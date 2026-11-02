package com.mobiinfer.demo

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.os.Process as AndroidProcess
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.io.StringWriter
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max

/**
 * Debug-only persistent logging. The logcat collector is restricted to this
 * app's PID, and ApplicationExitInfo supplies native crash traces after the
 * process is restarted.
 */
object DebugLogStore {
    private const val TAG = "DebugLogStore"
    private const val PREFS = "mobiinfer_debug_logs"
    private const val KEY_ENABLED = "enabled"
    private const val KEY_LAST_EXIT_TIMESTAMP = "last_exit_timestamp"
    private const val KEY_LAST_CRASH_HASH = "last_crash_hash"
    private const val LOG_DIR = "debug-logs"
    private const val LOG_NAME = "latest.log"
    private const val MAX_LOG_BYTES = 512 * 1024L
    private const val MAX_TRACE_CHARS = 128 * 1024
    private const val MAX_EXIT_RECORDS = 8

    private val writeLock = Any()

    @Volatile
    private var installed = false

    @Volatile
    private var collectorProcess: java.lang.Process? = null

    @Volatile
    private var collectorThread: Thread? = null

    @JvmStatic
    @Volatile
    private var appContext: Context? = null

    fun install(context: Context) {
        val app = context.applicationContext
        appContext = app
        if (installed) return
        installed = true

        val previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            if (isEnabled(app)) {
                appendCrash(app, thread, error)
            }
            if (previousHandler != null) {
                previousHandler.uncaughtException(thread, error)
            } else {
                AndroidProcess.killProcess(AndroidProcess.myPid())
                kotlin.system.exitProcess(10)
            }
        }

        if (isEnabled(app)) {
            appendSessionHeader(app)
            captureExitInfo(app)
            captureCrashBuffer(app)
            startCollector(app)
        }
    }

    @JvmStatic
    fun isEnabled(context: Context): Boolean =
        prefs(context).getBoolean(KEY_ENABLED, false)

    @JvmStatic
    fun setEnabled(context: Context, enabled: Boolean) {
        val app = context.applicationContext
        prefs(app).edit().putBoolean(KEY_ENABLED, enabled).commit()
        if (enabled) {
            append(app, "===== debug logging enabled =====")
            appendSessionHeader(app)
            captureExitInfo(app)
            captureCrashBuffer(app)
            startCollector(app)
        } else {
            append(app, "===== debug logging disabled =====")
            stopCollector()
        }
    }

    @JvmStatic
    fun readLog(context: Context): String =
        try {
            logFile(context).takeIf { it.exists() }?.readText().orEmpty()
        } catch (error: Throwable) {
            "Unable to read debug log: ${error.message}"
        }

    @JvmStatic
    fun clearLog(context: Context) {
        synchronized(writeLock) {
            logFile(context).delete()
        }
    }

    private fun appendSessionHeader(context: Context) {
        val packageInfo = try {
            context.packageManager.getPackageInfo(context.packageName, 0)
        } catch (_: Throwable) {
            null
        }
        val versionCode = if (Build.VERSION.SDK_INT >= 28) {
            packageInfo?.longVersionCode ?: 0L
        } else {
            @Suppress("DEPRECATION")
            (packageInfo?.versionCode ?: 0).toLong()
        }
        append(
            context,
            buildString {
                appendLine()
                appendLine("===== ${timestamp()} app session =====")
                appendLine("version=${packageInfo?.versionName ?: "?"} ($versionCode)")
                appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
                appendLine("android=${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})")
                append("pid=${AndroidProcess.myPid()}")
            }
        )
    }

    private fun appendCrash(context: Context, thread: Thread, error: Throwable) {
        val stack = StringWriter().also { writer ->
            PrintWriter(writer).use { error.printStackTrace(it) }
        }.toString()
        append(
            context,
            buildString {
                appendLine()
                appendLine("===== FATAL ${timestamp()} =====")
                appendLine("thread=${thread.name}")
                append(stack)
            }
        )
    }

    private fun captureExitInfo(context: Context) {
        if (Build.VERSION.SDK_INT < 30) return
        val activityManager = context.getSystemService(ActivityManager::class.java) ?: return
        val exits = try {
            activityManager.getHistoricalProcessExitReasons(
                context.packageName,
                0,
                MAX_EXIT_RECORDS
            )
        } catch (error: Throwable) {
            Log.w(TAG, "Unable to read process exit history", error)
            return
        }

        val prefs = prefs(context)
        val lastTimestamp = prefs.getLong(KEY_LAST_EXIT_TIMESTAMP, 0L)
        var newestTimestamp = lastTimestamp
        exits.sortedBy { it.timestamp }.forEach { exit ->
            newestTimestamp = max(newestTimestamp, exit.timestamp)
            if (exit.timestamp <= lastTimestamp || !isReportableExit(exit.reason)) return@forEach

            append(
                context,
                buildString {
                    appendLine()
                    appendLine("===== previous abnormal exit ${formatTime(exit.timestamp)} =====")
                    appendLine("reason=${exitReason(exit.reason)} (${exit.reason})")
                    appendLine("status=${exit.status} importance=${exit.importance}")
                    exit.description?.takeIf { it.isNotBlank() }?.let { appendLine("description=$it") }
                }
            )
            try {
                exit.traceInputStream?.bufferedReader()?.use { reader ->
                    val trace = reader.readText().take(MAX_TRACE_CHARS)
                    if (trace.isNotBlank()) append(context, trace)
                }
            } catch (error: Throwable) {
                append(context, "Unable to read exit trace: ${error.message}")
            }
        }
        prefs.edit().putLong(KEY_LAST_EXIT_TIMESTAMP, newestTimestamp).apply()
    }

    private fun captureCrashBuffer(context: Context) {
        val main = readLogcat("-b", "main")
        val crash = readLogcat("-b", "crash")

        // The main buffer carries the ggml/fastrpc diagnostics that precede a
        // native abort. Both buffers are filtered down to this process: the
        // device emits thousands of framework lines per launch, which would
        // otherwise consume the whole record budget before the real failure.
        val appLines = (main + crash)
            .lineSequence()
            .filter { it.contains(context.packageName) || it.contains("ggml") || it.contains("llama") }
            .toList()

        val output = if (appLines.isEmpty()) "" else appLines.joinToString("\n")
        if (output.isBlank()) return

        val hash = MessageDigest.getInstance("SHA-256")
            .digest(output.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        val prefs = prefs(context)
        if (prefs.getString(KEY_LAST_CRASH_HASH, null) == hash) return

        append(
            context,
            buildString {
                appendLine()
                appendLine("===== Android crash buffer ${timestamp()} =====")
                append(output.take(MAX_TRACE_CHARS * 2))
            }
        )
        prefs.edit().putString(KEY_LAST_CRASH_HASH, hash).apply()
    }

    private fun readLogcat(vararg args: String): String =
        try {
            val process = ProcessBuilder("logcat", "-d", *args, "-v", "threadtime")
                .redirectErrorStream(true)
                .start()
            val text = process.inputStream.bufferedReader().use { it.readText() }
            process.waitFor()
            text
        } catch (error: Throwable) {
            Log.w(TAG, "Unable to read logcat buffer ${args.joinToString(" ")}", error)
            ""
        }

    private fun isReportableExit(reason: Int): Boolean =
        reason == ApplicationExitInfo.REASON_CRASH ||
            reason == ApplicationExitInfo.REASON_CRASH_NATIVE ||
            reason == ApplicationExitInfo.REASON_ANR ||
            reason == ApplicationExitInfo.REASON_INITIALIZATION_FAILURE ||
            reason == ApplicationExitInfo.REASON_LOW_MEMORY ||
            reason == ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE ||
            reason == ApplicationExitInfo.REASON_SIGNALED

    private fun exitReason(reason: Int): String =
        when (reason) {
            ApplicationExitInfo.REASON_CRASH -> "JAVA_CRASH"
            ApplicationExitInfo.REASON_CRASH_NATIVE -> "NATIVE_CRASH"
            ApplicationExitInfo.REASON_ANR -> "ANR"
            ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "INIT_FAILURE"
            ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY"
            ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "RESOURCE_LIMIT"
            ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED"
            else -> "OTHER"
        }

    @Synchronized
    private fun startCollector(context: Context) {
        if (collectorProcess?.isAlive == true) return
        val process = try {
            ProcessBuilder(
                "logcat",
                "-v",
                "threadtime",
                "--pid",
                AndroidProcess.myPid().toString()
            ).redirectErrorStream(true).start()
        } catch (error: Throwable) {
            append(context, "Unable to start logcat collector: ${error.message}")
            return
        }
        collectorProcess = process
        collectorThread = Thread(
            {
                try {
                    val fold = RepeatingLineFold()
                    process.inputStream.bufferedReader().useLines { lines ->
                        lines.forEach { line -> fold.accept(line)?.let { append(context, it) } }
                    }
                    fold.flush()?.let { append(context, it) }
                } catch (_: Throwable) {
                    // Process destruction during shutdown is expected.
                } finally {
                    finishCollector(process)
                }
            },
            "MobiInferDebugLogcat"
        ).apply {
            isDaemon = true
            start()
        }
    }

    /**
     * Drops logcat noise emitted by the platform (view system, insets, BLAST
     * buffer plumbing, HyperOS helpers) and folds runs of identical messages
     * into one line plus a repeat count, so the 512 KB file holds real signal.
     */
    internal class RepeatingLineFold {
        private var lastRecord: String? = null
        private var lastMessage: String? = null
        private var lastTimestamp: String? = null
        private var repeated = 0

        internal fun accept(line: String): String? {
            val parsed = parse(line) ?: return line
            if (isNoiseTag(parsed.tag, parsed.message)) return null
            if (parsed.message == lastMessage) {
                repeated++
                return null
            }
            val flushed = flush()
            lastRecord = parsed.record
            lastMessage = parsed.message
            lastTimestamp = parsed.timestamp
            repeated = 0
            return flushed
        }

        internal fun flush(): String? {
            val record = lastRecord ?: return null
            val timestamp = lastTimestamp
            lastRecord = null
            lastMessage = null
            lastTimestamp = null
            val count = repeated
            repeated = 0
            if (count == 0 || timestamp == null) return record
            return record + "\n" + timestamp + "     DEBUG repeating line $count more time(s)"
        }

        private fun parse(line: String): ParsedLine? {
            // threadtime format: MM-DD HH:MM:SS.mmm  PID  TID L TAG     : message
            if (line.length < 40) return null
            val timestampEnd = 18
            if (line.getOrNull(2) != '-' || line.getOrNull(5) != ' ') return null
            val messageSeparator = line.indexOf(": ")
            if (messageSeparator < 0) return null
            var pos = timestampEnd
            fun skipSpaces() {
                while (pos < messageSeparator && line[pos] == ' ') pos++
            }
            fun skipToken() {
                while (pos < messageSeparator && line[pos] != ' ') pos++
            }

            // Skip the PID and TID columns. Parsing their values is unnecessary,
            // but consuming both tokens puts us at the severity column.
            repeat(2) {
                skipSpaces()
                skipToken()
                if (pos >= messageSeparator) return null
            }
            skipSpaces()
            val level = line.getOrNull(pos) ?: return null
            if (level !in 'A'..'Z') return null
            if (line.getOrNull(++pos) != ' ') return null
            skipSpaces()
            if (pos >= messageSeparator) return null
            return ParsedLine(
                record = line,
                timestamp = line.substring(0, timestampEnd),
                tag = line.substring(pos, messageSeparator).trim(),
                message = line.substring(messageSeparator + 2)
            )
        }

        private fun isNoiseTag(tag: String?, text: String): Boolean {
            if (tag == null) return false
            if (tag in NOISE_TAGS) return true
            if (NOISE_TAG_PREFIXES.any { tag.startsWith(it) }) return true
            return text.contains("getMiuiFreeformStackInfo")
        }
    }

    private class ParsedLine(
        val record: String,
        val timestamp: String,
        val tag: String,
        val message: String
    )

    private val NOISE_TAGS = setOf(
        "Activity",
        "InsetsSource",
        "InsetsController",
        "DecorViewImmersiveImpl",
        "HandWritingStubImpl",
        "ImeTracker",
        "HWUI",
        "libEGL",
        "ContentCatcher_TaplusBlackListHelper"
    )

    private val NOISE_TAG_PREFIXES = listOf(
        "VRI[",
        "BufferQueue",
        "MiuiMultiWindow"
    )

    @Synchronized
    private fun stopCollector() {
        val process = collectorProcess
        collectorProcess = null
        collectorThread?.interrupt()
        collectorThread = null
        process?.destroy()
    }

    @Synchronized
    private fun finishCollector(process: java.lang.Process) {
        if (collectorProcess === process) {
            collectorProcess = null
            collectorThread = null
        }
    }

    /**
     * Writes a diagnostic block (already formatted) to the store. Called
     * reflectively from [DebugLogBridge], which exists in release builds too;
     * the context is kept as an application-wide reference installed by
     * [install].
     */
    @JvmStatic
    fun dump(tag: String, text: String) {
        val context = appContext ?: return
        append(context, buildString {
            appendLine()
            appendLine("---- [$tag] ${timestamp()} ----")
            append(text)
        })
    }

    private fun append(context: Context, text: String) {
        if (text.isEmpty()) return
        synchronized(writeLock) {
            try {
                val file = logFile(context)
                file.parentFile?.mkdirs()
                FileOutputStream(file, true).use { output ->
                    output.write(text.toByteArray(Charsets.UTF_8))
                    if (!text.endsWith('\n')) output.write('\n'.code)
                    output.flush()
                }
                if (file.length() > MAX_LOG_BYTES) {
                    trimLog(file)
                }
            } catch (error: Throwable) {
                Log.w(TAG, "Unable to write debug log", error)
            }
        }
    }

    private fun trimLog(file: File) {
        val bytes = file.readBytes()
        val keepFrom = max(0, bytes.size - (MAX_LOG_BYTES / 2).toInt())
        file.writeBytes(bytes.copyOfRange(keepFrom, bytes.size))
    }

    private fun logFile(context: Context): File =
        File(context.filesDir, "$LOG_DIR/$LOG_NAME")

    private fun prefs(context: Context) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private fun timestamp(): String = formatTime(System.currentTimeMillis())

    private fun formatTime(timeMs: Long): String =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date(timeMs))
}
