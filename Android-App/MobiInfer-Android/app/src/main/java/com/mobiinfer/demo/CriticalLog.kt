package com.mobiinfer.demo

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Last-resort diagnostic trail for the text-to-image pipeline.
 *
 * logcat is useless there: the Qualcomm DSP driver prints hundreds of lines a
 * second while a model runs, so whatever we log is rotated away before the
 * failure can be read back. [DebugLogBridge] has the same problem only in
 * release builds; it needs the debug source set, and this file must work in
 * every build type. Each line is therefore appended to a small file in the app
 * directory that survives a process death.
 *
 * Deliberately tiny and allocation-cheap: it runs on the image path.
 */
object CriticalLog {

    private const val TAG = "CriticalLog"
    private const val FILE_NAME = "pipeline.log"
    private const val MAX_BYTES = 256 * 1024L
    private val format = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)
    private val lock = Any()

    @Volatile
    private var context: Context? = null

    fun install(context: Context) {
        this.context = context.applicationContext
    }

    fun log(component: String, message: String) {
        val stackTop = Throwable().stackTrace
            .firstOrNull { it.className != CriticalLog::class.java.name }
            ?.let { "${it.fileName}:${it.lineNumber}" } ?: "?"
        val line = "${format.format(Date())} ${android.os.Process.myPid()} [$component] $message @$stackTop"
        Log.i(TAG, line)
        val ctx = context ?: return
        synchronized(lock) {
            try {
                val file = File(ctx.filesDir, FILE_NAME)
                if (file.length() > MAX_BYTES) {
                    val keep = file.readLines().takeLast(2000)
                    file.writeText(keep.joinToString("\n", postfix = "\n"))
                }
                file.appendText(line + "\n")
            } catch (_: Exception) {
                // Diagnostics must never break the feature.
            }
        }
    }

    fun text(context: Context): String =
        File(context.filesDir, FILE_NAME).takeIf { it.exists() }?.readText().orEmpty()
}
