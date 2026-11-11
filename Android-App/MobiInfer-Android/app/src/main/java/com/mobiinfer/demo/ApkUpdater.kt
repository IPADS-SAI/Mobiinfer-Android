package com.mobiinfer.demo

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.core.content.FileProvider
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Downloads an update APK from the mirror list in update.json and installs it
 * through the system package installer.
 *
 * Two checks gate the install, in this order:
 *  1. SHA-256 must match the manifest — catches truncated or corrupted bodies.
 *  2. The APK signer must equal the installed app's signer — this is the real
 *     security boundary. Without it the system rejects the install anyway
 *     (INSTALL_FAILED_UPDATE_INCOMPATIBLE); checking first gives a clear error
 *     and makes third-party mirrors safe to use.
 */
object ApkUpdater {
    private const val TAG = "ApkUpdater"
    private const val BUFFER_SIZE = 64 * 1024

    /** Same path as the cache-path declared in res/xml/file_paths.xml. */
    fun apkFile(context: Context): File = File(context.cacheDir, "updates/update.apk")

    /**
     * Tries each mirror in order. Returns the verified APK, or null when every
     * mirror failed. [onProgress] is called from an IO thread with 0..100.
     */
    suspend fun download(
        context: Context,
        info: UpdateInfo,
        onProgress: (Int) -> Unit,
    ): File? = withContext(Dispatchers.IO) {
        val out = apkFile(context)
        out.parentFile?.mkdirs()
        for (url in info.apkUrls) {
            ensureActive()
            try {
                if (fetch(url, out, info, onProgress) && isInstallable(context, out)) {
                    return@withContext out
                }
                Log.w(TAG, "rejected $url")
            } catch (e: CancellationException) {
                File(out.parentFile, "update.apk.part").delete()
                throw e
            } catch (t: Throwable) {
                Log.w(TAG, "failed $url: ${t.javaClass.simpleName}")
            }
        }
        out.delete()
        null
    }

    private fun fetch(url: String, out: File, info: UpdateInfo, onProgress: (Int) -> Unit): Boolean {
        val tmp = File(out.parentFile, "update.apk.part")
        tmp.delete()

        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 10_000
            readTimeout = 15_000
            instanceFollowRedirects = true
            setRequestProperty("Accept", "application/octet-stream")
        }
        try {
            if (conn.responseCode != HttpURLConnection.HTTP_OK) {
                Log.w(TAG, "HTTP ${conn.responseCode} for $url")
                return false
            }
            val total = conn.contentLengthLong.takeIf { it > 0 } ?: info.sizeBytes
            val digest = MessageDigest.getInstance("SHA-256")
            var read = 0L
            var lastPercent = -1

            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        read += n
                        if (info.sizeBytes > 0 && read > info.sizeBytes) {
                            Log.w(TAG, "body exceeds declared size for $url")
                            tmp.delete()
                            return false
                        }
                        output.write(buffer, 0, n)
                        digest.update(buffer, 0, n)
                        if (total > 0) {
                            val percent = (read * 100 / total).toInt()
                            if (percent != lastPercent && percent % 2 == 0) {
                                lastPercent = percent
                                onProgress(percent)
                            }
                        }
                    }
                }
            }

            val sha256 = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
            if (!sha256.equals(info.sha256, ignoreCase = true)) {
                Log.w(TAG, "sha256 mismatch for $url")
                tmp.delete()
                return false
            }
            if (info.sizeBytes > 0 && read != info.sizeBytes) {
                Log.w(TAG, "size mismatch for $url: $read != ${info.sizeBytes}")
                tmp.delete()
                return false
            }
            out.delete()
            if (!tmp.renameTo(out)) {
                tmp.delete()
                return false
            }
            return true
        } finally {
            conn.disconnect()
        }
    }

    /** Package name, version and signer must all match what is installed. */
    private fun isInstallable(context: Context, apk: File): Boolean {
        val pm = context.packageManager
        return try {
            val archive = pm.getPackageArchiveInfo(apk.absolutePath, signerFlags()) ?: return false
            if (archive.packageName != context.packageName) return false

            val archiveCode = if (Build.VERSION.SDK_INT >= 28) {
                archive.longVersionCode
            } else {
                @Suppress("DEPRECATION")
                archive.versionCode.toLong()
            }
            if (archiveCode < UpdateChecker.installedVersionCode(context)) return false

            val mine = signerDigest(context, null) ?: return false
            val theirs = signerDigest(context, apk) ?: return false
            mine.contentEquals(theirs)
        } catch (t: Throwable) {
            Log.w(TAG, "installability check failed: ${t.javaClass.simpleName}")
            false
        }
    }

    /**
     * Android 9/10 only collect certificates for an *archive* when GET_SIGNATURES
     * is set: with GET_SIGNING_CERTIFICATES alone getPackageArchiveInfo() returns a
     * PackageInfo whose signingInfo is null, which made every download fail the
     * signer check on those releases. Ask for both and read whichever is filled.
     */
    private fun signerFlags(): Int {
        @Suppress("DEPRECATION")
        val legacy = PackageManager.GET_SIGNATURES
        return if (Build.VERSION.SDK_INT >= 28) {
            legacy or PackageManager.GET_SIGNING_CERTIFICATES
        } else {
            legacy
        }
    }

    private fun signerDigest(context: Context, apk: File?): ByteArray? {
        val pm = context.packageManager
        val info = if (apk == null) {
            pm.getPackageInfo(context.packageName, signerFlags())
        } else {
            pm.getPackageArchiveInfo(apk.absolutePath, signerFlags()) ?: return null
        }
        @Suppress("DEPRECATION")
        val signer = info.signingInfo?.apkContentsSigners?.singleOrNull()
            ?: info.signatures?.singleOrNull()
            ?: return null
        return MessageDigest.getInstance("SHA-256").digest(signer.toByteArray())
    }

    /** Android 8+ needs per-app consent before an app may prompt the installer. */
    fun canInstall(context: Context): Boolean =
        Build.VERSION.SDK_INT < 26 || context.packageManager.canRequestPackageInstalls()

    fun requestInstallPermission(activity: Activity): Boolean {
        if (Build.VERSION.SDK_INT < 26) return false
        return runCatching {
            activity.startActivity(
                Intent(
                    Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                    Uri.parse("package:${activity.packageName}")
                )
            )
        }.onFailure { Log.w(TAG, "no unknown-sources settings activity", it) }.isSuccess
    }

    /**
     * Hands the APK to the system installer. The app process is killed once the
     * install completes, so callers must not run this while a model is generating.
     */
    fun install(activity: Activity, apk: File): Boolean {
        if (!apk.exists()) {
            Log.w(TAG, "cached APK is gone: $apk")
            return false
        }
        return runCatching {
            val uri = FileProvider.getUriForFile(activity, "${activity.packageName}.fileprovider", apk)
            activity.startActivity(
                Intent(Intent.ACTION_VIEW).apply {
                    setDataAndType(uri, "application/vnd.android.package-archive")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
            )
        }.onFailure { Log.w(TAG, "no package installer available", it) }.isSuccess
    }
}
