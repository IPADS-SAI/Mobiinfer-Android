package com.mobiinfer.demo

import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * An update offered by the manifest served from mobiinfer.yiguzhaojin.cn.
 *
 * [apkUrls] is ordered: the first reachable mirror whose bytes match [sha256]
 * and whose signature matches the installed app wins.
 */
data class UpdateInfo(
    val versionCode: Long,
    val versionName: String,
    val sha256: String,
    val sizeBytes: Long,
    val apkUrls: List<String>,
    val notes: String,
    val minSupportedVersionCode: Long,
)

/** Outcome of one manifest check, so callers can tell "no update" from "offline". */
sealed interface UpdateCheck {
    data object UpToDate : UpdateCheck
    data object Failed : UpdateCheck
    data class Available(val info: UpdateInfo) : UpdateCheck
}

/**
 * Fetches the update manifest and compares its versionCode with ours.
 *
 * The manifest is a small JSON file hosted next to the landing page:
 * https://mobiinfer.yiguzhaojin.cn/update.json
 *
 * Checks are throttled to once a day. The response is cached with ETag, so a
 * repeat check costs one 304. Every failure (offline, 404, malformed JSON) is
 * reported as [UpdateCheck.Failed] and never interrupts the user.
 */
object UpdateChecker {
    private const val TAG = "UpdateChecker"
    private const val MANIFEST_URL = "https://mobiinfer.yiguzhaojin.cn/update.json"
    private const val PREFS = "updater"
    private const val CHECK_INTERVAL_MS = 24L * 60 * 60 * 1000

    /** versionCode of the installed package; 0 if the lookup fails. */
    fun installedVersionCode(context: Context): Long = runCatching {
        val info = context.packageManager.getPackageInfo(context.packageName, 0)
        if (Build.VERSION.SDK_INT >= 28) {
            info.longVersionCode
        } else {
            @Suppress("DEPRECATION")
            info.versionCode.toLong()
        }
    }.getOrDefault(0L)

    fun installedVersionName(context: Context): String = runCatching {
        context.packageManager.getPackageInfo(context.packageName, 0).versionName.orEmpty()
    }.getOrDefault("")

    /**
     * @param force skip the daily throttle (manual "check for updates").
     */
    suspend fun check(context: Context, force: Boolean = false): UpdateCheck =
        withContext(Dispatchers.IO) {
            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val now = System.currentTimeMillis()
            if (!force && now - prefs.getLong("lastCheck", 0L) < CHECK_INTERVAL_MS) {
                // Throttled: report "up to date" so the caller stays silent.
                return@withContext UpdateCheck.UpToDate
            }

            var conn: HttpURLConnection? = null
            try {
                conn = (URL(MANIFEST_URL).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 8000
                    readTimeout = 8000
                    instanceFollowRedirects = true
                    setRequestProperty("Accept", "application/json")
                    prefs.getString("etag", null)?.let { setRequestProperty("If-None-Match", it) }
                }
                val code = conn.responseCode
                prefs.edit().putLong("lastCheck", now).apply()
                if (code == HttpURLConnection.HTTP_NOT_MODIFIED) {
                    return@withContext UpdateCheck.UpToDate
                }
                if (code != HttpURLConnection.HTTP_OK) {
                    Log.w(TAG, "manifest returned HTTP $code")
                    return@withContext UpdateCheck.Failed
                }

                val body = conn.inputStream.bufferedReader().use { it.readText() }
                conn.getHeaderField("ETag")?.let { prefs.edit().putString("etag", it).apply() }

                val info = parse(context, body) ?: return@withContext UpdateCheck.Failed
                if (info.versionCode <= installedVersionCode(context)) {
                    UpdateCheck.UpToDate
                } else {
                    UpdateCheck.Available(info)
                }
            } catch (e: Exception) {
                // Offline, DNS failure, TLS error, truncated body: all silent.
                Log.w(TAG, "check failed: ${e.javaClass.simpleName}")
                UpdateCheck.Failed
            } finally {
                conn?.disconnect()
            }
        }

    /**
     * The landing page is a SPA: unknown paths return index.html with a 200.
     * Anything that is not the expected JSON shape is treated as "no manifest".
     */
    private fun parse(context: Context, body: String): UpdateInfo? = runCatching {
        val o = JSONObject(body)
        val urls = o.optJSONArray("apkUrls")
            ?.let { array -> List(array.length()) { array.getString(it) } }
            ?.filter { it.startsWith("https://") }
            .orEmpty()
        if (urls.isEmpty()) return null
        UpdateInfo(
            versionCode = o.getLong("versionCode"),
            versionName = o.optString("versionName"),
            sha256 = o.getString("sha256").lowercase(),
            sizeBytes = o.optLong("sizeBytes", 0L),
            apkUrls = urls,
            notes = pickNotes(context, o),
            minSupportedVersionCode = o.optLong("minSupportedVersionCode", 0L),
        )
    }.getOrNull()

    private fun pickNotes(context: Context, o: JSONObject): String {
        val notes = o.optJSONObject("notes") ?: return ""
        val chinese = LocaleManager.currentLanguage(context) == LocaleManager.AppLanguage.ZH
        return notes.optString(if (chinese) "zh" else "en").ifBlank { notes.optString("zh") }
    }
}
