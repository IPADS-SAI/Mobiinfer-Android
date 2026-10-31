package com.mobiinfer.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Foreground service that runs [LlamaEngine.downloadModels] outside the
 * Activity's lifecycle. Survives:
 *   - user switching to another app
 *   - the screen turning off (Doze still throttles network somewhat, but the
 *     PARTIAL_WAKE_LOCK keeps the CPU awake long enough to keep the socket
 *     alive; combined with the HTTP Range resume in LlamaEngine that's
 *     enough to make multi-GB downloads reliable)
 *   - Activity being destroyed (we're a service, not bound to it)
 *
 * Coordinates with the UI through [ModelDownloadController] (a simple
 * StateFlow singleton) - we deliberately don't expose a binder so observing
 * the service requires no IPC plumbing in the Activity.
 */
class ModelDownloadService : Service() {

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var downloadJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var currentModelId: String? = null
    private var lastFraction: Float = 0f

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        ensureNotificationChannel(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: ACTION_START

        if (action == ACTION_PAUSE || action == ACTION_CANCEL) {
            val modelId = currentModelId ?: ModelDownloadController.runningModelId()
            Log.i(TAG, "Pause/cancel requested (model=$modelId)")
            // Flag the stop *before* cancelling: the job may already be past
            // every suspension point (inside a blocking socket read), and the
            // flag is what keeps it from resuming or reporting completion.
            if (modelId != null) {
                LlamaEngine.flagDownloadStop(modelId)
            }
            downloadJob?.cancel()
            if (modelId != null) {
                if (action == ACTION_CANCEL) {
                    ModelDownloadController.markCancelled(modelId)
                } else {
                    ModelDownloadController.markPaused(modelId, lastFraction)
                }
            }
            releaseWakeLock()
            stopForegroundCompat()
            stopSelf()
            return START_NOT_STICKY
        }

        val modelId = intent?.getStringExtra(EXTRA_MODEL_ID)
        if (modelId == null) {
            Log.w(TAG, "No model id supplied")
            stopSelf()
            return START_NOT_STICKY
        }

        // One download at a time. Same model -> re-attach; different -> ignore.
        if (downloadJob?.isActive == true) {
            if (currentModelId == modelId) {
                startForegroundWithNotification(buildNotification(getString(R.string.status_downloading)))
            } else {
                // Tell the user instead of dropping the request silently: the
                // UI would otherwise show "not downloaded" as if the tap did
                // nothing.
                Log.w(TAG, "Download already running for $currentModelId, ignoring $modelId")
                ModelDownloadController.markBusy(modelId, currentModelId ?: "")
            }
            return START_NOT_STICKY
        }

        // Starting a download is an explicit user action (or the chat asking
        // to fetch the image model), so it lifts a previous pause of *this*
        // model. Scoped to modelId on purpose: pausing A and immediately
        // starting B must not clear A's stop marker, or A's loop keeps running
        // and the pause looks ignored. Placed after the busy check so a request
        // that is dropped does not clear anything either.
        LlamaEngine.resumeRequested(modelId)

        currentModelId = modelId
        lastFraction = 0f
        startForegroundWithNotification(buildNotification(getString(R.string.download_preparing)))
        acquireWakeLock()
        ModelDownloadController.markStarted(modelId)

        downloadJob = scope.launch {
            try {
                LlamaEngine.downloadModels(applicationContext, modelId) { progress ->
                    lastFraction = progress.fraction
                    ModelDownloadController.publishProgress(modelId, progress.message, progress.fraction)
                    updateNotification(progress.message, progress.fraction)
                }
                // Publish "completed" only if nothing replaced this session
                // meanwhile. The stop marker is checked rather than the
                // controller status, because a resume clears the status to
                // Running while the paused loop is still draining.
                if (ModelDownloadController.runningModelId() == modelId) {
                    ModelDownloadController.markCompleted(modelId)
                    updateNotification(getString(R.string.download_complete_short))
                }
            } catch (e: CancellationException) {
                // Pause/cancel path already updated the controller; keep .tmp
                // so the next start() resumes via HTTP Range.
                Log.i(TAG, "Download paused/cancelled: $modelId")
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                ModelDownloadController.markFailed(modelId, e.message ?: e::class.java.simpleName)
                updateNotification(getString(R.string.download_failed, e.message))
            } finally {
                // Only clean up what this session still owns: a paused session
                // can finish unwinding after its successor already started, and
                // an unconditional reset would wipe the new session's state.
                if (currentModelId == modelId) currentModelId = null
                LlamaEngine.clearDownloadJob(modelId)
                releaseWakeLock()
                stopForegroundCompat()
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        downloadJob?.cancel()
        releaseWakeLock()
        scope.coroutineContext[Job]?.cancel()
        super.onDestroy()
    }

    private fun startForegroundWithNotification(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_DETACH)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(false)
        }
    }

    private fun updateNotification(message: String, fraction: Float? = null) {
        val nm = ContextCompat.getSystemService(this, NotificationManager::class.java)
            ?: return
        nm.notify(NOTIFICATION_ID, buildNotification(message, fraction))
    }

    private fun buildNotification(message: String, fraction: Float? = null): Notification {
        val openIntent = Intent(this, ModelManagerActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val contentPi = PendingIntent.getActivity(
            this,
            0,
            openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val cancelPi = PendingIntent.getService(
            this,
            1,
            Intent(this, ModelDownloadService::class.java).apply { action = ACTION_CANCEL },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_download_title))
            .setContentText(message)
            .setStyle(NotificationCompat.BigTextStyle().bigText(message))
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setContentIntent(contentPi)
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                getString(R.string.cancel),
                cancelPi
            )

        // Determinate bar once we know the overall fraction; indeterminate
        // (no bar) for the preparing / terminal states where fraction is null.
        if (fraction != null) {
            builder.setProgress(100, (fraction * 100).roundToInt().coerceIn(0, 100), false)
        }

        return builder.build()
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = ContextCompat.getSystemService(this, PowerManager::class.java) ?: return
        // PARTIAL_WAKE_LOCK keeps the CPU running with the screen off, but
        // *not* the radio - that's fine, OkHttp/HttpURLConnection only need
        // the CPU awake to drive the socket; the modem stays in low power.
        // 30 minute hard timeout as a safety belt against leaks: a typical
        // download is well under that.
        wakeLock = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "MiniCPMV:ModelDownload"
        ).apply {
            setReferenceCounted(false)
            acquire(30 * 60 * 1000L)
        }
    }

    private fun releaseWakeLock() {
        try {
            wakeLock?.takeIf { it.isHeld }?.release()
        } catch (e: Throwable) {
            Log.w(TAG, "WakeLock release failed: ${e.message}")
        }
        wakeLock = null
    }

    companion object {
        private const val TAG = "ModelDownloadService"
        private const val CHANNEL_ID = "model_download"
        private const val NOTIFICATION_ID = 0xD0_C1

        const val ACTION_START = "com.mobiinfer.demo.action.DOWNLOAD_START"
        const val ACTION_CANCEL = "com.mobiinfer.demo.action.DOWNLOAD_CANCEL"
        const val ACTION_PAUSE = "com.mobiinfer.demo.action.DOWNLOAD_PAUSE"
        const val EXTRA_MODEL_ID = "model_id"

        fun start(context: Context, modelId: String) {
            ensureNotificationChannel(context)
            val intent = Intent(context, ModelDownloadService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_MODEL_ID, modelId)
            }
            ContextCompat.startForegroundService(context, intent)
        }

        fun pause(context: Context) {
            val intent = Intent(context, ModelDownloadService::class.java).apply {
                action = ACTION_PAUSE
            }
            context.startService(intent)
        }

        fun cancel(context: Context) {
            val intent = Intent(context, ModelDownloadService::class.java).apply {
                action = ACTION_CANCEL
            }
            // Use startService instead of startForegroundService for the
            // cancel signal to avoid a stray foreground transition.
            context.startService(intent)
        }

        private fun ensureNotificationChannel(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
            val nm = ContextCompat.getSystemService(context, NotificationManager::class.java)
                ?: return
            if (nm.getNotificationChannel(CHANNEL_ID) != null) return
            val channel = NotificationChannel(
                CHANNEL_ID,
                context.getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = context.getString(R.string.notification_channel_desc)
                setShowBadge(false)
            }
            nm.createNotificationChannel(channel)
        }
    }
}

/**
 * Singleton that decouples the service from any specific Activity. The
 * Activity (or other UI) collects [status] to render progress; the service
 * publishes updates via the package-private [publishProgress] / [markFoo]
 * functions.
 *
 * This lives outside the service so that observers don't need to bind to the
 * service or worry about IPC - the JVM-local StateFlow is enough for our
 * single-process app.
 */
object ModelDownloadController {

    sealed class Status {
        object Idle : Status()
        data class Running(val modelId: String, val message: String, val fraction: Float) : Status()
        data class Paused(val modelId: String, val fraction: Float) : Status()
        data class Completed(val modelId: String) : Status()
        data class Cancelled(val modelId: String) : Status()
        data class Failed(val modelId: String, val message: String) : Status()

        /**
         * A start request was dropped because [runningModelId] is already
         * downloading. Carries both ids so the UI can explain why nothing
         * happened; it is transient and never becomes the running state.
         */
        data class Busy(val modelId: String, val runningModelId: String) : Status()
    }

    private val _status = MutableStateFlow<Status>(Status.Idle)
    val status: StateFlow<Status> = _status.asStateFlow()

    // One-shot "this tap did nothing" notices, kept apart from [status] so they
    // cannot replace the live Running state of the download in progress.
    private val _busyEvents = MutableSharedFlow<Status.Busy>(extraBufferCapacity = 4)
    val busyEvents: SharedFlow<Status.Busy> = _busyEvents.asSharedFlow()

    val isRunning: Boolean
        get() = _status.value is Status.Running

    /** Id of the model currently downloading, or null when idle. */
    fun runningModelId(): String? = (_status.value as? Status.Running)?.modelId

    internal fun markStarted(modelId: String) {
        _status.value = Status.Running(modelId, "", 0f)
    }

    internal fun publishProgress(modelId: String, message: String, fraction: Float) {
        val cur = _status.value
        if (cur is Status.Running && cur.modelId == modelId) {
            _status.value = Status.Running(modelId, message, fraction)
        }
    }

    internal fun markCompleted(modelId: String) {
        _status.value = Status.Completed(modelId)
    }

    internal fun markFailed(modelId: String, reason: String) {
        _status.value = Status.Failed(modelId, reason)
    }

    internal fun markPaused(modelId: String, fraction: Float) {
        _status.value = Status.Paused(modelId, fraction)
    }

    internal fun markCancelled(modelId: String) {
        _status.value = Status.Cancelled(modelId)
    }

    /**
     * A start request was dropped because another model is downloading.
     *
     * Deliberately published through a separate channel instead of [status]:
     * overwriting the live [Status.Running] would blank the row and the
     * notification of the download that is actually running.
     */
    internal fun markBusy(modelId: String, runningModelId: String) {
        _busyEvents.tryEmit(Status.Busy(modelId, runningModelId))
    }

    /** Drop a terminal state (Completed/Cancelled/Failed/Paused) back to Idle once acknowledged. */
    fun acknowledge() {
        val cur = _status.value
        if (cur is Status.Completed || cur is Status.Cancelled || cur is Status.Failed || cur is Status.Paused) {
            _status.value = Status.Idle
        }
    }
}
