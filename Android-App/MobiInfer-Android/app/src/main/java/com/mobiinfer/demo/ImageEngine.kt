package com.mobiinfer.demo

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withTimeoutOrNull
import java.util.concurrent.Executors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

sealed class ImageState {
    object Uninitialized : ImageState()
    object Initializing : ImageState()
    object LoadingModel : ImageState()
    object Ready : ImageState()
    object Generating : ImageState()
    data class Error(val exception: Exception) : ImageState()
}

/**
 * Text-to-image engine backed by stable-diffusion.cpp.
 *
 * stable-diffusion.cpp ships its own ggml, so it is built into a separate
 * native library ("imagegen") loaded independently of the llama "mobiinfer"
 * library. All native calls are serialised on a single-threaded dispatcher.
 */
class ImageEngine private constructor(
    private val context: Context
) {
    companion object {
        private val TAG = ImageEngine::class.java.simpleName

        // Stages reported by the native step callback (see imagegen_stage_t).
        // The step/steps pair only makes sense inside one stage.
        const val STAGE_LOADING = 0
        const val STAGE_ENCODING = 1
        const val STAGE_SAMPLING = 2
        const val STAGE_DECODING = 3
        const val STAGE_DONE = 4

        private const val INIT_TIMEOUT_MS = 30_000L

        // One bridge for the whole process; rebuilt engines re-point it.
        private val sharedBridge = Bridge(null)

        private fun bridgeFor(engine: ImageEngine): Bridge = sharedBridge.also { it.engine = engine }

        @Volatile
        private var instance: ImageEngine? = null

        fun getInstance(context: Context): ImageEngine =
            instance ?: synchronized(this) {
                ImageEngine(context.applicationContext).also { instance = it }
            }
    }

    // Dedicated single thread: native work must stay serialised, but it must
    // not share Dispatchers.IO's pool. limitedParallelism(1) still queues
    // behind every IO task scheduled before it, and nativeGenerate() is a
    // blocking multi-second call - a queued compute is invisible from the
    // outside, which is exactly how "generating image…" could hang forever.
    private val imageExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "imagegen-native").apply { isDaemon = true }
    }
    private val imageDispatcher = imageExecutor.asCoroutineDispatcher()
    private val imageScope = CoroutineScope(imageDispatcher + SupervisorJob())

    private val _state = MutableStateFlow<ImageState>(ImageState.Uninitialized)
    val state: StateFlow<ImageState> = _state.asStateFlow()

    @Volatile
    var isLoaded = false
        private set

    @Volatile
    var modelVersion: String = "Unknown"
        private set

    private external fun nativeSetenv(key: String, value: String)
    private external fun nativeInit(
        diffusionModelPath: String, llmPath: String, vaePath: String,
        tokenizerPath: String, nThreads: Int, backend: String
    ): Boolean
    // Returns the seed the image was drawn with (>= 0), or -1 on failure.
    private external fun nativeGenerate(
        prompt: String, negative: String,
        width: Int, height: Int, steps: Int, cfg: Float,
        seed: Long, outPngPath: String
    ): Long
    private external fun nativeLastError(): String
    // Receives onStep/onPreview from the native generation thread.
    private external fun nativeSetProgressListener(listener: Any?)
    private external fun nativeModelVersion(): String
    private external fun nativeFree()

    /**
     * Live generation progress. The object is also handed to native code,
     * which calls [onStep] / [onPreview] on the generation thread; both are
     * invoked through this interface so the JNI lookup always finds them.
     */
    interface ProgressListener {
        /**
         * [stage] is one of the STAGE_* constants, and the step/steps pair only
         * means something inside that stage: loading counts tensors, sampling
         * counts denoising steps and the tiled VAE decode counts tiles.
         * [samplePass] numbers the sampling batches of one generate call.
         * Returns false to cancel the generation.
         */
        fun onStep(samplePass: Int, stage: Int, step: Int, steps: Int, elapsedMs: Double): Boolean

        /**
         * Downscaled RGB preview (width*height*3 bytes) of the current latent.
         * Only pass 1 is reported: a later pass restarts from noise and would
         * make the picture jump back to a blur.
         */
        fun onPreview(samplePass: Int, step: Int, width: Int, height: Int, rgb: ByteArray)
    }

    @Volatile
    private var progressListenerOwner: Long = -1L

    @Volatile
    var progressListener: ProgressListener? = null
        private set

    /**
     * Installs [listener] for the generation identified by [token]. A later
     * generation replaces it; only the owner may clear it again, which kept a
     * previous run's cleanup from silencing the next one.
     */
    @Synchronized
    fun setProgressListener(token: Long, listener: ProgressListener?) {
        if (listener == null) {
            if (progressListenerOwner != token) return
            progressListener = null
            progressListenerOwner = -1L
            return
        }
        progressListener = listener
        progressListenerOwner = token
    }

    // Stable JNI target: native keeps a global reference to it for the whole
    // process, so it must never be replaced by a new engine instance.
    private val bridge: Bridge = bridgeFor(this)

    /**
     * True while nativeGenerate() is on the native thread. Freeing the context
     * underneath a running generate would be a use-after-free, so destroy()
     * and destroyIfLoaded() wait for it instead.
     */
    @Volatile
    var isGenerating = false
        private set

    // Resolved once the library is loaded (or loading failed). Every native
    // entry point awaits it, so loadModel can never race the initial dlopen.
    private val initialized = CompletableDeferred<Boolean>()

    init {
        imageScope.launch {
            try {
                check(_state.value is ImageState.Uninitialized)
                _state.value = ImageState.Initializing
                // Load first: nativeSetenv lives in libimagegen.so. Loading the
                // .so does NOT start the backend; the HTP stack is only brought
                // up at new_sd_ctx() time, so setting env right after load is
                // still in time. HOSTBUF=1 is required for mixed-backend
                // scheduling (see sdcpp-hexagon devdocs 04-delivery §3).
                Log.i(TAG, "Loading native library for image generation...")
                System.loadLibrary("imagegen")
                nativeSetenv("GGML_HEXAGON_HOSTBUF", "1")
                nativeSetenv("GGML_HEXAGON_ARCH", hexagonArchForDevice())
                nativeSetenv("ADSP_LIBRARY_PATH",
                    "${context.applicationInfo.nativeLibraryDir};/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp")
                nativeSetProgressListener(bridge)
                Log.i(TAG, "Imagegen native library loaded")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load native library", e)
                _state.value = ImageState.Error(e)
                throw e
            }
        }.also { job ->
            job.invokeOnCompletion { cause ->
                initialized.complete(cause == null && _state.value !is ImageState.Error)
            }
        }
    }

    /**
     * Static JNI target: native holds a reference to this object, not to the
     * engine, so a recreated engine does not leave a dangling listener.
     */
    internal class Bridge(@Volatile var engine: ImageEngine?) {
        @Suppress("unused")  // called from native
        fun onStep(pass: Int, stage: Int, step: Int, steps: Int, elapsedMs: Double): Boolean =
            engine?.progressListener?.onStep(pass, stage, step, steps, elapsedMs) ?: true

        @Suppress("unused")  // called from native
        fun onPreview(pass: Int, step: Int, width: Int, height: Int, rgb: ByteArray) {
            engine?.progressListener?.onPreview(pass, step, width, height, rgb)
        }
    }

    private suspend fun awaitInitialized(): Boolean {
        // Bounded wait with an explicit reason: a build without
        // libimagegen.so, or a dlopen that died, must not leave the caller
        // hanging on a deferred nobody will complete.
        val ok = withTimeoutOrNull(INIT_TIMEOUT_MS) { initialized.await() }
            ?: run {
                CriticalLog.log("imagegen", "native library init timed out (${INIT_TIMEOUT_MS} ms)")
                throw IllegalStateException("Imagegen native library did not finish loading")
            }
        if (!ok) throw IllegalStateException("Imagegen native library failed to load")
        return ok
    }

    /**
     * Frees the native context if one is loaded, and waits for it to happen on
     * the image dispatcher. Needed before loading another checkpoint: the
     * native side has a single global context, and [loadModel] must not run
     * while a previous context is still alive.
     */
    suspend fun destroyIfLoaded() {
        if (!isLoaded) return
        // Serialised on the native thread: if a generate is running there it
        // finishes first, because it occupies that same thread.
        withContext(imageDispatcher) { freeNative() }
    }

    private fun freeNative() {
        // A freed context has no meaningful error left; keeping it would show
        // a stale message on the next attempt.
        lastError = null
        if (!isLoaded) return
        runCatching { nativeFree() }
            .onFailure { Log.w(TAG, "nativeFree failed", it) }
        isLoaded = false
        _state.value = ImageState.Uninitialized
        Log.i(TAG, "Image engine freed")
    }

    // GGML_HEXAGON_ARCH: v79 for Snapdragon 8 Elite (SM8750 / OnePlus 13),
    // v81 for 8 Elite Gen5 (SM8850). Defaults to v79; harmless if the skel for
    // the detected arch is present alongside.
    private fun hexagonArchForDevice(): String {
        val soc = runCatching {
            (Class.forName("android.os.SystemProperties")
                .getMethod("get", String::class.java)
                .invoke(null, "ro.soc.model") as? String)
        }.getOrNull() ?: ""
        return when {
            // Matches the libggml-htp-v*.so skels the Gradle build ships.
            soc.contains("SM8850", true) -> "v81"
            else -> "v79"
        }
    }

    suspend fun loadModel(
        diffusionModelPath: String,
        llmPath: String? = null,
        vaePath: String? = null,
        tokenizerPath: String? = null,
        nThreads: Int = 0,
        backend: String = "diffusion=HTP0,te=cpu,vae=HTP0"
    ): Boolean = withContext(imageDispatcher) {
        CriticalLog.log("imagegen", "loadModel scheduled on the native thread")
        awaitInitialized()
        check(File(diffusionModelPath).exists()) { "Diffusion model not found: $diffusionModelPath" }
        if (!llmPath.isNullOrBlank()) {
            check(File(llmPath).exists()) { "Text encoder not found: $llmPath" }
        }
        if (!vaePath.isNullOrBlank()) {
            check(File(vaePath).exists()) { "VAE not found: $vaePath" }
        }
        if (!tokenizerPath.isNullOrBlank()) {
            check(File(tokenizerPath).exists()) { "Tokenizer not found: $tokenizerPath" }
        }

        lastError = null
        // DiT + text encoder + VAE are mmapped/loaded into RAM by sdcpp; the
        // file sizes are a good enough upper bound for deciding whether the
        // language model must be unloaded first.
        estimatedFootprintBytes = listOfNotNull(
            File(diffusionModelPath), llmPath?.let { File(it) }, vaePath?.let { File(it) }, tokenizerPath?.let { File(it) }
        ).sumOf { it.length() }

        try {
            _state.value = ImageState.LoadingModel
            Log.i(TAG, "Initializing imagegen: dit=$diffusionModelPath te=${llmPath ?: "-"} vae=${vaePath ?: "-"} backend=$backend")
            CriticalLog.log("imagegen", "new_sd_ctx start (footprint ${estimatedFootprintBytes / (1024 * 1024)} MB)")
            val ok = nativeInit(
                diffusionModelPath, llmPath ?: "", vaePath ?: "",
                tokenizerPath ?: "", nThreads, backend
            )
            if (!ok) {
                val e = RuntimeException("Imagegen init failed: ${nativeLastError()}")
                lastError = e.message
                _state.value = ImageState.Error(e)
                return@withContext false
            }
            isLoaded = true
            modelVersion = runCatching { nativeModelVersion() }.getOrDefault("Unknown")
            CriticalLog.log("imagegen", "new_sd_ctx done: $modelVersion")
            _state.value = ImageState.Ready
            Log.i(TAG, "Imagegen ready ($modelVersion)")
            return@withContext true
        } catch (e: Exception) {
            Log.e(TAG, "Error loading image model", e)
            lastError = e.message
            _state.value = ImageState.Error(e)
            return@withContext false
        }
    }

    /** Message of the last failed [loadModel]/[generate], if any. */
    @Volatile
    var lastError: String? = null
        private set

    /**
     * Rough resident footprint of [loadModel]'s weights, derived from the
     * file sizes on disk. Used to decide whether the language model has to go
     * before the image stack is loaded (see MainActivity's image tool).
     */
    var estimatedFootprintBytes: Long = 0L
        private set

    /**
     * Draws one image and returns the seed used, or null on failure (details
     * in [lastError]). A negative [seed] asks the engine to pick a random one.
     */
    suspend fun generate(
        prompt: String,
        negativePrompt: String = "",
        width: Int = LlamaEngine.DEFAULT_IMAGE_GEN_SIDE,
        height: Int = LlamaEngine.DEFAULT_IMAGE_GEN_SIDE,
        steps: Int = LlamaEngine.DEFAULT_IMAGE_GEN_STEPS,
        cfg: Float = LlamaEngine.DEFAULT_IMAGE_GEN_CFG,
        seed: Long = LlamaEngine.RANDOM_IMAGE_SEED,
        outputPath: String
    ): Long? {
        // Logged before and after entering the dispatcher: if only the first
        // line appears, the coroutine never reached the native thread, which
        // is a queue problem; if both appear and nothing follows, the
        // blocking native call itself is stuck.
        CriticalLog.log("imagegen", "generate() entered, waiting for the native thread")
        val queuedAt = SystemClock.elapsedRealtime()
        val result = withContext(imageDispatcher) {
            CriticalLog.log(
                "imagegen",
                "generate() running on the native thread after ${SystemClock.elapsedRealtime() - queuedAt} ms"
            )
            generateOnNativeThread(prompt, negativePrompt, width, height, steps, cfg, seed, outputPath)
        }
        CriticalLog.log("imagegen", "generate() returned ${result ?: "null"}")
        return result
    }

    /**
     * The actual draw, executed on the native thread. Everything critical is
     * logged from a finally block: any failure (check, exception, return)
     * leaves exactly one line that says how it ended.
     */
    private fun generateOnNativeThread(
        prompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        outputPath: String
    ): Long? {
        var outcome = "not started"
        try {
            if (!initialized.isCompleted) {
                outcome = "native library not initialised"
                throw IllegalStateException("Imagegen native library is not ready")
            }
            if (!initialized.getCompleted()) {
                outcome = "native library load failed"
                throw IllegalStateException("Imagegen native library failed to load")
            }
            check(isLoaded) { "Image engine not loaded" }
            check(prompt.isNotBlank()) { "Prompt must not be empty" }
            check(_state.value is ImageState.Ready) { "Engine not ready: ${_state.value}" }
            lastError = null
            _state.value = ImageState.Generating

            CriticalLog.log(
                "imagegen",
                "generate(${width}x$height steps=$steps cfg=$cfg seed=$seed) -> native call"
            )
            isGenerating = true
            val usedSeed = try {
                nativeGenerate(prompt, negativePrompt, width, height, steps, cfg, seed, outputPath)
            } finally {
                isGenerating = false
            }
            if (usedSeed < 0) {
                // The context may be damaged (sampler stopped, HTP queue
                // wedged). The engine is marked Error so the next request
                // reloads instead of reusing a broken context - the second
                // picture in one conversation used to fail for that reason.
                outcome = "native failure: ${nativeLastError()}"
                _state.value = ImageState.Error(RuntimeException(outcome))
                lastError = outcome
                return null
            }
            outcome = "ok seed=$usedSeed"
            _state.value = ImageState.Ready
            Log.i(TAG, "Image generated with seed $usedSeed: $outputPath (${File(outputPath).length() / 1024} KB)")
            return usedSeed
        } catch (e: Exception) {
            outcome = "error: ${e.javaClass.simpleName}: ${e.message}"
            _state.value = ImageState.Error(e)
            lastError = e.message
            Log.e(TAG, "Error during image generation", e)
            return null
        } finally {
            CriticalLog.log("imagegen", "generate finished -> $outcome")
        }
    }

    fun destroy() {
        imageScope.launch {
            // Queued behind any in-flight generate on the same thread.
            freeNative()
            Log.i(TAG, "Image engine destroyed")
        }.invokeOnCompletion {
            imageScope.cancel()
        }
    }

    /** Releases the dedicated native thread; call when the engine is replaced. */
    fun shutdownThread() {
        runCatching { imageExecutor.shutdown() }
    }
}
