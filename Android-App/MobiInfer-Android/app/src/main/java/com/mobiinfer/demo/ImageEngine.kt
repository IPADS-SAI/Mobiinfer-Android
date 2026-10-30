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

        private const val INIT_TIMEOUT_MS = 30_000L

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
    private external fun nativeModelVersion(): String
    private external fun nativeFree()

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
            val usedSeed = nativeGenerate(
                prompt, negativePrompt, width, height, steps, cfg, seed, outputPath
            )
            if (usedSeed < 0) {
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
