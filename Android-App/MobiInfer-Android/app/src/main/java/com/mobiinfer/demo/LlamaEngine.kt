package com.mobiinfer.demo

import android.app.ActivityManager
import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.nio.channels.FileLock
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.roundToInt

sealed class LlamaState {
    object Uninitialized : LlamaState()
    object Initializing : LlamaState()
    object Initialized : LlamaState()
    object LoadingModel : LlamaState()
    object ModelReady : LlamaState()
    object ProcessingSystemPrompt : LlamaState()
    object PrefillingImage : LlamaState()
    object ProcessingUserPrompt : LlamaState()
    object Generating : LlamaState()
    object UnloadingModel : LlamaState()
    data class Error(val exception: Exception) : LlamaState()
}

class LlamaEngine private constructor(
    private val context: Context,
    private val nativeLibDir: String
) {

    companion object {
        private val TAG = LlamaEngine::class.java.simpleName

        @Volatile
        private var instance: LlamaEngine? = null

        /** Id of the model currently loaded in the engine (null when none). */
        @Volatile
        var loadedModelId: String? = null
            private set

        // Per-turn token budget. iOS demo (MTMDParams.swift) uses 100 because
        // ANE makes generation fast enough that the cap is rarely hit; on
        // Android with CPU-only inference users complained about replies
        // getting truncated mid-sentence, so we raise it to a safer ceiling.
        // 512 tokens is roughly 250-350 Chinese characters per turn, which
        // covers the vast majority of single-turn answers; the n_ctx=4096
        // buffer plus shift_context() in llama_jni.cpp still keeps multi-turn
        // chats stable.
        const val DEFAULT_PREDICT_LENGTH = 1024

        const val MODEL_SUBDIR = "models"

        private const val PREFS_NAME = "model_prefs"
        private const val KEY_SELECTED_MODEL = "selected_model_id"
        // Tracks the built-in default that has already been force-applied so a
        // changed DEFAULT_MODEL rolls the stored selection forward exactly
        // once, instead of on every launch (which would clobber a user's
        // manual choice).
        private const val KEY_DEFAULT_MODEL_APPLIED = "default_model_applied_id"
        private const val KEY_IMAGE_MAX_SLICE = "image_max_slice_nums"
        private const val KEY_IMAGE_SCALE_FACTOR = "image_scale_factor"
        private const val KEY_MEMORY_BUDGET_MB = "minicpm47_memory_budget_v2_mb"
        private const val KEY_RUNTIME_RELOAD = "minicpm47_memory_budget_reload_pending"
        private const val KEY_CONTEXT_SIZE = "context_size_tokens"
        const val MIN_MEMORY_BUDGET_MB = 5200
        const val DEFAULT_MEMORY_BUDGET_MB = 8000
        const val ABSOLUTE_MAX_MEMORY_BUDGET_MB = 13000
        private const val MEMORY_RESERVE_MB = 3072

        // Hard deadlines for the blocking native load path. Mapping the ~19 GB
        // bundle over slow UFS plus DSP setup is normally well under a minute;
        // these only trip when the backend hangs on a rejected layout.
        const val LOAD_TIMEOUT_SECONDS = 180L
        const val PREPARE_TIMEOUT_SECONDS = 120L

        const val MIN_CONTEXT_SIZE = 1024
        const val DEFAULT_CONTEXT_SIZE = 4096
        const val MAX_CONTEXT_SIZE = 16384
        const val CONTEXT_SIZE_STEP = 1024

        /**
         * Upper bound for the runtime memory budget on this device.
         *
         * The model, vision encoder, context and expert cache all live in this
         * budget, so we leave a reserve for Android, the UI and the HTP driver.
         * The product ceiling is 13 decimal GB, including on larger devices.
         */
        fun getMaxMemoryBudgetMb(context: Context): Int {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val info = ActivityManager.MemoryInfo()
            am.getMemoryInfo(info)
            val totalMb = (info.totalMem / 1_000_000L).toInt()
            val reserveMb = maxOf(MEMORY_RESERVE_MB, totalMb / 5)
            val usableMb = ((totalMb - reserveMb) / 100) * 100
            return usableMb
                .coerceAtLeast(MIN_MEMORY_BUDGET_MB)
                .coerceAtMost(ABSOLUTE_MAX_MEMORY_BUDGET_MB)
        }

        fun getMemoryBudgetMb(context: Context): Int {
            val maxMb = getMaxMemoryBudgetMb(context)
            val preferences = prefs(context)
            val stored = preferences.getInt(KEY_MEMORY_BUDGET_MB, DEFAULT_MEMORY_BUDGET_MB)
            val bounded = stored.coerceIn(MIN_MEMORY_BUDGET_MB, maxMb)
            if (stored != bounded) {
                preferences.edit().putInt(KEY_MEMORY_BUDGET_MB, bounded)
                    .putBoolean(KEY_RUNTIME_RELOAD, true).apply()
            }
            return bounded
        }

        fun setMemoryBudgetMb(context: Context, value: Int) {
            val maxMb = getMaxMemoryBudgetMb(context)
            require(value in MIN_MEMORY_BUDGET_MB..maxMb && value % 100 == 0)
            val current = prefs(context).getInt(KEY_MEMORY_BUDGET_MB, DEFAULT_MEMORY_BUDGET_MB)
            val editor = prefs(context).edit().putInt(KEY_MEMORY_BUDGET_MB, value)
            // The budget is consumed by the native load path, so the model has
            // to be loaded again before the new value can take effect.
            if (value != current) editor.putBoolean(KEY_RUNTIME_RELOAD, true)
            editor.apply()
        }

        fun getMaxContextSize(context: Context): Int =
            if (getSelectedModel(context).id == "minicpm-v-4_7") 4096 else MAX_CONTEXT_SIZE

        fun getContextSize(context: Context): Int = prefs(context)
            .getInt(KEY_CONTEXT_SIZE, DEFAULT_CONTEXT_SIZE)
            .coerceIn(MIN_CONTEXT_SIZE, getMaxContextSize(context))

        fun setContextSize(context: Context, value: Int) {
            require(value in MIN_CONTEXT_SIZE..getMaxContextSize(context) && value % CONTEXT_SIZE_STEP == 0)
            val current = getContextSize(context)
            val editor = prefs(context).edit().putInt(KEY_CONTEXT_SIZE, value)
            if (value != current) editor.putBoolean(KEY_RUNTIME_RELOAD, true)
            editor.apply()
        }

        /** Returns true once after a load-time setting changed. */
        fun consumeRuntimeReloadPending(context: Context): Boolean {
            val prefs = prefs(context)
            val pending = prefs.getBoolean(KEY_RUNTIME_RELOAD, false)
            if (pending) prefs.edit().putBoolean(KEY_RUNTIME_RELOAD, false).apply()
            return pending
        }
        private const val KEY_MODEL_SWITCHED = "model_switched"
        private const val KEY_ENABLE_THINKING = "enable_thinking"

        // Text-to-image generation defaults. Kept in prefs (not per-model) so a
        // MOSS fp16 <-> q4 switch keeps the user's size/step choices.
        private const val KEY_IMAGE_WIDTH = "image_gen_width"
        private const val KEY_IMAGE_HEIGHT = "image_gen_height"
        private const val KEY_IMAGE_STEPS = "image_gen_steps"
        private const val KEY_IMAGE_ENCODE_MS = "image_timing_encode_ms"
        private const val KEY_IMAGE_PER_STEP_MS = "image_timing_per_step_ms"
        private const val KEY_IMAGE_DECODE_MS = "image_timing_decode_ms"
        private const val KEY_IMAGE_CFG = "image_gen_cfg"
        private const val KEY_IMAGE_NEGATIVE = "image_gen_negative_prompt"
        private const val KEY_IMAGE_SEED = "image_gen_seed"
        private const val KEY_IMAGE_LAST_SEED = "image_gen_last_seed"
        private const val KEY_IMAGE_CHAT_TOOL = "image_gen_chat_tool"
        const val MIN_IMAGE_GEN_SIDE = 256
        const val MAX_IMAGE_GEN_SIDE = 1024
        const val IMAGE_GEN_SIDE_STEP = 64
        const val MIN_IMAGE_GEN_STEPS = 1
        const val MAX_IMAGE_GEN_STEPS = 50
        const val DEFAULT_IMAGE_GEN_SIDE = 512
        const val DEFAULT_IMAGE_GEN_STEPS = 8
        const val DEFAULT_IMAGE_GEN_CFG = 3.0f
        // -1 means "pick a fresh random seed for every image". Any other value
        // is reused verbatim so the same prompt can be reproduced.
        const val RANDOM_IMAGE_SEED = -1L

        /**
         * Seeds are shown and typed as four digits: long enough to feel like a
         * code one can share, short enough to read out loud. The picker stays
         * inside 0..9999 so every seed fits the field.
         */
        const val IMAGE_SEED_MAX = 9999L

        fun randomImageSeed(): Long = java.util.Random().nextInt(10_000).toLong()

        fun getImageGenWidth(context: Context): Int =
            prefs(context).getInt(KEY_IMAGE_WIDTH, DEFAULT_IMAGE_GEN_SIDE)
                .coerceIn(MIN_IMAGE_GEN_SIDE, MAX_IMAGE_GEN_SIDE)

        fun setImageGenWidth(context: Context, width: Int) {
            val clamped = width.coerceIn(MIN_IMAGE_GEN_SIDE, MAX_IMAGE_GEN_SIDE)
            prefs(context).edit().putInt(KEY_IMAGE_WIDTH, clamped).apply()
        }

        fun getImageGenHeight(context: Context): Int =
            prefs(context).getInt(KEY_IMAGE_HEIGHT, DEFAULT_IMAGE_GEN_SIDE)
                .coerceIn(MIN_IMAGE_GEN_SIDE, MAX_IMAGE_GEN_SIDE)

        fun setImageGenHeight(context: Context, height: Int) {
            val clamped = height.coerceIn(MIN_IMAGE_GEN_SIDE, MAX_IMAGE_GEN_SIDE)
            prefs(context).edit().putInt(KEY_IMAGE_HEIGHT, clamped).apply()
        }

        fun getImageGenSteps(context: Context): Int =
            prefs(context).getInt(KEY_IMAGE_STEPS, DEFAULT_IMAGE_GEN_STEPS)
                .coerceIn(MIN_IMAGE_GEN_STEPS, MAX_IMAGE_GEN_STEPS)

        fun setImageGenSteps(context: Context, steps: Int) {
            val clamped = steps.coerceIn(MIN_IMAGE_GEN_STEPS, MAX_IMAGE_GEN_STEPS)
            prefs(context).edit().putInt(KEY_IMAGE_STEPS, clamped).apply()
        }

        fun getImageGenCfg(context: Context): Float =
            prefs(context).getFloat(KEY_IMAGE_CFG, DEFAULT_IMAGE_GEN_CFG).coerceIn(1f, 15f)

        fun setImageGenCfg(context: Context, cfg: Float) {
            // Keep one decimal place so the UI never shows 3.0000001.
            val stepped = (cfg.coerceIn(1f, 15f) * 10).roundToInt() / 10f
            prefs(context).edit().putFloat(KEY_IMAGE_CFG, stepped).apply()
        }

        /** Seed of the most recently drawn picture (for "reuse this seed"). */
        fun getLastImageSeed(context: Context): Long =
            prefs(context).getLong(KEY_IMAGE_LAST_SEED, RANDOM_IMAGE_SEED)

        fun setLastImageSeed(context: Context, seed: Long) {
            prefs(context).edit().putLong(KEY_IMAGE_LAST_SEED, seed).apply()
        }

        /**
         * Milliseconds the last draw spent in each phase, used to predict the
         * remaining time of the next one. Zero means "never measured", and the
         * caller falls back to its own defaults.
         */
        data class ImageTiming(
            val encodeMs: Long,
            val perStepMs: Long,
            val decodeMs: Long
        )

        fun getImageTiming(context: Context): ImageTiming = ImageTiming(
            encodeMs = prefs(context).getLong(KEY_IMAGE_ENCODE_MS, 0L),
            perStepMs = prefs(context).getLong(KEY_IMAGE_PER_STEP_MS, 0L),
            decodeMs = prefs(context).getLong(KEY_IMAGE_DECODE_MS, 0L)
        )

        fun setImageTiming(context: Context, encodeMs: Long, perStepMs: Long, decodeMs: Long) {
            if (encodeMs <= 0 || perStepMs <= 0 || decodeMs <= 0) return
            prefs(context).edit()
                .putLong(KEY_IMAGE_ENCODE_MS, encodeMs)
                .putLong(KEY_IMAGE_PER_STEP_MS, perStepMs)
                .putLong(KEY_IMAGE_DECODE_MS, decodeMs)
                .apply()
        }

        fun getImageGenSeed(context: Context): Long =
            prefs(context).getLong(KEY_IMAGE_SEED, RANDOM_IMAGE_SEED)

        /** Clamps to sdcpp's 32-bit seed range; negative keeps it random. */
        fun setImageGenSeed(context: Context, seed: Long) {
            val stored = if (seed < 0) RANDOM_IMAGE_SEED else seed and 0xffffffffL
            prefs(context).edit().putLong(KEY_IMAGE_SEED, stored).apply()
        }

        /**
         * Whether chat models are told about the generate_image tool. Off
         * means the model never hears about image generation; the chat-side
         * text-to-image switch (for image checkpoints) is unaffected.
         */
        fun isImageToolEnabled(context: Context): Boolean =
            prefs(context).getBoolean(KEY_IMAGE_CHAT_TOOL, true)

        fun setImageToolEnabled(context: Context, enabled: Boolean) {
            prefs(context).edit().putBoolean(KEY_IMAGE_CHAT_TOOL, enabled).apply()
        }

        fun getImageNegativePrompt(context: Context): String =
            prefs(context).getString(KEY_IMAGE_NEGATIVE, "") ?: ""

        fun setImageNegativePrompt(context: Context, prompt: String) {
            prefs(context).edit().putString(KEY_IMAGE_NEGATIVE, prompt.trim()).apply()
        }

        /** Directory holding generated PNGs (outside cacheDir so history survives). */
        fun imageOutputDir(context: Context): File =
            File(context.filesDir, "generated_images").apply { mkdirs() }

        // MiniCPM-V's hard upper bound on slice count.  Values higher than 9
        // get clamped by clip.cpp::get_best_grid anyway; we cap on the UI
        // side to keep the slider semantics honest.
        const val MIN_IMAGE_SLICE = 1
        const val MAX_IMAGE_SLICE = 9
        // Out-of-the-box we run with MiniCPM-V's full slice budget (9 =
        // the model's built-in default) so first-launch image quality
        // matches what the model card promises.  Users who care about
        // prefill latency drop the chat-page slider down to 1 (no
        // slicing, ~9x fewer image tokens).
        const val DEFAULT_IMAGE_SLICE = MAX_IMAGE_SLICE
        const val MIN_IMAGE_SCALE_FACTOR = 0.1f
        const val MAX_IMAGE_SCALE_FACTOR = 1.0f
        const val DEFAULT_IMAGE_SCALE_FACTOR = 0.5f

        fun getInstance(context: Context): LlamaEngine =
            instance ?: synchronized(this) {
                instance?.let { return@synchronized it }
                val nativeLibDir = context.applicationInfo.nativeLibraryDir
                require(nativeLibDir.isNotBlank()) { "Expected a valid native library path!" }
                LlamaEngine(context, nativeLibDir).also { instance = it }
            }

        private fun prefs(context: Context): SharedPreferences =
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        fun getSelectedModel(context: Context): ModelInfo {
            val modelId = prefs(context).getString(KEY_SELECTED_MODEL, ModelInfo.DEFAULT_MODEL.id) ?: ModelInfo.DEFAULT_MODEL.id
            return ModelInfo.AVAILABLE_MODELS.find { it.id == modelId } ?: ModelInfo.DEFAULT_MODEL
        }

        fun setSelectedModel(context: Context, modelId: String) {
            // A manual choice must survive the next default-model rollover.
            prefs(context).edit()
                .putString(KEY_SELECTED_MODEL, modelId)
                .putString(KEY_DEFAULT_MODEL_APPLIED, modelId)
                .apply()
        }

        /**
         * Rolls the stored model selection forward to [ModelInfo.DEFAULT_MODEL]
         * once, on the first launch of a build that ships it.
         *
         * Only a selection that was never made explicitly is overridden: a
         * fresh install, or a choice rolled forward by an earlier rollover.
         * A model the user picked by hand (in the toolbar or in the
         * configuration page) is always preserved - otherwise every rebuild
         * with a new default would silently switch the user, e.g. away from
         * the text-to-image checkpoint they are working with.
         */
        fun applyDefaultModelIfNeeded(context: Context) {
            val prefs = prefs(context)
            if (prefs.getString(KEY_DEFAULT_MODEL_APPLIED, null) == ModelInfo.DEFAULT_MODEL.id) return
            val current = prefs.getString(KEY_SELECTED_MODEL, null)
            val editor = prefs.edit().putString(KEY_DEFAULT_MODEL_APPLIED, ModelInfo.DEFAULT_MODEL.id)
            // No stored selection at all: a fresh install, take the default.
            // Anything else is either this rollover's own bookkeeping or a
            // manual pick (setSelectedModel() records it), which is kept.
            if (current == null) {
                editor.putString(KEY_SELECTED_MODEL, ModelInfo.DEFAULT_MODEL.id)
            }
            editor.apply()
        }

        fun markModelSwitched(context: Context) {
            prefs(context).edit().putBoolean(KEY_MODEL_SWITCHED, true).apply()
        }

        fun consumeModelSwitched(context: Context): Boolean {
            val switched = prefs(context).getBoolean(KEY_MODEL_SWITCHED, false)
            if (switched) {
                prefs(context).edit().putBoolean(KEY_MODEL_SWITCHED, false).apply()
            }
            return switched
        }

        // Slice cap is persisted globally (not per-model) - users keep
        // their preferred speed / quality trade-off across model swaps,
        // and v4 / v4.6 / 2.6 all use the same slicing semantics.
        fun getImageMaxSliceNums(context: Context): Int =
            prefs(context).getInt(KEY_IMAGE_MAX_SLICE, DEFAULT_IMAGE_SLICE)
                .coerceIn(MIN_IMAGE_SLICE, MAX_IMAGE_SLICE)

        fun setImageMaxSliceNumsPref(context: Context, n: Int) {
            val clamped = n.coerceIn(MIN_IMAGE_SLICE, MAX_IMAGE_SLICE)
            prefs(context).edit().putInt(KEY_IMAGE_MAX_SLICE, clamped).apply()
        }

        fun getImageScaleFactor(context: Context): Float {
            val modelDefault = getSelectedModel(context).defaultImageScaleFactor
            return prefs(context).getFloat(KEY_IMAGE_SCALE_FACTOR, modelDefault)
                .coerceIn(MIN_IMAGE_SCALE_FACTOR, MAX_IMAGE_SCALE_FACTOR)
        }

        fun setImageScaleFactorPref(context: Context, factor: Float) {
            val clamped = factor.coerceIn(MIN_IMAGE_SCALE_FACTOR, MAX_IMAGE_SCALE_FACTOR)
            val stepped = (clamped * 10).roundToInt() / 10f
            prefs(context).edit().putFloat(KEY_IMAGE_SCALE_FACTOR, stepped).apply()
        }

        fun getEnableThinking(context: Context): Boolean =
            prefs(context).getBoolean(KEY_ENABLE_THINKING, false)

        fun setEnableThinkingPref(context: Context, enable: Boolean) {
            prefs(context).edit().putBoolean(KEY_ENABLE_THINKING, enable).apply()
        }

        fun modelDir(context: Context): String =
            File(context.filesDir, MODEL_SUBDIR).absolutePath

        fun modelDirFor(context: Context, model: ModelInfo): String =
            File(modelDir(context), model.id).absolutePath

        fun modelPath(context: Context): String = modelPath(context, getSelectedModel(context))

        fun modelPath(context: Context, model: ModelInfo): String =
            File(modelDirFor(context, model), model.ggufFileName).absolutePath

        fun mmprojPath(context: Context): String? {
            val model = getSelectedModel(context)
            val mmproj = model.mmprojFileName ?: return null
            return File(modelDirFor(context, model), mmproj).absolutePath
        }

        fun acousticPath(context: Context): String? {
            val model = getSelectedModel(context)
            val acoustic = model.acousticFileName ?: return null
            return File(modelDirFor(context, model), acoustic).absolutePath
        }

        // MOSS-Image accessory file paths (derived from extraFiles order:
        // [text encoder, VAE, tokenizer]). Return null if the model has no
        // matching extra files configured.
        fun imageTextEncoderPath(context: Context): String? =
            imageTextEncoderPath(context, getSelectedModel(context))

        fun imageVaePath(context: Context): String? =
            imageVaePath(context, getSelectedModel(context))

        fun imageTokenizerPath(context: Context): String? =
            imageTokenizerPath(context, getSelectedModel(context))

        fun imageTextEncoderPath(context: Context, model: ModelInfo): String? =
            imageExtraPath(context, model, 0)

        fun imageVaePath(context: Context, model: ModelInfo): String? =
            imageExtraPath(context, model, 1)

        fun imageTokenizerPath(context: Context, model: ModelInfo): String? =
            imageExtraPath(context, model, 2)

        private fun imageExtraPath(context: Context, model: ModelInfo, index: Int): String? {
            val name = model.extraFiles.getOrNull(index) ?: return null
            return File(modelDirFor(context, model), name).absolutePath
        }

        fun modelsExist(context: Context): Boolean =
            modelsExist(context, getSelectedModel(context))

        /** Per-model completeness check (any model, not just the selection). */
        fun modelsExist(context: Context, model: ModelInfo): Boolean {
            val dir = File(modelDirFor(context, model))
            if (missingExpertFile(context, model) != null) return false
            if (!File(dir, model.ggufFileName).exists()) return false
            if (model.extraFiles.isNotEmpty()) {
                return model.extraFiles.all { File(dir, it).let { f -> f.isFile && f.canRead() && f.length() > 0L } }
            }
            if (model.isTextOnly) return true
            if (model.isTts) {
                val acoustic = model.acousticFileName ?: return true
                return File(dir, acoustic).exists()
            }
            val mmproj = model.mmprojFileName ?: return true
            return File(dir, mmproj).exists()
        }

        /** Bytes currently on disk for [model] (including partial `.tmp` files). */
        fun downloadedBytes(context: Context, model: ModelInfo): Long {
            val dir = File(modelDirFor(context, model))
            if (!dir.exists()) return 0L
            var total = 0L
            dir.walkTopDown().forEach { if (it.isFile) total += it.length() }
            return total
        }

        /** True when some bytes are present but the model is still incomplete. */
        fun isPartial(context: Context, model: ModelInfo): Boolean =
            !modelsExist(context, model) && downloadedBytes(context, model) > 0L

        private fun missingExpertFile(context: Context, model: ModelInfo): File? {
            // Bundle payloads are enumerated by the SHA256SUMS manifest, so the
            // draft GGUF (if any) is not otherwise tracked here; check it too,
            // otherwise a missing draft would look like a complete download.
            model.draftFileName?.let { draft ->
                val draftFile = File(modelDirFor(context, model), draft)
                if (!draftFile.isFile || !draftFile.canRead() || draftFile.length() == 0L) {
                    return draftFile
                }
            }
            val name = model.expertsDirectory ?: return null
            val dir = File(modelDirFor(context, model), name)
            return (listOf(File(dir, "manifest.json")) + (0 until model.expertsLayerCount).map {
                File(dir, "layer_%03d.bin".format(java.util.Locale.ROOT, it))
            }).firstOrNull { !it.isFile || !it.canRead() || it.length() == 0L }
        }

        // Rename map for files that were previously sideloaded with a
        // different name and now need to match the iOS demo's filenames so a
        // re-push isn't required. Keyed by ModelInfo.id; values are
        // (oldName -> newName) pairs scoped to that model's sub-directory.
        // Maps any historical filename (left) to the current canonical name
        // (right) for files that may already exist in a user's per-model
        // sub-directory. Each entry is consulted on app start; if a left-hand
        // file is found it is renamed to the right-hand one. MD5 checks then
        // either accept it or trigger a re-download. Keep the rename targets
        // in sync with [ModelInfo.AVAILABLE_MODELS].
        private val LEGACY_FILE_RENAMES: Map<String, List<Pair<String, String>>> = mapOf(
            "minicpm-v-4_6-instruct" to listOf(
                // Earliest sideloaded LLM name, then our previous demo name,
                // both rolled forward to the released HF filename. The GGUF
                // content is byte-stable across these names (LFS-hashed on
                // HF, MD5-pinned on OBS) so renaming is safe.
                "MiniCPM-V4.6-instruct-Q4_K_M.gguf" to "MiniCPM-V-4_6-Q4_K_M.gguf",
                "minicpmv46-llm-Q4_K_M.gguf" to "MiniCPM-V-4_6-Q4_K_M.gguf"
                // NOTE: deliberately *not* renaming the legacy mmproj
                // filenames here; their bytes are not necessarily compatible
                // with the demo's current clip.cpp. They are purged outright
                // by [STALE_MMPROJ_NAMES] below.
            ),
            "minicpm5-0.9b" to listOf(
                // Pre-release name (`0.9B`) → post-release HF/MS name (`1B`).
                // The pre-release gguf came from a pre-public OBS bucket and
                // may be byte-identical to or slightly different from the
                // OpenBMB-published HF/MS variant; the downstream MD5/LFS
                // check inside downloadFile will accept the matching bytes
                // or quietly trigger a re-download, so renaming is always
                // safe.
                "MiniCPM5-0.9B-Q4_K_M.gguf" to "MiniCPM5-1B-Q4_K_M.gguf"
            ),
            "voxcpm2" to listOf(
                // BaseLM was quantized from F16 to Q4_K_M (~3.1 GB → ~1.0 GB).
                "VoxCPM2-BaseLM-F16.gguf" to "VoxCPM2-BaseLM-Q4_K_M.gguf"
            )
        )

        // Pre-v2.0 the v4.6 mmproj came from a temporary OBS bucket and went
        // through several incompatible revisions (pre-release / sealed
        // minicpmv4_6 / demo-merger). To keep the rollout clean we purge any
        // file that matches a known stale name; modelsExist() then reports
        // "missing" and the new HF/MS racing download fetches the canonical
        // OpenBMB-published `mmproj-model-f16.gguf` (whose MD5 is pinned in
        // ModelInfo and verified after download).
        private val STALE_MMPROJ_NAMES: Map<String, List<String>> = mapOf(
            "minicpm-v-4_6-instruct" to listOf(
                "mmproj-v46-model-f16.gguf",
                // v1.7-v1.9 demo: bytes converted on the Support-iOS-Demo
                // branch (clip.projector_type=merger). Upstream master mtmd
                // accepts the original minicpmv4_6 projector natively so we
                // no longer need this re-conversion; the file would also
                // fail the new MD5 anyway.
                "mmproj-model-merger-f16.gguf"
                // Intentionally NOT listing "mmproj-model-f16.gguf" here:
                // that IS the canonical name from v2.0 onwards. If a user
                // has a stale copy from an even earlier OBS revision sitting
                // under that name, the MD5 check inside downloadFile will
                // delete it and re-download cleanly.
            )
        )

        // One-shot migration: previously all model files lived flat under models/.
        // With multiple models (v4, v4.6, ...) some filenames collide (e.g. mmproj-model-f16.gguf),
        // so each model now gets its own subdirectory keyed by ModelInfo.id.
        // Also handles renaming files that have been sideloaded under a
        // legacy name (see [LEGACY_FILE_RENAMES]).
        fun migrateLegacyLayoutIfNeeded(context: Context) {
            val rootDir = File(modelDir(context))
            if (!rootDir.exists()) return

            for (model in ModelInfo.AVAILABLE_MODELS) {
                val targetDir = File(rootDir, model.id)
                val flatGguf = File(rootDir, model.ggufFileName)
                val flatMmproj = model.mmprojFileName?.let { File(rootDir, it) }
                if (flatGguf.exists() || (flatMmproj != null && flatMmproj.exists())) {
                    if (!targetDir.exists()) targetDir.mkdirs()
                    if (flatGguf.exists()) {
                        val dst = File(targetDir, model.ggufFileName)
                        if (!dst.exists() && flatGguf.renameTo(dst)) {
                            Log.i(TAG, "Migrated legacy ${model.ggufFileName} into ${model.id}/")
                        }
                    }
                    if (flatMmproj != null && flatMmproj.exists() && model.mmprojFileName != null) {
                        val dst = File(targetDir, model.mmprojFileName)
                        if (!dst.exists() && flatMmproj.renameTo(dst)) {
                            Log.i(TAG, "Migrated legacy ${model.mmprojFileName} into ${model.id}/")
                        }
                    }
                }

                LEGACY_FILE_RENAMES[model.id]?.let { renames ->
                    val perModelDir = File(rootDir, model.id)
                    if (!perModelDir.exists()) return@let
                    for ((oldName, newName) in renames) {
                        val src = File(perModelDir, oldName)
                        val dst = File(perModelDir, newName)
                        if (src.exists() && !dst.exists() && src.renameTo(dst)) {
                            Log.i(TAG, "Renamed legacy ${model.id}/$oldName -> $newName")
                        }
                    }
                }

                // Purge any stale mmproj cached under a previous filename:
                // the OBS object behind that name has been re-converted at
                // least once and the cached bytes may be incompatible with
                // the demo's current clip.cpp (would crash inside
                // mtmd_init_from_file). Companion .tmp from a partial old
                // download is dropped too. This is intentionally one-way -
                // we never try to "rescue" the bytes by renaming.
                STALE_MMPROJ_NAMES[model.id]?.let { stale ->
                    val perModelDir = File(rootDir, model.id)
                    if (!perModelDir.exists()) return@let
                    for (name in stale) {
                        val f = File(perModelDir, name)
                        if (f.exists() && f.delete()) {
                            Log.i(TAG, "Purged stale mmproj ${model.id}/$name (incompatible OBS revision)")
                        }
                        val tmp = File(perModelDir, "$name.tmp")
                        if (tmp.exists() && tmp.delete()) {
                            Log.i(TAG, "Purged stale mmproj tmp ${model.id}/$name.tmp")
                        }
                    }
                }
            }
        }

        // Maps a selected [ModelInfo] to the matching MiniCPM-V family version
        // tag that the (legacy) fork mtmd-ios.h would have returned via
        // mtmd_get_minicpmv_version.  Used in [loadModel] to push the value
        // down to native after loadMmproj.
        //   - 46  : MiniCPM-V-4.6 instruct (the demo's only V-4.6 build)
        //   - 5   : MiniCPM-V-4 (corresponds to upstream `clip.minicpmv_version=5`)
        //   - 0   : no match / unknown -> native treats as "no vision-version
        //           gating; use default n_ctx + plain assistant prefix"
        // Kept in the companion object so [downloadModels] / [loadModel] /
        // any future callers can reuse the same mapping without re-grepping
        // model ids in three places.
        fun inferMinicpmvVersion(model: ModelInfo): Int = when (model.id) {
            "minicpm-v-4_7"          -> 47
            "minicpm-v-4_6-instruct" -> 46
            "minicpm-v-4"            -> 5
            else                     -> 0
        }

        // Per-file source descriptor used by [downloadFileMultiSource] / racing.
        private data class FileSource(val label: String, val url: URL)

        /**
         * One tick of download progress. [fraction] is the *overall* share of
         * [ModelInfo.totalBytes] fetched so far (across every file), so the
         * UI can drive a single determinate bar instead of a per-file one.
         */
        data class DownloadProgress(
            val message: String,
            val downloadedBytes: Long,
            val totalBytes: Long,
        ) {
            val fraction: Float
                get() = if (totalBytes > 0L) {
                    (downloadedBytes.toDouble() / totalBytes.toDouble()).toFloat().coerceIn(0f, 1f)
                } else 0f
        }

        /**
         * Accumulates byte counts across the files of one download so each
         * tick reports a single overall fraction: [completedBytes] holds the
         * finalized files, and [overall] adds the bytes read so far in the
         * file currently streaming.
         */
        private class ProgressSink(
            private val context: Context,
            private val grandTotal: Long,
            private val emit: (DownloadProgress) -> Unit,
        ) {
            private var completedBytes = 0L
            private var lastSampleTime = 0L
            private var lastSampleBytes = 0L
            private var smoothedBytesPerSec = 0.0

            /** Emit an informational message at the current overall offset. */
            fun message(text: String) = emit(DownloadProgress(text, completedBytes, grandTotal))

            /** A file was already complete on disk: add it to progress, not to network speed. */
            fun fileSkipped(bytes: Long) {
                completedBytes += bytes
                lastSampleTime = 0L
                overall(0L)
            }

            /** We are resuming an on-disk `.tmp`: exclude its prefix from network speed. */
            fun resuming(resumeFrom: Long) {
                lastSampleTime = 0L
                overall(resumeFrom)
            }

            /** Emit ongoing progress: [downloadedInFile] bytes into the current file. */
            fun overall(downloadedInFile: Long) {
                val done = completedBytes + downloadedInFile
                val now = System.currentTimeMillis()
                if (lastSampleTime > 0L) {
                    val dtMs = now - lastSampleTime
                    if (dtMs > 0L) {
                        val inst = (done - lastSampleBytes).toDouble() * 1000.0 / dtMs
                        if (inst >= 0.0) {
                            smoothedBytesPerSec =
                                if (smoothedBytesPerSec <= 0.0) inst else smoothedBytesPerSec * 0.7 + inst * 0.3
                        }
                    }
                }
                lastSampleTime = now
                lastSampleBytes = done

                val pct = if (grandTotal > 0L) {
                    ((done * 100L) / grandTotal).toInt().coerceIn(0, 100)
                } else 0
                val mbPerSec = smoothedBytesPerSec / (1024.0 * 1024.0)
                val remaining = (grandTotal - done).coerceAtLeast(0L)
                val etaText = if (smoothedBytesPerSec > 1024.0) {
                    formatEta((remaining / smoothedBytesPerSec).toLong())
                } else {
                    context.getString(R.string.eta_unknown)
                }
                emit(DownloadProgress(
                    context.getString(R.string.download_overall_progress, pct, mbPerSec, etaText),
                    done,
                    grandTotal
                ))
            }

            private fun formatEta(totalSec: Long): String {
                val h = totalSec / 3600
                val m = (totalSec % 3600) / 60
                val sec = totalSec % 60
                return when {
                    h > 0 -> context.getString(R.string.eta_hours_minutes, h, m)
                    m > 0 -> context.getString(R.string.eta_minutes_seconds, m, sec)
                    else -> context.getString(R.string.eta_seconds, sec)
                }
            }

            /** Record a file that finished (or was already present) on disk. */
            fun fileFinished(actualBytes: Long) {
                completedBytes += actualBytes
                // Re-baseline the speed sampler so a skipped/completed file's
                // instantaneous byte jump isn't read as a burst of throughput.
                lastSampleTime = 0L
                overall(0L)
            }
        }

        // Racing winner state passed from header-probe goroutines to the
        // body-streaming logic. [conn] is the still-open HttpURLConnection
        // whose response headers we accepted; the loser connections are
        // disconnected before [body] is reached.
        private data class RaceWinner(
            val index: Int,
            val source: String,
            val conn: HttpURLConnection,
            val responseCode: Int,
            val contentLength: Long,
        )

        suspend fun downloadModels(
            context: Context,
            onProgress: (DownloadProgress) -> Unit
        ) = downloadModels(context, getSelectedModel(context), onProgress)

        suspend fun downloadModels(
            context: Context,
            modelId: String,
            onProgress: (DownloadProgress) -> Unit
        ) = downloadModels(
            context,
            ModelInfo.AVAILABLE_MODELS.firstOrNull { it.id == modelId } ?: getSelectedModel(context),
            onProgress
        )

        suspend fun downloadModels(
            context: Context,
            model: ModelInfo,
            onProgress: (DownloadProgress) -> Unit
        ) = withContext(Dispatchers.IO) {
            val dir = File(modelDirFor(context, model))
            if (!dir.exists()) dir.mkdirs()
            val sink = ProgressSink(context, model.totalBytes, onProgress)

            // Cross-process guard: two app processes (e.g. after the service is
            // restarted while a background download is still running) must never
            // write the same per-model directory concurrently - they would each
            // delete the other's `.tmp` and corrupt/reset the download. An OS
            // file lock on `<dir>/.download.lock` makes that exclusive.
            val lock = acquireDownloadLock(dir)
            if (lock == null) {
                Log.w(TAG, "Download already in progress for ${model.id} in another process; refusing to start")
                throw IllegalStateException("该模型正在另一个进程中下载，请稍后再试")
            }
            try {

            // Bundle models (e.g. the 4.7 Hexagon package) are more than a
            // single gguf/mmproj pair: a ModelScope-hosted checksum manifest
            // enumerates every payload file. Handle them on their own path and
            // skip the racing single-file logic below.
            val bundleManifest = model.bundleManifestFile
            if (bundleManifest != null) {
                val msBase = model.msRepo?.takeIf { it.isNotBlank() }
                    ?.let { "https://www.modelscope.cn/models/$it/resolve/${model.msBranch}" }
                    ?: throw RuntimeException("Model ${model.id} has no ModelScope repo for its bundle manifest")
                downloadBundle(context, dir, msBase, bundleManifest, sink)
                sink.message(context.getString(R.string.download_all_complete))
                return@withContext
            }

            // Per file, build the *full* candidate-source list. HF + MS run
            // in parallel; directGgufUrl/directMmprojUrl, if present, are
            // registered as a third candidate. The first source to deliver
            // valid response headers wins, the others get disconnected.
            // Mirrors iOS opt-r1's MBModelDownloadHelperV2.downloadV2.
            data class FileSpec(
                val name: String,
                val sources: List<FileSource>,
                val md5: String?,
                val bytes: Long?,
            )

            val ggufSources = mutableListOf<FileSource>()
            val mmprojSources = mutableListOf<FileSource>()
            val acousticSources = mutableListOf<FileSource>()

            if (model.hasRemoteSources) {
                // HF and MS are registered independently so a model can be
                // served by just one of them (e.g. Qwen3-VL is ModelScope-only).
                // A single source is passed to downloadFileMultiSource, which
                // bypasses the race plumbing and streams directly.
                val hfBase = model.hfRepo?.takeIf { it.isNotBlank() }
                    ?.let { "https://huggingface.co/$it/resolve/${model.hfBranch}" }
                val msBase = model.msRepo?.takeIf { it.isNotBlank() }
                    ?.let { "https://www.modelscope.cn/models/$it/resolve/${model.msBranch}" }

                fun addRemoteSources(target: MutableList<FileSource>, remotePath: String) {
                    hfBase?.let { target.add(FileSource("HuggingFace", URL("$it/$remotePath"))) }
                    msBase?.let { target.add(FileSource("ModelScope", URL("$it/$remotePath"))) }
                }

                addRemoteSources(ggufSources, model.ggufRemotePath)
                if (model.isTts) {
                    model.acousticRemotePath?.let { addRemoteSources(acousticSources, it) }
                } else if (!model.isTextOnly) {
                    model.mmprojRemotePath?.let { addRemoteSources(mmprojSources, it) }
                }
            }
            if (!model.directGgufUrl.isNullOrBlank()) {
                ggufSources.add(FileSource(context.getString(R.string.source_direct), URL(model.directGgufUrl)))
            }
            if (model.isTts && !model.directAcousticUrl.isNullOrBlank()) {
                acousticSources.add(FileSource(context.getString(R.string.source_direct), URL(model.directAcousticUrl)))
            }
            if (!model.isTextOnly && !model.isTts && !model.directMmprojUrl.isNullOrBlank()) {
                mmprojSources.add(FileSource(context.getString(R.string.source_direct), URL(model.directMmprojUrl)))
            }

            if (ggufSources.isEmpty()) {
                throw RuntimeException("Model ${model.id} has no GGUF download source configured")
            }
            if (model.isTts && acousticSources.isEmpty()) {
                throw RuntimeException("Model ${model.id} has no acoustic GGUF download source configured")
            }
            if (!model.isTextOnly && !model.isTts && mmprojSources.isEmpty()) {
                throw RuntimeException("Model ${model.id} has no mmproj download source configured")
            }

            val files = mutableListOf(
                FileSpec(model.ggufFileName, ggufSources, model.ggufMd5, model.ggufBytes),
            )
            if (model.isTts && model.acousticFileName != null) {
                files.add(FileSpec(model.acousticFileName, acousticSources, model.acousticMd5, model.acousticBytes))
            }
            if (!model.isTextOnly && !model.isTts && model.mmprojFileName != null) {
                files.add(FileSpec(model.mmprojFileName, mmprojSources, model.mmprojMd5, model.mmprojBytes))
            }
            // Extra accessory files (e.g. MOSS-Image TE / VAE / tokenizer),
            // fetched from the ModelScope repo only.
            if (model.extraFiles.isNotEmpty()) {
                val msBase = model.msRepo?.takeIf { it.isNotBlank() }
                    ?.let { "https://www.modelscope.cn/models/$it/resolve/${model.msBranch}" }
                    ?: throw RuntimeException("Model ${model.id} has extraFiles but no ModelScope repo")
                for (name in model.extraFiles) {
                    files.add(
                        FileSpec(
                            name,
                            listOf(FileSource("ModelScope", URL("$msBase/$name"))),
                            null,
                            model.extraFileBytes[name],
                        )
                    )
                }
            }

            val racedLabel = files.first().sources.joinToString("+") { it.label }
            sink.message(context.getString(R.string.download_race_start, racedLabel))

            for (file in files) {
                downloadFileMultiSource(context, dir, file.name, file.sources, file.md5, sink, expectedBytes = file.bytes)
            }

            sink.message(context.getString(R.string.download_all_complete))
            } finally {
                releaseDownloadLock(lock)
            }
        }

        // Cross-process download lock, keyed on the per-model directory.
        // A held lock keeps its RandomAccessFile open; we store it by path so
        // release can both drop the FileLock and close the RAF (closing alone
        // would release the OS lock, but we close explicitly for tidiness).
        private val downloadLocks = ConcurrentHashMap<String, RandomAccessFile>()

        private fun acquireDownloadLock(dir: File): FileLock? {
            val key = dir.absolutePath
            val raf = try {
                RandomAccessFile(File(dir, ".download.lock"), "rw")
            } catch (e: Exception) {
                Log.w(TAG, "acquireDownloadLock: cannot open lock file: ${e.message}")
                return null
            }
            return try {
                val l = raf.channel.tryLock()
                if (l == null) {
                    raf.close()
                    null
                } else {
                    downloadLocks[key] = raf
                    l
                }
            } catch (e: Exception) {
                Log.w(TAG, "acquireDownloadLock: tryLock failed: ${e.message}")
                try { raf.close() } catch (_: Exception) {}
                null
            }
        }

        private fun releaseDownloadLock(lock: FileLock) {
            try { if (lock.isValid) lock.release() } catch (_: Exception) {}
            // Close and forget every RAF whose channel no longer holds a lock.
            val it = downloadLocks.entries.iterator()
            while (it.hasNext()) {
                val (_, raf) = it.next()
                try {
                    val held = raf.channel.tryLock()
                    if (held != null) {
                        held.release()
                        raf.close()
                        it.remove()
                    }
                } catch (_: Exception) {
                    try { raf.close() } catch (_: Exception) {}
                    it.remove()
                }
            }
        }

        /**
         * Multi-source racing download.
         *
         * Launches one HttpURLConnection GET per [sources] entry in parallel.
         * The first connection whose response headers indicate success
         * (HTTP 200/206 with positive Content-Length, or 416 with a valid
         * Content-Range matching the local `.tmp`) wins; the others are
         * disconnected immediately. The winner's body is streamed to disk
         * exactly the same way [downloadFile] does it for the single-source
         * path, including resume semantics, MD5 verification, and progress
         * reporting tagged with the winning source label.
         *
         * If every source fails before any of them produces headers, the
         * race surfaces a single combined RuntimeException to the caller.
         *
         * Mirrors iOS opt-r1's MBModelDownloadHelperV2.downloadV2 (race the
         * URLSessionDownloadTask `didReceiveResponse` callbacks). Implementing
         * this on Android uses `HttpURLConnection.responseCode` as the
         * synchronous header gate — it blocks the worker dispatcher thread
         * until the server sends the first byte of the status line, which is
         * the exact moment we want to compare for a winner.
         */
        private suspend fun downloadFileMultiSource(
            context: Context,
            dir: File,
            fileName: String,
            sources: List<FileSource>,
            expectedMd5: String?,
            sink: ProgressSink,
            expectedSha256: String? = null,
            expectedBytes: Long? = null,
        ) {
            require(sources.isNotEmpty()) { "downloadFileMultiSource: empty sources" }

            // Single source -> bypass race plumbing entirely.
            if (sources.size == 1) {
                val (label, url) = sources.first()
                downloadFile(context, dir, fileName, url, label, expectedMd5, sink, expectedSha256, expectedBytes)
                return
            }

            val targetFile = File(dir, fileName)
            val tmpFile = File(dir, "$fileName.tmp")

            // Fast path: same as downloadFile - if target already exists and
            // matches its expectation, treat as done.
            if (targetFile.exists() &&
                (expectedMd5 != null || expectedSha256 != null || expectedBytes != null)
            ) {
                sink.message(context.getString(R.string.download_verifying_existing, fileName))
                if (matchesExpectation(targetFile, expectedMd5, expectedSha256, expectedBytes)) {
                    sink.fileSkipped(targetFile.length())
                    sink.message(context.getString(R.string.download_file_ready, fileName))
                    Log.i(TAG, "$fileName already present and matches expectation, skipping download")
                    if (tmpFile.exists()) tmpFile.delete()
                    return
                }
                Log.w(TAG, "$fileName already present but does not match expectation; re-downloading")
                targetFile.delete()
                if (tmpFile.exists()) tmpFile.delete()
            }

            val resumeFrom = if (tmpFile.exists()) tmpFile.length() else 0L
            if (resumeFrom > 0) {
                sink.message(context.getString(R.string.download_resuming, fileName, (resumeFrom / (1024 * 1024)).toInt()))
                Log.i(TAG, "Resuming $fileName from $resumeFrom bytes via ${sources.size}-way race")
            } else {
                sink.message(context.getString(R.string.download_race_fetching, sources.joinToString("+") { it.label }, fileName))
                Log.i(TAG, "Race-downloading $fileName from ${sources.size} sources")
            }

            coroutineScope {
                val winnerSlot = CompletableDeferred<RaceWinner>()
                val failedCount = AtomicInteger(0)
                val totalSources = sources.size
                val openConns = ConcurrentHashMap<Int, HttpURLConnection>()

                val jobs = sources.mapIndexed { idx, src ->
                    launch(Dispatchers.IO) {
                        var conn: HttpURLConnection? = null
                        try {
                            conn = (src.url.openConnection() as HttpURLConnection).apply {
                                connectTimeout = 10_000
                                readTimeout = 120_000
                                requestMethod = "GET"
                                instanceFollowRedirects = true
                                setRequestProperty("User-Agent", "MiniCPMV-demo/1.0")
                                if (resumeFrom > 0) {
                                    setRequestProperty("Range", "bytes=$resumeFrom-")
                                }
                            }
                            openConns[idx] = conn
                            Log.i(TAG, "race[$idx] ${src.label} -> ${src.url} start")

                            val code = conn.responseCode

                            // 416 == server thinks we already have everything.
                            // Treat as a valid winner with len=-1 so the body
                            // loop short-circuits and just finalises .tmp.
                            if (code == 416) {
                                val totalLen = parseContentRangeTotal(conn.getHeaderField("Content-Range"))
                                if (totalLen != null && tmpFile.length() == totalLen) {
                                    if (winnerSlot.complete(RaceWinner(idx, src.label, conn!!, 416, -1L))) return@launch
                                    conn.disconnect()
                                    return@launch
                                }
                                throw RuntimeException("$src.label: HTTP 416 but local .tmp size ${tmpFile.length()} != server $totalLen")
                            }

                            if (code !in 200..206) {
                                throw RuntimeException("${src.label}: HTTP $code")
                            }
                            val contentLen = conn.contentLength.toLong()
                            if (contentLen <= 0L) {
                                // Some mirrors return chunked-encoded responses
                                // without Content-Length. Still acceptable, the
                                // streaming loop just won't show a percentage.
                                Log.w(TAG, "race[$idx] ${src.label}: headers OK but no Content-Length")
                            }
                            Log.i(TAG, "race[$idx] ${src.label} headers OK: code=$code len=$contentLen")

                            if (!winnerSlot.complete(RaceWinner(idx, src.label, conn!!, code, contentLen))) {
                                // Lost the race - drop the connection so the
                                // server-side socket is closed promptly.
                                conn.disconnect()
                            }
                        } catch (e: CancellationException) {
                            conn?.disconnect()
                            throw e
                        } catch (e: Throwable) {
                            try { conn?.disconnect() } catch (_: Throwable) {}
                            val fc = failedCount.incrementAndGet()
                            Log.w(TAG, "race[$idx] ${src.label} failed (${fc}/${totalSources}): ${e.message}")
                            if (fc >= totalSources && !winnerSlot.isCompleted) {
                                winnerSlot.completeExceptionally(
                                    RuntimeException("$fileName: all $totalSources sources failed")
                                )
                            }
                        }
                    }
                }

                val winner = winnerSlot.await()
                sink.message(context.getString(R.string.download_race_winner, winner.source, fileName))
                Log.i(TAG, "race winner for $fileName: idx=${winner.index} source=${winner.source} code=${winner.responseCode}")

                // Cancel everyone else first so their sockets stop pulling
                // bytes before we start streaming the winner.
                jobs.forEachIndexed { idx, job ->
                    if (idx != winner.index) {
                        openConns[idx]?.disconnect()
                        job.cancel()
                    }
                }

                streamWinnerToDisk(context, winner, tmpFile, targetFile, fileName, expectedMd5, resumeFrom, sink, expectedSha256)
            }
        }

        /**
         * Streams a successful [winner] connection body into [tmpFile], then
         * promotes to [targetFile] and verifies MD5. Carries the resume
         * semantics from [downloadFile] (206 -> append, 200 -> truncate, 416
         * -> finalise as-is) so race winners are observationally equivalent
         * to single-source downloads.
         */
        private fun streamWinnerToDisk(
            context: Context,
            winner: RaceWinner,
            tmpFile: File,
            targetFile: File,
            fileName: String,
            expectedMd5: String?,
            resumeFrom: Long,
            sink: ProgressSink,
            expectedSha256: String? = null,
        ) {
            val source = winner.source
            val conn = winner.conn
            try {
                if (winner.responseCode == 416) {
                    Log.i(TAG, "$fileName: server reports complete (HTTP 416 via race)")
                } else {
                    val acceptedResume = winner.responseCode == HttpURLConnection.HTTP_PARTIAL // 206
                    val effectiveStart = if (acceptedResume) resumeFrom else 0L
                    val totalSize = if (winner.contentLength > 0) winner.contentLength + effectiveStart else -1L

                    if (winner.responseCode == HttpURLConnection.HTTP_OK && resumeFrom > 0) {
                        Log.w(TAG, "$fileName: server returned 200 instead of 206; restarting from 0")
                        if (tmpFile.exists()) tmpFile.delete()
                    }
                    sink.resuming(effectiveStart)

                    conn.inputStream.use { input ->
                        FileOutputStream(tmpFile, acceptedResume && resumeFrom > 0).use { output ->
                            val buffer = ByteArray(64 * 1024)
                            var totalRead = effectiveStart
                            var lastProgressTime = 0L
                            while (true) {
                                val read = input.read(buffer)
                                if (read == -1) break
                                output.write(buffer, 0, read)
                                totalRead += read
                                val now = System.currentTimeMillis()
                                if (now - lastProgressTime > 500) {
                                    lastProgressTime = now
                                    if (totalSize > 0) {
                                        Log.i(TAG, "$fileName: ${totalRead * 100 / totalSize}% ($source)")
                                    }
                                    sink.overall(totalRead)
                                }
                            }
                            output.flush()
                            try {
                                output.fd.sync()
                            } catch (e: Throwable) {
                                Log.w(TAG, "fsync failed for $fileName (continuing): ${e.message}")
                            }
                        }
                    }
                }

                if (targetFile.exists()) targetFile.delete()
                if (!tmpFile.renameTo(targetFile)) {
                    tmpFile.copyTo(targetFile, overwrite = true)
                    tmpFile.delete()
                }

                verifyOrThrow(context, targetFile, source, expectedMd5, expectedSha256, sink, tmpFile)

                sink.fileFinished(targetFile.length())
                sink.message(context.getString(R.string.download_file_complete, fileName, (targetFile.length() / (1024 * 1024)).toInt()))
                Log.i(TAG, "$fileName saved to ${targetFile.absolutePath} (winner=$source)")
            } finally {
                try { conn.disconnect() } catch (_: Throwable) {}
            }
        }

        // Downloads a single file with HTTP Range-based resume. The on-disk
        // staging file is `<fileName>.tmp`; if it already exists from a prior
        // (interrupted) attempt we re-issue the request with `Range: bytes=N-`
        // so the server only sends the missing tail. The .tmp file is
        // deliberately preserved across cancellations / process death so the
        // user can resume seamlessly the next time `downloadModels` runs.
        //
        // Server response handling:
        //   - 200 OK  : server ignored Range (or .tmp was empty) -> overwrite from 0
        //   - 206 Partial Content : append missing tail to existing .tmp
        //   - 416 Range Not Satisfiable : .tmp is already the full file -> finalize
        private fun downloadFile(
            context: Context,
            dir: File,
            fileName: String,
            url: URL,
            source: String,
            expectedMd5: String?,
            sink: ProgressSink,
            expectedSha256: String? = null,
            expectedBytes: Long? = null,
        ) {
            val targetFile = File(dir, fileName)
            val tmpFile = File(dir, "$fileName.tmp")

            // Fast path: if the file is already on disk and matches its
            // expectation (hash preferred, size otherwise), skip re-downloading.
            // Saves 500MB-1GB on app reinstall / dev iteration, and prevents a
            // partial re-download from re-fetching files that are already
            // complete (e.g. Qwen3-VL has no pinned hash, only a size).
            if (targetFile.exists() &&
                (expectedMd5 != null || expectedSha256 != null || expectedBytes != null)
            ) {
                sink.message(context.getString(R.string.download_verifying_existing_src, source, fileName))
                if (matchesExpectation(targetFile, expectedMd5, expectedSha256, expectedBytes)) {
                    sink.fileSkipped(targetFile.length())
                    sink.message(context.getString(R.string.download_file_ready, fileName))
                    Log.i(TAG, "$fileName already present and matches expectation, skipping download")
                    // Stale .tmp from an even older interrupted attempt - drop it.
                    if (tmpFile.exists()) tmpFile.delete()
                    return
                }
                Log.w(TAG, "$fileName already present but does not match expectation; re-downloading")
                targetFile.delete()
                // The cached .tmp (if any) belonged to a previous, mismatching
                // version; better to start clean.
                if (tmpFile.exists()) tmpFile.delete()
            }

            val resumeFrom = if (tmpFile.exists()) tmpFile.length() else 0L
            Log.i(TAG, "$fileName: start, tmpExists=${tmpFile.exists()} resumeFrom=$resumeFrom targetExists=${targetFile.exists()}")
            if (resumeFrom > 0) {
                val mb = resumeFrom / (1024 * 1024)
                sink.message(context.getString(R.string.download_resuming_src, source, fileName, mb.toInt()))
                Log.i(TAG, "Resuming $fileName from $resumeFrom bytes ($source: $url)")
            } else {
                sink.message(context.getString(R.string.download_fetching_file, source, fileName))
                Log.i(TAG, "Downloading $fileName from $source: $url")
            }

            val conn = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 10000
                readTimeout = 120000
                requestMethod = "GET"
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "MiniCPMV-demo/1.0")
                if (resumeFrom > 0) {
                    setRequestProperty("Range", "bytes=$resumeFrom-")
                }
            }

            try {
                val responseCode = conn.responseCode

                // 416: server thinks we already have everything. Verify size
                // matches Content-Range total before treating .tmp as complete.
                if (responseCode == 416 /* RANGE_NOT_SATISFIABLE */) {
                    val totalLen = parseContentRangeTotal(conn.getHeaderField("Content-Range"))
                    if (totalLen != null && tmpFile.length() == totalLen) {
                        Log.i(TAG, "$fileName: server reports complete (HTTP 416, size=$totalLen)")
                    } else {
                        // .tmp is somehow larger than server's file (different
                        // mirror? truncated download earlier?). Reset.
                        Log.w(TAG, "$fileName: HTTP 416 but local size ${tmpFile.length()} != server $totalLen; restarting")
                        tmpFile.delete()
                        throw RuntimeException(context.getString(R.string.download_416_mismatch, source))
                    }
                } else {
                    val acceptedResume = responseCode == HttpURLConnection.HTTP_PARTIAL // 206
                    val isOk = responseCode == HttpURLConnection.HTTP_OK // 200
                    if (!acceptedResume && !isOk) {
                        throw RuntimeException("$source returned $responseCode for $fileName")
                    }

                    // 200 means the server didn't honor Range - either we asked
                    // for bytes=0- (then 200/206 are equivalent), or the server
                    // doesn't support ranges. Either way, restart from 0.
                    if (isOk && resumeFrom > 0) {
                        Log.w(TAG, "$fileName: server returned 200 instead of 206; restarting from 0")
                        if (tmpFile.exists()) tmpFile.delete()
                    }

                    val effectiveStart = if (acceptedResume) resumeFrom else 0L
                    sink.resuming(effectiveStart)
                    val remaining = conn.contentLength.toLong() // -1 if unknown
                    val totalSize = if (remaining > 0) remaining + effectiveStart else -1L

                    conn.inputStream.use { input ->
                        // append=true only when we actually got 206 and are
                        // continuing the existing .tmp; otherwise truncate.
                        FileOutputStream(tmpFile, acceptedResume && resumeFrom > 0).use { output ->
                            val buffer = ByteArray(64 * 1024)
                            var totalRead = effectiveStart
                            var lastProgressTime = 0L

                            while (true) {
                                val read = input.read(buffer)
                                if (read == -1) break
                                output.write(buffer, 0, read)
                                totalRead += read

                                val now = System.currentTimeMillis()
                                if (now - lastProgressTime > 500) {
                                    lastProgressTime = now
                                    if (totalSize > 0) {
                                        Log.i(TAG, "$fileName: ${totalRead * 100 / totalSize}%")
                                    }
                                    sink.overall(totalRead)
                                }
                            }
                            output.flush()
                            // fd.sync() ensures bytes hit the storage device
                            // before we trust the .tmp size for the next
                            // resume attempt. It's cheap relative to a 1GB
                            // download but, on some Android devices/FS
                            // combinations, can throw SyncFailedException -
                            // wrap it so a fsync hiccup doesn't fail an
                            // otherwise-fine download.
                            try {
                                output.fd.sync()
                            } catch (e: Throwable) {
                                Log.w(TAG, "fsync failed for $fileName (continuing): ${e.message}")
                            }
                        }
                    }
                }

                // Promote .tmp -> final. Be defensive: another process, an OS
                // cleanup, or a 200-restart may have removed the .tmp between
                // the write and here. Never let a missing .tmp throw
                // NoSuchFileException - fall back to an already-complete target
                // if one exists, else fail with a clear message.
                if (targetFile.exists()) targetFile.delete()
                if (tmpFile.exists()) {
                    if (!tmpFile.renameTo(targetFile)) {
                        tmpFile.copyTo(targetFile, overwrite = true)
                        tmpFile.delete()
                    }
                } else if (!targetFile.exists()) {
                    // Neither .tmp nor final is present. If the server told us
                    // this was already complete (416) we cannot recover here;
                    // otherwise it's a genuine failure.
                    Log.e(TAG, "$fileName: .tmp vanished before finalize (and no final file)")
                    throw java.io.IOException("$fileName: temporary file disappeared before it could be finalized; will retry from scratch")
                } else {
                    Log.w(TAG, "$fileName: .tmp missing but final file already present; using it")
                }

                verifyOrThrow(context, targetFile, source, expectedMd5, expectedSha256, sink, tmpFile)

                sink.fileFinished(targetFile.length())
                sink.message(context.getString(R.string.download_file_complete, fileName, (targetFile.length() / (1024 * 1024)).toInt()))
                Log.i(TAG, "$fileName saved to ${targetFile.absolutePath}")
            } finally {
                conn.disconnect()
                // Note: we deliberately do NOT delete tmpFile in catch-all -
                // a transient network error / process kill should leave the
                // partial bytes on disk so the next call can resume.
            }
        }

        // Parse the "*/<total>" suffix out of a Content-Range header.
        // Returns null if header is missing or malformed.
        private fun parseContentRangeTotal(header: String?): Long? {
            if (header.isNullOrEmpty()) return null
            val slash = header.lastIndexOf('/')
            if (slash < 0 || slash == header.length - 1) return null
            return header.substring(slash + 1).trim().toLongOrNull()
        }

        // Streaming MD5 over a file. Buffer size kept small enough to avoid
        // memory pressure on cheaper devices while still being decently fast
        // (~150MB/s on the test phone for the 1GB mmproj).
        private fun computeDigest(file: File, algorithm: String): String {
            val md = MessageDigest.getInstance(algorithm)
            FileInputStream(file).use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    md.update(buffer, 0, read)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        private fun computeMd5(file: File): String = computeDigest(file, "MD5")

        // Whether [file] matches the first non-null expectation. Callers only
        // invoke this with at least one expectation set. Hash beats size:
        // SHA-256 > MD5 > exact byte length.
        private fun matchesExpectation(
            file: File,
            expectedMd5: String?,
            expectedSha256: String?,
            expectedBytes: Long?,
        ): Boolean = when {
            expectedSha256 != null -> computeDigest(file, "SHA-256").equals(expectedSha256, ignoreCase = true)
            expectedMd5 != null -> computeMd5(file).equals(expectedMd5, ignoreCase = true)
            expectedBytes != null -> file.length() == expectedBytes
            else -> true
        }

        // Verifies [file] against whichever hash is pinned (SHA-256 preferred,
        // MD5 fallback). Deletes the file and throws on mismatch. A no-op when
        // neither hash is set.
        private fun verifyOrThrow(
            context: Context,
            file: File,
            source: String,
            expectedMd5: String?,
            expectedSha256: String?,
            sink: ProgressSink,
            tmpFile: File? = null,
        ) {
            when {
                expectedSha256 != null -> {
                    sink.message(context.getString(R.string.download_verifying_sha256, source, file.name))
                    val actual = computeDigest(file, "SHA-256")
                    if (!actual.equals(expectedSha256, ignoreCase = true)) {
                        Log.e(TAG, "${file.name} SHA-256 mismatch: expected $expectedSha256, got $actual")
                        file.delete()
                        tmpFile?.let { if (it.exists()) it.delete() }
                        throw RuntimeException(
                            context.getString(R.string.download_sha256_failed, file.name, expectedSha256, actual)
                        )
                    }
                    Log.i(TAG, "${file.name} SHA-256 OK ($actual)")
                }
                expectedMd5 != null -> {
                    sink.message(context.getString(R.string.download_verifying_md5, source, file.name))
                    val actual = computeMd5(file)
                    if (!actual.equals(expectedMd5, ignoreCase = true)) {
                        Log.e(TAG, "${file.name} MD5 mismatch: expected $expectedMd5, got $actual")
                        file.delete()
                        tmpFile?.let { if (it.exists()) it.delete() }
                        throw RuntimeException(
                            context.getString(R.string.download_md5_failed, file.name, expectedMd5, actual)
                        )
                    }
                    Log.i(TAG, "${file.name} MD5 OK ($actual)")
                }
            }
        }

        // Downloads every file listed by a `sha256  <relative-path>` manifest
        // served from a ModelScope repo into the model directory, verifying
        // each entry. Used for multi-file bundles (see
        // [ModelInfo.bundleManifestFile]).
        private suspend fun downloadBundle(
            context: Context,
            dir: File,
            msBase: String,
            manifestName: String,
            sink: ProgressSink,
        ) {
            val manifestUrl = URL("$msBase/$manifestName")
            Log.i(TAG, "Fetching bundle manifest: $manifestUrl")
            val entries = fetchChecksumManifest(manifestUrl)
            if (entries.isEmpty()) throw RuntimeException("Empty bundle manifest: $manifestName")
            sink.message(context.getString(R.string.download_bundle_start, entries.size, manifestName))
            entries.forEachIndexed { index, (sha256, relativePath) ->
                val target = File(dir, relativePath)
                require(target.canonicalPath.startsWith(dir.canonicalPath + File.separator)) {
                    "Invalid bundle path: $relativePath"
                }
                target.parentFile?.let { if (!it.exists()) it.mkdirs() }
                sink.message(context.getString(R.string.download_bundle_progress, index + 1, entries.size, relativePath))
                downloadFileMultiSource(
                    context,
                    target.parentFile ?: dir,
                    target.name,
                    listOf(FileSource("ModelScope", URL("$msBase/$relativePath"))),
                    null,
                    sink,
                    sha256,
                )
            }
        }

        // Fetches and parses a `sha256  <path>` checksum manifest. Blank lines
        // are skipped; the hash field must be 64 hex chars so a path containing
        // spaces is preserved. Returns (lowercased-hash, path) pairs.
        private fun fetchChecksumManifest(url: URL): List<Pair<String, String>> {
            val conn = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 10_000
                readTimeout = 30_000
                requestMethod = "GET"
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "MiniCPMV-demo/1.0")
            }
            try {
                val code = conn.responseCode
                if (code != HttpURLConnection.HTTP_OK) {
                    throw RuntimeException("Checksum manifest HTTP $code: $url")
                }
                val entries = mutableListOf<Pair<String, String>>()
                conn.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { raw ->
                        val line = raw.trim()
                        if (line.isEmpty()) return@forEach
                        val parts = line.split(Regex("""\s+"""), limit = 2)
                        require(parts.size == 2 && parts[0].matches(Regex("[0-9a-fA-F]{64}"))) {
                            "Invalid checksum manifest entry"
                        }
                        val relativePath = parts[1].removePrefix("*")
                        require(relativePath.isNotBlank() && !File(relativePath).isAbsolute) {
                            "Invalid bundle path: $relativePath"
                        }
                        entries.add(parts[0].lowercase() to relativePath)
                    }
                }
                return entries
            } finally {
                conn.disconnect()
            }
        }
    }

    @Volatile
    private var _mmprojLoaded = false

    @Volatile
    private var _nativeCallTimedOut = false

    @ExperimentalCoroutinesApi
    private val llamaDispatcher = Dispatchers.IO.limitedParallelism(1)
    private val llamaScope = CoroutineScope(llamaDispatcher + SupervisorJob())

    private class NativeCallTimeoutException(
        message: String,
        cause: Throwable,
    ) : RuntimeException(message, cause)

    /**
     * Run a blocking native call with a hard wall-clock deadline.
     *
     * The Hexagon backend can hang forever inside sched_reserve()/fastrpc_mmap
     * when the DSP rejects the memory layout (e.g. on Hexagon v73 where a
     * single mapping exceeds the 32-bit window). Without a deadline the UI
     * stays on "loading model" with no way out. A stuck FastRPC ioctl cannot
     * be cancelled, so on timeout the worker thread is left behind and the app
     * is marked unusable for inference until restarted — but the user at least
     * gets an error instead of an endless spinner.
     */
    private fun <T> withNativeTimeout(
        what: String,
        timeoutSeconds: Long,
        block: () -> T,
    ): T {
        val worker = java.util.concurrent.Executors.newSingleThreadExecutor { r ->
            Thread(r, "native-$what").apply { isDaemon = true }
        }
        return try {
            val future = worker.submit<T> { block() }
            try {
                future.get(timeoutSeconds, java.util.concurrent.TimeUnit.SECONDS)
            } catch (e: java.util.concurrent.TimeoutException) {
                future.cancel(true)
                // The native call may still be running and cannot be interrupted.
                // Poison the engine so no later load/unload races with that worker.
                _nativeCallTimedOut = true
                throw NativeCallTimeoutException(
                    "$what timed out after ${timeoutSeconds}s. The Hexagon DSP likely " +
                    "rejected the memory layout (device may need a smaller memory budget " +
                    "or is unsupported). Please restart the app.", e)
            } catch (e: java.util.concurrent.ExecutionException) {
                throw (e.cause as? Exception) ?: e
            }
        } finally {
            worker.shutdown()
        }
    }

    private val _state = MutableStateFlow<LlamaState>(LlamaState.Uninitialized)
    val state: StateFlow<LlamaState> = _state.asStateFlow()

    /**
     * True when the model is loaded, readable without a round trip through the
     * StateFlow. loadModel() sets ModelReady on the llama dispatcher while its
     * caller is still on that dispatcher, so anything that checked
     * `state.value` right after loading raced the main-thread collector and
     * failed - it is why replaying the history after a reload silently did
     * nothing and the model lost the conversation (and the tool instructions).
     */
    val isReady: Boolean get() = _state.value is LlamaState.ModelReady

    @Volatile
    private var _cancelGeneration = false

    @Volatile
    private var _readyForSystemPrompt = false

    private external fun init(nativeLibDir: String)
    private external fun load(modelPath: String, expertsPath: String?, memoryBudgetMb: Int, contextSize: Int): Int
    private external fun setImageTokenLimitsNative(minTokens: Int, maxTokens: Int)
    // image_max_slice_nums: 1..9 (or -1 for model default).  See JNI for
    // the semantics; the upper layer should pass MIN_IMAGE_SLICE..MAX_IMAGE_SLICE.
    private external fun loadMmproj(mmprojPath: String, imageMaxSliceNums: Int): Int
    // Native counterpart of [setImageMaxSliceNums].  Renamed in JNI as
    // setImageMaxSliceNumsNative to avoid the name collision with the
    // public suspend wrapper above.
    private external fun setImageMaxSliceNumsNative(n: Int)
    // Pushes the MiniCPM-V family version (5 / 46 / 100045 / ...) down to
    // native so prepare()'s n_ctx selection and the assistant-turn prefix
    // logic see the right value.  Required since upstream master mtmd
    // dropped mtmd_get_minicpmv_version().  See LlamaEngine.loadModel.
    private external fun setMinicpmvVersionNative(version: Int)
    // MiniCPM5 / V-4.6 thinking toggle. Default off.
    private external fun setEnableThinkingNative(enable: Boolean)
    // 0 if no mmproj is loaded.  46 / 460 / 461 = MiniCPM-V-4.6 family.
    // Used by [isVideoUnderstandingSupported] to gate the video path.
    private external fun getMinicpmvVersionNative(): Int
    private external fun prepare(): Int
    private external fun systemInfo(): String
    // The active chat template (Jinja or built-in), or null when the model
    // ships none. Used to expose the generate_image tool in a shape the
    // model's own template understands (tools schema vs. system text).
    private external fun chatTemplateNative(): String?
    // OpenAI-compatible tool schema array for the next system prompt; pass
    // "[]" (or null) to clear. Only tool-aware templates render it.
    private external fun setToolSchemasNative(toolSchemasJson: String?)
    // Appends a finished text-to-image turn (assistant tool call + tool
    // result) to the template history used by tool-aware prompts.
    private external fun recordImageToolTurnNative(prompt: String, fileName: String)
    // Process the system prompt without dropping an existing short-term
    // state. Unlike the single-shot variant used at load time, this can be
    // called mid-conversation so a new system prompt never wipes history.
    private external fun processSystemPromptKeepStateNative(systemPrompt: String): Int
    private external fun processSystemPrompt(systemPrompt: String): Int
    private external fun processUserPrompt(userPrompt: String, predictLength: Int): Int
    private external fun generateNextToken(): String?
    private external fun prefillImage(imageData: ByteArray, imageSize: Int): Int
    private external fun restoreHistory(
        roles: Array<String>,
        texts: Array<String>,
        imageCounts: IntArray,
        images: Array<ByteArray>
    ): Int
    private external fun resetPerformanceStatsNative()
    private external fun readPerformanceStatsNative(): DoubleArray
    private external fun wasContextTruncatedNative(): Boolean
    private external fun fullReset()
    private external fun nativeCancelGeneration()
    private external fun unload()
    private external fun shutdown()

    init {
        llamaScope.launch {
            try {
                check(_state.value is LlamaState.Uninitialized) {
                    "Cannot load native library in ${_state.value.javaClass.simpleName}!"
                }
                _state.value = LlamaState.Initializing
                Log.i(TAG, "Loading native library...")
                Log.i(TAG, CpuFeatures.summary())

                val variant = CpuFeatures.bestGgmlCpuVariant()
                if (variant != null) {
                    try {
                        Log.i(TAG, "Pre-loading optimised ggml-cpu ($variant)")
                        System.loadLibrary("ggml-cpu-$variant")
                        Log.i(TAG, "Optimised ggml-cpu ($variant) loaded successfully")
                    } catch (e: UnsatisfiedLinkError) {
                        Log.w(TAG, "Optimised ggml-cpu-$variant not available, using baseline", e)
                    }
                }

                System.loadLibrary("mobiinfer")
                init(nativeLibDir)
                _state.value = LlamaState.Initialized
                Log.i(TAG, "Native library loaded! System info: \n${systemInfo()}")
            } catch (e: LinkageError) {
                Log.e(TAG, "Native library is incompatible with this device", e)
                _state.value = LlamaState.Error(RuntimeException(
                    "无法加载本地推理库，请检查安装包与设备兼容性。", e))
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load native library", e)
                _state.value = LlamaState.Error(e)
            }
        }
    }

    suspend fun loadModel(pathToModel: String, pathToMmproj: String? = null) =
        withContext(llamaDispatcher) {
            check(_state.value is LlamaState.Initialized) {
                "Cannot load model in ${_state.value.javaClass.simpleName}!"
            }
            check(!_nativeCallTimedOut) {
                "A previous native call timed out. Restart the app before loading a model."
            }
            try {
                Log.i(TAG, "Checking access to model file... \n$pathToModel")
                File(pathToModel).let {
                    require(it.exists()) { "File not found: $pathToModel" }
                    require(it.isFile) { "Not a valid file: $pathToModel" }
                    require(it.canRead()) { "Cannot read file: $pathToModel" }
                }

                Log.i(TAG, "Loading model... \n$pathToModel")
                _readyForSystemPrompt = false
                _state.value = LlamaState.LoadingModel
                val modelInfo = getSelectedModel(context)
                val missingExpert = missingExpertFile(context, modelInfo)
                require(missingExpert == null) { "Missing or unreadable expert file: $missingExpert" }
                // Experts live under the per-model root (not necessarily next to
                // the GGUF, which may be in a `models/` subdirectory), and only
                // vision models require an mmproj; text-only MoE bundles (e.g.
                // Qwen3.5) must not be forced to have one.
                val expertsPath = modelInfo.expertsDirectory?.let {
                    File(modelDirFor(context, modelInfo), it).absolutePath
                }
                if (modelInfo.mmprojFileName != null) {
                    require(pathToMmproj != null && File(pathToMmproj).canRead()) {
                        "${modelInfo.displayName} vision weights are missing or unreadable"
                    }
                }
                setImageTokenLimitsNative(modelInfo.imageMinTokens, modelInfo.imageMaxTokens)
                withNativeTimeout("load model", LOAD_TIMEOUT_SECONDS) {
                    load(pathToModel, expertsPath, getMemoryBudgetMb(context), getContextSize(context))
                }.let {
                    if (it != 0) throw RuntimeException("Failed to load model (code: $it)")
                }

                if (pathToMmproj != null) {
                    Log.i(TAG, "Checking access to mmproj file... \n$pathToMmproj")
                    File(pathToMmproj).let {
                        require(it.exists()) { "mmproj file not found: $pathToMmproj" }
                        require(it.isFile) { "Not a valid mmproj file: $pathToMmproj" }
                        require(it.canRead()) { "Cannot read mmproj file: $pathToMmproj" }
                    }

                    val sliceCap = getImageMaxSliceNums(context)
                    Log.i(TAG, "Loading mmproj (image_max_slice_nums=$sliceCap)... \n$pathToMmproj")
                    withNativeTimeout("load mmproj", LOAD_TIMEOUT_SECONDS) {
                        loadMmproj(pathToMmproj, sliceCap)
                    }.let {
                        if (it != 0) {
                            if (modelInfo.mmprojFileName != null) throw RuntimeException("Failed to load ${modelInfo.displayName} vision weights (code: $it)")
                            Log.w(TAG, "Failed to load mmproj (code: $it), continuing without vision support")
                        } else {
                            _mmprojLoaded = true
                            // Upstream master mtmd dropped mtmd_get_minicpmv_version, so the
                            // native side can no longer auto-detect which MiniCPM-V family
                            // an mmproj belongs to.  Push the version down from Kotlin based
                            // on the selected ModelInfo id; native uses this to (a) flip
                            // n_ctx to V46_CONTEXT_SIZE on prepare() and (b) pick the right
                            // assistant-turn prefix.  Mirrors iOS opt-r1, which routes the
                            // same hint through the MBMtmd bridge / MTMDParams.
                            val mv = inferMinicpmvVersion(getSelectedModel(context))
                            setMinicpmvVersionNative(mv)
                            Log.i(TAG, "mmproj loaded successfully! minicpmv_version=$mv")
                        }
                    }
                }

                withNativeTimeout("prepare resources", PREPARE_TIMEOUT_SECONDS) { prepare() }.let {
                    if (it != 0) throw RuntimeException("Failed to prepare resources (code: $it)")
                }
                Log.i(TAG, "Model loaded!")
                loadedModelId = modelInfo.id
                _readyForSystemPrompt = true
                _cancelGeneration = false
                setEnableThinkingNative(getEnableThinking(context))
                _state.value = LlamaState.ModelReady
            } catch (e: Exception) {
                Log.e(TAG, (e.message ?: "Error loading model") + "\n" + pathToModel, e)
                if (e !is NativeCallTimeoutException) {
                    unload()
                }
                loadedModelId = null
                _mmprojLoaded = false
                _state.value = LlamaState.Error(e)
                throw e
            }
        }

    val isVisionSupported: Boolean get() = _mmprojLoaded

    // Live update of the per-image slice cap.  Persists the new value to
    // SharedPreferences so the next mmproj reload picks it up too, and
    // (if mmproj is currently loaded) immediately patches the in-memory
    // clip context so the *next* image picked uses the new cap without
    // needing a model reload.  Cheap (no allocation, no warmup) - safe to
    // wire to a continuous slider.
    suspend fun setImageMaxSliceNums(n: Int) = withContext(llamaDispatcher) {
        val clamped = n.coerceIn(MIN_IMAGE_SLICE, MAX_IMAGE_SLICE)
        setImageMaxSliceNumsPref(context, clamped)
        if (_mmprojLoaded) {
            Log.i(TAG, "Live-updating image_max_slice_nums = $clamped")
            setImageMaxSliceNumsNative(clamped)
        } else {
            Log.i(TAG, "image_max_slice_nums = $clamped persisted; will apply on next mmproj load")
        }
    }

    fun setEnableThinking(enable: Boolean) {
        setEnableThinkingPref(context, enable)
        setEnableThinkingNative(enable)
        Log.i(TAG, "enable_thinking=$enable")
    }

    /**
     * Declares the tools the model may call. Has to run right before the
     * system prompt is processed: the prompt it applies to is the only one
     * that sees them.
     */
    suspend fun setToolSchemas(schemasJson: String?) = withContext(llamaDispatcher) {
        runCatching { setToolSchemasNative(schemasJson) }
            .onFailure { Log.w(TAG, "setToolSchemas failed", it) }
    }

    /** Records a generated picture as a generate_image tool turn. */
    suspend fun recordImageToolTurn(prompt: String, fileName: String) = withContext(llamaDispatcher) {
        runCatching { recordImageToolTurnNative(prompt, fileName) }
            .onFailure { Log.w(TAG, "recordImageToolTurn failed", it) }
    }

    /** The model's chat template source, or null when it has none. */
    suspend fun chatTemplate(): String? = withContext(llamaDispatcher) {
        runCatching { chatTemplateNative() }.getOrNull()?.takeIf { it.isNotBlank() }
    }

    /**
     * Applies a system prompt mid-conversation without resetting the context.
     * The KV prefix is recomputed from the system prompt on the next request;
     * the restored history is replayed afterwards by the caller.
     */
    suspend fun reapplySystemPrompt(prompt: String) = withContext(llamaDispatcher) {
        require(prompt.isNotBlank()) { "Cannot process empty system prompt!" }
        val state = _state.value
        if (state is LlamaState.ModelReady) {
            _state.value = LlamaState.ProcessingSystemPrompt
        }
        try {
            processSystemPromptKeepStateNative(prompt).let { result ->
                if (result != 0) throw RuntimeException("Failed to apply system prompt: $result")
            }
        } finally {
            if (state is LlamaState.ModelReady) _state.value = LlamaState.ModelReady
        }
        Log.i(TAG, "System prompt re-applied mid-conversation")
    }

    suspend fun setSystemPrompt(prompt: String) =
        withContext(llamaDispatcher) {
            require(prompt.isNotBlank()) { "Cannot process empty system prompt!" }
            check(_readyForSystemPrompt) { "System prompt must be set RIGHT AFTER model loaded!" }
            check(_state.value is LlamaState.ModelReady) {
                "Cannot process system prompt in ${_state.value.javaClass.simpleName}!"
            }

            Log.i(TAG, "Sending system prompt...")
            _readyForSystemPrompt = false
            _state.value = LlamaState.ProcessingSystemPrompt
            processSystemPrompt(prompt).let { result ->
                if (result != 0) {
                    RuntimeException("Failed to process system prompt: $result").also {
                        _state.value = LlamaState.Error(it)
                        throw it
                    }
                }
            }
            Log.i(TAG, "System prompt processed! Awaiting user prompt...")
            _state.value = LlamaState.ModelReady
        }

    suspend fun prefillImage(imageData: ByteArray) =
        withContext(llamaDispatcher) {
            check(_mmprojLoaded) { "Vision model not loaded!" }
            check(_state.value is LlamaState.ModelReady) {
                "Cannot prefill image in ${_state.value.javaClass.simpleName}!"
            }

            Log.i(TAG, "Prefilling image...")
            _state.value = LlamaState.PrefillingImage
            val result = prefillImage(imageData, imageData.size)
            if (result != 0) {
                _state.value = LlamaState.ModelReady
                throw RuntimeException("Failed to prefill image (code: $result)")
            }
            Log.i(TAG, "Image prefilled!")
            _state.value = LlamaState.ModelReady
        }

    suspend fun resetPerformanceStats() = withContext(llamaDispatcher) {
        resetPerformanceStatsNative()
    }

    suspend fun readVisionEncodingMs(): Long? = withContext(llamaDispatcher) {
        val values = readPerformanceStatsNative()
        values.getOrNull(4)?.takeIf { it.isFinite() && it >= 0.0 }?.toLong()
    }

    suspend fun readPerformanceRates(): Pair<Double?, Double?> = withContext(llamaDispatcher) {
        val values = readPerformanceStatsNative()
        if (values.size < 4) return@withContext null to null
        val prefillMs = values[0]
        val prefillTokens = values[1]
        val decodeMs = values[2]
        val decodeTokens = values[3]
        val prefillRate = if (prefillMs > 0.0 && prefillTokens > 0.0) prefillTokens * 1000.0 / prefillMs else null
        val decodeRate = if (decodeMs > 0.0 && decodeTokens > 0.0) decodeTokens * 1000.0 / decodeMs else null
        prefillRate to decodeRate
    }

    /** True when the last completed request hit the context limit. */
    suspend fun wasLastGenerationTruncated(): Boolean = withContext(llamaDispatcher) {
        wasContextTruncatedNative()
    }

    /**
     * True iff the currently loaded mmproj advertises a MiniCPM-V-4.6
     * family clip (46 / 460 / 461).  Earlier models share the same image
     * code path but the iOS demo also restricts video understanding to
     * V-4.6 because the perceiver token count + extended `n_ctx=8192`
     * combine to make multi-frame prefill cheap enough to be usable.
     * The Android backend mirrors that gate by reading
     * mtmd_get_minicpmv_version() back through getMinicpmvVersionNative.
     */
    val isVideoUnderstandingSupported: Boolean
        get() {
            if (!_mmprojLoaded) return false
            val v = try { getMinicpmvVersionNative() } catch (_: Throwable) { 0 }
            return v == 46 || v == 460 || v == 461
        }

    /**
     * Prefills a sequence of video frames into the model.  Iterates over
     * [frames] and feeds each one through the same native path as
     * [prefillImage] (which is exactly what iOS does via repeated
     * `mtmd_ios_prefill_frame`: the C++ implementation of `prefill_frame`
     * and `prefill_image` in `mtmd-ios.cpp` are byte-for-byte identical
     * apart from the error-log prefix).
     *
     * Side effects to match iOS [MBHomeViewController+CaptureVideo]:
     *  - Slice cap is temporarily forced to 1 for the duration of the
     *    video so the per-frame ViT cost stays manageable; the user's
     *    chat-page slider value is restored on exit (also on failure).
     *    The persisted SharedPreference is NOT touched - the slider
     *    keeps showing the user's chosen value.
     *  - State is parked in [LlamaState.PrefillingImage] across all
     *    frames so the UI shows a single "prefilling" badge rather
     *    than flickering between frames.
     *
     * @param onProgress invoked on this engine's serial dispatcher
     * after each successfully-prefilled frame.  Caller should
     * `withContext(Dispatchers.Main)` before touching UI.
     */
    suspend fun prefillVideoFrames(
        frames: List<ByteArray>,
        onProgress: suspend (current: Int, total: Int) -> Unit = { _, _ -> }
    ) = withContext(llamaDispatcher) {
        check(_mmprojLoaded) { "Vision model not loaded!" }
        check(isVideoUnderstandingSupported) {
            "Video understanding only supported on MiniCPM-V-4.6 (current version=" +
                "${getMinicpmvVersionNative()})"
        }
        check(_state.value is LlamaState.ModelReady) {
            "Cannot prefill video in ${_state.value.javaClass.simpleName}!"
        }
        require(frames.isNotEmpty()) { "No frames to prefill" }

        val savedSliceCap = getImageMaxSliceNums(context)
        val needSliceOverride = savedSliceCap > 1

        Log.i(TAG, "Prefilling ${frames.size} video frames (savedSlice=$savedSliceCap, override=$needSliceOverride)")
        _state.value = LlamaState.PrefillingImage
        try {
            if (needSliceOverride) {
                Log.i(TAG, "Temporarily forcing image_max_slice_nums=1 for video frames")
                setImageMaxSliceNumsNative(1)
            }
            for ((idx, frame) in frames.withIndex()) {
                val rc = prefillImage(frame, frame.size)
                if (rc != 0) {
                    throw RuntimeException(
                        "Failed to prefill video frame ${idx + 1}/${frames.size} (code: $rc)"
                    )
                }
                onProgress(idx + 1, frames.size)
            }
            Log.i(TAG, "Video frames prefilled successfully")
        } finally {
            if (needSliceOverride) {
                Log.i(TAG, "Restoring image_max_slice_nums=$savedSliceCap after video")
                setImageMaxSliceNumsNative(savedSliceCap)
            }
            _state.value = LlamaState.ModelReady
        }
    }

    suspend fun clearContext() =
        withContext(llamaDispatcher) {
            check(isReady) {
                "Cannot clear context in ${_state.value.javaClass.simpleName}"
            }
            try {
                fullReset()
            } catch (e: Exception) {
                // Native reset failed (e.g. context re-allocation failed). Mark
                // the engine as Error so no prompt is sent into a half-freed
                // native context.
                Log.e(TAG, "fullReset failed", e)
                _state.value = LlamaState.Error(e)
                throw e
            }
            _readyForSystemPrompt = true
            Log.i(TAG, "Context fully reset - context recreated, ready for new conversation")
        }

    /**
     * Re-seeds the engine with a previously saved conversation so the user can
     * keep chatting with context intact. Images are read back from their
     * persisted content-URI references; a URI that can no longer be opened is
     * skipped (that image turn is then replayed without the picture, which is
     * the graceful degradation we agreed on for reference-only storage).
     */
    suspend fun restoreConversation(turns: List<ChatHistoryTurn>) = withContext(llamaDispatcher) {
        check(isReady) {
            "Cannot restore conversation in ${_state.value.javaClass.simpleName}"
        }
        val roles = ArrayList<String>(turns.size)
        val texts = ArrayList<String>(turns.size)
        val counts = IntArray(turns.size)
        val images = ArrayList<ByteArray>()
        turns.forEachIndexed { index, turn ->
            roles.add(turn.role)
            texts.add(turn.text)
            var attached = 0
            for (uriString in turn.imageUris) {
                val bytes = readUriBytes(uriString)
                if (bytes != null) {
                    images.add(bytes)
                    attached++
                } else {
                    Log.w(TAG, "restoreConversation: image URI no longer readable, skipping: $uriString")
                }
            }
            counts[index] = attached
        }
        val result = restoreHistory(
            roles.toTypedArray(),
            texts.toTypedArray(),
            counts,
            images.toTypedArray()
        )
        if (result != 0) {
            throw IllegalStateException("Failed to restore conversation (code: $result)")
        }
        Log.i(TAG, "Restored conversation: ${turns.size} turns, ${images.size} images")
    }

    private fun readUriBytes(uriString: String): ByteArray? = try {
        context.contentResolver.openInputStream(android.net.Uri.parse(uriString))?.use { it.readBytes() }
    } catch (e: Exception) {
        Log.w(TAG, "Failed to read image URI: $uriString", e)
        null
    }

    fun sendUserPrompt(
        message: String,
        predictLength: Int = DEFAULT_PREDICT_LENGTH
    ): Flow<String> = flow {
        require(message.isNotEmpty()) { "User prompt must not be empty!" }
        check(_state.value is LlamaState.ModelReady) {
            "User prompt discarded due to: ${_state.value.javaClass.simpleName}"
        }

        try {
            _cancelGeneration = false
            Log.i(TAG, "Sending user prompt...")
            _readyForSystemPrompt = false
            _state.value = LlamaState.ProcessingUserPrompt

            processUserPrompt(message, predictLength).let { result ->
                if (result != 0) {
                    Log.e(TAG, "Failed to process user prompt: $result")
                    throw IllegalStateException("Failed to process user prompt (code: $result)")
                }
            }

            Log.i(TAG, "User prompt processed. Generating assistant prompt...")
            _state.value = LlamaState.Generating
            while (!_cancelGeneration) {
                generateNextToken()?.let { utf8token ->
                    if (utf8token.isNotEmpty()) emit(utf8token)
                } ?: break
            }
            if (_cancelGeneration) {
                Log.i(TAG, "Assistant generation aborted per requested.")
                throw CancellationException("Generation cancelled")
            } else {
                Log.i(TAG, "Assistant generation complete. Awaiting user prompt...")
            }
            _state.value = LlamaState.ModelReady
        } catch (e: CancellationException) {
            Log.i(TAG, "Assistant generation's flow collection cancelled.")
            nativeCancelGeneration()
            _state.value = LlamaState.ModelReady
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error during generation!", e)
            runCatching { nativeCancelGeneration() }
            _state.value = LlamaState.Error(e)
            throw e
        }
    }.flowOn(llamaDispatcher)

    fun cancelGeneration() {
        _cancelGeneration = true
    }

    /** True while a model is resident in memory (loading/generating excluded). */
    val isModelLoaded: Boolean
        get() = _state.value is LlamaState.ModelReady

    /**
     * Frees the model. Runs on a dispatcher that is NOT tied to the caller's
     * job on purpose: it is routinely called from inside a generation coroutine
     * that is being cancelled (generate_image frees the chat model before the
     * image stack loads), and withContext(llamaDispatcher) would abort with
     * that job's CancellationException before the unload even starts.
     */
    suspend fun unloadModel() {
        // withContext(NonCancellable) keeps the unload running when the caller
        // happens to be a coroutine that is being cancelled; the explicit
        // dispatcher keeps it serialised with every other native call.
        withContext(NonCancellable + llamaDispatcher) { unloadModelInner() }
    }

    private suspend fun unloadModelInner() {
        if (_nativeCallTimedOut) {
            Log.w(TAG, "Skipping unload after native timeout; restart the app")
            return
        }
        if (_state.value is LlamaState.ModelReady || _state.value is LlamaState.Error) {
            Log.i(TAG, "Unloading model...")
            _readyForSystemPrompt = false
            _mmprojLoaded = false
            _state.value = LlamaState.UnloadingModel
            unload()
            loadedModelId = null
            _state.value = LlamaState.Initialized
            Log.i(TAG, "Model unloaded")
        }
    }

    fun resetToInitialized() {
        loadedModelId = null
        _mmprojLoaded = false
        _readyForSystemPrompt = false
        _cancelGeneration = false
        _state.value = if (_nativeCallTimedOut) {
            LlamaState.Error(RuntimeException(
                "A native call timed out. Restart the app before loading a model."))
        } else {
            LlamaState.Initialized
        }
    }

    fun cleanUp() {
        _cancelGeneration = true
        if (_nativeCallTimedOut) {
            Log.w(TAG, "Skipping native cleanup after timeout; restart the app")
            return
        }
        runBlocking(llamaDispatcher) {
            when (val state = _state.value) {
                is LlamaState.ModelReady -> {
                    Log.i(TAG, "Unloading model and free resources...")
                    _readyForSystemPrompt = false
                    _mmprojLoaded = false
                    _state.value = LlamaState.UnloadingModel
                    unload()
                    loadedModelId = null
                    _state.value = LlamaState.Initialized
                    Log.i(TAG, "Model unloaded!")
                }
                is LlamaState.Error -> {
                    Log.i(TAG, "Resetting error states...")
                    unload()
                    loadedModelId = null
                    _mmprojLoaded = false
                    _state.value = LlamaState.Initialized
                }
                else -> throw IllegalStateException("Cannot unload model in ${state.javaClass.simpleName}")
            }
        }
    }

    fun destroy() {
        _cancelGeneration = true
        if (_nativeCallTimedOut) {
            Log.w(TAG, "Skipping native destroy after timeout; restart the app")
            llamaScope.cancel()
            return
        }
        runBlocking(llamaDispatcher) {
            _readyForSystemPrompt = false
            _mmprojLoaded = false
            when (_state.value) {
                is LlamaState.Uninitialized -> {}
                is LlamaState.Initialized -> shutdown()
                else -> { unload(); shutdown() }
            }
        }
        loadedModelId = null
        llamaScope.cancel()
        _state.value = LlamaState.Uninitialized
        synchronized(Companion) {
            if (instance === this) instance = null
        }
    }

}
