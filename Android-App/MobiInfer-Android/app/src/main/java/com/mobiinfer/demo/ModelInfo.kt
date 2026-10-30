package com.mobiinfer.demo

import android.content.Context

/**
 * Metadata for a downloadable MiniCPM-V variant.
 *
 * Source-of-truth for download URLs: HuggingFace ([hfRepo]) and ModelScope
 * ([msRepo]) are both registered for every model, and `LlamaEngine`'s
 * downloader runs them in a **race**: both GETs are launched in parallel and
 * the first one to deliver HTTP headers wins; the loser's connection is
 * closed immediately. Mirrors iOS opt-r1's `MBModelDownloadHelperV2.downloadV2`
 * so all three demos consume the same upstream artifacts via the same
 * topology (no OBS direct link).
 *
 * The previously used [directGgufUrl] / [directMmprojUrl] knobs are retained
 * as an optional escape hatch — if a model ever needs a temporary mirror
 * during a TestFlight period they can be set and the racer treats them as a
 * third source — but for the current release lineup they stay null.
 *
 * [ggufRemoteName] and [mmprojRemoteName] decouple the on-disk filename from
 * the path segment on HF/MS. MiniCPM-V-4.6 stores mmproj as
 * `mmproj-model-merger-f16.gguf` locally (to avoid colliding with stale
 * pre-release files left in the per-model directory) but the upstream HF/MS
 * object is `mmproj-model-f16.gguf`.
 *
 * When [ggufMd5] / [mmprojMd5] are provided, the downloader verifies the
 * downloaded file's MD5 and deletes + raises on mismatch. Mirrors iOS demo
 * `MBV4ModelDownloadManager` / `MBV46ModelDownloadManager`.
 */
data class ModelInfo(
    val id: String,
    val displayName: String,
    val descriptionResName: String,
    val ggufFileName: String,
    val mmprojFileName: String? = null,
    val acousticFileName: String? = null,
    val hfRepo: String? = null,
    val msRepo: String? = null,
    val hfBranch: String = "main",
    val msBranch: String = "master",
    val ggufRemoteName: String? = null,
    val mmprojRemoteName: String? = null,
    val acousticRemoteName: String? = null,
    val directGgufUrl: String? = null,
    val directMmprojUrl: String? = null,
    val directAcousticUrl: String? = null,
    val ggufMd5: String? = null,
    val mmprojMd5: String? = null,
    val acousticMd5: String? = null,
    /**
     * Exact byte lengths of the individual files. Used as a fallback
     * integrity/skip check when no hash is pinned for a file (e.g. Qwen3-VL
     * relies on ModelScope git-LFS instead of a client-side MD5): on a partial
     * re-download an already-complete file is skipped instead of re-fetched.
     */
    val ggufBytes: Long? = null,
    val mmprojBytes: Long? = null,
    val acousticBytes: Long? = null,
    /**
     * Optional dynamic-resolution limits for the vision encoder. Qwen-VL
     * family mmproj models need at least 1024 image tokens for reliable
     * reading/grounding; -1 keeps the mmproj metadata defaults.
     */
    val imageMinTokens: Int = -1,
    val imageMaxTokens: Int = -1,
    /** Initial image scale before the user overrides it in Model Management. */
    val defaultImageScaleFactor: Float = 0.5f,
    val expertsDirectory: String? = null,
    /** Number of tiled expert layer files (`layer_NNN.bin`) in [expertsDirectory]. */
    val expertsLayerCount: Int = 40,
    /**
     * On-disk name of the DSpark draft GGUF inside a bundle, relative to the
     * per-model directory. Bundle payloads are enumerated by [bundleManifestFile],
     * so this is only used to include the draft in the "is the model complete"
     * check ([LlamaEngine.modelsExist]) - without it a missing draft would be
     * silently treated as a complete download.
     */
    val draftFileName: String? = null,
    /**
     * Total size in bytes of every file this model downloads. Used only to
     * render the single overall progress bar: the downloader accumulates
     * bytes across all files and divides by this constant. Hardcoded from the
     * upstream artefacts (verified against the on-device copies) rather than
     * probed at runtime, so the bar is determinate from the first byte.
     * 0 means "unknown" -> the UI keeps the bar indeterminate.
     */
    val totalBytes: Long = 0L,
    /**
     * Name of a `sha256  <relative-path>` manifest inside [msRepo]. When set,
     * the ModelScope downloader treats the model as a multi-file *bundle*:
     * it fetches this manifest and then downloads + SHA-256-verifies every
     * listed entry into the per-model directory at the same relative path.
     * This avoids hard-coding a large, volatile file list (e.g. the 4.7
     * Hexagon bundle's 40 tiled expert layers) in the client.
     */
    val bundleManifestFile: String? = null,
    /**
     * Additional downloadable files (relative to [msRepo]) beyond
     * [ggufFileName], fetched into the per-model directory when no bundle
     * manifest is used. Used by text-to-image models (MOSS-Image) which need
     * a text encoder, VAE and tokenizer alongside the diffusion checkpoint.
     */
    val extraFiles: List<String> = emptyList(),
    /**
     * Exact byte lengths for [extraFiles] (keyed by file name). Pinning the
     * size lets the downloader skip files that are already complete instead of
     * re-fetching them on every run.
     */
    val extraFileBytes: Map<String, Long> = emptyMap()
) {
    /** True for text-only models that have no vision projector. */
    val isTextOnly: Boolean
        get() = mmprojFileName == null && acousticFileName == null

    /** True for TTS (text-to-speech) models with acoustic component. */
    val isTts: Boolean
        get() = acousticFileName != null

    /**
     * True for text-to-image models (stable-diffusion.cpp checkpoints).
     * Detected by id so registering an image model only needs a matching id
     * plus its single-file checkpoint in [ggufFileName].
     */
    val isImage: Boolean
        get() = id.startsWith("sd-") || id.contains("stable-diffusion")

    /** Only the MiniCPM-V 4.7 bundle has the multi-frame video path. */
    val supportsVideo: Boolean
        get() = id == "minicpm-v-4_7"

    /**
     * Models whose bundled chat template actually consumes `enable_thinking`.
     * Qwen3-VL-4B-Instruct's template does not expose this switch.
     */
    val supportsThinking: Boolean
        get() = id == "qwen35-35b-a3b" ||
            id == "sage-a1.5" ||
            id == "qwen3-30b-a3b" ||
            id == "minicpm-v-4_7"

    fun getDescription(context: Context): String {
        val resId = context.resources.getIdentifier(descriptionResName, "string", context.packageName)
        return if (resId != 0) context.getString(resId) else descriptionResName
    }

    /** Path segment to request on HF/MS for the gguf, falling back to local name. */
    val ggufRemotePath: String
        get() = ggufRemoteName ?: ggufFileName

    /** Path segment to request on HF/MS for the mmproj, falling back to local name. */
    val mmprojRemotePath: String?
        get() = mmprojFileName?.let { mmprojRemoteName ?: it }

    /** Path segment to request on HF/MS for the acoustic GGUF, falling back to local name. */
    val acousticRemotePath: String?
        get() = acousticFileName?.let { acousticRemoteName ?: it }

    /** Whether the model registers a direct (non-HF/MS) mirror URL for all required files. */
    val hasDirectUrls: Boolean
        get() = if (isTextOnly) !directGgufUrl.isNullOrBlank()
                else !directGgufUrl.isNullOrBlank() && !directMmprojUrl.isNullOrBlank()

    /** Whether the model registers at least one HF / MS repo for racing. */
    val hasRemoteSources: Boolean
        get() = !hfRepo.isNullOrBlank() || !msRepo.isNullOrBlank()

    companion object {
        // This NPU branch ships the Qwen3-VL-4B demo model plus the Hexagon
        // MoE bundles (MiniCPM-V 4.7, Qwen3.5-35B-A3B, Sage-A1.5, Qwen3-30B-A3B).
        // The upstream demo's other variants (V-4, V-4.6, MiniCPM5-1B/2B,
        // VoxCPM2) are intentionally not offered here.
        val AVAILABLE_MODELS = listOf(
            // MiniCPM-V 4.7 is the default and appears first in every selector.
            ModelInfo(
                id = "minicpm-v-4_7",
                displayName = "MiniCPM-V-4.7 35B-A3B (Q3, NPU)",
                descriptionResName = "model_desc_v47_npu",
                ggufFileName = "minicpm47-les-hexagon.gguf",
                mmprojFileName = "mmproj-MiniCPM-V-4_7-real-f16-pad4320.gguf",
                expertsDirectory = "experts-htp-tiled",
                msRepo = "fengerhu1/MiniCPM-V-4.7-35B-A3B-gguf-hexagon",
                bundleManifestFile = "SHA256SUMS",
                draftFileName = "dspark-MiniCPM-V-4_7-q8_0.gguf",
                totalBytes = 19_315_173_512L
            ),
            // Sage-A1.5 35B-A3B (qwen35moe) Hexagon bundle: vision + text
            // agentic MoE with a 41st MTP expert layer, no speculative draft.
            // Same hexagon-tiled-v1 expert layout as the Qwen3.5 bundle.
            ModelInfo(
                id = "sage-a1.5",
                displayName = "Sage-A1.5 35B-A3B (Q4_K_M, NPU)",
                descriptionResName = "model_desc_sage_a1_5",
                ggufFileName = "models/sage-a1.5-les-hexagon.gguf",
                mmprojFileName = "models/Sage-A1.5-mmproj.gguf",
                expertsDirectory = "experts-htp-tiled",
                expertsLayerCount = 41,
                msRepo = "fengerhu1/Sage-A1.5-gguf-hexagon",
                bundleManifestFile = "SHA256SUMS",
                imageMinTokens = 1024,
                imageMaxTokens = 1024,
                defaultImageScaleFactor = 1.0f,
                totalBytes = 24_632_376_251L
            ),
            // Qwen3.5-35B-A3B (qwen35moe) Hexagon bundle: text-only, same
            // tiled-expert layout as the MiniCPM 4.7 bundle.  Main GGUF and the
            // DFlash draft live under `models/`, experts under
            // `experts-htp-tiled/`; all enumerated by the SHA256SUMS manifest.
            ModelInfo(
                id = "qwen35-35b-a3b",
                displayName = "Qwen3.5-35B-A3B (Q4_K_M, NPU)",
                descriptionResName = "model_desc_qwen35_35b",
                ggufFileName = "models/qwen35-les-hexagon.gguf",
                mmprojFileName = "models/mmproj-Qwen3.5-35B-A3B-Q8_0.gguf",
                expertsDirectory = "experts-htp-tiled",
                draftFileName = "models/Qwen3.5-35B-A3B-DFlash-Q8_0.gguf",
                msRepo = "fengerhu1/Qwen3.5-35B-A3B-gguf-hexagon",
                bundleManifestFile = "SHA256SUMS",
                imageMinTokens = 1024,
                imageMaxTokens = 1024,
                defaultImageScaleFactor = 1.0f,
                totalBytes = 25_064_701_899L
            ),
            // Qwen3-30B-A3B (qwen3moe) Hexagon bundle: text-only MoE, no draft and
            // no mmproj. Same hexagon-tiled-v1 expert layout (48 layers).
            ModelInfo(
                id = "qwen3-30b-a3b",
                displayName = "Qwen3-30B-A3B (Q4_K_M, NPU)",
                descriptionResName = "model_desc_qwen3_30b",
                ggufFileName = "models/qwen3-30b-a3b-les-hexagon.gguf",
                expertsDirectory = "experts-htp-tiled",
                expertsLayerCount = 48,
                msRepo = "fengerhu1/Qwen3-30B-A3B-gguf-hexagon",
                bundleManifestFile = "SHA256SUMS",
                totalBytes = 20_331_058_440L
            ),
            // Qwen3-VL-4B-Instruct: vision-language model served from
            // ModelScope only (Qwen/Qwen3-VL-4B-Instruct-GGUF). No HuggingFace
            // mirror is registered, so the downloader falls back to a single
            // non-raced source. No MD5 on purpose: ModelScope serves via
            // git-LFS which already provides hash-based integrity checks.
            ModelInfo(
                id = "qwen3-vl-4b-instruct",
                displayName = "Qwen3-VL-4B (Q4_K_M)",
                descriptionResName = "model_desc_qwen3vl_4b",
                ggufFileName = "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
                mmprojFileName = "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
                msRepo = "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                ggufBytes = 2_497_281_664L,
                mmprojBytes = 836_180_256L,
                totalBytes = 3_333_461_920L
            ),
            // MOSS-Image text-to-image, fp16 8-step (best quality). Runs on the
            // Hexagon NPU via the sdcpp-hexagon fork (libimagegen). ggufFileName
            // is the DiT; extraFiles carry the text encoder, VAE and tokenizer.
            ModelInfo(
                id = "sd-moss-image-fp16",
                displayName = "MOSS-Image FP16 (8-step)",
                descriptionResName = "model_desc_moss_fp16",
                ggufFileName = "moss-sft-official-f16.gguf",
                msRepo = "YYDH21/moss-image-hexagon",
                extraFiles = listOf(
                    "qwen3.5-9b-q4km.gguf",
                    "flux2-vae.safetensors",
                    "tokenizer.json"
                ),
                ggufBytes = 10_164_640_640L,
                extraFileBytes = mapOf(
                    "qwen3.5-9b-q4km.gguf" to 5_680_522_464L,
                    "flux2-vae.safetensors" to 336_211_292L,
                    "tokenizer.json" to 19_989_325L
                ),
                totalBytes = 16_201_363_721L
            ),
            // MOSS-Image q4 8-step (smaller, faster). Same TE/VAE/tokenizer
            // as the fp16 variant; only the DiT is quantized.
            ModelInfo(
                id = "sd-moss-image-q4",
                displayName = "MOSS-Image Q4 (8-step)",
                descriptionResName = "model_desc_moss_q4",
                ggufFileName = "moss-turbo8-q4km-f16t.gguf",
                msRepo = "YYDH21/moss-image-hexagon",
                extraFiles = listOf(
                    "qwen3.5-9b-q4km.gguf",
                    "flux2-vae.safetensors",
                    "tokenizer.json"
                ),
                ggufBytes = 3_600_669_568L,
                extraFileBytes = mapOf(
                    "qwen3.5-9b-q4km.gguf" to 5_680_522_464L,
                    "flux2-vae.safetensors" to 336_211_292L,
                    "tokenizer.json" to 19_989_325L
                ),
                totalBytes = 9_637_392_649L
            )
        )

        // The default selection is the MiniCPM-V 4.7 NPU bundle. Falls back to
        // the first entry only if that model is ever removed from the lineup.
        val DEFAULT_MODEL = AVAILABLE_MODELS.firstOrNull { it.id == "minicpm-v-4_7" }
            ?: AVAILABLE_MODELS.first()
    }
}
