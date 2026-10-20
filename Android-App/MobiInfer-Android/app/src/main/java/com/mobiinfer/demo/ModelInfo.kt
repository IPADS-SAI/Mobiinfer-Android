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
    val bundleManifestFile: String? = null
) {
    /** True for text-only models that have no vision projector. */
    val isTextOnly: Boolean
        get() = mmprojFileName == null && acousticFileName == null

    /** True for TTS (text-to-speech) models with acoustic component. */
    val isTts: Boolean
        get() = acousticFileName != null

    /** Only the MiniCPM-V 4.7 bundle has the multi-frame video path. */
    val supportsVideo: Boolean
        get() = id == "minicpm-v-4_7"

    /** MiniCPM5 (text), MiniCPM-V-4.6 and Qwen3.5 expose a thinking toggle. */
    val supportsThinking: Boolean
        get() = isTextOnly || id == "minicpm-v-4_6-instruct" || id == "qwen35-35b-a3b"

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
        // This NPU branch ships only two entries: the Qwen3-VL-4B demo model
        // and the MiniCPM-V 4.7 Hexagon bundle. The upstream demo's other
        // variants (V-4, V-4.6, MiniCPM5-1B/2B, VoxCPM2) are intentionally
        // not offered here.
        val AVAILABLE_MODELS = listOf(
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
            // MiniCPM-V 4.7 35B-A3B Hexagon bundle (NPU). Not a single GGUF:
            // it ships a mixed-Q3 main model, a padded F16 mmproj, a Q8_0
            // DSpark draft and 40 tiled expert layers. Served from ModelScope;
            // the per-file list (and SHA-256s) come from the repo's
            // SHA256SUMS manifest so the 40 expert files stay out of the
            // client source. The repo is laid out so each entry's relative
            // path already matches the on-device per-model layout.
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
            )
        )

        // The default selection is the MiniCPM-V 4.7 NPU bundle. Falls back to
        // the first entry only if that model is ever removed from the lineup.
        val DEFAULT_MODEL = AVAILABLE_MODELS.firstOrNull { it.id == "minicpm-v-4_7" }
            ?: AVAILABLE_MODELS.first()
    }
}
