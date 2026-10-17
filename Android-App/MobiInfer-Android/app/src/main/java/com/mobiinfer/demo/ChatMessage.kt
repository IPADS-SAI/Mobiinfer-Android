package com.mobiinfer.demo

import android.graphics.Bitmap

sealed class ChatMessage {
    abstract val id: Long

    data class UserMessage(
        override val id: Long,
        val text: String,
        val imageBitmap: Bitmap? = null,
        val imageInfo: String? = null,
        val isPrefilling: Boolean = false,
        // True when [imageBitmap] is a video's first frame and the
        // cell should overlay a play icon to communicate "this was a
        // video, the model saw N sampled frames".  Mirrors iOS
        // MBImageTableViewCell's video-playback overlay.
        val isVideo: Boolean = false,
        // Persisted reference to the picked media (a SAF content:// URI).
        // Stored instead of the bitmap so a saved conversation can re-open the
        // original file for display and for context replay. Null for synthetic
        // messages (e.g. restored placeholders) that carry no media.
        val imageUri: String? = null
    ) : ChatMessage()

    data class AiMessage(
        override val id: Long,
        val text: String,
        val isGenerating: Boolean = false,
        val metrics: InferenceMetrics? = null,
        // True while this bubble is only a "thinking" placeholder shown during
        // model loading (not an actual generation). Hides the stop button.
        val isLoadingPlaceholder: Boolean = false
    ) : ChatMessage()

    data class WelcomeCard(
        override val id: Long = 0L,
        val isTextOnly: Boolean = false
    ) : ChatMessage()

    /**
     * Inline card shown in the chat stream when the user sends a request but the
     * selected model has not been downloaded yet.  It never uses a dialog: the
     * card itself offers the download confirmation and, once accepted, turns
     * into a progress indicator in place.
     */
    data class ModelDownloadPrompt(
        override val id: Long,
        val modelName: String,
        val downloading: Boolean = false,
        val paused: Boolean = false,
        val fraction: Float = 0f,
        val status: String? = null
    ) : ChatMessage()
}

data class InferenceMetrics(
    val imageProcessingMs: Long?,
    val prefillTokensPerSecond: Double?,
    val decodeTokensPerSecond: Double?,
    val totalMs: Long
)
