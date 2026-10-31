package com.mobiinfer.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.os.SystemClock
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import io.noties.markwon.Markwon

class ChatAdapter(
    private val markwon: Markwon
) : ListAdapter<ChatMessage, RecyclerView.ViewHolder>(DiffCallback()) {

    companion object {
        private const val TYPE_WELCOME = 0
        private const val TYPE_USER = 1
        private const val TYPE_AI = 2
        private const val TYPE_DOWNLOAD = 3
        private const val TYPE_IMAGE_RESULT = 4
        // Minimum gap between forced rebinds of an unbound image row. Matches
        // the Activity's preview throttle: frames do not arrive faster anyway.
        private const val REBIND_INTERVAL_MS = 500L
        // Decoded results are cached by path: ListAdapter rebinds every time the
        // list is submitted, and re-decoding a 1-2 MB PNG on the main thread on
        // each pass would stutter while the next image streams in.
        private val bitmapCache = androidx.collection.LruCache<String, Bitmap>(6)

        /**
         * The name of the selected checkpoint for a bound row, or null.
         *
         * A View's context is not usable before it is attached
         * (applicationContext is still null), and binding does run that early
         * for the first submitList(); resolving the model there threw.
         */
        private fun modelNameOrNull(view: View): String? {
            val context = (view.context as? Context)?.applicationContext ?: return null
            return LlamaEngine.getSelectedModel(context).displayName
        }
    }

    private var onSuggestionClick: ((String) -> Unit)? = null
    private var onSuggestionImageClick: (() -> Unit)? = null
    private var onStopClick: (() -> Unit)? = null
    private var onDownloadConfirm: (() -> Unit)? = null
    private var onDownloadCancel: (() -> Unit)? = null
    private var onDownloadPause: (() -> Unit)? = null
    private var onDownloadResume: (() -> Unit)? = null
    private var onImageClick: ((String) -> Unit)? = null
    private var onImageLongClick: ((String) -> Unit)? = null
    private var onImageErrorClick: ((String) -> Unit)? = null

    private var activeAiHolder: AiMessageViewHolder? = null
    private var activeAiId: Long = -1L
    private var activeDownloadHolder: DownloadPromptViewHolder? = null
    private var activeDownloadId: Long = -1L
    // Row currently showing an in-flight image; progress and preview frames
    // are pushed straight into it instead of re-submitting the list.
    private var activeImageHolder: ImageResultViewHolder? = null
    private var activeImageId: Long = -1L
    // Last time a rebind was forced for a row that was not bound, per id.
    // Without this the "row not bound yet" fallback would submit a new list on
    // every preview frame and reintroduce the flicker it exists to avoid.
    private val lastRebindAt = HashMap<Long, Long>()

    fun setOnSuggestionClick(listener: (String) -> Unit) {
        onSuggestionClick = listener
    }

    fun setOnSuggestionImageClick(listener: () -> Unit) {
        onSuggestionImageClick = listener
    }

    fun setOnStopClick(listener: () -> Unit) {
        onStopClick = listener
    }

    fun setOnDownloadConfirm(listener: () -> Unit) {
        onDownloadConfirm = listener
    }

    fun setOnDownloadCancel(listener: () -> Unit) {
        onDownloadCancel = listener
    }

    fun setOnDownloadPause(listener: () -> Unit) {
        onDownloadPause = listener
    }

    fun setOnDownloadResume(listener: () -> Unit) {
        onDownloadResume = listener
    }

    /** Full-screen preview of a generated picture. */
    fun setOnImageClick(listener: (String) -> Unit) {
        onImageClick = listener
    }

    /** Save a generated picture to the gallery. */
    fun setOnImageLongClick(listener: (String) -> Unit) {
        onImageLongClick = listener
    }

    /** Copy a generation error somewhere the user can paste it. */
    fun setOnImageErrorClick(listener: (String) -> Unit) {
        onImageErrorClick = listener
    }

    fun setActiveAiMessage(id: Long) {
        activeAiId = id
    }

    fun clearActiveAiMessage() {
        activeAiId = -1L
        activeAiHolder = null
    }

    fun setActiveDownload(id: Long) {
        activeDownloadId = id
        activeDownloadHolder = null
    }

    fun clearActiveDownload() {
        activeDownloadId = -1L
        activeDownloadHolder = null
    }

    /**
     * Direct, in-place progress update for the download card.  Going through
     * submitList() on every progress tick rebinds the item and restarts the
     * progress animation, which makes the card flicker.
     */
    fun updateDownloadProgress(id: Long, fraction: Float, status: String?) {
        if (id == activeDownloadId) {
            activeDownloadHolder?.updateProgress(fraction, status)
        }
    }

    /**
     * In-place progress update for the image bubble. Re-submitting the list
     * would rebind every row and restart the bar animation on each step.
     */
    /**
     * Latest state of every image row that is still drawing, by message id.
     *
     * The submitted list is only replaced when messages appear or disappear;
     * progress and preview frames change several times per second, and pushing
     * each of them through submitList() rebinds the row (restarting the
     * Material progress animation and re-setting the thumbnail), which is what
     * made the bar flicker continuously. onBindViewHolder() reads this map
     * instead, so a rebind - scrolling, a neighbouring message arriving - shows
     * the newest state rather than the progress 0 the list was submitted with.
     */
    private val liveImageItems = HashMap<Long, ChatMessage.ImageResult>()

    /** Records the newest state of a drawing row; repaints only its label. */
    fun updateImageItem(item: ChatMessage.ImageResult) {
        // Two paths write this row - the progress label and the preview frames -
        // and each carries whatever copy of the message its caller had. Keeping
        // the larger progress (and the newer frame) stops the older write from
        // undoing the newer one when a rebind reads the map back.
        val known = liveImageItems[item.id]
        val merged = if (known != null && known.isGenerating && item.isGenerating) {
            item.copy(
                progress = maxOf(item.progress, known.progress).coerceIn(0f, 1f),
                previewBitmap = item.previewBitmap ?: known.previewBitmap
            )
        } else {
            item
        }
        liveImageItems[item.id] = merged
        val holder = activeImageHolder
        val isBound = item.id == activeImageId && holder != null
        if (!isBound) {
            // The row is not the one currently bound: either another image is
            // drawing, or this row is not on screen yet. The frames must not be
            // dropped here - that is what made the preview appear only
            // sometimes, depending on whether the row happened to be bound when
            // the first frames arrived.
            //
            // notifyItemChanged() alone is not enough: it resolves the position
            // through currentList, and the image row is inserted ahead of the
            // assistant bubble, so while that insert is still pending the index
            // is -1 and the rebind is skipped. Re-submitting the list instead
            // always runs a diff, and onBindViewHolder() then reads this map.
            // A fresh list instance is required: submitList() compares by
            // identity first and would return early for currentList itself.
            // Throttled, because a rebind re-creates the row and a submit per
            // frame is exactly the flicker this avoids.
            if (item.isGenerating) {
                val now = SystemClock.elapsedRealtime()
                val last = lastRebindAt[item.id] ?: 0L
                if (now - last >= REBIND_INTERVAL_MS) {
                    lastRebindAt[item.id] = now
                    submitList(currentList.toMutableList())
                }
            } else {
                liveImageItems.remove(item.id)
            }
            return
        }
        holder.updateProgress(merged.progress, merged.stage)
        merged.previewBitmap?.let { holder.showPreview(it) }
    }

    /** Drops the live state of a row that finished or was removed. */
    fun forgetImageItem(id: Long) {
        liveImageItems.remove(id)
    }
    /**
     * Swaps in a live preview frame. Returns false when the row is no longer
     * bound, so the caller can recycle the bitmap.
     */
    fun showImagePreview(id: Long, bitmap: Bitmap): Boolean {
        val holder = activeImageHolder
        if (id != activeImageId || holder == null) return false
        return holder.showPreview(bitmap)
    }

    fun updateStreamingText(id: Long, text: String) {
        if (id == activeAiId) {
            activeAiHolder?.updateText(text)
        }
    }

    /**
     * Shows the "reply is coming" spinner on the bubble that [updateStreamingText]
     * will feed. Falls back to a rebind when the row is not bound yet (the
     * picture was just inserted/removed around it), because [updateStreamingText]
     * silently drops updates while there is no holder and the bubble would stay
     * frozen until the stream ends.
     */
    fun showPendingReply(id: Long, waiting: Boolean) {
        if (id != activeAiId) return
        val holder = activeAiHolder
        if (holder == null) {
            notifyItemChanged(indexOf(id))
            return
        }
        holder.showPending(waiting)
    }

    private fun indexOf(id: Long): Int {
        val items = currentList
        for (i in items.indices) {
            if (items[i].id == id) return i
        }
        return -1
    }

    fun setGeneratingDone(id: Long) {
        if (id == activeAiId) {
            activeAiHolder?.setStopButtonVisible(false)
        }
    }

    override fun getItemViewType(position: Int): Int {
        return when (getItem(position)) {
            is ChatMessage.WelcomeCard -> TYPE_WELCOME
            is ChatMessage.UserMessage -> TYPE_USER
            is ChatMessage.AiMessage -> TYPE_AI
            is ChatMessage.ModelDownloadPrompt -> TYPE_DOWNLOAD
            is ChatMessage.ImageResult -> TYPE_IMAGE_RESULT
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return when (viewType) {
            TYPE_WELCOME -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_welcome_card, parent, false)
                WelcomeViewHolder(view)
            }
            TYPE_USER -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_user_message, parent, false)
                UserMessageViewHolder(view)
            }
            TYPE_AI -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_ai_message, parent, false)
                AiMessageViewHolder(view)
            }
            TYPE_DOWNLOAD -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_download_prompt, parent, false)
                DownloadPromptViewHolder(view)
            }
            TYPE_IMAGE_RESULT -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_image_result, parent, false)
                ImageResultViewHolder(view)
            }
            else -> throw IllegalArgumentException("Unknown view type: $viewType")
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        when (val item = getItem(position)) {
            is ChatMessage.WelcomeCard -> (holder as WelcomeViewHolder).bind(item)
            is ChatMessage.UserMessage -> (holder as UserMessageViewHolder).bind(item)
            is ChatMessage.AiMessage -> {
                val aiHolder = holder as AiMessageViewHolder
                aiHolder.bind(item)
                if (item.id == activeAiId) {
                    activeAiHolder = aiHolder
                } else if (aiHolder == activeAiHolder) {
                    // The row was rebound with a newer copy of the streaming
                    // bubble (the code below the image tool replaces the whole
                    // message, so its id no longer matched). Dropping it left
                    // updateStreamingText() with nothing to update and the
                    // reply stayed invisible until the stream finished.
                    activeAiId = item.id
                }
            }
            is ChatMessage.ModelDownloadPrompt -> {
                val promptHolder = holder as DownloadPromptViewHolder
                promptHolder.bind(item)
                if (item.id == activeDownloadId) {
                    activeDownloadHolder = promptHolder
                }
            }
            is ChatMessage.ImageResult -> {
                val imageHolder = holder as ImageResultViewHolder
                // A rebind must show the newest progress/preview of a drawing
                // row, not the snapshot the list was submitted with.
                val live = if (item.isGenerating) liveImageItems[item.id] else null
                imageHolder.bind(live ?: item)
                // Identity check, not id: a finishing row can be bound into the
                // very holder that was tracking it, and leaving that holder as
                // the active target made the next draw's progress land on a row
                // that was no longer generating.
                if (imageHolder == activeImageHolder && !item.isGenerating) {
                    activeImageHolder = null
                    activeImageId = -1L
                }
                if (item.isGenerating) {
                    activeImageHolder = imageHolder
                    activeImageId = item.id
                } else {
                    liveImageItems.remove(item.id)
                    lastRebindAt.remove(item.id)
                }
            }
        }
    }

    override fun onViewRecycled(holder: RecyclerView.ViewHolder) {
        if (holder is AiMessageViewHolder && holder == activeAiHolder) {
            activeAiHolder = null
        }
        if (holder is DownloadPromptViewHolder && holder == activeDownloadHolder) {
            activeDownloadHolder = null
        }
        if (holder is ImageResultViewHolder && holder == activeImageHolder) {
            activeImageHolder = null
            activeImageId = -1L
        }
    }

    inner class WelcomeViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvWelcomeTitle: TextView = itemView.findViewById(R.id.tv_welcome_title)
        private val tvWelcomeDesc: TextView = itemView.findViewById(R.id.tv_welcome_desc)
        private val btnSuggestion1: MaterialButton = itemView.findViewById(R.id.btn_suggestion_1)
        private val btnSuggestion2: MaterialButton = itemView.findViewById(R.id.btn_suggestion_2)

        private val ivWelcomeIcon: ImageView = itemView.findViewById(R.id.iv_welcome_icon)

        fun bind(item: ChatMessage.WelcomeCard) {
            val ctx = itemView.context
            ivWelcomeIcon.visibility = if (item.isImageMode) View.VISIBLE else View.GONE
            if (item.isImageMode) {
                tvWelcomeTitle.setText(R.string.welcome_title_image)
                tvWelcomeDesc.setText(R.string.welcome_desc_image)
                btnSuggestion1.setText(R.string.suggestion_image_1)
                btnSuggestion2.setText(R.string.suggestion_image_2)
                btnSuggestion1.setIconResource(R.drawable.ic_image)
                btnSuggestion2.setIconResource(R.drawable.ic_image)
                btnSuggestion1.setOnClickListener { onSuggestionClick?.invoke(btnSuggestion1.text.toString()) }
                btnSuggestion2.setOnClickListener { onSuggestionClick?.invoke(btnSuggestion2.text.toString()) }
                return
            }
            if (item.isTextOnly) {
                tvWelcomeTitle.setText(R.string.welcome_title_text)
                tvWelcomeDesc.setText(R.string.welcome_desc_text)
                btnSuggestion1.setText(R.string.suggestion_1)
                btnSuggestion2.setText(R.string.suggestion_1_text)
                btnSuggestion1.setIconResource(R.drawable.ic_lightbulb)
                btnSuggestion2.setIconResource(R.drawable.ic_lightbulb)
            } else {
                tvWelcomeTitle.setText(R.string.welcome_title)
                tvWelcomeDesc.setText(R.string.welcome_desc)
                btnSuggestion1.setText(R.string.suggestion_1)
                btnSuggestion2.setText(R.string.suggestion_2)
                btnSuggestion1.setIconResource(R.drawable.ic_lightbulb)
                btnSuggestion2.setIconResource(R.drawable.ic_image)
            }
            val s1 = ctx.getString(R.string.suggestion_1_prompt)
            val s2 = btnSuggestion2.text.toString()
            btnSuggestion1.setOnClickListener { onSuggestionClick?.invoke(s1) }
            if (item.isTextOnly) {
                btnSuggestion2.setOnClickListener { onSuggestionClick?.invoke(s2) }
            } else {
                btnSuggestion2.setOnClickListener { onSuggestionImageClick?.invoke() }
            }
        }
    }

    inner class DownloadPromptViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvTitle: TextView = itemView.findViewById(R.id.tv_download_title)
        private val tvStatus: TextView = itemView.findViewById(R.id.tv_download_status)
        private val progress: LinearProgressIndicator = itemView.findViewById(R.id.progress_download)
        private val layoutButtons: View = itemView.findViewById(R.id.layout_download_buttons)
        private val btnConfirm: MaterialButton = itemView.findViewById(R.id.btn_download_confirm)
        private val btnCancel: MaterialButton = itemView.findViewById(R.id.btn_download_cancel)
        private val btnPause: MaterialButton = itemView.findViewById(R.id.btn_download_pause)
        private val btnResume: MaterialButton = itemView.findViewById(R.id.btn_download_resume)

        fun bind(item: ChatMessage.ModelDownloadPrompt) {
            tvTitle.text = itemView.context.getString(R.string.model_download_prompt, item.modelName)
            layoutButtons.visibility = View.VISIBLE
            if (item.downloading) {
                progress.visibility = View.VISIBLE
                progress.isIndeterminate = false
                progress.setProgressCompat((item.fraction * 100f).toInt().coerceIn(0, 100), false)
                applyStatus(item.status)
            } else {
                progress.visibility = View.GONE
                tvStatus.visibility = View.GONE
            }
            btnConfirm.visibility = if (!item.downloading) View.VISIBLE else View.GONE
            btnCancel.visibility = if (!item.downloading || item.paused) View.VISIBLE else View.GONE
            btnPause.visibility = if (item.downloading && !item.paused) View.VISIBLE else View.GONE
            btnResume.visibility = if (item.downloading && item.paused) View.VISIBLE else View.GONE

            btnConfirm.setOnClickListener { onDownloadConfirm?.invoke() }
            btnCancel.setOnClickListener { onDownloadCancel?.invoke() }
            btnPause.setOnClickListener { onDownloadPause?.invoke() }
            btnResume.setOnClickListener { onDownloadResume?.invoke() }
        }

        /** In-place progress update; avoids rebinding / animation restart. */
        fun updateProgress(fraction: Float, status: String?) {
            layoutButtons.visibility = View.GONE
            progress.visibility = View.VISIBLE
            progress.isIndeterminate = false
            progress.setProgressCompat((fraction * 100f).toInt().coerceIn(0, 100), false)
            applyStatus(status)
        }

        private fun applyStatus(status: String?) {
            if (status != null) {
                tvStatus.visibility = View.VISIBLE
                tvStatus.text = status
            } else {
                tvStatus.visibility = View.GONE
            }
        }
    }

    inner class ImageResultViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvModel: TextView = itemView.findViewById(R.id.tv_image_model)
        private val layoutProgress: View = itemView.findViewById(R.id.layout_image_progress)
        private val tvProgress: TextView = itemView.findViewById(R.id.tv_image_progress)
        private val ivResult: ImageView = itemView.findViewById(R.id.iv_image_result)
        private val tvError: TextView = itemView.findViewById(R.id.tv_image_error)
        private val tvParams: TextView = itemView.findViewById(R.id.tv_image_params)
        private val tvHint: TextView = itemView.findViewById(R.id.tv_image_hint)
        private val progressBar: LinearProgressIndicator = itemView.findViewById(R.id.progress_image_gen)
        // The model that drew this picture is recorded on the message itself
        // (the selection may have changed since), but ImageResult carries no
        // model name; the label is therefore only shown while generating.
        private var boundPath: String? = null
        // Id of the draw this row is showing, and the furthest the bar has got.
        // bind() receives whatever copy of the item the list holds, and the
        // progress label and the preview frames are written by two different
        // paths, so that copy is not always the newest one. Without a floor the
        // bar stepped back to the older value on every rebind - the flicker.
        private var progressItemId: Long = -1L
        private var maxProgress: Float = 0f

        fun bind(item: ChatMessage.ImageResult) {
            val ctx = itemView.context
            // A recycled view may still hold the previous draw's high-water mark;
            // the floor only makes sense within one message.
            if (item.id != progressItemId) {
                progressItemId = item.id
                maxProgress = 0f
            }
            // The card names the checkpoint that draws the picture, not the
            // selection: during a tool turn the selection is the language model.
            tvModel.text = item.imageModelName
                ?: modelNameOrNull(itemView)?.takeIf { it.isNotEmpty() }

            val params = mutableListOf<String>()
            item.paramsText?.let { params += it }
            item.elapsedMs?.let {
                params += ctx.getString(R.string.image_result_elapsed, formatDuration(it))
            }
            if (params.isNotEmpty() && !item.isGenerating && item.error == null) {
                tvParams.text = params.joinToString(" · ")
                tvParams.visibility = View.VISIBLE
            } else {
                tvParams.visibility = View.GONE
            }

            when {
                item.isGenerating -> {
                    layoutProgress.visibility = View.VISIBLE
                    tvError.visibility = View.GONE
                    tvHint.visibility = View.GONE
                    updateProgress(item.progress, item.stage)
                    val preview = item.previewBitmap
                    if (preview != null) {
                        // Only replace what the view shows when it is a newer
                        // frame. bind() can run after showPreview() painted a
                        // fresh one straight into the ImageView, and rebinding
                        // the slightly older frame made the thumbnail step
                        // backwards (a visible stutter on every rebind).
                        if (ivResult.drawable?.let { it as? BitmapDrawable }?.bitmap !== preview) {
                            ivResult.setImageBitmap(preview)
                        }
                        ivResult.visibility = View.VISIBLE
                    } else {
                        ivResult.setImageDrawable(null)
                        ivResult.visibility = View.GONE
                    }
                }
                item.error != null -> {
                    layoutProgress.visibility = View.GONE
                    ivResult.visibility = View.GONE
                    tvHint.visibility = View.GONE
                    tvError.visibility = View.VISIBLE
                    // Show the engine's own words: "image generation failed"
                    // alone gives the user (and us) nothing to act on. Tap
                    // copies it for a bug report.
                    tvError.text = ctx.getString(R.string.image_error, item.error) +
                        "\n" + ctx.getString(R.string.image_copy_error)
                    tvError.setOnClickListener { onImageErrorClick?.invoke(item.error) }
                }
                else -> {
                    layoutProgress.visibility = View.GONE
                    tvError.visibility = View.GONE
                    val path = item.pngPath
                    if (path == null) {
                        ivResult.visibility = View.GONE
                        return
                    }
                    ivResult.visibility = View.VISIBLE
                    if (boundPath != path) {
                        boundPath = path
                        val bitmap = bitmapCache[path] ?: runCatching {
                            ImageOutput.decodeForDisplay(path)
                        }.getOrNull()?.also { bitmapCache.put(path, it) }
                        ivResult.setImageBitmap(bitmap)
                        ivResult.visibility = if (bitmap != null) View.VISIBLE else View.GONE
                    }
                    val available = ivResult.drawable != null
                    tvHint.visibility = if (available) View.VISIBLE else View.GONE
                    // Tap opens the picture full screen, long-press offers to
                    // save it; both are also reachable from the viewer.
                    ivResult.setOnClickListener { if (available) onImageClick?.invoke(path) }
                    ivResult.setOnLongClickListener {
                        if (available) onImageLongClick?.invoke(path)
                        available
                    }
                    tvHint.setOnClickListener { if (available) onImageClick?.invoke(path) }
                }
            }
        }

        fun updateProgress(fraction: Float, stage: String?) {
            layoutProgress.visibility = View.VISIBLE
            if (fraction < 0f) {
                progressBar.isIndeterminate = true
            } else {
                progressBar.isIndeterminate = false
                // The bar never goes backwards within one draw: a stale copy of
                // the item (another write path got there first) must not pull it
                // back down to an earlier percentage.
                val shown = maxOf(fraction, maxProgress).coerceIn(0f, 1f)
                maxProgress = shown
                // No animation: a 500 ms cross-fade per step reads as jitter
                // while the bar is being restarted every step.
                progressBar.setProgressCompat((shown * 100f).toInt().coerceIn(0, 100), false)
            }
            stage?.let { tvProgress.text = it }
        }

        /** Shows a step preview in place of the (not yet existing) result. */
        fun showPreview(bitmap: Bitmap): Boolean {
            ivResult.setImageBitmap(bitmap)
            ivResult.visibility = View.VISIBLE
            return true
        }
        private fun formatDuration(milliseconds: Long): String =
            String.format(java.util.Locale.getDefault(), "%.1fs", milliseconds / 1000.0)
    }

    inner class UserMessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvText: TextView = itemView.findViewById(R.id.tv_user_text)
        private val flImageContainer: View = itemView.findViewById(R.id.fl_user_image_container)
        private val ivImage: ImageView = itemView.findViewById(R.id.iv_user_image)
        private val ivVideoBadge: ImageView = itemView.findViewById(R.id.iv_video_play_badge)
        private val tvImageInfo: TextView = itemView.findViewById(R.id.tv_image_info)
        private val progressImage: LinearProgressIndicator = itemView.findViewById(R.id.progress_image)

        fun bind(item: ChatMessage.UserMessage) {
            tvText.text = item.text
            tvText.visibility = if (item.text.isNotBlank()) View.VISIBLE else View.GONE

            if (item.imageBitmap != null) {
                ivImage.setImageBitmap(item.imageBitmap)
                flImageContainer.visibility = View.VISIBLE
                ivVideoBadge.visibility = if (item.isVideo) View.VISIBLE else View.GONE
                tvImageInfo.visibility = View.VISIBLE
                tvImageInfo.text = item.imageInfo ?: ""
                progressImage.visibility = if (item.isPrefilling) View.VISIBLE else View.GONE
            } else {
                flImageContainer.visibility = View.GONE
                ivVideoBadge.visibility = View.GONE
                tvImageInfo.visibility = View.GONE
                progressImage.visibility = View.GONE
            }
        }
    }

    inner class AiMessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvAiModel: TextView = itemView.findViewById(R.id.tv_ai_model)
        private val tvText: TextView = itemView.findViewById(R.id.tv_ai_text)
        private val tvStatus: TextView = itemView.findViewById(R.id.tv_ai_status)
        private val tvMetrics: TextView = itemView.findViewById(R.id.tv_ai_metrics)
        private val btnStop: MaterialButton = itemView.findViewById(R.id.btn_stop_generating)
        private val layoutThinking: View = itemView.findViewById(R.id.layout_thinking)
        private val layoutThinkingHeader: View = itemView.findViewById(R.id.layout_thinking_header)
        private val tvThinkingArrow: TextView = itemView.findViewById(R.id.tv_thinking_arrow)
        private val tvThinkingLabel: TextView = itemView.findViewById(R.id.tv_thinking_label)
        private val tvThinkingText: TextView = itemView.findViewById(R.id.tv_thinking_text)
        private val dividerThinking: View = itemView.findViewById(R.id.divider_thinking)

        private val prefillStatus: View = itemView.findViewById(R.id.layout_prefill_status)
        private val tvPrefillStatus: TextView = itemView.findViewById(R.id.tv_prefill_status)
        private var thinkingExpanded = false
        private var streamingMinWidth = 0

        fun bind(item: ChatMessage.AiMessage) {
            // A tool turn leaves the assistant bubble empty while the image is
            // being drawn: showing it would stack a second model header with no
            // content above the picture.
            val hideWholeBubble = item.waitingForImage && item.text.isBlank()
            itemView.visibility = if (hideWholeBubble) View.GONE else View.VISIBLE
            if (hideWholeBubble) return
            tvAiModel.text = modelNameOrNull(itemView).orEmpty()
            if (!item.status.isNullOrBlank()) {
                tvStatus.text = item.status
                tvStatus.visibility = View.VISIBLE
            } else {
                tvStatus.visibility = View.GONE
            }
            if (!item.isGenerating) {
                streamingMinWidth = 0
                (tvText.parent as? ViewGroup)?.minimumWidth = 0
            }
            // The same spinner block is used for two different waits: model
            // loading (placeholder bubble) and prefill/thinking. Keep the copy
            // distinct so they are not confused.
            tvPrefillStatus.text = itemView.context.getString(
                when {
                    item.waitingForImage -> R.string.tool_generating_image_short
                    item.isLoadingPlaceholder -> R.string.model_loading_in_progress
                    else -> R.string.thinking_in_progress
                }
            )
            renderWithThinking(item.text, item.isGenerating)
            val metrics = item.metrics
            if (!item.isGenerating && metrics != null) {
                val parts = mutableListOf<String>()
                metrics.imageProcessingMs?.let {
                    parts += itemView.context.getString(R.string.metric_image_time, formatDuration(it))
                }
                metrics.prefillTokensPerSecond?.let {
                    parts += itemView.context.getString(R.string.metric_prefill, it)
                }
                metrics.decodeTokensPerSecond?.let {
                    parts += itemView.context.getString(R.string.metric_decode, it)
                }
                parts += itemView.context.getString(R.string.metric_total, formatDuration(metrics.totalMs))
                tvMetrics.text = parts.joinToString(itemView.context.getString(R.string.metric_separator))
                tvMetrics.visibility = View.VISIBLE
            } else {
                tvMetrics.visibility = View.GONE
            }
            btnStop.visibility = if (item.isGenerating && !item.isLoadingPlaceholder) View.VISIBLE else View.GONE
            btnStop.setOnClickListener {
                onStopClick?.invoke()
            }
        }

        /**
         * Grows the bubble to say a reply is coming before the first token.
         *
         * The tool flow leaves the assistant bubble empty and collapsed while
         * the picture is drawn, and an empty bubble has nothing to render the
         * spinner into: the picture appeared and then nothing happened until
         * the first token, which on a reloaded model is many seconds later. It
         * shows the text line (blank) and the spinner below the model name,
         * and only once the model is clearly answering ([waiting] false).
         */
        fun showPending(waiting: Boolean) {
            itemView.visibility = View.VISIBLE
            tvText.text = ""
            tvText.visibility = View.VISIBLE
            prefillStatus.visibility = View.VISIBLE
            tvPrefillStatus.text = itemView.context.getString(
                if (waiting) R.string.tool_generating_image_short else R.string.thinking_in_progress
            )
        }

        private fun formatDuration(milliseconds: Long): String =
            String.format(java.util.Locale.getDefault(), "%.1fs", milliseconds / 1000.0)

        fun updateText(text: String) {
            val contentLayout = tvText.parent as? ViewGroup
            if (contentLayout != null && contentLayout.width > streamingMinWidth) {
                streamingMinWidth = contentLayout.width
            }
            contentLayout?.minimumWidth = streamingMinWidth
            renderWithThinking(text, true)
        }

        fun setStopButtonVisible(visible: Boolean) {
            btnStop.visibility = if (visible) View.VISIBLE else View.GONE
            if (!visible) prefillStatus.visibility = View.GONE
        }

        private fun renderWithThinking(raw: String, isGenerating: Boolean) {
            // The prefill spinner only makes sense for a plain token stream; a
            // status line (e.g. image tool progress) replaces it.
            prefillStatus.visibility =
                if (isGenerating && raw.isEmpty() && tvStatus.visibility != View.VISIBLE) View.VISIBLE else View.GONE
            val parsed = parseThinkingBlock(raw, isGenerating)

            if (parsed.thinkingText != null) {
                layoutThinking.visibility = View.VISIBLE

                if (parsed.isThinking) {
                    tvThinkingLabel.text = itemView.context.getString(R.string.thinking_in_progress)
                    thinkingExpanded = true
                } else {
                    tvThinkingLabel.text = itemView.context.getString(R.string.thinking_process)
                }

                tvThinkingArrow.text = if (thinkingExpanded) "▾" else "▸"
                tvThinkingText.visibility = if (thinkingExpanded) View.VISIBLE else View.GONE
                dividerThinking.visibility = if (!parsed.isThinking) View.VISIBLE else View.GONE

                markwon.setMarkdown(tvThinkingText,
                    MarkdownEscape.normalizeResponseText(parsed.thinkingText))

                layoutThinkingHeader.setOnClickListener {
                    thinkingExpanded = !thinkingExpanded
                    tvThinkingArrow.text = if (thinkingExpanded) "▾" else "▸"
                    tvThinkingText.visibility = if (thinkingExpanded) View.VISIBLE else View.GONE
                }
            } else {
                layoutThinking.visibility = View.GONE
            }

            val display = parsed.responseText
            if (display.isNotEmpty()) {
                tvText.visibility = View.VISIBLE
                markwon.setMarkdown(tvText, MarkdownEscape.normalizeResponseText(display))
            } else {
                tvText.visibility = View.GONE
            }
        }

        private fun parseThinkingBlock(text: String, isGenerating: Boolean): ParsedThinking {
            val thinkStart = text.indexOf("<think>")
            if (thinkStart < 0) {
                return ParsedThinking(null, text, false)
            }

            val contentAfterTag = text.substring(thinkStart + "<think>".length)
            val thinkEnd = contentAfterTag.indexOf("</think>")

            return if (thinkEnd < 0) {
                ParsedThinking(
                    thinkingText = contentAfterTag.trim(),
                    responseText = "",
                    isThinking = true
                )
            } else {
                ParsedThinking(
                    thinkingText = contentAfterTag.substring(0, thinkEnd).trim(),
                    responseText = contentAfterTag.substring(thinkEnd + "</think>".length).trimStart('\n'),
                    isThinking = false
                )
            }
        }
    }

    private data class ParsedThinking(
        val thinkingText: String?,
        val responseText: String,
        val isThinking: Boolean
    )

    class DiffCallback : DiffUtil.ItemCallback<ChatMessage>() {
        override fun areItemsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean {
            return when {
                oldItem is ChatMessage.UserMessage && newItem is ChatMessage.UserMessage ->
                    oldItem.text == newItem.text &&
                            oldItem.imageBitmap == newItem.imageBitmap &&
                            oldItem.imageInfo == newItem.imageInfo &&
                            oldItem.isPrefilling == newItem.isPrefilling &&
                            oldItem.isVideo == newItem.isVideo
                oldItem is ChatMessage.AiMessage && newItem is ChatMessage.AiMessage ->
                    // waitingForImage and status are part of the bubble: a
                    // submitList() that only changed them was treated as
                    // "same content" and never rebound, so the spinner the
                    // model's turn needs never appeared after the picture.
                    oldItem.isGenerating == newItem.isGenerating &&
                            oldItem.waitingForImage == newItem.waitingForImage &&
                            oldItem.status == newItem.status &&
                            (oldItem.isGenerating || (oldItem.text == newItem.text && oldItem.metrics == newItem.metrics))
                oldItem is ChatMessage.WelcomeCard && newItem is ChatMessage.WelcomeCard ->
                    oldItem.isTextOnly == newItem.isTextOnly &&
                            oldItem.isImageMode == newItem.isImageMode
                oldItem is ChatMessage.ImageResult && newItem is ChatMessage.ImageResult ->
                    oldItem.isGenerating == newItem.isGenerating &&
                            oldItem.progress == newItem.progress &&
                            oldItem.stage == newItem.stage &&
                            // Presence, not identity: every step brings a new
                            // Bitmap, and comparing identities rebound the row
                            // on every frame. The new frame is painted directly
                            // into the bound view by showPreview(), so a rebind
                            // is only needed when a preview first appears or
                            // disappears (which decides GONE vs VISIBLE).
                            (oldItem.previewBitmap != null) == (newItem.previewBitmap != null) &&
                            oldItem.pngPath == newItem.pngPath &&
                            oldItem.error == newItem.error &&
                            oldItem.paramsText == newItem.paramsText &&
                            oldItem.elapsedMs == newItem.elapsedMs
                oldItem is ChatMessage.ModelDownloadPrompt && newItem is ChatMessage.ModelDownloadPrompt ->
                    oldItem.modelName == newItem.modelName &&
                            oldItem.downloading == newItem.downloading &&
                            oldItem.paused == newItem.paused &&
                            oldItem.fraction == newItem.fraction &&
                            oldItem.status == newItem.status
                else -> false
            }
        }
    }
}
