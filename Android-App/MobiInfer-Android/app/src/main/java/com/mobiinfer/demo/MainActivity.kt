package com.mobiinfer.demo

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.InputMethodManager
import android.widget.ArrayAdapter
import android.widget.ImageButton
import android.widget.ListPopupWindow
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.AppBarLayout
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import io.noties.markwon.Markwon
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var recyclerChat: RecyclerView
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var etInput: TextInputEditText
    private lateinit var btnSend: ImageButton
    private lateinit var btnImage: ImageButton
    private lateinit var btnThink: MaterialButton
    private lateinit var btnModelManager: ImageButton
    private lateinit var btnModelSelect: ImageButton
    private lateinit var cardInputBar: View
    private lateinit var appBarLayout: AppBarLayout
    private lateinit var tvTitle: TextView
    private lateinit var tvTitleModel: TextView
    private lateinit var loadingOverlay: View
    private lateinit var tvLoadingText: TextView
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var recyclerConversations: RecyclerView
    private lateinit var conversationAdapter: ConversationAdapter

    private var currentConversationId: String? = null
    // Set when the UI is showing a conversation whose history still needs to be
    // pushed into the engine (cold start, model reload, or conversation switch).
    private var pendingRestore = false
    // Throttle for the in-flight streaming checkpoint (see [persistDuringStreaming]).
    private var lastStreamPersistAt = 0L
    // While the model streams, only keep the list pinned to the bottom when the
    // user has not scrolled away. Scrolling up disables it until they return to
    // the bottom (or send a new message).
    private var autoFollowStreaming = true

    private lateinit var engine: LlamaEngine
    private var generationJob: Job? = null
    private var isModelReady = false
    private var isImagePrefilled = false
    private var pendingImageProcessingMs: Long? = null
    private var pendingImageQuestion: String? = null
    // Image/video chosen before the model was ready; processed after it loads.
    private var pendingMediaUri: Uri? = null
    private var isProcessingVideo = false
    private var hasAutoLoaded = false
    private var loadedModelId: String? = null
    private var pendingPrompt: String? = null
    private var downloadPromptId: Long = -1L
    private var downloadModelId: String? = null
    private var isInlineDownloading = false
    private var isInlineModelLoad = false
    private var lastDownloadUiAt = 0L
    private var messageIdCounter = 1L
    private val messages = mutableListOf<ChatMessage>()
    private var createdWithLocale: String? = null
    private var isLocaleRestart = false

    private data class PreparedImage(
        val prompt: String,
        val bytes: ByteArray,
        val bitmap: Bitmap,
        val info: String,
        val preparationMs: Long
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createdWithLocale = LocaleManager.currentLanguage(this).tag

        // Roll an outdated stored selection forward to the shipped default
        // (MiniCPM). Must run before anything reads getSelectedModel().
        LlamaEngine.applyDefaultModelIfNeeded(applicationContext)

        // If the selected model is a TTS model, redirect to TtsActivity immediately.
        // The chat interface is only meaningful for LLM/VLM models.
        if (shouldRedirectToTts()) {
            startActivity(Intent(this, TtsActivity::class.java))
            finish()
            return
        }

        if (shouldRedirectToImage()) {
            startActivity(Intent(this, ImageActivity::class.java))
            finish()
            return
        }

        setContentView(R.layout.activity_main)

        // Edge-to-edge: pad the root content for status/nav bars and the IME
        // so the bottom input bar follows the soft keyboard up. Without this,
        // targetSdk=35+ draws content behind the IME and the input bar gets
        // covered.
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val rootContent = findViewById<View>(android.R.id.content)
        ViewCompat.setOnApplyWindowInsetsListener(rootContent) { v, insets ->
            val sysBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            v.updatePadding(
                left = sysBars.left,
                top = sysBars.top,
                right = sysBars.right,
                bottom = maxOf(sysBars.bottom, ime.bottom)
            )
            insets
        }

        LlamaEngine.migrateLegacyLayoutIfNeeded(applicationContext)

        initViews()
        syncSelectedModelUi()
        setupRecyclerView()

        val restoredConversation = savedInstanceState?.getString(STATE_CURRENT_CONVERSATION)
        if (restoredConversation != null) {
            // Activity recreation (rotation / process restore): keep the chat.
            openConversation(restoredConversation, restoreContext = true)
        } else {
            // Fresh launch: always begin a new conversation.  It stays a draft
            // (not in the history list) until the user actually sends something.
            openConversation(ConversationStore.create(applicationContext).id, restoreContext = false)
        }

        setupClickListeners()
        initEngine()
    }

    private fun initViews() {
        recyclerChat = findViewById(R.id.recycler_chat)
        etInput = findViewById(R.id.et_input)
        btnSend = findViewById(R.id.btn_send)
        btnImage = findViewById(R.id.btn_image)
        btnThink = findViewById(R.id.btn_think)
        btnModelManager = findViewById(R.id.btn_model_manager)
        btnModelSelect = findViewById(R.id.btn_model_select)
        cardInputBar = findViewById(R.id.card_input_bar)
        appBarLayout = findViewById(R.id.appBarLayout)
        tvTitle = findViewById(R.id.tv_title)
        tvTitleModel = findViewById(R.id.tv_title_model)
        syncSelectedModelUi()
        loadingOverlay = findViewById(R.id.loading_overlay)
        tvLoadingText = findViewById(R.id.tv_loading_text)
        drawerLayout = findViewById(R.id.drawer_layout)
        recyclerConversations = findViewById(R.id.recycler_conversations)
    }

    private fun showLoading(show: Boolean, textRes: Int = R.string.status_loading) {
        if (show) {
            tvLoadingText.setText(textRes)
            loadingOverlay.visibility = View.VISIBLE
        } else {
            loadingOverlay.visibility = View.GONE
        }
    }

    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter(MarkdownRenderer.create(this))
        chatAdapter.setOnStopClick {
            engine.cancelGeneration()
        }
        chatAdapter.setOnDownloadConfirm {
            startInlineDownload()
        }
        chatAdapter.setOnDownloadCancel {
            cancelDownloadPrompt()
        }
        chatAdapter.setOnDownloadPause {
            ModelDownloadService.pause(this)
        }
        chatAdapter.setOnDownloadResume {
            resumeInlineDownload()
        }
        chatAdapter.setOnSuggestionClick { suggestion ->
            if (isProcessingVideo) {
                Toast.makeText(this, R.string.toast_wait_video, Toast.LENGTH_SHORT).show()
            } else {
                etInput.setText(suggestion)
                handleUserInput()
            }
        }
        chatAdapter.setOnSuggestionImageClick {
            when {
                isProcessingVideo ->
                    Toast.makeText(this, R.string.toast_wait_video, Toast.LENGTH_SHORT).show()
                else -> {
                    // Pick the image first; the model is loaded only after the
                    // user has chosen a file (see handleSelectedMedia()).
                    pendingImageQuestion = getString(R.string.suggestion_2)
                    getMedia.launch(arrayOf("image/*"))
                }
            }
        }

        recyclerChat.layoutManager = LinearLayoutManager(this)
        recyclerChat.adapter = chatAdapter
        recyclerChat.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(view: RecyclerView, newState: Int) {
                if (newState == RecyclerView.SCROLL_STATE_DRAGGING) {
                    // The user grabbed the list: follow only if they are at the bottom.
                    autoFollowStreaming = isChatAtBottom()
                } else if (newState == RecyclerView.SCROLL_STATE_IDLE && isChatAtBottom()) {
                    autoFollowStreaming = true
                }
            }

            override fun onScrolled(view: RecyclerView, dx: Int, dy: Int) {
                // During a user drag/fling follow the viewport; while idle only
                // re-enable following, never disable it (layout changes from
                // growing messages also emit onScrolled).
                if (view.scrollState == RecyclerView.SCROLL_STATE_IDLE) {
                    if (isChatAtBottom()) autoFollowStreaming = true
                } else {
                    autoFollowStreaming = isChatAtBottom()
                }
            }
        })

        cardInputBar.viewTreeObserver.addOnGlobalLayoutListener {
            recyclerChat.setPadding(
                recyclerChat.paddingLeft,
                recyclerChat.paddingTop,
                recyclerChat.paddingRight,
                cardInputBar.height
            )
        }

        setupConversationDrawer()
    }

    // -------------------------------------------------------- conversation drawer

    private fun setupConversationDrawer() {
        conversationAdapter = ConversationAdapter(
            onOpen = { meta ->
                if (meta.id != currentConversationId) openConversation(meta.id, restoreContext = true)
                drawerLayout.closeDrawers()
            },
            onDelete = { meta -> confirmDeleteConversation(meta) },
            onRename = { meta -> showRenameDialog(meta) }
        )
        recyclerConversations.layoutManager = LinearLayoutManager(this)
        recyclerConversations.adapter = conversationAdapter
    }

    private fun refreshConversationList() {
        conversationAdapter.submit(
            ConversationStore.list(applicationContext),
            currentConversationId
        )
    }

    /**
     * Shows [id] in the chat and, when [restoreContext] is set, replays its
     * history into the engine so follow-up turns keep their context.
     */
    private fun cancelActiveGeneration() {
        val job = generationJob
        if (job != null && job.isActive) {
            if (::engine.isInitialized) {
                runCatching { engine.cancelGeneration() }
            }
            job.cancel()
        }
        generationJob = null
    }

    private fun openConversation(id: String, restoreContext: Boolean) {
        // Switching conversations must abort the in-flight generation; otherwise
        // it keeps running against a cleared message list (wasted compute, and
        // the new chat's input stays disabled until it finishes).
        cancelActiveGeneration()
        persistCurrentConversation()
        currentConversationId = id
        ConversationStore.setCurrentId(applicationContext, id)

        messages.clear()
        val loaded = ConversationStore.load(applicationContext, id)
        val selectedModel = LlamaEngine.getSelectedModel(applicationContext)
        if (loaded.isEmpty()) {
            messages.add(ChatMessage.WelcomeCard(isTextOnly = selectedModel.isTextOnly))
        }
        messages.addAll(loaded)
        messageIdCounter = (loaded.maxOfOrNull { it.id } ?: 0L) + 1L
        isImagePrefilled = false
        pendingImageProcessingMs = null
        chatAdapter.submitList(messages.toList())
        refreshConversationList()

        pendingRestore = restoreContext && loaded.isNotEmpty()
        if (pendingRestore && ::engine.isInitialized && engine.state.value is LlamaState.ModelReady) {
            restoreCurrentConversationHistory()
        }
    }

    private fun persistCurrentConversation() {
        val id = currentConversationId ?: return
        // `messages` is only mutated on the main thread; snapshotting from an
        // IO/Default coroutine could race (ConcurrentModificationException or a
        // torn list), so marshal the snapshot to the main thread.
        if (Looper.myLooper() != Looper.getMainLooper()) {
            runOnUiThread { persistCurrentConversation() }
            return
        }
        val snapshot = ArrayList<ChatMessage>(messages.size).apply { addAll(messages) }
        lifecycleScope.launch(Dispatchers.IO) {
            ConversationStore.save(applicationContext, id, snapshot)
        }
    }

    /**
     * Checkpoints the conversation while the assistant answer is still
     * streaming, throttled to [STREAM_PERSIST_INTERVAL_MS]. Without this a
     * process kill mid-generation would leave the saved assistant bubble empty;
     * now at most ~1s of tokens are lost.
     */
    private fun persistDuringStreaming() {
        val now = System.currentTimeMillis()
        if (now - lastStreamPersistAt < STREAM_PERSIST_INTERVAL_MS) return
        lastStreamPersistAt = now
        persistCurrentConversation()
    }

    private fun restoreCurrentConversationHistory() {
        val id = currentConversationId ?: return
        pendingRestore = false
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val turns = ConversationStore.toHistoryTurns(ConversationStore.load(applicationContext, id))
                if (turns.isEmpty()) return@launch
                engine.clearContext()
                engine.restoreConversation(turns)
                withContext(Dispatchers.Main) { refreshConversationList() }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to restore conversation history", e)
            }
        }
    }

    private fun startNewConversation() {
        persistCurrentConversation()
        val meta = ConversationStore.create(applicationContext)
        openConversation(meta.id, restoreContext = false)
        clearEngineContext()
        drawerLayout.closeDrawers()
    }

    private fun clearEngineContext() {
        if (!::engine.isInitialized || engine.state.value !is LlamaState.ModelReady) return
        lifecycleScope.launch(Dispatchers.IO) {
            try { engine.clearContext() } catch (e: Exception) { Log.w(TAG, "clearContext failed", e) }
        }
    }

    private fun confirmDeleteConversation(meta: ConversationMeta) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_conversation_title)
            .setMessage(R.string.delete_conversation_confirm)
            .setPositiveButton(R.string.delete) { _, _ ->
                val deletingCurrent = meta.id == currentConversationId
                ConversationStore.delete(applicationContext, meta.id)
                if (deletingCurrent) {
                    val next = ConversationStore.list(applicationContext).firstOrNull()
                        ?: ConversationStore.create(applicationContext)
                    openConversation(next.id, restoreContext = true)
                } else {
                    refreshConversationList()
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun showRenameDialog(meta: ConversationMeta) {
        val input = android.widget.EditText(this).apply {
            setText(meta.title)
            setHint(R.string.rename_conversation_hint)
            setSelection(text.length)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.rename_conversation)
            .setView(input)
            .setPositiveButton(R.string.confirm) { _, _ ->
                val name = input.text.toString().trim()
                if (name.isNotEmpty()) {
                    ConversationStore.rename(applicationContext, meta.id, name)
                    refreshConversationList()
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun setupClickListeners() {
        // Pick image OR video.  iOS demo's HXPhotoPicker exposes both
        // photo and video in a single picker; on Android we ask SAF
        // for either MIME, so the user gets the same "pick anything
        // viewable" affordance with no extra "video" button.  Video is
        // only fed to the model if the loaded model is V-4.6 (gated in
        // [handleSelectedMedia] / [LlamaEngine.isVideoUnderstandingSupported]).
        btnImage.setOnClickListener { getMedia.launch(arrayOf("image/*", "video/*")) }
        btnThink.setOnClickListener {
            val next = !LlamaEngine.getEnableThinking(this)
            engine.setEnableThinking(next)
            refreshThinkButton()
        }
        findViewById<View>(R.id.btn_language).setOnClickListener {
            val languages = LocaleManager.AppLanguage.entries
            AlertDialog.Builder(this)
                .setTitle(R.string.language_picker_title)
                .setSingleChoiceItems(
                    languages.map { it.displayName }.toTypedArray(),
                    languages.indexOf(LocaleManager.currentLanguage(this))
                ) { dialog, which ->
                    dialog.dismiss()
                    if (languages[which] != LocaleManager.currentLanguage(this)) {
                        isLocaleRestart = true
                        LocaleManager.setLanguageAndRestart(this, languages[which])
                    }
                }
                .show()
        }
        btnSend.setOnClickListener { handleUserInput() }
        tvTitleModel.setOnClickListener { showModelSelector() }
        btnModelSelect.setOnClickListener { showModelSelector() }
        btnModelManager.setOnClickListener {
            startActivity(Intent(this, ModelManagerActivity::class.java))
        }
        findViewById<ImageButton>(R.id.btn_menu).setOnClickListener {
            refreshConversationList()
            drawerLayout.openDrawer(androidx.core.view.GravityCompat.START)
        }
        findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_new_conversation)
            .setOnClickListener { startNewConversation() }

        etInput.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                collapseAppBar()
                scrollToBottom()
            }
        }
    }

    private fun collapseAppBar() {
        appBarLayout.setExpanded(false, true)
    }

    private fun isChatAtBottom(): Boolean = !recyclerChat.canScrollVertically(1)

    /**
     * Auto-scroll used by the streaming path: keeps the newest tokens visible
     * only while the user is following along at the bottom.
     */
    private fun followStreamingIfAtBottom() {
        scrollToBottom(onlyIfFollowing = true)
    }

    private fun scrollToBottom(onlyIfFollowing: Boolean = false) {
        recyclerChat.post {
            if (onlyIfFollowing && !autoFollowStreaming) return@post
            val adapterCount = chatAdapter.itemCount
            if (adapterCount == 0) return@post
            val layoutManager = recyclerChat.layoutManager as? LinearLayoutManager ?: return@post
            val lastView = layoutManager.findViewByPosition(adapterCount - 1)
            if (lastView != null) {
                val offset = recyclerChat.height - recyclerChat.paddingBottom - lastView.height
                layoutManager.scrollToPositionWithOffset(adapterCount - 1, offset.coerceAtMost(0))
            } else {
                recyclerChat.scrollToPosition(adapterCount - 1)
            }
        }
    }

    private fun showClearChatDialog() {
        AlertDialog.Builder(this)
            .setTitle(R.string.clear_chat)
            .setMessage(R.string.clear_chat_confirm)
            .setPositiveButton(R.string.confirm) { _, _ ->
                clearChat()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun clearChatUI() {
        messages.clear()
        val selectedModel = LlamaEngine.getSelectedModel(applicationContext)
        messages.add(ChatMessage.WelcomeCard(isTextOnly = selectedModel.isTextOnly))
        messageIdCounter = 1L
        isImagePrefilled = false
        pendingImageProcessingMs = null
        chatAdapter.submitList(messages.toList())
    }

    private fun takePersistableReadPermission(uri: Uri) {
        try {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (e: SecurityException) {
            Log.w(TAG, "Provider offers no persistable grant for $uri", e)
        }
    }

    private fun clearChat() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                engine.clearContext()
                withContext(Dispatchers.Main) {
                    clearChatUI()
                    persistCurrentConversation()
                    Toast.makeText(this@MainActivity, R.string.clear_chat_toast, Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error clearing context", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, getString(R.string.toast_clear_chat_failed, e.message), Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun initEngine() {
        lifecycleScope.launch(Dispatchers.Default) {
            engine = LlamaEngine.getInstance(applicationContext)
            withContext(Dispatchers.Main) {
                observeEngineState()
                observeDownloads()
            }
        }
    }

    private fun observeEngineState() {
        lifecycleScope.launch {
            engine.state.collect { state ->
                when (state) {
                    is LlamaState.Uninitialized,
                    is LlamaState.Initializing -> {
                        enableInput(false)
                    }
                    is LlamaState.Initialized -> {
                        showLoading(false)
                        if (!isModelReady) enableTextInput()
                    }
                    is LlamaState.LoadingModel -> {
                        enableInput(false)
                        if (!isInlineModelLoad) showLoading(true)
                    }
                    is LlamaState.ModelReady -> {
                        isModelReady = true
                        loadedModelId = LlamaEngine.getSelectedModel(applicationContext).id
                        showLoading(false)
                        enableInput(true)
                        updateUIForModelType()
                        if (pendingRestore) restoreCurrentConversationHistory()
                    }
                    is LlamaState.ProcessingSystemPrompt,
                    is LlamaState.ProcessingUserPrompt,
                    is LlamaState.Generating -> {
                        enableInput(false)
                    }
                    is LlamaState.PrefillingImage -> {
                        isModelReady = true
                        etInput.isEnabled = true
                        btnSend.isEnabled = !isProcessingVideo
                        btnImage.isEnabled = false
                    }
                    is LlamaState.UnloadingModel -> {
                        enableInput(false)
                    }
                    is LlamaState.Error -> {
                        isModelReady = false
                        showLoading(false)
                        enableTextInput()
                    }
                }
            }
        }
    }

    private fun enableInput(enable: Boolean) {
        etInput.isEnabled = enable
        btnSend.isEnabled = enable
        if (!enable) {
            btnImage.isEnabled = false
        } else {
            btnImage.isEnabled = engine.isVisionSupported
        }
    }

    /**
     * Before any model is loaded we still want the user to type and send, so the
     * request itself can trigger the download/load flow.  Thinking is stored as
     * a preference and applied when the selected model is loaded; vision stays
     * hidden until its mmproj is actually available.
     */
    private fun enableTextInput() {
        etInput.isEnabled = true
        btnSend.isEnabled = true
        btnImage.isEnabled = false
        btnImage.visibility = View.GONE
        val supportsThinking = LlamaEngine.getSelectedModel(applicationContext).supportsThinking
        btnThink.visibility = if (supportsThinking) View.VISIBLE else View.GONE
        if (supportsThinking) refreshThinkButton()
    }

    private fun shouldRedirectToTts(): Boolean {
        val model = LlamaEngine.getSelectedModel(applicationContext)
        return model.isTts
    }

    private fun shouldRedirectToImage(): Boolean {
        val model = LlamaEngine.getSelectedModel(applicationContext)
        return model.isImage
    }

    private fun updateUIForModelType() {
        val model = LlamaEngine.getSelectedModel(applicationContext)
        val isVision = engine.isVisionSupported

        tvTitle.setText(if (isVision) R.string.app_title else R.string.app_title_text)
        syncSelectedModelUi()
        btnImage.visibility = if (isVision) View.VISIBLE else View.GONE
        btnImage.isEnabled = isVision
        btnThink.visibility = if (model.supportsThinking) View.VISIBLE else View.GONE
        if (model.supportsThinking) {
            refreshThinkButton()
        }

        refreshWelcomeCard(model.isTextOnly)
    }

    private fun syncSelectedModelUi() {
        val modelName = LlamaEngine.getSelectedModel(applicationContext).displayName
        val switchHint = getString(R.string.select_model_named, modelName)
        tvTitleModel.text = getString(R.string.model_label, modelName)
        tvTitleModel.contentDescription = switchHint
        btnModelSelect.contentDescription = switchHint
    }

    /** Toolbar dropdown that changes [selectedModelId] only (lazy loading). */
    private fun showModelSelector() {
        val models = ModelInfo.AVAILABLE_MODELS
        val selectedId = LlamaEngine.getSelectedModel(this).id
        val labels = models.map { it.displayName }
        val popup = ListPopupWindow(this)
        popup.anchorView = btnModelSelect
        val popupAdapter = object : ArrayAdapter<String>(
            this, R.layout.item_model_option, R.id.tv_option_text, labels
        ) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getView(position, convertView, parent)
                view.findViewById<View>(R.id.iv_option_check).visibility =
                    if (models[position].id == selectedId) View.VISIBLE else View.INVISIBLE
                return view
            }
        }
        popup.setAdapter(popupAdapter)
        popup.isModal = true
        popup.width = (resources.displayMetrics.widthPixels * 0.8f).toInt()
        popup.setOnItemClickListener { _, _, position, _ ->
            popup.dismiss()
            val model = models[position]
            if (model.id == LlamaEngine.getSelectedModel(this).id) return@setOnItemClickListener
            LlamaEngine.setSelectedModel(this, model.id)
            syncSelectedModelUi()
            btnThink.visibility = if (model.supportsThinking) {
                View.VISIBLE
            } else {
                View.GONE
            }
            if (model.supportsThinking) refreshThinkButton()
            if (!LlamaEngine.modelsExist(this, model)) {
                Toast.makeText(this, R.string.model_not_downloaded_hint, Toast.LENGTH_SHORT).show()
            } else if (isModelReady && loadedModelId != model.id) {
                Toast.makeText(
                    this,
                    getString(R.string.model_switched_lazy, model.displayName),
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
        popup.show()
    }

    private fun refreshThinkButton() {
        val on = LlamaEngine.getEnableThinking(this)
        btnThink.isChecked = on
        btnThink.alpha = 1f
    }

    private fun refreshWelcomeCard(isTextOnly: Boolean) {
        val welcomeIndex = messages.indexOfFirst { it is ChatMessage.WelcomeCard }
        if (welcomeIndex >= 0) {
            messages[welcomeIndex] = ChatMessage.WelcomeCard(isTextOnly = isTextOnly)
            chatAdapter.submitList(messages.toList())
        }
    }

    private fun loadDefaultModel() {
        val ctx = applicationContext
        val model = LlamaEngine.getSelectedModel(ctx)
        val ggufFile = File(LlamaEngine.modelPath(ctx))
        val mmprojPathStr = LlamaEngine.mmprojPath(ctx)
        val mmprojFile = mmprojPathStr?.let { File(it) }

        val ggufMissing = !ggufFile.exists()
        val mmprojMissing = !model.isTextOnly && (mmprojFile == null || !mmprojFile.exists())

        // modelsExist() also validates bundle payloads (expert layers + draft),
        // so a partially-downloaded flow-through model is treated as "not
        // downloaded" and routed to the download prompt instead of a load
        // that would fail.
        if (!LlamaEngine.modelsExist(ctx)) {
            promptDownloadModels(
                ggufMissing = ggufMissing,
                mmprojMissing = mmprojMissing
            )
            return
        }

        showLoading(true)
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val mmprojArg = if (mmprojFile != null && mmprojFile.exists()) mmprojFile.absolutePath else null
                engine.loadModel(ggufFile.absolutePath, mmprojArg)
                loadedModelId = model.id
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                engine.resetToInitialized()
                hasAutoLoaded = false
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, getString(R.string.toast_model_load_failed, e.message), Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun promptDownloadModels(ggufMissing: Boolean, mmprojMissing: Boolean) {
        val message = when {
            ggufMissing && mmprojMissing ->
                getString(R.string.download_prompt_all_missing)
            mmprojMissing ->
                getString(R.string.download_prompt_mmproj_missing)
            else ->
                getString(R.string.download_prompt_incomplete)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.download_prompt_title)
            .setMessage(message)
            .setCancelable(false)
            .setPositiveButton(R.string.go_download) { _, _ ->
                startActivity(Intent(this, ModelManagerActivity::class.java))
            }
            .setNegativeButton(R.string.later) { _, _ ->
                Toast.makeText(
                    this,
                    R.string.download_prompt_hint,
                    Toast.LENGTH_LONG
                ).show()
            }
            .show()
    }

    private val getMedia = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            handleSelectedMedia(uri)
        } else {
            pendingImageQuestion = null
        }
    }

    private fun handleSelectedMedia(uri: Uri) {
        val mime = contentResolver.getType(uri).orEmpty()
        val selected = LlamaEngine.getSelectedModel(applicationContext)
        when {
            mime.startsWith("video/") -> {
                if (!selected.supportsVideo) {
                    Toast.makeText(this, R.string.video_only_v46, Toast.LENGTH_LONG).show()
                    return
                }
                if (!ensureModelLoadedForMedia(uri)) return
                if (!engine.isVideoUnderstandingSupported) {
                    Toast.makeText(this, R.string.video_only_v46, Toast.LENGTH_LONG).show()
                    return
                }
                handleSelectedVideo(uri)
            }
            mime.startsWith("image/") || mime.isEmpty() -> {
                if (!ensureModelLoadedForMedia(uri)) return
                handleSelectedImage(uri)
            }
            else -> {
                Toast.makeText(this, getString(R.string.toast_unsupported_file, mime), Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * The picker runs before the model is loaded, so the selected file is
     * stashed and the normal inline-loading flow resumes it afterwards.
     * Returns true only when the model is ready and the caller can proceed.
     */
    private fun ensureModelLoadedForMedia(uri: Uri): Boolean {
        if (isInlineModelLoad || isInlineDownloading) {
            Toast.makeText(this, R.string.toast_please_wait_model, Toast.LENGTH_SHORT).show()
            return false
        }
        val selected = LlamaEngine.getSelectedModel(applicationContext)
        val budgetReload = LlamaEngine.consumeRuntimeReloadPending(applicationContext)
        if (isModelReady && loadedModelId == selected.id && !budgetReload) return true
        pendingMediaUri = uri
        if (!LlamaEngine.modelsExist(applicationContext, selected)) {
            showDownloadPrompt()
        } else {
            loadModelThenSend()
        }
        return false
    }

    private fun prepareImage(uri: Uri): PreparedImage {
        val startNs = System.nanoTime()
        val prompt = pendingImageQuestion ?: getString(R.string.image_only_prompt)
        pendingImageQuestion = null
        val imageData = contentResolver.openInputStream(uri)?.use { input ->
            val bitmap = BitmapFactory.decodeStream(input)
                ?: throw RuntimeException(getString(R.string.error_decode_image))
            val scaleFactor = LlamaEngine.getImageScaleFactor(applicationContext)
            val scaledWidth = (bitmap.width * scaleFactor).roundToInt().coerceAtLeast(1)
            val scaledHeight = (bitmap.height * scaleFactor).roundToInt().coerceAtLeast(1)
            val modelBitmap = if (scaledWidth == bitmap.width && scaledHeight == bitmap.height) {
                bitmap
            } else {
                Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
            }
            val stream = ByteArrayOutputStream()
            modelBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
            if (modelBitmap !== bitmap) modelBitmap.recycle()
            Triple(stream.toByteArray(), bitmap, Pair(scaledWidth, scaledHeight))
        } ?: throw RuntimeException(getString(R.string.error_read_image))

        val (bytes, bitmap, scaledSize) = imageData
        val imageInfo = getString(
            R.string.image_info_scaled,
            bitmap.width,
            bitmap.height,
            scaledSize.first,
            scaledSize.second,
            bytes.size / 1024
        )
        return PreparedImage(
            prompt = prompt,
            bytes = bytes,
            bitmap = bitmap,
            info = imageInfo,
            preparationMs = (System.nanoTime() - startNs) / 1_000_000
        )
    }

    private fun addImageUserMessage(uri: Uri, image: PreparedImage): Long {
        val msgId = messageIdCounter++
        messages.add(
            ChatMessage.UserMessage(
                id = msgId,
                text = image.prompt,
                imageBitmap = image.bitmap,
                imageInfo = image.info,
                isPrefilling = true,
                imageUri = uri.toString()
            )
        )
        chatAdapter.submitList(messages.toList()) {
            scrollToBottom()
        }
        return msgId
    }

    private fun setImagePrefilling(msgId: Long, prefilling: Boolean) {
        val index = messages.indexOfFirst { it.id == msgId }
        if (index < 0) return
        val message = messages[index] as? ChatMessage.UserMessage ?: return
        messages[index] = message.copy(isPrefilling = prefilling)
        chatAdapter.submitList(messages.toList())
    }

    private fun handleSelectedImage(
        uri: Uri,
        preparedImage: PreparedImage? = null,
        userMessageId: Long? = null
    ) {
        takePersistableReadPermission(uri)
        lifecycleScope.launch(Dispatchers.IO) {
            var msgId = userMessageId
            try {
                val image = preparedImage ?: prepareImage(uri)
                if (msgId == null) {
                    withContext(Dispatchers.Main) {
                        msgId = addImageUserMessage(uri, image)
                    }
                }

                engine.resetPerformanceStats()
                val prefillStartNs = System.nanoTime()
                engine.prefillImage(image.bytes)
                pendingImageProcessingMs = image.preparationMs +
                    (System.nanoTime() - prefillStartNs) / 1_000_000

                isImagePrefilled = true

                withContext(Dispatchers.Main) {
                    msgId?.let { setImagePrefilling(it, false) }
                }
                persistCurrentConversation()

                withContext(Dispatchers.Main) {
                    sendPrompt(image.prompt, userMessageAlreadyAdded = true)
                }
            } catch (e: Exception) {
                pendingImageProcessingMs = null
                pendingImageQuestion = null
                Log.e(TAG, "Error processing image", e)
                withContext(Dispatchers.Main) {
                    msgId?.let { setImagePrefilling(it, false) }
                    Toast.makeText(this@MainActivity, getString(R.string.toast_image_failed, e.message), Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    /**
     * Video-understanding pipeline (iOS-equivalent
     * MBHomeViewController+CaptureVideo.processVideoFrame):
     * extract up to 64 uniformly-sampled frames off the IO dispatcher,
     * append a single chat cell with the first frame as thumbnail,
     * then hand the frames to [LlamaEngine.prefillVideoFrames] which
     * loops `prefillImage(...)` under a temporary slice=1 cap.
     *
     * Gated to MiniCPM-V-4.6 because that's where iOS enables the
     * feature and where the native nCtx bump to 8192 takes effect
     * (see prepare() in llama_jni.cpp).
     */
    private fun handleSelectedVideo(uri: Uri) {
        if (!engine.isVideoUnderstandingSupported) {
            Toast.makeText(this,
                R.string.video_only_v46,
                Toast.LENGTH_LONG).show()
            return
        }

        isProcessingVideo = true
        takePersistableReadPermission(uri)
        lifecycleScope.launch(Dispatchers.IO) {
            val msgId = messageIdCounter++
            val startNs = System.nanoTime()
            try {
                val scaleFactor = LlamaEngine.getImageScaleFactor(applicationContext)
                val extracted = VideoFrameExtractor.extract(applicationContext, uri, scaleFactor)
                val info = VideoFrameExtractor.formatVideoInfo(applicationContext, extracted)
                Log.i(TAG, "Video info: $info")

                withContext(Dispatchers.Main) {
                    val videoMessage = ChatMessage.UserMessage(
                        id = msgId,
                        text = "",
                        imageBitmap = extracted.thumbnail,
                        imageInfo = info,
                        isPrefilling = true,
                        isVideo = true,
                        imageUri = uri.toString()
                    )
                    messages.add(videoMessage)
                    chatAdapter.submitList(messages.toList()) {
                        scrollToBottom()
                    }
                }

                engine.resetPerformanceStats()
                engine.prefillVideoFrames(extracted.frames) { current, total ->
                    withContext(Dispatchers.Main) {
                        val index = messages.indexOfFirst { it.id == msgId }
                        if (index >= 0) {
                            val cur = messages[index] as ChatMessage.UserMessage
                            messages[index] = cur.copy(
                                imageInfo = getString(R.string.video_processing_progress, info, current, total)
                            )
                            chatAdapter.submitList(messages.toList())
                        }
                    }
                }

                isImagePrefilled = true

                val elapsedMs = (System.nanoTime() - startNs) / 1_000_000
                pendingImageProcessingMs = elapsedMs
                withContext(Dispatchers.Main) {
                    isProcessingVideo = false
                    val index = messages.indexOfFirst { it.id == msgId }
                    if (index >= 0) {
                        val cur = messages[index] as ChatMessage.UserMessage
                        messages[index] = cur.copy(
                            imageInfo = getString(R.string.video_preprocessing_done, info, elapsedMs / 1000.0),
                            isPrefilling = false
                        )
                        chatAdapter.submitList(messages.toList())
                    }
                }
                persistCurrentConversation()
            } catch (e: Exception) {
                pendingImageProcessingMs = null
                Log.e(TAG, "Error processing video", e)
                withContext(Dispatchers.Main) {
                    isProcessingVideo = false
                    val index = messages.indexOfFirst { it.id == msgId }
                    if (index >= 0) {
                        messages.removeAt(index)
                        chatAdapter.submitList(messages.toList())
                    }
                    Toast.makeText(this@MainActivity, getString(R.string.toast_video_failed, e.message), Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun getFileName(uri: Uri): String {
        val cursor = contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val nameIndex = it.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0) {
                    return it.getString(nameIndex)
                }
            }
        }
        return "file-${System.currentTimeMillis()}"
    }

    private fun handleUserInput() {
        val userMsg = etInput.text.toString().trim()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, R.string.toast_empty_input, Toast.LENGTH_SHORT).show()
            return
        }

        val selected = LlamaEngine.getSelectedModel(applicationContext)
        // A changed memory budget is only consumed while loading the model, so
        // force one reload on the next request instead of reusing the old one.
        val budgetReload = LlamaEngine.consumeRuntimeReloadPending(applicationContext)
        if (isModelReady && loadedModelId == selected.id && !budgetReload) {
            sendPrompt(userMsg)
            return
        }

        beginModelFlow(userMsg)
    }

    /**
     * First request before any model is loaded: make sure the files are present
     * (download inline in the chat stream if not) and then load the model, all
     * without leaving the home screen and without any dialog.
     */
    private fun beginModelFlow(userMsg: String) {
        if (isInlineDownloading || isInlineModelLoad) {
            Toast.makeText(this, R.string.toast_please_wait_model, Toast.LENGTH_SHORT).show()
            return
        }
        pendingPrompt = userMsg
        pendingMediaUri = null
        etInput.text = null
        etInput.clearFocus()
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager)
            .hideSoftInputFromWindow(etInput.windowToken, 0)
        collapseAppBar()

        val model = LlamaEngine.getSelectedModel(applicationContext)
        if (!LlamaEngine.modelsExist(applicationContext, model)) {
            showDownloadPrompt()
        } else {
            loadModelThenSend()
        }
    }

    private fun showDownloadPrompt() {
        removeDownloadPrompt()
        val model = LlamaEngine.getSelectedModel(applicationContext)
        val id = messageIdCounter++
        downloadPromptId = id
        downloadModelId = model.id
        messages.add(ChatMessage.ModelDownloadPrompt(id = id, modelName = model.displayName))
        chatAdapter.submitList(messages.toList()) {
            scrollToBottom()
        }
    }

    private fun updateDownloadPrompt(
        downloading: Boolean,
        fraction: Float = 0f,
        status: String? = null,
        paused: Boolean = false
    ) {
        val index = messages.indexOfFirst { it.id == downloadPromptId }
        if (index < 0) return
        val current = messages[index] as? ChatMessage.ModelDownloadPrompt ?: return
        messages[index] = current.copy(
            downloading = downloading,
            paused = paused,
            fraction = fraction,
            status = status
        )
        chatAdapter.submitList(messages.toList()) {
            scrollToBottom()
        }
    }

    private fun removeDownloadPrompt() {
        if (downloadPromptId < 0) return
        chatAdapter.clearActiveDownload()
        removeMessage(downloadPromptId)
        downloadPromptId = -1L
        downloadModelId = null
    }

    private fun cancelDownloadPrompt() {
        if (isInlineDownloading) return
        removeDownloadPrompt()
        pendingPrompt?.let { etInput.setText(it) }
        pendingPrompt = null
        pendingMediaUri = null
        pendingImageQuestion = null
        enableInput(true)
        etInput.requestFocus()
    }

    private fun startInlineDownload() {
        if (downloadPromptId < 0) return
        val modelId = downloadModelId ?: LlamaEngine.getSelectedModel(applicationContext).id
        if (ModelDownloadController.isRunning && ModelDownloadController.runningModelId() != modelId) {
            Toast.makeText(this, R.string.toast_downloading, Toast.LENGTH_SHORT).show()
            return
        }
        isInlineDownloading = true
        updateDownloadPrompt(
            downloading = true,
            fraction = 0f,
            status = getString(R.string.download_preparing),
            paused = false
        )
        chatAdapter.setActiveDownload(downloadPromptId)
        enableInput(false)
        ModelDownloadService.start(this, modelId)
    }

    private fun resumeInlineDownload() {
        val modelId = downloadModelId ?: return
        if (ModelDownloadController.isRunning) return
        isInlineDownloading = true
        updateDownloadPrompt(
            downloading = true,
            fraction = 0f,
            status = getString(R.string.download_preparing),
            paused = false
        )
        chatAdapter.setActiveDownload(downloadPromptId)
        enableInput(false)
        ModelDownloadService.start(this, modelId)
    }

    /**
     * Mirrors the shared [ModelDownloadController] into the inline chat card.
     * Downloads started elsewhere (model manager) are ignored here because
     * [downloadPromptId] is only set while a card exists.
     */
    private fun observeDownloads() {
        lifecycleScope.launch {
            ModelDownloadController.status.collect { status ->
                if (downloadPromptId < 0) return@collect
                val modelId = downloadModelId ?: return@collect
                when (status) {
                    is ModelDownloadController.Status.Idle -> Unit
                    is ModelDownloadController.Status.Running -> {
                        if (status.modelId == modelId) {
                            if (!isInlineDownloading) {
                                isInlineDownloading = true
                                chatAdapter.setActiveDownload(downloadPromptId)
                            }
                            chatAdapter.updateDownloadProgress(
                                downloadPromptId,
                                status.fraction,
                                status.message
                            )
                        }
                    }
                    is ModelDownloadController.Status.Paused -> {
                        if (status.modelId == modelId) {
                            isInlineDownloading = false
                            updateDownloadPrompt(
                                downloading = true,
                                paused = true,
                                fraction = status.fraction,
                                status = getString(
                                    R.string.model_paused_progress,
                                    (status.fraction * 100f).toInt().coerceIn(0, 100)
                                )
                            )
                            Toast.makeText(this@MainActivity, R.string.download_paused, Toast.LENGTH_SHORT).show()
                            ModelDownloadController.acknowledge()
                        }
                    }
                    is ModelDownloadController.Status.Completed -> {
                        if (status.modelId == modelId) {
                            isInlineDownloading = false
                            removeDownloadPrompt()
                            ModelDownloadController.acknowledge()
                            loadModelThenSend()
                        }
                    }
                    is ModelDownloadController.Status.Failed -> {
                        if (status.modelId == modelId) {
                            isInlineDownloading = false
                            updateDownloadPrompt(downloading = false)
                            Toast.makeText(
                                this@MainActivity,
                                getString(R.string.download_failed, status.message),
                                Toast.LENGTH_LONG
                            ).show()
                            pendingPrompt?.let { etInput.setText(it) }
                            pendingPrompt = null
                            pendingMediaUri = null
                            pendingImageQuestion = null
                            enableInput(true)
                            ModelDownloadController.acknowledge()
                        }
                    }
                    is ModelDownloadController.Status.Cancelled -> {
                        if (status.modelId == modelId) {
                            isInlineDownloading = false
                            updateDownloadPrompt(downloading = false)
                            enableInput(true)
                            ModelDownloadController.acknowledge()
                        }
                    }
                }
            }
        }
    }

    private fun loadModelThenSend() {
        val pendingText = pendingPrompt
        val mediaUri = pendingMediaUri
        if (pendingText == null && mediaUri == null) return
        if (isInlineModelLoad) return
        autoFollowStreaming = true
        isInlineModelLoad = true
        enableInput(false)
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager)
            .hideSoftInputFromWindow(etInput.windowToken, 0)

        val ggufFile = File(LlamaEngine.modelPath(applicationContext))
        val mmprojFile = LlamaEngine.mmprojPath(applicationContext)?.let { File(it) }

        lifecycleScope.launch(Dispatchers.IO) {
            var placeholderId = -1L
            var imageMessageId: Long? = null
            try {
                val preparedImage = mediaUri?.let { prepareImage(it) }

                // Show the selected image/question before model loading starts.
                // The loading placeholder follows the user's turn so the chat
                // order matches the already-loaded path.
                withContext(Dispatchers.Main) {
                    when {
                        pendingText != null -> {
                            messages.add(ChatMessage.UserMessage(id = messageIdCounter++, text = pendingText))
                        }
                        mediaUri != null && preparedImage != null -> {
                            imageMessageId = addImageUserMessage(mediaUri, preparedImage)
                        }
                    }
                    placeholderId = messageIdCounter++
                    messages.add(
                        ChatMessage.AiMessage(
                            id = placeholderId,
                            text = "",
                            isGenerating = true,
                            isLoadingPlaceholder = true
                        )
                    )
                    chatAdapter.setActiveAiMessage(placeholderId)
                    chatAdapter.submitList(messages.toList()) {
                        scrollToBottom()
                    }
                }

                // A different model may still be loaded (selection is lazy), so
                // drop it before loading the newly selected one.
                if (isModelReady) {
                    runCatching { engine.unloadModel() }
                    isModelReady = false
                    loadedModelId = null
                }
                val mmprojArg = mmprojFile?.takeIf { it.exists() }?.absolutePath
                engine.loadModel(ggufFile.absolutePath, mmprojArg)

                // The lazy (re)load starts from an empty context. Replay the
                // saved turns so follow-up messages keep the conversation
                // context instead of silently forgetting it. Media also needs
                // the history replayed before its image/audio prefill.
                if (pendingText != null || mediaUri != null) {
                    restoreHistoryForCurrentConversation()
                }

                withContext(Dispatchers.Main) {
                    isInlineModelLoad = false
                    chatAdapter.clearActiveAiMessage()
                    if (placeholderId >= 0) removeMessage(placeholderId)
                    isModelReady = true
                    loadedModelId = LlamaEngine.getSelectedModel(applicationContext).id
                    updateUIForModelType()
                    pendingPrompt = null
                    enableInput(true)
                    if (mediaUri != null && preparedImage != null) {
                        pendingMediaUri = null
                        handleSelectedImage(mediaUri, preparedImage, imageMessageId)
                    } else if (pendingText != null) {
                        sendPrompt(pendingText, userMessageAlreadyAdded = true)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Inline model load failed", e)
                withContext(Dispatchers.Main) {
                    isInlineModelLoad = false
                    chatAdapter.clearActiveAiMessage()
                    if (placeholderId >= 0) removeMessage(placeholderId)
                    imageMessageId?.let { setImagePrefilling(it, false) }
                    runCatching { engine.resetToInitialized() }
                    isModelReady = false
                    pendingPrompt?.let { etInput.setText(it) }
                    pendingPrompt = null
                    pendingMediaUri = null
                    pendingImageQuestion = null
                    enableInput(true)
                    Toast.makeText(
                        this@MainActivity,
                        getString(R.string.toast_model_load_failed, e.message),
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private suspend fun restoreHistoryForCurrentConversation() {
        val id = currentConversationId ?: return
        try {
            val turns = ConversationStore.toHistoryTurns(ConversationStore.load(applicationContext, id))
            if (turns.isEmpty()) return
            engine.clearContext()
            engine.restoreConversation(turns)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to replay conversation history", e)
        }
    }

    private fun removeMessage(id: Long) {
        val index = messages.indexOfFirst { it.id == id }
        if (index >= 0) {
            messages.removeAt(index)
            chatAdapter.submitList(messages.toList())
        }
    }

    private fun sendPrompt(userMsg: String, userMessageAlreadyAdded: Boolean = false) {
        autoFollowStreaming = true
        etInput.clearFocus()
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager)
            .hideSoftInputFromWindow(etInput.windowToken, 0)

        etInput.text = null
        enableInput(false)

        collapseAppBar()

        if (!userMessageAlreadyAdded) {
            val msgId = messageIdCounter++
            val userMessage = ChatMessage.UserMessage(
                id = msgId,
                text = userMsg,
                imageBitmap = null,
                imageInfo = null
            )
            messages.add(userMessage)
            chatAdapter.submitList(messages.toList()) {
                scrollToBottom()
            }
        }
        // Persist immediately so the new conversation enters the history list
        // (and the user's turn survives an interruption).
        persistCurrentConversation()

        val imageProcessingMs = if (isImagePrefilled) pendingImageProcessingMs else null
        val resetStatsForTextOnlyTurn = !isImagePrefilled
        isImagePrefilled = false
        pendingImageProcessingMs = null

        val aiMsgId = messageIdCounter++
        val aiMessage = ChatMessage.AiMessage(id = aiMsgId, text = "", isGenerating = true)
        messages.add(aiMessage)
        chatAdapter.setActiveAiMessage(aiMsgId)
        chatAdapter.submitList(messages.toList()) {
            scrollToBottom()
        }
        persistCurrentConversation()

        generationJob = lifecycleScope.launch(Dispatchers.Default) {
            val inferenceStartNs = System.nanoTime()
            if (resetStatsForTextOnlyTurn) engine.resetPerformanceStats()
            val fullResponse = StringBuilder()
            engine.sendUserPrompt(userMsg)
                .onCompletion { cause ->
                    withContext(NonCancellable) {
                        val (prefillRate, decodeRate) = runCatching {
                            engine.readPerformanceRates()
                        }.getOrDefault(null to null)
                        val visionEncodingMs = if (imageProcessingMs != null) {
                            runCatching { engine.readVisionEncodingMs() }.getOrNull()
                        } else null
                        val totalMs = imageProcessingMs?.plus((System.nanoTime() - inferenceStartNs) / 1_000_000)
                            ?: ((System.nanoTime() - inferenceStartNs) / 1_000_000)
                        val metrics = if (cause == null) InferenceMetrics(
                            imageProcessingMs = visionEncodingMs ?: imageProcessingMs,
                            prefillTokensPerSecond = prefillRate,
                            decodeTokensPerSecond = decodeRate,
                            totalMs = totalMs
                        ) else null
                        val contextTruncated = cause == null &&
                            runCatching { engine.wasLastGenerationTruncated() }.getOrDefault(false)
                        withContext(Dispatchers.Main) {
                            val index = messages.indexOfFirst { it.id == aiMsgId }
                            if (index >= 0) {
                                messages[index] = (messages[index] as ChatMessage.AiMessage).copy(
                                    text = fullResponse.toString(),
                                    isGenerating = false,
                                    metrics = metrics
                                )
                            }
                            chatAdapter.setGeneratingDone(aiMsgId)
                            chatAdapter.clearActiveAiMessage()
                            chatAdapter.submitList(messages.toList())
                            enableInput(true)
                            followStreamingIfAtBottom()
                            if (contextTruncated) {
                                Toast.makeText(
                                    this@MainActivity,
                                    R.string.context_limit_reached,
                                    Toast.LENGTH_LONG
                                ).show()
                            }
                        }
                        persistCurrentConversation()
                        withContext(Dispatchers.Main) { refreshConversationList() }
                    }
                }
                .catch { error ->
                    withContext(Dispatchers.Main) {
                        val message = error.message.orEmpty()
                        val text = if (message.contains("context size", ignoreCase = true) ||
                                      message.contains("context capacity", ignoreCase = true)) {
                            getString(R.string.context_limit_reached)
                        } else {
                            message.ifBlank { getString(android.R.string.dialog_alert_title) }
                        }
                        Toast.makeText(this@MainActivity, text, Toast.LENGTH_LONG).show()
                    }
                }
                .collect { token ->
                    fullResponse.append(token)
                    withContext(Dispatchers.Main) {
                        val currentText = fullResponse.toString()
                        val index = messages.indexOfFirst { it.id == aiMsgId }
                        if (index >= 0) {
                            messages[index] = ChatMessage.AiMessage(
                                id = aiMsgId,
                                text = currentText,
                                isGenerating = true
                            )
                        }
                        chatAdapter.updateStreamingText(aiMsgId, currentText)
                        followStreamingIfAtBottom()
                    }
                    persistDuringStreaming()
                }
        }
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.action == MotionEvent.ACTION_DOWN) {
            val v = currentFocus
            if (v is TextInputEditText) {
                val barRect = android.graphics.Rect()
                cardInputBar.getGlobalVisibleRect(barRect)
                if (!barRect.contains(ev.rawX.toInt(), ev.rawY.toInt())) {
                    v.clearFocus()
                    val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
                    imm.hideSoftInputFromWindow(v.windowToken, 0)
                }
            }
        }
        return super.dispatchTouchEvent(ev)
    }

    override fun onResume() {
        super.onResume()
        val currentTag = LocaleManager.currentLanguage(this).tag
        if (createdWithLocale != null && createdWithLocale != currentTag) {
            isLocaleRestart = true
            LocaleManager.recreateSeamlessly(this)
            return
        }
        // Re-check: if the model was switched to a TTS model while this
        // activity was in the background, redirect to TtsActivity.
        if (shouldRedirectToTts()) {
            startActivity(Intent(this, TtsActivity::class.java))
            finish()
            return
        }
        if (shouldRedirectToImage()) {
            startActivity(Intent(this, ImageActivity::class.java))
            finish()
            return
        }
        if (!::engine.isInitialized) return
        // Selection is lazy: a model switch (here or on the model manager page)
        // only updates the title/selection.  The engine keeps whatever model it
        // currently has until the next request, which then unloads + loads the
        // selected model.
        if (LlamaEngine.consumeModelSwitched(applicationContext)) {
            loadedModelId = LlamaEngine.getSelectedModel(applicationContext).id
            clearChatUI()
            updateUIForModelType()
            pendingRestore = currentConversationId != null
            if (engine.state.value is LlamaState.ModelReady) restoreCurrentConversationHistory()
        }
    }

    private fun reloadAfterModelSwitch() {
        enableInput(false)
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                if (engine.state.value is LlamaState.ModelReady) {
                    engine.unloadModel()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error unloading during model switch", e)
            }
            withContext(Dispatchers.Main) {
                clearChatUI()
                pendingRestore = currentConversationId != null
                loadDefaultModel()
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        currentConversationId?.let { outState.putString(STATE_CURRENT_CONVERSATION, it) }
        super.onSaveInstanceState(outState)
    }

    override fun onStop() {
        generationJob?.cancel()
        persistCurrentConversation()
        super.onStop()
    }

    override fun onDestroy() {
        if (isFinishing && !isLocaleRestart && ::engine.isInitialized) {
            engine.destroy()
        }
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val STREAM_PERSIST_INTERVAL_MS = 1000L
        private const val STATE_CURRENT_CONVERSATION = "state_current_conversation"
    }
}
