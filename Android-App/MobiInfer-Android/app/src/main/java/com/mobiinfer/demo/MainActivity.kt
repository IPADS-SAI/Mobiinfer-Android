package com.mobiinfer.demo

import android.content.ClipData
import android.content.ClipboardManager
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
import com.mobiinfer.demo.imagegen.ImageTool
import com.mobiinfer.demo.imagegen.MemoryBudget
import com.mobiinfer.demo.imagegen.SystemPrompts
import io.noties.markwon.Markwon
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.SupervisorJob
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

    // Lazily created: the image stack (libimagegen + its own ggml) is only
    // loaded the first time the user actually generates an image.
    private var imageEngine: ImageEngine? = null
    private var imageJob: Job? = null
    // Image work must not die with the Activity: leaving the screen (settings,
    // rotation fallback) used to cancel the job and report a failure even
    // though the native draw was still running. See onStop().
    private val imageWorkScope: CoroutineScope =
        CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    // Scope that owns the image flow started from a tool call; independent of
    // generationJob (cancelled before the image work starts).
    // Bumped for every image request so a late result from an abandoned or
    // superseded request cannot overwrite a newer one.
    private var imageGenerationToken = 0L
    private var isImageModelLoaded = false
    private var loadedImageModelId: String? = null
    // Whether the resident chat model was loaded with the generate_image tool
    // in its system prompt; null while no model is loaded.
    private var lastSystemPromptHadImageTool: Boolean? = null
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
        chatAdapter.setOnImageClick { path -> ImageViewer.show(this, path) }
        chatAdapter.setOnImageLongClick { path -> ImageViewer.saveToGallery(this, path) }
        chatAdapter.setOnImageErrorClick { message ->
            val clipboard = getSystemService(ClipboardManager::class.java)
            clipboard?.setPrimaryClip(ClipData.newPlainText("MobiInfer image error", message))
            Toast.makeText(this, R.string.image_error_copied, Toast.LENGTH_SHORT).show()
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
        // An image request belongs to the conversation that started it (or is
        // superseded by a newer one); never let it render into another chat.
        // Bumping the token once is enough: it invalidates every callback the
        // old draw could still deliver, whatever thread it lands on.
        imageJob?.cancel()
        imageJob = null
        imageGenerationToken++
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
            messages.add(
                ChatMessage.WelcomeCard(
                    isTextOnly = selectedModel.isTextOnly,
                    isImageMode = selectedModel.isImage && isImageMode()
                )
            )
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

    /**
     * Persist and wait for the write. Used before maintenance that inspects
     * the saved files (image pruning, history replay).
     */
    private fun persistCurrentConversationBlocking() {
        val id = currentConversationId ?: return
        val snapshot = ArrayList<ChatMessage>(messages.size).apply { addAll(messages) }
        runCatching {
            kotlinx.coroutines.runBlocking(Dispatchers.IO) {
                ConversationStore.save(applicationContext, id, snapshot)
            }
        }.onFailure { Log.w(TAG, "Synchronous conversation save failed", it) }
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

    /**
     * Rotation restarts the Activity; the image draw lives in the Activity's
     * scope and in a single native context, so a recreation would lose it.
     * Pin the screen to its current orientation for the duration instead.
     */
    private fun lockOrientationWhileGenerating(lock: Boolean) {
        requestedOrientation = if (lock) {
            if (resources.configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE) {
                android.content.pm.ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
            } else {
                android.content.pm.ActivityInfo.SCREEN_ORIENTATION_SENSOR_PORTRAIT
            }
        } else {
            android.content.pm.ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        }
    }

    private fun collapseAppBar() {
        appBarLayout.setExpanded(false, true)
    }

    /**
     * System message the chat model is loaded with. The generate_image tool is
     * described here; switching it off in the config page changes the text
     * and reloads the model (see onResume).
     */
    /**
     * Applies the chat system prompt to the freshly loaded model. The tool
     * schema has to be declared first: a tool-aware template (Qwen et al.)
     * only renders tools that were passed to common_chat_templates_apply(),
     * and that happens while the system prompt is formatted. Templates that
     * do not know about tools get them as text in the prompt instead.
     */
    private suspend fun applyChatSystemPrompt() {
        val toolEnabled = LlamaEngine.isImageToolEnabled(applicationContext)
        val template = engine.chatTemplate()
        val templateKnowsTools = SystemPrompts.templateSupportsTools(template)
        engine.setToolSchemas(
            if (templateKnowsTools) SystemPrompts.toolSchemasJson(toolEnabled) else null
        )
        DebugLogBridge.dump(
            "system-prompt",
            "templateKnowsTools=$templateKnowsTools toolEnabled=$toolEnabled\n--- template (head) ---\n" +
                template?.take(600)
        )
        engine.setSystemPrompt(chatSystemPrompt())
    }

    private fun chatSystemPrompt(): String {
        val ctx = applicationContext
        val locale = LocaleManager.currentLanguage(ctx).tag
        val toolEnabled = LlamaEngine.isImageToolEnabled(ctx)
        lastSystemPromptHadImageTool = toolEnabled
        val prompt = SystemPrompts.systemMessage(
            isChinese = locale.startsWith("zh"),
            toolEnabled = toolEnabled,
            defaultWidth = LlamaEngine.getImageGenWidth(ctx),
            defaultHeight = LlamaEngine.getImageGenHeight(ctx),
            defaultSteps = LlamaEngine.getImageGenSteps(ctx),
            defaultCfg = LlamaEngine.getImageGenCfg(ctx)
        )
        // The system prompt is the model's only source of truth about the
        // image tool; "it did not call the tool" is almost always this text
        // not having reached it, so keep a copy in the debug log.
        DebugLogBridge.dump("system-prompt", prompt)
        return prompt
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
        messages.add(
            ChatMessage.WelcomeCard(
                isTextOnly = selectedModel.isTextOnly,
                isImageMode = selectedModel.isImage && isImageMode()
            )
        )
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
        refreshImageModeUi()
    }

    private fun shouldRedirectToTts(): Boolean {
        val model = LlamaEngine.getSelectedModel(applicationContext)
        return model.isTts
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
        // A model switch may cross the chat/image boundary (or pick a different
        // image checkpoint): drop the stale image engine and re-sync the UI.
        if (loadedImageModelId != null && loadedImageModelId != model.id) {
            releaseImageEngine()
        }
        refreshImageModeUi()
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
            updateUIForModelType()
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

    // ------------------------------------------------------------------
    // Text-to-image (an image checkpoint is the selected model)
    // ------------------------------------------------------------------

    /** True when the selected model draws pictures instead of chatting. */
    private fun isImageMode(): Boolean =
        LlamaEngine.getSelectedModel(applicationContext).isImage

    /**
     * Re-syncs everything whose answer depends on the selected model: the
     * attachment / deep-thinking buttons, the input hint and, when the model
     * is a text-to-image checkpoint, the image prompt placeholder.
     *
     * There is deliberately no separate "text to image" switch: an image
     * checkpoint turns the input into an image prompt by itself, and a chat
     * model reaches the same engine through the generate_image tool.
     */
    private fun refreshImageModeUi() {
        val model = LlamaEngine.getSelectedModel(applicationContext)
        val imageModel = model.isImage
        if (imageModel) {
            // sdcpp takes text only and has no reasoning channel, but the deep
            // thinking button belongs to the chat flow: leave both of its
            // visibility rules alone (they follow model support) so the button
            // never disappears just because a picture model is selected.
            btnImage.visibility = View.GONE
            btnImage.isEnabled = false
            etInput.setHint(R.string.image_prompt_hint)
            etInput.contentDescription = getString(R.string.image_prompt_hint)
        } else {
            etInput.setHint(R.string.hint_input)
            etInput.contentDescription = getString(R.string.hint_input)
        }
        refreshWelcomeCard(model.isTextOnly)
    }

    /**
     * One request to the image engine: which prompt, with what parameters, and
     * where the result goes. Produced either by the image-mode switch
     * ([beginImageGeneration]) or by the language model calling the
     * generate_image tool ([runImageGenerationForTool]).
     */
    private data class ImageRequest(
        val prompt: String,
        val width: Int,
        val height: Int,
        val steps: Int,
        val cfg: Float,
        val negativePrompt: String,
        val seed: Long,
        val model: ModelInfo
    )

    private fun currentImageRequest(prompt: String, model: ModelInfo): ImageRequest {
        val ctx = applicationContext
        return ImageRequest(
            prompt = prompt,
            model = model,
            width = LlamaEngine.getImageGenWidth(ctx),
            height = LlamaEngine.getImageGenHeight(ctx),
            steps = LlamaEngine.getImageGenSteps(ctx),
            cfg = LlamaEngine.getImageGenCfg(ctx),
            negativePrompt = LlamaEngine.getImageNegativePrompt(ctx),
            seed = LlamaEngine.getImageGenSeed(ctx)
        )
    }

    private fun releaseImageEngine() {
        val destroyed = imageEngine
        imageEngine = null
        isImageModelLoaded = false
        loadedImageModelId = null
        destroyed?.destroy()
        destroyed?.shutdownThread()
    }

    /**
     * Makes sure libimagegen has [imageModel] loaded, downloading it first if
     * necessary, showing progress on the message [statusMessageId] and giving
     * up if that message leaves the screen.
     *
     * The image stack is completely separate from the llama one (its own
     * ggml), so the chat model is only unloaded when it does not fit in
     * memory next to the image stack.
     */
    // Why the last prepareImageEngine() returned null, for the error card.
    @Volatile
    private var lastImagePreparationError: String? = null

    private suspend fun prepareImageEngine(
        imageModel: ModelInfo,
        statusMessageId: Long
    ): ImageEngine? {
        lastImagePreparationError = null
        val ctx = applicationContext
        cancelActiveGeneration()

        if (!LlamaEngine.modelsExist(ctx, imageModel)) {
            updateAiStatus(
                statusMessageId,
                getString(R.string.tool_download_image_model, imageModel.displayName)
            )
            if (!downloadImageModelInline(imageModel, statusMessageId)) {
                lastImagePreparationError = getString(R.string.image_model_download_failed)
                return null
            }
        }

        val engine = imageEngine ?: ImageEngine.getInstance(ctx).also { imageEngine = it }
        // Reuse only a context that is still healthy: after a failed draw the
        // native side may have left the model in a bad state, and reusing it
        // made every following request in the same conversation fail too.
        val reusable = isImageModelLoaded &&
            loadedImageModelId == imageModel.id &&
            engine.isLoaded &&
            engine.state.value !is ImageState.Error
        if (reusable) return engine
        if (isImageModelLoaded && engine.isLoaded &&
            loadedImageModelId == imageModel.id && engine.state.value is ImageState.Error
        ) {
            CriticalLog.log("image", "previous draw left the context in an error state: reloading")
        }

        if (isImageModelLoaded) {
            // Different checkpoint: free the old one before loading the new.
            engine.destroyIfLoaded()
            isImageModelLoaded = false
            loadedImageModelId = null
        }

        if (shouldUnloadChatModelFor(imageModel)) {
            updateAiStatus(statusMessageId, getString(R.string.tool_unloading_model))
            unloadChatModelForImage()
        }

        CriticalLog.log(
            "image",
            "loading ${imageModel.id}: native heap ${nativeHeapMb()} MB, " +
                "free ${MemoryBudget.availableBytes(ctx) / (1024 * 1024)} MB"
        )
        val loadStart = SystemClock.elapsedRealtime()
        val ok = withContext(Dispatchers.IO) {
            engine.loadModel(
                diffusionModelPath = LlamaEngine.modelPath(ctx, imageModel),
                llmPath = LlamaEngine.imageTextEncoderPath(ctx, imageModel),
                vaePath = LlamaEngine.imageVaePath(ctx, imageModel),
                tokenizerPath = LlamaEngine.imageTokenizerPath(ctx, imageModel),
                nThreads = 0,
                backend = "diffusion=HTP0,te=cpu,vae=HTP0"
            )
        }
        CriticalLog.log(
            "image",
            "load ${if (ok) "OK" else "FAILED"} in ${SystemClock.elapsedRealtime() - loadStart} ms, " +
                "native heap ${nativeHeapMb()} MB" + (engine.lastError?.let { ", error=$it" } ?: "")
        )
        if (ok) {
            isImageModelLoaded = true
            loadedImageModelId = imageModel.id
        } else {
            Log.e(TAG, "Image model load failed: ${engine.lastError}")
            lastImagePreparationError = engine.lastError ?: getString(R.string.image_load_failed)
        }
        return if (ok) engine else null
    }

    /**
     * Loads [imageModel] through the regular download service and mirrors its
     * progress into the chat until it finishes. Returns false when the user
     * dismissed the card or the download failed.
     */
    private suspend fun downloadImageModelInline(imageModel: ModelInfo, statusMessageId: Long): Boolean {
        if (ModelDownloadController.isRunning) {
            updateAiStatus(statusMessageId, getString(R.string.toast_downloading))
            return false
        }
        ModelDownloadService.start(applicationContext, imageModel.id)
        val outcome = try {
            var terminal = DownloadOutcome.Progress
            ModelDownloadController.status
                .filter { status -> status.modelIdOrNull() == imageModel.id }
                .takeWhile { status ->
                    if (status is ModelDownloadController.Status.Running) {
                        val pct = (status.fraction * 100f).toInt().coerceIn(0, 100)
                        updateAiStatus(
                            statusMessageId,
                            "${getString(R.string.tool_download_image_model_short, imageModel.displayName)} $pct%"
                        )
                    }
                    terminal = status.toDownloadOutcome()
                    terminal == DownloadOutcome.Progress
                }
                .collect { }
            terminal
        } catch (e: kotlinx.coroutines.CancellationException) {
            cancelImageModelDownload(imageModel)
            throw e
        }
        return outcome == DownloadOutcome.Done
    }

    private enum class DownloadOutcome { Progress, Done, Aborted }

    private fun ModelDownloadController.Status.modelIdOrNull(): String? = when (this) {
        is ModelDownloadController.Status.Idle -> null
        is ModelDownloadController.Status.Running -> modelId
        is ModelDownloadController.Status.Paused -> modelId
        is ModelDownloadController.Status.Completed -> modelId
        is ModelDownloadController.Status.Cancelled -> modelId
        is ModelDownloadController.Status.Failed -> modelId
    }

    private fun ModelDownloadController.Status.toDownloadOutcome(): DownloadOutcome = when (this) {
        is ModelDownloadController.Status.Completed -> {
            ModelDownloadController.acknowledge()
            DownloadOutcome.Done
        }
        is ModelDownloadController.Status.Failed,
        is ModelDownloadController.Status.Cancelled -> {
            ModelDownloadController.acknowledge()
            DownloadOutcome.Aborted
        }
        else -> DownloadOutcome.Progress
    }

    private fun cancelImageModelDownload(imageModel: ModelInfo) {
        if (ModelDownloadController.runningModelId() == imageModel.id) {
            ModelDownloadService.pause(applicationContext)
        }
    }

    /**
     * Decides whether the chat model has to go before the image stack loads.
     * Uses the configured runtime budget as the chat model's share, so a
     * large MoE on a 12 GB phone releases memory and a small model on a
     * 24 GB phone keeps running for the follow-up question.
     */
    private fun shouldUnloadChatModelFor(imageModel: ModelInfo): Boolean {
        if (!isModelReady) return false
        val footprint = imageStackBytes(imageModel)
        val budgetMb = LlamaEngine.getMemoryBudgetMb(applicationContext)
        val fits = MemoryBudget.fitsAlongsideLanguageModel(applicationContext, footprint, budgetMb)
        // Logged both ways on purpose: "it unloaded my model" and "it did not
        // fit" are the two questions every memory report starts with.
        Log.i(
            TAG,
            "image stack ${footprint / (1024 * 1024)} MB vs free " +
                "${MemoryBudget.availableBytes(applicationContext) / (1024 * 1024)} MB " +
                "(chat budget ${budgetMb} MB) -> " + if (fits) "fits, keep chat model" else "unload chat model"
        )
        return !fits
    }

    /**
     * DiT + text encoder + VAE as they sit on disk: an upper bound for what
     * sdcpp maps into RAM. Falls back to the model's declared total when the
     * files are not all present yet.
     */
    private fun imageStackBytes(imageModel: ModelInfo): Long {
        val files = listOf(imageModel.ggufFileName) + imageModel.extraFiles
        val onDisk = files.sumOf { modelFileSize(imageModel, it) }
        return if (onDisk > 0) onDisk else imageModel.totalBytes
    }

    private fun modelFileSize(model: ModelInfo, fileName: String): Long =
        File(LlamaEngine.modelDirFor(applicationContext, model), fileName).length()

    private suspend fun unloadChatModelForImage() {
        val before = nativeHeapMb()
        Log.i(TAG, "Unloading the chat model for the image stack (native heap ${before} MB)")
        CriticalLog.log("image", "unload chat model: native heap before = ${before} MB")
        try {
            engine.cancelGeneration()
            engine.unloadModel()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to unload the chat model for the image stack", e)
            CriticalLog.log("image", "unload FAILED: ${e.javaClass.simpleName}: ${e.message}")
        }
        isModelReady = false
        loadedModelId = null
        // Give the allocator a chance to hand the freed pages back before a
        // multi-GB image stack asks for them; the Kotlin heap is small but its
        // native counterparts (mmap, ggml buffers) are what matters.
        runCatching {
            System.gc()
            Runtime.getRuntime().gc()
        }
        val after = nativeHeapMb()
        Log.i(TAG, "Chat model unloaded: native heap ${before} -> ${after} MB")
        CriticalLog.log("image", "unload chat model: native heap after = ${after} MB")
    }

    private fun nativeHeapMb(): Long = android.os.Debug.getNativeHeapAllocatedSize() / (1024 * 1024)

    /** Home-screen image mode entry point (the switch is on). */
    private fun beginImageGeneration(prompt: String) {
        val ctx = applicationContext
        val model = LlamaEngine.getSelectedModel(ctx)
        if (!LlamaEngine.modelsExist(ctx, model)) {
            beginModelFlow(prompt)
            return
        }
        if (isInlineDownloading || isInlineModelLoad) {
            Toast.makeText(this, R.string.toast_please_wait_model, Toast.LENGTH_SHORT).show()
            return
        }
        if (imageJob?.isActive == true) {
            Toast.makeText(this, R.string.image_generating, Toast.LENGTH_SHORT).show()
            return
        }

        etInput.text = null
        etInput.clearFocus()
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager)
            .hideSoftInputFromWindow(etInput.windowToken, 0)
        collapseAppBar()
        autoFollowStreaming = true

        lockOrientationWhileGenerating(true)
        val userMsgId = messageIdCounter++
        messages.add(
            ChatMessage.UserMessage(
                id = userMsgId,
                text = prompt,
                isImagePrompt = true
            )
        )
        val resultId = addImageResultPlaceholder(prompt, requestedByModel = false)
        chatAdapter.submitList(messages.toList()) { scrollToBottom() }
        persistCurrentConversation()
        enableInput(false)

        val request = currentImageRequest(prompt, model)
        val outFile = ImageOutput.newOutputFile(ctx)
        val token = ++imageGenerationToken
        imageJob = imageWorkScope.launch {
            val startNs = SystemClock.elapsedRealtime()
            try {
                val imageEngine = prepareImageEngine(model, resultId) ?: run {
                    failImageResult(resultId, lastImagePreparationError ?: getString(R.string.image_generate_failed))
                    return@launch
                }
                if (token != imageGenerationToken) return@launch

                imageEngine.setProgressListener(token, ImageProgressUi(resultId).also { it.onEncoding() })
                CriticalLog.log("image", "generate start: native heap ${nativeHeapMb()} MB")
                val usedSeed = withContext(NonCancellable) {
                    imageEngine.generate(
                        prompt = request.prompt,
                        negativePrompt = request.negativePrompt,
                        width = request.width,
                        height = request.height,
                        steps = request.steps,
                        cfg = request.cfg,
                        seed = request.seed,
                        outputPath = outFile.absolutePath
                    )
                }
                CriticalLog.log("image", "generate end: seed=$usedSeed, native heap ${nativeHeapMb()} MB")
                if (token != imageGenerationToken) return@launch
                if (usedSeed == null) {
                    failImageResult(resultId, imageEngine.lastError ?: getString(R.string.image_generate_failed))
                    return@launch
                }

                finishImageResult(resultId, outFile.absolutePath, request, usedSeed, (SystemClock.elapsedRealtime() - startNs))
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Image generation failed", e)
                if (token == imageGenerationToken) {
                    failImageResult(resultId, e.message ?: getString(R.string.image_generate_failed))
                }
            } finally {
                lockOrientationWhileGenerating(false)
                imageEngine?.setProgressListener(token, null)
                if (token == imageGenerationToken) {
                    enableInput(true)
                }
            }
        }
    }

    /**
     * Inserts the result cell and returns its id. Params stay empty until the
     * engine reports the seed it actually used.
     */
    private fun addImageResultPlaceholder(prompt: String, requestedByModel: Boolean): Long {
        val id = messageIdCounter++
        messages.add(
            ChatMessage.ImageResult(
                id = id,
                prompt = prompt,
                pngPath = null,
                paramsText = null,
                isGenerating = true,
                requestedByModel = requestedByModel,
                progress = 0f,
                stage = getString(R.string.image_stage_loading)
            )
        )
        return id
    }

    /**
     * Drives the image bubble's progress bar and replaces its picture with the
     * per-step latent preview.
     *
     * The callbacks arrive on the native generation thread. Everything UI
     * related is marshalled to the main thread; preview frames are throttled
     * and replaced in place through the adapter, because re-submitting the
     * whole list would restart the RecyclerView's animations every step.
     */
    private inner class ImageProgressUi(
        private val resultId: Long
    ) : ImageEngine.ProgressListener {

        private var lastPreviewStep = -1
        private var lastPreviewMs = 0L
        // Last fraction handed to the bar; a phase switch may lower the
        // "within this phase" value, so the bar is only ever moved forward.
        private var shownFraction = 0f

        /**
         * The native wrapper announces the phase explicitly: it emits a
         * step-less event on every transition and only the denoising phase
         * carries meaningful step numbers. Tiled VAE decode events (hundreds
         * of tiles) never move the bar as if they were steps.
         */
        override fun onStep(
            samplePass: Int, stage: Int, step: Int, steps: Int, elapsedMs: Double
        ): Boolean {
            when (stage) {
                ImageEngine.STAGE_LOADING -> {
                    // Weight/tensor loading shares the native progress channel
                    // with denoising (426 tensors vs 4 steps). It owns the
                    // first slice and never looks like a sampling step.
                    if (steps > 0) {
                        val within = (step.toFloat() / steps).coerceIn(0f, 1f)
                        postProgress(
                            LOADING_END * within,
                            getString(R.string.image_stage_loading_progress, step, steps)
                        )
                    }
                }
                ImageEngine.STAGE_SAMPLING -> {
                    if (steps > 0) {
                        val within = (step.toFloat() / steps).coerceIn(0f, 1f)
                        postProgress(
                            ENCODING_END + (SAMPLING_END - ENCODING_END) * within,
                            getString(R.string.image_stage_step, step, steps)
                        )
                    } else {
                        // Phase announcement: denoising is about to start.
                        postProgress(ENCODING_END, getString(R.string.image_stage_sampling))
                    }
                }
                ImageEngine.STAGE_DECODING -> {
                    // One announcement, then hundreds of tile events. Only the
                    // first one changes anything; the bar sits at the top of
                    // the denoising band until the PNG is written.
                    postProgress(SAMPLING_END, getString(R.string.image_stage_decoding))
                }
                ImageEngine.STAGE_ENCODING ->
                    postProgress(0f, getString(R.string.image_stage_encoding))
                else -> Unit
            }
            return imageJob?.isActive != false
        }

        override fun onPreview(samplePass: Int, step: Int, width: Int, height: Int, rgb: ByteArray) {
            // Pass 1 only (the native side filters as well), and never before
            // the latent has some structure.
            if (samplePass != 1 || step < PREVIEW_FIRST_STEP || width <= 0 || height <= 0) return
            // A repeated or older step must not overwrite a newer picture.
            if (step <= lastPreviewStep) return
            val now = SystemClock.elapsedRealtime()
            if (now - lastPreviewMs < PREVIEW_INTERVAL_MS) return
            lastPreviewStep = step
            lastPreviewMs = now

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val argb = IntArray(width * height)
            for (i in argb.indices) {
                val o = i * 3
                argb[i] = (0xFF shl 24) or
                    ((rgb[o].toInt() and 0xFF) shl 16) or
                    ((rgb[o + 1].toInt() and 0xFF) shl 8) or
                    (rgb[o + 2].toInt() and 0xFF)
            }
            bitmap.setPixels(argb, 0, width, 0, 0, width, height)
            runOnUiThread {
                if (!chatAdapter.showImagePreview(resultId, bitmap)) bitmap.recycle()
            }
        }

        /** Called once the model is in memory and the prompt is being read. */
        fun onEncoding() {
            postProgress(LOADING_END, getString(R.string.image_stage_encoding))
        }

        private fun postProgress(fraction: Float, stage: String) {
            val clamped = maxOf(shownFraction, fraction.coerceIn(0f, 1f))
            runOnUiThread {
                val index = messages.indexOfFirst { it.id == resultId }
                if (index < 0) return@runOnUiThread
                val current = messages[index] as? ChatMessage.ImageResult ?: return@runOnUiThread
                if (!current.isGenerating) return@runOnUiThread
                // Ignore sub-percent changes: each notification rebinds the row.
                if (current.stage == stage &&
                    kotlin.math.abs(current.progress - clamped) < PROGRESS_EPSILON
                ) {
                    return@runOnUiThread
                }
                shownFraction = clamped
                messages[index] = current.copy(progress = clamped, stage = stage)
                chatAdapter.notifyImageProgress(resultId, clamped, stage)
            }
        }
    }

    /**
     * A draw may finish after the screen is gone. The message list only exists
     * in memory, so the finished bubble is appended to the saved conversation
     * directly; otherwise the picture silently disappears from the history.
     */
    private fun persistDetachedImageResult(
        id: Long,
        pngPath: String,
        request: ImageRequest,
        seed: Long,
        elapsedMs: Long
    ) {
        val conversationId = currentConversationId ?: return
        lifecycleScope.launch(Dispatchers.IO) {
            runCatching {
                val saved = ConversationStore.load(applicationContext, conversationId)
                if (saved.any { it is ChatMessage.ImageResult && it.id == id }) return@runCatching
                val result = ChatMessage.ImageResult(
                    id = id,
                    prompt = request.prompt,
                    pngPath = pngPath,
                    paramsText = ImageOutput.paramsText(
                        applicationContext, request.width, request.height, request.steps, request.cfg, seed
                    ),
                    isGenerating = false,
                    requestedByModel = false,
                    progress = 1f,
                    elapsedMs = elapsedMs
                )
                ConversationStore.save(applicationContext, conversationId, saved + result)
            }.onFailure { Log.w(TAG, "Could not persist a detached image result", it) }
        }
    }

    private fun finishImageResult(
        id: Long,
        pngPath: String,
        request: ImageRequest,
        seed: Long,
        elapsedMs: Long
    ) {
        LlamaEngine.setLastImageSeed(applicationContext, seed)
        val index = messages.indexOfFirst { it.id == id }
        if (index < 0) {
            // The Activity that started this draw is gone (settings, or the
            // system reclaimed it). The picture still exists on disk, so put
            // it in the saved conversation instead of dropping it.
            persistDetachedImageResult(id, pngPath, request, seed, elapsedMs)
            return
        }
        val current = messages[index] as? ChatMessage.ImageResult ?: return
        // A pinned seed keeps producing the same picture; a random one is
        // surfaced as "last seed" so the user can pin it from the config page.
        messages[index] = current.copy(
            pngPath = pngPath,
            paramsText = ImageOutput.paramsText(
                applicationContext, request.width, request.height, request.steps, request.cfg, seed
            ),
            isGenerating = false,
            error = null,
            elapsedMs = elapsedMs
        )
        submitMessages(scrollToEnd = true)
        // Persist synchronously before pruning: save() runs on IO, and pruning
        // on another IO thread could otherwise see the old messages.json, treat
        // the fresh PNG as unreferenced and delete it - the history then loses
        // the picture even though the bubble was shown.
        persistCurrentConversationBlocking()
        refreshConversationList()
        pruneUnusedImages()
        // Tell the native side, so the next prompt sees this turn as a real
        // generate_image call + tool result (see tool_history_append_image_turn).
        // Only while a language model is resident: the history it renders is
        // discarded when that model is unloaded.
        if (isModelReady) {
            lifecycleScope.launch(Dispatchers.IO) {
                runCatching {
                    engine.recordImageToolTurn(request.prompt, File(pngPath).name)
                }.onFailure { Log.w(TAG, "recordImageToolTurn failed", it) }
            }
        }
    }

    /**
     * Drops PNGs no conversation refers to any more (deleted chats, images
     * pruned from history). Cheap enough to run after each picture; skipped
     * while a generation may still be writing files.
     */
    private fun pruneUnusedImages() {
        // A generation in flight owns a file that is not referenced yet.
        if (imageJob?.isActive == true) return
        lifecycleScope.launch(Dispatchers.IO) {
            val referenced = ConversationStore.allImagePaths(applicationContext)
            ImageOutput.pruneOrphans(applicationContext, referenced)
        }
    }

    private fun failImageResult(id: Long, reason: String) {
        val index = messages.indexOfFirst { it.id == id }
        if (index < 0) return
        val current = messages[index] as? ChatMessage.ImageResult ?: return
        messages[index] = current.copy(isGenerating = false, error = reason)
        submitMessages(scrollToEnd = true)
        persistCurrentConversation()
    }

    // ------------------------------------------------------------------
    // generate_image tool called by the language model
    // ------------------------------------------------------------------

    /** Image checkpoint the tool uses: the selected one, else the Q4 model. */
    private fun imageModelForTool(): ModelInfo {
        val selected = LlamaEngine.getSelectedModel(applicationContext)
        if (selected.isImage) return selected
        return ModelInfo.AVAILABLE_MODELS.firstOrNull { it.id == ModelInfo.DEFAULT_IMAGE_MODEL_ID }
            ?: ModelInfo.AVAILABLE_MODELS.first { it.isImage }
    }

    /**
     * Runs the image engine for a tool call the model made inside [aiMsgId].
     *
     * The turnaround is: download the checkpoint if needed (progress stays on
     * the assistant bubble) -> free the chat model when it does not fit ->
     * draw the picture as a new ImageResult -> reload the chat model and feed
     * the result back so the model can close the answer.
     */
    private suspend fun runImageGenerationForTool(
        aiMsgId: Long,
        call: ImageTool.Call
    ): Boolean {
        val ctx = applicationContext
        val imageModel = imageModelForTool()
        val prompt = call.prompt
        lockOrientationWhileGenerating(true)
        val resultId = addImageResultPlaceholder(prompt, requestedByModel = true)
        submitMessages(scrollToEnd = true)

        val startNs = SystemClock.elapsedRealtime()
        val request = currentImageRequest(prompt, imageModel)
        val outFile = ImageOutput.newOutputFile(ctx)
        // The stream that produced this call still owns the llama context; it
        // has to stop before the image stack can take the NPU (and before the
        // chat model may be unloaded).
        cancelActiveGeneration()
        val imageEngine = try {
            // Preparation runs detached: it unloads the chat model (which may
            // take a while) and must finish even though the stream that
            // produced this call is being cancelled right above.
            withContext(NonCancellable) { prepareImageEngine(imageModel, aiMsgId) }
        } catch (e: Exception) {
            Log.e(TAG, "Image tool: preparation failed", e)
            null
        }
        if (imageEngine == null) {
            if (messages.any { it.id == aiMsgId }) {
                failImageResult(resultId, lastImagePreparationError ?: getString(R.string.image_generate_failed))
            }
            return false
        }
        if (!messages.any { it.id == aiMsgId }) return false
        CriticalLog.log("image", "prepare returned a loaded engine (isLoaded=${imageEngine.isLoaded})")

        updateAiStatus(aiMsgId, getString(R.string.tool_generating_image, prompt))
        imageEngine.setProgressListener(
            imageGenerationToken,
            ImageProgressUi(resultId).also { it.onEncoding() }
        )
        Log.i(TAG, "Image tool: drawing ${request.width}x${request.height} steps=${request.steps} cfg=${request.cfg}")
        CriticalLog.log("image", "tool generate start: native heap ${nativeHeapMb()} MB")
        CriticalLog.log("image", "calling imageEngine.generate(...)")
        // NonCancellable: the tool flow is owned by a fresh scope, but the
        // llama stream that produced the call is still being torn down around
        // it and would otherwise cancel the generate() coroutine right here
        // (observed: "tool flow ended: JobCancellationException" one ms after
        // the call).
        val usedSeed = withContext(NonCancellable) {
            imageEngine.generate(
                prompt = request.prompt,
                negativePrompt = request.negativePrompt,
                width = request.width,
                height = request.height,
                steps = request.steps,
                cfg = request.cfg,
                seed = request.seed,
                outputPath = outFile.absolutePath
            )
        }
        imageEngine.setProgressListener(imageGenerationToken, null)
        CriticalLog.log("image", "tool generate end: seed=$usedSeed, native heap ${nativeHeapMb()} MB")
        if (usedSeed == null) {
            failImageResult(resultId, imageEngine.lastError ?: getString(R.string.image_generate_failed))
            return false
        }
        finishImageResult(resultId, outFile.absolutePath, request, usedSeed, (SystemClock.elapsedRealtime() - startNs))
        updateAiStatus(aiMsgId, getString(R.string.tool_image_done))

        // Give the model its turn back so it can answer around the picture.
        if (!isModelReady) {
            if (!reloadChatModel()) {
                appendAiText(
                    aiMsgId,
                    getString(R.string.tool_image_done_no_reply)
                )
                updateAiStatus(aiMsgId, null)
                return true
            }
        }
        val reply = ImageTool.formatImageReply(prompt, outFile.name)
        appendAiText(aiMsgId, reply, isGenerating = true)
        streamChatContinuation(aiMsgId)
        return true
    }

    /** Reloads the chat model after the image stack freed its memory. */
    private suspend fun reloadChatModel(): Boolean {
        if (!::engine.isInitialized) return false
        val ctx = applicationContext
        val model = LlamaEngine.getSelectedModel(ctx)
        if (!LlamaEngine.modelsExist(ctx, model)) return false
        return try {
            val mmproj = if (!model.isTextOnly) LlamaEngine.mmprojPath(ctx)?.takeIf { File(it).exists() } else null
            engine.loadModel(LlamaEngine.modelPath(ctx), mmproj)
            applyChatSystemPrompt()
            // The lazy reload starts from an empty context: replay the
            // conversation (minus the turn being answered) so follow-ups and
            // the image acknowledgement stay in context.
            restoreHistoryForCurrentConversation()
            isModelReady = true
            loadedModelId = model.id
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to reload the chat model after image generation", e)
            isModelReady = false
            loadedModelId = null
            false
        }
    }

    /** Streams the model's follow-up message into [aiMsgId]. */
    private fun streamChatContinuation(aiMsgId: Long) {
        generationJob = lifecycleScope.launch(Dispatchers.Default) {
            val tokens = runCatching { engine.sendUserPrompt("(continue)", predictLength = 256) }
                .getOrElse { error ->
                    Log.e(TAG, "Continuation failed", error)
                    withContext(Dispatchers.Main) { updateAiStatus(aiMsgId, null) }
                    return@launch
                }
            val text = StringBuilder()
            tokens
                .onCompletion { _ ->
                    withContext(Dispatchers.Main) {
                        updateAiStatus(aiMsgId, null)
                        setAiText(aiMsgId, text.toString(), isGenerating = false)
                        enableInput(true)
                        persistCurrentConversation()
                        refreshConversationList()
                    }
                }
                .collect { token ->
                    text.append(token)
                    withContext(Dispatchers.Main) {
                        setAiText(aiMsgId, text.toString(), isGenerating = true)
                    }
                }
        }
    }

    // ---- small helpers used by both paths -----------------------------

    private fun submitMessages(scrollToEnd: Boolean) {
        chatAdapter.submitList(messages.toList()) {
            if (scrollToEnd) followStreamingIfAtBottom()
        }
    }

    /** Replaces the AI message's text (used by status and continuation paths). */
    private fun setAiText(id: Long, text: String, isGenerating: Boolean) {
        val index = messages.indexOfFirst { it.id == id }
        if (index < 0) return
        val current = messages[index] as? ChatMessage.AiMessage ?: return
        messages[index] = current.copy(text = text, isGenerating = isGenerating)
        if (isGenerating) {
            chatAdapter.updateStreamingText(id, text)
        } else {
            chatAdapter.submitList(messages.toList())
        }
    }

    /** Appends to the assistant text and pushes it to the bound holder. */
    private fun appendAiText(id: Long, extra: String, isGenerating: Boolean = false) {
        val index = messages.indexOfFirst { it.id == id }
        if (index < 0) return
        val current = messages[index] as? ChatMessage.AiMessage ?: return
        val text = if (current.text.isBlank()) extra else current.text.trimEnd() + "\n\n" + extra
        messages[index] = current.copy(text = text, isGenerating = isGenerating)
        chatAdapter.updateStreamingText(id, text)
        if (!isGenerating) chatAdapter.submitList(messages.toList())
    }

    /** Shows (or clears, with null) the display-only status line of a bubble. */
    private fun updateAiStatus(id: Long, status: String?) {
        val index = messages.indexOfFirst { it.id == id }
        if (index < 0) return
        val current = messages[index] as? ChatMessage.AiMessage ?: return
        if (current.status == status) return
        messages[index] = current.copy(status = status)
        chatAdapter.submitList(messages.toList()) {
            if (status != null) followStreamingIfAtBottom()
        }
    }

    private fun refreshThinkButton() {
        val on = LlamaEngine.getEnableThinking(this)
        btnThink.isChecked = on
        btnThink.alpha = 1f
    }

    private fun refreshWelcomeCard(isTextOnly: Boolean) {
        val welcomeIndex = messages.indexOfFirst { it is ChatMessage.WelcomeCard }
        if (welcomeIndex >= 0) {
            messages[welcomeIndex] = ChatMessage.WelcomeCard(
                isTextOnly = isTextOnly,
                isImageMode = isImageMode()
            )
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
                applyChatSystemPrompt()
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
        if (selected.isImage && isImageMode()) {
            beginImageGeneration(userMsg)
            return
        }
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
                applyChatSystemPrompt()

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
            var fullResponse = StringBuilder()
            var toolInvoked = false
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
                                // Keep whatever the collector last rendered
                                // (protocol already stripped); a cancelled or
                                // failed stream must not paint the raw bytes.
                                messages[index] = (messages[index] as ChatMessage.AiMessage).copy(
                                    text = ImageTool.strip(fullResponse.toString()),
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
                    // The raw stream may already contain a tool call: keep the
                    // protocol out of the bubble from the first token on (a
                    // partial call is stripped too), so it never flashes.
                    val visible = ImageTool.strip(fullResponse.toString())
                    withContext(Dispatchers.Main) {
                        val index = messages.indexOfFirst { it.id == aiMsgId }
                        if (index >= 0) {
                            messages[index] = ChatMessage.AiMessage(
                                id = aiMsgId,
                                text = visible,
                                isGenerating = true
                            )
                        }
                        chatAdapter.updateStreamingText(aiMsgId, visible)
                        followStreamingIfAtBottom()
                    }
                    persistDuringStreaming()
                }

            // The model asked for a picture. The text so far is what it said
            // before the call; from here on the image path owns this bubble.
            val call = takeIf { LlamaEngine.isImageToolEnabled(applicationContext) }
                ?.let { ImageTool.parse("$fullResponse") }
            // Whether or not a call was found, keep what the model produced:
            // when the tool does not fire, this is the only record of how the
            // model phrased it (the chat bubble only shows the cleaned text).
            DebugLogBridge.dump(
                "image-tool",
                "call=${call?.prompt ?: "none"}\n--- model output ---\n" + fullResponse
            )
            if (call != null) {
                toolInvoked = true
                val visible = ImageTool.strip(fullResponse.toString())
                withContext(Dispatchers.Main) {
                    setAiText(aiMsgId, visible, isGenerating = true)
                    persistCurrentConversation()
                }
                // Dedicated scope: generationJob is cancelled above, and a
                // coroutine started from it (or from a NonCancellable block
                // whose parent is still that job) silently never resumes after
                // a suspension point -- this is the "generating image" that
                // never finishes. The image job owns its own parent.
                imageJob = imageWorkScope.launch {
                    val ok = try {
                        runImageGenerationForTool(aiMsgId, call)
                    } finally {
                        lockOrientationWhileGenerating(false)
                    }
                    Log.i(TAG, "Image tool finished: served=$ok")
                    CriticalLog.log("image", "tool flow finished: served=$ok")
                }.apply {
                    invokeOnCompletion { cause ->
                        // A cancellation here is routine: the streaming job
                        // that produced this call is cancelled on purpose once
                        // the tool fires, and leaving the screen cancels the
                        // image job as well. Only real failures are worth a
                        // line; the served= line above already records the rest.
                        if (cause != null && cause !is kotlinx.coroutines.CancellationException) {
                            CriticalLog.log("image", "tool flow failed: ${cause.javaClass.simpleName}: ${cause.message}")
                        }
                    }
                }
                // Wait for the image flow here so the outer coroutine keeps
                // its lifecycle (it is cancelled when the screen goes away,
                // which also cancels the image job through cancelActiveGeneration).
                imageJob?.join()
            }
            if (!toolInvoked) {
                // Any stray protocol text must never reach the screen.
                val raw = fullResponse.toString()
                val clean = ImageTool.strip(raw)
                if (clean != raw) {
                    withContext(Dispatchers.Main) { setAiText(aiMsgId, clean, isGenerating = false) }
                }
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
        if (!::engine.isInitialized) return
        // The image model / generation parameters may have changed in the
        // configuration page while we were in the background.
        if (loadedImageModelId != null &&
            loadedImageModelId != LlamaEngine.getSelectedModel(applicationContext).id
        ) {
            releaseImageEngine()
        }
        refreshImageModeUi()
        // The generate_image toggle changes the system prompt, which is only
        // read when a model is loaded: drop the resident model so the next
        // request picks the new prompt up.
        if (isModelReady && loadedModelId != null &&
            loadedModelId == LlamaEngine.getSelectedModel(applicationContext).id
        ) {
            val appliedTool = lastSystemPromptHadImageTool
            if (appliedTool != null && appliedTool != LlamaEngine.isImageToolEnabled(applicationContext)) {
                lifecycleScope.launch {
                    runCatching { engine.unloadModel() }
                    isModelReady = false
                    loadedModelId = null
                    Toast.makeText(
                        this@MainActivity,
                        R.string.config_reload_for_prompt,
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
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
        // The chat stream is tied to this screen: leaving it stops the answer.
        generationJob?.cancel()
        // An image draw is not. It runs for a minute on the native thread and
        // cannot be interrupted; cancelling the job here would report a
        // failure while the GPU is still working. It keeps running and the
        // result is written to the history even if we never come back.
        persistCurrentConversation()
        super.onStop()
    }

    override fun onDestroy() {
        // Never let the image stack go away underneath a running draw; the job
        // outlives this Activity and finishes on the native thread it owns.
        val imageBusy = imageJob?.isActive == true || imageEngine?.isGenerating == true
        if (imageBusy) {
            CriticalLog.log("image", "onDestroy during image generation: keeping the engine alive")
        } else {
            releaseImageEngine()
        }
        if (isFinishing && !isLocaleRestart && ::engine.isInitialized) {
            engine.destroy()
        }
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val STREAM_PERSIST_INTERVAL_MS = 1000L
        private const val STATE_CURRENT_CONVERSATION = "state_current_conversation"
        private const val KEY_IMAGE_PROMPTS = "recent_image_prompts"
        private const val MAX_IMAGE_PROMPTS = 10
        // Share of the bar that belongs to loading + text encoding; the rest
        // is the denoising loop, which reports its own step counts.
        // Bar layout: weights 0-40%, encoding 40-45%, denoising 45-90%, VAE
        // decode 90-100%. The native stages are disjoint, so switching phase
        // can never move the bar backwards.
        // The native stages are disjoint, so a phase switch can never move the
        // bar backwards (that is what produced the 426/426 -> 70/426 jump).
        private const val LOADING_END = 0.40f
        private const val ENCODING_END = 0.45f
        private const val SAMPLING_END = 0.90f
        private const val PROGRESS_EPSILON = 0.01f
        private const val PREVIEW_FIRST_STEP = 2
        // Two previews per second is plenty for a thumbnail and keeps the
        // downsampled JPEG-free bitmap churn off the UI thread budget.
        private const val PREVIEW_INTERVAL_MS = 500L
    }
}
