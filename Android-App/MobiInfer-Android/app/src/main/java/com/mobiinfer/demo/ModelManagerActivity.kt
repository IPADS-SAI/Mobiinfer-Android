package com.mobiinfer.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.roundToInt

/**
 * Configuration page for both model families.
 *
 * A segmented switch at the top selects the family:
 *  - "Language": chat/VLM checkpoints plus their runtime settings (context
 *    size, memory budget, image scale, debug logs);
 *  - "Image": text-to-image checkpoints plus the generation parameters
 *    (size, steps, CFG, negative prompt) that used to live on a dedicated
 *    screen.
 *
 * Selection is shared with the home page (same persisted `selectedModelId`);
 * nothing is loaded here - the home page loads lazily on the first request.
 */
class ModelManagerActivity : AppCompatActivity() {

    private lateinit var recyclerModels: RecyclerView
    private lateinit var sliderImageScale: Slider
    private lateinit var tvImageScaleValue: TextView

    private lateinit var engine: LlamaEngine
    private lateinit var modelAdapter: ModelAdapter

    private var lastRunningModelId: String? = null
    private var pendingDownloadModelId: String? = null

    // Which family the page is showing. Persisted so a return visit lands on
    // the same tab the user left.
    private var imageMode = false
    private var sliderImageWidth: Slider? = null
    private var sliderImageHeight: Slider? = null

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted) {
                Toast.makeText(this, getString(R.string.toast_notification_denied), Toast.LENGTH_LONG).show()
            }
            pendingDownloadModelId?.let { ModelDownloadService.start(this, it) }
            pendingDownloadModelId = null
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_manager)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val rootContent = findViewById<View>(android.R.id.content)
        ViewCompat.setOnApplyWindowInsetsListener(rootContent) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updatePadding(
                left = systemBars.left,
                top = systemBars.top,
                right = systemBars.right,
                bottom = systemBars.bottom
            )
            insets
        }

        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        toolbar.setNavigationOnClickListener { finish() }

        val toggle = findViewById<MaterialButtonToggleGroup>(R.id.toggle_config_mode)
        toggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            val target = checkedId == R.id.btn_mode_image
            if (target != imageMode) {
                imageMode = target
                applyMode()
            }
        }

        recyclerModels = findViewById(R.id.recycler_models)
        sliderImageScale = findViewById(R.id.slider_image_scale)
        tvImageScaleValue = findViewById(R.id.tv_image_scale_value)

        engine = LlamaEngine.getInstance(applicationContext)

        setupModelList()
        setupImageScale()
        setupMemoryBudget()
        setupContextSize()
        setupImageParameters()
        setupDebugLogs()

        imageMode = readStoredMode()
        toggle.check(if (imageMode) R.id.btn_mode_image else R.id.btn_mode_llm)
        applyMode()
        refreshStates()
        observeDownloads()
    }

    private fun setupDebugLogs() {
        val card = findViewById<View>(R.id.card_debug_logs)
        if (!DebugLogBridge.isAvailable()) {
            card.visibility = View.GONE
            return
        }

        card.visibility = View.VISIBLE
        val toggle = findViewById<MaterialSwitch>(R.id.switch_debug_logs)
        toggle.isChecked = DebugLogBridge.isEnabled(this)
        toggle.setOnCheckedChangeListener { _, enabled ->
            DebugLogBridge.setEnabled(this, enabled)
            Toast.makeText(
                this,
                if (enabled) R.string.debug_logging_enabled else R.string.debug_logging_disabled,
                Toast.LENGTH_SHORT
            ).show()
        }

        findViewById<MaterialButton>(R.id.btn_open_debug_logs).setOnClickListener {
            if (!DebugLogBridge.open(this)) {
                Toast.makeText(this, R.string.debug_logs_unavailable, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setupImageScale() {
        sliderImageScale.value = LlamaEngine.getImageScaleFactor(this)
        tvImageScaleValue.text = getString(R.string.image_scale_value, sliderImageScale.value)
        sliderImageScale.addOnChangeListener { _, value, fromUser ->
            tvImageScaleValue.text = getString(R.string.image_scale_value, value)
            if (fromUser) {
                LlamaEngine.setImageScaleFactorPref(this, value)
            }
        }
    }

    private fun setupMemoryBudget() {
        val slider = findViewById<Slider>(R.id.slider_memory_budget)
        val valueView = findViewById<TextView>(R.id.tv_memory_budget_value)
        fun showValue(mb: Int) {
            valueView.text = getString(R.string.memory_budget_value, mb / 1000f)
        }
        val maxMb = LlamaEngine.getMaxMemoryBudgetMb(this)
        slider.valueFrom = LlamaEngine.MIN_MEMORY_BUDGET_MB / 1000f
        slider.valueTo = maxMb / 1000f
        val budget = LlamaEngine.getMemoryBudgetMb(this).coerceIn(LlamaEngine.MIN_MEMORY_BUDGET_MB, maxMb)
        slider.value = budget / 1000f
        showValue(budget)
        slider.addOnChangeListener { _, value, fromUser ->
            val mb = (value * 10).roundToInt() * 100
            showValue(mb)
            if (fromUser) LlamaEngine.setMemoryBudgetMb(this, mb)
        }
        findViewById<MaterialButton>(R.id.btn_memory_budget_max).setOnClickListener {
            // Programmatic slider changes have fromUser=false.
            LlamaEngine.setMemoryBudgetMb(this, maxMb)
            slider.value = slider.valueTo
        }
        updateMemoryBudgetVisibility()
    }

    private fun setupContextSize() {
        val slider = findViewById<Slider>(R.id.slider_context_size)
        val valueView = findViewById<TextView>(R.id.tv_context_size_value)
        fun showValue(tokens: Int) {
            valueView.text = getString(R.string.context_size_value, tokens)
        }
        slider.valueFrom = LlamaEngine.MIN_CONTEXT_SIZE.toFloat()
        slider.valueTo = LlamaEngine.getMaxContextSize(this).toFloat()
        slider.stepSize = LlamaEngine.CONTEXT_SIZE_STEP.toFloat()
        val size = LlamaEngine.getContextSize(this)
        slider.value = size.toFloat()
        showValue(size)
        slider.addOnChangeListener { _, value, fromUser ->
            val tokens = (value / LlamaEngine.CONTEXT_SIZE_STEP).roundToInt() * LlamaEngine.CONTEXT_SIZE_STEP
            showValue(tokens)
            if (fromUser) LlamaEngine.setContextSize(this, tokens)
        }
    }

    private fun updateMemoryBudgetVisibility() {
        findViewById<View>(R.id.card_memory_budget).visibility =
            if (LlamaEngine.getSelectedModel(this).expertsDirectory != null) View.VISIBLE else View.GONE
        findViewById<Slider>(R.id.slider_context_size).let { slider ->
            val maxSize = LlamaEngine.getMaxContextSize(this).toFloat()
            if (slider.value > maxSize) slider.value = maxSize
            slider.valueTo = maxSize
            slider.value = LlamaEngine.getContextSize(this).toFloat()
        }
    }

    /** Models belonging to the family currently shown. */
    private fun modelsForMode(): List<ModelInfo> =
        ModelInfo.AVAILABLE_MODELS.filter { it.isImage == imageMode }

    /** Selected model of the currently shown family, if any. */
    private fun selectedModelForMode(): ModelInfo? {
        val selected = LlamaEngine.getSelectedModel(this)
        return if (selected.isImage == imageMode) selected else null
    }

    private fun setupModelList() {
        modelAdapter = ModelAdapter(
            models = modelsForMode(),
            selectedModelId = selectedModelForMode()?.id ?: "",
            onModelSelected = { model ->
                LlamaEngine.setSelectedModel(this, model.id)
                updateMemoryBudgetVisibility()
                // Selection is lazy: never unload/load here. Loading happens on
                // the home page when the user sends a request.
                if (!LlamaEngine.modelsExist(this, model)) {
                    Toast.makeText(this, R.string.model_not_downloaded_hint, Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(
                        this,
                        getString(R.string.model_switched_lazy, model.displayName),
                        Toast.LENGTH_SHORT
                    ).show()
                }
            },
            onDownload = { model -> requestDownload(model) },
            onPause = { ModelDownloadService.pause(this) },
            onResume = { model -> requestDownload(model) },
            onDelete = { model -> confirmDelete(model) }
        )

        recyclerModels.layoutManager = LinearLayoutManager(this)
        recyclerModels.adapter = modelAdapter
        // The list lives inside a ScrollView; a wrap_content RecyclerView there
        // measures against the viewport height and clips the last row(s). A
        // small unaveraged adapter keeps that from happening in the first
        // place, so the explicit height freeze is enough.
        recyclerModels.isNestedScrollingEnabled = false
        recyclerModels.setHasFixedSize(false)
    }

    /** Swap the list content when the language/image switch changes. */
    private fun rebuildModelList() {
        val selected = LlamaEngine.getSelectedModel(this)
        modelAdapter = ModelAdapter(
            models = modelsForMode(),
            selectedModelId = selectedModelForMode()?.id ?: "",
            onModelSelected = { model ->
                LlamaEngine.setSelectedModel(this, model.id)
                updateMemoryBudgetVisibility()
                if (!LlamaEngine.modelsExist(this, model)) {
                    Toast.makeText(this, R.string.model_not_downloaded_hint, Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(
                        this,
                        getString(R.string.model_switched_lazy, model.displayName),
                        Toast.LENGTH_SHORT
                    ).show()
                }
            },
            onDownload = { model -> requestDownload(model) },
            onPause = { ModelDownloadService.pause(this) },
            onResume = { model -> requestDownload(model) },
            onDelete = { model -> confirmDelete(model) }
        )
        recyclerModels.adapter = modelAdapter
        // The running id may belong to the other family.
        modelAdapter.setProgressTarget(ModelDownloadController.runningModelId())
        refreshStates()
    }

    // ------------------------------------------------------------------
    // Language / image switch
    // ------------------------------------------------------------------

    private fun modePrefs() = getSharedPreferences("config_page_prefs", MODE_PRIVATE)

    private fun readStoredMode(): Boolean = modePrefs().getBoolean(KEY_CONFIG_MODE_IMAGE, false)

    /** Shows the groups and list rows that belong to the selected family. */
    private fun applyMode() {
        modePrefs().edit().putBoolean(KEY_CONFIG_MODE_IMAGE, imageMode).apply()
        findViewById<View>(R.id.group_llm_settings).visibility =
            if (imageMode) View.GONE else View.VISIBLE
        findViewById<View>(R.id.group_image_settings).visibility =
            if (imageMode) View.VISIBLE else View.GONE
        rebuildModelList()
        refreshImageParameters()
    }

    // ------------------------------------------------------------------
    // Text-to-image generation parameters
    // ------------------------------------------------------------------

    private fun setupImageParameters() {
        val width = findViewById<Slider>(R.id.slider_image_width)
        val height = findViewById<Slider>(R.id.slider_image_height)
        sliderImageWidth = width
        sliderImageHeight = height

        width.valueFrom = LlamaEngine.MIN_IMAGE_GEN_SIDE.toFloat()
        width.valueTo = LlamaEngine.MAX_IMAGE_GEN_SIDE.toFloat()
        width.stepSize = LlamaEngine.IMAGE_GEN_SIDE_STEP.toFloat()
        height.valueFrom = LlamaEngine.MIN_IMAGE_GEN_SIDE.toFloat()
        height.valueTo = LlamaEngine.MAX_IMAGE_GEN_SIDE.toFloat()
        height.stepSize = LlamaEngine.IMAGE_GEN_SIDE_STEP.toFloat()
        refreshImageParameters()

        width.addOnChangeListener { _, value, fromUser ->
            // Keep the callback budget: the value is snapped to the 64 px
            // grid before it is persisted, otherwise two sliders can be off
            // by an unsupported odd size.
            if (!fromUser) return@addOnChangeListener
            val px = snapSide(value)
            LlamaEngine.setImageGenWidth(this, px)
            updateImageSizeLabel()
        }
        height.addOnChangeListener { _, value, fromUser ->
            if (!fromUser) return@addOnChangeListener
            val px = snapSide(value)
            LlamaEngine.setImageGenHeight(this, px)
            updateImageSizeLabel()
        }

        val steps = findViewById<Slider>(R.id.slider_image_steps)
        val stepsLabel = findViewById<TextView>(R.id.tv_image_steps_value)
        steps.valueFrom = LlamaEngine.MIN_IMAGE_GEN_STEPS.toFloat()
        steps.valueTo = LlamaEngine.MAX_IMAGE_GEN_STEPS.toFloat()
        steps.stepSize = 1f
        steps.value = LlamaEngine.getImageGenSteps(this).toFloat()
        stepsLabel.text = LlamaEngine.getImageGenSteps(this).toString()
        steps.addOnChangeListener { _, value, fromUser ->
            val n = value.roundToInt()
            stepsLabel.text = n.toString()
            if (fromUser) LlamaEngine.setImageGenSteps(this, n)
        }

        val cfg = findViewById<Slider>(R.id.slider_image_cfg)
        val cfgLabel = findViewById<TextView>(R.id.tv_image_cfg_value)
        cfg.valueFrom = 1.0f
        cfg.valueTo = 10.0f
        cfg.stepSize = 0.1f
        cfg.value = LlamaEngine.getImageGenCfg(this)
        cfgLabel.text = formatCfg(LlamaEngine.getImageGenCfg(this))
        cfg.addOnChangeListener { _, value, fromUser ->
            cfgLabel.text = formatCfg(value)
            if (fromUser) LlamaEngine.setImageGenCfg(this, value)
        }

        val negative = findViewById<TextInputEditText>(R.id.et_image_negative)
        negative.setText(LlamaEngine.getImageNegativePrompt(this))
        negative.setSelection(negative.text?.length ?: 0)
        negative.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: android.text.Editable?) {
                LlamaEngine.setImageNegativePrompt(this@ModelManagerActivity, s?.toString().orEmpty())
            }
        })

        // Seed: empty field means "random every time" (-1). A number pins the
        // sampler so the same prompt reproduces the same picture.
        val seed = findViewById<TextInputEditText>(R.id.et_image_seed)
        // Four digits: the input filter keeps stray characters out and the
        // range check stops a value that would not fit the field.
        seed.filters = arrayOf(
            android.text.InputFilter.LengthFilter(4),
            android.text.method.DigitsKeyListener.getInstance("0123456789")
        )
        seed.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: android.text.Editable?) {
                val text = s?.toString()?.trim().orEmpty()
                if (text.isEmpty()) {
                    LlamaEngine.setImageGenSeed(this@ModelManagerActivity, LlamaEngine.RANDOM_IMAGE_SEED)
                } else {
                    text.toLongOrNull()?.let { LlamaEngine.setImageGenSeed(this@ModelManagerActivity, it) }
                }
                updateSeedHint()
            }
        })
        findViewById<MaterialButton>(R.id.btn_image_seed_random).setOnClickListener {
            // Rolls a fresh four-digit seed instead of clearing the field: an
            // empty field looked like "no seed" rather than "a random one".
            val next = LlamaEngine.randomImageSeed()
            seed.setText(next.toString())
            seed.setSelection(seed.text?.length ?: 0)
            LlamaEngine.setImageGenSeed(this, next)
            updateSeedHint()
        }
        findViewById<MaterialButton>(R.id.btn_image_seed_last).setOnClickListener {
            val last = LlamaEngine.getLastImageSeed(this)
            if (last >= 0) {
                val shown = last % (LlamaEngine.IMAGE_SEED_MAX + 1)
                seed.setText(shown.toString())
                seed.setSelection(seed.text?.length ?: 0)
                LlamaEngine.setImageGenSeed(this, shown)
                updateSeedHint()
            }
        }

        // generate_image tool toggle: takes effect when the chat model reloads
        // (see MainActivity.onResume).
        val toolSwitch = findViewById<MaterialSwitch>(R.id.switch_image_tool)
        toolSwitch.isChecked = LlamaEngine.isImageToolEnabled(this)
        toolSwitch.setOnCheckedChangeListener { _, enabled ->
            LlamaEngine.setImageToolEnabled(this, enabled)
        }

        findViewById<MaterialButton>(R.id.btn_image_reset).setOnClickListener {
            LlamaEngine.setImageGenWidth(this, LlamaEngine.DEFAULT_IMAGE_GEN_SIDE)
            LlamaEngine.setImageGenHeight(this, LlamaEngine.DEFAULT_IMAGE_GEN_SIDE)
            LlamaEngine.setImageGenSteps(this, LlamaEngine.DEFAULT_IMAGE_GEN_STEPS)
            LlamaEngine.setImageGenCfg(this, LlamaEngine.DEFAULT_IMAGE_GEN_CFG)
            LlamaEngine.setImageNegativePrompt(this, "")
            LlamaEngine.setImageGenSeed(this, LlamaEngine.RANDOM_IMAGE_SEED)
            refreshImageParameters()
            Toast.makeText(this, R.string.image_param_saved, Toast.LENGTH_SHORT).show()
        }
    }

    /** Explains what the seed field currently does, and offers the last one. */
    private fun updateSeedHint() {
        val fixed = LlamaEngine.getImageGenSeed(this) != LlamaEngine.RANDOM_IMAGE_SEED
        val last = LlamaEngine.getLastImageSeed(this)
        // "Use the previous seed" only makes sense while a seed is pinned;
        // it used to disappear exactly then, which is the opposite of useful.
        findViewById<MaterialButton>(R.id.btn_image_seed_last).visibility =
            if (fixed && last >= 0) View.VISIBLE else View.GONE
        // The syntax is already in the helperText; this line only says what the
        // field currently does, and the buttons carry the rest.
        findViewById<TextView>(R.id.tv_image_seed_hint).text =
            if (fixed) getString(R.string.image_param_seed_fixed) else ""
        val useLast = findViewById<MaterialButton>(R.id.btn_image_seed_last)
        useLast.text = getString(R.string.image_param_seed_use_last, last)
    }

    private fun refreshImageParameters() {
        val w = LlamaEngine.getImageGenWidth(this)
        val h = LlamaEngine.getImageGenHeight(this)
        sliderImageWidth?.value = w.toFloat()
        sliderImageHeight?.value = h.toFloat()
        findViewById<TextView>(R.id.tv_image_steps_value).text =
            LlamaEngine.getImageGenSteps(this).toString()
        val cfg = LlamaEngine.getImageGenCfg(this)
        findViewById<TextView>(R.id.tv_image_cfg_value).text = formatCfg(cfg)
        findViewById<Slider>(R.id.slider_image_steps).value = LlamaEngine.getImageGenSteps(this).toFloat()
        findViewById<Slider>(R.id.slider_image_cfg).value = cfg
        // Only rewrite the field when it would actually change, so refreshing
        // (e.g. after a reset) does not jump the caret around.
        val seedValue = LlamaEngine.getImageGenSeed(this)
        val seedField = findViewById<TextInputEditText>(R.id.et_image_seed)
        val seedText = if (seedValue == LlamaEngine.RANDOM_IMAGE_SEED) "" else seedValue.toString()
        if (seedField.text?.toString() != seedText) seedField.setText(seedText)
        updateSeedHint()
        val negativeField = findViewById<TextInputEditText>(R.id.et_image_negative)
        val negativeValue = LlamaEngine.getImageNegativePrompt(this)
        if (negativeField.text?.toString() != negativeValue) {
            negativeField.setText(negativeValue)
        }
        updateImageSizeLabel()
    }

    private fun updateImageSizeLabel() {
        val w = sliderImageWidth?.value?.let { snapSide(it) } ?: LlamaEngine.getImageGenWidth(this)
        val h = sliderImageHeight?.value?.let { snapSide(it) } ?: LlamaEngine.getImageGenHeight(this)
        findViewById<TextView>(R.id.tv_image_size_value).text =
            getString(R.string.image_param_size_value, w, h)
        findViewById<TextView>(R.id.tv_image_width_value).text = w.toString()
        findViewById<TextView>(R.id.tv_image_height_value).text = h.toString()
    }

    private fun snapSide(value: Float): Int {
        val step = LlamaEngine.IMAGE_GEN_SIDE_STEP
        val snapped = (value / step).roundToInt() * step
        return snapped.coerceIn(LlamaEngine.MIN_IMAGE_GEN_SIDE, LlamaEngine.MAX_IMAGE_GEN_SIDE)
    }

    private fun formatCfg(value: Float): String =
        String.format(java.util.Locale.getDefault(), "%.1f", value)

    private fun fractionOf(model: ModelInfo): Float {
        if (model.totalBytes <= 0L) return 0f
        val downloaded = LlamaEngine.downloadedBytes(this, model)
        return (downloaded.toDouble() / model.totalBytes.toDouble()).toFloat().coerceIn(0f, 1f)
    }

    private fun refreshStates() {
        val runningId = ModelDownloadController.runningModelId()
        val running = ModelDownloadController.status.value as? ModelDownloadController.Status.Running
        val states = modelsForMode().associate { model ->
            val ui = when {
                runningId == model.id ->
                    ModelRowUi(ModelRowState.DOWNLOADING, running?.fraction ?: 0f, running?.message)
                LlamaEngine.modelsExist(this, model) ->
                    ModelRowUi(ModelRowState.READY)
                LlamaEngine.isPartial(this, model) ->
                    ModelRowUi(ModelRowState.PAUSED, fractionOf(model))
                else ->
                    ModelRowUi(ModelRowState.NOT_DOWNLOADED)
            }.copy(running = LlamaEngine.loadedModelId == model.id)
            model.id to ui
        }
        modelAdapter.setStates(states)
    }

    private fun observeDownloads() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                ModelDownloadController.status.collect { status ->
                    when (status) {
                        is ModelDownloadController.Status.Idle -> {
                            lastRunningModelId = null
                            modelAdapter.setProgressTarget(null)
                            refreshStates()
                        }
                        is ModelDownloadController.Status.Running -> {
                            if (lastRunningModelId != status.modelId) {
                                lastRunningModelId = status.modelId
                                modelAdapter.setProgressTarget(status.modelId)
                                refreshStates()
                            }
                            modelAdapter.updateProgress(status.modelId, status.fraction, status.message)
                        }
                        is ModelDownloadController.Status.Paused -> {
                            lastRunningModelId = null
                            modelAdapter.setProgressTarget(null)
                            refreshStates()
                        }
                        is ModelDownloadController.Status.Completed -> {
                            lastRunningModelId = null
                            modelAdapter.setProgressTarget(null)
                            refreshStates()
                            Toast.makeText(this@ModelManagerActivity, R.string.download_complete_toast, Toast.LENGTH_SHORT).show()
                            ModelDownloadController.acknowledge()
                        }
                        is ModelDownloadController.Status.Cancelled -> {
                            lastRunningModelId = null
                            modelAdapter.setProgressTarget(null)
                            refreshStates()
                            ModelDownloadController.acknowledge()
                        }
                        is ModelDownloadController.Status.Failed -> {
                            lastRunningModelId = null
                            modelAdapter.setProgressTarget(null)
                            refreshStates()
                            Toast.makeText(
                                this@ModelManagerActivity,
                                getString(R.string.download_failed, status.message),
                                Toast.LENGTH_LONG
                            ).show()
                            ModelDownloadController.acknowledge()
                        }
                    }
                }
            }
        }
    }

    private fun requestDownload(model: ModelInfo) {
        if (ModelDownloadController.isRunning) {
            Toast.makeText(this, R.string.toast_downloading, Toast.LENGTH_SHORT).show()
            return
        }
        if (LlamaEngine.modelsExist(this, model)) {
            Toast.makeText(this, R.string.toast_already_downloaded, Toast.LENGTH_SHORT).show()
            return
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            pendingDownloadModelId = model.id
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }

        ModelDownloadService.start(this, model.id)
    }

    private fun confirmDelete(model: ModelInfo) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_dialog_title)
            .setMessage(getString(R.string.delete_dialog_message, model.displayName, model.ggufFileName))
            .setPositiveButton(R.string.delete) { _, _ -> deleteModelFiles(model) }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun deleteModelFiles(model: ModelInfo) {
        lifecycleScope.launch(Dispatchers.IO) {
            // Stop an in-flight download for this model; keep the partial files
            // only if the user cancels deletion (they can't, we already asked).
            if (ModelDownloadController.runningModelId() == model.id) {
                ModelDownloadService.pause(applicationContext)
                kotlinx.coroutines.delay(300)
            }
            // Unload if this is the model currently held by the engine.
            val selected = LlamaEngine.getSelectedModel(applicationContext)
            if (selected.id == model.id && engine.state.value is LlamaState.ModelReady) {
                runCatching { engine.unloadModel() }
            }
            val dir = File(LlamaEngine.modelDirFor(applicationContext, model))
            val deleted = dir.exists() && dir.deleteRecursively()
            withContext(Dispatchers.Main) {
                refreshStates()
                val messageRes = if (deleted) R.string.model_files_deleted else R.string.toast_no_files_to_delete
                Toast.makeText(this@ModelManagerActivity, messageRes, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Selection may have changed on the home page.
        modelAdapter.updateSelection(selectedModelForMode()?.id ?: "")
        updateMemoryBudgetVisibility()
        refreshStates()
    }

    companion object {
        private val TAG = ModelManagerActivity::class.java.simpleName
        private const val KEY_CONFIG_MODE_IMAGE = "config_mode_image"
    }
}
