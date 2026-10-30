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
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.roundToInt

/**
 * Model download centre.  This page no longer selects/loads models for
 * inference: selection is shared with the home page (same persisted
 * `selectedModelId`) and loading happens lazily on the home page when the user
 * sends a request.  Here users only decide whether to download / pause / resume
 * / delete each model, plus the advanced settings (image scale, memory budget).
 */
class ModelManagerActivity : AppCompatActivity() {

    private lateinit var recyclerModels: RecyclerView
    private lateinit var sliderImageScale: Slider
    private lateinit var tvImageScaleValue: TextView

    private lateinit var engine: LlamaEngine
    private lateinit var modelAdapter: ModelAdapter

    private var lastRunningModelId: String? = null
    private var pendingDownloadModelId: String? = null

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

        recyclerModels = findViewById(R.id.recycler_models)
        sliderImageScale = findViewById(R.id.slider_image_scale)
        tvImageScaleValue = findViewById(R.id.tv_image_scale_value)

        engine = LlamaEngine.getInstance(applicationContext)

        setupModelList()
        setupImageScale()
        setupMemoryBudget()
        setupContextSize()
        setupDebugLogs()
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

    private fun setupModelList() {
        val selectedModel = LlamaEngine.getSelectedModel(this)
        modelAdapter = ModelAdapter(
            models = ModelInfo.AVAILABLE_MODELS,
            selectedModelId = selectedModel.id,
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
        // measures against the viewport height and clips the last row(s). Force
        // it to expand to its full content height instead.
        recyclerModels.isNestedScrollingEnabled = false
        recyclerModels.post {
            val lp = recyclerModels.layoutParams
            recyclerModels.measure(
                View.MeasureSpec.makeMeasureSpec(recyclerModels.width, View.MeasureSpec.EXACTLY),
                View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
            )
            lp.height = recyclerModels.measuredHeight
            recyclerModels.layoutParams = lp
        }
    }

    private fun fractionOf(model: ModelInfo): Float {
        if (model.totalBytes <= 0L) return 0f
        val downloaded = LlamaEngine.downloadedBytes(this, model)
        return (downloaded.toDouble() / model.totalBytes.toDouble()).toFloat().coerceIn(0f, 1f)
    }

    private fun refreshStates() {
        val runningId = ModelDownloadController.runningModelId()
        val running = ModelDownloadController.status.value as? ModelDownloadController.Status.Running
        val states = ModelInfo.AVAILABLE_MODELS.associate { model ->
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
        modelAdapter.updateSelection(LlamaEngine.getSelectedModel(this).id)
        updateMemoryBudgetVisibility()
        refreshStates()
    }

    companion object {
        private val TAG = ModelManagerActivity::class.java.simpleName
    }
}
