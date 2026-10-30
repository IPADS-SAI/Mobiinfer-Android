package com.mobiinfer.demo

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Text-to-image screen backed by stable-diffusion.cpp (see ImageEngine).
 *
 * The selected model (from ModelManager) supplies the diffusion checkpoint.
 * Output PNGs are written to the cache dir and shown in an ImageView.
 */
class ImageActivity : AppCompatActivity() {

    companion object {
        private val TAG = ImageActivity::class.java.simpleName
    }

    private lateinit var etPrompt: TextInputEditText
    private lateinit var etNegative: TextInputEditText
    private lateinit var etWidth: TextInputEditText
    private lateinit var etHeight: TextInputEditText
    private lateinit var etSteps: TextInputEditText
    private lateinit var etCfg: TextInputEditText
    private lateinit var etSeed: TextInputEditText
    private lateinit var ivResult: ImageView
    private lateinit var tvStatus: TextView
    private lateinit var btnGenerate: MaterialButton

    private lateinit var engine: ImageEngine
    private var generationJob: Job? = null
    private var lastPngPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_image)

        initViews()
        setupListeners()
        initEngine()
    }

    private fun initViews() {
        etPrompt = findViewById(R.id.et_prompt)
        etNegative = findViewById(R.id.et_negative)
        etWidth = findViewById(R.id.et_width)
        etHeight = findViewById(R.id.et_height)
        etSteps = findViewById(R.id.et_steps)
        etCfg = findViewById(R.id.et_cfg)
        etSeed = findViewById(R.id.et_seed)
        ivResult = findViewById(R.id.iv_result)
        tvStatus = findViewById(R.id.tv_status)
        btnGenerate = findViewById(R.id.btn_generate)

        findViewById<ImageButton>(R.id.btn_model_manager).setOnClickListener {
            startActivity(Intent(this, ModelManagerActivity::class.java))
        }
    }

    private fun setupListeners() {
        btnGenerate.setOnClickListener { onGenerateClicked() }
    }

    private fun initEngine() {
        engine = ImageEngine.getInstance(this)
        observeEngineState()

        val modelPath = LlamaEngine.modelPath(this)
        val tePath = LlamaEngine.imageTextEncoderPath(this)
        val vaePath = LlamaEngine.imageVaePath(this)
        val tokenizerPath = LlamaEngine.imageTokenizerPath(this)

        val missing = listOfNotNull(modelPath, tePath, vaePath, tokenizerPath).any { !File(it).exists() }
        if (missing) {
            promptDownloadModel()
            return
        }

        lifecycleScope.launch {
            val ok = engine.loadModel(
                diffusionModelPath = modelPath,
                llmPath = tePath,
                vaePath = vaePath,
                tokenizerPath = tokenizerPath,
                nThreads = 0,
                backend = "diffusion=HTP0,te=cpu,vae=HTP0"
            )
            if (!ok) {
                tvStatus.visibility = View.VISIBLE
                tvStatus.text = getString(R.string.image_error, "load failed")
            }
        }
    }

    private fun promptDownloadModel() {
        AlertDialog.Builder(this)
            .setTitle(R.string.image_download_prompt_title)
            .setMessage(R.string.image_model_missing)
            .setCancelable(false)
            .setPositiveButton(R.string.go_download) { _, _ ->
                startActivity(Intent(this, ModelManagerActivity::class.java))
                finish()
            }
            .setNegativeButton(R.string.cancel) { _, _ ->
                Toast.makeText(this, R.string.image_model_missing, Toast.LENGTH_LONG).show()
            }
            .show()
    }

    private fun observeEngineState() {
        lifecycleScope.launch {
            engine.state.collect { state ->
                when (state) {
                    is ImageState.Uninitialized, is ImageState.Initializing -> {
                        btnGenerate.isEnabled = false
                    }
                    is ImageState.LoadingModel -> {
                        btnGenerate.isEnabled = false
                        tvStatus.visibility = View.VISIBLE
                        tvStatus.setText(R.string.image_loading_model)
                    }
                    is ImageState.Ready -> {
                        btnGenerate.isEnabled = true
                        btnGenerate.setText(R.string.image_generate)
                        if (engine.isLoaded) tvStatus.visibility = View.GONE
                    }
                    is ImageState.Generating -> {
                        btnGenerate.isEnabled = false
                        btnGenerate.setText(R.string.image_generating)
                        tvStatus.visibility = View.VISIBLE
                        tvStatus.setText(R.string.image_generating)
                    }
                    is ImageState.Error -> {
                        btnGenerate.isEnabled = true
                        btnGenerate.setText(R.string.image_generate)
                        tvStatus.visibility = View.VISIBLE
                        tvStatus.text = getString(R.string.image_error, state.exception.message)
                    }
                }
            }
        }
    }

    private fun onGenerateClicked() {
        if (generationJob?.isActive == true) return

        val prompt = etPrompt.text?.toString()?.trim().orEmpty()
        if (prompt.isEmpty()) {
            Toast.makeText(this, R.string.image_prompt_empty, Toast.LENGTH_SHORT).show()
            return
        }

        val negative = etNegative.text?.toString()?.trim().orEmpty()
        val width = etWidth.text?.toString()?.toIntOrNull()?.coerceIn(128, 1024) ?: 512
        val height = etHeight.text?.toString()?.toIntOrNull()?.coerceIn(128, 1024) ?: 512
        val steps = etSteps.text?.toString()?.toIntOrNull()?.coerceIn(1, 150) ?: 8
        val cfg = etCfg.text?.toString()?.toFloatOrNull() ?: 3.0f
        val seed = etSeed.text?.toString()?.toLongOrNull() ?: -1L

        val outFile = File(cacheDir, "image_${System.currentTimeMillis()}.png")

        generationJob = lifecycleScope.launch {
            val ok = engine.generate(
                prompt = prompt,
                negativePrompt = negative,
                width = width,
                height = height,
                steps = steps,
                cfg = cfg,
                seed = seed,
                outputPath = outFile.absolutePath
            )
            if (ok) {
                showImage(outFile)
            } else {
                Toast.makeText(this@ImageActivity, R.string.image_generate_failed, Toast.LENGTH_LONG).show()
            }
        }
    }

    private suspend fun showImage(file: File) {
        val bitmap = withContext(Dispatchers.IO) {
            BitmapFactory.decodeFile(file.absolutePath)
        }
        if (bitmap != null) {
            lastPngPath = file.absolutePath
            ivResult.setImageBitmap(bitmap)
            Log.i(TAG, "Displayed generated image: ${file.absolutePath}")
        } else {
            Toast.makeText(this, R.string.image_generate_failed, Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        generationJob?.cancel()
    }
}
