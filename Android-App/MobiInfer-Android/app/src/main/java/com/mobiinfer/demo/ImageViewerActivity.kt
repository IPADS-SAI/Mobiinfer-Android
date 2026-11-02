package com.mobiinfer.demo

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Full-screen viewer for a generated picture: pinch-to-zoom, tap to dismiss,
 * and a save button that copies the PNG into the system gallery.
 */
class ImageViewerActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_URI = "extra_image_uri"
        private const val TAG = "ImageViewerActivity"
    }

    private var currentPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_image_viewer)
        WindowCompat.setDecorFitsSystemWindows(window, false)

        val uriString = intent.getStringExtra(EXTRA_URI)
        val uri = uriString?.let { Uri.parse(it) }
        currentPath = uri?.path

        val image = findViewById<ImageView>(R.id.iv_full_image)
        val status = findViewById<TextView>(R.id.tv_viewer_status)
        val btnSave = findViewById<MaterialButton>(R.id.btn_viewer_save)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(android.R.id.content)) { v, insets ->
            val bars = insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        val path = currentPath
        if (path == null || !File(path).exists()) {
            status.setText(R.string.image_missing)
            status.visibility = View.VISIBLE
            btnSave.isEnabled = false
            return
        }

        btnSave.isEnabled = true
        btnSave.setOnClickListener { ImageViewer.saveToGallery(this, currentPath) }
        findViewById<MaterialButton>(R.id.btn_viewer_share).setOnClickListener { share(path) }
        findViewById<View>(R.id.viewer_root).setOnClickListener { finish() }

        lifecycleScope.launch {
            // Decode at full size off the main thread; a 1024px PNG is a few
            // MB of pixels and would otherwise jank the open animation.
            val bitmap = withContext(Dispatchers.IO) {
                runCatching { BitmapFactory.decodeFile(path) }.getOrNull()
            }
            if (bitmap == null) {
                Log.w(TAG, "Cannot decode $path")
                status.setText(R.string.image_missing)
                status.visibility = View.VISIBLE
                return@launch
            }
            image.setImageBitmap(bitmap)
            image.visibility = View.VISIBLE
        }
    }

    private fun share(path: String) {
        val file = File(path)
        val authority = "$packageName.fileprovider"
        val uri = androidx.core.content.FileProvider.getUriForFile(this, authority, file)
        val send = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(android.content.Intent.EXTRA_STREAM, uri)
            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(android.content.Intent.createChooser(send, getString(R.string.image_share_title)))
    }
}
