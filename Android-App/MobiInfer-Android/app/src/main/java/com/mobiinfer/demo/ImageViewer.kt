package com.mobiinfer.demo

import android.content.ContentValues
import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

/**
 * Full-screen preview for a generated picture, plus "save to gallery".
 *
 * Saving goes through MediaStore so no storage permission is needed on any
 * supported Android version: the PNG is copied into
 * Pictures/MobiInfer and a content:// URI handed back for the toast/share
 * action.
 */
object ImageViewer {

    private const val TAG = "ImageViewer"
    private const val ALBUM = "MobiInfer"
    private val io = Executors.newSingleThreadExecutor { r -> Thread(r, "image-save").apply { isDaemon = true } }
    private val main = Handler(Looper.getMainLooper())

    /** Opens [path] full screen over [activity]. */
    fun show(activity: AppCompatActivity, path: String?) {
        if (path == null || !File(path).exists()) return
        val uri = Uri.fromFile(File(path))
        val intent = android.content.Intent(activity, ImageViewerActivity::class.java)
            .putExtra(ImageViewerActivity.EXTRA_URI, uri.toString())
        activity.startActivity(intent)
    }

    /** Ask where to save, then copy on a background thread. */
    fun saveToGallery(activity: AppCompatActivity, path: String?) {
        if (path == null || !File(path).exists()) {
            android.widget.Toast.makeText(activity, R.string.image_save_failed, android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        // MediaStore's RELATIVE_PATH/IS_PENDING only exist from Android 10; on
        // older releases a plain file in the public Pictures dir is used,
        // which is why WRITE_EXTERNAL_STORAGE is declared with maxSdkVersion 28.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            saveLegacy(activity, path)
            return
        }
        androidx.appcompat.app.AlertDialog.Builder(activity)
            .setTitle(R.string.image_save_title)
            .setMessage(R.string.image_save_message)
            .setPositiveButton(R.string.image_save_confirm) { _, _ -> copyToGallery(activity, path) }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun copyToGallery(activity: AppCompatActivity, path: String) {
        val appContext = activity.applicationContext
        val resolver = appContext.contentResolver
        io.execute {
            var uri: Uri? = null
            try {
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(path, bounds)
                val name = "MobiInfer_${System.currentTimeMillis()}.png"
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    if (bounds.outWidth > 0) put(MediaStore.Images.Media.WIDTH, bounds.outWidth)
                    if (bounds.outHeight > 0) put(MediaStore.Images.Media.HEIGHT, bounds.outHeight)
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$ALBUM")
                        put(MediaStore.Images.Media.IS_PENDING, 1)
                    }
                }
                uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: throw IllegalStateException("MediaStore refused the insert")
                val out = resolver.openOutputStream(uri)
                    ?: throw IllegalStateException("Cannot open output stream")
                out.use { File(path).inputStream().use { input -> input.copyTo(it) } }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                }
                main.post {
                    showSaved(activity, uri!!)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save image to gallery", e)
                uri?.let { runCatching { resolver.delete(it, null, null) } }
                main.post {
                    android.widget.Toast.makeText(activity, R.string.image_save_failed, android.widget.Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun saveLegacy(activity: AppCompatActivity, path: String) {
        if (androidx.core.content.ContextCompat.checkSelfPermission(
                activity, android.Manifest.permission.WRITE_EXTERNAL_STORAGE
            ) != android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            android.widget.Toast.makeText(activity, R.string.image_save_permission_needed, android.widget.Toast.LENGTH_LONG).show()
            return
        }
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), ALBUM)
        io.execute {
            try {
                if (!dir.exists() && !dir.mkdirs()) throw IllegalStateException("Cannot create $dir")
                val dest = File(dir, "MobiInfer_${System.currentTimeMillis()}.png")
                File(path).copyTo(dest)
                // Make the file show up in the gallery app immediately.
                android.media.MediaScannerConnection.scanFile(
                    activity.applicationContext, arrayOf(dest.absolutePath), arrayOf("image/png"), null
                )
                main.post {
                    android.widget.Toast.makeText(activity, R.string.image_save_done, android.widget.Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save image (legacy path)", e)
                main.post {
                    android.widget.Toast.makeText(activity, R.string.image_save_failed, android.widget.Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showSaved(activity: AppCompatActivity, uri: Uri) {
        if (activity.isFinishing || activity.isDestroyed) return
        androidx.appcompat.app.AlertDialog.Builder(activity)
            .setTitle(R.string.image_save_done_title)
            .setMessage(R.string.image_save_done)
            .setPositiveButton(R.string.image_open_gallery) { _, _ ->
                val view = android.content.Intent(android.content.Intent.ACTION_VIEW).apply {
                    setDataAndType(uri, "image/png")
                    addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                runCatching { activity.startActivity(view) }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
}
