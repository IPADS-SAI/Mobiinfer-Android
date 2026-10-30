package com.mobiinfer.demo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File

/**
 * Files produced by the text-to-image stack.
 *
 * Results live next to the app's private data (not cacheDir) so a picture
 * stays available to the chat history after the OS trims caches. Saving to
 * the gallery copies the PNG out through MediaStore, which needs no storage
 * permission on any supported Android version.
 */
object ImageOutput {

    private val fileNameRegex = Regex("^image_(\\d{13})\\.png$")

    /**
     * Reserves a new output file and returns its path. File names embed the
     * timestamp they were created at.
     */
    fun newOutputFile(context: android.content.Context): File {
        val dir = LlamaEngine.imageOutputDir(context)
        var stamp = System.currentTimeMillis()
        var file = File(dir, "image_$stamp.png")
        // Two requests inside the same millisecond would otherwise overwrite
        // each other, and a path already on disk must never be reused.
        while (file.exists()) {
            stamp++
            file = File(dir, "image_$stamp.png")
        }
        return file
    }

    /** Creation timestamp of a generated file, or null for foreign paths. */
    fun timestampOf(path: String): Long? =
        fileNameRegex.find(File(path).name)?.groupValues?.getOrNull(1)?.toLongOrNull()

    /** Decodes a result for display, sampling large images down. */
    fun decodeForDisplay(path: String, maxDim: Int = 1536): Bitmap? = try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        var sample = 1
        while (bounds.outWidth / (sample * 2) >= maxDim || bounds.outHeight / (sample * 2) >= maxDim) {
            sample *= 2
        }
        BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
    } catch (e: Exception) {
        null
    }

    /** Caption shown under a result: dimensions, steps, CFG and the seed. */
    fun paramsText(context: android.content.Context, width: Int, height: Int, steps: Int, cfg: Float, seed: Long): String =
        context.getString(
            R.string.image_result_params,
            width,
            height,
            steps,
            String.format(java.util.Locale.getDefault(), "%.1f", cfg),
            seed
        )

    /**
     * Deletes PNGs that no conversation refers to any more. Keeps the disk
     * cost bounded without touching pictures the user can still scroll back to.
     */
    fun pruneOrphans(context: android.content.Context, referencedPaths: Collection<String>) {
        val referenced = referencedPaths.toSet()
        LlamaEngine.imageOutputDir(context).listFiles()?.forEach { file ->
            if (file.isFile && file.name.endsWith(".png") && file.absolutePath !in referenced) {
                runCatching { file.delete() }
            }
        }
    }
}
