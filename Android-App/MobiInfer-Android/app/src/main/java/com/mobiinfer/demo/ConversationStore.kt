package com.mobiinfer.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import com.mobiinfer.demo.imagegen.ImagePromptMarker
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/**
 * A single turn fed to [LlamaEngine.restoreConversation] when the user opens a
 * saved conversation. `imageUris` are SAF content:// references, re-read at
 * restore time; if a reference has gone stale that turn is replayed without
 * its picture.
 */
data class ChatHistoryTurn(
    val role: String,
    val text: String,
    val imageUris: List<String> = emptyList()
)

/** Lightweight list entry shown in the conversation drawer. */
data class ConversationMeta(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long
)

/**
 * Disk-backed conversation history. Each conversation is a directory under
 * `filesDir/conversations/<id>/` holding a `messages.json`; an `index.json`
 * keeps the ordered list plus the currently-selected conversation. Messages are
 * serialized with org.json (no extra dependency). Image bitmaps are *not*
 * persisted - only the content-URI reference - and are re-decoded on load.
 */
object ConversationStore {

    private const val TAG = "ConversationStore"
    private const val DIR = "conversations"
    private const val INDEX_FILE = "index.json"
    private const val MESSAGES_FILE = "messages.json"
    private const val DEFAULT_TITLE = "新对话"
    private const val TITLE_MAX = 24
    private const val DECODE_MAX_DIM = 1024

    private fun root(context: Context): File = File(context.filesDir, DIR).apply { mkdirs() }

    private fun dirFor(context: Context, id: String): File = File(root(context), id)

    private fun messagesFile(context: Context, id: String): File = File(dirFor(context, id), MESSAGES_FILE)

    // ------------------------------------------------------------------ index

    private class Index(
        val conversations: MutableList<ConversationMeta>,
        var currentId: String?
    )

    private fun loadIndex(context: Context): Index {
        val file = File(root(context), INDEX_FILE)
        val conversations = mutableListOf<ConversationMeta>()
        var currentId: String? = null
        if (file.exists()) {
            try {
                val json = JSONObject(file.readText())
                currentId = json.optString("current", null)?.takeIf { it.isNotBlank() }
                val arr = json.optJSONArray("conversations") ?: JSONArray()
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    conversations.add(
                        ConversationMeta(
                            id = o.getString("id"),
                            title = o.optString("title", DEFAULT_TITLE),
                            createdAt = o.optLong("createdAt"),
                            updatedAt = o.optLong("updatedAt")
                        )
                    )
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to read conversation index, starting fresh", e)
            }
        }
        return Index(conversations, currentId)
    }

    private fun saveIndex(context: Context, index: Index) {
        val arr = JSONArray()
        for (meta in index.conversations) {
            arr.put(
                JSONObject().apply {
                    put("id", meta.id)
                    put("title", meta.title)
                    put("createdAt", meta.createdAt)
                    put("updatedAt", meta.updatedAt)
                }
            )
        }
        val json = JSONObject().apply {
            put("current", index.currentId ?: JSONObject.NULL)
            put("conversations", arr)
        }
        File(root(context), INDEX_FILE).writeText(json.toString())
    }

    // ------------------------------------------------------------------- api

    /** Conversation list, most-recently-updated first (createdAt as tie-break). */
    @Synchronized
    fun list(context: Context): List<ConversationMeta> =
        loadIndex(context).conversations.sortedWith(
            compareByDescending<ConversationMeta> { it.updatedAt }.thenByDescending { it.createdAt }
        )

    @Synchronized
    fun currentId(context: Context): String? = loadIndex(context).currentId

    @Synchronized
    fun setCurrentId(context: Context, id: String) {
        val index = loadIndex(context)
        index.currentId = id
        saveIndex(context, index)
    }

    /**
     * Returns the conversation to show on launch, creating a fresh one only when
     * there is nothing to resume.  If the saved `current` id is missing/stale we
     * fall back to the most recently updated conversation instead of silently
     * starting a new one on every restart.
     */
    @Synchronized
    fun ensureCurrent(context: Context): ConversationMeta {
        val index = loadIndex(context)
        val existing = index.currentId
            ?.takeIf { it.isNotBlank() }
            ?.let { id -> index.conversations.find { it.id == id } }
        if (existing != null) return existing

        val latest = index.conversations.maxByOrNull { it.updatedAt }
        if (latest != null) {
            index.currentId = latest.id
            saveIndex(context, index)
            return latest
        }
        return create(context)
    }

    /**
     * Creates a transient "new conversation" draft.  It is intentionally NOT
     * added to the index yet: an empty conversation should never show up in
     * the history list.  It is promoted into the index by [save] once it has
     * real content.
     */
    @Synchronized
    fun create(context: Context): ConversationMeta {
        val now = System.currentTimeMillis()
        val meta = ConversationMeta(
            id = UUID.randomUUID().toString(),
            title = DEFAULT_TITLE,
            createdAt = now,
            updatedAt = now
        )
        dirFor(context, meta.id).mkdirs()
        return meta
    }

    @Synchronized
    fun rename(context: Context, id: String, title: String) {
        val index = loadIndex(context)
        val i = index.conversations.indexOfFirst { it.id == id }
        if (i < 0) return
        val old = index.conversations[i]
        index.conversations[i] = old.copy(title = title)
        saveIndex(context, index)
    }

    @Synchronized
    fun delete(context: Context, id: String) {
        val index = loadIndex(context)
        index.conversations.removeAll { it.id == id }
        if (index.currentId == id) index.currentId = index.conversations.firstOrNull()?.id
        saveIndex(context, index)
        dirFor(context, id).deleteRecursively()
    }

    @Synchronized
    fun load(context: Context, id: String): MutableList<ChatMessage> {
        val file = messagesFile(context, id)
        val out = mutableListOf<ChatMessage>()
        if (!file.exists()) return out
        try {
            val arr = JSONArray(file.readText())
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                when (o.optString("type")) {
                    "user" -> {
                        val uri = o.optString("imageUri", null)
                        out.add(
                            ChatMessage.UserMessage(
                                id = o.getLong("id"),
                                text = o.optString("text", ""),
                                imageBitmap = uri?.let { decodeBitmap(context, it) },
                                imageInfo = o.optString("imageInfo", null),
                                isVideo = o.optBoolean("isVideo", false),
                                imageUri = uri,
                                isImagePrompt = o.optBoolean("isImagePrompt", false)
                            )
                        )
                    }
                    "ai" -> {
                        val metrics = o.optJSONObject("metrics")?.let { m ->
                            InferenceMetrics(
                                imageProcessingMs = if (m.has("imgMs")) m.getLong("imgMs") else null,
                                prefillTokensPerSecond = if (m.has("prefillTps")) m.getDouble("prefillTps") else null,
                                decodeTokensPerSecond = if (m.has("decodeTps")) m.getDouble("decodeTps") else null,
                                totalMs = m.optLong("totalMs")
                            )
                        }
                        out.add(
                            ChatMessage.AiMessage(
                                id = o.getLong("id"),
                                text = o.optString("text", ""),
                                metrics = metrics
                            )
                        )
                    }
                    "image" -> {
                        val pngPath = o.optString("pngPath", "").takeIf { it.isNotBlank() }
                        // Drop results whose file was cleaned up so the cell
                        // never shows a permanently blank frame.
                        if (pngPath == null || File(pngPath).exists()) {
                            out.add(
                                ChatMessage.ImageResult(
                                    id = o.getLong("id"),
                                    prompt = o.optString("prompt", ""),
                                    pngPath = pngPath,
                                    paramsText = o.optString("params", "").takeIf { it.isNotBlank() },
                                    elapsedMs = if (o.has("elapsedMs")) o.getLong("elapsedMs") else null
                                )
                            )
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read conversation $id", e)
        }
        return out
    }

    /** Persists the conversation (UI-only cards are skipped) and rolls the title forward. */
    @Synchronized
    fun save(context: Context, id: String, messages: List<ChatMessage>) {
        val arr = JSONArray()
        for (m in messages) {
            when (m) {
                is ChatMessage.UserMessage -> arr.put(
                    JSONObject().apply {
                        put("type", "user")
                        put("id", m.id)
                        put("text", m.text)
                        m.imageInfo?.let { put("imageInfo", it) }
                        put("isVideo", m.isVideo)
                        m.imageUri?.let { put("imageUri", it) }
                        if (m.isImagePrompt) put("isImagePrompt", true)
                    }
                )
                is ChatMessage.AiMessage -> {
                    // Never persist the transient "model loading" placeholder:
                    // it would reappear as an empty assistant bubble on restart.
                    if (!m.isLoadingPlaceholder) {
                        arr.put(
                            JSONObject().apply {
                                put("type", "ai")
                                put("id", m.id)
                                put("text", m.text)
                                m.metrics?.let { mt ->
                                    put("metrics", JSONObject().apply {
                                        mt.imageProcessingMs?.let { put("imgMs", it) }
                                        mt.prefillTokensPerSecond?.let { put("prefillTps", it) }
                                        mt.decodeTokensPerSecond?.let { put("decodeTps", it) }
                                        put("totalMs", mt.totalMs)
                                    })
                                }
                            }
                        )
                    }
                }
                is ChatMessage.ImageResult -> {
                    // Only persist finished results; an in-flight generation has
                    // no file yet and would come back as a blank frame.
                    if (!m.isGenerating && m.pngPath != null) {
                        arr.put(
                            JSONObject().apply {
                                put("type", "image")
                                put("id", m.id)
                                put("prompt", m.prompt)
                                put("pngPath", m.pngPath)
                                m.paramsText?.let { put("params", it) }
                                m.elapsedMs?.let { put("elapsedMs", it) }
                            }
                        )
                    }
                }
                else -> Unit
            }
        }
        // A brand-new conversation is a transient draft until it actually has
        // content (a user turn or a real assistant reply).  Empty "new chat"
        // entries must never appear in the history list.
        val hasContent = messages.any { m ->
            (m is ChatMessage.UserMessage && (m.text.isNotBlank() || m.imageUri != null)) ||
            (m is ChatMessage.AiMessage && !m.isLoadingPlaceholder && m.text.isNotBlank()) ||
            (m is ChatMessage.ImageResult && !m.isGenerating && m.pngPath != null) ||
            (m is ChatMessage.UserMessage && m.isImagePrompt)
        }
        if (!hasContent) return

        dirFor(context, id).mkdirs()
        val file = messagesFile(context, id)
        val newJson = arr.toString()
        val index = loadIndex(context)
        val i = index.conversations.indexOfFirst { it.id == id }

        // Merely switching / opening a conversation persists it without any
        // content change.  Don't rewrite or bump updatedAt then.
        val previous = if (file.exists()) runCatching { file.readText() }.getOrNull() else null
        if (previous == newJson && i >= 0) return
        file.writeText(newJson)

        val title = autoTitle(messages) ?: index.conversations.getOrNull(i)?.title ?: DEFAULT_TITLE
        val now = System.currentTimeMillis()
        if (i >= 0) {
            index.conversations[i] = index.conversations[i].copy(title = title, updatedAt = now)
        } else {
            index.conversations.add(
                ConversationMeta(id = id, title = title, createdAt = now, updatedAt = now)
            )
        }
        index.currentId = id
        saveIndex(context, index)
    }

    /** First user line, trimmed - used as the drawer label. Null if none yet. */
    fun autoTitle(messages: List<ChatMessage>): String? {
        val first = messages.firstOrNull { it is ChatMessage.UserMessage && it.text.isNotBlank() }
            as? ChatMessage.UserMessage ?: return null
        val clean = first.text.replace('\n', ' ').trim()
        return if (clean.length <= TITLE_MAX) clean else clean.substring(0, TITLE_MAX) + "…"
    }

    /** Every generated PNG referenced by any saved conversation. */
    fun allImagePaths(context: Context): Set<String> {
        val paths = mutableSetOf<String>()
        for (meta in list(context)) {
            messagesFile(context, meta.id).takeIf { it.exists() }?.let { file ->
                runCatching {
                    val arr = JSONArray(file.readText())
                    for (i in 0 until arr.length()) {
                        val o = arr.getJSONObject(i)
                        if (o.optString("type") == "image") {
                            o.optString("pngPath", "").takeIf { it.isNotBlank() }?.let { paths.add(it) }
                        }
                    }
                }.onFailure { Log.w(TAG, "Failed to scan images of ${meta.id}", it) }
            }
        }
        return paths
    }

    /** Persisted turns, in order, for context replay on conversation switch. */
    fun toHistoryTurns(messages: List<ChatMessage>): List<ChatHistoryTurn> {
        val turns = mutableListOf<ChatHistoryTurn>()
        for (m in messages) {
            when (m) {
                is ChatMessage.UserMessage -> {
                    // A user bubble may be a pure image turn (empty text); the
                    // model still needs a non-empty content string there.
                    val text = m.text.ifBlank { if (m.imageUri != null) "" else "" }
                    turns.add(
                        ChatHistoryTurn(
                            role = "user",
                            text = text,
                            imageUris = m.imageUri?.let { listOf(it) } ?: emptyList()
                        )
                    )
                    // Text-to-image turns: the picture was not drawn by the
                    // LLM, so tell it what happened instead of leaving a
                    // question that looks answered only by the next turn. The
                    // image path rides in the marker so the replay can use the
                    // file name (and therefore find the PNG again).
                    if (m.isImagePrompt) {
                        turns.add(
                            ChatHistoryTurn(
                                role = "assistant",
                                text = ImagePromptMarker.encode(text)
                            )
                        )
                    }
                }
                is ChatMessage.AiMessage -> {
                    if (m.text.isNotBlank() && !m.isLoadingPlaceholder) {
                        turns.add(ChatHistoryTurn(role = "assistant", text = m.text))
                    }
                }
                // Image results never enter the LLM context: they belong to the
                // text-to-image engine, which is a separate native stack. While
                // in image mode the chat model is not even loaded.
                is ChatMessage.ImageResult -> {
                    // The acknowledgement was already emitted above, next to
                    // the prompt; the result cell itself is UI only.
                }
                else -> Unit
            }
        }
        return turns
    }

    private fun decodeBitmap(context: Context, uriString: String): Bitmap? = try {
        val uri = Uri.parse(uriString)
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (bounds.outWidth / (sample * 2) >= DECODE_MAX_DIM || bounds.outHeight / (sample * 2) >= DECODE_MAX_DIM) {
            sample *= 2
        }
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    } catch (e: Exception) {
        Log.w(TAG, "Failed to decode image URI: $uriString", e)
        null
    }
}
