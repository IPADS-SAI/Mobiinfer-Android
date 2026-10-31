package com.mobiinfer.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import com.mobiinfer.demo.imagegen.ImagePromptMarker
import com.mobiinfer.demo.imagegen.ImageTool
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

    // Parsed index, replaced on every write. list() and currentId() run on the
    // main thread and used to re-read and re-parse the whole file each time; a
    // rebuild over hundreds of directories made that visibly slow, and it could
    // happen on every call while the index was unreadable.
    @Volatile
    private var cachedIndex: Index? = null

    // Whether the on-disk directories have been reconciled with the index in
    // this process (see saveIndex).
    @Volatile
    private var indexReconciled = false

    private fun loadIndex(context: Context): Index {
        cachedIndex?.let { return it }
        val parsed = parseIndex(context)
        cachedIndex = parsed
        return parsed
    }

    private fun parseIndex(context: Context): Index {
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
                // Do NOT treat an unreadable index as "no conversations". The
                // list is rewritten on the next save, so starting fresh here
                // permanently deletes every entry - which is exactly how the
                // drawer collapsed from 210 rows to 2 while all the data was
                // still on disk. Rebuild from the directories instead.
                Log.w(TAG, "Failed to read conversation index, rebuilding from disk", e)
                return rebuildIndex(context)
            }
        }
        return Index(conversations, currentId)
    }

    /**
     * Rebuilds the conversation list from the directories on disk.
     *
     * Every conversation is a directory with a messages.json, so the index is
     * only a cache of titles and timestamps. Used when it cannot be parsed, and
     * when the app finds entries on disk that the index does not know about -
     * a lost index must never lose the conversations themselves.
     */
    private fun rebuildIndex(context: Context): Index {
        val conversations = mutableListOf<ConversationMeta>()
        val dirs = root(context).listFiles()?.filter { it.isDirectory } ?: emptyList()
        for (dir in dirs) {
            val messages = File(dir, MESSAGES_FILE)
            if (!messages.exists()) continue
            var title = DEFAULT_TITLE
            var createdAt = dir.lastModified()
            var updatedAt = createdAt
            runCatching {
                val arr = JSONArray(messages.readText())
                // The title is what the user said, not what the model answered.
                var firstText: String? = null
                for (i in 0 until arr.length()) {
                    val o = arr.optJSONObject(i) ?: continue
                    val text = o.optString("text", "").takeIf { it.isNotBlank() } ?: continue
                    if (firstText == null) firstText = text
                    if (o.optString("type") == "user") { firstText = text; break }
                }
                firstText?.let { title = it.lineSequence().first().take(TITLE_MAX).trim() }
                updatedAt = messages.lastModified()
            }
            conversations.add(
                ConversationMeta(
                    id = dir.name,
                    title = title,
                    createdAt = createdAt,
                    updatedAt = updatedAt
                )
            )
        }
        conversations.sortByDescending { it.updatedAt }
        // This pass visited every directory, so a later save() has nothing
        // left to reconcile.
        indexReconciled = true
        Log.i(TAG, "Rebuilt conversation index from disk: ${conversations.size} conversation(s)")
        CriticalLog.log("chat", "index rebuilt from disk: ${conversations.size} conversation(s)")
        return Index(conversations, conversations.firstOrNull()?.id)
    }

    private fun saveIndex(context: Context, index: Index) {
        // Adoption pass: a conversation directory the list does not know about
        // is kept rather than dropped. The index only caches titles and times,
        // so a stale one must never delete conversations that still exist.
        val known = index.conversations.mapTo(HashSet()) { it.id }
        // Reconciled once per process, not on every save: save() runs on the
        // main thread during a stream, and a directory listing per save is
        // wasted work as soon as the index is known to be complete. A count
        // mismatch is enough to decide - an entry pointing at a deleted
        // directory and a directory the index never saw cancel out numerically,
        // so the first pass always scans and only a clean pass sets the flag.
        val maybeLost = if (!indexReconciled) {
            root(context).listFiles()
                ?.filter { it.isDirectory && it.name !in known && File(it, MESSAGES_FILE).exists() }
                ?: emptyList()
        } else {
            emptyList()
        }
        if (!indexReconciled) {
            indexReconciled = true
        }
        if (maybeLost.isNotEmpty()) {
            val adopted = rebuildIndex(context).conversations.filter { it.id !in known }
            index.conversations.addAll(adopted)
            index.conversations.sortByDescending { it.updatedAt }
            Log.i(TAG, "Adopted ${adopted.size} conversation(s) present on disk but missing from the index")
            CriticalLog.log("chat", "index adopted ${adopted.size} orphaned conversation(s)")
        }
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
        // Write to a sibling and rename: a process death or a kill in the
        // middle of a plain writeText() left a truncated index, and the next
        // launch then read it as "no conversations".
        val target = File(root(context), INDEX_FILE)
        val temp = File(root(context), "$INDEX_FILE.tmp")
        temp.writeText(json.toString())
        if (!temp.renameTo(target)) {
            // renameTo does not replace an existing file on every filesystem.
            target.delete()
            if (!temp.renameTo(target)) {
                Log.w(TAG, "Could not replace $INDEX_FILE atomically, writing in place")
                target.writeText(json.toString())
                temp.delete()
            }
        }
        // Adopted orphans were appended to this very object, so publishing it
        // keeps the cache identical to what just hit the disk.
        cachedIndex = index
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
        // Prompt of the picture that was just drawn, so the assistant turn that
        // follows it can mention what it produced.
        var previousImagePrompt: String? = null
        // Set when the picture turn above was just emitted, so the assistant
        // sentence that follows merges into it instead of starting a new turn.
        var pendingDrawnPrompt: String? = null
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
                        // Home-screen image mode: the picture was not drawn by
                        // the model, so replay it as a completed generate_image
                        // call rather than as a raw prompt. The \uE000 sentinel
                        // means nothing to the native side - it would reach the
                        // model as a stray control character glued to the text.
                        previousImagePrompt = text
                        turns.add(
                            ChatHistoryTurn(
                                role = "assistant",
                                text = ImagePromptMarker.toolRecord(text)
                            )
                        )
                    }
                }
                is ChatMessage.AiMessage -> {
                    if (m.text.isNotBlank() && !m.isLoadingPlaceholder) {
                        val drawnPrompt = pendingDrawnPrompt
                        if (drawnPrompt != null) {
                            // Fold the sentence into the picture turn that was
                            // just emitted, keeping one assistant turn.
                            // The record is wrapped in \u0001 markers, so append
                            // after the closing one without an extra space: the
                            // native parser must see exactly the JSON object
                            // between the markers, and anything glued to it -
                            // even whitespace - changes what it has to accept.
                            val last = turns.lastOrNull()
                            if (last != null && last.role == "assistant") {
                                turns[turns.size - 1] = last.copy(
                                    text = last.text + m.text
                                )
                            } else {
                                turns.add(ChatHistoryTurn(role = "assistant", text = m.text))
                            }
                            pendingDrawnPrompt = null
                        } else {
                            turns.add(ChatHistoryTurn(role = "assistant", text = m.text))
                        }
                        previousImagePrompt = null
                    }
                }
                // Image results never enter the LLM context: they belong to the
                // text-to-image engine, which is a separate native stack. While
                // in image mode the chat model is not even loaded.
                is ChatMessage.ImageResult -> {
                    // A finished picture is the assistant's answer to the turn
                    // above it. Omitting it made the model replay its own
                    // request with no reply and conclude it cannot draw;
                    // narrating it in extra prose made it treat the topic as
                    // closed. Recording it as the plain answer to that request
                    // is what keeps later prompts calling the tool.
                    if (!m.isGenerating && m.pngPath != null && previousImagePrompt == null) {
                        // The native side turns this record into a real
                        // generate_image call + result. Without it the model
                        // replays "user asked for a picture" with a plain
                        // sentence as the answer and stops calling the tool.
                        turns.add(
                            ChatHistoryTurn(
                                role = "assistant",
                                text = ImagePromptMarker.toolRecord(m.prompt)
                            )
                        )
                        pendingDrawnPrompt = m.prompt
                    }
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
