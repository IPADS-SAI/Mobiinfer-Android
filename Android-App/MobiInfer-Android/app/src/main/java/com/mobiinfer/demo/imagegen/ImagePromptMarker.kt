package com.mobiinfer.demo.imagegen

/**
 * Invisible sentinel that carries the image prompt inside a chat message.
 *
 * The picture is drawn by the text-to-image engine, not by the language
 * model, so the history replayed into the LLM would otherwise show a bare
 * "user prompt" with no assistant answer. The marker lets
 * [com.mobiinfer.demo.ConversationStore.toHistoryTurns] emit the prompt as a
 * user turn plus a short assistant acknowledgement instead.
 *
 * The character is a private-use code point that models never produce, and
 * the marker is stripped before anything is rendered.
 */
object ImagePromptMarker {

    private const val SENTINEL = '\uE000'

    /** Prefix the native side looks for; versioned so the shape can change. */
    private const val BRIDGE = "\u0001image-gen:v1:"
    fun encode(text: String): String = SENTINEL + text

    fun strip(text: String): String =
        if (text.indexOf(SENTINEL) < 0) text
        else text.split(SENTINEL).joinToString("")

    fun isImageTurn(text: String): Boolean {
        val at = text.indexOf(SENTINEL)
        if (at < 0) return false
        // Accept any text carrying the sentinel: marker-only, or a prompt on
        // either side of it, depending on how the replay composed the turn.
        return true
    }

    /**
     * Text that goes back into the LLM context for an image turn: the prompt
     * itself, and a machine-readable record of the draw.
     *
     * The native side turns the record into a real `generate_image` tool call
     * plus its result, which is what the model needs to see to call the tool
     * again on the next request. It is one JSON object wrapped in \u0001 on
     * both sides: a prompt containing `]`, quotes or newlines cannot truncate
     * it the way the older `[image generated: ...]` text could.
     */
    fun forLlm(text: String): Pair<String, String>? {
        if (!isImageTurn(text)) return null
        val prompt = strip(text).trim()
        val payload = """{"prompt":${quote(prompt)},"result":"The image was generated and shown to the user."}"""
        return prompt to "$BRIDGE$payload\u0001"
    }

    /**
     * Bridge record for a picture drawn through the tool: same payload as
     * [forLlm] but written straight from the prompt, no sentinel round trip.
     */
    fun toolRecord(prompt: String): String {
        val payload = """{"prompt":${quote(prompt)},"result":"The image was generated and shown to the user."}"""
        return "$BRIDGE$payload\u0001"
    }

    /** Minimal JSON string escaping; the payload never leaves this app. */
    private fun quote(value: String): String {
        val out = StringBuilder(value.length + 2)
        out.append('"')
        for (c in value) {
            when (c) {
                '"' -> out.append("\\\"")
                '\\' -> out.append("\\\\")
                '\n' -> out.append("\\n")
                '\r' -> out.append("\\r")
                '\t' -> out.append("\\t")
                else -> if (c < ' ') out.append(String.format("\\u%04x", c.code)) else out.append(c)
            }
        }
        out.append('"')
        return out.toString()
    }
}
