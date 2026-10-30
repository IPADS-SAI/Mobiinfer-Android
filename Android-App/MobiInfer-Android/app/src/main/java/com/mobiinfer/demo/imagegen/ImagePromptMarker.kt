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
     * itself, and a note that the picture was drawn (so the model does not
     * claim to have drawn it itself).
     */
    fun forLlm(text: String): Pair<String, String>? {
        if (!isImageTurn(text)) return null
        val prompt = strip(text).trim()
        return prompt to "[image generated: $prompt]"
    }
}
