package com.mobiinfer.demo.imagegen

/**
 * The `generate_image` tool that lets a language model ask the app to draw a
 * picture with the on-device text-to-image engine.
 *
 * The llama stack in this app has no native function-calling loop, so the
 * tool is described in the system prompt and called with a small text
 * protocol: the model emits a line of the form
 *
 * ```
 * <tool_call>{"name":"generate_image","arguments":{"prompt":"..."}}</tool_call>
 * ```
 *
 * [parse] recovers the call (tolerating the bare-JSON and missing-closing-tag
 * variations small models like to produce), [strip] removes it from the text
 * before it is rendered, and [formatImageReply] turns the engine's result
 * into the answer the model continues from.
 */
object ImageTool {

    const val NAME = "generate_image"
    const val OPEN = "<tool_call>"
    const val CLOSE = "</tool_call>"

    /** One parsed call. [prompt] is the user-facing description of the image. */
    data class Call(val prompt: String)

    /**
     * The system-prompt section describing the tool. Kept short: system
     * prompts are prefilled on every model load, and small models degrade
     * quickly with long instructions.
     */
    fun systemPromptSection(
        defaultWidth: Int,
        defaultHeight: Int,
        defaultSteps: Int,
        defaultCfg: Float,
        lora: String?
    ): String = buildString {
        appendLine("## Tool: $NAME")
        appendLine("You can draw an image on this device by calling the $NAME tool.")
        appendLine("When the user asks for a picture, photo, drawing, painting, poster or logo, call the tool INSTEAD of describing the image in words.")
        appendLine("Emit exactly one line and nothing after it:")
        appendLine("""$OPEN{"name":"$NAME","arguments":{"prompt":"<detailed English visual description>"}}$CLOSE""")
        appendLine("Write the prompt in English, describing subject, style, composition, colours and lighting. Do not add captions or text inside the image.")
        appendLine("The app renders the image with these defaults: ${defaultWidth}x$defaultHeight px, $defaultSteps steps, CFG $defaultCfg" +
            (lora?.let { ", LoRA $it" } ?: "") + ".")
        appendLine("After the tool result you may add one short sentence to the user. Never invent a tool result yourself.")
        appendLine("If the user asks for text, code, analysis or an edit to a previous picture, answer normally and do not call the tool.")
    }.trimEnd()

    /**
     * Finds the first tool call in [text]. Returns null when there is none, so
     * callers can keep streaming.
     */
    fun parse(text: String): Call? {
        val open = text.indexOf(OPEN)
        if (open >= 0) {
            val close = text.indexOf(CLOSE, open + OPEN.length)
            val body = if (close >= 0) {
                text.substring(open + OPEN.length, close)
            } else {
                // Small models regularly forget the closing tag; take the rest
                // of the text once it looks complete enough to parse.
                text.substring(open + OPEN.length)
            }
            body.toCall()?.let { return it }
        }
        // Fallback: a bare JSON object mentioning the tool, without tags.
        val bare = text.indexOf("{\"name\"")
        if (bare >= 0) {
            val end = text.lastIndexOf('}')
            if (end > bare) text.substring(bare, end + 1).toCall()?.let { return it }
        }
        return null
    }

    /** True when [text] ends in the middle of a tool call (keep buffering). */
    fun isPartialCall(text: String): Boolean {
        if (text.contains(CLOSE)) return false
        val open = text.indexOf(OPEN)
        if (open < 0) return false
        return text.indexOf('}', open) < 0
    }

    /** Removes every tool call from [text] so it can be rendered. */
    fun strip(text: String): String {
        if (!text.contains(OPEN) && !text.contains("{\"name\"")) return text
        var out = text
        while (true) {
            val open = out.indexOf(OPEN)
            if (open < 0) break
            val close = out.indexOf(CLOSE, open)
            out = if (close >= 0) {
                out.removeRange(open, close + CLOSE.length)
            } else {
                out.substring(0, open)
            }
        }
        val bare = out.indexOf("{\"name\"")
        if (bare >= 0) {
            val end = out.lastIndexOf('}')
            if (end > bare) out = out.removeRange(bare, end + 1)
        }
        return out.trim()
    }

    /**
     * The assistant turn appended after an image turn so the conversation
     * stays coherent for the LLM (the picture itself is not tokens).
     */
    fun formatImageReply(prompt: String, filename: String): String =
        "[image generated: $prompt] ($filename)"

    /**
     * Minimal reader for the call object.
     *
     * Deliberately not org.json: the format is fixed and tiny, a hand parser
     * keeps this file free of Android dependencies (unit testable on the JVM)
     * and it tolerates trailing commas / missing quotes that small models
     * emit, which a strict JSON parser would reject.
     */
    internal fun String.toCall(): Call? {
        // Small models love full-width punctuation ("prompt"： "…") and
        // Chinese quotes, which strict JSON rejects. Normalise the delimiter
        // characters only; the values themselves are left untouched.
        val normalized = normalizePunctuation(trim())
        val body = normalized.ifBlank { return null }
        val name = field(body, "name") ?: field(body, "tool") ?: return null
        if (!name.trim().equals(NAME, ignoreCase = true)) return null
        // Look for the prompt first inside an "arguments" object, then at top
        // level. Arguments given as a JSON string are handled by the second
        // pass over the raw body.
        val argsAt = listOf("\"arguments\"", "'arguments'", "\"args\"", "'args'")
            .map { body.indexOf(it) }
            .filter { it >= 0 }
            .minOrNull() ?: -1
        val argsStart = if (argsAt >= 0) body.indexOf('{', argsAt) else -1
        val prompt = if (argsStart >= 0) {
            val argsEnd = matchingBrace(body, argsStart)
            if (argsEnd > argsStart) {
                field(body.substring(argsStart, argsEnd + 1), "prompt")
                    ?: field(body.substring(argsStart, argsEnd + 1), "query")
            } else {
                null
            }
        } else {
            field(body, "prompt")
        } ?: field(body, "prompt")
        val clean = prompt?.trim()?.removeSurrounding("\"")?.trim().orEmpty()
        return if (clean.isBlank()) null else Call(clean)
    }

    /**
     * Full-width structural punctuation in an otherwise JSON-ish object:
     * `：` -> `:`, `“”` -> `"`, `，` -> `,`, and full-width braces/brackets.
     * Content inside a value string is preserved except for the quote
     * characters, which cannot be told apart cheaply - the cost is a prompt
     * where a Chinese quotation mark becomes an ASCII one.
     */
    internal fun normalizePunctuation(text: String): String {
        val sb = StringBuilder(text.length)
        for (c in text) {
            sb.append(
                when (c) {
                    '\uFF1A' -> ':'   // ：
                    '\uFF0C' -> ','   // ，
                    '\uFF1B' -> ';'   // ；
                    '\uFF5B' -> '{'   // ｛
                    '\uFF5D' -> '}'   // ｝
                    '\uFF02' -> '"'   // ＂
                    '\u201C', '\u201D' -> '"' // “ ”
                    '\u2018', '\u2019' -> '\'' // ‘ ’
                    else -> c
                }
            )
        }
        return sb.toString()
    }

    /** Value of the first `"key": "value"` (or unquoted value) pair. */
    private fun field(json: String, key: String): String? {
        return fieldQuoted(json, "\"$key\"") ?: fieldQuoted(json, "'$key'")
    }

    private fun fieldQuoted(json: String, marker: String): String? {
        var from = 0
        while (true) {
            val at = json.indexOf(marker, from)
            if (at < 0) return null
            var i = at + marker.length
            while (i < json.length && json[i].isWhitespace()) i++
            if (i < json.length && json[i] == ':') {
                i++
                while (i < json.length && json[i].isWhitespace()) i++
                if (i >= json.length) return null
                if (json[i] == '"' || json[i] == '\'') {
                    val end = closingQuote(json, i + 1, json[i])
                    return json.substring(i + 1, if (end < 0) json.length else end)
                }
                val end = json.indexOfAny(charArrayOf(',', '}', '\n'), i).let { if (it < 0) json.length else it }
                return json.substring(i, end)
            }
            from = at + marker.length
        }
    }

    private fun closingQuote(text: String, from: Int, quote: Char = '"'): Int {
        var i = from
        while (i < text.length) {
            if (text[i] == '\\') { i += 2; continue }
            if (text[i] == quote) return i
            i++
        }
        return -1
    }

    private fun matchingBrace(text: String, open: Int): Int {
        var depth = 0
        var inString = false
        var i = open
        while (i < text.length) {
            val c = text[i]
            when {
                inString && c == '\\' -> i++
                c == '"' -> inString = !inString
                !inString && c == '{' -> depth++
                !inString && c == '}' -> {
                    depth--
                    if (depth == 0) return i
                }
            }
            i++
        }
        return -1
    }
}
