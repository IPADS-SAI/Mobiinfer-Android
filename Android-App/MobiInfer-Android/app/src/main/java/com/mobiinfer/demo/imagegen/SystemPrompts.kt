package com.mobiinfer.demo.imagegen

import android.app.ActivityManager
import android.content.Context

/**
 * Builds the system message a chat model is loaded with.
 *
 * The base text is per-language. When the generate_image tool is available it
 * is appended as plain instructions, or - if the model's own chat template
 * already renders `tools` - as a template argument, so the model sees the tool
 * in the exact shape it was trained on. The identity of each call is the same
 * either way: [ImageTool.OPEN]/[ImageTool.CLOSE] wrapped JSON.
 */
object SystemPrompts {

    private const val BASE_ZH = "你是 MobiInfer，一个运行在手机本地的 AI 助手。请用中文回答，简洁准确。"
    private const val BASE_EN = "You are MobiInfer, a helpful assistant running locally on a phone. Answer in the user's language, concisely and accurately."

    /**
     * The OpenAI-compatible tool array handed to a tool-aware template
     * (`[{"type":"function", ...}]`), or null when the tool is off.
     */
    fun toolSchemasJson(toolEnabled: Boolean): String? =
        if (toolEnabled) "[${toolSchema()}]" else null

    /** The assistant persona without any tool instructions. */
    fun base(isChinese: Boolean): String = if (isChinese) BASE_ZH else BASE_EN

    fun systemMessage(
        isChinese: Boolean,
        toolEnabled: Boolean,
        defaultWidth: Int,
        defaultHeight: Int,
        defaultSteps: Int,
        defaultCfg: Float
    ): String {
        if (!toolEnabled) return base(isChinese)
        val section = ImageTool.systemPromptSection(
            defaultWidth = defaultWidth,
            defaultHeight = defaultHeight,
            defaultSteps = defaultSteps,
            defaultCfg = defaultCfg,
            lora = null
        )
        // English tool lines in a Chinese prompt are intentional: models
        // follow tool syntax best in English, and the instruction to answer
        // in the user's language is right above it.
        return base(isChinese) + "\n\n" + section
    }

    /**
     * True when the template renders tool definitions, i.e. it is a real
     * function-calling template rather than a plain chatml template.
     */
    fun templateSupportsTools(template: String?): Boolean {
        if (template.isNullOrBlank()) return false
        return template.contains("tools", ignoreCase = true)
    }

    /**
     * Tool schema passed to templates that support it. The regexes mirror the
     * literal protocol the model is also told about in the system text.
     */
    fun toolSchema(): String =
        """{"type":"function","function":{"name":"${ImageTool.NAME}",""" +
            """"description":"Generate an image on this device from a text prompt. Use it whenever the user asks for a picture.",""" +
            """"parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"Detailed English visual description of the image"}},""" +
            """"required":["prompt"]}}}"""
}

/** Memory helpers around the two model stacks. */
object MemoryBudget {

    /** Advisory heap-ish headroom the system reports as "available". */
    fun availableBytes(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        return info.availMem
    }

    /** Total physical RAM of the device. */
    fun totalBytes(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        return info.totalMem
    }

    /**
     * Whether the text-to-image stack of [imageFootprintBytes] can be loaded
     * alongside the resident language model. Uses the user's configured
     * runtime budget as the language model's share instead of guessing it from
     * the GGUF size, because the expert/bundle layouts differ.
     */
    fun fitsAlongsideLanguageModel(
        context: Context,
        imageFootprintBytes: Long,
        languageBudgetMb: Int
    ): Boolean {
        // Keep a slice for the UI, the bitmap of the result and the allocator.
        val reserve = 1536L * 1024 * 1024
        val need = imageFootprintBytes + reserve
        val free = availableBytes(context)
        val languageModelHold = if (languageBudgetMb > 0) languageBudgetMb.toLong() * 1024 * 1024 else 0L
        return free - languageModelHold >= need
    }
}
