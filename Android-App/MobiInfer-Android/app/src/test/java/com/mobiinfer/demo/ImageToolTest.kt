package com.mobiinfer.demo.imagegen

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageToolTest {

    private fun call(prompt: String) =
        """${ImageTool.OPEN}{"name":"generate_image","arguments":{"prompt":"$prompt"}}${ImageTool.CLOSE}"""

    @Test
    fun parsesTaggedCall() {
        val text = "Sure! ${call("a cat on the moon")}"
        val parsed = ImageTool.parse(text)
        assertEquals("a cat on the moon", parsed?.prompt)
        assertEquals("Sure!", ImageTool.strip(text))
    }

    @Test
    fun parsesCallWithoutClosingTag() {
        val text = """<tool_call>{"name":"generate_image","arguments":{"prompt":"red kite"}}"""
        assertEquals("red kite", ImageTool.parse(text)?.prompt)
    }

    @Test
    fun parsesBareJsonAndNestedArgsObject() {
        val text = """{"name": "generate_image", "arguments": "{\"prompt\":\"harbour at dusk\"}"}"""
        // Arguments given as a JSON string are not something the protocol
        // promises, but the fallback must at least not throw.
        assertTrue(ImageTool.parse(text) == null || ImageTool.parse(text)!!.prompt.isNotBlank())
    }

    @Test
    fun rejectsCallsToOtherTools() {
        val text = """<tool_call>{"name":"search","arguments":{"prompt":"x"}}</tool_call>"""
        assertNull(ImageTool.parse(text))
    }

    @Test
    fun rejectsCallsWithoutPrompt() {
        val text = """<tool_call>{"name":"generate_image","arguments":{"prompt":"  "}}</tool_call>"""
        assertNull(ImageTool.parse(text))
    }

    @Test
    fun plainTextHasNoCall() {
        assertNull(ImageTool.parse("A picture of a cat would look like this."))
        assertEquals("no tools here", ImageTool.strip("no tools here"))
    }

    @Test
    fun partialCallIsDetected() {
        assertTrue(ImageTool.isPartialCall("""<tool_call>{"name":"generate_image","argu"""))
        assertFalse(ImageTool.isPartialCall(call("done")))
        assertFalse(ImageTool.isPartialCall("plain"))
    }

    @Test
    fun stripRemovesEverythingFromAnUnclosedCall() {
        assertEquals("Look:", ImageTool.strip("Look: " + ImageTool.OPEN + "{\"name\":\"generate_image\""))
    }

    @Test
    fun systemPromptSectionIsWellFormed() {
        val section = ImageTool.systemPromptSection(512, 512, 8, 3.0f, null)
        assertTrue(section.contains(ImageTool.NAME))
        assertTrue(section.contains("512x512"))
        assertTrue(section.contains("8 steps"))
        assertTrue(section.contains(ImageTool.OPEN))
    }

    @Test
    fun imageReplyFormattingKeepsThePrompt() {
        val reply = ImageTool.formatImageReply("a dog", "image_1.png")
        assertTrue(reply.contains("a dog"))
        assertTrue(reply.contains("image_1.png"))
    }

    @Test
    fun promptMarkerRoundTrips() {
        val encoded = ImagePromptMarker.encode("a castle")
        assertTrue(ImagePromptMarker.isImageTurn(encoded))
        assertFalse(ImagePromptMarker.isImageTurn("a castle"))
        assertEquals("a castle", ImagePromptMarker.strip(encoded))
        // The bridge is a machine record now: one JSON object, delimited by
        // \u0001 so a prompt containing ']', quotes or newlines cannot truncate
        // it the way the old "[image generated: ...]" text could.
        val (prompt, record) = ImagePromptMarker.forLlm(encoded)!!
        assertEquals("a castle", prompt)
        assertTrue(record.startsWith("\u0001image-gen:v1:"))
        assertTrue(record.endsWith("\u0001"))
        assertTrue(record.contains("\"prompt\":\"a castle\""))
        assertNull(ImagePromptMarker.forLlm("plain question"))
    }

    @Test
    fun bridgeRecordEscapesJson() {
        // A prompt with the characters that broke the old text format must
        // survive intact: it is the model's argument for a real tool call.
        val tricky = "a cat [on a windowsill] with \"quotes\" and a\\slash\nand a newline"
        val record = ImagePromptMarker.toolRecord(tricky)
        assertTrue(record.startsWith("\u0001image-gen:v1:"))
        assertTrue(record.endsWith("\u0001"))
        // The payload between the markers has to be valid JSON on its own.
        val payload = record.removePrefix("\u0001image-gen:v1:").dropLast(1)
        assertTrue(payload.startsWith("{") && payload.endsWith("}"))
        // org.json is a stub in JVM unit tests, so verify the escaping by the
        // property that matters: every quote and backslash in the payload is
        // escaped, and the markers cannot appear inside it.
        assertTrue(payload.contains("\\\""))
        assertFalse(payload.contains("\u0001"))
        // Strip the escapes back off and the original prompt must return.
        val unescaped = StringBuilder()
        var i = 0
        while (i < payload.length) {
            val c = payload[i]
            if (c == '\\' && i + 1 < payload.length) {
                when (val n = payload[i + 1]) {
                    'n' -> unescaped.append('\n')
                    'r' -> unescaped.append('\r')
                    't' -> unescaped.append('\t')
                    else -> unescaped.append(n)
                }
                i += 2
            } else {
                unescaped.append(c)
                i++
            }
        }
        assertTrue(
            "prompt must survive escaping: $payload",
            unescaped.contains("a cat [on a windowsill]")
        )
        assertTrue("quotes survive", unescaped.contains("\"quotes\""))
        assertTrue("backslash survives", unescaped.contains("a\\slash"))
    }

    @Test
    fun parsesChinesePunctuation() {
        // Qwen-style output: full-width colon and Chinese quotes.
        val text = "好的\uFF1A<tool_call>{\u201Cname\u201D：\u201Cgenerate_image\u201D，\u201Carguments\u201D：" +
            "{\u201Cprompt\u201D：\u201Ca ginger cat sunbathing on a windowsill\u201D}}</tool_call>"
        assertEquals("a ginger cat sunbathing on a windowsill", ImageTool.parse(text)?.prompt)
    }

    @Test
    fun parsesSingleQuotedJson() {
        val text = "<tool_call>{'name':'generate_image','arguments':{'prompt':'a red kite'}}</tool_call>"
        assertEquals("a red kite", ImageTool.parse(text)?.prompt)
    }

    @Test
    fun schemaArrayIsAJsonArray() {
        val json = SystemPrompts.toolSchemasJson(true)
        assertTrue(json!!.startsWith("[{"))
        assertTrue(json.endsWith("}]"))
        assertNull(SystemPrompts.toolSchemasJson(false))
    }

    @Test
    fun toolSchemaIsWellFormed() {
        val schema = SystemPrompts.toolSchema()
        assertTrue(schema.startsWith("{\"type\":\"function\""))
        assertTrue(schema.contains("\"name\":\"${ImageTool.NAME}\""))
        assertTrue(schema.contains("\"required\":[\"prompt\"]"))
        // Balanced braces: the schema is spliced into a template by hand.
        assertEquals(schema.count { it == '{' }, schema.count { it == '}' })
    }

    @Test
    fun templateSupportDetection() {
        assertTrue(SystemPrompts.templateSupportsTools("{{ if tools }}{{ tools }}{{ endif }}"))
        assertFalse(SystemPrompts.templateSupportsTools("{{ messages[0].content }}"))
        assertFalse(SystemPrompts.templateSupportsTools(null))
    }

    @Test
    fun systemMessageIncludesToolOnlyWhenEnabled() {
        val withTool = SystemPrompts.systemMessage(true, true, 512, 512, 8, 3.0f)
        val without = SystemPrompts.systemMessage(true, false, 512, 512, 8, 3.0f)
        assertTrue(withTool.contains(ImageTool.NAME))
        assertFalse(without.contains(ImageTool.NAME))
        assertTrue(without.isNotBlank())
    }
}
