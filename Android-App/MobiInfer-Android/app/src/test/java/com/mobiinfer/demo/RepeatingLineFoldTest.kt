package com.mobiinfer.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RepeatingLineFoldTest {
    @Test
    fun foldsRepeatedThreadtimeLines() {
        val fold = DebugLogStore.RepeatingLineFold()
        val line = "10-29 15:18:21.123  1234  5678 I MobiInfer: repeated"

        assertNull(fold.accept(line))
        assertNull(fold.accept(line))
        assertEquals(
            line + "\n10-29 15:18:21.123     DEBUG repeating line 1 more time(s)",
            fold.accept("10-29 15:18:22.123  1234  5678 I MobiInfer: next"),
        )
    }

    @Test
    fun filtersKnownNoiseTags() {
        val fold = DebugLogStore.RepeatingLineFold()

        assertNull(fold.accept("10-29 15:18:21.123  1234  5678 I Activity: ignored"))
        assertNull(fold.accept("10-29 15:18:21.123  1234  5678 I VRI[MainActivity]: ignored"))
    }
}
