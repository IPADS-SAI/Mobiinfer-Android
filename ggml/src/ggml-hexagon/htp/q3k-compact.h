#ifndef HTP_Q3K_COMPACT_H
#define HTP_Q3K_COMPACT_H

#include <stddef.h>
#include <stdint.h>

// One group covers 32 rows and 256 K values: 8 k32 quant tiles (each 256 B of
// packed low-2-bit quants followed by 128 B of sign bits), then 32 fp16
// multipliers and 12 columns of 32 packed scale bytes.
#define HTP_Q3K_COMPACT_GROUP_BYTES   3520
#define HTP_Q3K_COMPACT_QUANTS_BYTES  3072
#define HTP_Q3K_COMPACT_SCALES_OFFSET 3136

// The layout only covers whole 256-value Q3_K super-blocks. Returns 0 for any
// other K so callers hit their existing slab_size > 0 checks instead of silently
// truncating the slab.
static inline size_t htp_q3k_compact_slab_size(uint32_t k) {
    if (k % 256 != 0) {
        return 0;
    }
    // Only the end of the 32-row slab is padded.
    const uint64_t bytes = ((uint64_t) (k / 256) * HTP_Q3K_COMPACT_GROUP_BYTES + 127) & ~(uint64_t) 127;
    return bytes <= SIZE_MAX ? (size_t) bytes : 0;
}

#endif // HTP_Q3K_COMPACT_H
