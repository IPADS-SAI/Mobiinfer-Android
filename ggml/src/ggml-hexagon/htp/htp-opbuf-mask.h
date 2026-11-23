#ifndef HTP_OPBUF_MASK_H
#define HTP_OPBUF_MASK_H

// 64-bit mmap-reuse bitmap for prep_op_bufs() (T-0326, follow-up of T-0304).
//
// htp_context has HTP_MAX_MMAPS (64) mmap slots.  The reuse mask used to be a
// uint32_t updated with a 32-bit int shift of 1 by i; for i >= 31 that shift is
// undefined (and even at i == 31 it lands in the sign bit), while a 32-bit mask
// can never record slots 32..63.  On x86 the shift count is masked to 5 bits,
// so shifting by 32 silently aliases slot 0: prep_op_bufs() would then consider
// a live mapping unused and drop it, leaving b->base dangling.  This header
// keeps the bitmap operation in one place, uses a 64-bit `1ull` shift, and
// makes out-of-range indices well defined (set/clear are ignored, test false).
//
// Pure header: no DSP/FastRPC/HAP dependencies, no allocation, no I/O.  Shared
// by the DSP skel (C) and host unit tests (C++), same convention as
// htp-opbuf-stats.h.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define HTP_OPBUF_MASK_BITS 64

// Native compile-time assert in each language: _Static_assert is a C11
// keyword that clang warns about in C++ (-Wc11-extensions).
#ifndef HTP_OPBUF_STATIC_ASSERT
#ifdef __cplusplus
#define HTP_OPBUF_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define HTP_OPBUF_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
#endif

// Standalone host tests include this header without htp-ctx.h.  The fallback
// matches htp-ctx.h:20; when the real header comes first the assert below
// checks the actual value.
#ifndef HTP_MAX_MMAPS
#define HTP_MAX_MMAPS 64
#endif

HTP_OPBUF_STATIC_ASSERT(HTP_MAX_MMAPS <= HTP_OPBUF_MASK_BITS,
                        "HTP_MAX_MMAPS must fit in the 64-bit opbuf reuse mask");

// Mark mmap slot i as reused.  i >= 64 is ignored (no shift, no UB).
static inline void htp_opbuf_mask_set(uint64_t * m, uint32_t i) {
    if (m == NULL || i >= HTP_OPBUF_MASK_BITS) {
        return;
    }
    *m |= (1ull << i);
}

// True iff mmap slot i is marked reused.  i >= 64 is false (no shift, no UB).
static inline bool htp_opbuf_mask_test(uint64_t m, uint32_t i) {
    if (i >= HTP_OPBUF_MASK_BITS) {
        return false;
    }
    return (m & (1ull << i)) != 0;
}

// Clear mmap slot i.  i >= 64 is ignored.
static inline void htp_opbuf_mask_clear(uint64_t * m, uint32_t i) {
    if (m == NULL || i >= HTP_OPBUF_MASK_BITS) {
        return;
    }
    *m &= ~(1ull << i);
}

#endif // HTP_OPBUF_MASK_H
