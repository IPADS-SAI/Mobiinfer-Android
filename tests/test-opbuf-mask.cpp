// Host unit test for the 64-bit opbuf mmap-reuse bitmap (T-0326).
//
// Pins the boundary behaviour that the old 32-bit mask with an `int` shift got
// wrong: slots 31/32/63 must be independent, slot 32 must not alias slot 0
// (x86 masks the shift count to 5 bits, so shifting by 32 wraps to 0), and
// out-of-range indices (64/100) must be defined no-ops / false instead of UB.
//
// Compile + run (host, g++ >= C++17), from the repo root:
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror -I ggml/src/ggml-hexagon/htp -o /tmp/test-opbuf-mask tests/test-opbuf-mask.cpp
//   /tmp/test-opbuf-mask
//
// Exit code 0 and "PASS" on success; any failure prints FAIL and exits 1.

#include "htp-opbuf-mask.h"

#include <cstdint>
#include <cstdio>

static int failures = 0;

static void expect_true(const char * what, bool got, int line) {
    if (!got) {
        ++failures;
        std::printf("FAIL %s:%d: %s\n", __FILE__, line, what);
    }
}

static void expect_u64(const char * what, uint64_t got, uint64_t want, int line) {
    if (got != want) {
        ++failures;
        std::printf("FAIL %s:%d: %s = 0x%016llx, want 0x%016llx\n", __FILE__, line, what,
                    (unsigned long long) got, (unsigned long long) want);
    }
}

static void expect_false(const char * what, bool got, int line) {
    if (got) {
        ++failures;
        std::printf("FAIL %s:%d: %s is true, want false\n", __FILE__, line, what);
    }
}

#define EXPECT_TRUE(expr)  expect_true(#expr, (bool) (expr), __LINE__)
#define EXPECT_FALSE(expr) expect_false(#expr, (bool) (expr), __LINE__)
#define EXPECT_U64(expr, want) expect_u64(#expr, (uint64_t) (expr), (uint64_t) (want), __LINE__)

static void test_single_bit_roundtrip(uint32_t i) {
    uint64_t m = 0;
    htp_opbuf_mask_set(&m, i);
    EXPECT_TRUE(htp_opbuf_mask_test(m, i));
    EXPECT_U64(m, 1ull << i);
    htp_opbuf_mask_clear(&m, i);
    EXPECT_FALSE(htp_opbuf_mask_test(m, i));
    EXPECT_U64(m, 0);
}

static void test_neighbours_are_independent() {
    uint64_t m = 0;

    // Setting 31 must not bleed into 32 (and vice versa below).
    htp_opbuf_mask_set(&m, 31);
    EXPECT_U64(m, 1ull << 31);
    EXPECT_TRUE(htp_opbuf_mask_test(m, 31));
    EXPECT_FALSE(htp_opbuf_mask_test(m, 32));

    // Setting 32 must not set 31, 63, or 0.  The 0 check is the regression for
    // `1 << 32` aliasing slot 0 on x86.
    m = 0;
    htp_opbuf_mask_set(&m, 32);
    EXPECT_U64(m, 1ull << 32);
    EXPECT_TRUE(htp_opbuf_mask_test(m, 32));
    EXPECT_FALSE(htp_opbuf_mask_test(m, 31));
    EXPECT_FALSE(htp_opbuf_mask_test(m, 63));
    EXPECT_FALSE(htp_opbuf_mask_test(m, 0));

    // Setting 63 stays in the top bit.
    m = 0;
    htp_opbuf_mask_set(&m, 63);
    EXPECT_U64(m, 1ull << 63);
    EXPECT_TRUE(htp_opbuf_mask_test(m, 63));
    EXPECT_FALSE(htp_opbuf_mask_test(m, 32));
    EXPECT_FALSE(htp_opbuf_mask_test(m, 0));
}

static void test_out_of_range() {
    const uint64_t pattern = 0x0123456789abcdefull;

    uint64_t m = pattern;
    htp_opbuf_mask_set(&m, 64);
    htp_opbuf_mask_set(&m, 100);
    EXPECT_U64(m, pattern);
    EXPECT_FALSE(htp_opbuf_mask_test(m, 64));
    EXPECT_FALSE(htp_opbuf_mask_test(m, 100));

    htp_opbuf_mask_clear(&m, 64);
    htp_opbuf_mask_clear(&m, 100);
    EXPECT_U64(m, pattern);

    // No crash for a null mask either.
    htp_opbuf_mask_set(nullptr, 0);
    htp_opbuf_mask_set(nullptr, 64);
    htp_opbuf_mask_clear(nullptr, 0);
    htp_opbuf_mask_clear(nullptr, 64);
}

static void test_all_bits_then_clear() {
    uint64_t m = 0;
    for (uint32_t i = 0; i < HTP_OPBUF_MASK_BITS; ++i) {
        htp_opbuf_mask_set(&m, i);
    }
    EXPECT_U64(m, ~0ull);
    for (uint32_t i = 0; i < HTP_OPBUF_MASK_BITS; ++i) {
        EXPECT_TRUE(htp_opbuf_mask_test(m, i));
    }

    // Clear every bit and re-test both directions.
    for (uint32_t i = 0; i < HTP_OPBUF_MASK_BITS; ++i) {
        htp_opbuf_mask_clear(&m, i);
        EXPECT_FALSE(htp_opbuf_mask_test(m, i));
    }
    EXPECT_U64(m, 0);

    // Clearing an already-clear bit is idempotent.
    htp_opbuf_mask_clear(&m, 32);
    EXPECT_U64(m, 0);
}

int main() {
    static_assert(HTP_OPBUF_MASK_BITS == 64, "reuse mask must cover HTP_MAX_MMAPS=64");
    static_assert(HTP_MAX_MMAPS <= HTP_OPBUF_MASK_BITS, "HTP_MAX_MMAPS must fit the mask");

    // Boundary indices required by the task: 0, 31, 32, 63.
    test_single_bit_roundtrip(0);
    test_single_bit_roundtrip(31);
    test_single_bit_roundtrip(32);
    test_single_bit_roundtrip(63);

    test_neighbours_are_independent();
    test_out_of_range();
    test_all_bits_then_clear();

    if (failures != 0) {
        std::printf("FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS: opbuf mask 0/31/32/63, independence, out-of-range, all-64\n");
    return 0;
}
