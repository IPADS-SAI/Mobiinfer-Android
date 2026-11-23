// Host-side unit test for the dense narrow-batch dispatch rule (ggml-hexagon).
//
//   g++ -std=c++11 -O2 -I ggml/src/ggml-hexagon tests/test-hexagon-mm-dispatch.cpp -o /tmp/t && /tmp/t
//
// The rule only selects between two already-tested kernels (HVX tiled 32x1/32x2 vs HMX 2D),
// so the whole risk surface is the decision itself -- which is why it is factored into
// htp-mm-dispatch.h and tested here on the development host instead of only on the DSP.
// The expected values encode the two on-device calibrations:
//   * vivo V2546A, Hexagon v81, hwinfo threads 10 / hvx 8 -> crossover m* = 6, rule fires at m = 5
//     (measured -12.8% on the 2048x248144 q6_K vocab projection at m = 5)
//   * OnePlus 13 / P0110 class, Hexagon v79, hwinfo threads 8 / hvx 6 -> crossover m* <= 4,
//     rule must stay inactive (measured neutral: vocab m=5 -0.16%, audit ABBA 3+3)

#include "htp-mm-dispatch.h"

#include <cstdio>
#include <cstdlib>

static int g_failures = 0;

#define CHECK_EQ(expr, want) do {                                                    \
        const int got_ = (expr);                                                     \
        if (got_ != (want)) {                                                        \
            std::printf("FAIL %s:%d  %s = %d, want %d\n", __FILE__, __LINE__,        \
                        #expr, got_, (int) (want));                                  \
            g_failures++;                                                            \
        }                                                                            \
    } while (0)

#define CHECK_TRUE(expr) do {                                                        \
        if (!(expr)) {                                                               \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #expr);              \
            g_failures++;                                                            \
        }                                                                            \
    } while (0)

int main() {
    // --- derived crossover ---------------------------------------------------------------
    // vivo v81 (hvx 8): measured crossover 6, one row of margin -> 5
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(8, -1), 5);
    // v79 with 6 HVX threads (P0110 / OnePlus 13 class): crossover 4 -> rule inactive
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(6, -1), 4);
    // fewer HVX threads than that must not enable the rule either
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(4, -1), 4);
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(2, -1), 4);
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(0, -1), 4);
    // more HVX threads: the crossover may only grow, and it stays below the HMX hard-win zone
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(10, -1), 6);
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(12, -1), 8);
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(16, -1), 8);  // clamped
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(64, -1), 8);  // clamped

    // monotone in n_hvx and always inside [HMX_MIN_NROWS, 8]
    int prev = 0;
    for (int n_hvx = 1; n_hvx <= 32; n_hvx++) {
        const int mr = htp_mm_dispatch_dense_hvx_max_rows(n_hvx, -1);
        CHECK_TRUE(mr >= HTP_MM_DISPATCH_HMX_MIN_NROWS && mr <= 8);
        CHECK_TRUE(mr >= prev);
        prev = mr;
    }

    // --- env override (measurement knob) -------------------------------------------------
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(8, 0), 4);  // 0 = rule off (m<=4 is HVX anyway)
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(8, 5), 5);
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(6, 5), 5);  // lets the v79 arm sweep the crossover
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(6, 8), 8);
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(6, 12), 8); // clamped
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(6, 3), 4);  // clamped up
    CHECK_EQ(htp_mm_dispatch_dense_hvx_max_rows(6, -2), 4); // any negative = derive

    // --- full predicate ------------------------------------------------------------------
    // vivo v81, wide weights, N = 248144 -> 7755 row tiles
    CHECK_TRUE (htp_mm_dispatch_dense_prefer_hvx(8, -1, 5, 7755));  // the measured win
    CHECK_TRUE (!htp_mm_dispatch_dense_prefer_hvx(8, -1, 6, 7755)); // tie at m=6 -> keep HMX
    CHECK_TRUE (!htp_mm_dispatch_dense_prefer_hvx(8, -1, 18, 7755));// HVX 2.7x worse
    // N = 512 -> 16 tiles, exactly the 2-tile-per-thread boundary for 8 HVX threads
    CHECK_TRUE (htp_mm_dispatch_dense_prefer_hvx(8, -1, 5, 16));
    CHECK_TRUE (!htp_mm_dispatch_dense_prefer_hvx(8, -1, 5, 15));   // N = 480: pipeline collapses
    CHECK_TRUE (!htp_mm_dispatch_dense_prefer_hvx(8, -1, 5, 1));    // N = 32: HMX wins
    // v79 / P0110 (hvx 6): default must be a no-op, env=5 must switch it on for measurement
    CHECK_TRUE (!htp_mm_dispatch_dense_prefer_hvx(6, -1, 5, 7755));
    CHECK_TRUE (htp_mm_dispatch_dense_prefer_hvx(6, 5, 5, 7755));
    CHECK_TRUE (!htp_mm_dispatch_dense_prefer_hvx(6, 5, 5, 11));    // < 2*6=12 tiles: still HMX
    CHECK_TRUE (htp_mm_dispatch_dense_prefer_hvx(6, 5, 5, 12));
    CHECK_TRUE (!htp_mm_dispatch_dense_prefer_hvx(6, 5, 6, 7755));  // m above the override

    if (g_failures) {
        std::printf("%d check(s) FAILED\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("test-hexagon-mm-dispatch: all checks passed\n");
    return EXIT_SUCCESS;
}
