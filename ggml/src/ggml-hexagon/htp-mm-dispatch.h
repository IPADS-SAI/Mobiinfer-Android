#ifndef HTP_MM_DISPATCH_H
#define HTP_MM_DISPATCH_H

// Pure decision logic for the dense (MUL_MAT / MUL_MAT+ADD) narrow-batch dispatch.
//
// Kept header-only and dependency-free so it can be unit tested on the development host
// (see tests/test-hexagon-mm-dispatch.cpp) instead of only on the DSP.
//
// Why a rule is needed at all
// ---------------------------
// The two dense kernels scale differently with the number of activation rows m:
//   - HVX tiled dots the quantised weight tiles straight from DDR: DDR-bound for m <= 3
//     (a wide q6_K GEMV reaches ~70 of the ~72.5 GB/s peak) and instruction throughput
//     bound from m = 4 up (a 2048->8192 q4_K GEMV runs at ~43 GB/s).
//   - HMX first dequantises every weight tile to f16 in VTCM before the tensor core reads
//     it, so its cost is nearly flat in m for thin batches: the 444.7 MB q6_K vocab
//     projection costs ~9.5 ms at m = 5 and ~10.3 ms at m = 18 (an effective ~47 GB/s).
// The crossover is where the HVX instruction cost (linear in m) meets that flat HMX cost.
// Measured on SM8850 / Hexagon v81 with 8 HVX threads, same binary,
// GGML_HEXAGON_MM_SELECT=2 (HMX for m >= 5) vs =1 (always HVX):
//   2048x248144 q6_K : m=5 HMX 9540 us vs HVX 8316 us (-12.8%); m=6 is a tie;
//                      m=18 HMX 10269 us vs HVX 28157 us (HVX 2.7x worse: keep HMX)
//   2048x8192   q6_K : m=5 HMX  325 us vs HVX  288 us (-11.5%)
// The HVX term scales as 1/n_hvx while the HMX term does not, so the crossover follows the
// number of HVX threads: m* ~= 6 * n_hvx / 8. One row of margin is taken because m = 6 is
// already a tie on 8 HVX threads.
//
// Second condition: HVX row-tile parallelism. The HVX dense kernel hands whole 32-row
// weight tiles to threads, so with one tile per thread the DMA pipeline collapses and HMX
// wins even for a wide-but-thin GEMV (2048x32 at m = 5: HMX 9.4 us vs HVX 22.1 us).
// The rule therefore also requires >= 2 row tiles per HVX thread (N >= 64 * n_hvx).

// Smallest number of activation rows that the HMX dense path accepts (HTP_MM_HMX_MIN_NROWS).
// Dense MUL_MAT with m <= 4 already goes HVX without any rule, so this is also the lower
// bound of "rows the rule can affect".
#define HTP_MM_DISPATCH_HMX_MIN_NROWS 4

// Rows per weight tile in the hexagon-tiled layout (HTP_TENSOR_TILE_N_ROWS).
#define HTP_MM_DISPATCH_TILE_N_ROWS 32

// Largest m for which the dense HVX path should be tried before HMX.
//
// n_hvx         : HVX-capable threads reported by the DSP (hwinfo)
// env_override  : -1 = derive from n_hvx, 0 = never prefer HVX, >0 = explicit value
//
// Returns a value in [HTP_MM_DISPATCH_HMX_MIN_NROWS, 8]; m <= 4 is HVX anyway, so returning
// 4 is equivalent to "rule inactive". Clamping to the minimum is deliberate: on devices with
// fewer HVX threads than the 8 that were measured, the predicted crossover is <= 4 and the
// rule must stay out of the way until a measurement says otherwise (the env override exists
// exactly for that measurement).
static inline int htp_mm_dispatch_dense_hvx_max_rows(int n_hvx, int env_override) {
    if (env_override >= 0) {
        int max_rows = env_override;
        if (max_rows > 8) {
            max_rows = 8;
        }
        return max_rows < HTP_MM_DISPATCH_HMX_MIN_NROWS ? HTP_MM_DISPATCH_HMX_MIN_NROWS : max_rows;
    }
    if (n_hvx <= 0) {
        return HTP_MM_DISPATCH_HMX_MIN_NROWS;  // no HVX threads: never prefer HVX
    }
    const int crossover = (6 * n_hvx) / 8;     // 8 HVX threads -> 6
    const int max_rows  = crossover - 1;       // margin: m = 6 is a tie on 8 HVX threads
    if (max_rows < HTP_MM_DISPATCH_HMX_MIN_NROWS) {
        return HTP_MM_DISPATCH_HMX_MIN_NROWS;
    }
    return max_rows > 8 ? 8 : max_rows;
}

// Should the dense HVX path be attempted (and, if it fits VTCM, preferred over HMX)?
//
// m            : activation rows (src1->ne[1] collapsed, i.e. n_tokens of the batch)
// n_row_tiles  : weight row tiles = ceil(N / 32), N = src0->ne[1]
static inline bool htp_mm_dispatch_dense_prefer_hvx(int n_hvx, int env_override, int m, int n_row_tiles) {
    if (m > htp_mm_dispatch_dense_hvx_max_rows(n_hvx, env_override)) {
        return false;
    }
    return n_row_tiles >= 2 * n_hvx;
}

#endif  // HTP_MM_DISPATCH_H
