#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#include "dma-queue.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-vtcm.h"
#include "htp-tensor.h"
#include <stdlib.h>
#include "work-queue.h"
#include "hmx-utils.h"
#include "hmx-queue.h"
#include "hmx-fa-kernels.h"
#include "conv2dhmx-ops.h"

// HMX direct conv: build the patch rows in VTCM, multiply with HMX.
// Rationale: devdocs/06-perf-profile.md 100 (im2col costs 40.8 GB of DDR) and
// 107.6 (a pure-HVX conv is 67x slower than im2col+HMX).

#define CV_TILE_ROWS 32
#define CV_TILE_COLS 32
#define CV_TILE_ELMS 1024

struct htp_conv2dhmx_ctx {
    const struct htp_conv2dhmx_kernel_params * kp;
    const float *  src;    // [W][H][IC]
    const float *  wf32;   // weight when f32
    const __fp16 * w16;    // weight when f16
    float *        dst;    // [OW][OH][OC]

    uint8_t * vtcm_act;    // [n_dot_tiles][1024] fp16 patch tiles
    uint8_t * vtcm_w;      // [n_dot_tiles][1024] fp16 weight tiles
    uint8_t * vtcm_out;    // one 32x32 fp16 result tile
    uint8_t * vtcm_stage;  // row-major staging for interleave (act or weight)
    void *    vtcm_scales; // 256 B identity column-scale tile for HMX
    int       dump_tiles;
    bool      failed;      // set when the HMX queue could not accept a job
    uint32_t  total_tiles;
    uint32_t  tiles_per_thread;
    uint32_t  stride_per_thread;   // bytes between workers' VTCM slabs
    uint32_t  vtcm_wall_bytes;     // size of the shared weight-tile region
    uint32_t  max_row_tiles;       // row tiles accumulated per HMX pass
    struct htp_context * ctx;
    uint32_t n_pack_threads;
    hmx_queue_t hmx_q;
};

static inline float cv_weight_at(const struct htp_conv2dhmx_ctx * c,
                                 const struct htp_conv2dhmx_kernel_params * kp,
                                 uint32_t kw, uint32_t kh, uint32_t ic, uint32_t oc) {
    // ggml: kw + KW*(kh + KH*(ic + IC*oc))
    const size_t i = (size_t) kw + (size_t) kp->KW * (kh + (size_t) kp->KH * (ic + (size_t) kp->IC * oc));
    return c->w16 ? (float) c->w16[i] : c->wf32[i];
}

static inline float cv_input_at(const struct htp_conv2dhmx_ctx * c,
                                const struct htp_conv2dhmx_kernel_params * kp,
                                int32_t ih, int32_t iw, uint32_t ic) {
    if (ih < 0 || ih >= (int32_t) kp->IH || iw < 0 || iw >= (int32_t) kp->IW) {
        return 0.0f;
    }
    return c->src[((size_t) ic * kp->IH + ih) * kp->IW + iw];
}

// Copy the 32 weight rows of one output-channel tile from DRAM into a row-major
// VTCM staging buffer, via the DMA engine when the weights are already F16.
//
// Why DMA: measured on device (IC=OC=512, 3x3, F16, 32x32), the CPU memcpy form
// of this copy cost ~5.53 ms of the ~5.81 ms weight-pack phase -- about 814 MB/s,
// far below DDR bandwidth, because it issues ~59万 load/store instructions that no
// hardware prefetch or pipelining hides. The matmul path (matmul-ops.c) never
// copies weights on the CPU for the same reason: it hands DRAM->VTCM to
// dma_queue_push and only repacks inside VTCM.
//
// The 32 rows are laid out so the whole tile is describable by one descriptor:
// source rows are kv apart, destination rows patch_stride apart, each row kv
// bytes. When kv == patch_stride (K a multiple of 64, which every 3x3 VAE conv
// with IC=512/256/128/64 satisfies) all three strides are equal, so this lands on
// dma_queue_push's contiguous fast path. F32 weights cannot use DMA: the hardware
// does not convert types, so that case stays on the CPU.
static void cv_copy_weight_rows_dma(dma_queue * dma_q,
                                    __fp16 * restrict stage,
                                    const struct htp_conv2dhmx_ctx * c,
                                    const struct htp_conv2dhmx_kernel_params * kp,
                                    uint32_t oc0) {
    const uint32_t kv      = kp->k_valid;
    const uint32_t n_lanes = MIN(CV_TILE_COLS, kp->OC - oc0);

    // Rows past the end of the output channels must read as zero. Only the tail
    // tile can have them, and the staging buffer is reused across tiles, so this
    // has to be a real memset rather than an assumption about prior contents.
    if (n_lanes < CV_TILE_COLS) {
        memset(stage + (size_t) n_lanes * kp->patch_stride, 0,
               (size_t) (CV_TILE_COLS - n_lanes) * kp->patch_stride * sizeof(__fp16));
    }

    const size_t row_bytes = (size_t) kv * sizeof(__fp16);
    const size_t dst_stride = (size_t) kp->patch_stride * sizeof(__fp16);
    // dma_queue_push returns false when the ring is full; unlike memcpy it is not
    // guaranteed to accept the descriptor. Flushing and retrying is the pattern
    // dma-copy.h uses.
    // Bound the retry: a ring that keeps refusing would otherwise spin here with
    // the HMX queue stalled, which reads on the host as a dead session.
    uint32_t retries = 0;
    while (!dma_queue_push(dma_q,
                           dma_make_data(stage, c->w16 + (size_t) oc0 * kv),
                           dst_stride, row_bytes, row_bytes, n_lanes)) {
        dma_queue_flush(dma_q);
        if (++retries > 1024) {
            for (uint32_t lane = 0; lane < n_lanes; lane++) {
                memcpy(stage + (size_t) lane * kp->patch_stride,
                       c->w16 + (size_t) (oc0 + lane) * kv, row_bytes);
            }
            break;
        }
    }

    // Rows are copied at the real K, so a padded tail left over from a previous
    // tile would corrupt the dot product; zero it while the DMA is in flight.
    if (kp->patch_stride > kv) {
        for (uint32_t lane = 0; lane < n_lanes; lane++) {
            memset(stage + (size_t) lane * kp->patch_stride + kv, 0,
                   (size_t) (kp->patch_stride - kv) * sizeof(__fp16));
        }
    }
    dma_queue_pop(dma_q);
}

// Build [n_col_tiles][n_dot_tiles][1024] fp16 weight tiles for oc0..oc0+31.
static void cv_build_weight_tiles(__fp16 * restrict tiles,
                                  const struct htp_conv2dhmx_ctx * c,
                                  const struct htp_conv2dhmx_kernel_params * kp,
                                  uint32_t oc0) {
    const uint32_t OC = kp->OC;
    for (uint32_t lane = 0; lane < CV_TILE_COLS; lane++) {
        const uint32_t oc = oc0 + lane;
        __fp16 * dst = tiles + (size_t) lane * kp->patch_stride;
        if (oc >= OC) {
            memset(dst, 0, (size_t) kp->patch_stride * sizeof(__fp16));
            continue;
        }
        // The weight rows in DRAM are packed at the real K (k_valid), so the
        // padded tail of the staging row must be zeroed explicitly.
        const uint32_t kv = kp->k_valid;
        if (c->w16) {
            memcpy(dst, c->w16 + (size_t) oc * kv, (size_t) kv * sizeof(__fp16));
            if (kp->patch_stride > kv) {
                memset(dst + kv, 0, (size_t) (kp->patch_stride - kv) * sizeof(__fp16));
            }
            continue;
        }
        for (uint32_t k = 0; k < kv; k += 64) {
            const float * src = c->wf32 + (size_t) oc * kv + k;
            const uint32_t n = MIN(64u, kv - k);
            if (n == 64) {
                hvx_vmem(dst + k) = hvx_vec_f32_to_f16(hvx_vmemu(src), hvx_vmemu(src + 32));
            } else {
                for (uint32_t j = 0; j < n; j++) dst[k + j] = (__fp16) src[j];
                for (uint32_t j = n; j < 64; j++) dst[k + j] = (__fp16) 0.0f;
            }
        }
        if (kp->patch_stride > kv) {
            memset(dst + kv, 0, (size_t) (kp->patch_stride - kv) * sizeof(__fp16));
        }

    }
}

// Build [n_dot_tiles][1024] fp16 activation tiles for 32 patch rows.
static void cv_build_act_tiles(__fp16 * restrict tiles,
                               const struct htp_conv2dhmx_ctx * c,
                               const struct htp_conv2dhmx_kernel_params * kp,
                               uint32_t row0, uint32_t rows) {
    const uint32_t IC = kp->IC, KH = kp->KH, KW = kp->KW;
    const uint32_t OW = kp->OW, OH = kp->OH;

    // The staging buffer always describes a full 32-row tile, and HMX reads all
    // 32 rows. Rows past the end of the output grid must therefore be zeroed
    // explicitly: a partial tile would otherwise hand HMX whatever the previous
    // batch left in VTCM, which lands in the output as a huge garbage value
    // rather than as an obvious out-of-range read.
    for (uint32_t lane = rows; lane < CV_TILE_ROWS; lane++) {
        __fp16 * dst = tiles + (size_t) lane * kp->patch_stride;
        for (uint32_t k = 0; k < kp->patch_stride; k++) {
            dst[k] = (__fp16) 0.0f;
        }
    }

    for (uint32_t lane = 0; lane < rows; lane++) {
        const uint32_t r = row0 + lane;
        if (r >= kp->total_rows) {
            break;
        }
        const uint32_t ow = r % OW;
        const uint32_t oh = (r / OW) % OH;
        const uint32_t n  = r / (OW * OH);
        const float * batch = c->src + (size_t) n * IC * kp->IH * kp->IW;

        // ggml stores the activation as [W][H][IC][N], and ne[0] is the
        // lowest, most contiguous dimension, so the flat index is
        // ((n*IC + ic)*IH + ih)*IW + iw: W is contiguous, then H, then C.
        // The patch column k must match the weight's own k ordering, which for
        // a kernel tensor of ne = [KW, KH, IC, OC] is
        // kw + KW*(kh + KH*ic), i.e. a stride of KW*KH per input channel.
        for (uint32_t kh = 0; kh < KH; kh++) {
            const int32_t ih = (int32_t) oh * kp->stride + (int32_t) kh - kp->pad;
            const bool ih_ok = (ih >= 0) && (ih < (int32_t) kp->IH);
            for (uint32_t kw = 0; kw < KW; kw++) {
                const int32_t iw = (int32_t) ow * kp->stride + (int32_t) kw - kp->pad;
                const bool ok = ih_ok && (iw >= 0) && (iw < (int32_t) kp->IW);
                __fp16 * dst = tiles + (size_t) lane * kp->patch_stride + kw + KW * kh;
                if (!ok) {
                    for (uint32_t ic = 0; ic < IC; ic++) {
                        dst[(size_t) ic * KW * KH] = (__fp16) 0.0f;
                    }
                    continue;
                }
                // Stride IH*IW between channels: this walk is channel-outer by
                // necessity, because C is the *least* contiguous axis of a
                // [W][H][C] tensor. Vectorising the ic loop would need a gather.
                const float * src = batch + (size_t) ih * kp->IW + iw;
                for (uint32_t ic = 0; ic < IC; ic++) {
                    dst[(size_t) ic * KW * KH] = (__fp16) src[(size_t) ic * kp->IH * kp->IW];
                }
            }
        }
    }
}

static void cv_pack_act_tiles(__fp16 * restrict tiles,
                              const struct htp_conv2dhmx_ctx * c,
                              HVX_Vector * restrict matrix,
                              uint32_t row0, uint32_t rows) {
    const struct htp_conv2dhmx_kernel_params * kp = c->kp;
    const uint32_t ow = row0 % kp->OW;
    const uint32_t oh = (row0 / kp->OW) % kp->OH;
    const uint32_t n = row0 / (kp->OW * kp->OH);
    const HVX_Vector zero = Q6_V_vzero();
    float edge[32] __attribute__((aligned(128)));
    struct {
        uint32_t lane, length, n, oh, ow;
    } segments[32];
    uint32_t n_segments = 0;
    if (kp->stride == 1 && ow + rows > kp->OW) {
        uint32_t cursor = row0;
        uint32_t remaining = rows;
        while (remaining) {
            const uint32_t col = cursor % kp->OW;
            const uint32_t length = MIN(remaining, kp->OW - col);
            segments[n_segments].lane = rows - remaining;
            segments[n_segments].length = length;
            segments[n_segments].n = cursor / (kp->OW * kp->OH);
            segments[n_segments].oh = (cursor / kp->OW) % kp->OH;
            segments[n_segments].ow = col;
            n_segments++;
            cursor += length;
            remaining -= length;
        }
    }
    for (uint32_t ic = 0; ic < kp->IC; ic++) {
        for (uint32_t kh = 0; kh < kp->KH; kh++) {
            for (uint32_t kw = 0; kw < kp->KW; kw++) {
                const uint32_t k = (ic * kp->KH + kh) * kp->KW + kw;
                const int32_t ih = (int32_t) (oh * kp->stride + kh) - kp->pad;
                const int32_t iw = (int32_t) (ow * kp->stride + kw) - kp->pad;
                HVX_Vector v;
                if (rows == 32 && kp->stride == 1 && ow + 32 <= kp->OW &&
                    ih >= 0 && ih < (int32_t) kp->IH && iw >= 0 && iw + 32 <= (int32_t) kp->IW) {
                    v = hvx_vmemu(c->src + (((size_t) n * kp->IC + ic) * kp->IH + ih) * kp->IW + iw);
                } else if (kp->stride == 1 && ow + rows <= kp->OW) {
                    v = zero;
                    if (ih >= 0 && ih < (int32_t) kp->IH) {
                        memset(edge, 0, sizeof(edge));
                        const int32_t first = iw < 0 ? -iw : 0;
                        const int32_t last = MIN((int32_t) rows, (int32_t) kp->IW - iw);
                        if (last > first) {
                            const float * src = c->src + (((size_t) n * kp->IC + ic) * kp->IH + ih) * kp->IW + (iw + first);
                            memcpy(edge + first, src, (size_t) (last - first) * sizeof(float));
                        }
                        v = hvx_vmem(edge);
                    }
                } else if (kp->stride == 1) {
                    memset(edge, 0, sizeof(edge));
                    for (uint32_t s = 0; s < n_segments; s++) {
                        const int32_t y = (int32_t) (segments[s].oh + kh) - kp->pad;
                        if (y < 0 || y >= (int32_t) kp->IH) {
                            continue;
                        }
                        const int32_t x = (int32_t) (segments[s].ow + kw) - kp->pad;
                        const int32_t first = x < 0 ? -x : 0;
                        const int32_t last = MIN((int32_t) segments[s].length, (int32_t) kp->IW - x);
                        if (last > first) {
                            const float * src = c->src +
                                (((size_t) segments[s].n * kp->IC + ic) * kp->IH + y) * kp->IW + (x + first);
                            memcpy(edge + segments[s].lane + first, src, (size_t) (last - first) * sizeof(float));
                        }
                    }
                    v = hvx_vmem(edge);
                } else {
                    for (uint32_t r = 0; r < 32; r++) {
                        float value = 0.0f;
                        if (r < rows) {
                            const uint32_t m = row0 + r;
                            const uint32_t bn = m / (kp->OW * kp->OH);
                            const int32_t y = (int32_t) (((m / kp->OW) % kp->OH) * kp->stride + kh) - kp->pad;
                            const int32_t x = (int32_t) ((m % kp->OW) * kp->stride + kw) - kp->pad;
                            if (y >= 0 && y < (int32_t) kp->IH && x >= 0 && x < (int32_t) kp->IW) {
                                value = c->src[(((size_t) bn * kp->IC + ic) * kp->IH + y) * kp->IW + x];
                            }
                        }
                        edge[r] = value;
                    }
                    v = hvx_vmem(edge);
                }
                matrix[k % 32] = hvx_vec_f32_to_f16(v, zero);
                if (k % 32 == 31) {
                    hvx_transpose_32x32_w_shuff((uint8_t *) (tiles + (size_t) (k / 32) * 1024),
                                              128, matrix, 16, 32);
                }
            }
        }
    }
    // Flush a trailing partial dot tile. The transpose above only fires on
    // k % 32 == 31, so when K is not a multiple of 32 the last group of k values
    // is left in `matrix` and never written. The unused lanes still hold the
    // previous group, so they must be re-zeroed before the transpose.
    const uint32_t k_rem = kp->k_valid % 32;
    if (k_rem) {
        const HVX_Vector zero2 = Q6_V_vzero();
        for (uint32_t t = k_rem; t < 32; t++) {
            matrix[t] = zero2;
        }
        hvx_transpose_32x32_w_shuff((uint8_t *) (tiles + (size_t) (kp->k_valid / 32) * 1024),
                                  128, matrix, 16, 32);
    }
}

struct cv_pack_job {
    struct htp_conv2dhmx_ctx * c;
    __fp16 * act;
    __fp16 * scratch;
    uint32_t batch;
    uint32_t count;
    size_t scratch_stride;
};

static void cv_pack_worker(unsigned int nth, unsigned int ith, void * data) {
    struct cv_pack_job * j = (struct cv_pack_job *) data;
    const struct htp_conv2dhmx_kernel_params * kp = j->c->kp;
    for (uint32_t i = ith; i < j->count; i += nth) {
        const uint32_t row0 = (j->batch + i) * 32;
        __fp16 * base = j->act + (size_t) i * kp->n_dot_tiles * 1024;
        // The packer only writes the ceil(k_valid/32) tiles that hold real taps.
        // When K was padded up to a multiple of 64 the remaining dot tile must be
        // zero, otherwise stale VTCM participates in the dot product.
        const uint32_t n_dot_valid = (kp->k_valid + 31u) / 32u;
        for (uint32_t t = n_dot_valid; t < kp->n_dot_tiles; t++) {
            memset(base + (size_t) t * 1024, 0, 1024 * sizeof(__fp16));
        }
        cv_pack_act_tiles(base, j->c,
                          (HVX_Vector *) (j->scratch + ith * j->scratch_stride),
                          row0, MIN(32, kp->total_rows - row0));
    }
}

struct cv_hmx_job {
    __fp16 * act;        // [n_row_tiles][n_dot][1024] patch-row tiles
    __fp16 * wtile;      // [n_dot][1024] one output-channel tile
    __fp16 * otile;      // [n_row_tiles][1024] results, same tile layout as act
    void *   scales;     // HMX column scales (identity), 256 B aligned
    uint32_t n_dot;
    uint32_t n_row_tiles;
};

static void conv2d_hmx_dot_worker(void * data) {
    struct cv_hmx_job * j = (struct cv_hmx_job *) data;
    // HMX needs an explicit output-scale tile; without HMX_SET_BIAS the bias
    // register holds garbage and the result is all-zero / NaN. FA does the same
    // with an identity tile (flash-attn-ops.c:1612).
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)(uintptr_t) j->scales));

    // `range` is a 16-bit field (max 65535 = 32 dot tiles), so long K has to be
    // walked in chunks of 32 dot tiles -- exactly what
    // core_mma_chunk_fp16() does for n_dot_tiles > 32.
    // Every row tile this worker owns is accumulated in one call, mirroring
    // core_dot_chunk_fp16(): the row loop wraps the whole K sweep and the tail
    // store. Pushing once per row tile instead costs one queue round trip per
    // tile, which for the VAE shapes is ~140k serialised round trips.
    const uint32_t n_loops = j->n_dot / 32;
    const uint32_t rem     = j->n_dot % 32;
    const size_t   dot_stride = (size_t) j->n_dot * CV_TILE_ELMS;

    for (uint32_t r = 0; r < j->n_row_tiles; r++) {
        const __fp16 * row_tiles = j->act + r * dot_stride;
        const __fp16 * col_tiles = j->wtile;

        asm volatile(HMX_CLRACC_F16());
        for (uint32_t l = 0; l < n_loops; l++) {
            asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0")
                         :
                         : "r"(65535), "r"(row_tiles), "r"(col_tiles));
            row_tiles += 32 * CV_TILE_ELMS;
            col_tiles += 32 * CV_TILE_ELMS;
        }
        if (rem > 0) {
            const uint32_t range = 2048u * rem - 1u;
            asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0")
                         :
                         : "r"(range), "r"(row_tiles), "r"(col_tiles));
        }
        asm volatile(HMX_STORE_AFTER_F16("%0", "%1")
                     :
                     : "r"((unsigned int)(uintptr_t)(j->otile + r * CV_TILE_ELMS)), "r"(0)
                     : "memory");
    }
}

static void conv2d_hmx_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_conv2dhmx_ctx * c = (struct htp_conv2dhmx_ctx *) data;
    const struct htp_conv2dhmx_kernel_params * kp = c->kp;

    const uint32_t tile_begin = ith * c->tiles_per_thread;
    const uint32_t tile_end   = MIN(tile_begin + c->tiles_per_thread, c->total_tiles);
    if (tile_begin >= tile_end) {
        return;
    }

    // The shared weight tiles sit at the base of VTCM; each worker's slab follows.
    __fp16 * wtile_all = (__fp16 *) c->vtcm_act;
    uint8_t * slab = c->vtcm_act + c->vtcm_wall_bytes + (size_t) ith * c->stride_per_thread;

    const uint32_t n_dot  = kp->n_dot_tiles;
    const uint32_t n_col  = kp->n_col_tiles;
    const uint32_t OC = kp->OC, OH = kp->OH, OW = kp->OW;
    const uint32_t max_rt = c->max_row_tiles;

    // Layout inside a worker slab: [max_rt][n_dot][1024] act, then
    // [max_rt][1024] results, then the 256 B scale tile, then one staging buffer.
    __fp16 * act    = (__fp16 *) slab;
    __fp16 * otile  = (__fp16 *) (slab + (size_t) max_rt * n_dot * CV_TILE_ELMS * sizeof(__fp16));
    void *   scales = (void *) ((uint8_t *) otile + (size_t) max_rt * CV_TILE_ELMS * sizeof(__fp16));
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
    __fp16 * stage_act = (__fp16 *) ((uint8_t *) scales + 256);

    for (uint32_t batch = tile_begin; batch < tile_end; batch += max_rt) {
        const uint32_t rt_end = MIN(batch + max_rt, tile_end);
        const uint32_t rt_n   = rt_end - batch;

        if (!c->dump_tiles) {
            struct cv_pack_job pack = { c, act, stage_act, batch, rt_n, 2048 };
            ((volatile float *) c->dst)[0] = 21.0f;   /* before activation pack */
            if (!work_queue_run(c->ctx->work_queue, cv_pack_worker, &pack, MIN(c->n_pack_threads, rt_n))) {
                c->failed = true;
                return;
            }
        }
        for (uint32_t i = 0; c->dump_tiles && i < rt_n; i++) {
            const uint32_t row0 = (batch + i) * CV_TILE_ROWS;
            const uint32_t rows = MIN(CV_TILE_ROWS, kp->total_rows - row0);
            cv_build_act_tiles(stage_act, c, kp, row0, MIN(rows, CV_TILE_ROWS));
            if (c->dump_tiles == 7 && batch == tile_begin && i == 0) {
                for (uint32_t r = 0; r < 32; r++) {
                    for (uint32_t k = 0; k < 32; k++) {
                        c->dst[r * 32 + k] = (float) stage_act[(size_t) r * kp->patch_stride + k];
                    }
                }
            }
            hmx_interleave_cols_to_tiles(act + (size_t) i * n_dot * CV_TILE_ELMS,
                                         stage_act, CV_TILE_ROWS, kp->patch_stride,
                                         (size_t) kp->patch_stride, 1,
                                         0, CV_TILE_ROWS);
        }

        for (uint32_t ct = 0; ct < n_col; ct++) {
            const uint32_t oc0 = ct * CV_TILE_COLS;
            __fp16 * wtile = wtile_all + (size_t) ct * n_dot * CV_TILE_ELMS;

            if (c->dump_tiles == 1) {
                // Diagnostic: dump the interleaved weight tile of this col tile
                // straight into dst[0..1023] so element i of the dump is fp16
                // element i of the tile. Only worker 0 and the first col tile may
                // write, which the single-producer restriction in
                // op_conv_2d_hmx() guarantees.
                float * dd = c->dst;
                for (uint32_t i = 0; i < CV_TILE_ELMS; i++) dd[i] = (float) wtile[i];
                return;
            }

            struct cv_hmx_job job = { act, wtile, otile, scales, n_dot, rt_n };
            // hmx_queue_push() returns false when the ring is full and writes
            // nothing. Ignoring that would leave otile holding the previous
            // tile's results while the write-back below happily consumes them,
            // so a full queue must fail the op instead of producing garbage.
            ((volatile float *) c->dst)[0] = 31.0f;   /* before hmx push */
            if (!hmx_queue_push(c->hmx_q, hmx_queue_make_desc(conv2d_hmx_dot_worker, &job))) {
                FARF(ERROR, "conv2d-hmx: hmx queue full, aborting op\n");
                c->failed = true;
                return;
            }
            ((volatile float *) c->dst)[0] = 32.0f;   /* pushed, waiting for HMX */
            hmx_queue_pop(c->hmx_q);
            ((volatile float *) c->dst)[0] = 33.0f;   /* HMX job completed */

            if (c->dump_tiles == 7) {
                for (uint32_t i = 0; i < CV_TILE_ELMS; i++) {
                    c->dst[1024 + i] = (float) act[i];
                    c->dst[2048 + i] = (float) wtile[i];
                    c->dst[3072 + i] = (float) otile[i];
                }
                // [4096] = n_col_tiles, [4097] = weight wall KiB.
                c->dst[4096] = (float) kp->n_col_tiles;
                c->dst[4097] = (float) (c->vtcm_wall_bytes / 1024);
                return;
            }
            if (c->dump_tiles == 4) {
                // Layout self-test: after the HMX result for this col tile is
                // available, recompute one 32x32 result block with plain scalar
                // math in the SAME invocation and report where the two disagree.
                // Filling dst with the scalar result makes the diff visible from
                // the host without a debugger, and it uses the already verified
                // float path so any mismatch is a tile-layout fact, not an
                // arithmetic one.
                float * dd = c->dst;
                // The col-tile loop runs after the row tiles were assembled; the
                // scalar block below is for the first row tile of this batch.
                // Cover the whole output so dst holds a complete reference.
                // Filling only this batch's row tile left the rest of the output
                // at whatever it held before, which made the comparison useless.
                const uint32_t m0 = 0;
                const uint32_t m1 = kp->total_rows;
                for (uint32_t rr = m0; rr < m1; rr++) {
                    for (uint32_t cc = 0; cc < OC; cc++) {
                        // scalar reference for (patch row m0+rr, oc oc0+cc)
                        double acc = 0.0;
                        const uint32_t rr_abs = rr;
                        const uint32_t ow_s = rr_abs % OW;
                        const uint32_t oh_s = (rr_abs / OW) % OH;
                        const uint32_t nn_s = rr_abs / (OW * OH);
                        const uint32_t occ  = cc;
                        if (occ < OC && rr_abs < kp->total_rows) {
                            for (uint32_t ic = 0; ic < kp->IC; ic++) {
                                for (uint32_t kh = 0; kh < kp->KH; kh++) {
                                    const int32_t ih = (int32_t) oh_s * (int32_t) kp->stride + (int32_t) kh - (int32_t) kp->pad;
                                    if (ih < 0 || ih >= (int32_t) kp->IH) continue;
                                    for (uint32_t kw = 0; kw < kp->KW; kw++) {
                                        const int32_t iw = (int32_t) ow_s * (int32_t) kp->stride + (int32_t) kw - (int32_t) kp->pad;
                                        if (iw < 0 || iw >= (int32_t) kp->IW) continue;
                                        const float xv = c->src[(((size_t) nn_s * kp->IC + ic) * kp->IH + ih) * kp->IW + iw];
                                        const float wv = cv_weight_at(c, kp, kw, kh, ic, occ);
                                        acc += (double) xv * (double) wv;
                                    }
                                }
                            }
                        }
                        // Write with the tensor's real addressing so this block is
                        // directly comparable to a correct result: dst is
                        // [OW, OH, OC, N] with OW contiguous, so the flat index is
                        // ((n*OC + oc)*OH + oh)*OW + ow.
                        if (occ < OC && rr_abs < kp->total_rows) {
                            dd[((size_t) nn_s * OC + occ) * OH * OW
                               + (size_t) oh_s * OW + ow_s] = (float) acc;
                        }
                    }
                }
                return;
            }
            if (c->dump_tiles == 3) {
                // Dump the raw 1024-element accumulator tile for the first col
                // tile, before any index translation. With a ramp input and unit
                // weights every element of the result is a distinct value, so the
                // tile layout can be read off directly instead of guessed.
                float * dd = c->dst;
                const __fp16 * ot = otile;
                for (uint32_t i = 0; i < CV_TILE_ELMS; i++) dd[i] = (float) ot[i];
                return;
            }
            if (c->dump_tiles == 5) {
                // Ground-truth correspondence: next to each raw accumulator
                // element, store the scalar result it should equal for the same
                // (patch row, output channel) slot, computed with the reference
                // path that is already verified against the host oracle. Reading
                // both out of one dump gives the tile layout directly, with no
                // guesswork and no dependence on how the probe generated data.
                const __fp16 * ot = otile;
                float * dd = c->dst;
                const uint32_t row_base = (batch + 0) * CV_TILE_ROWS;
                for (uint32_t rr = 0; rr < CV_TILE_ROWS; rr++) {
                    for (uint32_t cc = 0; cc < CV_TILE_COLS; cc++) {
                        const uint32_t rr_abs = row_base + rr;
                        double acc = 0.0;
                        if (rr_abs < kp->total_rows && (oc0 + cc) < OC) {
                            const uint32_t ow_s = rr_abs % OW;
                            const uint32_t oh_s = (rr_abs / OW) % OH;
                            const uint32_t nn_s = rr_abs / (OW * OH);
                            const uint32_t occ  = oc0 + cc;
                            for (uint32_t ic = 0; ic < kp->IC; ic++) {
                                for (uint32_t kh = 0; kh < kp->KH; kh++) {
                                    const int32_t ih = (int32_t) oh_s * (int32_t) kp->stride + (int32_t) kh - (int32_t) kp->pad;
                                    if (ih < 0 || ih >= (int32_t) kp->IH) continue;
                                    for (uint32_t kw = 0; kw < kp->KW; kw++) {
                                        const int32_t iw = (int32_t) ow_s * (int32_t) kp->stride + (int32_t) kw - (int32_t) kp->pad;
                                        if (iw < 0 || iw >= (int32_t) kp->IW) continue;
                                        const float xv = c->src[(((size_t) nn_s * kp->IC + ic) * kp->IH + ih) * kp->IW + iw];
                                        acc += (double) xv * (double) cv_weight_at(c, kp, kw, kh, ic, occ);
                                    }
                                }
                            }
                        }
                        // interleave raw and expected so the host can pair them
                        dd[(size_t)(rr * CV_TILE_COLS + cc) * 2 + 0] = (float) ot[rr * CV_TILE_COLS + cc];
                        dd[(size_t)(rr * CV_TILE_COLS + cc) * 2 + 1] = (float) acc;
                    }
                }
                return;
            }
            if (c->dump_tiles == 6) {
                // Report the readback geometry as the writer actually sees it:
                // for the first eight output channels of the first 32 rows, store
                // (computed tile index, value at that index). This removes every
                // assumption about which tile `ot` points at.
                float * dd = c->dst;
                const __fp16 * ot = otile;
                // First 64 elements: the raw tile, exactly as dump 3 reports it.
                // Next 64: the (index, value) pairs the writer would read.
                for (uint32_t i = 0; i < 64; i++) dd[i] = (float) ot[i];
                uint32_t w = 64;
                for (uint32_t rr = 0; rr < 32; rr++) {
                    // Report only the VALUE each row would read for oc = 0, with
                    // the index implied by position, so a mismatch between the
                    // linear dump above and this list is unambiguous.
                    const size_t oidx = (size_t) ((rr >> 1) * 64 + (rr & 1));
                    dd[w++] = (float) ot[oidx];
                }
                return;
            }
            if (c->dump_tiles == 2) {
                // Dump the assembled activation tile of this worker's first row
                // tile, so the act and weight layouts can be compared element by
                // element instead of by inference.
                float * dd = c->dst;
                for (uint32_t i = 0; i < CV_TILE_ELMS; i++) dd[i] = (float) act[i];
                return;
            }

            for (uint32_t i = 0; i < rt_n; i++) {
                const uint32_t tile = batch + i;
                const uint32_t row0 = tile * CV_TILE_ROWS;
                const uint32_t rows = MIN(CV_TILE_ROWS, kp->total_rows - row0);
                const __fp16 * ot = otile + (size_t) i * CV_TILE_ELMS;

                if (row0 / (OH * OW) == (row0 + rows - 1) / (OH * OW)) {
                    HVX_Vector matrix[32] __attribute__((aligned(128)));
                    HVX_Vector scratch[32] __attribute__((aligned(128)));
                    for (uint32_t r = 0; r < 32; r += 2) {
                        const HVX_VectorPair pair = hvx_vec_f16_to_f32_shuff(hvx_vmem(ot + r * 32));
                        matrix[r] = Q6_V_lo_W(pair);
                        matrix[r + 1] = Q6_V_hi_W(pair);
                    }
                    hvx_transpose_32x32_w_shuff_inplace(matrix, scratch);
                    const uint32_t n = row0 / (OH * OW);
                    const uint32_t pos = row0 % (OH * OW);
                    for (uint32_t l = 0; l < MIN(32, OC - oc0); l++) {
                        float * dst = c->dst + ((size_t) n * OC + oc0 + l) * OH * OW + pos;
                        hvx_vec_store_u(dst, rows * sizeof(float), matrix[l]);
                    }
                    continue;
                }
                for (uint32_t lane = 0; lane < rows; lane++) {
                    const uint32_t r = row0 + lane;
                    const uint32_t ow_i = r % OW;
                    const uint32_t oh_i = (r / OW) % OH;
                    const uint32_t n_i  = r / (OW * OH);
                    float * dpx = c->dst + ((size_t) n_i * OC * OH * OW)
                                        + (size_t) oh_i * OW + ow_i;
                    for (uint32_t l = 0; l < CV_TILE_COLS; l++) {
                        const uint32_t oc = oc0 + l;
                        if (oc >= OC) break;
                        // HMX interleaves the two rows of each output vector at
                        // halfword granularity, as in the existing matmul path.
                        const size_t oidx = (size_t) (lane >> 1) * 64 + l * 2 + (lane & 1);
                        dpx[(size_t) oc * OH * OW] = (float) ot[oidx];
                    }
                }
            }
        }
    }
}

int op_conv_2d_hmx(struct htp_ops_context * octx) {
    const struct htp_conv2dhmx_kernel_params * kp =
        (const struct htp_conv2dhmx_kernel_params *) octx->kernel_params;
    if (octx->dst->type != HTP_TYPE_F32 || octx->src[0] == NULL || octx->src[1] == NULL) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!octx->ctx->hmx_enabled) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!kp->IC || !kp->IH || !kp->IW || !kp->OC || !kp->OH || !kp->OW ||
        !kp->KH || !kp->KW || !kp->stride || !kp->total_rows ||
        !kp->patch_stride || kp->patch_stride % 64 || kp->dump_tiles > 7 ||
        !kp->k_valid || kp->k_valid > kp->patch_stride ||
        kp->patch_stride - kp->k_valid >= 64 ||
        (uint64_t) kp->IC * kp->KH * kp->KW != kp->k_valid ||
        kp->n_dot_tiles != kp->patch_stride / 32 ||
        kp->n_col_tiles != ((uint64_t) kp->OC + 31) / 32 ||
        octx->src[1]->type != HTP_TYPE_F32 ||
        (kp->kernel_type != HTP_TYPE_F16 && kp->kernel_type != HTP_TYPE_F32) ||
        octx->src[0]->type != kp->kernel_type) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->src[1]->ne[0] != kp->IW || octx->src[1]->ne[1] != kp->IH ||
        octx->src[1]->ne[2] != kp->IC || octx->src[0]->ne[0] != kp->KW ||
        octx->src[0]->ne[1] != kp->KH || octx->src[0]->ne[2] != kp->IC ||
        octx->src[0]->ne[3] != kp->OC || octx->dst->ne[0] != kp->OW ||
        octx->dst->ne[1] != kp->OH || octx->dst->ne[2] != kp->OC ||
        (uint64_t) kp->OW * kp->OH * octx->dst->ne[3] != kp->total_rows ||
        octx->src[1]->ne[3] != octx->dst->ne[3]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const size_t output_elements = (size_t) kp->total_rows * kp->OC;
    const size_t dump_elements = kp->dump_tiles == 7 ? 4100 :
                                 kp->dump_tiles == 5 ? 2048 :
                                 kp->dump_tiles == 6 ? 96 : 1024;
    if (kp->dump_tiles && kp->dump_tiles != 4 && output_elements < dump_elements) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t n_threads = kp->n_threads ? kp->n_threads : octx->ctx->n_threads;
    if (n_threads > octx->ctx->n_threads) {
        n_threads = octx->ctx->n_threads;
    }
    // The HMX queue requires a single producer and completion owner.
    n_threads = 1;
    if (!htp_ops_context_set_n_threads(octx, n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    // One row tile's worth of activation: n_dot tiles x 1024 elms already spans
    // all 32 rows x K columns, because each tile carries 32 rows by 32 k-cols.
    const size_t act_bytes = (size_t) kp->n_dot_tiles * CV_TILE_ELMS * sizeof(__fp16);
    const size_t w_bytes   = (size_t) kp->n_dot_tiles * CV_TILE_ELMS * sizeof(__fp16);
    const size_t out_bytes = (size_t) CV_TILE_ELMS * sizeof(__fp16);
    // 2 row-major staging buffers (32 rows x K fp16), 128 B aligned.
    /* phase timing captured during weight pack */
    const size_t stage_bytes = hex_round_up((uint32_t)((size_t) kp->patch_stride * 32 * sizeof(__fp16)), 128);
    const size_t scales_bytes = 256;

    // Weights are read-only and HMX is already serialised by hmx_queue, so one
    // shared copy is enough. Replicating them per worker is what pushed the
    // 3x3/512-channel VAE convs 2-5x over the 8 MB VTCM budget.
    const size_t wall_bytes = (size_t) kp->n_col_tiles * w_bytes;

    // Reserve one staging area, private pack scratch and an end guard.
    // HMX consumes a batch of row tiles per output-channel tile.
    const uint32_t n_pack_threads = kp->dump_tiles ? 1 : MIN(MAX(1, kp->n_threads), octx->ctx->n_threads);
    const size_t fixed = 256 + stage_bytes + (size_t) (n_pack_threads - 1) * 4096;
    const size_t avail = octx->ctx->vtcm_size > wall_bytes + 128
                       ? octx->ctx->vtcm_size - wall_bytes - 128 : 0;
    const size_t per_rt = act_bytes + out_bytes;

    const uint32_t n_tiles_all = kp->total_rows / CV_TILE_ROWS + (kp->total_rows % CV_TILE_ROWS != 0);
    const uint32_t max_rt = avail > fixed ? MIN((avail - fixed) / per_rt, n_tiles_all) : 0;
    if (max_rt == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const size_t per_thread = per_rt * max_rt + fixed;
    if (!htp_ops_context_set_n_threads(octx, n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_conv2dhmx_ctx c = { 0 };
    c.kp     = kp;
    c.ctx    = octx->ctx;
    c.n_pack_threads = n_pack_threads;
    c.src    = (const float *) octx->src[1]->data;
    c.wf32   = kp->kernel_type == HTP_TYPE_F16 ? NULL : (const float *) octx->src[0]->data;
    c.w16    = kp->kernel_type == HTP_TYPE_F16 ? (const __fp16 *) octx->src[0]->data : NULL;
    c.dst    = (float *) octx->dst->data;
    // The DSP's getenv() does not observe the host environment, so a dump
    // request has to arrive through kernel_params like every other value.
    c.dump_tiles = kp->dump_tiles;
    c.hmx_q  = octx->ctx->hmx_queue;
    if (c.hmx_q == NULL) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t pth = MIN(n_threads, n_tiles_all);
    const uint32_t per = (n_tiles_all + pth - 1) / pth;

    c.vtcm_act          = (uint8_t *) octx->ctx->vtcm_base;
    c.total_tiles       = n_tiles_all;
    c.tiles_per_thread  = per;
    c.stride_per_thread = (uint32_t) per_thread;
    c.vtcm_wall_bytes   = (uint32_t) wall_bytes;
    c.max_row_tiles     = max_rt;

    // Build the shared weight tiles once, before any worker runs. They sit at the
    // base of VTCM so every worker sees the same copy.
    {
        __fp16 * wtile_all = (__fp16 *) c.vtcm_act;
        // Built before work_queue_run(), so worker 0's slab is still free to use
        // as the staging buffer.
        __fp16 * stage_w   = (__fp16 *) ((uint8_t *) c.vtcm_act + wall_bytes);
        dma_queue * wdma = octx->ctx->dma[0];
        for (uint32_t ct = 0; ct < kp->n_col_tiles; ct++) {
            if (c.w16) {
                // DMA path: cv_copy_weight_rows_dma pops the descriptor before
                // returning, so the staging rows are complete when it is done.
                cv_copy_weight_rows_dma(wdma, stage_w, &c, kp, ct * CV_TILE_COLS);
            } else {
                cv_build_weight_tiles(stage_w, &c, kp, ct * CV_TILE_COLS);
            }
            hmx_interleave_rows_to_tiles(wtile_all + (size_t) ct * kp->n_dot_tiles * CV_TILE_ELMS,
                                         stage_w, CV_TILE_COLS, kp->patch_stride,
                                         (size_t) kp->patch_stride,
                                         0, CV_TILE_COLS);
        }
    }

    volatile uint32_t * guard = (volatile uint32_t *) ((uint8_t *) c.vtcm_act + wall_bytes + per_thread);
    for (uint32_t i = 0; i < 32; i++) {
        guard[i] = 0xc02d5afeu;
    }
    conv2d_hmx_thread(1, 0, &c);
    for (uint32_t i = 0; i < 32; i++) {
        if (guard[i] != 0xc02d5afeu) {
            return HTP_STATUS_INTERNAL_ERR;
        }
    }
    if (c.failed) {
        return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}
