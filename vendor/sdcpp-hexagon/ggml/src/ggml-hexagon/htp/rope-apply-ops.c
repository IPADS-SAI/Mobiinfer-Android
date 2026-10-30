#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

// ROPE_APPLY: dst[row, i] rotates src0 row-wise by a per-token table of 2x2
// matrices (see ggml_rope_apply()). ggml-common.h, htp-* and hex-* headers must
// be included by exactly one TU on the DSP side, so they are pulled in here
// rather than in the header.

#include <HAP_farf.h>

#include <string.h>

#include "dma-queue.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "rope-apply-ops.h"
#include "work-queue.h"

struct htp_rope_apply_context {
    struct htp_ops_context * octx;
    const struct htp_rope_apply_kernel_params * kparams;

    dma_addr_t data_src0;
    dma_addr_t data_pe;
    dma_addr_t data_dst;

    uint32_t tok_start;
    uint32_t n_tok;
    uint32_t n_tok_per_thread;

    uint8_t * vtcm_src;
    uint8_t * vtcm_dst;
    size_t    vtcm_half;

};

// One row: half_dim independent 2x2 rotations. Both the interleaved and the
// split layout read the same (cos, sin) pair from pe[token, j].
// One row: half_dim independent 2x2 rotations.
//   matrix = pe_row + j*pe_mat_stride
//   c  = matrix + 0*nb1   (cos)      s  = matrix + 1*nb1 (sin)
//   c1 = matrix + 1*nb1 + 1*nb0      neg_s = matrix + 1*nb0   (interleaved only)
// pe is a [2, 2, half_dim, ne2] view, so nb1 is NOT contiguous with nb0.
static void rope_apply_row_f32(const float * restrict src,
                               const uint8_t * restrict pe_row,
                               float * restrict dst,
                               uint32_t half_dim,
                               uint32_t pe_mat_stride,
                               uint32_t pe_nb0,
                               uint32_t pe_nb1,
                               uint32_t interleaved) {
    if (interleaved) {
        for (uint32_t j = 0; j < half_dim; j++) {
            const uint8_t * m = pe_row + j * pe_mat_stride;
            const float c     = *(const float *) (m);
            const float neg_s = *(const float *) (m + pe_nb0);
            const float s     = *(const float *) (m + pe_nb1);
            const float c1    = *(const float *) (m + pe_nb1 + pe_nb0);
            const float a     = src[2 * j + 0];
            const float b     = src[2 * j + 1];
            dst[2 * j + 0]    = a * c + b * neg_s;
            dst[2 * j + 1]    = a * s + b * c1;
        }
    } else {
        for (uint32_t j = 0; j < half_dim; j++) {
            const uint8_t * m = pe_row + j * pe_mat_stride;
            const float c     = *(const float *) (m);
            const float s     = *(const float *) (m + pe_nb1);
            const float a     = src[j];
            const float b     = src[j + half_dim];
            dst[j]            = a * c - b * s;
            dst[j + half_dim] = a * s + b * c;
        }
    }
}


// Vectorised non-interleaved path. src/dst halves are contiguous, so the only
// strided access is the pe table: cos = pe_row[j*pe_mat_stride], sin = pe_row[j*pe_mat_stride+pe_nb1].
// Pack the whole (cos,sin) table for this token once into scratch, then run the
// rotation 128 floats at a time.
static void rope_pack_cos_sin(const uint8_t * restrict pe_row, uint32_t half_dim,
                             uint32_t pe_mat_stride, uint32_t pe_nb1,
                             float * restrict cosv, float * restrict sinv) {
    for (uint32_t j = 0; j < half_dim; j++) {
        const uint8_t * m = pe_row + j * pe_mat_stride;
        cosv[j] = *(const float *) (m);
        sinv[j] = *(const float *) (m + pe_nb1);
    }
}

static void rope_apply_row_f32_hvx(const float * restrict src,
                                   const uint8_t * restrict pe_row,
                                   float * restrict dst,
                                   uint32_t half_dim,
                                   uint32_t pe_mat_stride,
                                   uint32_t pe_nb1,
                                   float * restrict cosv,
                                   float * restrict sinv) {
    (void) pe_row; (void) pe_mat_stride; (void) pe_nb1;

    const uint32_t epv = 128 / sizeof(float);   // 32 floats per HVX vector
    const float * restrict a = src;
    const float * restrict b = src + half_dim;
    float * restrict oa = dst;
    float * restrict ob = dst + half_dim;

    uint32_t j = 0;
    for (; j + epv <= half_dim; j += epv) {
        const HVX_Vector va = hvx_vmemu(a + j);
        const HVX_Vector vb = hvx_vmemu(b + j);
        const HVX_Vector vc = hvx_vmemu(cosv + j);
        const HVX_Vector vs = hvx_vmemu(sinv + j);

        const HVX_Vector o0 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(va, vc),
                                                  hvx_vec_mul_f32_f32(vb, vs));
        const HVX_Vector o1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(va, vs),
                                                  hvx_vec_mul_f32_f32(vb, vc));
        hvx_vmemu(oa + j) = o0;
        hvx_vmemu(ob + j) = o1;
    }
    for (; j < half_dim; j++) {
        const float c = cosv[j];
        const float s = sinv[j];
        const float x = a[j];
        const float y = b[j];
        oa[j] = x * c - y * s;
        ob[j] = x * s + y * c;
    }
}

static void rope_apply_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_rope_apply_context * rctx = (const struct htp_rope_apply_context *) data;
    struct htp_ops_context * octx = rctx->octx;
    const struct htp_rope_apply_kernel_params * kp = rctx->kparams;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t tok_begin = rctx->tok_start + rctx->n_tok_per_thread * ith;
    const uint32_t tok_end   = MIN(tok_begin + rctx->n_tok_per_thread, rctx->tok_start + rctx->n_tok);
    if (tok_begin >= tok_end) {
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * s_vtcm = rctx->vtcm_src + ith * rctx->vtcm_half;
    uint8_t * d_vtcm = rctx->vtcm_dst + ith * rctx->vtcm_half;

    const uint32_t ne1       = kp->ne1;
    const uint32_t ne2       = kp->ne2;
    const uint32_t row_bytes = kp->ne0 * sizeof(float);
    const uint32_t spad_row  = kp->spad_row_bytes;
    const uint32_t tok_bytes = ne1 * row_bytes;

    // dst rows are ordered (batch, head, token) with token fastest, so a token's
    // ne1 rows are ne2 rows apart in dst and cannot be moved as one run. src0
    // orders heads contiguously (src0_row_stride == row_bytes), so src0 does move
    // as one run per token; dst is written back row by row.
    for (uint32_t tok = tok_begin; tok < tok_end; tok++) {
        const uint32_t batch = tok / ne2;
        const uint32_t t     = tok % ne2;

        const size_t s0 = (size_t) batch * kp->src0_bat_stride + (size_t) t * kp->src0_tok_stride;

        dma_queue_push(dma_q, dma_make_data(s_vtcm, rctx->data_src0 + s0),
                       spad_row, row_bytes, row_bytes, ne1);
        dma_queue_pop(dma_q);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (tok & 0x7fff));

        const uint8_t * pe_r = (const uint8_t *) (uintptr_t) rctx->data_pe +
                               (size_t) t * kp->pe_row_stride;
        if (kp->interleaved) {
            for (uint32_t h = 0; h < ne1; h++) {
                rope_apply_row_f32((const float *) (s_vtcm + h * spad_row),
                                   pe_r,
                                   (float *) (d_vtcm + h * spad_row),
                                   kp->half_dim, kp->pe_mat_stride, kp->pe_nb0, kp->pe_nb1, 1);
            }
        } else {
            // pack cos/sin once per token, then vectorise every head row
            float * cosv = (float *) (s_vtcm + ne1 * spad_row);
            float * sinv = cosv + kp->half_dim;
            rope_pack_cos_sin(pe_r, kp->half_dim, kp->pe_mat_stride, kp->pe_nb1, cosv, sinv);
            for (uint32_t h = 0; h < ne1; h++) {
                rope_apply_row_f32_hvx((const float *) (s_vtcm + h * spad_row),
                                       (const uint8_t *) 0,
                                       (float *) (d_vtcm + h * spad_row),
                                       kp->half_dim, 0, 0, cosv, sinv);
            }
        }

        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (tok & 0x7fff));

        // dst: head h of this token lives at row (h*ne2 + t)
        for (uint32_t h = 0; h < ne1; h++) {
            dma_queue_push(dma_q, dma_make_data(rctx->data_dst +
                                                (size_t) (batch * ne1 * ne2 + h * ne2 + t) * row_bytes,
                                                (uintptr_t) (d_vtcm + h * spad_row)),
                           row_bytes, row_bytes, row_bytes, 1);
        }
        for (uint32_t h = 0; h < ne1; h++) {
            dma_queue_pop(dma_q);
        }
    }
}

int op_rope_apply(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * pe   = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32 || pe->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_rope_apply_kernel_params * kparams =
        (const struct htp_rope_apply_kernel_params *) octx->kernel_params;

    if (kparams->n_rows == 0 || kparams->half_dim == 0 || kparams->ne1 == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (octx->ctx->vtcm_size < (size_t) kparams->vtcm_bytes) {
        FARF(ERROR, "rope-apply : VTCM %zu < needed %u\n", octx->ctx->vtcm_size, kparams->vtcm_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    // Must match the host's half_bytes: the row block plus the packed (cos,sin)
    // scratch that the vectorised path appends to it.
    const size_t rope_scratch = (size_t) kparams->half_dim * 2 * sizeof(float);
    const size_t half = ((size_t) kparams->spad_row_bytes * kparams->ne1 + rope_scratch + 127) & ~(size_t) 127;

    struct htp_rope_apply_context rctx = {
        .octx              = octx,
        .kparams           = kparams,
        .data_src0         = src0->data,
        .data_pe           = pe->data,
        .data_dst          = dst->data,
        .tok_start         = 0,
        .n_tok             = kparams->n_rows / kparams->ne1,
        .n_tok_per_thread  = kparams->n_rows_per_thread / kparams->ne1,
        .vtcm_src          = base,
        .vtcm_dst          = base + half * octx->n_threads,
        .vtcm_half         = half,
    };

    work_queue_run(octx->ctx->work_queue, rope_apply_thread, &rctx, octx->n_threads);

    octx->status = HTP_STATUS_OK;
    return HTP_STATUS_OK;
}
