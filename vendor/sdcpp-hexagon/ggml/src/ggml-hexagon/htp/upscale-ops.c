#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <string.h>

#include "hex-common.h"
#include "hex-profile.h"

#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

// GGML_OP_UPSCALE with GGML_SCALE_MODE_NEAREST.
//
// ggml.h is not on the DSP include path, so mirror the two encodings we read
// from op_params. They must stay in sync with ggml.h.
#define UPSCALE_MODE_NEAREST          0
#define UPSCALE_FLAG_ALIGN_CORNERS    (1 << 8)
#define UPSCALE_FLAG_ANTIALIAS        (1 << 9)
//
// ggml's reference maps output index i to source index i / sf, so for an
// integer scale factor K every source element feeds exactly K adjacent
// outputs. The layout ggml hands us is the plain row-major NCHW tensor, so a
// pixel here is ne0 contiguous elements; we expand whole rows and then
// duplicate the row band vertically.
//
// The scale factors are per-axis and are not required to be equal, which is
// why the horizontal step is driven by a counter rather than an assumed ratio.

#define upscale_preamble                            \
    const struct htp_tensor * dst = octx->dst;      \
    const struct htp_tensor * src = octx->src[0];   \
                                                    \
    const uint32_t ne0 = src->ne[0];                \
    const uint32_t ne1 = src->ne[1];                \
    const uint32_t ne2 = src->ne[2];                \
    const uint32_t ne3 = src->ne[3];                \
                                                    \
    const uint32_t ow = dst->ne[0];                 \
    const uint32_t oh = dst->ne[1];                 \
                                                    \
    const uint32_t nb1 = src->nb[1];                \
    const uint32_t nb2 = src->nb[2];                \
    const uint32_t nb3 = src->nb[3];                \
                                                    \
    const uint32_t onb1 = dst->nb[1];               \
    const uint32_t onb2 = dst->nb[2];               \
    const uint32_t onb3 = dst->nb[3];

struct htp_upscale_context {
    struct htp_ops_context * octx;
    uint32_t scale_x;
    uint32_t scale_y;
    uint32_t rows_per_thread;
    uint32_t row_start;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    size_t   elem_size;
};

// One source row is expanded horizontally into the first output row of its
// band; the remaining rows of the band are byte copies of that row.
static void upscale_expand_row(const uint8_t * src_row,
                               uint8_t * dst_row,
                               uint32_t  ne0,
                               uint32_t  ow,
                               uint32_t  scale_x,
                               size_t    elem_size) {
    if (elem_size == 4 && scale_x == 2) {
        // f32 2x: duplicate each source element.
        //
        // This used to build the vector with Q6_W_vcombine_VV(v, v), which
        // concatenates the two operands rather than interleaving their lanes --
        // it emits [v0,v1,...,v15,v0,v1,...], not [v0,v0,v1,v1,...], so every
        // element after the first source vector landed in the wrong pixel.
        // Reproduced under hexagon-sim (maxerr 0.99); Q6_W_vshuff_VVR and
        // vror+vshuff variants were also measured and are not a lane-wise
        // duplicate either. Keep the plain loop until a correct interleave is
        // found and measured -- this op is not on the critical path.
        float *       d = (float *) dst_row;
        const float * s = (const float *) src_row;
        for (uint32_t x = 0; x < ne0; ++x) {
            d[2 * x]     = s[x];
            d[2 * x + 1] = s[x];
        }
    } else if (scale_x == 1) {
        memcpy(dst_row, src_row, (size_t) ne0 * elem_size);
    } else {
        // Incremental source index: a divide per output element is measurable
        // at these sizes, a counter is not.
        uint32_t sx  = 0;
        uint32_t cnt = 0;
        for (uint32_t ox = 0; ox < ow; ++ox) {
            memcpy(dst_row + (size_t) ox * elem_size,
                   src_row + (size_t) sx * elem_size, elem_size);
            if (++cnt == scale_x) {
                cnt = 0;
                ++sx;
            }
        }
    }
}

static void upscale_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_upscale_context * uctx = (const struct htp_upscale_context *) data;
    struct htp_ops_context           * octx = uctx->octx;
    upscale_preamble;

    const uint32_t ir0 = uctx->row_start + uctx->rows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + uctx->rows_per_thread, uctx->row_start + uctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const size_t out_row_bytes = (size_t) ow * uctx->elem_size;

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t i1 = ir % ne1;
        const uint32_t i2 = (ir / ne1) % ne2;
        const uint32_t i3 = ir / (ne1 * ne2);

        const uint8_t * src_row = (const uint8_t *) src->data + (size_t) i1 * nb1 + (size_t) i2 * nb2 + (size_t) i3 * nb3;
        // Output row band starts at i1 * scale_y.
        uint8_t * dst_row0 = (uint8_t *) dst->data + (size_t) (i1 * uctx->scale_y) * onb1 + (size_t) i2 * onb2 + (size_t) i3 * onb3;

        upscale_expand_row(src_row, dst_row0, ne0, ow, uctx->scale_x, uctx->elem_size);

        // Remaining rows of the band are identical to the first.
        for (uint32_t ry = 1; ry < uctx->scale_y; ++ry) {
            uint8_t * dst_rep = dst_row0 + (size_t) ry * onb1;
            for (size_t off = 0; off < out_row_bytes; off += HEX_L2_LINE_SIZE) {
                const size_t n = MIN((size_t) HEX_L2_LINE_SIZE, out_row_bytes - off);
                memcpy(dst_rep + off, dst_row0 + off, n);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(HIGH, "upscale %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

int op_upscale(struct htp_ops_context * octx) {
    upscale_preamble;

    if (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src->type != dst->type) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // Only nearest-neighbour with an exact integer ratio is supported; ggml's
    // scale mode is packed into the low byte of op_params[0].
    const int32_t mode_flags = octx->op_params[0];
    const int32_t mode       = mode_flags & 0xFF;
    if (mode != UPSCALE_MODE_NEAREST) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (mode_flags & UPSCALE_FLAG_ALIGN_CORNERS) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((mode_flags & UPSCALE_FLAG_ANTIALIAS) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (ow % ne0 != 0 || oh % ne1 != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t scale_x = ow / ne0;
    const uint32_t scale_y = oh / ne1;
    if (scale_x == 0 || scale_y == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = ne1 * ne2 * ne3;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(nrows, 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_upscale_context uctx = {
        .octx            = octx,
        .scale_x         = scale_x,
        .scale_y         = scale_y,
        .rows_per_thread = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div),
        .row_start       = row_start,
        .total_rows      = nrows,
        .elem_size       = (dst->type == HTP_TYPE_F32) ? sizeof(float) : sizeof(_Float16),
    };

    FARF(HIGH, "upscale: (%ux%ux%ux%u)->(%ux%ux%ux%u) scale=%ux%u type=%u\n",
         ne0, ne1, ne2, ne3, ow, oh, ne2, ne3, scale_x, scale_y, dst->type);

    work_queue_run(octx->ctx->work_queue, upscale_thread, &uctx, octx->n_threads);

    return HTP_STATUS_OK;
}
