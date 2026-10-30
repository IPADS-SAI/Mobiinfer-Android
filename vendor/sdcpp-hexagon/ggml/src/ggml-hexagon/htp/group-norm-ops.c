#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <math.h>
#include <string.h>

#include "hex-common.h"
#include "hex-profile.h"

#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

// GGML_OP_GROUP_NORM.
//
// ggml normalises over [ne0, ne1, channels_per_group] independently for each
// batch element, computing mean and variance in F32 regardless of the tensor
// type, then applies the per-channel affine (weight/bias) which the sd.cpp
// GroupNorm wrapper folds in as separate MUL/ADD nodes -- so this kernel only
// produces the normalised value.
//
// The reduction is the same shape as the layer-norm kernel in the MNN reference
// backend: accumulate vectors, then fold with vror adds. Clamping the vector
// count keeps the tail out of the hot loop.
//
// op_params:
//   [0] n_groups (int32)
//   [1] eps      (float, bit-cast into the int32 slot)

// Horizontal sum of the 32 f32 lanes in one vector.
//
// Use the project's single-vector helper. Note the two traps here:
//  - hvx_vec_reduce_sum_f32x2(a, b) de-interleaves *two* vectors first, so it is
//    not applicable when you only have one.
//  - the ladder must fold all the way down to a single lane (4B); stopping at
//    8B leaves two partial sums.
static inline float group_norm_reduce_sum_f32(HVX_Vector v) {
    union { HVX_Vector v; float f[32]; } u = { .v = hvx_vec_reduce_sum_f32(v) };
    return u.f[0];
}

struct htp_group_norm_context {
    struct htp_ops_context * octx;
    uint32_t n_groups;
    uint32_t cpg;         // channels per group
    uint32_t group_elems; // ne0 * ne1 * cpg
    float    eps;
    uint32_t groups_per_thread;
    uint32_t group_start;
    uint32_t total_groups; // n_groups * ne3
};

static void group_norm_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_group_norm_context * gctx = (const struct htp_group_norm_context *) data;
    struct htp_ops_context             * octx = gctx->octx;

    const struct htp_tensor * dst = octx->dst;
    const struct htp_tensor * src = octx->src[0];

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t nb1 = src->nb[1];
    const uint32_t nb2 = src->nb[2];
    const uint32_t nb3 = src->nb[3];

    const uint32_t ig0 = gctx->group_start + gctx->groups_per_thread * ith;
    const uint32_t ig1 = MIN(ig0 + gctx->groups_per_thread, gctx->group_start + gctx->total_groups);

    if (ig0 >= ig1) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ig0);

    const float inv_n = 1.0f / (float) gctx->group_elems;
    // kernel_params[0] is a debug switch from the host: 1 = scalar only.
    const bool scalar_only = octx->kernel_params[0] != 0;

    for (uint32_t ig = ig0; ig < ig1; ++ig) {
        const uint32_t g = ig % gctx->n_groups;
        const uint32_t i3 = ig / gctx->n_groups;

        const uint32_t c0 = g * gctx->cpg;
        const uint32_t c1 = MIN(c0 + gctx->cpg, ne2);

        // Pass 1: mean.
        float sum = 0.0f;
        for (uint32_t i2 = c0; i2 < c1; ++i2) {
            for (uint32_t i1 = 0; i1 < ne1; ++i1) {
                const float * x = (const float *) ((const uint8_t *) src->data + (size_t) i1 * nb1 + (size_t) i2 * nb2 + (size_t) i3 * nb3);
                if (scalar_only) {
                    for (uint32_t i0 = 0; i0 < ne0; ++i0) {
                        sum += x[i0];
                    }
                    continue;
                }
                HVX_Vector acc = Q6_V_vzero();
                uint32_t i0 = 0;
                for (; i0 + 32 <= ne0; i0 += 32) {
                    acc = Q6_Vsf_vadd_VsfVsf(acc, hvx_vmemu((const HVX_UVector *) (x + i0)));
                }
                float tail = 0.0f;
                for (; i0 < ne0; ++i0) {
                    tail += x[i0];
                }
                sum += group_norm_reduce_sum_f32(acc) + tail;
            }
        }
        const float mean = sum * inv_n;

        // Pass 2: variance (two-pass, matching ggml's reference: it subtracts
        // the mean before squaring, so this is the population variance).
        float sumsq = 0.0f;
        for (uint32_t i2 = c0; i2 < c1; ++i2) {
            for (uint32_t i1 = 0; i1 < ne1; ++i1) {
                const float * x = (const float *) ((const uint8_t *) src->data + (size_t) i1 * nb1 + (size_t) i2 * nb2 + (size_t) i3 * nb3);
                if (scalar_only) {
                    for (uint32_t i0 = 0; i0 < ne0; ++i0) {
                        const float d = x[i0] - mean;
                        sumsq += d * d;
                    }
                    continue;
                }
                // qf32 accumulator, matching hvx_fast_rms_norm_f32: f32
                // accumulation of squares loses precision.
                HVX_Vector acc = Q6_V_vzero();
                uint32_t i0 = 0;
                for (; i0 + 32 <= ne0; i0 += 32) {
                    const HVX_Vector v = hvx_vmemu((const HVX_UVector *) (x + i0));
                    const HVX_Vector d = Q6_Vsf_vsub_VsfVsf(v, hvx_vec_splat_f32(mean));
                    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(d, d));
                }
                float tail = 0.0f;
                for (; i0 < ne0; ++i0) {
                    const float d = x[i0] - mean;
                    tail += d * d;
                }
                sumsq += group_norm_reduce_sum_f32(Q6_Vsf_equals_Vqf32(acc)) + tail;
            }
        }
        const float variance = sumsq * inv_n;
        const float scale    = 1.0f / sqrtf(variance + gctx->eps);
        const HVX_Vector vscale = hvx_vec_splat_f32(scale);
        const HVX_Vector vmean  = hvx_vec_splat_f32(mean);

        // Pass 3: normalise.
        for (uint32_t i2 = c0; i2 < c1; ++i2) {
            for (uint32_t i1 = 0; i1 < ne1; ++i1) {
                const size_t off = (size_t) i1 * nb1 + (size_t) i2 * nb2 + (size_t) i3 * nb3;
                const float * x = (const float *) ((const uint8_t *) src->data + off);
                      float * y = (      float *) ((      uint8_t *) dst->data + off);
                if (scalar_only) {
                    for (uint32_t i0 = 0; i0 < ne0; ++i0) {
                        y[i0] = (x[i0] - mean) * scale;
                    }
                    continue;
                }
                uint32_t i0 = 0;
                for (; i0 + 32 <= ne0; i0 += 32) {
                    const HVX_Vector v = hvx_vmemu((const HVX_UVector *) (x + i0));
                    const HVX_Vector d = Q6_Vsf_vsub_VsfVsf(v, vmean);
                    hvx_vmemu((HVX_UVector *) (y + i0)) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(d, vscale));
                }
                for (; i0 < ne0; ++i0) {
                    y[i0] = (x[i0] - mean) * scale;
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ig1);
    FARF(HIGH, "group_norm %u/%u: groups %u:%u\n", ith, nth, ig0, ig1);
}

int op_group_norm(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;
    const struct htp_tensor * src = octx->src[0];

    // F32 only: sd.cpp's GroupNorm path produces F32 activations here, and the
    // reduction is written for F32.
    if (src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src->nb[0] != sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_groups = (uint32_t) octx->op_params[0];
    float eps;
    memcpy(&eps, &octx->op_params[1], sizeof(float));

    const uint32_t ne2 = src->ne[2];
    if (n_groups == 0 || ne2 % n_groups != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // A group spans a channel range inside one batch element, so batch is part
    // of the outer index space.
    if (src->ne[3] != dst->ne[3] || src->ne[2] != dst->ne[2] ||
        src->ne[1] != dst->ne[1] || src->ne[0] != dst->ne[0]) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_group_norm_context gctx = {
        .octx         = octx,
        .n_groups     = n_groups,
        .cpg          = ne2 / n_groups,
        .eps          = eps,
        .total_groups = n_groups * src->ne[3],
    };
    gctx.group_elems = src->ne[0] * src->ne[1] * gctx.cpg;

    if (gctx.total_groups == 0) {
        return HTP_STATUS_OK;
    }

    gctx.groups_per_thread = fastdiv(gctx.total_groups + octx->n_threads - 1, &octx->n_threads_div);
    gctx.group_start       = 0;

    FARF(HIGH, "group_norm: (%ux%ux%ux%u) groups=%u cpg=%u eps=%f type=%u\n",
         src->ne[0], src->ne[1], src->ne[2], src->ne[3], n_groups, gctx.cpg, eps, dst->type);

    work_queue_run(octx->ctx->work_queue, group_norm_thread, &gctx, octx->n_threads);

    return HTP_STATUS_OK;
}
