#ifndef HTP_CONV2DHMX_OPS_H
#define HTP_CONV2DHMX_OPS_H

#include <stdint.h>

// Direct conv2d on HMX: the patch matrix is assembled in VTCM instead of being
// materialised to DDR by im2col-ops.c, which dominates VAE decode at 40.8 GB of
// DDR traffic (devdocs/06-perf-profile.md 100).
//
// Weight layout follows ggml conv2d: ne = [KW, KH, IC, OC], so the flat index is
// kw + KW*(kh + KH*(ic + IC*oc)) with OC outermost.
struct htp_conv2dhmx_kernel_params {
    uint32_t IC, IH, IW;
    uint32_t OC, OH, OW;
    uint32_t KH, KW;
    uint32_t stride;
    uint32_t pad;
    uint32_t total_rows;      // OW * OH * N patch rows
    uint32_t k_valid;         // real K = KW * KH * IC, in elements
    uint32_t patch_stride;    // k_valid rounded up to a multiple of 64, in elements.
                              // The tail is zero-filled so the extra dot products
                              // contribute nothing; this keeps shapes whose K is not
                              // a multiple of 64 on the fused path instead of CPU.
    uint32_t n_dot_tiles;     // ceil(patch_stride / 32)
    uint32_t n_col_tiles;     // ceil(OC / 32)
    uint32_t n_threads;
    uint32_t kernel_type;     // HTP_TYPE_F32 or HTP_TYPE_F16
    // Diagnostics replace the output: 1 weight, 2 activation, 3 accumulator,
    // 4 scalar reference, 5 raw/reference pairs, 6 index samples,
    // 7 staging/activation/weight/accumulator (4096 floats). Capacity is checked.
    // Host environment values must travel through params; DSP getenv is separate.
    uint32_t dump_tiles;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_conv2dhmx_kernel_params) <= 128, "htp_conv2dhmx_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_conv2dhmx_kernel_params) <= 128, "htp_conv2dhmx_kernel_params is too large");
#endif

int op_conv_2d_hmx(struct htp_ops_context * octx);

#endif /* HTP_CONV2DHMX_OPS_H */
