#ifndef HTP_SUM_ROWS_OPS_H
#define HTP_SUM_ROWS_OPS_H

#include <stdint.h>

struct htp_sum_rows_kernel_params {
    uint32_t n_hvx;
    uint32_t elems_per_hvx;
};

// opt-in flag carried in the op's op_params[15]: clamp the summed value with op_params[0]/[1]
// (f32 bit patterns). 0 = plain SUM_ROWS (ggml zeroes op_params, so the default path is unchanged).
#define HTP_SUM_ROWS_FLAG_CLAMP 1

#if defined(__cplusplus)
static_assert(sizeof(struct htp_sum_rows_kernel_params) <= 128, "htp_sum_rows_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_sum_rows_kernel_params) <= 128, "htp_sum_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SUM_ROWS_OPS_H
