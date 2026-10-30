#ifndef HTP_ROPE_APPLY_OPS_H
#define HTP_ROPE_APPLY_OPS_H

#include <stdint.h>

#define HTP_ROPE_APPLY_SPAD_ROWS  8
#define HTP_ROPE_APPLY_SPAD_SLOTS 4
#define HTP_ROPE_APPLY_BATCH      (HTP_ROPE_APPLY_SPAD_ROWS * HTP_ROPE_APPLY_SPAD_SLOTS)

// dst[row, 0:ne0] = pe[token] applied to src0[row, 0:ne0] as half_dim 2x2 rotations.
// row indexes (head, token, batch) with token varying fastest, matching the CPU reference.
struct htp_rope_apply_kernel_params {
    uint32_t n_threads;
    uint32_t n_rows;
    uint32_t n_rows_per_thread;
    uint32_t ne0;          // elements per row (even)
    uint32_t half_dim;     // ne0 / 2
    uint32_t ne2;          // tokens per head
    uint32_t ne1;          // heads
    uint32_t interleaved;
    uint32_t src0_row_stride;  // bytes between rows of src0 (head-major)
    uint32_t src0_tok_stride;  // bytes between tokens within a head
    uint32_t src0_bat_stride;  // bytes between batches
    uint32_t pe_row_stride;    // bytes per pe token entry (all half_dim matrices)
    uint32_t pe_mat_stride;    // bytes per pe matrix entry
    uint32_t pe_nb0;           // bytes between the 2 columns of a matrix
    uint32_t pe_nb1;           // bytes between the 2 rows of a matrix
    uint32_t spad_row_bytes;   // VTCM bytes per staged row (128B aligned)
    uint32_t vtcm_bytes;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_rope_apply_kernel_params) <= 128, "htp_rope_apply_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_rope_apply_kernel_params) <= 128, "htp_rope_apply_kernel_params is too large");
#endif

int op_rope_apply(struct htp_ops_context * octx);

#endif /* HTP_ROPE_APPLY_OPS_H */
