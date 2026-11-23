#ifndef HTP_COMMON_H
#define HTP_COMMON_H

#include "htp-opnode.h"
#include "htp-repack.h"

extern int    opt_arch;
extern size_t opt_ndev;
extern size_t opt_nhvx;
extern int    opt_nhmx;
extern size_t opt_vmem;
extern int    opt_etm;
extern int    opt_verbose;
extern int    opt_profile;
extern bool   opt_hostbuf;
extern bool   opt_dma64;
extern int    opt_power_mode;

extern size_t opt_mbuf_dyn;
extern size_t opt_mbuf_static;
extern size_t opt_mbuf_total;

extern int    opt_mm_select;
extern bool   opt_moe_hmx_pipeline;
extern bool   opt_moe_hmx_serial;
// MUL_MAT_ID HVX-first window (n_tokens <= value goes HVX first; HMX stays the
// fallback for shapes HVX cannot stage). 0 = derive from the session's HVX count.
extern int    opt_mm_moe_hvx_first_max_nrows;
// HVX-first attempt cap for MUL_MAT_ID: with >= HTP_MOE_HVX_WINDOW_NHVX_PLATEAU HVX
// contexts, batches up to HTP_MOE_HVX_WINDOW_PLATEAU tokens try HVX first; fewer contexts
// keep HTP_MM_MOE_HMX_MIN_NROWS as a conservative fallback. The plateau is the smallest
// measured HVX-context count for the 32-token window: v79/SM8750 reports 6 and v81/SM8850
// reports 8, and both are faster on HVX at n=18/32 (nubia v79 same-binary A/B: -76.1% /
// -67.4%; n=64 unchanged). This is a session-derived heuristic for *which solver to try
// first*; whether a concrete shape fits the VTCM budget is decided by the HVX layout
// solver, whose UNSUPPORTED verdict falls back to HMX (also for shapes above the window).
// One policy function so the kparams solver and the session log can never disagree.
#define HTP_MOE_HVX_WINDOW_NHVX_PLATEAU 6
#define HTP_MOE_HVX_WINDOW_PLATEAU      32
int htp_moe_hvx_first_window(uint32_t n_hvx);
// PR-06: dense MUL_MAT narrow-batch window (-1 = derive from n_hvx, 0 = never,
// >0 = explicit m limit).  Independent of the MUL_MAT_ID window above.
extern int    opt_mm_dense_hvx_max_rows;
extern int    opt_fa_select;
extern int    opt_fa_split;
extern int    opt_li_select;
extern int    opt_gdn_select;
extern int    opt_ar_select;
extern int    opt_ar_scatter;

extern int    opt_opbatch;
extern int    opt_opqueue;
extern int    opt_optrace;
extern int    opt_oppoll;
extern int    opt_opfusion;
extern int    opt_rt_repack;

#ifndef HEX_VERBOSE
#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)
#endif

static inline uint32_t ggml_hexagon_tensor_get_flags(const struct ggml_tensor * t) {
    return (uint32_t) (uintptr_t) t->extra;
}

static inline void ggml_hexagon_tensor_set_flags(struct ggml_tensor * t, uint32_t flags) {
    t->extra = (void *) (uintptr_t) (ggml_hexagon_tensor_get_flags(t) | flags);
}

static inline void ggml_hexagon_tensor_clear_flags(struct ggml_tensor * t, uint32_t flags) {
    t->extra = (void *) (uintptr_t) (ggml_hexagon_tensor_get_flags(t) & ~flags);
}

static inline bool ggml_hexagon_tensor_is_fuseable(const struct ggml_tensor * t) {
    return (ggml_hexagon_tensor_get_flags(t) & HTP_TENSOR_FUSEABLE) != 0;
}

static inline bool ggml_hexagon_tensor_is_tiled(const struct ggml_tensor * t) {
    return (ggml_hexagon_tensor_get_flags(t) & HTP_TENSOR_TILED) != 0;
}

// `extra` is shared with every other ggml backend. Only interpret it as Hexagon
// flags when the tensor's storage is owned by this backend's alloc-only buffer
// type. Tensors that are not allocated yet can only carry a marker written by
// ggml_backend_mark_moe_expert_compact(); the marker is converted to real flags
// in buffer_init_tensor().
bool ggml_hexagon_tensor_extra_is_owned(const struct ggml_tensor * t);
bool ggml_hexagon_tensor_is_pending_compact(const struct ggml_tensor * t);
void ggml_hexagon_tensor_mark_pending_compact(struct ggml_tensor * t);
void ggml_hexagon_tensor_forget_pending_compact(struct ggml_tensor * t);

static inline bool ggml_hexagon_tensor_is_q3k_compact(const struct ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_Q3_K) {
        return false;
    }
    const auto * bank = t->view_src ? t->view_src : t;
    // A pending mark is a token, not a flag word. It must be recognized before
    // the buffer is attached; buffer_init_tensor() consumes it and writes the
    // real HTP_TENSOR_Q3K_COMPACT flag afterwards.
    if (ggml_hexagon_tensor_is_pending_compact(bank)) {
        return true;
    }
    if (!ggml_hexagon_tensor_extra_is_owned(bank)) {
        // A foreign buffer may keep a private pointer or integer in `extra`.
        // Never decode those bits as Hexagon flags (and never abort on them).
        return false;
    }
    const uintptr_t flags = (uintptr_t) bank->extra;
    constexpr uintptr_t allowed = HTP_TENSOR_WEIGHT | HTP_TENSOR_TILED | HTP_TENSOR_FENCE |
                                  HTP_TENSOR_FUSEABLE | HTP_TENSOR_Q3K_COMPACT;
    if ((flags & HTP_TENSOR_Q3K_COMPACT) == 0) {
        return false;
    }
    // The op encoding forwards the compact flag to the DSP, so a tensor with the
    // compact bit plus unknown flag bits would have the host write it as tiled
    // while the DSP reads it as compact. Refuse to guess: report and stop.
    if ((flags & ~allowed) != 0) {
        GGML_LOG_ERROR("ggml-hex: tensor %s : compact Q3_K set with unexpected flags %#x\n",
                       t->name, (unsigned) (flags & ~allowed));
        GGML_ABORT("ggml-hex: compact Q3_K tensor carries unknown flag bits");
    }
    return true;
}

static inline bool ggml_hexagon_tensor_can_tile(const struct ggml_tensor * t) {
    if (opt_rt_repack == 0) {
        return false;
    }
    if (t->buffer) {
        if (ggml_backend_buft_is_host(t->buffer->buft) ||
            ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            return false;
        }
    }
    return htp_is_tiled_weight_tensor(t);
}

static inline bool ggml_hexagon_tensor_will_tile(const struct ggml_tensor * t) {
    return ggml_hexagon_tensor_is_tiled(t) || ggml_hexagon_tensor_is_q3k_compact(t) || ggml_hexagon_tensor_can_tile(t);
}

static inline bool htp_tensors_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t b0 = (uintptr_t) b->data;
    const uintptr_t a1 = a0 + ggml_nbytes(a);
    const uintptr_t b1 = b0 + ggml_nbytes(b);

    return a0 < b1 && b0 < a1;
}

static inline bool htp_can_row_partition(const struct ggml_tensor * t) {
    if (t->ne[1] > 1 && (t->nb[1] & 127) != 0) return false;
    if (t->ne[2] > 1 && (t->nb[2] & 127) != 0) return false;
    if (t->ne[3] > 1 && (t->nb[3] & 127) != 0) return false;
    return true;
}

struct ggml_hexagon_session;

const char * htp_session_name(const struct ggml_hexagon_session * sess);
size_t       htp_session_vtcm_size(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_n_hvx(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_n_hmx(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_mdev_count(const struct ggml_hexagon_session * sess);
bool         htp_session_has_capability(const struct ggml_hexagon_session * sess, uint32_t cap);
// True when the tensor's buffer will be mapped with the extended (DMA64) mapping, which the
// fused ROWNORM kernel cannot access. Uses the same decision as the enqueue path.
bool         htp_session_tensor_is_extended(const struct ggml_hexagon_session * sess, const struct ggml_tensor * t);

htp_op_code  htp_op_remap(const struct ggml_tensor * t);

#endif // HTP_COMMON_H
