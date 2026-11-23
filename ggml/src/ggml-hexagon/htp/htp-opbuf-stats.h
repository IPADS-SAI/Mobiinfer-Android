#ifndef HTP_OPBUF_STATS_H
#define HTP_OPBUF_STATS_H

// P0 counters for prep_op_bufs() (T-0304, follow-up of T-0293).
//
// This header is shared by the DSP skel (C) and the host library / host unit
// tests (C++).  It must stay a pure data structure: no DSP, FastRPC or HAP
// dependencies, no allocation, no I/O.
//
// Semantics (all counters describe one op batch):
//   n_bufs       buffers in the batch (req->n_bufs)
//   b_reuse      buffers whose mapping was found by reuse_buf() in pass 0
//   drop_n       htp_munmap() calls made by drop_mmap() (non-empty entries only)
//   drop_bytes   sum of m->size for those calls
//   map_n        successful htp_mmap() calls for new mappings (mmap_buf())
//   map_bytes    sum of b->size for those successful calls
//   m_vmem       pass-1 sum of non-extended mapped bytes, before drops
//                (0 when the all-reuse early return skips pass 1)
//   e_vmem       pass-0 sum of non-reused, non-extended buffer bytes
//   max_vmem     ctx->max_vmem used by the drop decision
//   defrag_n     number of defragmentation passes entered (0 or 1)
//   scan_cmp     mmap-table entries examined by reuse_buf()/mmap_buf()
//
// Conservation (checked by bench/opbuf_stats.py and the host unit test):
//   without defrag:  n_bufs == b_reuse + map_n
//   with defrag:     n_bufs >= b_reuse, map_n >= n_bufs - b_reuse
// A defrag pass drops every non-extended mapping and remaps buffers that were
// already mapped in pass 1, so map_n can exceed n_bufs - b_reuse there.
//
// The counters never participate in allocation/reuse/drop decisions.  They are
// written after (or next to) the existing comparisons and do not change any
// branch condition or system call count.

#include <stdint.h>

// _Static_assert is a C11 keyword that clang warns about in C++
// (-Wc11-extensions); use the native form in each language.
#ifdef __cplusplus
#define HTP_OPBUF_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define HTP_OPBUF_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

// ABI magic carried in struct htp_opbatch_rsp.  Bump this whenever the
// response layout changes; host and skel must be rebuilt together.
#define HTP_OPBATCH_ABI_MAGIC 0x4f505331u // "OPS1"
#define HTP_OPBUF_STATS_VERSION 1u

struct htp_opbuf_stats {
    uint32_t n_bufs;
    uint32_t b_reuse;
    uint32_t drop_n;
    uint32_t map_n;
    uint32_t defrag_n;
    uint64_t drop_bytes;
    uint64_t map_bytes;
    uint64_t m_vmem;
    uint64_t e_vmem;
    uint64_t max_vmem;
    uint64_t scan_cmp;
};

static inline void htp_opbuf_stats_zero(struct htp_opbuf_stats * s) {
    s->n_bufs     = 0;
    s->b_reuse    = 0;
    s->drop_n     = 0;
    s->map_n      = 0;
    s->defrag_n   = 0;
    s->drop_bytes = 0;
    s->map_bytes  = 0;
    s->m_vmem     = 0;
    s->e_vmem     = 0;
    s->max_vmem   = 0;
    s->scan_cmp   = 0;
}

static inline void htp_opbuf_stats_init(struct htp_opbuf_stats * s, uint32_t n_bufs, uint64_t max_vmem) {
    htp_opbuf_stats_zero(s);
    s->n_bufs   = n_bufs;
    s->max_vmem = max_vmem;
}

// One mmap-table entry examined by a lookup loop.
static inline void htp_opbuf_stats_scan(struct htp_opbuf_stats * s, uint32_t entries) {
    s->scan_cmp += entries;
}

// One buffer served by an existing mapping (reuse_buf() hit).
static inline void htp_opbuf_stats_reuse(struct htp_opbuf_stats * s) {
    s->b_reuse++;
}

// One entry actually unmapped by drop_mmap().
static inline void htp_opbuf_stats_drop(struct htp_opbuf_stats * s, uint64_t bytes) {
    s->drop_n++;
    s->drop_bytes += bytes;
}

// One successful htp_mmap() for a new mapping.
static inline void htp_opbuf_stats_map(struct htp_opbuf_stats * s, uint64_t bytes) {
    s->map_n++;
    s->map_bytes += bytes;
}

// One defragmentation pass entered (after a pass-1 mmap failure).
static inline void htp_opbuf_stats_defrag(struct htp_opbuf_stats * s) {
    s->defrag_n++;
}

static inline void htp_opbuf_stats_add(struct htp_opbuf_stats * acc, const struct htp_opbuf_stats * s) {
    acc->n_bufs     += s->n_bufs;
    acc->b_reuse    += s->b_reuse;
    acc->drop_n     += s->drop_n;
    acc->map_n      += s->map_n;
    acc->defrag_n   += s->defrag_n;
    acc->drop_bytes += s->drop_bytes;
    acc->map_bytes  += s->map_bytes;
    acc->m_vmem     += s->m_vmem;
    acc->e_vmem     += s->e_vmem;
    acc->max_vmem   += s->max_vmem;
    acc->scan_cmp   += s->scan_cmp;
}

// Counter conservation law; see the comment at the top of this file.
static inline int htp_opbuf_stats_conserved(const struct htp_opbuf_stats * s) {
    if (s->n_bufs < s->b_reuse) {
        return 0;
    }
    if (s->defrag_n == 0) {
        return s->n_bufs == s->b_reuse + s->map_n;
    }
    return s->map_n >= s->n_bufs - s->b_reuse;
}

#endif // HTP_OPBUF_STATS_H
