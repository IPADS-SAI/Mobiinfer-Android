#ifndef GGML_HEXAGON_OPPOLL_H
#define GGML_HEXAGON_OPPOLL_H

#include <cstdint>

// GGML_HEXAGON_OPPOLL=2 ("auto") spins on batch completion only for
// decode-sized batches and blocks for prefill/load/init.  The decision is
// derived from the number of activation rows (tokens) in the graph that is
// being enqueued, never from the number of ops (a prefill graph and a verify
// graph have the same op count but very different token counts).
//
// This header is deliberately dependency-free so the rule can be unit-tested
// on the host without the Hexagon SDK / DSP toolchain.  See
// tests/test-oppoll-hint.cpp.
//
// Conservative by construction: no candidate, conflicting candidates, or a
// non-positive row count all yield poll_decode() == false, i.e. the blocking
// wait.  A false negative only loses the decode spin on an unusual graph; a
// false positive would burn host CPU during prefill/load, which is exactly
// what =2 is meant to avoid.
//
// The row-count fix below is per graph: MUL_MAT_ID takes its token count from
// the ids tensor (src[2]->ne[1]), not from the expert activation (src[1]),
// whose ne[1] is n_expert_used.  The batch-level AND in ggml-hexagon.cpp
// (`graph_poll = graph_poll && sess->open_batch_poll`) is deliberately kept:
// it is a conservative amplification when several graphs share one open batch,
// not the cause of the single-graph false negative.  Fix the row accounting,
// keep the AND.

// 4-row speculative verify and single-token draft decode are <= 8.  Prefill
// ubatches are normally much larger (e.g. 128), so 8 keeps a safe margin.
static constexpr uint32_t GGML_HEXAGON_OPPOLL_DECODE_MAX_TOKENS = 8;

struct ggml_hexagon_oppoll_hint {
    bool     have_candidate = false;
    bool     consistent     = true;
    uint64_t n_tokens       = 0;
    uint32_t n_candidates   = 0;

    // Register the row count of one MUL_MAT / MUL_MAT_ID activation operand.
    void add(uint64_t rows) {
        ++n_candidates;
        if (!have_candidate) {
            have_candidate = true;
            n_tokens       = rows;
            return;
        }
        if (rows != n_tokens) {
            consistent = false;
        }
    }

    bool poll_decode() const {
        return have_candidate && consistent && n_tokens > 0 &&
               n_tokens <= GGML_HEXAGON_OPPOLL_DECODE_MAX_TOKENS;
    }
};

// Transport-neutral op kinds so the row-count rule below can be exercised by
// the host test without ggml headers.
enum ggml_hexagon_oppoll_op {
    GGML_HEXAGON_OPPOLL_OP_OTHER      = 0,
    GGML_HEXAGON_OPPOLL_OP_MUL_MAT    = 1,
    GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID = 2,
};

// Register one compute node's token row count.
//
//   MUL_MAT:    ne is src[1]->ne = {n_embd, rows, ...}; the DSP processes
//               ne[1]*ne[2]*ne[3] activation rows.
//   MUL_MAT_ID: ne is still src[1]->ne (the expert activation, e.g.
//               [n_ff, n_expert_used, n_tokens]); its ne[1] is the number of
//               experts used, NOT the token count.  Pass ids_ne =
//               src[2]->ne (the ids tensor, [n_expert_used, n_tokens, 1, 1]);
//               the token row count is ids_ne[1].
//
// A null ids_ne (missing src[2]), or ids_ne[1] <= 0, records 0 rows (block).
// It never falls back to src[1]'s product: for the down projection that would
// be n_expert_used * n_tokens (32 for top-8 x 4), a guess that made every
// MoE graph inconsistent.  Non-positive MUL_MAT dims are likewise 0 rows.
// Returns true when the node is a matmul candidate.
inline bool ggml_hexagon_oppoll_add_node(ggml_hexagon_oppoll_hint & hint, int op,
                                         const int64_t ne[4], const int64_t * ids_ne = nullptr) {
    if (op == GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID) {
        if (ids_ne == nullptr || ids_ne[1] <= 0) {
            hint.add(0);
            return true;
        }
        hint.add((uint64_t) ids_ne[1]);
        return true;
    }
    if (op != GGML_HEXAGON_OPPOLL_OP_MUL_MAT) {
        return false;
    }
    if (ne[1] <= 0 || ne[2] <= 0 || ne[3] <= 0) {
        hint.add(0);
        return true;
    }
    const uint64_t rows = (uint64_t) ne[1] * (uint64_t) ne[2] * (uint64_t) ne[3];
    hint.add(rows);
    return true;
}

// Effective dspqueue_read() wait policy.
//   0 -> always block (DSPQUEUE_TIMEOUT), historical
//   1 -> always spin  (timeo 0), historical
//   2 -> spin only when this batch was classified as decode
// A batch without a hint is false (block); any other mode blocks.
inline bool ggml_hexagon_oppoll_spin(int mode, bool batch_poll_decode) {
    if (mode == 1) {
        return true;
    }
    if (mode == 2) {
        return batch_poll_decode;
    }
    return false;
}

// Normalize a parsed GGML_HEXAGON_OPPOLL value.  0/1/2 pass through; anything
// else is clamped to 2 (auto) and *clamped is set so the caller can warn once.
inline int ggml_hexagon_oppoll_normalize(long value, bool & clamped) {
    if (value < 0 || value > 2) {
        clamped = true;
        return 2;
    }
    clamped = false;
    return (int) value;
}

#endif // GGML_HEXAGON_OPPOLL_H
