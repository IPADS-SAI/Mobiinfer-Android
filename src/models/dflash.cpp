#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-context.h" // context-owned DFlash A4 state (llama_dflash_a4_state)
#include "llama-ext.h" // T10 aux head channel: llama_set_aux_head / llama_get_aux_logits (C++ linkage)

#include <algorithm>
#include <vector>

// ===========================================================================
// G2-A4 (draft-pool) support
//
// The A4 arm keeps the 4-row target verify batch unchanged and only replaces the
// *draft* side full-vocab LM head with a restricted one:
//
//   block  = 7 rows [anchor, MASK x 6] at positions q..q+6, one ubatch
//   rows 0..head_k-1   -> candidates (propagated by the Markov chain)
//   rows head_k+1..    -> lookahead rows, their post-norm hidden goes to the
//                         target aux head, whose logits build the next pool
//
// In pool mode the graph
//   * gathers n_pool lm_head rows per pool (first + shared) and applies them to the
//     head_k candidate hidden rows -> base logits [n_pool, head_k]
//   * runs the Markov chain inside that restricted space
//       z_i = base_i[pool_i] + W2[pool_i] . W1[prev_i],  prev_0 = anchor token
//     where W1 is gathered full (all vocab rows) and W2 only for the pool rows
//   * publishes z through the masked pre-norm channel (see the layout note below)
//
// PLUMBING:
//   * the arm topology (block/head/aux rows, pool size) is owned by the draft
//     context and copied into llm_graph_params, so it participates in graph
//     reuse and cannot leak between two contexts
//   * the per-decode pool ids / anchor / mode stay in the context-owned state and
//     are read by the graph input at set_input() (still one drafting sequence per
//     draft context, matching --parallel 1)
//   * the aux hidden rows and z are published by *reusing* the masked pre-norm
//     channel (llama_get_embeddings_pre_norm) instead of a dedicated graph output;
//     layout per block of block_rows:  slot 0..aux_rows-1 = aux hidden rows,
//                                     slot head_k..head_k+aux_rows-1 = z[0..n_pool) padded
//   * the dev-only pool source (draft's own full-vocab rows) is not the A4 arm
// ===========================================================================

namespace {

// vocab-id ordered pool -> the ids are used verbatim for both the lm_head gather and
// the W2 gather, so the two gathers are guaranteed to be in the same order.
bool a4_ids_valid(const std::vector<int32_t> & ids, int32_t n_vocab) {
    for (int32_t id : ids) {
        if (id < 0 || id >= n_vocab) {
            return false;
        }
    }
    return true;
}

} // namespace

// `llama_dflash_a4_*` are declared (with LLAMA_API) in llama-ext.h. C++ linkage there and here.

// whether the loaded draft model really carries the DSpark Markov head. Selecting the
// `draft-dspark` speculative type is not proof: a plain DFlash GGUF would pass that check and
// then abort in the pool graph.
bool llama_model_dflash_a4_supported(const struct llama_model * model) {
    if (model == nullptr || model->arch != LLM_ARCH_DFLASH) {
        return false;
    }
    const auto * model_dflash = static_cast<const llama_model_dflash *>(model);
    return model_dflash->dspark_markov_w1 != nullptr && model_dflash->dspark_markov_w2 != nullptr;
}

bool llama_dflash_a4_supported(const struct llama_context * ctx_dft) {
    return ctx_dft != nullptr && llama_model_dflash_a4_supported(llama_get_model(ctx_dft));
}

// arm the A4 draft graph for a single draft context (called from common/speculative.cpp)
void llama_dflash_a4_arm(
        struct llama_context * ctx_dft,
        int32_t n_pool, int32_t head_k, int32_t aux_rows, int32_t block_rows) {
    auto & st = ctx_dft->dflash_a4();
    st.armed      = n_pool > 0 && head_k > 0 && block_rows > head_k;
    st.n_pool     = n_pool;
    st.head_k     = head_k;
    st.aux_rows   = aux_rows;
    st.block_rows = block_rows;
    st.pool_mode  = false;
    st.decode_active = false;
    st.seq_id     = -1;
    st.anchor     = -1;
    st.ids_first.assign(n_pool > 0 ? n_pool : 0, 0);
    st.ids_shared.assign(n_pool > 0 ? n_pool : 0, 0);
}

// release the A4 graph state when its owning speculative implementation is destroyed
void llama_dflash_a4_disarm(struct llama_context * ctx_dft) {
    ctx_dft->dflash_a4() = {};
}

// clear the per-decode A4 marker after the draft decode (success or failure)
void llama_dflash_a4_clear_decode(struct llama_context * ctx_dft) {
    ctx_dft->dflash_a4().decode_active = false;
}

// fill the per-decode pool state (called right before the draft llama_decode)
void llama_dflash_a4_set_pool(
        struct llama_context * ctx_dft, llama_seq_id seq_id,
        const int32_t * ids_first, const int32_t * ids_shared, int32_t n_pool,
        int32_t anchor, bool pool_mode) {
    auto & st = ctx_dft->dflash_a4();
    GGML_ASSERT(st.armed && "G2-A4: set_pool() before arm()");
    GGML_ASSERT(n_pool == st.n_pool && "G2-A4: pool size changed between arm and set_pool");

    if (st.seq_id >= 0 && st.seq_id != seq_id) {
        // a different sequence is drafting -> the previous pool cannot be reused
        GGML_ABORT("%s: G2-A4 prototype supports a single drafting sequence (seq %d -> %d)\n",
                __func__, (int) st.seq_id, (int) seq_id);
    }

    st.seq_id    = seq_id;
    st.anchor    = anchor;
    st.pool_mode = pool_mode;
    // mark exactly this decode as the A4 draft block; an ordinary decode on the same context
    // (same token count, all logits) must not be mistaken for it
    st.decode_active = true;
    if (pool_mode && n_pool > 0) {
        GGML_ASSERT(ids_first && ids_shared && "G2-A4: missing pool ids");
        st.ids_first .assign(ids_first,  ids_first  + n_pool);
        st.ids_shared.assign(ids_shared, ids_shared + n_pool);

        const llama_model * model   = llama_get_model(ctx_dft);
        const int32_t       n_vocab = (int32_t) llama_vocab_n_tokens(llama_model_get_vocab(model));
        GGML_ASSERT(n_vocab > 0);
        GGML_ASSERT(a4_ids_valid(st.ids_first,  n_vocab) &&
                    a4_ids_valid(st.ids_shared, n_vocab) &&
                    "G2-A4: pool id outside of the vocabulary");
    }
}

namespace {

// graph inputs of the A4 pool arm: pool ids, anchor token and the block row indices
class llm_graph_input_a4_pool : public llm_graph_input_i {
public:
    llm_graph_input_a4_pool(const llama_dflash_a4_state * state,
            int32_t n_pool, int32_t block_rows, int32_t head_k, int32_t aux_rows, bool pool_mode) :
        state(state), n_pool(n_pool), block_rows(block_rows), head_k(head_k), aux_rows(aux_rows),
        pool_mode(pool_mode) {}

    void set_input(const llama_ubatch * ubatch) override {
        // single block per ubatch (prototype): block rows are contiguous, block-major
        GGML_ASSERT(ubatch->n_seqs_unq == 1);
        GGML_ASSERT((int32_t) ubatch->n_tokens == block_rows);

        if (pool_mode) {
            GGML_ASSERT(state != nullptr && state->armed && state->pool_mode);
            GGML_ASSERT(state->ids_first.size()  == (size_t) n_pool);
            GGML_ASSERT(ids_first != nullptr);
            // ids_shared only exists when there is more than one candidate row (head_k > 1); a
            // graph input that is never used is not allocated and writing it would crash.
            GGML_ASSERT(ids_shared == nullptr || state->ids_shared.size() == (size_t) n_pool);

            ggml_backend_tensor_set(ids_first,  state->ids_first .data(), 0, sizeof(int32_t)*n_pool);
            if (ids_shared != nullptr) {
                ggml_backend_tensor_set(ids_shared, state->ids_shared.data(), 0, sizeof(int32_t)*n_pool);
            }

            GGML_ASSERT(anchor != nullptr);
            const int32_t anchor_v = state->anchor;
            ggml_backend_tensor_set(anchor, &anchor_v, 0, sizeof(int32_t));
        } else if (ids_first != nullptr) {
            // task-22 diagnostic (LLAMA_A4_FAKEGATHER): the full arm may carry the same cross-context
            // get_rows as the pool arm; keep the ids in range even when no real pool was built.
            std::vector<int32_t> ids(n_pool);
            const bool have = state != nullptr && state->ids_first.size() == (size_t) n_pool;
            for (int32_t i = 0; i < n_pool; ++i) {
                ids[i] = have ? state->ids_first[i] : (int32_t) i; // ids < n_pool <= n_embd < n_vocab
            }
            ggml_backend_tensor_set(ids_first,  ids.data(), 0, sizeof(int32_t)*n_pool);
            if (ids_shared != nullptr) {
                ggml_backend_tensor_set(ids_shared, ids.data(), 0, sizeof(int32_t)*n_pool);
            }
        }

        // candidate rows 0..head_k-1 (pool arm only) and aux (lookahead) rows head_k+1..block_rows-1
        if (idx_cand != nullptr) {
            std::vector<int32_t> idx_c(head_k);
            for (int32_t i = 0; i < head_k; ++i) {
                idx_c[i] = i;
            }
            ggml_backend_tensor_set(idx_cand, idx_c.data(), 0, sizeof(int32_t)*head_k);
        }

        if (idx_aux != nullptr) {
            std::vector<int32_t> idx_a(aux_rows);
            for (int32_t i = 0; i < aux_rows; ++i) {
                idx_a[i] = head_k + 1 + i;
            }
            ggml_backend_tensor_set(idx_aux, idx_a.data(), 0, sizeof(int32_t)*aux_rows);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const uint64_t expected_mask = head_k >= 64 ? ~(uint64_t) 0 : ((uint64_t) 1 << head_k) - 1;
        return params.a4_armed && params.a4_decode &&
            llama_graph_a4_output_mask(params.ubatch, block_rows) == expected_mask &&
            params.ubatch.n_tokens   == (uint32_t) block_rows &&
            params.ubatch.n_seqs_unq == 1 &&
            // the pool arm and the full arm have a different topology -> never reuse across modes
            pool_mode == params.a4_pool_mode &&
            n_pool    == params.a4_n_pool    &&
            head_k    == params.a4_head_k    &&
            aux_rows  == params.a4_aux_rows  &&
            block_rows == params.a4_block_rows;
    }

    const llama_dflash_a4_state * state;
    const int32_t n_pool;
    const int32_t block_rows;
    const int32_t head_k;
    const int32_t aux_rows;
    const bool    pool_mode;

    ggml_tensor * ids_first  = nullptr; // I32 [n_pool]
    ggml_tensor * ids_shared = nullptr; // I32 [n_pool]
    ggml_tensor * anchor     = nullptr; // I32 [1]
    ggml_tensor * idx_cand   = nullptr; // I32 [head_k]
    ggml_tensor * idx_aux    = nullptr; // I32 [aux_rows]
};

// pre-norm channel packing: [n_embd, block_rows] per block
//   slot 0..aux_rows-1            -> draft post-norm hidden of block rows head_k+1..block_rows-1
//   slot head_k..head_k+aux_rows-1-> z[0..n_pool) padded to n_embd (pool mode only)
//   remaining slots               -> zero padding
// Reusing the masked pre-norm channel is a prototype transfer hack: llama_get_embeddings_pre_norm()
// hands exactly n_outputs rows of n_embd floats back to the host.
ggml_tensor * a4_pack_hidden(
        llm_graph_context & g, ggml_tensor * h_aux, ggml_tensor * z_pad,
        int32_t n_embd, int32_t block_rows, int32_t head_k) {
    ggml_context * ctx0 = g.ctx0;

    // layout contract (shared with the reader in common/speculative.cpp):
    //   slot 0 .. aux_rows-1              -> aux hidden (block rows head_k+1 .. block_rows-1)
    //   slot head_k .. head_k+aux_rows-1  -> z[0..pool_size) (pool arm only)
    // The z block therefore starts at head_k, NOT at aux_rows: pad the aux part up to head_k first,
    // otherwise any aux_rows != head_k (i.e. block_rows != 2*head_k+1) would silently misalign z.
    ggml_tensor * packed = h_aux; // [n_embd, aux_rows]
    GGML_ASSERT(h_aux->ne[1] <= block_rows && "G2-A4: aux rows must fit in the block");
    GGML_ASSERT(head_k >= h_aux->ne[1] && "G2-A4: head_k must be >= aux_rows (see the config validation)");
    if (h_aux->ne[1] < head_k) {
        packed = ggml_pad(ctx0, packed, 0, (int) (head_k - h_aux->ne[1]), 0, 0);
    }

    if (z_pad != nullptr) {
        packed = ggml_concat(ctx0, packed, z_pad, 1);
    }
    if (packed->ne[1] < block_rows) {
        packed = ggml_pad(ctx0, packed, 0, (int) (block_rows - packed->ne[1]), 0, 0);
    }
    GGML_ASSERT(packed->ne[1] == block_rows && packed->ne[0] == n_embd &&
                "G2-A4: packed hidden must cover exactly block_rows (head_k + max(aux_rows, head_k) <= block_rows)");
    return packed;
}

} // namespace

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    LLAMA_LOG_INFO("%s: DFlash extract_layers = [", __func__);
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        LLAMA_LOG_INFO("%d%s", target_layer_ids[i], i + 1 < target_layer_ids.size() ? ", " : "");
    }
    LLAMA_LOG_INFO("]\n");

    // optional interleaved sliding-window attention with per-layer pattern array.
    // DFlash has a single rope, so the SWA rope == main rope.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
    //
    // TODO: only Qwen3-style backbones are supported for now; other backbones (e.g. Gemma4)
    //       need their own conversion path and graph tweaks
    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { dspark_markov_rank, n_vocab }, 0);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { dspark_markov_rank, n_vocab }, 0);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, 0);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long) dspark_markov_rank);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_pre_norm = cur;

    ggml_build_forward_expand(gf, cur);
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
// G2-A4: n_chain > 0 restricts the Markov chain to the first n_chain block rows
// (the proposal rows); the remaining rows keep their unbiased base logits so that they
// can be used as "base" pool logits. n_chain == 0 keeps the legacy full-block chain
// (and the confidence broadcast channel).
// n_block_rows: real ubatch rows per block (7 for the A4 block). When the head only consumes the
// head_k candidate rows, base->ne[1] is smaller than the block, so the anchor view must use this.
static void build_dspark_markov_head(
        llm_graph_context & g, const llama_model & model, ggml_tensor * tokens,
        int64_t n_chain, bool conf_channel, int64_t n_block_rows = 0) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && model.dspark_conf_proj && "DSpark markov/confidence weights not loaded");

    ggml_tensor * base = res->t_logits; // [n_vocab, n_tokens]
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    const auto it = model.gguf_kv.find("dflash.block_size");
    GGML_ASSERT(it != model.gguf_kv.end() && "DSpark draft requires 'dflash.block_size' in GGUF metadata");
    const int64_t block_size = std::stoi(it->second);
    GGML_ASSERT(block_size > 0);

    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DSpark markov head requires equal-size blocks");
    // runtime tokens per block in this ubatch (anchor + drafted positions), bounded by training block_size
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    // anchor (committed last) token of every block: token 0 of each block, i.e. a strided view
    const int64_t n_rows_blk   = n_block_rows > 0 ? n_block_rows : block_drafts;
    const size_t  token_stride = (size_t) n_rows_blk * tokens->nb[0];
    const size_t base_stride = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    // confidence head input: predicts per-position acceptance
    ggml_tensor * conf_inp = res->t_embd; // [n_embd, n_tok]

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
    //       token pick, not the Markov conditioning path
    const int64_t n_chained = n_chain > 0 ? std::min<int64_t>(n_chain, block_drafts) : block_drafts;

    for (int64_t i = 0; i < block_drafts; ++i) {
        // position i of every block: strided view [n_vocab, n_blocks]
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);

        // G2-A4: rows beyond the proposal rows keep their unbiased base logits (pool "base" rows)
        if (i >= n_chained) {
            cat = cat ? ggml_concat(ctx0, cat, base_i, 1) : base_i;
            continue;
        }

        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev);   // [R, n_blocks]
        ggml_tensor * bias    = ggml_mul_mat(ctx0, w2, w1_prev); // [n_vocab, n_blocks]
        ggml_tensor * col     = ggml_add(ctx0, base_i, bias);

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        if (conf_channel) {
            // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
            ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                                                    (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
            ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), w1_prev, 0);
            ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
            if (model.dspark_conf_proj_b) {
                conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
            }
            conf = ggml_sigmoid(ctx0, conf);

            cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;
        }

        if (i + 1 < n_chained) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    if (conf_channel) {
        GGML_ASSERT(cat_conf != nullptr);
        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);

        // note: broadcast the [1, n_tok] confidences to n_embd-wide rows to be able to reuse `llama_get_embeddings_nextn`
        conf = ggml_repeat(ctx0, conf, res->t_embd);
        res->t_h_pre_norm = conf;
        ggml_build_forward_expand(g.gf, conf);
    }

    res->t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

// G2-A4 pool arm: restricted draft LM head + in-pool Markov chain.
//
//   base_i = lm_head[pool_i] . h_i          -> [n_pool, 1]  (only the pool rows are gathered)
//   z_i    = base_i + W2[pool_i] . W1[prev_i]
//   prev_0 = anchor token (block row 0),  prev_{i+1} = pool_i[argmax(z_i)]
//
// W1 is gathered for every pool row (and once for the anchor), W2 only for the pool rows,
// so the lm_head gather and the W2 gather always use the same vocab-id ordering.
// The candidates are z[0..n_pool) per proposal row; they are published through the masked
// pre-norm channel together with the aux hidden rows (see a4_pack_hidden).
static void build_dspark_pool_head(
        llm_graph_context & g, const llama_model & model,
        ggml_tensor * output,
        ggml_tensor * output_s,
        const llm_graph_input_a4_pool * inp,
        int32_t n_embd) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && "DSpark pool arm requires the Markov weights");

    const int32_t n_pool     = inp->n_pool;
    const int32_t head_k     = inp->head_k;
    const int32_t aux_rows   = inp->aux_rows;
    const int32_t block_rows = inp->block_rows;

    GGML_ASSERT(n_pool <= n_embd && "G2-A4: pool larger than n_embd is not supported by the packing channel");

    ggml_tensor * h = res->t_embd; // [n_embd, block_rows] post final-norm draft hidden
    GGML_ASSERT(h != nullptr && h->ne[0] == n_embd && h->ne[1] == block_rows);

    ggml_tensor * h_c = ggml_get_rows(ctx0, h, inp->idx_cand); // [n_embd, head_k]
    ggml_tensor * h_a = ggml_get_rows(ctx0, h, inp->idx_aux);  // [n_embd, aux_rows]
    GGML_ASSERT(h_a->ne[1] == aux_rows && "G2-A4: aux rows must be the trailing lookahead rows");

    // restricted lm_head + chain weights, in vocab-id pool order.
    // The shared pool rows are only consumed by candidate rows 1..head_k-1; with head_k == 1 the
    // gather would not feed the graph, so do not create (or upload) it at all.
    const bool need_shared = head_k > 1;

    ggml_tensor * lm_first  = ggml_get_rows(ctx0, output, inp->ids_first);  // [n_embd, n_pool]
    ggml_tensor * lm_shared = need_shared ? ggml_get_rows(ctx0, output, inp->ids_shared) : nullptr;

    ggml_tensor * w1_first  = ggml_get_rows(ctx0, w1, inp->ids_first);  // [R, n_pool]
    ggml_tensor * w1_shared = need_shared ? ggml_get_rows(ctx0, w1, inp->ids_shared) : nullptr;
    ggml_tensor * w2_first  = ggml_get_rows(ctx0, w2, inp->ids_first);  // [R, n_pool]
    ggml_tensor * w2_shared = need_shared ? ggml_get_rows(ctx0, w2, inp->ids_shared) : nullptr;

    // prev_0 = anchor (block row 0 token): W1 is gathered full, only a single row is read
    ggml_tensor * prev = ggml_get_rows(ctx0, w1, inp->anchor); // [R, 1]

    ggml_tensor * cat = nullptr; // [n_pool, head_k]
    for (int32_t i = 0; i < head_k; ++i) {
        const bool first = (i == 0);

        ggml_tensor * lm_i = first ? lm_first  : lm_shared;
        ggml_tensor * w2_i = first ? w2_first  : w2_shared;
        ggml_tensor * w1_i = first ? w1_first  : w1_shared;

        ggml_tensor * h_i  = ggml_view_2d(ctx0, h_c, n_embd, 1, h_c->nb[1], (size_t) i*h_c->nb[1]);
        ggml_tensor * base = ggml_mul_mat(ctx0, lm_i, h_i);                    // [n_pool, 1]
        if (output_s != nullptr) {
            // same scale semantics as the full A4 arm (build_lora_mm(..., output_s)) and the target
            // aux head, otherwise the Markov bias would get a different relative weight.
            base = ggml_mul(ctx0, base, output_s);
        }
        ggml_tensor * z_i  = ggml_add(ctx0, base, ggml_mul_mat(ctx0, w2_i, prev));

        cat = cat ? ggml_concat(ctx0, cat, z_i, 1) : z_i;

        if (i + 1 < head_k) {
            ggml_tensor * idx_prev = ggml_argmax(ctx0, z_i);       // [1]
            prev = ggml_get_rows(ctx0, w1_i, idx_prev);            // [R, 1]
        }
    }

    ggml_tensor * z = ggml_cont(ctx0, cat); // [n_pool, head_k]
    ggml_tensor * z_pad = z;
    if (n_pool < n_embd) {
        z_pad = ggml_pad(ctx0, z, (int) (n_embd - n_pool), 0, 0, 0); // [n_embd, head_k], first n_pool cols = z
    }

    ggml_tensor * packed = a4_pack_hidden(g, h_a, z_pad, n_embd, block_rows, head_k);
    res->t_h_pre_norm = packed;
    ggml_build_forward_expand(g.gf, packed);

    // no full-vocab logits in the pool arm
    res->t_logits = nullptr;
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos  = build_inp_pos();

    // optional iSWA: pick the matching attention input
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto build_rope = [&](ggml_tensor * cur, ggml_tensor * pos) {
        return rope_type == GGML_ROPE_TYPE_MROPE
            ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext(ctx0, cur, pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
    };

    // KV cache injection
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, inp_g);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_rope(Kcur, inp_pos);
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            if (use_iswa) {
                // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
                const bool    is_swa = hparams.is_swa(il);
                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
            } else {
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, noise_norm);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

        Qcur = build_rope(Qcur, inp_pos);
        Kcur = build_rope(Kcur, inp_pos);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head from the target model (shared via ctx_other)
    auto * output = model.output;
    ggml_tensor * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires the target model's output projection");
        output = model_other->output;
        // G2-A4: the target output scale (LLM_TENSOR_OUTPUT/"scale") must be applied here as well,
        // otherwise the draft logits differ from the target head for any non-unit output_s.
        // TODO(G2-A4): the target's LoRA adapters are still not applied here (the draft context
        //              has its own lora set).
        output_s = model_other->output_s;
    }

    // The A4 layout only applies to the noise-block decode ([anchor, MASK x (block_rows-1)],
    // one block per sequence), and only when the caller explicitly marked *this* decode via
    // llama_dflash_a4_set_pool(). A normal token decode with the same token count (e.g. a
    // 7-token caller batch with all logits) must keep the legacy head/chain. The output mask is
    // part of the identity: only the first head_k rows may request logits.
    const uint64_t a4_mask = llama_graph_a4_output_mask(ubatch, params.a4_block_rows);
    const uint64_t a4_expected_mask = params.a4_head_k >= 64
        ? ~(uint64_t) 0 : ((uint64_t) 1 << params.a4_head_k) - 1;
    const bool a4_out_mask = ubatch.token != nullptr && a4_mask == a4_expected_mask;
    const bool a4_arm = params.a4_armed && params.a4_decode && a4_out_mask &&
        ubatch.n_seqs_unq == 1 && (int32_t) ubatch.n_tokens == params.a4_block_rows;
    if (params.a4_decode && !a4_arm) {
        // A marked A4 decode must never silently fall back to the legacy head: the caller still
        // unpacks pre-norm rows using the A4 layout, so a mismatched batch would corrupt the
        // aux/pool channel.
        GGML_ABORT("%s: A4 decode marked but batch does not match the A4 layout "
                "(tokens=%u block_rows=%d n_seqs=%u mask=0x%llx expected=0x%llx)\n",
                __func__, ubatch.n_tokens, params.a4_block_rows, ubatch.n_seqs_unq,
                (unsigned long long) a4_mask, (unsigned long long) a4_expected_mask);
    }
    // The pool contents (ids/anchor/mode) are per-decode state owned by the same context. The
    // graph input keeps a stable pointer and reads it at set_input(); the topology is fixed in
    // params and checked by allow_reuse().
    const llama_dflash_a4_state * a4_state = params.ctx != nullptr ? &params.ctx->dflash_a4() : nullptr;

    // G2-A4 pool arm: restricted head + in-pool chain. No full-vocab logits are produced.
    if (a4_arm && params.a4_pool_mode) {
        auto inp_a4 = std::make_unique<llm_graph_input_a4_pool>(
                a4_state, params.a4_n_pool, params.a4_block_rows, params.a4_head_k, params.a4_aux_rows,
                /*pool_mode=*/ true);

        inp_a4->ids_first  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_n_pool);
        // only candidate rows 1..head_k-1 read the shared pool; with head_k == 1 the tensor would
        // be an unused graph input (not allocated -> set_input would crash)
        inp_a4->ids_shared = params.a4_head_k > 1
            ? ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_n_pool)
            : nullptr;
        inp_a4->anchor     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        inp_a4->idx_cand   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_head_k);
        inp_a4->idx_aux    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_aux_rows);
        for (ggml_tensor * t : { inp_a4->ids_first, inp_a4->ids_shared, inp_a4->anchor,
                                 inp_a4->idx_cand, inp_a4->idx_aux }) {
            if (t != nullptr) {
                ggml_set_input(t);
            }
        }

        build_dspark_pool_head(*this, model, output, output_s, inp_a4.get(), (int32_t) n_embd);

        res->add_input(std::move(inp_a4));
        return;
    }

    // G2-A4 full arm: the full-vocab head is only needed for the head_k candidate rows. The
    // lookahead rows (head_k+1 .. block_rows-1) never go through the draft head - their logits come
    // from the target aux head. Applying the head to all 7 rows only added compute + 4 x 992 KB of
    // logits writes (the Q6_K weight traffic is the same for M=3 and M=7).
    auto inp_a4 = std::make_unique<llm_graph_input_a4_pool>(
            a4_state, params.a4_n_pool, params.a4_block_rows, params.a4_head_k, params.a4_aux_rows,
            /*pool_mode=*/ false);

    if (a4_arm) {
        inp_a4->idx_cand = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_head_k);
        inp_a4->idx_aux  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_aux_rows);
        ggml_set_input(inp_a4->idx_cand);
        ggml_set_input(inp_a4->idx_aux);

        ggml_tensor * h_c = ggml_get_rows(ctx0, res->t_embd, inp_a4->idx_cand); // [n_embd, head_k]

        cur = build_lora_mm(output, h_c, output_s); // [n_vocab, head_k]
        cb(cur, "result_output", -1);
        res->t_logits = cur;
        ggml_build_forward_expand(gf, cur);

        // task-22 diagnostic (LLAMA_A4_FAKEGATHER=1): run exactly the pool arm's cross-context gathers
        // on the FULL arm. If the full arm then starts to drift like the pool arm, that op has a side
        // effect on the target's weight/device buffers; if it stays lossless, the pool arm's other ops
        // (3x MUL_MAT / pad / concat) or the extra graph shape are to blame.
        static const bool fake_gather = [] {
            const char * v = std::getenv("LLAMA_A4_FAKEGATHER");
            return v && *v && *v != '0';
        }();
        if (fake_gather) {
            inp_a4->ids_first  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_n_pool);
            inp_a4->ids_shared = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, params.a4_n_pool);
            ggml_set_input(inp_a4->ids_first);
            ggml_set_input(inp_a4->ids_shared);

            ggml_tensor * f1 = ggml_get_rows(ctx0, output, inp_a4->ids_first);  // cross-context Q6_K gather
            ggml_tensor * f2 = ggml_get_rows(ctx0, output, inp_a4->ids_shared);
            ggml_format_name(f1, "a4_fakegather_0");
            ggml_format_name(f2, "a4_fakegather_1");
            ggml_set_output(f1); // must be materialised (never read on the host)
            ggml_set_output(f2);
            ggml_build_forward_expand(gf, f1);
            ggml_build_forward_expand(gf, f2);
        }
    } else {
        // keep the legacy (pre-A4) head for non-A4 decodes: adding output_s here would silently
        // change default dspark behaviour, so the scale stays an A4-only change.
        cur = build_lora_mm(output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
        ggml_build_forward_expand(gf, cur);
    }

    // DSpark: bias the draft logits with the Markov head
    if (model.dspark_markov_w1) {
        // G2-A4: only the proposal rows run the chain (in the A4 full arm the head above already
        // only produced those rows), and the pre-norm channel is repurposed for the aux-hidden/z
        // packing instead of the confidence broadcast.
        build_dspark_markov_head(*this, model, inp_tokens,
                a4_arm ? params.a4_head_k : 0, /*conf_channel=*/ !a4_arm,
                a4_arm ? params.a4_block_rows : 0);
    }

    if (a4_arm) {
        // publish the lookahead rows (block rows head_k+1..block_rows-1) for the target aux head;
        // they leave the draft context through the pre-norm channel
        ggml_tensor * h_a = ggml_get_rows(ctx0, res->t_embd, inp_a4->idx_aux); // [n_embd, aux_rows]

        ggml_tensor * packed = a4_pack_hidden(*this, h_a, nullptr, (int32_t) n_embd,
                                              params.a4_block_rows, params.a4_head_k);

        res->t_h_pre_norm = packed;
        ggml_build_forward_expand(gf, packed);
    }

    if (a4_arm || inp_a4->idx_aux != nullptr) {
        res->add_input(std::move(inp_a4));
    }
}
