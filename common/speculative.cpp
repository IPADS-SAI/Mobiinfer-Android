#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_pre_norm / llama_get_embeddings_pre_norm_ith (used by MTP)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <unordered_set>
#include <cstring>
#include <iomanip>
#include <stdexcept>

#include <limits>
#include <map>
#include <set>
#include <cinttypes>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// ---------------------------------------------------------------------------
// G2-A4 (draft-pool) interfaces
//
// The aux head channel (llama_set_aux_head / llama_set_aux_hidden / llama_get_aux_logits) and the
// draft-side arm/pool API (llama_dflash_a4_*) are all declared in ../src/llama-ext.h and defined
// by the owning translation units; there are no weak stubs or process-global slots here.
//
// The per-decode pool ids / anchor / mode are owned by the draft context (dflash_a4 state), so
// they cannot leak between two draft contexts and they take part in graph reuse.
// ---------------------------------------------------------------------------

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

// true if any device of the model is the Hexagon/HTP backend (device names are "HTP0", ...).
// Layers actually executed by the ordinary qwen35moe verify graph. NextN/MTP tail blocks are
// loaded as extra decoder blocks but skipped in the main pass, so a DSpark target_layer_ids entry
// pointing at them must be rejected even though it is inside llama_model_n_layer().
static int32_t common_a4_target_layer_limit(const llama_model * model_tgt) {
    char arch[64] = {};
    if (llama_model_meta_val_str(model_tgt, "general.architecture", arch, sizeof(arch)) < 0) {
        return 0;
    }
    const std::string key = std::string(arch) + ".nextn_predict_layers";
    char buf[32] = {};
    int32_t nextn = 0;
    if (llama_model_meta_val_str(model_tgt, key.c_str(), buf, sizeof(buf)) >= 0) {
        nextn = std::atoi(buf);
    }
    int32_t n_layer = llama_model_n_layer(model_tgt);
    if (nextn > 0 && nextn < n_layer) {
        n_layer -= nextn;
    }
    return n_layer;
}

static bool common_speculative_model_is_htp(const llama_model * model) {
    if (model == nullptr) {
        return false;
    }
    const int32_t n_devices = llama_model_n_devices(model);
    for (int32_t i = 0; i < n_devices; ++i) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);
        const char * name = dev != nullptr ? ggml_backend_dev_name(dev) : nullptr;
        if (name != nullptr && std::strncmp(name, "HTP", 3) == 0) {
            return true;
        }
    }
    return false;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // Actual number of proposals sent to the verifier after every caller-side truncation.
    virtual void set_verified(llama_seq_id /*seq_id*/, uint16_t /*n_verified*/) {}
    // A drafting cycle that produced no proposal (budget/block-fit/guard); no accept() follows.
    virtual void draft_empty(llama_seq_id /*seq_id*/) {}

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_pre_norm() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        LOG_INF("%s: adding speculative implementation 'draft-simple'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;


        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        LOG_INF("%s: adding speculative implementation 'draft-eagle3'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, params.draft.n_max, params.draft.n_min, params.draft.p_min);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_pre_norm(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        __func__, rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    bool need_embd() const override {
        return false;
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    // G2-A4 freezes the confidence truncation off (p_min == 0): a fixed-size block with a
    // truncated candidate list would desync the proposal count from the block layout.
    bool p_min_is_greedy() const { return params.p_min == 0.0f; }

    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool is_mrope = false;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    // -----------------------------------------------------------------------
    // G2-A4 (draft-pool) state
    // -----------------------------------------------------------------------
    struct a4_seq_state {
        int32_t last_accepted = -1; // accepted drafts of the last verify (-1 = none yet)
        int32_t last_n_draft  = 0;  // drafts proposed in the last cycle (may be < proposal_k when the budget/ctx is tight)
        bool    have_pool     = false;
        bool    pool_from_target = false; // true if the pool logits came from the target aux head

        std::vector<int32_t> pool_first;  // [pool_size] vocab ids, first-candidate pool
        std::vector<int32_t> pool_shared; // [pool_size] vocab ids, shared 2nd/3rd pool

        // draft post-norm hidden of the lookahead rows, [aux_rows * n_embd_dec], row-major
        std::vector<float> aux_hidden;
    };

    int32_t a4_block_rows  = 0;
    int32_t a4_proposal_k  = 0;
    int32_t a4_head_k      = 0;
    int32_t a4_aux_rows    = 0;
    int32_t a4_pool_size   = 0;
    bool    a4_armed       = false;
    bool    a4_pool_src_draft = false; // DEV ONLY fallback: pool from the draft's own full-vocab rows

    std::vector<a4_seq_state> a4;

    // cycle statistics (prototype observability: proves the pool/full switching happens)
    // T13 consumes the LLAMA_SPEC_A4_STATS=1 per-cycle lines:
    //   A4STAT seq=.. cycle=.. mode=pool|full n_draft=.. a_k=.. committed=.. trunc=.. src=target|draft
    bool    a4_stats = false;
    bool    a4_timing = false;           // LLAMA_SPEC_A4_TIMING=1: per-cycle stage timings (A4TIME lines)
    int64_t a4_n_pool_cycles = 0;
    int64_t a4_n_full_cycles = 0;
    int64_t a4_n_pool_target = 0;        // pool cycles whose logits came from the target aux head
    int64_t a4_n_trunc_cycles = 0;       // cycles where the caller truncated the draft (budget/ctx)
    int64_t a4_n_valid_cycles = 0;       // completed, non-truncated verify cycles
    int64_t a4_n_empty_cycles = 0;       // cycles that produced no proposal and never called accept()
    int64_t a4_sum_a_valid   = 0;        // accepted tokens, only for non-truncated cycles
    std::vector<int64_t> a4_hist_a_k;    // [proposal_k+1]
    std::vector<int64_t> a4_hist_n_draft;// [proposal_k+1]
    // pending per-cycle record, filled by draft() and emitted by accept() (which gets a_k)
    int64_t a4_last_pool_us = 0;         // pool construction (aux logits read + top-k), set by a4_prepare

    // ---- task-22 roadmap knobs (both default OFF) -------------------------------------------
    // (1a) LLAMA_A4_RESYNC_AFTER_REJECT=1: after an all-reject cycle (last_accepted == 0) emit no
    //      proposals for the next cycle, so that the next commit is decided by a 1-row decode
    //      (numerically identical to the plain sync decode).
    // (2)  LLAMA_A4_GUARD_TAU=<float>: read the target's verify-row margins (already on the host) and
    //      if min(top1-top2) over rows 0..3 is below tau, emit no proposals this cycle and force a
    //      full (no-pool) draft next cycle. tau = 0 disables the guard.
    static bool a4_resync_after_reject() {
        static const bool on = [] {
            const char * v = std::getenv("LLAMA_A4_RESYNC_AFTER_REJECT");
            return v && *v && *v != '0';
        }();
        return on;
    }

    static bool a4_guard_log() {
        static const bool on = [] {
            const char * v = std::getenv("LLAMA_A4_GUARD_LOG");
            return v && *v && *v != '0';
        }();
        return on;
    }

    static float a4_guard_tau() {
        static const float v = [] {
            const char * s = std::getenv("LLAMA_A4_GUARD_TAU");
            return (s && *s) ? (float) std::atof(s) : 0.0f;
        }();
        return v;
    }

    static bool a4_allow_htp_pool() {
        static const bool on = [] {
            const char * v = std::getenv("LLAMA_A4_ALLOW_HTP_POOL");
            // Only the exact string "1" enables the known-divergent HTP pool graph. Values such as
            // off/false/yes must not silently turn it on.
            return v != nullptr && std::strcmp(v, "1") == 0;
        }();
        return on;
    }

    int64_t a4_last_nospec_cycle = -100; // last cycle where we emitted 0 proposals (anti-stickiness)
    bool a4_guard_force_full = false;    // pending full request for the NEXT draft(), set by the guard
    bool a4_force_full_cycle = false;    // effective full request for THIS draft(), consumed from the pending flag
    bool a4_guard_unsafe     = false;    // set by the guard for THIS cycle (emit no proposals)
    bool a4_warned_htp_pool  = false;    // instance-local warning throttle (no cross-context races)
    bool a4_warned_htp_env   = false;    // invalid LLAMA_A4_ALLOW_HTP_POOL value warned once
    bool a4_warned_aux_logits = false;   // instance-local warning throttle
    bool a4_have_verify      = false;    // a target verify batch with output rows was processed
    std::vector<int32_t> a4_verify_rows; // batch token indices that actually have logits this verify
    int64_t a4_last_draft_end_us = 0;    // end of the previous draft(): the next entry measures the caller gap
    // per-mode cumulative stage sums for the A4TIME means
    int64_t a4tm_n_pool = 0, a4tm_n_full = 0;
    int64_t a4tm_draft_pool = 0, a4tm_draft_full = 0;
    int64_t a4tm_target_pool = 0, a4tm_target_full = 0;
    int64_t a4tm_pool_us_sum = 0, a4tm_aux_pool = 0, a4tm_aux_full = 0, a4tm_full_us_sum = 0;
    std::vector<int32_t> a4_pend_mode;     // [n_seq]
    std::vector<int32_t> a4_pend_ndraft;   // [n_seq] proposals produced by draft()
    std::vector<int32_t> a4_pend_verified; // [n_seq] proposals actually sent to the verifier

    int32_t a4_n_vocab = 0;              // target vocab size (pool ids / aux logits width)
    bool    a4_htp = false;              // HTP backend: the pool graph has a known divergence
    bool    a4_ctx_armed = false;        // the borrowed contexts were modified (needs restore)
    std::vector<int32_t> a4_idx_buf;     // [n_vocab] top-k scratch
    std::vector<float>   a4_merged_buf;  // [n_vocab] per-token max scratch

    bool a4_active() const { return a4_armed; }

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        // If anything below throws after the A4 arm touched the borrowed contexts, undo it before
        // the object is considered constructed (the destructor will not run in that case).
        struct a4_arm_guard {
            common_speculative_impl_draft_dflash * self;
            bool committed = false;
            ~a4_arm_guard() {
                if (!committed) {
                    self->a4_restore_contexts();
                }
            }
        } a4_guard{this};

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        // Validate A4 model eligibility before touching target_layer_ids: a plain GGUF paired
        // with draft-dspark must fail with a diagnostic, not a GGML_ASSERT during construction.
        if (this->params.a4()) {
            if (!is_dspark) {
                throw std::invalid_argument("G2-A4 (--spec-block-rows > 0) requires the DSpark Markov head");
            }
            if (!llama_dflash_a4_supported(ctx_dft)) {
                throw std::invalid_argument("G2-A4: the draft model did not load DSpark Markov weights "
                        "(a plain GGUF/DFlash model cannot run the pool chain)");
            }
        }

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n == 0) {
            throw std::invalid_argument("DFlash draft model has no target_layer_ids metadata "
                    "(not a DFlash/DSpark draft)");
        }
        if (this->params.a4()) {
            const int32_t n_layer_tgt = common_a4_target_layer_limit(model_tgt);
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                const int32_t id = target_layer_ids[k];
                if (id < 0 || id >= n_layer_tgt) {
                    throw std::invalid_argument("G2-A4: draft target_layer_ids[" + std::to_string(k) +
                            "]=" + std::to_string(id) + " is outside target layer range [0, " +
                            std::to_string(n_layer_tgt) + ") - the draft does not match this target");
                }
            }
        }

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u\n", __func__, block_size, mask_token_id, target_layer_ids_n);

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, DSpark yield a full block_size draft tokens
        const int32_t n_draft_max = is_dspark ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }

        // -------------------------------------------------------------------
        // G2-A4 (draft-pool): split of the runtime knobs
        //   block_rows=7 [anchor, MASK x 6] / proposal_k=3 / head_k=3 / aux_rows=3,
        //   pool_size=256. The block is decoded in a single ubatch and only the first
        //   proposal_k rows are read as candidates.
        // -------------------------------------------------------------------
        if (this->params.a4()) {
            if (!is_dspark) {
                throw std::invalid_argument("G2-A4 (--spec-block-rows > 0) requires the DSpark Markov head");
            }
            if (!p_min_is_greedy()) {
                throw std::invalid_argument("G2-A4 fixes p_min = 0 (confidence truncation is frozen off)");
            }
            if (n_embd_dec != n_embd_tgt) {
                throw std::invalid_argument("G2-A4: aux hidden needs the draft hidden size == target n_embd");
            }
            if (n_seq != 1 || llama_n_seq_max(ctx_dft) != 1) {
                throw std::invalid_argument("G2-A4: the draft-pool arm supports exactly one drafting "
                        "sequence (use --parallel 1)");
            }

            // Validate the *actually loaded* models before touching either context. The enum
            // `draft-dspark` only names the requested implementation; a plain DFlash GGUF has no
            // Markov weights and would otherwise abort in the pool graph.
            if (!llama_dflash_a4_supported(ctx_dft)) {
                throw std::invalid_argument("G2-A4: the draft model did not load DSpark Markov weights "
                        "(a plain DFlash GGUF cannot run the pool chain)");
            }

            char arch_tgt[64] = {};
            if (llama_model_meta_val_str(model_tgt, "general.architecture", arch_tgt, sizeof(arch_tgt)) < 0 ||
                std::strcmp(arch_tgt, "qwen35moe") != 0) {
                throw std::invalid_argument(std::string("G2-A4: the aux head is only implemented for the "
                        "qwen35moe target graph (target arch = ") + arch_tgt + ")");
            }

            const int32_t n_vocab_tgt = (int32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            const int32_t n_vocab_dft = (int32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
            if (n_vocab_tgt != n_vocab_dft) {
                throw std::invalid_argument("G2-A4: target and draft vocab sizes must be identical "
                        "(the target pool ids are used verbatim in the draft gathers): " +
                        std::to_string(n_vocab_tgt) + " != " + std::to_string(n_vocab_dft));
            }

            // Resolve the final shape before validating it. In particular, clamp block_rows to the
            // trained block size FIRST and derive aux_rows from that final value, otherwise a
            // clamped block can pass validation with stale derived dimensions.
            const int32_t block_rows = std::min(this->params.block_rows, n_draft_max);
            if (block_rows != this->params.block_rows) {
                LOG_WRN("%s: A4 block_rows=%d exceeds the trained block size %d -- clamping to %d\n",
                        __func__, this->params.block_rows, n_draft_max, block_rows);
            }

            // The raw shape was already validated during argument parsing / need_n_rs_seq; resolve
            // it again against the *clamped* trained block size. Both paths use the same helper, so
            // proposal/head/aux and the 2*head_k test cannot drift apart.
            common_a4_shape shape;
            try {
                shape = common_a4_resolve_shape(this->params, block_rows, this->params.n_max);
            } catch (const std::invalid_argument & err) {
                throw std::invalid_argument(std::string("G2-A4 invalid shape after block clamp: ") + err.what());
            }
            const int32_t proposal_k = shape.proposal_k;
            const int32_t head_k     = shape.head_k;
            const int32_t aux_rows   = shape.aux_rows;
            const int32_t pool_size  = this->params.pool_size;

            auto invalid = [&](const char * why) {
                throw std::invalid_argument(std::string("G2-A4 invalid shape (") +
                        "block_rows=" + std::to_string(block_rows) +
                        ", proposal_k=" + std::to_string(proposal_k) +
                        ", head_k=" + std::to_string(head_k) +
                        ", aux_rows=" + std::to_string(aux_rows) + "): " + why);
            };
            if (aux_rows > LLAMA_MAX_AUX_ROWS) {
                invalid("aux_rows exceeds LLAMA_MAX_AUX_ROWS");
            }
            if (pool_size > n_embd_dec) {
                invalid("pool larger than n_embd cannot be published");
            }
            if (pool_size > n_vocab_tgt) {
                invalid("pool larger than the vocabulary (pool ids would be out of range)");
            }

            a4_block_rows = block_rows;
            a4_proposal_k = proposal_k;
            a4_head_k     = head_k;
            a4_aux_rows   = aux_rows;
            a4_pool_size  = pool_size;

            // n_max is the legacy "candidate count"; keep it aligned with proposal_k
            this->params.n_max = a4_proposal_k;
            this->params.n_min = std::min(this->params.n_min, a4_proposal_k);

            a4_armed = true;

            // The real A4 arm builds the pool from the target aux head logits (T10). Until that
            // channel exists a DEV ONLY fallback can build the pool from the draft's own
            // full-vocab rows; it must be requested explicitly (flag or env) and is logged.
            static const bool dev_pool_env = [] {
                const char * value = std::getenv("LLAMA_DSPARK_A4_DEV_POOL");
                return value && *value && *value != '0';
            }();
            a4_pool_src_draft = this->params.pool_src == COMMON_SPECULATIVE_POOL_SRC_DRAFT ||
                (this->params.pool_src == COMMON_SPECULATIVE_POOL_SRC_AUTO && dev_pool_env);
            if (a4_pool_src_draft) {
                // the A4 full arm now runs the full-vocab head only on the head_k candidate rows, so the
                // lookahead rows (head_k+1..) have no draft logits any more -> the dev-only source cannot
                // be built; the target aux head (T10) is the real path
                a4_pool_src_draft = false;
                LOG_WRN("%s: G2-A4: the DEV ONLY draft-logits pool source is no longer available "
                        "(the full head only covers head_k rows) - using the target aux channel\n", __func__);
            }

            a4.resize(n_seq);
            for (auto & s : a4) {
                s.pool_first .assign(a4_pool_size, 0);
                s.pool_shared.assign(a4_pool_size, 0);
                s.aux_hidden.assign((size_t) a4_aux_rows * n_embd_dec, 0.0f);
            }
            a4_n_vocab = (int32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            a4_htp = common_speculative_model_is_htp(model_tgt) || common_speculative_model_is_htp(model_dft);
            {
                const char * v = std::getenv("LLAMA_A4_ALLOW_HTP_POOL");
                if (v != nullptr && *v != '\0' && std::strcmp(v, "1") != 0 && !a4_warned_htp_env) {
                    a4_warned_htp_env = true;
                    LOG_WRN("%s: LLAMA_A4_ALLOW_HTP_POOL='%s' is not exactly '1' - HTP pool stays "
                            "disabled\n", __func__, v);
                }
            }
            a4_idx_buf   .assign(a4_n_vocab, 0);
            a4_merged_buf.assign(a4_n_vocab, 0.0f);

            {
                const char * stats_env = std::getenv("LLAMA_SPEC_A4_STATS");
                a4_stats = stats_env && *stats_env && *stats_env != '0';

                const char * timing_env = std::getenv("LLAMA_SPEC_A4_TIMING");
                a4_timing = timing_env && *timing_env && *timing_env != '0';
            }
            a4_hist_a_k    .assign(a4_proposal_k + 1, 0);
            a4_hist_n_draft.assign(a4_proposal_k + 1, 0);
            a4_pend_mode    .assign(n_seq, 0);
            a4_pend_ndraft  .assign(n_seq, -1);
            a4_pend_verified.assign(n_seq, -1);
            if (a4_stats) {
                LOG_INF("%s: G2-A4 per-cycle stats enabled (LLAMA_SPEC_A4_STATS)\n", __func__);
            }

            LOG_INF("%s: G2-A4 draft-pool arm: n_rs_seq=%u (LLAMA_A4_RS_SEQ override = %s)\n", __func__,
                    llama_n_rs_seq(ctx_tgt), std::getenv("LLAMA_A4_RS_SEQ") ? std::getenv("LLAMA_A4_RS_SEQ") : "-");
            LOG_INF("%s: G2-A4 draft-pool arm: block_rows=%d proposal_k=%d head_k=%d aux_rows=%d pool_size=%d mode=%s src=%s\n",
                    __func__, a4_block_rows, a4_proposal_k, a4_head_k, a4_aux_rows, a4_pool_size,
                    this->params.pool_mode == COMMON_SPECULATIVE_POOL_MODE_POOL ? "pool" :
                    this->params.pool_mode == COMMON_SPECULATIVE_POOL_MODE_FULL ? "full" : "auto",
                    this->params.pool_src  == COMMON_SPECULATIVE_POOL_SRC_TARGET ? "target" :
                    this->params.pool_src  == COMMON_SPECULATIVE_POOL_SRC_DRAFT  ? "draft(DEV ONLY)" : "auto");

            llama_dflash_a4_arm(ctx_dft, a4_pool_size, a4_head_k, a4_aux_rows, a4_block_rows);
            a4_ctx_armed = true;

            // reserve the aux rows on the target context (T10)
            llama_set_aux_head(ctx_tgt, (uint32_t) a4_aux_rows);

            // T20 (device top-k over the aux rows) is a measured **NO-GO** on the phone: the HTP
            // TOP_K whole-row select/sort moves the cost into the target decode
            // (dev-host per-pair +1.4/+4.7/+2.2/+5.4/+6.7/+9.9 ms => +5.05 +- 3.25 ms/cycle,
            //  target_us +4.88, pool_us -1.54, total +3.45) - it does NOT pay for the 3 MB D2H it
            // removes. Default is therefore the host top-k path; the device channel stays behind an
            // explicit opt-in for reproduction only.
            static const bool aux_topk_dev = [] {
                const char * v = std::getenv("LLAMA_AUX_TOPK_DEV");
                return v && *v && *v != '0';
            }();
            static const bool aux_topk_host = [] {   // legacy override: force the host path
                const char * v = std::getenv("LLAMA_AUX_TOPK_HOST");
                return v && *v && *v != '0';
            }();
            // the device channel is only defined for 3 aux rows and k <= 4096 - anything else must
            // explicitly fall back to the host F32 channel instead of hitting the setter assert
            const bool dev_topk_supported = a4_aux_rows == 3 && a4_pool_size <= 4096;
            const bool use_dev_topk = aux_topk_dev && !aux_topk_host && dev_topk_supported;
            if (aux_topk_dev && !aux_topk_host && !dev_topk_supported) {
                LOG_WRN("%s: G2-A4: LLAMA_AUX_TOPK_DEV ignored (aux_rows=%d, pool_size=%d; the "
                        "device channel needs aux_rows == 3 and k <= 4096) - using host top-k\n",
                        __func__, a4_aux_rows, a4_pool_size);
            }
            if (use_dev_topk) {
                llama_set_aux_top_k(ctx_tgt, (uint32_t) a4_pool_size);
            }
            LOG_INF("%s: G2-A4 pool source: %s (top_k channel = %u)\n", __func__,
                    use_dev_topk ? "device top-k ids+vals [T20 NO-GO, opt-in]" : "host top-k over F32 aux logits (default)",
                    llama_n_aux_top_k(ctx_tgt));

            if (a4_htp) {
                LOG_WRN("%s: G2-A4: HTP backend detected - the pool graph has an unexplained output "
                        "divergence there, so pool cycles fall back to the full-vocab head "
                        "(set LLAMA_A4_ALLOW_HTP_POOL=1 only for experiments)\n", __func__);
            }
        }

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 1;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // A4 publishes aux hidden / z through the pre-norm channel for *all* block rows, but the copy
        // length of the masked mode is n_outputs (only the head_k rows request logits) -> use the
        // unmasked mode for A4 so every block row shows up in the channel.
        llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ !a4_armed);
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention

        a4_guard.committed = true;
    }

    // Undo everything the A4 arm changed on the borrowed target/draft contexts. Called both from
    // the destructor and from the constructor's failure path, so a failed arm cannot leave the
    // target with the aux LM-head shape or the device top-k channel configured.
    void a4_restore_contexts() {
        if (!a4_ctx_armed) {
            return;
        }
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        if (ctx_tgt != nullptr) {
            // top_k first: it may still hold host buffers tied to the aux row count
            llama_set_aux_top_k(ctx_tgt, 0);
            llama_set_aux_head (ctx_tgt, 0);
        }
        if (ctx_dft != nullptr) {
            llama_dflash_a4_disarm(ctx_dft);
            // restore the legacy masked pre-norm channel (A4 had switched it to unmasked)
            llama_set_embeddings_pre_norm(ctx_dft, true, true);
        }
        a4_ctx_armed = false;
    }

    ~common_speculative_impl_draft_dflash() override {
        // emit the summary of the trailing (partial) 8-cycle batch, otherwise a short run that
        // stops between two summary points would report nothing
        if (a4_armed && a4_stats) {
            const int64_t n_cycles = a4_n_pool_cycles + a4_n_full_cycles;
            if (n_cycles > 0 && n_cycles % 8 != 0) {
                a4_emit_summary(n_cycles);
            }
        }
        a4_restore_contexts();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // Reset all per-request A4 state even for a BOS-only prompt (prompt.size() == 0): the
        // guard/resync/verify-row state must never leak into the next request.
        if (a4_armed) {
            a4_guard_force_full = false;
            a4_force_full_cycle = false;
            a4_guard_unsafe     = false;
            a4_have_verify      = false;
            a4_verify_rows.clear();
            a4_last_nospec_cycle = -100;
            a4_pend_mode    [seq_id] = 0;
            a4_pend_ndraft  [seq_id] = -1;
            a4_pend_verified[seq_id] = -1;
            // discard any pending aux upload / readback from the previous request
            llama_aux_clear_seq(params.ctx_tgt, seq_id);

            // a new generation invalidates the pool and the acceptance history of this sequence
            auto & st = a4[seq_id];
            st.last_accepted  = -1;
            st.have_pool      = false;
            st.pool_from_target = false;
            llama_dflash_a4_set_pool(params.ctx_dft, seq_id,
                    st.pool_first.data(), st.pool_shared.data(), a4_pool_size, -1, false);
            // begin() only resets the pool contents; the preceding prompt/feature decodes are not
            // the A4 draft block, so do not leave the per-decode marker armed.
            llama_dflash_a4_clear_decode(params.ctx_dft);
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        static const bool media_features_enabled = [] {
            const char * value = std::getenv("LLAMA_DSPARK_MEDIA_FEATURES");
            return value && *value && *value != '0';
        }();
        // A vision model with mmproj prefills even text prompts as one embedding
        // batch ("route A"). DFlash must process those too, otherwise its KV is
        // never populated during prefill (ctx_dft pos_max == -1) and drafting
        // degrades / llama_decode fails. The injection logic below only uses
        // media_features to pick the M-RoPE position rows, so it is valid for
        // both DFlash and DSpark.
        const bool media_features = media_features_enabled &&
            batch_in.token == nullptr && batch_in.embd != nullptr;
        if (!media_features && (batch_in.token == nullptr || batch_in.embd != nullptr)) {
            return true;
        }

        // Remember which rows of the target verify batch actually requested logits. The next
        // draft() may only query those indices; llama_get_logits_ith() aborts on a valid-looking
        // but non-output index in Debug builds.
        if (a4_armed) {
            a4_record_verify_batch(batch_in);
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather this chunk's target features, interleaved by extract layer
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                // fuse extracted features through DFlash encoder
                // M-RoPE drafts read 4 position rows per token from embd batches, so pass them explicitly
                std::vector<llama_pos> enc_pos;
                if (is_mrope) {
                    enc_pos.resize((size_t) 4 * n_chunk);
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const int32_t row = i_batch_beg[seq_id] + offset + i;
                        const llama_pos p = batch_in.pos[row];
                        // Embedding media batches carry section-major M-RoPE
                        // coordinates. Text token batches carry one scalar row.
                        for (int32_t axis = 0; axis < 4; ++axis) {
                            enc_pos[axis * n_chunk + i] = media_features
                                ? batch_in.pos[axis * n_tokens + row]
                                : (axis == 3 ? 0 : p);
                        }
                    }
                }

                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ is_mrope ? enc_pos.data() : nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_pre_norm(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                // inject the DFlash decoder K/V cache at the tokens' target positions
                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const int32_t row = i_batch_beg[seq_id] + offset + i;
                    const llama_pos p = batch_in.pos[row];
                    batch_inject.pos[i] = p;
                    if (is_mrope) {
                        for (int32_t axis = 1; axis < 4; ++axis) {
                            batch_inject.pos[axis * n_chunk + i] = media_features
                                ? batch_in.pos[axis * n_tokens + row]
                                : (axis == 3 ? 0 : p);
                        }
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
                if (media_features) {
                    LOG_INF("DSpark: injected %d media target-feature rows for seq %d (%d position rows)\n",
                            n_chunk, seq_id, is_mrope ? 4 : 1);
                }
            }
        }

        return true;
    }

    // -----------------------------------------------------------------------
    // G2-A4 helpers
    //
    // pool = top-`pool_size` vocab ids. The id order is the contract with the draft
    // graph: the same vector is used for the lm_head row gather, the W2 row gather and
    // the host-side index -> token mapping, so all three are in the same order.
    // -----------------------------------------------------------------------
    // Top-k vocab ids of one logits row.
    // note: a single sequential pass + a size-k min-heap is ~10-20x faster than
    // std::partial_sort over n_vocab index entries (the indirect v[] lookups made it the host
    // hotspot: ~248k random accesses per pool, twice per cycle).
    void a4_top_k(const float * v, int32_t k, std::vector<int32_t> & out) {
        GGML_ASSERT(k <= a4_n_vocab);
        out.clear();
        out.reserve(k);

        // Full order: higher value first, then smaller token id. The heap must use the same
        // order as the final sort, otherwise a tie at the k-th boundary keeps an arbitrary id
        // (e.g. [0,0,1], k=2 used to return [2,1] instead of [2,0]).
        auto better = [v](int32_t a, int32_t b) {
            return v[a] > v[b] || (v[a] == v[b] && a < b);
        };
        // min-heap w.r.t. `better`: out.front() is the worst kept id (lowest value, then
        // largest id), so replacing it keeps the best k under the same full order.
        for (int32_t i = 0; i < a4_n_vocab; ++i) {
            if ((int32_t) out.size() < k) {
                out.push_back(i);
                std::push_heap(out.begin(), out.end(), better);
            } else if (better(i, out.front())) {
                std::pop_heap(out.begin(), out.end(), better);
                out.back() = i;
                std::push_heap(out.begin(), out.end(), better);
            }
        }

        std::sort(out.begin(), out.end(), better);
    }

    // T20: rebuild the pools from the device top-k channel.
    //   ids  = [row0 top-k | row1 top-k | row2 top-k]   (3k)
    //   vals = [values of (ids1||ids2) on row1 | on row2] (4k)
    // The shared pool is the exact top_k(max(row1,row2)): that set is always contained in the union
    // of the two per-row top-k lists, and every union member has both values -> exact merge.
    void a4_build_pool_from_topk(const int32_t * ids, const float * vals, a4_seq_state & st) {
        const int32_t k = a4_pool_size;

        // first-candidate pool = aux row 0 top-k (already sorted by the device)
        std::copy(ids, ids + k, st.pool_first.begin());

        struct cand { float v; int32_t id; };
        std::vector<cand> cs;
        cs.reserve(2*k);
        for (int32_t j = 0; j < 2*k; ++j) {
            const float v1 = vals[j];
            const float v2 = vals[2*k + j];
            cs.push_back({ v1 > v2 ? v1 : v2, ids[k + j] });
        }
        std::sort(cs.begin(), cs.end(), [](const cand & a, const cand & b) {
            return a.v > b.v || (a.v == b.v && a.id < b.id);
        });

        // dedupe (keep the highest-scoring copy), then pad by repeating the last id (gathering the
        // same row twice is harmless for the draft graph)
        std::unordered_set<int32_t> seen;
        seen.reserve(2*k);
        int32_t n = 0;
        for (const auto & c : cs) {
            if (n >= k) {
                break;
            }
            if (seen.insert(c.id).second) {
                st.pool_shared[n++] = c.id;
            }
        }
        GGML_ASSERT(n > 0);
        for (int32_t j = n; j < k; ++j) {
            st.pool_shared[j] = st.pool_shared[n - 1];
        }

        st.have_pool = true;
    }

    // rows[0] -> first-candidate pool, per-token max of rows[1..] -> shared 2nd/3rd pool
    void a4_build_pool(const std::vector<const float *> & rows, a4_seq_state & st) {
        GGML_ASSERT(!rows.empty() && (int32_t) rows.size() == a4_aux_rows);

        a4_top_k(rows[0], a4_pool_size, st.pool_first);

        if (rows.size() == 1) {
            st.pool_shared = st.pool_first;
        } else {
            GGML_ASSERT((int32_t) a4_merged_buf.size() == a4_n_vocab);
            std::copy(rows[1], rows[1] + a4_n_vocab, a4_merged_buf.begin());
            for (size_t r = 2; r < rows.size(); ++r) {
                for (int32_t i = 0; i < a4_n_vocab; ++i) {
                    a4_merged_buf[i] = std::max(a4_merged_buf[i], rows[r][i]);
                }
            }
            a4_top_k(a4_merged_buf.data(), a4_pool_size, st.pool_shared);
        }

        st.have_pool = true;
    }

    // refresh the pool (if a source is available) and decide the draft head mode of this cycle
    void a4_prepare(llama_seq_id seq_id, llama_token id_last, llama_context * ctx_dft, bool & pool_mode) {
        auto & st = a4[seq_id];
        int64_t t_pool_us = 0;
        a4_last_pool_us = 0;

        // "fully accepted" = the target accepted every token we proposed last cycle. The proposal
        // count can be < proposal_k when the caller truncated the draft (context/budget limit).
        // A real aux pool is only valid after the previous verify actually accepted every one of
        // the proposal_k rows. A caller-truncated "full" (last_n_draft < proposal_k) or a partial
        // acceptance is not row-aligned with the lookahead rows.
        const bool a4_pool_aligned   = a4_head_k == a4_proposal_k;
        const bool fully_accepted_last = st.last_accepted >= a4_proposal_k &&
            st.last_n_draft >= a4_proposal_k;

        // Decide FIRST whether this cycle can use a pool, and only then pay for building it.
        // Previously the aux-logits fetch + two full-vocab top-k ran unconditionally, i.e. a full
        // cycle (rejection or forced-full) wasted ~3 MB of DMA and 2 x top-256 for a pool it never used.
        // task-22 experiment: LLAMA_A4_FORCE_POOL_CYCLE=K forces the pool graph on exactly one cycle
        // (A4STAT 1-based index). By default the forced cycle builds the REAL pool (aux-derived ids) so
        // the accept pattern stays comparable to the baseline; set LLAMA_A4_POOL_STATIC=1 to use fixed
        // ids instead (which makes every proposal garbage -> a_k=0 -> different chunking).
        // accepts a single cycle or a comma separated list, e.g. "41,42,43" (multi-point injection)
        static const std::set<int64_t> force_pool_cycles = [] {
            std::set<int64_t> out;
            const char * v = std::getenv("LLAMA_A4_FORCE_POOL_CYCLE");
            if (v == nullptr || *v == 0) {
                return out;
            }
            const char * p = v;
            while (*p) {
                char * end = nullptr;
                const long k = std::strtol(p, &end, 10);
                if (end == p) {
                    break;
                }
                if (k > 0) {
                    out.insert((int64_t) k);
                }
                p = (*end == ',') ? end + 1 : end;
            }
            return out;
        }();
        const int64_t cycle_1based = a4_n_pool_cycles + a4_n_full_cycles + 1;
        const bool force_this = force_pool_cycles.count(cycle_1based) != 0;

        bool want_pool = false;
        switch (this->params.pool_mode) {
            case COMMON_SPECULATIVE_POOL_MODE_FULL: want_pool = false; break;
            case COMMON_SPECULATIVE_POOL_MODE_POOL:
            default:                                want_pool = fully_accepted_last && a4_pool_aligned; break;
        }

        if (a4_force_full_cycle) {
            want_pool = false; // one-shot guard request from the previous draft()
        }
        if (force_this) {
            want_pool = true; // request only: the protections below can still veto it
        }

        // Pool admission is decided here, after the force request. Two conditions are mandatory:
        //   * the backend is not HTP unless the experiment explicitly allows the divergent graph
        //   * the packed layout needs 2*head_k <= block_rows (a full-only shape can be smaller)
        // Force must not bypass either one; otherwise it can run the known-divergent graph or
        // trip the packing assertion in dflash.cpp.
        const bool pool_shape_ok = a4_head_k <= a4_block_rows/2;
        const bool pool_htp_ok   = !a4_htp || a4_allow_htp_pool();
        if (want_pool && (!a4_pool_aligned || !fully_accepted_last)) {
            if (force_this) {
                LOG_WRN("%s: A4FORCE skipped: pool needs head_k == proposal_k (%d/%d) and a fully "
                        "accepted previous verify (accepted=%d n_draft=%d proposal_k=%d) - staying "
                        "on the full-vocab head\n", __func__, a4_head_k, a4_proposal_k,
                        (int) st.last_accepted, (int) st.last_n_draft, a4_proposal_k);
            }
            want_pool = false;
        }
        if (want_pool && !pool_shape_ok) {
            if (force_this) {
                LOG_WRN("%s: A4FORCE skipped: pool layout needs 2*head_k <= block_rows "
                        "(block_rows=%d head_k=%d) - staying on the full-vocab head\n",
                        __func__, a4_block_rows, a4_head_k);
            }
            want_pool = false;
        }
        if (want_pool && !pool_htp_ok) {
            if (!a4_warned_htp_pool) {
                a4_warned_htp_pool = true;
                LOG_WRN("%s: G2-A4: HTP pool graph disabled (known divergence) - falling back to "
                        "the full-vocab head; set LLAMA_A4_ALLOW_HTP_POOL=1 for experiments\n", __func__);
            }
            if (force_this) {
                LOG_WRN("%s: A4FORCE skipped: HTP pool graph is not allowed without "
                        "LLAMA_A4_ALLOW_HTP_POOL=1 - staying on the full-vocab head\n", __func__);
            }
            want_pool = false;
        }
        if (force_this && want_pool) {
            LOG_INF("%s: A4FORCE: pool graph forced on cycle=%d committed=%" PRId64 " (one shot)\n",
                    __func__, (int) cycle_1based, a4_committed_tokens);
        }

        // real A4 arm: the pool comes from the target aux head logits of the previous verify
        if (want_pool && !st.have_pool && !a4_pool_src_draft) {
            const int64_t t0 = ggml_time_us();

            // T20: preferred path - the target graph already computed per-row top_k on the device
            const uint32_t kdev = llama_n_aux_top_k(params.ctx_tgt);
            const int32_t * ids  = kdev == (uint32_t) a4_pool_size ? llama_aux_top_k_ids (params.ctx_tgt, seq_id) : nullptr;
            const float   * vals = kdev == (uint32_t) a4_pool_size ? llama_aux_top_k_vals(params.ctx_tgt, seq_id) : nullptr;

            if (ids != nullptr && vals != nullptr) {
                a4_build_pool_from_topk(ids, vals, st);
                st.pool_from_target = true;
                t_pool_us = ggml_time_us() - t0;
                a4_last_pool_us = t_pool_us;
            } else {
            const float * aux_logits = llama_get_aux_logits(params.ctx_tgt, seq_id); // [aux_rows][n_vocab]
            if (aux_logits != nullptr) {
                std::vector<const float *> rows(a4_aux_rows);
                for (int32_t r = 0; r < a4_aux_rows; ++r) {
                    rows[r] = aux_logits + (size_t) r * a4_n_vocab;
                }
                a4_build_pool(rows, st);
                st.pool_from_target = true;
                t_pool_us = ggml_time_us() - t0;
                a4_last_pool_us = t_pool_us;
            } else {
                if (st.last_accepted >= 0 && !a4_warned_aux_logits) {
                    a4_warned_aux_logits = true;
                    LOG_WRN("%s: G2-A4: target aux logits unavailable (neither the top-k channel nor "
                            "llama_get_aux_logits) -> staying on the full-vocab draft head\n", __func__);
                }
            }
            }
        }

        // task-22 experiment: decouple the pool CONTENT from the pool GRAPH.
        // K512/K256/K32 produce bit-identical diverging output on the phone, i.e. the proposals do not
        // matter but "a pool cycle happened" does. Static ids answer: is it the aux->pool data path,
        // or merely the presence of the pool graph in the draft decode?
        static const bool pool_static = [] {
            const char * v = std::getenv("LLAMA_A4_POOL_STATIC");
            return v && *v && *v != '0';
        }();
        if (want_pool && pool_static) {
            for (int32_t i = 0; i < a4_pool_size; ++i) {
                st.pool_first [i] = (llama_token) ((i*7919 +   11) % a4_n_vocab);
                st.pool_shared[i] = (llama_token) ((i*7919 + 4099) % a4_n_vocab);
            }
            st.have_pool = true;
        }

        pool_mode = want_pool && st.have_pool;

        // hand the pool to the draft graph (process-level slot: prototype, --parallel 1)
        llama_dflash_a4_set_pool(ctx_dft, seq_id,
                st.pool_first.data(), st.pool_shared.data(), a4_pool_size,
                (int32_t) id_last, pool_mode);

        if (pool_mode) {
            a4_n_pool_cycles++;
            if (st.pool_from_target) {
                a4_n_pool_target++;
            }
            // consumed: a fresh cycle has to rebuild it (the target aux path rebuilds it above)
            st.have_pool = false;
        } else {
            a4_n_full_cycles++;
        }

        const int64_t n_cycles = a4_n_pool_cycles + a4_n_full_cycles;
        if (n_cycles % 8 == 0) {
            LOG_INF("%s: G2-A4 cycles=%" PRId64 " pool=%" PRId64 " (from_target=%" PRId64 ") full=%" PRId64 "\n",
                    __func__, n_cycles, a4_n_pool_cycles, a4_n_pool_target, a4_n_full_cycles);
        }
    }

    // task-22: hash of the TARGET sequence state (attention KV + recurrent/GDN snapshots) once per
    // cycle, so two arms can be compared AT THE SAME COMMITTED OFFSET. If the hashes differ, the
    // target state really diverged (and the first differing offset localises the cycle); if they are
    // equal while the logits differ, the divergence is in the weights/buffers instead.
    //   LLAMA_A4_STATE_HASH=1  ->  A4STATE cycle=.. committed=.. bytes=.. hash=..
    static bool a4_state_hash_on() {
        static const bool on = [] {
            const char * v = std::getenv("LLAMA_A4_STATE_HASH");
            return v && *v && *v != '0';
        }();
        return on;
    }

    void a4_state_hash(llama_seq_id seq_id, int64_t cycle, int64_t committed) {
        const size_t sz = llama_state_seq_get_size(params.ctx_tgt, seq_id);
        if (sz == 0) {
            LOG_INF("A4STATE seq=%d cycle=%" PRId64 " committed=%" PRId64 " bytes=0\n",
                    (int) seq_id, cycle, committed);
            return;
        }
        a4_state_buf.resize(sz);
        const size_t got = llama_state_seq_get_data(params.ctx_tgt, a4_state_buf.data(), sz, seq_id);
        uint64_t h = 1469598103934665603ull; // FNV-1a
        for (size_t i = 0; i < got; ++i) {
            h ^= (uint8_t) a4_state_buf[i];
            h *= 1099511628211ull;
        }
        LOG_INF("A4STATE seq=%d cycle=%" PRId64 " committed=%" PRId64 " bytes=%zu hash=%016" PRIx64 "\n",
                (int) seq_id, cycle, committed, got, h);
    }

    std::vector<uint8_t> a4_state_buf;
    int64_t a4_committed_tokens = 0; // committed tokens seen through accept() (A4STATE alignment)

    // Record the output rows of a successful target verify batch. Only a batch that follows a
    // draft() (a4_pend_ndraft[0] >= 0) is a verify; prompt/prefill batches must not become the
    // margin/anchor source. Rows without a logits flag are deliberately excluded.
    void a4_record_verify_batch(const llama_batch & b) {
        a4_have_verify = false;
        a4_verify_rows.clear();
        if (!a4_armed || n_seq != 1 || a4_pend_verified.empty() || a4_pend_verified[0] <= 0) {
            return;
        }
        if (b.n_tokens <= 0 || b.n_tokens > a4_proposal_k + 1 || b.logits == nullptr) {
            return;
        }
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            if (b.logits[i]) {
                a4_verify_rows.push_back(i);
            }
        }
        a4_have_verify = !a4_verify_rows.empty();
    }

    // Only rows recorded from the last successful target verify are safe to query: in a Debug
    // build llama_get_logits_ith() aborts on an index that did not request logits.
    const float * a4_verify_logits(int32_t df_arg) const {
        if (!a4_have_verify || df_arg < 0 || df_arg >= (int32_t) a4_verify_rows.size()) {
            return nullptr;
        }
        return llama_get_logits_ith(params.ctx_tgt, a4_verify_rows[df_arg]);
    }

    // H1/H2 diagnostic (task-22): dump the target's verify-row margins of the LAST decode.
    // Rationale: with the merged (verify+aux) head the verify rows are computed by a different M
    // matmul than the baseline (M=7 vs M=4), which can flip an argmax on a near tie. If the A4 / A4-off
    // arms produce different committed tokens, this dump shows whether the ids differ (H1: row
    // mapping) or only the margins are tiny (H2: numeric near tie).
    //   LLAMA_SPEC_MARGIN_DBG=1  -> one line per cycle, one entry per verify row
    void a4_margin_dbg(llama_seq_id seq_id) {
        const int n_vocab = a4_n_vocab;
        std::string line;
        char buf[160];
        for (size_t k = 0; k < a4_verify_rows.size(); ++k) {
            const int32_t row_idx = a4_verify_rows[k];
            const float * lg = a4_verify_logits((int32_t) k);
            if (lg == nullptr) {
                line += " row" + std::to_string(row_idx) + "=null";
                continue;
            }
            int32_t b1 = 0, b2 = -1;
            for (int32_t v = 1; v < n_vocab; ++v) {
                if (lg[v] > lg[b1]) { b2 = b1; b1 = v; }
                else if (b2 < 0 || lg[v] > lg[b2]) { b2 = v; }
            }
            snprintf(buf, sizeof(buf), " row%d=%d(%.6f)/%d(%.6f)/gap=%.2e",
                    (int) row_idx, b1, lg[b1], b2, b2 < 0 ? 0.0f : lg[b2],
                    b2 < 0 ? 0.0f : (double) (lg[b1] - lg[b2]));
            line += buf;
        }
        LOG_INF("SPECMARGIN seq=%d rows=%zu %s\n", (int) seq_id, a4_verify_rows.size(), line.c_str());
    }


    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // Consume the one-shot guard request at the start of the draft it applies to. The guard
        // below may set a new pending request for the *next* cycle; it must not leak further.
        a4_force_full_cycle = a4_guard_force_full;
        a4_guard_force_full = false;

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);
        // G2-A4: block index inside the single ubatch + draft head mode of this cycle
        std::vector<int32_t> a4_block_idx(n_seq, -1);
        std::vector<uint8_t> a4_pool_mode(n_seq, 0);
        int32_t n_blocks_added = 0;

        // A4TIME: the time between the end of the previous draft() and now is everything the caller
        // does in between: the target verify decode + host sampling + batch build. The impl does not
        // own the target decode call, so this gap is the closest host-side "target_us".

        static const bool margin_dbg = [] {
            const char * v = std::getenv("LLAMA_SPEC_MARGIN_DBG");
            return v && *v && *v != '0';
        }();
        static const bool state_hash = [] {
            const char * v = std::getenv("LLAMA_A4_STATE_HASH");
            return v && *v && *v != '0';
        }();
        if (a4_armed) {
            if (margin_dbg && a4_have_verify) {
                a4_margin_dbg(0);
            }
            if (state_hash) {
                a4_state_hash(0, a4_n_pool_cycles + a4_n_full_cycles, a4_committed_tokens);
            }
        }
        // roadmap (2): margin guard. The verify logits of the last target decode are already on the host,
        // so top1/top2 can be scanned here. min gap < tau  =>  emit no proposals (1-row commit) and
        // force the next cycle to full.
        a4_guard_unsafe = false;
        // LLAMA_A4_GUARD_LOG=1 is a pure probe: log the per-cycle min gap WITHOUT changing behaviour
        // (that gives the gap distribution and the gap at the exact cycle where an unguarded run
        // diverges -> that is what determines the required tau).
        if (a4_armed && (a4_guard_tau() > 0.0f || a4_guard_log()) && a4_have_verify) {
            const int n_vocab = a4_n_vocab;
            float gap_min = std::numeric_limits<float>::infinity();
            int32_t row_min = -1;
            for (size_t k = 0; k < a4_verify_rows.size(); ++k) {
                const int32_t row_idx = a4_verify_rows[k];
                const float * lg = a4_verify_logits((int32_t) k);
                if (lg == nullptr) {
                    continue;
                }
                int32_t b1 = 0, b2 = -1;
                for (int32_t v = 1; v < n_vocab; ++v) {
                    if (lg[v] > lg[b1]) { b2 = b1; b1 = v; }
                    else if (b2 < 0 || lg[v] > lg[b2]) { b2 = v; }
                }
                const float gap = b2 < 0 ? std::numeric_limits<float>::infinity() : lg[b1] - lg[b2];
                if (gap < gap_min) {
                    gap_min = gap;
                    row_min = row_idx;
                }
            }
            if (a4_guard_log()) {
                LOG_INF("%s: A4GAP cycle=%" PRId64 " committed=%" PRId64 " row=%d min_gap=%.6e tau=%.3f\n",
                        __func__, a4_n_pool_cycles + a4_n_full_cycles + 1, a4_committed_tokens,
                        (int) row_min, (double) gap_min, (double) a4_guard_tau());
            }
            if (a4_guard_tau() > 0.0f && gap_min < a4_guard_tau()) {
                a4_guard_unsafe     = true;
                a4_guard_force_full = true;
                LOG_INF("%s: A4GUARD: min gap %.3e < tau %.3f -> no proposals this cycle, full next\n",
                        __func__, (double) gap_min, (double) a4_guard_tau());
            }
        }

        const int64_t t_enter_us = a4_timing ? ggml_time_us() : 0;
        const int64_t t_target_us = (a4_timing && a4_last_draft_end_us > 0 && t_enter_us > a4_last_draft_end_us)
            ? (t_enter_us - a4_last_draft_end_us) : 0;

        // G2-A4 pre-flight: the batch is allocated with the draft context's n_batch, and the A4
        // block must also land in one ubatch. Check BOTH capacities before common_batch_add()
        // writes the block, otherwise a block larger than the batch overruns the arrays and the
        // later post-fill check is too late.
        if (a4_armed) {
            int32_t n_block_tokens_total = 0;
            for (const auto & dp : dparams) {
                if (dp.drafting) {
                    n_block_tokens_total += a4_block_rows;
                }
            }
            const int32_t n_batch_dft  = (int32_t) llama_n_batch(ctx_dft);
            const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
            if (n_block_tokens_total > n_batch_dft || n_block_tokens_total > n_ubatch_dft) {
                LOG_WRN("%s: G2-A4 needs the whole %d-row block in the batch (n_batch=%d) and in one "
                        "ubatch (n_ubatch=%d) -- skipping the draft\n",
                        __func__, n_block_tokens_total, n_batch_dft, n_ubatch_dft);
                for (auto & dp : dparams) {
                    if (dp.drafting) {
                        dp.result->clear();
                    }
                }
                return;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            int32_t n = (int32_t) dp.n_past;
            static const bool use_target_positions = [] {
                const char * value = std::getenv("LLAMA_DSPARK_TARGET_POSITIONS");
                return value && *value && *value != '0';
            }();
            if (is_dspark && use_target_positions) {
                // Multimodal token count differs from the target's logical
                // M-RoPE position. Draft queries must follow the accepted prefix.
                const llama_pos last = llama_memory_seq_pos_max(llama_get_memory(params.ctx_tgt), seq_id);
                if (last >= 0) {
                    n = last + 1;
                    if (n != dp.n_past) {
                        LOG_DBG("DSpark: target draft position seq=%d pos=%d token_count=%d\n",
                                seq_id, n, (int) dp.n_past);
                    }
                }
            }

            // G2-A4: pick the draft head mode + refresh the candidate pool of this cycle
            if (a4_armed) {
                bool pool_mode = false;
                a4_prepare(seq_id, dp.id_last, ctx_dft, pool_mode);
                a4_pool_mode[seq_id] = pool_mode ? 1 : 0;
            }

            const int32_t n_draft = params.n_max;

            if (a4_state_hash_on()) {
                LOG_INF("A4CYCLE seq=%d committed=%" PRId64 " n_past=%d anchor=%d n_draft_max=%d\n",
                        (int) seq_id, a4_committed_tokens, (int) dp.n_past, (int) dp.id_last, (int) dp.n_max);
            }

            // G2-A4: the block is [anchor, MASK x (block_rows-1)]; only the first proposal_k
            // rows are read back as candidates, the trailing rows are lookahead rows.
            const int32_t n_block_tokens = a4_armed ? a4_block_rows : n_draft + (is_dspark ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            a4_block_idx[seq_id] = n_blocks_added++;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                // G2-A4: the draft head only produces head_k logits rows (the lookahead rows never go
                // through it), so only those rows may request outputs - n_outputs drives the logits
                // copy in llama_context::decode(), which must not exceed the tensor size.
                const bool want_logits = !a4_armed || i < a4_head_k;
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, want_logits);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        // note: trunk (7 rows) + pool/full head + chain all live in this one decode - a per-op split
        //       needs cparams.cb_eval (a context-creation parameter, not reachable from common/).
        //       T17's 4-arm zero-code decomposition provides the trunk/head/chain split; A4TIME gives
        //       the surrounding host stages.
        // note: there is deliberately no process-global backend knob here (the old
        //       LLAMA_A4_POOL_NOFUSE wrote the non-atomic ggml-hexagon opt_opfusion global, which
        //       could race with other contexts building/reading it).

        const int64_t t_draft0 = a4_timing ? ggml_time_us() : 0;
        int ret = llama_decode(ctx_dft, batch);
        if (a4_armed) {
            // consume the per-decode A4 marker even when the decode failed, so a later ordinary
            // decode on this context cannot inherit the A4 graph shape
            llama_dflash_a4_clear_decode(ctx_dft);
        }

        const int64_t t_draft_us = a4_timing ? ggml_time_us() - t_draft0 : 0;
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (a4_armed) {
                // -----------------------------------------------------------
                // G2-A4: publish the lookahead hidden rows to the target aux head and
                // read back the proposals.
                //
                // The draft graph publishes, per block, through the masked pre-norm channel:
                //   slot 0 .. aux_rows-1              -> post-final-norm hidden of block rows
                //                                        head_k+1 .. block_rows-1
                //   slot head_k .. head_k+aux_rows-1  -> z[0..pool_size) padded to n_embd (pool arm)
                // and the proposals are either argmax(z) mapped through the pool ids (pool arm)
                // or the sampled chained logits of block rows 0..proposal_k-1 (full arm).
                // -----------------------------------------------------------
                auto & st = a4[seq_id];

                const int32_t b    = a4_block_idx[seq_id];
                const size_t  row0 = (size_t) b * a4_block_rows; // prototype: one block per ubatch

                const float * chan = llama_get_embeddings_pre_norm(ctx_dft);
                if (chan == nullptr || !llama_pre_norm_valid(ctx_dft)) {
                    // A non-NONE pooling mode (or a graph that did not emit t_h_pre_norm) skips the
                    // readback; the buffer address alone cannot prove this cycle wrote fresh rows.
                    LOG_WRN("%s: A4 pre-norm channel was not written this decode (pooling=%d) - "
                            "dropping this cycle instead of uploading stale/zero aux hidden\n",
                            __func__, (int) llama_pooling_type(ctx_dft));
                    result.clear();
                    continue;
                }

                // aux hidden = block rows head_k+1 .. block_rows-1, packed at slot 0..aux_rows-1
                std::memcpy(st.aux_hidden.data(),
                        chan + row0 * n_embd_dec,
                        (size_t) a4_aux_rows * n_embd_dec * sizeof(float));

                // hand them to the target context (T10)
                const int64_t t_aux0 = a4_timing ? ggml_time_us() : 0;
                llama_set_aux_hidden(params.ctx_tgt, seq_id, st.aux_hidden.data());
                const int64_t t_aux_us = a4_timing ? ggml_time_us() - t_aux0 : 0;

                int64_t t_full_us = 0;
                if (a4_pool_mode[seq_id] != 0) {
                    // pool arm: candidate i = pool_i[argmax(z_i)], z published at slot head_k+i
                    // (the graph already ran the chain over the pool, so the argmax here matches it)
                    for (int32_t i = 0; i < a4_proposal_k; ++i) {
                        const float * z = chan + (row0 + a4_head_k + i) * n_embd_dec;
                        const std::vector<int32_t> & pool = i == 0 ? st.pool_first : st.pool_shared;

                        int32_t best = 0;
                        for (int32_t j = 1; j < a4_pool_size; ++j) {
                            // ggml_argmax keeps the first maximum. The graph uses that same pool
                            // index to gather the next W1 row, so the host must not re-tie-break
                            // by vocab id here; otherwise the emitted token and the Markov
                            // condition can diverge.
                            if (z[j] > z[best]) {
                                best = j;
                            }
                        }
                        result.push_back((llama_token) pool[best]);
                    }
                    if (LOG_LEVEL_DEBUG <= 0) {} // no-op guard
                    if (a4_stats || std::getenv("LLAMA_SPEC_MARGIN_DBG")) {
                        static const bool pool_dbg = [] {
                            const char * v = std::getenv("LLAMA_SPEC_MARGIN_DBG");
                            return v && *v && *v != '0';
                        }();
                        if (pool_dbg) {
                            // log exactly the candidates produced by this cycle; proposal_k can be
                            // smaller than 3 (e.g. --spec-proposal-k 1), so indexing the last
                            // three entries would underflow the result vector.
                            std::string cands;
                            for (int32_t i = 0; i < a4_proposal_k; ++i) {
                                if (i > 0) {
                                    cands += ' ';
                                }
                                cands += std::to_string((int) result[result.size() - a4_proposal_k + i]);
                            }
                            LOG_INF("A4POOL seq=%d n=%d cands=[%s]\n", (int) seq_id,
                                    (int) result.size(), cands.c_str());
                        }
                    }
                } else {
                    const int64_t t_full0 = a4_timing ? ggml_time_us() : 0;
                    // full arm: read the chained logits of the proposal rows only
                    for (int32_t i = 0; i < a4_proposal_k; ++i) {
                        common_sampler_sample(smpl, ctx_dft, beg + i, true);

                        const auto * cur_p = common_sampler_get_candidates(smpl, true);
                        const llama_token id = cur_p->data[0].id;

                        common_sampler_accept(smpl, id, true);
                        result.push_back(id);
                    }

                    // note: the former DEV ONLY draft-logits pool source is gone - the A4 full head
                    // only produces head_k logits rows, so the lookahead rows are only reachable
                    // through the target aux head
                    t_full_us = a4_timing ? ggml_time_us() - t_full0 : 0;
                }

                // task-22 grouping control: LLAMA_A4_FORCE_REJECT_CYCLE=N keeps the pool graph and the
                // REAL pool ids, but replaces the proposals with deliberately wrong tokens so the caller
                // rejects all of them (a_k=0, one token committed). Isolates "all-reject grouping" from
                // "pool graph execution" (the latter is clean per the B43r control).
                static const std::set<int64_t> force_reject_cycles = [] {
                    std::set<int64_t> out;
                    const char * v = std::getenv("LLAMA_A4_FORCE_REJECT_CYCLE");
                    if (v == nullptr || *v == 0) {
                        return out;
                    }
                    const char * p = v;
                    while (*p) {
                        char * end = nullptr;
                        const long k = std::strtol(p, &end, 10);
                        if (end == p) {
                            break;
                        }
                        if (k > 0) {
                            out.insert((int64_t) k);
                        }
                        p = (*end == ',') ? end + 1 : end;
                    }
                    return out;
                }();
                // a4_prepare() already counted THIS cycle, so the current index is `counters` (no +1).
                // (the FORCE knob matches before that increment, where +1 is correct -> hence the two
                //  different expressions; always cross-check with the A4CYCLE committed value.)
                const int64_t cyc_now = a4_n_pool_cycles + a4_n_full_cycles;

                // roadmap (1a)/(2): drop all proposals for this cycle -> the caller decodes a single row
                if (a4_armed && !result.empty()) {
                    // note: require that the previous cycle actually PROPOSED something - otherwise a
                    // 0-proposal cycle (n_draft == 0) would look like "all rejected" and the resync
                    // would stick forever, turning the whole run into non-speculative decoding.
                    const bool resync = a4_resync_after_reject() &&
                        a4[seq_id].last_n_draft > 0 && a4[seq_id].last_accepted == 0 &&
                        cyc_now != a4_last_nospec_cycle + 1; // never two resync cycles in a row
                    if (resync || a4_guard_unsafe) {
                        LOG_INF("%s: A4NOSPEC: cycle=%" PRId64 " (%s) -> 0 proposals (last_accepted=%d)\n",
                                __func__, cyc_now, resync ? "resync after reject" : "margin guard",
                                (int) a4[seq_id].last_accepted);
                        result.clear();
                        // the target will decode only the committed row; do not leave this
                        // cycle's aux upload pending for that non-verify decode
                        llama_set_aux_hidden(params.ctx_tgt, seq_id, nullptr);
                        a4_last_nospec_cycle = cyc_now;
                    }
                }
                if (a4_armed && force_reject_cycles.count(cyc_now) != 0 && !result.empty()) {
                    for (size_t i = 0; i < result.size(); ++i) {
                        // ids 7,8,9,... : valid tokens, effectively never the target argmax here
                        result[i] = (llama_token) (7 + (int) (i % 8));
                    }
                    LOG_INF("%s: A4REJECT: proposals overridden on cycle=%d committed=%" PRId64 " (forcing a_k=0)\n",
                            __func__, (int) cyc_now, a4_committed_tokens);
                }

                if (a4_timing) {
                    // A4TIME: host/backend stages of one cycle (us).
                    //   draft_us  = the whole draft decode (7-row trunk + pool/full head + chain)
                    //   target_us = caller gap since the previous draft() end, i.e. target verify
                    //               decode + host sampling + batch build (the impl does not call the
                    //               target decode itself, so this is the host-side proxy)
                    const bool pool_now = a4_pool_mode[seq_id] != 0;
                    if (pool_now) {
                        a4tm_n_pool++; a4tm_draft_pool += t_draft_us; a4tm_target_pool += t_target_us;
                        a4tm_aux_pool += t_aux_us; a4tm_pool_us_sum += a4_last_pool_us;
                    } else {
                        a4tm_n_full++; a4tm_draft_full += t_draft_us; a4tm_target_full += t_target_us;
                        a4tm_aux_full += t_aux_us; a4tm_full_us_sum += t_full_us;
                    }
                    const int64_t t_total = t_target_us + t_draft_us + t_aux_us + a4_last_pool_us + t_full_us;
                    LOG_INF("A4TIME seq=%d cycle=%" PRId64 " mode=%s target_us=%" PRId64 " draft_us=%" PRId64
                            " aux_us=%" PRId64 " pool_us=%" PRId64 " full_us=%" PRId64 " total_us=%" PRId64
                            " | mean_pool target=%" PRId64 " draft=%" PRId64 " aux=%" PRId64 " pool=%" PRId64
                            " | mean_full target=%" PRId64 " draft=%" PRId64 " aux=%" PRId64 " full=%" PRId64 "\n",
                            (int) seq_id, a4_n_pool_cycles + a4_n_full_cycles, pool_now ? "pool" : "full",
                            t_target_us, t_draft_us, t_aux_us, a4_last_pool_us, t_full_us, t_total,
                            a4tm_n_pool ? a4tm_target_pool/a4tm_n_pool : 0, a4tm_n_pool ? a4tm_draft_pool/a4tm_n_pool : 0,
                            a4tm_n_pool ? a4tm_aux_pool/a4tm_n_pool : 0, a4tm_n_pool ? a4tm_pool_us_sum/a4tm_n_pool : 0,
                            a4tm_n_full ? a4tm_target_full/a4tm_n_full : 0, a4tm_n_full ? a4tm_draft_full/a4tm_n_full : 0,
                            a4tm_n_full ? a4tm_aux_full/a4tm_n_full : 0, a4tm_n_full ? a4tm_full_us_sum/a4tm_n_full : 0);
                }

                // T13 instrumentation: remember this cycle's mode + proposal count; accept() adds a_k
                a4_pend_mode    [seq_id] = a4_pool_mode[seq_id] ? 1 : 0;
                a4_pend_ndraft  [seq_id] = (int32_t) result.size();
                a4_pend_verified[seq_id] = (int32_t) result.size();
                if (result.empty()) {
                    llama_set_aux_hidden(params.ctx_tgt, seq_id, nullptr);
                    a4_pend_ndraft[seq_id] = -1;
                    a4_pend_verified[seq_id] = -1;
                }
                st.last_n_draft = dp.n_max > 0 ? std::min((int32_t) result.size(), (int32_t) dp.n_max)
                                               : (int32_t) result.size();
                if (a4_timing) {
                    a4_last_draft_end_us = ggml_time_us();
                }
            } else if (is_dspark) {
                // DSpark predicts the next token from position 0 and optionally truncates
                // at the first position below the confidence threshold.
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_pre_norm(ctx_dft) : nullptr;

                for (int32_t i = 0; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }
        }
    }

    void set_verified(llama_seq_id seq_id, uint16_t n_verified) override {
        if (!a4_armed || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (n_verified == 0) {
            llama_set_aux_hidden(params.ctx_tgt, seq_id, nullptr);
            a4_pend_ndraft[seq_id] = -1;
            a4_pend_verified[seq_id] = -1;
            a4[seq_id].last_n_draft = 0;
            return;
        }
        a4_pend_verified[seq_id] = (int32_t) n_verified;
        // pool admission must see the number actually sent to the verifier, not the untruncated
        // draft() result
        a4[seq_id].last_n_draft = (int32_t) n_verified;
    }

    void draft_empty(llama_seq_id seq_id) override {
        if (!a4_armed || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        a4_n_empty_cycles++;
        llama_set_aux_hidden(params.ctx_tgt, seq_id, nullptr);
        a4_pend_ndraft[seq_id] = -1;
        a4_pend_verified[seq_id] = -1;
        if (a4_stats) {
            const int64_t cycle = a4_n_pool_cycles + a4_n_full_cycles;
            LOG_INF("A4STAT seq=%d cycle=%" PRId64 " mode=none n_draft=0 a_k=0 committed=0 "
                    "trunc=0 empty=1 src=none\n", (int) seq_id, cycle);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (!a4_armed || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        // remember how many drafts the target accepted: a full acceptance unlocks the pool arm
        // on the next cycle (PLAN §3 "pool 仅当上一轮全接受")
        a4[seq_id].last_accepted = (int32_t) n_accepted;
        a4_committed_tokens += (int64_t) n_accepted + 1;

        if (!a4_stats) {
            a4_pend_verified[seq_id] = -1;
            a4_pend_ndraft[seq_id] = -1;
            return;
        }
        const int32_t n_verified = a4_pend_verified[seq_id] >= 0
            ? a4_pend_verified[seq_id] : a4_pend_ndraft[seq_id];
        if (n_verified <= 0) {
            a4_pend_verified[seq_id] = -1;
            return; // empty cycles are reported by draft_empty(), not accept()
        }
        const int32_t n_draft = std::min(n_verified, a4_proposal_k);
        const int32_t a_k   = std::min<int32_t>((int32_t) n_accepted, n_draft);
        const int32_t mode  = a4_pend_mode[seq_id];
        const bool    trunc = n_verified < a4_proposal_k;

        if (trunc) {
            a4_n_trunc_cycles++;
        } else {
            a4_n_valid_cycles++;
            a4_sum_a_valid += a_k;
        }
        a4_hist_a_k    [a_k]++;
        a4_hist_n_draft[n_draft]++;

        const int64_t cycle = a4_n_pool_cycles + a4_n_full_cycles;
        // frozen semantics: the `src` field describes *this* cycle's pool, so a full cycle is "none"
        // (previously the stale pool_from_target flag leaked "target" into full rows - T13 caught it)
        const char * src = !mode ? "none"
                          : (a4[seq_id].pool_from_target ? "target" : (a4_pool_src_draft ? "draft" : "none"));
        LOG_INF("A4STAT seq=%d cycle=%" PRId64 " mode=%s n_draft=%d a_k=%d committed=%d trunc=%d src=%s\n",
                (int) seq_id, cycle, mode ? "pool" : "full", n_draft, a_k, a_k + 1, trunc ? 1 : 0, src);

        a4_pend_verified[seq_id] = -1;
        a4_pend_ndraft  [seq_id] = -1;

        // every 8 cycles: short runs (n=64 -> ~20-30 cycles) must also produce at least one summary
        if (cycle > 0 && cycle % 8 == 0) {
            a4_emit_summary(cycle);
        }
    }

    // cumulative A4STAT snapshot (also emitted from the destructor for the trailing partial batch)
    void a4_emit_summary(int64_t cycle) {
        std::string a_k_hist;
        std::string n_draft_hist;
        for (int32_t i = 0; i <= a4_proposal_k; ++i) {
            if (a4_hist_a_k[i] > 0) {
                a_k_hist += (a_k_hist.empty() ? "" : ",") + std::to_string(i) + ":" + std::to_string(a4_hist_a_k[i]);
            }
            if (a4_hist_n_draft[i] > 0) {
                n_draft_hist += (n_draft_hist.empty() ? "" : ",") + std::to_string(i) + ":" + std::to_string(a4_hist_n_draft[i]);
            }
        }
        // TPC numerator and denominator use the same non-truncated, verified cycles; empty
        // (zero-proposal) cycles are reported separately and never enter either side.
        LOG_INF("A4STAT summary cycles=%" PRId64 " pool=%" PRId64 " full=%" PRId64 " target=%" PRId64
                " trunc=%" PRId64 " valid=%" PRId64 " empty=%" PRId64
                " a_k_hist=%s n_draft_hist=%s tpc=%.4f\n",
                cycle, a4_n_pool_cycles, a4_n_full_cycles, a4_n_pool_target, a4_n_trunc_cycles,
                a4_n_valid_cycles, a4_n_empty_cycles,
                a_k_hist.empty() ? "-" : a_k_hist.c_str(),
                n_draft_hist.empty() ? "-" : n_draft_hist.c_str(),
                a4_n_valid_cycles > 0 ? 1.0 + (double) a4_sum_a_valid / (double) a4_n_valid_cycles : 0.0);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd(llama_get_model(ctx_dft));

        LOG_INF("%s: adding speculative implementation 'draft-mtp'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_pre_norm(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ true);

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        common_batch_clear(batch);

        for (int k = 0; k < n_tokens; ++k) {
            common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
        }

        // shift the tgt embeddings to the right by one position
        // assumes that the tokens in the batch are sequential for each sequence
        // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
        //                                                       ^--- this is a problem
        // TODO:this is generally true, but would be nice to assert it
        {
            const float * h_tgt = llama_get_embeddings_pre_norm(ctx_tgt);
            std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));

            //{
            //    // string with seq_ids in the batch
            //    std::stringstream ss;
            //    for (int i = 0; i < n_tokens; ++i) {
            //        ss << batch_in.seq_id[i][0] << ",";
            //    }
            //    LOG_WRN("%s: batch_in.seq_id = %s\n", __func__, ss.str().c_str());
            //}
        }

        // fill the pending embeddings from a previous run
        auto set_h = [&](int idx, const float * h_row) {
            std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
        }

        const int32_t rc = llama_decode(ctx_dft, batch);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
            return false;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_pre_norm_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                h_row = llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_pre_norm() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-simple'\n", __func__);
        LOG_INF("%s: - size_n=%d, size_m=%d, min_hits=%d\n", __func__,
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(this->type).c_str());
        LOG_INF("%s: - size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", __func__,
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: adding speculative implementation 'ngram-mod'\n", __func__);
        LOG_INF("%s: - n_match=%d, n_max=%d, n_min=%d\n", __func__,
                this->params.n_match, this->params.n_max, this->params.n_min);
        LOG_INF("%s: - mod size=%zu (%.3f MB)\n", __func__,
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) - resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-cache'\n", __func__);
        LOG_INF("%s: - n_draft=%d, cache_static=%s, cache_dynamic=%s\n", __func__,
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

void common_speculative_validate_a4_models(const common_params_speculative & params,
        const struct llama_model * model_tgt, const struct llama_model * model_dft) {
    if (!params.a4_selected()) {
        return;
    }
    if (model_tgt == nullptr || model_dft == nullptr) {
        throw std::invalid_argument("G2-A4: target and draft models must be loaded before validation");
    }
    if (!llama_model_dflash_a4_supported(model_dft)) {
        throw std::invalid_argument("G2-A4: the draft model did not load DSpark Markov weights "
                "(a plain DFlash GGUF cannot run the pool chain)");
    }
    char arch_tgt[64] = {};
    if (llama_model_meta_val_str(model_tgt, "general.architecture", arch_tgt, sizeof(arch_tgt)) < 0 ||
        std::strcmp(arch_tgt, "qwen35moe") != 0) {
        throw std::invalid_argument(std::string("G2-A4: the aux head is only implemented for the "
                "qwen35moe target graph (target arch = ") + arch_tgt + ")");
    }
    const int32_t n_embd_tgt = llama_model_n_embd(model_tgt);
    const int32_t n_embd_dft = llama_model_n_embd(model_dft);
    if (n_embd_tgt != n_embd_dft) {
        throw std::invalid_argument("G2-A4: aux hidden needs draft n_embd == target n_embd (" +
                std::to_string(n_embd_dft) + " != " + std::to_string(n_embd_tgt) + ")");
    }
    const int32_t n_vocab_tgt = (int32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    const int32_t n_vocab_dft = (int32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
    if (n_vocab_tgt != n_vocab_dft) {
        throw std::invalid_argument("G2-A4: target and draft vocab sizes must be identical "
                "(the target pool ids are used verbatim in the draft gathers): " +
                std::to_string(n_vocab_tgt) + " != " + std::to_string(n_vocab_dft));
    }
    const int32_t * target_layer_ids = llama_model_target_layer_ids(model_dft);
    const uint32_t target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
    if (target_layer_ids == nullptr || target_layer_ids_n == 0) {
        throw std::invalid_argument("G2-A4: the draft model has no target_layer_ids metadata");
    }
    const int32_t n_layer_tgt = common_a4_target_layer_limit(model_tgt);
    for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
        const int32_t id = target_layer_ids[k];
        if (id < 0 || id >= n_layer_tgt) {
            throw std::invalid_argument("G2-A4: draft target_layer_ids[" + std::to_string(k) +
                    "]=" + std::to_string(id) + " is outside target layer range [0, " +
                    std::to_string(n_layer_tgt) + ") - the draft does not match this target");
        }
    }
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}
// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_model_path = !params.draft.mparams.path.empty();

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3)) && params.draft.ctx_dft != nullptr;
        bool has_draft_mtp    = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP))    && params.draft.ctx_dft != nullptr;
        bool has_draft_dflash = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) && params.draft.ctx_dft != nullptr;
        bool has_draft_dspark = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) && params.draft.ctx_dft != nullptr;



        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            if (!has_draft_model_path) {
                LOG_WRN("%s: draft model is not specified - cannot use 'draft' type\n", __func__);
                has_draft_simple = false;
            }
        } else if (has_draft_model_path && !has_draft_mtp && !has_draft_eagle3 &&
                   !has_draft_dflash && !has_draft_dspark) {
            LOG_WRN("%s: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it\n", __func__);
            has_draft_simple = true;
        }

        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_draft_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
        if (has_draft_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params));
        }
        if (has_draft_dspark) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_pre_norm(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_pre_norm()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                // report the post-wrapper count; callers may truncate further (budget) and call
                // common_speculative_set_verified() again before accept()
                impl->set_verified(seq_id, (uint16_t) result.size());

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            } else if (dp.drafting && result.empty()) {
                impl->draft_empty(seq_id);
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_set_verified(common_speculative * spec, llama_seq_id seq_id, uint16_t n_verified) {
    common_speculative_impl * impl = spec->impl_last[seq_id];
    if (impl != nullptr) {
        impl->set_verified(seq_id, n_verified);
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
