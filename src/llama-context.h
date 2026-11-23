#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_moe_slot_runtime;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// Mutable per-decode state of the DFlash draft-pool (A4) arm. Owned by the draft context so that
// two contexts cannot observe or terminate each other's arm, and so that the graph builder never
// reads process-global mutable state.
struct llama_dflash_a4_state {
    bool    armed       = false;
    bool    decode_active = false; // set for the single draft decode that owns this A4 block
    bool    pool_mode   = false;
    int32_t n_pool      = 0;
    int32_t head_k      = 0;
    int32_t aux_rows    = 0;
    int32_t block_rows  = 0;
    int32_t anchor      = -1;
    llama_seq_id seq_id = -1;

    std::vector<int32_t> ids_first;  // [n_pool] vocab rows of the first-candidate pool
    std::vector<int32_t> ids_shared; // [n_pool] vocab rows of the shared 2nd/3rd pool
};

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_pre_norm();
    float * get_embeddings_pre_norm_ith(int32_t i);
    bool    pre_norm_valid() const { return embd_pre_norm_valid; }

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_pre_norm(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);

    // target aux LM-head rows (draft-pool A4, see llama-ext.h):
    // enable n_rows extra head rows per sequence (0 = disable), upload one sequence's rows and
    // read back the aux logits of the last decode. the aux rows never enter the batch.
    // the graph is rebuilt/reserved when n_rows changes.
    void set_aux_head(uint32_t n_rows);
    bool set_aux_hidden(llama_seq_id seq_id, const float * data);
    void clear_aux_seq(llama_seq_id seq_id);
    uint32_t n_aux_rows() const;
    float * get_aux_logits(llama_seq_id seq_id);

    // T20: device top-k over the aux rows (ids + shared-pool candidate values only)
    void            set_aux_top_k(uint32_t k);
    uint32_t        n_aux_top_k() const;
    const int32_t * get_aux_top_k_ids(llama_seq_id seq_id);
    const float   * get_aux_top_k_vals(llama_seq_id seq_id);

    // DFlash draft-pool A4 state, owned by this (draft) context (see src/models/dflash.cpp)
    llama_dflash_a4_state       & dflash_a4()       { return dflash_a4_state; }
    const llama_dflash_a4_state & dflash_a4() const { return dflash_a4_state; }

    void set_causal_attn(bool value);
    void set_warmup(bool value);
    void set_moe_lookahead_prefetch_depth(int32_t depth);
    int32_t get_moe_lookahead_prefetch_depth() const;
    void park_moe_expert_memory();
    void prewarm_moe_experts();
    void set_moe_prefill_final(bool value);
    // 0 restores hparams top-k. Clamped to [1, hparams.n_expert_used].
    void set_n_expert_used(int32_t n);
    int32_t n_expert_used() const;

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    // discard pending aux uploads/outputs after a failed/cancelled decode
    void aux_decode_abort();
    // async-copy the aux head logits (if any) of this ubatch from the backend into aux_head.logits
    void extract_aux_logits(const llm_graph_result * res, const llama_ubatch & ubatch);
    // T20: async-copy the per-row aux top-k ids and the shared-pool candidate values
    void extract_aux_top_k(const llm_graph_result * res, const llama_ubatch & ubatch);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    bool moe_prefill_final = false;

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype,
                          bool             reserve = false) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;
    std::unique_ptr<llama_moe_slot_runtime> moe_slots;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state before the final output norm (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_pre_norm is enabled and the model graph
    // sets llm_graph_result::t_h_pre_norm
    buffer_view<float> embd_pre_norm = {nullptr, 0};
    bool embd_pre_norm_valid = false; // this decode actually wrote the pre-norm channel

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    // target aux LM-head rows (draft-pool A4). Extra rows for the output head, uploaded from the
    // host and returned in a channel fully separate from logits/embd. The rows are not batch
    // tokens: no KV entry, no position, no output_ids entry.
    struct aux_head_info {
        uint32_t n_rows = 0; // rows per sequence; 0 = disabled

        // host-side input rows, F32 [n_seq_max][n_rows][n_embd] (seq-major)
        std::vector<float> hidden;

        // set for sequences with a hidden upload that has not been consumed by a decode yet.
        // An aux-logits readback is only valid for a sequence whose rows were uploaded for *this*
        // verify; prefill decodes (or a stale upload from an earlier request) must not publish a
        // pool source.
        std::vector<uint8_t> hidden_valid;

        // host-side aux logits of the last decode, F32 [n_seq_max][n_rows][n_vocab] (seq-major)
        buffer_view<float> logits = {nullptr, 0};

        // set for sequences whose aux logits were computed by the last decode
        std::vector<uint8_t> logits_valid;

        // T20: device top-k channel - ids [3*k] + shared-candidate values [4*k] per sequence
        uint32_t             top_k = 0;
        std::vector<int32_t> top_k_ids;   // [n_seq_max][3*k]
        std::vector<float>   top_k_vals;  // [n_seq_max][4*k]
        std::vector<uint8_t> top_k_valid; // [n_seq_max]
    };

    aux_head_info aux_head;

    // draft-pool A4 arm/graph state, owned by this context
    llama_dflash_a4_state dflash_a4_state;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
