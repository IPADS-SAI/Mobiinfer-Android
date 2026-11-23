#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

//
// pre-norm embeddings (hidden state before the final output norm)
//

// Set whether the context outputs pre-norm embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_pre_norm    (struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);
LLAMA_API bool    llama_pre_norm_valid             (const struct llama_context * ctx);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// target aux LM-head rows (draft-pool A4)
//
// The target LM head can project a few extra hidden rows in a channel separate from the verify
// rows, e.g. the draft-pool A4 scheme where the draft publishes the post-norm hidden of its
// block rows 4,5,6. The verify rows always keep the exact baseline LM-head path; the aux rows are
// projected independently while the phone pool-arm divergence is investigated.
//
// These rows are NOT batch tokens: they never enter the batch/ubatch, the KV cache, the
// positions or output_ids, and they do not change n_tokens, llama_get_logits() or
// llama_get_logits_ith(). The aux logits come back in a separate host channel.
//
// Layouts (row-major, "ggml order", i.e. ne0 is contiguous):
//   aux hidden  F32 [n_rows][n_embd]   per sequence, rows r0..r_(n_rows-1) = draft block rows 4,5,6
//                                      (already post-norm on the draft side - the target does NOT
//                                      apply its output norm to them, only the aux head MUL_MAT)
//   aux logits  F32 [n_rows][n_vocab]  per sequence, same row order as the uploaded hidden
//
// The caller is expected to:
//   1) llama_set_aux_head(ctx_tgt, 3) once when the draft/aux channel is set up
//   2) after the draft decode, llama_set_aux_hidden(ctx_tgt, seq_id, rows) for every sequence
//   3) run the normal target verify decode (batch unchanged)
//   4) after the decode, llama_get_aux_logits(ctx_tgt, seq_id) to read the aux rows
//

// the maximum number of aux rows per sequence supported by this channel
#define LLAMA_MAX_AUX_ROWS 32

// Enable/disable the aux head rows. n_rows is per sequence, 0 disables the channel and restores
// the exact baseline graph. Changing n_rows rebuilds/reserves the compute graphs.
LLAMA_API void llama_set_aux_head(struct llama_context * ctx, uint32_t n_rows);

// Upload the aux hidden rows of one sequence: data is F32 [n_rows][n_embd], row-major.
// data == nullptr clears the sequence's rows. Returns false if the channel is disabled or the
// sequence id is invalid. Must be called after the draft decode and before the target decode.
LLAMA_API bool llama_set_aux_hidden(struct llama_context * ctx, llama_seq_id seq_id, const float * data);

// Number of aux rows per sequence currently configured (0 = disabled).
LLAMA_API uint32_t llama_n_aux_rows(const struct llama_context * ctx);

// Invalidate all aux channel state (pending upload, F32 logits, device top-k) for one sequence.
// Used at request boundaries so a new request cannot consume a previous request's hidden/logits.
LLAMA_API void llama_aux_clear_seq(struct llama_context * ctx, llama_seq_id seq_id);

// Aux head logits of the last decode for one sequence: F32 [n_rows][n_vocab], row-major.
// Returns nullptr if the channel is disabled or if this sequence did not get aux logits from the
// last decode. Like llama_get_logits(), the pointer is only valid for the most recent decode
// (this call synchronizes the context).
LLAMA_API float * llama_get_aux_logits(struct llama_context * ctx, llama_seq_id seq_id);

//
// aux head device top-k (G2-A4 draft-pool, T20)
//
// Goal: keep the [n_rows][n_vocab] F32 aux logits on the device. With k > 0 the target graph runs
// one top_k per aux row on the backend (per row on purpose: the HTP top_k layout solver takes the
// fast chunked path only for rows == 1) and also gathers the F32 values of the shared-pool
// candidate union (aux rows 1,2), so only ids + values are copied back:
//   ids  I32 [3*k]  : row0 top-k, then row1 top-k, then row2 top-k      (~6 KB for k=512)
//   vals F32 [4*k]  : values of (ids_row1 || ids_row2) on row1, then on row2 (~8 KB for k=512)
// The caller can then rebuild the exact `top_k(max(row1,row2))` shared pool on the host (the union
// of the two per-row top-k lists always contains it).
// With k > 0 the F32 `llama_get_aux_logits` channel is not filled any more (returns nullptr).
LLAMA_API void           llama_set_aux_top_k (struct llama_context * ctx, uint32_t k);
LLAMA_API uint32_t       llama_n_aux_top_k   (const struct llama_context * ctx);
LLAMA_API const int32_t * llama_aux_top_k_ids (struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API const float   * llama_aux_top_k_vals(struct llama_context * ctx, llama_seq_id seq_id);

//
// DFlash draft-pool A4 arm (G2-A4)
//
// These functions are called from common/speculative.cpp and defined by the dflash model graph.
// They live in this shared header (C++ linkage) so that both the CLI/server side and the model
// side use one LLAMA_API declaration; hidden-visibility shared builds need the export attribute.
//
// llama_dflash_a4_supported() reports whether the draft model actually loaded the DSpark Markov
// weights. Selecting the `draft-dspark` enum alone is not enough: a plain DFlash GGUF would pass
// the enum check and only fail later in the pool graph.
LLAMA_API bool llama_model_dflash_a4_supported(const struct llama_model * model);
LLAMA_API bool llama_dflash_a4_supported(const struct llama_context * ctx_dft);

// Arm/disarm the A4 draft graph topology on a draft context. Sizes are per decode block.
LLAMA_API void llama_dflash_a4_arm(struct llama_context * ctx_dft, int32_t n_pool, int32_t head_k,
                                   int32_t aux_rows, int32_t block_rows);
LLAMA_API void llama_dflash_a4_disarm(struct llama_context * ctx_dft);

// Per-decode pool contents (pool ids, anchor token, pool/full mode) for one drafting sequence.
// set_pool also marks the next decode as the A4 draft block; clear_decode must be called after
// that decode (success or failure) so ordinary decodes fall back to the legacy head.
LLAMA_API void llama_dflash_a4_set_pool(struct llama_context * ctx_dft, llama_seq_id seq_id,
                                        const int32_t * ids_first, const int32_t * ids_shared,
                                        int32_t n_pool, int32_t anchor, bool pool_mode);
LLAMA_API void llama_dflash_a4_clear_decode(struct llama_context * ctx_dft);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);
