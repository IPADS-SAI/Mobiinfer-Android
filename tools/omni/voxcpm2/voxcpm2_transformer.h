#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>

struct VoxCPM2GGUFWeightStore {
    ggml_backend_t        backend       = nullptr; // not owned
    ggml_context *        ctx_meta      = nullptr;
    ggml_context *        ctx_weights   = nullptr;
    gguf_context *        ctx_gguf      = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;

    std::unordered_map<std::string, ggml_tensor *> tensors;

    ~VoxCPM2GGUFWeightStore();

    VoxCPM2GGUFWeightStore() = default;
    VoxCPM2GGUFWeightStore(const VoxCPM2GGUFWeightStore &) = delete;
    VoxCPM2GGUFWeightStore & operator=(const VoxCPM2GGUFWeightStore &) = delete;
    VoxCPM2GGUFWeightStore(VoxCPM2GGUFWeightStore &&) = delete;
    VoxCPM2GGUFWeightStore & operator=(VoxCPM2GGUFWeightStore &&) = delete;

    bool load(const std::string & path, ggml_backend_t backend, const std::vector<std::string> & prefixes);
    void free();

    ggml_tensor * get(const std::string & name) const;
    bool get_u32(const char * key, int & dst) const;
    bool get_f32(const char * key, float & dst) const;
};

struct VoxCPM2TransformerConfig {
    int   hidden_size       = 1024;
    int   intermediate_size = 4096;
    int   n_layer           = 12;
    int   n_heads           = 16;
    int   n_kv_heads        = 2;
    int   head_dim          = 64;
    int   max_length        = 4096;
    float rms_norm_eps      = 1.0e-5f;
    float rope_freq_base    = 10000.0f;
    int   rope_original_max = 32768;
    bool  no_rope           = false;
    bool  use_flash_attn    = false; // CUDA flash-attn is unsafe for VoxCPM2 short non-causal sequences.
    std::vector<float> rope_factors;  // Per-dimension-pair frequency factors for LongRoPE
};

struct VoxCPM2TransformerLayerWeights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq        = nullptr;
    ggml_tensor * wk        = nullptr;
    ggml_tensor * wv        = nullptr;
    ggml_tensor * wo        = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
    ggml_tensor * ffn_gate  = nullptr;
    ggml_tensor * ffn_up    = nullptr;
    ggml_tensor * ffn_down  = nullptr;
};

struct VoxCPM2TransformerWeights {
    ggml_tensor * norm         = nullptr;
    ggml_tensor * freq_factors = nullptr;   // [head_dim/2] for LongRoPE, can be null
    ggml_context * aux_ctx     = nullptr;   // owner ctx for freq_factors
    ggml_backend_buffer_t aux_buf = nullptr; // backend buffer for freq_factors
    std::vector<VoxCPM2TransformerLayerWeights> layers;

    ~VoxCPM2TransformerWeights() {
        if (aux_buf) { ggml_backend_buffer_free(aux_buf); aux_buf = nullptr; }
        if (aux_ctx) { ggml_free(aux_ctx); aux_ctx = nullptr; }
    }
};

int voxcpm2_infer_layer_count(const std::unordered_map<std::string, ggml_tensor *> & tensors,
                              const std::string & prefix);

bool voxcpm2_bind_transformer_weights(const std::unordered_map<std::string, ggml_tensor *> & tensors,
                                      const std::string & prefix,
                                      VoxCPM2TransformerConfig & cfg,
                                      VoxCPM2TransformerWeights & weights);

ggml_tensor * voxcpm2_add_bias(ggml_context * ctx, ggml_tensor * output, ggml_tensor * bias);

ggml_tensor * voxcpm2_linear(ggml_context * ctx,
                             ggml_tensor * weight,
                             ggml_tensor * bias,
                             ggml_tensor * input);

ggml_tensor * voxcpm2_build_positions(ggml_context * ctx, int n_tokens);
ggml_tensor * voxcpm2_build_cfg_pair_positions(ggml_context * ctx, int branch_len);
ggml_tensor * voxcpm2_build_cfg_pair_attention_mask(ggml_context * ctx, int branch_len);

// Get VoxCPM2 LongRoPE short/long factors (64 elements for head_dim=128)
const std::vector<float> & voxcpm2_get_rope_factors();

// Build a persistent freq_factors tensor from rope_factors vector.
// backend is used to allocate a buffer for the tensor (GPU or CPU).
// *out_ctx is set to the owning context; caller must ggml_free(*out_ctx) when done.
// *out_buf is set to the backend buffer; caller must ggml_backend_buffer_free(*out_buf) when done.
ggml_tensor * voxcpm2_build_freq_factors(const std::vector<float> & rope_factors,
                                         ggml_backend_t backend,
                                         ggml_context ** out_ctx,
                                         ggml_backend_buffer_t * out_buf);

ggml_tensor * voxcpm2_transformer_forward(ggml_context * ctx,
                                          const VoxCPM2TransformerConfig & cfg,
                                          const VoxCPM2TransformerWeights & weights,
                                          ggml_tensor * input,
                                          ggml_tensor * positions = nullptr,
                                          ggml_tensor * attention_mask = nullptr);
