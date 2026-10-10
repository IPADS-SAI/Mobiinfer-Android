#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

struct llama_moe_sidecar_entry {
    int32_t layer = -1;
    std::string tensor_name;
    std::string tensor_family;
    std::string path;
    ggml_type type = GGML_TYPE_COUNT;
    size_t offset = 0;
    size_t tensor_bytes = 0;
    size_t bytes_per_expert = 0;
    size_t expert_stride = 0;
    int32_t expert_count = 0;
};

class llama_moe_sidecar {
public:
    explicit llama_moe_sidecar(const char * input_path);

    const llama_moe_sidecar_entry * find(const char * tensor_name) const;

    int32_t expert_count() const;
    int32_t expert_used_count() const;
    int32_t layer_count() const;
    size_t bytes_per_slot_all_layers() const;

private:
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_layer = 0;
    size_t bytes_per_slot = 0;
    std::unordered_map<std::string, llama_moe_sidecar_entry> entries;
};
