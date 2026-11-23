#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_moe_packed_shard {
    size_t stride = 0;
    std::vector<uint32_t> checksums;
    /// Sparse packs list the materialized experts explicitly; legacy/full shards
    /// leave this empty and are treated as fully materialized.
    std::vector<uint8_t> materialized;
};

// Incremental IEEE CRC32. Start with zero, then pass the return value for each span in file order.
uint32_t llama_moe_expert_crc32(uint32_t checksum, const void * data, size_t size);

struct llama_moe_experts_entry {
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
    bool compact_q3k = false;
    std::shared_ptr<const llama_moe_packed_shard> packed;
};

class llama_moe_experts {
public:
    explicit llama_moe_experts(const char * input_path);

    const llama_moe_experts_entry * find(const char * tensor_name) const;

    /// 每个 routed 专家 entry 是否都带 packed 校验索引（manifest 的 expert_checksums）。
    /// 共享 decode 的槽位校验依赖它，所以模型加载期就要能问这个问题。
    bool packed_index_complete() const;

    int32_t expert_count() const;
    int32_t expert_used_count() const;
    int32_t layer_count() const;
    size_t bytes_per_slot_all_layers() const;

    /// 逻辑 expert id -> 打包文件里的位置（position）映射。
    ///
    /// manifest 的 `expert_order.orders["<layer>"]` 是 **position -> logical**
    /// （`dpsk-pack-order-v1`，见 tools/moe/pack_hexagon.py）。这里在加载时转成
    /// logical -> position，运行时只做 `position * stride` 的 offset 计算。
    /// 未提供 `expert_order` 或 `LLAMA_MOE_EXPERT_ORDER=0` 时返回 nullptr（identity）。
    const std::vector<int32_t> * expert_order(int32_t layer) const;

    /// manifest 是否带合法 `expert_order` 且未被 `LLAMA_MOE_EXPERT_ORDER=0` 关掉。
    bool has_expert_order() const;

private:
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_layer = 0;
    size_t bytes_per_slot = 0;
    bool order_enabled = false;
    std::map<int32_t, std::vector<int32_t>> expert_order_; // layer -> (logical -> position)
    std::unordered_map<std::string, llama_moe_experts_entry> entries;
};
