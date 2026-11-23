#include "llama-moe-experts.h"

#include "llama-impl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#if defined(__linux__) && defined(__aarch64__) && !defined(__AARCH64EB__) && (defined(__GNUC__) || defined(__clang__))
#define LLAMA_MOE_ARM_CRC32
#include <arm_acle.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace {

constexpr uint32_t crc32_polynomial = 0xedb88320u;

#ifdef LLAMA_MOE_ARM_CRC32
// Multiply reflected IEEE CRC32 residues to shift a lane by a number of bytes.
uint32_t crc_multiply(uint32_t a, uint32_t b) {
    uint32_t product = 0;
    while (a) {
        if (a & 0x80000000u) {
            product ^= b;
        }
        a <<= 1;
        b = (b >> 1) ^ ((0u - (b & 1u)) & crc32_polynomial);
    }
    return product;
}

uint32_t crc_shift(size_t bytes) {
    uint32_t result = 0x80000000u;
    uint32_t power = 0x00800000u;
    while (bytes) {
        if (bytes & 1u) {
            result = crc_multiply(result, power);
        }
        bytes >>= 1;
        if (bytes) {
            power = crc_multiply(power, power);
        }
    }
    return result;
}

__attribute__((target("+crc")))
uint32_t crc32_arm(uint32_t crc, const uint8_t * data, size_t size) {
    const auto load_word = [](const uint8_t * ptr) {
        uint64_t word;
        std::memcpy(&word, ptr, sizeof(word));
        return word;
    };

    if (size >= 32 * 1024) {
        const size_t lane_size = size / (3 * sizeof(uint64_t)) * sizeof(uint64_t);
        uint32_t crc1 = 0;
        uint32_t crc2 = 0;
        for (size_t i = 0; i < lane_size; i += sizeof(uint64_t)) {
            crc  = __crc32d(crc,  load_word(data + i));
            crc1 = __crc32d(crc1, load_word(data + lane_size + i));
            crc2 = __crc32d(crc2, load_word(data + 2 * lane_size + i));
        }
        const uint32_t advance = crc_shift(lane_size);
        crc = crc_multiply(advance, crc) ^ crc1;
        crc = crc_multiply(advance, crc) ^ crc2;
        data += 3 * lane_size;
        size -= 3 * lane_size;
    }
    while (size >= sizeof(uint64_t)) {
        crc = __crc32d(crc, load_word(data));
        data += sizeof(uint64_t);
        size -= sizeof(uint64_t);
    }
    while (size--) {
        crc = __crc32b(crc, *data++);
    }
    return crc;
}
#endif

ggml_type parse_type(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
        return (char) std::tolower(value);
    });

    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = (ggml_type) value;
        const char * type_name = ggml_type_name(type);
        if (!type_name) {
            continue;
        }
        std::string candidate(type_name);
        std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char item) {
            return (char) std::tolower(item);
        });
        if (candidate == name) {
            return type;
        }
    }

    throw std::runtime_error(format("unknown MoE experts quant type '%s'", name.c_str()));
}

bool routed_family(const std::string & family) {
    return family == "ffn_gate_exps" ||
           family == "ffn_up_exps" ||
           family == "ffn_down_exps" ||
           family == "ffn_gate_up_exps";
}

/// Parse a JSON array that must be a permutation of [0, n_expert).
std::vector<int32_t> parse_bijection(const nlohmann::json & value, int32_t n_expert,
                                     const std::string & where) {
    if (!value.is_array() || value.size() != (size_t) n_expert) {
        throw std::runtime_error(format(
            "MoE expert order '%s' must be an array of length %d", where.c_str(), n_expert));
    }
    std::vector<int32_t> result((size_t) n_expert);
    std::vector<uint8_t> seen((size_t) n_expert, 0);
    for (size_t i = 0; i < result.size(); ++i) {
        if (!value[i].is_number_integer()) {
            throw std::runtime_error(format(
                "MoE expert order '%s' contains a non-integer at index %zu", where.c_str(), i));
        }
        const int64_t raw = value[i].get<int64_t>();
        if (raw < 0 || raw >= n_expert) {
            throw std::runtime_error(format(
                "MoE expert order '%s' has out-of-range value %lld at index %zu",
                where.c_str(), (long long) raw, i));
        }
        const int32_t id = (int32_t) raw;
        if (seen[(size_t) id]) {
            throw std::runtime_error(format(
                "MoE expert order '%s' repeats expert %d (not a bijection)", where.c_str(), id));
        }
        seen[(size_t) id] = 1;
        result[i] = id;
    }
    return result;
}

int32_t parse_layer_key(const std::string & key) {
    if (key.empty()) {
        throw std::runtime_error("MoE expert order has an empty layer key");
    }
    size_t used = 0;
    long value = 0;
    try {
        value = std::stol(key, &used);
    } catch (const std::exception &) {
        throw std::runtime_error(format("MoE expert order has invalid layer key '%s'", key.c_str()));
    }
    if (used != key.size() || value < 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(format("MoE expert order has invalid layer key '%s'", key.c_str()));
    }
    return (int32_t) value;
}

}

uint32_t llama_moe_expert_crc32(uint32_t checksum, const void * bytes, size_t size) {
    const auto * data = static_cast<const uint8_t *>(bytes);
#ifdef LLAMA_MOE_ARM_CRC32
    static const bool has_crc32 = getauxval(AT_HWCAP) & HWCAP_CRC32;
    if (has_crc32) {
        return ~crc32_arm(~checksum, data, size);
    }
#endif
    static constexpr auto table = [] {
        std::array<uint32_t, 256> table {};
        for (uint32_t i = 0; i < table.size(); ++i) {
            uint32_t crc = i;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc >> 1) ^ ((crc & 1) ? crc32_polynomial : 0);
            }
            table[i] = crc;
        }
        return table;
    }();

    uint32_t crc = ~checksum;
    for (size_t i = 0; i < size; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xff];
    }
    return ~crc;
}

llama_moe_experts::llama_moe_experts(const char * input_path) {
    namespace fs = std::filesystem;

    if (!input_path || input_path[0] == '\0') {
        throw std::invalid_argument("MoE experts path is empty");
    }

    const fs::path input(input_path);
    const fs::path manifest_path = fs::is_directory(input) ? input / "manifest.json" : input;
    const fs::path manifest_dir = manifest_path.parent_path();

    std::ifstream stream(manifest_path);
    if (!stream.is_open()) {
        throw std::runtime_error(format("failed to open MoE experts manifest '%s'", manifest_path.string().c_str()));
    }

    nlohmann::json manifest;
    stream >> manifest;

    if (manifest.value("schema_version", 0) != 1) {
        throw std::runtime_error("unsupported MoE experts schema version");
    }
    // 兼容旧 manifest：早期写的是 `sidecar_kind`。数据格式改名不该让已生成的
    // 13 GiB 产物作废，所以这里两个 key 都收。
    const std::string kind = manifest.contains("experts_kind")
        ? manifest.value("experts_kind", std::string())
        : manifest.value("sidecar_kind", std::string());
    if (kind != "omni_moe_gguf" && kind != "flashmoe_gguf") {
        throw std::runtime_error(format("unsupported MoE experts kind '%s'", kind.c_str()));
    }
    const std::string layout = manifest.value("layout", std::string());
    if (layout != "layer_major_expert" && layout != "expert_major") {
        throw std::runtime_error(format("MoE experts layout '%s' is not expert-major", layout.c_str()));
    }

    const auto backend_layout = manifest.value("backend_layout", std::string());
    const bool compact = backend_layout == "hexagon-tiled-q3k-compact-v1";
    const char * packed_dir = std::getenv("LLAMA_MOE_HTP_PACKED_DIR");
    if (manifest.contains("backend_layout")) {
        if (backend_layout != "hexagon-tiled-v1" && !compact) {
            throw std::runtime_error("unsupported backend expert layout");
        }
        if (!packed_dir || !*packed_dir || !fs::equivalent(fs::path(packed_dir), manifest_dir)) {
            throw std::runtime_error("tiled expert manifest requires LLAMA_MOE_HTP_PACKED_DIR pointing at its directory");
        }
    } else if (packed_dir && *packed_dir) {
        const fs::path packed_manifest = fs::path(packed_dir) / "manifest.json";
        if (fs::is_regular_file(packed_manifest) && !fs::equivalent(manifest_path, packed_manifest)) {
            std::ifstream packed_stream(packed_manifest);
            const auto packed = nlohmann::json::parse(packed_stream);
            const auto packed_layout = packed.value("backend_layout", std::string());
            if ((!packed_layout.empty() && packed_layout != "hexagon-tiled-v1") ||
                    packed.value("hexagon_conversion", nlohmann::json::object()).contains("expert_checksums")) {
                throw std::runtime_error("this Hexagon expert layout requires --moe-experts pointing at its packed manifest");
            }
        }
    }

    const auto & model = manifest.at("model");
    n_expert = model.at("expert_count").get<int32_t>();
    n_expert_used = model.at("expert_used_count").get<int32_t>();
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert) {
        throw std::runtime_error("invalid expert counts in MoE experts manifest");
    }

    std::unordered_map<std::string, std::shared_ptr<const llama_moe_packed_shard>> packed_shards;
    const auto conversion = manifest.value("hexagon_conversion", nlohmann::json::object());
    if (conversion.contains("expert_checksums")) {
        const auto & index = conversion.at("expert_checksums");
        if (backend_layout.empty() || !index.is_object() || index.empty()) {
            throw std::runtime_error("invalid Hexagon expert checksum index");
        }
        for (const auto & [name, shard] : index.items()) {
            const fs::path filename(name);
            auto packed = std::make_shared<llama_moe_packed_shard>();
            const auto & stride = shard.at("stride");
            const auto & checksums = shard.at("crc32");
            if (filename.has_root_path() || !filename.has_filename() || filename.filename() == "." ||
                std::find(filename.begin(), filename.end(), fs::path("..")) != filename.end() ||
                !stride.is_number_unsigned() || stride.get<uint64_t>() == 0 ||
                stride.get<uint64_t>() > std::numeric_limits<size_t>::max() / (size_t) n_expert ||
                !checksums.is_array() || checksums.size() != (size_t) n_expert) {
                throw std::runtime_error("invalid Hexagon packed expert shard: " + name);
            }
            packed->stride = stride.get<size_t>();
            packed->checksums.reserve(n_expert);
            for (const auto & value : checksums) {
                if (!value.is_number_unsigned() || value.get<uint64_t>() > UINT32_MAX) {
                    throw std::runtime_error("invalid Hexagon expert checksum: " + name);
                }
                packed->checksums.push_back(value.get<uint32_t>());
            }
            if (shard.contains("materialized_experts")) {
                const auto & materialized = shard.at("materialized_experts");
                if (!materialized.is_array()) {
                    throw std::runtime_error("invalid Hexagon materialized expert list: " + name);
                }
                packed->materialized.assign((size_t) n_expert, 0);
                for (const auto & value : materialized) {
                    if (!value.is_number_unsigned() || value.get<uint64_t>() >= (uint64_t) n_expert) {
                        throw std::runtime_error("invalid Hexagon materialized expert index: " + name);
                    }
                    packed->materialized[value.get<size_t>()] = 1;
                }
            }
            packed_shards.emplace(name, std::move(packed));
        }
    }
    if (compact && packed_shards.empty()) {
        throw std::runtime_error("compact Q3_K experts require a checksum index");
    }

    std::unordered_set<int32_t> layers;
    for (const auto & item : manifest.at("entries")) {
        const std::string family = item.at("tensor_family").get<std::string>();
        if (!routed_family(family)) {
            continue;
        }

        llama_moe_experts_entry entry;
        entry.layer = item.at("layer").get<int32_t>();
        entry.tensor_name = item.at("tensor_name").get<std::string>();
        entry.tensor_family = family;
        const auto filename = item.at("repacked_file").get<std::string>();
        entry.path = (manifest_dir / filename).string();
        entry.type = parse_type(item.at("quant_type").get<std::string>());
        entry.offset = item.at("repacked_offset").get<size_t>();
        entry.tensor_bytes = item.at("exact_byte_length").get<size_t>();
        entry.bytes_per_expert = item.at("bytes_per_expert").get<size_t>();
        entry.expert_stride = item.at("expert_stride").get<size_t>();
        entry.expert_count = item.value(
            "expert_count",
            entry.bytes_per_expert == 0 ? 0 : (int32_t) (entry.tensor_bytes / entry.bytes_per_expert));
        entry.compact_q3k = compact && entry.type == GGML_TYPE_Q3_K;
        if (!packed_shards.empty()) {
            const auto found = packed_shards.find(filename);
            if (found == packed_shards.end()) {
                throw std::runtime_error("missing Hexagon expert checksums: " + filename);
            }
            entry.packed = found->second;
        }

        if (entry.layer < 0 || entry.bytes_per_expert == 0 || entry.expert_count != n_expert) {
            throw std::runtime_error(format("invalid MoE experts entry '%s'", entry.tensor_name.c_str()));
        }
        if (entry.offset > std::numeric_limits<size_t>::max() - entry.bytes_per_expert) {
            throw std::runtime_error(format("MoE experts entry '%s' overflows family extent", entry.tensor_name.c_str()));
        }
        const size_t family_end = entry.offset + entry.bytes_per_expert;
        if (entry.expert_stride < family_end) {
            throw std::runtime_error(format("invalid expert stride for MoE experts entry '%s'", entry.tensor_name.c_str()));
        }
        if ((size_t) (entry.expert_count - 1) >
            (std::numeric_limits<size_t>::max() - family_end) / entry.expert_stride) {
            throw std::runtime_error(format("MoE experts entry '%s' overflows file extent", entry.tensor_name.c_str()));
        }

        const size_t required = (size_t) (entry.expert_count - 1) * entry.expert_stride +
                                family_end;
        std::error_code error;
        const size_t file_size = (size_t) fs::file_size(entry.path, error);
        if (error || (entry.packed ? file_size != entry.packed->stride * (size_t) n_expert : file_size < required)) {
            throw std::runtime_error(format(
                "MoE experts file '%s' is too small for tensor '%s'",
                entry.path.c_str(), entry.tensor_name.c_str()));
        }

        bytes_per_slot += entry.bytes_per_expert;
        layers.insert(entry.layer);
        if (!entries.emplace(entry.tensor_name, std::move(entry)).second) {
            throw std::runtime_error("duplicate tensor in MoE experts manifest");
        }
    }

    if (entries.empty()) {
        throw std::runtime_error("MoE experts manifest has no routed expert tensors");
    }
    n_layer = (int32_t) layers.size();

    // 可选：打包文件里的专家置换。T-0055 的 order.json 与 pack_hexagon.py 都采用
    // `orders[layer]` = position -> logical；这里在加载时转成 logical -> position，
    // 运行时只拿它算文件 offset，slot/LRU/router 仍全部使用逻辑 id。
    if (manifest.contains("expert_order")) {
        const auto & order_doc = manifest.at("expert_order");
        if (!order_doc.is_object()) {
            throw std::runtime_error("MoE expert_order must be an object");
        }
        if (order_doc.contains("format") &&
            order_doc.value("format", std::string()) != "dpsk-pack-order-v1") {
            throw std::runtime_error(format("unsupported MoE expert_order format '%s'",
                order_doc.value("format", std::string()).c_str()));
        }
        const std::string direction = order_doc.value("direction", std::string("position_to_logical"));
        bool position_to_logical = true;
        if (direction == "position_to_logical" || direction == "position->logical") {
            position_to_logical = true;
        } else if (direction == "logical_to_position" || direction == "logical->position") {
            position_to_logical = false;
        } else {
            throw std::runtime_error(format("unsupported MoE expert_order direction '%s'",
                direction.c_str()));
        }
        if (!order_doc.contains("orders") || !order_doc.at("orders").is_object()) {
            throw std::runtime_error("MoE expert_order.orders must be an object keyed by layer");
        }
        for (const auto & item : order_doc.at("orders").items()) {
            const int32_t layer = parse_layer_key(item.key());
            const std::string where = "orders[" + item.key() + "]";
            std::vector<int32_t> values = parse_bijection(item.value(), n_expert, where);
            if (position_to_logical) {
                // values 是 position -> logical；转成运行时要的 logical -> position。
                std::vector<int32_t> inverse((size_t) n_expert);
                for (int32_t position = 0; position < n_expert; ++position) {
                    inverse[(size_t) values[(size_t) position]] = position;
                }
                values.swap(inverse);
            }
            // direction=logical_to_position 时 values 已经是 logical -> position。
            if (!expert_order_.emplace(layer, std::move(values)).second) {
                throw std::runtime_error(format("MoE expert_order repeats layer %d", layer));
            }
        }
        for (const int32_t layer : layers) {
            if (expert_order_.find(layer) == expert_order_.end()) {
                throw std::runtime_error(format(
                    "MoE expert_order is missing layer %d", layer));
            }
        }
        if (expert_order_.size() != layers.size()) {
            throw std::runtime_error("MoE expert_order has layers with no routed expert tensors");
        }
        order_enabled = true;
        const char * env = std::getenv("LLAMA_MOE_EXPERT_ORDER");
        if (env && std::strcmp(env, "0") == 0) {
            order_enabled = false;
            LLAMA_LOG_INFO("%s: LLAMA_MOE_EXPERT_ORDER=0, using identity expert order\n", __func__);
        } else {
            LLAMA_LOG_INFO("%s: loaded expert order for %zu layers (position -> logical in manifest)\n",
                __func__, expert_order_.size());
        }
    }
}

const llama_moe_experts_entry * llama_moe_experts::find(const char * tensor_name) const {
    const auto found = entries.find(tensor_name);
    return found == entries.end() ? nullptr : &found->second;
}

bool llama_moe_experts::packed_index_complete() const {
    return !entries.empty() && std::all_of(entries.begin(), entries.end(),
        [](const auto & item) { return item.second.packed != nullptr; });
}

int32_t llama_moe_experts::expert_count() const {
    return n_expert;
}

int32_t llama_moe_experts::expert_used_count() const {
    return n_expert_used;
}

int32_t llama_moe_experts::layer_count() const {
    return n_layer;
}

size_t llama_moe_experts::bytes_per_slot_all_layers() const {
    return bytes_per_slot;
}

const std::vector<int32_t> * llama_moe_experts::expert_order(int32_t layer) const {
    if (!order_enabled) {
        return nullptr;
    }
    const auto found = expert_order_.find(layer);
    return found == expert_order_.end() ? nullptr : &found->second;
}

bool llama_moe_experts::has_expert_order() const {
    return order_enabled;
}
