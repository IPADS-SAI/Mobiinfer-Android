#include "llama-moe-sidecar.h"

#include "llama-impl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {

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

    throw std::runtime_error(format("unknown MoE sidecar quant type '%s'", name.c_str()));
}

bool routed_family(const std::string & family) {
    return family == "ffn_gate_exps" ||
           family == "ffn_up_exps" ||
           family == "ffn_down_exps" ||
           family == "ffn_gate_up_exps";
}

}

llama_moe_sidecar::llama_moe_sidecar(const char * input_path) {
    namespace fs = std::filesystem;

    if (!input_path || input_path[0] == '\0') {
        throw std::invalid_argument("MoE sidecar path is empty");
    }

    const fs::path input(input_path);
    const fs::path manifest_path = fs::is_directory(input) ? input / "manifest.json" : input;
    const fs::path manifest_dir = manifest_path.parent_path();

    std::ifstream stream(manifest_path);
    if (!stream.is_open()) {
        throw std::runtime_error(format("failed to open MoE sidecar manifest '%s'", manifest_path.string().c_str()));
    }

    nlohmann::json manifest;
    stream >> manifest;

    if (manifest.value("schema_version", 0) != 1) {
        throw std::runtime_error("unsupported MoE sidecar schema version");
    }
    const std::string kind = manifest.value("sidecar_kind", std::string());
    if (kind != "omni_moe_gguf" && kind != "flashmoe_gguf") {
        throw std::runtime_error(format("unsupported MoE sidecar kind '%s'", kind.c_str()));
    }
    const std::string layout = manifest.value("layout", std::string());
    if (layout != "layer_major_expert" && layout != "expert_major") {
        throw std::runtime_error(format("MoE sidecar layout '%s' is not expert-major", layout.c_str()));
    }

    const auto & model = manifest.at("model");
    n_expert = model.at("expert_count").get<int32_t>();
    n_expert_used = model.at("expert_used_count").get<int32_t>();
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert) {
        throw std::runtime_error("invalid expert counts in MoE sidecar manifest");
    }

    std::unordered_set<int32_t> layers;
    for (const auto & item : manifest.at("entries")) {
        const std::string family = item.at("tensor_family").get<std::string>();
        if (!routed_family(family)) {
            continue;
        }

        llama_moe_sidecar_entry entry;
        entry.layer = item.at("layer").get<int32_t>();
        entry.tensor_name = item.at("tensor_name").get<std::string>();
        entry.tensor_family = family;
        entry.path = (manifest_dir / item.at("repacked_file").get<std::string>()).string();
        entry.type = parse_type(item.at("quant_type").get<std::string>());
        entry.offset = item.at("repacked_offset").get<size_t>();
        entry.tensor_bytes = item.at("exact_byte_length").get<size_t>();
        entry.bytes_per_expert = item.at("bytes_per_expert").get<size_t>();
        entry.expert_stride = item.at("expert_stride").get<size_t>();
        entry.expert_count = item.value(
            "expert_count",
            entry.bytes_per_expert == 0 ? 0 : (int32_t) (entry.tensor_bytes / entry.bytes_per_expert));

        if (entry.layer < 0 || entry.bytes_per_expert == 0 || entry.expert_count != n_expert) {
            throw std::runtime_error(format("invalid MoE sidecar entry '%s'", entry.tensor_name.c_str()));
        }
        if (entry.offset > std::numeric_limits<size_t>::max() - entry.bytes_per_expert) {
            throw std::runtime_error(format("MoE sidecar entry '%s' overflows family extent", entry.tensor_name.c_str()));
        }
        const size_t family_end = entry.offset + entry.bytes_per_expert;
        if (entry.expert_stride < family_end) {
            throw std::runtime_error(format("invalid expert stride for MoE sidecar entry '%s'", entry.tensor_name.c_str()));
        }
        if ((size_t) (entry.expert_count - 1) >
            (std::numeric_limits<size_t>::max() - family_end) / entry.expert_stride) {
            throw std::runtime_error(format("MoE sidecar entry '%s' overflows file extent", entry.tensor_name.c_str()));
        }

        const size_t required = (size_t) (entry.expert_count - 1) * entry.expert_stride +
                                family_end;
        std::error_code error;
        const size_t file_size = (size_t) fs::file_size(entry.path, error);
        if (error || file_size < required) {
            throw std::runtime_error(format(
                "MoE sidecar file '%s' is too small for tensor '%s'",
                entry.path.c_str(), entry.tensor_name.c_str()));
        }

        bytes_per_slot += entry.bytes_per_expert;
        layers.insert(entry.layer);
        if (!entries.emplace(entry.tensor_name, std::move(entry)).second) {
            throw std::runtime_error("duplicate tensor in MoE sidecar manifest");
        }
    }

    if (entries.empty()) {
        throw std::runtime_error("MoE sidecar manifest has no routed expert tensors");
    }
    n_layer = (int32_t) layers.size();
}

const llama_moe_sidecar_entry * llama_moe_sidecar::find(const char * tensor_name) const {
    const auto found = entries.find(tensor_name);
    return found == entries.end() ? nullptr : &found->second;
}

int32_t llama_moe_sidecar::expert_count() const {
    return n_expert;
}

int32_t llama_moe_sidecar::expert_used_count() const {
    return n_expert_used;
}

int32_t llama_moe_sidecar::layer_count() const {
    return n_layer;
}

size_t llama_moe_sidecar::bytes_per_slot_all_layers() const {
    return bytes_per_slot;
}
