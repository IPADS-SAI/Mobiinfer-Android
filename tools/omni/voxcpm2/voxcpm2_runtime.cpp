#include "voxcpm2_runtime.h"

#include "ggml-alloc.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

constexpr int   kAudioStartToken       = 101;
constexpr float kDefaultEmbeddingScale = 12.0f;

constexpr size_t kSmallGraphMem  = 32ull * 1024ull * 1024ull;
constexpr size_t kMediumGraphMem = 128ull * 1024ull * 1024ull;
constexpr size_t kLargeGraphMem  = 256ull * 1024ull * 1024ull;

constexpr size_t kSmallGraphNodes  = 8192;
constexpr size_t kMediumGraphNodes = 65536;
constexpr size_t kLargeGraphNodes  = 262144;

struct GgmlContextGuard {
    ggml_context * ctx = nullptr;

    explicit GgmlContextGuard(size_t mem_size, bool no_alloc = true) {
        ggml_init_params params{};
        params.mem_size   = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc   = no_alloc;
        ctx               = ggml_init(params);
    }

    ~GgmlContextGuard() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    ggml_context * get() const { return ctx; }
};

struct BackendBufferGuard {
    ggml_backend_buffer_t buffer = nullptr;

    explicit BackendBufferGuard(ggml_backend_buffer_t buf) : buffer(buf) {}

    ~BackendBufferGuard() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

static std::vector<float> tensor_to_vector(ggml_tensor * tensor) {
    std::vector<float> out(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

static std::vector<float> make_causal_mask(int n_tokens) {
    const int   padded  = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<float> mask(static_cast<size_t>(n_tokens) * static_cast<size_t>(padded), neg_inf);
    for (int q = 0; q < n_tokens; ++q) {
        for (int k = 0; k <= q; ++k) {
            mask[static_cast<size_t>(k) + static_cast<size_t>(q) * static_cast<size_t>(n_tokens)] = 0.0f;
        }
    }
    return mask;
}

static void copy_token(const std::vector<float> & src, int hidden, int token_idx, std::vector<float> & dst) {
    dst.resize(static_cast<size_t>(hidden));
    std::copy_n(src.data() + static_cast<size_t>(token_idx) * static_cast<size_t>(hidden), static_cast<size_t>(hidden),
                dst.data());
}

static bool read_embedding_scale(const std::string & path, float & scale) {
    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        return false;
    }

    const char * keys[] = {
        "minicpm.embedding_scale",
        "llama.embedding_scale",
    };
    bool ok = false;
    for (const char * key : keys) {
        const int64_t id = gguf_find_key(ctx, key);
        if (id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_FLOAT32) {
            scale = gguf_get_val_f32(ctx, id);
            ok    = true;
            break;
        }
    }

    gguf_free(ctx);
    if (meta) {
        ggml_free(meta);
    }
    return ok;
}

static bool read_gguf_string(const std::string & path, const char * key, std::string & value) {
    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        return false;
    }

    bool ok = false;
    const int64_t id = gguf_find_key(ctx, key);
    if (id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING) {
        value = gguf_get_val_str(ctx, id);
        ok    = true;
    }

    gguf_free(ctx);
    if (meta) {
        ggml_free(meta);
    }
    return ok;
}

static bool finite_vector(const std::vector<float> & data) {
    return std::all_of(data.begin(), data.end(), [](float v) { return std::isfinite(v); });
}

static std::string remove_spm_space_marker(std::string text) {
    static const std::string marker = "\xE2\x96\x81";
    size_t                   pos    = 0;
    while ((pos = text.find(marker, pos)) != std::string::npos) {
        text.erase(pos, marker.size());
    }
    return text;
}

static bool utf8_to_codepoints(const std::string & text, std::vector<char32_t> & out) {
    out.clear();
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        char32_t            cp = 0;
        size_t              n  = 0;
        if (c < 0x80) {
            cp = c;
            n  = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            n  = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            n  = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            n  = 4;
        } else {
            return false;
        }
        if (i + n > text.size()) {
            return false;
        }
        for (size_t j = 1; j < n; ++j) {
            const unsigned char cc = static_cast<unsigned char>(text[i + j]);
            if ((cc & 0xC0) != 0x80) {
                return false;
            }
            cp = (cp << 6) | (cc & 0x3F);
        }
        out.push_back(cp);
        i += n;
    }
    return true;
}

static std::string utf8_from_codepoint(char32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

static bool is_cjk_codepoint(char32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x20000 && cp <= 0x2A6DF);
}

}  // namespace

bool VoxCPM2TokenEmbeddingTable::load(const std::string & gguf_path) {
    free();

    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;

    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf || !meta) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: failed to open GGUF: %s\n", gguf_path.c_str());
        if (ctx_gguf) {
            gguf_free(ctx_gguf);
        }
        if (meta) {
            ggml_free(meta);
        }
        return false;
    }

    const int64_t tensor_id = gguf_find_tensor(ctx_gguf, "token_embd.weight");
    if (tensor_id < 0) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: token_embd.weight not found in %s\n", gguf_path.c_str());
        gguf_free(ctx_gguf);
        ggml_free(meta);
        return false;
    }

    ggml_tensor * tensor = ggml_get_tensor(meta, "token_embd.weight");
    if (!tensor || ggml_n_dims(tensor) < 2) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: invalid token_embd.weight metadata\n");
        gguf_free(ctx_gguf);
        ggml_free(meta);
        return false;
    }

    path        = gguf_path;
    type        = tensor->type;
    n_embd      = static_cast<int>(tensor->ne[0]);
    n_vocab     = static_cast<int>(tensor->ne[1]);
    row_bytes   = ggml_row_size(type, n_embd);
    data_offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, tensor_id);

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && (!traits || !traits->to_float)) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: unsupported token embedding type %s\n", ggml_type_name(type));
        free();
        gguf_free(ctx_gguf);
        ggml_free(meta);
        return false;
    }

    LOG_INF("VoxCPM2TokenEmbeddingTable: loaded metadata vocab=%d embd=%d type=%s\n", n_vocab, n_embd,
            ggml_type_name(type));

    gguf_free(ctx_gguf);
    ggml_free(meta);
    return true;
}

bool VoxCPM2TokenEmbeddingTable::embedding_for_token(int32_t token_id, std::vector<float> & dst) const {
    if (path.empty() || n_embd <= 0 || n_vocab <= 0 || row_bytes == 0) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: table is not loaded\n");
        return false;
    }
    if (token_id < 0 || token_id >= n_vocab) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: token id %d out of range [0, %d)\n", token_id, n_vocab);
        return false;
    }

    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: failed to open %s\n", path.c_str());
        return false;
    }

    const size_t offset = data_offset + static_cast<size_t>(token_id) * row_bytes;
    fin.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!fin) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: seek failed for token %d\n", token_id);
        return false;
    }

    std::vector<uint8_t> row(row_bytes);
    fin.read(reinterpret_cast<char *>(row.data()), static_cast<std::streamsize>(row.size()));
    if (!fin) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: read failed for token %d\n", token_id);
        return false;
    }

    dst.resize(static_cast<size_t>(n_embd));
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst.data(), row.data(), dst.size() * sizeof(float));
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        LOG_ERR("VoxCPM2TokenEmbeddingTable: unsupported token embedding type %s\n", ggml_type_name(type));
        return false;
    }
    traits->to_float(row.data(), dst.data(), n_embd);
    return true;
}

void VoxCPM2TokenEmbeddingTable::free() {
    path.clear();
    data_offset = 0;
    row_bytes   = 0;
    n_embd      = 0;
    n_vocab     = 0;
    type        = GGML_TYPE_COUNT;
}

VoxCPM2ResidualLM::~VoxCPM2ResidualLM() {
    free();
}

bool VoxCPM2ResidualLM::init_from_gguf(const std::string & path, ggml_backend_t backend_in) {
    free();
    if (!backend_in) {
        LOG_ERR("VoxCPM2ResidualLM: backend is null\n");
        return false;
    }
    backend = backend_in;

    store = std::make_unique<VoxCPM2GGUFWeightStore>();
    if (!store->load(path, backend, { "residual_lm." })) {
        free();
        return false;
    }

    store->get_u32("voxcpm2.residual_lm.n_layer", config.n_layer);
    store->get_u32("voxcpm2.residual_lm.n_embd", config.hidden_size);
    config.n_heads        = 16;
    config.n_kv_heads     = 2;
    config.head_dim       = config.hidden_size / config.n_heads;
    config.max_length     = 32768;
    config.no_rope        = true;
    config.use_flash_attn = false;

    if (!voxcpm2_bind_transformer_weights(store->tensors, "residual_lm", config, weights)) {
        free();
        return false;
    }

    LOG_INF("VoxCPM2ResidualLM: loaded layers=%d hidden=%d heads=%d kv_heads=%d\n", config.n_layer, config.hidden_size,
            config.n_heads, config.n_kv_heads);
    return true;
}

std::vector<float> VoxCPM2ResidualLM::forward(const std::vector<float> & input, int seq_len) const {
    if (!backend || seq_len <= 0 || config.hidden_size <= 0 ||
        input.size() != static_cast<size_t>(config.hidden_size) * static_cast<size_t>(seq_len)) {
        LOG_ERR("VoxCPM2ResidualLM::forward: invalid input\n");
        return {};
    }

    GgmlContextGuard ctx_guard(kMediumGraphMem, true);
    ggml_context *   ctx = ctx_guard.get();
    if (!ctx) {
        LOG_ERR("VoxCPM2ResidualLM::forward: failed to create context\n");
        return {};
    }

    ggml_tensor * input_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, seq_len);
    ggml_set_input(input_t);
    ggml_tensor * mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, GGML_PAD(seq_len, GGML_KQ_MASK_PAD));
    ggml_set_input(mask_t);

    ggml_tensor * output = voxcpm2_transformer_forward(ctx, config, weights, input_t, nullptr, mask_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kMediumGraphNodes, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    BackendBufferGuard buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buffer.buffer) {
        LOG_ERR("VoxCPM2ResidualLM::forward: failed to allocate graph tensors\n");
        return {};
    }

    const std::vector<float> mask = make_causal_mask(seq_len);
    ggml_backend_tensor_set(input_t, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("VoxCPM2ResidualLM::forward: graph compute failed with status %d\n", static_cast<int>(status));
        return {};
    }

    return tensor_to_vector(output);
}

std::vector<float> VoxCPM2ResidualLM::forward_last(const std::vector<float> & input, int seq_len) const {
    std::vector<float> all = forward(input, seq_len);
    if (all.empty()) {
        return {};
    }
    std::vector<float> last;
    copy_token(all, config.hidden_size, seq_len - 1, last);
    return last;
}

void VoxCPM2ResidualLM::free() {
    store.reset();
    weights = {};
    config  = {};
    backend = nullptr;
}

VoxCPM2Runtime::VoxCPM2Runtime() : rng(0) {}

VoxCPM2Runtime::~VoxCPM2Runtime() {
    free();
}

bool VoxCPM2Runtime::fail(const std::string & message) {
    last_error_msg = message;
    LOG_ERR("VoxCPM2Runtime: %s\n", message.c_str());
    return false;
}

void VoxCPM2Runtime::clear_error() {
    last_error_msg.clear();
}

bool VoxCPM2Runtime::init(const std::string & base_lm_path,
                          const std::string & acoustic_path,
                          int                 n_gpu_layers,
                          bool                use_gpu_backend) {
    free();
    clear_error();

    ggml_backend_load_all();

    if (use_gpu_backend) {
        backend = ggml_backend_init_best();
        if (!backend) {
            LOG_WRN("VoxCPM2Runtime: no GPU backend found, falling back to CPU\n");
        }
    }
    if (!backend) {
        backend = ggml_backend_init_by_name("cpu", nullptr);
    }
    if (!backend) {
        return fail("failed to initialize ggml backend");
    }

    LOG_INF("VoxCPM2Runtime: custom component backend=%s\n", ggml_backend_name(backend));

    if (!base_lm.init(base_lm_path, n_gpu_layers)) {
        return fail("failed to initialize BaseLM");
    }
    if (!token_embeddings.load(base_lm_path)) {
        return fail("failed to load BaseLM token embeddings");
    }
    if (token_embeddings.n_embd != base_lm.n_embd) {
        return fail("token embedding size does not match BaseLM hidden size");
    }
    if (!build_text_tokenizer_metadata(base_lm_path)) {
        return fail("failed to initialize text tokenizer metadata");
    }

    embedding_scale = kDefaultEmbeddingScale;
    if (!read_embedding_scale(base_lm_path, embedding_scale)) {
        LOG_WRN("VoxCPM2Runtime: embedding_scale metadata missing, using %.3f\n", embedding_scale);
    }

    if (!fsq.init_from_gguf(acoustic_path, backend)) {
        return fail("failed to initialize FSQ");
    }
    if (!residual_lm.init_from_gguf(acoustic_path, backend)) {
        return fail("failed to initialize ResidualLM");
    }
    if (!loc_enc.init_from_gguf(acoustic_path, backend)) {
        return fail("failed to initialize LocEnc");
    }
    if (!loc_dit.init_from_gguf(acoustic_path, backend)) {
        return fail("failed to initialize LocDiT");
    }
    if (!projections.init_from_gguf(acoustic_path, backend)) {
        return fail("failed to initialize projections");
    }
    if (!stop_predictor.init_from_gguf(acoustic_path, backend)) {
        return fail("failed to initialize stop predictor");
    }
    if (!audio_vae.init_from_gguf(acoustic_path, backend)) {
        return fail("failed to initialize AudioVAE");
    }

    cfm_solver.config.cfg_rate = loc_dit.config.cfg_rate;
    rng.seed(0);

    is_initialized = true;
    reset_state();
    LOG_INF("VoxCPM2Runtime: initialized (embedding_scale=%.3f)\n", embedding_scale);
    return true;
}

bool VoxCPM2Runtime::build_text_tokenizer_metadata(const std::string & base_lm_path) {
    tokenizer_available = false;
    cjk_split_map.clear();
    cjk_prefix_single_split_map.clear();
    cjk_token_ids.clear();

    std::string tokenizer_model;
    if (!read_gguf_string(base_lm_path, "tokenizer.ggml.model", tokenizer_model) ||
        tokenizer_model == "no_vocab" || tokenizer_model == "none") {
        LOG_WRN("VoxCPM2Runtime: BaseLM GGUF has no text tokenizer; generate(text) is disabled\n");
        return true;
    }

    const llama_vocab * vocab = base_lm.vocab;
    if (!vocab) {
        return false;
    }

    const int n_tokens = llama_vocab_n_tokens(vocab);
    std::unordered_map<std::string, int32_t> piece_to_id;
    piece_to_id.reserve(static_cast<size_t>(n_tokens));
    for (int32_t id = 0; id < n_tokens; ++id) {
        const char * text = llama_vocab_get_text(vocab, id);
        if (text) {
            piece_to_id.emplace(text, id);
        }
    }

    const auto marker_it = piece_to_id.find("\xE2\x96\x81");
    const int32_t spm_space_id = marker_it == piece_to_id.end() ? -1 : marker_it->second;

    std::vector<char32_t> codepoints;
    for (int32_t id = 0; id < n_tokens; ++id) {
        const char * raw = llama_vocab_get_text(vocab, id);
        if (!raw) {
            continue;
        }

        const std::string raw_text = raw;
        const std::string clean    = remove_spm_space_marker(raw);
        if (!utf8_to_codepoints(clean, codepoints)) {
            continue;
        }
        if (codepoints.size() < 2) {
            if (codepoints.size() == 1 && is_cjk_codepoint(codepoints[0])) {
                cjk_token_ids.insert(id);
                if (spm_space_id >= 0 && raw_text.rfind("\xE2\x96\x81", 0) == 0) {
                    const std::string ch = utf8_from_codepoint(codepoints[0]);
                    const auto        it = piece_to_id.find(ch);
                    if (it != piece_to_id.end() && it->second != 0) {
                        cjk_prefix_single_split_map.emplace(id, std::vector<int32_t>{ spm_space_id, it->second });
                    }
                }
            }
            continue;
        }
        if (!std::all_of(codepoints.begin(), codepoints.end(), is_cjk_codepoint)) {
            continue;
        }
        cjk_token_ids.insert(id);

        std::vector<int32_t> split_ids;
        split_ids.reserve(codepoints.size());
        bool can_split = true;
        for (const char32_t cp : codepoints) {
            const std::string ch = utf8_from_codepoint(cp);
            const auto        it = piece_to_id.find(ch);
            if (it == piece_to_id.end() || it->second == 0) {
                can_split = false;
                break;
            }
            split_ids.push_back(it->second);
        }
        if (can_split) {
            cjk_token_ids.insert(split_ids.begin(), split_ids.end());
            cjk_split_map.emplace(id, std::move(split_ids));
        }
    }

    tokenizer_available = true;
    LOG_INF("VoxCPM2Runtime: text tokenizer enabled (%d tokens, %zu CJK split entries, %zu CJK prefix entries)\n",
            n_tokens, cjk_split_map.size(), cjk_prefix_single_split_map.size());
    return true;
}

void VoxCPM2Runtime::reset_state() {
    lm_hidden.clear();
    residual_hidden.clear();
    prefix_feat_cond.assign(static_cast<size_t>(feat_dim()) * static_cast<size_t>(patch_size()), 0.0f);
    residual_input_history.clear();
    output_pool.clear();
    current_position  = 0;
    audio_frame_count = 0;
    state_ready       = false;
    base_lm.clear_kv_cache();
}

bool VoxCPM2Runtime::prefill_tokens(const std::vector<int32_t> & token_ids) {
    VoxCPM2PrefillInputs inputs;
    inputs.token_ids = token_ids;
    inputs.text_mask.assign(token_ids.size(), 1);
    inputs.feat_mask.assign(token_ids.size(), 0);
    return prefill(inputs);
}

bool VoxCPM2Runtime::prefill(const VoxCPM2PrefillInputs & inputs) {
    clear_error();
    if (!is_initialized) {
        return fail("runtime is not initialized");
    }
    if (inputs.token_ids.empty()) {
        return fail("prefill requires at least one token");
    }

    const int seq_len     = static_cast<int>(inputs.token_ids.size());
    const int hidden      = base_lm.n_embd;
    const int fdim        = feat_dim();
    const int psize       = patch_size();
    const int patch_elems = fdim * psize;

    std::vector<int32_t> text_mask = inputs.text_mask;
    std::vector<int32_t> feat_mask = inputs.feat_mask;
    if (text_mask.empty()) {
        text_mask.assign(static_cast<size_t>(seq_len), 1);
    }
    if (feat_mask.empty()) {
        feat_mask.assign(static_cast<size_t>(seq_len), 0);
    }
    if (text_mask.size() != static_cast<size_t>(seq_len) || feat_mask.size() != static_cast<size_t>(seq_len)) {
        return fail("prefill mask sizes must match token_ids");
    }

    const bool has_feat = std::any_of(feat_mask.begin(), feat_mask.end(), [](int32_t v) { return v != 0; });
    if (has_feat && inputs.audio_feat.size() != static_cast<size_t>(seq_len) * static_cast<size_t>(patch_elems)) {
        return fail("audio_feat must be [feat_dim, patch_size, seq_len] when feat_mask is set");
    }

    reset_state();

    std::vector<float> combined(static_cast<size_t>(hidden) * static_cast<size_t>(seq_len), 0.0f);
    std::vector<float> token_embed;
    for (int i = 0; i < seq_len; ++i) {
        if (text_mask[static_cast<size_t>(i)] == 0) {
            continue;
        }
        if (!token_embeddings.embedding_for_token(inputs.token_ids[static_cast<size_t>(i)], token_embed)) {
            return fail("failed to read token embedding");
        }
        std::copy_n(token_embed.data(), static_cast<size_t>(hidden),
                    combined.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden));
    }

    std::vector<float> feat_embed(static_cast<size_t>(hidden) * static_cast<size_t>(seq_len), 0.0f);
    if (has_feat) {
        std::vector<float> loc_hidden = run_locenc_sequence(inputs.audio_feat, seq_len);
        if (loc_hidden.empty()) {
            return fail("LocEnc sequence forward failed");
        }
        feat_embed = run_enc_to_lm(loc_hidden, seq_len);
        if (feat_embed.empty()) {
            return fail("enc_to_lm projection failed");
        }

        for (int i = 0; i < seq_len; ++i) {
            if (feat_mask[static_cast<size_t>(i)] == 0) {
                continue;
            }
            std::copy_n(feat_embed.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden),
                        static_cast<size_t>(hidden),
                        combined.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden));
        }
    }

    std::vector<float> base_input = combined;
    if (embedding_scale != 0.0f) {
        for (float & v : base_input) {
            v /= embedding_scale;
        }
    }

    if (!base_lm.prefill(base_input.data(), seq_len, 0)) {
        return fail("BaseLM prefill failed");
    }
    std::vector<float> base_hidden = base_lm.get_all_hidden();
    if (base_hidden.size() != static_cast<size_t>(hidden) * static_cast<size_t>(seq_len)) {
        return fail("BaseLM did not return all prefill hidden states");
    }

    std::vector<float> blended = base_hidden;
    if (has_feat) {
        std::vector<float> fsq_hidden = run_fsq(base_hidden, seq_len);
        if (fsq_hidden.empty()) {
            return fail("FSQ prefill forward failed");
        }
        for (int i = 0; i < seq_len; ++i) {
            if (feat_mask[static_cast<size_t>(i)] == 0) {
                continue;
            }
            std::copy_n(fsq_hidden.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden),
                        static_cast<size_t>(hidden),
                        blended.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden));
        }
    }

    std::vector<float> feat_embed_masked(static_cast<size_t>(hidden) * static_cast<size_t>(seq_len), 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        if (feat_mask[static_cast<size_t>(i)] == 0) {
            continue;
        }
        std::copy_n(combined.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden), static_cast<size_t>(hidden),
                    feat_embed_masked.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden));
    }

    residual_input_history = run_residual_fusion(blended, feat_embed_masked, seq_len);
    if (residual_input_history.empty()) {
        return fail("residual fusion prefill failed");
    }

    std::vector<float> residual_all = residual_lm.forward(residual_input_history, seq_len);
    if (residual_all.size() != static_cast<size_t>(hidden) * static_cast<size_t>(seq_len)) {
        return fail("ResidualLM prefill failed");
    }

    copy_token(blended, hidden, seq_len - 1, lm_hidden);
    copy_token(residual_all, hidden, seq_len - 1, residual_hidden);

    prefix_feat_cond.assign(static_cast<size_t>(patch_elems), 0.0f);
    output_pool.clear();
    if (has_feat) {
        int last_feat_idx = -1;
        for (int i = 0; i < seq_len; ++i) {
            if (feat_mask[static_cast<size_t>(i)] == 0) {
                continue;
            }
            last_feat_idx = i;
        }
        if (last_feat_idx >= 0) {
            const float * patch =
                inputs.audio_feat.data() + static_cast<size_t>(last_feat_idx) * static_cast<size_t>(patch_elems);
            std::copy_n(patch, static_cast<size_t>(patch_elems), prefix_feat_cond.data());
        }
    }

    current_position  = seq_len;
    audio_frame_count = 0;
    state_ready       = true;

    return true;
}

std::vector<float> VoxCPM2Runtime::random_noise() {
    const int                       n = feat_dim() * patch_size();
    std::vector<float>              noise(static_cast<size_t>(n));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float & v : noise) {
        v = dist(rng);
    }
    return noise;
}

std::vector<float> VoxCPM2Runtime::run_locenc_sequence(const std::vector<float> & feat, int seq_len) {
    GgmlContextGuard ctx_guard(kLargeGraphMem, true);
    ggml_context *   ctx = ctx_guard.get();
    if (!ctx) {
        return {};
    }

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, feat_dim(), patch_size(), seq_len);
    ggml_set_input(input);
    ggml_tensor * output = loc_enc.forward_sequence(ctx, input);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kLargeGraphNodes, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    BackendBufferGuard buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buffer.buffer) {
        return {};
    }

    ggml_backend_tensor_set(input, feat.data(), 0, feat.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return {};
    }
    return tensor_to_vector(output);
}

std::vector<float> VoxCPM2Runtime::run_enc_to_lm(const std::vector<float> & locenc_hidden, int seq_len) {
    GgmlContextGuard ctx_guard(kSmallGraphMem, true);
    ggml_context *   ctx = ctx_guard.get();
    if (!ctx) {
        return {};
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, loc_enc.config.transformer.hidden_size, seq_len);
    ggml_set_input(input);
    ggml_tensor * output = projections.enc_to_lm(ctx, input);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kSmallGraphNodes, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    BackendBufferGuard buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buffer.buffer) {
        return {};
    }

    ggml_backend_tensor_set(input, locenc_hidden.data(), 0, locenc_hidden.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return {};
    }
    return tensor_to_vector(output);
}

std::vector<float> VoxCPM2Runtime::run_fsq(const std::vector<float> & input_vec, int seq_len) {
    GgmlContextGuard ctx_guard(kSmallGraphMem, true);
    ggml_context *   ctx = ctx_guard.get();
    if (!ctx) {
        return {};
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fsq.config.hidden_size, seq_len);
    ggml_set_input(input);
    ggml_tensor * output = fsq.forward(ctx, input);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kMediumGraphNodes, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    BackendBufferGuard buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buffer.buffer) {
        return {};
    }

    ggml_backend_tensor_set(input, input_vec.data(), 0, input_vec.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return {};
    }
    return tensor_to_vector(output);
}

std::vector<float> VoxCPM2Runtime::run_residual_fusion(const std::vector<float> & blended,
                                                       const std::vector<float> & feat_embed,
                                                       int                        seq_len) {
    GgmlContextGuard ctx_guard(kSmallGraphMem, true);
    ggml_context *   ctx = ctx_guard.get();
    if (!ctx) {
        return {};
    }

    ggml_tensor * blended_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, projections.config.lm_hidden_size, seq_len);
    ggml_tensor * feat_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, projections.config.lm_hidden_size, seq_len);
    ggml_set_input(blended_t);
    ggml_set_input(feat_t);
    ggml_tensor * output = projections.build_residual_fusion(ctx, blended_t, feat_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kSmallGraphNodes, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    BackendBufferGuard buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buffer.buffer) {
        return {};
    }

    ggml_backend_tensor_set(blended_t, blended.data(), 0, blended.size() * sizeof(float));
    ggml_backend_tensor_set(feat_t, feat_embed.data(), 0, feat_embed.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return {};
    }
    return tensor_to_vector(output);
}

bool VoxCPM2Runtime::run_decode_front_half(const std::vector<float> &    noise,
                                           const VoxCPM2GenerateParams & params,
                                           VoxCPM2DecodeStepResult &     result) {
    const int patch_elems = feat_dim() * patch_size();
    if (noise.size() != static_cast<size_t>(patch_elems)) {
        return fail("decode noise size does not match feat_dim * patch_size");
    }

    GgmlContextGuard ctx_guard(kLargeGraphMem, true);
    ggml_context *   ctx = ctx_guard.get();
    if (!ctx) {
        return fail("failed to create decode graph context");
    }

    ggml_tensor * noise_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim(), patch_size());
    ggml_tensor * prefix_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim(), patch_size());
    ggml_tensor * lm_t     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, base_lm.n_embd);
    ggml_tensor * res_t    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, residual_lm.config.hidden_size);
    ggml_set_input(noise_t);
    ggml_set_input(prefix_t);
    ggml_set_input(lm_t);
    ggml_set_input(res_t);

    ggml_tensor * mu      = projections.build_dit_condition(ctx, lm_t, res_t);
    ggml_tensor * patch   = cfm_solver.solve(ctx, noise_t, mu, prefix_t, loc_dit, params.inference_timesteps,
                                             params.cfg_value, params.temperature, cfm_solver.config.sway_sampling_coef,
                                             cfm_solver.config.use_cfg_zero_star, nullptr);
    ggml_tensor * encoded = loc_enc.forward_patch(ctx, patch);
    ggml_tensor * embed   = projections.enc_to_lm(ctx, encoded);
    ggml_tensor * stop    = stop_predictor.forward(ctx, lm_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kLargeGraphNodes, false);
    ggml_set_output(patch);
    ggml_set_output(embed);
    ggml_set_output(stop);
    ggml_build_forward_expand(graph, patch);
    ggml_build_forward_expand(graph, embed);
    ggml_build_forward_expand(graph, stop);

    BackendBufferGuard buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buffer.buffer) {
        return fail("failed to allocate decode graph tensors");
    }

    ggml_backend_tensor_set(noise_t, noise.data(), 0, noise.size() * sizeof(float));
    ggml_backend_tensor_set(prefix_t, prefix_feat_cond.data(), 0, prefix_feat_cond.size() * sizeof(float));
    ggml_backend_tensor_set(lm_t, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
    ggml_backend_tensor_set(res_t, residual_hidden.data(), 0, residual_hidden.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return fail("decode front-half graph compute failed");
    }

    result.latent_patch            = tensor_to_vector(patch);
    result.current_embed           = tensor_to_vector(embed);
    std::vector<float> stop_logits = tensor_to_vector(stop);
    if (stop_logits.size() >= 2) {
        result.continue_logit = stop_logits[0];
        result.stop_logit     = stop_logits[1];
        result.should_stop    = stop_logits[1] > stop_logits[0];
    }
    result.position = current_position;
    return true;
}

VoxCPM2DecodeStepResult VoxCPM2Runtime::decode_step(const VoxCPM2GenerateParams & params) {
    if (params.seed != 0) {
        rng.seed(params.seed + static_cast<uint32_t>(current_position));
    }
    return decode_step(random_noise(), params);
}

VoxCPM2DecodeStepResult VoxCPM2Runtime::decode_step(const std::vector<float> &    noise,
                                                    const VoxCPM2GenerateParams & params) {
    clear_error();
    VoxCPM2DecodeStepResult result;
    if (!is_initialized || !state_ready) {
        fail("decode_step requires initialized runtime and successful prefill");
        return result;
    }

    if (!run_decode_front_half(noise, params, result)) {
        return result;
    }
    if (!finite_vector(result.latent_patch) || !finite_vector(result.current_embed)) {
        fail("decode_step produced non-finite patch or embedding");
        return {};
    }

    const int          hidden     = base_lm.n_embd;
    std::vector<float> base_input = result.current_embed;
    if (embedding_scale != 0.0f) {
        for (float & v : base_input) {
            v /= embedding_scale;
        }
    }

    if (!base_lm.decode_step(base_input.data(), current_position)) {
        fail("BaseLM decode_step failed");
        return {};
    }

    std::vector<float> base_hidden = base_lm.get_last_hidden();
    if (base_hidden.size() != static_cast<size_t>(hidden)) {
        fail("BaseLM decode_step did not return a hidden state");
        return {};
    }

    lm_hidden = run_fsq(base_hidden, 1);
    if (lm_hidden.size() != static_cast<size_t>(hidden)) {
        fail("FSQ decode hidden forward failed");
        return {};
    }

    std::vector<float> fusion = run_residual_fusion(lm_hidden, result.current_embed, 1);
    if (fusion.size() != static_cast<size_t>(hidden)) {
        fail("residual fusion decode failed");
        return {};
    }
    residual_input_history.insert(residual_input_history.end(), fusion.begin(), fusion.end());
    const int residual_seq_len = static_cast<int>(residual_input_history.size() / static_cast<size_t>(hidden));
    residual_hidden            = residual_lm.forward_last(residual_input_history, residual_seq_len);
    if (residual_hidden.size() != static_cast<size_t>(hidden)) {
        fail("ResidualLM decode forward failed");
        return {};
    }

    prefix_feat_cond = result.latent_patch;
    output_pool.push_back(result.latent_patch);
    ++audio_frame_count;
    ++current_position;
    state_ready = true;
    return result;
}

void VoxCPM2Runtime::decode_loop(const VoxCPM2GenerateParams &                                params,
                                 const std::function<void(const VoxCPM2DecodeStepResult &)> & callback) {
    for (int i = 0; i < params.max_steps; ++i) {
        VoxCPM2DecodeStepResult step = decode_step(params);
        if (step.latent_patch.empty()) {
            break;
        }
        if (callback) {
            callback(step);
        }
        if (params.stop_on_predictor && step.should_stop) {
            break;
        }
    }
}

std::vector<float> VoxCPM2Runtime::decode_to_waveform(int target_sr) {
    clear_error();
    if (!is_initialized) {
        fail("runtime is not initialized");
        return {};
    }
    if (output_pool.empty()) {
        return {};
    }

    const int fdim         = feat_dim();
    const int psize        = patch_size();
    const int n_patches    = static_cast<int>(output_pool.size());
    const int total_frames = n_patches * psize;

    std::vector<float> latents(static_cast<size_t>(total_frames) * static_cast<size_t>(fdim), 0.0f);
    for (int p = 0; p < n_patches; ++p) {
        const std::vector<float> & patch = output_pool[static_cast<size_t>(p)];
        if (patch.size() != static_cast<size_t>(fdim * psize)) {
            fail("output_pool contains an invalid latent patch");
            return {};
        }
        for (int t = 0; t < psize; ++t) {
            const int frame = p * psize + t;
            for (int c = 0; c < fdim; ++c) {
                latents[static_cast<size_t>(frame) + static_cast<size_t>(c) * static_cast<size_t>(total_frames)] =
                    patch[static_cast<size_t>(c) + static_cast<size_t>(t) * static_cast<size_t>(fdim)];
            }
        }
    }

    GgmlContextGuard ctx_guard(kLargeGraphMem, true);
    ggml_context *   ctx = ctx_guard.get();
    if (!ctx) {
        fail("failed to create AudioVAE decode context");
        return {};
    }

    ggml_tensor * latents_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_frames, fdim);
    ggml_set_input(latents_t);
    ggml_tensor * waveform =
        audio_vae.decode(ctx, latents_t, target_sr > 0 ? target_sr : audio_vae.config.output_sample_rate());

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kLargeGraphNodes, false);
    ggml_set_output(waveform);
    ggml_build_forward_expand(graph, waveform);

    BackendBufferGuard buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buffer.buffer) {
        fail("failed to allocate AudioVAE decode graph tensors");
        return {};
    }

    ggml_backend_tensor_set(latents_t, latents.data(), 0, latents.size() * sizeof(float));
    audio_vae.prepare_decode_inputs();
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode graph compute failed");
        return {};
    }

    return tensor_to_vector(waveform);
}

std::vector<int32_t> VoxCPM2Runtime::expand_multichar_cjk_tokens(const std::vector<int32_t> & ids) const {
    std::vector<int32_t> expanded;
    expanded.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];
        const auto it = cjk_split_map.find(id);
        if (it == cjk_split_map.end()) {
            const auto prefix_it = cjk_prefix_single_split_map.find(id);
            if (prefix_it != cjk_prefix_single_split_map.end() && i + 1 < ids.size() &&
                cjk_token_ids.find(ids[i + 1]) != cjk_token_ids.end()) {
                expanded.insert(expanded.end(), prefix_it->second.begin(), prefix_it->second.end());
            } else {
                expanded.push_back(id);
            }
        } else {
            expanded.insert(expanded.end(), it->second.begin(), it->second.end());
        }
    }
    return expanded;
}

std::vector<int32_t> VoxCPM2Runtime::tokenize_text(const std::string & text,
                                                   bool                add_special,
                                                   bool                parse_special) {
    clear_error();
    if (!is_initialized) {
        fail("runtime is not initialized");
        return {};
    }
    if (!tokenizer_available || !base_lm.vocab) {
        fail("text tokenization is unavailable; re-export BaseLM GGUF with tokenizer.json metadata");
        return {};
    }

    std::vector<llama_token> tokens(static_cast<size_t>(text.size()) + 8);
    int32_t n_tokens = llama_tokenize(base_lm.vocab, text.data(), static_cast<int32_t>(text.size()),
                                      tokens.data(), static_cast<int32_t>(tokens.size()),
                                      add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        fail("text tokenization overflowed int32_t token count");
        return {};
    }
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(base_lm.vocab, text.data(), static_cast<int32_t>(text.size()),
                                  tokens.data(), static_cast<int32_t>(tokens.size()),
                                  add_special, parse_special);
    }
    if (n_tokens < 0) {
        fail("llama_tokenize failed");
        return {};
    }

    tokens.resize(static_cast<size_t>(n_tokens));
    std::vector<int32_t> ids(tokens.begin(), tokens.end());
    return expand_multichar_cjk_tokens(ids);
}

std::vector<float> VoxCPM2Runtime::generate_tokens(const std::vector<int32_t> &  token_ids,
                                                   const VoxCPM2GenerateParams & params) {
    clear_error();
    if (params.seed != 0) {
        rng.seed(params.seed);
    }

    std::vector<int32_t> prompt = token_ids;
    if (params.append_audio_start && (prompt.empty() || prompt.back() != kAudioStartToken)) {
        prompt.push_back(kAudioStartToken);
    }
    if (!prefill_tokens(prompt)) {
        return {};
    }
    decode_loop(params, nullptr);
    if (!last_error_msg.empty()) {
        return {};
    }
    return decode_to_waveform(params.target_sr);
}

std::vector<float> VoxCPM2Runtime::generate(const std::string & text, const VoxCPM2GenerateParams & params) {
    std::vector<int32_t> token_ids = tokenize_text(text, true, true);
    if (token_ids.empty()) {
        if (last_error_msg.empty()) {
            fail("text tokenization produced no tokens");
        }
        return {};
    }
    return generate_tokens(token_ids, params);
}

void VoxCPM2Runtime::free() {
    reset_state();
    audio_vae.free();
    stop_predictor.free();
    projections.free();
    loc_dit.free();
    loc_enc.free();
    residual_lm.free();
    fsq.free();
    token_embeddings.free();
    base_lm.free();

    if (backend) {
        ggml_backend_free(backend);
        backend = nullptr;
    }

    is_initialized  = false;
    state_ready     = false;
    tokenizer_available = false;
    cjk_split_map.clear();
    cjk_prefix_single_split_map.clear();
    cjk_token_ids.clear();
    embedding_scale = kDefaultEmbeddingScale;
}
