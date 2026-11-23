#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Routing tracer used by T-0041.  Differences from tools/moe/profile_hot_experts.cpp:
//   * tokenizes the prompt verbatim (no chat template), so wikitext / held-out
//     documents are profiled as corpus text rather than as chat turns;
//   * also captures ffn_moe_probs-<layer> and emits the router probability for
//     every selected expert (per-token weights, not just ids).
//
// The callback synchronizes the backend, so timings from this binary are not
// benchmarks.

static int env_int(const char * name, int fallback, int lo, int hi) {
    const char * value = std::getenv(name);
    const int parsed = value ? std::atoi(value) : fallback;
    return std::max(lo, std::min(hi, parsed));
}

struct route_trace {
    std::ofstream output;
    std::string sample;
    const char * phase = "prefill";
    size_t records = 0;
    std::unordered_map<int, std::vector<float>> probs; // layer -> [n_expert, n_tokens]
    std::unordered_map<int, int64_t> probs_tokens;
    int n_expert = 0;
};

static bool parse_exact(const std::string & name, const char * prefix, int * layer) {
    const size_t n = std::strlen(prefix);
    if (name.compare(0, n, prefix) != 0) return false;
    const std::string rest = name.substr(n);
    if (rest.empty() || !std::all_of(rest.begin(), rest.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return false;
    }
    *layer = std::stoi(rest);
    return true;
}

static bool is_topk(const std::string & name, int * layer) {
    return parse_exact(name, "ffn_moe_topk-", layer);
}

static bool is_probs(const std::string & name, int * layer) {
    return parse_exact(name, "ffn_moe_probs-", layer);
}

static bool capture_routes(ggml_tensor * tensor, bool ask, void * opaque) {
    const std::string name = ggml_get_name(tensor);
    int layer = -1;
    const bool topk = is_topk(name, &layer);
    const bool probs = is_probs(name, &layer);
    if (!topk && !probs) return false;
    if (ask) {
        if (std::getenv("PROFILE_DEBUG") && layer == 0) {
            std::cerr << "ASK " << name << "\n";
        }
        return true;
    }

    auto & trace = *static_cast<route_trace *>(opaque);
    if (probs) {
        if (tensor->type != GGML_TYPE_F32) throw std::runtime_error("router probs must be f32");
        const int64_t n_expert = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        std::vector<float> values(static_cast<size_t>(n_expert * n_tokens));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        trace.probs[layer] = std::move(values);
        trace.probs_tokens[layer] = n_tokens;
        trace.n_expert = static_cast<int>(n_expert);
        return true;
    }

    if (tensor->type != GGML_TYPE_I32) throw std::runtime_error("router ids must be int32");
    const int64_t k = tensor->ne[0];
    const int64_t n_tokens = tensor->ne[1];
    std::vector<uint8_t> ids_bytes(static_cast<size_t>(ggml_nbytes(tensor)));
    ggml_backend_tensor_get(tensor, ids_bytes.data(), 0, ids_bytes.size());

    const auto it = trace.probs.find(layer);
    const bool have_weights = it != trace.probs.end() && trace.probs_tokens[layer] == n_tokens;
    auto rows = nlohmann::json::array();
    auto wrows = nlohmann::json::array();
    for (int64_t row = 0; row < n_tokens; ++row) {
        auto experts = nlohmann::json::array();
        auto weights = nlohmann::json::array();
        for (int64_t rank = 0; rank < k; ++rank) {
            int32_t expert;
            std::memcpy(&expert, ids_bytes.data() + row * tensor->nb[1] + rank * tensor->nb[0],
                        sizeof(int32_t));
            experts.push_back(expert);
            if (have_weights) {
                weights.push_back(it->second[static_cast<size_t>(row) * trace.n_expert + expert]);
            }
        }
        rows.push_back(std::move(experts));
        if (have_weights) wrows.push_back(std::move(weights));
    }
    nlohmann::json record = {{"sample", trace.sample}, {"phase", trace.phase},
                             {"layer", layer}, {"experts", rows}};
    if (have_weights) record["weights"] = wrows;
    trace.output << record.dump() << '\n';
    trace.records++;
    // The probs row is only needed by the topk of the same evaluation.
    trace.probs.erase(layer);
    trace.probs_tokens.erase(layer);
    return true;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: llama-moe-profile-raw MODEL.gguf SAMPLES.jsonl ROUTES.jsonl\n";
            return 1;
        }
        std::ifstream samples(argv[2]);
        if (!samples) throw std::runtime_error("cannot open input samples");
        route_trace trace{std::ofstream(argv[3])};
        if (!trace.output) throw std::runtime_error("cannot open output trace");
        common_init();
        llama_backend_init();
        common_params params;
        params.model.path = argv[1];
        params.n_gpu_layers = 99;
        const int n_ctx = env_int("LLAMA_MOE_PROFILE_CTX", 2048, 256, 32768);
        const int n_gen = env_int("LLAMA_MOE_PROFILE_NGEN", 64, 0, 512);
        params.n_ctx = n_ctx;
        params.n_batch = params.n_ubatch = 256;
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 8;
        params.warmup = false;
        params.fit_params = false;
        params.cb_eval = capture_routes;
        params.cb_eval_user_data = &trace;
        auto init = common_init_from_params(params);
        auto * ctx = init->context();
        auto * model = init->model();
        if (!ctx || !model) throw std::runtime_error("model initialization failed");
        const auto * vocab = llama_model_get_vocab(model);
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_greedy(), llama_sampler_free);
        size_t count = 0;
        std::string line;
        while (std::getline(samples, line)) {
            if (line.empty()) continue;
            auto sample = nlohmann::json::parse(line);
            trace.sample = sample.at("id").get<std::string>();
            const std::string prompt = sample.at("prompt").get<std::string>();
            auto tokens = common_tokenize(ctx, prompt, true, true);
            if (tokens.empty() || (int) tokens.size() + n_gen > n_ctx) {
                throw std::runtime_error("sample exceeds context budget");
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            trace.phase = "prefill";
            const auto before = trace.records;
            for (size_t pos = 0; pos < tokens.size(); pos += 256) {
                int n = (int) std::min<size_t>(256, tokens.size() - pos);
                if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, n))) {
                    throw std::runtime_error("prefill failed");
                }
            }
            trace.phase = "decode";
            int generated = 0;
            for (; generated < n_gen; ++generated) {
                llama_token token = llama_sampler_sample(sampler.get(), ctx, -1);
                if (llama_vocab_is_eog(vocab, token)) break;
                if (llama_decode(ctx, llama_batch_get_one(&token, 1))) throw std::runtime_error("decode failed");
            }
            if (trace.records == before) throw std::runtime_error("no router tensors captured");
            trace.output.flush();
            std::cerr << "PROFILE sample=" << trace.sample << " prompt=" << tokens.size()
                      << " generated=" << generated << " records=" << trace.records - before << '\n';
            count++;
        }
        if (!count) throw std::runtime_error("empty sample dataset");
        std::cerr << "PROFILE complete samples=" << count << " records=" << trace.records << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "PROFILE ERROR: " << error.what() << '\n';
        return 1;
    }
}
