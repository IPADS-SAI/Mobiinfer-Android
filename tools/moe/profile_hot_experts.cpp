#include "common.h"
#include "chat.h"
#include "ggml-backend.h"
#include "llama.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int env_int(const char * name, int fallback, int lo, int hi) {
    const char * value = std::getenv(name);
    const int parsed = value ? std::atoi(value) : fallback;
    return std::max(lo, std::min(hi, parsed));
}

// A profiling-only scheduler callback: pulling router ids synchronizes the
// backend, so timings from this executable must not be used as benchmarks.
struct route_trace {
    std::ofstream output;
    std::string sample;
    const char * phase = "prefill";
    size_t records = 0;
};

static bool capture_routes(ggml_tensor * tensor, bool ask, void * opaque) {
    const std::string name = ggml_get_name(tensor);
    const std::string prefix = "ffn_moe_topk-";
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    if (ask) return true;
    if (tensor->type != GGML_TYPE_I32) throw std::runtime_error("router ids must be int32");
    auto & trace = *static_cast<route_trace *>(opaque);
    const int layer = std::stoi(name.substr(prefix.size()));
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    auto rows = nlohmann::json::array();
    for (int64_t row = 0; row < tensor->ne[1]; ++row) {
        auto experts = nlohmann::json::array();
        for (int64_t rank = 0; rank < tensor->ne[0]; ++rank) {
            int32_t expert;
            std::memcpy(&expert, bytes.data() + row * tensor->nb[1] + rank * tensor->nb[0], sizeof(expert));
            experts.push_back(expert);
        }
        rows.push_back(std::move(experts));
    }
    trace.output << nlohmann::json({{"sample", trace.sample}, {"phase", trace.phase},
        {"layer", layer}, {"experts", rows}}).dump() << '\n';
    if (!trace.output) throw std::runtime_error("route trace write failed");
    trace.records++;
    return true;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: llama-moe-profile MODEL.gguf SAMPLES.jsonl ROUTES.jsonl\n";
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
        auto templates = common_chat_templates_init(model, "");
        const auto * vocab = llama_model_get_vocab(model);
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_greedy(), llama_sampler_free);
        size_t count = 0;
        std::string line;
        while (std::getline(samples, line)) {
            if (line.empty()) continue;
            auto sample = nlohmann::json::parse(line);
            trace.sample = sample.at("id").get<std::string>();
            common_chat_templates_inputs input;
            common_chat_msg message;
            message.role = "user";
            message.content = sample.at("prompt").get<std::string>();
            input.messages.push_back(message);
            input.add_generation_prompt = true;
            input.use_jinja = true;
            input.enable_thinking = false;
            auto formatted = common_chat_templates_apply(templates.get(), input);
            auto tokens = common_tokenize(ctx, formatted.prompt, true, true);
            if (tokens.empty() || (int) tokens.size() + n_gen > n_ctx) throw std::runtime_error("sample exceeds context budget");
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
