#include "minicpm47_runtime.h"
#include "minicpm47_memory.h"
#include "logging.h"
#include "server-context.h"
#include "speculative.h"
#include "chat.h"
#include "log.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <mutex>

// Defined in llama_jni.cpp: number of HTP sessions the device exposes
// (1 on single-NSP SoCs, 2 where a second NSP is available).
extern "C" int mobiinfer_htp_session_count();

struct MiniCPM47Runtime::Impl {
    common_params params;
    std::unique_ptr<server_context> server;
    std::unique_ptr<server_response_reader> reader;
    std::thread worker;
    std::mutex failure_mutex;
    std::string worker_failure;

    bool failed() {
        std::lock_guard<std::mutex> lock(failure_mutex);
        return !worker_failure.empty();
    }

    void check_worker() {
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (!worker_failure.empty()) throw std::runtime_error(worker_failure);
    }

    void fail_worker(const char * detail) {
        std::lock_guard<std::mutex> lock(failure_mutex);
        worker_failure = std::string("Inference stopped; reload the model. ") + detail;
        LOGe("MiniCPM47 worker failed: %s", worker_failure.c_str());
    }
    std::vector<common_chat_msg> messages;
    std::vector<raw_buffer> files;
    std::string pending_media;
    std::string response;
    bool thinking = false;
    bool vision_htp = true;
    bool dspark = true;
    bool cache_prompt = false;
    int64_t request_start = 0;
    int64_t first_token = 0;
    std::string directory;
    std::string vision_path;
    bool vision_loaded = false;
    bool benchmark_eager_mmproj = true;
    std::string label;
    int benchmark_max_tokens = 0;
    int memory_budget_mb = 6000;
    int cache_slots = 0;
    bool context_truncated = false;
    std::array<double, 5> last_performance_stats = {0.0, 0.0, 0.0, 0.0, 0.0};

    void finish() {
        if (!response.empty()) {
            common_chat_msg msg;
            msg.role = "assistant";
            msg.content = response;
            messages.push_back(std::move(msg));
        }
        response.clear();
        reader.reset();
    }
};

MiniCPM47Runtime::MiniCPM47Runtime(const std::string & model, const std::string & experts, int memory_budget_mb, int context_size)
    : impl(std::make_unique<Impl>()) {
    auto & p = impl->params;
    const auto memory = minicpm47_memory_config(memory_budget_mb);
    impl->memory_budget_mb = memory.budget_mb;
    impl->cache_slots = memory.cache_slots;
    int prefetch = memory.prefetch;
    int io_threads = 4;
    int benchmark_cache_slots = -1;
    int benchmark_ubatch = memory.ubatch;
    int benchmark_bank = memory.bank;
    const auto dir = model.substr(0, model.find_last_of('/'));
    impl->directory = dir;
    int cfg_draft_n_max = -1;
    int cfg_batch       = 0;
    std::string cfg_layer_windows;
    std::string cfg_prewarm;
    // Optional local benchmark overrides. Normal use defaults to HTP + DSpark.
    std::ifstream config_file(dir + "/runtime.json");
    if (config_file) {
        const auto config = nlohmann::json::parse(config_file);
        impl->vision_htp = config.value("vision_htp", true);
        impl->dspark = config.value("dspark", true);
        impl->label = config.value("label", std::string());
        impl->benchmark_max_tokens = config.value("max_tokens", 0);
        impl->benchmark_eager_mmproj = config.value("eager_mmproj", true);
        cfg_draft_n_max = config.value("draft_n_max", -1);
        cfg_batch       = config.value("batch", 0);
        cfg_layer_windows = config.value("layer_windows", std::string());
        cfg_prewarm     = config.value("prewarm", std::string());
        prefetch = config.value("prefetch", prefetch);
        io_threads = config.value("io_threads", io_threads);
        benchmark_cache_slots = config.value("cache_slots", -1);
        benchmark_ubatch = config.value("ubatch", benchmark_ubatch);
        benchmark_bank = config.value("bank", config.value("moe_slot_bank", benchmark_bank));
        if (benchmark_bank < 8 || benchmark_bank > 256) {
            throw std::invalid_argument("Benchmark bank must be 8..256");
        }
        if (benchmark_ubatch < 1 || benchmark_ubatch > memory.batch) {
            throw std::invalid_argument("Benchmark ubatch must be 1..batch");
        }
        if (benchmark_cache_slots != -1 && (benchmark_cache_slots < 8 || benchmark_cache_slots > 128)) {
            throw std::invalid_argument("Benchmark cache_slots must be 8..128");
        }
        if (prefetch < 0 || prefetch > 16 || io_threads < 1 || io_threads > 8) {
            throw std::invalid_argument("Benchmark prefetch must be 0..16 and io_threads 1..8");
        }
    }
    if (benchmark_bank < 256 && benchmark_ubatch > benchmark_bank / 8) {
        throw std::invalid_argument("MiniCPM47 ubatch exceeds the expert bank capacity");
    }
    if (cfg_batch < 0 || (cfg_batch > 0 && cfg_batch < benchmark_ubatch)) {
        throw std::invalid_argument("Benchmark batch must be >= ubatch");
    }
    // Keep the total decode window within the selected memory budget when
    // comparing prefetch depths. Disabled prefetch returns its slots to cache.
    impl->cache_slots = memory.cache_slots + memory.prefetch - prefetch;
    // Explicit local experiments may exceed the budget-derived slot estimate.
    if (benchmark_cache_slots != -1) {
        impl->cache_slots = benchmark_cache_slots;
    }
    if (impl->cache_slots < 8) {
        throw std::invalid_argument("Benchmark prefetch leaves fewer than 8 expert cache slots");
    }
    p.model.path = model;
    p.moe_experts = experts;
    p.moe_slot_bank = benchmark_bank;
    p.n_ctx = context_size > 0 ? context_size : 4096;
    p.n_batch = cfg_batch > 0 ? cfg_batch : memory.batch;
    p.n_ubatch = benchmark_ubatch;
    p.n_ctx_checkpoints = 1;
    p.n_parallel = 1;
    p.n_gpu_layers = 99;
    // One HTP device on single-NSP SoCs (8 Gen 5 / SM8850), two where the
    // platform brings up a second NSP (8 Elite / SM8750). Kept in sync with
    // llama_jni.cpp's detection via mobiinfer_htp_session_count().
    const int n_htp = mobiinfer_htp_session_count();
    ggml_backend_dev_t htp0 = ggml_backend_dev_by_name("HTP0");
    ggml_backend_dev_t htp1 = n_htp > 1 ? ggml_backend_dev_by_name("HTP1") : nullptr;
    if (!htp0 || (n_htp > 1 && !htp1)) {
        throw std::runtime_error(std::string("MiniCPM-V 4.7 requires HTP0")
                                 + (n_htp > 1 ? " and HTP1" : ""));
    }
    p.devices = {htp0, htp1, nullptr};
    if (n_htp > 1) {
        p.split_mode = LLAMA_SPLIT_MODE_LAYER;
        p.tensor_split[0] = p.tensor_split[1] = 1.0f;
    }
    p.fit_params = false;
    p.warmup = false;
    p.ctx_shift = false;
    p.cpuparams.n_threads = p.cpuparams_batch.n_threads = 4;
    p.mmproj_use_gpu = impl->vision_htp;
    p.image_max_tokens = 64;
    p.enable_reasoning = 0;
    p.sampling.temp = 0.0f;
    p.sampling.seed = 42;
    p.sampling.no_perf = false;
    if (impl->dspark) {
        // Draft path: an explicit runtime.json "draft" wins; otherwise auto-detect
        // a DSpark/DFlash draft sitting next to the main GGUF.  If none is found
        // we disable speculative decoding instead of failing the whole load, so
        // the model still runs (just slower).
        std::string draft_path;
        std::string spec_override;
        std::ifstream draft_cfg(dir + "/runtime.json");
        if (draft_cfg) {
            const auto config = nlohmann::json::parse(draft_cfg);
            draft_path = config.value("draft", std::string());
            spec_override = config.value("speculative_type", std::string());
        }
        if (draft_path.empty()) {
            std::error_code ec;
            for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
                if (!entry.is_regular_file()) continue;
                std::string low = entry.path().filename().string();
                std::transform(low.begin(), low.end(), low.begin(),
                               [](unsigned char c) { return (char) std::tolower(c); });
                if (low.size() > 5 && low.substr(low.size() - 5) == ".gguf" &&
                    (low.find("dspark") != std::string::npos || low.find("dflash") != std::string::npos)) {
                    draft_path = entry.path().filename().string();
                    break;
                }
            }
        }
        if (draft_path.empty()) {
            LOGw("MiniCPM47 runtime: no draft GGUF found in %s; disabling speculative decoding", dir.c_str());
            impl->dspark = false;
        } else {
            if (draft_path.front() != '/') draft_path = dir + "/" + draft_path;

            // Pick the speculative decoder from the draft itself: MiniCPM ships
            // a DSpark draft, Qwen3.5 ships a DFlash draft. Forcing DSpark for
            // both is wrong. An explicit runtime.json "speculative_type"
            // (dflash/dspark/simple/mtp/eagle3) always wins.
            std::string spec_low = spec_override.empty() ? draft_path : spec_override;
            std::transform(spec_low.begin(), spec_low.end(), spec_low.begin(),
                           [](unsigned char c) { return (char) std::tolower(c); });
            common_speculative_type spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
            if (spec_low.find("dflash") != std::string::npos) {
                spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
            } else if (spec_low.find("dspark") != std::string::npos) {
                spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
            } else if (spec_low.find("eagle3") != std::string::npos) {
                spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3;
            } else if (spec_low.find("mtp") != std::string::npos) {
                spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
            } else if (spec_low.find("simple") != std::string::npos) {
                spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE;
            }
            p.speculative.types = {spec_type};
            LOGi("MiniCPM47 runtime: speculative type = %s",
                 common_speculative_type_to_str(spec_type).c_str());
            p.speculative.draft.mparams.path = draft_path;
            p.speculative.draft.n_gpu_layers = 99;
            p.speculative.draft.devices = p.devices;
            // DFlash is fastest at n_max=5; DSpark keeps the memory-config value.
            const int default_n_max = spec_type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH
                ? 5 : memory.draft;
            p.speculative.draft.n_max = cfg_draft_n_max > 0 ? cfg_draft_n_max : default_n_max;
            if ((spec_type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK ||
                 spec_type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) &&
                p.speculative.draft.n_max >= p.n_ubatch) {
                throw std::invalid_argument("Speculative decoding requires ubatch >= draft length + 1");
            }
            p.speculative.draft.cpuparams.n_threads = p.speculative.draft.cpuparams_batch.n_threads = 4;
            LOGi("MiniCPM47 runtime: draft = %s", draft_path.c_str());
        }
    }
    setenv("LLAMA_MOE_DECODE_WINDOW", std::to_string(impl->cache_slots).c_str(), 1);
    // Optional per-layer decode windows ("llama,windows" style: "0:16,1:19,...").
    // Overrides the uniform DECODE_WINDOW above; needed to reallocate the same
    // RAM across layers (dispersed early layers need bigger windows than
    // concentrated late layers).
    if (!cfg_layer_windows.empty()) {
        setenv("LLAMA_MOE_LAYER_WINDOWS", cfg_layer_windows.c_str(), 1);
    } else {
        unsetenv("LLAMA_MOE_LAYER_WINDOWS");
    }
    // Optional static hot-expert table (one layer per line, expert ids).
    // With the persistent-pin patch these stay resident (LRU never evicts them).
    // Relative paths resolve against the model directory.
    if (!cfg_prewarm.empty()) {
        const std::string p = cfg_prewarm.front() == '/'
                ? cfg_prewarm : impl->directory + "/" + cfg_prewarm;
        setenv("LLAMA_MOE_PREWARM", p.c_str(), 1);
    } else {
        unsetenv("LLAMA_MOE_PREWARM");
    }
    setenv("LLAMA_MOE_IO_THREADS", std::to_string(io_threads).c_str(), 1);
    setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH", prefetch > 0 ? "1" : "0", 1);
    setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", std::to_string(prefetch).c_str(), 1);
    LOGi("MiniCPM47 expert configuration: cache=%d prefetch=%d io_threads=%d",
         impl->cache_slots, prefetch, io_threads);
    setenv("LLAMA_MOE_PREFILL_WARM_ROWS", "0", 1);
    setenv("LLAMA_MOE_PREFETCH_MAX_ROWS", "8", 1);
    setenv("LLAMA_MOE_PREFILL_SORT_READS", "1", 1);
    setenv("MB_MEDIA_EMBD_BATCH", "1", 1);
    setenv("LLAMA_DSPARK_MEDIA_FEATURES", "1", 1);
    setenv("LLAMA_DSPARK_TARGET_POSITIONS", "1", 1);
}

MiniCPM47Runtime::~MiniCPM47Runtime() {
    impl->reader.reset();
    if (impl->server) impl->server->terminate();
    if (impl->worker.joinable()) impl->worker.join();
}

void MiniCPM47Runtime::setVisionPath(const std::string & path) {
    impl->vision_path = path;
    if (impl->benchmark_eager_mmproj) impl->params.mmproj.path = path;
}

bool MiniCPM47Runtime::start() {
    common_log_set_file(common_log_main(), (impl->directory + "/runtime.log").c_str());
    common_log_set_timestamps(common_log_main(), true);
    impl->server = std::make_unique<server_context>();
    LOGi("MiniCPM47 mmproj: %s", impl->params.mmproj.path.empty() ? "deferred until image input" : "loading");
    LOGi("MiniCPM47 runtime: vision=%s dspark=%d bank=%d batch=%d ubatch=%d n_max=%d budget=%dMB cache=%d temp=0 seed=42",
         impl->vision_htp ? "HTP" : "CPU", impl->dspark, impl->params.moe_slot_bank,
         impl->params.n_batch, impl->params.n_ubatch, impl->params.speculative.draft.n_max,
         impl->memory_budget_mb, impl->cache_slots);
    if (!impl->server->load_model(impl->params)) {
        if (impl->dspark) {
            LOGw("MiniCPM47 runtime: load_model failed with draft; retrying without speculative decoding");
            impl->dspark = false;
            impl->params.speculative.types.clear();
            impl->params.speculative.draft.mparams.path.clear();
            impl->server = std::make_unique<server_context>();
            if (!impl->server->load_model(impl->params)) return false;
        } else {
            return false;
        }
    }
    impl->vision_loaded = !impl->params.mmproj.path.empty();
    impl->worker = std::thread([this] {
        try {
            impl->server->start_loop();
        } catch (const std::exception & e) {
            impl->fail_worker(e.what());
        } catch (...) {
            impl->fail_worker("Unknown native error");
        }
    });
    return true;
}

void MiniCPM47Runtime::setThinking(bool enabled) { impl->thinking = enabled; }

void MiniCPM47Runtime::setSystemPrompt(const std::string & text) {
    common_chat_msg msg;
    msg.role = "system";
    msg.content = text;
    impl->messages.push_back(std::move(msg));
}

void MiniCPM47Runtime::addImage(const uint8_t * data, size_t size) {
    if (impl->reader) throw std::runtime_error("Cannot attach an image during generation");
    impl->files.emplace_back(data, data + size);
    impl->pending_media += get_media_marker();
}

void MiniCPM47Runtime::submit(const std::string & text, int n_predict) {
    impl->check_worker();
    if (impl->reader) throw std::runtime_error("A completion is already active");
    if (!impl->files.empty() && !impl->vision_loaded) {
        if (impl->vision_path.empty()) {
            throw std::runtime_error("Image input requires an mmproj model");
        }
        std::ifstream projector(impl->vision_path, std::ios::binary);
        if (!projector) {
            throw std::runtime_error("Missing mmproj model: " + impl->vision_path);
        }
        // Multimodal slots, positions and DSpark state are initialized together
        // by server_context. Recreate them before submitting any image tokens.
        // Release the text runtime first so two target/draft models never coexist.
        LOGi("MiniCPM47 first image: switching to vision runtime; preserving chat history");
        impl->server->terminate();
        if (impl->worker.joinable()) impl->worker.join();
        impl->server.reset();
        impl->cache_prompt = false;
        impl->params.mmproj.path = impl->vision_path;
        try {
            if (!start()) throw std::runtime_error("Failed to load vision runtime");
        } catch (const std::exception & e) {
            impl->fail_worker(e.what());
            throw;
        } catch (...) {
            impl->fail_worker("Failed to load vision runtime");
            throw;
        }
    }
    common_chat_msg msg;
    msg.role = "user";
    msg.content = impl->pending_media + text;
    impl->pending_media.clear();
    impl->last_performance_stats = {0.0, 0.0, 0.0, 0.0, 0.0};
    impl->context_truncated = false;
    impl->messages.push_back(std::move(msg));
    const auto meta = impl->server->get_meta();
    common_chat_templates_inputs inputs;
    inputs.messages = impl->messages;
    inputs.use_jinja = meta.chat_params.use_jinja;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = impl->thinking;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    auto formatted = common_chat_templates_apply(meta.chat_params.tmpls.get(), inputs);
    impl->reader = std::make_unique<server_response_reader>(impl->server->get_response_reader());
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = impl->reader->get_new_id();
    task.cli = true;
    task.cli_prompt = formatted.prompt;
    task.cli_files = impl->files;
    task.params.n_predict = impl->benchmark_max_tokens > 0 ? impl->benchmark_max_tokens : n_predict;
    task.params.stream = true;
    task.params.cache_prompt = impl->cache_prompt;
    impl->cache_prompt = true;
    task.params.timings_per_token = true;
    task.params.sampling = impl->params.sampling;
    task.params.speculative = impl->params.speculative;
    task.params.chat_parser_params = common_chat_parser_params(formatted);
    if (!formatted.parser.empty()) task.params.chat_parser_params.parser.load(formatted.parser);
    impl->request_start = ggml_time_ms();
    impl->first_token = 0;
    LOGi("MiniCPM47 request: vision=%s dspark=%d images=%zu max_tokens=%d",
         impl->vision_htp ? "HTP" : "CPU", impl->dspark, impl->files.size(), n_predict);
    impl->reader->post_task(std::move(task));
}

std::string MiniCPM47Runtime::next(bool & done) {
    done = false;
    if (!impl->reader) { done = true; return {}; }
    impl->check_worker();
    while (auto result = impl->reader->next([this] { return impl->failed(); })) {
        if (result->is_error()) {
            const auto error = result->to_json().dump();
            impl->finish();
            throw std::runtime_error(error);
        }
        if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
            std::string chunk;
            for (const auto & diff : partial->oaicompat_msg_diffs) chunk += diff.content_delta;
            if (!chunk.empty()) {
                if (!impl->first_token) {
                    impl->first_token = ggml_time_ms();
                    LOGi("MiniCPM47 first_token_ms=%lld", (long long)(impl->first_token - impl->request_start));
                }
                impl->response += chunk;
                return chunk;
            }
        }
        if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
            impl->last_performance_stats = {
                final->timings.prompt_ms,
                static_cast<double>(final->timings.prompt_n),
                final->timings.predicted_ms,
                static_cast<double>(final->timings.predicted_n),
                final->timings.vision_encode_ms
            };
            impl->context_truncated = final->truncated;
            const int64_t wall = ggml_time_ms() - impl->request_start;
            const int64_t ttft = impl->first_token ? impl->first_token - impl->request_start : -1;
            json record = {
                {"vision_htp", impl->vision_htp}, {"dspark", impl->dspark},
                {"label", impl->label}, {"images", impl->files.size()},
                {"wall_ms", wall}, {"ttft_ms", ttft},
                {"timings", final->timings.to_json()}, {"response", impl->response}
            };
            std::ofstream(impl->directory + "/benchmark.jsonl", std::ios::app) << record.dump() << '\n';
            common_log_flush(common_log_main());
            LOGi("MiniCPM47 BENCH vision=%s dspark=%d wall_ms=%lld ttft_ms=%lld timings=%s",
                 impl->vision_htp ? "HTP" : "CPU", impl->dspark,
                 (long long)(ggml_time_ms() - impl->request_start),
                 (long long)(impl->first_token ? impl->first_token - impl->request_start : -1),
                 final->timings.to_json().dump().c_str());
            impl->finish();
            done = true;
            return {};
        }
    }
    impl->finish();
    impl->check_worker();
    done = true;
    return {};
}

void MiniCPM47Runtime::cancel() { impl->finish(); }

void MiniCPM47Runtime::reset() {
    cancel();
    impl->messages.clear();
    impl->files.clear();
    impl->pending_media.clear();
    // SLOT_ERASE rejects multimodal contexts. The next completion instead starts
    // with n_past=0, which clears both target and draft state on the server thread.
    impl->cache_prompt = false;
}

void MiniCPM47Runtime::restore(const std::vector<RestoreTurn> & turns) {
    cancel();
    impl->messages.clear();
    impl->files.clear();
    impl->pending_media.clear();
    impl->response.clear();
    for (const auto & turn : turns) {
        common_chat_msg msg;
        msg.role = turn.role;
        std::string content = turn.text;
        // Media markers must precede the turn's text, one per attached image,
        // mirroring addImage()/submit(). The server maps markers to cli_files
        // positionally, so the file order must match the marker order.
        for (const auto & image : turn.images) {
            content = std::string(get_media_marker()) + content;
            impl->files.emplace_back(image.begin(), image.end());
        }
        msg.content = std::move(content);
        impl->messages.push_back(std::move(msg));
    }
    // Force the next submit() to evaluate the whole restored history instead
    // of trying to reuse a (now stale) KV prefix.
    impl->cache_prompt = false;
    LOGi("MiniCPM47Runtime::restore: %zu turns, %zu images", turns.size(), impl->files.size());
}

void MiniCPM47Runtime::resetPerformanceStats() {
    impl->last_performance_stats = {0.0, 0.0, 0.0, 0.0, 0.0};
}

std::array<double, 5> MiniCPM47Runtime::performanceStats() const {
    return impl->last_performance_stats;
}

bool MiniCPM47Runtime::contextTruncated() const {
    return impl->context_truncated;
}
