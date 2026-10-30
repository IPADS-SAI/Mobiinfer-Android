#include <android/log.h>
#include <jni.h>
#include <dlfcn.h>
#include <iomanip>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/system_properties.h>
#include <unistd.h>
#include <vector>

#include "logging.h"
#include "chat.h"
#include "nlohmann/json.hpp"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "minicpm47_runtime.h"
#include "minicpm47_memory.h"

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

// Number of Hexagon (HTP) sessions to open, decided once at init(). Two
// sessions (layers split across HTP0 + HTP1) is the dual-NSP layout the 4.7
// tiled-expert MoE path was built for. Detected via FastRPC when possible;
// otherwise a conservative single-session default, with only Snapdragon
// 8 Elite (SM8750) keeping two. Overridable with MOBIINFER_HTP_SESSIONS.
static int g_htp_sessions = 2;

// Minimal copies of the Hexagon SDK fastrpc structs (remote.h) needed to issue
// FASTRPC_GET_DOMAINS. Kept local so this file needs no SDK include path.
namespace htp_probe {
    constexpr int MAX_DOMAIN_NAMELEN = 30;
    constexpr int FASTRPC_GET_DOMAINS = 0;   // system_req_id
    constexpr int FASTRPC_NSP         = 1;   // fastrpc_domain_type
    constexpr int TYPE_POS            = 5;

    struct domain {
        int          id;
        int          instance_id;
        char         name[MAX_DOMAIN_NAMELEN];
        int          type;       // fastrpc_domain_type
        int          status;     // 0 = down, non-zero = up
        unsigned int card;
        unsigned int soc_id;
        unsigned int reserved[10];
    };

    struct domains_info {
        domain *           domains;
        int                max_domains;
        int                num_domains;
        unsigned long long flags;
    };

    struct system_req_payload {
        int          id;
        domains_info sys;
    };

    inline unsigned long long nsp_flag() {
        return (unsigned long long) (FASTRPC_NSP & ((1 << TYPE_POS) - 1));
    }
} // namespace htp_probe

// Guard the local copies against accidental divergence from the SDK ABI
// (verified against remote.h: 96 / 24 / 32 bytes, sys at offset 8).
static_assert(sizeof(htp_probe::domain)           == 96, "fastrpc_domain ABI");
static_assert(sizeof(htp_probe::domains_info)     == 24, "fastrpc_domains_info ABI");
static_assert(sizeof(htp_probe::system_req_payload) == 32, "system_req_payload ABI");

static int detect_htp_sessions(std::string & soc_out) {
    char soc[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.model", soc);
    soc_out = soc;

    const char * override = getenv("MOBIINFER_HTP_SESSIONS");
    if (override && *override) {
        int n = atoi(override);
        if (n >= 1 && n <= 2) return n;
    }

    // Conservative default when we cannot measure: one session. Only the one
    // platform we know brings up a second NSP gets two. Everything else
    // (8 Gen 3 / 8 Gen 5 / unknown) stays at one until proven otherwise.
    auto fallback = [&]() -> int {
        if (strncmp(soc, "SM8750", 6) == 0) {   // Snapdragon 8 Elite (8 Gen 4)
            return 2;
        }
        return 1;
    };

    // Measured path: ask the FastRPC driver how many NSP (CDSP / HTP) domains
    // are up. Available on SM8850 (8 Gen 5, reports one); absent on SM8750.
    void * lib = dlopen("libcdsprpc.so", RTLD_NOW);
    if (!lib) {
        LOGi("libcdsprpc.so not loadable - using fallback HTP sessions");
        return fallback();
    }

    using rsr_t = int (*)(htp_probe::system_req_payload *);
    rsr_t rsr = (rsr_t) dlsym(lib, "remote_system_request");
    if (!rsr) {
        LOGi("remote_system_request symbol absent - using fallback HTP sessions");
        return fallback();
    }

    htp_probe::system_req_payload req = {};
    req.id              = htp_probe::FASTRPC_GET_DOMAINS;
    req.sys.domains     = nullptr;
    req.sys.max_domains = 0;
    req.sys.flags       = htp_probe::nsp_flag();

    if (rsr(&req) != 0 || req.sys.num_domains <= 0) {
        LOGw("FastRPC FASTRPC_GET_DOMAINS query failed - using fallback HTP sessions");
        return fallback();
    }

    const int n = req.sys.num_domains;
    LOGi("FastRPC reports %d NSP domain(s)", n);
    return n >= 2 ? 2 : 1;
}

// Exposed to minicpm47_runtime.cpp so the MoE runtime picks the same number
// of HTP devices as the JNI layer.
extern "C" int mobiinfer_htp_session_count() {
    return g_htp_sessions;
}

// True when the device's HTP needs a reduced layout to fit in the per-session
// mapping window. Overridable with MOBIINFER_LOW_VMEM=0/1 for A/B testing.
extern "C" int mobiinfer_low_vmem_required() {
    const char * override = getenv("MOBIINFER_LOW_VMEM");
    if (override && *override) {
        return *override != '0';
    }

    using get_arch_fn = int (*)(ggml_backend_reg_t);
    get_arch_fn get_arch = nullptr;
    const auto reg = ggml_backend_reg_by_name("HTP");
    if (reg) {
        get_arch = reinterpret_cast<get_arch_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_arch"));
    }
    if (!get_arch) {
        LOGw("Hexagon arch unavailable; using the conservative low-vmem layout");
        return 1;
    }

    const int arch = get_arch(reg);
    LOGi("Hexagon arch v%d%s", arch,
         arch > 0 && arch < 75 ? " (low-vmem layout required)" : "");
    if (arch <= 0) {
        LOGw("Hexagon arch query returned v%d; using the conservative low-vmem layout", arch);
        return 1;
    }
    return arch < 75 ? 1 : 0;
}

// Inference parameters mirror the iOS demo defaults (MTMDParams.swift / mtmd-ios.cpp):
//   nThreads=4, nCtx=4096 (8192 for MiniCPM-V-4.6 to fit video frames),
//   nBatch=2048, temperature=0.7, top_k=0, top_p=1.0, penalty_repeat=1.0,
//   nPredict=100. Keeping Android in lockstep avoids per-platform
//   divergence in generation quality and prefill latency.
constexpr int   N_THREADS               = 4;

constexpr int   DEFAULT_CONTEXT_SIZE    = 4096;
// MiniCPM-V-4.6 video understanding feeds up to 64 frames * ~64 visual
// tokens each into the KV cache; 4096 is not enough.  iOS demo
// (MBHomeViewController+LoadModel.swift) bumps n_ctx to 8192 only on
// V-4.6 load so older / non-vision models keep the cheaper 4096 budget.
constexpr int   V46_CONTEXT_SIZE        = 8192;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 2048;
// Aligned with the model's generation_config.json (do_sample=true,
// temperature=0.7, top_k=0, top_p=1.0, repetition_penalty=1.0). top_k=0 and
// top_p=1.0 effectively disable those filters, so sampling is pure
// temperature-only as the model card recommends.
constexpr float DEFAULT_SAMPLER_TEMP    = 0.7f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static bool                               g_moe = false;
static std::unique_ptr<MiniCPM47Runtime>    g_moe_runtime;
static std::string                        g_experts_path;
static int                                g_batch_size = BATCH_SIZE;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
// Tool schemas (OpenAI-compatible JSON array) that the next system prompt
// should expose to a tool-aware chat template. Filled from Kotlin before the
// system prompt is processed; "pending" is consumed by that call so a later
// prompt without tools does not inherit them.
static std::string                        g_tool_schemas_pending_json;
static std::string                        g_tool_schemas_json;
static common_sampler                   * g_sampler;
static mtmd_context                     * g_ctx_vision;
// MiniCPM-V family version of the loaded mmproj. 0 = not minicpmv / no mmproj.
// 5  = V-4.0
// 6  = o-4.0
// 100045 = o-4.5
// 46/460/461 = V-4.6 (instruct/thinking)
static int                                g_minicpmv_version = 0;

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_resetPerformanceStatsNative(
        JNIEnv *, jobject) {
    if (g_moe_runtime) {
        g_moe_runtime->resetPerformanceStats();
    }
    if (g_context) {
        llama_perf_context_reset(g_context);
    }
}

extern "C"
JNIEXPORT jdoubleArray JNICALL
Java_com_mobiinfer_demo_LlamaEngine_readPerformanceStatsNative(
        JNIEnv *env, jobject) {
    double values[5] = {0.0, 0.0, 0.0, 0.0, -1.0};
    if (g_moe_runtime) {
        const auto perf = g_moe_runtime->performanceStats();
        for (int i = 0; i < 5; ++i) values[i] = perf[i];
    } else if (g_context) {
        const auto perf = llama_perf_context(g_context);
        values[0] = perf.t_p_eval_ms;
        values[1] = perf.n_p_eval;
        values[2] = perf.t_eval_ms;
        values[3] = perf.n_eval;
    }
    jdoubleArray result = env->NewDoubleArray(5);
    if (result) {
        env->SetDoubleArrayRegion(result, 0, 5, values);
    }
    return result;
}

// Most recent slice cap requested by the upper layer.  Persists across
// loadMmproj calls so a user-chosen value survives e.g. an unload/reload
// of the model.  Initial fallback = 9 (MiniCPM-V's built-in upper bound)
// so anything that calls into native before Kotlin has had a chance to
// thread the user-preference value through gets the "high quality"
// default rather than the legacy "no slicing" one.  In practice every
// real call site uses LlamaEngine.loadMmproj(path, n) which passes the
// persisted preference, so this only matters for the very first
// setImageMaxSliceNumsNative call before mmproj is loaded.
static int                                g_image_max_slice_nums = 9;
static int                                g_image_min_tokens = -1;
static int                                g_image_max_tokens = -1;

// Effective n_ctx used to create g_context.  Set by prepare(); reads by
// the few stay-under-context guards scattered through this file.  Kept
// as a global rather than a llama_n_ctx() call so we don't depend on
// llama-side defaults when llama_context_default_params changes.
static int                                g_n_ctx = DEFAULT_CONTEXT_SIZE;
static int                                g_requested_context_size = DEFAULT_CONTEXT_SIZE;

// MiniCPM5 / V-4.6 thinking toggle. Default off so the first reply is
// not a long <think> dump.
static bool                               g_enable_thinking = false;

static void apply_minicpm5_thinking_suffix(std::string &prompt) {
    static const char kAss[] = "<|im_start|>assistant\n";
    static const char kOpen[] = "<think>\n";
    static const char kClosed[] = "<think>\n\n</think>\n\n";
    const size_t assLen = sizeof(kAss) - 1;
    const auto pos = prompt.rfind(kAss);
    if (pos != std::string::npos) {
        const std::string rest = prompt.substr(pos + assLen);
        if (rest.empty() || rest == kOpen || rest == kClosed) {
            prompt.resize(pos);
        }
    }
    prompt += kAss;
    prompt += g_enable_thinking ? kOpen : kClosed;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    llama_log_set(minicpm_android_log_callback, nullptr);
    mtmd_log_set(minicpm_android_log_callback, nullptr);

    try {
        const auto *native_path = env->GetStringUTFChars(nativeLibDir, 0);
        if (!native_path) return; // JVM already has a pending allocation exception.
        const std::string backend_path(native_path);
        env->ReleaseStringUTFChars(nativeLibDir, native_path);
        const auto *path_to_backend = backend_path.c_str();
        LOGi("Loading backends from %s", path_to_backend);
        setenv("ADSP_LIBRARY_PATH", path_to_backend, 1);
        // Register the HTP sessions before the backend's one-time registry
        // initialization. Two sessions on dual-CDSP SoCs (8 Elite / SM8750);
        // one on single-CDSP SoCs (8 Gen 5 / SM8850). Dense models still
        // explicitly select only HTP0.
        std::string soc_model;
        g_htp_sessions = detect_htp_sessions(soc_model);
        setenv("GGML_HEXAGON_DEVICES", g_htp_sessions > 1 ? "2" : "1", 1);
        LOGi("HTP sessions: %d (ro.soc.model=%s)", g_htp_sessions, soc_model.c_str());
        // Default to polling via the dedicated nonblocking DSP queue API.
        // Read before backend registration, which caches this setting once.
        // adb shell setprop debug.mobiinfer.htp_poll 0 disables polling; restart the App.
        char poll_property[PROP_VALUE_MAX] = {0};
        __system_property_get("debug.mobiinfer.htp_poll", poll_property);
        const bool poll = std::strcmp(poll_property, "0") != 0;
        setenv("GGML_HEXAGON_OPPOLL", poll ? "1" : "0", 1);
        LOGi("HTP response wait: %s", poll ? "poll" : "blocking");
        char profile_property[PROP_VALUE_MAX] = {0};
        __system_property_get("debug.mobiinfer.htp_profile", profile_property);
        const bool profile = std::strcmp(profile_property, "1") == 0;
        setenv("GGML_HEXAGON_PROFILE", profile ? "1" : "0", 1);
        setenv("LLAMA_PROFILE_TARGET", profile ? "1" : "0", 1);
        if (profile) {
            // Keep detailed DSP records in common_log's runtime.log instead of
            // flooding logcat. Profiling runs are separate from speed runs.
            common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
            llama_log_set(common_log_default_callback, nullptr);
            LOGi("HTP profiling enabled; detailed records use runtime.log");
        }
        setenv("GGML_HEXAGON_MBUF", "256M,512M,4G", 1);
        LOGi("ADSP_LIBRARY_PATH=%s", getenv("ADSP_LIBRARY_PATH"));
        ggml_backend_load_all_from_path(path_to_backend);

        if (ggml_backend_reg_count() == 0) {
            LOGw("%s: No backends loaded from path, trying ggml_backend_load_all()", __func__);
            ggml_backend_load_all();
        }

        LOGi("%s: Backend count: %zu", __func__, ggml_backend_reg_count());

        llama_backend_init();
        LOGi("Backend initiated; Log handler set.");
    } catch (const std::exception & e) {
        LOGe("Backend initialization failed: %s", e.what());
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    } catch (...) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "Unknown backend initialization error");
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_load(JNIEnv *env, jobject, jstring jmodel_path,
                                                jstring jexperts_path, jint memory_budget_mb, jint context_size) {
    llama_model_params model_params = llama_model_default_params();
    g_requested_context_size = context_size > 0 ? context_size : DEFAULT_CONTEXT_SIZE;
    model_params.n_gpu_layers = 99;

    g_moe = jexperts_path != nullptr;
    g_experts_path.clear();
    g_batch_size = g_moe ? 8 : BATCH_SIZE;
    g_minicpmv_version = 0;
    if (g_moe) {
        const char *path = env->GetStringUTFChars(jexperts_path, nullptr);
        g_experts_path = path;
        env->ReleaseStringUTFChars(jexperts_path, path);
        setenv("LLAMA_MOE_HTP_PACKED_DIR", g_experts_path.c_str(), 1);
        setenv("LLAMA_MOE_DECODE_WINDOW", "64", 1);
        setenv("LLAMA_MOE_WINDOW_MAX_ROWS", "8", 1);
        setenv("LLAMA_MOE_IO_POOL", "1", 1);
        setenv("LLAMA_MOE_IO_THREADS", "4", 1);
        setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1", 1);
        setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "4", 1);
        setenv("LLAMA_MOE_PREFILL_WARM_ROWS", "0", 1);
        model_params.moe_experts_path = g_experts_path.c_str();
        model_params.moe_slot_bank = 64;
        LOGi("MoE expert bundle: %s; using MiniCPM47 runtime", g_experts_path.c_str());
    } else {
        unsetenv("LLAMA_MOE_HTP_PACKED_DIR");
    }
    const bool dual_htp = g_moe && g_htp_sessions > 1;
    ggml_backend_dev_t devices[] = {
        ggml_backend_dev_by_name("HTP0"),
        dual_htp ? ggml_backend_dev_by_name("HTP1") : nullptr,
        nullptr,
    };
    if (!devices[0]) {
        LOGe("Hexagon backend did not expose HTP0");
        return 3;
    }
    if (dual_htp && !devices[1]) {
        LOGe("MiniCPM-V 4.7 tiled experts require HTP0 and HTP1");
        return 3;
    }
    const float tensor_split[] = {1.0f, 1.0f};
    model_params.devices = devices;
    if (dual_htp) {
        // Two physical CDSP cores available: layer-split the MoE model.
        model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        model_params.tensor_split = tensor_split;
    }

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    LOGi("%s: Loading model from: \n%s\n", __func__, model_path);

    FILE *f = fopen(model_path, "rb");
    if (!f) {
        LOGe("%s: Cannot open model file! Permission denied or file not found: %s", __func__, model_path);
        env->ReleaseStringUTFChars(jmodel_path, model_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);
    LOGi("%s: Model file size: %ld bytes (%.2f GB)", __func__, file_size, file_size / (1024.0 * 1024.0 * 1024.0));

    if (g_moe) {
        try {
            g_moe_runtime = std::make_unique<MiniCPM47Runtime>(
                model_path, g_experts_path, memory_budget_mb, g_requested_context_size,
                g_image_min_tokens, g_image_max_tokens);
        } catch (const std::exception & e) {
            LOGe("MiniCPM47 configuration failed: %s", e.what());
            env->ReleaseStringUTFChars(jmodel_path, model_path);
            return 4;
        }
        env->ReleaseStringUTFChars(jmodel_path, model_path);
        return 0; // The CLI core loads target, draft and vision together in prepare().
    }

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        LOGe("%s: llama_model_load_from_file returned null! Check logcat for llama.cpp internal errors.", __func__);
        return 2;
    }
    g_model = model;
    LOGi("%s: Model loaded successfully!", __func__);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_loadMmproj(JNIEnv *env, jobject,
                                                        jstring jmmproj_path,
                                                        jint jimage_max_slice_nums) {
    if (g_moe_runtime) {
        const char *path = env->GetStringUTFChars(jmmproj_path, nullptr);
        g_moe_runtime->setVisionPath(path);
        env->ReleaseStringUTFChars(jmmproj_path, path);
        return 0;
    }
    if (!g_model) {
        LOGe("%s: Base model must be loaded first!", __func__);
        return 1;
    }

    const auto *mmproj_path = env->GetStringUTFChars(jmmproj_path, 0);
    LOGd("%s: Loading mmproj from: \n%s\n", __func__, mmproj_path);

    mtmd_context_params mparams = mtmd_context_params_default();
    char architecture[64] = {};
    llama_model_meta_val_str(g_model, "general.architecture", architecture, sizeof(architecture));
    // Qwen3-VL uses the registered Hexagon GPU device for vision by default.
    mparams.use_gpu = std::string(architecture) == "qwen3vl";
    LOGi("Vision backend request: architecture=%s backend=%s", architecture,
         mparams.use_gpu ? "HTP" : "CPU");
    mparams.print_timings = false;

    g_image_max_slice_nums       = (jint) jimage_max_slice_nums;
    mparams.image_max_slice_nums = g_image_max_slice_nums;

    mparams.n_threads = N_THREADS;
    if (g_image_min_tokens > 0) mparams.image_min_tokens = g_image_min_tokens;
    if (g_image_max_tokens > 0) mparams.image_max_tokens = g_image_max_tokens;

    g_ctx_vision = mtmd_init_from_file(mmproj_path, g_model, mparams);
    env->ReleaseStringUTFChars(jmmproj_path, mmproj_path);

    if (!g_ctx_vision) {
        LOGe("%s: Failed to load mmproj model!", __func__);
        return 2;
    }

    // mtmd_get_minicpmv_version() was a fork-only API that lived in the
    // demo-private mtmd-ios.h.  Upstream master mtmd does not expose the
    // mmproj's MiniCPM-V family version.  The Kotlin layer now sets
    // g_minicpmv_version explicitly via setMinicpmvVersionNative() based
    // on the selected ModelInfo id (4.6 -> 46).  See iOS opt-r1's
    // MBHomeViewController+LoadModel.swift for the symmetric move.

    LOGi("%s: mmproj model loaded successfully! Vision: %s, slice-cap: %d (applied)",
         __func__,
         mtmd_support_vision(g_ctx_vision) ? "yes" : "no",
         g_image_max_slice_nums);
    return 0;
}

// Returns the MiniCPM-V family version of the currently loaded mmproj
// (0 if no mmproj is loaded).  Backed by [setMinicpmvVersionNative]: the
// Kotlin layer infers the version from the selected ModelInfo id (4.6 ->
// 46) and pushes it down after loadMmproj.  Replaces the legacy
// mtmd_get_minicpmv_version() introspection that the fork mtmd-ios.h used
// to expose but upstream master mtmd removed.
extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_getMinicpmvVersionNative(JNIEnv * /*env*/, jobject) {
    return (jint) g_minicpmv_version;
}

// Kotlin-driven setter for the mmproj version (see comment on
// getMinicpmvVersionNative).  Affects prepare()'s n_ctx selection and
// the assistant-turn prefix logic; should be called immediately after
// loadMmproj() succeeds.  Passing 0 disables both branches.
extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_setMinicpmvVersionNative(JNIEnv * /*env*/,
                                                                      jobject,
                                                                      jint jversion) {
    g_minicpmv_version = (int) jversion;
    LOGi("%s: minicpmv_version set to %d", __func__, g_minicpmv_version);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_setImageMaxSliceNumsNative(JNIEnv * /*env*/,
                                                                        jobject,
                                                                        jint jn) {
    g_image_max_slice_nums = (int) jn;
    if (g_ctx_vision) {
        mtmd_set_image_max_slice_nums(g_ctx_vision, g_image_max_slice_nums);
        LOGi("%s: image_max_slice_nums set to %d (applied)", __func__, g_image_max_slice_nums);
    } else {
        LOGi("%s: image_max_slice_nums set to %d (stored)", __func__, g_image_max_slice_nums);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_setEnableThinkingNative(JNIEnv * /*env*/,
                                                                     jobject,
                                                                     jboolean jenable) {
    g_enable_thinking = (bool) jenable;
    if (g_moe_runtime) g_moe_runtime->setThinking(g_enable_thinking);
    LOGi("%s: enable_thinking=%d", __func__, (int) g_enable_thinking);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_setImageTokenLimitsNative(JNIEnv * /*env*/,
                                                              jobject,
                                                              jint min_tokens,
                                                              jint max_tokens) {
    g_image_min_tokens = (int) min_tokens;
    g_image_max_tokens = (int) max_tokens;
    LOGi("%s: image_tokens=%d..%d", __func__, g_image_min_tokens, g_image_max_tokens);
}

static llama_context *init_context(llama_model *model, const int n_ctx = DEFAULT_CONTEXT_SIZE) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    LOGi("%s: Using %d threads (aligned with iOS demo MTMDParams)", __func__, N_THREADS);

    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, n_ctx);
    }
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = g_batch_size;
    ctx_params.n_ubatch = g_batch_size;
    ctx_params.n_threads = N_THREADS;
    ctx_params.n_threads_batch = N_THREADS;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static common_sampler *new_sampler(float temp) {
    // Match the model's generation_config defaults: pure temperature sampling
    // with top_k / top_p disabled and no repetition penalty. Keep this in
    // lockstep with mtmd-ios.cpp so iOS and Android produce identical
    // distributions for a given seed.
    common_params_sampling sparams;
    sparams.temp = temp;
    sparams.top_k = 0;            // disabled
    sparams.top_p = 1.0f;         // disabled
    sparams.penalty_repeat = 1.0f; // disabled
    return common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_prepare(JNIEnv * /*env*/, jobject /*unused*/) {
    if (g_moe_runtime) {
        try {
            return g_moe_runtime->start() ? 0 : 1;
        } catch (const std::exception & e) {
            LOGe("MiniCPM47 startup failed: %s", e.what());
            return 1;
        }
    }
    // MiniCPM-V-4.6 video understanding (up to 64 frames per turn) needs
    // a larger KV budget than the 4096 default; everything else stays at
    // 4096 to keep memory pressure low on older / non-vision models.
    // Matches the iOS demo's MTMDParams.nCtx = 8192 on V46MultiModel.
    const bool is_v46 = (g_minicpmv_version == 46) ||
                        (g_minicpmv_version == 460) ||
                        (g_minicpmv_version == 461);
    const int  n_ctx  = g_requested_context_size > 0
                      ? g_requested_context_size
                      : (g_moe ? 2048 : (is_v46 ? V46_CONTEXT_SIZE : DEFAULT_CONTEXT_SIZE));
    LOGi("%s: minicpmv_version=%d -> n_ctx=%d", __func__, g_minicpmv_version, n_ctx);

    auto *context = init_context(g_model, n_ctx);
    if (!context) { return 1; }
    g_context = context;
    g_n_ctx = n_ctx;
    g_batch = llama_batch_init(g_batch_size, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler(DEFAULT_SAMPLER_TEMP);
    return 0;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_mobiinfer_demo_LlamaEngine_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;
static llama_pos generation_start_position;
static bool g_image_prefilled = false;
static bool g_vision_mode = false;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;
    g_image_prefilled = false;
    g_vision_mode = false;

    if (clear_kv_cache && g_context)
        llama_memory_clear(llama_get_memory(g_context), false);
}

// Mirror of iOS MBMtmd.mm prefill_text role="user":
// MiniCPM-V-4.6 (46/460/461) follows g_enable_thinking, same as MiniCPM5.
// v4.0 / v2.x use plain ChatML and have no thinking toggle.
static const char * assistant_turn_prefix() {
    switch (g_minicpmv_version) {
        case 46:
        case 460:
        case 461:
        case 47:
            return g_enable_thinking
                ? "<|im_start|>assistant\n<think>\n"
                : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        default:
            return "<|im_start|>assistant\n";
    }
}

static void shift_context() {
    int n_discard = (current_position - system_prompt_position) / 2;
    // The system block (which carries the tools when a template renders them)
    // must survive as long as possible: only drop conversation history.
    if (n_discard < 1 && current_position > system_prompt_position + 1) {
        n_discard = 1;
    }
    if (n_discard < 1) {
        LOGw("%s: nothing left to discard (position=%d, system ends at %d)",
             __func__, current_position, system_prompt_position);
        return;
    }
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content,
                                       bool exclude_from_template = false) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;

    // Tool schemas are rendered only by common_chat_templates_apply(), so a
    // tool-aware system turn is formatted through the full apply path. The
    // formatter derives the whole prompt from the message list, so the new
    // message must be part of it.
    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = role == ROLE_USER;
    inputs.use_jinja = true;
    // Same toggle as assistant_turn_prefix()/the thinking suffix: the
    // template must agree with the assistant prefix we decode against.
    inputs.enable_thinking = g_enable_thinking;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    for (const auto & msg : chat_msgs) {
        inputs.messages.push_back(msg);
    }
    if (!exclude_from_template) {
        inputs.messages.push_back(new_msg);
    }

    std::string formatted;
    const bool tools = role == ROLE_SYSTEM && !g_tool_schemas_pending_json.empty();
    if (tools) {
        try {
            const auto parsed = nlohmann::ordered_json::parse(g_tool_schemas_pending_json);
            inputs.tools = common_chat_tools_parse_oaicompat(parsed);
        } catch (const std::exception & e) {
            LOGe("%s: bad tool schema JSON: %s", __func__, e.what());
        }
    }
    if (role == ROLE_SYSTEM) {
        // The apply path returns the prompt plus parser metadata; only the
        // text is decoded here (the tool call is parsed from the output).
        const auto applied = common_chat_templates_apply(g_chat_templates.get(), inputs);
        formatted = applied.prompt;
        LOGi("%s: system turn rendered with %zu tool(s); prompt starts with: %.160s",
             __func__, inputs.tools.size(), formatted.c_str());
        if (!inputs.tools.empty() && formatted.find("\"generate_image\"") == std::string::npos) {
            LOGe("%s: tool schema was passed but the system block has no generate_image "
                 "definition - this model's template does not render tools; the text "
                 "instructions are the only guidance", __func__);
        }
    } else {
        formatted = common_chat_format_single(
                g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, true);
    }
    chat_msgs.push_back(new_msg);
    LOGi("%s: Formatted and added %s message: \n%s\n", __func__, role.c_str(), formatted.c_str());
    return formatted;
}

static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += g_batch_size) {
        const int cur_batch_size = std::min((int) tokens.size() - i, g_batch_size);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        if (start_pos + i + cur_batch_size >= g_n_ctx - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

static int apply_system_prompt(JNIEnv *env, jstring jsystem_prompt, bool keep_state) {
    if (g_moe_runtime) {
        const char *text = env->GetStringUTFChars(jsystem_prompt, nullptr);
        g_moe_runtime->setSystemPrompt(text);
        env->ReleaseStringUTFChars(jsystem_prompt, text);
        return 0;
    }
    if (!g_context || !g_chat_templates) {
        LOGe("%s: engine not ready (context=%p templates=%p); refusing system prompt",
             __func__, (void *) g_context, (void *) g_chat_templates.get());
        return 3;
    }
    if (!keep_state) {
        reset_short_term_states();
    }

    const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOGd("%s: System prompt received: \n%s", __func__, system_prompt);
    std::string formatted_system_prompt(system_prompt);

    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        // chat_msgs must hold exactly this system turn: the formatter derives
        // the prompt from the message list, and a stale (or empty) list would
        // drop the system prompt entirely.
        chat_msgs.clear();
        std::string text = system_prompt;
        if (!g_tool_schemas_json.empty()) {
            // Tool-aware template: if it renders `tools`, the tool schema is
            // part of the system block and the text that only described the
            // calling convention is redundant. The trigger itself stays: it
            // is a tool, not a template feature.
            const char *marker = "## Tool:";
            const size_t at = text.find(marker);
            if (at != std::string::npos) {
                text.resize(at);
                while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
                    text.pop_back();
                }
            }
        }
        g_tool_schemas_json.swap(g_tool_schemas_pending_json);
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, text);
    }
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);

    if (g_ctx_vision) {
        mtmd_input_text input_text;
        input_text.text = formatted_system_prompt.c_str();
        input_text.add_special = current_position == 0;
        input_text.parse_special = true;

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &input_text, nullptr, 0);
        if (res != 0) {
            LOGe("%s: mtmd_tokenize failed with code %d", __func__, res);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        llama_pos new_n_past;
        if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                    current_position, 0, g_batch_size,
                                    true, &new_n_past)) {
            LOGe("%s: mtmd_helper_eval_chunks failed!", __func__);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        current_position = new_n_past;
        generation_start_position = current_position;
        mtmd_input_chunks_free(chunks);
    } else {
        const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                                   current_position == 0, true);
        for (auto id: system_tokens) {
            LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
        }

        const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
        if ((int) system_tokens.size() > max_batch_size) {
            LOGe("%s: System prompt too long for context! %d tokens, max: %d",
                 __func__, (int) system_tokens.size(), max_batch_size);
            return 1;
        }

        if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position, true)) {
            LOGe("%s: llama_decode() failed!", __func__);
            return 2;
        }

        current_position += (int) system_tokens.size();
    }

    system_prompt_position = current_position;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    return apply_system_prompt(env, jsystem_prompt, /*keep_state=*/false);
}

// Mid-conversation variant used when the generate_image tool is switched on
// or off: it must not throw away the tokens already in the KV cache.
extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_processSystemPromptKeepStateNative(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    return apply_system_prompt(env, jsystem_prompt, /*keep_state=*/true);
}

// Declares the tools the next system prompt should expose, as an
// OpenAI-compatible JSON array of {"type":"function","function":{...}}
// objects ("[]" clears them). Only tool-aware templates render this.
extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_setToolSchemasNative(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jtool_schemas
) {
    g_tool_schemas_pending_json.clear();
    if (jtool_schemas == nullptr) {
        return;
    }
    const char *json = env->GetStringUTFChars(jtool_schemas, nullptr);
    const std::string text(json);
    env->ReleaseStringUTFChars(jtool_schemas, json);
    if (text != "[]" && !text.empty()) {
        g_tool_schemas_pending_json = text;
    }
    LOGi("%s: %zu chars of tool schemas queued for the next system prompt",
         __func__, g_tool_schemas_pending_json.size());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_mobiinfer_demo_LlamaEngine_chatTemplateNative(
        JNIEnv *env,
        jobject /*unused*/
) {
    if (!g_model) {
        return nullptr;
    }
    const char *tmpl = llama_model_chat_template(g_model, /*name=*/nullptr);
    if (tmpl == nullptr) {
        return nullptr;
    }
    return env->NewStringUTF(tmpl);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_prefillImage(
        JNIEnv *env,
        jobject /*unused*/,
        jbyteArray jimage_data,
        jint jimage_size
) {
    if (g_moe_runtime) {
        jbyte *data = env->GetByteArrayElements(jimage_data, nullptr);
        if (!data) return 2;
        try {
            g_moe_runtime->addImage(reinterpret_cast<const uint8_t *>(data), jimage_size);
        } catch (const std::exception & e) {
            env->ReleaseByteArrayElements(jimage_data, data, JNI_ABORT);
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
            return 2;
        }
        env->ReleaseByteArrayElements(jimage_data, data, JNI_ABORT);
        return 0;
    }
    if (!g_ctx_vision) {
        LOGe("%s: mmproj not loaded!", __func__);
        return 1;
    }
    if (!g_context) {
        LOGe("%s: context not initialised!", __func__);
        return 1;
    }

    jbyte *data = env->GetByteArrayElements(jimage_data, nullptr);
    if (!data) {
        LOGe("%s: Failed to get image data from Java", __func__);
        return 2;
    }

    auto *bitmap = mtmd_helper_bitmap_init_from_buf(
        g_ctx_vision,
        reinterpret_cast<const unsigned char *>(data),
        static_cast<size_t>(jimage_size)
    );

    env->ReleaseByteArrayElements(jimage_data, data, JNI_ABORT);

    if (!bitmap) {
        LOGe("%s: Failed to create bitmap from image buffer", __func__);
        return 3;
    }

    mtmd_input_text text;
    // 4.7's image embeddings belong inside the user turn, before its text.
    // Keep the turn open until processUserPrompt appends the question.
    const std::string image_prompt =
        std::string(g_moe && !g_image_prefilled ? "<|im_start|>user\n" : "") + mtmd_default_marker();
    text.text = image_prompt.c_str();
    text.add_special = current_position == 0;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps_cptr[] = { bitmap };
    int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &text,
                                bitmaps_cptr, 1);

    mtmd_bitmap_free(bitmap);

    if (res != 0) {
        LOGe("%s: mtmd_tokenize failed with code %d", __func__, res);
        mtmd_input_chunks_free(chunks);
        return 4;
    }

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                current_position, 0, g_batch_size,
                                false, &new_n_past)) {
        LOGe("%s: mtmd_helper_eval_chunks failed!", __func__);
        mtmd_input_chunks_free(chunks);
        return 5;
    }

    current_position = new_n_past;
    mtmd_input_chunks_free(chunks);
    g_image_prefilled = true;
    g_vision_mode = true;

    LOGi("%s: Image prefilled, current_position: %d", __func__, current_position);
    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_fullReset(JNIEnv *env, jobject) {
    if (g_moe_runtime) {
        try { g_moe_runtime->reset(); }
        catch (const std::exception & e) { env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what()); }
        return;
    }
    if (!g_model || !g_context) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Model is not loaded; reload before resetting the conversation");
        return;
    }
    reset_long_term_states();
    reset_short_term_states();

    common_sampler_free(g_sampler);
    g_sampler = nullptr;
    g_chat_templates.reset();
    g_tool_schemas_json.clear();
    g_tool_schemas_pending_json.clear();
    llama_batch_free(g_batch);
    g_batch = {};
    llama_free(g_context);
    g_context = nullptr;

    auto *context = init_context(g_model, g_n_ctx);
    if (!context) {
        LOGe("%s: Failed to reinitialize context!", __func__);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Failed to recreate model context; reload the model");
        return;
    }
    g_context = context;
    g_batch = llama_batch_init(g_batch_size, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler(DEFAULT_SAMPLER_TEMP);

    LOGi("%s: Full reset complete - context recreated, KV cache fresh, sampler reinitialized", __func__);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_nativeCancelGeneration(JNIEnv *, jobject) {
    if (g_moe_runtime) { g_moe_runtime->cancel(); return; }
    common_sampler_reset(g_sampler);
    assistant_ss.str("");
    assistant_ss.clear();
    cached_token_chars.clear();
    LOGi("%s: Generation cancelled, sampler reset, KV cache preserved at position %d",
         __func__, current_position);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    if (g_moe_runtime) {
        const char *text = env->GetStringUTFChars(juser_prompt, nullptr);
        const std::string prompt(text);
        env->ReleaseStringUTFChars(juser_prompt, text);
        try { g_moe_runtime->submit(prompt, n_predict); }
        catch (const std::exception & e) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
            return 1;
        }
        return 0;
    }
    if (!g_model || !g_context || !g_chat_templates || !g_sampler) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Model is not ready; reload before sending a prompt");
        return 1;
    }
    reset_short_term_states();

    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("%s: User prompt received: \n%s", __func__, user_prompt);

    std::string content_for_format(user_prompt);
    if (content_for_format.empty()) {
        content_for_format = " ";
    }

    std::string formatted_user_prompt;
    if (g_ctx_vision) {
        // No default system prompt: aligned with iOS opt-r1's MBMtmd.mm.
        // An English-only "You are a helpful assistant." was observed to
        // bias MiniCPM-V into answering Chinese queries in English. The
        // reference Python pipeline (`AutoModel.chat(...)` /
        // `apply_chat_template`) does not inject one either when the
        // caller passes only a user message. If a system message is
        // required it must be added through processSystemPrompt()
        // *before* the first processUserPrompt() call.
        if (!(g_moe && g_image_prefilled)) formatted_user_prompt += "<|im_start|>user\n";
        formatted_user_prompt += content_for_format + "<|im_end|>\n";
        formatted_user_prompt += assistant_turn_prefix();

        common_chat_msg new_msg;
        new_msg.role = ROLE_USER;
        new_msg.content = content_for_format;
        chat_msgs.push_back(new_msg);

        LOGi("%s: Formatted user prompt (mtmd, image=%s, minicpmv=%d): \n%s\n",
             __func__,
             g_image_prefilled ? "yes" : "no",
             g_minicpmv_version,
             formatted_user_prompt.c_str());

        g_image_prefilled = false;
    } else {
        const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
        if (has_chat_template) {
            formatted_user_prompt = chat_add_and_format(ROLE_USER, content_for_format.c_str());
        } else {
            formatted_user_prompt = content_for_format;
        }
        apply_minicpm5_thinking_suffix(formatted_user_prompt);
    }
    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    if (formatted_user_prompt.size() >= 8 &&
        formatted_user_prompt.compare(formatted_user_prompt.size() - 8, 8, "<think>\n") == 0) {
        cached_token_chars = "<think>\n";
    }

    if (g_ctx_vision) {
        mtmd_input_text text;
        text.text          = formatted_user_prompt.c_str();
        text.add_special   = current_position == 0;
        text.parse_special = true;

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &text, nullptr, 0);
        if (res != 0) {
            LOGe("%s: mtmd_tokenize failed with code %d", __func__, res);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        llama_pos new_n_past;
        if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                    current_position, 0, g_batch_size,
                                    true, &new_n_past)) {
            LOGe("%s: mtmd_helper_eval_chunks failed!", __func__);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        current_position = new_n_past;
        generation_start_position = current_position;
        mtmd_input_chunks_free(chunks);
    } else {
        auto user_tokens = common_tokenize(g_context, formatted_user_prompt, current_position == 0, true);
        for (auto id: user_tokens) {
            LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
        }

        const int user_prompt_size = (int) user_tokens.size();
        const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
        if (user_prompt_size > max_batch_size) {
            const int skipped_tokens = user_prompt_size - max_batch_size;
            user_tokens.resize(max_batch_size);
            LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
        }

        if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
            LOGe("%s: llama_decode() failed!", __func__);
            return 2;
        }

        current_position += user_prompt_size;
        generation_start_position = current_position;
    }

    stop_generation_position = current_position + n_predict;
    return 0;
}


// ---------------------------------------------------------------------------
// Generic (non-MoE / mtmd) conversation restore.
//
// The MoE runtime keeps an in-memory message list and can swap it wholesale.
// The generic path instead has to replay the saved turns into the KV cache,
// re-encoding images through mtmd, so follow-up turns keep both text and image
// context (otherwise a reopened image chat answers as if the image was never
// there).
// ---------------------------------------------------------------------------

static int generic_restore_prefill_image(const std::vector<uint8_t> & image) {
    if (!g_ctx_vision || image.empty()) {
        return 0;
    }

    auto * bitmap = mtmd_helper_bitmap_init_from_buf(
        g_ctx_vision, image.data(), image.size());
    if (!bitmap) {
        return 3;
    }

    mtmd_input_text text;
    // Open the user turn once, before its first image; keep it open for text.
    const std::string image_prompt =
        std::string(!g_image_prefilled ? "<|im_start|>user\n" : "") + mtmd_default_marker();
    text.text          = image_prompt.c_str();
    text.add_special   = current_position == 0;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps_cptr[] = { bitmap };
    int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &text, bitmaps_cptr, 1);
    mtmd_bitmap_free(bitmap);
    if (res != 0) {
        mtmd_input_chunks_free(chunks);
        return 4;
    }

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                current_position, 0, g_batch_size, false, &new_n_past)) {
        mtmd_input_chunks_free(chunks);
        return 5;
    }
    current_position = new_n_past;
    mtmd_input_chunks_free(chunks);
    g_image_prefilled = true;
    g_vision_mode = true;
    return 0;
}

static int generic_restore_eval_text(const std::string & formatted) {
    if (formatted.empty()) {
        return 0;
    }
    if (g_ctx_vision) {
        mtmd_input_text text;
        text.text          = formatted.c_str();
        text.add_special   = current_position == 0;
        text.parse_special = true;

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &text, nullptr, 0);
        if (res != 0) {
            mtmd_input_chunks_free(chunks);
            return 2;
        }
        llama_pos new_n_past;
        if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                    current_position, 0, g_batch_size, false, &new_n_past)) {
            mtmd_input_chunks_free(chunks);
            return 2;
        }
        current_position = new_n_past;
        mtmd_input_chunks_free(chunks);
    } else {
        auto tokens = common_tokenize(g_context, formatted, current_position == 0, true);
        const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
        if ((int) tokens.size() > max_batch_size) {
            tokens.resize(max_batch_size);
        }
        if (decode_tokens_in_batches(g_context, g_batch, tokens, current_position, false)) {
            return 2;
        }
        current_position += (int) tokens.size();
    }
    return 0;
}

static int generic_restore(const std::vector<RestoreTurn> & turns) {
    if (!g_context) {
        LOGe("%s: context not initialised", __func__);
        return 2;
    }
    if (!g_ctx_vision && !g_chat_templates) {
        LOGe("%s: chat templates not initialised", __func__);
        return 2;
    }
    reset_long_term_states();

    for (const auto & turn : turns) {
        if (turn.role == ROLE_USER) {
            for (const auto & image : turn.images) {
                const int ir = generic_restore_prefill_image(image);
                if (ir != 0) {
                    LOGe("%s: image prefill failed (%d)", __func__, ir);
                    return ir;
                }
            }

            const std::string content = turn.text.empty() ? " " : turn.text;
            std::string formatted;
            if (g_ctx_vision) {
                if (!g_image_prefilled) {
                    formatted += "<|im_start|>user\n";
                }
                formatted += content + "<|im_end|>\n";
                formatted += assistant_turn_prefix();

                common_chat_msg msg;
                msg.role = ROLE_USER;
                msg.content = content;
                chat_msgs.push_back(msg);
                g_image_prefilled = false;
            } else {
                const bool has_tmpl = common_chat_templates_was_explicit(g_chat_templates.get());
                formatted = has_tmpl ? chat_add_and_format(ROLE_USER, content.c_str(), true) : content;
                apply_minicpm5_thinking_suffix(formatted);
            }
            if (generic_restore_eval_text(formatted) != 0) {
                return 2;
            }
        } else if (turn.role == ROLE_ASSISTANT) {
            if (turn.text.empty()) {
                continue;
            }
            std::string formatted;
            if (g_ctx_vision) {
                // The assistant header was already emitted by the user turn; the
                // reply must be closed with <|im_end|> so the turn boundary is
                // complete (the live path decodes the EOG token too).
                formatted = turn.text + "<|im_end|>\n";
                common_chat_msg msg;
                msg.role = ROLE_ASSISTANT;
                msg.content = turn.text;
                chat_msgs.push_back(msg);
            } else {
                const bool has_tmpl = common_chat_templates_was_explicit(g_chat_templates.get());
                formatted = has_tmpl ? chat_add_and_format(ROLE_ASSISTANT, turn.text.c_str(), true) : turn.text;
            }
            if (generic_restore_eval_text(formatted) != 0) {
                return 2;
            }
        } else if (turn.role == ROLE_SYSTEM) {
            std::string formatted = turn.text;
            if (common_chat_templates_was_explicit(g_chat_templates.get())) {
                formatted = chat_add_and_format(ROLE_SYSTEM, turn.text.c_str());
            }
            if (generic_restore_eval_text(formatted) != 0) {
                return 2;
            }
            system_prompt_position = current_position;
        }
    }

    reset_short_term_states();
    g_image_prefilled = false;
    generation_start_position = current_position;
    stop_generation_position = current_position;
    LOGi("%s: replayed %d turns into the generic context, position %d",
         __func__, (int) turns.size(), current_position);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_mobiinfer_demo_LlamaEngine_restoreHistory(
        JNIEnv *env,
        jobject /*unused*/,
        jobjectArray jroles,
        jobjectArray jtexts,
        jintArray jimage_counts,
        jobjectArray jimages
) {
    const jsize turns_n  = env->GetArrayLength(jroles);
    const jsize images_n = jimages ? env->GetArrayLength(jimages) : 0;
    jint * counts = jimage_counts ? env->GetIntArrayElements(jimage_counts, nullptr) : nullptr;

    std::vector<RestoreTurn> turns;
    turns.reserve(turns_n);
    jsize image_cursor = 0;
    for (jsize i = 0; i < turns_n; ++i) {
        RestoreTurn turn;
        auto jr = (jstring) env->GetObjectArrayElement(jroles, i);
        auto jt = (jstring) env->GetObjectArrayElement(jtexts, i);
        if (jr) {
            const char * s = env->GetStringUTFChars(jr, nullptr);
            turn.role = s;
            env->ReleaseStringUTFChars(jr, s);
            env->DeleteLocalRef(jr);
        }
        if (jt) {
            const char * s = env->GetStringUTFChars(jt, nullptr);
            turn.text = s;
            env->ReleaseStringUTFChars(jt, s);
            env->DeleteLocalRef(jt);
        }
        const int n_for_turn = counts ? counts[i] : 0;
        for (int j = 0; j < n_for_turn && image_cursor < images_n; ++j, ++image_cursor) {
            auto jb = (jbyteArray) env->GetObjectArrayElement(jimages, image_cursor);
            if (!jb) continue;
            const jsize len = env->GetArrayLength(jb);
            jbyte * data = env->GetByteArrayElements(jb, nullptr);
            if (data) {
                turn.images.emplace_back(
                    reinterpret_cast<uint8_t *>(data),
                    reinterpret_cast<uint8_t *>(data) + len);
                env->ReleaseByteArrayElements(jb, data, JNI_ABORT);
            }
            env->DeleteLocalRef(jb);
        }
        turns.push_back(std::move(turn));
    }
    if (counts) env->ReleaseIntArrayElements(jimage_counts, counts, JNI_ABORT);

    if (g_moe_runtime) {
        try {
            g_moe_runtime->restore(turns);
        } catch (const std::exception & e) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
            return 2;
        }
    } else {
        int restore_rc = 0;
        try {
            restore_rc = generic_restore(turns);
        } catch (const std::exception & e) {
            // A C++ exception must never cross the JNI boundary (std::terminate).
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
            return 2;
        }
        if (restore_rc != 0) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "Failed to restore conversation for this model");
            return 3;
        }
    }
    LOGi("%s: restored %d turns", __func__, (int) turns.size());
    return 0;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mobiinfer_demo_LlamaEngine_wasContextTruncatedNative(
        JNIEnv * /*env*/,
        jobject /*unused*/
) {
    return g_moe_runtime && g_moe_runtime->contextTruncated() ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_mobiinfer_demo_LlamaEngine_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    if (g_moe_runtime) {
        try {
            bool done;
            const auto chunk = g_moe_runtime->next(done);
            return done ? nullptr : env->NewStringUTF(chunk.c_str());
        } catch (const std::exception & e) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
            return nullptr;
        }
    }
    if (!g_context || !g_model || !g_sampler || !g_batch.token) {
        LOGe("%s: engine not ready (context=%p model=%p sampler=%p); stopping",
             __func__, (void *) g_context, (void *) g_model, (void *) g_sampler);
        return nullptr;
    }
    if (current_position >= g_n_ctx - OVERFLOW_HEADROOM) {
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    if (current_position >= stop_generation_position) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        return nullptr;
    }

    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    const bool is_eog = llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id);

    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token", __func__);
        return nullptr;
    }

    current_position++;

    if (is_eog) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        std::string assistant_content = assistant_ss.str();
        auto think_end = assistant_content.find("</think>");
        if (think_end != std::string::npos) {
            assistant_content = assistant_content.substr(think_end + 8);
            while (!assistant_content.empty() && assistant_content[0] == '\n') {
                assistant_content.erase(0, 1);
            }
        }
        chat_add_and_format(ROLE_ASSISTANT, assistant_content);
        return nullptr;
    }

    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_unload(JNIEnv * /*env*/, jobject /*unused*/) {
    if (g_moe_runtime) {
        g_moe_runtime.reset();
        g_moe = false;
        g_minicpmv_version = 0;
        g_experts_path.clear();
        return;
    }
    reset_long_term_states();
    reset_short_term_states();

    if (g_ctx_vision) {
        mtmd_free(g_ctx_vision);
        g_ctx_vision = nullptr;
    }

    common_sampler_free(g_sampler);
    g_chat_templates.reset();
    g_tool_schemas_json.clear();
    g_tool_schemas_pending_json.clear();
    llama_batch_free(g_batch);
    g_batch = {};
    llama_free(g_context);
    llama_model_free(g_model);
    g_context = nullptr;
    g_model = nullptr;
    g_sampler = nullptr;
    g_minicpmv_version = 0;
    g_moe = false;
    g_experts_path.clear();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_LlamaEngine_shutdown(JNIEnv *, jobject /*unused*/) {
    llama_backend_free();
}
