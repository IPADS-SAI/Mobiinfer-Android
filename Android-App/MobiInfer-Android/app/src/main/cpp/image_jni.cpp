// JNI bridge for text-to-image generation via the sdcpp-hexagon fork of
// stable-diffusion.cpp (MOSS-Image on Qualcomm Hexagon / HTP).
//
// This lives in its own shared library (libimagegen.so) and is deliberately
// independent of the llama.cpp-omni chain: sdcpp-hexagon vendors its own
// (patched) ggml with the ggml-hexagon backend which would collide with the
// llama ggml at link time. Kotlin loads "imagegen" separately from "mobiinfer".
//
// Exposes a minimal API to Kotlin (class com.mobiinfer.demo.ImageEngine):
//   - nativeInit(diffusion, llm, vae, tokenizer, nThreads, backend) -> bool
//   - nativeGenerate(prompt, negative, w, h, steps, cfg, seed, outPng) -> seed
//     (the seed actually used, or -1 on failure)
//   - nativeLastError() -> String
//   - nativeModelVersion() -> String
//   - nativeFree()

#include <jni.h>
#include <android/log.h>

#include <cstdarg>
#include <ctime>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "imagegen.h"

#define TAG "image_jni"

// Persistent trail: the DSP driver floods logcat while sd runs, so the
// interesting lines are appended to filesDir/pipeline.log (see
// com.mobiinfer.demo.CriticalLog) and are readable after a process death.
static void crit(const char *fmt, ...) {
    char msg[1024] = {0};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    __android_log_print(ANDROID_LOG_INFO, TAG, "%s", msg);
    const char *path = "/data/user/0/com.mobiinfer.demo/files/pipeline.log";
    FILE *f = fopen(path, "a");
    if (f) {
        // Timestamp in ms since boot: the phase boundaries (weights / encode /
        // denoise / decode) are only visible as these lines, so without a clock
        // the log can say what happened but not how long it took.
        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        fprintf(f, "[image_jni] %lld %s\n",
                static_cast<long long>(ts.tv_sec) * 1000LL + ts.tv_nsec / 1000000LL, msg);
        fclose(f);
    }
}
#define LOG_I(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOG_E(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static imagegen_ctx * g_ctx = nullptr;
static std::string    g_last_error;

// Progress/preview bridge. The callbacks fire on the thread that runs
// imagegen_generate(); they take the listener under a lock, call into it and
// never keep a JNI reference across calls.
static JavaVM *   g_vm                = nullptr;
static jobject    g_progress_listener = nullptr;  // global ref
static std::mutex g_listener_mutex;

static JNIEnv * attach_env() {
    if (g_vm == nullptr) {
        return nullptr;
    }
    JNIEnv * env = nullptr;
    if (g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
        return env;
    }
    return g_vm->AttachCurrentThread(&env, nullptr) == JNI_OK ? env : nullptr;
}

// Forwards one diagnostic line from the wrapper into pipeline.log.
static void trace_trampoline(const char * line, void *) {
    crit("%s", line);
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void *) {
    g_vm = vm;
    imagegen_set_trace_callback(trace_trampoline, nullptr);
    return JNI_VERSION_1_6;
}

static void deliver_preview(int pass, int step, int width, int height, const unsigned char * rgb) {
    std::lock_guard<std::mutex> lock(g_listener_mutex);
    if (g_progress_listener == nullptr) {
        return;
    }
    JNIEnv * env = attach_env();
    if (env == nullptr) {
        return;
    }
    const jsize n      = static_cast<jsize>(width) * height * 3;
    jbyteArray pixels  = env->NewByteArray(n);
    if (pixels != nullptr) {
        env->SetByteArrayRegion(pixels, 0, n, reinterpret_cast<const jbyte *>(rgb));
        jclass cls    = env->GetObjectClass(g_progress_listener);
        // onPreview(samplePass: Int, step: Int, width: Int, height: Int, rgb: ByteArray)
        jmethodID mid = env->GetMethodID(cls, "onPreview", "(IIII[B)V");
        if (mid != nullptr) {
            env->CallVoidMethod(g_progress_listener, mid, pass, step, width, height, pixels);
        }
        env->DeleteLocalRef(pixels);
        env->DeleteLocalRef(cls);
    }
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
}

static bool deliver_step(int pass, int stage, int step, int steps, double elapsed_ms) {
    std::lock_guard<std::mutex> lock(g_listener_mutex);
    if (g_progress_listener == nullptr) {
        return true;
    }
    JNIEnv * env = attach_env();
    if (env == nullptr) {
        return true;
    }
    bool keep_going = true;
    jclass cls      = env->GetObjectClass(g_progress_listener);
    // onStep(samplePass: Int, stage: Int, step: Int, steps: Int, elapsedMs: Double): Boolean
    jmethodID mid = env->GetMethodID(cls, "onStep", "(IIIID)Z");
    if (mid != nullptr) {
        keep_going = env->CallBooleanMethod(g_progress_listener, mid, pass, stage, step, steps, elapsed_ms) == JNI_TRUE;
    }
    env->DeleteLocalRef(cls);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        keep_going = true;
    }
    return keep_going;
}

static void preview_trampoline(int pass, int step, int width, int height,
                               const unsigned char * rgb, void *) {
    deliver_preview(pass, step, width, height, rgb);
}

static bool step_trampoline(int pass, int stage, int step, int steps, double elapsed_ms, void *) {
    return deliver_step(pass, stage, step, steps, elapsed_ms);
}

static std::string jstringToStdString(JNIEnv * env, jstring jStr) {
    if (!jStr) return {};
    const char * chars = env->GetStringUTFChars(jStr, nullptr);
    std::string result(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(jStr, chars);
    return result;
}

extern "C" {

// setenv for the Hexagon runtime (must run before the HTP backend reads env).
JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeSetenv(
        JNIEnv * env, jclass /*clazz*/, jstring key, jstring value) {
    const std::string k = jstringToStdString(env, key);
    const std::string v = jstringToStdString(env, value);
    if (!k.empty()) {
        setenv(k.c_str(), v.c_str(), 1);
        LOG_I("setenv %s=%s", k.c_str(), v.c_str());
    }
}

JNIEXPORT jboolean JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeInit(
        JNIEnv * env, jobject /*thiz*/,
        jstring diffusionModelPath, jstring llmPath, jstring vaePath,
        jstring tokenizerPath, jint nThreads, jstring backend) {
    if (g_ctx != nullptr) {
        imagegen_free(g_ctx);
        g_ctx = nullptr;
    }

    const std::string model = jstringToStdString(env, diffusionModelPath);
    const std::string llm   = jstringToStdString(env, llmPath);
    const std::string vae   = jstringToStdString(env, vaePath);
    const std::string tok   = jstringToStdString(env, tokenizerPath);
    const std::string be    = jstringToStdString(env, backend);

    if (model.empty()) {
        g_last_error = "diffusion model path is empty";
        LOG_E("%s", g_last_error.c_str());
        return JNI_FALSE;
    }

    imagegen_params params{};
    params.diffusion_model_path = model.c_str();
    params.llm_path             = llm.empty() ? nullptr : llm.c_str();
    params.vae_path             = vae.empty() ? nullptr : vae.c_str();
    params.tokenizer            = tok.empty() ? nullptr : tok.c_str();
    params.n_threads            = nThreads;
    params.backend              = be.empty() ? nullptr : be.c_str();
    params.enable_mmap          = true;
    // Weight loading reports through the same listener the draw uses, so the
    // progress bar starts at 0 instead of appearing at "loading done".
    params.on_load_step         = step_trampoline;
    params.on_load_step_user_data = nullptr;

    LOG_I("imagegen init: dit=%s te=%s vae=%s tok=%s threads=%d backend=%s",
          model.c_str(), llm.c_str(), vae.c_str(), tok.c_str(), nThreads, be.c_str());

    g_ctx = imagegen_init(&params);
    if (g_ctx == nullptr) {
        g_last_error = "new_sd_ctx failed";
        LOG_E("imagegen_init returned null");
        return JNI_FALSE;
    }

    LOG_I("imagegen model version: %s", imagegen_model_version(g_ctx));
    return JNI_TRUE;
}

// Returns the seed the image was generated with (>= 0) or -1 on failure; the
// reason is available through nativeLastError().
JNIEXPORT jlong JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeGenerate(
        JNIEnv * env, jobject /*thiz*/,
        jstring prompt, jstring negative,
        jint width, jint height, jint steps, jfloat cfg,
        jlong seed, jstring outPngPath) {
    if (g_ctx == nullptr) {
        g_last_error = "context not initialized";
        return -1;
    }

    const std::string p   = jstringToStdString(env, prompt);
    const std::string n   = jstringToStdString(env, negative);
    const std::string out = jstringToStdString(env, outPngPath);
    if (out.empty()) {
        g_last_error = "output path is empty";
        return -1;
    }

    // Whatever the model wrote ends up here: keep it bounded and one line so
    // the log stays readable (the scene description can be long).
    const std::string brief = p.size() > 120 ? p.substr(0, 120) + "..." : p;

    imagegen_gen_params gp{};
    gp.prompt          = p.c_str();
    gp.negative_prompt = n.empty() ? nullptr : n.c_str();
    gp.width           = width;
    gp.height          = height;
    gp.steps           = steps;
    gp.cfg_scale       = cfg;  // <0 -> model default
    gp.seed            = seed;  // <0 -> imagegen picks and reports one
    gp.vae_tiling      = true;  // bound peak memory on mobile
    gp.on_preview           = preview_trampoline;
    gp.on_preview_user_data = nullptr;
    gp.on_step              = step_trampoline;
    gp.on_step_user_data    = nullptr;

    crit("generate start %dx%d steps=%d cfg=%.1f seed=%lld prompt=\"%.60s\"", width, height, steps, cfg, (long long) seed, brief.c_str());
    LOG_I("imagegen generate: %dx%d steps=%d cfg=%.1f seed=%lld prompt=\"%s\" -> %s",
          width, height, steps, cfg, (long long) seed, brief.c_str(), out.c_str());

    const char * err = nullptr;
    int64_t used_seed = -1;
    const bool ok = imagegen_generate(g_ctx, &gp, out.c_str(), nullptr, nullptr, &err, &used_seed);
    if (!ok) {
        g_last_error = err ? err : "generation failed";
        LOG_E("imagegen_generate failed: %s", g_last_error.c_str());
        return -1;
    }
    crit("generate done seed=%lld", (long long) used_seed);
    LOG_I("imagegen generate done: seed=%lld", (long long) used_seed);
    return static_cast<jlong>(used_seed);
}

JNIEXPORT jstring JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeLastError(JNIEnv * env, jobject /*thiz*/) {
    return env->NewStringUTF(g_last_error.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeModelVersion(JNIEnv * env, jobject /*thiz*/) {
    const char * v = g_ctx != nullptr ? imagegen_model_version(g_ctx) : "Unknown";
    return env->NewStringUTF(v ? v : "Unknown");
}

JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeSetProgressListener(
        JNIEnv * env, jobject /*thiz*/, jobject listener) {
    std::lock_guard<std::mutex> lock(g_listener_mutex);
    if (g_progress_listener != nullptr) {
        env->DeleteGlobalRef(g_progress_listener);
        g_progress_listener = nullptr;
    }
    if (listener != nullptr) {
        g_progress_listener = env->NewGlobalRef(listener);
    }
}

JNIEXPORT void JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeFree(JNIEnv * /*env*/, jobject /*thiz*/) {
    if (g_ctx != nullptr) {
        imagegen_free(g_ctx);
        g_ctx = nullptr;
        LOG_I("imagegen freed");
    }
}

} // extern "C"
