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
//   - nativeGenerate(prompt, negative, w, h, steps, cfg, seed, outPng) -> bool
//   - nativeLastError() -> String
//   - nativeModelVersion() -> String
//   - nativeFree()

#include <jni.h>
#include <android/log.h>

#include <cstdlib>
#include <string>

#include "imagegen.h"

#define TAG "image_jni"
#define LOG_I(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOG_E(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static imagegen_ctx * g_ctx = nullptr;
static std::string    g_last_error;

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

JNIEXPORT jboolean JNICALL
Java_com_mobiinfer_demo_ImageEngine_nativeGenerate(
        JNIEnv * env, jobject /*thiz*/,
        jstring prompt, jstring negative,
        jint width, jint height, jint steps, jfloat cfg,
        jlong seed, jstring outPngPath) {
    if (g_ctx == nullptr) {
        g_last_error = "context not initialized";
        return JNI_FALSE;
    }

    const std::string p   = jstringToStdString(env, prompt);
    const std::string n   = jstringToStdString(env, negative);
    const std::string out = jstringToStdString(env, outPngPath);
    if (out.empty()) {
        g_last_error = "output path is empty";
        return JNI_FALSE;
    }

    imagegen_gen_params gp{};
    gp.prompt          = p.c_str();
    gp.negative_prompt = n.empty() ? nullptr : n.c_str();
    gp.width           = width;
    gp.height          = height;
    gp.steps           = steps;
    gp.cfg_scale       = cfg;
    gp.seed            = seed;
    gp.vae_tiling      = true;  // bound peak memory on mobile

    LOG_I("imagegen generate: %dx%d steps=%d cfg=%.1f seed=%lld -> %s",
          width, height, steps, cfg, (long long) seed, out.c_str());

    const char * err = nullptr;
    const bool ok = imagegen_generate(g_ctx, &gp, out.c_str(), nullptr, nullptr, &err);
    if (!ok) {
        g_last_error = err ? err : "generation failed";
        LOG_E("imagegen_generate failed: %s", g_last_error.c_str());
        return JNI_FALSE;
    }
    return JNI_TRUE;
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
Java_com_mobiinfer_demo_ImageEngine_nativeFree(JNIEnv * /*env*/, jobject /*thiz*/) {
    if (g_ctx != nullptr) {
        imagegen_free(g_ctx);
        g_ctx = nullptr;
        LOG_I("imagegen freed");
    }
}

} // extern "C"
