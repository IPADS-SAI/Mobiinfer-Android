#pragma once

// Thin wrapper around the sdcpp-hexagon fork of stable-diffusion.cpp for
// text-to-image generation on Qualcomm Hexagon (HTP/NPU).
//
// This fork vendors its own (patched) ggml with the ggml-hexagon backend, so
// nothing here touches the llama.cpp-omni ggml or model stack. Everything is
// opaque and C-friendly so it can be driven from JNI or another host language.

#include <cstdint>

#ifndef IMAGEGEN_API
#ifdef _WIN32
#define IMAGEGEN_API __declspec(dllexport)
#else
#define IMAGEGEN_API __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct imagegen_ctx;

typedef bool (*imagegen_step_cb)(int sample_pass, int stage, int step, int steps, double elapsed_ms,
                                 void * user_data);
typedef void (*imagegen_preview_cb)(int sample_pass, int step, int width, int height,
                                    const unsigned char * rgb, void * user_data);

struct imagegen_params {
    // MOSS-Image needs several files. All are required for MOSS.
    const char * diffusion_model_path; // DiT, e.g. moss-image-sft-q4k-f16t.gguf (required)
    const char * llm_path;             // text encoder, e.g. qwen3.5-9b-q4km.gguf (required for MOSS)
    const char * vae_path;             // e.g. flux2-vae.safetensors (required)
    const char * tokenizer;            // e.g. tokenizer.json (required)

    int n_threads;                     // <=0 -> physical cores
    const char * wtype;                // weight type string, e.g. "auto"; NULL -> auto
    // Per-module backend placement, e.g. "diffusion=HTP0,te=cpu,vae=HTP0".
    // NULL -> library default (auto). For hexagon the recommended value keeps
    // the text encoder on CPU and runs DiT + VAE on the NPU.
    const char * backend;              // e.g. "diffusion=HTP0,te=cpu,vae=HTP0"
    bool enable_mmap;                  // mmap weights from disk instead of reading
    // Free the TE params+buffers after the prompt is encoded (saves memory,
    // re-loads TE on the next prompt). Useful on memory-tight devices.
    bool release_te_after_encode;

    // Called while the weights are being loaded, so a UI can show the first
    // part of the run instead of a spinner. Same stage/step convention as
    // imagegen_step_cb, always with stage IMAGEGEN_STAGE_LOADING. Optional.
    imagegen_step_cb on_load_step;
    void * on_load_step_user_data;
};

// sample_pass counts sampling batches inside one generate_image() call (a
// refiner or a second guidance pass starts a new one); preview only carries
// frames for pass 1, because a later pass starts from noise again and would
// flash the picture back to a blur.
// stage: which phase of the run produced this event, as imagegen_stage_t
// (declared below). A sampling pass reports its denoising steps (step/steps),
// the VAE decode reports its tile grid (also step/steps) - do not mix them.
typedef enum imagegen_stage_t {
    IMAGEGEN_STAGE_LOADING   = 0,  // weights / context being created
    IMAGEGEN_STAGE_ENCODING  = 1,  // text encoder running
    IMAGEGEN_STAGE_SAMPLING  = 2,  // denoising loop, the "real" steps
    IMAGEGEN_STAGE_DECODING  = 3,  // VAE decode (tiled: one event per tile)
    IMAGEGEN_STAGE_DONE      = 4,
} imagegen_stage_t;

struct imagegen_gen_params {
    const char * prompt;
    const char * negative_prompt;      // NULL -> empty

    int width;                         // <=0 -> 512
    int height;                        // <=0 -> 512
    int steps;                         // <=0 -> model default (8 for MOSS)
    float cfg_scale;                   // <=0 -> model default (3.0 for MOSS)
    // <0 -> pick a random seed in 32-bit range. sdcpp takes the seed as a
    // uint64 (and seeds only the low 32 bits), so a 64-bit value would not be
    // reproducible; callers must pass something < 2^32.
    int64_t seed;

    bool vae_tiling;                   // enable tiled VAE decode to bound peak memory

    // Called once per denoising step (after the preview projection exists):
    // step/steps describe the current sampling pass, elapsed_ms is the time of
    // the step that just finished. Return false to cancel the generation.
    imagegen_step_cb on_step;
    void * on_step_user_data;

    // Called once per step with a downscaled RGB preview of the current
    // latent (PREVIEW_PROJ, no VAE). width*height*3 bytes, needs no other
    // header. Issued right before on_step.
    imagegen_preview_cb on_preview;
    void * on_preview_user_data;
};

// Progress callback: called once before generation starts (step negotiated by
// the library API has no per-step hook); return false to cancel.
typedef bool (*imagegen_progress_cb)(int step, int steps, void * user_data);


// Optional diagnostics sink: receives one line per progress event, so an
// embedder can see which native phase produced which step/steps pair. Pass
// NULL to disable. The callback runs on the generation thread.
typedef void (*imagegen_trace_cb)(const char * line, void * user_data);
IMAGEGEN_API void imagegen_set_trace_callback(imagegen_trace_cb cb, void * user_data);

// Loads the model. Returns NULL on failure.
IMAGEGEN_API imagegen_ctx * imagegen_init(const imagegen_params * params);
IMAGEGEN_API void imagegen_free(imagegen_ctx * ctx);

// Generates one image and writes it as PNG to out_png_path.
// Returns true on success. On failure *err (if non-NULL) receives a short
// message describing the reason.
// seed_out (optional) receives the seed the image was actually drawn with, so
// callers can display it and reproduce the result later.
IMAGEGEN_API bool imagegen_generate(imagegen_ctx * ctx,
                                    const imagegen_gen_params * params,
                                    const char * out_png_path,
                                    imagegen_progress_cb cb,
                                    void * cb_user_data,
                                    const char ** err,
                                    int64_t * seed_out);

// Returns the detected model version name, or "Unknown".
IMAGEGEN_API const char * imagegen_model_version(const imagegen_ctx * ctx);

#ifdef __cplusplus
}
#endif
