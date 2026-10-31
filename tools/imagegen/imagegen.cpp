#include "imagegen.h"

#include "stable-diffusion.h"

// Internal sdcpp header: sd_progress_is_uploading() distinguishes weight
// loading from denoising, which share the public progress callback.
#include "core/util.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct imagegen_ctx {
    sd_ctx_t * sd = nullptr;
};

// Kept in a buffer with the preview/cancel context: "the sampler stopped
// early" and "the model blew up" both used to surface as kErrGen.
static std::string     g_last_failure;
static std::atomic<int> g_preview_calls{0};
static std::atomic<int> g_step_calls{0};


namespace {

// Context whose generate_image() call owns the thread that runs sdcpp; the
// step callback uses it to request a cancel.
thread_local imagegen_ctx * g_active_ctx = nullptr;

// Thread-local "currently generating" context: the sdcpp preview/progress
// callbacks are process-global function pointers, so they need a way back to
// the call that installed them.
thread_local const imagegen_gen_params * tls_params = nullptr;
// Which phase the current generate_image() call is in. Set by the wrapper
// itself (not guessed from step numbers): the same process-global callback
// reports text encoding, denoising and tiled VAE decoding with unrelated
// step/total pairs (8 denoising steps vs 426 decode tiles).
thread_local imagegen_stage_t tls_stage = IMAGEGEN_STAGE_LOADING;
// Set at the first preview of the sampling pass, so nothing before it can be
// mistaken for denoising.
thread_local bool tls_sampling_seen = false;
// Set only while imagegen_init() builds the context; the step callback uses it
// to forward weight-loading progress through imagegen_params::on_load_step.
thread_local const imagegen_params * tls_load_params = nullptr;

const char * stage_name(imagegen_stage_t stage) {
    switch (stage) {
        case IMAGEGEN_STAGE_LOADING:  return "loading";
        case IMAGEGEN_STAGE_ENCODING: return "encoding";
        case IMAGEGEN_STAGE_SAMPLING: return "sampling";
        case IMAGEGEN_STAGE_DECODING: return "decoding";
        case IMAGEGEN_STAGE_DONE:     return "done";
    }
    return "?";
}

void imagegen_trace(const char * fmt, ...);  // defined below

// Announces a phase change once, so listeners do not have to infer it.
// The trace line is what makes the phase durations measurable: each boundary is
// written once with the stage name, so a profile can subtract them.
void announce_stage(imagegen_stage_t stage) {
    if (tls_stage == stage) {
        return;
    }
    tls_stage = stage;
    imagegen_trace("PHASE %s", stage_name(stage));
    const imagegen_gen_params * p = tls_params;
    if (p != nullptr && p->on_step != nullptr) {
        p->on_step(0, static_cast<int>(stage), 0, 0, 0.0, p->on_step_user_data);
    }
}
thread_local int64_t tls_last_step_us = 0;

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Optional trace hook a host can install (the Android JNI layer appends to
// pipeline.log through it). Kept dumb on purpose: a null hook costs one branch.
static imagegen_trace_cb g_trace_cb   = nullptr;
static void *            g_trace_data = nullptr;

void imagegen_trace(const char * fmt, ...) {
    if (g_trace_cb == nullptr) {
        return;
    }
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    g_trace_cb(msg, g_trace_data);
}

// Keeps the preview cheap: latent-resolution RGB is blown up by
// preview_latent_video(), a 64 px long side is plenty for a progress thumbnail
// and keeps the JNI copy at ~12 KB.
constexpr int kPreviewMaxSide = 64;

struct DownsampledPreview {
    std::vector<unsigned char> rgb;
    int width  = 0;
    int height = 0;
};

DownsampledPreview downsample_rgb(const sd_image_t & img) {
    DownsampledPreview out;
    if (img.data == nullptr || img.width == 0 || img.height == 0) {
        return out;
    }
    const int src_w    = static_cast<int>(img.width);
    const int src_h    = static_cast<int>(img.height);
    const int channels = img.channel > 0 ? static_cast<int>(img.channel) : 3;
    const int step     = std::max(1, std::max(src_w, src_h) / kPreviewMaxSide);
    out.width          = (src_w + step - 1) / step;
    out.height         = (src_h + step - 1) / step;
    out.rgb.resize(static_cast<size_t>(out.width) * out.height * 3);
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const int sx = std::min(src_w - 1, x * step);
            const int sy = std::min(src_h - 1, y * step);
            const unsigned char * px = img.data + (static_cast<size_t>(sy) * src_w + sx) * channels;
            unsigned char * dst     = out.rgb.data() + (static_cast<size_t>(y) * out.width + x) * 3;
            dst[0] = px[0];
            dst[1] = channels > 1 ? px[1] : px[0];
            dst[2] = channels > 2 ? px[2] : px[0];
        }
    }
    return out;
}

void preview_cb(int step, int frame_count, sd_image_t * frames, bool /*is_noisy*/, void * /*data*/) {
    ++g_preview_calls;
    const sd_preview_info_t info = sd_get_preview_info();
    const int pass               = info.sample_pass;
    // A preview only exists once the denoising pass runs, so this is the most
    // reliable end-of-encoding signal even when nobody wants the frames.
    if (pass == 1 && !tls_sampling_seen) {
        tls_sampling_seen = true;
        imagegen_trace("first preview: encoding -> sampling (pass=%d step=%d/%d)",
                       pass, step, info.total_steps);
        announce_stage(IMAGEGEN_STAGE_SAMPLING);
    }
    const imagegen_gen_params * p = tls_params;
    if (p == nullptr) {
        return;
    }
    // sdcpp's progress callback only fires for steps it considers "complete",
    // which for MOSS is about one in eight - useless for a progress bar. The
    // preview fires on every step, so it reports the step count as well; the
    // Android side drives both the bar and the ETA from it.
    // It honours the same contract as step_cb: false means "stop".
    // sdcpp's own progress callback now fires for every denoising step, so this
    // path no longer has to synthesise one: reporting both made every step
    // arrive twice and skewed the per-step timing.
    (void) 0;
    // Pass 1 only: later passes start from noise again and the thumbnail would
    // jump back to an unreadable blur.
    if (pass != 1 || p->on_preview == nullptr || frame_count <= 0 || frames == nullptr) {
        return;
    }
    const DownsampledPreview small = downsample_rgb(frames[0]);
    if (!small.rgb.empty()) {
        p->on_preview(pass, step, small.width, small.height, small.rgb.data(),
                      p->on_preview_user_data);
    }
}

void step_cb(int step, int steps, float /*secs*/, void * /*data*/) {
    ++g_step_calls;
    const imagegen_gen_params * p = tls_params;
    if (p == nullptr || p->on_step == nullptr) {
        return;
    }
    const int64_t t      = now_us();
    const double elapsed = tls_last_step_us > 0 ? (t - tls_last_step_us) / 1000.0 : 0.0;

    // The wrapper owns the phase; this callback only carries step numbers that
    // are meaningful inside it (denoising steps, or decode tiles). Weight
    // loading reports through the same callback with tensor counts (426 of
    // 426!), so it is tagged as its own stage instead of being mistaken for a
    // sampling step.
    const int pass       = std::max(0, sd_get_preview_info().sample_pass);
    const bool uploading = sd_progress_is_uploading();
    // sdcpp knows when denoising ends and the tiled VAE decode begins; the
    // wrapper only sees the two as one call. Trust sdcpp for that split, and
    // keep the byte-progress flag for the lazily (re)loaded text encoder,
    // which also reports through this channel during a draw.
    // Weight loading during imagegen_init(): forward through the load hook.
    if (tls_load_params != nullptr) {
        if (tls_load_params->on_load_step != nullptr) {
            tls_load_params->on_load_step(std::max(0, sd_get_preview_info().sample_pass),
                                          static_cast<int>(IMAGEGEN_STAGE_LOADING),
                                          step, steps, elapsed, tls_load_params->on_load_step_user_data);
        }
        return;
    }
    const bool decoding = sd_progress_get_stage() == SD_PROGRESS_STAGE_DECODING;
    // The decode announces itself here, on its first tile, because the line
    // below the generate_image() call runs only after the whole decode is
    // done - announcing there made the phase look 11 ms long and dumped all of
    // its real cost into denoising.
    if (decoding) {
        announce_stage(IMAGEGEN_STAGE_DECODING);
    }
    // The real weight load happens in imagegen_init() and is forwarded through
    // on_load_step, so it never reaches this point. By the time a draw runs,
    // "uploading" only means the text encoder is lazily (re)loading its own
    // weights, which happens *during* encoding - reporting that as LOADING made
    // the label flicker loading -> encoding -> loading -> ... and hid the real
    // phase. The phase the wrapper declared (ENCODING from the start of the
    // draw, SAMPLING at the first preview, DECODING at the first tile) is the
    // answer, and `uploading` no longer overrides it.
    //
    // A previous version kept `uploading && tls_stage == LOADING` as a safety
    // net "so the first draw of a process still says loading weights". That arm
    // was unreachable: imagegen_generate() sets tls_stage = ENCODING before
    // calling generate_image(), and LOADING is never announced - the initial
    // weight load is reported through on_load_step instead.
    const int stage = decoding
                          ? static_cast<int>(IMAGEGEN_STAGE_DECODING)
                          : static_cast<int>(tls_stage);
    if (g_trace_cb != nullptr) {
        // Log the stage that is actually reported, not the raw flags. The old
        // line preferred the byte-progress flag, so during the VAE decode -
        // where that flag is also raised while the tile weights load - the log
        // said "UPLOAD" even though the UI was correctly told "decoding". That
        // mismatch is what made "the weights reloading during decode" look like
        // a real phase error and sent this investigation the wrong way.
        imagegen_trace("progress stage=%s pass=%d step=%d/%d upload=%d tls=%d",
                       stage_name(static_cast<imagegen_stage_t>(stage)),
                       pass, step, steps, uploading ? 1 : 0, static_cast<int>(tls_stage));
    }
    // Advance the per-step clock only for events that actually are steps. The
    // wrapper's phase announcements arrive here with step == 0; resetting the
    // clock on those made the first real step report ~1 ms, which was then
    // stored as the per-step time and made every later estimate nonsense.
    if (step > 0) {
        tls_last_step_us = t;
    }
    if (!p->on_step(pass, stage, step, steps, elapsed, p->on_step_user_data)) {
        if (g_active_ctx != nullptr && g_active_ctx->sd != nullptr) {
            sd_cancel_generation(g_active_ctx->sd, SD_CANCEL_ALL);
        }
    }
}

// sdcpp's preview and progress callbacks are process-global, so they are
// installed once and read the per-call parameters from the thread-locals above.
bool install_callbacks() {
    sd_set_preview_callback(preview_cb, PREVIEW_PROJ, /*interval=*/1,
                            /*denoised=*/true, /*noisy=*/false, nullptr);
    sd_set_progress_callback(step_cb, nullptr);
    return true;
}

}  // namespace

extern "C" IMAGEGEN_API void imagegen_set_trace_callback(imagegen_trace_cb cb, void * user_data) {
    g_trace_cb   = cb;
    g_trace_data = user_data;
}

static const char * kErrInit = "failed to create sd context";
static const char * kErrGen  = "generate_image failed";
static const char * kErrSave = "failed to write PNG";

imagegen_ctx * imagegen_init(const imagegen_params * params) {
    if (params == nullptr || params->diffusion_model_path == nullptr) {
        return nullptr;
    }

    sd_ctx_params_t sp;
    sd_ctx_params_init(&sp);

    sp.diffusion_model_path = params->diffusion_model_path;
    sp.llm_path             = params->llm_path;
    sp.vae_path             = params->vae_path;
    sp.tokenizer            = params->tokenizer;

    if (params->n_threads > 0) {
        sp.n_threads = params->n_threads;
    }
    if (params->wtype != nullptr) {
        sp.wtype = str_to_sd_type(params->wtype);
    }
    if (params->backend != nullptr) {
        sp.backend = params->backend;
    }
    sp.enable_mmap                  = params->enable_mmap;
    sp.release_conditioner_after_encode = params->release_te_after_encode;

    // The weights are the first third of a run on this device, and the SDK
    // reports their progress through the same channel it uses for denoising.
    // Arm the hooks before building the context, otherwise the events happen
    // while nothing is listening and the bar starts at "done".
    const imagegen_gen_params * prev_params = tls_params;
    tls_params        = nullptr;   // no generate-level callbacks yet
    tls_load_params   = params;
    install_callbacks();

    sd_ctx_t * sd = new_sd_ctx(&sp);
    tls_load_params = nullptr;
    tls_params      = prev_params;
    if (sd == nullptr) {
        return nullptr;
    }
    auto * ctx = new imagegen_ctx();
    ctx->sd = sd;
    return ctx;
}

void imagegen_free(imagegen_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->sd != nullptr) {
        free_sd_ctx(ctx->sd);
    }
    delete ctx;
}

const char * imagegen_model_version(const imagegen_ctx * ctx) {
    if (ctx == nullptr || ctx->sd == nullptr) {
        return "Unknown";
    }
    return sd_get_model_version_name(ctx->sd);
}

bool imagegen_generate(imagegen_ctx * ctx,
                       const imagegen_gen_params * params,
                       const char * out_png_path,
                       imagegen_progress_cb cb,
                       void * cb_user_data,
                       const char ** err,
                       int64_t * seed_out) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (seed_out != nullptr) {
        *seed_out = -1;
    }
    if (ctx == nullptr || ctx->sd == nullptr || params == nullptr || out_png_path == nullptr) {
        if (err != nullptr) { *err = "invalid arguments"; }
        return false;
    }

    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);

    gp.prompt          = params->prompt != nullptr ? params->prompt : "";
    gp.negative_prompt = params->negative_prompt;
    gp.width           = params->width  > 0 ? params->width  : 512;
    gp.height          = params->height > 0 ? params->height : 512;
    // sdcpp's own resolve_seed() re-seeds rand() from time() for negative
    // values, which makes two requests inside the same second identical.
    // Draw the seed here instead and hand sdcpp an explicit one.
    int64_t seed = params->seed;
    if (seed < 0) {
        std::random_device rd;
        seed = static_cast<int64_t>(rd() & 0x7fffffffULL);
    }
    // sdcpp seeds only the low 32 bits of the RNG, so anything wider would be
    // reported back but never reproducible. Keep it inside 32 bits.
    seed &= 0xffffffffLL;
    gp.seed = seed;

    if (params->steps > 0) {
        gp.sample_params.sample_steps = params->steps;
    }
    if (params->cfg_scale > 0.0f) {
        gp.sample_params.guidance.txt_cfg = params->cfg_scale;
    }
    if (params->vae_tiling) {
        gp.vae_tiling_params.enabled = true;
    }

    // This API version has no per-step callback; the callback is only used to
    // allow the caller to cancel before the (blocking) call.
    if (cb != nullptr && !cb(0, gp.sample_params.sample_steps, cb_user_data)) {
        if (err != nullptr) { *err = "cancelled"; }
        return false;
    }

    // Arm the per-step hooks for this call only.
    g_last_failure.clear();
    g_preview_calls = 0;
    g_step_calls    = 0;
    // Weight loading happens in imagegen_init(); nothing is uploaded during a
    // draw except the lazily (re)loaded text encoder, which the byte-progress
    // flag already tags. Starting at LOADING made a second draw open on
    // "loading weights" with a stale count.
    tls_stage         = IMAGEGEN_STAGE_ENCODING;
    tls_sampling_seen = false;
    // A listener that threw on the previous run could leave the byte-progress
    // flag behind; it would relabel every later denoising step as loading.
    sd_progress_mark_uploading(false);
    const imagegen_gen_params * prev_params = tls_params;
    imagegen_ctx * const prev_ctx          = g_active_ctx;
    tls_params      = params;
    tls_last_step_us = now_us();
    g_active_ctx    = ctx;

    // Text encoding happens inside generate_image() before the first preview;
    // the tiled VAE decode happens inside it after the last one. The wrapper
    // announces ENCODING now, SAMPLING on the first preview, DECODING after
    // the call returns, so every step number lands in the right phase.
    announce_stage(IMAGEGEN_STAGE_ENCODING);
    sd_image_generation_result_t * result = nullptr;
    const bool generated = generate_image(ctx->sd, &gp, &result);
    announce_stage(IMAGEGEN_STAGE_DECODING);
    tls_params           = prev_params;
    g_active_ctx         = prev_ctx;
    tls_stage            = IMAGEGEN_STAGE_DONE;
    if (!generated || result == nullptr || result->count <= 0) {
        if (result != nullptr) {
            free_sd_image_generation_result(result);
        }
        char detail[256];
        char tail[200] = "";
        if (!g_last_failure.empty()) {
            std::snprintf(tail, sizeof(tail), ", %s", g_last_failure.c_str());
        }
        std::snprintf(detail, sizeof(detail),
                      "generate_image failed (steps seen=%d, previews=%d%s)",
                      g_step_calls.load(), g_preview_calls.load(), tail);
        g_last_failure.assign(detail);
        if (err != nullptr) { *err = g_last_failure.c_str(); }
        return false;
    }

    bool ok = false;
    const sd_image_t & img = result->items[0].image;
    if (img.data != nullptr && img.width > 0 && img.height > 0) {
        int comp = img.channel > 0 ? static_cast<int>(img.channel) : 3;
        ok = stbi_write_png(out_png_path,
                            static_cast<int>(img.width),
                            static_cast<int>(img.height),
                            comp,
                            img.data,
                            static_cast<int>(img.width) * comp) != 0;
        if (!ok && err != nullptr) {
            *err = kErrSave;
        }
    }

    free_sd_image_generation_result(result);
    if (ok && seed_out != nullptr) {
        *seed_out = seed;
    }
    return ok;
}
