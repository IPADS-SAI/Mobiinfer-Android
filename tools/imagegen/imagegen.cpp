#include "imagegen.h"

#include "stable-diffusion.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct imagegen_ctx {
    sd_ctx_t * sd = nullptr;
};

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

    sd_ctx_t * sd = new_sd_ctx(&sp);
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
                       const char ** err) {
    if (err != nullptr) {
        *err = nullptr;
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
    gp.seed            = params->seed;

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

    sd_image_generation_result_t * result = nullptr;
    if (!generate_image(ctx->sd, &gp, &result) || result == nullptr || result->count <= 0) {
        if (result != nullptr) {
            free_sd_image_generation_result(result);
        }
        if (err != nullptr) { *err = kErrGen; }
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
    return ok;
}
