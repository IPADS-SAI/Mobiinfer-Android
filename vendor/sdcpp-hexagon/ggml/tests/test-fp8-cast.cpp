#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float fp8_e4m3_to_fp32(uint8_t x) {
    const uint32_t sign     = x >> 7;
    const uint32_t exponent = (x >> 3) & 0x0f;
    const uint32_t mantissa = x & 0x07;

    if (exponent == 0x0f && mantissa == 0x07) {
        return sign ? -NAN : NAN;
    }

    float value;
    if (exponent == 0) {
        value = std::ldexp((float) mantissa, -9);
    } else {
        value = std::ldexp(1.0f + (float) mantissa / 8.0f, (int) exponent - 7);
    }
    return sign ? -value : value;
}

static float fp8_e5m2_to_fp32(uint8_t x) {
    const uint32_t sign     = x >> 7;
    const uint32_t exponent = (x >> 2) & 0x1f;
    const uint32_t mantissa = x & 0x03;

    if (exponent == 0x1f) {
        const float value = mantissa == 0 ? INFINITY : NAN;
        return sign ? -value : value;
    }

    float value;
    if (exponent == 0) {
        value = std::ldexp((float) mantissa, -16);
    } else {
        value = std::ldexp(1.0f + (float) mantissa / 4.0f, (int) exponent - 15);
    }
    return sign ? -value : value;
}

static bool check_result(const char * name, const std::vector<ggml_fp16_t> & actual, float (*decode)(uint8_t)) {
    for (int i = 0; i < 256; ++i) {
        const float expected_f32 = decode((uint8_t) i);
        const float actual_f32   = ggml_fp16_to_fp32(actual[i]);
        if (std::isnan(expected_f32)) {
            if (!std::isnan(actual_f32)) {
                std::fprintf(stderr, "%s: value 0x%02x should be NaN\n", name, i);
                return false;
            }
        } else if (actual[i] != ggml_fp32_to_fp16(expected_f32)) {
            std::fprintf(stderr, "%s: value 0x%02x converted to %g, expected %g\n", name, i, actual_f32, expected_f32);
            return false;
        }
    }
    return true;
}

static bool check_result(const char * name, const std::vector<ggml_bf16_t> & actual, float (*decode)(uint8_t)) {
    for (int i = 0; i < 256; ++i) {
        const float expected_f32 = decode((uint8_t) i);
        const float actual_f32   = ggml_bf16_to_fp32(actual[i]);
        if (std::isnan(expected_f32)) {
            if (!std::isnan(actual_f32)) {
                std::fprintf(stderr, "%s: value 0x%02x should be NaN\n", name, i);
                return false;
            }
        } else if (actual[i].bits != ggml_fp32_to_bf16(expected_f32).bits) {
            std::fprintf(stderr, "%s: value 0x%02x converted to %g, expected %g\n", name, i, actual_f32, expected_f32);
            return false;
        }
    }
    return true;
}

static bool test_backend(ggml_backend_t backend, bool require_bf16) {
    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * e4m3     = ggml_new_tensor_1d(ctx, GGML_TYPE_F8_E4M3, 256);
    ggml_tensor * e5m2     = ggml_new_tensor_1d(ctx, GGML_TYPE_F8_E5M2, 256);
    ggml_set_input(e4m3);
    ggml_set_input(e5m2);
    ggml_tensor * e4m3_f16 = ggml_cast(ctx, e4m3, GGML_TYPE_F16);
    ggml_tensor * e5m2_f16 = ggml_cast(ctx, e5m2, GGML_TYPE_F16);
    ggml_tensor * e4m3_bf16 = ggml_cast(ctx, e4m3, GGML_TYPE_BF16);
    ggml_tensor * e5m2_bf16 = ggml_cast(ctx, e5m2, GGML_TYPE_BF16);

    GGML_ASSERT(ggml_backend_supports_op(backend, e4m3_f16));
    GGML_ASSERT(ggml_backend_supports_op(backend, e5m2_f16));
    const bool supports_bf16 = ggml_backend_supports_op(backend, e4m3_bf16) &&
                               ggml_backend_supports_op(backend, e5m2_bf16);
    if (require_bf16 && !supports_bf16) {
        std::fprintf(stderr, "backend %s does not support FP8 -> BF16\n", ggml_backend_name(backend));
        ggml_free(ctx);
        return false;
    }

    ggml_set_output(e4m3_f16);
    ggml_set_output(e5m2_f16);
    if (supports_bf16) {
        ggml_set_output(e4m3_bf16);
        ggml_set_output(e5m2_bf16);
    }

    GGML_ASSERT(ggml_type_size(GGML_TYPE_F8_E4M3) == 1);
    GGML_ASSERT(ggml_type_size(GGML_TYPE_F8_E5M2) == 1);
    GGML_ASSERT(ggml_nbytes(e4m3) == 256);
    GGML_ASSERT(ggml_nbytes(e5m2) == 256);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, e4m3_f16);
    ggml_build_forward_expand(graph, e5m2_f16);
    if (supports_bf16) {
        ggml_build_forward_expand(graph, e4m3_bf16);
        ggml_build_forward_expand(graph, e5m2_bf16);
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);

    std::vector<uint8_t> input(256);
    for (int i = 0; i < 256; ++i) {
        input[i] = (uint8_t) i;
    }
    ggml_backend_tensor_set(e4m3, input.data(), 0, input.size());
    ggml_backend_tensor_set(e5m2, input.data(), 0, input.size());

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<ggml_fp16_t> e4m3_output(256);
    std::vector<ggml_fp16_t> e5m2_output(256);
    ggml_backend_tensor_get(e4m3_f16, e4m3_output.data(), 0, ggml_nbytes(e4m3_f16));
    ggml_backend_tensor_get(e5m2_f16, e5m2_output.data(), 0, ggml_nbytes(e5m2_f16));

    bool ok = check_result("E4M3 -> F16", e4m3_output, fp8_e4m3_to_fp32) &&
              check_result("E5M2 -> F16", e5m2_output, fp8_e5m2_to_fp32);
    if (supports_bf16) {
        std::vector<ggml_bf16_t> e4m3_bf16_output(256);
        std::vector<ggml_bf16_t> e5m2_bf16_output(256);
        ggml_backend_tensor_get(e4m3_bf16, e4m3_bf16_output.data(), 0, ggml_nbytes(e4m3_bf16));
        ggml_backend_tensor_get(e5m2_bf16, e5m2_bf16_output.data(), 0, ggml_nbytes(e5m2_bf16));
        ok = check_result("E4M3 -> BF16", e4m3_bf16_output, fp8_e4m3_to_fp32) &&
             check_result("E5M2 -> BF16", e5m2_bf16_output, fp8_e5m2_to_fp32) && ok;
    }
    if (!ok) {
        std::fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

static bool run_backend(ggml_backend_t backend, bool require_bf16) {
    if (backend == nullptr) {
        return true;
    }
    const bool ok = test_backend(backend, require_bf16);
    ggml_backend_free(backend);
    return ok;
}

int main() {
    bool ok = run_backend(ggml_backend_cpu_init(), true);
#ifdef GGML_USE_CUDA
    ok = run_backend(ggml_backend_cuda_init(0), true) && ok;
#endif
#ifdef GGML_USE_METAL
    ok = run_backend(ggml_backend_metal_init(), false) && ok;
#endif
#ifdef GGML_USE_VULKAN
    ok = run_backend(ggml_backend_vk_init(0), true) && ok;
#endif
    return ok ? 0 : 1;
}
