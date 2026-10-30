#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifdef GGML_USE_CUDA
static bool check_output(
        const char * name,
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float tolerance) {
    for (size_t i = 0; i < actual.size(); ++i) {
        const float limit = tolerance * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > limit) {
            std::fprintf(
                stderr, "%s: value %zu is %g, expected %g (tolerance %g)\n",
                name, i, actual[i], expected[i], limit);
            return false;
        }
    }
    return true;
}

static bool test_type(ggml_backend_t backend, ggml_type type) {
    constexpr int64_t K = 32;
    constexpr int64_t M = 16;
    constexpr int64_t N = 16;

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, K, M);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_input(weight);
    ggml_set_input(input);

    ggml_tensor * native = ggml_mul_mat(ctx, weight, input);
    ggml_tensor * precise = ggml_mul_mat(ctx, weight, input);
    ggml_mul_mat_set_prec(precise, GGML_PREC_F32);
    ggml_set_output(native);
    ggml_set_output(precise);

    if (!ggml_backend_supports_op(backend, native) ||
        !ggml_backend_supports_op(backend, precise)) {
        std::fprintf(stderr, "backend %s does not support raw FP8 matmul\n", ggml_backend_name(backend));
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, native);
    ggml_build_forward_expand(graph, precise);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);

    const uint8_t weight_codes_e4m3[] = { 0x38, 0xb8, 0x30, 0xb0 };
    const uint8_t weight_codes_e5m2[] = { 0x3c, 0xbc, 0x38, 0xb8 };
    const float weight_values[] = { 1.0f, -1.0f, 0.5f, -0.5f };
    const float input_values[] = { -1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 1.0f };
    const uint8_t * weight_codes =
        type == GGML_TYPE_F8_E4M3 ? weight_codes_e4m3 : weight_codes_e5m2;

    std::vector<uint8_t> weight_data(K * M);
    std::vector<float> input_data(K * N);
    std::vector<float> expected(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            weight_data[k + m * K] = weight_codes[(k + 3 * m) % 4];
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            input_data[k + n * K] = input_values[(2 * k + n) % 7];
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            float value = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                value += weight_values[(k + 3 * m) % 4] * input_data[k + n * K];
            }
            expected[m + n * M] = value;
        }
    }

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> native_output(M * N);
    std::vector<float> precise_output(M * N);
    if (computed) {
        ggml_backend_tensor_get(native, native_output.data(), 0, native_output.size() * sizeof(float));
        ggml_backend_tensor_get(precise, precise_output.data(), 0, precise_output.size() * sizeof(float));
    }

    const char * type_name = type == GGML_TYPE_F8_E4M3 ? "E4M3" : "E5M2";
    const bool ok = computed &&
        check_output(type_name, native_output, expected, 1e-4f) &&
        check_output("forced F32 fallback", precise_output, expected, 1e-4f);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}
#endif

int main() {
#ifdef GGML_USE_CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        return 0;
    }
    const bool ok = test_type(backend, GGML_TYPE_F8_E4M3) &&
                    test_type(backend, GGML_TYPE_F8_E5M2);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
#else
    return 0;
#endif
}
