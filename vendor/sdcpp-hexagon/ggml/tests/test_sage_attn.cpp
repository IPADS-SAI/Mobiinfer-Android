#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

bool sage_test_poison_shared_memory();

static bool test_attention(ggml_backend_t backend, int d, int nq, int nk, int hq, int hk, int batch, bool zero, float bias, float scale_factor, ggml_sage_attn_mode mode, bool zero_v = false) {
    ggml_context* ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
    ggml_tensor* q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, hq, batch);
    ggml_tensor* k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nk, hk, batch);
    ggml_tensor* v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nk, hk, batch);
    const float scale = scale_factor / std::sqrt(float(d));
    ggml_tensor* out  = ggml_sage_attn(ctx, q, k, v, scale, mode);
    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);
    ggml_set_output(out);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "SageAttention is not supported by this CUDA build/device\n");
        ggml_free(ctx);
        return false;
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 0.6f);
    std::vector<float> qdata(ggml_nelements(q)), kdata(ggml_nelements(k)), vdata(ggml_nelements(v));
    std::vector<ggml_fp16_t> vhalf(vdata.size());
    for (float& x : qdata) {
        x = zero ? 0.0f : normal(rng);
    }
    for (size_t i = 0; i < kdata.size(); ++i) {
        kdata[i] = zero ? 0.0f : normal(rng) + bias * std::sin(float(i % d));
    }
    for (float& x : vdata) {
        x = zero_v ? 0.0f : normal(rng);
    }
    ggml_fp32_to_fp16_row(vdata.data(), vhalf.data(), vdata.size());
    ggml_fp16_to_fp32_row(vhalf.data(), vdata.data(), vdata.size());
    ggml_backend_tensor_set(q, qdata.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kdata.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, vhalf.data(), 0, ggml_nbytes(v));
    bool ok = true;
    // Repeated execution also exercises CUDA graph capture when enabled in the backend.
    for (int repeat = 0; repeat < 3; ++repeat) {
        ok = sage_test_poison_shared_memory() && ok;
        ok = ok && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }
    std::vector<float> actual(ggml_nelements(out));
    ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
    double error = 0.0, norm = 0.0, max_error = 0.0;
    std::vector<double> scores(nk);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < hq; ++h) {
            const int kh = h / (hq / hk);
            for (int t = 0; t < nq; ++t) {
                for (int s = 0; s < nk; ++s) {
                    double dot = 0.0;
                    for (int c = 0; c < d; ++c) {
                        dot += double(qdata[((b * hq + h) * nq + t) * d + c]) *
                               kdata[((b * hk + kh) * nk + s) * d + c];
                    }
                    scores[s] = dot * scale;
                }
                const double maximum = *std::max_element(scores.begin(), scores.end());
                double sum           = 0.0;
                for (double& x : scores) {
                    x = std::exp(x - maximum);
                    sum += x;
                }
                for (int c = 0; c < d; ++c) {
                    double expected = 0.0;
                    for (int s = 0; s < nk; ++s) {
                        expected += scores[s] / sum * vdata[((b * hk + kh) * nk + s) * d + c];
                    }
                    const float result = actual[((b * nq + t) * hq + h) * d + c];
                    ok                 = ok && std::isfinite(result);
                    const double diff  = result - expected;
                    error += diff * diff;
                    norm += expected * expected;
                    max_error = std::max(max_error, std::abs(diff));
                }
            }
        }
    }
    const double relative_error = std::sqrt(error / std::max(norm, 1e-20));
    ok                          = ok && relative_error < 0.05 && max_error < 0.05;
    std::printf("mode=%d D=%d Q=%d K=%d H=%d/%d B=%d zero=%d zero_v=%d bias=%.0f scale=%.2f: rel=%.6f max=%.6f %s\n",
                int(mode), d, nq, nk, hq, hk, batch, zero, zero_v, bias, scale_factor, relative_error, max_error, ok ? "PASS" : "FAIL");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

static bool test_support(ggml_backend_t cuda) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_context* ctx  = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    auto* q            = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 80, 128, 4, 1);
    auto* k            = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 80, 128, 2, 1);
    auto* v            = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 80, 128, 2, 1);
    auto* out          = ggml_sage_attn(ctx, q, k, v, 0.1f, GGML_SAGE_ATTN_AUTO);
    bool ok            = !ggml_backend_supports_op(cuda, out) && !ggml_backend_supports_op(cpu, out);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    std::printf("Unsupported dimension/CPU: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool supports_mode(ggml_backend_t backend, ggml_sage_attn_mode mode) {
    auto* ctx            = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    auto* q              = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 1, 1);
    auto* k              = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 1, 1);
    auto* v              = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 128, 1, 1);
    auto* out            = ggml_sage_attn(ctx, q, k, v, 0.1f, mode);
    const bool supported = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return supported;
}

static bool benchmark_attention(ggml_backend_t backend, int length, bool flash, ggml_sage_attn_mode mode) {
    auto* ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
    auto* q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, length, 48, 1);
    auto* k   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, length, 12, 1);
    auto* v   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, length, 12, 1);
    ggml_tensor* out;
    if (flash) {
        out = ggml_flash_attn_ext(ctx, q, ggml_cast(ctx, k, GGML_TYPE_F16), v, nullptr, 1.f / std::sqrt(128.f), 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    } else {
        out = ggml_sage_attn(ctx, q, k, v, 1.f / std::sqrt(128.f), mode);
    }
    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);
    ggml_set_output(out);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok    = ggml_backend_supports_op(backend, out) && ggml_gallocr_alloc_graph(alloc, graph);
    if (ok) {
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 0.6f);
        for (auto* tensor : {q, k, v}) {
            std::vector<float> values(ggml_nelements(tensor));
            for (float& value : values) {
                value = normal(rng);
            }
            if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(values.size());
                ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
                ggml_backend_tensor_set(tensor, half.data(), 0, ggml_nbytes(tensor));
            } else {
                ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
            }
        }
        for (int i = 0; i < 10; ++i) {
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS && ok;
        }
        std::vector<double> times;
        for (int repeat = 0; repeat < 5; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) {
                ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS && ok;
            }
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 10);
        }
        std::sort(times.begin(), times.end());
        std::printf("BENCH %s mode=%d D=128 N=%d H=48/12: %.4f ms (median of 5 x 10 runs, including preprocessing and dispatch)\n", flash ? "FA" : "Sage", int(mode), length, times[2]);
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        return 77;
    }
    if (!supports_mode(backend, GGML_SAGE_ATTN_AUTO)) {
        std::puts("SKIP: SageAttention requires CUDA 12.0 and SM80 or newer kernels/device");
        ggml_backend_free(backend);
        return 77;
    }
    bool ok          = test_support(backend);
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    for (auto mode : {GGML_SAGE_ATTN_AUTO, GGML_SAGE_ATTN_2, GGML_SAGE_ATTN_2_PLUS_PLUS}) {
        if (!supports_mode(backend, mode)) {
            std::printf("mode=%d: SKIP (not supported by this toolkit/device)\n", int(mode));
            continue;
        }
        for (int d : {64, 128}) {
            ok = test_attention(backend, d, 1, 1, 4, 2, 2, false, 0.0f, 1.0f, mode) && ok;
            ok = test_attention(backend, d, 17, 33, 4, 1, 2, false, 0.0f, 1.0f, mode) && ok;
            ok = test_attention(backend, d, 127, 65, 4, 2, 2, false, 32.0f, 0.7f, mode) && ok;
            ok = test_attention(backend, d, 129, 193, 4, 4, 1, false, 0.0f, 1.0f, mode) && ok;
            ok = test_attention(backend, d, 256, 256, 8, 2, 1, false, 0.0f, 1.0f, mode) && ok;
            ok = test_attention(backend, d, 17, 65, 4, 2, 2, true, 0.0f, 1.0f, mode) && ok;
            ok = test_attention(backend, d, 17, 1042, 4, 2, 1, false, 0.0f, 1.0f, mode) && ok;
            ok = test_attention(backend, d, 17, 33, 4, 2, 2, false, 0.0f, 1.0f, mode, true) && ok;
        }
        if (bench) {
            for (int length : {1042, 4096, 8192}) {
                ok = benchmark_attention(backend, length, false, mode) && ok;
            }
        }
    }
    if (bench) {
        for (int length : {1042, 4096, 8192}) {
            ok = benchmark_attention(backend, length, true, GGML_SAGE_ATTN_AUTO) && ok;
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
