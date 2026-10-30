#include "sage-attn.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDACC_VER_MAJOR__ >= 12

#include <climits>
#include "convert.cuh"
#include "sageattention/attn.cuh"
#include "sageattention/attn_fp8.cuh"

static constexpr int sage_cuda_version = __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10;

static ggml_sage_attn_mode sage_mode(int device, const ggml_tensor* op) {
    const auto requested = static_cast<ggml_sage_attn_mode>(op->op_params[1]);
    if (requested != GGML_SAGE_ATTN_AUTO) {
        return requested;
    }
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10) >= 12040
    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc >= GGML_CUDA_CC_ADA_LOVELACE && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_ADA_LOVELACE) {
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10) >= 12080
        if (cc != GGML_CUDA_CC_HOPPER) {
            return GGML_SAGE_ATTN_2_PLUS_PLUS;
        }
#endif
        return GGML_SAGE_ATTN_2;
    }
#endif
    return GGML_SAGE_ATTN_AUTO;
}

struct SageWorkspace {
    size_t q;
    size_t k;
    size_t q_scale;
    size_t k_scale;
    size_t k_mean;
    size_t v;
    size_t v_scale;
    size_t output;
    size_t size;

    explicit SageWorkspace(const ggml_tensor* dst) {
        const ggml_tensor* qt = dst->src[0];
        const ggml_tensor* kt = dst->src[1];
        size                  = GGML_PAD(ggml_nbytes(dst), 256);
        q                     = reserve(ggml_nelements(qt));
        k                     = reserve(ggml_nelements(kt));
        q_scale               = reserve(GGML_PAD(qt->ne[1], 128) / 32 * 8 * qt->ne[2] * qt->ne[3] * sizeof(float));
        k_scale               = reserve(GGML_PAD(kt->ne[1], 64) / 64 * 4 * kt->ne[2] * kt->ne[3] * sizeof(float));
        k_mean                = reserve(kt->ne[0] * kt->ne[2] * kt->ne[3] * sizeof(float));
        v                     = reserve(kt->ne[0] * GGML_PAD(kt->ne[1], 64) * kt->ne[2] * kt->ne[3]);
        v_scale               = reserve(kt->ne[0] * kt->ne[2] * kt->ne[3] * sizeof(float));
        output                = reserve(ggml_nelements(dst) * sizeof(half));
    }

    size_t reserve(size_t bytes) {
        const size_t offset = size;
        size += GGML_PAD(bytes, 256);
        return offset;
    }
};

template <int D>
static __global__ void sage_key_mean(const float* k, float* mean, int length) {
    const int channel = blockIdx.x * 32 + threadIdx.x % 32;
    const int row     = threadIdx.x / 32;
    const size_t base = size_t(blockIdx.y) * length * D;
    float sum         = 0.0f;
    for (int token = row; token < length; token += 8) {
        sum += k[base + size_t(token) * D + channel];
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = sum;
    __syncthreads();
    if (row == 0) {
        for (int i = 1; i < 8; ++i) {
            sum += sums[threadIdx.x + i * 32];
        }
        mean[size_t(blockIdx.y) * D + channel] = sum / length;
    }
}

template <int D, int TOKENS, bool SMOOTH>
static __global__ void sage_quantize(const float* input, const float* mean, int8_t* output, float* scales, int length) {
    constexpr int VALUES = D * TOKENS / 256;
    const int head       = blockIdx.y;
    const int first      = blockIdx.x * TOKENS * D;
    const size_t base    = size_t(head) * length * D;
    float values[VALUES];
    float amax = 1e-7f;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = first + threadIdx.x + i * 256;
        float value     = 0.0f;
        if (index < length * D) {
            value = input[base + index];
            if constexpr (SMOOTH) {
                value -= mean[size_t(head) * D + index % D];
            }
        }
        values[i] = value;
        amax      = fmaxf(amax, fabsf(value));
    }
    amax = warp_reduce_max(amax);
    __shared__ float maxima[8];
    if (threadIdx.x % 32 == 0) {
        maxima[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        amax = threadIdx.x < 8 ? maxima[threadIdx.x] : 1e-7f;
        amax = warp_reduce_max(amax);
        if (threadIdx.x == 0) {
            maxima[0]                                     = amax;
            scales[size_t(head) * gridDim.x + blockIdx.x] = amax / 127.0f;
        }
    }
    __syncthreads();
    const float inverse_scale = 127.0f / maxima[0];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = first + threadIdx.x + i * 256;
        if (index < length * D) {
            output[base + index] = int8_t(__float2int_rn(values[i] * inverse_scale));
        }
    }
}

template <int D, bool KEY>
static __global__ void sage_quantize_per_thread(const float* input, const float* mean, int8_t* output, float* scales, int length) {
    constexpr int GROUPS = KEY ? 4 : 8;
    constexpr int TOKENS = KEY ? 64 : 32;
    constexpr int VALUES = D * TOKENS / GROUPS / 32;
    const int group      = threadIdx.x / 32;
    const int lane       = threadIdx.x % 32;
    const size_t base    = size_t(blockIdx.y) * length * D;
    float values[VALUES];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = lane + i * 32;
        const int row   = index / D;
        const int token = blockIdx.x * TOKENS + (KEY ? row / 2 * 8 + group * 2 + row % 2 : row * 8 + group);
        float value     = token < length ? input[base + size_t(token) * D + index % D] : 0.0f;
        if constexpr (KEY) {
            if (token < length) {
                value -= mean[size_t(blockIdx.y) * D + index % D];
            }
        }
        values[i] = value;
        amax      = fmaxf(amax, fabsf(value));
    }
    const float scale = warp_reduce_max(amax) / 127.0f + 1e-7f;
    if (lane == 0) {
        scales[(size_t(blockIdx.y) * gridDim.x + blockIdx.x) * GROUPS + group] = scale;
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = lane + i * 32;
        const int row   = index / D;
        const int token = blockIdx.x * TOKENS + (KEY ? row / 2 * 8 + group * 2 + row % 2 : row * 8 + group);
        if (token < length) {
            const float value                            = values[i] / scale;
            output[base + size_t(token) * D + index % D] = int8_t(__float2int_rz(value + copysignf(0.5f, value)));
        }
    }
}

template <int D>
static __global__ void sage_value_scale(const half* v, float* scales, int length, float scale_max) {
    const int channel = blockIdx.x * 32 + threadIdx.x % 32;
    const int row     = threadIdx.x / 32;
    const size_t base = size_t(blockIdx.y) * length * D;
    float amax        = 1e-7f;
    for (int token = row; token < length; token += 8) {
        amax = fmaxf(amax, fabsf(__half2float(v[base + size_t(token) * D + channel])));
    }
    __shared__ float maxima[256];
    maxima[threadIdx.x] = amax;
    __syncthreads();
    if (row == 0) {
        for (int i = 1; i < 8; ++i) {
            amax = fmaxf(amax, maxima[threadIdx.x + i * 32]);
        }
        scales[size_t(blockIdx.y) * D + channel] = amax / scale_max;
    }
}

template <int D>
static __global__ void sage_quantize_value(const half* v, const float* scales, int8_t* output, int length, int padded) {
    __shared__ uint8_t tile[32][D + 4];
    const int head    = blockIdx.y;
    const int first   = blockIdx.x * 32;
    const size_t base = size_t(head) * length * D;
    for (int i = threadIdx.x; i < 32 * D; i += 256) {
        const int token   = first + i / D;
        const int channel = i % D;
        const float value = token < length ? __half2float(v[base + size_t(token) * D + channel]) / scales[size_t(head) * D + channel] : 0.0f;
        // FP8 MMA consumes pairs of tokens in a permuted, transposed V tile.
        const int row           = i / D;
        const int permuted      = row / 16 * 16 + row % 16 / 8 * 2 + row % 8 / 2 * 4 + row % 2;
        tile[permuted][channel] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * D; i += 256) {
        output[(size_t(head) * D + i / 32) * padded + first + i % 32] = tile[i % 32][i / 32];
    }
}

template <int D, bool PLUS_PLUS>
static void sage_attn_fp8(ggml_backend_cuda_context& ctx, ggml_tensor* dst, const SageWorkspace& work) {
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    const int lq = q->ne[1], lk = k->ne[1], padded = GGML_PAD(lk, 64);
    const int hq = q->ne[2], hk = k->ne[2], batch = q->ne[3];
    char* data                = static_cast<char*>(dst->data);
    auto* qi                  = reinterpret_cast<int8_t*>(data + work.q);
    auto* ki                  = reinterpret_cast<int8_t*>(data + work.k);
    auto* vi                  = reinterpret_cast<int8_t*>(data + work.v);
    auto* qs                  = reinterpret_cast<float*>(data + work.q_scale);
    auto* ks                  = reinterpret_cast<float*>(data + work.k_scale);
    auto* vs                  = reinterpret_cast<float*>(data + work.v_scale);
    auto* km                  = reinterpret_cast<float*>(data + work.k_mean);
    auto* out                 = reinterpret_cast<half*>(data + work.output);
    const cudaStream_t stream = ctx.stream();
    sage_key_mean<D><<<dim3(D / 32, hk * batch), 256, 0, stream>>>(static_cast<const float*>(k->data), km, lk);
    sage_quantize_per_thread<D, false><<<dim3(GGML_PAD(lq, 128) / 32, hq * batch), 256, 0, stream>>>(
        static_cast<const float*>(q->data), nullptr, qi, qs, lq);
    sage_quantize_per_thread<D, true><<<dim3(padded / 64, hk * batch), 128, 0, stream>>>(
        static_cast<const float*>(k->data), km, ki, ks, lk);
    // The smaller V range keeps the FP16 instruction accumulator finite in SageAttention2++.
    sage_value_scale<D><<<dim3(D / 32, hk * batch), 256, 0, stream>>>(static_cast<const half*>(v->data), vs, lk, PLUS_PLUS ? 2.25f : 448.0f);
    sage_quantize_value<D><<<dim3(padded / 32, hk * batch), 256, 0, stream>>>(static_cast<const half*>(v->data), vs, vi, lk, padded);
    auto kernel = sageattention::qk_int_sv_f8_attn_kernel<128, 64, 32, 64, D,
                                                          sageattention::DataType::kInt8, sageattention::QuantGranularity::kPerThread,
                                                          sageattention::QuantGranularity::kPerThread, float, true, half,
                                                          sageattention::ComputeUnit::kCudaCore, sageattention::MaskMode::kNone, false, true, false, PLUS_PLUS>;
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    kernel<<<dim3(GGML_PAD(lq, 128) / 128, hq, batch), dim3(32, 4), 256 * D, stream>>>(
        qi, ki, vi, out, nullptr, qs, ks, vs, nullptr, lq, lk, hq / hk,
        D * lq * hq, D, D * lq, D * lk * hk, D, D * lk,
        D * padded * hk, D * padded, padded, D * lq * hq, D * hq, D, scale);
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(out, static_cast<float*>(dst->data), ggml_nelements(dst), stream);
}

template <int D>
static void sage_attn(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    const int lq = q->ne[1], lk = k->ne[1];
    const int hq = q->ne[2], hk = k->ne[2], batch = q->ne[3];
    const SageWorkspace work(dst);
    const auto mode = sage_mode(ctx.device, dst);
    if (mode == GGML_SAGE_ATTN_2_PLUS_PLUS) {
        sage_attn_fp8<D, true>(ctx, dst, work);
        return;
    }
    if (mode == GGML_SAGE_ATTN_2) {
        sage_attn_fp8<D, false>(ctx, dst, work);
        return;
    }
    char* data                = static_cast<char*>(dst->data);
    auto* qi                  = reinterpret_cast<int8_t*>(data + work.q);
    auto* ki                  = reinterpret_cast<int8_t*>(data + work.k);
    auto* qs                  = reinterpret_cast<float*>(data + work.q_scale);
    auto* ks                  = reinterpret_cast<float*>(data + work.k_scale);
    auto* km                  = reinterpret_cast<float*>(data + work.k_mean);
    auto* out                 = reinterpret_cast<half*>(data + work.output);
    const cudaStream_t stream = ctx.stream();

    sage_key_mean<D><<<dim3(D / 32, hk * batch), 256, 0, stream>>>(static_cast<const float*>(k->data), km, lk);
    sage_quantize<D, 32, false><<<dim3(GGML_PAD(lq, 128) / 32, hq * batch), 256, 0, stream>>>(
        static_cast<const float*>(q->data), nullptr, qi, qs, lq);
    sage_quantize<D, 64, true><<<dim3(GGML_PAD(lk, 64) / 64, hk * batch), 256, 0, stream>>>(
        static_cast<const float*>(k->data), km, ki, ks, lk);

    constexpr size_t shared_bytes = (128 + 64 + 64 * 2) * D;
    auto kernel                   = sageattention::qk_int_sv_f16_attn_kernel<128, 64, 32, 64, D,
                                                           sageattention::DataType::kInt8, sageattention::QuantGranularity::kPerWarp,
                                                           sageattention::QuantGranularity::kPerWarp, float, false, half,
                                                           sageattention::ComputeUnit::kTensorCore, sageattention::MaskMode::kNone, false, false>;
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    kernel<<<dim3(GGML_PAD(lq, 128) / 128, hq, batch), dim3(32, 4), shared_bytes, stream>>>(
        qi, ki, static_cast<half*>(v->data), out, nullptr, qs, ks, nullptr,
        lq, lk, hq / hk,
        D * lq * hq, D, D * lq,
        D * lk * hk, D, D * lk,
        D * lk * hk, D, D * lk,
        D * lq * hq, D * hq, D,
        scale);
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(out, static_cast<float*>(dst->data), ggml_nelements(dst), stream);
}

bool ggml_cuda_sage_attn_supported(int device, const ggml_tensor* op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE ||
        ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const auto mode = sage_mode(device, op);
    if (mode < GGML_SAGE_ATTN_AUTO || mode > GGML_SAGE_ATTN_2_PLUS_PLUS) {
        return false;
    }
    if (mode != GGML_SAGE_ATTN_AUTO && (sage_cuda_version < 12040 || cc < GGML_CUDA_CC_ADA_LOVELACE ||
                                        ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_ADA_LOVELACE)) {
        return false;
    }
    if (mode == GGML_SAGE_ATTN_2_PLUS_PLUS && sage_cuda_version < 12080) {
        return false;
    }
    const ggml_tensor* q = op->src[0];
    const ggml_tensor* k = op->src[1];
    const ggml_tensor* v = op->src[2];
    if (!q || !k || !v || op->src[3] || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32 || (q->ne[0] != 64 && q->ne[0] != 128)) {
        return false;
    }
    for (const ggml_tensor* t : {q, k, v, op}) {
        if (!ggml_is_contiguous(t) || ggml_nelements(t) > INT_MAX || t->ne[2] * t->ne[3] > 65535) {
            return false;
        }
    }
    if (k->ne[0] * GGML_PAD(k->ne[1], 64) * k->ne[2] * k->ne[3] > INT_MAX) {
        return false;
    }
    return q->ne[0] == k->ne[0] && ggml_are_same_shape(k, v) && q->ne[2] % k->ne[2] == 0 &&
           q->ne[3] == k->ne[3] && op->ne[0] == q->ne[0] && op->ne[1] == q->ne[2] &&
           op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3];
}

size_t ggml_cuda_sage_attn_get_alloc_size(const ggml_tensor* op) {
    // Scratch belongs to the graph allocation, so graph capture and VRAM planning see its full lifetime.
    return SageWorkspace(op).size;
}

void ggml_cuda_sage_attn(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(ggml_cuda_sage_attn_supported(ctx.device, dst));
    if (dst->src[0]->ne[0] == 64) {
        sage_attn<64>(ctx, dst);
    } else {
        sage_attn<128>(ctx, dst);
    }
}

#else

bool ggml_cuda_sage_attn_supported(int, const ggml_tensor*) {
    return false;
}

size_t ggml_cuda_sage_attn_get_alloc_size(const ggml_tensor* op) {
    return ggml_nbytes(op);
}

void ggml_cuda_sage_attn(ggml_backend_cuda_context&, ggml_tensor*) {
    GGML_ABORT("SageAttention was not compiled into this backend");
}

#endif
