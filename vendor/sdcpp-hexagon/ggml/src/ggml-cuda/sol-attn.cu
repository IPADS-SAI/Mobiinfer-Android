// Sol-Attn diagonal-threshold algorithm:
// https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn
#include "sol-attn.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDACC_VER_MAJOR__ >= 12

#include <climits>
#include "mma.cuh"

static constexpr int SOL_DIM   = 128;
static constexpr int SOL_BLOCK = 64;

struct SolWorkspace {
    size_t q, k, v, kc, vc, qm, stats, threshold, size;
    int tokens, blocks;

    explicit SolWorkspace(const ggml_tensor* dst) {
        const auto* input  = dst->src[0];
        const size_t heads = input->ne[2] * input->ne[3];
        tokens             = GGML_PAD(input->ne[1], SOL_BLOCK);
        blocks             = GGML_PAD(tokens / SOL_BLOCK, SOL_BLOCK);
        size               = GGML_PAD(ggml_nbytes(dst), 256);
        q                  = reserve(heads * tokens * SOL_DIM * sizeof(nv_bfloat16));
        k                  = reserve(heads * tokens * SOL_DIM * sizeof(nv_bfloat16));
        v                  = reserve(heads * tokens * SOL_DIM * sizeof(nv_bfloat16));
        kc                 = reserve(heads * blocks * SOL_DIM * sizeof(nv_bfloat16));
        vc                 = reserve(heads * blocks * SOL_DIM * sizeof(nv_bfloat16));
        qm                 = reserve(heads * (tokens / SOL_BLOCK) * SOL_DIM * sizeof(float));
        stats              = reserve(heads * 2 * SOL_DIM * sizeof(float));
        threshold          = reserve(heads * (tokens / SOL_BLOCK) * sizeof(float));
    }

    size_t reserve(size_t bytes) {
        const size_t offset = size;
        size += GGML_PAD(bytes, 256);
        return offset;
    }
};

static __global__ void sol_pack(const float* q, const float* k, const float* v, nv_bfloat16* qb, nv_bfloat16* kb, nv_bfloat16* vb, int length, int padded) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= padded * SOL_DIM) {
        return;
    }
    const int token   = index / SOL_DIM;
    const int channel = index % SOL_DIM;
    const size_t src  = size_t(blockIdx.y) * length * SOL_DIM + index;
    const size_t dst  = size_t(blockIdx.y) * padded * SOL_DIM + index;
    qb[dst]           = __float2bfloat16(token < length ? q[src] : 0.f);
    kb[dst]           = __float2bfloat16(token < length ? k[src] : 0.f);
    vb[(size_t(blockIdx.y) * SOL_DIM + channel) * padded + token] =
        __float2bfloat16(token < length ? v[src] : 0.f);
}

static __global__ void sol_summaries(const nv_bfloat16* q, const nv_bfloat16* k, const nv_bfloat16* v, nv_bfloat16* kc, nv_bfloat16* vc, float* qm, int length, int padded, int padded_blocks) {
    const int channel = threadIdx.x;
    const int block   = blockIdx.x;
    const int count   = min(SOL_BLOCK, max(0, length - block * SOL_BLOCK));
    float sum_q = 0.f, sum_k = 0.f, sum_v = 0.f;
    for (int i = 0; i < count; ++i) {
        const int token    = block * SOL_BLOCK + i;
        const size_t index = (size_t(blockIdx.y) * padded + token) * SOL_DIM + channel;
        sum_q += __bfloat162float(q[index]);
        sum_k += __bfloat162float(k[index]);
        sum_v += __bfloat162float(v[(size_t(blockIdx.y) * SOL_DIM + channel) * padded + token]);
    }
    kc[(size_t(blockIdx.y) * padded_blocks + block) * SOL_DIM + channel] =
        __float2bfloat16(count ? sum_k / count : 0.f);
    vc[(size_t(blockIdx.y) * SOL_DIM + channel) * padded_blocks + block] = __float2bfloat16(sum_v);
    if (count) {
        qm[(size_t(blockIdx.y) * (padded / SOL_BLOCK) + block) * SOL_DIM + channel] = sum_q / count;
    }
}

static __global__ void sol_statistics(const nv_bfloat16* kc, float* stats, int blocks, int padded_blocks) {
    const int channel = threadIdx.x;
    float sum = 0.f, sum_sq = 0.f;
    for (int i = 0; i < blocks; ++i) {
        const float value = __bfloat162float(kc[(size_t(blockIdx.x) * padded_blocks + i) * SOL_DIM + channel]);
        sum += value;
        sum_sq += value * value;
    }
    const float mean                                            = sum / blocks;
    stats[size_t(blockIdx.x) * 2 * SOL_DIM + channel]           = mean;
    stats[size_t(blockIdx.x) * 2 * SOL_DIM + SOL_DIM + channel] = fmaxf(sum_sq / blocks - mean * mean, 0.f);
}

static __global__ void sol_thresholds(const float* qm, const float* stats, float* threshold, int blocks, float scale, float tau) {
    const int channel = threadIdx.x;
    const float q     = qm[(size_t(blockIdx.y) * blocks + blockIdx.x) * SOL_DIM + channel];
    float mean        = q * stats[size_t(blockIdx.y) * 2 * SOL_DIM + channel];
    float var         = q * q * stats[size_t(blockIdx.y) * 2 * SOL_DIM + SOL_DIM + channel];
    mean              = warp_reduce_sum(mean);
    var               = warp_reduce_sum(var);
    __shared__ float means[4], vars[4];
    if (channel % 32 == 0) {
        means[channel / 32] = mean;
        vars[channel / 32]  = var;
    }
    __syncthreads();
    if (channel == 0) {
        mean = means[0] + means[1] + means[2] + means[3];
        var  = vars[0] + vars[1] + vars[2] + vars[3];
        threshold[size_t(blockIdx.y) * blocks + blockIdx.x] =
            mean * scale + tau * sqrtf(fmaxf(var * scale * scale, 0.f) + 1.e-6f);
    }
}

using SolQTile = ggml_cuda_mma::tile<16, 8, nv_bfloat162>;
using SolKTile = ggml_cuda_mma::tile<8, 8, nv_bfloat162>;
using SolTile  = ggml_cuda_mma::tile<16, 8, float>;

static __device__ __forceinline__ void sol_scores(const SolQTile (&q)[8], const nv_bfloat16* k, SolTile (&scores)[8], float scale) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        scores[n] = SolTile{};
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            SolKTile kt;
            ggml_cuda_mma::load_generic(kt, reinterpret_cast<const nv_bfloat162*>(k + n * 8 * SOL_DIM + d * 16), SOL_DIM / 2);
            ggml_cuda_mma::mma(scores[n], q[d], kt);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            scores[n].x[i] *= scale;
        }
    }
}

template <bool PROXY>
static __device__ __forceinline__ void sol_update(SolTile (&scores)[8], SolTile (&output)[16], float (&row_max)[2], float (&row_sum)[2], nv_bfloat16* probabilities, const nv_bfloat16* values, int stride, int start, int length, const int* exact) {
    float maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int column = n * 8 + SolTile::get_j(i);
            const bool valid = PROXY ? ((start + column) * SOL_BLOCK < length && !exact[column])
                                     : (start + column < length);
            scores[n].x[i]   = valid ? scores[n].x[i] : -INFINITY;
            maximum[i / 2]   = fmaxf(maximum[i / 2], scores[n].x[i]);
        }
    }
    float alpha[2], sums[2] = {0.f, 0.f};
#pragma unroll
    for (int row = 0; row < 2; ++row) {
        maximum[row]     = fmaxf(maximum[row], __shfl_xor_sync(0xffffffff, maximum[row], 1));
        maximum[row]     = fmaxf(maximum[row], __shfl_xor_sync(0xffffffff, maximum[row], 2));
        const float next = fmaxf(row_max[row], maximum[row]);
        alpha[row]       = row_max[row] == next ? 1.f : exp2f(row_max[row] - next);
        row_max[row]     = next;
    }
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int column = n * 8 + SolTile::get_j(i);
            const float p    = scores[n].x[i] == -INFINITY ? 0.f : exp2f(scores[n].x[i] - row_max[i / 2]);
            const int count  = PROXY ? min(SOL_BLOCK, max(0, length - (start + column) * SOL_BLOCK)) : 1;
            sums[i / 2] += p * count;
            probabilities[SolTile::get_i(i) * SOL_BLOCK + column] = __float2bfloat16(p);
        }
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
        sums[row] += __shfl_xor_sync(0xffffffff, sums[row], 1);
        sums[row] += __shfl_xor_sync(0xffffffff, sums[row], 2);
        row_sum[row] = row_sum[row] * alpha[row] + sums[row];
    }
    __syncwarp();
#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            output[n].x[i] *= alpha[i / 2];
        }
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            SolQTile pt;
            SolKTile vt;
            ggml_cuda_mma::load_generic(pt, reinterpret_cast<const nv_bfloat162*>(probabilities + d * 16), SOL_BLOCK / 2);
            ggml_cuda_mma::load_generic(vt, reinterpret_cast<const nv_bfloat162*>(values + n * 8 * stride + d * 16), stride / 2);
            ggml_cuda_mma::mma(output[n], pt, vt);
        }
    }
    __syncwarp();
}

static __global__ __launch_bounds__(128) void sol_forward(
    const nv_bfloat16* q,
    const nv_bfloat16* k,
    const nv_bfloat16* v,
    const nv_bfloat16* kc,
    const nv_bfloat16* vc,
    const float* thresholds,
    float* dst,
    int length,
    int padded,
    int padded_blocks,
    int heads,
    float scale) {
#ifdef AMPERE_MMA_AVAILABLE
    const int warp    = threadIdx.y;
    const int lane    = threadIdx.x;
    const int tid     = warp * 32 + lane;
    const int blocks  = padded / SOL_BLOCK;
    const int q_start = blockIdx.x * SOL_BLOCK;
    const int q_len   = min(SOL_BLOCK, length - q_start);
    const size_t base = size_t(blockIdx.y) * padded * SOL_DIM;
    SolQTile qt[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        ggml_cuda_mma::load_generic(qt[d], reinterpret_cast<const nv_bfloat162*>(q + base + (q_start + warp * 16) * SOL_DIM + d * 16), SOL_DIM / 2);
    }
    SolTile output[16];
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.f, 0.f};
    __shared__ nv_bfloat16 probabilities[SOL_BLOCK * SOL_BLOCK];
    __shared__ float routes[4][SOL_BLOCK];
    __shared__ int exact[SOL_BLOCK];
    const float threshold = thresholds[size_t(blockIdx.y) * blocks + blockIdx.x];
    for (int group = 0; group < blocks; group += SOL_BLOCK) {
        SolTile scores[8];
        sol_scores(qt, kc + (size_t(blockIdx.y) * padded_blocks + group) * SOL_DIM, scores, scale);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int column = 0; column < 2; ++column) {
                float sum = scores[n].x[column] + scores[n].x[column + 2];
                sum += __shfl_xor_sync(0xffffffff, sum, 4);
                sum += __shfl_xor_sync(0xffffffff, sum, 8);
                sum += __shfl_xor_sync(0xffffffff, sum, 16);
                if (lane < 4) {
                    routes[warp][n * 8 + 2 * lane + column] = sum;
                }
            }
        }
        __syncthreads();
        if (tid < SOL_BLOCK) {
            const int block   = group + tid;
            const float score = (routes[0][tid] + routes[1][tid] + routes[2][tid] + routes[3][tid]) / q_len;
            exact[tid]        = block < blocks && (score > threshold || abs(int(blockIdx.x) - block) <= 1);
        }
        __syncthreads();
        sol_update<true>(scores, output, row_max, row_sum, probabilities + warp * 16 * SOL_BLOCK,
                         vc + size_t(blockIdx.y) * padded_blocks * SOL_DIM + group,
                         padded_blocks, group, length, exact);
        for (int block = 0; block < SOL_BLOCK && group + block < blocks; ++block) {
            if (!exact[block]) {
                continue;
            }
            const int start = (group + block) * SOL_BLOCK;
            sol_scores(qt, k + base + start * SOL_DIM, scores, scale);
            sol_update<false>(scores, output, row_max, row_sum, probabilities + warp * 16 * SOL_BLOCK,
                              v + base + start, padded, start, length, nullptr);
        }
        __syncthreads();
    }
    const int head  = blockIdx.y % heads;
    const int batch = blockIdx.y / heads;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int token = q_start + warp * 16 + SolTile::get_i(i);
            if (token < length) {
                const int channel = n * 8 + SolTile::get_j(i);
                dst[((size_t(batch) * length + token) * heads + head) * SOL_DIM + channel] =
                    __bfloat162float(__float2bfloat16(output[n].x[i] / row_sum[i / 2]));
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}

bool ggml_cuda_sol_attn_supported(int device, const ggml_tensor* op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE ||
        ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const auto* q = op->src[0];
    const auto* k = op->src[1];
    const auto* v = op->src[2];
    if (!q || !k || !v || op->src[3] || q->ne[0] != SOL_DIM || q->ne[1] <= 0 ||
        !ggml_are_same_shape(q, k) || !ggml_are_same_shape(q, v) ||
        q->ne[2] * q->ne[3] > 65535 || GGML_PAD(q->ne[1], SOL_BLOCK) > INT_MAX / SOL_DIM) {
        return false;
    }
    for (const auto* tensor : {q, k, v, op}) {
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
            return false;
        }
    }
    float params[2];
    memcpy(params, op->op_params, sizeof(params));
    return std::isfinite(params[0]) && params[0] > 0.f && std::isfinite(params[1]) &&
           op->ne[0] == SOL_DIM && op->ne[1] == q->ne[2] && op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3];
}

size_t ggml_cuda_sol_attn_get_alloc_size(const ggml_tensor* op) {
    return SolWorkspace(op).size;
}

void ggml_cuda_sol_attn(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(ggml_cuda_sol_attn_supported(ctx.device, dst));
    const auto* q = dst->src[0];
    const SolWorkspace work(dst);
    char* data       = static_cast<char*>(dst->data);
    auto* qb         = reinterpret_cast<nv_bfloat16*>(data + work.q);
    auto* kb         = reinterpret_cast<nv_bfloat16*>(data + work.k);
    auto* vb         = reinterpret_cast<nv_bfloat16*>(data + work.v);
    auto* kc         = reinterpret_cast<nv_bfloat16*>(data + work.kc);
    auto* vc         = reinterpret_cast<nv_bfloat16*>(data + work.vc);
    auto* qm         = reinterpret_cast<float*>(data + work.qm);
    auto* stats      = reinterpret_cast<float*>(data + work.stats);
    auto* threshold  = reinterpret_cast<float*>(data + work.threshold);
    const int length = q->ne[1], heads = q->ne[2], batch_heads = heads * q->ne[3];
    const int blocks = work.tokens / SOL_BLOCK;
    float params[2];
    memcpy(params, dst->op_params, sizeof(params));
    const float scale = params[0] * 1.4426950408889634f;
    const auto stream = ctx.stream();
    sol_pack<<<dim3((work.tokens * SOL_DIM + 255) / 256, batch_heads), 256, 0, stream>>>(
        static_cast<const float*>(q->data), static_cast<const float*>(dst->src[1]->data), static_cast<const float*>(dst->src[2]->data),
        qb, kb, vb, length, work.tokens);
    sol_summaries<<<dim3(work.blocks, batch_heads), SOL_DIM, 0, stream>>>(qb, kb, vb, kc, vc, qm, length, work.tokens, work.blocks);
    sol_statistics<<<batch_heads, SOL_DIM, 0, stream>>>(kc, stats, blocks, work.blocks);
    sol_thresholds<<<dim3(blocks, batch_heads), SOL_DIM, 0, stream>>>(qm, stats, threshold, blocks, scale, params[1]);
    sol_forward<<<dim3(blocks, batch_heads), dim3(32, 4), 0, stream>>>(qb, kb, vb, kc, vc, threshold,
                                                                       static_cast<float*>(dst->data), length, work.tokens, work.blocks, heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_sol_attn_supported(int, const ggml_tensor*) {
    return false;
}

size_t ggml_cuda_sol_attn_get_alloc_size(const ggml_tensor* op) {
    return ggml_nbytes(op);
}

void ggml_cuda_sol_attn(ggml_backend_cuda_context&, ggml_tensor*) {
    GGML_ABORT("Sol-Attn was not compiled into this backend");
}

#endif
