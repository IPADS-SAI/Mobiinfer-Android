#ifndef GGML_CUDA_SAGE_ATTN_CUH
#define GGML_CUDA_SAGE_ATTN_CUH

#include "common.cuh"

bool ggml_cuda_sage_attn_supported(int device, const ggml_tensor* op);
size_t ggml_cuda_sage_attn_get_alloc_size(const ggml_tensor* op);
void ggml_cuda_sage_attn(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

#endif
