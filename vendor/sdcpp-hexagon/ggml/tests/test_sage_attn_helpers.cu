#include <cuda_runtime.h>
#include <cstdio>

static __global__ void poison_attention_shared_memory() {
    // Exercise short K tails after kernels that leave FP16 NaNs in shared memory.
    extern __shared__ volatile unsigned int storage[];
    for (int i = threadIdx.x; i < 48 * 1024 / int(sizeof(unsigned int)); i += blockDim.x) {
        storage[i] = 0x7fff7fff;
    }
}

bool sage_test_poison_shared_memory() {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return false;
    }
    poison_attention_shared_memory<<<properties.multiProcessorCount * 16, 256, 48 * 1024>>>();
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Shared-memory poison failed: %s\n", cudaGetErrorString(status));
    }
    return status == cudaSuccess;
}
