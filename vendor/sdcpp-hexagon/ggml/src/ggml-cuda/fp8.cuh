#pragma once

#include "common.cuh"

struct ggml_fp8_e4m3_cuda {
    uint8_t value;

    __device__ operator float() const {
        const uint32_t sign     = value >> 7;
        const uint32_t exponent = (value >> 3) & 0x0f;
        const uint32_t mantissa = value & 0x07;

        if (exponent == 0x0f && mantissa == 0x07) {
            return sign ? -nanf("") : nanf("");
        }

        float result;
        if (exponent == 0) {
            result = ldexpf((float) mantissa, -9);
        } else {
            result = ldexpf(1.0f + (float) mantissa / 8.0f, (int) exponent - 7);
        }
        return sign ? -result : result;
    }
};

struct ggml_fp8_e5m2_cuda {
    uint8_t value;

    __device__ operator float() const {
        const uint32_t sign     = value >> 7;
        const uint32_t exponent = (value >> 2) & 0x1f;
        const uint32_t mantissa = value & 0x03;

        if (exponent == 0x1f) {
            const float result = mantissa == 0 ? INFINITY : nanf("");
            return sign ? -result : result;
        }

        float result;
        if (exponent == 0) {
            result = ldexpf((float) mantissa, -16);
        } else {
            result = ldexpf(1.0f + (float) mantissa / 4.0f, (int) exponent - 15);
        }
        return sign ? -result : result;
    }
};

static_assert(sizeof(ggml_fp8_e4m3_cuda) == 1, "unexpected E4M3 storage size");
static_assert(sizeof(ggml_fp8_e5m2_cuda) == 1, "unexpected E5M2 storage size");
