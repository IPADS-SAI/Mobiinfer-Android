// Host regression test for the weighted Q1_0 quantizer (imatrix / quant_weights).
//
// Q1_0 stores only a per-block fp16 scale d plus one sign bit per value, so an
// importance matrix can only influence d.  This test pins down the contract of
// the weighted path added in T-0022:
//   * nullptr quant_weights keeps the output bit-identical to
//     quantize_row_q1_0_ref() (the historical path);
//   * the weighted path lowers the imatrix-weighted MSE on heterogeneous data;
//   * the sign bits are bit-identical to the unweighted path (format unchanged);
//   * the weighted path is deterministic and has the same byte size;
//   * degenerate weights (all zero / NaN / negative) fall back block by block
//     instead of producing NaNs, and negative ones are counted.
//
// The test is fully deterministic: no RNG, no model, no backend.

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Internal symbols exported by ggml/src/ggml-quants.c.  The block type itself
// is private to ggml, so the reference entry point is declared with void*.
extern "C" void quantize_row_q1_0_ref(const float * x, void * y, int64_t k);
#if defined(__GNUC__) && !defined(_WIN32)
// Weak so the same test binary can be linked against a pre-T-0022 libggml to
// dump the historical bytes for the sha256 comparison.  Absent -> skip the
// negative-weight counter assertions.
extern "C" size_t ggml_q1_0_negative_weight_count(void) __attribute__((weak));
#define HAVE_NEG_COUNTER (ggml_q1_0_negative_weight_count != nullptr)
#else
extern "C" size_t ggml_q1_0_negative_weight_count(void);
#define HAVE_NEG_COUNTER 1
#endif

static constexpr int64_t NROWS    = 8;
static constexpr int64_t NPER_ROW = 128*4;
static constexpr int64_t NBLOCK   = NPER_ROW/128;

static int g_failures = 0;

static bool write_file(const char * path, const void * data, size_t size) {
    FILE * f = std::fopen(path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    const bool ok = size == 0 || std::fwrite(data, 1, size, f) == size;
    std::fclose(f);
    return ok;
}

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        ++g_failures;                                                           \
    }                                                                           \
} while (0)

// Deterministic heterogeneous data: a smooth magnitude ramp, sparse large
// outliers, mixed signs and a few exact zeros (whose sign bit must still be 1).
// The imatrix puts a large weight on exactly the outlier columns, which is the
// case a per-column importance matrix is meant to capture.
static void generate_data(std::vector<float> & x, std::vector<float> & qw) {
    x.assign((size_t) (NROWS*NPER_ROW), 0.0f);
    qw.assign((size_t) NPER_ROW, 0.0f);

    for (int64_t c = 0; c < NPER_ROW; ++c) {
        const int u = (int) ((c*37 + 11) % 101);
        qw[c] = (c % 16 == 0) ? 50.0f : 1.0f + 0.01f*(u % 7);
    }

    for (int64_t r = 0; r < NROWS; ++r) {
        for (int64_t c = 0; c < NPER_ROW; ++c) {
            const int u = (int) ((c*37 + 11 + r*13) % 101);
            float mag = 0.2f + 0.03f*u;
            if (c % 16 == 0) {
                mag *= 8.0f;
            }
            float v = ((u + r) % 2) ? mag : -mag;
            if (c % 97 == 0) {
                v = 0.0f;
            }
            x[(size_t) (r*NPER_ROW + c)] = v;
        }
    }
}

static size_t row_bytes() {
    return ggml_row_size(GGML_TYPE_Q1_0, NPER_ROW);
}

// Reference output produced row by row with the historical quantizer.
static std::vector<uint8_t> reference_quantize(const std::vector<float> & x) {
    const size_t rs = row_bytes();
    std::vector<uint8_t> out(rs*NROWS);
    for (int64_t r = 0; r < NROWS; ++r) {
        quantize_row_q1_0_ref(x.data() + r*NPER_ROW, out.data() + r*rs, NPER_ROW);
    }
    return out;
}

// Dequantize through the registered type traits (the same entry point the
// engine uses), then average the squared error with the imatrix weights.
static double weighted_mse(const std::vector<uint8_t> & q, const std::vector<float> & x,
                           const std::vector<float> & qw) {
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q1_0);
    CHECK(traits != nullptr);
    CHECK(traits->to_float != nullptr);
    if (traits == nullptr || traits->to_float == nullptr) {
        return NAN;
    }

    const size_t n = (size_t) (NROWS*NPER_ROW);
    std::vector<float> deq(n);
    traits->to_float(q.data(), deq.data(), (int64_t) n);

    double sum = 0.0;
    double wsum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = (double) x[i] - (double) deq[i];
        sum  += (double) qw[i % (size_t) NPER_ROW] * diff*diff;
        wsum += (double) qw[i % (size_t) NPER_ROW];
    }
    return wsum > 0.0 ? sum/wsum : NAN;
}

// The packed sign bits (everything except the fp16 scale) must never change.
static bool sign_bits_equal(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, size_t rs, size_t type_size) {
    if (a.size() != b.size() || type_size < 3) {
        return false;
    }
    for (int64_t r = 0; r < NROWS; ++r) {
        for (int64_t ib = 0; ib < NBLOCK; ++ib) {
            const size_t off = (size_t) (r*rs + ib*type_size);
            if (std::memcmp(a.data() + off + 2, b.data() + off + 2, type_size - 2) != 0) {
                return false;
            }
        }
    }
    return true;
}

// Compare one block of two buffers.
static bool block_equal(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, size_t off, size_t n) {
    return std::memcmp(a.data() + off, b.data() + off, n) == 0;
}

int main(int argc, char ** argv) {
    const char * dump_prefix = nullptr;
    bool nullptr_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--nullptr-only") == 0) {
            nullptr_only = true;
        } else {
            std::fprintf(stderr, "usage: %s [--dump <prefix>] [--nullptr-only]\n", argv[0]);
            return 2;
        }
    }

    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q1_0);
    CHECK(traits != nullptr);
    if (traits == nullptr) {
        std::fprintf(stderr, "no Q1_0 type traits\n");
        return 1;
    }

    const size_t rs    = row_bytes();
    const size_t ts    = traits->type_size;
    const size_t bytes = rs*NROWS;

    // 1) Layout / size contract: unaffected by the weighting.
    std::printf("row_size(Q1_0, %lld) = %zu, type_size = %zu, block = %lld\n",
                (long long) NPER_ROW, rs, traits->type_size, (long long) traits->blck_size);
    CHECK(traits->blck_size == 128);
    CHECK(rs == (size_t) NBLOCK*traits->type_size);
    CHECK(traits->type_size == 18);

    std::vector<float> x;
    std::vector<float> qw;
    generate_data(x, qw);

    // 2) Unweighted path: the historical bytes, and the reference function.
    std::vector<uint8_t> q_un(bytes, 0xAA);
    const size_t n_un = ggml_quantize_chunk(GGML_TYPE_Q1_0, x.data(), q_un.data(), 0, NROWS, NPER_ROW, nullptr);
    const std::vector<uint8_t> q_ref = reference_quantize(x);
    const size_t n_ref = q_ref.size();
    CHECK(n_un == bytes);
    CHECK(n_ref == n_un);
    CHECK(std::memcmp(q_un.data(), q_ref.data(), bytes) == 0);
    std::printf("nullptr path: %zu bytes, byte-identical to quantize_row_q1_0_ref: %s\n",
                n_un, std::memcmp(q_un.data(), q_ref.data(), bytes) == 0 ? "yes" : "NO");

    if (dump_prefix != nullptr) {
        const std::string base(dump_prefix);
        CHECK(write_file((base + ".src.f32").c_str(), x.data(), x.size()*sizeof(float)));
        CHECK(write_file((base + ".imatrix.f32").c_str(), qw.data(), qw.size()*sizeof(float)));
        CHECK(write_file((base + ".unweighted.bin").c_str(), q_un.data(), q_un.size()));
    }
    if (nullptr_only) {
        std::printf("--nullptr-only: skipping weighted assertions\n");
        return g_failures != 0 ? 1 : 0;
    }

    // 3) Weighted path: same size, deterministic, sign bits unchanged.
    std::vector<uint8_t> q_w1(bytes, 0x55);
    std::vector<uint8_t> q_w2(bytes, 0x55);
    const size_t n_w1 = ggml_quantize_chunk(GGML_TYPE_Q1_0, x.data(), q_w1.data(), 0, NROWS, NPER_ROW, qw.data());
    const size_t n_w2 = ggml_quantize_chunk(GGML_TYPE_Q1_0, x.data(), q_w2.data(), 0, NROWS, NPER_ROW, qw.data());
    CHECK(n_w1 == bytes);
    CHECK(n_w2 == bytes);
    CHECK(n_w1 == n_un);
    CHECK(std::memcmp(q_w1.data(), q_w2.data(), bytes) == 0);
    CHECK(std::memcmp(q_w1.data(), q_un.data(), bytes) != 0); // weighting must have an effect
    CHECK(sign_bits_equal(q_w1, q_un, rs, ts));
    std::printf("weighted path: %zu bytes, deterministic: %s, sign bits == unweighted: %s, differs from unweighted: %s\n",
                n_w1,
                std::memcmp(q_w1.data(), q_w2.data(), bytes) == 0 ? "yes" : "NO",
                sign_bits_equal(q_w1, q_un, rs, ts) ? "yes" : "NO",
                std::memcmp(q_w1.data(), q_un.data(), bytes) != 0 ? "yes" : "NO");

    // 4) Weighted MSE must improve substantially on this weighted data.
    const double mse_u = weighted_mse(q_un, x, qw);
    const double mse_w = weighted_mse(q_w1, x, qw);
    const double gain  = mse_u > 0.0 ? 1.0 - mse_w/mse_u : 0.0;
    std::printf("imatrix-weighted MSE: unweighted=%.6f weighted=%.6f reduction=%.2f%% (threshold 5.00%%)\n",
                mse_u, mse_w, 100.0*gain);
    CHECK(std::isfinite(mse_u));
    CHECK(std::isfinite(mse_w));
    CHECK(mse_w < 0.95*mse_u);

    // 5) Data validation (no NaN/Inf scales) for both paths.
    CHECK(ggml_validate_row_data(GGML_TYPE_Q1_0, q_un.data(), n_un));
    CHECK(ggml_validate_row_data(GGML_TYPE_Q1_0, q_w1.data(), n_w1));
    std::printf("ggml_validate_row_data: unweighted=%s weighted=%s\n",
                ggml_validate_row_data(GGML_TYPE_Q1_0, q_un.data(), n_un) ? "ok" : "FAILED",
                ggml_validate_row_data(GGML_TYPE_Q1_0, q_w1.data(), n_w1) ? "ok" : "FAILED");

    // 6) All-zero weights: sum(w) == 0 in every block -> per-block reference
    //    fallback, so the bytes equal the unweighted output.
    std::vector<float> qw_zero((size_t) NPER_ROW, 0.0f);
    std::vector<uint8_t> q_zero(bytes, 0);
    const size_t n_zero = ggml_quantize_chunk(GGML_TYPE_Q1_0, x.data(), q_zero.data(), 0, NROWS, NPER_ROW, qw_zero.data());
    CHECK(n_zero == bytes);
    CHECK(std::memcmp(q_zero.data(), q_ref.data(), bytes) == 0);
    std::printf("all-zero weights: fallback to ref bytes = %s\n",
                std::memcmp(q_zero.data(), q_ref.data(), bytes) == 0 ? "yes" : "NO");

    // 7) A single NaN weight only poisons its own block (column 5 -> block 0 of
    //    every row); the remaining blocks stay weighted.
    std::vector<float> qw_nan = qw;
    qw_nan[5] = std::nanf("");
    std::vector<uint8_t> q_nan(bytes, 0);
    const size_t n_nan = ggml_quantize_chunk(GGML_TYPE_Q1_0, x.data(), q_nan.data(), 0, NROWS, NPER_ROW, qw_nan.data());
    CHECK(n_nan == bytes);
    bool nan_ok = true;
    for (int64_t r = 0; r < NROWS; ++r) {
        nan_ok = nan_ok && block_equal(q_nan, q_ref, (size_t) (r*rs), ts);
        for (int64_t ib = 1; ib < NBLOCK; ++ib) {
            nan_ok = nan_ok && block_equal(q_nan, q_w1, (size_t) (r*rs + ib*ts), ts);
        }
    }
    CHECK(nan_ok);
    std::printf("one NaN weight: block 0 falls back to ref, other blocks stay weighted = %s\n",
                nan_ok ? "yes" : "NO");

    // 8) Negative weights are clamped to 0 and counted, never used as-is.
    std::vector<float> qw_neg = qw;
    int64_t neg_per_row = 0;
    for (int64_t c = 0; c < NPER_ROW; ++c) {
        if (c % 32 == 0) {
            qw_neg[(size_t) c] = -qw_neg[(size_t) c];
            ++neg_per_row;
        }
    }
    const size_t neg_before = HAVE_NEG_COUNTER ? ggml_q1_0_negative_weight_count() : 0;
    std::vector<uint8_t> q_neg(bytes, 0);
    const size_t n_neg = ggml_quantize_chunk(GGML_TYPE_Q1_0, x.data(), q_neg.data(), 0, NROWS, NPER_ROW, qw_neg.data());
    const size_t neg_after = HAVE_NEG_COUNTER ? ggml_q1_0_negative_weight_count() : 0;
    CHECK(n_neg == bytes);
    if (HAVE_NEG_COUNTER) {
        CHECK(neg_after - neg_before == (size_t) (NROWS*neg_per_row));
    }
    CHECK(std::memcmp(q_neg.data(), q_ref.data(), bytes) != 0); // some columns still weighted
    CHECK(sign_bits_equal(q_neg, q_ref, rs, ts));
    CHECK(ggml_validate_row_data(GGML_TYPE_Q1_0, q_neg.data(), n_neg));
    std::printf("negative weights: clamped=0, counted %zu (expected %lld, counter %s), sign bits == ref: %s\n",
                neg_after - neg_before, (long long) (NROWS*neg_per_row),
                HAVE_NEG_COUNTER ? "present" : "absent in this libggml",
                sign_bits_equal(q_neg, q_ref, rs, ts) ? "yes" : "NO");

    // 9) All-negative weights: every block has sum(w) == 0 -> full fallback.
    std::vector<float> qw_allneg((size_t) NPER_ROW, -1.0f);
    std::vector<uint8_t> q_allneg(bytes, 0);
    const size_t n_allneg = ggml_quantize_chunk(GGML_TYPE_Q1_0, x.data(), q_allneg.data(), 0, NROWS, NPER_ROW, qw_allneg.data());
    CHECK(n_allneg == bytes);
    CHECK(std::memcmp(q_allneg.data(), q_ref.data(), bytes) == 0);
    if (HAVE_NEG_COUNTER) {
        CHECK(ggml_q1_0_negative_weight_count() == neg_after + (size_t) (NROWS*NPER_ROW));
    }
    std::printf("all-negative weights: full fallback to ref bytes = %s\n",
                std::memcmp(q_allneg.data(), q_ref.data(), bytes) == 0 ? "yes" : "NO");

    if (g_failures != 0) {
        std::fprintf(stderr, "test-quantize-q1-0-weighted: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("test-quantize-q1-0-weighted: all checks passed\n");
    return 0;
}
