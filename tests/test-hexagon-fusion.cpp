// Graph-level acceptance test for the Hexagon (HTP) op-fusion rules that cannot be covered by
// test-backend-ops: those cases build single ops, so no consumer chain exists and no fusion fires.
//
// This program builds small multi-op graphs, runs them on the HTP device and on the CPU backend,
// and compares the results. It also counts the backend's own "fused <NAME>" log lines, which is the
// only direct proof that a fusion rule actually triggered.
//
// Usage (on device, next to the backend .so files; HTP runs need the device lock):
//   GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_OPFUSION=1023 ~/csm/moe-exp/devlock -w 600
//     ./test-hexagon-fusion --arm unfused --dump /tmp/before.bin
//   GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_OPFUSION=1 ~/csm/moe-exp/devlock -w 600
//     ./test-hexagon-fusion --arm fused --dump /tmp/after.bin
//   ./test-hexagon-fusion --diff /tmp/before.bin /tmp/after.bin
//
// Covered graphs:
//   1. ggml_cont(ggml_argsort_top_k(x, k))        correctness only (the TOP_K rewrite was rejected)
//   2. ggml_cont(ggml_concat(a, b, 0))            -> HTP_FUSE_CONCAT_COPY (2 ops -> 1)
//   3. ggml_add(ggml_mul_mat(a, b), c)            -> HTP_FUSE_MUL_MAT_ADD  (multi-row needs
//                                                    GGML_HEXAGON_MMADD_MULTIROW=1)
//   4. SUM_ROWS -> CLAMP -> DIV                   -> HTP_FUSE_ROWNORM_DIV (3 ops -> 1)
//
// --arm selects the expected arm, so a process where neither new rule fires fails.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------
// fusion-hit counting via the ggml log (the backend reports "ggml-hex: <sess> fused <NAME> (...)")
// ---------------------------------------------------------------------------------------------

struct fuse_stats {
    int n_hits = 0;
    int n_ops_submitted = 0;   // "execute-op" lines: one per op handed to the DSP (fused ops count once)
    std::vector<std::string> ops_seen; // op names in submission order (for --verbose-ops)
    std::vector<std::string> names;

    int count(const char * name) const {
        int n = 0;
        for (const auto & s : names) {
            if (s == name) {
                n++;
            }
        }
        return n;
    }
};

static fuse_stats g_fuse_stats;

static void fusion_log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;

    const char * op_p = strstr(text, "execute-op ");
    if (op_p) {
        g_fuse_stats.n_ops_submitted++;
        op_p += strlen("execute-op ");
        const char * op_e = op_p;
        while (*op_e && *op_e != '|' && *op_e != '\n') {
            op_e++;
        }
        g_fuse_stats.ops_seen.emplace_back(op_p, (size_t) (op_e - op_p));
    }

    const char * p = strstr(text, " fused ");
    if (p) {
        p += strlen(" fused ");
        const char * e = p;
        while (*e && *e != ' ' && *e != '\n' && *e != '(') {
            e++;
        }
        g_fuse_stats.n_hits++;
        g_fuse_stats.names.emplace_back(p, (size_t) (e - p));
    }

    fputs(text, stderr);
}

// ---------------------------------------------------------------------------------------------
// deterministic tensor data (identical for the CPU reference run and the HTP run)
// ---------------------------------------------------------------------------------------------

static uint32_t hash_name(const char * s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h = (h ^ (uint8_t) *s++) * 16777619u;
    }
    return h;
}

static uint32_t lcg_next(uint32_t & s) {
    s = s * 1103515245u + 12345u;
    return s >> 8;
}

// fill a tensor with deterministic data; ties > 0 collapses f32 values to `i % ties` so that the
// sort has many equal elements (tie behaviour is the interesting part of the ARGSORT/TOP_K rewrite)
static void gen_tensor_data(const ggml_tensor * t, int ties, bool row_values, std::vector<uint8_t> & out) {
    out.resize(ggml_nbytes(t));

    const int64_t n = ggml_nelements(t);
    uint32_t seed = hash_name(t->name);

    if (t->type == GGML_TYPE_F32) {
        float * d = (float *) out.data();
        for (int64_t i = 0; i < n; i++) {
            if (row_values) {
                d[i] = (float) (i / t->ne[0] + 1);
            } else if (ties > 0) {
                d[i] = (float) (i % ties);
            } else {
                seed = lcg_next(seed);
                d[i] = (float) ((int32_t) (seed % 2001) - 1000) / 1000.0f;
            }
        }
    } else if (t->type == GGML_TYPE_I32) {
        int32_t * d = (int32_t *) out.data();
        for (int64_t i = 0; i < n; i++) {
            seed = lcg_next(seed);
            d[i] = (int32_t) (seed % 1024);
        }
    } else {
        // unsupported input type for this test: zero it (cases that need it are skipped earlier)
        memset(out.data(), 0, out.size());
    }
}

// ---------------------------------------------------------------------------------------------
// graph cases
// ---------------------------------------------------------------------------------------------

struct case_params {
    int64_t ne0 = 0; // argsort columns / concat width / matmul m
    int64_t ne1 = 0; // tokens / rows / matmul n
    int64_t k   = 0; // top-k / matmul k
    int     ties = 0;
    bool    row_values = false; // fill each row with 1, 2, 3, ...
};

typedef ggml_tensor * (*build_fn_t)(ggml_context * ctx, const case_params & p);

// 1. ggml_cont(ggml_argsort_top_k(x, k)) -- exactly what src/models/qwen35moe.cpp:488 builds
static ggml_tensor * build_argsort_topk_cont(ggml_context * ctx, const case_params & p) {
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.ne0, p.ne1);
    ggml_set_name(logits, "logits");

    ggml_tensor * top = ggml_argsort_top_k(ctx, logits, (int) p.k);
    ggml_tensor * out = ggml_cont(ctx, top);
    ggml_set_name(out, "out");
    return out;
}

// 2. ggml_cont(ggml_concat(a, b, 0)) -- what src/models/qwen35moe.cpp:500 builds
static ggml_tensor * build_concat_cont(ggml_context * ctx, const case_params & p) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, p.ne0, p.ne1);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, p.k,   p.ne1);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    ggml_tensor * cat = ggml_concat(ctx, a, b, 0);
    ggml_tensor * out = ggml_cont(ctx, cat);
    ggml_set_name(out, "out");
    return out;
}

// 2b. Negative case: ggml_cont(concat(a, b, 1)) where the CONT output aliases b.
// Concatenating on the outer dimension gives b and the CONT destination the same strides even
// though their shapes differ. The fusion must reject this: CONCAT writes a into the destination
// first, which would overwrite b before CONCAT reads it.
//
// b is a view of a same-sized backing tensor so the aliased destination (a+b rows) fits in b's
// storage without overlapping the CONCAT intermediate. The test harness skips view tensors when
// filling inputs; the backing tensor carries the data.
static ggml_tensor * build_concat_cont_overlap(ggml_context * ctx, const case_params & p) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.ne0, p.ne1);
    ggml_set_name(a, "a");

    ggml_tensor * b_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.ne0, p.ne1 + p.k);
    ggml_set_name(b_storage, "b_storage");
    ggml_tensor * b = ggml_view_2d(ctx, b_storage, p.ne0, p.k, b_storage->nb[1], 0);
    ggml_set_name(b, "b");

    ggml_tensor * cat = ggml_concat(ctx, a, b, 1);
    ggml_tensor * out = ggml_cont(ctx, cat);
    ggml_set_name(out, "out");
    return out;
}

// 3. ggml_add(ggml_mul_mat(a, b), c) -- the residual pattern of every transformer layer
static ggml_tensor * build_mul_mat_add(ggml_context * ctx, const case_params & p) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.k,   p.ne0);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.k,   p.ne1);
    ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.ne0, p.ne1);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_name(c, "c");

    ggml_tensor * mm  = ggml_mul_mat(ctx, a, b);
    ggml_tensor * out = ggml_add(ctx, mm, c);
    ggml_set_name(out, "out");
    return out;
}

// 4. MoE top-k weight normalisation: sum_rows(w) -> clamp(sum) -> div(w, sum)  (3 ops -> 1)
//    exactly what src/llama-graph.cpp builds before the expert weights are applied
static ggml_tensor * build_moe_wnorm(ggml_context * ctx, const case_params & p) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.k, p.ne1);
    ggml_set_name(w, "w");

    ggml_tensor * sums    = ggml_sum_rows(ctx, w);
    ggml_tensor * clamped = ggml_clamp(ctx, sums, 6.103515625e-5f, INFINITY);
    ggml_tensor * out     = ggml_div(ctx, w, clamped);
    ggml_set_name(out, "out");
    return out;
}

// Negative case: SUM_ROWS consumes a shorter view at the same address as DIV's numerator. Pointer
// equality on data alone would incorrectly fuse this graph and sum the full numerator.
static ggml_tensor * build_moe_wnorm_shared_data_view(ggml_context * ctx, const case_params & p) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.k, p.ne1);
    ggml_set_name(w, "w");

    ggml_tensor * view    = ggml_view_1d(ctx, w, p.k / 2, 0);
    ggml_tensor * sums    = ggml_sum_rows(ctx, view);
    ggml_tensor * clamped = ggml_clamp(ctx, sums, 6.103515625e-5f, INFINITY);
    ggml_tensor * out     = ggml_div(ctx, w, clamped);
    ggml_set_name(out, "out");
    return out;
}

// Negative case: DIV uses a zero-offset (1,1) VIEW of the clamped (1,4) sum. The view has the same
// data address as the CLAMP output, but it is not the same tensor: DIV must broadcast the first
// denominator, whereas the fused ROWNORM kernel would use each row's own denominator.
static ggml_tensor * build_moe_wnorm_divisor_view(ggml_context * ctx, const case_params & p) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.k, p.ne1);
    ggml_set_name(w, "w");

    ggml_tensor * sums    = ggml_sum_rows(ctx, w);
    ggml_tensor * clamped = ggml_clamp(ctx, sums, 6.103515625e-5f, INFINITY);
    ggml_tensor * divisor = ggml_view_2d(ctx, clamped, 1, 1, clamped->nb[1], 0);
    ggml_tensor * out     = ggml_div(ctx, w, divisor);
    ggml_set_name(out, "out");
    return out;
}

// Negative case: two CLAMPs in series must not overwrite the first clamp's bounds.
static ggml_tensor * build_moe_wnorm_double_clamp(ggml_context * ctx, const case_params & p) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.k, p.ne1);
    ggml_set_name(w, "w");

    ggml_tensor * sums       = ggml_sum_rows(ctx, w);
    ggml_tensor * clamped_lo = ggml_clamp(ctx, sums, 0.0f, 1.0f);
    ggml_tensor * clamped_hi = ggml_clamp(ctx, clamped_lo, 2.0f, 3.0f);
    ggml_tensor * out        = ggml_div(ctx, w, clamped_hi);
    ggml_set_name(out, "out");
    return out;
}

struct test_graph {
    const char * name;
    const char * fusion;    // which fusion rule it is supposed to exercise (empty = negative control)
    build_fn_t   build;
    case_params  params;
    // Comparison mode against the CPU reference:
    //  - i32 outputs are compared exactly, EXCEPT when value_key is set: then the returned indices
    //    are mapped through that input tensor and the *values* are compared. Sort implementations
    //    differ in how they order equal elements, so comparing raw indices is not meaningful.
    const char * value_key; // input tensor name to map i32 indices through (nullptr = compare raw)
    double       rel_tol;   // >0: relative tolerance for f32 outputs (0 = exact)
    const char * forbidden_fusion = nullptr; // must not appear in the backend's fusion log
    int          ops_unfused = -1; // exact submitted-op count in --arm unfused (-1 = not asserted)
    int          ops_fused = -1;   // exact submitted-op count in --arm fused (-1 = not asserted)
    const char * alias_output_to = nullptr; // test only: point the graph output at this input's data
};

enum test_arm {
    TEST_ARM_UNFUSED,
    TEST_ARM_FUSED,
};

static const char * arm_name(test_arm arm) {
    return arm == TEST_ARM_FUSED ? "fused" : "unfused";
}

// ---------------------------------------------------------------------------------------------
// run + compare
// ---------------------------------------------------------------------------------------------

static const size_t GRAPH_SIZE = 64;

struct run_result {
    bool                 ok = false;
    bool                 skipped = false;
    ggml_type            type = GGML_TYPE_COUNT;
    std::vector<uint8_t> data;
    std::string          error;
    double               wall_ms = 0.0;
};

// inputs: name -> bytes; filled on the first run (CPU) and reused verbatim for the HTP run
static bool run_graph(ggml_backend_t backend, ggml_backend_dev_t dev, const test_graph & g,
                      std::vector<std::pair<std::string, std::vector<uint8_t>>> & inputs,
                      run_result & res, int reps) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(GRAPH_SIZE, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        res.error = "ggml_init failed";
        return false;
    }

    ggml_tensor * out = g.build(ctx, g.params);
    res.type = out->type;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        res.error = std::string("failed to allocate tensors on ") + ggml_backend_name(backend);
        ggml_free(ctx);
        return false;
    }

    // Force a controlled aliasing case: the graph output points at an input's storage. This is a
    // test-only setup; both the CPU reference and the HTP arm run the same graph.
    if (g.alias_output_to) {
        ggml_tensor * alias = nullptr;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t != out && strcmp(t->name, g.alias_output_to) == 0) {
                alias = t;
                break;
            }
        }
        if (!alias) {
            res.error = std::string("missing alias input: ") + g.alias_output_to;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }
        // Same data pointer and therefore the same strides as the aliased input.
        out->data = alias->data;
    }

    // refuse unsupported graphs instead of letting the backend abort on an unsupported op
    if (dev) {
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (!ggml_backend_dev_supports_op(dev, node)) {
                res.skipped = true;
                res.error   = std::string("unsupported op on HTP: ") + ggml_op_desc(node);
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                return true;
            }
        }
    }

    // fill the graph inputs (op == NONE) with data shared by both runs
    {
        size_t idx = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            // Views are not independent inputs: their data belongs to view_src, which is filled
            // separately. Filling a view with its own generated bytes would diverge from the
            // backing tensor for aliasing tests such as concat_cont_overlap.
            if (t->op != GGML_OP_NONE || t->view_src != nullptr || ggml_nelements(t) == 0) {
                continue;
            }
            const std::string name = t->name;
            bool found = false;
            for (auto & kv : inputs) {
                if (kv.first == name) {
                    ggml_backend_tensor_set(t, kv.second.data(), 0, kv.second.size());
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::vector<uint8_t> data;
                gen_tensor_data(t, g.params.ties, g.params.row_values, data);
                ggml_backend_tensor_set(t, data.data(), 0, data.size());
                inputs.emplace_back(name, std::move(data));
            }
            idx++;
        }
        (void) idx;
    }

    const int64_t t0 = ggml_time_us();
    for (int r = 0; r < reps; r++) {
        ggml_backend_graph_compute(backend, gf);
    }
    const int64_t t1 = ggml_time_us();
    res.wall_ms = (double) (t1 - t0) / 1000.0 / (double) reps;

    res.data.resize(ggml_nbytes(out));
    ggml_backend_tensor_get(out, res.data.data(), 0, res.data.size());
    res.ok = true;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// map an i32 index output through a float input tensor.
// The output has shape (k, n_cols) laid out contiguously, and the source has shape (src_ne0, n_cols):
// index i belongs to column j = i / k, so the source element is src[j * src_ne0 + idx[i]].
static bool gather_values(const run_result & r, const std::vector<uint8_t> & src_bytes, int64_t k, int64_t src_ne0,
                          std::vector<float> & out, std::string & err) {
    if (r.data.empty() || r.data.size() % sizeof(int32_t) != 0) {
        err = "unexpected index output size";
        return false;
    }
    if (k <= 0 || src_ne0 <= 0) {
        err = "bad shape for value mapping";
        return false;
    }
    const int32_t * idx  = (const int32_t *) r.data.data();
    const size_t    n    = r.data.size() / sizeof(int32_t);
    const size_t    nsrc = src_bytes.size() / sizeof(float);
    const float *   src  = (const float *) src_bytes.data();
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        const size_t  col = i / (size_t) k;
        const int32_t ix  = idx[i];
        if (ix < 0 || (int64_t) ix >= src_ne0) {
            err = "index out of range in argsort output";
            return false;
        }
        const size_t si = col * (size_t) src_ne0 + (size_t) ix;
        if (si >= nsrc) {
            err = "index out of range in argsort output (column overflow)";
            return false;
        }
        out[i] = src[si];
    }
    return true;
}

static bool compare_results(const run_result & ref, const run_result & got, std::string & msg, double rel_tol = 0.0) {
    if (ref.data.size() != got.data.size()) {
        msg = "size mismatch";
        return false;
    }

    if (ref.type == GGML_TYPE_I32) {
        const int32_t * a = (const int32_t *) ref.data.data();
        const int32_t * b = (const int32_t *) got.data.data();
        const size_t n = ref.data.size() / sizeof(int32_t);
        for (size_t i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                char buf[128];
                snprintf(buf, sizeof(buf), "i32 mismatch at %zu (cpu=%d htp=%d)", i, a[i], b[i]);
                msg = buf;
                return false;
            }
        }
        msg = "exact";
        return true;
    }

    const float * a = (const float *) ref.data.data();
    const float * b = (const float *) got.data.data();
    const size_t n = ref.data.size() / sizeof(float);

    double max_abs = 0.0;
    double max_ref = 0.0;
    size_t max_i = 0;
    for (size_t i = 0; i < n; i++) {
        const double da = fabs((double) a[i] - (double) b[i]);
        if (da > max_abs) {
            max_abs = da;
            max_i   = i;
        }
        if (fabs((double) a[i]) > max_ref) {
            max_ref = fabs((double) a[i]);
        }
        if (std::isnan(b[i]) || std::isnan(a[i])) {
            char buf[128];
            snprintf(buf, sizeof(buf), "NaN at %zu", i);
            msg = buf;
            return false;
        }
    }
    double nmse = 0.0;
    {
        double num = 0.0;
        double den = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double d = (double) a[i] - (double) b[i];
            num += d * d;
            den += (double) a[i] * (double) a[i];
        }
        nmse = den > 0.0 ? num / den : num;
    }
    const double tol = rel_tol > 0.0 ? rel_tol * (max_ref > 0.0 ? max_ref : 1.0)
                                     : 1e-6 * (max_ref > 1.0 ? max_ref : 1.0);
    char buf[224];
    snprintf(buf, sizeof(buf), "max_abs=%.3g @%zu (rel %.2g, nmse %.2g) tol=%.3g", max_abs, max_i,
             max_ref > 0.0 ? max_abs / max_ref : max_abs, nmse, tol);
    msg = buf;
    return max_abs <= tol;
}

// ---------------------------------------------------------------------------------------------
// dump / diff (cross-process A/B: fused vs unfused must be bit-identical)
// ---------------------------------------------------------------------------------------------

struct dump_file {
    std::string          name;
    ggml_type            type = GGML_TYPE_COUNT;
    std::vector<uint8_t> data;
    int                  n_ops = -1;
    int                  fusion_hits = -1;
    bool                 expects_fusion = false;
    int                  expected_ops_unfused = -1;
    int                  expected_ops_fused = -1;
};

static bool write_dump(const char * path, const std::vector<dump_file> & dumps, int n_ops_submitted, test_arm arm) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    const uint32_t magic = 0x48465433u; // "HFT3"
    const uint32_t n     = (uint32_t) dumps.size();
    const uint32_t ops   = (uint32_t) n_ops_submitted;
    const uint32_t arm_u = (uint32_t) arm;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&n, sizeof(n), 1, f);
    fwrite(&ops, sizeof(ops), 1, f);
    fwrite(&arm_u, sizeof(arm_u), 1, f);
    for (const auto & d : dumps) {
        const uint32_t len           = (uint32_t) d.name.size();
        const uint32_t type          = (uint32_t) d.type;
        const uint64_t size          = (uint64_t) d.data.size();
        const int32_t  n_ops         = d.n_ops;
        const int32_t  fusion_hits   = d.fusion_hits;
        const uint32_t expects       = d.expects_fusion ? 1u : 0u;
        const int32_t  expected_unf  = d.expected_ops_unfused;
        const int32_t  expected_fus  = d.expected_ops_fused;
        fwrite(&len, sizeof(len), 1, f);
        fwrite(d.name.data(), 1, len, f);
        fwrite(&type, sizeof(type), 1, f);
        fwrite(&size, sizeof(size), 1, f);
        fwrite(d.data.data(), 1, d.data.size(), f);
        fwrite(&n_ops, sizeof(n_ops), 1, f);
        fwrite(&fusion_hits, sizeof(fusion_hits), 1, f);
        fwrite(&expects, sizeof(expects), 1, f);
        fwrite(&expected_unf, sizeof(expected_unf), 1, f);
        fwrite(&expected_fus, sizeof(expected_fus), 1, f);
    }
    fclose(f);
    return true;
}

static bool read_dump(const char * path, std::vector<dump_file> & dumps, int * n_ops_submitted, test_arm * arm) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    uint32_t magic = 0, n = 0, ops = 0, arm_u = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x48465433u ||
        fread(&n, sizeof(n), 1, f) != 1 ||
        fread(&ops, sizeof(ops), 1, f) != 1 ||
        fread(&arm_u, sizeof(arm_u), 1, f) != 1 ||
        arm_u > (uint32_t) TEST_ARM_FUSED) {
        fclose(f);
        return false;
    }
    if (n_ops_submitted) {
        *n_ops_submitted = (int) ops;
    }
    if (arm) {
        *arm = (test_arm) arm_u;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = 0, type = 0;
        uint32_t expects = 0;
        uint64_t size = 0;
        int32_t n_ops = -1, fusion_hits = -1, expected_unf = -1, expected_fus = -1;
        if (fread(&len, sizeof(len), 1, f) != 1) { fclose(f); return false; }
        dump_file d;
        d.name.resize(len);
        if (len && fread(&d.name[0], 1, len, f) != len) { fclose(f); return false; }
        if (fread(&type, sizeof(type), 1, f) != 1 || fread(&size, sizeof(size), 1, f) != 1) { fclose(f); return false; }
        d.type = (ggml_type) type;
        d.data.resize(size);
        if (size && fread(d.data.data(), 1, size, f) != size) { fclose(f); return false; }
        if (fread(&n_ops, sizeof(n_ops), 1, f) != 1 ||
            fread(&fusion_hits, sizeof(fusion_hits), 1, f) != 1 ||
            fread(&expects, sizeof(expects), 1, f) != 1 ||
            fread(&expected_unf, sizeof(expected_unf), 1, f) != 1 ||
            fread(&expected_fus, sizeof(expected_fus), 1, f) != 1) {
            fclose(f);
            return false;
        }
        d.n_ops = n_ops;
        d.fusion_hits = fusion_hits;
        d.expects_fusion = expects != 0;
        d.expected_ops_unfused = expected_unf;
        d.expected_ops_fused = expected_fus;
        dumps.push_back(std::move(d));
    }
    fclose(f);
    return true;
}

static bool compare_bytes_exact(const dump_file & a, const dump_file & b, std::string & msg) {
    if (a.type != b.type) {
        msg = std::string("type mismatch: ") + ggml_type_name(a.type) + " vs " + ggml_type_name(b.type);
        return false;
    }
    if (a.data.size() != b.data.size()) {
        msg = "size mismatch: " + std::to_string(a.data.size()) + " vs " + std::to_string(b.data.size());
        return false;
    }
    if (!a.data.empty() && memcmp(a.data.data(), b.data.data(), a.data.size()) != 0) {
        size_t i = 0;
        while (i < a.data.size() && a.data[i] == b.data[i]) {
            i++;
        }
        msg = "byte mismatch at " + std::to_string(i);
        return false;
    }
    msg = "exact bytes";
    return true;
}

static int do_diff(const char * path_a, const char * path_b) {
    std::vector<dump_file> a, b;
    int ops_a = -1, ops_b = -1;
    test_arm arm_a = TEST_ARM_UNFUSED;
    test_arm arm_b = TEST_ARM_UNFUSED;
    if (!read_dump(path_a, a, &ops_a, &arm_a)) {
        fprintf(stderr, "cannot read %s\n", path_a);
        return 1;
    }
    if (!read_dump(path_b, b, &ops_b, &arm_b)) {
        fprintf(stderr, "cannot read %s\n", path_b);
        return 1;
    }
    if (a.empty() || b.empty()) {
        fprintf(stderr, "empty case set: %zu vs %zu\n", a.size(), b.size());
        return 1;
    }
    if (arm_a == arm_b) {
        fprintf(stderr, "both dumps are from the %s arm\n", arm_name(arm_a));
        return 1;
    }
    if (a.size() != b.size()) {
        fprintf(stderr, "dump size mismatch: %zu vs %zu\n", a.size(), b.size());
        return 1;
    }

    const std::vector<dump_file> & unfused = arm_a == TEST_ARM_UNFUSED ? a : b;
    const std::vector<dump_file> & fused   = arm_a == TEST_ARM_FUSED   ? a : b;

    int n_fail = 0;
    int n_enforced = 0;
    printf("\n== cross-process diff: %s  vs  %s ==\n", path_a, path_b);
    printf("arms: A=%s, B=%s\n", arm_name(arm_a), arm_name(arm_b));
    if (ops_a >= 0 && ops_b >= 0) {
        printf("ops submitted to DSP: %d vs %d  (delta %+d)\n",
               ops_a, ops_b, ops_b - ops_a);
    }
    for (size_t i = 0; i < unfused.size(); i++) {
        const dump_file & u = unfused[i];
        const dump_file & f = fused[i];
        if (u.name != f.name) {
            printf("%-28s NAME-MISMATCH\n", u.name.c_str());
            n_fail++;
            continue;
        }
        std::string msg;
        bool ok = compare_bytes_exact(u, f, msg);

        std::string protocol;
        if (u.expects_fusion) {
            n_enforced++;
            const bool hits_ok = u.fusion_hits == 0 && f.fusion_hits > 0;
            const bool unfused_ops_ok = u.expected_ops_unfused < 0 || u.n_ops == u.expected_ops_unfused;
            const bool fused_ops_ok   = u.expected_ops_fused < 0 || f.n_ops == u.expected_ops_fused;
            if (!hits_ok) {
                protocol = "fusion-hit mismatch (" + std::to_string(u.fusion_hits) + "->" +
                           std::to_string(f.fusion_hits) + ")";
            } else if (!unfused_ops_ok || !fused_ops_ok) {
                protocol = "op-count mismatch";
            } else if (u.expected_ops_unfused >= 0 && u.expected_ops_fused >= 0) {
                protocol = "ops " + std::to_string(u.n_ops) + "->" + std::to_string(f.n_ops);
            }
            if (!protocol.empty() && protocol.compare(0, 4, "ops ") != 0) {
                ok = false;
            }
        }

        printf("%-28s type=%-4s %-28s %-16s hits=%d->%d %s\n", u.name.c_str(),
               ggml_type_name(u.type), msg.c_str(), protocol.c_str(), u.fusion_hits, f.fusion_hits,
               ok ? "IDENTICAL" : "FAIL");
        if (!ok) {
            n_fail++;
        }
    }
    if (n_enforced == 0) {
        fprintf(stderr, "no fusion/op-count assertions were present in the dumps\n");
        n_fail++;
    }
    printf("\n%s (protocol cases=%d)\n",
           n_fail == 0 ? "CROSS-PROCESS: PASS (identical outputs; fusion counts and op reductions verified)"
                       : "CROSS-PROCESS: FAIL",
           n_enforced);
    return n_fail == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------

static void usage(const char * prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --arm <fused|unfused>               expected fusion arm (required for HTP runs)\n");
    printf("  --graph <all|argsort|concat|wnorm|mmadd>  select graphs (default: all)\n");
    printf("  --ties <n>                          collapse f32 inputs to n distinct values (0 = random)\n");
    printf("  --reps <n>                          graph_compute repetitions for timing (default 1)\n");
    printf("  --device <name>                     HTP backend device name (default: first HTP device)\n");
    printf("  --dump <path>                       write HTP outputs for cross-process comparison\n");
    printf("  --verbose-ops                       print the ops each graph actually submitted\n");
    printf("  --diff <a> <b>                      compare two dump files (fused vs unfused arms)\n");
    printf("  --list                              list graphs and exit\n");
}

int main(int argc, char ** argv) {
    std::string graph_sel = "all";
    std::string device_name;
    test_arm arm = TEST_ARM_UNFUSED;
    bool arm_set = false;
    const char * dump_path = nullptr;
    int ties = 0;
    int reps = 1;
    bool verbose_ops = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--arm" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "fused") {
                arm = TEST_ARM_FUSED;
            } else if (value == "unfused") {
                arm = TEST_ARM_UNFUSED;
            } else {
                fprintf(stderr, "unknown arm: %s\n", value.c_str());
                return 1;
            }
            arm_set = true;
        } else if (a == "--graph" && i + 1 < argc) {
            graph_sel = argv[++i];
        } else if (a == "--ties" && i + 1 < argc) {
            ties = atoi(argv[++i]);
        } else if (a == "--reps" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (a == "--device" && i + 1 < argc) {
            device_name = argv[++i];
        } else if (a == "--dump" && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (a == "--diff" && i + 2 < argc) {
            const char * path_a = argv[++i];
            const char * path_b = argv[++i];
            return do_diff(path_a, path_b);
        } else if (a == "--verbose-ops") {
            verbose_ops = true;
        } else if (a == "--list") {
            graph_sel = "list";
        } else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    auto params_of = [](int64_t ne0, int64_t ne1, int64_t k, int ties_, bool row_values = false) {
        case_params p;
        p.ne0        = ne0;
        p.ne1        = ne1;
        p.k          = k;
        p.ties       = ties_;
        p.row_values = row_values;
        return p;
    };

    // rel_tol: f32 outputs are compared with a relative tolerance. The HTP matmul converts f32
    // weights to f16 tiles (htp_kparams: wtype = F16 for tiled f32), so a CPU f32 reference cannot
    // be matched to ~1e-6; 5e-3 relative is ~10x the observed difference.
    // value_key: i32 outputs are mapped through that input tensor before comparing, because sort
    // implementations order *equal* elements differently (HTP bitonic vs CPU std::sort).
    std::vector<test_graph> graphs = {
        // argsort + cont: production shape (256 experts, top-8). The TOP_K rewrite was rejected, so
        // these are correctness cases, not fusion-hit cases.
        { "argsort_topk_cont_256x4",   "",             build_argsort_topk_cont, params_of(256, 4, 8, ties), "logits", 0.0 },
        { "argsort_topk_cont_256x1",   "",             build_argsort_topk_cont, params_of(256, 1, 8, ties), "logits", 0.0 },
        { "argsort_topk_cont_256x4_t", "",             build_argsort_topk_cont, params_of(256, 4, 8, 1),    "logits", 0.0 },
        { "argsort_topk_cont_256x4_3t","",             build_argsort_topk_cont, params_of(256, 4, 8, 3),    "logits", 0.0 },
        { "argsort_topk_cont_2048x4",  "",             build_argsort_topk_cont, params_of(2048, 4, 8, 1),   "logits", 0.0 },
        // concat + cont (i32 expert id assembly): pure integer move, exact
        { "concat_cont_8x4",           "CONCAT+CONT",  build_concat_cont,       params_of(8, 4, 8, ties), nullptr, 0.0, nullptr, 2, 1 },
        { "concat_cont_8x1",           "CONCAT+CONT",  build_concat_cont,       params_of(8, 1, 8, ties), nullptr, 0.0, nullptr, 2, 1 },
        // CONT destination aliases concat's right input (same data pointer and stride): must stay
        // 2 ops in both arms and produce the same bytes as the unfused run.
        { "concat_cont_overlap_8x4",   "",             build_concat_cont_overlap, params_of(8, 4, 4, ties), nullptr, 0.0,
          "CONCAT+CONT", 2, 2, "b" },
        // MoE top-k weight normalisation chain: SUM_ROWS + CLAMP + DIV -> 1 op (op count 3 -> 1)
        { "moe_wnorm_k8_t4",           "SUM_ROWS+CLAMP+DIV", build_moe_wnorm, params_of(1, 4, 8, ties), nullptr, 5e-3, nullptr, 3, 1 },
        { "moe_wnorm_k8_t1",           "SUM_ROWS+CLAMP+DIV", build_moe_wnorm, params_of(1, 1, 8, ties), nullptr, 5e-3, nullptr, 3, 1 },
        { "moe_wnorm_view_start",      "",             build_moe_wnorm_shared_data_view, params_of(1, 1, 8, ties), nullptr, 5e-3,
          "SUM_ROWS+CLAMP+DIV" },
        { "moe_wnorm_divisor_view",    "",             build_moe_wnorm_divisor_view, params_of(1, 4, 8, ties, true), nullptr, 5e-3,
          "SUM_ROWS+CLAMP+DIV" },
        { "moe_wnorm_double_clamp",    "",             build_moe_wnorm_double_clamp, params_of(1, 1, 8, ties), nullptr, 5e-3,
          "SUM_ROWS+CLAMP+DIV" },
        // mul_mat + add (residual): correctness coverage; multi-row needs GGML_HEXAGON_MMADD_MULTIROW=1
        { "mul_mat_add_n1",            "",             build_mul_mat_add,       params_of(2048, 1,  2048, ties), nullptr, 5e-3 },
        { "mul_mat_add_n4",            "",             build_mul_mat_add,       params_of(2048, 4,  2048, ties), nullptr, 5e-3 },
        { "mul_mat_add_n5",            "",             build_mul_mat_add,       params_of(2048, 5,  2048, ties), nullptr, 5e-3 },
    };

    if (graph_sel == "list") {
        for (const auto & g : graphs) {
            printf("%-30s %s\n", g.name, g.fusion);
        }
        return 0;
    }
    if (graph_sel != "all" && graph_sel != "argsort" && graph_sel != "concat" &&
        graph_sel != "wnorm" && graph_sel != "mmadd") {
        fprintf(stderr, "unknown graph selection: %s\n", graph_sel.c_str());
        return 1;
    }
    if (!arm_set) {
        fprintf(stderr, "FAIL: --arm <fused|unfused> is required for an HTP run\n");
        return 1;
    }

    // print the effective configuration (these env vars are read when the backend is loaded, so
    // they can only be changed between processes -- see FUSION_TEST.md)
    const char * env_fusion = getenv("GGML_HEXAGON_OPFUSION");
    const char * env_mmadd  = getenv("GGML_HEXAGON_MMADD_MULTIROW");
    printf("== hexagon fusion acceptance test ==\n");
    printf("arm                        = %s\n", arm_name(arm));
    printf("GGML_HEXAGON_OPFUSION      = %s\n", env_fusion ? env_fusion : "(unset -> 1, all on)");
    printf("GGML_HEXAGON_MMADD_MULTIROW= %s\n", env_mmadd ? env_mmadd : "(unset -> 0, opt-in off)");
    printf("ties = %d, reps = %d\n\n", ties, reps);

    ggml_backend_load_all();
    ggml_log_set(fusion_log_cb, nullptr);

    ggml_backend_dev_t dev_htp = nullptr;
    if (!device_name.empty()) {
        dev_htp = ggml_backend_dev_by_name(device_name.c_str());
    } else {
        // This acceptance test is HTP-only; do not silently fall back to another GPU backend.
        dev_htp = ggml_backend_dev_by_name("HTP0");
        if (!dev_htp) {
            for (size_t r = 0; r < ggml_backend_reg_count() && !dev_htp; r++) {
                ggml_backend_reg_t reg = ggml_backend_reg_get(r);
                for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); d++) {
                    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, d);
                    if (strstr(ggml_backend_dev_name(dev), "HTP") != nullptr) {
                        dev_htp = dev;
                        break;
                    }
                }
            }
        }
    }
    if (!dev_htp) {
        fprintf(stderr, "FAIL: no HTP device found (loaded backends: %zu)\n", ggml_backend_reg_count());
        return 1;
    }
    if (strstr(ggml_backend_dev_name(dev_htp), "HTP") == nullptr) {
        fprintf(stderr, "FAIL: selected backend device '%s' is not an HTP device\n",
                ggml_backend_dev_name(dev_htp));
        return 1;
    }
    ggml_backend_t htp = ggml_backend_dev_init(dev_htp, nullptr);
    if (!htp) {
        fprintf(stderr, "FAIL: cannot init backend for device %s\n", ggml_backend_dev_name(dev_htp));
        return 1;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "FAIL: cannot init CPU backend\n");
        return 1;
    }

    printf("device = %s (%s), cpu = %s\n\n",
           ggml_backend_dev_name(dev_htp), ggml_backend_dev_description(dev_htp),
           ggml_backend_name(cpu));

    std::vector<dump_file> dumps;
    int n_pass = 0;
    int n_fail = 0;
    int n_skip = 0;
    int n_selected = 0;

    for (const auto & g : graphs) {
        const bool selected = (graph_sel == "all") || (graph_sel == "argsort" && strstr(g.name, "argsort")) ||
                              (graph_sel == "concat" && strstr(g.name, "concat")) ||
                              (graph_sel == "wnorm" && strstr(g.name, "moe_wnorm")) ||
                              (graph_sel == "mmadd" && strstr(g.name, "mul_mat"));
        if (!selected) {
            continue;
        }
        n_selected++;

        const bool expects_fusion = g.fusion && g.fusion[0] != '\0';
        const int hits_before = expects_fusion ? g_fuse_stats.count(g.fusion) : 0;
        const int forbidden_before = g.forbidden_fusion ? g_fuse_stats.count(g.forbidden_fusion) : 0;

        // CPU reference first: it defines the input data for both runs
        std::vector<std::pair<std::string, std::vector<uint8_t>>> inputs;
        run_result ref, got;
        if (!run_graph(cpu, nullptr, g, inputs, ref, 1) || !ref.ok) {
            printf("%-30s CPU-RUN-FAILED: %s\n", g.name, ref.error.c_str());
            n_fail++;
            continue;
        }

        const int ops_prev = g_fuse_stats.n_ops_submitted;
        const size_t ops_prev_seen = g_fuse_stats.ops_seen.size();
        std::vector<std::pair<std::string, std::vector<uint8_t>>> inputs2 = inputs;
        if (!run_graph(htp, dev_htp, g, inputs2, got, reps) || !got.ok) {
            printf("%-30s HTP-RUN-FAILED: %s\n", g.name, got.error.c_str());
            n_fail++;
            continue;
        }
        if (got.skipped) {
            printf("%-30s SKIPPED (%s)\n", g.name, got.error.c_str());
            if (expects_fusion || g.ops_unfused >= 0 || g.ops_fused >= 0) {
                n_fail++;
            } else {
                n_skip++;
            }
            continue;
        }

        std::string msg;
        bool ok = false;
        if (g.value_key) {
            // compare the *values* the returned indices point at (tie-order independent)
            std::vector<uint8_t> src_bytes;
            for (const auto & kv : inputs) {
                if (kv.first == g.value_key) {
                    src_bytes = kv.second;
                    break;
                }
            }
            std::vector<float> vref, vgot;
            std::string err;
            run_result rr = ref, rg = got;
            rr.type = GGML_TYPE_I32;
            rg.type = GGML_TYPE_I32;
            if (src_bytes.empty()) {
                msg = "missing value_key input";
            } else if (!gather_values(rr, src_bytes, g.params.k, g.params.ne0, vref, err) ||
                       !gather_values(rg, src_bytes, g.params.k, g.params.ne0, vgot, err)) {
                msg = err;
            } else {
                run_result fr, fg;
                fr.type = GGML_TYPE_F32; fg.type = GGML_TYPE_F32;
                fr.data.assign((const uint8_t *) vref.data(), (const uint8_t *) vref.data() + vref.size() * sizeof(float));
                fg.data.assign((const uint8_t *) vgot.data(), (const uint8_t *) vgot.data() + vgot.size() * sizeof(float));
                ok  = compare_results(fr, fg, msg, 0.0);
                msg = "values: " + msg;
            }
        } else {
            ok = compare_results(ref, got, msg, g.rel_tol);
        }

        const int hits  = expects_fusion ? g_fuse_stats.count(g.fusion) - hits_before : 0;
        const int forbidden_hits = g.forbidden_fusion
            ? g_fuse_stats.count(g.forbidden_fusion) - forbidden_before
            : 0;
        const int n_ops = g_fuse_stats.n_ops_submitted - ops_prev;
        if (forbidden_hits > 0) {
            if (!msg.empty()) {
                msg += "; ";
            }
            msg += "forbidden fusion " + std::string(g.forbidden_fusion) + " fired " +
                   std::to_string(forbidden_hits) + "x";
            ok = false;
        }
        if (expects_fusion) {
            const int expected_hits = arm == TEST_ARM_FUSED ? 1 : 0;
            if (hits != expected_hits) {
                if (!msg.empty()) {
                    msg += "; ";
                }
                msg += "expected " + std::to_string(expected_hits) + " " + g.fusion +
                       " hit(s), got " + std::to_string(hits);
                ok = false;
            }
        }
        const int expected_ops = arm == TEST_ARM_FUSED ? g.ops_fused : g.ops_unfused;
        if (expected_ops >= 0 && n_ops != expected_ops) {
            if (!msg.empty()) {
                msg += "; ";
            }
            msg += "expected " + std::to_string(expected_ops) + " submitted op(s), got " +
                   std::to_string(n_ops);
            ok = false;
        }

        printf("%-30s type=%-4s %-52s fused=%-2d ops=%-2d %6.2f ms  %s\n",
               g.name, ggml_type_name(got.type), msg.c_str(), hits, n_ops, got.wall_ms,
               ok ? "PASS" : "FAIL");
        if (expects_fusion) {
            const int expected_hits = arm == TEST_ARM_FUSED ? 1 : 0;
            printf("%-30s   expected fusion: %s, hits=%d (expected %d)\n",
                   "", g.fusion, hits, expected_hits);
        }
        if (verbose_ops) {
            printf("%-30s   submitted ops:", "");
            for (size_t i = ops_prev_seen; i < g_fuse_stats.ops_seen.size(); i++) {
                printf(" %s", g_fuse_stats.ops_seen[i].c_str());
            }
            printf("\n");
        }

        dump_file d;
        d.name = g.name;
        d.type = got.type;
        d.data = got.data;
        d.n_ops = n_ops;
        d.fusion_hits = hits;
        d.expects_fusion = expects_fusion;
        d.expected_ops_unfused = g.ops_unfused;
        d.expected_ops_fused = g.ops_fused;
        dumps.push_back(std::move(d));

        if (ok) {
            n_pass++;
        } else {
            n_fail++;
        }
    }

    if (n_selected == 0) {
        fprintf(stderr, "FAIL: graph selection '%s' selected no cases\n", graph_sel.c_str());
        n_fail++;
    }

    printf("\nops submitted to DSP (execute-op lines): %d\n", g_fuse_stats.n_ops_submitted);
    printf("  ^ per-case op counts above are checked against --arm and in --diff\n");
    printf("\nfusion hits seen: %d\n", g_fuse_stats.n_hits);
    {
        std::vector<std::string> uniq = g_fuse_stats.names;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        for (const auto & u : uniq) {
            printf("  %-24s x%d\n", u.c_str(), g_fuse_stats.count(u.c_str()));
        }
    }

    if (dump_path) {
        if (dumps.empty()) {
            printf("\ncannot write %s: no completed cases\n", dump_path);
            n_fail++;
        } else if (write_dump(dump_path, dumps, g_fuse_stats.n_ops_submitted, arm)) {
            printf("\nwrote %zu outputs to %s\n", dumps.size(), dump_path);
        } else {
            printf("\ncannot write %s\n", dump_path);
            n_fail++;
        }
    }

    printf("\nSUMMARY: %d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);

    ggml_backend_free(cpu);
    ggml_backend_free(htp);

    return n_fail == 0 ? 0 : 1;
}
