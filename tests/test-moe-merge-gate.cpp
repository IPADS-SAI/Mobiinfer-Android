// R4a 合并门控真值表 host 单测（不需要模型/后端/HTP）。
//
// 编译 + 运行（host，任意 g++ >= C++11）：
//   g++ -std=c++11 -O2 -Wall -Wextra -o /tmp/test-moe-merge-gate tests/test-moe-merge-gate.cpp
//   /tmp/test-moe-merge-gate
//
// 覆盖：
//   * 162 组 `MERGE_READS(0/1/2) × MERGE_PREFETCH(0/1/2) × io_budget(0/1) ×
//     read_mode(0/1/2) × managed{none,all,mixed}`，n>1 固定；
//   * 期望值由本文件**独立实现**的参考布尔式给出（不调用被测 decide() 生成期望）；
//   * reason 优先级：env_off < io_budget < read_mode < mixed_managed < single_item < ok；
//   * n==1 的另一组 162 组（单专家不排序、不合并、不投影）；
//   * `MERGE_READS=1` 在 demand 侧是纯 no-op（不排序、不合并）；
//   * 越界 mode 的夹紧语义。
//
// 刻意保留的现状细节（见 src/llama-moe-merge-gate.h 顶部注释）：
//   - sort_demand 不看 read_mode（read_mode!=0 时仍排序/切 run，再逐专家回退）；
//   - project_prefetch 带 n>1（改动前 n==1 早退、不投影，保持旧计数逐字节可比）。

#include "../src/llama-moe-merge-gate.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mg = llama_moe::merge_gate;

static int failures = 0;

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __func__, __LINE__, #c); ++failures; } } while (false)

struct expected {
    bool sort_demand;
    bool merge_demand;
    bool project_prefetch;
    bool merge_prefetch;
    const char * reason;
};

// 独立参考实现：只用 spec 的布尔式与固定优先级，不调用 decide()。
static expected reference(int reads, int prefetch, int io_budget, int read_mode,
                          int n, bool any_managed, bool all_managed) {
    const bool multi = n > 1;
    const bool managed_ok = !any_managed || all_managed;
    expected e;
    e.sort_demand  = reads == 2 && io_budget == 0 && multi && managed_ok;
    e.merge_demand = reads == 2 && io_budget == 0 && read_mode == 0 && multi && managed_ok;
    e.project_prefetch = prefetch >= 1 && multi;
    e.merge_prefetch   = prefetch == 2 && io_budget == 0 && read_mode == 0 && multi;

    if (reads < 2 && prefetch < 2) {
        e.reason = "env_off";
    } else if (io_budget > 0) {
        e.reason = "io_budget";
    } else if (read_mode != 0) {
        e.reason = "read_mode";
    } else if (reads == 2 && multi && any_managed && !all_managed) {
        e.reason = "mixed_managed";
    } else if (!multi) {
        e.reason = "single_item";
    } else {
        e.reason = "ok";
    }
    return e;
}

static mg::inputs make_inputs(int reads, int prefetch, int io_budget, int read_mode,
                              int n, const char * managed) {
    mg::inputs in;
    in.merge_reads = reads;
    in.merge_prefetch = prefetch;
    in.io_budget = io_budget;
    in.read_mode = read_mode;
    in.n = n;
    if (std::strcmp(managed, "none") == 0) {
        in.any_managed = false;
        in.all_managed = true;
    } else if (std::strcmp(managed, "all") == 0) {
        in.any_managed = true;
        in.all_managed = true;
    } else { // mixed
        in.any_managed = true;
        in.all_managed = false;
    }
    return in;
}

static void compare(const mg::inputs & in, const expected & e, int reads, int prefetch,
                    int io_budget, int read_mode, int n, const char * managed) {
    const mg::decision d = mg::decide(in);
    if (d.sort_demand != e.sort_demand || d.merge_demand != e.merge_demand ||
        d.project_prefetch != e.project_prefetch || d.merge_prefetch != e.merge_prefetch ||
        std::strcmp(mg::reason_name(d.why), e.reason) != 0) {
        std::fprintf(stderr,
            "FAIL combo reads=%d prefetch=%d budget=%d read_mode=%d n=%d managed=%s\n"
            "  got  sort=%d demand=%d project=%d pfmerge=%d reason=%s\n"
            "  want sort=%d demand=%d project=%d pfmerge=%d reason=%s\n",
            reads, prefetch, io_budget, read_mode, n, managed,
            (int) d.sort_demand, (int) d.merge_demand, (int) d.project_prefetch,
            (int) d.merge_prefetch, mg::reason_name(d.why),
            (int) e.sort_demand, (int) e.merge_demand, (int) e.project_prefetch,
            (int) e.merge_prefetch, e.reason);
        ++failures;
    }
}

static void test_truth_table_n_gt_1() {
    const char * managed_names[] = { "none", "all", "mixed" };
    int combos = 0;
    for (int reads = 0; reads <= 2; ++reads) {
        for (int prefetch = 0; prefetch <= 2; ++prefetch) {
            for (int budget = 0; budget <= 1; ++budget) {
                for (int read_mode = 0; read_mode <= 2; ++read_mode) {
                    for (int m = 0; m < 3; ++m) {
                        const int n = 8;   // n>1 固定
                        const mg::inputs in = make_inputs(reads, prefetch, budget, read_mode,
                                                          n, managed_names[m]);
                        compare(in, reference(reads, prefetch, budget, read_mode, n,
                                              in.any_managed, in.all_managed),
                                reads, prefetch, budget, read_mode, n, managed_names[m]);
                        ++combos;
                    }
                }
            }
        }
    }
    CHECK(combos == 162);
    std::printf("ok   162 combos (n>1)\n");
}

static void test_truth_table_n_eq_1() {
    const char * managed_names[] = { "none", "all", "mixed" };
    int combos = 0;
    for (int reads = 0; reads <= 2; ++reads) {
        for (int prefetch = 0; prefetch <= 2; ++prefetch) {
            for (int budget = 0; budget <= 1; ++budget) {
                for (int read_mode = 0; read_mode <= 2; ++read_mode) {
                    for (int m = 0; m < 3; ++m) {
                        const int n = 1;
                        const mg::inputs in = make_inputs(reads, prefetch, budget, read_mode,
                                                          n, managed_names[m]);
                        const expected e = reference(reads, prefetch, budget, read_mode, n,
                                                     in.any_managed, in.all_managed);
                        // 单专家：不排序、不合并、不投影（改动前 n==1 早退）。
                        CHECK(!e.sort_demand && !e.merge_demand);
                        CHECK(!e.project_prefetch && !e.merge_prefetch);
                        compare(in, e, reads, prefetch, budget, read_mode, n, managed_names[m]);
                        ++combos;
                    }
                }
            }
        }
    }
    CHECK(combos == 162);
    std::printf("ok   162 combos (n==1)\n");
}

static void test_reason_priority() {
    // env_off 压过所有禁用条件。
    {
        const auto d = mg::decide(make_inputs(0, 0, 1, 1, 8, "mixed"));
        CHECK(d.why == mg::reason::env_off);
    }
    // io_budget 压过 read_mode / mixed_managed / single_item。
    {
        const auto d = mg::decide(make_inputs(2, 2, 1, 1, 8, "mixed"));
        CHECK(d.why == mg::reason::io_budget);
    }
    {
        const auto d = mg::decide(make_inputs(2, 0, 1, 0, 1, "none"));
        CHECK(d.why == mg::reason::io_budget);
    }
    // read_mode 压过 mixed_managed / single_item。
    {
        const auto d = mg::decide(make_inputs(2, 2, 0, 1, 8, "mixed"));
        CHECK(d.why == mg::reason::read_mode);
    }
    {
        const auto d = mg::decide(make_inputs(0, 2, 0, 2, 1, "none"));
        CHECK(d.why == mg::reason::read_mode);
    }
    // mixed_managed 压过 single_item（n>1 时 mixed 才会出现）。
    {
        const auto d = mg::decide(make_inputs(2, 0, 0, 0, 8, "mixed"));
        CHECK(d.why == mg::reason::mixed_managed);
    }
    // 单专家。
    {
        const auto d = mg::decide(make_inputs(2, 2, 0, 0, 1, "none"));
        CHECK(d.why == mg::reason::single_item);
    }
    // 全允许。
    {
        const auto d = mg::decide(make_inputs(2, 2, 0, 0, 8, "none"));
        CHECK(d.why == mg::reason::ok);
    }
    // 优先级数值必须严格递增，日志消费方可直接比较。
    CHECK(mg::reason_priority(mg::reason::env_off) <
          mg::reason_priority(mg::reason::io_budget));
    CHECK(mg::reason_priority(mg::reason::io_budget) <
          mg::reason_priority(mg::reason::read_mode));
    CHECK(mg::reason_priority(mg::reason::read_mode) <
          mg::reason_priority(mg::reason::mixed_managed));
    CHECK(mg::reason_priority(mg::reason::mixed_managed) <
          mg::reason_priority(mg::reason::single_item));
    CHECK(mg::reason_priority(mg::reason::single_item) <
          mg::reason_priority(mg::reason::ok));
    std::printf("ok   reason priority\n");
}

static void test_current_semantics_quirks() {
    // read_mode!=0 时仍排序/切 run（sort_demand=true），但不合并。
    {
        const auto d = mg::decide(make_inputs(2, 0, 0, 1, 8, "none"));
        CHECK(d.sort_demand);
        CHECK(!d.merge_demand);
        CHECK(d.why == mg::reason::read_mode);
    }
    // managed 混合：不排序也不合并。
    {
        const auto d = mg::decide(make_inputs(2, 2, 0, 0, 8, "mixed"));
        CHECK(!d.sort_demand);
        CHECK(!d.merge_demand);
        CHECK(d.merge_prefetch);   // prefetch 不看 managed
    }
    // io_budget 同时禁用 demand 与 prefetch，但投影仍保留（供对照）。
    {
        const auto d = mg::decide(make_inputs(2, 2, 1, 0, 8, "none"));
        CHECK(!d.sort_demand);
        CHECK(!d.merge_demand);
        CHECK(d.project_prefetch);
        CHECK(!d.merge_prefetch);
    }
    // read_mode 同样禁用两条合并，投影保留。
    {
        const auto d = mg::decide(make_inputs(2, 2, 0, 2, 8, "none"));
        CHECK(!d.merge_demand);
        CHECK(d.project_prefetch);
        CHECK(!d.merge_prefetch);
    }
    // MERGE_READS=1 是纯 no-op：arch 形状统计不算门控，读法不变。
    {
        const auto d = mg::decide(make_inputs(1, 0, 0, 0, 8, "none"));
        CHECK(!d.sort_demand);
        CHECK(!d.merge_demand);
        CHECK(d.why == mg::reason::env_off);
    }
    // 只有 prefetch 处于 merge 模式：reason 不应是 env_off。
    {
        const auto d = mg::decide(make_inputs(0, 2, 0, 0, 8, "none"));
        CHECK(d.merge_prefetch);
        CHECK(d.why == mg::reason::ok);
    }
    // 越界 mode 夹紧（与 slot.cpp 的 merge_reads_mode()/parse_mode() 一致）。
    {
        const auto d = mg::decide(make_inputs(3, -1, 0, 0, 8, "none"));
        CHECK(d.merge_demand);
        CHECK(!d.merge_prefetch);
        CHECK(d.why == mg::reason::ok);
    }
    std::printf("ok   current semantics quirks\n");
}

int main() {
    test_truth_table_n_gt_1();
    test_truth_table_n_eq_1();
    test_reason_priority();
    test_current_semantics_quirks();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all merge-gate tests passed\n");
    return 0;
}
