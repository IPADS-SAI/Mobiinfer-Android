// R4a host 单测：预读读合并的规划/执行（不需要模型/后端/HTP）。
//
// 编译 + 运行（host，任意 g++ >= C++11）：
//   g++ -std=c++11 -O2 -Wall -Wextra -o /tmp/test-moe-prefetch-merge tests/test-moe-prefetch-merge.cpp
//   /tmp/test-moe-prefetch-merge
//
// 覆盖：空 batch / 单专家 / 连续 run / 间隔 run / run > max / 跨层拒绝 /
// 排序稳定性 / 部分读（含异常）整段回退 / MERGE_PREFETCH=0|1|2 语义。

#include "../src/llama-moe-prefetch-merge.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace pm = llama_moe::prefetch_merge;

static int failures = 0;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __func__, __LINE__, #c); ++failures; } } while (false)

static pm::item item(int layer, int expert, int slot, int pos) {
    pm::item it;
    it.layer = layer; it.expert = expert; it.slot = slot; it.pos = pos;
    return it;
}

static std::vector<pm::item> make_items(const std::vector<int> & positions, int layer = 3,
                                        int first_slot = 0) {
    std::vector<pm::item> items;
    for (size_t i = 0; i < positions.size(); ++i) {
        items.push_back(item(layer, 100 + (int) i, first_slot + (int) i, positions[i]));
    }
    return items;
}

static void expect_mode(const char * text, int want) {
    CHECK(pm::parse_mode(text) == want);
}

static void test_modes() {
    expect_mode(nullptr, 0);
    expect_mode("", 0);
    expect_mode("0", 0);
    expect_mode("1", 1);
    expect_mode("2", 2);
    expect_mode("3", 2);      // 夹到 2
    expect_mode("-5", 0);     // 负数夹到 0
    expect_mode("junk", 0);   // atoi 语义 = 0
    std::printf("ok   parse_mode\n");
}

static void test_empty() {
    const auto p = pm::plan_batch({}, 2, 4, true);
    CHECK(p.order.empty());
    CHECK(p.units.empty());
    CHECK(!p.mixed_layers);
    CHECK(p.projection.runs == 0 && p.projection.syscalls == 0);
    std::printf("ok   empty batch\n");
}

static void test_single() {
    const auto items = make_items({17});
    for (int mode = 0; mode <= 2; ++mode) {
        const auto p = pm::plan_batch(items, mode, 4, true);
        CHECK(p.units.size() == 1);
        CHECK(p.units[0].count == 1);
        CHECK(!p.units[0].merged);
        CHECK(p.order.size() == 1 && p.order[0] == 0);
        if (mode == 0) {
            CHECK(p.projection.experts == 0);   // mode=0：计数 disabled
        } else {
            CHECK(p.projection.experts == 1);
            CHECK(p.projection.runs == 1);
            CHECK(p.projection.syscalls == 1);
        }
    }
    std::printf("ok   single expert\n");
}

static void test_consecutive() {
    const auto items = make_items({10, 11, 12, 13});
    const auto p = pm::plan_batch(items, 2, 4, true);
    CHECK(p.units.size() == 1);
    CHECK(p.units[0].merged);
    CHECK(p.units[0].count == 4);
    CHECK(p.units[0].slots_consecutive);
    CHECK(p.projection.runs == 1);
    CHECK(p.projection.syscalls == 1);
    CHECK(p.projection.zerocopy_runs == 1);
    for (int i = 0; i < 4; ++i) CHECK(p.order[(size_t) i] == i);
    std::printf("ok   consecutive run\n");
}

static void test_gapped() {
    // 输入故意乱序：12,10,13,11 -> 排序后两个连续段 {10,11,12,13} 其实是一段。
    const auto four = make_items({12, 10, 13, 11});
    const auto p4 = pm::plan_batch(four, 2, 4, true);
    CHECK(p4.units.size() == 1 && p4.units[0].count == 4);
    CHECK(p4.projection.runs == 1 && p4.projection.syscalls == 1);
    // 真正的间隔：10,11 与 20,21。
    const auto gap = make_items({10, 11, 20, 21});
    const auto p = pm::plan_batch(gap, 2, 4, true);
    CHECK(p.units.size() == 2);
    CHECK(p.units[0].count == 2 && p.units[0].merged);
    CHECK(p.units[1].count == 2 && p.units[1].merged);
    CHECK(p.projection.runs == 2);
    CHECK(p.projection.syscalls == 2);
    std::printf("ok   gapped run\n");
}

static void test_over_max() {
    std::vector<int> positions;
    for (int i = 0; i < 10; ++i) positions.push_back(30 + i);
    const auto items = make_items(positions);
    const auto p = pm::plan_batch(items, 2, 4, true);
    CHECK(p.units.size() == 3);          // 4 + 4 + 2
    CHECK(p.units[0].count == 4);
    CHECK(p.units[1].count == 4);
    CHECK(p.units[2].count == 2);
    CHECK(p.projection.runs == 1);       // 一个连续段
    CHECK(p.projection.syscalls == 3);   // ceil(10/4)
    CHECK(p.projection.max_run == 10);

    const auto one = pm::plan_batch(items, 2, 64, true);
    CHECK(one.units.size() == 1 && one.units[0].count == 10);  // 上限 64 -> 不切

    std::printf("ok   run > max\n");
}

static void test_cross_layer() {
    auto items = make_items({0, 1, 2, 3});
    items[2].layer = 4;
    const auto p = pm::plan_batch(items, 2, 4, true);
    CHECK(p.mixed_layers);
    CHECK(p.fallback);
    CHECK(p.units.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(!p.units[(size_t) i].merged);
        CHECK(p.order[(size_t) i] == i);   // 保持输入顺序逐专家
    }
    CHECK(p.projection.runs == 0);         // 跨层拒绝，不产出合并投影
    std::printf("ok   cross-layer rejected\n");
}

static void test_mode_semantics() {
    // 输入乱序；mode=0/1 的实际执行顺序必须保持输入顺序。
    const auto items = make_items({9, 3, 8, 2});
    const auto p0 = pm::plan_batch(items, 0, 4, true);
    CHECK(p0.units.size() == 4);
    for (int i = 0; i < 4; ++i) CHECK(p0.order[(size_t) i] == i);
    CHECK(p0.projection.runs == 0);        // mode=0：计数为 0 / disabled
    CHECK(p0.projection.syscalls == 0);

    const auto p1 = pm::plan_batch(items, 1, 4, false);
    CHECK(p1.units.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(!p1.units[(size_t) i].merged);
        CHECK(p1.order[(size_t) i] == i);   // 不改变实际读顺序
    }
    CHECK(p1.projection.experts == 4);
    CHECK(p1.projection.runs == 2);        // 位置 2,3 与 8,9 -> 2 段
    CHECK(p1.projection.syscalls == 2);

    // mode=2 但调用方禁止合并（io_budget / dummy / 不支持的布局）：
    // 实际仍逐专家，且顺序不变；投影照算。
    const auto p2 = pm::plan_batch(items, 2, 4, false);
    CHECK(p2.fallback);
    CHECK(p2.units.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(!p2.units[(size_t) i].merged);
        CHECK(p2.order[(size_t) i] == i);
    }
    CHECK(p2.projection.runs == 2);
    CHECK(p2.projection.syscalls == 2);

    const auto p3 = pm::plan_batch(items, 2, 4, true);
    CHECK(p3.units.size() == 2);
    CHECK(p3.units[0].count == 2 && p3.units[1].count == 2);
    CHECK(p3.order[0] == 3 && p3.order[1] == 1);   // pos 2,3
    CHECK(p3.order[2] == 2 && p3.order[3] == 0);   // pos 8,9
    std::printf("ok   mode 0/1/2 semantics\n");
}

static void test_nonconsecutive_slots() {
    // position 连续但 slot 不连续：R4a 的零拷贝 iovec 可以处理（文件顺序才是硬约束），
    // 规划仍给出 merged，并用 slots_consecutive 标记实际路径。
    auto items = make_items({40, 41, 42}, 7, 0);
    items[1].slot = 9;
    items[2].slot = 20;
    const auto p = pm::plan_batch(items, 2, 4, true);
    CHECK(p.units.size() == 1 && p.units[0].merged);
    CHECK(!p.units[0].slots_consecutive);
    CHECK(p.projection.zerocopy_runs == 0);
    CHECK(p.projection.scatter_runs == 1);

    // slot 全是 -1（还不认识落点）时视为不可判定 -> scatter，不冒充 zero-copy。
    const auto unknown = make_items({40, 41}, 7, -1);
    const auto pu = pm::plan_batch(unknown, 2, 4, true);
    CHECK(pu.units.size() == 1 && pu.units[0].merged);
    CHECK(!pu.units[0].slots_consecutive);
    CHECK(pu.projection.scatter_runs == 1);
    std::printf("ok   non-consecutive slots\n");
}

static void test_stable_sort() {
    // 相同 position（理论上不该有，但 data race/重复预测可能出现）保持输入顺序。
    const auto items = make_items({5, 5, 4, 5}, 1, 0);
    const auto p = pm::plan_batch(items, 2, 4, true);
    CHECK(p.order.size() == 4);
    CHECK(p.order[0] == 2);            // pos 4
    CHECK(p.order[1] == 0);            // 三个 pos 5 按输入顺序
    CHECK(p.order[2] == 1);
    CHECK(p.order[3] == 3);
    std::printf("ok   stable sort\n");
}

static void test_execute_success() {
    const auto items = make_items({0, 1, 2, 3});
    const auto p = pm::plan_batch(items, 2, 4, true);
    int run_calls = 0, one_calls = 0, merged_calls = 0;
    const auto s = pm::execute_plan(p,
        [&](const std::vector<int> &, const pm::unit & u) { ++run_calls; return u.count == 4; },
        [&](int) { ++one_calls; },
        [&](const std::vector<int> &, const pm::unit &) { ++merged_calls; });
    CHECK(run_calls == 1 && one_calls == 0 && merged_calls == 1);
    CHECK(s.merged_runs == 1 && s.fallback_runs == 0);
    CHECK(s.experts == 4 && s.syscalls == 1 && s.syscalls_saved() == 3);
    CHECK(s.zerocopy_runs == 1 && s.scatter_runs == 0);
    std::printf("ok   execute merged\n");
}

static void test_partial_read_fallback_data() {
    // 合成 buffer：合并读先写了第 0 个专家就「短读失败」；回退必须把
    // 四个专家全部重新读进各自 slot，最终 buffer 必须逐字节等于预期。
    const auto items = make_items({0, 1, 2, 3});
    const auto p = pm::plan_batch(items, 2, 4, true);
    std::vector<int> buf(4, -1);
    int one_calls = 0, merged_calls = 0;

    const auto s = pm::execute_plan(p,
        [&](const std::vector<int> & order, const pm::unit & u) {
            // 模拟部分读：只落第一个 item，然后返回 false（等价 preadv 短读）。
            buf[(size_t) items[(size_t) order[(size_t) u.first]].slot] =
                items[(size_t) order[(size_t) u.first]].expert;
            return false;
        },
        [&](int index) {
            ++one_calls;
            buf[(size_t) items[(size_t) index].slot] = items[(size_t) index].expert;
        },
        [&](const std::vector<int> &, const pm::unit &) { ++merged_calls; });

    CHECK(one_calls == 4);          // 整段回退，不是只重读第一个
    CHECK(merged_calls == 0);
    CHECK(s.fallback_runs == 1 && s.merged_runs == 0);
    CHECK(s.experts == 4 && s.syscalls == 4 && s.syscalls_saved() == 0);
    for (int i = 0; i < 4; ++i) CHECK(buf[(size_t) i] == items[(size_t) i].expert);
    std::printf("ok   partial read -> whole-run fallback (data)\n");
}

static void test_read_exception_fallback() {
    const auto items = make_items({0, 1, 2, 3});
    const auto p = pm::plan_batch(items, 2, 4, true);
    int one_calls = 0, merged_calls = 0;
    const auto s = pm::execute_plan(p,
        [&](const std::vector<int> &, const pm::unit &) -> bool {
            throw std::runtime_error("EINVAL");
        },
        [&](int) { ++one_calls; },
        [&](const std::vector<int> &, const pm::unit &) { ++merged_calls; });
    CHECK(one_calls == 4 && merged_calls == 0);
    CHECK(s.fallback_runs == 1 && s.merged_runs == 0);
    std::printf("ok   preadv exception -> whole-run fallback\n");
}

static void test_partial_then_ok_units() {
    // 两个 run：第一个失败、第二个成功；失败 run 的每个专家都要逐读，
    // 成功 run 只走一次 read_run + on_merged。
    const auto items = make_items({0, 1, 10, 11});
    const auto p = pm::plan_batch(items, 2, 4, true);
    int run_calls = 0, one_calls = 0, merged_calls = 0;
    const auto s = pm::execute_plan(p,
        [&](const std::vector<int> &, const pm::unit & u) {
            ++run_calls;
            return u.first != p.units[0].first;   // 第一个 run 失败
        },
        [&](int) { ++one_calls; },
        [&](const std::vector<int> &, const pm::unit &) { ++merged_calls; });
    CHECK(run_calls == 2);
    CHECK(one_calls == 2);          // 只回退失败的 run
    CHECK(merged_calls == 1);
    CHECK(s.merged_runs == 1 && s.fallback_runs == 1);
    CHECK(s.experts == 4 && s.syscalls == 3 && s.syscalls_saved() == 1);
    std::printf("ok   per-run fallback isolation\n");
}

static void test_single_no_merge_call() {
    const auto items = make_items({7});
    const auto p = pm::plan_batch(items, 2, 4, true);
    int run_calls = 0, one_calls = 0;
    const auto s = pm::execute_plan(p,
        [&](const std::vector<int> &, const pm::unit &) { ++run_calls; return true; },
        [&](int) { ++one_calls; },
        [&](const std::vector<int> &, const pm::unit &) {});
    CHECK(run_calls == 0 && one_calls == 1);
    CHECK(s.experts == 1 && s.syscalls == 1 && s.syscalls_saved() == 0);
    std::printf("ok   single expert never calls read_run\n");
}

int main() {
    test_modes();
    test_empty();
    test_single();
    test_consecutive();
    test_gapped();
    test_over_max();
    test_cross_layer();
    test_mode_semantics();
    test_nonconsecutive_slots();
    test_stable_sort();
    test_execute_success();
    test_partial_read_fallback_data();
    test_read_exception_fallback();
    test_partial_then_ok_units();
    test_single_no_merge_call();

    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("all prefetch-merge tests passed\n");
    return 0;
}
