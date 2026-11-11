// `LLAMA_MOE_LIVE_UNION` 的准入决策单测（纯函数，不需要模型/HTP/链接 llama）。
//
// 编译 + 运行（host，任意 g++ ≥ C++11）：
//   g++ -std=c++11 -O2 -Wall -Wextra -o /tmp/test-moe-live-union tests/test-moe-live-union.cpp && /tmp/test-moe-live-union
//
// 覆盖的是「会做错决定」的那些边界：恰好够、差一格、差很多、负数/退化输入，
// 以及 docs/live-union.md 里写的那条不变量（avail + evictable = 本层可容纳量）。

#include "../src/llama-moe-live-union.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

using llama_moe::admission_decision;
using llama_moe::decide_admission;
using llama_moe::live_union_max_rows;
using verdict = llama_moe::admission_decision::verdict;

static int failures = 0;

static const char * name(verdict v) {
    return v == verdict::window ? "window" : v == verdict::evict ? "evict" : "bank";
}

static void expect(int32_t u, int32_t avail, int32_t evictable, verdict want, int32_t want_evict) {
    const admission_decision got = decide_admission(u, avail, evictable);
    const bool ok = got.v == want && (want != verdict::evict || got.evict == want_evict);
    if (!ok) {
        ++failures;
        std::printf("FAIL decide_admission(union=%d, avail=%d, evictable=%d) = %s/%d, want %s/%d\n",
                    (int) u, (int) avail, (int) evictable, name(got.v), (int) got.evict,
                    name(want), (int) want_evict);
        return;
    }
    std::printf("ok   decide_admission(union=%d, avail=%d, evictable=%d) = %s evict=%d\n",
                (int) u, (int) avail, (int) evictable, name(got.v), (int) got.evict);
}

int main() {
    // ── 任务里给出的验收例子 ──────────────────────────────────────────────
    expect(20, 16, 6, verdict::evict, 4);    // 淘汰 4 个 LRU 常驻即可
    expect(26, 16, 6, verdict::bank, 0);     // 差 10 > 可淘汰 6 -> 装不下

    // ── 边界：恰好够 / 差一格 ────────────────────────────────────────────
    expect(16, 16, 6, verdict::window, 0);   // 一格都不用淘汰
    expect(17, 16, 6, verdict::evict, 1);
    expect(22, 16, 6, verdict::evict, 6);    // 用完可淘汰池
    expect(23, 16, 6, verdict::bank, 0);     // 只差一格 -> 也不许硬撑

    // ── 退化输入：不应该出现负容量/负并集，出现了也要给出确定答案 ──────────
    expect(0, 0, 0, verdict::window, 0);     // 空批
    expect(1, 0, 0, verdict::bank, 0);       // 一格都没有
    expect(1, 0, 1, verdict::evict, 1);      // 全靠淘汰
    expect(8, 0, 8, verdict::evict, 8);
    expect(8, 0, 7, verdict::bank, 0);
    expect(-3, 5, 0, verdict::window, 0);    // union<0 夹成 0
    expect(3, -2, 3, verdict::evict, 3);     // avail<0 夹成 0
    expect(5, 2, -1, verdict::bank, 0);      // evictable<0 夹成 0

    // ── 真机 trace 的形状：W24/K8（物理 32 格），4 行 verify ────────────────
    // union p50 16 / p95 22 / max 24；空池 32、无 pin、预读在途 0 时
    expect(16, 32, 0, verdict::window, 0);
    expect(22, 32, 0, verdict::window, 0);
    expect(24, 32, 0, verdict::window, 0);
    // 一个**可实现**的占用组合：物理 32 格、8 格被在途预读写者占住 → 可用的常驻池 24 格；
    // 其中 8 格空着（avail 的空槽部分）、16 格是可按 LRU 淘汰的常驻（evictable）。
    // 所以 avail + evictable = 24 = 物理 32 - 在途 8（在途既不在 avail 也不在 evictable）。
    expect(16, 16, 8, verdict::window, 0);   // 8 空 + 8 淘汰 = 16，直接进
    expect(20, 16, 8, verdict::evict, 4);    // 需要腾 4 格
    expect(25, 16, 8, verdict::bank, 0);     // 超过 24（=32-8），装不下
    // 单 token（|U| <= top_k）：pins + 在途把可用格压到 7 时，必须判出需要让位/淘汰，
    // 不能像旧实现那样因为 n_rows==1 就跳过（见 llama-moe-slot.cpp 的准入调用点）。
    expect(8, 7, 0, verdict::bank, 0);       // 7 格可用、无可淘汰 -> 必须先 drain 再重判
    expect(8, 8, 0, verdict::window, 0);     // drain 之后重判通过
    expect(30, 24, 8, verdict::evict, 6);
    expect(32, 24, 8, verdict::evict, 8);    // 32 <= 24+8：最坏并集仍装得下（只是要淘汰）

    // ── 不变量：avail + evictable 就是本层可容纳量 ────────────────────────
    // 只要 union <= avail+evictable，判定就必须是 window 或 evict（绝不允许 bank）；
    // 反之 union > avail+evictable 必须是 bank。用穷举把这条钉死。
    for (int32_t avail = 0; avail <= 40; ++avail) {
        for (int32_t evictable = 0; evictable <= 16; ++evictable) {
            for (int32_t u = 0; u <= 80; ++u) {
                const auto d = decide_admission(u, avail, evictable);
                const bool fits = u <= avail + evictable;
                if (fits != (d.v != verdict::bank)) {
                    ++failures;
                    std::printf("FAIL invariant at union=%d avail=%d evictable=%d -> %s\n",
                                (int) u, (int) avail, (int) evictable, name(d.v));
                }
                if (d.v == verdict::evict && (d.evict != u - avail || d.evict <= 0 || d.evict > evictable)) {
                    ++failures;
                    std::printf("FAIL evict count at union=%d avail=%d evictable=%d -> %d\n",
                                (int) u, (int) avail, (int) evictable, (int) d.evict);
                }
            }
        }
    }
    std::printf("ok   invariant sweep: avail 0..40 × evictable 0..16 × union 0..80\n");

    // ── init 那一半：live_union_max_rows（静态可容纳性）──────────────────────
    // W24/K8（物理宽度 32）+ top-8 + 0 pin：4 行成立（4×8=32）；扣掉 1 格就不行。
    struct rows_case { int32_t want, per_row, n_expert, cap, want_rows; };
    const rows_case row_cases[] = {
        {4, 8, 256, 32, 4},   // 任务里的目标：W24/K8 能让 4 行进窗口
        {4, 8, 256, 31, 3},   // 少一格 -> 退回 3
        {4, 8, 256, 24, 3},   // 旧夹紧的 c=24 -> 3（这就是要修的那个）
        {4, 8, 256, 40, 4},   // 再宽也不会超过 rows_want
        {1, 8, 256, 32, 1},
        {8, 8, 256, 32, 4},   // rows_want 更大时仍按容量夹
        {4, 8, 256,  0, 1},   // 一格都没有 -> 最少留 1（1 行 = 单 token，由 top_k 保证）
        {4, 1, 256,  3, 3},
        {4, 300, 256, 200, 1},// per_row > n_expert：min 生效
        {4, 300, 256, 256, 4},// min(4×300, 256)=256 <= 256 -> 满 4 行
        // 图侧硬约束：批宽必须 < 32（mul_mat_id 寻址不到 .dec 视图），
        // 容量再大也只能到 31。31/32/64 三点覆盖边界。
        {31,  8, 256, 4096, 31},
        {32,  8, 256, 4096, 31},
        {64,  8, 256, 4096, 31},
    };
    for (const auto & c : row_cases) {
        const int32_t got = live_union_max_rows(c.want, c.per_row, c.n_expert, c.cap);
        if (got != c.want_rows) {
            ++failures;
            std::printf("FAIL live_union_max_rows(want=%d, per_row=%d, n_expert=%d, cap=%d) = %d, want %d\n",
                        (int) c.want, (int) c.per_row, (int) c.n_expert, (int) c.cap, (int) got, (int) c.want_rows);
        } else {
            std::printf("ok   live_union_max_rows(want=%d, per_row=%d, n_expert=%d, cap=%d) = %d\n",
                        (int) c.want, (int) c.per_row, (int) c.n_expert, (int) c.cap, (int) got);
        }
    }

    // 静态不变量穷举：只要 rows_want 装得下，返回值必须是 rows_want；
    // 否则返回值必须装得下（除非退到 1 行、而 1 行本身就装不下 —— 那是容量根本不够）。
    for (int32_t want = 1; want <= 8; ++want) {
        for (int32_t per_row = 1; per_row <= 8; ++per_row) {
            for (int32_t cap = 0; cap <= 80; ++cap) {
                const int32_t n_expert = 256;
                const int32_t r = live_union_max_rows(want, per_row, n_expert, cap);
                const int32_t worst = (int32_t) std::min<int64_t>((int64_t) r * per_row, n_expert);
                const bool fits = worst <= cap;
                if (r > 1 && !fits) {
                    ++failures;
                    std::printf("FAIL static invariant: want=%d per_row=%d cap=%d -> rows=%d worst=%d\n",
                                (int) want, (int) per_row, (int) cap, (int) r, (int) worst);
                }
                if (r != want) {
                    // 退让只能是因为「want 真的装不下」，而且 r 必须是**最大**可行值
                    // （r+1 必须装不下）—— 这两条一起才叫"夹到最大可行批宽"。
                    const int32_t want_worst = (int32_t) std::min<int64_t>((int64_t) want * per_row, n_expert);
                    const int32_t next_worst = (int32_t) std::min<int64_t>((int64_t) (r + 1) * per_row, n_expert);
                    if (want_worst <= cap || (r < want && next_worst <= cap)) {
                        ++failures;
                        std::printf("FAIL clamp shape: want=%d per_row=%d cap=%d -> rows=%d (want_worst=%d next_worst=%d)\n",
                                    (int) want, (int) per_row, (int) cap, (int) r, (int) want_worst, (int) next_worst);
                    }
                }
            }
        }
    }
    std::printf("ok   static invariant sweep: want 1..8 × per_row 1..8 × cap 0..80\n");

    if (failures) {
        std::printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("\nall live-union admission cases passed\n");
    return 0;
}
