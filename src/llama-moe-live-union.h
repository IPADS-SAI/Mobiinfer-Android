#pragma once

#include <algorithm>
#include <cstdint>

// `LLAMA_MOE_LIVE_UNION` 的**准入决策**：纯函数、无状态、无 I/O。
//
// 为什么单独抽出来：这是整个开关唯一"会做错决定"的地方。把它写成纯函数，
// 就能在 host 上用最小的代价把边界（恰好够 / 差一格 / 差很多）全部覆盖，
// 不必起 HTP、不必加载模型。
//
// 三个输入的语义（都由 llama-moe-slot.cpp 在持层锁时统计，见 live_union_view）：
//
//   union     —— 本批**该层**实际用到的不重复专家数 |U|（去重后）。
//                上界是 min(rows × top_k, n_expert)；去重只会让它更小。
//   avail     —— 不淘汰任何东西时，|U| 里**能直接落位**的专家数：
//                = 空槽数 + 已经驻留着 U 成员的槽数。
//                它排除了：在途写者占住的槽（预读/等待中的 demand）、
//                受保护且不属于本批的常驻槽（pin/lease/persistent）。
//   evictable —— 还能腾出来的常驻槽数：常驻、不属于本批、不受保护
//                （即 LRU 可以淘汰的那些；在途槽不算，它们不能被淘汰）。
//
// 判定：
//   |U| <= avail                  -> window（不用淘汰）
//   |U| <= avail + evictable      -> evict=need（先淘汰 need 个再进窗口）
//   否则                          -> bank（本层装不下）
//
// 注意 avail + evictable 恰好等于"本层最多能同时容纳多少个不同专家"，
// 所以第三条判据就是 |U| 超过物理可容纳量 —— 不是启发式。
namespace llama_moe {

// 图侧硬约束：窗口批宽必须 **< 32**。再大时 mul_mat_id 走的是 prefill 路径，
// 寻址不到每层 `.dec` 视图（见 llama-moe-slot.cpp 中 window_max_rows 的注释）。
// 旧实现只在注释里写了这条约束、没有在代码里夹；本函数把它变成可测的不变量。
inline constexpr int32_t kWindowMaxRowsGraphLimit = 31;

struct admission_decision {
    enum class verdict {
        window,   // 直接进窗口
        evict,    // 淘汰 evict 个 LRU 常驻槽后进窗口
        bank,     // 装不下
    };
    verdict v = verdict::bank;
    int32_t evict = 0;

    admission_decision() = default;
    admission_decision(verdict v_, int32_t evict_) : v(v_), evict(evict_) {}

    bool operator==(const admission_decision & o) const { return v == o.v && evict == o.evict; }
};

inline admission_decision decide_admission(int32_t union_size, int32_t avail, int32_t evictable) {
    if (union_size < 0) union_size = 0;
    if (avail < 0)      avail = 0;
    if (evictable < 0)  evictable = 0;
    if (union_size <= avail) {
        return {admission_decision::verdict::window, 0};
    }
    const int32_t need = union_size - avail;
    if (need <= evictable) {
        return {admission_decision::verdict::evict, need};
    }
    return {admission_decision::verdict::bank, 0};
}

// init 段的那一半：把窗口批宽夹到「**最坏**并集仍装得进 cap_min」的最大行数。
//
// `cap_min` = min over enabled layers of (demand_capacity(il) - pins(il))；
// `per_row` = top_k；`n_expert` 用来封顶（全库只有这么多专家）。
//
// 它保证的不变量（也是 docs/live-union.md §3 第 6 条）：
//   min(live_union_max_rows(...) × per_row, n_expert) <= cap_min
// 只要 rows_want 满足该式，返回值就等于 rows_want；否则逐行往下退，最少退到 1。
// 因为 |U| <= min(rows × per_row, n_expert)，所以运行期 decide_admission 的
// `bank` 分支在窗口批次上不可达 —— 这就是「绝不崩」的来源。
inline int32_t live_union_max_rows(int32_t rows_want, int32_t per_row, int32_t n_expert, int32_t cap_min) {
    if (rows_want < 1) rows_want = 1;
    if (rows_want > kWindowMaxRowsGraphLimit) rows_want = kWindowMaxRowsGraphLimit;
    if (per_row    < 1) per_row    = 1;
    if (n_expert   < 1) n_expert   = 1;
    int32_t rows = rows_want;
    while (rows > 1 && (int32_t) std::min<int64_t>((int64_t) rows * per_row, n_expert) > cap_min) {
        --rows;
    }
    return rows;
}

} // namespace llama_moe
