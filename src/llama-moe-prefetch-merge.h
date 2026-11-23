#pragma once

// R4a：预读读合并的**纯规划/执行**逻辑。
//
// 预读（lookahead）在 worker 里是「一个专家一个 task」。R4a 允许把同一层、
// 同一预测批次（同一 lookahead epoch）的 task 收集成一个 batch，先按
// `expert_to_pos` 排序，再把物理 position 连续的专家切成 run，一次 preadv
// 读完。这里只放不依赖 llama/ggml/后端、可在 host 上单测的部分：
//
//   * `parse_mode` / `env_mode`：`LLAMA_MOE_MERGE_PREFETCH=0|1|2` 的语义；
//   * `plan_batch`：空 batch / 单专家 / 连续 run / 间隔 run / run > max /
//     跨层拒绝 / 只统计模式；
//   * `execute_plan`：真正执行时「部分读失败必须重读整个 run」的保守回退。
//
// 关键不变量：
//   1. 规划只允许**同一层**的 experts（mixed_layers=true 时整批回退逐专家）；
//   2. mode=0 时 order / units 与输入逐项一致，调用方不会改变旧读顺序；
//   3. mode=1 只产出 projection，actual units 仍是逐专家（不改变实际读）；
//   4. run 长度上限由调用方给出（复用 `LLAMA_MOE_MERGE_MAX_EXPERTS`）；
//   5. 任何 run 的合并读失败（false 或异常）都必须把该 run 的**每个**专家
//      重新逐专家读一遍 —— 只重读第一个会让其余 managed request 永远停在
//      loading（R3 踩过的坑）。
//
// 头文件不 include 其它项目头文件，避免把后端告警带进 host 单测。

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace llama_moe {
namespace prefetch_merge {

enum class mode : int {
    off     = 0,   // 旧路径：逐专家读，顺序也不变
    project = 1,   // 只统计「如果合并能省多少」，不改变实际读
    merge   = 2,   // 排序 + 连续段一次 preadv
};

inline int parse_mode(const char * value) {
    const int raw = value ? std::atoi(value) : 0;
    return raw < 0 ? 0 : (raw > 2 ? 2 : raw);
}

inline int env_mode() {
    return parse_mode(std::getenv("LLAMA_MOE_MERGE_PREFETCH"));
}

struct item {
    int32_t layer  = -1;
    int32_t expert = -1;
    int32_t slot   = -1;
    int32_t pos    = -1;   // expert_to_pos[expert]；-1 = 无法映射（不参与合并）
};

struct unit {
    int  first = 0;        // 在 plan::order 里的起点
    int  count = 0;
    bool merged = false;   // true = 调用方应尝试一次 preadv
    bool slots_consecutive = false;

    unit() = default;
    unit(int f, int c, bool m, bool s) : first(f), count(c), merged(m), slots_consecutive(s) {}
};

struct stats {
    int runs          = 0; // 按 position 数出来的连续段数
    int syscalls      = 0; // 按 max_experts 切段后的 preadv 次数投影
    int experts       = 0;
    int max_run       = 0;
    int zerocopy_runs = 0; // slot 也连号的段数（投影/实际口径见调用方）
    int scatter_runs  = 0;
};

struct plan {
    bool mixed_layers = false;
    bool fallback     = false;   // 整批不可合并（跨层/调用方禁用）
    std::vector<int>  order;     // 执行顺序：元素是输入下标
    std::vector<unit> units;
    stats projection;
};

namespace detail {

inline bool same_layer(const std::vector<item> & items) {
    if (items.empty()) return true;
    const int32_t layer = items.front().layer;
    if (layer < 0) return false;
    for (const auto & it : items) {
        if (it.layer != layer) return false;
    }
    return true;
}

inline bool slots_consecutive(const std::vector<item> & items, const std::vector<int> & order,
                              int first, int count) {
    if (count <= 1) return true;
    if (items[(size_t) order[(size_t) first]].slot < 0) return false;
    for (int j = 1; j < count; ++j) {
        const int32_t prev = items[(size_t) order[(size_t) (first + j - 1)]].slot;
        const int32_t cur  = items[(size_t) order[(size_t) (first + j)]].slot;
        if (prev < 0 || cur != prev + 1) return false;
    }
    return true;
}

// 按 position 升序；相同 position 保持输入顺序（stable），这样合成 buffer
// 对比时落点可复现。
inline void sort_by_pos(const std::vector<item> & items, std::vector<int> & order) {
    order.resize(items.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const int32_t pa = items[(size_t) a].pos;
        const int32_t pb = items[(size_t) b].pos;
        if (pa != pb) return pa < pb;
        return a < b;
    });
}

inline void project(const std::vector<item> & items, int max_experts, stats & out) {
    out = stats{};
    if (items.empty()) return;
    std::vector<int> order;
    sort_by_pos(items, order);
    const int chunk = std::max(1, max_experts);
    const int n = (int) order.size();
    int i = 0;
    while (i < n) {
        int j = i + 1;
        const int32_t p0 = items[(size_t) order[(size_t) i]].pos;
        while (j < n && p0 >= 0 &&
               items[(size_t) order[(size_t) j]].pos == p0 + (j - i)) {
            ++j;
        }
        const int len = j - i;
        out.runs++;
        out.experts += len;
        out.max_run = std::max(out.max_run, len);
        out.syscalls += (len + chunk - 1) / chunk;
        if (slots_consecutive(items, order, i, len)) {
            out.zerocopy_runs++;
        } else {
            out.scatter_runs++;
        }
        i = j;
    }
}

} // namespace detail

/// 规划一个 batch。`allow_merge` 由调用方根据运行期条件给出（io_budget 开、
/// dummy/skip 读、非 2 模式等都必须传 false）。mode=project 时无论
/// allow_merge 如何都只产生逐专家 units，但 projection 会填好。
inline plan plan_batch(const std::vector<item> & items, int mode_value, int max_experts,
                       bool allow_merge) {
    const int mode = mode_value < 0 ? 0 : (mode_value > 2 ? 2 : mode_value);
    plan out;
    const int n = (int) items.size();
    out.order.resize((size_t) n);
    std::iota(out.order.begin(), out.order.end(), 0);
    if (n == 0) return out;

    const int chunk = std::max(1, max_experts);
    out.mixed_layers = !detail::same_layer(items);
    if (out.mixed_layers) {
        // 跨层请求绝不合并；整批保持输入顺序逐专家执行。
        out.fallback = true;
        out.units.resize((size_t) n);
        for (int i = 0; i < n; ++i) {
            out.units[(size_t) i] = unit{i, 1, false, true};
        }
        return out;
    }

    // 投影与开关无关地算（mode>=1），供 prefetch_runs/syscalls 计数。
    if (mode >= 1) {
        detail::project(items, chunk, out.projection);
    }

    if (mode == 0) {
        out.units.resize((size_t) n);
        for (int i = 0; i < n; ++i) out.units[(size_t) i] = unit{i, 1, false, true};
        return out;
    }

    if (mode == 1 || !allow_merge) {
        // 只统计 / 运行期条件不允许：actual 仍是逐专家，顺序保持输入顺序
        //（mode=1 的承诺是「不改变实际读」，包括读顺序）。
        out.fallback = true;
        out.units.resize((size_t) n);
        for (int i = 0; i < n; ++i) out.units[(size_t) i] = unit{i, 1, false, true};
        return out;
    }

    // mode=2：按 position 排序后切 run；每个 run 再按 max_experts 切块。
    detail::sort_by_pos(items, out.order);
    int i = 0;
    while (i < n) {
        int j = i + 1;
        const int32_t p0 = items[(size_t) out.order[(size_t) i]].pos;
        while (j < n && p0 >= 0 &&
               items[(size_t) out.order[(size_t) j]].pos == p0 + (j - i)) {
            ++j;
        }
        for (int a = i; a < j; a += chunk) {
            const int count = std::min(chunk, j - a);
            unit u;
            u.first  = a;
            u.count  = count;
            u.merged = count > 1;
            u.slots_consecutive = detail::slots_consecutive(items, out.order, a, count);
            out.units.push_back(u);
        }
        i = j;
    }
    return out;
}

struct exec_stats {
    int merged_runs   = 0;
    int fallback_runs = 0;
    int experts       = 0;
    int syscalls      = 0;
    int zerocopy_runs = 0;
    int scatter_runs  = 0;
    int syscalls_saved() const { return experts > syscalls ? experts - syscalls : 0; }
};

/// 执行 plan。
///   read_run(order, unit) -> bool  : 一次 preadv；false/异常都表示本 run 未成功
///   read_one(index)                : 逐专家读（index 是输入下标）
///   on_merged(order, unit)         : 合并读成功后的发布/记账（逐专家）
///
/// 回退语义是这里的核心：read_run 失败时**整个 unit 的每个专家都调用
/// read_one**，不会只读第一个。
template <class ReadRun, class ReadOne, class OnMerged>
exec_stats execute_plan(const plan & p, ReadRun read_run, ReadOne read_one, OnMerged on_merged) {
    exec_stats out;
    for (const auto & u : p.units) {
        const bool want_merge = u.merged && u.count > 1;
        bool ok = false;
        if (want_merge) {
            try {
                ok = read_run(p.order, u) ? true : false;
            } catch (...) {
                ok = false;
            }
        }
        if (want_merge && ok) {
            out.merged_runs++;
            out.experts += u.count;
            out.syscalls += 1;
            if (u.slots_consecutive) ++out.zerocopy_runs;
            else                     ++out.scatter_runs;
            on_merged(p.order, u);
            continue;
        }
        if (want_merge) {
            // 部分读 / EINVAL / EFAULT / IOV_MAX 等：整段回退，不能只读第一个。
            out.fallback_runs++;
        }
        for (int j = 0; j < u.count; ++j) {
            read_one(p.order[(size_t) (u.first + j)]);
            out.experts++;
            out.syscalls++;
        }
    }
    return out;
}

} // namespace prefetch_merge
} // namespace llama_moe
