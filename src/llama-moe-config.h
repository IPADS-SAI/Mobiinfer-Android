#pragma once

// MoE 运行时配置 env 的**纯解析函数**。
//
// 这些开关（`LLAMA_MOE_LAYER_WINDOWS` / `LLAMA_MOE_EVICT[_LAYERS]` /
// `LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH` / `LLAMA_MOE_PREFETCH_REFRESH`）过去把
// 非法输入静默忽略：一次配置笔误不会报错，只会让上机 A/B 白跑。把解析抽到这里
// 是为了能在 host 上直接单测（`tests/test-moe-config.cpp`，不需要模型/后端/HTP）。
//
// **约束：解析语义与运行期默认行为逐位不变。** 非法项仍按原行为处理（跳过 / 截断 /
// 返回 0 / 抛错），只是额外返回 `ignored` / `warnings`，让调用方能把「你写的配置
// 没生效」打进日志。默认配置（env 未设或为空）不产生任何诊断。
//
// 头文件不依赖 llama / ggml / 后端，也不 include 其它项目头文件（避免把
// `llama-moe-slot-manager.h` 的既有告警带进新的编译单元、污染告警基线）。

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace llama_moe {
namespace config {

// ── LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH ────────────────────────────────────
//
// 预取深度既要装得进 slot bank，又要有上限（再深 gate 预测精度会崩）。
// 语义（`llama-moe-slot.h` 旧实现，保持不变）：
//   负数            -> 0（关闭）
//   > max_depth(16) -> 16
//   bank < 2*depth+8 -> 0（bank 太小，预取静默关闭）
inline constexpr int max_prefetch_depth = 16;

inline int normalize_prefetch_depth(int bank, int requested) {
    const int depth = requested < 0 ? 0 : (requested > max_prefetch_depth ? max_prefetch_depth : requested);
    return bank >= 2 * depth + 8 ? depth : 0;
}

// ── LLAMA_MOE_LAYER_WINDOWS ───────────────────────────────────────────────
//
// 格式 `层:宽度[,...]`，未列出的层用 base（= 全局 decode window）。
// 行为与旧实现一致：
//   * 空串 / nullptr : 全部用 base；
//   * 宽度夹到 [lo, hi]（lo = max(1, n_expert_used)，hi = 1024）；
//   * 负层号 / 越界层号 / 宽度 <= 0 : **跳过该项**，继续解析后面的项；
//   * 缺少 `:` 或宽度不是数字 : **停止解析**（已解析的项仍生效）；
//   * 同层重复出现 : 后者覆盖前者。
// 被跳过/截断的项写进 `ignored`，调用方负责打日志（旧实现完全静默）。
struct layer_windows {
    std::vector<int32_t>    widths;    // 大小 = max(n_layers, 0)，初值 base
    std::vector<std::string> ignored;  // 每个未生效的项一条说明
};

namespace detail {

inline std::string window_item_text(const char * p) {
    const char * end = p;
    while (*end && *end != ',') ++end;
    return std::string(p, (size_t) (end - p));
}

inline std::string window_layer_range(int32_t n_layers) {
    return "[0," + std::to_string((long long) n_layers) + ")";
}

} // namespace detail

inline layer_windows parse_layer_windows(const char * spec, int32_t n_layers, int32_t base,
                                         int32_t lo, int32_t hi) {
    layer_windows out;
    out.widths.assign(n_layers > 0 ? (size_t) n_layers : 0, base);
    if (base <= 0 || !spec || !*spec || n_layers <= 0) {
        return out;                     // 未启用窗口 / 没有按层覆盖
    }

    const char * p = spec;
    while (*p) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;

        const char * item = p;
        char * end = nullptr;
        const long il = std::strtol(p, &end, 10);
        if (end == p || *end != ':') {
            out.ignored.push_back("stopped at '" + detail::window_item_text(item) +
                                  "': expected layer:width");
            break;
        }
        p = end + 1;
        const long w = std::strtol(p, &end, 10);
        if (end == p) {
            out.ignored.push_back("stopped at '" + detail::window_item_text(item) +
                                  "': expected width after ':'");
            break;
        }
        p = end;

        if (il >= 0 && il < n_layers && w > 0) {
            out.widths[(size_t) il] = (int32_t) std::min<long>(std::max<long>(w, lo), hi);
        } else if (il < 0 || il >= n_layers) {
            out.ignored.push_back("ignored '" + detail::window_item_text(item) +
                                  "': layer out of range " + detail::window_layer_range(n_layers));
        } else {
            out.ignored.push_back("ignored '" + detail::window_item_text(item) +
                                  "': width must be > 0");
        }
    }
    return out;
}

// ── LLAMA_MOE_EVICT / LLAMA_MOE_EVICT_LAYERS ──────────────────────────────
//
// `LLAMA_MOE_EVICT = lru（默认）| lfu | lfu<N>`；`<N>` 是计数减半的周期，
// 0 / 缺省 = 不衰减。
enum class evict_policy { lru, lfu };

struct evict_spec {
    evict_policy policy;
    uint32_t decay;        // 仅 lfu 有意义
    std::string error;     // 非空 = 调用方必须抛错（旧行为）
    std::string warning;   // 非空 = 值按旧行为解释，但配置可能写错了

    evict_spec() : policy(evict_policy::lru), decay(0) {}
};

inline evict_spec parse_evict_spec(const char * value, const char * what) {
    evict_spec out;
    if (!value || !*value || std::strcmp(value, "lru") == 0) {
        return out;                     // 默认 lru
    }
    if (std::strncmp(value, "lfu", 3) != 0) {
        out.error = std::string(what) + " must be lru | lfu | lfu<N>";
        return out;
    }

    out.policy = evict_policy::lfu;
    const char * suffix = value + 3;
    if (*suffix) {
        char * end = nullptr;
        // 与旧实现一致：strtoul 的成功/失败都不报错；解析不出来就是 0。
        out.decay = (uint32_t) std::strtoul(suffix, &end, 10);
        if (!(*suffix >= '0' && *suffix <= '9') || *end) {
            out.warning = std::string("decay suffix '") + suffix +
                          "' is not a plain integer; using " + std::to_string((unsigned long) out.decay);
        }
    }
    return out;
}

struct evict_layers {
    std::map<int32_t, evict_spec> layers;  // 层号 -> 覆盖；同层后者胜
    std::string error;                     // 非空 = 调用方必须抛错
    std::vector<std::string> warnings;     // 非致命异常（旧实现静默接受）
};

inline evict_layers parse_evict_layers(const char * value) {
    evict_layers out;
    if (!value || !*value) {
        return out;
    }

    const std::string spec(value);
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? spec.size() + 1 : comma + 1;
        if (item.empty()) continue;

        const size_t colon = item.find(':');
        if (colon == std::string::npos) {
            out.error = "LLAMA_MOE_EVICT_LAYERS expects layer:policy[,...]";
            return out;
        }

        const std::string layer_text = item.substr(0, colon);
        char * end = nullptr;
        const long parsed = std::strtol(layer_text.c_str(), &end, 10);
        const int32_t layer = (int32_t) parsed;
        if (end == layer_text.c_str() || *end) {
            // 旧实现不检查 end：'abc:lru' 会被当成 layer 0。保持数值不变，只记录告警。
            out.warnings.push_back("layer '" + layer_text + "' is not a plain integer; using " +
                                   std::to_string((long long) layer));
        }
        if (layer < 0) {
            out.error = "LLAMA_MOE_EVICT_LAYERS: layer must be >= 0";
            return out;
        }

        const evict_spec parsed_spec = parse_evict_spec(item.substr(colon + 1).c_str(), "LLAMA_MOE_EVICT_LAYERS");
        if (!parsed_spec.error.empty()) {
            out.error = parsed_spec.error;
            return out;
        }
        if (!parsed_spec.warning.empty()) {
            out.warnings.push_back("layer " + std::to_string((long long) layer) + ": " + parsed_spec.warning);
        }
        out.layers[layer] = parsed_spec;   // 重复层号：后者覆盖（旧行为）
    }
    return out;
}

// ── LLAMA_MOE_PREFETCH_REFRESH 前置条件 ───────────────────────────────────
//
// REFRESH 依赖 Ready window + 打开的预取；不满足时旧实现直接抛
// `LLAMA_MOE_PREFETCH_REFRESH requires ...`。这里把判据拆成标志位，
// 让单测能覆盖每个前置条件，也让运行期错误信息能指出**到底缺哪一个**。
struct refresh_state {
    bool managers_active;   // Ready window 的 slot manager 已建立
    bool window_mode;       // 预取走 decode 窗口
    bool matmul_split;      // matmul_split > 0
    bool prefetch_enabled;  // depth 归一化后 > 0
    int  read_n;            // 每层预读多少个专家（<=0 = 关）
    int  horizon;           // 跨层流水 horizon（<=1 = 只预测下一层）
    bool io_budget;         // LLAMA_MOE_EXPERT_IO_BUDGET != 0
    bool h1_front;          // LLAMA_MOE_PREFETCH_H1_FRONT != 0
};

enum refresh_requirement : unsigned {
    refresh_ready_window   = 1u << 0,
    refresh_split_zero     = 1u << 1,
    refresh_horizon_gt_one = 1u << 2,
    refresh_prefetch_read  = 1u << 3,
    refresh_io_budget_zero = 1u << 4,
    refresh_h1_front_zero  = 1u << 5,
};

inline unsigned prefetch_refresh_missing(const refresh_state & state) {
    unsigned missing = 0;
    if (!state.managers_active || !state.window_mode) missing |= refresh_ready_window;
    if (state.matmul_split)                        missing |= refresh_split_zero;
    if (state.horizon <= 1)                        missing |= refresh_horizon_gt_one;
    if (!state.prefetch_enabled || state.read_n <= 0) missing |= refresh_prefetch_read;
    if (state.io_budget)                           missing |= refresh_io_budget_zero;
    if (state.h1_front)                            missing |= refresh_h1_front_zero;
    return missing;
}

inline std::string refresh_missing_text(unsigned missing) {
    struct entry { unsigned bit; const char * text; };
    static const entry entries[] = {
        { refresh_ready_window,   "ready-window" },
        { refresh_split_zero,     "matmul_split=0" },
        { refresh_horizon_gt_one, "horizon>1" },
        { refresh_prefetch_read,  "prefetch/read>0" },
        { refresh_io_budget_zero, "io-budget=0" },
        { refresh_h1_front_zero,  "h1-front=0" },
    };
    std::string out;
    for (const entry & e : entries) {
        if (!(missing & e.bit)) continue;
        if (!out.empty()) out += ", ";
        out += e.text;
    }
    return out;
}

} // namespace config
} // namespace llama_moe
