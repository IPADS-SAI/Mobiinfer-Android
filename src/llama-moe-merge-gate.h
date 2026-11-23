#pragma once

// R4a：MoE 合并读的门控决策（纯函数，不依赖 slot 内部状态）。
//
// 改动前，同一个「这一步要不要合并」的判断散落在三个地方，而且只写在内联
// 布尔表达式里：
//
//   * demand：`install_packed_wave()` 的 `merge_active`（排序/切 run）和
//     `read_run_directv()` 内部的 `merge_reads_mode()==2 && read_mode()==0`；
//   * prefetch：`install_prefetch_batch()` 的 `mode==1 || io_budget || read_mode()!=0`；
//   * deferred demand：`process_deferred()` 自己的 `merge_mode==2 && n>1`。
//
// 任何一处静默失效时，日志只能看到「merge=2 / prefetch_mode=2」却没有任何
// merge-exec 计数，无法区分「开关没打开」「io_budget 禁用了」「read_mode 禁用」
// 「managed 混合」和「单专家」。这里把它们收成一个可单测的纯函数 + 稳定的
// reason code；调用点只读 `decide()` 的结果，不再各写一份布尔式。
//
// 语义与改动前的代码逐项对齐（不是理想语义）：
//
//   sort_demand      = reads==2 && !io_budget && n>1 && (!any_managed || all_managed)
//   merge_demand     = sort_demand && read_mode==0
//   project_prefetch = prefetch>=1 && n>1
//   merge_prefetch   = prefetch==2 && !io_budget && read_mode==0 && n>1
//
// 两个刻意保留的现状细节：
//
//   1. `sort_demand` **不**看 read_mode。改动前 `merge_active` 只看
//      merge_mode/io_budget/managed，read_mode!=0 时仍会排序并切 run，只是
//      `read_run_directv()` 随后拒绝合并、逐专家回退。为保持实际读顺序不变，
//      排序条件照旧；`merge_demand` 才是「真的发一次 preadv」的条件。
//   2. `project_prefetch` 带 `n>1`。spec 的公式写的是 prefetch>=1，但改动前
//      `install_prefetch_batch()` 在 n==1 时**先早退、不投影**；若单专家也投影，
//      `merge-prefetch-scan` 会凭空多出计数，旧日志就不再逐字节可比。162 组
//      真值表固定 n>1，因此与 spec 公式一致；n==1 的行为以代码为准（见
//      run 目录 notes.md 的差异记录）。
//
// reason 优先级固定（数值越小越优先，便于测试断言）：
//
//   env_off < io_budget < read_mode < mixed_managed < single_item < ok
//
// `env_off` 表示两条路径都没有处于 merge 模式（reads<2 且 prefetch<2）；
// 只有一条处于 merge 模式时，原因按该路径的实际禁用条件给出。
//
// 头文件不 include 项目内其它头文件，避免把后端依赖带进 host 单测。

#include <cstdint>

namespace llama_moe {
namespace merge_gate {

enum class reason : int {
    env_off      = 0,   // 两条路径都没有要求合并（0/1 都算；1 只统计/只投影）
    io_budget    = 1,   // io_budget>0：per-expert permit，合并会改变限流语义
    read_mode    = 2,   // read_mode!=0：dummy/skip 诊断读，不碰真实合并路径
    mixed_managed = 3,  // 同一 wave 里 managed 与 legacy 混合：request 无法逐个 finish
    single_item  = 4,   // n<=1：没有可合并的相邻专家
    ok           = 5,   // 门控允许
};

inline const char * reason_name(reason r) {
    switch (r) {
        case reason::env_off:       return "env_off";
        case reason::io_budget:     return "io_budget";
        case reason::read_mode:     return "read_mode";
        case reason::mixed_managed: return "mixed_managed";
        case reason::single_item:   return "single_item";
        case reason::ok:            return "ok";
    }
    return "unknown";
}

/// reason 的稳定优先级（越小越优先）。测试用它断言「两个禁用条件同时成立时
/// 原因不漂移」；也方便日志消费方排序。
inline int reason_priority(reason r) {
    return (int) r;
}

inline int clamp_mode(int v) {
    return v <= 0 ? 0 : (v >= 2 ? 2 : v);
}

struct inputs {
    int  merge_reads    = 0;   // LLAMA_MOE_MERGE_READS，已夹到 0..2
    int  merge_prefetch = 0;   // LLAMA_MOE_MERGE_PREFETCH，已夹到 0..2
    int  io_budget      = 0;   // >0 = LLAMA_MOE_EXPERT_IO_BUDGET 打开
    int  read_mode      = 0;   // read_mode()：0=正常；!=0 dummy/skip
    int  n              = 0;   // 本批专家数（同一 wave / 同一 prefetch batch）
    bool any_managed    = false;
    bool all_managed    = true;   // 空批时按 true（与调用方初值一致）
};

struct decision {
    bool sort_demand      = false;   // 可以按 expert_to_pos 排序并切 run
    bool merge_demand     = false;   // 真的允许一次 preadv（含 read_mode）
    bool project_prefetch = false;   // 计算「如果合并能省多少」
    bool merge_prefetch   = false;   // 真的允许预读批合并
    reason why            = reason::env_off;
};

inline decision decide(const inputs & in) {
    decision out;

    const int reads    = clamp_mode(in.merge_reads);
    const int prefetch = clamp_mode(in.merge_prefetch);
    const bool no_budget = in.io_budget <= 0;
    const bool normal_read = in.read_mode == 0;
    const bool multi = in.n > 1;
    const bool managed_compatible = !in.any_managed || in.all_managed;

    out.sort_demand      = reads == 2 && no_budget && multi && managed_compatible;
    out.merge_demand     = out.sort_demand && normal_read;
    // 单专家保持改动前的早退：不算投影，避免 merge-prefetch-scan 多计数。
    out.project_prefetch = prefetch >= 1 && multi;
    out.merge_prefetch   = prefetch == 2 && no_budget && normal_read && multi;

    // reason 只看「是否要求合并」，不看 n/managed 是否让本批真的能合并。
    // 这样 `merge=2` + `single_item` 与 `merge=0` + `env_off` 能在日志里区分。
    if (reads < 2 && prefetch < 2) {
        out.why = reason::env_off;
    } else if (!no_budget) {
        out.why = reason::io_budget;
    } else if (!normal_read) {
        out.why = reason::read_mode;
    } else if (reads == 2 && multi && in.any_managed && !in.all_managed) {
        out.why = reason::mixed_managed;
    } else if (!multi) {
        out.why = reason::single_item;
    } else {
        out.why = reason::ok;
    }
    return out;
}

} // namespace merge_gate
} // namespace llama_moe
