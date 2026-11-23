// `LLAMA_MOE_LAYER_WINDOWS` / `LLAMA_MOE_EVICT[_LAYERS]` /
// `LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH` / `LLAMA_MOE_PREFETCH_REFRESH`
// 配置解析的纯函数单测。
//
// 背景：这些开关过去对非法输入静默处理（跳过、截断、归一化成 0、或静默接受
// 非数字层号），一次配置笔误不会报错，只会白跑一次上机 A/B。这里把解析抽到
// src/llama-moe-config.h 后逐条 pin 住边界行为，保证「修可观测性」不改变语义。
//
// 编译 + 运行（host，任意 g++ >= C++17）：
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test-moe-config tests/test-moe-config.cpp
//   /tmp/test-moe-config
//
// 覆盖：空串 / nullptr / 非法项 / 停止解析 / 越界与负层号 / 宽度夹紧 / 重复项 /
// 负 depth / 超过上限的 depth / BANK 太小 / REFRESH 的每个前置条件。

#include "../src/llama-moe-config.h"
#include "../src/llama-moe-prefetch-merge.h"
#include "../ggml/src/ggml-hexagon/htp/htp-ops.h"

#include <cstdint>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <vector>

namespace config = llama_moe::config;

static int failures = 0;

static void expect_int(const char * text, long got, long want, int line) {
    if (got != want) {
        ++failures;
        std::printf("FAIL %s:%d: %s = %ld, want %ld\n", __FILE__, line, text, got, want);
    }
}

static void expect_true(const char * text, bool ok, int line) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s:%d: %s\n", __FILE__, line, text);
    }
}

#define EXPECT_INT(expr, want)  expect_int(#expr, (long) (expr), (long) (want), __LINE__)
#define EXPECT_TRUE(expr)       expect_true(#expr, (bool) (expr), __LINE__)

static void expect_u64(const char * text, uint64_t got, uint64_t want, int line) {
    if (got != want) {
        ++failures;
        std::printf("FAIL %s:%d: %s = %llu, want %llu\n", __FILE__, line, text,
                    (unsigned long long) got, (unsigned long long) want);
    }
}

#define EXPECT_U64(expr, want) expect_u64(#expr, (uint64_t) (expr), (uint64_t) (want), __LINE__)

static bool all_equal(const std::vector<int32_t> & values, int32_t want) {
    for (int32_t value : values) {
        if (value != want) return false;
    }
    return true;
}

static config::layer_windows parse_windows(const char * spec, int32_t n_layers = 40,
                                           int32_t base = 80, int32_t lo = 8, int32_t hi = 1024) {
    return config::parse_layer_windows(spec, n_layers, base, lo, hi);
}

static config::refresh_state refresh_ok() {
    config::refresh_state state{};
    state.managers_active  = true;
    state.window_mode      = true;
    state.matmul_split     = false;
    state.prefetch_enabled = true;
    state.read_n           = 8;
    state.horizon          = 2;
    state.io_budget        = false;
    state.h1_front         = false;
    return state;
}


// gate.sh 的 host ctest 正则只点名 test-moe-{ready,io-admission,live-union,config}；
// 把 R4a 的关键边界也钉在 config 测试里，保证 build gate 一定会跑到（独立的
// test-moe-prefetch-merge 仍保留更细的用例）。
static void test_prefetch_merge_contract() {
    namespace pm = llama_moe::prefetch_merge;
    auto items = [](std::vector<int> pos) {
        std::vector<pm::item> out;
        for (size_t i = 0; i < pos.size(); ++i) {
            pm::item it; it.layer = 2; it.expert = 50 + (int) i; it.slot = (int) i; it.pos = pos[i];
            out.push_back(it);
        }
        return out;
    };

    EXPECT_INT(pm::parse_mode(nullptr), 0);
    EXPECT_INT(pm::parse_mode("2"), 2);
    EXPECT_INT(pm::parse_mode("9"), 2);

    const auto empty = pm::plan_batch({}, 2, 4, true);
    EXPECT_TRUE(empty.units.empty());

    const auto single = pm::plan_batch(items({7}), 2, 4, true);
    EXPECT_INT(single.units.size(), 1);
    EXPECT_TRUE(!single.units[0].merged);

    const auto consec = pm::plan_batch(items({10, 11, 12, 13}), 2, 4, true);
    EXPECT_INT(consec.units.size(), 1);
    EXPECT_TRUE(consec.units[0].merged && consec.units[0].count == 4);
    EXPECT_INT(consec.projection.syscalls, 1);

    const auto gapped = pm::plan_batch(items({10, 11, 20, 21}), 2, 4, true);
    EXPECT_INT(gapped.units.size(), 2);
    EXPECT_INT(gapped.projection.runs, 2);

    const auto over = pm::plan_batch(items({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), 2, 4, true);
    EXPECT_INT(over.units.size(), 3);              // 4 + 4 + 2
    EXPECT_INT(over.projection.syscalls, 3);

    auto cross = items({0, 1, 2, 3});
    cross[2].layer = 3;
    const auto mixed = pm::plan_batch(cross, 2, 4, true);
    EXPECT_TRUE(mixed.mixed_layers);
    for (int i = 0; i < 4; ++i) EXPECT_TRUE(!mixed.units[(size_t) i].merged);

    // mode=1 只统计，实际 units 仍逐专家且保持输入顺序。
    const auto projection_only = pm::plan_batch(items({9, 3, 8, 2}), 1, 4, false);
    EXPECT_INT(projection_only.units.size(), 4);
    for (int i = 0; i < 4; ++i) EXPECT_TRUE(!projection_only.units[(size_t) i].merged);
    EXPECT_INT(projection_only.projection.runs, 2);

    // 部分读/异常 -> 整个 run 的每个专家都回退重读，不能只读第一个。
    const auto plan = pm::plan_batch(items({0, 1, 2, 3}), 2, 4, true);
    int one = 0, merged = 0;
    const auto stats = pm::execute_plan(plan,
        [](const std::vector<int> &, const pm::unit &) -> bool {
            throw std::runtime_error("short preadv");
        },
        [&](int) { ++one; },
        [&](const std::vector<int> &, const pm::unit &) { ++merged; });
    EXPECT_INT(one, 4);
    EXPECT_INT(merged, 0);
    EXPECT_INT(stats.fallback_runs, 1);
    EXPECT_INT(stats.syscalls_saved(), 0);
    std::printf("ok   prefetch-merge contract (gate-covered)\n");
}

// ---------------------------------------------------------------------------
// T-0304: htp_opbuf_stats accounting contract
//
// The production loops live in ggml/src/ggml-hexagon/htp/main.c, which is
// DSP-only.  This harness mirrors prep_op_bufs() pass 0 / drop / pass 1 on a
// synthetic 64-entry mmap table and uses the exact same inline accounting
// helpers from htp-opbuf-stats.h, so scan_cmp/drop_n/map_n can be checked
// against a hand calculation without a device.  Keep the mirror in sync if
// prep_op_bufs() changes.
//
// Counted semantics (see the header): scan_cmp is the number of mmap-table
// entries examined by reuse_buf()/mmap_buf(); drop_n/map_n count actual
// htp_munmap()/htp_mmap() calls; a defrag pass remaps non-extended buffers
// that pass 1 already mapped, so map_n can exceed n_bufs - b_reuse there.
// ---------------------------------------------------------------------------
static constexpr uint32_t kMaxMmaps   = 64; // HTP_MAX_MMAPS
static constexpr uint32_t kBufExtended = 1u; // HTP_BUF_EXTENDED

struct sim_mmap {
    uint64_t size;
    uint32_t fd;
    uint32_t flags;
};

struct sim_buf {
    uint64_t size;
    uint32_t fd;
    uint32_t flags;
    bool     base;
};

static void sim_reset(sim_mmap (&tab)[kMaxMmaps]) {
    for (auto & m : tab) {
        m = {0, 0xffffffffu, 0};
    }
}

static bool sim_reuse(sim_mmap (&tab)[kMaxMmaps], uint32_t & mask, sim_buf & b, htp_opbuf_stats & st) {
    b.base = false;
    for (uint32_t i = 0; i < kMaxMmaps; i++) {
        htp_opbuf_stats_scan(&st, 1);
        const sim_mmap & m = tab[i];
        if (m.size && m.fd == b.fd && m.flags == b.flags) {
            b.base = true;
            // The production mask is uint32_t; hits at index >= 32 are outside
            // this harness (and outside the production mask's range).
            mask |= (1u << i);
            htp_opbuf_stats_reuse(&st);
            return true;
        }
    }
    return false;
}

static bool sim_map(sim_mmap (&tab)[kMaxMmaps], sim_buf & b, htp_opbuf_stats & st) {
    if (b.base) {
        return true;
    }
    for (uint32_t i = 0; i < kMaxMmaps; i++) {
        htp_opbuf_stats_scan(&st, 1);
        if (!tab[i].size) {
            tab[i] = {b.size, b.fd, b.flags};
            b.base = true;
            htp_opbuf_stats_map(&st, b.size);
            return true;
        }
    }
    return false;
}

static void sim_prep(sim_mmap (&tab)[kMaxMmaps], sim_buf * bufs, uint32_t n, uint64_t max_vmem,
                     htp_opbuf_stats & st) {
    uint32_t mask    = 0;
    uint32_t b_reuse = 0;
    uint64_t e_vmem  = 0;

    htp_opbuf_stats_init(&st, n, max_vmem);

    for (uint32_t i = 0; i < n; i++) {
        sim_buf & b = bufs[i];
        if (sim_reuse(tab, mask, b, st)) {
            b_reuse++;
        } else if (!(b.flags & kBufExtended)) {
            e_vmem += b.size;
        }
    }
    st.e_vmem = e_vmem;

    if (b_reuse == n) {
        return; // all bufs reuse existing mappings
    }

    uint64_t m_vmem = 0;
    for (auto & m : tab) {
        if (!(m.flags & kBufExtended)) {
            m_vmem += m.size;
        }
    }
    st.m_vmem = m_vmem;

    if (m_vmem + e_vmem > max_vmem) {
        for (uint32_t i = 0; i < kMaxMmaps; i++) {
            const bool used = i < 32 && (mask & (1u << i)) != 0;
            if (!used && !(tab[i].flags & kBufExtended) && tab[i].size) {
                htp_opbuf_stats_drop(&st, tab[i].size);
                tab[i] = {0, 0xffffffffu, 0};
            }
        }
    }

    bool mmap_ok = true;
    for (uint32_t i = 0; i < n; i++) {
        if (!sim_map(tab, bufs[i], st)) {
            mmap_ok = false;
            break;
        }
    }

    if (!mmap_ok) {
        // Defrag pass: drop every non-extended mapping, then remap.
        htp_opbuf_stats_defrag(&st);
        for (uint32_t i = 0; i < kMaxMmaps; i++) {
            if (!(tab[i].flags & kBufExtended) && tab[i].size) {
                htp_opbuf_stats_drop(&st, tab[i].size);
                tab[i] = {0, 0xffffffffu, 0};
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            if (!(bufs[i].flags & kBufExtended)) {
                bufs[i].base = false;
            }
            // The production path aborts if this fails; the harness only
            // needs the successful accounting here.
            sim_map(tab, bufs[i], st);
        }
    }
}

static void test_opbuf_stats_contract() {
    // 0) ABI: sizes are pinned and a legacy/short response is rejected.  The
    //    pre-T-0304 htp_opbatch_rsp was 88 bytes (no magic, no counters).
    EXPECT_U64(sizeof(htp_opbuf_stats), 72);
    EXPECT_U64(sizeof(htp_opbatch_rsp), 168);
    EXPECT_TRUE(htp_opbatch_rsp_size_ok(sizeof(htp_opbatch_rsp), 1));
    EXPECT_TRUE(!htp_opbatch_rsp_size_ok(88, 1));
    EXPECT_TRUE(!htp_opbatch_rsp_size_ok(sizeof(htp_opbatch_rsp), 2));
    EXPECT_TRUE(htp_opbatch_rsp_magic_ok(HTP_OPBATCH_ABI_MAGIC));
    EXPECT_TRUE(!htp_opbatch_rsp_magic_ok(0));
    EXPECT_TRUE(!htp_opbatch_rsp_magic_ok(HTP_OPBATCH_ABI_MAGIC ^ 1u));

    // 1) Zero/default record: all fields zero, conservation holds.
    {
        htp_opbuf_stats st;
        htp_opbuf_stats_init(&st, 0, 0);
        EXPECT_U64(st.n_bufs, 0);
        EXPECT_U64(st.b_reuse, 0);
        EXPECT_U64(st.drop_n, 0);
        EXPECT_U64(st.map_n, 0);
        EXPECT_U64(st.defrag_n, 0);
        EXPECT_U64(st.drop_bytes, 0);
        EXPECT_U64(st.map_bytes, 0);
        EXPECT_U64(st.m_vmem, 0);
        EXPECT_U64(st.e_vmem, 0);
        EXPECT_U64(st.max_vmem, 0);
        EXPECT_U64(st.scan_cmp, 0);
        EXPECT_TRUE(htp_opbuf_stats_conserved(&st));
    }

    // 2) n_bufs = 7, empty table: pass0 = 7*64 = 448 scans, pass1 = 1+..+7 = 28,
    //    map 7 buffers / 2800 B, no drops, e_vmem = 2800.
    {
        sim_mmap tab[kMaxMmaps];
        sim_reset(tab);
        sim_buf bufs[7];
        for (uint32_t i = 0; i < 7; i++) {
            bufs[i] = {100ull * (i + 1), 10 + i, 0, false};
        }
        htp_opbuf_stats st;
        sim_prep(tab, bufs, 7, 1000000, st);
        EXPECT_U64(st.n_bufs, 7);
        EXPECT_U64(st.b_reuse, 0);
        EXPECT_U64(st.scan_cmp, 448 + 28);
        EXPECT_U64(st.map_n, 7);
        EXPECT_U64(st.map_bytes, 2800);
        EXPECT_U64(st.drop_n, 0);
        EXPECT_U64(st.drop_bytes, 0);
        EXPECT_U64(st.e_vmem, 2800);
        EXPECT_U64(st.m_vmem, 0);
        EXPECT_U64(st.max_vmem, 1000000);
        EXPECT_TRUE(htp_opbuf_stats_conserved(&st));
    }

    // 3) n_bufs = 7, all reuse (early return): scan = 1+..+7 = 28, no map/drop.
    {
        sim_mmap tab[kMaxMmaps];
        sim_reset(tab);
        sim_buf bufs[7];
        for (uint32_t i = 0; i < 7; i++) {
            tab[i] = {100ull * (i + 1), 10 + i, 0};
            bufs[i] = {100ull * (i + 1), 10 + i, 0, false};
        }
        htp_opbuf_stats st;
        sim_prep(tab, bufs, 7, 1000000, st);
        EXPECT_U64(st.n_bufs, 7);
        EXPECT_U64(st.b_reuse, 7);
        EXPECT_U64(st.scan_cmp, 28);
        EXPECT_U64(st.map_n, 0);
        EXPECT_U64(st.map_bytes, 0);
        EXPECT_U64(st.drop_n, 0);
        EXPECT_U64(st.e_vmem, 0);
        EXPECT_U64(st.m_vmem, 0); // early return: pass 1 not run
        EXPECT_TRUE(htp_opbuf_stats_conserved(&st));
    }

    // 4) n_bufs = 1, empty table: 64 + 1 scans, one mapping.
    {
        sim_mmap tab[kMaxMmaps];
        sim_reset(tab);
        sim_buf buf = {4096, 7, 0, false};
        htp_opbuf_stats st;
        sim_prep(tab, &buf, 1, 1000000, st);
        EXPECT_U64(st.scan_cmp, 65);
        EXPECT_U64(st.map_n, 1);
        EXPECT_U64(st.map_bytes, 4096);
        EXPECT_TRUE(htp_opbuf_stats_conserved(&st));
    }

    // 5) n_bufs = 64, full table, max_vmem = 0: drop all 64 (640 B), then
    //    pass1 scans 1+..+64 = 2080; pass0 = 4096; total 6176; map 64 / 66016 B.
    {
        sim_mmap tab[kMaxMmaps];
        sim_reset(tab);
        for (uint32_t i = 0; i < kMaxMmaps; i++) {
            tab[i] = {10, 1000 + i, 0};
        }
        sim_buf bufs[64];
        uint64_t total = 0;
        for (uint32_t i = 0; i < 64; i++) {
            bufs[i] = {1000ull + i, i, 0, false};
            total += bufs[i].size;
        }
        htp_opbuf_stats st;
        sim_prep(tab, bufs, 64, 0, st);
        EXPECT_U64(st.n_bufs, 64);
        EXPECT_U64(st.b_reuse, 0);
        EXPECT_U64(st.scan_cmp, 4096 + 2080);
        EXPECT_U64(st.drop_n, 64);
        EXPECT_U64(st.drop_bytes, 640);
        EXPECT_U64(st.map_n, 64);
        EXPECT_U64(st.map_bytes, total);
        EXPECT_U64(st.e_vmem, total);
        EXPECT_U64(st.m_vmem, 640);
        EXPECT_TRUE(htp_opbuf_stats_conserved(&st));
    }

    // 6) Forced pass-1 failure (full table, max_vmem large): one defrag pass,
    //    drop 64 (640 B), then map the single buffer.  scan = 64 + 64 + 1.
    {
        sim_mmap tab[kMaxMmaps];
        sim_reset(tab);
        for (uint32_t i = 0; i < kMaxMmaps; i++) {
            tab[i] = {10, 1000 + i, 0};
        }
        sim_buf buf = {4096, 7, 0, false};
        htp_opbuf_stats st;
        sim_prep(tab, &buf, 1, 1000000000ull, st);
        EXPECT_U64(st.defrag_n, 1);
        EXPECT_U64(st.scan_cmp, 64 + 64 + 1);
        EXPECT_U64(st.drop_n, 64);
        EXPECT_U64(st.drop_bytes, 640);
        EXPECT_U64(st.map_n, 1);
        EXPECT_U64(st.map_bytes, 4096);
        EXPECT_TRUE(htp_opbuf_stats_conserved(&st));
    }

    // 7) Conservation law: defrag is the only case where map_n may exceed
    //    n_bufs - b_reuse.
    {
        htp_opbuf_stats bad;
        htp_opbuf_stats_init(&bad, 7, 0);
        bad.b_reuse = 3;
        bad.map_n   = 3;
        EXPECT_TRUE(!htp_opbuf_stats_conserved(&bad));

        bad.map_n   = 4;
        bad.defrag_n = 1;
        EXPECT_TRUE(htp_opbuf_stats_conserved(&bad));

        htp_opbuf_stats over;
        htp_opbuf_stats_init(&over, 3, 0);
        over.b_reuse = 4;
        EXPECT_TRUE(!htp_opbuf_stats_conserved(&over));
    }

    // 8) Aggregation is exact and additive.
    {
        htp_opbuf_stats a;
        htp_opbuf_stats_init(&a, 7, 1000);
        a.b_reuse = 2; a.map_n = 5; a.map_bytes = 500; a.drop_n = 1; a.drop_bytes = 64;
        a.m_vmem = 128; a.e_vmem = 500; a.scan_cmp = 476;
        htp_opbuf_stats b;
        htp_opbuf_stats_init(&b, 3, 1000);
        b.b_reuse = 3; b.scan_cmp = 12;
        htp_opbuf_stats acc;
        htp_opbuf_stats_zero(&acc);
        htp_opbuf_stats_add(&acc, &a);
        htp_opbuf_stats_add(&acc, &b);
        EXPECT_U64(acc.n_bufs, 10);
        EXPECT_U64(acc.b_reuse, 5);
        EXPECT_U64(acc.map_n, 5);
        EXPECT_U64(acc.map_bytes, 500);
        EXPECT_U64(acc.drop_n, 1);
        EXPECT_U64(acc.drop_bytes, 64);
        EXPECT_U64(acc.m_vmem, 128);
        EXPECT_U64(acc.e_vmem, 500);
        EXPECT_U64(acc.scan_cmp, 488);
        EXPECT_U64(acc.max_vmem, 2000); // sums, as documented for the total line
        EXPECT_TRUE(htp_opbuf_stats_conserved(&acc));
    }
}

int main() {
    // ── 1. normalize_prefetch_depth：负 depth / 上限 / BANK 太小 ──────────────
    // bank >= 2*depth + 8 才保留 depth；否则归一化成 0（预取静默关闭）。
    EXPECT_INT(config::normalize_prefetch_depth(40, -1), 0);     // 负 depth -> 0
    EXPECT_INT(config::normalize_prefetch_depth(40, -100), 0);
    EXPECT_INT(config::normalize_prefetch_depth(40, 0), 0);      // 0 = 关闭
    EXPECT_INT(config::normalize_prefetch_depth(40, 8), 8);      // 40 >= 24
    EXPECT_INT(config::normalize_prefetch_depth(23, 8), 0);      // 23 < 2*8+8：BANK 太小
    EXPECT_INT(config::normalize_prefetch_depth(24, 8), 8);      // 恰好够
    EXPECT_INT(config::normalize_prefetch_depth(39, 16), 0);     // 16 需要 bank >= 40
    EXPECT_INT(config::normalize_prefetch_depth(40, 16), 16);
    EXPECT_INT(config::normalize_prefetch_depth(100, 99), 16);   // 先夹到上限 16
    EXPECT_INT(config::normalize_prefetch_depth(-5, 4), 0);      // 退化 bank
    EXPECT_INT(config::max_prefetch_depth, 16);

    // ── 2. LAYER_WINDOWS：空串 / 未设置 / base 关闭 ──────────────────────────
    for (const char * empty : { (const char *) nullptr, "" }) {
        const auto parsed = parse_windows(empty);
        EXPECT_INT(parsed.widths.size(), 40);
        EXPECT_TRUE(all_equal(parsed.widths, 80));
        EXPECT_INT(parsed.ignored.size(), 0);
    }

    const auto no_base = parse_windows("0:128", 4, 0);           // 全局窗口关闭 -> 不看覆盖
    EXPECT_INT(no_base.widths.size(), 4);
    EXPECT_TRUE(all_equal(no_base.widths, 0));
    EXPECT_INT(no_base.ignored.size(), 0);

    const auto no_layers = parse_windows("0:128", 0, 80);        // 没有 MoE 层
    EXPECT_INT(no_layers.widths.size(), 0);

    // ── 3. LAYER_WINDOWS：合法项 / 空白 / 逗号 / 重复项 ─────────────────────
    const auto basic = parse_windows("0:128,1:80");
    EXPECT_INT(basic.widths[0], 128);
    EXPECT_INT(basic.widths[1], 80);
    EXPECT_INT(basic.widths[2], 80);                             // 未列出的层保持 base
    EXPECT_INT(basic.ignored.size(), 0);

    const auto spaced = parse_windows(" 0:128 , 1:96 ");
    EXPECT_INT(spaced.widths[0], 128);
    EXPECT_INT(spaced.widths[1], 96);
    EXPECT_INT(spaced.ignored.size(), 0);

    const auto commas = parse_windows(",,0:128,,,");
    EXPECT_INT(commas.widths[0], 128);
    EXPECT_INT(commas.ignored.size(), 0);

    const auto trailing = parse_windows("0:128,");
    EXPECT_INT(trailing.widths[0], 128);
    EXPECT_INT(trailing.ignored.size(), 0);

    const auto dup = parse_windows("0:100,0:200");               // 同层后者胜
    EXPECT_INT(dup.widths[0], 200);
    EXPECT_INT(dup.ignored.size(), 0);

    // ── 4. LAYER_WINDOWS：宽度夹紧 ───────────────────────────────────────────
    const auto clamp_lo = parse_windows("0:3", 40, 80, 8);       // < n_expert_used -> 抬到 8
    EXPECT_INT(clamp_lo.widths[0], 8);
    EXPECT_INT(clamp_lo.ignored.size(), 0);

    const auto clamp_hi = parse_windows("0:5000");               // > 1024 -> 夹到 1024
    EXPECT_INT(clamp_hi.widths[0], 1024);
    EXPECT_INT(clamp_hi.ignored.size(), 0);

    // ── 5. LAYER_WINDOWS：非法项（跳过 / 截断，语义与旧实现一致） ────────────
    const auto bad_first = parse_windows("abc:8");               // 缺层号 -> 停止解析
    EXPECT_TRUE(all_equal(bad_first.widths, 80));
    EXPECT_INT(bad_first.ignored.size(), 1);

    const auto missing_colon = parse_windows("0=128");
    EXPECT_TRUE(all_equal(missing_colon.widths, 80));
    EXPECT_INT(missing_colon.ignored.size(), 1);

    const auto missing_width = parse_windows("0:");
    EXPECT_TRUE(all_equal(missing_width.widths, 80));
    EXPECT_INT(missing_width.ignored.size(), 1);

    const auto stop_after_valid = parse_windows("0:128,garbage");
    EXPECT_INT(stop_after_valid.widths[0], 128);                 // 前面的合法项已生效
    EXPECT_INT(stop_after_valid.widths[1], 80);
    EXPECT_INT(stop_after_valid.ignored.size(), 1);

    const auto zero_width = parse_windows("0:0");                // 宽度必须 > 0 -> 跳过
    EXPECT_INT(zero_width.widths[0], 80);
    EXPECT_INT(zero_width.ignored.size(), 1);

    const auto negative_layer = parse_windows("-1:8,1:96");      // 负层号跳过，继续后面
    EXPECT_INT(negative_layer.widths[0], 80);
    EXPECT_INT(negative_layer.widths[1], 96);
    EXPECT_INT(negative_layer.ignored.size(), 1);

    const auto high_layer = parse_windows("40:8");               // 越界（合法范围 0..39）
    EXPECT_TRUE(all_equal(high_layer.widths, 80));
    EXPECT_INT(high_layer.ignored.size(), 1);

    const auto last_layer = parse_windows("39:8");
    EXPECT_INT(last_layer.widths[39], 8);
    EXPECT_INT(last_layer.ignored.size(), 0);

    // ── 6. LLAMA_MOE_EVICT / EVICT_LAYERS：合法值 ────────────────────────────
    const auto evict_default = config::parse_evict_spec(nullptr, "LLAMA_MOE_EVICT");
    EXPECT_TRUE(evict_default.error.empty());
    EXPECT_INT((int) evict_default.policy, (int) config::evict_policy::lru);
    EXPECT_INT(evict_default.decay, 0);

    const auto evict_empty = config::parse_evict_spec("", "LLAMA_MOE_EVICT");
    EXPECT_TRUE(evict_empty.error.empty());
    EXPECT_INT((int) evict_empty.policy, (int) config::evict_policy::lru);

    const auto evict_lfu = config::parse_evict_spec("lfu", "LLAMA_MOE_EVICT");
    EXPECT_TRUE(evict_lfu.error.empty());
    EXPECT_INT((int) evict_lfu.policy, (int) config::evict_policy::lfu);
    EXPECT_INT(evict_lfu.decay, 0);

    const auto evict_decay = config::parse_evict_spec("lfu512", "LLAMA_MOE_EVICT");
    EXPECT_INT((int) evict_decay.policy, (int) config::evict_policy::lfu);
    EXPECT_INT(evict_decay.decay, 512);
    EXPECT_TRUE(evict_decay.warning.empty());

    const auto evict_bad = config::parse_evict_spec("lru2", "LLAMA_MOE_EVICT");
    EXPECT_TRUE(!evict_bad.error.empty());                       // 仍按旧行为报错

    const auto evict_bad_decay = config::parse_evict_spec("lfuX", "LLAMA_MOE_EVICT");
    EXPECT_INT((int) evict_bad_decay.policy, (int) config::evict_policy::lfu);
    EXPECT_INT(evict_bad_decay.decay, 0);                        // 旧行为：解析失败 = 0
    EXPECT_TRUE(!evict_bad_decay.warning.empty());               // 但不再静默

    // ── 7. LLAMA_MOE_EVICT_LAYERS：空 / 合法 / 重复 / 非法 ───────────────────
    const auto layers_none = config::parse_evict_layers(nullptr);
    EXPECT_TRUE(layers_none.error.empty());
    EXPECT_INT(layers_none.layers.size(), 0);

    const auto layers_empty = config::parse_evict_layers("");
    EXPECT_TRUE(layers_empty.error.empty());
    EXPECT_INT(layers_empty.layers.size(), 0);

    const auto layers_ok = config::parse_evict_layers("0:lfu512,1:lru");
    EXPECT_TRUE(layers_ok.error.empty());
    EXPECT_INT(layers_ok.layers.size(), 2);
    EXPECT_INT(layers_ok.layers.at(0).decay, 512);
    EXPECT_INT((int) layers_ok.layers.at(1).policy, (int) config::evict_policy::lru);

    const auto layers_trailing = config::parse_evict_layers("0:lfu512,");
    EXPECT_TRUE(layers_trailing.error.empty());
    EXPECT_INT(layers_trailing.layers.size(), 1);

    const auto layers_dup = config::parse_evict_layers("3:lfu,3:lru");   // 后者胜
    EXPECT_TRUE(layers_dup.error.empty());
    EXPECT_INT(layers_dup.layers.size(), 1);
    EXPECT_INT((int) layers_dup.layers.at(3).policy, (int) config::evict_policy::lru);

    const auto layers_no_colon = config::parse_evict_layers("0");
    EXPECT_TRUE(!layers_no_colon.error.empty());
    EXPECT_INT(layers_no_colon.layers.size(), 0);

    const auto layers_negative = config::parse_evict_layers("-1:lru");
    EXPECT_TRUE(layers_negative.error.find("layer must be >= 0") != std::string::npos);

    const auto layers_bad_policy = config::parse_evict_layers("0:foo");
    EXPECT_TRUE(layers_bad_policy.error.find("must be lru | lfu | lfu<N>") != std::string::npos);

    const auto layers_non_numeric = config::parse_evict_layers("abc:lru");
    EXPECT_TRUE(layers_non_numeric.error.empty());
    EXPECT_INT(layers_non_numeric.layers.size(), 1);             // 旧行为：非数字层号 = 0
    EXPECT_INT(layers_non_numeric.warnings.size(), 1);           // 新行为：打一条告警

    const auto layers_bad_decay = config::parse_evict_layers("0:lfu512x");
    EXPECT_TRUE(layers_bad_decay.error.empty());
    EXPECT_INT(layers_bad_decay.layers.at(0).decay, 512);        // 旧行为：取数字前缀
    EXPECT_INT(layers_bad_decay.warnings.size(), 1);

    // ── 8. PREFETCH_REFRESH：每个前置条件单独覆盖 ───────────────────────────
    const auto ok = refresh_ok();
    EXPECT_INT(config::prefetch_refresh_missing(ok), 0);

    auto no_managers = ok;
    no_managers.managers_active = false;
    EXPECT_INT(config::prefetch_refresh_missing(no_managers), config::refresh_ready_window);

    auto no_window = ok;
    no_window.window_mode = false;
    EXPECT_INT(config::prefetch_refresh_missing(no_window), config::refresh_ready_window);

    auto split = ok;
    split.matmul_split = true;
    EXPECT_INT(config::prefetch_refresh_missing(split), config::refresh_split_zero);

    auto no_prefetch = ok;
    no_prefetch.prefetch_enabled = false;
    EXPECT_INT(config::prefetch_refresh_missing(no_prefetch), config::refresh_prefetch_read);

    auto no_read = ok;
    no_read.read_n = 0;
    EXPECT_INT(config::prefetch_refresh_missing(no_read), config::refresh_prefetch_read);

    auto h1 = ok;
    h1.horizon = 1;
    EXPECT_INT(config::prefetch_refresh_missing(h1), config::refresh_horizon_gt_one);

    auto h0 = ok;
    h0.horizon = 0;
    EXPECT_INT(config::prefetch_refresh_missing(h0), config::refresh_horizon_gt_one);

    auto budget = ok;
    budget.io_budget = true;
    EXPECT_INT(config::prefetch_refresh_missing(budget), config::refresh_io_budget_zero);

    auto front = ok;
    front.h1_front = true;
    EXPECT_INT(config::prefetch_refresh_missing(front), config::refresh_h1_front_zero);

    auto all_bad = ok;
    all_bad.managers_active  = false;
    all_bad.window_mode      = false;
    all_bad.matmul_split     = true;
    all_bad.prefetch_enabled = false;
    all_bad.read_n           = 0;
    all_bad.horizon          = 1;
    all_bad.io_budget        = true;
    all_bad.h1_front         = true;
    const unsigned every = config::refresh_ready_window | config::refresh_split_zero |
                           config::refresh_horizon_gt_one | config::refresh_prefetch_read |
                           config::refresh_io_budget_zero | config::refresh_h1_front_zero;
    EXPECT_INT(config::prefetch_refresh_missing(all_bad), every);
    EXPECT_TRUE(config::refresh_missing_text(every).find("horizon>1") != std::string::npos);

    // BANK 太小 -> depth 归一化成 0 -> REFRESH 的 prefetch/read 前置条件随之不满足。
    // 这条把「静默归一化」和「REFRESH 报错」串起来：配置笔误不会再白跑 A/B。
    const int too_small = config::normalize_prefetch_depth(20, 8);
    EXPECT_INT(too_small, 0);
    auto bank_case = ok;
    bank_case.prefetch_enabled = too_small > 0;
    bank_case.read_n = too_small;
    EXPECT_TRUE((config::prefetch_refresh_missing(bank_case) & config::refresh_prefetch_read) != 0);

    test_prefetch_merge_contract();
    test_opbuf_stats_contract();

    if (failures) {
        std::printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("all config parse checks passed\n");
    return 0;
}
