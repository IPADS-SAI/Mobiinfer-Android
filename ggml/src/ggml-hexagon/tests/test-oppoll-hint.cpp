// Host-only unit test for the GGML_HEXAGON_OPPOLL=2 decode/prefill hint rule.
//
// The Hexagon backend itself is cross-compiled, so this test includes only the
// dependency-free decision header.  Build/run with:
//   ggml/src/ggml-hexagon/tests/run-oppoll-hint-test.sh
//
// The "constructed graph" cases feed the exact helper the backend calls
// (ggml_hexagon_oppoll_add_node) with the activation shapes of a Qwen3-MoE
// decode/prefill graph, so the printed log line is the same one a real
// HEX_VERBOSE run emits:
//   ggml-hex: oppoll=2 poll_decode=%d n_tokens=%d
//
// MUL_MAT_ID rows come from the ids tensor (src[2]->ne[1]), not from the
// expert activation src[1] whose ne[1] is n_expert_used.  The down projection
// is [n_ff, n_expert_used, n_tokens] = [512, 8, 4] with ids [8, 4], so the
// token count is 4 (not 8*4 = 32).

#include "ggml-hexagon-oppoll.h"

#include <cstdio>
#include <initializer_list>

static int g_failures = 0;

struct mock_node {
    int     op;
    int64_t ne[4];
    int64_t ids_ne[4] = {0, 0, 0, 0};  // MUL_MAT_ID src[2] (ids) shape
    bool    has_ids   = false;
};

static ggml_hexagon_oppoll_hint graph_hint(std::initializer_list<mock_node> nodes) {
    ggml_hexagon_oppoll_hint h;
    for (const mock_node & n : nodes) {
        ggml_hexagon_oppoll_add_node(h, n.op, n.ne, n.has_ids ? n.ids_ne : nullptr);
    }
    return h;
}

static void expect(bool got, bool want, const char * what) {
    if (got != want) {
        std::printf("FAIL %-64s got=%d want=%d\n", what, got ? 1 : 0, want ? 1 : 0);
        ++g_failures;
    } else {
        std::printf("ok   %-64s %d\n", what, got ? 1 : 0);
    }
}

static ggml_hexagon_oppoll_hint expect_graph(const char * name, std::initializer_list<mock_node> nodes, bool want) {
    const ggml_hexagon_oppoll_hint h = graph_hint(nodes);
    std::printf("ggml-hex: oppoll=2 poll_decode=%d n_tokens=%d\n",
                h.poll_decode() ? 1 : 0, (int) h.n_tokens);
    std::printf("     constructed graph: %s (candidates=%u consistent=%d)\n",
                name, (unsigned) h.n_candidates, h.consistent ? 1 : 0);
    expect(h.poll_decode(), want, name);
    return h;
}

int main() {
    // No MUL_MAT/MUL_MAT_ID candidate: block.
    expect(graph_hint({}).poll_decode(), false, "no activation operand -> block");

    // Decode-sized graphs spin.
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 1, 1, 1}}}).poll_decode(), true,
           "1-row draft decode");
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 4, 1, 1}, true}}).poll_decode(), true,
           "4-row speculative verify (MUL_MAT_ID via ids)");
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 8, 1, 1}}}).poll_decode(), true,
           "8 rows (inclusive threshold)");

    // Prefill/load graphs block.
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 9, 1, 1}}}).poll_decode(), false,
           "9 rows (just over threshold)");
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 128, 1, 1}}}).poll_decode(), false,
           "128-token prefill");

    // Conflicting candidates and degenerate values are never decode.
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 4, 1, 1}},
                       {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 128, 1}, {8, 128, 1, 1}, true}}).poll_decode(),
           false, "mixed decode/prefill graph -> block");
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 8, 1, 1}},
                       {GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 1, 1, 1}}}).poll_decode(),
           false, "conflicting rows -> block");
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 0, 1, 1}}}).poll_decode(), false,
           "zero rows -> block");
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_OTHER, {2048, 4, 1, 1}}}).poll_decode(), false,
           "non-matmul node ignored -> no candidate");

    // 3D MUL_MAT activation: rows are ne[1]*ne[2]*ne[3].
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 2, 2, 1}}}).poll_decode(), true,
           "2x2x1 = 4 activation rows");
    expect(graph_hint({{GGML_HEXAGON_OPPOLL_OP_MUL_MAT, {2048, 2, 2, 3}}}).poll_decode(), false,
           "2x2x3 = 12 rows -> prefill");

    // Mode 0/1 are the historical policies and must not depend on the hint.
    // Mode 2 follows the per-batch decode classification (missing -> block).
    expect(ggml_hexagon_oppoll_spin(0, true), false, "mode 0 blocks even with decode hint");
    expect(ggml_hexagon_oppoll_spin(0, false), false, "mode 0 always blocks");
    expect(ggml_hexagon_oppoll_spin(1, true), true, "mode 1 spins with decode hint");
    expect(ggml_hexagon_oppoll_spin(1, false), true, "mode 1 always spins");
    expect(ggml_hexagon_oppoll_spin(2, true), true, "mode 2 spins for decode");
    expect(ggml_hexagon_oppoll_spin(2, false), false, "mode 2 blocks for prefill/load");

    // Parsing/normalization: 0/1/2 pass through, >2 (and invalid negatives)
    // clamp to 2 with a warning flag; the engine default stays 0.
    bool clamped = false;
    expect(ggml_hexagon_oppoll_normalize(0, clamped) == 0 && !clamped, true, "normalize 0 -> 0");
    expect(ggml_hexagon_oppoll_normalize(1, clamped) == 1 && !clamped, true, "normalize 1 -> 1");
    expect(ggml_hexagon_oppoll_normalize(2, clamped) == 2 && !clamped, true, "normalize 2 -> 2");
    expect(ggml_hexagon_oppoll_normalize(3, clamped) == 2 && clamped, true, "normalize 3 -> 2 (warn)");
    expect(ggml_hexagon_oppoll_normalize(42, clamped) == 2 && clamped, true, "normalize 42 -> 2 (warn)");
    expect(ggml_hexagon_oppoll_normalize(-1, clamped) == 2 && clamped, true, "normalize -1 -> 2 (warn)");

    // ---------------------------------------------------------------------
    // MUL_MAT_ID row accounting: token rows come from ids src[2]->ne[1].
    // These are the exact real shapes from T-0341 / T-0297 (top-8, 4-row
    // speculative verify) and the regression the old src[1] product caused.
    // ---------------------------------------------------------------------

    // 1. target verify4: 4-row MUL_MAT + down MUL_MAT_ID [512,8,4] + ids [8,4].
    {
        const auto h = expect_graph("verify4 target: MUL_MAT 4 + MUL_MAT_ID ids 4 -> spin",
            {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 4, 1, 1}, true},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}}}, true);
        expect(h.n_tokens == 4, true, "verify4 target n_tokens == 4");
        expect(h.consistent, true, "verify4 target consistent");
    }

    // 2. prefill128: ids carry 128 tokens even though n_expert_used stays 8.
    {
        const auto h = expect_graph("prefill128: MUL_MAT_ID ids 128 -> block",
            {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 128, 1}, {8, 128, 1, 1}, true}}, false);
        expect(h.n_tokens == 128, true, "prefill128 n_tokens == 128");
    }

    // 3. draft1: one token, ids [8,1].
    {
        const auto h = expect_graph("draft1: MUL_MAT_ID ids 1 -> spin",
            {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 1, 1}, {8, 1, 1, 1}, true}}, true);
        expect(h.n_tokens == 1, true, "draft1 n_tokens == 1");
    }

    // 4. Missing src[2] must block.  The old rule would have guessed 8*4 = 32;
    //    there is deliberately no fallback to src[1]'s product.
    {
        const auto h = expect_graph("missing ids: MUL_MAT_ID src[2]=null -> block (not 32/4)",
            {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}}}, false);
        expect(h.have_candidate, true, "missing ids still records a candidate");
        expect(h.n_tokens == 0, true, "missing ids n_tokens == 0 (no src[1] fallback)");
    }

    // 4b. A non-positive ids token dim is also 0 rows.
    {
        const auto h = expect_graph("zero-token ids: MUL_MAT_ID ids[1]=0 -> block",
            {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 0, 1, 1}, true}}, false);
        expect(h.n_tokens == 0, true, "zero-token ids n_tokens == 0");
    }

    // 5. Conflict: a 4-row MUL_MAT and a MUL_MAT_ID whose ids say 16 tokens.
    expect_graph("conflict: MUL_MAT 4 + MUL_MAT_ID ids 16 -> block",
        {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 16, 1}, {8, 16, 1, 1}, true}}, false);

    // 6. The old fixture {2048,4,1,1} as MUL_MAT_ID src[1] is NOT a token count.
    //    Without ids it now blocks; the assertion moved to the ids-shaped cases
    //    above rather than being dropped.
    expect_graph("old MUL_MAT_ID src[1] shape without ids -> block",
        {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {2048, 4, 1, 1}}}, false);

    // 7. A 40-layer-style target mix: 4 dense MUL_MAT rows + 4 MoE down
    //    MUL_MAT_ID rows all saying 4 tokens -> spin; the same downs with 128
    //    tokens -> block.
    {
        const auto h = expect_graph("40-layer mixed target 4 rows -> spin",
            {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 4, 1, 1}, true},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 4, 1, 1}, true},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 4, 1, 1}, true},
             {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 4, 1, 1}, true}}, true);
        expect(h.n_candidates == 8, true, "40-layer mixed target candidates == 8");
    }
    expect_graph("40-layer mixed target 128-token downs -> block",
        {{GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 128, 1}, {8, 128, 1, 1}, true},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 128, 1}, {8, 128, 1, 1}, true},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 128, 1}, {8, 128, 1, 1}, true},
         {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 128, 1}, {8, 128, 1, 1}, true}}, false);

    // Constructed Qwen3-MoE graphs with the real MUL_MAT_ID shapes: every
    // matmul operand now reports the same token count (the real invariant the
    // backend relies on).
    const mock_node verify4[] = {
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},              // attn q/k/v/o, router
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 4, 1}, {8, 4, 1, 1}, true},  // down projection
        {GGML_HEXAGON_OPPOLL_OP_OTHER,      {2048, 4, 1, 1}},              // add/norm, ignored
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 4, 1, 1}},              // lm_head
    };
    const mock_node prefill128[] = {
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 128, 1, 1}},
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 128, 1}, {8, 128, 1, 1}, true},
        {GGML_HEXAGON_OPPOLL_OP_OTHER,      {2048, 128, 1, 1}},
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 128, 1, 1}},
    };
    const mock_node draft1[] = {
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 1, 1, 1}},
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT_ID, {512, 8, 1, 1}, {8, 1, 1, 1}, true},
        {GGML_HEXAGON_OPPOLL_OP_MUL_MAT,    {2048, 1, 1, 1}},
    };
    expect_graph("Qwen3-MoE 4-row verify", {verify4[0], verify4[1], verify4[2], verify4[3]}, true);
    expect_graph("Qwen3-MoE 128-token prefill", {prefill128[0], prefill128[1], prefill128[2], prefill128[3]}, false);
    expect_graph("Qwen3-MoE 1-token draft decode", {draft1[0], draft1[1], draft1[2]}, true);

    if (g_failures != 0) {
        std::printf("oppoll-hint: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("oppoll-hint: all checks passed\n");
    return 0;
}
