// Host unit test for the RTT v1/v2 JSON formatters (T-0327).
//
// Pins two contracts that cannot be checked on a phone without a device run:
//   1. GGML_HEXAGON_RTT_TRACE_V2 unset -> the v1 line is byte-for-byte the
//      historical fprintf() output (same keys, order, decimal formatting and
//      trailing newline).  The three golden strings below were generated from
//      the pre-change fprintf() implementation (see the run's artifacts/).
//   2. V2=1 -> the same v1 body with schema bumped to ggml.hex.rtt.v2 plus four
//      appended graph fields.  A dropped record must keep null graph fields
//      (never fabricate identity/times).
//
// Compile + run (host, g++ >= C++17), from the repo root:
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror -I ggml/src/ggml-hexagon -o /tmp/test-hexagon-rtt tests/test-hexagon-rtt.cpp
//   /tmp/test-hexagon-rtt
//
// Exit code 0 and "PASS" on success; any failure prints FAIL and exits 1.

#include "ggml-hexagon-rtt.h"

#include <cstdint>
#include <cstdio>
#include <string>

static int failures = 0;

static void expect_str_eq(const char * what, const std::string & got, const std::string & want, int line) {
    if (got == want) {
        return;
    }
    ++failures;
    std::printf("FAIL %s:%d: %s mismatch\n  got  (%zu): %s\n  want (%zu): %s\n",
                __FILE__, line, what, got.size(), got.c_str(), want.size(), want.c_str());
    const size_t n = got.size() < want.size() ? got.size() : want.size();
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != want[i]) {
            std::printf("  first diff at byte %zu: got 0x%02x '%c', want 0x%02x '%c'\n",
                        i, (unsigned char) got[i], got[i], (unsigned char) want[i], want[i]);
            break;
        }
    }
}

static void expect_true(const char * what, bool got, int line) {
    if (!got) {
        ++failures;
        std::printf("FAIL %s:%d: %s is false\n", __FILE__, line, what);
    }
}

static void expect_false(const char * what, bool got, int line) {
    if (got) {
        ++failures;
        std::printf("FAIL %s:%d: %s is true\n", __FILE__, line, what);
    }
}

#define EXPECT_STR_EQ(got, want) expect_str_eq(#got, (got), (want), __LINE__)
#define EXPECT_TRUE(expr)        expect_true(#expr, (bool) (expr), __LINE__)
#define EXPECT_FALSE(expr)       expect_false(#expr, (bool) (expr), __LINE__)

// Fixed slot shared by the v1 and v2 tests.
static ggml_hexagon_rtt_slot make_slot() {
    ggml_hexagon_rtt_slot s;
    s.seq            = 42;
    s.t_write_ns     = 1234567890123ull;
    s.t_write_ret_ns = 1234567890456ull;
    s.t_first_poll_ns = 1234567891000ull;
    s.t_done_ns      = 1234567899999ull;
    s.n_ops          = 8;
    s.n_ops_rsp      = 3;
    s.rsp_status     = 0;
    s.oppoll         = 1;
    s.used           = true;
    s.has_first_poll = true;
    return s;
}

// Golden output of the pre-change fprintf() path, captured verbatim.
static const char * V1_GOLDEN_FIRST_POLL =
    "{\"schema\":\"ggml.hex.rtt.v1\",\"session\":7,\"role\":\"main\",\"seq\":42,"
    "\"t_write_ns\":1234567890123,\"t_write_ret_ns\":1234567890456,"
    "\"t_first_poll_ns\":1234567891000,\"t_done_ns\":1234567899999,"
    "\"rtt_ns\":9876,\"write_ns\":333,\"poll_wait_ns\":8999,"
    "\"n_ops\":8,\"n_ops_rsp\":3,\"rsp_status\":0,\"oppoll\":1,\"dropped\":false}\n";

static const char * V1_GOLDEN_NO_FIRST_POLL =
    "{\"schema\":\"ggml.hex.rtt.v1\",\"session\":7,\"role\":\"main\",\"seq\":43,"
    "\"t_write_ns\":1234567890123,\"t_write_ret_ns\":1234567890456,"
    "\"t_first_poll_ns\":null,\"t_done_ns\":1234567899999,"
    "\"rtt_ns\":9876,\"write_ns\":333,\"poll_wait_ns\":null,"
    "\"n_ops\":8,\"n_ops_rsp\":3,\"rsp_status\":0,\"oppoll\":1,\"dropped\":false}\n";

static const char * V1_GOLDEN_DROPPED =
    "{\"schema\":\"ggml.hex.rtt.v1\",\"session\":7,\"role\":\"main\",\"seq\":44,"
    "\"t_write_ns\":null,\"t_write_ret_ns\":null,\"t_first_poll_ns\":null,"
    "\"t_done_ns\":null,\"rtt_ns\":null,\"write_ns\":null,\"poll_wait_ns\":null,"
    "\"n_ops\":null,\"n_ops_rsp\":5,\"rsp_status\":2,\"oppoll\":0,\"dropped\":true}\n";

static void test_v1_golden() {
    const ggml_hexagon_rtt_slot s = make_slot();
    EXPECT_STR_EQ(ggml_hexagon_rtt_json_v1(s, 7), V1_GOLDEN_FIRST_POLL);

    ggml_hexagon_rtt_slot no_poll = s;
    no_poll.seq            = 43;
    no_poll.has_first_poll = false;
    EXPECT_STR_EQ(ggml_hexagon_rtt_json_v1(no_poll, 7), V1_GOLDEN_NO_FIRST_POLL);

    ggml_hexagon_rtt_slot dropped;
    dropped.seq        = 44;
    dropped.n_ops_rsp  = 5;
    dropped.rsp_status = 2;
    dropped.oppoll     = 0;
    dropped.dropped    = true;
    EXPECT_STR_EQ(ggml_hexagon_rtt_json_v1(dropped, 7), V1_GOLDEN_DROPPED);

    // v1 never grows graph keys.
    EXPECT_FALSE(ggml_hexagon_rtt_json_v1(s, 7).find("graph_") != std::string::npos);
}

static void test_v2_boundary_zero_graph() {
    // graph_uid=0 / cache_hit=false / prep_ns=0 / nodes=0 is the "no uid, no
    // prep measured" case; the keys must still be present with real zeros.
    const ggml_hexagon_rtt_slot s = make_slot();
    const std::string got = ggml_hexagon_rtt_json_v2(s, 7);
    const std::string want =
        "{\"schema\":\"ggml.hex.rtt.v2\",\"session\":7,\"role\":\"main\",\"seq\":42,"
        "\"t_write_ns\":1234567890123,\"t_write_ret_ns\":1234567890456,"
        "\"t_first_poll_ns\":1234567891000,\"t_done_ns\":1234567899999,"
        "\"rtt_ns\":9876,\"write_ns\":333,\"poll_wait_ns\":8999,"
        "\"n_ops\":8,\"n_ops_rsp\":3,\"rsp_status\":0,\"oppoll\":1,\"dropped\":false,"
        "\"graph_uid\":0,\"graph_cache_hit\":false,\"graph_prep_ns\":0,\"graph_nodes\":0}\n";
    EXPECT_STR_EQ(got, want);

    // v1 keys stay in place and the four v2 keys are appended exactly once.
    EXPECT_TRUE(got.find("\"schema\":\"ggml.hex.rtt.v2\"") != std::string::npos);
    EXPECT_TRUE(got.find("\"dropped\":false,\"graph_uid\":0") != std::string::npos);
    EXPECT_TRUE(got.find("\"graph_uid\"") != std::string::npos);
    EXPECT_TRUE(got.find("\"graph_cache_hit\"") != std::string::npos);
    EXPECT_TRUE(got.find("\"graph_prep_ns\"") != std::string::npos);
    EXPECT_TRUE(got.find("\"graph_nodes\"") != std::string::npos);
}

static void test_v2_attributed() {
    ggml_hexagon_rtt_slot s = make_slot();
    // 2^53 + 1 cannot be represented exactly by a double, so a formatter that
    // went through double/printf("%g") would round it to ...992.
    s.graph_uid       = 9007199254740993ull;
    s.graph_cache_hit = true;
    s.graph_prep_ns   = 18446744073709551615ull; // UINT64_MAX
    s.graph_nodes     = 81;
    const std::string got = ggml_hexagon_rtt_json_v2(s, 7);
    const std::string want =
        "{\"schema\":\"ggml.hex.rtt.v2\",\"session\":7,\"role\":\"main\",\"seq\":42,"
        "\"t_write_ns\":1234567890123,\"t_write_ret_ns\":1234567890456,"
        "\"t_first_poll_ns\":1234567891000,\"t_done_ns\":1234567899999,"
        "\"rtt_ns\":9876,\"write_ns\":333,\"poll_wait_ns\":8999,"
        "\"n_ops\":8,\"n_ops_rsp\":3,\"rsp_status\":0,\"oppoll\":1,\"dropped\":false,"
        "\"graph_uid\":9007199254740993,\"graph_cache_hit\":true,"
        "\"graph_prep_ns\":18446744073709551615,\"graph_nodes\":81}\n";
    EXPECT_STR_EQ(got, want);

    // No scientific notation and no locale grouping separators in numbers.
    EXPECT_FALSE(got.find("e+") != std::string::npos);
    EXPECT_FALSE(got.find("e-") != std::string::npos);
    EXPECT_FALSE(got.find("E+") != std::string::npos);
    EXPECT_FALSE(got.find("1,234") != std::string::npos);
    EXPECT_FALSE(got.find("18,446") != std::string::npos);
}

static void test_v2_dropped_nulls() {
    ggml_hexagon_rtt_slot dropped;
    dropped.seq        = 44;
    dropped.n_ops_rsp  = 5;
    dropped.rsp_status = 2;
    dropped.oppoll     = 0;
    dropped.dropped    = true;
    // Even if stale graph data were present, a dropped record must emit null.
    dropped.graph_uid       = 1234;
    dropped.graph_cache_hit = true;
    dropped.graph_prep_ns   = 5678;
    dropped.graph_nodes     = 81;

    const std::string got = ggml_hexagon_rtt_json_v2(dropped, 7);
    const std::string want =
        "{\"schema\":\"ggml.hex.rtt.v2\",\"session\":7,\"role\":\"main\",\"seq\":44,"
        "\"t_write_ns\":null,\"t_write_ret_ns\":null,\"t_first_poll_ns\":null,"
        "\"t_done_ns\":null,\"rtt_ns\":null,\"write_ns\":null,\"poll_wait_ns\":null,"
        "\"n_ops\":null,\"n_ops_rsp\":5,\"rsp_status\":2,\"oppoll\":0,\"dropped\":true,"
        "\"graph_uid\":null,\"graph_cache_hit\":null,\"graph_prep_ns\":null,\"graph_nodes\":null}\n";
    EXPECT_STR_EQ(got, want);

    // The dropped record still carries the v1 response counters, never times.
    EXPECT_FALSE(got.find("\"t_done_ns\":null") == std::string::npos);
    EXPECT_FALSE(got.find("\"graph_uid\":null") == std::string::npos);
    EXPECT_FALSE(got.find("\"graph_cache_hit\":null") == std::string::npos);
}

static void test_v2_preserves_v1_body() {
    // Apart from the schema value and the appended tail, v2 must be the v1
    // body byte-for-byte: this is the compatibility guarantee for existing
    // ggml.hex.rtt.v1 parsers that read known keys positionally.
    const ggml_hexagon_rtt_slot s = make_slot();
    std::string v1 = ggml_hexagon_rtt_json_v1(s, 7);
    const std::string v2 = ggml_hexagon_rtt_json_v2(s, 7);

    const std::string v1_schema = "\"schema\":\"ggml.hex.rtt.v1\"";
    const std::string v2_schema = "\"schema\":\"ggml.hex.rtt.v2\"";
    const std::string::size_type p = v1.find(v1_schema);
    EXPECT_TRUE(p != std::string::npos);
    v1.replace(p, v1_schema.size(), v2_schema);
    v1.resize(v1.size() - 2); // strip "}\n"
    EXPECT_TRUE(v2.compare(0, v1.size(), v1) == 0);
    EXPECT_TRUE(v2.compare(v1.size(), std::string::npos,
                           ",\"graph_uid\":0,\"graph_cache_hit\":false,"
                           "\"graph_prep_ns\":0,\"graph_nodes\":0}\n") == 0);
}

int main() {
    test_v1_golden();
    test_v2_boundary_zero_graph();
    test_v2_attributed();
    test_v2_dropped_nulls();
    test_v2_preserves_v1_body();

    if (failures != 0) {
        std::printf("FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS: RTT v1 golden + v2 additive fields\n");
    return 0;
}
