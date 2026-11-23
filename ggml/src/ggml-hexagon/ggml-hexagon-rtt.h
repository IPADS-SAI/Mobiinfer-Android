#ifndef GGML_HEXAGON_RTT_H
#define GGML_HEXAGON_RTT_H

#include <cstdint>
#include <string>

// Host-only RTT trace formatting (T-0327, follow-up to T-0181/T-0306).
//
// v1 schema (ggml.hex.rtt.v1) is the wire format that has been emitted since
// T-0181.  ggml_hexagon_rtt_json_v1() must stay byte-for-byte identical to the
// fprintf() sequence it replaced: same keys, same order, same decimal
// formatting, same trailing newline.  tools/moe/analyze_rtt.py and any phone
// logs already in the archive depend on that.
//
// v2 (ggml.hex.rtt.v2) is purely additive.  It keeps every v1 key in the same
// position and appends the graph attribution fields after "dropped":
//
//   "graph_uid":<u64>|null
//   "graph_cache_hit":<true|false>|null
//   "graph_prep_ns":<u64>|null
//   "graph_nodes":<u32>|null
//
// A dropped record (ring wrap / response without a live slot) never fabricates
// times or graph identity, so all four v2 fields are null there.  The v1
// dropped record keeps its historical shape (n_ops_rsp/rsp_status/oppoll are
// carried, every timestamp and n_ops is null).
//
// This header is deliberately dependency-free (no ggml / DSP / Hexagon SDK) so
// the formatting can be unit-tested on the host.  See tests/test-hexagon-rtt.cpp.

#define GGML_HEXAGON_RTT_RING 256

struct ggml_hexagon_rtt_slot {
    // v1 fields.
    uint64_t seq             = 0;
    uint64_t t_write_ns      = 0;
    uint64_t t_write_ret_ns  = 0;
    uint64_t t_first_poll_ns = 0;
    uint64_t t_done_ns       = 0;
    uint32_t n_ops           = 0;
    uint32_t n_ops_rsp       = 0;
    uint32_t rsp_status      = 0;
    uint32_t oppoll          = 0;
    bool     used            = false;
    bool     has_first_poll  = false;

    // v1 record kind: true -> all times null, dropped:true (never fabricate).
    bool     dropped         = false;

    // v2 additive fields: the graph that opened this batch (first graph if a
    // batch spans graphs) and the host-side prep cost paid for it.
    uint64_t graph_uid       = 0;
    uint64_t graph_prep_ns   = 0;
    uint32_t graph_nodes     = 0;
    bool     graph_cache_hit = false;
};

inline const char * ggml_hexagon_rtt_bool(bool v) {
    return v ? "true" : "false";
}

inline std::string ggml_hexagon_rtt_u64_or_null(bool present, uint64_t v) {
    return present ? std::to_string(v) : std::string("null");
}

// Byte-for-byte replacement for the historical v1 fprintf() blocks.
inline std::string ggml_hexagon_rtt_json_v1(const ggml_hexagon_rtt_slot & s, uint32_t session_id) {
    std::string out;
    out.reserve(320);

    out += "{\"schema\":\"ggml.hex.rtt.v1\",\"session\":";
    out += std::to_string(session_id);
    out += ",\"role\":\"main\",\"seq\":";
    out += std::to_string(s.seq);

    if (s.dropped) {
        out += ",\"t_write_ns\":null,\"t_write_ret_ns\":null,\"t_first_poll_ns\":null,"
               "\"t_done_ns\":null,\"rtt_ns\":null,\"write_ns\":null,\"poll_wait_ns\":null,"
               "\"n_ops\":null,\"n_ops_rsp\":";
        out += std::to_string(s.n_ops_rsp);
        out += ",\"rsp_status\":";
        out += std::to_string(s.rsp_status);
        out += ",\"oppoll\":";
        out += std::to_string(s.oppoll);
        out += ",\"dropped\":true}\n";
        return out;
    }

    // Unsigned subtraction is intentional: the caller only fills a slot after
    // t_write_ret >= t_write and t_done >= t_write (monotonic clock).
    const uint64_t rtt_ns        = s.t_done_ns - s.t_write_ns;
    const uint64_t write_ns      = s.t_write_ret_ns - s.t_write_ns;
    const uint64_t poll_wait_ns  = s.t_done_ns - s.t_first_poll_ns;

    out += ",\"t_write_ns\":";
    out += std::to_string(s.t_write_ns);
    out += ",\"t_write_ret_ns\":";
    out += std::to_string(s.t_write_ret_ns);
    out += ",\"t_first_poll_ns\":";
    out += ggml_hexagon_rtt_u64_or_null(s.has_first_poll, s.t_first_poll_ns);
    out += ",\"t_done_ns\":";
    out += std::to_string(s.t_done_ns);
    out += ",\"rtt_ns\":";
    out += std::to_string(rtt_ns);
    out += ",\"write_ns\":";
    out += std::to_string(write_ns);
    out += ",\"poll_wait_ns\":";
    out += ggml_hexagon_rtt_u64_or_null(s.has_first_poll, poll_wait_ns);
    out += ",\"n_ops\":";
    out += std::to_string(s.n_ops);
    out += ",\"n_ops_rsp\":";
    out += std::to_string(s.n_ops_rsp);
    out += ",\"rsp_status\":";
    out += std::to_string(s.rsp_status);
    out += ",\"oppoll\":";
    out += std::to_string(s.oppoll);
    out += ",\"dropped\":false}\n";
    return out;
}

// v1 keys in v1 order, schema bumped to v2, graph fields appended at the end.
inline std::string ggml_hexagon_rtt_json_v2(const ggml_hexagon_rtt_slot & s, uint32_t session_id) {
    std::string out = ggml_hexagon_rtt_json_v1(s, session_id);

    // Replace the schema value in place.  The prefix is fixed and short, so a
    // find+replace is both safe and keeps the v1 body as the single source of
    // truth for key order / number formatting.
    static const std::string v1_schema = "\"schema\":\"ggml.hex.rtt.v1\"";
    static const std::string v2_schema = "\"schema\":\"ggml.hex.rtt.v2\"";
    const std::string::size_type pos = out.find(v1_schema);
    if (pos != std::string::npos) {
        out.replace(pos, v1_schema.size(), v2_schema);
    }

    // Strip the closing "}\n" so the graph fields land after the last v1 key.
    const std::string::size_type end = out.size() >= 2 ? out.size() - 2 : std::string::npos;
    if (end == std::string::npos) {
        return out;
    }
    out.resize(end);

    if (s.dropped) {
        out += ",\"graph_uid\":null,\"graph_cache_hit\":null,"
               "\"graph_prep_ns\":null,\"graph_nodes\":null}\n";
        return out;
    }

    out += ",\"graph_uid\":";
    out += std::to_string(s.graph_uid);
    out += ",\"graph_cache_hit\":";
    out += ggml_hexagon_rtt_bool(s.graph_cache_hit);
    out += ",\"graph_prep_ns\":";
    out += std::to_string(s.graph_prep_ns);
    out += ",\"graph_nodes\":";
    out += std::to_string(s.graph_nodes);
    out += "}\n";
    return out;
}

#endif // GGML_HEXAGON_RTT_H
