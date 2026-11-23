#pragma once

#include "llama-moe-slot-manager.h"

#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>

namespace llama_moe {

struct evict_spec {
    slot_manager::evict_policy policy = slot_manager::evict_policy::lru;
    uint32_t decay_period = 0;
};

namespace detail {

// Process-wide one-shot logging. The callback runs only for the thread that wins
// the atomic exchange; every other caller returns immediately.
//
// Deliberately not std::call_once: the callback may invoke a user log handler that
// waits for another MoE runtime's initialization. If other callers blocked on the
// same once-flag until the callback returned, the two runtimes could deadlock (the
// other runtime cannot finish initialization until the handler returns). The relaxed
// exchange only elects the logger; whatever is logged must be safely published by the
// caller (e.g. function-local statics initialized first).
template <typename F>
inline void invoke_log_once(std::atomic<bool> & logged, F && log) {
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        log();
    }
}

inline uint32_t parse_decimal_uint32(std::string_view text, const std::string & error) {
    if (text.empty()) {
        throw std::runtime_error(error);
    }
    for (const char c : text) {
        if (c < '0' || c > '9') {
            throw std::runtime_error(error);
        }
    }

    uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::runtime_error(error);
    }
    return value;
}

} // namespace detail

inline evict_spec parse_evict_spec(std::string_view text, const char * what, bool empty_is_default = false) {
    const std::string error = std::string(what) + " must be lru | lfu | lfu<N>";
    if (text.empty() && empty_is_default) {
        return {};
    }
    if (text == "lru") {
        return {slot_manager::evict_policy::lru, 0};
    }
    if (text == "lfu") {
        return {slot_manager::evict_policy::lfu, 0};
    }
    if (text.size() > 3 && text.substr(0, 3) == "lfu") {
        return {slot_manager::evict_policy::lfu, detail::parse_decimal_uint32(text.substr(3), error)};
    }
    throw std::runtime_error(error);
}

inline std::unordered_map<int32_t, evict_spec> parse_evict_layers(
        std::string_view text, const char * what = "LLAMA_MOE_EVICT_LAYERS") {
    std::unordered_map<int32_t, evict_spec> out;
    if (text.empty()) {
        return out;
    }

    std::size_t pos = 0;
    while (pos <= text.size()) {
        const std::size_t comma = text.find(',', pos);
        const std::string_view item = text.substr(
            pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos);
        pos = comma == std::string_view::npos ? text.size() + 1 : comma + 1;

        const std::size_t colon = item.find(':');
        if (item.empty() || colon == std::string_view::npos) {
            throw std::runtime_error(std::string(what) + " expects layer:policy[,...]");
        }

        const std::string layer_error = std::string(what) +
            ": layer must be a decimal integer in 0..2147483647";
        const uint32_t parsed_layer = detail::parse_decimal_uint32(item.substr(0, colon), layer_error);
        if (parsed_layer > (uint32_t) std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error(layer_error);
        }
        out[(int32_t) parsed_layer] = parse_evict_spec(item.substr(colon + 1), what);
    }
    return out;
}

} // namespace llama_moe
