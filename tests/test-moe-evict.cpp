#include "../src/llama-moe-evict.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

using llama_moe::evict_spec;
using llama_moe::parse_evict_layers;
using llama_moe::parse_evict_spec;
using llama_moe::slot_manager;

static int failures = 0;

static void expect_spec(std::string_view text, slot_manager::evict_policy policy, uint32_t decay,
                        bool empty_is_default = false) {
    try {
        const evict_spec got = parse_evict_spec(text, "TEST_EVICT", empty_is_default);
        if (got.policy != policy || got.decay_period != decay) {
            std::printf("FAIL parse_evict_spec(\"%.*s\") = policy=%d decay=%u, want policy=%d decay=%u\n",
                        (int) text.size(), text.data(), (int) got.policy, got.decay_period,
                        (int) policy, decay);
            ++failures;
        }
    } catch (const std::exception & e) {
        std::printf("FAIL parse_evict_spec(\"%.*s\") threw: %s\n",
                    (int) text.size(), text.data(), e.what());
        ++failures;
    }
}

static void expect_spec_error(std::string_view text, bool empty_is_default = false) {
    try {
        const evict_spec got = parse_evict_spec(text, "TEST_EVICT", empty_is_default);
        std::printf("FAIL parse_evict_spec(\"%.*s\") accepted policy=%d decay=%u\n",
                    (int) text.size(), text.data(), (int) got.policy, got.decay_period);
        ++failures;
    } catch (const std::exception &) {
    }
}

static void expect_layers_ok(std::string_view text,
                             const std::initializer_list<std::pair<int32_t, evict_spec>> & want) {
    try {
        const auto got = parse_evict_layers(text, "TEST_LAYERS");
        bool ok = got.size() == want.size();
        for (const auto & item : want) {
            const auto it = got.find(item.first);
            ok = ok && it != got.end() &&
                it->second.policy == item.second.policy &&
                it->second.decay_period == item.second.decay_period;
        }
        if (!ok) {
            std::printf("FAIL parse_evict_layers(\"%.*s\") returned an unexpected map\n",
                        (int) text.size(), text.data());
            ++failures;
        }
    } catch (const std::exception & e) {
        std::printf("FAIL parse_evict_layers(\"%.*s\") threw: %s\n",
                    (int) text.size(), text.data(), e.what());
        ++failures;
    }
}

static void expect_layers_error(std::string_view text) {
    try {
        const auto got = parse_evict_layers(text, "TEST_LAYERS");
        std::printf("FAIL parse_evict_layers(\"%.*s\") accepted %zu entries\n",
                    (int) text.size(), text.data(), got.size());
        ++failures;
    } catch (const std::exception &) {
    }
}

static void test_one_shot_log_does_not_wait_for_callback() {
    std::atomic<bool> logged{false};
    std::atomic<int> log_calls{0};
    std::atomic<int> returns{0};
    std::atomic<bool> first_in_callback{false};
    std::atomic<bool> second_returned{false};
    std::atomic<bool> second_returned_while_winner_waiting{false};
    std::mutex mutex;
    std::condition_variable cv;

    // The winner simulates moe_log() invoking a user callback that waits for another
    // runtime's initialization to finish. That other runtime calls log_once below.
    // A regression to std::call_once blocks the second caller until the callback
    // returns; the callback's bounded wait detects that and fails the test.
    std::thread winner([&] {
        llama_moe::detail::invoke_log_once(logged, [&] {
            ++log_calls;
            {
                std::lock_guard<std::mutex> lock(mutex);
                first_in_callback.store(true);
            }
            cv.notify_all();

            std::unique_lock<std::mutex> lock(mutex);
            const bool returned = cv.wait_for(lock, std::chrono::seconds(5),
                [&] { return second_returned.load(); });
            second_returned_while_winner_waiting.store(returned && second_returned.load());
        });
        ++returns;
    });

    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return first_in_callback.load(); });
    }

    std::thread second([&] {
        llama_moe::detail::invoke_log_once(logged, [&] {
            ++log_calls;
        });
        second_returned.store(true);
        ++returns;
        cv.notify_all();
    });

    winner.join();
    second.join();

    if (log_calls.load() != 1 || returns.load() != 2 ||
        !second_returned_while_winner_waiting.load() || !logged.load()) {
        std::printf("FAIL one-shot log claim: log_calls=%d returns=%d "
                    "second_returned_while_winner_waiting=%d logged=%d\n",
                    log_calls.load(), returns.load(),
                    (int) second_returned_while_winner_waiting.load(), (int) logged.load());
        ++failures;
    }
}

int main() {
    using policy = slot_manager::evict_policy;

    test_one_shot_log_does_not_wait_for_callback();

    expect_spec("", policy::lru, 0, true);
    expect_spec("lru", policy::lru, 0);
    expect_spec("lfu", policy::lfu, 0);
    expect_spec("lfu0", policy::lfu, 0);
    expect_spec("lfu1", policy::lfu, 1);
    expect_spec("lfu512", policy::lfu, 512);
    expect_spec("lfu4294967295", policy::lfu, UINT32_MAX);

    expect_spec_error("");
    expect_spec_error("lru0");
    expect_spec_error("lr j");
    expect_spec_error("LFU");
    expect_spec_error("lfugarbage");
    expect_spec_error("lfu512junk");
    expect_spec_error("lfu-1");
    expect_spec_error("lfu+1");
    expect_spec_error("lfu 1");
    expect_spec_error("lfu1.5");
    expect_spec_error("lfu4294967296");
    expect_spec_error("lfu999999999999999999999");

    expect_layers_ok("", {});
    expect_layers_ok("0:lru", {{0, {policy::lru, 0}}});
    expect_layers_ok("0:lfu,39:lfu512", {
        {0, {policy::lfu, 0}},
        {39, {policy::lfu, 512}},
    });
    expect_layers_ok("2147483647:lfu", {{2147483647, {policy::lfu, 0}}});
    expect_layers_ok("0:lru,0:lfu512", {{0, {policy::lfu, 512}}});

    expect_layers_error(",");
    expect_layers_error("0:lfu,");
    expect_layers_error(",0:lfu");
    expect_layers_error("0:lfu,,1:lru");
    expect_layers_error("0");
    expect_layers_error(":lfu");
    expect_layers_error("-1:lfu");
    expect_layers_error("+1:lfu");
    expect_layers_error("bogus:lfu512");
    expect_layers_error("4294967296:lfu");
    expect_layers_error("2147483648:lfu");
    expect_layers_error("0:");
    expect_layers_error("0:lfugarbage");
    expect_layers_error("0:lfu512junk");
    expect_layers_error("0:lfu:extra");

    if (failures) {
        std::printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("all MoE eviction parser cases passed\n");
    return 0;
}
