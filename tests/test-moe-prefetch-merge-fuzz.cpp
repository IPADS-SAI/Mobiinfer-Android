// R4a host fuzz/differential tests: randomized planning, systematic failure
// injection, concurrent batch extraction, and caller-boundary contracts.
//
// Everything here is pure host code: no model, no backend, no HTP.  The
// production logic under test is the header-only planning/execution layer in
// src/llama-moe-prefetch-merge.h; the reference implementation below is
// written independently from the header so a wrong shared shortcut cannot
// make both sides agree.
//
// Build/run (host, C++17):
//   g++ -std=c++17 -O2 -Wall -Wextra -pthread -o /tmp/test-moe-prefetch-merge-fuzz tests/test-moe-prefetch-merge-fuzz.cpp
//   /tmp/test-moe-prefetch-merge-fuzz
//
// Fixed seed: 0x5EED20261122.  The ctest logs and STATS lines are kept in the
// task run directory.

#include "../src/llama-moe-prefetch-merge.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pm = llama_moe::prefetch_merge;

static int failures = 0;
static int reports = 0;
static int g_batch_no = -1;

static bool check_at(bool ok, const char * expr, const char * file, int line) {
    if (!ok) {
        ++failures;
        if (reports++ < 50) {
            std::fprintf(stderr, "FAIL %s:%d: %s (batch %d)\n", file, line, expr, g_batch_no);
        }
    }
    return ok;
}

#define CHECK(c) check_at((c), #c, __FILE__, __LINE__)

static pm::item mk_item(int layer, int expert, int slot, int pos) {
    pm::item it;
    it.layer  = layer;
    it.expert = expert;
    it.slot   = slot;
    it.pos    = pos;
    return it;
}

static std::vector<pm::item> consecutive_items(int n, int layer = 3, int pos0 = 100, int slot0 = 0) {
    std::vector<pm::item> items;
    items.reserve((size_t) n);
    for (int i = 0; i < n; ++i) {
        items.push_back(mk_item(layer, 100 + i, slot0 + i, pos0 + i));
    }
    return items;
}

static int rand_int(std::mt19937 & rng, int n) {
    return n <= 0 ? 0 : (int) (rng() % (uint32_t) n);
}

// ---------------------------------------------------------------------------
// Test-local reference implementation (independent of the header).
// ---------------------------------------------------------------------------

struct ref_stats {
    int runs          = 0;
    int syscalls      = 0;
    int experts       = 0;
    int max_run       = 0;
    int zerocopy_runs = 0;
    int scatter_runs  = 0;
};

struct ref_unit {
    int  first = 0;
    int  count = 0;
    bool merged = false;
    bool slots_consecutive = false;
};

struct ref_plan {
    bool mixed_layers = false;
    bool fallback     = false;
    std::vector<int> order;
    std::vector<ref_unit> units;
    ref_stats projection;
};

static bool ref_same_layer(const std::vector<pm::item> & items) {
    if (items.empty()) return true;
    const int32_t layer = items.front().layer;
    if (layer < 0) return false;
    for (const auto & it : items) {
        if (it.layer != layer) return false;
    }
    return true;
}

static std::vector<int> ref_order_by_pos(const std::vector<pm::item> & items) {
    std::vector<int> order(items.size());
    for (size_t i = 0; i < items.size(); ++i) order[i] = (int) i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (items[(size_t) a].pos != items[(size_t) b].pos) {
            return items[(size_t) a].pos < items[(size_t) b].pos;
        }
        return a < b;
    });
    return order;
}

static bool ref_slots_consecutive(const std::vector<pm::item> & items, const std::vector<int> & order,
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

static ref_stats ref_project(const std::vector<pm::item> & items, int max_experts) {
    ref_stats out;
    if (items.empty()) return out;
    const std::vector<int> order = ref_order_by_pos(items);
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
        if (ref_slots_consecutive(items, order, i, len)) ++out.zerocopy_runs;
        else                                           ++out.scatter_runs;
        i = j;
    }
    return out;
}

static ref_plan ref_plan_batch(const std::vector<pm::item> & items, int mode_value, int max_experts,
                               bool allow_merge) {
    const int mode = mode_value < 0 ? 0 : (mode_value > 2 ? 2 : mode_value);
    ref_plan out;
    const int n = (int) items.size();
    out.order.resize((size_t) n);
    for (int i = 0; i < n; ++i) out.order[(size_t) i] = i;
    if (n == 0) return out;

    out.mixed_layers = !ref_same_layer(items);
    if (out.mixed_layers) {
        out.fallback = true;
        out.units.resize((size_t) n);
        for (int i = 0; i < n; ++i) out.units[(size_t) i] = ref_unit{i, 1, false, true};
        return out;
    }

    if (mode >= 1) out.projection = ref_project(items, max_experts);

    if (mode == 0) {
        out.units.resize((size_t) n);
        for (int i = 0; i < n; ++i) out.units[(size_t) i] = ref_unit{i, 1, false, true};
        return out;
    }

    if (mode == 1 || !allow_merge) {
        out.fallback = true;
        out.units.resize((size_t) n);
        for (int i = 0; i < n; ++i) out.units[(size_t) i] = ref_unit{i, 1, false, true};
        return out;
    }

    out.order = ref_order_by_pos(items);
    const int chunk = std::max(1, max_experts);
    int i = 0;
    while (i < n) {
        int j = i + 1;
        const int32_t p0 = items[(size_t) out.order[(size_t) i]].pos;
        while (j < n && p0 >= 0 &&
               items[(size_t) out.order[(size_t) j]].pos == p0 + (j - i)) {
            ++j;
        }
        for (int a = i; a < j; a += chunk) {
            ref_unit u;
            u.first  = a;
            u.count  = std::min(chunk, j - a);
            u.merged = u.count > 1;
            u.slots_consecutive = ref_slots_consecutive(items, out.order, a, u.count);
            out.units.push_back(u);
        }
        i = j;
    }
    return out;
}

static void check_stats(const pm::stats & got, const ref_stats & want) {
    CHECK(got.runs == want.runs);
    CHECK(got.syscalls == want.syscalls);
    CHECK(got.experts == want.experts);
    CHECK(got.max_run == want.max_run);
    CHECK(got.zerocopy_runs == want.zerocopy_runs);
    CHECK(got.scatter_runs == want.scatter_runs);
}

static bool is_permutation(const std::vector<int> & order, int n) {
    if ((int) order.size() != n) return false;
    std::vector<int> seen((size_t) n, 0);
    for (int idx : order) {
        if (idx < 0 || idx >= n) return false;
        if (++seen[(size_t) idx] != 1) return false;
    }
    for (int s : seen) {
        if (s != 1) return false;
    }
    return true;
}

static void check_plan(const std::vector<pm::item> & items, int mode, int max_experts, bool allow_merge) {
    const int n = (int) items.size();
    const pm::plan got = pm::plan_batch(items, mode, max_experts, allow_merge);
    const ref_plan want = ref_plan_batch(items, mode, max_experts, allow_merge);

    CHECK(got.mixed_layers == want.mixed_layers);
    CHECK(got.fallback == want.fallback);
    CHECK((int) got.order.size() == n);
    for (int i = 0; i < n && i < (int) got.order.size(); ++i) {
        CHECK(got.order[(size_t) i] == want.order[(size_t) i]);
    }
    check_stats(got.projection, want.projection);

    CHECK(got.units.size() == want.units.size());
    int cursor = 0;
    for (size_t i = 0; i < got.units.size() && i < want.units.size(); ++i) {
        const pm::unit & u = got.units[i];
        const ref_unit & r = want.units[i];
        CHECK(u.first == r.first);
        CHECK(u.first == cursor);          // units cover order contiguously
        CHECK(u.count == r.count);
        CHECK(u.merged == r.merged);
        CHECK(u.slots_consecutive == r.slots_consecutive);
        CHECK(u.count >= 1);
        if (u.merged) CHECK(u.count > 1);
        cursor += u.count;
    }
    CHECK(cursor == n);

    if (mode == 0) {
        if (!got.mixed_layers) CHECK(!got.fallback);
        CHECK(is_permutation(got.order, n));
        for (int i = 0; i < n; ++i) CHECK(got.order[(size_t) i] == i);
        for (const auto & u : got.units) CHECK(!u.merged);
        check_stats(got.projection, ref_stats{});
    }
    if (mode == 1) {
        CHECK(is_permutation(got.order, n));
        for (int i = 0; i < n; ++i) CHECK(got.order[(size_t) i] == i);
        for (const auto & u : got.units) CHECK(!u.merged);
    }
    if (mode == 2 && !got.mixed_layers) {
        if (!got.fallback) {
            CHECK(is_permutation(got.order, n));
            for (const auto & u : got.units) {
                CHECK(u.count <= std::max(1, max_experts));
                if (!u.merged) continue;
                const int32_t layer = items[(size_t) got.order[(size_t) u.first]].layer;
                int32_t prev = -1;
                for (int j = 0; j < u.count; ++j) {
                    const pm::item & it = items[(size_t) got.order[(size_t) (u.first + j)]];
                    CHECK(it.layer == layer);
                    CHECK(it.pos >= 0);                       // pos=-1 never merged
                    if (j == 0) prev = it.pos;
                    else {
                        CHECK(it.pos == prev + 1);            // physical continuity
                        prev = it.pos;
                    }
                }
            }
        } else {
            CHECK(is_permutation(got.order, n));
            for (int i = 0; i < n; ++i) CHECK(got.order[(size_t) i] == i);
            for (const auto & u : got.units) CHECK(!u.merged);
        }
    }
    if (got.mixed_layers) {
        CHECK(got.fallback);
        CHECK(is_permutation(got.order, n));
        for (int i = 0; i < n; ++i) CHECK(got.order[(size_t) i] == i);
        for (const auto & u : got.units) CHECK(!u.merged);
    }
}

// ---------------------------------------------------------------------------
// Random batch generation.
// ---------------------------------------------------------------------------

static std::vector<pm::item> random_batch(std::mt19937 & rng, bool allow_mixed) {
    const int n = rand_int(rng, 65);                  // 0..64
    const int base_layer = rand_int(rng, 4);          // 0..3
    const bool mixed = allow_mixed && n > 1 && rand_int(rng, 5) == 0;
    const int style = rand_int(rng, 4);
    int32_t cur = (int32_t) rand_int(rng, 200);
    int32_t slot_cur = (int32_t) rand_int(rng, 400);
    int32_t expert = (int32_t) rand_int(rng, 32);

    std::vector<pm::item> items;
    items.reserve((size_t) n);
    for (int i = 0; i < n; ++i) {
        pm::item it;
        it.layer = base_layer;
        if (mixed && (i == n / 2 || rand_int(rng, 7) == 0)) {
            it.layer = (base_layer + 1 + rand_int(rng, 3)) % 4;
        }

        if (style == 0) {
            it.pos = rand_int(rng, 257) - 1;          // -1..255, no locality
        } else if (style == 1) {
            const int roll = rand_int(rng, 10);
            if (roll < 7) {
                it.pos = cur;
                cur = (cur + 1) % 200;
            } else if (roll < 9) {
                cur = (cur + 2 + rand_int(rng, 4)) % 200;   // gap
                it.pos = cur;
                cur = (cur + 1) % 200;
            } else if (roll == 9) {
                it.pos = -1;
            } else {
                it.pos = cur > 0 ? cur - 1 : 0;       // duplicate position
            }
        } else if (style == 2) {
            const int roll = rand_int(rng, 12);
            if (roll == 0) {
                it.pos = -1;
            } else if (roll == 1) {
                it.pos = cur > 0 ? cur - 1 : 0;       // duplicate
            } else {
                it.pos = cur;
                cur = (cur + 1) % 200;
            }
        } else {
            it.pos = rand_int(rng, 3) == 0 ? -1 : rand_int(rng, 8);  // duplicates
        }

        if (rand_int(rng, 4) == 0 && !items.empty()) {
            it.expert = items.back().expert;          // duplicate expert
        } else {
            expert = (expert + 1 + rand_int(rng, 3)) % 256;
            it.expert = expert;
        }

        if (rand_int(rng, 2) == 0) {
            it.slot = slot_cur;
            slot_cur = (slot_cur + 1) % 500;
        } else {
            it.slot = rand_int(rng, 513) - 1;         // -1..511
        }
        if (rand_int(rng, 20) == 0) it.slot = -1;

        items.push_back(it);
    }
    return items;
}

// ---------------------------------------------------------------------------
// 1. Randomized differential planning.
// ---------------------------------------------------------------------------

static int merged_unit_count(const pm::plan & p);

struct coverage_counters {
    long batches       = 0;
    long mixed_layers  = 0;
    long pos_negative  = 0;
    long duplicate_pos = 0;
    long merge_batches = 0;
    long merged_units  = 0;
    long split_batches = 0;    // projection run was longer than max_experts
};

static coverage_counters g_cov;

static void test_random_differential(unsigned seed) {
    const int kBatches = 1200;
    std::mt19937 rng(seed);
    for (int b = 0; b < kBatches; ++b) {
        g_batch_no = b;
        const std::vector<pm::item> items = random_batch(rng, true);
        const int max_experts = 1 + rand_int(rng, 8);

        ++g_cov.batches;
        const ref_plan ref = ref_plan_batch(items, 2, max_experts, true);
        if (ref.mixed_layers) ++g_cov.mixed_layers;
        std::vector<int> positions;
        positions.reserve(items.size());
        for (const auto & it : items) {
            positions.push_back(it.pos);
            if (it.pos < 0) ++g_cov.pos_negative;
        }
        std::sort(positions.begin(), positions.end());
        for (size_t i = 1; i < positions.size(); ++i) {
            if (positions[i] == positions[i - 1]) { ++g_cov.duplicate_pos; break; }
        }
        const int n_merged = merged_unit_count(pm::plan_batch(items, 2, max_experts, true));
        g_cov.merged_units += n_merged;
        if (n_merged > 0) ++g_cov.merge_batches;
        if (!ref.mixed_layers && ref.projection.runs > 0 &&
            ref.units.size() > (size_t) ref.projection.runs) {
            ++g_cov.split_batches;
        }

        check_plan(items, 0, max_experts, true);
        check_plan(items, 0, max_experts, false);
        check_plan(items, 1, max_experts, true);
        check_plan(items, 1, max_experts, false);
        check_plan(items, 2, max_experts, true);
        check_plan(items, 2, max_experts, false);
    }
    g_batch_no = -1;
    std::printf("ok   random differential: %d batches x 6 plan variants\n", kBatches);
}

// ---------------------------------------------------------------------------
// 2. Systematic failure injection on every merged unit.
// ---------------------------------------------------------------------------

struct injection_counters {
    long batches      = 0;
    long merged_units = 0;
    long cases_false  = 0;
    long cases_throw  = 0;
    long cases_partial = 0;
    long read_one_calls = 0;
    long experts      = 0;
};

static injection_counters g_inj;

static int merged_unit_count(const pm::plan & p) {
    int n = 0;
    for (const auto & u : p.units) {
        if (u.merged) ++n;
    }
    return n;
}

// kind: 0 = read_run returns false, 1 = read_run throws, 2 = read_run writes
// only the first k experts and then returns false.
static void check_injection_case(const std::vector<pm::item> & items, const pm::plan & p,
                                 int target, int kind) {
    const int n = (int) items.size();
    const int n_merged = merged_unit_count(p);
    const int n_single = (int) p.units.size() - n_merged;

    std::vector<int> consumed((size_t) n, 0);
    std::vector<int> writes((size_t) n, 0);
    std::vector<char> finished(256, 0);
    std::vector<int> one_seq;

    int run_calls = 0;
    int one_calls = 0;
    int merged_calls = 0;
    const int partial_k = std::max(1, p.units[(size_t) target].count / 2);

    auto mark_unit = [&](const pm::unit & u) {
        for (int j = 0; j < u.count; ++j) {
            const int idx = p.order[(size_t) (u.first + j)];
            consumed[(size_t) idx]++;
            writes[(size_t) idx]++;
            const int e = items[(size_t) idx].expert;
            if (e >= 0 && e < (int) finished.size()) finished[(size_t) e] = 1;
        }
    };

    const pm::exec_stats s = pm::execute_plan(p,
        [&](const std::vector<int> & order, const pm::unit & u) -> bool {
            ++run_calls;
            if (&u != &p.units[(size_t) target]) return true;   // on_merged publishes it
            if (kind == 1) throw std::runtime_error("injected read_run failure");
            if (kind == 2) {
                for (int j = 0; j < partial_k && j < u.count; ++j) {
                    writes[(size_t) order[(size_t) (u.first + j)]]++;
                }
            }
            return false;
        },
        [&](int idx) {
            ++one_calls;
            one_seq.push_back(idx);
            consumed[(size_t) idx]++;
            writes[(size_t) idx]++;
            const int e = items[(size_t) idx].expert;
            if (e >= 0 && e < (int) finished.size()) finished[(size_t) e] = 1;
        },
        [&](const std::vector<int> &, const pm::unit & u) {
            ++merged_calls;
            mark_unit(u);
        });

    // Every merged unit is attempted exactly once; single units never call read_run.
    CHECK(run_calls == n_merged);
    CHECK(merged_calls == n_merged - 1);
    CHECK(s.merged_runs == n_merged - 1);
    CHECK(s.fallback_runs == 1);
    CHECK(s.experts == n);
    CHECK(s.syscalls == (n_merged - 1) + p.units[(size_t) target].count + n_single);

    // Expected read_one sequence: every single unit, plus the *whole* failed run.
    std::vector<int> expected;
    for (size_t i = 0; i < p.units.size(); ++i) {
        const pm::unit & u = p.units[i];
        if ((int) i != target && u.merged) continue;
        for (int j = 0; j < u.count; ++j) expected.push_back(p.order[(size_t) (u.first + j)]);
    }
    CHECK(one_calls == (int) expected.size());
    CHECK(one_seq == expected);

    for (int i = 0; i < n; ++i) {
        CHECK(consumed[(size_t) i] == 1);       // exactly one completion per item
        CHECK(writes[(size_t) i] >= 1);         // data reached the slot at least once
    }
    for (const auto & it : items) {
        if (it.expert >= 0 && it.expert < (int) finished.size()) {
            CHECK(finished[(size_t) it.expert] == 1);   // no expert left loading
        }
    }

    g_inj.read_one_calls += one_calls;
    g_inj.experts += n;
}

static void test_failure_injection(unsigned seed) {
    const int kBatches = 240;
    std::mt19937 rng(seed);
    for (int b = 0; b < kBatches; ++b) {
        g_batch_no = b;
        const std::vector<pm::item> items = random_batch(rng, false);   // single layer
        const int max_experts = 1 + rand_int(rng, 8);
        const pm::plan p = pm::plan_batch(items, 2, max_experts, true);
        if (p.fallback || p.mixed_layers) continue;
        const int n_merged = merged_unit_count(p);
        if (n_merged == 0) continue;

        ++g_inj.batches;
        g_inj.merged_units += n_merged;
        for (size_t t = 0; t < p.units.size(); ++t) {
            if (!p.units[t].merged) continue;
            check_injection_case(items, p, (int) t, 0); ++g_inj.cases_false;
            check_injection_case(items, p, (int) t, 1); ++g_inj.cases_throw;
            check_injection_case(items, p, (int) t, 2); ++g_inj.cases_partial;
        }
    }

    // mode 0/1: read_run must never be called, and every item is read once.
    g_batch_no = -1;
    for (int b = 0; b < 200; ++b) {
        g_batch_no = 10000 + b;
        const std::vector<pm::item> items = random_batch(rng, true);
        const int n = (int) items.size();
        for (int mode = 0; mode <= 1; ++mode) {
            const pm::plan p = pm::plan_batch(items, mode, 4, true);
            std::vector<int> consumed((size_t) n, 0);
            int run_calls = 0;
            int one_calls = 0;
            const pm::exec_stats s = pm::execute_plan(p,
                [&](const std::vector<int> &, const pm::unit &) { ++run_calls; return true; },
                [&](int idx) { ++one_calls; consumed[(size_t) idx]++; },
                [&](const std::vector<int> &, const pm::unit &) {});
            CHECK(run_calls == 0);
            CHECK(one_calls == n);
            CHECK(s.experts == n);
            CHECK(s.merged_runs == 0);
            for (int i = 0; i < n; ++i) CHECK(consumed[(size_t) i] == 1);
            CHECK(p.order.size() == (size_t) n);
            for (int i = 0; i < n; ++i) CHECK(p.order[(size_t) i] == i);
        }
    }
    g_batch_no = -1;

    CHECK(g_inj.batches > 0);
    CHECK(g_inj.cases_false > 0 && g_inj.cases_throw > 0 && g_inj.cases_partial > 0);
    std::printf("ok   failure injection: batches=%ld merged_units=%ld cases(false/throw/partial)=%ld/%ld/%ld\n",
                g_inj.batches, g_inj.merged_units, g_inj.cases_false, g_inj.cases_throw, g_inj.cases_partial);
}

// ---------------------------------------------------------------------------
// 3. Concurrent batch extraction (mutex + condition_variable, 8 workers).
// ---------------------------------------------------------------------------

struct work_item {
    int id    = 0;
    int layer = 0;
    int epoch = 0;
    int expert = 0;
    int pos   = 0;
    int slot  = 0;
};

class batch_queue {
public:
    explicit batch_queue(int groups) : groups_((size_t) groups), next_(0), size_(0), closed_(false) {}

    void push(const work_item & it) {
        std::lock_guard<std::mutex> lk(m_);
        const size_t g = (size_t) (it.layer * 2 + it.epoch);
        groups_[g].push_back(it);
        ++size_;
        cv_.notify_one();
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_.notify_all();
    }

    // Take up to 8 items from one (layer, epoch) group.  A batch never mixes
    // groups because a worker only pops from a single sub-queue.
    bool pop(std::vector<work_item> & out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return closed_ || size_ > 0; });
        if (size_ == 0) return false;
        for (size_t k = 0; k < groups_.size(); ++k) {
            const size_t g = (next_ + k) % groups_.size();
            if (groups_[g].empty()) continue;
            next_ = (g + 1) % groups_.size();
            const size_t take = std::min<size_t>(8, groups_[g].size());
            out.clear();
            out.reserve(take);
            for (size_t i = 0; i < take; ++i) {
                out.push_back(groups_[g].front());
                groups_[g].pop_front();
            }
            size_ -= take;
            return true;
        }
        return false;
    }

    bool empty() {
        std::lock_guard<std::mutex> lk(m_);
        return size_ == 0;
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<std::deque<work_item>> groups_;
    size_t next_;
    size_t size_;
    bool closed_;
};

static void test_concurrency() {
    const int kLayers = 2;
    const int kEpochs = 2;
    const int kPerGroup = 500;
    const int kGroups = kLayers * kEpochs;
    const int kWorkers = 8;
    const int kItems = kLayers * kEpochs * kPerGroup;

    batch_queue q(kGroups);
    std::vector<std::atomic<int>> consumed((size_t) kItems);
    for (auto & c : consumed) c.store(0);

    std::atomic<int> errors{0};
    std::atomic<long> batches{0};
    std::atomic<long> items_done{0};
    std::atomic<long> read_one{0};
    std::atomic<long> read_run{0};
    std::atomic<long> merged_units{0};

    auto worker = [&]() {
        std::vector<work_item> batch;
        while (q.pop(batch)) {
            if (batch.empty()) {
                ++errors;
                continue;
            }
            const int layer = batch.front().layer;
            const int epoch = batch.front().epoch;
            for (const auto & w : batch) {
                if (w.layer != layer || w.epoch != epoch) ++errors;
            }

            std::vector<pm::item> items;
            items.reserve(batch.size());
            for (const auto & w : batch) {
                items.push_back(mk_item(w.layer, w.expert, w.slot, w.pos));
            }

            const pm::plan p = pm::plan_batch(items, 2, 8, true);
            if (p.mixed_layers || p.fallback) ++errors;

            int cursor = 0;
            for (const auto & u : p.units) {
                if (u.first != cursor || u.count < 1 || u.count > 8) ++errors;
                cursor += u.count;
                if (!u.merged) continue;
                ++merged_units;
                int32_t prev = -1;
                for (int j = 0; j < u.count; ++j) {
                    const pm::item & it = items[(size_t) p.order[(size_t) (u.first + j)]];
                    if (it.layer != layer || it.pos < 0) ++errors;
                    if (j == 0) prev = it.pos;
                    else {
                        if (it.pos != prev + 1) ++errors;
                        prev = it.pos;
                    }
                }
            }
            if (cursor != (int) items.size()) ++errors;

            const pm::exec_stats s = pm::execute_plan(p,
                [&](const std::vector<int> & order, const pm::unit & u) {
                    ++read_run;
                    for (int j = 0; j < u.count; ++j) {
                        const int idx = order[(size_t) (u.first + j)];
                        consumed[(size_t) batch[(size_t) idx].id].fetch_add(1);
                    }
                    return true;
                },
                [&](int idx) {
                    ++read_one;
                    consumed[(size_t) batch[(size_t) idx].id].fetch_add(1);
                },
                [&](const std::vector<int> &, const pm::unit &) {});
            if (s.experts != (int) items.size()) ++errors;

            ++batches;
            items_done += (long) items.size();
            batch.clear();
        }
    };

    // Pre-fill one full batch per group so the merge path is exercised even if
    // the producer thread happens to be slow, then let workers and producer
    // run concurrently.
    int next_id = 0;
    for (int i = 0; i < 8; ++i) {
        for (int layer = 0; layer < kLayers; ++layer) {
            for (int epoch = 0; epoch < kEpochs; ++epoch) {
                work_item w;
                w.id = next_id++;
                w.layer = layer;
                w.epoch = epoch;
                w.expert = i;
                w.pos = i;
                w.slot = i % 512;
                q.push(w);
            }
        }
    }

    std::vector<std::thread> workers;
    workers.reserve((size_t) kWorkers);
    for (int i = 0; i < kWorkers; ++i) workers.emplace_back(worker);

    std::thread producer([&]() {
        for (int i = 8; i < kPerGroup; ++i) {
            for (int layer = 0; layer < kLayers; ++layer) {
                for (int epoch = 0; epoch < kEpochs; ++epoch) {
                    work_item w;
                    w.id = next_id++;
                    w.layer = layer;
                    w.epoch = epoch;
                    w.expert = i;
                    w.pos = i;
                    w.slot = i % 512;
                    q.push(w);
                }
            }
        }
        q.close();
    });

    producer.join();
    for (auto & t : workers) t.join();

    CHECK(errors.load() == 0);
    CHECK(q.empty());
    CHECK(items_done.load() == kItems);
    CHECK(batches.load() >= 4);
    CHECK(merged_units.load() >= 4);
    CHECK(read_one.load() + read_run.load() > 0);
    for (int i = 0; i < kItems; ++i) CHECK(consumed[(size_t) i].load() == 1);

    std::printf("ok   concurrency: workers=%d items=%d batches=%ld merged_units=%ld read_one=%ld read_run=%ld\n",
                kWorkers, kItems, batches.load(), merged_units.load(), read_one.load(), read_run.load());
}

// ---------------------------------------------------------------------------
// 4. Caller-boundary contracts.
// ---------------------------------------------------------------------------

// Batch-level conditions in src/llama-moe-slot.cpp: the caller must pass
// allow_merge=false (or skip the batch path) when any of these is active.
struct caller_gate {
    bool io_budget = false;
    int  read_mode = 0;
    bool raw_upload_hook = false;
    bool family_stride_matches = true;

    bool permits_batch_merge() const {
        return !io_budget && read_mode == 0 && !raw_upload_hook && family_stride_matches;
    }
};

// Per-run conditions inside read_prefetch_run_directv: returning false makes
// execute_plan re-read the whole run one expert at a time.
static bool permits_run(int count, int n_fam, bool o_direct, size_t offset, size_t total,
                        bool upload_hook, bool stride_mismatch) {
    if (upload_hook) return false;
    if (stride_mismatch) return false;
    if ((size_t) count * (size_t) n_fam > 1024) return false;          // IOV_MAX
    if (o_direct && ((offset % 512) != 0 || (total % 512) != 0)) return false;
    return true;
}

static void check_batch_gate_case(const char * name, caller_gate gate, bool expect_merge) {
    (void) name;
    const std::vector<pm::item> items = consecutive_items(8);
    const bool allow = gate.permits_batch_merge();
    CHECK(allow == expect_merge);
    const pm::plan p = pm::plan_batch(items, 2, 4, allow);
    if (expect_merge) {
        CHECK(merged_unit_count(p) == 2);
        CHECK(!p.fallback);
    } else {
        CHECK(p.fallback);
        CHECK(merged_unit_count(p) == 0);
        for (const auto & u : p.units) CHECK(!u.merged);
        for (int i = 0; i < 8; ++i) CHECK(p.order[(size_t) i] == i);
    }

    int run_calls = 0;
    int one_calls = 0;
    pm::execute_plan(p,
        [&](const std::vector<int> &, const pm::unit &) { ++run_calls; return true; },
        [&](int) { ++one_calls; },
        [&](const std::vector<int> &, const pm::unit &) {});
    if (expect_merge) {
        CHECK(run_calls == 2 && one_calls == 0);
    } else {
        CHECK(run_calls == 0 && one_calls == 8);
    }
}

static void check_run_gate_case(const char * name, int n, int max_experts, bool expect_success,
                                int n_fam = 1, bool o_direct = true, size_t offset = 1024,
                                size_t total = 2048, bool upload_hook = false,
                                bool stride_mismatch = false) {
    (void) name;
    const std::vector<pm::item> items = consecutive_items(n);
    const pm::plan p = pm::plan_batch(items, 2, max_experts, true);
    const int n_merged = merged_unit_count(p);
    CHECK(n_merged > 0);
    for (const auto & u : p.units) CHECK(u.count <= std::max(1, max_experts));

    int run_calls = 0;
    int one_calls = 0;
    const pm::exec_stats s = pm::execute_plan(p,
        [&](const std::vector<int> &, const pm::unit & u) {
            ++run_calls;
            return permits_run(u.count, n_fam, o_direct, offset, total, upload_hook, stride_mismatch);
        },
        [&](int) { ++one_calls; },
        [&](const std::vector<int> &, const pm::unit &) {});

    CHECK(run_calls == n_merged);
    if (expect_success) {
        CHECK(one_calls == 0);
        CHECK(s.fallback_runs == 0);
        CHECK(s.merged_runs == n_merged);
    } else {
        CHECK(one_calls == n);                       // whole-run fallback, every expert
        CHECK(s.fallback_runs == n_merged);
        CHECK(s.merged_runs == 0);
        CHECK(s.experts == n);
    }
}

static void test_caller_boundaries() {
    caller_gate ok;
    check_batch_gate_case("baseline", ok, true);

    caller_gate budget = ok;
    budget.io_budget = true;
    check_batch_gate_case("io_budget", budget, false);
    // The header trusts the caller's allow_merge flag: with io_budget on, the
    // caller must not pass true.  This asserts both sides of that contract.
    {
        const std::vector<pm::item> items = consecutive_items(8);
        const pm::plan would = pm::plan_batch(items, 2, 4, true);
        CHECK(merged_unit_count(would) == 2);
        const pm::plan must_not = pm::plan_batch(items, 2, 4, budget.permits_batch_merge());
        CHECK(merged_unit_count(must_not) == 0);
        CHECK(must_not.fallback);
    }

    caller_gate mode1 = ok;
    mode1.read_mode = 1;
    check_batch_gate_case("non-zero read_mode", mode1, false);

    caller_gate staging = ok;
    staging.raw_upload_hook = true;
    check_batch_gate_case("raw staging/upload hook", staging, false);

    caller_gate stride = ok;
    stride.family_stride_matches = false;
    check_batch_gate_case("family stride mismatch", stride, false);

    // run > max: split into chunks, each <= max, all still mergeable.
    check_run_gate_case("run > max", 10, 4, true);
    {
        const std::vector<pm::item> items = consecutive_items(10);
        const pm::plan p = pm::plan_batch(items, 2, 4, true);
        CHECK(p.units.size() == 3);
        CHECK(p.units[0].count == 4 && p.units[1].count == 4 && p.units[2].count == 2);
    }

    // Aligned O_DIRECT run succeeds.
    check_run_gate_case("O_DIRECT aligned", 8, 4, true, 1, true, 1024, 2048);
    // Offset / total not 512-aligned: read_run returns false -> whole-run fallback.
    check_run_gate_case("O_DIRECT unaligned offset", 8, 4, false, 1, true, 1000, 2048);
    check_run_gate_case("O_DIRECT unaligned total", 8, 4, false, 1, true, 1024, 2049);
    // Alignment only matters when O_DIRECT is on.
    check_run_gate_case("buffered unaligned", 8, 4, true, 1, false, 1000, 2049);
    // IOV_MAX: 4 experts x 300 families = 1200 > 1024.
    check_run_gate_case("IOV_MAX", 8, 4, false, 300, true, 1024, 2048);
    // 4 x 256 = 1024 is exactly the limit and must still pass.
    check_run_gate_case("IOV_MAX boundary", 8, 4, true, 256, true, 1024, 2048);
    // Raw staging upload hook / family stride mismatch fall back per run.
    check_run_gate_case("upload hook", 8, 4, false, 1, true, 1024, 2048, true, false);
    check_run_gate_case("family stride mismatch", 8, 4, false, 1, true, 1024, 2048, false, true);

    // Slot discontinuity is *not* a fallback condition (iovec carries each slot).
    {
        std::vector<pm::item> items = consecutive_items(4);
        items[1].slot = 77;
        items[2].slot = 5;
        const pm::plan p = pm::plan_batch(items, 2, 4, true);
        CHECK(merged_unit_count(p) == 1);
        CHECK(!p.fallback);
        CHECK(!p.units[0].slots_consecutive);
        CHECK(p.projection.scatter_runs == 1);
        int run_calls = 0;
        int one_calls = 0;
        pm::execute_plan(p,
            [&](const std::vector<int> &, const pm::unit &) { ++run_calls; return true; },
            [&](int) { ++one_calls; },
            [&](const std::vector<int> &, const pm::unit &) {});
        CHECK(run_calls == 1 && one_calls == 0);
    }

    std::printf("ok   caller boundaries (io_budget/read_mode/staging/stride/O_DIRECT/IOV_MAX/slot)\n");
}

// ---------------------------------------------------------------------------
// env_mode: serial only, so no worker thread observes a changing environment.
// ---------------------------------------------------------------------------

static void test_env_mode_serial() {
    unsetenv("LLAMA_MOE_MERGE_PREFETCH");
    CHECK(pm::env_mode() == 0);                    // default remains 0

    setenv("LLAMA_MOE_MERGE_PREFETCH", "2", 1);
    CHECK(pm::env_mode() == 2);
    setenv("LLAMA_MOE_MERGE_PREFETCH", "1", 1);
    CHECK(pm::env_mode() == 1);
    setenv("LLAMA_MOE_MERGE_PREFETCH", "-7", 1);
    CHECK(pm::env_mode() == 0);
    setenv("LLAMA_MOE_MERGE_PREFETCH", "99", 1);
    CHECK(pm::env_mode() == 2);
    setenv("LLAMA_MOE_MERGE_PREFETCH", "junk", 1);
    CHECK(pm::env_mode() == 0);

    unsetenv("LLAMA_MOE_MERGE_PREFETCH");
    CHECK(pm::env_mode() == 0);
    std::printf("ok   env_mode 0/1/2 + clamps (serial)\n");
}

int main() {
    const unsigned seed = 0x5EED20261122u & 0x7fffffffu;
    std::printf("seed=%u\n", seed);

    test_env_mode_serial();

    g_batch_no = -1;
    test_random_differential(seed);
    test_failure_injection(seed ^ 0x9E3779B9u);
    test_concurrency();
    test_caller_boundaries();

    std::printf("STATS seed=%u random_batches=1200 plan_variants=6\n", seed);
    std::printf("STATS coverage batches=%ld mixed_layers=%ld pos_negative=%ld duplicate_pos=%ld merge_batches=%ld merged_units=%ld split_batches=%ld\n",
                g_cov.batches, g_cov.mixed_layers, g_cov.pos_negative, g_cov.duplicate_pos,
                g_cov.merge_batches, g_cov.merged_units, g_cov.split_batches);
    std::printf("STATS injection batches=%ld merged_units=%ld cases_false=%ld cases_throw=%ld cases_partial=%ld read_one_calls=%ld experts=%ld\n",
                g_inj.batches, g_inj.merged_units, g_inj.cases_false, g_inj.cases_throw,
                g_inj.cases_partial, g_inj.read_one_calls, g_inj.experts);

    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("all prefetch-merge fuzz tests passed\n");
    return 0;
}
