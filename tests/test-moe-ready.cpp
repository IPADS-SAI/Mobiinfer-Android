#include "../src/llama-moe-slot-manager.h"
#include "../src/llama-moe-prefetch-schedule.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>

static std::atomic<bool> count_allocations{false};
static std::atomic<size_t> allocations{0};
void * operator new(size_t n) {
    if (count_allocations.load(std::memory_order_relaxed)) allocations.fetch_add(1, std::memory_order_relaxed);
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
[[gnu::noinline]] void operator delete(void * p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void * p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void * p, size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void * p, size_t) noexcept { std::free(p); }

using manager = llama_moe::slot_manager;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #c); std::abort(); } } while (false)
struct fixture {
    std::mutex mutex;
    manager table;
    explicit fixture(int slots = 2, int cache = 2, bool unified = true, bool probation = false) :
        table(mutex, 32, slots, cache, unified, -1, probation) {}
    manager::ticket resident(int e) {
        const auto r = table.acquire(e, {1, 0}); CHECK(r.owner); CHECK(table.complete(r.request)); return r.request;
    }
};
struct gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool open = false;
    void signal() { std::lock_guard<std::mutex> lock(mutex); open = true; cv.notify_all(); }
    void wait() { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return open; }); }
};

template<class F> static void throws(F fn) { bool caught = false; try { fn(); } catch (const std::exception &) { caught = true; } CHECK(caught); }

static void horizon_coalescing() {
    fixture f;
    std::lock_guard<std::mutex> lock(f.mutex);
    const auto a = f.table.enqueue(4, {1, 10});
    CHECK(a.action == manager::outcome::created && a.request.slot == -1 && !a.owner);
    const auto b = f.table.enqueue(4, {2, 20});
    CHECK(b.action == manager::outcome::coalesced && a.request.generation == b.request.generation);
    CHECK(f.table.lookup(4).producer.epoch == 1 && f.table.lookup(4).producer.job == 10);
    CHECK(f.table.writers() == 0 && f.table.slot_owner(0).state == manager::phase::absent);
    const auto loading = f.table.start_queued(a.request); CHECK(loading.owner);
    CHECK(f.table.enqueue(4, {3, 30}).request.generation == a.request.generation);
    CHECK(f.table.complete(loading.request));
    CHECK(f.table.enqueue(4, {4, 40}).request.generation == a.request.generation);
    CHECK(f.table.acquire(4, {5, 0}).action == manager::outcome::reused);
}

static void queued_demand_promotion() {
    fixture f(1, 1);
    std::lock_guard<std::mutex> lock(f.mutex);
    const auto queued = f.table.enqueue(1, {1, 1});
    const int32_t ids[] = {1, 1}; f.table.pin_selected(ids, 2);
    const auto own = f.table.acquire(1, {1, 0});
    CHECK(own.owner && own.action == manager::outcome::promoted);
    CHECK(own.previous.generation == queued.request.generation && own.request.generation != queued.request.generation);
    CHECK(!f.table.start_queued(queued.request).owner);
    CHECK(!f.table.cancel_queued(queued.request));
    CHECK(f.table.complete(own.request)); CHECK(f.table.pins() == 1);
    CHECK(!f.table.acquire(2, {1, 0}).request);
    f.table.release_selected(); CHECK(f.table.acquire(2, {1, 0}).owner);
}

static void queued_claim_race() {
    for (int i = 0; i < 64; ++i) {
        fixture f(1, 1); manager::result pf, demand; manager::ticket queued;
        { std::lock_guard<std::mutex> lock(f.mutex); queued = f.table.enqueue(1, {1, 1}).request; }
        gate start;
        std::thread worker([&] { start.wait(); std::lock_guard<std::mutex> lock(f.mutex); pf = f.table.start_queued(queued); });
        std::thread main([&] { start.wait(); std::lock_guard<std::mutex> lock(f.mutex); const int32_t ids[] = {1}; f.table.pin_selected(ids, 1); demand = f.table.acquire(1, {2, 0}); });
        start.signal(); worker.join(); main.join();
        std::lock_guard<std::mutex> lock(f.mutex);
        CHECK((int) pf.owner + (int) demand.owner == 1);
        if (pf.owner) CHECK(demand.action == manager::outcome::joined && demand.request.generation == pf.request.generation);
        else CHECK(demand.action == manager::outcome::promoted);
        CHECK(f.table.complete(pf.owner ? pf.request : demand.request));
        CHECK(f.table.writers() == 0); f.table.release_selected();
    }
}

static void family_publication_and_wait() {
    fixture f(1, 1); manager::ticket request;
    { std::lock_guard<std::mutex> lock(f.mutex); auto q = f.table.enqueue(1, {1, 1}); request = f.table.start_queued(q.request).request; }
    std::array<int, 3> families{}; gate two_written, allow_last, waiter_entered;
    std::thread writer([&] {
        families[0] = 11; families[1] = 22; two_written.signal(); allow_last.wait();
        families[2] = 33; std::lock_guard<std::mutex> lock(f.mutex); CHECK(f.table.complete(request));
    });
    two_written.wait();
    { std::lock_guard<std::mutex> lock(f.mutex); CHECK(f.table.resident_slot(1) < 0); CHECK(f.table.acquire(1, {2, 0}).action == manager::outcome::joined); }
    std::thread waiter([&] {
        std::unique_lock<std::mutex> lock(f.mutex); waiter_entered.signal(); f.table.wait(request, lock);
        CHECK(families[0] == 11 && families[1] == 22 && families[2] == 33);
    });
    waiter_entered.wait(); allow_last.signal(); writer.join(); waiter.join();
}

static void own_misses_before_join() {
    fixture f(2, 2); manager::ticket pf, own;
    { std::lock_guard<std::mutex> lock(f.mutex); auto q = f.table.enqueue(1, {1, 1}); pf = f.table.start_queued(q.request).request;
      const int32_t ids[] = {1, 2, 1}; f.table.pin_selected(ids, 3);
      CHECK(f.table.acquire(1, {1, 0}).action == manager::outcome::joined); own = f.table.acquire(2, {1, 0}).request; CHECK(own); }
    gate own_started;
    std::thread worker([&] { own_started.wait(); std::lock_guard<std::mutex> lock(f.mutex); CHECK(f.table.complete(pf)); });
    // The production caller submits all owned IO before this wait. If it joined
    // first, the controlled PF below could not progress: this seam needs no sleep.
    { std::lock_guard<std::mutex> lock(f.mutex); CHECK(f.table.complete(own)); } own_started.signal();
    { std::unique_lock<std::mutex> lock(f.mutex); f.table.wait(pf, lock); CHECK(f.table.resident_slot(2) == own.slot); f.table.release_selected(); }
    worker.join();
}

static void pin_loading_and_resident() {
    fixture f(2, 2); std::lock_guard<std::mutex> lock(f.mutex);
    auto q = f.table.enqueue(1, {1, 1}); auto w = f.table.start_queued(q.request);
    const int32_t ids[] = {1, 2}; f.table.pin_selected(ids, 2); auto d = f.table.acquire(2, {1, 0}); CHECK(d.owner);
    CHECK(!f.table.acquire(3, {1, 0}).request); CHECK(f.table.complete(w.request));
    CHECK(!f.table.acquire(3, {1, 0}).request); CHECK(f.table.complete(d.request));
    CHECK(!f.table.acquire(3, {1, 0}).request); throws([&] { f.table.reset_binding(); });
    f.table.release_selected(); CHECK(f.table.acquire(3, {1, 0}).owner);
}

static void failed_generation_retry() {
    fixture f(1, 1); manager::ticket old, retry; gate waiter_entered;
    { std::lock_guard<std::mutex> lock(f.mutex); auto q = f.table.enqueue(1, {1, 1}); old = f.table.start_queued(q.request).request; }
    bool caught = false;
    std::thread waiter([&] { std::unique_lock<std::mutex> lock(f.mutex); waiter_entered.signal(); try { f.table.wait(old, lock); } catch (const std::exception &) { caught = true; } });
    waiter_entered.wait();
    { std::lock_guard<std::mutex> lock(f.mutex); CHECK(f.table.fail(old, std::make_exception_ptr(std::runtime_error("prefetch EIO"))));
      retry = f.table.acquire(1, {1, 0}).request; CHECK(retry && retry.generation != old.generation); }
    waiter.join(); CHECK(caught);
    std::lock_guard<std::mutex> lock(f.mutex); CHECK(!f.table.complete(old)); CHECK(!f.table.fail(old, {})); CHECK(f.table.complete(retry));
}

static void cancellation_binding_and_aba() {
    fixture f(1, 1); std::lock_guard<std::mutex> lock(f.mutex);
    auto q = f.table.enqueue(1, {1, 1}); CHECK(f.table.cancel_queued(q.request)); CHECK(!f.table.start_queued(q.request).owner);
    auto first = f.resident(1); auto second = f.resident(2); auto third = f.resident(1);
    CHECK(first.slot == third.slot && first.generation != third.generation);
    CHECK(!f.table.complete(first) && !f.table.fail(first, {}));
    CHECK(f.table.resident_slot(1) == third.slot);
    const auto binding = f.table.binding_generation(); f.table.reset_binding(); CHECK(f.table.binding_generation() != binding);
    auto current = f.table.acquire(1, {9, 0}); CHECK(current.request.binding != third.binding);
    CHECK(!f.table.complete(third) && !f.table.start_queued(q.request).owner);
    CHECK(f.table.complete(current.request)); CHECK(!f.table.fail(second, {}));
}

static void phase_retain_and_rebind() {
    fixture f; std::lock_guard<std::mutex> lock(f.mutex);
    const auto warm = f.table.acquire(1, {0, 0}, manager::origin::prewarm, true); CHECK(warm.owner);
    CHECK(f.table.complete(warm.request));
    // Physically separate bank prefill does not touch this table or bytes.
    CHECK(f.table.acquire(1, {2, 0}).action == manager::outcome::reused);
    const auto active = f.table.acquire(2, {2, 0}); throws([&] { f.table.reset_binding(); });
    CHECK(f.table.fail(active.request, {})); f.table.reset_binding();
    CHECK(!f.table.complete(warm.request)); CHECK(f.table.resident_slot(1) < 0);
}

static void eviction_hints_and_age() {
    fixture f(3, 3); std::lock_guard<std::mutex> lock(f.mutex);
    auto oldest = f.resident(1); f.resident(2); f.resident(3);
    f.table.set_hints({1}); auto d = f.table.acquire(4, {1, 0}); CHECK(d.victim.expert == 2); CHECK(f.table.complete(d.request));
    f.table.set_hints({}); auto next = f.table.acquire(5, {1, 0}); CHECK(next.victim.expert == 1 && next.request.slot == oldest.slot); CHECK(f.table.complete(next.request));
    f.table.set_hints({3}); auto q = f.table.enqueue(6, {1, 6}); auto pf = f.table.start_queued(q.request);
    CHECK(pf.victim.expert == 3); // PF keeps upstream pure LRU, even if hinted.
}

static void partition_and_persistent_policy() {
    fixture f(3, 2, false); std::lock_guard<std::mutex> lock(f.mutex);
    auto one = f.resident(1); f.table.set_persistent(1, true); auto two = f.resident(2);
    auto q = f.table.enqueue(3, {1, 1}); auto pf = f.table.start_queued(q.request); CHECK(pf.owner && pf.request.slot == 2); CHECK(f.table.complete(pf.request));
    auto d = f.table.acquire(4, {1, 0}); CHECK(d.owner && d.victim.expert == 2 && d.request.slot == two.slot);
    CHECK(f.table.resident_slot(1) == one.slot); CHECK(f.table.complete(d.request));
    f.table.set_persistent(1, false); f.table.touch(4); auto warm = f.table.acquire(5, {1, 0}, manager::origin::prewarm, true);
    CHECK(warm.owner && warm.request.slot < 2 && warm.victim.expert == 1);
}

static void independent_layers() {
    fixture a(1, 1), b(1, 1); std::lock_guard<std::mutex> la(a.mutex), lb(b.mutex);
    auto qa = a.table.enqueue(1, {1, 1}); auto qb = b.table.enqueue(1, {1, 1});
    CHECK(a.table.start_queued(qa.request).owner && b.table.start_queued(qb.request).owner);
    CHECK(a.table.writers() == 1 && b.table.writers() == 1);
}

static void selected_queued_blocks_worker_and_capacity_retry() {
    fixture f(1, 1); std::lock_guard<std::mutex> lock(f.mutex);
    f.resident(2); f.table.pin(2); auto queued = f.table.enqueue(1, {1, 1});
    auto failed = f.table.start_queued(queued.request); CHECK(!failed.owner && failed.action == manager::outcome::capacity);
    CHECK(f.table.lookup(1).state == manager::phase::cancelled); f.table.unpin(2);
    const int32_t ids[] = {1}; f.table.pin_selected(ids, 1); auto own = f.table.acquire(1, {1, 0}); CHECK(own.owner);
    CHECK(!f.table.start_queued(queued.request).owner); CHECK(f.table.complete(own.request)); f.table.release_selected();
    f.table.pin(3); CHECK(f.table.enqueue(3, {1, 3}).action == manager::outcome::selected); f.table.unpin(3);
}

static void prefetch_writer_budget() {
    std::mutex mutex; manager table(mutex, 32, 8, 6, true, 2);
    std::array<manager::ticket, 16> queued; std::array<manager::result, 16> popped;
    { std::lock_guard<std::mutex> lock(mutex); for (int i = 0; i < 16; ++i) queued[i] = table.enqueue(i, {1, (uint64_t) i + 1}).request; }
    gate start; std::vector<std::thread> workers;
    for (int i = 0; i < 16; ++i) workers.emplace_back([&, i] { start.wait(); std::lock_guard<std::mutex> lock(mutex); popped[i] = table.start_queued(queued[i]); });
    start.signal(); for (auto & worker : workers) worker.join();
    std::lock_guard<std::mutex> lock(mutex); int owners = 0, limited = 0;
    for (int i = 0; i < 16; ++i) {
        if (popped[i].owner) ++owners;
        else { ++limited; CHECK(popped[i].action == manager::outcome::writer_limit); CHECK(table.lookup(i).state == manager::phase::cancelled); }
    }
    CHECK(owners == 2 && limited == 14 && table.prefetch_writers() == 2);
    throws([&] { table.set_prefetch_limit(1); });
    auto demand = table.acquire(20, {2, 0}); CHECK(demand.owner && table.prefetch_writers() == 2); CHECK(table.complete(demand.request));
    bool failed = false;
    for (const auto & result : popped) if (result.owner) {
        if (!failed) { CHECK(table.fail(result.request, {})); failed = true; }
        else CHECK(table.complete(result.request));
        CHECK(!table.fail(result.request, {})); // stale callbacks cannot refund twice.
    }
    CHECK(table.prefetch_writers() == 0 && table.writers() == 0);
    auto q = table.enqueue(21, {3, 21}); auto pf = table.start_queued(q.request); CHECK(pf.owner && table.prefetch_writers() == 1);
    CHECK(table.complete(pf.request)); table.reset_binding(); CHECK(table.prefetch_writers() == 0);
    table.set_prefetch_limit(0); auto disabled = table.enqueue(22, {4, 22});
    CHECK(table.start_queued(disabled.request).action == manager::outcome::writer_limit);
    CHECK(table.lookup(22).state == manager::phase::cancelled);
    table.set_prefetch_limit(2); auto restarted = table.enqueue(22, {5, 22});
    auto started = table.start_queued(restarted.request); CHECK(started.owner); CHECK(table.complete(started.request));
}

// Compare the indexed allocator to an independent tiny scan under hint,
// permanent-pin and touch churn. This checks heap moves preserve old LRU age.
static void indexed_allocator_matches_scan() {
    fixture f(8, 6, true); std::lock_guard<std::mutex> lock(f.mutex);
    struct ref_slot { int expert = -1; uint64_t age = 0; };
    std::array<ref_slot, 8> slots{}; std::array<bool, 32> hint{}, persistent{};
    uint64_t age = 0; uint32_t random = 12345;
    auto draw = [&]() { random = random * 1664525u + 1013904223u; return random; };
    auto find = [&](int expert) { for (int i = 0; i < 8; ++i) if (slots[i].expert == expert) return i; return -1; };
    std::vector<int32_t> hints; hints.reserve(4);
    for (int step = 0; step < 10000; ++step) {
        if (step % 5 == 0) {
            hint.fill(false); hints.clear();
            for (int i = 0; i < 4; ++i) { int e = (int) (draw() % 32); hint[e] = true; hints.push_back(e); }
            f.table.set_hints(hints);
        }
        if (step % 7 == 0) { int e = (int) (draw() % 32); persistent[e] = !persistent[e]; f.table.set_persistent(e, persistent[e]); }
        int expert = (int) (draw() % 32), expected = find(expert);
        if (step % 3 == 0) {
            f.table.touch(expert); if (expected >= 0) slots[expected].age = ++age; continue;
        }
        const bool pf = step % 2 == 0;
        if (expected < 0) {
            for (int i = 0; i < 8; ++i) if (slots[i].expert < 0) { expected = i; break; }
            for (int pass = 0; pass < 2 && expected < 0; ++pass) {
                uint64_t oldest = UINT64_MAX;
                for (int i = 0; i < 8; ++i) {
                    const int resident = slots[i].expert;
                    if (resident < 0 || persistent[resident] || (!pf && pass == 0 && hint[resident])) continue;
                    if (slots[i].age < oldest) { oldest = slots[i].age; expected = i; }
                }
            }
        }
        auto r = f.table.acquire(expert, {(uint64_t) step, 0}, pf ? manager::origin::prefetch : manager::origin::demand);
        CHECK((bool) r.request == (expected >= 0));
        if (expected < 0) continue;
        CHECK(r.request.slot == expected);
        if (r.owner) { CHECK(f.table.complete(r.request)); slots[expected] = {expert, ++age}; }
        else CHECK(slots[expected].expert == expert && r.action == manager::outcome::reused);
    }
}

static manager::ticket prefetched(fixture & f, int expert) {
    const auto q = f.table.enqueue(expert, {7, 77}); CHECK(q.action == manager::outcome::created);
    const auto r = f.table.start_queued(q.request); CHECK(r.owner); CHECK(f.table.complete(r.request)); return r.request;
}

static void probation_touch_hints_and_coalescing() {
    {
        fixture f(2, 2, true, true); std::lock_guard<std::mutex> lock(f.mutex);
        f.resident(1); const auto pf = prefetched(f, 2);
        f.table.set_hints({2}); CHECK(f.table.enqueue(2, {8, 88}).action == manager::outcome::coalesced);
        CHECK(f.table.acquire(2, {8, 0}).action == manager::outcome::reused);
        CHECK(f.table.probation_stats().promotions == 0);
        const auto demand = f.table.acquire(3, {8, 0});
        CHECK(demand.victim.expert == 2 && demand.victim.generation == pf.generation && demand.probation_victim);
        CHECK(f.table.complete(demand.request)); CHECK(f.table.probation_stats().demand_victims == 1);
    }
    {
        fixture f(2, 2, true, true); std::lock_guard<std::mutex> lock(f.mutex);
        const auto pf = prefetched(f, 1); f.resident(2); f.table.set_hints({1});
        CHECK(f.table.touch(1)); CHECK(!f.table.touch(1)); CHECK(!f.table.touch(2));
        const auto after = f.table.lookup(1);
        CHECK(after.owner == manager::origin::prefetch && after.request.generation == pf.generation);
        CHECK(after.request.binding == pf.binding && after.producer.epoch == 7 && after.producer.job == 77);
        // Once genuinely touched, the older PF follows the normal hint rule.
        const auto demand = f.table.acquire(3, {8, 0}); CHECK(demand.victim.expert == 2 && !demand.probation_victim);
        CHECK(f.table.complete(demand.request)); CHECK(f.table.probation_stats().promotions == 1);
    }
}

static void probation_selected_loading_and_persistent() {
    fixture f(3, 3, true, true); std::lock_guard<std::mutex> lock(f.mutex);
    prefetched(f, 1); const auto q = f.table.enqueue(2, {7, 77}); const auto loading = f.table.start_queued(q.request);
    CHECK(loading.owner && !f.table.touch(2)); f.resident(3); f.table.set_persistent(3, true);
    const int32_t selected[] = {1, 2, 1}; f.table.pin_selected(selected, 3);
    CHECK(!f.table.acquire(4, {8, 0}).request); CHECK(f.table.complete(loading.request));
    CHECK(!f.table.acquire(4, {8, 0}).request);
    CHECK(f.table.touch(1) && f.table.touch(2)); f.table.release_selected();
    const auto demand = f.table.acquire(4, {8, 0}); CHECK(demand.victim.expert == 1 && !demand.probation_victim);
    CHECK(f.table.complete(demand.request)); CHECK(f.table.probation_stats().demand_victims == 0);
    f.table.set_persistent(2, true); f.table.set_persistent(4, true);
    f.table.set_persistent(3, false); const auto pf = f.table.acquire(5, {9, 99}, manager::origin::prefetch);
    CHECK(pf.owner); CHECK(f.table.complete(pf.request)); f.table.set_persistent(5, true);
    CHECK(!f.table.acquire(6, {9, 0}).request); f.table.set_persistent(5, false);
    CHECK(f.table.acquire(6, {9, 0}).probation_victim);
}

static void probation_serial_prefetch_and_prewarm_lru() {
    fixture f(4, 4, true, true); std::lock_guard<std::mutex> lock(f.mutex);
    for (int e = 1; e <= 4; ++e) f.resident(e);
    std::array<manager::ticket, 3> queued;
    for (int i = 0; i < 3; ++i) queued[i] = f.table.enqueue(i + 5, {7, 77}).request;
    for (int i = 0; i < 3; ++i) {
        const auto pf = f.table.start_queued(queued[i]); CHECK(pf.owner && pf.victim.expert == i + 1 && !pf.probation_victim);
        CHECK(f.table.complete(pf.request)); f.table.set_hints({5, 6, 7});
    }
    for (int e = 5; e <= 7; ++e) CHECK(f.table.resident_slot(e) >= 0);
    auto next = f.table.acquire(8, {7, 77}, manager::origin::prefetch); CHECK(next.victim.expert == 4); CHECK(f.table.complete(next.request));
    next = f.table.acquire(9, {7, 77}, manager::origin::prefetch); CHECK(next.victim.expert == 5); CHECK(f.table.complete(next.request));
    const auto warm = f.table.acquire(10, {8, 0}, manager::origin::prewarm, true);
    CHECK(warm.victim.expert == 6 && !warm.probation_victim); CHECK(f.table.complete(warm.request));
    CHECK(f.table.probation_stats().demand_victims == 0);
}

static void probation_regions_and_cache_only() {
    for (bool unified : {false, true}) {
        fixture f(4, 2, unified, true); std::lock_guard<std::mutex> lock(f.mutex);
        f.resident(1); f.resident(2); const auto first = prefetched(f, 3); prefetched(f, 4);
        CHECK(first.slot == 2);
        const auto demand = f.table.acquire(5, {8, 0});
        CHECK(demand.victim.expert == (unified ? 3 : 1)); CHECK(demand.probation_victim == unified); CHECK(f.table.complete(demand.request));
        const auto warm = f.table.acquire(6, {8, 0}, manager::origin::prewarm, true);
        CHECK(warm.request.slot < 2 && warm.victim.expert == (unified ? 1 : 2) && !warm.probation_victim);
        CHECK(f.table.complete(warm.request)); CHECK(f.table.resident_slot(4) == 3);
    }
}

static void probation_failure_aba_and_reset() {
    fixture f(1, 1, true, true); std::lock_guard<std::mutex> lock(f.mutex);
    auto q = f.table.enqueue(1, {1, 1}); const auto failed = f.table.start_queued(q.request).request;
    CHECK(f.table.fail(failed, {})); auto demand = f.table.acquire(2, {1, 0});
    CHECK(demand.owner && !demand.probation_victim); CHECK(f.table.complete(demand.request)); CHECK(!f.table.touch(2));
    const auto pf = prefetched(f, 1); CHECK(!f.table.complete(failed) && !f.table.fail(failed, {}));
    CHECK(f.table.touch(1)); CHECK(f.table.probation_stats().promotions == 1);
    f.table.reset_binding(); demand = f.table.acquire(1, {2, 0});
    CHECK(demand.request.binding != pf.binding && !demand.probation_victim);
    CHECK(f.table.complete(demand.request)); CHECK(!f.table.touch(1)); CHECK(!f.table.complete(pf));
    q = f.table.enqueue(4, {2, 2}); CHECK(f.table.cancel_queued(q.request)); CHECK(!f.table.start_queued(q.request).owner);
    prefetched(f, 4); demand = f.table.acquire(5, {2, 0}); CHECK(demand.probation_victim);
    CHECK(f.table.complete(demand.request)); CHECK(!f.table.touch(5));
    CHECK(f.table.probation_stats().promotions == 1 && f.table.probation_stats().demand_victims == 1);
}

static void probation_heap_allocation_gate() {
    std::mutex mutex; std::array<size_t, 2> counts{};
    for (int enabled = 0; enabled < 2; ++enabled) {
        allocations = 0; count_allocations = true;
        { manager table(mutex, 32, 8, 6, true, -1, enabled != 0); }
        count_allocations = false; counts[enabled] = allocations.load();
    }
    CHECK(counts[1] == counts[0] + 2); // Only the enabled path reserves heaps 6/7.
}

static void probation_indexed_allocator_matches_scan() {
    fixture f(8, 6, true, true); std::lock_guard<std::mutex> lock(f.mutex);
    struct ref_slot { int expert = -1; uint64_t age = 0; bool unused = false; };
    std::array<ref_slot, 8> slots{}; std::array<bool, 32> hint{}, persistent{}, pinned{};
    uint64_t age = 0, promotions = 0, victims = 0; uint32_t random = 54321;
    auto draw = [&]() { random = random * 1664525u + 1013904223u; return random; };
    auto find = [&](int expert) { for (int i = 0; i < 8; ++i) if (slots[i].expert == expert) return i; return -1; };
    std::vector<int32_t> hints; hints.reserve(4);
    for (int step = 0; step < 10000; ++step) {
        if (step == 5000) { f.table.release_selected(); pinned.fill(false); f.table.reset_binding(); slots = {}; hint.fill(false); }
        if (step % 5 == 0) {
            hint.fill(false); hints.clear();
            for (int i = 0; i < 4; ++i) { const int e = (int) (draw() % 32); hint[e] = true; hints.push_back(e); }
            f.table.set_hints(hints);
        }
        if (step % 7 == 0) { const int e = (int) (draw() % 32); persistent[e] = !persistent[e]; f.table.set_persistent(e, persistent[e]); }
        if (step % 11 == 0) {
            f.table.release_selected(); pinned.fill(false); const int e = slots[(size_t) (draw() % 8)].expert;
            if (e >= 0) { f.table.pin_selected(&e, 1); pinned[e] = true; }
        }
        const int expert = (int) (draw() % 32); const int resident = find(expert);
        if (step % 3 == 0) {
            const bool promoted = resident >= 0 && slots[resident].unused;
            CHECK(f.table.touch(expert) == promoted);
            if (resident >= 0) { slots[resident].age = ++age; slots[resident].unused = false; }
            promotions += promoted; continue;
        }
        const auto who = step % 17 == 0 ? manager::origin::prewarm : step % 2 == 0 ? manager::origin::prefetch : manager::origin::demand;
        const bool cache_only = who == manager::origin::prewarm;
        int expected = resident;
        if (expected < 0) {
            const int limit = cache_only ? 6 : 8;
            for (int i = 0; i < limit; ++i) if (slots[i].expert < 0) { expected = i; break; }
            if (expected < 0) {
                int best_rank = 4; uint64_t oldest = UINT64_MAX;
                for (int i = 0; i < limit; ++i) {
                    const int e = slots[i].expert; if (persistent[e] || pinned[e]) continue;
                    const int rank = who != manager::origin::demand ? 0 : slots[i].unused ? 0 : hint[e] ? 2 : 1;
                    if (rank < best_rank || (rank == best_rank && slots[i].age < oldest)) {
                        expected = i; best_rank = rank; oldest = slots[i].age;
                    }
                }
            }
        }
        const bool victim = resident < 0 && expected >= 0 && slots[expected].expert >= 0 &&
            who == manager::origin::demand && slots[expected].unused;
        const auto r = f.table.acquire(expert, {(uint64_t) step, 0}, who, cache_only);
        CHECK((bool) r.request == (expected >= 0)); CHECK(r.probation_victim == victim);
        if (expected < 0) continue;
        CHECK(r.request.slot == expected);
        if (r.owner) { CHECK(f.table.complete(r.request)); slots[expected] = {expert, ++age, who == manager::origin::prefetch}; victims += victim; }
        else CHECK(r.action == manager::outcome::reused);
    }
    f.table.release_selected(); const auto counters = f.table.probation_stats();
    CHECK(counters.promotions == promotions && counters.demand_victims == victims);
}

static void hot_metadata_no_allocations() {
    for (bool enabled : {false, true}) {
        fixture f(8, 8, true, enabled); std::vector<int32_t> hints{1, 2, 3};
        std::lock_guard<std::mutex> lock(f.mutex);
        allocations = 0; count_allocations = true;
        for (int i = 0; i < 1000; ++i) {
            const int expert = i % 16;
            auto q = f.table.enqueue(expert, {(uint64_t) i, (uint64_t) i});
            auto r = q.action == manager::outcome::created ? f.table.start_queued(q.request) : f.table.acquire(expert, {(uint64_t) i, 0});
            CHECK(r.request); if (r.owner) CHECK(f.table.complete(r.request));
            const int32_t ids[] = {expert, expert}; f.table.pin_selected(ids, 2); f.table.touch(expert); f.table.set_hints(hints); f.table.release_selected();
        }
        count_allocations = false; CHECK(allocations == 0);
    }
}

using schedule = llama_moe::prefetch_schedule;

static void refresh_reserves_near_budget_after_far_completion() {
    fixture f(8, 8); schedule plans; plans.resize(40);
    CHECK(plans.begin(2, 1, 8) && plans.allowance(2, 1, false) == 6);
    std::lock_guard<std::mutex> lock(f.mutex);
    for (int e = 0; e < 6; ++e) {
        const auto q = f.table.enqueue(e, {1, 1}); CHECK(q.action == manager::outcome::created);
        plans.created(2, 1, e); const auto w = f.table.start_queued(q.request); CHECK(w.owner); CHECK(f.table.complete(w.request));
    }
    CHECK(plans.refine(2, 1)); CHECK(!plans.refine(2, 1));
    CHECK(plans.allowance(2, 1, true) == 2);
    // Completed far owners coalesce and consume none of the reserved requests.
    for (int e = 0; e < 6; ++e) CHECK(f.table.enqueue(e, {1, 2}).action == manager::outcome::coalesced);
    for (int e = 6; e < 8; ++e) {
        CHECK(f.table.enqueue(e, {1, 2}).action == manager::outcome::created); plans.created(2, 1, e);
    }
    CHECK(plans.allowance(2, 1, true) == 0); throws([&] { plans.created(2, 1, 8); });
}

static void refresh_refunds_only_unstarted_exact_tickets() {
    fixture f(8, 8); schedule plans; plans.resize(40); plans.begin(3, 1, 8);
    std::lock_guard<std::mutex> lock(f.mutex);
    std::array<manager::ticket, 6> pending;
    for (int e = 0; e < 6; ++e) { pending[e] = f.table.enqueue(e, {1, 1}).request; plans.created(3, 1, e); }
    const auto writing = f.table.start_queued(pending[0]); CHECK(writing.owner);
    CHECK(!f.table.cancel_queued(pending[0])); CHECK(plans.find(3, 1)->spent == 6);
    CHECK(f.table.cancel_queued(pending[1])); CHECK(plans.refund(3, 1, 1));
    CHECK(!plans.refund(3, 1, 1)); CHECK(plans.allowance(3, 1, true) == 3);
    const auto replacement = f.table.enqueue(1, {1, 2}); CHECK(replacement.action == manager::outcome::created);
    CHECK(!f.table.cancel_queued(pending[1])); CHECK(f.table.start_queued(replacement.request).owner);
    CHECK(f.table.complete(writing.request));
}

static void refresh_epoch_and_replacement_do_not_overflow() {
    schedule plans; plans.resize(40); plans.begin(4, 1, 16);
    CHECK(plans.allowance(4, 1, false) == 14);
    for (int e = 0; e < 14; ++e) plans.created(4, 1, e);
    for (int e = 0; e < 14; ++e) CHECK(plans.refund(4, 1, e));
    CHECK(plans.refine(4, 1));
    for (int e = 16; e < 32; ++e) plans.created(4, 1, e);
    CHECK(plans.find(4, 1)->spent == 16 && plans.allowance(4, 1, true) == 0);
    CHECK(plans.begin(4, 2, 8)); CHECK(!plans.refund(4, 1, 16)); CHECK(!plans.refine(4, 1));
    CHECK(plans.find(4, 2)->spent == 0 && plans.allowance(4, 2, false) == 6);
}

static void refresh_deadline_is_stable_and_expires_consumed_work() {
    struct candidate { uint64_t epoch; int layer; int order; };
    std::vector<candidate> q{{2, 2, 0}, {1, 9, 1}, {1, 3, 2}, {1, 3, 3}};
    const auto first = std::min_element(q.begin(), q.end(), [](const candidate & a, const candidate & b) {
        return schedule::earlier(a.epoch, a.layer, b.epoch, b.layer);
    });
    CHECK(first->order == 2);
    CHECK(schedule::due(1, 3, 1, 3)); CHECK(schedule::due(1, 39, 2, 0));
    CHECK(!schedule::due(1, 4, 1, 3)); CHECK(!schedule::due(2, 0, 1, 39));
}

static void refresh_budget_metadata_does_not_allocate() {
    schedule plans; plans.resize(40);
    allocations = 0; count_allocations = true;
    for (uint64_t ep = 1; ep < 1000; ++ep) {
        CHECK(plans.begin(3, ep, 8));
        for (int e = 0; e < 6; ++e) plans.created(3, ep, e);
        CHECK(plans.refine(3, ep)); CHECK(plans.refund(3, ep, 1));
        for (int e = 6; e < 9; ++e) plans.created(3, ep, e);
    }
    count_allocations = false; CHECK(allocations == 0);
}

int main() {
    struct test { const char * name; void (*run)(); };
    const test tests[] = {
        {"refresh far completion reserve", refresh_reserves_near_budget_after_far_completion},
        {"refresh exact queued refund", refresh_refunds_only_unstarted_exact_tickets},
        {"refresh epoch and replacement budget", refresh_epoch_and_replacement_do_not_overflow},
        {"refresh stable deadlines", refresh_deadline_is_stable_and_expires_consumed_work},
        {"refresh metadata allocation gate", refresh_budget_metadata_does_not_allocate},
        {"horizon coalescing", horizon_coalescing}, {"queued demand promotion", queued_demand_promotion},
        {"queued claim race", queued_claim_race}, {"family publication and wait", family_publication_and_wait},
        {"own misses before join", own_misses_before_join}, {"pin loading and resident", pin_loading_and_resident},
        {"failed generation retry", failed_generation_retry}, {"cancellation binding ABA", cancellation_binding_and_aba},
        {"phase retain and rebind", phase_retain_and_rebind}, {"eviction hint and age", eviction_hints_and_age},
        {"partition and persistent policy", partition_and_persistent_policy}, {"independent layers", independent_layers},
        {"selected and capacity retry", selected_queued_blocks_worker_and_capacity_retry}, {"hot metadata no allocations", hot_metadata_no_allocations},
        {"indexed allocator matches scan", indexed_allocator_matches_scan},
        {"prefetch writer budget", prefetch_writer_budget},
        {"probation touch hints coalescing", probation_touch_hints_and_coalescing},
        {"probation selected loading persistent", probation_selected_loading_and_persistent},
        {"probation serial PF and prewarm LRU", probation_serial_prefetch_and_prewarm_lru},
        {"probation regions cache only", probation_regions_and_cache_only},
        {"probation failure ABA reset", probation_failure_aba_and_reset},
        {"probation heap allocation gate", probation_heap_allocation_gate},
        {"probation indexed allocator matches scan", probation_indexed_allocator_matches_scan},
    };
    for (const auto & t : tests) { t.run(); std::printf("PASS %s\n", t.name); }
    std::printf("%zu tests passed; ticket=%zu result=%zu\n", sizeof(tests) / sizeof(tests[0]), sizeof(manager::ticket), sizeof(manager::result));
}
