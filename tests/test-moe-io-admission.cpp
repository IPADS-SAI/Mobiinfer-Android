#include "../src/llama-moe-io-admission.h"
#include "../src/llama-moe-slot-manager.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>
#include <thread>

using admission = llama_moe::io_admission;
using manager = llama_moe::slot_manager;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #c); std::abort(); } } while (false)
static std::atomic<bool> count_new{false};
static std::atomic<size_t> new_count{0};
void * operator new(size_t n) { if (count_new.load(std::memory_order_relaxed)) ++new_count;
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
[[gnu::noinline]] void operator delete(void * p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void * p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void * p, size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void * p, size_t) noexcept { std::free(p); }

struct latch {
    std::mutex mutex; std::condition_variable cv; bool opened = false;
    void open() { std::lock_guard<std::mutex> lock(mutex); opened = true; cv.notify_all(); }
    void wait() { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return opened; }); }
};
struct probe {
    std::mutex mutex; std::condition_variable cv;
    std::array<admission::observation, 4096> events{}; size_t count = 0;
    static void observe(void * self, const admission::observation & event) noexcept {
        auto & p = *static_cast<probe *>(self); std::lock_guard<std::mutex> lock(p.mutex);
        CHECK(p.count < p.events.size()); p.events[p.count++] = event; p.cv.notify_all();
    }
    size_t total(admission::action action) const {
        size_t n = 0; for (size_t i = 0; i < count; ++i) n += events[i].event == action; return n;
    }
    void until(admission::action action, size_t n) {
        std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return total(action) >= n; });
    }
    std::vector<int> grants_after(size_t first = 0) {
        std::lock_guard<std::mutex> lock(mutex); std::vector<int> ids;
        for (size_t i = first; i < count; ++i) if (events[i].event == admission::action::grant) ids.push_back(events[i].request.expert);
        return ids;
    }
};
static admission::key key(int e, uint64_t generation = 1, int layer = 0, uint64_t binding = 1) { return {layer,e,binding,generation}; }
static admission::key key(manager::ticket t, int layer = 0) { return {layer,t.expert,t.binding,t.generation}; }
static void balanced(admission & gate) { auto s = gate.snapshot(); CHECK(s.active == 0 && s.waiting == 0 && s.grants == s.releases); }

static void cap_and_no_new_threads() {
    probe p; admission gate(8,32,probe::observe,&p); latch release; std::vector<std::thread> workers;
    for (int i = 0; i < 16; ++i) workers.emplace_back([&,i] { auto lease = gate.acquire(key(i), i % 2 ? admission::origin::prefetch : admission::origin::demand); release.wait(); });
    p.until(admission::action::wait,16); CHECK(gate.snapshot().active == 8 && gate.snapshot().waiting == 8);
    release.open(); for (auto & worker : workers) worker.join(); CHECK(gate.snapshot().peak == 8); balanced(gate);
}
static void inherited_priority_and_fifo() {
    probe p; admission gate(1,32,probe::observe,&p); auto blocker = gate.acquire(key(0),admission::origin::demand);
    std::thread background([&] { auto lease = gate.acquire(key(1),admission::origin::prefetch); });
    p.until(admission::action::wait,2);
    std::thread needed([&] { auto lease = gate.acquire(key(2,7),admission::origin::prefetch); });
    p.until(admission::action::wait,3);
    std::thread own([&] { auto lease = gate.acquire(key(3),admission::origin::demand); });
    p.until(admission::action::wait,4);
    const auto wanted = key(2,7); auto scope = gate.begin_needed(&wanted,1);
    CHECK(gate.snapshot().boosted == 1); blocker.reset(); background.join(); needed.join(); own.join(); scope.reset();
    const auto order = p.grants_after(); CHECK(order.size() == 4);
    // budget1 means at most one foreground grant then one background grant.
    CHECK(order[0] == 0 && order[1] == 2 && order[2] == 1 && order[3] == 3); balanced(gate);
}
static void late_needed_and_exact_aba() {
    probe p; admission gate(1,32,probe::observe,&p); auto blocker = gate.acquire(key(0),admission::origin::demand);
    const auto wanted = key(3,9,2,4); auto scope = gate.begin_needed(&wanted,1);
    std::thread stale_generation([&] { auto lease = gate.acquire(key(3,8,2,4),admission::origin::prefetch); });
    p.until(admission::action::wait,2);
    std::thread stale_binding([&] { auto lease = gate.acquire(key(3,9,2,3),admission::origin::prefetch); });
    p.until(admission::action::wait,3);
    std::thread exact([&] { auto lease = gate.acquire(wanted,admission::origin::prefetch); });
    p.until(admission::action::wait,4); blocker.reset(); stale_generation.join(); stale_binding.join(); exact.join(); scope.reset();
    const auto s = gate.snapshot(); CHECK(s.needed_grants == 1 && s.background_grants == 2 && s.boosted == 0); balanced(gate);
}
static void fairness_eight_to_one() {
    probe p; admission gate(8,64,probe::observe,&p); std::vector<admission::permit> blockers;
    for (int i=0;i<8;++i) blockers.push_back(gate.acquire(key(i),admission::origin::demand));
    std::thread background([&] { auto lease = gate.acquire(key(50),admission::origin::prefetch); });
    p.until(admission::action::wait,9);
    std::vector<std::thread> foreground;
    for (int i=8;i<24;++i) { foreground.emplace_back([&,i] { auto lease = gate.acquire(key(i),admission::origin::demand); }); p.until(admission::action::wait,(size_t)i+2); }
    for (auto & lease : blockers) lease.reset();
    for (auto & thread : foreground) thread.join();
    background.join();
    const auto order = p.grants_after(); CHECK(order.size() == 25 && order[16] == 50); balanced(gate);
}
static void scope_exception_demotes() {
    probe p; admission gate(1,32,probe::observe,&p); auto blocker = gate.acquire(key(0),admission::origin::demand);
    std::thread background([&] { auto lease = gate.acquire(key(1),admission::origin::prefetch); }); p.until(admission::action::wait,2);
    std::thread needed([&] { auto lease = gate.acquire(key(2),admission::origin::prefetch); }); p.until(admission::action::wait,3);
    try { const auto wanted = key(2); auto scope = gate.begin_needed(&wanted,1); throw std::runtime_error("own IO failed"); } catch (const std::runtime_error &) {}
    blocker.reset(); background.join(); needed.join(); const auto order = p.grants_after(); CHECK(order[1] == 1 && order[2] == 2); balanced(gate);
}
static void lease_move_and_exception() {
    admission gate(1,32);
    try { auto first = gate.acquire(key(1),admission::origin::demand); admission::permit second(std::move(first)); admission::permit third; third = std::move(second); throw std::runtime_error("pread EIO"); }
    catch (const std::runtime_error &) {}
    auto s = gate.snapshot(); CHECK(s.grants == 1 && s.releases == 1); balanced(gate);
}
static void close_cancels_not_active_io() {
    probe p; admission gate(1,32,probe::observe,&p); auto active = gate.acquire(key(0),admission::origin::prefetch);
    std::atomic<int> cancelled{0}; std::vector<std::thread> workers;
    for(int i=1;i<5;++i) workers.emplace_back([&,i] { try { auto lease = gate.acquire(key(i),admission::origin::demand); CHECK(false); } catch(const llama_moe::io_admission_cancelled &) { ++cancelled; } });
    p.until(admission::action::wait,5); gate.close(); for(auto & t:workers)t.join();
    CHECK(cancelled == 4 && gate.snapshot().active == 1 && gate.snapshot().waiting == 0);
    active.reset(); gate.close(); CHECK(gate.snapshot().cancelled == 4); balanced(gate);
    try { auto lease = gate.acquire(key(6),admission::origin::demand); CHECK(false); } catch (const llama_moe::io_admission_cancelled &) {}
    balanced(gate);
}
static void combined_manager_join_and_failure() {
    std::mutex layer; manager table(layer,32,4,2,true,2); probe p; admission gate(1,32,probe::observe,&p);
    auto blocker = gate.acquire(key(30),admission::origin::demand); manager::ticket pf;
    { std::lock_guard<std::mutex> lock(layer); auto q = table.enqueue(1,{1,1}); pf=table.start_queued(q.request).request; }
    std::atomic<int> io_count{0}; std::array<int,3> families{};
    std::thread worker([&] {
        { auto lease=gate.acquire(key(pf),admission::origin::prefetch); ++io_count; families={1,2,3}; }
        // No I permit/lock may survive into L publication.
        CHECK(gate.snapshot().active == 0); std::lock_guard<std::mutex> lock(layer); CHECK(table.complete(pf));
    });
    p.until(admission::action::wait,2); admission::key snapshot;
    { std::lock_guard<std::mutex> lock(layer); const int32_t ids[]={1};table.pin_selected(ids,1);
      auto joined=table.acquire(1,{2,0});CHECK(!joined.owner && joined.action==manager::outcome::joined);snapshot=key(joined.request); }
    { auto needed=gate.begin_needed(&snapshot,1); blocker.reset(); std::unique_lock<std::mutex> lock(layer);table.wait(pf,lock);CHECK(families[0]==1 && families[2]==3); }
    worker.join();CHECK(io_count==1);{std::lock_guard<std::mutex> lock(layer);table.release_selected();} balanced(gate);
    manager::ticket failed;
    {std::lock_guard<std::mutex> lock(layer);auto q=table.enqueue(2,{3,2});failed=table.start_queued(q.request).request;}
    try {auto lease=gate.acquire(key(failed),admission::origin::prefetch);throw std::runtime_error("PF EIO");}
    catch(...){CHECK(gate.snapshot().active==0);std::lock_guard<std::mutex> lock(layer);CHECK(table.fail(failed,std::current_exception()));}
    manager::ticket retry;
    {std::unique_lock<std::mutex> lock(layer);try{table.wait(failed,lock);CHECK(false);}catch(const std::runtime_error&){}retry=table.acquire(2,{3,0}).request;}
    {auto lease=gate.acquire(key(retry),admission::origin::demand);CHECK(retry.generation!=failed.generation);}
    {std::lock_guard<std::mutex> lock(layer);CHECK(table.complete(retry));} balanced(gate);
}
static void normal_stop_drains_without_cancellation() {
    probe p; admission gate(2,32,probe::observe,&p); latch release; std::vector<std::thread> workers;
    for(int i=0;i<8;++i)workers.emplace_back([&,i]{auto lease=gate.acquire(key(i),admission::origin::prefetch);release.wait();});
    p.until(admission::action::wait,8); release.open(); for(auto & t:workers)t.join();gate.close();
    CHECK(gate.snapshot().cancelled==0 && gate.snapshot().grants==8);balanced(gate);
}
static void invalid_scope_is_transactional() {
    admission gate(1,32); const auto bad=key(40);bool failed=false;
    try{auto scope=gate.begin_needed(&bad,1);}catch(const std::invalid_argument&){failed=true;}CHECK(failed);
    const auto good=key(1);{auto scope=gate.begin_needed(&good,1);auto lease=gate.acquire(good,admission::origin::prefetch);}balanced(gate);
}
static void no_hot_allocations_and_off_branch() {
    admission gate(8,32);std::unique_ptr<admission> off; new_count=0;count_new=true;
    for(int i=0;i<1000;++i){if(off){auto lease=off->acquire(key(1),admission::origin::demand);}auto wanted=key(i%32,(uint64_t)i+1);auto scope=gate.begin_needed(&wanted,1);auto lease=gate.acquire(wanted,admission::origin::prefetch);}
    count_new=false;CHECK(new_count==0);balanced(gate);
}
int main(){
    struct test{const char*name;void(*run)();};const test tests[]={
        {"cap",cap_and_no_new_threads},{"inherited FIFO",inherited_priority_and_fifo},{"late needed ABA",late_needed_and_exact_aba},
        {"8:1 fairness",fairness_eight_to_one},{"exception demotion",scope_exception_demotes},{"lease exception",lease_move_and_exception},
        {"exceptional close",close_cancels_not_active_io},{"manager join and retry",combined_manager_join_and_failure},
        {"normal stop",normal_stop_drains_without_cancellation},{"scope validation",invalid_scope_is_transactional},{"hot no allocations",no_hot_allocations_and_off_branch}};
    for(const auto&t:tests){t.run();std::printf("PASS %s\n",t.name);}std::printf("%zu tests passed\n",sizeof(tests)/sizeof(tests[0]));
}
