#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace llama_moe {

// The caller holds one layer mutex for every operation. No method takes a queue
// lock or performs IO. Queued requests own no slot; only Loading owns a writer.
class slot_manager {
public:
    enum class phase { absent, queued, loading, resident, failed, cancelled };
    enum class origin { prefetch, demand, prewarm };
    enum class outcome { created, coalesced, promoted, joined, reused, stale, capacity, selected, writer_limit };
    struct ticket {
        int32_t expert = -1;
        int32_t slot = -1;
        uint64_t binding = 0;
        uint64_t generation = 0;
        explicit operator bool() const { return expert >= 0 && generation != 0; }
    };
    struct provenance { uint64_t epoch = 0, job = 0; };
    struct observation {
        phase state = phase::absent;
        ticket request;
        origin owner = origin::prefetch;
        provenance producer;
    };
    struct result {
        ticket request;
        ticket previous; // queued ticket promoted, or resident victim evicted
        ticket victim;
        outcome action = outcome::stale;
        bool owner = false;
        bool probation_victim = false;
    };
    struct probation_counters { uint64_t promotions = 0, demand_victims = 0; };
    /// `LLAMA_MOE_LIVE_UNION` 的准入视图。只读，不改任何状态；调用方必须持层锁。
    /// 语义见 llama-moe-live-union.h：capacity 之外的三项对应 avail/evictable 的分母。
    struct occupancy {
        int32_t resident  = 0;   // 常驻（含受保护、含本批要用的）
        int32_t loading   = 0;   // 在途写者占住的槽（不可淘汰、不可复用）
        int32_t pinned    = 0;   // 常驻但受保护（pin/lease/persistent）且不属于本批
        int32_t evictable = 0;   // 常驻、不属于本批、可被 LRU 淘汰
    };

private:
    struct entry {
        observation value;
        std::exception_ptr error;
        uint32_t pins = 0;
        bool persistent = false;
        bool hint = false;
        uint64_t selected_stamp = 0;
    };
    struct slot {
        ticket owner;
        uint64_t age = 0;
        int32_t heap = -1, position = -1;
        int32_t region = 0;
        bool pf_unselected = false;
    };
    std::mutex & mutex;
    std::condition_variable changed;
    std::vector<entry> entries;
    std::vector<slot> slots;
    // 0/1: free cache/staging; 2/3: unhinted; 4/5: hinted.
    // Indexed, reserved heaps preserve exact age ordering when a hint/pin
    // changes. No heap node allocation and no expert/window victim scan.
    // 6/7: unselected PF residents, used only by demand-time probation.
    std::array<std::vector<int32_t>, 8> heaps;
    std::vector<int32_t> selected, hinted;
    size_t selected_count = 0, hinted_count = 0;
    uint64_t binding = 1, generation = 0, age = 0, selected_stamp = 0;
    uint32_t loading_count = 0, prefetch_loading = 0, pin_count = 0;
    int32_t prefetch_limit;
    int32_t cache_slots;
    bool unified;
    bool pf_probation;
    probation_counters probation;

    void check_expert(int32_t expert) const {
        if (expert < 0 || (size_t) expert >= entries.size()) throw std::out_of_range("MoE expert outside request table");
    }
    static bool same(ticket a, ticket b) {
        return a.expert == b.expert && a.binding == b.binding && a.generation == b.generation;
    }
    bool current(ticket t) const {
        return t && t.binding == binding && (size_t) t.expert < entries.size() && same(entries[t.expert].value.request, t);
    }
    bool before(int32_t a, int32_t b, int32_t heap) const {
        if (heap >= 2 && slots[a].age != slots[b].age) return slots[a].age < slots[b].age;
        return a < b;
    }
    void exchange(int32_t heap, size_t a, size_t b) {
        auto & h = heaps[heap];
        const int32_t tmp = h[a]; h[a] = h[b]; h[b] = tmp;
        slots[h[a]].position = (int32_t) a; slots[h[b]].position = (int32_t) b;
    }
    void sift_up(int32_t heap, size_t i) {
        while (i && before(heaps[heap][i], heaps[heap][(i - 1) / 2], heap)) {
            exchange(heap, i, (i - 1) / 2); i = (i - 1) / 2;
        }
    }
    void sift_down(int32_t heap, size_t i) {
        auto & h = heaps[heap];
        for (;;) {
            size_t best = i, left = i * 2 + 1, right = left + 1;
            if (left < h.size() && before(h[left], h[best], heap)) best = left;
            if (right < h.size() && before(h[right], h[best], heap)) best = right;
            if (best == i) return;
            exchange(heap, i, best); i = best;
        }
    }
    void remove(int32_t s) {
        auto & node = slots[s];
        if (node.heap < 0) return;
        const int32_t heap = node.heap;
        auto & h = heaps[heap];
        const size_t i = (size_t) node.position;
        exchange(heap, i, h.size() - 1); h.pop_back();
        node.heap = node.position = -1;
        if (i < h.size()) {
            if (i && before(h[i], h[(i - 1) / 2], heap)) sift_up(heap, i);
            else sift_down(heap, i);
        }
    }
    void insert(int32_t s, int32_t heap) {
        auto & node = slots[s];
        if (node.heap >= 0) throw std::logic_error("MoE slot present in two allocator sets");
        node.heap = heap; node.position = (int32_t) heaps[heap].size();
        heaps[heap].push_back(s); sift_up(heap, heaps[heap].size() - 1);
    }
    void make_available(int32_t expert) {
        const auto & e = entries[expert];
        if (e.value.state == phase::resident && !e.pins && !e.persistent) {
            const auto & s = slots[e.value.request.slot];
            insert(e.value.request.slot, (pf_probation && s.pf_unselected ? 6 : e.hint ? 4 : 2) + s.region);
        }
    }
    int32_t pick_pair(int32_t first, int32_t regions) const {
        int32_t chosen = -1;
        for (int32_t r = 0; r < 2; ++r) {
            if (!(regions & (1 << r)) || heaps[first + r].empty()) continue;
            const int32_t s = heaps[first + r][0];
            if (chosen < 0 || before(s, chosen, first)) chosen = s;
        }
        return chosen;
    }
    int32_t choose(origin who, bool cache_only) const {
        const int regions = cache_only ? 1 : unified ? 3 : who == origin::prefetch ? 2 : 1;
        int32_t s = pick_pair(0, regions);
        if (s >= 0) return s;
        const int32_t unused_pf = pf_probation ? pick_pair(6, regions) : -1;
        if (who == origin::demand && unused_pf >= 0) return unused_pf;
        s = pick_pair(2, regions);
        const int32_t hinted = pick_pair(4, regions);
        // Match upstream: demand protects predictions in its first eviction
        // pass; background/prewarm allocation uses the oldest eligible slot.
        if (s < 0 || (who != origin::demand && hinted >= 0 && before(hinted, s, 2))) s = hinted;
        // PF/prewarm retain pure age LRU: a new PF must not immediately evict
        // the earlier members of its own job merely because they are unselected.
        if (who != origin::demand && unused_pf >= 0 && (s < 0 || before(unused_pf, s, 2))) s = unused_pf;
        return s;
    }
    ticket evict(int32_t s) {
        remove(s);
        const ticket previous = slots[s].owner;
        if (previous) {
            auto & e = entries[previous.expert];
            if (e.value.state != phase::resident || e.pins || e.persistent) throw std::logic_error("MoE evicted a live slot");
            e.value.state = phase::absent;
        }
        slots[s].owner = {};
        slots[s].pf_unselected = false;
        return previous;
    }
    result start(int32_t expert, origin who, provenance producer, bool preserve_request, bool cache_only) {
        auto & e = entries[expert];
        if (who == origin::prefetch && prefetch_loading >= (uint32_t) prefetch_limit) {
            return {{}, e.value.request, {}, outcome::writer_limit, false};
        }
        const int32_t s = choose(who, cache_only);
        if (s < 0) return {{}, e.value.request, {}, outcome::capacity, false};
        const ticket previous = e.value.state == phase::queued ? e.value.request : ticket{};
        const bool probation_victim = pf_probation && who == origin::demand && slots[s].pf_unselected;
        const ticket victim = evict(s);
        if (probation_victim) ++probation.demand_victims;
        const uint64_t g = preserve_request ? e.value.request.generation : ++generation;
        const ticket t = {expert, s, binding, g};
        e.value = {phase::loading, t, who, producer}; e.error = {};
        slots[s].owner = t; slots[s].pf_unselected = pf_probation && who == origin::prefetch; ++loading_count;
        if (who == origin::prefetch) ++prefetch_loading;
        return {t, previous, victim, previous && !preserve_request ? outcome::promoted : outcome::created, true, probation_victim};
    }
    bool writing(ticket t) const {
        return current(t) && t.slot >= 0 && (size_t) t.slot < slots.size() &&
            entries[t.expert].value.state == phase::loading && entries[t.expert].value.request.slot == t.slot && same(slots[t.slot].owner, t);
    }

public:
    slot_manager(std::mutex & mutex, int32_t experts, int32_t count, int32_t cache, bool unified, int32_t pf_limit = -1,
                 bool pf_probation = false) :
        mutex(mutex), entries((size_t) experts), slots((size_t) count), selected((size_t) experts),
        hinted((size_t) experts), prefetch_limit(pf_limit < 0 ? count : pf_limit), cache_slots(cache), unified(unified),
        pf_probation(pf_probation) {
        if (experts <= 0 || count <= 0 || cache < 0 || cache > count) throw std::invalid_argument("invalid MoE slot-manager dimensions");
        for (size_t h = 0; h < (pf_probation ? heaps.size() : 6); ++h) heaps[h].reserve((size_t) count);
        for (int32_t s = 0; s < count; ++s) { slots[s].region = s >= cache; insert(s, slots[s].region); }
    }
    uint64_t binding_generation() const { return binding; }
    probation_counters probation_stats() const { return probation; }
    uint32_t writers() const { return loading_count; }
    uint32_t prefetch_writers() const { return prefetch_loading; }
    void set_prefetch_limit(int32_t limit) {
        if (limit < 0 || (size_t) limit > slots.size() || prefetch_loading) {
            throw std::logic_error("cannot change a live MoE prefetch writer budget");
        }
        prefetch_limit = limit;
    }
    uint32_t pins() const { return pin_count; }
    /// 供 live-union 准入统计用：`requested` 是本批并集位图（可为 null）。
    /// 只数状态，不移动任何堆/槽 —— 判定的后果仍由 acquire() 的 LRU 淘汰执行。
    occupancy occupancy_of(const uint8_t * requested) const {
        occupancy o;
        o.loading = (int32_t) loading_count;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto & s = slots[i];
            // owner 必须真的指向这一格：陈旧 owner（若出现过）不能被重复计数。
            if (!s.owner || s.owner.slot != (int32_t) i) continue;
            const auto & e = entries[s.owner.expert];
            if (e.value.state != phase::resident) continue;   // loading/queued 不占"常驻"
            ++o.resident;
            const bool used  = requested && requested[(size_t) s.owner.expert] != 0;
            const bool fixed = e.pins != 0 || e.persistent;
            if (used) continue;                               // 本批要用 -> 既不算保护也不算可淘汰
            if (fixed) ++o.pinned;
            else       ++o.evictable;
        }
        return o;
    }
    const int32_t * selected_ids() const { return selected.data(); }
    size_t selected_size() const { return selected_count; }
    observation lookup(int32_t expert) const { check_expert(expert); return entries[expert].value; }
    int32_t resident_slot(int32_t expert) const {
        const auto o = lookup(expert); return o.state == phase::resident ? o.request.slot : -1;
    }
    observation slot_owner(int32_t s) const {
        if (s < 0 || (size_t) s >= slots.size() || !slots[s].owner) return {};
        return entries[slots[s].owner.expert].value;
    }
    result enqueue(int32_t expert, provenance producer) {
        check_expert(expert); auto & e = entries[expert];
        if (e.value.state == phase::queued || e.value.state == phase::loading || e.value.state == phase::resident) {
            return {e.value.request, {}, {}, outcome::coalesced, false};
        }
        if (e.pins) return {{}, {}, {}, outcome::selected, false};
        const ticket t = {expert, -1, binding, ++generation};
        e.value = {phase::queued, t, origin::prefetch, producer}; e.error = {};
        return {t, {}, {}, outcome::created, false};
    }
    result acquire(int32_t expert, provenance producer, origin who = origin::demand, bool cache_only = false) {
        check_expert(expert); auto & e = entries[expert];
        if (e.value.state == phase::resident) return {e.value.request, {}, {}, outcome::reused, false};
        if (e.value.state == phase::loading) return {e.value.request, {}, {}, outcome::joined, false};
        return start(expert, who, producer, false, cache_only);
    }
    result start_queued(ticket t) {
        if (!current(t) || entries[t.expert].value.state != phase::queued) return {{}, t, {}, outcome::stale, false};
        const auto producer = entries[t.expert].value.producer;
        auto result = start(t.expert, origin::prefetch, producer, true, false);
        if (!result.owner) cancel_queued(t);
        return result;
    }
    bool complete(ticket t) {
        if (!writing(t)) return false;
        if (entries[t.expert].value.owner == origin::prefetch) --prefetch_loading;
        entries[t.expert].value.state = phase::resident; --loading_count;
        slots[t.slot].age = ++age; make_available(t.expert); changed.notify_all(); return true;
    }
    bool fail(ticket t, std::exception_ptr error) {
        if (!writing(t)) return false;
        if (entries[t.expert].value.owner == origin::prefetch) --prefetch_loading;
        auto & e = entries[t.expert]; e.value.state = phase::failed; e.error = error; --loading_count;
        slots[t.slot].owner = {}; slots[t.slot].pf_unselected = false;
        insert(t.slot, slots[t.slot].region); changed.notify_all(); return true;
    }
    bool cancel_queued(ticket t) {
        if (!current(t) || entries[t.expert].value.state != phase::queued) return false;
        entries[t.expert].value.state = phase::cancelled; changed.notify_all(); return true;
    }
    void wait(ticket t, std::unique_lock<std::mutex> & lock) {
        if (!lock.owns_lock() || lock.mutex() != &mutex) throw std::logic_error("MoE wait requires its layer mutex");
        changed.wait(lock, [&]() { return !current(t) || entries[t.expert].value.state != phase::loading; });
        if (current(t) && entries[t.expert].value.state == phase::resident) return;
        if (current(t) && entries[t.expert].value.state == phase::failed && entries[t.expert].error) std::rethrow_exception(entries[t.expert].error);
        throw std::runtime_error("MoE request failed, cancelled or superseded");
    }
    void pin(int32_t expert) {
        check_expert(expert); auto & e = entries[expert];
        if (!e.pins++ && e.value.state == phase::resident) remove(e.value.request.slot);
        ++pin_count;
    }
    void unpin(int32_t expert) {
        check_expert(expert); auto & e = entries[expert];
        if (!e.pins) throw std::logic_error("unbalanced MoE expert lease");
        --pin_count; if (!--e.pins) make_available(expert);
    }
    void pin_selected(const int32_t * ids, size_t count) {
        if (selected_count) throw std::logic_error("MoE selected lease was not released");
        ++selected_stamp;
        for (size_t i = 0; i < count; ++i) {
            check_expert(ids[i]); auto & e = entries[ids[i]];
            if (e.selected_stamp == selected_stamp) continue;
            e.selected_stamp = selected_stamp; selected[selected_count++] = ids[i]; pin(ids[i]);
        }
    }
    void release_selected() {
        while (selected_count) unpin(selected[--selected_count]);
    }
    void set_persistent(int32_t expert, bool value) {
        check_expert(expert); auto & e = entries[expert];
        if (e.persistent == value) return;
        if (e.value.state == phase::resident) remove(e.value.request.slot);
        e.persistent = value; make_available(expert);
    }
    bool touch(int32_t expert) {
        check_expert(expert); auto & e = entries[expert];
        if (e.value.state != phase::resident) return false;
        auto & s = slots[e.value.request.slot];
        const bool promoted = pf_probation && s.pf_unselected;
        remove(e.value.request.slot); s.pf_unselected = false; s.age = ++age; make_available(expert);
        if (promoted) ++probation.promotions;
        return promoted;
    }
    void set_hints(const std::vector<int32_t> & ids) {
        while (hinted_count) {
            const int32_t expert = hinted[--hinted_count]; auto & e = entries[expert];
            if (e.value.state == phase::resident) remove(e.value.request.slot);
            e.hint = false; make_available(expert);
        }
        for (int32_t expert : ids) {
            if (expert < 0 || (size_t) expert >= entries.size() || entries[expert].hint) continue;
            auto & e = entries[expert]; if (e.value.state == phase::resident) remove(e.value.request.slot);
            e.hint = true; hinted[hinted_count++] = expert; make_available(expert);
        }
    }
    // Cold operation only. Call after draining old queued/active producers and
    // the existing graph dependency has released all consumer leases.
    void reset_binding() {
        if (loading_count || pin_count) throw std::logic_error("cannot replace a live MoE binding");
        ++binding; hinted_count = selected_count = 0;
        for (auto & h : heaps) h.clear();
        for (auto & e : entries) { const bool persistent = e.persistent; e = {}; e.persistent = persistent; }
        for (size_t s = 0; s < slots.size(); ++s) {
            slots[s] = {}; slots[s].region = (int32_t) s >= cache_slots; insert((int32_t) s, slots[s].region);
        }
        changed.notify_all();
    }
};

} // namespace llama_moe
