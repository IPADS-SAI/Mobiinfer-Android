#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llama_moe {

class io_admission_cancelled : public std::runtime_error {
public:
    io_admission_cancelled() : std::runtime_error("expert IO admission closed") {}
};

// Independent of slot/queue locks: no method calls an IO function or takes a
// layer/queue mutex. Callers must release those locks before entering this API.
class io_admission {
public:
    struct key {
        int32_t layer = -1, expert = -1;
        uint64_t binding = 0, generation = 0;
    };
    enum class origin { demand, prefetch };
    enum class priority { background, demand, needed_prefetch };
    enum class action { wait, grant, release, cancel, boost, needed_begin, needed_end };
    struct observation {
        action event;
        key request;
        origin source;
        priority category;
        uint64_t sequence;
        uint32_t active, waiting, capacity;
        const void * payload;
    };
    // Diagnostic callback only, under the admission mutex. It must be noexcept,
    // nonblocking, and must not call this gate or any layer/queue API.
    using observer = void (*)(void *, const observation &) noexcept;
    struct statistics {
        uint64_t grants = 0, releases = 0, cancelled = 0, boosted = 0;
        uint64_t demand_grants = 0, needed_grants = 0, background_grants = 0;
        uint32_t active = 0, waiting = 0, peak = 0;
        bool closed = false;
    };
private:
    struct waiter {
        key request;
        origin source;
        priority category;
        uint64_t sequence;
        const void * payload;
        waiter * previous = nullptr, * next = nullptr;
        bool foreground = false, queued = false, granted = false, cancelled = false;
    };
    struct fifo { waiter * first = nullptr, * last = nullptr; };
    std::mutex mutex;
    std::condition_variable changed;
    const uint32_t limit;
    observer observe;
    void * observer_context;
    statistics stats;
    uint64_t next_sequence = 0;
    uint32_t foreground_streak = 0;
    fifo foreground, background;
    std::vector<uint64_t> needed_generations;
    std::vector<int32_t> needed_ids;
    size_t needed_count = 0;
    int32_t needed_layer = -1;
    uint64_t needed_binding = 0;
    bool needed_active = false;
    const void * needed_payload = nullptr;

    void emit(action a, const waiter & w) noexcept {
        if (observe) observe(observer_context, {a, w.request, w.source, w.category, w.sequence,
            stats.active, stats.waiting, limit, w.payload});
    }
    void emit_scope(action a) noexcept {
        if (observe) observe(observer_context, {a, {needed_layer, -1, needed_binding, 0}, origin::demand,
            priority::demand, 0, stats.active, stats.waiting, limit, needed_payload});
    }
    bool needed(key request) const {
        return needed_active && request.layer == needed_layer && request.binding == needed_binding &&
            request.expert >= 0 && (size_t) request.expert < needed_generations.size() &&
            request.generation != 0 && needed_generations[(size_t) request.expert] == request.generation;
    }
    static void remove(fifo & queue, waiter & w) {
        if (w.previous) w.previous->next = w.next; else queue.first = w.next;
        if (w.next) w.next->previous = w.previous; else queue.last = w.previous;
        w.previous = w.next = nullptr; w.queued = false;
    }
    static void append(fifo & queue, waiter & w) {
        w.previous = queue.last; w.next = nullptr; w.queued = true;
        if (queue.last) queue.last->next = &w; else queue.first = &w;
        queue.last = &w;
    }
    // Promotions/demotions preserve original FIFO age. The walk is bounded by
    // waiting worker threads, never the number of experts or queued jobs.
    static void ordered_insert(fifo & queue, waiter & w) {
        waiter * at = queue.last;
        while (at && at->sequence > w.sequence) at = at->previous;
        if (!at) {
            w.previous = nullptr; w.next = queue.first;
            if (queue.first) queue.first->previous = &w; else queue.last = &w;
            queue.first = &w;
        } else {
            w.previous = at; w.next = at->next;
            if (at->next) at->next->previous = &w; else queue.last = &w;
            at->next = &w;
        }
        w.queued = true;
    }
    void dispatch() noexcept {
        while (!stats.closed && stats.active < limit && (foreground.first || background.first)) {
            // A background waiter gets one grant after at most `limit`
            // foreground grants while both classes have pending work.
            const bool high = foreground.first && (!background.first || foreground_streak < limit);
            fifo & queue = high ? foreground : background;
            waiter & w = *queue.first;
            if (high && background.first) ++foreground_streak; else foreground_streak = 0;
            remove(queue, w); --stats.waiting;
            w.granted = true; ++stats.active; ++stats.grants;
            stats.peak = std::max(stats.peak, stats.active);
            if (w.category == priority::demand) ++stats.demand_grants;
            else if (w.category == priority::needed_prefetch) ++stats.needed_grants;
            else ++stats.background_grants;
            emit(action::grant, w);
        }
        changed.notify_all();
    }
    void release(key request, origin source, priority category, uint64_t sequence, const void * payload) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        --stats.active; ++stats.releases;
        const waiter w{request, source, category, sequence, payload};
        emit(action::release, w); dispatch();
    }
    void end_needed() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        emit_scope(action::needed_end);
        for (waiter * w = foreground.first; w;) {
            waiter * next = w->next;
            if (w->source == origin::prefetch) {
                remove(foreground, *w); w->foreground = false; w->category = priority::background;
                ordered_insert(background, *w);
            }
            w = next;
        }
        while (needed_count) needed_generations[(size_t) needed_ids[--needed_count]] = 0;
        needed_active = false; needed_layer = -1; needed_binding = 0; needed_payload = nullptr;
        dispatch();
    }
public:
    class permit {
        friend class io_admission;
        io_admission * gate = nullptr;
        key request;
        origin source = origin::demand;
        priority category = priority::demand;
        uint64_t sequence = 0;
        const void * payload = nullptr;
        permit(io_admission & gate, const waiter & w) noexcept : gate(&gate), request(w.request),
            source(w.source), category(w.category), sequence(w.sequence), payload(w.payload) {}
    public:
        permit() = default;
        permit(const permit &) = delete;
        permit & operator=(const permit &) = delete;
        permit(permit && other) noexcept : gate(std::exchange(other.gate, nullptr)), request(other.request),
            source(other.source), category(other.category), sequence(other.sequence), payload(other.payload) {}
        permit & operator=(permit && other) noexcept {
            if (this != &other) { reset(); gate = std::exchange(other.gate, nullptr); request = other.request;
                source = other.source; category = other.category; sequence = other.sequence; payload = other.payload; }
            return *this;
        }
        ~permit() { reset(); }
        void reset() noexcept {
            if (gate) { auto * owner = std::exchange(gate, nullptr); owner->release(request, source, category, sequence, payload); }
        }
    };
    class needed_scope {
        friend class io_admission;
        io_admission * gate = nullptr;
        explicit needed_scope(io_admission & gate) noexcept : gate(&gate) {}
    public:
        needed_scope() = default;
        needed_scope(const needed_scope &) = delete;
        needed_scope & operator=(const needed_scope &) = delete;
        needed_scope(needed_scope && other) noexcept : gate(std::exchange(other.gate, nullptr)) {}
        needed_scope & operator=(needed_scope && other) noexcept {
            if (this != &other) { reset(); gate = std::exchange(other.gate, nullptr); } return *this;
        }
        ~needed_scope() { reset(); }
        void reset() noexcept { if (gate) std::exchange(gate, nullptr)->end_needed(); }
    };

    io_admission(uint32_t capacity, size_t experts, observer callback = nullptr, void * context = nullptr) :
        limit(capacity), observe(callback), observer_context(context), needed_generations(experts, 0), needed_ids(experts) {
        if (!capacity || capacity > 64 || !experts) throw std::invalid_argument("invalid expert IO admission dimensions");
    }
    io_admission(const io_admission &) = delete;
    io_admission & operator=(const io_admission &) = delete;
    permit acquire(key request, origin source, const void * payload = nullptr) {
        std::unique_lock<std::mutex> lock(mutex);
        const priority category = source == origin::demand ? priority::demand :
            needed(request) ? priority::needed_prefetch : priority::background;
        waiter w{request, source, category, ++next_sequence, payload};
        w.foreground = category != priority::background;
        append(w.foreground ? foreground : background, w); ++stats.waiting; emit(action::wait, w);
        if (stats.closed) {
            remove(w.foreground ? foreground : background, w); --stats.waiting;
            ++stats.cancelled; emit(action::cancel, w); throw io_admission_cancelled();
        }
        dispatch();
        try { changed.wait(lock, [&] { return w.granted || w.cancelled; }); }
        catch (...) {
            if (w.queued) { remove(w.foreground ? foreground : background, w); --stats.waiting; }
            if (w.granted) { --stats.active; ++stats.releases; emit(action::release, w); }
            else if (!w.cancelled) { ++stats.cancelled; emit(action::cancel, w); }
            dispatch(); throw;
        }
        if (w.cancelled) throw io_admission_cancelled();
        return permit(*this, w);
    }
    needed_scope begin_needed(const key * requests, size_t count, const void * payload = nullptr) {
        std::lock_guard<std::mutex> lock(mutex);
        if (stats.closed) throw io_admission_cancelled();
        if (needed_active || count > needed_generations.size()) throw std::logic_error("overlapping expert IO needed scopes");
        // Validate before publishing any state, so a bad key cannot leave an
        // active scope without its RAII owner.
        for (size_t i = 0; i < count; ++i) {
            if (requests[i].expert < 0 || (size_t) requests[i].expert >= needed_generations.size() ||
                !requests[i].generation || requests[i].layer != requests[0].layer || requests[i].binding != requests[0].binding) {
                throw std::invalid_argument("invalid expert IO needed ticket");
            }
        }
        needed_active = true; needed_payload = payload;
        if (count) { needed_layer = requests[0].layer; needed_binding = requests[0].binding; }
        for (size_t i = 0; i < count; ++i) {
            const auto & request = requests[i];
            if (!needed_generations[(size_t) request.expert]) needed_ids[needed_count++] = request.expert;
            needed_generations[(size_t) request.expert] = request.generation;
        }
        emit_scope(action::needed_begin);
        for (waiter * w = background.first; w;) {
            waiter * next = w->next;
            if (needed(w->request)) {
                remove(background, *w); w->foreground = true; w->category = priority::needed_prefetch;
                ordered_insert(foreground, *w); ++stats.boosted; emit(action::boost, *w);
            }
            w = next;
        }
        dispatch(); return needed_scope(*this);
    }
    // Normal shutdown calls this only after producer joins. Exceptional abort
    // may close first: waiting owners wake and fail their tickets outside I.
    void close() noexcept {
        std::lock_guard<std::mutex> lock(mutex); stats.closed = true;
        for (fifo * queue : {&foreground, &background}) while (queue->first) {
            waiter & w = *queue->first; remove(*queue, w); --stats.waiting;
            w.cancelled = true; ++stats.cancelled; emit(action::cancel, w);
        }
        changed.notify_all();
    }
    statistics snapshot() { std::lock_guard<std::mutex> lock(mutex); return stats; }
};

} // namespace llama_moe
