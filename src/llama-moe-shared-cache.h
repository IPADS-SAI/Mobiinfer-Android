#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llama_moe {

// Demand-only expert cache for `LLAMA_MOE_SHARED_DECODE=1`: one fixed bank of slots shared
// by every routed layer, evicted ClassLRU-style (8 log2 frequency classes, 128-step half
// life). It is built for **single-row decode**: process() only consults it when the batch
// has exactly one row.
//
// Speculative verification (draft n_max > 0) turns every decode step into a multi-row
// batch. Those batches use the plain bank path, which clears this whole map first because
// they share the same storage -- so with speculation enabled the cache is cleared every
// step and its benefit is gone. That stays correct but slow; the runtime logs it once.
// Do not widen this class to multi-row batches without reworking that shared-storage
// invariant.
//
// One LRU list per frequency class, shared by all participating layers.
// The caller serializes plan, fill, and graph execution.
class shared_slot_cache {
    static constexpr int class_count = 8;
    static constexpr float half_life = 128.0f;

    struct expert_entry {
        int64_t last_use = -1;
        float frequency = 0.0f;
        int32_t slot = -1;
    };

    struct slot_entry {
        int32_t key = -1;
        int32_t prev = -1;
        int32_t next = -1;
        int32_t frequency_class = 0;
    };

    struct class_list {
        int32_t head = -1;
        int32_t tail = -1;
    };

    const int32_t expert_count;
    std::vector<expert_entry> entries;
    std::vector<int64_t> clocks;
    std::vector<slot_entry> slots;
    std::array<class_list, class_count> classes;

    static std::size_t entry_count(int32_t layers, int32_t experts) {
        const int64_t count = int64_t(layers) * experts;
        if (layers <= 0 || experts <= 0 || count > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("invalid MoE shared cache dimensions");
        }
        return count;
    }

    static int32_t checked_capacity(int32_t capacity) {
        if (capacity <= 0) {
            throw std::invalid_argument("invalid MoE shared cache capacity");
        }
        return capacity;
    }

    int32_t key(int32_t layer, int32_t expert) const {
        assert(layer >= 0 && std::size_t(layer) < clocks.size());
        assert(expert >= 0 && expert < expert_count);
        return layer * expert_count + expert;
    }

    static int class_of(const expert_entry & entry) {
        assert(entry.frequency >= 0.0f);
        // claim() 可能在新条目还没被 touch() 时就调用本函数（frequency == 0）。NDEBUG 下
        // assert 不生效，log2(0) = -inf 转 int 是 UB（classes[INT_MIN] 越界）。夹到 1 即可：
        // 频率 0 和 1 本来就属于 0 档。
        return std::min(class_count - 1, int(std::log2(std::max(1.0f, entry.frequency))));
    }

    void promote(int32_t index, int frequency_class) {
        auto & entry = slots[index];
        auto & old_list = classes[entry.frequency_class];
        auto & new_list = classes[frequency_class];
        if (entry.frequency_class == frequency_class && old_list.tail == index) {
            return;
        }

        if (entry.prev >= 0) {
            slots[entry.prev].next = entry.next;
        } else {
            old_list.head = entry.next;
        }
        if (entry.next >= 0) {
            slots[entry.next].prev = entry.prev;
        } else {
            old_list.tail = entry.prev;
        }

        entry.prev = new_list.tail;
        entry.next = -1;
        entry.frequency_class = frequency_class;
        if (new_list.tail >= 0) {
            slots[new_list.tail].next = index;
        } else {
            new_list.head = index;
        }
        new_list.tail = index;
    }

    int32_t victim(int32_t layer, const std::vector<uint8_t> & selected) const {
        int32_t best = -1;
        double best_score = -1.0;
        for (int c = 0; c < class_count; ++c) {
            for (int32_t index = classes[c].head; index >= 0; index = slots[index].next) {
                const int32_t owner = slots[index].key;
                if (owner < 0) {
                    return index;
                }
                const int32_t owner_layer = owner / expert_count;
                if (owner_layer == layer && selected[owner % expert_count]) {
                    continue;
                }
                const int64_t age = clocks[owner_layer] - entries[owner].last_use;
                const double score = std::ldexp(double(age), -c); // age / 2^class
                if (score > best_score) {
                    best = index;
                    best_score = score;
                }
                break;
            }
        }
        if (best < 0) {
            throw std::runtime_error("no MoE shared cache victim");
        }
        return best;
    }

public:
    shared_slot_cache(int32_t layers, int32_t experts, int32_t capacity)
        : expert_count(experts),
          entries(entry_count(layers, experts)),
          clocks(layers),
          slots(checked_capacity(capacity)) {
        clear();
    }

    void clear() noexcept {
        std::fill(entries.begin(), entries.end(), expert_entry{});
        std::fill(clocks.begin(), clocks.end(), 0);
        classes.fill({});

        const int32_t capacity = static_cast<int32_t>(slots.size());
        classes[0] = { 0, capacity - 1 };
        for (int32_t index = 0; index < capacity; ++index) {
            slots[index] = { -1, index - 1, index + 1 < capacity ? index + 1 : -1, 0 };
        }
    }

    int32_t slot(int32_t layer, int32_t expert) const {
        return entries[key(layer, expert)].slot;
    }

    // Record each selected expert once, including misses, before claiming slots.
    void touch(int32_t layer, int32_t expert) {
        auto & entry = entries[key(layer, expert)];
        const int64_t age = clocks[layer] - entry.last_use;
        entry.frequency = entry.frequency * std::exp2(-float(age) / half_life) + 1.0f;
        entry.last_use = clocks[layer];
        if (entry.slot >= 0) {
            promote(entry.slot, class_of(entry));
        }
    }

    // The mask protects all experts selected in this layer, including those already claimed.
    int32_t claim(int32_t layer, int32_t expert, const std::vector<uint8_t> & selected) {
        const int32_t expert_key = key(layer, expert);
        auto & entry = entries[expert_key];
        assert(selected.size() == std::size_t(expert_count) && selected[expert]);
        if (entry.slot >= 0) {
            return entry.slot;
        }

        const int32_t index = victim(layer, selected);
        auto & owner = slots[index].key;
        if (owner >= 0) {
            entries[owner].slot = -1;
        }
        owner = expert_key;
        entry.slot = index;
        promote(index, class_of(entry));
        return index;
    }

    void advance(int32_t layer) {
        assert(layer >= 0 && std::size_t(layer) < clocks.size());
        ++clocks[layer];
    }
};

} // namespace llama_moe
