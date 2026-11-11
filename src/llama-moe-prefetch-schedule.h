#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llama_moe {

// Caller holds the prefetch queue mutex. Plans outlive their worker jobs: an
// early job can finish before the adjacent layer supplies its refined routing.
class prefetch_schedule {
public:
    static constexpr int32_t max_reads = 16;
    struct plan {
        uint64_t epoch = 0;
        int32_t limit = 0;
        int32_t spent = 0;
        bool refined = false;
        std::array<int32_t, max_reads> experts{};
    };

private:
    std::vector<plan> plans;

public:
    void resize(size_t layers) { plans.resize(layers); }
    bool begin(int32_t layer, uint64_t epoch, int32_t limit) {
        if (!epoch || limit < 0 || limit > max_reads) throw std::invalid_argument("invalid prefetch plan");
        auto & p = plans.at((size_t) layer);
        if (p.epoch == epoch) return false;
        p = {}; p.epoch = epoch; p.limit = limit;
        return true;
    }
    const plan * find(int32_t layer, uint64_t epoch) const {
        const auto & p = plans.at((size_t) layer);
        return p.epoch == epoch ? &p : nullptr;
    }
    bool refine(int32_t layer, uint64_t epoch) {
        auto & p = plans.at((size_t) layer);
        if (p.epoch != epoch || p.refined) return false;
        p.refined = true; return true;
    }
    int32_t allowance(int32_t layer, uint64_t epoch, bool near) const {
        const auto * p = find(layer, epoch);
        if (!p) return 0;
        const int32_t limit = near ? p->limit : p->limit - std::min(2, p->limit);
        return std::max(0, limit - p->spent);
    }
    void created(int32_t layer, uint64_t epoch, int32_t expert) {
        auto & p = plans.at((size_t) layer);
        if (p.epoch != epoch || p.spent >= p.limit ||
            std::find(p.experts.begin(), p.experts.begin() + p.spent, expert) != p.experts.begin() + p.spent) {
            throw std::logic_error("prefetch plan exceeds its distinct-request budget");
        }
        p.experts[(size_t) p.spent++] = expert;
    }
    bool refund(int32_t layer, uint64_t epoch, int32_t expert) {
        auto & p = plans.at((size_t) layer);
        if (p.epoch != epoch) return false;
        auto end = p.experts.begin() + p.spent;
        const auto it = std::find(p.experts.begin(), end, expert);
        if (it == end) return false;
        std::move(it + 1, end, it); --p.spent; return true;
    }
    static bool earlier(uint64_t epoch_a, int32_t layer_a, uint64_t epoch_b, int32_t layer_b) {
        return epoch_a < epoch_b || (epoch_a == epoch_b && layer_a < layer_b);
    }
    static bool due(uint64_t epoch, int32_t layer, uint64_t consumer_epoch, int32_t consumer_layer) {
        return !earlier(consumer_epoch, consumer_layer, epoch, layer);
    }
};

} // namespace llama_moe
