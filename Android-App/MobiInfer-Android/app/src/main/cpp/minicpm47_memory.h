#pragma once

#include <cstdint>
#include <stdexcept>

// Estimates for the tiled MiniCPM-V 4.7/Q8 DSpark/pad4320 vision bundle.
// Decimal MB/GB; this is not an allocator-enforced process limit.
struct MiniCPM47MemoryConfig {
    static constexpr int min_mb = 5200;
    static constexpr int max_mb = 8000;
    static constexpr int batch = 512;
    static constexpr int draft = 3;
    static constexpr int64_t slot_bytes = 40LL * 1572864;

    int budget_mb;
    int bank;
    int ubatch;
    int prefetch;
    int cache_slots;
    int64_t estimated_bytes;
};

inline MiniCPM47MemoryConfig minicpm47_memory_config(int budget_mb) {
    if (budget_mb < MiniCPM47MemoryConfig::min_mb || budget_mb > MiniCPM47MemoryConfig::max_mb) {
        throw std::invalid_argument("MiniCPM-V 4.7 memory budget must be 5200..8000 MB");
    }
    // Smaller budgets must also reduce the prefill bank and graph reservation.
    // Fixed terms include the resident model, draft, eager vision, context and
    // a reserve for UI/variation. A different bundle needs new calibration.
    const bool low = budget_mb < 5900;
    const bool medium = budget_mb < 6500;
    const int bank = low ? 32 : medium ? 128 : 256;
    const int ubatch = low ? 4 : medium ? 16 : 128;
    const int prefetch = low ? 0 : 4;
    const int64_t fixed_bytes = low ? 4650000000LL : medium ? 5050000000LL : 5700000000LL;
    const int total_slots = int((int64_t(budget_mb) * 1000000 - fixed_bytes)
                               / MiniCPM47MemoryConfig::slot_bytes);
    if (total_slots < prefetch + 8) {
        throw std::invalid_argument("MiniCPM-V 4.7 budget leaves fewer than 8 expert cache slots");
    }
    return {budget_mb, bank, ubatch, prefetch, total_slots - prefetch,
            fixed_bytes + total_slots * MiniCPM47MemoryConfig::slot_bytes};
}
