#pragma once

#include <cstdint>
#include <stdexcept>

// Calibrated for the tiled MiniCPM-V 4.7/Q8 DSpark/pad4320 vision bundle.
// Decimal MB/GB, not MiB/GiB. See Android-App/HEXAGON_RUNTIME.md.
struct MiniCPM47MemoryConfig {
    static constexpr int min_mb = 5500;
    static constexpr int max_mb = 8000;
    static constexpr int bank = 256; // one shared prefill bank per HTP device
    static constexpr int batch = 128;
    static constexpr int draft = 3;
    static constexpr int prefetch = 4;
    // Actual packaged App libraries have higher fixed memory than standalone CLI.
    // Reserve ~200 MB beyond the observed native multi-request upper bound.
    static constexpr int64_t fixed_bytes = 4740000000LL;
    static constexpr int64_t slot_bytes = 40LL * 1572864; // all 40 layers

    int budget_mb;
    int cache_slots;
    int64_t estimated_bytes;
};

inline MiniCPM47MemoryConfig minicpm47_memory_config(int budget_mb) {
    if (budget_mb < MiniCPM47MemoryConfig::min_mb || budget_mb > MiniCPM47MemoryConfig::max_mb) {
        throw std::invalid_argument("MiniCPM-V 4.7 memory budget must be 5500..8000 MB");
    }
    // Approximate budget: round to the nearest whole layer-wide slot (8 GB -> 52).
    // This is not an allocator-enforced limit.
    const int total_slots = int((int64_t(budget_mb) * 1000000 - MiniCPM47MemoryConfig::fixed_bytes
                                + MiniCPM47MemoryConfig::slot_bytes / 2)
                               / MiniCPM47MemoryConfig::slot_bytes);
    return {budget_mb, total_slots - MiniCPM47MemoryConfig::prefetch,
            MiniCPM47MemoryConfig::fixed_bytes + total_slots * MiniCPM47MemoryConfig::slot_bytes};
}
