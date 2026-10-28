#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

// Estimates for the tiled MiniCPM-V 4.7/Q8 DSpark/pad4320 vision bundle.
// Decimal MB/GB; this is not an allocator-enforced process limit.
struct MiniCPM47MemoryConfig {
    static constexpr int min_mb = 5200;
    static constexpr int max_mb = 13000;
    static constexpr int batch = 512;
    static constexpr int draft = 1;
    static constexpr bool phase_reuse = true;
    static constexpr int64_t slot_bytes = 40LL * 1572864;
    // Qwen35MoE in this bundle has 256 experts per layer. A wider per-layer
    // decode window cannot hold more distinct experts, so RAM above this is
    // simply wasted.
    static constexpr int max_useful_slots = 256;

    int budget_mb;
    int bank;
    int ubatch;
    int prefetch;
    int cache_slots;
    int64_t estimated_bytes;
};

inline MiniCPM47MemoryConfig minicpm47_memory_config(int budget_mb) {
    if (budget_mb < MiniCPM47MemoryConfig::min_mb || budget_mb > MiniCPM47MemoryConfig::max_mb) {
        throw std::invalid_argument("MiniCPM-V 4.7 memory budget must be 5200..13000 MB");
    }
    // Keep the graph/bank profile fixed so every additional 62.9 decimal MB
    // buys one slot per layer. Changing profiles at 5.9 GB previously made
    // the slot count fall when the user increased the slider.
    // The shared phase pool permits a larger prompt chunk without reserving
    // a second full expert bank. A short draft keeps verify in the decode
    // window at the upper budgets; draft3/8 spill into the bank and reread it.
    // Calibration includes eager vision + Q8 draft1 and phase reuse.
    const int bank = 256;
    const int ubatch = 128;
    const int prefetch = 0;
    const int64_t fixed_bytes = 4650000000LL;
    const int total_slots = int(std::min<int64_t>(
        (int64_t(budget_mb) * 1000000 - fixed_bytes) / MiniCPM47MemoryConfig::slot_bytes,
        MiniCPM47MemoryConfig::max_useful_slots));
    if (total_slots < prefetch + 8) {
        throw std::invalid_argument("MiniCPM-V 4.7 budget leaves fewer than 8 expert cache slots");
    }
    return {budget_mb, bank, ubatch, prefetch, total_slots - prefetch,
            fixed_bytes + total_slots * MiniCPM47MemoryConfig::slot_bytes};
}

// Low-VMEM variant for Hexagon v73 and older (Snapdragon 8 Gen 2 / 8 Gen 1).
// Those parts map the DSP session through a 32-bit HAP_mmap with a ~2 GiB
// single-block limit, and `ndev` is forced to 1, so the whole session must fit
// under that. The normal layout needs ~2.9 GiB (weights 1.41 G + bank 0.53 G +
// vision 0.39 G + decode window 0.40 G + KV/RS 0.29 G) and fails to map.
// This variant shrinks the expert bank and ubatch. The caller must additionally
// run vision on the CPU and disable speculative decoding (ubatch=1..2 conflicts
// with the draft-length assertion); see minicpm47_runtime.cpp.
inline MiniCPM47MemoryConfig minicpm47_low_vmem_config(int budget_mb) {
    if (budget_mb < MiniCPM47MemoryConfig::min_mb || budget_mb > MiniCPM47MemoryConfig::max_mb) {
        throw std::invalid_argument("MiniCPM-V 4.7 memory budget must be 5200..13000 MB");
    }
    // Small fixed bank: the DSP mapping is the constraint here, not host RAM.
    // bank=16 with ubatch=2 keeps the published `ubatch <= bank / 8` rule.
    const int bank = 16;
    const int ubatch = 2;
    const int prefetch = 0;
    const int cache_slots = 8;
    const int64_t fixed_bytes = 4650000000LL;
    return {budget_mb, bank, ubatch, prefetch, cache_slots - prefetch,
            fixed_bytes + (int64_t) cache_slots * MiniCPM47MemoryConfig::slot_bytes};
}
