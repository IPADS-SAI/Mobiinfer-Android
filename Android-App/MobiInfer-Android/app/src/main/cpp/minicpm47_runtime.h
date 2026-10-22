#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <vector>

// One restored conversation turn: role is "system" / "user" / "assistant";
// images are raw encoded bytes (png/jpeg) attached to a user turn, in the
// order their media markers appear in text.
struct RestoreTurn {
    std::string role;
    std::string text;
    std::vector<std::vector<uint8_t>> images;
};

// In-process CLI core: owns target/draft contexts, media history, streaming
// requests and the worker loop. No HTTP listener or external process.
class MiniCPM47Runtime {
public:
    MiniCPM47Runtime(const std::string & model, const std::string & experts, int memory_budget_mb = 6000,
                     int context_size = 4096, int image_min_tokens = -1, int image_max_tokens = -1);
    ~MiniCPM47Runtime();
    void setVisionPath(const std::string & path);
    bool start();
    void setThinking(bool enabled);
    void setSystemPrompt(const std::string & text);
    void addImage(const uint8_t * data, size_t size);
    void submit(const std::string & text, int n_predict);
    std::string next(bool & done);
    void cancel();
    void reset();
    // Replace the in-memory chat history (used when the user switches back to
    // a saved conversation). Does not prefill immediately: the KV prefix is
    // rebuilt by the next submit(), which the flag below forces to start from
    // scratch so the restored turns are actually evaluated.
    void restore(const std::vector<RestoreTurn> & turns);
    void resetPerformanceStats();
    std::array<double, 5> performanceStats() const;
    bool contextTruncated() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
