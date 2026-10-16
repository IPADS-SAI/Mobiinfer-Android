#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <string>

// In-process CLI core: owns target/draft contexts, media history, streaming
// requests and the worker loop. No HTTP listener or external process.
class MiniCPM47Runtime {
public:
    MiniCPM47Runtime(const std::string & model, const std::string & experts, int memory_budget_mb = 6000);
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
    void resetPerformanceStats();
    std::array<double, 5> performanceStats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
