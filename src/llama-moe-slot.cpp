#include "llama-moe-slot.h"
#include "llama-moe-slot-manager.h"
#include "llama-moe-io-admission.h"
#include "llama-moe-prefetch-schedule.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-experts.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <new>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/mman.h>
#endif
#endif

namespace {
// Parallel pread threads per layer wave.
//
// This path is bandwidth-bound (measured: on-device decode time tracks total bytes
// read, not critical-path bytes), so whether the storage can serve more concurrent
// reads decides whether raising this helps at all. It is a tunable rather than a
// constant so it can be swept on device without a rebuild -- set
// LLAMA_MOE_IO_THREADS. Defaults to 4, the value all earlier measurements used.
constexpr int kMoeSsdIoThreadsDefault = 4;

static int moe_ssd_io_threads() {
    static const int n = []() {
        const char * s = std::getenv("LLAMA_MOE_IO_THREADS");
        const int v = s ? std::atoi(s) : 0;
        return (v > 0 && v <= 64) ? v : kMoeSsdIoThreadsDefault;
    }();
    return n;
}

static bool slot_manager_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LLAMA_MOE_EXPERT_READY");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static bool pf_probation_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LLAMA_MOE_PF_PROBATION");
        if (!value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") != 0) throw std::runtime_error("LLAMA_MOE_PF_PROBATION must be 0 or 1");
        return true;
    }();
    return enabled;
}

static bool h1_front_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LLAMA_MOE_PREFETCH_H1_FRONT");
        if (!value || !*value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") != 0) throw std::runtime_error("LLAMA_MOE_PREFETCH_H1_FRONT must be 0 or 1");
        return true;
    }();
    return enabled;
}

static bool prefetch_refresh_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LLAMA_MOE_PREFETCH_REFRESH");
        if (!value || !*value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") != 0) throw std::runtime_error("LLAMA_MOE_PREFETCH_REFRESH must be 0 or 1");
        return true;
    }();
    return enabled;
}

// Parse once, before any candidate IO is submitted. Disabled creates no gate.
static uint32_t expert_io_budget() {
    static const uint32_t value = []() -> uint32_t {
        const char * text = std::getenv("LLAMA_MOE_EXPERT_IO_BUDGET");
        if (!text || !*text) return 0;
        if (*text < '0' || *text > '9') throw std::runtime_error("LLAMA_MOE_EXPERT_IO_BUDGET must be an integer in 0..64");
        char * end = nullptr; errno = 0;
        const unsigned long parsed = std::strtoul(text, &end, 10);
        if (errno || *end || parsed > 64) throw std::runtime_error("LLAMA_MOE_EXPERT_IO_BUDGET must be an integer in 0..64");
        return (uint32_t) parsed;
    }();
    return value;
}

// O_DIRECT：同一个文件、同样 1.5 MiB 随机专家，缓冲读 1.9 GiB/s、O_DIRECT 3.6 GiB/s
// （tools/moe/bench/expert_read_bench.c，见 exps/roofline-saturate-20261105 §4.45）。
// 差 1.87× 的原因是缓冲路径的 readahead 把物理字节放大了 1.88×（phys_read 72192 vs 38400 KiB）。
// 生产路径的 iovec 指向 slot 张量里 gate|up|down 三段，base 不保证 4 KiB 对齐，
// 所以走「对齐 scratch + memcpy」：多一次 1.5 MiB 拷贝，换设备少读 40% 的字节。
// CPU 占用只有 26%（6 核里 1.6 个），这次拷贝是免费的。
static std::atomic<bool> g_o_direct_ok{false};

/// LLAMA_MOE_O_DIRECT 的两种取值：
///   未设 / "0"  → 全程缓冲读（旧行为）
///   "1"         → 全程 O_DIRECT（实测最好，见 exps/roofline §4.48）
///
/// 历史上有过第三档 `:auto`（按阶段切读法：prefill 走 O_DIRECT、decode 走缓冲），
/// 它需要两套 fd 表 + 一个"当前阶段"的全局量，而那个全局量只在 process() 里更新，
/// 预读 worker 从第一个图就开始读 → 行为取决于启动顺序，不可复现。
/// **§3.10 实测 `:auto` 两个阶段都输给全开 O_DIRECT**，所以这一档删掉了：
/// 不是为了修 bug（它语义本身是对的），而是维护成本不值。
/// 老脚本里传 "auto" 会被当成 "1" 并打一次警告。
static int o_direct_mode() {
    static const int m = [] {
        const char * v = std::getenv("LLAMA_MOE_O_DIRECT");
        // 默认**开**：这一步只换读法，不多占一字节内存，且开机先跑一次真实对齐读
        // 探测（probe_o_direct），探不过就整程退回缓冲读。回退用 LLAMA_MOE_O_DIRECT=0。
        if (!v || !*v) {
            return 1;
        }
        if (std::strcmp(v, "0") == 0) {
            return 0;
        }
        if (std::strcmp(v, "auto") == 0) {
            fprintf(stderr, "[moe-ssd] O_DIRECT: \"auto\" 已删除（实测输给全开），按 1 处理\n");
        }
        return 1;
    }();
    return m;
}

static bool moe_o_direct_want() {
    return o_direct_mode() != 0;
}

static bool moe_o_direct() {
    return g_o_direct_ok.load(std::memory_order_relaxed) && o_direct_mode() == 1;
}


// 跨层流水的 horizon：一次提交几层的 gate 预测。
// 1 = 旧行为（只预测下一层）；默认 2；上限 8 —— 再远用 h_il 近似 h_{il+k} 精度会崩，
// 而且预测本身要读 `layers[il+k].ffn_gate_inp` 的全部权重（d_model × 256，很小）。
static int moe_lookahead_horizon() {
    static const int h = []() {
        const char * v = std::getenv("LLAMA_MOE_LOOKAHEAD_HORIZON");
        // **默认 2**：同一份二进制、只改本 env 的对照里 h2 优于 h1 与 h4
        // （数值与命令见 exps/pipe-in-layer-20261105/README.md §3.33；
        //  这里不复制具体 t/s，避免与文档里的口径漂移）。
        // 注意尾部若干层凑不满 horizon 块（il+k 越界），运行期会按实际块数夹紧并计数。
        const int n = v ? std::atoi(v) : 2;
        return std::max(1, std::min(n, 8));
    }();
    return h;
}


// Optional persistent helpers avoid pthread creation for every MoE layer.
// Each caller owns a pool; prefetch never competes with main-path jobs here.
class moe_io_pool {
    std::mutex call_mutex, mutex;
    std::condition_variable go, done;
    std::vector<std::thread> threads;
    std::function<void()> job;
    uint64_t generation = 0;
    int active = 0, remaining = 0;
    bool quit = false;
    bool unavailable = false;

    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); quit = true; }
        go.notify_all();
        for (auto & thread : threads) if (thread.joinable()) thread.join();
    }
public:
    ~moe_io_pool() { stop(); }
    bool run(int count, const std::function<void()> & fn) {
        const char * enabled = std::getenv("LLAMA_MOE_IO_POOL");
        if (!enabled || *enabled != '1') return false;
        std::lock_guard<std::mutex> caller(call_mutex);
        if (unavailable) return false;
        if (threads.empty()) {
            try {
                for (int id = 0; id < moe_ssd_io_threads(); ++id) {
                    threads.emplace_back([this, id]() {
                        uint64_t seen = 0;
                        std::unique_lock<std::mutex> lock(mutex);
                        for (;;) {
                            go.wait(lock, [&]() { return quit || seen != generation; });
                            if (quit) return;
                            seen = generation;
                            if (id >= active) continue;
                            auto fn = job;
                            lock.unlock();
                            fn();
                            lock.lock();
                            if (--remaining == 0) done.notify_one();
                        }
                    });
                }
            } catch (...) { stop(); threads.clear(); unavailable = true; return false; }
        }
        std::unique_lock<std::mutex> lock(mutex);
        active = remaining = count;
        job = fn;
        ++generation;
        go.notify_all();
        done.wait(lock, [&]() { return remaining == 0; });
        job = nullptr;
        return true;
    }
};

// slot_to_expert 的哨兵值。
//   kSlotFree    -1：空位，可分配
//   kSlotPending -2：已被预读 worker **认领**、内容尚未写完。主路径必须跳过它
//                    （既不能覆盖，也不能当成有效缓存条目）。
constexpr int32_t kSlotFree    = -1;
constexpr int32_t kSlotPending = -2;

// One I/O wave per layer. A layer's unique set never exceeds n_slots (<= 256), so this is
// always a single wave: the wave loop only adds a join + thread respawn per boundary, and
// the disk idles across it. Measured 2026-09-12 (paired A/B, 8/8 reps positive): 13 waves ->
// 1 wave is worth +3.65% prefill (261.9 -> 271.2 t/s), at zero memory cost.
// Kept as a named constant (rather than deleting the loop) so the cap stays explicit.
constexpr int kMoeSsdIoChunk = 256;

static int64_t now_ns() {
#ifndef _WIN32
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

// Xcode 能抓住 stderr（和 [MBMtmd] 一样）。llama-cli 的 callback 会吞掉纯 LLAMA_LOG_INFO。
static void moe_log(const char * fmt, ...) {
    char buf[768];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[moe-ssd] %s\n", buf);
    fflush(stderr);
    LLAMA_LOG_INFO("[moe-ssd] %s\n", buf);
}

struct moe_handoff_context {
    uint64_t job_id;
    uint64_t epoch;
    int32_t layer;
    int32_t rows;
    const char * source;
    uint64_t generation = 0;
    const char * domain = "unknown";
    uint64_t binding = 0;
    uint64_t io_id = 0; // diagnostic identity; independent of request generations
};

struct moe_handoff_detail {
    const char * reason = "none";
    int32_t related_expert = -1, related_slot = -1;
    uint64_t related_generation = 0;
    int32_t scanned = 0, capacity_slots = 0, free_slots = 0, loading_slots = 0;
    int32_t protected_slots = 0, pinned_slots = 0, eligible_slots = 0;
    uint64_t file_offset = 0, requested_pages = 0;
    int64_t resident_pages = -1, probe_begin_ns = 0, probe_end_ns = 0;
    int32_t probe_errno = 0;
    uint64_t admit_seq = 0;
    const char * admit_class = "none";
    uint32_t admit_active = 0, admit_waiters = 0, admit_capacity = 0;
};

static moe_handoff_context handoff_request(moe_handoff_context context, llama_moe::slot_manager::ticket ticket) {
    context.generation = ticket.generation;
    context.binding = ticket.binding;
    return context;
}

// Diagnostic only: each caller owns its buffer. Producers reserve distinct POD
// entries; the caller flushes only after its prefetch worker has been joined.
class moe_handoff_trace {
    struct event {
        int64_t ns;
        uint64_t tid;
        const char * name;
        moe_handoff_context context;
        moe_handoff_context active;
        int32_t expert;
        int32_t slot;
        int32_t result;
        uint64_t bytes;
        moe_handoff_detail detail;
    };
    static constexpr uint64_t default_capacity = 262144;
    uint64_t capacity = default_capacity;
    std::unique_ptr<event[]> events;
    std::atomic<uint64_t> next{0}, next_io{0};
    std::atomic<uint64_t> lost{0};
    FILE * file = nullptr;
    std::string path;

public:
    moe_handoff_trace() {
        const char * value = std::getenv("LLAMA_MOE_HANDOFF_TRACE");
        if (!value || !*value) return;
        if (const char * count = std::getenv("LLAMA_MOE_HANDOFF_TRACE_CAPACITY")) {
            char * end = nullptr;
            const unsigned long long requested = std::strtoull(count, &end, 10);
            if (*count && end && !*end && requested >= 1024 && requested <= 1048576) capacity = requested;
            else moe_log("handoff-trace: invalid capacity %s; using %llu", count, (unsigned long long) capacity);
        }
        path = value;
        file = std::fopen(path.c_str(), "wx");
        if (!file && errno == EEXIST) {
            // Multiple MoE callers must never share a live buffer or overwrite
            // each other's trace. Keep the requested path for the first caller.
            path += "." + std::to_string((uintptr_t) this) + ".jsonl";
            file = std::fopen(path.c_str(), "wx");
        }
        if (!file) {
            moe_log("handoff-trace: cannot create %s: %s", path.c_str(), std::strerror(errno));
            return;
        }
        events.reset(new (std::nothrow) event[capacity]);
        if (!events) {
            moe_log("handoff-trace: allocation failed for %s", path.c_str());
            std::fclose(file);
            file = nullptr;
            return;
        }
        moe_log("handoff-trace: path=%s capacity=%llu", path.c_str(), (unsigned long long) capacity);
    }

    ~moe_handoff_trace() {
        if (file) std::fclose(file);
    }

    bool enabled() const { return events != nullptr; }
    uint64_t new_io_id() { return events ? next_io.fetch_add(1, std::memory_order_relaxed) + 1 : 0; }

    void emit(const char * name, moe_handoff_context context, int32_t expert = -1,
              int32_t slot = -1, int32_t result = 0, uint64_t bytes = 0,
              moe_handoff_context active = {0, 0, -1, 0, "none"}, moe_handoff_detail detail = {}) {
        if (!events) return;
        const uint64_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= capacity) {
            lost.fetch_add(1, std::memory_order_relaxed);
            return;
        }
#if defined(__linux__)
        static thread_local const uint64_t tid = (uint64_t) syscall(SYS_gettid);
#else
        static thread_local const uint64_t tid = (uint64_t) std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
        events[index] = {now_ns(), tid, name, context, active, expert, slot, result, bytes, detail};
    }

    // No producer may still be running when this is called.
    void flush() {
        if (!file) return;
        const uint64_t attempted = next.load(std::memory_order_relaxed);
        const uint64_t written = std::min(attempted, capacity);
        std::fprintf(file, "{\"schema\":\"llama.moe.handoff.v2\",\"clock\":\"CLOCK_MONOTONIC\","
                          "\"unit\":\"ns\",\"capacity\":%llu,\"entry_bytes\":%llu,\"buffer_bytes\":%llu,"
                          "\"attempted\":%llu,\"written\":%llu,\"lost\":%llu}\n",
                     (unsigned long long) capacity, (unsigned long long) sizeof(event),
                     (unsigned long long) (capacity * sizeof(event)), (unsigned long long) attempted,
                     (unsigned long long) written, (unsigned long long) lost.load(std::memory_order_relaxed));
        for (uint64_t i = 0; i < written; ++i) {
            const auto & e = events[i];
            std::fprintf(file, "{\"index\":%llu,\"ns\":%lld,\"tid\":%llu,\"event\":\"%s\","
                              "\"source\":\"%s\",\"job_id\":%llu,\"epoch\":%llu,\"layer\":%d,\"rows\":%d,"
                              "\"expert\":%d,\"slot\":%d,\"result\":%d,\"bytes\":%llu,"
                              "\"active_job_id\":%llu,\"active_epoch\":%llu,\"active_layer\":%d,\"active_rows\":%d,"
                              "\"generation\":%llu,\"domain\":\"%s\",\"binding_generation\":%llu,\"io_id\":%llu,\"reason\":\"%s\","
                              "\"related_expert\":%d,\"related_slot\":%d,\"related_generation\":%llu,"
                              "\"scanned\":%d,\"capacity_slots\":%d,\"free_slots\":%d,\"loading_slots\":%d,"
                              "\"protected_slots\":%d,\"pinned_slots\":%d,\"eligible_slots\":%d,"
                              "\"file_offset\":%llu,\"requested_pages\":%llu,\"resident_pages\":%lld,"
                              "\"probe_begin_ns\":%lld,\"probe_end_ns\":%lld,\"probe_errno\":%d,"
                              "\"admit_seq\":%llu,\"admit_class\":\"%s\",\"admit_active\":%u,"
                              "\"admit_waiters\":%u,\"admit_capacity\":%u}\n",
                         (unsigned long long) i, (long long) e.ns, (unsigned long long) e.tid, e.name,
                         e.context.source, (unsigned long long) e.context.job_id,
                         (unsigned long long) e.context.epoch, e.context.layer, e.context.rows, e.expert, e.slot, e.result,
                         (unsigned long long) e.bytes, (unsigned long long) e.active.job_id,
                         (unsigned long long) e.active.epoch, e.active.layer, e.active.rows,
                         (unsigned long long) e.context.generation, e.context.domain, (unsigned long long) e.context.binding, (unsigned long long) e.context.io_id, e.detail.reason,
                         e.detail.related_expert, e.detail.related_slot, (unsigned long long) e.detail.related_generation,
                         e.detail.scanned, e.detail.capacity_slots, e.detail.free_slots, e.detail.loading_slots,
                         e.detail.protected_slots, e.detail.pinned_slots, e.detail.eligible_slots,
                         (unsigned long long) e.detail.file_offset, (unsigned long long) e.detail.requested_pages,
                         (long long) e.detail.resident_pages, (long long) e.detail.probe_begin_ns,
                         (long long) e.detail.probe_end_ns, e.detail.probe_errno,
                         (unsigned long long) e.detail.admit_seq, e.detail.admit_class, e.detail.admit_active,
                         e.detail.admit_waiters, e.detail.admit_capacity);
        }
        const bool failed = std::ferror(file) != 0;
        const int close_result = std::fclose(file);
        file = nullptr;
        moe_log("handoff-trace: complete path=%s events=%llu lost=%llu write_error=%d",
                path.c_str(), (unsigned long long) written,
                (unsigned long long) lost.load(std::memory_order_relaxed), (int) (failed || close_result != 0));
    }
};

struct moe_handoff_thread_context {
    moe_handoff_trace * trace;
    moe_handoff_context context;
    int32_t expert;
    int32_t slot;
};
static thread_local moe_handoff_thread_context handoff_thread = {nullptr, {0, 0, -1, 0, "other"}, -1, -1};

struct moe_handoff_scope {
    moe_handoff_thread_context previous;
    moe_handoff_scope(moe_handoff_trace * trace, moe_handoff_context context, int32_t expert = -1, int32_t slot = -1) :
        previous(handoff_thread) {
        handoff_thread = {trace, context, expert, slot};
    }
    ~moe_handoff_scope() { handoff_thread = previous; }
};

struct moe_handoff_span {
    moe_handoff_trace & trace;
    const char * end;
    moe_handoff_context context;
    int32_t expert;
    int32_t slot;
    uint64_t bytes;
    int32_t result = -1;
    moe_handoff_span(moe_handoff_trace & trace, const char * begin, const char * end,
                     moe_handoff_context context, int32_t expert = -1, int32_t slot = -1, uint64_t bytes = 0) :
        trace(trace), end(end), context(context), expert(expert), slot(slot), bytes(bytes) {
        trace.emit(begin, context, expert, slot, 0, bytes);
    }
    ~moe_handoff_span() { trace.emit(end, context, expert, slot, result, bytes); }
    void complete() { result = 0; }
};

// Track each held segment separately. IO and condition-variable waits happen
// after release(), so they are never reported as time holding the layer mutex.
struct moe_handoff_lock_scope {
    moe_handoff_trace & trace;
    std::unique_lock<std::mutex> & guard;
    moe_handoff_context context;
    bool success = false;
    moe_handoff_lock_scope(moe_handoff_trace & trace, std::unique_lock<std::mutex> & guard,
                           moe_handoff_context context) : trace(trace), guard(guard), context(context) {
        trace.emit("PROCESS_LOCKED", context);
    }
    ~moe_handoff_lock_scope() {
        if (guard.owns_lock()) trace.emit("PROCESS_LOCK_RELEASING", context, -1, -1, success ? 0 : -1);
    }
    void release() {
        trace.emit("PROCESS_LOCK_RELEASING", context);
        guard.unlock();
    }
    void acquire() {
        if (guard.owns_lock()) return;
        trace.emit("PROCESS_LOCK_WAIT", context);
        guard.lock();
        trace.emit("PROCESS_LOCKED", context);
    }
    void complete() { success = true; }
};


/// O_DIRECT 能不能用，用一次真实的对齐读来判定（不是靠猜）。
/// 判不过就整程退回缓冲读 —— 半路 EINVAL 会让 worker 抛异常，宁可保守。
static bool probe_o_direct(const std::string & path) {
#ifdef O_DIRECT
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) {
        moe_log("O_DIRECT: off（open %s 失败: %s）", path.c_str(), std::strerror(errno));
        return false;
    }
    void * buf = nullptr;
    if (posix_memalign(&buf, 4096, 4096) != 0) {
        close(fd);
        return false;
    }
    const ssize_t n = pread(fd, buf, 4096, 0);
    const int saved = errno;
    free(buf);
    close(fd);
    if (n != 4096) {
        moe_log("O_DIRECT: off（4 KiB 对齐探测 rc=%zd errno=%s）",
                n, n < 0 ? std::strerror(saved) : "short");
        return false;
    }
    moe_log("O_DIRECT: on（对齐探测通过；读进 4 KiB 对齐 scratch 再 memcpy 到 slot）");
    return true;
#else
    GGML_UNUSED(path);
    moe_log("O_DIRECT: off（平台没有 O_DIRECT）");
    return false;
#endif
}

static bool park_window_enabled() {
    const char * e = std::getenv("LLAMA_MOE_PARK_WINDOW");
    if (!e || !*e) {
        return true;
    }
    return e[0] != '0';
}

using metal_buf_fn = bool (*)(ggml_backend_buffer_t);

static metal_buf_fn lookup_metal_buf_fn(ggml_backend_buffer_t buf, const char * name) {
    if (!buf) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (!buft) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return nullptr;
    }
    return (metal_buf_fn) ggml_backend_reg_get_proc_address(reg, name);
}
} // namespace

// Next-layer router × current hidden (attn_post_norm). Decode only. LLAMA_MOE_LOOKAHEAD_PREFETCH=1.
// Predicted ids are preadv'd into a ping-pong tail of the shared bank after install, overlapping
// the current layer's GPU work. Hits remap for free; misses still install on the critical path.
struct lookahead_expert_prefetch {
    static constexpr int32_t kMaxDepth = llama_moe_slot_runtime::max_prefetch_depth;
    enum class read_result { skipped, published, discarded };
    struct layer_timing {
        uint64_t epoch = 0;
        int64_t arm_ns = 0;
        int64_t overlap_ns = 0;
        uint64_t overlap_n = 0;
        uint64_t late_n = 0;
        int64_t read_wall_ns = 0;
        uint64_t reads = 0;
        uint64_t jobs = 0;
    };
    bool timing_enabled = false;
    bool trace_enabled = false;
    std::vector<layer_timing> timings; // protected by mtx until worker is joined

    struct hit {
        int32_t expert;
        int32_t slot;
    };

    bool enabled = false;
    int32_t depth = 8;
    /// 真正**读盘**的专家数；0 = 只算预测、不读盘（把预测用来指导淘汰）。
    /// 由 LLAMA_MOE_PREFETCH_READ 控制，默认 = depth（旧行为）。
    int32_t read_n = 0;
    int32_t n_slots = 0;
    int32_t n_expert = 0;
    int32_t last_moe_layer = -1;

    int32_t slot_floor() const { return n_slots - 2 * depth; }
    int32_t bank_base(int k) const { return slot_floor() + depth * (k & 1); }

    /// 一个待读的专家。**跨层流水的最小调度单位**。
    /// 老实现是「一层一个 job，job 之间互斥」：上一个 job 没跑完时 arm() 直接丢掉
    /// （`n_arm_skipped++`），端到端实测丢掉了 83% 的 arm 机会 —— 被丢掉的那一层只能让
    /// 主路径同步读，IO 全暴露在关键路径上（§4.46 的 75% late_jobs 就是这么来的）。
    /// 现在拆成「每专家一个 task」进 FIFO，由一组常驻线程并行消费：
    ///   - 顺序天然是层号升序（先要的层先读），所以提前 K 层提交的预测会真的排在前面；
    ///   - 并发度 = io_threads 个常驻线程，不再每个 job 现开线程（省 30–50 µs × 任务数）；
    ///   - 只有队列满（默认 4 个 job 的等价深度）才丢，丢了也计数。
    struct task {
        int32_t  layer  = -1;
        int32_t  slot   = -1;    // 乒乓区里的格号（to_window=false 时用）
        /// **读进去要写到哪**：true = 目标层自己的窗口（由窗口 LRU 认领），false = 乒乓区 slot。
        /// 这个语义在 arm() 时**一次性固化**，worker 不再读 lookahead 的全局标志 ——
        /// 全局量在 arm 与真正读盘之间有可能被改（set_depth / resolve_decode_window），
        /// 一旦翻转，乒乓格号就会被当成窗口内偏移写进别的专家的位置（静默读错专家）。
        bool     to_window = true;
        int32_t  expert = -1;
        uint64_t job    = 0;
        int64_t  arm_ns = 0;
        uint64_t epoch = 0;
        int32_t rows = 0;
        llama_moe::slot_manager::ticket request;
    };
    struct job_meta {
        int32_t  layer     = -1;
        int32_t  n         = 0;
        int32_t  remaining = 0;
        bool     to_window = true;   // 与 task 同语义，只用于 trace/日志
        int64_t  arm_ns    = 0;
        int32_t  expert[kMaxDepth] = {};
        uint64_t epoch = 0;
        int32_t rows = 0;
    };

    std::mutex              mtx;
    std::condition_variable cv_go;     // 兼容旧名字（worker 唤醒）
    std::condition_variable cv_done;   // job/队列全部完成
    bool                    quit = false;
    std::deque<task>        queue;
    std::map<uint64_t, job_meta> jobs;
    uint64_t                next_job    = 1;
    int32_t                 pending_jobs = 0;   // 已入队、未完成的 job 数（队列深度按它算）
    int32_t                 queue_cap    = 4;   // 队列里最多同时存在几个 job
    int32_t                 n_worker     = 8;   // 常驻读线程数（默认 = io_threads）
    std::vector<std::thread> workers;

    int32_t arm_layer = -1;
    int32_t arm_bank = -1;
    int32_t arm_n = 0;
    int32_t arm_expert[kMaxDepth] = {};

    /// 读一个专家。(layer, expert, slot, to_window)：to_window=true 表示写进目标层窗口，
    /// 此时 slot 无意义；false 表示写进乒乓区里的 slot 格。
    std::function<read_result(int, int, int, bool, llama_moe::slot_manager::ticket)> read_fn;
    std::function<llama_moe::slot_manager::result(int, int, uint64_t, uint64_t, int)> enqueue_fn;
    std::function<bool(int, llama_moe::slot_manager::ticket)> cancel_fn;
    moe_handoff_trace * handoff = nullptr;
    /// 是否让预读写进窗口。**只在 init 的 resolve_decode_window() 里确定**；
    /// 传给 worker 的是 arm 时固化的 task::to_window，而不是这个字段。
    bool window_mode = false;
    bool refresh = false;
    llama_moe::prefetch_schedule refresh_schedule;
    uint64_t refresh_plans = 0, refresh_near_updates = 0;
    uint64_t refresh_far_created = 0, refresh_near_created = 0;
    uint64_t refresh_cancelled = 0, refresh_expired = 0, refresh_reprioritized = 0;
    std::atomic<uint64_t> n_hits{0};
    std::atomic<uint64_t> n_armed{0};
    // 诊断：预读到底有没有在工作。见 arm()/take()。
    std::atomic<uint64_t> n_arm_skipped{0};   // 因上一个 job 未完成而放弃的次数
    std::atomic<uint64_t> n_waits{0};         // take() 里真的阻塞等待的次数
    std::atomic<int64_t>  n_wait_ns{0};       // 阻塞等待的总时长
    std::atomic<uint64_t> n_pred_read{0};     // worker 实际读进去的专家数
    std::atomic<uint64_t> n_pred_published{0};// 预读发布进窗口（真正变成缓存条目）的次数
    std::atomic<uint64_t> n_pred_discarded{0};
    std::atomic<uint64_t> n_pred_skipped{0};
    std::atomic<uint64_t> n_pred_errors{0};
    std::atomic<uint64_t> n_predict_only{0};  // 只预测不读盘的次数（轻量组合模式）
    std::atomic<uint64_t> n_evict_saved{0};   // 因「预测会用」而被跳过的淘汰次数
    std::atomic<uint64_t> n_evict_forced{0};  // pass0 找不到非预测候选、被迫进入 pass1 的次数
    std::atomic<uint64_t> n_pred_cached{0};   // arm 时因「目标层已有」而被过滤掉的预测数
    std::atomic<uint64_t> n_used_from_pf{0};  // 主路径用到的专家里，落在**预读区**的次数
                                              // （用完不保留，本层结束就被重填冲掉）
    std::atomic<uint64_t> n_evict_total{0};   // 淘汰总次数
    std::atomic<uint64_t> n_pred_bytes{0};    // worker 实际读进去的字节
    bool ready = false;
    moe_io_pool io_pool;

    ~lookahead_expert_prefetch() { stop(); }

    void init(int32_t slots, int32_t experts, int32_t last_moe,
              std::function<read_result(int, int, int, bool, llama_moe::slot_manager::ticket)> fn, moe_handoff_trace * trace_sink) {
        handoff = trace_sink;
        n_slots = slots;
        n_expert = experts;
        last_moe_layer = last_moe;
        refresh = prefetch_refresh_enabled();
        refresh_schedule.resize((size_t) last_moe + 1);
        static_assert(kMaxDepth == llama_moe::prefetch_schedule::max_reads, "prefetch budget width changed");
        read_fn = std::move(fn);
        const char * profile = std::getenv("LLAMA_MOE_PREFETCH_TIMING");
        timing_enabled = profile && *profile && *profile != '0';
        const char * trace = std::getenv("LLAMA_MOE_PREFETCH_TRACE");
        trace_enabled = timing_enabled && trace && *trace && *trace != '0';
        timings.resize((size_t) last_moe + 1);
        ready = true;

        int want = 0;
        const char * on = std::getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH");
        if (on && on[0] != '\0' && on[0] != '0') {
            want = 8;
            if (const char * d = std::getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH")) {
                want = atoi(d);
            }
        } else {
            // the mechanism was called FATE before this rename; keep the old spellings loud
            if (std::getenv("LLAMA_MOE_FATE") || std::getenv("LLAMA_MOE_FATE_DEPTH")) {
                moe_log("lookahead-prefetch: off, and LLAMA_MOE_FATE* is no longer read; "
                        "use LLAMA_MOE_LOOKAHEAD_PREFETCH[=_DEPTH]");
            } else {
                moe_log("lookahead-prefetch: off (LLAMA_MOE_LOOKAHEAD_PREFETCH=%s)", on ? on : "unset");
            }
        }
        {
            const char * rs = std::getenv("LLAMA_MOE_PREFETCH_READ");
            read_n = rs ? std::atoi(rs) : -1;   // -1 = 跟随 depth
        }
        // 跨层流水：队列里允许多少个 job 同时在飞、多少个常驻读线程。
        {
            const char * q = std::getenv("LLAMA_MOE_PREFETCH_QUEUE");
            const int v = q ? std::atoi(q) : 0;
            queue_cap = (v > 0 && v <= 64) ? v : 4;
            // 预读线程数可以**独立于**主路径的 io_threads 调。
            // 动机：设备是单一串行资源，预读线程越多，主路径的同步读排得越靠后
            // （§4.46 的 75% late_jobs 就是这么来的）。调小它能把时间还给主路径。
            const char * w = std::getenv("LLAMA_MOE_PREFETCH_WORKERS");
            const int vw = w ? std::atoi(w) : 0;
            n_worker = (vw > 0 && vw <= 64) ? vw : moe_ssd_io_threads();
        }
        set_depth(want);
        if (read_n < 0 || read_n > depth) {
            read_n = depth;                     // 默认与 depth 一致（旧行为）
        }
    }

    void set_depth(int32_t want) {
        if (!ready) {
            return;
        }
        const int32_t configured = want;
        want = llama_moe_slot_runtime::normalize_prefetch_depth(n_slots, want);
        if (configured > 0 && want == 0) {
            moe_log("lookahead-prefetch: off, bank %d too small for depth %d", (int) n_slots, (int) configured);
        }
        const int32_t cur = enabled ? depth : 0;
        if (want == cur && (want == 0 || !workers.empty())) {
            return;
        }

        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_done.wait(lk, [this]() { return !queue.empty() ? false : jobs.empty(); });
            arm_layer = -1;
            arm_n = 0;
            if (want > 0) {
                depth = want;
                enabled = true;
            } else {
                enabled = false;
            }
        }

        if (enabled && workers.empty()) {
            quit = false;
            const int32_t n = std::max(1, n_worker);
            try {
                for (int32_t i = 0; i < n; ++i) workers.emplace_back([this]() { worker_loop(); });
            } catch (...) { stop(); throw; }
        }

        if (enabled) {
            moe_log("lookahead-prefetch: on depth=%d reserved=[%d,%d) workers=%d queue_cap=%d",
                    (int) depth, (int) slot_floor(), (int) n_slots,
                    (int) workers.size(), (int) queue_cap);
        } else {
            moe_log("lookahead-prefetch: off (depth=0)");
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            quit = true;
        }
        cv_go.notify_all();
        for (auto & t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
        workers.clear();
        enabled = false;
    }

    void drain() {
        if (!enabled) {
            return;
        }
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this]() { return queue.empty() && jobs.empty(); });
        arm_layer = -1;
        arm_n = 0;
    }

    void report() const {
        if (n_armed.load(std::memory_order_relaxed) == 0) {
            return;
        }
        const uint64_t armed = n_armed.load(std::memory_order_relaxed);
        const uint64_t hits = n_hits.load(std::memory_order_relaxed);
        const uint64_t preds = armed * (uint64_t) depth;
        if (window_mode) {
            // 窗口模式下命中由窗口 LRU 负责统计（cache-summary 的 window_hit），
            // `n_hits` 在这里**结构性恒为 0**（老实现也如此）。打「0.0%」会误导，
            // 所以明确写 n/a 并指向真正的计数器。
            moe_log("lookahead-prefetch: armed=%llu hits=n/a（窗口模式，命中看 cache-summary 的 window_hit）"
                    " read=%llu published=%llu",
                (unsigned long long) armed,
                (unsigned long long) n_pred_read.load(),
                (unsigned long long) n_pred_published.load());
            return;
        }
        moe_log("lookahead-prefetch: armed=%llu hits=%llu (%.1f%% of predictions)",
            (unsigned long long) armed,
            (unsigned long long) hits,
            preds ? 100.0 * (double) hits / (double) preds : 0.0);
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mtx);
        arm_layer = -1;
        arm_n = 0;
    }

    void observe_demand(int layer, uint64_t epoch, int rows) {
        if (refresh) cancel_refresh_tasks([&](const task & t) {
            return llama_moe::prefetch_schedule::due(t.epoch, t.layer, epoch, layer);
        }, true);
        if (!timing_enabled || layer < 0 || layer >= (int) timings.size()) return;
        const int64_t demand_ns = now_ns();
        std::lock_guard<std::mutex> lk(mtx);
        auto & t = timings[(size_t) layer];
        if (t.arm_ns > 0 && t.epoch == epoch) {
            // 本层还有 task 没读完 = 迟到（老的判定是「唯一的那个 job 还在跑」，
            // 现在队列里可能有多个 job，所以按**本层**是否还有剩余任务判）。
            bool still_pending = false;
            for (const auto & kv : jobs) {
                if (kv.second.layer == layer) { still_pending = true; break; }
            }
            if (trace_enabled) {
                moe_log("prefetch-demand: layer=%d epoch=%llu rows=%d window_us=%.3f late=%d",
                        layer, (unsigned long long) epoch, rows, (demand_ns - t.arm_ns) / 1000.0,
                        (int) still_pending);
            }
            t.overlap_ns += std::max<int64_t>(0, demand_ns - t.arm_ns);
            t.overlap_n++;
            if (still_pending) t.late_n++;
            t.arm_ns = 0;
        }
    }

    // Call only after stop(): the final in-flight job must be included.
    void report_completed() const {
        moe_log("prefetch-refresh-summary: enabled=%d plans=%llu near_updates=%llu far_created=%llu near_created=%llu cancelled=%llu expired=%llu queued_reprioritized=%llu",
                (int) refresh, (unsigned long long) refresh_plans, (unsigned long long) refresh_near_updates,
                (unsigned long long) refresh_far_created, (unsigned long long) refresh_near_created,
                (unsigned long long) refresh_cancelled, (unsigned long long) refresh_expired,
                (unsigned long long) refresh_reprioritized);
        moe_log("prefetch-summary: read=%llu published=%llu discarded=%llu skipped=%llu errors=%llu",
                (unsigned long long) n_pred_read.load(),
                (unsigned long long) n_pred_published.load(),
                (unsigned long long) n_pred_discarded.load(),
                (unsigned long long) n_pred_skipped.load(),
                (unsigned long long) n_pred_errors.load());
        if (!timing_enabled) return;
        for (size_t layer = 0; layer < timings.size(); ++layer) {
            const auto & t = timings[layer];
            if (!t.jobs && !t.overlap_n) continue;
            // 口径（老版本没写清，容易被误读）：
            //   overlap_samples = 需求观测次数（**per-hop**，一层一次）
            //   late_demands    = 其中「需求到达时本层还有 task 没读完」的次数（per-hop，
            //                     老实现叫 late_jobs，是 per-job，口径已变）
            //   jobs            = 完成的任务组数（per-job）  read_experts = 实际读的专家数（per-task）
            moe_log("prefetch-timing: layer=%zu overlap_samples=%llu overlap_mean_us=%.3f "
                    "late_demands=%llu read_jobs=%llu read_experts=%llu read_us_per_expert=%.3f",
                    layer, (unsigned long long) t.overlap_n,
                    t.overlap_n ? t.overlap_ns / (1000.0 * t.overlap_n) : 0.0,
                    (unsigned long long) t.late_n, (unsigned long long) t.jobs,
                    (unsigned long long) t.reads,
                    t.reads ? t.read_wall_ns / (1000.0 * t.reads) : 0.0);
        }
    }

    void take(int32_t layer, std::vector<hit> & out) {
        if (!enabled) {
            return;
        }
        std::unique_lock<std::mutex> lk(mtx);

        if (window_mode) {
            // 窗口模式：worker 与缓存**共用同一片 slot**，但**不需要等它**。
            //
            // 为什么可以不等（这是「缓存 + 预读结合」的关键）：
            //   1. worker 认领 slot 时标 `kSlotPending`，主路径分配/淘汰都跳过它 ——
            //      既不会覆盖 worker 正在写的位置，也不会把半成品当成有效条目；
            //   2. 主路径要的专家如果 worker 已经**发布**完成，`expert_to_slot` 里就有，
            //      自然算命中；没发布就按未命中自己读（最多重复读一次，代价有界）；
            //   3. worker 读盘时**不持锁**（见 install_prefetch_into_window），
            //      所以不会把主线程挡在 mutex 上。
            //
            // 以前这里无条件等整批 K 个读完，实测每 token 白等 43ms ——
            // 那才是「缓存与预读互斥」的真正原因，不是两者本质冲突。
            arm_layer = -1;
            arm_n = 0;
            return;
        }

        // 乒乓模式：worker 写尾部专用区 [slot_floor, n_slots)，主路径只用
        // [0, slot_floor)（见 process() 里 alloc_hi = slot_floor），两者**不重叠**，
        // 所以不需要等 worker。
        //
        // 以前这里无条件 cv_done.wait()，等的是**整批 K 个专家**（含主路径根本
        // 不需要的那些）。实测代价很大：K=8 时每 token 白等 54.6ms、K=16 时 259ms
        // —— 比整步还长，直接把预读的收益吃光。
        // 现在改成：**只接受已经完成、且正是本层**的预读结果；没好的就跳过，
        // 主路径按未命中正常读（本来也要读）。
        if (!jobs.empty()) {
            // 结果还没好。arm_layer 仍是上一次的值，下面的一致性检查会拦住它。
            return;
        }
        if (arm_layer != layer || arm_n <= 0) {
            arm_layer = -1;
            arm_n = 0;
            return;
        }
        const int32_t base = bank_base(arm_bank);
        for (int k = 0; k < arm_n; ++k) {
            out.push_back({ arm_expert[k], base + k });
        }
        arm_layer = -1;
        arm_n = 0;
    }

    // Queue bookkeeping never calls the layer authority while mtx is held.
    // A removed task remains in remaining until its exact queued ticket has
    // been cancelled, just as a popped worker remains counted until return.
    void finish_refresh_task_locked(const task & t) {
        auto it = jobs.find(t.job);
        if (it == jobs.end() || it->second.remaining <= 0) throw std::logic_error("unbalanced refreshed prefetch job");
        if (--it->second.remaining) return;
        const auto done = it->second;
        jobs.erase(it); --pending_jobs;
        if (timing_enabled) ++timings[(size_t) done.layer].jobs;
        if (handoff) handoff->emit("JOB_DONE", {t.job, done.epoch, done.layer, done.rows, "prefetch", 0, "window"});
        cv_done.notify_all();
    }

    template<class Predicate> void cancel_refresh_tasks(Predicate obsolete, bool expired) {
        std::vector<task> removed;
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (std::none_of(queue.begin(), queue.end(), [&](const task & t) { return t.request && obsolete(t); })) return;
            // Reserve before the first erase: allocation failure must not leave
            // an unowned removed task contributing to job.remaining forever.
            removed.reserve(queue.size());
            for (auto it = queue.begin(); it != queue.end();) {
                if (!it->request || !obsolete(*it)) { ++it; continue; }
                removed.push_back(*it); it = queue.erase(it);
            }
        }
        std::exception_ptr error;
        for (const auto & t : removed) {
            bool cancelled = false;
            try { cancelled = cancel_fn(t.layer, t.request); }
            catch (...) { if (!error) error = std::current_exception(); }
            std::lock_guard<std::mutex> lock(mtx);
            if (cancelled) {
                refresh_schedule.refund(t.layer, t.epoch, t.expert);
                ++refresh_cancelled; if (expired) ++refresh_expired;
            }
            auto & job = jobs.at(t.job);
            for (int k = 0; k < job.n; ++k) if (job.expert[k] == t.expert) {
                std::move(job.expert + k + 1, job.expert + job.n, job.expert + k);
                --job.n; break;
            }
            if (handoff) handoff->emit(expired ? "REFRESH_EXPIRE" : "REFRESH_CANCEL",
                handoff_request({t.job, t.epoch, t.layer, t.rows, "prefetch", 0, "window"}, t.request),
                t.expert, t.request.slot, cancelled ? 0 : -1);
            finish_refresh_task_locked(t);
        }
        if (error) std::rethrow_exception(error);
    }

    std::vector<int32_t> refreshed_predictions(int32_t layer, uint64_t epoch) {
        std::lock_guard<std::mutex> lock(mtx);
        const auto * p = refresh_schedule.find(layer, epoch);
        return p ? std::vector<int32_t>(p->experts.begin(), p->experts.begin() + p->spent) : std::vector<int32_t>{};
    }

    bool arm_refreshed(int32_t layer, const std::vector<int32_t> & pred, uint64_t epoch, int rows, bool near) {
        auto context = moe_handoff_context{0, epoch, layer, rows, "prefetch", 0, "window"};
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (near) {
                if (!refresh_schedule.refine(layer, epoch)) return false;
                ++refresh_near_updates;
            } else {
                if (!refresh_schedule.begin(layer, epoch, read_n)) return false;
                ++refresh_plans;
            }
            if (handoff) handoff->emit(near ? "REFRESH_NEAR" : "REFRESH_PLAN", context, -1, -1,
                refresh_schedule.allowance(layer, epoch, near));
        }
        if (near) cancel_refresh_tasks([&](const task & t) {
            return t.epoch == epoch && t.layer == layer && std::find(pred.begin(), pred.end(), t.expert) == pred.end();
        }, false);

        std::unique_lock<std::mutex> lock(mtx);
        const int32_t limit = refresh_schedule.allowance(layer, epoch, near);
        if (!enabled || quit || !limit || pred.empty()) return true;
        if (handoff) handoff->emit("ARM_ATTEMPT", context);
        auto existing = std::find_if(jobs.begin(), jobs.end(), [&](const auto & entry) {
            return entry.second.to_window && entry.second.layer == layer && entry.second.epoch == epoch;
        });
        const bool new_job = existing == jobs.end();
        if (new_job && pending_jobs >= std::max(1, queue_cap)) {
            ++n_arm_skipped;
            if (handoff) handoff->emit("ARM_SKIP", context, -1, -1, pending_jobs);
            return true;
        }
        const uint64_t id = new_job ? next_job++ : existing->first;
        context.job_id = id;
        const int64_t submitted_ns = timing_enabled || handoff ? now_ns() : 0;
        if (new_job) {
            job_meta job; job.layer = layer; job.epoch = epoch; job.rows = rows;
            job.arm_ns = submitted_ns; job.to_window = true;
            jobs.emplace(id, job); ++pending_jobs;
            // In refresh mode these two markers carry the admission upper
            // bound, not a byte/task count. REFRESH_ENQUEUE is the actual ledger.
            if (handoff) {
                handoff->emit("ARM_ACCEPT", context, -1, -1, limit);
                handoff->emit("JOB_BEGIN", context, -1, -1, limit);
            }
        }
        // This submission guard prevents a far worker from erasing a job while
        // near registration temporarily releases mtx to enter the layer mutex.
        ++jobs.at(id).remaining;
        const auto * plan = refresh_schedule.find(layer, epoch);
        const auto prior = *plan;
        lock.unlock();

        std::array<task, kMaxDepth> tasks;
        int32_t count = 0;
        try {
            std::array<int32_t, kMaxDepth> seen{};
            int32_t seen_count = 0;
            for (int32_t expert : pred) {
                if (count >= limit) break;
                if (expert < 0 || expert >= n_expert ||
                    std::find(prior.experts.begin(), prior.experts.begin() + prior.spent, expert) != prior.experts.begin() + prior.spent) continue;
                // The prediction is already deduplicated by the caller; retain
                // this bounded check for direct callers without heap allocation.
                if (std::find(seen.begin(), seen.begin() + seen_count, expert) != seen.begin() + seen_count) continue;
                const auto registered = enqueue_fn(layer, expert, epoch, id, rows);
                if (registered.action != llama_moe::slot_manager::outcome::created) continue;
                seen[(size_t) seen_count++] = expert;
                tasks[(size_t) count++] = task{layer, -1, true, expert, id, submitted_ns, epoch, rows, registered.request};
            }
            lock.lock();
            if (quit) throw std::runtime_error("MoE prefetch stopped during refresh");
            auto & job = jobs.at(id);
            if (job.n + count > kMaxDepth) throw std::logic_error("refreshed prefetch job exceeds its read budget");
            size_t pushed = 0;
            try {
                for (int i = 0; i < count; ++i) { queue.push_back(tasks[(size_t) i]); ++pushed; }
            } catch (...) {
                while (pushed) { queue.pop_back(); --pushed; }
                throw;
            }
            for (int i = 0; i < count; ++i) {
                const auto & t = tasks[(size_t) i];
                refresh_schedule.created(layer, epoch, t.expert);
                job.expert[job.n++] = t.expert;
                if (handoff) handoff->emit("REFRESH_ENQUEUE", handoff_request(context, t.request),
                    t.expert, t.request.slot, near ? 1 : 0);
            }
            job.remaining += count;
            if (near) refresh_near_created += count; else refresh_far_created += count;
            if (timing_enabled) { timings[(size_t) layer].epoch = epoch; timings[(size_t) layer].arm_ns = submitted_ns; }
            ++n_armed;
            finish_refresh_task_locked(task{layer, -1, true, -1, id, submitted_ns, epoch, rows, {}});
            lock.unlock(); cv_go.notify_all();
        } catch (...) {
            const auto error = std::current_exception();
            if (lock.owns_lock()) lock.unlock();
            for (int i = 0; i < count; ++i) {
                try { cancel_fn(layer, tasks[(size_t) i].request); } catch (...) {}
            }
            lock.lock();
            if (handoff) handoff->emit("REFRESH_SUBMIT_FAILED", context, -1, -1, -1);
            finish_refresh_task_locked(task{layer, -1, true, -1, id, submitted_ns, epoch, rows, {}});
            lock.unlock(); cv_done.notify_all();
            std::rethrow_exception(error);
        }
        return true;
    }

    void arm_managed(int32_t layer, const std::vector<int32_t> & pred, uint64_t epoch, int rows) {
        moe_handoff_context context = {0, epoch, layer, rows, "prefetch", 0, "window"};
        if (!enabled || layer < 1 || layer > last_moe_layer || pred.empty()) {
            if (handoff) handoff->emit("ARM_SKIP_INVALID", context);
            return;
        }
        if (read_n <= 0) {
            ++n_predict_only;
            if (handoff) handoff->emit("ARM_SKIP_PREDICT_ONLY", context);
            return;
        }
        job_meta job;
        std::unique_lock<std::mutex> lock(mtx);
        if (pending_jobs >= std::max(1, queue_cap)) {
            ++n_arm_skipped;
            if (handoff) handoff->emit("ARM_SKIP", context, -1, -1, pending_jobs);
            return;
        }
        std::vector<uint8_t> seen((size_t) n_expert, 0);
        for (int32_t expert : pred) {
            if (job.n >= read_n) break;
            if (expert < 0 || expert >= n_expert || seen[(size_t) expert]) continue;
            seen[(size_t) expert] = 1; job.expert[job.n++] = expert;
        }
        if (!job.n) return;
        for (int k = job.n; k < kMaxDepth; ++k) job.expert[k] = -1;
        job.layer = layer; job.epoch = epoch; job.rows = rows; job.to_window = true;
        job.arm_ns = timing_enabled || handoff ? now_ns() : 0;
        const uint64_t id = next_job++;
        context.job_id = id;
        jobs.emplace(id, job); ++pending_jobs;
        if (handoff) { handoff->emit("ARM_ACCEPT", context, -1, -1, job.n); handoff->emit("JOB_BEGIN", context, -1, -1, job.n); }
        lock.unlock();

        std::array<task, kMaxDepth> tasks;
        int count = 0;
        try {
            // No queue mutex is held while the layer authority registers or
            // coalesces requests. Queued entries reserve no physical memory.
            for (int k = 0; k < job.n; ++k) {
                const auto registered = enqueue_fn(layer, job.expert[k], epoch, id, rows);
                if (registered.action != llama_moe::slot_manager::outcome::created) continue;
                tasks[(size_t) count++] = task{layer, -1, true, job.expert[k], id, job.arm_ns, epoch, rows, registered.request};
            }
            lock.lock();
            if (quit) throw std::runtime_error("MoE prefetch stopped during submission");
            size_t pushed = 0;
            try {
                for (int i = 0; i < count; ++i) { queue.push_back(tasks[(size_t) i]); ++pushed; }
            } catch (...) {
                while (pushed) { queue.pop_back(); --pushed; }
                throw;
            }
            jobs.at(id).remaining = count;
            if (!count) {
                jobs.erase(id); --pending_jobs;
                if (handoff) handoff->emit("JOB_DONE", context);
                cv_done.notify_all();
            }
            if (timing_enabled) { timings[(size_t) layer].epoch = epoch; timings[(size_t) layer].arm_ns = job.arm_ns; }
            ++n_armed;
            lock.unlock(); cv_go.notify_all();
        } catch (...) {
            if (lock.owns_lock()) lock.unlock();
            for (int i = 0; i < count; ++i) cancel_fn(layer, tasks[(size_t) i].request);
            lock.lock();
            if (jobs.erase(id)) --pending_jobs;
            if (handoff) handoff->emit("JOB_CANCEL", context, -1, -1, -1);
            lock.unlock(); cv_done.notify_all();
            throw;
        }
    }

    void arm(int32_t layer_next, const std::vector<int32_t> & pred, uint64_t epoch, int rows) {
        const moe_handoff_context context = {0, epoch, layer_next, rows, "prefetch", 0, window_mode ? "window" : "bank"};
        if (handoff) handoff->emit("ARM_ATTEMPT", context);
        if (enqueue_fn && window_mode) { arm_managed(layer_next, pred, epoch, rows); return; }
        // rows 只用于日志；队列化之后 job 里不再记它（完成时按层聚合）。
        GGML_UNUSED(rows);
        if (!enabled || layer_next < 1 || layer_next > last_moe_layer || pred.empty()) {
            if (handoff) handoff->emit("ARM_SKIP_INVALID", context);
            return;
        }

        // read_n == 0：只要预测、不读盘。这是「缓存 + 预测」的轻量组合形态。
        // 预测本身由调用方（impl::process）存下来供淘汰参考。
        if (read_n <= 0) {
            n_predict_only.fetch_add(1, std::memory_order_relaxed);
            if (handoff) handoff->emit("ARM_SKIP_PREDICT_ONLY", context);
            return;
        }

        job_meta job;
        std::unique_lock<std::mutex> lk(mtx);
        // 队列深度按**job 数**算：窗口模式下允许多层同时在飞（跨层流水的前提），
        // 乒乓模式下只能有 1 个（slot 是 [base, base+depth) 的固定区，两层同奇偶会撞）。
        const int32_t cap = window_mode ? std::max(1, queue_cap) : 1;
        if (pending_jobs >= cap) {
            // 队列满 = 读真的跟不上算。这个计数就是「流水不够深」的直接度量。
            n_arm_skipped.fetch_add(1, std::memory_order_relaxed);
            if (handoff) handoff->emit("ARM_SKIP", context, -1, -1, pending_jobs);
            return;
        }
        std::vector<uint8_t> seen((size_t) n_expert, 0);
        for (int32_t e : pred) {
            if (job.n >= read_n) {   // 只读 read_n 个（<= depth）；read_n==0 已在上面返回
                break;
            }
            if (e < 0 || e >= n_expert || seen[(size_t) e]) {
                continue;
            }
            seen[(size_t) e] = 1;
            job.expert[job.n++] = e;
        }
        if (job.n <= 0) {
            return;
        }
        for (int k = job.n; k < kMaxDepth; ++k) {
            job.expert[k] = -1;
        }
        job.layer     = layer_next;
        job.remaining = job.n;
        job.to_window = window_mode;
        job.arm_ns    = timing_enabled || handoff ? now_ns() : 0;
        job.epoch = epoch; job.rows = rows;
        const uint64_t id = next_job++;
        jobs.emplace(id, job);
        pending_jobs++;
        if (handoff) { auto c = context; c.job_id = id; handoff->emit("ARM_ACCEPT", c, -1, -1, job.n); handoff->emit("JOB_BEGIN", c, -1, -1, job.n); }

        // 语义在这里固化：worker 只认 task::to_window，不再回头看全局的 window_mode。
        const int32_t base = bank_base(layer_next & 1);
        for (int k = 0; k < job.n; ++k) {
            queue.push_back(task{ layer_next, base + k, window_mode, job.expert[k], id, job.arm_ns, epoch, rows, {} });
        }
        if (timing_enabled) {
            auto & t = timings[(size_t) layer_next];
            t.epoch = epoch;
            t.arm_ns = job.arm_ns;
        }
        n_armed.fetch_add(1, std::memory_order_relaxed);
        lk.unlock();
        cv_go.notify_all();
    }

    /// 常驻读线程：从 FIFO 取一个 (层, 专家)，读进目标层窗口并发布。
    /// 并发度由线程数决定（默认 = LLAMA_MOE_IO_THREADS），不再是「每 job 现开 n 个线程」。
    void worker_loop() {
        for (;;) {
            task t;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_go.wait(lk, [this]() { return quit || !queue.empty(); });
                if (quit && queue.empty()) {
                    return;
                }
                if (refresh) {
                    const auto next = std::min_element(queue.begin(), queue.end(), [](const task & a, const task & b) {
                        return llama_moe::prefetch_schedule::earlier(a.epoch, a.layer, b.epoch, b.layer);
                    });
                    if (next != queue.begin()) ++refresh_reprioritized;
                    t = *next; queue.erase(next);
                } else {
                    t = queue.front(); queue.pop_front();
                }
            }

            const moe_handoff_context task_context = handoff_request({t.job, t.epoch, t.layer, t.rows, "prefetch", 0, t.to_window ? "window" : "bank"}, t.request);
            moe_handoff_scope task_scope(handoff, task_context, t.expert, t.slot);
            if (handoff) handoff->emit("TASK_BEGIN", task_context, t.expert, t.slot);
            const int64_t start_ns = timing_enabled ? now_ns() : 0;
            try {
                // to_window=true：让 impl 按**目标层**的窗口 LRU 认领并发布 ——
                // 预读成果直接变成那层的缓存命中，不需要再搬一次。
                const auto result = read_fn(t.layer, t.expert, t.slot, t.to_window, t.request);
                if (handoff) handoff->emit("PF_RESULT", task_context, t.expert, t.slot, (int32_t) result);
                if (result == read_result::skipped) {
                    n_pred_skipped.fetch_add(1, std::memory_order_relaxed);
                } else {
                    n_pred_read.fetch_add(1, std::memory_order_relaxed);
                    if (result == read_result::published) {
                        n_pred_published.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        n_pred_discarded.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            } catch (...) {
                if (handoff) handoff->emit("PF_ERROR", task_context, t.expert, t.slot, -1);
                n_pred_errors.fetch_add(1, std::memory_order_relaxed);
            }
            if (handoff) handoff->emit("TASK_DONE", task_context, t.expert, t.slot);

            job_meta done_job;
            bool finished = false;
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (timing_enabled) {
                    auto & tm = timings[(size_t) t.layer];
                    tm.read_wall_ns += now_ns() - start_ns;
                    tm.reads++;
                    tm.overlap_ns += std::max<int64_t>(0, start_ns - t.arm_ns);
                    tm.overlap_n++;
                }
                auto it = jobs.find(t.job);
                if (it != jobs.end() && --it->second.remaining == 0) {
                    done_job = it->second;
                    jobs.erase(it);
                    pending_jobs--;
                    finished = true;
                }
            }
            if (finished) {
                std::lock_guard<std::mutex> lk(mtx);
                if (trace_enabled) {
                    moe_log("prefetch-job: layer=%d n=%d", done_job.layer, done_job.n);
                }
                if (timing_enabled) {
                    timings[(size_t) done_job.layer].jobs++;
                }
                arm_layer = done_job.layer;
                arm_bank = done_job.layer & 1;
                arm_n = done_job.n;
                memcpy(arm_expert, done_job.expert, sizeof(arm_expert));
                // notify 在锁内，与老实现一致：drain()/set_depth() 的等待方醒来时
                // 已经能看到 arm_* 全部落定。
                if (handoff) handoff->emit("JOB_DONE", {t.job, done_job.epoch, done_job.layer, done_job.rows, "prefetch", 0, done_job.to_window ? "window" : "bank"});
                cv_done.notify_all();
            }
        }
    }
};

struct llama_moe_slot_runtime::impl {
    struct binding {
        ggml_tensor * tensor = nullptr;
        const llama_moe_experts_entry * entry = nullptr;
    };

    // 一次读取要用的「打包视图」：prefill 和 decode 各有自己的一套张量与文件布局。
    // 抽出来是为了让 read_expert_directv 能按阶段选，不用复制读取逻辑。
    struct pack_view {
        const std::vector<binding> * bindings = nullptr;
        const std::vector<size_t>  * order    = nullptr;
        size_t base = 0;
        size_t stride = 0;
        std::string path;
        bool ok = false;

        bool valid() const { return ok && bindings && order; }
    };

    struct layer_state {
        bool enabled = false;
        bool pack_ok = false;
        int32_t window_start = 0;   // 本层 decode 窗口的起始 slot（0 表示未启用窗口）
        /// 本层窗口的**总宽度**（缓存区 c_il + 预读暂存区 K）。各层可不同 ——
        /// 见 llama_model::moe_layer_windows()：热层给大、冷层给小。
        /// window_start 恒为 0（每层有自己的 decode 张量，slot 空间就是 [0, width)）。
        int32_t window_width = 0;
        size_t pack_base = 0;
        size_t pack_bytes = 0;
        size_t pack_stride = 0;
        std::string pack_path;
        std::vector<binding> bindings;
        std::vector<size_t> pack_order; // bindings sorted by file offset, i.e. experts packed order

        // decode 专用的一套（每层独立窗口，ne[2] = c）。prefill 用上面那套。
        bool dec_pack_ok = false;
        size_t dec_pack_base = 0;
        size_t dec_pack_stride = 0;
        std::string dec_pack_path;
        std::vector<binding> dec_bindings;
        std::vector<size_t> dec_pack_order;

        /// 预读成功发布进本窗口的次数（诊断用；与 n_hits 的区别是后者统计的是
        /// 主路径命中，这里统计 worker 的产出）。
        uint64_t n_prefetch_published = 0;
        std::vector<int32_t> slot_to_expert;
        std::vector<int32_t> expert_to_slot;
        std::vector<uint64_t> slot_age;
        std::vector<uint8_t> protected_experts;
        /// Persistently pinned hot experts (`LLAMA_MOE_PREWARM`, unless
        /// `LLAMA_MOE_PREWARM_PIN=0`). Unlike `protected_experts` (cleared every
        /// step), these are never evicted, so the resident hot set stays put.
        std::vector<uint8_t> pinned_experts;
        uint64_t age = 0;
        std::mutex mutex;
        std::unique_ptr<llama_moe::slot_manager> manager;
        std::vector<llama_moe::slot_manager::result> requests;
    };
    struct map_userdata {
        llama_moe_slot_runtime * runtime = nullptr;
        int layer = -1;
    };

    struct pending_install {
        int32_t expert = -1;
        int32_t slot = -1;
        llama_moe::slot_manager::ticket request;
        uint64_t io_id = 0;
        bool retry = false;
    };

    moe_io_pool io_pool;

    struct install_timing {
        int n = 0;
        int n_th = 0;
        bool pack = false;
        int64_t wall_ns = 0;
        int64_t pread_sum_ns = 0;
        int64_t pread_max_ns = 0;
        size_t bytes = 0;
    };

    // 「合并相邻专家读」可行性统计。见 accumulate_runs() 的注释。
    struct run_stats {
        int n          = 0;   // 连续段总数（仅按 expert id 连续划分）
        int experts    = 0;   // 落在段里的专家数（应等于 miss 总数）
        int max        = 0;   // 最长段
        int ok_n       = 0;   // 段内 slot 也连号的段数（可零拷贝合并）
        int ok_ex      = 0;   // 这些段覆盖的专家数
        int scatter_n  = 0;   // slot 不连号、需 scratch 中转的段数
        int scatter_ex = 0;
    };

    struct ubatch_acc {
        int n_tokens = 0;
        int n_used = 0;
        int n_layers = 0;
        int unique_sum = 0;
        int miss_sum = 0;
        int n_th = 0;
        int pack_layers = 0;
        int64_t wall_ns = 0;
        int64_t pread_sum_ns = 0;
        size_t bytes = 0;
        run_stats runs;
    };

    struct io_stats {
        std::atomic<uint64_t> n_miss{0};
        std::atomic<uint64_t> n_bytes{0};
        std::atomic<int64_t>  pread_ns{0};
    };

    /// 每层「下一步预测会用到的专家」列表，由上一层 arm() 写入、本层 process() 消费。
    /// 用途是**零 I/O 成本**地改进淘汰：预测会用的条目不该被换出去。
    /// 每层**被 arm 的那份预测**（保护集，见 process() 的写入处注释）
    std::vector<std::vector<int32_t>> predicted;
    /// 每层紧邻上一层给出的预测（recall 更高，只作额外淘汰提示）
    std::vector<std::vector<int32_t>> pred_hint;
    /// pred_hist[L] = 给层 L 的所有预测，元素是 (提交它的层号, 专家列表)。
    /// 因为 horizon>1 时同一层会被 K 个不同来源各预测一次（层 L-1 猜它、L-2 也猜它…），
    /// 只留最新那份就没法算 recall(k) —— 所以这里保留全部，到层 L 执行时逐条结算。
    std::vector<std::vector<std::pair<int32_t, std::vector<int32_t>>>> pred_hist;
    uint64_t recall_hit[9]  = {};
    uint64_t recall_pred[9] = {};
    std::mutex pred_mtx;

    // 整个进程的完整 I/O 计量（主路径 + 预读），见 readv_span。
    std::atomic<uint64_t> io_bytes_all{0};
    /// 预读 publish 时 slot 已不属于本专家的次数（应为 0；非 0 = slot 空间混用）
    std::atomic<uint64_t> n_slot_collision{0};
    /// horizon 被按实际块数夹紧的**本 runtime 实例累计次数**（不是全进程；全进程共享的
    /// 只有下面日志限流用的 static clamp_logs 那 16 条额度）。收尾打 horizon-clamp-summary。
    std::atomic<uint64_t> g_n_horizon_clamped{0};
    /// LLAMA_MOE_VERIFY_SLOT 的自检计数
    std::atomic<uint64_t> n_verify_ok{0};
    std::atomic<uint64_t> n_verify_bad{0};
    std::atomic<uint64_t> io_calls_all{0};
    uint64_t last_io_bytes = 0;
    uint64_t last_io_calls = 0;
    int64_t last_io_flush_ns = now_ns();
    // 「图执行期间的专家管理耗时」累计，见 add_process_ns。
    std::atomic<int64_t> process_ns{0};
    // 诊断：把 process() 内部再拆三层，用来定位「记账 3.5 → 19.5 ms」那笔异常。
    // 三者的和是 process_ns 的子集，差值就是其余部分（分配、淘汰、统计等）。
    std::atomic<int64_t> readids_ns{0};
    std::atomic<int64_t> install_ns{0};
    std::atomic<int64_t> writeids_ns{0};
    // 诊断：各层「用到的专家并集」大小之和（= 图里真正要做的 (行,专家) matvec 次数）。
    // 用来区分 ε 是「按并集大小（kernel 次数）」还是「按字节」增长。
    std::atomic<int64_t> unique_sum{0};
    // 诊断：逐层 miss 累计（只统计 decode 步）。用来判断「每层均匀 1.x 个」还是
    // 「少数层很多」—— 前者意味着 40 次串行等待，后者只需解决几层。
    std::vector<uint64_t> miss_per_layer;
    /// miss 在 selected_experts（按分数降序）里的**位置**直方图。
    /// 用来判断能否用「固定尾部预留」代替排序：
    /// 若 miss 集中在末尾几位，就把末尾几位划成第二组，求和顺序不变（可保逐字节一致）。
    /// A2：每层 MoE 的 n_expert_used 里，**末尾**这么多位置允许延后读取。
    ///
    /// 动机：miss 高度集中在分数顺序的末尾（实测 0:0.03 ... 7:0.22）。把这些位置
    /// 的缺失专家延后到「前面那些已驻留专家算完之后」再读，读就与 GPU 计算重叠了。
    /// 固定按位置切（而不是按是否驻留排序切）能保持求和顺序不变 → 输出逐字节一致。
    int32_t matmul_split = 0;

    /// 每层被延后读取的 (expert, slot)，由 phase A 写入、phase B 消费。
    std::vector<std::vector<pending_install>> deferred;

    uint64_t miss_pos[16] = {};
    uint64_t miss_pos_steps = 0;
    uint64_t miss_per_layer_steps = 0;
    int64_t last_process_ns = 0;
    uint64_t last_flush_idx = 0;
    // 预读诊断字段的增量基线，见 flush_ubatch。
    uint64_t last_arm_skipped = 0;
    uint64_t last_wait_count = 0;
    int64_t  last_wait_ns = 0;
    uint64_t last_pred_read = 0;
    uint64_t last_pf_hits = 0;
    uint64_t last_pf_pub = 0;

    explicit impl(const llama_model & model) :
        model(model),
        n_expert((int32_t) model.hparams.n_expert),
        n_slots(model.moe_slot_bank_size()),
        layers(model.layers.size()),
        predicted(model.layers.size()),
        pred_hint(model.layers.size()),
        deferred(model.layers.size()),
        miss_per_layer(model.layers.size(), 0),
        userdata(model.layers.size()) {
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto & state = layers[il];
            const auto & layer = model.layers[il];

            bind(state, layer.ffn_gate_up_exps, n_slots);
            bind(state, layer.ffn_gate_exps,    n_slots);
            bind(state, layer.ffn_up_exps,      n_slots);
            bind(state, layer.ffn_down_exps,    n_slots);

            if (state.bindings.empty()) {
                continue;
            }
            if (!layer.ffn_down_exps ||
                (!layer.ffn_gate_up_exps && (!layer.ffn_gate_exps || !layer.ffn_up_exps))) {
                throw std::runtime_error(format("incomplete MoE slot bindings for layer %zu", il));
            }

            state.enabled = true;
            if (first_moe_layer < 0) first_moe_layer = (int) il;
            finalize_pack(state);
            state.slot_to_expert.assign(n_slots, kSlotFree);
            state.expert_to_slot.assign(n_expert, kSlotFree);
            state.protected_experts.assign(n_expert, 0);
            state.pinned_experts.assign(n_expert, 0);
            state.slot_age.assign(n_slots, 0);
            // window_start 不在这里算 —— decode_window 要到 resolve_decode_window()
            // 才确定，此处还是 0。在那边统一赋值，否则所有层会共用窗口起点 [0, W)，
            // 层间互相覆盖（曾经的实际 bug）。
            userdata[il] = { nullptr, (int) il };
            n_enabled++;
            last_moe_layer = (int) il;
            if (state.pack_ok) {
                n_pack_ok++;
            }
            for (const auto & b : state.bindings) {
                if (b.tensor && b.tensor->view_src) {
                    shared_bank = true;
                    break;
                }
            }
        }
        size_t stride0 = 0;
        if (last_moe_layer >= 0) {
            stride0 = layers[(size_t) last_moe_layer].pack_bytes;
            for (const auto & layer : layers) {
                if (layer.enabled && layer.pack_bytes > 0) {
                    stride0 = layer.pack_bytes;
                    break;
                }
            }
        }
        // Experts loading preadv's the packed expert span straight into its slot, which requires both a
        // packed layout and CPU-addressable slot tensors. Do not guess either, and do not silently fall
        // back to an upload path: private Metal buffers (non-unified memory) expose tensor->data as a
        // virtual address that is not writable from the CPU.
        for (const auto & layer : layers) {
            if (layer.enabled && !layer.pack_ok) {
                throw std::runtime_error(
                    "MoE experts layer is not packed; preadv loading requires gate/up/down contiguous per expert");
            }
            for (const auto & b : layer.bindings) {
                if (!b.tensor->buffer || !ggml_backend_buffer_is_host_addressable(b.tensor->buffer)) {
                    throw std::runtime_error(format(
                        "MoE slot tensor '%s' is not CPU-addressable; preadv loading requires host-visible slots",
                        ggml_get_name(b.tensor)));
                }
            }
        }

        moe_log("init pack=%s pack_layers=%d/%d bank=%d shared=%s experts=%d io_threads=%d stride0=%zu preadv=on park_window=%s",
            n_pack_ok == n_enabled && n_enabled > 0 ? "on" : (n_pack_ok > 0 ? "partial" : "off"),
            n_pack_ok, n_enabled, (int) n_slots, shared_bank ? "on" : "off",
            (int) n_expert, moe_ssd_io_threads(), stride0,
            park_window_enabled() ? "on" : "off");

        // recall(k) 的预测历史：按层存「谁在哪层提交的预测」，到该层执行时逐条结算。
        // 在 init() 里定容，避免塞进 ctor 的初始化列表（那边字段顺序很挤，加一项会触发 -Wreorder-ctor）。
        pred_hist.assign(layers.size(), {});

        if (moe_o_direct_want()) {
            std::string probe_path;
            for (const auto & st : layers) {
                if (!st.enabled) continue;
                probe_path = !st.dec_pack_path.empty() ? st.dec_pack_path : st.pack_path;
                if (!probe_path.empty()) break;
            }
            if (!probe_path.empty() && probe_o_direct(probe_path)) {
                g_o_direct_ok.store(true, std::memory_order_relaxed);
            }
        }

        lookahead.init((int32_t) n_slots, n_expert, last_moe_layer,
            [this](int layer, int expert, int slot, bool to_window, llama_moe::slot_manager::ticket request) {
                if (to_window && layers[(size_t) layer].manager) return install_managed_prefetch(layer, expert, request);
                if (to_window) {
                    return install_prefetch_into_window(layer, expert);
                }
                read_expert_directv(layers[(size_t) layer], expert, slot);
                return lookahead_expert_prefetch::read_result::published;
            }, handoff_trace.enabled() ? &handoff_trace : nullptr);
        resolve_decode_window();
        {
            const char * e = std::getenv("LLAMA_MOE_MATMUL_SPLIT");
            matmul_split = e ? std::atoi(e) : 0;
            if (matmul_split > 0) {
                moe_log("matmul-split: 末尾 %d 个位置的缺失专家延后读取（读与算重叠）",
                        (int) matmul_split);
            }
        }

        // decode 窗口那一套绑定要等 resolve_decode_window() —— 它依赖 shared_bank
        // 和 last_moe_layer，两者都在上面的层循环里才确定。
        if (decode_window > 0) {
            // 模型侧每层张量的宽度是 c_il + K（缓存区 + 预读暂存区），**按层不同**。
            // 见 qwen35moe.cpp 与 install_prefetch_into_window 的分区说明。
            int n_dec = 0;
            std::string fail;
            for (size_t il = 0; il < model.layers.size() && fail.empty(); ++il) {
                auto & state = layers[il];
                if (!state.enabled) {
                    continue;
                }
                const auto & layer = model.layers[il];
                const std::pair<ggml_tensor *, const char *> fam[4] = {
                    { layer.ffn_gate_up_exps_dec, "gate_up" },
                    { layer.ffn_gate_exps_dec,    "gate"    },
                    { layer.ffn_up_exps_dec,      "up"      },
                    { layer.ffn_down_exps_dec,    "down"    },
                };
                for (int k = 0; k < 4 && fail.empty(); ++k) {
                    if (!fam[k].first) {
                        continue;
                    }
                    try {
                        const int32_t expect = layer_width[il] > 0 ? layer_width[il]
                                                                   : decode_window + prefetch_slots;
                        bind(state, fam[k].first, expect, &state.dec_bindings);
                    } catch (const std::exception & e) {
                        ggml_tensor * t = fam[k].first;
                        fail = format("layer %zu family %s: name=%s ne2=%lld nbytes=%zu "
                                      "win=%d :: %s",
                                      il, fam[k].second, ggml_get_name(t),
                                      (long long) t->ne[2], ggml_nbytes(t),
                                      (int) decode_window, e.what());
                    }
                }
                if (fail.empty() && state.dec_bindings.empty()) {
                    fail = format("layer %zu has no decode tensors", il);
                }
                if (fail.empty()) {
                    finalize_pack(state.dec_bindings, state.dec_pack_order, state.dec_pack_ok,
                                  state.dec_pack_base, state.dec_pack_stride, state.dec_pack_path);
                    n_dec++;
                }
            }
            if (!fail.empty()) {
                // 绑不上就退回共享 bank（那是正确路径），但必须**大声**报出来 ——
                // 静默回退会让人以为窗口生效了，测出来的数是假的。
                moe_log("decode-cache: 绑定失败，回退到共享 bank（窗口关闭）。原因: %s",
                        fail.c_str());
                for (auto & st : layers) {
                    st.dec_bindings.clear();
                    st.dec_pack_order.clear();
                    st.dec_pack_ok = false;
                }
                decode_window = 0;
            } else {
                moe_log("decode-cache: %d 层绑定 decode 张量（每层 %d = cache %d + prefetch %d slot，pack %s）",
                        n_dec, (int) (decode_window + prefetch_slots), (int) decode_window,
                        (int) prefetch_slots, layers[0].dec_pack_ok ? "ok" : "off");
            }
        }
        sweep_init();
        prewarm_init();
        try { init_slot_managers(); init_io_admission(); init_h1_front(); init_prefetch_refresh(); }
        catch (...) { lookahead.stop(); if (io_budget) io_budget->close(); throw; }
    }

    bool managers_active() const {
        return first_moe_layer >= 0 && layers[(size_t) first_moe_layer].manager != nullptr;
    }

    void init_slot_managers() {
        pf_probation = pf_probation_enabled();
        if (pf_probation && (!slot_manager_enabled() || decode_window <= 0 || matmul_split > 0 || first_moe_layer < 0)) {
            throw std::runtime_error("LLAMA_MOE_PF_PROBATION requires LLAMA_MOE_EXPERT_READY=1, decode window and matmul_split=0");
        }
        if (!slot_manager_enabled() || decode_window <= 0 || matmul_split > 0) return;
        for (auto & st : layers) {
            if (!st.enabled) continue;
            if (st.window_start != 0) throw std::runtime_error("managed MoE requires layer-local window indices");
            const int32_t persistent = (int32_t) std::count(st.pinned_experts.begin(), st.pinned_experts.end(), (uint8_t) 1);
            const int32_t cache = st.window_width - prefetch_slots;
            const int32_t worst_selected = std::min(n_expert - persistent,
                window_max_rows * (int32_t) model.hparams.n_expert_used);
            // Keep the existing K writer reserve even for a unified window.
            // Strict permanent pins cannot consume the demand safety margin.
            if (persistent + worst_selected > cache) {
                throw std::runtime_error(format("slot-manager: permanent pin budget %d plus worst selected %d exceeds cache %d; reduce prewarm pins",
                    persistent, worst_selected, cache));
            }
            st.manager.reset(new llama_moe::slot_manager(st.mutex, n_expert, st.window_width,
                st.window_width - prefetch_slots, prefetch_unified, prefetch_slots, pf_probation));
            st.requests.resize((size_t) n_expert);
            for (int32_t e = 0; e < n_expert; ++e) st.manager->set_persistent(e, st.pinned_experts[(size_t) e] != 0);
        }
        lookahead.enqueue_fn = [this](int layer, int expert, uint64_t epoch, uint64_t job, int rows) {
            auto & st = layers[(size_t) layer];
            std::lock_guard<std::mutex> lock(st.mutex);
            const auto r = st.manager->enqueue(expert, {epoch, job});
            const auto context = handoff_request({job, epoch, layer, rows, "prefetch", 0, "window"}, r.request);
            handoff_trace.emit(r.action == llama_moe::slot_manager::outcome::created ? "REQUEST_ENQUEUED" :
                               r.action == llama_moe::slot_manager::outcome::coalesced ? "ENQUEUE_COALESCE" : "ENQUEUE_SELECTED_SKIP",
                               context, expert, r.request.slot, (int32_t) r.action);
            if (r.action == llama_moe::slot_manager::outcome::coalesced) ++manager_coalesced;
            return r;
        };
        lookahead.cancel_fn = [this](int layer, llama_moe::slot_manager::ticket request) {
            auto & st = layers[(size_t) layer];
            std::lock_guard<std::mutex> lock(st.mutex);
            const bool cancelled = st.manager->cancel_queued(request);
            if (cancelled) handoff_trace.emit("REQUEST_CANCELLED",
                handoff_request({0, 0, layer, 0, "control", 0, "window"}, request), request.expert, request.slot);
            return cancelled;
        };
    }

    struct io_trace_payload {
        moe_handoff_context context;
        int32_t slot = -1, needed_count = 0;
    };

    static void observe_io_admission(void * data, const llama_moe::io_admission::observation & event) noexcept {
        auto & owner = *static_cast<impl *>(data);
        const auto & payload = *static_cast<const io_trace_payload *>(event.payload);
        auto context = payload.context;
        if (event.request.expert >= 0) {
            context.layer = event.request.layer; context.binding = event.request.binding; context.generation = event.request.generation;
        }
        using action = llama_moe::io_admission::action;
        const char * name = event.event == action::wait ? "IO_ADMIT_WAIT" : event.event == action::grant ? "IO_ADMIT_GRANT" :
            event.event == action::release ? "IO_ADMIT_RELEASE" : event.event == action::cancel ? "IO_ADMIT_CANCEL" :
            event.event == action::boost ? "IO_ADMIT_BOOST" : event.event == action::needed_begin ? "IO_NEEDED_SCOPE_BEGIN" : "IO_NEEDED_SCOPE_END";
        moe_handoff_detail detail;
        detail.admit_seq = event.sequence;
        detail.admit_class = event.category == llama_moe::io_admission::priority::background ? "background" :
            event.category == llama_moe::io_admission::priority::needed_prefetch ? "needed_prefetch" : "demand";
        detail.admit_active = event.active; detail.admit_waiters = event.waiting; detail.admit_capacity = event.capacity;
        owner.handoff_trace.emit(name, context, event.request.expert, payload.slot, payload.needed_count, 0, {}, detail);
    }

    void init_io_admission() {
        const uint32_t limit = expert_io_budget();
        if (!limit) return;
        if (!managers_active() || matmul_split > 0) {
            throw std::runtime_error("LLAMA_MOE_EXPERT_IO_BUDGET requires LLAMA_MOE_EXPERT_READY=1, decode window and matmul_split=0");
        }
        io_needed_snapshot.resize((size_t) n_expert);
        io_budget.reset(new llama_moe::io_admission(limit, (size_t) n_expert,
            handoff_trace.enabled() ? observe_io_admission : nullptr, this));
        moe_log("io-budget: enabled=1 limit=%u scope=ready-window", limit);
    }

    void init_h1_front() {
        h1_front = h1_front_enabled();
        if (!h1_front) return;
        if (!managers_active() || matmul_split > 0 || !lookahead.enabled || !lookahead.window_mode || lookahead.read_n <= 0 || io_budget) {
            throw std::runtime_error("LLAMA_MOE_PREFETCH_H1_FRONT requires Ready window, split=0, active prefetch/read>0 and IO budget=0");
        }
        moe_log("h1-front: enabled=1 first_moe_layer=%d (adds H1 only when actual horizon>1)", first_moe_layer);
    }

    void init_prefetch_refresh() {
        if (!lookahead.refresh) return;
        if (!managers_active() || matmul_split > 0 || !lookahead.enabled || !lookahead.window_mode ||
            lookahead.read_n <= 0 || moe_lookahead_horizon() <= 1 || io_budget || h1_front) {
            throw std::runtime_error("LLAMA_MOE_PREFETCH_REFRESH requires Ready window, split=0, horizon>1, prefetch/read>0 and IO budget/H1 front=0");
        }
        moe_log("prefetch-refresh: enabled=1 reserve=%d read=%d deadline=epoch-layer", std::min(2, lookahead.read_n), lookahead.read_n);
    }

    int64_t read_managed_expert(layer_state & state, llama_moe::slot_manager::ticket request,
                               llama_moe::io_admission::origin origin, bool retry = false) {
        // Policy uses explicit immutable owner/key values even when tracing is
        // disabled. The trace payload is diagnostic only and outlives the lease.
        auto read = [&]() {
            ++manager_reads;
            if (origin == llama_moe::io_admission::origin::demand) ++manager_demand_reads; else ++manager_prefetch_reads;
            if (retry) ++manager_retry_reads;
            return read_expert_directv(state, request.expert, request.slot, true);
        };
        if (!io_budget) return read();
        const io_trace_payload payload{handoff_thread.context, request.slot, 0};
        auto permit = io_budget->acquire({(int32_t) (&state - layers.data()), request.expert,
            request.binding, request.generation}, origin, &payload);
        // permit unwinds before callers take the layer mutex to publish/fail.
        return read();
    }

    int32_t cached_slot(layer_state & st, int32_t expert, bool window) {
        return window && st.manager ? st.manager->resident_slot(expert) : st.expert_to_slot[(size_t) expert];
    }

    void trace_managed_claim(layer_state & st, const llama_moe::slot_manager::result & r, moe_handoff_context context) {
        if (r.victim) {
            ++manager_evicted;
            handoff_trace.emit("SLOT_EVICT", handoff_request(context, r.victim), r.victim.expert, r.victim.slot);
            if (r.probation_victim) {
                moe_handoff_detail detail; detail.reason = "unused_prefetch";
                handoff_trace.emit("PF_PROBATION_VICTIM", handoff_request(context, r.victim), r.victim.expert, r.victim.slot, 0, 0, {}, detail);
            }
        }
        if (r.action == llama_moe::slot_manager::outcome::promoted) {
            ++manager_promoted;
            moe_handoff_detail detail; detail.related_generation = r.previous.generation;
            handoff_trace.emit("QUEUED_PROMOTION", handoff_request(context, r.request), r.request.expert, r.request.slot, 0, 0, {}, detail);
        }
        if (r.owner) {
            handoff_trace.emit("SLOT_CLAIM", handoff_request(context, r.request), r.request.expert, r.request.slot);
        }
        GGML_UNUSED(st);
    }

    bool finish_managed(layer_state & st, llama_moe::slot_manager::ticket request,
                        moe_handoff_context context, bool success, std::exception_ptr error = {}) {
        const bool accepted = success ? st.manager->complete(request) : st.manager->fail(request, error);
        handoff_trace.emit(success ? "SLOT_READY" : "SLOT_FAILED", handoff_request(context, request),
                           request.expert, request.slot, accepted ? 0 : -1);
        if (!success && accepted) ++manager_failed;
        if (success && !accepted) ++n_slot_collision;
        return accepted;
    }

    lookahead_expert_prefetch::read_result install_managed_prefetch(int layer, int32_t expert, llama_moe::slot_manager::ticket request) {
        auto & st = layers[(size_t) layer];
        auto context = handoff_request(handoff_thread.context, request);
        handoff_trace.emit("PF_CLAIM_WAIT", context, expert);
        llama_moe::slot_manager::result r;
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            handoff_trace.emit("PF_CLAIM_LOCKED", context, expert);
            r = st.manager->start_queued(request);
            context = handoff_request(context, r.request ? r.request : request);
            handoff_trace.emit("PF_CLAIM_RESULT", context, expert, r.request.slot, (int32_t) r.action);
            if (r.owner) trace_managed_claim(st, r, context);
        }
        if (!r.owner) {
            if (r.action == llama_moe::slot_manager::outcome::stale) {
                ++manager_stale; handoff_trace.emit("CANCELLED_STALE_TASK", context, expert);
            } else {
                moe_handoff_detail detail;
                const bool quota = r.action == llama_moe::slot_manager::outcome::writer_limit;
                detail.reason = quota ? "writer_limit" : "capacity";
                if (quota) ++manager_quota_skipped; else ++manager_capacity_skipped;
                handoff_trace.emit("REQUEST_CANCELLED", context, expert, -1, (int32_t) r.action, 0, {}, detail);
            }
            return lookahead_expert_prefetch::read_result::skipped;
        }
        context.io_id = handoff_trace.new_io_id();
        moe_handoff_scope scope(handoff_trace.enabled() ? &handoff_trace : nullptr, context, expert, r.request.slot);
        handoff_trace.emit("IO_OWNER", context, expert, r.request.slot, (int32_t) llama_moe::slot_manager::origin::prefetch);
        handoff_trace.emit("PF_READ_SUBMIT", context, expert, r.request.slot);
        try {
            read_managed_expert(st, r.request, llama_moe::io_admission::origin::prefetch);
        } catch (...) {
            std::lock_guard<std::mutex> lock(st.mutex);
            finish_managed(st, r.request, context, false, std::current_exception());
            throw;
        }
        handoff_trace.emit("PF_PUBLISH_WAIT", context, expert, r.request.slot);
        std::lock_guard<std::mutex> lock(st.mutex);
        handoff_trace.emit("PF_PUBLISH_LOCKED", context, expert, r.request.slot);
        const bool accepted = finish_managed(st, r.request, context, true);
        if (accepted) ++st.n_prefetch_published;
        handoff_trace.emit("PF_PUBLISH_RESULT", context, expert, r.request.slot, accepted ? 1 : 0);
        return accepted ? lookahead_expert_prefetch::read_result::published : lookahead_expert_prefetch::read_result::discarded;
    }

    /// 预读专用：把一个 expert 读进**目标层自己的窗口**（按窗口 LRU 挑 slot）。
    ///
    /// 与 process() 的安装路径共用同一套 slot 选择规则，所以预读的成果对那层来说
    /// 就是一次普通的缓存命中 —— 不需要乒乓区、也不需要事后搬运。
    ///
    /// 调用时机保证安全：预读在层 L-1 的 process() 里 arm、在层 L 的 process()
    /// 开头 take() 处等它结束，所以 worker 动层 L 的窗口时，层 L 没有在跑。
    /// 清空一层缓存的映射表。
    ///
    /// 保留 `kSlotPending`：那是预读 worker 已认领、正在写的 slot。正常路径上
    /// 清空只发生在 prefill→decode 边界（那时没有 worker 在飞），但保留判断更安全
    /// —— 否则会把 worker 的位置标成空闲，主路径分配过去就撞车了。
    static void clear_layer_cache(layer_state & st) {
        for (auto & e : st.slot_to_expert) {
            if (e != kSlotPending) {
                e = kSlotFree;
            }
        }
        // expert_to_slot 不能整体清：pending 的专家此时还没登记，所以直接全清是安全的。
        std::fill(st.expert_to_slot.begin(), st.expert_to_slot.end(), kSlotFree);
    }

    /// 预读：把一个预测专家读进**目标层自己的窗口**，与缓存共用同一片 slot 池。
    ///
    /// 关键设计：**锁内只做「认领 slot」，读盘在锁外，读完再锁内「发布」**。
    ///
    /// 为什么必须这样：以前这个函数从头到尾持着 `st.mutex` 读盘（每次读 ~1ms），
    /// 而主线程的 `process()` 要拿同一把锁 —— 于是主线程被 worker 的读盘挡住，
    /// 实测每 token 白等 43ms。**这才是「缓存与预读互斥」的真正原因**，
    /// 不是两者本质上冲突。
    ///
    /// 认领期间该 slot 标成 `kSlotPending`：主路径分配/淘汰都会跳过它，
    /// 所以主线程既不会覆盖它、也不会误用未写完的内容。
    /// 认领一个槽位并标 `kSlotPending`（内容尚未写入）。**调用方必须已持 st.mutex**。
    ///
    /// `cache_only` 决定从哪一段挑：
    ///   true  → 只在**缓存区** [window_start, window_start + c_il) 里挑（预热用；
    ///           预热的是「会跨 token 复用」的专家，本就该进缓存区）
    ///   false → 在**预读区**里挑（预读用；统一模式下就是整层窗口）
    ///
    /// 返回 -1 = 该专家已在窗口里，或无处可放。
    int32_t claim_slot_in_window(layer_state & st, int32_t expert, bool cache_only,
                                 const moe_handoff_context * trace_context = nullptr) {
        if (decode_window <= 0 || expert < 0 || expert >= n_expert) {
            return -1;
        }
        if (st.expert_to_slot[(size_t) expert] >= 0) {
            return -1;   // 已经在（缓存区或预读区）—— 白读，跳过
        }
        const int32_t w_il = st.window_width > 0 ? st.window_width : (decode_window + prefetch_slots);
        const int32_t c_il = std::max<int32_t>(1, w_il - prefetch_slots);
        const int32_t lo = st.window_start;
        const int32_t hi = cache_only ? st.window_start + c_il
                                      : st.window_start + w_il;
        const int32_t from = cache_only ? lo
                                        : (prefetch_unified ? lo : st.window_start + c_il);

        int32_t slot = -1;
        for (int32_t cand = from; cand < hi; ++cand) {
            if (st.slot_to_expert[(size_t) cand] == kSlotFree) {
                slot = cand;
                break;
            }
        }
        if (slot < 0) {
            uint64_t oldest = UINT64_MAX;
            for (int32_t cand = from; cand < hi; ++cand) {
                if (st.slot_to_expert[(size_t) cand] == kSlotPending) {
                    continue;
                }
                const int32_t resident = st.slot_to_expert[(size_t) cand];
                if (resident >= 0 && (st.protected_experts[(size_t) resident] ||
                                      st.pinned_experts[(size_t) resident])) continue;
                if (st.slot_age[(size_t) cand] < oldest) {
                    oldest = st.slot_age[(size_t) cand];
                    slot = cand;
                }
            }
        }
        if (slot < 0) {
            return -1;
        }
        const int32_t evicted = st.slot_to_expert[(size_t) slot];
        if (trace_context && evicted >= 0) handoff_trace.emit("LEGACY_SLOT_EVICT", *trace_context, evicted, slot);
        if (evicted >= 0) {
            st.expert_to_slot[(size_t) evicted] = kSlotFree;
        }
        st.slot_to_expert[(size_t) slot] = kSlotPending;   // 认领，内容尚未写入
        return slot;
    }

    /// 读完后登记为有效缓存条目。**调用方必须已持 st.mutex**。
    /// 返回是否真的登记（专家在读盘期间被主路径抢先装好时不登记，丢弃这份）。
    bool publish_slot_in_window(layer_state & st, int32_t expert, int32_t slot) {
        // 读盘期间主路径可能已经把同一个专家装好了（它自己读的），那就丢弃这份。
        if (st.expert_to_slot[(size_t) expert] >= 0) {
            st.slot_to_expert[(size_t) slot] = kSlotFree;
            return false;
        }
        // 不变量：publish 的这个 slot 必须仍是**本专家**认领的 kSlotPending。
        // 如果它变成了别的专家的有效条目（slot 空间混用 / 认领被抢），
        // 这里强行写就会覆盖别人的数据 —— 计一次碰撞并放弃，宁可不缓存。
        // 这条检查是评审指出的「乒乓格号被当成窗口偏移」那一类错误的运行时探针。
        if (st.slot_to_expert[(size_t) slot] != kSlotPending) {
            n_slot_collision++;
            return false;
        }
        st.slot_to_expert[(size_t) slot] = expert;
        st.expert_to_slot[(size_t) expert] = slot;
        st.slot_age[(size_t) slot] = ++st.age;
        st.n_prefetch_published++;
        return true;
    }

    static int prefill_warm_rows() {
        const char * value = std::getenv("LLAMA_MOE_PREFILL_WARM_ROWS");
        return value ? std::max(0, std::min(256, std::atoi(value))) : 0;
    }

    // Preserve recently used prefill experts in the physically separate decode
    // window. The existing prewarm placement path republishes their mappings
    // when decode begins; no extra file reads or expert allocations are needed.
    // Called with the layer mutex held, after all shared-bank reads completed.
    void warm_from_prefill(layer_state & st, int layer, const std::vector<int32_t> & experts,
                           const std::vector<int32_t> & slots, int n_rows) {
        // Copying into overlapping decode views would corrupt live prefill weights.
        if (model.moe_phase_reuse()) return;
        const int rows = std::min(prefill_warm_rows(), n_rows);
        if (rows <= 0 || !shared_bank || decode_window <= 0 || st.dec_bindings.empty()) return;
        const int capacity = st.window_width - prefetch_slots;
        const size_t ranks = experts.size() / n_rows;
        if (capacity <= 0 || ranks == 0 || experts.size() != slots.size()) return;
        if (st.manager && (st.manager->writers() || st.manager->pins())) return;

        using pack_fn = size_t (*)(void *, const void *, int, int64_t, int64_t);
        struct family_copy { const binding * src; const binding * dst; size_t bytes; };
        std::vector<family_copy> families;
        for (const auto & src : st.bindings) {
            auto dst = std::find_if(st.dec_bindings.begin(), st.dec_bindings.end(),
                [&](const binding & b) { return b.entry == src.entry; });
            if (dst == st.dec_bindings.end()) throw std::runtime_error("prefill warm family missing");
            auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(src.tensor->buffer));
            auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            auto pack = reg ? (pack_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_pack_moe_expert") : nullptr;
            // This path is specific to the verified Hexagon tiled layout.
            if (!pack) return;
            if (src.tensor->type != dst->tensor->type || src.tensor->ne[0] != dst->tensor->ne[0] ||
                src.tensor->ne[1] != dst->tensor->ne[1] ||
                !ggml_backend_buffer_is_host_addressable(src.tensor->buffer) ||
                !ggml_backend_buffer_is_host_addressable(dst->tensor->buffer)) {
                throw std::runtime_error("prefill warm incompatible family");
            }
            const size_t bytes = pack(nullptr, nullptr, src.tensor->type, src.tensor->ne[0], src.tensor->ne[1]);
            if (!bytes) throw std::runtime_error("prefill warm unsupported type");
            const uintptr_t src_begin = (uintptr_t) src.tensor->data;
            const uintptr_t dst_begin = (uintptr_t) dst->tensor->data;
            if (src_begin < dst_begin + bytes * dst->tensor->ne[2] &&
                dst_begin < src_begin + bytes * src.tensor->ne[2]) {
                throw std::runtime_error("prefill and decode warm buffers overlap");
            }
            families.push_back({ &src, &*dst, bytes });
        }

        const int64_t start = now_ns();
        std::vector<uint8_t> seen((size_t) n_expert, 0);
        std::vector<prewarm_place> placed;
        size_t copied = 0, verified = 0;
        const bool verify = verify_slot_enabled();
        const char * rank_order = std::getenv("LLAMA_MOE_PREFILL_WARM_RANK_FIRST");
        const bool rank_first = rank_order && *rank_order && *rank_order != '0';
        // Slot selection only: required experts and routing weights are unchanged.
        // Default traversal preserves the previous reverse-row order exactly.
        for (size_t candidate = 0; candidate < (size_t) rows * ranks &&
                (int) placed.size() < capacity; ++candidate) {
                const int row = n_rows - 1 - (int) (rank_first ? candidate % rows : candidate / ranks);
                const size_t rank = rank_first ? candidate / rows : candidate % ranks;
                const size_t index = (size_t) row * ranks + rank;
                const int expert = experts[index];
                if (seen[(size_t) expert]) continue;
                seen[(size_t) expert] = 1;
                int dst_slot = (int) placed.size();
                llama_moe::slot_manager::result managed;
                auto warm_context = handoff_thread.context; warm_context.source = "prewarm"; warm_context.domain = "window";
                if (st.manager) {
                    managed = st.manager->acquire(expert, {graph_epoch, 0}, llama_moe::slot_manager::origin::prewarm, true);
                    if (!managed.request) continue;
                    dst_slot = managed.request.slot;
                    if (!managed.owner) { placed.push_back({layer, expert, dst_slot}); continue; }
                    trace_managed_claim(st, managed, warm_context);
                }
                try {
                for (const auto & family : families) {
                    if (slots[index] < 0 || slots[index] >= family.src->tensor->ne[2] ||
                        dst_slot >= family.dst->tensor->ne[2]) {
                        throw std::runtime_error("prefill warm slot out of range");
                    }
                    std::memcpy((uint8_t *) family.dst->tensor->data + dst_slot * family.bytes,
                                (uint8_t *) family.src->tensor->data + slots[index] * family.bytes,
                                family.bytes);
                    copied += family.bytes;
                }
                if (verify) {
                    const char * packed_dir = std::getenv("LLAMA_MOE_HTP_PACKED_DIR");
                    if (!packed_dir || !*packed_dir || !st.dec_pack_ok || st.dec_pack_path.empty() ||
                        st.dec_pack_order.empty() || st.dec_pack_order.size() > 8 ||
                        st.dec_pack_order.size() != families.size()) {
                        throw std::runtime_error("prefill warm verification requires offline tiled experts");
                    }
                    struct iovec iov[8];
                    size_t stride = 0;
                    for (size_t i = 0; i < st.dec_pack_order.size(); ++i) {
                        const auto & dst = st.dec_bindings.at(st.dec_pack_order[i]);
                        const auto family = std::find_if(families.begin(), families.end(),
                            [&](const family_copy & f) { return f.dst == &dst; });
                        if (family == families.end()) throw std::runtime_error("prefill warm verification family missing");
                        iov[i] = { (uint8_t *) dst.tensor->data + dst_slot * family->bytes, family->bytes };
                        stride += family->bytes;
                    }
                    const size_t slash = st.dec_pack_path.find_last_of("/");
                    const std::string path = std::string(packed_dir) + "/" +
                        st.dec_pack_path.substr(slash == std::string::npos ? 0 : slash + 1);
                    if (!verify_slot_bytes(path, (size_t) expert * stride, iov, (int) st.dec_pack_order.size())) {
                        throw std::runtime_error("prefill warm destination verification failed");
                    }
                    verified += stride;
                }
                if (st.manager) finish_managed(st, managed.request, warm_context, true);
                } catch (...) {
                    if (verify) moe_log("prefill-warm-verify: layer=%d copied=%zu verified=%zu bad=%zu",
                            layer, copied, verified, copied - verified);
                    if (st.manager && managed.owner) finish_managed(st, managed.request, warm_context, false, std::current_exception());
                    throw;
                }
                placed.push_back({ layer, expert, dst_slot });
        }
        prewarm_placed.erase(std::remove_if(prewarm_placed.begin(), prewarm_placed.end(),
            [&](const prewarm_place & p) { return p.layer == layer; }), prewarm_placed.end());
        prewarm_placed.insert(prewarm_placed.end(), placed.begin(), placed.end());
        prewarm_ready = !prewarm_placed.empty();
        moe_log("prefill-warm: layer=%d rows=%d rank_first=%d experts=%zu copied=%.1fMiB wall=%.3fms",
                layer, rows, (int) rank_first, placed.size(), mib_of(copied), ms_of(now_ns() - start));
        if (verify) moe_log("prefill-warm-verify: layer=%d copied=%zu verified=%zu bad=%zu",
                layer, copied, verified, copied - verified);
    }

    lookahead_expert_prefetch::read_result install_prefetch_into_window(int layer, int32_t expert) {
        if (layer < 0 || layer >= (int) layers.size() || decode_window <= 0 || prefetch_slots <= 0) {
            return lookahead_expert_prefetch::read_result::skipped;
        }
        auto & st = layers[(size_t) layer];
        if (!st.enabled || expert < 0 || expert >= n_expert) {
            return lookahead_expert_prefetch::read_result::skipped;
        }

        // 只在**预读暂存区** [window_start + c, window_start + c + K) 里分配。
        // 这一块每步重填，不参与跨 token 的 LRU 淘汰 —— 所以预读**不会**把缓存的
        // 条目换出去。这是「缓存 + 预读」并集覆盖的前提：
        //   预测到的专家已在缓存区 -> 白读（跳过，见下面的检查）
        //   不在缓存区 -> 读进暂存区，本层用时不落关键路径
        // 统一模式下预读可写在整层窗口里（见 prefetch_unified 的说明）。
        auto context = handoff_thread.context; context.io_id = handoff_trace.new_io_id();
        handoff_trace.emit("PF_CLAIM_WAIT", context, expert);
        int32_t slot = -1;
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            handoff_trace.emit("PF_CLAIM_LOCKED", context, expert);
            slot = claim_slot_in_window(st, expert, /*cache_only=*/false, handoff_trace.enabled() ? &context : nullptr);
            handoff_trace.emit("PF_CLAIM_RESULT", context, expert, slot, slot >= 0 ? 1 : 0);
            if (slot >= 0) handoff_trace.emit("LEGACY_SLOT_CLAIM", context, expert, slot);
        }
        if (slot < 0) {
            return lookahead_expert_prefetch::read_result::skipped;
        }

        // 锁外读盘：这一步不再阻塞主线程。
        moe_handoff_scope scope(handoff_trace.enabled() ? &handoff_trace : nullptr, context, expert, slot);
        handoff_trace.emit("IO_OWNER", context, expert, slot, (int32_t) llama_moe::slot_manager::origin::prefetch);
        try { read_expert_directv(st, expert, slot, true); }
        catch (...) { handoff_trace.emit("LEGACY_SLOT_FAILED", context, expert, slot, -1); throw; }

        handoff_trace.emit("PF_PUBLISH_WAIT", context, expert, slot);
        std::lock_guard<std::mutex> lock(st.mutex);
        handoff_trace.emit("PF_PUBLISH_LOCKED", context, expert, slot);
        const bool published = publish_slot_in_window(st, expert, slot);
        handoff_trace.emit("PF_PUBLISH_RESULT", context, expert, slot, published ? 1 : 0);
        handoff_trace.emit(published ? "LEGACY_SLOT_READY" : "LEGACY_SLOT_DISCARDED", context, expert, slot);
        return published ? lookahead_expert_prefetch::read_result::published : lookahead_expert_prefetch::read_result::discarded;
    }

    /// ── 缓存预热（LLAMA_MOE_PREWARM）───────────────────────────────────
    ///
    /// 每轮生成开始时各层窗口都是空的（进窗口模式时整表清空 / park 后又清）。
    /// 预热把一张**静态表**（每层若干专家 id）读进各层 decode 窗口，让第一步不再是
    /// 「全 miss」。
    ///
    /// 关键：decode 窗口与 prefill 段**物理分离**（见 qwen35moe.cpp 的 base 布局），
    /// 所以只要在 prefill 之前把数据读进去，prefill 不会覆盖它，阶段切换时只需
    /// **重新发布映射**（`publish_prewarm_placed`，纯整数操作），不必重读盘。
    /// 这就是 App 侧「mmproj 卸载后、prefill 之前」调用 `prewarm_experts()` 的意义：
    /// 那笔读完全离开首 token 的关键路径。
    ///
    /// 三段式（与 install_prefetch_into_window 同构，但批量做）：
    ///   1. 认领：逐层持锁，给每个专家在**缓存区**找空位 / 淘汰最老的；
    ///   2. 读盘：不持任何层锁，用 io_threads 个线程并行读；
    ///   3. 发布：逐层持锁，并记下落位（层,专家,slot）供阶段切换重挂。
    ///
    /// `cur_layer` 的锁**已被调用方（process）持有**时传该层号；从公开 API 调用
    /// （此时不持任何层锁）传 -1。`std::mutex` 不可重入，再锁会自死锁。
    void prewarm_managed(int cur_layer) {
        if (!prewarm_enabled) return;
        struct item { int32_t layer; llama_moe::slot_manager::ticket request; std::exception_ptr error; };
        size_t count = 0;
        for (const auto & ids : prewarm_table) count += ids.size();
        std::vector<item> pending; pending.reserve(count);
        const uint64_t epoch = graph_epoch;
        for (size_t layer = 0; layer < layers.size() && layer < prewarm_table.size(); ++layer) {
            auto & st = layers[layer]; if (!st.manager) continue;
            auto claim = [&]() {
                for (int32_t expert : prewarm_table[layer]) {
                    const auto r = st.manager->acquire(expert, {epoch, 0}, llama_moe::slot_manager::origin::prewarm, true);
                    if (!r.owner) continue;
                    trace_managed_claim(st, r, {0, epoch, (int32_t) layer, 0, "prewarm", 0, "window"});
                    pending.push_back({(int32_t) layer, r.request, {}});
                }
            };
            if ((int) layer == cur_layer) claim();
            else { std::lock_guard<std::mutex> lock(st.mutex); claim(); }
        }
        std::atomic<size_t> next{0};
        auto read = [&]() {
            for (;;) {
                const size_t i = next.fetch_add(1, std::memory_order_relaxed); if (i >= pending.size()) return;
                auto & item = pending[i]; auto & st = layers[(size_t) item.layer];
                const auto context = handoff_request({0, epoch, item.layer, 0, "prewarm", 0, "window"}, item.request);
                moe_handoff_scope scope(handoff_trace.enabled() ? &handoff_trace : nullptr, context, item.request.expert, item.request.slot);
                handoff_trace.emit("IO_OWNER", context, item.request.expert, item.request.slot, (int32_t) llama_moe::slot_manager::origin::prewarm);
                try { ++manager_reads; ++manager_prewarm_reads; read_expert_directv(st, item.request.expert, item.request.slot, true); }
                catch (...) { item.error = std::current_exception(); }
            }
        };
        const int threads = std::max(1, std::min(moe_ssd_io_threads(), (int) pending.size()));
        if (threads == 1) read();
        else {
            bool pooled = false;
            try { pooled = io_pool.run(threads, read); }
            catch (...) { /* Submission did not start; retain synchronous fallback. */ }
            if (!pooled) {
            std::vector<std::thread> helpers;
            try { helpers.reserve((size_t) threads); for (int i = 0; i < threads; ++i) helpers.emplace_back(read); }
            catch (...) { read(); }
            for (auto & helper : helpers) helper.join();
            }
        }
        size_t completed = 0;
        for (auto & item : pending) {
            auto & st = layers[(size_t) item.layer];
            auto finish = [&]() {
                const auto context = handoff_request({0, epoch, item.layer, 0, "prewarm", 0, "window"}, item.request);
                if (finish_managed(st, item.request, context, !item.error, item.error) && !item.error) ++completed;
            };
            if (item.layer == cur_layer) finish(); else { std::lock_guard<std::mutex> lock(st.mutex); finish(); }
        }
        prewarm_ready = true;
        moe_log("managed-prewarm: requested=%zu own=%zu completed=%zu", count, pending.size(), completed);
    }

    void prewarm_windows(int cur_layer) {
        if (managers_active()) { prewarm_managed(cur_layer); return; }
        if (!prewarm_enabled || decode_window <= 0) {
            return;
        }
        const int64_t t0 = now_ns();
        struct flat { int layer; int32_t expert; int32_t slot; };

        std::vector<std::vector<std::pair<int32_t, int32_t>>> tasks(layers.size());
        size_t n_req = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (il >= prewarm_table.size() || prewarm_table[il].empty()) {
                continue;
            }
            auto & st = layers[il];
            if (!st.enabled) {
                continue;
            }
            n_req += prewarm_table[il].size();
            auto claim_all = [&]() {
                for (int32_t e : prewarm_table[il]) {
                    const int32_t s = claim_slot_in_window(st, e, /*cache_only=*/true);
                    if (s >= 0) {
                        tasks[il].push_back({ e, s });
                    }
                }
            };
            if ((int) il == cur_layer) {
                claim_all();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                claim_all();
            }
        }

        std::vector<flat> q;
        for (size_t il = 0; il < tasks.size(); ++il) {
            for (const auto & t : tasks[il]) {
                q.push_back({ (int) il, t.first, t.second });
            }
        }
        const int n = (int) q.size();
        const int n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
        std::atomic<int> next{0};
        std::atomic<int> n_read{0};
        std::atomic<int64_t> n_bytes{0};
        auto read_one = [&]() {
            for (;;) {
                const int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) {
                    break;
                }
                auto & st = layers[(size_t) q[i].layer];
                try {
                    read_expert_directv(st, q[i].expert, q[i].slot, true);
                    n_read.fetch_add(1, std::memory_order_relaxed);
                    n_bytes.fetch_add((int64_t) st.pack_bytes, std::memory_order_relaxed);
                } catch (...) {
                    q[i].slot = -1;
                }
            }
        };
        if (n > 0) {
            if (n_th == 1) {
                read_one();
            } else {
                std::vector<std::thread> th;
                th.reserve((size_t) n_th);
                for (int t = 0; t < n_th; ++t) {
                    th.emplace_back(read_one);
                }
                for (auto & h : th) {
                    h.join();
                }
            }
        }

        size_t n_pub = 0;
        std::vector<prewarm_place> placed;
        placed.reserve((size_t) n);
        for (size_t il = 0; il < tasks.size(); ++il) {
            if (tasks[il].empty()) {
                continue;
            }
            auto & st = layers[il];
            auto pub_all = [&]() {
                for (const auto & t : tasks[il]) {
                    if (t.second >= 0 && publish_slot_in_window(st, t.first, t.second)) {
                        placed.push_back({ (int32_t) il, t.first, t.second });
                        n_pub++;
                    }
                }
            };
            if ((int) il == cur_layer) {
                pub_all();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                pub_all();
            }
        }
        // 记下落位：阶段切换（prefill → decode）时用它们「重新发布」映射，不必重读盘。
        prewarm_placed = std::move(placed);
        prewarm_ready = !prewarm_placed.empty();

        moe_log("prewarm: 表 %zu 专家 -> 认领 %d / 读入 %d / 发布 %zu，%.1f MiB，wall=%.1f ms "
                "(%d 线程，缓存区)",
                n_req, n, (int) n_read.load(), n_pub,
                (double) n_bytes.load() / (1024.0 * 1024.0), ms_of(now_ns() - t0), n_th);
    }

    /// 阶段切换（prefill → decode）时用预热落位记录**重新发布**映射。
    ///
    /// 数据本身还在（decode 段与 prefill 段物理分离），只是切换时映射表被清了 ——
    /// 所以这里纯整数操作、不读盘。`cur_layer` 的锁已由调用方持有（同 prewarm_windows）。
    void publish_prewarm_placed(int cur_layer) {
        if (managers_active()) return; // trusted physical Residents already belong to the authority
        for (const auto & p : prewarm_placed) {
            if (p.layer < 0 || p.layer >= (int) layers.size()) {
                continue;
            }
            auto & st = layers[(size_t) p.layer];
            if (!st.enabled) {
                continue;
            }
            if (p.expert < 0 || p.expert >= n_expert ||
                p.slot < 0 || p.slot >= (int32_t) st.slot_to_expert.size()) {
                continue;
            }
            auto do_pub = [&]() {
                // 刚清过映射（全部 kSlotFree），这里直接登记。
                // 若该 slot/专家已被占用（理论上不会），跳过以免制造冲突。
                if (st.slot_to_expert[(size_t) p.slot] == kSlotFree &&
                    st.expert_to_slot[(size_t) p.expert] < 0) {
                    st.slot_to_expert[(size_t) p.slot] = p.expert;
                    st.expert_to_slot[(size_t) p.expert] = p.slot;
                    st.slot_age[(size_t) p.slot] = ++st.age;
                }
            };
            if ((int) p.layer == cur_layer) {
                do_pub();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                do_pub();
            }
        }
    }

    /// 公开入口（`llama_moe_prewarm_experts`）：把静态表读进各层 decode 窗口并登记映射。
    /// 在 prefill 之前、空闲期调用（App：mmproj 卸载后）。调用期间不得有 decode 在飞。
    void prewarm_experts() {
        // Prefill will overwrite the overlapped storage. Static warming, if
        // requested, is deferred until window entry after prefill has completed.
        if (model.moe_phase_reuse()) return;
        if (!prewarm_enabled) {
            return;
        }
        if (managers_active()) lookahead.drain();
        prewarm_windows(-1);   // -1 = 不持任何层锁，全部自取
    }

    /// 决定 decode 缓存的窗口宽度，并把结果打出来。
    ///
    /// 为什么 prefill 用不了窗口：实测 prefill 一层要用 105~215 个专家（`unique/l`），
    /// 而窗口只有 bank/40 个 slot，根本装不下。所以 prefill 必须保持「整块 bank + 逐层清空」，
    /// decode 才切窗口 —— 两者是天然分离的阶段，在 `process()` 里按 `n_tokens == 1` 切换。
    void resolve_decode_window() {
        const char * s = std::getenv("LLAMA_MOE_DECODE_WINDOW");
        if (!s || !*s) {
            moe_log("decode-cache: off（设 LLAMA_MOE_DECODE_WINDOW 开启，留空=按 bank 自动取最大）");
            return;
        }
        if (!shared_bank) {
            moe_log("decode-cache: off（仅共享 bank 模式支持按层窗口）");
            return;
        }
        // 窗口宽度以**模型侧**为唯一来源（`moe_decode_window()`）。
        // 模型加载时按它建了张量布局，运行时若自己另算一遍就会对不上 ——
        // 而且旧的上限（bank / 层数）属于共享张量方案，现在不再成立：
        // 每层有自己的张量，c 只受 RAM 和 ne[2] <= 1024 约束。
        const int32_t want = model.moe_decode_window();
        if (want < 1) {
            moe_log("decode-cache: off（模型侧未开启窗口）");
            return;
        }
        decode_window = want;
        // 预读暂存区大小 = 预读深度（K）。没开预读就是 0，等于退回「只有缓存区」。
        prefetch_slots = lookahead.enabled ? std::max(0, (int32_t) lookahead.depth) : 0;
        {
            // 默认统一 LRU：预读与主路径共用整个窗口 (c+K)。
            // `LLAMA_MOE_PREFETCH_UNIFIED=0` 退回分区模式（只作 A/B 用）。
            const char * u = std::getenv("LLAMA_MOE_PREFETCH_UNIFIED");
            prefetch_unified = !(u && *u && *u == '0');
            moe_log("decode-cache: %s（窗口 %d slot，K=%d）",
                    prefetch_unified ? "预读与缓存共用一个 LRU 窗口" : "预读/缓存分区",
                    (int) (decode_window + prefetch_slots), (int) prefetch_slots);
        }

        // 各层窗口起点**不再是 il*c** —— 现在每层有自己的 decode 张量
        // （base 上的 view），它的 slot 空间就是 [0, 宽度)，所以起点恒为 0。
        // 这一点让运行时更简单：层间天然不冲突。
        //
        // **宽度按层**：模型侧已按 LLAMA_MOE_LAYER_WINDOWS 建好布局（base 容量是
        // 各层宽度之和），这里必须用**同一份**数据，否则张量对不上。
        {
            const std::vector<int32_t> per_layer_c = model.moe_layer_windows();
            layer_width.assign(layers.size(), 0);
            for (size_t il = 0; il < layers.size(); ++il) {
                const int32_t c_il = il < per_layer_c.size() ? per_layer_c[il] : decode_window;
                const int32_t w = c_il + prefetch_slots;
                layers[il].window_start = 0;
                layers[il].window_width = w;
                // Runtime metadata must cover both the prefill bank and the
                // complete per-layer cache/prefetch window.
                const size_t capacity = (size_t) std::max(n_slots, w);
                layers[il].slot_to_expert.resize(capacity, kSlotFree);
                layers[il].slot_age.resize(capacity, 0);
                layer_width[il] = w;
            }
            // 打印一下分布，确认按层生效
            std::string dist;
            for (size_t il = 0; il < layer_width.size() && il < 6; ++il) {
                dist += format("%zu:%d ", il, (int) layer_width[il]);
            }
            moe_log("decode-cache: 按层宽度 %s（前几层 %s）",
                    (getenv("LLAMA_MOE_LAYER_WINDOWS") ? "on" : "off（均匀）"), dist.c_str());
        }

        // ── 窗口覆盖多大的批宽 ──────────────────────────────────────────────
        //
        // 单 token decode 之外，投机验证（DSpark）也会送进批宽 = n_max+1 的小批次。
        // 让它们走窗口，吃的是**跨轮**的专家复用（连续解码同一段文本，路由高度重合）。
        // 上限由**容量**决定：必须保证「该批宽在最坏情况下的专家并集」装得进缓存区。
        //   最坏并集 = min(行数 × top-k, n_expert)——每行最多 top-k 个，全库只有 n_expert 个。
        //
        // 取**各层里最窄的那个**来夹：窗口可逐层不同（LLAMA_MOE_LAYER_WINDOWS），
        // 夹紧必须对最窄的层也成立，否则那一层会在 process() 里抛异常。
        {
            const char * rows_env = std::getenv("LLAMA_MOE_WINDOW_MAX_ROWS");
            const int32_t rows_want = std::max(1, rows_env && *rows_env ? atoi(rows_env) : 1);

            int32_t cache_min = 0;
            for (size_t il = 0; il < layer_width.size() && il < layers.size(); ++il) {
                if (!layers[il].enabled || layer_width[il] <= 0) {
                    continue;
                }
                // A unified LRU still contains slots claimed by in-flight
                // prefetch reads. Demand allocation cannot evict kSlotPending,
                // so reserve that capacity when bounding a graph's worst-case
                // expert union, just as in the partitioned layout.
                const int32_t usable = layer_width[il] - prefetch_slots;
                if (usable > 0 && (cache_min == 0 || usable < cache_min)) {
                    cache_min = usable;
                }
            }
            if (cache_min <= 0) {
                cache_min = decode_window;
            }

            const int32_t n_used = (int32_t) model.hparams.n_expert_used;
            int32_t rows = rows_want;
            while (rows > 1 && cache_min < std::min(rows * n_used, n_expert)) {
                rows--;
            }
            window_max_rows = rows;

            if (rows != rows_want) {
                moe_log("decode-cache: 缓存区只有 %d 格，装不下 %d 行的最坏并集（%d），"
                        "窗口适用的批宽从 %d 夹到 %d",
                        (int) cache_min, (int) rows_want,
                        (int) std::min(rows_want * n_used, n_expert), (int) rows_want, (int) rows);
            }
            moe_log("decode-cache: 窗口覆盖批宽 1..%d%s", (int) window_max_rows,
                    window_max_rows > 1 ? "（投机验证也走窗口）" : "（只服务单 token decode）");
        }

        // 预读**保留**，并改成写进目标层自己的窗口（见 install_prefetch_into_window）。
        //
        // 为什么不关掉它：预读的字节是在上一层计算期间并发读的，只要
        // `预读字节 / 带宽 <= 单层计算时间` 就完全被隐藏 —— 那部分是白赚的。
        // 真正加在耗时上的是**关键路径**字节（卡住下一层开始的那些）。
        // 缓存的另一个作用正是把预读从「超出预算、部分暴露」压回「完全隐藏」：
        //   无缓存 depth 8: 预读 14.2 MiB/层 → 8.2ms > 6ms 预算  ❌
        //   缓存 24 + K=8 : 预读 6.6 MiB/层 → 3.8ms < 6ms 预算  ✅
        // 所以两者是互补的，不该二选一。
        if (lookahead.enabled && !lookahead.window_mode) {
            lookahead.window_mode = true;
            moe_log("decode-cache: 预读保留，改为写进目标层窗口（depth=%d）",
                    (int) lookahead.depth);
        }

        const int32_t n_moe_layers = model.moe_experts_layer_count();
        const size_t total = (size_t) decode_window * (size_t) n_moe_layers;
        moe_log("decode-cache: on cache=%d + prefetch=%d slots/layer × %d 层 = %zu slots "
                "(%.0f MiB，与 %d-slot 的 prefill 共享 bank 同处一块 base)",
                (int) decode_window, (int) prefetch_slots, (int) n_moe_layers, total,
                (double) total * (double) layers[0].pack_bytes / 1024.0 / 1024.0,
                (int) n_slots);
    }

    ~impl() {
        if (cur.n_layers > 0) {
            flush_ubatch(cur.n_tokens <= 1 ? "decode" : "prefill",
                cur.n_tokens <= 1 ? n_decode_step : n_prefill_ubatch);
        }
        lookahead.report();
        moe_log("cache-summary: scope=all_graphs_including_initialization "
                "window_unique=%llu window_hit=%llu window_miss=%llu "
                "bank_unique=%llu bank_miss=%llu",
                (unsigned long long) window_unique,
                (unsigned long long) (window_unique - window_miss),
                (unsigned long long) window_miss,
                (unsigned long long) bank_unique,
                (unsigned long long) bank_miss);
        if (miss_pos_steps > 0) {
            std::string s;
            for (int i = 0; i < 16; ++i) {
                if (miss_pos[i] == 0 && i >= 8) break;
                s += format("%d:%.2f ", i, (double) miss_pos[i] / (double) miss_pos_steps);
            }
            moe_log("miss position (score order, avg over %llu steps): %s",
                    (unsigned long long) miss_pos_steps, s.c_str());
        }
        if (miss_per_layer_steps > 0 && decode_window > 0) {
            // 逐层 miss 分布：判断「每层都有几个」还是「少数层很多」。
            std::string s;
            for (size_t il = 0; il < miss_per_layer.size(); ++il) {
                if (miss_per_layer[il] == 0) continue;
                s += format("%zu:%.2f ", il, (double) miss_per_layer[il] / (double) miss_per_layer_steps);
            }
            moe_log("miss/layer (avg over %llu steps): %s",
                    (unsigned long long) miss_per_layer_steps, s.c_str());
        }
        {
            std::string s;
            for (int k = 1; k <= 8; ++k) {
                if (recall_pred[k] == 0) continue;
                s += format("k=%d:%.1f%%(%llu/%llu) ", k,
                            100.0 * (double) recall_hit[k] / (double) recall_pred[k],
                            (unsigned long long) recall_hit[k],
                            (unsigned long long) recall_pred[k]);
            }
            if (!s.empty()) {
                moe_log("horizon-recall: %s", s.c_str());
            }
        }
        lookahead.stop();
        if (io_budget) io_budget->close();
        const auto admission = io_budget ? io_budget->snapshot() : llama_moe::io_admission::statistics{};
        moe_log("io-budget-summary: enabled=%d limit=%u scope=ready-window peak=%u grants=%llu releases=%llu cancelled=%llu "
                "boosted=%llu demand_grants=%llu needed_grants=%llu background_grants=%llu active_final=%u waiters_final=%u",
                (int) (io_budget != nullptr), io_budget ? expert_io_budget() : 0, admission.peak,
                (unsigned long long) admission.grants, (unsigned long long) admission.releases,
                (unsigned long long) admission.cancelled, (unsigned long long) admission.boosted,
                (unsigned long long) admission.demand_grants, (unsigned long long) admission.needed_grants,
                (unsigned long long) admission.background_grants, admission.active, admission.waiting);
        lookahead.report_completed();
        llama_moe::slot_manager::probation_counters probation;
        for (const auto & st : layers) if (st.manager) {
            const auto counts = st.manager->probation_stats();
            probation.promotions += counts.promotions; probation.demand_victims += counts.demand_victims;
        }
        moe_log("pf-probation-summary: enabled=%d promotions=%llu demand_victims=%llu",
                (int) pf_probation, (unsigned long long) probation.promotions, (unsigned long long) probation.demand_victims);
        moe_log("h1-front-summary: enabled=%d eligible=%llu attempted=%llu candidate_ids=%llu",
                (int) h1_front, (unsigned long long) h1_front_eligible,
                (unsigned long long) h1_front_attempted, (unsigned long long) h1_front_candidate_ids);
        moe_log("slot-manager-summary: enabled=%d requested=%d coalesced=%llu promoted=%llu joined=%llu reused=%llu "
                "reads=%llu evicted=%llu stale_tasks=%llu failed=%llu own_reads=%llu retry_reads=%llu pf_reads=%llu prewarm_reads=%llu quota_skipped=%llu capacity_skipped=%llu",
                (int) managers_active(), (int) slot_manager_enabled(),
                (unsigned long long) manager_coalesced.load(), (unsigned long long) manager_promoted.load(),
                (unsigned long long) manager_joined.load(), (unsigned long long) manager_reused.load(),
                (unsigned long long) manager_reads.load(), (unsigned long long) manager_evicted.load(),
                (unsigned long long) manager_stale.load(), (unsigned long long) manager_failed.load(),
                (unsigned long long) manager_demand_reads.load(), (unsigned long long) manager_retry_reads.load(),
                (unsigned long long) manager_prefetch_reads.load(), (unsigned long long) manager_prewarm_reads.load(),
                (unsigned long long) manager_quota_skipped.load(), (unsigned long long) manager_capacity_skipped.load());
        handoff_trace.flush();
        if (g_n_horizon_clamped.load()) {
            moe_log("horizon-clamp-summary: 本 runtime 累计夹紧 %llu 次（可见日志条数≠实际次数）",
                    (unsigned long long) g_n_horizon_clamped.load());
        }
        if (n_verify_ok.load() || n_verify_bad.load() || n_slot_collision.load()) {
            moe_log("verify-slot: ok=%llu bad=%llu slot_collision=%llu",
                    (unsigned long long) n_verify_ok.load(),
                    (unsigned long long) n_verify_bad.load(),
                    (unsigned long long) n_slot_collision.load());
        }
        moe_log("io-summary: scope=all_graphs_including_initialization_and_prefetch "
                "requested_bytes=%llu span_requests=%llu",
                (unsigned long long) io_bytes_all.load(std::memory_order_relaxed),
                (unsigned long long) io_calls_all.load(std::memory_order_relaxed));
        log_io("final");
        report_merge();
        if (sweep_fp) {
            fclose(sweep_fp);
            sweep_fp = nullptr;
        }
        for (const auto & item : fds) {
#ifdef _WIN32
            _close(item.second);
#else
            close(item.second);
#endif
        }
    }

    void bind(layer_state & state, ggml_tensor * tensor, int32_t expect_slots,
              std::vector<binding> * out = nullptr) {
        if (!tensor) {
            return;
        }
        const auto * entry = model.moe_experts_entry(ggml_get_name(tensor));
        if (!entry) {
            throw std::runtime_error(format("missing experts entry for tensor '%s'", ggml_get_name(tensor)));
        }
        if (tensor->ne[2] != expect_slots ||
            ggml_nbytes(tensor) != entry->bytes_per_expert * (size_t) expect_slots) {
            throw std::runtime_error(format("invalid virtual slot tensor '%s'", ggml_get_name(tensor)));
        }
        (out ? *out : state.bindings).push_back({ tensor, entry });
    }

    using pack_out = layer_state;

    static void finalize_pack(std::vector<binding> & bindings, std::vector<size_t> & order,
                              bool & ok, size_t & base, size_t & stride, std::string & path) {
        ok = false;
        if (bindings.empty()) {
            return;
        }
        const std::string & p = bindings[0].entry->path;
        const size_t st = bindings[0].entry->expert_stride;
        if (st == 0) {
            return;
        }
        size_t min_off = (size_t) -1;
        size_t max_end = 0;
        for (const auto & b : bindings) {
            if (b.entry->path != p || b.entry->expert_stride != st) {
                return;
            }
            min_off = std::min(min_off, b.entry->offset);
            max_end = std::max(max_end, b.entry->offset + b.entry->bytes_per_expert);
        }
        if (min_off == (size_t) -1 || max_end <= min_off || (max_end - min_off) > st) {
            return;
        }
        for (const auto & b : bindings) {
            if (b.entry->offset < min_off) {
                return;
            }
            if ((b.entry->offset - min_off) + b.entry->bytes_per_expert > st) {
                return;
            }
        }
        ok = true;
        base = min_off;
        stride = st;
        path = p;

        order.resize(bindings.size());
        for (size_t i = 0; i < bindings.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return bindings[a].entry->offset < bindings[b].entry->offset;
        });
    }

    static void finalize_pack(layer_state & state) {
        finalize_pack(state.bindings, state.pack_order, state.pack_ok,
                      state.pack_base, state.pack_stride, state.pack_path);
        state.pack_bytes = state.pack_stride;
    }

    /// 取 fd（按路径缓存）。O_DIRECT 只能在 open() 时指定、不能用 fcntl 改已打开的 fd，
    /// 但读法在进程生命周期里是固定的（只有 0/1 两档，没有阶段切换），所以一张表就够。
    int fd_for(const std::string & path) {
        std::lock_guard<std::mutex> lock(fd_mutex);
        const bool direct = moe_o_direct();
        auto & table = fds;
        const auto found = table.find(path);
        if (found != table.end()) {
            return found->second;
        }
#ifdef _WIN32
        const int fd = _open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
        int open_flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECT
        if (direct) {
            open_flags |= O_DIRECT;
        }
#endif
        const int fd = open(path.c_str(), open_flags);
#endif
        if (fd < 0) {
            throw std::runtime_error(format("failed to open MoE experts file '%s': %s", path.c_str(), std::strerror(errno)));
        }
#if defined(__linux__)
        const char * random_reads = std::getenv("LLAMA_MOE_FADVISE_RANDOM");
        if (random_reads && *random_reads != '0') {
            const int advice_error = posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
            if (advice_error) moe_log("random-read advice failed: %s", std::strerror(advice_error));
        }
#endif
        table.emplace(path, fd);
        return fd;
    }

    /// O_DIRECT 版读：先读进 4096 对齐的 thread-local scratch，再 memcpy 给各 iovec。
    /// 目的地址和长度都不保证对齐（gate|up|down 三段），所以不能直接 preadv 进去。
    /// LLAMA_MOE_VERIFY_SLOT=1：每个专家读完就把 slot 里的字节和文件里的字节比一遍。
    /// 这条检查证明「算出来的文件偏移 → 落到的内存」没错位（评审第 2 条那一类错误的
    /// 可执行判据；md5 对比反而证明不了这个，因为这套运行时本来就不是 bit-exact）。
    static bool verify_slot_enabled() {
        static const bool on = [] {
            const char * v = std::getenv("LLAMA_MOE_VERIFY_SLOT");
            return v && *v && *v != '0';
        }();
        return on;
    }

    bool verify_slot_bytes(const std::string & path, size_t offset,
                           const struct iovec * iov, int niov) {
        size_t total = 0;
        for (int i = 0; i < niov; ++i) total += iov[i].iov_len;
        if (!total) return false;
        void * buf = nullptr;
        if (posix_memalign(&buf, 4096, total) != 0) {
            n_verify_bad.fetch_add(1, std::memory_order_relaxed);
            moe_log("VERIFY-SLOT: allocation failed %s offset=%zu bytes=%zu", path.c_str(), offset, total);
            return false;
        }
        size_t done = 0;
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            while (done < total) {
                const ssize_t r = pread(fd, (uint8_t *) buf + done, total - done, (off_t) (offset + done));
                if (r <= 0) break;
                done += (size_t) r;
            }
            close(fd);
        }
        int bad = 0;
        size_t at = 0;
        for (int i = 0; i < niov && done == total; ++i) {
            if (memcmp((const uint8_t *) buf + at, iov[i].iov_base, iov[i].iov_len) != 0) bad++;
            at += iov[i].iov_len;
        }
        if (done != total) bad = -1;
        if (bad) {
            n_verify_bad.fetch_add(1, std::memory_order_relaxed);
            moe_log("VERIFY-SLOT: **不一致** %s offset=%zu bytes=%zu 坏段=%d read=%zu",
                    path.c_str(), offset, total, bad, done);
        } else {
            n_verify_ok.fetch_add(1, std::memory_order_relaxed);
        }
        free(buf);
        return bad == 0;
    }

    void readv_span_direct(int fd, size_t offset, struct iovec * iov, int niov) {
        size_t total = 0;
        for (int i = 0; i < niov; ++i) {
            total += iov[i].iov_len;
        }
        io_bytes_all.fetch_add(total, std::memory_order_relaxed);
        io_calls_all.fetch_add(1, std::memory_order_relaxed);
        if (total == 0) {
            return;
        }
        // 长度也得是 512 的倍数才能进 O_DIRECT；不是就退回缓冲读（少见，专家 stride 是 512 的倍数）。
        if (total % 512 != 0 || offset % 512 != 0) {
            readv_span(fd, offset, iov, niov);
            return;
        }
        static thread_local uint8_t * scratch = nullptr;
        static thread_local size_t scratch_cap = 0;
        const size_t need = (total + 4095) & ~(size_t) 4095;
        if (scratch_cap < need) {
            // aligned_alloc() is API 28+ on Android (minSdk 24); posix_memalign() is available
            // on every supported API level and gives the 4096-byte alignment O_DIRECT needs.
            // 先分配成功再释放旧 buffer：失败时 scratch/scratch_cap 仍指向原 buffer，
            // 不会留下悬空指针（否则同线程下一次调用会 double-free）。
            void * aligned = nullptr;
            if (posix_memalign(&aligned, 4096, need) != 0 || aligned == nullptr) {
                throw std::runtime_error("failed to allocate aligned MoE read buffer");
            }
            if (scratch) {
                free(scratch);
            }
            scratch = (uint8_t *) aligned;
            scratch_cap = need;
        }
        {
            moe_handoff_span read_span(handoff_trace, "PREAD_BEGIN", "PREAD_DONE", handoff_thread.context,
                                   handoff_thread.expert, handoff_thread.slot, total);
            size_t done = 0;
            while (done < total) {
                const ssize_t result = pread(fd, scratch + done, total - done, (off_t) (offset + done));
                if (result < 0 && errno == EINTR) {
                    continue;
                }
                if (result <= 0) {
                    throw std::runtime_error(format(
                        "failed to read MoE experts (O_DIRECT) at offset %zu: %s",
                        offset + done, result < 0 ? std::strerror(errno) : "short read"));
                }
                done += (size_t) result;
            }
            read_span.complete();
        }
        size_t at = 0;
        for (int i = 0; i < niov; ++i) {
            memcpy(iov[i].iov_base, scratch + at, iov[i].iov_len);
            at += iov[i].iov_len;
        }
    }

    // read one contiguous file span into several disjoint buffers, adjusting the iovecs on partial reads
    void readv_span(int fd, size_t offset, struct iovec * iov, int niov) {
        if (moe_o_direct()) {
            readv_span_direct(fd, offset, iov, niov);
            return;
        }
        {
            size_t n = 0;
            for (int i = 0; i < niov; ++i) {
                n += iov[i].iov_len;
            }
            // 唯一能给出「完整 I/O」的埋点：主路径和预读 worker 都走这里。
            // 只看 decode 行里的 bytes= 会漏掉预读读的字节（实测那是大头）。
            io_bytes_all.fetch_add(n, std::memory_order_relaxed);
            io_calls_all.fetch_add(1, std::memory_order_relaxed);
        }
        size_t total = 0;
        for (int i = 0; i < niov; ++i) {
            total += iov[i].iov_len;
        }
        moe_handoff_span pread(handoff_trace, "PREAD_BEGIN", "PREAD_DONE", handoff_thread.context,
                               handoff_thread.expert, handoff_thread.slot, total);
        size_t done = 0;
        while (done < total) {
            struct iovec cur[8];
            int cn = 0;
            size_t skip = done;
            for (int i = 0; i < niov; ++i) {
                if (skip >= iov[i].iov_len) {
                    skip -= iov[i].iov_len;
                    continue;
                }
                cur[cn].iov_base = (uint8_t *) iov[i].iov_base + skip;
                cur[cn].iov_len  = iov[i].iov_len - skip;
                skip = 0;
                cn++;
            }
#ifdef _WIN32
            std::lock_guard<std::mutex> lock(io_mutex);
            if (_lseeki64(fd, (int64_t) (offset + done), SEEK_SET) < 0) {
                throw std::runtime_error("failed to seek MoE experts");
            }
            const ssize_t result = _read(fd, cur[0].iov_base, (unsigned int) cur[0].iov_len);
#else
            const ssize_t result = preadv(fd, cur, cn, (off_t) (offset + done));
#endif
            if (result < 0 && errno == EINTR) {
                continue;
            }
            if (result <= 0) {
                throw std::runtime_error(format(
                    "failed to read MoE experts at offset %zu", offset + done));
            }
            done += (size_t) result;
        }
        pread.complete();
    }

    // Reads the packed gate|up|down span (one contiguous run in the experts) with a single preadv,
    // writing each family straight into its own slot. One syscall, one write per byte: no staging
    // buffer and no second copy into the device buffer.
    int64_t read_expert_directv(const layer_state & state, int32_t expert, int32_t slot,
                                bool window_mode = false) {
        const int32_t trace_layer = (int32_t) (&state - layers.data());
        auto context = handoff_thread.trace == &handoff_trace ? handoff_thread.context
            : moe_handoff_context{0, 0, trace_layer, 0, "other", 0, window_mode ? "window" : "bank"};
        if (!context.io_id) context.io_id = handoff_trace.new_io_id();
        moe_handoff_scope scope(handoff_trace.enabled() ? &handoff_trace : nullptr, context, expert, slot);
        const bool pf = handoff_trace.enabled() && std::strcmp(context.source, "prefetch") == 0;
        const bool demand = handoff_trace.enabled() && std::strcmp(context.source, "demand") == 0;
        moe_handoff_span reader(handoff_trace, pf ? "PF_READ_BEGIN" : demand ? "DEMAND_READ_BEGIN" : "OTHER_READ_BEGIN",
            pf ? "PF_READ_DONE" : demand ? "DEMAND_READ_DONE" : "OTHER_READ_DONE", context, expert, slot);
        // 诊断：LLAMA_MOE_SKIP_READ=1 时走完所有记账但不真读盘。
        //
        // 这些模式**只用来观察耗时**，不能用 step 之间的差值去做归因：
        // 它们走的读取路径不同、落点不同、且后两者输出无效（路由与投机接受率都会变）。
        // 输出是错的（专家内容是垃圾），只看时间。
        // 诊断三态（**只用于观察耗时，不能互相相减做归因**）：
        //   0（默认）正常：preadv 直接写进 slot 张量（GPU 可见）
        //   1 dummy   ：**诊断模式，不是 IO-only 下界**。它跳过 packed/tiled 正常路径
        //               （`packed_dir && mode == 0` 不再成立），改走 raw 路径读进**同一个**
        //               scratch，所有 family 共用一个缓冲区；MoE 图仍然照常计算，
        //               用的是 slot 里的旧数据 ⇒ **输出无效**。
        //               所以它的耗时不能当成"只读不算"的地板（曾被这样误用，见
        //               exps/pipe-in-layer-20261105/README.md §3.34 的更正）。
        //   2 skip    ：完全不读（图仍然照算、slot 里是旧数据）⇒ **同样输出无效**。
        //               **不能**当成"有效推理的算力地板"，只能当"不读盘时的诊断耗时"
        //               （见 exps/pipe-in-layer-20261105/README.md §3.34）。
        // 三种模式**只用于观察耗时**，不能互相相减来归因"成本在写 GPU 缓冲还是 I/O"：
        // 它们走的读取路径不同（dummy/skip 跳过 packed/tiled 分支）、落点不同、
        // 且后两者输出无效、路由与投机接受率都会变 —— 差值没有可比性（§3.34）。
        static const int mode = [] {
            const char * d = std::getenv("LLAMA_MOE_DUMMY_READ");
            if (d && *d && *d != '0') return 1;
            const char * s = std::getenv("LLAMA_MOE_SKIP_READ");
            if (s && *s && *s != '0') return 2;
            return 0;
        }();

        // decode 阶段读进**本层自己的窗口张量**（base 上的 view），prefill 读进共享那段。
        // 两套的张量、文件偏移、步长都不同，所以整组一起切。
        const std::vector<binding> & bindings = window_mode ? state.dec_bindings   : state.bindings;
        const std::vector<size_t>  & order    = window_mode ? state.dec_pack_order : state.pack_order;
        const std::string          & path     = window_mode ? state.dec_pack_path : state.pack_path;
        const size_t base   = window_mode ? state.dec_pack_base   : state.pack_base;
        const size_t stride = window_mode ? state.dec_pack_stride : state.pack_stride;

        // Optional offline tiled files: retain the raw manifest and compute the
        // backend strides independently. Never feed tiled bytes to raw upload.
        const char * packed_dir = std::getenv("LLAMA_MOE_HTP_PACKED_DIR");
        if (packed_dir && *packed_dir && mode == 0) {
            using pack_fn = size_t (*)(void *, const void *, int, int64_t, int64_t);
            struct iovec packed_iov[8];
            size_t packed_stride = 0;
            if (order.empty() || order.size() > 8) throw std::runtime_error("invalid packed expert families");
            for (size_t i = 0; i < order.size(); ++i) {
                const binding & b = bindings[order[i]];
                auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(b.tensor->buffer));
                auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
                auto pack = reg ? (pack_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_pack_moe_expert") : nullptr;
                if (!pack) throw std::runtime_error("offline tiled experts require the Hexagon backend");
                const size_t bytes = pack(nullptr, nullptr, b.tensor->type, b.tensor->ne[0], b.tensor->ne[1]);
                if (!bytes) throw std::runtime_error("unsupported offline expert type");
                packed_iov[i].iov_base = (uint8_t *) b.tensor->data + (size_t) slot * bytes;
                packed_iov[i].iov_len = bytes;
                packed_stride += bytes;
            }
            const size_t slash = path.find_last_of("/");
            const std::string packed_path = std::string(packed_dir) + "/" + path.substr(slash == std::string::npos ? 0 : slash + 1);
            const int64_t start = now_ns();
            const int packed_fd = fd_for(packed_path);
            const size_t packed_offset = (size_t) expert * packed_stride;
            readv_span(packed_fd, packed_offset, packed_iov, (int) order.size());
            if (verify_slot_enabled() &&
                !verify_slot_bytes(packed_path, packed_offset, packed_iov, (int) order.size())) {
                throw std::runtime_error("offline tiled expert slot verification failed");
            }
#if defined(__linux__)
            // Once retained in a DSP cache slot, the same expert pages need not
            // occupy Linux page cache too. This advice changes no file bytes.
            const char * drop_pages = std::getenv("LLAMA_MOE_DROP_CACHED_EXPERT_PAGES");
            if (window_mode && drop_pages && *drop_pages != '0') {
                const int advice_error = posix_fadvise(packed_fd, packed_offset, packed_stride, POSIX_FADV_DONTNEED);
                if (advice_error) moe_log("expert page-cache advice failed: %s", std::strerror(advice_error));
            }
#endif
            const int64_t elapsed = now_ns() - start;
            stats.pread_ns.fetch_add(elapsed, std::memory_order_relaxed);
            stats.n_bytes.fetch_add(packed_stride, std::memory_order_relaxed);
            stats.n_miss.fetch_add(1, std::memory_order_relaxed);
            reader.complete();
            return elapsed;
        }

        struct iovec iov[8];
        const int n = (int) order.size();
        if (n <= 0 || n > 8) {
            throw std::runtime_error("invalid MoE packed family count for preadv");
        }
        using upload_fn = bool (*)(ggml_tensor *, int32_t, const void *, size_t);
        std::vector<upload_fn> uploads((size_t) n, nullptr);
        bool staging = false;
        for (int i = 0; i < n; ++i) {
            const binding & b = bindings[order[(size_t) i]];
            auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(b.tensor->buffer));
            if (dev) {
                auto reg = ggml_backend_dev_backend_reg(dev);
                uploads[(size_t) i] = (upload_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_moe_expert");
                staging = staging || uploads[(size_t) i] != nullptr;
            }
        }
        size_t nbytes = 0;
        size_t max_exp = 0;
        for (int i = 0; i < n; ++i) {
            max_exp = std::max(max_exp, bindings[order[(size_t) i]].entry->bytes_per_expert);
        }
        if (mode == 1) {
            // dummy：读进一块常驻 scratch（不碰 GPU 缓冲区）。同样是真读盘，
            // 所以 I/O 量与正常模式一致，差别只有「数据落到哪里」。
            static std::vector<uint8_t> scratch;
            if (scratch.size() < max_exp) {
                scratch.resize(max_exp);
            }
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                iov[i].iov_base = scratch.data();
                iov[i].iov_len  = b.entry->bytes_per_expert;
                nbytes += b.entry->bytes_per_expert;
            }
        } else if (staging && mode == 0) {
            // HTP consumes tiled weights, so raw disk bytes require backend upload.
            static thread_local std::vector<uint8_t> raw;
            size_t total = 0;
            for (int i = 0; i < n; ++i) total += bindings[order[(size_t) i]].entry->bytes_per_expert;
            raw.resize(total);
            size_t pos = 0;
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                iov[i].iov_base = raw.data() + pos;
                iov[i].iov_len = b.entry->bytes_per_expert;
                pos += iov[i].iov_len;
            }
            nbytes = total;
        } else {
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                iov[i].iov_base = (uint8_t *) b.tensor->data + (size_t) slot * b.entry->bytes_per_expert;
                iov[i].iov_len  = b.entry->bytes_per_expert;
                nbytes += b.entry->bytes_per_expert;
            }
        }
        const int fd = fd_for(path);
        const size_t offset = base + (size_t) expert * stride;

        if (mode == 2) {
            // 记账照做（让上层以为读成功了），但不落盘 —— 用于隔离 I/O 与结构开销。
            stats.n_miss.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        // LLAMA_MOE_READ_DELAY_US：给每次读加固定延迟。
        // 用来判断「读时间是否 1:1 落在关键路径上」—— 若总时间按 delay × 读次数
        // 等量增长，说明读在关键路径上是线性的；若增长更多，说明有放大效应。
        static const int delay_us = [] {
            const char * d = std::getenv("LLAMA_MOE_READ_DELAY_US");
            return d ? std::atoi(d) : 0;
        }();
        if (delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }

        const int64_t t0 = now_ns();
        readv_span(fd, offset, iov, n);
        if (staging && mode == 0) {
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                if (uploads[(size_t) i]) {
                    if (!uploads[(size_t) i](b.tensor, slot, iov[i].iov_base, iov[i].iov_len)) {
                        throw std::runtime_error(format("backend cannot upload MoE expert slot: %s", ggml_get_name(b.tensor)));
                    }
                } else {
                    memcpy((uint8_t *) b.tensor->data + (size_t) slot * b.entry->bytes_per_expert, iov[i].iov_base, iov[i].iov_len);
                }
            }
        }
        const int64_t dt = now_ns() - t0;
        stats.pread_ns.fetch_add(dt, std::memory_order_relaxed);
        stats.n_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        stats.n_miss.fetch_add(1, std::memory_order_relaxed);
        reader.complete();
        return dt;
    }

    static const uint8_t * host_data(const ggml_tensor * tensor) {
        if (!tensor || !tensor->data || (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer))) {
            return nullptr;
        }
        return (const uint8_t *) tensor->data;
    }

    static uint8_t * host_data(ggml_tensor * tensor) {
        return const_cast<uint8_t *>(host_data((const ggml_tensor *) tensor));
    }

    void read_i32_ids(const ggml_tensor * tensor, std::vector<int32_t> & ids) const {
        const int64_t n_ids = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        const size_t row_bytes = (size_t) n_ids * sizeof(int32_t);
        ids.resize((size_t) (n_ids * n_tokens));
        if (const uint8_t * src = host_data(tensor)) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                memcpy(ids.data() + token * n_ids, src + (size_t) token * tensor->nb[1], row_bytes);
            }
            return;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_backend_tensor_get(
                tensor,
                ids.data() + token * n_ids,
                (size_t) token * tensor->nb[1],
                row_bytes);
        }
    }

    void write_i32_ids(ggml_tensor * tensor, const std::vector<int32_t> & ids) const {
        const int64_t n_ids = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        const size_t row_bytes = (size_t) n_ids * sizeof(int32_t);
        if (ids.size() != (size_t) (n_ids * n_tokens)) {
            throw std::runtime_error("MoE slot id count does not match tensor shape");
        }
        if (uint8_t * dst = host_data(tensor)) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                memcpy(dst + (size_t) token * tensor->nb[1], ids.data() + token * n_ids, row_bytes);
            }
            return;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_backend_tensor_set(
                tensor,
                ids.data() + token * n_ids,
                (size_t) token * tensor->nb[1],
                row_bytes);
        }
    }

    // 统计 pending 的「连续段」结构，用来判断「合并相邻专家的读」是否可行。
    //
    // 背景：experts 是 expert-major，expert k 的字节在文件里位于
    // `base + k * expert_stride`，所以**专家 id 连号 == 文件地址连号**。
    // 而 pending 是按 expert id 升序构建的，因此天然带着可用于合并的连续段。
    //
    // 两类可合并：
    //   1. slot 也连号 → 可以直接用一组 iovec 覆盖整段，**零拷贝**、syscall 数减到 1/段长
    //   2. slot 不连号 → 得先读进 scratch 再散写到各 slot，多一次内存拷贝
    //
    // 共享 bank 的 prefill 里 LRU 每层清空、slot 从 0 顺序分配，而 pending 又是按
    // expert 升序压入的，所以 1 类应该占绝大多数。这个统计就是来验证这一点的。
    static run_stats accumulate_runs(const std::vector<pending_install> & pending) {
        run_stats st;
        const size_t n = pending.size();
        if (n == 0) {
            return st;
        }
        size_t i = 0;
        while (i < n) {
            size_t j = i + 1;   // 只按 expert 连号扩展，slot 单独判
            while (j < n && pending[j].expert == pending[j - 1].expert + 1) {
                ++j;
            }
            const int run_len = (int) (j - i);

            bool slots_consecutive = true;
            for (size_t k = i + 1; k < j; ++k) {
                if (pending[k].slot != pending[k - 1].slot + 1) {
                    slots_consecutive = false;
                    break;
                }
            }

            st.n++;
            st.experts += run_len;
            st.max = std::max(st.max, run_len);
            if (slots_consecutive) {
                st.ok_n++;
                st.ok_ex += run_len;
            } else {
                st.scatter_n++;
                st.scatter_ex += run_len;
            }
            i = j;
        }
        return st;
    }

    static void add_runs(run_stats & dst, const run_stats & src) {
        dst.n          += src.n;
        dst.experts    += src.experts;
        dst.max         = std::max(dst.max, src.max);
        dst.ok_n       += src.ok_n;
        dst.ok_ex      += src.ok_ex;
        dst.scatter_n  += src.scatter_n;
        dst.scatter_ex += src.scatter_ex;
    }

    install_timing install_pending(layer_state & state, const std::vector<pending_install> & pending,
                                   bool window_mode) {
        install_timing total;
        const int n = (int) pending.size();
        if (n <= 0) {
            return total;
        }
        for (int offset = 0; offset < n; offset += kMoeSsdIoChunk) {
            const int n_wave = std::min(kMoeSsdIoChunk, n - offset);
            std::vector<pending_install> wave(
                pending.begin() + offset, pending.begin() + offset + n_wave);
            const install_timing t = install_packed_wave(state, wave, window_mode);
            total.n += t.n;
            total.n_th = std::max(total.n_th, t.n_th);
            total.pack = true;
            total.wall_ns += t.wall_ns;
            total.pread_sum_ns += t.pread_sum_ns;
            total.pread_max_ns = std::max(total.pread_max_ns, t.pread_max_ns);
            total.bytes += t.bytes;
        }
        return total;
    }

    install_timing install_packed_wave(layer_state & state, const std::vector<pending_install> & pending,
                                       bool window_mode) {
        install_timing t;
        const int n = (int) pending.size();
        if (n <= 0) {
            return t;
        }

        t.n = n;
        t.pack = true;
        t.bytes = (size_t) n * state.pack_bytes;
        std::vector<std::exception_ptr> errs((size_t) n);
        std::vector<int64_t> dts((size_t) n, 0);
        // Snapshot on the submitting thread, before the pool starts. Readers
        // must never load the caller's mutable graph_epoch themselves.
        const moe_handoff_context trace_context = {0, graph_epoch, (int32_t) (&state - layers.data()),
            handoff_thread.trace == &handoff_trace ? handoff_thread.context.rows : 0, "demand", 0, window_mode ? "window" : "bank"};
        auto read_one = [&](int i) {
            auto request_context = handoff_request(trace_context, pending[(size_t) i].request);
            request_context.io_id = pending[(size_t) i].io_id;
            moe_handoff_scope trace_scope(handoff_trace.enabled() ? &handoff_trace : nullptr, request_context);
            handoff_trace.emit("IO_OWNER", request_context,
                pending[(size_t) i].expert, pending[(size_t) i].slot, (int32_t) llama_moe::slot_manager::origin::demand);
            try {
                const auto & item = pending[(size_t) i];
                dts[(size_t) i] = item.request ? read_managed_expert(state, item.request,
                    llama_moe::io_admission::origin::demand, item.retry) :
                    read_expert_directv(state, item.expert, item.slot, window_mode);
                if (pending[(size_t) i].request) {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    if (!finish_managed(state, pending[(size_t) i].request, request_context, true)) {
                        throw std::runtime_error("demand expert request lost its slot ownership");
                    }
                }
                handoff_trace.emit("DEMAND_READY", request_context, pending[(size_t) i].expert, pending[(size_t) i].slot);
            } catch (...) {
                errs[(size_t) i] = std::current_exception();
                if (pending[(size_t) i].request) {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    finish_managed(state, pending[(size_t) i].request, request_context, false, errs[(size_t) i]);
                }
            }
        };

        t.n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
        const int64_t wall0 = now_ns();
        handoff_trace.emit("DEMAND_WAVE_SUBMIT", trace_context, -1, -1, n, t.bytes);
        // 注意：单线程不能只调 read_one(0) —— 那只会装第一个专家，
        // 其余 N-1 个被静默跳过，slot 里留着上一层/上一个 token 的旧权重，
        // 模型输出会错但不报错。这条路径只在 io_threads==1 时进入，
        // 默认 4 所以长期潜伏；把它做成可调后才会踩到。
        // 单线程还有个副作用：pending 是按 expert id 升序构建的，而 experts 是
        // expert-major（offset = base + expert * stride），所以单线程等于**严格
        // 顺序读文件**，可能接近顺序读速率（实测比 4 线程的随机交织快）。
        if (t.n_th == 1) {
            for (int i = 0; i < n; ++i) {
                read_one(i);
            }
        } else {
            std::atomic<int> next{0};
            auto work = [&]() {
                for (;;) {
                    const int i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= n) break;
                    read_one(i);
                }
            };
            if (!io_pool.run(t.n_th, work)) {
                std::vector<std::thread> threads;
                threads.reserve((size_t) t.n_th);
                try {
                    for (int th = 0; th < t.n_th; ++th) threads.emplace_back(work);
                } catch (...) {
                    for (auto & th : threads) th.join();
                    throw;
                }
                for (auto & th : threads) th.join();
            }
        }
        t.wall_ns = now_ns() - wall0;
        for (int i = 0; i < n; ++i) {
            if (errs[(size_t) i]) std::rethrow_exception(errs[(size_t) i]);
            t.pread_sum_ns += dts[(size_t) i];
            t.pread_max_ns = std::max(t.pread_max_ns, dts[(size_t) i]);
        }
        return t;
    }

    llama_moe::slot_manager::ticket wait_managed(layer_state & st, llama_moe::slot_manager::ticket request,
                                                moe_handoff_context context, install_timing & total) {
        {
            moe_handoff_span waiting(handoff_trace, "JOIN_BEGIN", "JOIN_DONE", handoff_request(context, request), request.expert, request.slot);
            try {
                std::unique_lock<std::mutex> lock(st.mutex);
                st.manager->wait(request, lock);
                waiting.complete(); return request;
            } catch (const llama_moe::io_admission_cancelled &) { throw; }
            catch (...) { /* A speculative failure gets one demand retry. */ }
        }
        std::vector<pending_install> retry(1); // allocate before acquiring ownership
        llama_moe::slot_manager::result r;
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            r = st.manager->acquire(request.expert, {context.epoch, 0});
            if (!r.request) throw std::runtime_error("no MoE slot for failed speculative read retry");
            trace_managed_claim(st, r, context);
        }
        if (!r.owner) {
            std::unique_lock<std::mutex> lock(st.mutex); st.manager->wait(r.request, lock); return r.request;
        }
        context = handoff_request(context, r.request);
        context.io_id = handoff_trace.new_io_id();
        handoff_trace.emit("DEMAND_READ_SUBMIT", context, r.request.expert, r.request.slot);
        retry[0] = {r.request.expert, r.request.slot, r.request, context.io_id, true};
        try {
            const auto timing = install_pending(st, retry, true);
            total.n += timing.n; total.n_th = std::max(total.n_th, timing.n_th);
            total.pread_sum_ns += timing.pread_sum_ns; total.pread_max_ns = std::max(total.pread_max_ns, timing.pread_max_ns);
            total.bytes += timing.bytes;
        } catch (...) {
            std::lock_guard<std::mutex> lock(st.mutex); finish_managed(st, r.request, context, false, std::current_exception()); throw;
        }
        return r.request;
    }

    static double ms_of(int64_t ns) {
        return ns / 1e6;
    }

    static double mib_of(size_t bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    static double par_of(int64_t sum_ns, int64_t wall_ns) {
        if (wall_ns <= 0) {
            return 0.0;
        }
        return (double) sum_ns / (double) wall_ns;
    }

    bool should_log_ubatch(int n_tokens, uint64_t idx) const {
        // LLAMA_MOE_LOG_EVERY_STEP=1：每步都打，供性能归因用。
        // 稀疏打印时 counter 的累计窗口和 span 容易搞错（实测踩过），
        // 逐步打印后 span 恒为 1，消费方不用做任何归一。
        static const bool every = [] {
            const char * s = std::getenv("LLAMA_MOE_LOG_EVERY_STEP");
            return s && *s && *s != '0';
        }();
        if (every) {
            return true;
        }
        if (idx <= 8) {
            return true;
        }
        if (n_tokens <= 1) {
            return idx <= 4 || (idx % 32) == 0;
        }
        return (idx % 8) == 0;
    }

    void flush_ubatch(const char * kind, uint64_t idx) {
        if (cur.n_layers <= 0) {
            return;
        }
        // 本步的**完整** I/O：主路径 + 预读。用全局计数器的增量，避免口径混淆。
        const uint64_t all_now = io_bytes_all.load(std::memory_order_relaxed);
        const uint64_t calls_now = io_calls_all.load(std::memory_order_relaxed);
        const int64_t proc_now = process_ns.load(std::memory_order_relaxed);
        const uint64_t arm_skip_now = lookahead.n_arm_skipped.load(std::memory_order_relaxed);
        const uint64_t wait_now     = lookahead.n_waits.load(std::memory_order_relaxed);
        const int64_t  wait_ns_now  = lookahead.n_wait_ns.load(std::memory_order_relaxed);
        const uint64_t pred_read_now = lookahead.n_pred_read.load(std::memory_order_relaxed);
        const uint64_t pf_hits_now = lookahead.n_hits.load(std::memory_order_relaxed);
        const uint64_t pf_pub_now  = lookahead.n_pred_published.load(std::memory_order_relaxed);
        const uint64_t all_delta = all_now - last_io_bytes;
        const uint64_t calls_delta = calls_now - last_io_calls;
        const int64_t proc_delta = proc_now - last_process_ns;
        const uint64_t arm_skip_delta = arm_skip_now - last_arm_skipped;
        const uint64_t wait_delta = wait_now - last_wait_count;
        const int64_t wait_ns_delta = wait_ns_now - last_wait_ns;
        const uint64_t pred_read_delta = pred_read_now - last_pred_read;
        const uint64_t pf_hits_delta = pf_hits_now - last_pf_hits;
        const uint64_t pf_pub_delta  = pf_pub_now - last_pf_pub;
        last_io_bytes = all_now;
        last_io_calls = calls_now;
        last_process_ns = proc_now;
        last_arm_skipped = arm_skip_now;
        last_wait_count = wait_now;
        last_wait_ns = wait_ns_now;
        last_pred_read = pred_read_now;
        last_pf_hits = pf_hits_now;
        last_pf_pub  = pf_pub_now;

        // 这些计数器只在 flush 时前进，而 flush 受 should_log_ubatch 节流。
        // 所以必须把「本次 flush 覆盖了多少步」一起打出来，否则消费方会把
        // 多步累计当成单步值（实测虚报过 24 倍）。`span=` 就是干这个的。
        const uint64_t span = idx > last_flush_idx ? idx - last_flush_idx : 1;
        last_flush_idx = idx;
        const int64_t flush_ns = now_ns();
        const double interval_s = (flush_ns - last_io_flush_ns) / 1e9;
        last_io_flush_ns = flush_ns;
        const double wall_s = cur.wall_ns / 1e9;
        const double mibs = wall_s > 0.0 ? (cur.bytes / (1024.0 * 1024.0)) / wall_s : 0.0;
        const double uniq = cur.n_layers > 0 ? (double) cur.unique_sum / (double) cur.n_layers : 0.0;
        const double avg_run = cur.runs.n > 0 ? (double) cur.runs.experts / (double) cur.runs.n : 0.0;
        moe_log("%s #%llu tok=%d used=%d layers=%d unique/l=%.0f miss=%d hit=%d pack=%d/%d nth=%d "
                "pread_wall=%.0fms pread_sum=%.0fms bytes=%.0fMiB par=%.2f io=%.0fMiB/s "
                "readall=%.0fMiB calls=%llu realio=%.0fMiB/s interval_ms=%.1f span=%llu pmgmt=%.1fms "
                "pfskip=%llu pfwait=%llu/%.1fms pfread=%llu pfpub=%llu pfhit=%llu pfpredict=%llu pevsave=%llu pevforced=%llu pfcached=%llu usedpf=%llu "
                "runs=%d avg_run=%.1f max_run=%d zerocopy=%.0f%%",
            kind,
            (unsigned long long) idx,
            cur.n_tokens,
            cur.n_used,
            cur.n_layers,
            uniq,
            cur.miss_sum,
            cur.unique_sum - cur.miss_sum,
            cur.pack_layers,
            cur.n_layers,
            cur.n_th,
            ms_of(cur.wall_ns),
            ms_of(cur.pread_sum_ns),
            mib_of(cur.bytes),
            par_of(cur.pread_sum_ns, cur.wall_ns),
            mibs,
            mib_of(all_delta),
            (unsigned long long) calls_delta,
            // Complete bytes and elapsed time cover the same logging interval.
            // This is process throughput, including compute between callbacks.
            interval_s > 0.0 ? (double) all_delta / (1024.0 * 1024.0) / interval_s : 0.0,
            interval_s * 1000.0,
            (unsigned long long) span,
            ms_of(proc_delta),
            (unsigned long long) arm_skip_delta,
            (unsigned long long) wait_delta,
            ms_of(wait_ns_delta),
            (unsigned long long) pred_read_delta,
            (unsigned long long) pf_pub_delta,
            (unsigned long long) pf_hits_delta,
            (unsigned long long) lookahead.n_predict_only.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_evict_saved.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_evict_forced.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_pred_cached.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_used_from_pf.load(std::memory_order_relaxed),
            cur.runs.n,
            avg_run,
            cur.runs.max,
            cur.runs.experts > 0 ? 100.0 * (double) cur.runs.ok_ex / (double) cur.runs.experts : 0.0);
        cur = ubatch_acc{};
    }

    /// 整个进程的「合并读」可行性汇总。用来回答一个问题：
    /// **把相邻专家的读合并成大 preadv，到底能减少多少次 syscall、传输会变成多大。**
    void report_merge() {
        if (g_runs.experts <= 0) {
            return;
        }
        const double avg_run = (double) g_runs.experts / (double) g_runs.n;
        const double zc_pct  = 100.0 * (double) g_runs.ok_ex / (double) g_runs.experts;
        // 假设全合并：syscall 数从「每专家一次」降到「每段一次」
        const double syscall_ratio = (double) g_runs.n / (double) g_runs.experts;
        moe_log("merge-scan: experts=%d runs=%d avg_run=%.2f max_run=%d "
                "zerocopy=%.1f%% scatter_experts=%d "
                "syscalls=%.1f%% of current (avg read = %.2f experts)",
            g_runs.experts, g_runs.n, avg_run, g_runs.max,
            zc_pct, g_runs.scatter_ex,
            100.0 * syscall_ratio, avg_run);
    }

    void log_io(const char * tag) {        const uint64_t miss = stats.n_miss.load(std::memory_order_relaxed);
        if (miss == 0) {
            return;
        }
        const int64_t pread_ns = stats.pread_ns.load(std::memory_order_relaxed);
        const uint64_t bytes = stats.n_bytes.load(std::memory_order_relaxed);
        moe_log("total[%s]: miss=%llu bytes=%.1f MiB pread=%.0fms",
            tag,
            (unsigned long long) miss,
            mib_of((size_t) bytes),
            pread_ns / 1e6);
    }

    const llama_model & model;
    int32_t n_expert;
    int32_t n_slots;
    int n_enabled = 0;
    int n_pack_ok = 0;
    bool shared_bank = false;
    // 按层 decode 缓存：每层分到 [layer*W, (layer+1)*W) 一段私有 slot 窗口，
    // 窗口内 LRU 跨 token 保留。prefill 不用窗口（一层要用 100+ 专家，装不下 W），
    // 仍然是「整块 bank + 每层清空」。
    //
    // W = 0 表示关闭（退回原行为：所有层共享全部 slot、每层清空）。
    int32_t decode_window = 0;
    /// 窗口适用的最大批宽（行数）。1 = 只服务单 token decode（旧行为）；更大的值让
    /// 投机验证（DSpark，批宽 = n_max+1）这类**小批次**也走窗口，靠跨轮专家复用省读盘。
    ///
    /// 由 LLAMA_MOE_WINDOW_MAX_ROWS 给出想要值，再由 resolve_decode_window() 按
    /// 缓存区容量夹紧 —— 夹紧是必须的：装不下时 process() 是抛异常（运行期崩），
    /// 不是降级。上限另外受「批宽 < 32」约束：再大就走 mul_mat_id 路径，寻址不到
    /// 每层的 `.dec` 视图。
    int32_t window_max_rows = 1;
    /// 每层**预读暂存区**的 slot 数（= 预读深度 K）。
    ///
    /// 为什么单列一块：预读的条目是「下一步可能用到」，与「跨 token 复用」是两种
    /// 不同的价值。共用一个 LRU 时，预读会在窗口满时**把跨 token 的缓存条目淘汰掉**
    /// —— 等于预读破坏缓存。分区之后两者互不干扰，覆盖变成**并集**：
    ///   缓存区 [window_start, window_start + c)          跨 token LRU
    ///   预读区 [window_start + c, window_start + c + K)  每步重填，不参与 LRU 淘汰
    int32_t prefetch_slots = 0;
    /// 每层窗口的总宽度（c_il + K），下标 = layer index。见 layer_state::window_width。
    std::vector<int32_t> layer_width;
    /// 统一 LRU：预读与主路径共用**整个窗口**（不分区）。**默认开启**。
    ///
    /// 为什么分区反而慢（实测，每 24 步）：
    ///   分区   miss 53 | 预读写入 3022 | 用到的专家来自预读区 4568 | pmgmt 174ms
    ///   统一   miss 29 | 预读写入 1932 | 用到的专家来自预读区  888 | pmgmt 101ms
    /// 分区下预读区只保留 ~3 步（每步被重填），而 78% 被用到的专家都落在那里 ——
    /// 于是「热专家」用完就被冲掉、下次重读。统一 LRU 让它们自然进入跨 token LRU。
    ///
    /// 分区当初是为了防「预读把缓存成批挤出去」，但那个前提已被 arm 时的过滤消掉：
    /// 预读每层每步只新增约 2.4 个（而不是 8 个），不会造成成批淘汰。
    bool prefetch_unified = true;
    bool pf_probation = false;
    bool was_window = false;
    int protected_layer = -1;
    int last_moe_layer = -1;
    int first_moe_layer = -1;
    uint64_t graph_epoch = 0;
    bool h1_front = false;
    uint64_t h1_front_eligible = 0, h1_front_attempted = 0, h1_front_candidate_ids = 0;
    uint64_t n_prefill_ubatch = 0;
    uint64_t n_decode_step = 0;
    uint64_t window_unique = 0;
    uint64_t window_miss = 0;
    uint64_t bank_unique = 0;
    uint64_t bank_miss = 0;
    ubatch_acc cur;
    run_stats g_runs;   // 整进程累计；process() 只在 ith==0 上跑，单线程，无需原子
    std::vector<layer_state> layers;
    std::vector<map_userdata> userdata;
    std::unordered_map<std::string, int> fds;         // 按路径缓存的 fd（缓冲或 O_DIRECT，全进程一致）
    std::mutex fd_mutex;
    std::mutex io_mutex;
    io_stats stats;
    moe_handoff_trace handoff_trace;
    std::atomic<uint64_t> manager_coalesced{0}, manager_promoted{0}, manager_joined{0}, manager_reused{0};
    std::atomic<uint64_t> manager_reads{0}, manager_evicted{0}, manager_stale{0}, manager_failed{0};
    std::atomic<uint64_t> manager_demand_reads{0}, manager_prefetch_reads{0}, manager_prewarm_reads{0}, manager_retry_reads{0};
    std::atomic<uint64_t> manager_quota_skipped{0}, manager_capacity_skipped{0};
    // Must outlive lookahead workers, including constructor unwinding.
    std::unique_ptr<llama_moe::io_admission> io_budget;
    std::vector<llama_moe::io_admission::key> io_needed_snapshot;
    lookahead_expert_prefetch lookahead;
    ggml_tensor * lookahead_pending = nullptr;

    std::atomic<bool> parked{false};
    std::mutex park_mutex;

    // ── 命中率扫描（诊断，默认关闭）──────────────────────────────────────
    // LLAMA_MOE_SWEEP=<path> 打开，把每个 decode 步、每一层的原始记录写成一行：
    //
    //   L<layer> E:<本层真实用的专家> P:<上一层给本层预测的专家（按分数降序）>
    //
    // 为什么 dump 原始数据而不是在 C++ 里算表：命中率只取决于路由本身，与硬件无关，
    // 所以可以在 Mac 上跑一次、把文件拿来随便分析。dump 之后 cap / K 想怎么扫都行，
    // 不用为每个组合重编译。分析脚本见项目文档里的 analyze_hits.py。
    FILE * sweep_fp = nullptr;
    std::vector<std::vector<int32_t>> sweep_pred;   // sweep_pred[L] = 给层 L 的预测（来自层 L-1）

    // ── 缓存预热表（LLAMA_MOE_PREWARM=<path>）────────────────────────────
    // 文件：每行一层，行号 = 模型层号（与 sweep dump 的 `L<layer>` 同一编号），
    // 行内是逗号/空白分隔的专家 id（`#` 起注释，空行/注释行不占层号）。
    // 第一行 = 层 0，第二行 = 层 1……
    std::vector<std::vector<int32_t>> prewarm_table;
    bool prewarm_enabled = false;
    bool prefill_warm_enabled = false;

    /// 预热读入后的落位记录（层、专家、窗口内 slot 下标）。
    /// 阶段切换时用它「重新发布」映射 —— 数据没被 prefill 碰（物理分离），只有映射被清。
    struct prewarm_place { int32_t layer; int32_t expert; int32_t slot; };
    std::vector<prewarm_place> prewarm_placed;
    /// 是否已读入且尚未被阶段切换消费。consumed 时置回 false（一次性），
    /// park（每轮结束）也会清掉 —— 下一轮必须重新 prewarm_experts()。
    bool prewarm_ready = false;

    void prewarm_init() {
        const char * path = std::getenv("LLAMA_MOE_PREWARM");
        if (!path || !*path) {
            return;
        }
        FILE * fp = std::fopen(path, "r");
        if (!fp) {
            moe_log("prewarm: 打不开表文件 %s", path);
            return;
        }
        prewarm_table.assign(layers.size(), {});
        char buf[16384];
        int li = 0;
        int total = 0;
        while (li < (int) layers.size() && std::fgets(buf, sizeof(buf), fp)) {
            const char * p = buf;
            while (*p == ' ' || *p == '\t') {
                ++p;
            }
            if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') {
                continue;   // 注释 / 空行不占层号
            }
            std::vector<int32_t> ids;
            while (*p) {
                while (*p == ' ' || *p == '\t' || *p == ',') {
                    ++p;
                }
                if (!*p || *p == '\n' || *p == '\r' || *p == '#') {
                    break;
                }
                char * end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) {
                    ++p;
                    continue;
                }
                if (v >= 0 && v < n_expert) {
                    ids.push_back((int32_t) v);
                }
                p = end;
            }
            prewarm_table[(size_t) li] = std::move(ids);
            total += (int) prewarm_table[(size_t) li].size();
            li++;
        }
        std::fclose(fp);
        prewarm_enabled = total > 0 && decode_window > 0;
        moe_log("prewarm: 表 %s -> %d 层 / %d 个专家, decode_window=%d, enabled=%s",
                path, li, total, (int) decode_window, prewarm_enabled ? "on" : "off");
        if (decode_window <= 0) {
            moe_log("prewarm: 窗口未开（LLAMA_MOE_DECODE_WINDOW=0），预热不会生效");
        }
        // 持久 pin：把热表钉进窗口，LRU 永不淘汰（`LLAMA_MOE_PREWARM_PIN=0` 可关）。
        // 与每步清的 protected_experts 不同，这份是常驻的。
        if (prewarm_enabled) {
            const char * pv = std::getenv("LLAMA_MOE_PREWARM_PIN");
            const bool pin = !(pv && *pv && *pv == '0');
            int n_pin_layers = 0, n_pin_total = 0;
            for (size_t il = 0; il < prewarm_table.size() && il < layers.size(); ++il) {
                auto & st = layers[il];
                std::fill(st.pinned_experts.begin(), st.pinned_experts.end(), 0);
                if (!pin) {
                    continue;
                }
                int n_pin = 0;
                for (int32_t e : prewarm_table[il]) {
                    if (e >= 0 && e < (int32_t) st.pinned_experts.size() && !st.pinned_experts[(size_t) e]) {
                        st.pinned_experts[(size_t) e] = 1;
                        ++n_pin;
                    }
                }
                const int32_t cap = st.window_width > 0 ? st.window_width - prefetch_slots : 0;
                if (cap > 0 && n_pin > cap) {
                    moe_log("prewarm: layer %zu 的 pin %d 超过容量 %d，本层不 pin", il, n_pin, (int) cap);
                    std::fill(st.pinned_experts.begin(), st.pinned_experts.end(), 0);
                    n_pin = 0;
                }
                if (n_pin > 0) {
                    ++n_pin_layers;
                    n_pin_total += n_pin;
                }
            }
            moe_log("prewarm: pin %s（%d 层 / %d 个专家常驻，LRU 不淘汰）",
                    pin ? "on" : "off", n_pin_layers, n_pin_total);
        }
    }

    void sweep_init() {
        const char * path = std::getenv("LLAMA_MOE_SWEEP");
        if (!path || !*path) {
            return;
        }
        sweep_fp = fopen(path, "w");
        if (!sweep_fp) {
            moe_log("sweep: 无法打开 %s", path);
            return;
        }
        sweep_pred.assign(layers.size(), {});
        fprintf(sweep_fp, "# layer real_experts | predicted_experts(desc)\n");
        moe_log("sweep: 记录到 %s（每 decode 步每层一行）", path);
    }

    void sweep_write(int layer, const std::vector<int32_t> & used, const std::vector<int32_t> & pred) {
        if (!sweep_fp) {
            return;
        }
        fprintf(sweep_fp, "L%d E:", layer);
        for (size_t i = 0; i < used.size(); ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", used[i]);
        }
        fprintf(sweep_fp, " P:");
        for (size_t i = 0; i < pred.size(); ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", pred[i]);
        }
        fprintf(sweep_fp, "\n");
    }

    /// prefill 时记录**最后一个 token** 用的专家（`P<layer> E:...`）。
    ///
    /// 用途：验证「prefill 能否给 decode 留一份初始缓存」。
    /// prefill 每层要用 100+ 个专家，装不进按层窗口（c=24），所以 prefill 必须用
    /// 整块 bank、逐层清空 —— 结束时 bank 里只剩最后一层的数据。但**最后一个 token
    /// 用的 8 个专家是 decode 第一步最可能的邻居**，可以单独留下来当种子。
    void sweep_write_prefill(int layer, const int32_t * ids, int n_ids, int n_tokens) {
        if (!sweep_fp || n_tokens <= 0) {
            return;
        }
        const int32_t * last = ids + (size_t) (n_tokens - 1) * n_ids;
        fprintf(sweep_fp, "P%d E:", layer);
        for (int i = 0; i < n_ids; ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", last[i]);
        }
        fprintf(sweep_fp, "\n");
    }

    std::vector<ggml_backend_buffer_t> expert_buffers() const {
        std::set<ggml_backend_buffer_t> uniq;
        auto add = [&](const std::vector<binding> & bs) {
            for (const auto & b : bs) {
                ggml_tensor * t = b.tensor;
                if (!t) {
                    continue;
                }
                ggml_backend_buffer_t buf = t->buffer;
                if (!buf && t->view_src) {
                    buf = t->view_src->buffer;
                }
                if (buf) {
                    uniq.insert(buf);
                }
            }
        };
        for (const auto & st : layers) {
            add(st.bindings);
            add(st.dec_bindings);
        }
        return { uniq.begin(), uniq.end() };
    }

    void park_expert_memory() {
        if (!park_window_enabled()) {
            return;
        }
        lookahead.drain();
        std::lock_guard<std::mutex> lk(park_mutex);
        if (parked.load(std::memory_order_relaxed)) {
            return;
        }
        // 一轮结束：窗口数据随本轮 decode churn 过，落位记录作废。
        prewarm_ready = false;
        prewarm_placed.clear();
        int n_ok = 0;
        for (ggml_backend_buffer_t buf : expert_buffers()) {
            metal_buf_fn fn = lookup_metal_buf_fn(buf, "ggml_backend_metal_buffer_park");
            if (fn && fn(buf)) {
                n_ok++;
            }
        }
        for (auto & st : layers) {
            if (st.enabled) {
                std::lock_guard<std::mutex> state_lock(st.mutex);
                if (st.manager) {
                    st.manager->release_selected(); st.manager->reset_binding();
                    handoff_trace.emit("BINDING_RESET", {0, graph_epoch, (int32_t) (&st - layers.data()), 0, "control", 0,
                        "window", st.manager->binding_generation()});
                }
                clear_layer_cache(st);
            }
        }
        protected_layer = -1;
        was_window = false;
        parked.store(true, std::memory_order_release);
        moe_log("park expert memory: %d metal buffer(s)", n_ok);
    }

    void ensure_unparked() {
        if (!parked.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> lk(park_mutex);
        if (!parked.load(std::memory_order_relaxed)) {
            return;
        }
        for (ggml_backend_buffer_t buf : expert_buffers()) {
            metal_buf_fn fn = lookup_metal_buf_fn(buf, "ggml_backend_metal_buffer_unpark");
            if (fn && !fn(buf)) {
                throw std::runtime_error("failed to unpark MoE expert Metal buffer");
            }
        }
        parked.store(false, std::memory_order_release);
        moe_log("unparked expert memory");
    }
};

void llama_moe_slot_runtime::map_custom(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    // 这里在**图执行期间**被调用（CPU 自定义 op），所以这段耗时直接落在关键路径上。
    // 每层一次、每 token 40 次，量出来才能知道它是不是瓶颈。
    const int64_t t0 = now_ns();
    data->runtime->process(data->layer, src, dst);
    data->runtime->add_process_ns(now_ns() - t0);
}

void llama_moe_slot_runtime::map_custom_lookahead(
        ggml_tensor * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    const int64_t t0 = now_ns();
    data->runtime->process(data->layer, a, dst, b);
    data->runtime->add_process_ns(now_ns() - t0);
}

llama_moe_slot_runtime::llama_moe_slot_runtime(const llama_model & model) :
    pimpl(std::make_unique<impl>(model)) {
    for (auto & item : pimpl->userdata) {
        item.runtime = this;
    }
}

llama_moe_slot_runtime::~llama_moe_slot_runtime() = default;

bool llama_moe_slot_runtime::uses_layer(int layer) const {
    return layer >= 0 && layer < (int) pimpl->layers.size() && pimpl->layers[layer].enabled;
}

int llama_moe_slot_runtime::matmul_split() const {
    return pimpl->matmul_split;
}

/// A2 的 phase B 节点：把 slot_ids 拷一份作为输出（这样本节点自然依赖 phase A），
/// 同时依赖前段专家的 matmul 输出 `b` —— 于是调度器会把它排在前段算完之后、
/// 后段 matmul 之前执行。
void llama_moe_slot_runtime::map_custom_deferred(
        ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
        int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(b);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    memcpy(dst->data, a->data, ggml_nbytes(a));
    data->runtime->process_deferred(data->layer);
}

ggml_tensor * llama_moe_slot_runtime::build_deferred_trigger(
        ggml_context * ctx, ggml_tensor * slot_ids, ggml_tensor * dep, int layer) {
    return ggml_map_custom2(ctx, slot_ids, dep, map_custom_deferred, 1, &pimpl->userdata[(size_t) layer]);
}

void llama_moe_slot_runtime::add_process_ns(int64_t ns) {
    pimpl->process_ns.fetch_add(ns, std::memory_order_relaxed);
}

bool llama_moe_slot_runtime::lookahead_prefetch_enabled() const {
    return pimpl->lookahead.enabled;
}

int llama_moe_slot_runtime::lookahead_prefetch_depth() const {
    return pimpl->lookahead.enabled ? (int) pimpl->lookahead.depth : 0;
}

/// 跨层流水 horizon：一次性把接下来 horizon 层的 gate 都算出来（都用当前层的
/// attn_post_norm 当 hidden 的近似），于是读盘可以提前 horizon 层发起。
/// **默认 2**；1 = 旧行为；上限 8（再多预测精度会崩，见 exps/pipe-in-layer 的 recall(k)）。
int llama_moe_slot_runtime::lookahead_horizon() const {
    // 这里**直接给值**，不再看 pimpl->lookahead.enabled。
    //
    // 原因（2026-11-05 由独立评审纠正过一次）：**在启用预取且 depth 有效时**，
    // `enabled` 在构造期就由 init() → set_depth() 置 true（见 `:424` → `:451`），
    // 建图时早已是 true；env 关掉预取、或 bank 太小导致 depth 归一化为 0 时才是 false。
    // 而唯一调用方 qwen35moe 外层已经用 `lookahead_prefetch_enabled()`（= 同一个 enabled）
    // 闸过一遍：它为 false 时**一块预测都不建**。所以原来的 `if (!enabled) return 1;`
    // 在可达路径上不可达，删掉它是去冗余检查，**不是** bug 修复 —— 曾一度误当成
    // 「horizon 从未生效」的证据，已更正（exps/pipe-in-layer-20261105/README.md §3.33）。
    // 保留这条约束的语义：建不建块由调用方判，这里只负责给值。
    return moe_lookahead_horizon();
}

// Opt-in: speculative verification can overlap next-layer expert I/O too.
// Clamp to the established window capacity so prefill stays on its shared bank.
int llama_moe_slot_runtime::lookahead_prefetch_max_rows() const {
    const char * value = std::getenv("LLAMA_MOE_PREFETCH_MAX_ROWS");
    const int requested = value ? std::atoi(value) : 1;
    return std::max(1, std::min(requested, (int) pimpl->window_max_rows));
}

int llama_moe_slot_runtime::decode_window() const {
    return (int) pimpl->decode_window;
}

int llama_moe_slot_runtime::window_max_rows() const {
    return (int) pimpl->window_max_rows;
}

void llama_moe_slot_runtime::park_expert_memory() {
    pimpl->park_expert_memory();
}

void llama_moe_slot_runtime::ensure_unparked() {
    pimpl->ensure_unparked();
}

void llama_moe_slot_runtime::set_prefill_warm_enabled(bool enabled) {
    pimpl->prefill_warm_enabled = enabled;
}

void llama_moe_slot_runtime::prewarm_experts() {
    pimpl->prewarm_experts();
}

llama_moe_slot_runtime::counters llama_moe_slot_runtime::counters_snapshot() const {
    counters c;
    c.process  = pimpl->process_ns.load(std::memory_order_relaxed);
    c.readids  = pimpl->readids_ns.load(std::memory_order_relaxed);
    c.install  = pimpl->install_ns.load(std::memory_order_relaxed);
    c.writeids = pimpl->writeids_ns.load(std::memory_order_relaxed);
    c.miss     = (int64_t) pimpl->stats.n_miss.load(std::memory_order_relaxed);
    c.bytes    = (int64_t) pimpl->stats.n_bytes.load(std::memory_order_relaxed);
    c.unique   = pimpl->unique_sum.load(std::memory_order_relaxed);
    return c;
}

void llama_moe_slot_runtime::set_lookahead_prefetch_depth(int depth) {
    if (pimpl->managers_active()) {
        if (depth > 0 && (!pimpl->lookahead.window_mode || depth > pimpl->prefetch_slots)) {
            moe_log("slot-manager: reject depth=%d (reserved=%d window=%d); tensor layout rebuild required",
                    depth, pimpl->prefetch_slots, (int) pimpl->lookahead.window_mode);
            return;
        }
        // A phase/depth mutation never races an old task writing its binding.
        pimpl->lookahead.drain();
        if (depth <= 0) for (auto & state : pimpl->layers) {
            if (!state.manager) continue;
            std::lock_guard<std::mutex> lock(state.mutex); state.manager->set_hints({});
        }
    }
    pimpl->lookahead.set_depth(depth);
    if (pimpl->managers_active()) for (auto & state : pimpl->layers) {
        if (!state.manager) continue;
        std::lock_guard<std::mutex> lock(state.mutex);
        state.manager->set_prefetch_limit(pimpl->lookahead.enabled ? pimpl->lookahead.depth : 0);
    }
}

void llama_moe_slot_runtime::set_lookahead_pred(ggml_tensor * pred_ids) {
    pimpl->lookahead_pending = pred_ids;
}

ggml_tensor * llama_moe_slot_runtime::build_slot_ids_tensor(
        ggml_context * ctx,
        ggml_tensor * expert_ids,
        int layer) {
    if (!uses_layer(layer)) {
        return expert_ids;
    }
    ggml_tensor * pred = pimpl->lookahead_pending;
    pimpl->lookahead_pending = nullptr;
    if (pimpl->lookahead.enabled && pred) {
        return ggml_map_custom2(ctx, expert_ids, pred, map_custom_lookahead, 1, &pimpl->userdata[(size_t) layer]);
    }
    return ggml_map_custom1(ctx, expert_ids, map_custom, 1, &pimpl->userdata[layer]);
}

void llama_moe_slot_runtime::process(
        int layer,
        const ggml_tensor * expert_ids,
        ggml_tensor * slot_ids,
        const ggml_tensor * lookahead_pred) {
    pimpl->ensure_unparked();
    if (!uses_layer(layer) ||
        expert_ids->type != GGML_TYPE_I32 ||
        slot_ids->type != GGML_TYPE_I32 ||
        expert_ids->ne[0] != slot_ids->ne[0] ||
        expert_ids->ne[1] != slot_ids->ne[1]) {
        throw std::runtime_error(format(
            "invalid MoE slot map tensors at layer %d type=%s/%s shape=%lldx%lld vs %lldx%lld",
            layer,
            ggml_type_name(expert_ids ? expert_ids->type : GGML_TYPE_COUNT),
            ggml_type_name(slot_ids ? slot_ids->type : GGML_TYPE_COUNT),
            expert_ids ? (long long) expert_ids->ne[0] : -1,
            expert_ids ? (long long) expert_ids->ne[1] : -1,
            slot_ids ? (long long) slot_ids->ne[0] : -1,
            slot_ids ? (long long) slot_ids->ne[1] : -1));
    }

    if (layer == pimpl->first_moe_layer) pimpl->graph_epoch++;
    const moe_handoff_context trace_context = {0, pimpl->graph_epoch, layer, (int32_t) expert_ids->ne[1], "demand", 0,
        pimpl->decode_window > 0 && expert_ids->ne[1] <= pimpl->window_max_rows ? "window" : "bank"};
    moe_handoff_scope process_scope(pimpl->handoff_trace.enabled() ? &pimpl->handoff_trace : nullptr, trace_context);
    moe_handoff_span callback(pimpl->handoff_trace, "CALLBACK_BEGIN", "CALLBACK_DONE", trace_context);
    pimpl->lookahead.observe_demand(layer, pimpl->graph_epoch, (int) expert_ids->ne[1]);

    // This callback depends on the preceding layer output, so the previous
    // layer weights are no longer in use. Keep current weights protected from
    // late prefetch eviction until the next dependent callback is reached.
    if (pimpl->protected_layer >= 0) {
        auto & previous = pimpl->layers[(size_t) pimpl->protected_layer];
        std::lock_guard<std::mutex> release(previous.mutex);
        if (previous.manager) {
            if (pimpl->handoff_trace.enabled()) {
                auto c = trace_context; c.layer = pimpl->protected_layer; c.domain = "window";
                for (size_t i = 0; i < previous.manager->selected_size(); ++i) {
                    const int32_t e = previous.manager->selected_ids()[i]; const auto resident = previous.manager->lookup(e);
                    pimpl->handoff_trace.emit("LEASE_RELEASE", handoff_request(c, resident.request), e, resident.request.slot);
                }
            }
            previous.manager->release_selected();
        } else std::fill(previous.protected_experts.begin(), previous.protected_experts.end(), 0);
        pimpl->protected_layer = -1;
    }
    auto & state = pimpl->layers[layer];

    const int32_t n_rows = (int32_t) expert_ids->ne[1];

    const bool decode = n_rows == 1;
    // 窗口也覆盖投机验证这类**小批次**（批宽 = n_max+1）：它们与单 token decode 走
    // 同一条 mul_mv_id 路径、同一套每层 `.dec` 张量，跨轮复用同一批专家。
    // 大批次（prefill）不走 —— 它拿的是 prefill 共享视图，只能用 [0, prefill_cap) 那段。
    //
    // 上限由 window_max_rows 按缓存区容量夹紧（见 resolve_decode_window），
    // 「并集装不下」的风险在那里就被消掉，不会落到这里的运行期。
    const bool window_mode = pimpl->decode_window > 0 && n_rows <= pimpl->window_max_rows;
    const bool managed = window_mode && state.manager != nullptr;
    const bool phase_reuse = pimpl->model.moe_phase_reuse();
    const bool lookahead_step = pimpl->lookahead.enabled && n_rows <= lookahead_prefetch_max_rows() &&
                                (!phase_reuse || window_mode);

    // **只在窗口模式下**需要在加锁之前等预读收工。
    //
    // 窗口模式里预读 worker 会锁**目标层**的 mutex 去写它的窗口；若本函数已持
    // state.mutex 再等 worker，就成了锁序反转 → 死锁（实际踩到过）。
    // 非窗口模式 worker 只写尾部乒乓区、不碰任何层锁，所以照旧在锁后 take 取结果。
    if (lookahead_step && window_mode) {
        std::vector<lookahead_expert_prefetch::hit> ignored;   // 窗口模式不返回 hit
        pimpl->lookahead.take(layer, ignored);
    }

    if (pimpl->managers_active() && window_mode != pimpl->was_window) {
        pimpl->lookahead.reset(); pimpl->lookahead.drain();
        if (phase_reuse) {
            for (size_t il = 0; il < pimpl->layers.size(); ++il) {
                auto & old = pimpl->layers[il]; if (!old.manager) continue;
                std::lock_guard<std::mutex> reset(old.mutex); old.manager->reset_binding();
                auto c = trace_context; c.layer = (int32_t) il; c.binding = old.manager->binding_generation(); c.domain = "window";
                pimpl->handoff_trace.emit("BINDING_RESET", c);
            }
        }
    }

    if (!window_mode && ((pimpl->prefill_warm_enabled && pimpl->prefill_warm_rows() > 0) || phase_reuse) && pimpl->lookahead.enabled) {
        // Drain before taking a layer lock: the worker also locks its target.
        pimpl->lookahead.reset();
        pimpl->lookahead.drain();
    }

    if (phase_reuse && !window_mode) {
        // Bank writes invalidate any placement snapshot, including one left by
        // an initialization probe. Window entry will clear all live mappings.
        pimpl->prewarm_ready = false;
        pimpl->prewarm_placed.clear();
    }

    pimpl->handoff_trace.emit("PROCESS_LOCK_WAIT", trace_context);
    std::unique_lock<std::mutex> lock(state.mutex);
    moe_handoff_lock_scope process_lock(pimpl->handoff_trace, lock, trace_context);

    // 阶段切换：prefill 用的是整块 bank / prefill 共享视图，与每层窗口在物理上可能
    // 落在同一块内存，所以**进入窗口阶段时**必须把每层的映射整表清空。
    //
    // 只清窗口内那 W 个 slot 是不够的：prefill 留下的 `expert_to_slot` 里还有大量
    // 指向**窗口之外**的陈旧映射，那些专家会被 process() 误判成「已在内存」而跳过安装，
    // 于是 matmul 读到的是别的专家权重 —— 输出错但不报错。（这是实际踩到的 bug。）
    if (window_mode && !pimpl->was_window) {
        for (auto & st : pimpl->layers) {
            if (!st.enabled) {
                continue;
            }
            // 当前层（layer 0）的 mutex 已经被上面那个 lock_guard 持有了，
            // 再锁一次会**自死锁**（std::mutex 不可重入）。它由下面那段直接清。
            if (&st == &state) {
                continue;
            }
            std::lock_guard<std::mutex> lk(st.mutex);
            llama_moe_slot_runtime::impl::clear_layer_cache(st);
        }
        // 当前层自己在锁内清（不用再锁）
        llama_moe_slot_runtime::impl::clear_layer_cache(state);

        // 缓存预热：窗口刚开、缓存刚清空（每轮生成都会走到这里一次）。
        //
        // 若这轮已经在 prefill 之前调过 `prewarm_experts()`（App 的正常路径），
        // 那批专家**已经读进窗口**且 prefill 碰不到它（decode 段与 prefill 段物理分离），
        // 这里只需**重新发布映射**（纯整数操作、零 I/O）；否则退回「现在读」的老路径。
        // 此刻 layer 0 的锁已被本函数持有，两个函数对当前层都不再加锁。
        if (pimpl->prewarm_ready) {
            const size_t n_before = pimpl->prewarm_placed.size();
            pimpl->publish_prewarm_placed(layer);
            moe_log("prewarm: 重挂 %zu 个已预热专家（无 I/O，数据活过 prefill）", n_before);
        } else {
            pimpl->prewarm_windows(layer);
        }
        // Both paths have installed live mappings. Window computation can now
        // evict these slots, so their placement snapshot must not be republished
        // after a later prefill (including the initialization seq-rm probe).
        pimpl->prewarm_ready = false;
    }
    // 阶段判据必须是 `window_mode` 而不是 `decode`：投机验证（批宽 = n_max+1）与
    // 单 token decode 属于**同一个阶段**，它们之间不能清缓存 —— 否则窗口每步都被
    // 清空，跨轮复用无从谈起（实测那样会让窗口命中率归零）。
    pimpl->was_window = window_mode;

    if (pimpl->shared_bank && !window_mode) {
        llama_moe_slot_runtime::impl::clear_layer_cache(state);
    }
    std::vector<int32_t> experts;
    {
        const int64_t t0 = now_ns();
        pimpl->read_i32_ids(expert_ids, experts);
        pimpl->readids_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
    const size_t count = experts.size();
    std::vector<int32_t> slots(count);
    std::vector<uint8_t> requested(pimpl->n_expert, 0);

    // 诊断：这个 token 里哪些**位置**会 miss（该位置还不在缓存里）。
    // 注意此时槽位分配还没做，所以只用 expert_to_slot 判断「已驻留」。
    if (window_mode && expert_ids->ne[1] == 1) {
        pimpl->miss_pos_steps++;
        for (size_t i = 0; i < count && i < 16; ++i) {
            if (experts[i] >= 0 && experts[i] < pimpl->n_expert &&
                pimpl->cached_slot(state, experts[i], window_mode) < 0) {
                pimpl->miss_pos[i]++;
            }
        }
    }

    int32_t resident_at_lookup = 0;
    for (size_t i = 0; i < count; ++i) {
        if (experts[i] < 0 || experts[i] >= pimpl->n_expert) {
            throw std::runtime_error("MoE expert id is out of range");
        }
        if (pimpl->handoff_trace.enabled()) pimpl->handoff_trace.emit("DEMAND_ROUTE", trace_context, experts[i], -1, (int32_t) i);
        if (!requested[(size_t) experts[i]]) {
            if (managed) {
                state.requests[(size_t) experts[i]] = {};
                resident_at_lookup += state.manager->resident_slot(experts[i]) >= 0 ? 1 : 0;
            }
            if (pimpl->handoff_trace.enabled()) {
                const auto o = managed ? state.manager->lookup(experts[i]) : llama_moe::slot_manager::observation{};
                const int32_t slot = pimpl->cached_slot(state, experts[i], window_mode);
                const auto name = slot >= 0 ? "LOOKUP_READY" : managed && o.state == llama_moe::slot_manager::phase::queued ? "LOOKUP_QUEUED" :
                    managed && o.state == llama_moe::slot_manager::phase::loading ? "LOOKUP_LOADING" : "LOOKUP_MISS";
                pimpl->handoff_trace.emit(name, handoff_request(trace_context, o.request), experts[i], slot >= 0 ? slot : o.request.slot);
            }
        }
        requested[experts[i]] = 1;
    }

    if (managed) {
        state.manager->pin_selected(experts.data(), experts.size());
        pimpl->protected_layer = layer;
    }
    struct selected_cleanup {
        llama_moe_slot_runtime::impl & owner;
        llama_moe_slot_runtime::impl::layer_state & state;
        moe_handoff_lock_scope & lock;
        const std::vector<int32_t> & experts;
        moe_handoff_context context;
        bool active, success = false;
        ~selected_cleanup() {
            if (!active || success) return;
            lock.acquire();
            for (int32_t e : experts) {
                const auto & r = state.requests[(size_t) e];
                if (r.owner) owner.finish_managed(state, r.request, context, false, {});
            }
            state.manager->release_selected(); owner.protected_layer = -1;
        }
    } selected_guard{*pimpl, state, process_lock, experts, trace_context, managed};

    // ── recall(k)：这份预测是提前 k 层提交的，命中了多少 ────────────────
    // 只有实测才知道 horizon 能推到几层：k 越大，用 h_il 近似 h_{il+k} 越失真。
    if (pimpl->lookahead.enabled && layer < (int) pimpl->pred_hist.size()) {
        std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
        for (const auto & entry : pimpl->pred_hist[(size_t) layer]) {
            const int32_t k = layer - entry.first;
            if (k < 1 || k > 8) {
                continue;
            }
            uint64_t hit = 0;
            for (int32_t e : entry.second) {
                if (e >= 0 && e < pimpl->n_expert && requested[(size_t) e]) {
                    hit++;
                }
            }
            pimpl->recall_hit[k]  += hit;
            pimpl->recall_pred[k] += entry.second.size();
        }
        pimpl->pred_hist[(size_t) layer].clear();
    }

    if (pimpl->lookahead.enabled && !lookahead_step) {
        pimpl->lookahead.reset();
    }
    // 非窗口模式：在这里取预读结果（命中已在乒乓区，需搬到目标 slot）。
    // 窗口模式的结果由 worker 直接写进目标窗口了，这里无事可做。
    if (lookahead_step && !window_mode) {
        std::vector<lookahead_expert_prefetch::hit> hits;
        pimpl->lookahead.take(layer, hits);
        for (const auto & h : hits) {
            if (h.expert < 0 || h.expert >= pimpl->n_expert) {
                continue;
            }
            if (!requested[(size_t) h.expert] || state.expert_to_slot[h.expert] >= 0) {
                continue;
            }
            state.slot_to_expert[h.slot] = h.expert;
            state.expert_to_slot[h.expert] = h.slot;
            state.slot_age[h.slot] = ++state.age;
            pimpl->lookahead.n_hits.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // 分配范围。窗口模式下只在**本层窗口**内找空位 / 淘汰 —— 这是整个机制的要点：
    // 每层有独立的 slot 段，所以层间的 LRU 互不干扰，缓存能跨 token 存活。
    // 非窗口模式仍是整块 bank（预读开启时让出尾部乒乓区）。
    int32_t alloc_lo = 0;
    int32_t alloc_hi = pimpl->n_slots;
    if (window_mode) {
        alloc_lo = state.window_start;
        // 分区模式：主路径只从缓存区分配，暂存区留给预读（互不挤占）。
        // 统一模式：主路径也可用整个窗口 —— 用到的专家因此天然进入跨 token LRU，
        // 不会被「预读区每步重填」冲掉。
        // 边界用**本层宽度**（各层不同，见 layer_state::window_width）。
        const int32_t w_il = state.window_width > 0
            ? state.window_width : (pimpl->decode_window + pimpl->prefetch_slots);
        alloc_hi = state.window_start + w_il -
                   (pimpl->prefetch_unified ? 0 : pimpl->prefetch_slots);
    } else if (lookahead_step && !pimpl->lookahead.window_mode) {
        alloc_hi = pimpl->lookahead.slot_floor();
    }

    // 「缓存 + 预测」的结合点（零 I/O 成本）：
    // 上一层给出的预测里，有些专家本层这次没用到、但**下一步**会用。淘汰时必须
    // 避开它们 —— 否则「刚被预测会用」的条目恰好是最老的，正好被换出去。
    //
    // 这不是预读（不读任何字节），只是把已有的预测信号用来改进淘汰决策。
    std::vector<uint8_t> keep;
    if (window_mode && pimpl->lookahead.enabled) {
        std::vector<int32_t> pred;
        {
            std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
            auto & armed = pimpl->predicted[(size_t) layer];
            auto & hint  = pimpl->pred_hint[(size_t) layer];
            if (!armed.empty()) {
                pred.swap(armed);          // 用完即清（与老实现一致）
            }
            if (!hint.empty()) {
                for (int32_t e : hint) { pred.push_back(e); }
                hint.clear();              // 用完即清
            }
        }
        if (managed) state.manager->set_hints(pred);
        if (!managed && !pred.empty()) {
            keep.assign((size_t) pimpl->n_expert, 0);
            for (int32_t e : pred) {
                if (e >= 0 && e < pimpl->n_expert) {
                    keep[(size_t) e] = 1;
                }
            }
        }
    }

    std::vector<llama_moe_slot_runtime::impl::pending_install> pending;
    pending.reserve((size_t) std::min(managed ? alloc_hi - alloc_lo : (int32_t) pimpl->n_slots, pimpl->n_expert));

    // A2：按**名次**决定「现在读」还是「延后读」。末尾 matmul_split 个**名次**的
    // 缺失专家延后到 phase B —— 实测 miss 高度集中在尾部（名次 7 的 miss 是名次 0
    // 的 7 倍），那里最可能真的有东西要读。
    //
    // 按**固定名次**切而不是按「是否驻留」排序切，是为了保持求和顺序不变 ——
    // 输出因此可以逐字节校验。
    //
    // 判据必须用**名次**（`pos % n_used`）而不是线性下标：展平顺序是
    // `pos = 名次 + n_used * 行号`（ne[0] 变化最快，见 read_i32_ids），所以
    // 「末尾 K 个线性下标」多行时只覆盖最后 K/n_used 行的**全部**名次；而图侧的切分
    // （build_moe_ffn 的 slice_ids）切的是**所有行**的名次 [split_at, n_used)。
    // 两边对不上，phase A 就会读到 phase B 还没写入的槽位（静默读错专家权重）。
    // 单 token 时 pos == 名次，两种写法等价。
    auto & defer_list = pimpl->deferred[(size_t) layer];
    defer_list.clear();
    const int32_t n_used_row = (int32_t) expert_ids->ne[0];
    const int32_t split_at   = n_used_row - pimpl->matmul_split;
    const bool defer_by_rank = window_mode && pimpl->matmul_split > 0 && split_at > 0;

    for (size_t pos = 0; pos < count; ++pos) {
        const int32_t expert = experts[pos];
        if (managed) {
            auto & r = state.requests[(size_t) expert];
            if (!r.request) {
                r = state.manager->acquire(expert, {pimpl->graph_epoch, 0});
                if (!r.request) {
                    pimpl->handoff_trace.emit("SLOT_CAPACITY_FAIL", trace_context, expert);
                    throw std::runtime_error("selected MoE experts exceed unpinned managed window capacity");
                }
                pimpl->trace_managed_claim(state, r, trace_context);
                const auto c = handoff_request(trace_context, r.request);
                if (r.owner) {
                    auto submitted = c; submitted.io_id = pimpl->handoff_trace.new_io_id();
                    pending.push_back({expert, r.request.slot, r.request, submitted.io_id});
                    pimpl->handoff_trace.emit("DEMAND_READ_SUBMIT", submitted, expert, r.request.slot);
                } else if (r.action == llama_moe::slot_manager::outcome::joined) {
                    ++pimpl->manager_joined; pimpl->handoff_trace.emit("JOIN_LOADING", c, expert, r.request.slot);
                } else { ++pimpl->manager_reused; pimpl->handoff_trace.emit("RESIDENT_REUSE", c, expert, r.request.slot); }
            }
            slots[pos] = r.request.slot; continue;
        }
        if (state.expert_to_slot[expert] >= 0) {
            slots[pos] = state.expert_to_slot[expert];   // 已驻留，无需读取
            continue;
        }

        int32_t slot = -1;
        for (int32_t candidate = alloc_lo; candidate < alloc_hi; ++candidate) {
            if (state.slot_to_expert[candidate] == kSlotFree) {
                slot = candidate;
                break;
            }
        }
        if (slot < 0) {
            // 淘汰时跳过 kSlotPending：那是预读 worker 正在写的 slot，
            // 动它会让 worker 把内容写到一个已经属于别人的位置。
            //
            // 两轮挑选：第一轮只考虑「预测下一步不会用」的条目（keep[]==0），
            // 第二轮才放宽。这样预测会用的条目能多留一步 —— 零 I/O 成本的收益。
            for (int pass = 0; pass < 2 && slot < 0; ++pass) {
                uint64_t oldest = UINT64_MAX;
                for (int32_t candidate = alloc_lo; candidate < alloc_hi; ++candidate) {
                    const int32_t resident = state.slot_to_expert[candidate];
                    if (resident == kSlotPending) {
                        continue;
                    }
                    if (resident == kSlotFree) {
                        slot = candidate;
                        break;
                    }
                    if (resident < 0 || resident >= pimpl->n_expert) {
                        throw std::runtime_error("invalid resident expert in MoE eviction");
                    }
                    if (pass == 0 && !keep.empty() && keep[(size_t) resident]) {
                        continue;   // 预测下一步要用，先不动它
                    }
                    if (!requested[resident] && state.slot_age[candidate] < oldest) {
                        oldest = state.slot_age[candidate];
                        slot = candidate;
                    }
                }
                if (pass == 0) {
                    if (slot >= 0) {
                        pimpl->lookahead.n_evict_saved.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        // pass0 一个都没找到 —— 说明候选**全在预测集里**，
                        // 也就是「保护预测条目」这个策略真的起了约束作用。
                        pimpl->lookahead.n_evict_forced.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
        if (slot < 0) {
            throw std::runtime_error(format(
                "layer %d requires more than %d unique experts in one graph; reduce ubatch or increase --moe-slot-bank",
                layer, alloc_hi - alloc_lo));
        }

        auto submitted = trace_context; submitted.io_id = pimpl->handoff_trace.new_io_id();
        const int32_t evicted = state.slot_to_expert[slot];
        if (evicted >= 0) {
            pimpl->handoff_trace.emit("LEGACY_SLOT_EVICT", submitted, evicted, slot);
            state.expert_to_slot[evicted] = kSlotFree;
        }
        state.slot_age[slot] = ++state.age;
        slots[pos] = slot;
        const bool defer_this = defer_by_rank &&
            (int32_t) (pos % (size_t) n_used_row) >= split_at;
        if (defer_this) {
            // 延后：槽位标 kSlotPending 占住（别处不会动它），phase B 读完再发布。
            state.slot_to_expert[slot] = kSlotPending;
            defer_list.push_back({ expert, slot, {} });
        } else {
            state.slot_to_expert[slot] = expert;
            state.expert_to_slot[expert] = slot;
            pending.push_back({ expert, slot, {}, submitted.io_id });
            pimpl->handoff_trace.emit("DEMAND_READ_SUBMIT", submitted, expert, slot);
            pimpl->handoff_trace.emit("LEGACY_MAPPING_VISIBLE", submitted, expert, slot);
        }
    }

    if (managed && pimpl->handoff_trace.enabled()) {
        for (size_t i = 0; i < state.manager->selected_size(); ++i) {
            const int32_t expert = state.manager->selected_ids()[i];
            const auto request = state.requests[(size_t) expert].request;
            pimpl->handoff_trace.emit("LEASE_ACQUIRE", handoff_request(trace_context, request), expert, request.slot);
        }
    }
    size_t io_needed_count = 0;
    llama_moe_slot_runtime::impl::io_trace_payload needed_trace{trace_context, -1, 0};
    if (managed && pimpl->io_budget) {
        needed_trace.context.binding = state.manager->binding_generation();
        for (size_t i = 0; i < state.manager->selected_size(); ++i) {
            const int32_t expert = state.manager->selected_ids()[i]; const auto & r = state.requests[(size_t) expert];
            if (r.action != llama_moe::slot_manager::outcome::joined) continue;
            pimpl->io_needed_snapshot[io_needed_count++] = {layer, expert, r.request.binding, r.request.generation};
        }
        needed_trace.needed_count = (int32_t) io_needed_count;
    }
    int n_unique = 0;
    for (int32_t expert = 0; expert < pimpl->n_expert; ++expert) {
        if (requested[expert]) {
            n_unique++;
        }
    }
    pimpl->unique_sum.fetch_add(n_unique, std::memory_order_relaxed);

    // Reading in expert-file order changes only the I/O schedule. Slot assignments
    // and router rank order remain untouched, so the matmul reduction is identical.
    static const bool sort_prefill_reads = [] {
        const char * value = std::getenv("LLAMA_MOE_PREFILL_SORT_READS");
        return value && *value && *value != '0';
    }();
    if (!window_mode && sort_prefill_reads) {
        std::sort(pending.begin(), pending.end(), [](const auto & a, const auto & b) {
            return a.expert < b.expert;
        });
    }

    llama_moe_slot_runtime::impl::install_timing timing;
    {
        moe_handoff_span loading(pimpl->handoff_trace, "LOAD_WAIT_BEGIN", "LOAD_WAIT_END", trace_context);
        const int64_t begin = now_ns();
        if (managed) {
            process_lock.release();
            try {
                {
                    auto needed = pimpl->io_budget ? pimpl->io_budget->begin_needed(pimpl->io_needed_snapshot.data(), io_needed_count, &needed_trace) :
                        llama_moe::io_admission::needed_scope{};
                    timing = pimpl->install_pending(state, pending, true);
                    // All own misses are in flight/completed before joining only the
                    // selected Loading generations; queued PF was stolen above.
                    for (size_t i = 0; i < state.manager->selected_size(); ++i) {
                        const int32_t expert = state.manager->selected_ids()[i];
                        auto & r = state.requests[(size_t) expert];
                        if (r.action == llama_moe::slot_manager::outcome::joined) r.request = pimpl->wait_managed(state, r.request, trace_context, timing);
                    }
                } // Needed scope is cleared before taking L, including exceptions.
                process_lock.acquire();
                for (size_t pos = 0; pos < count; ++pos) {
                    const int32_t expert = experts[pos]; const auto o = state.manager->lookup(expert);
                    if (o.state != llama_moe::slot_manager::phase::resident) throw std::runtime_error("selected MoE request did not become Resident");
                    slots[pos] = o.request.slot;
                }
            } catch (...) { process_lock.acquire(); throw; }
            timing.wall_ns = now_ns() - begin;
        } else timing = pimpl->install_pending(state, pending, window_mode);
        loading.complete();
    }
    pimpl->install_ns.fetch_add(timing.wall_ns, std::memory_order_relaxed);

    if (!pending.empty()) {
        // 仅统计，不影响读取路径。按层取增量，避免 cur 累计后重复计数。
        const auto st = llama_moe_slot_runtime::impl::accumulate_runs(pending);
        llama_moe_slot_runtime::impl::add_runs(pimpl->cur.runs, st);
        llama_moe_slot_runtime::impl::add_runs(pimpl->g_runs, st);
    }

    if (lookahead_step && lookahead_pred) {
        // pred 的形状是 [horizon*depth, rows]：ne0 上连着 horizon 个目标层，
        // 第 kb 块（0 基）= 目标层 `layer + kb + 1`。horizon==1 时与旧行为完全一致。
        const int32_t ranks_total = (int32_t) lookahead_pred->ne[0];
        const int32_t rows        = (int32_t) lookahead_pred->ne[1];
        const int32_t depth       = std::max(1, pimpl->lookahead.depth);
        const int32_t horizon_env = moe_lookahead_horizon();
        // 建图侧按 depth 插了 H 块、运行侧按 horizon 切块；两边不一致会**沿 ne0 切错**，
        // 把第 A 层的分数当成第 B 层的。所以这里以**张量实际宽度**为准夹紧：
        // ranks_total 必须是 depth 的整数倍，horizon 不能超过实际块数。
        int32_t horizon = std::max(1, std::min(horizon_env, ranks_total / depth));
        if (ranks_total % depth != 0) {
            // 预测张量宽度不是 depth 的整数倍 —— 只能按 1 块处理（防御）。
            horizon = 1;
        }
        if (horizon != horizon_env) {
            // 出现这种夹紧有**两种正常来源**，不能笼统解释成"尾部层现象"：
            //   1. qwen35moe 的建图循环 `k <= horizon && il + k < n_layer` —— 靠后的层
            //      能建出的块数依次是 H-1、H-2……（H=8 时尾部 7…1 块）；
            //   2. qwen3moe 等只建 1 块预测的路径 —— **每一层**都会退到 1 块。
            // 日志带 layer，且只打前 16 条；**限流后可见条数不等于实际次数**，
            // 所以另有 per-runtime 的累计计数 g_n_horizon_clamped，收尾打一行 summary。
            static std::atomic<int> clamp_logs{0};
            pimpl->g_n_horizon_clamped.fetch_add(1, std::memory_order_relaxed);
            const int n_logged = clamp_logs.fetch_add(1, std::memory_order_relaxed);
            if (n_logged < 16) {   // 量化前 16 条，避免每层每步刷屏
                moe_log("lookahead-horizon: layer=%d 只建出 %d 块（请求 %d，depth=%d），夹到 %d%s",
                        (int) layer, (int) (ranks_total / depth), (int) horizon_env, (int) depth,
                        (int) horizon, n_logged == 15 ? "（后续同类日志已节流）" : "");
            }
        }

        std::vector<int32_t> all_ids;
        {
            auto c = trace_context; c.source = "prediction";
            moe_handoff_span get(pimpl->handoff_trace, "PRED_GET_BEGIN", "PRED_GET_DONE", c);
            pimpl->read_i32_ids(lookahead_pred, all_ids); get.complete();
        }

        for (int32_t kb = 0; kb < horizon; ++kb) {
            const int32_t target = layer + kb + 1;
            if (target > pimpl->last_moe_layer) {
                break;
            }
            const bool extra_h1_front = pimpl->h1_front && layer == pimpl->first_moe_layer && kb == 0 && horizon > 1;
            if (extra_h1_front) ++pimpl->h1_front_eligible;
            const int32_t width = ranks_total / horizon;   // 本块的 rank 数（= depth）
            if (pimpl->handoff_trace.enabled()) {
                auto c = trace_context; c.layer = target; c.source = "prediction";
                for (int32_t row = 0; row < rows; ++row) for (int32_t rank = 0; rank < width; ++rank) {
                    pimpl->handoff_trace.emit("PRED_CANDIDATE", c, all_ids[(size_t) row * ranks_total + kb * width + rank], row, rank);
                }
            }
            std::vector<int32_t> pred;
            pred.reserve((size_t) width * (size_t) rows);
            // all_ids 是 [ranks_total, rows] 的列主序：index = row*ranks_total + j。
            if (rows > 1) {
                // 多行（投机验证）按 rank 交替，去掉重复，让有限的读深度先服务多个候选 token。
                std::vector<uint8_t> seen((size_t) pimpl->n_expert, 0);
                for (int32_t rank = 0; rank < width; ++rank) {
                    for (int32_t row = 0; row < rows; ++row) {
                        const int32_t expert = all_ids[(size_t) row * ranks_total + kb * width + rank];
                        if (expert >= 0 && expert < pimpl->n_expert && !seen[(size_t) expert]) {
                            seen[(size_t) expert] = 1;
                            pred.push_back(expert);
                        }
                    }
                }
            } else {
                for (int32_t rank = 0; rank < width; ++rank) {
                    const int32_t expert = all_ids[(size_t) kb * width + rank];
                    if (expert >= 0 && expert < pimpl->n_expert) {
                        pred.push_back(expert);
                    }
                }
            }
            if (pred.empty()) {
                continue;
            }

            // 「缓存 + 预测」的结合点（零 I/O）：把预测留给**目标层**，
            // 它下一步淘汰时可以避开这些条目。不是预读 —— 不读任何字节。
            //
            // **两份预测，语义不同，必须分开存**（评审指出过这里曾经混在一起）：
            //   - `predicted[L]`  = **真正被 arm 的那一份**（来自 L-horizon，recall ≈71.6%）
            //     → 这是"已经读进缓存、绝不能又被淘汰出去"的集合，保护集必须与它一致；
            //   - `pred_hint[L]`  = 紧邻上一层给的预测（来自 L-1，recall ≈83%）
            //     → 精度更高，但不是我们读的那一份，只作为额外的淘汰提示。
            // 只用一个槽位时，后写覆盖先写（L-1 那份），保护集就和 arm 集分家了
            // —— 被读进来的专家反而没有保护，这正是白读的来源。
            {
                std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
                if (kb + 1 == horizon) {
                    pimpl->predicted[(size_t) target] = pred;
                }
                if (kb == 0) {
                    pimpl->pred_hint[(size_t) target] = pred;
                }
                pimpl->pred_hist[(size_t) target].emplace_back(layer, pred);
            }
            // arm 前先去掉**目标层缓存里已有**的预测专家。
            // 它们本来就不需要读（worker 也会查表跳过），但会让批次看起来更大、
            // worker 跑得更久 —— 而队列满时 arm 是直接丢的。
            std::vector<int32_t> need_read;
            need_read.reserve(pred.size());
            {
                auto & tgt = pimpl->layers[(size_t) target];
                std::lock_guard<std::mutex> lk(tgt.mutex);
                for (int32_t e : pred) {
                    if (e >= 0 && e < pimpl->n_expert && pimpl->cached_slot(tgt, e, true) < 0) {
                        need_read.push_back(e);
                    }
                }
            }
            pimpl->lookahead.n_pred_cached.fetch_add(pred.size() - need_read.size(),
                                                     std::memory_order_relaxed);
            // **只给最远那一层 arm**。这一条是量出来的关键：
            //   每个目标层会被 K 个来源各预测一次（层 T-1 猜它=1 层提前、T-2 猜它=2 层…），
            //   如果每块都 arm，目标层实际读到的是 K 份预测的**并集**，密度远高于单份 →
            //   总字节上涨（实测 h4q4 比 h1q1 多读 28%），而带宽是瓶颈，于是更慢。
            //   只 arm 最远那份（k=horizon），每个目标层就只读一份预测，字节数与
            //   horizon=1 相同，但**发起时刻提前了 horizon 层** —— 这正是流水要的。
            // Optional first-layer coverage only: the original farthest job
            // keeps its read budget and still follows H1 in the existing loop.
            const bool refresh_far = pimpl->lookahead.refresh && horizon > 1 && kb + 1 == horizon && target < pimpl->last_moe_layer;
            const bool refresh_near = pimpl->lookahead.refresh && horizon > 1 && kb == 0 && target < pimpl->last_moe_layer;
            if (refresh_far || refresh_near) {
                // An empty need_read still closes/refines the near stage. The
                // manager is authoritative for coalesced/Loading/Resident IDs.
                if (pimpl->lookahead.arm_refreshed(target, pred, pimpl->graph_epoch, n_rows, refresh_near)) {
                    auto armed = pimpl->lookahead.refreshed_predictions(target, pimpl->graph_epoch);
                    std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
                    pimpl->predicted[(size_t) target] = std::move(armed);
                }
            } else if ((kb + 1 == horizon || extra_h1_front) && !need_read.empty()) {
                if (extra_h1_front) {
                    ++pimpl->h1_front_attempted;
                    pimpl->h1_front_candidate_ids += need_read.size();
                    {
                        std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
                        pimpl->predicted[(size_t) target] = pred;
                    }
                    auto c = trace_context; c.layer = target; c.source = "prefetch";
                    pimpl->handoff_trace.emit("H1_FRONT_ARM", c, -1, -1, (int32_t) need_read.size());
                }
                pimpl->lookahead.arm(target, need_read, pimpl->graph_epoch, n_rows);
            }
            // 记下 horizon 内各目标层的预测，供该层做 recall(k) 统计与扫描记录。
            if (pimpl->sweep_fp && target < (int) pimpl->sweep_pred.size()) {
                pimpl->sweep_pred[target] = pred;
            }
        }
    }

    // 命中率扫描：decode 记录整批的 E（真实用到）；prefill 只记**最后一个 token** 的 E
    // （用于判断 prefill 能否给 decode 留初始缓存）。
    if (pimpl->sweep_fp) {
        if (decode) {
            std::vector<int32_t> used;
            used.reserve(32);
            for (int32_t e = 0; e < pimpl->n_expert; ++e) {
                if (requested[e]) {
                    used.push_back(e);
                }
            }
            pimpl->sweep_write(layer, used, pimpl->sweep_pred[layer]);
        } else {
            pimpl->sweep_write_prefill(layer, experts.data(),
                                       (int) expert_ids->ne[0], (int) expert_ids->ne[1]);
        }
    }

    const int n_tokens = (int) expert_ids->ne[1];
    const int n_used = (int) expert_ids->ne[0];
    const int n_miss = (int) pending.size();
    const int n_hit = n_unique - n_miss;
    // Count every layer and batch, including multi-token draft verification.
    // Periodic ubatch diagnostics sample batches and cannot establish this rate.
    if (window_mode) {
        pimpl->window_unique += (uint64_t) n_unique;
        pimpl->window_miss += (uint64_t) (managed ? n_unique - resident_at_lookup : n_miss);
    } else {
        pimpl->bank_unique += (uint64_t) n_unique;
        pimpl->bank_miss += (uint64_t) n_miss;
    }

    if (layer == 0) {
        if (pimpl->cur.n_layers > 0) {
            pimpl->flush_ubatch(pimpl->cur.n_tokens <= 1 ? "decode" : "prefill",
                pimpl->cur.n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch);
        }
        if (n_tokens <= 1) {
            pimpl->n_decode_step++;
            pimpl->miss_per_layer_steps++;   // 每 decode 步一次（不是每层）
        } else {
            pimpl->n_prefill_ubatch++;
        }
        pimpl->cur.n_tokens = n_tokens;
        pimpl->cur.n_used = n_used;
    }

    pimpl->cur.n_layers++;
    pimpl->cur.unique_sum += n_unique;
    pimpl->cur.miss_sum += n_miss;
    pimpl->cur.n_th = std::max(pimpl->cur.n_th, timing.n_th);
    pimpl->cur.pack_layers += state.pack_ok ? 1 : 0;
    pimpl->cur.wall_ns += timing.wall_ns;
    pimpl->cur.pread_sum_ns += timing.pread_sum_ns;
    pimpl->cur.bytes += timing.bytes;
    if (window_mode && n_tokens <= 1 && layer < (int) pimpl->miss_per_layer.size()) {
        pimpl->miss_per_layer[(size_t) layer] += (uint64_t) n_miss;
    }

    const uint64_t ubatch_idx = n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch;
    const bool sample_layer = (layer == 0 || layer == 1 ||
        layer == pimpl->last_moe_layer / 2 || layer == pimpl->last_moe_layer);
    if (ubatch_idx <= 2 && sample_layer && (n_miss > 0 || layer == 0)) {
        moe_log("layer l=%d tok=%d used=%d unique=%d miss=%d hit=%d pack=%d nth=%d "
                "pread_wall=%.1fms pread_sum=%.1fms pread_max=%.1fms bytes=%.1fMiB par=%.2f",
            layer, n_tokens, n_used, n_unique, n_miss, n_hit,
            timing.pack ? 1 : 0, timing.n_th,
            llama_moe_slot_runtime::impl::ms_of(timing.wall_ns),
            llama_moe_slot_runtime::impl::ms_of(timing.pread_sum_ns),
            llama_moe_slot_runtime::impl::ms_of(timing.pread_max_ns),
            llama_moe_slot_runtime::impl::mib_of(timing.bytes),
            llama_moe_slot_runtime::impl::par_of(timing.pread_sum_ns, timing.wall_ns));
    }

    if (layer == pimpl->last_moe_layer) {
        const char * kind = n_tokens <= 1 ? "decode" : "prefill";
        const uint64_t idx = n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch;
        if (pimpl->should_log_ubatch(n_tokens, idx)) {
            pimpl->flush_ubatch(kind, idx);
        } else {
            pimpl->cur = llama_moe_slot_runtime::impl::ubatch_acc{};
        }
        if (idx == 8 || (idx > 8 && (idx % 8) == 0)) {
            pimpl->log_io(kind);
        }
    }

    // slots[] 已在分配循环里按位置填好（被延后的也有有效槽位地址）。这里只补
    // LRU 时间戳：位置越靠后越新，与原来一致。
    for (size_t i = 0; i < count; ++i) {
        if (slots[i] < 0) {
            throw std::runtime_error("MoE expert was not installed");
        }
        if (managed) {
            const bool promoted = state.manager->touch(experts[i]);
            if (promoted && pimpl->handoff_trace.enabled()) {
                const auto request = state.manager->lookup(experts[i]).request;
                pimpl->handoff_trace.emit("PF_PROBATION_PROMOTE", handoff_request(trace_context, request), experts[i], request.slot);
            }
        } else state.slot_age[(size_t) slots[i]] = ++state.age;
        // 诊断：这个专家来自预读区吗？预读区每步重填，所以它**用完就没了** ——
        // 下次再需要就得重读。理想情况应该把它升格到缓存区。
        if (window_mode && pimpl->prefetch_slots > 0 &&
            slots[i] >= state.window_start + (state.window_width > 0
                        ? state.window_width - pimpl->prefetch_slots : pimpl->decode_window)) {
            pimpl->lookahead.n_used_from_pf.fetch_add(1, std::memory_order_relaxed);
        }
    }
    {
        const int64_t t0 = now_ns();
        if (window_mode && !managed) {
            state.protected_experts = requested;
            pimpl->protected_layer = layer;
        }
        pimpl->write_i32_ids(slot_ids, slots);
        pimpl->writeids_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
    if (!window_mode && pimpl->prefill_warm_enabled && pimpl->prefill_warm_rows() > 0) {
        pimpl->warm_from_prefill(state, layer, experts, slots, n_rows);
    }
    selected_guard.success = true;
    process_lock.complete(); callback.complete();
}

/// A2 的 phase B：把 phase A 里被延后的缺失专家读进来，并发布到窗口。
///
/// 调用时机：**在「前段专家（位置 < n - matmul_split）已经算完」之后**。
/// 那时 GPU 正在算那些已驻留的专家，这里的读就与它重叠了 —— 这是整个 A2 的收益来源。
///
/// 锁：读盘不持锁（槽位在 phase A 已标 kSlotPending，别处不会动它）；只有发布
/// （写 expert_to_slot / slot_to_expert）才持层锁，和预读 worker 的写法一致。
void llama_moe_slot_runtime::process_deferred(int layer) {
    if (layer < 0 || layer >= (int) pimpl->deferred.size()) {
        return;
    }
    auto & list = pimpl->deferred[(size_t) layer];
    if (list.empty()) {
        return;
    }
    auto & st = pimpl->layers[(size_t) layer];
    if (!st.enabled) {
        list.clear();
        return;
    }

    for (const auto & d : list) {
        pimpl->read_expert_directv(st, d.expert, d.slot, true);
    }
    {
        std::lock_guard<std::mutex> lk(st.mutex);
        for (const auto & d : list) {
            st.slot_to_expert[(size_t) d.slot] = d.expert;
            st.expert_to_slot[(size_t) d.expert] = d.slot;
        }
        st.n_prefetch_published += list.size();
    }
    list.clear();
}
