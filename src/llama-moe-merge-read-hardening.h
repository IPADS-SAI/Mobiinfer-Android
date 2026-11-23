#pragma once

// R4a demand 读路径硬化：不依赖 llama/ggml/后端、可在 host 上单测的纯逻辑。
//
// 这个头文件只放两类东西：
//
//   1. O_DIRECT 对齐判据 `span_needs_buffered_fallback()`。`readv_span()` 在
//      O_DIRECT 打开时走 `readv_span_direct()`；后者原先在 offset/长度不是
//      512B 倍数时回调 `readv_span()`，于是两个函数互相递归（RK-3）。判据抽成
//      纯函数后，`readv_span_direct()` 的非对齐分支只调用 `readv_span_buffered()`，
//      不会再进入带 O_DIRECT 派发的 `readv_span()`。
//
//   2. `readv_span_buffered_fd()` 的缓冲 preadv，以及 demand 合并读的降级
//      编排 `read_demand_unit()`。后者保证「merge 返回 false 或抛异常」时，
//      **非 managed** unit 的每个专家恰好被逐专家读一次（RK-2）；managed
//      unit 保持 fail-all + rethrow（ownership 语义，不在本任务改）。
//
// 头文件不 include 项目内其它头文件，避免把后端依赖带进 host 单测。

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace llama_moe {
namespace merge_read_hardening {

// Linux / Android 上 O_DIRECT 的偏移、长度、用户缓冲区都要求这个对齐。
// 这里只判 offset/total；用户缓冲区由调用方负责（直接路径用 4096 对齐 scratch）。
static const size_t k_o_direct_align = 512;

/// 纯判据：O_DIRECT 下这次 span 读是否必须走缓冲路径。
///
/// total == 0 时没有任何字节要读，返回 false（调用方应先处理空读，避免
/// `0 % 512 == 0` 被误读成“对齐”）。
inline bool span_needs_buffered_fallback(bool o_direct, size_t offset, size_t total) {
    if (!o_direct || total == 0) {
        return false;
    }
    return (offset % k_o_direct_align) != 0 || (total % k_o_direct_align) != 0;
}

/// 缓冲 preadv：把一个连续文件区间读进若干个不相交 iovec，部分读时推进 iovec。
/// `scratch` 由调用方持有（生产路径是 thread_local，热路径不分配）。
///
/// 注意：这个函数**不能**用在以 O_DIRECT 打开的 fd 上做非对齐读 —— O_DIRECT
/// 的 offset/长度/用户缓冲区都必须对齐，否则内核返回 EINVAL。生产路径在
/// `readv_span_buffered()` 里会用同一文件的普通 fd（无 O_DIRECT）来读。
inline void readv_span_buffered_fd(int fd, size_t offset, const struct iovec * iov, int niov,
                                   std::vector<struct iovec> & scratch) {
    size_t total = 0;
    for (int i = 0; i < niov; ++i) {
        total += iov[i].iov_len;
    }
    scratch.resize((size_t) niov);
    size_t done = 0;
    while (done < total) {
        struct iovec * cur = scratch.data();
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
            throw std::runtime_error("failed to read MoE experts at offset " +
                std::to_string((unsigned long long) (offset + done)));
        }
        done += (size_t) result;
    }
}

enum class demand_unit_result {
    merged,     // 一次合并读成功
    per_expert, // 没有合并，或合并失败后已逐专家重读
};

/// demand unit 的读编排（RK-2）。
///
///   read_run()        : 尝试一次合并读；true = 成功，false = 布局不支持；
///                       抛异常 = 读失败（部分读/EINVAL/EFAULT/校验失败等）。
///   read_one(j)       : 读 unit 内第 j 个专家（j 是 unit 内下标，不是全局下标）。
///   on_merged()       : 合并成功后的发布/记账（只调用一次）。
///   on_fallback(reason): 非 managed 合并失败时的日志/计数钩子。
///   on_managed_failure(reason): managed 合并抛异常时的钩子；随后异常原样重抛。
///
/// 关键不变量：
///   * `count > 1` 且 read_run 返回 false / 抛异常时，非 managed unit 的
///     `read_one` 对每个 j **恰好调用一次**（不会只读第一个，也不会重复读）；
///   * managed unit 的合并异常不降级：先调 on_managed_failure，再重抛，由调用方
///     fail-all（finish_managed(false)），绝不留 loading；
///   * count <= 1 时不调用 read_run，直接逐专家读。
template <class ReadRun, class ReadOne, class OnMerged, class OnFallback, class OnManagedFailure>
demand_unit_result read_demand_unit(int count, bool unit_managed,
                                    ReadRun read_run, ReadOne read_one,
                                    OnMerged on_merged,
                                    OnFallback on_fallback,
                                    OnManagedFailure on_managed_failure) {
    if (count > 1) {
        bool ok = false;
        std::exception_ptr error;
        try {
            ok = read_run() ? true : false;
        } catch (...) {
            error = std::current_exception();
        }
        if (ok) {
            on_merged();
            return demand_unit_result::merged;
        }
        if (error) {
            std::string reason = "unknown exception";
            try {
                std::rethrow_exception(error);
            } catch (const std::exception & e) {
                reason = e.what();
            } catch (...) {
            }
            if (unit_managed) {
                on_managed_failure(reason.c_str());
                std::rethrow_exception(error);
            }
            on_fallback(reason.c_str());
        } else if (!unit_managed) {
            on_fallback("merged layout unsupported");
        }
    }
    for (int j = 0; j < count; ++j) {
        read_one(j);
    }
    return demand_unit_result::per_expert;
}

} // namespace merge_read_hardening
} // namespace llama_moe
