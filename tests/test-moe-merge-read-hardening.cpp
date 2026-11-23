// R4a demand 读路径硬化 host 单测（不需要模型/后端/HTP）。
//
// 编译 + 运行（host）：
//   g++ -std=c++11 -O2 -Wall -Wextra -o /tmp/test-moe-merge-read-hardening
//       tests/test-moe-merge-read-hardening.cpp
//   /tmp/test-moe-merge-read-hardening
//
// 覆盖：
//   * O_DIRECT 对齐判据真值表（o_direct × offset%512 × total%512）；
//   * 临时文件非对齐 O_DIRECT 读：O_DIRECT fd 直接读会 EINVAL，缓冲回退字节正确；
//   * 注入「合并读抛异常」：非 managed 的 k 个专家各被读一次；
//   * 注入「合并读返回 false」：同上；
//   * managed 合并异常保持 fail-all + rethrow，不逐专家重读。

#include "../src/llama-moe-merge-read-hardening.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace h = llama_moe::merge_read_hardening;

static int failures = 0;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __func__, __LINE__, #c); ++failures; } } while (false)

static void test_truth_table() {
    struct test_case {
        bool o_direct;
        size_t offset;
        size_t total;
        bool want;
    };
    const test_case cases[] = {
        // O_DIRECT 关闭：永远不需要回退。
        { false, 0,    0,    false },
        { false, 0,    4096, false },
        { false, 1,    1000, false },
        { false, 511,  511,  false },
        // O_DIRECT 打开：offset 与 total 都必须 512B 对齐。
        { true,  0,    0,    false },   // 空读不递归
        { true,  0,    512,  false },
        { true,  0,    4096, false },
        { true,  512,  1024, false },
        { true,  1024, 512,  false },
        { true,  1,    512,  true  },
        { true,  511,  512,  true  },
        { true,  512,  1,    true  },
        { true,  512,  511,  true  },
        { true,  1,    1,    true  },
        { true,  513,  1025, true  },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const test_case & c = cases[i];
        const bool got = h::span_needs_buffered_fallback(c.o_direct, c.offset, c.total);
        if (got != c.want) {
            std::fprintf(stderr, "FAIL truth[%zu]: o_direct=%d offset=%zu total=%zu got=%d want=%d\n",
                         i, (int) c.o_direct, c.offset, c.total, (int) got, (int) c.want);
            ++failures;
        }
    }
    std::printf("ok   truth table (%zu cases)\n", sizeof(cases) / sizeof(cases[0]));
}

static bool write_all(int fd, const uint8_t * data, size_t n) {
    size_t done = 0;
    while (done < n) {
        const ssize_t w = write(fd, data + done, n - done);
        if (w <= 0) return false;
        done += (size_t) w;
    }
    return true;
}

static void test_nonaligned_odirect_file() {
    char path[] = "/tmp/llama-moe-merge-hardening-XXXXXX";
    const int wfd = mkstemp(path);
    CHECK(wfd >= 0);
    if (wfd < 0) return;
    const size_t n = 8192;
    std::vector<uint8_t> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = (uint8_t) ((i * 37u + 11u) & 0xffu);
    }
    CHECK(write_all(wfd, data.data(), data.size()));
    CHECK(close(wfd) == 0);

    // 生产路径会先用 O_DIRECT fd 尝试合并读；非对齐时退回缓冲实现。
    // 这里先证明同一个 O_DIRECT fd **不能**直接做非对齐读（否则回退仍是坏的）。
#ifdef O_DIRECT
    const int dfd = open(path, O_RDONLY | O_DIRECT);
    bool odirect_supported = dfd >= 0;
    if (odirect_supported) {
        const size_t offset = 17;
        const size_t total  = 4000;
        void * aligned = nullptr;
        CHECK(posix_memalign(&aligned, 4096, total) == 0);
        struct iovec iov;
        iov.iov_base = aligned;
        iov.iov_len  = total;
        errno = 0;
        const ssize_t r = preadv(dfd, &iov, 1, (off_t) offset);
        CHECK(r < 0);   // O_DIRECT 要求 offset/长度对齐 -> EINVAL
        CHECK(errno == EINVAL);
        free(aligned);
        close(dfd);
    } else {
        std::printf("skip O_DIRECT direct-read probe (open failed: %s)\n", std::strerror(errno));
    }
#else
    const bool odirect_supported = false;
    std::printf("skip O_DIRECT direct-read probe (O_DIRECT not defined)\n");
#endif
    (void) odirect_supported;

    const size_t offset = 17;
    const size_t a_len = 1500;
    const size_t b_len = 2500;
    CHECK(h::span_needs_buffered_fallback(true, offset, a_len + b_len));

    // 缓冲回退使用无 O_DIRECT 的普通 fd；两个不相交 iovec，验证部分读推进。
    const int bfd = open(path, O_RDONLY);
    CHECK(bfd >= 0);
    if (bfd < 0) {
        unlink(path);
        return;
    }
    std::vector<uint8_t> a(a_len, 0), b(b_len, 0);
    struct iovec iov[2];
    iov[0].iov_base = a.data();
    iov[0].iov_len  = a_len;
    iov[1].iov_base = b.data();
    iov[1].iov_len  = b_len;
    std::vector<struct iovec> scratch;
    try {
        h::readv_span_buffered_fd(bfd, offset, iov, 2, scratch);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL buffered non-aligned read threw: %s\n", e.what());
        ++failures;
    }
    CHECK(std::memcmp(a.data(), data.data() + offset, a_len) == 0);
    CHECK(std::memcmp(b.data(), data.data() + offset + a_len, b_len) == 0);
    close(bfd);

    // 读到文件尾之外必须抛异常（不能静默留下半截数据）。
    const int efd = open(path, O_RDONLY);
    CHECK(efd >= 0);
    if (efd >= 0) {
        uint8_t tail[1024];
        struct iovec tiov;
        tiov.iov_base = tail;
        tiov.iov_len  = sizeof(tail);
        bool threw = false;
        try {
            h::readv_span_buffered_fd(efd, n - 100, &tiov, 1, scratch);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        close(efd);
    }
    unlink(path);
    std::printf("ok   non-aligned O_DIRECT buffered fallback (bytes + EOF)\n");
}

static void test_merge_exception_fallback() {
    const int k = 5;
    std::vector<int> hits((size_t) k, 0);
    int fallbacks = 0;
    std::string reason;
    auto result = h::read_demand_unit(
        k, false,
        []() -> bool { throw std::runtime_error("injected merged read failure"); },
        [&](int j) { hits[(size_t) j]++; },
        []() { CHECK(false); },
        [&](const char * why) { ++fallbacks; reason = why; },
        [&](const char *) { CHECK(false); });
    CHECK(result == h::demand_unit_result::per_expert);
    CHECK(fallbacks == 1);
    CHECK(reason.find("injected merged read failure") != std::string::npos);
    for (int j = 0; j < k; ++j) {
        CHECK(hits[(size_t) j] == 1);
    }
    std::printf("ok   merged exception -> %d experts read once each\n", k);
}

static void test_merge_false_fallback() {
    const int k = 4;
    std::vector<int> hits((size_t) k, 0);
    int fallbacks = 0;
    auto result = h::read_demand_unit(
        k, false,
        []() -> bool { return false; },
        [&](int j) { hits[(size_t) j]++; },
        []() { CHECK(false); },
        [&](const char * why) { ++fallbacks; CHECK(std::strstr(why, "unsupported") != nullptr); },
        [](const char *) { CHECK(false); });
    CHECK(result == h::demand_unit_result::per_expert);
    CHECK(fallbacks == 1);
    for (int j = 0; j < k; ++j) {
        CHECK(hits[(size_t) j] == 1);
    }
    std::printf("ok   merged false -> %d experts read once each\n", k);
}

static void test_managed_fail_all() {
    bool managed_failure = false;
    bool read_one_called = false;
    bool threw = false;
    try {
        (void) h::read_demand_unit(
            3, true,
            []() -> bool { throw std::runtime_error("injected managed failure"); },
            [&](int) { read_one_called = true; },
            []() { CHECK(false); },
            [](const char *) { CHECK(false); },
            [&](const char * why) {
                managed_failure = true;
                CHECK(std::strstr(why, "injected managed failure") != nullptr);
            });
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(managed_failure);
    CHECK(!read_one_called);
    std::printf("ok   managed exception -> fail-all + rethrow, no per-expert read\n");
}

static void test_managed_false_reads_once() {
    // 布局不支持合并时，managed 仍走现有「逐专家读 + 逐个 finish」语义。
    const int k = 3;
    std::vector<int> hits((size_t) k, 0);
    auto result = h::read_demand_unit(
        k, true,
        []() -> bool { return false; },
        [&](int j) { hits[(size_t) j]++; },
        []() { CHECK(false); },
        [](const char *) { CHECK(false); },   // managed 的 false 不算“降级”
        [](const char *) { CHECK(false); });
    CHECK(result == h::demand_unit_result::per_expert);
    for (int j = 0; j < k; ++j) {
        CHECK(hits[(size_t) j] == 1);
    }
    std::printf("ok   managed layout-false -> per-expert once (existing semantics)\n");
}

static void test_single_and_merged_success() {
    bool run_called = false;
    int one_calls = 0;
    auto r1 = h::read_demand_unit(
        1, false,
        [&]() -> bool { run_called = true; return true; },
        [&](int) { ++one_calls; },
        []() { CHECK(false); },
        [](const char *) { CHECK(false); },
        [](const char *) { CHECK(false); });
    CHECK(r1 == h::demand_unit_result::per_expert);
    CHECK(!run_called);       // count=1 不尝试合并
    CHECK(one_calls == 1);

    int merged_calls = 0;
    one_calls = 0;
    auto r2 = h::read_demand_unit(
        3, false,
        []() -> bool { return true; },
        [&](int) { ++one_calls; },
        [&]() { ++merged_calls; },
        [](const char *) { CHECK(false); },
        [](const char *) { CHECK(false); });
    CHECK(r2 == h::demand_unit_result::merged);
    CHECK(merged_calls == 1);
    CHECK(one_calls == 0);    // 合并成功不再逐专家读
    std::printf("ok   single expert / merged success paths\n");
}

int main() {
    test_truth_table();
    test_nonaligned_odirect_file();
    test_merge_exception_fallback();
    test_merge_false_fallback();
    test_managed_fail_all();
    test_managed_false_reads_once();
    test_single_and_merged_success();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all merge-read-hardening tests passed\n");
    return 0;
}
