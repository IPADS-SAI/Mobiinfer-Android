// iotrace: 单线程、逐次时间戳化的 O_DIRECT 随机读探针 —— 用来验证
//   "我们的读延迟是否与前台 App（游戏）的渲染帧相位对齐"。
//
//   iotrace <文件路径> <次数> <每次字节MiB> <间隔us> <输出csv路径>
//
// 单线程**逐次**记录 submit/done 时间戳，产出可与系统帧/负载周期对齐的事件流。
//
// 关键设计（都是刻意的，别顺手改）：
//   1. 偏移用**固定种子**的 rand_r 生成：不同运行的地址序列完全一致。
//      相位分析要求"同样的读序列在不同运行/不同负载下可比"，随机种子会让
//      两个 run 读到不同的盘面区域，延迟差异混入布局差异。
//   2. t_submit 在 pread 调用前一刻取（CLOCK_MONOTONIC，ns）。系统帧/负载时间戳
//      若也走 CLOCK_MONOTONIC，本 CSV 可直接与其做绝对时间对齐（无需对时）。
//   3. 节流是"追加式"：t_next = t_submit + interval + lat，等价于读完成后再等
//      interval。interval=0 即背靠背连续读（此时周期 = 延迟本身，符合
//      "磁盘随机读延迟"的测量意图）。
//   4. 只统计成功读；短读/失败计数并继续（不让一次 EIO 毁掉整条时间轴）。
//
// 输出 CSV（以 '#' 开头的是注释，非注释行全部是数据，可直接喂给外部相位分析）：
//   # iotrace path=... iters=... bytes=... interval_us=... seed=...
//   #seq,t_submit_ns,lat_ns,offset,bytes
//   0,1234567890123,1420000,8796093022208,1572864
//   ...
// 结束时往 stderr 打一行汇总（次数 / 平均 / 中位 / p95 / p99 / 总耗时）。
//
// 构建：见 shared/build.sh（NDK aarch64-linux-android，动态 PIE，与其它工具一致）。
//   产物：build/iotrace（build/ 在 .gitignore 里）
//
// 未验证：手机被占用（本轮未调用 adb/ssh，没在设备上跑过）。本机 NDK 交叉编译通过。
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>

// 固定种子：换掉它 = 换掉地址序列 = 与历史 run 不可比。要对比新序列请另开文件，
// 不要改这里。
#define FIXED_SEED 20261107u
#define ALIGN 4096

static long long now_ns(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

// 睡到绝对时刻 t；被信号打断就继续睡（不提前返回，否则节流会漂）
static void sleep_until_ns(long long t){
    long long d;
    while((d = t - now_ns()) > 0){
        struct timespec ts = {(time_t)(d / 1000000000LL), (long)(d % 1000000000LL)};
        if(nanosleep(&ts, &ts) != 0 && errno != EINTR) break;
    }
}

static int cmp_d(const void *a, const void *b){
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
// 最近秩法分位数：数据少时也不会像线性插值那样给出不存在的值
static double pct(const double *s, int n, double p){
    if(n <= 0) return 0.0;
    int i = (int)(p * (double)n);
    if(i >= n) i = n - 1;
    return s[i];
}

int main(int c, char **v){
    if(c < 6){
        fprintf(stderr,
            "usage: %s <path> <iters> <MiB-per-read> <interval_us> <out.csv>\n"
            "  iters        次数（>0）\n"
            "  MiB-per-read 每次读多少 MiB（会向下对齐到 4096）\n"
            "  interval_us  每次读完成后的追加等待（us，0=背靠背连续读）\n"
            "  out.csv      逐次时间戳输出（seq,t_submit_ns,lat_ns,offset,bytes）\n", v[0]);
        return 2;
    }
    const char *path = v[1];
    long n_iter = atol(v[2]);
    double mb = atof(v[3]);
    long iv_us = atol(v[4]);
    const char *outp = v[5];
    if(n_iter <= 0){ fprintf(stderr, "iters 必须 >0\n"); return 2; }
    if(iv_us < 0){ fprintf(stderr, "interval_us 不能为负\n"); return 2; }

    size_t bs = (size_t)(mb * 1024.0 * 1024.0) & ~(size_t)(ALIGN - 1);
    if(bs == 0){ fprintf(stderr, "每次字节数太小（%.3f MiB），至少要 %d B\n", mb, ALIGN); return 2; }

    void *buf = NULL;
    int rc = posix_memalign(&buf, ALIGN, bs);
    if(rc != 0){ fprintf(stderr, "posix_memalign(%zu): %s\n", bs, strerror(rc)); return 1; }
    memset(buf, 0, bs);

    int fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if(fd < 0){ fprintf(stderr, "open(%s): %s\n", path, strerror(errno)); free(buf); return 1; }

    struct stat st;
    if(fstat(fd, &st) != 0){ fprintf(stderr, "fstat: %s\n", strerror(errno)); close(fd); free(buf); return 1; }
    off_t fsize = st.st_size;
    if((off_t)bs > fsize){
        fprintf(stderr, "文件只有 %lld B，小于单次读 %zu B\n", (long long)fsize, bs);
        close(fd); free(buf); return 1;
    }
    // 可选起点数量（都按 4096 对齐）。rand_r 只给 31 位，超过 2^31 个起点也只会用前
    // 2^31 个；顺便避免 (unsigned)nblk 回绕成 0 造成除零。
    long long nblk = (long long)((fsize - (off_t)bs) / ALIGN) + 1;
    if(nblk > 2147483647LL) nblk = 2147483647LL;

    double *lats = malloc(sizeof(double) * (size_t)n_iter);
    if(!lats){ fprintf(stderr, "malloc lats 失败\n"); close(fd); free(buf); return 1; }

    FILE *out = fopen(outp, "w");
    if(!out){ fprintf(stderr, "open %s: %s\n", outp, strerror(errno)); close(fd); free(buf); free(lats); return 1; }
    fprintf(out, "# iotrace path=%s iters=%ld bytes=%zu interval_us=%ld seed=%u fsize=%lld clock=CLOCK_MONOTONIC\n",
            path, n_iter, bs, iv_us, (unsigned)FIXED_SEED, (long long)fsize);
    fprintf(out, "#seq,t_submit_ns,lat_ns,offset,bytes\n");

    unsigned seed = FIXED_SEED;
    long long interval_ns = (long long)iv_us * 1000LL;
    long long t_first = 0, t_last = 0;
    long ok = 0, bad = 0;
    for(long i = 0; i < n_iter; i++){
        // 固定种子 -> 地址序列可复现；% nblk 只取决于文件大小，因此换盘器不影响
        unsigned r = (unsigned)rand_r(&seed);
        off_t off = (off_t)((long long)(r % (unsigned)nblk) * ALIGN);

        long long t0 = now_ns();
        ssize_t got = pread(fd, buf, bs, off);
        long long t1 = now_ns();

        if(got != (ssize_t)bs){
            bad++;
            fprintf(stderr, "第 %ld 次读失败 @%lld: got=%zd want=%zu errno=%s\n",
                    i, (long long)off, got, bs, got < 0 ? strerror(errno) : "short read");
            continue;   // 时间轴继续，坏样本不进 CSV（相位分析不应被半截读污染）
        }
        long long lat = t1 - t0;
        fprintf(out, "%ld,%lld,%lld,%lld,%zu\n", i, t0, lat, (long long)off, bs);
        lats[ok++] = lat;
        if(!t_first) t_first = t0;
        t_last = t1;
        if((ok & 63) == 0) fflush(out);   // 中途被杀也留得下大部分数据

        sleep_until_ns(t0 + interval_ns + lat);
    }
    fclose(out);

    if(ok == 0){
        fprintf(stderr, "iotrace: 0 次成功读（%ld 次失败）\n", bad);
        close(fd); free(buf); free(lats); return 1;
    }
    double *s = lats; qsort(s, (size_t)ok, sizeof(double), cmp_d);
    double sum = 0.0;
    for(long i = 0; i < ok; i++) sum += s[i];
    double dt = (double)(t_last - t_first) / 1e9;
    double avg_ns = sum / (double)ok;
    // 实际提交周期：nanosleep 会过冲（几十 us），所以真实周期总比 interval+lat 大。
    // 相位分析靠的是时间戳本身，但报告里必须给出真实周期，否则没法判断
    // "采样周期是不是已经接近帧周期（16.7/8.3 ms）-> 相位混叠"。
    double period_ms = ok > 1 ? (double)(t_last - t_first) / 1e9 / (double)(ok - 1) * 1000.0 : 0.0;
    fprintf(stderr,
        "iotrace: %ld/%ld 次成功（%ld 次失败），%zu B/次 = %.2f MiB，interval=%ld us\n"
        "延迟 ms: avg=%.3f p50=%.3f p95=%.3f p99=%.3f min=%.3f max=%.3f\n"
        "总耗时 %.3f s（%.1f MiB/s，%.2f 次/s），实际提交周期 avg=%.3f ms -> %s\n",
        ok, n_iter, bad, bs, bs / 1048576.0, iv_us,
        avg_ns / 1e6, pct(s, (int)ok, 0.50) / 1e6, pct(s, (int)ok, 0.95) / 1e6,
        pct(s, (int)ok, 0.99) / 1e6, s[0] / 1e6, s[ok - 1] / 1e6,
        dt, (double)ok * bs / 1048576.0 / (dt > 0 ? dt : 1e-9),
        dt > 0 ? (double)ok / dt : 0.0, period_ms, outp);
    // 相位实验的坑：采样周期一旦接近帧周期（16.7/8.3 ms）的整数倍/整数分之一，
    // 相位就会锁定在固定格点上（混叠），看起来"有相位依赖"其实只是采样混叠。
    // 磁盘读本身要 1.4~11 ms，所以采样不可能真做到亚毫秒 —— 只要别和帧周期成简单整数比。
    // 外部相位分析需再检查：采样周期与帧周期相差 < 20% 时应判"相位不可信"。
    if(period_ms > 4.0)
        fprintf(stderr,
            "⚠️ 提交周期 %.2f ms（稀疏采样）：相位分析仍可用（每点相位独立）。"
            "但若它恰好等于帧周期 16.7/8.3 ms 的整数倍或整数分之一，相位会混叠；"
            "interval_us=0（背靠背，周期=延迟本身）是单线程下能拿到的最快采样。\n",
            period_ms);

    close(fd); free(buf); free(s);
    return bad ? 1 : 0;   // 有坏读返回非零，但数据仍然可用
}
