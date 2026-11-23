#!/usr/bin/env python3
"""analyze.py —— 汇总 front-app 实验，给出"被干扰的是什么"的归因。

输入目录（with_llm.sh 产出）：
  rounds.tsv        每轮 config(C=app+llm / B=llm-only)、gen_tps、fg_lost、round_t0_ms
  llm_windows.log   每行 "tag start_ms end_ms rc fg_lost"（手机时钟）
  <tag>.dev.csv     runp 每秒累计：dev_busy_jiffies / dev_iowait_jiffies / u_cpu(App, jiffies)
  <tag>.app.csv     RSS / 网络 / App CPU

输出：判定表 + 归因结论。

归因逻辑（只能到"哪一类资源"层面，不能给带宽绝对值）：
  1. decode tok/s 是否显著下降（B vs C，中位数，且要求两组方向一致）
  2. 若下降：
     a) 设备 iowait 绝对增量 >= 5 ms/s 且 相对增幅 >= 50%
        -> 干扰源以 UFS / 块层 为主（前台 App 在读写）
        （阈值不能写成"翻倍"：B 组基线本身就因 llama 读专家而很高，
          翻倍门槛会让 197% 的真实上升被判成"不变"。）
     b) App CPU 高但 iowait 不变
        -> 干扰源以 CPU 调度 / LPDDR 带宽 为主（见第 3 条如何再区分）
     c) App CPU 很低
        -> 干扰很弱，tok/s 差异可能只是噪声
  3. 无法直接测内存带宽（无 root，PMU 被 SELinux 挡）。若需要区分
     "CPU 抢占" vs "LPDDR 带宽"，必须额外用 skills/profile 的 qtimer skel
     比较 DSP 计算时长：DSP 变慢 -> 带宽被挤；DSP 不变但 IO 变慢 -> UFS。
"""
import csv
import os
import statistics as st
import sys


def read_csv(p, delim=","):
    try:
        with open(p, newline="") as f:
            return list(csv.DictReader(f, delimiter=delim))
    except FileNotFoundError:
        return []


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_windows(out):
    """tag -> (start_ms, end_ms, rc, fg_lost)；start/end 与 dev.csv 的 epoch_ms 同一手机墙钟。"""
    wins = {}
    path = os.path.join(out, "llm_windows.log")
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    wins[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return wins


def median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def llama_io(out, tag):
    """从 llama-cli 的 stdout 抓 moe-ssd 的 UFS 统计 —— 独立于 iowait 的硬证据。

    行格式：[moe-ssd] total[final]: miss=5131 bytes=7696.5 MiB pread=14429ms
    返回 (MiB, pread_ms, MiB/s) 或 None。
    """
    import re
    txt = ""
    # tok/s 在 stdout，moe-ssd 的 IO 统计在 stderr —— 两个都读
    for suf in ("llm.out", "llm.err"):
        try:
            txt += open(os.path.join(out, f"{tag}.{suf}"), errors="ignore").read()
        except FileNotFoundError:
            pass
    if not txt:
        return None
    m = re.findall(r"total\[final\]:\s*miss=\d+\s+bytes=([0-9.]+)\s*MiB\s+pread=(\d+)ms", txt)
    if not m:
        return None
    mib, ms = float(m[-1][0]), int(m[-1][1])
    return (mib, ms, mib / (ms / 1000) if ms else 0.0)


def config_stats(out, cfg, rounds, wins):
    """某一组（C 或 B）的 tok/s 与资源指标中位数。

    只有同时满足以下条件的轮次才进入统计（tok/s 与资源用同一批轮次）：
      - fg_lost=0；
      - 有 llm_windows.log 窗口；
      - 窗口内 >=2 行采样，且采样跨度 >= 窗口长度的 80%。
    等 devlock、预热、轮间冷却、采样提前结束都不应该混进吞吐/资源窗口。
    """
    all_rows = [r for r in rounds if r.get("config") == cfg]
    clean_rows = [r for r in all_rows if int(num(r.get("fg_lost") or 0)) == 0]

    valid = []          # (round, dev_rows, app_rows, dur_s, llama_io)
    window_missing = 0
    no_tps = 0
    for r in clean_rows:
        tag = r["tag"]
        dev = read_csv(os.path.join(out, f"{tag}.dev.csv"))
        app = read_csv(os.path.join(out, f"{tag}.app.csv"))
        win = wins.get(tag)
        io = llama_io(out, tag)

        if not win:
            window_missing += 1
            continue
        dev_w = [x for x in dev if win[0] <= num(x.get("epoch_ms") or x.get("ts_ms")) <= win[1]]
        app_w = [x for x in app if win[0] <= num(x.get("ts")) <= win[1]]
        if len(dev_w) < 2:
            window_missing += 1
            continue
        win_s = max((win[1] - win[0]) / 1000.0, 0.001)
        t_first = num(dev_w[0].get("epoch_ms") or dev_w[0].get("ts_ms"))
        t_last = num(dev_w[-1].get("epoch_ms") or dev_w[-1].get("ts_ms"))
        span = max((t_last - t_first) / 1000.0, 0.0)
        # 采样只覆盖窗口的一小段时，速率不能代表整个 LLM 窗口 -> 拒绝该轮。
        if span < 0.8 * win_s:
            window_missing += 1
            continue
        # 资源与 tok/s 必须用同一批轮次：没有有效 tok/s 的失败轮不能污染资源中位数。
        if r.get("gen_tps") in (None, "", "NA") or num(r.get("gen_tps")) <= 0:
            no_tps += 1
            continue
        valid.append((r, dev_w, app_w, max(span, 0.001), io))

    rows = [r for (r, _d, _a, _s, _io) in valid]
    tps = [num(r["gen_tps"]) for r in rows]
    prm = [num(r["prompt_tps"]) for r in rows if r.get("prompt_tps") not in (None, "", "NA")]

    app_cpu_pct, dev_busy_pct, iowait_ms_per_s, rss_mib = [], [], [], []
    pread_ms, pread_mibs, pread_mibs_vals = [], [], []
    mem_avail, swap_used, majflt = [], [], []
    for _r, dev, app, dur, io in valid:
        # runp 的 *_jiffies 列本身是"自开始以来的累计值"
        el = [(num(x.get("epoch_ms") or x.get("ts_ms")) / 1000.0, num(x.get("u_cpu"))) for x in dev]
        busy = [num(x.get("dev_busy_jiffies")) for x in dev]
        iow = [num(x.get("dev_iowait_jiffies")) for x in dev]
        # App CPU：只在有效区间求均值（包未启动时 jiffies 不变）；1%=1 个核的 1/100
        live = cpu = 0.0
        for (t0, c0), (t1, c1) in zip(el, el[1:]):
            if t1 > t0 and c1 >= c0:
                live += t1 - t0
                cpu += c1 - c0
        app_cpu_pct.append(cpu / live if live else 0.0)
        # HZ=100：jiffies/(100*dur) = 核数；再除以 ncpu 得到全部核的占比（勿再乘 100）
        ncpu = int(num(dev[-1].get("ncpu"), 8)) or 8
        dev_busy_pct.append((busy[-1] - busy[0]) / dur / ncpu)
        mj = [num(x.get("u_majflt")) for x in dev if x.get("u_majflt") not in (None, "")]
        if mj:
            majflt.append(mj[-1])
        iowait_ms_per_s.append((iow[-1] - iow[0]) * 10 / dur)

        if io:
            pread_ms.append(io[1]); pread_mibs.append(io[0]); pread_mibs_vals.append(io[2])
        if app:
            v = [int(num(x["rssKiB"])) for x in app if num(x["rssKiB"]) > 0]
            if v:
                rss_mib.append(max(v) / 1024)
            ma = [num(x["memAvailKiB"]) / 1048576 for x in app if num(x.get("memAvailKiB")) > 0]
            if ma:
                mem_avail.append(median(ma))
            sw = [num(x["swapUsedKiB"]) / 1024 for x in app if num(x.get("swapUsedKiB")) > 0]
            if sw:
                swap_used.append(max(sw))
    return dict(
        n=len(tps), n_all=len(all_rows), n_clean=len(clean_rows), n_res=len(valid), no_tps=no_tps,
        tps=tps, tps_med=median(tps), prm_med=median(prm),
        app_cpu=median(app_cpu_pct), dev_busy=median(dev_busy_pct),
        iowait_ms_s=median(iowait_ms_per_s), rss=median(rss_mib),
        pread_ms=median(pread_ms), pread_mib=median(pread_mibs), ufs_mibs=median(pread_mibs_vals),
        mem_avail=median(mem_avail), swap=max(swap_used) if swap_used else None,
        majflt=median(majflt),
        fg_lost=sum(int(num(r.get("fg_lost"))) for r in all_rows),
        window_used=len(valid), window_missing=window_missing,
    )


def main(out):
    rounds = read_csv(os.path.join(out, "rounds.tsv"), delim="\t")
    if not rounds:
        print(f"没有找到 {out}/rounds.tsv")
        return 1

    wins = read_windows(out)
    C = config_stats(out, "C", rounds, wins)
    B = config_stats(out, "B", rounds, wins)

    L = []
    L.append(f"实验目录: {out}")
    # 预热状态：首次运行 vs 稳态，两者干扰相差一个数量级，必须写进报告
    try:
        warm = dict(l.strip().split("=", 1) for l in open(os.path.join(out, "warmup.txt")) if "=" in l)
    except FileNotFoundError:
        warm = {}
    if warm.get("warm_secs", "0") == "0":
        L.append("⚠️ 未预热（warm_secs=0）：若被测 App 是首次运行，结论会严重高估干扰。")
    else:
        dv = float(warm.get("warm_disk_delta_mib") or 0)
        L.append(f"预热: {warm.get('warm_secs')}s，期间 /data 净占用变化 {dv:+.1f} MiB"
                 f"（{warm.get('warm_disk_rate_mib_s','?')} MiB/s）"
                 + ("  ← 负值 = 净释放（缓存淘汰/清理），df 只反映净值" if dv < 0 else ""))
    L.append("")
    L.append("== 每轮 ==")
    L.append(f"{'config':<7}{'#':<4}{'gen t/s':>9}{'prompt t/s':>12}{'fg_lost':>9}")
    for r in rounds:
        L.append(f"{r['config']:<7}{r['round']:<4}{r.get('gen_tps','NA'):>9}{r.get('prompt_tps','NA'):>12}{r.get('fg_lost','0'):>9}")

    L.append("")
    L.append("== 组间中位数 ==")
    L.append(f"{'指标':<26}{'B llm-only':>14}{'C app+llm':>14}{'变化':>12}")

    def fmt(v):
        return "NA" if v is None else f"{v:.2f}"

    def row(name, b, c, unit=""):
        # 两侧各自独立；B 组 App 没启动时 App 指标本就没有值，不该把 C 组也变成 NA
        if b is None or c is None or b == 0:
            d = "-"
        else:
            d = f"{(c - b) / b * 100:+.1f}%"
        L.append(f"{name:<26}{fmt(b):>14}{fmt(c):>14}{d:>12}")

    row("decode tok/s", B["tps_med"], C["tps_med"])
    row("prefill tok/s", B["prm_med"], C["prm_med"])
    row("App CPU (%)", B["app_cpu"], C["app_cpu"])
    row("App RSS (MiB)", B["rss"], C["rss"])
    row("设备 CPU (%)", B["dev_busy"], C["dev_busy"])
    row("设备 iowait (ms/s)", B["iowait_ms_s"], C["iowait_ms_s"])
    if B["pread_ms"] or C["pread_ms"]:
        L.append("")
        L.append("-- llama 自身 UFS 统计（独立证据，来自 moe-ssd total[final]）--")
        row("专家读 pread 耗时 (ms)", B["pread_ms"], C["pread_ms"])
        row("专家读字节 (MiB)", B["pread_mib"], C["pread_mib"])
        row("有效读带宽 (MiB/s)", B["ufs_mibs"], C["ufs_mibs"])
    if B["mem_avail"] or C["mem_avail"]:
        L.append("")
        L.append("-- 内存（判断是否 direct reclaim 拖慢读路径）--")
        row("MemAvailable (GiB)", B["mem_avail"], C["mem_avail"])
        row("swap 已用 (MiB)", B["swap"], C["swap"])
        row("App 进程 major fault", B["majflt"], C["majflt"])

    L.append("")
    L.append(f"轮数: B={B['n']}/{B['n_all']} C={C['n']}/{C['n_all']}（有效 tok/s / 总轮）；"
             f"C 组前台丢失={C['fg_lost']}（已整轮剔除）")
    L.append(f"资源窗口: B={B['window_used']}/{B['window_used']+B['window_missing']} "
             f"C={C['window_used']}/{C['window_used']+C['window_missing']} "
             "（按 llm_windows.log 对齐；缺窗口的轮次不参与资源统计）")
    L.append(f"无有效 tok/s: B={B['no_tps']} C={C['no_tps']}（不进入 tok/s/资源统计）")
    if B["n"] == 0 or C["n"] == 0:
        L.append("⚠️ 有组别完全没有有效 tok/s —— 先查该轮的 *.llm.log：")
        L.append("   - 'Permission denied' 或 cd 失败 => llama 不是从 Termux uid 起的（须走 ssh $PHONE_SSH）")
        L.append("   - '0x72' / 'err 114'            => CDSP 卡死，重启手机")
        L.append("   - 有 Generation 行但这里 NA     => runs/<tag>-n<N>.out 的 tag 或 N 不匹配")

    # ---------------- 归因 ----------------
    L.append("")
    L.append("== 归因 ==")
    if B["tps_med"] is None or C["tps_med"] is None:
        L.append("数据不足：至少需要 B 和 C 各一轮带有效 tok/s 的样本。")
        print("\n".join(L))
        return 0

    if B["iowait_ms_s"] is None or C["iowait_ms_s"] is None:
        L.append("")
        L.append("⚠️ 资源窗口不足：B/C 至少一组没有落在 llm_windows.log 窗口内的采样，")
        L.append("   只报告 tok/s，不做 CPU/iowait 资源归因。检查 llm_windows.log 与 dev.csv 的 ts 列。")
        print("\n".join(L))
        return 0

    slow = (B["tps_med"] - C["tps_med"]) / B["tps_med"] * 100
    app_cpu = C["app_cpu"] or 0
    d_iow = (C["iowait_ms_s"] or 0) - (B["iowait_ms_s"] or 0)
    d_busy = (C["dev_busy"] or 0) - (B["dev_busy"] or 0)
    d_rss = (C["rss"] or 0) - (B["rss"] or 0)

    L.append(f"decode 变慢: {slow:.1f}%（B {B['tps_med']:.2f} -> C {C['tps_med']:.2f} t/s）")
    L.append(f"App CPU: {app_cpu:.1f}%   RSS: +{d_rss:.0f} MiB")
    rel_io = (d_iow / B["iowait_ms_s"] * 100) if B["iowait_ms_s"] else float("inf")
    L.append(f"设备 iowait 变化: {d_iow:+.1f} ms/s（{rel_io:+.0f}%）  设备 CPU 变化: {d_busy:+.1f}%")
    d_ma = None
    if B["mem_avail"] and C["mem_avail"]:
        d_ma = C["mem_avail"] - B["mem_avail"]
        L.append(f"MemAvailable 变化: {d_ma:+.2f} GiB"
                 + (f"（swap {B['swap']:.0f} -> {C['swap']:.0f} MiB）"
                    if B["swap"] is not None and C["swap"] is not None else ""))

    # 读数变慢：只是"I/O 路径变慢"的证据，**不能**直接归因到 UFS 带宽。
    # 原因：pread 的墙钟时间包含线程被调度走的时间；内存压力下读路径里的 direct
    # reclaim 也会拉长它。要区分需要额外证据（见下）。
    io_slow = False
    dbytes = dtime = None
    if B["pread_ms"] and C["pread_ms"] and B["pread_mib"] and C["pread_mib"]:
        dbytes = abs(C["pread_mib"] - B["pread_mib"]) / B["pread_mib"]
        dtime = (C["pread_ms"] - B["pread_ms"]) / B["pread_ms"]
        if dbytes < 0.15 and dtime > 0.15:
            io_slow = True
            L.append(f"读数变慢: 字节数基本相同（差 {dbytes*100:.1f}%），pread 耗时 "
                     f"{B['pread_ms']:.0f} -> {C['pread_ms']:.0f} ms（{dtime*100:+.0f}%），"
                     f"有效带宽 {B['ufs_mibs']:.0f} -> {C['ufs_mibs']:.0f} MiB/s。")
            L.append("  注意：这只说明 I/O 路径变慢，不能直接归因到 UFS 带宽——")
            L.append("  pread 墙钟时间包含被调度走的时间，内存压力下的 direct reclaim 也会拉长它。")

    if slow < 3:
        L.append("判定: 未观察到显著干扰（<3%），或干扰被 App 的低活动量掩盖。")
    elif io_slow:
        if d_ma is not None and d_ma < -1.0:
            L.append(f"判定: **内存压力先于 UFS** —— C 组可用内存少了 {-d_ma:.1f} GiB。")
            L.append("  内存紧张时读路径会做 direct reclaim，pread 自然变慢，与 UFS 带宽无关。")
            L.append("  看上方 'App 进程 major fault'：若 C 组明显更高，说明页被换出后反复缺页。")
        elif d_ma is not None:
            L.append(f"判定: 内存压力不明显（MemAvailable 只变 {d_ma:+.2f} GiB），"
                     "I/O 路径变慢更可能来自 CPU 抢占或 UFS。")
            L.append("  下一步: 查 Termux 的 cpuset 是否为 /background（见 skills/bench），")
            L.append("          再用 skills/profile 比较两组 preadv 的 p50/p99。")
        else:
            L.append("判定: I/O 路径明显变慢，但**归因未定**（本轮没有内存指标）。")
            L.append("  重新采一次（sp_helper 已记录 MemAvailable/swap）即可排除内存压力。")
        L.append("  排查顺序：1) 内存压力  2) CPU 抢占  3) 才是 UFS")
    elif app_cpu < 5 and d_iow < 2:
        L.append("判定: tok/s 下降但 App 几乎没活动、iowait 也没涨 -> 更像噪声或与 App 无关的漂移。"
                 "增加轮数复测。")
    elif d_iow >= 5.0 and (d_iow / (B["iowait_ms_s"] or 0.1)) >= 0.5:
        L.append("判定: **UFS / 块层 是主要干扰源**（iowait 明显上升）。")
        L.append("  下一步: 用 skills/bench 的 GGML_HEXAGON_PROFILE 日志比较 C 组与 B 组的")
        L.append("          preadv (read_end_us-begin_us) 分布，确认是 UFS 服务时间变长。")
    else:
        L.append("判定: iowait 基本不变但 CPU 被占 -> 干扰更可能在 **CPU 调度 / LPDDR 带宽**。")
        L.append("  这两者本工具无法区分（无 root 拿不到 PMU）。要区分请用 skills/profile 的")
        L.append("  qtimer skel：比较 C/B 两组 DSP 计算时长 ——")
        L.append("    DSP 也变慢 => LPDDR 带宽被 GPU/CPU 挤占；DSP 不变 => 纯 CPU 调度抢占。")

    if C["fg_lost"]:
        L.append(f"⚠️ C 组有 {C['fg_lost']} 次前台丢失，已从统计中剔除；需要补齐有效轮次。")

    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
