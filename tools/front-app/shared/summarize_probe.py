#!/usr/bin/env python3
"""summarize_probe.py <前缀，如 out/probe> <app名> —— 单 App probe 的一行摘要"""
import csv, sys

prefix, name = sys.argv[1], sys.argv[2]
def rows(p):
    try: return list(csv.DictReader(open(p)))
    except FileNotFoundError: return []

d, a = rows(f"{prefix}.dev.csv"), rows(f"{prefix}.app.csv")
if not d:
    print(f"{name}: 无 collect 数据"); sys.exit(1)
el = [(int(float(r["elapsed_s"])), float(r["u_cpu"] or 0)) for r in d]
busy = [float(r["dev_busy_jiffies"]) for r in d]
io = [float(r["dev_iowait_jiffies"]) for r in d]
dur = max(el[-1][0] - el[0][0], 1)
live = cpu = 0.0
for (t0, c0), (t1, c1) in zip(el, el[1:]):
    if t1 > t0 and c1 >= c0:
        live += t1 - t0; cpu += c1 - c0
rss = [int(float(r["rssKiB"])) for r in a if float(r["rssKiB"]) > 0] if a else []
peak_cpu = max((c1 - c0) for (_, c0), (_, c1) in zip(el, el[1:]) if c1 >= c0) if len(el) > 1 else 0
ma = [int(float(r.get("memAvailKiB") or 0)) for r in a] if a and "memAvailKiB" in (a[0] if a else {}) else []
ncpu = int(float(d[-1].get("ncpu") or 8)) or 8
ma_txt = f" | MemAvailable {min(ma)/1048576:.1f}~{max(ma)/1048576:.1f} GiB" if ma else ""
print(f"{name} probe: {dur:.0f}s | App CPU 均值 {cpu/live if live else 0:.0f}%（峰值 {peak_cpu:.0f}%/s；100% = 1 核）"
      f" | RSS 峰 {max(rss)//1024 if rss else 0} MiB | 全设备 {(busy[-1]-busy[0])/dur/ncpu:.0f}% of {ncpu}核"
      f" | iowait {io[-1]-io[0]:.0f} j = {(io[-1]-io[0])*10:.0f} ms" + ma_txt)
