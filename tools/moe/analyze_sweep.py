#!/usr/bin/env python3
"""Analyze a LLAMA_MOE_SWEEP trace: per-layer prefetch precision vs demand.

Input (written by src/llama-moe-slot.cpp sweep_write when LLAMA_MOE_SWEEP=<path>
is set, one line per decode step per MoE layer, in layer order):

    L17 E:203,222,74,1,9,60,33,88 P:203,222,74,1,9,60,88,120

`E:` is the expert set actually used (top-k), `P:` the lookahead prediction.

What this computes, per layer:
  demand/step   misses per decode step of an LRU window of capacity `--capacity`
                (exactly the engine's batch-LRU: 8 experts requested together,
                same-step experts never evict each other)
  useful_r      step fraction where prediction rank r is correct AND not resident
                (i.e. it really saved a demand read)
  waste_r       step fraction where prediction rank r is wrong (a wasted read;
                the engine filters correct+resident ones via n_pred_cached)
  precision_r   step fraction where prediction rank r is in E

Then the per-layer prefetch-bank size K is chosen from the tradeoff
useful(K) vs waste(K) (see --lam).

Also parses LLAMA_MOE_PREFETCH_TIMING=1 lines if present:
    prefetch-timing: layer=17 overlap_samples=.. overlap_mean_us=.. late_jobs=.. read_jobs=.. read_experts=.. batch_wall_per_expert_us=..

usage (run next to llama-moe-profile, or on any host):
  tools/moe/analyze_sweep.py SWEEP.txt [--capacity 72] [--layers 40] [--topk 8] [--lam 0.5]
                  [--csv out.csv] [--json out.json]
"""
import argparse
import json
import re
from collections import defaultdict

SWEEP_RE = re.compile(r"^L(\d+)\s+E:([0-9,]*)\s+P:([0-9,]*)\s*$")
# Two formats exist in the field: the A100/host build prints late_jobs/batch_wall_per_expert_us,
# the phone build (release-v81-sync) prints late_demands/read_us_per_expert. Accept both.
TIMING_RES = (
    re.compile(r"prefetch-timing: layer=(\d+) overlap_samples=(\d+) overlap_mean_us=([\d.]+) "
               r"late_jobs=(\d+) read_jobs=(\d+) read_experts=(\d+) batch_wall_per_expert_us=([\d.]+)"),
    re.compile(r"prefetch-timing: layer=(\d+) overlap_samples=(\d+) overlap_mean_us=([\d.]+) "
               r"late_demands=(\d+) read_jobs=(\d+) read_experts=(\d+) read_us_per_expert=([\d.]+)"),
)


def parse(path):
    """Return per-layer list of (used list, predicted list), plus timing rows."""
    per_layer = defaultdict(list)
    timings = {}
    with open(path, errors="replace") as fp:
        for line in fp:
            line = line.strip()
            m = SWEEP_RE.match(line)
            if m:
                layer = int(m.group(1))
                used = [int(x) for x in m.group(2).split(",") if x]
                pred = [int(x) for x in m.group(3).split(",") if x]
                per_layer[layer].append((used, pred))
                continue
            for t in (re_ for re_ in (r.search(line) for r in TIMING_RES) if re_):
                timings[int(t.group(1))] = dict(
                    overlap_samples=int(t.group(2)), overlap_mean_us=float(t.group(3)),
                    late=int(t.group(4)), read_jobs=int(t.group(5)),
                    read_experts=int(t.group(6)), us_per_expert=float(t.group(7)))
    return per_layer, timings


def lru_misses(steps, capacity):
    """Per-step miss counts of the engine's batch LRU; returns list aligned with steps."""
    order, resident = [], set()
    misses = []
    for used, _pred in steps:
        miss = [e for e in used if e not in resident]
        misses.append(len(miss))
        need = len(miss)
        if capacity is not None:
            while len(resident) + need > capacity:
                victim = next((e for e in order if e not in used), None)
                if victim is None:
                    break
                order.remove(victim)
                resident.discard(victim)
            for e in miss:
                resident.add(e)
                order.append(e)
        # touch in rank order
        for e in used:
            if e in order:
                order.remove(e)
            order.append(e)
    return misses


def analyze_layer(steps, capacity, topk, lam):
    n = len(steps)
    if not n:
        return None
    misses = lru_misses(steps, capacity)

    # residency must be tracked alongside the prediction accounting
    order, resident = [], set()
    correct = [0] * topk
    useful = [0] * topk
    waste = [0] * topk
    exposure = 0  # steps where the prediction list was shorter than topk
    for i, (used, pred) in enumerate(steps):
        used_set = set(used)
        if len(pred) < topk:
            exposure += 1
        for r in range(min(topk, len(pred))):
            e = pred[r]
            if e in used_set:
                correct[r] += 1
                if e not in resident:
                    useful[r] += 1
            else:
                waste[r] += 1
        # advance cache exactly as lru_misses did
        miss = [e for e in used if e not in resident]
        while len(resident) + len(miss) > capacity:
            victim = next((e for e in order if e not in used_set), None)
            if victim is None:
                break
            order.remove(victim)
            resident.discard(victim)
        for e in miss:
            resident.add(e)
            order.append(e)
        for e in used:
            if e in order:
                order.remove(e)
            order.append(e)

    # K tradeoff: maximize hidden reads minus lam * wasted reads
    best_k, best_score = 0, 0.0
    curve = []
    u_acc = w_acc = 0.0
    for k in range(1, topk + 1):
        u_acc += useful[k - 1] / n
        w_acc += waste[k - 1] / n
        score = u_acc - lam * w_acc
        curve.append(dict(k=k, useful=round(u_acc, 4), waste=round(w_acc, 4),
                          precision=round(correct[k - 1] / n, 4), score=round(score, 4)))
        if score > best_score:
            best_score, best_k = score, k
    return dict(
        steps=n,
        demand_per_step=round(sum(misses) / n, 3),
        miss_total=sum(misses),
        precision=[round(correct[r] / n, 4) for r in range(topk)],
        useful=[round(useful[r] / n, 4) for r in range(topk)],
        waste=[round(waste[r] / n, 4) for r in range(topk)],
        pred_short_steps=exposure,
        curve=curve,
        best_k=best_k,
        best_score=round(best_score, 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--capacity", type=int, default=72, help="LRU window per layer (experts)")
    ap.add_argument("--layers", type=int, default=0, help="expected layer count (0 = infer)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="weight of a wasted read relative to a hidden demand read")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    args = ap.parse_args()

    per_layer, timings = parse(args.sweep)
    if not per_layer:
        raise SystemExit("no 'L<n> E:.. P:..' lines found")
    layers = sorted(per_layer)
    if args.layers and len(layers) != args.layers:
        print(f"warning: found {len(layers)} layers, expected {args.layers}")

    report = {"capacity": args.capacity, "lam": args.lam,
              "sweep": args.sweep, "layers": {}, "timing": {}}
    for L in layers:
        rep = analyze_layer(per_layer[L], args.capacity, args.topk, args.lam)
        if rep is None:
            continue
        if L in timings:
            rep["timing"] = timings[L]
        report["layers"][str(L)] = rep

    tot_steps = sum(r["steps"] for r in report["layers"].values())
    tot_demand = sum(r["miss_total"] for r in report["layers"].values())
    print(f"layers={len(layers)} steps/layer≈{tot_steps/len(layers):.0f} capacity={args.capacity}")
    print(f"总 demand（E: 的 LRU 缺失）= {tot_demand} 次 / {tot_steps/len(layers):.0f} 步")
    print()
    print(f"{'层':>4} {'steps':>6} {'demand/步':>10} {'prec r0':>8} {'prec r3':>8} {'prec r7':>8} "
          f"{'useful K=2':>10} {'waste K=2':>10} {'useful K=8':>10} {'waste K=8':>10} {'K*':>4}")
    for L in layers:
        r = report["layers"].get(str(L))
        if not r:
            continue
        p = r["precision"]
        c = {x["k"]: x for x in r["curve"]}
        print(f"L{L:<3d} {r['steps']:6d} {r['demand_per_step']:10.2f} "
              f"{p[0]*100:7.1f}% {p[min(3,len(p)-1)]*100:7.1f}% {p[max(0,len(p)-1)]*100:7.1f}% "
              f"{c[min(2,max(c))]['useful']:10.2f} {c[min(2,max(c))]['waste']:10.2f} "
              f"{c[max(c)]['useful']:10.2f} {c[max(c)]['waste']:10.2f} {r['best_k']:4d}")
    ks = [r["best_k"] for r in report["layers"].values()]
    if ks:
        print(f"\nK* 分布: " + ", ".join(f"K={k}:{ks.count(k)} 层" for k in sorted(set(ks))))
    if timings:
        print("\n== LLAMA_MOE_PREFETCH_TIMING 实测（按层）==")
        tot_s = sum(t["overlap_samples"] for t in timings.values())
        tot_l = sum(t["late"] for t in timings.values())
        if tot_s:
            print(f"迟到率合计 {100.0 * tot_l / tot_s:.1f}%（{tot_l}/{tot_s}）：低=预取赶上了需求，"
                  f"高=预取白忙（此时预取只值 bytes，不值延迟）")
        print(f"{'层':>4} {'read_experts':>13} {'late':>7} {'迟到%':>7} {'overlap_samples':>16} "
              f"{'overlap_us':>11} {'us/expert':>10}")
        for L in sorted(timings):
            t = timings[L]
            rate = 100.0 * t["late"] / t["overlap_samples"] if t["overlap_samples"] else 0.0
            print(f"L{L:<3d} {t['read_experts']:13d} {t['late']:7d} {rate:6.1f}% {t['overlap_samples']:16d} "
                  f"{t['overlap_mean_us']:11.1f} {t['us_per_expert']:10.1f}")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(report, fp, indent=1)
        print(f"\nwrote {args.json}")
    if args.csv:
        with open(args.csv, "w") as fp:
            fp.write("layer,steps,demand_per_step,precision_r0,precision_r3,precision_r7,"
                     "useful_k2,waste_k2,useful_k4,waste_k4,useful_k8,waste_k8,best_k\n")
            for L in layers:
                r = report["layers"].get(str(L))
                if not r:
                    continue
                c = {x["k"]: x for x in r["curve"]}
                g = lambda k: c.get(k, c[max(c)])  # noqa: E731
                p = r["precision"]
                fp.write(f"{L},{r['steps']},{r['demand_per_step']},"
                         f"{p[0]:.4f},{p[min(3,len(p)-1)]:.4f},{p[max(0,len(p)-1)]:.4f},"
                         f"{g(2)['useful']:.4f},{g(2)['waste']:.4f},"
                         f"{g(4)['useful']:.4f},{g(4)['waste']:.4f},"
                         f"{g(8)['useful']:.4f},{g(8)['waste']:.4f},{r['best_k']}\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
