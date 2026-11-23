#!/usr/bin/env python3
"""Turn measured prefetch quality into a slot plan: per-layer cache c_i and prefetch K_i.

Inputs
  --win     window_sweep CSV (per-layer LRU load curve at 8..256 slots, train+held)
  --sweep   analyze_sweep.py JSON (per-layer prediction precision, demand, useful/waste).
            Optional: without it only K=0 (pure cache) scenarios are computed, using
            the held-out LRU curve in --win as the demand estimate.
  --budget  total slots per layer (cache + prefetch), e.g. 80
  --max-c   hard per-layer cap for cache + prefetch (default 256; phone-safe 96)
  --min-c   per-layer floor (default 8; 64 keeps WINDOW_MAX_ROWS=8 from clamping)
  --prefetch candidate uniform K values to evaluate, e.g. 0,2,4,8

Accounting model (all in "expert reads per decode step, summed over layers"):
  demand_i      reads the layer would do without prefetch (= LRU misses)
  hidden_i(K)   reads taken off the critical path      (= min(demand, useful(K)))
  extra_i(K)    reads added by wrong predictions       (= waste(K))
  total reads   demand - hidden + extra

Prefetch can *reduce* total reads (a published prefetch also fills the window, so the
demand read disappears) until the wasted reads take over; past that point it only buys
latency. So the plan is a Pareto choice, not a single optimum - see the K sweep.

The per-layer allocations are emitted both as a JSON map and as a ready-to-paste
`LLAMA_MOE_LAYER_WINDOWS` string (`layer:cache,...`).  Note that the engine adds
`LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH` prefetch slots on top of each cache width;
`--budget` here counts cache + prefetch, so a phone plan that keeps K=8 must use
`--budget 88` to end up with 80 cache slots.

usage:
  # cache-only plan from an offline LRU curve (no sweep trace needed)
  tools/moe/solve_k.py --win win_minicpm.csv --budget 80 --max-c 96 --prefetch 0

  # full prefetch tradeoff
  tools/moe/solve_k.py --win win_qwen35.csv --sweep sweep_qwen35.json --budget 80 \
      --max-c 96 --prefetch 0,2,4,8
"""
import argparse
import csv
import json

CAPS = [8 * (i + 1) for i in range(32)]
UNIT = 8


def load_window(path):
    tr, he = {}, {}
    for r in csv.DictReader(open(path)):
        L = int(r["layer"])
        tr[L] = [float(r[f"train{c}"]) for c in CAPS]
        he[L] = [float(r[f"held{c}"]) for c in CAPS]
    return tr, he


def dp_alloc(cost, budget_units, max_units=len(CAPS), min_units=1):
    """min sum cost[i][u_i] s.t. sum u_i = budget_units, min_units <= u_i <= max_units.

    Units are 8 slots.  `max_units` is the per-layer hard cap; without it the DP
    would happily put 200+ slots on a hot layer even when the caller asked for an
    average of 80 (phone safety limit is 96).
    """
    max_units = max(1, min(int(max_units), len(CAPS)))
    min_units = max(1, min(int(min_units), max_units))
    if budget_units < len(cost) * min_units:
        return None
    L = len(cost)
    INF = float("inf")
    dp = [[INF] * (budget_units + 1) for _ in range(L + 1)]
    ch = [[0] * (budget_units + 1) for _ in range(L + 1)]
    dp[0][0] = 0.0
    for i in range(L):
        row = dp[i]
        for b in range(budget_units + 1):
            base = row[b]
            if base == INF:
                continue
            for u in range(min_units, max_units + 1):
                nb = b + u
                if nb > budget_units:
                    break
                v = base + cost[i][u - 1]
                if v < dp[i + 1][nb] - 1e-12:
                    dp[i + 1][nb] = v
                    ch[i + 1][nb] = u
    if dp[L][budget_units] == INF:
        return None
    b = budget_units
    alloc = [0] * L
    for i in range(L, 0, -1):
        u = ch[i][b]
        alloc[i - 1] = u
        b -= u
    return alloc


def window_spec(layers, alloc):
    """`0:96,1:88,...` - exactly the LLAMA_MOE_LAYER_WINDOWS format."""
    return ",".join(f"{L}:{UNIT * u}" for L, u in zip(layers, alloc))


def heldout_demand(he, layers, alloc):
    """Expert reads per decode step over all layers, from the held-out LRU curve.

    window_sweep stores loads per 1000 expert references and each decode step
    references `topk` (8) experts, so miss/step = load / 1000 * 8.
    """
    return sum(he[L][u - 1] for L, u in zip(layers, alloc)) * (UNIT / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", required=True)
    ap.add_argument("--sweep", help="analyze_sweep.py JSON; omit for K=0-only cache plans")
    ap.add_argument("--budget", type=int, default=80, help="slots per layer: cache + prefetch")
    ap.add_argument("--max-c", type=int, default=256,
                    help="hard per-layer cap on cache + prefetch (96 = phone sweep safety limit)")
    ap.add_argument("--min-c", type=int, default=8,
                    help="per-layer floor; 64 keeps LLAMA_MOE_WINDOW_MAX_ROWS=8 from clamping")
    ap.add_argument("--prefetch", default="0,2,4,8")
    ap.add_argument("--json", help="write the per-K scenarios here")
    args = ap.parse_args()

    if args.max_c < UNIT or args.max_c > max(CAPS):
        ap.error(f"--max-c must be in [{UNIT}, {max(CAPS)}]")
    if args.max_c % UNIT:
        ap.error(f"--max-c must be a multiple of {UNIT}")
    if args.min_c < UNIT or args.min_c > args.max_c or args.min_c % UNIT:
        ap.error(f"--min-c must be a multiple of {UNIT} in [{UNIT}, {args.max_c}]")
    min_units = args.min_c // UNIT

    tr, he = load_window(args.win)
    layers = sorted(tr)
    sweep = json.load(open(args.sweep)) if args.sweep else None
    if sweep is not None and len(sweep["layers"]) != len(layers):
        print(f"warning: sweep has {len(sweep['layers'])} layers, window curve has {len(layers)}")

    print(f"budget={args.budget}/层 (cache+K)  min_c={args.min_c}  max_c={args.max_c}  "
          f"sweep={'yes' if sweep else 'no (K=0 only)'}")
    print(f"{'K':>3} {'缓存均值':>8} {'单层最大':>8} {'cache miss%':>12} {'藏住':>7} "
          f"{'白读':>7} {'总读/步':>9} {'vs K=0':>8}")
    base = None
    out = []
    for K in [int(x) for x in args.prefetch.split(",")]:
        if K < 0:
            ap.error(f"prefetch K={K} must be non-negative")
        if K > 0 and sweep is None:
            ap.error("--sweep is required for K>0 (no offline prefetch precision data)")
        cache_per = args.budget - K
        if cache_per < args.min_c:
            print(f"{K:3d}  预算不足")
            continue
        # --max-c is a per-layer hard cap; --budget is the average pool.
        # Capping at `cache_per` would silently force a uniform plan.
        cache_units = (args.max_c - K) // UNIT
        if cache_units < 1:
            print(f"{K:3d}  预算不足")
            continue
        alloc = dp_alloc([tr[L] for L in layers], (cache_per * len(layers)) // UNIT,
                         max_units=cache_units, min_units=min_units)
        if alloc is None:
            print(f"{K:3d}  预算不足")
            continue
        capacities = [UNIT * u for u in alloc]
        miss = sum(he[L][u - 1] for L, u in zip(layers, alloc)) / len(layers) / 10.0

        if sweep is None:
            demand = heldout_demand(he, layers, alloc)
            hidden = extra = 0.0
        else:
            demand = hidden = extra = 0.0
            for L in layers:
                r = sweep["layers"].get(str(L))
                if not r:
                    continue
                demand += r["demand_per_step"]
                if K == 0:                  # no prefetch at all: pure cache baseline
                    continue
                curve = {c["k"]: c for c in r["curve"]}
                g = curve.get(K, curve[max(curve)])
                hidden += min(r["demand_per_step"], g["useful"])
                extra += g["waste"]

        reads = demand - hidden + extra
        if K == 0:
            base = reads
        spec = window_spec(layers, alloc)
        scenario = dict(
            K=K,
            cache_per_layer=cache_per,
            min_c=args.min_c,
            max_c=args.max_c,
            cache_avg=round(sum(capacities) / len(capacities), 2),
            cache_max=max(capacities),
            slots_max=max(capacities) + K,
            total_cache_slots=sum(capacities),
            miss=round(miss, 2),
            demand=round(demand, 2),
            hidden=round(hidden, 2),
            extra=round(extra, 2),
            reads=reads,
            capacities={str(L): c for L, c in zip(layers, capacities)},
            layer_windows=spec,
        )
        out.append(scenario)
        pct = "n/a" if base is None else f"{100 * (reads - base) / base:+7.1f}%"
        print(f"{K:3d} {scenario['cache_avg']:8.1f} {scenario['cache_max']:8d} "
              f"{miss:11.2f}% {hidden:7.2f} {extra:7.2f} {reads:9.2f} {pct:>8}")

    if not out:
        ap.error(f"no feasible prefetch candidates in --prefetch={args.prefetch} "
                 f"for budget={args.budget}")
    print("\n读法：'总读/步' 是物理专家读取次数；'藏住' 只把读取移出关键路径。")
    for s in out:
        print(f"K={s['K']} LLAMA_MOE_LAYER_WINDOWS=\"{s['layer_windows']}\"")
    if args.json:
        json.dump(dict(budget=args.budget, min_c=args.min_c, max_c=args.max_c, scenarios=out),
                  open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
