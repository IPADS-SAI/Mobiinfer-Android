#!/usr/bin/env python3
"""Generate phone-ready LLAMA_MOE_LAYER_WINDOWS plans from an offline route dump.

Pipeline (all offline; no model forward pass):

    *-routes.jsonl  --split train/heldout-->  window_sweep  --win.csv-->
        solve_k.py (DP allocator)  -->  {caps, layer_windows, heldout miss}

`window_sweep` replays the engine's batch-LRU (8 experts/step, same-step experts
never evict each other) for capacities 8..256 step 8, on both splits.  The DP in
solve_k then minimises train miss for a fixed per-layer cache budget while
holding a hard per-layer cap (default 96 slots, bench/sweep.sh's safety limit).

Held-out data is only used for evaluation; it never enters the allocation.

Usage:
  tools/moe/gen_layer_windows.py \
      --routes exp/pack-order-20261122/routes.jsonl \
      --prompts exp/pack-order-20261122/prompts.jsonl \
      --nlayers 40 --nexperts 256 --split pack-order \
      --window-sweep ./window_sweep --budgets 64,72,80,88 --max-c 96 \
      --baseline "t0018_D=0:96,1:96,..." \
      --out-dir t-0029
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

UNIT = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_order_heldout(pid):
    """Exact split used by exps/pack-order-20261122 (32 CN + 32 EN, 1/4 heldout)."""
    n = int(pid.lstrip("pP"))
    if n < 32:
        return n % 4 == 0
    return (n - 32) % 4 == 1


def split_routes(routes, heldout_ids, train_path, held_path):
    n_train = n_held = 0
    with open(routes) as src, open(train_path, "w") as tr, open(held_path, "w") as he:
        for line in src:
            rec = json.loads(line)
            if rec["sample"] in heldout_ids:
                he.write(line)
                n_held += 1
            else:
                tr.write(line)
                n_train += 1
    return n_train, n_held


def run(cmd, log_path):
    with open(log_path, "w") as log:
        log.write("+ " + " ".join(str(c) for c in cmd) + "\n")
        log.flush()
        proc = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
    if proc.returncode:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
                         f"see {log_path}")
    return proc.stdout


def load_curve(path):
    tr, he, steps = {}, {}, {}
    with open(path) as fp:
        reader = csv.DictReader(fp)
        caps = [int(c[len("train"):]) for c in reader.fieldnames if c.startswith("train")]
        for row in reader:
            L = int(row["layer"])
            tr[L] = [float(row[f"train{c}"]) for c in caps]
            he[L] = [float(row[f"held{c}"]) for c in caps]
            steps[L] = int(row["steps"])
    return caps, tr, he, steps


def eval_caps(caps, he, steps):
    """Held-out miss/step (summed over layers) and 8-way hit rate for a cap map."""
    layers = sorted(caps)
    total_steps = sum(steps[L] for L in layers)
    miss = 0.0
    for L in layers:
        c = caps[L]
        if c % UNIT or not (UNIT <= c <= 256):
            raise SystemExit(f"bad cap L{L}={c} (must be 8..256 step {UNIT})")
        miss += he[L][c // UNIT - 1] / 1000.0 * (UNIT * steps[L])
    per_step = miss / (total_steps / len(layers))
    per_layer_step = miss / total_steps
    hit = 100.0 * (1.0 - miss / (UNIT * total_steps))
    return dict(heldout_miss_total=round(miss, 2),
                heldout_miss_per_step=round(per_step, 4),
                heldout_miss_per_layer_step=round(per_layer_step, 4),
                heldout_hit_pct=round(hit, 3),
                steps_per_layer=round(total_steps / len(layers), 1))


def validate_plan(cmap, layers, budget, min_c, max_c):
    if set(cmap) != set(layers):
        raise SystemExit(f"plan layers {sorted(cmap)} != {layers}")
    if sum(cmap.values()) != budget * len(layers):
        raise SystemExit(f"plan total {sum(cmap.values())} != {budget * len(layers)}")
    for L, c in cmap.items():
        if c % UNIT or not (min_c <= c <= max_c):
            raise SystemExit(f"bad cap L{L}={c} (need {min_c}..{max_c} step {UNIT})")
    spec = ",".join(f"{L}:{cmap[L]}" for L in layers)
    if parse_spec(spec, len(layers)) != cmap:
        raise SystemExit("layer_windows string does not round-trip")
    return spec


def parse_spec(spec, nlayers):
    caps = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        layer, width = item.split(":")
        caps[int(layer)] = int(width)
    missing = [L for L in range(nlayers) if L not in caps]
    if missing:
        raise SystemExit(f"baseline spec misses layers {missing[:5]}")
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--nlayers", type=int, required=True)
    ap.add_argument("--nexperts", type=int, required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--replay-mode", choices=["per-sample", "concat"], default="per-sample",
                    help="reset the offline LRU at sample boundaries (default) or concatenate")
    ap.add_argument("--split", choices=["pack-order", "ids"], default="pack-order",
                    help="pack-order = the deterministic 32+32 split of exps/pack-order-20261122")
    ap.add_argument("--heldout-ids", help="file with one prompt id per line (--split ids)")
    ap.add_argument("--window-sweep", help="compiled tools/moe/window_sweep binary")
    ap.add_argument("--win-csv", help="reuse an existing window_sweep CSV instead of replaying")
    ap.add_argument("--solve-k", default=str(Path(__file__).with_name("solve_k.py")))
    ap.add_argument("--budgets", default="64,72,80,88", help="cache slots per layer")
    ap.add_argument("--min-c", type=int, default=8,
                    help="per-layer floor; 64 avoids the WINDOW_MAX_ROWS=8 clamp")
    ap.add_argument("--max-c", type=int, default=96, help="hard per-layer cache cap")
    ap.add_argument("--baseline", action="append", default=[],
                    help='extra baseline: "name=0:96,1:96,..." (repeatable)')
    ap.add_argument("--model", default="MiniCPM-V 4.7 35B-A3B")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.topk != 8:
        ap.error("window_sweep is fixed at top-8; --topk must be 8")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)

    prompts = [json.loads(line) for line in open(args.prompts) if line.strip()]
    if args.split == "pack-order":
        heldout = {p["id"] for p in prompts if pack_order_heldout(p["id"])}
    else:
        if not args.heldout_ids:
            ap.error("--split ids needs --heldout-ids")
        heldout = {line.strip() for line in open(args.heldout_ids) if line.strip()}
    if not heldout or len(heldout) >= len(prompts):
        ap.error(f"bad heldout split: {len(heldout)}/{len(prompts)}")

    train_jsonl = work / "train-routes.jsonl"
    held_jsonl = work / "heldout-routes.jsonl"
    n_train, n_held = split_routes(args.routes, heldout, train_jsonl, held_jsonl)

    commands = []
    if args.win_csv:
        win_csv = Path(args.win_csv)
        sweep_log = work / "window_sweep.log"
        sweep_log.write_text(f"reused {win_csv}\n")
    else:
        if not args.window_sweep:
            ap.error("need --window-sweep or --win-csv")
        win_csv = work / "win.csv"
        cmd = [args.window_sweep, train_jsonl, held_jsonl, args.nlayers,
               args.nexperts, "d"]
        if args.replay_mode == "per-sample":
            cmd.append("per-sample")
        commands.append(" ".join(map(str, cmd)) + f" > {win_csv}")
        csv_text = run(cmd, work / "window_sweep.log")
        win_csv.write_text(csv_text)

    caps_grid, tr, he, steps = load_curve(win_csv)
    if len(tr) != args.nlayers:
        sys.exit(f"curve has {len(tr)} layers, expected {args.nlayers}")
    layers = sorted(tr)

    plans = []
    for budget in [int(x) for x in args.budgets.split(",")]:
        scenario_path = work / f"scenario-c{budget}.json"
        cmd = [sys.executable, args.solve_k, "--win", win_csv, "--budget", str(budget),
               "--min-c", str(args.min_c), "--max-c", str(args.max_c),
               "--prefetch", "0", "--json", scenario_path]
        commands.append(" ".join(map(str, cmd)))
        run(cmd, work / f"solve_k-c{budget}.log")
        scenario = json.load(open(scenario_path))["scenarios"][0]
        cmap = {int(L): int(c) for L, c in scenario["capacities"].items()}
        spec = validate_plan(cmap, layers, budget, args.min_c, args.max_c)
        metrics = eval_caps(cmap, he, steps)
        uniform = {L: budget for L in layers}
        if budget % UNIT == 0 and sum(uniform.values()) == sum(cmap.values()):
            u = eval_caps(uniform, he, steps)
            metrics["uniform_same_budget_miss_per_step"] = u["heldout_miss_per_step"]
            metrics["vs_uniform_same_budget_pct"] = round(
                100.0 * (metrics["heldout_miss_per_step"] - u["heldout_miss_per_step"])
                / u["heldout_miss_per_step"], 3)
        plan = dict(
            id=f"dp-c{budget}" + (f"-min{args.min_c}" if args.min_c != UNIT else ""),
            cache_budget_per_layer=budget,
            cache_floor=args.min_c,
            prefetch_k=0,
            cache_slots_total=sum(cmap.values()),
            cache_avg=round(sum(cmap.values()) / len(cmap), 2),
            cache_min=min(cmap.values()),
            cache_max=max(cmap.values()),
            peak_window_slots=scenario["cache_max"],
            peak_window_slots_with_k8=scenario["cache_max"] + 8,
            total_window_slots_with_k8=sum(cmap.values()) + 8 * len(cmap),
            layer_windows=spec,
            caps={str(L): cmap[L] for L in layers},
            heldout=metrics,
        )
        plans.append(plan)

    baselines = []
    uniform80 = {L: 80 for L in layers}
    baselines.append(dict(id="uniform80", layer_windows=",".join(f"{L}:80" for L in layers),
                          caps={str(L): 80 for L in layers},
                          heldout=eval_caps(uniform80, he, steps)))
    for item in args.baseline:
        name, spec = item.split("=", 1)
        cmap = parse_spec(spec, args.nlayers)
        baselines.append(dict(id=name, layer_windows=spec, caps={str(L): cmap[L] for L in layers},
                              heldout=eval_caps(cmap, he, steps)))

    report = dict(
        schema=1,
        model=args.model,
        generated_by="tools/moe/gen_layer_windows.py",
        budget_semantics=("cache slots per layer with prefetch_k=0; the engine adds "
                          "LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH slots on top. "
                          "peak_window_slots_with_k8 assumes the bench default DEPTH=8."),
        trace=dict(
            routes=str(Path(args.routes).resolve()),
            routes_sha256=sha256(args.routes),
            prompts=str(Path(args.prompts).resolve()),
            prompts_sha256=sha256(args.prompts),
            n_layers=args.nlayers, n_experts=args.nexperts, topk=args.topk,
            split=args.split,
            train_samples=len(prompts) - len(heldout), heldout_samples=len(heldout),
            train_records=n_train, heldout_records=n_held,
            heldout_ids=sorted(heldout),
            steps_per_layer=round(sum(steps.values()) / len(steps), 1),
        ),
        replay=dict(window_sweep_csv=str(win_csv.resolve()),
                    caps=caps_grid, phase="decode",
                    lru="batch-8 all-associative, same-step experts never evict",
                    sample_reset=(args.replay_mode == "per-sample")),
        plans=plans,
        baselines=baselines,
        commands=commands,
    )
    plans_path = out / "layer-windows-plans.json"
    plans_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(f"{'plan':>10} {'cache/层':>8} {'min':>4} {'max':>4} {'总槽':>6} "
          f"{'heldout miss/步':>15} {'vs uniform':>11}")
    for p in plans:
        u = p["heldout"].get("vs_uniform_same_budget_pct")
        print(f"{p['id']:>10} {p['cache_avg']:8.2f} {p['cache_min']:4d} {p['cache_max']:4d} "
              f"{p['cache_slots_total']:6d} {p['heldout']['heldout_miss_per_step']:15.4f} "
              f"{('n/a' if u is None else f'{u:+.2f}%'):>11}")
    for b in baselines:
        print(f"{b['id']:>10} {'':>8} {'':>4} {'':>4} {sum(b['caps'].values()):6d} "
              f"{b['heldout']['heldout_miss_per_step']:15.4f}")
    print(f"wrote {plans_path} ({plans_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
