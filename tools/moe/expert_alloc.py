#!/usr/bin/env python3
"""Rank (layer, family, expert) quantisation upgrades and emit allocations.

Inputs
------
* ``routing-expert.tsv`` from :mod:`expert_routing_stats` (decode counts are the
  allocation signal; prefill is reported separately).
* ``expert-quant-stats.tsv`` from :mod:`expert_quant_stats` (``mse1``, ``mse2``,
  ``diff``, ``energy`` per expert slice).

Score
-----
``score = p_decode(layer, expert) * (mse1 - mse2) / energy``  (when BF16 stats
exist) or ``p_decode * diff / energy`` otherwise.  ``p`` is the decode routing
frequency of the expert inside its layer (top-8 over all decode rows), so the
score is an expected output-perturbation reduction per upgraded expert-family.
Every upgrade costs the same number of bytes (``n_params * 0.1875``), which makes
the greedy selection a knapsack with unit weights.

The tool writes allocation TSVs (``layer family expert`` per line) for the
carrier builder, plus the equivalent ``blk.L.family.expert=Q2_K`` rules for a
future per-expert manifest, and per-layer top-4/bottom-4 tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
SHORT = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}


def read_quant(path: Path) -> dict[tuple[int, str, int], dict[str, float]]:
    out = {}
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            key = (int(row["layer"]), row["family"], int(row["expert"]))
            out[key] = {k: float(v) for k, v in row.items() if k not in ("layer", "family", "expert")}
    return out


def read_routing(path: Path, tag: str, phase: str) -> dict[tuple[int, int], float]:
    """Return (layer, expert) -> probability within the layer."""
    counts: dict[tuple[int, int], float] = {}
    totals: dict[int, float] = defaultdict(float)
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        ccol = f"{tag}_{phase}_count"
        if ccol not in (reader.fieldnames or []):
            raise SystemExit(f"{path} has no column {ccol}")
        for row in reader:
            layer = int(row["layer"])
            expert = int(row["expert"])
            c = float(row[ccol])
            counts[(layer, expert)] = c
            totals[layer] += c
    return {k: (v / totals[k[0]] if totals[k[0]] else 0.0) for k, v in counts.items()}


def fmt_rule(key: tuple[int, str, int]) -> str:
    layer, family, expert = key
    return f"blk.{layer}.{family}.{expert}=Q2_K"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routing", required=True, type=Path)
    ap.add_argument("--quant", required=True, type=Path)
    ap.add_argument("--tag", default="wiki")
    ap.add_argument("--phase", default="decode")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--budgets", default="0.3,0.6", help="decimal GB, comma separated")
    ap.add_argument("--family-mult", default="gate=1,up=1,down=1")
    ap.add_argument("--layer-controls", default="38-39:0.3,36-39:0.6",
                    help="name:budget or layers:budget, e.g. 38-39:0.3")
    ap.add_argument("--layer-prior", type=Path, default=None,
                    help="TSV layer<TAB>weight; multiplies the per-expert score (layer sensitivity prior)")
    ap.add_argument("--arm-prefix", default="pe", help="prefix for budget allocations")
    args = ap.parse_args()

    mult = {}
    for item in args.family_mult.split(","):
        name, value = item.split("=", 1)
        mult[name] = float(value)
    fam_mult = {
        "ffn_gate_exps": mult.get("gate", 1.0),
        "ffn_up_exps": mult.get("up", 1.0),
        "ffn_down_exps": mult.get("down", 1.0),
    }

    quant = read_quant(args.quant)
    prob = read_routing(args.routing, args.tag, args.phase)
    layer_prior = {}
    if args.layer_prior is not None:
        for line in args.layer_prior.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            layer, weight = line.split("\t")
            layer_prior[int(layer)] = float(weight)
        if len(layer_prior) != 40:
            raise SystemExit(f"layer prior has {len(layer_prior)} rows, expected 40")

    rows = []
    for key, q in quant.items():
        layer, family, expert = key
        p = prob.get((layer, expert), 0.0)
        energy = q.get("energy") or 0.0
        if "mse1" in q and "mse2" in q:
            delta = max(q["mse1"] - q["mse2"], 0.0)
            basis = "mse1-mse2"
        else:
            delta = q.get("diff", 0.0)
            basis = "diff"
        rel = delta / energy if energy > 0 else delta
        score = p * rel * fam_mult[family] * layer_prior.get(layer, 1.0)
        rows.append({
            "layer": layer, "family": family, "expert": expert,
            "prob_decode": p, "rel_delta": rel, "score": score,
            "n_params": int(q["n_params"]),
            "mse1": q.get("mse1", float("nan")),
            "mse2": q.get("mse2", float("nan")),
            "diff": q.get("diff", float("nan")),
            "energy": energy, "basis": basis,
        })
    if not rows:
        raise SystemExit("no rows")
    if len({(r["layer"], r["family"], r["expert"]) for r in rows}) != len(rows):
        raise SystemExit("duplicate quant-stats rows")

    # Per-family cost is identical here, but keep it data-driven.
    cost = {}
    for r in rows:
        cost[r["family"]] = r["n_params"] * (84 / 256 - 18 / 128)
    if len(set(cost.values())) != 1:
        print(f"warning: family upgrade costs differ: {cost}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rules").mkdir(exist_ok=True)

    rows.sort(key=lambda r: (-r["score"], r["layer"], FAMILIES.index(r["family"]), r["expert"]))
    with (args.out / "expert-ranking.tsv").open("w", encoding="utf-8") as out:
        cols = ["rank", "layer", "family", "expert", "prob_decode", "rel_delta", "score",
                "n_params", "cost_bytes", "mse1", "mse2", "diff", "energy"]
        out.write("\t".join(cols) + "\n")
        for i, r in enumerate(rows, 1):
            out.write("\t".join([
                str(i), str(r["layer"]), r["family"], str(r["expert"]),
                f"{r['prob_decode']:.8f}", f"{r['rel_delta']:.8e}", f"{r['score']:.8e}",
                str(r["n_params"]), str(int(cost[r["family"]])),
                f"{r['mse1']:.6e}", f"{r['mse2']:.6e}", f"{r['diff']:.6e}", f"{r['energy']:.6e}",
            ]) + "\n")

    # Per-layer top-4 / bottom-4 per family (the spec's requested sample grid).
    by_lf: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_lf[(r["layer"], r["family"])].append(r)
    with (args.out / "per-layer-top4-bottom4.tsv").open("w", encoding="utf-8") as out:
        out.write("layer\tfamily\tsample\trank\texpert\tprob_decode\trel_delta\tscore\n")
        for layer in sorted({r["layer"] for r in rows}):
            for family in FAMILIES:
                items = sorted(by_lf[(layer, family)], key=lambda r: -r["score"])
                for label, subset in (("top", items[:4]), ("bottom", items[-4:][::-1])):
                    for j, r in enumerate(subset, 1):
                        out.write(f"{layer}\t{family}\t{label}\t{j}\t{r['expert']}\t"
                                  f"{r['prob_decode']:.8f}\t{r['rel_delta']:.8e}\t{r['score']:.8e}\n")

    # Greedy per-budget selections.
    def select(budget_gb: float) -> tuple[list[dict], float]:
        budget = budget_gb * 1e9
        chosen = []
        used = 0.0
        for r in rows:
            c = cost[r["family"]]
            if used + c > budget:
                break
            chosen.append(r)
            used += c
        return chosen, used

    manifests = {}
    for b in [float(x) for x in args.budgets.split(",") if x]:
        chosen, used = select(b)
        name = f"{args.arm_prefix}{int(round(b * 1000)):04d}"
        with (args.out / f"alloc-{name}.tsv").open("w", encoding="utf-8") as out:
            for r in chosen:
                out.write(f"{r['layer']}\t{r['family']}\t{r['expert']}\n")
        with (args.out / "rules" / f"expert-{name}.txt").open("w", encoding="utf-8") as out:
            out.write(f"# {name}: {len(chosen)} expert-family upgrades, {used:.6e} bytes "
                      f"({used/1e9:.4f} GB), Q2_K on top of Q1_0; rest Q1_0\n")
            for r in chosen:
                out.write(fmt_rule((r["layer"], r["family"], r["expert"])) + "\n")
        manifests[name] = {
            "budget_gb": b, "used_bytes": used, "used_gb": used / 1e9,
            "upgrades": len(chosen), "layers": sorted({r["layer"] for r in chosen}),
            "families": {SHORT[f]: sum(1 for r in chosen if r["family"] == f) for f in FAMILIES},
            "min_score": chosen[-1]["score"] if chosen else None,
        }

    # Layer controls at the same byte budget (the existing layer-sensitive rule).
    for item in args.layer_controls.split(","):
        if not item:
            continue
        spec, budget_s = item.split(":", 1)
        a, bb = (int(x) for x in spec.split("-", 1))
        budget = float(budget_s) * 1e9
        chosen = []
        used = 0.0
        # Whole layers, most sensitive layer first (38-39 -> 39 first then 38).
        layers = list(range(bb, a - 1, -1))
        for layer in layers:
            layer_rows = [r for r in rows if r["layer"] == layer]
            layer_cost = sum(cost[r["family"]] for r in layer_rows)
            if used + layer_cost > budget:
                break
            chosen.extend(layer_rows)
            used += layer_cost
        name = f"layer{a}-{bb}"
        with (args.out / f"alloc-{name}.tsv").open("w", encoding="utf-8") as out:
            for r in chosen:
                out.write(f"{r['layer']}\t{r['family']}\t{r['expert']}\n")
        layers_txt = "|".join(str(x) for x in range(a, bb + 1))
        with (args.out / "rules" / f"{name}.txt").open("w", encoding="utf-8") as out:
            out.write(f"blk\\.({layers_txt})\\.ffn_(gate|up|down)_exps\\.weight=Q2_K\n")
            out.write("ffn_(gate|up|down)_exps\\.weight=Q1_0\n")
        manifests[name] = {
            "budget_gb": float(budget_s), "used_bytes": used, "used_gb": used / 1e9,
            "upgrades": len(chosen), "layers": sorted({r["layer"] for r in chosen}),
            "families": {SHORT[f]: sum(1 for r in chosen if r["family"] == f) for f in FAMILIES},
        }

    # Coverage of the expert-level selection against held-out routing is done
    # separately; here we record concentration to explain the byte budget.
    top = rows[:1000]
    summary = {
        "rows": len(rows),
        "basis": rows[0]["basis"],
        "tag": args.tag,
        "phase": args.phase,
        "family_mult": fam_mult,
        "layer_prior": str(args.layer_prior) if args.layer_prior else None,
        "cost_per_upgrade_bytes": next(iter(cost.values())),
        "arms": manifests,
        "top1000_layer_histogram": {str(l): sum(1 for r in top if r["layer"] == l)
                                    for l in sorted({r["layer"] for r in top})},
        "top1000_family": {SHORT[f]: sum(1 for r in top if r["family"] == f) for f in FAMILIES},
    }
    (args.out / "alloc-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
