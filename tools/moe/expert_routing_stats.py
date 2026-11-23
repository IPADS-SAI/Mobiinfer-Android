#!/usr/bin/env python3
"""Summarize per-layer/per-expert routing from ``llama-moe-profile`` traces.

Trace rows (one JSON object per line) look like::

    {"sample":"wiki:0001","phase":"decode","layer":17,
     "experts":[[203,222,74,1,9,60,33,88], ...],
     "weights":[[0.31,0.22,...], ...]}          # weights optional

``experts`` is one row per token/eval row; ``weights`` (when present) are the
router probabilities for those top-k ids, so ``sum_weight`` is the routed mass.

Outputs (in ``--out``):

* ``routing-expert.tsv`` -- layer, expert, per-tag phase counts and weight sums
* ``routing-layer.tsv``  -- per-layer skew (top-4 share, bottom-4 share,
  effective experts, top-k coverage), decode/prefill separately
* ``routing-summary.json``
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

N_LAYERS = 40
N_EXPERTS = 256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # (tag, phase) -> layer -> expert -> [count, sum_weight]
    stats: dict[tuple[str, str], list[list[list[float]]]] = {}
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * N_LAYERS)
    layers_seen: dict[tuple[str, str], set[int]] = defaultdict(set)
    samples: dict[str, set[str]] = defaultdict(set)
    has_weights = False

    for path in args.traces:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample = str(row["sample"])
                tag = sample.split(":", 1)[0]
                phase = row["phase"]
                layer = int(row["layer"])
                if phase not in ("prefill", "decode"):
                    raise ValueError(f"bad phase {phase!r}")
                if not 0 <= layer < N_LAYERS:
                    raise ValueError(f"bad layer {layer}")
                key = (tag, phase)
                if key not in stats:
                    stats[key] = [[[0.0, 0.0] for _ in range(N_EXPERTS)] for _ in range(N_LAYERS)]
                experts_rows = row["experts"]
                weights_rows = row.get("weights")
                if weights_rows is not None:
                    has_weights = True
                    if len(weights_rows) != len(experts_rows):
                        raise ValueError("weights/experts row count mismatch")
                for r, experts in enumerate(experts_rows):
                    if len(experts) != len(set(experts)):
                        raise ValueError(f"duplicate experts in {path} L{layer}")
                    totals[key][layer] += 1
                    ws = weights_rows[r] if weights_rows is not None else None
                    if ws is not None and len(ws) != len(experts):
                        raise ValueError("weights/experts length mismatch")
                    for j, expert in enumerate(experts):
                        cell = stats[key][layer][int(expert)]
                        cell[0] += 1.0
                        if ws is not None:
                            cell[1] += float(ws[j])
                samples[tag].add(sample)
                layers_seen[key].add(layer)

    if not stats:
        raise SystemExit("no routing rows")
    for key, seen in layers_seen.items():
        if seen != set(range(N_LAYERS)):
            raise ValueError(f"incomplete layer coverage for {key}: {sorted(seen)}")

    args.out.mkdir(parents=True, exist_ok=True)
    tags = sorted({k[0] for k in stats})
    phases = ["decode", "prefill"]
    columns = ["layer", "expert"]
    for tag in tags:
        for phase in phases:
            columns += [f"{tag}_{phase}_count", f"{tag}_{phase}_sumw"]

    with (args.out / "routing-expert.tsv").open("w", encoding="utf-8") as out:
        out.write("\t".join(columns) + "\n")
        for layer in range(N_LAYERS):
            for expert in range(N_EXPERTS):
                row = [str(layer), str(expert)]
                for tag in tags:
                    for phase in phases:
                        table = stats.get((tag, phase))
                        c, w = table[layer][expert] if table is not None else (0.0, 0.0)
                        row += [f"{int(c)}", f"{w:.6f}" if has_weights else "nan"]
                out.write("\t".join(row) + "\n")

    summary: dict = {
        "tags": tags,
        "has_weights": has_weights,
        "samples": {tag: len(v) for tag, v in samples.items()},
        "layers": N_LAYERS,
        "experts": N_EXPERTS,
        "layer_stats": {},
    }
    with (args.out / "routing-layer.tsv").open("w", encoding="utf-8") as out:
        out.write("tag\tphase\tlayer\trows\tselections\ttop4_share\tbottom4_share\teffective_experts\tn90\tn95\tn99\n")
        for tag in tags:
            for phase in phases:
                key = (tag, phase)
                if key not in stats:
                    continue
                per_layer = {}
                for layer in range(N_LAYERS):
                    counts = [stats[key][layer][e][0] for e in range(N_EXPERTS)]
                    selections = max(sum(counts), 1.0)
                    ordered = sorted(counts, reverse=True)
                    probs = [c / selections for c in counts if c > 0]
                    entropy = -sum(p * math.log2(p) for p in probs)
                    eff = 2 ** entropy
                    cum = 0
                    nk: dict[int, int | None] = {90: None, 95: None, 99: None}
                    for i, c in enumerate(ordered, 1):
                        cum += c
                        for pct in nk:
                            if nk[pct] is None and cum >= pct / 100.0 * selections:
                                nk[pct] = i
                    stat = {
                        "rows": totals[key][layer],
                        "selections": int(selections),
                        "top4_share": sum(ordered[:4]) / selections,
                        "bottom4_share": sum(ordered[-4:]) / selections,
                        "effective_experts": eff,
                        "n90": nk[90],
                        "n95": nk[95],
                        "n99": nk[99],
                    }
                    per_layer[str(layer)] = stat
                    out.write(
                        f"{tag}\t{phase}\t{layer}\t{totals[key][layer]}\t{stat['selections']}\t"
                        f"{stat['top4_share']:.6f}\t{stat['bottom4_share']:.6f}\t"
                        f"{eff:.3f}\t{nk[90]}\t{nk[95]}\t{nk[99]}\n"
                    )
                summary["layer_stats"][f"{tag}_{phase}"] = per_layer

    (args.out / "routing-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tags", "has_weights", "samples")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
