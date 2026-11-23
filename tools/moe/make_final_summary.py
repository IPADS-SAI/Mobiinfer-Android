#!/usr/bin/env python3
"""Assemble the final T-0041 evidence JSON from the A100 logs.

Run from ``~/workspace/csm/moe-exp/t-0041`` after all PPL runs finish.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_expert_arms import parse_log  # type: ignore

ROOT = Path(__file__).resolve().parents[2] / "t-0041"
OUT = ROOT / "stats"

DIRECT = {
    "all_q1": ("logs/ppl-all_q1.log", 6619289760),
    "l38-39": ("logs/ppl-l38-39.log", 6921279648),
    "l36-39": ("logs/ppl-l36-39.log", 7223269536),
    "best-layer-2-6": ("logs/ppl-best-layer-2-6.log", 6921279648),
    "best-layer-6-7": ("logs/ppl-best-layer-6-7.log", 6921279648),
    "best-layer-6-33": ("logs/ppl-best-layer-6-33.log", 6921279648),
    "best-layer-2567": ("logs/ppl-best-layer-2567.log", 7223269536),
    "all_q2k": ("logs/ppl-all_q2k.log", 12659087520),
}
CARRIER = {
    "c-allq1": ("logs/ppl-c-allq1.log", 36314962080),
    "c-layer38-39": ("logs/ppl-c-layer38-39.log", 36314962080),
    "c-layer36-39": ("logs/ppl-c-layer36-39.log", 36314962080),
    "c-pe0302": ("logs/ppl-c-pe0302.log", 36314962080),
    "c-pe0604": ("logs/ppl-c-pe0604.log", 36314962080),
    "c-pef0302": ("logs/ppl-c-pef0302.log", 36314962080),
    "c-pe20302": ("logs/ppl-c-pe20302.log", 36314962080),
    "c-pe20604": ("logs/ppl-c-pe20604.log", 36314962080),
    "c-pe20302p": ("logs/ppl-c-pe20302p.log", 36314962080),
}


def ppl_of(path: Path) -> tuple[float, list[float], int]:
    p = parse_log(path)
    return p["ppl"], p["inc_ll"], p["chunks"]


def main() -> int:
    out: dict = {"model": {
        "name": "MiniCPM-V-4.7-35B-A3B", "arch": "qwen35moe", "layers": 40,
        "experts": 256, "top_k": 8, "n_embd": 2048, "n_ff_exp": 512,
        "expert_params_per_family": 2048 * 512,
        "q1_0_bpw": 18 / 128 * 8, "q2_K_bpw": 84 / 256 * 8,
        "delta_bytes_per_expert_family": 196608,
    }, "ppl": {}, "comparisons": {}}

    for kind, table, base_name in (("direct", DIRECT, "all_q1"), ("carrier", CARRIER, "c-allq1")):
        base_ppl, base_inc, chunks = ppl_of(ROOT / table[base_name][0])
        out["ppl"][kind] = {}
        for name, (rel, nbytes) in table.items():
            ppl, inc, n = ppl_of(ROOT / rel)
            if n != chunks:
                raise SystemExit(f"{name}: {n} chunks != {chunks}")
            out["ppl"][kind][name] = {
                "log": rel, "bytes": nbytes, "ppl": ppl,
                "delta_vs_base": ppl - base_ppl,
                "delta_pct_vs_base": 100.0 * (ppl / base_ppl - 1.0),
                "chunks": n,
            }

    # Paired bootstrap comparisons already written by analyze_expert_arms.py.
    comp_specs = {
        "pe0302_vs_layer38-39": "stats/ppl-pe-vs-layer03/ppl-analysis.json",
        "pe0604_vs_layer36-39": "stats/ppl-pe-vs-layer06/ppl-analysis.json",
        "pe20302_vs_best_layer_2-6": "stats/ppl-pe2-vs-bestlayer03/ppl-analysis.json",
        "pe20604_vs_best_layer_2567": "stats/ppl-pe2-vs-bestlayer06/ppl-analysis.json",
        "pe20302_vs_pe0302": "stats/ppl-pe2-vs-pe03/ppl-analysis.json",
        "pe20604_vs_pe0604": "stats/ppl-pe2-vs-pe06/ppl-analysis.json",
        "pe20302p_vs_best_layer_2-6": "stats/ppl-pe2p-vs-bestlayer03/ppl-analysis.json",
        "pe20302p_vs_pe20302": "stats/ppl-pe2p-vs-pe2/ppl-analysis.json",
    }
    for label, rel in comp_specs.items():
        data = json.loads((ROOT / rel).read_text())
        base = data["baseline"]
        arms = {k: v for k, v in data["arms"].items() if k != base}
        out["comparisons"][label] = {
            "baseline": base,
            "baseline_ppl": data["arms"][base]["ppl"],
            "arms": {k: {kk: vv for kk, vv in v.items() if kk != "inc_ll"} for k, v in arms.items()},
        }

    # Routing summary.
    routing = json.loads((ROOT / "stats/routing-summary.json").read_text())
    gen = json.loads((ROOT / "stats/routing-generalization.json").read_text())
    out["routing"] = {
        "samples": routing["samples"],
        "has_weights": routing["has_weights"],
        "decode_rows_per_layer_wiki": 48 * 128,
        "decode_rows_per_layer_docs": 16 * 128,
        "generalization": gen,
    }
    lay = list(csv.DictReader((ROOT / "stats/routing-layer.tsv").open(), delimiter="\t"))
    w = [r for r in lay if r["tag"] == "wiki" and r["phase"] == "decode"]
    out["routing"]["layer_skew_decode_wiki"] = {
        "top4_share_min": min(float(r["top4_share"]) for r in w),
        "top4_share_max": max(float(r["top4_share"]) for r in w),
        "effective_experts_min": min(float(r["effective_experts"]) for r in w),
        "effective_experts_max": max(float(r["effective_experts"]) for r in w),
        "n90_min": min(int(r["n90"]) for r in w), "n90_max": max(int(r["n90"]) for r in w),
        "n95_min": min(int(r["n95"]) for r in w), "n95_max": max(int(r["n95"]) for r in w),
    }

    # Layer sensitivity (single-layer arms).
    sens = {}
    for L in range(40):
        ppl, _, _ = ppl_of(ROOT / f"logs/single/ppl-{L}.log")
        sens[str(L)] = {"ppl": ppl, "delta_vs_all_q1": ppl - out["ppl"]["direct"]["all_q1"]["ppl"]}
    out["layer_sensitivity"] = sens

    # Coverage of each allocation on wiki/docs decode routing.
    rows = list(csv.DictReader((ROOT / "stats/routing-expert.tsv").open(), delimiter="\t"))
    counts = {}
    for r in rows:
        for tag in ("wiki", "docs"):
            for ph in ("decode", "prefill"):
                counts[(tag, ph, int(r["layer"]), int(r["expert"]))] = float(r[f"{tag}_{ph}_count"])
    fams = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")

    def load_alloc(rel: str) -> set:
        s = set()
        p = ROOT / rel
        if not p.exists():
            return s
        for line in p.read_text().splitlines():
            if line.strip():
                l, f, e = line.split("\t")
                s.add((int(l), f, int(e)))
        return s

    allocs = {
        "pe0302": "alloc/alloc-pe0302.tsv",
        "pe0604": "alloc/alloc-pe0604.tsv",
        "pef0302": "alloc-freq/alloc-pef0302.tsv",
        "pe20302": "alloc-measured/alloc-pe20302.tsv",
        "pe20604": "alloc-measured/alloc-pe20604.tsv",
        "pe20302p": "alloc-measured/alloc-pe20302p.tsv",
        "layer38-39": "alloc/alloc-layer38-39.tsv",
        "layer36-39": "alloc/alloc-layer36-39.tsv",
    }
    out["coverage"] = {}
    for name, rel in allocs.items():
        sel = load_alloc(rel)
        cov = {}
        for tag in ("wiki", "docs"):
            for ph in ("decode", "prefill"):
                tot = hit = 0.0
                for (t, p, l, e), c in counts.items():
                    if t != tag or p != ph:
                        continue
                    tot += c * 3
                    for f in fams:
                        if (l, f, e) in sel:
                            hit += c
                cov[f"{tag}_{ph}"] = hit / tot if tot else None
        out["coverage"][name] = cov

    # Arm metadata.
    out["arms"] = {
        "X1_best_layer_2-6": {
            "kind": "deployable_now", "budget_gb": 0.302, "upgrades": 1536,
            "layers": [2, 6], "arm_bytes": 6921279648,
            "expert_tiled_bytes": 4831838208, "deployed_bytes": 6494579904,
            "arm_ppl": out["ppl"]["direct"]["best-layer-2-6"]["ppl"],
            "projected_deployed_ppl": out["ppl"]["direct"]["best-layer-2-6"]["ppl"] + 0.36,
        },
        "X2_pe20302p": {
            "kind": "hot_cold_split_required", "budget_gb": 0.302, "upgrades": 1536,
            "arm_bytes": 6921279648, "expert_tiled_bytes": 4831838208,
            "deployed_bytes": 6494579904,
            "arm_ppl": out["ppl"]["carrier"]["c-pe20302p"]["ppl"],
            "projected_deployed_ppl": out["ppl"]["carrier"]["c-pe20302p"]["ppl"] + 0.36,
        },
        "X3_pe20604": {
            "kind": "hot_cold_split_required", "budget_gb": 0.604, "upgrades": 3072,
            "arm_bytes": 7223269536, "expert_tiled_bytes": 5133828096,
            "deployed_bytes": 6796569792,
            "arm_ppl": out["ppl"]["carrier"]["c-pe20604"]["ppl"],
            "projected_deployed_ppl": out["ppl"]["carrier"]["c-pe20604"]["ppl"] + 0.36,
        },
    }

    (OUT / "final-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["ppl"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
