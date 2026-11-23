#!/usr/bin/env python3
"""Parse llama-perplexity logs and run a paired chunk bootstrap.

``llama-perplexity --chunks N`` prints cumulative PPL after every chunk::

    [1]34.97,[2]40.49,...

The per-chunk mean log-NLL is recovered as
``inc_ll_i = i*ln(PPL_i) - (i-1)*ln(PPL_{i-1})`` (the same reduction used by
T-0014).  Arms that score the same corpus have identical chunk boundaries, so a
paired bootstrap over the 128 chunk indices gives a CI for ``PPL_arm - PPL_base``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

CHUNK_RE = re.compile(r"\[(\d+)\]([0-9]+(?:\.[0-9]+)?)")
FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    values: dict[int, float] = {}
    for m in CHUNK_RE.finditer(text):
        values[int(m.group(1))] = float(m.group(2))
    if not values:
        raise SystemExit(f"no cumulative PPL in {path}")
    last = max(values)
    cumulative = [values[i] for i in range(1, last + 1)]
    inc = []
    prev = 0.0
    for i, ppl in enumerate(cumulative, 1):
        ll = math.log(ppl)
        inc.append(i * ll - (i - 1) * prev)
        prev = ll
    final = FINAL_RE.search(text)
    return {
        "path": str(path),
        "chunks": last,
        "ppl": math.exp(sum(inc) / len(inc)),
        "final_reported": float(final.group(1)) if final else None,
        "inc_ll": inc,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ppl", action="append", required=True, help="name=path")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4141)
    args = ap.parse_args()

    arms: dict[str, dict] = {}
    for spec in args.ppl:
        name, path = spec.split("=", 1)
        arms[name] = parse_log(Path(path))
    if args.baseline not in arms:
        raise SystemExit("baseline not in --ppl")
    n = {a["chunks"] for a in arms.values()}
    if len(n) != 1:
        raise SystemExit(f"arms have different chunk counts: {n}")
    n = n.pop()

    base = np.asarray(arms[args.baseline]["inc_ll"])
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.boot, n))

    out = {"baseline": args.baseline, "chunks": n, "bootstrap": args.boot,
           "seed": args.seed, "arms": {}}
    for name, arm in arms.items():
        x = np.asarray(arm["inc_ll"])
        ppl = math.exp(float(x.mean()))
        if name == args.baseline:
            out["arms"][name] = {
                "path": arm["path"], "ppl": ppl,
                "final_reported": arm["final_reported"],
                "delta_ppl": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
            }
            continue
        # Paired bootstrap: same chunk indices for both arms.
        boot_base = np.exp(base[idx].mean(axis=1))
        boot_arm = np.exp(x[idx].mean(axis=1))
        delta = boot_arm - boot_base
        out["arms"][name] = {
            "path": arm["path"],
            "ppl": ppl,
            "final_reported": arm["final_reported"],
            "delta_ppl": ppl - math.exp(float(base.mean())),
            "ci95_low": float(np.percentile(delta, 2.5)),
            "ci95_high": float(np.percentile(delta, 97.5)),
            "delta_pct": 100.0 * (ppl / math.exp(float(base.mean())) - 1.0),
            "delta_pct_ci95_low": float(np.percentile(100.0 * (boot_arm / boot_base - 1.0), 2.5)),
            "delta_pct_ci95_high": float(np.percentile(100.0 * (boot_arm / boot_base - 1.0), 97.5)),
            "inc_ll": x.tolist(),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ppl-analysis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    with (args.out / "ppl-table.tsv").open("w", encoding="utf-8") as t:
        t.write("arm\tppl\tfinal_reported\tdelta_ppl\tci95_low\tci95_high\tdelta_pct\n")
        for name, a in out["arms"].items():
            t.write(f"{name}\t{a['ppl']:.6f}\t{a['final_reported']}\t{a['delta_ppl']:.6f}\t"
                    f"{a['ci95_low']:.6f}\t{a['ci95_high']:.6f}\t{a.get('delta_pct', 0.0):.4f}\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "inc_ll"}
                      for k, v in out["arms"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
