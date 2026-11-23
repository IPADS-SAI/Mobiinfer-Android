#!/usr/bin/env python3
"""Build the locality viewer data.json from `llama-moe-locality` output.

Pipeline (see docs/hot-cold-expert/LOCALITY.md):

    llama-moe-profile  SAMPLES.jsonl  ->  *-routes.jsonl
    llama-moe-locality TRAIN EVAL ... ->  locality-<model>.csv   (per layer: LRU / pin / reuse distance)
    build_locality_data.py            ->  <model>.json           (what profile-web /moe/locality renders)

Usage:

    build_locality_data.py --csv locality-qwen3.csv --model qwen3 \\
        --label "Qwen3-30B-A3B · 中文日常 1024/256" \\
        --train-samples 1024 --heldout-samples 256 --dataset "中文日常 1024/256" \\
        --caps cfg-qwen3-default.csv --out profile-web/data/locality/qwen3.json

    build_locality_data.py --index-only --data-dir profile-web/data/locality

The output keeps the viewer dumb: every number the page draws is precomputed here,
and miss rates are stored as percentages (0-100) per layer.
"""
import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path


def capacities(header):
    """Capacities are discovered from the CSV columns (lru8 ... st128)."""
    caps = []
    for row in csv.reader([header]):
        for cell in row:
            m = re.fullmatch(r"lru(\d+)", cell)
            if m:
                caps.append(int(m.group(1)))
    if not caps:
        raise SystemExit("no lru<N> columns in the CSV header")
    return caps


def parse_caps_file(path):
    """Optional per-layer capacity suggestion: 'layer,slots' lines."""
    slots = {}
    if not path:
        return slots
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        layer, cap = line.split(",")[:2]
        slots[int(layer)] = int(cap)
    return slots


def build(args):
    text = Path(args.csv).read_text()
    header, *lines = text.splitlines()
    caps = capacities(header)
    rows = list(csv.DictReader([header, *lines]))
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    cfg = parse_caps_file(args.cfg_caps) if args.cfg_caps else {}
    has_cfg = "cap" in (rows[0] or {})

    layers, total_refs, total_steps = [], 0, 0
    weighted_lru = [0.0] * len(caps)
    weighted_st = [0.0] * len(caps)
    for r in rows:
        refs = int(r["refs"])
        steps = int(r["steps"])
        layer = {
            "l": int(r["layer"]),
            "steps": steps,
            "refs": refs,
            "distinct": int(r["distinct"]),
            "lru": [round(float(r[f"lru{c}"]) / 10.0, 2) for c in caps],   # per 1000 refs -> percent
            "st": [round(float(r[f"st{c}"]) / 10.0, 2) for c in caps],
            "hot32": round(float(r["hot32"]), 1),
            "sd": [round(float(r["sd50"]), 0), round(float(r["sd90"]), 0), round(float(r["sd99"]), 0)],
            "comp": round(float(r["compulsory"]), 2),
            "top8": round(float(r["top8share"]), 4),
            "n95": int(r["n95"]),
        }
        if has_cfg and r.get("cap"):
            layer["cfg"] = {
                "cap": int(r["cap"]),
                "lru": round(float(r["lru_cfg"]) / 10.0, 2),
                "st": round(float(r["st_cfg"]) / 10.0, 2),
            }
        elif layer["l"] in cfg:
            layer["cfg"] = {"cap": cfg[layer["l"]]}
        layers.append(layer)
        total_refs += refs
        total_steps += steps
        for i, c in enumerate(caps):
            weighted_lru[i] += float(r[f"lru{c}"]) * refs
            weighted_st[i] += float(r[f"st{c}"]) * refs

    total = {
        "steps": total_steps,
        "refs": total_refs,
        "lru": [round(v / total_refs / 10.0, 2) for v in weighted_lru],
        "st": [round(v / total_refs / 10.0, 2) for v in weighted_st],
    }

    # Capacity of the reference budget = what the engine would actually be given.
    suggested = [l["cfg"]["cap"] for l in layers if "cfg" in l]
    meta = {
        "model": args.model,
        "label": args.label or args.model,
        "n_layers": len(layers),
        "n_experts": args.experts or 0,
        "train_samples": args.train_samples,
        "heldout_samples": args.heldout_samples,
        "dataset": args.dataset,
        "phase": args.phase,
        "ranking": "训练集 decode 频率（与冷热页同口径）",
        "method": "逐 token 重放路由轨迹到全相联 LRU（每步 8 个专家，同步请求互不淘汰）",
        "note": args.note,
        "generated": args.generated or date.today().isoformat(),
    }
    out = {
        "meta": meta,
        "caps": caps,
        "layers": layers,
        "total": total,
        "budget": {
            "name": args.budget_name,
            "slots_total": sum(suggested) if suggested else 0,
            "per_layer": bool(suggested),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes) "
          f"{len(layers)}L x {meta['n_experts']}E caps={caps} budget={out['budget']['slots_total']}")


def build_index(args):
    data_dir = Path(args.data_dir)
    entries = []
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        data = json.loads(path.read_text())
        meta = data.get("meta", {})
        entries.append({
            "model": meta.get("model", path.stem),
            "title": meta.get("label", path.stem),
            "layers": meta.get("n_layers"),
            "experts": meta.get("n_experts"),
            "dataset": meta.get("dataset", ""),
            "note": meta.get("note", ""),
            "slots": (data.get("budget") or {}).get("slots_total", 0),
        })
    out = data_dir / "index.json"
    out.write_text(json.dumps({"entries": entries}, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(entries)} models)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="llama-moe-locality CSV for one model")
    ap.add_argument("--model", help="model id used in the URL, e.g. qwen3")
    ap.add_argument("--label", default="", help="human readable title")
    ap.add_argument("--experts", type=int, default=0, help="experts per layer (128 / 256)")
    ap.add_argument("--train-samples", type=int, default=0)
    ap.add_argument("--heldout-samples", type=int, default=0)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--phase", default="decode")
    ap.add_argument("--note", default="")
    ap.add_argument("--cfg-caps", help="optional layer,slots CSV (suggested window per layer)")
    ap.add_argument("--budget-name", default="default")
    ap.add_argument("--generated", default="")
    ap.add_argument("--out")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--data-dir", help="directory to index (with --index-only)")
    args = ap.parse_args()

    if args.index_only:
        if not args.data_dir:
            ap.error("--index-only needs --data-dir")
        return build_index(args)
    for required in ("csv", "model", "out"):
        if not getattr(args, required):
            ap.error(f"--{required.replace('_', '-')} is required")
    return build(args)


if __name__ == "__main__":
    main()
