#!/usr/bin/env python3
"""Emit per-layer hot/cold group specifications from an allocation TSV."""
import argparse, collections, csv
from pathlib import Path

FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True, type=Path)
    ap.add_argument("--alloc", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    freq = collections.defaultdict(dict)
    with args.routing.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            freq[int(r["layer"])][int(r["expert"])] = float(r["wiki_decode_count"])
    sel = collections.defaultdict(list)
    for line in args.alloc.read_text(encoding="utf-8").splitlines():
        if line.strip():
            l, f, e = line.split("\t")
            sel[(int(l), f)].append(int(e))
    total = 0
    with args.out.open("w", encoding="utf-8") as out:
        out.write("layer\tfamily\thot_count\thot_experts_by_frequency_rank\n")
        for (l, f), es in sorted(sel.items()):
            es = sorted(es, key=lambda e: (-freq[l][e], e))
            out.write("%d\t%s\t%d\t%s\n" % (l, f, len(es), ",".join(map(str, es))))
            total += len(es)
    print("groups=%d experts=%d" % (len(sel), total))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
