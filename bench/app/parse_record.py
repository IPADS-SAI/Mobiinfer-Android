#!/usr/bin/env python3
"""parse a MobiInfer benchmark.jsonl tail into one TSV row (and a record JSON)."""
import json
import math
import sys

path, run, tag, prompt, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if not rows:
    sys.exit(2)
r = rows[-1]
t = r.get("timings") or {}
pm, pn = t.get("prompt_ms"), t.get("prompt_n")
dm, dn = t.get("predicted_ms"), t.get("predicted_n")
wall, ttft = r.get("wall_ms"), r.get("ttft_ms")

def valid(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and x >= 0)

if not all(valid(x) for x in (pm, pn, dm, dn, wall, ttft)):
    sys.exit(3)
# Zero tokens / zero durations are not a usable throughput sample.
if pn <= 0 or dn <= 0 or pm <= 0 or dm <= 0:
    sys.exit(3)
decode_tps = dn * 1000.0 / dm if dm else 0.0
prefill_tps = pn * 1000.0 / pm if pm else 0.0
row = [tag, run, "ok", r.get("wall_ms"), r.get("ttft_ms"), pn, round(pm, 1),
       dn, round(dm, 1), round(decode_tps, 2), round(prefill_tps, 2),
       len(r.get("response", "")), '"' + prompt.replace('"', "") + '"']
print("\t".join(str(x) for x in row))
with open(f"{outdir}/run{run}.record.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)
