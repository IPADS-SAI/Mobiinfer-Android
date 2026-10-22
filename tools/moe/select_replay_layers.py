#!/usr/bin/env python3
"""Select layers from held-out routes; report cache misses, not device latency."""
import argparse
from collections import OrderedDict, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capacity", type=int, default=32)
    args = parser.parse_args()
    if args.capacity < 8:
        parser.error("capacity must be at least 8")
    sequences = defaultdict(list)
    with args.routes.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["phase"] == "decode":
                sequences[record["sample"], int(record["layer"])].extend(record["experts"])
    totals = defaultdict(lambda: {"requests": 0, "misses": 0, "steps": 0,
                                  "sorted_runs": 0, "ordered_runs": 0})
    for (_, layer), rows in sequences.items():
        cache = OrderedDict()
        stat = totals[layer]
        for row in rows:
            requested = list(dict.fromkeys(row))
            if len(requested) > args.capacity:
                raise ValueError("one request exceeds cache capacity")
            # Protect every current request before allocating any missing entry.
            protected = set(requested)
            missing = [expert for expert in requested if expert not in cache]
            for expert in requested:
                if expert in cache:
                    cache.move_to_end(expert)
            for expert in missing:
                if len(cache) == args.capacity:
                    victim = next(e for e in cache if e not in protected)
                    del cache[victim]
                cache[expert] = None
            ordered = missing
            sorted_ids = sorted(missing)
            for key, ids in [("ordered_runs", ordered), ("sorted_runs", sorted_ids)]:
                stat[key] += sum(i == 0 or expert != ids[i - 1] + 1
                                 for i, expert in enumerate(ids))
            stat["requests"] += len(requested)
            stat["misses"] += len(missing)
            stat["steps"] += 1
    if len(totals) < 2:
        raise ValueError("need at least two layers")
    ranked = sorted(totals, key=lambda layer: (totals[layer]["misses"] /
                                             totals[layer]["steps"], layer))
    typical = ranked[len(ranked) // 2]
    high_miss = ranked[-1]
    report = {"source": str(args.routes), "capacity": args.capacity,
              "cache_start": "empty per sample; decode only; no prefill seed",
              "limitations": "sequential target routes; no speculative or device timing",
              "typical_layer": typical, "high_miss_layer": high_miss,
              "layers": {str(k): dict(v, misses_per_step=v["misses"] / v["steps"])
                         for k, v in sorted(totals.items())}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    selected = {typical, high_miss}
    # Include predecessors for subsequent adjacent-layer scheduling experiments.
    selected.update(layer - 1 for layer in list(selected) if layer > 0)
    with (args.output / "routes.jsonl").open("w") as stream:
        for (sample, layer), rows in sequences.items():
            if layer in selected:
                stream.write(json.dumps({"sample": sample, "layer": layer,
                                         "experts": rows}) + "\n")
    print(json.dumps({k: report[k] for k in ("typical_layer", "high_miss_layer")}))


if __name__ == "__main__":
    main()
