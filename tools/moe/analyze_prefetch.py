#!/usr/bin/env python3
"""Validate and summarize an LLAMA_MOE_PREFETCH_TIMING=1 run.

Analyze one runtime per log. Duplicate logging sinks are tolerated; concatenated
runs must be split first. Timing includes initialization unless excluded upstream.
"""
import argparse
import statistics
from collections import defaultdict
import json
import re
from pathlib import Path


def analyze(text):
    summaries = re.findall(
        r"prefetch-summary: read=(\d+) published=(\d+) discarded=(\d+) skipped=(\d+) errors=(\d+)", text
    )
    if not summaries or len(set(summaries)) != 1:
        raise ValueError("expected one completed runtime summary (duplicate sinks allowed)")
    totals = dict(zip(("read", "published", "discarded", "skipped", "errors"), map(int, summaries[0])))
    if totals["read"] != totals["published"] + totals["discarded"]:
        raise ValueError("completed reads must equal published plus discarded")
    pattern = (
        r"prefetch-timing: layer=(\d+) overlap_samples=(\d+) overlap_mean_us=([\d.]+) "
        r"late_jobs=(\d+) read_jobs=(\d+) read_experts=(\d+) batch_wall_per_expert_us=([\d.]+)"
    )
    layers = {}
    for layer, samples, window, late, jobs, reads, cost in re.findall(pattern, text):
        row = dict(layer=int(layer), overlap_samples=int(samples), overlap_mean_us=float(window),
                   late_jobs=int(late), read_jobs=int(jobs), read_experts=int(reads),
                   batch_wall_per_expert_us=float(cost))
        if row["late_jobs"] > row["overlap_samples"]:
            raise ValueError(f"layer {layer}: more late jobs than observed demands")
        if layer in layers and row != layers[layer]:
            raise ValueError(f"layer {layer}: multiple distinct summaries; split concatenated runs")
        layers[layer] = row
    if not layers or sum(row["read_experts"] for row in layers.values()) != totals["read"]:
        raise ValueError("missing per-layer timing or completed-read totals disagree")
    return dict(totals=totals, layers=sorted(layers.values(), key=lambda row: row["layer"]), notes=[
        "Published means installed, not necessarily used by a subsequent demand.",
        "Overlap pairs accepted prefetch arm and next-layer demand in the same graph epoch.",
        "Late means the whole job remained pending/running at demand; no demand wait is added.",
        "Batch wall per expert measures parallel I/O throughput, not isolated read latency.",
        "Means do not establish a safe per-layer admission bound; compare depths and lateness.",
        "Initialization is included; this is not a pure steady-state decode measurement.",
    ])


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=values[min(len(values) - 1, int(.9 * (len(values) - 1)))])


def analyze_trace(text):
    """Pair CPU arm/demand windows; separately sum post-routing DSP FFN ops.

    CPU and DSP clocks are not aligned. Neither summary establishes a per-expert
    useful-read deadline. Duplicate log sinks are removed by (layer,epoch,rows).
    """
    jobs, demands = {}, {}
    for m in re.finditer(r"prefetch-job: layer=(\d+) epoch=(\d+) rows=(\d+) requested=(\d+) reads=(\d+) queue_us=([\d.]+) wall_us=([\d.]+)", text):
        layer, epoch, rows, requested, reads = map(int, m.groups()[:5])
        key = (layer, epoch, rows)
        value = dict(layer=layer, epoch=epoch, rows=rows, requested=requested, reads=reads,
                     queue_us=float(m[6]), wall_us=float(m[7]))
        if key in jobs and jobs[key] != value:
            raise ValueError("distinct jobs with same epoch/layer; split concatenated logs")
        jobs[key] = value
    for m in re.finditer(r"prefetch-demand: layer=(\d+) epoch=(\d+) rows=(\d+) window_us=([\d.]+) late=(\d+)", text):
        key = tuple(map(int, m.groups()[:3]))
        value = dict(window_us=float(m[4]), late=int(m[5]))
        if key in demands and demands[key] != value:
            raise ValueError("distinct demands with same epoch/layer")
        demands[key] = value
    groups = defaultdict(list)
    for key, job in jobs.items():
        groups[job['rows']].append(dict(job, **demands.get(key, {})))
    cpu = []
    for rows, group in sorted(groups.items()):
        paired = [r for r in group if 'window_us' in r]
        cpu.append(dict(rows=rows, jobs=len(group), matched_demands=len(paired),
                        late_fraction=statistics.mean(r['late'] for r in paired) if paired else None,
                        requested=distribution([r['requested'] for r in group]),
                        reads=distribution([r['reads'] for r in group]),
                        queue_us=distribution([r['queue_us'] for r in group]),
                        job_wall_us=distribution([r['wall_us'] for r in group]),
                        window_us=distribution([r['window_us'] for r in paired])))

    active, ffn = {}, []
    for line in text.splitlines():
        m = re.search(r"ggml-hex: (HTP\d+) profile-op (.*)", line)
        if not m:
            continue
        device, payload = m.groups()
        parts = payload.split('|')
        if len(parts) < 7 or parts[0] == 'OPBATCH':
            continue
        usec = re.search(r"usec (\d+)", parts[6])
        if not usec:
            continue
        output = parts[1].split(' -> ')[-1]
        start = re.fullmatch(r"ffn_moe_(?:up|gate_up)-(\d+)", output)
        if start and '.weight.dec' in parts[1] and parts[0].startswith('MUL_MAT_ID'):
            dims = list(map(int, parts[2].split(' -> ')[-1].split(':')))
            active[device] = dict(layer=int(start[1]), rows=dims[2], dsp_us=0, sparse_matmul_us=0)
        if device not in active:
            continue
        current = active[device]
        if not parts[0].startswith('FENCE'):
            current['dsp_us'] += int(usec[1])
        if parts[0].startswith('MUL_MAT_ID'):
            current['sparse_matmul_us'] += int(usec[1])
        if output == f"ffn_out-{current['layer']}":
            ffn.append(active.pop(device))
    dsp = []
    for rows in sorted(set(r['rows'] for r in ffn)):
        group = [r for r in ffn if r['rows'] == rows]
        dsp.append(dict(rows=rows, layers=distribution([r['dsp_us'] for r in group]),
                        sparse_matmul_us=distribution([r['sparse_matmul_us'] for r in group]),
                        per_layer=[dict(layer=layer,
                            dsp_us=distribution([r['dsp_us'] for r in group if r['layer'] == layer]))
                            for layer in sorted(set(r['layer'] for r in group))]))
    return dict(cpu=cpu, dsp_ffn=dsp, jobs=list(jobs.values()), demands=[dict(layer=k[0], epoch=k[1], rows=k[2], **v) for k,v in demands.items()], notes=[
        'rows=4 selects the full-width target verify batch when draft length is 3; rows=2 includes initialization and partial verifies.',
        'DSP FFN starts at sparse gate/up matmul and ends at shared+sparse ffn_out; excludes router, expert disk reads and host/RPC overhead.',
        'CPU window includes FFN plus subsequent attention/router and inter-device transfer; it is not FFN time.',
        'Queue and wall times describe whole prefetch jobs, including skipped callbacks; reads counts completed disk expert reads.',
        'Profile logs perturb CPU/DSP timing. Use separate uninstrumented throughput runs.',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = args.log.read_text(errors="backslashreplace")
    result = json.dumps(dict(**analyze(text), trace=analyze_trace(text)), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
