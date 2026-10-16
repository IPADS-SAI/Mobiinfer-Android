#!/usr/bin/env python3
"""Rank experts on training routes; report held-out coverage, per-layer skew and a
per-layer decode-window suggestion for LLAMA_MOE_LAYER_WINDOWS."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

N_LAYERS = 40
N_EXPERTS = 256
N_ROUTED = 8


def load(path):
    counts = {phase: [Counter() for _ in range(N_LAYERS)] for phase in ('prefill', 'decode')}
    seen = defaultdict(set)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            phase, layer = row['phase'], row['layer']
            if phase not in counts or not 0 <= layer < N_LAYERS:
                raise ValueError('expected MiniCPM 4.7 route phase/layer')
            for experts in row['experts']:
                if len(experts) != N_ROUTED or len(set(experts)) != N_ROUTED or any(not 0 <= e < N_EXPERTS for e in experts):
                    raise ValueError(f'expected {N_ROUTED} distinct experts in [0,{N_EXPERTS})')
                counts[phase][layer].update(experts)
            seen[(row['sample'], phase)].add(layer)
    samples = {sample for sample, _ in seen}
    if not samples:
        raise ValueError('empty routing trace')
    for sample in samples:
        for phase in counts:
            if seen[(sample, phase)] != set(range(N_LAYERS)):
                raise ValueError(f'incomplete {N_LAYERS}-layer coverage: {sample} {phase}')
    return counts, samples


def skew(counter):
    total = sum(counter.values())
    if total <= 0:
        return None
    per_expert = [counter.get(e, 0) for e in range(N_EXPERTS)]
    ordered = sorted(per_expert, reverse=True)
    cumulative = []
    acc = 0
    for c in ordered:
        acc += c
        cumulative.append(acc)

    def n_for(frac):
        target = frac * total
        for i, a in enumerate(cumulative):
            if a >= target:
                return i + 1
        return N_EXPERTS

    probs = [c / total for c in per_expert if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    mean = total / N_EXPERTS
    return {
        'total': total,
        'distinct_active': sum(1 for c in per_expert if c > 0),
        'zero_count': sum(1 for c in per_expert if c == 0),
        'cold_lt_0.1mean': sum(1 for c in per_expert if c < 0.1 * mean),
        'cold_lt_0.01mean': sum(1 for c in per_expert if c < 0.01 * mean),
        'entropy_bits': entropy,
        'entropy_norm': entropy / math.log2(N_EXPERTS),
        'effective_experts': 1.0 / sum(p * p for p in probs),
        'top1_share': ordered[0] / total,
        'top8_share': sum(ordered[:N_ROUTED]) / total,
        'n_50pct': n_for(0.50),
        'n_80pct': n_for(0.80),
        'n_90pct': n_for(0.90),
        'n_95pct': n_for(0.95),
        'n_99pct': n_for(0.99),
    }


def suggest_windows(skew_by_layer, budget, lo, hi):
    weights = {}
    for layer, phases in skew_by_layer.items():
        stat = phases.get('decode', {}).get('train')
        weights[layer] = max(1, stat['n_95pct'] if stat else lo)
    total = sum(weights.values())
    out = {}
    for layer, w in weights.items():
        c = round(budget * w / total)
        out[layer] = max(lo, min(hi, c))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=Path)
    parser.add_argument('heldout', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--slot-budget', type=int, default=0,
                        help='total per-layer decode-window slots to distribute (0 = suggest n_95pct per layer)')
    parser.add_argument('--min-slots', type=int, default=8, help='lower clamp per layer')
    parser.add_argument('--max-slots', type=int, default=1024, help='upper clamp per layer')
    args = parser.parse_args()
    train, train_samples = load(args.train)
    heldout, heldout_samples = load(args.heldout)
    if train_samples & heldout_samples:
        raise ValueError('training and held-out sample ids overlap')

    ranking = [sorted(range(N_EXPERTS), key=lambda e: (-train['decode'][layer][e], e)) for layer in range(N_LAYERS)]
    report = dict(train_samples=len(train_samples), heldout_samples=len(heldout_samples),
                  ranking_phase='decode', ranking_source='training only', layers=[])
    for layer in range(N_LAYERS):
        result = dict(layer=layer, ranked_experts=ranking[layer], coverage={})
        for capacity in (8, 16, 24, 32, 48, 64):
            selected = set(ranking[layer][:capacity])
            coverage = {}
            for split, counts in (('train', train), ('heldout', heldout)):
                for phase in ('prefill', 'decode'):
                    total = sum(counts[phase][layer].values())
                    hits = sum(n for expert, n in counts[phase][layer].items() if expert in selected)
                    coverage[f'{split}_{phase}'] = dict(selected=hits, total=total, fraction=hits/total if total else None)
            result['coverage'][capacity] = coverage
        report['layers'].append(result)

    skew_by_layer = {}
    for layer in range(N_LAYERS):
        skew_by_layer[layer] = {
            phase: {'train': skew(train[phase][layer]), 'heldout': skew(heldout[phase][layer])}
            for phase in ('prefill', 'decode')
        }

    args.output.mkdir(parents=True, exist_ok=True)
    table = '# MiniCPM-V 4.7: training decode-frequency rank; %d layers, %d experts per layer\n' % (N_LAYERS, N_EXPERTS)
    table += ''.join(' '.join(map(str, ids)) + '\n' for ids in ranking)
    (args.output / 'prewarm-ranked.txt').write_text(table)

    report['trace_sha256'] = {str(path): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                              for path in (args.train, args.heldout)}
    (args.output / 'coverage.json').write_text(json.dumps(report, indent=2))

    with (args.output / 'expert-counts.tsv').open('w') as out:
        out.write('layer\texpert\ttrain_prefill\ttrain_decode\theldout_prefill\theldout_decode\n')
        for layer in range(N_LAYERS):
            for expert in range(N_EXPERTS):
                out.write('%d\t%d\t%d\t%d\t%d\t%d\n' % (
                    layer, expert,
                    train['prefill'][layer].get(expert, 0), train['decode'][layer].get(expert, 0),
                    heldout['prefill'][layer].get(expert, 0), heldout['decode'][layer].get(expert, 0)))

    (args.output / 'layer-skew.json').write_text(json.dumps(skew_by_layer, indent=2))

    print('layer  dec_total  active  zero  cold<0.1m  n80  n90  n95  eff    top8')
    for layer in sorted(range(N_LAYERS), key=lambda l: -skew_by_layer[l]['decode']['train']['n_95pct']):
        s = skew_by_layer[layer]['decode']['train']
        print('L%02d  %9d  %6d  %4d  %9d  %3d  %3d  %3d  %5.1f  %.3f' % (
            layer, s['total'], s['distinct_active'], s['zero_count'], s['cold_lt_0.1mean'],
            s['n_80pct'], s['n_90pct'], s['n_95pct'], s['effective_experts'], s['top8_share']))

    windows = (suggest_windows(skew_by_layer, args.slot_budget, args.min_slots, args.max_slots)
               if args.slot_budget > 0 else
               {layer: max(args.min_slots, min(args.max_slots, skew_by_layer[layer]['decode']['train']['n_95pct']))
                for layer in range(N_LAYERS)})
    spec = ','.join('%d:%d' % (layer, windows[layer]) for layer in range(N_LAYERS))
    (args.output / 'layer-windows-suggested.txt').write_text(
        '# decode-window suggestion from training decode n_95pct%s\n' % (' scaled to slot budget' if args.slot_budget > 0 else '') +
        'LLAMA_MOE_LAYER_WINDOWS="%s"\n' % spec)
    print('LLAMA_MOE_LAYER_WINDOWS="%s"' % spec)

    for capacity in (8, 16, 24, 32, 48, 64):
        rows = [row['coverage'][capacity]['heldout_decode'] for row in report['layers']]
        print(f"slots={capacity} heldout_decode_coverage={sum(r['selected'] for r in rows)/sum(r['total'] for r in rows):.4f}")
    print('Offline frequency coverage is not an online cache-hit-rate or speedup estimate.')


if __name__ == '__main__':
    main()
