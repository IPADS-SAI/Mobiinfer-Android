#!/usr/bin/env python3
"""Validate expert residency counts and export one layer for sched_repro.

Counts are NOT page bitmaps or router traces. Scaling to another expert size
requires --target-pages and preserves fractions, not physical cache layout.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def validate(snapshot):
    page = snapshot['page_size']
    if not isinstance(page, int) or page <= 0:
        raise ValueError('invalid page_size')
    result = []
    seen = set()
    for layer in snapshot['layers']:
        counts = layer['resident_pages_per_expert']
        size = layer['bytes']
        if not counts or size <= 0 or size % (page * len(counts)):
            raise ValueError('layer size is not an integral number of expert pages')
        pages = size // (page * len(counts))
        if any(type(n) is not int or not 0 <= n <= pages for n in counts):
            raise ValueError('resident count outside expert; do not clamp')
        if sum(counts) * page != layer['resident_bytes']:
            raise ValueError('expert counts do not match resident_bytes')
        if layer['layer'] in seen:
            raise ValueError('duplicate layer')
        seen.add(layer['layer'])
        result.append((layer['layer'], pages, counts))
    if not result:
        raise ValueError('empty snapshot')
    return result


def export_profile(layers, layer_id, target_pages=None):
    matches = [x for x in layers if x[0] == layer_id]
    if len(matches) != 1:
        raise ValueError('layer not found')
    _, pages, counts = matches[0]
    target = pages if target_pages is None else target_pages
    if target < 1:
        raise ValueError('target pages must be positive')
    scaled = [(n * target + pages // 2) // pages for n in counts]
    return f'{len(counts)} {target}\n' + ' '.join(map(str, scaled)) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot', type=Path)
    ap.add_argument('--layer', type=int)
    ap.add_argument('--target-pages', type=int)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    layers = validate(snapshot)
    ratios = [n / pages for _, pages, counts in layers for n in counts]
    counts = collections.Counter(n for _, _, ns in layers for n in ns)
    report = dict(sha256=hashlib.sha256(raw).hexdigest(), layers=len(layers),
                  experts=len(ratios), source_pages=sorted({p for _, p, _ in layers}),
                  mean_residency=sum(ratios)/len(ratios),
                  hot=sum(r >= .999 for r in ratios),
                  cold=sum(r <= .02 for r in ratios),
                  partial=sum(.02 < r < .999 for r in ratios),
                  common_page_counts=counts.most_common(8),
                  semantics=snapshot.get('semantics'), page_positions_available=False)
    if args.output:
        if args.layer is None:
            ap.error('--output requires --layer')
        text = export_profile(layers, args.layer, args.target_pages)
        # Never overwrite source data or an existing experiment artifact.
        with args.output.open('x') as f:
            f.write(text)
        report.update(layer=args.layer, target_pages=args.target_pages,
                      profile=str(args.output), profile_sha256=hashlib.sha256(text.encode()).hexdigest())
        with Path(str(args.output)+'.json').open('x') as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
