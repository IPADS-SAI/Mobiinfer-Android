#!/usr/bin/env python3
"""Finite-candidate calibration; never claim a global scheduling optimum.

Run on the device with a compiled sched_repro. Uses whole batches from one
exported layer profile. Independent seeds are mandatory for evaluation.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess


def candidates(workers):
    for order, cap, release, chunk in itertools.product(
            ('random', 'cold', 'hot', '1c1h', '2c1h', '1c2h', 'proportional'),
            range(1, workers+1), (0, 1), (0, 262144, 524288)):
        if release and cap == workers:
            continue
        yield f'{order}:{cap}:{release}:{chunk}'


def score(row):
    # All candidates pay residency-probe + dispatch-plan cost. A genuine
    # no-probe baseline needs a separate experiment, not this arithmetic.
    return row['ms'] + row['probe_ms'] + row['plan_ms']


def measure(args, policies, seeds, reps, log, phase):
    data = {p: [] for p in policies}
    rng = random.Random(917)
    for seed in seeds:
        for rep in range(reps):
            order = list(policies)
            rng.shuffle(order)
            for policy in order:
                env = dict(os.environ, SCHED_POLICY=policy,
                           SCHED_PROFILE=str(args.profile.resolve()), SCHED_LAYOUT=args.layout)
                proc = subprocess.run([str(args.bench.resolve()), str(args.file.resolve()),
                                       str(args.k), str(args.workers), '0', str(seed), '1'],
                                      env=env, text=True, capture_output=True, check=True,
                                      timeout=60)
                rows = [json.loads(line) for line in proc.stdout.splitlines()]
                if len(rows) != 1 or not rows[0]['valid']:
                    raise ValueError(f'invalid cache or physical I/O measurement: {proc.stdout}')
                row = rows[0]
                row.update(phase=phase, trial_rep=rep)
                log.write(json.dumps(row)+'\n')
                log.flush()
                data[policy].append(row)
    return data


def rank(data):
    return sorted(data, key=lambda p: statistics.mean(score(r) for r in data[p]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('train', 'evaluate'))
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--file', type=Path, required=True)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--layout', choices=('prefix', 'spread'), default='prefix')
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.k <= 256 or not 1 <= args.workers <= 16 or args.reps < 1 or args.top < 1:
        ap.error('invalid dimensions')
    if len(set(args.seeds)) != len(args.seeds) or any(not 1 <= s < 2**31 for s in args.seeds):
        ap.error('seeds must be distinct positive signed integers')
    identity = dict(k=args.k, workers=args.workers, layout=args.layout,
                    profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                    file_size=args.file.stat().st_size,
                    bench_sha256=hashlib.sha256(args.bench.read_bytes()).hexdigest())
    all_policies = list(candidates(args.workers))
    if args.mode == 'train':
        if args.config.exists():
            ap.error('refusing to overwrite config')
        with args.log.open('x') as log:
            coarse = measure(args, all_policies, args.seeds, 1, log, 'coarse')
            finalists = rank(coarse)[:args.top]
            # Keep baselines in the independent remeasurement stage.
            finalists = list(dict.fromkeys(finalists + [f'{p}:{args.workers}:0:0' for p in ('random','cold','1c1h')]))
            fine = measure(args, finalists, args.seeds, args.reps, log, 'fine')
        config = dict(version=1, identity=identity, seeds=args.seeds,
                      selected=rank(fine)[0], finalists=rank(fine),
                      candidates=len(all_policies), objective='read+probe+plan_ms')
        with args.config.open('x') as f:
            json.dump(config, f, indent=2)
        print(json.dumps(config, indent=2))
    else:
        config = json.loads(args.config.read_text())
        if config['identity'] != identity:
            ap.error('configuration geometry/profile/layout/binary mismatch; recalibrate')
        if set(config['seeds']) & set(args.seeds):
            ap.error('evaluation seeds overlap calibration seeds')
        with args.log.open('x') as log:
            data = measure(args, all_policies, args.seeds, args.reps, log, 'heldout')
        selected = config['selected']
        means = {p: statistics.mean(score(r) for r in rows) for p, rows in data.items()}
        best = min(means, key=means.get)
        per_seed = []
        for seed in args.seeds:
            m = {p: statistics.mean(score(r) for r in rows if r['seed']==seed) for p,rows in data.items()}
            per_seed.append(100*(m[selected]/min(m.values())-1))
        print(json.dumps(dict(selected=selected, selected_ms=means[selected],
                              best_candidate=best, best_candidate_ms=means[best],
                              aggregate_regret_pct=100*(means[selected]/means[best]-1),
                              per_seed_regret_pct=per_seed,
                              worst_seed_regret_pct=max(per_seed),
                              caveat='finite-candidate empirical oracle, noise-biased; not global optimum'), indent=2))


if __name__ == '__main__':
    main()
