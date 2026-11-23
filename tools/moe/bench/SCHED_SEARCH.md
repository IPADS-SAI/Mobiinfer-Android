# Experimental profile-driven dispatch search

These standalone tools calibrate a finite set of dynamic pull-queue policies.
They do not change slot loading or claim a global optimum. Calibration files
and measurement logs belong outside the repository. Review AI-assisted changes
before proposing a contribution, as required by CONTRIBUTING.md.

## Import a snapshot

`sched_snapshot.py` validates per-layer file size, expert count, page size,
resident counts, and aggregate resident bytes. It never clamps out-of-range
counts. A snapshot with 384 pages per expert must not be divided by 330.

```sh
python3 tools/moe/bench/sched_snapshot.py /path/snapshot.json \
  --layer 8 --output /path/artifacts/layer8.profile
```

A profile contains `EXPERT_COUNT PAGES_PER_EXPERT`, then one resident-page count
per expert id. The current `sched_repro` file geometry is 256 experts, each
1351680 bytes; other geometry is rejected. To explicitly transfer *fractions*
from a different source geometry, add `--target-pages 330`. This is a synthetic
scaled replay, NOT a replay of the original physical file or its page positions.
An adjacent JSON sidecar records the source hash and conversion provenance.

Counts do not reveal page locations. `SCHED_LAYOUT=prefix|spread` selects a
synthetic prefix or evenly dispersed pattern with exactly the requested count.
Before each policy the benchmark drops the test file's cache, prewarms specified
pages, and checks **every page bit** against the target. It never drops global
cache. Do not run concurrently with inference using this file.

Snapshot expert ids are retained. Batch ids are sampled uniformly without
replacement within a single layer, NOT from an actual router trace. Evaluate
multiple layers separately; a global histogram hides layer correlations.

## Benchmark a policy

```sh
cc -O2 -Wall -Wextra -pthread tools/moe/bench/sched_repro.c -o /path/sched_repro
SCHED_PROFILE=/path/layer8.profile SCHED_LAYOUT=prefix \
SCHED_POLICY=proportional:3:1:524288 \
/path/sched_repro /path/layer_000.bin 8 4 0 42 3
```

Policy syntax: `ORDER:COLD_LIMIT:RELEASE_LIMIT:CHUNK_BYTES`.
Orders: `random`, `cold`, `hot`, `1c1h`, `2c1h`, `1c2h`, `proportional`.
Cold-class jobs have any missing pages and are ordered by descending missing
bytes; hot-class jobs have none. This is continuous work ranking but still a
binary admission/class interleaving heuristic, not a continuous resource model.
`proportional` distributes classes in proportion to their job counts, not bytes.
For `random`, input order is preserved. Release=1 restores the worker-count limit
after all fully hot jobs finish. Chunk size is 0 (whole expert), 262144, or 524288.
Reads within an expert remain nonpreemptive at the application task level.

Optional `maskch...` orders (up to 16 tasks) permit small binary-class order
experiments. They do not enumerate permutations of unequal partial experts.

With a profile, CLI PCT is ignored (use zero). Without a profile, the existing
whole-hot/cold PCT setup remains available. Without SCHED_POLICY, the original
seven comparison policies run, in randomized order.

JSONL includes expert ids, actual resident counts, completion times, probe/plan
costs, physical read bytes and extra read bytes. A valid run requires the exact
starting bitmap and physical read bytes >= missing bytes. Extra reads are
reported, NOT hidden or forced to zero: partial-page patterns and request
boundaries may amplify I/O. This inequality alone cannot identify the cause of
extra reads. Unavailable I/O counters or under-accounting invalidate a run.

## Calibrate and evaluate (on device)

```sh
python sched_search.py train --bench ./sched_repro --file /path/layer_000.bin \
  --profile /path/layer8.profile --layout prefix --k 8 --workers 4 \
  --seeds 11 23 47 --reps 3 --config /path/config.json --log /path/train.jsonl
python sched_search.py evaluate --bench ./sched_repro --file /path/layer_000.bin \
  --profile /path/layer8.profile --layout prefix --k 8 --workers 4 \
  --seeds 101 211 307 --reps 3 --config /path/config.json --log /path/eval.jsonl
```

Search covers seven orders, all cold limits 1..W, fixed/released limits, and
three request sizes (147 unique candidates at W=4). A coarse pass ranks all
candidates; the leading candidates plus baselines are remeasured. Objective is
read makespan + measured probe + plan cost. The selected policy is frozen in
JSON before testing; evaluation rejects overlapping seeds and changed profile,
layout, binary, K/W, or file size. It remeasures the complete candidate set and
reports regret against its empirical minimum, including per-seed worst regret.

This minimum is noise-biased and is NOT a global optimum or a formal approximation
bound. Small sample counts do not justify a <=5% near-optimality claim. Every
candidate pays probe cost here: compare a genuine no-probe baseline separately
before production decisions. Process startup is outside the read timer; each
policy subprocess starts a fresh worker pool and allocator, then barrier-starts
the measured work. Device state, CPU placement, and thermal drift remain risks.
The saved policy is an experimental exact-profile selection, not a production
online lookup/fallback system. Cross-profile/layer transfer is not implemented.

## Tests

```sh
python3 -m unittest discover -s tools/moe/bench -p test_sched_snapshot.py
cc -O2 -Wall -Wextra -Werror tools/moe/bench/test_sched_policy.c -o /tmp/test-policy
/tmp/test-policy
```
