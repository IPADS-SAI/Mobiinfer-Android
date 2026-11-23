# Dispatch scheduling for MoE expert reads — goal: minimize total time (makespan)

Problem: a decode step reads K experts from flash; W reader threads pull them.
Some experts are already in page cache (fast, ~0.3 ms/expert, but still costs a
worker + a DDR→user memcpy), some must be read from UFS (slow, ~1.3–1.8 ms/expert).
Given we can probe per-expert residency with `mincore`, how (if at all) should we
*order / assign* the reads to minimize the **total time until the whole batch is
done (makespan)**?

Devices: vivo V2546A / SM8850 / UFS 4.0; MiniCPM-V 4.7 Q3_K layer_000.bin
(346,030,080 B = 256 experts × **1,351,680 B** = 1.29 MiB/expert).

> STATUS: This document was corrected on 2026-11-04. An earlier version claimed
> "cold-first is the makespan optimum". That is **wrong** — see §2. Cold-first is
> a *candidate heuristic*. Cold/hot **interleaving** reproducibly beats it in
> several mixed workloads; indiscriminate cold-concurrency caps are usually worse;
> request granularity is another independent dimension; and the `mincore` probe
> cost can erase the gain. No fixed rule is globally optimal here.

## 1. Model (corrected)

- **Task**: read one expert. Work `w_i` = missing bytes (mincore) — for a hot
  expert this is 0 *UFS bytes* but **not zero time**: it still copies page-cache →
  user buffer on a worker thread. Ignoring hot-copy cost is the key mistake of the
  original model.
- **Shared channel**: all W workers share one UFS queue/bandwidth pool. The
  per-window service curve `B(n, request_size, locality, state)` is empirical, not
  a constant 4.1 GiB/s.
- **Goal**: makespan = time until the *last required expert* completes, including
  scheduling/probe overhead.

## 2. Cold-first is NOT optimal (two independent demonstrations)

### 2a. Device experiment (reproduced independently, see §5)

Cold/hot **alternating** dispatch (interleave cold and hot in the dynamic queue,
no static partition) beats cold-first in mixed workloads. Confirmation, 7 seeds ×
8 reps, mean ms (lower better):

| K | W | hot% | random | cold-first | alternating | alt vs cold-first |
|---|---|---|---:|---:|---:|---:|
| 8 | 4 | 50 | 2.562 | 2.755 | 2.415 | **+12.3%** (CI 10.0–14.5) |
| 25 | 4 | 80 | 4.396 | 4.622 | 4.196 | **+9.2%** (CI 6.8–11.2) |
| 25 | 8 | 50 | 5.433 | 6.042 | 5.317 | **+12.0%** (CI 10.6–13.7) |
| 25 | 8 | 80 | 3.977 | 3.869 | 3.871 | tied (not replicated) |

Independent re-implementation (`sched_repro.c`, different thread/queue design)
reproduced the same direction and magnitude (alt vs cold: +12.1%, +9.6%, +6.4%,
+1.2%). Absolute ms differ between implementations, as expected.

**Why cold-first loses**: the cold reads do finish slightly sooner, but the
hot-copy work then piles up in a tail. Example timeline (K=25 W=4 hot=80%):
cold-first starts its first hot read at 1.93 ms and has a 1.41 ms hot-only tail;
alternating starts hot at 0.02 ms and has a 0.89 ms tail → whole batch finishes
earlier. The objective is *overlap of cold reads with hot copies + short tail*,
not minimum cold-finish time.

### 2b. Pure shared-bandwidth counterexample (`sched_counterexample.py`)

Even with NO hot-copy cost, cold-first is not makespan-optimal for a shared
channel. W=2, B(1)=1, B(2)=1.5 (equal sharing), work [3,3,2,2,2]:
- cold-first (longest-first) makespan = **8.667**
- order (3,2,2,3,2) makespan = **8.000** = lower bound Σwork/Bmax → optimal.
Verified by exhaustive enumeration (independent continuous-time simulator; the
original used a discrete stepped simulator — both agree).

## 3. Other findings

- **Indiscriminate cold-concurrency caps are usually worse.** All-cold K=8 W=8:
  whole-expert 3.843 ms; cap cold concurrency 4 → 4.673 ms, cap2 → 6.994,
  cap1 → 12.156 ms. Big-file "4 workers saturate" does NOT transfer to small
  random expert batches; app threads ≠ UFS hardware queue depth.
- **Request granularity is an independent lever.** All-cold K=8 W=8: 512 KiB pread
  chunks → 3.457 ms (**~10% faster**, CI 8.8–11.9%) vs whole-expert; 128 KiB → worse.
  This is per-expert sequential chunking, not application-level preemption.
- **Reserve-slot policies (hold W-1/W-2 cold slots while hot remains) also win some,**
  lose others — not a universal rule.
- **Probe cost can erase the gain.** Per-expert mmap/mincore/munmap ≈ 0.17–0.20 ms
  (K=8) and 0.50–0.56 ms (K=25) in this harness. Added to cold-first it often
  wipes out its small advantage over a no-probe random queue. (Arithmetic
  accounting, not a direct no-probe end-to-end run.)

## 4. Known methodology flaws in the earlier tools (do not trust their old numbers)

- `expert_sched_bench.c`: prewarms the first P% of ids and dispatches those same
  ids in original order for "naive" → naive is a hot-prefix, not a fair random
  baseline. Timed region includes thread create / per-read alloc / open. Does not
  verify `read_bytes` or re-probe each strategy's true starting residency; policy
  order is fixed, not randomized. Its `wsrpt` sorts work/urgency DESC, but Smith's
  rule is ASC and is non-preemptive — so it is NOT weighted-SRPT.
- `expert_bubble_bench.c`: static LPT treats hot experts as zero load, so hot jobs
  pile onto one worker; that is an implementation artifact, not evidence LPT is
  inherently bad.
- Stride typo history: some earlier text said 1,353,680; the correct value is
  **1,351,680** (manifest and all commands).

The corrected harness (`sched_repro.c`) fixes these: persistent barrier-started
workers, pre-touched reusable buffers, per-policy FADV_DONTNEED + identical
prewarm membership, per-expert mincore probe, `read_bytes`==missing validation,
randomized policy order, independent hot-membership vs queue order, per-job
timestamps. It is the tool to trust for scheduling claims here.

## 5. How to test

### 5.1 What must hold for a number to be trustworthy
1. **Identical cache state before each policy**: FADV_DONTNEED whole file →
   re-prewarm the *same* hot membership → run. Never compare a warm 2nd run vs a
   cold 1st run.
2. **Honest cold check**: per-expert `mincore` (hot≈full resident, cold≈absent;
   allow small readahead tolerance) + `/proc/self/io` `read_bytes` delta ==
   missing bytes. Abort if a read is short.
3. **Hot membership independent of queue order** (avoid accidental hot-prefix).
4. **Same expert set across policies**; only order/assignment differs.
5. **Many seeds × reps, mean**; confirm on a *new* seed set.
6. **Randomize policy order** and interleave A/B to spread thermal/frequency drift.
7. **Report probe + sort cost separately**; add it back to judge net benefit.
8. **Match objective to metric**: for total time use makespan (max completion).

### 5.2 Reproduce (device, Termux, no root)
```sh
clang -O2 -pthread sched_repro.c -o sched_repro
# FILE K W PCT SEED REPS ; one JSON line per (rep, policy)
./sched_repro layer_000.bin 8 4 50 42 4
python3 sched_repro_analyze.py layer_000.bin        # aggregates the table above
python3 sched_counterexample.py                     # the shared-bw counterexample
```

### 5.3 Verify the mechanism
- All orders ~equal at 0% prewarm (uniform work, bandwidth-bound).
- cold-first gain grows only with a cold/hot mix; alternating wins by overlapping
  hot copies with cold reads (check per-job timestamps: first-hot-start, tail).
- If order matters hugely even at 0% prewarm → suspect measurement error.

## 6. Bottom line

Cold-first is a candidate heuristic, not an optimum. On this phone, sensible
cold/hot **overlap** beats pure cold-first; blindly lowering cold concurrency is
usually worse; **request granularity** is an independent win; the `mincore` probe
cost can cancel the ordering benefit. Whether any of this improves token/s must
still be validated on the real slot/rpcmem read path with compute overlap — this
document only covers the read-dispatch microbenchmark.
