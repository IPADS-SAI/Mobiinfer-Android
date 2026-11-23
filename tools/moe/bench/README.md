# MoE expert-read I/O benchmark

Microbenchmarks for measuring how MoE expert weights are read from UFS/flash,
to answer the questions that decide a read strategy:

1. **What is the real UFS read bandwidth ceiling?** → `ufs_read_bench.c`
2. **Does it matter WHICH experts a decode step reads (sequential vs random),
   and how many workers?** → `expert_read_bench.c`
3. **How does page-cache hit ratio change things (0/30/50/80/100% prewarmed),
   buffered vs O_DIRECT?** → `expert_read_prewarm.c`
4. **Does coalescing consecutive experts into one big pread help?** →
   `expert_read_bench.c` with `ER_MERGE_SPAN=N`
5. **How much is per-read fixed overhead, and how expensive is a mincore
   residency probe?** → `latency_probe.c`
6. **Given partial page-cache residency, how should we order/assign reads to
   minimize total time?** → `sched_repro.c` (+ `sched_counterexample.py`);
   see **[SCHEDULING.md](SCHEDULING.md)** for the corrected model, results and
   methodology. (The older `expert_bubble_bench.c` / `expert_sched_bench.c` have
   known methodology flaws documented there — prefer `sched_repro.c`.)

All are self-contained C (POSIX + Linux `/proc/self/io`), build with clang/gcc,
run on the Android device with Termux (no root needed for reading your own files).

## Build & run (on device)

```sh
clang -O2 -pthread ufs_read_bench.c      -o ufs_read_bench
clang -O2 -pthread expert_read_bench.c   -o expert_read_bench
clang -O2 -pthread expert_read_prewarm.c -o expert_read_prewarm
clang -O2 -pthread latency_probe.c       -o latency_probe

# 1) file-level bandwidth: FILE REPS CHUNK_KIB FADV_RANDOM
./ufs_read_bench layer_000.bin 3 1536 0

# 2a) per-expert, seq vs rand, with WORKERS:
#     FILE STRIDE K MODE TURNS DIRECT SEED WORKERS [REPS]
./expert_read_bench layer_000.bin 1351680 25 seq  1 0 1 4 5   # 25 contiguous, buffered, 4 workers
./expert_read_bench layer_000.bin 1351680 25 rand 1 0 42 4 5  # 25 random, buffered, 4 workers

# 2b) merge consecutive experts into one pread of MERGE_SPAN*stride bytes:
ER_MERGE_SPAN=5 ./expert_read_bench layer_000.bin 1351680 25 seq 1 1 1 4 5

# 3) prewarm ratio: FILE STRIDE K MODE PREWARM_PCT WORKERS SEED [REPS]
./expert_read_prewarm layer_000.bin 1351680 25 seq  50 4 1 3   # 50% prewarmed, buffered
ER_DIRECT=1 ./expert_read_prewarm layer_000.bin 1351680 25 rand 80 4 42 3  # 80% prewarmed, O_DIRECT

# 5) per-read cost breakdown: FILE STRIDE [N] [SCOPE=expert|file]
./latency_probe layer_000.bin 1351680 50 expert   # mincore one expert range
./latency_probe layer_000.bin 1351680 10 file     # mincore the whole file
```

Every run is honestly cold first: `FADV_DONTNEED` the whole file, `mincore` probe,
`/proc/self/io` `read_bytes` delta reported so you can prove real device reads.
`expert_read_prewarm` deliberately does NOT drop cache after prewarming — it reads
a mixture of resident and cold experts.

- `STRIDE` = bytes per expert (gate|up|down contiguous). Q3_K MiniCPM-V 4.7 =
  1,353,680 B (1.29 MiB). Qwen3-30B Q4_K tiled = different value; read it from
  the layer manifest.
- `MODE` = `seq` (experts 0..K-1) or `rand` (K distinct random experts).
- `TURNS` = how many times to read the K-expert set within one timed window.
- `DIRECT` = 0 buffered / 1 O_DIRECT (aligned ordinary buffers). `ER_DIRECT=1`
  does the same for `expert_read_prewarm`.
- `WORKERS` = reader threads pulling from a shared queue (W<=K).
- `ER_MERGE_SPAN=N` = read N consecutive experts with ONE pread of N*stride
  bytes (seq only). N=1 (default) is the per-expert baseline.


## Measured — vivo V2546A / SM8850 / UFS 4.0

File: `minicpm47/layer_000.bin` (330 MiB, Q3_K, 256 experts, stride 1,353,680 B).
Cold every run (mincore resident == 0). 1 turn = 25 experts read concurrently.

### File-level UFS bandwidth (`ufs_read_bench`)

| mode | 1w | 2w | 4w | 8w |
|---|---:|---:|---:|---:|
| buffered | 2.30 | 3.69 | 4.08 | 4.07 |
| O_DIRECT | 2.01 | 3.49 | 4.17 | 4.15 |

→ UFS 4.0 sequential ceiling ≈ **4.1–4.2 GiB/s**, saturated at **4 workers**.
8 workers add nothing. Block ≥512 KiB needed; 256 KiB single-thread collapses to
~1.5 GiB/s.

### Per-expert: 25 contiguous vs 25 random (one decode-step shape)

`per_expert_ms` = mean wall per expert. Random = mean over seeds 1/7/42/123/999.

| selection | mode | wall_ms (25) | per_expert_ms |
|---|---|---:|---:|
| **seq** 25 | buffered | 10.88 | 0.435 |
| **seq** 25 | O_DIRECT | 8.27 | 0.331 |
| **rand** 25 | buffered | ~16.6 | ~0.66–0.71 |
| **rand** 25 | O_DIRECT | ~8.6 | ~0.34–0.36 |

**Key findings:**
- **Random 25 is much slower than sequential 25 for buffered reads** (~16.6 ms
  vs ~10.9 ms, ~1.5×): page cache / readahead reward contiguity; random jumps
  defeat it and inflate physical reads.
- **O_DIRECT erases most of the seq/rand gap** (~8.3 vs ~8.6 ms): it bypasses
  page cache, so both are pure UFS, and both sit near the physical limit. For a
  genuinely random, uncached decode working set, O_DIRECT is the honest path.
- **O_DIRECT is also fastest for the contiguous case** here (8.27 < 10.88): the
  buffered path pays readahead + cache-fill overhead it can't reuse.

### Worker count (25 experts = 32.2 MiB per "step"; ms / GiB·s⁻¹)

`seq`:

| W | buffered | O_DIRECT |
|---:|---:|---:|
| 1 | 31.9 / 0.99 | 20.2 / 1.55 |
| 2 | 19.5 / 1.62 | 14.3 / 2.20 |
| 4 | 12.4 / 2.54 | 9.15 / 3.44 |
| **8** | 9.54 / 3.30 | **8.41 / 3.74** |
| 16 | 9.78 / 3.22 | 8.58 / 3.67 |

`rand` (mean over seeds; ms / GiB·s⁻¹):

| W | buffered | O_DIRECT |
|---:|---:|---:|
| 1 | ~48.7 / 0.65 | ~26.4 / 1.19 |
| 2 | ~26.9 / 1.17 | ~13.8 / 2.28 |
| 4 | ~17.5 / 1.80 | ~8.6 / 3.66 |
| **8** | ~16.8 / 1.87 | ~8.8 / 3.58 |
| 16 | ~16.4 / 1.92 | ~9.1 / 3.46 |

→ 1 worker uses ~1/32 of the UFS queue and is slowest everywhere. **4 workers**
saturate the queue; 8 adds little; 16 regresses. For random O_DIRECT, 4 workers
already matches sequential. buffered random tops out ~1.9 GiB/s (random I/O wall).

### Page-cache hit ratio (`expert_read_prewarm`, W=4; ms / GiB·s⁻¹)

Rows = prewarmed fraction of the 25 experts. buffered benefits; O_DIRECT ignores it.

`seq`:

| prewarm | buffered | O_DIRECT |
|---:|---:|---:|
| 0% | 12.2 / 2.57 | 9.37 / 3.36 |
| 30% | 10.3 / 3.05 | 9.05 / 3.48 |
| 50% | 9.23 / 3.41 | 9.60 / 3.28 |
| 80% | 7.82 / 4.02 | 9.33 / 3.37 |
| 100% | 7.49 / 4.20 | 8.78 / 3.58 |

`rand` (mean over seeds):

| prewarm | buffered | O_DIRECT |
|---:|---:|---:|
| 0% | 18.7 / 1.68 | 9.03 / 3.49 |
| 30% | 13.6 / 2.32 | 8.73 / 3.60 |
| 50% | 12.4 / 2.55 | 8.89 / 3.54 |
| 80% | 9.44 / 3.34 | 9.11 / 3.46 |
| 100% | 6.74 / 4.67 | 9.10 / 3.46 |

→ **O_DIRECT is flat (~3.5 GiB/s) regardless of prewarm** — it bypasses page
cache. buffered starts slower (cold) but overtakes O_DIRECT once enough experts
are resident (seq ~50–80%, rand ~80–100%), and at 100% reaches 4.2–4.7 GiB/s by
reading from DDR instead of UFS. Choose by expected hit ratio, not by dogma.

### Merge consecutive experts (`ER_MERGE_SPAN`, seq 25)

Reading N consecutive experts as one big pread of N*stride bytes (into scratch,
then scattered) vs one pread per expert. Full span × worker wall_ms matrix
(buffered / O_DIRECT):

| span \ W | 1 buf | 1 dir | 2 buf | 2 dir | 4 buf | 4 dir | 8 buf | 8 dir |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33.7 | 21.0 | 19.9 | 13.7 | 15.1 | 9.3 | 11.0 | **8.6** |
| 2 | — | — | 16.4 | 11.7 | **11.2** | 9.1 | 10.1 | 9.6 |
| 5 | 30.5 | 18.6 | 16.3 | **11.5** | 15.5 | 9.9 | 12.6 | 10.1 |
| 10 | — | — | 19.3 | 13.9 | 13.2 | 10.4 | 13.1 | 10.7 |
| 25 | 30.8 | 17.1 | 30.1 | 16.2 | 26.4 | 16.6 | 30.2 | 16.7 |

→ **span=2 is the only reliably helpful merge**: buffered W=4 15.1→11.2 ms (-26%);
O_DIRECT barely moves. Bigger spans lose more parallelism than they save syscalls;
**span=25 (whole run) is worst everywhere** because only 1 job remains. The rule:
`span` must stay ≪ K/W, otherwise the reader pool starves. Random experts are not
contiguous in the file, so merging does not apply to `rand` (the tool falls back
to per-expert). Combined with the ~15 µs fixed overhead (see below), syscall
savings are tiny next to a ~1.3 ms read — merging only helps at small spans where
it keeps workers busy.

### Per-read cost breakdown (`latency_probe`, MiniCPM Q3 1.29 MiB/expert)


| component | p50 | p95 | note |
|---|---:|---:|---|
| fixed startup (open+malloc+close, no read) | 12–20 µs | ~21 µs | ~1% of a read |
| cold per-expert read (1.29 MiB) | 1.25–1.84 ms | ~1.4–2.2 ms | varies with cache/thermal |
| mincore probe, ONE expert range (384 pg) | ~12 µs | ~17 µs | ~1% of a read — cheap |
| mincore probe, WHOLE 330 MiB file (84480 pg) | ~2.2–4.0 ms | ~5.5 ms | ~2–3× a single read — do NOT do per-read |

→ Probing one expert's residency (~12 µs) is negligible versus reading it
(~1.3 ms), so a per-read `mincore` gate is practical. Probing the whole file per
read costs more than the read itself — only scan the experts you will touch.
Fixed open/malloc/close overhead (~15 µs) is also negligible, which is why
syscall-reduction from merging buys almost nothing.

## Caveats / traps

- These are **single-layer, 25-expert, cold** measurements. Not full-model
  decode tok/s. The real interleaving of read/compute across layers differs.
- K concurrent threads model "all 25 ready at once". A real runtime may use a
  4-thread reader pool; that changes concurrency but not the seq/rand physics.
- Do not divide wall time by LPDDR peak and call it utilization. UFS is the wall.
- `read_bytes` may include readahead; compare against `mincore` and requested
  bytes. Verify coldness every run.
- Real decode reads experts across ALL layers (each its own `layer_XXX.bin`);
  here everything is within one layer file. Cross-file random access will be no
  better than the in-file random number, and likely worse (more seeks).

## Profile-Driven Scheduling Search

See [SCHED_SEARCH.md](SCHED_SEARCH.md) for experimental snapshot validation,
partial-residency replay, finite-candidate calibration, and held-out evaluation.
These tools do not modify the production expert loader. Store generated
profiles, configurations, and benchmark logs outside the repository.

## Where this came from

Distilled from the on-device experiments under
`~/moe-speculative/hexagon-results/minicpm-belle-full-20261020/` (bandwidth,
direct-vs-buffered, exported DMA-BUF) and reproduced on vivo V2546A. See the
`moe-expert-io` agent skill for the full read-path / page-cache model.
