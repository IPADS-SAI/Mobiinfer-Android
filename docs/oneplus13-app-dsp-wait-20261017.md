# OnePlus 13 App DSP wait experiments (2026-10-17)

Target: reproduce the historical 512-token Chinese prompt / 128-token output on the original PJZ110 / SM8750. User subsequently requested decode as close as possible to 15 tok/s. This target is not yet achieved. Historical supported result is approximately 73.15 tok/s prefill and 14.06 tok/s decode; no equivalent 15 tok/s run has been verified.

## Evidence location

Full commands, per-run Markdown, JSON results, library/APK hashes, original logs and scripts:

`/home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/`

Historical prompt SHA256: `32e4bf4a46971fb43552ab91ca17b7de204495d0bf7784baa5a0da513a52cb52`.

Old CLI/core backup: `/home/yydh/moe-speculative/hexagon-results/old-app-cli-backup-20261017/old-app-cli-backup.tgz`. Preserve it independently of new builds.

## Results so far

All rows use 512 input / 128 output, temp0/seed42, dual HTP, bank256, batch/ubatch128, IO4, draft3, no expert prewarm. Sequential runs do not control OS page cache. Different binaries are explicitly separated.

| Binary and wait mode | Cache + prefetch | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Old core CLI, blocking | 48+4 | 42.63 / 45.04 | 9.79 / 9.73 |
| Old core CLI, polling | 48+4 | 51.62 | 13.15 |
| Old core CLI, polling | 52+0 | 53.89 | 12.11 |
| New App, blocking, original run | 48+4 | 40.15 | 10.28 |
| New App, dedicated nonblocking read, initial run | 48+4 | 33.45 | 12.04 |
| Same App, blocking repeat | 48+4 | 42.07 | 10.48 |
| Same App, nonblocking repeat1 | 48+4 | 48.54 | 13.03 |
| Same App, nonblocking repeat2 | 48+4 | 48.24 | 13.02 |
| Same App, nonblocking repeat3 | 48+4 | 48.78 | 12.98 |

The three App nonblocking repeats have identical response strings, accepted76/150 draft tokens and zero cached prompt tokens. They establish repeatability for this prompt and environment, not general quality, long-run thermal stability or cold-cache throughput. Old CLI accepted75/150 and generated different wording, so full numerical generation equivalence is not established.

## Changes and interpretation

Android previously used timed `dspqueue_read(..., 0)` for polling. The driver can log empty polls as timeouts. The driver wrapper now uses its dynamically loaded `dspqueue_read_noblock` function when available, retaining the timed-read fallback for older drivers. This mirrors an existing Windows path. A timeout alone does not prove a stalled DSP. Captured logs must be checked for dropped records before claiming absence of timeout messages.

App diagnostic properties, applied before backend initialization:

```sh
adb -s 192.168.61.132:39565 shell setprop debug.mobiinfer.htp_poll 1
adb -s 192.168.61.132:39565 shell setprop debug.mobiinfer.htp_profile 0
adb -s 192.168.61.132:39565 shell am force-stop com.mobiinfer.demo
```

Missing/0 polling property retains the blocking default. Property1 is experimental. Detailed profile property1 routes HTP records into App `runtime.log`; it perturbs throughput and must be measured separately. Direct queue-read and target-batch timers now have a separate instrumented measurement; see below.

Optional local model-directory `runtime.json` supports `max_tokens`, `label`, `prefetch` and `io_threads` for experiments. Prefetch0 returns its slots to the cache without increasing the budget's total slots. Normal defaults remain cache48+prefetch4 at8GB and IO4. Remove the temporary file and restart after experiments. Validate actual initialization logs rather than assuming requested overrides took effect.

## Prefill and memory caveats

Historical tok128 chunks reported626/573/664ms expert read wall time. Current App chunks often require1.1–3sec for similar logical bytes. Same-App prefill variability closely tracks read-wall variability. OS page-cache, storage and scheduling effects remain unresolved; counters are not physical UFS bandwidth. `pread_wall` overlaps expert-management time and these must not be added.

Repeat3 kernel RSS high-water:985520KiB (about962.42MiB). Its end-of-run61 unique DMA-BUFs total7814733824bytes (about7.28GiB). DMA is a snapshot, not a peak; CPU/DMA mappings may overlap and must not be summed as independent physical memory. PSS and Swap are in the per-run memory JSON. Missing earlier peak samples remain explicitly missing.

Detailed profile1 shows aggregate DSP op service3928.477ms during prefill and5951.543ms during decode. Decode includes MoE matmul1455.731ms, GDN483.440ms and target+draft vocabulary projection798.240ms. OPBATCH records are excluded to avoid double counting. These session sums are not the end-to-end critical path and cannot be subtracted from wall time to infer pure host wait.

## Remaining work

Reproduce yesterday's cache52/prefetch0 configuration in App; verify approximately70tok/s prefill and approach15tok/s decode. Collect direct target/queue timers, separate target and draft costs, investigate read-path variability, repeat across requests with memory evidence, and select production defaults only after validation. No commit or push has been performed for these changes.

## Historical App configuration and direct timers

The current App now accepts per-run `prefetch` and `io_threads` overrides with validation; total decode slots remain budget-derived. Initial App cache52/prefetch0/IO4/nonblocking run measured36.02prefill and12.73decode tok/s, accepted76/150. Installed APK SHA256 is recorded per run and matched the local artifact. This does not reproduce the historical73.15/14.06 result.

A separate same-configuration profile measured inclusive target decode10144.032ms (51batches/201evaluated tokens including rejected proposals), draft generation1089.313ms and queue API duration4050.154ms. Queue time includes target and draft, overlaps DSP execution and must not be added to target time. Per-poll clock probes recorded39572138empty reads and can significantly perturb the run. Ordinary throughput remains the non-profile result. Details, memory and raw logs: `app-historical-poll-pf0-profile1.md` and its direct-summary JSON in the evidence directory.

IO8-only App follow-up (cache52/prefetch0) measured38.54prefill/12.02decode; no decode benefit over IO4 demonstrated. Layer000 expert bytes match between App and Termux; other39layers not yet checked in this comparison.

## Vision placement and process storage accounting follow-up

A diagnostic App run with vision_htp=false (vision still loaded on CPU), cache48+prefetch4, IO4 and nonblocking reads measured38.04prefill/12.74decode tok/s. RSS high-water1951212KiB; final DMA-BUF snapshot6693203968bytes (6.23GiB), versus7.28GiB with HTP vision. Reducing HTP buffers did not demonstrate improved text throughput in this single sequential comparison.

The normal HTP-vision follow-up `app-poll-pf4-io-account1` measured38.20prefill/12.65decode tok/s, accepted76/150,512new input tokens and128output tokens. RSS high-water988372KiB; final DMA snapshot7814733824bytes. New runner snapshots /proc/PID/io before submit and after observing completion: rchar49768304378bytes (46.35GiB), read_bytes40361312256bytes (37.59GiB). This interval includes model initialization after submit and all App threads; it is not a phase-specific or expert-only measurement. The counters support substantial storage traffic, but cannot establish an exact cache hit rate or explain decode time alone.

Page-cache prewarming may move storage latency into loading rather than remove total latency. The packed expert files total15GiB and active App DMA snapshots exceed7GiB, so retaining all experts in cache is not assured on this24GB device. fadvise(WILLNEED) is advisory; explicit pread confirms a read but does not pin pages. Any prewarming experiment must record its elapsed time, per-stage storage counters, throughput and memory; do not treat warm-cache microbenchmarks as model throughput. Detailed standalone evidence is in the per-run Markdown/JSON files.

## Approximate phase storage measurements

`app-poll-pf4-io-phases1`: same normal HTP-vision App, nonblocking,48+4slots/IO4. Prefill37.85/decode12.56tok/s, accepted76/150. External log-triggered boundaries approximate request-to-first-token and first-token-to-completion, not exact prefill/decode boundaries. ADB delivery/snapshot latency and all App threads are included. No engine instrumentation was enabled.

request_to_first_token: 13.513 s, logical rchar 28.453 GiB, kernel storage read_bytes 27.720 GiB.

first_token_to_complete: 10.229 s, logical rchar 13.563 GiB, kernel storage read_bytes 5.803 GiB.

These measurements show substantial storage traffic during prompt processing. They do not prove that page-cache warming can eliminate it: cache capacity/reclaim and repeated expert accesses still need controlled investigation.

## Full expert page-cache prewarming experiment

Old verified CLI, original OnePlus13,512/128,cache52/prefetch0,IO4,draft3,poll1. App stopped. First requested file-specific DONTNEED on all40 App expert files; system Cached fell8626796→3006824KiB. CLI then measured46.79prefill/12.25decode,accepted75/150; full-process peak RSS778256KiB (GNU time). This does not reproduce~70prefill.

Then explicitly read all40 CLI expert files (15GiB) before another identical CLI launch. Prewarming cost2.822s and incurred7466102784bytes of storage reads; Cached reached16442556KiB before model allocation. CLI measured51.29prefill/12.50decode,accepted75/150,peak RSS778232KiB. Prefill saved approximately0.960s relative to the immediately preceding run, less than the2.822s prewarming cost. No net latency benefit demonstrated in this single sequential comparison. Pages were not pinned and model allocation could reclaim them. Raw commands, per-file data, timings and memory: old-cli-after-app-pages-drop1.md,old-cli-all-experts-prewarm1.md and corresponding files in the evidence directory. No default prewarming change applied.

## Read sorting and CPU frequency observations

Correction: the zerocopy percentage in MoE logs is an opportunity statistic from accumulate_runs, not measured copies. Sorting changes expert adjacency versus slot adjacency; it does not enable scratch scatter in the active install_packed_wave path. Packed reads still use per-expert readv_span directly into tensor slots.

Old verified CLI explicit LLAMA_MOE_PREFILL_SORT_READS=1,512/128,cache52/prefetch0,bank256,b/ub128,draft3,poll1: IO4 measured52.47prefill/13.09decode,IO8 measured56.57prefill/12.75decode; both accepted75/150. Full-process maximum RSS778268/778148KiB, respectively. Sequential runs after prior full-file warming are not a randomized sorting A/B. Device state/raw evidence saved per run.

During this series, policy6 scaling_max_freq was1689600kHz versus cpuinfo_max_freq4320000kHz; battery40.2–40.3C, thermal status1. Battery saver was OFF when queried after IO8. CPU frequency capping is a candidate environmental contributor, not a proven cause of the historical performance gap. No thermal/frequency limits were overridden.

## IO8 sorting-off follow-up and existing performance setting

Old CLI sort0/IO8 measured55.19prefill/12.32decode,accepted75/150,maximum process RSS778392KiB (GNU time). Same512/128,cache52/prefetch0,draft3 configuration. DMA was not sampled for this run. Sort1/IO8 was56.57/12.75; single sequential runs with uncontrolled cache/temperature do not establish a sorting improvement.

Read-only settings inspection found high_performance_mode_on=1 (also shutdown key1), while policy6 scaling_max_freq remained1689600kHz. Thus simply recommending enabling the already-set performance flag does not resolve this observation. A settings flag alone does not establish vendor policy is honored; the owner of the cap remains unidentified. No frequency or thermal policy was modified.

## File-cache microbenchmark measurement contract

The earlier expert-read-probe experiment measures pread from a384MiB packed file into ordinary malloc buffers using4threads and1.5MiB expert records, not pure RAM memcpy and not DMA writes. File-specific DONTNEED before cold runs is advisory, but the observed per-process read_bytes delta was384MiB for every cold run and0 for every immediate warm run. Raw cold throughput2887.55–3494.45MiB/s (2.82–3.41GiB/s), warm7523.76–13307.99MiB/s (7.35–13.00GiB/s). These are short-file measurements and cannot be substituted for full-model storage/DMA bandwidth. Probe code and8raw results: expert-read-probe.c / expert-read-path-ab.jsonl in the evidence directory.

## App IO2 versus adjacent IO4 follow-up

Same App/APK, historical512input/128output,cache48+prefetch4,draft3,HTPvision,nonblocking,no detailed profiling: IO2 measured30.64prefill/10.95decode; adjacent IO4 measured34.36prefill/11.83decode. Both accepted76/150 and generated identical response SHA256. Current environment did not retain earlier~13decode; do not claim stable performance across thermal/cache conditions. Per-run RSS peak and sampled DMA are stored with each result; DMA is not peak. Full summary io2-io4-followup-summary.json. No IO2 default selected.

## CPU writes to rpcmem microbenchmark

Original OnePlus13 PJZ110/SM8750. Single384MiB Termux packed expert file,4threads,per-thread1.5MiB allocation reused across256permuted expert reads. rpcmem_alloc(25,1,size), matching backend SYSTEM heap/cached flags, loaded from libcdsprpc.so. CPU mapping only: no DSP mapping/session, compute, coherency handoff or queue synchronization. This is not full expert-slot performance. App process26410 was present before this probe series; no exclusive idle guarantee.

Cold uses file-specific DONTNEED; observed kernel storage read_bytes384MiB for each cold run,0 for each warm run. Order malloc,rpcmem,rpcmem,malloc, each cold then warm. Thread allocation/free and first touch are inside measured interval.

| Destination | File state | GiB/s range (2runs) |
|---|---|---:|
| malloc | cold | 3.22–3.30 |
| malloc | warm | 10.38–13.85 |
| rpcmem | cold | 3.33–3.49 |
| rpcmem | warm | 10.79–11.53 |

No major cold-read regression from writing rpcmem demonstrated here. Warm-read ranges overlap; cannot assign a reliable DMA penalty with2runs and uncontrolled interference. Small reused buffers do not reproduce large slot access/cache behavior. Pure memcpy still unmeasured. Per-process RSS peaks are in each raw record; DMA peak was not sampled.

Source expert-read-rpcmem-probe.c; binary expert-read-rpcmem-probe. Reproduce using raw command fields in expert-read-rpcmem-ab-system-first-raw.json. An initial vendor-first LD_LIBRARY_PATH failed resolving a system binder symbol; raw failure preserved in expert-read-rpcmem-ab-raw.json. System-first library order succeeded. No weight-file writes or production source changes.

## Full expert identity verification

SHA256 for all40App packed files and all40Termux packed files now matches layer by layer (exit0,30.394s). This closes the previous limitation that only layer000 was checked. Raw80hash/path lines and summary are all-app-cli-expert-sha256.txt / all-app-cli-expert-sha256-summary.json in the evidence directory. The hash operation itself reads30GiB and changes file-cache state; it is not a throughput benchmark.

A later read-only state sample showed battery39.6C,policy0 scaling_max_freq3532800kHz andpolicy6 2438400kHz, versus earlier1785600/1689600kHz. Frequency caps are dynamic; the responsible policy and their causal contribution remain unproven. App process26410 was present and was not stopped during this file check.

## App repeat after frequency-cap recovery

Same APK,512/128,48+4slots,IO4,draft3,HTPvision,nonblocking: app-poll-pf4-unlimited-start1 measured44.95prefill/13.21decode,accepted76/150,identical response hash to earlier App runs. Frequency sampling began just before request submission and continued until benchmark observation. 28 samples; big-core allowed maximum ranged 2438400–4320000 kHz. It was not continuously unrestricted. Sampling adds adb/dumpsys activity, so comparison with older unsampled runs is not fully controlled. Earlier full-file SHA256 checks changed cache state.

RSS peak 1092740KiB; DMA final snapshot 7814733824bytes (not peak). Response SHA256 d1c1e974b018acb5a3aa47a8eb5824d47bcbf0b5e9bfaa2c949eb19e63d1e654.
CPU allowed maxima are sampled environmental evidence, not proof of causal speedup; page-cache state also changed. Historical~70prefill is still unreproduced.

## Checkpoint limit alignment

Original OnePlus13 PJZ110/SM8750, verified old core. Historical512input/128output,temp0/seed42,cache52/prefetch0,IO4,bank256,b/ub128,draft3,warm0,poll1,LLAMA_MOE_PREFILL_SORT_READS=1. App stopped, exclusive device lock. New script cli-checkpoint1-reproduction.sh differs from historical warm-length-run.sh only by --ctx-checkpoints1 (argument and value separated in actual script). Remote script repro-checkpoint1.sh; GNU time -v wraps execution.

Prefill47.39tok/s,decode13.57tok/s,accepted75/150. Final maximum process RSS777980KiB. Raw stdout/stderr, sampled memory and device state share run label. DMA sample is not peak. Full reproduction script retained.

Both this run (limit1) and earlier old-cli-sort1 (limit32) actually created only one checkpoint, at token384,size66.571MiB. Thus configured limit disparity did not translate into a larger checkpoint count for this512-token case. Different cache/temperature/order prevent attributing throughput differences to the limit. Historical~70prefill still unreproduced.

## Random file-advice A/B/A sequence

Verified old CLI,512/128,cache52/prefetch0,IO4,sort1,checkpoint1,draft3,poll1. LLAMA_MOE_FADVISE_RANDOM sequence0/1/0 produced:

- old-cli-checkpoint1: prefill47.39,decode13.57tok/s; GNU time filesystem inputs45720144,maximum RSS777980KiB.
- old-cli-random-advice1: prefill56.83,decode13.52tok/s; GNU time filesystem inputs27085416,maximum RSS778388KiB.
- old-cli-random-advice0-repeat: prefill63.36,decode13.76tok/s; GNU time filesystem inputs21566480,maximum RSS778604KiB.

All accepted75/150 and exited0. Sequential cache and thermal state were not controlled, so do not attribute improvement solely to fadvise. No production default changed. Raw evidence and full command in each run Markdown; random-advice-aba-summary.json preserves raw counters.

## Original historical CLI parameter repeats after cooldown

Reverted checkpoint default32,sort0,random advice0;old verified CLI,512/128,cache52/prefetch0,IO4,bank256,b/ub128,draft3,warm0,poll1. Two repeats yielded65.17/13.87 and63.53/13.97prefill/decode tok/s;accepted75/150 each. Peak RSS778760/778672KiB from GNU time. Battery35.9/35.6C;big-core allowed-max snapshots4.0896/3.072GHz,not full-run clocks. No explicit prewarming between these runs;cache history remains uncontrolled. This restores~14decode on CLI but does not reproduce~70prefill. Per-run command,raw evidence and memory scope: old-cli-original-repeat-final1.md/final2.md in evidence directory.

## App historical configuration after cooler CLI repeats

App52cache/prefetch0/IO4,HTPvision,draft3,nonblocking; same installed APK verified per run. Prefill32.9962/decode11.9990tok/s,accepted76/150. Big-core allowed-max samples ranged2438400–4320000kHz; sampling is not continuous frequency tracing. CLI/App use independent file copies and cache history differs; this run is not a controlled old-core versus App comparison. No full-file App prewarming applied.
Results retain a substantial prefill gap versus CLI63–65tok/s, and do not establish robust App>=12 under all environmental conditions. Raw timing,storage deltas,frequency samples,APK hash and kernel RSS high-water/DMA snapshot saved per run.

## App full-file prewarming control

Treatment: explicitly read all40 App packed expert files(15GiB) into ordinary CPU buffers before launching App. Cost3.368139s;kernel storage reads9866219520bytes. Pages not pinned; App model allocation can reclaim them. No file changes. Raw prewarming app-all-experts-prewarm1.jsonl and summary.

Same App52cache/prefetch0/IO4,draft3,HTPvision,nonblocking,512new input/128output: prefill41.15793tok/s(12439.887ms),decode12.64652tok/s(10121.362ms),accepted76/150. Previous no-explicit-warming run33.00/12.00. Prefill reduced3077.033ms,less than3368.139ms warming cost. Including decode, total request savings exceed warming cost slightly, but uncontrolled sequential environment prevents a net benefit claim.

External marker-triggered request-to-first-token storage reads25282437120bytes(23.55GiB),previous34569236480bytes(32.19GiB). First-token-to-completion5299298304bytes(4.94GiB). These are process-wide delayed snapshots,not exact expert-only phase counters. Significant reads persist after prewarming.

RSS high-water986780KiB;final DMA snapshot7814733824bytes,not peak. Per-run APK hash,response,frequency samples and raw results retained. No production warming default applied.

## CLI with current App core

Current APK extracted libraries were transferred with rsync to an isolated experiments/current-app-core-20261017/lib directory; old backup preserved. SHA256 manifest current-app-cli-lib-sha256.json includes APK and21libraries. CLI script cli-current-app-core-reproduction.sh puts current core directory before old fallback library paths. /proc/PID/maps confirms current libllama.so and libggml-hexagon.so; old libllama-cli-impl.so remains the CLI frontend. This is a mixed frontend/core compatibility experiment,not a fully rebuilt same-version CLI. Exit0 shows completion,not exhaustive ABI correctness.

Original OnePlus13 SM8750,historical512input/128output,temp0/seed42,52cache/no prefetch,IO4,bank256,b/ub128,draft3,warm0,poll1,sort1,random advice0,checkpoint1. App stopped,exclusive lockrun device. Raw command script,stdout/stderr,device state and maps saved.

Prefill32.81tok/s,decode12.44tok/s,accepted75/150. Full-process RSS peak777920KiB from GNU time;DMA snapshot during inference is not peak. Current-core CLI does not automatically preserve old CLI63–65prefill in this sequence. Cache/thermal/order uncontrolled; this does not establish a core regression or isolate App UI cost. CLI no-vision versus App loaded vision and independent file copies remain differences.

## Immediate old-core control

Original OnePlus13 SM8750,same old CLI frontend,Termux weight files and512prompt/128output,seed42/temp0,cache52/prefetch0,IO4,bank256,b/ub128,draft3,warm0,poll1,sort1,random advice0,checkpoint1. App stopped,exclusive lockrun. Uses old backup core path;actual mappings captured in old-core-after-current1-maps.txt. Command GNU time -v env LLAMA_MOE_FADVISE_RANDOM=0 LLAMA_MOE_PREFILL_SORT_READS=1 BENCH_LENGTH=512 BENCH_PREFETCH=0 BENCH_CACHE=52 BENCH_WARM_ROWS=0 bash ~/csm/moe/experiments/cli-app-equivalence-20261016/repro-checkpoint1.sh via ssh oneplus13-reverse.

Prefill35.04tok/s,decode12.71tok/s,accepted75/150,exit0. Final RSS peak778480KiB from GNU time. GNU time raw filesystem inputs80009264 accounting blocks; substantially above original-repeat-final2's18925320. DMA sample not peak. Both versions now slow: current core32.81/12.44,old35.04/12.71. No controlled cache/thermal/randomized experiment; cannot prove exact version penalty, but this sequence does not support treating earlier63–65CLI versus33App difference as solely core version. Library manifests,raw logs,maps,device state and sampled memory preserved.

## Direct expert-file cache residency

file-cache-residency.c uses read-only mmap and mincore,never dereferences mapped weight pages. It does not intentionally fault/read weight data; allocates only the residency vector. Android may restrict mincore information; root used here. Snapshots are sequential across40files and can race with reads/reclaim. No page pinning,full-cache-residency or kernel-reclaim-cause claim.

Idle: App resident3785871360bytes(3.53GiB),CLI7048519680bytes(6.56GiB),each total15GiB. During the next CLI snapshot,CLI resident6.950447GiB; sample stage/log tail saved separately. A single during-run sample does not demonstrate cache fell after model allocation; observed residency was higher than idle.

CLI experiment old-cli-residency1: old verified core,historical512/128,52cache/prefetch0,IO4,bank256,b/ub128,draft3,warm0,poll1,checkpoint default32,sort0,random advice0;exclusive device lock,App stopped. Command historical warm-length-run.sh under GNU time -v with explicit BENCH_LENGTH512/BENCH_CACHE52/BENCH_PREFETCH0/BENCH_WARM_ROWS0.

```text
0.28.633.725 I slot print_timing: id  0 | task 0 | prompt eval time =   13426.05 ms /   512 tokens (   26.22 ms per token,    38.13 tokens per second)
0.28.633.735 I slot print_timing: id  0 | task 0 |        eval time =   10193.59 ms /   128 tokens (   79.64 ms per token,    12.56 tokens per second)
0.28.633.738 I slot print_timing: id  0 | task 0 | draft acceptance = 0.50000 (   75 accepted /   150 generated)
	Maximum resident set size (kbytes): 778136
```

Raw stdout/stderr,memory files,cache-residency-cli-during1.jsonl and stage.txt retained. Idle raw data cache-residency-idle-app/cli.jsonl; summary JSON. Per-run sampled DMA is not peak. mincore execution adds brief CPU/metadata activity,so this is an instrumented residency run.

## Matched prefill chunk timing attribution

Four logged chunks(128+128+128+124=508tokens),original OnePlus13,historical512input/128output,cache52/no prefetch,IO4,draft3,poll1. Remaining4prompt tokens and final timing bookkeeping not included in this chunk sum. Raw fields are pread_wall and interval_ms; interval-minus-read is a descriptive residual,not isolated DSP compute. pmgmt overlaps read;do not sum them.

| Run | Read wall ms | Chunk interval ms | Residual ms | Logical read MiB |
|---|---:|---:|---:|---:|
| old-cli-original-repeat-final2 | 3759 | 7886.8 | 4127.8 | 29285 |
| old-cli-residency1 | 9051 | 13174.8 | 4123.8 | 29285 |
| old-core-after-current1 | 10223 | 14358.3 | 4135.3 | 29285 |
| current-app-core-cli1 | 11160 | 15346.8 | 4186.8 | 29285 |

Old-core fast original-repeat-final2 and slower residency1 read the same29285MiB logical expert data with same chunk shapes. Read wall rises3759→9051ms(+5292ms),chunk intervals7886.8→13174.8ms(+5288ms);residual4127.8→4123.8ms is nearly unchanged. This localizes observed prefill variation primarily to expert installation/read path in these logged chunks. It does not distinguish storage waits,page-cache effects,CPU copy,thread scheduling or thermal effects within that path. Residency instrumentation differs and frequency/cache uncontrolled. No claim every App/CLI gap has the same cause.

Raw JSON prefill-chunk-fast-slow-comparison.json;source *.stderr by row label. Current-core uses old CLI frontend,current App core;old-core-after-current1 uses old core. sort/checkpoint differences across some rows documented in per-run Markdown;matched old-core fast/slow comparison uses sort0/default32 in both.

## External thread wait sampling

Original OnePlus13 SM8750,verified old CLI/core,historical512/128,52cache/no prefetch,IO4,bank256,b/ub128,draft3,warm0,poll1,sort0,random advice0,checkpoint default32. App stopped,exclusive lockrun. Historical warm-length-run.sh under GNU time -v. Sampling script sample-cli-thread-waits.py retained.

Prefill37.20tok/s,decode12.19tok/s,accepted75/150,exit0;final process RSS peak778280KiB. DMA snapshot not peak. 25external /proc snapshots at roughly0.5s plus collection latency;sampling has overhead and covers partial run,not exact engine phases.

Four active threads4435–4438 each accumulated3.61–4.10CPU seconds and0.165–0.214s runqueue wait between first/last samples. Their samples include folio_wait_bit_common(12total across4threads),futex waits,runnable/unreported wchan0,and2__flush_work observations. This is consistent with mixed CPU work and page-related waits;not proof storage-only bottleneck. Thread I/O counters were not collected,so IO-pool identity remains unverified. Wait-channel counts are not duration fractions. Main inference/scheduler thread4439 accumulated8.78CPU seconds (polling can account for CPU work);do not interpret CPU time as useful compute. Idle FastRPC helper wait channels do not prove DSP stalls.

Raw old-cli-thread-waits1-thread-samples.jsonl,summary JSON,stdout/stderr and memory retained. Need per-thread read counters and phase-aligned probes before attributing precise expert-read cost fractions.

## Per-thread read attribution

Original OnePlus13 SM8750,verified old CLI/core,historical512input/128output,52cache/no prefetch,IO4,bank256,b/ub128,draft3,warm0,poll1,sort0,random advice0,checkpoint default32. App stopped;exclusive lockrun;historical warm-length-run.sh under GNU time -v.

Prefill37.23tok/s,decode12.22tok/s,accepted75/150,exit0. Final RSS peak778120KiB. DMA snapshot not peak. Sampling script sample-cli-thread-waits.py uses SAMPLE_LABEL;raw thread snapshots and summary JSON retained.

21valid external snapshots cover15.859s between first/last,partial prefill+decode,not exact phase boundaries. Actual reading workers7073/7074/7075/7077 have logical rchar deltas5.4–6.3GiB and storage read_bytes deltas4.0–4.5GiB each. CPU deltas3.25–3.74s and scheduler runqueue wait0.150–0.178s each. All have folio_wait_bit_common samples. The main thread7035 appears in cumulative-rchar filter but has zero read delta in this interval;do not classify it as an active IO worker.

This confirms active read threads experience both CPU work and page waits. It does not quantify duration fractions for storage versus copy;wchan samples are not duration measurements,CPU time includes kernel work not just memcpy,storage counters can include readahead. Scheduling starvation alone is not supported as the dominant cause in this sampled interval. Root snapshot activity adds diagnostic overhead.

## Stage-specific IO implementation constraints

Current source src/llama-moe-slot.cpp: moe_ssd_io_threads caches LLAMA_MOE_IO_THREADS in a function-local static. Changing env between stages cannot change it. moe_io_pool creates that many persistent threads once;run(count) sets remaining=count and only existing workers decrement it. A future stage-specific count must not exceed pool capacity,otherwise completion can hang.

install_packed_wave receives window_mode and currently uses one global thread count. Shared-bank prefill and per-layer decode windows are distinguished earlier in the callback. DSpark verify batches can contain multiple tokens while still using decode windows;token count>1 is not a valid prefill classifier. Prefetch has a separate caller-owned pool and should retain explicit depth/worker limits. Current App initializes LLAMA_MOE_IO_POOL=1 in llama_jni.cpp.

Potential follow-up: choose prefill/decode concurrency using existing window-mode decision,allocate pool capacity to max validated stage concurrency,keep baseline defaults4/4 and prefetch policy fixed. This is a proposed experiment,not implemented or proven improvement. Earlier sequential IO8 gains were cache/thermal-confounded;must compare4/4 and8/4 on same binary,prompt and recorded state before changing defaults. No production source edit or benchmark in this audit.

## App chunk512 five-request result

See [dedicated experiment](app-chunk512-five-runs-20261017.md). Same-process512/128,new conversation each round: prefill81.05/94.27/94.32/94.28/94.51;decode13.60/13.78/13.65/13.65/13.64tok/s. Batch/ubatch512,bank256,48+4slots,IO4,draft3,nonblocking. This changes chunk size from the historical128 configuration and does not count as reproducing historical parameters.

## Decode cache capacity sweep

See [App cache48/56/64 experiment](app-decode-cache-slots-20261017.md). Same APK,chunk512,prefetch4,IO4,draft3,512/128: cache48decode13.61/13.81;56decode13.46/13.62;64decode12.64/12.66;return48decode13.39tok/s. All7responses identical,accepted81/136. Larger cache reduced logical reads but not demonstrated throughput;DMA increased. Default48+4 retained,temporary overrides removed.
