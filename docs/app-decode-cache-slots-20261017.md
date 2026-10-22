# App decode cache slots48/56/64

Original OnePlus13 PJZ110/SM8750,App debug APK SHA256b8df86d925db1dead4973974b41736b088fd34fb621d89cbfb2daffbdf46d51f. Local benchmark-only cache_slots override added to runtime.json;normal budget-derived defaults unchanged. Installed hash saved per run;actual cache count verified in startup logs. Build successful,install successful.

All runs: batch/ubatch512,bank256,512historical prompt/128output,cache_n0,IO4,prefetch4,DSpark Q8 draft3,temp0/seed42,HTPvision,dualHTP,nonblocking DSP wait,profile0.8GB configured estimate is not a hard limit;explicit experiment overrides can exceed it. All outputs identical across7requests and accepted81/136.

Two consecutive new conversations in the same process per size;restart between48→56→64→48return. No explicit file-cache flush/prewarm or cooldown;automation/sampling gaps exist. Sequence is not randomized,frequency/temperature/cache uncontrolled. Return48single-run check is additional evidence of temporal variation.

| Cache + prefetch | Prefill tok/s | Decode tok/s | DMA end sample GiB | Max process RSS high-water MiB |
|---|---|---|---:|---:|
| 48+4 | 82.88 / 87.89 | 13.61 / 13.81 | 7.803 | 1433.14 |
| 56+4 | 80.65 / 92.04 | 13.46 / 13.62 | 8.262 | 1308.25 |
| 64+4 | 81.21 / 91.08 | 12.64 / 12.66 | 8.730 | 1299.09 |
| 48+4 return | 81.51 | 13.39 | 7.793 | 1298.06 |

## Approximate generation storage accounting

Measured process-wide from external first-token log snapshot to observed benchmark completion;includes marker/adb delay and all process threads,not exact expert-only decode accounting. Logical rchar and kernel read_bytes are different accounting scopes;do not calculate an exact cache hit ratio.

| Cache slots | Logical read GiB (two runs) | Kernel storage read GiB (two runs) |
|---|---|---|
| 48 | 11.81 / 11.77 | 7.97 / 7.84 |
| 56 | 10.40 / 10.37 | 7.79 / 7.70 |
| 64 | 9.38 / 9.33 | 7.77 / 7.78 |

Increasing slots reduced logical reads,but did not demonstrate faster decode:56similar/slightly lower,64lower than48in this sequence. Storage reads barely fall while DMA allocation rises;competition with page-cache is plausible but not proven causal. Retain normal48cache+4prefetch. More repetitions/randomized matched thermal/cache states needed to estimate small56vs48differences.

RSS high-water is since process start,cumulative within each pair. DMA sampled at request end,not peak. CPU/DMA mappings may overlap,do not add. Text quality inspection limited to coherent repeated Chinese plan and identical response hash,not broad numerical/quality equivalence.

## Reproduction and artifacts

Directory /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/. Run run-cache-slots-sweep.py;it uses run-app-ab.py with BENCH_CACHE48/56/64,BENCH_PREFETCH4,BENCH_IO4,BENCH_POLL1. Second request sets BENCH_REUSE_PROCESS1 and BENCH_RUNTIME_LABEL to first request label. Files app-chunk512-cache{48-a,56-b,64-c,48-return}-rN include raw results,installed hashes,config,runtime/logcat,frequency samples,IO snapshots,memory. cache-slots-sweep-summary.json preserves all row metrics. summarize-cache-slots.py reconstructs summary.

Temporary runtime.json removed after experiments;App restarted with budget-derived48+4defaults,batch512 retained. Nonblocking debug property1 retained as measured on test phone. No commits/pushes.

更小的 32/40 槽追加实验见 [小槽位测试](app-decode-cache-small-slots-20261017.md)。
