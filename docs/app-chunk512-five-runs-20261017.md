# App chunk512: five consecutive requests

Original OnePlus13 PJZ110 / SM8750. Changed MiniCPM47MemoryConfig::batch128→512,so App n_batch=n_ubatch512. Bank256,cache48+prefetch4,IO4,DSpark Q8 draft3,vision model on HTP,ctx2048,temp0/seed42,8GB configured budget estimate. Dedicated nonblocking DSP polling enabled through debug.mobiinfer.htp_poll=1;profile0.

Built debug APK and installed successfully. APK SHA256 e2ecda981b683f78c04aec9199fa6735673f1e66ec85957dd4e49ab7322ff056;installed hash recorded each round. App log confirms batch512 and tok512 expert step. Prompt SHA25632e4bf4a46971fb43552ab91ca17b7de204495d0bf7784baa5a0da513a52cb52,512input/128output.

Five requests in one process,PID10072,new conversation each time;cache_n0 throughout. No deliberate cooldown,file-cache flushing or explicit prewarming. UI automation/sampling introduces gaps,so this is consecutive interactive requests,not a zero-gap stress test.

| Round | Prefill tok/s | Decode tok/s | Accepted/drafted | Process RSS high-water MiB | DMA snapshot GiB |
|---|---:|---:|---:|---:|---:|
| 1 | 81.05 | 13.60 | 81/136 | 1302.15 | 7.793 |
| 2 | 94.27 | 13.78 | 81/136 | 1307.71 | 7.793 |
| 3 | 94.32 | 13.65 | 81/136 | 1333.07 | 7.793 |
| 4 | 94.28 | 13.65 | 81/136 | 1352.63 | 7.793 |
| 5 | 94.51 | 13.64 | 81/136 | 1364.16 | 7.793 |

Median prefill94.28tok/s,decode13.65tok/s. All5response hashes identical: 84b85854d492a8c22e2a658bd15c0d8891c676188bc5a8d928179aac4aaf6a3b. Output is coherent Chinese planning text and deliberately truncated at128tokens;identity across repeats is not a general quality test. Chunk512 output differs from128chunk results; no bitwise cross-batch equivalence claim.

RSS high-water is cumulative since common process launch;DMA is a final snapshot per request,not peak. CPU/DMA may overlap;do not add them. Budget8000MB is an estimator,not a hard allocation limit;observed DMA alone8367480832bytes exceeds8GB decimal. Relative to earlier chunk128 snapshot7814733824bytes,DMA rose552747008bytes(about527.14MiB). No strict8GB physical-memory compliance claim.

First512chunk reported readall10322MiB versus29285MiB across prior128chunks;larger chunks deduplicate experts across more tokens and reduce repeated expert installation. This is a mechanism supported by logs,not a controlled speedup ratio: temperature/cache history and output differ from prior runs.

## Reproduction

Scripts/raw evidence directory: /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/. run-app-ab.py verifies exact prompt before submit and saves new benchmark rows,installed APK hash,runtime/logcat,process IO,frequency samples and memory. First request:

```sh
BENCH_LABEL=app-chunk512-continuous-r1 BENCH_POLL=1 BENCH_PREFETCH=4 BENCH_IO=4 python3 run-app-ab.py
```

Subsequent requests,for N=2..5:

```sh
BENCH_LABEL=app-chunk512-continuous-rN BENCH_RUNTIME_LABEL=app-chunk512-continuous-r1 BENCH_REUSE_PROCESS=1 BENCH_POLL=1 BENCH_PREFETCH=4 BENCH_IO=4 python3 run-app-ab.py
```

Replace rN with each actual round. Runtime label stays r1 because native configuration is read once;artifact filenames and distinct benchmark rows identify rounds. Summary chunk512-five-runs-summary.json;raw app-chunk512-continuous-r{1..5}-result.json and corresponding logs/config/memory/frequency files. Build output chunk512-build.log. Default batch512 remains in source and installed APK;temporary benchmark runtime.json is removed afterward. Polling property1 is retained on this test device for the measured wait mode. No commit/push.
