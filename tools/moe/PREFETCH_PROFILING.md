# Prefetch overlap measurements

Build and deploy a coherent current `llama-cli` and shared-library bundle. Run on
the phone; CPU/CUDA results do not establish Hexagon timing. Keep model hashes,
environment, full stdout/stderr, exit status and memory samples with each run.

Example environment for the tiled mixed-Q3 MiniCPM-V 4.7 bundle:

```sh
export LLAMA_MOE_PREFETCH_TIMING=1
export LLAMA_MOE_LOOKAHEAD_PREFETCH=1
export LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=4
export LLAMA_MOE_DECODE_WINDOW=16
export LLAMA_MOE_PREFETCH_MAX_ROWS=8
export LLAMA_MOE_WINDOW_MAX_ROWS=8
export LLAMA_MOE_IO_POOL=1
export LLAMA_MOE_IO_THREADS=8
export LLAMA_MOE_HTP_PACKED_DIR="$PWD/experts-htp-tiled"
export LLAMA_MOE_PREWARM=
export LLAMA_MOE_PREFILL_WARM_ROWS=0
export LLAMA_MOE_PHASE_REUSE=0
export GGML_HEXAGON_DEVICES=2
export GGML_HEXAGON_MBUF=256M,512M,4G

./bin/llama-cli -m models/minicpm47-les-hexagon.gguf \
  --moe-experts experts --moe-slot-bank 128 \
  --device HTP0,HTP1 --split-mode layer --tensor-split 1,1 -ngl 99 \
  -c 2048 -b 16 -ub 16 -fit off --ctx-checkpoints 1 --no-warmup \
  --spec-type draft-dspark --model-draft models/dspark-MiniCPM-V-4_7-q8_0.gguf \
  --device-draft HTP0,HTP1 -ngld 99 --spec-draft-n-max 1 \
  --temp 0 --seed 42 --single-turn --reasoning off -n 256 \
  -p '请介绍一个适合全家参加的周末活动，并说明具体安排。' \
  > prefetch.stdout 2> prefetch.log

python3 tools/moe/analyze_prefetch.py prefetch.log --output prefetch.json
```

Configure the deployed library and DSP skeleton search paths for the actual
installation; do not mix arbitrary old/new libraries. This example does not
measure or enforce a whole-process memory cap. Add images and long-context cases
separately, and measure unique DMA plus CPU memory as well as speed.

`prefetch-summary` separates completed reads, published results, duplicate results
discarded because demand already installed the expert, callbacks skipped because
no read was needed/possible, and errors. A published result is not proof that it
was subsequently consumed. Completed reads equal published plus discarded.

`prefetch-timing` pairs accepted prefetch arm and next-layer demand in the same
graph epoch, preventing an old job from masquerading as a long overlap window.
It reports mean overlap, jobs still pending/running at demand, completed expert
reads, and batch wall time per completed expert. The last metric includes parallel
I/O and worker overhead; it is throughput, not an isolated expert's read latency.
Whole-job lateness does not establish whether an individual useful expert was
late. Initialization is included. The worker is joined before final summaries.

Timing is opt-in and adds a timestamp and mutex observation to the CPU callback;
use profiling to explain behavior and separate runs with timing disabled to judge
end-to-end speed. Means alone are insufficient to establish safe admission limits.

For an equal-memory depth sweep, keep cache + reserved prefetch slots fixed, e.g.
depth0/1/2/4/8 with cache20/19/18/16/12. Depth0 disables prediction as well as I/O,
and smaller demand capacity can change speculative verification's window/bank
admission; report those changes. Compare three alternating repeats, logical total
I/O, output stability, TTFT, prefill/decode and whole-request time.

Physical capacity is fixed at model load. Future adaptation should cap admitted
reads within that reserved capacity, rather than assume changing worker depth
resizes model tensors. Keep demand nonblocking and preserve authoritative top-8
routing. No timing-based adaptation is enabled by this instrumentation.

For per-job CPU observations set `LLAMA_MOE_PREFETCH_TRACE=1` as well as
`LLAMA_MOE_PREFETCH_TIMING=1`. `prefetch-job` records graph epoch, source row
count, target layer, admitted expert count, completed reads, worker queue delay
and whole-job wall time. `prefetch-demand` records the matching demand window
and whether the job is still pending/running. This trace is verbose and perturbs
timing; it is disabled by default and should not be enabled for speed selection.
`LLAMA_MOE_PREFETCH_READ=0/1/2/4/8` can cap reads independently of the configured
prediction depth and physical reserve. Keep depth and reserve constant to isolate
admission count. Zero reads still computes predictions and uses eviction hints.

The analyzer also emits `trace.cpu` grouped by token rows and, if
`GGML_HEXAGON_PROFILE=1` and verbose logging are enabled, `trace.dsp_ffn`.
For draft length 3, full-width target verification has rows=4; a rows=2
initialization probe must not be folded into it. DSP FFN starts at the sparse
gate/up matmul and ends at `ffn_out` (including shared experts), excludes routing,
expert disk reads and host/RPC time, and sums reported non-fence op durations.
The CPU arm-to-demand window is broader: FFN plus the following layer's
attention/router and transfers. CPU and DSP clocks are not aligned by this
analyzer, and job completion does not establish that each useful expert was ready.
