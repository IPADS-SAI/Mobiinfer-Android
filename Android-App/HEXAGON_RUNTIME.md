# MiniCPM-V 4.7 Hexagon runtime notes

Status: experimental, 2026-10-16. Final defaults are still under phone validation.
Bundle: tiled mixed-Q3 experts, Q8 DSpark, real F16 pad4320 vision projector,
HTP0/HTP1, context 2048, top-8 routing. These estimates do not apply to arbitrary GGUF files.

## Memory setting

The persistent model-manager preference accepts 5.5–6.0 decimal GB in 0.1 GB steps.
Changes apply on the next model load. JNI passes the budget to the runtime constructor
before allocation. `minicpm47_memory.h` owns the mapping:

```
total_slots = floor((budget_MB * 1,000,000 - 4,740,000,000) / 62,914,560)
cache_slots = total_slots - 4
```

One expert occupies 1,572,864 bytes in one layer; one slot across all 40 layers
costs 62,914,560 bytes. Four slots per layer reserve asynchronous prefetch capacity.
The resulting demand-cache capacities at 5.5/5.6/5.7/5.8/5.9/6.0 GB are
8/9/11/12/14/16. This is a quantized linear mapping, not an allocator-enforced cap.
The fixed term includes a measured baseline plus reserve; a model change requires
recalibration. Context checkpoints are limited to one to avoid history-driven growth.

Candidate configuration: prefill bank 128 per HTP backend, batch/ubatch 16,
DSpark draft length 1, IO workers 8. A bank is shared by layers on its backend;
two devices therefore allocate two banks. A 128-expert bank is 192 MiB per device.
Static hot-expert prewarming is disabled by default.

Phone memory sampling uses unique process FD DMA objects (deduplicated by inode)
plus full process PSS every 0.5 s. Their sum is a conservative upper bound because
mapped DMA pages may also appear in PSS. Sampling can miss short peaks and excludes
Java/UI when running the standalone native probe. It is not physical flash usage
or a hard guarantee of a 6 GB maximum.

The original CLI-calibrated 20-cache-slot candidate measured 5.737–5.739 GB during
two image/256-token runs, but the actual packaged App native libraries reached
6.047 GB over text, three images, cancellation, error and recovery. That candidate
was rejected. The mapping was reduced to at most 16 cache slots. Two repeats at
each revised endpoint, retaining conversation history through three image requests,
passed with sampled upper bounds 5.744/5.744 GB at 6 GB and 5.241/5.240 GB at
5.5 GB. Both runs also passed cancellation/error recovery. This leaves about
256 MB at the upper endpoint for Java/UI and variation; actual UI-process peaks
and all context-length extremes have not been validated. The setting is a budget
estimate and does not implement a strict whole-process allocation cap.

## Timings and request lifecycle

The native performance array contains prompt time/count, generation time/count,
and actual vision encoder time. `mtmd_encode` measures encoder work; queuing an
image is not encoding. Text requests have zero vision time. UI total elapsed time
uses the request wall clock and includes media preparation once. Failed/cancelled
requests do not display successful completion rates. The waiting indicator is
shown before the first output and hidden on output, cancellation or failure.

The packaged native runtime passed phone SSH tests for text, three consecutive
image requests, cancellation followed by generation, and oversized-prompt error
followed by generation. Vision times were 380–407 ms; decode was 11.7–13.1 token/s
for the short successful requests. These runs used the rejected 20-slot memory
candidate, so they establish lifecycle behavior, not final-budget performance.
App UI device interaction was waived by the user because ADB is unavailable;
APK compilation is the available UI evidence.
With retained image history, the request may re-encode earlier images: measured
encoder work for one/two/three images was about 408/782/1140 ms in one run. The
metric reports actual work performed for that request, including history replay.

## Static prewarm evidence

A100 routing capture used 128 Chinese BELLE instruction prompts for ranking and
32 disjoint prompts for validation. The pinned dataset and capture instructions
are in `tools/moe/HOT_EXPERTS.md`. Top-48 offline decode-frequency coverage was
65.38%; this is not an online cache hit rate.

A stale prewarm placement snapshot could survive fallback loading and later
republish slots whose contents had been evicted during initialization/decode.
Window entry now consumes the snapshot after either entry branch. After the fix,
two Chinese prompts × two repeats × off/static modes produced identical greedy
answer bodies for all eight runs. This is a regression sample, not a universal
numerical proof.

| Prompt | Decode off → static | Prefill off → static | Server-launch TTFT off → static |
| --- | --- | --- | --- |
| belle-1687 | 10.545 → 10.715 t/s | 39.095 → 35.700 t/s | 2225 → 2439 ms |
| belle-839 | 9.920 → 10.075 t/s | 21.050 → 18.435 t/s | 1329 → 1519 ms |

Each speed is the median of two runs. One belle-839 baseline lacks a launch
timestamp, so its TTFT has only one observation. TTFT excludes model loading.
The ~1.6% decode change is insufficient evidence of benefit; prefill and TTFT
worsened. Existing interval diagnostics omit the final interval and sample cache
events, so their sums are not complete I/O totals or hit rates. New cumulative
`io-summary` and `cache-summary` counters include initialization, demand loading
and prefetch; requested bytes are logical bytes, not physical storage traffic.
The full small-cache comparison is now in `tools/moe/HOT_EXPERTS.md`: static
warming improved window hit ratio by about one percentage point but increased
logical requested bytes, with mixed decode changes. It remains disabled.

## Capacity correctness

Window admission reserves in-flight prefetch slots even under unified LRU;
pending slots cannot be evicted. A verification graph whose worst-case expert
union exceeds usable window capacity falls back to the prefill bank.
When lookahead writes separate window buffers, it does not reserve a tail in
the independent prefill bank. This allows the full bank capacity to be used.
Higher draft lengths can therefore increase bank fallback and I/O despite fewer
target invocations. Final selection requires equal-budget repeated measurements.

Raw evidence and reproducible command JSON files are maintained separately in the
experiment artifacts `hexagon-results/{hot-expert-profile,memory-budget}`.
Nothing in this note constitutes an accepted 60–70 token/s prefill reproduction.


## Native failure containment

App initialization now reports library linkage errors through `LlamaState.Error`
instead of rethrowing from the initialization coroutine. JNI backend initialization
converts C++ exceptions into Java runtime errors. MiniCPM's server worker catches
escaping C++ exceptions, records the failure under a mutex, and lets the response
reader stop on its regular poll instead of waiting forever. Subsequent submissions
on that runtime fail with a request to reload the model. No automatic request retry
or CPU fallback is performed on a failed context.

These boundaries cannot catch native signals, backend `GGML_ABORT`, or Android
low-memory process termination. Inference is still in the UI application's process;
containing those failures requires a separate inference service process and IPC.
This change does not establish compatibility with additional Snapdragon devices.
