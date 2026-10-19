# MiniCPM-V 4.7 Hexagon runtime notes

Status: experimental, memory mapping recalibrated on OnePlus 13 on 2026-10-19.
Bundle: tiled mixed-Q3 experts, Q8 DSpark, real F16 pad4320 vision projector,
HTP0/HTP1, context 2048, top-8 routing. These estimates do not apply to arbitrary GGUF files.

## Memory setting

The model-manager preference accepts 5.2–8.0 decimal GB in 0.1 GB steps.
Changes apply on the next model load. `minicpm47_memory.h` owns the mapping;
Kotlin and the slider use the same endpoints. The selected value is an estimated
budget, not an allocator-enforced process limit.

The prefill bank and microbatch now decrease with the budget, as well as the
expert cache.
Batch remains 512, context 2048, DSpark draft length 3, and mmproj loads eagerly.

| Budget GB | Prefill bank | Ubatch | Prefetch depth | Fixed estimate GB |
| --- | ---: | ---: | ---: | ---: |
| 5.2–5.8 | 32 | 4 | 0 | 4.65 |
| 5.9–6.4 | 128 | 16 | 4 | 5.05 |
| 6.5–8.0 | 256 | 128 | 4 | 5.70 |

```
total_slots = floor((budget_MB * 1,000,000 - fixed_bytes) / 62,914,560)
cache_slots = total_slots - prefetch_depth
```

One expert slot is 1,572,864 bytes per layer, or 62,914,560 bytes across 40 layers.
The fixed estimates include resident weights, Q8 draft, eager vision, context,
profile-specific graph/bank reservations and a margin for UI/variation. At
5.2/6.0/8.0 GB, demand cache capacities are 8/11/32 and prefetch depths are 0/4/4.
At a profile boundary, a larger prefill bank/ubatch trades off against cache
capacity; increasing the budget need not increase the slot count at that step.
These estimates apply only to this model bundle. They do not cover arbitrary
context sizes, image resolutions, other models or unlimited conversation history.

DSpark draft3 needs ubatch>=4. A smaller bank must fit the worst-case top-8 expert
union for its microbatch. Invalid local benchmark overrides fail before inference
instead of triggering the non-causal-attention assertion observed with ubatch1.

Phone memory sampling records smaps_rollup RSS/PSS and DMA objects exposed by
process fdinfo, deduplicated by inode. RSS alone misses most HTP buffers. Their
simultaneous PSS+DMA sum is a conservative estimate for observed resources:
mapped DMA pages may be counted twice, short peaks can be missed, and driver
internal allocations and system file cache are not fully included. Native probe
measurements exclude Java/UI; the estimates retain a reserve rather than claiming
a strict whole-device physical memory bound.

## Timings and request lifecycle

The native performance array contains prompt time/count, generation time/count,
and actual vision encoder time. `mtmd_encode` measures encoder work; queuing an
image is not encoding. Text requests have zero vision time. UI total elapsed time
uses the request wall clock and includes media preparation once. Failed/cancelled
requests do not display successful completion rates. The waiting indicator is
shown before the first output and hidden on output, cancellation or failure.

In the earlier 2026-10-16 measurements, the packaged native runtime passed phone
SSH tests for text, three consecutive
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
