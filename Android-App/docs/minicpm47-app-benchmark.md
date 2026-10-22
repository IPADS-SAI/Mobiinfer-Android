# MiniCPM-V 4.7 Android HTP / DSpark measurement

2026-10-15, OnePlus 13 / PJZ110, Hexagon v79. All requests were made through the App UI. Each case starts a new App process; OS file caches were not cleared. One observation per case, not a repeated-run median. Main model: minicpm47-les-hexagon.gguf; 40 tiled expert layers; original pad4320 F16 mmproj; Q8_0 DSpark draft. Same ctx=2048, bank=256, batch/ubatch=128, threads=4, image-max-tokens=64, greedy seed=42 and output cap=128 in every mode. DSpark n_max=3.

| Mode | Case | Images | First visible token (s) | Decode token/s | Output tokens | Total (s) | Draft accepted/generated |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | text | 0 | 2.001 | 11.85 | 128 | 12.802 | —/— |
| cpu | image | 1 | 8.395 | 11.22 | 108 | 18.022 | —/— |
| htp | text | 0 | 1.889 | 11.79 | 128 | 12.748 | —/— |
| htp | image | 1 | 3.781 | 11.26 | 108 | 13.369 | —/— |
| dspark | text | 0 | 2.132 | 13.90 | 128 | 11.340 | 77/147 |
| dspark | image | 1 | 4.187 | 12.64 | 128 | 14.314 | 77/150 |

`cpu`: CPU vision + HTP target, no draft. `htp`: HTP vision + HTP target, no draft. `dspark`: HTP vision + HTP target + HTP Q8_0 DSpark. Every image case attaches the same 224×224 image (red square left, blue circle right); every response identifies both correctly. Text cases contain no images. Prompts and complete responses are in [raw results](minicpm47-app-benchmark.json).

HTP reduces image TTFT from 8.395s to 3.781s (55%). DSpark increases text decode throughput from 11.79 to 13.90 token/s (18%), image decode throughput from 11.26 to 12.64 token/s (12%). Image output lengths differ (108 vs 128), so total response times cannot be used as a fixed-work speedup. DSpark changes some image-answer wording; this is not proof of exact greedy equivalence or a broad quality evaluation.

The local llama-cli uses server_context and accepts --spec-type draft-dspark / --model-draft. App integration links that inference core in process. No external llama-cli or HTTP listener is launched.

Continuous repeated requests are not yet established as stable: an earlier run completed two text requests, but the third lacked an App completion record and coincided with DSP queue error logs. Its native server timing is excluded. The successful measurements above restart the App between cases; they do not resolve that reliability issue. SLOT_ERASE rejects multimodal contexts; App reset now invalidates cache reuse for the next request, causing target and draft state to restart at position zero on the server thread.
