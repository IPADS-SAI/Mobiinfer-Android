<div align="center">

# MobiInfer · Android

### On-device Large-Model Inference on Snapdragon NPU

**Running large models on the Qualcomm Hexagon NPU: fitting a 35B-class MoE into a phone**

[![Platform](https://img.shields.io/badge/platform-Android%20%7C%20Snapdragon-3DDC84)](#)
[![Backend](https://img.shields.io/badge/backend-Hexagon%20NPU%20(HTP)-blue)](#)
[![Base](https://img.shields.io/badge/built%20on-llama.cpp--omni-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

[中文](README.md) | English

</div>

---

## What we did

MobiInfer-Android ports the full **Qualcomm Hexagon NPU (HTP/HTA DSP) backend** into [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) and builds a complete on-device path for **large sparse MoE models**: experts stay in flash, stream into the NPU on demand, I/O latency is hidden by lookahead scheduling, and an adapted **DSpark speculative sampler** pushes decode speed further.

In one line:

> **A single Snapdragon phone running a 35B-A3B-class MoE on the NPU, with no capability loss.**

---

## Highlights

- 🚀 **Native Hexagon NPU backend**
  A complete `ggml-hexagon` implementation ported from upstream llama.cpp: HVX/HMX tensor kernels, Flash-Attention, Gated Delta Net, DMA queues, and multi-device partitioning (single device, virtual-session layer split, physical-core tensor split, core-group row split). Models offload to the HTP with "GPU" semantics.

- 🧠 **On-device MoE expert offload**
  Expert weights live in flash; an `--moe-experts` manifest plus resident `--moe-slot-bank` slots stream routed experts into the HTP. Supports cross-device TopK and tiled expert allocation/upload, pooled expert I/O, bounded multi-row prefetch, and optional prefill cache warming.

- ⚡ **LES (Lookahead Expert Scheduling)**
  Cross-layer routing prediction prefetches the next layer's likely experts; loaded ones compute first while the rest load in parallel. Prediction is used only for **prefetch and scheduling** — compute still strictly follows the model's own token-level routing, missed experts are loaded normally, and **no extra training or capability loss** is involved.

- 🎯 **DSpark speculative sampling**
  DFlash (multi-layer hidden-state extraction, multi-token single-step prediction) plus a semi-autoregressive Markov head and a confidence head, adapted into the on-device pipeline (`speculative.type = dspark`) and working together with expert scheduling.

- 📦 **Expert packing toolchain**
  `experts.py` (extract/verify) → `strip_experts.py` (strip duplicated experts from the GGUF) → `pack_hexagon.py` (convert to the HTP tiled `hexagon-tiled-v1` layout). Packing calls the backend's `ggml_backend_pack_moe_expert` directly, guaranteeing the layout matches at runtime.

- 💾 **Weight access optimization**
  Layer-wise mixed-precision quantization plus contiguous expert repacking to reduce scattered flash reads, improve sequential read efficiency, and lower the storage bandwidth required for expert loading.

---

## Supported models

| Model | Scale | Notes |
|---|---|---|
| **MiniCPM-V 4.7 35B-A3B** | 35B total / 3B active | Primary on-device MoE target, multimodal (vision + text) |
| Qwen3.5 / 3.6 MoE | 35B-A3B / 122B-A10B / 397B-A17B | `qwen35moe`, linear attention + gated delta net |
| DFlash / DSpark draft models | — | Speculative sampling (`dflash` / `dspark`) |

Architecture coverage includes `qwen35moe`, `dflash`, `minicpm` families, and the rest of the common architectures within the llama.cpp ecosystem.

---

## Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │                llama.cpp-omni                │
                  │   (CLI / Server / multimodal / TTS pipeline) │
                  └───────────────┬──────────────────────────────┘
                                  │  -ngl / --device / --split-mode
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
      ┌───────────┐        ┌───────────┐         ┌───────────┐
      │  CPU/ARM  │        │  Adreno   │         │  Hexagon  │  ← this repo
      │  (NEON)   │        │  (OpenCL) │         │  NPU/HTP  │
      └───────────┘        └───────────┘         └─────┬─────┘
                                                       │
                          ┌────────────────────────────┼───────────────────────────┐
                          ▼                            ▼                           ▼
                   ┌─────────────┐            ┌───────────────┐           ┌───────────────┐
                   │  HVX / HMX  │            │  Expert       │           │  DSpark draft │
                   │  tensor     │            │  streaming    │           │  speculative  │
                   │  kernels    │            │  + LES sched  │           │  sampling     │
                   └─────────────┘            └───────┬───────┘           └───────────────┘
                                                      │ pread (tiled, contiguous layout)
                                                      ▼
                                              ┌───────────────┐
                                              │  UFS / SDD    │
                                              │  external     │
                                              │  expert weights│
                                              └───────────────┘
```

At inference, dense weights (attention, shared experts, embeddings) stay resident on the NPU; routed expert weights remain in flash, are predicted by LES and streamed into resident slots (slot bank), and expert matvec runs on the HTP. The DSpark draft model shares this expert scheduling path with the main model.

---

## Performance

> 📱 Measured on OnePlus 13 (HTP0 + HTP1), 512-token input / 128-token output, text-only CLI.

Best current config: **LES prefetch cap 8 + DSpark draft3** (Q3 main model + HTP tiled experts).

| Metric | Value |
|---|---:|
| Prefill throughput | **66.22 token/s** |
| Decode throughput | **14.64 token/s** |
| Total inference (prefill + decode) | **16.47 s** |

> 512/128, ctx2048, draft3, fixed 512-token Chinese input; three runs produced identical output SHA256. Raising the prefetch cap from 8 to 14/16 actually lowered decode, so 8 is used. Full conditions, raw logs, and reproduction commands are in [`docs/benchmark-prefill512-decode128.md`](docs/benchmark-prefill512-decode128.md) and the [on-device reproduction notes](docs/on-device-decode-reproduction.md).

---

## Quick Start

Three steps: **install the app → download the model → run.** The app ships with the optimal NPU defaults (dual HTP sessions + HTP vision + Q8_0 DSpark speculation) — no extra setup.

### 1. Install the app

Download the latest Android APK from [Releases](https://github.com/IPADS-SAI/Mobiinfer-Android/releases) and install it. On first install, allow installation from this source (Settings → Apps → Install unknown apps).

```bash
adb install -r MobiInfer-Android-<version>.apk
```

### 2. Download the model weights (~17 GB)

Open the app → **Model Manager** → select **MiniCPM-V 4.7 35B-A3B (Q3, NPU)** → **Download**. The app pulls the full bundle (main model + mmproj + DSpark draft + 40-layer tiled experts, ~**17 GB**) from ModelScope and verifies every file against `SHA256SUMS`. Use Wi‑Fi and keep ~17 GB free.

### 3. Run

Once the download completes, select the model in **Model Manager** and start chatting — text or image.

> Need manual sideloading, CLI usage, building from source, multi-device split, or troubleshooting? See **[DEPLOY.md](DEPLOY.md)** and [`docs/backend/snapdragon/README.md`](docs/backend/snapdragon/README.md).

---

## Key flags

| Flag | Purpose |
|---|---|
| `--moe-experts PATH` | Expert directory/manifest path (MoE with expert-major routing) |
| `--moe-slot-bank N` | Number of resident routed-expert slots per layer |
| `--moe-k-prefill N` | top-k for the prompt eval stage |
| `--moe-k-decode N` / `--moe-k-decode-cycle 8,4` | top-k for the generation stage (cyclable) |
| `--device HTP0,...` | Select Hexagon devices/sessions (Hexagon is equivalent to GPU under `-ngl` semantics) |
| `--split-mode layer\|tensor` | Cross-session layer split / cross-physical-core tensor split |
| `-ngl N` / `--load-mode` | Number of layers to offload / load mode (`auto`/`none`/`mmap`/`mlock`/`dio`) |
| `speculative.type = dspark` | Enable DSpark speculative sampling |

Hexagon environment variables (`GGML_HEXAGON_DEVICES` / `GGML_HEXAGON_POWER_MODE` / `GGML_HEXAGON_PROFILE` / `GGML_HEXAGON_OPFILTER`, etc.) are documented in the Snapdragon backend docs.

---

## Layout

```
ggml/src/ggml-hexagon/     Hexagon backend host side (scheduling, multi-device, DMA) + htp/ (DSP-side HVX/HMX kernels)
src/llama-moe-slot.*       MoE expert slot runtime: prefetch, scheduling, I/O thread pool, diagnostic counters
src/llama-moe-experts.*    External expert manifest and loading
src/models/qwen35moe.cpp   Qwen3.5/3.6 MoE (incl. MiniCPM-V 4.7 35B-A3B)
src/models/dflash.cpp      DFlash / DSpark draft model
common/speculative.cpp     Speculative sampling (DSpark adaptation)
tools/moe/                 Expert extract / strip / pack tools
docs/backend/snapdragon/   Snapdragon build, run, multi-device docs
docs/les/                  LES lookahead expert scheduling docs (CN/EN)
```

---

## Relation to upstream

MobiInfer-Android is built on [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) (itself built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)). The Hexagon backend and supporting op/API changes are ported from upstream llama.cpp; MoE expert offload, LES scheduling, and DSpark integration are this project's on-device extensions. The Android App (`Android-App/`) is based on [OpenBMB/MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps).

---

## Credits & License

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps) — Android app upstream
- [Qualcomm Snapdragon Toolchain](https://github.com/snapdragon-toolchain)
- MiniCPM-V model family

Licensed under MIT — see [LICENSE](LICENSE).

---

## Build & run on Snapdragon

```sh
./scripts/snapdragon/build.py --target adb --push        # build + deploy to a connected device

export LLAMA_MOE_HTP_PACKED_DIR=/models/experts-htp-tiled
./bin/llama-cli -m /models/model-les.gguf \
    --moe-experts /models/experts-htp-tiled --moe-slot-bank 32 \
    --device HTP0,HTP1 --split-mode layer --tensor-split 1,1 -ngl 99 \
    -fit off -c 2048 -b 4 -ub 4 -p 'Hello'
```

See [`tools/moe/HEXAGON.md`](tools/moe/HEXAGON.md) for the expert preparation pipeline and [`docs/backend/snapdragon/README.md`](docs/backend/snapdragon/README.md) for all build/run/multi-device options.
