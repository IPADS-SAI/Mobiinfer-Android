<div align="center">

# MobiInfer · Android


**Running a 35B-class MoE locally on a phone**

[![Platform](https://img.shields.io/badge/platform-Android%20%7C%20Snapdragon-3DDC84)](#)
[![Backend](https://img.shields.io/badge/backend-Hexagon%20NPU%20(HTP)-blue)](#)
[![Base](https://img.shields.io/badge/built%20on-llama.cpp--omni-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

[中文](README.md) | English

</div>

---

## Introduction

> **A Snapdragon phone running a 35B-A3B-class MoE on the NPU**

MobiInfer-Android ports the full **Qualcomm Hexagon NPU (HTP/HTA DSP) backend** into [llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni) and builds a complete on-device path for **large sparse MoE models**: expert weights stay resident in flash, are streamed into the NPU on demand, lookahead scheduling masks the I/O latency, and an adapted **DSpark speculative sampler** pushes decode speed further.

The Android app ([`Android-App/`](Android-App/README.md), based on [OpenBMB/MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps)) ships with this repo — install it and run directly on the phone.

---

## Highlights

- 🚀 **Native Hexagon NPU backend**
  HVX/HMX tensor cores, Flash-Attention, Gated Delta Net, DMA queues and multi-device split

- 🧠 **On-device large-MoE expert offload**
  Expert weights in external flash, streamed into the HTP on demand

- ⚡ **LES (Lookahead Expert Scheduling)**
  Cross-layer prediction prefetches the next layer's experts; hits compute first, misses load in parallel

- 🎯 **DSpark speculative sampling**
  DFlash + a semi-autoregressive Markov head and a confidence head, adapted on-device and coordinated with expert scheduling

- 📦 **Expert packing toolchain**
  Packs expert weights and emits the HTP tiled layout.

- 💾 **Weight access optimization**
  Layer-wise mixed-precision quantization + contiguous expert repacking to reduce scattered flash reads.

---

## Supported models

| Model | Scale | Notes |
|---|---|---|
| **MiniCPM-V 4.7 35B-A3B** | 35B total / 3B active | Primary on-device MoE target, multimodal (vision + text) |
| **Qwen3.5-35B-A3B** | 35B total / 3B active | `qwen35moe`, vision + text, LES + DFlash |
| **Qwen3-30B-A3B** | 30B total / 3B active | `qwen3moe`, text, LES + external experts |
| **Sage-A1.5 35B-A3B** | 35B total / 3B active | `qwen35moe`, vision + text (agentic), LES + external experts |
| **Qwen3-0.6B** | 0.6B (dense) | `qwen3`, text |
| **Qwen3-1.7B** | 1.7B (dense) | `qwen3`, text |
| **Qwen3-4B** | 4B (dense) | `qwen3`, text |
| **Qwen3-VL-2B** | 2B (dense) | `qwen3vl`, vision + text |
| **Qwen3-VL-4B** | 4B (dense) | `qwen3vl`, vision + text |
| DFlash / DSpark draft models | — | Speculative sampling (`dflash` / `dspark`) |

Architecture coverage includes `qwen3`/`qwen3moe`/`qwen3next`, `qwen3vl`/`qwen3vlmoe`, `qwen35moe`, `dflash`, `minicpm` families, and the rest of the common architectures within the llama.cpp ecosystem.

---

## Performance

> 📱 Measured on OnePlus 13 (Snapdragon 8 Elite, HTP0 + HTP1), text-only, 128-token output.

| Model | SoC | Prefill (token/s) | Decode (token/s) |
|---|---|---:|---:|
| MiniCPM-V 4.7 35B-A3B (Q3, MoE + LES + DSpark) | Snapdragon 8 Elite | 94.28 | 13.65 |
| MiniCPM-V 4.7 35B-A3B (same, app, short prompt, 41 tokens) | Snapdragon 8 Elite | 22.02 | 17.98 |
| Qwen3-VL-4B-Instruct (Q4_K_M) | Snapdragon 8 Elite | 1013.74 | 18.13 |

MiniCPM rows are app measurements: the short prompt uses the Python Fibonacci example without mmproj; the 512-token result is the [median of five runs](docs/app-chunk512-five-runs-20261017.md) with mmproj loaded. Qwen3 is measured with the CLI.

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
Android-App/               Android app source
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

## Credits & License

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni)
- [MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps) — Android app upstream
- [Qualcomm Snapdragon Toolchain](https://github.com/snapdragon-toolchain)
- MiniCPM-V model family

Licensed under MIT — see [LICENSE](LICENSE).
