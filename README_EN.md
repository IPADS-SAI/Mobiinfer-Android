<div align="center">

<h1 align="center">
  <img src="Android-App/MobiInfer-Android/app/src/main/res/mipmap-xxxhdpi/ic_launcher.webp" width="64" height="64" alt="MobiInfer App icon" />
  MobiInfer · Android
</h1>


**Running a 35B-class MoE locally on a phone**

[![Release](https://img.shields.io/badge/release-v0.1--alpha-blue)](https://github.com/IPADS-SAI/Mobiinfer-Android/releases)
[![Platform](https://img.shields.io/badge/platform-Android%20%7C%20Snapdragon-3DDC84)](#)
[![Backend](https://img.shields.io/badge/backend-Hexagon%20NPU%20(HTP)-blue)](#)
[![Base](https://img.shields.io/badge/built%20on-llama.cpp--omni-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

[中文](README.md) | English

</div>

---

## News

- [2026.11.11] 🚀 **MobiInfer-Android v0.4.5: MiniCPM-V 4.7 MoE decode peaks at 23.3 tok/s**
  - ⚡ **Cross-layer expert pipeline**: O_DIRECT expert reads, gate predictions submitted two layers ahead (H+2), one FIFO task per expert with resident reader threads — the I/O is hidden under compute. Same device, same prompt: 23.3 tok/s.
  - 🧵 **Runs alongside other apps**: MoE inference no longer needs exclusive access; generation keeps going while other apps are in the foreground, at 21.3–21.8 tok/s.
  - 🧩 **Hexagon v85**: skels for v73 / v75 / v79 / v81 / v85 ship in the APK; backend synced to upstream 2026-11-10 (repack, op fusion, long-context FA).
  - 🐛 Fixed a low-vmem misdetection that forced vision onto the CPU and silently disabled DSpark; fixed exit code 139.
- [2026.11.02] 🎨 **On-device text-to-image with MOSS-Image**: 9B TE + fp16 5B DiT running on the Hexagon NPU.
- [2026.10.22] 🔥 Day-0 support for InternScience's latest Sage-A1.5 35B-A3B, with 41 expert layers streamed from an HTP-tiled bundle.
- [2026.10.19] 🛠️ Added support for Qwen3-30B-A3B / Qwen3.5-MoE-35B-A3B, with MoE experts using external slots and LES.
- [2026.10.17] 🚀 Released MobiInfer-Android v0.1 with model management, one-tap downloads, and on-device NPU inference.

---

## Introduction

> **A Snapdragon phone running a 35B-A3B-class MoE on the NPU**

MobiInfer-Android brings the complete **Qualcomm Hexagon NPU backend** and covers the full **on-device large-MoE** path: expert weights remain in flash, stream into the NPU on demand, lookahead scheduling hides I/O latency, and **DSpark speculative sampling** further accelerates decoding.

The repo provides the [Android APK](https://github.com/IPADS-SAI/Mobiinfer-Android/releases) and source (`Android-App/`); install it and run directly on the phone.

**Supported devices**


| SoC | Phone models |
|--|--|
| Snapdragon 8 Elite | OnePlus 13, Redmi K80 Pro, Xiaomi 15 / Xiaomi 15 Ultra, realme GT7 Pro, iQOO 13, Red Magic 11S Pro, HONOR Magic7 series, iQOO Neo10 Pro+ |
| Snapdragon 8 Gen 5 | OnePlus Ace 6T, iQOO Z11 Turbo, vivo S50 Pro mini, Redmi K100 Pro, realme GT8, Lenovo Legion Y70, STEPX Neo Pioneer Edition |
| Snapdragon 8 Elite Gen 5 | Xiaomi 17 series (including Pro / Pro Max / Ultra), OnePlus 15, realme GT8 Pro, Redmi K90 Pro Max, iQOO 15, HONOR Magic8 series, Samsung Galaxy S26 series, Red Magic gaming phones, vivo X300 series |

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

| Model | Scale | Weight size | Notes |
|---|---|---|---|
| **[MiniCPM-V 4.7 35B-A3B](https://www.modelscope.cn/models/fengerhu1/MiniCPM-V-4.7-35B-A3B-gguf-hexagon)** | 35B total / 3B active | ~19.3G | Primary on-device MoE target, multimodal (vision + text) |
| **[Qwen3.5-35B-A3B](https://www.modelscope.cn/models/fengerhu1/Qwen3.5-35B-A3B-gguf-hexagon)** | 35B total / 3B active | ~25.1G | Vision + text, LES + DFlash |
| **[Qwen3-30B-A3B](https://www.modelscope.cn/models/fengerhu1/Qwen3-30B-A3B-gguf-hexagon)** | 30B total / 3B active | ~20.3G | Text, LES + external experts |
| **[Sage-A1.5 35B-A3B](https://www.modelscope.cn/models/fengerhu1/Sage-A1.5-gguf-hexagon)** | 35B total / 3B active | ~24.6G | Vision + text (agentic), LES + external experts |
| **[Qwen3 series](https://www.modelscope.cn/models/Qwen/Qwen3-4B-GGUF)** | 0.6 / 1.7 / 4 B | 0.64 / 1.83 / 2.5 G | `qwen3`, text |
| **[Qwen3-VL series](https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct-GGUF)** | 2 / 4 B | ~1.9 / 3.3 G | `qwen3vl`, vision + text |
| DFlash / DSpark series | — | ~0.42–0.43 G | Speculative sampling (`dflash` / `dspark`) |
| **[MOSS-Image FP16 / Q4](https://www.modelscope.cn/models/fengerhu1/moss-image-hexagon)** | 5B DiT + 9B TE | ~3.6–9.6 G | text-to-image, fp16 / Q4_K_M, shares the text encoder with chat |

Architecture coverage includes `qwen3`/`qwen3moe`/`qwen3next`, `qwen3vl`/`qwen3vlmoe`, `qwen35moe`, `dflash`, `minicpm` families, and other common architectures within llama.cpp.

---

## Performance

> 📱 Measured on OnePlus 13 (Snapdragon 8 Elite, HTP0 + HTP1), text-only, 128-token output.
> 📱 iQOO 15 Ultra (Snapdragon 8 Elite Gen5, Hexagon v81), text-only, 260-token prompt, DSpark draft3.

| Model | Configuration | SoC | Prompt length | Prefill (token/s) | Decode (token/s) |
|---|---|---|---|---:|---:|
| MiniCPM-V 4.7 35B-A3B | Q3 tiled, H+2 pipeline + O_DIRECT + DSpark3 | 8 Elite Gen5 | 260 | 38.5 | **23.3** |
| MiniCPM-V 4.7 35B-A3B | as above, with other apps running | 8 Elite Gen5 | 260 | — | 21.3–21.8 |
| MiniCPM-V 4.7 35B-A3B | Q3, LES + DSpark | 8 Elite | 41 | 22.02 | 17.98 |
| MiniCPM-V 4.7 35B-A3B | Q3, LES + DSpark | 8 Elite | 512 | 94.28 | 13.65 |
| Qwen3-VL-4B-Instruct | Q4_K_M | 8 Elite | 512 | 1013.74 | 18.13 |

> Concurrency figures were re-measured with WeChat, the launcher and system services running (no force-stop); DSpark acceptance 83%.

---

## Text-to-image performance

> 📱 Snapdragon 8 Elite (Hexagon HTP), MOSS-Image FP16. End-to-end, including prompt encoding and the VAE decode.

| Resolution | Steps | End to end | Per step | Unit efficiency |
|---|---|---:|---:|---:|
| 256 × 256 | 4 | **10 s** | 2.5 s | 38.1 s/(Mpx·step) |
| 768 × 1024 | 8 | **100.1 s** | 12.5 s | **15.9 s/(Mpx·step)** |

---

## Quick Start

Three steps: **install the app → download the model → run.** The app ships with the optimal NPU defaults (dual HTP sessions + HTP vision + Q8_0 DSpark speculation) — no extra setup.

### 1. Install the app

Download the latest Android APK from [Releases](https://github.com/IPADS-SAI/Mobiinfer-Android/releases) and install it. On first install, allow installation from this source (Settings → Apps → Install unknown apps).

```bash
adb install -r MobiInfer-Android-<version>.apk
```

### 2. Download the model weights (~19 GB)

Open the app → **Model Manager** → select **MiniCPM-V 4.7 35B-A3B (Q3, NPU)** → **Download**. The app pulls the full bundle (main model + mmproj + DSpark draft + 40-layer tiled experts, ~**19 GB**) from ModelScope and verifies every file against `SHA256SUMS`. Use Wi‑Fi and keep ~19 GB free.

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
