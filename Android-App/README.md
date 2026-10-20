# MobiInfer · Android App

**English** | [中文](README_zh.md)

The on-device **Android** app shipped with **MobiInfer-Android**. It runs large multimodal models **fully locally on a Qualcomm Hexagon NPU (HTP)** — no server, no data leaves the device.

> See **[DEPLOY.md](../DEPLOY.md)** at the repository root for deployment and installation, and **[README](../README.md)** for the engine's capabilities and architecture.

---

## Current capabilities

| Capability | Status |
| --- | --- |
| Fully on-device inference | ✅ No remote server; all inference runs on the phone |
| **Qualcomm Hexagon NPU (HTP)** acceleration | ✅ LLM **and** vision encoder run on the NPU |
| **MiniCPM-V 4.7 35B-A3B** (MoE) on-device | ✅ Routed experts streamed from flash on demand (LES) |
| **DSpark** speculative decoding | ✅ Q8_0 draft, up to 3 draft tokens (enabled by default) |
| Multimodal I/O | ✅ Text chat + image Q&A |
| In-app **Model Manager** | ✅ Downloads model bundles from ModelScope with `SHA256SUMS` verification |
| Qwen3-VL-4B (HTP) | ✅ Available as a second model |

> Keep in mind: some vision operators (e.g. position-embedding `UPSCALE`) currently fall back to CPU. Only Hexagon **v79** (validated) is targeted; v73+ builds are produced but not validated on this branch.

## Supported models

This NPU branch offers four entries in **Model Manager** (the upstream demo's other variants are intentionally not shipped here):

| Model | Quant | Size | Notes |
| --- | --- | --- | --- |
| **MiniCPM-V 4.7 35B-A3B** | Q3 (mixed, tiled experts) | **~17 GB** | Main target. MoE experts streamed from flash; needs Hexagon v79 + ≥12 GB RAM |
| **Qwen3.5-35B-A3B** | Q4_K_M (tiled experts) | **~25 GB** | `qwen35moe`, vision + text, LES + DFlash |
| **Qwen3-30B-A3B** | Q4_K_M (tiled experts) | **~20 GB** | `qwen3moe`, text-only, LES + external experts |
| **Qwen3-VL-4B-Instruct** | Q4_K_M | ~3.3 GB | Vision-language; language + vision on HTP |

The 4.7 bundle is **not a single GGUF**. It ships a resident main model, a padded F16 `mmproj`, a Q8_0 DSpark draft, and 40 tiled expert layer files:

```text
files/models/minicpm-v-4_7/
├── minicpm47-les-hexagon.gguf                     # main model (no routed experts)
├── mmproj-MiniCPM-V-4_7-real-f16-pad4320.gguf     # vision encoder (F16)
├── dspark-MiniCPM-V-4_7-q8_0.gguf                 # DSpark draft (Q8_0, required)
└── experts-htp-tiled/
    ├── manifest.json
    └── layer_000.bin ... layer_039.bin
```

Keep the main model and `experts-htp-tiled/` from the same bundle — a mismatch or missing expert file refuses to load.

## Requirements

- Qualcomm Snapdragon device with **Hexagon v79** NPU
- Android 8.0 (API 26)+, **arm64-v8a**
- **≥ 12 GB RAM** (4.7 35B-A3B validated on 12 GB devices)
- **≥ 17 GB free fast (UFS) storage** — experts are read from flash on demand
- Wi‑Fi for the initial ~17 GB model download

## Install & run

Pre-built APKs are published in the [Releases](https://github.com/IPADS-SAI/Mobiinfer-Android/releases) of the parent repository. Install, then download the model from the in-app **Model Manager** and start chatting.

Full step-by-step (including manual sideloading, CLI usage, performance figures, and troubleshooting) is in **[../DEPLOY.md](../DEPLOY.md)**.

## Build from source

Requirements: Android Studio (Giraffe+), Android SDK + NDK `27.0.12077973`, CMake `3.22.1`, and the Hexagon SDK `6.6.0.0` (tools `19.0.07`).

```bash
# from Android-App/MobiInfer-Android
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk-6.6.0.0/6.6.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The inference engine lives at the top of the enclosing repository and is reached directly by relative path — there is **no** `llama.cpp-omni` submodule or symlink here, and no `git submodule update` is needed.

- Application ID / package: **`com.mobiinfer.demo`**
- Native library: `libmobiinfer.so` (host-side Hexagon driver + `libggml-htp-v79.so` DSP skeleton)

## Repository layout

| Path | Contents |
| --- | --- |
| `MobiInfer-Android/` | Android client (Kotlin + NDK) |
| `../` (repository root) | The shared inference engine (built by the client via `-DLLAMA_SRC`) |
| `docs/` | app-level benchmark data |
| `assets/` | benchmark assets |
