<div align="center">

# MobiInfer · Android

### On-device Large-Model Inference on Snapdragon NPU

**在骁龙 Hexagon NPU 上跑大模型：把 35B 级 MoE 装进手机**

[![Platform](https://img.shields.io/badge/platform-Android%20%7C%20Snapdragon-3DDC84)](#)
[![Backend](https://img.shields.io/badge/backend-Hexagon%20NPU%20(HTP)-blue)](#)
[![Base](https://img.shields.io/badge/built%20on-llama.cpp--omni-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

中文 | [English](#english)

</div>

---

## 我们做了什么 / What we did

MobiInfer-Android 把 **Qualcomm Hexagon NPU（HTP/HTA DSP）后端**完整引入 [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)，并在此基础上打通了**端侧大 MoE 模型**的完整链路：专家权重常驻闪存、按需流式加载到 NPU、用前瞻调度掩盖 I/O 延迟，再叠加 **DSpark 投机采样**进一步提速。

> MobiInfer-Android ports the full **Qualcomm Hexagon NPU (HTP) backend** into [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) and builds a complete on-device path for **large sparse MoE models**: experts stay in flash, stream into the NPU on demand, I/O latency is hidden by lookahead scheduling, and an adapted **DSpark speculative sampler** pushes decode speed further.

一句话 / In one line:

> **一台骁龙手机，用 NPU 跑 35B-A3B 级 MoE 大模型，能力不缩水。**

---

## 核心亮点 / Highlights

- 🚀 **原生 Hexagon NPU 后端 / Native Hexagon NPU backend**
  从上游 llama.cpp 移植的完整 `ggml-hexagon` 实现：HVX/HMX 张量核、Flash-Attention、Gated Delta Net、DMA 队列、以及多设备切分（单设备、虚拟会话层切分、物理核张量切分、核组行切分）。模型以“GPU”语义卸载到 HTP。

- 🧠 **端侧大 MoE 专家卸载 / On-device MoE expert offload**
  专家权重外置到闪存，通过 `--moe-experts` 清单 + `--moe-slot-bank` 常驻槽位，把被路由到的专家流式加载进 HTP。支持跨 HTP 设备的 TopK 与分块专家分配/上传、池化专家 I/O、有界多行预取，以及可选的前缀缓存预热。

- ⚡ **LES 前瞻专家调度 / Lookahead Expert Scheduling**
  跨层路由预测提前把下一层可能用到的专家读进来；已命中的专家先算，未命中的并行加载。预测只用于**预取与调度**，实际计算严格遵循模型自身 token 级路由，未命中的专家照常加载使用，**无需额外训练、无损能力**。

- 🎯 **DSpark 投机采样 / DSpark speculative sampling**
  DFlash（提取多层隐状态、单步预测多 token）叠加一个半自回归 Markov 头与置信度头。已在端侧流程中适配（`speculative.type = dspark`），与专家调度协同工作。

- 📦 **专家打包工具链 / Expert packing toolchain**
  `experts.py`（提取/校验）→ `strip_experts.py`（剥离 GGUF 内重复专家）→ `pack_hexagon.py`（转换为 HTP tiled 布局，`hexagon-tiled-v1`）。打包直接调用后端的 `ggml_backend_pack_moe_expert`，保证与运行时布局一致。

- 💾 **权重存取优化 / Weight access optimization**
  逐层混合精度量化 + 专家权重连续重排，减少闪存上的分散读、提升顺序读效率，降低专家加载对端侧存储带宽的要求。

---

## 支持的模型 / Supported models

| 模型 / Model | 规模 / Scale | 说明 / Notes |
|---|---|---|
| **MiniCPM-V 4.7 35B-A3B** | 35B 总参 / 3B 激活 | 端侧 MoE 主线目标，多模态（视觉 + 文本） |
| Qwen3.5 / 3.6 MoE | 35B-A3B / 122B-A10B / 397B-A17B | `qwen35moe`，线性注意力 + gated delta net |
| DFlash / DSpark 草稿模型 | — | 投机采样（`dflash` / `dspark`） |

模型架构覆盖 `qwen35moe`、`dflash`、`minicpm` 系列，以及 llama.cpp 体系内的其余常见架构。

---

## 系统架构 / Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │                llama.cpp-omni                │
                  │  （CLI / Server / 多模态 / TTS 全链路）       │
                  └───────────────┬──────────────────────────────┘
                                  │  -ngl / --device / --split-mode
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
      ┌───────────┐        ┌───────────┐         ┌───────────┐
      │  CPU/ARM  │        │  Adreno   │         │  Hexagon  │  ← 本仓库重点
      │  (NEON)   │        │  (OpenCL) │         │  NPU/HTP  │
      └───────────┘        └───────────┘         └─────┬─────┘
                                                       │
                          ┌────────────────────────────┼───────────────────────────┐
                          ▼                            ▼                           ▼
                   ┌─────────────┐            ┌───────────────┐           ┌───────────────┐
                   │  HVX / HMX  │            │  专家流式加载  │           │  DSpark 草稿  │
                   │  张量核内核  │            │  + LES 调度   │           │  投机采样      │
                   └─────────────┘            └───────┬───────┘           └───────────────┘
                                                      │ pread (tiled, 连续布局)
                                                      ▼
                                              ┌───────────────┐
                                              │  UFS / SDD 闪存 │
                                              │  外置专家权重   │
                                              └───────────────┘
```

推理时，密集权重（attention、共享专家、嵌入）常驻 NPU；路由专家权重留在闪存，由 LES 预测并流式读取到常驻槽位（slot bank），再在 HTP 上完成专家 matvec。DSpark 草稿模型与主模型共享这套专家调度路径。

---

## 性能数据 / Performance

> 🚧 **本节待补充（Android / Snapdragon 实测）。** 下面先给出量表结构，数据回填后即可发布。
>
> 🚧 **To be filled (measured on Android / Snapdragon).** The table structure is ready; numbers will be filled in.

### 端到端解码速度 / End-to-end decode speed

| 阶段 / Stage | 解码速度 / Decode | 相比上一阶段 / vs prev | 相比基准 / vs baseline |
|---|---:|---:|---:|
| 基准：专家从闪存加载，无优化 / Baseline | — token/s | — | 1.00× |
| + LES 前瞻调度与加载优化 / Lookahead schedule + load opt | — token/s | — | — |
| + 逐层混合精度量化 / Layer-wise mixed-precision | — token/s | — | — |
| **+ DSpark 投机采样 / Full LES + DSpark** | **— token/s** | — | — |

### 首 token 延迟与内存 / TTFT & memory

| 指标 / Metric | 数值 / Value | 说明 / Notes |
|---|---:|---|
| Time to First Token (TTFT) | — ms | 端侧 MoE |
| 常驻内存 / Resident memory | — GB | LLM 密集部分 + 槽位 |
| 专家目录大小 / Expert store size | — GB | 量化后 |
| Prefill 吞吐 / Prefill throughput | — token/s | prompt eval |

---

## 快速开始 / Quick Start

三步：**装 App → 下模型 → 开跑**。App 默认就是 NPU 最优配置（双 HTP 会话 + HTP 视觉 + Q8_0 DSpark 投机采样），开箱即用。

> Three steps: **install the app → download the model → run.** The app ships with the optimal NPU defaults (dual HTP sessions + HTP vision + Q8_0 DSpark speculation) — no extra setup.

### 1. 安装 App / Install the app

从 [Releases](https://github.com/IPADS-SAI/Mobiinfer-Android/releases) 下载最新 Android APK 并安装（首次需在「设置 → 应用 → 安装未知应用」中允许来源）。

> Download the latest Android APK from **Releases** and install it. On first install, allow installation from this source.

```bash
adb install -r MobiInfer-Android-<version>.apk
```

### 2. 下载模型权重 / Download the model weights (~17 GB)

打开 App → **模型管理 / Model Manager** → 选择 **MiniCPM-V 4.7 35B-A3B (Q3, NPU)** → **下载 / Download**。App 会从 ModelScope 拉取整包（主模型 + mmproj + DSpark 草稿 + 40 层 tiled 专家，约 **17 GB**），并按 `SHA256SUMS` 逐一校验。建议 Wi‑Fi，预留约 17 GB 可用空间。

> Open the app → **Model Manager** → select **MiniCPM-V 4.7 35B-A3B (Q3, NPU)** → **Download**. The app pulls the full bundle (~17 GB) from ModelScope and verifies every file against `SHA256SUMS`. Use Wi‑Fi and keep ~17 GB free.

### 3. 开跑 / Run

下载完成后，在 **模型管理** 中选中该模型进入对话，直接输入文字，或附图提问。

> Select the model and start chatting — text or image.

> 需要手动侧载、命令行运行、从源码构建、多设备切分或排查问题？见 **[部署指南 / DEPLOY.md](DEPLOY.md)** 与骁龙后端文档 [`docs/backend/snapdragon/README.md`](docs/backend/snapdragon/README.md)。
>
> For manual sideloading, CLI usage, building from source, multi-device split, or troubleshooting, see **[DEPLOY.md](DEPLOY.md)** and [`docs/backend/snapdragon/README.md`](docs/backend/snapdragon/README.md).

---

## 常用开关 / Key flags

| 开关 / Flag | 作用 / Purpose |
|---|---|
| `--moe-experts PATH` | 专家目录 / 清单路径（专家主体路由的 MoE） |
| `--moe-slot-bank N` | 每层常驻路由专家槽位数 |
| `--moe-k-prefill N` | prompt eval 阶段的 top-k |
| `--moe-k-decode N` / `--moe-k-decode-cycle 8,4` | 生成阶段的 top-k（可周期切换） |
| `--device HTP0,...` | 选择 Hexagon 设备 / 会话（Hexagon 在 `-ngl` 语义上等同 GPU） |
| `--split-mode layer\|tensor` | 跨会话层切分 / 跨物理核张量切分 |
| `-ngl N` / `--load-mode` | 卸载层数 / 加载模式（`auto`/`none`/`mmap`/`mlock`/`dio`） |
| `speculative.type = dspark` | 启用 DSpark 投机采样 |

Hexagon 环境变量（`GGML_HEXAGON_DEVICES` / `GGML_HEXAGON_POWER_MODE` / `GGML_HEXAGON_PROFILE` / `GGML_HEXAGON_OPFILTER` 等）见骁龙后端文档。

---

## 仓库结构 / Layout

```
ggml/src/ggml-hexagon/     Hexagon 后端宿主侧（调度、多设备、DMA）+ htp/（DSP 侧 HVX/HMX 内核）
src/llama-moe-slot.*       MoE 专家槽位运行时：预取、调度、I/O 线程池、诊断计数
src/llama-moe-experts.*    外置专家清单与加载
src/models/qwen35moe.cpp   Qwen3.5/3.6 MoE（含 MiniCPM-V 4.7 35B-A3B）
src/models/dflash.cpp      DFlash / DSpark 草稿模型
common/speculative.cpp     投机采样（DSpark 适配）
tools/moe/                 专家提取 / 剥离 / 打包工具
docs/backend/snapdragon/   骁龙构建、运行、多设备文档
docs/les/                  LES 前瞻专家调度说明（中英）
```

---

## 与上游的关系 / Relation to upstream

本仓库基于 [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)（其自身基于 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)）。Hexagon 后端与相关 op/API 变更从上游 llama.cpp 移植；MoE 专家卸载、LES 调度与 DSpark 适配为本项目在端侧方向的扩展。

MobiInfer-Android is built on [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) (itself built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)). The Hexagon backend and supporting op/API changes are ported from upstream llama.cpp; MoE expert offload, LES scheduling, and DSpark integration are this project's on-device extensions.

---

## 致谢与许可 / Credits & License

- [llama.cpp](https://github.com/ggml-org/llama.cpp) 与 [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [Qualcomm Snapdragon Toolchain](https://github.com/snapdragon-toolchain)
- MiniCPM-V 系列模型 / MiniCPM-V model family

遵循 MIT 许可，详见 [LICENSE](LICENSE)。

---

<a name="english"></a>
## English

**MobiInfer-Android** brings a full **Qualcomm Hexagon NPU (HTP) backend** to [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) and turns it into a practical engine for **large sparse MoE models on a phone**.

- **Native Hexagon backend** — HVX/HMX tensor kernels, Flash-Attention, Gated Delta Net, DMA queues, and multi-device partitioning (single session, virtual-session layer split, physical-core tensor split, core-group row split). Models offload to the HTP with normal "GPU" semantics (`-ngl`, `--device`, `--split-mode`).
- **On-device MoE expert offload** — expert weights live in flash and stream into the NPU via an expert-major manifest (`--moe-experts`) and resident slots (`--moe-slot-bank`). Includes cross-device TopK and tiled expert allocation/upload, pooled expert I/O, bounded multi-row prefetch, and optional prefill cache warming.
- **LES (Lookahead Expert Scheduling)** — cross-layer routing prediction prefetches the next layer's likely experts; loaded ones compute first, the rest load in parallel. Prediction is used only for prefetch/scheduling — compute still follows the model's own token-level routing, missed experts are loaded normally, and **no extra training or capability loss** is involved.
- **DSpark speculative sampling** — DFlash (multi-layer hidden-state extraction, multi-token single-step prediction) plus a semi-autoregressive Markov head and a confidence head, adapted into the on-device pipeline (`speculative.type = dspark`).
- **Expert packing toolchain** — `experts.py` (extract/verify) → `strip_experts.py` (strip duplicated experts from the GGUF) → `pack_hexagon.py` (convert to the HTP `hexagon-tiled-v1` layout through the backend's `ggml_backend_pack_moe_expert` API, so runtime layout always matches).
- **Weight access optimization** — layer-wise mixed-precision quantization plus contiguous expert repacking to lower flash bandwidth demands.

**Supported models:** MiniCPM-V 4.7 35B-A3B, Qwen3.5/3.6 MoE (35B-A3B / 122B-A10B / 397B-A17B), and DFlash/DSpark draft models.

**Build & run on Snapdragon:**

```sh
./scripts/snapdragon/build.py --target adb --push        # build + deploy to a connected device

export LLAMA_MOE_HTP_PACKED_DIR=/models/experts-htp-tiled
./bin/llama-cli -m /models/model-les.gguf \
    --moe-experts /models/experts-htp-tiled --moe-slot-bank 32 \
    --device HTP0,HTP1 --split-mode layer --tensor-split 1,1 -ngl 99 \
    -fit off -c 2048 -b 4 -ub 4 -p 'Hello'
```

See [`tools/moe/HEXAGON.md`](tools/moe/HEXAGON.md) for the expert preparation pipeline and [`docs/backend/snapdragon/README.md`](docs/backend/snapdragon/README.md) for all build/run/multi-device options.
