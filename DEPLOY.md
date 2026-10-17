<div align="center">

# 部署指南 · Deployment Guide

**MobiInfer · Android — MiniCPM-V 4.7 35B-A3B on Snapdragon Hexagon NPU**

中文 | [English](#english)

</div>

本指南覆盖：**装 App → 下模型 → 跑**，以及手动侧载、命令行运行、从源码构建和多设备切分。
This guide covers **install → download → run**, plus manual sideloading, CLI usage, building from source, and multi-device split.

> 仓库结构 / Layout：`Android-App/`（Android 端侧 App）与本次的 Hexagon 引擎（`ggml/src/ggml-hexagon/`、`src/llama-moe-*`）同处本仓库。

---

## 1. 环境要求 / Requirements

| 项目 / Item | 要求 / Requirement |
|---|---|
| SoC / NPU | 高通骁龙，**Hexagon v79**（v73+ 亦可，v79 为当前验证版本） |
| 系统 / OS | Android 8.0 (API 26)+，**arm64-v8a** |
| 内存 / RAM | **≥ 12 GB**（4.7 35B-A3B 已验证于 12 GB 设备）|
| 存储 / Storage | **≥ 17 GB 可用**，且为**快速 UFS**（专家权重按需从闪存流式读取，存储带宽直接影响解码速度）|
| 网络 / Network | 首次下载模型建议 Wi‑Fi（约 17 GB）|

> 说明：MG 专家权重常驻闪存而非全部驻留内存，这是 35B 级模型能在手机上运行的关键；因此**内部存储的随机读带宽**对体验影响很大。

---

## 2. 安装 App / Install the app

**方式 A — 下载 APK（推荐）**

1. 打开 [Releases](https://github.com/IPADS-SAI/Mobiinfer-Android/releases)，下载最新 `MobiInfer-Android-<version>.apk`。
2. 在设备上点击安装；若出现安全提示，在 **设置 → 应用 → 安装未知应用** 中允许来源。
3. 或用 adb 安装：

```bash
adb install -r MobiInfer-Android-<version>.apk
```

**方式 B — 从源码构建（Android Studio）**

见 [§6 从源码构建](#6-从源码构建--build-from-source)。

---

## 3. 下载模型权重 / Download the model weights

### 3.1 App 内下载（推荐）

打开 App → **模型管理 / Model Manager** → 选择 **MiniCPM-V 4.7 35B-A3B (Q3, NPU)** → **下载 / Download**。

- 来源：ModelScope 仓库 `fengerhu1/MiniCPM-V-4.7-35B-A3B-gguf-hexagon`
- App 会下载整包并按仓库内 `SHA256SUMS` 清单**逐文件校验**。
- 整包约 **17 GB**：主模型 + mmproj + DSpark 草稿 + 40 层 tiled 专家。

> 下载完成后，**请勿单独替换或删除其中任何文件**。主模型自身不含被路由专家权重，必须与 `experts-htp-tiled/` 目录配对使用，否则加载前的磁盘一致性校验会失败。

### 3.2 手动侧载（adb）

如需离线部署，可将整包推入 App 私有目录：

```bash
adb push <本地模型目录>/minicpm-v-4_7/ \
    /data/data/com.mobiinfer.demo/files/models/minicpm-v-4_7/
```

> App 私有目录通常需 `run-as`（debug 包）或 root 才能写入。也可以先推到 `/sdcard/` 再在 App 内导入（若你的版本提供导入入口）。

期望的目录布局 / Expected layout：

```text
files/models/minicpm-v-4_7/
├── minicpm47-les-hexagon.gguf                     # 主模型（不含路由专家）
├── mmproj-MiniCPM-V-4_7-real-f16-pad4320.gguf     # 视觉编码器（F16，pad4320）
├── dspark-MiniCPM-V-4_7-q8_0.gguf                 # DSpark 草稿模型（Q8_0，必需）
└── experts-htp-tiled/
    ├── manifest.json
    └── layer_000.bin ... layer_039.bin            # 40 层 tiled 专家
```

- **主模型与 `experts-htp-tiled/` 必须来自同一整包。**
- 专家文件缺失或为空会导致加载失败。
- 侧载时可逐一核对 `SHA256SUMS`，与 App 内下载等价。

---

## 4. 运行 / Run

1. 打开 App，进入 **模型管理**，选中 **MiniCPM-V-4.7 35B-A3B (Q3, NPU)**。
2. 等待模型加载（首次加载需从闪存建立各层专家索引与常驻槽位）。
3. 进入对话，输入文字或附图提问。

**默认运行配置 / Default runtime config**（无需手动设置）：

| 参数 / Item | 默认值 / Value |
|---|---|
| HTP 会话 / HTP sessions | 2（layer split） |
| 专家常驻槽位 / slot bank | 256 |
| 上下文 / context | 2048 |
| batch / ubatch | 128 |
| 图像 token 上限 / image tokens | 64 |
| 语言模型 + 视觉编码器 | 均在 HTP（NPU） |
| 投机采样 / speculation | DSpark（Q8_0 草稿，n_max = 3，**必需**） |

**可选覆盖 / Optional overrides**：在模型目录放置 `runtime.json` 可覆盖 `vision_htp`、`dspark`、`max_tokens`、`label`，用于受控基准测试。正常 HTP + DSpark 运行无需该文件。

> 连续多轮请求的稳定性仍在评估中：如遇到偶发中断，可在请求之间重启 App。

---

## 5. 预期性能 / Expected performance

以下为 **App 页面实测**（OnePlus 13 / PJZ110，Hexagon v79，单次样本，每例重启 App 进程，未清系统文件缓存；ctx=2048、bank=256、batch=128、threads=4、image-max-tokens=64、greedy seed=42、输出上限 128，DSpark n_max=3）：

| 模式 / Mode | 用例 / Case | 图片 / Images | 首 token / TTFT (s) | 解码 / Decode (token/s) |
|---|---|---:|---:|---:|
| CPU 视觉 + HTP 目标，无草稿 | text | 0 | 2.001 | 11.85 |
| CPU 视觉 + HTP 目标，无草稿 | image | 1 | 8.395 | 11.22 |
| HTP 视觉 + HTP 目标，无草稿 | text | 0 | 1.889 | 11.79 |
| HTP 视觉 + HTP 目标，无草稿 | image | 1 | 3.781 | 11.26 |
| **HTP 视觉 + HTP 目标 + DSpark** | text | 0 | 2.132 | **13.90** |
| **HTP 视觉 + HTP 目标 + DSpark** | image | 1 | 4.187 | **12.64** |

- HTP 视觉把图片 **TTFT 从 8.395 s 降到 3.781 s（−55%）**。
- DSpark 把文字解码从 11.79 提升到 **13.90 token/s（+18%）**，图片从 11.26 提升到 **12.64 token/s（+12%）**。

---

## 6. 从源码构建 / Build from source

### 6.1 构建 App

**依赖 / Requirements**：Android Studio（Giraffe+）、Android SDK + NDK `27.0.12077973`、CMake `3.22.1`、Hexagon SDK `6.6.0.0`（tools `19.0.07`）。

```bash
# 指向本仓库内的引擎源码（App 将其作为 llama.cpp-omni 子模块 / 软链）
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export ANDROID_HOME=/path/to/Android/Sdk
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk-6.6.0.0/6.6.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07

cd Android-App/MobiInfer-Android
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

构建时需确保 CMake 传入 Hexagon 相关开关（`-DGGML_HEXAGON=ON -DPREBUILT_LIB_DIR=toolv19_v79 -DHEXAGON_SDK_ROOT=... -DHEXAGON_TOOLS_ROOT=...`），并保留 DSP skel（`libggml-htp-v79.so` 不能被 strip）。

### 6.2 独立构建 Hexagon 引擎（命令行）

```bash
# Android 预设（Hexagon + OpenCL）
cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --preset arm64-android-snapdragon-release -B build-snapdragon
cmake --build build-snapdragon -j
```

或使用骁龙工具链脚本（自动拉取 Docker 镜像、编译、安装，可选推送到设备）：

```bash
./scripts/snapdragon/build.py --target adb --push
```

### 6.3 在设备上运行引擎 / Run the engine on device

专家权重外置、常驻部分与打包专家分开存放，**两者必须一起使用**：

```sh
export LLAMA_MOE_HTP_PACKED_DIR=/models/experts-htp-tiled

./bin/llama-cli -m /models/model-les.gguf \
    --moe-experts /models/experts-htp-tiled --moe-slot-bank 32 \
    --device HTP0,HTP1 --split-mode layer --tensor-split 1,1 -ngl 99 \
    -fit off -c 2048 -b 4 -ub 4 -p 'Hello'
```

对超过单 NPU 会话地址空间（约 3.5 GB）的模型，后端会自动在推理中映射/解映射权重缓冲，单会话即可：

```sh
./scripts/snapdragon/run.py --target adb --devices HTP0:0 -- \
    llama-cli -m /models/model-les.gguf --moe-experts /models/experts-htp-tiled \
    --moe-slot-bank 32 -ngl 99 -p "Hello"
```

### 6.4 专家打包（自行准备模型时）

把 MoE 专家从 GGUF 提取、剥离并转换为 HTP tiled 布局：

```sh
python tools/moe/experts.py extract --model /models/input.gguf --out-dir /models/experts-raw
python tools/moe/experts.py verify  --model /models/input.gguf --experts /models/experts-raw
python tools/moe/strip_experts.py --model /models/input.gguf \
    --experts /models/experts-raw --out /models/model-les.gguf
python tools/moe/pack_hexagon.py \
    --experts /models/experts-raw \
    --out-dir /models/experts-htp-tiled \
    --lib-dir /path/to/matching/lib
```

详见 [`tools/moe/HEXAGON.md`](tools/moe/HEXAGON.md)。

---

## 7. 常用引擎开关 / Key engine flags

| 开关 / Flag | 作用 / Purpose |
|---|---|
| `--moe-experts PATH` | 专家目录 / 清单路径（专家主体路由的 MoE） |
| `--moe-slot-bank N` | 每层常驻路由专家槽位数 |
| `--moe-k-prefill N` | prompt eval 阶段的 top-k |
| `--moe-k-decode N` / `--moe-k-decode-cycle 8,4` | 生成阶段的 top-k（可周期切换） |
| `--device HTP0,...` | 选择 Hexagon 设备 / 会话（Hexagon 在 `-ngl` 语义上等同 GPU） |
| `--split-mode layer\|tensor` | 跨会话层切分 / 跨物理核张量切分 |
| `-ngl N` / `--load-mode` | 卸载层数 / 加载模式（`auto`/`none`/`mmap`/`mlock`/`dio`） |
| `--spec-type draft-dspark` | 启用 DSpark 投机采样 |

**多设备组合 / Multi-device split**（`--devices` / `GGML_HEXAGON_DEVICES`）：

- 虚拟会话层切分：`HTP0:0,HTP0:1`
- 物理核张量切分：`HTP0:0,HTP1:0`
- 核组行切分：`HTP0[0-1]`

其余 Hexagon 环境变量（`GGML_HEXAGON_POWER_MODE` / `GGML_HEXAGON_PROFILE` / `GGML_HEXAGON_OPFILTER` 等）见 [`docs/backend/snapdragon/README.md`](docs/backend/snapdragon/README.md)。

---

## 8. 故障排查 / Troubleshooting

| 现象 / Symptom | 原因与处理 / Cause & fix |
|---|---|
| `HTP0 failed to open session ... 0x80000406`（DSP 无法加载 skel） | APK 未把 `.so` 解压到磁盘。确保构建时启用 `packaging.jniLibs.useLegacyPackaging = true`（即 `extractNativeLibs=true`），且 `libggml-htp-v79.so` 位于 App 私有 `lib/arm64/` |
| 看不到 `libcdsprpc.so` / Hexagon 后端未注册 | `AndroidManifest.xml` 需声明 `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>`；且 `ggml-base` 编译期须带 `GGML_HEXAGON`（仅丢一个预编译 `.so` 不会自动注册） |
| 模型加载失败 / 拒绝加载 | 主模型与 `experts-htp-tiled/` 不匹配（来源不同整包），或专家文件缺失/为空 |
| 偶发中断 / DSP queue 报错 | 连续多轮可靠性仍在评估；在请求之间重启 App |
| 与 Termux 中的 HTP 会话互相阻塞（约 10 s 超时） | VTCM 争用：同一时刻只保留一个 HTP 使用方 |
| 视觉部分算子未走 NPU | 部分视觉算子（如位置编码插值 `UPSCALE`）暂回落到 CPU，属预期行为 |
| 内存不足 / OOM | 使用 `>= 12 GB` 内存设备；或在 App 内降低上下文 / 专家缓存（会变慢） |

---

<a name="english"></a>
## English

This guide covers **install → download → run**, plus manual sideloading, CLI usage, building from source, and multi-device split.

**1. Requirements** — Snapdragon with **Hexagon v79** NPU (v73+ also supported; v79 is the validated target), Android 8.0+ (`arm64-v8a`), **≥ 12 GB RAM**, **≥ 17 GB free fast UFS storage**, Wi‑Fi for the initial download.

**2. Install the app** — Download the latest `*.apk` from **Releases** and install it (allow the source in Settings if prompted), or `adb install -r MobiInfer-Android-<version>.apk`.

**3. Download the model (~17 GB)** — Open the app → **Model Manager** → select **MiniCPM-V 4.7 35B-A3B (Q3, NPU)** → **Download**. The app pulls the whole bundle from ModelScope (`fengerhu1/MiniCPM-V-4.7-35B-A3B-gguf-hexagon`) and verifies every file against `SHA256SUMS`. Keep the main model and `experts-htp-tiled/` together — a mismatch or missing expert file refuses to load.

To sideload manually, push the bundle into the app-private directory:

```bash
adb push <local>/minicpm-v-4_7/ \
    /data/data/com.mobiinfer.demo/files/models/minicpm-v-4_7/
```

**4. Run** — Select the model and chat (text or image). Defaults: 2 HTP sessions, 256-slot expert bank, ctx 2048, batch 128, image tokens 64, HTP vision, Q8_0 DSpark (n_max=3, required). A local `runtime.json` can override `vision_htp` / `dspark` / `max_tokens` / `label` for benchmarks.

**5. Performance** — Measured through the app UI (OnePlus 13 / PJZ110, Hexagon v79): HTP vision cuts image TTFT from **8.395 s → 3.781 s (−55%)**; DSpark lifts text decode **11.79 → 13.90 token/s (+18%)** and image decode **11.26 → 12.64 token/s (+12%)**.

**6. Build from source** — Requirements: Android Studio, NDK `27.0.12077973`, CMake `3.22.1`, Hexagon SDK `6.6.0.0` (tools `19.0.07`). Then `./gradlew :app:assembleDebug`. The engine can also be built standalone via the Snapdragon presets or `./scripts/snapdragon/build.py --target adb --push`. See [`tools/moe/HEXAGON.md`](tools/moe/HEXAGON.md) for packing experts from your own model.

**7. Key flags** — `--moe-experts`, `--moe-slot-bank`, `--moe-k-prefill`, `--moe-k-decode[-cycle]`, `--device HTP0,...`, `--split-mode layer|tensor`, `--spec-type draft-dspark`. Multi-device: `HTP0:0,HTP0:1` (layer split), `HTP0:0,HTP1:0` (tensor split), `HTP0[0-1]` (row split).

**8. Troubleshooting** — `0x80000406` = the `.so` was not extracted (`useLegacyPackaging = true`); missing Hexagon backend = declare `<uses-native-library libcdsprpc.so>` and compile `ggml-base` with `GGML_HEXAGON`; intermittent failures = restart the app between requests; VTCM contention = only one HTP consumer at a time.

---

## 许可 / License

遵循 MIT 许可，详见 [LICENSE](LICENSE)。
