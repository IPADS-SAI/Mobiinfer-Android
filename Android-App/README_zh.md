# MobiInfer · Android App

[English](README.md) | **中文**

**MobiInfer-Android** 自带的端侧 **Android** App。可**完全本地**在高通 **Hexagon NPU（HTP）** 上运行大型多模态模型——无服务器，数据不出设备。

> 部署与安装见仓库根目录的 **[部署指南 / DEPLOY.md](../DEPLOY.md)**；引擎能力与架构见 **[README](../README.md)**。

---

## 当前能力

| 能力 | 状态 |
| --- | --- |
| 完全端侧推理 | ✅ 无远程服务器，全部在手机上运行 |
| **高通 Hexagon NPU（HTP）**加速 | ✅ 语言模型 **和** 视觉编码器都跑在 NPU |
| **MiniCPM-V 4.7 35B-A3B**（MoE）端侧运行 | ✅ 路由专家按需从闪存流式加载（LES） |
| **DSpark** 投机采样 | ✅ Q8_0 草稿，默认最多 3 个草稿 token（默认启用） |
| 多模态输入 | ✅ 文本对话 + 图片问答 |
| App 内 **模型管理** | ✅ 从 ModelScope 下载整包并按 `SHA256SUMS` 校验 |
| Qwen3-VL-4B（HTP） | ✅ 作为第二个模型提供 |

> 注意：部分视觉算子（如位置编码插值 `UPSCALE`）当前回落到 CPU。本分支仅以 Hexagon **v79**（已验证）为目标；v73+ 虽可构建但未在此分支验证。

## 支持的模型

本 NPU 分支在 **模型管理** 中提供两个条目（上游 demo 的其它变体已刻意不在此提供）：

| 模型 | 量化 | 体积 | 说明 |
| --- | --- | --- | --- |
| **MiniCPM-V 4.7 35B-A3B** | Q3（混合，tiled 专家） | **~17 GB** | 主线目标。MoE 专家从闪存流式加载；需 Hexagon v79 + ≥12 GB 内存 |
| **Qwen3-VL-4B-Instruct** | Q4_K_M | ~3.3 GB | 视觉语言模型；语言 + 视觉均在 HTP |

4.7 整包**不是单个 GGUF**，而是常驻主模型 + padding 过的 F16 `mmproj` + Q8_0 DSpark 草稿 + 40 层 tiled 专家文件：

```text
files/models/minicpm-v-4_7/
├── minicpm47-les-hexagon.gguf                     # 主模型（不含路由专家）
├── mmproj-MiniCPM-V-4_7-real-f16-pad4320.gguf     # 视觉编码器（F16）
├── dspark-MiniCPM-V-4_7-q8_0.gguf                 # DSpark 草稿（Q8_0，必需）
└── experts-htp-tiled/
    ├── manifest.json
    └── layer_000.bin ... layer_039.bin
```

主模型与 `experts-htp-tiled/` 必须来自同一整包；不匹配或专家文件缺失会导致加载失败。

## 环境要求

- 高通骁龙设备，**Hexagon v79** NPU
- Android 8.0 (API 26)+，**arm64-v8a**
- **≥ 12 GB 内存**（4.7 35B-A3B 已验证于 12 GB 设备）
- **≥ 17 GB 可用快速（UFS）存储**——专家按需从闪存读取
- 首次下载约 17 GB 模型建议 Wi‑Fi

## 安装与运行

预编译 APK 发布在父仓库的 [Releases](https://github.com/IPADS-SAI/Mobiinfer-Android/releases)。安装后，在 App 内 **模型管理** 下载模型，即可开始对话。

完整步骤（含手动侧载、命令行运行、性能数据、故障排查）见 **[../DEPLOY.md](../DEPLOY.md)**。

## 从源码构建

依赖：Android Studio（Giraffe+）、Android SDK + NDK `27.0.12077973`、CMake `3.22.1`、Hexagon SDK `6.6.0.0`（tools `19.0.07`）。

```bash
# 在 Android-App/MobiInfer-Android 下
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk-6.6.0.0/6.6.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

推理引擎位于父仓库根目录，各端通过相对路径（`-DLLAMA_SRC`）直接引用，**不再使用 submodule 或软链**，无需 `git submodule update`。

- Application ID / 包名：**`com.mobiinfer.demo`**
- 原生库：`libmobiinfer.so`（宿主侧 Hexagon 驱动 + `libggml-htp-v73/v75/v79/v81.so` DSP skel；可用 `-PhtpSkels=v79` 只构建指定版本）

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| 路径 | 内容 |
| --- | --- |
| `MobiInfer-Android/` | Android 客户端（Kotlin + NDK） |
| `../`（仓库根目录） | 共享推理引擎（客户端通过 `-DLLAMA_SRC` 构建） |
| `docs/` | App 级基准数据 |
| `assets/` | 基准测试素材 |
