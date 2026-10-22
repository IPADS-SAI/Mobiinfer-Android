# HANDOFF — Qwen3-VL-4B on Qualcomm Hexagon (NPU) in the MiniCPM-V Android app

日期：2026-10-15

## Qwen 视觉默认 HTP（2026-10-15 14:16）

- 按用户要求，loadMmproj按general.architecture=qwen3vl默认启用mparams.use_gpu，mtmd日志接入Android回调。未修改后端子模块。
- assembleDebug成功，已无线ADB install -r并启动。实际日志：Vision backend request architecture=qwen3vl backend=HTP；CLIP using HTP0 backend。
- 启动时唯一列出的不支持算子：UPSCALE f32 [92,92,1024,1]，来源Qwen3-VL resize_position_embeddings的插值；由CPU回落执行。不可声称全视觉算子都在DSP。
- App实测224×224图 -> Red square and blue circle；清空后1280×610图 -> 识别1869年法军俄国战役地图、损失色带及温度曲线。证明图片能运行，未证明读数/OCR全面准确或性能提升比例。
- /tmp/qwen-htp-build.log、/tmp/qwen-htp-small.log、/tmp/qwen-htp-verified.log。手机留在Qwen大图回答页面，默认配置已安装；MiniCPM4.7原有HTP路径保留。

## Qwen3-VL-4B 图片复核（2026-10-15）

- 手机当前通过无线ADB连接，USB serial暂不可用；模型管理选中qwen3-vl-4b-instruct。
- 当前已安装App无需改代码即可通过页面完成：224×224测试图 -> Red square, blue circle；同一对话追问右侧颜色 -> Blue。
- 清空后重新附加用户1280×610图，Qwen识别出拿破仑俄国战役地图、兵力色带和底部温度曲线；色带含义回答存在不准确，不可声称图表问答质量通过。
- 起初页面有“未提供图像”回答，本次新会话重新附图没有复现。图片实际预填充日志与问题提交日志image=yes可确认；原失败原因仍未确认。
- Qwen主模型为HTP0，视觉编码mparams.use_gpu=false仍为CPU，与MiniCPM4.7 HTP视觉不同。
- 自动化脚本长文本导致输入框变高，旧坐标可能点不到发送；需输入文字后重新dump再点btn_send。一次误将未发送等待当作预填充耗时，已向用户纠正。
- 本次仅运行诊断，未改推理实现；证据/tmp/qwen-before.log、/tmp/qwen-image-verified.log。手机留在Qwen大图回答页面。

## 最新状态（2026-10-15，HTP 视觉 + DSpark）

- Android 4.7 改为进程内复用当前分支 llama-cli 的 server_context，启用 HTP 视觉和 Q8_0 draft-dspark（n_max=3）；bank=256、batch=128、ctx=2048、image tokens=64。
- 已构建、安装，App 页面实际完成文字及图片请求。三种配置各一条文字、一条图片，图片均识别红方块/蓝圆形，DSpark 文字接受77/147、图片77/150草稿 token。
- 首次请求对照：CPU视觉图片TTFT8.395s，HTP3.781s；DSpark文字13.90token/s（无草稿11.79），图片12.64（无草稿11.26）。每项新App进程、单次样本，系统文件缓存未清理。详见 docs/minicpm47-app-benchmark.md 及 JSON。
- 仍需排查连续多轮可靠性：早先第三次文字请求缺少App完成记录，伴随DSP queue报错；不得据单次成功声称稳定。多模态不支持SLOT_ERASE，清空改为下一请求禁用缓存前缀复用，由server线程重置主/草稿状态。
- 启动曾因设备列表缺少nullptr终止符崩溃，已修复；当前构建通过。llama-cli本身支持--spec-type draft-dspark，不必使用独立llama-speculative示例。

## 此前状态（保留排查历史）

### 2026-10-15 13:04

- `useLegacyPackaging=true` 已验证生效：App 可创建 HTP session；Qwen3-VL 页面文字推理返回 Paris。
- 已新增 Android 本地模型条目 `minicpm-v-4_7`，并将当前使用的混合 Q3 主模型、pad4320 mmproj、40 层 tiled 专家、Q8_0 DSpark 草稿复制到手机 App 私有目录 `files/models/minicpm-v-4_7/`。45 个文件逐一 SHA-256 对比通过，App UID/SELinux 权限已设好；Termux 原文件保留。
- 4.7 JNI 加载已接入 `moe_experts_path`、`LLAMA_MOE_HTP_PACKED_DIR` 和双 HTP layer split，slot bank=64、decode window=64、prefetch depth=4、batch/ubatch=8、ctx=2048、image-max-tokens=64。视觉编码器暂走 CPU。
- Kotlin 检查 manifest 和 40 层文件存在、可读、非空；本地条目不展示下载按钮。4.7 图片预填充先打开 user turn，问题接在图像后；清空对话重建上下文保留模型对应 ctx/batch。
- 已 `:app:assembleDebug` 成功，`adb install -r` 安装。页面实际选中 4.7：日志 `pack_layers=40/40`，HTP0/HTP1 KV 和 RS buffer 创建成功。
- 页面实测：法国首都 -> `Paris`；上传红色方块/蓝色圆形图片 -> `Red square, blue circle`；追问右边形状 -> `Circle`。清空对话后上传图片的流程通过。
- DSpark 草稿仅已复制，**未接入投机解码**；当前成功路径是主模型常规解码。图片路径有 `find_slot: non-consecutive token position` 警告，以上简单图和多轮问答通过，复杂图片及长上下文未验证。
- 证据：`/tmp/minicpm47-app-load.log`、`/tmp/minicpm47-text.log`、`/tmp/minicpm47-vision.log`、`/tmp/minicpm47-app-success.png`。手机保留在 4.7 图片问答页面。

Worktree：`/home/yydh/MiniCPM-V-Apps-NPU`（分支 `qwen3-vl-npu`）
主仓库：`/home/yydh/MiniCPM-V-Apps`（分支 `main`）
推理后端：`/home/yydh/llama.cpp-omni-hexagon`（分支 `hexagon-minicpm-dspark`，remote `yydhYYDH/llama.cpp-omni`）
目标设备：adb `3B1F6PE5MS116NYH`（还有一条无线 adb），型号 `PJZ110`，CDSP v79，已 root（Magisk，`su -c` 可用）

---

## 1. 目标

把 Qwen3-VL-4B-Instruct 跑起来，最终**通过 Android app 页面**完成推理，并让计算走 Qualcomm Hexagon NPU（HTP）。

背景：Android app 原本不支持 Qwen3-VL，也没有任何 NPU 后端。本分支新增了 Qwen3-VL 模型条目（下载/加载/多模态链路），并接入 `ggml-hexagon` 后端。

---

## 2. 已完成

### 2.1 工具链 / 环境
- 装了缺的 SDK 组件（`/home/yydh/Android/Sdk`）：
  - NDK `27.0.12077973`（项目要求）、CMake `4.1.2`（项目要求）
  - `platforms;android-36.1`、`build-tools;36.0.0`
- `local.properties`（worktree 内，gitignored）：`sdk.dir=/home/yydh/Android/Sdk`
- **不要删 NDK `25.2.9519653`**：被 `/home/yydh/ClawMate-Android` 和 `/home/yydh/clawmate-task-ui` 引用。
- Hexagon SDK（本地、非 Docker）：`/home/yydh/third/hexagon-sdk-6.6.0.0/6.6.0.0`，tools `19.0.07`。
  - 注意：shell 里已存在 `HEXAGON_SDK_ROOT=/home/yydh/hexagon-tutorial/tools/hexagon-sdk`（另一套，skills 用），构建前需覆盖成 `third/...` 那套。
- 独立编 hexagon 后端时用 conda `et` 的 cmake 3.31.10（系统 cmake 3.22.1 太低，子构建要求 ≥3.22.2）：
  `/home/yydh/miniconda3/envs/et/lib/python3.12/site-packages/cmake/data/bin/cmake`

### 2.2 llama 源码接入
- worktree 的 `llama.cpp-omni/` 是软链 → `/home/yydh/llama.cpp-omni-hexagon`（本地 `T` typechange，**别提交**）。
- 该树 git log 关键提交：`3c9cfbd` 移植 Hexagon 后端、`55e48ff` MiniCPM MoE offload、`7b7dccc` 专家打包工具。
- 确认该树支持 Qwen3-VL：`PROJECTOR_TYPE_QWEN3VL`、`tools/mtmd/models/qwen3vl.cpp`。

### 2.3 模型
- 已用 app 内下载器从 ModelScope（`Qwen/Qwen3-VL-4B-Instruct-GGUF`）下好，落在 app 私有目录：
  `/data/user/0/com.mobiinfer.demo/files/models/qwen3-vl-4b-instruct/`
  - `Qwen3VL-4B-Instruct-Q4_K_M.gguf`（2 497 281 664 B，对上）
  - `mmproj-Qwen3VL-4B-Instruct-F16.gguf`（836 180 256 B，对上）
- 该条目**没填 MD5**，所以下载完不校验；加载也只查存在性、不校验 MD5。

### 2.4 app 改动（worktree，均未提交）
- `ModelInfo.kt`：新增 `qwen3-vl-4b-instruct` 条目；`hasHfMsSources` → `hasRemoteSources`（支持单源）。
- `LlamaEngine.kt`：下载源改为 HF/MS 各自独立，允许只有 ModelScope 单源。
- `res/values{,-en}/strings.xml`：`model_desc_qwen3vl_4b`。
- `app/src/main/cpp/llama_jni.cpp`：
  - `init()` 里 `setenv("ADSP_LIBRARY_PATH", nativeLibDir, 1)`（FastRPC 找 skel）。
  - `load()` 里 `model_params.n_gpu_layers = 99`（offload 到 HTP；HTP 对 `-ngl` 表现为 GPU device）。
- `app/build.gradle.kts`：
  - externalNativeBuild cmake 增 `-DGGML_HEXAGON=ON -DPREBUILT_LIB_DIR=toolv19_v79 -DHEXAGON_SDK_ROOT=... -DHEXAGON_TOOLS_ROOT=...`；
  - env `KLEIDIAI_SRC` 存在时透传 `-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD=...`（离线跳过 KleidiAI 下载）；
  - `packaging.jniLibs.keepDebugSymbols += "**/libggml-htp-*.so"`（DSP skel 不能 strip）；
  - `packaging.jniLibs.useLegacyPackaging = true`（**关键，见 §4**）。
- `AndroidManifest.xml`：`<uses-native-library android:name="libcdsprpc.so" android:required="false"/>`（否则 app linker namespace 看不到 `/vendor/lib64/libcdsprpc.so`）。

### 2.5 独立编出的 hexagon 产物（app 里 host 侧已由 gradle 编，skel 用预编译）
- host：`/home/yydh/llama.cpp-omni-hexagon/build-hexagon-app/bin/libggml-hexagon.so`（arm64）
- skel：`.../build-hexagon-app/ggml/src/ggml-hexagon/libggml-htp-v79.so`（QUALCOMM DSP6）
- 该 skel 已拷进 `app/src/main/jniLibs/arm64-v8a/`（CMake 的 ExternalProject 产物不会被 gradle 自动打包，所以 skel 靠 jniLibs 带入）。

---

## 3. 验证过的结果

- App 页面加载 Qwen3-VL **成功**（模型 4.02B q4_K；mmproj `Vision: yes`；ctx 4096；KV 576 MiB），**CPU** 推理正常：输入 “capital_of_France_is” 正确回答 “The capital of France is **Paris**.”。
- 开启 hexagon 后：
  - 后端注册：`Backend count: 2`（CPU + Hexagon）。
  - `libcdsprpc.so` 加载成功（加 `uses-native-library` 后）。
  - 识别并选中 `HTP0 (Hexagon)`。
  - **但打开 session 失败**：`HTP0 failed to open session : uri file:///libggml-htp-v79.so?... error 0x80000406`。

---

## 4. 当前卡点与修复（未验证）

`0x80000406` = `AEE_EUNABLETOLOAD`（`AEE_EOFFSET 0x80000400 + 0x006`），DSP 无法加载 skel 文件。

根因：app `extractNativeLibs=false`，`nativeLibraryDir`（`/data/app/.../lib/arm64/`）**是空目录**，.so 只存在于 APK 内。FastRPC 需要磁盘上的真实文件，故找不到 `libggml-htp-v79.so`。

已在 `build.gradle.kts` 加 `packaging.jniLibs.useLegacyPackaging = true`（= `extractNativeLibs=true`）。**尚未重编验证。**

其他排查结论（供参考）：
- hexagon 后端只导出 `ggml_backend_hexagon_reg`，没有 `ggml_backend_init`/`ggml_backend_score`，所以**不能**靠丢一个预编译 `libggml-hexagon.so` 触发自动加载；必须让 `ggml-base` 编译期带 `GGML_HEXAGON`（`ggml-backend-reg.cpp:149-150` 静态注册）。
- 后端会调 `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`（允许未签名 skel），该步未报错。
- 设备另一套可用的 MiniCPM 4.7 HTP 环境在 termux `~/csm/moe`（`lib/`、`experts-htp-tiled/`、`run-*.sh`），通过 `LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:$PWD/lib`、`ADSP_LIBRARY_PATH=$PWD/lib` 运行。**注意 VTCM 争用**：它和 app 的 HTP 会话不能同时跑，否则互相阻塞（约 10s 超时）。
- 设备 termux 可用 `run-as com.termux` 或 `su -c` 访问内部文件。

---

## 5. 下一步

1. **重编 + 安装 + 验证**（应该只需 repackage）：
   ```bash
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
   export ANDROID_HOME=/home/yydh/Android/Sdk
   export HEXAGON_SDK_ROOT=/home/yydh/third/hexagon-sdk-6.6.0.0/6.6.0.0
   export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07
   export KLEIDIAI_SRC=/home/yydh/MiniCPM-V-Apps-NPU/MobiInfer-Android/app/.cxx/Release/2n1w69yz/arm64-v8a/_deps/kleidiai_download-src
   cd /home/yydh/MiniCPM-V-Apps-NPU/MobiInfer-Android
   ./gradlew :app:assembleDebug
   adb -s 3B1F6PE5MS116NYH install -r app/build/outputs/apk/debug/app-debug.apk
   adb -s 3B1F6PE5MS116NYH shell am force-stop com.mobiinfer.demo
   adb -s 3B1F6PE5MS116NYH logcat -c
   adb -s 3B1F6PE5MS116NYH shell am start -n com.mobiinfer.demo/com.example.minicpmv_v_demo.MainActivity  # 修正为实际组件
   # 期望日志：ggml-hex: HTP0 new session ... ；load_tensors 出现 HTP0 model buffer size
   ```
   成功判据：logcat 出现 `new session`（session-id/domain-id），以及 `load_tensors: HTP0 model buffer size = ...`（不再是纯 CPU_Mapped/CPU_REPACK）。
   若仍 `0x80000406`：确认 `lib/arm64/libggml-htp-v79.so` 已落到磁盘；否则改用「asset 拷到可读目录 + 设 ADSP_LIBRARY_PATH」的方案。

2. **内存/设备配置**：Qwen3-VL 4B q4 ≈ 2.4 GiB，HTP 单 session ~2 GiB（文档称 >3.5 GB 才启用动态映射）。若 OOM/装不下，可用多 session（`GGML_HEXAGON_DEVICES=HTP0,HTP1` + `--split-mode layer`），app 里需要给 `llama_model_params` 配 `devices`/`tensor_split` 或 setenv。

3. **排查多模态问题**：用户反馈「输入图片后多模态推理有问题」，尚未定位。怀疑点：
   - app 的 `processUserPrompt`/`prefillImage` 用的是 MiniCPM-V 的 ChatML 手写格式化与 `assistant_turn_prefix()`（基于 `g_minicpmv_version`，Qwen3-VL 时为 0）；
   - Qwen3-VL 的图像 token/占位符格式与 MiniCPM-V 不同，需确认 `mtmd` 的 Qwen3-VL 模板路径被正确走到、以及图片 token 预算/切片设置。
   复现：app 里选 Qwen3-VL，发图 + 提问，抓 `minicpm-v` tag 的 logcat。

---

## 6. 关键命令备忘

- 设备截图：`adb -s 3B1F6PE5MS116NYH exec-out screencap -p > s.png`
- UI 坐标：`adb -s 3B1F6PE5MS116NYH shell uiautomator dump /sdcard/u.xml && adb shell cat /sdcard/u.xml`
- app 私有模型目录：`adb shell su -c 'ls -l /data/user/0/com.mobiinfer.demo/files/models/qwen3-vl-4b-instruct/'`
- 独立编 hexagon（如需重编）：
  ```bash
  CM=/home/yydh/miniconda3/envs/et/lib/python3.12/site-packages/cmake/data/bin/cmake
  cd /home/yydh/llama.cpp-omni-hexagon
  $CM -S . -B build-hexagon-app -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=/home/yydh/Android/Sdk/ndk/27.0.12077973/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_HEXAGON=ON -DPREBUILT_LIB_DIR=toolv19_v79 \
    -DHEXAGON_SDK_ROOT=/home/yydh/third/hexagon-sdk-6.6.0.0/6.6.0.0 \
    -DHEXAGON_TOOLS_ROOT=/home/yydh/third/hexagon-sdk-6.6.0.0/6.6.0.0/tools/HEXAGON_Tools/19.0.07 \
    -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
  $CM --build build-hexagon-app --target ggml-hexagon htp-v79 -j $(nproc)
  ```

---

## 7. 已知坑（本会话踩过）

- SDK `hexagon_fun.cmake` 顶部 `string(FIND ${PREBUILT_LIB_DIR} ...)`：`PREBUILT_LIB_DIR` 为空会直接 configure 失败 → 传 `-DPREBUILT_LIB_DIR=toolv19_v79`。
- Gradle 加了新 cmake 参数会换 `.cxx` 目录哈希，导致 KleidiAI 重新联网下载（代理 TLS 失败）→ 用 `KLEIDIAI_SRC` 指向已缓存源码。
- `libcdsprpc.so` 在 `/vendor/etc/public.libraries.txt` 中，但 app 仍需 `<uses-native-library>` 声明才能 dlopen。
- hexagon skel 是 DSP6 ELF，不能被 NDK strip（用 `keepDebugSymbols`）。
- `llama.cpp-omni` 是软链到 worktree，`git status` 会显示 `T`，别 stage。
