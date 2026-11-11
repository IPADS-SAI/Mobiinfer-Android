# bench/app/ — App 内推理 bench（与 `bench/bench.sh` 同口径）

在 **MobiInfer Android app** 里跑同一条 prompt，采集 app 自己写出的
`files/models/<model>/benchmark.jsonl`（`wall_ms` / `ttft_ms` / `prompt_*` /
`predicted_*`）和 UI 指标，用来回答两个问题：

1. 同一个 prompt，**App 前台** 与 **Termux 前台 `llama-cli`** 的 decode/prefill 有没有区别；
2. 区别里有多少来自 **引擎配置**（window / prefetch / ctx），多少来自 **App 前端**
   （server_context + slot、每 token 流式渲染、多轮 prompt cache）。

## 1. 用法

```sh
# 需要 adb 连到手机，app 已安装、模型已就位
cd bench/app
./app_bench.sh -t rain -r 3 -f ../prompts/rain.txt

# 或直接给 prompt（单行；不能带双引号/换行）
./app_bench.sh -t rain -r 3 -p "Explain how rain forms in clear simple English."
```

输出（`runs/<tag>-<YYYYmmdd-HHMMSS>/`）：

| 文件 | 内容 |
|---|---|
| `results.tsv` | 每条 run 一行：`wall_ms` `ttft_ms` `prompt_n` `prompt_ms` `decode_n` `decode_ms` `decode_tps` `prefill_tps` `chars` |
| `runN.record.json` | app `benchmark.jsonl` 里该 run 的完整记录（含 response） |
| `runN.benchmark.jsonl` | 该 run 结束时 app 侧文件快照 |
| `runN.png` / `runN.*.xml` | 结束时截图 / 输入框与发送键的 UI dump（排障用） |

## 1.1 设备独占锁与清理

脚本在主机上通过 `ssh $PHONE_SSH`（默认 `vivo`）调用手机上的 `~/csm/moe-exp/devlock`，
取得和 CLI 实验**同一个** flock。默认 `DEV_LOCK=on`：拿不到锁/ssh 不可达直接退出；
`DEV_LOCK=off` 在真实设备运行中不再允许（会直接报错），因为 App 一定会碰 DSP。

启动前会核对 adb 目标与 `$PHONE_SSH` 的 `ro.product.model` + `ro.build.fingerprint`，
不一致立即失败（`SKIP_DEVICE_CHECK=1` 可跳过）；持锁期间主机每 30 s 续一次心跳，
主机失联 3 分钟后锁自动释放（避免永久占锁）。

退出（正常/失败/Ctrl-C）时只 `force-stop` **本次启动**的 App（不会误杀别的持锁进程），
确认 App 停止后才释放锁（`am force-stop` 后会轮询 `pidof` 确认进程消失）；停不下来时
**保留锁**并打印恢复命令。运行期间每 3 s 检查一次持锁进程，掉了就中止该轮并停 App。
没有“保留 App 但释放锁”的开关。相关 env：`PHONE_SSH`、
`DEV_LOCK`、`LOCK_WAIT`、`SKIP_DEVICE_CHECK`。

主机若被 SIGKILL/断网，心跳会变旧，但持有端**不会**在 App 可能仍在跑时自动释放锁
（避免 DSP 被第二个实验并发使用）。恢复步骤：

```sh
adb shell am force-stop com.mobiinfer.demo     # 先确认 App 已停
ssh vivo 'cd ~/csm/moe-exp && touch .appbench-<token>.release'   # 只结束本次 holder
```

**不要** `rm .device.lock` 或全局 `pkill`：等待者可能还阻塞在旧 inode 上，
删文件/乱杀会破坏 flock 协议。holder 无响应时先确认 App 已停，再重启手机兜底。

## 2. 与 `bench/bench.sh` 的口径对照

| 项 | `bench.sh`（Termux CLI） | app（`MiniCPM47Runtime`） |
|---|---|---|
| prompt 文本 | `-p` / `-f`（默认 rain 句） | 同一段文本；`bench/prompts/rain.txt` 两边共用 |
| chat template | `--single-turn --reasoning off`，无 system prompt | 同一份 gguf 模板，无 system prompt；`enable_thinking` 由 UI「深度思考」决定 |
| prompt token | rain = **22**（`prompt eval ... / 22 tokens`） | rain = **22**（`benchmark.jsonl.prompt_n`） |
| 采样 | `--seed 42 --temp 0` | `seed 42, temp 0`（`minicpm47_runtime.cpp`） |
| 投机 | `--spec-type draft-dspark -n_max 3` | DSpark，`n_max=3` |
| 生成长度 | `-n 64/128` | 固定 `n_predict=1024`（遇 EOG 提前停） |
| 上下文 | `-c 2048` | `n_ctx=4096` |
| 专家窗口 | `WINDOW=80`（v3 仍是 80） | 默认 80（10200 MB → 88 slot = 80 窗口 + 8 预读）→ `LLAMA_MOE_DECODE_WINDOW` |
| 预读 | `PREFETCH=1 DEPTH=8 READ=8` | `prefetch=8`（= depth；READ 未设时跟随 depth） |
| IO 线程 | v3 默认 `IO_THREADS=8`（v2 是 4） | 默认 8（= v3；`runtime.json` 可覆盖） |
| `READY/REFRESH` | v3 bench 默认 `1/1`（21.9 t/s）；产品臂 `0/0`（20.7–20.9） | 默认 `1/1`（= v3 最快臂，人类要求） |
| bank / ubatch / batch | 256 / 128 / 512 | 256 / 128 / 512 |
| 每次 run 的会话 | 单轮、无历史 | **force-stop 重启 → 新会话**，保证第一轮 prompt 与 CLI 一致 |
| 计时口径 | `Generation:` / `Prompt:` = 原生 eval time | `predicted_n/predicted_ms`、`prompt_n/prompt_ms` = 同一套 server timings |
| 额外 | 无 | prompt cache（跨轮）、prefill 走 `media-embd-batch`、UI 每 token 渲染 |

结论：**prompt 本身没有区别**（同一模板、无 system prompt，rain 都是 22 token）。
可比性差异在「窗口/预读/ctx」这些引擎配置，以及 app 的 server_context/media-embd 前端路径。

## 3. 把 app 配置对齐到 CLI（可选）

app 从模型目录下的 `runtime.json` 读配置（`minicpm47_runtime.cpp`）。脚本支持：

```sh
RUNTIME_JSON=/path/to/runtime.json ./app_bench.sh -t rain-cli-parity -r 3 -f ../prompts/rain.txt
```

`runtime.json` 常用键（app 的 `cache_slots` 就是 `LLAMA_MOE_DECODE_WINDOW`，所以
CLI `WINDOW=80` ↔ `cache_slots: 80`）：

```json
{"bank": 256, "ubatch": 128, "batch": 512,
 "cache_slots": 80, "prefetch": 8, "io_threads": 8, "draft_n_max": 3,
 "ready": 1, "refresh": 1}
```

注意：

- `cache_slots` 允许 8..128，且不再按内存预算裁剪；`80 + 8` 个 slot 在 15 GB 手机上
  可能顶到 HTP mapping 上限，加载失败就退回默认（删掉 `runtime.json`）。
- 脚本会把**原有** `runtime.json` 备份到 run 目录的 `runtime.json.orig`；
  默认没有该文件，测完删掉即可回到默认配置。
- `vision_htp` / `dspark` / `max_tokens` 也能改，但会改变模型行为，bench 不建议动。

## 4. 约束

- **一次只跑一个 HTP 用户**：app 或 `llama-cli` 只能二选一。CLI 侧必须先
  `am force-stop com.mobiinfer.demo`，并按 `bench/devlock` 串行化。
- app 的 decode 时间从 `request_start` 起算，**模型加载不计入**；每条 run 都冷启动，
  所以加载时间也不会混进 `wall_ms`。
- Termux 能否开 DSP 取决于设备和当次启动：vivo V2546A 在 `runas_app` 域下
  `/dev/adsprpc-smd` 不可见时会报 `HTP0 failed to open session ... 0x72`
  （见 `AGENTS.md`）；此时 CLI 侧跑不了，只能测 app 侧。
- 长 prompt 用 `-f`；脚本按词分块（尽量 ≤40 字符、不拆单词）`input text`，避免 IME 丢字和切断 `%s`。
