# front-app/ —— 前台 App 对后台 LLM 生成的干扰测量

量化「**前台 App 在跑时，后台 `llama-cli`（MoE on HTP）的 decode/prefill 被干扰多少，
干扰主要来自 UFS 还是 CPU/内存带宽**」。

设备基线：vivo V2546A（SM8850 / Android 16 / 无 root）。其它 Android 设备可用，
但前台包名、坐标、启动路径需要按机型调整。

---

## 1. 能力边界

| 问题 | 能否回答 | 依据 |
|---|---|---|
| App 把 LLM 拖慢了多少（tok/s） | ✅ | B/C 两组 `Generation: t/s` |
| 干扰源是不是 **UFS / 块层** | ✅ | 设备 `iowait` 上升；可选 `iotrace` 看单次读延迟 |
| 干扰源是不是 **LPDDR 带宽** | ⚠️ 只能间接判断 | 无 root 时 PMU 被 SELinux 挡住；需用 `skills/profile` 的 qtimer skel 比较 DSP 计算时长 |
| App 自己读了多少字节 | ❌ | `/proc/<pid>/io` 对 app uid 关闭 |
| App 自己的内存带宽 | ❌ | 同上，`perf_event_open` 返回 `EACCES` |

结论只能到「哪一类资源」，不能给带宽绝对值。

> **方法学前提**：同一 CDSP 同时只允许一个 HTP 使用者。本工具用 `devlock` 串行化
> `llama-cli`；前台 App 与 llama 的冲突点在共享的 UFS / LPDDR / CPU 调度，
> 这正是要测的东西。见 [`skills/profile/SKILL.md`](../../skills/profile/SKILL.md)。

---

## 2. 目录

```
tools/front-app/
├── shared/
│   ├── lib.sh             # adb/ssh 封装、编译并推送设备侧工具、启动并等 RSS 稳定、采样
│   ├── with_llm.sh        # 通用 B/C 交错轮转实验
│   ├── llm_round.sh       # 手机侧：devlock 内 dropcache → sweep.sh，记录墙钟窗口
│   ├── analyze.py         # 汇总 rounds.tsv + 采样 → 归因判定
│   ├── summarize_probe.py # 单 App probe 的一行摘要
│   ├── runp.c             # 设备侧采集器（全设备 CPU/iowait + 进程 CPU）
│   ├── sp_helper.sh       # 设备侧每秒采样（RSS / 网络 / App CPU）
│   ├── directio.c         # O_DIRECT 真值工具（iowait 有效性对照）
│   ├── iotrace.c          # 逐次时间戳的 O_DIRECT 读探针（可选）
│   └── build.sh           # 交叉编译上述 C 探针
├── wechat/    {launch,probe,appwork,with_llm}.sh
├── tiktok/    {launch,probe,appwork,with_llm}.sh
└── genshin/   {launch,probe,appwork,with_llm}.sh
```

---

## 3. 依赖与部署

本机需要：

- `adb`（能连到被测手机；多设备时设 `DEVICE_SERIAL`）
- `ssh` 到手机的 Termux（设 `PHONE_SSH`，例如 `vivo`）
- `ANDROID_NDK`（编译设备侧 C 探针；未设时自动找 `$HOME/env/android/ndk` 下最高版本）
- `MODEL_DIR`（可选：手机侧模型目录；不设时 `dropcache.py` 只清默认 MiniCPM 4.7 路径）
- Python 3（只用标准库）

手机侧需要（`PHONE_MOE`，默认 `~/csm/moe-exp`）：

- 可运行的 `llama-cli`、模型与专家文件
- 仓库 [`bench/`](../../bench/README.md) 里的四个文件放在 `$PHONE_MOE/` 根目录：
  `bench.sh`、`sweep.sh`、`dropcache.py`、`devlock`

```sh
# 手机侧一次性部署（必须走 Termux 的 ssh/scp；adb shell 是 uid 2000，
# 进不了 /data/data/com.termux 这个 App 私有目录）
scp bench/bench.sh bench/sweep.sh bench/dropcache.py bench/devlock vivo:~/csm/moe-exp/
ssh vivo 'chmod 755 ~/csm/moe-exp/bench.sh ~/csm/moe-exp/sweep.sh ~/csm/moe-exp/devlock'
ssh vivo 'mkdir -p ~/csm/moe-exp/runs'
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE_SERIAL` | 空（`adb` 自动选唯一设备） | 多设备时指定序列号 |
| `PHONE_SSH` | 空，调用 llama 时必填 | Termux 的 ssh host/alias |
| `PHONE_HOME` | `/data/data/com.termux/files/home` | Termux HOME |
| `PHONE_MOE` | `$PHONE_HOME/csm/moe-exp` | 手机侧仓库/bench 部署目录 |
| `ROOT` | `$HOME/csm/moe-exp` | 手机侧 `sweep.sh` 使用的根目录 |
| `MODEL_DIR` | 空 | 手机侧模型目录；设了才按它清 page cache（默认只清 dropcache.py 里的 MiniCPM 4.7 路径） |
| `ANDROID_NDK` | 自动探测（`$HOME/env/android/ndk`，优先 27.0.12077973） | NDK 根目录 |
| `ANDROID_API` | `35` | 设备侧探针的 target API |
| `FRONT_APP_DEVICE_DIR` | `/data/local/tmp/sp` | 手机侧工具目录（adb shell 可写） |

```sh
export PHONE_SSH=vivo          # 或 user@host / ssh config alias
export DEVICE_SERIAL=...       # 只有一台设备时可省略
```

---

## 4. 三种模式

| 模式 | 脚本 | 用途 |
|---|---|---|
| **probe** | `<app>/probe.sh --dur 120` | 只跑 App，得到该 App 的 CPU/RSS/iowait 基线 |
| **llm-only（B）** | 由 `with_llm.sh` 自动跑 | 只跑 `llama-cli`（Termux 切后台），无干扰对照 |
| **with-app（C）** | `<app>/with_llm.sh --rounds 4 --dur 120` | App 前台 + `llama-cli` 后台 |

B/C 轮转顺序为 **C B B C C B B C**（ABBA 交错），避免热漂移把结论带偏。

---

## 5. 用法

> 这些脚本会 `force-stop` 目标 App、操作手机前台、并向 `/data/local/tmp/sp` 推送工具；
> 只在你自己的测试设备上跑。

```sh
cd tools/front-app

# 单 App 基线
./tiktok/probe.sh --dur 120
./wechat/probe.sh --dur 120

# 前台 App + 后台 LLM（4 轮 ABBA，每轮 120 s）
./tiktok/with_llm.sh --rounds 4 --dur 120 --n 64 --warm 120
./wechat/with_llm.sh --rounds 4 --dur 120 --warm 120

# 原神：脚本会尝试自动进入游戏世界（约 90–120 s）
./genshin/with_llm.sh --rounds 4 --dur 180 --warm 0
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rounds` | `2` | B/C 总轮数；做结论至少 `4` |
| `--dur` | `120` | 每轮采样秒数 |
| `--n` | `64` | 每轮生成 token 数；同一组实验必须固定 |
| `--warm` | `0` | 首次运行前预热 App 的秒数；首次测某 App 建议 `120` |
| `--bench-env` | 空 | 透传给 `bench.sh` 的 env，如 `"WINDOW=32 IO_THREADS=8"` |
| `--outdir` | `artifacts/<label>-with-llm-<时间戳>` | 产物目录 |

---

## 6. 产物

```
artifacts/<label>-with-llm-<时间戳>/
├── rounds.tsv       每轮 config / gen_tps / prompt_tps / fg_lost / 起始时间
├── <tag>.dev.csv    runp：dev_busy / dev_iowait / App CPU（累计 jiffies）
├── <tag>.app.csv    RSS / 网络 / App CPU
├── <tag>.fg.txt     采样期前台记录（probe 用）
├── <tag>.fg_round.txt  本轮 llama 窗口前台守护（with_llm 用；缺失/空焦点=整轮无效）
├── <tag>.llm.log    llama-cli 本轮输出
├── llm_windows.log  每轮 llama 的墙钟窗口 tag/start/end/rc/fg_lost（手机时钟）
├── warmup.txt       预热写盘量（若启用）
└── summary.txt      analyze.py 的结论
```

`artifacts/` 与 `shared/build/` 均被 `.gitignore` 排除，不进 git。

---

## 7. 判定规则

`analyze.py` 按以下顺序归因：

1. B vs C 的 decode tok/s 中位数是否显著下降，且两组方向一致；
2. 若下降：
   - 设备 `iowait` 绝对增量 ≥ 5 ms/s 且相对增幅 ≥ 50% → 以 **UFS / 块层** 为主；
   - App CPU 高但 `iowait` 不变 → 以 **CPU 调度 / LPDDR 带宽** 为主；
   - App CPU 很低 → 干扰很弱，差异可能只是噪声；
3. 要进一步区分「CPU 抢占」与「LPDDR 带宽」，需用 `skills/profile` 的 qtimer skel：
   DSP 计算变慢 → 带宽被挤；DSP 不变但 IO 变慢 → UFS。

阈值不能写成「iowait 翻倍」：B 组基线本身就因 llama 读专家而很高，翻倍会把真实的
197% 上升误判成「不变」。

---

## 8. 方法学要求（不做这些，数字不可信）

- **ABBA 交错、组间不重叠**：至少 4 轮；只比较中位数，不看单轮。
- **固定 `N`**：`N` 会显著改变测到的 decode（N=8/16/32/64 差别很大），
  同一组实验必须固定，且与基线一致。
- **首次运行先预热**：App 首次启动会解压资源/建缓存/dexopt，写盘量可达稳态的数十倍；
  不预热会把一次性初始化算成干扰。
- **前台守护**：由 host 侧 adb 每秒采 `<tag>.fg_round.txt`，覆盖整个 llama 窗口
  （含 devlock 等待）。文件缺失、样本 <2、空焦点或任何一次不是目标包，整轮标记
  `fg_lost=1` 并从统计中剔除。不要改用 Termux 里的 `dumpsys window`——普通 App
  进程没有 `android.permission.DUMP`，会全部误判。
- **同一 CDSP 只跑一个 HTP 使用者**：`llm_round.sh` 内部已用 `devlock`；
  不要在实验期间再跑其它 llama/CLI/App。
- **记录原始产物**：`artifacts/` 保留原始 CSV 与日志，结论必须能回溯到这些文件。
- **窗口覆盖**：只有采样跨度 ≥ `llm_windows.log` 窗口 80% 的轮次才进入统计；
  覆盖不足的轮次直接跳过（不拿部分窗口外推整轮）。

---

## 9. 已知限制

- 无 root：拿不到 PMU 内存带宽事件、每 App 块层字节数、App 内部帧耗时。
- `dumpsys` 在高负载下会变慢；采样器已按真实秒数记时间轴，但极端负载下仍有抖动。
- App 版本更新会改变负载形态；换版本后要重新做 probe/预热。
- 原神的登录、区服、横屏坐标与机型相关；脚本里的坐标只保证在 3168×1440 下可用。
- 设备 `0x72` / `err 114` 打不开 DSP 会话是设备状态问题，重启手机，不要怀疑构建。

---

## 10. 排障

| 现象 | 处理 |
|---|---|
| `adb: more than one device` | `export DEVICE_SERIAL=<serial>` |
| `ssh: Could not resolve hostname` | `export PHONE_SSH=<alias>` |
| `sweep.sh: No such file` | 按 §3 把 `bench/` 四个文件部署到 `$PHONE_MOE` |
| `failed to open session ... 0x72` | 重启手机；确认没有残留 HTP 进程 |
| App 一直被判 `fg_lost` | 检查包名/悬浮窗/分屏；先在手机上手动确认前台 |

---

## 11. 相关文档

- [`bench/README.md`](../../bench/README.md)：`bench.sh` / `sweep.sh` / `dropcache.py` / `devlock`
- [`skills/bench/SKILL.md`](../../skills/bench/SKILL.md)：端到端 tok/s 测量
- [`skills/profile/SKILL.md`](../../skills/profile/SKILL.md)：DSP/IO 重叠与 page cache 剖析
- [`AGENTS.md`](../../AGENTS.md)：设备独占锁、实验记录规范
