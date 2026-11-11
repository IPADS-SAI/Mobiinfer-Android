# bench/ — 端侧 MoE 端到端 bench 脚本（单一来源）

这里放**手机上实际在跑**的四个脚本，是 bench 流程的唯一权威副本。
`skills/bench/SKILL.md` 的方法说明指向本目录。

> 手机端位置：`~/csm/moe-exp/{bench.sh,sweep.sh,dropcache.py,devlock}`。
> **2026-11-10 起 `bench.sh` 为 v3**（默认 `release-v81-sync` / IO8 / `MAX_ROWS=8` /
> `READY=1 + REFRESH=1` 的最快配置 / config banner）。本目录 `bench/bench.sh` 与手机端**逐字节一致**
> （md5 `29795fb1c4e690be72b0366f90c7c2ab`）；旧的 v2 备份在手机
> `~/csm/moe-exp/bench-v2-20261110.sh`（md5 `ee2b14e528e7a00b2c5d498e44ffe9c7`）。
> 改任何一份都要两边同步，并把 md5 记在这里。

## 四个脚本

| 脚本 | 作用 |
|---|---|
| `bench.sh` | 组装 env + 参数，`exec llama-cli`。默认值见下表，全部可用 env 覆盖 |
| `sweep.sh` | 跑一次 `bench.sh`（`stdbuf -oL` 行缓冲），解析 `Generation:`/`Prompt:` t/s，追加一行到 `runs/results.tsv` |
| `dropcache.py` | `posix_fadvise(DONTNEED)` 丢模型/专家的 page cache，无需 root；让每次采样从冷缓存开始 |
| `devlock` | 内核 `flock(2)` 独占锁，串行化 DSP 使用者；进程结束（含 SIGKILL）自动释放 |

## v3 默认值（2026-11-10）

| 项 | 默认 | 说明 |
|---|---|---|
| `REL` | `release-v81-sync` | main + hexagon 11-10；旧基线用 `release-v81-pipe` |
| `WINDOW` / `DEPTH` / `READ` | 80 / 8 / 8 | 每层窗口 + 预读深度/读数 |
| `HORIZON` / `QUEUE` | 2 / 4 | 跨层流水提前量、队列深度 |
| `IO_THREADS` | **8** | 旧版是 4；历史 8 对 ABBA +7.7%（2026-11-05），本会话单次对照未复现（在噪声内） |
| `MAX_ROWS` / `WINDOW_ROWS` | **8** / 8 | 预取/窗口覆盖的批宽；旧版 `MAX_ROWS` 硬编码 4，`SPECN≥4` 时会静默关预取 |
| `SPECN` | 3 | DSpark 草稿长度；verify 行数 = `SPECN+1` |
| `READY` / `REFRESH` | **1 / 1** | 默认测“实测最快”配置（21.90 t/s）；生产默认仍是 0/0，测基线要显式 `READY=0 REFRESH=0` |
| `H1_FRONT` / `PROBATION` / `IO_BUDGET` | 0 / 0 / 0 | 其余 opt-in，显式写出 |
| `VERIFY_SLOT` | 0 | 1 = 每读一个专家就做字节级落点自检（很慢，仅正确性用） |
| `SPLIT` | 0 | `LLAMA_MOE_MATMUL_SPLIT`；与 REFRESH 互斥（>0 时须显式 `REFRESH=0`） |

启动时 stderr 打一行：

```
[bench] rel=... lib=<sha12> n=64 window=80 depth=8 read=8 io=8 spec=3
        horizon=2 queue=4 maxrows=8 ready=1 refresh=1 split=0 ...
```

## 最简跑法

```sh
# 默认 = 最快配置（sync + IO8 + READY=1 + REFRESH=1，本机实测 21.8–21.9 t/s）
ssh vivo 'cd ~/csm/moe-exp && ./devlock -w 600 bash -c "
  python3 dropcache.py >/dev/null 2>&1; N=64 bash bench.sh"'

# 测基线（关掉两个 opt-in，等价于 main 默认行为）
ssh vivo 'cd ~/csm/moe-exp && READY=0 REFRESH=0 ./devlock -w 600 bash -c "
  python3 dropcache.py >/dev/null 2>&1; N=64 bash bench.sh"'

# 走 sweep.sh 落 TSV
ssh vivo 'cd ~/csm/moe-exp && TAG=smoke N=64 ./devlock -w 600 bash -c "
  python3 dropcache.py >/dev/null 2>&1; exec bash sweep.sh"'

# opt-in 臂（例：READY+REFRESH）
ssh vivo 'cd ~/csm/moe-exp && REL=$HOME/csm/moe/release-v81-sync N=64 \
  READY=1 REFRESH=1 ./devlock -w 600 bash -c "
  python3 dropcache.py >/dev/null 2>&1; exec bash sweep.sh"'
```

`dropcache` 与单轮 run 都要在**锁内**（否则排队等锁期间缓存会被别的实验改掉）。
`sweep.sh` 自己**不** dropcache，必须由调用方在循环里做。

## 调用链

```
devlock  →  [ dropcache.py  →  sweep.sh  →  bench.sh  →  llama-cli ]
                                                   ↓
                                runs/<TAG>-n<N>.out / .err
                                                   ↓
                                runs/results.tsv   (tag n wall_ms rc gen_tps prompt_tps)
```

`devlock` 持有锁的进程是 `bash -c "...; bash bench.sh"`，`dropcache` 必须在**锁内**做。

## App 内 bench（`bench/app/`）

`bench/app/app_bench.sh` 在 App 里跑**同一条 prompt**（`bench/prompts/`），从
`benchmark.jsonl` 取 `wall_ms/ttft_ms/prompt_*/predicted_*`，口径与 `bench.sh` 的
`Generation:`/`Prompt:` 一致。方法与差异表见 [`app/README.md`](app/README.md)。

- App 与 CLI 的 prompt 必须共用 `bench/prompts/` 里的文件，避免两边各写一句；
- App/CLI 是同一个 HTP 的两个用户，必须二选一：CLI 前 `am force-stop com.mobiinfer.demo`，
  并保持 `devlock` 串行；
- App 的窗口/预读由模型目录 `runtime.json` 控制（`cache_slots` 就是
  `LLAMA_MOE_DECODE_WINDOW`），对齐 CLI 时用 `RUNTIME_JSON=` 传入。

## env 命名规则（v3）

- `WINDOW` / `DEPTH` / `READ` / `IO_THREADS` / `PREFETCH` / `SPECN` 这类**只认短名**；
  直接传对应 `LLAMA_MOE_*` 会被脚本默认值覆盖（旧行为，未改）。
- 流水/新增开关 `HORIZON`/`QUEUE`/`MAX_ROWS`/`WINDOW_ROWS`/`READY`/`REFRESH`/`H1_FRONT`/
  `PROBATION`/`IO_BUDGET`/`VERIFY_SLOT`/`SPLIT`：**短名优先，未给短名时回退读同名
  `LLAMA_MOE_*`**，最后才是默认值（兼容旧脚本，例如 `LLAMA_MOE_LOOKAHEAD_HORIZON=4`）。
- `REFRESH=1` 必须同时 `READY=1`，否则 `init_prefetch_refresh()` 直接抛异常。
- 空串 `REFRESH=""` 按"没给"处理（否则会被当成显式、跳过降级，最后在 init 抛异常）。
- `REFRESH` 默认 1；若**没显式给**且前置条件不满足，会自动降级为 0 并打一行带原因的提示：
  `READY!=1` / `HORIZON<2` / `PREFETCH=0` / `READ=0`（**负数不降级**，引擎里负值=跟随 depth）/
  `WINDOW<=0` / `DEPTH<=0` / `BANK<2*DEPTH+8`（否则 depth 被归一化成 0）/
  `H1_FRONT=1` / `IO_BUDGET!=0` / `SPLIT!=0`。显式 `REFRESH=1` 时**不**降级（让异常尽早暴露）。

## 规则

- 一次只跑一个 HTP 实验：批量/对照一律 `devlock` 串行。
- 只有**轮转**的多臂对照（≥4 轮、组间不重叠）才能宣称端到端收益；单次采样只算机制结果。
- 每轮前 `dropcache.py`；`devlock` 只管串行，不保证被测命令本身稳定。
- 老构建（`release-v81` / `release-v81-pipe`）在 teardown 段 segfault（rc=139），
  **CLI 的 `[Prompt|Generation]` 行来不及打印**，t/s 只在 stderr 的
  `slot print_timing: ... eval time = ... tokens per second` 里 —— 解析时要加 fallback。

## 待补（TODO）

- `ab.sh`：多臂轮转 + "组间不重叠"判定 + 计数器（bytes/miss/hit/prefetch/refresh）落 TSV。
- `sweep.sh` 的 stderr `print_timing` fallback（现在会把 rc=139 但已跑完的 run 记成 `NA`）。
- 结果 JSON（`runs/<tag>.json`），字段固定，便于后续分析脚本直接消费。
