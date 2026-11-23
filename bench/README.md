# bench/ — 端侧 MoE 端到端 bench 脚本（单一来源）

这里放 bench 流程的**仓库权威副本**（`bench.sh` / `bench-nubia.sh` / `sweep.sh` /
`dropcache.py` / `devlock`）；nubia 的 `bench-nubia.sh` 是 `bench.sh` 的薄 wrapper
（改 `REL`/`ARCH`/`HOSTBUF`/`EXPERTS` 默认值，并在 bench 前打印解析身份），
gate 会把本目录整体推到手机 workdir，不再用手机隐式副本。
`skills/bench/SKILL.md` 的方法说明指向本目录。

> **2026-11-22 设备默认变更：手机实验默认写 nubia（P0110 / SM8750 / Hexagon v79）。**
> vivo（V2546A / SM8850 / v81）电池低，只在需要 **vivo 特有数据/构建**时用。
>
> - nubia：`ssh nubia`（不通时 `adb -s EP0110MZ0BB291368W forward tcp:2224 tcp:8022`）、
>   `REL=$HOME/csm/moe/release-v81-chunkpaging-nubia`（含 `libggml-htp-v79.so`）、
>   bench 用 `~/csm/moe-exp/bench-nubia.sh`、所有 DSP 命令用 `~/csm/moe-exp/devlock` 串行；
> - **跨设备绝对 t/s 不可比**：nubia 结论标 `nubia-only`；只有在 vivo 复现才算通用。
>
> 来源：`docs/dpsk-queue/GATES.md` §3.3、`ops/dpsk-queue/AGENT_BRIEF.md` §4、T-0012/T-0031。

> 手机端位置：`~/csm/moe-exp/{bench.sh,sweep.sh,bench-nubia.sh,dropcache.py,devlock}`。
> **2026-11-10 起 `bench.sh` 为 v3**（默认 `release-v81-sync` / IO8 / `MAX_ROWS=8` /
> `READY=1 + REFRESH=1` 的最快配置 / config banner）。
>
> **2026-11-23 T-0098 实测：仓库副本与手机副本当前并不逐字节一致**：手机 `bench.sh`
> md5 `e09c3457941dbd17e368efd5b2b53aba`（8436 B），仓库 `bench/bench.sh`
> md5 `bcdfe4751daaf6504261d9319b0b3680`（8930 B，含 2026-11-22 的 DSpark
> draft 设备/线程/offload 参数，默认行为等价）。部署/对照前先 `diff`，不要照抄旧 md5
> （旧 README 的 `29795fb…` 已过期；`GATES.md` §0 仍把 bench md5 修正记成 T-0042，
> 但 T-0042 实际是“验证 T-0014”，nubia gate 由 T-0100 跟踪）。
> 旧的 v2 备份在手机 `~/csm/moe-exp/bench-v2-20261110.sh`（md5 `ee2b14e528e7a00b2c5d498e44ffe9c7`）。
> 改任何一份都要两边同步，并把 md5 记在这里。

## 脚本

| 脚本 | 作用 |
|---|---|
| `bench.sh` | vivo 默认：组装 env + 参数，`exec llama-cli`。默认值见下表，全部可用 env 覆盖 |
| `bench-nubia.sh` | nubia 默认：薄 wrapper → `bench.sh`；`REL=release-v81-chunkpaging-nubia`、`ARCH=79`（`ARCH=` 空=后端自检）、`HOSTBUF=0`（v79 上 hostbuf=1 会 IOVA mapping fail）、`EXPERTS=experts-q2k-idnow`（28.1 基线）；启动时在 stderr 打印 `nubia-config`/`nubia-identity`；`BENCH_CONFIG_ONLY=1` 只打印身份后退出；其余参数继承 `bench.sh` |
| `sweep.sh` | 跑一次 `$BENCH`（默认 `bench.sh`，nubia 传 `BENCH=bench-nubia.sh`；`stdbuf -oL` 行缓冲），解析 `Generation:`/`Prompt:` t/s，写本轮 `runs/results-<run_id>.tsv`（首行注释头）并 append 到兼容聚合 `runs/results.tsv`；按单次 rc 传播 gate 语义（见下） |
| `dropcache.py` | `posix_fadvise(DONTNEED)` 丢模型/专家的 page cache，无需 root；T-0095 起支持 `--model/--experts/--draft` 显式 scope（旧副本不支持） |
| `devlock` | 内核 `flock(2)` 独占锁，串行化 DSP 使用者；进程结束（含 SIGKILL）自动释放 |

## v3 默认值（2026-11-10）

| 项 | 默认 | 说明 |
|---|---|---|
| `REL` | `release-v81-sync` | **vivo** v81；nubia 用 wrapper 默认的 `release-v81-chunkpaging-nubia`（v79）；旧基线用 `release-v81-pipe` |
| `ARCH` / `HOSTBUF` | 81 / 1 | 短名覆盖 `GGML_HEXAGON_ARCH`/`GGML_HEXAGON_HOSTBUF`；未设时 vivo 行为不变，nubia wrapper 传 79/0 |
| `WINDOW` / `DEPTH` / `READ` | 80 / 8 / 8 | 每层窗口 + 预读深度/读数 |
| `HORIZON` / `QUEUE` | 2 / 4 | 跨层流水提前量、队列深度 |
| `IO_THREADS` | **8** | 旧版是 4；历史 8 对 ABBA +7.7%（2026-11-05），本会话单次对照未复现（在噪声内） |
| `MAX_ROWS` / `WINDOW_ROWS` | **8** / 8 | 预取/窗口覆盖的批宽；旧版 `MAX_ROWS` 硬编码 4，`SPECN≥4` 时会静默关预取 |
| `SPECN` | 3 | DSpark 草稿长度；verify 行数 = `SPECN+1` |
| `READY` / `REFRESH` | **1 / 1** | 默认测“实测最快”配置（21.90 t/s）；生产默认仍是 0/0，测基线要显式 `READY=0 REFRESH=0` |
| `H1_FRONT` / `PROBATION` / `IO_BUDGET` | 0 / 0 / 0 | 其余 opt-in，显式写出 |
| `VERIFY_SLOT` | 0 | 1 = 每读一个专家就做字节级落点自检（很慢，仅正确性用） |
| `SPLIT` | 0 | `LLAMA_MOE_MATMUL_SPLIT`；与 REFRESH 互斥（>0 时须显式 `REFRESH=0`） |

> **nubia 专家包默认（T-0317）**：`bench-nubia.sh` 默认
> `EXPERTS=$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q2k-idnow`
> （Q1_0 + L38/39 Q2_K，4.83 GB，28.1 t/s 基线），不是 `bench.sh` 的
> `experts-htp-tiled`（Q3_K，16.1 GB）。目录不存在时 wrapper 直接 `exit 2`
> 并列出 `experts-q2k-idnow`/`experts-q1-30tps`/`experts-htp-tiled` 三个候选，
> **不会静默回退**；显式选 `experts-htp-tiled` 允许，但会打印
> `WARN: Q3_K htp-tiled is not the 28.1 baseline`。

启动时 stderr 打一行：

```
[bench] rel=... lib=<sha12> n=64 window=80 depth=8 read=8 io=8 spec=3
        horizon=2 queue=4 maxrows=8 ready=1 refresh=1 split=0 ...
```

## nubia 最简跑法（默认）

> **dropcache scope 前提（T-0095）**：仓库 `bench/dropcache.py` 已支持
> `--model/--experts/--draft`（本分支 339 行，commit `253c237e7`）。但截至
> **2026-11-23 01:2x**，两台手机上的副本仍是旧 41 行脚本
> （`sha256 300d4df5630b5187b06e62d0d770b342a68a967d8d9fb75c1457198a0fe2ce64`，T-0098
> 只读 preflight），**不认**这些选项：`--model` 等会被当成路径，只丢实际存在的文件，
> 而且不会展开 `experts-*/*.bin`。先检查：
>
> ```sh
> ssh nubia 'grep -q -- "--print-scope" ~/csm/moe-exp/dropcache.py \
>   && echo dropcache=scope-ok || echo dropcache=OLD'
> ```
>
> 只有 `scope-ok` 才能用下面的显式 scope；`OLD` 要先部署 T-0095 的
> `bench/dropcache.py`，并用 `--print-scope` 确认 scope。**未部署前不要声称冷缓存**。
>
> 2026-11-23 T-0098 用 T-0095 脚本在 nubia 上只做 `--print-scope`（未 drop、未跑 DSP）：
> `--experts experts-q2k-idnow` 解析出 **43 文件 / 6.92 GB**，包含该臂 layer_038/039 +
> symlink 到 `experts-q1-30tps` 的 0–37 层 + model + draft（证据
> `runs/T-0098/.../artifacts/nubia-print-scope.json`）。这说明 scope 会跟 symlink 到真实
> 文件；也说明“裸 `dropcache.py` 只丢默认目录”会漏掉这些层。

```sh
# 0) 只读 preflight（不占 DSP）
ssh nubia 'ls -l $HOME/csm/moe-exp/bench-nubia.sh \
  $HOME/csm/moe/release-v81-chunkpaging-nubia/lib/libggml-htp-v79.so; \
  df -h /data | tail -1'
# Termux 读不到 /sys/class/power_supply（Permission denied，T-0098 实测）；电量/温度走 adb：
adb -s EP0110MZ0BB291368W shell dumpsys battery | grep -E 'level|temperature'

# 1) 单轮：显式 dropcache scope + nubia release + 被测专家包，全程 devlock
#    EXPERTS 必须与 dropcache --experts 一致；bench-nubia 默认 EXPERTS=experts-q2k-idnow
#    （28.1 基线），显式给 EXPERTS 仍会覆盖默认。
ssh nubia 'cd ~/csm/moe-exp && M=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon && ./devlock -w 900 bash -c "
  set -e
  mkdir -p runs
  python3 dropcache.py --model \"$M/minicpm47-les-hexagon.gguf\" \
    --experts \"$M/experts-q2k-idnow\" \
    --draft \"$M/dspark-MiniCPM-V-4_7-q8_0.gguf\" --strict --json
  env REL=$HOME/csm/moe/release-v81-chunkpaging-nubia EXPERTS=\"$M/experts-q2k-idnow\" \
    N=64 WINDOW=80 DEPTH=16 READ=8 IO_THREADS=8 MAX_ROWS=8 SPECN=3 \
    bash ./bench-nubia.sh > runs/nubia-smoke-n64.out 2> runs/nubia-smoke-n64.err"'
ssh nubia 'grep -E "Prompt:|Generation:" ~/csm/moe-exp/runs/nubia-smoke-n64.out | tail -2'
```

仓库版 `bench-nubia.sh` 是 `bench.sh` 的**薄 wrapper**：默认
`REL=$HOME/csm/moe/release-v81-chunkpaging-nubia`、`ARCH=79`、`HOSTBUF=0`
（`ARCH=` 空让后端自检；`HOSTBUF=1` 在 2026-11-23 的集成构建上会
`HTP0 buffer mapping failed ... error 0x0000000e`，旧 release 不受影响），
以及 `EXPERTS=$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q2k-idnow`
（28.1 t/s 基线）。它不认 `TAG`、不写 `runs/results-*.tsv`；要 TSV/门禁证据时经
`sweep.sh` 跑（`BENCH=bench-nubia.sh`），gate 已这样做。手机上的旧
`bench-nubia.sh` 是与 `bench.sh` 分叉的副本（缺 `DRAFT_DEVICE`/`HOSTBUF`，默认 vivo
REL），不要再当权威。

wrapper 在 exec `bench.sh` 前向 stderr 打印两行身份（stdout 的 t/s 解析不变）：

```
nubia-config: REL=... ARCH=... HOSTBUF=... MODEL=... EXPERTS=... DRAFT=...
nubia-identity: experts_basename=... experts_bytes=... manifest_sha256=...
```

manifest 依次找 `manifest.json`、`experts-manifest.json`、`SHA256SUMS`；缺失时
`manifest_sha256=unknown`（不失败），`experts_bytes` 来自 `du -sb`。host 测试/人工
sanity check 可用 `BENCH_CONFIG_ONLY=1 bash bench-nubia.sh`：只打印上述两行后
`exit 0`，不 exec `bench.sh`、不碰 DSP。`bench/test-bench-nubia.sh` 覆盖默认包、
显式 htp-tiled WARN、缺失目录 `exit 2` 和 `ARCH=` 空值语义。

> **验证状态（2026-11-23 T-0098）**：preflight 与 `--print-scope` 已实测；单轮/4+4 的
> shell 语法已检查，但**没有跑 DSP**（手机 dropcache 仍是旧版，本任务 resource=none）。
> 部署 T-0095 后第一次跑，把结果从“待验证”改成实测并附 `.out/.err`。

4+4 轮转（A=基线、B=候选；同一把 `devlock` 内串行，每轮先显式 dropcache；
第 2/4 轮用 BA 抵消顺序漂移）：

```sh
ssh nubia 'cd ~/csm/moe-exp && ./devlock -w 1800 bash -s' <<'EOS'
set -u
M=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon
REL_A=$HOME/csm/moe/dpsk-gate/BASE_TASK/release     # 替换 BASE_TASK 为基线 release
REL_B=$HOME/csm/moe/dpsk-gate/CAND_TASK/release     # 替换 CAND_TASK 为候选 release
mkdir -p runs
for r in 1 2 3 4; do
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for arm in $order; do
    case $arm in A) rel=$REL_A ;; B) rel=$REL_B ;; esac
    python3 dropcache.py --model "$M/minicpm47-les-hexagon.gguf" \
      --experts "$M/experts-q2k-idnow" --draft "$M/dspark-MiniCPM-V-4_7-q8_0.gguf" \
      --strict --json > "runs/nubia-r$r-$arm.dropcache.json" || exit 2
    rc=0
    env REL="$rel" EXPERTS="$M/experts-q2k-idnow" N=64 WINDOW=80 DEPTH=16 \
      READ=8 IO_THREADS=8 MAX_ROWS=8 SPECN=3 \
      bash ./bench-nubia.sh > "runs/nubia-r$r-$arm.out" 2> "runs/nubia-r$r-$arm.err" || rc=$?
    echo "[nubia] r=$r arm=$arm rc=$rc $(grep -h 'Generation:' "runs/nubia-r$r-$arm.out" | tail -1)"
  done
done
EOS
```

nubia 手工 A/B 只能标 `nubia-only`；**不要把这里的绝对 t/s 和 vivo 比**。gate 的
`full` 模式自 T-0100 起支持 nubia（`GATE_PHONE=nubia`，用仓库 `bench-nubia.sh` + v79），
见 [`docs/dpsk-queue/GATES.md` §3.3](../docs/dpsk-queue/GATES.md)。gate 通过
`sweep.sh` 的 `BENCH=bench-nubia.sh` 落 per-run TSV；它**始终显式传 `EXPERTS`**
（`GATE_EXPERTS` 默认仍是 `experts-htp-tiled`），所以 wrapper 的 q2k-idnow 默认
不影响 gate 的 Q3_K 路径；手工 A/B 证据仍是 `.out/.err` + 每轮 `.dropcache.json`。

## vivo 最简跑法（非默认；仅 vivo 特有数据）

> vivo 只在需要 **vivo 特有数据/构建**时用；`release-v81-sync` 是 vivo 的 v81 构建，
> **nubia 上用不了**（缺 `libggml-htp-v79.so`）。nubia 默认见上一节。

```sh
# 默认 = 最快配置（sync + IO8 + READY=1 + REFRESH=1，历史同会话 21.8–21.9 t/s）
ssh vivo 'cd ~/csm/moe-exp && M=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon && ./devlock -w 600 bash -c "
  python3 dropcache.py --model \"$M/minicpm47-les-hexagon.gguf\" \
    --experts \"$M/experts-htp-tiled\" --draft \"$M/dspark-MiniCPM-V-4_7-q8_0.gguf\" --strict
  N=64 bash bench.sh"'

# 测基线（关掉两个 opt-in，等价于 main 默认行为）
ssh vivo 'cd ~/csm/moe-exp && M=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon && READY=0 REFRESH=0 ./devlock -w 600 bash -c "
  python3 dropcache.py --model \"$M/minicpm47-les-hexagon.gguf\" \
    --experts \"$M/experts-htp-tiled\" --draft \"$M/dspark-MiniCPM-V-4_7-q8_0.gguf\" --strict
  N=64 bash bench.sh"'

# 走 sweep.sh 落 TSV
ssh vivo 'cd ~/csm/moe-exp && M=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon && TAG=smoke N=64 ./devlock -w 600 bash -c "
  python3 dropcache.py --model \"$M/minicpm47-les-hexagon.gguf\" \
    --experts \"$M/experts-htp-tiled\" --draft \"$M/dspark-MiniCPM-V-4_7-q8_0.gguf\" --strict
  exec bash sweep.sh"'

# opt-in 臂（例：READY+REFRESH）
ssh vivo 'cd ~/csm/moe-exp && M=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon && REL=$HOME/csm/moe/release-v81-sync N=64 \
  READY=1 REFRESH=1 ./devlock -w 600 bash -c "
  python3 dropcache.py --model \"$M/minicpm47-les-hexagon.gguf\" \
    --experts \"$M/experts-htp-tiled\" --draft \"$M/dspark-MiniCPM-V-4_7-q8_0.gguf\" --strict
  exec bash sweep.sh"'
```

`dropcache` 与单轮 run 都要在**锁内**（否则排队等锁期间缓存会被别的实验改掉）。
`sweep.sh` 自己**不** dropcache，必须由调用方在循环里做；上面的显式 scope 同样要求手机
副本已是 T-0095 后的版本（见「nubia 最简跑法」开头的检查）。

> `release-v81-sync` = vivo v81（`libggml-htp-v81.so`）；
> `release-v81-chunkpaging-nubia` = nubia v79（`libggml-htp-v79.so`，T-0012/T-0031）。
> 两者不能互换，也不能拿同一份绝对 t/s 跨设备比较。

## 调用链

```
devlock  →  [ dropcache.py  →  sweep.sh  →  bench.sh  →  llama-cli ]
                                                   ↓
                                runs/<TAG>-n<N>.out / .err
                                                   ↓
        runs/results-<run_id>.tsv  (本轮专属；首行注释头 + 数据行)
        runs/results.tsv           (legacy 聚合；只有数据行，可能含历史行)
```

`devlock` 持有锁的进程是 `bash -c "...; bash bench.sh"`，`dropcache` 必须在**锁内**做。

nubia 走同一条链，只是 `sweep.sh` 里 `BENCH=bench-nubia.sh`（wrapper 再 exec `bench.sh`）：
`devlock → [dropcache.py → sweep.sh → bench-nubia.sh → bench.sh → llama-cli]`。
手工单跑可以绕过 `sweep.sh`，但那样只有 `.out/.err`，没有 per-run TSV。

## per-run results 文件（T-0099）

`sweep.sh` 每次调用生成一个**不可混淆**的本轮文件 `runs/results-<run_id>.tsv`，
同时继续 append 旧的 `runs/results.tsv` 聚合：

- `run_id` 取 `SWEEP_RUN_ID`；未设时默认 `<本地时间>-<pid>`（如
  `20261122T123526-661538`），所以默认每次调用都不同。只允许
  `[A-Za-z0-9._-]`，非法值直接 `exit 3`（不写任何结果）。
- 新文件先写一行注释头（仅新文件；重复 id 不重复写头）：
  `# sweep-results v1 run_id=<id> tag n wall_ms rc gen_tps prompt_tps [mem...] [io...] status`；
  数据行与旧格式**完全一致**（默认 7 列；`MEMMARK=1` 26 列；`IOMARK=1` 10 列；
  两者都开 29 列；`status` 永远在最后）。
- 同一 gate run 的 4/8 轮显式传同一个 id，方便只读本轮数据：

  ```sh
  SWEEP_RUN_ID=gate-20261122T1300 TAG=gate-r1-A bash sweep.sh
  ```

- **重复 id 语义 = 追加**（不覆盖）：同一 `SWEEP_RUN_ID` 跑第二次只追加数据行，
  注释头只有一份；分析时用 `status=ok` 和轮次 tag 过滤。
- `legacy` 是兼容聚合：**不写注释头**，列序/列数不变，现有 `awk '$1'` / `stats.py`
  读取不受影响；它可能含历史行，不能用来判定"这一轮跑了几次"。
- **gate/分析应优先读 `results-<run_id>.tsv`**；legacy 只留给尚未改的解析器过渡。
  `stats.py speed --a ... --b ...` 可直接读 per-run 文件（注释头没有 tab、字段数 < 6，
  `read_tsv()` 会跳过）。
- 写入顺序：先 per-run，再 legacy；两者在同一把锁内各写一行（单次 `printf` +
  `O_APPEND`）。优先 `flock -w 60 runs/.results.lock`；Termux 没有 `flock(1)` 时退回
  `mkdir runs/.results.lock.d` 自旋锁（60s 超时，失败 `exit 3`）。锁只保护 append，
  不改变 `devlock` 的设备串行语义。
- 不自动清理旧 `results-*.tsv`（janitor 管）。
- 回归：`bash bench/test-sweep.sh`（临时 ROOT + mock `bench.sh`，不访问手机/DSP；
  覆盖两次调用隔离、legacy 列数、重复 id 追加、非法 id、MEMMARK/IOMARK 列数、并发 append）。

## IOMARK：`/proc/<pid>/io` 读放大代理（T-0225）

`bench.sh` 最后 `exec "$REL/bin/llama-cli"`，所以 `sweep.sh` 里的 `$pid` 就是
**llama-cli 进程**（不是中间 shell）。`IOMARK=1` 时在启动后与结束前各取一次
`/proc/<pid>/io`，把差值写进 `status` 前的 3 列：

| 列 | 来源 | 含义 / 注意 |
|---|---|---|
| `io_rchar` | `rchar` 差值 | 进程读过的总字节，**包含 buffered/伪文件读**；不是块层读 |
| `io_read_bytes` | `read_bytes` 差值 | 进程导致的块层读；只有 **O_DIRECT** 下才接近真实设备读，buffered 时可能为 0 或很小 |
| `io_syscr` | `syscr` 差值 | 读系统调用次数，辅助判断读粒度 |

- 开关：`IOMARK=1`；默认 `IOMARK=0`，此时表头/行**完全不变**，也不读
  `/proc/<pid>/io`。gate 的 full speed 采样默认 `GATE_IOMARK=1`（`GATE_IOMARK=0` 可关）。
- 列顺序：`BASE + MEM + HEALTH + IO + status`；只开 IOMARK 时为
  `BASE + IO + status`（10 列），`status` 仍是最后一列。
- 采样是 **best-effort**：`/proc` 不可读、字段缺失、权限拒绝（Termux/Android 的
  `EACCES` 很常见）都写 `NA`，绝不写 0、绝不让 sweep 失败或改变 rc。
- **不要把 `io_read_bytes` 当成运行时 `requested_bytes`/`read_experts`**：前者是
  `/proc` 的块层读代理，后者是引擎逻辑字节；两者口径不同。关闭 O_DIRECT
  （buffered）时 page-cache 命中不会产生 `read_bytes`，更不能拿它当真实设备读。
- 采样复用 MEMMARK 既有的 0.4s 循环（只开 IOMARK 时用轻量 0.1s 轮询）；最后读到的
  存活快照作为 end，取不到 end 时差值为 `NA`。

## IOSTAT：逐 run UFS/GC/内存回收侧车（T-0296）

`IOSTAT=1` 时 `sweep.sh` 在**每个 run 开始前/结束后**各读一次内核计数器，写
`runs/<tag>-iostat.json`（`iostat-v1`）。它**不改 TSV 表头/列数/行**，也不改
`bench.sh` 的 argv、t/s、rc 或 gate verdict；默认 `IOSTAT=0` 时连 `/proc`、`/sys`
都不读。

- 采样点：`sweep.sh` 在启动 `bench.sh` 前、结束后各一次；与 `MEMMARK=1` 的
  `snapshot_device_health` 同一阶段。侧车不依赖 `MEMMARK`/`IOMARK`。
- 块设备：用 `MODEL`/`EXPERTS`/`DRAFT` 的实际路径（`readlink -f` + `df -P`）找
  mount 的 `/dev/...`。手工直接跑 `sweep.sh` 时要显式给这三个路径（gate 的
  `run_phone_sample` 已经设置）；只跑 `bench.sh` 不会产生侧车。`/sys/block/<dev>/stat` 优先；模型在分区上时用整盘
  （`sda1 → sda`、`mmcblk0p1 → mmcblk0`）的计数器，因为 UFS GC/队列是整盘行为。
  `/sys/block` 不可读时回退 `/proc/diskstats`；都拿不到就写 `null` + `missing`。
- 主机 fixture 测试可覆盖根路径：`IOSTAT_PROC_ROOT`、`IOSTAT_SYS_ROOT`（默认
  `/proc`、`/sys`），因此不需要真机/真内核。

JSON 字段（`iostat-v1`）：

| 字段 | 来源 | 含义 |
|---|---|---|
| `vmstat.pgscan_direct` | `/proc/vmstat` | 直接回收扫描次数（run 内 after−before） |
| `vmstat.pgsteal_direct` | `/proc/vmstat` | 直接回收偷走的页数 |
| `vmstat.allocstall` | `/proc/vmstat` | 分配停顿次数；与上面两项一起看内存压力 |
| `vmstat.nr_dirty` / `nr_writeback` / `nr_unevictable` | `/proc/vmstat` | 采样时刻的脏页/回写/不可回收页 |
| `disk.dev` | `df -P`/`readlink -f` | 实际取 stat 的块设备名（整盘） |
| `disk.read_ios` / `read_sectors` / `read_ticks` | `/sys/block/<dev>/stat` 或 `/proc/diskstats` | 读次数/扇区/读耗时（ms） |
| `disk.write_ios` / `write_sectors` / `write_ticks` | 同上 | 写次数/扇区/写耗时（ms） |
| `disk.in_flight` | 同上 | 采样瞬间在途 I/O（gauge，不参与 delta） |
| `disk.io_ticks` / `time_in_queue` | 同上 | 设备忙时间 / 加权队列时间（ms） |
| `before_ts_ns` / `after_ts_ns` | `date +%s%N` | 采样点墙钟 |
| `before` / `after` | 上述计数器 | 原始快照，便于对账 |
| `delta` | after−before | 计数器差值；`disk.in_flight` 固定 `null` |
| `sources` | — | 实际尝试的 proc/sys/df 路径 |
| `missing` | — | 读不到的字段/来源（`vmstat.*`、`disk.*`、`sources.*`） |

规则：

- **所有读不到的字段都是 `null` 并进 `missing`，绝不写 0**；任何权限/路径/解析
  失败都只影响侧车，`sweep.sh` 的 rc、TSV、t/s 完全不变。
- 侧车 delta 覆盖**整个 run**（prompt eval + decode），不是只覆盖 prefill 段。
  关联 T-0272 的 `pread_wall`：同一个 `gate-r<N>-A|B` tag 的 `.err` 里有
  `prefill #k pread_wall ...`，把快/慢态的 prefill run 与同 tag 的
  `delta.disk.read_ticks` / `time_in_queue` / `vmstat.pgscan_direct` 逐轮对齐；
  若慢态这些计数器同步升高，才支持“UFS GC/内存回收”假设。
- gate 的 full speed 采样默认 `GATE_IOSTAT=0`；`GATE_IOSTAT=1` 时把
  `IOSTAT=1` 传给 `sweep.sh`，随 `runs/` 一起拉回 `*-iostat.json`，由
  `stats.py speed --iostat-dir` 聚合每臂 `pgscan_direct/pgsteal_direct/
  read_ticks/time_in_queue` 的 delta 中位数，写入 `gate.json.speed.io_state`；
  聚合明细同时留在 `speed/stats.json` 的 `io_state` 字段，**不参与
  `pass`/`prefill_ok` 判据**。
- Termux/Android 上 `/proc/vmstat` 通常可读，`/sys/block/*/stat` 可能
  `EACCES` 或不存在；下次 nubia run 需要实测哪些字段真正有值（见 result.json
  的 risks）。若 `disk.*` 全 null，仍可用 `vmstat.*` 判断直接回收压力。

回归：`bash bench/test-sweep.sh` 覆盖正常、部分缺失、无块设备三种 host fixture，
并断言 `IOSTAT=0` 时 TSV 表头/列数逐字不变。

## OPBUFSTAT：prep_op_bufs 的 P0 计数（T-0304）

`GGML_HEXAGON_OPBUFSTAT=1` 打开 op-buffer 观测：DSP skel 在 `prep_op_bufs()` 里把
每个 batch 的计数填进 batch rsp，宿主收到后格式化到 stderr。它用来回答“本 batch
有没有 drop/remap、扫了多少 mmap 表项、`m_vmem + e_vmem` 是否越过 `max_vmem`”
这类归因问题。它**只做观测**：计数只是旁路自增，不改变 reuse/drop/mmap 的
任何分支条件或系统调用次数；默认关闭。

- 开关：`GGML_HEXAGON_OPBUFSTAT=1`。未设或解析为 0 时**零输出**：不注册
  `atexit`、不格式化 JSONL、不碰 stderr（`opt_opbufstat == 0` 的两个 `if` 都不进）。
- 输出全在 **stderr**：每个 batch 一行纯 JSON（无 `ggml-hex:` 前缀，写完 `fflush`），
  进程正常退出时再由 `atexit` 打一行 `{"opbuf_total":{...}}`。JSONL 与普通日志同流，
  消费方要按行找 `{"seq"`；`opbuf_stats.py` 已做这种过滤。
- **SIGKILL/超时/abort/`_exit` 不会走 `atexit`**，因此可能缺 `opbuf_total`；已经写完的
  逐 batch 行仍然有效。脚本会打印 `total line: absent`，但**只要逐行守恒通过仍退出 0**
  ——判断“数据全不全”要看这一行或 `--json` 的 `total_line`，不能只看退出码。
- 逐 batch 行示例（字段顺序即输出顺序）：

  ```json
  {"seq":1,"n_bufs":7,"reuse":7,"drop_n":0,"drop_bytes":0,"map_n":0,"map_bytes":0,"m_vmem":0,"e_vmem":0,"max_vmem":3347161088,"defrag":0,"scan_cmp":28}
  ```

字段（**JSONL 键 / 结构体字段** 是两套名字，`reuse`=`b_reuse`、
`defrag`=`defrag_n`）：

| JSONL 键 | 结构体字段 | 单位 | 含义 |
|---|---|---|---|
| `seq` | `rsp.seq` | 计数 | batch 序号，宿主按 1 递增校验 |
| `n_bufs` | `n_bufs` | 计数 | 本 batch 的 buffer 数（`req->n_bufs`） |
| `reuse` | `b_reuse` | 计数 | pass 0 `reuse_buf()` 命中的 buffer 数 |
| `drop_n` | `drop_n` | 计数 | `drop_mmap()` 实际 `htp_munmap()` 的非空表项数 |
| `drop_bytes` | `drop_bytes` | **bytes** | 上述 unmap 的 `m->size` 之和 |
| `map_n` | `map_n` | 计数 | `mmap_buf()` 成功的 `htp_mmap()` 新映射数 |
| `map_bytes` | `map_bytes` | **bytes** | 上述成功映射的 `b->size` 之和 |
| `m_vmem` | `m_vmem` | **bytes** | pass 1 在 drop 前统计的非 extended 已映射字节；**all-reuse 提前返回时为 0** |
| `e_vmem` | `e_vmem` | **bytes** | pass 0 未复用、非 extended 的 buffer 字节；**all-reuse 提前返回时为 0** |
| `max_vmem` | `max_vmem` | **bytes** | drop 判定用的 `ctx->max_vmem`（每 session 常量） |
| `defrag` | `defrag_n` | 计数 | 进入 defrag/pass 2 的次数（0 或 1） |
| `scan_cmp` | `scan_cmp` | 计数 | `reuse_buf()`/`mmap_buf()` 检查的 mmap 表项数（不是 bytes） |

**`m_vmem`/`e_vmem` 的 0 不是“没占内存”**：当 `b_reuse == n_bufs`（JSONL 的
`reuse`）时 `prep_op_bufs()` 在 pass 0 后直接 return，pass 1 根本没跑，所以这两个
字段保持 0，同时 `map_n == drop_n == defrag == 0`。此时用 `n_bufs == b_reuse`
判断 all-reuse。

守恒式（`bench/opbuf_stats.py` 逐行检查；total 行另行核对 `batches` 与各 sum 字段）：

- `defrag == 0`：`n_bufs == b_reuse + map_n`（JSONL 键 `reuse` 即 `b_reuse`）；
- `defrag > 0`：`n_bufs >= b_reuse` 且 `map_n >= n_bufs - b_reuse`。defrag 会把 pass 1
  已映射的非 extended buffer 重新映射一次，所以 `map_n` 可以大于
  `n_bufs - b_reuse`；这时不要套无 defrag 的等式。

`opbuf_total` 的 `batches` 是已输出 batch 行数；除 `max_vmem`（取观测最大值）外，
其余字段都是逐行求和。脚本会核对 `batches`/每个 sum 字段，并列出每个不一致字段的
`got`/`want`。

### ABI：88 → 168 B，host 与 skel 必须同源

- `struct htp_opbatch_rsp` 从 88 B 扩到 **168 B** = 88 B legacy + 8 B
  `abi_magic`/`abi_reserved` + 72 B `struct htp_opbuf_stats`；头文件有
  `sizeof == 168` 的 static assert。
- `abi_magic = HTP_OPBATCH_ABI_MAGIC = 0x4f505331`（`"OPS1"`），
  `HTP_OPBUF_STATS_VERSION = 1`，`abi_reserved = 0`（保留位，当前不校验）。
- 宿主收 rsp 时做**双校验**：先 `rsp_size == sizeof(rsp) && n_dbufs == 1`，再
  `rsp.abi_magic == 0x4f505331`。任一不符都 `GGML_ABORT`，消息分别是
  `response ABI mismatch: got size ... expected size 168 ...` 和
  `response ABI magic mismatch: got ... expected 0x4f505331 ...`；不会静默读短结构。
- **host 库与 v79/v81 skel 必须用同一份头文件重建，发布时整体交付**（lib +
  `libggml-htp-v79.so` + `libggml-htp-v81.so`）。只换 lib 或只换 skel 都会在第一次
  batch rsp 时 abort；旧 host + 新 skel 也过不了旧 host 的 size 检查。
- 这个 ABI 要求**与 `GGML_HEXAGON_OPBUFSTAT` 开不开无关**：rsp 布局变了，开关=0
  也要求同源。部署新 lib 前先确认手机 release 目录里的 skel 是同一构建。

### 工具：`bench/opbuf_stats.py`

```sh
python3 bench/opbuf_stats.py run.err
python3 bench/opbuf_stats.py run.err --bucket-by n_bufs --bucket-width 8 --json summary.json
python3 bench/opbuf_stats.py --self-test
```

- 默认按 `n_bufs` 精确值分桶（`--bucket-width 0`）；`--bucket-width 8` 按 8 的区间
  聚合。输出每字段 p50/p95/max/total、drop/map 总量、守恒结果、total 行核对，
  以及每桶的 `scan_cmp` 分位与 drop/map 总量。
- `--json` 把同一份 summary 写文件（含 `malformed_lines`）；stdout 报告不变。
- 退出码：`0` = 有记录且逐行守恒通过；`1` = 没有任何可解析记录，或守恒失败。
  注意 **total 行缺失/不一致不改变退出码**（只写 `total line: absent|FAIL` 和
  `--json` 的 `total_line`），所以消费方必须显式检查 `total_line`（缺失时为 `null`）
  及其 `ok`/`batches`，不能只看 rc。
- `--self-test` 覆盖 20/100 条守恒与百分位、defrag 例外、total 行校验和混合 stderr
  解析。
- 输入可以是整份 stderr：非 JSON 的普通日志行被跳过；带前缀的 JSON（如
  `ggml-hex: {...}`）会从第一个 `{` 起解析；以 `{` 开头但缺字段/解析失败的行计入
  `malformed`（`--json` 的 `malformed_lines`）。

### 与 `GGML_HEXAGON_VERBOSE=1` 日志代理的区别

不开 OPBUFSTAT 时，可以拿 DSP 的 `mmap :` / `unmap :`（`FARF(ALWAYS)`）数行，
以及 `prep-bufs : pass1 ...`（`FARF(HIGH)`，可能被编译掉）看单 batch 摘要；
旧笔记把这条路径简称 `HEX_VERBOSE=1`。它只是**日志代理**：

- `mmap :`/`unmap :` 是文本行，没有 batch/seq 关联，也没有 `scan_cmp`/`defrag`，
  不能做逐 batch 守恒检查；
- `prep-bufs : pass1` 是 HIGH 级，同源 v79 构建默认可能不出现，不能当必有输出；
- 没有累计行，drop/map 的 bytes 只能从多行文本拼，和计数器口径不可互换。

要 bytes、守恒或 `scan_cmp`，用 `GGML_HEXAGON_OPBUFSTAT=1`；日志代理只适合
“有没有发生 mmap/unmap”这种粗判断。

### 最小采集示例（命令骨架，不跑设备）

下面用具体值 `t0308-r1-A` 作 tag（替换成自己的）；整个 run 在 `devlock` 内串行，
stderr 直接落 `runs/t0308-r1-A-opbuf.jsonl`：

```sh
ssh nubia 'cd ~/csm/moe-exp && mkdir -p runs && \
  GGML_HEXAGON_OPBUFSTAT=1 \
  REL=$HOME/csm/moe/release-v81-chunkpaging-nubia \
  EXPERTS=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q2k-idnow \
  N=128 WINDOW=80 DEPTH=16 READ=8 IO_THREADS=8 MAX_ROWS=8 SPECN=3 \
  ./devlock -w 900 bash ./bench-nubia.sh \
    > runs/t0308-r1-A.out 2> runs/t0308-r1-A-opbuf.jsonl'

# 回到仓库根目录后
mkdir -p runs
scp nubia:'~/csm/moe-exp/runs/t0308-r1-A-opbuf.jsonl' runs/
python3 bench/opbuf_stats.py runs/t0308-r1-A-opbuf.jsonl \
  --bucket-by n_bufs --bucket-width 8 --json runs/t0308-r1-A-opbuf-summary.json
```

- `dropcache`、A/B 轮转、冷却等按 T-0308 的模板走；这里只固定 env 与 stderr 落盘。
  `EXPERTS` 必须与 `dropcache --experts` 一致（见上面的 nubia 单轮示例）。
- 经 `sweep.sh` 跑时，`llama-cli` 的 stderr 被收进 `runs/<TAG>-n<N>.err`；OPBUFSTAT
  的 JSONL 就在这个 `.err` 里，直接 `python3 bench/opbuf_stats.py runs/<TAG>-n<N>.err`
  即可，tag 与同轮 `.out`/TSV 一致。上面的骨架是绕过 sweep 的单跑，所以单独落
  `-opbuf.jsonl`。
- **计时 A/B 的两臂必须用同一个 OPBUFSTAT 设置**：不要一臂开、一臂关——多出的
  stderr 写入和计数本身会改变两臂的 I/O/时序。具体两臂都开还是都关，由 T-0308
  的 spec 决定；本节的示例只用于采集，不作为 t/s 结论。
- 拉回后先看 `conservation: OK` 和 `total line: OK`。若 `total line: absent`，说明
  进程被信号/超时打断，逐 batch 行仍可分析，但不要拿它当完整 run 的总量。

## rc → gate 映射（`sweep.sh` 传播单次 run 的 rc）

`results.tsv`（以及 per-run `results-<run_id>.tsv`）每行最后新增一列 `status`（`ok|invalid`）：**只有解析到 `Generation:`
计时行、且单次 rc ∈ {0,139} 才是 `ok`**，其余一律 `invalid`。同一行的 `rc` 保留原始值，
便于对账；`stats.py` 仍按 `rc∈{0,139}` + 有 `gen_tps` 判有效，多出的 `status` 列不会被
当成数值列。

`sweep.sh` 不再无条件 `exit 0`，而是按单次 bench 的 rc 退出：

| 单次 rc | `sweep.sh` 退出码 | TSV `status` | gate 语义 |
|---|---|---|---|
| 0 | 0 | `ok` | 正常 |
| 139 | 0 | 有计时行→`ok`，否则 `invalid` | teardown segv；计时行已落盘，按有效样本（`stats.py` 接受 0/139） |
| 75 / 124 / 255 | 原样 75 / 124 / 255 | `invalid` | 设备忙 / 超时 / ssh 断链：infra（gate 现有分支映射为 2） |
| 134 / 137 | **2** | `invalid` | Scudo abort / OOM kill：infra 语义（2） |
| 其它非 0（如安全闸 3） | 原样 | `invalid` | 配置/产品失败（1），不是 infra |

> `134/137 → 2` 是 infra 约定码（`CONTRACT.md` §8）。`gate.sh` 的
> `phone_failure_or_fail()` 目前只把 75/124/255 当 infra，rc=2 的分支由 T-0059 的
> gate 改动消费；本任务按 spec **不碰 gate.sh**。在 gate 更新前，134/137 仍可能落到
> “无有效轮次”的产品失败(1)——此时以每行 `status=invalid` 和原始 `.err`
> （`grep -E 'Scudo|OOM|err 114|0x72'`）为准，不要当成候选代码变慢。
>
> 修复前 `sweep.sh` 总是 `exit 0`，所以 gate 看到的是“ssh 成功”，134/137 只会在
> `stats.py` 里表现为 `no valid rows`，无法在采样当场判 infra。

## rc=139 计时 fallback（T-0046）

老构建在 teardown 段 segfault（rc=139）。如果 `[ Prompt: ... | Generation: ... ]` 汇总行
没来得及写到 stdout（缓冲/崩溃时序），`sweep.sh` 会从 stderr 里最后一条 server 计时行恢复：

```
0.06.696.974 I slot print_timing: id  0 | task 0 | prompt eval time = 1752.25 ms / 22 tokens ( 79.65 ms per token, 12.56 tokens per second)
0.06.696.982 I slot print_timing: id  0 | task 0 |        eval time =  682.02 ms /  8 tokens ( 85.25 ms per token, 11.73 tokens per second)
```

- **只**在 `rc=139` 且对应 stdout 字段为 `NA` 时启用；`rc=0` 和其它 rc 完全走原路径，
  `results.tsv` 列数/顺序不变。
- 匹配 `slot print_timing:`（同时兼容 `llama_print_timings:`），gen 用裸 `eval time`、
  prompt 用 `prompt eval time`，取行尾 `, <n> tokens per second` 的数。
- 恢复成功时 `status` 仍按原规则记 `ok`，并向 stderr 打一行
  `[sweep] note: rc=139 ... recovered from stderr print_timing ...` 便于审计；
  恢复失败（stderr 也没有计时行）仍是 `NA` / `invalid`。
- 自测（无需手机，fake bench 覆盖 7 种 rc/输出组合）与真机 rc=139 日志 replay 见队列
  run T-0046 的 `artifacts/test-sweep-fallback.sh`、`replay-real-phone-log.sh`。

## devlock：探测不截断 holder 行

- `devlock` 用 `exec 9<>"$LOCKFILE"`（读写打开、**不带 `O_TRUNC`**）持有锁；只有
  `flock` 成功后才 `: >"$LOCKFILE"` 重写 holder 行，并写回**同一个 inode**
  （`flock` 是按 inode 加锁的，不能换成 rename 新文件）。
- 因此 `devlock -n` / `devlock -w N` 在锁被占用时只读 holder 行，不会再把上一任的
  40B 信息截成 0B；忙碌语义不变（`-n` 立刻 75，`-w N` 等不到 75）。
- 拿锁后的命令 rc 原样传播（`"$@" || rc=$?`），不再被 `set -e` 吞掉。

本地回归（不需要手机）：

```sh
# 终端 A：持锁
DEVICE_LOCK_FILE=/tmp/dl.lock bash bench/devlock -q bash -c 'sleep 30' &
sleep 0.5   # 等 holder 真正拿到锁并写入 holder 行
# 终端 B：两次探测，holder 行必须逐字节不变
before=$(cat /tmp/dl.lock)
DEVICE_LOCK_FILE=/tmp/dl.lock bash bench/devlock -n true;    echo "rc=$?"   # 75
DEVICE_LOCK_FILE=/tmp/dl.lock bash bench/devlock -w 1 true;  echo "rc=$?"   # 75
after=$(cat /tmp/dl.lock)
[ "$before" = "$after" ] && echo "holder intact"
kill %1 2>/dev/null || true
```

排障时仍以 `fuser -v ~/csm/moe-exp/.device.lock` 为准；holder 行只是辅助信息。

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

## 记录模板（speed 必填）

每次 A/B 至少把下面字段写进 run notes / `result.json`；缺一项就不能宣称 speed 结论：

```text
device=... (nubia|vivo)  REL=...  lib_sha256=...  commit=...
bench: N=... WINDOW=... DEPTH=... READ=... IO_THREADS=... MAX_ROWS=... SPECN=... READY=... REFRESH=...
runs: A=... B=...（4+4 逐轮；ABBA/BAAB 顺序）
median: A=... B=... gain=...   A/A 噪声=...
acceptance: A=... B=...   calls_g: A=... B=...   accepted/generated: ...
repeat: A=<同一句重复次数/最长重复片段>  B=...
cpu_cap_mhz / battery / temp: ...
headroom_min_mb / vmswap_peak_kb: ...
evidence: runs/results-<run_id>.tsv（vivo）或 runs/<tag>.out|.err + .dropcache.json（nubia）
```

- `acceptance` / `calls_g` 从 llama-cli 的 speculative-decoding 日志行取
  （T-0070 的 `tools/moe/bench_attrib.py` 可解析），**不要只抄 t/s**。
- 重复度至少记“同一句重复次数 / 最长重复片段”；T-0079 的证据是 baseline 在 rain prompt
  上重复 14–17 次、weighted 臂只重复 1 次，同时 acceptance 0.902 vs 0.556、calls 34 vs 48。
- 这些字段覆盖 `docs/dpsk-queue/GATES.md` §3.4 的记录项，并额外要求
  acceptance/重复度（GATES 目前只要求 `cpu_cap_mhz`/内存/TSV；T-0079 后续）。

## 规则

- 一次只跑一个 HTP 实验：批量/对照一律 `devlock` 串行。
- 只有**轮转**的多臂对照（≥4 轮、组间不重叠）才能宣称端到端收益；单次采样只算机制结果。
- 每轮 dropcache 必须显式 `--model/--experts/--draft`；裸 `dropcache.py` 只覆盖默认目录
  `*.gguf` + `experts-htp-tiled/*.bin`，**不覆盖 `experts-*` 臂目录**（T-0082）。手机副本
  未升级到 T-0095 时不要声称冷缓存（见「nubia 最简跑法」开头）。
- speed 结论必须同时记录 acceptance/calls/重复度；baseline 输出退化会把 acceptance 抬高，
  看起来像“更快”（T-0079）。
- 电量 <30% 或 `/data` <6 GiB：只做 4 跑或先充电，**不声称 8 轮**。电量/温度用
  `adb -s <serial> shell dumpsys battery`（Termux 读 `/sys/class/power_supply` 会
  `Permission denied`，T-0098 实测）。
- gate full 自 T-0100 起**支持 nubia**：`GATE_PHONE=nubia` 自动用本目录的
  `bench-nubia.sh` + v79 release/skel，缺脚本/v79 skel/release 时 preflight `exit 2`。
  手工 bench 结论仍标 `nubia-only`，并引用
  [`docs/dpsk-queue/GATES.md` §3.3](../docs/dpsk-queue/GATES.md)。
- 老构建（`release-v81` / `release-v81-pipe`）在 teardown 段 segfault（rc=139），
  **CLI 的 `[Prompt|Generation]` 行可能来不及打印**，t/s 只在 stderr 的
  `slot print_timing: ... eval time = ... tokens per second` 里 —— `sweep.sh` 已有
  rc=139 fallback（见上节），不会再把这些样本记成 `NA`。

## 待补（TODO）

- `ab.sh`：多臂轮转 + "组间不重叠"判定 + 计数器（bytes/miss/hit/prefetch/refresh）落 TSV。
- 结果 JSON（`runs/<tag>.json`），字段固定，便于后续分析脚本直接消费。
- 把 T-0095 的 `bench/dropcache.py` 部署到 vivo/nubia，并把 `--print-scope` 实测结果补进本文
  （当前两台手机仍是旧 41 行脚本，sha256 `300d4df…`；T-0098）。
- `GATES.md` §0 的 bench md5 修正仍误引 `T-0042`（T-0042 实际是“验证 T-0014”；nubia gate
  由 T-0100 跟踪）；`GATES.md` 由 gate 任务维护，本文件不把 `T-0042` 当 nubia gate 任务号。
