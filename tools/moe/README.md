# MOE 读账本工具

## `analyze_moe_reads.py`

离线解析 `[moe-ssd]` 读账本日志（Python 3 stdlib only，不访问手机/A100/网络）。
它取代 T-0078 run 目录里的 `analyze.py`：后者把所有逐步字段都除以 `span`，但
`runs=` / `miss=` 是**最后一个 ubatch** 的 `cur` 值，不能除。

### 日志字段口径

| 字段 | 来源 | 语义 | 除以 `span`？ |
|---|---|---|---|
| `calls=` | `io_calls_all` 全局增量 | 自上次 flush 的所有 `readv_span` 调用（含预读） | **是** |
| `pfread=` | `lookahead.n_pred_read` 全局增量 | 自上次 flush 的预读专家数 | **是** |
| `bytes=` / `pread_sum=` | 全局增量 | 读字节 / 预读耗时 | **是** |
| `runs=` | `cur.runs.n` | **最后一个 ubatch** 的 demand 物理连续段数 | **否** |
| `miss=` | `cur.miss_sum` | **最后一个 ubatch** 的 pending 专家数 | **否** |
| `avg_run=` / `scatter=` | `cur` 派生比率 | 形状指标 | 否 |
| `span=` | `idx - last_flush_idx` | 本行覆盖多少个 ubatch | — |

关键点：`cur` 在每个 ubatch（无论是否打印）后清空，所以 `span=8` 的 `#16` 行里
`calls/pfread` 是 8 步聚合，而 `runs/miss` 只是第 8 步本身。SPECN=3 的 verify 批是
4 行，运行时也打成 `prefill #… tok=4`；真 prefill 是 `tok=2/18`。解析器按 `tok`
而不是按日志前缀区分二者。

源码定位（行号随 HEAD 漂移，可用 `grep -n` 重新找）：
`src/llama-moe-slot.cpp` 的 `n_pred_read`、`io_calls_all`、`last_flush_idx`、
`cur.runs` / `cur.miss_sum`、`flush_ubatch` 的 `cur` 清零处。

### 两种可引用口径

- **published**：丢掉 `#3`（第一个 decode 窗口）；`calls/pfread/bytes/pread` 除以
  `span`，`runs/miss` 保持原值；对 per-run 的 per-step 值取中位数。
  T-0078 的 published 表由 5 个 `span=1` 窗口主导，所以 runs/miss 中位数与旧脚本
  相同（96.5 / 86.5 / 101.5），但 `#16` 行不再被错误地记成 5.75 / 6.0。
- **total**：进程级 `merge-scan.runs` 减去真 prefill（`tok!=4`）的 `runs`；
  `calls/pfread` 用 `#3-#8 + #16` 的精确增量求和，再除以 `sum(span)`。
  `#9-#15` 被 logger 节流，只能由 `#16` 的 8 步聚合代表。

T-0096 的验收集（A/C/B）：published `runs/step` 96.5 / 86.5 / 86.5，
`calls/step` 373 / 373 / 355.5，`pfread/step` 262；total `runs/round`
124.96 / 110.71 / 110.71，`calls/round` 299.04 / 298.93 / 283.04。
`pfread/calls` 在 total 口径是 63.2–66.7%，published 口径是 70.2–73.7%——
两个量的分母不同，引用时必须写清口径。

### 供其他脚本复用的 API（T-0126）

`exps/roofline-saturate-20261105/collect/ledger_adapter.py` 只做薄适配，不复制解析：

- `parse_file(path, with_totals=False)`：默认输出不变；`with_totals=True` 额外暴露
  `task_line` 与进程级 `total[<kind>]`（`miss/bytes/pread`）行，供阶段切分使用。
- `parse_ledger(path)`：等价于 `parse_file(..., with_totals=True)`，是适配层唯一入口；
  `parse_run()` 在此基础上只做 T-0097 的 P/G 切分，不新增账本 regex。
- `summarize_run(parsed)` 的 `published` / `total` 语义不变；roofline 的
  `--stage-split` 是第三种问题（整段 decode 累计账 = `total[final]` − 真 prefill），
  由适配层显式登记为 `scope("stage-split")`，不要把它混进 published 中位数。

### 用法

```sh
# 一个 runs/ 目录（也接受含 runs/ 子目录的 phone artifact 目录）
python3 tools/moe/analyze_moe_reads.py \
  --runs /home/yydh/.local/state/dpsk-queue/runs/T-0078/run-20261122T092959-w1-1/artifacts/phone/runs \
  --out /tmp/ledger.json --tsv /tmp/ledger.tsv

# 一个或多个文件
python3 tools/moe/analyze_moe_reads.py --err idA-r1-n64.err --tag idA-r1 --out /tmp/x.json

# T-0078 离线自检（<60s，不碰设备；写 selftest.json/selftest.txt）
python3 tools/moe/analyze_moe_reads.py \
  --runs /home/yydh/.local/state/dpsk-queue/runs/T-0078/run-20261122T092959-w1-1/artifacts/phone/runs \
  --out "$QUEUE_RUN_DIR/artifacts/ledger.json" --self-test
```

选项：

| 选项 | 说明 |
|---|---|
| `--runs DIR` | runs 目录、phone artifact 目录；可重复 |
| `--err FILE` | 单个 `.err`；可重复 |
| `--tag TAG` | 只解析该 tag（单文件时也作为标签） |
| `--out FILE` | 输出完整 JSON 账本 |
| `--tsv FILE` | 每行一个 flush 窗口的 TSV |
| `--self-test` | 用 T-0078 `runs/` + `analysis.json` 做离线对账，exit 0 = 通过 |
| `--legacy-span` | **LEGACY/DO-NOT-CITE**：连 `runs/miss/syscalls` 也除以 `span`，复现旧 `analyze.py` |
| `--allow-partial` | 缺字段时仍 exit 0（JSON 里保留 `missing_fields`） |

默认遇到缺失/不完整日志（缺 `calls/pfread/runs/miss/span`、缺 decode 窗口、
缺 `merge-scan.runs` 等）会在 JSON 写 `missing_fields` 并以非 0 退出，不会静默当 0。
带 `results.tsv` 且 `status!=ok` 的采样会被列进 `skipped`，不参与统计。

### JSON 结构（节选）

```
{
  "schema": "moe-read-ledger-v1",
  "legacy_span": false,
  "citation": "OK-TO-CITE",
  "arms": {
    "<label>": {
      "published": {"runs_per_step", "calls_per_step", "pfread_per_step", "miss_per_step", ...},
      "total": {"runs_per_round", "calls_per_round", "pfread_per_round", ...},
      "prefetch_fraction": {"published", "total"},
      "merge_scan": {...}, "merge_exec": {...}, "per_run": [...]
    }
  },
  "files": {"<tag>": {...}},
  "records": [{"tag","idx","tok","span","kind","raw", "per_step", "legacy_per_step"}],
  "merge_scan": {...}, "merge_exec": {...},
  "missing_fields": [], "skipped": []
}
```

`--self-test` 写 `$QUEUE_RUN_DIR/artifacts/selftest.json` 与 `selftest.txt`，检查：

1. A/C/B 的 published/total 复现（中位数容差 ±0.51）；
2. `idA-r1 #16` 的 `runs=46`、`miss=48`（不除 8），`calls/step=195.0`、
   `pfread/step=131.0`；
3. 7 个臂 × 7 个 published 字段与 T-0078 `analysis.json` 对账（旧值只用于对照）；
4. `--legacy-span` 确实复现 `#16` 的 5.75 / 6.0，且默认输出带 `OK-TO-CITE`。

---

## `compare_moe_runs.py`

离线比较两个 `[moe-ssd]` 读账本 run（Python 3 stdlib only，不访问手机/A100/网络）。
它 **import** `analyze_moe_reads.py` 复用逐行解析与 published/total 口径，不修改后者的语义；
`calls/pfread/bytes/pread` 仍按 `span` 归一，`runs/miss` 保持 per-ubatch 原值，`tok==4`
仍识别为 SPECN=3 verify（日志前缀是 `prefill`），并与真 prefill（`tok=2/18`）分开输出。

### 用法

```sh
# R4 的 M0P2 vs M0P0（每个 run 目录里有 idA/ordB/... 多个 tag）
python3 tools/moe/compare_moe_runs.py \
  --a "$QUEUE_HOME/runs/T-00xx/<run>/artifacts/phone/runs" \
  --b "$QUEUE_HOME/runs/T-00yy/<run>/artifacts/phone/runs" \
  --tags-a idA --tags-b ordB \
  --json "$QUEUE_RUN_DIR/artifacts/compare.json" \
  --markdown "$QUEUE_RUN_DIR/artifacts/compare.md" \
  --scope published --reps 10000 --seed 20261122

# 同一文件自比：所有 delta=0、CI=[0,0]、p=1
python3 tools/moe/compare_moe_runs.py --a idA-r1-n64.err --b idA-r1-n64.err \
  --json /tmp/self.json --markdown /tmp/self.md

# 总量口径（per-run，按 replicate 配对；不是 per-step）
python3 tools/moe/compare_moe_runs.py --a <dir> --b <dir> \
  --tags-a idA --tags-b ordA --scope total \
  --json /tmp/total.json --markdown /tmp/total.md

# 离线自检（读 T-0078 + T-0110 ledger；exit 0 = 通过）
python3 tools/moe/compare_moe_runs.py --self-test
```

| 选项 | 说明 |
|---|---|
| `--a/--b FILE_OR_DIR` | 一个 `.err`，或 run/artifact 目录（自动找 `runs/` 子目录） |
| `--tags-a/--tags-b TAG` | 选择臂；可逗号分隔/重复，但每侧只允许一个臂 |
| `--scope published` | 默认。丢第一个 decode 窗口；`calls/pfread/bytes/pread` 除以 `span`，`runs/miss` 原值；按窗口 `#N` 配对 |
| `--scope total` | 进程级 per-run 总量；`runs = merge-scan.runs - 真 prefill runs`；按 replicate 配对，`n=run 数` |
| `--reps N` | bootstrap 次数，默认 10000 |
| `--seed N` | bootstrap 固定种子，默认 20261122；同一输入+种子 CI 完全一致 |
| `--json FILE` | 机器可读结果（schema `moe-read-compare-v1`） |
| `--markdown FILE` | 人读表格（A/B 中位数、ratio、delta、CI、符号检验、verify/prefill 分解） |
| `--self-test` | T-0078 A/C/B 复现 T-0110 published/total；自比；缺/重/乱序 step |
| `--selftest-dir DIR` | selftest.json/txt 输出目录（默认 `$QUEUE_RUN_DIR/artifacts`） |

### 配对与错误处理

- `published`：配对单位是窗口号 `#N`；每个 step 的 log-ratio 先对匹配的 replicate
  （`r1/r2/...`）取中位数，再对 step 取中位数 / 做 bootstrap。T-0078 有 7 个
  `tok==4` 窗口，丢掉冷启动 `#3` 后 `n_pairs=6`。
- `total`：只有 per-run 总量，配对单位是 replicate（run）；T-0078 四轮 `n_pairs=4`。
  这是 **不是** per-step 统计，Markdown/JSON 都显式标注。
- 一个臂的 tag 只匹配 `TAG` 本身或 `TAG-rN`；`ordA-v` 这类 verify-slot 采样不会被
  混进 `ordA` 的 replicate。
- A/B 的 replicate 集合必须一致；每个 replicate 的 decode step 集合必须一致。
- **缺失 step、重复 `#N`、`#N` 倒退（乱序）、缺 `calls/pfread/runs/miss/span`、
  无法按 `tok` 分类** 一律 exit code 2，且在写 JSON/Markdown **之前** 失败，
  不会留下“看似有效”的结果。
- `n_pairs < 4` 不算 CI，metric 里 `insufficient_pairs=true`，顶层 `status=insufficient_pairs`；
  仍给出 A/B 中位数与（若有）符号检验。
- 计数为 0 时 log-ratio 用 Haldane-Anscombe `+0.5` 修正，并在 metric 的
  `zero_corrected_pairs` / `warnings` 中记录；两边都是 0 视为无差异。

### 统计口径

- `ratio_median = exp(median(paired log-ratio))`；`delta_pct = (ratio_median - 1) * 100`。
- 同时给 `delta_pct_of_medians = (A 中位数 / B 中位数 - 1) * 100`。两者分母不同：
  前者是逐 step 配对后的中位数，后者是臂级中位数之比，引用时必须写清。
- `ci95_log_ratio` 是固定种子 percentile bootstrap 2.5/97.5；`ci95_ratio = exp(CI)`。
- `sign_test` 是精确单侧二项检验，ties（log-ratio 正好 0）不计入 `n_eff`：
  `p_one_sided_less` 对应 H1 “A<B”（读更少），`p_one_sided_greater` 对应 H1 “A>B”。
  全部 ties 时两者都是 1.0。

### JSON 结构（schema `moe-read-compare-v1`）

```
{
  "schema": "moe-read-compare-v1",
  "status": "ok|insufficient_pairs",
  "scope": "published|total",
  "scope_note": "...",
  "inputs": {
    "a|b": {"path","tags","replicates",
            "files":[{"tag","replicate","file","steps"}], "skipped":[...]}
  },
  "pairing": {"unit":"window_index|replicate","steps","replicates",
              "dropped_first_decode","note"},
  "stats": {"reps","seed","ci_method","min_pairs_for_ci","insufficient_pairs"},
  "metrics": {
    "<calls|pfread|runs|miss|syscalls|bytes|pread_sum|avg_run|demand_calls>": {
      "field": "calls_per_step|...",
      "a_median": 0.0, "b_median": 0.0,
      "ratio_median": 0.0, "log_ratio_median": 0.0,
      "delta_pct": 0.0, "delta_pct_of_medians": 0.0,
      "ci95_log_ratio": [lo, hi] | null,
      "ci95_ratio": [lo, hi] | null,
      "sign_test": {"n_positive","n_negative","n_zero","n_eff",
                    "p_one_sided_less","p_one_sided_greater"},
      "n_pairs": 6, "insufficient_pairs": false,
      "zero_corrected_pairs": 0,
      "pairs": [{"unit":"#4","a":...,"b":...,"log_ratio":...,"ratio":...,
                "ratio_of_medians":...}]
    }
  },
  "decomposition": {
    "a|b": {
      "verify": {"definition","windows_total","windows_used","indices","indices_all",
                 "span_sum","per_step_median":{...},"runs_sum_per_replicate_median"},
      "prefill": {"definition","windows_total","indices","indices_all","span_sum",
                  "per_step_median":{...},"runs_sum_per_replicate_median"},
      "process_totals": {"merge_scan_runs_median","prefill_runs_median",
                         "decode_runs_total_median","rounds_median"}
    }
  },
  "validation": {"a_steps","b_steps","a_prefill_steps","b_prefill_steps",
                 "missing_steps_in_a","missing_steps_in_b","duplicate_steps","order_ok"},
  "warnings": [...]
}
```

`decomposition` 的 `verify` 只含 `tok==4`，`prefill` 只含 `tok!=4`；两者 indices
不相交。`process_totals` 的 `prefill.runs` 就是 total 口径里从 `merge-scan.runs`
减掉的那部分。

### `--self-test`

读 `T-0078/.../artifacts/phone/runs` 与 `T-0110/.../artifacts/ledger.json`，
写 `$QUEUE_RUN_DIR/artifacts/selftest.{json,txt}`。检查：

1. A/C/B published `runs/step` 96.5/86.5/86.5、`calls/step` 373/373/355.5、
   `pfread/step` 262，容差 **±0.05（仅显示舍入）**；不符即 exit 非 0，并打印期望/实际。
2. A/C/B total 与 T-0110 `ledger.json` 的 `arms.<label>.total.*` 浮点值按 1e-9 对账，
   同时核对 124.96/110.71/110.71、299.04/298.93/283.04、188.86/188.79/188.86。
3. 同一臂自比：published（6 个 step）与 total（4 个 replicate）所有 `delta_pct=0`、
   `ci95_log_ratio=[0,0]`、两个单侧 `p=1`；单文件 published 自比同样成立（total 单文件
   只有 1 个 pair，按 n<4 规则不给 CI）。
4. verify/prefill 分解：verify indices `[3,4,5,6,7,8,16]`、prefill `[1,2]` 不重叠；
   A/C 的 prefill runs 为 2177/1773。
5. 固定种子 bootstrap 两次结果完全一致。
6. 缺 step、重复 step、乱序 step：CLI 都返回 exit 2，且不写 JSON。

---

## `q1_provenance.py` — tiled 包 provenance sidecar（Q1_0 防静默回退）

weighted / unweighted Q1_0 的 tiled 包**大小、布局、类型完全相同**，只有专家字节
（以及 manifest 的 `files_sha256`）不同；把 unweighted 包当 weighted 部署会静默损失
PPL。本工具只写一个独立的 `provenance.json`，**不改** `manifest.json` / `layer_*.bin`：

```sh
# 记录 arm/imatrix/rules/quantizer/manifest 的 sha 与 weighted 标志
python3 tools/moe/q1_provenance.py stamp \
  --tiled tiled/a-w --arm models/a-w.gguf \
  --imatrix imatrix.gguf --rules rules/q1-all.txt \
  --quantizer-src ggml/src/ggml-quants.c \
  --weighted-q1-0 auto --out tiled/a-w/provenance.json

# unweighted 臂没有 imatrix：省略 --imatrix，sidecar 写 imatrix_sha256/bytes=null
python3 tools/moe/q1_provenance.py stamp \
  --tiled tiled/a-u --arm models/a-u.gguf \
  --rules rules/q1-all.txt --weighted-q1-0 false \
  --out tiled/a-u/provenance.json

# 校验 sidecar 自洽 + manifest/120 entries/40 层/backend_layout +
# 逐个流式校验 layer_*.bin 的 sha256
python3 tools/moe/q1_provenance.py verify \
  --tiled tiled/a-w --provenance tiled/a-w/provenance.json \
  --expect-arm-sha "$ARM_SHA" --expect-imatrix-sha "$IMATRIX_SHA"
```

- `weighted_q1_0`：`auto` 只读 arm GGUF 的 imatrix 相关 KV（`quantize.imatrix.*` /
  `imatrix.*`）；有则记 `true`（并保留 `arm_imatrix` 原始值），没有则记 `unknown`，
  **绝不按文件大小推断**。部署声明应显式传 `--weighted-q1-0 true|false`。
- `--imatrix` 可选（T-0088）。省略时 sidecar 的 `imatrix_sha256`/`imatrix_bytes`/
  `source_paths.imatrix` 都是 `null`，且只允许 `weighted_q1_0=false|unknown`；显式传
  `--weighted-q1-0 true` 会以 exit 2 拒绝。`verify` 接受 null；若同时给了
  `--expect-imatrix-sha`，对 null sidecar 以 exit 3 失败。
- 旧包没有 sidecar：`verify` 打印 warning、仍校验 layer sha 并退出 0（无论 `--provenance`
  是默认路径还是显式给出）；加 `--require-provenance` 才失败。
- 退出码：`0` 通过；`2` 输入/基础设施错误（缺文件、坏 JSON、CLI 参数错）；
  `3` 校验失败（manifest/layer/布局/计数/`--expect-*` 不匹配，或要求 provenance 但缺失）。
- `verify` 只读并流式读取 layer 文件，不把 4.5 GB 包载入内存；`stamp` 保护
  `manifest.json` / `layer_*.bin` 不被 `--out` 覆盖。
- **`quant_arm.py` 集成**：arm JSON 的 `pack.enabled=true` 时，默认在 `pack` 后追加
  `stamp` + `verify --require-provenance`（`provenance.enabled=false` 可关闭；
  `provenance.quantizer_src` 可覆盖 quantizer 源）。`run` 把 sidecar 路径/sha、
  `verify_ok`、`weighted_q1_0` 和 arm/imatrix/rules sha 写进
  `arm-result.json.provenance`；stamp/verify 失败会让本次 run 失败（`status=failure`，
  `provenance.status=stamp_failed|verify_failed`）。`--resume` 只在既有 sidecar
  verify 通过时跳过 stamp，sidecar 缺失或损坏会重跑。
- 测试：`python3 -m unittest tests.test_q1_provenance -v`（或
  `python3 -m pytest tests/test_q1_provenance.py -q`）；
  `python3 -m unittest discover -s tools/moe/bench -p 'test_quant_arm_provenance.py' -v`。

## `bench_attrib.py`

离线解析 bench 日志里的 `slot print_timing` / `statistics draft-dspark` /
`[ Prompt: .. | Generation: .. ]`，每个 session 一行固定 TSV，并按 arm 给
median/min/max。支持一个 log 内多段 ABBA；arm 先用文件名 regex，再用
`[run-start]` marker 序列配对。字段口径、`per_call_ms` 的两种定义和 T-0032
0.902→0.556 / 34→48 / 0.7124 复算见 [`bench/ATTRIB.md`](bench/ATTRIB.md)。

```sh
python3 tools/moe/bench_attrib.py --logs /path/to/runs \
  --out sessions.tsv --json > attrib.json
python3 -m unittest discover -s tools/moe/bench -p 'test_bench_attrib.py' -v
```

## `verify_io.py` — verify 四段分解（`profile-data.json`）

把一次 profile 采集里的 **verify cycle 墙钟**严格拆成
`overlap / dsp_only / read_only / neither` 四段，纯 stdlib、离线、只读输入。
它是 T-0340 一次性脚本 `analyze_verify_io.py` 的固化版本；输入是
`profile-data*.json`，不需要手机、DSP、构建或网络。

### 输入

- `forwards[]` 中 `phase=="verify"` 的条目，每个是一个 cycle，`[a,b)` 单位 µs；
- `events[]` 按 `fw == forwards[].id` 归属 cycle，`session` 默认全收，
  `--session HTP0` 只收该 session；
- `pagecache[]` 按 `fw_ids` 包含 cycle id 归属；`prefetch==false` 为 demand，
  `true` 为 prefetch。

### 口径

```text
wall      = b - a
dsp       = |union(events [a,b))|
read      = |union(demand [a,b)) ∪ union(prefetch [a,b))|
overlap   = |dsp ∩ read|
dsp_only  = |dsp − read|
read_only = |read − dsp|
neither   = wall − |dsp ∪ read|          # 宿主空洞 / 无 I/O
四段之和 == wall（容差 1e-6）
demand_share = |demand| / |read|          # 时间并集比
eff_gbs      = Σ(nonresident_bytes) / |read| * 1e-3
```

- **建模口径，不是硬件计数器。** `dsp` 是 host 校准 qtimer 的 event 并集，
  event 自带 `placement_error_us`；`read` 是 host `preadv` 时间并集，不是
  UFS 利用率。不能据此宣称“DSP 精确忙时”或预测 t/s。
- 跨 cycle 的 read 仍计入该 cycle；四段分解只用落在 cycle 内的部分，超出部分
  记在 JSON `clipped_us`，不会静默丢弃。若一个 read 的 `fw_ids` 命中多个
  verify cycle，它在每个 cycle 的 `reads/bytes` 里各计一次，summary 是逐
  cycle 求和。
- `demand_share` 按 spec 是**时间并集**比；字节比另见 summary 的
  `demand_byte_share` / `demand_bytes` / `prefetch_bytes`。11-10 两者分别约
  55.8% 和 44.1%——T-0340 的 44.1/55.9 是**字节**口径，引用时必须写清。
- `eff_gbs` 按 spec 只用 `nonresident_bytes`（真正从设备读的字节）。
  `eff_gbs_total_bytes` 用 `bytes`（含 resident/partial），用于和 T-0340
  的 cyc4 3.13 GB/s 对账；11-10 / 11-04 两份全 cold，两者相同。
- `--per-layer` 输出每层 demand/prefetch 的 `n/p50/p90/max`，读的是
  **单次读延迟 `b-a`**，不是等待时间；percentile 用 T-0340 的
  `sorted[round((n-1)*p)]`。只用于复现 L04/L05 异常，不做因果推断。

### 用法

```sh
# 人类摘要（每 profile 的 cycles、wall、四段百分比、eff GB/s）
python3 tools/moe/verify_io.py \
  --profile /path/to/profile-data-20261110.json

# 多 profile + 逐 cycle TSV + 完整 JSON + per-layer
python3 tools/moe/verify_io.py \
  --profile profile-data-20261110.json \
  --profile profile-data-20261104.json \
  --out-dir /tmp/verify-io --per-layer

# 显式指定输出，只统计 HTP0
python3 tools/moe/verify_io.py --profile p.json --session HTP0 \
  --out-tsv /tmp/verify-io.tsv --out-json /tmp/verify-io.json
```

`--out-dir DIR` 默认写 `verify-io.tsv`、`verify-io.json`（加 `--per-layer`
时再写 `verify-io-per-layer.tsv`）；显式 `--out-tsv/--out-json` 优先。

TSV 列固定为
`profile,cycle,fw,wall_ms,dsp_ms,demand_ms,prefetch_ms,read_union_ms,overlap_ms,dsp_only_ms,read_only_ms,neither_ms,demand_share,eff_gbs,clipped_us`。
JSON 为 `{version:1, definitions:{...}, profiles:[{path,cycles:[...],summary:{...}}]}`
（`cycles[]` 另含 raw read union、bytes、clipped 等字段，summary 另含
`demand_byte_share`、`eff_gbs_total_bytes`、`skipped/skipped_reasons`）。

空输入、零 read union 输出 0 而不是除零；坏 JSON、缺 `pagecache`、坏 entry
计入 `skipped` 并打印到 stderr，不静默丢。

### 回归（T-0340 输入，只读）

| capture | cycles | wall ms | DSP busy | read covered | overlap | dsp_only | read_only | neither | eff (nonres) | eff (all bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11-10 | 8 | 1688.8 | 39.94% | 86.66% | 32.04% | 7.90% | 54.63% | 5.43% | 3.514 | 3.514 |
| 11-04 | 1 | 309.7 | 25.32% | 93.02% | 22.78% | 2.54% | 70.24% | 4.44% | 3.555 | 3.555 |
| 11-04 cyc4 | 4 | 1441.6 | 26.10% | 80.36% | 17.46% | 8.65% | 62.91% | 10.99% | 2.339 | 3.125 |

11-10 与 T-0340 §4.1 逐项一致；11-04 两份的 DSP busy 25–26%、纯读等待
63–70% 方向一致。cyc4 有 489 次 resident / 53+ 次 partial 读（共 0.77 GB），
非驻留口径 eff 为 2.34 GB/s，按全部 `bytes` 复算才是 T-0340 的 3.13 GB/s。

测试：

```sh
python3 -m unittest tools/moe/test_verify_io.py -v
python3 -m py_compile tools/moe/verify_io.py
```

## `analyze_rtt.py` — RTT v1/v2 聚合（graph / UID / prep）

离线聚合 `GGML_HEXAGON_RTT_TRACE=<path>` 写出的 host-only JSONL trace，纯 stdlib，
不碰设备/DSP。输入同时接受两种 schema：

- `ggml.hex.rtt.v1`：T-0181 以来的原始格式；键名、顺序、校验规则不变。
- `ggml.hex.rtt.v2`（T-0327，`GGML_HEXAGON_RTT_TRACE_V2=1`）：在 v1 键之后追加
  `graph_uid`（uint64，打开该 batch 的 graph）、`graph_cache_hit`（bool，host 侧
  `cached_uid` 是否命中）、`graph_prep_ns`（uint64，miss 时 host graph prep 墙钟；
  hit 为 0）、`graph_nodes`（uint32，该 graph 的节点数）。这些字段是 **batch 级快照**，
  一个 graph 可产生多个 batch，batch 数与 graph 数不是一一对应。

dropped 行（ring wrap / 无 live slot）在 v2 里四个 graph 字段为 `null`；analyzer
不会把它们伪造成 0/false，也不计入 `graph_records`。非 dropped 的 v2 行缺字段报
`missing-field`，类型/范围不对（含负 `graph_uid`、非 bool 的 `graph_cache_hit`）报
`bad-field`；v1 行完全不参与 graph 分母。

混合 v1/v2 文件按行分别计数：顶层 `input_schemas` 与每个 trace 的 `schemas` 给出
两种 schema 的 valid 行数；每个 group 的 `graph.records` 只统计非 dropped v2 行，
`graph.cache_hit_rate = cache_hits / records`（无 v2 行为 `null`）。`AGG_SCHEMA` 仍是
`ggml.hex.rtt.agg.v1`，只追加 `input_schemas` 和 group 的 `graph`
（`records/cache_hits/cache_hit_rate/unique_graph_uids/prep_ns{p50,p95,mean,max}/nodes{...}`）。
TSV 旧列顺序不变，末尾追加
`graph_records,graph_cache_hits,graph_cache_hit_rate,graph_unique_uids,graph_prep_p50_ns,graph_prep_p95_ns,graph_nodes_dist`。

```sh
# 人类摘要（schema 与 cache hit rate；JSON 仍写文件）
python3 tools/moe/analyze_rtt.py trace-v2.jsonl --summary

# 标准用法：JSON + TSV
python3 tools/moe/analyze_rtt.py trace-v2.jsonl --out-json rtt.json --out-tsv rtt.tsv
```

```sh
python3 -m unittest -v tools.moe.test_analyze_rtt
python3 -m py_compile tools/moe/analyze_rtt.py
```
