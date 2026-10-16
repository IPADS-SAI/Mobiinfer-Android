# Hot expert profiling / 按层窗口规划

- **需求**：用真实数据采 MoE 路由，得到**每层、每个专家的出现次数**，据此判断**哪些层热专家多、哪些层冷专家多**，并产出一份 **`LLAMA_MOE_LAYER_WINDOWS` 建议**（每层 decode 窗口开多大）。
- **工具**：`tools/moe/profile_hot_experts.cpp`（`llama-moe-profile`）+ `tools/moe/summarize_hot_experts.py`。
- **一句话**：`LLAMA_MOE_LAYER_WINDOWS` 就是"每个 MoE 层 decode 窗口开多大"的按层覆盖开关。

---

## 1. 需求背景

- 之前只用 128 条 prompt，采到的数据太少，且只产出"排名 + 一个覆盖率数字"，看不出**每层冷热分布**。
- 现在要用 **A100 上的「日常回答」数据集**（经 chat 模板、非思考、贪心）采更大规模的路由。
- 输出要能直接回答：
  1. 每个专家的**出现次数**（每层）；
  2. 哪些层**热专家多**（激活分散、需要更大窗口）、哪些层**冷专家多**（激活集中、窗口可小）；
  3. 一份可直接粘贴的 `LLAMA_MOE_LAYER_WINDOWS="0:200,1:96,..."`。

---

## 2. 采集：`llama-moe-profile`

输入 JSONL，每行必须有唯一 `id` 和 `prompt`：

```json
{"id": "daily-0001", "prompt": "帮我安排一个周末活动"}
```

用法：

```sh
cmake --build build --target llama-moe-profile
# 训练集（用于排名）与留出集（用于验证泛化）必须不重叠
build/bin/llama-moe-profile model.gguf train.jsonl  train-routes.jsonl
build/bin/llama-moe-profile model.gguf heldout.jsonl heldout-routes.jsonl
```

环境变量（默认与旧行为一致）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `LLAMA_MOE_PROFILE_CTX` | 2048 | 上下文长度（256–32768） |
| `LLAMA_MOE_PROFILE_NGEN` | 64 | 每条 prompt 生成多少 token（0–512）——**调大可采到更多 decode 路由** |

> 采路由会同步后端（`ggml_backend_tensor_get`），所以**这些运行不能当性能基准**。

输出 `ROUTES.jsonl`，每行一条记录：

```json
{"sample":"daily-0001","phase":"prefill","layer":17,"experts":[[203,222,74,...],[...]]}
```

`experts` 是**每行一个 8 专家集合**；prefill 一次一个 chunk、decode 每 token 一行。

---

## 3. 汇总：`summarize_hot_experts.py`

```sh
python3 tools/moe/summarize_hot_experts.py train-routes.jsonl heldout-routes.jsonl out \
    [--slot-budget 3840] [--min-slots 8] [--max-slots 1024]
```

校验：空轨迹、非法 top-8、40 层覆盖不全、训练/留出 id 重叠都会报错。

**产出文件**：

| 文件 | 内容 |
|---|---|
| `prewarm-ranked.txt` | 40 行，每行一层，256 专家按**训练 decode 频率**降序 |
| `coverage.json` | 按容量的覆盖率（train/heldout × prefill/decode），含轨迹 sha256 |
| **`expert-counts.tsv`** | **每层每专家的出现次数**：`layer, expert, train_prefill, train_decode, heldout_prefill, heldout_decode` |
| **`layer-skew.json`** | **每层冷热分布统计**（train/heldout × prefill/decode） |
| **`layer-windows-suggested.txt`** | 一份 `LLAMA_MOE_LAYER_WINDOWS="..."` 建议 |

`layer-skew` 每层的字段：`total`（该层激活总次数）、`distinct_active`（出现过≥1次
的专家数）、`zero_count`（0 次专家数）、`cold_lt_0.1mean` / `cold_lt_0.01mean`（低于
均值 10%/1% 的冷专家数）、`entropy_bits`/`entropy_norm`、`effective_experts`（参与度
比 `1/Σp²`）、`top1_share`/`top8_share`、`n_50pct`…`n_99pct`（覆盖 50–99% 激活所需专家数）。

**终端会打印一张排序过的表**（按 `n_95pct` 降序，即"最需要大窗口的层"在前），列：
`layer dec_total active zero cold<0.1m n80 n90 n95 eff top8`。

---

## 4. 窗口建议 → `LLAMA_MOE_LAYER_WINDOWS`

推荐规则：**每层窗口 ≈ 该层覆盖 95% decode 激活所需专家数 `n_95pct`**（窗口越大，能常驻的热专家越多、miss 越少）。

- 不指定 `--slot-budget`：`c_il = clamp(n_95pct, min-slots, max-slots)`；
- 指定 `--slot-budget`（所有层窗口之和）：按各层 `n_95pct` 占比分配，再夹到 `[min-slots, max-slots]`。

拿到建议后直接设置（示例）：

```sh
export LLAMA_MOE_LAYER_WINDOWS="0:200,1:96,2:96,..."
```

约束（引擎 `llama-model.cpp` 会强制）：每层 `c_il ≥ n_expert_used`，且 `c_il + K ≤ 1024`（`K` = `LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH`）。

---

## 5. 复现命令（端到端）

```sh
# 1) 构建采集器
cmake --build build --target llama-moe-profile

# 2) 采集（A100「日常回答」集；训练/留出分开）
LLAMA_MOE_PROFILE_CTX=2048 LLAMA_MOE_PROFILE_NGEN=128 \
  CUDA_VISIBLE_DEVICES=0 build/bin/llama-moe-profile model.gguf train.jsonl  train-routes.jsonl
CUDA_VISIBLE_DEVICES=0 build/bin/llama-moe-profile model.gguf heldout.jsonl heldout-routes.jsonl

# 3) 汇总出次数、冷热、窗口建议
python3 tools/moe/summarize_hot_experts.py train-routes.jsonl heldout-routes.jsonl out --slot-budget 3840

# 4) 用建议设置窗口（手机上）
export LLAMA_MOE_LAYER_WINDOWS="$(sed -n 's/^LLAMA_MOE_LAYER_WINDOWS="\(.*\)"/\1/p' out/layer-windows-suggested.txt)"
```

---

## 6. 结果

### 6.1 工具自检（合成轨迹，2026-10-16）

用合成轨迹（低层分散、高层集中）跑通，产出 5 个文件，`layer-windows-suggested.txt` 正确区分了
热层（`n_95pct`≈190）与冷层（`n_95pct`≈15）。**仅为格式自检，非真实数据结论。**

### 6.2 真机预热实验（MiniCPM-V 4.7，2026-10-16）

数据集 `BelleGroup/train_0.5M_CN`（revision `29c35e9b…`），160 样本（128 排名 / 32 留出）。
留出集 decode 频率覆盖率 top8/16/24/32/48/64 = **21.78 / 34.87 / 45.33 / 53.34 / 65.38 / 73.96 %**。

手机：mixed-Q3 tiled 专家，Q8 DSpark length1，bank128，batch/ubatch16，cache16 + prefetch4，
IO8，ctx2048，一个 context checkpoint，greedy seed42。

| Prompt/policy | 窗口命中 | 逻辑请求 GB | Decode t/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| belle-1687/off | 68.20% | 33.150 | 11.420 | 3849 |
| belle-1687/static | 69.24% | 34.804 | 11.345 | 3920 |
| belle-1687/prompt | 69.48% | 32.708 | 11.840 | 4303 |
| belle-839/off | 72.93% | 19.461 | 10.440 | 1383 |
| belle-839/static | 74.15% | 21.112 | 10.700 | 1355 |
| belle-839/prompt | 74.24% | 19.008 | 10.755 | 1581 |

结论：静态预热**略提升命中但增加 I/O**，decode 有正有负；prompt warm（`warm_from_prefill`）
**减少 I/O 但 TTFT 变差**；**默认关闭**。

### 6.3 A100「日常回答」集 profile（待补）

命令见 §5，产出 `expert-counts.tsv` / `layer-skew.json` / `layer-windows-suggested.txt`；
填回本表后据此设置 `LLAMA_MOE_LAYER_WINDOWS` 并在手机 A/B。

---

## 7. 注意

- **冷热是按层**的：同一专家在不同层冷热不同，不要做全局排名。
- 覆盖率是**离线频率统计**，不是在线 LRU 命中率或加速预测；必须在手机上 A/B 验证。
- 静态预热对比要**等内存容量**、并计入预加载时间与字节；与 `LLAMA_MOE_PREFILL_WARM_ROWS`
  分开测（后者会用每个 prompt 的真实路由覆盖静态表）。
- 记录首请求与重复请求、总 I/O（含预取）、内存（含 unique shared/DMA）、输出行为；
  **不要静默改 top-k 路由**。
- 数据集/权重/原始路由轨迹是实验产物，不入仓库；保留来源、revision 与许可信息。
