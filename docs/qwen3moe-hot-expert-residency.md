# Qwen3-30B-A3B（qwen3moe）端侧热专家驻留：profiling 结果与 BigMoMo 参考

状态：profiling 已完成（A100），设备端 A/B 待做。
工具：`tools/moe/profile_hot_experts.cpp`（`llama-moe-profile`）+ `tools/moe/summarize_hot_experts.py`。

## 1. 背景

端侧大 MoE 的瓶颈是**专家权重从闪存搬到 NPU 的 I/O**。已有机制里，每层是一个
LRU 窗口（`LLAMA_MOE_LAYER_WINDOWS` / `LLAMA_MOE_DECODE_WINDOW`，c_i 个槽）+ 预取暂存
（K）。「热专家常驻」= 让每层最常激活的专家**始终留在窗口内**，减少 miss 重读。

此前只在 MiniCPM-V 4.7（256 专家/top-8）上验证过：静态预热`LLAMA_MOE_PREWARM`命中
略升但额外 I/O 抵消，decode 有正有负，默认关闭（见 `tools/moe/HOT_EXPERTS.md §6.2`）。
原因是 256 专家激活太分散。**Qwen3-30B-A3B 只有 128 专家，预期更集中**，值得重测。

## 2. 方法

- 数据：与 MiniCPM 同源的日常问答集 **BelleGroup/train_0.5M_CN**，
  `~/workspace/csm/moe-task/hot-expert-profile/{train,heldout}.jsonl`（160 条：128 排名 / 32 留出）。
- 采集：`LLAMA_MOE_PROFILE_CTX=2048 LLAMA_MOE_PROFILE_NGEN=128`，
  `llama-moe-profile Qwen3-30B-A3B-Q4_K_M.gguf train.jsonl q3moe-train-routes.jsonl`（GPU，greedy，thinking off）。
- 汇总：把 `summarize_hot_experts.py` 的 `N_LAYERS=40/N_EXPERTS=256` 改成 **48/128** 后运行。
- 产物：`~/workspace/csm/moe-task/hot-expert-profile/q3moe-summary{,-1536,-3072}/`。

## 3. 结果

### 3.1 留出集 decode 覆盖率（每层槽位 vs 命中）

| 每层槽位 | Qwen3-30B-A3B | MiniCPM 4.7（旧） |
|---:|---:|---:|
| 8  | 33.8% | 21.8% |
| 16 | 50.8% | 34.9% |
| 24 | 62.4% | 45.3% |
| 32 | **71.6%** | 53.3% |
| 48 | 84.2% | 65.4% |
| 64 | **91.8%** | 74.0% |

**结论：qwen3moe 激活明显比 MiniCPM 集中**（同样 64 槽/层，91.8% vs 74.0%）。
⇒ 热专家驻留在 qwen3moe 上应比 MiniCPM 有效得多。

### 3.2 按层冷热

- 低层 **L00–L40**：分散，`n_95pct` ≈ 63–88（`entropy_norm` 0.25–0.40）。
- 高层 **L41–L47**：集中，`n_95pct` ≈ 27–54（`entropy_norm` 0.49–0.61，`effective_experts` ~17–25）。
  （L46:27 L47:34 L45:40 L42:40 L44:48 L41:49 L43:50）

⇒ 窗口不应均匀：低层开大、高层可开小。

### 3.3 建议的按层窗口

`LLAMA_MOE_LAYER_WINDOWS`（Σ 槽位 / 估算 RAM，按 ~1.1 MB/专家/层）：

- **预算 3072（≈64/层，RAM ≈ 3.4 GB）**
  `"0:70,1:82,2:65,3:78,4:63,5:71,6:61,7:67,8:69,9:69,10:75,11:78,12:73,13:71,14:67,15:69,16:65,17:66,18:62,19:69,20:69,21:65,22:75,23:74,24:72,25:71,26:67,27:67,28:65,29:65,30:62,31:68,32:67,33:62,34:73,35:70,36:71,37:64,38:62,39:64,40:59,41:46,42:37,43:47,44:45,45:37,46:25,47:32"`
- **预算 1536（≈32/层，RAM ≈ 1.7 GB）**
  `"0:35,1:41,2:33,3:39,4:31,5:36,6:30,7:34,8:35,9:35,10:37,11:39,12:36,13:36,14:34,15:35,16:33,17:33,18:31,19:35,20:35,21:32,22:37,23:37,24:36,25:36,26:34,27:34,28:33,29:33,30:31,31:34,32:34,33:31,34:36,35:35,36:36,37:32,38:31,39:32,40:29,41:23,42:19,43:23,44:22,45:19,46:13,47:16"`

约束（`llama-model.cpp`）：每层 `c_i ≥ top-k(8)`，且 `c_i + K ≤ 1024`（K=`LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH`）。

> 覆盖率是**离线频率统计**，不等于在线 LRU 命中率或加速；必须在真机 A/B。

## 4. 参考：BigMoMo 的「驻留 VTCM」

论文：*BigMoMo: Efficient Inference of Large-Scale MoE with Speculative Decoding on Mobile Devices*
(arXiv:2609.14643，北大；基于 llama.cpp + Hexagon backend，~5k 行)。

要点（与本项目高度相关）：

- **移动 NPU 的多级搬移**：专家权重路径
  `T_e = T_Flash + T_Map + T_Vtcm + T_HMX`。
  即 UFS→DRAM→（映射进）**NPU mapped buffer**→**VTCM**（8 MiB on-chip scratchpad，双缓冲）→HMX 计算。
  NPU 只能访问映射进其地址空间的区域，容量有限（"NPU buffer"）。
- **驻留集**：按全局激活频率，把最热的专家**常驻在 NPU mapped buffer** 里，
  使其搬移只剩 map→VTCM（省掉 Flash 读与 DRAM→map），并作为**兜底计算组**填补
  搬移造成的 HMX 气泡。规格：**8 Elite / 8 Elite Gen 5 上驻留 24 个专家，8 Gen 3 上 8 个**。
- 用**投机解码的多 token 验证窗口**把「搬移」和「单 token 执行」解耦：权重复用、闪存连续读、
  搬移与计算重叠；再配合分支/专家剪枝（按接受率 + 路由影响 / 搬移代价）、
  运行时闪存布局重排（co-loading 超图 + anchor region）、跨层级搬移调度。
- 结果：相对按需自回归卸载 4.83×，相对最好的投机 MoE 基线 1.82×，支持到 80B。

## 5. 与本项目的差异 / 可借鉴点

| 维度 | 本项目现状 | BigMoMo |
|---|---|---|
| 常驻位置 | 「slot bank」窗口（每层 c_i 槽，prefill/decode 视图），在 host-可见内存，preadv 填充 | **NPU mapped buffer** 常驻热专家，直接 VTCM 搬移 |
| 常驻选择 | 离线 profile → `LLAMA_MOE_LAYER_WINDOWS`（按层窗口）；`LLAMA_MOE_PREWARM` 静态预热 | 全局激活频率 top-24 驻留 |
| 作用 | 减少 flash miss | 既省搬移，又当**兜底计算组**填 HMX 气泡 |
| 依赖投机 | 已有 DSpark/DFlash（qwen3moe 暂无草稿） | 强依赖投机验证窗口 |

## 6. 下一步

1. 真机 A/B：vivo(v81) 上设 `LLAMA_MOE_LAYER_WINDOWS`（先 3072 档）+ 相应 `--moe-slot-bank`，
   对比默认（均匀 64）。App 侧可用 `runtime.json` 覆盖。
2. 若为正，再考虑**真 pin**：在 `llama-moe-slot.cpp` 里把每层窗口切出 pinned 段（预载后不淘汰）。
3. 对齐 BigMoMo：评估「常驻集既是缓存又是兜底计算组」的可移植性（需 Hexagon 侧把常驻专家放在
   NPU mapped buffer 并参与调度）。
4. 投机：给 qwen3moe 找/训草稿（如 Qwen3-0.6B 做草稿，或 DFlash 路线），
   以吃到多 token 验证窗口带来的搬移复用。
