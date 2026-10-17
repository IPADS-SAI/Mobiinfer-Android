# 端侧 MoE 优化：可落地点清单（面向 LES + DSpark）

本文件把 [on-device-moe-offloading-papers.md](on-device-moe-offloading-papers.md) 里三篇论文（BigMoMo / SpecPrefetch / NPUMoE）的机制，映射到本项目**具体能做什么**，按落地难度和收益分三档：

- **一档**：改配置 / 开关即可试，不动代码；
- **二档**：需要改代码，收益最大；
- **三档**：概念参考，当前形态下不宜直接照搬。

> 对照的代码位置：`src/llama-moe-slot.{h,cpp}`（专家槽位运行时）、`tools/moe/*`（专家提取/剥离/打包）、`common/speculative.cpp`（DSpark）、`common/arg.cpp`（命令行开关）、`ggml/src/ggml-hexagon/`（Hexagon 后端）。

---

## 一档：现在就能用（改配置 / 开关）

| # | 借鉴点 | 出处 | 怎么落到本项目 | 预期 |
|---|---|---|---|---|
| 1 | **用重叠窗口给预取设上限** | SpecPrefetch `M_max` | 已有 `lookahead_prefetch_depth`（0–16，`llama-moe-slot.h:11`）与 `set_lookahead_prefetch_depth`。当前深度由 `normalize_prefetch_depth(bank, requested)` 按 **slot bank 大小**（`bank ≥ 2*depth+8`）决定；改为**按测得的重叠窗口 / 单专家搬运时间**决定，即 `M_max` 的等价物 | 避免"预取过多→冗余 I/O"导致的吞吐反降 |
| 2 | **多预取≠更快，找拐点** | SpecPrefetch 真机 `M_max=8` 拐点 | 扫 `lookahead_prefetch_depth` × `--moe-slot-bank`，定位本平台吞吐拐点 | 拿到本平台最优预算 |
| 3 | **预测只做预取、router 权威（无损）** | SpecPrefetch / 已有 LES | 本项目已是此设计，无需改；继续作为核心卖点 | 保持无损 |
| 4 | **常驻/预取不是越多越好** | BigMoMo（24 专家拐点：过多与 attention 争 NPU mapped buffer） | 扫 `--moe-slot-bank` 与 `decode_window`，验证是否同样存在拐点 | 找到 bank 的最优点 |
| 5 | **"搬运准入"= 减少无用专家请求** | BigMoMo node/expert pruning | 已有 `--moe-k-decode` / `--moe-k-decode-cycle`（`common/arg.cpp`），本身就是简化版准入；先用它扫质量-速度权衡 | 质量-速度曲线 |
| 6 | **收益在慢存储上最明显** | SpecPrefetch（慢 UFS +20%，快 NVMe 收益有限） | 回填性能数据时**区分快/慢存储**，避免快 flash 掩盖收益 | 更可信的对比口径 |

**一档结论**：几乎零成本。**先做 1、2**——把预取深度的决策依据从"bank 大小"换成"重叠窗口估计"，并扫出拐点。

---

## 二档：改代码，收益最大（按优先级）

### 2.1 运行时 flash 布局重排（Anchor Region + 共加载超图）— **最值得做**

- **现状**：专家是**静态**打包（`tools/moe/pack_hexagon.py`，格式 `hexagon-tiled-v1`），拿不到动态共加载收益。
- **改法**：
  1. 在 `llama_moe_slot_runtime::process()` 中已经拿到每层实际选中的专家集合，顺手记录**最近 K 轮的共加载集合**（超图：顶点=专家，超边=一轮共加载集）；
  2. 周期性为常共加载的集合找 **Anchor Region**（连续 flash 区），把成员专家增量聚拢（专家对交换）；
  3. 交换只在两个专家都已进 DRAM 后做；**有延迟敏感读到来时暂停后台重排**。
- **预期**（BigMoMo 实测）：每输出 token **读次数 −15%**、有效带宽 **+1.23×**；几乎不影响计算正确性。
- **风险**：低（只动布局，不改计算语义）。

### 2.2 投机窗口放大搬运粒度（接 DSpark）

- **现状**：DSpark 已在（§`common/speculative.cpp`，草稿 `n_max=3`）；LES lookahead 目前按"单 token 下一层"预取。
- **改法**：
  1. 把预取目标从"下一个 token 的下一层"扩成"**整棵草稿树的专家并集**"，一次搬运服务多个候选 token（权重复用 + 连续读）；
  2. 借鉴 BigMoMo 的**接受率加权**：低价值草稿分支的专家**不预取**（甚至不验证）。
- **预期**（BigMoMo）：把暴露的专家搬运 stall 从 142 ms/token 量级压到 13 ms/token 量级（其特定配置）。

### 2.3 跨层级就绪时间调度（VTCM 双缓冲 + 就绪分组）

- **现状**：已有 `process_deferred` / `matmul_split`（A2 phase B 延后读），是同一问题的雏形。
- **改法**：把"等所有专家就绪"改成"按**剩余就绪时间** `R_t = max(0, τ_t^next − W_t)` 分组执行"：
  - 有组满足 `T_b^HMX ≥ R_t` → 选 HMX 时间最短的；
  - 否则选 HMX 时间最长的（尽量盖住）。
- **配套**：对齐 BigMoMo 的多级搬运时间模型 `T_e = T_flash + T_map + T_vtcm + T_hmx`。
- **预期**（BigMoMo）：HTP 利用率从 ~19–45% 提到 ~84–98%（特定模型/设备）。

### 2.4 全静态激活量化（W4A8，正交方向）

- **论文**：Quant.npu（[on-device-npu-quantization.md](on-device-npu-quantization.md)），针对 Qualcomm Hexagon。
- **现状**：本项目权重侧是 Q3_K/Q4_K，**激活**多半仍较高精度；HTP 端整数 GEMM 未被激活量化吃满。
- **改法**：走 **per-tensor 激活 + per-channel 权重**的**全静态**方案；只保留**离线融合旋转（R1/R2）**、去掉在线旋转；用**相对量化误差**做敏感度指标，只把少数高敏感 down_proj 激活提到 16-bit。
- **预期**（论文）：Hexagon 上 W4A8 per-tensor 比 W4A16 per-block 快 ~20%；同等延迟下精度更高（W4A8 PPL 28.78→19.16）。
- **前提**：HTP 侧需有对应的 W4A8/INT8 激活 GEMM 内核。
- **注意**：与 LES/DSpark **正交**，可叠加。

### 2.5 用现有 counters 自适应定每层预取深度

- **现状**：`llama-moe-slot.h:59-68` 的 `counters`（`process/readids/install/writeids/miss/bytes/unique`）已能测出"非 I/O 记账"和真实读字节。
- **改法**：基于每层 `install`（搬运时间）与计算窗口之比，**自适应设定该层的 `lookahead_prefetch_depth`**。这就是一档第 1 条的落地实现。
- **风险**：低。

---

## 三档：概念参考，当前不宜直接照搬

| 机制 | 出处 | 为什么不直接可用 | 可借鉴处 |
|---|---|---|---|
| 容量静态分档 + 溢出剪枝 | NPUMoE | 前提是**权重全驻留、不卸载**（统一内存）+ Apple Core ML 静态图；本项目 35B 从 flash 流式加载，regime 不同 | 分档思想可用于 prefill 批处理；"溢出按激活显著性剪枝"可参考 |
| 分组专家执行（多专家融进一个大图） | NPUMoE | 面向 Core ML 图粒度的派发开销；Hexagon 侧机制不同 | 对**Hexagon/DSP 子图派发开销**有参考：小/冷专家是否值得上 DSP，可否回退 CPU |
| 冷/热专家放置 | NPUMoE | 同上 | Hexagon 上"冷专家回退 CPU / 小图不上 DSP"值得验证 |
| 低秩 adapter 预测下一层专家 | SpecPrefetch | **需要训练**（0.8–6.5M 参数），与 LES "无需额外训练、无损"卖点冲突 | 若现有免训练预测命中率不足，可作备选；接口上"预测仅用于预取"一致 |
| 投机节点/专家准入剪枝（值函数 `V_d`、路由影响 `I_e`、搬运代价 `C_e`） | BigMoMo | 引入约 0.05–0.26% 的质量损失（近似，非无损） | 可做成**可选开关**，默认无损路径不变 |

---

## 落地优先级建议

1. **立即**：一档 #1/#2 — 预取深度改为"窗口驱动"，扫出本平台拐点。
2. **随后**：二档 2.1 — 运行时 flash 重排（与现状"静态打包"差距最大、收益最确定、风险最低）。
3. **并行**：二档 2.5 — 用 counters 自适应深度（2.1 的输入之一）；2.4 全静态激活量化为正交方向，可单独推进。
4. **再评估**：二档 2.2（接 DSpark 的窗口复用）与 2.3（跨层级调度）。
5. **仅作参考**：三档；如要做，一律做成**可选开关**，默认保持无损。

---

## 与现有代码/开关的映射

| 论文概念 | 本项目对应 |
|---|---|
| `M_max`（每层预取上限） | `lookahead_prefetch_depth` + `normalize_prefetch_depth()`（`llama-moe-slot.h:11,13-16`） |
| 预取深度设置 | `set_lookahead_prefetch_depth()`（`llama-moe-slot.h:37`） |
| 搬运/计算分项测量 | `counters`（`llama-moe-slot.h:59-68`） |
| 延后读 / phase B | `process_deferred()` / `build_deferred_trigger()` / `matmul_split()`（`llama-moe-slot.h:71-77`） |
| 静态预热 | `prewarm_experts()`（`llama-moe-slot.h:84`） |
| 专家槽位预算 | `--moe-slot-bank`（`common/arg.cpp`） |
| 搬运准入（简化版） | `--moe-k-decode` / `--moe-k-decode-cycle` |
| 投机 | DSpark（`common/speculative.cpp`，`n_max=3`） |
| flash 布局 | `tools/moe/pack_hexagon.py`，`hexagon-tiled-v1` |
| 多设备切分 | `--device` / `--split-mode`（Hexagon 后端） |

---

参考论文与详述见 [on-device-moe-offloading-papers.md](on-device-moe-offloading-papers.md)。

### SpecPrefetch 大规模训练中途证据（2026-10-16，未完成）

A100 已启动 500,000 条 BELLE 中文指令的单轮训练，另留 2,000 验证和 2,000 测试；规模按用户要求，数据组成不等于论文原始配方。冻结实际量化 GGUF 教师，训练 39 对相邻 MoE 层的 rank32/rank64 KL 预测器，仅对给定答案位置计算训练损失。所有训练数据排除了此前 pilot 的提示。训练阶段尚未完成，以下不能视为最终结果。

64,000 条训练后的检查点，在 256 条独立验证对话、每层 14,355 个答案 token 上，同预算预取 8 个专家的召回率：原生 81.80%，rank64 72.44%，rank32 65.27%。

另用此前已检查过、但从大训练语料排除的 32 条自回归轨迹做辅助诊断，69,000 条训练后的检查点结果：原生 prefill/decode 召回率 83.08%/83.72%；rank64 为 63.98%/74.48%。该辅助集合不能替代最终未使用的 2,000 条测试集。

当前证据不支持以训练预测器替换原生预取。更少的预测器乘法也不自动意味着端到端提速：错误预取增加磁盘读取、缓存污染和主线程补读。完整训练后先用全量验证集冻结 rank，再进行独立测试及实际缓存/延迟比较。原始代码与结果位于 sai-a100 `/home/ma-user/workspace/csm/moe-task/specprefetch` 和本地 `/home/yydh/moe-speculative/hexagon-results/specprefetch`。

后续 122,000 条训练后的检查点，在完全相同的 256 条验证对话上，预算 8 个专家时 rank64 召回率提高到 76.35%，rank32 为 68.76%；原生仍为 81.80%。这是同样本上的改善，但 rank64 仍落后原生 5.44 个百分点，暂不支持替换。
