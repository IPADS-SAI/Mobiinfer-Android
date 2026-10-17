# 端侧 MoE 推理论文整理：BigMoMo / SpecPrefetch / NPUMoE

本文件整理三篇与「端侧 / NPU 上跑稀疏 MoE 大模型」直接相关的论文，重点提炼它们的**专家搬运、预取、计算与搬运重叠**机制，以及对本项目 LES + DSpark 的借鉴点。

| 论文 | arXiv | 定位一句话 |
|---|---|---|
| **BigMoMo** | [2609.14643](https://arxiv.org/abs/2609.14643) | 手机 NPU 上，用**投机解码的多 token 验证窗口**把专家搬运从单 token 执行中解耦 |
| **SpecPrefetch** | [2607.24787](https://arxiv.org/abs/2607.24787) | **本文件核心**：预测下一层专家只做异步预取，原生 router 决定执行（无损），并用重叠窗口定预取预算 `M_max` |
| **NPUMoE** | [2604.18788](https://arxiv.org/abs/2604.18788) | Apple ANE 上**权重全驻留**，专门解决 MoE 动态形状 / 静态图 / 调度开销问题（prefill 加速） |

> 三条路线：BigMoMo = 卸载 + 投机窗口；SpecPrefetch = 卸载 + 预测式预取；NPUMoE = 不卸载 + NPU 静态图适配。

---

## 1. BigMoMo（PKU，2026）

**平台**：手机 NPU，基于 **llama.cpp + Hexagon 后端**（HVX/HMX/VTCM），约 5000 行；设备 OnePlus 15 / 13 / Ace 5；模型 Qwen3-30B-A3B、LFM2-24B-A2B、Gemma-4-26B-A4B、ERNIE-4.5-21B-A3B（Q4_0），扩到 Qwen3-Next-80B-A3B。

### 核心思想

从「模型层参数解耦」转向「**系统层数据搬运解耦**」：用投机解码的**多 token 验证窗口**把搬运粒度放大——一次搬运服务多 token（权重复用）、把碎片读聚成连续读、并让搬运与计算重叠。

### 三个端侧痛点

1. **搬运延迟大**：DRAM 小 → 更多专家留在 flash；单 token 计算摊不平一次搬运。
2. **flash 随机读开销高**：路由导致稀疏不连续读，UFS（~4 GB/s）带宽用不满。
3. **NPU 利用率低**：多级路径 `T_e = T_flash + T_map + T_vtcm + T_hmx`，专家**就绪时间不同**，同步等待产生流水线气泡。

### 三个机制

1. **Reuse-Oriented Expert Movement Admission（复用导向的搬运准入）**
   - Acceptance-Aware Node Pruning：节点价值 `V_d = p̂_{h(d)} · ∏_{u∈path(d)} q_u`（深度历史接受率 × 路径草稿概率），按价值降序保留累计价值 ≥ ρ 的最小子集（论文 `ρ=0.9`）。
   - Hierarchy-Aware Expert Pruning：路由影响 `I_e = Σ_{d∈D_e} V_d · min_j |s_{d,e} − s_{d,j}|`；剪枝优先级 `P_e = ΔI_e / C_e`，其中 `C_e` 是**该专家跨内存层级的搬运代价（离线 profile）**。在路由影响预算 τ 内迭代剪枝，把被剪专家的路由权**重分配给路由分最接近的保留专家**。
   - 实测：投机树中 82.1% 验证计算、81.8% 专家搬运来自不贡献接受 token 的节点。

2. **Runtime-Adaptive On-Flash Expert Organization（运行时自适应 flash 布局）**
   - 用**超图**建模最近 K 轮的共加载（顶点=专家，超边=一轮验证共加载集合）。
   - 为每条超边找 **Anchor Region**（含最多成员专家的连续 flash 区），增量做**专家对交换**把共加载集聚到连续区。
   - 代价用离线 profile 的按请求大小闪存读时间 `T_S^Flash` 估算；收益在最近 K 轮历史窗口累计。
   - 工程细节：**只在两个专家都已加载进 DRAM 后才交换**；**有延迟敏感的读到来时暂停后台重排**。

3. **Cross-Hierarchy Movement Scheduling（跨层级搬运调度）**
   - VTCM 双缓冲（两块权重缓冲 + 128 KiB 激活区）+ 计算分组，**不等所有专家就绪**。
   - 未覆盖搬运时间 `R_t = max(0, τ_t^next − W_t)`；有组满足 `T_b^HMX ≥ R_t` 时选 **HMX 时间最短**的，否则选 **HMX 时间最长**的。
   - 动态重排 flash 读顺序，优先放「剩余加载+映射能在当前窗口内被隐藏」的请求。

### 关键结果

- 平均 **4.83×** vs OnDemand 自回归卸载，**3.00×** vs 投机版 OnDemand，**1.76×**（跨设备最高 1.82×）vs 最好的投机 MoE baseline。
- 按模型 vs 最优 baseline：Qwen3-30B 1.73×、LFM2-24B 1.47×、Gemma-4-26B 1.80×、ERNIE-4.5-21B 2.03×。
- TPOT 拆解（Qwen3-30B-A3B / 8 Elite Gen 5，ms/token）：草稿 12.03（不变）；专家计算 38.26→31.51；**暴露的搬运 stall 142.41→13.62**。
- 运行时 flash 布局：每输出 token 读次数 **−15.06%**，有效带宽 **+1.23×**。
- HTP 利用率 **18.96–45.03% → 83.78–98.30%**。
- 绝对吞吐（Qwen3-30B-A3B Q4_0）：OnePlus 13（18GB 可用）**13.5 tok/s**；OnePlus 15（11.5GB 可用）**3.3 tok/s**；Ace 5（6GB 可用）**1.7 tok/s** → **可用 DRAM 决定天花板**。

---

## 2. SpecPrefetch（ShanghaiTech 等，2026）

**平台**：模型级验证 + **Snapdragon 8 Elite** 真机（Adreno 825）。模型 Qwen3-VL-30B-A3B、DeepSeek-VL2-Tiny（**VL MoE**）。

### 核心思想

把「**传输预测**」和「**执行路由**」彻底分开：
- 预测只用于**异步搬运**；
- 最终执行的专家仍由**冻结的原生 router** 决定。

因此预测误差只影响搬运效率，**不影响输出质量**（无损）。

### 三个方法

1. **Expert Prediction Adapter（低秩 adapter 预测下一层）**
   - 每个**相邻 MoE 层对**挂一个低秩 adapter：`ẑ^{l+1} = B^l A^l x^l`，`A ∈ R^{r×D}`、`B ∈ R^{n×r}`、`r ≪ D`。
   - 输入是**第 l 层的 router 输入**，输出是**第 l+1 层的专家 logits**；softmax 得 token 级分布 `π̂`。
   - batch 级聚合为专家优先级 `s_e^{l+1} = Σ_{(b,s)∈Ω} π̂_{b,s,e}`（预计服务更多 token 的专家优先）。
   - 预测候选集 `C_M = Top_M(s)`。
   - 参数量 `(L−1)·r·(D+n)`：Qwen3-VL 6.48M、DeepSeek-VL2-Tiny 0.82M。

2. **Window-Aware Prefetch Budgeting（窗口感知预取预算）—— 见下节 `M_max`**

3. **Training Objective**
   - 用冻结的原生 router 分布作监督：`KL( stopgrad(softmax(R^{l+1}(x^{l+1}))) ‖ π̂ )`。
   - 只训 adapter，主干冻结；一层 AdamW、lr 1e-4、cosine、warmup 0.03、grad clip 0.5、bf16。
   - 蒸馏**完整路由分布**（而非只 Top-K），以保留优先级排序。

### 关键结果

- ExpertRecall：30 个设置中 **27 个最优/并列**；Qwen3-VL 全部最优。
- **Snapdragon 8 Elite 真机**（DeepSeek-VL2-Tiny 4bit，11 层 MoE、64 专家、top-6、1.94MB/专家，物理 NVMe）：
  - OFF **3.753 tok/s**；PRED0（只跑预测+控制、不搬运）**3.730 tok/s（−0.53%，预测开销≈0）**；SpecPrefetch(`M_max=8`) **4.055 tok/s（+8.12%）**。
  - 同预算 vs Naive 预取（用上一 token 专家）：ReadyRecall **23.4% → 91.7%**；cold NVMe +6.97%、慢存储 +15.43%。
  - 存储越慢收益越大：慢 UFS **≈+20%**、最慢 SD 卡 +17.02%；与计算优化互补。
- 参数量对比：SpecPrefetch 6.48M/0.82M，仅为 ProMoE 的 3.1%（Qwen3-VL）。

---

## 3. NPUMoE（UVA，2026，Apple ANE）

**平台**：Apple M2 Max / M2 Ultra。**假设权重全驻留统一内存（不卸载）**，把 **prefill** 搬到 ANE（长上下文 prefill 占 ~80% 端到端时间、~70% CPU 周期）。

### 背景与挑战

- MoE 推理是动态的（路由 top-k、scatter/gather、每专家 token 数不定），与 NPU 的**静态形状 / 预编译图**冲突。
- 不规则算子（top-k、动态索引、mask、reduction）NPU 不友好 → 回退 CPU/GPU。
- 并发有限 + 频繁派发：小图串行、CPU-NPU 同步在最坏情况可占 **>60%** 运行时。

### 三个机制

1. **Static Tiers for Expert Capacity（专家容量静态分档）**
   - 离线标定每层路由不均衡比 `r_ℓ = max_e n_{ℓ,e} / mean_e n_{ℓ,e}`，基础容量 `C_base = B/E`，最忙专家估计 `n_{ℓ,max} ≈ r_ℓ · B/E`。
   - 把容量分为少数档位 `T = {C⁽¹⁾, C⁽²⁾, C⁽³⁾}`，每个专家映射到能容纳其期望负载的最小档；高频专家给更大容量（如 top 四分位给 4× 基础容量）。
   - 溢出 token 按**激活 L2 显著性**剪掉，精度损失 <1.1%。
   - 原因：Core ML 在编译期静态决定图与设备放置，无法把溢出 token 动态塞给 CPU/GPU，所以只能静态估容。

2. **Grouped Expert Execution（分组专家执行）**
   - 把多个**同容量档**专家打包进**一个静态 FFN 图**，一次 launch 算多个专家，摊薄派发/调度开销、避免 subgraph 爆炸。
   - 流程：按专家分组 → 分配 `X∈R^{(G·C)×H}` → 连续切片 `X_g` → 打包 token 并 zero-pad → `Y_g=f_{e_g}(X_g)` → 加权 scatter 回原位。
   - 存在**可行性边界**：图过大（如 M2 Ultra >1.2GB）会回退 CPU；把 16 个专家塞进一个大图并不最优。

3. **Load-Aware Compute Graph Residency（负载感知的图驻留）**
   - 离线标定专家热度；**热专家组常驻 NPU**（摊薄 launch/同步），**冷专家 / 小结组丢给 CPU**（短突发不值得 CPU-NPU 同步）。
   - 观测：主导专家**不强烈依赖数据集**（一个 benchmark 上热的专家在另一个上也热）。

### 关键结果

- M2 Max/Ultra 上（Phi-3.5-MoE、Phi-tiny-MoE、Qwen3-30B-A3B，4 个长上下文负载）：prefill 延迟 **1.32–5.55×**、能耗 **1.81–7.37×**、CPU 周期 **1.78–5.54×**，精度损失 <1.1%。

---

## 4. 横向对比

| 维度 | BigMoMo | SpecPrefetch | NPUMoE |
|---|---|---|---|
| 平台 | 手机 NPU（Hexagon） | Snapdragon 8 Elite | Apple ANE |
| 是否卸载 | ✅ Flash→DRAM→NPU buffer→VTCM | ✅ host/flash→加速器 | ❌ 权重全驻留 |
| 主要阶段 | decode（投机） | decode（标准卸载推理） | **prefill** |
| 预取/预测 | 无传统预取；**搬运准入剪枝** | **低秩 adapter 预测下一层** | 离线热度标定（不预测下一层） |
| 是否需训练 | ❌ 免训练 | ⚠️ 训 adapter（0.8–6.5M 参数） | ⚠️ 离线标定（不重训权重） |
| 预测用途 | 搬运准入 + 调度 | **仅异步搬运**，router 权威 | 容量分档 + 冷热放置 |
| 搬运↔计算重叠 | 跨层级就绪时间调度 + VTCM 双缓冲 | **窗口感知预算** + 异步搬运 | CPU/NPU 图放置 + 分组 |
| 布局优化 | **运行时 flash 重排**（超图 + Anchor Region） | 无 | 无 |
| 静态形状处理 | NPU buffer/VTCM 分级 | — | **容量分档 + 分组图** |
| 是否用投机 | ✅ 核心 | ❌ | ❌ |
| 无损 | 近似（剪枝影响 <0.26%） | ✅ 无损（router 不变） | 近似（溢出剪枝 <1.1%） |
| 关键加速 | 4.83× vs OnDemand；1.76–1.82× vs 最优 baseline | +8% cold / +15–20% 慢存储 | prefill 1.32–5.55× |

---

## 5. 关键概念：重叠窗口（overlap window）与 `M_max`

这三个数都来自 SpecPrefetch §3.3，解决的是「**预测很准，但不代表预取有用**」的问题。

### 重叠窗口 `W_l`

在层 `l`，
- `t_l^{pred}`：第 l 层**预测结果可用的时刻**；
- `t_{l+1}^{exec}`：第 l+1 层**被路由选中的专家开始计算的时刻**。

于是层 l 到 l+1 之间、**可以用来搬运下一层专家的时间窗**是：

```
W_l = t_{l+1}^{exec} − t_l^{pred}
```

只有在这段窗口里完成的搬运，才是「被计算盖住」的有效搬运；窗口外的搬运就变成暴露的 stall。

### `M_max` 是什么

**`M_max` = 每个目标 MoE 层最多预取多少个专家（最大预取预算，单位是专家个数）。**

它的确定方式：
1. **离线 profile** 目标平台的**计算窗口**和**单个专家的搬运延迟**；
2. 由此定出 `M_max`——大致就是「在一个重叠窗口内能搬完的专家数量」；
3. 运行时对每个目标层，把「尚未就绪、也不在搬运中」的专家按预测优先级排序，取 **`Top_{M_max}`** 发起异步搬运：

```
P^{l+1}(t) = Top_{M_max}( { s_e^{l+1} | e ∉ A^{l+1}(t) ∪ U^{l+1}(t) } )
```

其中 `A` 是已就绪、`U` 是在途的专家集合。实际发起的搬运数 `m_l = |P| ≤ M_max` 恒成立。

**为什么要有这个上限**：
- 预测再多，**窗口装不下**就只能变成冗余 I/O + 缓存压力；
- 论文动机里实测：该窗口大约只够 ~2 次搬运，而 top-4 / top-8 需求分别在 37% / 89% 的事件里超过窗口；
- 真机上 **`M_max=8` 是吞吐拐点**：继续加大预算，ReadyRecall 还涨（82.1%→97.5%）但**吞吐反而下降**。

一句话：**`M_max` 是「按重叠窗口给预取设的硬上限」，把「覆盖度」和「冗余搬运」平衡在一个由硬件 profile 决定的预算上**——它对应到本项目里，就是 LES 的 **`lookahead_prefetch_depth` / `max_prefetch_depth=16`** 以及 slot bank 归一化（`normalize_prefetch_depth`）想表达的同一件事。

---

## 6. 对本项目 LES + DSpark 的借鉴点

1. **显式的重叠窗口预算**：SpecPrefetch 用 profile 出的 `W_l` 定 `M_max`，而不是只按缓存大小定深度。可在 `src/llama-moe-slot.*` 的诊断计数（`process/readids/install/writeids/miss/bytes`）基础上估一个 `W`，据此定每层预取深度。
2. **预测与执行解耦**已具备（预测仅预取）；若现有免训练预测命中率不足，可评估加一个低秩 adapter（代价：需训练，与「免训练」卖点冲突，需权衡）。
3. **收益在慢存储上最明显**：回填性能数据时应区分快/慢存储，避免快 flash 掩盖收益。
4. **投机窗口放大搬运粒度**（BigMoMo）：DSpark 的多 token 验证窗口可用来做权重复用 / 连续读 / 跨层级重叠。
5. **运行时 flash 布局重排**（BigMoMo）：当前是静态打包（`hexagon-tiled-v1`），动态共加载模式未被利用。
6. **NPU 图/调度开销**（NPUMoE）：分组执行、热/冷专家放置、CPU-NPU（或 CPU-DSP）同步开销在 Hexagon 侧同样成立，值得验证「小/冷专家是否值得上 NPU」。
7. **常驻专家不是越多越好**（BigMoMo）：24 为拐点，过多与 attention 争 NPU mapped buffer。

---

> 这些机制的**可落地清单**（改哪些开关、改哪段代码、预期收益、优先级）见 [on-device-moe-offloading-adoption.md](on-device-moe-offloading-adoption.md)。

## 7. 参考链接

- BigMoMo — arXiv:2609.14643 — https://arxiv.org/abs/2609.14643
- SpecPrefetch — arXiv:2607.24787 — https://arxiv.org/abs/2607.24787 （code: https://github.com/wei390/SpecPrefetch）
- NPUMoE — arXiv:2604.18788 — https://arxiv.org/abs/2604.18788
- 相关：FATE (2502.12224)、MoE-Infinity (2401.14361)、Klotski (2502.06888)、Pre-gated MoE (2308.12066)、APEX (2608.11688)、EdgeMoE (2308.14352)
