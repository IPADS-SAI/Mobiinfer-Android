# BigMoMo 阅读笔记 + 端侧 MoE 常驻/调度对照

论文：*BigMoMo: Efficient Inference of Large-Scale MoE with Speculative Decoding on
Mobile Devices*（arXiv:2609.14643v2，北大；基于 llama.cpp + Hexagon 后端，约 5k 行改动）。

相关实现：本仓库的 MoE-SSD（slot bank / 外置专家 / 前瞻预取）与 DFlash/DSpark。

---

## 1. 论文要解决的问题

移动端 MoE 的瓶颈是**专家搬移**，不是算力。一个专家权重要经过：

```
T_e = T_Flash + T_Map + T_Vtcm + T_HMX
        UFS        DRAM→NPU映射   DDR→VTCM     HMX 计算
```

- 手机 DRAM 带宽 ~78 GB/s，UFS ~4 GB/s（小随机读更差）。
- NPU 只能访问**映射进其地址空间**的内存（"NPU mapped buffer"，容量有限）；VTCM 是
  **片上 8 MiB scratchpad**，是 DDR→VTCM→HMX 的最后一跳。
- 单 token 只激活少数专家，一次搬移服务不了几个 token → 稀疏计算被权重传输拖住。

核心思想：**用投机解码的多 token 验证窗口，把「搬移」和「单 token 执行」解耦**——
权重复用、闪存连续读、搬移与计算重叠。

---

## 2. 三块技术 + 常驻专家

### 2.1 投机驱动的搬移准入（Reuse-Oriented Expert Movement Admission）
- **接受率感知的节点剪枝**：按「深度接受概率 × draft 路径置信度」给 draft 树节点打分，
  剪掉低价值分支（论文测：这些节点占 82.1% 验证计算、81.8% 搬移）。
- **层级感知的专家剪枝**：按「路由影响 ΔI / 搬移代价 C」排序剪专家（受总影响预算约束），
  被剪路由权重转给最接近的专家。

### 2.2 运行时闪存布局重排（Runtime-Adaptive On-Flash Organization）
- 用**超图**记录最近 K 轮「同一验证轮一起被加载」的专家集合。
- 为每个 co-loading 集找 **anchor region**（重叠最多的连续闪存区），做**增量专家交换**让它们
  连续，把碎片读变成大块顺序读。
- 交换在专家已进 DRAM 后后台做，遇到关键读请求就暂停。

### 2.3 跨层级搬移-计算调度（Cross-Hierarchy Movement Scheduling）
- 跟踪专家处于 Flash→DRAM→mapped→VTCM 哪一阶段，按**剩余搬移时间**排程。
- **VTCM 双缓冲**（两块权重缓冲 + 128 KiB 激活区），大专家按 N 以 32 为组切片。
- 把已就绪专家组成**计算组**，选能盖住剩余搬移时间的组（Eq.5/6）；常驻专家当**兜底组**
  防 HMX 气泡。
- 动态重排待读请求，优先读「能在当前计算窗口内被藏住」的。

### 2.4 常驻热专家（Expert Residency）——重要澄清
论文原文：
> "keep the most frequently activated experts in the **NPU mapped buffer** for direct
> **VTCM weight movement**. We use 24 resident experts on Snapdragon 8 Elite and
> Snapdragon 8 Elite Gen 5, and 8 on Snapdragon 8 Gen 3."

敏感性分析（§5.5）：
> "varying the number of resident experts in the **fixed-capacity NPU mapped buffer**
> from 8 to 40 ... TPOT decreases from 8 to 24, but rises with further increases ...
> greater **mapped-buffer contention with attention weights executed by HVX** ...
> The same 24-expert residency is used on Snapdragon 8 Elite, which provides the same
> NPU-buffer capacity. On Snapdragon 8 Gen 3, the smaller buffer capacity limits the
> number of resident experts to 8."

结论：
- **常驻位置是 NPU mapped buffer（DDR），不是 VTCM**；VTCM 只有 8 MiB，装不下。
  「direct VTCM movement」= 省掉 Flash 与 DRAM→映射两跳，只剩 DDR→VTCM→HMX。
- **24 / 8 是受 mapped-buffer 容量约束的全局预算**（8 Elite 与 8 Elite Gen5 都 24；
  8 Gen 3 只够 8），不是「每层 24」。
- 超过 24 反而变慢：常驻专家会**和 attention 权重抢同一块 mapped buffer**。

---

## 3. 对照本仓库现状

| 维度 | 本仓库 | BigMoMo |
|---|---|---|
| 常驻位置 | slot bank = Hexagon `alloc_only` 缓冲（rpcmem DMA，DDR） | NPU mapped buffer（DDR）→ **位置相同** |
| 常驻粒度 | **每层 top-K**（`LLAMA_MOE_PREWARM` + 持久 pin），占每层窗口缓存区 | **全局预算**（24/8），共享池 |
| 常驻角色 | 仅当缓存（减 miss） | 缓存 + **兜底计算组**填 HMX 气泡 |
| 读合并 | 有 `accumulate_runs`，但实测 `runs=150 avg_run=1.0 max_run=1`（专家在 tiled 文件里不相邻 → 合并无效） | 运行时闪存重排让读连续 |
| 搬移/计算重叠 | A2 split（驻留算 / 缺失读重叠），简化版 | 跨层级调度 + VTCM 双缓冲 |
| 投机 | DFlash/DSpark 有（MiniCPM/Qwen3.5），**qwen3moe 无草稿** | 每个目标模型配草稿（Qwen3-30B-A3B 用 Qwen3-0.6B） |

### 3.1 本仓库的实测（qwen3moe，vivo v81，同 prompt，`max_tokens=128`）

| 配置 | Σ槽位(48层) | 估算 RAM | 命中 | tg (3 次) |
|---|---:|---:|---:|---|
| 均匀 c=32 | 1536 | ~4.5G | 91% | 5.77 / 5.94 / 6.37（中位 **5.94**） |
| per-layer Σ≈720 | ~720 | ~3.4G | 61% | 5.36 / 5.32 / 5.28（中位 **5.32**） |
| per-layer + pin top-8/层（384 专家） | ~720 | +1.0G | — | **4.89** |

- per-layer 在「半 RAM」下略慢于均匀；pin top-8 反而更慢：per-layer 高层窗口只有 8 格，
  pin 8 就**把整层窗口钉满**，LRU 没余量。与 `tools/moe/HOT_EXPERTS.md §6.2`（MiniCPM 静态
  预热略负）一致。
- K=2 的 pin 测试因设备断开未完成。

---

## 4. 可借鉴的优化（按性价比）

1. **给 qwen3moe 配草稿（上限最高）**
   BigMoMo 对 Qwen3-30B-A3B 用的就是 **Qwen3-0.6B**。有了多 token 验证窗口，才有
   「一次搬移服务多 token / 连续读 / 搬移与计算重叠」。仓库 `/temp/models/Qwen3-0.6B-GGUF` 已存在。
   需接一个跨架构 draft（simple/draft spec type）并 A/B。

2. **改为全局 top-N 常驻**
   把「每层 K」改成「全局热度前 N 个 (layer, expert) 对」的共享常驻池（N 量级 24~32），
   不挤占每层 LRU；注意别和 attention 抢 mapped buffer。比每层 pin 更省 RAM。

3. **闪存布局重排（本仓库暂缓）**
   `avg_run=1.0` 说明读完全碎片化；静态按共激活重排 tiled 文件即可让 `accumulate_runs`
   命中（运行时零改动）。

4. **跨层级调度（深水区）**
   VTCM 双缓冲 + 计算分组 + 按剩余搬移时间选组 + 常驻集当兜底组。工作量大，后置。

5. **投机驱动剪枝（依赖 1）**
   接受率感知节点剪枝 + 层级感知专家剪枝。研究性强，先跳过。

---

## 5. 备注：投机类型可组合

`common/speculative.cpp` 里 `common_params_speculative::types` 是 `std::vector`，
多个类型会各自创建实现并按优先级放入 `impls`（`common_speculative_init`）。
因此 `dspark` + `ngram-*` 理论上可同时启用；是否会合并成更宽的验证树取决于
`generate_draft` 的用法（待进一步确认）。
