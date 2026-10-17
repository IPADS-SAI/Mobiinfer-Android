# 端侧 MoE / Hexagon 优化方法清单（菜单）

本文件汇总可用的优化手段，按**存储、调度、缓存、算法、投机、系统**六类整理，标注本项目（手机 + Qualcomm Hexagon + 35B MoE 流式解码）的现状、预期与代价，供挑选组合。

> 关联文档：[on-device-moe-offloading-papers.md](on-device-moe-offloading-papers.md)（三篇论文详解）、[on-device-moe-offloading-adoption.md](on-device-moe-offloading-adoption.md)（可落地点清单）、[on-device-npu-quantization.md](on-device-npu-quantization.md)（Quant.npu 量化）。
>
> ⚠️ 一个总纲：**decode 是延迟受限（搬运暴露，靠重叠/预取），prefill 是带宽受限（I/O 总量，靠少读/连续读）**。选方法前先确定你优化的是哪个阶段。

---

## 一、存储 / I/O 侧（prefill 最吃这里）

| # | 方法 | 说明 | 现状 | 预期 | 代价 |
|---|---|---|---|---|---|
| 1 | **连续读 / 布局重排** | 运行时按共加载把专家聚成连续读（BigMoMo Anchor Region） | 缺（现为静态 tiled 打包） | 读次数 −15%、有效带宽 +1.23× | 中等（改动 + 后台重排写） |
| 2 | **flash 页对齐** | 每次读对齐 4K/16K 页边界，避免随机小读的读放大 | 待查 | 直接提升有效带宽 | 低 |
| 3 | **量化减字节** | 全静态 W4A8 / 逐层混合精度（Quant.npu） | 部分有（Q3_K/Q4_K 权重） | 总量下降 → prefill 下界下降；Hexagon 上 W4A8 per-tensor 比 W4A16 per-block 快 ~20% | 需 HTP INT8 激活内核 |
| 4 | **直接 I/O vs page cache** | `--load-mode dio` 避免污染页缓存 | 有开关 | 视复用情况而定 | 低 |
| 5 | **专家压缩** | 低秩 / 稀疏表示减字节 | 缺 | 减字节 | 可能损质量 |

## 二、计算 / 调度重叠

| # | 方法 | 说明 | 现状 | 预期 | 代价 |
|---|---|---|---|---|---|
| 6 | **层间流水** | 当前层计算盖下一层加载（Klotski 多 batch 流水） | 缺 | 对 prefill 特别有效（窗口大） | 中 |
| 7 | **跨层级就绪调度 + VTCM 双缓冲** | 不等全部就绪，按剩余就绪时间分组执行（BigMoMo） | 部分（`process_deferred`/`matmul_split`） | HTP 利用率 19%→98%（其配置） | 中 |
| 8 | **投机窗口复用** | 一次搬运服务整棵草稿树的多个候选（权重复用+连续读） | 缺 | decode 收益（BigMoMo 核心） | 中 |
| 9 | **异构执行：冷/小专家回退 CPU** | 小/突发专家上 NPU 不合算（Fiddler / NPUMoE） | 缺 | 减少 dispatch/同步开销 | 中 |
| 10 | **专家分组执行** | 多专家打进一个图，摊薄 NPU 子图派发（NPUMoE） | 缺 | 提升 NPU 利用率 | 中 |

## 三、缓存 / 常驻

| # | 方法 | 说明 | 现状 | 预期 | 代价 |
|---|---|---|---|---|---|
| 11 | **冷启动预热** | `warm_from_prefill`（真实路由）+ 静态热表兜底 | 有 | 降低首 token 冷启动 miss | 一次 host memcpy |
| 12 | **更好的缓存替换** | 稀疏感知/激活 trace（MoE-Infinity）、ML 缓存替换（FlashMoE） | 缺（现为 LRU） | 更高命中率 | 中 |
| 13 | **逐层窗口容量** | 扫每层 `c_il`（layer 0 需 c≈200，多数层 c≈96） | 有（`LLAMA_MOE_LAYER_WINDOWS`） | 找到最优点 | 低 |
| 14 | **KV cache 量化/压缩** | 省 DRAM/NPU buffer，喂给专家常驻 | 待查 | 提高专家命中 | 可能损质量 |

## 四、算法 / 路由

| # | 方法 | 说明 | 现状 | 预期 | 代价 |
|---|---|---|---|---|---|
| 15 | **路由预测** | 免训练 FATE / 低秩 adapter（SpecPrefetch） | 有（LES 免训练预测） | 更高预取命中 | adapter 需训练，与"无损免训练"冲突 |
| 16 | **专家剪枝/合并** | 少专家 = 少搬（CoSMoEs 等） | 缺 | 减搬运 | 质量风险 |
| 17 | **搬运准入剪枝** | 按价值/影响剪投机节点与专家（BigMoMo） | 部分（`--moe-k-decode`） | 少搬（−18%~−23% TPOT） | 约 0.1–0.3% 质量损失（近似） |

## 五、投机 / 解码（本项目有 DSpark）

| # | 方法 | 说明 | 现状 | 预期 | 代价 |
|---|---|---|---|---|---|
| 18 | **接受率加权草稿树剪枝** | 低价值分支的专家不预取/不验证（BigMoMo：82% 验证是白干） | 缺 | 减无用搬运 | 需接受率统计 |
| 19 | **草稿计算盖住目标加载** | DSpark 的 draft forward 藏 target 专家搬运 | 缺 | 藏延迟 | 中 |
| 20 | **D/B 参数扫描** | BigMoMo 发现 D=2,B=2 最优 | 有（`n_max`），可扫 | 找最优点 | 低 |

## 六、系统 / 工程

| # | 方法 | 说明 | 现状 | 预期 | 代价 |
|---|---|---|---|---|---|
| 21 | **I/O 线程数调优** | `LLAMA_MOE_IO_THREADS`（默认 4） | 有 | ⚠️ decode 跟总读字节走（带宽受限），加线程未必有用，需按存储并发能力调 | 低 |
| 22 | **会话常驻 / 避免重初始化** | 减少每轮重载开销 | 部分 | 降低轮间开销 | 低 |
| 23 | **park/unpark 释放物理页** | 现在 Metal-only；可给 Hexagon 加接口 | 缺 | decode 期释放 prefill 段省内存 | 需后端支持 |
| 24 | **功耗模式** | `GGML_HEXAGON_POWER_MODE`（sustained_high_performance / burst） | 有 | 防热降频；`burst` 会长跑热节流 | 低 |
| 25 | **多设备切分** | layer/tensor/row split，跨物理核分摊 I/O 与计算 | 有 | 提升吞吐 | 配置 |

---

## 优先级（按性价比）

**若只挑三个：**
1. **flash 页对齐 + 连续读**（存储层，prefill 直接受益，改动小、风险低）；
2. **运行时 flash 重排**（BigMoMo Anchor Region；见 adoption 文档 2.1）；
3. **层间流水 / 跨层级就绪调度**（把 decode 的 ~52% 暴露搬运压下去）。

**成本低、先做的：** #2 对齐、#13 逐层容量扫、#20 D/B 扫、#21 IO 线程、#24 功耗、#25 多设备。
**收益大、改代码的：** #1 布局重排、#6 层间流水、#7 就绪调度、#8/18/19 投机相关。
**谨慎/近似的：** #16 专家剪枝、#17 准入剪枝（破坏"无损"）。
**与卖点冲突的：** #15 低秩 adapter（需训练）。

---

## 决策点（选方法前先定）

1. **主优化目标**：decode 速度 还是 prefill/TTFT？（工具完全不同）
2. **可用 DRAM / slot bank**：还能不能再挤？（BigMoMo：可用 DRAM 决定天花板）
3. **能否接受近似**：准入剪枝/混合精度剪枝约 0.1–0.3% 质量损失，还是坚持无损？
4. **是否允许训练**：决定要不要上 learned predictor（否则走免训练 FATE / LES 路线）。

---

## 参考

- 论文详解与链接：见 [on-device-moe-offloading-papers.md](on-device-moe-offloading-papers.md)
- 可落地点与代码映射：见 [on-device-moe-offloading-adoption.md](on-device-moe-offloading-adoption.md)
- 全静态量化：见 [on-device-npu-quantization.md](on-device-npu-quantization.md)
