# PPL：Q1_0 专家不变，output/head 与 attention/dense 精度（T-0004）

- 日期：2026-11-22
- 任务：T-0004「PPL：output/head/dense 精度」；分支 `dpsk/w5`；run `run-20261122T155210-w5-1`
- 设备：sai-a100（A100-80GB ×8，CUDA `build-cuda`）+ vivo V2546A（SM8850 / Hexagon v81 / HTP0）
- 目标：**Q1_0 专家固定不变**，分别试 output/head 的 Q4_K/Q2_K、attention/dense 的 Q3_K/Q4_K；
  记录 A100 PPL 与 vivo N=128 t/s，保持 ≥30 t/s。**不使用全 Q1_0 dense**（已知 11.2 t/s）。

## 1. 结论（先看这里）

1. **Q1_0 专家下，Q2_K dense 是 PPL 的主要瓶颈**：attention/SSM/shared-FFN 从 Q2_K 默认混合
   升到 **Q4_K**，PPL 从 **103.07 → 36.21（−64.8%）**；升到 Q3_K 只到 41.36（−59.9%），
   再多花字节升 Q4_K 还能再降 12.4%，因此 **Q3_K 这一档不值得**。
2. **收益主要来自 SSM/shared-FFN，不是 attention**：只升 attention 到 Q4_K 是 90.60（−12.1%）；
   只升 SSM+shared-FFN 到 Q4_K 是 40.65（−60.6%）；两者都升是 36.21，对数增量近似可加。
3. **output head Q4_K 基本是“免费”的**：Q6_K→Q4_K 只 +0.16%（uniform Q4 dense）/
   +0.20%（原 dense 类型），省 125 MiB。**Q2_K head 不免费**：+13.4%（d4）/+15.3%（d2），
   不建议。
4. **部署中的 33.3 t/s 配置 PPL 是 34.63，不是之前写的 103.07**。之前的 103.07 对应
   `Q2_K` dense+output（`arm-a`），而手机上 33.3 t/s 用的是原 `minicpm47-les-hexagon.gguf`
   的 Q4/Q6 dense。按原张量类型重建成 Q1_0 专家模型后 PPL 34.63；仍比 Q3_K 对照 6.39
   差 5.4×，**不可用**，下一步必须升专家位宽（T-0003/T-0005 在做）。
5. **推荐点（仍保留 Q1_0 专家）**：`orig_o4` = 原 dense 类型 + `output.weight=Q4_K`。
   PPL **34.70**（vs 原类型 34.63，+0.2%），dense 文件 1.553 GB（比原 1.663 GB 小 110 MB），
   vivo N=128 decode **32.6–33.5 t/s**。若再省内存可用 `d4o4t4`
   （uniform Q4_K 含 token_embd + Q4 output，PPL 35.54，dense 约 1.36 GiB），但未上机测速。

## 2. 实验设计

### 2.1 固定项与变量

- 源权重：`/home/ma-user/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf`
  （69,375,191,008 B，`qwen35moe`，40 层 / 256 专家 / top-8）。
- **专家固定 Q1_0**：120 个 `ffn_{gate,up,down}_exps.weight`，A100 全模型里 4.22 GiB；
  vivo 用同一份 `experts-q1-30tps` tiled 包（4.3 GB，layer SHA256 与 A100 `tiled-a` 一致）。
- 变量：
  - attention：100 个 `attn_*` 张量；
  - SSM：90 个 `ssm_{out,alpha,beta}.weight`；
  - shared-FFN：120 个 `ffn_{gate,up,down}_shexp.weight`；
  - `token_embd.weight`（默认保持 Q2_K，除 `d4o4t4`/`orig*`）；
  - `output.weight`：Q2_K / Q4_K / Q6_K。
- 其余 1D/F32 张量不量化，所有臂一致。

### 2.2 臂定义（实际量化类型，来自 quantize 日志审计）

| 臂 | attention (100) | SSM (90) | shared-FFN (120) | token_embd | output | 专家 |
|---|---|---|---|---|---|---|
| `d2o2` / `d2o4` / `d2o6` | Q2_K 50 + Q3_K 10 + Q4_K 40（llama-quantize Q2_K 默认混合） | Q2_K 90 | Q2_K 80 + Q3_K 40 | Q2_K | Q2_K / Q4_K / Q6_K | Q1_0 |
| `d3o2` / `d3o4` / `d3o6` | Q3_K 100 | Q3_K 90 | Q3_K 120 | Q2_K | Q2_K / Q4_K / Q6_K | Q1_0 |
| `d4o2` / `d4o4` / `d4o6` | Q4_K 100 | Q4_K 90 | Q4_K 120 | Q2_K | Q2_K / Q4_K / Q6_K | Q1_0 |
| `d4o4t4` | Q4_K 100 | Q4_K 90 | Q4_K 120 | **Q4_K** | Q4_K | Q1_0 |
| `a4o6` | **Q4_K 100** | Q2_K 90 | Q2_K 80 + Q3_K 40 | Q2_K | Q6_K | Q1_0 |
| `s4o6` | Q2_K 50 + Q3_K 10 + Q4_K 40 | **Q4_K 90** | **Q4_K 120** | Q2_K | Q6_K | Q1_0 |
| `orig` | Q4_K 70 + Q6_K 30 | Q4_K 90 | Q4_K 100 + Q6_K 20 | Q4_K | Q6_K | Q1_0 |
| `orig_o4` | 同 `orig` | 同 `orig` | 同 `orig` | Q4_K | **Q4_K** | Q1_0 |

`orig` 的规则由原 `minicpm47-les-hexagon.gguf` 的逐张量类型表生成（312 条精确规则），
因此它复现的是“手机上 33.3 t/s 那个 dense 类型组合 + Q1_0 专家”，而不是 Q2_K dense。

### 2.3 PPL 口径

- wikitext-2 test，`t-0037/corpus-wikitext2.txt`（1,292,013 B，sha256 `aca2f467…`），
  128 chunk × 2048 ctx，`-c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128`。
- 同一份语料/同一 tokenizer/同一 chunk 划分；`d2o6` 与 T-0037/T-0098 的 `arm-a` **逐字节相同**
  （sha256 `ef8684c2…`，PPL 103.0652 与历史 103.07 一致），证明流水线可复现。
- 配对统计：同 chunk 的 `log(PPL_arm) − log(PPL_d2o6)`，2000 次 bootstrap 95% CI。

## 3. 结果

### 3.1 A100 PPL 矩阵（Q1_0 专家固定）

| 臂 | PPL | ± | Δ vs `d2o6`（配对几何均值） | 95% CI | 全模型 MiB |
|---|---:|---:|---:|---|---:|
| `d2o6`（Q2_K dense, Q6 output） | 103.0652 | 1.1000 | 0 | — | 5529.3 |
| `d2o4` | 103.4929 | 1.1030 | +0.39% | [+0.38, +0.41] | 5404.3 |
| `d2o2` | 118.8372 | 1.3010 | +15.51% | [+15.37, +15.68] | 5290.7 |
| `d3o6` | 41.3620 | 0.4012 | −59.38% | [−59.59, −59.16] | 5538.7 |
| `d3o4` | 41.4083 | 0.4011 | −59.33% | [−59.55, −59.11] | 5413.8 |
| `d3o2` | 47.3045 | 0.4696 | −53.47% | [−53.75, −53.22] | 5300.2 |
| `d4o6` | 36.2091 | 0.3444 | −64.40% | [−64.56, −64.23] | 5717.2 |
| `d4o4` | 36.2684 | 0.3443 | −64.34% | [−64.52, −64.15] | 5592.2 |
| `d4o2` | 41.1402 | 0.3998 | −59.53% | [−59.74, −59.34] | 5478.6 |
| `d4o4t4`（+token Q4） | **35.5412** | 0.3361 | −65.01% | [−65.18, −64.84] | 5705.8 |
| `a4o6`（只升 attention） | 90.6045 | 0.9337 | −11.72% | [−11.82, −11.62] | 5636.0 |
| `s4o6`（只升 SSM+shared） | 40.6517 | 0.3978 | −60.03% | [−60.24, −59.82] | 5610.5 |
| `orig`（原 dense 类型, Q6 output） | **34.6260** | 0.3283 | −65.91% | [−66.07, −65.73] | 5915.9 |
| `orig_o4`（原 dense 类型, Q4 output） | **34.6963** | 0.3284 | −65.83% | [−66.00, −65.67] | 5790.9 |

Q3_K 对照（T-0037，Q3_K 专家 + Q6_K 其他）PPL = **6.3928**；Q1_0 专家最好也只有 34.63，
差 5.4×。

关键比值（用最终 PPL 直接相除）：

- dense Q2→Q4：36.268 / 103.065 = **−64.8%**；Q2→Q3：41.408 / 103.065 = −59.8%；
  Q3→Q4：36.268 / 41.408 = **−12.4%**。
- output Q6→Q4：36.268 / 36.209 = +0.16%（d4）；34.696 / 34.626 = +0.20%（orig）。
- output Q4→Q2：41.140 / 36.268 = **+13.4%**（d4）；118.837 / 103.065 = +15.3%（d2）。
- token_embd Q2→Q4：35.541 / 36.268 = **−2.0%**（`d4o4t4` vs `d4o4`）。
- attention-only Q4：90.604 / 103.065 = −12.1%；SSM+shared-only Q4：40.652 / 103.065 = −60.6%。

### 3.2 归因（为什么 Q3_K 不值得）

- `a4o6`（只升 attention）−12.1%，`s4o6`（只升 SSM+shared-FFN）−60.6%，合并 `d4o6` −64.4%。
  对数增量近似可加（`exp(-0.129-0.930)=0.347` → 35.7，实测 36.2）。
- 本模型 40 层里大量层是 SSM 路径，shared-FFN/SSM 的 2-bit 误差主导输出；attention 的
  100 个张量只占小部分字节和误差。**若只能升一个 family，先升 SSM/shared-FFN。**
- Q3_K 处在中间：比 Q2 好很多（−59.8%），但比 Q4 差 12.4%，而 Q4 只多约 0.15 GiB dense；
  所以 **Q3_K 是“花了字节但没买到最优”的档位**，不建议作为部署点。

### 3.3 vivo N=128（同包 Q1_0 专家，`release-v81-sync`）

命令：`N=128 SPECN=3 WINDOW=80 DEPTH=16`，`devlock` 串行，每跑前 `dropcache.py`；
两轮 ABBA（每轮 base/cand/cand/base）。

| 配置 | dense 文件 B | PPL | decode t/s（2 跑） | prefill t/s（2 跑） |
|---|---:|---:|---|---|
| 原 `minicpm47-les-hexagon.gguf`（Q4/Q6 dense） | 1,662,732,800 | 34.63* | **35.3 / 33.9**；**35.8 / 33.5** | 33.9 / 31.4；32.4 / 31.7 |
| `d4o4` dense（uniform Q4，output Q4） | 1,345,019,360 | 36.27 | **33.7 / 33.4** | 34.4 / 30.9 |
| `orig_o4` dense（原类型，output Q4） | 1,553,339,360 | 34.70 | **33.5 / 32.6** | 31.8 / 30.5 |

\* `34.63` 是按原张量类型重建的 Q1_0 专家模型 PPL，不是原文件直接测的（原文件专家是 Q4/Q6，
不能与 Q1_0 专家包混测）。

- 三个配置 **N=128 都 ≥30 t/s**（最低 32.6）。候选组间有重叠，且每臂只有 2 跑：
  `d4o4` 相邻配对 −1.5%，`orig_o4` 相邻配对 −2.7%；不满足 GATES「≥3% 且组间不重叠」，
  只能判定“没有明显回退”，不能宣称提速。
- 电池全程 13–14%（充电中但净电流约 −1.4 A），热状态未做长程控制；速度只作 ≥30 的
  通过/不通过证据，不作为精确 A/B。

## 4. 对 `Q1_SPEED_30TPS.md` 的更正

`Q1_SPEED_30TPS.md` 写“同一 Q1_0 专家包的 PPL = 103.07”，但该 PPL 来自 `arm-a`
（Q2_K dense + Q2_K/混合 attention + Q6 output）；手机上 33.3 t/s 的模型是
`minicpm47-les-hexagon.gguf`（原 Q4/Q6 dense），两者不是同一个 dense。按原张量类型重建后：

- `orig`（Q1_0 专家 + 原 dense 类型 + Q6 output）PPL = **34.63**；
- `orig_o4`（只把 output 换 Q4_K）PPL = **34.70**。

所以 33.3 t/s 配置的真实精度代价是 **PPL 34.6（vs Q3_K 6.39，5.4×）**，而不是 103。
结论不变（Q1_0 专家仍不可用），但“103”不能再作为该手机配置的精度数字引用。

## 5. 建议

1. **不要部署 Q2_K dense**：它是 103 PPL 的主因；也不要用 uniform Q3_K（41.4），
   性价比低于 Q4_K（36.2）。
2. **保留 Q1_0 专家的前提下，最佳质量/体积点是 `orig_o4`**：原 dense 类型 + output Q4_K，
   PPL 34.70、dense 1.553 GB、N=128 32.6–33.5 t/s。output Q4_K 是唯一几乎免费的降级；
   output Q2_K 不要用。
3. 若要把 PPL 从 34 继续往下压，**必须动专家位宽**（Q1_0→Q2_K/Q2_0 或按层混合），
   dense 侧已经到 Q4/Q6 的边际收益地板；T-0003/T-0005 的专家混合是正确方向。
4. 若 dense 内存再紧，`d4o4t4`（uniform Q4 含 token_embd + Q4 output）PPL 35.54、
   dense ~1.38 GB，可作为备选，但本轮未上机测速。

## 6. 复现

A100（`~/workspace/csm/moe-exp/q2-30tps/t0004-dense/`）：

```sh
B=$HOME/workspace/csm/moe-exp/build-cuda/bin
SRC=/home/ma-user/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf
C=$HOME/workspace/csm/moe-exp/t-0037/corpus-wikitext2.txt

# 规则：专家 Q1_0 + 指定 dense family/类型（见 artifacts/a100/rules_dense_*.txt）
$B/llama-quantize --tensor-type-file rules/rules_dense_4.txt \
  --output-tensor-type Q4_K "$SRC" out/arm-d4o4.gguf Q2_K 16

# 原 dense 类型（312 条精确规则，由原 GGUF 类型表生成）
$B/llama-quantize --tensor-type-file rules/rules_orig_q1.txt \
  --output-tensor-type Q4_K "$SRC" out/arm-orig_o4.gguf Q2_K 16

# PPL（每臂一张 GPU）
CUDA_VISIBLE_DEVICES=0 $B/llama-perplexity -m out/arm-d4o4.gguf -f "$C" \
  -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128
```

手机（devlock 串行）：

```sh
ssh vivo 'cd ~/csm/moe-exp && ./devlock -w 1800 bash t0004-bench2.sh'
# 脚本内：N=128 SPECN=3 WINDOW=80 DEPTH=16 MODEL=<dense gguf> \
#         EXPERTS=$HOME/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q1-30tps
#         ./bench.sh
```

## 7. 产物

- 本 run：`/home/yydh/.local/state/dpsk-q2/runs/T-0004/run-20261122T155210-w5-1/`
  - `PPL_TUNING_DENSE.md`、`summary.md`、`result.json`
  - `artifacts/a100/logs/`（quant/ppl/strip 原始日志）
  - `artifacts/a100/ppl-matrix.{json,tsv}`、`artifacts/a100/rules_dense_*.txt`
  - `artifacts/vivo/t0004-bench-*/`、`artifacts/vivo/t0004-bench2-*/`（bench 原始日志）
- A100：`sai-a100:~/workspace/csm/moe-exp/q2-30tps/t0004-dense/`
  （`out/arm-*.gguf` 约 70 GB，保留在 A100；`out/arm-*-dense.gguf` 是剥离专家后的手机模型）
- vivo：`~/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/`
  - `dense-q1exp-d4o4-t0004.gguf`（sha256 `aba12340…`）
  - `dense-q1exp-orig-o4-t0004.gguf`（sha256 `566bd302…`）
  - 专家包 `experts-q1-30tps`（未改）

## 8. 限制与风险

- `orig`/`orig_o4` 是“原张量类型表 + 从 BF16 重新量化”，不是原文件的逐字节复制；原文件可能
  用过 imatrix/不同舍入，因此 34.63/34.70 有少量重建误差。趋势（Q4 head ≈ 免费）与
  uniform 臂一致。
- PPL 只在 wikitext-2 上测，未做任务级/中文质量评估；Q1_0 专家的 34.6 仍属不可用档。
- vivo 每配置仅 2 跑，电池 13–14%、设备与其他 worker 共享；速度差 <5% 不显著。
- 未测 `d4o4t4` 与 `orig_o4` 的 4+4 轮 A/B；若要把 output Q4 作为默认，建议补 4 轮轮转。
- 未使用 imatrix；专家 Q1_0 + imatrix 是另一条可能降低 PPL 的路径（T-0002）。
