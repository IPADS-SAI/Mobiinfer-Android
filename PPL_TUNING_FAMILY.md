# PPL 调优：per-family / Q1_0-Q2_K-Q2_0 混合（T-0003）

- 日期：2026-11-22
- 分支：`dpsk/w1`（run `run-20261122T172403-w1-2`）
- A100：sai-a100，8×A100-80GB；量化器 `T-0005/src/build/bin/llama-quantize`（含 Q2_0），
  PPL `T-0005/src/build-cuda/bin/llama-perplexity`
- 设备：vivo V2546A（SM8850 / Hexagon v81），`release-v81-sync`，N=128 / W80 / D16 / SPECN=3
- 语料：`t-0037/corpus-wikitext2.txt`（1,292,013 B），PPL 128 chunks × 2048 ctx
- 手机 dense 固定为 `minicpm47-les-hexagon.gguf`（613 张量，Q4_K/Q6_K，
  sha256 `a1e700ae…f2f83`），与线上/历史测速同一文件

## 结论（TL;DR）

1. **≥30 t/s 档拿到 −14.8% PPL 的严格改进**：`sens4_guq2k_dq1`（只把 36–39 层的
   `gate/up` 升 Q2_K、`down` 保持 Q1_0）部署 PPL **33.7477 → 28.7474**，
   vivo N=128 decode **33.25 t/s**（同批 base 33.35，4+4 轮交替），
   **速度打平、PPL 严格更好**，且包只从 4.22 GiB 增到 4.59 GiB。
2. **per-family 的收益主要在 `gate/up`，不在 `down`**：12 层 `gate/up` Q2_K 臂
   PPL **17.2239（−49.0%）**，但 HTP 上 Q2_K `gate/up` 的 kernel 代价很大，
   decode 掉到 **18.8 t/s**；反向（全 40 层 `gate/up` Q1_0 + `down` Q2_K）
   PPL 18.9554、每步字节 +44%，是更差的投放方式。
3. **Q2_K 仍是 20 t/s 档锚点**：全 40 层 `gate/up` Q2_K 的 PPL 9.0994（−73.0%），
   但每步 +88.9%，会落到 ~20 t/s；`sens12`（17.22 @ 18.8 t/s）在速度/PPL 上
   都被 uniform Q2_K（8.66 @ 20.05 t/s）支配，不应部署。
4. **Q2_0 与 Q1_0 混合可用但收益最小**：12 层只升 `down` Q2_0，PPL
   **30.3121（−10.2%）**，包 4.64 GiB；同体积下被 `sens4`（28.75 @ 4.59 GiB）
   支配，与 T-0005 “Q2_0 在 HTP 上不划算”的结论一致。
5. 所有臂的 dense/非专家权重完全一致（手机 `minicpm47-les-hexagon.gguf` 的
   Q4_K/Q6_K 原始字节）；差异只来自 120 个 routed-expert 张量的 family 类型；
   未使用 imatrix（对 Q1_0 是 no-op，T-0002 已证）。

## 1. 实验内容

**假设**：Q1_0 专家只有 1.125 bpw，部署 PPL 33.75 不可用；把**同一层内不同 family**
（`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`）分配不同位宽，可以在“只动一部分
张量”的前提下换到更大 PPL 收益，从而在 ≥30 t/s 档拿到可用精度。

- **变量**：只改 120 个 routed expert 张量的类型；每层三族独立选择
  Q1_0（1.125 bpw，block 128）/ Q2_K（2.625 bpw，block 256）/ Q2_0（2.25 bpw，block 64）。
  非专家张量在量化时用 Q2_K 占位，PPL/部署前用 `merge_deployed.py` 换成手机 dense
  （Q4_K/Q6_K）的原始量化字节，保证与端侧一致。
- **臂**（5 个新臂 + 基线）：

  | id | gate/up | down | 层 | 每步专家字节 |
  |---|---|---|---|---:|
  | `base` | Q1_0 | Q1_0 | 全部 40 | 135.0 MiB |
  | `guq1_dq2k` | Q1_0 | Q2_K | 全部 40 | 195.0 MiB (+44.4%) |
  | `guq2k_dq1` | Q2_K | Q1_0 | 全部 40 | 255.0 MiB (+88.9%) |
  | `sens4_guq2k_dq1` | Q2_K | Q1_0 | 36–39 的 gate/up 升 Q2_K | 147.0 MiB (+8.9%) |
  | `sens12_guq2k_dq1` | Q2_K | Q1_0 | 8–11,32–39 的 gate/up 升 Q2_K | 171.0 MiB (+26.7%) |
  | `q20_d_sens12` | Q1_0 | Q2_0 | 8–11,32–39 的 down 升 Q2_0 | 148.5 MiB (+10.0%) |

  （每步 = top-8 × 40 层 × 8 的专家读取量；Q1_0 family 147,456 B/专家，
  Q2_K 344,064 B，Q2_0 294,912 B。）
- **敏感层定义**：沿用 T-0005 的 8–11/32–39（12 层）；另加 36–39（4 层）保守臂，
  因为 T-0002 证明 36–39 全 family Q2_K 的部署 PPL 收益最大。
- **对照**：`base` = T-0002 `deployed-a.gguf`（Q1_0 全专家 + 手机 dense），
  本次在 GPU5 重跑，PPL **33.7477**，与历史值逐位一致（33.7477 ±0.32337）。
- **Q2_0 的 CUDA 限制**：`ggml-cuda` 没有 Q2_0 kernel；`q20_d_sens12` 先用
  `llama-quantize --allow-requantize … F16` 无损反量化（69.4 GB）再跑同一 128 chunk。
  T-0005 的 CPU 对照显示单 chunk 差 ≤1.4%，绝对 PPL 以同法为准。

## 2. 复现命令

### 2.1 规则（first-match-wins，具体规则在前）

```sh
# sens12_guq2k_dq1：12 层 gate/up Q2_K，down 与其余层 Q1_0
blk\.(8|9|10|11|32|33|34|35|36|37|38|39)\.ffn_(gate|up)_exps\.weight=Q2_K
blk\.(8|9|10|11|32|33|34|35|36|37|38|39)\.ffn_down_exps\.weight=Q1_0
ffn_(gate|up|down)_exps\.weight=Q1_0

# q20_d_sens12：12 层 down Q2_0，其余 Q1_0
blk\.(8|9|10|11|32|33|34|35|36|37|38|39)\.ffn_down_exps\.weight=Q2_0
ffn_(gate|up|down)_exps\.weight=Q1_0
```

其余三臂见 `artifacts/a100/rules/`。量化/合并/PPL/pack 全流程脚本
`artifacts/a100/arm_pipeline.sh`、`run_all.sh`。

### 2.2 量化 + 部署口径合并 + PPL

```sh
T=~/workspace/csm/moe-exp/T-0003; cd $T
Q=~/workspace/csm/moe-exp/T-0005/src/build/bin/llama-quantize
P=~/workspace/csm/moe-exp/T-0005/src/build-cuda/bin/llama-perplexity
$Q --tensor-type-file rules/<arm>.txt \
  /home/ma-user/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf models/<arm>.gguf Q2_K 16
python3 merge_deployed.py \
  --dense ~/workspace/csm/moe-exp/t-0040/model/minicpm47-les-hexagon.gguf \
  --arm models/<arm>.gguf --experts-manifest manifest-experts.json \
  --out deployed/<arm>.gguf
# Q2_0 臂额外：$Q --allow-requantize deployed/<arm>.gguf deployed/<arm>-f16.gguf F16 16
CUDA_VISIBLE_DEVICES=<g> $P -m deployed/<arm>.gguf -f ~/workspace/csm/moe-exp/t-0037/corpus-wikitext2.txt \
  -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128
```

### 2.3 extract + Hexagon tiled pack

```sh
LIB=~/workspace/csm/moe-exp/T-0005/lib-q20   # 含 Q2_0 的 nb[] host shim
python3 tools/moe/experts.py extract --model models/<arm>.gguf --out-dir raw/<arm> --force
LD_LIBRARY_PATH=$LIB python3 tools/moe/pack_hexagon.py \
  --experts raw/<arm> --out-dir tiled/<arm> --lib-dir $LIB
sha256sum tiled/<arm>/layer_*.bin > tiled/<arm>.sha256   # 40/40 层
```

### 2.4 vivo N=128（devlock 串行，base/arm 交替）

```sh
ssh vivo 'cd ~/csm/moe-exp && ./devlock -w 1800 bash t0003-bench-ab.sh <tag> <experts_dir>'
# 每轮先 dropcache.py，再
# env N=128 SPECN=3 WINDOW=80 DEPTH=16 MODEL=minicpm47-les-hexagon.gguf \
#     EXPERTS=<dir> ./bench.sh
```

## 3. 结果

### 3.1 A100 PPL（部署口径，128 chunks × 2048 ctx）

| 臂 | 专家类型 | PPL | ± | 相对 base | tiled 包 |
|---|---|---:|---:|---:|---:|
| `base` | Q1_0 ×120 | 33.7477 | 0.32337 | — | 4.22 GiB |
| `guq1_dq2k` | 40×(gate/up Q1_0 + down Q2_K) | **18.9554** | 0.17339 | −43.8% | 6.09 GiB |
| `guq2k_dq1` | 40×(gate/up Q2_K + down Q1_0) | **9.0994** | 0.07456 | −73.0% | 7.97 GiB |
| `sens4_guq2k_dq1` | 4 层 gate/up Q2_K，down Q1_0 | **28.7474** | 0.26702 | −14.8% | 4.59 GiB |
| `sens12_guq2k_dq1` | 12 层 gate/up Q2_K，down Q1_0 | **17.2239** | 0.15232 | −49.0% | 5.34 GiB |
| `q20_d_sens12` | 12 层 down Q2_0，其余 Q1_0 | **30.3121** | 0.28874 | −10.2% | 4.64 GiB |

关键对比：

- 同样只动 12 层，`gate/up` Q2_K（17.22）远好于 `down` Q2_0（30.31），
  且包只大 0.7 GiB；Q2_K 的 block-256 + 双 scale 量化比 Q2_0 的 group-64
  更适配 routed experts。
- 反向臂（`down` Q2_K 全 40 层，18.96）与“12 层 gate/up Q2_K”（17.22）PPL 接近，
  但前者改了 3.3× 的张量数、每步字节 +44%，说明 **`gate/up` 是更高效的投放位置**。
- 全 40 层 `gate/up` Q2_K 的 PPL 9.10 已接近 uniform Q2_K（8.66，T-0005），
  只保留 `down` 的 1-bit，是“精度优先”臂。
- Q2_0 路线的 PPL 增益最小（−10.2%），并被同体积 Q2_K 臂支配。

### 3.2 端侧速度（vivo N=128，`release-v81-sync`，devlock 串行）

> 18:03 首次尝试时 vivo 只剩 1–2% 电量、CPU 被限到 1.38 GHz，base/sens4 都只有
> 9.2 t/s；该批只能证明相对不回退，绝对值无效，已丢弃。充电到 13% 后于 19:42/19:48
> 重跑，CPU 恢复正常（base 33 t/s），下表只用这次满速数据。

| 臂 | 轮次 | Generation（4 轮） | 中位 | Prompt 中位 | PPL |
|---|---:|---|---:|---:|---:|
| `base`（Q1_0） | 8（跨两批） | 30.0 / 33.3 / 33.4 / 33.4；32.9 / 32.1 / 33.3 / 33.3 | **33.35** | 27.4 | 33.7477 |
| `sens4_guq2k_dq1` | 4 | 33.0 / 33.3 / 33.2 / 33.3 | **33.25** | 26.3 | 28.7474 |
| `sens12_guq2k_dq1` | 4 | 18.8 / 18.7 / 18.8 / 18.9 | **18.8** | 23.4 | 17.2239 |
| `guq1_dq2k` / `guq2k_dq1` / `q20_d_sens12` | — | 未测（时间/电量） | — | — | 18.9554 / 9.0994 / 30.3121 |

- **`sens4` 是 ≥30 t/s 的验收点**：33.25 t/s（比值 0.997 vs base），PPL 28.75
  （−14.8%）——在同样 33 t/s 下严格优于 Q1_0 基线，是本次唯一“免费”的精度收益。
- **`sens12` 掉到 18.8 t/s**：每步专家字节只 +26.7%，速度却 −44%，说明瓶颈不是 I/O
  而是 **Q2_K `gate/up` 的 HTP kernel**；12 层就足以把 decode 拉进 20 t/s 档。
  它的 PPL 17.22 仍差于 uniform Q2_K 的 8.66（同 18.8–20 t/s 档），因此被支配。
- `sens4` 的 4 层 Q2_K `gate/up` 几乎不付速度代价（33.25 vs 33.35，在 ±0.3 噪声内），
  与 T-0002 “36–39 全 family Q2_K 速度中性（30.05 vs 30.25）”一致；Q2_K 的代价
  在层数增加后非线性放大（4 层 ≈0%，12 层 −44%）。

### 3.3 证据

- A100 PPL/量化/合并/pack 原始日志：`artifacts/a100/logs/`（含 `ppl-*.log`、
  `types-*.json`、`verify-*.log`、`pack-*.log`、`size-*.txt`）
- tiled manifest 与逐层 sha256：`artifacts/a100/manifests/`、`artifacts/a100/sha/`
- vivo 原始测速日志与 summary：`artifacts/phone/sens4/`
- 大文件（arm/deployed GGUF、tiled 包）留在 A100
  `~/workspace/csm/moe-exp/T-0003/{tiled,logs}`，未回传本机。

## 4. 结论与下一步

1. **推荐部署 `sens4_guq2k_dq1`**：PPL 28.75（−14.8%）、decode 33.25 t/s、
   tiled 4.59 GiB。它严格支配 Q1_0 基线（同速、更低 PPL），是 30 t/s 档的新默认候选；
   代价只有 36–39 四层 `gate/up` 从 Q1_0 换 Q2_K。
2. **`sens12` 不作为 30 t/s 方案**：17.22 PPL 很好，但 18.8 t/s 已进入 20 t/s 档，
   且同档 uniform Q2_K（8.66 @ 20.05）PPL 更好，直接支配它。若未来给 Q2_K
   `gate/up` 写更快的 HVX kernel，12 层方案才值得重开。
3. **未测速的两臂**：`guq1_dq2k`（PPL 18.96，每步 +44%）预计 ~24 t/s，
   可能落在 sens4 与 Q2_K 之间的 Pareto 点；`guq2k_dq1`（9.10，+89%）预计 ~20 t/s，
   与 uniform Q2_K 接近但 PPL 略差（8.66 vs 9.10），优先级低。`q20_d_sens12`
   （30.31，+10%）已被 sens4 支配，无需再测。
4. **Q2_0 结论**：down-only 12 层只换到 −10.2% PPL，且 HTP 上 Q2_0 kernel 慢；
   在 30 t/s 档没有位置，除非专用 kernel。
5. 下一步：把 `sens4` 推进到 verifier/集成；再做 per-layer 敏感度搜索
   （当前 36–39 来自 T-0002 的粗粒度结论），目标是在 ≥30 t/s 下继续加层；
   同时测 `guq1_dq2k` 的速度以补全 Pareto 前沿。

## 5. 设备电量记录（2026-11-22）

| 时间 | 电量 | 电压 | 电流 | CPU7 max | 备注 |
|---|---:|---:|---:|---:|---|
| 17:52 | 2% | 3.117 V | −62 mA | — | devlock 拒绝（<5%） |
| 17:56 | 2% | 3.160 V | +166 mA | — | 关屏后转正 |
| 18:03 | 2% | 3.29 V | +529 mA | 1.38 GHz | 短批 6 次，base/sens4 均 9.2 t/s（无效） |
| 18:05 | 1% | 3.203 V | — | 1.38 GHz | 立即停止占用设备 |
| 18:11 | 1% | 3.370 V | +409 mA | 1.38 GHz | Charge counter 75,000，充电中 |
| 19:38 | 13% | 3.693 V | +495 mA | 1.38 GHz* | 满速复测 base 33.4 t/s（*读数滞后） |
| 19:42 | 13% | — | — | — | `sens4` 4+4 轮，33.25 vs 33.35 |
| 19:48 | 13% | — | — | — | `sens12` 4+4 轮，18.8 vs 33.1 |

> 决策：低电时没有伪造电量或绕过 devlock；只用真实值 2% 跑了一次短批，
> 发现绝对速度无效后立即停手；充电到 13% 才重跑正式速度门禁。
> 手机侧 `t0003-sens4`/`t0003-sens12` 包在测速后已删除（A100 仍保留 tiled 包）。
