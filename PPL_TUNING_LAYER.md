# PPL 调优：逐层混合 A1/A2/A3（36-39 / +8-11 / +32-35）

- 日期：2026-11-22
- 任务：T-0001「PPL：逐层混合 A1/A2/A3」；分支 `dpsk/w2`；run `run-20261122T173636-w2-2`
- 设备：sai-a100（A100-80GB，CUDA `build-cuda`）+ vivo V2546A（SM8850 / Hexagon v81 / HTP0）
- 目标：按 T-0098 敏感度 36-39 > 8-11 > 32-35，把敏感层专家从 Q1_0 升到 Q2_K，
  记录每臂 PPL / vivo N=128 t/s / 包大小，并保持 ≥30 t/s。

## 1. 结论（先看这里）

1. **只有 A1 同时满足 ≥30 t/s 和明显降 PPL**：36-39 层专家 Q2_K、其余 Q1_0，
   部署口径 PPL **33.75 → 26.31（−22.0%，95% CI [−22.6, −21.5]）**；
   vivo N=128 decode **34.1 t/s**（同批基线 34.2，比值 0.997），专家包 4.78 GiB。
2. **A2（+8-11）与 A3（+32-35）精度收益更大，但速度跌破 30**：
   A2 部署 PPL **17.61（−47.8%）**、A3 **13.37（−60.4%）**；然而 t/s 掉到
   **17.7 / 16.8**（约为基线的 50–52%）。原因是专家包 5.34 / 5.91 GiB 越过了
   vivo 的专家包尺寸悬崖（本机约 4.8–5.3 GiB；与 T-0003 的 4.93 GB 通过 /
   5.03 GB 跌破 30 一致），不是 Q2_K 计算本身慢——A1 只加 0.56 GiB 仍是 34.1 t/s。
   机制：`src/llama-moe-slot.cpp:1761` 对**每一层**按 `n_slots = moe_slot_bank_size()`
   （默认 BANK=256 = 全部专家）绑定居留槽，常驻内存≈整个专家包大小；A2/A3 超过设备
   可用 RAM 后进入 zram/回收抖动。T-0003 已试 BANK/窗口/NOMAP/IO 调参，无法把
   5 GB 以上的包拉回 30 t/s。
3. **A1 是当前唯一可部署点**：相对旧 Q1_0 包精度 −22%，速度中性；A2/A3 需要更小的
   常驻/流式方案（或更大 RAM 的设备）才可能回到 ≥30，当前不建议部署。
4. 「其他权重 Q2_K」的 arm 级 PPL 为 83.97 / 58.35 / 41.62（A1/A2/A3），这是量化臂本身
   的数值；手机实际 dense 是 Q4_K/Q6_K，故上表用**部署口径**（手机 dense + 该臂专家）
   作为精度结论。两套数字都记录在 §4。

## 2. 实验内容

- **假设**：T-0098 的层敏感度排序（36-39 > 8-11 > 32-35 > …）在 Q1_0→Q2_K 升级上成立；
  逐层加 4 层 Q2_K 的 PPL 收益递减，而速度由专家包总大小（不是层数）决定。
- **固定项**：
  - 源权重 `/home/ma-user/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf`
    （`qwen35moe`，40 层 / 256 专家 / top-8）；
  - 量化器 `build-cuda/bin/llama-quantize`，基础类型 `Q2_K`（即非专家权重 Q2_K），
    `-t 16`，`--tensor-type-file` 只覆盖 120 个 `ffn_{gate,up,down}_exps.weight`；
  - PPL：`t-0037/corpus-wikitext2.txt`，128 chunk × 2048 ctx，确定性；
  - 手机：`release-v81-sync`，N=128 / SPECN=3 / WINDOW=80 / DEPTH=16，`devlock` 串行。
- **变量（三臂）**：
  | 臂 | 专家 Q2_K 层 | 其余专家 | 非专家 |
  |---|---|---|---|
  | A1 | 36,37,38,39 | Q1_0 | Q2_K |
  | A2 | 36-39 + 8,9,10,11 | Q1_0 | Q2_K |
  | A3 | 36-39 + 8-11 + 32,33,34,35 | Q1_0 | Q2_K |
- **对照**：手机现包 `experts-q1-30tps`（40 层全 Q1_0，4.22 GiB）；
  部署口径基线 PPL 33.7477（T-0002 同流水线、无 imatrix）。

## 3. 复现命令

### 3.1 A100：规则 + 量化

```sh
T=~/workspace/csm/moe-exp/T-0001
B=~/workspace/csm/moe-exp/build-cuda/bin
SRC=/home/ma-user/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf

cat > $T/rules/a1.txt <<'EOF'
blk\.(36|37|38|39)\.ffn_(gate|up|down)_exps\.weight=Q2_K
blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=Q1_0
EOF
# a2: (36|37|38|39|8|9|10|11)；a3: (32|33|34|35|36|37|38|39|8|9|10|11)

$B/llama-quantize --tensor-type-file $T/rules/a1.txt "$SRC" $T/models/a1.gguf Q2_K 16
# A1 实际直接复用 T-0002 的 out/arm-icn.gguf（同源/同规则，T-0002 用 -t 32 生成；
# 本次重测 PPL 83.9691 与 T-0002 逐位一致，故未重跑量化）。A2/A3 用同一命令新量化
# （-t 16），三臂 PPL/pack 并行，量化实测 133 s/臂。
```

### 3.2 A100：arm 级 PPL 与部署口径 PPL

```sh
P=$B/llama-perplexity
C=~/workspace/csm/moe-exp/t-0037/corpus-wikitext2.txt
DENSE=~/workspace/csm/moe-exp/t-0040/model/minicpm47-les-hexagon.gguf   # 与手机 dense 同字节

# arm 级（Q2_K 非专家，spec 臂）
CUDA_VISIBLE_DEVICES=2 $P -m $T/models/a1.gguf -f $C \
  -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128

# 部署口径：手机 dense + 该臂 120 个专家张量
python3 $T/merge_deployed.py --dense $DENSE --arm $T/models/a1.gguf \
  --experts-manifest $T/manifest-experts.json --out $T/deployed/a1.gguf
CUDA_VISIBLE_DEVICES=2 $P -m $T/deployed/a1.gguf -f $C \
  -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128
```

### 3.3 extract / pack tiled

```sh
LIB=~/workspace/csm/moe-exp/T-0005/lib-q20
python3 $T/tools/moe/experts.py extract --model $T/models/a1.gguf --out-dir $T/raw/a1 --force
LD_LIBRARY_PATH=$LIB python3 $T/tools/moe/pack_hexagon.py \
  --experts $T/raw/a1 --out-dir $T/tiled/a1 --lib-dir $LIB
sha256sum $T/tiled/a1/layer_*.bin > $T/tiled/a1.sha256
```

### 3.4 vivo：adb 推送 + symlink 复用 + 测速

12 个不同的 Q2_K 层（8-11、32-39，每层 264,241,152 B，共 3.17 GB）经 adb 推到
`/sdcard/Download/t0001-q2k/`，再拷到 model 目录 `t0001-q2k/`；三个臂目录中
非 Q2_K 层 symlink 到现包 `experts-q1-30tps`（0-35 层 sha256 与现包逐字节相同），
Q2_K 层指向共享的 `t0001-q2k/`，每个臂只放自己的 `manifest.json`。

```sh
ssh vivo 'cd ~/csm/moe-exp && ./devlock -w 1200 -q env OUT=$HOME/csm/moe-exp/t0001-bench \
  ./t0001-bench-ab.sh \
  base=.../experts-q1-30tps a1=.../experts-t0001-a1 a2=.../experts-t0001-a2 a3=.../experts-t0001-a3 \
  a3=... a2=... a1=... base=...  # 4 轮轮转，实际因电池保护只完成到第 16 跑'
```

每跑前 `dropcache.py` 丢弃 dense + 当前专家包的 page cache；`bench.sh` 内
`N=128 SPECN=3 WINDOW=80 DEPTH=16`，其余默认。

## 4. 结果

### 4.1 PPL（A100，128 chunk × 2048 ctx）

| 臂 | arm PPL（其他权重 Q2_K，spec 臂） | 部署 PPL（手机 dense Q4/Q6） | Δ vs 基线（配对 bootstrap 95% CI） |
|---|---:|---:|---:|
| 基线 Q1_0（`experts-q1-30tps`） | 103.0652 | 33.7477 ± 0.323 | — |
| **A1**（36-39 Q2_K） | 83.9691 | **26.3097 ± 0.239** | **−22.04% [−22.59, −21.46]** |
| **A2**（+8-11） | 58.3501 | **17.6147 ± 0.161** | **−47.81% [−48.74, −46.87]** |
| **A3**（+32-35） | 41.6187 | **13.3654 ± 0.122** | **−60.40% [−61.35, −59.43]** |

> arm 级 PPL 使用 spec 规定的「其他权重 Q2_K」，Q2_K dense 本身是 103 PPL 的主因；
> 手机实际 dense 是 Q4_K/Q6_K，部署口径才是被测包的真实精度。arm 级数字与
> T-0002 的 `icn` 臂（83.9691）逐位一致，证明流水线可复现。

### 4.2 vivo N=128 decode（`release-v81-sync`，devlock 串行）

| 臂 | 专家包 | 包大小 (GiB) | decode t/s（本会话） | 有效窗口 t/s | 判定 |
|---|---|---:|---|---:|---|
| 基线 Q1_0 | `experts-q1-30tps` | 4.22 | 34.2 / 25.1 / 24.0 / 22.9 | 34.2 | 对照 |
| **A1** | `experts-t0001-a1` | **4.78** | **34.1** / 26.2 / 24.2 / 22.1 | **34.1** | **≥30，速度中性（0.997×）** |
| A2 | `experts-t0001-a2` | **5.34** | 18.7 / 16.7 / 14.4 / 9.8 | 18.7 / 16.7 | **<30（0.55×），不可部署** |
| A3 | `experts-t0001-a3` | **5.91** | 17.2 / 16.4 / 13.6 / 5.8 | 17.2 / 16.4 | **<30（0.50×），不可部署** |

- 每臂 4 跑，按 `base,a1,a2,a3` 轮转；**前 6 跑是有效窗口**（电池 5%、无省电降频）。
  第 7 跑起电池降到 4%，Android 省电把**基线本身**也压到 25→23 t/s（第 16 跑 5.8 t/s），
  因此后段只用于看比值，不作为绝对 t/s。
- A1 与基线在四轮里的比值分别为 0.997 / 1.044 / 1.008 / 0.965（中位 **1.003**），
  即使降频段也同步，说明 **A1 无速度回退**。
- A2/A3 的 18.7/17.2 是在与 A1 同一有效窗口、同一冷缓存条件下取得，跌幅真实；
  与 T-0003 的结论一致：专家包 ~4.9 GB 可 ≥30，~5.0-5.3 GB 起跌到 27 以下，
  5.34/5.91 GiB 进一步跌到 ~17。**Q2_K 的层数与总字节是速度主因，不是 kernel。**
  代码层面：每层按 `moe_slot_bank_size()`（默认 256）绑定居留槽
  （`src/llama-moe-slot.cpp:1761`），BANK=256 时常驻≈整包；包超过设备可用 RAM 即抖动。
- prefill 同步下降（A1 30.2 vs 基线 32.7；A2 27.0；A3 24.9），不作为验收指标。

### 4.3 包大小与部署

| 臂 | tiled 专家包 bytes | GiB | Δ vs Q1_0 | + dense 1,662,732,800 B |
|---|---:|---:|---:|---:|
| Q1_0 | 4,529,927,461 | 4.22 | — | 6.19 GB |
| A1 | 5,133,911,587 | 4.78 | +0.56 GiB | 6.80 GB |
| A2 | 5,737,891,375 | 5.34 | +1.13 GiB | 7.40 GB |
| A3 | 6,341,871,163 | 5.91 | +1.69 GiB | 8.00 GB |

手机侧实际只新增 12 个 Q2_K 层（3.17 GB，三臂共享），其余 Q1_0 层用 symlink 复用
`experts-q1-30tps`；已抽检 `layer_000/001/020` 与现包 sha256 完全一致，三个臂目录
分别有 4/8/12 个 Q2_K 链接。正式发布若不做 symlink，按上表 standalone 大小计。

## 5. 证据与产物

- 本 run：`/home/yydh/.local/state/dpsk-q2/runs/T-0001/run-20261122T173636-w2-2/`
  - `PPL_TUNING_LAYER.md`、`summary.md`、`result.json`
  - `artifacts/a100/logs/`：quant / arm-PPL / merge / deployed-PPL / extract / pack 原始日志
  - `artifacts/a100/{a1,a2,a3}.sha256`：三臂 40 层 tiled sha256
  - `artifacts/a100/pushed-layers.sha256`：实际推到手机的 12 层 sha256
  - `artifacts/a100/ppl-deltas.json`：配对 bootstrap 结果
  - `artifacts/vivo/t0001-bench/summary.txt` + 16 个原始 bench 日志、`speed-table.tsv`
- A100 大文件（保留，不回收）：`sai-a100:~/workspace/csm/moe-exp/T-0001/{models,deployed,tiled}/a{1,2,3}*`
- vivo：`~/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/`
  `experts-t0001-a{1,2,3}/`（symlink + manifest）、`t0001-q2k/`（12 个共享 Q2_K 层）

## 6. 风险与下一步

1. **电池/热降频**：vivo 开跑时仅 5%，第 7 跑起 4% 触发省电降频，基线从 34.2 掉到 22.9；
   有效窗口只有前 6 跑（A1 与基线各 1 跑 ≥30，A2/A3 各 2 跑）。A1 的 34.1 与
   T-0002 同款 36-39 包 8 跑中位 30.05、上一轮队列 T-0001 的 32.8 一致，故 ≥30 结论可交叉验证；
   若要更严格的 4+4 轮 A/B，应在电池 ≥20% 时补测。
2. **A2/A3 的 17 t/s 是设备内存/包大小悬崖**，不是 Q2_K 算力；换更大 RAM 设备或实现
   按窗口流式驻留（减少整包常驻）才可能让 5.3-5.9 GiB 包回到 ≥30。当前不建议部署。
3. 部署 PPL 是 CUDA 上用「手机 dense + 臂专家」合并后测的，HTP 数值路径可能略有差异；
   量化字节相同，排序应稳定。
4. 手机 `SHA256SUMS` 的 manifest 行历史上有旧哈希，本次以实际 `sha256sum` 为准；
   12 层推送后已在手机端 `sha256sum -c` 全过。
5. 未做任务级/中文质量评估；wikitext-2 PPL 只作工程近似。
