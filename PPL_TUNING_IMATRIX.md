# PPL 调优：imatrix × Q1_0 专家 / 混合层（T-0002）

- 日期：2026-11-22
- 分支：`exp/extreme-quant-30tps`（worker `dpsk/w3`）
- A100：sai-a100，8×A100-80GB，`build-cuda`（build 378 @ `6e24e00`）
- 设备：vivo V2546A（SM8850 / HTP v81），`release-v81-sync`，N=128 / W80 / D16 / SPECN=3
- 语料：`t-0037/corpus-wikitext2.txt`（1,292,013 B），PPL 128 chunks × 2048 ctx

## 结论（TL;DR）

1. **`--imatrix` 对 Q1_0 专家是 no-op**：本树的 `quantize_q1_0()` 收到
   `quant_weights` 后直接走无权重路径（`ggml/src/ggml-quants.c:2098-2111`），
   专家张量逐字节相同（3 个抽样张量 sha256 在 no-imatrix / wt2-imatrix /
   calib-imatrix 三臂完全一致；40 层 tiled 包 0-35 层 sha256 也完全一致）。
2. **imatrix 在 Q2_K 专家层上也没有可测收益**：36-39 层 Q2_K 的部署口径 PPL
   为 26.31（无 imatrix）vs 26.46（有 imatrix），差异在噪声内（±0.24）。
3. imatrix 的“大改善”只出现在 **非专家 Q2_K/Q3_K/Q4_K 权重**（attention/dense）上：
   同一批 Q1_0 专家，arm 级 PPL 从 103.07（无 imatrix）降到 40.53（wt2 imatrix）。
   但手机部署的 dense 是固定的 Q4_K/Q6_K（`minicpm47-les-hexagon.gguf`），
   所以这部分收益**不会**落到端侧。
4. **真正可部署的收益来自“混合层”**：把 36-39 层专家升到 Q2_K（Q1_0 其余层不变），
   部署口径 PPL 33.75 → **26.46（−21.6%）**；vivo N=128 decode 中位 **30.05 t/s**
   （8 次，28.3-32.4），同一批基线中位 30.25（8 次）——**速度基本不变（比值 0.993）**，
   绝对中位仍 ≥30。38-39 两层版本部署 PPL 28.97（−14.2%），但同批 t/s 中位 29.5
   （3 次，设备负载漂移下与 36-39 版无显著差异）。
5. 下一步：若要让 imatrix 真正作用于 1-bit 专家，需要实现 **weighted Q1_0 scale**
   （`d = Σw|x| / Σw`，格式不变）；当前 `--imatrix` 对 Q1_0 无任何效果。

## 1. 实验内容

- **假设**：用 imatrix 重新量化 Q1_0 专家可降低 PPL，同时保持 Q1_0 的 1.125 bpw
  与 vivo ≥30 t/s；必要时把最敏感的 36-39 层专家升到 Q2_K。
- **变量**：
  - imatrix：无 / t-0037 wikitext2（`-c 512 --chunks 128`）/ 既有 calib（4.1 MB，128 chunks）
  - 专家类型：全 Q1_0；Q1_0 + 36-39 Q2_K；Q1_0 + 38-39 Q2_K
  - dense/attention：Q2_K（arm 级对照）/ 手机固定 Q4_K+Q6_K（部署口径）
- **对照**：no-imatrix 的 `arm-a`（Q1_0 专家 + Q2_K 其他，历史 PPL 103.07）；
  手机基线包 `experts-q1-30tps`（同一份 Q1_0 专家，历史 33.3 t/s）。
- **量化器事实**：`llama-quantize` 先用 `--imatrix` 算 `quant_weights`，再按
  `--tensor-type-file` 覆盖专家类型；Q1_0 / Q2_0 的 quantize 函数当前**不使用**
  `quant_weights`，Q2_K/Q3_K/Q4_K 使用。

## 2. 复现命令

A100 工作目录：`~/workspace/csm/moe-exp/T-0002/`（脚本见 §4）。

### 2.1 imatrix（BF16 源 + t-0037 语料）

```sh
T=~/workspace/csm/moe-exp/T-0002
B=~/workspace/csm/moe-exp/build-cuda/bin
CUDA_VISIBLE_DEVICES=1 $B/llama-imatrix \
  -m /home/ma-user/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf \
  -f ~/workspace/csm/moe-exp/t-0037/corpus-wikitext2.txt \
  -o $T/out/imatrix-wt2.gguf -c 512 -b 2048 -ub 512 -ngl 99 -t 32 \
  --chunks 128 --no-ppl
# 实测 8m18s，184 MB；日志 logs/imatrix-wt2.log
```

### 2.2 量化（Q1_0 专家 + Q2_K 其他；混合层用规则文件）

```sh
cat > rules/q1-exp.txt <<'EOF'
blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=Q1_0
EOF
# 36-39 层 Q2_K：具体规则放前面（首条匹配生效）
cat > rules/q1-exp-l36-39-q2k.txt <<'EOF'
blk\.(36|37|38|39)\.ffn_(gate|up|down)_exps\.weight=Q2_K
blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=Q1_0
EOF
# 38-39 同理，把 (36|37|38|39) 换成 (38|39)

$B/llama-quantize --imatrix out/imatrix-wt2.gguf \
  --tensor-type-file rules/q1-exp-l36-39-q2k.txt \
  /home/ma-user/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf \
  out/arm-ic.gguf Q2_K 32
```

### 2.3 PPL（arm 级，Q2_K dense）

```sh
CUDA_VISIBLE_DEVICES=1 $B/llama-perplexity -m out/arm-ic.gguf \
  -f corpus-wt2.txt -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128
```

### 2.4 部署口径（手机 dense + 专家）合并后 PPL

手机 dense 是 `t-0040/model/minicpm47-les-hexagon.gguf`（613 张量，Q4_K/Q6_K，
无专家）；用 `merge_deployed.py` 把 arm 的 120 张专家张量按原始量化字节拼回去：

```sh
python3 merge_deployed.py \
  --dense /home/ma-user/workspace/csm/moe-exp/t-0040/model/minicpm47-les-hexagon.gguf \
  --arm out/arm-ic.gguf --experts-manifest raw-ic/manifest.json \
  --out out/deployed-ic.gguf
CUDA_VISIBLE_DEVICES=1 $B/llama-perplexity -m out/deployed-ic.gguf \
  -f corpus-wt2.txt -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128
```

> 直接用 `llama-perplexity --moe-experts` 不行：流式专家路径要求后端有
> alloc-only buft（HTP 有，CUDA 没有），因此用合并后的等价 GGUF 评估。

### 2.5 extract / pack（Hexagon tiled）

```sh
LIB=~/workspace/csm/moe-exp/t-0082/lib
python3 tools/moe/experts.py extract --model out/arm-ic.gguf --out-dir raw-ic --force
LD_LIBRARY_PATH=$LIB python3 tools/moe/pack_hexagon.py \
  --experts raw-ic --out-dir tiled-ic --lib-dir $LIB
sha256sum tiled-ic/layer_*.bin > tiled-ic.sha256
```

### 2.6 vivo 部署与测速

```sh
# 本机中转（sai-a100 → yydh 约 48 MB/s；adb push 约 39 MB/s）
rsync -a sai-a100:.../tiled-ic/ /tmp/t0002-ic/
adb -s 10AG9L1QAA003LR push /tmp/t0002-delta /sdcard/Download/   # 只推 36-39 层 + manifest

# 手机侧：0-35 层与 experts-q1-30tps 逐字节相同，用 symlink 复用（省 4 GB）
cd ~/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon
mkdir experts-q1-imatrix-l36-39
for i in $(seq -w 0 35); do ln -s ../experts-q1-30tps/layer_0$i.bin experts-q1-imatrix-l36-39/layer_0$i.bin; done
cp /sdcard/Download/t0002-delta/layer_03[6-9].bin experts-q1-imatrix-l36-39/
cp /sdcard/Download/t0002-delta/manifest.json experts-q1-imatrix-l36-39/

# A/B：devlock 内 ABBA 交替，N=128/W80/D16/SPECN=3
./devlock -w 1700 -q ./t0002-bench-ab.sh \
  old=.../experts-q1-30tps new=.../experts-q1-imatrix-l36-39 \
  new=.../experts-q1-imatrix-l36-39 old=.../experts-q1-30tps
```

## 3. 结果

### 3.1 A100 arm 级 PPL（Q2_K dense，128 chunks）

| arm | 专家 | imatrix | PPL | 量化体积 |
|---|---|---|---:|---:|
| BF16（T-0037 参考） | bf16 | — | 6.2321 | 64.61 GiB |
| a | Q1_0 全部 | 无 | 103.0652 | 5529.28 MiB（1.34 bpw） |
| ia | Q1_0 全部 | wt2 | 40.5342 | 5529.28 MiB |
| ib | Q1_0 全部 | calib | 41.8985 | 5529.28 MiB |
| ic | Q1_0 + 36-39 Q2_K | wt2 | 31.7171 | 6105.28 MiB（1.48 bpw） |
| icn | Q1_0 + 36-39 Q2_K | 无 | 83.9691 | 6105.28 MiB |
| id | Q1_0 + 38-39 Q2_K | wt2 | 34.7848 | 5817.28 MiB（1.41 bpw） |

注：a/ia/ib 的专家张量逐字节相同；ia/ib 的 PPL 改善全部来自非专家权重。
ic vs icn 的大差异也主要来自 Q2_K dense 权重，而不是 36-39 层专家（见 3.2）。

### 3.2 部署口径 PPL（Q4_K/Q6_K dense + 专家，真实手机组合）

| 专家包 | imatrix | PPL | Δ vs 基线 |
|---|---|---:|---:|
| Q1_0 全部（旧包） | 无 | 33.7477 ±0.323 | — |
| Q1_0 + 36-39 Q2_K | 无 | 26.3097 ±0.239 | **−22.0%** |
| Q1_0 + 36-39 Q2_K | wt2 | 26.4563 ±0.241 | **−21.6%** |
| Q1_0 + 38-39 Q2_K | wt2 | 28.9658 ±0.267 | **−14.2%** |

→ 对 1-bit / 2-bit 专家，imatrix 与 no-imatrix 的部署口径差异 ≤0.6%（噪声内）；
混合层本身的收益是真实的。

### 3.3 vivo N=128 decode t/s（devlock 串行，ABBA）

| 包 | 运行（t/s） | 中位 | 备注 |
|---|---|---:|---|
| experts-q1-30tps（Q1_0） | 32.2 / 29.3 / 34.7 / 30.7 / 31.5 / 29.8 / 29.2 / 28.3 | **30.25** | 前 8 次（ic 对照窗口） |
| experts-q1-imatrix-l36-39（36-39 Q2_K） | 28.8 / 30.2 / 32.4 / 30.4 / 29.9 / 31.4 / 28.7 / 28.3 | **30.05** | 前 8 次，≥30 |
| experts-q1-imatrix-l38-39（38-39 Q2_K） | 29.8 / 29.5 / 28.2 | **29.5** | 3 次，与基线同批 |
| （负载加重后补测）baseline / l36-39 | old 29.7/26.4/28.8；ic 28.7/28.6/27.9 | old 28.8 / ic 28.6 | 见下方漂移说明，不作为验收值 |

> 负载加重后的补测（17:00 左右）中，**基线本身**也跌到 26.4-29.7，ic 27.9-28.7；
> 同一时段 vivo 上有其他 worker 的 `rsync .../experts-a4` 正在写 6 GB 包、
> `/sdcard/Download` 达 31 GB，属设备 I/O 争用。历史轻载同配置基线 33.3 t/s；
> 独立任务 T-0001 的同款 36-39 Q2_K 包（无 imatrix）在轻载下为 **32.8 t/s**
> （基线 33.3，−1.5%），与本次 ic/baseline 比值 0.993 一致。

prefill 同步波动（27-32 t/s），不作为验收指标。包体积：tiled Q1_0 4.53 GiB；
tiled 36-39 版 5.13 GB（+0.6 GB，手机侧新增数据 1.0 GB，其余 symlink）；
tiled 38-39 版 4.83 GB。

**关于绝对 t/s 的漂移**：本次会话内同一条基线包从 34.7 逐步降到 26.4（约 −24%），
同时段设备 load avg 6.5、`/sdcard/Download` 被其他 worker 堆到 31 GB，属队列并发 I/O
争用，不是包本身变化。历史（负载较轻时）同包为 33.3 t/s，T-0001 的 36-39 Q2_K
同款包为 32.8 t/s。因此验收看同批交替的中位与比值；绝对中位 30.05 是在
前 8 轮（争用较轻）窗口内取得，后 3 轮争用加重时基线同样跌到 28.8。

## 4. 证据与产物

运行目录：`/home/yydh/.local/state/dpsk-q2/runs/T-0002/run-20261122T155210-w3-1/`

- `artifacts/a100/logs/`：imatrix / quantize / perplexity / merge / pack 全部原始日志
- `artifacts/a100/expert-tensor-sha256.json`：3 个专家张量在三臂的 sha256（逐字节相同）
- `artifacts/a100/tiled-layer-compare.json`：tiled-ic 40 层中 36 层与 no-imatrix 包相同
- `artifacts/a100/tiled-ic.sha256`、`tiled-ic/manifest.json`
- `artifacts/vivo-bench/summary.txt`：10 次 A/B 原始 t/s
- `artifacts/merge_deployed.py`、`artifacts/a100_quant_ppl.sh`、`artifacts/a100_pack.sh`、
  `artifacts/t0002-bench-ab.sh`
- A100 大文件（不回收）：`~/workspace/csm/moe-exp/T-0002/out/arm-{ia,ib,ic,icn,id}.gguf`、
  `out/deployed-*.gguf`、`tiled-*/`

## 5. 风险与下一步

- **imatrix 对 Q1_0 无效是代码事实**，不是测量误差；`quantize_q1_0()` 的
  `quant_weights` 分支未被使用。若要让 imatrix 真正作用于 1-bit 专家，需实现
  weighted scale（格式不变，HTP 无需改动），再评 PPL/速度。
- 部署口径 PPL 是用合并 GGUF 在 CUDA 上测的；手机 HTP 的数值路径可能略有差异，
  但量化字节相同，排序应稳定。
- 手机侧 0-35 层用 symlink 复用旧包，实测可加载/可跑；正式发布应改为完整拷贝。
- 设备负载高（load avg ~6.5，其他 worker 同时写包），会话内基线漂移 −24%；验收用前 8 轮交替中位，另列后 3 轮作负载敏感性。
- 38-39 两层版本（id）见 §3，若速度余量优先可选它；36-39 版本 PPL 更优。
