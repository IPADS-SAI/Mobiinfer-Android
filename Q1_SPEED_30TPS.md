# Q1_0 专家包：30 t/s 速度档与 weighted Q1_0 部署口径

- 日期：2026-11-22（本修订：T-0049，run `run-20261122T214739-w1-1`）
- 分支：`exp/extreme-quant-30tps`；PR #53
- 状态：**weighted Q1_0 已实现并验证**（T-0022 实现，T-0023 部署 PPL；T-0034 独立复现同一
  `a-w` 包 sha）。Q1_0 不再是“PPL 103.07”的单一档：同一份 4.53 GB 专家包，unweighted 部署
  PPL **33.75**，weighted（wt2 imatrix）**17.23**。
- 一句话边界：**33–35 t/s 只在 vivo + unweighted Q1_0 + N=128 热缓存下成立；weighted 的
  vivo 速度尚未测（pending）；质量仍非 Q3_K 等价，只能按“低内存档/非等价”部署。**

## 1. 包选型（部署口径，wikitext2 128 chunks）

| 包 | 专家规则（first-match） | imatrix | 专家 tiled 字节 | 部署包总字节 | wiki2 PPL | 来源 | 状态 |
|---|---|---|---|---:|---:|---:|---|---|
| `a`（a-now） | 全部 Q1_0 | 无 | 4,529,848,320 B（4.53 GB / 4.22 GiB） | 6,192,590,016 B | 33.7477 | T-0023（T-0033 128c 锚点逐位相同） | 旧基线；T-0011/T-0012 的 33–35 t/s 就是它 |
| **`a-w`** | 全部 Q1_0 | wt2 | 4,529,848,320 B（4.53 GB / 4.22 GiB） | 6,192,590,016 B | **17.2316** | T-0023；T-0034 复现 0.000% | **推荐零字节档**；部署包 sha `128919c4…` |
| **`id-w`** | 38–39 层 Q2_K，其余 Q1_0 | wt2 | 4,831,838,208 B（4.83 GB / 4.50 GiB） | 6,494,579,904 B | **15.5664** | T-0023 | T-0032 nubia 速度未过（19.65 vs 27.55，比值 0.7132）；先不要部署 |
| **`ic-w`** | 36–39 层 Q2_K，其余 Q1_0 | wt2 | 5,133,828,096 B（5.13 GB / 4.78 GiB） | 6,796,569,792 B | **14.5837** | T-0023 | 贴着 4.78 GiB 崖：4.7813 GiB 恰为 T-0025 轻载 ≥30 的实测上限（bracket 4.785–5.34 GiB）；速度待验证，谨慎 |
| `q3k` 参考 | 120 × Q3_K | — | 13,841,203,200 B | 15,503,944,896 B | 6.4794（32c 为 6.3870） | T-0033 受控合并 | 只作参考；不是原模型绝对质量 |

说明：

- 字节以 tiled 专家包与部署 GGUF 的实测值为准（`T-0023/gate/ppl.json`、`T-0024/artifacts/ppl-table.tsv`、
  `T-0033/artifacts/ppl.tsv`）。旧汇总把 4.53/4.83/5.13 写成 GiB，按字节实际是 **GB**
  （分别 4.22/4.50/4.78 GiB）；本表两种单位都列，避免再混。
- PPL 口径：手机 dense `t-0040/model/minicpm47-les-hexagon.gguf` + 120 个专家张量，
  `-c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128`。表中 `a`/`a-w`/`id-w`/`ic-w`
  均来自 T-0023；`q3k` 来自 T-0033 的同 dense 受控合并。
- `a-w` 是**零字节成本**的首选（同字节 33.7477 → 17.2316，−48.94%，95% CI
  [−49.98, −47.88]）；`id-w` 多 0.30 GB 换到 15.5664；`ic-w` 再多 0.30 GB 换到 14.5837，
  但跨过 4.78 GiB 崖，且 T-0032 已显示 +0.3 GB 档在当前 nubia release 上有严重速度回退。
- 旧汇总里的 Q3_K “6.39–6.8”：T-0033 `ppl.tsv` 能复现的是 6.3870（32c）/ 6.4794（128c）；
  **6.8 未在该表复现，标“待验证”**，不要当第二个锚点。

## 2. 历史：103.07 是什么，为什么不能再用

- **103.07 是历史 arm-level 值**：T-0002/T-0011 的 `arm-a.gguf`（Q1_0 专家 +
  `llama-quantize` 默认 Q2_K dense/output）在 A100 上的 PPL，不是手机部署包，也不是
  33.3 t/s 那个端点（T-0011 `notes/claim-audit.md` §0.5、T-0012 runbook §2.3）。
- 33.3 t/s 端点按原张量类型重建的 PPL 是 **34.63**（T-0004，`rebuilt`，非逐字节复现）。
- 现在统一用**部署口径**：unweighted `a` = **33.7477**（T-0023/T-0033），weighted `a-w` =
  **17.2316**（T-0023）。所以 **103.07 只保留为 unweighted/旧包的历史值**，不能再写成
  “当前 Q1_0 包的 PPL”。

## 3. 复现：BF16 + wt2 imatrix + rules → quantize → merge → perplexity

### 3.1 固定身份（先核对，任一不符就停）

| 对象 | 值 | 来源 |
|---|---|---|
| weighted Q1_0 实现 | T-0022 commit `5d3d014b967b09fb0cac89bcfab706d76cc7d70d` | T-0022 / T-0023 `gate/ppl.json` |
| 量化器源码 | `ggml/src/ggml-quants.c` sha256 `3cb38ef3ef619b0db09446f849bc332948ec8d7d17bd630c473649216a154674` | T-0023 `gate/ppl.json` |
| wt2 imatrix | `$HOME/workspace/csm/moe-exp/T-0002/out/imatrix-wt2.gguf`，sha256 `a8418f9e9ac3843b3cf1d72f587b46357ddf740267936d40de2b358c9151be72` | T-0034 `artifacts/imatrix.sha256`（T-0002 生成，184 MB） |
| BF16 源 | `$HOME/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf`，69,375,191,008 B | T-0012 runbook；**sha256 未记录，首次复现请自行登记（待验证）** |
| 手机 dense | `$HOME/workspace/csm/moe-exp/t-0040/model/minicpm47-les-hexagon.gguf`，1,662,732,800 B，sha256 `a1e700ae…f2f83` | T-0011 / T-0012 |
| 语料 | `$HOME/workspace/csm/moe-exp/t-0037/corpus-wikitext2.txt`，1,292,013 B | T-0012 runbook |

以下命令以 `$QUEUE_HOME` 指队列运行根目录（默认 `$HOME/.local/state/dpsk-q2`），
`$HOME` 在 A100 上是 `/home/ma-user`、在手机上是 Termux home；不写死私有绝对路径。

### 3.2 取源码并构建量化器（A100，CPU 路径）

```sh
# yydh：任取一个含 T-0022 的 worktree；队列 slot 示例为 w1
QUEUE_HOME=${QUEUE_HOME:-$HOME/.local/state/dpsk-q2}
REPO=${REPO:-$QUEUE_HOME/wt/w1}
git -C "$REPO" merge-base --is-ancestor 5d3d014b967b09fb0cac89bcfab706d76cc7d70d HEAD
sha256sum "$REPO/ggml/src/ggml-quants.c"   # 必须是 3cb38ef3…
ssh sai-a100 'mkdir -p ~/workspace/csm/moe-exp/t-0049-repro/src'
rsync -a --exclude .git --exclude 'build*' --exclude '*.gguf' \
  "$REPO/" sai-a100:workspace/csm/moe-exp/t-0049-repro/src/
```

```sh
# A100
ssh sai-a100
set -euo pipefail
B=$HOME/workspace/csm/moe-exp
T=$B/t-0049-repro
mkdir -p "$T"/{rules,models,raw,deployed,logs}
cmake -S "$T/src" -B "$T/build-q1w" -G Ninja -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF
cmake --build "$T/build-q1w" --target llama-quantize -j"$(nproc)"
```

### 3.3 规则（`first-match-wins`；具体层必须写在 catch-all 前）

```sh
cat > "$T/rules/q1-all.txt" <<'EOF'
blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=Q1_0
EOF

cat > "$T/rules/q1-l38-39-q2k.txt" <<'EOF'
blk\.(38|39)\.ffn_(gate|up|down)_exps\.weight=Q2_K
blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=Q1_0
EOF

cat > "$T/rules/q1-l36-39-q2k.txt" <<'EOF'
blk\.(36|37|38|39)\.ffn_(gate|up|down)_exps\.weight=Q2_K
blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=Q1_0
EOF
```

### 3.4 量化 → extract/manifest → merge → PPL

```sh
Q=$T/build-q1w/bin/llama-quantize
P=$B/build-cuda/bin/llama-perplexity
SRC=$HOME/moe-work/mixed-quant/minicpm47-bf16-nomtp.gguf
IM=$B/T-0002/out/imatrix-wt2.gguf
CORPUS=$B/t-0037/corpus-wikitext2.txt
DENSE=$B/t-0040/model/minicpm47-les-hexagon.gguf
M=$B/T-0002/merge_deployed.py

sha256sum "$IM"   # 必须是 a8418f9e…

# arm 级：Q6_K 只作量化载体；部署 PPL 的 dense 来自手机文件（见下）
"$Q" --tensor-type-file "$T/rules/q1-all.txt" \
  "$SRC" "$T/models/a-now.gguf" Q6_K 32
"$Q" --imatrix "$IM" --tensor-type-file "$T/rules/q1-all.txt" \
  "$SRC" "$T/models/a-w.gguf" Q6_K 32
"$Q" --imatrix "$IM" --tensor-type-file "$T/rules/q1-l38-39-q2k.txt" \
  "$SRC" "$T/models/id-w.gguf" Q6_K 32
"$Q" --imatrix "$IM" --tensor-type-file "$T/rules/q1-l36-39-q2k.txt" \
  "$SRC" "$T/models/ic-w.gguf" Q6_K 32

for arm in a-now a-w id-w ic-w; do
  python3 "$T/src/tools/moe/experts.py" extract \
    --model "$T/models/$arm.gguf" --out-dir "$T/raw/$arm" --force
  python3 "$M" --dense "$DENSE" --arm "$T/models/$arm.gguf" \
    --experts-manifest "$T/raw/$arm/manifest.json" \
    --out "$T/deployed/$arm.gguf"
  CUDA_VISIBLE_DEVICES=0 "$P" -m "$T/deployed/$arm.gguf" -f "$CORPUS" \
    -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128 \
    2>&1 | tee "$T/logs/ppl-$arm.log"
done
```

期望的 `Final estimate`（128 chunks，部署口径）：`a-now` 33.7477、`a-w` 17.2316、
`id-w` 15.5664、`ic-w` 14.5837（T-0023 `gate/ppl.json`）。`sha256sum "$T/deployed/a-w.gguf"`
应为 `128919c4274ee8d2a50ae79a1872e7049d458300511efa61934991006dbe0dbc`。

> 只复现 `a-w` 时，量化 `a-now` + `a-w` 两臂即可；这两臂是“同字节硬检查”的最小集合，
> `id-w`/`ic-w` 可按需追加。中间 GGUF/raw 很大，PPL 后及时删除。

## 4. 部署前硬检查（缺一不可）

1. `sha256sum "$IM"` = `a8418f9e…`；`sha256sum "$T/src/ggml/src/ggml-quants.c"` = `3cb38ef3…`，
   且 T-0022 commit 在当前分支祖先里。任一不符：**停**。
2. 同一份 dense、同一套规则、同字节的 weighted/unweighted 两包，必须满足：
   **120/120 专家张量 sha256 不同**、张量名/类型/nbytes 布局相同、文件大小相同；
   非专家抽样张量（dense）sha256 相同。T-0023 的 `verify_deployed.py` 可直接跑：

   ```sh
   # 在 yydh 上：把 T-0023 的两个小脚本拷到 A100 的 $T（$T 见 §3.2）
   scp "$QUEUE_HOME/runs/T-0023/run-20261122T201215-w3-2/artifacts/a100/gguf_sha.py" \
       "$QUEUE_HOME/runs/T-0023/run-20261122T201215-w3-2/artifacts/a100/verify_deployed.py" \
       sai-a100:workspace/csm/moe-exp/t-0049-repro/
   # 在 A100 上：
   install -m 755 "$T/gguf_sha.py" /tmp/gguf_sha.py
   python3 "$T/verify_deployed.py" --dir "$T" --pairs a-now:a-w
   ```

   期望：`expert_sha_differ=120`、`expert_sha_identical=0`、`layout_identical=true`、
   `same_file_size=true`、`dense_samples_identical=true`。
3. `a-w` 部署包 sha256 = `128919c4…`（T-0023；T-0034 独立复现同一值）。
4. **若 120 个专家 sha 有任何一个相同，说明实际用了 unweighted / `--imatrix` 没生效：
   停止部署**，先查 `--imatrix` 路径、sha 和 T-0022 源码。

## 5. 速度证据边界（不要把不同设备/不同包混在一起）

| 配置 | decode t/s | 证据 / 状态 |
|---|---:|---|
| vivo V2546A，**unweighted** Q1_0，N=128，W80 D16，**热缓存** | 33.3–35.8（n=10，中位 33.4；原始两次均 33.3） | T-0011 `notes/claim-audit.md`；T-0012 runbook。冷缓存未测 |
| 同上，N=64 | 25.9–29.9（最好 29.9） | T-0011；**低于 30** |
| vivo，**weighted**（a-w / id-w） | **pending（未测）** | 没有数据；不要写成已通过 |
| nubia v79，unweighted Q1_0 基线 | 27.55（27.4–27.7，6 跑） | T-0032；v79、非满血（cpu7_max 2.6496 GHz） |
| nubia v79，`id-w` weighted | 19.65（19.6–19.7）；比值 0.7132（95% CI [0.7127, 0.7138]，n=4） | T-0032；**速度门禁未过，回退约 29%** |
| `ic-w` | 待验证 | 无速度数据 |

- vivo 33–35 t/s 只对应 **unweighted Q1_0 + N=128 + 热缓存**；N=64 只有 29.9，冷缓存没有证据
  （T-0011）。不要把它写成“任何 N/任何缓存都 ≥30”。
- **nubia 27.5 不能拿来宣称 30**，也不能和 vivo 33–35 直接比较：设备/v79 release/CPU cap
  都不同（T-0032）。
- **weighted 的 vivo 速度仍是 pending**；T-0032 只在 nubia 上测了 `id-w`，结果是 19.65 vs 27.55，
  未过门禁。其回退疑似 Q2_K tiled kernel 慢或 4.83 GB 包触发缓存崖，未区分（T-0032 §4.3）。

## 6. 质量警告：Q1_0 是低内存档，不是 Q3_K 等价

- T-0033 的受控 Q3_K 合并口径（同手机 dense，wiki 32c）显示 unweighted Q1_0 相对 Q3_K 的
  PPL 倍数为 **wiki 4.95× / docs 3.44× / code 2.26× / zh 3.85×**，即 **3.4–5.0×**；
  token-level 仅 **6/12 top-1 一致**、|Δlogprob| 中位 **0.4532**、KL 均值 **3.1144 nats**
  （最差 zh-03 KL 18.64）。`+36–39 Q2_K` 把 PPL 降 17–21%，但 top-1 只到 7/12，仍不等价。
- **weighted 质量已复评（T-0051）**：`a-w` 对 Q3_K 的 4 语料 32c PPL 仍是
  **2.563× / 2.067× / 1.543× / 2.330×**（wiki/docs/code/zh），`id-w` 是 2.327× / 1.898× /
  1.450× / 2.130×；12-prompt 上 `a-w` **top-1 7/12**、中位 |Δlogprob| **0.170**、top-5
  **0.583**、KL 均值 **1.245 nats**，`id-w` 6/12、0.322、0.633、**1.154**；对照 unweighted
  a 是 6/12、0.453、0.500、**3.114**。weighted 把 KL 降约 60%（真实分布改善），但 top-1
  只从 6/12→7/12，**仍远不在 KL≤0.02 / top-5≥0.90 的等价门禁内**。详见
  [`exps/q1-weighted-imatrix-20261122/README.md#35-weighted-质量复评t-0051`](exps/q1-weighted-imatrix-20261122/README.md#35-weighted-质量复评t-0051)。
- 即使只看 wiki，按 128c 锚点 weighted `a-w` 仍是 Q3_K 的约 **2.66×**（17.2316/6.4794），
  `id-w` 约 2.40×，`ic-w` 约 2.25×；按 T-0051 的 32c 同表则是 `a-w` **2.563×**、`id-w`
  **2.327×**（两套 chunk 数不可混用）。
- 结论：Q1_0（含 weighted）定位为**低内存/非等价档**；weighted 只能作为显式降级的低内存档，
  默认/高质量档仍是 Q3_K。**全 weighted 在 SPECN=3 下已被 T-0053 证明 acceptance
  0.50–0.56、18–20 t/s，不可部署；X1-w 仍是全 weighted，只有 PPL 结论；下一步是 X1-u
  （无权重 + L2/L6 Q2_K）与 tail4w（只加权 36–39），见 T-0084/T-0085/T-0086。**

## 7. 下一阶段

1. **weighted 质量复评：已完成（T-0051）**。`a-w`/`id-w` 的 4 语料 32c PPL + 12-prompt
   token-level 已落盘：KL 比 unweighted 降约 60%，但 `a-w` top-1 只 6/12→7/12、KL 1.245
   nats，**仍非 Q3_K 等价**（见 §6 与
   [`exps/q1-weighted-imatrix-20261122/README.md#35-weighted-质量复评t-0051`](exps/q1-weighted-imatrix-20261122/README.md#35-weighted-质量复评t-0051)）。
   不要再排“weighted 质量待复评”。
2. **weighted vivo 速度**：N=128 热缓存 + 冷缓存，ABBA ≥4 轮，`ssh vivo` +
   `~/csm/moe-exp/devlock` 串行；当前状态 **pending**。
3. **T-0032 的 29% 不是 Q2_K kernel 慢（T-0054），而是 weighting 改变 acceptance
   （T-0053）**：T-0054 已证明每次验证耗时 136.8→136.0 ms、验证次数 34→48、
   `34/48=0.708≈0.713`，只有 2/40 层 Q2_K，kernel 静态 1.28–2.40×、整体计算约 −6%，
   要解释 0.713 需每层 9.04×，给不出。**T-0053 用 id-now（无权重 + 38–39 Q2_K）
   acceptance 0.902、27.94 t/s vs a-w（全 weighted）0.503、18.38 t/s 证明 Q2_K 位置
   不是原因；weighting 才是**；`SPEC=0` 下 `B/C=0.981`（同 Q2_K 布局、只差
   weighted）、`B/A=1.062`（Q2_K 4.83 GB 甚至比 Q1_0 4.53 GB 略快）。
   **不要再用 T-0032 的 19.65 否定 +0.3 GB 方向**；§5 里“疑似 kernel 慢/缓存崖、未区分”
   的旧表述以本条为准。
4. **X1-w 的 PPL 已完成（14.6058），但它仍是全 weighted；T-0066 是
   weighted-vs-weighted，不构成 30 t/s 闭环。不要把 27.5/0.902 当作 a-w 的基线。**
   X1-w = L2+L6 两层 Q2_K + 其余 weighted Q1_0，tiled 4,831,838,208 B；T-0065 部署
   PPL 14.6058（wiki128）、token-level 7/12、KL 1.1733。按 T-0053 的机制，全 weighted
   预期继承 ~0.50 acceptance、~18–19 t/s，**不能当作 30 t/s 候选**；T-0066 的 a-w 对照
   必须用 a-w 自己的 18.38 t/s、acceptance 0.503，不是 unweighted 基线的
   27.55/0.902。
   **X1-u（无权重 + L2/L6 Q2_K）与 tail4w（0–35 无权重 + 36–39 weighted）仍 pending**：
   PPL/token-level/tiled 由 T-0084/T-0085 产出，nubia 速度/acceptance 由 T-0086 测；
   本文不预写其结果。`id-w` 仍不可部署。
5. **expert-level 分配（T-0041）**：把 +0.3/+0.6 GB 预算落到具体专家而不是整层；
   L2+L6 组合已转 X1-w（T-0065/T-0066），本条保留更细的专家级分配。
6. **provenance 校验**：部署清单记录 imatrix path/sha256、T-0022 commit、
   `ggml-quants.c` sha256、部署包 sha256；`tools/moe/quant_arm.py` 的 imatrix 字段
   （T-0035 实现，T-0045 build gate pass）可替代手写 shell，合并状态以集成分支为准。
7. **旧说法归档**：“Q1_0 + imatrix 只是下一阶段候选”已被 T-0022/T-0023 关闭；
   “Q1_0 PPL 103.07”只作为 unweighted/旧包历史值保留（见 §2）。

## 8. 证据索引

- 实验文档：`exps/q1-weighted-imatrix-20261122/README.md`（T-0023）、
  `exps/q2k-subblock-budget-20261122/README.md`（T-0024）。
- T-0023：`$QUEUE_HOME/runs/T-0023/run-20261122T201215-w3-2/gate/ppl.json`、
  `artifacts/a100/verify_deployed.json`、`artifacts/a100/logs/`。
- T-0024：`$QUEUE_HOME/runs/T-0024/run-20261122T200153-w2-2/artifacts/ppl-table.tsv`。
- T-0033：`$QUEUE_HOME/runs/T-0033/run-20261122T200508-w6-1/notes/quality-report.md`、
  `artifacts/ppl.tsv`、`numeric/summary.json`。
- T-0032：`$QUEUE_HOME/runs/T-0032/run-20261122T195853-w5-2/result.json`、
  `notes/winner-ab.md`、`artifacts/ab.tsv`。
- T-0011/T-0012：`$QUEUE_HOME/runs/T-0011/run-20261122T165027-w6-1/notes/claim-audit.md`、
  `$QUEUE_HOME/runs/T-0012/run-20261122T194008-w3-3/notes/30tps-runbook.md`。
- T-0034（imatrix sha / a-w 复现）：`$QUEUE_HOME/runs/T-0034/run-20261122T211529-w3-1/artifacts/imatrix.sha256`。
- T-0025（4.78 GiB 崖/字节-吞吐模型）：`$QUEUE_HOME/runs/T-0025/run-20261122T171753-w4-1/notes/q1-bytes-cliff.md`。
