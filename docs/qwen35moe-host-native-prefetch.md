# Qwen3.5-35B-A3B 原生预取验证（host，2026-10-17）

验证目标：确认 `Qwen3.5-35B-A3B-Q4_K_M.gguf`（arch=`qwen35moe`）能用与 MiniCPM-V 4.7
text 塔**同一套原生 lookahead 预取**（`src/models/qwen35moe.cpp` 的
`prefetch_router_logits`/`prefetch_topk`，`src/llama-moe-slot.cpp` 的
`lookahead_expert_prefetch`），无需另训预测器。

结论：**能用**。在 sai-a100 的 CPU host 上用 experts 包跑通，日志显示
`lookahead-prefetch: on depth=4`，decode 每步有 `pfread/pfpub/pfhit`，收尾
`lookahead-prefetch: armed=805 hits=1814 (56.3% of predictions)`，且输出正常。

## 重要：用 `llama-completion`，不要用 `llama-cli`

本仓库的 `llama-cli` 走 server/interactive 路径，**会进入 chat 模式**：即使给了 `-p`
和 `-n`，生成结束后仍反复打印 `>` 提示符，日志被刷成几千万行且进程不退出（
`-no-cnv`、stdin 接 `/dev/null` 都不能避免）。做单次 completion / 预取验证请用
**`llama-completion`**，它跑完即退出、stdout 是干净的生成文本，stderr 是 moe 统计。

## 复现步骤（host CPU）

1. 抽 experts 包（arch 无关，按 `ffn_*_exps` 名字匹配）：

```bash
MODEL=/temp/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf
python /home/ma-user/workspace/csm/llama.cpp-omni-hexagon/tools/moe/experts.py extract \
  --model "$MODEL" --out-dir ~/csm/moe-task/qwen35-native-prefetch/experts --force
```

2. 构建 CPU host（含当前 source）：

```bash
cd /home/ma-user/workspace/csm/llama.cpp-omni-hexagon
cmake --build build -j"$(nproc)" --target llama-completion
```

3. 运行（native lookahead 通过环境变量开启）：

```bash
cd /home/ma-user/workspace/csm/llama.cpp-omni-hexagon
export LD_LIBRARY_PATH="$PWD/build/bin"
export LLAMA_MOE_LOOKAHEAD_PREFETCH=1
export LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=4
export LLAMA_MOE_PREFETCH_MAX_ROWS=8
exec ./build/bin/llama-completion \
  -m "$MODEL" \
  --moe-experts ~/csm/moe-task/qwen35-native-prefetch/experts \
  --moe-slot-bank 64 \
  -ngl 0 --no-mmap -nr -c 512 -b 512 -ub 8 -t 64 --temp 0 -n 24 \
  -p "Explain how rain forms in clear simple English." < /dev/null
```

## 为什么必须 `-nr --no-mmap`（CPU host）

- MoE 专家张量必须落在 **host-addressable** 的缓冲上（`llama-moe-slot.cpp` 的
  `preadv loading requires host-visible slots` 检查）。
- `alloc_only_buft`（`make_alloc_only_buft`）目前只有 **Metal / Hexagon** 后端提供；
  CPU 后端没有，于是 `create_tensor_own_alloc` 回退到普通分配，Q4_K/Q5_K 专家会被
  装进 CPU 的 **repack extra buffer**（`is_host_addressable` 为空 → 判定失败）。
- `-nr`（`--no-repack`）关掉 repack，专家回到普通 host buffer；`--no-mmap` 避免被文件
  映射包成 view。两者都只是 **host 验证的 workaround**，不改设备路径。
- `--moe-slot-bank` 必须 ≥ 单图唯一专家数：`-ub 8`、top-8 时上界 64，故取 64。

## 其他边角

- 该 GGUF 的专家几何与 MiniCPM text 塔一致：40 层、256 专家、top-8、专家 2048×512；
  `ffn_gate_exps`/`ffn_up_exps`=Q4_K，`ffn_down_exps`=**Q5_K**。raw preadv 路径不挑量化
  档位，Q5_K 直接可用；若走手机 HTP 的 tiled pack，需要在 host-pack 生成器里补 Q5_K
  （`wrap repack_q5_K_tiled` 已在 `ggml/src/ggml-hexagon/ggml-hexagon.cpp` 存在）。
- `llama-model.cpp` 的专家卸载分支**硬编码只允许 qwen35moe**，本模型正好命中。
- 无 MTP/nextn 层（`nextn_predict_layers` 缺失），无额外复杂度。

## Tiled experts 导出（hexagon-tiled-v1，仍 Q4_K_M）

目标：给手机 HTP 生成 `experts-htp-tiled`（与 MiniCPM 同一 `hexagon-tiled-v1` 布局）。

1. 扩 host 侧 packer：现有 `host-pack.cpp` 只含 Q3_K/Q4_K/Q6_K。Qwen3.5 Q4_K_M 的
   `ffn_down_exps` 是 **Q5_K**，所以补上 `#define HTP_MM_WEIGHT_TILE_SIZE_Q5_K 768`、
   `repack_q5_K_tiled`（逐字取自 `ggml/src/ggml-hexagon/ggml-hexagon.cpp`）、并让
   `pack_expert` 支持 `type==13`。产物 `host-pack-q5k.cpp` / `libhost-pack-q5k.so`
   （`-DNDEBUG -fPIC -shared`）。

2. 打包脚本 `pack-htp-qwen35.py`：按 `experts/manifest.json` 的 `repacked_offset`
   顺序（gate→up→down）、逐专家调用 `pack_expert(type, ne0, ne1)` 写 tiled 文件。
   每层 stride = Σ pack 字节；本模型每专家 2,097,152 B（gate 655360 + up 655360 +
   down 786432），每层 512 MiB，40 层共 **21 GiB**，输出在
   `~/csm/moe-task/qwen35-native-prefetch/experts-htp-tiled/`。

```bash
# 运行
/opt/conda/bin/python ~/csm/moe-task/qwen35-native-prefetch/pack-htp-qwen35.py
```

3. 正确性：用新 lib 复现已知-good 的 MiniCPM Q3_K tiled 包，逐字节 SHA256 一致
   （layer0 `39bedf6999…`，与手机 `modelscope-experts-pack.log` 记录相同），说明抽取
   与尺寸公式正确；Q4_K/Q6_K 路径未改动，Q5_K 为后端逐字拷贝。

手机侧使用（与 MiniCPM 相同）：
```
--moe-experts experts            # 原始 manifest（无 backend_layout）
LLAMA_MOE_HTP_PACKED_DIR=$PWD/experts-htp-tiled
```

## DFlash 草稿：转换 + A100 接受率（2026-10-17）

下载 `z-lab/Qwen3.5-35B-A3B-DFlash`（hf-mirror，736MiB，sha256 校验通过），转 GGUF Q8_0，
与 Qwen3.5-35B-A3B-Q4_K_M 全 GPU 配对测接受率。

转换要点（本仓库无 DFlash 的 python 转换器，自写 `convert_dflash_to_gguf.py`）：
- arch=`dflash`；KV：block_count/context_length/embedding_length/feed_forward_length、
  attention.head_count=32/head_count_kv=8/key_length=128/value_length=128、
  rope.freq_base=1e7、layer_norm_rms_eps=1e-6、`dflash.block_size=16`、
  `dflash.target_layers=[1,6,11,16,22,27,32,37]`。
- **必须**加 `tokenizer.ggml.mask_token_id`（=config.mask_token_id=248077）。缺了它
  `llama_vocab_mask` 返回 -1，草稿 batch 会带 token -1 → `invalid token[1] = -1` 满天飞、
  接受率归零。tokenizer.* 从 target GGUF 原样拷贝（同一 tokenizer）。
- 张量**不转置**：GGUF ne 顺序 = gguf-py 直接写 numpy，HF `nn.Linear` 的 [out,in] 原样即可
  （与 Qwen3.6 DSpark 参考 GGUF 一致）。fc/attn_*/ffn_*/norm 名字见脚本。
- 未写 sliding window（与可用的 Qwen3.6 DSpark 参考 GGUF 一致；加 SWA 反而触发 iswa 崩溃）。

运行（`llama-server` 在本仓库不存在；用 `llama-cli`，它走 server-context 会设 ctx_other）：
```
llama-cli -m Qwen3.5-35B-A3B-Q4_K_M.gguf -md Qwen3.5-35B-A3B-DFlash-Q8_0.gguf \
  --spec-type draft-dflash --spec-draft-n-max <N> \
  -ngl 99 -ngld 99 -c 2048 -b 512 -ub 512 -t 64 --temp 0 -n 256 --fit off -lv 3 \
  -p "Explain how rain forms in clear simple English."
```

接受率结果（贪心，~256 输出，单条短样本，`draft acceptance = acc/gen`）：

| n_max | 每 token 接受率 | 生成 tok/s | 相对无投机 |
| --- | --- | --- | --- |
| 无投机 | — | 126.8 | 1.00× |
| 3 | 64.9% | 164.8 | 1.30× |
| 5 | 55.5% | **174.5** | **1.38×** |
| 7 | 43.3% | 161.6 | 1.27× |
| 15 | 20.5% | 92.8 | 0.73× |

结论：DFlash 可用，`n_max=5` 最优（≈1.4×）；每次 verify 实际稳收 ~3 个 token，所以
把 `n_max` 放到 15 只会浪费算力反而变慢。接受率偏低的一部分原因：target 是 Q4_K_M 量化、
且草稿是对 BF16 target 特征训练的。

## 备注

- 上线（手机）前，远端工作树里 `src/llama-moe-slot.cpp` 曾有一处重复定义
  （`prefill_warm_rows`/`warm_from_prefill` 被加了第二份）导致编译失败，已 `git
  checkout` 回退；HEAD 本身已含该特性。
- 上述 host CPU 跑只为**功能验证**，不是性能证据；真机 HTP 路径需另做 tiled pack 与部署。
