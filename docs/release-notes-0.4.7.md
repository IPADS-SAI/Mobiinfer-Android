## MobiInfer-Android 0.4.7

Qwen3.5-35B-A3B 多了两个精度档，按显存从紧到松都能挑：

- Qwen3.5-35B-A3B 新增 **Q2_K**（15.1 GB）与 **Q3_K_M**（20.4 GB）两档，加上原有的
  Q4_K_M（25.1 GB），同一套 hexagon-tiled 专家布局（41 层 = 40 个 block + MTP 层），
  只差 dense 与专家的量化档位。两档都用 wikitext2 128-chunk 的 imatrix 加权。
- MiniCPM-V 4.7 的 Q2_K 档补上了配套的 DSpark 草稿文件。此前应用不检查草稿是否存在，
  缺了也显示「已下载」，引擎会静默退回无投机的自回归解码，速度白白少三成。

其他：
- 修复：DFlash 草稿长度维持 5。实测短草稿省下的验证开销抵不过少接受的 token
  （n_max=1/3/5/7 → 17.80/19.27/21.02/19.92 tok/s），默认值不动，仍可用
  `runtime.json` 的 `draft_n_max` 自行覆盖。
- 新增：`runtime.json` 里 `speculative_type: "mtp"` 可用模型自带的 MTP 头做自投机，
  不需要单独的草稿模型。实测在本机是负收益（n_max=1 时 1.04×），保留接口、不作为默认。
- 内存预算说明改为反映实际可调上限与专家窗口饱和点。

Qwen3.5-35B-A3B now ships in three precision tiers, so it fits devices with less
headroom:

- New **Q2_K** (15.1 GB) and **Q3_K_M** (20.4 GB) tiers alongside the existing
  Q4_K_M (25.1 GB). Same hexagon-tiled expert layout (41 layers = 40 blocks + MTP),
  only the quantisation of the dense weights and experts differs. Both new tiers use
  the same wikitext2-128-chunk importance matrix.
- The MiniCPM-V 4.7 Q2_K tier now names its DSpark draft. The app did not check for
  the draft before, so a bundle without it still reported "downloaded" and the engine
  silently fell back to autoregressive decoding.

Other:
- Fixed: the DFlash draft length stays at 5. Shorter drafts save verify work but lose
  more accepted tokens than they save (n_max=1/3/5/7 -> 17.80/19.27/21.02/19.92 tok/s).
  `runtime.json` `draft_n_max` still overrides it.
- New: `speculative_type: "mtp"` in `runtime.json` drafts with the model's own MTP head,
  no separate draft model needed. Measured as a net loss here (1.04x at n_max=1); the
  hook is kept but is not a default.
- The memory-budget description now reflects the real ceiling and the point where the
  expert window saturates.
