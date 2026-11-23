## MobiInfer-Android 0.4.6

Q2 tiled 专家包把 MiniCPM-V 4.7 的解码推到 31.1 tok/s，其他 App 在前台时 29.2 tok/s：

- Q2_K tiled 专家包：路由专家按 Q2_K tiled 存放，读盘量明显低于 Q3 tiled；8 Elite Gen5
  上解码 23.3 → 31.1 tok/s，41-token prompt 下 prefill 24.57 tok/s。
- 可与其他 App 并行推理：其他应用前台运行时解码 29.2 tok/s。
- MoE 运行时同步：窗口淘汰支持 LRU / LFU（`LLAMA_MOE_EVICT`）、共享专家缓存、
  预读合并与读加固、A4 draft-pool 投机解码（opt-in）。

其他：
- Hexagon 后端同步：Q3_K compact slab、小 batch 的 MUL_MAT / MUL_MAT_ID HVX-first 派发、
  SUM_ROWS+CLAMP+DIV 等小算子链融合、opbuf mask/stats、decode 期轮询与 RTT 环。
- Q2_K / Q1_0 量化内核，以及 `tools/moe` 的量化打包与分析工具链。
- 模型列表恢复 Q2_K 专家包条目，已经下载过该包、之前选中过的设备不需要重新下载。

MiniCPM-V 4.7 decodes at 31.1 tok/s with the Q2 tiled expert pack, and at 29.2 tok/s while
other apps are in the foreground:

- Q2 tiled experts: routed experts are stored in Q2_K tiled, well below the read volume of
  Q3 tiled. On an 8 Elite Gen5 decode goes 23.3 -> 31.1 tok/s, with prefill at 24.57 tok/s
  on a 41-token prompt.
- Runs alongside other apps: 29.2 tok/s with other apps in the foreground.
- MoE runtime sync: LRU/LFU window eviction (`LLAMA_MOE_EVICT`), shared expert cache,
  prefetch merging with read hardening, and the A4 draft pool (opt-in).

Other:
- Hexagon backend sync: compact Q3_K slab, HVX-first dispatch for small-batch MUL_MAT /
  MUL_MAT_ID, elementwise chain fusion, opbuf mask/stats, decode-time polling and the RTT ring.
- Q2_K / Q1_0 quantisation kernels, plus the `tools/moe` packing and analysis toolchain.
- The Q2_K expert pack is listed again, so devices that already downloaded and selected it
  do not re-download.
