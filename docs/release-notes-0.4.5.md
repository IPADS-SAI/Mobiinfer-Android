## MobiInfer-Android 0.4.5

MiniCPM-V 4.7 MoE 解码峰值 23.3 tok/s，并且可以和其他 App 同时跑：

- 跨层专家流水：O_DIRECT 直接读 + 提前 2 层提交 gate 预测 + 每专家任务队列与常驻读线程，
  I/O 基本藏到计算下面。
- 可与其它 App 并行推理：不要求独占设备，其他应用在前台运行时仍可持续生成。
- 准入按物理宽度计算，小窗口设备不再整批掉回 bank 路径（W24/W28 实测 +88% / +96%）。
- 修复未知 DSP 架构被误判为低内存布局、导致 vision 回 CPU 与 DSpark 关闭的回归。

其他：
- Hexagon 后端同步到上游 2026-11-10（repack / op fusion / 长上下文 FA / 新算子）。
- v73 / v75 / v79 / v81 / v85 全部随包构建；文生图也有自己的 v85 skel。
- 文生图与 MoE 各自使用自己的 skel，不再互相覆盖。
- 修复退出码 139；新增 `-PskipImageGen` 构建开关，文生图改为增量构建。

MiniCPM-V 4.7 MoE decode peaks at 23.3 tok/s and keeps running while other apps are
in the foreground:

- Cross-layer expert pipeline: O_DIRECT reads, gate predictions issued two layers
  ahead, one FIFO task per expert with resident reader threads -- I/O hides under compute.
- Runs alongside other apps; no exclusive device access needed.
- Physical-width admission, so small decode windows no longer fall back to the bank path.
- Fixed a low-vmem misdetection that forced vision onto the CPU and disabled DSpark.
- Hexagon backend synced to upstream 2026-11-10 (repack, op fusion, long-context FA).
- v73 / v75 / v79 / v81 / v85 skels ship by default, and text-to-image now has its own
  v85 skel. Exit-139 fix, plus `-PskipImageGen` for faster builds.
