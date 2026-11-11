## MobiInfer-Android 0.4.5

MiniCPM-V 4.7 解码明显加速（同机同 prompt：13.3 → 21.8 t/s，bench 口径 21.9 t/s）：

- MoE 专家读盘改为跨层流水：O_DIRECT 直接读 + 提前 2 层提交 gate 预测 + 每专家
  任务队列与常驻读线程，I/O 基本藏到计算下面。
- 准入按物理宽度计算，小窗口设备不再整批掉回 bank 路径（W24/W28 实测 +88%/+96%）。
- 修复未知 DSP 架构被误判为低内存布局，导致 vision 回 CPU、DSpark 关闭的回归。

其他：
- Hexagon 后端同步到上游 2026-11-10（repack/op fusion/FA 长上下文/新算子），
  新增 v85 skel，v73/v75/v79/v81/v85 默认全部随包构建。
- 文生图与 MoE 各自使用自己的 skel，不再互相覆盖。
- 修复退出码 139；新增 `-PskipImageGen` 构建开关，文生图改增量构建。

Faster MiniCPM-V 4.7 decode on-device (13.3 -> 21.8 t/s on the same prompt):
cross-layer expert prefetch (O_DIRECT reads, horizon 2, per-expert task queue),
physical-width admission for small decode windows, and a fix for the low-vmem
misdetection that silently disabled vision-on-HTP and DSpark. Hexagon backend
synced to upstream 2026-11-10, v85 skels ship by default, imagegen and MoE now
use separate skel files, exit-139 fix, plus `-PskipImageGen` for faster builds.
