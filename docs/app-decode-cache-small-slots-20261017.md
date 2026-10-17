# App 小 decode 槽位测试（2026-10-17）

原一加 13，App 实机测试。固定历史 prompt 512 tokens、输出 128 tokens；batch/ubatch=512，bank=256，prefetch=4，IO=4，DSpark Q8 draft=3，temperature=0、seed=42，双 HTP，非阻塞 DSP 等待，profile=0。APK SHA256：`b8df86d925db1dead4973974b41736b088fd34fb621d89cbfb2daffbdf46d51f`。

顺序为 32 槽两轮 → 40 槽两轮 → 48 槽回测一轮。同一槽位的两轮共用进程、各开新会话；换槽位重启进程。没有主动清页缓存、预热或降温，顺序未随机化，不能把小幅差异归因于槽位本身。

| Decode cache + prefetch | 轮次 | Prefill tok/s | Decode tok/s | DMA 结束采样 GiB | 进程峰值 RSS MiB |
|---|---:|---:|---:|---:|---:|
| 32 + 4 | 1 | 80.90 | 13.32 | 6.855 | 1326.32 |
| 32 + 4 | 2 | 97.09 | 13.94 | 6.855 | 1333.53 |
| 40 + 4 | 1 | 85.57 | 13.49 | 7.324 | 1298.07 |
| 40 + 4 | 2 | 95.52 | 13.52 | 7.324 | 1300.20 |
| 48 + 4 | 1 | 89.55 | 13.72 | 7.793 | 1290.88 |

32 槽平均 decode 13.63 tok/s，40 槽平均 13.51 tok/s，48 槽本次回测 13.72 tok/s。32 槽相对 48 槽减少约 0.94 GiB DMA 分配，尚未显示稳定速度优势；40 槽也没有提速。保留正常默认 cache=48、prefetch=4；需要节省内存时，32 槽值得继续多轮验证。本次未测更小槽位或长上下文，因此不能声称 32 为最低可用值或全场景等速。

五轮均 cache_n=0、prompt_n=512、predicted_n=128，接受草稿 81/136，输出文本 SHA256 全部一致：`84b85854d492a8c22e2a658bd15c0d8891c676188bc5a8d928179aac4aaf6a3b`。

## IO 与内存口径

外部首 token 标记至观察到完成的进程级逻辑读取：32 槽首轮约 16.45 GiB，40 槽两轮约 13.74/13.80 GiB，48 槽约 11.80 GiB。槽位越少逻辑读取越多，但端到端速度还受页缓存等影响，不能据此换算精确专家命中率。32 槽第二轮缺失首 token IO 快照，未填补该数据；完整请求 IO 记录仍保留。

RSS 峰值为内核 VmHWM，同进程第二轮为累计峰值；DMA 为请求结束时采样，非峰值。RSS 与 DMA 可能重叠，不能相加。8GB 是配置预算估计，不是硬内存上限。

## 复现与文件

原始目录：`/home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/`。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/run-cache-slots-small-sweep.py
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/summarize-cache-slots.py
```

脚本调用 `run-app-ab.py`，设置 BENCH_CACHE=32/40/48，BENCH_PREFETCH=4、BENCH_IO=4、BENCH_POLL=1；同进程第二轮使用 BENCH_REUSE_PROCESS=1。每轮 `app-chunk512-cache*-small*-rN` 文件保存配置、结果、runtime/logcat、频率采样、进程 IO 和内存。汇总为 `cache-slots-sweep-summary.json`。

测试完成已删除临时 runtime.json 并重启 App，恢复预算推导的 cache=48、prefetch=4，batch512 保留；测试手机非阻塞 debug 属性保留为 1。未提交或推送。
