# vivo V2546A App 诊断与默认轮询调整

2026-10-17，SM8850，已安装 APK 与一加测试版本相同（旧 SHA256 b8df86d925db1dead4973974b41736b088fd34fb621d89cbfb2daffbdf46d51f）。日志确认 HTP0、v81 skel，cache48/prefetch4/IO4、ubatch512，视觉 HTP、DSpark 开启。未发生 CPU 全模型回退。

用户已有请求日志：1105 输入/940 输出为 prefill32.59、decode7.92 tok/s；16 输入/595 输出为8.71/7.62；图片请求93输入/298输出为25.24/7.25，视觉编码201.52ms。请求内容和长度不同，不能直接对比一加512/128结果。

现场电池45.8°C、thermal status4，skin约47–48°C；CPU policy6 scaling_max_freq1382400，而 cpuinfo_max_freq4608000，policy0上限1555200而硬件3628800。明确存在温控与频率限制，但尚无降温同输入对照，不能量化其耗时占比。设备 MemTotal15596872kB、MemAvailable约3GiB；与一加24GB环境不同。DSP轮询属性为空，因此旧App使用阻塞模式。不能将所有差异归因于等待方式、单HTP或页缓存。

用户要求后，将App默认GGML_HEXAGON_OPPOLL改为1；debug.mobiinfer.htp_poll显式0仍可禁用，重启进程生效。底层Hexagon通用默认保持原有值。默认使用已有专用非阻塞队列路径。assembleDebug离线构建成功；新APK SHA256699f0040c7fd9d133e3cbb7b02867533c7dd9dfeb74657dbe105594ee4a6d4a4。

原始日志、thermal、battery、频率、benchmark与安装包旧hash保存在 /home/yydh/moe-speculative/hexagon-results/vivo-app-20261017/。本次诊断未主动运行新推理、未更改温控设置，尚无新默认下性能结论。


## 新默认实机测试

新APK启动日志明确为 `HTP response wait: poll`，Android调试属性保持为空。固定历史prompt512、输出128，batch/ubatch512，cache48+prefetch4、IO4、DSpark草稿3，profile0，HTP0。新对话、cache_n=0，同进程连续两次请求。未主动清缓存或降温。

| 轮次 | Prefill tok/s | Decode tok/s | 进程累计峰值RSS MiB | 电池温度°C |
|---|---:|---:|---:|---|
| 1 | 81.37 | 11.34 | 1015.21 | 34.9–34.9 |
| 2 | 88.54 | 12.01 | 1015.21 | 34.7–34.7 |

两轮接受82/131草稿，文本输出SHA一致。RSS为进程启动以来VmHWM，两轮累计；本次未获取DMA总量，不能把RSS当总内存。结果仍受温度、CPU限频、页缓存和输入内容影响；与之前不同prompt的7.25不能解释为纯轮询加速比例，尚无同输入blocking对照。

首次自动化误用一加屏幕坐标，没有进入新对话，带入历史记录并观察超时；该次排除。vivo脚本现按btn_new_conversation控件ID点击。有效文件为vivo-default-poll-clean-r1/r2，保存结果、日志、IO、频率采样、状态；运行命令：

```bash
BENCH_LABEL=vivo-default-poll-clean-r1 BENCH_POLL=default python3 /home/yydh/moe-speculative/hexagon-results/vivo-app-20261017/run-app-ab.py
BENCH_LABEL=vivo-default-poll-clean-r2 BENCH_POLL=default BENCH_REUSE_PROCESS=1 BENCH_RUNTIME_LABEL=vivo-default-poll-clean-r1 python3 /home/yydh/moe-speculative/hexagon-results/vivo-app-20261017/run-app-ab.py
```

测试后移除临时runtime.json，恢复正常生成长度并重新启动App。默认轮询开启。未提交或推送。
