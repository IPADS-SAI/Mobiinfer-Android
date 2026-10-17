# MiniCPM App 按需加载 mmproj

> 当前默认已改回随模型加载 mmproj。下文保留按需加载方案及其历史实验；实验可用 `runtime.json` 的 `eager_mmproj=false` 复现。

MiniCPM47Runtime 默认启动文本运行时，setVisionPath 只记录文件路径，不把 mmproj 传入初次 load_model。首次 submit 检测到当前或恢复历史中的图片后，检查 mmproj 文件，停止并释放文本 server，再创建带 mmproj 的视觉 server。对话消息与图片字节保留，禁用旧 KV 前缀复用并重新 prefill。后续图片复用已加载 projector，新对话也继续复用，直到模型卸载。

这是 App 层按需切换，不是 server 内热插入。视觉 slots、MRoPE、DSpark 和计算缓冲区由 server 初始化一并确定，因此首次图片需要重新加载 target/draft，产生额外等待；先释放旧运行时，避免同时驻留两份主模型。首次图片加载开销发生在请求计时开始前，现有 benchmark wall/TTFT 不包含这次切换，不能用其数字声称用户端首图等待更短。首次加载失败会记录运行时失败，要求重载模型，避免继续使用不完整 server。没有更改通用server实现。

assembleDebug离线构建成功；未执行新版本文本→图片推理验证，尚无速度或内存收益测量。新行为仅MiniCPM47运行时，其他模型路径不变。未提交推送。


## 按需加载版本：App纯文本实测

一加13 App，英文Explain how rain forms in clear simple English.，22输入/128输出，两次新会话同进程。为对照上一短组使用b512/ub128、bank256、cache48+prefetch4、IO4、draft3、双HTP、轮询1。日志明确mmproj deferred until image input，未出现loaded multimodal；ub128生效。未清页缓存或主动降温。

| 轮次 | Prefill tok/s | Decode tok/s | Max RSS MiB | DMA结束GiB | 电池°C | 草稿接受 |
|---|---:|---:|---:|---:|---|---|
| 1 | 15.02 | 15.38 | 965.82 | 6.233 | 31.5–31.9 | 83/131 |
| 2 | 20.87 | 16.71 | 965.82 | 6.233 | 32.1–32.3 | 83/131 |

两轮输出hash一致。此前加载mmproj短组decode14.54/15.40，本组15.38/16.71；尚无交错且草稿接受不同，不能把差值全部归因mmproj或保证固定提速。RSS峰值同进程累计，DMA结束采样不是峰值，二者不能相加。本次仅验证纯文本，首次图片切换尚未验证。原始app-lazy-mmproj-text-r1/r2文件保存日志、结果、内存、频率温度和IO。测试后删除runtime.json并重启，正常ub512恢复。


## 追加三轮纯文本

配置与前两轮相同。此前已恢复并重启App，因此追加第3轮重新启动ub128测试进程，第4/5轮复用此进程，均新会话、cache_n0、22输入128输出；不是五轮单一连续进程。日志确认mmproj延迟加载，无图片。

| 总轮次 | Prefill tok/s | Decode tok/s | Max RSS MiB | DMA结束GiB | 电池°C |
|---|---:|---:|---:|---:|---|
| 3 | 15.79 | 15.59 | 961.80 | 6.233 | 32.7–32.8 |
| 4 | 21.35 | 16.38 | 961.80 | 6.233 | 32.8–32.9 |
| 5 | 21.71 | 16.56 | 961.80 | 6.233 | 33.0–33.1 |

合计五轮decode平均16.12，中位16.38 tok/s；全部输出hash一致，草稿接受83/131。仍未做加载/不加载mmproj交错对照，不能把速度差全部归因mmproj。追加原始app-lazy-mmproj-text-r3/r4/r5文件和repeat-lazy-text-extra.py保留。结束删除runtime.json并重启恢复默认ub512。


## 默认ub512，纯文本连续五轮

同App未加载mmproj、b512/bank256/cache48+prefetch4/IO4/draft3/双HTP/轮询1。英文22输入、128输出，新会话cache_n0，五轮共用一个进程。runtime.json没有ubatch字段，因此确实使用默认512（启动日志确认），只临时限制输出128并标记实验。未清页缓存或主动降温。

| 轮次 | Prefill tok/s | Decode tok/s | 峰值RSS MiB | DMA结束GiB | 电池°C |
|---|---:|---:|---:|---:|---|
| 1 | 15.93 | 15.54 | 962.62 | 7.206 | 33.5–33.5 |
| 2 | 17.08 | 16.44 | 962.62 | 7.206 | 33.5–33.6 |
| 3 | 19.11 | 15.56 | 962.62 | 7.206 | 33.6–33.7 |
| 4 | 18.08 | 16.34 | 962.62 | 7.206 | 33.7–33.9 |
| 5 | 18.78 | 15.58 | 962.62 | 7.206 | 34.0–34.1 |

Decode平均15.89、中位15.58 tok/s，输出hash一致，草稿均83/131。此前ub128五轮平均16.12，但两组非交错、进程分组也不同，不能据小幅差异认定ub影响。RSS为同进程累计峰值，DMA为结束采样、非峰值，不能相加。测试后删除runtime.json并重启，保留正常默认。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/repeat-default-ub512-short.py
```

原始同目录app-lazy-default-ub512-r1..r5保存结果、logcat/runtime、IO、配置、频率温度和内存。


## 启动时加载mmproj，默认ub512，五轮对照

新增仅供本地对照的runtime.json eager_mmproj=true；正常默认仍按需加载。重新构建debug并安装成功，运行时日志确认loaded multimodal model，文件mmproj-MiniCPM-V-4_7-real-f16-pad4320.gguf。固定同短英文22输入/128输出，b/ub512、bank256、cache48+prefetch4、IO4、draft3、双HTP、轮询1。五轮同进程各新会话，cache_n0，无主动清缓存或降温。

| 轮次 | Prefill tok/s | Decode tok/s | 峰值RSS MiB | DMA结束GiB | 电池°C |
|---|---:|---:|---:|---:|---|
| 1 | 15.82 | 15.89 | 974.14 | 8.251 | 34.0–34.2 |
| 2 | 16.95 | 15.97 | 974.14 | 8.251 | 34.2–34.2 |
| 3 | 17.47 | 16.02 | 974.14 | 8.251 | 34.3–34.4 |
| 4 | 17.42 | 15.84 | 974.14 | 8.251 | 34.4–34.6 |
| 5 | 17.36 | 15.84 | 974.14 | 8.251 | 34.6–34.8 |

Decode平均15.91、中位15.89 tok/s。未加载mmproj此前平均15.89，本组15.91，约0.1%差异，没有测出明显decode性能差距。两组非交错、重新构建APK，草稿接受本组85/126而未加载83/131，输出/计算轨迹并不完全相同，不能据此认定严格等效或mmproj无任何成本。RSS为同进程累计峰值，DMA结束采样非峰值，两者可能重叠不能相加。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/repeat-eager-ub512-short.py
```

原始app-eager-default-ub512-r1..r5文件保存配置、安装包hash、结果、日志、IO、频率温度和内存。测试后删除runtime.json并重启，恢复正常按需加载mmproj与输出上限；未提交推送。


## App关闭预取：短输入默认ub512五轮

同一句短英文22输入/128输出，mmproj未加载，b/ub512、bank256、IO4、draft3、双HTP、轮询1。临时prefetch0，预算归还4槽使cache52，总窗口仍52。日志确认cache52/prefetch0且lookahead off。五轮同进程新会话cache_n0，无主动清缓存/降温。

| 轮次 | Prefill tok/s | Decode tok/s | 峰值RSS MiB | DMA结束GiB | 电池°C |
|---|---:|---:|---:|---:|---|
| 1 | 14.38 | 15.19 | 960.54 | 7.206 | 31.5–31.6 |
| 2 | 15.27 | 15.12 | 1020.85 | 7.206 | 31.6–31.8 |
| 3 | 15.46 | 14.99 | 1027.28 | 7.206 | 31.8–32.0 |
| 4 | 15.09 | 15.18 | 1032.69 | 7.206 | 32.1–32.2 |
| 5 | 15.30 | 15.16 | 1032.69 | 7.206 | 32.4–32.5 |

Prefill平均15.10、decode平均15.13、中位15.16 tok/s。此前预取开decode15.89，本组未显示关闭提速，顺序非交错仍不能将差值全部归因预取。回复组内一致，草稿83/131。RSS同进程累计峰值，DMA结束采样不是峰值，不能相加。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/repeat-noprefetch-ub512-short.py
```

原始app-lazy-noprefetch-ub512-r1..r5文件保存全部结果、配置、安装包hash、日志、IO、内存、频率温度。测试后删除runtime.json重启恢复默认cache48+prefetch4，未改正常默认预取策略。未提交推送。
