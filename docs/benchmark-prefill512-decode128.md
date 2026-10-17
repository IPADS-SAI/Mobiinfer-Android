# 512-token Prefill / 128-token Decode 实机数据

日期：2026-10-16。设备：OnePlus 13，HTP0 + HTP1。以下均为真实手机 CLI 推理，非 A100 或模拟器数据。原生日志确认实际输入512 tokens、生成128 tokens。

## 最近一轮：预取数量 8 / 10 / 14 / 16

每组3次，表内取各指标中位数。总推理时间是 prompt eval + eval，不含模型加载，不等同 App 点击到回复结束的耗时。

| 每次最多预取专家 | Prefill tokens/s | Decode tokens/s | 总推理秒 | 逻辑专家读取 GiB | 三次 Decode tokens/s |
| --- | --- | --- | --- | --- | --- |
| 8 | 34.18 | 11.80 | 25.833 | 55.45 | 11.80, 12.10, 11.74 |
| 10 | 33.35 | 11.77 | 26.337 | 57.34 | 11.77, 11.88, 11.65 |
| 14 | 33.28 | 11.60 | 26.467 | 60.02 | 11.60, 11.78, 11.54 |
| 16 | 32.97 | 11.43 | 26.706 | 61.16 | 11.47, 11.43, 11.31 |

固定条件：

- Target：`models/minicpm47-les-hexagon.gguf`；专家：`experts-htp-tiled`。
- DSpark：`models/dspark-MiniCPM-V-4_7-q8_0.gguf`，draft长度 **3**。
- context2048，batch/ubatch128，prefill bank256，CPU线程4，I/O线程4。
- 预测候选depth16，每层36个需求缓存槽 + 16个预取槽，总52槽；只调整实际读取上限。
- 最终prefill chunk最近32行专家预热开启，phase-reuse关闭。
- 纯文本，无mmproj；temperature0、seed42；吞吐测试关闭CPU计时和DSP profile。
- 输入为中文对话背景及次日计划请求，完整格式化输入见实验目录 `stage/prompt-512.txt`。

执行顺序：8/10/14/16 → 16/14/10/8 → 8/10/14/16。12次均正常退出，生成正文SHA256一致：
`a987367b2f85ca33bb2469cf9966c0320b223b0a89f8d747d2a432d06d3671f0`。

本组16相比8的decode中位慢约3.1%，逻辑读取量增加约10.3%，不支持在此配置下加到14/16。
逻辑读取量包含初始化、prefill和decode，并非物理UFS流量。温度31.9–39.7°C；未锁频恒温，也未验证整个进程6GB内存上限。

## 较早一轮：固定预测depth8，扫描实际读取数量

同样512输入/128输出、draft3，每组3次；预测开启组每层44缓存槽+8预取槽。
关闭组使用52需求缓存槽，总物理槽数量相同。其余主要参数同上。

| 每次最多预取专家 | Prefill tokens/s | Decode tokens/s | 总推理秒 |
| --- | --- | --- | --- |
| 关闭整个lookahead | 59.71 | 13.52 | 18.044 |
| 0 | 59.63 | 13.42 | 18.141 |
| 1 | 59.26 | 14.04 | 17.758 |
| 2 | 63.14 | 14.06 | 17.215 |
| 4 | 64.38 | 14.38 | 16.857 |
| 8 | 66.22 | 14.64 | 16.466 |

读取0仍计算预测并用于淘汰提示，与关闭lookahead不同。这一轮温度约36–42°C。
最近一轮整体速度明显较低，原因尚未隔离；两轮的预测候选、缓存分配和设备状态不同，不能跨表直接归因于预取上限。
上述值不是App端到端速度，也不是draft8/16结果；增大DSpark draft尚未开展本组对照。

## 复现命令

手机需已有私有CLI/共享库/HTP skel及权重。完整脚本保存实际环境和CLI参数。

单次最近配置（预取上限可改10/14/16）：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- ssh oneplus13-reverse \
  'BENCH_LENGTH=512 BENCH_CACHE=36 BENCH_DEPTH=16 BENCH_PREFETCH=1 BENCH_READ=8 BENCH_TIMING=0 BENCH_PROFILE=0 BENCH_OUTPUT=128 bash ~/csm/moe/experiments/prefetch-budget-20261016/run.sh'
```

完整三轮矩阵及额外CPU计时：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- ssh oneplus13-reverse \
  'bash ~/csm/moe/experiments/prefetch-budget-20261016/high-sweep.sh'
```

复跑会覆盖该脚本同名日志，需先归档已有结果或修改输出目录。

## 原始数据

本地：`/home/yydh/moe-speculative/hexagon-results/prefetch-budget-20261016/`。

- `high-{0,1,2}-{8,10,14,16}.stderr/stdout`：最近12次原始日志和回复。
- 同名 `.exit`、`.battery-before`、`.battery-after`：退出码与电池状态。
- `sweep-*.stderr/stdout`：较早depth8扫描。
- `summary.json`、`analyze_sweep.py`：逐次数据、汇总及计算脚本。
- `stage/run.sh`、`stage/high-sweep.sh`、`stage/prompt-512.txt`：复现参数、扫描与实际输入。
- `high-timing-*.json`、`timing-summary.json`：另行采集的预取任务计时，不用于吞吐表。

详细背景见 [性能复现记录](on-device-decode-reproduction.md)。未修改App默认设置，未安装新APK。
