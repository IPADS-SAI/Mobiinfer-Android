# Qwen3-VL-4B 手机测速（2026-10-16）

OnePlus 13，HTP0+HTP1；手机已有 Qwen3-VL-4B-Instruct-Q4_0.gguf（约2.3GiB）。
纯文本测试，无视觉投影、无图片编码；未使用DSpark，不能称为视觉端到端性能。
当前维护源码CLI/bench与coherent库，b/ub128、CPU4线程、ngl99、layer split 1:1。

## llama-bench 三次重复

独立pp512和tg128，非512上下文之后连续decode的组合请求。每次有基准内部预热。

| 测试 | 平均tokens/s | 标准差 | 三次tokens/s |
| --- | --- | --- | --- |
| pp512 / tg0 | 350.08 | 0.37 | [350.198, 349.666, 350.369] |
| pp0 / tg128 | 17.23 | 0.35 | [17.5027, 16.8374, 17.3587] |
| pp512 / tg0 | 348.24 | 0.49 | [347.779, 348.198, 348.754] |
| pp0 / tg128 | 17.46 | 0.02 | [17.4788, 17.4528, 17.4378] |
| pp512 / tg0 | 340.27 | 2.95 | [340.003, 343.342, 337.456] |
| pp0 / tg128 | 17.61 | 0.37 | [17.1867, 17.7336, 17.8951] |
| pp512 / tg0 | 345.37 | 0.55 | [344.801, 345.909, 345.387] |
| pp0 / tg128 | 17.58 | 0.04 | [17.6254, 17.5715, 17.537] |

## 中文真实请求

复用MoE实验的中文输入文件，但Qwen的tokenizer和chat template不同，实际547输入tokens、128输出tokens，非精确512。context2048、temperature0、seed42、关闭warmup。

| 运行 | Prefill tokens/s | Decode tokens/s | 原生日志prompt+eval秒 |
| --- | --- | --- | --- |
| cli-0 | 855.38 | 18.32 | 7.628 |
| cli-1 | 857.16 | 18.05 | 7.730 |
| cli-2 | 852.95 | 18.13 | 7.700 |

CLI prefill计时口径与llama-bench明显不同，尚未隔离差异，不能把CLI的较高prefill数值当作同口径超过bench的证据；不含模型加载时间。
已检查首轮输出为正常中文计划，输出上限128导致回答被截断，并非完整回答。没有找到配套mmproj，尚未测图片理解耗时。

## 复现与日志

完整真实生成脚本：`/home/yydh/moe-speculative/hexagon-results/qwen3-vl-4b-20261016/run-cli.sh`。
手机同名目录：`~/csm/moe/experiments/qwen3-vl-4b-20261016/`。
依赖核心库目录 `~/csm/moe/experiments/prefetch-budget-20261016/lib`。

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- ssh oneplus13-reverse \
  'bash ~/csm/moe/experiments/qwen3-vl-4b-20261016/run-cli.sh'
```

基准参数：

```text
llama-bench -m ~/llama.cpp/models/Qwen3-VL-4B-Instruct-Q4_0.gguf -p 512 -n 128 -r 3 -b 128 -ub 128 -t 4 -ngl 99 --device HTP0,HTP1 --split-mode layer --tensor-split 1,1 -o json
```

本地原始数据 `bench.json/bench.stderr/bench.exit`、`cli-*.stdout/stderr/exit`、`cli-summary.json` 均在上述本地实验目录。没有修改模型、App或默认配置。


## 单 HTP0 对照

单HTP使用 `GGML_HEXAGON_DEVICES=1 --device HTP0 --split-mode none`，其余权重、
batch128、ubatch128、CPU4线程等同前。日志确认ndev1，设备仅HTP0；模型完整加载、三次基准和三次中文生成均退出0。

| 配置 | bench pp512 平均tokens/s | bench tg128 平均tokens/s | 中文547输入/128输出 decode中位tokens/s |
| --- | --- | --- | --- |
| 双HTP | 340.27 | 17.58 | 18.13 |
| 单HTP0 | 349.70 | 18.17 | 18.71 |

本次单HTP的bench prefill约快2.8%、decode约快3.3%；中文decode约快3.2%。
两组按时间先后测量，未交错恒温复测，不能把全部差异确认为会话切分造成。
该Q4_0模型可以单HTP推理，当前结果没有使用双HTP的性能理由；仍未测图片输入。

单HTP中文CLI三次（prefill口径与bench差异仍未隔离）：

| 运行 | Prefill tokens/s | Decode tokens/s | prompt+eval秒 |
| --- | --- | --- | --- |
| single-cli-0 | 878.33 | 18.71 | 7.463 |
| single-cli-1 | 884.20 | 18.83 | 7.416 |
| single-cli-2 | 881.19 | 18.59 | 7.506 |

复现脚本本地实验目录 `run-single.sh`；手机同名目录可运行：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- ssh oneplus13-reverse \
  'bash ~/csm/moe/experiments/qwen3-vl-4b-20261016/run-single.sh'
```

原始数据：`single-bench.json/stderr/exit`、`single-cli-*.stdout/stderr/exit`、
`single-cli-summary.json`，均保存在同一本地实验目录。脚本复跑覆盖同名输出，应先归档。


## 默认batch2048 / ubatch512，单HTP0

保持相同模型、单HTP和其他参数，仅将128/128改为CLI默认的2048/512。为使bench与CLI一致，两者均显式传入默认数值。

| 测试 | Prefill tokens/s | Decode tokens/s |
| --- | --- | --- |
| bench pp512/tg128，三次平均 | 378.17 | 18.33 |
| 中文547输入/128输出，三次中位 | 1013.74 | 18.71 |

中文各次原生日志：

| 运行 | Prefill tokens/s | Decode tokens/s | prompt+eval秒 |
| --- | --- | --- | --- |
| default-cli-0 | 1016.62 | 18.78 | 7.353 |
| default-cli-1 | 1013.66 | 18.71 | 7.380 |
| default-cli-2 | 1013.74 | 18.71 | 7.383 |

仍未测图片输入。bench与CLI计时差异尚未隔离，不应混用。原始default-bench.*、default-cli-*与run-default.sh保存在同一本地实验目录；手机运行run-default.sh可复现（会覆盖同名输出）。两种batch配置未交错恒温重复，性能差异需按此限制解读。
