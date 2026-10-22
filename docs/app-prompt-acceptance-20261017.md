# 一加13 App：不同prompt的DSpark接受率

2026-10-17，App默认b/ub512、bank256、cache48+prefetch4、IO4、Q8 DSpark draft3、双HTP、非阻塞等待、profile0、temperature0/seed42，不加载mmproj。8组各两轮：同组同进程新会话，组间重启，不清页缓存或主动降温。输出上限128；中文降雨自然结束102，其他均128。cache_n均0。接受率=accepted/generated草稿token，不等于输出token占比。Prefill/decode为两轮算术平均。

| 组 | 输入tokens | 输出tokens | 草稿接受率 | Prefill平均 tok/s | Decode平均 tok/s | Decode两轮 |
|---|---:|---:|---:|---:|---:|---|
| rain | 22 | 128 | 63.4% | 14.93 | 15.42 | 15.45 / 15.39 |
| count | 36 | 128 | 100.0% | 20.03 | 19.33 | 19.27 / 19.39 |
| repeat | 40 | 128 | 100.0% | 20.23 | 20.41 | 20.46 / 20.36 |
| python | 34 | 128 | 80.2% | 20.57 | 16.53 | 16.45 / 16.62 |
| chinese | 34 | 102 | 38.3% | 19.44 | 11.85 | 11.84 / 11.85 |
| plan-cn | 48 | 128 | 58.8% | 24.78 | 14.43 | 14.21 / 14.65 |
| explain-cn | 41 | 128 | 32.0% | 22.00 | 10.97 | 10.92 / 11.02 |
| steps-en | 46 | 128 | 57.6% | 25.07 | 14.54 | 14.50 / 14.57 |

## 实际提示

### rain

Explain how rain forms in clear simple English.

### count

Write the integers from 1 to 100 in ascending order, separated by commas. Output only the numbers.

### repeat

Write the sentence "The sun rises in the east." exactly 30 times, one sentence per line. Do not add anything else.

### python

Write a Python function that returns the Fibonacci sequence up to n terms. Include clear comments and an example call.

### chinese

请用简单清楚的中文解释雨是怎样形成的，按水分蒸发、凝结成云、降雨三个步骤说明。

### plan-cn

请为一个平时工作忙、每天只有三十分钟空闲的上班族，制定简单可行的一周运动计划。以快走和徒手练习为主，说明怎样循序渐进，语言简洁。

### explain-cn

请用初中生能理解的中文解释：为什么夏天装冷水的杯子外面会出现水珠？说明水从哪里来，并举一个类似的生活现象。

### steps-en

Explain how to grow a bean plant in a small pot at home. Give simple steps for planting, watering, sunlight, and daily care. Use clear everyday English.

## 结论与限制

计数/机械重复接受100%、decode约19–20，只作容易预测的对照，不代表实用任务。实用组Python代码接受80.2%、decode16.53最高，其次英文降雨约63.4%/15.42；英文种豆57.6%/14.54、中文运动计划58.8%/14.43，中文概念解释较低。只有少量提示、各两轮，不可外推“所有中文低于英文”或“简单一定更快”。未关闭DSpark做同提示对照，因此不能给出投机解码相对普通解码加速比。不同输出内容、实际长度、缓存/温度/频率影响耗时。每轮内存、IO、频率温度保留；RSS同进程累计峰值，DMA结束采样非峰值，不能相加。

## 复现

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/run-prompt-acceptance.py
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/run-practical-prompts.py
```

同目录app-prompt-{case}-r1/r2保存结果、配置、logcat/runtime、安装包hash、IO、频率温度和内存；prompt-cases-summary.json保存接受率与内存摘要。测试后删除临时runtime.json重启，恢复正常输出上限和默认设置。未提交推送。


## 中文Python斐波那契追加对照

提示：请编写一个 Python 函数，返回斐波那契数列的前 n 项。请包含清晰的注释，并给出一个调用示例。

同App默认配置、mmproj未加载，两轮同进程新会话，输入41输出128，cache_n0。

| 轮次 | Prefill tok/s | Decode tok/s | 接受 | 峰值RSS MiB | DMA结束GiB | 电池°C |
|---|---:|---:|---|---:|---:|---|
| 1 | 21.22 | 17.36 | 91/105 | 964.69 | 7.206 | 36.6–36.7 |
| 2 | 25.64 | 17.81 | 91/105 | 964.69 | 7.206 | 36.7–36.7 |

接受率86.7%，平均prefill23.43、decode17.59。两轮输出hash一致：True。此前英文版80.2%/16.53，本例中文不低于英文；两种回复内容并非相同token序列，非交错且样本少，不能归因于语言或外推。RSS/DMA口径同上。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/run-python-cn.py
```

原始app-prompt-python-cn-r1/r2全量记录与prompt文件保留，测试后移除runtime.json重启恢复正常默认。
