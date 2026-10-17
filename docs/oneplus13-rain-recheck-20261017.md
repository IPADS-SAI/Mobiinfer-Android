# 一加13短英文历史配置复测

2026-10-17，按历史 app-libs-text-fast.sh 原样运行：Explain how rain forms in clear simple English.，实际22输入、128输出，Q3 LES+Q8 DSpark，draft3，双HTP，batch/ubatch4，bank32，decode窗口52，关闭lookahead，IO4，OPPOLL1，ctx2048，不加载mmproj，seed42/temp0。关闭App后使用设备锁串行执行，未清页缓存、未主动预热。

| 轮次 | Prefill tok/s | Decode tok/s | 草稿接受 |
|---|---:|---:|---|
| 1 | 10.03 | 15.05 | 84/124 |
| 2 | 11.30 | 15.13 | 84/124 |
| 3 | 11.65 | 15.40 | 84/124 |

用户提示温度高后立即执行停止命令；第三轮原始日志已完成128输出，故保留完整计时，并未再启动测试。停止时电池41.3°C、PhoneTemp40°C，不能作为降温后与历史17–18的公平对比，也不能证明差异全部由温度造成。本轮未采集进程峰值RSS，属于记录缺项，不能引用此前其他进程内存代替。

原始目录：/home/yydh/moe-speculative/hexagon-results/oneplus-rain-recheck-20261017/，r1/r2/r3.log、device-before.txt。

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- ssh oneplus13-reverse 'bash ~/csm/moe/experiments/cli-app-equivalence-20261016/app-libs-text-fast.sh'
```

vivo当前默认App同一句英文两轮22/128：prefill10.18/12.14，decode12.07/13.13，草稿84/127。原始文件vivo-app-20261017/vivo-rain-poll-r1/r2，测试临时runtime.json已移除。其batch512/bank256、单HTP与本组不同，不能只据速度差归因芯片。

## 相同小bank配置，512-token长输入

用户要求继续实测后，使用同一历史prompt-512.txt，仅将原短英文-p替换为-f文件；其余保持batch/ubatch4、bank32、window52、lookahead0、IO4、draft3、旧App核心库纯文本CLI配置。单轮输入512、输出128，prefill21356.99ms（23.97 tok/s），decode8879.07ms（14.42 tok/s），草稿接受78/146。进程Max RSS778348KiB（约760.1MiB），不是包含全部DMA的总内存。总进程墙钟35.40秒，退出0；未清页缓存。开始电池36.3°C，结束35.3°C，只有前后读数而非连续温度轨迹。

因此该小chunk配置的512-token prefill实测约24 tok/s；与此前batch512配置80–95不同轮次、不同环境，不能声称严格加速倍数。本次未运行1024或更长输入。

完整脚本保存在原始目录long512-smallbank.sh，日志long512-smallbank.log，结束温度long512-battery-after.txt。复现（先关闭App，避免HTP争用）：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- ssh oneplus13-reverse '/system/bin/time -v bash ~/csm/moe/experiments/cli-app-equivalence-20261016/long512-smallbank-20261017.sh'
```

## 长输入之后回测短英文

同一历史脚本app-libs-text-fast.sh（batch/ubatch4、bank32、window52、lookahead0、draft3），独立新进程，22输入/128输出：prefill15.22 tok/s（1445.75ms），decode16.40 tok/s（7804.08ms），接受84/124。电池开始33.2°C、结束32.8°C，Max RSS778312KiB，退出0。未清页缓存；单轮，不据此认定温度为此前差异唯一原因。

原始short-now.log、short-now-battery-before/after.txt保存在同目录。复现使用/system/bin/time -v包裹上述app-libs-text-fast.sh。


## 连续五轮短英文复测

保持相同历史CLI脚本，每轮独立进程，无主动降温、无清页缓存，输出均128、输入均22 tokens。循环脚本repeat-short.py，原始short-continuous-r1..r5.log及每轮前后温度，汇总short-continuous-summary.json。

| 轮次 | Prefill tok/s | Decode tok/s | 电池前→后°C | Max RSS MiB |
|---|---:|---:|---|---:|
| 1 | 15.35 | 16.15 | 32.2→32.1 | 759.97 |
| 2 | 15.43 | 16.32 | 32.1→32.0 | 759.93 |
| 3 | 15.26 | 16.12 | 32.0→31.8 | 760.18 |
| 4 | 19.99 | 17.55 | 31.8→31.4 | 759.73 |
| 5 | 22.80 | 17.71 | 31.4→31.3 | 759.76 |

Decode中位数16.32 tok/s，平均16.77，范围16.12–17.71。五轮均退出0、接受84/124。后两轮达到17.55/17.71，历史17级别短输入速度再次出现；前后电池温度读数下降，但未采集连续频率、逐阶段IO或热状态，不能据此认定温度是唯一原因。RSS不包含全部DMA，不能作为总内存。

## ub512 对照与容量限制

用户要求调大ub连续五轮。首先保留b4、设置ub512运行五轮，decode17.62/17.81/17.80/17.75/17.64；但运行时n_ubatch=min(n_batch,requested_ubatch)，该组不是有效ub512对照，不能据此判断ub512速度。原始short-ub512-r*.log与summary.json保留。

为让ub512真正生效，追加b512/ub512，bank32等其他参数不变，日志确认n_batch/n_ubatch均512。首轮失败退出6：`layer 0 requires more than 32 unique experts in one graph; reduce ubatch or increase --moe-slot-bank`。未继续重复相同失败，也未擅自提高bank以混淆变量。没有获得有效ub512 decode速度，无法完成有效五轮。原始short-bub512-r1.log、short-bub512-summary.json与复现脚本保留。日志等级从3改4以确认实际参数，该次未产生性能结论。


## b/ub512、bank256 连续五轮

用户授权增加bank后，使用short-bub512-bank256.sh；b/ub512、bank256，其他保持历史纯文本CLI：短英文22输入/128输出，window52、prefetch0、IO4、draft3、双HTP、OPPOLL1、旧App核心库、不加载mmproj、ctx2048、seed42/temp0。每轮新进程，不清页缓存、不主动降温。日志确认target和draft n_batch/n_ubatch均512。为核对参数使用-lv4，之前小配置为-lv3，额外日志也是比较限制。

| 轮次 | Prefill tok/s | Decode tok/s | 电池前→后°C | Max RSS MiB | 草稿接受 |
|---|---:|---:|---|---:|---|
| 1 | 17.28 | 15.66 | 31.6→31.6 | 759.97 | 84/126 |
| 2 | 15.50 | 16.43 | 31.6→31.6 | 759.78 | 84/126 |
| 3 | 18.58 | 15.99 | 31.6→31.6 | 759.79 | 84/126 |
| 4 | 16.11 | 16.46 | 31.6→31.6 | 760.09 | 84/126 |
| 5 | 18.89 | 16.13 | 31.6→31.7 | 759.76 | 84/126 |

平均decode16.13，中位16.13 tok/s，范围15.66–16.46，五轮退出0。该组合未保持此前小配置17级速度，但同时改变batch、ubatch、bank，且顺序非随机、CPU频率/页缓存未控制，不能单独归因ub。RSS为内核峰值而非包含全部DMA的总内存。

原始short-bub512-bank256-r1..r5.log、前后电池、summary.json和循环脚本均保存在原始目录。复现：

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-rain-recheck-20261017/repeat-short-bub512-bank256.py
```

未改变App默认配置，未提交或推送。


## 固定b512/bank256，仅ub降至128，连续五轮

与上一组相比仅命令参数ub512→128，日志确认target/draft均n_batch512、n_ubatch128；其余prompt、模型库、bank256、window52、prefetch0、IO4、draft3、日志等级4相同，每轮新进程，不清页缓存、不主动降温。

| 轮次 | Prefill tok/s | Decode tok/s | 电池前→后°C | Max RSS MiB | 草稿接受 |
|---|---:|---:|---|---:|---|
| 1 | 21.93 | 17.13 | 31.8→31.8 | 759.70 | 84/126 |
| 2 | 20.71 | 17.18 | 31.8→31.7 | 759.48 | 84/126 |
| 3 | 20.67 | 16.90 | 31.7→31.7 | 759.96 | 84/126 |
| 4 | 19.53 | 16.08 | 31.7→31.7 | 760.17 | 84/126 |
| 5 | 21.83 | 17.36 | 31.7→31.7 | 759.80 | 84/126 |

Decode平均16.93、中位17.13 tok/s，全部退出0、输入22输出128。上一ub512组平均16.13，本组约高4.9%；顺序测试未随机交错，频率/页缓存等仍可能变化，不能认定全部差异由ub造成。RSS不等于包含DMA的总内存。未改App默认配置。原始short-b512-ub128-bank256-r1..r5.log、summary和前后温度均保留。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-rain-recheck-20261017/repeat-short-b512-ub128-bank256.py
```


## ub128/512 ABBA交错12轮

固定b512、bank256、window52、prefetch0、IO4、draft3、双HTP、OPPOLL1、同旧App核心库纯文本CLI、日志等级4。短英文22输入/128输出。按[128,512,512,128]重复3组，每轮独立新进程，不清页缓存、不主动降温。

| 轮次 | ub | Prefill tok/s | Decode tok/s | 电池前→后°C | Max RSS MiB | 草稿接受 |
|---|---:|---:|---:|---|---:|---|
| 1 | 128 | 21.89 | 17.16 | 31.6→31.5 | 760.26 | 84/126 |
| 2 | 512 | 16.14 | 16.99 | 31.5→31.5 | 759.77 | 84/126 |
| 3 | 512 | 19.97 | 16.95 | 31.5→31.4 | 759.96 | 84/126 |
| 4 | 128 | 22.25 | 17.00 | 31.4→31.4 | 759.81 | 84/126 |
| 5 | 128 | 21.55 | 17.03 | 31.4→31.4 | 759.57 | 84/126 |
| 6 | 512 | 17.52 | 16.80 | 31.4→31.5 | 759.98 | 84/126 |
| 7 | 512 | 19.95 | 17.07 | 31.5→31.6 | 759.95 | 84/126 |
| 8 | 128 | 21.29 | 16.94 | 31.6→31.8 | 759.95 | 84/126 |
| 9 | 128 | 21.33 | 16.99 | 31.8→31.8 | 759.95 | 84/126 |
| 10 | 512 | 19.97 | 16.67 | 31.8→31.9 | 760.00 | 84/126 |
| 11 | 512 | 19.21 | 17.09 | 31.9→32.1 | 759.93 | 84/126 |
| 12 | 128 | 21.42 | 17.03 | 32.1→32.2 | 759.76 | 84/126 |

| ub | Decode均值 | 中位数 | 范围 | Prefill均值 |
|---|---:|---:|---|---:|
| 128 | 17.025 | 17.015 | 16.94–17.16 | 21.622 |
| 512 | 16.928 | 16.970 | 16.67–17.09 | 18.793 |

交错后ub128平均decode仅比ub512高0.57%，范围重叠，未复现先前顺序组约5%差距，不支持“ub512会明显拖慢decode”的结论。温度31.4–32.2°C，未记录连续CPU频率与阶段IO，不能确定小差异机制。全部退出0。RSS为内核峰值，不含全部DMA总内存。短输入prefill指标不能外推长上下文。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-rain-recheck-20261017/interleave-ub.py
```

原始interleave-r01..r12-ub*.log、前后温度、interleave-summary.json和循环脚本均保留。未改App默认配置、未提交或推送。


## 512-token长输入，ub128/512 ABBA

固定b512、bank256、window52、prefetch0、IO4、draft3、双HTP、OPPOLL1、同旧App核心库纯文本CLI、日志等级4。实际输入512输出128，每轮独立进程，不清页缓存、不主动降温。历史prompt-512.txt。

| 轮次 | ub | Prefill tok/s | Decode tok/s | 电池前→后°C | Max RSS MiB | 草稿接受 |
|---|---:|---:|---:|---|---:|---|
| 1 | 128 | 65.34 | 13.92 | 31.8→31.7 | 759.50 | 75/150 |
| 2 | 512 | 99.34 | 12.85 | 31.7→31.8 | 1095.93 | 76/152 |
| 3 | 512 | 95.38 | 12.82 | 31.8→31.9 | 1095.27 | 76/152 |
| 4 | 128 | 64.53 | 13.74 | 31.9→32.1 | 759.58 | 75/150 |

仅各两轮：ub512提高本组长输入prefill，但其后decode较低；未采集连续频率、阶段IO/完整DMA，无法确定decode差异机制或外推更长输入。内核Max RSS不是包含全部DMA的总内存。首个试跑prefill60.01/decode13.40后因日志UTF-8解析报错，修正为errors=replace后重新执行完整ABBA；试跑同名日志被覆盖，因此不纳入正式汇总，正式4轮原始字节日志保留。全部正式轮退出0。

```bash
python3 /home/yydh/moe-speculative/hexagon-results/oneplus-rain-recheck-20261017/interleave-long512.py
```

原始long512-interleave-r01..r04日志、温度与long512-interleave-summary.json保存于同目录。未改App默认、未提交推送。
