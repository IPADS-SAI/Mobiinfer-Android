# OnePlus 13 decode 实验记录（2026-10-16）

## 输入和计时口径

固定纯文本提示：

> Explain how rain forms in clear simple English.

原生日志中的实际 prompt 为 **22 tokens**，包含模型输入格式带来的 token。
这里的 context capacity 是 2048，不代表输入长度是 2048。
CLI 对照统一生成上限 128 tokens，各成功测试实际生成 128 tokens。
手机已安装 App 默认上限 1024，本次自然结束为 204 tokens。
因此 App 与 128-token CLI 不能直接作为相同工作量的速度对比。
Decode 是原生 `eval time` 的整段生成速度，不含 prefill、模型加载或 UI 时间。

设备为 OnePlus 13，双 HTP0/HTP1，Q3 LES 主模型、打包专家、Q8_0 DSpark。
同一手机按设备互斥锁串行执行，进程间未清理系统文件缓存。
App 测试前后均冷启动；CLI 每轮为独立进程。未采集全程温度或本轮峰值内存。

## 实机结果

| 路径 | batch/ubatch | bank | draft | IO | lookahead | decode tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| 旧 CLI 库，纯文本 | 4 | 32 | 3 | 4 | 关 | 16.17 / 16.31 / 16.08 |
| 旧 CLI 库，纯文本 | 4 | 32 | 3 | 8 | 关 | 16.22 |
| 旧 CLI 库，纯文本 | 4 | 32 | 3 | 2 | 关 | 13.34 |
| 旧 CLI 库，纯文本 | 4 | 32 | 2 | 4 | 关 | 15.26 |
| 旧 CLI 库，纯文本 | 8 | 64 | 4 | 4 | 关 | 12.72 |
| App 核心库，加载 mmproj，媒体输入路径 | 4 | 32 | 3 | 4 | 关 | 15.98 / 16.18 / 16.13 |
| App 核心库，加载 mmproj，默认参数，上限128 | 128 | 256 | 3 | 4 | 开 | 16.05 |
| **App 核心库，纯文本，不加载 mmproj** | **4** | **32** | **3** | **4** | **关** | **18.04 / 18.07 / 18.29** |

最快组窗口为每层52，草稿接受84/124=67.74%，三轮回复正文一致。
该组 prefill 为20.99 / 20.96 / 22.03 tokens/s；decode 中位数18.07，单次最高18.29。
这只是短英文提示的实机证据，不是长上下文或图片任务的性能保证。
Draft4同时改变batch和bank，不能把全部下降归因于草稿长度。
小范围顺序扫描未经随机化，不能把相近的单次结果当作确定性提升。

已安装App（v2.5，com.tianchi.minicpmv）实际为batch128、bank256、draft3、
IO4、统一窗口52（cache48+prefetch4）、开启lookahead。
其204-token三轮decode为15.10 / 15.38 / 15.05，中位数15.10；prefill为
12.42 / 13.15 / 15.14。自然结束的App核心库CLI生成195 tokens，15.18 tokens/s，
接受126/210；App生成204 tokens，接受132/213。完整路径等价仍未建立。

历史2026-10-15预取A/B：开预取16.80/17.46/17.23，中位17.23；
关预取18.01/17.89/17.12，中位17.89。每层总槽52，不能解释为开启后简单少4个可用槽。
当前实验同时涉及库、输入处理和输出长度等变化，不能只用新旧速度差定位某一个原因。

## 库与权重对齐边界

App和Termux的主模型、草稿、mmproj逐字节cmp一致；未逐文件比较全部专家。
从已安装App复制核心so到独立实验目录，包含libllama、libllama-common、
libggml、libmtmd和v79 DSP库，不替换公共库。原有CLI启动器与CLI/server glue仍来自
旧CLI构建；补充私有NDK25 libc++_shared.so满足其链接符号，故不是与Java/JNI进程完全相同的运行时。

- App libllama.so SHA256：`1c0b22293f1fe52a76a1a30c866a2c58d5e2268687b2d8dbf1b837cc1884ab6d`
- App libggml-htp-v79.so SHA256：`d49f2101fd08eb2b365aa98b076f4a7f1cd720cee95e88dc17ad929a24cb1ef6`

## 最快组复现

手机已准备的独立目录为 `~/csm/moe/experiments/cli-app-equivalence-20261016`。
其中lib/为App核心so加旧CLI glue，libc++_shared.so为所需NDK25运行库。
脚本依赖此目录，不是任意手机可直接执行的安装工具。
先关闭App以避免DSP并发，再在本地获取设备锁运行：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- \
  ssh oneplus13-reverse \
  'bash ~/csm/moe/experiments/cli-app-equivalence-20261016/app-libs-text-fast.sh'
```

手机端最快组完整脚本：

```bash
#!/data/data/com.termux/files/usr/bin/bash
set -eu
cd "$HOME/csm/moe"
case_dir="$PWD/experiments/cli-app-equivalence-20261016"
export LD_LIBRARY_PATH="/system/lib64:/vendor/lib64:$case_dir/lib:$PWD/lib"
export ADSP_LIBRARY_PATH="$case_dir/lib"
export GGML_HEXAGON_DEVICES=2
export GGML_HEXAGON_MBUF=256M,512M,4G
export GGML_HEXAGON_OPPOLL=1
export LLAMA_MOE_HTP_PACKED_DIR="$PWD/experts-htp-tiled"
export LLAMA_MOE_DECODE_WINDOW=52
export LLAMA_MOE_WINDOW_MAX_ROWS=8
export LLAMA_MOE_IO_POOL=1
export LLAMA_MOE_IO_THREADS=4
export LLAMA_MOE_LOOKAHEAD_PREFETCH=0
export LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=4
export LLAMA_MOE_PREFETCH_MAX_ROWS=8
export MB_MEDIA_EMBD_BATCH=1
export LLAMA_DSPARK_MEDIA_FEATURES=1
export LLAMA_DSPARK_TARGET_POSITIONS=1
export LD_PRELOAD="$case_dir/libc++_shared.so"
exec ./bin/llama-cli -m models/minicpm47-les-hexagon.gguf --moe-experts experts --moe-slot-bank 32 --device HTP0,HTP1 --split-mode layer --tensor-split 1,1 -ngl 99 -c 2048 -b 4 -ub 4 -t 4 -tb 4 -fit off --no-warmup --seed 42 --temp 0 --single-turn --reasoning off -n 128 -p 'Explain how rain forms in clear simple English.' --spec-type draft-dspark --model-draft models/dspark-MiniCPM-V-4_7-q8_0.gguf --device-draft HTP0,HTP1 -ngld 99 --spec-draft-n-max 3 --image-max-tokens 64 -lv 3
```

其他组：旧库使用 `fast.sh -lv 3`；IO以 `BENCH_IO_THREADS=2` 或8覆盖；
草稿2增加 `--spec-draft-n-max 2`；草稿4增加
`-b 8 -ub 8 --moe-slot-bank 64 --spec-draft-n-max 4`。
App核心库媒体小batch用 `app-libs-fast.sh`；默认参数128-token对照用
`app-libs-default128.sh`。脚本在本地实验目录和手机上述目录保留。

本地完整原始数据：`moe-speculative/hexagon-results/decode-fast-repro-20261016/`，
包含README、summary.json、各组脚本和stdout/stderr；相邻
`current-app-default-20261016/`保存App原始JSON/日志和UI复现脚本；
`cli-app-equivalence-20261016/`保存库对齐过程。未将APK、权重或原始大日志提交到仓库。
其中旧库媒体路径run2与权重cmp发生IO重叠，不用作干净的性能基线。

## lookahead 是什么

当前实现用**当前层的隐藏状态乘下一层已有的router权重**，近似预测下一层会用的专家。
取预测top-k专家ID后，后台提前读取其权重，希望与当前层计算重叠。
`LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=4`表示预测4个专家，不是向前预测4层。
此路径用于decode及满足小批宽条件的投机验证；不是正常大chunk prefill的通用预取。

这不需要另外训练预测器，也不是DSpark：DSpark预测后续token，lookahead预测后续层的专家。
关闭lookahead后，专家按需加载和缓存仍工作，DSpark仍工作。
预测误差、额外router计算、后台读取和缓存管理可能抵消命中收益；
本次没有用独立消融量化各项开销，不能断言关闭更快只由某一项导致。
代码参考：`src/models/qwen35moe.cpp`中的prefetch_router_logits/prefetch_topk，
以及`src/llama-moe-slot.cpp`中的lookahead_expert_prefetch。

## Prefill 已激活专家预热 A/B（2026-10-16）

对用户指出的warm_from_prefill进行实机验证。App核心库CLI媒体路径，
batch/ubatch128、bank256、draft3、IO4、lookahead开、窗口52（cache48+prefetch4），
相同22-token输入、128-token输出。只改变LLAMA_MOE_PREFILL_WARM_ROWS=0或32。
执行顺序off/on/on/off/off/on，每组3次；六次输出正文SHA256相同，接受数84/127相同。

| warm rows | Prefill tokens/s（三次） | Decode tokens/s（三次） | prefill+decode中位秒数 |
| --- | --- | --- | --- |
| 0 | 25.98 / 23.03 / 24.41 | 16.39 / 16.65 / 16.95 | 8.642 |
| 32 | 18.44 / 18.51 / 18.69 | 16.96 / 16.86 / 16.79 | 8.780 |

warm=32实际只选本批22行，每层达到48个专家上限，40层累计复制2880MiB，
日志累计memcpy段wall约225–228ms。这是累计拷贝流量，不是新增常驻内存。
Prefill中位耗时901.36→1188.65ms；decode中位16.65→16.86 tokens/s（约+1.3%），
两组范围重叠；总推理中位耗时反而增加约1.6%。本短提示不支持把它作为明确整体加速。
未修改App安装包；本地最新源码已设warm_rows=32，当前安装版实验用环境变量开关。

warm_from_prefill是在本层shared-bank加载完成后，把本批末尾若干行实际路由到的专家
去重后拷贝至独立decode窗口，并保存落位供切换时恢复。默认倒序遍历token行，
每行按rank顺序展开，直到容量满；rank-first是另一个可选遍历开关。
它不会额外读专家文件，但增加memcpy和缓存管理。当前源码phase-reuse模式会跳过它。
调用受!window_mode条件限制；小batch如果已直接走窗口，设置warm_rows也可能没有复制。
不能把这项warm和A100统计得到的静态热专家预热混为一谈。

lookahead的模型图条件是ubatch.n_tokens<=prefetch_max_rows（本次为8），
不是严格的“只在decode”判断：小批prefill也可能触发，大chunk正常prefill通常跳过。
因此没有理由预期它直接加速大chunk prefill；若刻意放大阈值，需要另做性能和容量验证。
这次A/B保持lookahead开启，不是lookahead对prefill的独立消融。

复现单组（替换0为32测试开启；在本地持设备锁执行）：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- \
  ssh oneplus13-reverse \
  'LLAMA_MOE_PREFILL_WARM_ROWS=32 bash ~/csm/moe/experiments/cli-app-equivalence-20261016/app-libs-default128.sh'
```

完整六轮顺序脚本、本次基础配置run.sh、原始stdout/stderr及summary.json保存在
`moe-speculative/hexagon-results/prefill-warm-ab-20261016/`。

## 长上下文 lookahead A/B（2026-10-16）

OnePlus 13实机，App核心库CLI纯文本路径；非App UI测试。
batch/ubatch=128、bank=256、draft=3、IO=4、ctx=2048，固定输出128 tokens。
warm_from_prefill明确关闭：LLAMA_MOE_PREFILL_WARM_ROWS=0。
预取关闭cache52；开启cache48+prefetch4；两组统一窗口均52。
每个长度执行off/on/on/off/off/on，未清理OS文件缓存。

输入是固定中文日常对话背景，附详细次日计划问题；生成器按UTF8字符边界裁剪背景，
使用模型自身vocab tokenizer对完整模板计数到512/1024，必要时用少量rain填充尾差。
原始推理日志再次确认实际prompt为512/1024、实际输出128，不存在context容量冒充输入长度。
prompt-512.txt、prompt-1024.txt是实际输入；这是两条合成背景样本，不是泛化评测。

| 输入tokens | lookahead | Prefill tokens/s中位 | Decode tokens/s中位 | 总推理秒数中位 |
| --- | --- | --- | --- | --- |
| 512 | off | 62.86 | 14.10 | 17.243 |
| 512 | on | 61.69 | 13.94 | 17.483 |
| 1024 | off | 59.91 | 13.44 | 26.633 |
| 1024 | on | 56.14 | 14.21 | 27.272 |

512-token：开启后decode约-1.1%，总推理约+1.4%。
1024-token：开启后decode约+5.7%，但prefill未加速，总推理约+2.4%。
总时间为prefill+decode，不含加载和UI；各项分别取中位，不能直接由中位速度相加重建总时间。
每个输入六次回复正文一致，SHA256见summary.json；输出内容正常但在128-token上限处截断。
CLI显示长prompt时主动截断预览，可能切断UTF8字符；它不代表实际输入文件损坏或输入被截断。
回复哈希解析已去掉该预览、加载动画及计时尾部。

正常prefill批宽128>lookahead max_rows8，因此不直接运行下一层预测；
两个输入长度均为128的整数倍，没有小尾批。prefill差异不能解释成lookahead直接加速/减速大批前向。
初始化、缓存状态、IO和设备状态仍可能影响时间，本次没有进一步隔离这些因素。
这两条样本不支持统一开启lookahead：1024只改善decode，总请求时间仍未改善。

复现单组（本地获取设备锁，长度可改1024；关闭用BENCH_PREFETCH=0 BENCH_CACHE=52）：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- \
  ssh oneplus13-reverse \
  'BENCH_LENGTH=512 BENCH_PREFETCH=1 BENCH_CACHE=48 bash ~/csm/moe/experiments/cli-app-equivalence-20261016/long-lookahead-run.sh'
```

复现完整12轮：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- \
  ssh oneplus13-reverse \
  'bash ~/csm/moe/experiments/cli-app-equivalence-20261016/long-lookahead-sweep.sh'
```

手机需保留前述私有App库、CLI glue及NDK25运行库。
本地完整记录：moe-speculative/hexagon-results/lookahead-long-ab-20261016/，
包括完整run.sh/sweep.sh、prompt文件、生成器源码、原始stdout/stderr、analyze.py和summary.json。
没有修改App设置或安装包。

## warm_from_prefill 长度对照（2026-10-16）

OnePlus 13，App核心库CLI纯文本路径；batch/ubatch128、bank256、draft3、IO4、
ctx2048、lookahead关闭、每层窗口52，固定输出128 tokens。
每个长度按off/on/on/off/off/on执行，warm rows为0或32，其余相同。
128输入由同一中文日常对话背景和计划问题生成；512/1024复用前述实际输入文件。
原生日志确认输入和输出计数。每长度六次回复正文SHA256一致。

| 输入tokens | warm rows | Prefill中位tokens/s | Decode中位tokens/s | 总推理中位秒 | 累计复制MiB中位 | 复制段累计ms中位 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 0 | 62.85 | 12.19 | 12.533 | 0.0 | 0.0 |
| 128 | 32 | 47.12 | 11.80 | 13.563 | 3120.0 | 286.4 |
| 512 | 0 | 73.15 | 14.06 | 16.104 | 0.0 | 0.0 |
| 512 | 32 | 62.85 | 13.70 | 17.533 | 12442.5 | 870.0 |
| 1024 | 0 | 69.91 | 13.50 | 24.172 | 0.0 | 0.0 |
| 1024 | 32 | 59.02 | 14.08 | 26.442 | 24927.0 | 2177.4 |

128/512/1024长度下总耗时中位分别增加约8.2%/8.9%/9.4%。
1024下decode中位提高约4.3%，未抵消prefill增加的成本。
128组prefill波动较大（关闭47.42/62.85/64.06，开启45.91/47.12/60.14）；
本次是小样本顺序交错观察，不代表全部差异都由开关单独造成。

开启组实际warm调用次数分别为40/160/320：每个prefill chunk、每层都复制，
512/1024累计约12.15/24.34GiB，属于累计拷贝流量，不是新增常驻内存。
落位表每层会被后续chunk更新，故值得研究仅在最终prefill chunk预热，
或者直接保留最终chunk已装载专家，以避免反复复制。本轮未实现该优化。
没有因本轮结果修改App默认值或安装APK。

复现单组，BENCH_WARM_ROWS可设0，BENCH_LENGTH可设128/512/1024：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- \
  ssh oneplus13-reverse \
  'BENCH_LENGTH=1024 BENCH_PREFETCH=0 BENCH_CACHE=52 BENCH_WARM_ROWS=32 bash ~/csm/moe/experiments/cli-app-equivalence-20261016/warm-length-run.sh'
```

完整18轮使用同一目录warm-length-sweep.sh。
本地实验文件：moe-speculative/hexagon-results/prefill-warm-lengths-20261016/，
包含run.sh、sweep.sh、analyze.py、summary.json及原始stdout/stderr；
输入文件512/1024在相邻lookahead-long-ab-20261016/，本目录另保存prompt-128.txt。
手机依赖前述私有App库、CLI glue及NDK25 libc++，不是任意手机直接运行的脚本。


## 仅最终 prefill 预热最近 32 tokens（2026-10-16）

实现位于本维护仓库，尚未安装到 App。`LLAMA_MOE_PREFILL_WARM_ROWS=32`
启用后，server-context（也供本版本 llama-cli 使用）显式标记最终 prompt batch；
llama-context 只给其最终 microbatch 开放专家复制，完成后同步并关闭标记。
不根据 logits 或 token 数猜测 prompt 是否结束，初始化计算与中间 chunk 不预热。

每层只访问最终 microbatch 最近 32 行的真实路由结果，从最近 token 开始、按
rank 遍历并去重，最多填满 decode 缓存区（本轮52槽）；不额外读取专家文件，
直接从已加载的 prefill bank 复制到独立 decode 窗口，并在切换时恢复映射。
这是最后32个 token 的专家候选集合，并非只加载32个专家，也不改变路由选择。
槽位数量与常驻权重预算不增加；本轮每层最多78MiB、40层累计最多3120MiB复制流量。
phase-reuse 的重叠视图仍禁止此路径。混合序列 batch 跳过预热。

为避免原有 checkpoint 把最终块截成4个 token，开启时 checkpoint 尾部改为32；
必要时前一块提前结束，保证完整未缓存文本 prompt 的尾块至少32行。
短 prompt、已缓存前缀导致本次新增不足32行，或 batch/ubatch小于32时，
只能使用本次最终 microbatch 实际计算的行，不能凭空获取未计算路由。
其他直接调用 llama_decode 的客户端需要用新增的
`llama_set_moe_prefill_final(ctx, true/false)` 括住最终 prompt batch。

### 分块与正确性检查

128-token测试中，关闭时原有checkpoint分块为124+4，开启时为96+32，
两者生成正文不同，不能直接归因于专家缓存错误。
另构建仅禁用复制、保留相同96+32分块的诊断版本：其128-token正文与开启复制
逐字一致，SHA256均为
`b3cd26d3bf6290a1dc5237d96059a1d1defca3620b35213235ae238dc4cc097c`。
124+4基线正文SHA256为
`9dbf430bd9a1788f8397feacddd0348c73088ba2046d0025a66828936753d23e`。
因此该样本的差异由分块路径引入，未复现出复制开关导致的正文差异。
这不是全部输入数值正确性的证明。诊断改动已从维护源码移除。

本轮使用当前源码统一编译的CLI及全部核心库、HTP skel、NDK27运行库；
与前文旧App核心库实验不是相同二进制，不应直接跨表归因。
复现命令（依赖手机该私有目录的库、模型、prompt）：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- \
  ssh oneplus13-reverse \
  'BENCH_LENGTH=1024 BENCH_PREFETCH=0 BENCH_CACHE=52 BENCH_WARM_ROWS=32 bash ~/csm/moe/experiments/prefill-final32-20261016/run.sh'
```

原始日志、诊断库、运行脚本和分析脚本位于
`moe-speculative/hexagon-results/prefill-final32-20261016/`。

### 当前源码实机结果

128每组3次取中位，512/1024每组仅1次；均生成128 tokens。

| 输入tokens | warm rows | Prefill tokens/s | Decode tokens/s | 总推理秒 | 累计复制MiB |
| --- | --- | --- | --- | --- | --- |
| 128 | 0 | 62.55 | 12.13 | 12.567 | 0.0 |
| 128 | 32 | 46.31 | 12.97 | 12.559 | 3120.0 |
| 512 | 0 | 68.74 | 14.32 | 16.385 | 0.0 |
| 512 | 32 | 63.97 | 14.23 | 16.997 | 3114.0 |
| 1024 | 0 | 66.20 | 14.14 | 24.522 | 0.0 |
| 1024 | 32 | 67.18 | 13.89 | 24.459 | 3117.0 |

所有完成的开启组均严格40次调用，每层一次，rows=32。1024输入复制流量从旧实现约24.34GiB降为约3.04GiB（减少约87.5%）；此为调用/流量比较，不是跨二进制的性能因果对照。
当前1024开启/关闭总时间相近，512开启反而更慢；分块不同且生成正文不同，decode数据不能解读为同轨迹纯缓存加速。当前证据不支持声称总推理稳定提速。
尚未安装新App；源码App runtime共用此server-context路径，需重新构建安装才会应用。


## 预取上限 8/10/14/16 实机对照（恢复连接后）

所有组固定预测depth16、cache36+prefetch16=52槽、bank256、IO4、draft3、
b/ub128、ctx2048、最后32行专家预热；实际512输入/128输出，纯文本CLI，未安装新App。
每组3次，8/10/14/16 → 16/14/10/8 → 8/10/14/16；全部退出0、12次正文SHA256一致。
温度31.9–39.7°C。设备此时读取和整体推理低于前轮，原因未隔离，不能跨时段直接归因。

| 最多实际预取 | Prefill中位tokens/s | Decode中位tokens/s | 总推理中位秒 | 整次逻辑专家读取GiB |
| --- | --- | --- | --- | --- |
| 8 | 34.18 | 11.80 | 25.833 | 55.45 |
| 10 | 33.35 | 11.77 | 26.337 | 57.34 |
| 14 | 33.28 | 11.60 | 26.467 | 60.02 |
| 16 | 32.97 | 11.43 | 26.706 | 61.16 |

16比8的decode中位慢约3.1%，整次逻辑读取增加约10.3%。读10与8接近；该样本不支持继续增加至14/16。
读取量含初始化、prefill和decode的逻辑文件读，不是物理UFS流量。

另以64生成token单次采集CPU事件、关闭DSP profile，仅取rows=4的完整verify：

| 读取上限 | 已发起任务数 | 实际平均完成读取专家 | 整批读取均值ms | 发起到需求均值ms | 需求时整批未完成 |
| --- | --- | --- | --- | --- | --- |
| 8 | 503 | 7.29 | 4.51 | 2.74 | 88.3% |
| 10 | 489 | 8.58 | 4.87 | 2.75 | 90.8% |
| 14 | 494 | 10.16 | 5.45 | 2.73 | 88.3% |
| 16 | 564 | 10.36 | 3.82 | 2.73 | 67.4% |

任务均值均超过约2.7ms可重叠窗口，支持按窗口限制准入的方向，但不是逐专家有效读取迟到率。
16组读取均值反而低于14组，说明设备状态/读缓存/样本选择有波动，不能据此声称16的单专家读取更快。
主线程仍不等待整批，未完成时按原有需求读取；没有启用自动预算调节，也没有改App默认值。
此前DSP profile的每层FFN约0.93ms、40层约37.34ms来自另一时段，不是本轮FFN实测值。

复现完整矩阵（依赖手机私有库和权重）：

```bash
/home/yydh/executorch/.tmp/locks/lockrun device -- ssh oneplus13-reverse \
  'bash ~/csm/moe/experiments/prefetch-budget-20261016/high-sweep.sh'
```

脚本stage/high-sweep.sh、原始stdout/stderr/exit/温度、summary.json、timing-summary.json
均保存在本地 moe-speculative/hexagon-results/prefetch-budget-20261016/。
