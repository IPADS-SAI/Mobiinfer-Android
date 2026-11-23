#!/data/data/com.termux/files/usr/bin/bash
# sweep.sh v2 — 一次采样；用 stdbuf -oL 让 llama-cli 的计时行在 teardown 崩溃前落盘。
# 输出 runs/results.tsv: tag n wall_ms rc gen_tps prompt_tps
#   [dmabuf_mb vmhwm_kb headroom_min_mb memavail_min_mb zram_peak_mb vmswap_peak_kb cached_min_mb cpu_cap_mhz]
#
# MEMMARK=1 时额外采内存（只在开的时候多三列，默认行格式不变）：
#   dmabuf_mb        专家窗口等 HOSTBUF 缓冲的实际归属 —— /proc/<pid>/smaps 里 /dmabuf:system 的 Size 之和
#   vmhwm_kb         进程常驻峰值（VmHWM）。注意：**窗口内存在 dmabuf 里，RSS/PSS 里是 0**，
#                    所以只看 RSS 会低估一个数量级（实测 WINDOW=80 时 dmabuf≈8.0GB、RSS≈0.2GB）。
#   headroom_min_mb  真实余量最小值 = MemFree + (Cached - Shmem) + SReclaimable。
#                    **这是唯一能直接喂给 dmabuf 的量**（dmabuf 不可换出，只能吃 free + 可回收 cache/slab）
#   memavail_min_mb  MemAvailable 最小值。⚠️ 这台 16+16 的机器上它不可信：ZRAM 让内核把
#                    「可换出的 anon」也算成 available（空闲时高估到 10.4GB），贴近低水位时又转保守
#                    （WINDOW=120 只剩 20MB 而实际还有 427MB cache）。两个都记，便于对账。
#   cached_min_mb    文件缓存最小值（Cached − Shmem）= 可回收 page cache 被挤到什么程度
#   zram_peak_mb     ZRAM 占用峰值（SwapTotal-SwapFree）= 内存压力预警：开始涨说明内核在换出 anon
#   cpu_cap_mhz      **热限频指标**：跑的过程中 min(cluster 的 scaling_max_freq)，单位 MHz。
#                    满血大核 4608、小核 3628；被热压到 2xxx 时 decode 会掉 8~12%，
#                    这时**不能**拿 t/s 跟凉机数据比（实测 42°C 时 cap=2361/2438 → 19.0~19.7，
#                    而凉机同配置历史记录是 20.9~21.6）。
#   vmswap_peak_kb   **本进程**被换出的峰值（/proc/<pid>/status 的 VmSwap）。>0 说明热路径被换出过，
#                    这个 run 的 t/s 不能用来比较。系统级换页/refault 只能从 adb 采（Termux 读不到
#                    /proc/vmstat），见 tools/moe/refault_probe.sh
#   采样开销：状态每 0.4s、smaps 每 1.6s；MEMMARK 开关不要在同一组 A/B 里混用。
set -u
ROOT=${ROOT:-$HOME/csm/moe-exp}
TAG=${TAG:-run}
N=${N:-64}
MEMMARK=${MEMMARK:-0}
# ── 内存安全闸 ────────────────────────────────────────────────────────────
# 实测（vivo V2546A，16G RAM + 16G ZRAM，MiniCPM 40 层）：
#   WINDOW+DEPTH=88  → dmabuf ~8.5GB，真实余量 ~2.3GB，安全
#   WINDOW+DEPTH=128 → dmabuf ~11.4GB，余量 ~0.77GB，43k 页 anon 被换出又读回、t/s −7%，
#                      并且**把手机上的 sshd 都熬死了**（靠 adb run-as 才恢复）
# 所以默认把每层槽数限制在 96；确需越界要显式 ALLOW_DANGEROUS=1。
#
# 这里必须按 bench.sh 的**最终有效默认值**检查，而不是把未设置的变量当 0。
# bench.sh 只认短名 WINDOW/DEPTH；未给时固定为 80/8。
WINDOW=${WINDOW:-80}
DEPTH=${DEPTH:-8}
BANK=${BANK:-256}
_safe_limit=96
_max_decimal=1000000

_normalize_uint() {
    local value=${1:-}
    case "$value" in
        ''|*[!0-9]*) return 1 ;;
    esac
    # Bash arithmetic treats a leading zero as octal. Normalize to decimal
    # before any comparisons or additions, and bound the input so arithmetic
    # cannot overflow or be used to bypass the memory gate.
    while [ "${#value}" -gt 1 ] && [ "${value#0}" != "$value" ]; do
        value=${value#0}
    done
    if [ "${#value}" -gt 7 ] || [ "$value" -gt "$_max_decimal" ]; then
        return 1
    fi
    printf '%s\n' "$value"
}

_refuse() {
    echo "[sweep] 拒绝执行：$*" >&2
    echo "[sweep] 无法确认每层内存占用时不做危险采样；确需越界要显式加 ALLOW_DANGEROUS=1" >&2
    exit 3
}

_check_slots() {
    local slots=$1 source=$2
    if [ "$slots" -gt "$_safe_limit" ] && [ "${ALLOW_DANGEROUS:-0}" != "1" ]; then
        _refuse "每层槽数 ${slots}（${source}）超过安全上限 ${_safe_limit}"
    fi
}

if ! _normalized=$(_normalize_uint "$DEPTH"); then
    _refuse "DEPTH=${DEPTH} 不是不超过 ${_max_decimal} 的非负十进制整数"
fi
DEPTH=$_normalized

if [ "$WINDOW" = "auto" ]; then
    # 模型侧按 BANK / MoE 层数计算后再夹到至少 top-k。sweep 不知道模型的
    # MoE 层数，除非调用者显式给出 MOE_LAYERS；不能猜，否则安全闸会失效。
    if [ "${ALLOW_DANGEROUS:-0}" = "1" ]; then
        echo "[sweep] ALLOW_DANGEROUS=1：放行 WINDOW=auto（无法预检实际槽数，原样传给 bench.sh）" >&2
    elif _moe_layers=$(_normalize_uint "${MOE_LAYERS:-}") &&
         [ "$_moe_layers" -gt 0 ] &&
         _bank=$(_normalize_uint "$BANK"); then
        MOE_LAYERS=$_moe_layers
        BANK=$_bank
        WINDOW=$((BANK / MOE_LAYERS))
        [ "$WINDOW" -ge 8 ] || WINDOW=8   # 模型侧至少装下 n_expert_used=8
        echo "[sweep] WINDOW=auto → ${WINDOW}（BANK=${BANK} / MOE_LAYERS=${MOE_LAYERS}）" >&2
    else
        _refuse "WINDOW=auto 无法在脚本内确认最终容量；请给数值 WINDOW，或同时给 MOE_LAYERS（BANK 默认 256）"
    fi
fi

if [ "$WINDOW" = "auto" ]; then
    # 被 ALLOW_DANGEROUS=1 显式放行的 WINDOW=auto：没有可校验的数值，保持原样。
    :
else
    if ! _normalized=$(_normalize_uint "$WINDOW"); then
        _refuse "WINDOW=${WINDOW} 不是不超过 ${_max_decimal} 的非负十进制整数"
    fi
    WINDOW=$_normalized
    _check_slots "$((WINDOW + DEPTH))" "WINDOW=${WINDOW} + DEPTH=${DEPTH}"
fi

if [ -n "${LAYER_WINDOWS:-}" ]; then
    # `read` 只读第一行，多行输入会让后续行绕过校验却被整体传给 bench.sh。
    case "$LAYER_WINDOWS" in
        *$'\n'*|*$'\r'*)
            _refuse "LAYER_WINDOWS 不允许换行（只有第一行会被校验，其余行不会被检查）"
            ;;
    esac
    _lw_items=()
    IFS=',' read -r -a _lw_items <<< "$LAYER_WINDOWS"
    for _item in "${_lw_items[@]}"; do
        _item=${_item//[[:space:]]/}
        [ -n "$_item" ] || continue
        case "$_item" in
            *:*) ;;
            *) _refuse "LAYER_WINDOWS 项 '${_item}' 格式非法（应为 层:槽数）" ;;
        esac
        case "$_item" in
            :*|*:|*:*:*) _refuse "LAYER_WINDOWS 项 '${_item}' 格式非法（应为 层:槽数）" ;;
        esac
        _layer=${_item%%:*}
        _width=${_item#*:}
        if ! _normalized_layer=$(_normalize_uint "$_layer") ||
           ! _normalized_width=$(_normalize_uint "$_width") ||
           [ "$_normalized_width" -eq 0 ]; then
            _refuse "LAYER_WINDOWS 项 '${_item}' 不是有效的 层:正整数"
        fi
        _layer=$_normalized_layer
        _width=$_normalized_width
        _check_slots "$((_width + DEPTH))" "LAYER_WINDOWS ${_layer}=${_width} + DEPTH=${DEPTH}"
    done
fi
mkdir -p "$ROOT/runs"
out="$ROOT/runs/$TAG-n$N.out"; err="$ROOT/runs/$TAG-n$N.err"
t0=$(date +%s%N)
if [ "$MEMMARK" = "1" ]; then
    N=$N WINDOW=$WINDOW DEPTH=$DEPTH stdbuf -oL -eL bash "$ROOT/bench.sh" >"$out" 2>"$err" &
    pid=$!
    dma_mb=0; hwm_kb=0; avail_min=0; head_min=0; zram_peak=0; vmswap_peak=0; cached_min=0; cpu_cap=0; i=0
    while kill -0 "$pid" 2>/dev/null; do
        st=/proc/$pid/status
        if [ -r "$st" ]; then
            h=$(awk '/VmHWM/{print $2}' "$st" 2>/dev/null)
            [ -n "$h" ] && [ "$h" -gt "$hwm_kb" ] && hwm_kb=$h
            sw=$(awk '/^VmSwap:/{print $2}' "$st" 2>/dev/null)
            [ -n "$sw" ] && [ "$sw" -gt "$vmswap_peak" ] && vmswap_peak=$sw
        fi
        # 一次读完需要的 meminfo 字段：Free / Available / Cached / Shmem / SReclaimable / SwapTotal / SwapFree
        set -- $(awk '/^MemFree:/{f=$2} /^MemAvailable:/{a=$2} /^Cached:/{c=$2} /^Shmem:/{sh=$2} \
                       /^SReclaimable:/{sr=$2} /^SwapTotal:/{st=$2} /^SwapFree:/{sf=$2} \
                       END{printf "%d %d %d %d %d %d %d", f/1024, a/1024, (c-sh)/1024, sr/1024, st/1024, sf/1024, 0}' \
                      /proc/meminfo 2>/dev/null) || set -- "" "" "" "" "" "" ""
        if [ $# -ge 6 ] && [ -n "$1" ]; then
            head=$(( $1 + $3 + $4 ))          # Free + 文件缓存 + 可回收 slab
            zram=$(( $5 - $6 ))
            if [ "$avail_min" -eq 0 ] || [ "$2" -lt "$avail_min" ]; then avail_min=$2; fi
            if [ "$head_min"  -eq 0 ] || [ "$head" -lt "$head_min"  ]; then head_min=$head; fi
            if [ "$cached_min" -eq 0 ] || [ "$3" -lt "$cached_min" ]; then cached_min=$3; fi
            [ "$zram" -gt "$zram_peak" ] && zram_peak=$zram
            # 热限频：取各 cluster 当前上限的最小值（MHz），记录整轮里最差的一次
            cap=$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0) / 1000 ))
            b=$(( $(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq 2>/dev/null || echo 0) / 1000 ))
            [ "$b" -gt 0 ] && [ "$b" -lt "$cap" ] && cap=$b
            [ "$cap" -gt 0 ] && { [ "$cpu_cap" -eq 0 ] || [ "$cap" -lt "$cpu_cap" ]; } && cpu_cap=$cap
        fi
        i=$((i + 1))
        if [ $((i % 4)) -eq 0 ] && [ -r "/proc/$pid/smaps" ]; then
            d=$(awk '/^Size:/{sz=$2} /dmabuf:system/{t+=sz} END{printf "%d", t/1024}' "/proc/$pid/smaps" 2>/dev/null)
            [ -n "$d" ] && [ "$d" -gt "$dma_mb" ] && dma_mb=$d
        fi
        sleep 0.4
    done
    wait "$pid"; rc=$?
else
    N=$N WINDOW=$WINDOW DEPTH=$DEPTH stdbuf -oL -eL bash "$ROOT/bench.sh" >"$out" 2>"$err"
    rc=$?
fi
t1=$(date +%s%N)
ms=$(( (t1 - t0) / 1000000 ))
gen=$(grep -aE "Generation: [0-9.]+ t/s" "$out" | tail -1 | sed -E 's/.*Generation: ([0-9.]+) t\/s.*/\1/')
prm=$(grep -aE "Prompt: [0-9.]+ t/s" "$out" | tail -1 | sed -E 's/.*Prompt: ([0-9.]+) t\/s.*/\1/')
if [ "$MEMMARK" = "1" ]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$TAG" "$N" "$ms" "$rc" "${gen:-NA}" "${prm:-NA}" "${dma_mb:-NA}" "${hwm_kb:-NA}" "${head_min:-NA}" "${avail_min:-NA}" "${zram_peak:-NA}" "${vmswap_peak:-NA}" "${cached_min:-NA}" "${cpu_cap:-NA}" >> "$ROOT/runs/results.tsv"
    echo "[sweep] $TAG n=$N ${ms}ms rc=$rc gen=${gen:-NA} t/s dmabuf=${dma_mb}MB 余量=${head_min}MB cache=${cached_min}MB zram=${zram_peak}MB cpu_cap=${cpu_cap}MHz"
else
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$TAG" "$N" "$ms" "$rc" "${gen:-NA}" "${prm:-NA}" >> "$ROOT/runs/results.tsv"
    echo "[sweep] $TAG n=$N ${ms}ms rc=$rc gen=${gen:-NA} t/s"
fi
