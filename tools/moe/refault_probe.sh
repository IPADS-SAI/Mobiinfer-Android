#!/usr/bin/env bash
# refault_probe.sh — 用 adb 采「匿名页被换出又读回」（refault）的次数，包住一次 bench 运行。
#
# 为什么需要它：窗口缓冲在 /dmabuf:system（不可换出），内存吃紧时内核只能把**别的进程的 anon 页**
# 压进 ZRAM；这些页再被访问时要解压回内存（refault），会抢 CPU 并让 t/s 掉 5~7%。
# /proc/vmstat 在 Termux 里被 SELinux 挡住（读不到 pswpin/pswpout），只有 adb shell 能读，
# 所以这个采样必须在手机外面做。
#
# 用法：
#   tools/moe/refault_probe.sh <label> -- <命令...>
# 例：
#   tools/moe/refault_probe.sh eF1 -- ssh -p 2223 u0_a371@127.0.0.1 \
#       'cd ~/csm/moe-exp && ./devlock env REL=$HOME/csm/moe/lfu-test TAG=eF1 N=128 \
#        MEMMARK=1 WINDOW=80 DEPTH=8 READ=8 bash ./sweep.sh'
#
# 输出：一行 TSV（stdout），并追加到 REFOUT（默认 ./runs/refault.tsv）
#   label  wall_s  refault_anon  refault_file  activate_anon  free_mb_start  free_mb_end
#
# 判读：refault_anon 大于约 10000 页（≈40MB）就说明这次运行已经被换页污染，
#       它的 t/s 不能拿去跟别的配置比 —— 要么降窗口，要么关掉后台应用重跑。
set -u
ADB=${ADB:-adb}
REFOUT=${REFOUT:-runs/refault.tsv}

if [ $# -lt 3 ] || [ "$2" != "--" ]; then
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
label=$1
shift 2

grab() {
    $ADB shell "grep -E '^(workingset_refault_anon|workingset_refault_file|workingset_activate_anon|nr_free_pages) ' /proc/vmstat" \
        2>/dev/null | tr -d '\r' |
        awk '{ v[$1] = $2 }
             END { printf "%d %d %d %d\n",
                        v["workingset_refault_anon"], v["workingset_refault_file"],
                        v["workingset_activate_anon"], v["nr_free_pages"] }'
}

if ! $ADB get-state >/dev/null 2>&1; then
    echo "[refault] adb 不可用（$ADB get-state 失败）" >&2
    exit 1
fi

read -r ra0 rf0 aa0 nf0 <<<"$(grab)"
if [ -z "${nf0:-}" ] || [ "$nf0" = "0" ]; then
    echo "[refault] 读不到 /proc/vmstat（adb 权限？）" >&2
    exit 1
fi

t0=$(date +%s)
"$@"
rc=$?
t1=$(date +%s)

read -r ra1 rf1 aa1 nf1 <<<"$(grab)"
wall=$((t1 - t0))
line=$(printf "%s\t%d\t%d\t%d\t%d\t%d\t%d" \
    "$label" "$wall" \
    "$((ra1 - ra0))" "$((rf1 - rf0))" "$((aa1 - aa0))" \
    "$((nf0 * 4 / 1024))" "$((nf1 * 4 / 1024))")
echo "$line"
mkdir -p "$(dirname "$REFOUT")"
if [ ! -s "$REFOUT" ]; then
    printf "label\twall_s\trefault_anon\trefault_file\tactivate_anon\tfree_mb_start\tfree_mb_end\n" >"$REFOUT"
fi
printf "%s\n" "$line" >>"$REFOUT"

if [ "$((ra1 - ra0))" -gt 10000 ]; then
    echo "[refault] ⚠️ 本 run 的 anon refault = $((ra1 - ra0)) 页（>10000）→ 受换页污染，t/s 不可比" >&2
fi
exit $rc
