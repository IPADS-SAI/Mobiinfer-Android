#!/bin/bash
# genshin/probe.sh [--dur 180] —— 单 App 采集：原神游戏世界内的 CPU/RSS/iowait
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
DUR=180
while [ $# -gt 0 ]; do case "$1" in --dur) DUR=$2; shift 2;; *) die "未知参数 $1";; esac; done
PKG=com.miHoYo.Yuanshen
OUT=$FRONT_APP_HOME/artifacts/genshin-probe-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
fa_install_tools
[[ "$(focused_pkg)" = "$PKG" ]] || die "请先手动把原神点进游戏世界，再运行本脚本"
( fa_sample "$PKG" "$DUR" "$OUT" probe )
python3 "$(dirname "$0")/../shared/summarize_probe.py" "$OUT/probe" genshin
log "完成 -> $OUT"
