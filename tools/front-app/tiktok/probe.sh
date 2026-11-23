#!/bin/bash
# tiktok/probe.sh [--dur 120] —— 单 App 采集：抖音刷视频的 CPU/RSS/iowait
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
DUR=120
while [ $# -gt 0 ]; do case "$1" in --dur) DUR=$2; shift 2;; *) die "未知参数 $1";; esac; done
PKG=com.ss.android.ugc.aweme
OUT=$FRONT_APP_HOME/artifacts/tiktok-probe-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
fa_install_tools
bash "$(dirname "$0")/appwork.sh" "$OUT/never.stop" 0 >"$OUT/feed.log" 2>&1 &
( fa_sample "$PKG" "$DUR" "$OUT" probe )
touch "$OUT/never.stop"; wait 2>/dev/null
fa_force_stop "$PKG"
python3 "$(dirname "$0")/../shared/summarize_probe.py" "$OUT/probe" tiktok
log "完成 -> $OUT"
