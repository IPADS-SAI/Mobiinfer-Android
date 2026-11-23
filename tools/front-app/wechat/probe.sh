#!/bin/bash
# wechat/probe.sh [--dur 120] —— 单 App 采集（不跑 LLM），得到微信的 CPU/RSS/iowait 基线
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
DUR=120
while [ $# -gt 0 ]; do case "$1" in --dur) DUR=$2; shift 2;; *) die "未知参数 $1";; esac; done
PKG=com.tencent.mm
OUT=$FRONT_APP_HOME/artifacts/wechat-probe-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
fa_install_tools
fa_launch_and_settle "$PKG" 60
log "采集 ${DUR}s ..."
( fa_sample "$PKG" "$DUR" "$OUT" probe )
fa_force_stop "$PKG"
python3 "$(dirname "$0")/../shared/summarize_probe.py" "$OUT/probe" wechat
log "完成 -> $OUT"
