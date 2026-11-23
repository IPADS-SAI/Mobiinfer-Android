#!/bin/bash
# wechat/appwork.sh <stop-file> [时长秒=0] —— 微信前台闲置负载；0 = 一直跑到 stop 文件
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
PKG=com.tencent.mm
STOP=$1; DUR=${2:-0}
fa_install_tools
fa_launch_and_settle "$PKG" 60 || true
touch "$1.ready"
# 保持前台：定期确认仍在微信，掉出去就重新拉起一次（记为事件）
T0=$(date +%s)
while [ ! -f "$STOP" ]; do
  el=$(( $(date +%s) - T0 ))
  [ "$DUR" -gt 0 ] && [ "$el" -ge "$DUR" ] && break
  fa_is_foreground "$PKG" || { log "微信掉出前台，重新拉起"; fa_wake; sh_ "export PATH=/system/bin:\$PATH; monkey -p $PKG -c android.intent.category.LAUNCHER 1" >/dev/null 2>&1; }
  sleep 3
done
