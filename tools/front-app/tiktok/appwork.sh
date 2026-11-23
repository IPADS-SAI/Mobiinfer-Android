#!/bin/bash
# tiktok/appwork.sh <stop-file> [时长秒=120] —— 抖音前台持续刷视频；0 = 一直跑到 stop 文件
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
PKG=com.ss.android.ugc.aweme
STOP=$1; DUR=${2:-120}
bash "$(dirname "$0")/launch.sh" >/dev/null 2>&1 || log "WARN: 抖音未能确认进入推荐流，继续尝试"
touch "$1.ready"
T0=$(date +%s)
while [ ! -f "$STOP" ]; do
  el=$(( $(date +%s) - T0 ))
  [ "$DUR" -gt 0 ] && [ "$el" -ge "$DUR" ] && break
  fa_is_foreground "$PKG" || { log "抖音掉出前台，尝试拉回"; fa_wake; sh_ "export PATH=/system/bin:\$PATH; am start -n $PKG/.splash.SplashActivity" >/dev/null 2>&1; sleep 2; continue; }
  sh_ 'input swipe 720 2300 720 900 300' >/dev/null 2>&1
  sleep 5
done
