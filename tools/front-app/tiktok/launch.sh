#!/bin/bash
# tiktok/launch.sh —— 拉起抖音并进入推荐流（检测 ViewPager）
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
PKG=com.ss.android.ugc.aweme
fa_install_tools
fa_wake
fa_force_stop "$PKG"; sleep 2
log "拉起抖音 ..."
sh_ "export PATH=/system/bin:\$PATH; monkey -p $PKG -c android.intent.category.LAUNCHER 1" >/dev/null 2>&1
for i in $(seq 1 60); do
  sleep 1
  [ "$(focused_pkg)" = "$PKG" ] || continue
  for _ in 1 2 3; do
    sh_ 'export PATH=/system/bin:$PATH; uiautomator dump /data/local/tmp/dy.xml >/dev/null 2>&1'
    if sh_ 'cat /data/local/tmp/dy.xml 2>/dev/null' | grep -q 'ViewPager'; then
      log "抖音进入推荐流（${i}s）"; exit 0
    fi
    sleep 1
  done
done
log "WARN: 未检测到推荐流（可能有弹窗，需人工处理）"; exit 1
