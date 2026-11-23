#!/bin/bash
# wechat/launch.sh [--settle] —— 冷启动微信并等到 RSS 稳定
#   --settle：等到"启动完成"（RSS 连续 3 次变化 <3%）再返回，默认等待
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
PKG=com.tencent.mm
fa_install_tools
fa_launch_and_settle "$PKG" 60
log "微信就绪：rss=$(sh_ "sh $D/sp_helper.sh rss $PKG" | tr -d '\r')KiB"
