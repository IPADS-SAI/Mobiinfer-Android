#!/bin/bash
# lib.sh —— front-app 公共函数（被各 App 的 launch/probe/with_llm 脚本 source）
#
# 约定：
#   DEVICE_SERIAL   adb 序列号；不设时用 adb 默认设备（多设备必须设）
#   D               手机侧工具目录（默认 /data/local/tmp/sp）
#   FRONT_APP_HOME  本目录的上一级（tools/front-app）
#   PHONE_HOME      Termux HOME（手机上）
#   PHONE_SSH       Termux 的 ssh host；只有跑 llama 时才必须
#
# 采集能力边界（见 README.md §1）：
#   能拿到：每 App CPU(jiffies)、RSS、全设备 CPU busy/iowait、llama 的 preadv 时序
#   拿不到：PMU 内存带宽事件、每 App 块层字节数（需 root）
set -u

ADB=(adb)
if [ -n "${DEVICE_SERIAL:-}" ]; then
  ADB+=(-s "$DEVICE_SERIAL")
fi
D=${FRONT_APP_DEVICE_DIR:-/data/local/tmp/sp}
SHARED=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRONT_APP_HOME=$(cd "$SHARED/.." && pwd)
REPO=$(cd "$FRONT_APP_HOME/../.." && pwd)
PHONE_HOME=${PHONE_HOME:-/data/data/com.termux/files/home}
# llama-cli / 模型在 Termux 私有目录，adb shell(uid 2000) 进不去，
# 所以一切 llama 相关命令必须走 ssh（Termux uid）。
PHONE_SSH=${PHONE_SSH:-}
phone_ssh() {
  [ -n "$PHONE_SSH" ] || die "需要 PHONE_SSH（Termux 的 ssh host/alias），例如 export PHONE_SSH=vivo"
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$PHONE_SSH" "$@"
}
PHONE_MOE=${PHONE_MOE:-$PHONE_HOME/csm/moe-exp}

sh_()  { "${ADB[@]}" shell "$@"; }
log()  { echo "[$(date +%T)] $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

focused_pkg() {
  sh_ 'dumpsys window 2>/dev/null | grep -m1 mCurrentFocus' \
    | sed -n 's/.*u0 \([^/ ]*\).*/\1/p' | tr -d '\r'
}

# ---------- 工具就位 ----------
fa_install_tools() {
  mkdir -p "$SHARED/build"
  [ -x "$SHARED/build.sh" ] || die "缺少 $SHARED/build.sh（设备侧探针构建入口）"
  # 单一构建入口：lib.sh 和 build.sh 不能再各自维护一套 flags/ABI。
  ANDROID_NDK="${ANDROID_NDK:-}" ANDROID_API="${ANDROID_API:-35}" \
    "$SHARED/build.sh" runp directio iotrace >&2 || die "设备侧工具编译失败"
  sh_ "mkdir -p $D/out" >/dev/null 2>&1
  "${ADB[@]}" push "$SHARED/build/runp" "$D/runp" >/dev/null || die "push runp"
  "${ADB[@]}" push "$SHARED/build/directio" "$D/directio" >/dev/null || die "push directio"
  "${ADB[@]}" push "$SHARED/build/iotrace" "$D/iotrace" >/dev/null || die "push iotrace"
  "${ADB[@]}" push "$SHARED/sp_helper.sh" "$D/sp_helper.sh" >/dev/null || die "push sp_helper"
  sh_ "chmod 755 $D/runp $D/directio $D/iotrace $D/sp_helper.sh" >/dev/null
}

# ---------- App 状态 ----------
fa_uid()      { sh_ "dumpsys package $1 2>/dev/null | awk '/appId=/{sub(/.*=/,\"\");print;exit}'" | tr -d '\r'; }
fa_pids()     { sh_ "pidof $1" | tr -d '\r'; }
fa_force_stop() { sh_ "export PATH=/system/bin:\$PATH; am force-stop $1" >/dev/null 2>&1; }
fa_wake()     { sh_ 'input keyevent KEYCODE_WAKEUP; input keyevent KEYCODE_MENU' >/dev/null 2>&1; }

# 拉起 App 并等到“启动完成”：前台是该包，且 RSS 在连续 3 次采样里变化 < 3%
#   fa_launch_and_settle <pkg> <max_wait_s>
fa_launch_and_settle() {
  local pkg=$1 maxs=${2:-60} prev=0 stable=0 rss i
  fa_wake
  fa_force_stop "$pkg"; sleep 2
  log "拉起 $pkg ..."
  sh_ "export PATH=/system/bin:\$PATH; monkey -p $pkg -c android.intent.category.LAUNCHER 1" >/dev/null 2>&1
  for ((i=0; i<maxs; i++)); do
    sleep 1
    [ "$(focused_pkg)" = "$pkg" ] || continue
    rss=$(sh_ "sh $D/sp_helper.sh rss $pkg" | tr -d '\r'); rss=${rss:-0}
    if [ "$rss" -gt 0 ] && [ "$prev" -gt 0 ]; then
      local d=$(( rss > prev ? rss - prev : prev - rss ))
      if [ $(( d * 100 / prev )) -lt 3 ]; then stable=$((stable+1)); else stable=0; fi
      [ $stable -ge 3 ] && { log "$pkg 启动完成：rss=$((rss/1024))MiB (${i}s)"; return 0; }
    fi
    prev=$rss
  done
  log "WARN: $pkg 在 ${maxs}s 内未稳定，按当前状态继续"
  return 1
}

# 守护：App 掉出前台就返回 1（用于中止实验，避免测到"后台 App"）
fa_is_foreground() { [ "$(focused_pkg)" = "$1" ]; }

# ---------- 采集 ----------
# fa_sample <pkg> <秒> <outdir> <tag>
#   产出 dev.csv（runp：设备 busy/iowait + App CPU）、app.csv（RSS/网络/CPU）、
#   另起 watchdog 记录前台变化
fa_sample() {
  local pkg=$1 dur=$2 out=$3 tag=$4
  mkdir -p "$out"
  # 清理上次遗留的采样进程：游戏高负载下 dumpsys 会变慢，残留进程会互相拖累
  sh_ 'pkill -f "sp_helper.sh sample" 2>/dev/null; pkill -f "runp /data/local/tmp/sp/out/" 2>/dev/null' >/dev/null 2>&1
  sleep 1
  sh_ "rm -f $D/out/$tag.dev.csv $D/out/$tag.app.csv $D/out/$tag.fg.txt" >/dev/null 2>&1
  ( sh_ "$D/runp $D/out/$tag.dev.csv $((dur + 6)) --pkg $pkg" >/dev/null 2>&1 ) & local p1=$!
  ( sh_ "sh $D/sp_helper.sh sample $pkg $((dur + 2)) $D/out/$tag.app.csv" >/dev/null 2>&1 ) & local p2=$!
  ( local i=0; while [ $i -lt $((dur + 3)) ]; do
      echo "$(date +%s) $(focused_pkg)" >> "$out/$tag.fg.txt"; sleep 1; i=$((i+1)); done ) & local p3=$!
  wait $p1 $p2 $p3
  "${ADB[@]}" pull "$D/out/$tag.dev.csv" "$out/$tag.dev.csv" >/dev/null 2>&1
  "${ADB[@]}" pull "$D/out/$tag.app.csv" "$out/$tag.app.csv" >/dev/null 2>&1
}

# 两次采样的“同一秒”对齐参考点：返回当前 epoch ms
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
