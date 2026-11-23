#!/bin/bash
# genshin/appwork.sh <stop-file> <时长秒>
# 原神在前台"真实操作"的负载：周期性转视角 + 摇杆移动 + 普攻，而不是静止挂机。
# ⚠️ 横屏坐标 3168x1440（见 launch.sh 顶部说明）。
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
PKG=com.miHoYo.Yuanshen
STOP=$1; DUR=${2:-180}

READY="$1.ready"
# 若不在世界里，先拉起并进世界（约 90-120s）
if [ "$(focused_pkg)" != "$PKG" ]; then
  log "原神不在前台，执行 launch.sh --enter ..."
  bash "$(dirname "$0")/launch.sh" --enter >> "$(dirname "$1")/appwork-launch.log" 2>&1 || log "WARN: launch 返回非 0"
fi
# 校验是否真的进了世界：世界内 RSS ~2.9 GiB，标题页只有 ~1.7 GiB
rss_of() { sh_ "sh $D/sp_helper.sh rss $PKG" 2>/dev/null | tr -d '\r'; }
r=$(rss_of); r=${r:-0}
if [ "$r" -lt $((2500*1024)) ]; then
  log "RSS $((r/1024)) MiB 偏低（像标题页），补点中间推进"
  sh_ 'input tap 1584 720' >/dev/null 2>&1; sleep 25
  sh_ 'input tap 1584 1330' >/dev/null 2>&1; sleep 40
  r=$(rss_of); r=${r:-0}
fi
log "进世界校验：RSS $((r/1024)) MiB $([ "$r" -ge $((2500*1024)) ] && echo '(在世界内)' || echo '(可能仍是标题页)')"
touch "$READY"   # 通知 with_llm 可以开始采样了
log "原神已就绪，开始注入操作"
T0=$(date +%s)
n=0
while [ ! -f "$STOP" ]; do
  el=$(( $(date +%s) - T0 ))
  [ "$DUR" -gt 0 ] && [ "$el" -ge "$DUR" ] && break
  if ! fa_is_foreground "$PKG"; then log "原神掉出前台"; sleep 2; continue; fi
  sh_ 'input swipe 2400 900 1900 760 200' >/dev/null 2>&1   # 右侧转视角
  sleep 1
  sh_ 'input swipe 400 1150 520 1000 400' >/dev/null 2>&1   # 左侧摇杆移动
  sleep 1
  n=$((n+1))
done
log "原神操作结束：$n 轮"
