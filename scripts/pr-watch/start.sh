#!/usr/bin/env bash
# 启动 pr-watch 守护进程（脱离当前 shell，关闭终端也继续跑）。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
STATE=$PRWATCH_HOME
mkdir -p "$STATE"
export PRWATCH_HOME

if [ -f "$STATE/daemon.pid" ] && kill -0 "$(cat "$STATE/daemon.pid")" 2>/dev/null; then
  echo "已在运行，pid=$(cat "$STATE/daemon.pid")"
  exit 0
fi

LOG="$STATE/daemon.log"
PRWATCH_HOME="$PRWATCH_HOME" setsid nohup "$SCRIPT_DIR/watch.sh" >> "$LOG" 2>&1 < /dev/null &
sleep 2

if [ -f "$STATE/daemon.pid" ] && kill -0 "$(cat "$STATE/daemon.pid")" 2>/dev/null; then
  echo "已启动，pid=$(cat "$STATE/daemon.pid")，日志：$LOG"
else
  echo "启动失败，最近日志："
  tail -20 "$LOG" 2>/dev/null || true
  exit 1
fi
