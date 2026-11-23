#!/usr/bin/env bash
# 停止 pr-watch 守护进程（连同整棵 worker 进程树）。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
PIDFILE="$PRWATCH_HOME/daemon.pid"

if [ ! -f "$PIDFILE" ]; then
  echo "未运行（无 pid 文件）"; exit 0
fi

pid=$(cat "$PIDFILE")
if ! kill -0 "$pid" 2>/dev/null; then
  echo "进程 $pid 已不存在，清理 pid 文件"; rm -f "$PIDFILE"; exit 0
fi

# watch.sh 由 setsid 启动：pid 同时是 session id 与 pgid。
# GNU timeout 会给子进程新建进程组（但不换 session），所以只杀进程组会漏掉
# timeout 的子进程；必须连 session 一起收。
alive() { kill -0 -- "-$pid" 2>/dev/null || pgrep -s "$pid" >/dev/null 2>&1; }

kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
pkill -TERM -s "$pid" 2>/dev/null || true

for _ in $(seq 1 40); do
  alive || break
  sleep 0.5
done

if alive; then
  echo "优雅退出超时，强制结束 session $pid"
  kill -KILL -- "-$pid" 2>/dev/null || true
  pkill -KILL -s "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    alive || break
    sleep 0.5
  done
fi

if alive; then
  echo "警告：session $pid 仍有成员存活，请手动检查：ps -o pid,pgid,sid,cmd -s $pid" >&2
  exit 1
fi

rm -f "$PIDFILE"
echo "已停止 session $pid"
