#!/usr/bin/env bash
# 查看 pr-watch 状态：进程、锁、最近日志、open PR 与评审状态。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
STATE=$PRWATCH_HOME
REPO=$(git -C "$REPO_ROOT" remote get-url origin \
  | sed -E 's#^git@github\.com:##; s#^https://github\.com/##; s#\.git$##')

echo "== 进程 =="
if [ -f "$STATE/daemon.pid" ] && kill -0 "$(cat "$STATE/daemon.pid")" 2>/dev/null; then
  pid=$(cat "$STATE/daemon.pid")
  echo "运行中 pid=$pid"
  ps -o pid,etime,cmd -p "$pid" 2>/dev/null | tail -1
else
  echo "未运行"
fi

echo
echo "== 锁 =="
if flock -n "$STATE/.lock" true 2>/dev/null; then echo "空闲"; else echo "被占用（守护进程在跑）"; fi

echo
echo "== 配置 =="
printf 'model=%s poll=%ss policy=%s rate=%s/h\n' "$REVIEW_MODEL" "$POLL_SECONDS" "$MERGE_POLICY" "$MAX_REVIEWS_PER_HOUR"

echo
echo "== 最近日志 =="
tail -10 "$STATE/watch.log" 2>/dev/null || echo "（无）"

echo
echo "== open PR =="
gh pr list --repo "$REPO" --state open --limit 20 \
  --json number,title,headRefOid,labels,isDraft \
  --jq '.[] | "#\(.number) \(.title[0:40]) head=\(.headRefOid[0:8]) labels=\([.labels[].name]|join(",")) draft=\(.isDraft)"' 2>/dev/null || echo "（查询失败）"

echo
echo "== 待合并（只重试门禁）=="
if [ -s "$STATE/pending-merge.txt" ]; then cat "$STATE/pending-merge.txt"; else echo "（无）"; fi

echo
echo "== 最近审计 =="
tail -5 "$STATE/audit.jsonl" 2>/dev/null || echo "（无）"
