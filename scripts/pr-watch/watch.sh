#!/usr/bin/env bash
# pr-watch 守护进程：轮询本仓库 open PR，有新 head 就跑 review-one.sh。
#
# 启动：scripts/pr-watch/start.sh
# 停止：scripts/pr-watch/stop.sh
# 状态：scripts/pr-watch/status.sh
# 单次：scripts/pr-watch/watch.sh --once
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
STATE=$PRWATCH_HOME
mkdir -p "$STATE"

exec 9>"$STATE/.lock"
if ! flock -n 9; then
  echo "pr-watch 已在运行（$STATE/.lock）" >&2
  exit 0
fi

echo $$ > "$STATE/daemon.pid"
trap 'rm -f "$STATE/daemon.pid"' EXIT
trap 'exit 0' INT TERM

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATE/watch.log"; }

REPO=$(git -C "$REPO_ROOT" remote get-url origin \
  | sed -E 's#^git@github\.com:##; s#^https://github\.com/##; s#\.git$##')

SEEN="$STATE/seen.txt"
RATE="$STATE/rate.txt"
PENDING="$STATE/pending-merge.txt"
touch "$SEEN" "$RATE" "$PENDING"

once=0
[ "${1:-}" = "--once" ] && once=1

log "启动：repo=$REPO poll=${POLL_SECONDS}s model=$REVIEW_MODEL policy=$MERGE_POLICY"

# 已 approve 但门禁暂未通过的 PR：只重试合并门禁，不重跑评审。
process_pending() {
  [ -s "$PENDING" ] || return 0
  : > "$PENDING.tmp"
  while read -r p s o; do
    [ -z "$p" ] && continue
    cur=$(gh pr view "$p" --repo "$REPO" --json state,headRefOid,labels \
          --jq '.state + " " + .headRefOid + " " + ([.labels[].name] | join(","))' 2>/dev/null || echo "ERR")
    st=${cur%% *}; rest=${cur#* }; sh=${rest%% *}; lbs=${rest#* }
    if [ "$st" != OPEN ] || [ "$sh" != "$s" ]; then
      log "待合并 #$p 已关闭或 head 变化，出队"
      continue
    fi
    case ",$lbs," in
      *,review:again,*)
        log "待合并 #$p 要求重审，出队"
        gh pr edit "$p" --repo "$REPO" --remove-label review:again >/dev/null 2>&1 || true
        continue;;
    esac
    if "$SCRIPT_DIR/merge-ready.sh" "$p" "$s" "$o"; then
      echo "$p:$s" >> "$SEEN"
      log "待合并 #$p 已合并"
    else
      printf '%s %s %s\n' "$p" "$s" "$o" >> "$PENDING.tmp"
    fi
  done < "$PENDING"
  mv "$PENDING.tmp" "$PENDING"
}

tick() {
  local prs numbers n sha key labels
  process_pending
  if ! prs=$(gh pr list --repo "$REPO" --state open --limit 50 \
        --json number,headRefOid,isDraft,labels,isCrossRepository 2>>"$STATE/watch.log"); then
    log "gh pr list 失败，60s 后重试"
    return 1
  fi
  numbers=$(jq -r '.[].number' <<<"$prs" | tr '\n' ' ')

  # 清理已关闭 PR 的 seen 记录
  if [ -s "$SEEN" ]; then
    awk -v open=" $numbers " '{ split($0, a, ":"); if (index(open, " " a[1] " ")) print }' "$SEEN" > "$SEEN.tmp" || true
    mv "$SEEN.tmp" "$SEEN"
  fi

  while IFS= read -r pr; do
    [ -z "$pr" ] && continue
    n=$(jq -r '.number' <<<"$pr")
    sha=$(jq -r '.headRefOid' <<<"$pr")
    labels=$(jq -r '[.labels[].name] | join(",")' <<<"$pr")
    key="$n:$sha"

    if [ "$(jq -r '.isDraft' <<<"$pr")" = true ]; then continue; fi
    if [ "$(jq -r '.isCrossRepository' <<<"$pr")" = true ] && [ "$ALLOW_FORK_PR" != true ]; then continue; fi
    case ",$labels," in
      *,hold,*|*,do-not-merge,*) continue;;
    esac

    if grep -q "^$n " "$PENDING"; then
      case ",$labels," in *,review:again,*) ;; *) continue;; esac
    fi

    # 基础设施失败（网络等）退避：同一 head 连续失败 3 次就先放下，
    # 不再每 120s 重试；head 一变计数自动清零。
    fails=$(cat "$STATE/fail-$n" 2>/dev/null || true)
    f_sha=${fails%% *}
    f_cnt=${fails##* }
    [ "$f_sha" = "$sha" ] || f_cnt=0
    if [ "${f_cnt:-0}" -ge 3 ]; then continue; fi

    if grep -qxF "$key" "$SEEN"; then
      case ",$labels," in
        *,review:again,*)
          gh pr edit "$n" --repo "$REPO" --remove-label review:again >/dev/null 2>&1 || true
          ;;
        *) continue;;
      esac
    fi

    # 每小时限额
    cutoff=$(( $(date +%s) - 3600 ))
    awk -v c="$cutoff" '$1 > c' "$RATE" > "$RATE.tmp" 2>/dev/null || true
    mv "$RATE.tmp" "$RATE"
    if [ "$(wc -l < "$RATE" | tr -d ' ')" -ge "$MAX_REVIEWS_PER_HOUR" ]; then
      log "已达每小时上限 $MAX_REVIEWS_PER_HOUR，跳过 #$n"
      continue
    fi

    log "发现待审 PR #$n @ ${sha:0:8}"
    # 总预算要覆盖 review-one 里实际会跑的三段（configure/build/ctest）+ 评审。
    wants_build=$RUN_BUILD_CHECK
    case ",$labels," in *",run-build,"*) wants_build=1;; esac
    # 评审可能跑多轮（材料不足自动补一轮），外层预算要按轮数放大，
    # 否则第二轮会被 timeout 掐断、按基础设施失败计费并反复重试。
    review_attempts=${PACKET_MAX_ATTEMPTS:-2}
    [ "$REVIEW_AUTO_EXPAND" = 1 ] || review_attempts=1
    # 每轮不只是正式评审：review-packet.sh 还会为低风险文件跑一次摘要模型
    # （PACKET_SUMMARY=1 时，最长 PACKET_SUMMARY_TIMEOUT）。只算 REVIEW_TIMEOUT 会
    # 让两轮在"每个内层调用都没超时"的情况下被外层 timeout 掐断。
    per_round=$((REVIEW_TIMEOUT + (PACKET_SUMMARY == 1 ? PACKET_SUMMARY_TIMEOUT : 0)))
    budget=$((per_round * review_attempts + 600))
    if [ "$wants_build" = 1 ]; then budget=$((budget + 3 * BUILD_TIMEOUT)); fi
    set +e
    timeout "$budget" "$SCRIPT_DIR/review-one.sh" "$n" "$sha"
    rc=$?
    set -e
    case "$rc" in
      0)  echo "$key" >> "$SEEN"; date +%s >> "$RATE"; rm -f "$STATE/fail-$n";;
      10) printf '%s %s %s\n' "$n" "$sha" "$STATE/pr-$n-$sha" >> "$PENDING"
          log "PR #$n 已 approve 但门禁未过，进入待合并队列"; date +%s >> "$RATE"; rm -f "$STATE/fail-$n";;
      11) echo "$key" >> "$SEEN"; date +%s >> "$RATE"; rm -f "$STATE/fail-$n";;
      3)  log "PR #$n 本轮跳过";;
      12) f_cnt=$(( ${f_cnt:-0} + 1 ))
          printf '%s %s\n' "$sha" "$f_cnt" > "$STATE/fail-$n"
          # 这一轮**真的调用过模型**（只是超时/额度/网络失败），必须计入小时限额：
          # 不记 seen（允许重试），但也不该让模型故障期间无限重试绕过 MAX_REVIEWS_PER_HOUR。
          date +%s >> "$RATE"
          log "PR #$n 评审未完成（额度/网络/超时，rc=12，第 $f_cnt 次），不记 seen、计入限额，下轮重试";;
      *)  f_cnt=$(( ${f_cnt:-0} + 1 ))
          printf '%s %s\n' "$sha" "$f_cnt" > "$STATE/fail-$n"
          log "PR #$n 基础设施失败（rc=$rc，第 $f_cnt 次），不占模型配额";;
    esac
    if [ "$once" = 1 ]; then break; fi
  done < <(jq -c '.[]' <<<"$prs")
}

while :; do
  tick || true
  [ "$once" = 1 ] && break
  sleep "$POLL_SECONDS"
done

log "退出"
