#!/usr/bin/env bash
# 合并门禁：只有 verdict=approve 且 T0 通过、head 未变、无 hold 标签时才合并。
#
# 用法：merge-ready.sh <pr-number> <head-sha> [review-out-dir]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PR=${1:?usage: merge-ready.sh <pr-number> <head-sha> [out-dir]}
SHA=${2:?usage: merge-ready.sh <pr-number> <head-sha> [out-dir]}
PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
STATE=$PRWATCH_HOME
OUT=${3:-$STATE/pr-$PR-$SHA}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

REPO=$(git -C "$REPO_ROOT" remote get-url origin \
  | sed -E 's#^git@github\.com:##; s#^https://github\.com/##; s#\.git$##')

if [ "$MERGE_POLICY" = report ]; then
  log "MERGE_POLICY=report，只审不合"; exit 0
fi

gate="$OUT/gate.json"
verdict="$OUT/verdict.json"
[ -s "$gate" ] || { log "门禁：没有 gate.json（拒绝只凭 verdict 合并）"; exit 1; }
[ -s "$verdict" ] || { log "门禁：没有 verdict.json"; exit 1; }
[ "$(jq -r '.pr' "$gate")" = "$PR" ] || { log "门禁：gate.json 的 PR 不匹配"; exit 1; }
[ "$(jq -r '.sha' "$gate")" = "$SHA" ] || { log "门禁：gate.json 的 head 不匹配"; exit 1; }
[ "$(jq -r '.base' "$gate")" = "$MERGE_BASE_BRANCH" ] \
  || { log "门禁：gate 的 base 不是 $MERGE_BASE_BRANCH"; exit 1; }
[ "$(jq -r '.t0_pass' "$gate")" = true ] || { log "门禁：T0 未通过"; exit 1; }
[ "$(jq -r '.t1' "$gate")" != "fail" ] || { log "门禁：T1 构建/测试失败"; exit 1; }
[ "$(jq -r '.conclusion' "$gate")" = approve ] || { log "门禁：结论不是 approve"; exit 1; }
[ "$(jq -r '(.policy_violations // []) | length' "$gate")" -eq 0 ] \
  || { log "门禁：存在 policy_violations"; exit 1; }
[ "$(jq -r '.blocking' "$gate")" -eq 0 ] || { log "门禁：仍有 blocking"; exit 1; }
[ "$(jq -r '.conclusion' "$verdict")" = approve ] || { log "门禁：verdict 结论不是 approve"; exit 1; }

if [ "$MERGE_POLICY" = label ]; then
  labels=$(gh pr view "$PR" --repo "$REPO" --json labels --jq '[.labels[].name] | join(",")')
  case ",$labels," in *",merge-ok,"*) :;; *) log "门禁：MERGE_POLICY=label 但缺 merge-ok 标签"; exit 1;; esac
fi

info=$(gh pr view "$PR" --repo "$REPO" \
  --json isDraft,isCrossRepository,mergeable,baseRefName,headRefOid,state,labels)
[ "$(jq -r '.state' <<<"$info")" = OPEN ] || { log "门禁：PR 不是 open"; exit 1; }
[ "$(jq -r '.isDraft' <<<"$info")" = false ] || { log "门禁：draft"; exit 1; }
if [ "$(jq -r '.isCrossRepository' <<<"$info")" = true ] && [ "$ALLOW_FORK_MERGE" != true ]; then
  log "门禁：fork PR 不自动合并（ALLOW_FORK_MERGE=false）"; exit 1
fi
[ "$(jq -r '.baseRefName' <<<"$info")" = "$MERGE_BASE_BRANCH" ] \
  || { log "门禁：目标分支不是 $MERGE_BASE_BRANCH"; exit 1; }
[ "$(jq -r '.mergeable' <<<"$info")" = MERGEABLE ] || { log "门禁：不可合并（$(jq -r .mergeable <<<"$info")）"; exit 1; }
[ "$(jq -r '.headRefOid' <<<"$info")" = "$SHA" ] || { log "门禁：head 已从 $SHA 变化"; exit 1; }

labels=$(jq -r '[.labels[].name] | join(",")' <<<"$info")
IFS=',' read -r -a hold_arr <<<"$HOLD_LABELS"
for h in "${hold_arr[@]}"; do
  [ -z "$h" ] && continue
  case ",$labels," in *",$h,"*) log "门禁：带 hold 标签 $h"; exit 1;; esac
done

changed=$(git -C "$REPO_ROOT" diff --name-only "origin/$MERGE_BASE_BRANCH...$SHA" 2>/dev/null || true)
if grep -qE '^\.github/workflows/' <<<"$changed"; then
  log "门禁：改了 workflow 文件，不自动合并"; exit 1
fi

# bench 门禁：改到运行时路径必须有可校验的 bench 证据（BENCH_GATE=auto/required）
need_bench=0
if [ "$BENCH_GATE" = required ]; then
  need_bench=1
elif [ "$BENCH_GATE" = auto ]; then
  if git -C "$REPO_ROOT" diff --name-only "origin/$MERGE_BASE_BRANCH...$SHA" 2>/dev/null \
       | grep -qE '^(src|ggml|tools|common|examples)/'; then
    need_bench=1
  fi
fi
if [ "$need_bench" = 1 ]; then
  if ! "$SCRIPT_DIR/bench-gate.sh" "$PR" "$SHA"; then
    log "门禁：bench 证据不足/未达标（$PRWATCH_HOME/bench/pr-$PR-$SHA.json）"
    exit 1
  fi
fi

case "$MERGE_METHOD" in
  squash) mflag=--squash;;
  merge)  mflag=--merge;;
  rebase) mflag=--rebase;;
  *) log "未知 MERGE_METHOD=$MERGE_METHOD"; exit 1;;
esac

subject=$(jq -r '.title' "$OUT/meta.json" 2>/dev/null || gh pr view "$PR" --repo "$REPO" --json title -q .title)
log "合并 PR #$PR（$SHA，$MERGE_METHOD）"
if gh pr merge "$PR" --repo "$REPO" "$mflag" --delete-branch \
     --match-head-commit "$SHA" --subject "$subject (#$PR)"; then
  jq -nc --arg ts "$(date -Is)" --argjson pr "$PR" --arg sha "$SHA" \
    '{ts:$ts,pr:$pr,sha:$sha,action:"merged",method:"'"$MERGE_METHOD"'"}' >> "$STATE/audit.jsonl"
  gh pr comment "$PR" --repo "$REPO" \
    --body "已由 pr-watch 按 \`MERGE_POLICY=$MERGE_POLICY\` 合并（squash，head \`$SHA\`）。" \
    >/dev/null 2>&1 || true
  log "PR #$PR 合并完成"
else
  jq -nc --arg ts "$(date -Is)" --argjson pr "$PR" --arg sha "$SHA" \
    '{ts:$ts,pr:$pr,sha:$sha,action:"merge_failed"}' >> "$STATE/audit.jsonl"
  log "PR #$PR 合并失败"
  exit 1
fi
