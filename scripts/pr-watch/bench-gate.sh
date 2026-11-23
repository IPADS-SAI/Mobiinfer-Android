#!/usr/bin/env bash
# bench 门禁：读 $PRWATCH_HOME/bench/pr-<N>-<sha>.json 并跑 verify_bench.py。
#
# 用法：bench-gate.sh <pr> <sha> [result-file]
# 退出码：0 通过/跳过；1 不通过；2 输入不合法
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PR=${1:?usage: bench-gate.sh <pr> <sha> [result-file]}
SHA=${2:?usage: bench-gate.sh <pr> <sha> [result-file]}
PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
RESULT=${3:-$PRWATCH_HOME/bench/pr-$PR-$SHA.json}

if [ "$BENCH_GATE" = off ]; then
  echo "bench 门禁：BENCH_GATE=off，跳过"
  exit 0
fi

if [ ! -s "$RESULT" ]; then
  echo "bench 门禁 FAIL：没有结果文件 $RESULT"
  echo "  跑完 bench 后执行：scripts/pr-watch/bench-record.sh $PR $SHA <bench-result.json>"
  exit 1
fi

CUR_MAIN=$(git -C "$REPO_ROOT" rev-parse "origin/$MERGE_BASE_BRANCH" 2>/dev/null || true)
MAIN_ARGS=()
if [ -n "$CUR_MAIN" ]; then
  MAIN_ARGS=(--current-main "$CUR_MAIN")
  [ "$BENCH_REQUIRE_CURRENT_MAIN" = 1 ] && MAIN_ARGS+=(--require-current-main)
fi

EVIDENCE_ARGS=()
if [ "$BENCH_REQUIRE_EVIDENCE" = 1 ]; then
  EVIDENCE_ARGS=(--evidence "$PRWATCH_HOME/bench/pr-$PR-$SHA-evidence")
fi

exec python3 "$SCRIPT_DIR/verify_bench.py" "$RESULT" \
  --pr "$PR" --sha "$SHA" \
  --min-gain "$BENCH_MIN_GAIN" \
  --noise-mult "$BENCH_NOISE_MULT" \
  --min-rounds "$BENCH_MIN_ROUNDS" \
  --prefill-max-regression "$BENCH_PREFILL_MAX_REGRESSION" \
  --rss-max-regression "$BENCH_RSS_MAX_REGRESSION" \
  --max-age-hours "$BENCH_MAX_AGE_HOURS" "${MAIN_ARGS[@]}" "${EVIDENCE_ARGS[@]}"
