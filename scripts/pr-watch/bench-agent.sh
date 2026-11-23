#!/usr/bin/env bash
# Stage 1：让便宜的复现 agent（默认 group/wb-deepseek）按 skills/bench 跑出 bench 证据。
#
# 用法：
#   bench-agent.sh <pr> <sha> [--check] [--wait-lock SEC] [--model M] [--out DIR]
#
#   --check        只检查前置条件（设备锁、SDK/NDK、codex），不调模型
#   --wait-lock N  设备被占时最多等 N 秒（默认 0：立刻退出，不排队）
#   --model M      覆盖模型（默认 $BENCH_AGENT_MODEL）
#
# 产出：
#   ~/.local/state/pr-watch/bench/pr-<N>-<sha>.json           （bench-result.json）
#   ~/.local/state/pr-watch/bench/pr-<N>-<sha>-evidence/      （manifest + run-*.log）
#
# 退出码：0 成功；2 前置条件不满足；3 设备被占；4 构建/复现失败；5 校验不通过
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PR=${1:?usage: bench-agent.sh <pr> <sha> [--check]}
SHA=${2:?usage: bench-agent.sh <pr> <sha> [--check]}
shift 2

MODE=run
WAIT_LOCK=0
MODEL=$BENCH_AGENT_MODEL
OUTDIR_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check; shift;;
    --wait-lock) WAIT_LOCK=${2:?}; shift 2;;
    --model) MODEL=${2:?}; shift 2;;
    --out) OUTDIR_OVERRIDE=${2:?}; shift 2;;
    *) echo "未知参数：$1" >&2; exit 2;;
  esac
done

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
STATE=$PRWATCH_HOME
mkdir -p "$STATE/bench" "$STATE/bench-wt"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

# 本机 cmake/ninja 不在默认 PATH（systemd 下尤其），补上
for d in "$HOME"/env/android/cmake/*/bin; do
  [ -x "$d/cmake" ] && PATH="$d:$PATH"
done
export PATH

# Hexagon / NDK：优先环境变量，其次在 ~/env 下取最高版本（可用 env 覆盖）
if [ -z "${HEXAGON_SDK_ROOT:-}" ] || [ ! -d "${HEXAGON_SDK_ROOT:-/nonexistent}" ]; then
  HEXAGON_SDK_ROOT=$(ls -d "$HOME"/env/hexagon/* 2>/dev/null | sort -V | tail -1)
fi
if [ -z "${HEXAGON_TOOLS_ROOT:-}" ] || [ ! -d "${HEXAGON_TOOLS_ROOT:-/nonexistent}" ]; then
  HEXAGON_TOOLS_ROOT=$(ls -d "$HEXAGON_SDK_ROOT"/tools/HEXAGON_Tools/* 2>/dev/null | sort -V | tail -1)
fi
if [ -z "${ANDROID_NDK:-}" ] || [ ! -d "${ANDROID_NDK:-/nonexistent}" ]; then
  ANDROID_NDK=$(ls -d "$HOME"/env/android/ndk/* 2>/dev/null | sort -V | tail -1)
fi
export HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT ANDROID_NDK BENCH_LOCK_WAIT

WT="$STATE/bench-wt/pr-$PR"
OUTDIR=${OUTDIR_OVERRIDE:-$STATE/bench/pr-$PR-$SHA.d}

fail() { echo "bench-agent: $*" >&2; exit 2; }

# ---------- 前置条件 ----------
command -v codex >/dev/null 2>&1 || fail "找不到 codex"
command -v python3 >/dev/null 2>&1 || fail "找不到 python3"
command -v cmake >/dev/null 2>&1 || fail "找不到 cmake（$HOME/env/android/cmake/*/bin）"
command -v ninja >/dev/null 2>&1 || fail "找不到 ninja"
[ -n "${HEXAGON_SDK_ROOT:-}" ] && [ -d "$HEXAGON_SDK_ROOT" ] || fail "HEXAGON_SDK_ROOT 无效"
[ -n "${ANDROID_NDK:-}" ] && [ -d "$ANDROID_NDK" ] || fail "ANDROID_NDK 无效"
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 vivo true >/dev/null 2>&1 \
  || fail "ssh vivo 不通"

device_free() {
  timeout 20 ssh -o BatchMode=yes vivo 'cd ~/csm/moe-exp && ./devlock -n -q true' >/dev/null 2>&1
}

if ! device_free; then
  if [ "$WAIT_LOCK" -gt 0 ]; then
    log "设备被占，最多等 ${WAIT_LOCK}s"
    waited=0
    while ! device_free; do
      [ "$waited" -ge "$WAIT_LOCK" ] && { log "等锁超时"; exit 3; }
      sleep 20; waited=$((waited + 20))
    done
  else
    log "设备被占（devlock -n 失败）。用 --wait-lock N 排队，或稍后重试"
    exit 3
  fi
fi

# ---------- 基线缓存 ----------
BASELINE_DIR=$("$SCRIPT_DIR/bench-baseline.sh" path 2>/dev/null || true)
if [ -z "$BASELINE_DIR" ] && [ "$MODE" != check ] && [ "$BENCH_BASELINE_BUILD" = 1 ]; then
  log "没有 main 基线缓存，先构建一次（按 main sha 复用）"
  "$SCRIPT_DIR/bench-baseline.sh" ensure || log "基线构建失败，交给 agent 自建"
  BASELINE_DIR=$("$SCRIPT_DIR/bench-baseline.sh" path 2>/dev/null || true)
fi
export BASELINE_DIR

if [ "$MODE" = check ]; then
  log "前置条件 OK"
  echo "model:   $MODEL"
  echo "worktree: $WT"
  echo "outdir:   $OUTDIR"
  echo "sdk:      $HEXAGON_SDK_ROOT"
  echo "tools:    $HEXAGON_TOOLS_ROOT"
  echo "ndk:      $ANDROID_NDK"
  echo "device:   free"
  echo "baseline: ${BASELINE_DIR:-（无缓存，首次会构建）}"
  exit 0
fi

# ---------- worktree ----------
mkdir -p "$OUTDIR/evidence"
git -C "$REPO_ROOT" fetch -q --no-tags origin "pull/$PR/head"
git -C "$REPO_ROOT" fetch -q --no-tags origin "$MERGE_BASE_BRANCH"
if [ -e "$WT/.git" ]; then
  git -C "$WT" checkout -q --detach "$SHA"
else
  git -C "$REPO_ROOT" worktree add -q --detach --force "$WT" "$SHA"
fi
git -C "$WT" reset -q --hard "$SHA"
git -C "$WT" clean -qffdx

TITLE=$(gh pr view "$PR" --repo "$(git -C "$REPO_ROOT" remote get-url origin | sed -E 's#^git@github\.com:##; s#^https://github\.com/##; s#\.git$##')" --json title -q .title 2>/dev/null || echo "")

{
  printf '# 复现任务\n\n'
  printf -- '- PR：#%s《%s》\n' "$PR" "$TITLE"
  printf -- '- head：`%s`\n' "$SHA"
  printf -- '- worktree（只读源码，禁止修改）：`%s`\n' "$WT"
  printf -- '- 输出目录（**唯一可写位置**）：`%s`\n' "$OUTDIR"
  printf -- '- 设备：vivo（`ssh vivo`），必须用 `~/csm/moe-exp/devlock` 独占\n'
  printf -- '- 取锁命令固定为 `./devlock -w %s`；退出码 75 表示等锁超时，写 failure.md 后退出，不要无限等待\n' "$BENCH_LOCK_WAIT"
  printf -- '- 基线（A 臂）：`%s`\n' "${BASELINE_DIR:-（无缓存，需要自己构建 origin/$MERGE_BASE_BRANCH）}"
  printf -- '- 基线已构建好，**不要重建**；B 臂才从本 worktree 构建\n\n' 
  printf '## 输出文件\n\n'
  printf -- '- `%s/bench-result.json`\n' "$OUTDIR"
  printf -- '- `%s/manifest.json`\n' "$OUTDIR"
  printf -- '- `%s/evidence/run-*.log`\n\n' "$OUTDIR"
  cat "$SCRIPT_DIR/bench-prompt.md"
} > "$OUTDIR/prompt.md"

log "调用 $MODEL 复现 PR #$PR @ ${SHA:0:8}（timeout ${BENCH_AGENT_TIMEOUT}s）"
set +e
env -u GH_TOKEN -u GITHUB_TOKEN -u GH_ENTERPRISE_TOKEN -u GITHUB_ENTERPRISE_TOKEN \
  CODEX_HOME="$HOME/.codex" \
  timeout "$BENCH_AGENT_TIMEOUT" codex exec \
    -C "$WT" \
    -c approval_policy=never \
    -m "$MODEL" \
    --json - \
    < "$OUTDIR/prompt.md" > "$OUTDIR/agent.jsonl" 2>&1
rc=$?
set -e
log "agent 结束（rc=$rc），日志：$OUTDIR/agent.jsonl"

RESULT="$OUTDIR/bench-result.json"
if [ ! -s "$RESULT" ]; then
  log "没有产出 bench-result.json，见 $OUTDIR/failure.md"
  exit 4
fi

if ! python3 "$SCRIPT_DIR/verify_bench.py" "$RESULT" --pr "$PR" --sha "$SHA" \
      --min-gain "$BENCH_MIN_GAIN" --noise-mult "$BENCH_NOISE_MULT" \
      --min-rounds "$BENCH_MIN_ROUNDS" \
      --prefill-max-regression "$BENCH_PREFILL_MAX_REGRESSION" \
      --rss-max-regression "$BENCH_RSS_MAX_REGRESSION" \
      --max-age-hours "$BENCH_MAX_AGE_HOURS" \
      --evidence "$OUTDIR"; then
  log "统计/证据校验不通过"
  exit 5
fi

"$SCRIPT_DIR/bench-record.sh" "$PR" "$SHA" "$RESULT" "$OUTDIR"
