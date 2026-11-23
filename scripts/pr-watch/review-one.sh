#!/usr/bin/env bash
# 审查单个 PR：T0 静态门禁 → codex review（gpt-6.1-sol）→ 贴评论/打标签 →（按策略）合并。
#
# 用法：review-one.sh <pr-number> [head-sha]
#   给了 head-sha 时，若远端 head 已变化则直接跳过（避免审旧代码）。
#
# 环境变量：
#   DRY_RUN=1        只产出 artifacts，不发评论、不打标签、不合并
#   PRWATCH_HOME     状态目录（默认 $XDG_STATE_HOME/pr-watch 或 ~/.local/state/pr-watch）
#   WT_DIR           worktree 根目录（默认 ~/pr-review）
#   CODEX_HOME       codex 家目录（默认 ~/.codex）
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"
# shellcheck source=sandbox-args.sh
source "$SCRIPT_DIR/sandbox-args.sh"

PR=${1:?usage: review-one.sh <pr-number> [head-sha]}
EXPECTED_SHA=${2:-}
DRY_RUN=${DRY_RUN:-0}

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
WT_DIR=${WT_DIR:-$HOME/pr-review}
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
REAL_HOME=$HOME   # 后面会把 HOME 指向临时目录，bwrap 里仍要用真实家目录
GIT_COMMON=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
GIT_COMMON=$(cd "$REPO_ROOT" && cd "$GIT_COMMON" 2>/dev/null && pwd || echo "")
STATE=$PRWATCH_HOME
mkdir -p "$STATE" "$WT_DIR"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

REPO=$(git -C "$REPO_ROOT" remote get-url origin \
  | sed -E 's#^git@github\.com:##; s#^https://github\.com/##; s#\.git$##')

# ---------- 1. 取 PR 元信息 ----------
gh_retry() {
  local i=1 delay=5
  while [ "$i" -le 5 ]; do
    if "$@"; then return 0; fi
    sleep "$delay"
    delay=$((delay * 2))
    i=$((i + 1))
  done
  return 1
}

meta=$(gh_retry gh pr view "$PR" --repo "$REPO" \
  --json number,title,headRefName,headRefOid,baseRefName,isDraft,isCrossRepository,mergeable,labels,author) \
  || { log "PR #$PR：gh pr view 连续失败（网络），本轮放弃"; exit 4; }
sha=$(jq -r '.headRefOid' <<<"$meta")
title=$(jq -r '.title' <<<"$meta")
base=$(jq -r '.baseRefName' <<<"$meta")
is_draft=$(jq -r '.isDraft' <<<"$meta")
is_fork=$(jq -r '.isCrossRepository' <<<"$meta")
labels=$(jq -r '[.labels[].name] | join(",")' <<<"$meta")

# 只审目标为 main 的 PR：否则审批绑定的 diff（相对旧 base）在 base 被改成 main 后
# 仍会被复用，未审的提交就跟着进了 main。
if [ "$base" != "$MERGE_BASE_BRANCH" ]; then
  log "PR #$PR 目标分支是 $base，不是 $MERGE_BASE_BRANCH，按策略跳过"
  exit 3
fi

if [ -n "$EXPECTED_SHA" ] && [ "$sha" != "$EXPECTED_SHA" ]; then
  log "PR #$PR head 已变化（$EXPECTED_SHA -> $sha），跳过本轮"
  exit 3
fi
if [ "$is_draft" = true ]; then log "PR #$PR 是 draft，跳过"; exit 3; fi
if ! command -v bwrap >/dev/null 2>&1 && [ "$ALLOW_UNSANDBOXED_REVIEW" != 1 ]; then
  log "PR #$PR：没有 bwrap，拒绝在无隔离环境跑评审（如需临时放开设 ALLOW_UNSANDBOXED_REVIEW=1）"
  exit 4
fi
if [ "$is_fork" = true ] && [ "$ALLOW_FORK_PR" != true ]; then
  log "PR #$PR 来自 fork，按策略跳过"; exit 3
fi
IFS=',' read -r -a hold_arr <<<"$HOLD_LABELS"
for h in "${hold_arr[@]}"; do
  [ -z "$h" ] && continue
  case ",$labels," in *",$h,"*) log "PR #$PR 带 $h 标签，跳过"; exit 3;; esac
done

OUT="$STATE/pr-$PR-$sha"
mkdir -p "$OUT"
printf '%s\n' "$meta" > "$OUT/meta.json"

# ---------- 2. 隔离 worktree ----------
WT="$WT_DIR/pr-$PR"
git -C "$REPO_ROOT" fetch -q --no-tags origin "pull/$PR/head"
git -C "$REPO_ROOT" fetch -q --no-tags origin "$base"
if ! git -C "$REPO_ROOT" cat-file -e "$sha^{commit}" 2>/dev/null; then
  sha=$(git -C "$REPO_ROOT" rev-parse FETCH_HEAD)
  OUT="$STATE/pr-$PR-$sha"
  mkdir -p "$OUT"
  printf '%s\n' "$meta" > "$OUT/meta.json"
fi
if [ -e "$WT/.git" ]; then
  git -C "$WT" checkout -q --detach "$sha"
else
  git -C "$REPO_ROOT" worktree add -q --detach --force "$WT" "$sha"
fi
git -C "$WT" reset -q --hard "$sha"
git -C "$WT" clean -qffdx

base_ref="origin/$base"
changed="$OUT/changed-files.txt"
# diff 拿不到 ≠ 没有改动。拿不到 base 引用 / merge-base / diff 一律中止，绝不放行。
if ! git -C "$WT" rev-parse --verify -q "$base_ref" >/dev/null \
   || ! git -C "$WT" merge-base "$base_ref" "$sha" >/dev/null 2>&1; then
  log "PR #$PR：拿不到 $base_ref 或 merge-base，中止"
  exit 4
fi
if ! git -C "$WT" diff --name-only --diff-filter=d "$base_ref...$sha" > "$changed"; then
  log "PR #$PR：diff 计算失败，中止"
  exit 4
fi
files_count=$(wc -l < "$changed" | tr -d ' ')
# 含换行的文件名会让逐行解析漏检（git 默认还会给特殊字符加引号），
# 用 -z 输出的 NUL 数 vs 按换行切出的行数比较，不一致就拒绝处理。
nul_count=$(git -C "$WT" diff --name-only -z --diff-filter=d "$base_ref...$sha" | tr -cd '\0' | wc -c | tr -d ' ')
line_count=$(git -C "$WT" diff --name-only -z --diff-filter=d "$base_ref...$sha" | tr '\0' '\n' | wc -l | tr -d ' ')
if [ "$nul_count" != "$line_count" ]; then
  log "PR #$PR：changed 文件名含换行/特殊字符，拒绝处理"
  exit 4
fi

# ---------- 3. T0 静态门禁（脚本判定，不靠模型） ----------
t0="$OUT/t0.txt"
: > "$t0"
t0_fail=0
add() { printf '%s\t%s\n' "$1" "$2" >> "$t0"; [ "$1" = FAIL ] && t0_fail=1 || true; }

add PASS "changed files: $files_count"

bad=$(grep -E '(^|/)artifacts/|\.(gguf|jsonl|stderr|stdout)$|(^|/)node_modules/|(^|/)profile-data\.json$|(^|/)validation\.json$' "$changed" || true)
if [ -n "$bad" ]; then add FAIL "疑似数据/大文件进 git：$(tr '\n' ' ' <<<"$bad")"; else add PASS "无数据文件（gguf/artifacts/jsonl/node_modules）"; fi

big=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  size=$(git -C "$WT" cat-file -s "HEAD:$f" 2>/dev/null || echo 0)
  [ "$size" -gt 5242880 ] && big+="$f($size) "
done < "$changed"
if [ -n "$big" ]; then add FAIL "单文件 >5MB：$big"; else add PASS "无 >5MB 文件"; fi

# 只报位置与命中数：把 token 原文写进 t0/评论会把泄露再放大一次。
leak=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in *.gguf|*.bin|*.safetensors) continue;; esac
  n=$(git -C "$WT" diff -U0 "$base_ref...$sha" -- "$f" 2>/dev/null \
        | grep -E '^\+' | grep -vE '^\+\+\+' \
        | grep -oE 'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY' \
        | wc -l || true)
  if [ "${n:-0}" -gt 0 ]; then leak+="$f($n 处) "; fi
done < "$changed"
if [ -n "$leak" ]; then add FAIL "疑似密钥（已脱敏，只报位置）：$leak"; else add PASS "无密钥特征"; fi

missing=""
while IFS= read -r f; do
  case "$f" in exps/*/README.md) ;; *) continue;; esac
  body=$(git -C "$WT" show "HEAD:$f" 2>/dev/null || true)
  for h in 实验内容 复现命令 结果; do
    grep -qE "^#+.*$h" <<<"$body" || missing+="$f 缺「$h」; "
  done
done < "$changed"
if [ -n "$missing" ]; then add FAIL "exps 文档三块不全：$missing"; else add PASS "exps 文档三块齐全（或未改 exps）"; fi

wf=$(grep -E '^\.github/workflows/' "$changed" || true)
if [ -n "$wf" ]; then add FAIL "改了 workflow 文件（当前 token 无 workflow scope）：$(tr '\n' ' ' <<<"$wf")"; else add PASS "未改 .github/workflows"; fi

if [ "$t0_fail" -eq 0 ]; then log "PR #$PR T0 通过"; else log "PR #$PR T0 失败（详见 $t0）"; fi

# ---------- 4. 可选 T1：沙箱构建 + ctest ----------
# 未审代码只在 bwrap 里执行：--unshare-all 断网，/home 与 /tmp 换成空 tmpfs，
# 只把 worktree 绑到 /work，拿不到 ~/.ssh、gh 凭证或审计目录。
run_isolated() {
  if [ "$ALLOW_UNSANDBOXED_BUILD" = 1 ]; then
    "$@"
    return
  fi
  bwrap --unshare-all --die-with-parent --new-session --clearenv \
    --ro-bind / / --dev /dev --proc /proc \
    --tmpfs /home --tmpfs /tmp \
    --bind "$WT" /tmp/work --chdir /tmp/work \
    --setenv HOME /tmp --setenv TMPDIR /tmp --setenv PATH "$PATH" --setenv LANG C.UTF-8 \
    "$@"
}

build_ok="skipped"
want_build=$RUN_BUILD_CHECK
case ",$labels," in *",run-build,"*) want_build=1;; esac
if [ "$want_build" = 1 ]; then
  if ! command -v cmake >/dev/null 2>&1; then
    add WARN "T1 跳过：本机没有 cmake"
  elif [ "$ALLOW_UNSANDBOXED_BUILD" != 1 ] && ! command -v bwrap >/dev/null 2>&1; then
    add FAIL "T1 需要 bwrap 沙箱（或显式 ALLOW_UNSANDBOXED_BUILD=1）"
    build_ok=fail
  else
    log "PR #$PR 开始沙箱构建（timeout ${BUILD_TIMEOUT}s）"
    if ( cd "$WT" && run_isolated timeout "$BUILD_TIMEOUT" cmake -B build-pr \
           -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF > "$OUT/build.log" 2>&1 ) \
       && ( cd "$WT" && run_isolated timeout "$BUILD_TIMEOUT" cmake --build build-pr \
              -j"$(nproc)" >> "$OUT/build.log" 2>&1 ); then
      if command -v ctest >/dev/null 2>&1; then
        if ( cd "$WT" && run_isolated timeout "$BUILD_TIMEOUT" ctest --test-dir build-pr \
               --output-on-failure -j"$(nproc)" >> "$OUT/build.log" 2>&1 ); then
          build_ok=pass; add PASS "T1 构建 + ctest 通过"
        else
          build_ok=fail; add FAIL "T1 ctest 失败（$OUT/build.log）"
        fi
      else
        build_ok=pass; add WARN "T1 构建通过，但 ctest 不可用，测试未执行"
      fi
    else
      build_ok=fail; add FAIL "T1 构建失败（$OUT/build.log）"
    fi
  fi
else
  add WARN "T1 构建未运行（RUN_BUILD_CHECK=0）"
fi

# ---------- 5. 材料包 + prompt + 评审（材料不足时自动补一轮） ----------
# 2026-11-10 修：旧版有三处让评审必然"材料不足"，于是 PR #4 之后再也没有 approve——
#   1) review-packet.sh 只认 src|ggml|tools|common|examples，只改 bench/scripts 的
#      PR 拿到 0 字节 diff（PR #16 的材料总共 1.1 KB）；
#   2) packet 模式把 CODEX_CWD 设成材料包，但沙箱工作目录写死成 worktree，
#      模型按 packet.md 找 files/<path>、AGENTS.md 全都读不到（PR #14）；
#   3) needs_full_diff 只能靠评论里"人工升级"，没人跑就永远 block。
# 现在材料缺口由这里自动补一轮（PACKET_MAX_ATTEMPTS，默认 2 轮）。
CODEX_CWD=$WT
packet_extra=${PACKET_EXTRA:-}   # 人工升级（PACKET_EXTRA=... review-one.sh）要能透传
expanded=0
attempt=1
max_attempts=${PACKET_MAX_ATTEMPTS:-2}
[ "$REVIEW_AUTO_EXPAND" = 1 ] || max_attempts=1

build_prompt() {
  if [ "$CODEX_CWD" != "$WT" ]; then
    {
      cat "$OUT/packet/packet.md"
      printf '\n\n'
      cat "$SCRIPT_DIR/prompt-packet.md"
    } > "$OUT/prompt.md"
  else
    {
      printf '# 本次 PR\n\n- 编号：#%s\n- 标题：%s\n- base：%s ← %s\n- 评审 head：%s\n- 改动文件数：%s\n\n' \
        "$PR" "$title" "$base" "$(jq -r .headRefName <<<"$meta")" "$sha" "$files_count"
      printf '## 改动文件\n\n'
      head -80 "$changed" | sed 's/^/- /'
      [ "$files_count" -gt 80 ] && printf '\n（其余 %s 个文件省略，可用 git diff --name-only 查看）\n' "$((files_count - 80))"
      printf '\n## T0 门禁结果\n\n'
      sed 's/^/- /' "$t0"
      printf '\n'
      cat "$SCRIPT_DIR/prompt.md"
    } > "$OUT/prompt.md"
  fi
}

# 从 needs_full_diff 的自由文本里抽出"本 PR 真的改过"的文件路径：
# 模型常写成 "src/foo.cpp 全文：重点包括…" / "`Makefile` 和 verdict.schema.json"，
# 切词与成员校验都在 pick-paths.sh 里做（精确匹配 changed-files.txt）。
expand_paths() {
  # 具体切词/匹配规则见 pick-paths.sh（有独立回归用例）：这里只负责把 JSON 摊平。
  jq -r '.needs_full_diff[]?' "$1" 2>/dev/null \
    | "$SCRIPT_DIR/pick-paths.sh" "$OUT/changed-files.txt"
}
# 本机 codex 是 wrapper，强制 --dangerously-bypass-approvals-and-sandbox，
# 所以 -s workspace-write 实际不生效，必须在外层用 bwrap 兜住：
#   - /home 换成 tmpfs（~/.ssh、~/.config/gh 不可见），只留 .codex/.local 两个软链
#   - 全量模式挂 worktree；材料包模式挂材料包目录（并把工作目录设成它）
#   - 保留网络供模型 API 使用；没有 bwrap 时退化为空 HOME（弱隔离）
run_home=""
REVIEW_CODEX_BIN=${REVIEW_CODEX_BIN:-codex}   # 便于离线用 stub 跑流程测试
run_codex() {
  if command -v bwrap >/dev/null 2>&1; then
    prwatch_sandbox_args "$WT" "$CODEX_CWD" "$OUT" "$CODEX_HOME" "$REAL_HOME" \
      "$SCRIPT_DIR" "$GIT_COMMON"
    prwatch_codex_args "$PRWATCH_BWRAP_CWD" "$REVIEW_MODEL" \
      /tmp/scripts/verdict.schema.json /tmp/out/verdict.json
    bwrap "${PRWATCH_ARGS[@]}" \
      timeout "$REVIEW_TIMEOUT" "$REVIEW_CODEX_BIN" "${PRWATCH_CODEX_ARGS[@]}" -
  else
    prwatch_codex_args "$CODEX_CWD" "$REVIEW_MODEL" \
      "$SCRIPT_DIR/verdict.schema.json" "$OUT/verdict.json"
    timeout "$REVIEW_TIMEOUT" "$REVIEW_CODEX_BIN" "${PRWATCH_CODEX_ARGS[@]}" -
  fi
}

verdict="$OUT/verdict.json"
conclusion=error
while :; do
  CODEX_CWD=$WT
  if [ "$REVIEW_PACKET" = 1 ]; then
    if PACKET_EXTRA="$packet_extra" "$SCRIPT_DIR/review-packet.sh" \
         "$PR" "$sha" "$WT" "$OUT" "$base_ref"; then
      CODEX_CWD="$OUT/packet"
    else
      log "PR #$PR 第 $attempt 轮：packet 构建失败，回退全量模式"
      CODEX_CWD=$WT
    fi
  fi
  build_prompt
  cp "$OUT/prompt.md" "$OUT/prompt.$attempt.md" 2>/dev/null || true

  codex_jsonl="$OUT/codex.jsonl"
  [ "$attempt" -gt 1 ] && codex_jsonl="$OUT/codex.$attempt.jsonl"
  log "PR #$PR 第 $attempt 轮调用 codex（$REVIEW_MODEL, timeout ${REVIEW_TIMEOUT}s, cwd=$CODEX_CWD）"

  rm -f "$OUT/codex-out/verdict.json" "$verdict"
  mkdir -p "$OUT/codex-out"
  # 每轮保留自己的日志：codex.jsonl 是首轮，codex.<n>.jsonl 是补材料轮
  run_home=$(mktemp -d "${TMPDIR:-/tmp}/prwatch-home.XXXXXX")
  mkdir -p "$run_home/.config"
  set +e
  ( unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN
    HOME="$run_home" XDG_CONFIG_HOME="$run_home/.config" TMPDIR="$run_home" \
    GIT_CONFIG_GLOBAL=/dev/null CODEX_HOME="$CODEX_HOME"
    run_codex ) < "$OUT/prompt.md" > "$codex_jsonl" 2>&1
  codex_rc=$?
  set -e
  rm -rf "$run_home"
  run_home=""
  if [ -s "$OUT/codex-out/verdict.json" ]; then
    cp "$OUT/codex-out/verdict.json" "$verdict"
  fi

  cp "$verdict" "$OUT/verdict.$attempt.json" 2>/dev/null || true
  conclusion=error
  review_failed=0
  fail_reason=""
  if [ "$codex_rc" -eq 0 ] && [ -s "$verdict" ] && jq -e '.conclusion' "$verdict" >/dev/null 2>&1; then
    conclusion=$(jq -r '.conclusion' "$verdict")
  else
    # 模型调用失败（额度/网络/超时）不是"评审不通过"：记 error 并把原因带出来，
    # 否则审计里全是 blocked，看不出"根本没评"和"评了但拦下"的区别。
    review_failed=1
    fail_reason=$(jq -r 'select(.type=="error") | .message' "$codex_jsonl" 2>/dev/null | tail -1 | tr '\n' ' ' || true)
    [ -z "$fail_reason" ] && fail_reason="codex 退出码 $codex_rc（见 $(basename "$codex_jsonl")）"
    log "PR #$PR codex 失败（rc=$codex_rc）：$fail_reason"
    jq -nc --arg r "$fail_reason" --arg rc "$codex_rc" \
      '{conclusion:"blocked", blocking:[], non_blocking:[], policy_violations:[], summary:("评审未完成，不是代码/材料结论： " + $r + " (exit " + $rc + ")")}' > "$verdict"
  fi

  # 材料缺口 → 用 needs_full_diff 里点名的文件重开一轮（只补一轮，避免死循环）
  more=""
  [ -s "$verdict" ] && more=$(expand_paths "$verdict" | sed -E 's/^ +//; s/ +$//')
  if [ "$attempt" -lt "$max_attempts" ] && [ -n "$more" ]; then
    # 上一轮已经给过全文的文件要一起带上并排在前面，否则它们会被新点名的
    # 大文件挤出预算——那等于"补了一轮，反而把关键文件弄丢了"。
    prev_full=$(jq -r '.full_files[]?' "$OUT/packet/stats.json" 2>/dev/null || true)
    log "PR #$PR 材料不足，自动展开后重审（保留 $(grep -c . <<<"$prev_full" || true) 个已给全文的文件）：$(paste -sd, - <<<"$more")"
    # 换行分隔而不是空格：路径里可能有空格，按空格拼会把它拆坏。
    # 人工指定的 PACKET_EXTRA 也要继续带着，别在补材料那一轮丢掉。
    packet_extra=$(printf '%s\n%s\n%s' "${PACKET_EXTRA:-}" "$prev_full" "$more")
    expanded=1
    attempt=$((attempt + 1))
    continue
  fi
  break
done
attempts_used=$attempt

# fail closed：补材料轮跑完仍缺材料时，approve 降级为 blocked（见 fail-closed.sh）。
if [ -n "${more:-}" ] && [ "$conclusion" = approve ]; then
  tmpv=$(mktemp)
  if "$SCRIPT_DIR/fail-closed.sh" "$verdict" "$more" "$attempts_used" > "$tmpv"; then
    mv "$tmpv" "$verdict"
    conclusion=blocked
    log "PR #$PR：补材料轮后仍缺材料（$(paste -sd, - <<<"$more")），approve 降级为 blocked"
  else
    rm -f "$tmpv"
  fi
fi

# 评审期间 worktree 必须仍等于被审 head；否则等于给"被改过的内容"签审批。
integrity=ok
dirty=$(git -C "$WT" status --porcelain 2>/dev/null | head -5 || true)
head_now=$(git -C "$WT" rev-parse HEAD 2>/dev/null || echo "")
if [ -n "$dirty" ] || [ "$head_now" != "$sha" ]; then
  integrity=bad
  log "PR #$PR：worktree 在评审期间被修改，结论作废"
  add FAIL "评审期间 worktree 被修改（HEAD=${head_now:0:12}）"
  git -C "$WT" reset -q --hard "$sha"
  git -C "$WT" clean -qffdx
fi

# ---------- 6. 落盘门禁结果（merge-ready 只信这个文件，不信内存变量） ----------
if [ "$t0_fail" -ne 0 ]; then
  conclusion=request_changes
fi
if [ "$integrity" = bad ]; then
  conclusion=blocked
fi
if [ "$review_failed" = 1 ]; then
  conclusion=error
fi
# bench 证据：碰到运行时路径就要求 bench-result.json（BENCH_GATE=auto/required）
bench_status=skip
if [ "$BENCH_GATE" != off ]; then
  need_bench=0
  if [ "$BENCH_GATE" = required ]; then need_bench=1; fi
  if [ "$BENCH_GATE" = auto ] && grep -qE '^(src|ggml|tools|common|examples)/' "$changed"; then
    need_bench=1
  fi
  if [ "$need_bench" = 1 ]; then
    if "$SCRIPT_DIR/bench-gate.sh" "$PR" "$sha" > "$OUT/bench-gate.txt" 2>&1; then
      bench_status=pass
    else
      bench_status=fail
    fi
  fi
fi

jq -nc --argjson pr "$PR" --arg sha "$sha" --arg base "$base" \
  --argjson t0 "$([ "$t0_fail" -eq 0 ] && echo true || echo false)" \
  --arg t1 "$build_ok" --arg c "$conclusion" \
  --argjson pv "$(jq -c '.policy_violations // []' "$verdict")" \
  --argjson bl "$(jq -c '(.blocking // []) | length' "$verdict")" \
  --arg model "$REVIEW_MODEL" --arg ts "$(date -Is)" --arg bench "$bench_status" \
  '{pr:$pr,sha:$sha,base:$base,t0_pass:$t0,t1:$t1,conclusion:$c,policy_violations:$pv,blocking:$bl,model:$model,bench:$bench,ts:$ts}' \
  > "$OUT/gate.json"

# ---------- 7. 贴评论 + 打标签 ----------

comment="$OUT/comment.md"
{
  printf '## pr-watch 评审：`%s`\n\n' "$conclusion"
  printf -- '- PR：#%s《%s》\n' "$PR" "$title"
  printf -- '- 对比：`%s` ← `%s`，评审 head：`%s`\n' "$base" "$(jq -r .headRefName <<<"$meta")" "$sha"
  printf -- '- 模型：`%s`，T0：%s，T1：%s\n\n' "$REVIEW_MODEL" \
    "$([ "$t0_fail" -eq 0 ] && echo '✅ pass' || echo '❌ fail')" "$build_ok"
  if [ "${review_failed:-0}" = 1 ]; then
    printf '> ⚠️ 本轮**评审没有跑完**（不是代码/材料结论）：`%s`\n> 先解决这个再看下面的空结论，否则本评论没有评审意义。\n\n' "$fail_reason"
  fi
  printf '### 阻塞项\n\n'
  if [ "$(jq -r '(.blocking // []) | length' "$verdict")" -eq 0 ]; then
    printf -- '- 无\n'
  else
    jq -r '.blocking[]? | "- " + .' "$verdict"
  fi
  printf '\n### 非阻塞项\n\n'
  if [ "$(jq -r '(.non_blocking // []) | length' "$verdict")" -eq 0 ]; then
    printf -- '- 无\n'
  else
    jq -r '.non_blocking[]? | "- " + .' "$verdict"
  fi
  printf '\n### 政策检查\n\n'
  if [ "$(jq -r '(.policy_violations // []) | length' "$verdict")" -eq 0 ]; then
    printf -- '- 未发现 AGENTS.md 违规\n'
  else
    jq -r '.policy_violations[]? | "- " + .' "$verdict"
  fi
  printf '\n### bench 门禁\n\n'
  case "$bench_status" in
    pass) printf -- '- ✅ 通过\n\n```\n%s\n```\n' "$(tail -8 "$OUT/bench-gate.txt" 2>/dev/null)";;
    fail) printf -- '- ❌ 未通过，**禁止合并**\n\n```\n%s\n```\n' "$(cat "$OUT/bench-gate.txt" 2>/dev/null | head -20)";;
    *)    printf -- '- 跳过（PR 未触及 bench 门禁路径，或 BENCH_GATE=off）——bench 缺失不作为阻塞理由\n';;
  esac
  if [ -f "$OUT/packet/stats.json" ]; then
    printf '\n### 评审材料\n\n- packet 模式：高风险 %s 个文件 / 低风险 %s 个，diff %s KB，材料 %s KB\n' \
      "$(jq -r '.high_files' "$OUT/packet/stats.json")" \
      "$(jq -r '.low_files' "$OUT/packet/stats.json")" \
      "$(jq -r '(.high_diff_bytes/1024)|floor' "$OUT/packet/stats.json")" \
      "$(jq -r '(.packet_bytes/1024)|floor' "$OUT/packet/stats.json")"
    if [ "$(jq -r '.omitted_files // 0' "$OUT/packet/stats.json")" -gt 0 ]; then
      printf -- '- 未放全文：%s 个文件（材料预算上限，属材料缺口、不是代码缺陷）\n' \
        "$(jq -r '.omitted_files' "$OUT/packet/stats.json")"
    fi
    if [ "$(jq -r '.diff_truncated // false' "$OUT/packet/stats.json")" = true ]; then
      printf -- '- diff 已截断：全文 %s KB，材料里只给了前 %s KB\n' \
        "$(jq -r '(.high_diff_bytes/1024)|floor' "$OUT/packet/stats.json")" \
        "$(jq -r '(.high_diff_kept_bytes/1024)|floor' "$OUT/packet/stats.json")"
    fi
    if [ "$expanded" = 1 ]; then
      printf -- '- 共 %s 轮：已按 needs_full_diff 自动展开材料后重审\n' "$attempts_used"
    fi
  fi
  if [ "$(jq -r '(.needs_full_diff // []) | length' "$verdict")" -gt 0 ]; then
    printf '\n### 需要展开的文件（材料不足）\n\n'
    jq -r '.needs_full_diff[] | "- `" + . + "`"' "$verdict"
    printf '\n（已自动尝试补一轮；仍缺则可用 `PACKET_EXTRA="<这些路径>" scripts/pr-watch/review-one.sh %s %s` 人工升级）\n' "$PR" "$sha"
  fi
  printf '\n### 摘要\n\n%s\n\n' "$(jq -r '.summary // ""' "$verdict")"
  printf '<details><summary>T0 明细</summary>\n\n```\n'
  cat "$t0"
  printf '```\n\n</details>\n\n'
  printf '<sub>pr-watch · head `%s` · %s · 审计：`%s/audit.jsonl`</sub>\n' \
    "$sha" "$(date '+%F %T %Z')" "$STATE"
} > "$comment"
# 发出去之前再脱敏一次：模型输出里可能带出 token（例如被 diff 诱导回显）。
sed -i -E 's/gh[pousr]_[A-Za-z0-9]{20,}/[REDACTED-GH]/g; s/github_pat_[A-Za-z0-9_]{20,}/[REDACTED-GH]/g; s/sk-[A-Za-z0-9]{20,}/[REDACTED-KEY]/g' "$comment"

set +e
if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN：评论写到 $comment（未发布）"
else
  gh pr comment "$PR" --repo "$REPO" --body-file "$comment" >/dev/null 2>&1 \
    || log "PR #$PR 评论发布失败"
  case "$conclusion" in
    approve)          keep=review:approve; drop="review:changes review:blocked review:error";;
    request_changes)  keep=review:changes; drop="review:approve review:blocked review:error";;
    blocked)          keep=review:blocked; drop="review:approve review:changes review:error";;
    *)                keep=review:error;   drop="review:approve review:changes review:blocked";;
  esac
  for l in $drop; do gh pr edit "$PR" --repo "$REPO" --remove-label "$l" >/dev/null 2>&1 || true; done
  gh label create "$keep" --repo "$REPO" --color ededed --force >/dev/null 2>&1 || true
  gh pr edit "$PR" --repo "$REPO" --add-label "$keep" >/dev/null 2>&1 || true
fi
set -e

# ---------- 8. 审计 ----------
jq -nc --arg ts "$(date -Is)" --argjson pr "$PR" --arg sha "$sha" \
  --arg c "$conclusion" --arg model "$REVIEW_MODEL" --argjson t0 "$([ "$t0_fail" -eq 0 ] && echo true || echo false)" \
  --arg t1 "$build_ok" --argjson attempts "$attempts_used" \
  '{ts:$ts,pr:$pr,sha:$sha,conclusion:$c,model:$model,t0_pass:$t0,t1:$t1,attempts:$attempts}' >> "$STATE/audit.jsonl"

# ---------- 9. 合并（按策略） ----------
# 评审没跑完（额度/网络/超时）：返回 12，让 watch.sh 走"基础设施失败"计数与退避，
# **不要**返回 0——返回 0 会被记进 seen.txt，这个 head 就再也不会被自动重评了。
if [ "$review_failed" = 1 ]; then
  log "PR #$PR 评审未完成（$fail_reason），返回 12 让 watcher 重试"
  exit 12
fi
if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN：跳过合并门禁"
  exit 0
fi
if [ "$MERGE_POLICY" = report ]; then
  log "MERGE_POLICY=report，只审不合"
  exit 11
fi
if [ "$conclusion" = approve ] && [ "$t0_fail" -eq 0 ] && [ "$build_ok" != fail ]; then
  if "$SCRIPT_DIR/merge-ready.sh" "$PR" "$sha" "$OUT"; then
    log "PR #$PR 已合并"
    exit 0
  fi
  log "PR #$PR 已 approve 但门禁暂未通过，进入待合并队列（下轮只重试门禁，不重跑评审）"
  exit 10
fi

exit 0
