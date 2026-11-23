#!/usr/bin/env bash
# 材料包（Stage 3）离线自测：不联网、不碰手机、不叫模型。
#
# 复现的两类事故：
#   - PR #16：只改 bench/bench.sh + docs 的 PR 被判"高风险 0 个文件 / diff 0 KB"，
#     模型看不到任何 diff，只能 blocked；
#   - PR #14：packet.md 说"全文在 packet/files/ 下"，但评审沙箱工作目录是 worktree，
#     模型按这个相对路径找不到文件，于是又是一条"材料缺失"。
#
# 用法：scripts/pr-watch/test-packet.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fails=0
ok()   { printf 'PASS\t%s\n' "$*"; }
bad()  { printf 'FAIL\t%s\n' "$*"; fails=$((fails + 1)); }
check_eq()   { [ "$1" = "$2" ] && ok "$3" || bad "$3（期望 $2，实际 $1）"; }
check_has()  { grep -q -- "$2" "$1" && ok "$3" || bad "$3（$1 里没有匹配：$2）"; }
check_not()  { ! grep -q -- "$2" "$1" && ok "$3" || bad "$3（$1 里不该出现：$2）"; }

# ---------- 造一个最小仓库 ----------
R="$TMP/repo"
git init -q -b main "$R"
git -C "$R" config user.email t@example.com
git -C "$R" config user.name tester
git -C "$R" config commit.gpgsign false
mkdir -p "$R/src" "$R/bench" "$R/docs"
printf '# 政策\n' > "$R/AGENTS.md"
printf 'int main(void){return 0;}\n' > "$R/src/foo.cpp"
printf 'int same(void){return 0;}\n' > "$R/src/same.cpp"
printf '#!/bin/sh\necho v1\n' > "$R/bench/bench.sh"
printf '# doc v1\n' > "$R/docs/x.md"
printf 'cmake_minimum_required(VERSION 3.20)\n' > "$R/CMakeLists.txt"
mkdir -p "$R/tests"
printf 'add_test(foo foo)\n' > "$R/tests/CMakeLists.txt"
printf 'all:\n\techo hi\n' > "$R/Makefile"
git -C "$R" add -A
git -C "$R" commit -qm base
git -C "$R" checkout -q -b feat
printf '#!/bin/sh\necho v2\nWINDOW=80\n' > "$R/bench/bench.sh"
printf 'int main(void){return 1;}\n' > "$R/src/foo.cpp"
printf '# doc v2\n' > "$R/docs/x.md"
printf 'cmake_minimum_required(VERSION 3.22)\n' > "$R/CMakeLists.txt"
printf 'add_test(foo foo)\nadd_test(bar bar)\n' > "$R/tests/CMakeLists.txt"
printf 'all:\n\techo hi there\n' > "$R/Makefile"
git -C "$R" add -A
git -C "$R" commit -qm feat
SHA=$(git -C "$R" rev-parse HEAD)

mk_out() { # mk_out <dir> <files...>
  local d=$1; shift
  mkdir -p "$d"
  printf '{"title":"合成 PR","author":{"login":"tester"},"headRefName":"feat","headRefOid":"%s","baseRefName":"main"}\n' \
    "$SHA" > "$d/meta.json"
  printf 'PASS\tchanged files: %s\n' "$#" > "$d/t0.txt"
  printf '%s\n' "$@" > "$d/changed-files.txt"
}
run_packet() { # run_packet <outdir>
  PRWATCH_HOME="$TMP/state" PACKET_SUMMARY=0 \
    "$SCRIPT_DIR/review-packet.sh" 1 "$SHA" "$R" "$1" main
}

# ---------- 用例 1：只改 bench 脚本 + docs 的 PR 必须有 diff ----------
OUT1="$TMP/out1"
mk_out "$OUT1" bench/bench.sh docs/x.md
run_packet "$OUT1" >/dev/null
st="$OUT1/packet/stats.json"
check_eq "$(jq -r '.high_files' "$st")" 1 "bench/bench.sh 被算作高风险（旧版是 0）"
check_eq "$(jq -r '.low_files' "$st")" 1 "docs/*.md 仍算低风险"
check_has "$OUT1/packet/high.diff" 'WINDOW=80' "只改 bench/ 的 PR 也拿到真实 diff（PR #16 回归）"
check_not "$OUT1/packet/packet.md" 'packet/files' "packet.md 不再引用模型读不到的 packet/files/"
check_has "$OUT1/packet/packet.md" 'files/<路径>' "packet.md 用材料包内相对路径"
check_not "$OUT1/packet/packet.md" 'packet/AGENTS.md' "AGENTS.md 引用与工作目录一致"
check_has "$OUT1/packet/packet.md" '不适用' "bench-only PR 的 bench 门禁标为不适用"
check_has "$OUT1/packet/low.summary.md" '逐文件改动统计' "低风险文件也给出逐文件改动统计"
check_has "$OUT1/packet/low.summary.md" 'docs/x.md' "低风险摘要列出真实文件名"
check_eq "$(jq -r '.diff_truncated' "$st")" false "未超限时不标截断"

# ---------- 用例 2：全文预算不足时显式标注省略，而不是静默丢 ----------
OUT2="$TMP/out2"
mk_out "$OUT2" bench/bench.sh src/foo.cpp docs/x.md
PACKET_MAX_FILES_BYTES=10 PRWATCH_HOME="$TMP/state" PACKET_SUMMARY=0 \
  "$SCRIPT_DIR/review-packet.sh" 1 "$SHA" "$R" "$OUT2" main >/dev/null
check_eq "$(jq -r '.omitted_files' "$OUT2/packet/stats.json")" 2 "超预算的文件被记为 omitted"
check_has "$OUT2/packet/packet.md" '未提供全文的文件' "packet.md 显式列出未提供全文的文件"
if [ -z "$(ls -A "$OUT2/packet/files" 2>/dev/null)" ]; then ok "被省略的文件没有塞进 files/"; else bad "files/ 不该有文件"; fi
# 关键：不能再出现"清单写着 full、其实没有全文"的谎话
if grep -q '| `bench/bench.sh` | full' "$OUT2/packet/packet.md"; then
  bad "被省略的文件不应标成 full"
else ok "被省略的文件没有标成 full"; fi

# ---------- 用例 3：diff 截断必须留痕 ----------
OUT3="$TMP/out3"
mk_out "$OUT3" bench/bench.sh src/foo.cpp
PACKET_MAX_DIFF_BYTES=20 PRWATCH_HOME="$TMP/state" PACKET_SUMMARY=0 \
  "$SCRIPT_DIR/review-packet.sh" 1 "$SHA" "$R" "$OUT3" main >/dev/null
check_eq "$(jq -r '.diff_truncated' "$OUT3/packet/stats.json")" true "diff 超限时 stats 标截断"
check_has "$OUT3/packet/packet.md" 'diff 已截断' "packet.md 里写明截断"
check_eq "$(jq -r '.high_diff_kept_bytes' "$OUT3/packet/stats.json")" 20 \
  "kept_bytes 报的是实际给出的字节数，不是全文大小（否则评论会写\"全文 X KB、只给前 X KB\"）"

# ---------- 用例 4：列了高风险文件却拿不到 diff → 拒绝产出空材料 ----------
OUT4="$TMP/out4"
mk_out "$OUT4" src/same.cpp
set +e
PRWATCH_HOME="$TMP/state" PACKET_SUMMARY=0 \
  "$SCRIPT_DIR/review-packet.sh" 1 "$SHA" "$R" "$OUT4" main >/dev/null 2>&1
rc=$?
set -e
[ "$rc" -ne 0 ] && ok "diff 为空时拒绝产出（退出码 $rc）" || bad "diff 为空时应当失败"

# ---------- 用例 5：PACKET_EXTRA 点名的低风险文件不重复计数 ----------
OUT5="$TMP/out5"
mk_out "$OUT5" bench/bench.sh docs/x.md
PACKET_EXTRA="docs/x.md" PRWATCH_HOME="$TMP/state" PACKET_SUMMARY=0 \
  "$SCRIPT_DIR/review-packet.sh" 1 "$SHA" "$R" "$OUT5" main >/dev/null
check_eq "$(jq -r '.extra_files' "$OUT5/packet/stats.json")" 1 "PACKET_EXTRA 的文件记为 extra"
check_eq "$(jq -r '.low_files' "$OUT5/packet/stats.json")" 0 "点名后不再重复计入低风险"
check_has "$OUT5/packet/low.summary.md" '没有低风险文件' "低风险为空时有明确说明"

# ---------- 用例 6：构建文件必须按高风险处理（CMakeLists.txt 别被 *.txt 吃掉） ----------
OUT6="$TMP/out6"
mk_out "$OUT6" CMakeLists.txt tests/CMakeLists.txt Makefile
run_packet "$OUT6" >/dev/null
check_eq "$(jq -r '.high_files' "$OUT6/packet/stats.json")" 3 "CMakeLists.txt / Makefile 算高风险"
check_has "$OUT6/packet/high.diff" 'cmake_minimum_required' "构建文件拿到了真实 diff"
check_eq "$(jq -r '.low_files' "$OUT6/packet/stats.json")" 0 "构建文件不会被误判为低风险"

# ---------- 用例 7：PACKET_EXTRA 重复路径只算一次 ----------
OUT7="$TMP/out7"
mk_out "$OUT7" bench/bench.sh docs/x.md
PACKET_EXTRA="docs/x.md docs/x.md" PRWATCH_HOME="$TMP/state" PACKET_SUMMARY=0 \
  "$SCRIPT_DIR/review-packet.sh" 1 "$SHA" "$R" "$OUT7" main >/dev/null
check_eq "$(jq -r '.extra_files' "$OUT7/packet/stats.json")" 1 "重复的 PACKET_EXTRA 只计一次"

# ---------- 用例 8：needs_full_diff 的路径挑选（含 Makefile/*.json/中文标点） ----------
LIST="$TMP/changed-for-pick.txt"
printf 'Makefile\nscripts/pr-watch/verdict.schema.json\nCMakeLists.txt\nbench/foo bar.sh\n' > "$LIST"
got=$(printf '%s\n' \
  '- `Makefile` 全文；还有 scripts/pr-watch/verdict.schema.json。' \
  '- 需要 CMakeLists.txt，以及不存在的 src/nope.cpp' \
  '- `bench/foo bar.sh` 也要' \
  | "$SCRIPT_DIR/pick-paths.sh" "$LIST" | tr '\n' ' ' | sed 's/ *$//')
check_eq "$got" "Makefile scripts/pr-watch/verdict.schema.json CMakeLists.txt bench/foo bar.sh" \
  "无扩展名/带点/含空格的文件名都能选中，且不把 verdict.schema.json 截断"
got2=$(printf 'src/not-in-pr.cpp\n' | "$SCRIPT_DIR/pick-paths.sh" "$LIST" | tr -d '\n')
check_eq "$got2" "" "不在本 PR 改动清单里的路径被丢弃"
# 句末英文句号是路径字符，严格边界会漏掉（PR #19 二轮评审指出的真 bug）
printf 'scripts/pr-watch/watch.sh\n' > "$TMP/list-period.txt"
got3=$(printf 'Please provide scripts/pr-watch/watch.sh.\n' | "$SCRIPT_DIR/pick-paths.sh" "$TMP/list-period.txt")
check_eq "$got3" "scripts/pr-watch/watch.sh" "句末带英文句号也能选中"
got4=$(printf 'need scripts/pr-watch/watch.sh. then go on\n' | "$SCRIPT_DIR/pick-paths.sh" "$TMP/list-period.txt")
check_eq "$got4" "scripts/pr-watch/watch.sh" "句中带英文句号也能选中"
got5=$(printf 'scripts/pr-watch/watch.sh.bak\n' | "$SCRIPT_DIR/pick-paths.sh" "$TMP/list-period.txt")
check_eq "$got5" "" "watch.sh.bak 是另一个文件名，不该被当成 watch.sh 选中"

# ---------- 用例 9：预算不够时优先保住改动量大的文件 ----------
OUT9="$TMP/out9"
mk_out "$OUT9" bench/bench.sh src/foo.cpp
run_packet "$OUT9" >/dev/null
check_eq "$(jq -r '.full_files[0]' "$OUT9/packet/stats.json")" "bench/bench.sh" \
  "改动行数最多的文件排在全文章节第一位（避免 PR #14 那种关键文件被挤出预算）"

# ---------- 用例 10：补轮用尽仍缺材料时 approve 必须 fail closed ----------
VJ="$TMP/verdict.json"
printf '%s' '{"conclusion":"approve","blocking":[],"non_blocking":[],"policy_violations":[],"summary":"ok"}' > "$VJ"
set +e
out=$("$SCRIPT_DIR/fail-closed.sh" "$VJ" "src/llama-moe-slot.cpp" 2); rc=$?
set -e
check_eq "$rc" 0 "approve + 未满足材料请求 → 触发降级"
check_eq "$(jq -r '.conclusion' <<<"$out")" "blocked" "降级后结论是 blocked"
check_has <(printf '%s' "$out") 'llama-moe-slot.cpp' "降级理由写明缺哪个文件"
set +e
"$SCRIPT_DIR/fail-closed.sh" "$VJ" "" 2 >/dev/null; rc2=$?
set -e
check_eq "$rc2" 3 "没有未满足请求时不降级"
printf '%s' '{"conclusion":"request_changes","blocking":["x"],"non_blocking":[],"policy_violations":[],"summary":"s"}' > "$VJ"
set +e
"$SCRIPT_DIR/fail-closed.sh" "$VJ" "src/a.cpp" 2 > /dev/null; rc3=$?
set -e
check_eq "$rc3" 3 "非 approve 结论不降级（避免把 request_changes 弄成 blocked）"

# ---------- 用例 11：评审沙箱（用生产同一份 sandbox-args.sh 构造参数） ----------
# 关键：这里**不手写 bwrap 参数**。生产若把 -C/--chdir 退回 worktree，这一条就会失败。
if command -v bwrap >/dev/null 2>&1; then
  mkdir -p "$OUT1/codex-out"
  ch=$( ( source "$SCRIPT_DIR/sandbox-args.sh"
          prwatch_sandbox_args "$R" "$OUT1/packet" "$OUT1" "${CODEX_HOME:-$HOME/.codex}" \
            "$HOME" "$SCRIPT_DIR" ""
          printf '%s' "$PRWATCH_BWRAP_CWD" ) )
  check_eq "$ch" "/tmp/packet" "材料包模式下沙箱工作目录就是材料包"
  got=$( ( source "$SCRIPT_DIR/sandbox-args.sh"
            prwatch_sandbox_args "$R" "$OUT1/packet" "$OUT1" "${CODEX_HOME:-$HOME/.codex}" \
              "$HOME" "$SCRIPT_DIR" ""
            bwrap "${PRWATCH_ARGS[@]}" \
              bash -c 'test -f files/bench/bench.sh && test -f AGENTS.md && printf ok' ) 2>/dev/null || true )
  check_eq "$got" "ok" "沙箱里按 files/<路径> 能读到全文，AGENTS.md 可读"

  # 生产 codex 的启动参数：材料包模式必须 -C /tmp/packet，且显式跳过 git 仓库检查
  argv=$( ( source "$SCRIPT_DIR/sandbox-args.sh"
             prwatch_codex_args /tmp/packet "stub-model" /tmp/scripts/verdict.schema.json /tmp/out/verdict.json
             printf '%s ' "${PRWATCH_CODEX_ARGS[@]}" ) )
  case "$argv" in *"-C /tmp/packet "*) ok "codex 在材料包模式下 -C 指向材料包";; *) bad "codex 的 -C 不是 /tmp/packet：$argv";; esac
  case "$argv" in *"--skip-git-repo-check "*) ok "codex 显式 --skip-git-repo-check（材料包不是 git 仓库）";; *) bad "缺少 --skip-git-repo-check";; esac
  case "$argv" in *"--output-schema /tmp/scripts/verdict.schema.json "*) ok "codex 使用沙箱内 schema 路径";; *) bad "schema 路径不对：$argv";; esac

  ch2=$( ( source "$SCRIPT_DIR/sandbox-args.sh"
            prwatch_sandbox_args "$R" "$R" "$OUT1" "${CODEX_HOME:-$HOME/.codex}" \
              "$HOME" "$SCRIPT_DIR" ""
            printf '%s' "$PRWATCH_BWRAP_CWD" ) )
  check_eq "$ch2" "/tmp/work" "全量模式（无材料包）工作目录仍是 worktree"
else
  bad "没有 bwrap：评审沙箱无法工作，本机不该在没有 bwrap 的情况下跑 auto review"
fi

if [ "$fails" -eq 0 ]; then
  printf '\n全部通过\n'
else
  printf '\n%d 项失败\n' "$fails"
  exit 1
fi
