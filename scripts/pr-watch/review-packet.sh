#!/usr/bin/env bash
# Stage 3 材料包：把"GPT 该看的东西"裁成一个目录，让贵模型只看证据和高风险 diff。
#
# 用法：review-packet.sh <pr> <sha> <worktree> <outdir> [base-ref]
# 环境：PACKET_EXTRA="path1 path2"  额外要求展开的文件（人工升级 / 自动重审用）。
#       这些文件优先塞进预算，且不受总预算的"先到先得"顺序影响。
#
# 产出：<outdir>/packet/
#   packet.md        给 GPT 的材料（meta + T0 + bench + 低风险摘要 + 高风险 diff）
#   high.diff        高风险文件的完整 diff（超限时截断，且在 packet.md 里显式标注）
#   files/<path>     高风险文件的完整内容（带上下文用；被省略的在 packet.md 里标注）
#   low.summary.md   便宜模型给出的低风险文件摘要（不是结论）
#   stats.json       材料规模统计（用于成本追踪）
#
# 约定（2026-11-10 修）：评审沙箱把本目录 bind 成工作目录，所以 packet.md 里
# 一律用"相对本材料包根目录"的路径（files/<path>），不再写一个模型找不到的
# `packet/files/...`。回归用例见 test-packet.sh。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PR=${1:?usage: review-packet.sh <pr> <sha> <worktree> <outdir> [base-ref]}
SHA=${2:?}
WT=${3:?}
OUT=${4:?}
BASE=${5:-origin/$MERGE_BASE_BRANCH}

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
PKG="$OUT/packet"
rm -rf "$PKG"
mkdir -p "$PKG/files"

# ---------- 风险分类 ----------
# 判据是"这段改动会不会影响行为/测量"，不是它放在哪个顶层目录。
# 之前只认 src|ggml|tools|common|examples，导致只改 bench/ scripts/ include/ tests/
# 的 PR 拿到 high_files=0、high_diff=0 的空材料，评审只能判 blocked（PR #16）。
# 现在：散文类（md/txt/图片/数据）仍算低风险；代码、脚本、构建文件一律高风险。
PROSE_RE='\.(md|markdown|txt|rst|png|jpe?g|svg|gif|jsonl)$'
CODE_EXT_RE='\.(c|cc|cpp|cxx|h|hh|hpp|m|mm|cu|metal|s|S|asm|sh|bash|py|ps1|bat|cmake)$'
CODE_DIR_RE='^(src|ggml|tools|common|examples|include|tests|app|ci|cmake|bench|scripts)/'

is_high() {
  local f=$1
  # 构建文件先判：CMakeLists.txt 会被 *.txt 的散文规则吃掉，必须排在前面。
  case "$f" in
    CMakeLists.txt|*/CMakeLists.txt|Makefile|*/Makefile|CMakePresets.json|Dockerfile) return 0;;
  esac
  [[ "$f" =~ $PROSE_RE ]] && return 1
  [[ "$f" =~ $CODE_EXT_RE ]] && return 0
  [[ "$f" =~ $CODE_DIR_RE ]] && return 0
  return 1
}

mapfile -t all < "$OUT/changed-files.txt"
high=()
low=()
for f in "${all[@]}"; do
  [ -z "$f" ] && continue
  if is_high "$f"; then high+=("$f"); else low+=("$f"); fi
done

# 人工/自动要求展开的文件：存在才算数，且排在最前（先占预算）。
extra=()
if [ -n "${PACKET_EXTRA:-}" ]; then
  # 每行一个路径；若某行不是有效路径，再按空格拆（兼容 README 里的
  # PACKET_EXTRA="a b" 写法）。这样带空格的路径不会被拆坏。
  cand=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    if git -C "$WT" cat-file -e "$SHA:$line" 2>/dev/null; then
      cand+=("$line")
    else
      read -r -a parts <<<"$line"
      cand+=("${parts[@]}")
    fi
  done < <(printf '%s\n' "$PACKET_EXTRA")
  for f in "${cand[@]}"; do
    [ -z "$f" ] && continue
    dup=0
    for g in "${extra[@]:-}"; do [ "$g" = "$f" ] && { dup=1; break; }; done
    [ "$dup" -eq 1 ] && continue
    if ! git -C "$WT" cat-file -e "$SHA:$f" 2>/dev/null; then
      printf 'packet: PACKET_EXTRA 里的 %s 在被审 head 里不存在，忽略\n' "$f" >&2
      continue
    fi
    extra+=("$f")
  done
fi

# 被 PACKET_EXTRA 点名的低风险文件已经在高风险清单里给了全文，别在低风险里重复一遍
if [ ${#extra[@]} -gt 0 ]; then
  kept=()
  for f in "${low[@]}"; do
    dup=0
    for g in "${extra[@]}"; do [ "$g" = "$f" ] && { dup=1; break; }; done
    [ "$dup" -eq 1 ] || kept+=("$f")
  done
  if [ ${#kept[@]} -gt 0 ]; then low=("${kept[@]}"); else low=(); fi
fi

# 高风险文件按"改动行数"从大到小排：预算不够时先保住真正被大改的文件。
# （PR #14 按字母序先塞了 4 个只改几行的头文件，把改了 577 行的
#   src/llama-moe-slot.cpp 挤出预算，评审因此判"材料缺失"。）
diff_lines() {
  git -C "$WT" diff --numstat "$BASE...$SHA" -- "$1" 2>/dev/null \
    | awk 'NF>=2 {print $1+$2; exit}' || true
}
# 输出 "added deleted"；二进制文件 numstat 是 "-  -"，归一成 0 0
numstat_pair() {
  local f=$1 a d _
  IFS=$'\t' read -r a d _ < <(git -C "$WT" diff --numstat "$BASE...$SHA" -- "$f" 2>/dev/null | head -1) || true
  case "${a:-}" in ''|-) a=0;; esac
  case "${d:-}" in ''|-) d=0;; esac
  printf '%s %s' "$a" "$d"
}
declare -A _rank
for f in "${high[@]}"; do _rank["$f"]=$(diff_lines "$f"); done
mapfile -t high_sorted < <(
  for f in "${high[@]}"; do printf '%s\t%s\n' "${_rank[$f]:-0}" "$f"; done \
    | sort -t $'\t' -k1,1nr -k2 | cut -f2-
)
# extra 优先：先按给定顺序排前面，再补其余高风险文件，去重。
ordered=()
for f in "${extra[@]}"; do ordered+=("$f"); done
for f in "${high_sorted[@]}"; do
  dup=0
  for g in "${ordered[@]}"; do [ "$g" = "$f" ] && { dup=1; break; }; done
  [ "$dup" -eq 1 ] || ordered+=("$f")
done
high=("${ordered[@]}")

# ---------- 高风险 diff ----------
: > "$PKG/high.diff"
if [ ${#high[@]} -gt 0 ]; then
  git -C "$WT" diff --unified=8 "$BASE...$SHA" -- "${high[@]}" > "$PKG/high.diff" 2>/dev/null || true
fi
# 新增文件也必须有正文，否则 --diff-filter=d 之外的路径（重命名等）会漏
high_diff_full=$(wc -c < "$PKG/high.diff" | tr -d ' ')

# 有高风险文件却拿不到 diff = 材料本身就是坏的，别把空包发出去让模型猜。
if [ ${#high[@]} -gt 0 ] && [ "$high_diff_full" -eq 0 ]; then
  printf 'packet: 高风险文件 %s 存在但 diff 为 0 字节（base=%s），拒绝产出空材料\n' \
    "${#high[@]}" "$BASE" >&2
  exit 1
fi

# ---------- 高风险文件全文（按 ordered 顺序，超预算的显式标注省略） ----------
# 有 PACKET_EXTRA（= 评审明确点名要全文）时，本轮是"补材料"轮，预算放宽：
# 否则上一轮已给的大文件会把新点名的文件挤掉，补一轮等于白补。
file_budget=$PACKET_MAX_FILES_BYTES
if [ ${#extra[@]} -gt 0 ]; then
  file_budget=${PACKET_EXTRA_MAX_FILES_BYTES:-600000}
fi
copied=0
omitted=()   # "path<TAB>bytes<TAB>原因"
full=()      # 实际拿到全文的文件（下一轮要保住它们，别被新增的 PACKET_EXTRA 挤掉）
for f in "${high[@]}"; do
  dst="$PKG/files/$f"
  sz=$(git -C "$WT" cat-file -s "$SHA:$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt "$file_budget" ]; then
    omitted+=("$f"$'\t'"$sz"$'\t'"单文件超过全文总预算")
    continue
  fi
  if [ $((copied + sz)) -gt "$file_budget" ]; then
    omitted+=("$f"$'\t'"$sz"$'\t'"超出全文总预算（前序文件已用满）")
    continue
  fi
  mkdir -p "$(dirname "$dst")" || { omitted+=("$f"$'\t'"$sz"$'\t'"建目录失败"); continue; }
  if ! git -C "$WT" show "$SHA:$f" > "$dst" 2>/dev/null; then
    rm -f "$dst"   # 别留下空文件让清单"看起来有全文"
    omitted+=("$f"$'\t'"$sz"$'\t'"读取失败")
    continue
  fi
  copied=$((copied + sz))
  full+=("$f")
done

# ---------- 低风险摘要（便宜模型；失败则退回文件清单） ----------
low_list=$(printf '%s\n' "${low[@]}" 2>/dev/null | sed '/^$/d')
: > "$PKG/low.summary.md"
# 先把"逐文件改动统计"写死（事实），再叠模型摘要（观点）。
# 只给模型摘要时，PR #16 的 5 个文件最后只剩一句对不存在路径的猜测，
# 真正的 bench.sh 改动（+40/−8）在材料里完全看不见。
{
  if [ -z "$low_list" ]; then
    echo "（没有低风险文件：改动全部按高风险处理）"
  fi
  if [ -n "$low_list" ]; then
    echo "### 逐文件改动统计"
    while IFS= read -r f; do
      read -r a d <<<"$(numstat_pair "$f")"
      printf -- '- `%s` (+%s −%s)\n' "$f" "$a" "$d"
    done <<<"$low_list"
    echo
  fi
} > "$PKG/low.summary.md"

if [ "$PACKET_SUMMARY" = 1 ] && [ ${#low[@]} -gt 0 ]; then
  git -C "$WT" diff --unified=2 "$BASE...$SHA" -- "${low[@]}" 2>/dev/null \
    | head -c "$PACKET_LOW_DIFF_BYTES" > "$PKG/low.diff" || true
  if [ -s "$PKG/low.diff" ]; then
    {
      printf '你在为一个 PR 做"低风险文件摘要"，供昂贵的 review 模型做政策检查。\n'
      printf '要求：逐个文件一行，写成 `路径: 改了什么`；只描述 diff 里能看到的事实，\n'
      printf '不要写 approve/reject，不要猜测"某文件不存在"或其它 diff 之外的问题，不要改文件。\n'
      printf '文件清单：\n%s\n\n改动：\n' "$low_list"
      cat "$PKG/low.diff"
    } > "$PKG/low.prompt"
    if timeout "${PACKET_SUMMARY_TIMEOUT:-900}" codex exec -C "$WT" -m "$PACKET_SUMMARY_MODEL" \
         -c approval_policy=never --json - \
         < "$PKG/low.prompt" > "$PKG/low.agent.jsonl" 2>&1; then
      # 取**最后一个** agent_message 的完整 text：jq -r 输出是多行，再接 tail -1
      # 只会留下最后一行，逐文件摘要就被截成一句（PR #19 三轮评审指出）。
      s=$(jq -rs '[.[] | select(.type=="item.completed" and .item.type=="agent_message")
                   | .item.text] | last // empty' "$PKG/low.agent.jsonl" 2>/dev/null || true)
      if [ -n "$s" ]; then printf '\n### 模型摘要（观点，仅供政策检查）\n\n%s\n' "$s" >> "$PKG/low.summary.md"; fi
    fi
  fi
fi

# 政策文本也放进材料包（GPT 需要对照 AGENTS.md 判违规）
if [ -f "$WT/AGENTS.md" ]; then cp "$WT/AGENTS.md" "$PKG/AGENTS.md"; fi

# ---------- bench 证据 ----------
# 关键：把"门禁是否适用"讲清楚。旧版对任何 PR 都写"没有 bench 结果（性能类 PR
# 不能 approve）"，于是仅改 bench/ 或 docs 的 PR 永远拿不到 approve（PR #16）。
bench_file="$PRWATCH_HOME/bench/pr-$PR-$SHA.json"
need_bench=0
case "$BENCH_GATE" in
  required) need_bench=1;;
  auto) grep -qE '^(src|ggml|tools|common|examples)/' "$OUT/changed-files.txt" && need_bench=1 || true;;
esac
{
  echo "### bench 门禁状态"
  echo
  if [ "$need_bench" -eq 1 ]; then
    if [ -s "$bench_file" ]; then
      echo "- 判定：**适用**，已找到本 head 的 bench 结果"
    else
      echo "- 判定：**适用且缺失** → 性能结论不成立，属 blocking（本 PR 改了运行时路径）"
    fi
  else
    echo "- 判定：**不适用**（本 PR 未触及 bench 门禁路径，或 BENCH_GATE=off）"
    echo "- 口径：bench 结果缺失**不是** blocking 理由；若 PR 里有性能宣称但材料没证据，"
    echo "  写进 non_blocking，不要据此判 blocked。文档/实验要求的核对见 AGENTS.md §1。"
  fi
  echo
  echo "### bench 结果"
  if [ -s "$bench_file" ]; then
    echo '```json'
    jq -c '{metrics, correctness, created_at}' "$bench_file" 2>/dev/null || cat "$bench_file"
    echo '```'
    echo
    echo "### verify_bench 输出"
    echo '```'
    if [ "$BENCH_REQUIRE_EVIDENCE" = 1 ]; then
      python3 "$SCRIPT_DIR/verify_bench.py" "$bench_file" --pr "$PR" --sha "$SHA" \
        --evidence "$PRWATCH_HOME/bench/pr-$PR-$SHA-evidence" 2>&1 || true
    else
      python3 "$SCRIPT_DIR/verify_bench.py" "$bench_file" --pr "$PR" --sha "$SHA" 2>&1 || true
    fi
    echo '```'
  else
    echo "没有本 head 的 bench 结果文件。"
  fi
} > "$PKG/bench.md"

# ---------- packet.md ----------
meta=$(jq -r '"\(.title)\n作者：\(.author.login)\nhead：\(.headRefName) @ \(.headRefOid)\nbase：\(.baseRefName)"' "$OUT/meta.json" 2>/dev/null || echo "")
{
  echo "# 评审材料（packet 模式）"
  echo
  echo "本文件所在目录就是你的工作目录（沙箱里是只读的），仓库不可访问。"
  echo
  echo "- 高风险文件全文：\`files/<路径>\`（下表标 \`full\` 的才有，标 \`omitted\` 的没有）"
  echo "- 基线政策：\`AGENTS.md\`"
  echo "- 下表 \`omitted\` 的文件如果确实必须逐行核对，写进 \`needs_full_diff\`"
  echo "  （只写本 PR 改动过的路径），编排器会自动重开一轮把全文补进来。"
  echo
  echo "## PR"
  echo '```'
  printf '%s\n' "$meta"
  echo '```'
  echo
  echo "> diff 以 merge-base 为基准：若 PR 分支落后 main，diff 里可能出现已经在 main 上的文件"
  echo "> （那不是本 PR 的新增内容）。"
  echo
  echo "## T0 静态门禁"
  echo '```'
  cat "$OUT/t0.txt" 2>/dev/null
  echo '```'
  echo
  echo "T1（构建/ctest）未运行是 WARN，不是 blocking；只有 T1 明确 FAIL 才 blocking。"
  echo
  echo "## bench"
  cat "$PKG/bench.md"
  echo
  echo "## AGENTS.md"
  echo
  echo "完整内容见 \`AGENTS.md\`（与本文件同目录）。"
  echo
  echo "## 低风险文件（摘要，仅供政策检查；不要当成结论）"
  echo
  cat "$PKG/low.summary.md"
  echo
  echo "## 高风险文件清单"
  echo
  echo "| 文件 | 全文 | 字节 | +/- |"
  echo "|---|---|---|---|"
  for f in "${high[@]}"; do
    sz=$(git -C "$WT" cat-file -s "$SHA:$f" 2>/dev/null || echo 0)
    read -r a d <<<"$(numstat_pair "$f")"
    mark=full
    for o in "${omitted[@]}"; do
      case "$o" in "$f"$'\t'*) mark='omitted';; esac
    done
    printf '| `%s` | %s | %s | +%s −%s |\n' "$f" "$mark" "$sz" "$a" "$d"
  done
  echo
  if [ ${#omitted[@]} -gt 0 ]; then
    echo "### 未提供全文的文件（材料缺口，不是代码问题）"
    echo
    while IFS=$'\t' read -r f sz why; do
      printf -- '- `%s`（%s 字节）：%s，全文未放入材料；diff 仍在下面\n' "$f" "$sz" "$why"
    done <<<"$(printf '%s\n' "${omitted[@]}")"
    echo
  fi
  echo "## 高风险 diff"
  echo '```diff'
  if [ "$high_diff_full" -gt "$PACKET_MAX_DIFF_BYTES" ]; then
    head -c "$PACKET_MAX_DIFF_BYTES" "$PKG/high.diff" 2>/dev/null || true
    printf '\n... [diff 已截断：全文 %s KB，这里只有前 %s KB；后续文件请用 needs_full_diff 点名]\n' \
      "$((high_diff_full / 1024))" "$((PACKET_MAX_DIFF_BYTES / 1024))"
  else
    cat "$PKG/high.diff" 2>/dev/null || true
  fi
  echo '```'
} > "$PKG/packet.md"

# high.diff 落盘保留全文，但材料里只给前 PACKET_MAX_DIFF_BYTES：统计要报"实际给了多少"，
# 否则评论会写成"全文 2700 KB，只给前 2700 KB"这种自相矛盾的话。
kept_diff=$high_diff_full
[ "$high_diff_full" -gt "$PACKET_MAX_DIFF_BYTES" ] && kept_diff=$PACKET_MAX_DIFF_BYTES
jq -nc --argjson high "${#high[@]}" --argjson low "${#low[@]}" \
  --argjson high_diff "$high_diff_full" \
  --argjson high_diff_kept "$kept_diff" \
  --argjson packet "$(wc -c < "$PKG/packet.md" | tr -d ' ')" \
  --argjson files "$copied" --argjson omitted "${#omitted[@]}" \
  --argjson extra "${#extra[@]}" \
  --argjson full "$(printf '%s\n' "${full[@]:-}" | sed '/^$/d' | jq -R . | jq -sc .)" \
  --argjson budget "$file_budget" \
  --argjson truncated "$([ "$high_diff_full" -gt "$PACKET_MAX_DIFF_BYTES" ] && echo true || echo false)" \
  '{high_files:$high, low_files:$low, high_diff_bytes:$high_diff, high_diff_kept_bytes:$high_diff_kept, packet_bytes:$packet, file_bytes:$files, omitted_files:$omitted, extra_files:$extra, full_files:$full, full_budget_bytes:$budget, diff_truncated:$truncated}' \
  > "$PKG/stats.json"
