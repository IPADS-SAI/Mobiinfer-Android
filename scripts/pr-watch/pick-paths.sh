#!/usr/bin/env bash
# 从自由文本（评审模型的 needs_full_diff 条目）里挑出"确实在本 PR 改动清单里"的路径。
#
# 用法：jq -r '.needs_full_diff[]?' verdict.json | pick-paths.sh <changed-files.txt>
#
# 做法：反过来遍历改动清单，看每条路径是否在文本里"以完整路径的形态"出现过。
# 匹配两侧要求不是路径字符（[A-Za-z0-9_./-]），所以 `src/a.c` 不会命中 `src/a.cpp`。
#
# 为什么不用"按扩展名抠 token"：
#   - `Makefile`、`Dockerfile`、`config.env`、`CMakePresets.json` 没有代码扩展名会被漏；
#   - 反过来会把 `verdict.schema.json` 抠成 `verdict.s`（命中了 `s` 这个汇编扩展名）；
#   - 按空格切词会把**含空格的路径**（`bench/foo bar.sh`）拆坏；
#   - 中文标点是多字节，tr 按字节处理还会切出乱码 token。
# 只有与改动清单逐字一致的路径才会被选中，模型写什么都塞不进无关文件。
set -euo pipefail

list=${1:?usage: pick-paths.sh <changed-files.txt>}
[ -f "$list" ] || exit 0

raw=$(mktemp)
trap 'rm -f "$raw"' EXIT
cat > "$raw"
[ -s "$raw" ] || exit 0

B='[^A-Za-z0-9_./-]'   # 路径边界：不能是路径字符

while IFS= read -r path; do
  [ -z "$path" ] && continue
  # ERE 转义：把正则元字符加上反斜杠
  esc=$(printf '%s' "$path" | sed 's/[][\\.^$*+?(){}|]/\\&/g')
  # 两种合法出现形态：
  #   1) 前后都不是路径字符（普通引用，如 `src/a.cpp` 或行尾）；
  #   2) 后面紧跟句末标点再跟空白/行尾（如 "please provide src/a.cpp."）——
  #      句号本身是路径字符，严格边界会把它当成 `src/a.cpp.` 的一部分而漏掉。
  if grep -qE "(^|${B})${esc}(${B}|\$)" "$raw" \
     || grep -qE "(^|${B})${esc}[.,;:!?]([[:space:]]|\$)" "$raw"; then
    printf '%s\n' "$path"
  fi
done < "$list"
