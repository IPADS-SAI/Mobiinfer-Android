#!/usr/bin/env bash
# fail closed：材料补轮用尽后，模型仍点名要"本 PR 改过的文件"的全文，说明它没看到
# 自己想看的东西。此时若结论是 approve，等于给"没审到的那部分"签了字——降级为 blocked。
#
# 用法：fail-closed.sh <verdict.json> <未满足的路径（换行分隔）> <已跑轮数>
#   需要降级：把改写后的 verdict 打到 stdout，退出码 0
#   无需降级：不输出，退出码 3
set -euo pipefail

V=${1:?usage: fail-closed.sh <verdict.json> <paths> <attempts>}
MORE=${2:-}
ATTEMPTS=${3:-1}

[ -s "$V" ] || exit 3
[ -n "$MORE" ] || exit 3
[ "$(jq -r '.conclusion // ""' "$V" 2>/dev/null)" = approve ] || exit 3

joined=$(printf '%s' "$MORE" | paste -sd, -)
jq --arg m "$joined" --argjson n "$ATTEMPTS" '
  .conclusion = "blocked"
  | .blocking = ((.blocking // [])
      + ["材料在 \($n) 轮后仍不足，模型点名需要：" + $m
         + "。为避免对未审到的代码放行，approve 自动降级为 blocked（可用 PACKET_EXTRA 补全后再审）。"])
' "$V"
