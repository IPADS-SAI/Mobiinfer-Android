#!/bin/bash
# genshin/with_llm.sh [--rounds 4] [--dur 180] [--n 64] [--outdir DIR]
# 原神前台（游戏世界）+ llama-cli 后台。要求：开始前已手动进入游戏世界。
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../shared/lib.sh"
# 不再要求先进世界：appwork.sh 会在需要时执行 launch.sh --enter 自动重进
# （进度在服务器，force-stop 后 launch + 点"点击进入" + 等 ~110s 即可）
args=(--pkg com.miHoYo.Yuanshen --app-run "$HERE/appwork.sh" --label genshin)
while [ $# -gt 0 ]; do case "$1" in
  --rounds|--dur|--n|--outdir|--warm|--bench-env) args+=("$1" "$2"); shift 2;;
  *) echo "未知参数 $1"; exit 2;; esac; done
exec bash "$HERE/../shared/with_llm.sh" "${args[@]}"
