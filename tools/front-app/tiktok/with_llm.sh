#!/bin/bash
# tiktok/with_llm.sh [--rounds 4] [--dur 120] [--n 64] [--outdir DIR]
# 抖音前台刷视频 + llama-cli 后台：量化视频流对 LLM 生成的干扰
HERE=$(cd "$(dirname "$0")" && pwd)
args=(--pkg com.ss.android.ugc.aweme --app-run "$HERE/appwork.sh" --label tiktok)
while [ $# -gt 0 ]; do case "$1" in
  --rounds|--dur|--n|--outdir|--warm|--bench-env) args+=("$1" "$2"); shift 2;;
  *) echo "未知参数 $1"; exit 2;; esac; done
exec bash "$HERE/../shared/with_llm.sh" "${args[@]}"
