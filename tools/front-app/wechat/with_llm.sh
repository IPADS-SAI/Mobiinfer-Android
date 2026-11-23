#!/bin/bash
# wechat/with_llm.sh [--rounds 4] [--dur 120] [--n 64] [--outdir DIR]
# 微信前台 + llama-cli 后台：量化"微信在跑时 LLM 生成被干扰多少、干扰在哪类资源"
HERE=$(cd "$(dirname "$0")" && pwd)
args=(--pkg com.tencent.mm --app-run "$HERE/appwork.sh" --label wechat)
while [ $# -gt 0 ]; do case "$1" in
  --rounds|--dur|--n|--outdir|--warm|--bench-env) args+=("$1" "$2"); shift 2;;
  *) echo "未知参数 $1"; exit 2;; esac; done
exec bash "$HERE/../shared/with_llm.sh" "${args[@]}"
