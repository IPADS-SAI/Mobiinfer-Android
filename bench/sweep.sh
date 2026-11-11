#!/data/data/com.termux/files/usr/bin/bash
# sweep.sh v2 — 一次采样；用 stdbuf -oL 让 llama-cli 的计时行在 teardown 崩溃前落盘。
# 输出 runs/results.tsv: tag n wall_ms rc gen_tps prompt_tps
set -u
ROOT=${ROOT:-$HOME/csm/moe-exp}
TAG=${TAG:-run}
N=${N:-64}
mkdir -p "$ROOT/runs"
out="$ROOT/runs/$TAG-n$N.out"; err="$ROOT/runs/$TAG-n$N.err"
t0=$(date +%s%N)
N=$N stdbuf -oL -eL bash "$ROOT/bench.sh" >"$out" 2>"$err"
rc=$?
t1=$(date +%s%N)
ms=$(( (t1 - t0) / 1000000 ))
gen=$(grep -aE "Generation: [0-9.]+ t/s" "$out" | tail -1 | sed -E 's/.*Generation: ([0-9.]+) t\/s.*/\1/')
prm=$(grep -aE "Prompt: [0-9.]+ t/s" "$out" | tail -1 | sed -E 's/.*Prompt: ([0-9.]+) t\/s.*/\1/')
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$TAG" "$N" "$ms" "$rc" "${gen:-NA}" "${prm:-NA}" >> "$ROOT/runs/results.tsv"
echo "[sweep] $TAG n=$N ${ms}ms rc=$rc gen=${gen:-NA} t/s"
