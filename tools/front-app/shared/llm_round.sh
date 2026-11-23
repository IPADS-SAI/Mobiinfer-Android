#!/data/data/com.termux/files/usr/bin/bash
# llm_round.sh —— 手机侧：在 devlock 内跑一轮 llama-cli（sweep.sh），并记录本轮的
# 墙钟窗口 + 前台守护结果，供 analyze.py 与 App 采样对齐。
#
# 用法：llm_round.sh <tag> <watch_pkg|-> <llm_windows.log 的手机侧路径>
#   tag        本轮标签（写进 runs/results.tsv）
#   watch_pkg  兼容保留；前台守护改由 host 侧 adb 的 .fg.txt 做
#   logpath    手机侧日志路径，会追加 "tag start_ms end_ms rc fg_lost"
#
# B 组（llm-only）与 C 组（with app）都走本脚本，唯一差别是 watch_pkg，
# 这样两组的 CPU 调度域（Termux 在后台）完全一致。
set -u
TAG=${1:?tag}; WATCH=${2:--}; LOGPATH=${3:-}
export PATH=/system/bin:$PATH
ROOT=${ROOT:-$HOME/csm/moe-exp}
export ROOT
cd "$ROOT" || exit 1
for f in bench.sh sweep.sh dropcache.py devlock; do
  [ -e "$ROOT/$f" ] || {
    echo "[llm_round] 缺少 $ROOT/$f；按 tools/front-app/README.md §3 部署 bench/ 四个文件" >&2
    exit 1
  }
done

run_round() {
  # 1) 冷缓存（必须在锁内）。dropcache.py 无参数时只清默认 MiniCPM 4.7 目录；
  #    换模型时由 with_llm.sh 通过 MODEL_DIR 传入实际的 gguf/experts 路径。
  if [ -n "${MODEL_DIR:-}" ]; then
    python3 ./dropcache.py "$MODEL_DIR"/*.gguf "$MODEL_DIR"/experts-htp-tiled/*.bin >/dev/null 2>&1 || true
  else
    python3 ./dropcache.py >/dev/null 2>&1 || true
  fi

  # 2) 跑 sweep.sh
  local t0 t1 rc
  t0=$(date +%s%3N)
  TAG="$TAG" bash ./sweep.sh   # 其余 env（WINDOW/IO_THREADS/SPEC...）由调用方 export
  rc=$?
  t1=$(date +%s%3N)

  # 前台守护不在这里做：Termux 进程通常没有 android.permission.DUMP，
  # dumpsys window 会返回空，把整轮误判成 fg_lost。改由 host 侧 adb 的
  # fa_sample 写 <tag>.fg.txt，with_llm.sh 读完再计数（adb shell 有 DUMP 权限）。
  echo "[llm_round] window ${t0}..${t1} rc=$rc fg_lost=0" >&2
  [ -n "$LOGPATH" ] && echo "$TAG $t0 $t1 $rc 0" >> "$LOGPATH"
  return $rc
}

export -f run_round 2>/dev/null || true
export TAG LOGPATH
./devlock bash -c 'run_round'
