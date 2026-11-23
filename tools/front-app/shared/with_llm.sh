#!/bin/bash
# with_llm.sh —— 前台 App + 后台 llama-cli 同时跑，采同一时间轴（所有 App 共用）
#
# 用法（由各 App 的 with_llm.sh 调用）：
#   with_llm.sh --pkg <包名> --app-run <启动+制造负载的脚本> --label <名> \
#               [--rounds 4] [--dur 120] [--n 64] [--outdir <dir>]
#
# 每组配置：
#   B  llm-only    : 只跑 llama-cli（Termux 切后台），作对照
#   C  with-app    : App 在前台制造负载 + 同时跑 llama-cli
# 顺序：ABBA 交错（C B B C C B B C …），避免热漂移把结论带偏。
#
# 产出（<outdir>/）：
#   rounds.tsv            每轮：round config tag gen_tps prompt_tps wall_ms fg_lost
#   <tag>.dev.csv         runp：设备 busy/iowait + App CPU
#   <tag>.app.csv         RSS / 网络 / App CPU
#   <tag>.fg.txt          前台守护采样
#   llm_windows.log       每轮 llama 的墙钟窗口
#   summary.txt           analyze.py 的结论
set -u
SHARED=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SHARED/lib.sh"

PKG=""; APP_RUN=""; LABEL=""; ROUNDS=2; DUR=120; N=64; WARM=0; PERSIST=0; BENCH_ENV=""
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pkg) PKG=$2; shift 2 ;;
    --app-run) APP_RUN=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --rounds) ROUNDS=$2; shift 2 ;;
    --dur) DUR=$2; shift 2 ;;
    --n) N=$2; shift 2 ;;
    --outdir) OUT=$2; shift 2 ;;
    --warm) WARM=$2; shift 2 ;;
    --bench-env) BENCH_ENV=$2; shift 2 ;;   # 透传给 bench.sh 的额外 env，如 "WINDOW=32 IO_THREADS=8"
    --persist) PERSIST=1; shift ;;   # App 状态保留在后台，不每轮 force-stop
                                      # （原神进世界要 ~90s，不能每轮重进）
    *) die "未知参数 $1" ;;
  esac
done
[ -n "$PKG" ] || die "需要 --pkg"
[ -n "$APP_RUN" ] && [ -x "$APP_RUN" ] || die "--app-run 必须给一个可执行脚本"
LABEL=${LABEL:-$(basename "$PKG")}

STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-$FRONT_APP_HOME/artifacts/$LABEL-with-llm-$STAMP}
mkdir -p "$OUT"
MOB_OUT=$PHONE_MOE/runs/front-app/$LABEL-$STAMP
phone_ssh "mkdir -p $MOB_OUT" >/dev/null 2>&1 || true

fa_install_tools
sh_ "cat > $D/llm_round.sh" < "$SHARED/llm_round.sh"
sh_ "chmod 755 $D/llm_round.sh" >/dev/null
# 拷贝到 Termux 侧（devlock/bench 都在 ~/csm/moe-exp 下工作）
scp_phone() { "${ADB[@]}" push "$1" "$D/$(basename "$1")" >/dev/null 2>&1; }
scp_phone "$SHARED/llm_round.sh"

# 手机侧执行一轮 llama（与 skills/bench 相同的调用路径：bench.sh -> llama-cli）。
# N 等短名 env 在这里 export，llm_round.sh 内部调 sweep.sh 时会继承。
# 必须在 Termux uid 下跑：llama-cli 与模型都在 ~/csm/moe-exp（u0_a371, 700），
# adb shell 是 uid 2000，进不去这个目录。所以 llama 一律走 ssh（$PHONE_SSH）。
phone_llm_round() {   # $1=tag  $2=watch_pkg|-  $3=llm_windows.log 路径
  phone_ssh "export N=$N ROOT=\"$PHONE_MOE\" MODEL_DIR=\"${MODEL_DIR:-}\" $BENCH_ENV; cd \"$PHONE_MOE\" && bash $D/llm_round.sh '$1' '$2' '$3'"
}

round_tag() { echo "$LABEL-$1-r$2"; }

gen_from_out() {  # 从 Termux 侧 runs/<tag>-n<N>.out 取 Generation t/s
  phone_ssh "grep -aE 'Generation: [0-9.]+ t/s' \"$PHONE_MOE/runs/$1-n$N.out\" 2>/dev/null | tail -1 | sed -E 's/.*Generation: ([0-9.]+) t\\/s.*/\\1/'" | tr -d '\r'
}
prompt_from_out() {
  phone_ssh "grep -aE 'Prompt: [0-9.]+ t/s' \"$PHONE_MOE/runs/$1-n$N.out\" 2>/dev/null | tail -1 | sed -E 's/.*Prompt: ([0-9.]+) t\\/s.*/\\1/'" | tr -d '\r'
}

# ---------- 预热（一次性）----------
# 目的：让 App 把"首次运行才会做的大动作"（资源包解压/建缓存/建 DB/dexopt）做完。
# 实测抖音：首次 1218 MiB/60s -> 预热后 82 MiB/60s，干扰从 -28% 降到 -3%。
# 注意：预热的是 App 自己的持久化状态，不是 llama 的专家 page cache（每轮都被 dropcache 丢掉）。
free_blocks() { sh_ 'df /data | tail -1 | awk "{print \$4}"' | tr -d '\r'; }

if [ "$WARM" -gt 0 ]; then
  log "预热 App ${WARM}s（让首次初始化落盘）..."
  fa_force_stop "$PKG"; sleep 2
  W0=$(free_blocks); WT0=$(now_ms)
  rm -f "$OUT/warmup.stop" "$OUT/warmup.stop.ready"
  # 后台跑 appwork，等它就绪后计 WARM 秒，再放 stop；不能同步调用：
  # 微信等 appwork 只等 stop 文件，同步调用会永久挂住。
  "$APP_RUN" "$OUT/warmup.stop" "$WARM" > "$OUT/warmup.log" 2>&1 &
  warm_pid=$!
  for i in $(seq 1 240); do
    [ -f "$OUT/warmup.stop.ready" ] && break
    sleep 1
  done
  sleep "$WARM"
  touch "$OUT/warmup.stop"
  wait "$warm_pid" 2>/dev/null || true
  rm -f "$OUT/warmup.stop" "$OUT/warmup.stop.ready"
  WT1=$(now_ms); W1=$(free_blocks)
  WMI=$(python3 -c "print(round(($W1-$W0)*4/1024,1))")
  fa_force_stop "$PKG"; sleep 3
  {
    echo "warm_secs=$WARM"
    echo "warm_wall_ms=$((WT1-WT0))"
    echo "warm_disk_delta_mib=$WMI"
    echo "warm_disk_rate_mib_s=$(python3 -c "print(round($WMI/max($WARM,1),3))")"
  } | tee "$OUT/warmup.txt"
  log "预热完成：写盘 ${WMI} MiB（$(python3 -c "print(round($WMI/max($WARM,1),2))") MiB/s）"
else
  log "⚠️ 未指定 --warm：若该 App 是首次运行，C 组可能撞上一次性初始化，"
  log "   干扰会被高估（实测差 40 倍）。建议加 --warm 120。"
  echo "warm_secs=0" > "$OUT/warmup.txt"
fi

printf "round\tconfig\ttag\tgen_tps\tprompt_tps\tfg_lost\tround_t0_ms\n" > "$OUT/rounds.tsv"

# ABBA：组 C / B 的排布
seq_abba() {
  local r
  for ((r=1; r<=ROUNDS; r++)); do
    if [ $(( r % 4 )) -eq 1 ] || [ $(( r % 4 )) -eq 0 ]; then echo C; else echo B; fi
  done
}

r=0
for cfg in $(seq_abba); do
  r=$((r+1))
  tag=$(round_tag "$cfg" "$r")
  log "round $r/$ROUNDS  config=$cfg  tag=$tag"

  # --- 两组共同的调度域：Termux 切后台
  sh_ 'input keyevent KEYCODE_HOME' >/dev/null 2>&1; sleep 1

  # persist 模式下，B 组必须把 App 停掉才是真正的 llm-only
  if [ "$PERSIST" = 1 ] && [ "$cfg" = B ]; then
    fa_force_stop "$PKG"; sleep 3
  fi

  if [ "$cfg" = C ]; then
    # --- C 组：先把 App 拉到前台并稳定，再起 llama
    rm -f "$OUT/$tag.stop" "$OUT/$tag.stop.ready"
    # 0 = 不按自身时长退出，由本轮 llama 结束后的 stop 文件停止
    "$APP_RUN" "$OUT/$tag.stop" 0 >"$OUT/$tag.apprun.log" 2>&1 &
    local_app=$!
    # 等 App 就绪信号（原神要先进世界，可能要 2-3 分钟；微信/抖音几秒）
    for i in $(seq 1 240); do
      [ -f "$OUT/$tag.stop.ready" ] && break
      sleep 1
    done
    if [ -f "$OUT/$tag.stop.ready" ]; then
      log "  App 就绪（${i}s）"
    else
      log "  WARN: 等了 240s 没收到就绪信号，按当前状态继续"
    fi
    fa_is_foreground "$PKG" || log "  WARN: $PKG 不在前台（analyze 会标记）"
  fi

  # --- 起采集 + 起 llama（同一时刻开始，analyze 靠 llm_windows.log 对齐）
  ( fa_sample "$PKG" "$DUR" "$OUT" "$tag" ) &
  SMP=$!
  # 本轮前台守护：覆盖整个 llama 窗口（含 devlock 等待），不能只靠 fa_sample 的 DUR+3。
  # focused_pkg 走 adb shell（有 DUMP 权限）；空输出记 UNKNOWN，视为无效证据。
  FG_PID=""
  FG_STOP=""
  if [ "$cfg" = C ]; then
    FG_STOP="$OUT/$tag.fg.stop"
    rm -f "$FG_STOP" "$OUT/$tag.fg_round.txt"
    ( while [ ! -f "$FG_STOP" ]; do
        fg=$(focused_pkg)
        echo "$(now_ms) ${fg:-UNKNOWN}" >> "$OUT/$tag.fg_round.txt"
        sleep 1
      done ) &
    FG_PID=$!
  fi
  # 本轮开始时间，供 analyze.py 对齐（单位 ms，与 llm_windows.log 同源）
  ROUND_T0=$(now_ms)
  phone_llm_round "$tag" "$([ "$cfg" = C ] && echo "$PKG" || echo "-")" "$MOB_OUT/llm_windows.log" \
      > "$OUT/$tag.llm.log" 2>&1
  wait "$SMP" 2>/dev/null
  if [ -n "$FG_PID" ]; then
    touch "$FG_STOP"
    wait "$FG_PID" 2>/dev/null || true
  fi

  [ "$cfg" = C ] && { echo "stop" > "$OUT/$tag.stop" 2>/dev/null; wait "$local_app" 2>/dev/null; }

  # 注意：moe-ssd 的 UFS 统计打在 stderr（.err），tok/s 打在 stdout（.out），两个都要
  phone_ssh "cat \"$PHONE_MOE/runs/$tag-n$N.out\" 2>/dev/null" > "$OUT/$tag.llm.out" 2>/dev/null || true
  phone_ssh "cat \"$PHONE_MOE/runs/$tag-n$N.err\" 2>/dev/null" > "$OUT/$tag.llm.err" 2>/dev/null || true
  gen=$(gen_from_out "$tag"); prm=$(prompt_from_out "$tag")
  # 前台守护用 host 侧 adb 采的 <tag>.fg_round.txt（Termux 进程通常没有 DUMP 权限）。
  # fail-closed：文件缺失、样本不足、空/UNKNOWN、任何一次不是目标包 => 整轮 fg_lost=1。
  # B 组本来就该是后台，不参与这个判定。
  lost=0
  if [ "$cfg" = C ]; then
    fg_file="$OUT/$tag.fg_round.txt"
    fg_n=0
    fg_bad=1
    if [ -s "$fg_file" ]; then
      fg_n=$(wc -l < "$fg_file")
      fg_bad=$(awk -v p="$PKG" 'NF < 2 || $2 == "" || $2 == "UNKNOWN" || $2 != p {n++} END {print n+0}' "$fg_file")
    fi
    if [ "$fg_n" -lt 2 ] || [ "$fg_bad" -gt 0 ]; then
      lost=1
    fi
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$r" "$cfg" "$tag" "${gen:-NA}" "${prm:-NA}" "$lost" "$ROUND_T0" >> "$OUT/rounds.tsv"
  log "  -> gen=${gen:-NA} t/s  prompt=${prm:-NA} t/s"

  # 轮间冷却。--persist 时不关 App（否则原神要重新登录进世界）
  if [ "$PERSIST" = 1 ]; then
    [ "$cfg" = B ] && fa_force_stop "$PKG"      # B 组要真正的 llm-only
    sleep 5
  else
    fa_force_stop "$PKG"; sleep 5
  fi
done

phone_ssh "cat $MOB_OUT/llm_windows.log 2>/dev/null" > "$OUT/llm_windows.log" 2>/dev/null
python3 "$SHARED/analyze.py" "$OUT" | tee "$OUT/summary.txt"
rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ]; then
  log "完成 -> $OUT"
else
  log "分析失败（rc=$rc），原始产物保留在 $OUT"
fi
exit "$rc"
