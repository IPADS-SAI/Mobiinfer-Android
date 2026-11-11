#!/usr/bin/env bash
# app_bench.sh — 在 MobiInfer Android app 里跑一条 prompt，采集 app 自己写的
# benchmark.jsonl（wall / ttft / prompt / decode）与 UI 指标，按 CLI bench 的口径记录。
#
# 与 bench/bench.sh 的对应关系见 bench/app/README.md。要点：
#   * 每条 run 先 force-stop 再启动，保证是**新会话**（无历史），prompt token 可比；
#   * app 的 n_predict 固定 1024（CLI 用 -n）；decode t/s 是原生 server timing，可直接比；
#   * 记录 app 私有的 files/models/<model>/benchmark.jsonl 最后一条，不改 app 源码。
#   * 任一 run 失败（输入框/输入/超时/进程消失/记录非法）都会写 status 并以非 0 退出。
#
# 用法:
#   ./app_bench.sh [-t TAG] [-r RUNS] [-p PROMPT | -f PROMPT_FILE]
#                  [-m MODEL_ID] [-s SERIAL] [-o OUT_ROOT]
# 例:
#   ./app_bench.sh -t rain -r 3 -f ../prompts/rain.txt
set -euo pipefail
set -f   # prompt 里的 * ? [ ] 不做路径展开

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SERIAL=${SERIAL:-}
PKG=${PKG:-com.mobiinfer.demo}
MODEL_ID=${MODEL_ID:-minicpm-v-4_7}
TAG=${TAG:-app}
RUNS=${RUNS:-1}
OUT_ROOT=${OUT_ROOT:-$HERE/runs}
PROMPT=${PROMPT:-"Explain how rain forms in clear simple English."}
PROMPT_FILE=${PROMPT_FILE:-}
WAIT_S=${WAIT_S:-360}
PHONE_SSH=${PHONE_SSH:-vivo}
DEV_LOCK=${DEV_LOCK:-on}        # on | off; any non-off value requires the lock
LOCK_WAIT=${LOCK_WAIT:-300}

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

while getopts ":t:r:p:f:m:s:o:h" opt; do
  case "$opt" in
    t) TAG=$OPTARG ;;
    r) RUNS=$OPTARG ;;
    p) PROMPT=$OPTARG ;;
    f) PROMPT_FILE=$OPTARG; PROMPT=$(cat "$PROMPT_FILE") ;;
    m) MODEL_ID=$OPTARG ;;
    s) SERIAL=$OPTARG ;;
    o) OUT_ROOT=$OPTARG ;;
    h) usage; exit 0 ;;
    :) echo "app_bench: -$OPTARG requires an argument" >&2; exit 2 ;;
    \?) echo "app_bench: unknown option -$OPTARG" >&2; exit 2 ;;
  esac
done

case "$RUNS" in ''|*[!0-9]*) echo "app_bench: RUNS must be a positive integer" >&2; exit 2 ;; esac
[ "$RUNS" -ge 1 ] || { echo "app_bench: RUNS must be >= 1" >&2; exit 2; }
case "$WAIT_S" in ''|*[!0-9]*) echo "app_bench: WAIT_S must be a positive integer" >&2; exit 2 ;; esac
[ "$WAIT_S" -ge 10 ] || { echo "app_bench: WAIT_S must be >= 10" >&2; exit 2; }

if [ -z "$SERIAL" ]; then
  SERIAL=$(adb devices | awk 'NR==2 && $2=="device"{print $1}')
fi
[ -n "$SERIAL" ] || { echo "app_bench: no adb device (set SERIAL=)" >&2; exit 1; }
ADB=(adb -s "$SERIAL")

DEV_DIR="files/models/$MODEL_ID"
BENCH_REMOTE="$DEV_DIR/benchmark.jsonl"
RUN_DIR="$OUT_ROOT/$TAG-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
printf 'tag\trun\tstatus\twall_ms\tttft_ms\tprompt_n\tprompt_ms\tdecode_n\tdecode_ms\tdecode_tps\tprefill_tps\tchars\tprompt\n' > "$RUN_DIR/results.tsv"

# prompt 只允许单行；空格用 %s 传给 `input text`，其余字符走远端 shell 单引号转义。
case "$PROMPT" in
  *$'\n'*|*$'\r'*) echo "app_bench: prompt must be single-line" >&2; exit 2 ;;
esac
PROMPT=$(printf '%s' "$PROMPT" | tr -s ' ' | sed 's/^ //; s/ $//')
[ -n "$PROMPT" ] || { echo "app_bench: empty prompt" >&2; exit 2; }

shell_quote() { printf '%s' "$1" | sed "s/'/'\\\\''/g"; }

# 按词切块，避免把 %s 切成 % + s（input text 只认连续的 %s）。
chunks() {
  local s=$1 cur=""
  for w in $s; do
    if [ -z "$cur" ]; then cur=$w
    elif [ ${#cur} -lt 40 ]; then cur="$cur $w"
    else printf '%s \n' "$cur"; cur=$w   # trailing space keeps the word gap
    fi
  done
  [ -n "$cur" ] && printf '%s\n' "$cur"
}

if [ "${SELFTEST:-0}" = 1 ]; then
  tp="Explain how rain forms in clear simple English. Include condensation and evaporation in the answer."
  got=""
  while IFS= read -r c; do got+="$c"; done < <(chunks "$tp")
  if [ "$got" != "$tp" ]; then
    echo "chunks selftest FAILED" >&2
    echo "  want: $tp" >&2
    echo "  got : $got" >&2
    exit 1
  fi
  echo "chunks selftest ok (multi-chunk, spaces preserved)" >&2
  exit 0
fi

dump_ui() {  # $1=local path; 返回非 0 表示 dump/读取失败（调用方必须当失败处理）
  local remote="/sdcard/appbench.$$.$RANDOM.xml"
  "${ADB[@]}" shell rm -f "$remote" >/dev/null 2>&1 || true
  if ! "${ADB[@]}" shell uiautomator dump "$remote" >/dev/null 2>&1; then return 1; fi
  if ! "${ADB[@]}" shell cat "$remote" > "$1" 2>/dev/null; then return 1; fi
  "${ADB[@]}" shell rm -f "$remote" >/dev/null 2>&1 || true
  [ -s "$1" ]
}

bounds_center() {  # $1=xml $2=resource-id -> "x y"
  python3 - "$1" "$2" <<'PY'
import re,sys
xml=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.search(r'resource-id="%s"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'%re.escape(sys.argv[2]), xml)
if not m: sys.exit(1)
a,b,c,d=map(int,m.groups()); print((a+c)//2,(b+d)//2)
PY
}

node_text() {  # $1=xml $2=resource-id -> 解码 XML 实体后的 text
  python3 - "$1" "$2" <<'PY'
import re,sys,html
xml=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.search(r'text="([^"]*)"[^>]*resource-id="%s"'%re.escape(sys.argv[2]), xml)
print(html.unescape(m.group(1)) if m else "")
PY
}

app_pid() { "${ADB[@]}" shell pidof "$PKG" 2>/dev/null | tr -d '\r'; }

wait_for_record() {  # $1=旧 size，$2=超时秒；stdout=新 size；返回 0 成功 / 1 超时 / 2 进程消失
  local before=$1 timeout=$2 t=0 size pid
  while [ "$t" -lt "$timeout" ]; do
    if [ "$LOCK_HELD" = 1 ] && ! kill -0 "$LOCK_PID" 2>/dev/null; then return 3; fi
    sleep 3; t=$((t+3))
    pid=$(app_pid)
    [ -n "$pid" ] || return 2
    size=$("${ADB[@]}" shell run-as "$PKG" stat -c %s "$BENCH_REMOTE" 2>/dev/null | tr -d '\r' || echo 0)
    [ -n "$size" ] || size=0
    if [ "$size" -gt "$before" ]; then echo "$size"; return 0; fi
  done
  return 1
}

classify_error() {  # $1=xml -> 打印可见错误文本（用于失败时留证）
  python3 - "$1" <<'PY'
import re,sys
xml=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
for t in re.findall(r'text="([^"]{6,120})"', xml):
    if any(k in t for k in ("错误", "失败", "Error", "error", "无法", "超时")):
        print(t)
PY
}

LOCK_TOKEN="appbench-$$-$(date +%s)"
LOCK_PID=""
LOCK_HELD=0
LOCK_KEEP=0
HB_PID=""
APP_STARTED=0
CLEANED=0

adb_ident() {
  "${ADB[@]}" shell 'getprop ro.product.model; getprop ro.build.fingerprint; cat /proc/sys/kernel/random/boot_id' 2>/dev/null | tr -d '\r'
}
ssh_ident() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$PHONE_SSH" \
    'getprop ro.product.model; getprop ro.build.fingerprint; cat /proc/sys/kernel/random/boot_id' 2>/dev/null | tr -d '\r'
}
verify_same_device() {
  [ "${SKIP_DEVICE_CHECK:-0}" = 1 ] && { echo "[app_bench] WARN: SKIP_DEVICE_CHECK=1" >&2; return 0; }
  local a s
  a=$(adb_ident)
  s=$(ssh_ident)
  if [ -z "$a" ] || [ -z "$s" ]; then
    echo "app_bench: cannot read device identity (adb='$a' ssh='$s'); refusing (SKIP_DEVICE_CHECK=1 overrides)" >&2
    exit 4
  fi
  if [ "$a" != "$s" ]; then
    echo "app_bench: adb device != ssh device:" >&2
    echo "  adb: $a" | tr '\n' ' ' >&2; echo >&2
    echo "  ssh: $s" | tr '\n' ' ' >&2; echo >&2
    exit 4
  fi
  echo "[app_bench] device identity ok ($(printf '%s' "$a" | head -1))" >&2
}
start_heartbeat() {
  ( while kill -0 "$LOCK_PID" 2>/dev/null; do
      ssh -o BatchMode=yes -o ConnectTimeout=4 "$PHONE_SSH" \
        "cd ~/csm/moe-exp && touch .appbench-$LOCK_TOKEN.hb" >/dev/null 2>&1 || true
      sleep 30
    done ) &
  HB_PID=$!
}
acquire_devlock() {
  if [ "$DEV_LOCK" = off ]; then
    echo "app_bench: DEV_LOCK=off is not allowed for real device runs (AGENTS §4)" >&2
    exit 4
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$PHONE_SSH" true 2>/dev/null; then
    echo "app_bench: devlock required but ssh $PHONE_SSH is unavailable (DEV_LOCK=off bypasses explicitly)" >&2
    exit 4
  fi
  verify_same_device
  ssh -o BatchMode=yes "$PHONE_SSH" "cd ~/csm/moe-exp && exec ./devlock -w $LOCK_WAIT bash -c 'touch .appbench-$LOCK_TOKEN.held .appbench-$LOCK_TOKEN.hb; while [ ! -e .appbench-$LOCK_TOKEN.release ]; do if [ -e .appbench-$LOCK_TOKEN.hb ] && [ -n \"\$(find .appbench-$LOCK_TOKEN.hb -mmin +3 2>/dev/null)\" ]; then echo stale > .appbench-$LOCK_TOKEN.stale; fi; sleep 2; done; rm -f .appbench-$LOCK_TOKEN.held .appbench-$LOCK_TOKEN.hb'" >/dev/null 2>&1 &
  LOCK_PID=$!
  local n=0
  while [ "$n" -lt "$LOCK_WAIT" ]; do
    if ! kill -0 "$LOCK_PID" 2>/dev/null; then
      echo "app_bench: devlock holder exited before acquisition (check ~/csm/moe-exp/.device.lock)" >&2
      exit 4
    fi
    if ssh -o BatchMode=yes -o ConnectTimeout=4 "$PHONE_SSH" "test -e ~/csm/moe-exp/.appbench-$LOCK_TOKEN.held" 2>/dev/null; then
      LOCK_HELD=1
      start_heartbeat
      echo "[app_bench] devlock acquired ($LOCK_TOKEN)" >&2
      return 0
    fi
    sleep 2; n=$((n+2))
  done
  echo "app_bench: timeout waiting for devlock ($PHONE_SSH busy)" >&2
  exit 4
}
release_devlock() {
  [ -n "$LOCK_PID" ] || return 0
  if [ "${LOCK_KEEP:-0}" = 1 ]; then
    echo "app_bench: devlock kept (app/DSP state unknown); see recovery above" >&2
    return 0
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=4 "$PHONE_SSH" \
    "cd ~/csm/moe-exp && touch .appbench-$LOCK_TOKEN.release; rm -f .appbench-$LOCK_TOKEN.held .appbench-$LOCK_TOKEN.hb" \
    >/dev/null 2>&1 || true
  if [ -n "$HB_PID" ]; then
    kill "$HB_PID" 2>/dev/null || true
    wait "$HB_PID" 2>/dev/null || true
    HB_PID=""
  fi
  for _ in $(seq 1 20); do kill -0 "$LOCK_PID" 2>/dev/null || break; sleep 0.2; done
  kill "$LOCK_PID" 2>/dev/null || true
  wait "$LOCK_PID" 2>/dev/null || true
  LOCK_PID=""
  LOCK_HELD=0
  return 0
}
app_state() {  # 0=gone, 1=running, 2=cannot query
  local out pid rc
  out=$("${ADB[@]}" shell "pidof $PKG; echo rc:\$?" 2>/dev/null | tr -d '\r')
  [ -n "$out" ] || return 2
  pid=$(printf '%s\n' "$out" | grep -v '^rc:' | head -1)
  rc=$(printf '%s\n' "$out" | sed -n 's/^rc://p' | tail -1)
  case "$rc" in
    0) return 1 ;;
    1) [ -z "$pid" ] && return 0 || return 1 ;;
    *) return 2 ;;
  esac
}
stop_app() {
  local n=0 st
  "${ADB[@]}" shell am force-stop "$PKG" >/dev/null 2>&1 || true
  while [ "$n" -lt 10 ]; do
    app_state; st=$?
    case "$st" in
      0) return 0 ;;
      2) echo "app_bench: WARN: cannot query app state over adb" >&2; return 2 ;;
    esac
    sleep 0.5; n=$((n+1))
  done
  echo "app_bench: WARN: app is still running after force-stop" >&2
  return 1
}
stop_owned_app() {
  # The lock may only be released after the app (the DSP user) is gone.
  # No bypass: if this runner started the app, it always force-stops it.
  if [ "$APP_STARTED" = 1 ]; then
    if ! stop_app; then
      LOCK_KEEP=1
      echo "app_bench: app state not confirmed stopped; KEEPING devlock. Recover with:" >&2
      echo "  adb -s $SERIAL shell am force-stop $PKG" >&2
      echo "  ssh $PHONE_SSH 'cd ~/csm/moe-exp && touch .appbench-$LOCK_TOKEN.release'" >&2
      echo "  (do NOT delete ~/csm/moe-exp/.device.lock or pkill other runners)" >&2
    fi
  fi
  return 0
}
cleanup() {
  [ "$CLEANED" = 1 ] && return 0
  CLEANED=1
  restore_runtime || true
  stop_owned_app || true
  release_devlock || true
  return 0
}
on_int()  { trap - INT TERM; exit 130; }
on_term() { trap - INT TERM; exit 143; }

RUNTIME_PUSHED=0
RUNTIME_HAD_ORIG=0
restore_runtime() {
  [ "$RUNTIME_PUSHED" = 1 ] || return 0
  [ "${KEEP_RUNTIME:-0}" = 1 ] && { echo "[app_bench] kept runtime.json (KEEP_RUNTIME=1)" >&2; return 0; }
  if [ "$RUNTIME_HAD_ORIG" = 1 ]; then
    if "${ADB[@]}" shell run-as "$PKG" tee "$DEV_DIR/runtime.json" < "$RUN_DIR/runtime.json.orig" >/dev/null; then
      echo "[app_bench] restored original runtime.json" >&2
    else
      echo "[app_bench] WARN: restore failed; recover with:" >&2
      echo "  ${ADB[*]} shell run-as $PKG tee $DEV_DIR/runtime.json < $RUN_DIR/runtime.json.orig" >&2
    fi
  else
    "${ADB[@]}" shell run-as "$PKG" rm -f "$DEV_DIR/runtime.json" >/dev/null 2>&1 || true
    echo "[app_bench] removed temporary runtime.json" >&2
  fi
}

trap on_int INT
trap on_term TERM
trap cleanup EXIT

acquire_devlock

if [ -n "${RUNTIME_JSON:-}" ]; then
  if "${ADB[@]}" exec-out run-as "$PKG" cat "$DEV_DIR/runtime.json" > "$RUN_DIR/runtime.json.orig" 2>/dev/null; then
    RUNTIME_HAD_ORIG=1
  elif "${ADB[@]}" shell run-as "$PKG" ls "$DEV_DIR/runtime.json" >/dev/null 2>&1; then
    echo "app_bench: cannot back up existing runtime.json; refusing to overwrite" >&2
    exit 3
  else
    rm -f "$RUN_DIR/runtime.json.orig"
  fi
  # Enable the restore trap *before* touching the remote file: a partial write
  # still leaves us able to restore/remove it on exit.
  RUNTIME_PUSHED=1
  if ! "${ADB[@]}" shell run-as "$PKG" tee "$DEV_DIR/runtime.json" < "$RUNTIME_JSON" >/dev/null; then
    echo "app_bench: failed to push $RUNTIME_JSON (trap will restore)" >&2
    exit 3
  fi
  echo "[app_bench] pushed $RUNTIME_JSON -> $DEV_DIR/runtime.json" >&2
fi

echo "[app_bench] tag=$TAG runs=$RUNS model=$MODEL_ID serial=$SERIAL prompt=${PROMPT:0:60}..." >&2
ok_runs=0
for i in $(seq 1 "$RUNS"); do
  status=""
  if [ "$LOCK_HELD" = 1 ] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "app_bench: lost devlock before run $i (holder gone)" >&2
    exit 5
  fi
  # 每条 run 冷启动 → 新会话；模型加载时间不计入 benchmark（app 从 request_start 起算）。
  "${ADB[@]}" shell am force-stop "$PKG" >/dev/null 2>&1 || true
  sleep 2
  APP_STARTED=1
  "${ADB[@]}" shell am start -n "$PKG/.MainActivity" >/dev/null 2>&1

  ok=0
  for _ in $(seq 1 30); do
    if dump_ui "$RUN_DIR/run$i.launch.xml" && bounds_center "$RUN_DIR/run$i.launch.xml" "$PKG:id/et_input" >/dev/null 2>&1; then
      ok=1; break
    fi
    sleep 1
  done
  if [ "$ok" -ne 1 ]; then
    echo "[app_bench] run$i: input box never appeared" >&2
    "${ADB[@]}" exec-out screencap -p > "$RUN_DIR/run$i.noinput.png" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t"%s"\n' "$TAG" "$i" noinput 0 0 0 0 0 0 NA NA 0 "$PROMPT" >> "$RUN_DIR/results.tsv"
    continue
  fi
  read -r IX IY < <(bounds_center "$RUN_DIR/run$i.launch.xml" "$PKG:id/et_input")

  typed_ok=0
  for attempt in 1 2 3; do
    "${ADB[@]}" shell input tap "$IX" "$IY"
    sleep 2
    while IFS= read -r chunk; do
      esc=${chunk// /%s}
      "${ADB[@]}" shell "input text '$(shell_quote "$esc")'"
      sleep 0.35
    done < <(chunks "$PROMPT")
    sleep 1
    if ! dump_ui "$RUN_DIR/run$i.typed.xml"; then
      echo "[app_bench] run$i: uiautomator dump failed (attempt $attempt)" >&2
      continue
    fi
    txt=$(node_text "$RUN_DIR/run$i.typed.xml" "$PKG:id/et_input")
    if [ "$txt" = "$PROMPT" ]; then typed_ok=1; break; fi
    echo "[app_bench] run$i: text mismatch (attempt $attempt): got '${txt:0:60}'" >&2
    # 清空：先移到末尾，再按当前长度+20 个退格
    "${ADB[@]}" shell input keyevent 123 >/dev/null 2>&1 || true
    n=$(( ${#txt} + 20 ))
    "${ADB[@]}" shell input keyevent $(printf '67 %.0s' $(seq 1 "$n")) >/dev/null 2>&1 || true
    sleep 0.5
  done
  if [ "$typed_ok" -ne 1 ]; then
    echo "[app_bench] run$i: could not type prompt exactly" >&2
    "${ADB[@]}" exec-out screencap -p > "$RUN_DIR/run$i.notyped.png" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t"%s"\n' "$TAG" "$i" notyped 0 0 0 0 0 0 NA NA 0 "$PROMPT" >> "$RUN_DIR/results.tsv"
    continue
  fi

  if ! bounds_out=$(bounds_center "$RUN_DIR/run$i.typed.xml" "$PKG:id/btn_send"); then
    echo "[app_bench] run$i: send button not found" >&2
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t"%s"\n' "$TAG" "$i" nosend 0 0 0 0 0 0 NA NA 0 "$PROMPT" >> "$RUN_DIR/results.tsv"
    continue
  fi
  read -r SX SY <<< "$bounds_out"
  before=$("${ADB[@]}" shell run-as "$PKG" stat -c %s "$BENCH_REMOTE" 2>/dev/null | tr -d '\r' || echo 0)
  [ -n "$before" ] || before=0
  "${ADB[@]}" shell input tap "$SX" "$SY"

  sleep 4
  if ! dump_ui "$RUN_DIR/run$i.sent.xml"; then
    echo "[app_bench] run$i: dump after send failed" >&2
  elif ! grep -q '终止生成' "$RUN_DIR/run$i.sent.xml"; then
    read -r SX SY < <(bounds_center "$RUN_DIR/run$i.sent.xml" "$PKG:id/btn_send" 2>/dev/null || echo "$SX $SY")
    "${ADB[@]}" shell input tap "$SX" "$SY"
    sleep 3
  fi

  rc=0
  size=$(wait_for_record "$before" "$WAIT_S") || rc=$?
  if [ "$rc" -eq 2 ]; then status=died
  elif [ "$rc" -eq 3 ]; then status=locklost
  elif [ "$rc" -ne 0 ]; then status=timeout
  fi
  if [ -n "$status" ]; then
    echo "[app_bench] run$i: $status waiting for benchmark.jsonl" >&2
    if [ "$status" = locklost ]; then
      stop_app || true
      echo "app_bench: lost devlock during run $i; aborting" >&2
      exit 5
    fi
    dump_ui "$RUN_DIR/run$i.timeout.xml" && classify_error "$RUN_DIR/run$i.timeout.xml" >&2 || true
    "${ADB[@]}" exec-out screencap -p > "$RUN_DIR/run$i.$status.png" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t"%s"\n' "$TAG" "$i" "$status" 0 0 0 0 0 0 NA NA 0 "$PROMPT" >> "$RUN_DIR/results.tsv"
    continue
  fi

  "${ADB[@]}" exec-out run-as "$PKG" cat "$BENCH_REMOTE" > "$RUN_DIR/run$i.benchmark.jsonl"
  if ! python3 "$HERE/parse_record.py" "$RUN_DIR/run$i.benchmark.jsonl" "$i" "$TAG" "$PROMPT" "$RUN_DIR" >> "$RUN_DIR/results.tsv"; then
    echo "[app_bench] run$i: benchmark record is invalid" >&2
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t"%s"\n' "$TAG" "$i" badrecord 0 0 0 0 0 0 NA NA 0 "$PROMPT" >> "$RUN_DIR/results.tsv"
    continue
  fi
  "${ADB[@]}" exec-out screencap -p > "$RUN_DIR/run$i.png" 2>/dev/null || true
  if [ "${CAPTURE_RUNTIME_LOG:-1}" = 1 ]; then
    "${ADB[@]}" exec-out run-as "$PKG" cat "$DEV_DIR/runtime.log" > "$RUN_DIR/run$i.runtime.log" 2>/dev/null || true
  fi
  ok_runs=$((ok_runs+1))
  tail -1 "$RUN_DIR/results.tsv" >&2
done

echo "[app_bench] done: $ok_runs/$RUNS ok, results=$RUN_DIR/results.tsv" >&2
[ "$ok_runs" -eq "$RUNS" ]
