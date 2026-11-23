#!/usr/bin/env bash
# test-sweep.sh — host-only regression for sweep.sh per-run results (T-0099).
#
# 不访问手机/DSP：所有 bench 都由临时 ROOT 里的 mock bench.sh 完成；PATH 前面
# 还放了一个会报错的 ssh stub，真被调用就会失败。
# 覆盖：per-run 文件 + 注释头、tag 不混、legacy 聚合格式、重复 run_id 追加、
#       非法 SWEEP_RUN_ID exit 3、MEMMARK 列数、失败行也落盘、默认 run_id 唯一、
#       bash -n。
set -u

here=$(cd "$(dirname "$0")" && pwd)
sweep="$here/sweep.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/test-sweep.XXXXXX") || exit 1
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM

checks=0
failures=0
ok() {
    checks=$((checks + 1))
    printf 'ok   %s\n' "$1"
}
bad() {
    checks=$((checks + 1))
    failures=$((failures + 1))
    printf 'FAIL %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3" >&2
}
check_eq() {
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "$2" "$3"; fi
}
check_contains() {
    case "$2" in
        *"$3"*) ok "$1" ;;
        *) bad "$1" "contains: $3" "$2" ;;
    esac
}

mkdir -p "$tmp/bin" "$tmp/runs"

# mock bench.sh：固定计时行；MOCK_RC 可模拟失败（默认 0）。
cat > "$tmp/bench.sh" <<'MOCK'
#!/usr/bin/env bash
sleep "${MOCK_SLEEP:-0}"
printf 'llama mock start\n'
printf 'Prompt: 50.0 t/s\n'
printf 'Generation: 20.0 t/s\n'
printf 'llama mock end\n'
exit "${MOCK_RC:-0}"
MOCK
chmod +x "$tmp/bench.sh"

# 任何 ssh 调用都视为测试失败。
cat > "$tmp/bin/ssh" <<EOF
#!/usr/bin/env bash
printf 'unexpected ssh: %s\n' "\$*" >> "$tmp/ssh.log"
exit 99
EOF
chmod +x "$tmp/bin/ssh"

# 极简环境（没有 coreutils stdbuf）时给个透传 stub；有真 stdbuf 就用真的。
if ! command -v stdbuf >/dev/null 2>&1; then
    cat > "$tmp/bin/stdbuf" <<'STUB'
#!/usr/bin/env bash
shift 2   # -oL -eL
exec "$@"
STUB
    chmod +x "$tmp/bin/stdbuf"
fi

run_seq=0
run_sweep() {  # run_sweep <tag> <run_id> [ENV=VAL ...]
    tag=$1
    rid=$2
    shift 2
    run_seq=$((run_seq + 1))
    out_log="$tmp/run-$run_seq.log"
    env ROOT="$tmp" TAG="$tag" SWEEP_RUN_ID="$rid" PATH="$tmp/bin:$PATH" "$@" \
        bash "$sweep" >"$out_log" 2>"$out_log.err"
    rc=$?
}
run_sweep_noid() {  # 不设置 SWEEP_RUN_ID，验证默认值每次不同
    tag=$1
    run_seq=$((run_seq + 1))
    out_log="$tmp/run-$run_seq.log"
    env -u SWEEP_RUN_ID ROOT="$tmp" TAG="$tag" PATH="$tmp/bin:$PATH" \
        bash "$sweep" >"$out_log" 2>"$out_log.err"
    rc=$?
}

legacy="$tmp/runs/results.tsv"
data_cols() { awk -F '\t' '!/^#/ {print NF}' "$1"; }
data_tags() { awk -F '\t' '!/^#/ {printf "%s%s", (n++ ? " " : ""), $1}' "$1"; }
header_count() { grep -c '^# sweep-results v1 ' "$1" 2>/dev/null || true; }
line_count() { wc -l < "$1" | tr -d ' '; }

# ── 0. syntax ────────────────────────────────────────────────────────────
bash -n "$sweep"; check_eq "bash -n bench/sweep.sh" 0 "$?"
bash -n "$here/test-sweep.sh"; check_eq "bash -n bench/test-sweep.sh" 0 "$?"

# ── 1. 两个 run_id = 两个独立文件，tag 不混 ──────────────────────────────
run_sweep t1 r1
check_eq "r1 sweep exit" 0 "$rc"
check_contains "r1 stdout prints run_id" "$(cat "$out_log")" "run_id=r1"
f1="$tmp/runs/results-r1.tsv"
check_eq "results-r1.tsv exists" yes "$([ -f "$f1" ] && echo yes || echo no)"
check_eq "results-r1.tsv lines (header+1)" 2 "$(line_count "$f1")"
check_eq "results-r1.tsv header" 1 "$(header_count "$f1")"
check_eq "results-r1.tsv header has run_id" 1 \
    "$(grep -c $'^# sweep-results v1 run_id=r1 \t' "$f1" || true)"
check_eq "results-r1.tsv header has named columns" 1 \
    "$(grep -c $'^# sweep-results v1 run_id=r1 \ttag\tn\twall_ms\trc\tgen_tps\tprompt_tps\tstatus$' "$f1" || true)"
check_eq "results-r1.tsv tag" t1 "$(data_tags "$f1")"
check_eq "results-r1.tsv columns (legacy format)" 7 "$(data_cols "$f1")"

run_sweep t2 r2
check_eq "r2 sweep exit" 0 "$rc"
check_contains "r2 stdout prints run_id" "$(cat "$out_log")" "run_id=r2"
f2="$tmp/runs/results-r2.tsv"
check_eq "results-r2.tsv lines (header+1)" 2 "$(line_count "$f2")"
check_eq "results-r2.tsv header" 1 "$(header_count "$f2")"
check_eq "results-r2.tsv tag" t2 "$(data_tags "$f2")"
check_eq "results-r2.tsv has no t1" 0 "$(grep -c '^t1	' "$f2" || true)"
check_eq "results-r1.tsv has no t2" 0 "$(grep -c '^t2	' "$f1" || true)"

# ── 2. legacy 聚合：2 行、无注释头、列数不变 ─────────────────────────────
check_eq "legacy results.tsv lines" 2 "$(line_count "$legacy")"
check_eq "legacy has no comment header" 0 "$(grep -c '^#' "$legacy" || true)"
check_eq "legacy tags in call order" "t1 t2" "$(data_tags "$legacy")"
check_eq "legacy columns" "7 7" "$(data_cols "$legacy" | tr '\n' ' ' | sed 's/ $//')"

# ── 3. 重复 run_id：追加而不是覆盖，头只写一次 ───────────────────────────
run_sweep t1 r1
check_eq "duplicate r1 sweep exit" 0 "$rc"
check_eq "duplicate r1 keeps one header" 1 "$(header_count "$f1")"
check_eq "duplicate r1 appends row" 3 "$(line_count "$f1")"
check_eq "duplicate r1 data rows" "t1 t1" "$(data_tags "$f1")"
check_eq "duplicate r1 legacy row appended" 3 "$(line_count "$legacy")"

# ── 4. 非法 SWEEP_RUN_ID -> exit 3，不写任何文件 ─────────────────────────
run_sweep bad 'a/b'
check_eq "invalid run_id exit 3" 3 "$rc"
check_eq "invalid run_id writes no per-run file" 0 "$(compgen -G "$tmp/runs/results-a*" | wc -l | tr -d ' ')"
check_eq "invalid run_id leaves legacy alone" 3 "$(line_count "$legacy")"
check_eq "invalid run_id prints nothing to stdout" "" "$(cat "$out_log")"

# ── 5. MEMMARK=1 仍是 26 列（legacy 列序），头也只有一份 ─────────────────
run_sweep t3 r3 MEMMARK=1
check_eq "memmark sweep exit" 0 "$rc"
f3="$tmp/runs/results-r3.tsv"
check_eq "memmark per-run lines" 2 "$(line_count "$f3")"
check_eq "memmark header" 1 "$(header_count "$f3")"
check_eq "memmark columns" 26 "$(data_cols "$f3")"
check_eq "memmark legacy columns" 26 "$(data_cols "$legacy" | tail -1)"
check_eq "memmark legacy rows" 4 "$(line_count "$legacy")"

# ── 6. 失败行也写（rc=137 -> sweep exit 2, status=invalid） ──────────────
run_sweep t4 r4 MOCK_RC=137
check_eq "failed run maps rc 137 -> 2" 2 "$rc"
f4="$tmp/runs/results-r4.tsv"
check_eq "failed run still writes per-run row" 2 "$(line_count "$f4")"
check_eq "failed run rc column" 137 "$(awk -F '\t' '!/^#/ {print $4}' "$f4")"
check_eq "failed run status column" invalid "$(awk -F '\t' '!/^#/ {print $7}' "$f4")"
check_eq "failed run appended to legacy" 5 "$(line_count "$legacy")"

# ── 7. SWEEP_RUN_ID 未设：每次调用 run_id 不同 ───────────────────────────
run_sweep_noid default_a
check_eq "default run a exit" 0 "$rc"
rid_a=$(sed -n 's/^run_id=//p' "$out_log")
run_sweep_noid default_b
check_eq "default run b exit" 0 "$rc"
rid_b=$(sed -n 's/^run_id=//p' "$out_log")
if [ -n "$rid_a" ] && [ -n "$rid_b" ] && [ "$rid_a" != "$rid_b" ]; then
    ok "default run_id differs per call ($rid_a != $rid_b)"
else
    bad "default run_id differs per call" "two different non-empty ids" "a=$rid_a b=$rid_b"
fi
check_eq "default run files both exist" yes \
    "$([ -f "$tmp/runs/results-$rid_a.tsv" ] && [ -f "$tmp/runs/results-$rid_b.tsv" ] && echo yes || echo no)"
check_eq "default runs appended to legacy" 7 "$(line_count "$legacy")"

# ── 8. 并发 append：legacy 不交错、每个 per-run 头只有一份 ──────────────
conc_before=$(line_count "$legacy")
for i in 1 2 3 4; do
    env ROOT="$tmp" TAG="c$i" SWEEP_RUN_ID="c$i" MOCK_SLEEP=0.05 \
        PATH="$tmp/bin:$PATH" bash "$sweep" >"$tmp/conc-$i.log" 2>&1 &
done
wait
check_eq "concurrent runs all appended" $((conc_before + 4)) "$(line_count "$legacy")"
check_eq "concurrent legacy rows keep 7 columns" 0 \
    "$(tail -4 "$legacy" | awk -F '\t' 'NF!=7' | wc -l | tr -d ' ')"
check_eq "concurrent legacy tags are distinct" 4 \
    "$(tail -4 "$legacy" | awk -F '\t' '{print $1}' | sort -u | wc -l | tr -d ' ')"
check_eq "concurrent per-run files each have one header" 4 \
    "$(for i in 1 2 3 4; do header_count "$tmp/runs/results-c$i.tsv"; done | grep -c '^1$' || true)"
check_eq "concurrent per-run files each have one row" 4 \
    "$(for i in 1 2 3 4; do line_count "$tmp/runs/results-c$i.tsv"; done | grep -c '^2$' || true)"

# ── 8.5 IOMARK：可选 3 列 /proc/<pid>/io 差值，status 永远最后 ───────────
# mock bench 睡 0.5s，保证启动后至少一次 end 快照（仍允许 NA，但不允许 0 冒充缺失）。
io_ok() { awk -F '\t' -v n="$2" '!/^#/ {v=$n; if (v !~ /^[0-9]+$/ && v != "NA") bad=1} END {print bad+0}' "$1"; }
run_sweep t5 r5 IOMARK=1 MOCK_SLEEP=0.5
check_eq "iomark-only sweep exit" 0 "$rc"
f5="$tmp/runs/results-r5.tsv"
check_eq "iomark-only header has io_rchar" 1 "$(grep -c 'io_rchar' "$f5" || true)"
check_eq "iomark-only header has io_read_bytes" 1 "$(grep -c 'io_read_bytes' "$f5" || true)"
check_eq "iomark-only header has io_syscr" 1 "$(grep -c 'io_syscr' "$f5" || true)"
check_eq "iomark-only columns (base+io+status)" 10 "$(data_cols "$f5")"
check_eq "iomark-only status stays last" ok "$(awk -F '\t' '!/^#/ {print $10}' "$f5")"
for col in 7 8 9; do
    check_eq "iomark-only column $col numeric or NA" 0 "$(io_ok "$f5" "$col")"
done
check_eq "iomark-only values are numeric" 3 \
    "$(awk -F '\t' '!/^#/ {n=0; for (i=7;i<=9;i++) if ($i ~ /^[0-9]+$/) n++; print n}' "$f5")"
check_eq "iomark-only legacy columns" 10 "$(data_cols "$legacy" | tail -1)"

run_sweep t6 r6 IOMARK=1 MEMMARK=1 MOCK_SLEEP=0.5
check_eq "iomark+memmark sweep exit" 0 "$rc"
f6="$tmp/runs/results-r6.tsv"
check_eq "iomark+memmark header has all io columns" 1 \
    "$(grep -c $'io_rchar\tio_read_bytes\tio_syscr' "$f6" || true)"
check_eq "iomark+memmark columns (base+mem+health+io+status)" 29 "$(data_cols "$f6")"
check_eq "iomark+memmark status stays last" ok "$(awk -F '\t' '!/^#/ {print $29}' "$f6")"
for col in 26 27 28; do
    check_eq "iomark+memmark column $col numeric or NA" 0 "$(io_ok "$f6" "$col")"
done
check_eq "iomark+memmark legacy columns" 29 "$(data_cols "$legacy" | tail -1)"

# IOMARK=0 仍然只有 7 列（不能因为新开关泄漏列）
run_sweep t7 r7 IOMARK=0 MOCK_SLEEP=0
f7="$tmp/runs/results-r7.tsv"
check_eq "iomark=0 columns unchanged" 7 "$(data_cols "$f7")"
check_eq "iomark=0 header has no io columns" 0 "$(grep -c 'io_rchar' "$f7" || true)"

# gate 的 speed 采样固定 MEMMARK=1；GATE_IOMARK=0 时 IOMARK=0，必须回到 26 列。
run_sweep t8 r8 MEMMARK=1 IOMARK=0 MOCK_SLEEP=0
f8="$tmp/runs/results-r8.tsv"
check_eq "memmark+iomark=0 columns (old width)" 26 "$(data_cols "$f8")"
check_eq "memmark+iomark=0 header has no io columns" 0 "$(grep -c 'io_rchar' "$f8" || true)"
check_eq "memmark+iomark=0 status stays last" ok "$(awk -F '\t' '!/^#/ {print $26}' "$f8")"

# ── 8.6 IOSTAT：可选 runs/<tag>-iostat.json 侧车，不改 TSV ────────────────
# 全部用 fake /proc、/sys 根；模型路径只是为了让 df/readlink 找到块设备。
iostat_root="$tmp/iostat"
iostat_models="$tmp/iostat-models"
mkdir -p "$iostat_root/proc" "$iostat_root/sys/block" "$iostat_models"
: > "$iostat_models/model.gguf"
iostat_df=$(df -P "$iostat_models/model.gguf" 2>/dev/null | awk 'NR == 2 { print $1 }')
iostat_base=$(basename "$(readlink -f "$iostat_df" 2>/dev/null || printf '%s' "$iostat_df")")
iostat_dev=$iostat_base
case "$iostat_base" in
    dm-*|loop*|ram*|zram*) ;;
    *p[0-9]*) iostat_dev=${iostat_base%p[0-9]*} ;;
    *[0-9]) iostat_dev=${iostat_base%%[0-9]*} ;;
esac
mkdir -p "$iostat_root/sys/block/$iostat_dev"
printf '100 0 2000 10 50 0 1000 20 1 30 40\n' > "$iostat_root/sys/block/$iostat_dev/stat"
printf '8 0 %s 100 0 2000 10 50 0 1000 20 1 30 40\n' "$iostat_dev" > "$iostat_root/proc/diskstats"
cat > "$iostat_root/proc/vmstat" <<'EOF'
pgscan_direct 1000
pgsteal_direct 900
allocstall 3
nr_dirty 10
nr_writeback 2
nr_unevictable 5
pgscan_kswapd 999999
EOF

jget() { # jget <json> <dotted.key>
    python3 - "$1" "$2" <<'PY'
import json, sys
try:
    cur = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("ERR")
    raise SystemExit(0)
for key in sys.argv[2].split("."):
    if not isinstance(cur, dict):
        cur = None
        break
    cur = cur.get(key)
if cur is None:
    print("null")
elif isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PY
}
jlen() { # jlen <json> <dotted.key>
    python3 - "$1" "$2" <<'PY'
import json, sys
try:
    cur = json.load(open(sys.argv[1], encoding="utf-8"))
    for key in sys.argv[2].split("."):
        cur = cur[key]
    print(len(cur))
except Exception:
    print("ERR")
PY
}

run_sweep io1 r-io1 IOSTAT=1 IOSTAT_PROC_ROOT="$iostat_root/proc" \
    IOSTAT_SYS_ROOT="$iostat_root/sys" MODEL="$iostat_models/model.gguf" \
    EXPERTS="$iostat_models" DRAFT="$iostat_models/model.gguf" MOCK_SLEEP=0
check_eq "iostat normal run exit" 0 "$rc"
fio="$tmp/runs/io1-iostat.json"
check_eq "iostat sidecar exists" yes "$([ -f "$fio" ] && echo yes || echo no)"
check_eq "iostat version" iostat-v1 "$(jget "$fio" version)"
check_eq "iostat tag" io1 "$(jget "$fio" tag)"
check_eq "iostat vmstat pgscan_direct" 1000 "$(jget "$fio" vmstat.pgscan_direct)"
check_eq "iostat vmstat allocstall" 3 "$(jget "$fio" vmstat.allocstall)"
check_eq "iostat disk dev" "$iostat_dev" "$(jget "$fio" disk.dev)"
check_eq "iostat disk read_ios" 100 "$(jget "$fio" disk.read_ios)"
check_eq "iostat disk read_ticks" 10 "$(jget "$fio" disk.read_ticks)"
check_eq "iostat disk time_in_queue" 40 "$(jget "$fio" disk.time_in_queue)"
check_eq "iostat delta read_ios" 0 "$(jget "$fio" delta.disk.read_ios)"
check_eq "iostat delta in_flight is gauge/null" null "$(jget "$fio" delta.disk.in_flight)"
check_eq "iostat normal has no missing" 0 "$(jlen "$fio" missing)"
check_eq "iostat run keeps 7 TSV columns" 7 "$(data_cols "$tmp/runs/results-r-io1.tsv")"
check_eq "iostat run adds no TSV column name" 0 \
    "$(grep -c 'iostat' "$tmp/runs/results-r-io1.tsv" || true)"

# 部分缺失：vmstat 少 4 个字段、sys stat 少 time_in_queue；必须 null 且 run 仍成功。
iostat_part="$tmp/iostat-part"
mkdir -p "$iostat_part/proc" "$iostat_part/sys/block/$iostat_dev"
printf 'pgscan_direct 5\nnr_dirty 7\n' > "$iostat_part/proc/vmstat"
printf '5 0 10 1 2 0 3 4 0 5\n' > "$iostat_part/sys/block/$iostat_dev/stat"
printf '8 0 %s 5 0 10 1 2 0 3 4 0 5\n' "$iostat_dev" > "$iostat_part/proc/diskstats"
run_sweep io2 r-io2 IOSTAT=1 IOSTAT_PROC_ROOT="$iostat_part/proc" \
    IOSTAT_SYS_ROOT="$iostat_part/sys" MODEL="$iostat_models/model.gguf" \
    EXPERTS="$iostat_models" DRAFT="$iostat_models/model.gguf" MOCK_SLEEP=0
check_eq "iostat partial run exit" 0 "$rc"
fio2="$tmp/runs/io2-iostat.json"
check_eq "iostat partial pgscan_direct still read" 5 "$(jget "$fio2" vmstat.pgscan_direct)"
check_eq "iostat partial allocstall null" null "$(jget "$fio2" vmstat.allocstall)"
check_eq "iostat partial time_in_queue null" null "$(jget "$fio2" disk.time_in_queue)"
check_eq "iostat partial missing allocstall" True \
    "$(python3 -c 'import json,sys; print("vmstat.allocstall" in json.load(open(sys.argv[1]))["missing"])' "$fio2")"
check_eq "iostat partial missing time_in_queue" True \
    "$(python3 -c 'import json,sys; print("disk.time_in_queue" in json.load(open(sys.argv[1]))["missing"])' "$fio2")"
check_eq "iostat partial still writes TSV row" ok "$(awk -F '\t' '!/^#/ {print $7}' "$tmp/runs/results-r-io2.tsv")"

# 找不到块设备：所有 disk 字段 null + missing，但 run/TSV 不受影响。
run_sweep io3 r-io3 IOSTAT=1 IOSTAT_PROC_ROOT="$iostat_root/proc" \
    IOSTAT_SYS_ROOT="$tmp/iostat-empty-sys" MODEL="$tmp/nope.gguf" \
    EXPERTS="$tmp/nope-experts" DRAFT="$tmp/nope-draft.gguf" MOCK_SLEEP=0
check_eq "iostat no-device run exit" 0 "$rc"
fio3="$tmp/runs/io3-iostat.json"
check_eq "iostat no-device dev null" null "$(jget "$fio3" disk.dev)"
check_eq "iostat no-device read_ios null" null "$(jget "$fio3" disk.read_ios)"
check_eq "iostat no-device missing disk.dev" True \
    "$(python3 -c 'import json,sys; print("disk.dev" in json.load(open(sys.argv[1]))["missing"])' "$fio3")"
check_eq "iostat no-device row written" ok "$(awk -F '\t' '!/^#/ {print $7}' "$tmp/runs/results-r-io3.tsv")"

# IOSTAT=0：不写侧车，TSV 表头/列数与旧格式逐字一致。
run_sweep io0 r-io0 IOSTAT=0 MODEL="$iostat_models/model.gguf" MOCK_SLEEP=0
check_eq "iostat=0 run exit" 0 "$rc"
check_eq "iostat=0 writes no sidecar" no "$([ -f "$tmp/runs/io0-iostat.json" ] && echo yes || echo no)"
check_eq "iostat=0 keeps legacy header" \
    $'# sweep-results v1 run_id=r-io0 \ttag\tn\twall_ms\trc\tgen_tps\tprompt_tps\tstatus' \
    "$(head -n1 "$tmp/runs/results-r-io0.tsv")"
check_eq "iostat=0 keeps 7 columns" 7 "$(data_cols "$tmp/runs/results-r-io0.tsv")"

# ── 9. 全程没有访问手机 ──────────────────────────────────────────────────
check_eq "no ssh invocation" no "$([ -e "$tmp/ssh.log" ] && echo yes || echo no)"

printf '\n%s: %d checks, %d failure(s)\n' \
    "$([ "$failures" -eq 0 ] && echo PASS || echo FAIL)" "$checks" "$failures"
[ "$failures" -eq 0 ]
