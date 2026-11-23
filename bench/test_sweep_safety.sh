#!/usr/bin/env bash
# Offline checks for sweep.sh's memory gate. No phone or model is needed.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0

cat > "$tmp/bench.sh" <<'EOF'
#!/usr/bin/env bash
printf 'WINDOW=%s DEPTH=%s\n' "${WINDOW:-unset}" "${DEPTH:-unset}" > "$ROOT/bench-called-${TAG:-unset}"
echo "fake bench: WINDOW=${WINDOW:-unset} DEPTH=${DEPTH:-unset} LAYER_WINDOWS=${LAYER_WINDOWS:-unset}"
EOF
chmod +x "$tmp/bench.sh"

check_rc() {
    local label=$1 want=$2 got=$3
    if [ "$got" -ne "$want" ]; then
        echo "FAIL $label: rc=$got want=$want" >&2
        fail=1
    fi
}

check_contains() {
    local label=$1 needle=$2 text=$3
    case "$text" in
        *"$needle"*) ;;
        *) echo "FAIL $label: output missing '$needle'" >&2; fail=1 ;;
    esac
}

check_bench_not_called() {
    local label=$1 tag=$2
    if [ -e "$tmp/bench-called-$tag" ]; then
        echo "FAIL $label: bench.sh was called after the gate rejected the run" >&2
        fail=1
    fi
}

check_file_contains() {
    local label=$1 file=$2 needle=$3
    if ! grep -Fq -- "$needle" "$file" 2>/dev/null; then
        echo "FAIL $label: file missing '$needle'" >&2
        fail=1
    fi
}

# The memory gate resolves bench.sh's defaults itself, so those resolved values
# must reach the child even when the caller did not set WINDOW/DEPTH.
out=$(unset WINDOW DEPTH; env ROOT="$tmp" TAG=defaults N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc defaults 0 "$rc"
check_file_contains defaults "$tmp/bench-called-defaults" "WINDOW=80 DEPTH=8"

out=$(unset WINDOW DEPTH; env ROOT="$tmp" MEMMARK=1 TAG=defaults-memmark N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc defaults-memmark 0 "$rc"
check_file_contains defaults-memmark "$tmp/bench-called-defaults-memmark" "WINDOW=80 DEPTH=8"

out=$(env ROOT="$tmp" WINDOW=96 TAG=window96 N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc window96 3 "$rc"
check_contains window96 "104" "$out"
check_bench_not_called window96 window96

out=$(env ROOT="$tmp" WINDOW=120 DEPTH=08 TAG=leadingzero N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc leading-zero-depth 3 "$rc"
check_contains leading-zero-depth "128" "$out"
case "$out" in
    *"value too great for base"*) echo "FAIL leading-zero-depth: octal arithmetic error" >&2; fail=1 ;;
esac
check_bench_not_called leading-zero-depth leadingzero

out=$(env ROOT="$tmp" WINDOW=08 DEPTH=8 TAG=leadingwindow N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc leading-zero-window 0 "$rc"
check_file_contains leading-zero-window "$tmp/bench-called-leadingwindow" "WINDOW=8 DEPTH=8"

out=$(env ROOT="$tmp" WINDOW=auto TAG=auto N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc auto-unresolved 3 "$rc"
check_contains auto-unresolved "无法在脚本内确认" "$out"
case "$out" in
    *"unbound variable"*) echo "FAIL auto-unresolved: shell unbound-variable crash" >&2; fail=1 ;;
esac
check_bench_not_called auto-unresolved auto

out=$(env ROOT="$tmp" WINDOW=auto BANK=256 MOE_LAYERS=40 TAG=auto40 N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc auto-resolved 0 "$rc"
check_contains auto-resolved "WINDOW=auto → 8" "$out"

out=$(env ROOT="$tmp" WINDOW=auto BANK=256 MOE_LAYERS=08 TAG=auto08 N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc auto-leading-zero 0 "$rc"
check_contains auto-leading-zero "WINDOW=auto → 32" "$out"

out=$(env ROOT="$tmp" WINDOW=80 DEPTH=8 LAYER_WINDOWS=0:89 TAG=layerover N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc layer-override 3 "$rc"
check_contains layer-override "97" "$out"
check_bench_not_called layer-override layerover

out=$(env ROOT="$tmp" WINDOW=80 LAYER_WINDOWS=bogus TAG=badlayer N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc malformed-layer 3 "$rc"
check_contains malformed-layer "格式非法" "$out"
check_bench_not_called malformed-layer badlayer

out=$(env ROOT="$tmp" WINDOW=80 LAYER_WINDOWS=0:089 TAG=leadinglayer N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc leading-zero-layer 3 "$rc"
check_contains leading-zero-layer "97" "$out"
case "$out" in
    *"value too great for base"*) echo "FAIL leading-zero-layer: octal arithmetic error" >&2; fail=1 ;;
esac
check_bench_not_called leading-zero-layer leadinglayer

out=$(env ROOT="$tmp" WINDOW=96 TAG=allowed N=1 ALLOW_DANGEROUS=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc allow-dangerous 0 "$rc"

out=$(env ROOT="$tmp" WINDOW=auto DEPTH=8 TAG=autoallow N=1 ALLOW_DANGEROUS=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc auto-allow 0 "$rc"
check_contains auto-allow "放行 WINDOW=auto" "$out"
check_file_contains auto-allow "$tmp/bench-called-autoallow" "WINDOW=auto DEPTH=8"

# read(1) only parses the first line, so a dangerous second line must not slip
# through and reach bench.sh inside the raw LAYER_WINDOWS value.
out=$(env ROOT="$tmp" WINDOW=80 DEPTH=8 LAYER_WINDOWS=$'0:80\n,1:120' TAG=multiline N=1 bash "$HERE/sweep.sh" 2>&1)
rc=$?
check_rc multiline-layer 3 "$rc"
check_contains multiline-layer "不允许换行" "$out"
check_bench_not_called multiline-layer multiline

[ "$fail" -eq 0 ] && echo "sweep safety selftest ok"
exit "$fail"
