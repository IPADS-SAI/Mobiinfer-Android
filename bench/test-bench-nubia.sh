#!/usr/bin/env bash
# test-bench-nubia.sh — host-only regression for the nubia wrapper defaults.
#
# No phone/DSP access: the wrapper runs the real bench.sh against a fake
# llama-cli that only prints the resolved EXPERTS/ARCH environment.  Covers:
#   * default EXPERTS = experts-q2k-idnow
#   * BENCH_CONFIG_ONLY=1 prints identity and never starts llama-cli
#   * manifest sha256 / experts_bytes identity fields (unknown when absent)
#   * explicit htp-tiled is allowed but warns
#   * missing EXPERTS exits 2, lists all candidates, and starts nothing
#   * ARCH= (empty) still leaves GGML_HEXAGON_ARCH unset
set -u

here=$(cd "$(dirname "$0")" && pwd)
wrapper="$here/bench-nubia.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/test-bench-nubia.XXXXXX") || exit 1
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
check_match() {
    if [[ "$2" =~ $3 ]]; then ok "$1"; else bad "$1" "matches: $3" "$2"; fi
}

root="$tmp/root"
rel="$tmp/rel"
model_dir="$root/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon"
q2k="$model_dir/experts-q2k-idnow"
q1="$model_dir/experts-q1-30tps"
q3="$model_dir/experts-htp-tiled"
model="$model_dir/minicpm47-les-hexagon.gguf"
draft="$model_dir/dspark-MiniCPM-V-4_7-q8_0.gguf"
mkdir -p "$q2k" "$q1" "$q3" "$rel/bin"

cat > "$rel/bin/llama-cli" <<'MOCK'
#!/usr/bin/env bash
printf 'EXPERTS=%s\n' "$EXPERTS"
printf 'GGML_HEXAGON_ARCH=%s\n' "${GGML_HEXAGON_ARCH-unset}"
printf 'call\n' >> "${MOCK_CALL_LOG:-/dev/null}"
MOCK
chmod +x "$rel/bin/llama-cli"
: > "$tmp/calls.log"

out="$tmp/out"
err="$tmp/err"
rc=0
run_wrapper() { # run_wrapper <name> [ENV=VAL ...]
    local name=$1
    shift
    out="$tmp/$name.out"
    err="$tmp/$name.err"
    env -u ROOT -u REL -u ARCH -u HOSTBUF -u MODEL -u EXPERTS -u DRAFT \
        -u BENCH_CONFIG_ONLY -u GGML_HEXAGON_ARCH \
        ROOT="$root" REL="$rel" MOCK_CALL_LOG="$tmp/calls.log" "$@" \
        bash "$wrapper" >"$out" 2>"$err"
    rc=$?
}
call_count() { wc -l < "$tmp/calls.log" | tr -d ' '; }
clear_calls() { : > "$tmp/calls.log"; }

# ── 0. syntax ────────────────────────────────────────────────────────────
bash -n "$wrapper"
check_eq "bash -n bench/bench-nubia.sh" 0 "$?"
bash -n "$here/test-bench-nubia.sh"
check_eq "bash -n bench/test-bench-nubia.sh" 0 "$?"

# ── 1. default config-only: q2k-idnow, identity, no llama-cli ────────────
clear_calls
run_wrapper cfg_default BENCH_CONFIG_ONLY=1
check_eq "config-only default exit" 0 "$rc"
check_contains "config-only default full config line" "$(cat "$err")" \
    "nubia-config: REL=$rel ARCH=79 HOSTBUF=0 MODEL=$model EXPERTS=$q2k DRAFT=$draft"
check_contains "identity basename" "$(cat "$err")" "experts_basename=experts-q2k-idnow"
check_match "identity bytes numeric" "$(cat "$err")" 'experts_bytes=[0-9]+'
check_contains "identity manifest unknown without manifest" "$(cat "$err")" "manifest_sha256=unknown"
check_eq "config-only default does not call llama-cli" 0 "$(call_count)"
check_eq "config-only default stdout empty" "" "$(cat "$out")"

# ── 1b. default package missing (no EXPERTS): still exit 2, no fallback ──
empty_root="$tmp/empty-root"
mkdir -p "$empty_root"
clear_calls
run_wrapper default_missing ROOT="$empty_root"
check_eq "default EXPERTS missing exit" 2 "$rc"
check_contains "default EXPERTS missing names q2k" "$(cat "$err")" \
    "$empty_root/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q2k-idnow"
check_eq "default EXPERTS missing starts no llama-cli" 0 "$(call_count)"

# ── 2. manifest sha256 is reported when present ──────────────────────────
printf 'q2k manifest fixture\n' > "$q2k/manifest.json"
manifest_sha=$(sha256sum "$q2k/manifest.json" | awk '{print $1}')
clear_calls
run_wrapper cfg_manifest BENCH_CONFIG_ONLY=1
check_eq "config-only manifest exit" 0 "$rc"
check_contains "identity manifest sha256" "$(cat "$err")" "manifest_sha256=$manifest_sha"
check_eq "config-only manifest does not call llama-cli" 0 "$(call_count)"

# ── 3. explicit htp-tiled: allowed + WARN, real bench path ───────────────
clear_calls
run_wrapper explicit_q3 EXPERTS="$q3"
check_eq "explicit htp-tiled exit" 0 "$rc"
check_contains "explicit htp-tiled reaches llama-cli" "$(cat "$out")" "EXPERTS=$q3"
check_contains "explicit htp-tiled warns" "$(cat "$err")" \
    "WARN: Q3_K htp-tiled is not the 28.1 baseline"
check_eq "explicit htp-tiled starts llama-cli once" 1 "$(call_count)"

# ── 4. missing EXPERTS: exit 2, all candidates, no llama-cli ────────────
clear_calls
missing="$model_dir/experts-does-not-exist"
run_wrapper missing EXPERTS="$missing"
check_eq "missing EXPERTS exit" 2 "$rc"
check_contains "missing EXPERTS error has path" "$(cat "$err")" \
    "ERROR: nubia EXPERTS directory not found: $missing"
check_contains "missing EXPERTS lists q2k candidate" "$(cat "$err")" "$q2k"
check_contains "missing EXPERTS lists q1 candidate" "$(cat "$err")" "$q1"
check_contains "missing EXPERTS lists htp-tiled candidate" "$(cat "$err")" "$q3"
check_eq "missing EXPERTS starts no llama-cli" 0 "$(call_count)"
check_eq "missing EXPERTS stdout empty" "" "$(cat "$out")"

# ── 5. ARCH= (empty) still means "let the backend auto-detect" ───────────
clear_calls
run_wrapper arch_empty ARCH=
check_eq "ARCH= full run exit" 0 "$rc"
check_contains "ARCH= full run uses q2k default" "$(cat "$out")" "EXPERTS=$q2k"
check_contains "ARCH= leaves GGML_HEXAGON_ARCH unset" "$(cat "$out")" \
    "GGML_HEXAGON_ARCH=unset"
check_eq "ARCH= starts llama-cli once" 1 "$(call_count)"

# ── 6. default full run passes q2k to the bench path ────────────────────
clear_calls
run_wrapper default_full
check_eq "default full run exit" 0 "$rc"
check_contains "default full run uses q2k" "$(cat "$out")" "EXPERTS=$q2k"
check_contains "default full run prints config line" "$(cat "$err")" "nubia-config:"
check_contains "default full run prints identity line" "$(cat "$err")" "nubia-identity:"
check_eq "default full run starts llama-cli once" 1 "$(call_count)"

printf '\n%d checks, %d failure(s)\n' "$checks" "$failures"
[ "$failures" -eq 0 ]
