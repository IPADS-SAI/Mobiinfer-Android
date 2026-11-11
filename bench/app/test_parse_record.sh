#!/usr/bin/env bash
# Offline checks for parse_record.py (no device needed).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0

run() { python3 "$HERE/parse_record.py" "$1" 1 t p "$tmp" >/dev/null 2>&1; echo $?; }
check() { [ "$2" = "$3" ] || { echo "FAIL $1: got $2 want $3" >&2; fail=1; }; }

cat > "$tmp/ok.jsonl" <<'EOF'
{"wall_ms":1000,"ttft_ms":100,"response":"x","timings":{"prompt_ms":100,"prompt_n":2,"predicted_ms":900,"predicted_n":9}}
EOF
check ok "$(run "$tmp/ok.jsonl")" 0
[ -f "$tmp/run1.record.json" ] || { echo "FAIL ok: no record.json" >&2; fail=1; }

: > "$tmp/empty.jsonl"
check empty "$(run "$tmp/empty.jsonl")" 2

printf 'not json\n' > "$tmp/bad.jsonl"
rc=$(run "$tmp/bad.jsonl"); [ "$rc" -ne 0 ] || { echo "FAIL badjson: accepted" >&2; fail=1; }

cat > "$tmp/zero.jsonl" <<'EOF'
{"wall_ms":1000,"ttft_ms":100,"timings":{"prompt_ms":100,"prompt_n":0,"predicted_ms":900,"predicted_n":9}}
EOF
check zero "$(run "$tmp/zero.jsonl")" 3

cat > "$tmp/missing.jsonl" <<'EOF'
{"wall_ms":1000,"ttft_ms":100,"timings":{"prompt_ms":100,"prompt_n":2}}
EOF
check missing "$(run "$tmp/missing.jsonl")" 3

cat > "$tmp/nan.jsonl" <<'EOF'
{"wall_ms":NaN,"ttft_ms":100,"timings":{"prompt_ms":100,"prompt_n":2,"predicted_ms":900,"predicted_n":9}}
EOF
rc=$(run "$tmp/nan.jsonl"); [ "$rc" -ne 0 ] || { echo "FAIL nan: accepted" >&2; fail=1; }

[ "$fail" = 0 ] && echo "parse_record selftest ok"
exit "$fail"
