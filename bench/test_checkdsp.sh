#!/usr/bin/env bash
# Offline checks for checkdsp.sh. A fake llama-cli covers success/failure shapes.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
mkdir -p "$tmp/rel/bin"

cat > "$tmp/rel/bin/llama-cli" <<'EOF'
#!/usr/bin/env bash
case "${MOCK_DSP:-}" in
    ok)
        for i in $(seq 1 10); do echo "diagnostic line $i"; done
        echo "Available devices:"
        echo "  HTP0: Hexagon (0 MiB, 0 MiB free)"
        ;;
    late-error)
        for i in $(seq 1 12); do echo "diagnostic line $i"; done
        echo "E ggml-hex: FastRPC capability query failed (err 114)"
        ;;
    abort)
        for i in $(seq 1 12); do echo "diagnostic line $i"; done
        echo "Scudo ERROR: invalid chunk state"
        exit 134
        ;;
    no-htp)
        for i in $(seq 1 12); do echo "diagnostic line $i"; done
        echo "Available devices:"
        ;;
    htp-open-error)
        # Enumeration succeeds and rc=0, but the HTP session fails to open.
        for i in $(seq 1 12); do echo "diagnostic line $i"; done
        echo "Available devices:"
        echo "  HTP0: Hexagon (0 MiB, 0 MiB free)"
        echo "HTP0 failed to open session : uri file:///libggml-htp-v81.so error 0x72"
        ;;
    *) echo "unknown MOCK_DSP=${MOCK_DSP:-}"; exit 2 ;;
esac
EOF
chmod +x "$tmp/rel/bin/llama-cli"

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

out=$(REL="$tmp/rel" SELINUX_DOMAIN='u:r:untrusted_app_27:s0' MOCK_DSP=ok bash "$HERE/checkdsp.sh" 2>&1)
rc=$?
check_rc success 0 "$rc"
check_contains success "diagnostic line 10" "$out"
check_contains success "RESULT: ✅ DSP 正常" "$out"

out=$(REL="$tmp/rel" SELINUX_DOMAIN='u:r:runas_app:s0' MOCK_DSP=ok bash "$HERE/checkdsp.sh" 2>&1)
rc=$?
check_rc bad-domain 1 "$rc"
check_contains bad-domain "SELinux 域不是 untrusted_app_27" "$out"

out=$(REL="$tmp/rel" SELINUX_DOMAIN='u:r:untrusted_app_27:s0' MOCK_DSP=late-error bash "$HERE/checkdsp.sh" 2>&1)
rc=$?
check_rc late-error 1 "$rc"
check_contains late-error "diagnostic line 12" "$out"
check_contains late-error "err 114" "$out"

out=$(REL="$tmp/rel" SELINUX_DOMAIN='u:r:untrusted_app_27:s0' MOCK_DSP=abort bash "$HERE/checkdsp.sh" 2>&1)
rc=$?
check_rc abort 134 "$rc"
check_contains abort "diagnostic line 12" "$out"

out=$(REL="$tmp/rel" SELINUX_DOMAIN='u:r:untrusted_app_27:s0' MOCK_DSP=no-htp bash "$HERE/checkdsp.sh" 2>&1)
rc=$?
check_rc no-htp 1 "$rc"
check_contains no-htp "没有枚举出可用的 Hexagon HTP" "$out"

out=$(REL="$tmp/rel" SELINUX_DOMAIN='u:r:untrusted_app_27:s0' MOCK_DSP=htp-open-error bash "$HERE/checkdsp.sh" 2>&1)
rc=$?
check_rc htp-open-error 1 "$rc"
check_contains htp-open-error "DSP 初始化报错" "$out"
case "$out" in
    *"RESULT: ✅ DSP 正常"*) echo "FAIL htp-open-error: reported DSP normal despite 0x72" >&2; fail=1 ;;
esac

[ "$fail" -eq 0 ] && echo "checkdsp selftest ok"
exit "$fail"
