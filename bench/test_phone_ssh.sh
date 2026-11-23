#!/usr/bin/env bash
# Offline checks for phone-ssh.sh check-path status propagation and device locking.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
mkdir -p "$tmp/bin"

cat > "$tmp/bin/adb" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    "devices") printf 'SERIAL\tdevice\n' ;;
    "forward --list") printf 'SERIAL tcp:2223 tcp:18022\n' ;;
    *) printf 'unexpected adb args: %s\n' "$*" >&2; exit 2 ;;
esac
EOF

cat > "$tmp/bin/ssh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SSH_LOG"
[ "$1" = "-p" ] || { echo "unexpected ssh args: $*" >&2; exit 2; }
shift 2
[ "$#" -eq 2 ] || { echo "unexpected ssh args: $*" >&2; exit 2; }
remote_cmd=$2
HOME="$MOCK_REMOTE_HOME" bash -c "$remote_cmd"
EOF
chmod +x "$tmp/bin/adb" "$tmp/bin/ssh"

mkdir -p "$tmp/remote-home/csm/moe-exp" "$tmp/default-root" "$tmp/custom-root"
cat > "$tmp/remote-home/csm/moe-exp/devlock" <<'EOF'
#!/usr/bin/env bash
printf 'lock cwd=%s args=%s\n' "$PWD" "$*" >> "$DEVICE_LOCK_LOG"
exec "$@"
EOF
for root in "$tmp/default-root" "$tmp/custom-root"; do
    cat > "$root/checkdsp.sh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$PWD" >> "$CHECK_LOG"
exit "${CHECK_RC:-0}"
EOF
done
chmod +x "$tmp/remote-home/csm/moe-exp/devlock" \
    "$tmp/default-root/checkdsp.sh" "$tmp/custom-root/checkdsp.sh"

check() {
    local label=$1 want=$2 got=$3
    if [ "$got" -ne "$want" ]; then
        echo "FAIL $label: rc=$got want=$want" >&2
        fail=1
    fi
}

check_contains() {
    local label=$1 needle=$2 file=$3
    grep -Fq -- "$needle" "$file" || {
        echo "FAIL $label: log missing '$needle'" >&2
        fail=1
    }
}

PATH="$tmp/bin:$PATH" SERIAL=SERIAL SSH_LOG="$tmp/ssh.log" \
    MOCK_REMOTE_HOME="$tmp/remote-home" REMOTE_ROOT="$tmp/default-root" \
    CHECK_LOG="$tmp/check.log" DEVICE_LOCK_LOG="$tmp/lock.log" CHECK_RC=42 \
    bash "$HERE/phone-ssh.sh" check >/dev/null 2>&1
check remote-failure 42 "$?"
check_contains remote-failure 'bash "$HOME/csm/moe-exp/devlock" bash ./checkdsp.sh' "$tmp/ssh.log"
check_contains remote-failure "$tmp/default-root" "$tmp/check.log"

PATH="$tmp/bin:$PATH" SERIAL=SERIAL SSH_LOG="$tmp/ssh-ok.log" \
    MOCK_REMOTE_HOME="$tmp/remote-home" REMOTE_ROOT="$tmp/default-root" \
    CHECK_LOG="$tmp/check-ok.log" DEVICE_LOCK_LOG="$tmp/lock-ok.log" CHECK_RC=0 \
    bash "$HERE/phone-ssh.sh" check >/dev/null 2>&1
check remote-success 0 "$?"

if [ -e "$tmp/custom-root/devlock" ]; then
    echo "FAIL custom-root-no-devlock: unexpected custom devlock" >&2
    fail=1
fi
PATH="$tmp/bin:$PATH" SERIAL=SERIAL SSH_LOG="$tmp/ssh-custom.log" \
    MOCK_REMOTE_HOME="$tmp/remote-home" REMOTE_ROOT="$tmp/custom-root" \
    CHECK_LOG="$tmp/check-custom.log" DEVICE_LOCK_LOG="$tmp/lock-custom.log" CHECK_RC=23 \
    bash "$HERE/phone-ssh.sh" check >/dev/null 2>&1
check custom-root-status 23 "$?"
check_contains custom-root-check "$tmp/custom-root" "$tmp/check-custom.log"
check_contains custom-root-standard-lock \
    "lock cwd=$tmp/custom-root args=bash ./checkdsp.sh" "$tmp/lock-custom.log"

[ "$fail" -eq 0 ] && echo "phone-ssh selftest ok"
exit "$fail"
