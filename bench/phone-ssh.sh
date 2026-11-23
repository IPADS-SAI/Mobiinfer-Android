#!/usr/bin/env bash
# phone-ssh.sh — 一条命令连/修 vivo 的 SSH（经 adb 端口转发）。
#
#   ./phone-ssh.sh connect   # 检查设备 + 建转发 + 试 SSH（默认动作）
#   ./phone-ssh.sh check     # 连上后跑 DSP 体检（checkdsp.sh），并打印 SELinux 域
#   ./phone-ssh.sh repair    # SSH 不通时修复：拉 Termux 前台 → 停旧 sshd → 在终端里起 sshd
#   ./phone-ssh.sh scp <本地> <远端相对路径>   # 用 adb 转发通道传文件
#
# 关键约束（见 skills/bench/SKILL.md）：
#   * sshd 必须从 **Termux 前台终端** 启动，落在 u:r:untrusted_app_27 域才有 DSP 权限；
#     用 `adb shell run-as com.termux … sshd` 起会落到 runas_app 域 → err 114 → 推理 abort，
#     且该状态重启也不恢复。
#   * 端口：手机 18022 → 本机 2223（Host vivo 已在 ~/.ssh/config 里）。
set -u

SERIAL=${SERIAL:-10AG9L1QAA003LR}
LADDR=${LADDR:-2223}          # 本机端口
PADDR=${PADDR:-18022}         # 手机 sshd 端口
USER_HOST=${USER_HOST:-u0_a371@127.0.0.1}
REMOTE_ROOT=${REMOTE_ROOT:-/data/data/com.termux/files/home/csm/moe-exp}
TERMUX=${TERMUX:-/data/data/com.termux/files/usr}
PKG=${PKG:-com.termux}

die() { echo "[phone-ssh] $*" >&2; exit 1; }
say() { echo "[phone-ssh] $*"; }

need_adb() {
    command -v adb >/dev/null || die "找不到 adb"
    adb devices | grep -q "^${SERIAL}[[:space:]]*device" || die "看不到设备 $SERIAL（插 USB / 授权调试？）"
}

ensure_forward() {
    if adb forward --list | grep -q "tcp:${LADDR} tcp:${PADDR}"; then
        say "转发已存在：tcp:${LADDR} → tcp:${PADDR}"
    else
        adb forward --remove "tcp:${LADDR}" >/dev/null 2>&1
        adb forward "tcp:${LADDR}" "tcp:${PADDR}" >/dev/null || die "建转发失败"
        say "已建转发：tcp:${LADDR} → tcp:${PADDR}"
    fi
}

sshd_listening() {
    adb shell "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null" 2>/dev/null | tr -d '\r' |
        awk -v p="$(printf '%04X' "$PADDR")" 'NR>1 { split($2,a,":"); if (a[2]==p && $4=="0A") { found=1 } } END { exit !found }'
}

ssh_ok() {
    timeout 15 ssh -o ConnectTimeout=8 -o BatchMode=yes -p "$LADDR" "$USER_HOST" true 2>/dev/null
}

# 在 Termux 里执行一条命令（adb input 代敲，用于必须落在 untrusted_app 域的动作）
type_in_terminal() {
    adb shell input text "$(printf '%s' "$1" | sed 's/ /%s/g')"
    adb shell input keyevent KEYCODE_ENTER
}

bring_termux_front() {
    adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
    sleep 1
    adb shell input swipe 540 2200 540 900 200 >/dev/null 2>&1   # 这台是「上滑打开」，无密码
    sleep 1
    adb shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
    sleep 3
    say "Termux 已拉到前台"
}

kill_stale_sshd() {
    local pids
    pids=$(adb shell "ps -A -o PID,NAME | grep -w sshd | awk '{print \$1}'" 2>/dev/null | tr -d '\r')
    [ -z "$pids" ] && { say "没有残留 sshd"; return 0; }
    for pid in $pids; do
        # 跨域 SELinux 拒绝 kill，必须用 run-as 以 App 身份杀
        adb shell "run-as ${PKG} kill -9 ${pid}" >/dev/null 2>&1
        say "已杀 sshd pid=$pid"
    done
    sleep 2
}

start_sshd_in_terminal() {
    say "在 Termux 前台终端里启动 sshd -p ${PADDR}"
    type_in_terminal "sshd -p ${PADDR}"
    sleep 4
}

verify() {
    ssh_ok || die "仍然连不上（看手机屏幕：Termux 里是否有报错）"
    local dom
    dom=$(ssh -p "$LADDR" "$USER_HOST" 'cat /proc/self/attr/current' 2>/dev/null | tr -d '\r\0')
    say "SSH 通了，SELinux 域 = ${dom:-未知}"
    case "$dom" in
        *untrusted_app_27*) ;;
        *) die "域不是 untrusted_app_27（当前=${dom:-未知}）—— 跑推理会报 err 114，执行 repair" ;;
    esac
}

cmd_connect() {
    need_adb
    ensure_forward
    if ssh_ok; then
        verify
    else
        say "SSH 不通：手机侧 18022 $(sshd_listening && echo 在监听 || echo 没监听)"
        die "先执行：./phone-ssh.sh repair"
    fi
}

run_dsp_check() {
    local qroot
    qroot=$(printf '%q' "$REMOTE_ROOT")
    # The whole check runs under the phone-side device lock; the remote shell
    # returns checkdsp's status unchanged (no tail/pipe mask).
    ssh -p "$LADDR" "$USER_HOST" \
        "cat /proc/self/attr/current; cd ${qroot} && bash \"\$HOME/csm/moe-exp/devlock\" bash ./checkdsp.sh"
}

cmd_check() {
    need_adb
    ensure_forward
    run_dsp_check
}

cmd_repair() {
    need_adb
    ensure_forward
    bring_termux_front
    kill_stale_sshd
    start_sshd_in_terminal
    ensure_forward
    verify
    say "修复完成；DSP 体检："
    run_dsp_check
}

cmd_scp() {
    [ $# -ge 2 ] || die "用法：./phone-ssh.sh scp <本地文件> <手机上的相对路径（相对 \$HOME）>"
    need_adb
    ensure_forward
    scp -P "$LADDR" "$1" "${USER_HOST}:$2"
}

action=${1:-connect}
shift || true
case "$action" in
    connect) cmd_connect ;;
    check)   cmd_check ;;
    repair)  cmd_repair ;;
    scp)     cmd_scp "$@" ;;
    *) die "未知动作：$action（connect|check|repair|scp）" ;;
esac
