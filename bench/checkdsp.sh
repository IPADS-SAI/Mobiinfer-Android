#!/data/data/com.termux/files/usr/bin/bash
# checkdsp.sh — 5 秒 DSP 体检：跑 bench 之前先过这一关。
#
# 判据（全部满足才返回 0）：
#   1. `llama-cli --list-devices` 返回 0，且完整输出里没有 FastRPC/abort 错误；
#   2. 输出里确实枚举出了 HTP（`HTP0: Hexagon ...`）；
#   3. 当前进程的 SELinux 域是 `u:r:untrusted_app_27:…`（Termux 前台终端 / 正常启动的 sshd）。
#
# 背景：用 `adb shell run-as com.termux …` 起的进程（包括这样起的 sshd 及其所有 SSH 会话）
# 落在 `u:r:runas_app` 域，该域没有 fastrpc/DSP 权限 → err 114 → ggml-hex 把 HTP 版本降级成
# v73 → 厂商库 libcdsprpc.so 在出错清理路径 double-free → `Scudo ERROR` / rc=134 abort。
# 详见 skills/bench/SKILL.md 的 “Known blocker on vivo” 一节。
#
# 用法：把本脚本和 bench.sh/sweep.sh 一起放到 ~/csm/moe-exp/，然后
#   ssh … 'cd ~/csm/moe-exp && ./devlock bash ./checkdsp.sh'
# 或在前台 Termux 终端里同样用 devlock；它会真正打开 DSP，不能与其它 HTP 用户并发。
# 测试可设 SELINUX_DOMAIN 覆盖 /proc/self/attr/current。
set -u
ROOT=${ROOT:-$HOME/csm/moe-exp}
REL=${REL:-$HOME/csm/moe/release-v81-sync}

export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:$REL/lib
export ADSP_LIBRARY_PATH="$REL/lib;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"

if [ -n "${SELINUX_DOMAIN:-}" ]; then
    dom=$SELINUX_DOMAIN
else
    dom=$(tr -d '\r\n\0' < "${SELINUX_ATTR_FILE:-/proc/self/attr/current}" 2>/dev/null)
fi
session=前台终端
[ -n "${SSH_CONNECTION:-}" ] && session=ssh
echo "=== uid=$(id -u 2>/dev/null) 域=${dom:-未知} 会话=$session rel=$REL ==="

if [ ! -x "$REL/bin/llama-cli" ]; then
    echo "RESULT: 找不到 $REL/bin/llama-cli（用 REL= 指定 release 目录）"
    exit 2
fi

out=$("$REL/bin/llama-cli" --list-devices 2>&1)
rc=$?
echo "$out"

if [ "$rc" -ne 0 ]; then
    echo "RESULT: ❌ llama-cli --list-devices 失败（rc=${rc}）"
    exit "$rc"
fi

if printf '%s\n' "$out" | grep -Eq 'err 114|FastRPC capability query failed|failed to query HTP version|failed to open session|error 0x72|Scudo ERROR|SIGABRT|Aborted'; then
    echo "RESULT: ❌ DSP 初始化报错（完整输出见上）—— err 114 常见于 runas_app 域（见 SKILL.md §3.5）；0x72 是 CDSP 卡死，重启手机"
    exit 1
fi

if ! printf '%s\n' "$out" | grep -Eq '^[[:space:]]*HTP[0-9]+:[[:space:]]+Hexagon'; then
    echo "RESULT: ❌ 没有枚举出可用的 Hexagon HTP 设备"
    exit 1
fi

case "${dom:-}" in
    *untrusted_app_27*) ;;
    *)
        echo "RESULT: ❌ SELinux 域不是 untrusted_app_27（当前=${dom:-未知}）"
        exit 1
        ;;
esac

echo "RESULT: ✅ DSP 正常"
