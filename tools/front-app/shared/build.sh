#!/bin/bash
# build.sh —— 交叉编译本目录所有手机侧 C 探针（aarch64-linux-android，动态 PIE）。
#
# 背景：这些工具原来没有构建脚本，都是各脚本里手敲 clang（见 lib.sh 的
# fa_install_tools 只编 runp/directio，其余是手工编译的）。本脚本把它们收拢，
# 顺带加入 iotrace。
#
# 用法：
#   ./build.sh                 # 全部编到 build/
#   ./build.sh iotrace         # 只编指定的（可给多个名字，不带 .c）
#   ./build.sh --check         # 只检查 build/ 里已有产物的 ELF 架构
#
# 环境变量：
#   ANDROID_NDK   NDK 根目录（默认 $HOME/env/android/ndk/27.0.12077973，
#                 否则取 $HOME/env/android/ndk 下版本号最大的一个）
#   ANDROID_API   target API level（默认 35，与 lib.sh 一致）
#   NDK_CC        直接指定编译器（覆盖上面两个）
#
# 约定（与 lib.sh / 现有产物一致）：产物在 build/（.gitignore 已排除），
# 一律 -O2 -pie -fPIE（动态 PIE），ABI aarch64，无需 push 到手机即可 `adb push` 使用。
#
# 验证状态：本脚本用 NDK 27.0（aarch64-linux-android35）交叉编译，
# `./build.sh --check` 可复核 ELF 架构。产物需要在手机上运行验证。
set -u
SHARED=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD=$SHARED/build
HOST_TAG=linux-x86_64
API=${ANDROID_API:-35}

log() { echo "[build] $*" >&2; }
die() { echo "[build] ERROR: $*" >&2; exit 1; }

resolve_cc() {
  [ -n "${NDK_CC:-}" ] && { echo "$NDK_CC"; return; }
  local ndk=${ANDROID_NDK:-}
  if [ -z "$ndk" ]; then
    if [ -d "$HOME/env/android/ndk/27.0.12077973" ]; then
      ndk=$HOME/env/android/ndk/27.0.12077973
    elif [ -d "$HOME/env/android/ndk" ]; then
      # 版本号最大的那个（sort -V 对 25.2 < 27.0 正确）
      ndk=$(ls -d "$HOME"/env/android/ndk/*/ 2>/dev/null | sed 's:/*$::' | sort -V | tail -1)
    fi
  fi
  [ -n "$ndk" ] && [ -d "$ndk" ] || die "找不到 NDK：请设 ANDROID_NDK（或 NDK_CC）"
  local cc=$ndk/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android${API}-clang
  [ -x "$cc" ] || die "找不到编译器 $cc（换 ANDROID_API？）"
  echo "$cc"
}

check_elf() {
  local f=$1 n
  n=$(basename "$f")
  [ -f "$f" ] || { printf "%-12s 缺失\n" "$n"; return 1; }
  if command -v readelf >/dev/null 2>&1; then
    local info
    # LC_ALL=C：readelf 的字段名在中文 locale 下会变成"类别/系统架构"，
    # 不加这一条解析会静默得到空字符串。
    info=$(LC_ALL=C readelf -h "$f" 2>/dev/null | sed -n 's/^ *Class: *//p; s/^ *Machine: *//p' | paste -sd' ' -)
    printf "%-12s %-24s %s MiB\n" "$n" "$info" "$(du -m "$f" | cut -f1)"
  else
    file "$f"
  fi
}

ALL="runp directio iotrace"

mkdir -p "$BUILD"

if [ "${1:-}" = "--check" ]; then
  for n in $ALL; do check_elf "$BUILD/$n"; done
  exit 0
fi

TARGETS=""
for a in "$@"; do
  case "$a" in
    --help|-h)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) TARGETS="$TARGETS ${a%.c}" ;;
  esac
done
[ -n "$TARGETS" ] || TARGETS=$ALL

CC=$(resolve_cc) || exit 1
log "CC=$CC"
fail=0
for n in $TARGETS; do
  src=$SHARED/$n.c
  [ -f "$src" ] || { log "跳过 $n：没有 $src"; fail=1; continue; }
  out=$BUILD/$n
  # 源码比产物新才重编（和 lib.sh 的 -nt 判断一致）
  if [ -x "$out" ] && [ ! "$src" -nt "$out" ]; then
    log "$n: 已是最新"
    continue
  fi
  log "编译 $n ..."
  # shellcheck disable=SC2086
  # 必须**动态 PIE**：Android 的 linker 不接受 ET_EXEC（报 "unexpected e_type: 2"），
  # 而 -static-pie 编出来的静态 PIE 在 Termux 里会段错误。实测 -pie -fPIE（动态）可用。
  # 历史：早期用 -static 编出 ET_EXEC，在 Termux 一直能跑，**设备重启后突然全部失效**。
  if ! $CC -O2 -pie -fPIE -o "$out" "$src"; then
    log "$n 编译失败"
    fail=1
    continue
  fi
  log "$n -> build/$n ($(du -m "$out" | cut -f1) MiB)"
done


if [ $fail -eq 0 ]; then
  log "全部完成。push 示例： adb push $BUILD/iotrace /data/local/tmp/sp/iotrace"
else
  log "有工具编译失败，见上文"
fi
exit $fail
