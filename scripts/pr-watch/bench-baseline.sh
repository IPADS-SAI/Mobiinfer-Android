#!/usr/bin/env bash
# 基线缓存：main 的 v81 构建只做一次，之后按 main sha 复用。
#
# 用法：
#   bench-baseline.sh path            # 打印可用基线目录（没有则空）
#   bench-baseline.sh ensure [--force] # 确保当前 origin/main 有缓存，必要时构建
#   bench-baseline.sh status
#
# 缓存位置：$PRWATCH_HOME/baseline/<main-sha>/
#   bin/                 llama-cli、llama-mtmd-cli、*.so
#   skel/                libggml-htp-v81.so
#   manifest.json        main_sha / cmake 参数 / sha256 / flags_hash / 时间
#   .complete            构建完成的标记
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
STATE=$PRWATCH_HOME
BASE_ROOT="$STATE/baseline"
mkdir -p "$BASE_ROOT"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

for d in "$HOME"/env/android/cmake/*/bin; do [ -x "$d/cmake" ] && PATH="$d:$PATH"; done
export PATH
if [ -z "${HEXAGON_SDK_ROOT:-}" ] || [ ! -d "${HEXAGON_SDK_ROOT:-/nonexistent}" ]; then
  HEXAGON_SDK_ROOT=$(ls -d "$HOME"/env/hexagon/* 2>/dev/null | sort -V | tail -1)
fi
if [ -z "${HEXAGON_TOOLS_ROOT:-}" ] || [ ! -d "${HEXAGON_TOOLS_ROOT:-/nonexistent}" ]; then
  HEXAGON_TOOLS_ROOT=$(ls -d "$HEXAGON_SDK_ROOT"/tools/HEXAGON_Tools/* 2>/dev/null | sort -V | tail -1)
fi
if [ -z "${ANDROID_NDK:-}" ] || [ ! -d "${ANDROID_NDK:-/nonexistent}" ]; then
  ANDROID_NDK=$(ls -d "$HOME"/env/android/ndk/* 2>/dev/null | sort -V | tail -1)
fi

CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28
  -DGGML_HEXAGON=ON -DGGML_OPENCL=OFF -DGGML_VULKAN=OFF
  -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"
  -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT"
  -DPREBUILT_LIB_DIR=android_aarch64
  -DGGML_HEXAGON_PREBUILT_LIB_DIR=android_aarch64
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_VOXCPM2=OFF
)
FLAGS_HASH=$(printf '%s\n' "${CMAKE_ARGS[@]}" | sha256sum | cut -d' ' -f1)

main_sha() {
  git -C "$REPO_ROOT" fetch -q --no-tags origin "$MERGE_BASE_BRANCH" 2>/dev/null || true
  git -C "$REPO_ROOT" rev-parse "origin/$MERGE_BASE_BRANCH"
}

dir_for() { echo "$BASE_ROOT/$1"; }

prune() {
  find "$BASE_ROOT" -maxdepth 1 -type d -name '[0-9a-f]*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk -v k="${BENCH_BASELINE_KEEP:-2}" 'NR>k{print $2}' \
    | xargs -r rm -rf
}

valid() { # valid <sha> <dir>
  local sha=$1 dir=$2
  [ -f "$dir/.complete" ] || return 1
  [ -f "$dir/manifest.json" ] || return 1
  [ "$(jq -r '.main_sha' "$dir/manifest.json" 2>/dev/null)" = "$sha" ] || return 1
  [ "$(jq -r '.flags_hash' "$dir/manifest.json" 2>/dev/null)" = "$FLAGS_HASH" ] || return 1
  [ -x "$dir/bin/llama-cli" ] || return 1
  return 0
}

cmd=${1:?usage: bench-baseline.sh path|ensure|status}
case "$cmd" in
  path)
    sha=$(main_sha)
    dir=$(dir_for "$sha")
    valid "$sha" "$dir" && echo "$dir"
    ;;
  status)
    sha=$(main_sha)
    dir=$(dir_for "$sha")
    echo "main=$sha"
    echo "flags_hash=${FLAGS_HASH:0:12}"
    if valid "$sha" "$dir"; then
      echo "baseline=ready $dir"
      jq -c '{main_sha, created_at, llama_cli_sha256}' "$dir/manifest.json"
    else
      echo "baseline=missing $dir"
    fi
    ls -1 "$BASE_ROOT" 2>/dev/null | head -5
    ;;
  ensure)
    sha=$(main_sha)
    dir=$(dir_for "$sha")
    if [ "${2:-}" != "--force" ] && valid "$sha" "$dir"; then
      log "基线已存在：$dir（main=${sha:0:8}）"
      exit 0
    fi
    if [ -d "$dir" ]; then
      log "缓存失效（main 更新或构建参数变化）：$dir，重建"
    fi
    command -v cmake >/dev/null || { log "缺 cmake"; exit 1; }
    command -v ninja >/dev/null || { log "缺 ninja"; exit 1; }
    [ -d "$ANDROID_NDK" ] || { log "ANDROID_NDK 无效"; exit 1; }

    wt="$BASE_ROOT/wt-$sha"
    if [ -e "$wt/.git" ]; then git -C "$wt" checkout -q --detach "$sha"; else
      git -C "$REPO_ROOT" worktree add -q --detach --force "$wt" "$sha"
    fi
    build="$wt/build-hexagon"
    log "构建 main@${sha:0:8} → $dir"
    cmake -B "$build" "${CMAKE_ARGS[@]}" > "$BASE_ROOT/cmake-$sha.log" 2>&1
    ninja -C "$build" -j"$(nproc)" llama-cli llama-mtmd-cli > "$BASE_ROOT/build-$sha.log" 2>&1

    rm -rf "$dir"; mkdir -p "$dir/bin" "$dir/skel"
    cp -a "$build"/bin/. "$dir/bin/"
    [ -d "$build/lib" ] && cp -a "$build/lib" "$dir/lib"
    skel="$build/ggml/src/ggml-hexagon/libggml-htp-v81.so"
    [ -f "$skel" ] && cp -a "$skel" "$dir/skel/"

    jq -nc --arg sha "$sha" --arg flags "$FLAGS_HASH" \
      --arg created "$(date -Is)" \
      --arg cli_sha "$(sha256sum "$dir/bin/llama-cli" | cut -d' ' -f1)" \
      --arg cmake "$(printf '%s ' "${CMAKE_ARGS[@]}")" \
      '{main_sha:$sha, flags_hash:$flags, created_at:$created, llama_cli_sha256:$cli_sha, cmake:$cmake}' \
      > "$dir/manifest.json"
    touch "$dir/.complete"
    prune
    git -C "$REPO_ROOT" worktree remove --force "$wt" 2>/dev/null || true
    log "基线完成：$dir"
    ;;
  seed)
    src=${2:?usage: bench-baseline.sh seed <build-dir> [main-sha]}
    sha=${3:-$(main_sha)}
    dir=$(dir_for "$sha")
    [ -x "$src/bin/llama-cli" ] || { log "seed：$src/bin/llama-cli 不存在"; exit 1; }
    rm -rf "$dir"; mkdir -p "$dir/bin" "$dir/skel"
    cp -a "$src"/bin/. "$dir/bin/"
    [ -d "$src/lib" ] && cp -a "$src/lib" "$dir/lib"
    skel="$src/ggml/src/ggml-hexagon/libggml-htp-v81.so"
    [ -f "$skel" ] && cp -a "$skel" "$dir/skel/"
    jq -nc --arg sha "$sha" --arg flags "$FLAGS_HASH" \
      --arg created "$(date -Is)" \
      --arg cli_sha "$(sha256sum "$dir/bin/llama-cli" | cut -d' ' -f1)" \
      --arg cmake "(seeded from $src)" \
      '{main_sha:$sha, flags_hash:$flags, created_at:$created, llama_cli_sha256:$cli_sha, cmake:$cmake}' \
      > "$dir/manifest.json"
    touch "$dir/.complete"
    log "已从 $src 播种基线：$dir"
    ;;
  *) echo "unknown command: $cmd" >&2; exit 2;;
esac
