#!/bin/sh
# Build the host-only tiled-pack shim used by tools/moe/pack_hexagon.py.
#
# Usage:
#   tools/moe/build_host_pack.sh <repo-root> <build-dir>
#
# <build-dir> must already contain bin/libggml-base.so (any normal CMake
# configuration of this repository produces it).  The output is
# <build-dir>/bin/libggml-hexagon.so, which can be passed to
# `pack_hexagon.py --lib-dir <build-dir>/bin`.
set -eu

root=${1:?usage: $0 <repo-root> <build-dir>}
build=${2:?usage: $0 <repo-root> <build-dir>}

root=$(cd "$root" && pwd)
build=$(cd "$build" && pwd)
test -f "$build/bin/libggml-base.so"

cxx=${CXX:-g++}
"$cxx" -O2 -fPIC -shared -std=c++17 \
    -I "$root/ggml/include" \
    -I "$root/ggml/src" \
    -I "$root/ggml/src/ggml-hexagon" \
    -I "$root/ggml/src/ggml-hexagon/htp" \
    "$root/tools/moe/host_pack.cpp" \
    "$root/ggml/src/ggml-hexagon/htp-repack.cpp" \
    -o "$build/bin/libggml-hexagon.so" \
    -L "$build/bin" -lggml-base -Wl,-rpath,'$ORIGIN'

echo "built $build/bin/libggml-hexagon.so"
