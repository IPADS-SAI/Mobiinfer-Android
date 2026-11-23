#!/usr/bin/env bash
# Host-only unit test runner for ggml-hexagon-oppoll.h (GGML_HEXAGON_OPPOLL=2).
# No Hexagon SDK, DSP, or device is touched; only a host C++ compiler is needed.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CXX=${CXX:-c++}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/dpsk-oppoll-hint.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror \
    -I "$HERE/.." \
    "$HERE/test-oppoll-hint.cpp" \
    -o "$TMP/test-oppoll-hint"

"$TMP/test-oppoll-hint"
