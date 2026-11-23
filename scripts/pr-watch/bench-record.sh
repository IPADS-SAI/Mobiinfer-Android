#!/usr/bin/env bash
# 把一份 bench 结果安装到状态目录（按 PR+head 绑定）并立刻校验。
#
# 用法：bench-record.sh <pr> <sha> <bench-result.json>
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PR=${1:?usage: bench-record.sh <pr> <sha> <bench-result.json>}
SHA=${2:?usage: bench-record.sh <pr> <sha> <bench-result.json>}
SRC=${3:?usage: bench-record.sh <pr> <sha> <bench-result.json> [evidence-dir]}
EVD_SRC=${4:-}

PRWATCH_HOME=${PRWATCH_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/pr-watch}
mkdir -p "$PRWATCH_HOME/bench"
DST="$PRWATCH_HOME/bench/pr-$PR-$SHA.json"

python3 - "$SRC" "$PR" "$SHA" <<'PY'
import json, sys
src, pr, sha = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(src, encoding="utf-8"))
if d.get("pr") != pr:
    sys.exit(f"结果里的 pr={d.get('pr')} 与参数 {pr} 不一致")
if d.get("sha") != sha:
    sys.exit(f"结果里的 sha={d.get('sha')} 与参数 {sha} 不一致")
PY

cp "$SRC" "$DST"
if [ -n "$EVD_SRC" ] && [ -d "$EVD_SRC" ]; then
  EVD_DST="$PRWATCH_HOME/bench/pr-$PR-$SHA-evidence"
  rm -rf "$EVD_DST"
  mkdir -p "$EVD_DST"
  [ -f "$EVD_SRC/manifest.json" ] && cp "$EVD_SRC/manifest.json" "$EVD_DST/"
  [ -d "$EVD_SRC/evidence" ] && cp -a "$EVD_SRC/evidence" "$EVD_DST/"
  echo "已安装证据：$EVD_DST"
fi
echo "已安装：$DST"
exec "$SCRIPT_DIR/bench-gate.sh" "$PR" "$SHA"
