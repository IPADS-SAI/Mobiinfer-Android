#!/data/data/com.termux/files/usr/bin/bash
# One CLI arm for the app-vs-CLI comparison (runs on the phone, inside devlock).
# Usage (from yydh):
#   scp bench/app/run_arm.sh vivo:~/csm/moe-exp/run_arm.sh
#   ssh vivo 'cd ~/csm/moe-exp && ./devlock env TAG=rain-cli208 N=208 CTX=2048 READY=1 REFRESH=1 bash ./run_arm.sh'
set -u

TAG=${TAG:?TAG required}
N=${N:-208}
CTX=${CTX:-2048}
PROMPT_FILE=${PROMPT_FILE:-$HOME/csm/moe-exp/rain.txt}
REL=${REL:-$HOME/csm/moe/release-v81-sync}
READY=${READY:-1}
REFRESH=${REFRESH:-1}

cd "$HOME/csm/moe-exp" || exit 1
[ "${DROPCACHE:-1}" = 1 ] && python3 dropcache.py >/dev/null 2>&1 || true

env REL="$REL" TAG="$TAG" N="$N" CTX="$CTX" PROMPT_FILE="$PROMPT_FILE" \
  WINDOW=80 PREFETCH=1 DEPTH=8 READ=8 IO_THREADS=8 \
  READY="$READY" REFRESH="$REFRESH" bash ./sweep.sh
