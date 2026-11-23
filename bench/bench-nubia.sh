#!/usr/bin/env bash
# bench-nubia.sh — nubia (P0110 / SM8750 / Hexagon v79) entry point.
#
# Thin wrapper around the single bench implementation (bench.sh).  It changes
# the nubia defaults:
#   REL  = $HOME/csm/moe/release-v81-chunkpaging-nubia (contains htp-v79)
#   ARCH = 79 (GGML_HEXAGON_ARCH); set ARCH= (empty) to let the backend
#          auto-detect, or ARCH=<n> to override.
#   HOSTBUF = 0.  Current integration builds hit
#          "HTP0 buffer mapping failed ... error 0x0000000e" on nubia with
#          GGML_HEXAGON_HOSTBUF=1 (287 MiB mapping at draft-context setup);
#          HOSTBUF=0 uses the non-host path and is the v79 default here.
#   EXPERTS = experts-q2k-idnow, the 28.1 t/s nubia baseline (Q1_0 + L38/39
#          Q2_K, 4.83 GB).  bench.sh's default experts-htp-tiled is Q3_K
#          (16.1 GB) and must not be used implicitly for nubia.
# Everything else (WINDOW/... and the llama-cli argv) is bench.sh.
#
# The wrapper also prints the resolved identity on stderr before exec'ing
# bench.sh, and BENCH_CONFIG_ONLY=1 stops after that without touching the DSP.
# A missing EXPERTS directory is a hard error (exit 2): silently falling back
# to htp-tiled is what caused the T-0289 package drift.
#
# The previous phone-local ~/csm/moe-exp/bench-nubia.sh was a stale fork of
# bench.sh (it missed DRAFT_DEVICE/HOSTBUF and defaulted REL to the vivo
# release).  Keeping a wrapper avoids a second copy drifting from bench.sh.
set -u

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$HOME/csm/moe-exp}
REL=${REL:-$HOME/csm/moe/release-v81-chunkpaging-nubia}
# ${ARCH-79}: an explicitly empty ARCH means "do not export GGML_HEXAGON_ARCH".
ARCH=${ARCH-79}
HOSTBUF=${HOSTBUF:-0}
MODEL=${MODEL:-$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/minicpm47-les-hexagon.gguf}
EXPERTS=${EXPERTS:-$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q2k-idnow}
DRAFT=${DRAFT:-$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/dspark-MiniCPM-V-4_7-q8_0.gguf}
# Export ROOT/MODEL/DRAFT too so bench.sh resolves exactly the paths printed
# below (a caller-supplied ROOT must not make the identity line lie).
export ROOT REL ARCH HOSTBUF MODEL EXPERTS DRAFT

if [ ! -d "$EXPERTS" ]; then
    {
        echo "ERROR: nubia EXPERTS directory not found: $EXPERTS"
        echo "candidates:"
        printf '  %s\n' \
            "$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q2k-idnow" \
            "$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-q1-30tps" \
            "$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-htp-tiled"
        echo "Set EXPERTS (or ROOT) to one of the candidates."
    } >&2
    exit 2
fi

if [ "$(basename "$EXPERTS")" = "experts-htp-tiled" ]; then
    echo "WARN: Q3_K htp-tiled is not the 28.1 baseline" >&2
fi

experts_basename=$(basename "$EXPERTS")
experts_bytes=$(du -sb "$EXPERTS" 2>/dev/null | awk '{print $1}')
[ -n "$experts_bytes" ] || experts_bytes=unknown

manifest_sha256=unknown
for manifest in manifest.json experts-manifest.json SHA256SUMS; do
    if [ -f "$EXPERTS/$manifest" ]; then
        manifest_sha256=$(sha256sum "$EXPERTS/$manifest" 2>/dev/null | awk '{print $1}')
        [ -n "$manifest_sha256" ] || manifest_sha256=unknown
        break
    fi
done

printf 'nubia-config: REL=%s ARCH=%s HOSTBUF=%s MODEL=%s EXPERTS=%s DRAFT=%s\n' \
    "$REL" "$ARCH" "$HOSTBUF" "$MODEL" "$EXPERTS" "$DRAFT" >&2
printf 'nubia-identity: experts_basename=%s experts_bytes=%s manifest_sha256=%s\n' \
    "$experts_basename" "$experts_bytes" "$manifest_sha256" >&2

if [ "${BENCH_CONFIG_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$SELF_DIR/bench.sh" "$@"
