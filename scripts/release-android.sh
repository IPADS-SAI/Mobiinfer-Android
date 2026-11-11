#!/usr/bin/env bash
#
# Release MobiInfer-Android end to end:
#
#   1. build the signed release APK (or reuse --apk)
#   2. verify versionCode / versionName and the v2+v3 signature
#   3. enforce the MobiInfer-Android-v<versionName>.apk name
#   4. create the git tag and push it to GitCode
#   5. create the GitCode release and upload the APK + SHA256SUMS
#   6. publish update.json (GitCode -> Chengdu -> GitHub) and mirror the APK
#
# The App reads update.json from the endpoint baked into the build (see the
# updateManifestUrl property in app/build.gradle.kts) at most once a day and
# installs the first mirror whose bytes match sha256 and whose signer matches
# the installed app.
#
# Usage:
#   scripts/release-android.sh --tag v1.0.0 [options]
#
# Options:
#   --tag <tag>            Release tag (default: v<versionName>)
#   --apk <path>           Use an existing APK instead of building
#   --notes-file <path>    Release notes (default: generated one-liner)
#   --min-supported <code> versionCode below which the update is mandatory
#   --prerelease           Mark the GitCode release as a pre-release
#   --no-build             Do not run gradle (requires --apk or a fresh build)
#   --with-debug           Also build a debug APK and publish the "debug"
#                          channel (GitCode + Chengdu)
#   --replace              Re-upload assets when the release already exists
#                          (refused by default: published versions are immutable)
#   --no-verify            Skip the post-publish download/checksum checks
#   --no-push              Do not create/push the tag, touch GitCode or publish
#                          update.json (implies --no-chengdu unless --chengdu)
#   --no-chengdu           Do not upload the APK / update.json to Chengdu
#   --chengdu              Publish to Chengdu even together with --no-push
#   --allow-dirty          Tag HEAD even if the working tree has changes
#   --allow-unpinned-cert  Skip the CERT_SHA256 signer-identity check
#
# Environment overrides:
#   GITCODE_REPO, GITCODE_REMOTE, CHENGDU_HOST, CHENGDU_WEBROOT, PUBLIC_BASE,
#   GITHUB_REPO, CERT_SHA256, ANDROID_HOME
#
# Per-machine values belong in scripts/release.local.env (gitignored); see
# scripts/release.local.env.example. GITHUB_REPO defaults to the `origin`
# remote when unset.
#
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
APP_DIR="$ROOT/Android-App/MobiInfer-Android"
GRADLE_FILE="$APP_DIR/app/build.gradle.kts"
DIST="$APP_DIR/app/build/release-dist"

if [[ -f "$ROOT/scripts/release.local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/scripts/release.local.env"
    set +a
fi

GITCODE_REPO=${GITCODE_REPO:-}
GITCODE_REMOTE=${GITCODE_REMOTE:-gitcode}
GITHUB_REPO=${GITHUB_REPO:-}
CHENGDU_HOST=${CHENGDU_HOST:-}
CHENGDU_WEBROOT=${CHENGDU_WEBROOT:-}
PUBLIC_BASE=${PUBLIC_BASE:-}
CERT_SHA256=${CERT_SHA256:-}

if [[ -z "$GITHUB_REPO" ]]; then
    origin_url=$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)
    GITHUB_REPO=$(sed -nE 's#.*github\.com[:/]([^/]+/[^/]+?)(\.git)?$#\1#p' <<<"$origin_url")
fi

TAG=""
APK=""
NOTES_FILE=""
MIN_SUPPORTED=0
DO_BUILD=1
DO_PUSH=1
DO_CHENGDU=1
CHENGDU_EXPLICIT=0
ALLOW_DIRTY=0
PRERELEASE=0
REPLACE=0
VERIFY=1
DO_DEBUG=0
ALLOW_UNPINNED=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
log() { printf '\n== %s\n' "$*"; }

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

size_of() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

usage() {
    awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)           TAG=${2:?}; shift 2 ;;
        --apk)           APK=${2:?}; DO_BUILD=0; shift 2 ;;
        --notes-file)    NOTES_FILE=${2:?}; shift 2 ;;
        --min-supported) MIN_SUPPORTED=${2:?}; shift 2 ;;
        --prerelease)    PRERELEASE=1; shift ;;
        --no-build)      DO_BUILD=0; shift ;;
        --no-push)       DO_PUSH=0; (( CHENGDU_EXPLICIT )) || DO_CHENGDU=0; shift ;;
        --no-chengdu)    DO_CHENGDU=0; shift ;;
        --chengdu)       DO_CHENGDU=1; CHENGDU_EXPLICIT=1; shift ;;
        --allow-dirty)   ALLOW_DIRTY=1; shift ;;
        --allow-unpinned-cert) ALLOW_UNPINNED=1; shift ;;
        --replace)       REPLACE=1; shift ;;
        --no-verify)     VERIFY=0; shift ;;
        --with-debug)    DO_DEBUG=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "unknown option: $1 (try --help)" ;;
    esac
done

if [[ -z "$PUBLIC_BASE" ]]; then
    die "PUBLIC_BASE is unset; copy scripts/release.local.env.example to scripts/release.local.env"
fi
if (( DO_PUSH )) && [[ -z "$GITCODE_REPO" ]]; then
    die "GITCODE_REPO is unset (scripts/release.local.env)"
fi
if (( DO_CHENGDU )); then
    [[ -n "$CHENGDU_HOST" ]] || die "CHENGDU_HOST is unset (scripts/release.local.env)"
    [[ -n "$CHENGDU_WEBROOT" ]] || die "CHENGDU_WEBROOT is unset (scripts/release.local.env)"
fi
if [[ -z "$CERT_SHA256" ]] && (( ! ALLOW_UNPINNED )); then
    die "CERT_SHA256 is unset; pin the release certificate or pass --allow-unpinned-cert"
fi

[[ -f "$GRADLE_FILE" ]] || die "missing $GRADLE_FILE"

VERSION_CODE=$(sed -nE 's/^[[:space:]]*versionCode = ([0-9]+).*/\1/p' "$GRADLE_FILE" | head -1)
VERSION_NAME=$(sed -nE 's/^[[:space:]]*versionName = "([^"]+)".*/\1/p' "$GRADLE_FILE" | head -1)
[[ -n "$VERSION_CODE" && -n "$VERSION_NAME" ]] || die "cannot read version from $GRADLE_FILE"
TAG=${TAG:-v$VERSION_NAME}

log "release $TAG ($VERSION_NAME, versionCode $VERSION_CODE)"

# ---------------------------------------------------------------- build / verify

if (( ! ALLOW_DIRTY )) && {
        ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet
    }; then
    die "working tree is dirty; commit or stash first (or pass --allow-dirty)"
fi

if (( DO_BUILD )); then
    log "building :app:assembleRelease"
    # Keep the Gradle user home at its default: the release signing
    # properties live in ~/.gradle/gradle.properties and are only visible there.
    ( cd "$APP_DIR" && ./gradlew :app:assembleRelease )
fi

if [[ -z "$APK" ]]; then
    APK="$APP_DIR/app/build/outputs/apk/release/app-release.apk"
    [[ -f "$APK" ]] || die "no signed APK at $APK (did gradle produce *-unsigned.apk?)"
fi
[[ -f "$APK" ]] || die "APK not found: $APK"

ANDROID_HOME=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}
[[ -n "$ANDROID_HOME" ]] || die "ANDROID_HOME is not set"
BUILD_TOOLS=$(ls -d "$ANDROID_HOME"/build-tools/*/ 2>/dev/null \
    | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
[[ -n "$BUILD_TOOLS" ]] || die "no build-tools under $ANDROID_HOME"
AAPT2="${BUILD_TOOLS}aapt2"
APKSIGNER="${BUILD_TOOLS}apksigner"

badging=$("$AAPT2" dump badging "$APK")
apk_code=$(sed -nE "s/.*versionCode='([0-9]+)'.*/\1/p" <<<"$badging" | head -1)
apk_name=$(sed -nE "s/.*versionName='([^']+)'.*/\1/p" <<<"$badging" | head -1)
[[ "$apk_code" == "$VERSION_CODE" ]] || die "APK versionCode $apk_code != gradle $VERSION_CODE"
[[ "$apk_name" == "$VERSION_NAME" ]] || die "APK versionName $apk_name != gradle $VERSION_NAME"

"$APKSIGNER" verify --verbose "$APK" >/dev/null 2>&1 || die "APK signature invalid"
verify_out=$("$APKSIGNER" verify --verbose "$APK" 2>/dev/null)
grep -q 'v2 scheme (APK Signature Scheme v2): true' <<<"$verify_out" || die "APK is not v2-signed"
grep -q 'v3 scheme (APK Signature Scheme v3): true' <<<"$verify_out" || die "APK is not v3-signed"

# A valid signature is not enough: it must be the key users already have,
# otherwise every installed client rejects the update.
if [[ -n "$CERT_SHA256" ]]; then
    apk_cert=$("$APKSIGNER" verify --print-certs "$APK" 2>/dev/null \
        | sed -n 's/^Signer #1 certificate SHA-256 digest: //p' | head -1)
    [[ "$apk_cert" == "$CERT_SHA256" ]] \
        || die "APK signer $apk_cert != pinned $CERT_SHA256 (wrong keystore?)"
fi

mkdir -p "$DIST"
DIST_APK="$DIST/MobiInfer-Android-v$VERSION_NAME.apk"
cp -f "$APK" "$DIST_APK"
sha=$(sha256_of "$DIST_APK")
size=$(size_of "$DIST_APK")
( cd "$DIST" && printf '%s  %s\n' "$sha" "$(basename "$DIST_APK")" > SHA256SUMS )
log "APK $DIST_APK ($size bytes, sha256 $sha)"

# ---------------------------------------------------------------- tag + GitCode

if (( DO_PUSH )); then
    command -v gc >/dev/null || die "gc (GitCode CLI) not on PATH"

    if git -C "$ROOT" rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
        tagged=$(git -C "$ROOT" rev-list -n1 "$TAG")
        [[ "$tagged" == "$(git -C "$ROOT" rev-parse HEAD)" ]] \
            || die "tag $TAG already exists on another commit"
    else
        git -C "$ROOT" tag -a "$TAG" -m "$TAG"
    fi

    log "pushing $TAG to $GITCODE_REMOTE"
    git -C "$ROOT" push "$GITCODE_REMOTE" "refs/tags/$TAG"

    if [[ -z "$NOTES_FILE" ]]; then
        NOTES_FILE="$DIST/release-notes-$TAG.md"
        {
            printf '## MobiInfer-Android %s\n\n' "$TAG"
            printf -- '- versionName `%s` (versionCode %s)\n' "$VERSION_NAME" "$VERSION_CODE"
            printf -- '- 安装包：`MobiInfer-Android-v%s.apk`\n' "$VERSION_NAME"
            printf -- '- 校验：见 `SHA256SUMS`\n'
        } > "$NOTES_FILE"
    fi

    log "publishing GitCode release $TAG"
    if gc release view "$TAG" -R "$GITCODE_REPO" >/dev/null 2>&1; then
        asset_list=$(gc release view "$TAG" -R "$GITCODE_REPO" --json 2>/dev/null \
            | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
assets = data.get("assets", []) if isinstance(data, dict) else []
for a in assets:
    if a.get("id"):
        print(a.get("id"), a.get("name"))
' || true)

        if grep -q " $(basename "$DIST_APK")\$" <<<"$asset_list"; then
            if (( ! REPLACE )); then
                die "release $TAG already has $(basename "$DIST_APK"); bump versionName or pass --replace"
            fi
            log "replacing published assets (--replace)"
            while read -r asset_id asset_name; do
                [[ -n "$asset_id" ]] || continue
                case "$asset_name" in
                    MobiInfer-Android-*.apk|SHA256SUMS)
                        gc release delete-asset "$TAG" "$asset_id" -R "$GITCODE_REPO" --yes >/dev/null
                        ;;
                esac
            done <<<"$asset_list"
        fi
    else
        create_args=(release create "$TAG" -R "$GITCODE_REPO" --title "$TAG" --notes-file "$NOTES_FILE")
        (( PRERELEASE )) && create_args+=(--prerelease)
        gc "${create_args[@]}"
    fi

    gc release upload "$TAG" "$DIST_APK" "$DIST/SHA256SUMS" -R "$GITCODE_REPO"
fi

# ---------------------------------------------------------------- debug channel

DEBUG_SHA=""
DEBUG_SIZE=""
if (( DO_DEBUG )); then
    log "building debug channel APK"
    ( cd "$APP_DIR" && ./gradlew :app:assembleDebug )

    src_debug="$APP_DIR/app/build/outputs/apk/debug/app-debug.apk"
    [[ -f "$src_debug" ]] || die "no debug APK at $src_debug"

    debug_badging=$("$AAPT2" dump badging "$src_debug")
    debug_code=$(sed -nE "s/.*versionCode='([0-9]+)'.*/\1/p" <<<"$debug_badging" | head -1)
    debug_name=$(sed -nE "s/.*versionName='([^']+)'.*/\1/p" <<<"$debug_badging" | head -1)
    [[ "$debug_code" == "$VERSION_CODE" ]] \
        || die "debug APK versionCode $debug_code != $VERSION_CODE"
    [[ "$debug_name" == "$VERSION_NAME-debug" ]] \
        || die "debug APK versionName $debug_name != $VERSION_NAME-debug"

    DEBUG_APK="$DIST/MobiInfer-Android-v$VERSION_NAME-debug.apk"
    cp -f "$src_debug" "$DEBUG_APK"
    DEBUG_SHA=$(sha256_of "$DEBUG_APK")
    DEBUG_SIZE=$(size_of "$DEBUG_APK")
    log "debug APK $DEBUG_APK ($DEBUG_SIZE bytes, sha256 $DEBUG_SHA)"

    if (( DO_PUSH )); then
        gc release upload "$TAG" "$DEBUG_APK" -R "$GITCODE_REPO"
    fi
    if (( DO_CHENGDU )); then
        scp -F "${HOME}/.ssh/config" -o BatchMode=yes \
            "$DEBUG_APK" "$CHENGDU_HOST:$CHENGDU_WEBROOT/apk/"
    fi
fi

# ---------------------------------------------------------------- update.json

GITCODE_URL=""
[[ -n "$GITCODE_REPO" ]] && GITCODE_URL="https://gitcode.com/$GITCODE_REPO/releases/download/$TAG/MobiInfer-Android-v$VERSION_NAME.apk"
CHENGDU_URL="$PUBLIC_BASE/apk/MobiInfer-Android-v$VERSION_NAME.apk"
GITHUB_URL=""
[[ -n "$GITHUB_REPO" ]] && GITHUB_URL="https://github.com/$GITHUB_REPO/releases/download/$TAG/MobiInfer-Android-v$VERSION_NAME.apk"
GITCODE_DEBUG_URL=""
[[ -n "$GITCODE_REPO" ]] && GITCODE_DEBUG_URL="https://gitcode.com/$GITCODE_REPO/releases/download/$TAG/MobiInfer-Android-v$VERSION_NAME-debug.apk"
CHENGDU_DEBUG_URL="$PUBLIC_BASE/apk/MobiInfer-Android-v$VERSION_NAME-debug.apk"

NOTES_ZH="MobiInfer-Android $TAG"
if [[ -n "$NOTES_FILE" && -f "$NOTES_FILE" ]]; then
    NOTES_ZH=$(sed '1{/^## /d;}' "$NOTES_FILE" | sed '/./,$!d')
fi

UPDATE_JSON="$DIST/update.json"
VERSION_CODE="$VERSION_CODE" VERSION_NAME="$VERSION_NAME" MIN_SUPPORTED="$MIN_SUPPORTED" \
SHA256="$sha" SIZE="$size" NOTES_ZH="$NOTES_ZH" \
GITCODE_URL="$GITCODE_URL" CHENGDU_URL="$CHENGDU_URL" GITHUB_URL="$GITHUB_URL" \
DEBUG_ENABLED="$DO_DEBUG" DEBUG_SHA256="$DEBUG_SHA" DEBUG_SIZE="$DEBUG_SIZE" \
GITCODE_DEBUG_URL="$GITCODE_DEBUG_URL" CHENGDU_DEBUG_URL="$CHENGDU_DEBUG_URL" \
python3 - > "$UPDATE_JSON" <<'PY'
import json, os, sys
from datetime import datetime, timezone

env = os.environ
doc = {
    "versionCode": int(env["VERSION_CODE"]),
    "versionName": env["VERSION_NAME"],
    "minSupportedVersionCode": int(env["MIN_SUPPORTED"]),
    "apkUrls": [u for u in (env["GITCODE_URL"], env["CHENGDU_URL"], env["GITHUB_URL"]) if u],
    "sha256": env["SHA256"],
    "sizeBytes": int(env["SIZE"]),
    "publishedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "notes": {"zh": env["NOTES_ZH"], "en": env["NOTES_ZH"]},
}
if env.get("DEBUG_ENABLED") == "1":
    doc["debug"] = {
        "versionCode": int(env["VERSION_CODE"]),
        "versionName": env["VERSION_NAME"] + "-debug",
        "apkUrls": [u for u in (env["GITCODE_DEBUG_URL"], env["CHENGDU_DEBUG_URL"]) if u],
        "sha256": env["DEBUG_SHA256"],
        "sizeBytes": int(env["DEBUG_SIZE"]),
    }
json.dump(doc, sys.stdout, ensure_ascii=False, indent=2)
print()
PY

log "update.json"
cat "$UPDATE_JSON"

# ---------------------------------------------------------------- Chengdu

if (( DO_CHENGDU )); then
    log "deploying to $CHENGDU_HOST"
    scp_opts=(-F "${HOME}/.ssh/config" -o BatchMode=yes)
    ssh "${scp_opts[@]}" "$CHENGDU_HOST" "mkdir -p '$CHENGDU_WEBROOT/apk'"
    scp "${scp_opts[@]}" "$DIST_APK" "$DIST/SHA256SUMS" \
        "$CHENGDU_HOST:$CHENGDU_WEBROOT/apk/"
    scp "${scp_opts[@]}" "$UPDATE_JSON" "$CHENGDU_HOST:$CHENGDU_WEBROOT/update.json"

    remote_sha=$(ssh "${scp_opts[@]}" "$CHENGDU_HOST" \
        "sha256sum '$CHENGDU_WEBROOT/apk/$(basename "$DIST_APK")'" | cut -d' ' -f1)
    [[ "$remote_sha" == "$sha" ]] || die "uploaded APK hash mismatch on $CHENGDU_HOST"
    log "Chengdu hash verified"
fi

# ---------------------------------------------------------------- verify public URLs

if (( VERIFY )); then
    log "verifying public URLs"
    verify_sha() {
        local url=$1 expected=${2:-$sha} tmp got
        tmp=$(mktemp)
        if ! curl -fsSL --max-time 300 -o "$tmp" "$url"; then
            rm -f "$tmp"
            die "cannot download $url"
        fi
        got=$(sha256_of "$tmp")
        rm -f "$tmp"
        [[ "$got" == "$expected" ]] || die "sha256 mismatch: $url"
        log "ok $url"
    }

    if (( DO_PUSH )) && [[ -n "$GITCODE_URL" ]]; then
        verify_sha "$GITCODE_URL"
        if (( DO_DEBUG )); then
            verify_sha "$GITCODE_DEBUG_URL" "$DEBUG_SHA"
        fi
    fi
    if (( DO_CHENGDU )); then
        verify_sha "$CHENGDU_URL"
        if (( DO_DEBUG )); then
            verify_sha "$CHENGDU_DEBUG_URL" "$DEBUG_SHA"
        fi
        manifest_code=$(curl -fsS --max-time 30 "$PUBLIC_BASE/update.json" \
            | python3 -c 'import json, sys; print(json.load(sys.stdin)["versionCode"])')
        [[ "$manifest_code" == "$VERSION_CODE" ]] \
            || die "update.json advertises $manifest_code, expected $VERSION_CODE"
    fi

    if [[ -n "$GITHUB_URL" ]] && (( DO_PUSH || DO_CHENGDU )) \
        && ! curl -fsSIL --max-time 30 "$GITHUB_URL" >/dev/null 2>&1; then
        log "note: GitHub fallback not published yet ($GITHUB_URL)"
    fi
fi

log "done"
printf '  APK      %s\n' "$GITCODE_URL"
printf '  manifest %s/update.json\n' "$PUBLIC_BASE"
printf '  local    %s\n' "$DIST_APK"
