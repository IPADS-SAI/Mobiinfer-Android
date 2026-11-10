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
# The App reads update.json from https://mobiinfer.yiguzhaojin.cn/update.json
# at most once a day and installs the first mirror whose bytes match sha256
# and whose signer matches the installed app.
#
# Usage:
#   scripts/release-android.sh --tag v0.4.2 [options]
#
# Options:
#   --tag <tag>            Release tag (default: v<versionName>)
#   --apk <path>           Use an existing APK instead of building
#   --notes-file <path>    Release notes (default: generated one-liner)
#   --min-supported <code> versionCode below which the update is mandatory
#   --prerelease           Mark the GitCode release as a pre-release
#   --no-build             Do not run gradle (requires --apk or a fresh build)
#   --no-push              Do not create/push the tag or touch GitCode
#   --no-chengdu           Do not upload the APK / update.json to Chengdu
#
# Environment overrides:
#   GITCODE_REPO, GITCODE_REMOTE, CHENGDU_HOST, CHENGDU_WEBROOT, PUBLIC_BASE,
#   GITHUB_REPO, ANDROID_HOME
#
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
APP_DIR="$ROOT/Android-App/MobiInfer-Android"
GRADLE_FILE="$APP_DIR/app/build.gradle.kts"
DIST="$APP_DIR/app/build/release-dist"

GITCODE_REPO=${GITCODE_REPO:-YYDH2121/MobiInfer-Android}
GITCODE_REMOTE=${GITCODE_REMOTE:-gitcode}
GITHUB_REPO=${GITHUB_REPO:-IPADS-SAI/Mobiinfer-Android}
CHENGDU_HOST=${CHENGDU_HOST:-chengdu}
CHENGDU_WEBROOT=${CHENGDU_WEBROOT:-/var/www/mobiinfer}
PUBLIC_BASE=${PUBLIC_BASE:-https://mobiinfer.yiguzhaojin.cn}

TAG=""
APK=""
NOTES_FILE=""
MIN_SUPPORTED=0
DO_BUILD=1
DO_PUSH=1
DO_CHENGDU=1
PRERELEASE=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
log() { printf '\n== %s\n' "$*"; }

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)           TAG=${2:?}; shift 2 ;;
        --apk)           APK=${2:?}; DO_BUILD=0; shift 2 ;;
        --notes-file)    NOTES_FILE=${2:?}; shift 2 ;;
        --min-supported) MIN_SUPPORTED=${2:?}; shift 2 ;;
        --prerelease)    PRERELEASE=1; shift ;;
        --no-build)      DO_BUILD=0; shift ;;
        --no-push)       DO_PUSH=0; shift ;;
        --no-chengdu)    DO_CHENGDU=0; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "unknown option: $1 (try --help)" ;;
    esac
done

[[ -f "$GRADLE_FILE" ]] || die "missing $GRADLE_FILE"

VERSION_CODE=$(sed -nE 's/^[[:space:]]*versionCode = ([0-9]+).*/\1/p' "$GRADLE_FILE" | head -1)
VERSION_NAME=$(sed -nE 's/^[[:space:]]*versionName = "([^"]+)".*/\1/p' "$GRADLE_FILE" | head -1)
[[ -n "$VERSION_CODE" && -n "$VERSION_NAME" ]] || die "cannot read version from $GRADLE_FILE"
TAG=${TAG:-v$VERSION_NAME}

log "release $TAG ($VERSION_NAME, versionCode $VERSION_CODE)"

# ---------------------------------------------------------------- build / verify

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
BUILD_TOOLS=$(ls -d "$ANDROID_HOME"/build-tools/*/ 2>/dev/null | sort -V | tail -1)
[[ -n "$BUILD_TOOLS" ]] || die "no build-tools under $ANDROID_HOME"
AAPT2="${BUILD_TOOLS}aapt2"
APKSIGNER="${BUILD_TOOLS}apksigner"

badging=$("$AAPT2" dump badging "$APK")
apk_code=$(sed -nE "s/.*versionCode='([0-9]+)'.*/\1/p" <<<"$badging" | head -1)
apk_name=$(sed -nE "s/.*versionName='([^']+)'.*/\1/p" <<<"$badging" | head -1)
[[ "$apk_code" == "$VERSION_CODE" ]] || die "APK versionCode $apk_code != gradle $VERSION_CODE"
[[ "$apk_name" == "$VERSION_NAME" ]] || die "APK versionName $apk_name != gradle $VERSION_NAME"

"$APKSIGNER" verify --verbose "$APK" >/dev/null 2>&1 || die "APK signature invalid"
if ! "$APKSIGNER" verify --verbose "$APK" 2>/dev/null | grep -q 'v2 scheme (APK Signature Scheme v2): true'; then
    die "APK is not v2-signed"
fi

mkdir -p "$DIST"
DIST_APK="$DIST/MobiInfer-Android-v$VERSION_NAME.apk"
cp -f "$APK" "$DIST_APK"
sha=$(sha256sum "$DIST_APK" | cut -d' ' -f1)
size=$(stat -c%s "$DIST_APK")
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
        # Replace assets with the same names so reruns are idempotent.
        gc release view "$TAG" -R "$GITCODE_REPO" --json 2>/dev/null \
            | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
assets = data.get("assets", data if isinstance(data, list) else [])
for a in assets:
    # Source archives have no id; only attached assets can be deleted.
    if a.get("id") and str(a.get("name", "")).endswith((".apk", "SHA256SUMS")):
        print(a.get("id"), a.get("name"))
' | while read -r asset_id asset_name; do
            [[ -n "$asset_id" ]] || continue
            case "$asset_name" in
                MobiInfer-Android-*.apk|SHA256SUMS)
                    log "removing stale asset $asset_name"
                    gc release delete-asset "$TAG" "$asset_id" -R "$GITCODE_REPO" --yes >/dev/null
                    ;;
            esac
        done
    else
        create_args=(release create "$TAG" -R "$GITCODE_REPO" --title "$TAG" --notes-file "$NOTES_FILE")
        (( PRERELEASE )) && create_args+=(--prerelease)
        gc "${create_args[@]}"
    fi

    gc release upload "$TAG" "$DIST_APK" "$DIST/SHA256SUMS" -R "$GITCODE_REPO"
fi

# ---------------------------------------------------------------- update.json

GITCODE_URL="https://gitcode.com/$GITCODE_REPO/releases/download/$TAG/MobiInfer-Android-v$VERSION_NAME.apk"
CHENGDU_URL="$PUBLIC_BASE/apk/MobiInfer-Android-v$VERSION_NAME.apk"
GITHUB_URL="https://github.com/$GITHUB_REPO/releases/download/$TAG/MobiInfer-Android-v$VERSION_NAME.apk"

NOTES_ZH="MobiInfer-Android $TAG"
if [[ -n "$NOTES_FILE" && -f "$NOTES_FILE" ]]; then
    NOTES_ZH=$(sed '1{/^## /d;}' "$NOTES_FILE" | sed '/./,$!d')
fi

UPDATE_JSON="$DIST/update.json"
VERSION_CODE="$VERSION_CODE" VERSION_NAME="$VERSION_NAME" MIN_SUPPORTED="$MIN_SUPPORTED" \
SHA256="$sha" SIZE="$size" NOTES_ZH="$NOTES_ZH" \
GITCODE_URL="$GITCODE_URL" CHENGDU_URL="$CHENGDU_URL" GITHUB_URL="$GITHUB_URL" \
python3 - > "$UPDATE_JSON" <<'PY'
import json, os, sys
from datetime import datetime, timezone

env = os.environ
doc = {
    "versionCode": int(env["VERSION_CODE"]),
    "versionName": env["VERSION_NAME"],
    "minSupportedVersionCode": int(env["MIN_SUPPORTED"]),
    "apkUrls": [env["GITCODE_URL"], env["CHENGDU_URL"], env["GITHUB_URL"]],
    "sha256": env["SHA256"],
    "sizeBytes": int(env["SIZE"]),
    "publishedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "notes": {"zh": env["NOTES_ZH"], "en": env["NOTES_ZH"]},
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

log "done"
printf '  APK      %s\n' "$GITCODE_URL"
printf '  manifest %s/update.json\n' "$PUBLIC_BASE"
printf '  local    %s\n' "$DIST_APK"
