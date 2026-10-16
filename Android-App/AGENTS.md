# AGENTS.md — Working with the MobiInfer Android app

> Conventions for AI coding agents (Cursor / Claude Code / Codex / etc.) working on this
> repository. Read this once before making your first commit or PR. Human contributors are
> welcome to follow the same conventions.

---

## 0. What this repo is

The on-device **Android** demo for the MiniCPM-V family of multimodal LLMs. It runs fully
on-device via a shared inference engine kept at the top of the **enclosing repository**
(`MobiInfer-Android`). This `Android-App/` directory is a sub-directory of that repository —
there is **no `llama.cpp-omni` submodule and no symlink**.

This repository holds **app source code + docs only**. Model weights are distributed via
HuggingFace / ModelScope / OBS and are **not** stored here. Compiled artefacts (Gradle build
outputs, IDE caches) are intentionally **not tracked** — they are produced locally as part of
the build (see §3 and the README).

---

## 1. Repository layout

```
MobiInfer-Android/
├── ggml/ src/ common/ tools/ ...   inference engine (shared at repo root)
├── Android-App/
│   ├── MobiInfer-Android/           Android Kotlin + NDK sources
│   ├── docs/                        app-level benchmark data
│   ├── assets/                      benchmark assets
│   └── README{,_zh}.md              setup walkthroughs (read these first)
└── DEPLOY.md / README.md            deployment guide + engine overview
```

On first entry, the agent should `Read README.md` (or `README_zh.md`). It contains the
"install Android Studio → first-time build → open & run" walkthrough. The engine build is
reached by relative path from the client (`-DLLAMA_SRC`).

---

## 2. The inference engine (at the repository root)

The engine is **not** a git submodule here. It is the git repository that contains this
`Android-App/` directory. The client locates it by relative path:

- Android: `CMakeLists.txt` sets `LLAMA_SRC` six levels up (repo root); override
  with `-DLLAMA_SRC=<path>`.

**Agent notes**

- ❌ Do **not** re-introduce a `llama.cpp-omni` git submodule or symlink under `Android-App/`
  — a symlink to an ancestor (`..`) makes recursive tools (`find`, `grep`, `git add`) loop.
- ✅ To point the client at an engine kept elsewhere, pass `-DLLAMA_SRC=<path>` (CMake)
  instead of creating a symlink.

---

## 3. What is *not* in the repo (and must stay out)

Already covered by `.gitignore`. Agents must double-check the staging list before each
`git commit` and `git restore --staged` anything that slipped in:

| Category | Examples |
|---|---|
| Android build / cache | `MobiInfer-Android/{app/,}build/`, `.cxx/`, `.gradle/`, `app/src/main/jniLibs/`, `local.properties` |
| IDE per-user state | `MobiInfer-Android/.idea/`, any `*.iml` |
| AI agent local notes | `CLAUDE.md`, `.claude/`, `.cursor/` (this file `AGENTS.md` **is** tracked) |

---

## 4. Building from source

The full walkthrough lives in `README.md` / `README_zh.md`. Agents touching build
configuration should re-read the relevant section first.

```bash
cd MobiInfer-Android
./gradlew assembleDebug         # quick local test
./gradlew :app:assembleRelease  # release artefact (signed via keystore in ~/.gradle/gradle.properties)
```

JBR used for the release build: `export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`.

---

## 5. Release artefact naming (strict)

Every release goes out under a fixed name so end users can tell APKs apart at a glance.
**No three-segment versions, no commit hashes, no `-release` / `-signed` suffixes in
user-facing filenames.**

- Fields: `versionCode` + `versionName` in `MobiInfer-Android/app/build.gradle.kts`.
- `versionCode` is **monotonically increasing**, +1 per release, never reused.
- **Release APK filename (hard rule)**:
  ```
  MobiInfer-Android-v<versionName>.apk
  ```
  - ✅ `MobiInfer-Android-v2.1.apk`
  - ❌ `MobiInfer-1.2.3.apk` (missing platform, three-segment version)
  - ❌ `MobiInfer-Android-release-<sha>.apk` (commit hash, internal-only)
- Post-build verification:
  - `aapt dump badging <apk> | grep ^package` — confirm `versionCode` / `versionName`
    match the `build.gradle.kts` you just edited.
  - `apksigner verify -v <apk>` — confirm v2 / v3 signature schemes are present.

---

## 6. Model consistency

When rotating a model or re-converting the mmproj, keep the client's MD5 constants in sync:

- Android: `MobiInfer-Android/app/src/main/java/com/mobiinfer/demo/ModelInfo.kt`
  (`ggufMd5` / `mmprojMd5`).

> **mmproj migration caveat**: this app's branch of llama.cpp-omni expects
> `clip.projector_type=merger`, which differs from the upstream `minicpmv4_6` type
> currently exported by the standard conversion path. When swapping mmproj layouts,
> rename the file on disk (so legacy on-device copies are not blindly reused) and add
> the old filename to the client's "stale layout" purge list. MD5 fast-paths alone are
> not sufficient — some load paths check existence first and skip the MD5 check.

---

## 7. Git commit & PR conventions

### Commit message

- **Single-line subject**, no body. Fold scope + summary + version bumps into one sentence.
  If something genuinely needs prose, put it in the PR description, not the commit body.
- **No AI / tool attribution** — no `Co-authored-by: Cursor <…>`, no `Generated with
  Claude`, no `🤖 …` watermark, no `--trailer` variants. Same for code comments.
- **No emoji.**
- English subject with optional Chinese scope is fine:
  - ✅ `android: drop hardcoded English sys prompt`
  - ✅ `android: bump app version 2.1 -> 2.2 (slimmer download)`
  - ❌ `feat: ...\n\n* bullet 1\n* bullet 2\nCo-authored-by: Cursor <…>`

### Force push

- `main` is **not** force-pushed routinely. The only legitimate force-push to `main` is
  history-rewriting cleanup (e.g. `git filter-repo`-driven removal of accidentally
  committed binaries), and only with explicit user authorization, using
  `git push --force-with-lease` (never `--force`).
- Feature branches may be force-pushed after user confirmation.

### PRs

- Target `IPADS-SAI/Mobiinfer-Android:main`.
- PR description should cover: motivation, files touched, whether the reviewer needs to
  re-pull a model, and any user-visible behaviour change.
- Reference open issues with `Fixes #N` so GitHub auto-links and auto-closes them.

---

## 8. Common pitfalls

| Symptom | Real cause / fix |
|---|---|
| CMake: `Inference engine (llama.cpp-omni) not found at ...` | The client was built outside the MobiInfer-Android tree. Build from within the repo, or pass `-DLLAMA_SRC=<path>`. |
| Android `INSTALL_FAILED_USER_RESTRICTED` | Xiaomi / HyperOS disables USB-side installation by default; enable both "USB debugging (security settings)" and "Install via USB" in Developer Options. |
| Android `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | Previously installed APK was signed with a different key. `adb uninstall <pkg>` then reinstall. |

---

## 9. Do-not list (recap)

- ❌ Commit any compiled artefact: `build/`, `.cxx/`, `.idea/`, `app/src/main/jniLibs/`, …
- ❌ Commit any file containing a local absolute path or credential: `local.properties`,
  `gradle.properties` with `signing.*`, keystore files, `*.p12`.
- ❌ Commit message with AI / tool attribution trailers.
- ❌ Re-introduce a `llama.cpp-omni` git submodule or an ancestor-pointing symlink under
  `Android-App/` (it creates a recursive tree for `find` / `grep` / `git add`).
- ❌ Decorate release artefact filenames with three-segment versions, commit hashes,
  `-release`, `-signed`, dates, etc.

---

## 10. Out of scope

For repository hygiene, the following are deliberately **not** documented here:

- GitHub PATs, API tokens, keystore passwords, OBS / HF / MS credentials.
- Internal maintainer SOPs and cross-repo workflows.
- Absolute paths / shell aliases on any specific developer's machine.

If an agent needs any of those (e.g. to comment on an issue as a maintainer, or to push
a release), it should ask the user rather than scanning the local filesystem for keys.
