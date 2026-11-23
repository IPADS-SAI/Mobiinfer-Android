#!/usr/bin/env python3
"""dropcache.py — 用 posix_fadvise(DONTNEED) 丢弃模型/专家文件的 page cache（无需 root）。

用法：
  python3 dropcache.py [--model FILE] [--experts DIR] [--draft FILE]
                       [--print-scope] [--json] [--strict]
                       [--max-gib N] [--force] [--allow-outside] [PATH ...]

scope 来源（优先级从高到低）：
  --model/--experts/--draft 显式参数
  MODEL/EXPERTS/DRAFT 环境变量（bench.sh 用同名变量，但本脚本只读 env）
  位置参数 PATH（兼容旧用法）
  以上都没有时：旧默认 root 下的 *.gguf + experts-htp-tiled/*.bin

关键语义：
  * ``--experts DIR`` 只展开 ``DIR/*.bin`` + ``DIR/manifest.json``（若存在），
    不递归、不碰 ``*.tmp`` 或子目录。
  * ``--model``/``--draft`` 接受单个文件或 glob（如 ``dir/*.gguf``）。
  * ``--print-scope`` 只打印 scope JSON，不 drop。
  * ``--json`` 在 drop 后打印同一份 scope JSON（含 ``dropped``/``dropped_files``）。
  * ``--strict``：任一显式声明的路径不存在/不可读/展开为空 → 退出 2，且一个文件都不 drop
    （先解析、校验，再统一 drop）。
  * ``--max-gib N``：解析后总字节 > N GiB 且没有 ``--force`` → 退出 2，不 drop。
  * 拒绝把 ``/``、``$HOME``、``$HOME/csm/moe-exp`` 本身作为 target（防手滑）；
    ``--allow-outside`` 仅供显式调试时放开。
  * realpath 后去重；默认不递归整个 models/，也不改 bench.sh/sweep.sh 的任何默认值。

无参数且无 env 时保持旧行为：默认 root 为
``~/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon``，
只打印一行 ``dropped N files, X GB``。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

DEFAULT_ROOT = "~/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon"
DEFAULT_EXPERTS_SUBDIR = "experts-htp-tiled"
GIB = 1024.0 ** 3

_KINDS = ("model", "experts", "draft", "extra")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def has_glob(path: str) -> bool:
    return any(c in path for c in "*?[")


def expand(spec: str) -> str:
    return os.path.abspath(os.path.expanduser(spec))


def dangerous_roots() -> set[str]:
    home = os.path.expanduser("~")
    return {
        os.path.realpath("/"),
        os.path.realpath(home),
        os.path.realpath(os.path.join(home, "csm", "moe-exp")),
    }


def is_dangerous_target(spec: str) -> bool:
    # Compare the normalized target itself.  A glob like
    # ``$HOME/csm/moe-exp/*`` is not the directory itself and is allowed.
    return os.path.realpath(expand(spec)) in dangerous_roots()


class Scope:
    """Resolved drop scope.  Resolution never drops anything."""

    def __init__(self) -> None:
        self.paths: dict[str, str] = {kind: [] for kind in _KINDS}
        self.entries: dict[str, int] = {}       # realpath -> bytes
        self.missing: list[str] = []
        self.dropped_files = 0

    # -- resolution -----------------------------------------------------
    def _add_file(self, path: str, kind: str, declared: str) -> None:
        real = os.path.realpath(path)
        if not os.path.isfile(real):
            self.missing.append(f"{kind}:{declared} (not a regular file: {real})")
            warn(f"{kind} target is not a regular file: {real}")
            return
        try:
            fd = os.open(real, os.O_RDONLY)
            os.close(fd)
        except OSError as exc:
            self.missing.append(f"{kind}:{declared} (unreadable: {real}: {exc})")
            warn(f"{kind} target is unreadable: {real}: {exc}")
            return
        if real not in self.entries:
            try:
                self.entries[real] = os.path.getsize(real)
            except OSError as exc:
                self.missing.append(f"{kind}:{declared} (stat failed: {real}: {exc})")
                warn(f"cannot stat {real}: {exc}")
                return
        if real not in self.paths[kind]:
            self.paths[kind].append(real)

    def add_generic(self, spec: str, kind: str) -> None:
        expanded = expand(spec)
        if has_glob(expanded):
            matches = sorted(glob.glob(expanded))
        else:
            matches = [expanded] if os.path.lexists(expanded) else []
        if not matches:
            self.missing.append(f"{kind}:{spec} (no match)")
            warn(f"{kind} target not found: {spec}")
            return
        for match in matches:
            if os.path.isdir(match):
                self.missing.append(f"{kind}:{spec} (directory: {match})")
                warn(f"{kind} expects a file/glob, got a directory: {match}")
                continue
            self._add_file(match, kind, spec)

    def add_experts(self, spec: str) -> None:
        expanded = expand(spec)
        if has_glob(expanded):
            bases = sorted(glob.glob(expanded))
        else:
            bases = [expanded] if os.path.lexists(expanded) else []
        if not bases:
            self.missing.append(f"experts:{spec} (no match)")
            warn(f"experts target not found: {spec}")
            return
        for base in bases:
            if os.path.isdir(base):
                files = sorted(glob.glob(os.path.join(base, "*.bin")))
                manifest = os.path.join(base, "manifest.json")
                if os.path.isfile(manifest):
                    files.append(manifest)
                files = [f for f in files if os.path.isfile(f)]
                if not files:
                    self.missing.append(
                        f"experts:{spec} (no *.bin/manifest.json under {base})")
                    warn(f"experts dir has no *.bin/manifest.json: {base}")
                    continue
                for path in files:
                    self._add_file(path, "experts", spec)
            elif os.path.isfile(base):
                self._add_file(base, "experts", spec)
            else:
                self.missing.append(f"experts:{spec} (not found: {base})")
                warn(f"experts target not found: {base}")

    # -- reporting ------------------------------------------------------
    @property
    def total_bytes(self) -> int:
        return sum(self.entries.values())

    @property
    def total_files(self) -> int:
        return len(self.entries)

    def files(self) -> list[dict]:
        return [{"path": p, "bytes": self.entries[p]}
                for p in sorted(self.entries)]

    def root(self, legacy_root: str | None) -> str:
        if legacy_root is not None:
            return legacy_root
        dirs = []
        for path in self.entries:
            dirs.append(os.path.dirname(path))
        if dirs:
            try:
                return os.path.realpath(os.path.commonpath(dirs))
            except ValueError:
                pass
        return ""

    def as_dict(self, root: str, strict: bool, dropped: bool) -> dict:
        doc = {
            "version": 1,
            "root": root,
            "model": sorted(self.paths["model"]),
            "experts": sorted(self.paths["experts"]),
            "draft": sorted(self.paths["draft"]),
            "extra": sorted(self.paths["extra"]),
            "files": self.files(),
            "total_files": len(self.entries),
            "total_bytes": self.total_bytes,
            "strict": bool(strict),
            "dropped": bool(dropped),
        }
        if self.missing:
            doc["missing"] = self.missing
        if dropped:
            doc["dropped_files"] = self.dropped_files
        return doc

    # -- drop -----------------------------------------------------------
    def drop(self) -> int:
        n = 0
        for path in sorted(self.entries):
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError as exc:
                warn(f"skip {path}: {exc}")
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            except (AttributeError, OSError) as exc:
                warn(f"posix_fadvise failed for {path}: {exc}")
            finally:
                os.close(fd)
        self.dropped_files = n
        return n


def legacy_scope() -> Scope:
    """Old no-argument behaviour: one model root, *.gguf + experts-htp-tiled/*.bin."""
    scope = Scope()
    root = expand(DEFAULT_ROOT)
    for path in sorted(glob.glob(os.path.join(root, "*.gguf"))):
        if os.path.isfile(path):
            scope._add_file(path, "model", path)
    for path in sorted(glob.glob(os.path.join(
            root, DEFAULT_EXPERTS_SUBDIR, "*.bin"))):
        if os.path.isfile(path):
            scope._add_file(path, "experts", path)
    return scope


def build_scope(args: argparse.Namespace) -> tuple[Scope, str | None, bool]:
    """Return (scope, legacy_root_or_None, explicit_mode)."""
    model_specs = list(args.model or [])
    experts_specs = list(args.experts or [])
    draft_specs = list(args.draft or [])
    if not model_specs and os.environ.get("MODEL"):
        model_specs = [os.environ["MODEL"]]
    if not experts_specs and os.environ.get("EXPERTS"):
        experts_specs = [os.environ["EXPERTS"]]
    if not draft_specs and os.environ.get("DRAFT"):
        draft_specs = [os.environ["DRAFT"]]
    extra_specs = list(args.paths or [])

    explicit = bool(model_specs or experts_specs or draft_specs or extra_specs)
    if not explicit:
        return legacy_scope(), expand(DEFAULT_ROOT), False

    for spec in model_specs + experts_specs + draft_specs + extra_specs:
        if not args.allow_outside and is_dangerous_target(spec):
            err(f"refusing dangerous target '{spec}' "
                f"({os.path.realpath(expand(spec))}); use --allow-outside only for debugging")
            raise SystemExit(2)

    scope = Scope()
    for spec in model_specs:
        scope.add_generic(spec, "model")
    for spec in experts_specs:
        scope.add_experts(spec)
    for spec in draft_specs:
        scope.add_generic(spec, "draft")
    for spec in extra_specs:
        scope.add_generic(spec, "extra")
    return scope, None, True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", metavar="FILE", action="append",
                    help="target GGUF file or glob (repeatable; env: MODEL)")
    ap.add_argument("--experts", metavar="DIR", action="append",
                    help="expert dir: expands DIR/*.bin + DIR/manifest.json "
                         "(repeatable; env: EXPERTS)")
    ap.add_argument("--draft", metavar="FILE", action="append",
                    help="draft model file or glob (repeatable; env: DRAFT)")
    ap.add_argument("--print-scope", action="store_true",
                    help="print the resolved scope JSON and do not drop anything")
    ap.add_argument("--json", action="store_true",
                    help="after dropping, print the scope JSON (dropped=true)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 without dropping if any explicit target is missing/unreadable")
    ap.add_argument("--max-gib", type=float, default=None, metavar="N",
                    help="refuse (exit 2, no drop) if the resolved scope exceeds N GiB "
                         "unless --force is given")
    ap.add_argument("--force", action="store_true",
                    help="ignore --max-gib")
    ap.add_argument("--allow-outside", action="store_true",
                    help="allow /, $HOME or $HOME/csm/moe-exp themselves as targets "
                         "(debug only)")
    ap.add_argument("paths", nargs="*", metavar="PATH",
                    help="legacy positional file/glob targets (deduplicated)")
    args = ap.parse_args(argv)

    if args.max_gib is not None and args.max_gib < 0:
        err("--max-gib must be >= 0")
        return 2

    scope, legacy_root, _explicit = build_scope(args)
    root = scope.root(legacy_root)
    strict = bool(args.strict)

    # Validate first, then enforce the size budget, and only then drop: a
    # strict or max-gib failure must never leave a half-cold cache behind.
    if strict and scope.missing:
        for item in scope.missing:
            err(f"strict: missing {item}")
        err("strict: refusing to drop any file (scope validation failed)")
        return 2
    if args.max_gib is not None and not args.force \
            and scope.total_bytes > args.max_gib * GIB:
        err(f"scope is {scope.total_bytes / GIB:.2f} GiB > --max-gib {args.max_gib:g} "
            f"({scope.total_files} files); refusing to drop (use --force to override)")
        return 2

    if args.print_scope:
        print(json.dumps(scope.as_dict(root, strict, False),
                         ensure_ascii=False, sort_keys=True))
        return 0

    n = scope.drop()
    if args.json:
        print(json.dumps(scope.as_dict(root, strict, True),
                         ensure_ascii=False, sort_keys=True))
    else:
        print(f"dropped {n} files, {scope.total_bytes / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
