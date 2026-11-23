#!/usr/bin/env python3
"""Unit tests for bench/dropcache.py (stdlib only, no phone / no models).

Run:  python3 bench/test_dropcache.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().with_name("dropcache.py")

# A sitecustomize that records every posix_fadvise() call.  It lets the tests
# assert "validated before drop" / "print-scope does not drop" without relying
# on page-cache observability (tmpfs may treat DONTNEED as a no-op).
SITECUSTOMIZE = r'''
import os
_orig = os.posix_fadvise
def _spy(fd, offset, length, advice):
    path = os.environ.get("FADV_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("%d %d %d\n" % (fd, offset, length))
    return _orig(fd, offset, length, advice)
os.posix_fadvise = _spy
'''


class DropcacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.spy = self.tmp / "spy"
        self.spy.mkdir()
        (self.spy / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
        self.fadv_log = self.tmp / "fadv.log"

    def tearDown(self) -> None:
        self._td.cleanup()

    # -- helpers --------------------------------------------------------
    def env(self, extra: dict | None = None) -> dict:
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODEL", "EXPERTS", "DRAFT", "PYTHONPATH", "FADV_LOG")}
        env["HOME"] = str(self.home)
        env["PYTHONPATH"] = str(self.spy)
        env["FADV_LOG"] = str(self.fadv_log)
        if extra:
            env.update(extra)
        return env

    def run_dc(self, *args: str, cwd: pathlib.Path | None = None,
               env_extra: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=str(cwd or self.tmp), env=self.env(env_extra),
            capture_output=True, text=True, timeout=60,
        )

    def write(self, rel: str, size: int) -> pathlib.Path:
        path = self.tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def fadv_calls(self) -> int:
        if not self.fadv_log.exists():
            return 0
        return len([ln for ln in self.fadv_log.read_text(encoding="utf-8").splitlines() if ln])

    def scope(self, *args: str, cwd: pathlib.Path | None = None,
              env_extra: dict | None = None) -> dict:
        cp = self.run_dc(*args, "--print-scope", cwd=cwd, env_extra=env_extra)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        return json.loads(cp.stdout)

    # 1 ---------------------------------------------------------------
    def test_01_no_args_missing_default_root_is_clean(self) -> None:
        cp = self.run_dc()
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(cp.stderr, "")
        self.assertRegex(cp.stdout.strip(), r"^dropped 0 files, 0\.00 GB$")

    # 2 ---------------------------------------------------------------
    def test_02_explicit_three_categories_and_bytes(self) -> None:
        self.write("m/a.gguf", 100)
        self.write("e/layer_000.bin", 200)
        self.write("e/layer_001.bin", 300)
        self.write("e/manifest.json", 50)
        self.write("d/d.gguf", 400)
        doc = self.scope("--model", str(self.tmp / "m/a.gguf"),
                         "--experts", str(self.tmp / "e"),
                         "--draft", str(self.tmp / "d/d.gguf"))
        self.assertEqual(len(doc["model"]), 1)
        self.assertEqual(len(doc["experts"]), 3)
        self.assertEqual(len(doc["draft"]), 1)
        self.assertEqual(doc["total_files"], 5)
        self.assertEqual(doc["total_bytes"], 100 + 200 + 300 + 50 + 400)
        by_path = {f["path"]: f["bytes"] for f in doc["files"]}
        self.assertEqual(by_path[os.path.realpath(self.tmp / "m/a.gguf")], 100)
        self.assertFalse(doc["dropped"])

    # 3 ---------------------------------------------------------------
    def test_03_experts_only_bins_and_manifest(self) -> None:
        self.write("e/layer_000.bin", 10)
        self.write("e/manifest.json", 5)
        self.write("e/ignored.tmp", 7)
        self.write("e/sub/nested.bin", 9)
        doc = self.scope("--experts", str(self.tmp / "e"))
        names = sorted(os.path.basename(p) for p in doc["experts"])
        self.assertEqual(names, ["layer_000.bin", "manifest.json"])
        self.assertEqual(doc["total_files"], 2)
        self.assertEqual(doc["total_bytes"], 15)

    # 4 ---------------------------------------------------------------
    def test_04_print_scope_does_not_drop(self) -> None:
        self.write("m/a.gguf", 10)
        doc = self.scope("--model", str(self.tmp / "m/a.gguf"))
        self.assertFalse(doc["dropped"])
        self.assertNotIn("dropped_files", doc)
        self.assertEqual(self.fadv_calls(), 0)

    # 5 ---------------------------------------------------------------
    def test_05_strict_missing_model_drops_nothing(self) -> None:
        self.write("e/layer_000.bin", 10)
        cp = self.run_dc("--model", str(self.tmp / "m/missing.gguf"),
                         "--experts", str(self.tmp / "e"), "--strict")
        self.assertEqual(cp.returncode, 2, cp.stderr)
        self.assertIn("strict", cp.stderr)
        self.assertEqual(self.fadv_calls(), 0)

    # 6 ---------------------------------------------------------------
    def test_06_non_strict_missing_model_still_drops_experts(self) -> None:
        self.write("e/layer_000.bin", 10)
        cp = self.run_dc("--model", str(self.tmp / "m/missing.gguf"),
                         "--experts", str(self.tmp / "e"))
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn("warning", cp.stderr)
        self.assertEqual(self.fadv_calls(), 1)
        self.assertRegex(cp.stdout.strip(), r"^dropped 1 files, 0\.00 GB$")

    # 7 ---------------------------------------------------------------
    def test_07_max_gib_blocks_then_force_allows(self) -> None:
        self.write("e/layer_000.bin", 4096)
        cp = self.run_dc("--experts", str(self.tmp / "e"), "--max-gib", "0.000001")
        self.assertEqual(cp.returncode, 2, cp.stderr)
        self.assertIn("max-gib", cp.stderr)
        self.assertEqual(self.fadv_calls(), 0)
        cp = self.run_dc("--experts", str(self.tmp / "e"),
                         "--max-gib", "0.000001", "--force")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(self.fadv_calls(), 1)

    # 8 ---------------------------------------------------------------
    def test_08_realpath_dedup_relative_absolute_glob(self) -> None:
        self.write("a.gguf", 10)
        rel = "a.gguf"
        absolute = str(self.tmp / "a.gguf")
        doc = self.scope(rel, absolute, "*.gguf")
        self.assertEqual(doc["total_files"], 1)
        self.assertEqual(doc["total_bytes"], 10)

    # 9 ---------------------------------------------------------------
    def test_09_refuses_dangerous_roots(self) -> None:
        (self.home / "csm" / "moe-exp").mkdir(parents=True)
        for target in (str(self.home / "csm" / "moe-exp"), "/"):
            cp = self.run_dc("--experts", target, "--print-scope")
            self.assertEqual(cp.returncode, 2, cp.stderr)
            self.assertIn("refusing dangerous target", cp.stderr)
        self.assertEqual(self.fadv_calls(), 0)

    # 10 --------------------------------------------------------------
    def test_10_json_is_parseable_with_required_fields(self) -> None:
        self.write("m/a.gguf", 10)
        self.write("e/layer_000.bin", 20)
        cp = self.run_dc("--model", str(self.tmp / "m/a.gguf"),
                         "--experts", str(self.tmp / "e"), "--json")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        doc = json.loads(cp.stdout)
        for key in ("version", "root", "model", "experts", "draft", "files",
                    "total_files", "total_bytes", "strict", "dropped"):
            self.assertIn(key, doc)
        self.assertEqual(doc["version"], 1)
        self.assertTrue(doc["dropped"])
        self.assertEqual(doc["dropped_files"], 2)
        self.assertEqual(doc["total_files"], 2)
        self.assertEqual(doc["total_bytes"], 30)

    # 11 --------------------------------------------------------------
    def test_11_env_scope(self) -> None:
        self.write("m/a.gguf", 10)
        self.write("e/layer_000.bin", 20)
        self.write("d/d.gguf", 30)
        doc = self.scope(env_extra={
            "MODEL": str(self.tmp / "m/a.gguf"),
            "EXPERTS": str(self.tmp / "e"),
            "DRAFT": str(self.tmp / "d/d.gguf"),
        })
        self.assertEqual(len(doc["model"]), 1)
        self.assertEqual(len(doc["experts"]), 1)
        self.assertEqual(len(doc["draft"]), 1)
        self.assertEqual(doc["total_bytes"], 60)

    # 12 --------------------------------------------------------------
    def test_12_legacy_default_scope_still_works(self) -> None:
        root = self.home / "csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon"
        self.write(str(root / "base.gguf"), 10)
        self.write(str(root / "experts-htp-tiled/layer_000.bin"), 20)
        self.write(str(root / "experts-q1-30tps/layer_000.bin"), 40)
        cp = self.run_dc("--json")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        doc = json.loads(cp.stdout)
        self.assertEqual(doc["total_files"], 2)
        self.assertEqual(doc["total_bytes"], 30)
        self.assertEqual(doc["root"], str(root))
        self.assertNotIn(str(root / "experts-q1-30tps/layer_000.bin"),
                         [f["path"] for f in doc["files"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
