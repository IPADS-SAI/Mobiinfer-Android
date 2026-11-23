#!/usr/bin/env python3
"""Unit tests for tools/moe/gate_diff.py (stdlib unittest, no device).

Run:
    python3 -m unittest tools/moe/test_gate_diff.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate_diff as G  # noqa: E402

TOOL = os.path.join(HERE, "gate_diff.py")

Q3K = (
    "/home/yydh/.local/state/dpsk-q2/runs/T-0252/"
    "run-20261123T083851-w5-1/gate/gate.json"
)
Q2K = (
    "/home/yydh/.local/state/dpsk-q2/runs/T-0282/"
    "run-20261123T105956-w2-1/gate-nubia-cap-paired/gate.json"
)
DRY = (
    "/home/yydh/.local/state/dpsk-q2/runs/T-0366/"
    "run-20261123T123702-w2-1/artifacts/gate-dry/gate.json"
)
DRY_Q2K = (
    "/home/yydh/.local/state/dpsk-q2/runs/T-0366/"
    "run-20261123T123702-w2-1/artifacts/gate-dry-q2k2/gate.json"
)
HAVE_REAL = all(os.path.exists(path) for path in (Q3K, Q2K, DRY, DRY_Q2K))


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def entries_by_key(entries):
    return {entry["key"]: entry for entry in entries}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._counter = 0

    def write(self, data, name=None):
        self._counter += 1
        path = os.path.join(self.tmp.name, name or f"gate-{self._counter}.json")
        with open(path, "w", encoding="utf-8") as handle:
            if isinstance(data, str):
                handle.write(data)
            else:
                json.dump(data, handle)
        return path

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, TOOL, *args],
            cwd=os.path.dirname(HERE),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


class FlattenTests(Base):
    def test_absent_is_omitted_and_null_is_kept(self):
        flat = G.flatten_gate({"phone": None})
        self.assertIn("phone", flat)
        self.assertIsNone(flat["phone"])
        self.assertNotIn("arch", flat)

    def test_package_aliases(self):
        document = {
            "speed": {
                "baseline_package": {
                    "pin_id": "q2k-idnow",
                    "manifest_sha256": "abc",
                    "bytes": 42,
                    "stride0": 442368,
                    "requested_bytes": 3141697536,
                }
            }
        }
        flat = G.flatten_gate(document)
        self.assertEqual(flat["baseline_package.id"], "q2k-idnow")
        self.assertEqual(flat["baseline_package.sha"], "abc")
        self.assertEqual(flat["baseline_package.bytes"], 42)
        self.assertEqual(flat["baseline_package.stride0"], 442368)
        self.assertEqual(flat["baseline_package.requested_bytes"], 3141697536)

    def test_quality_and_health_summaries_do_not_echo_whole_blocks(self):
        document = {
            "speed": {
                "health": {"ok": True, "arms": {"A": {"runs": 4}}},
                "quality": {
                    "effective_result": "pass",
                    "definitions": {"long": "x" * 1000},
                    "arm_a": {"acceptance": 0.81},
                },
            }
        }
        flat = G.flatten_gate(document)
        self.assertEqual(flat["health.ok"], True)
        self.assertEqual(flat["health.arms.A.runs"], 4)
        self.assertEqual(flat["quality.effective_result"], "pass")
        self.assertEqual(flat["quality.arm_a.acceptance"], 0.81)
        self.assertNotIn("quality.definitions.long", flat)

    def test_missing_and_null_are_distinct_in_diff(self):
        entries = G.diff_dicts(G.flatten_gate({"phone": None}), G.flatten_gate({}))
        phone = entries_by_key(entries)["phone"]
        self.assertEqual(phone["status"], "missing")

        same_entries = G.diff_dicts(
            G.flatten_gate({"phone": None}), G.flatten_gate({"phone": None})
        )
        self.assertEqual(entries_by_key(same_entries)["phone"]["status"], "same")


class RealFixtureTests(Base):
    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_identical_file_is_comparable_and_has_no_diff(self):
        report = G.build_report(load(Q2K), load(Q2K), path_a=Q2K, path_b=Q2K)
        self.assertTrue(report["comparable"])
        self.assertEqual(report["incomparable_reasons"], [])
        for group, entries in report["groups"].items():
            for entry in entries:
                self.assertNotEqual(entry["status"], "diff", (group, entry))

    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_q3k_vs_q2k_is_incomparable_and_keeps_after_tps(self):
        report = G.build_report(load(Q3K), load(Q2K), path_a=Q3K, path_b=Q2K)
        self.assertFalse(report["comparable"])
        reasons = set(report["incomparable_reasons"])
        self.assertIn("package_sha_diff", reasons)
        self.assertIn("package_bytes_diff", reasons)
        self.assertIn("package_stride0_diff", reasons)
        self.assertIn("package_requested_bytes_diff", reasons)

        after = entries_by_key(report["groups"]["result"])["after_tps"]
        self.assertEqual(after["a"], 15.6)
        self.assertEqual(after["b"], 25.45)
        self.assertEqual(after["status"], "diff")

        package = entries_by_key(report["groups"]["package"])
        self.assertEqual(package["baseline_package.stride0"]["a"], 1351680)
        self.assertEqual(package["baseline_package.stride0"]["b"], 442368)
        self.assertTrue(
            report["package_evidence"]["a"]["stride0"]["source"].endswith(".err")
        )

    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_dry_run_vs_real_gate(self):
        report = G.build_report(load(DRY_Q2K), load(Q2K), path_a=DRY_Q2K, path_b=Q2K)
        self.assertFalse(report["comparable"])
        self.assertIn("dry_run", report["incomparable_reasons"])
        dry = entries_by_key(report["groups"]["identity"])["dry_run"]
        self.assertEqual(dry["a"], True)
        self.assertEqual(dry["b"], False)
        self.assertEqual(dry["status"], "diff")

    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_strict_json_mode_does_not_read_run_logs(self):
        report = G.build_report(
            load(Q3K), load(Q2K), path_a=Q3K, path_b=Q2K, package_evidence="off"
        )
        self.assertFalse(report["comparable"])
        self.assertNotIn("package_stride0_diff", report["incomparable_reasons"])
        self.assertEqual(report["package_evidence"]["a"], {})

    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_same_file_require_comparable_exits_zero(self):
        result = self.run_cli("--a", Q2K, "--b", Q2K, "--require-comparable")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("comparable: true", result.stdout)

    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_incomparable_require_comparable_exits_one(self):
        result = self.run_cli(
            "--a", Q3K, "--b", Q2K, "--require-comparable", "--package-evidence", "off"
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("comparable: false", result.stdout)

    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_plain_diff_still_exits_zero(self):
        result = self.run_cli("--a", Q3K, "--b", Q2K, "--package-evidence", "off")
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(HAVE_REAL, "real gate fixtures are not present on this host")
    def test_json_and_markdown_outputs(self):
        json_path = os.path.join(self.tmp.name, "diff.json")
        md_path = os.path.join(self.tmp.name, "diff.md")
        result = self.run_cli(
            "--a", Q3K, "--b", Q2K,
            "--json", json_path, "--markdown", md_path,
            "--label-a", "q3k", "--label-b", "q2k",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(json_path, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["schema"], "gate-diff.v1")
        self.assertEqual(report["a"]["label"], "q3k")
        self.assertFalse(report["comparable"])
        self.assertIn("incomparable_reasons", report)
        for group in ("identity", "package", "config", "health", "result"):
            self.assertIn(group, report["groups"])
            for entry in report["groups"][group]:
                self.assertEqual(set(entry), {"key", "a", "b", "status"})

        with open(md_path, encoding="utf-8") as handle:
            markdown = handle.read()
        for group in ("identity", "package", "config", "health", "result"):
            self.assertIn(f"## {group}", markdown)
        self.assertIn("Comparable: false", markdown)
        self.assertIn("package_stride0_diff", markdown)


class SyntheticTests(Base):
    def test_missing_speed_and_health_are_reported_missing(self):
        with_speed = {
            "task": "A", "mode": "build", "overall": "pass", "phone": "nubia",
            "arch": "79", "bench": "b.sh", "skel": "s.so",
            "gate_version": "v", "dry_run": False,
            "build": {"ok": True}, "accuracy": {"ok": True},
            "speed": {"ok": True, "after_tps": 1.0, "runs": "4"},
        }
        without_speed = {
            "task": "B", "mode": "build", "overall": "pass", "phone": "nubia",
            "arch": "79", "bench": "b.sh", "skel": "s.so",
            "gate_version": "v", "dry_run": False,
            "build": {"ok": True}, "accuracy": {"ok": True},
        }
        report = G.build_report(with_speed, without_speed)
        flat = G.flatten_gate(without_speed)
        self.assertNotIn("speed.ok", flat)
        self.assertNotIn("health.ok", flat)
        result_entries = entries_by_key(report["groups"]["result"])
        self.assertEqual(result_entries["speed.ok"]["status"], "missing")
        self.assertEqual(result_entries["after_tps"]["status"], "missing")

    def test_bad_json_exits_two(self):
        bad = self.write("{not json")
        good = self.write({"task": "ok"})
        result = self.run_cli("--a", bad, "--b", good)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("invalid JSON", result.stderr)

    def test_non_dict_json_exits_two(self):
        array = self.write("[]")
        good = self.write({"task": "ok"})
        result = self.run_cli("--a", array, "--b", good)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("expected object", result.stderr)

    def test_missing_input_exits_two(self):
        good = self.write({"task": "ok"})
        result = self.run_cli("--a", os.path.join(self.tmp.name, "nope.json"), "--b", good)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_require_comparable_on_synthetic_pair(self):
        left = self.write({"task": "A", "phone": "nubia", "arch": "79"})
        right = self.write({"task": "B", "phone": "vivo", "arch": "79"})
        result = self.run_cli("--a", left, "--b", right, "--require-comparable")
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_markdown_has_five_groups_even_when_empty(self):
        left = self.write({"task": "A"})
        right = self.write({"task": "B"})
        md_path = os.path.join(self.tmp.name, "empty.md")
        result = self.run_cli("--a", left, "--b", right, "--markdown", md_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(md_path, encoding="utf-8") as handle:
            markdown = handle.read()
        for group in ("identity", "package", "config", "health", "result"):
            self.assertIn(f"## {group}", markdown)
        self.assertIn("Comparable: true", markdown)


if __name__ == "__main__":
    unittest.main()
