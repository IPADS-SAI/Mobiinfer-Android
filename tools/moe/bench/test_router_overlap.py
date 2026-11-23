#!/usr/bin/env python3
"""Tests for T-0097: tools/moe/router_overlap.py (stdlib unittest only).

Run from the repository root:

    python3 tools/moe/bench/test_router_overlap.py
    python3 -m unittest discover -s tools/moe/bench -p 'test_router_overlap.py' -v

Every fixture is a tiny synthetic LLAMA_MOE_SWEEP trace written with tempfile;
no model, phone, A100, or build is involved.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS_DIR / "router_overlap.py"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import router_overlap as ro  # noqa: E402


def _ids(values) -> str:
    return ",".join(str(value) for value in values)


def make_lines(layers, steps, used_fn, pred_fn=None) -> list[str]:
    """Build a decode sweep in runtime order: for each step, each layer."""
    lines: list[str] = []
    for step in range(steps):
        for layer in layers:
            used = used_fn(layer, step)
            pred = used_fn(layer, step) if pred_fn is None else pred_fn(layer, step)
            lines.append(f"L{layer} E:{_ids(used)} P:{_ids(pred)}")
    return lines


def unique_count(trace: ro.Trace) -> int:
    """Mirror window_unique/n_unique: sum of distinct experts over layer-steps."""
    return sum(len(set(step.used)) for steps in trace.layers.values() for step in steps)


class RouterOverlapTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write(self, name: str, lines) -> Path:
        path = self.tmp / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            capture_output=True,
            text=True,
            cwd=self.tmp,
        )

    def load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    # 1. Identical traces: every metric is 1 / 0 delta.
    def test_identical_traces(self) -> None:
        layers = [0, 1]
        lines = make_lines(layers, 3, lambda l, s: range(8), lambda l, s: range(8))
        a = self.write("a.txt", lines)
        b = self.write("b.txt", lines)
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("layers=2 steps=3", proc.stdout)

        report = self.load_json(out)
        for layer in ("0", "1"):
            row = report["layers"][layer]
            self.assertEqual(row["jaccard_E"], 1.0)
            self.assertEqual(row["exact_match_rate"], 1.0)
            self.assertEqual(row["mean_size_delta"], 0.0)
            self.assertEqual(row["jaccard_P"], 1.0)
            self.assertEqual(row["rbo_p90"], 1.0)
            self.assertEqual(row["rbo_p98"], 1.0)
            self.assertEqual(row["top1_agreement_P"], 1.0)
        agg = report["aggregate"]
        self.assertEqual(agg["n_layers"], 2)
        self.assertEqual(agg["n_steps"], 3)
        self.assertEqual(agg["n_mismatch_layers"], 0)
        self.assertEqual(agg["mean_jaccard_E"], 1.0)
        self.assertEqual(agg["exact_match_rate"], 1.0)

    # 2. T-0080 counterexample: same count (8/step, 80 total), disjoint ids.
    def test_disjoint_t080_counterexample(self) -> None:
        layers = [0, 1]
        a_lines = make_lines(layers, 5, lambda l, s: range(0, 8))
        b_lines = make_lines(layers, 5, lambda l, s: range(8, 16))
        a = self.write("a.txt", a_lines)
        b = self.write("b.txt", b_lines)
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)

        report = self.load_json(out)
        agg = report["aggregate"]
        self.assertEqual(agg["n_layers"], 2)
        self.assertEqual(agg["n_steps"], 5)
        self.assertEqual(agg["mean_jaccard_E"], 0.0)
        self.assertEqual(agg["exact_match_rate"], 0.0)
        self.assertEqual(agg["n_mismatch_layers"], 2)
        # The blind spot: the runtime counter only counts how many experts were
        # requested, so both arms report window_unique=8*5*2=80.
        self.assertEqual(unique_count(ro.parse_trace(str(a))), 80)
        self.assertEqual(unique_count(ro.parse_trace(str(b))), 80)
        self.assertEqual(agg["mean_jaccard_E"], 0.0)
        # E is not rank, so no E top-1 is emitted; P top-1 is disjoint here too.
        self.assertEqual(report["layers"]["0"]["top1_agreement_P"], 0.0)

    # 3. Half overlap: |A∩B|/|A∪B| = 2/6.
    def test_half_overlap(self) -> None:
        a = self.write("a.txt", ["L0 E:0,1,2,3 P:0,1,2,3"])
        b = self.write("b.txt", ["L0 E:2,3,4,5 P:2,3,4,5"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = self.load_json(out)["layers"]["0"]
        self.assertAlmostEqual(row["jaccard_E"], 2.0 / 6.0, delta=1e-12)
        self.assertAlmostEqual(row["mean_overlap_E"], 2.0 / 4.0, delta=1e-12)
        self.assertEqual(row["exact_match_rate"], 0.0)

    # 4. Different sizes: A=8, B=4; overlap_fraction and signed size_delta.
    def test_different_sizes(self) -> None:
        a = self.write("a.txt", ["L0 E:0,1,2,3,4,5,6,7 P:0,1,2,3,4,5,6,7"])
        b = self.write("b.txt", ["L0 E:0,1,2,3 P:0,1,2,3"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = self.load_json(out)["layers"]["0"]
        self.assertAlmostEqual(row["mean_overlap_E"], 4.0 / 8.0, delta=1e-12)
        self.assertEqual(row["mean_size_delta"], -4.0)
        self.assertAlmostEqual(row["jaccard_E"], 4.0 / 8.0, delta=1e-12)
        self.assertEqual(row["steps"][0]["n_A"], 8)
        self.assertEqual(row["steps"][0]["n_B"], 4)

    # 5. P on one side only: P metrics null, E metrics still work, no crash.
    def test_one_side_without_p(self) -> None:
        a = self.write("a.txt", ["L0 E:0,1,2,3 P:3,2,1,0"])
        b = self.write("b.txt", ["L0 E:0,1,2,3 P:"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        row = report["layers"]["0"]
        self.assertIsNone(row["jaccard_P"])
        self.assertIsNone(row["rbo_p90"])
        self.assertIsNone(row["rbo_p98"])
        self.assertIsNone(row["top1_agreement_P"])
        self.assertEqual(row["jaccard_E"], 1.0)
        self.assertIn("P metrics are null", "\n".join(report["warnings"]))

    # 6. Layer sets differ: exit 2, then subset mode explicitly skips a layer.
    def test_layer_mismatch_fail_closed_and_subset(self) -> None:
        layers_a = [0, 1]
        layers_b = [0, 1, 2]
        a = self.write("a.txt", make_lines(layers_a, 2, lambda l, s: range(8)))
        b = self.write("b.txt", make_lines(layers_b, 2, lambda l, s: range(8)))

        proc = self.run_cli(a, b)
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("layer=2", proc.stderr)
        self.assertIn("steps_A=0", proc.stderr)
        self.assertIn("steps_B=2", proc.stderr)
        self.assertEqual(proc.stdout, "")

        out = self.tmp / "subset.json"
        proc = self.run_cli(a, b, "--allow-layer-subset", "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        agg = self.load_json(out)["aggregate"]
        self.assertEqual(agg["n_layers"], 2)
        self.assertEqual(agg["skipped_layers"], [2])

    # 7. Same layers, different step counts: still fail-closed, even in subset mode.
    def test_step_mismatch_fail_closed(self) -> None:
        a = self.write("a.txt", make_lines([0], 2, lambda l, s: range(8)))
        b = self.write("b.txt", make_lines([0], 3, lambda l, s: range(8)))
        proc = self.run_cli(a, b)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("layer=0 steps_A=2 steps_B=3", proc.stderr)

        proc = self.run_cli(a, b, "--allow-layer-subset")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("per-layer step counts differ", proc.stderr)

    # 8. Blank/comment/prefill/bad lines: skipped and counted, never empty sets.
    def test_blank_malformed_and_prefill_lines(self) -> None:
        lines = [
            "# layer real_experts | predicted_experts(desc)",
            "",
            "L0 E:0,1 P:0,1",
            "L0 E:0,x P:0,1",
            "garbage",
            "P0 E:9,10",
            "L0 E:0,1 P:0,1",
            "L0 E:1,2 P:1,2",
        ]
        a = self.write("a.txt", lines)
        b = self.write("b.txt", ["L0 E:0,1 P:0,1", "L0 E:0,1 P:0,1", "L0 E:1,2 P:1,2"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        stats = report["trace_stats"]["a"]
        self.assertEqual(stats["malformed_lines"], 2)
        self.assertEqual(stats["prefill_lines_skipped"], 1)
        self.assertEqual(stats["blank_lines"], 1)
        self.assertEqual(stats["comment_lines"], 1)
        self.assertEqual(stats["decode_lines"], 3)
        # If the two bad lines had been treated as empty sets, n_steps would be 5
        # and the mean Jaccard would drop below 1.0.
        self.assertEqual(report["layers"]["0"]["n_steps"], 3)
        self.assertEqual(report["layers"]["0"]["jaccard_E"], 1.0)
        self.assertEqual(report["aggregate"]["malformed_lines"]["a"], 2)

    # 9. RBO against a hand calculation, both contract p values.
    def test_rbo_known_example(self) -> None:
        # P_A=[1,2,3], P_B=[1,3,2]:
        # d=1 X=1, d=2 X=1, d=3 X=3.
        # p=0.9 : 0.1*(1 + 0.9/2 + 0.81*3/3) + 0.9^3*3/3 = 0.955
        # p=0.98: 0.02*(1 + 0.98/2 + 0.9604) + 0.98^3 = 0.9902
        self.assertAlmostEqual(ro.rbo([1, 2, 3], [1, 3, 2], 0.9), 0.955, delta=1e-9)
        self.assertAlmostEqual(ro.rbo([1, 2, 3], [1, 3, 2], 0.98), 0.9902, delta=1e-9)
        self.assertAlmostEqual(ro.rbo([1, 2, 3], [1, 2, 3], 0.9), 1.0, delta=1e-12)
        self.assertEqual(ro.rbo([1, 2], [3, 4], 0.9), 0.0)
        self.assertEqual(ro.rbo([], [], 0.9), 1.0)
        self.assertEqual(ro.rbo([1], [], 0.9), 0.0)

    # 10. Explicit count-blind-spot assertion, separate from the CLI report.
    def test_window_unique_same_but_jaccard_zero(self) -> None:
        a = self.write("a.txt", make_lines([0, 1], 5, lambda l, s: range(0, 8)))
        b = self.write("b.txt", make_lines([0, 1], 5, lambda l, s: range(8, 16)))
        ta, tb = ro.parse_trace(str(a)), ro.parse_trace(str(b))
        self.assertEqual(unique_count(ta), 80)
        self.assertEqual(unique_count(tb), 80)
        report = ro.build_report(ta, tb)
        self.assertEqual(report["aggregate"]["mean_jaccard_E"], 0.0)
        self.assertEqual(report["aggregate"]["n_mismatch_layers"], 2)

    # 11. TSV header is exactly the contract, values are per-layer means.
    def test_tsv_header_and_values(self) -> None:
        lines = make_lines([0, 1], 2, lambda l, s: range(8), lambda l, s: range(8))
        a = self.write("a.txt", lines)
        b = self.write("b.txt", lines)
        out = self.tmp / "out.tsv"
        proc = self.run_cli(a, b, "--out-tsv", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            rows[0].split("\t"),
            [
                "layer",
                "n_steps",
                "jaccard_E",
                "mean_overlap_E",
                "exact_match_rate",
                "mean_size_delta",
                "jaccard_P",
                "rbo_p90",
                "rbo_p98",
                "top1_agreement_P",
            ],
        )
        self.assertEqual(len(rows), 3)
        fields = rows[1].split("\t")
        self.assertEqual(fields[0], "0")
        self.assertEqual(fields[1], "2")
        self.assertEqual(fields[2], "1.000000")
        self.assertEqual(fields[4], "1.000000")
        self.assertEqual(fields[7], "1.000000")
        self.assertEqual(fields[8], "1.000000")
        self.assertEqual(fields[9], "1.000000")

    # 12. Default JSON redaction omits absolute paths; opt-out keeps them.
    def test_json_redacts_paths_by_default(self) -> None:
        a = self.write("a.txt", ["L0 E:0,1 P:0,1"])
        b = self.write("b.txt", ["L0 E:0,1 P:0,1"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        self.assertEqual(report["a"], "a.txt")
        self.assertEqual(report["b"], "b.txt")
        self.assertTrue(report["redacted"])
        self.assertNotIn(str(self.tmp), out.read_text(encoding="utf-8"))

        out2 = self.tmp / "out2.json"
        proc = self.run_cli(a, b, "--out-json", out2, "--no-redact-paths")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.load_json(out2)["a"], str(a))

    # 13. --min-steps is fail-closed with layer/step details.
    def test_min_steps_fail_closed(self) -> None:
        a = self.write("a.txt", ["L0 E:0,1 P:0,1"])
        b = self.write("b.txt", ["L0 E:0,1 P:0,1"])
        proc = self.run_cli(a, b, "--min-steps", "2")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("layer=0", proc.stderr)
        self.assertIn("min_steps=2", proc.stderr)

    # 14. --layers checks the expected count instead of silently accepting it.
    def test_expected_layer_count(self) -> None:
        a = self.write("a.txt", ["L0 E:0,1 P:0,1"])
        b = self.write("b.txt", ["L0 E:0,1 P:0,1"])
        proc = self.run_cli(a, b, "--layers", "2")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("expected 2 layers", proc.stderr)
        proc = self.run_cli(a, b, "--layers", "1")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    # 15. Prefill P<n> lines are counted but never become decode layers.
    def test_prefill_lines_are_not_mixed(self) -> None:
        a = self.write(
            "a.txt",
            ["P0 E:5,6", "L0 E:0,1 P:0,1", "P1 E:7,8"],
        )
        b = self.write("b.txt", ["L0 E:0,1 P:0,1"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        self.assertEqual(report["trace_stats"]["a"]["prefill_lines_skipped"], 2)
        self.assertEqual(report["aggregate"]["n_layers"], 1)
        self.assertEqual(report["aggregate"]["n_steps"], 1)
        self.assertNotIn("5", report["layers"])
        self.assertTrue(
            any("prefill" in warning for warning in report["warnings"]),
            report["warnings"],
        )

    # 16. Both E sets empty: Jaccard is defined as 1.0 and counted.
    def test_both_empty_sets(self) -> None:
        a = self.write("a.txt", ["L0 E: P:"])
        b = self.write("b.txt", ["L0 E: P:"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        row = report["layers"]["0"]
        self.assertEqual(row["jaccard_E"], 1.0)
        self.assertEqual(row["exact_match_rate"], 1.0)
        self.assertEqual(row["empty_steps"], 1)
        self.assertEqual(report["aggregate"]["empty_steps"], 1)

    # 17. Partial P availability: only steps with P on both sides contribute.
    def test_partial_p_steps(self) -> None:
        a = self.write("a.txt", ["L0 E:0,1 P:1,2", "L0 E:0,1 P:"])
        b = self.write("b.txt", ["L0 E:0,1 P:1,3", "L0 E:0,1 P:4,5"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        row = report["layers"]["0"]
        self.assertEqual(row["n_steps_P"], 1)
        self.assertAlmostEqual(row["jaccard_P"], 1.0 / 3.0, delta=1e-12)
        self.assertEqual(row["top1_agreement_P"], 1.0)
        self.assertAlmostEqual(row["rbo_p90"], 0.55, delta=1e-12)
        self.assertIn("cover 1/2", "\n".join(report["warnings"]))

    # 18. Duplicate ids: dedupe for sets/ranks, but report the line count.
    def test_duplicate_ids(self) -> None:
        a = self.write("a.txt", ["L0 E:1,1,2 P:3,3,4"])
        b = self.write("b.txt", ["L0 E:1,2 P:3,4"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        stats = report["trace_stats"]["a"]
        self.assertEqual(stats["duplicate_E_lines"], 1)
        self.assertEqual(stats["duplicate_P_lines"], 1)
        self.assertEqual(report["layers"]["0"]["jaccard_E"], 1.0)
        self.assertEqual(report["layers"]["0"]["steps"][0]["E_A"], [1, 2])
        self.assertEqual(report["layers"]["0"]["steps"][0]["P_A"], [3, 4])

    # 19. E order is not rank: reordered E is an exact set match; P order still matters.
    def test_e_is_not_rank_but_p_order_is(self) -> None:
        a = self.write("a.txt", ["L0 E:9,1,5 P:1,2,3"])
        b = self.write("b.txt", ["L0 E:5,1,9 P:3,2,1"])
        out = self.tmp / "out.json"
        proc = self.run_cli(a, b, "--out-json", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = self.load_json(out)
        row = report["layers"]["0"]
        self.assertEqual(row["jaccard_E"], 1.0)
        self.assertEqual(row["exact_match_rate"], 1.0)
        self.assertEqual(row["jaccard_P"], 1.0)
        self.assertEqual(row["top1_agreement_P"], 0.0)
        self.assertLess(row["rbo_p90"], 1.0)
        self.assertNotIn("top1_agreement_E", row)
        self.assertTrue(
            any("E[0] is NOT top-1" in warning for warning in report["warnings"]),
            report["warnings"],
        )

    # 20. --help is complete enough to be discoverable.
    def test_help(self) -> None:
        proc = self.run_cli("--help")
        self.assertEqual(proc.returncode, 0)
        for option in (
            "--out-json",
            "--out-tsv",
            "--layers",
            "--min-steps",
            "--p",
            "--allow-layer-subset",
            "--redact-paths",
        ):
            self.assertIn(option, proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
