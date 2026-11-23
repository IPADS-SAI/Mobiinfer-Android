#!/usr/bin/env python3
"""Unit tests for tools/moe/bench_attrib.py (host-only, stdlib unittest)."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import bench_attrib as ba  # noqa: E402

# The exact sample from the task spec (T-0032 A1): acceptance 0.90196,
# 92/102 tokens, 34 calls, 33 graphs, tg 27.44, dur_g 589.643 ms.
SESSION_A = """\
[ Prompt: 24.5 t/s | Generation: 27.6 t/s ]
0.08.188.311 I slot print_timing: id  0 | task 0 | n_decoded =    103, tg =  27.44 t/s
0.09.057.574 I slot print_timing: id  0 | task 0 | prompt eval time =     818.03 ms /    22 tokens (   37.18 ms per token,    26.89 tokens per second)
0.09.057.581 I slot print_timing: id  0 | task 0 |        eval time =    4623.54 ms /   128 tokens (   36.12 ms per token,    27.68 tokens per second)
0.09.057.582 I slot print_timing: id  0 | task 0 |       total time =    5441.57 ms /   150 tokens
0.09.057.582 I slot print_timing: id  0 | task 0 |    graphs reused =         33
0.09.057.583 I slot print_timing: id  0 | task 0 | draft acceptance = 0.90196 (   92 accepted /   102 generated)
0.09.057.595 I statistics     draft-dspark: #calls(b,g,a) =    1     34     34, #gen drafts =     34, #acc drafts =     33, #gen tokens =    102, #acc tokens =    92, dur(b,g,a) = 0.001, 589.643, 0.004 ms
"""

SESSION_B = """\
[ Prompt: 24.9 t/s | Generation: 19.6 t/s ]
0.09.863.810 I slot print_timing: id  0 | task 0 | n_decoded =    103, tg =  19.75 t/s
0.11.196.607 I slot print_timing: id  0 | task 0 | prompt eval time =     882.72 ms /    22 tokens (   40.12 ms per token,    24.92 tokens per second)
0.11.196.614 I slot print_timing: id  0 | task 0 |        eval time =    6546.68 ms /   128 tokens (   51.15 ms per token,    19.55 tokens per second)
0.11.196.615 I slot print_timing: id  0 | task 0 |       total time =    7429.39 ms /   150 tokens
0.11.196.616 I slot print_timing: id  0 | task 0 |    graphs reused =         46
0.11.196.617 I slot print_timing: id  0 | task 0 | draft acceptance = 0.55634 (   79 accepted /   142 generated)
0.11.196.630 I statistics     draft-dspark: #calls(b,g,a) =    1     48     48, #gen drafts =     48, #acc drafts =     34, #gen tokens =    142, #acc tokens =    79, dur(b,g,a) = 0.001, 826.468, 0.005 ms
"""

RESET = "0.03.700.508 I common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dspark'\n"


class ParseSessionTests(unittest.TestCase):
    def test_single_session_full(self):
        sessions = ba.parse_log(SESSION_A)
        self.assertEqual(len(sessions), 1)
        s = sessions[0]
        self.assertEqual(s["acceptance"], 0.90196)
        self.assertEqual(s["accepted"], 92)
        self.assertEqual(s["generated"], 102)
        self.assertEqual(s["calls_g"], 34)
        self.assertEqual(s["calls_a"], 34)
        self.assertEqual(s["graphs_reused"], 33)
        self.assertEqual(s["tg_tps"], 27.44)
        self.assertEqual(s["n_decoded"], 103)
        self.assertEqual(s["prompt_tps"], 26.89)
        self.assertEqual(s["eval_ms"], 4623.54)
        self.assertEqual(s["gen_tokens"], 102)
        self.assertEqual(s["acc_tokens"], 92)
        self.assertEqual(s["draft_gen"], 34)
        self.assertEqual(s["draft_acc"], 33)
        self.assertAlmostEqual(s["per_call_ms"], 589.643 / 34, places=9)
        self.assertAlmostEqual(s["per_call_ms_eval"], 4623.54 / 34, places=9)
        self.assertEqual(s["missing"], [])

    def test_two_sessions_do_not_cross(self):
        text = RESET + SESSION_A + "\n" + RESET + SESSION_B
        sessions = ba.parse_log(text)
        self.assertEqual(len(sessions), 2)
        a, b = sessions
        self.assertEqual((a["acceptance"], a["calls_g"], a["graphs_reused"], a["tg_tps"]),
                         (0.90196, 34, 33, 27.44))
        self.assertEqual((b["acceptance"], b["calls_g"], b["graphs_reused"], b["tg_tps"]),
                         (0.55634, 48, 46, 19.75))
        self.assertEqual((a["accepted"], a["generated"]), (92, 102))
        self.assertEqual((b["accepted"], b["generated"]), (79, 142))

    def test_missing_acceptance_is_none_and_listed(self):
        text = "\n".join(
            line for line in SESSION_A.splitlines() if "draft acceptance" not in line
        )
        sessions = ba.parse_log(text)
        self.assertEqual(len(sessions), 1)
        s = sessions[0]
        self.assertIsNone(s["acceptance"])
        self.assertIsNone(s["accepted"])
        self.assertIsNone(s["generated"])
        self.assertIn("acceptance", s["missing"])
        self.assertIn("accepted", s["missing"])
        self.assertIn("generated", s["missing"])
        # Unrelated fields still parse.
        self.assertEqual(s["calls_g"], 34)
        self.assertEqual(s["tg_tps"], 27.44)

    def test_malformed_and_empty_do_not_crash(self):
        self.assertEqual(ba.parse_log(""), [])
        self.assertEqual(ba.parse_log("garbage\nslot print_timing without fields\n"), [])
        # A malformed tg value must become None (or use the summary fallback),
        # not raise or become 0.  Without a [Generation:] line it is missing.
        no_summary = "\n".join(SESSION_A.splitlines()[1:])
        broken = no_summary.replace("tg =  27.44", "tg =  nope")
        sessions = ba.parse_log(broken)
        self.assertEqual(len(sessions), 1)
        self.assertIsNone(sessions[0]["tg_tps"])
        self.assertIn("tg_tps", sessions[0]["missing"])
        # With the summary present, the documented fallback still works.
        self.assertEqual(ba.parse_log(SESSION_A.replace("tg =  27.44", "tg =  nope"))[0]["tg_tps"], 27.6)
        # parse_session_block is also tolerant of arbitrary junk.
        junk = ba.parse_session_block(["n_decoded = nope", "graphs reused = x", "?"])
        self.assertIsNone(junk["n_decoded"])
        self.assertIsNone(junk["graphs_reused"])

    def test_empty_calls_gives_empty_per_call(self):
        text = SESSION_A.replace(
            "#calls(b,g,a) =    1     34     34", "#calls(b,g,a) =    1      0      0"
        )
        s = ba.parse_log(text)[0]
        self.assertEqual(s["calls_g"], 0)
        self.assertIsNone(s["per_call_ms"])
        self.assertIn("per_call_ms", s["missing"])


class ArmTests(unittest.TestCase):
    def test_arm_from_filename(self):
        self.assertEqual(ba.arm_from_path("A-1.log"), "A")
        self.assertEqual(ba.arm_from_path("B-2.log"), "B")
        self.assertEqual(ba.arm_from_path("foo.log"), "unknown")
        self.assertEqual(ba.arm_from_path("/tmp/runs/A_3.log"), "A")

    def test_arm_from_marker_sequence(self):
        text = (
            f"[run-start] A 2026-11-22T21:20:52+08:00\n{RESET}{SESSION_A}\n"
            f"[run-start] B 2026-11-22T21:21:21+08:00\n{RESET}{SESSION_B}\n"
        )
        sessions = ba.parse_log(text, path="combined.log")
        self.assertEqual([s["arm"] for s in sessions], ["A", "B"])
        self.assertEqual(sessions[0]["acceptance"], 0.90196)
        self.assertEqual(sessions[1]["acceptance"], 0.55634)

    def test_failed_b_markers_do_not_steal_a_sessions(self):
        # abba-4a/b shape: A ok, B load failed, B load failed, A ok; only the
        # two A timing sessions exist.
        text = (
            f"[run-start] A t=1\n{RESET}{SESSION_A}\n"
            "[run-start] B t=2\nFailed to load the model\n"
            "[run-start] B t=3\nFailed to load the model\n"
            f"[run-start] A t=4\n{RESET}{SESSION_A}\n"
        )
        sessions = ba.parse_log(text, path="combined.log")
        self.assertEqual(len(sessions), 2)
        self.assertEqual([s["arm"] for s in sessions], ["A", "A"])

    def test_filename_regex_wins(self):
        text = f"[run-start] B t=1\n{RESET}{SESSION_A}\n"
        sessions = ba.parse_log(text, path="A-1.log")
        self.assertEqual(sessions[0]["arm"], "A")

    def test_custom_arm_regex(self):
        text = f"{RESET}{SESSION_A}\n"
        sessions = ba.parse_log(text, path="arm-baseline-01.log", arm_regex=r"arm-(baseline)")
        self.assertEqual(sessions[0]["arm"], "BASELINE")


class AggregateAndOutputTests(unittest.TestCase):
    def _sessions(self):
        a1 = ba.parse_log(SESSION_A)[0]
        a2 = ba.parse_log(SESSION_A.replace("27.44", "27.54"))[0]
        b1 = ba.parse_log(SESSION_B)[0]
        for s, arm in ((a1, "A"), (a2, "A"), (b1, "B")):
            s["file"] = "test.log"
            s["arm"] = arm
        return [a1, a2, b1]

    def test_aggregate_median_min_max(self):
        agg = ba.aggregate(self._sessions())
        self.assertEqual(agg["A"]["n"], 2)
        self.assertEqual(agg["A"]["tg_tps"]["n"], 2)
        self.assertAlmostEqual(agg["A"]["tg_tps"]["median"], 27.49)
        self.assertEqual(agg["A"]["tg_tps"]["min"], 27.44)
        self.assertEqual(agg["A"]["tg_tps"]["max"], 27.54)
        self.assertEqual(agg["B"]["n"], 1)
        self.assertEqual(agg["B"]["calls_g"], {"n": 1, "median": 48, "min": 48, "max": 48})
        self.assertEqual(agg["B"]["acceptance"]["median"], 0.55634)

    def test_tsv_fixed_header_and_empty_for_none(self):
        sessions = ba.parse_log(
            "\n".join(line for line in SESSION_A.splitlines() if "draft acceptance" not in line)
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sessions.tsv"
            ba.write_tsv(out, sessions)
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.reader(fh, delimiter="\t"))
        self.assertEqual(rows[0], ba.TSV_FIELDS)
        self.assertEqual(len(rows[1]), len(ba.TSV_FIELDS))
        row = dict(zip(rows[0], rows[1]))
        self.assertEqual(row["acceptance"], "")
        self.assertEqual(row["calls_g"], "34")
        self.assertEqual(row["tg_tps"], "27.44")


PREFILL_1 = "[moe-ssd] total[prefill]: miss=1000 bytes=100.0 MiB pread=1000ms\n"
PREFILL_2 = "[moe-ssd] total[prefill]: miss=2000 bytes=200.0 MiB pread=2000ms\n"
FINAL = "[moe-ssd] total[final]: miss=3000 bytes=300.0 MiB pread=3000ms\n"
PHASE_LOG = PREFILL_1 + SESSION_A + FINAL

# The legacy fixed columns must not change when the phase subcommand is added.
LEGACY_TSV_FIELDS = [
    "file", "session", "arm", "n_decoded", "tg_tps", "prompt_tps", "eval_ms",
    "graphs_reused", "acceptance", "accepted", "generated", "calls_g", "calls_a",
    "per_call_ms", "gen_tokens", "acc_tokens", "draft_gen", "draft_acc",
]

FIXTURE = HERE / "fixtures" / "t0102-q1-r1-snippet.log.txt"

# The 20 T-0128 phase columns; T-0344 appends package columns after these.
OLD_PHASE_TSV_FIELDS = [
    "run", "log", "arm", "group", "tg", "eval_ms", "n_tokens", "calls_b",
    "calls_g", "calls_a", "draft_gen_ms", "acceptance", "p_total_ms",
    "p_draft_ms", "p_target_ms", "pread_ms", "prefill_pread_ms",
    "decode_pread_ms", "p_target_ex_io", "parse_warnings",
]

P3_STRIDE0 = 1351680
P1_STRIDE0 = 442368
P3_REQUESTED = 12724469760
P1_REQUESTED = 2962096128
P3_BYTES = 16106407640
P1_BYTES = 4529927461


def _identity_lines(stride0=P3_STRIDE0, requested=P3_REQUESTED, experts=P3_BYTES):
    return (
        "[moe-ssd] init pack=on pack_layers=40/40 bank=256 shared=on "
        f"experts=256 io_threads=8 stride0={stride0} preadv=on park_window=on\n"
        f"[moe-ssd] experts_bytes={experts}\n"
    )


def _package_log(stride0=P3_STRIDE0, requested=P3_REQUESTED):
    """A complete phase log plus the real init/io-summary identity lines."""
    return (
        _identity_lines(stride0=stride0)
        + PHASE_LOG
        + "[moe-ssd] io-summary: scope=all_graphs_including_initialization_and_prefetch "
        f"requested_bytes={requested} span_requests=6696\n"
    )


def _run_phase(argv):
    """Run ``bench_attrib.py phase`` in-process and return its JSON output."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ba.main(["phase", *argv])
    return rc, buf.getvalue()


def _groups(payload):
    return {k: v for k, v in payload["aggregate"].items() if k != "excluded_runs"}


class PhaseTests(unittest.TestCase):
    def test_normal_formula(self):
        row = ba.build_phase_result([FIXTURE])["runs"][0]
        self.assertEqual(row["eval_ms"], 4584.79)
        self.assertEqual(row["n_tokens"], 128)
        self.assertEqual(row["calls_g"], 34)
        self.assertEqual(row["draft_gen_ms"], 576.490)
        self.assertEqual(row["acceptance"], 0.90196)
        self.assertAlmostEqual(row["p_total_ms"], 4584.79 / 34, places=9)
        self.assertAlmostEqual(row["p_draft_ms"], 576.490 / 34, places=9)
        self.assertAlmostEqual(row["p_target_ms"], (4584.79 - 576.490) / 34, places=9)
        self.assertEqual(row["pread_ms"], 9635)
        self.assertEqual(row["prefill_pread_ms"], 8408)
        self.assertEqual(row["decode_pread_ms"], 1227)
        self.assertAlmostEqual(row["p_target_ex_io"], (4584.79 - 576.490) / 34 - 1227 / 34, places=9)
        self.assertEqual(row["tg"], 27.92)  # final eval t/s, not the 27.88 snapshot
        self.assertEqual(row["parse_warning_count"], 0)
        self.assertEqual(row["missing"], [])

    def test_calls_g_zero_per_call_null(self):
        text = PHASE_LOG.replace(
            "#calls(b,g,a) =    1     34     34", "#calls(b,g,a) =    1      0      0"
        )
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "zero.log"
            log.write_text(text, encoding="utf-8")
            row = ba.build_phase_result([log])["runs"][0]
        self.assertEqual(row["calls_g"], 0)
        for field in ("p_total_ms", "p_draft_ms", "p_target_ms", "p_target_ex_io"):
            self.assertIsNone(row[field], field)
        self.assertEqual(row["pread_ms"], 3000)
        self.assertEqual(row["decode_pread_ms"], 2000)

    def test_missing_total_final(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "nofinal.log"
            log.write_text(PREFILL_1 + SESSION_A, encoding="utf-8")
            result = ba.build_phase_result([log])
        row = result["runs"][0]
        self.assertIsNone(row["pread_ms"])
        self.assertIsNone(row["decode_pread_ms"])
        self.assertEqual(row["prefill_pread_ms"], 1000)
        # Everything that does not depend on total[final] still works.
        self.assertAlmostEqual(row["p_target_ms"], (4623.54 - 589.643) / 34, places=9)
        self.assertIn("missing total[final] pread", row["parse_warnings"])
        self.assertGreater(row["parse_warning_count"], 0)

    def test_multiple_prefill_takes_last_and_largest_index(self):
        text = PREFILL_1 + PREFILL_2 + SESSION_A + FINAL
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "multi.log"
            log.write_text(text, encoding="utf-8")
            row = ba.build_phase_result([log])["runs"][0]
        self.assertEqual(row["prefill_pread_ms"], 2000)
        self.assertEqual(row["decode_pread_ms"], 1000)

        numbered = (
            "[moe-ssd] total[prefill#10]: miss=1 bytes=1.0 MiB pread=5000ms\n"
            "[moe-ssd] total[prefill#9]: miss=2 bytes=2.0 MiB pread=4000ms\n"
            + SESSION_A + FINAL
        )
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "numbered.log"
            log.write_text(numbered, encoding="utf-8")
            row = ba.build_phase_result([log])["runs"][0]
        self.assertEqual(row["prefill_pread_ms"], 5000)
        self.assertEqual(row["decode_pread_ms"], -2000)

    def test_two_number_calls_form_uses_second_as_calls_g(self):
        text = PHASE_LOG.replace(
            "#calls(b,g,a) =    1     34     34,", "#calls(b,g,a) =    1     34,"
        )
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "two.log"
            log.write_text(text, encoding="utf-8")
            row = ba.build_phase_result([log])["runs"][0]
        self.assertEqual(row["calls_b"], 1)
        self.assertEqual(row["calls_g"], 34)
        self.assertIsNone(row["calls_a"])
        self.assertAlmostEqual(row["p_total_ms"], 4623.54 / 34, places=9)
        self.assertIn("two-number #calls(b,g,a)", "\n".join(row["parse_warnings"]))

    def test_missing_out_file_still_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            logdir = Path(td) / "logs"
            logdir.mkdir()
            (logdir / "only-err.log").write_text(PHASE_LOG, encoding="utf-8")
            self.assertFalse((logdir / "only-err.out").exists())
            out_json = Path(td) / "out.json"
            rc, _ = _run_phase(["--logs", str(logdir), "--out-json", str(out_json)])
            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["runs"]), 1)
        self.assertIsNone(payload["runs"][0]["companion"])

    def test_out_companion_only_fills_missing_fields(self):
        text = "\n".join(
            line for line in PHASE_LOG.splitlines() if "draft acceptance" not in line
        )
        with tempfile.TemporaryDirectory() as td:
            err = Path(td) / "pair.err"
            out = Path(td) / "pair.out"
            err.write_text(text, encoding="utf-8")
            out.write_text(
                "0.08.744.429 I slot print_timing: id  0 | task 0 | "
                "draft acceptance = 0.90196 (   92 accepted /   102 generated)\n",
                encoding="utf-8",
            )
            row = ba.build_phase_result([err])["runs"][0]
        self.assertEqual(row["acceptance"], 0.90196)
        self.assertEqual(row["companion"], "pair.out")
        # The primary file's eval/calls values are kept.
        self.assertEqual(row["eval_ms"], 4623.54)

    def test_bad_lines_are_counted_not_silently_dropped(self):
        text = PHASE_LOG.replace(
            "draft acceptance = 0.90196 (   92 accepted /   102 generated)",
            "draft acceptance = nope",
        ).replace(
            "#calls(b,g,a) =    1     34     34", "#calls(b,g,a) =    1   nope     34"
        ).replace(
            "dur(b,g,a) = 0.001, 589.643, 0.004 ms", "dur(b,g,a) = 0.001, 589.643 ms"
        ).replace("pread=3000ms", "pread=3000furlongs")
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "bad.log"
            log.write_text(text, encoding="utf-8")
            row = ba.build_phase_result([log])["runs"][0]
        self.assertIsNone(row["acceptance"])
        self.assertIsNone(row["calls_g"])
        self.assertIsNone(row["pread_ms"])
        self.assertGreaterEqual(row["parse_warning_count"], 4)
        joined = "\n".join(row["parse_warnings"])
        self.assertIn("draft acceptance", joined)
        self.assertIn("#calls(b,g,a)", joined)
        self.assertIn("dur(b,g,a)", joined)
        self.assertIn("total[final]", joined)
        self.assertIn("line 8: unparsed draft acceptance", joined)  # exact line numbers

    def test_real_t0102_fixture(self):
        payload = ba.build_phase_result([FIXTURE])
        self.assertEqual(len(payload["runs"]), 1)
        row = payload["runs"][0]
        self.assertEqual(row["run"], "t0102-q1-r1-snippet.log")
        self.assertEqual(row["eval_ms"], 4584.79)
        self.assertEqual(row["tg"], 27.92)
        self.assertEqual(row["calls_g"], 34)
        self.assertEqual(row["pread_ms"], 9635)
        self.assertEqual(row["prefill_pread_ms"], 8408)
        self.assertEqual(row["decode_pread_ms"], 1227)
        self.assertAlmostEqual(row["p_target_ex_io"], 117.89117647058823 - 1227 / 34, places=6)
        # aggregate keeps the additive excluded_runs audit key plus one group.
        self.assertEqual(payload["aggregate"]["excluded_runs"], [])
        groups = {k: v for k, v in payload["aggregate"].items() if k != "excluded_runs"}
        self.assertEqual(len(groups), 1)
        self.assertEqual(next(iter(groups.values()))["n"], 1)

    def test_redact_paths_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "redact.log"
            log.write_text(PHASE_LOG, encoding="utf-8")
            out_json = Path(td) / "redacted.json"
            _run_phase(["--logs", str(log), "--out-json", str(out_json)])
            redacted = out_json.read_text(encoding="utf-8")
            out_json2 = Path(td) / "absolute.json"
            _run_phase(["--logs", str(log), "--out-json", str(out_json2),
                        "--no-redact-paths"])
            absolute = out_json2.read_text(encoding="utf-8")
        self.assertNotIn("/home/yydh", redacted)
        self.assertNotIn(td, redacted)
        self.assertEqual(json.loads(redacted)["runs"][0]["log"], "redact.log")
        self.assertIn(td, absolute)
        self.assertIn(str(log), absolute)

    def test_group_regex_and_aggregate_medians(self):
        with tempfile.TemporaryDirectory() as td:
            nubia_dir = Path(td) / "T-0031" / "artifacts" / "runs"
            vivo_dir = Path(td) / "T-0030" / "artifacts" / "runs"
            nubia_dir.mkdir(parents=True)
            vivo_dir.mkdir(parents=True)
            nubia_log = PHASE_LOG.replace("4623.54", "4600.00")
            vivo_log = PHASE_LOG.replace("4623.54", "3700.00")
            (nubia_dir / "q1-r1.log").write_text(nubia_log, encoding="utf-8")
            (vivo_dir / "specn-3-r1.log").write_text(vivo_log, encoding="utf-8")
            out_json = Path(td) / "agg.json"
            _run_phase([
                "--logs", str(nubia_dir), str(vivo_dir),
                "--group-regex", r"(?P<nubia>T-0031)|(?P<vivo>T-0030)",
                "--out-json", str(out_json),
            ])
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(
            sorted(k for k in payload["aggregate"] if k != "excluded_runs"),
            ["nubia", "vivo"],
        )
        self.assertAlmostEqual(payload["aggregate"]["nubia"]["tg_median"], 27.68)
        self.assertAlmostEqual(payload["aggregate"]["nubia"]["p_total_ms_median"], 4600.00 / 34)
        self.assertAlmostEqual(payload["aggregate"]["vivo"]["p_total_ms_median"], 3700.00 / 34)
        self.assertEqual(payload["aggregate"]["nubia"]["calls_g_values"], [34])

    def test_phase_help_exits_zero_and_documents_bounds(self):
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
            ba.main(["phase", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = buf.getvalue()
        self.assertIn("p_target_ex_io", text)
        self.assertIn("SERIAL UPPER BOUND", text)
        self.assertIn("LOWER BOUND", text)


class PackageGuardTests(unittest.TestCase):
    """T-0344: package identity guard on top of the phase subcommand."""

    def test_same_stride0_and_requested_pass(self):
        with tempfile.TemporaryDirectory() as td:
            logs = []
            for name in ("p3-r1.log", "p3-r2.log"):
                path = Path(td) / name
                path.write_text(_package_log(), encoding="utf-8")
                logs.append(path)
            payload = ba.build_phase_result(
                logs,
                expect_stride0=P3_STRIDE0,
                expect_requested_bytes=P3_REQUESTED,
                package_policy="exclude",
            )
        for row in payload["runs"]:
            self.assertIs(row["package_ok"], True)
            self.assertEqual(row["package_reason"], "match")
            self.assertFalse(row["excluded"])
            self.assertEqual(row["stride0"], P3_STRIDE0)
            self.assertEqual(row["requested_bytes"], P3_REQUESTED)
            self.assertEqual(row["pack_layers"], "40/40")
            self.assertEqual(row["experts"], 256)
        self.assertEqual(payload["package"]["excluded_runs"], [])
        self.assertEqual(payload["aggregate"]["excluded_runs"], [])
        self.assertEqual(payload["package"]["counts"]["match"], 2)
        self.assertEqual(next(iter(_groups(payload).values()))["n"], 2)

        # With no package flags at all, a consistent non-ambiguous stride0 group
        # also passes (legacy warn behaviour: mark, do not exclude).
        with tempfile.TemporaryDirectory() as td:
            logs = []
            for name in ("p3-r1.log", "p3-r2.log"):
                path = Path(td) / name
                path.write_text(_package_log(), encoding="utf-8")
                logs.append(path)
            legacy = ba.build_phase_result(logs)
        for row in legacy["runs"]:
            self.assertIs(row["package_ok"], True)
            self.assertEqual(row["package_reason"], "match")
            self.assertFalse(row["excluded"])
        self.assertEqual(legacy["package"]["excluded_runs"], [])

    def test_stride0_mismatch_excluded_under_exclude(self):
        with tempfile.TemporaryDirectory() as td:
            p3 = Path(td) / "p3.log"
            p1 = Path(td) / "p1.log"
            p3.write_text(_package_log(P3_STRIDE0, P3_REQUESTED), encoding="utf-8")
            p1.write_text(_package_log(P1_STRIDE0, P1_REQUESTED), encoding="utf-8")
            payload = ba.build_phase_result(
                [p3, p1], expect_stride0=P3_STRIDE0, package_policy="exclude"
            )
        rows = {row["run"]: row for row in payload["runs"]}
        self.assertIs(rows["p3"]["package_ok"], True)
        self.assertEqual(rows["p3"]["package_reason"], "match")
        self.assertIs(rows["p1"]["package_ok"], False)
        self.assertEqual(rows["p1"]["package_reason"], "stride0_mismatch")
        self.assertTrue(rows["p1"]["excluded"])
        self.assertEqual(payload["package"]["excluded_runs"], ["p1"])
        self.assertEqual(payload["aggregate"]["excluded_runs"], ["p1"])
        # The aggregate median is computed over the kept run only.
        self.assertEqual(next(iter(_groups(payload).values()))["n"], 1)
        self.assertTrue(any("p1: excluded" in w for w in payload["warnings"]))

    def test_requested_beyond_tol_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "off.log"
            # 12.9 GB vs 12.724 GB = 1.38% off, beyond the default 1% tolerance.
            log.write_text(_package_log(P3_STRIDE0, 12_900_000_000), encoding="utf-8")
            payload = ba.build_phase_result(
                [log],
                expect_stride0=P3_STRIDE0,
                expect_requested_bytes=P3_REQUESTED,
                package_policy="exclude",
            )
        row = payload["runs"][0]
        self.assertIs(row["package_ok"], False)
        self.assertEqual(row["package_reason"], "requested_mismatch")
        self.assertTrue(row["excluded"])
        # A wider tolerance accepts the same run.
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "off.log"
            log.write_text(_package_log(P3_STRIDE0, 12_900_000_000), encoding="utf-8")
            loose = ba.build_phase_result(
                [log],
                expect_stride0=P3_STRIDE0,
                expect_requested_bytes=P3_REQUESTED,
                requested_tol=0.02,
            )
        self.assertIs(loose["runs"][0]["package_ok"], True)

    def test_same_stride0_different_requested_is_ambiguous_not_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.log"
            b = Path(td) / "b.log"
            a.write_text(_package_log(P3_STRIDE0, P1_REQUESTED), encoding="utf-8")
            b.write_text(_package_log(P3_STRIDE0, P3_REQUESTED), encoding="utf-8")
            payload = ba.build_phase_result([a, b], package_policy="exclude")
        self.assertEqual(payload["package"]["excluded_runs"], [])
        for row in payload["runs"]:
            self.assertIsNone(row["package_ok"])
            self.assertEqual(row["package_reason"], "ambiguous_same_stride0")
            self.assertFalse(row["excluded"])
        self.assertEqual(sorted(payload["package"]["ambiguous_runs"]), ["a", "b"])
        self.assertEqual(next(iter(_groups(payload).values()))["n"], 2)

    def test_442368_alone_is_ambiguous_never_match(self):
        # T-0320: stride0=442368 is shared by P1 and P2; an explicit stride0
        # expectation must not silently "verify" it without a byte check.
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "p12.log"
            log.write_text(_package_log(P1_STRIDE0, P1_REQUESTED), encoding="utf-8")
            payload = ba.build_phase_result(
                [log], expect_stride0=P1_STRIDE0, package_policy="exclude"
            )
        row = payload["runs"][0]
        self.assertIsNone(row["package_ok"])
        self.assertEqual(row["package_reason"], "ambiguous_same_stride0")
        self.assertFalse(row["excluded"])
        # An explicit byte expectation does disambiguate it.
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "p12.log"
            log.write_text(_package_log(P1_STRIDE0, P1_REQUESTED), encoding="utf-8")
            payload = ba.build_phase_result(
                [log],
                expect_stride0=P1_STRIDE0,
                expect_requested_bytes=P1_REQUESTED,
                package_policy="exclude",
            )
        self.assertIs(payload["runs"][0]["package_ok"], True)
        self.assertEqual(payload["runs"][0]["package_reason"], "match")

    def test_missing_identity_is_missing_and_kept(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "noid.log"
            log.write_text(PHASE_LOG, encoding="utf-8")
            payload = ba.build_phase_result(
                [log], expect_stride0=P3_STRIDE0, package_policy="exclude"
            )
        row = payload["runs"][0]
        self.assertIsNone(row["package_ok"])
        self.assertEqual(row["package_reason"], "missing")
        self.assertFalse(row["excluded"])
        self.assertEqual(payload["package"]["excluded_runs"], [])
        self.assertIn("noid", payload["package"]["missing_runs"])

    def test_conflicting_stride0_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "conflict.log"
            text = (
                _identity_lines(stride0=P3_STRIDE0)
                + _identity_lines(stride0=P1_STRIDE0)
                + PHASE_LOG
            )
            log.write_text(text, encoding="utf-8")
            payload = ba.build_phase_result([log], package_policy="exclude")
        row = payload["runs"][0]
        self.assertIs(row["package_ok"], False)
        self.assertEqual(row["package_reason"], "conflicting_stride0")
        self.assertTrue(row["excluded"])
        self.assertIsNone(row["stride0"])
        self.assertEqual(row["stride0_values"], [P1_STRIDE0, P3_STRIDE0])

    def test_package_auto_majority_excludes_minority(self):
        with tempfile.TemporaryDirectory() as td:
            logs = []
            for i in (1, 2):
                path = Path(td) / f"p3-r{i}.log"
                path.write_text(_package_log(P3_STRIDE0, P3_REQUESTED), encoding="utf-8")
                logs.append(path)
            stray = Path(td) / "p1.log"
            stray.write_text(_package_log(P1_STRIDE0, P1_REQUESTED), encoding="utf-8")
            logs.append(stray)
            payload = ba.build_phase_result(
                logs, package_auto=True, package_policy="exclude"
            )
        self.assertEqual(payload["package"]["auto_stride0"], P3_STRIDE0)
        self.assertEqual(payload["package"]["excluded_runs"], ["p1"])
        self.assertEqual(next(iter(_groups(payload).values()))["n"], 2)

    def test_package_auto_tie_excludes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.log"
            b = Path(td) / "b.log"
            a.write_text(_package_log(P3_STRIDE0, P3_REQUESTED), encoding="utf-8")
            b.write_text(_package_log(P1_STRIDE0, P1_REQUESTED), encoding="utf-8")
            payload = ba.build_phase_result(
                [a, b], package_auto=True, package_policy="exclude"
            )
        self.assertIsNone(payload["package"]["auto_stride0"])
        self.assertEqual(payload["package"]["excluded_runs"], [])
        self.assertTrue(any("tied" in w for w in payload["warnings"]))

    def test_warn_policy_marks_but_does_not_exclude(self):
        with tempfile.TemporaryDirectory() as td:
            p3 = Path(td) / "p3.log"
            p1 = Path(td) / "p1.log"
            p3.write_text(_package_log(P3_STRIDE0, P3_REQUESTED), encoding="utf-8")
            p1.write_text(_package_log(P1_STRIDE0, P1_REQUESTED), encoding="utf-8")
            payload = ba.build_phase_result([p3, p1], expect_stride0=P3_STRIDE0)
        self.assertEqual(payload["package"]["policy"], "warn")
        self.assertEqual(payload["package"]["excluded_runs"], [])
        rows = {row["run"]: row for row in payload["runs"]}
        self.assertIs(rows["p1"]["package_ok"], False)
        self.assertFalse(rows["p1"]["excluded"])
        self.assertTrue(any("kept by warn policy" in w for w in payload["warnings"]))
        self.assertEqual(next(iter(_groups(payload).values()))["n"], 2)

    def test_old_timing_and_pread_columns_unchanged_by_guard(self):
        old = ba.build_phase_result([FIXTURE])["runs"][0]
        guarded = ba.build_phase_result(
            [FIXTURE], expect_stride0=P3_STRIDE0, package_policy="exclude"
        )["runs"][0]
        for field in ba.PHASE_NUMERIC_FIELDS + [
            "run", "log", "arm", "group", "session", "missing", "parse_warning_count"
        ]:
            self.assertEqual(old[field], guarded[field], field)

    def test_phase_tsv_appends_package_columns_only(self):
        self.assertEqual(ba.PHASE_TSV_FIELDS[:20], OLD_PHASE_TSV_FIELDS)
        self.assertEqual(
            ba.PHASE_TSV_FIELDS[20:],
            ["stride0", "requested_bytes", "package_ok", "package_reason"],
        )
        with tempfile.TemporaryDirectory() as td:
            p3 = Path(td) / "p3.log"
            p1 = Path(td) / "p1.log"
            p3.write_text(_package_log(P3_STRIDE0, P3_REQUESTED), encoding="utf-8")
            p1.write_text(_package_log(P1_STRIDE0, P1_REQUESTED), encoding="utf-8")
            out = Path(td) / "phase.tsv"
            _run_phase([
                "--logs", str(p3), str(p1),
                "--expect-stride0", str(P3_STRIDE0),
                "--package-policy", "exclude",
                "--arm-regex", r"(p[13])",
                "--out-tsv", str(out),
            ])
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.reader(fh, delimiter="\t"))
        self.assertEqual(rows[0], ba.PHASE_TSV_FIELDS)
        by_run = {row[0]: dict(zip(rows[0], row)) for row in rows[1:]}
        self.assertEqual(by_run["p3"]["package_ok"], "true")
        self.assertEqual(by_run["p3"]["package_reason"], "match")
        self.assertEqual(by_run["p1"]["package_ok"], "false")
        self.assertEqual(by_run["p1"]["package_reason"], "stride0_mismatch")
        self.assertEqual(by_run["p3"]["pread_ms"], "3000")

    def test_cli_flags_and_json_package_block(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "p3.log"
            log.write_text(_package_log(), encoding="utf-8")
            out_json = Path(td) / "out.json"
            rc, _ = _run_phase([
                "--logs", str(log),
                "--expect-stride0", str(P3_STRIDE0),
                "--expect-requested-bytes", str(P3_REQUESTED),
                "--requested-tol", "0.01",
                "--package-policy", "exclude",
                "--out-json", str(out_json),
            ])
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(rc, 0)
        self.assertEqual(payload["schema"], "bench_attrib.phase.v1")
        self.assertEqual(payload["package"]["policy"], "exclude")
        self.assertEqual(payload["package"]["expect_stride0"], P3_STRIDE0)
        self.assertEqual(payload["package"]["expect_requested_bytes"], P3_REQUESTED)
        self.assertEqual(payload["package"]["requested_tol"], 0.01)
        self.assertEqual(payload["aggregate"]["excluded_runs"], [])
        self.assertEqual(payload["runs"][0]["package_reason"], "match")

    def test_package_help_documents_flags(self):
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
            ba.main(["phase", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = buf.getvalue()
        for flag in (
            "--expect-stride0", "--expect-requested-bytes", "--requested-tol",
            "--package-policy", "--package-auto",
        ):
            self.assertIn(flag, text)
        self.assertIn("ambiguous_same_stride0", text)


class LegacyRegressionTests(unittest.TestCase):
    def test_legacy_tsv_columns_and_values_unchanged(self):
        self.assertEqual(ba.TSV_FIELDS, LEGACY_TSV_FIELDS)
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "legacy.log"
            log.write_text(PHASE_LOG, encoding="utf-8")
            out = Path(td) / "legacy.tsv"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = ba.main(["--logs", str(log), "--out", str(out)])
            self.assertEqual(rc, 0)
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.reader(fh, delimiter="\t"))
        self.assertEqual(rows[0], LEGACY_TSV_FIELDS)
        row = dict(zip(rows[0], rows[1]))
        self.assertEqual(row["eval_ms"], "4623.54")
        self.assertEqual(row["calls_g"], "34")
        self.assertEqual(row["acceptance"], "0.90196")
        self.assertEqual(row["tg_tps"], "27.44")  # legacy keeps the snapshot
        self.assertEqual(len(rows[1]), len(LEGACY_TSV_FIELDS))


if __name__ == "__main__":
    unittest.main()
