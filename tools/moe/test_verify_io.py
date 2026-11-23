#!/usr/bin/env python3
"""Unit tests for tools/moe/verify_io.py (stdlib unittest, no device).

Run:
    python3 -m unittest tools/moe/test_verify_io.py -v
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import verify_io as V  # noqa: E402

TOOL = os.path.join(HERE, "verify_io.py")

# Optional external regression fixture produced by T-0360.  The path is
# machine-local; when it is absent the regression test is skipped rather than
# failing an unrelated checkout.
QUEUE_HOME = os.environ.get("DPSK_QUEUE_HOME", "/home/yydh/.local/state/dpsk-q2")
T0360_FIXTURE = os.path.join(
    QUEUE_HOME,
    "runs/T-0360/run-20261123T123009-w2-1/artifacts/regression/verify-io.json",
)


def forward(fid, a, b, phase="verify", rows=4):
    return {"id": fid, "phase": phase, "a": a, "b": b, "rows": rows}


def event(fid, a, b, session="HTP0", layer=0, module="FFN 专家",
          placement_error_us=None):
    row = {
        "fw": fid,
        "phase": "verify",
        "a": a,
        "b": b,
        "us": b - a,
        "session": session,
        "layer": layer,
        "module": module,
    }
    if placement_error_us is not None:
        row["placement_error_us"] = placement_error_us
    return row


def read(a, b, nbytes=1000, fw_ids=(1,), prefetch=False, layer=0,
         nonresident=None):
    row = {
        "a": a,
        "b": b,
        "bytes": nbytes,
        "fw_ids": list(fw_ids),
        "prefetch": prefetch,
        "layer": layer,
    }
    if nonresident is not None:
        row["nonresident_bytes"] = nonresident
    return row


def capture(forwards, events=None, pagecache=None, **extra):
    data = {
        "forwards": forwards,
        "events": events or [],
        "pagecache": pagecache or [],
    }
    data.update(extra)
    return data


def four_cycle_capture(placement_error_us=10):
    """Four verify cycles with asymmetric DSP/read overlap.

    Hand-computed base totals (us): wall 3800, dsp 2300, read 1800,
    overlap 1100, dsp_only 1200, read_only 700, neither 800.
    """
    forwards = [
        forward(1, 0, 1000),
        forward(2, 2000, 3200),
        forward(3, 4000, 4600),
        forward(4, 6000, 7000),
    ]
    events = [
        event(1, 0, 400, placement_error_us=placement_error_us),
        event(1, 700, 900, placement_error_us=placement_error_us),
        event(2, 2000, 2600, placement_error_us=placement_error_us),
        event(3, 4000, 4300, placement_error_us=placement_error_us),
        event(4, 6000, 6800, placement_error_us=placement_error_us),
    ]
    reads = [
        read(200, 600, nbytes=1_000_000, fw_ids=(1,)),
        read(800, 1000, nbytes=1_000_000, fw_ids=(1,), prefetch=True),
        read(2400, 3000, nbytes=1_000_000, fw_ids=(2,)),
        read(6000, 6600, nbytes=1_000_000, fw_ids=(4,)),
    ]
    return capture(forwards, events, reads)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._counter = 0

    def write(self, data, name=None):
        self._counter += 1
        path = os.path.join(
            self.tmp.name, name or f"profile-{self._counter}.json"
        )
        with open(path, "w", encoding="utf-8") as handle:
            if isinstance(data, str):
                handle.write(data)
            else:
                json.dump(data, handle)
        return path

    def analyze(self, data, **kwargs):
        return V.analyze_profile(self.write(data), **kwargs)


class FourSegmentTests(Base):
    def test_exact_four_segments(self):
        # wall 100, dsp [0,40), read [20,80)
        report = self.analyze(
            capture(
                [forward(1, 1000, 1100)],
                [event(1, 1000, 1040)],
                [read(1020, 1080, nbytes=1_000_000)],
            )
        )
        cycle = report["cycles"][0]
        self.assertEqual(cycle["wall_us"], 100)
        self.assertEqual(cycle["dsp_us"], 40)
        self.assertEqual(cycle["demand_us"], 60)
        self.assertEqual(cycle["prefetch_us"], 0)
        self.assertEqual(cycle["read_union_us"], 60)
        self.assertEqual(cycle["overlap_us"], 20)
        self.assertEqual(cycle["dsp_only_us"], 20)
        self.assertEqual(cycle["read_only_us"], 40)
        self.assertEqual(cycle["neither_us"], 20)
        self.assertEqual(
            cycle["overlap_us"]
            + cycle["dsp_only_us"]
            + cycle["read_only_us"]
            + cycle["neither_us"],
            cycle["wall_us"],
        )
        self.assertAlmostEqual(cycle["demand_share"], 1.0)
        self.assertAlmostEqual(cycle["eff_gbs"], 1_000_000 / 60 * 1e-3)
        self.assertEqual(cycle["clipped_us"], 0)

    def test_read_covers_dsp(self):
        report = self.analyze(
            capture(
                [forward(1, 0, 100)],
                [event(1, 0, 40)],
                [read(0, 100, fw_ids=(1,))],
            )
        )
        cycle = report["cycles"][0]
        self.assertEqual(cycle["overlap_us"], 40)
        self.assertEqual(cycle["dsp_only_us"], 0)
        self.assertEqual(cycle["read_only_us"], 60)
        self.assertEqual(cycle["neither_us"], 0)

    def test_no_read(self):
        report = self.analyze(
            capture([forward(1, 0, 100)], [event(1, 0, 40)], [])
        )
        summary = report["summary"]
        self.assertEqual(summary["read_union_us"], 0)
        self.assertEqual(summary["read_only_pct"], 0)
        self.assertEqual(summary["neither_us"], 60)
        self.assertEqual(summary["dsp_only_us"], 40)
        self.assertEqual(summary["demand_share"], 0)
        self.assertEqual(summary["eff_gbs"], 0)

    def test_clipped_read_still_partitions_wall(self):
        # read runs 50 us past the cycle; the in-cycle part is 50 us.
        report = self.analyze(
            capture(
                [forward(1, 1000, 1100)],
                [event(1, 1000, 1020)],
                [read(1050, 1150, nbytes=500_000, nonresident=500_000)],
            )
        )
        cycle = report["cycles"][0]
        self.assertEqual(cycle["read_union_raw_us"], 100)
        self.assertEqual(cycle["read_union_us"], 50)
        self.assertEqual(cycle["clipped_us"], 50)
        self.assertEqual(cycle["dsp_only_us"], 20)
        self.assertEqual(cycle["read_only_us"], 50)
        self.assertEqual(cycle["neither_us"], 30)
        self.assertEqual(
            cycle["overlap_us"]
            + cycle["dsp_only_us"]
            + cycle["read_only_us"]
            + cycle["neither_us"],
            cycle["wall_us"],
        )
        self.assertAlmostEqual(cycle["eff_gbs"], 10.0)

    def test_session_filter(self):
        data = capture(
            [forward(1, 1000, 1100)],
            [event(1, 1000, 1040, session="HTP0"),
             event(1, 1060, 1100, session="HTP1")],
            [],
        )
        both = self.analyze(data)
        only0 = self.analyze(data, session="HTP0")
        self.assertEqual(both["cycles"][0]["dsp_us"], 80)
        self.assertEqual(only0["cycles"][0]["dsp_us"], 40)
        self.assertEqual(only0["cycles"][0]["dsp_only_us"], 40)
        self.assertEqual(only0["cycles"][0]["neither_us"], 60)


class BadInputTests(Base):
    def test_empty_profile_is_zero_not_division(self):
        report = self.analyze({})
        summary = report["summary"]
        self.assertEqual(summary["cycles"], 0)
        self.assertEqual(summary["wall_us"], 0)
        self.assertEqual(summary["eff_gbs"], 0)
        self.assertEqual(summary["demand_share"], 0)
        self.assertGreater(summary["skipped"], 0)

    def test_missing_pagecache(self):
        # Drop the pagecache key entirely.
        report = self.analyze(
            {
                "forwards": [forward(1, 0, 100)],
                "events": [event(1, 0, 10)],
            }
        )
        self.assertEqual(report["summary"]["read_union_us"], 0)
        self.assertEqual(report["summary"]["read_only_pct"], 0)
        self.assertGreater(report["summary"]["skipped"], 0)
        self.assertIn("missing_or_bad_pagecache", report["summary"]["skipped_reasons"])

    def test_bad_entries_do_not_raise(self):
        data = capture(
            [forward(1, 0, 100)],
            [event(1, 0, 10), {"fw": 1, "a": "x", "b": 20}],
            [
                read(20, 30),
                {"a": 30, "b": "nope", "fw_ids": [1]},
                {"a": 40, "b": 50, "fw_ids": []},
                {"a": 50, "b": 60, "fw_ids": [1]},  # no bytes at all
            ],
        )
        report = self.analyze(data)
        self.assertEqual(report["summary"]["cycles"], 1)
        self.assertGreaterEqual(report["summary"]["skipped"], 3)
        self.assertGreater(report["summary"]["read_union_us"], 0)

    def test_bad_json_file(self):
        path = self.write("{this is not json\n")
        report = V.analyze_profile(path)
        self.assertEqual(report["cycles"], [])
        self.assertEqual(report["summary"]["cycles"], 0)
        self.assertGreater(report["skipped"], 0)
        self.assertIn("error", report)


class MultiProfileTests(Base):
    def test_each_profile_owns_its_cycles(self):
        p1 = self.write(
            capture([forward(1, 0, 100)], [event(1, 0, 10)],
                    [read(20, 50)]),
            "p1.json",
        )
        p2 = self.write(
            capture(
                [forward(7, 0, 100), forward(8, 100, 300)],
                [event(7, 0, 30), event(8, 100, 160)],
                [read(20, 60, fw_ids=(7,)), read(120, 220, fw_ids=(8,))],
            ),
            "p2.json",
        )
        report = V.analyze_profiles([p1, p2])
        self.assertEqual(len(report["profiles"]), 2)
        first, second = report["profiles"]
        self.assertEqual(first["summary"]["cycles"], 1)
        self.assertEqual(first["summary"]["wall_us"], 100)
        self.assertEqual(second["summary"]["cycles"], 2)
        self.assertEqual(second["summary"]["wall_us"], 300)
        self.assertEqual([c["fw"] for c in second["cycles"]], [7, 8])
        self.assertEqual(first["cycles"][0]["dsp_us"], 10)
        self.assertEqual(second["cycles"][0]["dsp_us"], 30)


class PerLayerTests(Base):
    def test_percentiles_hand_checked(self):
        data = capture(
            [forward(1, 0, 100_000)],
            [event(1, 0, 1000)],
            [
                read(0, 10_000, layer=2),
                read(10_000, 30_000, layer=2),
                read(30_000, 60_000, layer=2),
                read(60_000, 100_000, layer=2),
                read(0, 5_000, layer=2, prefetch=True),
            ],
        )
        report = self.analyze(data, per_layer=True)
        rows = report["summary"]["per_layer"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["layer"], 2)
        self.assertEqual(row["n_demand"], 4)
        # sorted latencies [10,20,30,40] ms; T-0340 nearest-index percentile
        self.assertEqual(row["demand_p50_ms"], 30)
        self.assertEqual(row["demand_p90_ms"], 40)
        self.assertEqual(row["demand_max_ms"], 40)
        self.assertEqual(row["n_prefetch"], 1)
        self.assertEqual(row["prefetch_p50_ms"], 5)
        self.assertEqual(row["prefetch_p90_ms"], 5)
        self.assertEqual(row["prefetch_max_ms"], 5)

    def test_per_layer_absent_by_default(self):
        report = self.analyze(
            capture([forward(1, 0, 10)], [event(1, 0, 1)], [])
        )
        self.assertNotIn("per_layer", report["summary"])


class DefaultCompatTests(Base):
    def test_default_summary_has_no_new_keys(self):
        report = self.analyze(four_cycle_capture())
        summary = report["summary"]
        self.assertEqual(V.VERSION, 1)
        self.assertEqual(
            set(summary), set(V._empty_summary()) | {"skipped_reasons"}
        )
        self.assertNotIn("bootstrap", summary)
        self.assertNotIn("placement_sensitivity", summary)
        self.assertEqual(
            set(report["cycles"][0]), set(V.Cycle.__dataclass_fields__)
        )
        self.assertNotIn("placement_error_us", report["cycles"][0])

    def test_default_values_are_unchanged_hand_computed(self):
        summary = self.analyze(four_cycle_capture())["summary"]
        self.assertEqual(summary["wall_us"], 3800)
        self.assertEqual(summary["dsp_us"], 2300)
        self.assertEqual(summary["read_union_us"], 1800)
        self.assertEqual(summary["overlap_us"], 1100)
        self.assertEqual(summary["dsp_only_us"], 1200)
        self.assertEqual(summary["read_only_us"], 700)
        self.assertEqual(summary["neither_us"], 800)
        self.assertAlmostEqual(summary["read_only_pct"], 700 / 3800 * 100)
        self.assertAlmostEqual(summary["demand_share"], 1600 / 1800)
        self.assertEqual(summary["reads"], 4)
        self.assertEqual(summary["prefetch_reads"], 1)
        self.assertEqual(summary["read_bytes"], 4_000_000)

    @unittest.skipUnless(
        os.path.exists(T0360_FIXTURE), "T-0360 regression fixture not present"
    )
    def test_default_matches_t0360_fixture_old_keys(self):
        with open(T0360_FIXTURE, encoding="utf-8") as handle:
            fixture = json.load(handle)
        for expected in fixture["profiles"]:
            current = V.analyze_profile(expected["path"], per_layer=True)
            for key, value in expected["summary"].items():
                if key == "per_layer":
                    continue
                self.assertEqual(current["summary"][key], value, key)
            self.assertEqual(current["cycles"], expected["cycles"])


class BootstrapTests(Base):
    def test_same_seed_reproducible_different_seed_changes_metrics(self):
        path = self.write(four_cycle_capture())
        a = V.analyze_profile(path, bootstrap=200, seed=7)
        b = V.analyze_profile(path, bootstrap=200, seed=7)
        c = V.analyze_profile(path, bootstrap=200, seed=8)
        self.assertEqual(a["summary"]["bootstrap"], b["summary"]["bootstrap"])
        self.assertNotEqual(
            a["summary"]["bootstrap"]["metrics"],
            c["summary"]["bootstrap"]["metrics"],
        )

    def test_bootstrap_shape_and_ordering(self):
        path = self.write(four_cycle_capture())
        base = V.analyze_profile(path)["summary"]
        summary = V.analyze_profile(path, bootstrap=200, seed=20261123)["summary"]
        boot = summary["bootstrap"]
        self.assertEqual(boot["n"], 200)
        self.assertEqual(boot["seed"], 20261123)
        self.assertEqual(boot["n_eff"], 4)
        self.assertTrue(boot["estimable"])
        for name in V.BOOTSTRAP_METRICS + ("read_only_us_median",):
            metric = boot["metrics"][name]
            self.assertEqual(set(metric), {"p2_5", "p50", "p97_5"})
            self.assertLessEqual(metric["p2_5"], metric["p50"])
            self.assertLessEqual(metric["p50"], metric["p97_5"])
        # The point estimates themselves must not change.
        for key, value in base.items():
            self.assertEqual(summary[key], value, key)

    def test_single_cycle_is_degenerate_but_present(self):
        path = self.write(
            capture([forward(1, 0, 100)], [event(1, 0, 40)], [read(20, 80)])
        )
        boot = V.analyze_profile(path, bootstrap=200)["summary"]["bootstrap"]
        self.assertEqual(boot["n_eff"], 1)
        self.assertFalse(boot["estimable"])
        metric = boot["metrics"]["read_only_pct"]
        self.assertEqual(metric["p2_5"], metric["p50"])
        self.assertEqual(metric["p50"], metric["p97_5"])

    def test_empty_profile_reports_no_estimate(self):
        boot = V.analyze_profile(self.write({}), bootstrap=200)["summary"][
            "bootstrap"
        ]
        self.assertEqual(boot["n_eff"], 0)
        self.assertFalse(boot["estimable"])
        self.assertEqual(boot["metrics"], {})


class PlacementSensitivityTests(Base):
    def test_wide_and_narrow_directions(self):
        path = self.write(four_cycle_capture(placement_error_us=10))
        base = V.analyze_profile(path)["summary"]
        placement = V.analyze_profile(path, placement_sensitivity=True)[
            "summary"
        ]["placement_sensitivity"]
        wide = placement["wide"]
        narrow = placement["narrow"]
        # Widening the DSP intervals can only add overlap -> dsp up and
        # read_only (= read - dsp) down; narrowing does the opposite.  This is
        # the mathematical direction of the definitions in the module docstring
        # (it is the inverse of a literal reading of the task's test bullet).
        self.assertGreater(wide["dsp_busy_pct"], base["dsp_busy_pct"])
        self.assertGreater(wide["overlap_pct"], base["overlap_pct"])
        self.assertLess(wide["read_only_pct"], base["read_only_pct"])
        self.assertLess(narrow["dsp_busy_pct"], base["dsp_busy_pct"])
        self.assertLess(narrow["overlap_pct"], base["overlap_pct"])
        self.assertGreater(narrow["read_only_pct"], base["read_only_pct"])
        for record in (wide, narrow):
            for name, delta in record["delta_vs_base"].items():
                self.assertAlmostEqual(
                    delta, record[name] - base[name], places=12
                )

    def test_missing_placement_error_counts_and_does_not_crash(self):
        data = four_cycle_capture(placement_error_us=None)
        # Keep one event with a placement error so the counter is non-zero but
        # the remaining events exercise the missing-field fallback.
        data["events"][0]["placement_error_us"] = 10
        path = self.write(data)
        default = V.analyze_profile(path)["summary"]
        self.assertNotIn(
            "missing_placement_error_us", default["skipped_reasons"]
        )
        summary = V.analyze_profile(path, placement_sensitivity=True)["summary"]
        self.assertEqual(
            summary["skipped_reasons"]["missing_placement_error_us"], 4
        )
        self.assertEqual(summary["placement_sensitivity"]["wide"]["mode"], "wide")
        self.assertEqual(
            summary["placement_sensitivity"]["narrow"]["mode"], "narrow"
        )


class CliV2Tests(Base):
    def test_cli_rejects_bad_bootstrap_and_seed_before_reading(self):
        path = self.write(four_cycle_capture())
        cases = (
            ["--bootstrap", "99"],
            ["--bootstrap", "200.5"],
            ["--seed", "abc"],
        )
        for extra in cases:
            proc = subprocess.run(
                [sys.executable, TOOL, "--profile", path] + extra,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2, extra)
            self.assertIn("usage:", proc.stderr)

    def test_cli_bootstrap_json_and_human_lines(self):
        path = self.write(four_cycle_capture(placement_error_us=10))
        out = os.path.join(self.tmp.name, "v2.json")
        proc = subprocess.run(
            [
                sys.executable,
                TOOL,
                "--profile",
                path,
                "--bootstrap",
                "200",
                "--placement-sensitivity",
                "--out-json",
                out,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out, encoding="utf-8") as handle:
            report = json.load(handle)
        summary = report["profiles"][0]["summary"]
        self.assertEqual(summary["bootstrap"]["n"], 200)
        self.assertEqual(summary["bootstrap"]["seed"], 20261123)
        self.assertIn("wide", summary["placement_sensitivity"])
        self.assertIn("narrow", summary["placement_sensitivity"])
        self.assertIn("bootstrap     : read_only p50", proc.stdout)
        self.assertIn("placement     : wide/narrow read_only", proc.stdout)
        self.assertIn("p2.5", proc.stdout)
        self.assertIn("p97.5", proc.stdout)

    def test_tsv_byte_identical_with_and_without_new_flags(self):
        path = self.write(four_cycle_capture(placement_error_us=10))
        plain_dir = os.path.join(self.tmp.name, "plain")
        flags_dir = os.path.join(self.tmp.name, "flags")
        for out_dir, extra in (
            (plain_dir, []),
            (flags_dir, ["--bootstrap", "200", "--placement-sensitivity"]),
        ):
            proc = subprocess.run(
                [sys.executable, TOOL, "--profile", path, "--out-dir", out_dir]
                + extra,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(os.path.join(plain_dir, "verify-io.tsv"), "rb") as handle:
            plain = handle.read()
        with open(os.path.join(flags_dir, "verify-io.tsv"), "rb") as handle:
            flags = handle.read()
        self.assertEqual(plain, flags)
        header = plain.split(b"\n", 1)[0].decode("utf-8").split("\t")
        self.assertEqual(header, V.TSV_COLUMNS)


class CliTests(Base):
    def _sha256(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def test_cli_outputs_and_input_sha256_unchanged(self):
        path = self.write(
            capture(
                [forward(1, 0, 100)],
                [event(1, 0, 40)],
                [read(20, 80, layer=3)],
            )
        )
        before = self._sha256(path)
        out_dir = os.path.join(self.tmp.name, "out")
        proc = subprocess.run(
            [sys.executable, TOOL, "--profile", path, "--out-dir", out_dir,
             "--per-layer"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("read_only 40.00%", proc.stdout)
        self.assertEqual(self._sha256(path), before)

        tsv = os.path.join(out_dir, "verify-io.tsv")
        js = os.path.join(out_dir, "verify-io.json")
        layer_tsv = os.path.join(out_dir, "verify-io-per-layer.tsv")
        for output in (tsv, js, layer_tsv):
            self.assertTrue(os.path.exists(output), output)

        with open(tsv, encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            row = handle.readline().rstrip("\n").split("\t")
        self.assertEqual(header, V.TSV_COLUMNS)
        self.assertEqual(row[0], path)
        self.assertEqual(row[3], "0.1")  # wall_ms (synthetic wall is 100 us)
        self.assertEqual(row[12], "1")  # demand_share
        self.assertEqual(row[14], "0")  # clipped_us

        with open(js, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["version"], 1)
        self.assertEqual(len(report["profiles"]), 1)
        self.assertIn("overlap_us", report["definitions"])
        self.assertEqual(report["profiles"][0]["summary"]["cycles"], 1)

    def test_cli_survives_bad_json(self):
        path = self.write("{broken")
        out = os.path.join(self.tmp.name, "bad.json")
        proc = subprocess.run(
            [sys.executable, TOOL, "--profile", path, "--out-json", out],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["profiles"][0]["summary"]["skipped"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
