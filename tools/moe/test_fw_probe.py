#!/usr/bin/env python3
"""Unit tests for tools/moe/fw_probe.py (stdlib unittest, no device).

Run:
    python3 -m unittest tools.moe.test_fw_probe -v
"""

import csv
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

import fw_probe as F  # noqa: E402

TOOL = os.path.join(HERE, "fw_probe.py")

NEW_T = (
    "[fw] ctx=T gtype=1 rows=4 ms=12.345 moe=1.234 readids=0.123 "
    "install=0.456 writeids=0.078 call=7 first=0 reused=1 splits=41"
)
NEW_D = (
    "[fw] ctx=D gtype=1 rows=3 ms=3.456 moe=0.000 readids=0.000 "
    "install=0.000 writeids=0.000 call=7 first=0 reused=0 splits=1"
)
OLD = (
    "[fw] ctx=T gtype=1 rows=2 ms=5.500 moe=0.500 readids=0.050 "
    "install=0.200 writeids=0.040"
)
LEGACY_NO_MOE = "[fw] ctx=T gtype=1 rows=1 ms=1.000"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.counter = 0

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def write(self, text, name=None, newline="\n"):
        self.counter += 1
        path = self.path(name or "log-%d.log" % self.counter)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, TOOL, *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def run_json(self, *args):
        out = self.path("out-%d.json" % self.counter)
        self.counter += 1
        proc = self.run_cli(*args, "--json", out)
        data = None
        if os.path.exists(out):
            with open(out, encoding="utf-8") as handle:
                data = json.load(handle)
        return proc, data, out


class ParseLineTest(Base):
    def test_new_target_line_all_fields(self):
        row = F.parse_line(NEW_T, "t.log", 3)
        self.assertIsNotNone(row)
        self.assertEqual(row["file"], "t.log")
        self.assertEqual(row["line"], 3)
        self.assertEqual(row["ctx"], "T")
        self.assertEqual(row["gtype"], 1)
        self.assertEqual(row["rows"], 4)
        self.assertEqual(row["ms"], 12.345)
        self.assertEqual(row["moe"], 1.234)
        self.assertEqual(row["readids"], 0.123)
        self.assertEqual(row["install"], 0.456)
        self.assertEqual(row["writeids"], 0.078)
        self.assertEqual(row["call"], 7)
        self.assertEqual(row["first"], 0)
        self.assertEqual(row["reused"], 1)
        self.assertEqual(row["splits"], 41)
        self.assertEqual(row["extra"], {})
        self.assertEqual(row["missing_fields"], [])

    def test_new_draft_line_zeros_are_values(self):
        row = F.parse_line(NEW_D, "d.log", 1)
        self.assertEqual(row["ctx"], "D")
        self.assertEqual(row["moe"], 0.0)
        self.assertIsNotNone(row["moe"])
        self.assertEqual(row["reused"], 0)
        self.assertEqual(row["first"], 0)
        self.assertEqual(row["missing_fields"], [])

    def test_old_format_missing_new_fields_and_first_null(self):
        row = F.parse_line(OLD, "old.log", 2)
        self.assertIsNone(row["call"])
        self.assertIsNone(row["first"])
        self.assertIsNone(row["reused"])
        self.assertIsNone(row["splits"])
        self.assertEqual(
            row["missing_fields"], ["call", "first", "reused", "splits"]
        )
        # Values that *are* present still survive.
        self.assertEqual(row["moe"], 0.5)

    def test_splits_minus_one_is_not_missing(self):
        line = (
            "[fw] ctx=D gtype=1 rows=1 ms=1.000 moe=0.000 readids=0.000 "
            "install=0.000 writeids=0.000 call=1 first=1 reused=0 splits=-1"
        )
        row = F.parse_line(line, "x.log", 1)
        self.assertEqual(row["splits"], -1)
        self.assertNotIn("splits", row["missing_fields"])

        row = F.parse_line(OLD, "x.log", 1)
        self.assertIsNone(row["splits"])
        self.assertIn("splits", row["missing_fields"])

    def test_unknown_fields_are_raw_in_extra(self):
        line = (
            "[fw] ctx=T gtype=1 rows=4 ms=2.000 moe=0.100 readids=0.010 "
            "install=0.020 writeids=0.030 miss=44 mib=1.25 uniq=3 "
            "graph_uid=abc call=2 first=0 reused=1 splits=41"
        )
        row = F.parse_line(line, "x.log", 1)
        self.assertEqual(
            row["extra"], {"miss": "44", "mib": "1.25", "uniq": "3", "graph_uid": "abc"}
        )
        self.assertEqual(row["missing_fields"], [])
        self.assertEqual(row["call"], 2)

    def test_prefix_and_timestamp(self):
        line = "2026-11-23T13:00:00+08:00 llama-cli: " + NEW_T
        row = F.parse_line(line, "x.log", 1)
        self.assertIsNotNone(row)
        self.assertEqual(row["ctx"], "T")
        self.assertEqual(row["ms"], 12.345)
        # Anchor is the first [fw], even with a prefix that contains brackets.
        row = F.parse_line("[[noise]] [fw] ctx=D rows=1 ms=0.500", "x.log", 1)
        self.assertEqual(row["ctx"], "D")
        self.assertEqual(row["rows"], 1)

    def test_bad_required_fields_are_skipped(self):
        self.assertIsNone(F.parse_line("[fw] ctx=X gtype=1 rows=4 ms=1.0", "x", 1))
        self.assertIsNone(F.parse_line("[fw] ctx=T gtype=1 rows=4 ms=abc", "x", 1))
        self.assertIsNone(F.parse_line("[fw] ctx=T gtype=1 rows=4", "x", 1))
        self.assertIsNone(F.parse_line("[fw] ctx=T gtype=1 rows=4.0 ms=1.0", "x", 1))
        self.assertIsNone(F.parse_line("no probe here", "x", 1))
        self.assertIsNone(F.parse_line("[fw] ctx=T rows=4 ms=nan", "x", 1))

    def test_invalid_optional_value_becomes_missing_not_zero(self):
        line = "[fw] ctx=T gtype=1 rows=1 ms=1.000 moe=abc first=2 call=1 splits=-2"
        row = F.parse_line(line, "x", 1)
        self.assertIsNone(row["moe"])
        self.assertIsNone(row["first"])
        self.assertIsNone(row["splits"])
        self.assertIn("moe", row["missing_fields"])
        self.assertIn("first", row["missing_fields"])
        self.assertIn("splits", row["missing_fields"])
        # call=1 is valid and therefore still present.
        self.assertEqual(row["call"], 1)

    def test_legacy_line_without_optional_fields(self):
        row = F.parse_line(LEGACY_NO_MOE, "x", 1)
        self.assertEqual(
            row["missing_fields"],
            ["moe", "readids", "install", "writeids", "call", "first", "reused", "splits"],
        )
        self.assertIsNone(row["moe"])
        self.assertIsNone(row["first"])


class AggregateTest(Base):
    def test_empty_groups_use_null_not_zero(self):
        groups = F.aggregate([])
        self.assertEqual(tuple(groups), F.GROUP_KEYS)
        for key in F.GROUP_KEYS:
            group = groups[key]
            self.assertEqual(group["n"], 0)
            self.assertIsNone(group["reused_rate"])
            self.assertIsNone(group["call_min"])
            self.assertIsNone(group["call_max"])
            for field in F.STAT_FIELDS:
                self.assertEqual(
                    group[field], {"p50": None, "p95": None, "mean": None}
                )
            self.assertEqual(group["rows"], {})
            self.assertEqual(group["splits"], {})

    def test_groups_split_ctx_and_first(self):
        rows = [
            F.parse_line(NEW_T, "x", 1),
            F.parse_line(
                "[fw] ctx=T gtype=1 rows=4 ms=20.000 moe=2.000 readids=0.200 "
                "install=0.400 writeids=0.060 call=8 first=0 reused=0 splits=41",
                "x",
                2,
            ),
            F.parse_line(
                "[fw] ctx=D gtype=1 rows=3 ms=3.000 moe=0.000 readids=0.000 "
                "install=0.000 writeids=0.000 call=1 first=1 reused=0 splits=1",
                "x",
                3,
            ),
            F.parse_line(OLD, "x", 4),
        ]
        groups = F.aggregate(rows)
        self.assertEqual(groups["T/steady"]["n"], 2)
        self.assertEqual(groups["T/steady"]["ms"]["p50"], 12.345)
        self.assertEqual(groups["T/steady"]["ms"]["p95"], 20.0)
        self.assertEqual(groups["T/steady"]["ms"]["mean"], round((12.345 + 20.0) / 2, 6))
        self.assertEqual(groups["T/steady"]["reused_rate"], 0.5)
        self.assertEqual(groups["T/steady"]["call_min"], 7)
        self.assertEqual(groups["T/steady"]["call_max"], 8)
        self.assertEqual(groups["T/steady"]["rows"], {"4": 2})
        self.assertEqual(groups["T/steady"]["splits"], {"41": 2})
        self.assertEqual(groups["D/first"]["n"], 1)
        self.assertEqual(groups["D/first"]["splits"], {"1": 1})
        self.assertEqual(groups["T/first"]["n"], 0)
        self.assertEqual(groups["D/steady"]["n"], 0)
        # The legacy row (first=None) is parsed but belongs to no group.
        self.assertEqual(sum(group["n"] for group in groups.values()), 3)

    def test_distribution_sorts_numerically_and_keeps_minus_one(self):
        rows = [
            F.parse_line(
                "[fw] ctx=T gtype=1 rows=10 ms=1.0 moe=0 readids=0 install=0 "
                "writeids=0 call=3 first=0 reused=1 splits=41",
                "x",
                1,
            ),
            F.parse_line(
                "[fw] ctx=T gtype=1 rows=2 ms=2.0 moe=0 readids=0 install=0 "
                "writeids=0 call=2 first=0 reused=1 splits=-1",
                "x",
                2,
            ),
            F.parse_line(
                "[fw] ctx=T gtype=1 rows=10 ms=3.0 moe=0 readids=0 install=0 "
                "writeids=0 call=4 first=0 reused=1 splits=41",
                "x",
                3,
            ),
        ]
        group = F.aggregate(rows)["T/steady"]
        self.assertEqual(group["rows"], {"2": 1, "10": 2})
        self.assertEqual(group["splits"], {"-1": 1, "41": 2})
        self.assertEqual(group["reused_rate"], 1.0)


class PercentileTest(Base):
    def test_nearest_index_boundaries(self):
        self.assertIsNone(F.pctl([], 0.5))
        self.assertEqual(F.pctl([5.0], 0.5), 5.0)
        self.assertEqual(F.pctl([5.0], 0.95), 5.0)
        self.assertEqual(F.pctl([10.0, 20.0], 0.50), 10.0)
        self.assertEqual(F.pctl([10.0, 20.0], 0.95), 20.0)
        self.assertEqual(F.pctl([10, 20, 30, 40], 0.50), 30)
        self.assertEqual(F.pctl([10, 20, 30, 40], 0.95), 40)

    def test_stats_single_and_pair(self):
        self.assertEqual(F._stats([7.5]), {"p50": 7.5, "p95": 7.5, "mean": 7.5})
        self.assertEqual(
            F._stats([1.0, 2.0]), {"p50": 1.0, "p95": 2.0, "mean": 1.5}
        )
        self.assertEqual(
            F._stats([None]), {"p50": None, "p95": None, "mean": None}
        )


class CliTest(Base):
    def fixture(self):
        body = "\n".join(
            [
                NEW_T,
                NEW_D,
                OLD,
                "[fw] ctx=T gtype=1 rows=1 ms=644.000 moe=0.000 readids=0.000 "
                "install=0.000 writeids=0.000 call=1 first=1 reused=0 splits=-1",
                "[fw] ctx=X gtype=1 rows=4 ms=1.0",
                "plain log line",
                "",
            ]
        )
        return self.write(body, "mixed.log")

    def test_json_structure_roundtrip(self):
        path = self.fixture()
        proc, data, out = self.run_json(path, "--ctx", "T")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(out))
        self.assertEqual(data["schema"], "fw-probe.v1")
        self.assertEqual(
            set(data), {"schema", "inputs", "calls", "groups", "totals"}
        )
        self.assertEqual(tuple(data["groups"]), F.GROUP_KEYS)
        self.assertEqual(data["totals"]["lines"], 6)
        self.assertEqual(data["totals"]["parsed"], 4)
        self.assertEqual(data["totals"]["skipped"], 2)
        self.assertEqual(len(data["inputs"]), 1)
        with open(path, "rb") as handle:
            expected_sha = hashlib.sha256(handle.read()).hexdigest()
        self.assertEqual(data["inputs"][0]["sha256"], expected_sha)
        self.assertEqual(data["inputs"][0]["path"], path)
        # --ctx T applies to calls/groups, while totals keep raw parse counts.
        self.assertEqual([row["ctx"] for row in data["calls"]], ["T", "T", "T"])
        self.assertEqual(data["groups"]["T/steady"]["n"], 1)
        self.assertEqual(data["groups"]["T/first"]["n"], 1)
        self.assertEqual(data["groups"]["D/steady"]["n"], 0)
        legacy = [row for row in data["calls"] if row["first"] is None][0]
        self.assertIsNone(legacy["call"])
        self.assertIn("first", legacy["missing_fields"])

    def test_tsv_columns_and_empty_missing_values(self):
        path = self.fixture()
        tsv = self.path("out.tsv")
        proc = self.run_cli(path, "--tsv", tsv)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(tsv, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        self.assertEqual(tuple(rows[0]), F.TSV_COLUMNS)
        self.assertEqual(len(rows), 5)  # header + 4 parsed rows
        by_line = {int(row[1]): row for row in rows[1:]}
        self.assertEqual(by_line[1][2], "T")
        self.assertEqual(by_line[1][5], "12.345")
        # Legacy row has no call/first/reused/splits: empty strings, never 0.
        self.assertEqual(by_line[3][10:], ["", "", "", ""])
        # T/first splits=-1 is a value and must be rendered.
        self.assertEqual(by_line[4][13], "-1")

    def test_first_and_steady_filters_count_legacy_rows(self):
        path = self.fixture()
        proc, data, _ = self.run_json(path, "--first")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(data["totals"]["excluded_no_first"], 1)
        self.assertEqual([row["first"] for row in data["calls"]], [1])
        self.assertEqual(data["groups"]["T/first"]["n"], 1)

        proc, data, _ = self.run_json(path, "--steady")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(data["totals"]["excluded_no_first"], 1)
        self.assertEqual(len(data["calls"]), 2)
        self.assertTrue(all(row["first"] == 0 for row in data["calls"]))

    def test_ctx_filter_excludes_legacy_from_excluded_count(self):
        path = self.fixture()
        proc, data, _ = self.run_json(path, "--ctx", "D", "--first")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(data["totals"]["excluded_no_first"], 0)
        self.assertEqual(data["calls"], [])

    def test_min_ms_filter(self):
        path = self.fixture()
        proc, data, _ = self.run_json(path, "--min-ms", "10")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sorted(row["ms"] for row in data["calls"]), [12.345, 644.0])
        self.assertEqual(data["groups"]["T/first"]["n"], 1)
        self.assertEqual(data["groups"]["T/steady"]["n"], 1)

    def test_summary_prints_exactly_four_group_lines(self):
        path = self.fixture()
        proc = self.run_cli(path, "--summary")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        for key, line in zip(F.GROUP_KEYS, lines):
            self.assertTrue(line.startswith(key), line)
            self.assertIn("n=", line)

    def test_default_output_has_totals_and_groups(self):
        path = self.fixture()
        proc = self.run_cli(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("parsed=4", proc.stdout)
        self.assertIn("skipped=2", proc.stdout)
        for key in F.GROUP_KEYS:
            self.assertIn(key, proc.stdout)

    def test_multi_file_line_numbers_and_counts(self):
        first = self.write(NEW_T + "\n" + "noise\n", "one.log")
        second = self.write(NEW_D + "\n" + OLD, "two.log")
        proc, data, _ = self.run_json(first, second)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(data["totals"]["lines"], 4)
        self.assertEqual(data["totals"]["parsed"], 3)
        self.assertEqual(data["totals"]["skipped"], 1)
        self.assertEqual([info["path"] for info in data["inputs"]], [first, second])
        self.assertEqual([info["lines"] for info in data["inputs"]], [2, 2])
        self.assertEqual([row["file"] for row in data["calls"]], [first, second, second])
        self.assertEqual([row["line"] for row in data["calls"]], [1, 1, 2])

    def test_no_final_newline_still_parsed(self):
        path = self.path("no-newline.log")
        with open(path, "wb") as handle:
            handle.write(NEW_T.encode("utf-8"))
        proc, data, _ = self.run_json(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(data["totals"]["lines"], 1)
        self.assertEqual(data["totals"]["parsed"], 1)

    def test_empty_input_is_nonzero_without_traceback(self):
        path = self.write("", "empty.log")
        out = self.path("empty.json")
        proc = self.run_cli(path, "--json", out)
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("no [fw] line parsed", proc.stderr)
        with open(out, encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["totals"]["parsed"], 0)
        self.assertEqual(data["groups"]["T/first"]["n"], 0)

    def test_bad_file_is_nonzero_without_traceback(self):
        path = self.write("nothing but ordinary stderr\nstill nothing\n", "bad.log")
        proc = self.run_cli(path)
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)

    def test_missing_file_is_clean_error(self):
        proc = self.run_cli(self.path("does-not-exist.log"))
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("cannot read", proc.stderr)

    def test_strict_fails_on_skipped_lines(self):
        path = self.write(NEW_T + "\nnot a probe line\n", "mixed-bad.log")
        proc = self.run_cli(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        proc = self.run_cli(path, "--strict")
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("skipped under --strict", proc.stderr)

    def test_missing_optional_fields_are_null_not_zero_in_json(self):
        line = (
            "[fw] ctx=T gtype=1 rows=1 ms=1.000 call=1 first=1 reused=0 splits=-1"
        )
        path = self.write(line + "\n", "sparse.log")
        proc, data, out = self.run_json(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = data["calls"][0]
        for field in ("moe", "readids", "install", "writeids"):
            self.assertIsNone(row[field])
        self.assertEqual(row["splits"], -1)
        self.assertEqual(
            row["missing_fields"],
            ["moe", "readids", "install", "writeids"],
        )
        # Missing values must be JSON null, not numeric 0.
        with open(out, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn('"moe": null', text)
        self.assertNotIn('"moe": 0', text)


if __name__ == "__main__":
    unittest.main()
