#!/usr/bin/env python3
"""Unit tests for tools/moe/analyze_rtt.py (stdlib only)."""

import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analyze_rtt  # noqa: E402


RTT_VALUES = [1000 * (i + 1) for i in range(12)]
BASE = 1_000_000_000
WRITE_NS = 10


def make_row(index, n_ops=4, oppoll=0, role="main"):
    rtt = RTT_VALUES[index]
    t_write = BASE + index * 1_000_000
    return {
        "schema": analyze_rtt.SCHEMA,
        "session": 0,
        "role": role,
        "seq": index + 1,
        "t_write_ns": t_write,
        "t_write_ret_ns": t_write + WRITE_NS,
        "t_first_poll_ns": t_write + 20,
        "t_done_ns": t_write + rtt,
        "rtt_ns": rtt,
        "write_ns": WRITE_NS,
        "poll_wait_ns": rtt - 20,
        "n_ops": n_ops,
        "n_ops_rsp": 0,
        "rsp_status": 0,
        "oppoll": oppoll,
        "dropped": False,
    }


def make_v2_row(index, graph_uid=42, cache_hit=False, prep_ns=1234,
                nodes=41, **kwargs):
    row = make_row(index, **kwargs)
    row["schema"] = analyze_rtt.SCHEMA_V2
    row.update({
        "graph_uid": graph_uid,
        "graph_cache_hit": cache_hit,
        "graph_prep_ns": prep_ns,
        "graph_nodes": nodes,
    })
    return row


def write_lines(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row) + "\n")


class PercentileTests(unittest.TestCase):
    def test_known_series(self):
        values = RTT_VALUES
        self.assertEqual(analyze_rtt.percentile(values, 0.50), 6500)
        self.assertEqual(analyze_rtt.percentile(values, 0.95), 11450)
        self.assertEqual(analyze_rtt.percentile(values, 0.99), 11890)
        self.assertEqual(analyze_rtt.percentile([], 0.50), None)
        self.assertEqual(analyze_rtt.percentile([42], 0.95), 42)

    def test_unsorted_input(self):
        values = list(reversed(RTT_VALUES))
        self.assertEqual(analyze_rtt.percentile(values, 0.50), 6500)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_percentiles_and_fields(self):
        path = self.path("trace.jsonl")
        write_lines(path, [make_row(i) for i in range(12)])
        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["schema"], analyze_rtt.AGG_SCHEMA)
        self.assertEqual(report["totals"], {
            "traces": 1, "lines": 12, "valid": 12, "bad": 0, "dropped": 0})
        self.assertEqual(len(report["groups"]), 1)
        group = report["groups"][0]
        self.assertEqual(group["label"], "A")
        self.assertEqual(group["role"], "main")
        self.assertEqual(group["count"], 12)
        self.assertEqual(group["timed_count"], 12)
        for key, expected in (("p50", 6500), ("p95", 11450),
                              ("p99", 11890), ("mean", 6500), ("max", 12000)):
            self.assertLessEqual(abs(group["rtt_ns"][key] - expected), 1,
                                 "%s: %r != %r" % (key, group["rtt_ns"][key], expected))
        self.assertEqual(group["write_ns"]["p50"], WRITE_NS)
        self.assertEqual(group["write_ns"]["p95"], WRITE_NS)
        self.assertLessEqual(abs(group["poll_wait_ns"]["p50"] - 6480), 1)
        self.assertEqual(group["n_ops"], {"4": 12})
        self.assertEqual(group["oppoll"], {"0": 12})
        graph = group["graph"]
        self.assertEqual(graph["records"], 0)
        self.assertEqual(graph["cache_hits"], 0)
        self.assertIsNone(graph["cache_hit_rate"])
        self.assertEqual(graph["unique_graph_uids"], 0)
        self.assertIsNone(graph["prep_ns"]["p50"])
        self.assertIsNone(graph["nodes"]["p50"])
        self.assertEqual(graph["nodes"]["distribution"], {})

    def test_bad_schema_json_and_missing_field(self):
        path = self.path("bad.jsonl")
        rows = [make_row(i) for i in range(12)]
        bad_schema = make_row(0)
        bad_schema["schema"] = "ggml.hex.rtt.v0"
        missing = make_row(1)
        del missing["rtt_ns"]
        write_lines(path, rows + [bad_schema, "{not json", missing])

        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["totals"]["lines"], 15)
        self.assertEqual(report["totals"]["valid"], 12)
        self.assertEqual(report["totals"]["bad"], 3)
        reasons = report["traces"][0]["bad_reasons"]
        self.assertTrue(any(key.startswith("bad-schema") for key in reasons))
        self.assertTrue(any(key.startswith("bad-json") for key in reasons))
        self.assertTrue(any(key.startswith("missing-field") for key in reasons))
        self.assertEqual(len(report["traces"][0]["bad_lines"]), 3)
        self.assertEqual(report["groups"][0]["count"], 12)

    def test_inconsistent_derived_field_is_bad(self):
        path = self.path("inconsistent.jsonl")
        row = make_row(0)
        row["rtt_ns"] += 1
        write_lines(path, [row])
        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["totals"]["valid"], 0)
        self.assertEqual(report["totals"]["bad"], 1)
        self.assertIn("inconsistent: rtt_ns",
                      report["traces"][0]["bad_reasons"])

    def test_v2_graph_aggregation(self):
        path = self.path("v2.jsonl")
        write_lines(path, [make_v2_row(0)])
        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["input_schemas"], {
            analyze_rtt.SCHEMA: 0, analyze_rtt.SCHEMA_V2: 1})
        self.assertEqual(report["traces"][0]["schemas"], {
            analyze_rtt.SCHEMA: 0, analyze_rtt.SCHEMA_V2: 1})
        group = report["groups"][0]
        self.assertEqual(group["count"], 1)
        self.assertEqual(group["timed_count"], 1)
        graph = group["graph"]
        self.assertEqual(graph["records"], 1)
        self.assertEqual(graph["cache_hits"], 0)
        self.assertEqual(graph["cache_hit_rate"], 0.0)
        self.assertEqual(graph["unique_graph_uids"], 1)
        self.assertEqual(graph["prep_ns"]["p50"], 1234)
        self.assertEqual(graph["prep_ns"]["p95"], 1234)
        self.assertEqual(graph["prep_ns"]["mean"], 1234)
        self.assertEqual(graph["prep_ns"]["max"], 1234)
        self.assertEqual(graph["nodes"]["p50"], 41)
        self.assertEqual(graph["nodes"]["max"], 41)
        self.assertEqual(graph["nodes"]["distribution"], {"41": 1})

    def test_mixed_v1_v2_excludes_v1_from_graph(self):
        path = self.path("mixed.jsonl")
        write_lines(path, [make_row(0), make_v2_row(1, graph_uid=42,
                                                    cache_hit=True,
                                                    prep_ns=500, nodes=12)])
        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["input_schemas"], {
            analyze_rtt.SCHEMA: 1, analyze_rtt.SCHEMA_V2: 1})
        self.assertEqual(report["traces"][0]["schemas"], {
            analyze_rtt.SCHEMA: 1, analyze_rtt.SCHEMA_V2: 1})
        group = report["groups"][0]
        self.assertEqual(group["count"], 2)
        self.assertEqual(group["timed_count"], 2)
        self.assertEqual(group["graph"]["records"], 1)
        self.assertEqual(group["graph"]["cache_hits"], 1)
        self.assertEqual(group["graph"]["cache_hit_rate"], 1.0)
        self.assertEqual(group["graph"]["prep_ns"]["p50"], 500)

    def test_v2_bad_or_missing_graph_fields(self):
        cases = (
            ("missing-field", "graph_prep_ns", None, "graph_prep_ns"),
            ("bad-field", "graph_cache_hit", "true", "graph_cache_hit"),
            ("bad-field", "graph_uid", -1, "graph_uid"),
            ("bad-field", "graph_nodes", 1 << 32, "graph_nodes"),
        )
        for expected, field, value, needle in cases:
            with self.subTest(field=field, value=value):
                row = make_v2_row(0)
                if value is None:
                    del row[field]
                else:
                    row[field] = value
                path = self.path("bad-%s.jsonl" % field)
                write_lines(path, [row])
                with contextlib.redirect_stderr(io.StringIO()):
                    report = analyze_rtt.build_report([("A", path)])
                self.assertEqual(report["totals"]["valid"], 0)
                self.assertEqual(report["totals"]["bad"], 1)
                reasons = report["traces"][0]["bad_reasons"]
                self.assertTrue(any(reason.startswith(expected) and needle in reason
                                    for reason in reasons), reasons)

    def test_v2_dropped_graph_null_does_not_count(self):
        path = self.path("v2-dropped.jsonl")
        dropped = {
            "schema": analyze_rtt.SCHEMA_V2,
            "session": 0,
            "role": "main",
            "seq": 99,
            "t_write_ns": None,
            "t_write_ret_ns": None,
            "t_first_poll_ns": None,
            "t_done_ns": None,
            "rtt_ns": None,
            "write_ns": None,
            "poll_wait_ns": None,
            "n_ops": None,
            "n_ops_rsp": 5,
            "rsp_status": 2,
            "oppoll": 0,
            "dropped": True,
            "graph_uid": None,
            "graph_cache_hit": None,
            "graph_prep_ns": None,
            "graph_nodes": None,
        }
        write_lines(path, [make_v2_row(0), dropped])
        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["totals"]["valid"], 2)
        self.assertEqual(report["totals"]["dropped"], 1)
        group = report["groups"][0]
        self.assertEqual(group["count"], 2)
        self.assertEqual(group["dropped"], 1)
        self.assertEqual(group["graph"]["records"], 1)
        self.assertEqual(group["graph"]["unique_graph_uids"], 1)

    def test_bucket_n_ops(self):
        path = self.path("bucket.jsonl")
        rows = [make_row(i, n_ops=4) for i in range(6)]
        rows += [make_row(i, n_ops=8) for i in range(6, 12)]
        write_lines(path, rows)

        flat = analyze_rtt.build_report([("A", path)])
        self.assertEqual(len(flat["groups"]), 1)
        self.assertIsNone(flat["bucket_by"])

        bucketed = analyze_rtt.build_report([("A", path)], bucket_by="n_ops")
        self.assertEqual(bucketed["bucket_by"], "n_ops")
        self.assertEqual([g["bucket"] for g in bucketed["groups"]], [4, 8])
        self.assertEqual([g["count"] for g in bucketed["groups"]], [6, 6])
        self.assertEqual(bucketed["groups"][0]["n_ops"], {"4": 6})
        self.assertEqual(bucketed["groups"][1]["n_ops"], {"8": 6})

    def test_empty_file(self):
        path = self.path("empty.jsonl")
        open(path, "w", encoding="utf-8").close()
        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["totals"]["lines"], 0)
        self.assertEqual(report["totals"]["valid"], 0)
        self.assertEqual(report["groups"], [])
        self.assertEqual(report["traces"][0]["bad"], 0)

    def test_dropped_minimal_line(self):
        path = self.path("dropped.jsonl")
        write_lines(path, [make_row(i) for i in range(3)]
                    + ['{"schema":"ggml.hex.rtt.v1","seq":99,"dropped":true}'])
        report = analyze_rtt.build_report([("A", path)])
        self.assertEqual(report["totals"]["valid"], 4)
        self.assertEqual(report["totals"]["dropped"], 1)
        group = report["groups"][0]
        self.assertEqual(group["count"], 4)
        self.assertEqual(group["timed_count"], 3)
        self.assertEqual(group["dropped"], 1)

    def test_deterministic_group_order(self):
        pa = self.path("a.jsonl")
        pb = self.path("b.jsonl")
        write_lines(pa, [make_row(0, role="sub")])
        write_lines(pb, [make_row(1, role="main")])
        r1 = analyze_rtt.build_report([("B", pb), ("A", pa)])
        r2 = analyze_rtt.build_report([("B", pb), ("A", pa)])
        self.assertEqual(json.dumps(r1, sort_keys=True), json.dumps(r2, sort_keys=True))
        self.assertEqual([g["label"] for g in r1["groups"]], ["A", "B"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_cli_json_and_tsv(self):
        trace_a = self.path("trace-A.jsonl")
        trace_b = self.path("trace-B.jsonl")
        write_lines(trace_a, [make_row(i, oppoll=0) for i in range(12)])
        write_lines(trace_b, [make_row(i, oppoll=1) for i in range(12)])
        out_json = self.path("rtt.json")
        out_tsv = self.path("rtt.tsv")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = analyze_rtt.main([trace_a, trace_b, "--labels", "A", "B",
                                   "--out-json", out_json, "--out-tsv", out_tsv,
                                   "--json"])
        self.assertEqual(rc, 0)

        parsed_stdout = json.loads(stdout.getvalue())
        with open(out_json, encoding="utf-8") as handle:
            parsed_file = json.load(handle)
        self.assertEqual(parsed_stdout, parsed_file)
        self.assertEqual([g["label"] for g in parsed_file["groups"]], ["A", "B"])
        self.assertEqual(parsed_file["groups"][0]["oppoll"], {"0": 12})
        self.assertEqual(parsed_file["groups"][1]["oppoll"], {"1": 12})

        with open(out_tsv, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0].split("\t"), list(analyze_rtt.TSV_COLUMNS))
        self.assertEqual(len(lines), 3)  # header + A + B
        self.assertEqual(lines[1].split("\t")[0], "A")

    def test_tsv_graph_columns_readable_by_csv(self):
        old_columns = (
            "label", "role", "bucket", "count", "timed_count", "dropped",
            "rtt_p50_ns", "rtt_p95_ns", "rtt_p99_ns", "rtt_mean_ns",
            "rtt_max_ns", "write_p50_ns", "write_p95_ns",
            "poll_wait_p50_ns", "poll_wait_p95_ns", "n_ops_dist", "oppoll_dist",
        )
        self.assertEqual(
            list(analyze_rtt.TSV_COLUMNS)[:len(old_columns)], list(old_columns))
        trace = self.path("mixed.jsonl")
        write_lines(trace, [make_row(0),
                            make_v2_row(1, graph_uid=42, cache_hit=False,
                                        prep_ns=1234, nodes=41)])
        out_tsv = self.path("rtt.tsv")
        with contextlib.redirect_stderr(io.StringIO()):
            rc = analyze_rtt.main([trace, "--out-tsv", out_tsv, "--quiet"])
        self.assertEqual(rc, 0)
        with open(out_tsv, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
        self.assertEqual(reader.fieldnames, list(analyze_rtt.TSV_COLUMNS))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["label"], "mixed")
        self.assertEqual(row["count"], "2")
        self.assertEqual(row["graph_records"], "1")
        self.assertEqual(row["graph_cache_hits"], "0")
        self.assertEqual(row["graph_cache_hit_rate"], "0.0")
        self.assertEqual(row["graph_unique_uids"], "1")
        self.assertEqual(row["graph_prep_p50_ns"], "1234")
        self.assertEqual(row["graph_nodes_dist"], "41:1")

    def test_cli_summary_shows_schema_and_hit_rate(self):
        trace = self.path("v2.jsonl")
        write_lines(trace, [make_v2_row(0, cache_hit=True)])
        out_json = self.path("rtt.json")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = analyze_rtt.main([trace, "--summary", "--out-json", out_json])
        self.assertEqual(rc, 0)
        text = stdout.getvalue()
        self.assertIn("input schemas:", text)
        self.assertIn(analyze_rtt.SCHEMA_V2, text)
        self.assertIn("cache_hit_rate=100.00%", text)

    def test_cli_label_count_mismatch(self):
        trace = self.path("trace.jsonl")
        write_lines(trace, [make_row(0)])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                analyze_rtt.main([trace, "--labels", "A", "B"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
