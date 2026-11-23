#!/usr/bin/env python3
"""Unit tests for tools/moe/analyze_host_bubbles.py (stdlib unittest).

Run:
    python3 -m unittest tools/moe/test_analyze_host_bubbles.py -v
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analyze_host_bubbles as bubbles  # noqa: E402

MS = 1_000_000
BASE = 1_000_000_000


def event(name, ns, epoch, layer, source="other", **fields):
    row = {
        "event": name,
        "ns": ns,
        "tid": fields.pop("tid", 7),
        "epoch": epoch,
        "layer": layer,
        "rows": fields.pop("rows", 4),
        "expert": fields.pop("expert", -1),
        "slot": fields.pop("slot", -1),
        "source": source,
        "job_id": fields.pop("job_id", 1),
        "result": fields.pop("result", 0),
        "bytes": fields.pop("bytes", 0),
        "domain": fields.pop("domain", "window"),
        "io_id": fields.pop("io_id", 0),
        "reason": fields.pop("reason", "none"),
    }
    row.update(fields)
    return row


def synthetic_trace_lines():
    """Header + 3 epochs x 2 layers, covering all six bubble event types."""
    lines = [json.dumps({
        "schema": "llama.moe.handoff.v2",
        "clock": "CLOCK_MONOTONIC",
        "unit": "ns",
        "capacity": 262144,
        "attempted": 60,
        "written": 60,
        "lost": 0,
    })]

    def layer_block(epoch, layer, t0, map_ns, depth, wait_ns, arm_ns, decision,
                    demand_ns, prefetch_ns, demand_read_ns=MS, prefetch_read_ns=MS):
        rows = []
        rows.append(event("MAP_BEGIN", t0, epoch, layer, source="graph"))
        rows.append(event("MAP_END", t0 + map_ns, epoch, layer, source="graph",
                          map_ns=map_ns, bytes=map_ns))
        rows.append(event("QUEUE_ENQUEUE", t0 + map_ns + 100_000, epoch, layer,
                          source="prefetch", queue_depth=depth))
        rows.append(event("WORKER_WAKE", t0 + map_ns + 200_000, epoch, layer,
                          source="worker", wait_ns=wait_ns, queue_depth=depth))
        rows.append(event("ARM_BEGIN", t0 + map_ns + 300_000, epoch, layer,
                          source="prefetch", layer_next=layer))
        rows.append(event("ARM_END", t0 + map_ns + 300_000 + arm_ns, epoch, layer,
                          source="prefetch", arm_ns=arm_ns, decision=decision,
                          bytes=arm_ns, reason=decision))
        rows.append(event("PREAD_BEGIN", t0 + map_ns + demand_ns, epoch, layer,
                          source="demand", io_id=layer * 10 + 1))
        rows.append(event("PREAD_DONE", t0 + map_ns + demand_ns + demand_read_ns, epoch, layer,
                          source="demand", io_id=layer * 10 + 1))
        rows.append(event("PREAD_BEGIN", t0 + map_ns + prefetch_ns, epoch, layer,
                          source="prefetch", io_id=layer * 10 + 2))
        rows.append(event("PREAD_DONE", t0 + map_ns + prefetch_ns + prefetch_read_ns, epoch, layer,
                          source="prefetch", io_id=layer * 10 + 2))
        return rows

    # Epoch 1: hand-computed expectations in the test below.
    lines += [json.dumps(row) for row in layer_block(
        epoch=1, layer=1, t0=BASE, map_ns=5 * MS, depth=1, wait_ns=1 * MS,
        arm_ns=700_000, decision="accept", demand_ns=2 * MS, prefetch_ns=3_500_000,
        demand_read_ns=1 * MS, prefetch_read_ns=1 * MS)]
    lines += [json.dumps(row) for row in layer_block(
        epoch=1, layer=2, t0=BASE + 10 * MS, map_ns=2 * MS, depth=2, wait_ns=2 * MS,
        arm_ns=400_000, decision="skip", demand_ns=1_600_000, prefetch_ns=3 * MS,
        demand_read_ns=1 * MS, prefetch_read_ns=1 * MS)]

    for epoch in (2, 3):
        for layer in (1, 2):
            t0 = BASE + epoch * 100 * MS + layer * 20 * MS
            lines += [json.dumps(row) for row in layer_block(
                epoch=epoch, layer=layer, t0=t0, map_ns=MS, depth=layer,
                wait_ns=MS, arm_ns=500_000, decision="accept",
                demand_ns=MS, prefetch_ns=2_500_000)]
    return lines


class AnalyzeHostBubblesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        lines = synthetic_trace_lines()
        self.trace = os.path.join(self.tmp.name, "trace-A.jsonl")
        with open(self.trace, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            # three deliberately bad lines: malformed JSON, wrong schema, missing layer
            handle.write("{this is not json\n")
            handle.write(json.dumps({
                "schema": "llama.moe.handoff.v1", "event": "MAP_END",
                "ns": 1, "tid": 7, "epoch": 9, "layer": 1,
            }) + "\n")
            handle.write(json.dumps({
                "event": "PREAD_BEGIN", "ns": 2, "tid": 7, "epoch": 3,
            }) + "\n")
        self.json_path = os.path.join(self.tmp.name, "bubbles.json")
        self.tsv_path = os.path.join(self.tmp.name, "bubbles.tsv")
        rc = bubbles.main([
            self.trace, "--labels", "A",
            "--out-json", self.json_path, "--out-tsv", self.tsv_path,
        ])
        self.assertEqual(rc, 0)
        with open(self.json_path, encoding="utf-8") as handle:
            self.data = json.load(handle)

    def row(self, epoch):
        for row in self.data["by_verify"]:
            if row["label"] == "A" and row["epoch"] == epoch:
                return row
        self.fail("missing epoch %s" % epoch)

    def approx(self, actual, expected):
        self.assertAlmostEqual(actual, expected, delta=1e-6)

    def test_epoch1_manual_sums(self):
        row = self.row(1)
        # wall = last PREAD_DONE (B+16ms) - MAP_BEGIN (B+0)
        self.approx(row["verify_wall_ms"], 16.0)
        self.approx(row["map_ms"], 7.0)             # 5 ms + 2 ms
        self.approx(row["map_to_read_ms"], 3.6)     # 2.0 + 1.6
        self.approx(row["read_to_read_ms"], 5.0)    # 0.5 + 4.1 + 0.4
        self.approx(row["wake_ms"], 3.0)            # 1 ms + 2 ms
        self.approx(row["arm_ms"], 1.1)             # 0.7 + 0.4
        self.assertEqual(row["max_queue_depth"], 2)
        self.assertEqual(row["n_map"], 2)
        self.assertEqual(row["n_read"], 4)
        self.assertEqual(row["n_wake"], 2)
        self.assertEqual(row["n_bad"], 0)
        self.approx(row["map"]["p50_ms"], 3.5)      # (5 + 2) / 2
        self.approx(row["map"]["p95_ms"], 4.85)     # linear: 2 + 0.95 * 3

    def test_demand_prefetch_classification(self):
        row = self.row(1)
        self.assertEqual(row["reads_by_source"], {"demand": 2, "prefetch": 2, "unknown": 0})
        transitions = row["read_transitions"]
        self.assertIn("demand->prefetch", transitions)
        self.assertIn("prefetch->demand", transitions)
        self.assertEqual(transitions["demand->prefetch"]["count"], 2)
        self.approx(transitions["demand->prefetch"]["sum_ms"], 0.9)
        self.assertEqual(transitions["prefetch->demand"]["count"], 1)
        self.approx(transitions["prefetch->demand"]["sum_ms"], 4.1)

    def test_wake_by_tid_and_arm_by_decision(self):
        row = self.row(1)
        self.assertEqual(len(row["wake_by_tid"]), 1)
        self.assertEqual(row["wake_by_tid"][0]["tid"], 7)
        self.approx(row["wake_by_tid"][0]["sum_ms"], 3.0)
        self.approx(row["wake_by_tid"][0]["p50_ms"], 1.5)
        self.assertEqual(sorted(row["arm_by_decision"]), ["accept", "skip"])
        self.approx(row["arm_by_decision"]["accept"]["sum_ms"], 0.7)
        self.approx(row["arm_by_decision"]["skip"]["sum_ms"], 0.4)

    def test_bucket_layer_cross_layer_read_gap_attribution(self):
        bucket = {(row["epoch"], row["layer"]): row for row in self.data["by_bucket"]}
        self.approx(bucket[(1, 1)]["map"]["sum_ms"], 5.0)
        self.approx(bucket[(1, 1)]["map_to_read"]["sum_ms"], 2.0)
        # 0.5 ms into layer 1's prefetch + 4.1 ms into layer 2's demand read
        self.approx(bucket[(1, 1)]["read_to_read"]["sum_ms"], 4.6)
        self.approx(bucket[(1, 2)]["read_to_read"]["sum_ms"], 0.4)

    def test_bad_lines_counted(self):
        counts = self.data["counts"]
        self.assertEqual(counts["bad_json"], 1)
        self.assertEqual(counts["bad_schema"], 1)
        self.assertEqual(counts["missing_field"], 1)
        self.assertEqual(counts["events"], 60)
        self.assertEqual(counts["map_begin"], 6)
        self.assertEqual(counts["map_end"], 6)
        self.assertEqual(counts["worker_wake"], 6)
        self.assertEqual(counts["queue_enqueue"], 6)
        self.assertEqual(counts["arm_begin"], 6)
        self.assertEqual(counts["arm_end"], 6)

    def test_tsv_columns_and_values(self):
        with open(self.tsv_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0].split("\t"), list(bubbles.TSV_COLUMNS))
        header = lines[0].split("\t")
        row = lines[1].split("\t")
        record = dict(zip(header, row))
        self.assertEqual(record["label"], "A")
        self.assertEqual(record["epoch"], "1")
        self.approx(float(record["map_ms"]), 7.0)
        self.approx(float(record["map_to_read_ms"]), 3.6)
        self.approx(float(record["read_to_read_ms"]), 5.0)
        self.approx(float(record["wake_ms"]), 3.0)
        self.approx(float(record["arm_ms"]), 1.1)
        self.assertEqual(record["max_queue_depth"], "2")
        self.assertEqual(record["n_map"], "2")
        self.assertEqual(record["n_read"], "4")
        self.assertEqual(record["n_wake"], "2")
        self.assertEqual(record["n_bad"], "0")

    def test_read_source_origin_fallback(self):
        event_row = bubbles.Event(name="PREAD_BEGIN", ns=1, tid=7, epoch=1, layer=1,
                                  source="other", extra={"origin": "prefetch", "domain": "window"})
        self.assertEqual(bubbles._read_source(event_row), "prefetch")
        event_row.extra["origin"] = "demand"
        self.assertEqual(bubbles._read_source(event_row), "demand")
        event_row.extra["origin"] = "other"
        self.assertEqual(bubbles._read_source(event_row), "unknown")

    def test_empty_file_is_all_zero(self):
        empty = os.path.join(self.tmp.name, "empty.jsonl")
        open(empty, "w", encoding="utf-8").close()
        out = os.path.join(self.tmp.name, "empty.json")
        tsv = os.path.join(self.tmp.name, "empty.tsv")
        self.assertEqual(bubbles.main([empty, "--labels", "E", "--out-json", out, "--out-tsv", tsv]), 0)
        with open(out, encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["counts"]["total_lines"], 0)
        self.assertEqual(data["counts"]["events"], 0)
        self.assertEqual(data["by_verify"], [])
        self.assertEqual(data["totals"]["verifies"], 0)
        self.approx(data["totals"]["map_ms"], 0.0)
        self.approx(data["totals"]["map_to_read_ms"], 0.0)
        with open(tsv, encoding="utf-8") as handle:
            self.assertEqual(handle.read().splitlines(), ["\t".join(bubbles.TSV_COLUMNS)])

    def test_json_is_deterministic_and_loadable(self):
        again = os.path.join(self.tmp.name, "again.json")
        self.assertEqual(bubbles.main([
            self.trace, "--labels", "A", "--out-json", again, "--bucket", "epoch",
        ]), 0)
        with open(again, encoding="utf-8") as handle:
            second = json.load(handle)
        self.assertEqual(second["bucket"], "epoch")
        self.assertEqual(second["labels"], ["A"])
        # keys are stable and sorted by json.dumps(sort_keys=True)
        with open(again, encoding="utf-8") as handle:
            raw = handle.read()
        self.assertTrue(raw.startswith("{\n  \"bucket\""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
