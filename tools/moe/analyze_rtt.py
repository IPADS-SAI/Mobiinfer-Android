#!/usr/bin/env python3
"""Offline aggregator for ``ggml.hex.rtt.v1`` / ``ggml.hex.rtt.v2`` RTT traces.

The traces are produced by ``GGML_HEXAGON_RTT_TRACE=<path>`` in
``ggml/src/ggml-hexagon/ggml-hexagon.cpp``.  Each line measures the host-observed
interval between submitting a batch with ``dspqueue_write`` and
``dspqueue_read`` returning the matching ``rsp.seq``.  That is an upper bound for
DSP execution: it also contains host submit bookkeeping and queue wait.

v2 (``GGML_HEXAGON_RTT_TRACE_V2=1``) keeps every v1 key and appends
``graph_uid`` / ``graph_cache_hit`` / ``graph_prep_ns`` / ``graph_nodes``.
v1 lines stay valid and are excluded from the graph denominator; mixed files
are counted per line.  ``AGG_SCHEMA`` remains ``ggml.hex.rtt.agg.v1`` and only
gains additive fields.

Usage:
    python3 tools/moe/analyze_rtt.py trace-A.jsonl trace-B.jsonl \
        --labels A B --out-json rtt.json --out-tsv rtt.tsv [--bucket n_ops]

Only the Python standard library is used; no device or model access.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

SCHEMA = "ggml.hex.rtt.v1"
SCHEMA_V2 = "ggml.hex.rtt.v2"
SUPPORTED_SCHEMAS = (SCHEMA, SCHEMA_V2)
AGG_SCHEMA = "ggml.hex.rtt.agg.v1"

TIME_FIELDS = ("t_write_ns", "t_write_ret_ns", "t_first_poll_ns", "t_done_ns")
DERIVED_FIELDS = ("rtt_ns", "write_ns", "poll_wait_ns")
REQUIRED_FIELDS = (
    "schema",
    "session",
    "role",
    "seq",
    *TIME_FIELDS,
    *DERIVED_FIELDS,
    "n_ops",
    "rsp_status",
    "oppoll",
    "dropped",
)
GRAPH_FIELDS = ("graph_uid", "graph_cache_hit", "graph_prep_ns", "graph_nodes")
U64_MAX = (1 << 64) - 1
U32_MAX = (1 << 32) - 1


def _is_int(value) -> bool:
    # bool is a subclass of int; reject it for numeric fields.
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_graph_value(name, value) -> bool:
    if name == "graph_cache_hit":
        return isinstance(value, bool)
    if not _is_int(value) or value < 0:
        return False
    return value <= (U32_MAX if name == "graph_nodes" else U64_MAX)


def _check_graph_fields(rec, dropped):
    """Validate the v2-only fields.  Returns a reason string or None.

    A dropped record is allowed to omit the graph fields or to carry null; the
    writer emits null and the analyzer must not invent 0/false.  When a field is
    present it still has to have the right type.  Non-dropped records must have
    all four fields and none of them may be null.
    """
    if not dropped:
        missing = [name for name in GRAPH_FIELDS if name not in rec]
        if missing:
            return "missing-field: %s" % ",".join(missing)
        for name in GRAPH_FIELDS:
            if not _valid_graph_value(name, rec[name]):
                return "bad-field: %s" % name
        return None

    for name in GRAPH_FIELDS:
        if name not in rec or rec[name] is None:
            continue
        if not _valid_graph_value(name, rec[name]):
            return "bad-field: %s" % name
    return None


def percentile(values, q):
    """Linear-interpolated percentile (same definition as numpy's default).

    ``values`` may be unsorted.  Returns None for an empty input.  The result is
    rounded to the nearest integer nanosecond, so p50 of an even-length series
    can sit between two samples.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return int(ordered[0])
    h = (len(ordered) - 1) * float(q)
    lo = int(math.floor(h))
    hi = min(lo + 1, len(ordered) - 1)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (h - lo)
    return int(math.floor(value + 0.5))


def _mean(values):
    if not values:
        return None
    return int(math.floor(sum(values) / float(len(values)) + 0.5))


def parse_line(raw):
    """Parse and validate one JSONL line.

    Returns ``(record, None)`` on success or ``(None, reason)`` on failure.
    Unknown extra fields are tolerated (the writer may add fields later), but
    every field documented for the row's schema must be present.

    Both ``ggml.hex.rtt.v1`` and ``ggml.hex.rtt.v2`` are accepted.  The parsed
    record keeps the original ``schema`` field and gains an internal
    ``_schema`` marker so the aggregator can exclude v1 rows from graph stats.
    """
    try:
        rec = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, "bad-json: %s" % exc

    if not isinstance(rec, dict):
        return None, "not-an-object"

    schema = rec.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        return None, "bad-schema: %r" % (schema,)
    rec["_schema"] = schema

    # A dropped record may be minimal ({"seq":...,"dropped":true}); it carries
    # no timestamps and must never be turned into a fake sample.
    if rec.get("dropped") is True:
        if "seq" in rec and (not _is_int(rec["seq"]) or rec["seq"] < 0):
            return None, "bad-field: seq"
        if "role" in rec and (not isinstance(rec["role"], str) or not rec["role"]):
            return None, "bad-field: role"
        if schema == SCHEMA_V2:
            reason = _check_graph_fields(rec, dropped=True)
            if reason is not None:
                return None, reason
        return rec, None

    missing = [name for name in REQUIRED_FIELDS if name not in rec]
    if missing:
        return None, "missing-field: %s" % ",".join(missing)

    if not _is_int(rec["session"]):
        return None, "bad-field: session"
    if not _is_int(rec["seq"]) or rec["seq"] < 0:
        return None, "bad-field: seq"
    if not isinstance(rec["role"], str) or not rec["role"]:
        return None, "bad-field: role"
    if not isinstance(rec["dropped"], bool):
        return None, "bad-field: dropped"

    dropped = rec["dropped"]
    if dropped:
        # A dropped record explicitly has no timestamps; never invent any.
        if schema == SCHEMA_V2:
            reason = _check_graph_fields(rec, dropped=True)
            if reason is not None:
                return None, reason
        return rec, None

    for name in TIME_FIELDS:
        if not _is_int(rec[name]):
            return None, "bad-field: %s" % name
    for name in DERIVED_FIELDS:
        if rec[name] is not None and not _is_int(rec[name]):
            return None, "bad-field: %s" % name
    if not _is_int(rec["n_ops"]) or rec["n_ops"] < 0:
        return None, "bad-field: n_ops"
    if not _is_int(rec["rsp_status"]):
        return None, "bad-field: rsp_status"
    if not _is_int(rec["oppoll"]):
        return None, "bad-field: oppoll"

    t_write = rec["t_write_ns"]
    t_write_ret = rec["t_write_ret_ns"]
    t_first_poll = rec["t_first_poll_ns"]
    t_done = rec["t_done_ns"]

    if t_write_ret < t_write:
        return None, "inconsistent: t_write_ret_ns < t_write_ns"
    if t_done < t_write:
        return None, "inconsistent: t_done_ns < t_write_ns"

    expected_rtt = t_done - t_write
    expected_write = t_write_ret - t_write
    if rec["rtt_ns"] != expected_rtt:
        return None, "inconsistent: rtt_ns"
    if rec["write_ns"] != expected_write:
        return None, "inconsistent: write_ns"

    if t_first_poll is None:
        if rec["poll_wait_ns"] is not None:
            return None, "inconsistent: poll_wait_ns without t_first_poll_ns"
    else:
        if t_done < t_first_poll:
            return None, "inconsistent: t_done_ns < t_first_poll_ns"
        if rec["poll_wait_ns"] != t_done - t_first_poll:
            return None, "inconsistent: poll_wait_ns"

    # All v1 rules ran first; only then require the additive v2 fields.
    if schema == SCHEMA_V2:
        reason = _check_graph_fields(rec, dropped=False)
        if reason is not None:
            return None, reason

    return rec, None


class Group:
    __slots__ = ("rtt", "write", "poll", "n_ops", "oppoll", "count",
                 "timed_count", "dropped",
                 "graph_records", "graph_cache_hits", "graph_prep",
                 "graph_nodes", "graph_nodes_values", "graph_uids")

    def __init__(self):
        self.rtt = []
        self.write = []
        self.poll = []
        self.n_ops = {}
        self.oppoll = {}
        self.count = 0
        self.timed_count = 0
        self.dropped = 0
        # v2-only graph attribution.  v1 rows never enter these counters.
        self.graph_records = 0
        self.graph_cache_hits = 0
        self.graph_prep = []
        self.graph_nodes = {}
        self.graph_nodes_values = []
        self.graph_uids = set()

    def add(self, rec):
        self.count += 1
        if rec["dropped"]:
            self.dropped += 1
            return
        self.n_ops[rec["n_ops"]] = self.n_ops.get(rec["n_ops"], 0) + 1
        self.oppoll[rec["oppoll"]] = self.oppoll.get(rec["oppoll"], 0) + 1
        self.timed_count += 1
        self.rtt.append(rec["rtt_ns"])
        self.write.append(rec["write_ns"])
        if rec["poll_wait_ns"] is not None:
            self.poll.append(rec["poll_wait_ns"])
        if rec.get("_schema") == SCHEMA_V2:
            self.graph_records += 1
            if rec["graph_cache_hit"]:
                self.graph_cache_hits += 1
            self.graph_prep.append(rec["graph_prep_ns"])
            nodes = rec["graph_nodes"]
            self.graph_nodes[nodes] = self.graph_nodes.get(nodes, 0) + 1
            self.graph_nodes_values.append(nodes)
            self.graph_uids.add(rec["graph_uid"])

    def merge(self, other):
        self.count += other.count
        self.timed_count += other.timed_count
        self.dropped += other.dropped
        self.rtt.extend(other.rtt)
        self.write.extend(other.write)
        self.poll.extend(other.poll)
        for value, count in other.n_ops.items():
            self.n_ops[value] = self.n_ops.get(value, 0) + count
        for value, count in other.oppoll.items():
            self.oppoll[value] = self.oppoll.get(value, 0) + count
        self.graph_records += other.graph_records
        self.graph_cache_hits += other.graph_cache_hits
        self.graph_prep.extend(other.graph_prep)
        for value, count in other.graph_nodes.items():
            self.graph_nodes[value] = self.graph_nodes.get(value, 0) + count
        self.graph_nodes_values.extend(other.graph_nodes_values)
        self.graph_uids.update(other.graph_uids)

    def as_dict(self, label, role, bucket):
        if self.graph_records:
            cache_hit_rate = self.graph_cache_hits / float(self.graph_records)
        else:
            cache_hit_rate = None
        return {
            "label": label,
            "role": role,
            "bucket": bucket,
            "count": self.count,
            "timed_count": self.timed_count,
            "dropped": self.dropped,
            "rtt_ns": {
                "p50": percentile(self.rtt, 0.50),
                "p95": percentile(self.rtt, 0.95),
                "p99": percentile(self.rtt, 0.99),
                "mean": _mean(self.rtt),
                "max": max(self.rtt) if self.rtt else None,
            },
            "write_ns": {
                "p50": percentile(self.write, 0.50),
                "p95": percentile(self.write, 0.95),
            },
            "poll_wait_ns": {
                "p50": percentile(self.poll, 0.50),
                "p95": percentile(self.poll, 0.95),
            },
            "n_ops": _distribution(self.n_ops),
            "oppoll": _distribution(self.oppoll),
            "graph": {
                "records": self.graph_records,
                "cache_hits": self.graph_cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "unique_graph_uids": len(self.graph_uids),
                "prep_ns": {
                    "p50": percentile(self.graph_prep, 0.50),
                    "p95": percentile(self.graph_prep, 0.95),
                    "mean": _mean(self.graph_prep),
                    "max": max(self.graph_prep) if self.graph_prep else None,
                },
                "nodes": {
                    "p50": percentile(self.graph_nodes_values, 0.50),
                    "p95": percentile(self.graph_nodes_values, 0.95),
                    "mean": _mean(self.graph_nodes_values),
                    "max": max(self.graph_nodes_values) if self.graph_nodes_values else None,
                    "distribution": _distribution(self.graph_nodes),
                },
            },
        }


def _distribution(counter):
    # Sort by numeric key so output is deterministic (JSON keys become strings).
    return {str(key): counter[key] for key in sorted(counter)}


def _label_from_path(path, used):
    base = os.path.basename(path)
    for suffix in (".jsonl", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    label = base or "trace"
    candidate = label
    index = 2
    while candidate in used:
        candidate = "%s#%d" % (label, index)
        index += 1
    used.add(candidate)
    return candidate


def aggregate_file(label, path, bucket_by=None):
    """Read one trace and return (trace_meta, {group_key: Group})."""
    groups = {}
    bad_reasons = {}
    bad_lines = []
    schemas = {schema: 0 for schema in SUPPORTED_SCHEMAS}
    lines = 0
    valid = 0
    dropped = 0

    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit("analyze_rtt: cannot read %s: %s" % (path, exc))

    with handle:
        for lineno, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue  # blank lines are not records
            lines += 1
            rec, reason = parse_line(text)
            if reason is not None:
                bad_reasons[reason] = bad_reasons.get(reason, 0) + 1
                bad_lines.append({"file": path, "line": lineno, "reason": reason})
                print("analyze_rtt: bad line %s:%d: %s" % (path, lineno, reason),
                      file=sys.stderr)
                continue
            valid += 1
            schemas[rec["_schema"]] += 1
            if rec["dropped"]:
                dropped += 1
                # Dropped rows have no n_ops/timestamps: group them by role only
                # (bucket None) instead of inventing values.
                key = (rec.get("role", "main"),)
            elif bucket_by == "n_ops":
                key = (rec["role"], rec["n_ops"])
            else:
                key = (rec["role"],)
            group = groups.get(key)
            if group is None:
                group = groups[key] = Group()
            group.add(rec)

    meta = {
        "label": label,
        "path": path,
        "lines": lines,
        "valid": valid,
        "bad": len(bad_lines),
        "dropped": dropped,
        "bad_reasons": {key: bad_reasons[key] for key in sorted(bad_reasons)},
        "bad_lines": bad_lines,
        "schemas": schemas,
    }
    return meta, groups


def build_report(trace_specs, bucket_by=None):
    """Build the deterministic aggregate document.

    ``trace_specs`` is a list of ``(label, path)`` pairs.
    """
    traces = []
    merged = {}
    for label, path in trace_specs:
        meta, groups = aggregate_file(label, path, bucket_by)
        traces.append(meta)
        for key, group in groups.items():
            dest = merged.get((label,) + key)
            if dest is None:
                merged[(label,) + key] = group
                continue
            dest.merge(group)

    # Sort traces by label; groups by (label, role, bucket value).
    traces.sort(key=lambda item: item["label"])
    out_groups = []
    for key in sorted(merged, key=lambda k: (k[0], k[1], -1 if len(k) == 2 else k[2])):
        label, role = key[0], key[1]
        bucket = key[2] if len(key) > 2 else None
        out_groups.append(merged[key].as_dict(label, role, bucket))

    totals = {
        "traces": len(traces),
        "lines": sum(item["lines"] for item in traces),
        "valid": sum(item["valid"] for item in traces),
        "bad": sum(item["bad"] for item in traces),
        "dropped": sum(item["dropped"] for item in traces),
    }
    input_schemas = {
        schema: sum(item["schemas"].get(schema, 0) for item in traces)
        for schema in SUPPORTED_SCHEMAS
    }
    return {
        "schema": AGG_SCHEMA,
        "bucket_by": bucket_by,
        "input_schemas": input_schemas,
        "traces": traces,
        "totals": totals,
        "groups": out_groups,
    }


TSV_COLUMNS = (
    "label",
    "role",
    "bucket",
    "count",
    "timed_count",
    "dropped",
    "rtt_p50_ns",
    "rtt_p95_ns",
    "rtt_p99_ns",
    "rtt_mean_ns",
    "rtt_max_ns",
    "write_p50_ns",
    "write_p95_ns",
    "poll_wait_p50_ns",
    "poll_wait_p95_ns",
    "n_ops_dist",
    "oppoll_dist",
    "graph_records",
    "graph_cache_hits",
    "graph_cache_hit_rate",
    "graph_unique_uids",
    "graph_prep_p50_ns",
    "graph_prep_p95_ns",
    "graph_nodes_dist",
)


def _dist_text(dist):
    return "|".join("%s:%s" % (key, dist[key]) for key in dist)


def _cell(value):
    return "" if value is None else str(value)


def report_to_tsv(report):
    lines = ["\t".join(TSV_COLUMNS)]
    for group in report["groups"]:
        rtt = group["rtt_ns"]
        write = group["write_ns"]
        poll = group["poll_wait_ns"]
        graph = group["graph"]
        row = (
            group["label"],
            group["role"],
            _cell(group["bucket"]),
            _cell(group["count"]),
            _cell(group["timed_count"]),
            _cell(group["dropped"]),
            _cell(rtt["p50"]),
            _cell(rtt["p95"]),
            _cell(rtt["p99"]),
            _cell(rtt["mean"]),
            _cell(rtt["max"]),
            _cell(write["p50"]),
            _cell(write["p95"]),
            _cell(poll["p50"]),
            _cell(poll["p95"]),
            _dist_text(group["n_ops"]),
            _dist_text(group["oppoll"]),
            _cell(graph["records"]),
            _cell(graph["cache_hits"]),
            _cell(graph["cache_hit_rate"]),
            _cell(graph["unique_graph_uids"]),
            _cell(graph["prep_ns"]["p50"]),
            _cell(graph["prep_ns"]["p95"]),
            _dist_text(graph["nodes"]["distribution"]),
        )
        lines.append("\t".join(row))
    return "\n".join(lines) + "\n"


def _write_text(path, text):
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        raise SystemExit("analyze_rtt: cannot write %s: %s" % (path, exc))


def _print_summary(report, stream):
    """Small human-readable view; JSON/TSV remain the machine interfaces."""
    schemas = ", ".join("%s=%d" % (name, count)
                        for name, count in report["input_schemas"].items())
    print("input schemas: %s" % schemas, file=stream)
    for group in report["groups"]:
        graph = group["graph"]
        if graph["cache_hit_rate"] is None:
            rate = "n/a"
        else:
            rate = "%.2f%%" % (100.0 * graph["cache_hit_rate"])
        bucket = "" if group["bucket"] is None else " bucket=%s" % (group["bucket"],)
        print("%s/%s%s: count=%d rtt_p50_ns=%s graph_records=%d "
              "cache_hit_rate=%s prep_p50_ns=%s"
              % (group["label"], group["role"], bucket, group["count"],
                 _cell(group["rtt_ns"]["p50"]), graph["records"], rate,
                 _cell(graph["prep_ns"]["p50"])), file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate ggml.hex.rtt.v1/v2 host-only RTT traces (stdlib only).")
    parser.add_argument("traces", nargs="+", metavar="TRACE",
                        help="JSONL trace file(s)")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="one label per trace (default: file stem)")
    parser.add_argument("--out-json", metavar="PATH",
                        help="write the aggregate JSON document here")
    parser.add_argument("--out-tsv", metavar="PATH",
                        help="write the aggregate TSV table here")
    parser.add_argument("--bucket", choices=("n_ops",), default=None,
                        help="also split groups by request batch size")
    parser.add_argument("--json", action="store_true",
                        help="print the aggregate JSON document to stdout")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the JSON document to stdout by default")
    parser.add_argument("--summary", action="store_true",
                        help="print a short human-readable schema/cache-hit summary")
    args = parser.parse_args(argv)

    labels = args.labels
    if labels is not None and len(labels) != len(args.traces):
        parser.error("--labels needs exactly one value per trace (%d labels, %d traces)"
                     % (len(labels), len(args.traces)))
    if labels is None:
        used = set()
        labels = [_label_from_path(path, used) for path in args.traces]

    report = build_report(list(zip(labels, args.traces)), bucket_by=args.bucket)

    if args.out_json:
        _write_text(args.out_json, json.dumps(report, indent=2, sort_keys=False) + "\n")
    if args.out_tsv:
        _write_text(args.out_tsv, report_to_tsv(report))

    print_json = args.json or (
        not args.out_json and not args.out_tsv and not args.quiet)
    if print_json:
        print(json.dumps(report, indent=2, sort_keys=False))
    if args.summary:
        # Keep stdout valid JSON when the document is also printed there.
        _print_summary(report, sys.stderr if print_json else sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
