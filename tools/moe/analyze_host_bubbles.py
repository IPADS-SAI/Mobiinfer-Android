#!/usr/bin/env python3
"""Offline host-bubble decomposition for LLAMA_MOE_HANDOFF_TRACE.

Reads one or more ``llama.moe.handoff.v2`` JSONL traces produced by
``LLAMA_MOE_HANDOFF_TRACE=/path/trace.jsonl`` and reports, per verify (epoch),
the host-side pieces that the profile in T-0176 could not separate:

  * MAP_CUSTOM callback time (MAP_END.map_ns)
  * MAP_END -> first same-layer PREAD_BEGIN gap
  * PREAD_DONE -> next PREAD_BEGIN gap, split by demand/prefetch
  * worker cv_go wait time (WORKER_WAKE.wait_ns), grouped by tid
  * arm() time (ARM_END.arm_ns), grouped by decision
  * queue depth (QUEUE_ENQUEUE / WORKER_WAKE.queue_depth)

Only the Python standard library is used.  The parser counts bad JSON, wrong
schema and missing fields instead of silently dropping them.  Empty input is a
valid zero result (no divisions by zero).

The new events are additive: old event lines keep the exact v2 field set.  For
the six bubble events the C++ writer appends named fields (``map_ns``,
``arm_ns``, ``decision``, ``queue_depth``, ``wait_ns``, ``layer_next``); this
parser also accepts the pre-existing integer slots (``bytes``/``result``) as a
fallback so traces from older/alternate instrumentation still parse.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TRACE_SCHEMA = "llama.moe.handoff.v2"
OUT_SCHEMA = "dpsk.host_bubbles.v1"

MAP_BEGIN = "MAP_BEGIN"
MAP_END = "MAP_END"
ARM_BEGIN = "ARM_BEGIN"
ARM_END = "ARM_END"
QUEUE_ENQUEUE = "QUEUE_ENQUEUE"
WORKER_WAKE = "WORKER_WAKE"
PREAD_BEGIN = "PREAD_BEGIN"
PREAD_DONE = "PREAD_DONE"

BUBBLE_EVENTS = (MAP_BEGIN, MAP_END, ARM_BEGIN, ARM_END, QUEUE_ENQUEUE, WORKER_WAKE)

# Base fields every event line must carry.  New bubble events still use the
# existing emitter fields; extra per-event fields are checked below.
BASE_FIELDS = ("event", "ns", "tid", "epoch", "layer")


@dataclass
class Event:
    name: str
    ns: int
    tid: int
    epoch: int
    layer: int
    source: str = "other"
    job_id: int = 0
    rows: int = 0
    expert: int = -1
    slot: int = -1
    io_id: int = 0
    result: int = 0
    bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def value(self, named: str, fallback: Optional[str] = None) -> Optional[int]:
        if named in self.extra and self.extra[named] is not None:
            try:
                return int(self.extra[named])
            except (TypeError, ValueError):
                return None
        if fallback is not None and fallback in self.extra and self.extra[fallback] is not None:
            try:
                return int(self.extra[fallback])
            except (TypeError, ValueError):
                return None
        return None


@dataclass
class Trace:
    label: str
    path: str
    events: List[Event] = field(default_factory=list)
    total_lines: int = 0
    bad_json: int = 0
    bad_schema: int = 0
    missing_field: int = 0
    header_seen: bool = False
    # Parseable-but-rejected lines can still carry an epoch; remember it so
    # n_bad can be attributed to a verify where possible.
    bad_by_epoch: Dict[int, int] = field(default_factory=dict)


class BadLine(Exception):
    def __init__(self, kind: str, epoch: Optional[int] = None) -> None:
        super().__init__(kind)
        self.kind = kind
        self.epoch = epoch


def _as_int(obj: Dict[str, Any], key: str, default: int = 0) -> int:
    value = obj.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise BadLine("missing_field", _epoch_hint(obj))


def _epoch_hint(obj: Dict[str, Any]) -> Optional[int]:
    try:
        value = obj.get("epoch")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _classify_source(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "demand" in text:
        return "demand"
    if text in ("prefetch", "pf") or "prefetch" in text or text.startswith("pf_"):
        return "prefetch"
    return "unknown"


def _parse_event(obj: Dict[str, Any]) -> Event:
    for key in BASE_FIELDS:
        if key not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))
    name = str(obj["event"])
    if not name:
        raise BadLine("missing_field", _epoch_hint(obj))
    ns = _as_int(obj, "ns")
    if ns < 0:
        raise BadLine("missing_field", _epoch_hint(obj))
    extra = {k: obj[k] for k in obj.keys() if k not in {
        "index", "ns", "tid", "event", "source", "job_id", "epoch", "layer",
        "rows", "expert", "slot", "result", "bytes", "active_job_id",
        "active_epoch", "active_layer", "active_rows", "generation", "domain",
        "binding_generation", "io_id", "reason", "related_expert",
        "related_slot", "related_generation", "scanned", "capacity_slots",
        "free_slots", "loading_slots", "protected_slots", "pinned_slots",
        "eligible_slots", "file_offset", "requested_pages", "resident_pages",
        "probe_begin_ns", "probe_end_ns", "probe_errno", "admit_seq",
        "admit_class", "admit_active", "admit_waiters", "admit_capacity",
        "schema", "clock", "unit", "capacity", "entry_bytes", "buffer_bytes",
        "attempted", "written", "lost",
    }}
    # The named timing fields live in `extra`; also retain the pre-existing
    # slots so the fallback path is explicit.
    extra["bytes"] = obj.get("bytes", 0)
    extra["result"] = obj.get("result", 0)
    extra["reason"] = obj.get("reason", "none")

    if name == MAP_END:
        if "map_ns" not in obj and "bytes" not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))
    elif name == ARM_END:
        if "arm_ns" not in obj and "bytes" not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))
        if "decision" not in obj and "reason" not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))
    elif name == WORKER_WAKE:
        if "wait_ns" not in obj and "bytes" not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))
        if "queue_depth" not in obj and "result" not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))
    elif name == QUEUE_ENQUEUE:
        if "queue_depth" not in obj and "result" not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))
    elif name in (PREAD_BEGIN, PREAD_DONE):
        if "source" not in obj and "origin" not in obj:
            raise BadLine("missing_field", _epoch_hint(obj))

    return Event(
        name=name,
        ns=ns,
        tid=_as_int(obj, "tid"),
        epoch=_as_int(obj, "epoch"),
        layer=_as_int(obj, "layer", -1),
        source=str(obj.get("source", "other")),
        job_id=_as_int(obj, "job_id"),
        rows=_as_int(obj, "rows"),
        expert=_as_int(obj, "expert", -1),
        slot=_as_int(obj, "slot", -1),
        io_id=_as_int(obj, "io_id"),
        result=_as_int(obj, "result"),
        bytes=_as_int(obj, "bytes"),
        extra=extra,
    )


def load_trace(path: str, label: str) -> Trace:
    trace = Trace(label=label, path=path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            trace.total_lines += 1
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                trace.bad_json += 1
                continue
            if not isinstance(obj, dict):
                trace.bad_json += 1
                continue
            if "schema" in obj:
                if obj.get("schema") != TRACE_SCHEMA:
                    trace.bad_schema += 1
                    epoch = _epoch_hint(obj)
                    if epoch is not None:
                        trace.bad_by_epoch[epoch] = trace.bad_by_epoch.get(epoch, 0) + 1
                    continue
                if "event" not in obj:
                    trace.header_seen = True
                    continue
            try:
                event = _parse_event(obj)
            except BadLine as bad:
                if bad.kind == "missing_field":
                    trace.missing_field += 1
                else:  # defensive: _parse_event only raises missing_field today
                    trace.bad_schema += 1
                if bad.epoch is not None:
                    trace.bad_by_epoch[bad.epoch] = trace.bad_by_epoch.get(bad.epoch, 0) + 1
                continue
            trace.events.append(event)
    trace.events.sort(key=lambda event: (event.ns, event.name, event.tid))
    return trace


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo]) + (float(ordered[hi]) - float(ordered[lo])) * (pos - lo)


def ms_stats_ns(values_ns: Sequence[int]) -> Dict[str, Any]:
    values_ms = [value / 1_000_000.0 for value in values_ns]
    return {
        "count": len(values_ms),
        "sum_ms": sum(values_ms),
        "p50_ms": percentile(values_ms, 50.0),
        "p95_ms": percentile(values_ms, 95.0),
    }


def int_stats(values: Sequence[int]) -> Dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "max": ordered[-1] if ordered else 0,
        "median": percentile([float(v) for v in ordered], 50.0),
        "p95": percentile([float(v) for v in ordered], 95.0),
    }


def _read_source(event: Event) -> str:
    # PREAD events inherit handoff_thread.context, whose source is
    # demand/prefetch/prewarm.  Accept `origin` as an alias for traces that name
    # the classification field that way; otherwise fall back to domain/unknown.
    raw = event.source
    if not raw or raw == "other":
        raw = event.extra.get("origin")
    if not raw or raw == "other":
        raw = event.extra.get("domain")
    return _classify_source(raw)


def _gap_pairs(events: Sequence[Event]) -> List[Tuple[Event, Event, int, str]]:
    """Return (done, next_begin, gap_ns, transition) for PREAD_DONE -> next PREAD_BEGIN."""
    begins = [event for event in events if event.name == PREAD_BEGIN]
    begins.sort(key=lambda event: event.ns)
    begin_ns = [event.ns for event in begins]
    pairs: List[Tuple[Event, Event, int, str]] = []
    for done in events:
        if done.name != PREAD_DONE:
            continue
        pos = bisect.bisect_right(begin_ns, done.ns)
        if pos >= len(begins):
            continue
        nxt = begins[pos]
        transition = "%s->%s" % (_read_source(done), _read_source(nxt))
        pairs.append((done, nxt, nxt.ns - done.ns, transition))
    return pairs


def _first_read_after(events: Sequence[Event], layer: int, after_ns: int) -> Optional[Event]:
    candidates = [event for event in events
                  if event.name == PREAD_BEGIN and event.layer == layer and event.ns > after_ns]
    return min(candidates, key=lambda event: event.ns) if candidates else None


def _empty_summary() -> Dict[str, Any]:
    return {
        "map": ms_stats_ns([]),
        "map_to_read": ms_stats_ns([]),
        "map_to_read_unmatched": 0,
        "read_to_read": ms_stats_ns([]),
        "read_transitions": {},
        "reads_by_source": {"demand": 0, "prefetch": 0, "unknown": 0},
        "wake": ms_stats_ns([]),
        "wake_by_tid": [],
        "arm": ms_stats_ns([]),
        "arm_by_decision": {},
        "queue_depth": int_stats([]),
    }


def _summarize_group(events: Sequence[Event],
                     cross_layer_pairs: Optional[Sequence[Tuple[Event, Event, int, str]]] = None) -> Dict[str, Any]:
    summary = _empty_summary()
    map_ns: List[int] = []
    map_to_read_ns: List[int] = []
    unmatched = 0
    for event in events:
        if event.name != MAP_END:
            continue
        value = event.value("map_ns", "bytes")
        if value is not None:
            map_ns.append(value)
        nxt = _first_read_after(events, event.layer, event.ns)
        if nxt is None:
            unmatched += 1
        else:
            map_to_read_ns.append(nxt.ns - event.ns)

    pairs = list(cross_layer_pairs) if cross_layer_pairs is not None else _gap_pairs(events)
    read_gaps = [gap for _, _, gap, _ in pairs]
    transitions: Dict[str, Dict[str, Any]] = {}
    for _, _, gap, transition in pairs:
        transitions.setdefault(transition, []).append(gap)
    reads_by_source = {"demand": 0, "prefetch": 0, "unknown": 0}
    for event in events:
        if event.name == PREAD_BEGIN:
            reads_by_source[_read_source(event)] += 1

    wake_ns: List[int] = []
    wake_by_tid: Dict[int, List[int]] = {}
    arm_ns: List[int] = []
    arm_by_decision: Dict[str, List[int]] = {}
    queue_depth: List[int] = []
    for event in events:
        if event.name == WORKER_WAKE:
            value = event.value("wait_ns", "bytes")
            if value is not None:
                wake_ns.append(value)
                wake_by_tid.setdefault(event.tid, []).append(value)
            depth = event.value("queue_depth", "result")
            if depth is not None:
                queue_depth.append(depth)
        elif event.name == ARM_END:
            value = event.value("arm_ns", "bytes")
            if value is not None:
                arm_ns.append(value)
                decision = str(event.extra.get("decision", event.extra.get("reason", "skip")))
                if decision not in ("accept", "skip"):
                    decision = "skip" if event.result < 0 else "accept"
                arm_by_decision.setdefault(decision, []).append(value)
        elif event.name == QUEUE_ENQUEUE:
            depth = event.value("queue_depth", "result")
            if depth is not None:
                queue_depth.append(depth)

    summary["map"] = ms_stats_ns(map_ns)
    summary["map_to_read"] = ms_stats_ns(map_to_read_ns)
    summary["map_to_read_unmatched"] = unmatched
    summary["read_to_read"] = ms_stats_ns(read_gaps)
    summary["read_transitions"] = {
        transition: ms_stats_ns(gaps) for transition, gaps in sorted(transitions.items())
    }
    summary["reads_by_source"] = reads_by_source
    summary["wake"] = ms_stats_ns(wake_ns)
    summary["wake_by_tid"] = [
        dict(ms_stats_ns(values), tid=tid) for tid, values in sorted(wake_by_tid.items())
    ]
    summary["arm"] = ms_stats_ns(arm_ns)
    summary["arm_by_decision"] = {
        decision: ms_stats_ns(values) for decision, values in sorted(arm_by_decision.items())
    }
    summary["queue_depth"] = int_stats(queue_depth)
    return summary


def _bad_counts(traces: Sequence[Trace]) -> Dict[str, int]:
    return {
        "bad_json": sum(trace.bad_json for trace in traces),
        "bad_schema": sum(trace.bad_schema for trace in traces),
        "missing_field": sum(trace.missing_field for trace in traces),
    }


def analyze(traces: Sequence[Trace], bucket: str = "layer") -> Dict[str, Any]:
    if bucket not in ("layer", "epoch"):
        raise ValueError("bucket must be 'layer' or 'epoch'")
    groups: Dict[Tuple[str, int], List[Event]] = {}
    layer_groups: Dict[Tuple[str, int, int], List[Event]] = {}
    for trace in traces:
        for event in trace.events:
            groups.setdefault((trace.label, event.epoch), []).append(event)
            layer_groups.setdefault((trace.label, event.epoch, event.layer), []).append(event)

    labels = sorted({trace.label for trace in traces})
    pairs_by_epoch: Dict[Tuple[str, int], List[Tuple[Event, Event, int, str]]] = {}
    for (label, epoch), events in groups.items():
        pairs_by_epoch[(label, epoch)] = _gap_pairs(events)
    by_verify: List[Dict[str, Any]] = []
    for (label, epoch), events in sorted(groups.items()):
        summary = _summarize_group(events)
        bad_for_epoch = 0
        for trace in traces:
            if trace.label == label:
                bad_for_epoch += trace.bad_by_epoch.get(epoch, 0)
        wall_ns = max(event.ns for event in events) - min(event.ns for event in events)
        row = {
            "label": label,
            "epoch": epoch,
            "verify_wall_ms": wall_ns / 1_000_000.0,
            "map_ms": summary["map"]["sum_ms"],
            "map_to_read_ms": summary["map_to_read"]["sum_ms"],
            "read_to_read_ms": summary["read_to_read"]["sum_ms"],
            "wake_ms": summary["wake"]["sum_ms"],
            "arm_ms": summary["arm"]["sum_ms"],
            "max_queue_depth": summary["queue_depth"]["max"],
            "n_map": summary["map"]["count"],
            "n_read": sum(summary["reads_by_source"].values()),
            "n_wake": summary["wake"]["count"],
            "n_arm": summary["arm"]["count"],
            "n_queue": summary["queue_depth"]["count"],
            "n_bad": bad_for_epoch,
            "map": summary["map"],
            "map_to_read": summary["map_to_read"],
            "map_to_read_unmatched": summary["map_to_read_unmatched"],
            "read_to_read": summary["read_to_read"],
            "read_transitions": summary["read_transitions"],
            "reads_by_source": summary["reads_by_source"],
            "wake": summary["wake"],
            "wake_by_tid": summary["wake_by_tid"],
            "arm": summary["arm"],
            "arm_by_decision": summary["arm_by_decision"],
            "queue_depth": summary["queue_depth"],
        }
        by_verify.append(row)

    by_bucket: List[Dict[str, Any]] = []
    if bucket == "layer":
        for (label, epoch, layer), events in sorted(layer_groups.items()):
            # read->read gaps are global per verify (the interesting handoffs are
            # cross-layer); attribute each gap to the layer of its PREAD_DONE.
            all_pairs = pairs_by_epoch.get((label, epoch), [])
            layer_pairs = [pair for pair in all_pairs if pair[0].layer == layer]
            summary = _summarize_group(events, cross_layer_pairs=layer_pairs)
            by_bucket.append({
                "label": label,
                "epoch": epoch,
                "layer": layer,
                "n_events": len(events),
                **summary,
            })
    else:
        for row in by_verify:
            by_bucket.append({
                "label": row["label"],
                "epoch": row["epoch"],
                "n_events": len(groups[(row["label"], row["epoch"])]),
                "map": row["map"],
                "map_to_read": row["map_to_read"],
                "read_to_read": row["read_to_read"],
                "read_transitions": row["read_transitions"],
                "reads_by_source": row["reads_by_source"],
                "wake": row["wake"],
                "wake_by_tid": row["wake_by_tid"],
                "arm": row["arm"],
                "arm_by_decision": row["arm_by_decision"],
                "queue_depth": row["queue_depth"],
            })

    counts = {
        "files": len(traces),
        "total_lines": sum(trace.total_lines for trace in traces),
        "events": sum(len(trace.events) for trace in traces),
        "headers": sum(1 for trace in traces if trace.header_seen),
        "map_begin": sum(1 for trace in traces for event in trace.events if event.name == MAP_BEGIN),
        "map_end": sum(1 for trace in traces for event in trace.events if event.name == MAP_END),
        "arm_begin": sum(1 for trace in traces for event in trace.events if event.name == ARM_BEGIN),
        "arm_end": sum(1 for trace in traces for event in trace.events if event.name == ARM_END),
        "queue_enqueue": sum(1 for trace in traces for event in trace.events if event.name == QUEUE_ENQUEUE),
        "worker_wake": sum(1 for trace in traces for event in trace.events if event.name == WORKER_WAKE),
        "pread_begin": sum(1 for trace in traces for event in trace.events if event.name == PREAD_BEGIN),
        "pread_done": sum(1 for trace in traces for event in trace.events if event.name == PREAD_DONE),
    }
    counts.update(_bad_counts(traces))
    totals = {
        "verify_wall_ms": sum(row["verify_wall_ms"] for row in by_verify),
        "map_ms": sum(row["map_ms"] for row in by_verify),
        "map_to_read_ms": sum(row["map_to_read_ms"] for row in by_verify),
        "read_to_read_ms": sum(row["read_to_read_ms"] for row in by_verify),
        "wake_ms": sum(row["wake_ms"] for row in by_verify),
        "arm_ms": sum(row["arm_ms"] for row in by_verify),
        "verifies": len(by_verify),
    }
    return {
        "schema": OUT_SCHEMA,
        "source_schema": TRACE_SCHEMA,
        "bucket": bucket,
        "labels": labels,
        "counts": counts,
        "totals": totals,
        "by_verify": by_verify,
        "by_bucket": by_bucket,
    }


TSV_COLUMNS = (
    "label", "epoch", "verify_wall_ms", "map_ms", "map_to_read_ms",
    "read_to_read_ms", "wake_ms", "arm_ms", "max_queue_depth", "n_map",
    "n_read", "n_wake", "n_bad",
)


def _fmt_ms(value: float) -> str:
    return "%.6f" % (0.0 if value is None else float(value))


def render_tsv(result: Dict[str, Any]) -> str:
    lines = ["\t".join(TSV_COLUMNS)]
    for row in result.get("by_verify", []):
        lines.append("\t".join((
            str(row["label"]),
            str(row["epoch"]),
            _fmt_ms(row["verify_wall_ms"]),
            _fmt_ms(row["map_ms"]),
            _fmt_ms(row["map_to_read_ms"]),
            _fmt_ms(row["read_to_read_ms"]),
            _fmt_ms(row["wake_ms"]),
            _fmt_ms(row["arm_ms"]),
            str(row["max_queue_depth"]),
            str(row["n_map"]),
            str(row["n_read"]),
            str(row["n_wake"]),
            str(row["n_bad"]),
        )))
    return "\n".join(lines) + "\n"


def _write(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="+", help="llama.moe.handoff.v2 JSONL trace(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="one label per trace (default: file basename)")
    parser.add_argument("--out-json", default=None, help="JSON output path ('-' = stdout)")
    parser.add_argument("--out-tsv", default=None, help="TSV output path ('-' = stdout)")
    parser.add_argument("--bucket", choices=("layer", "epoch"), default="layer",
                        help="extra aggregation bucket (default: layer)")
    parser.add_argument("--json", action="store_true",
                        help="print the JSON result to stdout when --out-json is absent")
    args = parser.parse_args(argv)

    if args.labels is not None and len(args.labels) != len(args.traces):
        parser.error("--labels needs exactly one value per trace")

    traces: List[Trace] = []
    for index, path in enumerate(args.traces):
        if args.labels is not None:
            label = str(args.labels[index])
        else:
            label = os.path.basename(path)
            if label.endswith(".jsonl"):
                label = label[: -len(".jsonl")]
        try:
            traces.append(load_trace(path, label))
        except OSError as error:
            print("analyze_host_bubbles: cannot read %s: %s" % (path, error), file=sys.stderr)
            return 2

    result = analyze(traces, bucket=args.bucket)
    payload = json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n"

    if args.out_json:
        _write(args.out_json, payload)
    if args.json or (not args.out_json and not args.out_tsv):
        sys.stdout.write(payload)
    if args.out_tsv:
        _write(args.out_tsv, render_tsv(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
