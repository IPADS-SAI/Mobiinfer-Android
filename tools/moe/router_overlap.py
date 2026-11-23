#!/usr/bin/env python3
"""Compare expert-set identity between two LLAMA_MOE_SWEEP decode traces.

Input format (written by ``sweep_write()`` in ``src/llama-moe-slot.cpp``, one
line per decode step per MoE layer)::

    L17 E:203,222,74,1,9,60,33,88 P:203,222,74,1,9,60,88,120

``E:`` is the set of experts actually used by that layer at that decode step.
Its order is the engine/file order, **not** router rank, so ``E[0]`` is NOT
top-1.  ``P:`` is the previous layer's prediction for this layer, ordered by
descending score; only ``P[0]`` is a top-1 prediction.

This tool compares two traces as **sets of expert ids**, per layer and step:

  jaccard_E          |A & B| / |A | B|          (1.0 when both empty)
  overlap_E          |A & B| / max(|A|, |B|)    (direction of under/over-read)
  exact_set_match    1[A == B]
  size_delta         |B| - |A|
  jaccard_P          Jaccard of the P sets, only when both sides have P
  rbo_p90/rbo_p98    rank-biased overlap of P, p=0.9 and 0.98
  top1_agreement_P   1[P_A[0] == P_B[0]], only when both sides have P

Only decode ``L<n>`` lines are consumed in v1.  Prefill ``P<n> E:`` lines
(``sweep_write_prefill()``) are counted and skipped; they are never mixed into
decode statistics (prefill support is pending validation).

Aggregation is over all step pairs (not weighted by layer); per-layer values
are also emitted.  Default is fail-closed: differing layer sets or per-layer
step counts exit with code 2.  ``--allow-layer-subset`` explicitly compares
only the intersection and records ``skipped_layers``.

Exit codes: 0 = success, 2 = inconsistent/invalid input (no partial success).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Iterable, Sequence

DECODE_RE = re.compile(r"^L(\d+)\s+E:([0-9,]*)(?:\s+P:([0-9,]*))?\s*$")
PREFILL_RE = re.compile(r"^P(\d+)\s+E:([0-9,]*)\s*$")

MAX_MALFORMED_SAMPLES = 5
DEFAULT_P_VALUES = (0.9, 0.98)


class TraceError(Exception):
    """Invalid input or inconsistent traces (CLI exit code 2)."""

    def __init__(self, message: str, details: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.details = list(details)


@dataclass
class Step:
    used: tuple[int, ...]
    pred: tuple[int, ...]


@dataclass
class Trace:
    path: str
    layers: dict[int, list[Step]] = field(default_factory=dict)
    sequence: list[int] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)


def _parse_ids(payload: str) -> list[int]:
    """Parse a comma-separated id list; reject empty tokens such as ``1,,2``."""
    if payload == "":
        return []
    tokens = payload.split(",")
    if any(token == "" or not token.isdigit() for token in tokens):
        raise ValueError(f"bad id list {payload!r}")
    return [int(token) for token in tokens]


def _dedupe(ids: Sequence[int]) -> tuple[int, ...]:
    """Drop duplicate ids while preserving first-occurrence order."""
    return tuple(dict.fromkeys(ids))


def parse_trace(path: str) -> Trace:
    """Parse one sweep trace.  Bad lines are counted, never treated as empty sets."""
    stats: dict[str, object] = {
        "decode_lines": 0,
        "blank_lines": 0,
        "comment_lines": 0,
        "malformed_lines": 0,
        "malformed_samples": [],
        "prefill_lines_skipped": 0,
        "duplicate_E_lines": 0,
        "duplicate_P_lines": 0,
        "p_missing_lines": 0,
    }
    trace = Trace(path=path, stats=stats)
    try:
        fp = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TraceError(f"cannot read trace {path!r}: {exc.strerror}") from exc

    samples: list[dict[str, object]] = []
    with fp:
        for lineno, raw in enumerate(fp, 1):
            line = raw.strip().lstrip("\ufeff")
            if not line:
                stats["blank_lines"] = int(stats["blank_lines"]) + 1
                continue
            if line.startswith("#"):
                stats["comment_lines"] = int(stats["comment_lines"]) + 1
                continue

            m = DECODE_RE.match(line)
            if m:
                layer = int(m.group(1))
                try:
                    used_raw = _parse_ids(m.group(2))
                    pred_raw = _parse_ids(m.group(3) or "")
                except ValueError:
                    stats["malformed_lines"] = int(stats["malformed_lines"]) + 1
                    if len(samples) < MAX_MALFORMED_SAMPLES:
                        samples.append({"line": lineno, "text": line[:160]})
                    continue

                used = _dedupe(used_raw)
                pred = _dedupe(pred_raw)
                if len(used) != len(used_raw):
                    stats["duplicate_E_lines"] = int(stats["duplicate_E_lines"]) + 1
                if len(pred) != len(pred_raw):
                    stats["duplicate_P_lines"] = int(stats["duplicate_P_lines"]) + 1
                if m.group(3) is None:
                    stats["p_missing_lines"] = int(stats["p_missing_lines"]) + 1

                trace.layers.setdefault(layer, []).append(Step(used, pred))
                trace.sequence.append(layer)
                stats["decode_lines"] = int(stats["decode_lines"]) + 1
                continue

            if PREFILL_RE.match(line):
                # Prefill lines use the P<layer> prefix; v1 never mixes them in.
                stats["prefill_lines_skipped"] = int(stats["prefill_lines_skipped"]) + 1
                continue

            stats["malformed_lines"] = int(stats["malformed_lines"]) + 1
            if len(samples) < MAX_MALFORMED_SAMPLES:
                samples.append({"line": lineno, "text": line[:160]})

    stats["malformed_samples"] = samples
    stats["n_layers"] = len(trace.layers)
    return trace


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def overlap_fraction(a: set[int], b: set[int]) -> float:
    largest = max(len(a), len(b))
    if largest == 0:
        return 1.0
    return len(a & b) / largest


def rbo(a: Sequence[int], b: Sequence[int], p: float) -> float:
    """Rank-biased overlap with the standard residual-tail extrapolation.

    ``RBO = (1-p) * sum_{d=1..D} p^(d-1) * X_d / d + p^D * X_D / D``
    where ``D = min(len(a), len(b))`` and ``X_d`` is the size of the overlap of
    the two top-d prefixes.  Both empty -> 1.0; one empty -> 0.0.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    depth = min(len(a), len(b))
    prefix_a: set[int] = set()
    prefix_b: set[int] = set()
    overlap = 0
    weighted = 0.0
    for d in range(1, depth + 1):
        prefix_a.add(a[d - 1])
        prefix_b.add(b[d - 1])
        overlap = len(prefix_a & prefix_b)
        weighted += (p ** (d - 1)) * overlap / d
    return (1.0 - p) * weighted + (p ** depth) * overlap / depth


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def p_key(p: float) -> str:
    return f"p{format(p, '.12g')}"


def _p_scalar_name(p: float) -> str:
    # Stable names required by the TSV contract; other p values are appended.
    if abs(p - 0.9) < 1e-12:
        return "rbo_p90"
    if abs(p - 0.98) < 1e-12:
        return "rbo_p98"
    return f"rbo_p{format(p, '.12g')}"


def compare_layer(
    layer: int,
    steps_a: Sequence[Step],
    steps_b: Sequence[Step],
    p_values: Sequence[float],
) -> dict[str, object]:
    jaccards: list[float] = []
    overlaps: list[float] = []
    exacts: list[float] = []
    deltas: list[float] = []
    jaccards_p: list[float] = []
    rbos: dict[float, list[float]] = {p: [] for p in p_values}
    top1s: list[float] = []
    if len(steps_a) != len(steps_b):
        raise TraceError(
            f"layer={layer} step counts differ "
            f"(steps_A={len(steps_a)} steps_B={len(steps_b)})"
        )
    step_records: list[dict[str, object]] = []
    n_steps_p = 0
    empty_steps = 0
    one_sided_empty = 0

    for i, (sa, sb) in enumerate(zip(steps_a, steps_b)):
        set_a, set_b = set(sa.used), set(sb.used)
        jac = jaccard(set_a, set_b)
        ov = overlap_fraction(set_a, set_b)
        exact = 1.0 if set_a == set_b else 0.0
        delta = float(len(set_b) - len(set_a))
        empty = not set_a and not set_b
        if empty:
            empty_steps += 1
        elif not set_a or not set_b:
            one_sided_empty += 1

        both_p = bool(sa.pred) and bool(sb.pred)
        jac_p: float | None = None
        rbo_values: dict[str, float | None] = {p_key(p): None for p in p_values}
        top1: float | None = None
        if both_p:
            n_steps_p += 1
            jac_p = jaccard(set(sa.pred), set(sb.pred))
            jaccards_p.append(jac_p)
            for p in p_values:
                value = rbo(sa.pred, sb.pred, p)
                rbo_values[p_key(p)] = value
                rbos[p].append(value)
            top1 = 1.0 if sa.pred[0] == sb.pred[0] else 0.0
            top1s.append(top1)

        jaccards.append(jac)
        overlaps.append(ov)
        exacts.append(exact)
        deltas.append(delta)
        step_records.append({
            "step": i,
            "n_A": len(set_a),
            "n_B": len(set_b),
            "E_A": list(sa.used),
            "E_B": list(sb.used),
            "P_A": list(sa.pred),
            "P_B": list(sb.pred),
            "jaccard_E": jac,
            "overlap_E": ov,
            "exact_set_match": int(exact),
            "size_delta": delta,
            "jaccard_P": jac_p,
            "rbo": rbo_values,
            "top1_agreement_P": top1,
            "empty": empty,
        })

    jac_stats = _stats(jaccards)
    record: dict[str, object] = {
        "layer": layer,
        "n_steps": len(steps_a),
        "n_steps_P": n_steps_p,
        "jaccard_E": jac_stats["mean"],
        "median_jaccard_E": jac_stats["median"],
        "min_jaccard_E": jac_stats["min"],
        "max_jaccard_E": jac_stats["max"],
        "mean_overlap_E": _mean(overlaps),
        "exact_match_rate": _mean(exacts),
        "n_mismatch_steps": sum(1 for value in exacts if value == 0.0),
        "mean_size_delta": _mean(deltas),
        "median_size_delta": statistics.median(deltas) if deltas else None,
        "jaccard_P": _mean(jaccards_p),
        "median_jaccard_P": statistics.median(jaccards_p) if jaccards_p else None,
        "top1_agreement_P": _mean(top1s),
        "empty_steps": empty_steps,
        "one_sided_empty_steps": one_sided_empty,
        "rbo": {p_key(p): _mean(rbos[p]) for p in p_values},
        "steps": step_records,
    }
    # The two conventional p values get stable scalar names for TSV/gate consumers.
    for p in p_values:
        record[_p_scalar_name(p)] = _mean(rbos[p])
    record.setdefault("rbo_p90", None)
    record.setdefault("rbo_p98", None)
    return record


def _trace_stat(trace: Trace, key: str) -> int:
    return int(trace.stats.get(key, 0))


def _validate(
    trace_a: Trace,
    trace_b: Trace,
    expected_layers: int,
    min_steps: int,
    allow_layer_subset: bool,
) -> list[int]:
    layers_a = set(trace_a.layers)
    layers_b = set(trace_b.layers)
    common = sorted(layers_a & layers_b)

    if expected_layers:
        if allow_layer_subset:
            if len(common) != expected_layers:
                raise TraceError(
                    f"expected {expected_layers} common layers in --allow-layer-subset mode, "
                    f"got layers_common={len(common)} "
                    f"(layers_A={len(layers_a)} layers_B={len(layers_b)})"
                )
        elif len(layers_a) != expected_layers or len(layers_b) != expected_layers:
            raise TraceError(
                f"expected {expected_layers} layers, got layers_A={len(layers_a)} "
                f"layers_B={len(layers_b)}"
            )

    if not layers_a or not layers_b:
        raise TraceError(
            "no decode L<n> lines found "
            f"(layers_A={len(layers_a)} layers_B={len(layers_b)})"
        )

    if layers_a != layers_b:
        if not allow_layer_subset:
            details = []
            for layer in sorted(layers_a | layers_b):
                steps_a = len(trace_a.layers.get(layer, ()))
                steps_b = len(trace_b.layers.get(layer, ()))
                details.append(f"layer={layer} steps_A={steps_a} steps_B={steps_b}")
            raise TraceError(
                "trace layer sets differ (fail-closed); "
                "use --allow-layer-subset to compare the intersection",
                details,
            )
        if not common:
            raise TraceError(
                "no common layers between traces "
                f"(layers_A={sorted(layers_a)} layers_B={sorted(layers_b)})"
            )

    step_details = []
    for layer in common:
        steps_a = len(trace_a.layers[layer])
        steps_b = len(trace_b.layers[layer])
        if steps_a != steps_b:
            step_details.append(f"layer={layer} steps_A={steps_a} steps_B={steps_b}")
    if step_details:
        raise TraceError(
            "per-layer step counts differ (fail-closed); no silent truncation",
            step_details,
        )

    short_details = []
    for layer in common:
        steps_a = len(trace_a.layers[layer])
        if steps_a < min_steps:
            short_details.append(
                f"layer={layer} steps_A={steps_a} steps_B={steps_a} min_steps={min_steps}"
            )
    if short_details:
        raise TraceError(
            f"layer has fewer steps than --min-steps={min_steps} (fail-closed)",
            short_details,
        )

    return common


def _sequence_warning(trace_a: Trace, trace_b: Trace, compared_layers: Sequence[int]) -> str | None:
    allowed = set(compared_layers)
    seq_a = [layer for layer in trace_a.sequence if layer in allowed]
    seq_b = [layer for layer in trace_b.sequence if layer in allowed]
    if seq_a == seq_b:
        return None
    mismatch = next(
        (i for i, (la, lb) in enumerate(zip(seq_a, seq_b)) if la != lb),
        min(len(seq_a), len(seq_b)),
    )
    return (
        "layer_sequence_mismatch: the global order of decode lines differs "
        f"(first divergence at index {mismatch}); metrics align by per-layer occurrence, "
        "so verify the traces come from the same engine/layer order"
    )


def _warnings(
    trace_a: Trace,
    trace_b: Trace,
    compared_layers: Sequence[int],
    skipped_layers: Sequence[int],
    allow_layer_subset: bool,
    aggregate: dict[str, object],
) -> list[str]:
    warnings = [
        "v1 compares decode L<n> lines only; prefill P<n> E: lines are skipped "
        "(prefill support is pending validation).",
        "E is an unordered expert-id set in engine/file order; E[0] is NOT top-1. "
        "top-1 is P[0] only.",
    ]
    for side, trace in (("a", trace_a), ("b", trace_b)):
        prefill = _trace_stat(trace, "prefill_lines_skipped")
        malformed = _trace_stat(trace, "malformed_lines")
        missing_p = _trace_stat(trace, "p_missing_lines")
        dup_e = _trace_stat(trace, "duplicate_E_lines")
        dup_p = _trace_stat(trace, "duplicate_P_lines")
        if prefill:
            warnings.append(
                f"trace {side}: skipped {prefill} prefill P<n> line(s); they are not in any metric"
            )
        if malformed:
            warnings.append(
                f"trace {side}: skipped {malformed} malformed line(s); "
                f"samples are in trace_stats.{side}.malformed_samples"
            )
        if missing_p:
            warnings.append(
                f"trace {side}: {missing_p} decode line(s) had no P: field; "
                "treated as an empty prediction"
            )
        if dup_e:
            warnings.append(
                f"trace {side}: {dup_e} decode line(s) repeated an E expert id; "
                "deduplicated for set metrics"
            )
        if dup_p:
            warnings.append(
                f"trace {side}: {dup_p} decode line(s) repeated a P expert id; "
                "deduplicated preserving first rank"
            )

    has_p_a = any(step.pred for layer in compared_layers for step in trace_a.layers[layer])
    has_p_b = any(step.pred for layer in compared_layers for step in trace_b.layers[layer])
    n_steps_p = int(aggregate["n_steps_P"])
    total = int(aggregate["n_steps_total"])
    if not has_p_a and not has_p_b:
        warnings.append(
            "P metrics are null: neither trace contains a non-empty P: prediction list"
        )
    elif not has_p_a or not has_p_b:
        side = "a" if not has_p_a else "b"
        warnings.append(
            f"P metrics are null: trace {side} has no non-empty P: list, so top-1/RBO "
            "cannot be compared"
        )
    elif n_steps_p < total:
        warnings.append(
            f"P metrics cover {n_steps_p}/{total} step pairs; steps where either side "
            "had an empty P: list are null"
        )

    empty_steps = int(aggregate["empty_steps"])
    if empty_steps:
        warnings.append(
            f"{empty_steps} step pair(s) had empty E on both sides; Jaccard is defined as 1.0 "
            "and counted in empty_steps"
        )
    if allow_layer_subset:
        warnings.append(
            f"layer subset mode: compared {len(compared_layers)} layer(s); "
            f"skipped_layers={list(skipped_layers)}"
        )
    sequence_warning = _sequence_warning(trace_a, trace_b, compared_layers)
    if sequence_warning:
        warnings.append(sequence_warning)
    return warnings


def build_report(
    trace_a: Trace,
    trace_b: Trace,
    p_values: Sequence[float] = DEFAULT_P_VALUES,
    expected_layers: int = 0,
    min_steps: int = 1,
    allow_layer_subset: bool = False,
    redact_paths: bool = True,
) -> dict[str, object]:
    if not p_values:
        raise TraceError("at least one --p value is required")
    compared_layers = _validate(
        trace_a, trace_b, expected_layers, min_steps, allow_layer_subset
    )
    skipped_layers = sorted((set(trace_a.layers) | set(trace_b.layers)) - set(compared_layers))

    layer_reports: dict[str, dict[str, object]] = {}
    for layer in compared_layers:
        layer_reports[str(layer)] = compare_layer(
            layer, trace_a.layers[layer], trace_b.layers[layer], p_values
        )

    all_steps = [
        step
        for layer_report in layer_reports.values()
        for step in layer_report["steps"]  # type: ignore[union-attr]
    ]
    jaccards = [float(step["jaccard_E"]) for step in all_steps]
    overlaps = [float(step["overlap_E"]) for step in all_steps]
    exacts = [float(step["exact_set_match"]) for step in all_steps]
    deltas = [float(step["size_delta"]) for step in all_steps]
    jaccards_p = [
        float(step["jaccard_P"]) for step in all_steps if step["jaccard_P"] is not None
    ]
    top1s = [
        float(step["top1_agreement_P"])
        for step in all_steps
        if step["top1_agreement_P"] is not None
    ]
    rbo_values: dict[float, list[float]] = {p: [] for p in p_values}
    for step in all_steps:
        step_rbo = step["rbo"]
        assert isinstance(step_rbo, dict)
        for p in p_values:
            value = step_rbo.get(p_key(p))
            if value is not None:
                rbo_values[p].append(float(value))

    layer_step_counts = {int(report["n_steps"]) for report in layer_reports.values()}
    n_steps = next(iter(layer_step_counts)) if len(layer_step_counts) == 1 else None

    aggregate: dict[str, object] = {
        "n_layers": len(compared_layers),
        "n_steps": n_steps,
        "n_steps_total": len(all_steps),
        "n_steps_P": len(jaccards_p),
        "min_layer_steps": min(layer_step_counts) if layer_step_counts else 0,
        "max_layer_steps": max(layer_step_counts) if layer_step_counts else 0,
        "mean_jaccard_E": _mean(jaccards),
        "median_jaccard_E": statistics.median(jaccards) if jaccards else None,
        "min_jaccard_E": min(jaccards) if jaccards else None,
        "max_jaccard_E": max(jaccards) if jaccards else None,
        "mean_overlap_E": _mean(overlaps),
        "exact_match_rate": _mean(exacts),
        "n_mismatch_steps": sum(1 for value in exacts if value == 0.0),
        "n_mismatch_layers": sum(
            1 for report in layer_reports.values() if int(report["n_mismatch_steps"]) > 0
        ),
        "mean_size_delta": _mean(deltas),
        "mean_jaccard_P": _mean(jaccards_p),
        "median_jaccard_P": statistics.median(jaccards_p) if jaccards_p else None,
        "top1_agreement_P": _mean(top1s),
        "rbo": {p_key(p): _mean(rbo_values[p]) for p in p_values},
        "empty_steps": sum(1 for step in all_steps if step["empty"]),
        "one_sided_empty_steps": sum(
            1 for step in all_steps if step["n_A"] == 0 or step["n_B"] == 0
        ) - sum(1 for step in all_steps if step["empty"]),
        "skipped_layers": skipped_layers,
        "malformed_lines": {
            "a": _trace_stat(trace_a, "malformed_lines"),
            "b": _trace_stat(trace_b, "malformed_lines"),
        },
        "prefill_lines_skipped": {
            "a": _trace_stat(trace_a, "prefill_lines_skipped"),
            "b": _trace_stat(trace_b, "prefill_lines_skipped"),
        },
    }
    for p in p_values:
        aggregate[_p_scalar_name(p)] = _mean(rbo_values[p])
    aggregate.setdefault("rbo_p90", None)
    aggregate.setdefault("rbo_p98", None)

    def display(path: str) -> str:
        return os.path.basename(path) if redact_paths else path

    report: dict[str, object] = {
        "version": 1,
        "tool": "tools/moe/router_overlap.py",
        "a": display(trace_a.path),
        "b": display(trace_b.path),
        "redacted": redact_paths,
        "skipped_layers": skipped_layers,
        "options": {
            "p": list(p_values),
            "layers": expected_layers,
            "min_steps": min_steps,
            "allow_layer_subset": allow_layer_subset,
            "redact_paths": redact_paths,
        },
        "trace_stats": {
            "a": trace_a.stats,
            "b": trace_b.stats,
        },
        "layers": layer_reports,
        "aggregate": aggregate,
        "warnings": _warnings(
            trace_a, trace_b, compared_layers, skipped_layers, allow_layer_subset, aggregate
        ),
    }
    return report


def _format_tsv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_tsv(path: str, report: dict[str, object]) -> None:
    p_values = report["options"]["p"]  # type: ignore[index]
    assert isinstance(p_values, list)
    extra_p = [
        p for p in p_values if _p_scalar_name(float(p)) not in ("rbo_p90", "rbo_p98")
    ]
    columns = [
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
    ] + [_p_scalar_name(float(p)) for p in extra_p]
    lines = ["\t".join(columns)]
    layers = report["layers"]
    assert isinstance(layers, dict)
    for layer in sorted(layers, key=int):
        row = layers[layer]
        assert isinstance(row, dict)
        values: list[object] = [
            row["layer"],
            row["n_steps"],
            row["jaccard_E"],
            row["mean_overlap_E"],
            row["exact_match_rate"],
            row["mean_size_delta"],
            row["jaccard_P"],
            row["rbo_p90"],
            row["rbo_p98"],
            row["top1_agreement_P"],
        ] + [row.get(_p_scalar_name(float(p))) for p in extra_p]
        lines.append("\t".join(_format_tsv_value(value) for value in values))
    try:
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
    except OSError as exc:
        raise TraceError(f"cannot write TSV {path!r}: {exc.strerror}") from exc


def write_json(path: str, report: dict[str, object]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
    except OSError as exc:
        raise TraceError(f"cannot write JSON {path!r}: {exc.strerror}") from exc


def _summary_line(report: dict[str, object]) -> str:
    aggregate = report["aggregate"]
    assert isinstance(aggregate, dict)
    n_steps = aggregate["n_steps"]
    steps = (
        str(n_steps)
        if n_steps is not None
        else f"{aggregate['min_layer_steps']}..{aggregate['max_layer_steps']}"
    )
    return (
        f"layers={aggregate['n_layers']} steps={steps} "
        f"mean_jaccard_E={float(aggregate['mean_jaccard_E']):.6f} "
        f"exact={float(aggregate['exact_match_rate']):.6f}"
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float") from exc
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router_overlap.py",
        description=(
            "Compare expert-set identity between two LLAMA_MOE_SWEEP decode traces "
            "(decode L<n> lines only; prefill P<n> lines are skipped)."
        ),
        epilog=(
            "Input lines: L<n> E:<used expert ids> P:<previous-layer prediction, "
            "descending score>. E is an unordered set (engine/file order), not router "
            "rank; only P[0] is top-1.\n\n"
            "Metrics per layer/step: jaccard_E, overlap_E, exact_set_match, size_delta; "
            "P metrics (jaccard_P, RBO, top1_agreement_P) only when both sides have P.\n\n"
            "Exit codes: 0 success, 2 inconsistent/invalid input (fail-closed).\n\n"
            "Examples:\n"
            "  router_overlap.py A.txt B.txt --out-json router.json --out-tsv router.tsv\n"
            "  router_overlap.py A.txt B.txt --allow-layer-subset --layers 40\n"
            "  router_overlap.py A.txt B.txt --p 0.9 0.98 --min-steps 8\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace_a", help="first LLAMA_MOE_SWEEP trace (A)")
    parser.add_argument("trace_b", help="second LLAMA_MOE_SWEEP trace (B)")
    parser.add_argument("--out-json", metavar="PATH", help="write the version-1 JSON report")
    parser.add_argument(
        "--out-tsv",
        metavar="PATH",
        help=(
            "write per-layer TSV: layer n_steps jaccard_E mean_overlap_E "
            "exact_match_rate mean_size_delta jaccard_P rbo_p90 rbo_p98 "
            "top1_agreement_P (null metrics are empty fields)"
        ),
    )
    parser.add_argument(
        "--layers",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="expected layer count in each trace (0 = infer from the trace; default 0)",
    )
    parser.add_argument(
        "--min-steps",
        type=_positive_int,
        default=1,
        metavar="N",
        help="fail (exit 2) if any compared layer has fewer than N steps (default 1)",
    )
    parser.add_argument(
        "--p",
        nargs="+",
        type=_probability,
        default=list(DEFAULT_P_VALUES),
        metavar="P",
        dest="p_values",
        help="RBO persistence value(s), each in (0,1) (default: 0.9 0.98)",
    )
    parser.add_argument(
        "--allow-layer-subset",
        action="store_true",
        help="compare only the intersection of layer ids and record skipped_layers",
    )
    parser.add_argument(
        "--redact-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write only basenames for a/b in JSON (default: on; use --no-redact-paths to keep paths)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    p_values = list(dict.fromkeys(args.p_values))
    try:
        trace_a = parse_trace(args.trace_a)
        trace_b = parse_trace(args.trace_b)
        report = build_report(
            trace_a,
            trace_b,
            p_values=p_values,
            expected_layers=args.layers,
            min_steps=args.min_steps,
            allow_layer_subset=args.allow_layer_subset,
            redact_paths=args.redact_paths,
        )
        if args.out_json:
            write_json(args.out_json, report)
        if args.out_tsv:
            write_tsv(args.out_tsv, report)
    except TraceError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        for detail in exc.details:
            print(detail, file=sys.stderr)
        return 2

    print(_summary_line(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
