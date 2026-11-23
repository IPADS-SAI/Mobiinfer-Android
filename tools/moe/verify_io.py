#!/usr/bin/env python3
"""Decompose verify cycles from ``profile-data.json`` into four wall segments.

Python 3 standard library only; read-only offline analysis (no device, no
network).  This is the reusable version of the one-off T-0340
``analyze_verify_io.py``: it consumes the profile captures directly and reports

    wall      = b - a of one ``forwards[]`` entry with phase == "verify"
    dsp       = |union(events[] [a,b))|            (all sessions by default)
    read      = |union(pagecache[] [a,b))|         (demand + prefetch)
    overlap   = |dsp & read|
    dsp_only  = |dsp - read|                       (compute on the critical path)
    read_only = |read - dsp|                       (DSP idle while a read is in flight)
    neither   = wall - |dsp | read|                (host gap / no I/O)

The four segments are a partition of the cycle wall and must sum to it.  They
are a *modelling* decomposition based on co-occurrence, not hardware counters:
the DSP intervals come from host-calibrated qtimer timestamps and carry
``placement_error_us``; the read union is host ``preadv`` time, not UFS
utilisation.  Do not cite either as an exact hardware bus/busy number.

Reads are attributed to a cycle via ``pagecache[].fw_ids``.  A read may be
listed for more than one cycle; its in-cycle portion is used for the four-way
partition and the out-of-cycle portion is reported as ``clipped_us`` (this is
the T-0340 convention).  No read is silently dropped.

Usage:
    python3 tools/moe/verify_io.py --profile profile-data-20261110.json
    python3 tools/moe/verify_io.py --profile a.json --profile b.json \
        --out-dir out --per-layer

Optional uncertainty (additive; ``VERSION`` and the default JSON/TSV/human
output stay unchanged):

    --bootstrap N --seed S
        Resample verify cycles with replacement and report a nearest-index
        percentile interval for the wall-weighted percentages plus the
        per-cycle ``read_only_us`` median.  This is host co-occurrence
        modelling uncertainty, not a hardware-counter confidence interval.
    --placement-sensitivity
        Widen (``a-=e, b+=e``) or narrow (``a+=e, b-=e``) every DSP event by
        its ``placement_error_us``; reads and wall stay fixed.  Events without
        the field are treated as zero and counted in ``skipped_reasons``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

VERSION = 1

# --------------------------------------------------------------- interval ops


def merge(intervals: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """Merge half-open ``[a,b)`` intervals; drop non-positive ones."""
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    out: list[tuple[float, float]] = []
    for a, b in ordered:
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def total(intervals: Iterable[Sequence[float]]) -> float:
    """Total length of the union of ``intervals``."""
    return sum(b - a for a, b in merge(intervals))


def intersect_total(
    a_iv: Iterable[Sequence[float]], b_iv: Iterable[Sequence[float]]
) -> float:
    """Length of the intersection of two interval unions."""
    a = merge(a_iv)
    b = merge(b_iv)
    i = j = 0
    acc = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            acc += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return acc


def clip(
    intervals: Iterable[Sequence[float]], windows: Iterable[Sequence[float]]
) -> list[tuple[float, float]]:
    """Intersect each interval with the union of ``windows``."""
    win = merge(windows)
    out: list[tuple[float, float]] = []
    for a, b in merge(intervals):
        for wa, wb in win:
            lo, hi = max(a, wa), min(b, wb)
            if hi > lo:
                out.append((lo, hi))
    return out


# ------------------------------------------------------------------ datatypes


@dataclass
class Cycle:
    """One verify cycle (one ``forwards[]`` entry)."""

    cycle: int
    fw: Any
    rows: Any
    start_us: float
    end_us: float
    wall_us: float
    dsp_us: float
    demand_us: float
    prefetch_us: float
    read_union_us: float
    read_union_raw_us: float
    overlap_us: float
    dsp_only_us: float
    read_only_us: float
    neither_us: float
    demand_share: float
    eff_gbs: float
    eff_gbs_total_bytes: float
    clipped_us: float
    reads: int
    prefetch_reads: int
    read_bytes: int
    demand_bytes: int
    prefetch_bytes: int
    nonresident_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------- helpers


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_id(value: Any) -> bool:
    return isinstance(value, (int, str)) and not isinstance(value, bool)


def _section(data: dict, name: str, reasons: Counter) -> list:
    value = data.get(name)
    if not isinstance(value, list):
        reasons[f"missing_or_bad_{name}"] += 1
        return []
    return value


def _pctl(values: Sequence[float], p: float) -> Optional[float]:
    """Nearest-index percentile, same convention as T-0340's analyzer.

    ``values[round((n-1)*p)]`` on the sorted list, so a single sample returns
    itself and p90 of 4 samples is the maximum.
    """
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    return ordered[min(len(ordered) - 1, max(0, idx))]


def _empty_summary() -> dict:
    return {
        "cycles": 0,
        "wall_us": 0.0,
        "dsp_us": 0.0,
        "demand_us": 0.0,
        "prefetch_us": 0.0,
        "read_union_us": 0.0,
        "overlap_us": 0.0,
        "dsp_only_us": 0.0,
        "read_only_us": 0.0,
        "neither_us": 0.0,
        "dsp_busy_pct": 0.0,
        "read_covered_pct": 0.0,
        "overlap_pct": 0.0,
        "dsp_only_pct": 0.0,
        "read_only_pct": 0.0,
        "neither_pct": 0.0,
        "demand_share": 0.0,
        "demand_byte_share": 0.0,
        "eff_gbs": 0.0,
        "eff_gbs_total_bytes": 0.0,
        "read_bytes": 0,
        "demand_bytes": 0,
        "prefetch_bytes": 0,
        "nonresident_bytes": 0,
        "reads": 0,
        "prefetch_reads": 0,
        "clipped_us": 0.0,
        "skipped": 0,
    }


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


# ------------------------------------------------------ optional uncertainty

# Wall-weighted percentages reported by the optional bootstrap.  The extra
# ``read_only_us_median`` metric resamples the *per-cycle* read_only_us values,
# so it answers a different question from the wall-weighted read_only_pct.
BOOTSTRAP_METRICS = (
    "dsp_busy_pct",
    "read_covered_pct",
    "overlap_pct",
    "dsp_only_pct",
    "read_only_pct",
    "neither_pct",
)

_SUM_TO_PCT = {
    "dsp_busy_pct": "dsp_us",
    "read_covered_pct": "read_union_us",
    "overlap_pct": "overlap_us",
    "dsp_only_pct": "dsp_only_us",
    "read_only_pct": "read_only_us",
    "neither_pct": "neither_us",
}


def _bootstrap_summary(cycles: Sequence[Cycle], n: int, seed: int) -> dict:
    """Bootstrap wall-weighted percentages by resampling verify cycles.

    One draw samples ``len(cycles)`` cycles with replacement (the cycle is the
    independent unit); no other unit is resampled.  Percentiles use the same
    nearest-index convention as :func:`_pctl`.
    """
    n_eff = len(cycles)
    out: dict[str, Any] = {
        "n": n,
        "seed": seed,
        "n_eff": n_eff,
        "percentile_method": (
            "nearest-index: sorted[round((k-1)*p)], round-half-even"
        ),
        "metrics": {},
    }
    method_note = (
        "resampling unit: one verify cycle, drawn with replacement "
        f"({n_eff} cycles per draw); percentiles are nearest-index, the same "
        "convention as the existing _pctl.  This is host co-occurrence "
        "modelling uncertainty, not a hardware-counter confidence interval."
    )
    if n_eff == 0:
        out["estimable"] = False
        out["note"] = "no verify cycles; bootstrap not estimable. " + method_note
        return out

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        name: [] for name in BOOTSTRAP_METRICS
    }
    samples["read_only_us_median"] = []
    for _ in range(n):
        draw = [cycles[rng.randrange(n_eff)] for _ in range(n_eff)]
        wall = sum(c.wall_us for c in draw)
        sums = {
            "dsp_us": sum(c.dsp_us for c in draw),
            "read_union_us": sum(c.read_union_us for c in draw),
            "overlap_us": sum(c.overlap_us for c in draw),
            "dsp_only_us": sum(c.dsp_only_us for c in draw),
            "read_only_us": sum(c.read_only_us for c in draw),
            "neither_us": sum(c.neither_us for c in draw),
        }
        for name in BOOTSTRAP_METRICS:
            samples[name].append(_pct(sums[_SUM_TO_PCT[name]], wall))
        samples["read_only_us_median"].append(
            _pctl([c.read_only_us for c in draw], 0.5)
        )

    for name, values in samples.items():
        out["metrics"][name] = {
            "p50": _pctl(values, 0.50),
            "p2_5": _pctl(values, 0.025),
            "p97_5": _pctl(values, 0.975),
        }
    out["estimable"] = n_eff >= 2
    out["note"] = method_note
    if n_eff == 1:
        out["note"] = (
            "n_eff=1: every resample is the same cycle, so the interval is "
            "degenerate and not estimable. " + method_note
        )
    return out


def _decompose_dsp(
    wall: float,
    dsp_iv: Sequence[Sequence[float]],
    read_iv: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Four-way decomposition for an arbitrary (already clipped) DSP union."""
    dsp_us = total(dsp_iv)
    read_us = total(read_iv)
    overlap = intersect_total(dsp_iv, read_iv)
    dsp_only = dsp_us - overlap
    read_only = read_us - overlap
    neither = wall - dsp_us - read_us + overlap
    # Same floating-point hygiene as the main path.
    if -1e-9 < dsp_only < 0:
        dsp_only = 0.0
    if -1e-9 < read_only < 0:
        read_only = 0.0
    if -1e-9 < neither < 0:
        neither = 0.0
    return {
        "dsp_us": dsp_us,
        "read_union_us": read_us,
        "overlap_us": overlap,
        "dsp_only_us": dsp_only,
        "read_only_us": read_only,
        "neither_us": neither,
    }


def _placement_summary(
    cycle_inputs: Sequence[dict], base: dict, mode: str
) -> dict:
    """Recompute the decomposition after widening/narrowing DSP events.

    ``wide``:   a -= placement_error_us, b += placement_error_us
    ``narrow``: a += placement_error_us, b -= placement_error_us
    Non-positive adjusted intervals are dropped.  Read intervals and wall are
    untouched (the read union is host preadv time, not a DSP interval).
    """
    totals = {
        "dsp_us": 0.0,
        "read_union_us": 0.0,
        "overlap_us": 0.0,
        "dsp_only_us": 0.0,
        "read_only_us": 0.0,
        "neither_us": 0.0,
    }
    wall_total = 0.0
    for item in cycle_inputs:
        adjusted = []
        for a, b, err in item["events"]:
            if mode == "wide":
                lo, hi = a - err, b + err
            else:
                lo, hi = a + err, b - err
            if hi > lo:
                adjusted.append((lo, hi))
        dsp_iv = clip(adjusted, [(item["start"], item["end"])])
        parts = _decompose_dsp(item["wall"], dsp_iv, item["read_iv"])
        for key in totals:
            totals[key] += parts[key]
        wall_total += item["wall"]

    pcts = {
        name: _pct(totals[src], wall_total)
        for name, src in _SUM_TO_PCT.items()
    }
    record: dict[str, Any] = {
        "mode": mode,
        "dsp_busy_pct": pcts["dsp_busy_pct"],
        "overlap_pct": pcts["overlap_pct"],
        "dsp_only_pct": pcts["dsp_only_pct"],
        "read_only_pct": pcts["read_only_pct"],
        "neither_pct": pcts["neither_pct"],
        "delta_vs_base": {
            name: pcts[name] - base[name]
            for name in (
                "dsp_busy_pct",
                "overlap_pct",
                "dsp_only_pct",
                "read_only_pct",
                "neither_pct",
            )
        },
    }
    return record


def _add_optional_summaries(
    entry: dict,
    cycles: Sequence[Cycle],
    cycle_inputs: Sequence[dict],
    bootstrap: int,
    seed: int,
    placement_sensitivity: bool,
) -> None:
    """Attach the opt-in keys; never called on the default path."""
    if bootstrap:
        entry["summary"]["bootstrap"] = _bootstrap_summary(cycles, bootstrap, seed)
    if placement_sensitivity:
        base = entry["summary"]
        entry["summary"]["placement_sensitivity"] = {
            mode: _placement_summary(cycle_inputs, base, mode)
            for mode in ("wide", "narrow")
        }


# ------------------------------------------------------------------- analysis


def analyze_profile(
    path: str,
    session: Optional[str] = None,
    per_layer: bool = False,
    bootstrap: int = 0,
    seed: int = 20261123,
    placement_sensitivity: bool = False,
) -> dict:
    """Analyse one ``profile-data.json`` capture.

    Never raises on bad input: unreadable files and malformed entries are
    counted in ``summary.skipped`` / ``skipped_reasons`` instead.
    """
    reasons: Counter = Counter()
    entry: dict = {
        "path": os.path.abspath(path),
        "cycles": [],
        "summary": _empty_summary(),
        "skipped": 0,
    }

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        reasons["profile_load_error"] += 1
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["skipped"] = 1
        entry["skipped_reasons"] = dict(reasons)
        _finish_summary(entry, per_layer=per_layer)
        _add_optional_summaries(
            entry, [], [], bootstrap, seed, placement_sensitivity
        )
        return entry

    if not isinstance(data, dict):
        reasons["profile_not_object"] += 1
        entry["error"] = "top-level JSON value is not an object"
        entry["skipped"] = 1
        entry["skipped_reasons"] = dict(reasons)
        _finish_summary(entry, per_layer=per_layer)
        _add_optional_summaries(
            entry, [], [], bootstrap, seed, placement_sensitivity
        )
        return entry

    forwards = _section(data, "forwards", reasons)
    events = _section(data, "events", reasons)
    pagecache = _section(data, "pagecache", reasons)

    # --- verify forwards -------------------------------------------------
    verify_forwards: list[dict] = []
    verify_ids: set = set()
    for fw in forwards:
        if not isinstance(fw, dict):
            reasons["bad_forward"] += 1
            continue
        if fw.get("phase") != "verify":
            continue
        fid = fw.get("id")
        a, b = fw.get("a"), fw.get("b")
        if not _is_id(fid) or not _is_num(a) or not _is_num(b) or b <= a:
            reasons["bad_verify_forward"] += 1
            continue
        if fid in verify_ids:
            reasons["duplicate_verify_forward"] += 1
            continue
        verify_ids.add(fid)
        verify_forwards.append(fw)

    # --- events by forward ----------------------------------------------
    # Keep the placement error alongside each interval for the optional
    # sensitivity analysis; the default path uses only (a, b).
    events_by_fw: dict[Any, list[tuple[float, float, float]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            reasons["bad_event"] += 1
            continue
        fid = event.get("fw")
        if not _is_id(fid) or fid not in verify_ids:
            continue
        a, b = event.get("a"), event.get("b")
        if not _is_num(a) or not _is_num(b) or b <= a:
            reasons["bad_event"] += 1
            continue
        if session is not None and event.get("session") != session:
            continue
        placement_error = event.get("placement_error_us")
        if _is_num(placement_error) and placement_error >= 0:
            err = float(placement_error)
        else:
            err = 0.0
            if placement_sensitivity:
                reasons["missing_placement_error_us"] += 1
        events_by_fw[fid].append((float(a), float(b), err))

    # --- reads by forward ------------------------------------------------
    reads_by_fw: dict[Any, list[dict]] = defaultdict(list)
    reads_by_idx: dict[int, dict] = {}
    for idx, read in enumerate(pagecache):
        if not isinstance(read, dict):
            reasons["bad_read"] += 1
            continue
        a, b = read.get("a"), read.get("b")
        if not _is_num(a) or not _is_num(b) or b <= a:
            reasons["bad_read"] += 1
            continue
        fw_ids = read.get("fw_ids")
        if not isinstance(fw_ids, list) or not fw_ids:
            reasons["read_without_fw_ids"] += 1
            continue
        nbytes = read.get("bytes")
        nonresident = read.get("nonresident_bytes")
        if not _is_num(nbytes) and not _is_num(nonresident):
            reasons["read_without_bytes"] += 1
            continue
        if not _is_num(nbytes):
            nbytes = nonresident
        if not _is_num(nonresident):
            nonresident = nbytes
        rec = {
            "a": float(a),
            "b": float(b),
            "bytes": int(nbytes),
            "nonresident": int(nonresident),
            "prefetch": bool(read.get("prefetch")),
            "layer": read.get("layer"),
            "_idx": idx,
        }
        hit = False
        for fid in fw_ids:
            if _is_id(fid) and fid in verify_ids:
                reads_by_fw[fid].append(rec)
                hit = True
        if hit:
            reads_by_idx[idx] = rec

    # --- per-cycle decomposition ----------------------------------------
    cycles: list[Cycle] = []
    cycle_inputs: list[dict] = []
    seen_reads: set[int] = set()
    for number, fw in enumerate(verify_forwards, 1):
        fid = fw["id"]
        a = float(fw["a"])
        b = float(fw["b"])
        wall = b - a
        window = [(a, b)]

        raw_events = events_by_fw.get(fid, [])
        dsp_iv = clip(((ea, eb) for ea, eb, _err in raw_events), window)
        dsp_us = total(dsp_iv)

        reads = reads_by_fw.get(fid, [])
        for read in reads:
            seen_reads.add(read["_idx"])
        raw_read = merge((r["a"], r["b"]) for r in reads)
        read_iv = clip(raw_read, window)
        read_us = total(read_iv)
        read_raw_us = total(raw_read)
        clipped_us = max(0.0, read_raw_us - read_us)

        demand_raw = merge(
            (r["a"], r["b"]) for r in reads if not r["prefetch"]
        )
        prefetch_raw = merge(
            (r["a"], r["b"]) for r in reads if r["prefetch"]
        )
        demand_us = total(clip(demand_raw, window))
        prefetch_us = total(clip(prefetch_raw, window))

        overlap = intersect_total(dsp_iv, read_iv)
        dsp_only = dsp_us - overlap
        read_only = read_us - overlap
        neither = wall - dsp_us - read_us + overlap

        # Floating-point hygiene: the four segments must partition the wall.
        if -1e-9 < dsp_only < 0:
            dsp_only = 0.0
        if -1e-9 < read_only < 0:
            read_only = 0.0
        if -1e-9 < neither < 0:
            neither = 0.0
        if neither < -1e-6:
            reasons["negative_neither"] += 1
            neither = 0.0
        if abs(overlap + dsp_only + read_only + neither - wall) > 1e-6:
            reasons["invariant_violation"] += 1
            neither = wall - overlap - dsp_only - read_only

        demand_raw_us = total(demand_raw)
        demand_share = demand_raw_us / read_raw_us if read_raw_us else 0.0

        read_bytes = sum(r["bytes"] for r in reads)
        nonresident_bytes = sum(r["nonresident"] for r in reads)
        demand_bytes = sum(r["bytes"] for r in reads if not r["prefetch"])
        prefetch_bytes = read_bytes - demand_bytes
        eff_gbs = (
            nonresident_bytes / (read_us * 1e3) if read_us else 0.0
        )
        eff_total = read_bytes / (read_us * 1e3) if read_us else 0.0

        cycles.append(
            Cycle(
                cycle=number,
                fw=fid,
                rows=fw.get("rows"),
                start_us=a,
                end_us=b,
                wall_us=wall,
                dsp_us=dsp_us,
                demand_us=demand_us,
                prefetch_us=prefetch_us,
                read_union_us=read_us,
                read_union_raw_us=read_raw_us,
                overlap_us=overlap,
                dsp_only_us=dsp_only,
                read_only_us=read_only,
                neither_us=neither,
                demand_share=demand_share,
                eff_gbs=eff_gbs,
                eff_gbs_total_bytes=eff_total,
                clipped_us=clipped_us,
                reads=len(reads),
                prefetch_reads=sum(1 for r in reads if r["prefetch"]),
                read_bytes=read_bytes,
                demand_bytes=demand_bytes,
                prefetch_bytes=prefetch_bytes,
                nonresident_bytes=nonresident_bytes,
            )
        )
        cycle_inputs.append(
            {
                "start": a,
                "end": b,
                "wall": wall,
                "events": raw_events,
                "read_iv": read_iv,
            }
        )

    entry["cycles"] = [c.as_dict() for c in cycles]
    entry["skipped"] = sum(reasons.values())
    entry["skipped_reasons"] = dict(reasons)
    entry["summary"] = _summarize(cycles, reasons)
    if per_layer:
        entry["summary"]["per_layer"] = _per_layer(reads_by_idx, seen_reads)
    _add_optional_summaries(
        entry, cycles, cycle_inputs, bootstrap, seed, placement_sensitivity
    )
    return entry


def _summarize(cycles: Sequence[Cycle], reasons: Counter) -> dict:
    summary = _empty_summary()
    summary["cycles"] = len(cycles)
    summary["skipped"] = sum(reasons.values())
    summary["skipped_reasons"] = dict(reasons)

    summed = [
        "wall_us",
        "dsp_us",
        "demand_us",
        "prefetch_us",
        "read_union_us",
        "overlap_us",
        "dsp_only_us",
        "read_only_us",
        "neither_us",
        "clipped_us",
    ]
    for key in summed:
        summary[key] = sum(getattr(c, key) for c in cycles)
    for key in (
        "read_bytes",
        "demand_bytes",
        "prefetch_bytes",
        "nonresident_bytes",
        "reads",
        "prefetch_reads",
    ):
        summary[key] = sum(getattr(c, key) for c in cycles)

    wall = summary["wall_us"]
    summary["dsp_busy_pct"] = _pct(summary["dsp_us"], wall)
    summary["read_covered_pct"] = _pct(summary["read_union_us"], wall)
    summary["overlap_pct"] = _pct(summary["overlap_us"], wall)
    summary["dsp_only_pct"] = _pct(summary["dsp_only_us"], wall)
    summary["read_only_pct"] = _pct(summary["read_only_us"], wall)
    summary["neither_pct"] = _pct(summary["neither_us"], wall)

    demand_raw = sum(c.demand_share * c.read_union_raw_us for c in cycles)
    read_raw = sum(c.read_union_raw_us for c in cycles)
    summary["demand_share"] = demand_raw / read_raw if read_raw else 0.0
    summary["demand_byte_share"] = (
        summary["demand_bytes"] / summary["read_bytes"]
        if summary["read_bytes"]
        else 0.0
    )
    summary["eff_gbs"] = (
        summary["nonresident_bytes"] / (summary["read_union_us"] * 1e3)
        if summary["read_union_us"]
        else 0.0
    )
    summary["eff_gbs_total_bytes"] = (
        summary["read_bytes"] / (summary["read_union_us"] * 1e3)
        if summary["read_union_us"]
        else 0.0
    )
    return summary


def _per_layer(reads_by_idx: dict[int, dict], seen: set[int]) -> list[dict]:
    latencies: dict[Any, dict[str, list[float]]] = defaultdict(
        lambda: {"demand": [], "prefetch": []}
    )
    for idx in sorted(seen):
        read = reads_by_idx[idx]
        layer = read["layer"]
        if not isinstance(layer, int):
            continue
        kind = "prefetch" if read["prefetch"] else "demand"
        latencies[layer][kind].append((read["b"] - read["a"]) / 1e3)

    rows: list[dict] = []
    for layer in sorted(latencies):
        demand = latencies[layer]["demand"]
        prefetch = latencies[layer]["prefetch"]
        rows.append(
            {
                "layer": layer,
                "n_demand": len(demand),
                "demand_p50_ms": _pctl(demand, 0.50),
                "demand_p90_ms": _pctl(demand, 0.90),
                "demand_max_ms": max(demand) if demand else None,
                "n_prefetch": len(prefetch),
                "prefetch_p50_ms": _pctl(prefetch, 0.50),
                "prefetch_p90_ms": _pctl(prefetch, 0.90),
                "prefetch_max_ms": max(prefetch) if prefetch else None,
            }
        )
    return rows


def _finish_summary(entry: dict, per_layer: bool) -> None:
    """Recompute summary bookkeeping after an early return."""
    summary = entry["summary"]
    summary["skipped"] = entry.get("skipped", 0)
    summary["skipped_reasons"] = entry.get("skipped_reasons", {})
    if per_layer:
        summary["per_layer"] = []


# --------------------------------------------------------------------- output


TSV_COLUMNS = [
    "profile",
    "cycle",
    "fw",
    "wall_ms",
    "dsp_ms",
    "demand_ms",
    "prefetch_ms",
    "read_union_ms",
    "overlap_ms",
    "dsp_only_ms",
    "read_only_ms",
    "neither_ms",
    "demand_share",
    "eff_gbs",
    "clipped_us",
]

PER_LAYER_COLUMNS = [
    "profile",
    "layer",
    "n_demand",
    "demand_p50_ms",
    "demand_p90_ms",
    "demand_max_ms",
    "n_prefetch",
    "prefetch_p50_ms",
    "prefetch_p90_ms",
    "prefetch_max_ms",
]

DEFINITIONS = {
    "wall_us": "forward b - a for one forwards[] entry with phase == 'verify'",
    "dsp_us": "|union(events[].a,b)| for events with fw == cycle id; "
    "all sessions unless --session is given",
    "demand_us": "|union(pagecache[].a,b) with prefetch == false|, attributed "
    "to the cycle via fw_ids",
    "prefetch_us": "|union(pagecache[].a,b) with prefetch == true|, attributed "
    "to the cycle via fw_ids",
    "read_union_us": "|union(demand, prefetch)| clipped to the cycle wall",
    "read_union_raw_us": "read union before clipping to the cycle wall",
    "overlap_us": "|dsp & read_union|",
    "dsp_only_us": "|dsp - read_union|",
    "read_only_us": "|read_union - dsp|",
    "neither_us": "wall_us - |dsp | read_union| (host gap / no I/O)",
    "demand_share": "|demand union| / |read union|, both unclipped (time share)",
    "demand_byte_share": "demand_bytes / read_bytes (byte share, comparable "
    "with T-0340's 44.1/55.9)",
    "eff_gbs": "sum(nonresident_bytes) / read_union_us * 1e-3 (us -> ms -> s)",
    "eff_gbs_total_bytes": "sum(bytes) / read_union_us * 1e-3; includes "
    "resident/partial reads and reproduces T-0340's cyc4 3.13 GB/s",
    "clipped_us": "read union length outside the cycle wall; it is still "
    "attributed to the cycle but does not enter the four-way partition",
    "summary.*_pct": "segment / wall_us * 100",
    "per_layer": "read latency b - a (not wait time) per pagecache layer; "
    "nearest-index p50/p90 (T-0340 convention)",
}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def write_tsv(path: str, report: dict) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for profile in report["profiles"]:
            for cycle in profile["cycles"]:
                writer.writerow(
                    [
                        profile["path"],
                        cycle["cycle"],
                        cycle["fw"],
                        _fmt(cycle["wall_us"] / 1e3),
                        _fmt(cycle["dsp_us"] / 1e3),
                        _fmt(cycle["demand_us"] / 1e3),
                        _fmt(cycle["prefetch_us"] / 1e3),
                        _fmt(cycle["read_union_us"] / 1e3),
                        _fmt(cycle["overlap_us"] / 1e3),
                        _fmt(cycle["dsp_only_us"] / 1e3),
                        _fmt(cycle["read_only_us"] / 1e3),
                        _fmt(cycle["neither_us"] / 1e3),
                        _fmt(cycle["demand_share"]),
                        _fmt(cycle["eff_gbs"]),
                        _fmt(cycle["clipped_us"]),
                    ]
                )


def write_per_layer_tsv(path: str, report: dict) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PER_LAYER_COLUMNS)
        for profile in report["profiles"]:
            for row in profile["summary"].get("per_layer", []):
                writer.writerow(
                    [profile["path"]]
                    + [_fmt(row.get(col)) for col in PER_LAYER_COLUMNS[1:]]
                )


def write_json(path: str, report: dict) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def analyze_profiles(
    paths: Sequence[str],
    session: Optional[str] = None,
    per_layer: bool = False,
    bootstrap: int = 0,
    seed: int = 20261123,
    placement_sensitivity: bool = False,
) -> dict:
    return {
        "version": VERSION,
        "definitions": DEFINITIONS,
        "profiles": [
            analyze_profile(
                path,
                session=session,
                per_layer=per_layer,
                bootstrap=bootstrap,
                seed=seed,
                placement_sensitivity=placement_sensitivity,
            )
            for path in paths
        ],
    }


def print_report(report: dict) -> None:
    for profile in report["profiles"]:
        summary = profile["summary"]
        print(f"\n{profile['path']}")
        if profile.get("error"):
            print(f"  error         : {profile['error']}")
        print(f"  verify cycles : {summary['cycles']}")
        print(f"  wall          : {summary['wall_us'] / 1e3:.1f} ms")
        print(
            "  dsp union     : "
            f"{summary['dsp_us'] / 1e3:.1f} ms "
            f"({summary['dsp_busy_pct']:.2f}%)"
        )
        print(
            "  read union    : "
            f"{summary['read_union_us'] / 1e3:.1f} ms "
            f"({summary['read_covered_pct']:.2f}%)"
        )
        print(
            "  four segments : "
            f"overlap {summary['overlap_pct']:.2f}%  "
            f"dsp_only {summary['dsp_only_pct']:.2f}%  "
            f"read_only {summary['read_only_pct']:.2f}%  "
            f"neither {summary['neither_pct']:.2f}%"
        )
        print(
            "  demand share  : "
            f"time {summary['demand_share']:.4f}  "
            f"bytes {summary['demand_byte_share']:.4f}"
        )
        print(
            "  effective read: "
            f"{summary['eff_gbs']:.3f} GB/s "
            f"(nonresident; {summary['eff_gbs_total_bytes']:.3f} GB/s on all bytes)"
        )
        print(
            "  read bytes    : "
            f"{summary['read_bytes'] / 1e9:.3f} GB "
            f"({summary['reads']} reads, {summary['prefetch_reads']} prefetch)"
        )
        if summary["clipped_us"]:
            print(
                f"  clipped reads : {summary['clipped_us']:.1f} us outside cycles"
            )
        if summary["skipped"]:
            print(f"  skipped       : {summary['skipped']} bad/missing entries")
            for reason, count in sorted(summary.get("skipped_reasons", {}).items()):
                print(f"    - {count} x {reason}", file=sys.stderr)
        per_layer = summary.get("per_layer")
        if per_layer is not None:
            if not per_layer:
                print("  per-layer     : no reads with an integer layer")
            else:
                print(
                    "  per-layer read latency (ms, b-a): "
                    "layer  n_d  p50  p90  max  n_p  p50  p90  max"
                )
                for row in per_layer:
                    print(
                        "    L{layer:02d}  {n_demand:>3}  {d50}  {d90}  {dmax}"
                        "  {n_prefetch:>3}  {p50}  {p90}  {pmax}".format(
                            layer=row["layer"],
                            n_demand=row["n_demand"],
                            d50=_fmt(row["demand_p50_ms"], 3) or "-",
                            d90=_fmt(row["demand_p90_ms"], 3) or "-",
                            dmax=_fmt(row["demand_max_ms"], 3) or "-",
                            n_prefetch=row["n_prefetch"],
                            p50=_fmt(row["prefetch_p50_ms"], 3) or "-",
                            p90=_fmt(row["prefetch_p90_ms"], 3) or "-",
                            pmax=_fmt(row["prefetch_max_ms"], 3) or "-",
                        )
                    )
        boot = summary.get("bootstrap")
        if boot is not None:
            metric = boot.get("metrics", {}).get("read_only_pct")
            if metric and metric.get("p50") is not None:
                suffix = "" if boot.get("estimable") else "  [not estimable]"
                print(
                    "  bootstrap     : "
                    f"read_only p50 {metric['p50']:.2f}% "
                    f"[p2.5 {metric['p2_5']:.2f}%, "
                    f"p97.5 {metric['p97_5']:.2f}%] "
                    f"(n={boot['n']}, seed={boot['seed']}, "
                    f"n_eff={boot['n_eff']}){suffix}"
                )
            else:
                print(
                    "  bootstrap     : read_only p50 n/a "
                    f"(n={boot['n']}, seed={boot['seed']}, "
                    f"n_eff={boot['n_eff']}; not estimable)"
                )
        placement = summary.get("placement_sensitivity")
        if placement:
            wide = placement["wide"]
            narrow = placement["narrow"]
            print(
                "  placement     : wide/narrow read_only "
                f"{wide['read_only_pct']:.2f}% / "
                f"{narrow['read_only_pct']:.2f}%  "
                f"(dsp {wide['dsp_busy_pct']:.2f}% / "
                f"{narrow['dsp_busy_pct']:.2f}%)"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose verify cycles from profile-data.json into "
            "overlap / dsp_only / read_only / neither (stdlib only)."
        )
    )
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        metavar="FILE",
        help="profile-data JSON (repeatable)",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        help="write verify-io.tsv / verify-io.json (and per-layer.tsv) here",
    )
    parser.add_argument("--out-tsv", metavar="FILE", help="per-cycle TSV path")
    parser.add_argument("--out-json", metavar="FILE", help="full JSON path")
    parser.add_argument(
        "--session",
        metavar="HTP0",
        help="only count events from this DSP session (default: all)",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="aggregate read latency p50/p90/max per layer",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="wall-weighted cycle bootstrap, N >= 100 (default: 0 = off)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261123,
        metavar="S",
        help="bootstrap RNG seed (default: 20261123)",
    )
    parser.add_argument(
        "--placement-sensitivity",
        action="store_true",
        help="recompute the decomposition after +/- placement_error_us",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Validate before touching any profile: bad flags must exit 2 first.
    if args.bootstrap and args.bootstrap < 100:
        parser.error("--bootstrap must be 0 (off) or >= 100")
    report = analyze_profiles(
        args.profile,
        session=args.session,
        per_layer=args.per_layer,
        bootstrap=args.bootstrap,
        seed=args.seed,
        placement_sensitivity=args.placement_sensitivity,
    )

    tsv_path = args.out_tsv
    json_path = args.out_json
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        tsv_path = tsv_path or os.path.join(args.out_dir, "verify-io.tsv")
        json_path = json_path or os.path.join(args.out_dir, "verify-io.json")

    if tsv_path:
        write_tsv(tsv_path, report)
    if json_path:
        write_json(json_path, report)
    if args.out_dir and args.per_layer:
        write_per_layer_tsv(
            os.path.join(args.out_dir, "verify-io-per-layer.tsv"), report
        )

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
