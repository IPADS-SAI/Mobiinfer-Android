#!/usr/bin/env python3
"""A/B comparison of two `[moe-ssd]` read-ledger runs (offline, stdlib only).

This tool *imports* ``analyze_moe_reads.py`` for line parsing and for the
canonical ``published`` / ``total`` rollups.  It does not change that tool's
semantics: ``calls`` / ``pfread`` / ``bytes`` / ``pread`` are divided by
``span``, while ``runs`` / ``miss`` stay at their per-ubatch value; a window
with ``tok==4`` is the SPECN=3 verify graph (logged as ``prefill``) and is
reported separately from true prefill (``tok=2/18``).

What is added here
==================
* per-step pairing by window number ``#N`` (``published`` scope),
* per-run process-total pairing by replicate (``total`` scope),
* median paired log-ratio, fixed-seed percentile bootstrap 95% CI,
* exact one-sided sign test (ties excluded),
* a verify/prefill decomposition that never folds verify into prefill,
* hard errors (exit code 2) for missing, duplicate or out-of-order steps.

Scopes
======
``published`` (default)
    Drop the first decode window (cold ``#3``), use the remaining ``tok==4``
    windows.  The pairing unit is the window number; each step's log-ratio is
    the median over matched replicate files (``r1``/``r2``/...).  ``n_pairs``
    is therefore the number of decode windows (6 for T-0078), *not* the number
    of runs.

``total``
    Pair per replicate using the canonical per-run process totals
    (``runs_per_round = (merge-scan.runs - true-prefill.runs) / rounds`` and
    the exact calls/pfread deltas / rounds).  This is **not** a per-step
    statistic: ``n_pairs`` is the number of matched runs (4 for T-0078).
    It is labelled as such in JSON and Markdown.

Exit codes
==========
0  comparison written / self-test passed
1  self-test failed
2  malformed input: missing/duplicate/out-of-order windows, missing required
   fields, unmatched replicate or step sets.  No JSON/Markdown is written.

Usage
=====
    python3 tools/moe/compare_moe_runs.py \
      --a <err-or-dir> --b <err-or-dir> \
      --tags-a idA --tags-b ordB \
      --json out.json --markdown out.md \
      [--scope published|total] [--reps 10000] [--seed 20261122]

    python3 tools/moe/compare_moe_runs.py --self-test

The self-test is offline (no phone, no A100, no network) and reads the T-0078
run directory plus the T-0110 ledger by absolute path.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import statistics as st
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the canonical parser without copying it (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_moe_reads as amr  # noqa: E402

SCHEMA = "moe-read-compare-v1"
TOOL = "tools/moe/compare_moe_runs.py"

QUEUE_RUN_DIR = (Path(os.environ["QUEUE_RUN_DIR"]).expanduser()
                 if os.environ.get("QUEUE_RUN_DIR") else None)

DEFAULT_T0078_RUNS = Path(
    "/home/yydh/.local/state/dpsk-queue/runs/T-0078/"
    "run-20261122T092959-w1-1/artifacts/phone/runs"
)
DEFAULT_T0110_LEDGER = Path(
    "/home/yydh/.local/state/dpsk-queue/runs/T-0110/"
    "run-20261122T131308-w3-1/artifacts/ledger.json"
)

# Published metrics: (short key, field in analyze_moe_reads published summary,
# per-step field in the record).  The per-step field is what gets paired.
PUBLISHED_METRICS = (
    ("calls", "calls_per_step", "calls"),
    ("pfread", "pfread_per_step", "pfread"),
    ("runs", "runs_per_step", "runs"),
    ("miss", "miss_per_step", "miss"),
    ("syscalls", "syscalls_per_step", "syscalls"),
    ("bytes", "bytes_per_step", "bytes_mib"),
    ("pread_sum", "pread_sum_per_step", "pread_sum_ms"),
    ("avg_run", "avg_run", "avg_run"),
)

# Total metrics: (short key, field in analyze_moe_reads total summary).
TOTAL_METRICS = (
    ("runs", "runs_per_round"),
    ("calls", "calls_per_round"),
    ("pfread", "pfread_per_round"),
    ("bytes", "bytes_per_round"),
    ("demand_calls", "demand_calls_per_round"),
)

# Published values already published for T-0078 A/C/B (T-0096/T-0110).
# Tolerance is 0.05: display rounding only, deliberately *not* a loose
# "approximately equals" band.  Any real change must fail the self-test.
SELFTEST_TOL = 0.05
EXPECTED_PUBLISHED = {
    "idA": {"runs": 96.5, "calls": 373.0, "pfread": 262.0},
    "ordA": {"runs": 86.5, "calls": 373.0, "pfread": 262.0},
    "ordB": {"runs": 86.5, "calls": 355.5, "pfread": 262.0},
}
EXPECTED_TOTAL = {
    "idA": {"runs": 124.96428571428572, "calls": 299.0357142857143,
            "pfread": 188.85714285714286},
    "ordA": {"runs": 110.71428571428571, "calls": 298.92857142857144,
             "pfread": 188.78571428571428},
    "ordB": {"runs": 110.71428571428571, "calls": 283.0357142857143,
             "pfread": 188.85714285714286},
}
# Labels used by analyze_moe_reads.ARMS / the T-0110 ledger.
ARM_LABELS = {"idA": "A identity+merge0", "ordA": "C ordered+merge0",
              "ordB": "B ordered+merge2"}


class InputError(Exception):
    """Malformed / unpairable input.  CLI maps this to exit code 2."""


# ---------------------------------------------------------------------------
# small numeric helpers (kept local on purpose: gates/ is not imported)
# ---------------------------------------------------------------------------

def percentile(sorted_values, q):
    """Linear-interpolated percentile; sorted_values must be non-empty."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def log_ratio(a, b):
    """Return (log(a/b), zero_corrected).

    Counts can legitimately be zero.  A plain log is undefined then, so use
    the Haldane-Anscombe half-count correction and flag it in the output.
    Two zeros mean "no difference" and need no correction.
    """
    if a is None or b is None:
        return None, False
    if a < 0 or b < 0:
        return None, False
    if a == 0 and b == 0:
        return 0.0, False
    if a > 0 and b > 0:
        return math.log(a / b), False
    return math.log((a + 0.5) / (b + 0.5)), True


def bootstrap_ci(values, reps, seed):
    """Fixed-seed percentile bootstrap of the median of ``values``.

    Each metric gets its own ``random.Random(seed)``; because the draw order is
    identical, all metrics of one comparison share the same resampled units
    (a coherent paired bootstrap).  n < 4 is explicitly insufficient.
    """
    n = len(values)
    if n < 4:
        return None
    rng = random.Random(seed)
    meds = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        meds.append(st.median(sample))
    meds.sort()
    return [percentile(meds, 0.025), percentile(meds, 0.975)]


def sign_test(values):
    """Exact one-sided binomial sign test; exact zeros are ties (excluded)."""
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    zero = sum(1 for v in values if v == 0)
    n = pos + neg
    if n == 0:
        # All pairs are identical; both one-sided p-values are 1 by definition.
        return {"n_positive": pos, "n_negative": neg, "n_zero": zero,
                "n_eff": 0, "p_one_sided_less": 1.0,
                "p_one_sided_greater": 1.0}
    denom = 2 ** n
    p_less = sum(math.comb(n, i) for i in range(0, pos + 1)) / denom
    p_greater = sum(math.comb(n, i) for i in range(pos, n + 1)) / denom
    # Clamp floating point noise so an all-tie / all-positive case is exactly 1.
    p_less = min(1.0, max(0.0, p_less))
    p_greater = min(1.0, max(0.0, p_greater))
    return {"n_positive": pos, "n_negative": neg, "n_zero": zero,
            "n_eff": n, "p_one_sided_less": p_less,
            "p_one_sided_greater": p_greater}


# ---------------------------------------------------------------------------
# input loading / validation
# ---------------------------------------------------------------------------

_REP_RE = re.compile(r"r(\d+)")


def _parse_tags(values):
    tags = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if part and part not in tags:
                tags.append(part)
    return tags


def _replicate_for(full_tag, prefix):
    """Return the replicate key, or None if this file is not part of the arm.

    A tag matches when it is the arm itself (``idA``) or a replicate of it
    (``idA-r1``).  Other suffixes such as ``ordA-v`` are a different arm and
    are deliberately *not* folded in: verify-slot runs are not replicates.
    """
    if full_tag == prefix:
        return "single"
    if full_tag.startswith(prefix + "-"):
        rest = full_tag[len(prefix) + 1:]
        if re.fullmatch(r"r\d+", rest):
            return rest
    return None


def _rep_sort_key(rep):
    m = _REP_RE.fullmatch(rep)
    return (0, int(m.group(1))) if m else (1, rep)


def _side_prefixes(raw_files, tags, label):
    if tags:
        if len(tags) != 1:
            raise InputError(
                f"{label}: exactly one arm tag per side is supported; got {tags}. "
                "Run separate comparisons for different arms.")
        return tags
    prefixes = sorted({re.sub(r"-r\d+$", "", tag) for _, tag in raw_files})
    if len(prefixes) != 1:
        raise InputError(
            f"{label}: input contains multiple tags {prefixes}; pass --tags-{label} "
            "to select one arm.")
    return prefixes


def _validate_records(parsed, path):
    seen = set()
    previous = None
    for rec in parsed["records"]:
        idx = rec["idx"]
        if idx in seen:
            raise InputError(f"{path}: duplicate window #{idx}")
        if previous is not None and idx <= previous:
            raise InputError(
                f"{path}: window #{idx} is out of order after #{previous}; "
                "refusing to pair unordered steps")
        seen.add(idx)
        previous = idx
        if rec["kind"] not in ("verify", "prefill"):
            raise InputError(
                f"{path}: window #{idx} cannot be classified as verify/prefill "
                "(missing tok=)")
        raw = rec["raw"]
        missing = [f for f in ("tok", "span", "calls", "pfread", "runs", "miss")
                   if raw.get(f) is None]
        if missing:
            raise InputError(
                f"{path}: window #{idx} missing required field(s): {','.join(missing)}")


def load_side(path, tags, label):
    """Load one arm from an .err file or a directory (with optional runs/)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise InputError(f"{label}: input not found: {p}")
    try:
        raw_files, rows, skipped = amr.collect_inputs([str(p)])
    except SystemExit as exc:  # collect_inputs uses SystemExit for bad paths
        raise InputError(f"{label}: {exc}") from None
    if not raw_files:
        raise InputError(f"{label}: no *.err files under {p}")

    prefixes = _side_prefixes(raw_files, tags, label)
    prefix = prefixes[0]
    selected = []
    for err, full_tag in raw_files:
        rep = _replicate_for(full_tag, prefix)
        if rep is None:
            continue
        selected.append({"tag": full_tag, "replicate": rep, "path": str(err)})
    if not selected:
        available = sorted({tag for _, tag in raw_files})
        raise InputError(
            f"{label}: tag {prefix!r} matched no file under {p}; available: {available}")

    selected.sort(key=lambda f: _rep_sort_key(f["replicate"]))
    reps = [f["replicate"] for f in selected]
    dups = sorted({r for r in reps if reps.count(r) > 1})
    if dups:
        raise InputError(f"{label}: duplicate replicate(s) {dups} for tag {prefix!r}")

    files = []
    for item in selected:
        parsed = amr.parse_file(Path(item["path"]), tag=item["tag"])
        _validate_records(parsed, item["path"])
        verify = [r for r in parsed["records"] if r["kind"] == "verify"]
        prefill = [r for r in parsed["records"] if r["kind"] == "prefill"]
        if not verify:
            raise InputError(
                f"{item['path']}: no tok==4 verify/decode windows; cannot pair steps")
        used = verify[1:] if len(verify) > 1 else verify
        summary = amr.summarize_run(parsed)
        files.append({
            "tag": item["tag"],
            "replicate": item["replicate"],
            "file": item["path"],
            "parsed": parsed,
            "summary": summary,
            "verify": verify,
            "used": used,
            "prefill": prefill,
            "steps": [r["idx"] for r in used],
            "prefill_steps": [r["idx"] for r in prefill],
        })

    canonical_steps = files[0]["steps"]
    canonical_verify = [r["idx"] for r in files[0]["verify"]]
    for f in files[1:]:
        if f["steps"] != canonical_steps:
            raise InputError(
                f"{label}: replicate {f['replicate']} has decode steps {f['steps']}, "
                f"but replicate {files[0]['replicate']} has {canonical_steps}; "
                "refusing to pair incomplete replicates")
        if [r["idx"] for r in f["verify"]] != canonical_verify:
            raise InputError(
                f"{label}: replicate {f['replicate']} has verify windows "
                f"{[r['idx'] for r in f['verify']]}, but replicate "
                f"{files[0]['replicate']} has {canonical_verify}")

    return {
        "label": label,
        "path": str(p),
        "tags": [prefix],
        "replicates": reps,
        "files": files,
        "steps": canonical_steps,
        "verify_steps": canonical_verify,
        "prefill_steps": files[0]["prefill_steps"],
        "skipped": skipped,
        "results_tsv_rows": len(rows),
    }


def _side_by_rep(side):
    return {f["replicate"]: f for f in side["files"]}


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def _published_value(rec, field):
    value = rec["per_step"].get(field)
    return value


def _arm_median(side, scope, field):
    values = []
    for f in side["files"]:
        value = f["summary"][scope].get(field)
        if value is None:
            raise InputError(
                f"{side['label']}: {f['file']} has no {scope}.{field} "
                "(missing ledger fields or merge-scan)")
        values.append(value)
    if not values:
        raise InputError(f"{side['label']}: no values for {scope}.{field}")
    return st.median(values)


def _paired_published(a, b, metric_key, step_field):
    a_by_rep = _side_by_rep(a)
    b_by_rep = _side_by_rep(b)
    reps = sorted(a["replicates"], key=_rep_sort_key)
    logs = []
    pairs = []
    corrected = 0
    for idx in a["steps"]:
        step_logs = []
        a_vals = []
        b_vals = []
        for rep in reps:
            ra = next((r for r in a_by_rep[rep]["used"] if r["idx"] == idx), None)
            rb = next((r for r in b_by_rep[rep]["used"] if r["idx"] == idx), None)
            if ra is None or rb is None:
                raise InputError(f"missing step #{idx} in replicate {rep}")
            va = _published_value(ra, step_field)
            vb = _published_value(rb, step_field)
            lr, corr = log_ratio(va, vb)
            if lr is None:
                raise InputError(
                    f"{metric_key}: step #{idx} replicate {rep} has non-numeric/"
                    f"negative value A={va} B={vb}")
            corrected += int(corr)
            step_logs.append(lr)
            a_vals.append(va)
            b_vals.append(vb)
        step_log = st.median(step_logs)
        a_med = st.median(a_vals)
        b_med = st.median(b_vals)
        logs.append(step_log)
        pairs.append({"unit": f"#{idx}",
                      "a": a_med, "b": b_med,
                      "log_ratio": step_log, "ratio": math.exp(step_log),
                      "ratio_of_medians": (a_med / b_med) if b_med else None})
    return logs, pairs, corrected


def _paired_total(a, b, metric_field):
    a_by_rep = _side_by_rep(a)
    b_by_rep = _side_by_rep(b)
    reps = sorted(a["replicates"], key=_rep_sort_key)
    logs = []
    pairs = []
    corrected = 0
    for rep in reps:
        va = a_by_rep[rep]["summary"]["total"].get(metric_field)
        vb = b_by_rep[rep]["summary"]["total"].get(metric_field)
        lr, corr = log_ratio(va, vb)
        if lr is None:
            raise InputError(
                f"total {metric_field}: replicate {rep} has non-numeric/missing "
                f"value A={va} B={vb}")
        corrected += int(corr)
        logs.append(lr)
        pairs.append({"unit": rep, "a": va, "b": vb,
                      "log_ratio": lr, "ratio": math.exp(lr),
                      "ratio_of_medians": (va / vb) if vb else None})
    return logs, pairs, corrected


def _metric_block(a_median, b_median, logs, pairs, reps, seed, corrected):
    n = len(logs)
    insufficient = n < 4
    med_log = st.median(logs) if logs else None
    ci_log = bootstrap_ci(logs, reps, seed) if not insufficient else None
    ci_ratio = ([math.exp(ci_log[0]), math.exp(ci_log[1])]
                if ci_log is not None else None)
    ratio = math.exp(med_log) if med_log is not None else None
    return {
        "a_median": a_median,
        "b_median": b_median,
        "ratio_median": ratio,
        "log_ratio_median": med_log,
        "delta_pct": ((ratio - 1.0) * 100.0) if ratio is not None else None,
        "delta_pct_of_medians": (
            ((a_median / b_median) - 1.0) * 100.0
            if b_median not in (None, 0) and a_median is not None else None),
        "ci95_log_ratio": ci_log,
        "ci95_ratio": ci_ratio,
        "sign_test": sign_test(logs) if logs else None,
        "n_pairs": n,
        "insufficient_pairs": insufficient,
        "zero_corrected_pairs": corrected,
        "pairs": pairs,
    }


def _decompose(side):
    """Verify/prefill decomposition for one side; never mixes the two."""
    verify_fields = ("calls", "pfread", "runs", "miss", "syscalls", "avg_run")
    prefill_fields = ("calls", "pfread", "runs", "miss", "syscalls", "avg_run")

    def med_per_step(records, fields):
        out = {}
        for f in fields:
            vals = [r["per_step"].get(f) for r in records]
            vals = [v for v in vals if v is not None]
            out[f] = st.median(vals) if vals else None
        return out

    def med_or_none(values):
        vals = [v for v in values if v is not None]
        return st.median(vals) if vals else None

    verify_used = [r for f in side["files"] for r in f["used"]]
    prefill_all = [r for f in side["files"] for r in f["prefill"]]
    prefill_runs = []
    merge_runs = []
    runs_total = []
    rounds = []
    for f in side["files"]:
        total = f["summary"]["total"]
        prefill_runs.append(total.get("prefill_runs"))
        runs_total.append(total.get("runs_total"))
        rounds.append(total.get("rounds"))
        merge = (f["parsed"].get("merge_scan") or {}).get("runs")
        merge_runs.append(merge)
    return {
        "verify": {
            "definition": "tok==4 (SPECN=3 verify graph logged as prefill); first decode window dropped",
            "windows_total": sum(len(f["verify"]) for f in side["files"]),
            "windows_used": len(verify_used),
            "indices": side["steps"],
            "indices_all": side["verify_steps"],
            "span_sum": sum(r["span"] for r in verify_used),
            "per_step_median": med_per_step(verify_used, verify_fields),
            "runs_sum_per_replicate_median": st.median(
                [sum(r["raw"].get("runs") or 0 for r in f["used"])
                 for f in side["files"]]),
        },
        "prefill": {
            "definition": "tok!=4 (true prefill: tok=2/18), excluded from verify and from the paired test",
            "windows_total": len(prefill_all),
            "indices": sorted({r["idx"] for r in prefill_all}),
            "indices_all": sorted({r["idx"] for r in prefill_all}),
            "span_sum": sum(r["span"] for r in prefill_all),
            "per_step_median": med_per_step(prefill_all, prefill_fields),
            "runs_sum_per_replicate_median": med_or_none(prefill_runs),
        },
        "process_totals": {
            "merge_scan_runs_median": med_or_none(merge_runs),
            "prefill_runs_median": med_or_none(prefill_runs),
            "decode_runs_total_median": med_or_none(runs_total),
            "rounds_median": med_or_none(rounds),
        },
    }


def compare_sides(a, b, scope, reps=10000, seed=20261122):
    if scope not in ("published", "total"):
        raise InputError(f"unknown scope: {scope}")
    if reps < 1:
        raise InputError("--reps must be >= 1")

    if set(a["replicates"]) != set(b["replicates"]):
        raise InputError(
            f"replicate mismatch: A has {a['replicates']}, B has {b['replicates']}; "
            "refusing to compare unpaired runs")

    if scope == "published":
        if a["steps"] != b["steps"]:
            missing_a = sorted(set(b["steps"]) - set(a["steps"]))
            missing_b = sorted(set(a["steps"]) - set(b["steps"]))
            raise InputError(
                f"step mismatch: missing in A {missing_a}; missing in B {missing_b}; "
                f"A steps {a['steps']} B steps {b['steps']}")
        metrics = {}
        warnings = []
        for key, field, step_field in PUBLISHED_METRICS:
            logs, pairs, corrected = _paired_published(a, b, key, step_field)
            block = _metric_block(
                _arm_median(a, "published", field),
                _arm_median(b, "published", field),
                logs, pairs, reps, seed, corrected)
            block["field"] = field
            metrics[key] = block
            if corrected:
                warnings.append(
                    f"{key}: {corrected} paired value(s) had a zero; "
                    "Haldane-Anscombe +0.5 correction applied")
        pairing = {
            "unit": "window_index",
            "steps": a["steps"],
            "replicates": sorted(a["replicates"], key=_rep_sort_key),
            "dropped_first_decode": len(a["files"][0]["verify"]) > 1,
            "note": "each unit is a decode window #N; replicate log-ratios are medianed per step",
        }
    else:
        metrics = {}
        warnings = [
            "total scope is a per-run process total, not a per-step statistic; "
            "pairing unit is the replicate (run)",
            "total runs subtract true prefill (tok!=4) runs from merge-scan.runs",
        ]
        for key, field in TOTAL_METRICS:
            logs, pairs, corrected = _paired_total(a, b, field)
            block = _metric_block(
                _arm_median(a, "total", field),
                _arm_median(b, "total", field),
                logs, pairs, reps, seed, corrected)
            block["field"] = field
            metrics[key] = block
            if corrected:
                warnings.append(
                    f"{key}: {corrected} paired total(s) had a zero; "
                    "Haldane-Anscombe +0.5 correction applied")
        pairing = {
            "unit": "replicate",
            "steps": None,
            "replicates": sorted(a["replicates"], key=_rep_sort_key),
            "dropped_first_decode": len(a["files"][0]["verify"]) > 1,
            "note": "one pair per matched run/replicate",
        }

    any_insufficient = any(m["insufficient_pairs"] for m in metrics.values())

    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "status": "insufficient_pairs" if any_insufficient else "ok",
        "scope": scope,
        "scope_note": (
            "published: drop first decode; calls/pfread/bytes/pread /span; "
            "runs/miss raw; pair by window #N, median over replicates"
            if scope == "published" else
            "total: per-run process totals; runs = merge-scan.runs - true-prefill.runs; "
            "pair by replicate; NOT per-step"),
        "inputs": {
            "a": {"path": a["path"], "tags": a["tags"],
                  "replicates": a["replicates"],
                  "files": [{"tag": f["tag"], "replicate": f["replicate"],
                             "file": f["file"], "steps": f["steps"]}
                            for f in a["files"]],
                  "skipped": a["skipped"]},
            "b": {"path": b["path"], "tags": b["tags"],
                  "replicates": b["replicates"],
                  "files": [{"tag": f["tag"], "replicate": f["replicate"],
                             "file": f["file"], "steps": f["steps"]}
                            for f in b["files"]],
                  "skipped": b["skipped"]},
        },
        "pairing": pairing,
        "stats": {
            "reps": reps,
            "seed": seed,
            "ci_method": "percentile 2.5/97.5 over paired units; intervals are on log-ratio, ratio CI = exp(log CI)",
            "min_pairs_for_ci": 4,
            "insufficient_pairs": any_insufficient,
        },
        "metrics": metrics,
        "decomposition": {"a": _decompose(a), "b": _decompose(b)},
        "validation": {
            "a_steps": a["steps"],
            "b_steps": b["steps"],
            "a_prefill_steps": a["prefill_steps"],
            "b_prefill_steps": b["prefill_steps"],
            "missing_steps_in_a": sorted(set(b["steps"]) - set(a["steps"])),
            "missing_steps_in_b": sorted(set(a["steps"]) - set(b["steps"])),
            "duplicate_steps": [],
            "order_ok": True,
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _fmt_ci(ci, digits=6):
    if ci is None:
        return "insufficient_pairs"
    return f"[{_fmt(ci[0], digits)}, {_fmt(ci[1], digits)}]"


def render_markdown(result):
    scope = result["scope"]
    lines = [
        "# MoE read-ledger A/B comparison",
        "",
        f"- schema: `{result['schema']}`",
        f"- generated: {result['generated_at']}",
        f"- scope: **{scope}** — {result['scope_note']}",
        f"- A: `{result['inputs']['a']['path']}` tags={result['inputs']['a']['tags']} "
        f"replicates={result['inputs']['a']['replicates']}",
        f"- B: `{result['inputs']['b']['path']}` tags={result['inputs']['b']['tags']} "
        f"replicates={result['inputs']['b']['replicates']}",
        f"- pairing: {result['pairing']['unit']} × {len(result['pairing']['replicates'])} "
        f"replicate(s); dropped first decode="
        f"{result['pairing']['dropped_first_decode']}",
        f"- bootstrap: {result['stats']['reps']} reps, seed {result['stats']['seed']}, "
        "percentile 95% CI on paired log-ratio",
        "",
        "Positive log-ratio means A > B (more reads); negative means A < B. "
        "`delta_pct = (exp(median log-ratio) - 1) * 100`.",
        "",
        "## Paired metrics",
        "",
        "| metric | A median | B median | ratio (median log) | delta % | 95% CI log | "
        "95% CI ratio | sign +/−/0 | p(A<B) | p(A>B) | pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in result["metrics"].items():
        sign = m.get("sign_test") or {}
        ci = _fmt_ci(m.get("ci95_log_ratio"))
        if m["insufficient_pairs"]:
            ci = "insufficient_pairs"
        lines.append(
            f"| {key} (`{m['field']}`) | {_fmt(m['a_median'])} | {_fmt(m['b_median'])} | "
            f"{_fmt(m['ratio_median'])} | {_fmt(m['delta_pct'])} | {ci} | "
            f"{_fmt_ci(m.get('ci95_ratio')) if not m['insufficient_pairs'] else 'insufficient_pairs'} | "
            f"{sign.get('n_positive', 'n/a')}/{sign.get('n_negative', 'n/a')}/"
            f"{sign.get('n_zero', 'n/a')} | {_fmt(sign.get('p_one_sided_less'))} | "
            f"{_fmt(sign.get('p_one_sided_greater'))} | {m['n_pairs']} |")

    lines += ["", "## Verify / prefill decomposition", ""]
    for side_key in ("a", "b"):
        d = result["decomposition"][side_key]
        side = result["inputs"][side_key]
        lines += [
            f"### {side_key.upper()} (`{side['tags'][0]}`)",
            "",
            "| class | windows | indices | span sum | calls/step | pfread/step | "
            "runs/step | miss/step | runs sum (median across reps) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for kind in ("verify", "prefill"):
            v = d[kind]
            med = v["per_step_median"]
            lines.append(
                f"| {kind} | {v['windows_total']} (used {v.get('windows_used', v['windows_total'])}) | "
                f"{v['indices']} | {v['span_sum']} | {_fmt(med.get('calls'))} | "
                f"{_fmt(med.get('pfread'))} | {_fmt(med.get('runs'))} | "
                f"{_fmt(med.get('miss'))} | {_fmt(v['runs_sum_per_replicate_median'])} |")
        pt = d["process_totals"]
        lines += [
            "",
            f"process totals: merge-scan.runs={_fmt(pt['merge_scan_runs_median'])}, "
            f"prefill.runs={_fmt(pt['prefill_runs_median'])}, "
            f"decode runs total={_fmt(pt['decode_runs_total_median'])}, "
            f"rounds={_fmt(pt['rounds_median'])}",
            "",
        ]

    lines += ["## Validation", ""]
    v = result["validation"]
    lines += [
        f"- A steps: `{v['a_steps']}`; B steps: `{v['b_steps']}`",
        f"- A prefill steps: `{v['a_prefill_steps']}`; B prefill steps: `{v['b_prefill_steps']}`",
        f"- missing in A: `{v['missing_steps_in_a']}`; missing in B: `{v['missing_steps_in_b']}`",
        "- duplicate/out-of-order windows abort with exit code 2 before any output is written.",
    ]
    if result["stats"]["insufficient_pairs"]:
        lines += ["", "**insufficient_pairs**: n < 4 for at least one metric; no CI computed."]
    if result["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in result["warnings"]]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _check(checks, name, expected, actual, tol, detail=""):
    if tol is None:
        ok = expected == actual
    else:
        ok = (actual is not None and expected is not None
              and abs(actual - expected) <= tol)
    delta = (None if not isinstance(actual, (int, float))
             or not isinstance(expected, (int, float)) else actual - expected)
    checks.append({"name": name, "ok": bool(ok), "expected": expected,
                   "actual": actual, "tol": tol, "delta": delta,
                   "detail": detail})
    return ok


def _load_ledger(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read T-0110 ledger {path}: {exc}") from None


def _run_error_case(checks, name, mutate):
    base_a = (DEFAULT_T0078_RUNS / "idA-r1-n64.err").read_text(errors="replace")
    base_b = (DEFAULT_T0078_RUNS / "ordB-r1-n64.err").read_text(errors="replace")
    with tempfile.TemporaryDirectory(prefix="compare-moe-err-") as td:
        td = Path(td)
        a_text = mutate(base_a)
        a_path = td / "case-a.err"
        b_path = td / "case-b.err"
        a_path.write_text(a_text)
        b_path.write_text(base_b)
        out = td / "out.json"
        # The expected failure message is useful in a manual run but noise in
        # the self-test log; capture it so only the PASS/FAIL line remains.
        with contextlib.redirect_stderr(io.StringIO()):
            rc = main(["--a", str(a_path), "--b", str(b_path),
                       "--tags-a", "case-a", "--tags-b", "case-b",
                       "--json", str(out), "--scope", "published",
                       "--reps", "100", "--seed", "1", "--quiet"])
        wrote = out.exists()
    checks.append({"name": name, "ok": rc == 2 and not wrote,
                   "expected": "exit=2, no output", "actual": f"exit={rc}, wrote={wrote}",
                   "tol": None, "delta": None,
                   "detail": "malformed input must not produce a valid-looking result"})


def _remove_window(text, idx):
    lines = text.splitlines(keepends=True)
    out = []
    removed = False
    for line in lines:
        if re.search(rf"\[moe-ssd\]\s+(?:prefill|decode)\s+#{idx}\b", line):
            removed = True
            continue
        out.append(line)
    if not removed:
        raise AssertionError(f"test setup: window #{idx} not found")
    return "".join(out)


def _duplicate_window(text, idx):
    lines = text.splitlines(keepends=True)
    out = []
    copied = False
    for line in lines:
        out.append(line)
        if not copied and re.search(rf"\[moe-ssd\]\s+(?:prefill|decode)\s+#{idx}\b", line):
            out.append(line)
            copied = True
    if not copied:
        raise AssertionError(f"test setup: window #{idx} not found")
    return "".join(out)


def _swap_windows(text, idx1, idx2):
    lines = text.splitlines(keepends=True)
    positions = {}
    for i, line in enumerate(lines):
        for idx in (idx1, idx2):
            if re.search(rf"\[moe-ssd\]\s+(?:prefill|decode)\s+#{idx}\b", line):
                positions[idx] = i
    if len(positions) != 2:
        raise AssertionError("test setup: swap windows not found")
    i1, i2 = positions[idx1], positions[idx2]
    lines[i1], lines[i2] = lines[i2], lines[i1]
    return "".join(lines)


def run_self_test(args):
    start = time.time()
    runs_path = (Path(args.t0078_runs).expanduser()
                 if getattr(args, "t0078_runs", None) else DEFAULT_T0078_RUNS)
    ledger_path = (Path(args.t0110_ledger).expanduser()
                   if getattr(args, "t0110_ledger", None) else DEFAULT_T0110_LEDGER)
    if not runs_path.is_dir():
        raise SystemExit(f"self-test: T-0078 runs dir not found: {runs_path}")

    checks = []
    a = load_side(runs_path, ["idA"], "A")
    c = load_side(runs_path, ["ordA"], "C")
    b = load_side(runs_path, ["ordB"], "B")

    # 1) published A/C/B medians exactly as published by T-0096/T-0110.
    for tag, side in (("idA", a), ("ordA", c), ("ordB", b)):
        for key, field, _ in PUBLISHED_METRICS:
            if key not in EXPECTED_PUBLISHED[tag]:
                continue
            _check(checks, f"{tag} published.{field}",
                   EXPECTED_PUBLISHED[tag][key],
                   _arm_median(side, "published", field), SELFTEST_TOL,
                   "published median; tol ±0.05 display rounding")

    # 2) total A/C/B medians vs T-0110 ledger (canonical total scope).
    ledger = _load_ledger(ledger_path)
    for tag, side in (("idA", a), ("ordA", c), ("ordB", b)):
        label = ARM_LABELS[tag]
        arm = ledger.get("arms", {}).get(label)
        _check(checks, f"T-0110 ledger arm {label} present", True, arm is not None,
               None, "ledger must contain the canonical arm")
        if arm is None:
            continue
        for key, field in TOTAL_METRICS:
            if key not in EXPECTED_TOTAL[tag]:
                continue
            expected_rounded = EXPECTED_TOTAL[tag][key]
            actual = _arm_median(side, "total", field)
            _check(checks, f"{tag} total.{field} vs published value",
                   expected_rounded, actual, SELFTEST_TOL,
                   "±0.05 display rounding")
            _check(checks, f"{tag} total.{field} vs T-0110 ledger",
                   arm["total"].get(field), actual, 1e-9,
                   "must equal the T-0110 ledger float")

    # 3) paired comparisons must move in the expected direction.
    cmp_ac = compare_sides(a, c, "published", reps=args.reps, seed=args.seed)
    cmp_ab = compare_sides(a, b, "published", reps=args.reps, seed=args.seed)
    _check(checks, "A/C runs ratio > 1", True,
           cmp_ac["metrics"]["runs"]["ratio_median"] > 1.0, None,
           "A=idA 96.5 runs/step, C=ordA 86.5: A/C ratio must exceed 1")
    _check(checks, "A/C calls ratio", 1.0,
           cmp_ac["metrics"]["calls"]["ratio_median"], 1e-9,
           "order does not change calls")
    _check(checks, "A/C pfread ratio", 1.0,
           cmp_ac["metrics"]["pfread"]["ratio_median"], 1e-9,
           "order does not change pfread")
    _check(checks, "A/B calls ratio", 373.0 / 355.5,
           cmp_ab["metrics"]["calls"]["ratio_median"], 0.01,
           "A=idA 373 calls/step, B=ordB 355.5: ratio of medians ~1.049")
    _check(checks, "A/B runs ratio > 1", True,
           cmp_ab["metrics"]["runs"]["ratio_median"] > 1.0, None,
           "ordB is ordered, so it has fewer runs than identity idA")
    _check(checks, "published n_pairs (A/C)", 6,
           cmp_ac["metrics"]["runs"]["n_pairs"], None,
           "T-0078 has 7 decode windows, first is dropped")

    # 4) self-comparison: delta=0, CI contains 0, sign p=1.
    for scope in ("published", "total"):
        cmp_aa = compare_sides(a, a, scope, reps=args.reps, seed=args.seed)
        for key, metric in cmp_aa["metrics"].items():
            _check(checks, f"self {scope}/{key} delta_pct", 0.0,
                   metric["delta_pct"], 1e-12, "same file vs itself")
            ci = metric["ci95_log_ratio"]
            _check(checks, f"self {scope}/{key} CI contains 0", True,
                   ci is not None and ci[0] <= 0.0 <= ci[1], None,
                   "identical pairs must give a degenerate [0,0] interval")
            stest = metric["sign_test"]
            _check(checks, f"self {scope}/{key} sign p(A<B)", 1.0,
                   stest["p_one_sided_less"], None,
                   "all ties -> p=1 for both directions")
            _check(checks, f"self {scope}/{key} sign p(A>B)", 1.0,
                   stest["p_one_sided_greater"], None,
                   "all ties -> p=1 for both directions")
        _check(checks, f"self {scope} every delta_pct zero", True,
               all(m["delta_pct"] == 0.0 for m in cmp_aa["metrics"].values()),
               None, "")

    # 5) total-scope comparison still gives a valid CI (4 runs = 4 pairs).
    cmp_total_ac = compare_sides(a, c, "total", reps=args.reps, seed=args.seed)
    _check(checks, "total A/C n_pairs", 4,
           cmp_total_ac["metrics"]["runs"]["n_pairs"], None,
           "one pair per replicate")
    _check(checks, "total A/C runs ratio > 1", True,
           cmp_total_ac["metrics"]["runs"]["ratio_median"] > 1.0, None,
           "ordered reduces total runs/round (A=124.96, C=110.71)")
    _check(checks, "total A/C CI computed", True,
           cmp_total_ac["metrics"]["runs"]["ci95_log_ratio"] is not None, None,
           "n=4 is the minimum for a CI")

    # 6) decomposition must keep verify and true prefill apart.
    deco = cmp_ac["decomposition"]["a"]
    _check(checks, "A verify indices (all)", [3, 4, 5, 6, 7, 8, 16],
           deco["verify"]["indices_all"], None, "tok==4 windows")
    _check(checks, "A verify indices (used, first dropped)", [4, 5, 6, 7, 8, 16],
           deco["verify"]["indices"], None, "cold #3 dropped from the paired test")
    _check(checks, "A prefill indices", [1, 2],
           deco["prefill"]["indices"], None, "tok=2/18 windows")
    _check(checks, "A prefill excludes verify", True,
           not (set(deco["verify"]["indices_all"]) & set(deco["prefill"]["indices_all"])),
           None, "no tok==4 window may be counted as prefill")
    _check(checks, "A prefill runs sum", 2177.0,
           deco["prefill"]["runs_sum_per_replicate_median"], SELFTEST_TOL,
           "merge-scan.runs - decode runs = 2177")
    _check(checks, "C prefill runs sum", 1773.0,
           cmp_ac["decomposition"]["b"]["prefill"]["runs_sum_per_replicate_median"],
           SELFTEST_TOL, "ordered prefill has fewer runs")

    # 7) bootstrap determinism with a fixed seed.
    again = compare_sides(a, c, "published", reps=args.reps, seed=args.seed)
    _check(checks, "fixed-seed bootstrap deterministic", True,
           all(again["metrics"][k]["ci95_log_ratio"]
               == cmp_ac["metrics"][k]["ci95_log_ratio"]
               for k in cmp_ac["metrics"]), None,
           "same seed + same input => identical CI")

    # 8) stat helpers are unit-tested directly (fixed-seed bootstrap above).
    _check(checks, "percentile median", 1.5,
           percentile([0.0, 1.0, 2.0, 3.0], 0.5), 1e-12, "linear interpolation")
    _check(checks, "sign test all positive p(A>B)", 1.0 / 64.0,
           sign_test([1.0] * 6)["p_one_sided_greater"], 1e-12,
           "exact binomial, one-sided")
    _check(checks, "sign test ties p=1", 1.0,
           sign_test([0.0] * 3)["p_one_sided_less"], None,
           "ties are excluded; all-tie is degenerate p=1")
    _check(checks, "log ratio both zero", (0.0, False),
           log_ratio(0, 0), None, "0/0 means no difference")
    _check(checks, "log ratio zero correction", True,
           log_ratio(0, 5)[1], None, "Haldane-Anscombe +0.5 is flagged")

    # 9) n<4 must be reported as insufficient, without inventing a CI.
    id2a = load_side(runs_path, ["id2A"], "A2")
    cmp_ins = compare_sides(id2a, id2a, "total", reps=100, seed=args.seed)
    _check(checks, "total n<4 status", "insufficient_pairs",
           cmp_ins["status"], None, "id2A has only 2 replicates")
    _check(checks, "total n<4 no CI", True,
           all(m["ci95_log_ratio"] is None for m in cmp_ins["metrics"].values()),
           None, "n<4 must not compute a CI")

    # 10) malformed inputs: exit code 2 and no output file.
    _run_error_case(checks, "missing step -> exit 2, no output",
                    lambda text: _remove_window(text, 5))
    _run_error_case(checks, "duplicate step -> exit 2, no output",
                    lambda text: _duplicate_window(text, 4))
    _run_error_case(checks, "out-of-order step -> exit 2, no output",
                    lambda text: _swap_windows(text, 4, 5))

    passed = sum(1 for c in checks if c["ok"])
    failed = len(checks) - passed
    elapsed = time.time() - start

    selftest_dir = (Path(args.selftest_dir).expanduser()
                    if getattr(args, "selftest_dir", None)
                    else (QUEUE_RUN_DIR / "artifacts" if QUEUE_RUN_DIR
                          else Path("artifacts")))
    selftest_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "ok": failed == 0,
        "schema": "moe-read-compare-selftest-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "source": str(runs_path),
        "ledger": str(ledger_path),
        "elapsed_s": round(elapsed, 3),
        "checks_passed": passed,
        "checks_failed": failed,
        "checks": checks,
        "published": {
            tag: {key: _arm_median(side, "published", field)
                  for key, field, _ in PUBLISHED_METRICS
                  if key in EXPECTED_PUBLISHED[tag]}
            for tag, side in (("idA", a), ("ordA", c), ("ordB", b))
        },
        "total": {
            tag: {key: _arm_median(side, "total", field)
                  for key, field in TOTAL_METRICS
                  if key in EXPECTED_TOTAL[tag]}
            for tag, side in (("idA", a), ("ordA", c), ("ordB", b))
        },
        "comparison_a_c": {
            "paired_n": cmp_ac["metrics"]["runs"]["n_pairs"],
            "metrics": {k: {kk: m[kk] for kk in
                            ("a_median", "b_median", "ratio_median",
                             "delta_pct", "ci95_log_ratio", "ci95_ratio",
                             "sign_test", "n_pairs", "insufficient_pairs")}
                        for k, m in cmp_ac["metrics"].items()},
        },
        "comparison_a_b": {
            "paired_n": cmp_ab["metrics"]["runs"]["n_pairs"],
            "metrics": {k: {kk: m[kk] for kk in
                            ("a_median", "b_median", "ratio_median",
                             "delta_pct", "ci95_log_ratio", "ci95_ratio",
                             "sign_test", "n_pairs", "insufficient_pairs")}
                        for k, m in cmp_ab["metrics"].items()},
        },
        "total_comparison_a_c": {
            "paired_n": cmp_total_ac["metrics"]["runs"]["n_pairs"],
            "metrics": {k: {kk: m[kk] for kk in
                            ("a_median", "b_median", "ratio_median",
                             "delta_pct", "ci95_log_ratio", "ci95_ratio",
                             "sign_test", "n_pairs", "insufficient_pairs")}
                        for k, m in cmp_total_ac["metrics"].items()},
        },
        "notes": [
            "Imports analyze_moe_reads.py; its published/total semantics are not modified.",
            "Self-comparison must yield delta=0, a [0,0] log-ratio CI and p=1.",
            "Missing/duplicate/out-of-order windows exit 2 before writing output.",
            "Tolerance 0.05 is display rounding, not an approximate-equality escape hatch.",
        ],
    }

    (selftest_dir / "selftest.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = [
        f"# compare_moe_runs.py self-test ({'PASS' if result['ok'] else 'FAIL'})",
        f"source: {runs_path}",
        f"ledger: {ledger_path}",
        f"elapsed: {elapsed:.3f}s   checks: {passed} passed, {failed} failed",
        "",
    ]
    for c in checks:
        status = "PASS" if c["ok"] else "FAIL"
        if c["tol"] is None:
            lines.append(f"[{status}] {c['name']}: expected={c['expected']} "
                         f"actual={c['actual']}")
        else:
            lines.append(f"[{status}] {c['name']}: expected={c['expected']} "
                         f"actual={c['actual']} tol={c['tol']} delta={c['delta']}")
    lines += [
        "",
        "## published A/C/B (T-0110 values)",
        "idA runs=96.5 calls=373.0 pfread=262.0",
        "ordA runs=86.5 calls=373.0 pfread=262.0",
        "ordB runs=86.5 calls=355.5 pfread=262.0",
        "",
        "## total A/C/B (T-0110 ledger floats)",
        f"idA runs={result['total']['idA']['runs']} "
        f"calls={result['total']['idA']['calls']} "
        f"pfread={result['total']['idA']['pfread']}",
        f"ordA runs={result['total']['ordA']['runs']} "
        f"calls={result['total']['ordA']['calls']} "
        f"pfread={result['total']['ordA']['pfread']}",
        f"ordB runs={result['total']['ordB']['runs']} "
        f"calls={result['total']['ordB']['calls']} "
        f"pfread={result['total']['ordB']['pfread']}",
    ]
    (selftest_dir / "selftest.txt").write_text("\n".join(lines) + "\n")

    print(f"self-test: {'PASS' if result['ok'] else 'FAIL'} "
          f"({passed} passed, {failed} failed, {elapsed:.3f}s)")
    print(f"  {selftest_dir / 'selftest.json'}")
    print(f"  {selftest_dir / 'selftest.txt'}")
    for c in checks:
        if not c["ok"]:
            print(f"FAIL {c['name']}: expected={c['expected']} actual={c['actual']} "
                  f"tol={c['tol']}")
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Pair two [moe-ssd] read-ledger runs and compare them "
                    "(offline, stdlib only).")
    ap.add_argument("--a", help="A: one .err file or a run/artifact directory")
    ap.add_argument("--b", help="B: one .err file or a run/artifact directory")
    ap.add_argument("--tags-a", action="append", default=[],
                    help="arm tag in A, e.g. idA (comma separated or repeatable)")
    ap.add_argument("--tags-b", action="append", default=[],
                    help="arm tag in B, e.g. ordB (comma separated or repeatable)")
    ap.add_argument("--json", dest="json_out", help="write comparison JSON here")
    ap.add_argument("--markdown", dest="markdown_out",
                    help="write human-readable Markdown here")
    ap.add_argument("--scope", choices=("published", "total"),
                    default="published",
                    help="published per-step (default) or total per-run")
    ap.add_argument("--reps", type=int, default=10000,
                    help="bootstrap repetitions (default 10000)")
    ap.add_argument("--seed", type=int, default=20261122,
                    help="bootstrap seed (default 20261122)")
    ap.add_argument("--self-test", action="store_true",
                    help="offline self-test against T-0078 and the T-0110 ledger")
    ap.add_argument("--selftest-dir", default=None,
                    help="where selftest.json/txt go (default $QUEUE_RUN_DIR/artifacts)")
    ap.add_argument("--t0078-runs", default=None,
                    help="override the T-0078 runs dir for --self-test")
    ap.add_argument("--t0110-ledger", default=None,
                    help="override the T-0110 ledger.json for --self-test")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        try:
            return run_self_test(args)
        except InputError as exc:
            print(f"self-test input error: {exc}", file=sys.stderr)
            return 2

    if not args.a or not args.b:
        ap.error("give --a and --b (or --self-test)")

    tags_a = _parse_tags(args.tags_a)
    tags_b = _parse_tags(args.tags_b)
    try:
        side_a = load_side(args.a, tags_a, "a")
        side_b = load_side(args.b, tags_b, "b")
        result = compare_sides(side_a, side_b, args.scope,
                               reps=args.reps, seed=args.seed)
    except InputError as exc:
        # Important: nothing has been written yet, so no misleading result exists.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
    if args.markdown_out:
        out = Path(args.markdown_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(result))

    if not args.quiet:
        print(f"scope={result['scope']} status={result['status']} "
              f"pairs={result['metrics']['runs']['n_pairs']} "
              f"seed={args.seed} reps={args.reps}")
        for key, m in result["metrics"].items():
            ci = m["ci95_log_ratio"]
            ci_s = "insufficient_pairs" if ci is None else f"[{ci[0]:.4g},{ci[1]:.4g}]"
            print(f"  {key:12s} A={_fmt(m['a_median'])} B={_fmt(m['b_median'])} "
                  f"ratio={_fmt(m['ratio_median'])} delta={_fmt(m['delta_pct'])}% "
                  f"CI(log)={ci_s}")
        if args.json_out:
            print(f"wrote {args.json_out}")
        if args.markdown_out:
            print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
