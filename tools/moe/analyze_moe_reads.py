#!/usr/bin/env python3
"""Canonical parser for the `[moe-ssd]` read ledger (offline, stdlib only).

Why this tool exists
====================
T-0078's run-local ``analyze.py`` divided *every* per-window field by ``span``.
That is wrong for two counters.  In ``src/llama-moe-slot.cpp``:

* ``calls=``  is the process-global ``io_calls_all`` delta since the previous
  flush (``readv_span``); ``pfread=`` is the process-global
  ``lookahead.n_pred_read`` delta.  Both cover the whole ``span`` and *may* be
  divided by it.
* ``runs=`` / ``miss=`` are ``cur.runs.n`` / ``cur.miss_sum`` for the **last
  ubatch only**.  ``cur`` is cleared on every ubatch (printed or not), so these
  are already per-ubatch values and **must not** be divided by ``span``.
* the verify graph of SPECN=3 has 4 rows and is logged as
  ``prefill #... tok=4``.  Real prefill has ``tok=2/18``.  A parser that
  hard-codes ``tok=4`` as "decode" is fine, but one that treats every
  ``prefill`` line as a true prefill corrupts the total-scope prefill
  subtraction.

This tool keeps the raw counters, exposes two citable scopes, and has an
offline self-test against the T-0078 published data.

Scopes
======
``published``
    Drop ``#3`` (first decode window, cold), treat ``tok==4`` as verify/decode.
    ``calls/pfread/bytes/pread`` are divided by ``span``; ``runs/miss`` are
    kept raw.  Arm values are the median of per-run medians.  Five of the six
    logged windows have ``span=1``, so the published runs/miss medians are
    unchanged from T-0078 -- but the ``#16`` row is no longer divided by 8.

``total``
    Per-run process-level totals: ``merge-scan.runs`` minus the ``runs`` of the
    true prefill windows (``tok!=4``), and the exact sum of the ``calls`` /
    ``pfread`` deltas of the logged decode windows (``#3-#8`` + the ``#16``
    8-step aggregate).  Divided by the number of logged decode rounds
    (``sum(span)``).  ``#9-#15`` are throttled by the logger, so the ``#16``
    aggregate is the only evidence for them.

``--legacy-span``
    Opt-in reproduction of T-0078 ``analyze.py``: also divide ``runs/miss``
    (and ``merge_syscalls``) by ``span``.  Every output is tagged
    ``LEGACY/DO-NOT-CITE``; the default is canonical.

Usage
=====
    python3 tools/moe/analyze_moe_reads.py \
      --runs /home/yydh/.local/state/dpsk-queue/runs/T-0078/run-20261122T092959-w1-1/artifacts/phone/runs \
      --out "$QUEUE_RUN_DIR/artifacts/ledger.json" --self-test

    # single file / several files
    python3 tools/moe/analyze_moe_reads.py --err idA-r1-n64.err --tag idA-r1 --out /tmp/x.json

The self-test is offline (no phone, no A100, no network) and reads the T-0078
run directory by absolute path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "moe-read-ledger-v1"
LEGACY_WARNING = "LEGACY/DO-NOT-CITE"

QUEUE_RUN_DIR = Path(os.environ.get("QUEUE_RUN_DIR", "")).expanduser() if os.environ.get("QUEUE_RUN_DIR") else None

DEFAULT_T0078_ROOT = Path(
    "/home/yydh/.local/state/dpsk-queue/runs/T-0078/run-20261122T092959-w1-1/artifacts/phone"
)
DEFAULT_RUNS = DEFAULT_T0078_ROOT / "runs"
DEFAULT_ANALYSIS = DEFAULT_T0078_ROOT / "analysis.json"

# Same labels as T-0096's calls-breakdown.json / T-0078's analysis.json.
ARMS = {
    "idA": "A identity+merge0",
    "idB": "B0 identity+merge2",
    "ordA": "C ordered+merge0",
    "ordB": "B ordered+merge2",
    "ordS": "D ordered+merge1",
    "id2A": "A2 identity+merge0 (after restore)",
}

# One decode window line.  Tolerant to leading/suffix build noise and to
# `decode` vs `prefill` marker (the tok= value decides what it really is).
WINDOW_RE = re.compile(r"\[moe-ssd\]\s+(?P<kind>prefill|decode)\s+#(?P<idx>\d+)\b")
# key=value tokens after the header.  Values like `229ms`, `6%`, `398MiB`
# are accepted; the numeric prefix is what we keep.
KV_RE = re.compile(r"(?<![\w/])([A-Za-z_][A-Za-z0-9_]*)=(-?(?:\d+(?:\.\d*)?|\.\d+))")
TAG_N_RE = re.compile(r"-n\d+$")
# Process-level cumulative ledger line: `[moe-ssd] total[final]: miss=..`.
# Optional `with_totals=True` on parse_file() exposes these; the default CLI
# output is unchanged so the canonical per-window accounting stays stable.
TOTAL_RE = re.compile(
    r"\[moe-ssd\]\s+total\[(?P<kind>\w+)\]:\s*miss=(?P<miss>\d+)"
    r"\s+bytes=(?P<mib>[0-9.]+)\s*MiB\s+pread=(?P<pread>\d+)ms")

REQUIRED_WINDOW = ("idx", "tok", "span", "calls", "pfread", "runs", "miss")
PUBLISHED_FIELDS = (
    "runs_per_step",
    "syscalls_per_step",
    "miss_per_step",
    "calls_per_step",
    "pfread_per_step",
    "avg_run",
    "scatter_pct",
)


def _number(text: str):
    """int when the literal has no dot, float otherwise."""
    return float(text) if ("." in text or "e" in text.lower()) else int(text)


def _div(value, span):
    return None if value is None else value / span


def _median(values):
    vals = [v for v in values if v is not None]
    return st.median(vals) if vals else None


def _max(values):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _sum(values):
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def tag_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".err"):
        name = name[:-4]
    return TAG_N_RE.sub("", name)


def arm_for_tag(tag: str) -> str:
    if tag.endswith("-v"):
        return "V verify-slot runs"
    prefix = tag.split("-")[0]
    return ARMS.get(prefix, prefix)


# ---------------------------------------------------------------------------
# line parsing
# ---------------------------------------------------------------------------

def parse_window(line: str):
    m = WINDOW_RE.search(line)
    if not m:
        return None
    kv = {}
    for key, value in KV_RE.findall(line[m.end():]):
        if key not in kv:
            kv[key] = _number(value)
    tok = kv.get("tok")
    if tok == 4:
        kind = "verify"          # SPECN=3 verify graph is logged as "prefill"
    elif tok is not None:
        kind = "prefill"         # tok=2/18 -> real prefill
    else:
        kind = m.group("kind")
    if "span" in kv:
        kv["span"] = int(max(1, kv["span"]))
    idx = int(m.group("idx"))
    kv["idx"] = idx
    return {
        "kind": kind,
        "idx": idx,
        "raw": kv,
    }


def parse_total(line: str):
    """Parse one `[moe-ssd] total[<kind>]: miss=.. bytes=.. MiB pread=..ms` line."""
    m = TOTAL_RE.search(line)
    if not m:
        return None
    return {
        "kind": m.group("kind"),
        "miss": int(m.group("miss")),
        "mib": float(m.group("mib")),
        "pread_ms": int(m.group("pread")),
    }


def parse_merge_scan(line: str):
    for marker, kind in (("merge-prefetch-scan:", "prefetch"), ("merge-scan:", "demand")):
        if marker in line:
            tail = line.split(marker, 1)[1]
            kv = {}
            for key, value in KV_RE.findall(tail):
                if key not in kv:
                    kv[key] = _number(value)
            m = re.search(r"scatter_experts=(\d+)\s*\(([\d.]+)%\)", line)
            if m:
                kv["scatter_experts"] = int(m.group(1))
                kv["scatter_pct"] = float(m.group(2))
            m = re.search(r"syscalls=(\d+)\s*\(([\d.]+)%", line)
            if m:
                kv["syscalls"] = int(m.group(1))
                kv["syscall_pct"] = float(m.group(2))
            m = re.search(r"merge=(\d+)", line)
            if m:
                kv["merge"] = int(m.group(1))
            kv["kind"] = kind
            return kv
    return None


def parse_merge_exec(line: str):
    for marker, kind in (("merge-prefetch-exec:", "prefetch"), ("merge-exec:", "demand")):
        if marker in line:
            tail = line.split(marker, 1)[1]
            kv = {}
            for key, value in KV_RE.findall(tail):
                if key not in kv:
                    kv[key] = _number(value)
            kv["kind"] = kind
            return kv
    return None


def parse_cache_summary(line: str):
    if "cache-summary:" not in line:
        return None
    tail = line.split("cache-summary:", 1)[1]
    kv = {}
    for key, value in KV_RE.findall(tail):
        if key not in kv:
            kv[key] = _number(value)
    return kv


def parse_timing(line: str):
    if "print_timing" not in line and "prompt tokens" not in line:
        return None
    out = {}
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
    if m:
        out["prompt_ms"] = float(m.group(1))
        out["prompt_tokens"] = int(m.group(2))
    m = re.search(r"(?<!prompt )\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
    if m:
        out["eval_ms"] = float(m.group(1))
        out["eval_tokens"] = int(m.group(2))
    m = re.search(r"([\d.]+)\s*tokens per second", line)
    if m:
        if "prompt eval time" in line:
            out["prompt_tps"] = float(m.group(1))
        elif "eval time" in line:
            out["eval_tps"] = float(m.group(1))
    m = re.search(r"prompt tokens\s*=\s*(\d+)", line)
    if m:
        out["prompt_tokens"] = int(m.group(1))
    return out or None


def parse_verify_slot(line: str):
    if "verify-slot:" not in line:
        return None
    m = re.search(r"verify-slot:\s*ok=(\d+)\s+bad=(\d+)\s+slot_collision=(\d+)", line)
    if not m:
        return None
    return {"ok": int(m.group(1)), "bad": int(m.group(2)), "collision": int(m.group(3))}


# ---------------------------------------------------------------------------
# per-window / per-run accounting
# ---------------------------------------------------------------------------

def build_record(tag: str, arm: str, path: Path, line_no: int, window):
    raw = window["raw"]
    span = int(raw.get("span") or 1)
    if span < 1:
        span = 1

    def per_step(divide_cur: bool):
        return {
            # global deltas: always per ubatch
            "calls": _div(raw.get("calls"), span),
            "pfread": _div(raw.get("pfread"), span),
            "bytes_mib": _div(raw.get("bytes"), span),
            "pread_sum_ms": _div(raw.get("pread_sum"), span),
            "pread_wall_ms": _div(raw.get("pread_wall"), span),
            "interval_ms": _div(raw.get("interval_ms"), span),
            "pmgmt_ms": _div(raw.get("pmgmt"), span),
            # cur counters: raw by default, /span only in legacy mode
            "runs": _div(raw.get("runs"), span) if divide_cur else raw.get("runs"),
            "miss": _div(raw.get("miss"), span) if divide_cur else raw.get("miss"),
            "syscalls": _div(raw.get("merge_syscalls"), span) if divide_cur else raw.get("merge_syscalls"),
            # shape metrics are ratios, never divided
            "avg_run": raw.get("avg_run"),
            "scatter_pct": raw.get("scatter"),
        }

    return {
        "tag": tag,
        "arm": arm,
        "file": str(path),
        "line": line_no,
        "kind": window["kind"],
        "idx": window["idx"],
        "tok": raw.get("tok"),
        "span": span,
        "raw": dict(raw),
        "per_step": per_step(False),
        "legacy_per_step": per_step(True),
    }


def summarize_run(parsed: dict, legacy: bool = False):
    """Per-file summary.  `published` uses the selected scope, `total` is exact."""
    recs = parsed["records"]
    decode = [r for r in recs if r["kind"] == "verify"]      # tok==4 decode graphs
    prefill = [r for r in recs if r["kind"] == "prefill"]    # tok==2/18
    rounds = sum(r["span"] for r in decode)
    used = decode[1:] if len(decode) > 1 else decode        # drop #3 (cold)
    key = "legacy_per_step" if legacy else "per_step"

    def med(field, subset=None):
        subset = used if subset is None else subset
        return _median([r[key].get(field) for r in subset])

    published = {
        "runs_per_step": med("runs"),
        "syscalls_per_step": med("syscalls"),
        "miss_per_step": med("miss"),
        "calls_per_step": med("calls"),
        "pfread_per_step": med("pfread"),
        "avg_run": med("avg_run"),
        "scatter_pct": med("scatter_pct"),
        "max_run": _max([r["raw"].get("max_run") for r in used]),
        "bytes_per_step": med("bytes_mib"),
        "pread_sum_per_step": med("pread_sum_ms"),
        "windows": len(decode),
        "windows_used": len(used),
        "dropped_first": len(decode) > 1,
    }

    calls_total = _sum([r["raw"].get("calls") for r in decode])
    pfread_total = _sum([r["raw"].get("pfread") for r in decode])
    bytes_total = _sum([r["raw"].get("bytes") for r in decode])
    prefill_runs = _sum([r["raw"].get("runs") for r in prefill])
    merge_scan = parsed.get("merge_scan")
    runs_total = None
    if merge_scan is not None and merge_scan.get("runs") is not None and prefill_runs is not None:
        runs_total = merge_scan["runs"] - prefill_runs

    total = {
        "rounds": rounds,
        "decode_windows": len(decode),
        "prefill_windows": len(prefill),
        "prefill_runs": prefill_runs,
        "calls_total": calls_total,
        "pfread_total": pfread_total,
        "bytes_total": bytes_total,
        "runs_total": runs_total,
        "calls_per_round": _div(calls_total, rounds),
        "pfread_per_round": _div(pfread_total, rounds),
        "bytes_per_round": _div(bytes_total, rounds),
        "runs_per_round": _div(runs_total, rounds),
        "demand_calls_per_round": (
            _div((calls_total - pfread_total), rounds)
            if calls_total is not None and pfread_total is not None else None
        ),
        "prefetch_fraction": (
            pfread_total / calls_total
            if pfread_total is not None and calls_total else None
        ),
    }

    return {
        "tag": parsed["tag"],
        "arm": parsed["arm"],
        "file": parsed["file"],
        "published": published,
        "total": total,
        "prefetch_fraction": {
            "published": (published["pfread_per_step"] / published["calls_per_step"]
                          if published["pfread_per_step"] and published["calls_per_step"] else None),
            "total": total["prefetch_fraction"],
        },
        "merge_scan": merge_scan,
        "merge_prefetch_scan": parsed.get("merge_prefetch_scan"),
        "merge_exec": parsed.get("merge_exec"),
        "merge_prefetch_exec": parsed.get("merge_prefetch_exec"),
        "cache_summary": parsed.get("cache_summary"),
        "timing": parsed.get("timing"),
        "verify_slot": parsed.get("verify_slot"),
        "missing_fields": parsed["missing_fields"],
    }


# ---------------------------------------------------------------------------
# file / directory input
# ---------------------------------------------------------------------------

def parse_file(path: Path, tag: str | None = None, with_totals: bool = False):
    """Parse one log file.

    ``with_totals=True`` additionally exposes the process-level
    ``total[<kind>]`` lines and the ``processing task`` line number.  This is
    additive: the default output (and therefore the CLI/JSON schema) is
    unchanged.  ``ledger_adapter.py`` uses it for the stage split.
    """
    tag = tag or tag_from_path(path)
    arm = arm_for_tag(tag)
    out = {
        "tag": tag,
        "arm": arm,
        "file": str(path),
        "records": [],
        "merge_scan": None,
        "merge_prefetch_scan": None,
        "merge_exec": None,
        "merge_prefetch_exec": None,
        "cache_summary": None,
        "timing": {},
        "verify_slot": None,
        "missing_fields": [],
    }
    if with_totals:
        out["task_line"] = None
        out["totals"] = []
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        out["missing_fields"].append(f"{path.name}:unreadable:{exc}")
        return out

    for line_no, line in enumerate(text.splitlines(), 1):
        if with_totals and out["task_line"] is None and "processing task" in line:
            out["task_line"] = line_no
        if with_totals:
            total = parse_total(line)
            if total is not None:
                out["totals"].append(total)
                continue
        window = parse_window(line)
        if window:
            missing = [f for f in REQUIRED_WINDOW if window["raw"].get(f) is None]
            if missing:
                out["missing_fields"].append(
                    f"{path.name}:window#{window['idx']}:{','.join(missing)}"
                )
            out["records"].append(build_record(tag, arm, path, line_no, window))
            continue
        merge = parse_merge_scan(line)
        if merge is not None:
            if merge["kind"] == "prefetch":
                out["merge_prefetch_scan"] = merge
            else:
                out["merge_scan"] = merge
            continue
        merge = parse_merge_exec(line)
        if merge is not None:
            if merge["kind"] == "prefetch":
                out["merge_prefetch_exec"] = merge
            else:
                out["merge_exec"] = merge
            continue
        cache = parse_cache_summary(line)
        if cache is not None:
            out["cache_summary"] = cache
            continue
        timing = parse_timing(line)
        if timing is not None:
            out["timing"].update({f"slot_{k}": v for k, v in timing.items()})
            continue
        verify = parse_verify_slot(line)
        if verify is not None:
            out["verify_slot"] = verify

    decode = [r for r in out["records"] if r["kind"] == "verify"]
    prefill = [r for r in out["records"] if r["kind"] == "prefill"]
    if not decode:
        out["missing_fields"].append(f"{path.name}:decode_windows")
    if not prefill:
        out["missing_fields"].append(f"{path.name}:prefill_windows")
    if out["merge_scan"] is None:
        out["missing_fields"].append(f"{path.name}:merge_scan")
    elif out["merge_scan"].get("runs") is None:
        out["missing_fields"].append(f"{path.name}:merge_scan.runs")
    return out


def parse_ledger(path, tag: str | None = None):
    """Like ``parse_file`` plus the cumulative ``total[*]`` rows and task line.

    Public, side-effect-free entry point for the thin adapters in
    ``exps/roofline-saturate-20261105/collect/``.  It reuses the canonical
    window/kind accounting above; it does not add a second parser.
    """
    return parse_file(Path(path).expanduser(), tag=tag, with_totals=True)


def read_results_tsv(path: Path):
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        tag = fields[0]
        try:
            rows[tag] = {
                "n": int(fields[1]),
                "wall_ms": int(fields[2]),
                "rc": int(fields[3]),
                "tps": float(fields[4]) if fields[4] not in ("NA", "") else None,
                "status": fields[-1],
            }
        except ValueError:
            continue
    return rows


def collect_inputs(paths, tags=None):
    """Return (err_files, results_rows, skipped).  Directories may contain runs/."""
    files = []
    rows = {}
    skipped = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            d = p / "runs" if (p / "runs").is_dir() else p
            dir_rows = read_results_tsv(d / "results.tsv")
            for tag, row in dir_rows.items():
                rows.setdefault(tag, row)
            for err in sorted(d.glob("*.err")):
                tag = tag_from_path(err)
                if tags and tag not in tags:
                    continue
                row = dir_rows.get(tag)
                if row and row.get("status") not in (None, "ok"):
                    skipped.append({"tag": tag, "file": str(err),
                                    "reason": f"results.tsv status={row.get('status')} rc={row.get('rc')}"})
                    continue
                files.append((err, tag))
        elif p.is_file():
            tag = tag_from_path(p)
            if tags and tag not in tags and len(paths) > 1:
                continue
            files.append((p, tag))
        else:
            raise SystemExit(f"input not found: {p}")
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for err, tag in files:
        key = str(err.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append((err, tag))
    # A single explicit file may be renamed; `--tag` then labels it instead of
    # acting as a filter.
    if tags and len(unique) == 1 and all(Path(p).is_file() for p in paths):
        unique = [(unique[0][0], next(iter(tags)))]
    return unique, rows, skipped


def analyze(paths, tags=None, legacy=False):
    files, rows, skipped = collect_inputs(paths, tags=tags)
    parsed_runs = []
    records = []
    missing_fields = []
    merge_scan = {}
    merge_exec = {}
    merge_prefetch_scan = {}
    merge_prefetch_exec = {}
    cache_summary = {}
    timing = {}
    for err, tag in files:
        parsed = parse_file(err, tag=tag)
        summary = summarize_run(parsed, legacy=legacy)
        if tag in rows:
            summary["tps"] = rows[tag].get("tps")
            summary["n"] = rows[tag].get("n")
            summary["status"] = rows[tag].get("status")
            summary["rc"] = rows[tag].get("rc")
        parsed_runs.append(summary)
        records.extend(parsed["records"])
        missing_fields.extend(parsed["missing_fields"])
        if parsed["merge_scan"] is not None:
            merge_scan[tag] = parsed["merge_scan"]
        if parsed["merge_exec"] is not None:
            merge_exec[tag] = parsed["merge_exec"]
        if parsed["merge_prefetch_scan"] is not None:
            merge_prefetch_scan[tag] = parsed["merge_prefetch_scan"]
        if parsed["merge_prefetch_exec"] is not None:
            merge_prefetch_exec[tag] = parsed["merge_prefetch_exec"]
        if parsed["cache_summary"] is not None:
            cache_summary[tag] = parsed["cache_summary"]
        if parsed["timing"]:
            timing[tag] = parsed["timing"]

    arms = {}
    by_arm = {}
    for run in parsed_runs:
        by_arm.setdefault(run["arm"], []).append(run)
    for label, runs in by_arm.items():
        def med_pub(key):
            return _median([r["published"].get(key) for r in runs])

        def med_total(key):
            return _median([r["total"].get(key) for r in runs])

        pref_pub = [r["prefetch_fraction"]["published"] for r in runs
                    if r["prefetch_fraction"]["published"] is not None]
        pref_tot = [r["prefetch_fraction"]["total"] for r in runs
                    if r["prefetch_fraction"]["total"] is not None]
        arms[label] = {
            "tags": [r["tag"] for r in runs],
            "n": len(runs),
            "published": {k: med_pub(k) for k in
                          ("runs_per_step", "syscalls_per_step", "miss_per_step",
                           "calls_per_step", "pfread_per_step", "avg_run",
                           "scatter_pct", "max_run", "bytes_per_step",
                           "pread_sum_per_step")},
            "total": {k: med_total(k) for k in
                      ("runs_per_round", "calls_per_round", "pfread_per_round",
                       "bytes_per_round", "demand_calls_per_round", "rounds",
                       "prefill_runs")},
            "prefetch_fraction": {
                "published": _median(pref_pub),
                "total": _median(pref_tot),
            },
            "merge_scan": runs[0]["merge_scan"],
            "merge_exec": runs[0]["merge_exec"],
            "merge_prefetch_scan": runs[0]["merge_prefetch_scan"],
            "merge_prefetch_exec": runs[0]["merge_prefetch_exec"],
            "verify_slot": runs[0]["verify_slot"],
            "per_run": runs,
        }

    pref_pub_all = [r["prefetch_fraction"]["published"] for r in parsed_runs
                    if r["prefetch_fraction"]["published"] is not None]
    pref_tot_all = [r["prefetch_fraction"]["total"] for r in parsed_runs
                    if r["prefetch_fraction"]["total"] is not None]

    def _spread(values):
        return {
            "median": _median(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n": len(values),
        }

    return {
        "schema": SCHEMA,
        "tool": "tools/moe/analyze_moe_reads.py",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": [str(Path(p).expanduser()) for p in paths],
        "legacy_span": bool(legacy),
        "citation": LEGACY_WARNING if legacy else "OK-TO-CITE",
        "warning": LEGACY_WARNING if legacy else None,
        "semantics": {
            "published": "drop #3; tok==4 is verify/decode; calls/pfread/bytes/pread /span; runs/miss raw; median of per-run medians",
            "total": "per-run merge-scan.runs - true-prefill runs; sum of #3-#8 + #16 calls/pfread deltas / sum(span)",
            "legacy_span": "divides runs/miss/syscalls by span like T-0078 analyze.py; DO-NOT-CITE",
        },
        "arms": arms,
        "prefetch_fraction": {
            "published": _spread(pref_pub_all),
            "total": _spread(pref_tot_all),
            "by_arm": {label: arm["prefetch_fraction"] for label, arm in arms.items()},
            "note": "published = median(pfread/step)/median(calls/step); total = sum(pfread)/sum(calls) per round",
        },
        "files": {r["tag"]: r for r in parsed_runs},
        "records": records,
        "merge_scan": merge_scan,
        "merge_exec": merge_exec,
        "merge_prefetch_scan": merge_prefetch_scan,
        "merge_prefetch_exec": merge_prefetch_exec,
        "cache_summary": cache_summary,
        "timing": timing,
        "missing_fields": missing_fields,
        "skipped": skipped,
    }


def write_tsv(path: Path, records):
    cols = ["tag", "arm", "file", "line", "kind", "idx", "tok", "span",
            "calls", "calls_per_step", "pfread", "pfread_per_step",
            "runs", "runs_per_step", "miss", "miss_per_step",
            "bytes", "bytes_per_step", "pread_sum", "scatter_pct",
            "merge_syscalls", "syscalls_per_step", "avg_run", "max_run"]

    def value(r, col):
        raw = r["raw"]
        mapping = {
            "calls": raw.get("calls"), "pfread": raw.get("pfread"),
            "runs": raw.get("runs"), "miss": raw.get("miss"),
            "bytes": raw.get("bytes"), "pread_sum": raw.get("pread_sum"),
            "scatter_pct": raw.get("scatter"),
            "merge_syscalls": raw.get("merge_syscalls"),
            "avg_run": raw.get("avg_run"), "max_run": raw.get("max_run"),
            "calls_per_step": r["per_step"]["calls"],
            "pfread_per_step": r["per_step"]["pfread"],
            "runs_per_step": r["per_step"]["runs"],
            "miss_per_step": r["per_step"]["miss"],
            "bytes_per_step": r["per_step"]["bytes_mib"],
            "syscalls_per_step": r["per_step"]["syscalls"],
        }
        if col in mapping:
            return mapping[col]
        return r.get(col)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in records:
            f.write("\t".join("" if value(r, c) is None else str(value(r, c))
                             for c in cols) + "\n")


# ---------------------------------------------------------------------------
# offline self-test
# ---------------------------------------------------------------------------

# T-0096's citable numbers.  median tolerance 0.51; ratios 0.05.
EXPECTED = {
    "A identity+merge0": {
        "published": {"runs_per_step": 96.5, "calls_per_step": 373.0, "pfread_per_step": 262.0},
        "total": {"runs_per_round": 124.96, "calls_per_round": 299.04, "pfread_per_round": 188.86},
    },
    "C ordered+merge0": {
        "published": {"runs_per_step": 86.5, "calls_per_step": 373.0, "pfread_per_step": 262.0},
        "total": {"runs_per_round": 110.71, "calls_per_round": 298.93, "pfread_per_round": 188.79},
    },
    "B ordered+merge2": {
        "published": {"runs_per_step": 86.5, "calls_per_step": 355.5, "pfread_per_step": 262.0},
        "total": {"runs_per_round": 110.71, "calls_per_round": 283.04, "pfread_per_round": 188.86},
    },
}


def _check(checks, name, expected, actual, tol, detail=""):
    ok = actual is not None and abs(actual - expected) <= tol
    checks.append({
        "name": name,
        "ok": bool(ok),
        "expected": expected,
        "actual": actual,
        "tol": tol,
        "delta": None if actual is None else actual - expected,
        "detail": detail,
    })
    return ok


def run_self_test(args, out_path, selftest_dir):
    start = time.time()
    runs_path = Path(args.runs[0]).expanduser() if args.runs else DEFAULT_RUNS
    analysis_json = Path(args.analysis_json).expanduser() if args.analysis_json else DEFAULT_ANALYSIS
    if not runs_path.exists():
        raise SystemExit(f"self-test: T-0078 runs dir not found: {runs_path}")
    if not analysis_json.is_file():
        raise SystemExit(f"self-test: analysis.json not found: {analysis_json}")

    old = json.loads(analysis_json.read_text())
    old_arms = old.get("arms", {})
    tags = sorted({t for label in old_arms.values() for t in label.get("tags", [])})
    if not tags:
        raise SystemExit("self-test: analysis.json has no arm tags")

    analyzed = analyze([runs_path], tags=set(tags), legacy=False)
    legacy_analyzed = analyze([runs_path], tags={"idA-r1"}, legacy=True)

    checks = []

    # 1) three citable arms: published + total
    for label, expected in EXPECTED.items():
        arm = analyzed["arms"].get(label)
        if arm is None:
            checks.append({"name": f"arm {label} present", "ok": False,
                           "expected": "present", "actual": None, "tol": None, "detail": ""})
            continue
        for key, want in expected["published"].items():
            _check(checks, f"{label} published.{key}", want,
                   arm["published"].get(key), 0.51, "median-of-medians")
        for key, want in expected["total"].items():
            _check(checks, f"{label} total.{key}", want,
                   arm["total"].get(key), 0.51, "merge-scan / logged-round median")

    # 2) #16 must not be divided by span=8
    rec16 = next((r for r in analyzed["records"]
                  if r["tag"] == "idA-r1" and r["idx"] == 16), None)
    if rec16 is None:
        checks.append({"name": "idA-r1 #16 present", "ok": False,
                       "expected": "present", "actual": None, "tol": None, "detail": ""})
    else:
        checks.append({"name": "idA-r1 #16 runs raw", "ok": rec16["raw"].get("runs") == 46,
                       "expected": 46, "actual": rec16["raw"].get("runs"),
                       "tol": 0, "delta": None, "detail": "cur.runs must not be /span"})
        checks.append({"name": "idA-r1 #16 miss raw", "ok": rec16["raw"].get("miss") == 48,
                       "expected": 48, "actual": rec16["raw"].get("miss"),
                       "tol": 0, "delta": None, "detail": "cur.miss_sum must not be /span"})
        _check(checks, "idA-r1 #16 calls_per_step", 195.0,
               rec16["per_step"].get("calls"), 0.05, "1560/8")
        _check(checks, "idA-r1 #16 pfread_per_step", 131.0,
               rec16["per_step"].get("pfread"), 0.05, "1048/8")
        checks.append({"name": "idA-r1 #16 runs_per_step == raw runs",
                       "ok": rec16["per_step"].get("runs") == 46,
                       "expected": 46, "actual": rec16["per_step"].get("runs"),
                       "tol": 0, "delta": None,
                       "detail": "default scope keeps runs raw"})

    # 3) 7 arms x 7 published fields vs the old analysis.json
    comparison = {}
    for label, old_arm in sorted(old_arms.items()):
        new_arm = analyzed["arms"].get(label)
        comparison[label] = {}
        for field in PUBLISHED_FIELDS:
            want = old_arm.get(field)
            got = new_arm["published"].get(field) if new_arm else None
            tol = 0.05 if field in ("avg_run",) else 0.51
            ok = want is not None and got is not None and abs(got - want) <= tol
            comparison[label][field] = {"analysis_json": want, "tool": got,
                                        "delta": None if (want is None or got is None) else got - want,
                                        "tol": tol, "ok": bool(ok)}
            checks.append({"name": f"{label} vs analysis.json {field}", "ok": bool(ok),
                           "expected": want, "actual": got, "tol": tol,
                           "delta": None if (want is None or got is None) else got - want,
                           "detail": "legacy median reconciliation"})

    # 4) --legacy-span is opt-in and tagged
    legacy_rec16 = next((r for r in legacy_analyzed["records"]
                         if r["tag"] == "idA-r1" and r["idx"] == 16), None)
    legacy_runs = legacy_rec16["legacy_per_step"].get("runs") if legacy_rec16 else None
    legacy_miss = legacy_rec16["legacy_per_step"].get("miss") if legacy_rec16 else None
    _check(checks, "legacy-span #16 runs/8", 5.75, legacy_runs, 0.001, "old analyze.py value")
    _check(checks, "legacy-span #16 miss/8", 6.0, legacy_miss, 0.001, "old analyze.py value")
    checks.append({"name": "legacy-span warning", "ok": legacy_analyzed["citation"] == LEGACY_WARNING,
                   "expected": LEGACY_WARNING, "actual": legacy_analyzed["citation"],
                   "tol": None, "delta": None, "detail": ""})
    checks.append({"name": "default is not legacy", "ok": analyzed["legacy_span"] is False,
                   "expected": False, "actual": analyzed["legacy_span"],
                   "tol": None, "delta": None, "detail": ""})

    passed = sum(1 for c in checks if c["ok"])
    failed = len(checks) - passed
    elapsed = time.time() - start

    legacy_deviations = []
    for r in analyzed["records"]:
        if r["span"] > 1:
            legacy_deviations.append({
                "tag": r["tag"], "idx": r["idx"], "span": r["span"],
                "runs_correct": r["per_step"].get("runs"),
                "runs_legacy_div_span": r["legacy_per_step"].get("runs"),
                "miss_correct": r["per_step"].get("miss"),
                "miss_legacy_div_span": r["legacy_per_step"].get("miss"),
                "note": "only runs/miss are distorted; calls/pfread are genuinely /span",
            })

    result = {
        "ok": failed == 0,
        "schema": "moe-read-selftest-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": str(runs_path),
        "analysis_json": str(analysis_json),
        "elapsed_s": round(elapsed, 3),
        "checks_passed": passed,
        "checks_failed": failed,
        "checks": checks,
        "arms": {label: {"published": analyzed["arms"][label]["published"],
                         "total": analyzed["arms"][label]["total"],
                         "prefetch_fraction": analyzed["arms"][label]["prefetch_fraction"]}
                 for label in analyzed["arms"]},
        "analysis_json_comparison": comparison,
        "legacy_span_deviations": legacy_deviations,
        "legacy_span_example": {
            "tag": "idA-r1", "idx": 16, "span": 8,
            "runs_correct": 46, "runs_legacy": legacy_runs,
            "miss_correct": 48, "miss_legacy": legacy_miss,
        },
        "missing_fields": analyzed["missing_fields"],
        "notes": [
            "published: 5 of 6 logged windows have span=1, so the median is unchanged; #16 is the row where /span was wrong.",
            "the old analysis.json values are reproduced for reconciliation, not cited as canonical.",
            "total runs uses process-wide merge-scan.runs minus true prefill (tok=2/18) runs.",
        ],
    }

    selftest_dir.mkdir(parents=True, exist_ok=True)
    (selftest_dir / "selftest.json").write_text(json.dumps(result, indent=1) + "\n")

    lines = [
        f"# analyze_moe_reads.py self-test ({'PASS' if result['ok'] else 'FAIL'})",
        f"source: {runs_path}",
        f"analysis.json: {analysis_json}",
        f"elapsed: {elapsed:.3f}s   checks: {passed} passed, {failed} failed",
        "",
    ]
    for c in checks:
        status = "PASS" if c["ok"] else "FAIL"
        if c["tol"] is None:
            lines.append(f"[{status}] {c['name']}: expected={c['expected']} actual={c['actual']}")
        else:
            lines.append(f"[{status}] {c['name']}: expected={c['expected']} actual={c['actual']} "
                         f"tol={c['tol']} delta={c['delta']}")
    if legacy_deviations:
        lines += ["", "## legacy /span distortion (span>1 windows, not citable)"]
        for d in legacy_deviations:
            lines.append(
                f"- {d['tag']} #{d['idx']} span={d['span']}: "
                f"runs {d['runs_correct']} (correct) vs {d['runs_legacy_div_span']} (/span); "
                f"miss {d['miss_correct']} vs {d['miss_legacy_div_span']}"
            )
    (selftest_dir / "selftest.txt").write_text("\n".join(lines) + "\n")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(analyzed, indent=1) + "\n")
    tsv_path = getattr(args, "tsv", None)
    if tsv_path:
        write_tsv(Path(tsv_path).expanduser(), analyzed["records"])

    print(f"self-test: {'PASS' if result['ok'] else 'FAIL'} "
          f"({passed} passed, {failed} failed, {elapsed:.3f}s)")
    print(f"  {selftest_dir / 'selftest.json'}")
    print(f"  {selftest_dir / 'selftest.txt'}")
    if out_path is not None:
        print(f"  {out_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Canonical offline parser for [moe-ssd] read ledger logs.")
    ap.add_argument("--runs", action="append", default=[],
                    help="runs/ directory, phone artifact dir, or any dir with *.err")
    ap.add_argument("--err", action="append", default=[],
                    help="one .err file (repeatable)")
    ap.add_argument("--tag", default=None,
                    help="only parse this tag (e.g. idA-r1); also labels a single --err")
    ap.add_argument("--out", default=None, help="write the full JSON ledger here")
    ap.add_argument("--tsv", default=None, help="write one row per logged window here")
    ap.add_argument("--legacy-span", action="store_true",
                    help="LEGACY/DO-NOT-CITE: divide runs/miss by span like T-0078 analyze.py")
    ap.add_argument("--allow-partial", action="store_true",
                    help="do not fail on missing fields (still reported)")
    ap.add_argument("--self-test", action="store_true",
                    help="offline reconciliation against T-0078 runs + analysis.json")
    ap.add_argument("--analysis-json", default=None,
                    help="override T-0078 analysis.json for --self-test")
    ap.add_argument("--selftest-dir", default=None,
                    help="where selftest.json/txt go (default $QUEUE_RUN_DIR/artifacts)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.legacy_span:
        msg = (f"{LEGACY_WARNING}: --legacy-span divides runs/miss by span; "
               "this reproduces T-0078 analyze.py and must not be cited.")
        print(msg, file=sys.stderr)
        print(msg)

    inputs = list(args.err) + list(args.runs)
    if not inputs and not args.self_test:
        ap.error("give --runs DIR, --err FILE, or --self-test")

    if args.self_test and not inputs:
        inputs = [str(DEFAULT_RUNS)]

    tags = {args.tag} if args.tag else None
    if args.self_test:
        if args.selftest_dir:
            selftest_dir = Path(args.selftest_dir).expanduser()
        elif QUEUE_RUN_DIR:
            selftest_dir = QUEUE_RUN_DIR / "artifacts"
        elif args.out:
            selftest_dir = Path(args.out).expanduser().parent
        else:
            selftest_dir = Path("artifacts")
        out_path = Path(args.out).expanduser() if args.out else (selftest_dir / "ledger.json")
        result = run_self_test(args, out_path, selftest_dir)
        return 0 if result["ok"] else 1

    analyzed = analyze(inputs, tags=tags, legacy=args.legacy_span)
    out_path = Path(args.out).expanduser() if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(analyzed, indent=1) + "\n")
    if args.tsv:
        write_tsv(Path(args.tsv).expanduser(), analyzed["records"])

    if not args.quiet:
        if out_path is not None:
            print(f"wrote {out_path} ({len(analyzed['records'])} windows, "
                  f"{len(analyzed['arms'])} arms)")
            for label, arm in analyzed["arms"].items():
                pub = arm["published"]
                tot = arm["total"]
                pf = arm["prefetch_fraction"]
                print(f"  {label:34s} published runs={pub['runs_per_step']} "
                      f"calls={pub['calls_per_step']} pfread={pub['pfread_per_step']} "
                      f"| total runs/round={tot['runs_per_round']} "
                      f"calls/round={tot['calls_per_round']} "
                      f"pfread/calls={pf['total']}")
        else:
            print(json.dumps(analyzed, indent=1))

    if analyzed["missing_fields"] and not args.allow_partial:
        for item in analyzed["missing_fields"]:
            print(f"missing: {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
