#!/usr/bin/env python3
"""Attribute llama.cpp/llama-cli bench logs to speculative-decoding sessions.

Pure stdlib (re/pathlib/csv/json/statistics/argparse).  The parser is deliberately
tolerant: unknown lines are ignored, malformed numeric fields become ``None`` and
are reported in each session's ``missing`` list.  It never guesses a value that
was not in the log.

Input lines are the ones printed by ``slot print_timing`` / ``statistics
draft-dspark`` / ``[ Prompt: .. | Generation: .. ]``.  A *session* is one
``slot print_timing`` block (anchored at its ``n_decoded`` line) up to the next
block/bench boundary.  This prevents the common failure mode of one log with
several runs (e.g. an ABBA file) being aggregated across runs.

Arm attribution order:

1. explicit ``--arm-regex`` match on the file name (default ``^([AB])(?:[-_.]|$)``,
   so ``A-1.log`` -> A, ``B-2.log`` -> B);
2. otherwise ``[run-start] <arm>`` markers in the same log.  If the number of
   successful marker runs equals the number of parsed timing sessions, assign
   them positionally (this handles T-0032's ``abba-4c.log`` = A B B A).  If all
   successful markers agree, use that arm for every session (``abba-4a/b.log``
   have A-only successful runs; their B runs failed to load);
3. otherwise ``unknown``.

``bench_attrib.py phase ...`` adds a phase-decomposition subcommand on top of the
same parsers: ``p_total = eval_ms / calls_g``, ``p_draft = dur_g / calls_g``,
``p_target = p_total - p_draft`` (serial upper bound, includes unhidden I/O), and
``p_target_ex_io = p_target - decode_pread / calls_g`` (lower bound when decode
pread is fully serial).  The legacy single-command CLI is unchanged.

``per_call_ms`` follows the task's explicit unit-test definition:
``dur_g / calls_g`` where ``dur_g`` is the middle (g) field of
``dur(b,g,a) = <b>, <g>, <a> ms``; for the sample block it is 589.643/34.
T-0054's 136.8 ms is a different metric: ``eval_ms / calls_g``.  It is exposed as
``per_call_ms_eval`` in JSON (not in the fixed TSV schema) so both definitions
stay visible without mixing them.

T-0344 adds a package-identity guard to ``phase``: ``stride0`` and
``requested_bytes`` are parsed verbatim from the log, and ``--expect-stride0`` /
``--expect-requested-bytes`` declare the expected package.  Under
``--package-policy exclude`` only deterministic mismatches are dropped from the
aggregate; ``ambiguous_same_stride0`` (notably stride0=442368, shared by P1/P2)
and ``missing`` are kept and reported.  The schema stays
``bench_attrib.phase.v1``; all additions are keys/columns appended at the end.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_ARM_REGEX = r"^([AB])(?:[-_.]|$)"

TSV_FIELDS = [
    "file",
    "session",
    "arm",
    "n_decoded",
    "tg_tps",
    "prompt_tps",
    "eval_ms",
    "graphs_reused",
    "acceptance",
    "accepted",
    "generated",
    "calls_g",
    "calls_a",
    "per_call_ms",
    "gen_tokens",
    "acc_tokens",
    "draft_gen",
    "draft_acc",
]

# Numeric fields included in per-arm aggregation.  Keep the TSV fields first;
# the extras are useful for explaining the two per-call definitions.
AGG_FIELDS = TSV_FIELDS[3:] + [
    "per_call_ms_eval",
    "calls_b",
    "eval_tps",
    "prompt_eval_ms",
    "total_ms",
    "dur_b_ms",
    "dur_g_ms",
    "dur_a_ms",
]

# --- phase subcommand -------------------------------------------------------
# One row per run (log).  The columns are deliberately a superset of the
# minimum requested by T-0128; the legacy TSV_FIELDS above are untouched.
PHASE_TSV_FIELDS = [
    "run",
    "log",
    "arm",
    "group",
    "tg",
    "eval_ms",
    "n_tokens",
    "calls_b",
    "calls_g",
    "calls_a",
    "draft_gen_ms",
    "acceptance",
    "p_total_ms",
    "p_draft_ms",
    "p_target_ms",
    "pread_ms",
    "prefill_pread_ms",
    "decode_pread_ms",
    "p_target_ex_io",
    "parse_warnings",
    # T-0344 package-identity columns; appended so the legacy column order above
    # (and every legacy value) stays byte-for-byte unchanged.
    "stride0",
    "requested_bytes",
    "package_ok",
    "package_reason",
]

# T-0320 established that stride0=442368 is shared by the P1 and P2 expert
# packages, so a matching stride0 alone must never be reported as ``match`` for
# that value.  1351680 is unique to P3 in that audit.  Unknown stride0 values are
# not assumed ambiguous: an explicit expectation is the user's declaration.
AMBIGUOUS_STRIDE0 = frozenset({442368})

# package_ok=False reasons that are *deterministic* inconsistencies and may be
# excluded under --package-policy exclude.  ``missing`` / ``ambiguous_*`` are
# never excluded: the tool refuses to guess.
PACKAGE_EXCLUDABLE_REASONS = frozenset(
    {"stride0_mismatch", "requested_mismatch", "conflicting_stride0"}
)
PACKAGE_REASONS = (
    "match",
    "stride0_mismatch",
    "requested_mismatch",
    "ambiguous_same_stride0",
    "missing",
    "conflicting_stride0",
)

PHASE_NUMERIC_FIELDS = [
    "tg",
    "eval_ms",
    "n_tokens",
    "calls_b",
    "calls_g",
    "calls_a",
    "draft_gen_ms",
    "acceptance",
    "p_total_ms",
    "p_draft_ms",
    "p_target_ms",
    "pread_ms",
    "prefill_pread_ms",
    "decode_pread_ms",
    "p_target_ex_io",
]

PHASE_SUFFIXES = {".log", ".err", ".out"}


RE_N_DECODED_ANCHOR = re.compile(r"\bn_decoded\s*=")
RE_N_DECODED = re.compile(r"\bn_decoded\s*=\s*(\d+)")
RE_TG = re.compile(r"\btg\s*=\s*([0-9.]+)\s*t/s")
RE_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?"
    r"([0-9.]+)\s*tokens per second"
)
RE_EVAL = re.compile(
    r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?"
    r"([0-9.]+)\s*tokens per second"
)
RE_TOTAL = re.compile(r"\btotal time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens")
RE_GRAPHS = re.compile(r"graphs reused\s*=\s*(\d+)")
RE_ACCEPTANCE = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\s*\)"
)
RE_CALLS = re.compile(r"#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s+(\d+)")
# Older logs sometimes print only two numbers.  Phase reads that as (b,g);
# calls_a is then genuinely unknown and stays null.
RE_CALLS_TWO = re.compile(r"#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s*(?:,|$)")
RE_DUR = re.compile(
    r"dur\(b,g,a\)\s*=\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*ms"
)
RE_GEN_DRAFTS = re.compile(r"#gen drafts\s*=\s*(\d+)")
RE_ACC_DRAFTS = re.compile(r"#acc drafts\s*=\s*(\d+)")
RE_GEN_TOKENS = re.compile(r"#gen tokens\s*=\s*(\d+)")
RE_ACC_TOKENS = re.compile(r"#acc tokens\s*=\s*(\d+)")
RE_PROMPT_SUMMARY = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]"
)
RE_RUN_START = re.compile(r"\[run-start\]\s+([A-Za-z][A-Za-z0-9_.-]*)")
RE_BENCH_HEADER = re.compile(r"^\s*\[bench\]\b")
RE_SPEC_RESET = re.compile(
    r"adding speculative implementation\s+['\"]?draft-dspark['\"]?"
)
# Phase I/O anchors.  Both the numbered ("total[prefill#3]") and the plain
# ("total[prefill]") forms occur in the wild; only "ms" is documented in the
# task spec, but "us"/"s" are converted instead of being guessed at.
RE_FINAL_TOTAL = re.compile(
    r"\[moe-ssd\]\s+total\[final\]:(?P<rest>.*?)\bpread\s*=\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|us|s)\b"
)
RE_PREFILL_TOTAL = re.compile(
    r"\[moe-ssd\]\s+total\[prefill(?P<index>#[0-9]+)?\]:(?P<rest>.*?)\bpread\s*=\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|us|s)\b"
)
# Package identity anchors (T-0344).  The init line is printed once per process
# and the io-summary line once per run; both are read verbatim, never guessed.
RE_STRIDE0 = re.compile(r"\bstride0=(\d+)")
RE_REQUESTED_BYTES = re.compile(r"\brequested_bytes=(\d+)")
RE_PACK_LAYERS = re.compile(r"\bpack_layers=(\d+/\d+)")
RE_EXPERTS = re.compile(r"\bexperts=(\d+)")

# Fields that make a session useful for attribution.  Missing ones are listed in
# session["missing"]; None is written as an empty TSV cell (never as zero).
KEY_FIELDS = [
    "n_decoded",
    "tg_tps",
    "prompt_tps",
    "eval_ms",
    "graphs_reused",
    "acceptance",
    "accepted",
    "generated",
    "calls_g",
    "calls_a",
    "per_call_ms",
    "gen_tokens",
    "acc_tokens",
    "draft_gen",
    "draft_acc",
]


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_session_block(lines: Sequence[str] | str) -> dict[str, Any]:
    """Parse one ``slot print_timing`` block into a session dict.

    Unknown lines are skipped.  Every numeric field that is absent or malformed
    is ``None``; no exception is raised for bad input.
    """
    if isinstance(lines, str):
        lines = lines.splitlines()

    n_decoded: int | None = None
    tg_tps: float | None = None
    prompt_tps: float | None = None
    eval_ms: float | None = None
    eval_tps: float | None = None
    prompt_eval_ms: float | None = None
    total_ms: float | None = None
    total_tokens: int | None = None
    graphs_reused: int | None = None
    acceptance: float | None = None
    accepted: int | None = None
    generated: int | None = None
    calls_b: int | None = None
    calls_g: int | None = None
    calls_a: int | None = None
    gen_tokens: int | None = None
    acc_tokens: int | None = None
    draft_gen: int | None = None
    draft_acc: int | None = None
    dur_b_ms: float | None = None
    dur_g_ms: float | None = None
    dur_a_ms: float | None = None
    summary_prompt: float | None = None
    summary_gen: float | None = None

    for line in lines:
        if not isinstance(line, str):
            continue

        m = RE_PROMPT_SUMMARY.search(line)
        if m:
            summary_prompt = _float(m.group(1))
            summary_gen = _float(m.group(2))

        m = RE_N_DECODED.search(line)
        if m:
            n_decoded = _int(m.group(1))
        m = RE_TG.search(line)
        if m:
            tg_tps = _float(m.group(1))

        # "prompt eval time" also contains the substring "eval time"; handle it
        # first and use elif so the decode eval timer is not overwritten.
        if "prompt eval time" in line:
            m = RE_PROMPT_EVAL.search(line)
            if m:
                prompt_eval_ms = _float(m.group(1))
                prompt_tps = _float(m.group(3))
        elif "eval time" in line:
            m = RE_EVAL.search(line)
            if m:
                eval_ms = _float(m.group(1))
                eval_tps = _float(m.group(3))

        m = RE_TOTAL.search(line)
        if m:
            total_ms = _float(m.group(1))
            total_tokens = _int(m.group(2))

        m = RE_GRAPHS.search(line)
        if m:
            graphs_reused = _int(m.group(1))

        m = RE_ACCEPTANCE.search(line)
        if m:
            acceptance = _float(m.group(1))
            accepted = _int(m.group(2))
            generated = _int(m.group(3))

        m = RE_CALLS.search(line)
        if m:
            calls_b = _int(m.group(1))
            calls_g = _int(m.group(2))
            calls_a = _int(m.group(3))

        m = RE_GEN_DRAFTS.search(line)
        if m:
            draft_gen = _int(m.group(1))
        m = RE_ACC_DRAFTS.search(line)
        if m:
            draft_acc = _int(m.group(1))
        m = RE_GEN_TOKENS.search(line)
        if m:
            gen_tokens = _int(m.group(1))
        m = RE_ACC_TOKENS.search(line)
        if m:
            acc_tokens = _int(m.group(1))

        m = RE_DUR.search(line)
        if m:
            dur_b_ms = _float(m.group(1))
            dur_g_ms = _float(m.group(2))
            dur_a_ms = _float(m.group(3))

    # [Prompt/Generation] is only a fallback; the timing-block values win.
    if prompt_tps is None:
        prompt_tps = summary_prompt
    if tg_tps is None:
        tg_tps = summary_gen

    per_call_ms: float | None = None
    # Spec: empty when calls_a == 0.  If calls_a is absent (None) but calls_g is
    # present, calls_g is the only available denominator.
    if dur_g_ms is not None and calls_g not in (None, 0) and calls_a != 0:
        per_call_ms = dur_g_ms / calls_g

    per_call_ms_eval: float | None = None
    if eval_ms is not None and calls_g not in (None, 0):
        per_call_ms_eval = eval_ms / calls_g

    session: dict[str, Any] = {
        "file": None,
        "session": None,
        "arm": "unknown",
        "n_decoded": n_decoded,
        "tg_tps": tg_tps,
        "prompt_tps": prompt_tps,
        "eval_ms": eval_ms,
        "graphs_reused": graphs_reused,
        "acceptance": acceptance,
        "accepted": accepted,
        "generated": generated,
        "calls_g": calls_g,
        "calls_a": calls_a,
        "per_call_ms": per_call_ms,
        "gen_tokens": gen_tokens,
        "acc_tokens": acc_tokens,
        "draft_gen": draft_gen,
        "draft_acc": draft_acc,
        # Extra diagnostic fields (JSON only; not columns of the fixed TSV).
        "calls_b": calls_b,
        "eval_tps": eval_tps,
        "prompt_eval_ms": prompt_eval_ms,
        "total_ms": total_ms,
        "total_tokens": total_tokens,
        "dur_b_ms": dur_b_ms,
        "dur_g_ms": dur_g_ms,
        "dur_a_ms": dur_a_ms,
        "per_call_ms_eval": per_call_ms_eval,
        "line": None,
        "anchor_fallback": False,
        "missing": [],
    }
    session["missing"] = [f for f in KEY_FIELDS if session.get(f) is None]
    return session


def parse_marker_runs(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Parse ``[run-start] <arm>`` blocks and whether they produced a summary."""
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        m = RE_RUN_START.search(line)
        if m:
            if current is not None:
                runs.append(current)
            current = {
                "arm": m.group(1).upper(),
                "success": False,
                "prompt_tps": None,
                "tg_tps": None,
            }
            continue
        if current is None:
            continue
        sm = RE_PROMPT_SUMMARY.search(line)
        if sm:
            current["prompt_tps"] = _float(sm.group(1))
            current["tg_tps"] = _float(sm.group(2))
            current["success"] = True
    if current is not None:
        runs.append(current)
    return runs


def arm_from_path(path: str | Path | None, arm_regex: str = DEFAULT_ARM_REGEX) -> str:
    """Infer an arm from a file name (or its parent directory)."""
    if path is None:
        return "unknown"
    p = Path(path)
    names = [p.name] + [parent.name for parent in p.parents if parent.name]
    for name in names:
        try:
            m = re.search(arm_regex, name)
        except re.error:
            # Invalid user regex: fail closed rather than crash the whole run.
            return "unknown"
        if m:
            arm = m.group(1) if m.groups() else m.group(0)
            return arm.upper() if arm else "unknown"
    return "unknown"


def _find_summary_before(lines: Sequence[str], start: int, lookback: int = 200) -> tuple[float, float] | None:
    """Find the nearest [Prompt/Generation] summary before a timing anchor."""
    stop = max(-1, start - lookback)
    for j in range(start - 1, stop, -1):
        line = lines[j]
        if RE_N_DECODED_ANCHOR.search(line) or RE_BENCH_HEADER.search(line) or RE_SPEC_RESET.search(line):
            break
        m = RE_PROMPT_SUMMARY.search(line)
        if m:
            return (_float(m.group(1)), _float(m.group(2)))  # type: ignore[return-value]
    return None


def parse_log(
    text: str,
    path: str | Path | None = None,
    arm_regex: str = DEFAULT_ARM_REGEX,
) -> list[dict[str, Any]]:
    """Parse a whole bench log into one dict per ``slot print_timing`` session."""
    lines = text.splitlines()
    anchors = [i for i, line in enumerate(lines) if RE_N_DECODED_ANCHOR.search(line)]

    sessions: list[dict[str, Any]] = []
    # llm_print_timings() only prints the ``n_decoded =`` anchor when
    # n_decoded >= 100 (tools/server/server-context.cpp), while the gate/sweep
    # default is N=64.  A whole .err file is then a legal single-run log: use
    # the entire file as exactly one session.  A .out/.log companion must never
    # take this path (it would duplicate the .err session in phase/quality).
    fallback = (not anchors and path is not None
                and Path(path).suffix == ".err")
    blocks = [(0, len(lines))] if fallback else []
    for k, start in enumerate(anchors):
        end = anchors[k + 1] if k + 1 < len(anchors) else len(lines)
        # A new bench invocation / speculative-init line always starts a new run;
        # stop the block there so a later run's counters cannot leak in.
        for j in range(start + 1, end):
            if RE_SPEC_RESET.search(lines[j]) or RE_BENCH_HEADER.search(lines[j]):
                end = j
                break
        blocks.append((start, end))

    for k, (start, end) in enumerate(blocks):
        session = parse_session_block(lines[start:end])
        # A garbage .err must still be reported as "no parsable timing
        # session": only accept the fallback when the file yielded real
        # timing fields (a valid n=64 block has everything except n_decoded).
        if fallback and not any(session.get(field) is not None for field in (
                "eval_ms", "prompt_eval_ms", "total_ms", "graphs_reused",
                "acceptance", "calls_g", "dur_g_ms")):
            continue
        summary = _find_summary_before(lines, start)
        if summary:
            if session["prompt_tps"] is None:
                session["prompt_tps"] = summary[0]
            if session["tg_tps"] is None:
                session["tg_tps"] = summary[1]
        session["missing"] = [f for f in KEY_FIELDS if session.get(f) is None]
        session["session"] = k + 1
        session["line"] = 1 if fallback else start + 1
        session["file"] = Path(path).name if path is not None else None
        session["anchor_fallback"] = bool(fallback)
        sessions.append(session)

    # Arm attribution.
    path_arm = arm_from_path(path, arm_regex)
    if path_arm != "unknown":
        arms = [path_arm] * len(sessions)
    else:
        runs = parse_marker_runs(lines)
        successful = [r["arm"] for r in runs if r.get("success")]
        all_markers = [r["arm"] for r in runs]
        if len(successful) == len(sessions):
            arms = successful
        elif len(all_markers) == len(sessions):
            arms = all_markers
        elif successful and len(set(successful)) == 1:
            arms = [successful[0]] * len(sessions)
        elif all_markers and len(set(all_markers)) == 1:
            arms = [all_markers[0]] * len(sessions)
        else:
            arms = ["unknown"] * len(sessions)

    for session, arm in zip(sessions, arms):
        session["arm"] = arm
    return sessions


def aggregate(sessions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-arm median/min/max/n for every numeric field (None-safe)."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        by_arm.setdefault(str(session.get("arm") or "unknown"), []).append(session)

    out: dict[str, dict[str, Any]] = {}
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        entry: dict[str, Any] = {
            "n": len(rows),
            "files": sorted({str(r.get("file")) for r in rows if r.get("file")}),
            "sessions": [r.get("session") for r in rows],
        }
        for field in AGG_FIELDS:
            values = [r[field] for r in rows if r.get(field) is not None]
            if values:
                entry[field] = {
                    "n": len(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
            else:
                entry[field] = {"n": 0, "median": None, "min": None, "max": None}
        entry["missing_counts"] = {
            field: sum(1 for r in rows if r.get(field) is None) for field in KEY_FIELDS
        }
        out[arm] = entry
    return out


def definitions() -> dict[str, str]:
    return {
        "session": "1-based index of a slot print_timing block in one file: one per n_decoded anchor, or exactly one whole-.err fallback session when there is no anchor",
        "anchor_fallback": (
            "true only for an .err log with no n_decoded line (n=64 print_timings_tg "
            "returns before printing it): the whole file is parsed as one session, "
            "session=1/line=1 (only when the file yields a real timing block; "
            "garbage .err still has no parsable session).  Multiple runs in one "
            "anchorless file are unsupported; .out/.log companions never use this "
            "fallback.  false on anchored logs."
        ),
        "arm": (
            "explicit --arm-regex on file/dir name first; else [run-start] arm markers "
            "paired positionally with timing sessions when counts agree; else unknown"
        ),
        "tg_tps": "tg from the n_decoded slot print_timing line; fallback [Generation: N t/s]",
        "prompt_tps": "tokens per second from prompt eval time; fallback [Prompt: N t/s]",
        "eval_ms": "decode eval time in ms (the line that starts with 'eval time =')",
        "per_call_ms": (
            "dur_g / calls_g, where dur_g is the middle/g field of dur(b,g,a)=b,g,a ms; "
            "sample: 589.643/34 = 17.34235294; empty when calls_g==0"
        ),
        "per_call_ms_eval": (
            "eval_ms / calls_g; this is T-0054's 'per verification call' 136.8 ms metric, "
            "NOT the TSV per_call_ms. Reported in JSON only."
        ),
        "missing": "key fields absent/malformed in this session; never filled with 0",
    }


def expand_logs(patterns: Iterable[str]) -> list[Path]:
    """Expand directories, files and glob patterns into a sorted unique list."""
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.log")))
        elif p.is_file():
            paths.append(p)
        else:
            paths.extend(sorted(Path(x) for x in glob.glob(pattern, recursive=True) if Path(x).is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _num(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def write_tsv(path: str | Path, sessions: Sequence[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_FIELDS)
        for session in sessions:
            writer.writerow([_num(session.get(field)) for field in TSV_FIELDS])


def build_result(paths: Sequence[Path], arm_regex: str) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"bench_attrib: warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        parsed = parse_log(text, path=path, arm_regex=arm_regex)
        sessions.extend(parsed)
        files.append(
            {
                "file": path.name,
                "path": str(path),
                "sessions": len(parsed),
                "arm_markers": [r["arm"] for r in parse_marker_runs(text.splitlines())],
            }
        )
    return {
        "schema": "bench_attrib.v1",
        "definitions": definitions(),
        "files": files,
        "sessions": sessions,
        "arms": aggregate(sessions),
    }


def phase_definitions() -> dict[str, str]:
    """Documented phase semantics; also embedded in the JSON output."""
    return {
        "calls_g": "second number of #calls(b,g,a) (the g field); denominator of every per-call metric",
        "eval_ms": "final 'eval time = <ms> ms / N tokens' line (decode eval; prompt eval is excluded)",
        "n_tokens": "token count N from that same final eval line",
        "tg": "tokens per second from the final eval line (T-0102 convention, not the n_decoded snapshot)",
        "draft_gen_ms": "middle (g) field of dur(b,g,a) = <b>, <g>, <a> ms",
        "group": "aggregate label: --group-regex match, else arm, else parent directory",
        "p_total_ms": "eval_ms / calls_g",
        "p_draft_ms": "draft_gen_ms / calls_g",
        "p_target_ms": (
            "p_total_ms - p_draft_ms; SERIAL UPPER BOUND = target forward plus I/O that is "
            "not hidden by prefetch.  It is NOT pure HTP compute."
        ),
        "p_target_ex_io": (
            "p_target_ms - decode_pread_ms / calls_g; LOWER BOUND obtained by charging the "
            "whole decode-phase pread as a serial stall (I/O fully unhidden)."
        ),
        "pread_ms": "pread from the last total[final] line",
        "prefill_pread_ms": (
            "pread from the last/largest-index total[prefill] or total[prefill#N] line; "
            "null when the log has no prefill line"
        ),
        "decode_pread_ms": "pread_ms - prefill_pread_ms; null when either input is missing",
        "stride0": (
            "package layout stride from '[moe-ssd] init ... stride0=<n>'; null when the "
            "log has no stride0 or when several occurrences disagree (stride0_values)"
        ),
        "requested_bytes": (
            "run-level logical read from the last '[moe-ssd] io-summary ... "
            "requested_bytes=<n>' line; not a physical-read or generic package fingerprint"
        ),
        "pack_layers": "optional auxiliary field from the init line (e.g. 40/40)",
        "experts": "optional auxiliary expert count from the init line",
        "package_ok": (
            "true = matched an explicit/auto expectation, or a consistent non-ambiguous "
            "stride0 group with no expectation; false = deterministic mismatch; "
            "null = ambiguous or missing (never excluded)"
        ),
        "package_reason": (
            "match | stride0_mismatch | requested_mismatch | ambiguous_same_stride0 | "
            "missing | conflicting_stride0"
        ),
        "excluded": (
            "true only when --package-policy exclude and package_ok is false with a "
            "deterministic reason; excluded runs stay in runs[] but not in aggregate"
        ),
        "null": "missing or unparsable values are null, never 0",
        "parse_warnings": (
            "number of lines carrying a known bench marker (eval time, draft acceptance, "
            "#calls, dur(b,g,a), total[final], total[prefill]) that failed to parse; "
            "details are in runs[].parse_warnings"
        ),
    }


def _unit_to_ms(value: float, unit: str) -> float:
    if unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value


def _phase_raw(text: str, path: str | Path | None = None, arm_regex: str = DEFAULT_ARM_REGEX) -> dict[str, Any]:
    """Extract phase inputs from one log, reusing the legacy parsers.

    ``parse_log`` supplies eval/draft/calls/acceptance/arm; this function only
    reuses the existing module regexes to recover the eval token count and the
    ``[moe-ssd] total[...]`` pread anchors.  No regex is copied from T-0102.
    """
    lines = text.splitlines()
    sessions = parse_log(text, path=path, arm_regex=arm_regex)
    session = sessions[-1] if sessions else None
    warnings: list[str] = []
    if not sessions:
        warnings.append("no slot print_timing session found")
    elif len(sessions) > 1:
        warnings.append(f"{len(sessions)} print_timing sessions; phase uses the last one")

    raw: dict[str, Any] = {
        "eval_ms": session.get("eval_ms") if session else None,
        "n_tokens": None,
        "tg": session.get("tg_tps") if session else None,
        "calls_b": session.get("calls_b") if session else None,
        "calls_g": session.get("calls_g") if session else None,
        "calls_a": session.get("calls_a") if session else None,
        "draft_gen_ms": session.get("dur_g_ms") if session else None,
        "acceptance": session.get("acceptance") if session else None,
        "pread_ms": None,
        "prefill_pread_ms": None,
        # Package identity (T-0344).  None means "not printed in this log"; it
        # is never inferred from another file or from the package name.
        "stride0": None,
        "requested_bytes": None,
        "pack_layers": None,
        "experts": None,
        "stride0_values": [],
        "stride0_conflict": False,
        "session": session.get("session") if session else None,
        "arm": session.get("arm") if session else arm_from_path(path, arm_regex),
    }

    last_eval: tuple[int, float, int, float] | None = None
    last_final: tuple[int, float] | None = None
    last_two_calls: tuple[int, int] | None = None
    last_requested: tuple[int, int] | None = None
    stride0_values: list[int] = []
    prefills: list[tuple[int | None, int, float]] = []

    for lineno, line in enumerate(lines, 1):
        if not isinstance(line, str):
            continue
        if "prompt eval time" in line:
            if not RE_PROMPT_EVAL.search(line):
                warnings.append(f"line {lineno}: unparsed prompt eval time")
        elif "eval time" in line:
            m = RE_EVAL.search(line)
            if m:
                last_eval = (lineno, float(m.group(1)), int(m.group(2)), float(m.group(3)))
            else:
                warnings.append(f"line {lineno}: unparsed eval time")

        if "draft acceptance" in line and not RE_ACCEPTANCE.search(line):
            warnings.append(f"line {lineno}: unparsed draft acceptance")
        if "#calls(b,g,a)" in line:
            if not RE_CALLS.search(line):
                two = RE_CALLS_TWO.search(line)
                if two:
                    last_two_calls = (int(two.group(1)), int(two.group(2)))
                else:
                    warnings.append(f"line {lineno}: unparsed #calls(b,g,a)")
        if "dur(b,g,a)" in line and not RE_DUR.search(line):
            warnings.append(f"line {lineno}: unparsed dur(b,g,a)")

        if "total[final]" in line:
            m = RE_FINAL_TOTAL.search(line)
            if m:
                last_final = (lineno, _unit_to_ms(float(m.group("value")), m.group("unit")))
            else:
                warnings.append(f"line {lineno}: unparsed total[final] pread (unknown unit or format)")
        if "total[prefill" in line:
            m = RE_PREFILL_TOTAL.search(line)
            if m:
                index = _int(m.group("index")[1:]) if m.group("index") else None
                prefills.append(
                    (index, lineno, _unit_to_ms(float(m.group("value")), m.group("unit")))
                )
            else:
                warnings.append(f"line {lineno}: unparsed total[prefill] pread (unknown unit or format)")

        # Package identity.  Collect *all* stride0 values on a line (a malformed
        # concatenation must surface as a conflict, not as a silent last-wins).
        if "stride0=" in line:
            found = RE_STRIDE0.findall(line)
            if found:
                stride0_values.extend(int(v) for v in found)
            else:
                warnings.append(f"line {lineno}: unparsed stride0")
        if "requested_bytes=" in line:
            m = RE_REQUESTED_BYTES.search(line)
            if m:
                last_requested = (lineno, int(m.group(1)))
            else:
                warnings.append(f"line {lineno}: unparsed requested_bytes")
        if "pack_layers=" in line:
            m = RE_PACK_LAYERS.search(line)
            if m:
                raw["pack_layers"] = m.group(1)
        if "experts=" in line:
            m = RE_EXPERTS.search(line)
            if m:
                raw["experts"] = int(m.group(1))

    if last_eval is not None:
        _, eval_ms, n_tokens, eval_tps = last_eval
        raw["eval_ms"] = eval_ms
        raw["n_tokens"] = n_tokens
        raw["tg"] = eval_tps
    if last_final is not None:
        raw["pread_ms"] = last_final[1]
    else:
        warnings.append("missing total[final] pread")

    if prefills:
        numbered = [p for p in prefills if p[0] is not None]
        if numbered:
            chosen = max(numbered, key=lambda p: (p[0], p[1]))
        else:
            chosen = prefills[-1]
        raw["prefill_pread_ms"] = chosen[2]
    else:
        warnings.append("missing total[prefill] pread")

    # Companion .out files may contain only a few timing lines and therefore no
    # n_decoded anchor/session.  Reuse the existing line parser for those, but
    # never let an earlier session fill a field that the last session failed to
    # parse (that would hide a real bad line).
    if raw.get("calls_g") is None and last_two_calls is not None:
        # Two-number form: interpret as (b, g); there is no calls_a value.
        raw["calls_b"], raw["calls_g"] = last_two_calls
        raw["calls_a"] = None
        warnings.append("two-number #calls(b,g,a) form: calls_g is the second number, calls_a unknown")

    if session is None:
        block = parse_session_block(lines)
        fallbacks = {
            "eval_ms": "eval_ms",
            "tg": "tg_tps",
            "calls_b": "calls_b",
            "calls_g": "calls_g",
            "calls_a": "calls_a",
            "draft_gen_ms": "dur_g_ms",
            "acceptance": "acceptance",
        }
        for field, source in fallbacks.items():
            if raw.get(field) is None:
                raw[field] = block.get(source)

    # stride0 is only usable when every occurrence agrees.  A single log with
    # two different values is a deterministic inconsistency, not a "last wins".
    distinct_stride0 = sorted(set(stride0_values))
    raw["stride0_values"] = distinct_stride0
    if len(distinct_stride0) > 1:
        raw["stride0_conflict"] = True
        raw["stride0"] = None
        warnings.append(
            "conflicting stride0 values: " + ", ".join(str(v) for v in distinct_stride0)
        )
    elif distinct_stride0:
        raw["stride0"] = distinct_stride0[0]
    if last_requested is not None:
        # The last io-summary is the run-level total (scope=all_graphs_...).
        raw["requested_bytes"] = last_requested[1]

    raw["parse_warnings"] = warnings
    return raw


def _merge_missing_phase(primary: dict[str, Any], companion: dict[str, Any]) -> None:
    """Fill only missing phase inputs from a sibling ``.out`` file."""
    if primary.get("eval_ms") is None:
        primary["eval_ms"] = companion.get("eval_ms")
        primary["n_tokens"] = companion.get("n_tokens")
        primary["tg"] = companion.get("tg")
    elif primary.get("n_tokens") is None:
        primary["n_tokens"] = companion.get("n_tokens")
    if primary.get("tg") is None:
        primary["tg"] = companion.get("tg")
    for field in ("calls_b", "calls_g", "calls_a", "draft_gen_ms", "acceptance",
                  "pread_ms", "prefill_pread_ms"):
        if primary.get(field) is None:
            primary[field] = companion.get(field)
    if primary.get("session") is None:
        primary["session"] = companion.get("session")

    # Package identity is merged separately.  Any disagreement (including a
    # conflict already seen inside either file) becomes conflicting_stride0;
    # neither file is allowed to silently win.
    p_stride0 = primary.get("stride0")
    c_stride0 = companion.get("stride0")
    if primary.get("stride0_conflict") or companion.get("stride0_conflict"):
        values = set(primary.get("stride0_values") or [])
        values.update(companion.get("stride0_values") or [])
        if p_stride0 is not None:
            values.add(p_stride0)
        if c_stride0 is not None:
            values.add(c_stride0)
        primary["stride0_conflict"] = True
        primary["stride0_values"] = sorted(values)
        primary["stride0"] = None
    elif p_stride0 is not None and c_stride0 is not None and p_stride0 != c_stride0:
        primary["stride0_conflict"] = True
        primary["stride0_values"] = sorted({p_stride0, c_stride0})
        primary["stride0"] = None
    elif p_stride0 is None and c_stride0 is not None:
        primary["stride0"] = c_stride0
        primary["stride0_values"] = list(companion.get("stride0_values") or [c_stride0])
    for field in ("requested_bytes", "pack_layers", "experts"):
        if primary.get(field) is None:
            primary[field] = companion.get(field)


def _phase_group(path: Path, arm: str, group_regex: str | None, warnings: list[str]) -> str:
    if group_regex:
        try:
            m = re.search(group_regex, str(path.resolve()))
        except re.error as exc:
            warnings.append(f"invalid --group-regex {group_regex!r}: {exc}")
            m = None
        if m:
            # Named groups are handy for device labels:
            #   --group-regex '(?P<nubia>T-0031)|(?P<vivo>T-0030)'
            named = [name for name, value in m.groupdict().items() if value is not None]
            if named:
                return named[0]
            if m.lastindex:
                return m.group(1)
            return m.group(0)
    if arm and arm != "unknown":
        return arm
    parent = path.parent.name
    return parent or "unknown"


def _phase_row(raw: dict[str, Any], path: Path, group_regex: str | None, redact: bool,
               warnings: list[str], companion: Path | None = None) -> dict[str, Any]:
    calls_g = raw.get("calls_g")
    eval_ms = raw.get("eval_ms")
    draft_gen_ms = raw.get("draft_gen_ms")
    pread_ms = raw.get("pread_ms")
    prefill_pread_ms = raw.get("prefill_pread_ms")

    per_call_ok = calls_g is not None and calls_g > 0
    p_total_ms = eval_ms / calls_g if per_call_ok and eval_ms is not None else None
    p_draft_ms = draft_gen_ms / calls_g if per_call_ok and draft_gen_ms is not None else None
    p_target_ms = (
        p_total_ms - p_draft_ms
        if p_total_ms is not None and p_draft_ms is not None
        else None
    )
    decode_pread_ms = (
        pread_ms - prefill_pread_ms
        if pread_ms is not None and prefill_pread_ms is not None
        else None
    )
    p_target_ex_io = (
        p_target_ms - decode_pread_ms / calls_g
        if p_target_ms is not None and decode_pread_ms is not None and per_call_ok
        else None
    )

    row: dict[str, Any] = {
        "run": path.stem,
        "log": path.name if redact else str(path.resolve()),
        "arm": raw.get("arm") or "unknown",
        "group": _phase_group(path, str(raw.get("arm") or "unknown"), group_regex, warnings),
        "session": raw.get("session"),
        "tg": raw.get("tg"),
        "eval_ms": eval_ms,
        "n_tokens": raw.get("n_tokens"),
        "calls_b": raw.get("calls_b"),
        "calls_g": calls_g,
        "calls_a": raw.get("calls_a"),
        "draft_gen_ms": draft_gen_ms,
        "acceptance": raw.get("acceptance"),
        "p_total_ms": p_total_ms,
        "p_draft_ms": p_draft_ms,
        "p_target_ms": p_target_ms,
        "pread_ms": pread_ms,
        "prefill_pread_ms": prefill_pread_ms,
        "decode_pread_ms": decode_pread_ms,
        # Aliases requested by the task prose (kept in JSON only).
        "decode_pread_delta_ms": decode_pread_ms,
        "p_target_ms_ex_io": p_target_ex_io,
        "p_target_ex_io": p_target_ex_io,
        "parse_warnings": list(raw.get("parse_warnings") or []),
        "parse_warning_count": len(raw.get("parse_warnings") or []),
        # Package identity: raw observations are filled here; package_ok and
        # package_reason are decided once all runs are visible (cross-run
        # same-stride0 ambiguity needs the whole set).
        "stride0": raw.get("stride0"),
        "requested_bytes": raw.get("requested_bytes"),
        "pack_layers": raw.get("pack_layers"),
        "experts": raw.get("experts"),
        "stride0_values": list(raw.get("stride0_values") or []),
        "stride0_conflict": bool(raw.get("stride0_conflict")),
        "package_ok": None,
        "package_reason": None,
        "excluded": False,
        "companion": (companion.name if companion is not None else None) if redact
        else (str(companion.resolve()) if companion is not None else None),
    }
    row["missing"] = [field for field in PHASE_NUMERIC_FIELDS if row.get(field) is None]
    return row


def expand_phase_logs(patterns: Iterable[str]) -> list[Path]:
    """Expand files, directories (*.log/*.err/*.out) and glob patterns."""
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(
                sorted(
                    x for x in p.rglob("*")
                    if x.is_file() and x.suffix.lower() in PHASE_SUFFIXES
                )
            )
        elif p.is_file():
            paths.append(p)
        else:
            paths.extend(
                sorted(
                    Path(x) for x in glob.glob(pattern, recursive=True)
                    if Path(x).is_file()
                )
            )
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    # A sibling .out is only a companion of .err/.log; do not count it twice
    # when a whole directory is expanded.
    resolved = {p.resolve() for p in unique}
    filtered: list[Path] = []
    for p in unique:
        if p.suffix.lower() == ".out":
            siblings = {p.with_suffix(".err").resolve(), p.with_suffix(".log").resolve()}
            if siblings & resolved:
                continue
        filtered.append(p)
    return filtered


def aggregate_phase(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Median/min/max per group (group = --group-regex, arm, or directory)."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_group.setdefault(str(run.get("group") or "unknown"), []).append(run)

    out: dict[str, dict[str, Any]] = {}
    for group in sorted(by_group):
        rows = by_group[group]
        entry: dict[str, Any] = {
            "n": len(rows),
            "runs": [r.get("run") for r in rows],
            "metrics": {},
            "parse_warning_count": sum(int(r.get("parse_warning_count") or 0) for r in rows),
            "calls_g_values": sorted({r["calls_g"] for r in rows if r.get("calls_g") is not None}),
        }
        for field in PHASE_NUMERIC_FIELDS:
            values = [r[field] for r in rows if r.get(field) is not None]
            if values:
                entry["metrics"][field] = {
                    "n": len(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
            else:
                entry["metrics"][field] = {"n": 0, "median": None, "min": None, "max": None}
        # Flat *_median keys keep the JSON easy to consume (task example:
        # aggregate.nubia.tg_median == 27.89).
        for field in ("tg", "eval_ms", "draft_gen_ms", "acceptance", "p_total_ms",
                      "p_draft_ms", "p_target_ms", "pread_ms", "prefill_pread_ms",
                      "decode_pread_ms", "p_target_ex_io"):
            entry[f"{field}_median"] = entry["metrics"][field]["median"]
        out[group] = entry
    return out


def evaluate_packages(
    runs: Sequence[dict[str, Any]],
    expect_stride0: int | None = None,
    expect_requested_bytes: int | None = None,
    requested_tol: float = 0.01,
    policy: str = "warn",
    auto: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Classify each run's package identity and mark exclusions.

    Mutates ``package_ok`` / ``package_reason`` / ``excluded`` on every run and
    returns ``(package_block, warnings)``.  Only deterministic mismatches are
    excluded; ``ambiguous_same_stride0`` and ``missing`` are always kept, because
    the tool never guesses a package (T-0320: stride0=442368 is P1|P2).
    """
    if policy not in {"warn", "exclude"}:
        raise ValueError(f"unknown package policy {policy!r}")
    tol = max(0.0, float(requested_tol))
    warnings: list[str] = []

    stride0_counts: dict[int, int] = {}
    for run in runs:
        s = run.get("stride0")
        if s is not None:
            stride0_counts[int(s)] = stride0_counts.get(int(s), 0) + 1

    # --package-auto: majority stride0, but only when there is a strict
    # majority.  A tie must not be broken arbitrarily; nothing is excluded then.
    auto_stride0: int | None = None
    if expect_stride0 is None and auto and stride0_counts:
        ranked = sorted(stride0_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            auto_stride0 = ranked[0][0]
        else:
            warnings.append(
                "--package-auto: stride0 majority is tied "
                f"({', '.join(f'{s}x{n}' for s, n in ranked)}); no run excluded"
            )
    effective_expect_stride0 = expect_stride0 if expect_stride0 is not None else auto_stride0
    if effective_expect_stride0 is None and len(stride0_counts) > 1:
        warnings.append(
            "multiple stride0 groups without an expected package "
            f"({', '.join(f'{s}x{n}' for s, n in sorted(stride0_counts.items()))}); "
            "nothing excluded, use --expect-stride0 or --package-auto"
        )

    # Same-stride0 / different-requested runs cannot be told apart (P1 vs P2).
    # Cluster within each stride0 group only when the caller did not supply an
    # explicit byte expectation (which would make the check deterministic).
    ambiguous_groups: set[int] = set()
    if expect_requested_bytes is None:
        by_stride0: dict[int, list[dict[str, Any]]] = {}
        for run in runs:
            s = run.get("stride0")
            if s is not None:
                by_stride0.setdefault(int(s), []).append(run)
        for s, group_runs in by_stride0.items():
            values = [
                int(r["requested_bytes"]) for r in group_runs
                if r.get("requested_bytes") is not None
            ]
            if len(values) < 2:
                continue
            lo, hi = min(values), max(values)
            if hi > 0 and (hi - lo) / hi > tol:
                ambiguous_groups.add(s)

    for run in runs:
        s = run.get("stride0")
        q = run.get("requested_bytes")
        ok: bool | None
        reason: str
        if run.get("stride0_conflict"):
            ok, reason = False, "conflicting_stride0"
        elif s is None and q is None:
            ok, reason = None, "missing"
        elif effective_expect_stride0 is not None and s is None:
            ok, reason = None, "missing"
        elif effective_expect_stride0 is not None and int(s) != int(effective_expect_stride0):
            ok, reason = False, "stride0_mismatch"
        elif expect_requested_bytes is not None and q is None:
            ok, reason = None, "missing"
        elif expect_requested_bytes is not None:
            denom = abs(int(expect_requested_bytes)) or 1
            if abs(int(q) - int(expect_requested_bytes)) / denom > tol:
                ok, reason = False, "requested_mismatch"
            else:
                ok, reason = True, "match"
        elif s is None:
            ok, reason = None, "missing"
        elif effective_expect_stride0 is not None:
            # stride0 matched the explicit/auto expectation, but there is no
            # byte-level check.  442368 alone never proves P1 vs P2.
            if int(s) in ambiguous_groups or int(s) in AMBIGUOUS_STRIDE0:
                ok, reason = None, "ambiguous_same_stride0"
            else:
                ok, reason = True, "match"
        elif int(s) in ambiguous_groups:
            ok, reason = None, "ambiguous_same_stride0"
        elif int(s) in AMBIGUOUS_STRIDE0:
            # Even without an expectation, 442368 alone can be P1 or P2.
            ok, reason = None, "ambiguous_same_stride0"
        else:
            # No expectation at all: a single consistent, non-ambiguous stride0
            # group has no detected package contradiction (old warn behaviour).
            ok, reason = True, "match"

        run["package_ok"] = ok
        run["package_reason"] = reason
        run["excluded"] = bool(
            policy == "exclude" and ok is False and reason in PACKAGE_EXCLUDABLE_REASONS
        )

    guard_active = bool(
        expect_stride0 is not None
        or expect_requested_bytes is not None
        or auto
        or policy == "exclude"
    )
    for run in runs:
        reason = str(run.get("package_reason"))
        if run.get("excluded"):
            warnings.append(
                f"{run['run']}: excluded from aggregate ({reason}; "
                f"stride0={run.get('stride0')}, requested_bytes={run.get('requested_bytes')})"
            )
        elif guard_active and reason in PACKAGE_EXCLUDABLE_REASONS:
            warnings.append(
                f"{run['run']}: package mismatch ({reason}); kept by {policy} policy"
            )
        elif guard_active and reason == "ambiguous_same_stride0":
            warnings.append(
                f"{run['run']}: ambiguous package identity (stride0={run.get('stride0')}, "
                f"requested_bytes={run.get('requested_bytes')}); kept, not guessed"
            )
        elif guard_active and reason == "missing":
            warnings.append(f"{run['run']}: package identity missing; kept")

    counts = {reason: 0 for reason in PACKAGE_REASONS}
    for run in runs:
        reason = str(run.get("package_reason"))
        counts[reason] = counts.get(reason, 0) + 1

    package_block: dict[str, Any] = {
        "policy": policy,
        "auto": bool(auto),
        "guard_active": guard_active,
        "expect_stride0": expect_stride0,
        "auto_stride0": auto_stride0,
        "effective_expect_stride0": effective_expect_stride0,
        "expect_requested_bytes": expect_requested_bytes,
        "requested_tol": tol,
        "ambiguous_stride0": sorted(AMBIGUOUS_STRIDE0),
        "stride0_counts": {str(k): v for k, v in sorted(stride0_counts.items())},
        "counts": counts,
        "excluded_runs": [r["run"] for r in runs if r.get("excluded")],
        "ambiguous_runs": [
            r["run"] for r in runs if r.get("package_reason") == "ambiguous_same_stride0"
        ],
        "missing_runs": [
            r["run"] for r in runs if r.get("package_reason") == "missing"
        ],
    }
    return package_block, warnings


def _phase_tsv_cell(run: dict[str, Any], field: str) -> str:
    if field == "parse_warnings":
        return _num(run.get("parse_warning_count"))
    if field == "package_ok":
        value = run.get("package_ok")
        return "" if value is None else ("true" if value else "false")
    return _num(run.get(field))


def write_phase_tsv(path: str | Path, runs: Sequence[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(PHASE_TSV_FIELDS)
        for run in runs:
            writer.writerow([_phase_tsv_cell(run, field) for field in PHASE_TSV_FIELDS])


def build_phase_result(
    paths: Sequence[Path],
    arm_regex: str = DEFAULT_ARM_REGEX,
    group_regex: str | None = None,
    redact: bool = True,
    expect_stride0: int | None = None,
    expect_requested_bytes: int | None = None,
    requested_tol: float = 0.01,
    package_policy: str = "warn",
    package_auto: bool = False,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_runs: set[str] = set()

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warnings.append(f"cannot read {path.name}: {exc.strerror or exc}")
            print(f"bench_attrib: warning: cannot read {path}: {exc}", file=sys.stderr)
            continue

        raw = _phase_raw(text, path=path, arm_regex=arm_regex)
        companion: Path | None = None
        if path.suffix.lower() in {".err", ".log"}:
            candidate = path.with_suffix(".out")
            if candidate.is_file():
                try:
                    companion_text = candidate.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    warnings.append(f"cannot read companion {candidate.name}: {exc.strerror or exc}")
                else:
                    companion = candidate
                    _merge_missing_phase(raw, _phase_raw(companion_text, path=candidate, arm_regex=arm_regex))

        if not any(raw.get(field) is not None for field in (
            "eval_ms", "calls_g", "draft_gen_ms", "pread_ms", "prefill_pread_ms"
        )):
            warnings.append(f"{path.name}: no bench phase fields; skipped")
            continue

        row = _phase_row(raw, path, group_regex, redact, warnings, companion=companion)
        if row["run"] in seen_runs:
            warnings.append(f"duplicate run name {row['run']!r} (log column keeps basenames)")
        seen_runs.add(row["run"])
        runs.append(row)

    package_block, package_warnings = evaluate_packages(
        runs,
        expect_stride0=expect_stride0,
        expect_requested_bytes=expect_requested_bytes,
        requested_tol=requested_tol,
        policy=package_policy,
        auto=package_auto,
    )
    warnings.extend(package_warnings)

    kept = [run for run in runs if not run.get("excluded")]
    aggregate = aggregate_phase(kept)
    # Additive key (schema stays bench_attrib.phase.v1): which runs were dropped
    # from the medians, and why they are auditable in runs[]/package.
    aggregate["excluded_runs"] = list(package_block["excluded_runs"])

    return {
        "version": 1,
        "schema": "bench_attrib.phase.v1",
        "definitions": phase_definitions(),
        "package": package_block,
        "runs": runs,
        "aggregate": aggregate,
        "warnings": warnings,
    }


def phase_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_attrib.py phase",
        description=(
            "Decompose bench logs into per-verify-call phases: target (serial upper "
            "bound), draft, and I/O (pread).  Missing fields are null, never 0."
        ),
        epilog=(
            "PHASE DEFINITIONS\n"
            "  calls_g        = second number of #calls(b,g,a)\n"
            "  eval_ms        = final 'eval time = ... ms / N tokens' (decode eval)\n"
            "  draft_gen_ms   = middle/g field of dur(b,g,a)\n"
            "  p_total_ms     = eval_ms / calls_g\n"
            "  p_draft_ms     = draft_gen_ms / calls_g\n"
            "  p_target_ms    = p_total_ms - p_draft_ms\n"
            "                   SERIAL UPPER BOUND: target forward + I/O not hidden by\n"
            "                   prefetch.  It is NOT pure HTP compute.\n"
            "  pread_ms       = last total[final].pread\n"
            "  prefill_pread  = last/largest-index total[prefill(#N)].pread\n"
            "  decode_pread   = pread_ms - prefill_pread_ms\n"
            "  p_target_ex_io = p_target_ms - decode_pread_ms / calls_g\n"
            "                   LOWER BOUND if the whole decode pread is a serial stall.\n"
            "\n"
            "PACKAGE IDENTITY (additive, T-0344)\n"
            "  stride0 / requested_bytes are read from '[moe-ssd] init ...' and the\n"
            "  last '[moe-ssd] io-summary ... requested_bytes=' line.  stride0 must\n"
            "  agree everywhere in a log, else the reason is conflicting_stride0.\n"
            "  --expect-stride0 / --expect-requested-bytes declare the expected\n"
            "  package; --requested-tol (default 0.01) sets the byte tolerance.\n"
            "  --package-policy exclude drops only deterministic mismatches;\n"
            "  ambiguous_same_stride0 and missing are always kept.  stride0=442368\n"
            "  alone never proves P1 vs P2 (T-0320), so it stays ambiguous.\n"
            "  --package-auto picks the strict-majority stride0 when no explicit\n"
            "  expectation is given; ties are not broken and exclude nothing.\n"
            "\n"
            "Paths are redacted to basenames by default; use --no-redact-paths for\n"
            "absolute paths.  Without 'phase', the legacy single-command CLI is unchanged.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        metavar="PATH",
        help="bench log file(s) (.log/.err/.out), directory (recursive), or glob pattern",
    )
    parser.add_argument("--out-tsv", help="write the phase TSV here")
    parser.add_argument("--out-json", help="write the phase JSON here")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the phase JSON to stdout",
    )
    parser.add_argument(
        "--arm-regex",
        default=DEFAULT_ARM_REGEX,
        help=f"arm regex on file/dir name (default: {DEFAULT_ARM_REGEX})",
    )
    parser.add_argument(
        "--group-regex",
        default=None,
        help=(
            "regex on the full path used to label aggregate groups; a named group "
            "wins (e.g. '(?P<nubia>T-0031)|(?P<vivo>T-0030)'), else the first "
            "capture group, else the whole match.  Default: arm, then parent dir."
        ),
    )
    parser.add_argument(
        "--redact-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only basenames in log/companion fields (default: on)",
    )
    parser.add_argument(
        "--expect-stride0",
        type=int,
        default=None,
        metavar="N",
        help="expected package stride0; a deterministic different value is a mismatch",
    )
    parser.add_argument(
        "--expect-requested-bytes",
        type=int,
        default=None,
        metavar="N",
        help="expected run-level requested_bytes (last io-summary line)",
    )
    parser.add_argument(
        "--requested-tol",
        type=float,
        default=0.01,
        metavar="F",
        help="relative tolerance for --expect-requested-bytes (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--package-policy",
        choices=("warn", "exclude"),
        default="warn",
        help=(
            "warn (default) only marks package mismatches; exclude drops runs with "
            "a deterministic mismatch from the aggregate (ambiguous/missing are kept)"
        ),
    )
    parser.add_argument(
        "--package-auto",
        action="store_true",
        help=(
            "without an explicit expectation, use the strict-majority stride0 as the "
            "reference; a minority group is excluded under --package-policy exclude"
        ),
    )
    args = parser.parse_args(argv)
    if args.requested_tol < 0:
        parser.error("--requested-tol must be >= 0")

    paths = expand_phase_logs(args.logs)
    result = build_phase_result(
        paths,
        arm_regex=args.arm_regex,
        group_regex=args.group_regex,
        redact=args.redact_paths,
        expect_stride0=args.expect_stride0,
        expect_requested_bytes=args.expect_requested_bytes,
        requested_tol=args.requested_tol,
        package_policy=args.package_policy,
        package_auto=args.package_auto,
    )
    if not paths:
        display = [Path(p).name for p in args.logs] if args.redact_paths else list(args.logs)
        print(f"bench_attrib: no logs matched {display!r}", file=sys.stderr)
        result["warnings"].append(f"no logs matched {display!r}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_tsv:
        write_phase_tsv(args.out_tsv, result["runs"])
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif not args.out_tsv and not args.out_json:
        writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
        writer.writerow(PHASE_TSV_FIELDS)
        for run in result["runs"]:
            writer.writerow([_phase_tsv_cell(run, field) for field in PHASE_TSV_FIELDS])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "phase":
        return phase_main(args[1:])

    parser = argparse.ArgumentParser(
        description="Parse llama.cpp bench logs into per-session TSV and per-arm JSON.",
        epilog=(
            "Phase decomposition is available as: bench_attrib.py phase --help "
            "(p_total / p_target / p_draft / pread / p_target_ex_io)."
        ),
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        metavar="PATH",
        help="log file(s), directory (recursive *.log), or glob pattern",
    )
    parser.add_argument("--out", help="write the fixed-column sessions TSV here")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the sessions + per-arm aggregation JSON to stdout",
    )
    parser.add_argument(
        "--arm-regex",
        default=DEFAULT_ARM_REGEX,
        help=f"arm regex on file/dir name (default: {DEFAULT_ARM_REGEX})",
    )
    args = parser.parse_args(args)

    paths = expand_logs(args.logs)
    if not paths:
        print(f"bench_attrib: no logs matched {args.logs!r}", file=sys.stderr)
    result = build_result(paths, args.arm_regex)

    if args.out:
        write_tsv(args.out, result["sessions"])

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif not args.out:
        # No --out and no --json: emit TSV on stdout for quick inspection.
        writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_FIELDS)
        for session in result["sessions"]:
            writer.writerow([_num(session.get(field)) for field in TSV_FIELDS])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
