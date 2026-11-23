#!/usr/bin/env python3
"""Parse ``LLAMA_FW_PROBE=1`` ``[fw]`` lines from llama-cli / llama-server logs.

The probe itself lives in ``src/llama-context.cpp`` (``fw_probe_guard``).  Every
``process_ubatch()`` call emits one line to stderr.  Two formats exist:

    # legacy (before FW_PROBE v2)
    [fw] ctx=T gtype=1 rows=4 ms=12.345 moe=1.234 readids=0.123 install=0.456 writeids=0.078 ...
    # FW_PROBE v2 (T-0369): the same prefix plus four appended fields
    [fw] ctx=T gtype=1 rows=4 ms=12.345 ... call=7 first=0 reused=1 splits=41

``ms`` is the host wall time of this ``process_ubatch`` (``ggml_time_us`` around
the call, optionally after ``ggml_backend_sched_synchronize``); ``moe`` /
``readids`` / ``install`` / ``writeids`` are per-forward counter deltas in
milliseconds.  They are *not* DSP time and must not be turned into t/s by this
tool.

Format contract handled here (see T-0369 spec and this repo's notes):

* The anchor is the first ``[fw] `` in the line; arbitrary prefixes (``llama-cli:``,
  timestamps, ...) are allowed.
* Fields are whitespace-separated ``key=value`` tokens.  ``ctx`` must be ``T`` or
  ``D``; ``rows`` and ``ms`` are required.  Missing optional fields are ``None``
  in JSON and are listed in ``missing_fields``; they are never invented as 0.
* ``splits=-1`` is a real value (the backend sched has no graph), distinct from a
  missing ``splits`` field.
* Unknown keys are kept verbatim (raw strings) in ``extra``; this preserves the
  legacy ``miss=`` / ``mib=`` / ``uniq=`` fields without guessing their types.
* Optional fields that are present but unparsable (for example ``first=2`` or
  ``moe=abc``) are treated like missing values (``None`` + recorded in
  ``missing_fields``).  Bad/irrelevant lines (no anchor, ``ctx=X``, missing or
  non-numeric ``rows``/``ms``) are counted as ``skipped`` and do not abort the run.

Grouping uses ``(ctx, first)``: ``first=1`` is the first ``process_ubatch`` of a
context, ``first=0`` is steady state.  Legacy rows have ``first=None`` and are
therefore not part of any group; ``--first`` / ``--steady`` drop them and report
``excluded_no_first``.  The call index in v2 belongs to one ``llama_context``
(T and D each start at 1); it is a ``process_ubatch`` call number, not a token
index.

Exit codes
----------
0  inputs parsed and outputs written (filtered/empty groups are not an error)
1  ``--strict`` was given and at least one line was skipped
2  an input cannot be read, an output cannot be written, or no line parsed at all
   (empty file / pure garbage file).  The failure is a one-line message, never a
   traceback.

Only the Python standard library is used; no device, model or network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys

SCHEMA = "fw-probe.v1"
FW_ANCHOR = "[fw] "

TSV_COLUMNS = (
    "file",
    "line",
    "ctx",
    "gtype",
    "rows",
    "ms",
    "moe",
    "readids",
    "install",
    "writeids",
    "call",
    "first",
    "reused",
    "splits",
)

# Fixed order of the per-call JSON object (and of missing_fields, which never
# contains the required ctx/rows/ms keys).
CALL_FIELDS = (
    "file",
    "line",
    "ctx",
    "gtype",
    "rows",
    "ms",
    "moe",
    "readids",
    "install",
    "writeids",
    "call",
    "first",
    "reused",
    "splits",
    "extra",
    "missing_fields",
)
OPTIONAL_FIELDS = (
    "gtype",
    "moe",
    "readids",
    "install",
    "writeids",
    "call",
    "first",
    "reused",
    "splits",
)
STAT_FIELDS = ("ms", "moe", "readids", "install", "writeids")

GROUP_KEYS = ("T/first", "T/steady", "D/first", "D/steady")

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _to_int(raw):
    """Parse a strict decimal integer, rejecting ``1.0``, ``1e3`` and ``1_0``."""
    if raw is None or not _INT_RE.match(raw.strip()):
        return None
    return int(raw, 10)


def _to_float(raw):
    """Parse a finite decimal/scientific float; reject nan/inf and underscores."""
    if raw is None:
        return None
    text = raw.strip()
    if not _FLOAT_RE.match(text):
        return None
    value = float(text)
    if not math.isfinite(value):
        return None
    return value


def _optional_value(name, raw):
    """Parse one optional field; return None when absent/invalid/out of range."""
    if raw is None:
        return None
    if name in ("moe", "readids", "install", "writeids"):
        return _to_float(raw)
    value = _to_int(raw)
    if value is None:
        return None
    if name in ("first", "reused") and value not in (0, 1):
        return None
    if name == "splits" and value < -1:
        return None
    return value


def parse_line(line, file, lineno):
    """Parse one log line into a call dict, or return ``None`` if it is skipped.

    ``line`` is a text line (a trailing newline is harmless).  ``file`` and
    ``lineno`` are copied into the result so multi-file reports keep provenance.
    """
    anchor = line.find(FW_ANCHOR)
    if anchor < 0:
        return None

    fields = {}
    extra = {}
    for token in line[anchor + len(FW_ANCHOR):].split():
        key, sep, value = token.partition("=")
        if not sep or not key:
            continue
        if key == "ctx" or key in OPTIONAL_FIELDS or key in ("rows", "ms"):
            fields[key] = value
        else:
            extra[key] = value

    ctx = fields.get("ctx")
    if ctx not in ("T", "D"):
        return None
    rows = _to_int(fields.get("rows"))
    if rows is None or rows < 0:
        return None
    ms = _to_float(fields.get("ms"))
    if ms is None:
        return None

    values = {}
    missing = []
    for name in OPTIONAL_FIELDS:
        raw = fields.get(name)
        value = _optional_value(name, raw)
        values[name] = value
        if value is None:
            missing.append(name)

    record = {
        "file": file,
        "line": lineno,
        "ctx": ctx,
        "gtype": values["gtype"],
        "rows": rows,
        "ms": ms,
        "moe": values["moe"],
        "readids": values["readids"],
        "install": values["install"],
        "writeids": values["writeids"],
        "call": values["call"],
        "first": values["first"],
        "reused": values["reused"],
        "splits": values["splits"],
        "extra": extra,
        "missing_fields": missing,
    }
    assert tuple(record) == CALL_FIELDS
    return record


def pctl(sorted_values, q):
    """Nearest-index percentile on a sorted sequence (T-0340 convention).

    ``values[round((n - 1) * q)]`` with Python's round-half-even, exactly like
    ``tools/moe/verify_io.py::_pctl``.  A single sample returns itself; at n=2
    p50 is the lower sample and p95 is the upper sample.
    """
    if not sorted_values:
        return None
    ordered = sorted(sorted_values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[min(len(ordered) - 1, max(0, idx))]


def _round(value):
    if value is None or isinstance(value, int):
        return value
    return round(value, 6)


def _stats(values):
    present = [value for value in values if value is not None]
    if not present:
        return {"p50": None, "p95": None, "mean": None}
    ordered = sorted(present)
    return {
        "p50": _round(pctl(ordered, 0.50)),
        "p95": _round(pctl(ordered, 0.95)),
        "mean": _round(sum(ordered) / len(ordered)),
    }


def _int_distribution(values):
    counts = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def _reused_rate(calls):
    known = [call["reused"] for call in calls if call["reused"] is not None]
    if not known:
        return None
    return round(sum(known) / len(known), 6)


def _call_range(calls):
    known = [call["call"] for call in calls if call["call"] is not None]
    if not known:
        return None, None
    return min(known), max(known)


def _group_stats(calls):
    call_min, call_max = _call_range(calls)
    return {
        "n": len(calls),
        "ms": _stats([call["ms"] for call in calls]),
        "moe": _stats([call["moe"] for call in calls]),
        "readids": _stats([call["readids"] for call in calls]),
        "install": _stats([call["install"] for call in calls]),
        "writeids": _stats([call["writeids"] for call in calls]),
        "rows": _int_distribution([call["rows"] for call in calls]),
        "splits": _int_distribution([call["splits"] for call in calls]),
        "reused_rate": _reused_rate(calls),
        "call_min": call_min,
        "call_max": call_max,
    }


def aggregate(calls):
    """Aggregate parsed calls into the four fixed ``ctx/first`` groups.

    Empty groups are still present, with ``n=0`` and ``null`` statistics, so a
    consumer never has to distinguish "missing key" from "no samples".  Rows
    whose ``first`` is ``None`` are not assigned to any group.
    """
    buckets = {key: [] for key in GROUP_KEYS}
    for call in calls:
        first = call.get("first")
        if first not in (0, 1):
            continue
        key = "%s/%s" % (call["ctx"], "first" if first == 1 else "steady")
        if key in buckets:
            buckets[key].append(call)
    return {key: _group_stats(buckets[key]) for key in GROUP_KEYS}


def scan_file(path):
    """Read one log file and return ``(input_info, calls)``; raises ``OSError``.

    Hashing and parsing share one binary read pass.  Line numbers and the line
    count are 1-based / newline-based: a trailing newline does not create an
    extra empty line.
    """
    digest = hashlib.sha256()
    calls = []
    lines = 0
    skipped = 0
    with open(path, "rb") as handle:
        for raw in handle:
            lines += 1
            digest.update(raw)
            text = raw.decode("utf-8", "replace").rstrip("\r\n")
            record = parse_line(text, path, lines)
            if record is None:
                skipped += 1
            else:
                calls.append(record)
    info = {
        "path": path,
        "sha256": digest.hexdigest(),
        "lines": lines,
        "parsed": len(calls),
        "skipped": skipped,
    }
    return info, calls


def filter_calls(calls, ctx=None, first_mode=None, min_ms=None):
    """Apply --ctx / --first / --steady / --min-ms filters.

    ``excluded_no_first`` counts rows that pass the ctx and min-ms filters but
    have no ``first`` value, and would otherwise have been selected.
    """
    selected = []
    excluded_no_first = 0
    for call in calls:
        if ctx is not None and call["ctx"] != ctx:
            continue
        if min_ms is not None and call["ms"] < min_ms:
            continue
        if first_mode is not None:
            first = call["first"]
            if first is None:
                excluded_no_first += 1
                continue
            if first != first_mode:
                continue
        selected.append(call)
    return selected, excluded_no_first


def build_document(infos, selected, excluded_no_first):
    totals = {
        "lines": sum(info["lines"] for info in infos),
        "parsed": sum(info["parsed"] for info in infos),
        "skipped": sum(info["skipped"] for info in infos),
        "excluded_no_first": excluded_no_first,
    }
    return {
        "schema": SCHEMA,
        "inputs": infos,
        "calls": selected,
        "groups": aggregate(selected),
        "totals": totals,
    }


def _tsv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def write_json(path, document):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_tsv(path, calls):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for call in calls:
            writer.writerow([_tsv_value(call[column]) for column in TSV_COLUMNS])


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return ("%." + str(digits) + "f") % value


def _group_line(key, group):
    ms = group["ms"]
    moe = group["moe"]
    readids = group["readids"]
    install = group["install"]
    writeids = group["writeids"]
    calls = "-"
    if group["call_min"] is not None:
        calls = "%s..%s" % (group["call_min"], group["call_max"])
    return (
        "%-8s n=%-3d ms[p50=%s p95=%s mean=%s] "
        "moe[p50=%s] readids[p50=%s] install[p50=%s] writeids[p50=%s] "
        "rows=%s splits=%s reused=%s calls=%s"
        % (
            key,
            group["n"],
            _fmt(ms["p50"]),
            _fmt(ms["p95"]),
            _fmt(ms["mean"]),
            _fmt(moe["p50"]),
            _fmt(readids["p50"]),
            _fmt(install["p50"]),
            _fmt(writeids["p50"]),
            group["rows"] or "{}",
            group["splits"] or "{}",
            _fmt(group["reused_rate"]),
            calls,
        )
    )


def print_groups(groups):
    for key in GROUP_KEYS:
        print(_group_line(key, groups[key]))


def print_default(document):
    totals = document["totals"]
    print(
        "fw-probe: files=%d lines=%d parsed=%d skipped=%d excluded_no_first=%d"
        % (
            len(document["inputs"]),
            totals["lines"],
            totals["parsed"],
            totals["skipped"],
            totals["excluded_no_first"],
        )
    )
    for info in document["inputs"]:
        print(
            "  input %s: lines=%d parsed=%d skipped=%d sha256=%s"
            % (
                info["path"],
                info["lines"],
                info["parsed"],
                info["skipped"],
                info["sha256"][:12],
            )
        )
    print_groups(document["groups"])


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Parse LLAMA_FW_PROBE [fw] lines, emit per-call TSV/JSON and "
            "ctx x first/steady aggregates.  Never touches a device."
        )
    )
    parser.add_argument("logs", nargs="+", metavar="LOG", help="llama-cli/server stderr log")
    parser.add_argument("--json", metavar="OUT", help="write fw-probe.v1 JSON")
    parser.add_argument("--tsv", metavar="OUT", help="write per-call TSV")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print only the four group lines (default: inputs + totals + groups)",
    )
    parser.add_argument("--ctx", choices=("T", "D"), help="keep only target or draft rows")
    first_group = parser.add_mutually_exclusive_group()
    first_group.add_argument("--first", action="store_true", help="keep only first=1 rows")
    first_group.add_argument("--steady", action="store_true", help="keep only first=0 rows")
    parser.add_argument("--min-ms", type=float, metavar="X", help="keep rows with ms >= X")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any line was skipped (default: skipped lines are tolerated)",
    )
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.min_ms is not None and not math.isfinite(args.min_ms):
        parser.error("--min-ms must be a finite number")

    infos = []
    calls = []
    read_failed = False
    for path in args.logs:
        try:
            info, file_calls = scan_file(path)
        except OSError as exc:
            print("fw-probe: error: cannot read %s: %s" % (path, exc), file=sys.stderr)
            read_failed = True
            continue
        infos.append(info)
        calls.extend(file_calls)
    if read_failed:
        return 2

    first_mode = None
    if args.first:
        first_mode = 1
    elif args.steady:
        first_mode = 0
    selected, excluded_no_first = filter_calls(
        calls, ctx=args.ctx, first_mode=first_mode, min_ms=args.min_ms
    )
    document = build_document(infos, selected, excluded_no_first)

    try:
        if args.json:
            write_json(args.json, document)
        if args.tsv:
            write_tsv(args.tsv, selected)
    except OSError as exc:
        print("fw-probe: error: cannot write output: %s" % exc, file=sys.stderr)
        return 2

    if args.summary:
        print_groups(document["groups"])
    else:
        print_default(document)

    totals = document["totals"]
    if totals["parsed"] == 0:
        print(
            "fw-probe: error: no [fw] line parsed from %d input(s)" % len(infos),
            file=sys.stderr,
        )
        return 2
    if args.strict and totals["skipped"]:
        print(
            "fw-probe: error: %d line(s) skipped under --strict" % totals["skipped"],
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
