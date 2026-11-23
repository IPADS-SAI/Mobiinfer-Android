#!/usr/bin/env python3
"""opbuf_stats.py — summarize GGML_HEXAGON_OPBUFSTAT=1 JSONL (T-0304).

The host prints one pure-JSON line per op batch on stderr:

  {"seq":..,"n_bufs":..,"reuse":..,"drop_n":..,"drop_bytes":..,"map_n":..,
   "map_bytes":..,"m_vmem":..,"e_vmem":..,"max_vmem":..,"defrag":..,
   "scan_cmp":..}

and one aggregate line at process exit:

  {"opbuf_total":{"batches":..,"n_bufs":..,...}}

Usage:
  python3 bench/opbuf_stats.py run.err
  python3 bench/opbuf_stats.py run.err --bucket-by n_bufs --bucket-width 8
  python3 bench/opbuf_stats.py run.err --json summary.json
  python3 bench/opbuf_stats.py --self-test

Batch mode (T-0332; read-only, the single-file CLI above is unchanged):

  python3 bench/opbuf_stats.py --runs-dir runs/ \
    --glob '*-opbuf.jsonl' --arm-map arms.tsv \
    --out-tsv opbuf-arms.tsv --out-json opbuf-arms.json

  --runs-dir DIR   directory scanned (non-recursive)
  --glob PATTERN   filename glob, default '*-opbuf.jsonl'
  --arm-map FILE   TSV `pattern<TAB>arm` or JSON (list of
                   {"pattern":..,"arm":..}, list of [pattern, arm], or an
                   object {pattern: arm}); the first match wins, patterns
                   are regexes (literal substring if the regex is invalid),
                   and an unmatched tag goes to arm `unknown` with a warning
  --out-tsv FILE   per-arm TSV (plain header row, no comment prefix)
  --out-json FILE  per-arm + per-run JSON
  --strict         also exit 1 for a missing/mismatched opbuf_total or
                   malformed lines (default: single-file exit semantics)

Tag = file name with `-opbuf.jsonl` stripped (then a plain `.jsonl`).

Aggregation (no "median of medians" ambiguity, no cross-arm pooling):
  * every file is parsed independently with parse_stream(); the per-run
    summarize() keeps records/total/malformed and the conservation check;
  * per arm, `median_of_medians` is the median of the per-run p50/p95/max
    values (one vote per run, so one long run cannot dominate);
  * `pooled` p50/p95 sorts every batch of every run *of that arm* together;
  * drop/map totals are the sum of the per-run opbuf_total values; a run
    without a usable total contributes the sum of its own batches and is
    marked `derived` (TSV `drop_map_total_source` is exact|derived, with
    `drop_map_derived_runs` counting how many runs needed derivation);
  * exit code: 0 if every run has records and passes conservation, 1
    otherwise -- exactly the single-file rule.  Missing/mismatched total,
    malformed lines and unknown arm mappings are printed to stderr and
    counted; empty files always fail (single-file parity), and --strict
    additionally fails on missing/mismatched total and malformed lines.
    An unmatched tag becomes `unknown`, never merged into a real arm, and
    does not by itself fail the batch.

Only the Python standard library is used.

Semantics of the counters (same as htp/htp-opbuf-stats.h):
  n_bufs      buffers in the batch
  reuse       buffers served by an existing mapping (pass 0)
  drop_n      htp_munmap() calls, drop_bytes their total size
  map_n       successful htp_mmap() calls, map_bytes their total size
  m_vmem      pass-1 sum of non-extended mapped bytes before drops
  e_vmem      pass-0 sum of non-reused, non-extended buffer bytes
  max_vmem    ctx->max_vmem used by the drop decision
  defrag      defragmentation passes entered (0 or 1)
  scan_cmp    mmap-table entries examined by reuse_buf()/mmap_buf()

Conservation law (checked per record and for the aggregate):
  defrag == 0:  n_bufs == reuse + map_n
  defrag > 0:   n_bufs >= reuse and map_n >= n_bufs - reuse
A defrag pass remaps non-extended buffers that pass 1 already mapped, so
map_n can exceed n_bufs - reuse there.

Percentiles use linear interpolation between closest ranks (numpy default):
rank = p/100 * (n - 1); result = v[floor(rank)] + frac * (v[ceil(rank)] - v[floor]).
For the 100-value self-test set 1..50 (each twice) this gives p50 = 25.5,
p95 = 48.0, max = 50.0 by hand.
"""

import argparse
import contextlib
import glob as globmod
import io
import json
import os
import re
import sys
import tempfile
from collections import OrderedDict

RECORD_FIELDS = (
    "n_bufs",
    "reuse",
    "drop_n",
    "drop_bytes",
    "map_n",
    "map_bytes",
    "m_vmem",
    "e_vmem",
    "max_vmem",
    "defrag",
    "scan_cmp",
)

# Fields whose aggregate in the host total line is a sum.  max_vmem is a
# per-session constant, so the host reports the maximum observed instead.
SUM_FIELDS = tuple(f for f in RECORD_FIELDS if f != "max_vmem")

# Batch mode (T-0332).
BATCH_SCHEMA = "opbuf-arms v1"
DEFAULT_GLOB = "*-opbuf.jsonl"
DROP_MAP_FIELDS = ("drop_n", "drop_bytes", "map_n", "map_bytes")
ARM_BASE_COLUMNS = (
    "arm",
    "runs",
    "batches",
    "conservation_fail_runs",
    "missing_total_runs",
    "total_mismatch_runs",
    "malformed_lines",
    "empty_runs",
    "failed_runs",
)


def percentile(sorted_values, p):
    """Linear-interpolation percentile over an ascending list."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_values[lo]) + frac * (sorted_values[hi] - sorted_values[lo])


def conserved(rec):
    """Counter conservation law; see the module docstring."""
    n = int(rec["n_bufs"])
    reuse = int(rec["reuse"])
    map_n = int(rec["map_n"])
    defrag = int(rec["defrag"])
    if n < reuse:
        return False
    if defrag == 0:
        return n == reuse + map_n
    return map_n >= n - reuse


def parse_stream(stream):
    """Parse mixed stderr text; returns (records, total_line, ignored, malformed)."""
    records = []
    total = None
    ignored = 0
    malformed = 0
    for line in stream:
        line = line.strip()
        if not line:
            continue
        obj = None
        try:
            obj = json.loads(line)
        except ValueError:
            # Tolerate a prefix such as "ggml-hex: ..." before the JSON object;
            # lines that are not JSON at all are ordinary host logs.
            start = line.find("{")
            if start < 0:
                ignored += 1
                continue
            try:
                obj = json.loads(line[start:])
            except ValueError:
                obj = None
            if obj is None:
                malformed += 1
                continue
        if not isinstance(obj, dict):
            malformed += 1
            continue
        if "opbuf_total" in obj:
            total = obj["opbuf_total"]
            continue
        missing = [f for f in RECORD_FIELDS if f not in obj]
        if "seq" not in obj or missing:
            malformed += 1
            continue
        records.append(obj)
    if total is not None and not isinstance(total, dict):
        total = None
    return records, total, ignored, malformed


def _field_stats(values):
    ordered = sorted(values)
    return OrderedDict(
        (
            ("p50", percentile(ordered, 50)),
            ("p95", percentile(ordered, 95)),
            ("max", float(ordered[-1]) if ordered else None),
            ("total", sum(values)),
        )
    )


def _bucket_key(rec, bucket_by, width):
    value = int(rec[bucket_by])
    if width <= 0:
        return str(value), value
    lo = (value // width) * width
    return "%d-%d" % (lo, lo + width - 1), value


def summarize(records, total_line=None, bucket_by="n_bufs", bucket_width=0):
    fields = OrderedDict()
    for field in RECORD_FIELDS:
        fields[field] = _field_stats([int(r[field]) for r in records])

    buckets = OrderedDict()
    for rec in records:
        label, _ = _bucket_key(rec, bucket_by, bucket_width)
        buckets.setdefault(label, []).append(rec)

    bucket_rows = []
    for label in sorted(buckets, key=lambda s: int(s.split("-")[0])):
        group = buckets[label]
        row = OrderedDict()
        row["bucket"] = label
        row["batches"] = len(group)
        for field in ("scan_cmp", "m_vmem", "e_vmem"):
            row[field + "_p50"] = percentile(sorted(int(r[field]) for r in group), 50)
            row[field + "_p95"] = percentile(sorted(int(r[field]) for r in group), 95)
            row[field + "_max"] = float(max(int(r[field]) for r in group))
        row["drop_n"] = sum(int(r["drop_n"]) for r in group)
        row["drop_bytes"] = sum(int(r["drop_bytes"]) for r in group)
        row["map_n"] = sum(int(r["map_n"]) for r in group)
        row["map_bytes"] = sum(int(r["map_bytes"]) for r in group)
        row["defrag"] = sum(int(r["defrag"]) for r in group)
        bucket_rows.append(row)

    violations = [(i, r) for i, r in enumerate(records) if not conserved(r)]

    total_check = None
    if total_line is not None:
        expected = OrderedDict()
        expected["batches"] = len(records)
        for field in SUM_FIELDS:
            expected[field] = sum(int(r[field]) for r in records)
        expected["max_vmem"] = max((int(r["max_vmem"]) for r in records), default=0)
        diffs = OrderedDict()
        for key, want in expected.items():
            got = total_line.get(key)
            if got is None or int(got) != int(want):
                diffs[key] = {"got": got, "want": want}
        total_check = OrderedDict(
            (
                ("ok", not diffs),
                ("batches", total_line.get("batches")),
                ("expected_batches", len(records)),
                ("diffs", diffs),
            )
        )

    return OrderedDict(
        (
            ("batches", len(records)),
            ("fields", fields),
            ("buckets", bucket_rows),
            ("conservation", OrderedDict(
                (
                    ("ok", not violations),
                    ("checked", len(records)),
                    ("violations", len(violations)),
                    ("first", [
                        {"index": i, "seq": r.get("seq"), "n_bufs": r["n_bufs"],
                         "reuse": r["reuse"], "map_n": r["map_n"], "defrag": r["defrag"]}
                        for i, r in violations[:5]
                    ]),
                )
            )),
            ("total_line", total_check),
        )
    )


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return ("%.2f" % value).rstrip("0").rstrip(".")
    return str(value)


def print_report(summary, out=sys.stdout):
    write = out.write
    write("opbuf stats: %d batches\n" % summary["batches"])
    write("%-12s %12s %12s %12s %14s\n" % ("field", "p50", "p95", "max", "total"))
    for field, st in summary["fields"].items():
        write("%-12s %12s %12s %12s %14s\n" % (
            field, _fmt(st["p50"]), _fmt(st["p95"]), _fmt(st["max"]), _fmt(st["total"])))

    drop_bytes = summary["fields"]["drop_bytes"]["total"]
    map_bytes = summary["fields"]["map_bytes"]["total"]
    drop_n = summary["fields"]["drop_n"]["total"]
    map_n = summary["fields"]["map_n"]["total"]
    write("drop/map totals: drop_n=%s drop_bytes=%s map_n=%s map_bytes=%s\n"
          % (_fmt(drop_n), _fmt(drop_bytes), _fmt(map_n), _fmt(map_bytes)))

    cons = summary["conservation"]
    write("conservation: %s (%d records checked, %d violations)\n"
          % ("OK" if cons["ok"] else "FAIL", cons["checked"], cons["violations"]))
    for item in cons["first"]:
        write("  violation: seq=%s n_bufs=%s reuse=%s map_n=%s defrag=%s\n"
              % (item["seq"], item["n_bufs"], item["reuse"], item["map_n"], item["defrag"]))

    total = summary["total_line"]
    if total is None:
        write("total line: absent (stream truncated or process killed)\n")
    else:
        write("total line: %s (batches %s, expected %s)\n"
              % ("OK" if total["ok"] else "FAIL",
                 total["batches"], total["expected_batches"]))
        for key, diff in total["diffs"].items():
            write("  diff %s: got %s want %s\n" % (key, diff["got"], diff["want"]))

    if summary["buckets"]:
        write("\nper-bucket (grouped by n_bufs bucket):\n")
        write("%-10s %7s %10s %10s %10s %8s %12s %8s %12s\n" % (
            "bucket", "batches", "scan_p50", "scan_p95", "scan_max",
            "drop_n", "drop_bytes", "map_n", "map_bytes"))
        for row in summary["buckets"]:
            write("%-10s %7d %10s %10s %10s %8s %12s %8s %12s\n" % (
                row["bucket"], row["batches"], _fmt(row["scan_cmp_p50"]),
                _fmt(row["scan_cmp_p95"]), _fmt(row["scan_cmp_max"]),
                _fmt(row["drop_n"]), _fmt(row["drop_bytes"]),
                _fmt(row["map_n"]), _fmt(row["map_bytes"])))


# ---------------------------------------------------------------------------
# batch mode (T-0332)
# ---------------------------------------------------------------------------

def _split_arm_line(line):
    """Split one `pattern<TAB>arm` line (spaces are tolerated as a fallback)."""
    parts = line.split("\t")
    if len(parts) < 2:
        parts = line.split(None, 1)
    if len(parts) < 2:
        raise ValueError("expected two columns: %r" % line)
    pattern = parts[0].strip()
    arm = parts[1].strip()
    if not pattern or not arm:
        raise ValueError("empty pattern/arm in %r" % line)
    if any(p.strip() for p in parts[2:]):
        raise ValueError("more than two columns: %r" % line)
    return pattern, arm


def load_arm_map(path):
    """Load `[(pattern, arm), ...]` from TSV or JSON; order is significant."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if path.endswith(".json") or stripped.startswith(("[", "{")):
        try:
            obj = json.loads(text)
        except ValueError as exc:
            raise ValueError("invalid JSON arm map %s: %s" % (path, exc))
        entries = []
        if isinstance(obj, dict):
            if "pattern" in obj and "arm" in obj:
                entries = [(obj["pattern"], obj["arm"])]
            else:
                entries = list(obj.items())
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    if "pattern" not in item or "arm" not in item:
                        raise ValueError("arm-map object needs pattern+arm: %r" % (item,))
                    entries.append((item["pattern"], item["arm"]))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    entries.append((item[0], item[1]))
                else:
                    raise ValueError("bad arm-map entry: %r" % (item,))
        else:
            raise ValueError("arm map must be a JSON object, array or TSV")
        result = []
        for pattern, arm in entries:
            if not isinstance(pattern, str) or not isinstance(arm, str):
                raise ValueError("pattern/arm must be strings: %r" % ((pattern, arm),))
            if not pattern or not arm:
                raise ValueError("empty pattern/arm in arm map")
            result.append((pattern, arm))
        if not result:
            raise ValueError("arm map %s is empty" % path)
        return result

    result = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            result.append(_split_arm_line(line))
        except ValueError as exc:
            raise ValueError("%s:%d: %s" % (path, lineno, exc))
    if not result:
        raise ValueError("arm map %s is empty" % path)
    return result


def _match_arm(tag, arm_map):
    """First matching pattern wins; regex, with literal-substring fallback."""
    for pattern, arm in arm_map:
        try:
            if re.search(pattern, tag):
                return arm
        except re.error:
            if pattern in tag:
                return arm
    return "unknown"


def _tag_for(filename):
    """`A-r1-opbuf.jsonl` -> `A-r1`; a plain `.jsonl` suffix is also stripped."""
    if filename.endswith("-opbuf.jsonl"):
        return filename[: -len("-opbuf.jsonl")]
    if filename.endswith(".jsonl"):
        return filename[: -len(".jsonl")]
    return filename


def _run_drop_map(total, records, total_ok=None):
    """Return (totals, source) for drop/map; source is exact or derived.

    A total line that failed summarize()'s check is not used: the per-batch
    sum is trusted instead and the run is marked derived.
    """
    derived = {k: sum(int(r[k]) for r in records) for k in DROP_MAP_FIELDS}
    if total_ok is False or not isinstance(total, dict):
        return derived, "derived"
    if not all(k in total for k in DROP_MAP_FIELDS):
        return derived, "derived"
    try:
        exact = {k: int(total[k]) for k in DROP_MAP_FIELDS}
    except (TypeError, ValueError):
        return derived, "derived"
    return exact, "exact"


def _process_run_file(path, tag, arm, bucket_by, bucket_width):
    """Parse one file; keeps per-run stats plus a list of warning dicts."""
    warnings = []
    row = OrderedDict((
        ("tag", tag),
        ("path", path),
        ("arm", arm),
        ("batches", 0),
        ("malformed_lines", 0),
        ("ignored_lines", 0),
        ("conservation_ok", False),
        ("conservation_violations", 0),
        ("total_present", False),
        ("total_ok", None),
        ("total", None),
        ("empty", False),
        ("failed", False),
        ("strict_failed", False),
        ("failures", []),
        ("fields", None),
        ("drop_map_totals", {k: 0 for k in DROP_MAP_FIELDS}),
        ("drop_map_source", "derived"),
    ))
    values = {}
    read_error = None
    analysis_error = None
    records = []
    total = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            records, total, ignored, malformed = parse_stream(f)
    except OSError as exc:
        read_error = str(exc)
        ignored = 0
        malformed = 0

    row["malformed_lines"] = malformed
    row["ignored_lines"] = ignored
    row["total_present"] = isinstance(total, dict)
    row["total"] = total

    summary = None
    if read_error is None:
        try:
            summary = summarize(records, total, bucket_by, bucket_width)
        except (KeyError, TypeError, ValueError) as exc:
            analysis_error = "%s: %s" % (type(exc).__name__, exc)

    if summary is not None:
        row["batches"] = summary["batches"]
        row["conservation_ok"] = summary["conservation"]["ok"]
        row["conservation_violations"] = summary["conservation"]["violations"]
        row["fields"] = summary["fields"]
        values = {f: [int(r[f]) for r in records] for f in RECORD_FIELDS}
        if row["total_present"]:
            row["total_ok"] = summary["total_line"]["ok"]
        row["drop_map_totals"], row["drop_map_source"] = _run_drop_map(
            total, records, row["total_ok"])
    else:
        row["drop_map_totals"] = {k: 0 for k in DROP_MAP_FIELDS}

    row["empty"] = read_error is None and analysis_error is None and not records
    row["failed"] = bool(read_error or analysis_error or row["batches"] == 0
                         or not row["conservation_ok"])

    def add(reason, detail):
        row["failures"].append(reason)
        warnings.append(OrderedDict((
            ("path", path), ("tag", tag), ("arm", arm),
            ("reason", reason), ("detail", detail),
        )))

    if read_error:
        add("read_error", read_error)
    elif analysis_error:
        add("analysis_error", analysis_error)
    elif not records:
        add("empty", "no JSONL records (malformed=%d)" % malformed)
    else:
        if not row["conservation_ok"]:
            first = summary["conservation"]["first"]
            detail = "%d violation(s)" % row["conservation_violations"]
            if first:
                v = first[0]
                detail += "; first seq=%s n_bufs=%s reuse=%s map_n=%s defrag=%s" % (
                    v["seq"], v["n_bufs"], v["reuse"], v["map_n"], v["defrag"])
            add("conservation", detail)
        if not row["total_present"]:
            add("missing_total",
                "no opbuf_total line; drop/map totals derived from %d batches"
                % row["batches"])
        elif row["total_ok"] is False:
            diffs = summary["total_line"]["diffs"]
            add("total_mismatch", "opbuf_total mismatch in: %s"
                % (", ".join(diffs) if diffs else "unknown"))
        if malformed:
            add("malformed", "%d malformed line(s)" % malformed)

    row["strict_failed"] = bool(
        row["failed"] or malformed or (row["batches"] > 0 and not row["total_present"])
        or row["total_ok"] is False)
    row["_values"] = values
    row["_warnings"] = warnings
    return row


def _median(values):
    ordered = sorted(v for v in values if v is not None)
    return percentile(ordered, 50) if ordered else None


def _aggregate_arm(arm, rows):
    fields = OrderedDict()
    for field in RECORD_FIELDS:
        with_data = [r for r in rows if r["batches"] > 0 and r["fields"] is not None]
        mom = OrderedDict()
        for stat in ("p50", "p95", "max"):
            mom[stat] = _median(r["fields"][field][stat] for r in with_data)
        pooled_values = []
        for r in rows:
            pooled_values.extend(r["_values"].get(field, ()))
        pooled_values.sort()
        fields[field] = OrderedDict((
            ("median_of_medians", mom),
            ("pooled", OrderedDict((
                ("p50", percentile(pooled_values, 50)),
                ("p95", percentile(pooled_values, 95)),
            ))),
        ))

    totals = OrderedDict((k, sum(r["drop_map_totals"][k] for r in rows))
                         for k in DROP_MAP_FIELDS)
    sources = OrderedDict((r["tag"], r["drop_map_source"])
                          for r in rows if r["batches"] > 0)
    derived_runs = sum(1 for value in sources.values() if value == "derived")
    # Any derived component makes the arm total derived; the per-run map and
    # derived_runs keep the exact/derived split visible.
    source = "derived" if derived_runs else ("exact" if sources else None)
    totals["source"] = source
    totals["derived_runs"] = derived_runs
    totals["per_run_source"] = sources

    failed_runs = sum(1 for r in rows if r["failed"])
    return OrderedDict((
        ("arm", arm),
        ("runs", len(rows)),
        ("batches", sum(r["batches"] for r in rows)),
        ("conservation_fail_runs",
         sum(1 for r in rows if r["batches"] > 0 and not r["conservation_ok"])),
        ("missing_total_runs",
         sum(1 for r in rows if r["batches"] > 0 and not r["total_present"])),
        ("total_mismatch_runs", sum(1 for r in rows if r["total_ok"] is False)),
        ("malformed_lines", sum(r["malformed_lines"] for r in rows)),
        ("empty_runs", sum(1 for r in rows if r["empty"])),
        ("failed_runs", failed_runs),
        ("fields", fields),
        ("drop_map_totals", totals),
    ))


def run_batch(runs_dir, glob_pattern=DEFAULT_GLOB, arm_map=None, bucket_by="n_bufs",
              bucket_width=0, strict=False, arm_map_path=None):
    """Scan `runs_dir`, summarize every file and aggregate per arm.

    Returns (summary, warnings).  `summary["ok"]` is the exit-code verdict;
    warnings are dicts with path/tag/arm/reason/detail and are never silent.
    Files are opened read-only.
    """
    if arm_map is None:
        arm_map = []
    if not os.path.isdir(runs_dir):
        raise ValueError("--runs-dir is not a directory: %s" % runs_dir)
    pattern = glob_pattern or DEFAULT_GLOB
    if os.path.isabs(pattern):
        paths = globmod.glob(pattern)
    else:
        paths = globmod.glob(os.path.join(runs_dir, pattern))
    paths = sorted((p for p in paths if os.path.isfile(p)),
                   key=lambda p: (os.path.basename(p), p))

    rows = []
    warnings = []
    for path in paths:
        filename = os.path.basename(path)
        tag = _tag_for(filename)
        arm = _match_arm(tag, arm_map)
        if arm == "unknown":
            warnings.append(OrderedDict((
                ("path", path), ("tag", tag), ("arm", arm),
                ("reason", "unknown_arm"),
                ("detail", "tag %r matched no arm-map pattern (using 'unknown')" % tag),
            )))
        row = _process_run_file(path, tag, arm, bucket_by, bucket_width)
        rows.append(row)
        warnings.extend(row["_warnings"])

    by_arm = OrderedDict()
    if strict:
        # `failed` is the effective verdict for this invocation; strict_failed
        # keeps the always-on health view for JSON consumers.
        for row in rows:
            row["failed"] = row["strict_failed"]
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    order = sorted(a for a in by_arm if a != "unknown")
    if "unknown" in by_arm:
        order.append("unknown")
    arms = [_aggregate_arm(arm, by_arm[arm]) for arm in order]

    failed_runs = sum(a["failed_runs"] for a in arms)
    error = None
    if not paths:
        error = "no files matched %s in %s" % (pattern, runs_dir)
    summary = OrderedDict((
        ("schema", BATCH_SCHEMA),
        ("runs_dir", runs_dir),
        ("glob", pattern),
        ("arm_map", arm_map_path),
        ("strict", bool(strict)),
        ("ok", error is None and failed_runs == 0),
        ("error", error),
        ("failed_runs", failed_runs),
        ("unknown_runs", len(by_arm.get("unknown", ()))),
        ("warnings", warnings),
        ("runs", rows),
        ("arms", arms),
    ))
    for row in rows:
        row.pop("_values", None)
        row.pop("_warnings", None)
    return summary, warnings


def arm_tsv_columns():
    columns = list(ARM_BASE_COLUMNS)
    for field in RECORD_FIELDS:
        columns.extend((
            field + "_median_of_medians_p50",
            field + "_median_of_medians_p95",
            field + "_median_of_medians_max",
            field + "_pooled_p50", field + "_pooled_p95",
        ))
    columns.extend((
        "drop_n_total", "drop_bytes_total", "map_n_total", "map_bytes_total",
        "drop_map_total_source", "drop_map_derived_runs",
    ))
    return columns


def write_arm_tsv(summary, path):
    columns = arm_tsv_columns()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(columns) + "\n")
        for arm in summary["arms"]:
            values = {
                "arm": arm["arm"], "runs": arm["runs"], "batches": arm["batches"],
                "conservation_fail_runs": arm["conservation_fail_runs"],
                "missing_total_runs": arm["missing_total_runs"],
                "total_mismatch_runs": arm["total_mismatch_runs"],
                "malformed_lines": arm["malformed_lines"],
                "empty_runs": arm["empty_runs"],
                "failed_runs": arm["failed_runs"],
                "drop_n_total": arm["drop_map_totals"]["drop_n"],
                "drop_bytes_total": arm["drop_map_totals"]["drop_bytes"],
                "map_n_total": arm["drop_map_totals"]["map_n"],
                "map_bytes_total": arm["drop_map_totals"]["map_bytes"],
                "drop_map_total_source": arm["drop_map_totals"]["source"],
                "drop_map_derived_runs": arm["drop_map_totals"]["derived_runs"],
            }
            for field, stats in arm["fields"].items():
                mom = stats["median_of_medians"]
                pooled = stats["pooled"]
                values[field + "_median_of_medians_p50"] = mom["p50"]
                values[field + "_median_of_medians_p95"] = mom["p95"]
                values[field + "_median_of_medians_max"] = mom["max"]
                values[field + "_pooled_p50"] = pooled["p50"]
                values[field + "_pooled_p95"] = pooled["p95"]
            f.write("\t".join(_fmt(values[c]) for c in columns) + "\n")


def write_arm_json(summary, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, sort_keys=False)
        f.write("\n")


def print_arm_report(summary, out=None):
    if out is None:
        out = sys.stdout
    write = out.write
    write("opbuf arms: %d arm(s), %d run(s), %d failed, %d warning(s)%s\n" % (
        len(summary["arms"]), len(summary["runs"]), summary["failed_runs"],
        len(summary["warnings"]), " [strict]" if summary["strict"] else ""))
    if summary["error"]:
        write("error: %s\n" % summary["error"])
    write("%-10s %5s %8s %9s %9s %8s %11s %16s %s\n" % (
        "arm", "runs", "batches", "cons_fail", "no_total", "malformed",
        "failed", "drop_n_total", "total_source"))
    for arm in summary["arms"]:
        write("%-10s %5d %8d %9d %9d %8d %11d %16s %s\n" % (
            arm["arm"], arm["runs"], arm["batches"],
            arm["conservation_fail_runs"], arm["missing_total_runs"],
            arm["malformed_lines"], arm["failed_runs"],
            _fmt(arm["drop_map_totals"]["drop_n"]),
            _fmt(arm["drop_map_totals"]["source"])))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _record(seq, n_bufs, reuse, map_n, scan_cmp, defrag=0, drop_n=0, drop_bytes=0,
            map_bytes=0, m_vmem=0, e_vmem=0, max_vmem=1000):
    return {
        "seq": seq,
        "n_bufs": n_bufs,
        "reuse": reuse,
        "drop_n": drop_n,
        "drop_bytes": drop_bytes,
        "map_n": map_n,
        "map_bytes": map_bytes,
        "m_vmem": m_vmem,
        "e_vmem": e_vmem,
        "max_vmem": max_vmem,
        "defrag": defrag,
        "scan_cmp": scan_cmp,
    }


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append("%s: got %r want %r" % (name, got, want))

    def check_close(name, got, want):
        if got is None or abs(float(got) - float(want)) > 1e-9:
            failures.append("%s: got %r want %r" % (name, got, want))

    # 20 synthetic records: all-reuse path, n_bufs cycles 1..7.
    recs20 = []
    for i in range(20):
        n = (i % 7) + 1
        recs20.append(_record(i + 1, n, n, 0, 64 * n))
    s20 = summarize(recs20)
    check("20.batches", s20["batches"], 20)
    check("20.conservation", s20["conservation"]["ok"], True)
    check("20.n_bufs.total", s20["fields"]["n_bufs"]["total"], 77)
    check("20.reuse.total", s20["fields"]["reuse"]["total"], 77)
    check("20.map_n.total", s20["fields"]["map_n"]["total"], 0)
    check("20.drop_n.total", s20["fields"]["drop_n"]["total"], 0)
    check_close("20.n_bufs.p50", s20["fields"]["n_bufs"]["p50"], 4.0)
    check_close("20.n_bufs.p95", s20["fields"]["n_bufs"]["p95"], 7.0)
    check_close("20.scan_cmp.p50", s20["fields"]["scan_cmp"]["p50"], 256.0)
    check_close("20.scan_cmp.p95", s20["fields"]["scan_cmp"]["p95"], 448.0)

    total20 = {"batches": 20}
    for field in SUM_FIELDS:
        total20[field] = sum(int(r[field]) for r in recs20)
    total20["max_vmem"] = max(int(r["max_vmem"]) for r in recs20)
    s20t = summarize(recs20, total20)
    check("20.total_line.ok", s20t["total_line"]["ok"], True)

    # 100 synthetic records: n_bufs = 1..50 twice, all reuse.
    recs100 = []
    for i in range(100):
        n = (i % 50) + 1
        recs100.append(_record(i + 1, n, n, 0, 64 * n))
    s100 = summarize(recs100)
    check("100.batches", s100["batches"], 100)
    check("100.conservation", s100["conservation"]["ok"], True)
    check("100.n_bufs.total", s100["fields"]["n_bufs"]["total"], 2550)
    check_close("100.n_bufs.p50", s100["fields"]["n_bufs"]["p50"], 25.5)
    check_close("100.n_bufs.p95", s100["fields"]["n_bufs"]["p95"], 48.0)
    check_close("100.n_bufs.max", s100["fields"]["n_bufs"]["max"], 50.0)
    check_close("100.scan_cmp.p50", s100["fields"]["scan_cmp"]["p50"], 1632.0)
    check_close("100.scan_cmp.p95", s100["fields"]["scan_cmp"]["p95"], 3072.0)
    check_close("100.scan_cmp.max", s100["fields"]["scan_cmp"]["max"], 3200.0)

    total100 = {"batches": 100}
    for field in SUM_FIELDS:
        total100[field] = sum(int(r[field]) for r in recs100)
    total100["max_vmem"] = max(int(r["max_vmem"]) for r in recs100)
    s100t = summarize(recs100, total100)
    check("100.total_line.ok", s100t["total_line"]["ok"], True)
    # A wrong total must be detected, not silently accepted.
    bad_total = dict(total100)
    bad_total["drop_n"] = 1
    s100bad = summarize(recs100, bad_total)
    check("100.total_line.bad_detected", s100bad["total_line"]["ok"], False)

    # Conservation law, including the defrag exception.
    check("cons.valid", conserved(_record(1, 7, 3, 4, 0)), True)
    check("cons.short", conserved(_record(1, 7, 3, 3, 0)), False)
    check("cons.defrag", conserved(_record(1, 7, 3, 4, 0, defrag=1)), True)
    check("cons.defrag_short", conserved(_record(1, 7, 3, 3, 0, defrag=1)), False)
    check("cons.reuse_gt_n", conserved(_record(1, 3, 4, 0, 0)), False)

    # Parser: mixed stderr, prefixed JSON, malformed line, total line.
    stream = [
        "ggml-hex: opbuf stats enabled (GGML_HEXAGON_OPBUFSTAT=1)\n",
        '{"seq":1,"n_bufs":7,"reuse":7,"drop_n":0,"drop_bytes":0,"map_n":0,'
        '"map_bytes":0,"m_vmem":0,"e_vmem":0,"max_vmem":1000,"defrag":0,"scan_cmp":28}\n',
        'ggml-hex: prefix {"seq":2,"n_bufs":1,"reuse":0,"drop_n":0,"drop_bytes":0,'
        '"map_n":1,"map_bytes":4096,"m_vmem":0,"e_vmem":4096,"max_vmem":1000,'
        '"defrag":0,"scan_cmp":65}\n',
        '{"opbuf_total":{"batches":2,"n_bufs":8,"reuse":7,"drop_n":0,"drop_bytes":0,'
        '"map_n":1,"map_bytes":4096,"m_vmem":0,"e_vmem":4096,"max_vmem":1000,'
        '"defrag":0,"scan_cmp":93}}\n',
        "not json at all\n",
        '{"seq":3}\n',
    ]
    recs, total, ignored, malformed = parse_stream(stream)
    check("parse.records", len(recs), 2)
    check("parse.total", total is not None, True)
    check("parse.ignored", ignored, 2)
    check("parse.malformed", malformed, 1)
    check("parse.conservation", summarize(recs, total)["conservation"]["ok"], True)
    check("parse.total_line.ok", summarize(recs, total)["total_line"]["ok"], True)

    # Batch mode: 2 arms x 2 runs.  A is healthy; B-r1 has no opbuf_total and
    # B-r2 both violates conservation and carries one malformed line.
    with tempfile.TemporaryDirectory(prefix="opbuf-stats-selftest-") as tmp:
        def write_run(name, run_records, with_total=True, extra=()):
            path = os.path.join(tmp, name)
            lines = ["ggml-hex: opbuf stats enabled"]
            lines.extend(json.dumps(r, separators=(",", ":")) for r in run_records)
            lines.extend(extra)
            if with_total:
                run_total = {"batches": len(run_records)}
                for field in SUM_FIELDS:
                    run_total[field] = sum(int(r[field]) for r in run_records)
                run_total["max_vmem"] = max(
                    (int(r["max_vmem"]) for r in run_records), default=0)
                lines.append(json.dumps({"opbuf_total": run_total},
                                        separators=(",", ":")))
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            return path

        write_run("A-r1-opbuf.jsonl", [
            _record(1, 2, 1, 1, 128, drop_n=2, drop_bytes=100, map_bytes=200),
            _record(2, 4, 4, 0, 256, drop_n=1, drop_bytes=50),
        ])
        write_run("A-r2-opbuf.jsonl", [
            _record(1, 6, 2, 4, 384, drop_n=3, drop_bytes=300, map_bytes=800),
            _record(2, 8, 5, 3, 512, drop_n=4, drop_bytes=400, map_bytes=600),
        ])
        write_run("B-r1-opbuf.jsonl", [
            _record(1, 10, 10, 0, 640),
            _record(2, 20, 20, 0, 1280),
        ], with_total=False)
        write_run("B-r2-opbuf.jsonl", [
            _record(1, 7, 3, 3, 448, drop_n=5, drop_bytes=500),
        ], extra=['{"seq":9,"n_bufs":'])

        arm_map_path = os.path.join(tmp, "arms.tsv")
        with open(arm_map_path, "w", encoding="utf-8") as f:
            f.write("^A-\tA\n^B-\tB\n")
        arm_map = load_arm_map(arm_map_path)

        check("batch.arm.regex", _match_arm("A-r1", arm_map), "A")
        check("batch.arm.substring_fallback",
              _match_arm("A-r[1]", [("A-r[", "X")]), "X")
        check("batch.arm.unknown", _match_arm("C-r9", arm_map), "unknown")

        summary, warnings = run_batch(
            tmp, "*-opbuf.jsonl", arm_map, arm_map_path=arm_map_path)
        check("batch.schema", summary["schema"], BATCH_SCHEMA)
        check("batch.runs", len(summary["runs"]), 4)
        check("batch.arm_order", [a["arm"] for a in summary["arms"]], ["A", "B"])
        check("batch.warnings", len(warnings), 3)
        check("batch.failed_runs", summary["failed_runs"], 1)
        check("batch.ok", summary["ok"], False)
        check("batch.reasons", sorted(w["reason"] for w in warnings),
              ["conservation", "malformed", "missing_total"])
        check("batch.warn_paths", sorted(set(
            os.path.basename(w["path"]) for w in warnings)),
            ["B-r1-opbuf.jsonl", "B-r2-opbuf.jsonl"])
        check("batch.unknown_runs", summary["unknown_runs"], 0)
        by_tag = {r["tag"]: r for r in summary["runs"]}
        check("batch.A-r1.total_kept", by_tag["A-r1"]["total"]["batches"], 2)
        check("batch.B-r1.total_absent", by_tag["B-r1"]["total"], None)

        arm_a, arm_b = summary["arms"]
        check("batch.A.runs", arm_a["runs"], 2)
        check("batch.A.batches", arm_a["batches"], 4)
        check("batch.A.conservation_fail", arm_a["conservation_fail_runs"], 0)
        check("batch.A.missing_total", arm_a["missing_total_runs"], 0)
        check("batch.A.malformed", arm_a["malformed_lines"], 0)
        check("batch.A.failed", arm_a["failed_runs"], 0)
        check("batch.A.source", arm_a["drop_map_totals"]["source"], "exact")
        check("batch.A.drop_n_total", arm_a["drop_map_totals"]["drop_n"], 10)
        check("batch.A.drop_bytes_total", arm_a["drop_map_totals"]["drop_bytes"], 850)
        check("batch.A.map_n_total", arm_a["drop_map_totals"]["map_n"], 8)
        check("batch.A.map_bytes_total", arm_a["drop_map_totals"]["map_bytes"], 1600)
        check_close("batch.A.n_bufs.mom_p50",
                    arm_a["fields"]["n_bufs"]["median_of_medians"]["p50"], 5.0)
        check_close("batch.A.n_bufs.mom_p95",
                    arm_a["fields"]["n_bufs"]["median_of_medians"]["p95"], 5.9)
        check_close("batch.A.n_bufs.pooled_p50",
                    arm_a["fields"]["n_bufs"]["pooled"]["p50"], 5.0)
        check_close("batch.A.n_bufs.pooled_p95",
                    arm_a["fields"]["n_bufs"]["pooled"]["p95"], 7.7)
        check_close("batch.A.scan_cmp.mom_p50",
                    arm_a["fields"]["scan_cmp"]["median_of_medians"]["p50"], 320.0)

        check("batch.B.runs", arm_b["runs"], 2)
        check("batch.B.batches", arm_b["batches"], 3)
        check("batch.B.conservation_fail", arm_b["conservation_fail_runs"], 1)
        check("batch.B.missing_total", arm_b["missing_total_runs"], 1)
        check("batch.B.total_mismatch", arm_b["total_mismatch_runs"], 0)
        check("batch.B.malformed", arm_b["malformed_lines"], 1)
        check("batch.B.failed", arm_b["failed_runs"], 1)
        check("batch.B.source", arm_b["drop_map_totals"]["source"], "derived")
        check("batch.B.derived_runs", arm_b["drop_map_totals"]["derived_runs"], 1)
        check("batch.B.drop_n_total", arm_b["drop_map_totals"]["drop_n"], 5)
        check("batch.B.drop_bytes_total", arm_b["drop_map_totals"]["drop_bytes"], 500)
        check_close("batch.B.n_bufs.mom_p50",
                    arm_b["fields"]["n_bufs"]["median_of_medians"]["p50"], 11.0)
        check_close("batch.B.n_bufs.pooled_p50",
                    arm_b["fields"]["n_bufs"]["pooled"]["p50"], 10.0)
        check_close("batch.B.n_bufs.pooled_p95",
                    arm_b["fields"]["n_bufs"]["pooled"]["p95"], 19.0)

        tsv_path = os.path.join(tmp, "opbuf-arms.tsv")
        json_path = os.path.join(tmp, "opbuf-arms.json")
        write_arm_tsv(summary, tsv_path)
        write_arm_json(summary, json_path)
        with open(tsv_path, "r", encoding="utf-8") as f:
            tsv_lines = f.read().splitlines()
        check("batch.tsv.lines", len(tsv_lines), 3)
        check("batch.tsv.header", tsv_lines[0].split("\t"), arm_tsv_columns())
        tsv_a = dict(zip(tsv_lines[0].split("\t"), tsv_lines[1].split("\t")))
        check("batch.tsv.A.source", tsv_a["drop_map_total_source"], "exact")
        check("batch.tsv.A.mom_p50",
              tsv_a["n_bufs_median_of_medians_p50"], "5")
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        check("batch.json.arms", len(doc["arms"]), 2)
        check("batch.json.runs", len(doc["runs"]), 4)
        check("batch.json.no_internal_values",
              any("_values" in r for r in doc["runs"]), False)

        out_tsv = os.path.join(tmp, "cli-arms.tsv")
        out_json = os.path.join(tmp, "cli-arms.json")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = main(["--runs-dir", tmp, "--glob", "*-opbuf.jsonl",
                       "--arm-map", arm_map_path, "--out-tsv", out_tsv,
                       "--out-json", out_json])
        check("batch.cli.exit", rc, 1)
        err = stderr.getvalue()
        check("batch.cli.warn_missing", "B-r1-opbuf.jsonl" in err
              and "no opbuf_total line" in err, True)
        check("batch.cli.warn_conservation", "B-r2-opbuf.jsonl" in err
              and "conservation" in err, True)
        check("batch.cli.warn_malformed", "malformed line" in err, True)
        check("batch.cli.stdout", "opbuf arms:" in stdout.getvalue(), True)
        check("batch.cli.tsv_written", os.path.getsize(out_tsv) > 0, True)
        check("batch.cli.json_written", os.path.getsize(out_json) > 0, True)

        # Default exit code matches single-file mode: a missing total alone is
        # a warning, not a failure; --strict promotes it (and malformed lines).
        missing, missing_warnings = run_batch(
            tmp, "B-r1-opbuf.jsonl", arm_map, strict=False)
        check("batch.missing.default.ok", missing["ok"], True)
        check("batch.missing.default.failed", missing["failed_runs"], 0)
        check("batch.missing.default.warnings", len(missing_warnings), 1)
        strict, _ = run_batch(tmp, "B-r1-opbuf.jsonl", arm_map, strict=True)
        check("batch.missing.strict.ok", strict["ok"], False)
        check("batch.missing.strict.failed", strict["failed_runs"], 1)

    if failures:
        for f in failures:
            print("SELFTEST FAIL: " + f)
        return 1
    print("SELFTEST PASS: 20-record conservation, 100-record percentiles "
          "(p50=25.5 p95=48 max=50), defrag exception, total-line check, parser")
    print("SELFTEST PASS: batch 2 arms x 2 runs (MoM/pooled, 3 warnings, "
          "derived totals, exit code)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize GGML_HEXAGON_OPBUFSTAT=1 JSONL (T-0304).")
    parser.add_argument("path", nargs="?", default="-",
                        help="JSONL file (default: stdin; '-' for stdin)")
    parser.add_argument("--json", metavar="FILE",
                        help="write the full summary as JSON to FILE")
    parser.add_argument("--bucket-by", choices=("n_bufs", "n_ops"), default="n_bufs",
                        help="bucket key (default: n_bufs)")
    parser.add_argument("--bucket-width", type=int, default=0,
                        help="bucket width; 0 = exact value (default)")
    parser.add_argument("--runs-dir", metavar="DIR",
                        help="batch mode: aggregate every matching JSONL in DIR per arm")
    parser.add_argument("--glob", dest="glob_pattern", metavar="PATTERN", default=None,
                        help="batch filename glob (default: %s)" % DEFAULT_GLOB)
    parser.add_argument("--arm-map", metavar="FILE",
                        help="batch tag->arm map: TSV pattern<TAB>arm or JSON")
    parser.add_argument("--out-tsv", metavar="FILE",
                        help="batch mode: write per-arm TSV to FILE")
    parser.add_argument("--out-json", metavar="FILE",
                        help="batch mode: write per-arm/per-run JSON to FILE")
    parser.add_argument("--strict", action="store_true",
                        help="batch mode: also exit 1 on missing/mismatched total "
                             "and malformed lines")
    parser.add_argument("--self-test", action="store_true",
                        help="run built-in conservation/percentile self-test")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    batch_mode = bool(args.runs_dir or args.glob_pattern or args.arm_map
                      or args.out_tsv or args.out_json or args.strict)
    if batch_mode:
        if not args.runs_dir:
            parser.error("--runs-dir is required with batch options")
        if args.path != "-":
            parser.error("a JSONL path cannot be combined with --runs-dir")
        if args.json:
            parser.error("--json is single-file only; use --out-json with --runs-dir")
        if not args.arm_map:
            parser.error("--arm-map is required with --runs-dir")
        if args.out_tsv and args.out_json and (
                os.path.realpath(args.out_tsv) == os.path.realpath(args.out_json)):
            parser.error("--out-tsv and --out-json must be different files")
        try:
            arm_map = load_arm_map(args.arm_map)
        except (OSError, ValueError) as exc:
            print("opbuf_stats.py: bad --arm-map: %s" % exc, file=sys.stderr)
            return 2

        pattern = args.glob_pattern or DEFAULT_GLOB
        inputs = (globmod.glob(pattern) if os.path.isabs(pattern)
                  else globmod.glob(os.path.join(args.runs_dir, pattern)))
        input_real = {os.path.realpath(p) for p in inputs if os.path.isfile(p)}
        for label, out_path in (("--out-tsv", args.out_tsv),
                                ("--out-json", args.out_json)):
            if out_path and os.path.realpath(out_path) in input_real:
                parser.error("%s would overwrite an input file: %s" % (label, out_path))

        try:
            summary, warnings = run_batch(
                args.runs_dir, pattern, arm_map, args.bucket_by, args.bucket_width,
                args.strict, args.arm_map)
        except ValueError as exc:
            print("opbuf_stats.py: %s" % exc, file=sys.stderr)
            return 2
        for warning in warnings:
            print("opbuf_stats.py: warning: %s: %s: %s"
                  % (warning["path"], warning["reason"], warning["detail"]),
                  file=sys.stderr)
        print_arm_report(summary)
        try:
            if args.out_tsv:
                write_arm_tsv(summary, args.out_tsv)
            if args.out_json:
                write_arm_json(summary, args.out_json)
        except OSError as exc:
            print("opbuf_stats.py: cannot write batch output: %s" % exc,
                  file=sys.stderr)
            return 2
        return 0 if summary["ok"] else 1

    if args.path == "-":
        records, total, _ignored, malformed = parse_stream(sys.stdin)
    else:
        with open(args.path, "r", encoding="utf-8", errors="replace") as f:
            records, total, _ignored, malformed = parse_stream(f)

    if not records:
        print("opbuf_stats.py: no JSONL records found (malformed=%d)" % malformed,
              file=sys.stderr)
        return 1

    summary = summarize(records, total, args.bucket_by, args.bucket_width)
    summary["malformed_lines"] = malformed
    print_report(summary)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, sort_keys=False)
            f.write("\n")

    return 0 if summary["conservation"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
