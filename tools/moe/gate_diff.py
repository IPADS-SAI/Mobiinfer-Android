#!/usr/bin/env python3
"""Structured, read-only diff of two ``gate.json`` documents.

Why
---
A/B gate numbers are only comparable when the device identity, expert-package
identity and run configuration agree.  T-0366 showed that the same N=64/D8
configuration produced 15.65 t/s with the Q3_K package and 25.45 t/s with the
Q2_K package.  Subtracting those two numbers is meaningless.  This tool makes
the distinction machine-readable and can fail closed in CI/pre-review:

    identity / package / config / health / result

It never runs a device, never changes a gate, and never claims a t/s number.
The input JSON is read only.

Package identity caveat (2026-11-23 schema)
-------------------------------------------
``gate.sh`` currently persists ``speed.baseline_package`` with
``pin_id``/``manifest_sha256``/``bytes``/``mismatched``/``path``/``source``.
It does *not* persist ``stride0`` or ``requested_bytes``; those are only in the
run's ``.err`` log (``[moe-ssd] init ... stride0=...`` and the final
``[moe-ssd] io-summary ... requested_bytes=...``).  Without them a q3k/q2k diff
would silently lose the strongest package-identity signals.

Therefore, in the default ``--package-evidence auto`` mode, the tool fills
*only* missing ``stride0``/``requested_bytes`` values from the read-only run
logs referenced by the gate's ``speed.phase`` block (or its sibling
``speed/remote/runs`` directory).  Values already present in JSON always win,
including an explicit JSON ``null``.  Use ``--package-evidence off`` for a
strict JSON-only diff.  Every enriched value records its source under
``package_evidence`` in the output.

Exit codes
----------
0  inputs parsed and outputs written (differences are not an error by default)
1  ``--require-comparable`` was given and ``comparable`` is false
2  input missing/broken, top-level JSON is not an object, or output write failed
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys

SCHEMA = "gate-diff.v1"

IDENTITY_KEYS = (
    "task",
    "mode",
    "overall",
    "phone",
    "arch",
    "bench",
    "skel",
    "gate_version",
    "dry_run",
)

PACKAGE_ALIASES = (
    ("id", ("id", "pin_id")),
    ("sha", ("sha", "sha256", "manifest_sha256")),
    ("bytes", ("bytes", "size", "experts_bytes")),
    ("stride0", ("stride0",)),
    ("requested_bytes", ("requested_bytes", "io_requested_bytes")),
)

PACKAGE_KEYS = (
    "baseline_package",
    "baseline_package.id",
    "baseline_package.sha",
    "baseline_package.bytes",
    "baseline_package.stride0",
    "baseline_package.requested_bytes",
    "package_mismatch",
    "accuracy.criterion",
)

CONFIG_EXACT = (
    "n_predict",
    "calls",
    "verify_rows",
    "runs",
    "before_tps",
    "after_tps",
    "gain",
    "policy",
    "criterion",
)

HEALTH_EXACT = ("health", "thermal", "quality")
RESULT_KEYS = (
    "build.ok",
    "accuracy.ok",
    "speed.ok",
    "overall",
    "exit_code",
    "after_tps",
)

QUALITY_ARM_KEYS = (
    "acceptance",
    "acceptance_pooled",
    "calls_g",
    "gen_tokens",
    "repetition_ratio",
    "distinct4_ratio",
    "content_class",
    "n_predict",
    "per_call_comparable",
    "raw_ms_per_call",
    "steady_ms_per_call",
    "eval_ms",
    "first_decode_ms",
)

_MISSING = object()


class GateInputError(Exception):
    """Input is missing, unreadable, malformed, or not a JSON object."""


def load_gate(path):
    """Load a gate.json file and require a top-level JSON object."""
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as exc:
        raise GateInputError(f"{path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GateInputError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise GateInputError(
            f"{path}: top-level JSON is {type(document).__name__}, expected object"
        )
    return document


def _get_path(document, *keys):
    current = document
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _first_present(mapping, aliases):
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return _MISSING


def _summary_value(value):
    """Keep JSON scalars, summarise arrays/objects instead of echoing them."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, (list, tuple)):
        if len(value) <= 8 and all(
            item is None or isinstance(item, (bool, int, float, str))
            for item in value
        ):
            return list(value)
        return _fingerprint(value)
    if isinstance(value, dict):
        return _fingerprint(value)
    return _fingerprint(value)


def _fingerprint(value):
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except (TypeError, ValueError):
        payload = repr(value)
    digest = hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()
    if isinstance(value, (list, tuple)):
        return {"n": len(value), "sha256": digest[:16]}
    return {"keys": sorted(value) if isinstance(value, dict) else [], "sha256": digest[:16]}


def _flatten_into(flat, value, prefix, max_depth):
    if isinstance(value, dict):
        if not value:
            flat[prefix] = {}
            return
        for key, child in value.items():
            child_key = f"{prefix}.{key}"
            if isinstance(child, dict) and max_depth > 1:
                _flatten_into(flat, child, child_key, max_depth - 1)
            else:
                flat[child_key] = _summary_value(child)
        return
    flat[prefix] = _summary_value(value)


def _flatten_quality(flat, quality):
    for key in (
        "ok",
        "effective_result",
        "degraded",
        "reason",
        "quality_error",
        "allow_degrade",
        "allow_reason",
    ):
        if key in quality:
            flat[f"quality.{key}"] = _summary_value(quality[key])
    for section in ("checks", "delta"):
        value = quality.get(section, _MISSING)
        if isinstance(value, dict):
            for key, child in value.items():
                flat[f"quality.{section}.{key}"] = _summary_value(child)
        elif value is not _MISSING:
            flat[f"quality.{section}"] = _summary_value(value)
    for arm in ("arm_a", "arm_b"):
        block = quality.get(arm)
        if not isinstance(block, dict):
            continue
        for key in QUALITY_ARM_KEYS:
            if key in block:
                flat[f"quality.{arm}.{key}"] = _summary_value(block[key])


def flatten_gate(document):
    """Return a whitelisted ``path -> value`` map for one gate document.

    Absent keys are omitted; present keys with a JSON ``null`` are kept as
    ``None``.  This is what lets :func:`diff_dicts` distinguish "missing"
    from "explicitly null".
    """
    if not isinstance(document, dict):
        raise TypeError("flatten_gate expects a dict")

    flat = {}
    for key in IDENTITY_KEYS:
        if key in document:
            flat[key] = document[key]

    speed = document.get("speed", _MISSING)
    if isinstance(speed, dict):
        package = speed.get("baseline_package", _MISSING)
        if isinstance(package, dict):
            for name, aliases in PACKAGE_ALIASES:
                value = _first_present(package, aliases)
                if value is not _MISSING:
                    flat[f"baseline_package.{name}"] = value
        elif package is not _MISSING:
            flat["baseline_package"] = _summary_value(package)

        # Optional package identity fields in newer JSON, or in provenance /
        # phase blocks produced by future gate.sh versions.
        stride_sources = (
            ("speed", "stride0"),
            ("speed", "provenance", "arm_a", "stride0"),
            ("speed", "provenance", "arm_a", "effective_config", "stride0"),
            ("speed", "provenance", "stride0"),
            ("speed", "phase", "package", "auto_stride0"),
        )
        if "baseline_package.stride0" not in flat:
            for path in stride_sources:
                value = _get_path(document, *path)
                if value is not _MISSING:
                    flat["baseline_package.stride0"] = value
                    break

        requested_sources = (
            ("speed", "requested_bytes"),
            ("speed", "provenance", "arm_a", "requested_bytes"),
            ("speed", "provenance", "requested_bytes"),
            ("speed", "phase", "package", "auto_requested_bytes"),
        )
        if "baseline_package.requested_bytes" not in flat:
            for path in requested_sources:
                value = _get_path(document, *path)
                if value is not _MISSING:
                    flat["baseline_package.requested_bytes"] = value
                    break

        if "package_mismatch" in speed:
            flat["package_mismatch"] = speed["package_mismatch"]

        parity = speed.get("parity", _MISSING)
        if isinstance(parity, dict):
            _flatten_into(flat, parity, "parity", 4)
        elif parity is not _MISSING:
            flat["parity"] = _summary_value(parity)

        provenance = speed.get("provenance")
        effective = _MISSING
        if isinstance(provenance, dict):
            effective = _get_path(provenance, "arm_a", "effective_config")
        if isinstance(effective, dict):
            for key, value in effective.items():
                flat[f"effective_config.{key}"] = _summary_value(value)

        # Canonical run-config keys: prefer speed.parity, fall back to the
        # provenance/env actually recorded by T-0164 gates.
        canonical_sources = {
            "n_predict": (
                (parity, ("n_predict",)) if isinstance(parity, dict) else None,
                (effective, ("n_predict",)) if isinstance(effective, dict) else None,
                (provenance, ("n_predict",)) if isinstance(provenance, dict) else None,
            ),
            "verify_rows": (
                (parity, ("verify_rows",)) if isinstance(parity, dict) else None,
                (effective, ("verify_rows",)) if isinstance(effective, dict) else None,
                (provenance, ("verify_rows",)) if isinstance(provenance, dict) else None,
            ),
            # ``calls`` is a parity/config field only when parity records it.
            # Do not substitute quality.arm_a.calls_g: that count follows the
            # generated content/acceptance and is not a configuration mismatch.
            "calls": (
                (parity, ("calls",)) if isinstance(parity, dict) else None,
                (parity, ("calls_g",)) if isinstance(parity, dict) else None,
            ),
        }
        for name, sources in canonical_sources.items():
            for source in sources:
                if source is None:
                    continue
                mapping, keys = source
                if not isinstance(mapping, dict):
                    continue
                value = _get_path(mapping, *keys)
                if value is not _MISSING:
                    flat[name] = value
                    break

        if isinstance(parity, dict):
            env = parity.get("env", _MISSING)
            if isinstance(env, dict):
                for key, value in env.items():
                    flat[f"env.{key}"] = _summary_value(value)
            elif env is not _MISSING:
                flat["env"] = _summary_value(env)

        for key in ("runs", "before_tps", "after_tps", "gain", "policy", "criterion"):
            if key in speed:
                flat[key] = speed[key]

        for block_name in ("health", "thermal"):
            block = speed.get(block_name, _MISSING)
            if isinstance(block, dict):
                _flatten_into(flat, block, block_name, 6)
            elif block is not _MISSING:
                flat[block_name] = _summary_value(block)

        quality = speed.get("quality", _MISSING)
        if isinstance(quality, dict):
            _flatten_quality(flat, quality)
        elif quality is not _MISSING:
            flat["quality"] = _summary_value(quality)
    elif speed is not _MISSING:
        flat["speed"] = _summary_value(speed)

    accuracy = document.get("accuracy", _MISSING)
    if isinstance(accuracy, dict):
        if "criterion" in accuracy:
            flat["accuracy.criterion"] = accuracy["criterion"]
    elif accuracy is not _MISSING:
        flat["accuracy"] = _summary_value(accuracy)

    for block_name, key in (
        ("build", "build.ok"),
        ("accuracy", "accuracy.ok"),
        ("speed", "speed.ok"),
    ):
        block = document.get(block_name)
        if isinstance(block, dict) and "ok" in block:
            flat[key] = block["ok"]

    if "overall" in document:
        flat["overall"] = document["overall"]
    if "exit_code" in document:
        flat["exit_code"] = document["exit_code"]

    if isinstance(speed, dict) and "after_tps" in speed:
        # Kept in the result group as well as config: reviewers routinely ask
        # "was after_tps even recorded for this non-comparable pair?".
        flat["after_tps"] = speed["after_tps"]

    return flat


def _values_equal(left, right):
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return left == right


def diff_dicts(flat_a, flat_b):
    """Compare two flattened dicts over the union of their keys."""
    entries = []
    keys = list(flat_a) + [key for key in flat_b if key not in flat_a]
    for key in keys:
        present_a = key in flat_a
        present_b = key in flat_b
        value_a = _summary_value(flat_a[key]) if present_a else None
        value_b = _summary_value(flat_b[key]) if present_b else None
        if not present_a or not present_b:
            status = "missing"
        elif _values_equal(value_a, value_b):
            status = "same"
        else:
            status = "diff"
        entries.append(
            {"key": key, "a": value_a, "b": value_b, "status": status}
        )
    return entries


def _select_group(flat, exact, prefixes):
    selected = {}
    for key, value in flat.items():
        if key in exact or (prefixes and key.startswith(prefixes)):
            selected[key] = value
    return selected


def _group_entries(flat_a, flat_b):
    groups = {}
    rules = {
        "identity": (set(IDENTITY_KEYS), ()),
        "package": (set(PACKAGE_KEYS), ()),
        "config": (set(CONFIG_EXACT), ("parity.", "env.", "effective_config.")),
        "health": (set(HEALTH_EXACT), ("health.", "thermal.", "quality.")),
        "result": (set(RESULT_KEYS), ()),
    }
    for name, (exact, prefixes) in rules.items():
        groups[name] = diff_dicts(
            _select_group(flat_a, exact, prefixes),
            _select_group(flat_b, exact, prefixes),
        )
    return groups


def _entry_map(entries):
    return {entry["key"]: entry for entry in entries}


def _collect_reasons(groups):
    reasons = []

    def add(reason):
        if reason not in reasons:
            reasons.append(reason)

    def diffed(group, key, reason):
        entry = _entry_map(groups[group]).get(key)
        if entry is not None and entry["status"] == "diff":
            add(reason)

    identity = _entry_map(groups["identity"])
    for key in ("phone", "arch", "bench", "skel"):
        diffed("identity", key, f"{key}_diff")
    if identity.get("mode", {}).get("status") == "diff":
        add("mode_diff")

    for key in ("id", "sha", "bytes", "stride0", "requested_bytes"):
        diffed("package", f"baseline_package.{key}", f"package_{key}_diff")

    for key in ("n_predict", "calls", "verify_rows"):
        entry = _entry_map(groups["config"]).get(key)
        if entry is not None and entry["status"] == "diff":
            add(f"parity_{key}_diff")

    dry_run = identity.get("dry_run")
    if dry_run is not None and (dry_run["a"] is True or dry_run["b"] is True):
        add("dry_run")

    return reasons


# ---------------------------------------------------------------------------
# Optional read-only package identity from the run log (see module docstring)


def _parse_package_log(path):
    try:
        if os.path.getsize(path) > 5 * 1024 * 1024:
            return None, None
    except OSError:
        return None, None
    stride0 = None
    requested_bytes = None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if stride0 is None:
                    match = re.search(r"\bstride0=(\d+)", line)
                    if match:
                        stride0 = int(match.group(1))
                if "io-summary" in line:
                    match = re.search(r"\brequested_bytes=(\d+)", line)
                    if match:
                        requested_bytes = int(match.group(1))
    except OSError:
        return None, None
    return stride0, requested_bytes


def _phase_run_logs(document, gate_path):
    phase = _get_path(document, "speed", "phase")
    phase_path = None
    if isinstance(phase, dict):
        phase_path = phase.get("json") or phase.get("path")
    elif isinstance(phase, str):
        phase_path = phase

    gate_dir = os.path.dirname(os.path.abspath(gate_path))
    candidates = []

    if isinstance(phase_path, str) and phase_path:
        if not os.path.isabs(phase_path):
            phase_path = os.path.join(gate_dir, phase_path)
        phase_path = os.path.abspath(phase_path)
        phase_doc = None
        try:
            with open(phase_path, encoding="utf-8") as handle:
                phase_doc = json.load(handle)
        except (OSError, json.JSONDecodeError):
            phase_doc = None
        if isinstance(phase_doc, dict):
            phase_dir = os.path.dirname(phase_path)
            for run in phase_doc.get("runs") or []:
                if not isinstance(run, dict):
                    continue
                arm = str(run.get("arm") or run.get("group") or "").upper()
                if arm != "A":
                    continue
                log = run.get("log") or run.get("companion")
                if not isinstance(log, str) or not log:
                    continue
                for base in (phase_dir, os.path.join(phase_dir, "remote", "runs")):
                    candidate = log if os.path.isabs(log) else os.path.join(base, log)
                    if os.path.isfile(candidate):
                        candidates.append(candidate)

    if not candidates:
        pattern = os.path.join(gate_dir, "speed", "remote", "runs", "gate-r*-A-*.err")
        candidates = sorted(glob.glob(pattern))

    def rank(path):
        name = os.path.basename(path)
        return (
            0 if "-n64" in name else 1,
            0 if "dropcache" not in name else 1,
            0 if not name.startswith("gate-acc") else 1,
            name,
        )

    unique = []
    seen = set()
    for path in sorted(candidates, key=rank):
        resolved = os.path.abspath(path)
        if resolved in seen or not os.path.isfile(resolved):
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _enrich_package_identity(document, gate_path, flat):
    """Fill missing stride0/requested_bytes from read-only run logs.

    Returns an audit dict for ``package_evidence``.  JSON values always win.
    """
    needed = [
        key
        for key in ("baseline_package.stride0", "baseline_package.requested_bytes")
        if key not in flat
    ]
    if not needed:
        return {}

    logs = _phase_run_logs(document, gate_path)
    if not logs:
        return {"note": "no run log found for package identity"}

    stride_values = []
    requested_values = []
    for path in logs[:32]:
        stride0, requested = _parse_package_log(path)
        if stride0 is not None:
            stride_values.append((stride0, path))
        if requested is not None:
            requested_values.append((requested, path))

    audit = {"logs_scanned": min(len(logs), 32)}
    if "baseline_package.stride0" in needed and stride_values:
        values = sorted({value for value, _ in stride_values})
        if len(values) == 1:
            flat["baseline_package.stride0"] = values[0]
            audit["stride0"] = {"value": values[0], "source": stride_values[0][1]}
        else:
            audit["stride0"] = {"value": None, "conflict": values}

    if "baseline_package.requested_bytes" in needed and requested_values:
        values = sorted({value for value, _ in requested_values})
        # requested_bytes legitimately varies a little between runs; use the
        # first ranked A run and expose the observations for audit.
        flat["baseline_package.requested_bytes"] = requested_values[0][0]
        audit["requested_bytes"] = {
            "value": requested_values[0][0],
            "source": requested_values[0][1],
        }
        if len(values) > 1:
            audit["requested_bytes"]["observations"] = values[:8]
    return audit


# ---------------------------------------------------------------------------
# Report / rendering


def _header(path, label, document):
    return {
        "path": path,
        "label": label,
        "task": document.get("task"),
        "overall": document.get("overall"),
        "mode": document.get("mode"),
    }


def build_report(
    document_a,
    document_b,
    path_a="A",
    path_b="B",
    label_a=None,
    label_b=None,
    package_evidence="auto",
):
    """Build the ``gate-diff.v1`` report for two already-loaded documents."""
    label_a = label_a or (document_a.get("task") or path_a)
    label_b = label_b or (document_b.get("task") or path_b)

    flat_a = flatten_gate(document_a)
    flat_b = flatten_gate(document_b)

    evidence = {"mode": package_evidence, "a": {}, "b": {}}
    if package_evidence == "auto":
        evidence["a"] = _enrich_package_identity(document_a, path_a, flat_a)
        evidence["b"] = _enrich_package_identity(document_b, path_b, flat_b)

    groups = _group_entries(flat_a, flat_b)
    reasons = _collect_reasons(groups)

    return {
        "schema": SCHEMA,
        "a": _header(path_a, label_a, document_a),
        "b": _header(path_b, label_b, document_b),
        "comparable": not reasons,
        "incomparable_reasons": reasons,
        "package_evidence": evidence,
        "groups": groups,
    }


def _md_value(value):
    if isinstance(value, str):
        text = value.replace("\n", " ").replace("\r", " ")
    elif value is None:
        text = "null"
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.replace("|", "\\|")
    if len(text) > 180:
        text = text[:177] + "..."
    return text


def render_markdown(report):
    lines = []
    a = report["a"]
    b = report["b"]
    lines.append(f"# gate diff: {a['label']} vs {b['label']}")
    lines.append("")
    lines.append(
        f"- **A**: `{a['path']}` "
        f"(task={_md_value(a['task'])}, overall={_md_value(a['overall'])}, "
        f"mode={_md_value(a['mode'])})"
    )
    lines.append(
        f"- **B**: `{b['path']}` "
        f"(task={_md_value(b['task'])}, overall={_md_value(b['overall'])}, "
        f"mode={_md_value(b['mode'])})"
    )
    lines.append(f"- **Comparable: {'true' if report['comparable'] else 'false'}**")
    reasons = report["incomparable_reasons"]
    if reasons:
        lines.append("- **Incomparable reasons**: " + ", ".join(f"`{r}`" for r in reasons))
    else:
        lines.append("- **Incomparable reasons**: none")
    lines.append("")

    for group in ("identity", "package", "config", "health", "result"):
        lines.append(f"## {group}")
        lines.append("")
        entries = report["groups"][group]
        if not entries:
            lines.append("_No entries._")
            lines.append("")
            continue
        lines.append("| key | A | B | status |")
        lines.append("|---|---|---|---|")
        for entry in entries:
            lines.append(
                "| `{}` | {} | {} | **{}** |".format(
                    entry["key"],
                    _md_value(entry["a"]),
                    _md_value(entry["b"]),
                    entry["status"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_console(report):
    a = report["a"]
    b = report["b"]
    lines = [
        f"gate-diff: {a['label']} vs {b['label']}",
        f"  a: {a['path']} (task={a['task']}, overall={a['overall']}, mode={a['mode']})",
        f"  b: {b['path']} (task={b['task']}, overall={b['overall']}, mode={b['mode']})",
        f"  comparable: {'true' if report['comparable'] else 'false'}",
    ]
    if report["incomparable_reasons"]:
        lines.append("  reasons: " + ", ".join(report["incomparable_reasons"]))
    else:
        lines.append("  reasons: none")
    for group in ("identity", "package", "config", "health", "result"):
        entries = report["groups"][group]
        diffs = sum(1 for entry in entries if entry["status"] == "diff")
        lines.append(f"  {group}: {len(entries)} entries, {diffs} diff")
    return "\n".join(lines)


def _write_text(path, text):
    if path == "-":
        sys.stdout.write(text)
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only parity diff of two gate.json files. Differences do not "
            "fail by default; --require-comparable exits 1 for incomparable pairs."
        )
    )
    parser.add_argument("--a", required=True, help="first gate.json")
    parser.add_argument("--b", required=True, help="second gate.json")
    parser.add_argument("--json", dest="json_out", help="write gate-diff.v1 JSON here ('-' for stdout)")
    parser.add_argument("--markdown", dest="markdown_out", help="write Markdown here ('-' for stdout)")
    parser.add_argument(
        "--require-comparable",
        action="store_true",
        help="exit 1 when the pair has incomparable identity/package/config",
    )
    parser.add_argument("--label-a", help="display label for A (default: task or path)")
    parser.add_argument("--label-b", help="display label for B (default: task or path)")
    parser.add_argument(
        "--package-evidence",
        choices=("auto", "off"),
        default="auto",
        help=(
            "auto (default): fill only missing stride0/requested_bytes from the "
            "gate's read-only run logs; off: strict JSON-only"
        ),
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.json_out == "-" and args.markdown_out == "-":
        print("gate_diff: error: --json - and --markdown - cannot both use stdout", file=sys.stderr)
        return 2

    try:
        document_a = load_gate(args.a)
        document_b = load_gate(args.b)
    except GateInputError as exc:
        print(f"gate_diff: error: {exc}", file=sys.stderr)
        return 2

    report = build_report(
        document_a,
        document_b,
        path_a=args.a,
        path_b=args.b,
        label_a=args.label_a,
        label_b=args.label_b,
        package_evidence=args.package_evidence,
    )

    try:
        if args.json_out:
            _write_text(
                args.json_out,
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            )
        if args.markdown_out:
            _write_text(args.markdown_out, render_markdown(report))
    except OSError as exc:
        print(f"gate_diff: error: cannot write output: {exc}", file=sys.stderr)
        return 2

    console_stream = sys.stderr if args.json_out == "-" else sys.stdout
    print(render_console(report), file=console_stream)

    if args.require_comparable and not report["comparable"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
