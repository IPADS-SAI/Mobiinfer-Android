#!/usr/bin/env python3
"""Build a host-side hot/cold two-group tiled allocation plan for MoE experts.

This tool is deliberately **plan-only**.  It reads the raw manifest produced by
``tools/moe/experts.py extract`` plus a T-0041-style ``groups-*.tsv`` and emits a
deterministic JSON description of how a future packer would split every
``(layer, family)`` tensor into a hot group (default ``Q2_K``) and a cold group
(default ``Q1_0``).  It never reads or writes ``layer_*.bin`` payloads, never
calls ``pack_hexagon.py``, and does not touch GGUF metadata, kernels, the
runtime, or the device.

Input schema
------------

Raw manifest (all paths/offsets are as written by ``experts.py extract``)::

    {
      "layout": "layer_major_expert",
      "model": {"expert_count": 256, ...},          # or top-level expert_count
      "entries": [
        {
          "layer": 0,
          "tensor_family": "ffn_down_exps",          # "family" is accepted too
          "tensor_name": "blk.0.ffn_down_exps.weight",
          "repacked_file": "layer_000.bin",
          "repacked_offset": 294912,
          "expert_stride": 442368,
          "bytes_per_expert": 147456,
          "quant_type": "Q1_0",
          "shape": [512, 2048, 256]
        }, ...
      ]
    }

``shape`` is ``[rows, cols, n_expert]`` (the GGUF expert axis is last).  A
two-element ``[rows, cols]`` shape is also accepted.  ``repacked_offset`` is the
family's byte offset inside one expert record and ``expert_stride`` is the full
record stride.  A file may hold one family (then ``expert_stride ==
bytes_per_expert`` and ``repacked_offset == 0``) or several contiguous families
(the normal ``experts.py`` layout).  The validator enforces both forms.

Groups TSV::

    layer<TAB>family<TAB>hot_count<TAB>hot_experts_by_frequency_rank
    0<TAB>ffn_down_exps<TAB>2<TAB>31,152

``hot_experts`` keeps the frequency order from the TSV.  ``hot_count`` may be
``0`` (empty or ``-`` list) or ``n_expert``.  The real ``make_groups.py`` only
emits non-empty allocations, so a manifest pair that is absent from the TSV is
interpreted as ``hot_count=0``; pass ``--require-groups-exact`` to make that an
error instead.

Output schema (``--out plan.json``, keys sorted, byte-identical across runs)::

    {
      "version": 1,
      "layout": "hexagon-tiled-hotcold-v1",
      "n_expert": 256,
      "hot_type": "Q2_K",
      "cold_type": "Q1_0",
      "entries": [
        {
          "layer": 0, "family": "ffn_down_exps",
          "tensor_name": "blk.0.ffn_down_exps.weight",
          "source_file": "layer_000.bin", "source_offset": 294912,
          "source_expert_stride": 442368,
          "hot_count": 2, "hot_experts": [31, 152],
          "cold_experts": [0, 1, ...], "order": [31, 152, 0, 1, ...],
          "hot":  {"file": "layer_000.ffn_down_exps.hot.bin",
                   "offset": 0, "bytes_per_expert": 344064,
                   "count": 2, "bytes": 688128},
          "cold": {"file": "layer_000.ffn_down_exps.cold.bin",
                   "offset": 0, "bytes_per_expert": 147456,
                   "count": 254, "bytes": 37453824}
        }
      ],
      "totals": {
        "hot_bytes": 0, "cold_bytes": 0, "source_bytes": 0,
        "planned_bytes": 0, "delta_bytes": 0
      }
    }

Byte model
----------

Per-expert bytes are recomputed from the manifest shape (``n_params =
shape[0] * shape[1]``), never from a fixed multiplier:

* ``Q1_0``: ``n_params / 128 * 18``
* ``Q2_K``: ``n_params / 256 * 84``

The raw entry type must equal ``--cold-type`` and its ``bytes_per_expert`` must
match the cold formula; otherwise the tool stops instead of guessing.  Only the
two types above are supported because those are the ones T-0041 plans.

``totals.source_bytes`` is the raw cold-type payload:
``sum(expert_count * raw_bytes_per_expert)``.  ``hot_bytes + cold_bytes`` is
the **planned** package size and is reported separately as ``planned_bytes``;
``delta_bytes = planned_bytes - source_bytes`` is the quantization budget
increase.  When ``hot_type == cold_type`` the three values coincide.  For
``Q2_K``/``Q1_0`` they intentionally differ (T-0041's ``pe20302p`` is
``+301989888`` bytes = ``+0.302 GB``); forcing ``hot+cold == source`` would
misrepresent that budget.

File naming and order semantics
-------------------------------

Every ``(layer, family)`` gets two independent files::

    layer_{layer:03d}.{family}.hot.bin
    layer_{layer:03d}.{family}.cold.bin

Both file offsets start at 0 and are contiguous within that file.  ``order`` is
``hot_experts + cold_experts`` and defines the future packer's physical order
and the runtime's logical-expert -> position mapping: position ``p`` in a group
file stores logical expert ``order[p]``.  ``cold_experts`` is the ascending
complement of the hot set.

Boundary with ``pack_hexagon.py``
---------------------------------

``pack_hexagon.py`` consumes the **raw** ``layer_major_expert`` manifest and
tiles one type per family; it rejects manifests that already carry
``expert_order`` or a ``backend_layout``.  This planner is a separate upstream
step: a future hot/cold packer will consume this JSON plus the raw ``.bin``
files named by ``source_file``/``source_offset``/``source_expert_stride`` and
write the two new files.  Nothing here is a runnable model or a native pack.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

VERSION = 1
PLAN_LAYOUT = "hexagon-tiled-hotcold-v1"
RAW_LAYOUT = "layer_major_expert"

# Must match experts.py FAMILY_ORDER so entries are emitted in raw family order.
FAMILY_ORDER: dict[str, int] = {
    "ffn_gate_exps": 0,
    "ffn_up_exps": 1,
    "ffn_down_exps": 2,
    "ffn_gate_up_exps": 3,
}
FAMILIES = tuple(sorted(FAMILY_ORDER, key=FAMILY_ORDER.__getitem__))
TENSOR_RE = re.compile(r"^blk\.(\d+)\.(ffn_[^.]+)\.weight$")

# name -> (block size in weights, bytes per block)
QUANT_BYTES: dict[str, tuple[int, int]] = {
    "Q1_0": (128, 18),
    "Q2_K": (256, 84),
}

HOT_COUNT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1-16", 1, 16),
    ("17-128", 17, 128),
    ("129-255", 129, 255),
    ("256", 256, 256),
)

GROUPS_REQUIRED_COLUMNS = ("layer", "family", "hot_count", "hot_experts_by_frequency_rank")


class PlanError(ValueError):
    """Invalid input or internal consistency failure; the CLI exits 2."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise PlanError(f"{where}: expected an integer, got a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise PlanError(f"{where}: expected an integer, got {value!r}") from exc
    raise PlanError(f"{where}: expected an integer, got {value!r}")


def _as_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{where}: expected a JSON object, got {type(value).__name__}")
    return value


def _quant_name(value: str, where: str) -> str:
    name = str(value).strip().upper()
    if name not in QUANT_BYTES:
        supported = ", ".join(sorted(QUANT_BYTES))
        raise PlanError(f"{where}: unsupported quant type {value!r} (supported: {supported})")
    return name


def quant_bytes_per_expert(n_params: int, quant_type: str, where: str) -> int:
    """Return exact bytes per expert for a supported quant type."""
    block, block_bytes = QUANT_BYTES[quant_type]
    if n_params <= 0:
        raise PlanError(f"{where}: n_params must be positive, got {n_params}")
    if n_params % block:
        raise PlanError(
            f"{where}: n_params={n_params} is not divisible by {quant_type} block {block}"
        )
    return n_params // block * block_bytes


def _shape_n_params(shape: Any, n_expert: int, where: str) -> int:
    if not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3):
        raise PlanError(f"{where}: shape must be [rows, cols] or [rows, cols, n_expert], got {shape!r}")
    dims = [_as_int(item, f"{where}.shape[{index}]") for index, item in enumerate(shape)]
    if len(dims) == 3 and dims[2] != n_expert:
        raise PlanError(
            f"{where}: shape expert axis is {dims[2]}, manifest n_expert is {n_expert}"
        )
    rows, cols = dims[0], dims[1]
    if rows <= 0 or cols <= 0:
        raise PlanError(f"{where}: shape dimensions must be positive, got {dims}")
    return rows * cols


def _family_of(entry: dict[str, Any], where: str) -> str:
    family = entry.get("family") or entry.get("tensor_family")
    if family is None:
        match = TENSOR_RE.match(str(entry.get("tensor_name", "")))
        if match is not None:
            family = match.group(2)
    if not isinstance(family, str) or not family:
        raise PlanError(f"{where}: missing family/tensor_family and cannot derive it from tensor_name")
    if family not in FAMILY_ORDER:
        raise PlanError(f"{where}: unknown expert family {family!r}")
    return family


def _relative_output_name(name: Any, where: str) -> str:
    if not isinstance(name, str) or not name:
        raise PlanError(f"{where}: repacked_file must be a non-empty string")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PlanError(f"{where}: unsafe repacked_file {name!r}")
    return name


def hot_count_bucket(count: int) -> str:
    for label, low, high in HOT_COUNT_BUCKETS:
        if count >= low and (high is None or count <= high):
            return label
    raise PlanError(f"cannot bucket hot_count={count}")


def hot_count_histogram(counts: Iterable[int]) -> dict[str, int]:
    histogram = {label: 0 for label, _low, _high in HOT_COUNT_BUCKETS}
    for count in counts:
        histogram[hot_count_bucket(count)] += 1
    return histogram


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


class GroupSpec:
    """One validated groups-TSV row."""

    __slots__ = ("layer", "family", "hot_count", "hot_experts", "line")

    def __init__(self, layer: int, family: str, hot_count: int,
                 hot_experts: list[int], line: int) -> None:
        self.layer = layer
        self.family = family
        self.hot_count = hot_count
        self.hot_experts = hot_experts
        self.line = line

    def key(self) -> tuple[int, str]:
        return (self.layer, self.family)


def _parse_hot_list(raw: Any, where: str) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text == "-":
        return []
    result: list[int] = []
    for position, item in enumerate(text.split(",")):
        token = item.strip()
        if not token:
            raise PlanError(f"{where}: empty expert id at position {position}")
        result.append(_as_int(token, f"{where}[{position}]"))
    return result


def parse_groups(path: Path, n_expert: int) -> dict[tuple[int, str], GroupSpec]:
    """Read and validate a groups TSV; missing rows are *not* invented here."""
    if n_expert <= 0:
        raise PlanError(f"--n-expert must be positive, got {n_expert}")
    if not path.is_file():
        raise PlanError(f"groups file not found: {path}")

    groups: dict[tuple[int, str], GroupSpec] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        missing = [name for name in GROUPS_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise PlanError(
                f"{path}: missing groups header column(s) {missing}; "
                f"expected {list(GROUPS_REQUIRED_COLUMNS)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if not row or all(value is None or str(value).strip() == "" for value in row.values()):
                continue
            where = f"{path}:{line_number}"
            layer = _as_int(row.get("layer"), f"{where}.layer")
            if layer < 0:
                raise PlanError(f"{where}: layer must be >= 0, got {layer}")
            family = str(row.get("family") or "").strip()
            if family not in FAMILY_ORDER:
                raise PlanError(f"{where}: unknown expert family {family!r}")

            hot_count = _as_int(row.get("hot_count"), f"{where}.hot_count")
            if not 0 <= hot_count <= n_expert:
                raise PlanError(
                    f"{where}: hot_count={hot_count} outside [0, {n_expert}]"
                )
            hot_experts = _parse_hot_list(
                row.get("hot_experts_by_frequency_rank"), f"{where}.hot_experts"
            )
            if len(hot_experts) != hot_count:
                raise PlanError(
                    f"{where}: hot_count={hot_count} but {len(hot_experts)} hot experts listed"
                )
            seen: set[int] = set()
            for position, expert in enumerate(hot_experts):
                if not 0 <= expert < n_expert:
                    raise PlanError(
                        f"{where}: hot expert {expert} at position {position} outside [0, {n_expert})"
                    )
                if expert in seen:
                    raise PlanError(f"{where}: duplicate hot expert {expert}")
                seen.add(expert)

            spec = GroupSpec(layer, family, hot_count, hot_experts, line_number)
            if spec.key() in groups:
                raise PlanError(
                    f"{where}: duplicate groups row for layer={layer} family={family}"
                )
            groups[spec.key()] = spec

    if not groups:
        raise PlanError(f"{path}: no group rows found")
    return groups


def _manifest_n_expert(document: dict[str, Any]) -> int:
    candidates: list[tuple[str, int]] = []
    model = document.get("model")
    if isinstance(model, dict) and model.get("expert_count") is not None:
        candidates.append(("model.expert_count", _as_int(model["expert_count"], "model.expert_count")))
    if document.get("expert_count") is not None:
        candidates.append(("expert_count", _as_int(document["expert_count"], "expert_count")))
    for index, entry in enumerate(document.get("entries") or []):
        if isinstance(entry, dict) and entry.get("expert_count") is not None:
            candidates.append(
                (f"entries[{index}].expert_count",
                 _as_int(entry["expert_count"], f"entries[{index}].expert_count"))
            )
    if not candidates:
        raise PlanError("manifest: no model.expert_count, expert_count, or per-entry expert_count")
    values = {value for _name, value in candidates}
    if len(values) != 1:
        detail = ", ".join(f"{name}={value}" for name, value in candidates)
        raise PlanError(f"manifest: inconsistent expert counts: {detail}")
    n_expert = candidates[0][1]
    if n_expert <= 0:
        raise PlanError(f"manifest: expert_count must be positive, got {n_expert}")
    return n_expert


def load_manifest(path: Path, cold_type: str,
                  n_expert_override: int | None = None) -> tuple[list[dict[str, Any]], int]:
    """Load a raw layer_major_expert manifest and validate its structure."""
    if not path.is_file():
        raise PlanError(f"manifest not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"{path}: cannot parse JSON: {exc}") from exc
    document = _as_object(document, str(path))

    if document.get("backend_layout"):
        raise PlanError(
            f"{path}: already has backend_layout={document['backend_layout']!r}; "
            "expected a raw layer_major_expert manifest"
        )
    if document.get("layout") != RAW_LAYOUT:
        raise PlanError(
            f"{path}: layout={document.get('layout')!r}, expected {RAW_LAYOUT!r}"
        )

    n_expert = _manifest_n_expert(document)
    if n_expert_override is not None and n_expert_override != n_expert:
        raise PlanError(
            f"--n-expert={n_expert_override} disagrees with manifest expert_count={n_expert}"
        )

    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PlanError(f"{path}: entries must be a non-empty list")

    entries: list[dict[str, Any]] = []
    keys: set[tuple[int, str]] = set()
    tensor_names: set[str] = set()
    for index, raw in enumerate(raw_entries):
        where = f"{path}:entries[{index}]"
        entry = _as_object(raw, where)
        layer = _as_int(entry.get("layer"), f"{where}.layer")
        if layer < 0:
            raise PlanError(f"{where}: layer must be >= 0, got {layer}")
        family = _family_of(entry, where)
        tensor_name = entry.get("tensor_name")
        if not isinstance(tensor_name, str) or not tensor_name:
            raise PlanError(f"{where}: tensor_name must be a non-empty string")
        if tensor_name in tensor_names:
            raise PlanError(f"{where}: duplicate tensor_name {tensor_name!r}")
        tensor_names.add(tensor_name)

        key = (layer, family)
        if key in keys:
            raise PlanError(f"{where}: duplicate manifest entry for layer={layer} family={family}")
        keys.add(key)

        quant_type = _quant_name(entry.get("quant_type", ""), f"{where}.quant_type")
        if quant_type != cold_type:
            raise PlanError(
                f"{where}: quant_type={quant_type} != cold type {cold_type}; "
                "refusing to guess (source must be cold-type Q1_0)"
            )

        shape = entry.get("shape")
        n_params = _shape_n_params(shape, n_expert, where)
        bytes_per_expert = _as_int(entry.get("bytes_per_expert"), f"{where}.bytes_per_expert")
        if bytes_per_expert <= 0:
            raise PlanError(f"{where}: bytes_per_expert must be positive")
        expected_cold = quant_bytes_per_expert(n_params, cold_type, where)
        if bytes_per_expert != expected_cold:
            raise PlanError(
                f"{where}: bytes_per_expert={bytes_per_expert} disagrees with "
                f"{cold_type} formula from shape ({expected_cold})"
            )

        entry_count = entry.get("expert_count")
        if entry_count is not None and _as_int(entry_count, f"{where}.expert_count") != n_expert:
            raise PlanError(
                f"{where}: expert_count={entry_count} disagrees with manifest n_expert={n_expert}"
            )

        source_file = _relative_output_name(entry.get("repacked_file"), f"{where}.repacked_file")
        source_offset = _as_int(entry.get("repacked_offset"), f"{where}.repacked_offset")
        expert_stride = _as_int(entry.get("expert_stride"), f"{where}.expert_stride")
        if source_offset < 0:
            raise PlanError(f"{where}: repacked_offset must be >= 0")
        if expert_stride <= 0:
            raise PlanError(f"{where}: expert_stride must be positive")

        entries.append(
            {
                "layer": layer,
                "family": family,
                "tensor_name": tensor_name,
                "source_file": source_file,
                "source_offset": source_offset,
                "source_expert_stride": expert_stride,
                "bytes_per_expert": bytes_per_expert,
                "quant_type": quant_type,
                "shape": list(shape),
                "n_params": n_params,
                "n_expert": n_expert,
            }
        )

    _validate_repacked_groups(entries, path)
    entries.sort(key=lambda item: (item["layer"], FAMILY_ORDER[item["family"]]))
    return entries, n_expert


def _validate_repacked_groups(entries: list[dict[str, Any]], path: Path) -> None:
    """Validate contiguous family records, shared or one-family files alike."""
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_file[entry["source_file"]].append(entry)

    for file_name, group in by_file.items():
        group.sort(key=lambda item: item["source_offset"])
        stride = group[0]["source_expert_stride"]
        expected_offset = 0
        for entry in group:
            where = f"{path}:{entry['tensor_name']}"
            if entry["source_expert_stride"] != stride:
                raise PlanError(
                    f"{where}: expert_stride={entry['source_expert_stride']} disagrees with "
                    f"other entries in {file_name!r} (stride={stride})"
                )
            if entry["source_offset"] != expected_offset:
                raise PlanError(
                    f"{where}: repacked_offset={entry['source_offset']} is not contiguous; "
                    f"expected {expected_offset} in {file_name!r}"
                )
            expected_offset += entry["bytes_per_expert"]
        expected_stride = sum(entry["bytes_per_expert"] for entry in group)
        if stride != expected_stride:
            raise PlanError(
                f"{path}: {file_name!r}: expert_stride={stride} != sum(bytes_per_expert)="
                f"{expected_stride}"
            )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _file_name(layer: int, family: str, kind: str) -> str:
    return f"layer_{layer:03d}.{family}.{kind}.bin"


def build_plan_from(entries: list[dict[str, Any]], n_expert: int,
                    groups: dict[tuple[int, str], GroupSpec],
                    hot_type: str, cold_type: str,
                    require_groups_exact: bool = False) -> tuple[dict[str, Any], int]:
    """Build a plan from already-parsed inputs; returns (plan, implicit_zero_groups)."""
    entry_keys = {(entry["layer"], entry["family"]) for entry in entries}
    extra = sorted(set(groups) - entry_keys)
    if extra:
        shown = ", ".join(f"layer={layer} family={family}" for layer, family in extra[:5])
        raise PlanError(f"groups contain {len(extra)} (layer, family) pair(s) absent from manifest: {shown}")

    implicit_zero = 0
    source_total = 0
    plan_entries: list[dict[str, Any]] = []
    for entry in entries:
        key = (entry["layer"], entry["family"])
        spec = groups.get(key)
        if spec is None:
            if require_groups_exact:
                raise PlanError(
                    f"groups are missing layer={entry['layer']} family={entry['family']} "
                    "(--require-groups-exact)"
                )
            hot_experts: list[int] = []
            implicit_zero += 1
        else:
            hot_experts = list(spec.hot_experts)

        hot_set = set(hot_experts)
        cold_experts = [expert for expert in range(n_expert) if expert not in hot_set]
        order = hot_experts + cold_experts
        if len(hot_experts) + len(cold_experts) != n_expert or len(set(order)) != n_expert:
            raise PlanError(
                f"internal partition error for layer={entry['layer']} family={entry['family']}"
            )

        hot_bpe = quant_bytes_per_expert(entry["n_params"], hot_type, entry["tensor_name"])
        cold_bpe = entry["bytes_per_expert"]
        hot_bytes = len(hot_experts) * hot_bpe
        cold_bytes = len(cold_experts) * cold_bpe
        source_total += entry["n_expert"] * entry["bytes_per_expert"]

        plan_entries.append(
            {
                "layer": entry["layer"],
                "family": entry["family"],
                "tensor_name": entry["tensor_name"],
                "source_file": entry["source_file"],
                "source_offset": entry["source_offset"],
                "source_expert_stride": entry["source_expert_stride"],
                "hot_count": len(hot_experts),
                "hot_experts": hot_experts,
                "cold_experts": cold_experts,
                "order": order,
                "hot": {
                    "file": _file_name(entry["layer"], entry["family"], "hot"),
                    "offset": 0,
                    "bytes_per_expert": hot_bpe,
                    "count": len(hot_experts),
                    "bytes": hot_bytes,
                },
                "cold": {
                    "file": _file_name(entry["layer"], entry["family"], "cold"),
                    "offset": 0,
                    "bytes_per_expert": cold_bpe,
                    "count": len(cold_experts),
                    "bytes": cold_bytes,
                },
            }
        )

    plan_entries.sort(key=lambda item: (item["layer"], FAMILY_ORDER[item["family"]]))
    hot_total = sum(item["hot"]["bytes"] for item in plan_entries)
    cold_total = sum(item["cold"]["bytes"] for item in plan_entries)
    planned_total = hot_total + cold_total
    plan = {
        "version": VERSION,
        "layout": PLAN_LAYOUT,
        "n_expert": n_expert,
        "hot_type": hot_type,
        "cold_type": cold_type,
        "entries": plan_entries,
        "totals": {
            "hot_bytes": hot_total,
            "cold_bytes": cold_total,
            "source_bytes": source_total,
            "planned_bytes": planned_total,
            "delta_bytes": planned_total - source_total,
        },
    }
    validate_plan(plan)
    return plan, implicit_zero


def validate_plan(plan: dict[str, Any]) -> None:
    """Re-check every derived field so a bug can never emit a silently bad plan."""
    n_expert = plan["n_expert"]
    entries = plan["entries"]
    if not entries:
        raise PlanError("plan has no entries")
    if len({(item["layer"], item["family"]) for item in entries}) != len(entries):
        raise PlanError("plan has duplicate (layer, family) entries")

    hot_total = 0
    cold_total = 0
    for item in entries:
        hot = item["hot"]
        cold = item["cold"]
        where = f"layer={item['layer']} family={item['family']}"
        hot_experts = item["hot_experts"]
        cold_experts = item["cold_experts"]
        order = item["order"]
        if len(hot_experts) != item["hot_count"] or hot["count"] != len(hot_experts):
            raise PlanError(f"{where}: hot count mismatch")
        if cold["count"] != len(cold_experts):
            raise PlanError(f"{where}: cold count mismatch")
        if len(hot_experts) + len(cold_experts) != n_expert:
            raise PlanError(f"{where}: hot+cold counts != n_expert")
        if cold_experts != sorted(cold_experts):
            raise PlanError(f"{where}: cold_experts is not ascending")
        if set(hot_experts) & set(cold_experts):
            raise PlanError(f"{where}: hot/cold overlap")
        if order != hot_experts + cold_experts or sorted(order) != list(range(n_expert)):
            raise PlanError(f"{where}: order is not a hot+cold permutation")
        if hot["offset"] != 0 or cold["offset"] != 0:
            raise PlanError(f"{where}: group file offsets must start at 0")
        if hot["bytes"] != hot["count"] * hot["bytes_per_expert"]:
            raise PlanError(f"{where}: hot bytes != count * bytes_per_expert")
        if cold["bytes"] != cold["count"] * cold["bytes_per_expert"]:
            raise PlanError(f"{where}: cold bytes != count * bytes_per_expert")
        if hot["file"] != _file_name(item["layer"], item["family"], "hot"):
            raise PlanError(f"{where}: bad hot file name")
        if cold["file"] != _file_name(item["layer"], item["family"], "cold"):
            raise PlanError(f"{where}: bad cold file name")
        hot_total += hot["bytes"]
        cold_total += cold["bytes"]

    totals = plan["totals"]
    if totals["hot_bytes"] != hot_total:
        raise PlanError("totals.hot_bytes is inconsistent")
    if totals["cold_bytes"] != cold_total:
        raise PlanError("totals.cold_bytes is inconsistent")
    if totals["planned_bytes"] != hot_total + cold_total:
        raise PlanError("totals.planned_bytes != hot_bytes + cold_bytes")
    if totals["delta_bytes"] != totals["planned_bytes"] - totals["source_bytes"]:
        raise PlanError("totals.delta_bytes is inconsistent")
    if plan["hot_type"] == plan["cold_type"] and totals["planned_bytes"] != totals["source_bytes"]:
        raise PlanError("same hot/cold type must leave source bytes unchanged")
    if plan["hot_type"] != plan["cold_type"]:
        # planned - source == sum(hot_count * (hot_bpe - cold_bpe)); checked per entry.
        expected_delta = sum(
            item["hot_count"] * (item["hot"]["bytes_per_expert"] - item["cold"]["bytes_per_expert"])
            for item in entries
        )
        if totals["delta_bytes"] != expected_delta:
            raise PlanError("totals.delta_bytes disagrees with per-entry type deltas")


def summarize_plan(plan: dict[str, Any], explicit_groups: int, implicit_zero: int) -> dict[str, Any]:
    entries = plan["entries"]
    return {
        "ok": True,
        "mode": "manifest",
        "layout": plan["layout"],
        "n_expert": plan["n_expert"],
        "entries": len(entries),
        "layers": len({item["layer"] for item in entries}),
        "families": sorted({item["family"] for item in entries}, key=FAMILY_ORDER.__getitem__),
        "groups_explicit": explicit_groups,
        "groups_implicit_zero": implicit_zero,
        "hot_count_histogram": hot_count_histogram(item["hot_count"] for item in entries),
        "totals": plan["totals"],
    }


def check_groups_only(groups: dict[tuple[int, str], GroupSpec], n_expert: int) -> dict[str, Any]:
    layers = sorted({layer for layer, _family in groups})
    families = sorted({family for _layer, family in groups}, key=FAMILY_ORDER.__getitem__)
    counts = [spec.hot_count for spec in groups.values()]
    return {
        "ok": True,
        "mode": "groups-only",
        "n_expert": n_expert,
        "groups": len(groups),
        "layers": layers,
        "layer_count": len(layers),
        "families": families,
        "hot_experts": sum(counts),
        "hot_count_histogram": hot_count_histogram(counts),
        "note": "manifest not supplied; only groups schema/range/order validated",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hot_cold_plan.py",
        description=(
            "Plan hot/cold expert split files from a raw experts.py manifest and a "
            "T-0041 groups TSV (host-only; does not pack bytes or touch the runtime)."
        ),
        epilog=(
            "Output schema, byte model, file naming and order semantics are documented "
            "in the module docstring (python3 -c 'import hot_cold_plan; help(hot_cold_plan)')."
        ),
    )
    parser.add_argument("--manifest", type=Path,
                        help="raw layer_major_expert manifest.json from experts.py extract")
    parser.add_argument("--groups", required=True, type=Path,
                        help="groups TSV: layer, family, hot_count, hot_experts_by_frequency_rank")
    parser.add_argument("--out", type=Path,
                        help="write plan JSON here (required unless --check-only)")
    parser.add_argument("--hot-type", default="Q2_K", help="hot group quant label (default: Q2_K)")
    parser.add_argument("--cold-type", default="Q1_0", help="cold group quant label (default: Q1_0)")
    parser.add_argument("--n-expert", type=int, default=None,
                        help="expert count for groups-only checks (default: 256); "
                             "with --manifest it must match the manifest")
    parser.add_argument("--check-only", action="store_true",
                        help="validate inputs and print a JSON summary; do not write a plan")
    parser.add_argument("--require-groups-exact", action="store_true",
                        help="error when a manifest (layer, family) is absent from groups "
                             "instead of treating it as hot_count=0")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        hot_type = _quant_name(args.hot_type, "--hot-type")
        cold_type = _quant_name(args.cold_type, "--cold-type")
        if args.n_expert is not None and args.n_expert <= 0:
            raise PlanError(f"--n-expert must be positive, got {args.n_expert}")

        if args.manifest is None:
            n_expert = args.n_expert if args.n_expert is not None else 256
            groups = parse_groups(args.groups, n_expert)
            if not args.check_only:
                raise PlanError("--manifest is required unless --check-only is used")
            summary = check_groups_only(groups, n_expert)
            print(json.dumps(summary, sort_keys=True))
            return 0

        entries, n_expert = load_manifest(args.manifest, cold_type, args.n_expert)
        groups = parse_groups(args.groups, n_expert)
        plan, implicit_zero = build_plan_from(
            entries, n_expert, groups, hot_type, cold_type, args.require_groups_exact
        )
        summary = summarize_plan(plan, len(groups), implicit_zero)
        if args.check_only:
            print(json.dumps(summary, sort_keys=True))
            return 0
        if args.out is None:
            raise PlanError("--out is required unless --check-only")
        write_json_atomic(args.out, plan)
        summary["plan"] = str(args.out)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except PlanError as exc:
        print(f"hot_cold_plan: error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"hot_cold_plan: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
