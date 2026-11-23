#!/usr/bin/env python3
"""Record and verify provenance for a Hexagon-tiled MoE expert package.

The weighted and unweighted Q1_0 tiled packages have exactly the same file
sizes and layout; only the expert tensor bytes (and therefore the manifest's
``files_sha256``) differ.  Deploying an unweighted package as if it were
weighted silently loses a large amount of perplexity, so this tool writes a
small sidecar next to the tiled package:

    tiled/manifest.json
    tiled/layer_000.bin ...
    tiled/provenance.json      <- this tool, never touches the files above

``stamp`` records the arm/imatrix/rules/quantizer hashes, the manifest hash,
the tiled layout/counts, and the ``weighted_q1_0`` flag.  ``--imatrix`` is
optional: an unweighted arm has no imatrix file, in which case the sidecar
records ``imatrix_sha256``/``imatrix_bytes`` as ``null`` and only
``weighted_q1_0=false|unknown`` is allowed.  ``verify`` checks the sidecar
against the manifest and streams the layer files to compare their sha256
values.  Old packages without a sidecar still verify (with a warning) unless
``--require-provenance`` is given.

Exit codes (also printed by ``--help``):

    0  success / verification passed
    2  input or infrastructure error (missing/unreadable file, bad JSON,
       malformed CLI argument)
    3  provenance verification failed (hash/layout/count/expected-sha
       mismatch, or provenance required but absent)

Only the Python standard library is used.  Arm metadata is read with a small
header-only GGUF parser (the tensor-data region is never touched); if the arm
header cannot be parsed, ``--weighted-q1-0 auto`` records ``unknown`` instead
of guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# tools/moe/q1_provenance.py -> parents[2] = repository root.
LLAMA_ROOT = Path(__file__).resolve().parents[2]

TOOL_VERSION = "1.1.0"
SCHEMA_VERSION = 1
BACKEND_LAYOUT = "hexagon-tiled-v1"
MANIFEST_NAME = "manifest.json"
DEFAULT_PROVENANCE_NAME = "provenance.json"
DEFAULT_QUANTIZER_SOURCE = LLAMA_ROOT / "ggml" / "src" / "ggml-quants.c"
CHUNK_SIZE = 8 * 1024 * 1024

# Family order used by experts.py/pack_hexagon.py.  The tiled file stores the
# three families back-to-back for each expert position.
FAMILY_ORDER = {
    "ffn_gate_exps": 0,
    "ffn_up_exps": 1,
    "ffn_down_exps": 2,
}

# Keys written into a quantized arm by llama-quantize --imatrix, plus the
# metadata keys of the imatrix GGUF itself.  Their presence means an imatrix
# was used for this arm; it does not prove the flag by file size (which is
# identical for weighted/unweighted Q1_0 packages).
IMATRIX_METADATA_KEYS = (
    "quantize.imatrix.file",
    "quantize.imatrix.dataset",
    "quantize.imatrix.entries_count",
    "quantize.imatrix.chunks_count",
    "imatrix.datasets",
    "imatrix.chunk_count",
    "imatrix.chunk_size",
)
IMATRIX_PREFIXES = ("quantize.imatrix.", "imatrix.")
LAYER_FILE_RE = re.compile(r"^layer_(\d{3})\.bin$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_VERIFY = 3

EXIT_CODES = """\
exit codes:
  0  success / verification passed
  2  input or infrastructure error (missing/unreadable file, bad JSON,
     malformed CLI argument)
  3  provenance verification failed (hash/layout/count/expected-sha
     mismatch, or --require-provenance was given but no sidecar exists)
"""


class InputError(Exception):
    """Missing/unreadable/malformed input; maps to exit code 2."""


class VerificationError(Exception):
    """The sidecar and tiled package disagree; maps to exit code 3."""


@dataclass(frozen=True)
class TiledInfo:
    directory: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_raw: bytes
    entries: list[dict[str, Any]]
    layers: list[int]
    expert_count: int
    backend_layout: str
    files_sha256: dict[str, str]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` while streaming *path*."""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest(), size


def _sha256_text(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_int(value: Any, what: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VerificationError(f"{what}: expected an integer, got {value!r}")
    if value < minimum:
        raise VerificationError(f"{what}: expected >= {minimum}, got {value}")
    return value


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{what}: expected a non-empty string, got {value!r}")
    return value


def _require_sha256(value: Any, what: str) -> str:
    text = _require_str(value, what).strip().lower()
    if not SHA256_RE.match(text):
        raise VerificationError(f"{what}: not a sha256 hex digest: {value!r}")
    return text


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{what}: expected an object, got {type(value).__name__}")
    return value


def _json_safe(value: Any) -> Any:
    """Convert parser values to plain JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Defensive conversion for array-like values; bytes are made printable.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _resolved_dir(value: Path, what: str) -> Path:
    path = value.expanduser().resolve()
    if not path.is_dir():
        raise InputError(f"{what}: directory not found: {path}")
    return path


def _resolved_file(value: Path, what: str) -> Path:
    path = value.expanduser().resolve()
    if not path.is_file():
        raise InputError(f"{what}: file not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Manifest / tiled package validation
# ---------------------------------------------------------------------------


def load_tiled(tiled_dir: Path) -> TiledInfo:
    """Read and structurally validate a tiled package (read-only)."""
    directory = _resolved_dir(tiled_dir, "--tiled")
    manifest_path = directory / MANIFEST_NAME
    try:
        manifest_raw = manifest_path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read tiled manifest {manifest_path}: {exc}") from exc
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"tiled manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise InputError(f"tiled manifest is not a JSON object: {manifest_path}")

    backend_layout = manifest.get("backend_layout")
    if backend_layout != BACKEND_LAYOUT:
        raise VerificationError(
            f"manifest backend_layout is {backend_layout!r}, expected {BACKEND_LAYOUT!r} "
            "(this is not a Hexagon-tiled package)"
        )

    model = _require_mapping(manifest.get("model"), "manifest.model")
    expert_count = _require_int(model.get("expert_count"), "manifest.model.expert_count", 1)

    conversion = _require_mapping(manifest.get("hexagon_conversion"), "manifest.hexagon_conversion")
    files_raw = _require_mapping(conversion.get("files_sha256"), "manifest.hexagon_conversion.files_sha256")
    if not files_raw:
        raise VerificationError("manifest.hexagon_conversion.files_sha256 is empty")
    files_sha256: dict[str, str] = {}
    for name, digest in files_raw.items():
        if not isinstance(name, str) or not LAYER_FILE_RE.match(name):
            raise VerificationError(f"unexpected tiled file name in files_sha256: {name!r}")
        files_sha256[name] = _require_sha256(digest, f"files_sha256[{name}]")

    entries_raw = manifest.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise VerificationError("manifest.entries must be a non-empty list")

    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for index, raw in enumerate(entries_raw):
        what = f"manifest.entries[{index}]"
        entry = _require_mapping(raw, what)
        layer = _require_int(entry.get("layer"), f"{what}.layer")
        family = _require_str(entry.get("tensor_family"), f"{what}.tensor_family")
        if family not in FAMILY_ORDER:
            raise VerificationError(f"{what}.tensor_family: unsupported family {family!r}")
        if (layer, family) in seen:
            raise VerificationError(f"{what}: duplicate layer/family {layer}/{family}")
        seen.add((layer, family))

        repacked_file = _require_str(entry.get("repacked_file"), f"{what}.repacked_file")
        match = LAYER_FILE_RE.match(repacked_file)
        if match is None or int(match.group(1)) != layer:
            raise VerificationError(
                f"{what}.repacked_file: {repacked_file!r} does not match layer {layer}"
            )
        bytes_per_expert = _require_int(entry.get("bytes_per_expert"), f"{what}.bytes_per_expert", 1)
        expert_stride = _require_int(entry.get("expert_stride"), f"{what}.expert_stride", 1)
        repacked_offset = _require_int(entry.get("repacked_offset"), f"{what}.repacked_offset")
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) < 3:
            raise VerificationError(f"{what}.shape: expected a list with >= 3 dimensions")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
            raise VerificationError(f"{what}.shape: dimensions must be positive integers")
        if shape[2] != expert_count:
            raise VerificationError(
                f"{what}.shape[2] is {shape[2]}, expected expert_count {expert_count}"
            )
        exact = _require_int(entry.get("exact_byte_length"), f"{what}.exact_byte_length", 1)
        if exact != bytes_per_expert * expert_count:
            raise VerificationError(
                f"{what}.exact_byte_length is {exact}, "
                f"expected bytes_per_expert*expert_count={bytes_per_expert * expert_count}"
            )
        if repacked_offset + bytes_per_expert > expert_stride:
            raise VerificationError(
                f"{what}: family range {repacked_offset}+{bytes_per_expert} "
                f"exceeds expert_stride {expert_stride}"
            )
        entries.append(dict(entry))
        by_layer.setdefault(layer, []).append(entry)

    layers = sorted(by_layer)
    expected_files: set[str] = set()
    for layer, layer_entries in by_layer.items():
        if len(layer_entries) != len(FAMILY_ORDER):
            raise VerificationError(
                f"layer {layer}: expected {len(FAMILY_ORDER)} family entries, got {len(layer_entries)}"
            )
        families = {entry["tensor_family"] for entry in layer_entries}
        if families != set(FAMILY_ORDER):
            missing = sorted(set(FAMILY_ORDER) - families)
            raise VerificationError(f"layer {layer}: missing families {missing}")
        ordered = sorted(layer_entries, key=lambda item: FAMILY_ORDER[item["tensor_family"]])
        stride = ordered[0]["expert_stride"]
        if any(entry["expert_stride"] != stride for entry in ordered):
            raise VerificationError(f"layer {layer}: entries disagree on expert_stride")
        if stride != sum(entry["bytes_per_expert"] for entry in ordered):
            raise VerificationError(f"layer {layer}: expert_stride is not the sum of family sizes")
        offset = 0
        for entry in ordered:
            if entry["repacked_offset"] != offset:
                raise VerificationError(
                    f"layer {layer}: non-contiguous repacked_offset for {entry['tensor_family']} "
                    f"({entry['repacked_offset']}, expected {offset})"
                )
            offset += entry["bytes_per_expert"]
        if offset != stride:
            raise VerificationError(f"layer {layer}: family sizes do not fill expert_stride")

        file_name = ordered[0]["repacked_file"]
        expected_files.add(file_name)
        path = directory / file_name
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise VerificationError(f"layer file missing/unreadable: {path}: {exc}") from exc
        expected_size = stride * expert_count
        if size != expected_size:
            raise VerificationError(
                f"layer file size mismatch: {file_name}: got {size}, expected {expected_size}"
            )

    if set(files_sha256) != expected_files:
        missing = sorted(expected_files - set(files_sha256))
        extra = sorted(set(files_sha256) - expected_files)
        raise VerificationError(
            f"manifest files_sha256 set mismatch; missing={missing} extra={extra}"
        )

    return TiledInfo(
        directory=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_raw=manifest_raw,
        entries=entries,
        layers=layers,
        expert_count=expert_count,
        backend_layout=backend_layout,
        files_sha256=files_sha256,
    )


def verify_layer_files(info: TiledInfo) -> None:
    """Stream every declared layer file and compare its sha256."""
    for name in sorted(info.files_sha256):
        expected = info.files_sha256[name]
        actual, _ = sha256_file(info.directory / name)
        if actual != expected:
            raise VerificationError(
                f"layer file sha256 mismatch: {name}: expected {expected}, got {actual}"
            )
    declared = set(info.files_sha256)
    extras = sorted(
        path.name
        for path in info.directory.glob("layer_*.bin")
        if path.name not in declared
    )
    if extras:
        print(
            f"warning: tiled directory contains undeclared layer files: {', '.join(extras)}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Header-only GGUF metadata parser (stdlib only; no tensor data is read)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
SUPPORTED_GGUF_VERSIONS = (2, 3)
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<?", 1),   # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}
class _GgufError(Exception):
    pass


def _gguf_read_exact(handle: Any, size: int, file_size: int, what: str) -> bytes:
    if size < 0:
        raise _GgufError(f"negative size while reading {what}")
    position = handle.tell()
    if size > file_size - position:
        raise _GgufError(
            f"truncated GGUF while reading {what} at offset {position}: "
            f"need {size} bytes, only {file_size - position} left"
        )
    data = handle.read(size)
    if len(data) != size:
        raise _GgufError(f"short read while reading {what} at offset {position}")
    return data


def _gguf_read_u32(handle: Any, file_size: int, what: str) -> int:
    return struct.unpack("<I", _gguf_read_exact(handle, 4, file_size, what))[0]


def _gguf_read_u64(handle: Any, file_size: int, what: str) -> int:
    return struct.unpack("<Q", _gguf_read_exact(handle, 8, file_size, what))[0]


def _gguf_read_string(
    handle: Any, file_size: int, what: str, max_len: int = 1 << 26
) -> str:
    length = _gguf_read_u64(handle, file_size, f"{what} length")
    if length > max_len:
        raise _GgufError(f"{what} is too long ({length} bytes)")
    raw = _gguf_read_exact(handle, length, file_size, what)
    return raw.decode("utf-8", errors="replace")


def _gguf_skip_value(
    handle: Any, value_type: int, file_size: int, depth: int = 0
) -> None:
    """Skip one metadata value without retaining it."""
    if depth > 8:
        raise _GgufError("GGUF metadata arrays are nested too deeply")
    scalar = GGUF_SCALAR_FORMATS.get(value_type)
    if scalar is not None:
        _, size = scalar
        _gguf_read_exact(handle, size, file_size, "metadata scalar")
        return
    if value_type == GGUF_STRING:
        length = _gguf_read_u64(handle, file_size, "metadata string length")
        _gguf_read_exact(handle, length, file_size, "metadata string")
        return
    if value_type != GGUF_ARRAY:
        raise _GgufError(f"unknown GGUF metadata value type {value_type}")

    element_type = _gguf_read_u32(handle, file_size, "array element type")
    count = _gguf_read_u64(handle, file_size, "array count")
    element_scalar = GGUF_SCALAR_FORMATS.get(element_type)
    if element_scalar is not None:
        _, size = element_scalar
        remaining = file_size - handle.tell()
        if count * size > remaining:
            raise _GgufError("GGUF scalar array does not fit in the remaining header")
        handle.seek(count * size, os.SEEK_CUR)
        return
    if element_type == GGUF_STRING:
        if count * 8 > file_size - handle.tell():
            raise _GgufError("GGUF string array does not fit in the remaining header")
        for _ in range(count):
            length = _gguf_read_u64(handle, file_size, "metadata string array length")
            _gguf_read_exact(handle, length, file_size, "metadata string array item")
        return
    if element_type == GGUF_ARRAY:
        if count * 12 > file_size - handle.tell():
            raise _GgufError("GGUF nested array does not fit in the remaining header")
        for _ in range(count):
            _gguf_skip_value(handle, GGUF_ARRAY, file_size, depth + 1)
        return
    raise _GgufError(f"unknown GGUF metadata array element type {element_type}")


def _gguf_read_value(
    handle: Any, value_type: int, file_size: int, depth: int = 0
) -> Any:
    """Read one metadata value for an interesting key."""
    if depth > 8:
        raise _GgufError("GGUF metadata arrays are nested too deeply")
    scalar = GGUF_SCALAR_FORMATS.get(value_type)
    if scalar is not None:
        fmt, size = scalar
        return struct.unpack(fmt, _gguf_read_exact(handle, size, file_size, "metadata value"))[0]
    if value_type == GGUF_STRING:
        return _gguf_read_string(handle, file_size, "metadata string")
    if value_type != GGUF_ARRAY:
        raise _GgufError(f"unknown GGUF metadata value type {value_type}")
    element_type = _gguf_read_u32(handle, file_size, "array element type")
    count = _gguf_read_u64(handle, file_size, "array count")
    if count > (file_size - handle.tell()):
        raise _GgufError("GGUF metadata array count does not fit in the header")
    return [
        _gguf_read_value(handle, element_type, file_size, depth + 1)
        for _ in range(count)
    ]


def read_arm_imatrix_metadata(arm: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Return ``(metadata, error)`` for imatrix-related arm GGUF keys.

    A missing key is not an error: it simply means the arm does not advertise
    an imatrix.  Any parsing failure is returned as text so ``stamp`` can
    record it and fall back to ``unknown`` instead of guessing.
    """
    try:
        file_size = arm.stat().st_size
        with arm.open("rb") as handle:
            magic_raw = _gguf_read_exact(handle, 4, file_size, "magic")
            if struct.unpack("<I", magic_raw)[0] != GGUF_MAGIC:
                if struct.unpack(">I", magic_raw)[0] == GGUF_MAGIC:
                    raise _GgufError("big-endian GGUF is not supported")
                raise _GgufError(f"bad GGUF magic: {magic_raw!r}")
            version = _gguf_read_u32(handle, file_size, "version")
            if version not in SUPPORTED_GGUF_VERSIONS:
                raise _GgufError(f"unsupported GGUF version {version}")
            tensor_count = _gguf_read_u64(handle, file_size, "tensor count")
            metadata_count = _gguf_read_u64(handle, file_size, "metadata count")
            remaining = file_size - handle.tell()
            if tensor_count * 24 > remaining:
                raise _GgufError("tensor-info array does not fit in the file")
            if metadata_count * 13 > remaining:
                raise _GgufError("metadata array does not fit in the file")

            found: dict[str, Any] = {}
            for index in range(metadata_count):
                key = _gguf_read_string(
                    handle, file_size, f"metadata key #{index}", max_len=1 << 20
                )
                value_type = _gguf_read_u32(
                    handle, file_size, f"metadata type for {key!r}"
                )
                if key in IMATRIX_METADATA_KEYS or key.startswith(IMATRIX_PREFIXES):
                    found[key] = _json_safe(
                        _gguf_read_value(handle, value_type, file_size)
                    )
                else:
                    _gguf_skip_value(handle, value_type, file_size)
            return (found or None), None
    except (OSError, _GgufError, struct.error) as exc:
        return None, f"cannot read arm GGUF metadata: {exc}"


def infer_weighted(
    requested: str,
    arm_imatrix: Optional[dict[str, Any]],
) -> tuple[str, str, str]:
    """Return ``(flag, source, explanation)``; never infer from file size."""
    if requested in ("true", "false"):
        return requested, "explicit", f"explicit --weighted-q1-0 {requested}"
    if arm_imatrix:
        keys = ", ".join(sorted(arm_imatrix))
        return "true", "arm-metadata", f"arm metadata contains imatrix keys: {keys}"
    return "unknown", "unknown", "no imatrix metadata in arm; pass --weighted-q1-0 true|false"


# ---------------------------------------------------------------------------
# `stamp`
# ---------------------------------------------------------------------------


def _safe_output_path(out: Path, info: TiledInfo) -> Path:
    path = out.expanduser().resolve()
    if path == info.manifest_path:
        raise InputError("--out must not overwrite tiled/manifest.json")
    if path.parent == info.directory and path.name.startswith("layer_") and path.suffix == ".bin":
        raise InputError("--out must not overwrite a tiled layer file")
    return path


def stamp(args: argparse.Namespace) -> int:
    arm = _resolved_file(args.arm, "--arm")
    imatrix = _resolved_file(args.imatrix, "--imatrix") if args.imatrix else None
    rules = _resolved_file(args.rules, "--rules")
    quantizer = _resolved_file(args.quantizer_src, "--quantizer-src")
    try:
        info = load_tiled(args.tiled)
    except VerificationError as exc:
        raise InputError(f"invalid tiled package: {exc}") from exc

    out = _safe_output_path(args.out, info)

    arm_sha, arm_bytes = sha256_file(arm)
    imatrix_sha: Optional[str] = None
    imatrix_bytes: Optional[int] = None
    if imatrix is not None:
        imatrix_sha, imatrix_bytes = sha256_file(imatrix)
    rules_sha, rules_bytes = sha256_file(rules)
    quantizer_sha, quantizer_bytes = sha256_file(quantizer)
    manifest_sha = _sha256_text(info.manifest_raw)

    arm_imatrix, arm_imatrix_error = read_arm_imatrix_metadata(arm)
    if imatrix is None:
        # Without the imatrix file we cannot prove that the Q1_0 tensors were
        # weighted, so ``true`` is never a legal claim here.  ``auto`` records
        # ``unknown`` (even if the arm metadata mentions an imatrix, because
        # the referenced file is not part of this sidecar).
        if args.weighted_q1_0 == "true":
            raise InputError(
                "--weighted-q1-0 true requires --imatrix: without the imatrix "
                "file an unweighted package must not be labelled weighted"
            )
        if args.weighted_q1_0 == "false":
            weighted = "false"
            weighted_source = "explicit"
            weighted_why = "explicit --weighted-q1-0 false (no --imatrix)"
        else:
            weighted = "unknown"
            weighted_source = "missing-imatrix"
            weighted_why = (
                "no --imatrix given; cannot claim weighted_q1_0=true "
                "(pass --weighted-q1-0 false for a known-unweighted package)"
            )
    else:
        weighted, weighted_source, weighted_why = infer_weighted(
            args.weighted_q1_0, arm_imatrix
        )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm_sha256": arm_sha,
        "arm_bytes": arm_bytes,
        "imatrix_sha256": imatrix_sha,
        "imatrix_bytes": imatrix_bytes,
        "quantizer_source_sha256": quantizer_sha,
        "quantizer_source_bytes": quantizer_bytes,
        "rules_sha256": rules_sha,
        "rules_bytes": rules_bytes,
        "manifest_sha256": manifest_sha,
        "backend_layout": info.backend_layout,
        "entry_count": len(info.entries),
        "layer_count": len(info.layers),
        "expert_count": info.expert_count,
        "files_sha256": dict(sorted(info.files_sha256.items())),
        "weighted_q1_0": weighted,
        "weighted_q1_0_source": weighted_source,
        "weighted_q1_0_evidence": weighted_why,
        "arm_imatrix": arm_imatrix,
        "arm_imatrix_error": arm_imatrix_error,
        "source_paths": {
            "arm": str(arm),
            "imatrix": str(imatrix) if imatrix is not None else None,
            "rules": str(rules),
            "quantizer_source": str(quantizer),
        },
    }
    _atomic_write_json(out, record)

    imatrix_label = (
        f"{imatrix_sha[:16]}... ({imatrix_bytes} B)" if imatrix_sha else "none (null)"
    )
    print(
        f"stamped {out}: {len(info.entries)} entries / {len(info.layers)} layers, "
        f"backend_layout={info.backend_layout}, weighted_q1_0={weighted} ({weighted_source})"
    )
    print(
        f"  arm={arm_sha[:16]}... ({arm_bytes} B) imatrix={imatrix_label} "
        f"manifest={manifest_sha[:16]}..."
    )
    if arm_imatrix_error:
        print(f"warning: {arm_imatrix_error}; weighted_q1_0={weighted}", file=sys.stderr)
    if weighted == "unknown":
        print(
            "warning: weighted_q1_0 is unknown; pass --weighted-q1-0 true|false "
            "for a deployable claim",
            file=sys.stderr,
        )
    return EXIT_OK


# ---------------------------------------------------------------------------
# `verify`
# ---------------------------------------------------------------------------


def load_sidecar(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read provenance sidecar {path}: {exc}") from exc
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"provenance sidecar is not valid JSON: {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise VerificationError(f"provenance sidecar is not a JSON object: {path}")
    return record


def _check_sidecar(record: dict[str, Any], info: TiledInfo) -> None:
    if "schema_version" in record and record.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError(
            f"sidecar schema_version is {record.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    _require_str(record.get("tool_version"), "sidecar.tool_version")
    _require_str(record.get("created_at"), "sidecar.created_at")

    _require_sha256(record.get("arm_sha256"), "sidecar.arm_sha256")
    _require_sha256(record.get("quantizer_source_sha256"), "sidecar.quantizer_source_sha256")
    _require_sha256(record.get("rules_sha256"), "sidecar.rules_sha256")
    manifest_sha = _require_sha256(record.get("manifest_sha256"), "sidecar.manifest_sha256")
    _require_int(record.get("arm_bytes"), "sidecar.arm_bytes")
    # T-0088: an unweighted arm has no imatrix, so both fields may be null.
    # Old sidecars (with an imatrix) keep the exact sha256/integer contract.
    imatrix_sha = record.get("imatrix_sha256")
    imatrix_bytes = record.get("imatrix_bytes")
    if imatrix_sha is not None:
        _require_sha256(imatrix_sha, "sidecar.imatrix_sha256")
    if imatrix_bytes is not None:
        _require_int(imatrix_bytes, "sidecar.imatrix_bytes")
    if (imatrix_sha is None) != (imatrix_bytes is None):
        raise VerificationError(
            "sidecar.imatrix_sha256 and sidecar.imatrix_bytes must both be "
            "null or both be set"
        )
    if "rules_bytes" in record:
        _require_int(record.get("rules_bytes"), "sidecar.rules_bytes", 0)

    actual_manifest_sha = _sha256_text(info.manifest_raw)
    if manifest_sha != actual_manifest_sha:
        raise VerificationError(
            f"manifest sha256 mismatch: sidecar {manifest_sha}, "
            f"tiled/manifest.json {actual_manifest_sha}"
        )

    # The per-field copies below are optional.  A sidecar written by this tool
    # always has them, but a minimal sidecar is still valid as long as the
    # required provenance fields above are present; the manifest itself is
    # always validated and its layer hashes are always checked.
    if "backend_layout" in record and record["backend_layout"] != info.backend_layout:
        raise VerificationError(
            f"backend_layout mismatch: sidecar {record['backend_layout']!r}, "
            f"manifest {info.backend_layout!r}"
        )
    if "entry_count" in record:
        if _require_int(record["entry_count"], "sidecar.entry_count", 1) != len(info.entries):
            raise VerificationError(
                f"entry_count mismatch: sidecar {record['entry_count']}, "
                f"manifest {len(info.entries)}"
            )
    if "layer_count" in record:
        if _require_int(record["layer_count"], "sidecar.layer_count", 1) != len(info.layers):
            raise VerificationError(
                f"layer_count mismatch: sidecar {record['layer_count']}, "
                f"manifest {len(info.layers)}"
            )
    if "expert_count" in record:
        if _require_int(record["expert_count"], "sidecar.expert_count", 1) != info.expert_count:
            raise VerificationError(
                f"expert_count mismatch: sidecar {record['expert_count']}, "
                f"manifest {info.expert_count}"
            )

    if "files_sha256" in record:
        sidecar_files = _require_mapping(record["files_sha256"], "sidecar.files_sha256")
        normalized_files: dict[str, str] = {}
        for name, digest in sidecar_files.items():
            normalized_files[str(name)] = _require_sha256(
                digest, f"sidecar.files_sha256[{name}]"
            )
        if normalized_files != info.files_sha256:
            missing = sorted(set(info.files_sha256) - set(normalized_files))
            extra = sorted(set(normalized_files) - set(info.files_sha256))
            changed = sorted(
                name
                for name in set(info.files_sha256) & set(normalized_files)
                if info.files_sha256[name] != normalized_files[name]
            )
            raise VerificationError(
                "files_sha256 mismatch between sidecar and manifest; "
                f"missing={missing} extra={extra} changed={changed}"
            )

    weighted = record.get("weighted_q1_0")
    if weighted not in ("true", "false", "unknown"):
        raise VerificationError(
            f"sidecar.weighted_q1_0 must be true/false/unknown, got {weighted!r}"
        )
    if weighted == "true" and imatrix_sha is None:
        raise VerificationError(
            "sidecar.weighted_q1_0=true requires a non-null imatrix_sha256 "
            "(an unweighted/no-imatrix package cannot be labelled weighted)"
        )
    source = record.get("weighted_q1_0_source")
    if source is not None and not isinstance(source, str):
        raise VerificationError("sidecar.weighted_q1_0_source must be a string")
    arm_imatrix = record.get("arm_imatrix")
    if arm_imatrix is not None and not isinstance(arm_imatrix, dict):
        raise VerificationError("sidecar.arm_imatrix must be an object or null")
    error = record.get("arm_imatrix_error")
    if error is not None and not isinstance(error, str):
        raise VerificationError("sidecar.arm_imatrix_error must be a string or null")


def verify(args: argparse.Namespace) -> int:
    # Validate CLI-supplied hashes before touching the package so a malformed
    # argument is an input error (2), not a verification failure (3).
    try:
        expected_arm = (
            _require_sha256(args.expect_arm_sha, "--expect-arm-sha")
            if args.expect_arm_sha is not None
            else None
        )
        expected_imatrix = (
            _require_sha256(args.expect_imatrix_sha, "--expect-imatrix-sha")
            if args.expect_imatrix_sha is not None
            else None
        )
    except VerificationError as exc:
        raise InputError(str(exc)) from exc

    info = load_tiled(args.tiled)
    provenance_path = (
        args.provenance.expanduser().resolve()
        if args.provenance is not None
        else info.directory / DEFAULT_PROVENANCE_NAME
    )

    if not provenance_path.is_file():
        if args.require_provenance:
            raise VerificationError(
                f"provenance required but not found: {provenance_path}"
            )
        # Backward compatibility: an old package has no sidecar.  The path may
        # be explicit (the usual DIR/provenance.json) or the default; either
        # way it is a warning, not an error, unless --require-provenance.
        print(
            f"warning: no provenance sidecar at {provenance_path}; "
            "verifying tiled package for backward compatibility "
            "(pass --require-provenance to make this an error)",
            file=sys.stderr,
        )
        verify_layer_files(info)
        print(
            f"OK (legacy): {len(info.entries)} entries / {len(info.layers)} layers, "
            "layer sha256 values match manifest; provenance absent"
        )
        return EXIT_OK

    record = load_sidecar(provenance_path)
    _check_sidecar(record, info)
    verify_layer_files(info)

    if expected_arm is not None:
        actual = record["arm_sha256"].lower()
        if actual != expected_arm:
            raise VerificationError(
                f"arm sha256 mismatch: expected {expected_arm}, sidecar records {actual}"
            )
    if expected_imatrix is not None:
        actual_raw = record.get("imatrix_sha256")
        if actual_raw is None:
            raise VerificationError(
                "--expect-imatrix-sha was given but the sidecar records "
                "imatrix_sha256=null (unweighted/no-imatrix package)"
            )
        actual = actual_raw.lower()
        if actual != expected_imatrix:
            raise VerificationError(
                f"imatrix sha256 mismatch: expected {expected_imatrix}, "
                f"sidecar records {actual}"
            )

    weighted = record["weighted_q1_0"]
    print(
        f"OK: {len(info.entries)} entries / {len(info.layers)} layers, "
        f"backend_layout={info.backend_layout}, weighted_q1_0={weighted} "
        f"({record.get('weighted_q1_0_source', 'unknown')})"
    )
    imatrix_text = (
        f"{record['imatrix_sha256'][:16]}..."
        if record.get("imatrix_sha256")
        else "none (null)"
    )
    print(
        f"  arm={record['arm_sha256'][:16]}... imatrix={imatrix_text} "
        f"manifest={record['manifest_sha256'][:16]}..."
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXIT_CODES,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stamp_parser = subparsers.add_parser(
        "stamp",
        help="write DIR/provenance.json without touching tiled bytes",
        description=(
            "Hash the arm/imatrix/rules/quantizer inputs and tiled manifest, then "
            "write a provenance sidecar.  tiled/manifest.json and layer_*.bin are "
            "opened read-only and never modified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXIT_CODES,
    )
    stamp_parser.add_argument("--tiled", required=True, type=Path,
                              help="tiled package directory containing manifest.json")
    stamp_parser.add_argument("--arm", required=True, type=Path,
                              help="quantized arm GGUF that produced the tiled package")
    stamp_parser.add_argument("--imatrix", required=False, default=None, type=Path,
                              help="imatrix GGUF (or raw file) used by the arm; "
                                   "omit for an unweighted arm (sidecar records null)")
    stamp_parser.add_argument("--rules", required=True, type=Path,
                              help="llama-quantize --tensor-type-file rules.txt")
    stamp_parser.add_argument("--quantizer-src", type=Path, default=DEFAULT_QUANTIZER_SOURCE,
                              help="quantizer source to fingerprint "
                                   "(default: <repo>/ggml/src/ggml-quants.c)")
    stamp_parser.add_argument("--weighted-q1-0", choices=("auto", "true", "false"),
                              default="auto",
                              help="weighted Q1_0 flag; auto reads arm imatrix metadata "
                                   "and records 'unknown' if it cannot tell. Without "
                                   "--imatrix only false/unknown are legal")
    stamp_parser.add_argument("--out", required=True, type=Path,
                              help="sidecar output path, normally DIR/provenance.json")
    stamp_parser.set_defaults(func=stamp)

    verify_parser = subparsers.add_parser(
        "verify",
        help="check a tiled package against a provenance sidecar",
        description=(
            "Validate the sidecar, manifest sha256, backend_layout, entry/layer "
            "counts, and every layer file's sha256 by streaming.  With no "
            "--provenance, DIR/provenance.json is used; if it is absent the "
            "package is treated as legacy (warning + exit 0) unless "
            "--require-provenance is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXIT_CODES,
    )
    verify_parser.add_argument("--tiled", required=True, type=Path,
                               help="tiled package directory containing manifest.json")
    verify_parser.add_argument("--provenance", type=Path, default=None,
                               help="sidecar path (default: DIR/provenance.json)")
    verify_parser.add_argument("--expect-arm-sha", default=None,
                               help="require the sidecar's arm_sha256 to equal this sha256")
    verify_parser.add_argument("--expect-imatrix-sha", default=None,
                               help="require the sidecar's imatrix_sha256 to equal this sha256")
    verify_parser.add_argument("--require-provenance", action="store_true",
                               help="fail (exit 3) when no sidecar exists instead of warning")
    verify_parser.set_defaults(func=verify)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except VerificationError as exc:
        print(f"verification FAILED: {exc}", file=sys.stderr)
        return EXIT_VERIFY
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
