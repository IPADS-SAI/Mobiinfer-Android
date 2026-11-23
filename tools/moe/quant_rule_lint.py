#!/usr/bin/env python3
"""Static linter for llama.cpp ``--tensor-type-file`` quantization rules.

The linter reads only the GGUF header (metadata + tensor-info array).  It never
mmaps the model and never reads tensor data, so it is safe to run on a laptop
against a multi-GB checkpoint (a truncated header downloaded with ``head -c``
is enough).

Semantics mirrored from the C++ implementation
-----------------------------------------------
1. ``tools/quantize/quantize.cpp`` (``parse_tensor_type`` /
   ``parse_tensor_type_file``) reads whitespace-separated ``regex=GGML_TYPE``
   tokens and lowercases the regex text before compiling it.
2. ``src/llama-quant.cpp`` (``llama_tensor_get_type``) compiles the patterns
   once and then walks them in file order for every tensor using
   ``std::regex_search``.  The **first matching rule wins** and the loop
   ``break``s; later matching rules are shadowed.
3. Overrides are applied only when the caller's default type is a quantized
   type (``ggml_is_quantized(default_type)``), the tensor allows quantization,
   and ``--pure`` is off.  A tensor that does not allow quantization keeps its
   original GGUF type.  Shape-incompatible targets go through
   ``tensor_type_fallback`` (exact mapping reproduced below).

This tool deliberately does not model the non-manual ``ftype`` heuristics in
``llama_tensor_get_type_impl``.  If no rule matches, ``final_type`` is the
value passed to ``--default``; in a real ``llama-quantize`` run an ftype such
as ``Q4_K_M`` may choose a different per-tensor type before the fallback step.
The special ``--token-embedding-type`` / ``--output-tensor-type`` overrides are
not modelled either (both are off by default).

Patterns are compiled with Python's :mod:`re`, which is close to (but not
identical to) ``std::regex``'s ECMAScript grammar.  The rule syntax used by
this repository (literal names, ``.``, ``*``, character classes and
alternation) behaves the same.  As in C++, the regex text is lowercased before
compilation and matching is case-sensitive against the GGUF tensor name.

Byte estimates use ``ggml_row_size(type, ne0) * ne1 * ne2 * ne3`` with
``row_size = ne0 / block * block_bytes``.  They are header-only estimates:
GGUF metadata, padding/alignment and the output file's tensor-data layout are
not included.

Exit codes
----------
* 0 -- no diagnostics at all;
* 2 -- ``--strict`` and at least one warning (shadowed/unmatched/duplicate
  rule, impossible fallback, ...);
* 1 -- input error (bad GGUF, unreadable rules file, invalid regex, unknown
  GGML type, ...).

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# GGUF / GGML constants (kept locally: gguf-py imports numpy)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
SUPPORTED_GGUF_VERSIONS = (2, 3)
GGUF_DEFAULT_ALIGNMENT = 32

# GGUFValueType
GGUF_UINT8 = 0
GGUF_INT8 = 1
GGUF_UINT16 = 2
GGUF_INT16 = 3
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10
GGUF_INT64 = 11
GGUF_FLOAT64 = 12

GGUF_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    GGUF_UINT8: ("<B", 1),
    GGUF_INT8: ("<b", 1),
    GGUF_UINT16: ("<H", 2),
    GGUF_INT16: ("<h", 2),
    GGUF_UINT32: ("<I", 4),
    GGUF_INT32: ("<i", 4),
    GGUF_FLOAT32: ("<f", 4),
    GGUF_BOOL: ("<?", 1),
    GGUF_UINT64: ("<Q", 8),
    GGUF_INT64: ("<q", 8),
    GGUF_FLOAT64: ("<d", 8),
}

GGUF_TYPE_NAMES = {
    GGUF_UINT8: "UINT8",
    GGUF_INT8: "INT8",
    GGUF_UINT16: "UINT16",
    GGUF_INT16: "INT16",
    GGUF_UINT32: "UINT32",
    GGUF_INT32: "INT32",
    GGUF_FLOAT32: "FLOAT32",
    GGUF_BOOL: "BOOL",
    GGUF_STRING: "STRING",
    GGUF_ARRAY: "ARRAY",
    GGUF_UINT64: "UINT64",
    GGUF_INT64: "INT64",
    GGUF_FLOAT64: "FLOAT64",
}


@dataclass(frozen=True)
class GGMLType:
    """One ``ggml_type`` entry: block size and bytes per block."""

    type_id: int
    name: str
    block: int
    type_size: int

    @property
    def quantized(self) -> bool:
        # Every type with a block larger than one is a quantized type in this
        # tree; I8/I16/I32/I64/F32/F16/BF16 are not.
        return self.block > 1


# Reference: gguf-py/gguf/constants.py::GGML_QUANT_SIZES and
# ggml/include/ggml.h.  Kept in sync by tests for the types used by this fork.
_GGML_TYPE_ROWS = [
    # id, name, block, bytes/block
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 40),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (39, "MXFP4", 32, 17),
    (40, "NVFP4", 64, 36),
    (41, "Q1_0", 128, 18),
    (42, "Q2_0", 64, 18),
]

GGML_TYPES: dict[int, GGMLType] = {
    type_id: GGMLType(type_id, name, block, type_size)
    for type_id, name, block, type_size in _GGML_TYPE_ROWS
}
GGML_TYPES_BY_NAME: dict[str, GGMLType] = {
    t.name.upper(): t for t in GGML_TYPES.values()
}

# src/llama-quant.cpp::tensor_type_fallback
_FALLBACK_TO: dict[str, str] = {
    "IQ1_S": "IQ4_NL",
    "IQ1_M": "IQ4_NL",
    "IQ2_XXS": "IQ4_NL",
    "IQ2_XS": "IQ4_NL",
    "IQ2_S": "IQ4_NL",
    "IQ3_XXS": "IQ4_NL",
    "IQ3_S": "IQ4_NL",
    "IQ4_XS": "IQ4_NL",
    "Q2_0": "Q4_0",
    "Q2_K": "Q4_0",
    "Q3_K": "Q4_0",
    "TQ1_0": "Q4_0",
    "TQ2_0": "Q4_0",
    "Q4_K": "Q5_0",
    "Q5_K": "Q5_1",
    "Q6_K": "Q8_0",
}


# ---------------------------------------------------------------------------
# Errors and GGUF parsing
# ---------------------------------------------------------------------------


class LintError(Exception):
    """Fatal input error (bad GGUF, unreadable file, invalid CLI value)."""


@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: tuple[int, ...]  # GGUF order: ne[0], ne[1], ...
    type_id: int
    offset: int

    @property
    def shape(self) -> list[int]:
        return list(self.dims)

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def type(self) -> GGMLType:
        return GGML_TYPES[self.type_id]


@dataclass
class GGUFHeader:
    path: str
    file_size: int
    version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    header_end: int
    data_start: int
    warnings: list[str] = field(default_factory=list)


def _remaining(f, file_size: int) -> int:
    return file_size - f.tell()


def _read_exact(f, n: int, file_size: int, what: str) -> bytes:
    if n < 0:
        raise LintError(f"negative size while reading {what}")
    pos = f.tell()
    if n > file_size - pos:
        raise LintError(
            f"truncated GGUF while reading {what} at offset {pos}: "
            f"need {n} bytes, only {file_size - pos} left"
        )
    data = f.read(n)
    if len(data) != n:
        raise LintError(f"short read while reading {what} at offset {pos}")
    return data


def _skip_exact(f, n: int, file_size: int, what: str) -> None:
    if n < 0:
        raise LintError(f"negative size while skipping {what}")
    pos = f.tell()
    if n > file_size - pos:
        raise LintError(
            f"truncated GGUF while skipping {what} at offset {pos}: "
            f"need {n} bytes, only {file_size - pos} left"
        )
    f.seek(n, os.SEEK_CUR)


def _read_u32(f, file_size: int, what: str) -> int:
    return struct.unpack("<I", _read_exact(f, 4, file_size, what))[0]


def _read_u64(f, file_size: int, what: str) -> int:
    return struct.unpack("<Q", _read_exact(f, 8, file_size, what))[0]


def _read_string(f, file_size: int, what: str, max_len: int = 1 << 26) -> str:
    n = _read_u64(f, file_size, f"{what} length")
    if n > max_len:
        raise LintError(f"{what} at offset {f.tell() - 8} is too long ({n} bytes)")
    raw = _read_exact(f, n, file_size, what)
    return raw.decode("utf-8", errors="replace")


def _skip_string(f, file_size: int, what: str) -> None:
    n = _read_u64(f, file_size, f"{what} length")
    _skip_exact(f, n, file_size, what)


def _skip_gguf_value(f, value_type: int, file_size: int, depth: int = 0) -> None:
    """Skip one metadata value without retaining it."""
    if depth > 8:
        raise LintError("GGUF metadata arrays are nested too deeply")
    if value_type in GGUF_SCALAR_FORMATS:
        _skip_exact(f, GGUF_SCALAR_FORMATS[value_type][1], file_size,
                    f"metadata {GGUF_TYPE_NAMES[value_type]}")
        return
    if value_type == GGUF_STRING:
        _skip_string(f, file_size, "metadata string")
        return
    if value_type != GGUF_ARRAY:
        raise LintError(
            f"unknown GGUF metadata value type {value_type} at offset {f.tell() - 4}; "
            "cannot determine its size (known types are 0..12)"
        )

    pos = f.tell()
    elem_type = _read_u32(f, file_size, "array element type")
    count = _read_u64(f, file_size, "array count")
    if elem_type in GGUF_SCALAR_FORMATS:
        _skip_exact(f, count * GGUF_SCALAR_FORMATS[elem_type][1], file_size,
                    "metadata scalar array")
        return
    if elem_type == GGUF_STRING:
        # Every element is at least an 8-byte length; reject absurd counts
        # before entering a Python loop.
        if count * 8 > _remaining(f, file_size):
            raise LintError(
                f"GGUF string array at offset {pos} claims {count} elements, "
                "which does not fit in the remaining file"
            )
        for _ in range(count):
            _skip_string(f, file_size, "metadata string array item")
        return
    if elem_type == GGUF_ARRAY:
        if count * 12 > _remaining(f, file_size):
            raise LintError(
                f"GGUF nested array at offset {pos} claims {count} elements, "
                "which does not fit in the remaining file"
            )
        for _ in range(count):
            _skip_gguf_value(f, GGUF_ARRAY, file_size, depth + 1)
        return
    raise LintError(
        f"unknown GGUF metadata array element type {elem_type} at offset {pos}; "
        "cannot determine element size"
    )


def _interesting_key(key: str) -> bool:
    return (
        key in {
            "general.architecture",
            "general.alignment",
            "general.file_type",
            "general.quantization_version",
        }
        or key.endswith(".block_count")
        or key.endswith(".expert_count")
        or key.endswith(".expert_used_count")
    )


def _read_metadata_value(f, value_type: int, file_size: int) -> Any:
    if value_type in GGUF_SCALAR_FORMATS:
        fmt, size = GGUF_SCALAR_FORMATS[value_type]
        return struct.unpack(fmt, _read_exact(f, size, file_size, "metadata value"))[0]
    if value_type == GGUF_STRING:
        return _read_string(f, file_size, "metadata string value")
    # Arrays are not interesting for this tool.
    _skip_gguf_value(f, value_type, file_size)
    return None


def parse_gguf_header(path: str | os.PathLike[str]) -> GGUFHeader:
    """Parse a GGUF header without reading tensor data or using mmap."""
    path = os.fspath(path)
    try:
        file_size = os.stat(path).st_size
        f = open(path, "rb")
    except OSError as exc:
        raise LintError(f"cannot open model '{path}': {exc}") from exc

    warnings: list[str] = []
    with f:
        magic_raw = _read_exact(f, 4, file_size, "magic")
        magic = struct.unpack("<I", magic_raw)[0]
        if magic != GGUF_MAGIC:
            if struct.unpack(">I", magic_raw)[0] == GGUF_MAGIC:
                raise LintError("big-endian GGUF files are not supported")
            raise LintError(
                f"bad GGUF magic in '{path}': {magic_raw!r} (expected b'GGUF')"
            )

        version = _read_u32(f, file_size, "version")
        if version not in SUPPORTED_GGUF_VERSIONS:
            raise LintError(
                f"unsupported GGUF version {version} in '{path}' "
                f"(supported: {', '.join(map(str, SUPPORTED_GGUF_VERSIONS))})"
            )

        tensor_count = _read_u64(f, file_size, "tensor count")
        metadata_count = _read_u64(f, file_size, "metadata count")

        # Cheap sanity checks before any loops.
        if tensor_count * 24 > _remaining(f, file_size):
            raise LintError(
                f"GGUF claims {tensor_count} tensors, which cannot fit in the "
                f"remaining {_remaining(f, file_size)} header bytes"
            )
        if metadata_count * 13 > _remaining(f, file_size):
            raise LintError(
                f"GGUF claims {metadata_count} metadata entries, which cannot "
                f"fit in the remaining {_remaining(f, file_size)} header bytes"
            )

        metadata: dict[str, Any] = {}
        for i in range(metadata_count):
            key = _read_string(f, file_size, f"metadata key #{i}", max_len=1 << 20)
            value_type = _read_u32(f, file_size, f"metadata type for '{key}'")
            if _interesting_key(key):
                metadata[key] = _read_metadata_value(f, value_type, file_size)
            else:
                _skip_gguf_value(f, value_type, file_size)

        alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or alignment <= 0 or (alignment & (alignment - 1)):
            warnings.append(
                f"general.alignment={alignment!r} is not a non-zero power of two; "
                f"using {GGUF_DEFAULT_ALIGNMENT}"
            )
            alignment = GGUF_DEFAULT_ALIGNMENT

        tensors: list[TensorInfo] = []
        seen_names: set[str] = set()
        for i in range(tensor_count):
            name = _read_string(f, file_size, f"tensor name #{i}")
            if not name:
                raise LintError(f"tensor #{i} has an empty name")
            if name in seen_names:
                warnings.append(f"duplicate tensor name '{name}' (tensor #{i})")
            seen_names.add(name)
            n_dims = _read_u32(f, file_size, f"n_dims for tensor '{name}'")
            if n_dims == 0 or n_dims > 8:
                raise LintError(
                    f"tensor '{name}' has unsupported n_dims={n_dims} "
                    "(expected 1..8)"
                )
            dims = tuple(
                _read_u64(f, file_size, f"dim {j} for tensor '{name}'")
                for j in range(n_dims)
            )
            if any(d == 0 for d in dims):
                raise LintError(f"tensor '{name}' has a zero dimension: {dims}")
            type_id = _read_u32(f, file_size, f"ggml_type for tensor '{name}'")
            if type_id not in GGML_TYPES:
                raise LintError(
                    f"tensor '{name}' uses unknown ggml_type id {type_id} "
                    "(this build knows ids 0..42 with gaps for removed types)"
                )
            offset = _read_u64(f, file_size, f"offset for tensor '{name}'")
            tensors.append(TensorInfo(name, n_dims, dims, type_id, offset))

        header_end = f.tell()

    data_start = (header_end + alignment - 1) // alignment * alignment
    return GGUFHeader(
        path=path,
        file_size=file_size,
        version=version,
        tensor_count=len(tensors),
        metadata_count=metadata_count,
        alignment=alignment,
        metadata=metadata,
        tensors=tensors,
        header_end=header_end,
        data_start=data_start,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    index: int
    file: str
    line: int
    raw_pattern: str
    pattern: str  # lowercased, as in C++ parse_tensor_type()
    type_name: str
    type_id: Optional[int]
    regex: Optional[re.Pattern[str]] = None
    valid: bool = True
    error: Optional[str] = None
    duplicate_of: Optional[int] = None

    @property
    def ggml_type(self) -> Optional[GGMLType]:
        return GGML_TYPES.get(self.type_id) if self.type_id is not None else None

    def search(self, name: str) -> bool:
        return bool(self.regex is not None and self.regex.search(name))


def _strip_comment(line: str) -> str:
    """Drop a ``#`` comment; ``#`` inside a regex is kept unless whitespace-led."""
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


_GENERIC_FAMILY_TOKENS = {
    "blk", "block", "blocks", "layer", "layers", "weight", "bias",
    "exps", "expert", "experts", "ffn", "attn", "attention", "token",
    "tokens", "embd", "embed", "output", "input", "norm", "ssm", "conv",
    "proj", "linear", "model", "shexp", "shared", "gate", "up", "down",
}


def _rule_literal_family_tokens(pattern: str) -> list[str]:
    """Literal regex fragments that look like a tensor-family name."""
    tokens = [
        tok for tok in re.split(r"[^A-Za-z0-9_]+", pattern)
        if len(tok) >= 3 and any(c.isalpha() for c in tok)
    ]
    return [tok for tok in tokens if tok.lower() not in _GENERIC_FAMILY_TOKENS]


def _looks_like_ftype(name: str) -> bool:
    return bool(re.fullmatch(r"Q\d+_K_[SML]|Q\d+_K", name.upper()))


def parse_rule_text(
    text: str,
    *,
    source: str,
    start_index: int = 0,
) -> list[Rule]:
    """Parse one ``name_regex=Q_TYPE`` rule file body."""
    rules: list[Rule] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = _strip_comment(line).strip()
        if not stripped:
            continue
        for token in stripped.split():
            index = start_index + len(rules)
            if "=" not in token:
                rules.append(Rule(
                    index=index, file=source, line=line_no,
                    raw_pattern=token, pattern=token.lower(),
                    type_name="", type_id=None, valid=False,
                    error="malformed rule: expected name_regex=GGML_TYPE",
                ))
                continue
            raw_pattern, raw_type = token.split("=", 1)
            if not raw_pattern:
                rules.append(Rule(
                    index=index, file=source, line=line_no,
                    raw_pattern=raw_pattern, pattern="",
                    type_name=raw_type, type_id=None, valid=False,
                    error="missing tensor name regex before '='",
                ))
                continue
            if not raw_type:
                rules.append(Rule(
                    index=index, file=source, line=line_no,
                    raw_pattern=raw_pattern, pattern=raw_pattern.lower(),
                    type_name="", type_id=None, valid=False,
                    error="missing quantization type after '='",
                ))
                continue

            pattern = raw_pattern.lower()
            ggml_type = GGML_TYPES_BY_NAME.get(raw_type.strip().upper())
            if ggml_type is None:
                hint = ""
                if _looks_like_ftype(raw_type):
                    hint = (
                        " (this is a llama.cpp file type, not a ggml tensor type; "
                        "use e.g. Q4_K)"
                    )
                rules.append(Rule(
                    index=index, file=source, line=line_no,
                    raw_pattern=raw_pattern, pattern=pattern,
                    type_name=raw_type, type_id=None, valid=False,
                    error=f"unknown GGML type '{raw_type}'{hint}",
                ))
                continue

            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                rules.append(Rule(
                    index=index, file=source, line=line_no,
                    raw_pattern=raw_pattern, pattern=pattern,
                    type_name=ggml_type.name, type_id=ggml_type.type_id,
                    valid=False,
                    error=f"invalid regex: {exc}",
                ))
                continue

            rules.append(Rule(
                index=index, file=source, line=line_no,
                raw_pattern=raw_pattern, pattern=pattern,
                type_name=ggml_type.name, type_id=ggml_type.type_id,
                regex=compiled,
            ))
    return rules


def parse_rule_file(path: str) -> list[Rule]:
    if path == "-":
        return parse_rule_text(sys.stdin.read(), source="<stdin>")
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LintError(f"cannot read rules file '{path}': {exc}") from exc
    return parse_rule_text(text, source=path)


# ---------------------------------------------------------------------------
# llama-quant.cpp semantics
# ---------------------------------------------------------------------------

# Fragments excluded by tensor_allows_quantization().  The exact C++ function
# also checks a couple of architecture-dependent tensor names; these literal
# fragments cover the current tree.
_NOT_QUANTIZED_FRAGMENTS = (
    "_norm.weight",
    "ffn_gate_inp.weight",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "ssm_conv1d",
    "shortconv.conv.weight",
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
    ".position_embd",
    "sam.pos_embd",
    "sam.neck.",
    "sam.net_",
    ".rel_pos",
    ".patch_embd",
    ".patch_merger",
)


def tensor_allows_quantization(tensor: TensorInfo) -> bool:
    """Approximate ``tensor_allows_quantization`` for the default CLI flags.

    Assumes ``--only-copy`` and ``--quantize-output-tensor`` are off, matching
    llama-quantize defaults.
    """
    if tensor.n_dims < 2:
        return False
    name = tensor.name
    if not name.endswith("weight"):
        return False
    for fragment in _NOT_QUANTIZED_FRAGMENTS:
        if fragment in name:
            return False
    if name == "output.weight":
        return False
    if name in {"position_embd.weight", "token_types.weight"}:
        return False
    return True


def row_size(ggml_type: GGMLType, ne0: int) -> int:
    """``ggml_row_size``: exact block arithmetic, raises if incompatible."""
    if ne0 % ggml_type.block != 0:
        raise ValueError(
            f"ne0={ne0} is not divisible by {ggml_type.name} block size {ggml_type.block}"
        )
    return (ne0 // ggml_type.block) * ggml_type.type_size


def estimate_bytes(ggml_type: GGMLType, dims: Iterable[int]) -> int:
    """Header-only byte estimate for a tensor with GGUF dims ``[ne0, ne1, ...]``."""
    dims = tuple(dims)
    if not dims:
        raise ValueError("tensor has no dimensions")
    total = row_size(ggml_type, dims[0])
    for d in dims[1:]:
        total *= d
    return total


@dataclass
class TensorPlan:
    tensor: TensorInfo
    allows_quantization: bool
    default_type: str
    matched_rule: Optional[int]
    rule_applied: bool
    final_type: str
    fallback_from: Optional[str]
    bytes: Optional[int]
    original_type: str

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.tensor.name,
            "n_dims": self.tensor.n_dims,
            "shape": self.tensor.shape,
            "n_elements": self.tensor.n_elements,
            "ggml_type_id": self.tensor.type_id,
            "ggml_type": self.original_type,
            "allows_quantization": self.allows_quantization,
            "default_type": self.default_type,
            "matched_rule": self.matched_rule,
            "rule_applied": self.rule_applied,
            "final_type": self.final_type,
            "fallback_from": self.fallback_from,
            "estimated_bytes": self.bytes,
        }


@dataclass
class Diagnostic:
    severity: str  # "warning" | "error"
    code: str
    message: str
    rule_index: Optional[int] = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.rule_index is not None:
            out["rule_index"] = self.rule_index
        return out

    def render(self) -> str:
        prefix = "ERROR" if self.severity == "error" else "WARNING"
        rule = f" rule #{self.rule_index}" if self.rule_index is not None else ""
        return f"[{prefix}]{rule} {self.code}: {self.message}"


@dataclass
class LintReport:
    model: GGUFHeader
    default_type: GGMLType
    rules: list[Rule]
    plans: list[TensorPlan]
    diagnostics: list[Diagnostic]
    strict: bool
    overrides_enabled: bool
    rule_stats: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def warnings(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "warning")

    @property
    def errors(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "error")

    @property
    def exit_code(self) -> int:
        if self.errors:
            return 1
        if self.strict and self.warnings:
            return 2
        return 0

    def as_json(self) -> dict[str, Any]:
        return {
            "tool": "quant_rule_lint",
            "schema_version": 1,
            "status": "error" if self.errors else "ok",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "path": self.model.path,
                "file_size": self.model.file_size,
                "gguf_version": self.model.version,
                "tensor_count": self.model.tensor_count,
                "metadata_count": self.model.metadata_count,
                "alignment": self.model.alignment,
                "header_end": self.model.header_end,
                "data_start": self.model.data_start,
                "metadata": self.model.metadata,
                "tensor_data_read": False,
                "byte_estimate_note": (
                    "header-only estimate from ggml_row_size(type, ne0) * "
                    "ne1 * ne2 * ne3; excludes metadata, padding and file layout"
                ),
            },
            "default_type": {
                "name": self.default_type.name,
                "type_id": self.default_type.type_id,
                "block": self.default_type.block,
                "type_size": self.default_type.type_size,
                "quantized": self.default_type.quantized,
            },
            "overrides_enabled": self.overrides_enabled,
            "rules": self.rule_stats,
            "tensors": [p.as_json() for p in self.plans],
            "summary": self.summary,
            "diagnostics": [d.as_json() for d in self.diagnostics],
            "warning_count": self.warnings,
            "error_count": self.errors,
            "strict": self.strict,
            "exit_code": self.exit_code,
        }


def _format_bytes(n: Optional[int]) -> str:
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def _fallback_chain(target: GGMLType, ne0: int) -> tuple[GGMLType, Optional[str], list[Diagnostic]]:
    """Reproduce tensor_type_fallback(); returns (final, fallback_from, diags)."""
    diags: list[Diagnostic] = []
    if ne0 % target.block == 0:
        return target, None, diags

    fallback_name = _FALLBACK_TO.get(target.name)
    if fallback_name is None:
        diags.append(Diagnostic(
            "error", "fallback_unsupported",
            f"{target.name} needs ne0 divisible by {target.block}, got {ne0}; "
            "llama-quant.cpp has no fallback for this type and would abort",
        ))
        return target, None, diags

    fallback = GGML_TYPES_BY_NAME[fallback_name]
    if ne0 % fallback.block != 0:
        diags.append(Diagnostic(
            "warning", "fallback_f16",
            f"{target.name} -> {fallback.name} still incompatible with ne0={ne0}; "
            "llama-quant.cpp would use F16",
        ))
        return GGML_TYPES_BY_NAME["F16"], target.name, diags

    diags.append(Diagnostic(
        "warning", "shape_fallback",
        f"{target.name} needs ne0 divisible by {target.block}, got {ne0}; "
        f"llama-quant.cpp would fall back to {fallback.name}",
    ))
    return fallback, target.name, diags


def lint(
    model: GGUFHeader,
    rules: list[Rule],
    default_type: GGMLType,
    *,
    strict: bool = False,
) -> LintReport:
    diagnostics: list[Diagnostic] = [
        Diagnostic("warning", "gguf_warning", msg) for msg in model.warnings
    ]
    overrides_enabled = default_type.quantized
    if not overrides_enabled:
        diagnostics.append(Diagnostic(
            "warning", "overrides_ignored_default_not_quantized",
            f"--default {default_type.name} is not a quantized type; llama-quantize "
            "ignores --tensor-type-file overrides unless the default type is quantized",
        ))

    # Rule-level diagnostics from parsing.
    for rule in rules:
        if not rule.valid:
            diagnostics.append(Diagnostic(
                "error", "invalid_regex" if "regex" in (rule.error or "") else "invalid_rule",
                f"{rule.file}:{rule.line}: {rule.error} "
                f"(token: {rule.raw_pattern}={rule.type_name})",
                rule.index,
            ))

    # Duplicate exact rules.
    seen: dict[tuple[str, Optional[int]], int] = {}
    for rule in rules:
        if not rule.valid:
            continue
        key = (rule.pattern, rule.type_id)
        if key in seen:
            rule.duplicate_of = seen[key]
            diagnostics.append(Diagnostic(
                "warning", "duplicate_rule",
                f"{rule.file}:{rule.line}: rule #{rule.index} duplicates rule "
                f"#{seen[key]} ('{rule.raw_pattern}={rule.type_name}')",
                rule.index,
            ))
        else:
            seen[key] = rule.index

    # Per-tensor first-match resolution.
    allowed_by_name: dict[str, bool] = {
        t.name: tensor_allows_quantization(t) for t in model.tensors
    }
    matched_by_rule: list[list[TensorInfo]] = [[] for _ in rules]
    winning_rule_by_tensor: dict[str, int] = {}

    for tensor in model.tensors:
        if not allowed_by_name[tensor.name]:
            continue
        winner: Optional[int] = None
        for rule in rules:
            if rule.valid and rule.search(tensor.name):
                # Record every regex hit for shadow/usage diagnostics, but the
                # first hit is the only one that can determine the final type.
                matched_by_rule[rule.index].append(tensor)
                if winner is None:
                    winner = rule.index
        if winner is not None:
            winning_rule_by_tensor[tensor.name] = winner

    plans: list[TensorPlan] = []
    for tensor in model.tensors:
        allowed = allowed_by_name[tensor.name]
        matched_rule = winning_rule_by_tensor.get(tensor.name) if allowed else None
        rule = rules[matched_rule] if matched_rule is not None else None
        fallback_from: Optional[str] = None
        final = tensor.type
        rule_applied = bool(
            allowed and overrides_enabled and rule is not None and rule.ggml_type
        )

        if allowed:
            if rule is not None and rule.ggml_type is not None:
                target = rule.ggml_type
            else:
                target = default_type
            if overrides_enabled:
                final, fallback_from, fb_diags = _fallback_chain(target, tensor.dims[0])
                for d in fb_diags:
                    d.message = f"tensor '{tensor.name}': {d.message}"
                    if tensor.name in winning_rule_by_tensor:
                        d.rule_index = winning_rule_by_tensor[tensor.name]
                diagnostics.extend(fb_diags)
            else:
                final = default_type
        else:
            # C++ returns tensor->type before the rule loop.
            final = tensor.type

        try:
            nbytes: Optional[int] = estimate_bytes(final, tensor.dims)
        except ValueError as exc:
            diagnostics.append(Diagnostic(
                "error", "byte_estimate_failed",
                f"tensor '{tensor.name}': {exc}",
            ))
            nbytes = None

        plans.append(TensorPlan(
            tensor=tensor,
            allows_quantization=allowed,
            default_type=default_type.name,
            matched_rule=matched_rule if rule_applied else None,
            rule_applied=rule_applied,
            final_type=final.name,
            fallback_from=fallback_from,
            bytes=nbytes,
            original_type=tensor.type.name,
        ))

    # Rule usage / unmatched / shadow diagnostics.
    rule_stats: list[dict[str, Any]] = []
    for rule in rules:
        matched = matched_by_rule[rule.index] if rule.valid else []
        won = 0
        if rule.valid:
            won = sum(
                1 for t in matched
                if winning_rule_by_tensor.get(t.name) == rule.index
                and overrides_enabled
                and allowed_by_name[t.name]
            )
        shadowed = False
        if rule.valid and matched:
            shadowed = all(
                winning_rule_by_tensor.get(t.name) != rule.index for t in matched
            )
            if shadowed:
                diagnostics.append(Diagnostic(
                    "warning", "rule_shadowed",
                    f"{rule.file}:{rule.line}: rule #{rule.index} "
                    f"('{rule.raw_pattern}={rule.type_name}') is fully shadowed by an "
                    f"earlier rule for all {len(matched)} tensor(s) it matches",
                    rule.index,
                ))
        elif rule.valid and not matched:
            # Distinguish "matches no name at all" from "matches only tensors
            # that llama-quantize would never quantize".
            ineligible = [
                t.name for t in model.tensors
                if rule.search(t.name) and not allowed_by_name[t.name]
            ]
            detail = ""
            if ineligible:
                sample = ", ".join(ineligible[:3])
                detail = (
                    f" (it matches {len(ineligible)} tensor name(s) that do not "
                    f"allow quantization, e.g. {sample})"
                )
            diagnostics.append(Diagnostic(
                "warning", "rule_unmatched",
                f"{rule.file}:{rule.line}: rule #{rule.index} "
                f"('{rule.raw_pattern}={rule.type_name}') never matches a "
                f"quantization-eligible tensor{detail}",
                rule.index,
            ))
            shadowed = False

        # Heuristic for a misspelled/nonexistent tensor family: look at the
        # non-generic literal fragments in the regex and see whether any occurs
        # in a model tensor name.
        unknown_family = False
        if rule.valid and not matched:
            literals = _rule_literal_family_tokens(rule.pattern)
            if literals and not any(
                tok in tensor.name for tok in literals for tensor in model.tensors
            ):
                unknown_family = True
                diagnostics.append(Diagnostic(
                    "warning", "rule_unknown_family",
                    f"{rule.file}:{rule.line}: rule #{rule.index} literal token(s) "
                    f"{literals!r} do not occur in any tensor name; the rule may "
                    "target a nonexistent tensor family",
                    rule.index,
                ))

        rule_stats.append({
            "index": rule.index,
            "file": rule.file,
            "line": rule.line,
            "raw_pattern": rule.raw_pattern,
            "pattern": rule.pattern,
            "type": rule.type_name,
            "type_id": rule.type_id,
            "valid": rule.valid,
            "error": rule.error,
            "duplicate_of": rule.duplicate_of,
            "matched_tensors": len(matched),
            "won_tensors": won,
            "shadowed": shadowed,
            "unmatched": bool(rule.valid and not matched),
            "unknown_family": unknown_family,
        })

    # Summary by final type.
    by_type: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for plan in plans:
        entry = by_type.setdefault(plan.final_type, {"tensors": 0, "bytes": 0})
        entry["tensors"] += 1
        if plan.bytes is not None:
            entry["bytes"] += plan.bytes
            total_bytes += plan.bytes
    for name, entry in by_type.items():
        entry["human_bytes"] = _format_bytes(entry["bytes"])

    # Expert coverage: blk.<N>.ffn_{gate,up,down}_exps.weight.
    expert_re = re.compile(
        r"^blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight$"
    )
    expert_entries: list[dict[str, Any]] = []
    layers: set[int] = set()
    families: set[str] = set()
    for plan in plans:
        m = expert_re.match(plan.tensor.name)
        if not m:
            continue
        layer = int(m.group(1))
        family = m.group(2)
        layers.add(layer)
        families.add(family)
        covered = plan.rule_applied and plan.matched_rule is not None
        expert_entries.append({
            "name": plan.tensor.name,
            "layer": layer,
            "family": family,
            "matched_rule": plan.matched_rule,
            "final_type": plan.final_type,
            "estimated_bytes": plan.bytes,
            "covered_by_rule": covered,
        })
    uncovered = [
        e["name"] for e in expert_entries if not e["covered_by_rule"]
    ]
    expert_coverage: dict[str, Any] = {
        "pattern": "blk.<N>.ffn_(gate|up|down)_exps.weight",
        "tensor_count": len(expert_entries),
        "layer_count": len(layers),
        "layers": sorted(layers),
        "families": sorted(families),
        "covered_by_rule": sum(
            1 for e in expert_entries if e["covered_by_rule"]
        ),
        "uncovered": uncovered,
        "entries": expert_entries,
    }
    # Warn only when the rule set visibly tries to target expert tensors but
    # leaves some of them on --default.  A rules file that says nothing about
    # experts is not wrong just because the default type applies to them.
    expert_rules = [
        stat for stat in rule_stats
        if stat["valid"] and any(
            entry["matched_rule"] == stat["index"] for entry in expert_entries
        )
    ]
    if expert_entries and uncovered and expert_rules:
        diagnostics.append(Diagnostic(
            "warning", "expert_tensor_uncovered",
            f"{len(uncovered)} of {len(expert_entries)} expert tensor(s) fall back "
            f"to the default type instead of a rule; first: {uncovered[:5]}",
        ))

    # Recompute warning/error counts after all diagnostics are appended.
    summary = {
        "tensor_count": len(plans),
        "estimated_tensor_bytes": total_bytes,
        "human_estimated_tensor_bytes": _format_bytes(total_bytes),
        "by_final_type": dict(sorted(
            by_type.items(), key=lambda kv: kv[1]["bytes"], reverse=True
        )),
        "expert_coverage": expert_coverage,
        "quantization_eligible_tensors": sum(
            1 for p in plans if p.allows_quantization
        ),
        "rules_matched_tensors": sum(s["won_tensors"] for s in rule_stats),
    }

    return LintReport(
        model=model,
        default_type=default_type,
        rules=rules,
        plans=plans,
        diagnostics=diagnostics,
        strict=strict,
        overrides_enabled=overrides_enabled,
        rule_stats=rule_stats,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Rendering / CLI
# ---------------------------------------------------------------------------


def render_human(report: LintReport) -> str:
    lines: list[str] = []
    m = report.model
    lines.append("quant_rule_lint: GGUF header + --tensor-type-file preview")
    lines.append(
        "byte estimates are header-only (no tensor data read): "
        "row_size(type, ne0) * ne1 * ne2 * ne3"
    )
    lines.append("")
    lines.append(f"model   : {m.path} ({_format_bytes(m.file_size)})")
    lines.append(
        f"gguf    : version {m.version}, tensors {m.tensor_count}, "
        f"metadata {m.metadata_count}, alignment {m.alignment}, "
        f"header_end {m.header_end}, data_start {m.data_start}"
    )
    arch = m.metadata.get("general.architecture")
    if arch:
        block_count = m.metadata.get(f"{arch}.block_count")
        lines.append(
            f"arch    : {arch}" + (f", block_count {block_count}" if block_count else "")
        )
    lines.append(
        f"default : {report.default_type.name} "
        f"(block {report.default_type.block}, "
        f"{'quantized' if report.default_type.quantized else 'not quantized'})"
    )
    lines.append(
        f"rules   : {len(report.rules)} total, "
        f"{sum(1 for r in report.rules if r.valid)} valid, "
        f"overrides {'enabled' if report.overrides_enabled else 'DISABLED'}"
    )
    lines.append("")
    lines.append("per-tensor plan (rule = 0-based index; * = shape fallback):")
    lines.append(
        f"  {'name':<46} {'shape':<18} {'orig':<6} {'rule':<5} {'final':<7} {'bytes':>14}"
    )
    for plan in report.plans:
        rule = "-" if plan.matched_rule is None else str(plan.matched_rule)
        final = plan.final_type + ("*" if plan.fallback_from else "")
        shape = "[" + ", ".join(str(d) for d in plan.tensor.dims) + "]"
        lines.append(
            f"  {plan.tensor.name:<46} {shape:<18} {plan.original_type:<6} "
            f"{rule:<5} {final:<7} {_format_bytes(plan.bytes):>14}"
        )

    lines.append("")
    lines.append("rule usage:")
    if not report.rules:
        lines.append("  (no rules given)")
    for stat in report.rule_stats:
        status = []
        if not stat["valid"]:
            status.append("INVALID")
        if stat["shadowed"]:
            status.append("SHADOWED")
        if stat["unmatched"]:
            status.append("UNMATCHED")
        if stat["duplicate_of"] is not None:
            status.append(f"duplicate of #{stat['duplicate_of']}")
        flags = (" [" + ", ".join(status) + "]") if status else ""
        lines.append(
            f"  #{stat['index']} {stat['file']}:{stat['line']} "
            f"'{stat['raw_pattern']}' -> {stat['type'] or '?'}  "
            f"matches={stat['matched_tensors']} wins={stat['won_tensors']}{flags}"
        )

    summary = report.summary
    lines.append("")
    lines.append(
        f"estimated tensor bytes: {summary['human_estimated_tensor_bytes']} "
        f"({summary['estimated_tensor_bytes']} bytes)"
    )
    lines.append("by final type:")
    for name, entry in summary["by_final_type"].items():
        lines.append(
            f"  {name:<8} {entry['tensors']:>5} tensors  {entry['human_bytes']:>12}"
        )

    expert = summary["expert_coverage"]
    if expert["tensor_count"]:
        lines.append("")
        lines.append(
            f"expert coverage: {expert['covered_by_rule']}/{expert['tensor_count']} "
            f"tensors covered by a rule; layers={len(expert['layers'])} "
            f"({expert['layers'][0]}..{expert['layers'][-1]}), "
            f"families={expert['families']}"
        )
        if expert["uncovered"]:
            shown = expert["uncovered"][:12]
            more = len(expert["uncovered"]) - len(shown)
            suffix = f" ... (+{more} more; full list in --json)" if more else ""
            lines.append(f"  uncovered: {', '.join(shown)}{suffix}")
        else:
            lines.append("  all expert tensors hit a rule (none left on --default)")

    lines.append("")
    if report.diagnostics:
        lines.append(
            f"diagnostics: {report.errors} error(s), {report.warnings} warning(s)"
        )
        for diag in report.diagnostics:
            lines.append("  " + diag.render())
    else:
        lines.append("diagnostics: none")
    lines.append("")
    lines.append(
        f"exit code: {report.exit_code} "
        f"(0=clean, 2=strict warnings, 1=input error)"
    )
    return "\n".join(lines)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if path == "-":
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, target)


def _error_payload(message: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tool": "quant_rule_lint",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "model": {"path": args.model},
        "default_type": args.default,
        "rules": [],
        "tensors": [],
        "summary": {},
        "diagnostics": [{
            "severity": "error",
            "code": "input_error",
            "message": message,
        }],
        "warning_count": 0,
        "error_count": 1,
        "strict": bool(args.strict),
        "exit_code": 1,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant_rule_lint.py",
        description=(
            "Validate llama.cpp --tensor-type-file regex rules against the "
            "GGUF header (no tensor data is read)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 = no diagnostics; 2 = --strict and warnings; "
            "1 = input error.\n"
            "Rule files contain whitespace-separated name_regex=GGML_TYPE tokens; "
            "blank lines and # comments are allowed. The regex is lowercased and "
            "used with re.search(), first match wins (C++ semantics)."
        ),
    )
    parser.add_argument("--model", required=True, help="GGUF file (header is enough)")
    parser.add_argument(
        "--rules", action="append", default=[],
        help="rule file (repeatable; '-' reads stdin)",
    )
    parser.add_argument(
        "--default", required=True,
        help="default GGML type used by llama-quantize (e.g. Q6_K)",
    )
    parser.add_argument("--json", dest="json_out", help="write machine-readable JSON ('-' = stdout)")
    parser.add_argument(
        "--strict", action="store_true",
        help="return 2 when warnings are present",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    default_type = GGML_TYPES_BY_NAME.get(str(args.default).strip().upper())
    if default_type is None:
        message = (
            f"unknown --default GGML type '{args.default}'; valid names include "
            + ", ".join(sorted(GGML_TYPES_BY_NAME)[:8]) + ", ..."
        )
        print(f"quant_rule_lint: error: {message}", file=sys.stderr)
        if args.json_out:
            try:
                _write_json(args.json_out, _error_payload(message, args))
            except OSError as exc:
                print(f"quant_rule_lint: cannot write JSON: {exc}", file=sys.stderr)
        return 1

    try:
        model = parse_gguf_header(args.model)
        rules: list[Rule] = []
        for path in args.rules:
            file_rules = parse_rule_file(path)
            for rule in file_rules:
                rule.index += len(rules)
            rules.extend(file_rules)
        report = lint(model, rules, default_type, strict=args.strict)
    except LintError as exc:
        message = str(exc)
        print(f"quant_rule_lint: error: {message}", file=sys.stderr)
        if args.json_out:
            try:
                _write_json(args.json_out, _error_payload(message, args))
            except OSError as write_exc:
                print(
                    f"quant_rule_lint: cannot write JSON: {write_exc}",
                    file=sys.stderr,
                )
        return 1

    if args.json_out:
        try:
            _write_json(args.json_out, report.as_json())
        except OSError as exc:
            print(f"quant_rule_lint: cannot write JSON: {exc}", file=sys.stderr)
            return 1

    human = render_human(report)
    print(human, file=sys.stderr if args.json_out == "-" else sys.stdout)
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
