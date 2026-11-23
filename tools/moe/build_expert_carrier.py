#!/usr/bin/env python3
"""Build a Q8_0 "carrier" GGUF from the Q1_0 / Q2_K expert anchors.

Why a carrier?
--------------
A GGUF tensor has exactly one quantization type, so a per-expert mixed
Q1_0/Q2_K model cannot be represented in GGUF (and the tiled expert manifest is
also one type per tensor).  To *measure* a per-expert allocation with the real
PPL pipeline we re-encode each expert slice -- after the intended Q1_0 or Q2_K
quantization -- into a uniform high-precision carrier (Q8_0, 1.0625 B/param).
Q8_0 perturbs the already-quantized values by ~0.4 %, i.e. a few percent of the
Q1_0/Q2_K error, so pairwise PPL differences between carrier arms track the
underlying allocation.  An all-Q1_0 carrier is built as the calibration arm and
compared against the direct all-Q1_0 GGUF.

Expert slices are read from the all-Q1_0 / all-Q2_K anchors (so the exact
llama-quantize bytes are used, not a numpy re-implementation), dequantized with
the ports in ``expert_quant_stats.py``, and re-quantized to Q8_0 once per expert.
One pass can emit several arms because the Q8_0 encodings are shared.

Dense/non-expert tensors are copied byte-for-byte from ``--q1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "moe"))

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, GGUFValueType  # type: ignore
from gguf.quants import quantize  # type: ignore

from expert_quant_stats import bf16_to_f32, dequant_q1_0, dequant_q2_K  # type: ignore

EXPERT_RE = re.compile(r"^blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight$")
_G: dict[str, Any] = {}


def _init(q1: str, q2k: str, meta: dict[str, dict[str, Any]], dtype: str) -> None:
    _G["q1"] = np.memmap(q1, mode="r")
    _G["q2k"] = np.memmap(q2k, mode="r")
    _G["meta"] = meta
    _G["dtype"] = dtype


def _slice(item: tuple[int, str, int]) -> tuple[bytes, bytes]:
    layer, family, expert = item
    m = _G["meta"][f"blk.{layer}.{family}.weight"]
    q1 = _G["q1"]
    q2k = _G["q2k"]
    s1 = m["q1_nbytes"] // m["n_expert"]
    s2 = m["q2k_nbytes"] // m["n_expert"]
    d1 = dequant_q1_0(np.asarray(q1[m["q1_offset"] + expert * s1: m["q1_offset"] + (expert + 1) * s1]))
    d2 = dequant_q2_K(np.asarray(q2k[m["q2k_offset"] + expert * s2: m["q2k_offset"] + (expert + 1) * s2]))
    if _G["dtype"] == "F16":
        return d1.astype(np.float16).tobytes(), d2.astype(np.float16).tobytes()
    b1 = quantize(d1, GGMLQuantizationType.Q8_0).tobytes()
    b2 = quantize(d2, GGMLQuantizationType.Q8_0).tobytes()
    return b1, b2


def byte_shape(file_shape: list[int], nbytes: int) -> list[int]:
    """GGUF ne order -> writer numpy order with the contiguous dim in bytes."""
    shape = list(reversed(file_shape))
    outer = 1
    for d in shape[:-1]:
        outer *= d
    shape[-1] = nbytes // outer
    return shape


def add_kv(writer: GGUFWriter, reader: GGUFReader) -> None:
    skip = {"split.count", "split.no", "split.tensors.count"}
    for key, field in reader.fields.items():
        if key in skip or key.startswith("GGUF."):
            continue
        vtype = field.types[0]
        value = field.contents()
        subtype = field.types[-1] if len(field.types) > 1 else None
        writer.add_key_value(key, value, vtype, subtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q1", required=True, type=Path)
    ap.add_argument("--q2k", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True,
                    help="name=none or name=allocation.tsv (layer<TAB>family<TAB>expert)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tensors", type=int, default=0, help="debug: only first N tensors")
    ap.add_argument("--dtype", choices=("Q8_0", "F16"), default="Q8_0",
                    help="carrier type; F16 is exact for the Q1_0/Q2_K dequantized values")
    args = ap.parse_args()

    arms: dict[str, set[tuple[int, str, int]]] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"bad --arm {spec!r}")
        name, path = spec.split("=", 1)
        if path == "none":
            arms[name] = set()
            continue
        chosen: set[tuple[int, str, int]] = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            layer, family, expert = line.split("\t")
            chosen.add((int(layer), family, int(expert)))
        arms[name] = chosen

    q1 = GGUFReader(str(args.q1), "r")
    q2k = GGUFReader(str(args.q2k), "r")
    meta_q1 = {t.name: t for t in q1.tensors}
    meta_q2 = {t.name: t for t in q2k.tensors}
    if set(meta_q1) != set(meta_q2):
        raise SystemExit("anchor tensor sets differ")

    # Tensor order = anchor order; expert tensors are contiguous per layer.
    tensors = list(q1.tensors)
    if args.max_tensors:
        tensors = tensors[:args.max_tensors]
    expert_meta: dict[str, dict[str, Any]] = {}
    for t in tensors:
        if EXPERT_RE.match(t.name):
            q2t = meta_q2[t.name]
            if t.tensor_type != GGMLQuantizationType.Q1_0 or q2t.tensor_type != GGMLQuantizationType.Q2_K:
                raise SystemExit(f"unexpected anchor types for {t.name}")
            ne = [int(x) for x in t.shape]
            expert_meta[t.name] = {
                "q1_offset": int(t.data_offset), "q1_nbytes": int(t.n_bytes),
                "q2k_offset": int(q2t.data_offset), "q2k_nbytes": int(q2t.n_bytes),
                "n_expert": ne[-1], "shape": ne,
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, GGUFWriter] = {}
    files: dict[str, Any] = {}
    source_arch = str(q1.get_field("general.architecture").contents())
    for name in arms:
        writer = GGUFWriter(str(args.out_dir / f"{name}.gguf"), arch=source_arch)
        for key, field in q1.fields.items():
            if key == "general.architecture":
                continue
            if key in {"split.count", "split.no", "split.tensors.count"} or key.startswith("GGUF."):
                continue
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if len(field.types) > 1 else None)
        for t in tensors:
            if t.name in expert_meta:
                ne = expert_meta[t.name]["shape"]
                if args.dtype == "F16":
                    nbytes = int(t.n_elements) * 2
                    writer.add_tensor_info(t.name, list(reversed(ne)), np.float16, nbytes)
                else:
                    # Q8_0: 34 bytes per 32 weights.
                    nbytes = (int(t.n_elements) // 32) * 34
                    writer.add_tensor_info(t.name, byte_shape(ne, nbytes), np.uint8, nbytes,
                                           raw_dtype=GGMLQuantizationType.Q8_0)
            else:
                if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
                    dtype = np.float32 if t.tensor_type == GGMLQuantizationType.F32 else np.float16
                    writer.add_tensor_info(t.name, list(reversed([int(x) for x in t.shape])),
                                           dtype, int(t.n_bytes))
                else:
                    writer.add_tensor_info(t.name, byte_shape([int(x) for x in t.shape], int(t.n_bytes)),
                                           np.uint8, int(t.n_bytes), raw_dtype=t.tensor_type)
        path = args.out_dir / f"{name}.gguf"
        writer.write_header_to_file(path)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writers[name] = writer

    # The writers keep their own file handle; use that instead of reopening.
    for name, writer in writers.items():
        assert writer.fout is not None
        files[name] = writer.fout[0]
        # write_ti_data_to_file() leaves the cursor at the end of the tensor
        # infos; tensor data starts at the next data_alignment boundary.
        pos = files[name].tell()
        pad = (-pos) % writer.data_alignment
        if pad:
            files[name].write(b"\x00" * pad)

    digest = {name: hashlib.sha256() for name in arms}
    q1_data = np.memmap(args.q1, mode="r")
    task_list = [
        (int(EXPERT_RE.match(t.name).group(1)), EXPERT_RE.match(t.name).group(2), e)
        for t in tensors if t.name in expert_meta
        for e in range(expert_meta[t.name]["n_expert"])
    ]
    task_iter = iter(task_list)

    def emit(name: str, buf: bytes) -> None:
        f = files[name]
        f.write(buf)
        pad = (-len(buf)) % 32
        if pad:
            f.write(b"\x00" * pad)
        digest[name].update(buf)

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(str(args.q1), str(args.q2k), expert_meta, args.dtype)) as pool:
        slices = pool.imap(_slice, task_list, chunksize=2)
        for t in tensors:
            if t.name not in expert_meta:
                raw = np.asarray(q1_data[t.data_offset: t.data_offset + t.n_bytes])
                for name in arms:
                    emit(name, raw.tobytes())
                continue
            for e in range(expert_meta[t.name]["n_expert"]):
                task = next(task_iter)
                b1, b2 = next(slices)
                for name, chosen in arms.items():
                    key = (task[0], task[1], e)
                    emit(name, b2 if key in chosen else b1)

    for name, writer in writers.items():
        files[name].flush()
        files[name].close()
        size = (args.out_dir / f"{name}.gguf").stat().st_size
        print(json.dumps({
            "arm": name, "path": str(args.out_dir / f"{name}.gguf"),
            "bytes": size, "giB": size / 2**30,
            "upgrades": len(arms[name]), "sha256": digest[name].hexdigest(),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
