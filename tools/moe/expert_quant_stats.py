#!/usr/bin/env python3
"""Per-layer/per-expert quantization error statistics for packed MoE experts.

The deployed GGUF format stores one quantization type per expert tensor, so a
per-expert bit allocation cannot be expressed in a GGUF today.  This tool still
measures the quantity that such an allocation needs: for every
``(layer, family, expert)`` slice it dequantizes the all-Q1_0 and all-Q2_K
anchors and reports

* ``mse1`` / ``mse2``  -- weight MSE against the BF16 source,
* ``diff``             -- MSE between the Q1_0 and Q2_K dequantized weights,
* ``energy``           -- mean square of the BF16 weights,

Vectors are the per-expert slices of ``blk.N.ffn_{gate,up,down}_exps.weight``
(shape ``[n_embd, n_ff, n_expert]``, expert-major contiguous).

Q1_0 and Q2_K dequantization are ported from ``ggml/src/ggml-quants.c``
(``dequantize_row_q1_0`` / ``dequantize_row_q2_K``); the installed gguf-py has
no numpy implementation for these types.  Everything else uses gguf-py.

Only the standard library plus numpy/gguf-py (already required by the other
MoE tools) is used.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGMLQuantizationType, GGUFReader  # type: ignore

QK1_0 = 128
QK2_K = 256
FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
TENSOR_RE = r"^blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight$"

# Worker globals (memmaps are cheap to map; one map per process).
_G: dict[str, Any] = {}


def dequant_q1_0(raw: np.ndarray) -> np.ndarray:
    """Dequantize Q1_0 bytes (18 B / 128 weights) to float32."""
    blocks = raw.reshape(-1, 18)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    bits = np.unpackbits(blocks[:, 2:18], axis=1, bitorder="little")
    out = np.where(bits != 0, d, -d)
    return out.astype(np.float32, copy=False).reshape(-1)


def dequant_q2_K(raw: np.ndarray) -> np.ndarray:
    """Dequantize Q2_K bytes (84 B / 256 weights) to float32."""
    # block_q2_K layout: scales[16], qs[64], then {d, dmin} at the end.
    b = raw.reshape(-1, 84)
    scales = b[:, 0:16]
    qs = b[:, 16:80]
    d = b[:, 80:82].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    dmin = b[:, 82:84].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    nb = b.shape[0]
    out = np.empty((nb, QK2_K), dtype=np.float32)
    for half in range(2):
        q = qs[:, half * 32:(half + 1) * 32]
        for j in range(4):
            sc0 = scales[:, half * 8 + 2 * j][:, None]
            sc1 = scales[:, half * 8 + 2 * j + 1][:, None]
            dl0 = d * (sc0 & 0x0F)
            ml0 = dmin * (sc0 >> 4)
            dl1 = d * (sc1 & 0x0F)
            ml1 = dmin * (sc1 >> 4)
            shift = 2 * j
            v0 = (q[:, :16] >> shift) & 3
            v1 = (q[:, 16:] >> shift) & 3
            base = half * 128 + j * 32
            out[:, base:base + 16] = dl0 * v0 - ml0
            out[:, base + 16:base + 32] = dl1 * v1 - ml1
    return out.reshape(-1)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def _init(q1: str, q2k: str, bf16: str | None, tensors: dict[str, dict[str, Any]]) -> None:
    _G["q1"] = np.memmap(q1, mode="r")
    _G["q2k"] = np.memmap(q2k, mode="r")
    _G["bf16"] = np.memmap(bf16, mode="r") if bf16 else None
    _G["tensors"] = tensors


def _process(item: tuple[int, str]) -> list[dict[str, Any]]:
    layer, family = item
    meta = _G["tensors"][f"blk.{layer}.{family}.weight"]
    q1 = _G["q1"]
    q2k = _G["q2k"]
    bf = _G["bf16"]
    n_expert = meta["n_expert"]
    stride1 = meta["q1_nbytes"] // n_expert
    stride2 = meta["q2k_nbytes"] // n_expert
    strideb = meta["bf16_nbytes"] // n_expert if bf is not None else 0
    rows: list[dict[str, Any]] = []
    for e in range(n_expert):
        o1 = meta["q1_offset"] + e * stride1
        o2 = meta["q2k_offset"] + e * stride2
        d1 = dequant_q1_0(np.asarray(q1[o1:o1 + stride1]))
        d2 = dequant_q2_K(np.asarray(q2k[o2:o2 + stride2]))
        if d1.size != d2.size:
            raise RuntimeError(f"size mismatch L{layer} {family} e{e}: {d1.size} vs {d2.size}")
        diff = float(np.mean((d1.astype(np.float64) - d2.astype(np.float64)) ** 2))
        rec: dict[str, Any] = {
            "layer": layer,
            "family": family,
            "expert": e,
            "n_params": int(d1.size),
            "diff": diff,
        }
        if bf is not None:
            ob = meta["bf16_offset"] + e * strideb
            w = bf16_to_f32(np.asarray(bf[ob:ob + strideb]))
            if w.size != d1.size:
                raise RuntimeError(f"bf16 size mismatch L{layer} {family} e{e}")
            rec["energy"] = float(np.mean(w.astype(np.float64) ** 2))
            rec["mse1"] = float(np.mean((d1.astype(np.float64) - w.astype(np.float64)) ** 2))
            rec["mse2"] = float(np.mean((d2.astype(np.float64) - w.astype(np.float64)) ** 2))
        rows.append(rec)
    return rows


def read_tensor_meta(reader: GGUFReader, pattern: str) -> dict[str, dict[str, Any]]:
    import re
    rx = re.compile(pattern)
    out: dict[str, dict[str, Any]] = {}
    for t in reader.tensors:
        m = rx.match(t.name)
        if not m:
            continue
        out[t.name] = {
            "name": t.name,
            "tensor_type": t.tensor_type,
            "shape": [int(x) for x in t.shape],
            "nbytes": int(t.n_bytes),
            "offset": int(t.data_offset),
        }
    return out


def quant_type_name(t: GGMLQuantizationType) -> str:
    return GGMLQuantizationType(t).name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q1", required=True, type=Path)
    ap.add_argument("--q2k", required=True, type=Path)
    ap.add_argument("--bf16", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--layers", default=None, help="e.g. 0-39")
    args = ap.parse_args()

    q1_reader = GGUFReader(str(args.q1), "r")
    q2_reader = GGUFReader(str(args.q2k), "r")
    bf_reader = GGUFReader(str(args.bf16), "r") if args.bf16 else None

    q1_meta = read_tensor_meta(q1_reader, TENSOR_RE)
    q2_meta = read_tensor_meta(q2_reader, TENSOR_RE)
    bf_meta = read_tensor_meta(bf_reader, TENSOR_RE) if bf_reader else {}
    if not q1_meta or set(q1_meta) != set(q2_meta):
        raise SystemExit("Q1_0/Q2_K anchor expert tensors do not match")
    if bf_meta and set(bf_meta) != set(q1_meta):
        raise SystemExit("BF16 source expert tensors do not match anchors")

    layers: set[int] | None = None
    if args.layers:
        layers = set()
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                layers.update(range(int(a), int(b) + 1))
            elif part:
                layers.add(int(part))

    items: list[tuple[int, str]] = []
    meta_out: dict[str, dict[str, Any]] = {}
    for name, m in sorted(q1_meta.items()):
        layer = int(name.split(".")[1])
        family = name.split(".")[2]
        if layers is not None and layer not in layers:
            continue
        q2 = q2_meta[name]
        if m["shape"] != q2["shape"]:
            raise SystemExit(f"shape mismatch for {name}")
        if quant_type_name(m["tensor_type"]) != "Q1_0":
            raise SystemExit(f"{name} is {quant_type_name(m['tensor_type'])}, not Q1_0")
        if quant_type_name(q2["tensor_type"]) != "Q2_K":
            raise SystemExit(f"{name} is {quant_type_name(q2['tensor_type'])}, not Q2_K")
        rec = {
            "shape": m["shape"],
            "n_expert": m["shape"][-1],
            "q1_offset": m["offset"],
            "q1_nbytes": m["nbytes"],
            "q2k_offset": q2["offset"],
            "q2k_nbytes": q2["nbytes"],
        }
        if bf_meta:
            rec.update({"bf16_offset": bf_meta[name]["offset"], "bf16_nbytes": bf_meta[name]["nbytes"]})
        meta_out[name] = rec
        items.append((layer, family))

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with mp.Pool(
        args.workers,
        initializer=_init,
        initargs=(str(args.q1), str(args.q2k), str(args.bf16) if args.bf16 else None, meta_out),
    ) as pool:
        for result in pool.imap_unordered(_process, items, chunksize=1):
            rows.extend(result)
    rows.sort(key=lambda r: (r["layer"], FAMILIES.index(r["family"]), r["expert"]))
    if len(rows) != len(items) * 256:
        raise SystemExit(f"expected {len(items) * 256} rows, got {len(rows)}")

    cols = ["layer", "family", "expert", "n_params", "diff"]
    if bf_meta:
        cols += ["mse1", "mse2", "energy"]
    with (args.out / "expert-quant-stats.tsv").open("w", encoding="utf-8") as out:
        out.write("\t".join(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                cells.append(v if isinstance(v, str) else (str(v) if isinstance(v, int) else f"{v:.6e}"))
            out.write("\t".join(cells) + "\n")

    manifest = {
        "q1": str(args.q1),
        "q2k": str(args.q2k),
        "bf16": str(args.bf16) if args.bf16 else None,
        "rows": len(rows),
        "tensors": len(items),
        "layers": sorted({r["layer"] for r in rows}),
        "families": list(FAMILIES),
        "n_expert": 256,
        "bytes_q1_per_param": 18 / 128,
        "bytes_q2k_per_param": 84 / 256,
        "delta_bytes_per_param": 84 / 256 - 18 / 128,
        "dequant": "ported from ggml/src/ggml-quants.c",
    }
    (args.out / "expert-quant-stats.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
