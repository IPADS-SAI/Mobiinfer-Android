#!/usr/bin/env python3
"""Convert extracted raw expert files to the Hexagon backend's tiled layout."""
from __future__ import annotations

import argparse
import copy
import ctypes as C
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

from experts import atomic_file
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

# These are the quantizations exercised by the MiniCPM mixed-Q3 and Q4_K_M assets.
QUANT_TYPES = {"Q3_K": 11, "Q4_K": 12, "Q6_K": 14}


def relative_file(directory: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid expert file name: {name!r}")
    result = (directory / path).resolve()
    if not result.is_relative_to(directory):
        raise ValueError(f"expert file escapes directory: {name!r}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", required=True, type=Path, help="raw experts.py extract output")
    parser.add_argument("--out-dir", required=True, type=Path, help="new or empty tiled output directory")
    parser.add_argument("--lib-dir", required=True, type=Path, help="directory containing matching native Hexagon libraries")
    args = parser.parse_args()
    source = args.experts.expanduser().resolve()
    destination = args.out_dir.expanduser().resolve()
    libraries = args.lib_dir.expanduser().resolve()
    if source == destination or (destination.exists() and any(destination.iterdir())):
        raise ValueError("output directory must be distinct and empty; existing files are not reused")

    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("layout") != "layer_major_expert" or manifest.get("backend_layout"):
        raise ValueError("input must be a raw layer_major_expert manifest, not already tiled")
    count = int(manifest["model"]["expert_count"])
    if count <= 0 or not manifest["entries"]:
        raise ValueError("manifest must contain experts")
    groups: dict[str, list[dict]] = {}
    for entry in manifest["entries"]:
        name = entry["repacked_file"]
        relative_file(source, name)
        relative_file(destination, name)
        shape = entry["shape"]
        if (entry["quant_type"] not in QUANT_TYPES or len(shape) != 3 or
                shape[0] <= 0 or shape[0] % 256 or shape[1] <= 0 or shape[2] != count):
            raise ValueError(f"unsupported expert type/shape: {entry['tensor_name']}")
        block, block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType(QUANT_TYPES[entry["quant_type"]])]
        expected = shape[0] // block * block_bytes * shape[1]
        if entry["bytes_per_expert"] != expected:
            raise ValueError(f"raw expert size disagrees with shape/type: {entry['tensor_name']}")
        groups.setdefault(name, []).append(entry)
    # The runtime derives tiled offsets from raw family order, not from rewritten offsets.
    for name, entries in groups.items():
        entries.sort(key=lambda e: e["repacked_offset"])
        offset = 0
        stride = sum(int(e["bytes_per_expert"]) for e in entries)
        for entry in entries:
            if (entry["repacked_offset"] != offset or entry["expert_stride"] != stride or
                    entry["bytes_per_expert"] <= 0):
                raise ValueError(f"non-contiguous expert families in {name}")
            offset += entry["bytes_per_expert"]
        if relative_file(source, name).stat().st_size != stride * count:
            raise ValueError(f"unexpected raw expert file size: {name}")

    base = C.CDLL(str(libraries / "libggml-base.so"), mode=C.RTLD_GLOBAL)
    backend_path = libraries / "libggml-hexagon.so"
    backend = C.CDLL(str(backend_path), mode=C.RTLD_GLOBAL)
    backend.ggml_backend_hexagon_reg.restype = C.c_void_p
    base.ggml_backend_reg_get_proc_address.argtypes = [C.c_void_p, C.c_char_p]
    base.ggml_backend_reg_get_proc_address.restype = C.c_void_p
    address = base.ggml_backend_reg_get_proc_address(
        backend.ggml_backend_hexagon_reg(), b"ggml_backend_pack_moe_expert")
    if not address:
        raise ValueError("Hexagon library does not export ggml_backend_pack_moe_expert")
    pack = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_void_p, C.c_int, C.c_int64, C.c_int64)(address)

    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, entries in sorted(groups.items()):
        sizes = [pack(None, None, QUANT_TYPES[e["quant_type"]], *e["shape"][:2]) for e in entries]
        if not all(sizes):
            raise ValueError(f"backend cannot pack {name}")
        digest = hashlib.sha256()
        with ExitStack() as stack:
            src = stack.enter_context(relative_file(source, name).open("rb"))
            out = stack.enter_context(atomic_file(relative_file(destination, name), "wb"))
            for expert in range(count):
                for entry, size in zip(entries, sizes):
                    src.seek(entry["repacked_offset"] + expert * entry["expert_stride"])
                    raw = src.read(entry["bytes_per_expert"])
                    if len(raw) != entry["bytes_per_expert"]:
                        raise ValueError(f"short read: {name}, expert {expert}")
                    data = C.create_string_buffer(raw)
                    tiled = C.create_string_buffer(size)
                    written = pack(tiled, data, QUANT_TYPES[entry["quant_type"]], *entry["shape"][:2])
                    if written != size:
                        raise ValueError(f"backend size mismatch: {name}, expert {expert}")
                    view = memoryview(tiled).cast("B")
                    out.write(view)
                    digest.update(view)
            if out.tell() != sum(sizes) * count:
                raise ValueError(f"unexpected tiled output size: {name}")
        checksums[name] = digest.hexdigest()
        print(f"packed {name}: {sum(sizes) * count} bytes, sha256={checksums[name]}", flush=True)

    result = copy.deepcopy(manifest)
    result["backend_layout"] = "hexagon-tiled-v1"
    # Keep raw shape/type/stride metadata: the runtime computes backend strides separately.
    backend_digest = hashlib.sha256()
    with backend_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            backend_digest.update(chunk)
    result["hexagon_conversion"] = {
        "backend_sha256": backend_digest.hexdigest(),
        "files_sha256": checksums,
    }
    with atomic_file(destination / "manifest.json", "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(destination / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
