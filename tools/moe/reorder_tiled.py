#!/usr/bin/env python3
"""Permute an existing tiled expert pack **in place** (no second copy of the data).

`pack_hexagon.py --expert-order` writes the tiled pack in the requested expert
order, but it needs the raw extract and enough free disk for a second pack.  On a
phone that second copy is 14-16 GB, which usually does not fit.  This tool applies
the same permutation to an already tiled pack, one layer file at a time, using one
expert-sized buffer per cycle.  It also verifies the byte mapping while writing:

    sha256(new_file[position i]) == sha256(old_file[src[i]])

so `expert_order.orders[layer]` (position -> logical for the forward direction)
is proven per chunk, not just assumed.

The file layout is the one `pack_hexagon.py` produces: one `layer_XXX.bin` per
layer, expert-major, each expert a contiguous run of `expert_stride` bytes made of
the tiled families in manifest offset order.

Examples:

    # identity -> ordered, and write the new manifest with expert_order
    python3 reorder_tiled.py --dir ~/models/experts-htp-tiled --order order.json

    # ordered -> identity (restore; drops expert_order from the manifest)
    python3 reorder_tiled.py --dir ~/models/experts-htp-tiled --order order.json --inverse

    # only validate the layout/order, touch nothing
    python3 reorder_tiled.py --dir ... --order order.json --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_file(path: Path, mode: str = "w"):
    """Write via a temp file + os.replace, so a crash never leaves a torn manifest."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open(mode) as handle:
        yield handle
    os.replace(tmp, path)


def validate_order(values, count: int, where: str) -> list[int]:
    if not isinstance(values, list) or len(values) != count:
        raise SystemExit(f"{where}: expected a list of {count} expert ids")
    seen = [False] * count
    out = []
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < count:
            raise SystemExit(f"{where}[{index}]: invalid expert id {value!r}")
        if seen[value]:
            raise SystemExit(f"{where}: expert {value} appears twice (not a bijection)")
        seen[value] = True
        out.append(value)
    return out


def load_expert_order(path: Path, layers: list[int], count: int) -> dict[int, list[int]]:
    """Return layer -> (position -> logical), same convention as pack_hexagon.py."""
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise SystemExit("expert order document must be a JSON object")
    orders = document.get("orders", document)
    if not isinstance(orders, dict):
        raise SystemExit("expert order document must contain an 'orders' object")
    direction = document.get("direction", "position_to_logical")
    if direction not in ("position_to_logical", "logical_to_position"):
        raise SystemExit(f"unsupported expert order direction: {direction!r}")
    expected = {str(layer) for layer in layers}
    actual = {str(key) for key in orders}
    if actual != expected:
        raise SystemExit(f"expert order layers mismatch; missing={sorted(expected - actual)} "
                         f"extra={sorted(actual - expected)}")
    result = {}
    for layer in layers:
        values = validate_order(orders[str(layer)], count, f"orders[{layer}]")
        if direction == "logical_to_position":
            position_to_logical = [0] * count
            for logical, position in enumerate(values):
                position_to_logical[position] = logical
            values = position_to_logical
        result[layer] = values
    return result


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def layer_layout(manifest: dict, pack: Path, count: int) -> dict[int, dict]:
    """Layer -> {file, stride, size} for a tiled pack.

    The tiled file stride is **not** the manifest's `expert_stride`/`bytes_per_expert`
    (the manifest keeps the raw GGUF sizes; tiled sizes come from the backend
    `ggml_backend_pack_moe_expert` and are usually larger).  The only thing this
    permutation needs is the per-expert stride, which follows from the file size:

        stride = file_size / expert_count

    and is verified to divide exactly.  Family boundaries are irrelevant because
    whole expert chunks are moved as opaque byte ranges.
    """
    per_file: dict[str, list[dict]] = {}
    for entry in manifest["entries"]:
        per_file.setdefault(entry["repacked_file"], []).append(entry)
    layout: dict[int, dict] = {}
    for name, entries in per_file.items():
        layers = {int(e["layer"]) for e in entries}
        if len(layers) != 1:
            raise SystemExit(f"expert file {name} mixes layers: {sorted(layers)}")
        path = pack / name
        size = path.stat().st_size
        if count <= 0 or size % count:
            raise SystemExit(f"{path}: size {size} is not divisible by expert_count {count}")
        stride = size // count
        if stride <= 0:
            raise SystemExit(f"{path}: empty expert stride")
        layout[layers.pop()] = {"file": name, "stride": stride, "size": size,
                                "families": len(entries)}
    return layout


def read_chunk(handle, offset: int, size: int) -> bytes:
    handle.seek(offset)
    data = handle.read(size)
    if len(data) != size:
        raise SystemExit(f"short read at offset {offset}: got {len(data)} of {size}")
    return data


def hash_layer(path: Path, stride: int, count: int) -> list[str]:
    hashes = []
    with path.open("rb", buffering=0) as handle:
        for position in range(count):
            data = read_chunk(handle, position * stride, stride)
            hashes.append(sha256_bytes(data))
    return hashes


def permute_layer(path: Path, stride: int, count: int, src: list[int],
                  current: list[str]) -> list[int]:
    """In-place `new[i] = old[src[i]]`; returns written source index per position."""
    done = bytearray(count)
    written_src = [-1] * count
    fd = os.open(path, os.O_RDWR)
    try:
        def read_at(index: int) -> bytes:
            data = os.pread(fd, stride, index * stride)
            if len(data) != stride:
                raise SystemExit(f"{path}: short read at position {index}")
            return data

        def write_at(index: int, data: bytes) -> None:
            if len(data) != stride:
                raise SystemExit(f"{path}: bad chunk length {len(data)}")
            written = 0
            while written < stride:
                written += os.pwrite(fd, memoryview(data)[written:], index * stride + written)

        # Cycle following: new[cur] = old[src[cur]].  The only chunk that would be
        # overwritten before it is read is the first one of the cycle, so it is the
        # only one that needs a buffer.
        for start in range(count):
            if done[start]:
                continue
            first = read_at(start)
            digest = sha256_bytes(first)
            if digest != current[start]:
                raise SystemExit(f"{path}: chunk {start} changed since hashing")
            cur = start
            while True:
                nxt = src[cur]
                if nxt == start:
                    write_at(cur, first)
                    written_src[cur] = start
                    done[cur] = 1
                    break
                if done[nxt]:
                    raise SystemExit(f"{path}: cycle walked into an already written position {nxt}")
                data = read_at(nxt)
                digest = sha256_bytes(data)
                if digest != current[nxt]:
                    raise SystemExit(f"{path}: chunk {nxt} changed since hashing")
                write_at(cur, data)
                written_src[cur] = nxt
                done[cur] = 1
                cur = nxt
        os.fsync(fd)
    finally:
        os.close(fd)
    if any(w < 0 for w in written_src):
        raise SystemExit(f"{path}: permutation left positions unwritten")
    return written_src


def verify_layer(path: Path, stride: int, count: int, src: list[int],
                 current: list[str]) -> tuple[list[str], str]:
    """Re-read the file: chunk i must equal old chunk src[i]; returns (hashes, file sha)."""
    hashes = []
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for position in range(count):
            data = read_chunk(handle, position * stride, stride)
            digest.update(data)
            got = sha256_bytes(data)
            want = current[src[position]]
            if got != want:
                raise SystemExit(f"{path}: position {position} holds the wrong expert bytes")
            hashes.append(got)
    return hashes, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path, help="tiled pack directory (modified in place)")
    parser.add_argument("--order", required=True, type=Path,
                        help="dpsk-pack-order-v1 JSON; orders[layer] is position -> logical")
    parser.add_argument("--inverse", action="store_true",
                        help="apply the inverse permutation (ordered -> identity) and drop expert_order")
    parser.add_argument("--dry-run", action="store_true", help="validate layout/order only")
    parser.add_argument("--report", type=Path, help="write a JSON report here")
    args = parser.parse_args()

    pack = args.dir.expanduser().resolve()
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("backend_layout") != "hexagon-tiled-v1":
        raise SystemExit(f"{manifest_path}: not a tiled pack (backend_layout={manifest.get('backend_layout')!r})")
    count = int(manifest["model"]["expert_count"])
    layout = layer_layout(manifest, pack, count)
    layers = sorted(layout)
    order = load_expert_order(args.order.expanduser().resolve(), layers, count)
    if manifest.get("expert_order") and not args.inverse:
        raise SystemExit("manifest already has expert_order; pass --inverse to restore identity")
    if args.inverse and not manifest.get("expert_order"):
        raise SystemExit("--inverse needs a manifest with expert_order (nothing to undo)")

    src_by_layer: dict[int, list[int]] = {}
    for layer in layers:
        forward = order[layer]                      # position -> logical
        if args.inverse:
            inverse = [0] * count
            for position, logical in enumerate(forward):
                inverse[logical] = position
            src_by_layer[layer] = inverse
        else:
            src_by_layer[layer] = forward

    report: dict = {
        "dir": str(pack),
        "order": str(args.order),
        "inverse": args.inverse,
        "layers": len(layers),
        "experts": count,
        "layout": {str(layer): {"file": layout[layer]["file"], "stride": layout[layer]["stride"]}
                   for layer in layers},
        "per_layer": {},
    }

    started = time.time()
    if args.dry_run:
        report["dry_run"] = True
        report["elapsed_s"] = round(time.time() - started, 1)
        print(json.dumps({"layers": len(layers), "experts": count, "dry_run": True}))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    for layer in layers:
        info = layout[layer]
        path = pack / info["file"]
        stride, name = info["stride"], info["file"]
        t_hash = time.time()
        current = hash_layer(path, stride, count)
        t_perm = time.time()
        written_src = permute_layer(path, stride, count, src_by_layer[layer], current)
        t_ver = time.time()
        if written_src != src_by_layer[layer]:
            raise SystemExit(f"{name}: written source indices disagree with the permutation")
        hashes, file_sha = verify_layer(path, stride, count, src_by_layer[layer], current)
        # The semantic check for the forward direction: identity pack position p[i]
        # (logical p[i]) must now live at position i.
        for position in range(count):
            if hashes[position] != current[src_by_layer[layer][position]]:
                raise SystemExit(f"{name}: position {position} hash mismatch")
        identity_digest = hashlib.sha256("".join(current).encode()).hexdigest()
        ordered_digest = hashlib.sha256("".join(hashes).encode()).hexdigest()
        report["per_layer"][str(layer)] = {
            "file": name,
            "sha256": file_sha,
            "chunks": count,
            "mismatches": 0,
            "identity_hash_list_sha256": identity_digest,
            "ordered_hash_list_sha256": ordered_digest,
            "hash_s": round(t_perm - t_hash, 1),
            "permute_s": round(t_ver - t_perm, 1),
            "verify_s": round(time.time() - t_ver, 1),
        }
        print(f"layer {layer:3d} {name}: sha256={file_sha[:16]} "
              f"hash={t_perm - t_hash:.1f}s permute={t_ver - t_perm:.1f}s "
              f"verify={time.time() - t_ver:.1f}s", flush=True)

    result = dict(manifest)
    checksums = dict(result.get("hexagon_conversion", {}).get("files_sha256", {}))
    for layer in layers:
        checksums[layout[layer]["file"]] = report["per_layer"][str(layer)]["sha256"]
    result.setdefault("hexagon_conversion", {})["files_sha256"] = checksums
    if args.inverse:
        result.pop("expert_order", None)
    else:
        canonical = json.dumps({str(layer): order[layer] for layer in layers},
                               sort_keys=True, separators=(",", ":")).encode()
        result["expert_order"] = {
            "format": "dpsk-pack-order-v1",
            "direction": "position_to_logical",
            "orders_sha256": hashlib.sha256(canonical).hexdigest(),
            "orders": {str(layer): order[layer] for layer in layers},
        }
    with atomic_file(manifest_path, "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")

    report["elapsed_s"] = round(time.time() - started, 1)
    report["manifest"] = str(manifest_path)
    report["expert_order"] = result.get("expert_order", {}).get("orders_sha256")
    print(f"done: {len(layers)} layers, {count} experts/layer, elapsed {report['elapsed_s']}s")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
