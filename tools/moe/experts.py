#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# tools/moe/experts.py -> parents[2] = 仓库根（再拼 gguf-py）。脚本挪位置要同步改深度。
LLAMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

from gguf import GGML_QUANT_SIZES, GGUFReader  # type: ignore


ROUTED_FAMILIES = {
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps",
}
FAMILY_ORDER = {
    "ffn_gate_exps": 0,
    "ffn_up_exps": 1,
    "ffn_down_exps": 2,
    "ffn_gate_up_exps": 3,
}
TENSOR_RE = re.compile(r"^blk\.(\d+)\.(ffn_[^.]+)\.weight$")
SPLIT_RE = re.compile(r"^(?P<prefix>.+-)(?P<index>\d+)-of-(?P<count>\d+)(?P<suffix>\.gguf)$")


@contextmanager
def atomic_file(path: Path, mode: str) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    kwargs = {} if "b" in mode else {"encoding": "utf-8"}
    try:
        with tmp.open(mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def scalar(reader: GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def resolve_model_paths(model: Path) -> list[Path]:
    model = model.expanduser().resolve()
    reader = GGUFReader(str(model), "r")
    split_count = int(scalar(reader, "split.count", 1))
    if split_count == 1:
        return [model]

    match = SPLIT_RE.match(model.name)
    if match is None:
        raise SystemExit(f"cannot derive split names from '{model.name}'")

    width = len(match.group("index"))
    count_width = len(match.group("count"))
    paths = [
        model.with_name(
            f"{match.group('prefix')}{index:0{width}d}-of-"
            f"{split_count:0{count_width}d}{match.group('suffix')}"
        )
        for index in range(1, split_count + 1)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing GGUF shards: {', '.join(missing)}")
    return paths


def parse_layers(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    result: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if "-" in item:
            first, last = (int(value) for value in item.split("-", 1))
            if last < first:
                raise SystemExit(f"invalid layer range '{item}'")
            result.update(range(first, last + 1))
        elif item:
            result.add(int(item))
    return result


def tensor_info(name: str) -> tuple[int, str] | None:
    match = TENSOR_RE.match(name)
    if match is None or match.group(2) not in ROUTED_FAMILIES:
        return None
    return int(match.group(1)), match.group(2)


def collect_entries(model_paths: list[Path], layers: set[int] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readers = [GGUFReader(str(path), "r") for path in model_paths]
    first = readers[0]
    arch = str(scalar(first, "general.architecture", "unknown"))
    expert_count = int(scalar(first, f"{arch}.expert_count", 0))
    expert_used = int(scalar(first, f"{arch}.expert_used_count", 0))
    entries: list[dict[str, Any]] = []

    for shard_index, (path, reader) in enumerate(zip(model_paths, readers)):
        for tensor in reader.tensors:
            parsed = tensor_info(tensor.name)
            if parsed is None:
                continue
            layer, family = parsed
            if layers is not None and layer not in layers:
                continue

            shape = [int(value) for value in tensor.shape.tolist()]
            if len(shape) < 3 or shape[2] != expert_count:
                raise SystemExit(
                    f"'{tensor.name}' has shape {shape}, expected expert axis 2 with {expert_count} experts"
                )
            if tensor.n_bytes % expert_count != 0:
                raise SystemExit(f"'{tensor.name}' byte size is not divisible by expert count")
            block_size, _ = GGML_QUANT_SIZES[tensor.tensor_type]
            entries.append(
                {
                    "layer": layer,
                    "tensor_family": family,
                    "tensor_name": tensor.name,
                    "original_gguf_tensor_name": tensor.name,
                    "quant_type": tensor.tensor_type.name,
                    "block_size": int(block_size),
                    "shape": shape,
                    "exact_byte_length": int(tensor.n_bytes),
                    "bytes_per_expert": int(tensor.n_bytes // expert_count),
                    "source_shard": str(path),
                    "source_shard_index": shard_index,
                    "source_offset": int(tensor.data_offset),
                }
            )

    entries.sort(key=lambda item: (item["layer"], FAMILY_ORDER[item["tensor_family"]]))
    return entries, {
        "arch": arch,
        "expert_count": expert_count,
        "expert_used_count": expert_used,
    }


def copy_range(source: Any, output: Any, offset: int, size: int, label: str) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        data = source.read(min(8 << 20, remaining))
        if not data:
            raise SystemExit(f"unexpected EOF while reading '{label}'")
        output.write(data)
        remaining -= len(data)


def extract(args: argparse.Namespace) -> int:
    model_paths = resolve_model_paths(args.model)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and any(out_dir.iterdir()):
        raise SystemExit(f"output directory '{out_dir}' is not empty; pass --force to overwrite")

    entries, metadata = collect_entries(model_paths, parse_layers(args.layers))
    if not entries:
        raise SystemExit("no routed expert tensors found")

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_layer[entry["layer"]].append(entry)

    with ExitStack() as stack:
        sources = [stack.enter_context(path.open("rb")) for path in model_paths]
        for layer, layer_entries in sorted(by_layer.items()):
            layer_entries.sort(key=lambda item: FAMILY_ORDER[item["tensor_family"]])
            stride = sum(entry["bytes_per_expert"] for entry in layer_entries)
            family_offset = 0
            for entry in layer_entries:
                entry["repacked_file"] = f"layer_{layer:03d}.bin"
                entry["repacked_offset"] = family_offset
                entry["expert_stride"] = stride
                family_offset += entry["bytes_per_expert"]

            with atomic_file(out_dir / f"layer_{layer:03d}.bin", "wb") as output:
                for expert in range(metadata["expert_count"]):
                    for entry in layer_entries:
                        size = entry["bytes_per_expert"]
                        copy_range(
                            sources[entry["source_shard_index"]],
                            output,
                            entry["source_offset"] + expert * size,
                            size,
                            f"{entry['tensor_name']} expert {expert}",
                        )

    manifest = {
        "schema_version": 1,
        "experts_kind": "omni_moe_gguf",
        "layout": "layer_major_expert",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"model_files": [str(path) for path in model_paths]},
        "model": metadata,
        "entries": entries,
    }
    with atomic_file(out_dir / "manifest.json", "w") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")

    total = sum(entry["exact_byte_length"] for entry in entries)
    print(f"wrote {len(entries)} tensors in {len(by_layer)} layers ({total} bytes)")
    print(out_dir / "manifest.json")
    return 0


def verify(args: argparse.Namespace) -> int:
    model_paths = resolve_model_paths(args.model)
    manifest_path = args.experts.expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected, metadata = collect_entries(model_paths, parse_layers(args.layers))
    expected_by_name = {entry["tensor_name"]: entry for entry in expected}
    entries = manifest.get("entries", [])

    with ExitStack() as stack:
        sources = [stack.enter_context(path.open("rb")) for path in model_paths]
        experts: dict[str, Any] = {}
        for entry in entries:
            source = expected_by_name.get(entry["tensor_name"])
            if source is None:
                raise SystemExit(f"unexpected manifest tensor '{entry['tensor_name']}'")
            for key in ("shape", "quant_type", "exact_byte_length", "bytes_per_expert"):
                if entry[key] != source[key]:
                    raise SystemExit(f"metadata mismatch for '{entry['tensor_name']}': {key}")
            if args.metadata_only:
                continue

            experts = experts.get(entry["repacked_file"])
            if experts is None:
                experts = stack.enter_context((manifest_path.parent / entry["repacked_file"]).open("rb"))
                experts[entry["repacked_file"]] = experts
            size = entry["bytes_per_expert"]
            for expert in range(metadata["expert_count"]):
                sources[source["source_shard_index"]].seek(source["source_offset"] + expert * size)
                experts.seek(entry["repacked_offset"] + expert * entry["expert_stride"])
                remaining = size
                while remaining:
                    count = min(8 << 20, remaining)
                    if sources[source["source_shard_index"]].read(count) != experts.read(count):
                        raise SystemExit(f"byte mismatch for '{entry['tensor_name']}' expert {expert}")
                    remaining -= count

    if {entry["tensor_name"] for entry in entries} != set(expected_by_name):
        raise SystemExit("manifest tensor set does not match the requested GGUF tensors")
    mode = "metadata" if args.metadata_only else "metadata and bytes"
    print(f"verified {len(entries)} tensors ({mode})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract expert-major GGUF experts for Omni MoE experiments.")
    commands = parser.add_subparsers(dest="command", required=True)

    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("--model", required=True, type=Path)
    extract_parser.add_argument("--out-dir", required=True, type=Path)
    extract_parser.add_argument("--layers")
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.set_defaults(func=extract)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--model", required=True, type=Path)
    verify_parser.add_argument("--experts", required=True, type=Path)
    verify_parser.add_argument("--layers")
    verify_parser.add_argument("--metadata-only", action="store_true")
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
