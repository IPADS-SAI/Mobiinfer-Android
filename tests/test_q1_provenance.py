#!/usr/bin/env python3
"""Tests for tools/moe/q1_provenance.py (stdlib unittest only).

Run from the repository root:

    python3 -m unittest tests.test_q1_provenance -v
    # or
    python3 -m pytest tests/test_q1_provenance.py -q

All fixtures are synthetic: two tiled layers with three fake expert families,
tiny fake inputs, and a GGUF header built byte-by-byte.  No model is loaded
and no tiled package is modified.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.moe import q1_provenance as qp  # noqa: E402
from tests.test_quant_rule_lint import (  # noqa: E402
    build_gguf,
    kv_array,
    kv_string,
    kv_u32,
    pack_string,
)


EXPERT_COUNT = 4
BYTES_PER_EXPERT = 1024
FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
_UNSET = object()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def layer_bytes(layer: int, size: int) -> bytes:
    seed = hashlib.sha256(f"synthetic-layer-{layer}".encode()).digest()
    return (seed * ((size // len(seed)) + 1))[:size]


def make_tiled(tmp: Path, layers: tuple[int, ...] = (0, 1)) -> Path:
    """Write a minimal but structurally valid hexagon-tiled package."""
    tiled = tmp / "tiled"
    tiled.mkdir()
    stride = BYTES_PER_EXPERT * len(FAMILIES)
    entries: list[dict] = []
    files: dict[str, str] = {}
    for layer in layers:
        offset = 0
        for family in FAMILIES:
            entries.append(
                {
                    "layer": layer,
                    "tensor_family": family,
                    "tensor_name": f"blk.{layer}.{family}.weight",
                    "original_gguf_tensor_name": f"blk.{layer}.{family}.weight",
                    "quant_type": "Q1_0",
                    "block_size": 128,
                    "shape": [256, 8, EXPERT_COUNT],
                    "exact_byte_length": BYTES_PER_EXPERT * EXPERT_COUNT,
                    "bytes_per_expert": BYTES_PER_EXPERT,
                    "source_shard": str(tmp / "arm.gguf"),
                    "source_shard_index": 0,
                    "source_offset": 0,
                    "repacked_file": f"layer_{layer:03d}.bin",
                    "repacked_offset": offset,
                    "expert_stride": stride,
                }
            )
            offset += BYTES_PER_EXPERT
        name = f"layer_{layer:03d}.bin"
        data = layer_bytes(layer, stride * EXPERT_COUNT)
        (tiled / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()

    manifest = {
        "schema_version": 1,
        "experts_kind": "omni_moe_gguf",
        "layout": "layer_major_expert",
        "generated_at": "2026-11-22T00:00:00+00:00",
        "source": {"model_files": [str(tmp / "arm.gguf")]},
        "model": {
            "arch": "qwen35moe",
            "expert_count": EXPERT_COUNT,
            "expert_used_count": 2,
        },
        "backend_layout": "hexagon-tiled-v1",
        "entries": entries,
        "hexagon_conversion": {
            "backend_sha256": "a" * 64,
            "files_sha256": files,
        },
    }
    (tiled / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return tiled


def make_inputs(
    tmp: Path,
    *,
    imatrix_kv: bool = True,
    arm_name: str = "arm.gguf",
    imatrix_array: bool = False,
) -> dict[str, Path]:
    arm = tmp / arm_name
    imatrix = tmp / "imatrix.gguf"
    imatrix.write_bytes(b"synthetic-imatrix-data\n")
    metadata = [
        kv_string("general.architecture", "qwen35moe"),
        kv_u32("qwen35moe.expert_count", EXPERT_COUNT),
        kv_u32("general.alignment", 32),
    ]
    if imatrix_kv:
        metadata.append(kv_string("quantize.imatrix.file", str(imatrix)))
        metadata.append(kv_u32("quantize.imatrix.chunks_count", 128))
    if imatrix_array:
        metadata.append(
            kv_array(
                "imatrix.datasets",
                8,  # GGUF_STRING
                [pack_string("corpus-a.txt"), pack_string("corpus-b.txt")],
            )
        )
        # A future/unknown imatrix-prefixed key must also be recorded.
        metadata.append(kv_string("quantize.imatrix.sha256", "f" * 64))
        # An unrelated array exercises the "skip without retaining" path.
        metadata.append(
            kv_array("tokenizer.ggml.tokens", 8, [pack_string(f"tok{i}") for i in range(8)])
        )
    arm.write_bytes(build_gguf([], metadata=metadata))

    rules = tmp / "rules.txt"
    rules.write_text(
        "regex:ffn_(gate|up|down)_exps\\.weight,Q1_0\n", encoding="utf-8"
    )
    quantizer = tmp / "ggml-quants.c"
    quantizer.write_text("/* synthetic quantizer source */\n", encoding="utf-8")
    return {
        "arm": arm,
        "imatrix": imatrix,
        "rules": rules,
        "quantizer": quantizer,
    }


class ProvenanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="q1-provenance-")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.tiled = make_tiled(self.tmp)
        self.inputs = make_inputs(self.tmp)
        self.out = self.tmp / "out" / "provenance.json"

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = qp.main(argv)
        return rc, stdout.getvalue(), stderr.getvalue()

    def stamp(
        self,
        *,
        out: Path | None = None,
        weighted: str = "auto",
        tiled: Path | None = None,
        inputs: dict[str, Path] | None = None,
    ) -> tuple[int, str, str]:
        inputs = inputs or self.inputs
        return self.run_cli(
            [
                "stamp",
                "--tiled",
                str(tiled or self.tiled),
                "--arm",
                str(inputs["arm"]),
                "--imatrix",
                str(inputs["imatrix"]),
                "--rules",
                str(inputs["rules"]),
                "--quantizer-src",
                str(inputs["quantizer"]),
                "--weighted-q1-0",
                weighted,
                "--out",
                str(out or self.out),
            ]
        )

    def verify(
        self,
        *,
        provenance: Path | None | object = _UNSET,
        tiled: Path | None = None,
        expect_arm: str | None = None,
        expect_imatrix: str | None = None,
        require: bool = False,
    ) -> tuple[int, str, str]:
        argv = ["verify", "--tiled", str(tiled or self.tiled)]
        if provenance is _UNSET:
            provenance = self.out
        if provenance is not None:
            argv += ["--provenance", str(provenance)]
        if expect_arm is not None:
            argv += ["--expect-arm-sha", expect_arm]
        if expect_imatrix is not None:
            argv += ["--expect-imatrix-sha", expect_imatrix]
        if require:
            argv.append("--require-provenance")
        return self.run_cli(argv)

    def load_record(self) -> dict:
        return json.loads(self.out.read_text(encoding="utf-8"))

    def write_record(self, record: dict) -> None:
        self.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


class RoundTripTests(ProvenanceTestCase):
    def test_stamp_verify_round_trip_and_tiled_bytes_unchanged(self) -> None:
        before = file_hashes(self.tiled)
        rc, stdout, stderr = self.stamp()
        self.assertEqual(rc, 0, stderr)
        self.assertIn("stamped", stdout)

        record = self.load_record()
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["arm_sha256"], sha256(self.inputs["arm"]))
        self.assertEqual(record["arm_bytes"], self.inputs["arm"].stat().st_size)
        self.assertEqual(record["imatrix_sha256"], sha256(self.inputs["imatrix"]))
        self.assertEqual(record["imatrix_bytes"], self.inputs["imatrix"].stat().st_size)
        self.assertEqual(record["quantizer_source_sha256"], sha256(self.inputs["quantizer"]))
        self.assertEqual(record["rules_sha256"], sha256(self.inputs["rules"]))
        self.assertEqual(
            record["manifest_sha256"], sha256(self.tiled / "manifest.json")
        )
        self.assertEqual(record["backend_layout"], "hexagon-tiled-v1")
        self.assertEqual(record["layer_count"], 2)
        self.assertEqual(record["entry_count"], 6)
        self.assertEqual(record["expert_count"], EXPERT_COUNT)
        self.assertEqual(record["weighted_q1_0"], "true")
        self.assertEqual(record["weighted_q1_0_source"], "arm-metadata")
        self.assertEqual(
            record["arm_imatrix"]["quantize.imatrix.file"],
            str(self.inputs["imatrix"]),
        )

        # stamp must not touch manifest.json or any layer file.
        self.assertEqual(file_hashes(self.tiled), before)

        rc, stdout, stderr = self.verify(
            expect_arm=record["arm_sha256"],
            expect_imatrix=record["imatrix_sha256"],
        )
        self.assertEqual(rc, 0, stderr)
        self.assertIn("OK:", stdout)

    def test_stamp_outside_tiled_leaves_directory_listing_unchanged(self) -> None:
        before = {p.name for p in self.tiled.iterdir()}
        rc, _, err = self.stamp()
        self.assertEqual(rc, 0, err)
        self.assertEqual({p.name for p in self.tiled.iterdir()}, before)

    def test_stamp_inside_tiled_only_adds_sidecar(self) -> None:
        before = file_hashes(self.tiled)
        out = self.tiled / "provenance.json"
        rc, _, err = self.stamp(out=out)
        self.assertEqual(rc, 0, err)
        self.assertTrue(out.is_file())
        after = file_hashes(self.tiled)
        after.pop("provenance.json")
        self.assertEqual(after, before)
        self.assertEqual(self.verify(provenance=out)[0], 0)

    def test_minimal_sidecar_with_only_spec_fields_verifies(self) -> None:
        rc, _, err = self.stamp()
        self.assertEqual(rc, 0, err)
        record = self.load_record()
        minimal = {
            key: record[key]
            for key in (
                "arm_sha256",
                "arm_bytes",
                "imatrix_sha256",
                "imatrix_bytes",
                "quantizer_source_sha256",
                "rules_sha256",
                "manifest_sha256",
                "weighted_q1_0",
                "created_at",
                "tool_version",
            )
        }
        self.write_record(minimal)
        rc, _, err = self.verify()
        self.assertEqual(rc, 0, err)

    def test_stamp_refuses_to_overwrite_manifest(self) -> None:
        before = sha256(self.tiled / "manifest.json")
        rc, _, err = self.stamp(out=self.tiled / "manifest.json")
        self.assertEqual(rc, 2)
        self.assertIn("must not overwrite", err)
        self.assertEqual(sha256(self.tiled / "manifest.json"), before)


class WeightedFlagTests(ProvenanceTestCase):
    def test_auto_without_imatrix_kv_is_unknown(self) -> None:
        inputs = make_inputs(self.tmp, imatrix_kv=False, arm_name="plain.gguf")
        rc, _, err = self.stamp(inputs=inputs)
        self.assertEqual(rc, 0, err)
        record = self.load_record()
        self.assertEqual(record["weighted_q1_0"], "unknown")
        self.assertEqual(record["weighted_q1_0_source"], "unknown")
        self.assertIsNone(record["arm_imatrix"])
        self.assertIn("unknown", err)
        self.assertEqual(self.verify()[0], 0)

    def test_explicit_flag_overrides_auto(self) -> None:
        rc, _, err = self.stamp(weighted="false")
        self.assertEqual(rc, 0, err)
        record = self.load_record()
        self.assertEqual(record["weighted_q1_0"], "false")
        self.assertEqual(record["weighted_q1_0_source"], "explicit")
        self.assertEqual(self.verify()[0], 0)

        inputs = make_inputs(self.tmp, imatrix_kv=False, arm_name="plain2.gguf")
        out = self.tmp / "explicit-true.json"
        rc, _, err = self.stamp(out=out, weighted="true", inputs=inputs)
        self.assertEqual(rc, 0, err)
        record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(record["weighted_q1_0"], "true")
        self.assertEqual(record["weighted_q1_0_source"], "explicit")

    def test_array_metadata_is_recorded_and_unrelated_arrays_are_skipped(self) -> None:
        inputs = make_inputs(
            self.tmp, arm_name="array.gguf", imatrix_array=True
        )
        rc, _, err = self.stamp(inputs=inputs)
        self.assertEqual(rc, 0, err)
        record = self.load_record()
        self.assertEqual(record["weighted_q1_0"], "true")
        self.assertEqual(
            record["arm_imatrix"]["imatrix.datasets"],
            ["corpus-a.txt", "corpus-b.txt"],
        )
        self.assertEqual(
            record["arm_imatrix"]["quantize.imatrix.sha256"], "f" * 64
        )
        self.assertNotIn("tokenizer.ggml.tokens", record["arm_imatrix"])

    def test_does_not_infer_from_file_size(self) -> None:
        """Unweighted packages have identical sizes; auto must not claim false."""
        inputs = make_inputs(self.tmp, imatrix_kv=False, arm_name="plain3.gguf")
        rc, _, err = self.stamp(inputs=inputs)
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.load_record()["weighted_q1_0"], "unknown")


class TamperTests(ProvenanceTestCase):
    def setUp(self) -> None:
        super().setUp()
        rc, _, err = self.stamp()
        self.assertEqual(rc, 0, err)
        self.record = self.load_record()

    def test_changed_layer_byte_fails(self) -> None:
        path = self.tiled / "layer_000.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(data)
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("layer file sha256 mismatch", err)
        self.assertIn("layer_000.bin", err)

    def test_truncated_layer_file_fails(self) -> None:
        path = self.tiled / "layer_001.bin"
        path.write_bytes(path.read_bytes()[:-1])
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("size mismatch", err)

    def test_missing_layer_file_fails(self) -> None:
        (self.tiled / "layer_000.bin").unlink()
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("missing/unreadable", err)

    def test_changed_imatrix_sha_with_expectation_fails(self) -> None:
        original = self.record["imatrix_sha256"]
        self.record["imatrix_sha256"] = "0" * 64
        self.write_record(self.record)
        rc, _, err = self.verify(expect_imatrix=original)
        self.assertEqual(rc, 3)
        self.assertIn("imatrix sha256 mismatch", err)

    def test_changed_arm_sha_with_expectation_fails(self) -> None:
        original = self.record["arm_sha256"]
        self.record["arm_sha256"] = "0" * 64
        self.write_record(self.record)
        rc, _, err = self.verify(expect_arm=original)
        self.assertEqual(rc, 3)
        self.assertIn("arm sha256 mismatch", err)

    def test_changed_manifest_sha_fails(self) -> None:
        self.record["manifest_sha256"] = "0" * 64
        self.write_record(self.record)
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("manifest sha256 mismatch", err)

    def test_changed_manifest_bytes_fails(self) -> None:
        manifest = self.tiled / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("manifest sha256 mismatch", err)

    def test_changed_sidecar_files_sha_fails(self) -> None:
        name = sorted(self.record["files_sha256"])[0]
        self.record["files_sha256"][name] = "0" * 64
        self.write_record(self.record)
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("files_sha256 mismatch", err)

    def test_changed_backend_layout_fails(self) -> None:
        self.record["backend_layout"] = "raw-layer-major"
        self.write_record(self.record)
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("backend_layout mismatch", err)

    def test_changed_counts_fail(self) -> None:
        for key, value in (
            ("entry_count", 5),
            ("layer_count", 40),
            ("expert_count", 256),
        ):
            with self.subTest(key=key):
                record = dict(self.record)
                record[key] = value
                self.write_record(record)
                rc, _, err = self.verify()
                self.assertEqual(rc, 3)
                self.assertIn(f"{key} mismatch", err)

    def test_changed_weighted_flag_value_fails(self) -> None:
        self.record["weighted_q1_0"] = "maybe"
        self.write_record(self.record)
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("weighted_q1_0", err)

    def test_manifest_layout_removed_fails_even_without_sidecar(self) -> None:
        manifest = json.loads((self.tiled / "manifest.json").read_text(encoding="utf-8"))
        del manifest["backend_layout"]
        (self.tiled / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (self.out).unlink()
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("backend_layout", err)

    def test_bad_sidecar_schema_fails(self) -> None:
        del self.record["rules_sha256"]
        self.write_record(self.record)
        rc, _, err = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("rules_sha256", err)

    def test_expect_sha_must_be_full_hex(self) -> None:
        rc, _, err = self.verify(expect_arm="deadbeef")
        self.assertEqual(rc, 2)
        self.assertIn("sha256", err)


class LegacyCompatibilityTests(ProvenanceTestCase):
    def test_missing_default_sidecar_warns_and_passes(self) -> None:
        rc, stdout, err = self.verify(provenance=None)
        self.assertEqual(rc, 0, err)
        self.assertIn("warning: no provenance sidecar", err)
        self.assertIn("OK (legacy)", stdout)

    def test_require_provenance_fails_without_sidecar(self) -> None:
        rc, _, err = self.verify(provenance=None, require=True)
        self.assertEqual(rc, 3)
        self.assertIn("provenance required", err)

    def test_explicit_missing_sidecar_also_warns_and_passes(self) -> None:
        rc, stdout, err = self.verify(provenance=self.tmp / "does-not-exist.json")
        self.assertEqual(rc, 0, err)
        self.assertIn("warning: no provenance sidecar", err)
        self.assertIn("OK (legacy)", stdout)

        rc, _, err = self.verify(
            provenance=self.tmp / "does-not-exist.json", require=True
        )
        self.assertEqual(rc, 3)
        self.assertIn("provenance required", err)


class InputErrorTests(ProvenanceTestCase):
    def test_missing_arm_is_exit_2(self) -> None:
        inputs = dict(self.inputs)
        inputs["arm"] = self.tmp / "missing.gguf"
        rc, _, err = self.stamp(inputs=inputs)
        self.assertEqual(rc, 2)
        self.assertIn("file not found", err)

    def test_missing_tiled_is_exit_2(self) -> None:
        rc, _, err = self.verify(tiled=self.tmp / "nope")
        self.assertEqual(rc, 2)
        self.assertIn("directory not found", err)

    def test_malformed_sidecar_json_is_exit_2(self) -> None:
        self.out.parent.mkdir(parents=True)
        self.out.write_text("{not json", encoding="utf-8")
        rc, _, err = self.verify(provenance=self.out)
        self.assertEqual(rc, 2)
        self.assertIn("not valid JSON", err)

    def test_stamp_rejects_raw_manifest(self) -> None:
        manifest = json.loads((self.tiled / "manifest.json").read_text(encoding="utf-8"))
        del manifest["backend_layout"]
        (self.tiled / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rc, _, err = self.stamp()
        self.assertEqual(rc, 2)
        self.assertIn("invalid tiled package", err)


class HelpTests(unittest.TestCase):
    def run_help(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                qp.main(argv)
        return caught.exception.code or 0, stdout.getvalue()

    def test_top_level_help_documents_exit_codes(self) -> None:
        rc, text = self.run_help(["--help"])
        self.assertEqual(rc, 0)
        self.assertIn("exit codes:", text)
        self.assertIn("2  input", text)
        self.assertIn("3  provenance verification failed", text)

    def test_verify_help_documents_legacy_behavior(self) -> None:
        rc, text = self.run_help(["verify", "--help"])
        self.assertEqual(rc, 0)
        self.assertIn("--require-provenance", text)
        self.assertIn("exit codes:", text)


if __name__ == "__main__":
    unittest.main()
