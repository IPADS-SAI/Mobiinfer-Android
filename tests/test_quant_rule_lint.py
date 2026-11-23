#!/usr/bin/env python3
"""Tests for tools/moe/quant_rule_lint.py (stdlib unittest only).

Run from the repository root:

    python3 -m unittest tests.test_quant_rule_lint -v

The GGUF fixtures are synthetic headers built byte-by-byte; no tensor data is
written or read, which also exercises the linter's "header only" contract.
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.moe import quant_rule_lint as qrl  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic GGUF writer
# ---------------------------------------------------------------------------


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv_raw(key: str, value_type: int, payload: bytes) -> bytes:
    return pack_string(key) + struct.pack("<I", value_type) + payload


def kv_string(key: str, value: str) -> bytes:
    return kv_raw(key, qrl.GGUF_STRING, pack_string(value))


def kv_u32(key: str, value: int) -> bytes:
    return kv_raw(key, qrl.GGUF_UINT32, struct.pack("<I", value))


def kv_scalar(key: str, value_type: int, value) -> bytes:
    fmt, _ = qrl.GGUF_SCALAR_FORMATS[value_type]
    return kv_raw(key, value_type, struct.pack(fmt, value))


def kv_array(key: str, element_type: int, items: list[bytes]) -> bytes:
    payload = (
        struct.pack("<I", element_type)
        + struct.pack("<Q", len(items))
        + b"".join(items)
    )
    return kv_raw(key, qrl.GGUF_ARRAY, payload)


def tensor_info(name: str, dims: list[int], type_id: int, offset: int = 0) -> bytes:
    return (
        pack_string(name)
        + struct.pack("<I", len(dims))
        + b"".join(struct.pack("<Q", d) for d in dims)
        + struct.pack("<I", type_id)
        + struct.pack("<Q", offset)
    )


def build_gguf(
    tensors: list[tuple[str, list[int], int]],
    metadata: list[bytes] | None = None,
    *,
    version: int = 3,
    tensor_count: int | None = None,
    metadata_count: int | None = None,
) -> bytes:
    if metadata is None:
        metadata = [
            kv_string("general.architecture", "llama"),
            kv_u32("llama.block_count", 4),
            kv_u32("general.alignment", 32),
        ]
    if tensor_count is None:
        tensor_count = len(tensors)
    if metadata_count is None:
        metadata_count = len(metadata)

    body = b"".join(tensor_info(name, dims, type_id) for name, dims, type_id in tensors)
    return (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", metadata_count)
        + b"".join(metadata)
        + body
    )


def expert_tensors(layers: list[int]) -> list[tuple[str, list[int], int]]:
    return [
        (f"blk.{layer}.ffn_{family}_exps.weight", [256, 8, 4], 1)
        for layer in layers
        for family in ("gate", "up", "down")
    ]


class LintTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="quant-rule-lint-")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self._model_no = 0
        self._rules_no = 0
        self._json_no = 0

    def write_model(self, data: bytes) -> Path:
        self._model_no += 1
        path = self.tmp / f"model-{self._model_no}.gguf"
        path.write_bytes(data)
        return path

    def write_rules(self, text: str) -> Path:
        self._rules_no += 1
        path = self.tmp / f"rules-{self._rules_no}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def run_main(
        self,
        model: Path,
        *,
        rules: list[Path] | None = None,
        default: str = "Q6_K",
        strict: bool = False,
    ) -> tuple[int, dict, str]:
        self._json_no += 1
        json_path = self.tmp / f"out-{self._json_no}.json"
        argv = ["--model", str(model), "--default", default, "--json", str(json_path)]
        for path in rules or []:
            argv += ["--rules", str(path)]
        if strict:
            argv.append("--strict")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = qrl.main(argv)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return rc, payload, stdout.getvalue() + stderr.getvalue()

    @staticmethod
    def codes(payload: dict) -> set[str]:
        return {diag["code"] for diag in payload["diagnostics"]}

    @staticmethod
    def plans_by_name(payload: dict) -> dict[str, dict]:
        return {tensor["name"]: tensor for tensor in payload["tensors"]}


# ---------------------------------------------------------------------------
# GGUF parser
# ---------------------------------------------------------------------------


class GGUFParserTests(LintTestCase):
    def test_skips_every_metadata_type_and_arrays(self) -> None:
        metadata = [
            kv_string("general.architecture", "llama"),
            kv_u32("general.alignment", 32),
            kv_u32("llama.block_count", 4),
            kv_scalar("t.u8", qrl.GGUF_UINT8, 7),
            kv_scalar("t.i8", qrl.GGUF_INT8, -7),
            kv_scalar("t.u16", qrl.GGUF_UINT16, 65000),
            kv_scalar("t.i16", qrl.GGUF_INT16, -300),
            kv_scalar("t.u32", qrl.GGUF_UINT32, 123456),
            kv_scalar("t.i32", qrl.GGUF_INT32, -123),
            kv_scalar("t.f32", qrl.GGUF_FLOAT32, 1.5),
            kv_scalar("t.bool", qrl.GGUF_BOOL, True),
            kv_scalar("t.u64", qrl.GGUF_UINT64, 2**40),
            kv_scalar("t.i64", qrl.GGUF_INT64, -(2**40)),
            kv_scalar("t.f64", qrl.GGUF_FLOAT64, 3.25),
            kv_string("t.string", "hello"),
            kv_array("t.str_array", qrl.GGUF_STRING, [
                pack_string("a"), pack_string("bb"), pack_string("ccc"),
            ]),
            kv_array("t.u32_array", qrl.GGUF_UINT32, [
                struct.pack("<I", 1), struct.pack("<I", 2), struct.pack("<I", 3),
            ]),
            kv_array("t.f32_array", qrl.GGUF_FLOAT32, [
                struct.pack("<f", 0.5), struct.pack("<f", 1.5),
            ]),
        ]
        data = build_gguf(
            [("blk.0.ffn_gate_exps.weight", [256, 8, 4], 1)],
            metadata,
        )
        header = qrl.parse_gguf_header(self.write_model(data))
        self.assertEqual(header.version, 3)
        self.assertEqual(header.metadata_count, len(metadata))
        self.assertEqual(header.tensor_count, 1)
        self.assertEqual(header.metadata["general.architecture"], "llama")
        self.assertEqual(header.metadata["general.alignment"], 32)
        self.assertEqual(header.metadata["llama.block_count"], 4)
        self.assertEqual(header.tensors[0].dims, (256, 8, 4))
        self.assertEqual(header.data_start % 32, 0)

    def test_unknown_metadata_value_type_is_an_input_error(self) -> None:
        data = build_gguf(
            [],
            [kv_raw("bad.type", 99, b"")],
        )
        with self.assertRaisesRegex(qrl.LintError, "unknown GGUF metadata value type 99"):
            qrl.parse_gguf_header(self.write_model(data))

    def test_unknown_tensor_ggml_type_is_an_input_error(self) -> None:
        data = build_gguf([("blk.0.foo.weight", [256, 8], 31)])
        with self.assertRaisesRegex(qrl.LintError, "unknown ggml_type id 31"):
            qrl.parse_gguf_header(self.write_model(data))

    def test_bad_magic_and_truncated_header(self) -> None:
        good = build_gguf([("blk.0.foo.weight", [256, 8], 1)])
        bad_magic = self.write_model(b"NOPE" + good[4:])
        with self.assertRaisesRegex(qrl.LintError, "bad GGUF magic"):
            qrl.parse_gguf_header(bad_magic)

        truncated = self.write_model(good[:-10])
        with self.assertRaisesRegex(qrl.LintError, "truncated GGUF"):
            qrl.parse_gguf_header(truncated)

    def test_big_endian_magic_is_rejected_clearly(self) -> None:
        # A big-endian writer would emit magic 0x46554747 as b"FUGG".
        data = b"FUGG" + b"\x00\x00\x00\x03" + b"\x00" * 16
        with self.assertRaisesRegex(qrl.LintError, "big-endian"):
            qrl.parse_gguf_header(self.write_model(data))

    def test_wrong_version_is_rejected(self) -> None:
        data = build_gguf([], version=1)
        with self.assertRaisesRegex(qrl.LintError, "unsupported GGUF version 1"):
            qrl.parse_gguf_header(self.write_model(data))


# ---------------------------------------------------------------------------
# Rule file parsing
# ---------------------------------------------------------------------------


class RuleParsingTests(LintTestCase):
    def test_comments_blank_lines_and_multiple_tokens(self) -> None:
        rules = qrl.parse_rule_text(
            "\n".join([
                "# a comment",
                "",
                r"blk\.(0|1)\.ffn_.*=Q2_K   # trailing comment",
                "foo=Q4_K bar=Q8_0",
            ]),
            source="rules.txt",
        )
        self.assertEqual(len(rules), 3)
        self.assertEqual(rules[0].pattern, r"blk\.(0|1)\.ffn_.*")
        self.assertEqual(rules[0].type_id, 10)
        self.assertEqual(rules[1].line, 4)
        self.assertEqual(rules[2].type_id, 8)
        self.assertTrue(all(rule.valid for rule in rules))

    def test_invalid_regex_and_unknown_type_are_flagged(self) -> None:
        rules = qrl.parse_rule_text(
            r"blk\.(0=Q4_K" "\n" "foo=Q4_K_M\n" "nope\n",
            source="rules.txt",
        )
        self.assertEqual(len(rules), 3)
        self.assertFalse(rules[0].valid)
        self.assertIn("invalid regex", rules[0].error)
        self.assertFalse(rules[1].valid)
        self.assertIn("file type", rules[1].error)
        self.assertFalse(rules[2].valid)
        self.assertIn("malformed rule", rules[2].error)


# ---------------------------------------------------------------------------
# lint semantics
# ---------------------------------------------------------------------------


class LintSemanticsTests(LintTestCase):
    def test_layer_regex_does_not_match_longer_layer_numbers(self) -> None:
        model = self.write_model(build_gguf([
            (f"blk.{layer}.ffn_gate_exps.weight", [256, 8, 4], 1)
            for layer in (0, 1, 2, 3, 13, 30)
        ]))
        rules = self.write_rules(r"blk\.(0|1|2|3)\.ffn_.*=Q2_K" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q1_0")
        self.assertEqual(rc, 0)
        plans = self.plans_by_name(payload)
        for layer in (0, 1, 2, 3):
            tensor = plans[f"blk.{layer}.ffn_gate_exps.weight"]
            self.assertEqual(tensor["matched_rule"], 0)
            self.assertEqual(tensor["final_type"], "Q2_K")
        for layer in (13, 30):
            tensor = plans[f"blk.{layer}.ffn_gate_exps.weight"]
            self.assertIsNone(tensor["matched_rule"])
            self.assertEqual(tensor["final_type"], "Q1_0")

    def test_first_match_specific_rule_before_catch_all(self) -> None:
        model = self.write_model(build_gguf([
            (f"blk.{layer}.ffn_gate_exps.weight", [256, 8, 4], 1)
            for layer in (10, 36, 37)
        ]))
        rules = self.write_rules(
            r"blk\.(36|37)\.ffn_(gate|up|down)_exps\.weight=Q2_K" "\n"
            r"ffn_(gate|up|down)_exps\.weight=Q1_0" "\n"
        )
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 0)
        plans = self.plans_by_name(payload)
        self.assertEqual(plans["blk.36.ffn_gate_exps.weight"]["matched_rule"], 0)
        self.assertEqual(plans["blk.36.ffn_gate_exps.weight"]["final_type"], "Q2_K")
        self.assertEqual(plans["blk.37.ffn_gate_exps.weight"]["matched_rule"], 0)
        self.assertEqual(plans["blk.10.ffn_gate_exps.weight"]["matched_rule"], 1)
        self.assertEqual(plans["blk.10.ffn_gate_exps.weight"]["final_type"], "Q1_0")

    def test_specific_rule_after_catch_all_is_shadowed(self) -> None:
        model = self.write_model(build_gguf(
            expert_tensors([36, 37]),
        ))
        rules = self.write_rules(
            r"ffn_(gate|up|down)_exps\.weight=Q1_0" "\n"
            r"blk\.(36|37)\.ffn_(gate|up|down)_exps\.weight=Q2_K" "\n"
        )
        rc, payload, _ = self.run_main(
            model, rules=[rules], default="Q6_K", strict=True,
        )
        self.assertEqual(rc, 2)
        self.assertIn("rule_shadowed", self.codes(payload))
        self.assertTrue(payload["rules"][1]["shadowed"])
        self.assertEqual(payload["rules"][1]["won_tensors"], 0)
        for tensor in payload["tensors"]:
            self.assertEqual(tensor["final_type"], "Q1_0")

    def test_unmatched_rule_and_strict_exit(self) -> None:
        model = self.write_model(build_gguf(expert_tensors([0])))
        rules = self.write_rules(r"blk\.99\.ffn_.*=Q2_K" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q1_0")
        self.assertEqual(rc, 0)
        self.assertIn("rule_unmatched", self.codes(payload))
        self.assertTrue(payload["rules"][0]["valid"])

        rc_strict, payload_strict, _ = self.run_main(
            model, rules=[rules], default="Q1_0", strict=True,
        )
        self.assertEqual(rc_strict, 2)
        self.assertIn("rule_unmatched", self.codes(payload_strict))

    def test_rule_regex_is_lowercased_like_cpp(self) -> None:
        # parse_tensor_type() lowercases the regex before std::regex compiles
        # it, so an uppercase literal still matches lowercase GGUF names.
        model = self.write_model(build_gguf([
            ("blk.0.ffn_gate_exps.weight", [256, 8, 4], 1),
        ]))
        rules = self.write_rules(r"BLK\.0\.FFN_.*=Q4_K" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 0)
        self.assertEqual(payload["tensors"][0]["matched_rule"], 0)
        self.assertEqual(payload["rules"][0]["pattern"], r"blk\.0\.ffn_.*")
        self.assertEqual(payload["tensors"][0]["final_type"], "Q4_K")

    def test_multiple_rule_files_keep_file_order(self) -> None:
        model = self.write_model(build_gguf([
            ("blk.36.ffn_gate_exps.weight", [256, 8, 4], 1),
            ("blk.10.ffn_gate_exps.weight", [256, 8, 4], 1),
        ]))
        first = self.write_rules(
            r"blk\.36\.ffn_.*=Q2_K" + "\n"
        )
        second = self.write_rules(
            r"ffn_.*=Q1_0" + "\n"
        )
        rc, payload, _ = self.run_main(
            model, rules=[first, second], default="Q6_K",
        )
        self.assertEqual(rc, 0)
        self.assertEqual([rule["index"] for rule in payload["rules"]], [0, 1])
        plans = self.plans_by_name(payload)
        self.assertEqual(plans["blk.36.ffn_gate_exps.weight"]["matched_rule"], 0)
        self.assertEqual(plans["blk.10.ffn_gate_exps.weight"]["matched_rule"], 1)

    def test_unknown_tensor_family_is_reported(self) -> None:
        model = self.write_model(build_gguf([
            ("blk.0.attn_qkv.weight", [2048, 8192], 1),
        ]))
        rules = self.write_rules(r"blk\.\d+\.attn_qkvz\.weight=Q4_K" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 0)
        codes = self.codes(payload)
        self.assertIn("rule_unmatched", codes)
        self.assertIn("rule_unknown_family", codes)

    def test_invalid_type_is_input_error_exit_1(self) -> None:
        model = self.write_model(build_gguf([("blk.0.ffn_.weight", [256, 8], 1)]))
        rules = self.write_rules(r"blk\.0\.ffn_.*=Q9_Z" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 1)
        self.assertEqual(payload["status"], "error")
        self.assertIn("unknown GGML type", payload["diagnostics"][0]["message"])

    def test_invalid_regex_is_input_error_exit_1(self) -> None:
        model = self.write_model(build_gguf([("blk.0.ffn_.weight", [256, 8], 1)]))
        rules = self.write_rules(r"blk\.(0=Q4_K" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 1)
        self.assertIn("invalid_regex", self.codes(payload))
        self.assertEqual(payload["exit_code"], 1)

    def test_duplicate_rule_warning(self) -> None:
        model = self.write_model(build_gguf(expert_tensors([0])))
        rules = self.write_rules(
            r"blk\.0\.ffn_.*=Q4_K" "\n"
            r"blk\.0\.ffn_.*=Q4_K" "\n"
        )
        rc, payload, _ = self.run_main(
            model, rules=[rules], default="Q6_K", strict=True,
        )
        self.assertEqual(rc, 2)
        self.assertIn("duplicate_rule", self.codes(payload))
        self.assertEqual(payload["rules"][1]["duplicate_of"], 0)

    def test_non_quantizable_norm_tensor_keeps_original_type(self) -> None:
        model = self.write_model(build_gguf([
            ("blk.0.attn_norm.weight", [2048], 0),  # F32, 1-D
        ]))
        rules = self.write_rules(r"attn_norm\.weight=Q4_K" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 0)
        self.assertIn("rule_unmatched", self.codes(payload))
        tensor = payload["tensors"][0]
        self.assertFalse(tensor["allows_quantization"])
        self.assertIsNone(tensor["matched_rule"])
        self.assertEqual(tensor["final_type"], "F32")

    def test_default_not_quantized_disables_overrides(self) -> None:
        model = self.write_model(build_gguf(expert_tensors([0])))
        rules = self.write_rules(r"ffn_.*=Q1_0" + "\n")
        rc, payload, _ = self.run_main(
            model, rules=[rules], default="F16", strict=True,
        )
        self.assertEqual(rc, 2)
        self.assertFalse(payload["overrides_enabled"])
        self.assertIn("overrides_ignored_default_not_quantized", self.codes(payload))
        for tensor in payload["tensors"]:
            self.assertEqual(tensor["final_type"], "F16")
            self.assertFalse(tensor["rule_applied"])

    def test_shape_fallback_q6_k_to_q8_0(self) -> None:
        model = self.write_model(build_gguf([
            ("blk.0.attn_q.weight", [96, 4], 1),
        ]))
        rc, payload, _ = self.run_main(model, default="Q6_K")
        self.assertEqual(rc, 0)
        tensor = payload["tensors"][0]
        self.assertEqual(tensor["final_type"], "Q8_0")
        self.assertEqual(tensor["fallback_from"], "Q6_K")
        self.assertEqual(tensor["estimated_bytes"], 102 * 4)
        self.assertIn("shape_fallback", self.codes(payload))

    def test_shape_fallback_to_f16_when_no_32_multiple(self) -> None:
        model = self.write_model(build_gguf([
            ("blk.0.attn_q.weight", [40, 4], 1),
        ]))
        rc, payload, _ = self.run_main(model, default="Q6_K", strict=True)
        self.assertEqual(rc, 2)
        tensor = payload["tensors"][0]
        self.assertEqual(tensor["final_type"], "F16")
        self.assertEqual(tensor["fallback_from"], "Q6_K")
        self.assertIn("fallback_f16", self.codes(payload))

    def test_expert_coverage_all_covered_and_missing(self) -> None:
        model = self.write_model(build_gguf(expert_tensors([0, 1])))
        rules = self.write_rules(r"ffn_(gate|up|down)_exps\.weight=Q1_0" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 0)
        coverage = payload["summary"]["expert_coverage"]
        self.assertEqual(coverage["tensor_count"], 6)
        self.assertEqual(coverage["covered_by_rule"], 6)
        self.assertEqual(coverage["layer_count"], 2)
        self.assertEqual(coverage["uncovered"], [])

        partial = self.write_rules(
            r"blk\.0\.ffn_(gate|up|down)_exps\.weight=Q1_0" "\n"
        )
        rc2, payload2, _ = self.run_main(
            model, rules=[partial], default="Q6_K", strict=True,
        )
        self.assertEqual(rc2, 2)
        coverage2 = payload2["summary"]["expert_coverage"]
        self.assertEqual(coverage2["covered_by_rule"], 3)
        self.assertEqual(len(coverage2["uncovered"]), 3)
        self.assertIn("expert_tensor_uncovered", self.codes(payload2))

    def test_json_schema_and_header_only_contract(self) -> None:
        model = self.write_model(build_gguf(expert_tensors([0])))
        rules = self.write_rules(r"ffn_.*=Q1_0" + "\n")
        rc, payload, _ = self.run_main(model, rules=[rules], default="Q6_K")
        self.assertEqual(rc, 0)
        self.assertFalse(payload["model"]["tensor_data_read"])
        self.assertIn("byte_estimate_note", payload["model"])
        self.assertEqual(payload["tool"], "quant_rule_lint")
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("expert_coverage", payload["summary"])
        tensor = payload["tensors"][0]
        for key in (
            "name", "shape", "ggml_type", "default_type", "matched_rule",
            "final_type", "estimated_bytes", "allows_quantization",
        ):
            self.assertIn(key, tensor)


# ---------------------------------------------------------------------------
# Byte arithmetic (GGML_QUANT_SIZES block sizes)
# ---------------------------------------------------------------------------


class ByteEstimateTests(unittest.TestCase):
    def test_required_block_sizes(self) -> None:
        cases = {
            "Q1_0": (128, 18),
            "Q2_0": (64, 18),
            "Q2_K": (256, 84),
            "Q3_K": (256, 110),
            "Q4_K": (256, 144),
            "Q5_K": (256, 176),
            "Q6_K": (256, 210),
            "Q8_0": (32, 34),
            "F16": (1, 2),
            "BF16": (1, 2),
            "F32": (1, 4),
        }
        for name, (block, type_size) in cases.items():
            ggml_type = qrl.GGML_TYPES_BY_NAME[name]
            self.assertEqual(ggml_type.block, block, name)
            self.assertEqual(ggml_type.type_size, type_size, name)
            self.assertEqual(
                qrl.row_size(ggml_type, block),
                type_size,
                name,
            )

    def test_row_size_and_total_bytes(self) -> None:
        dims = (256, 3, 2)  # 6 rows
        cases = {
            "Q1_0": 36 * 6,
            "Q2_0": 72 * 6,
            "Q2_K": 84 * 6,
            "Q6_K": 210 * 6,
            "Q8_0": 272 * 6,
            "F16": 512 * 6,
            "BF16": 512 * 6,
            "F32": 1024 * 6,
        }
        for name, expected in cases.items():
            self.assertEqual(
                qrl.estimate_bytes(qrl.GGML_TYPES_BY_NAME[name], dims),
                expected,
                name,
            )

    def test_incompatible_ne0_raises(self) -> None:
        with self.assertRaises(ValueError):
            qrl.row_size(qrl.GGML_TYPES_BY_NAME["Q2_K"], 200)
        with self.assertRaises(ValueError):
            qrl.estimate_bytes(qrl.GGML_TYPES_BY_NAME["Q1_0"], [100, 2])


if __name__ == "__main__":
    unittest.main()
