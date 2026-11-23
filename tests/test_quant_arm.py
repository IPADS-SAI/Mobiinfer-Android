#!/usr/bin/env python3
"""Tests for tools/moe/quant_arm.py (stdlib unittest only).

Run from the repository root:

    python3 -m unittest tests.test_quant_arm -v

No llama.cpp binary and no GPU is used: ``plan`` is pure metadata, and the
resume tests inject a runner that fails if anything is executed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.moe import quant_arm as qa  # noqa: E402
from tools.moe import quant_rule_lint as qrl  # noqa: E402
from tests.test_quant_rule_lint import (  # noqa: E402
    build_gguf,
    kv_string,
    kv_u32,
)

SPECIFIC_RULE = {
    "regex": r"blk\.(36|37|38|39)\.ffn_(gate|up|down)_exps\.weight",
    "type": "Q2_K",
}
CATCH_ALL_RULE = {
    "regex": r"ffn_(gate|up|down)_exps\.weight",
    "type": "Q1_0",
}

EXPERT_NE0 = 2048
EXPERT_NE1 = 616
EXPERT_NE2 = 256
EXPERT_LAYERS = 40


def make_arm(
    tmp: Path,
    *,
    rules: list[dict[str, str]] | None = None,
    pack_enabled: bool = False,
    base_model: Path | None = None,
    imatrix: Path | None = None,
) -> Path:
    base = base_model or (tmp / "base.gguf")
    corpus = tmp / "corpus.txt"
    if base_model is None:
        base.write_bytes(b"fake-bf16")
    corpus.write_text("hello world\n", encoding="utf-8")
    arm = {
        "id": "test-arm",
        "base_model": str(base),
        "default_type": "Q6_K",
        "rules": rules or [SPECIFIC_RULE, CATCH_ALL_RULE],
        "ppl": {
            "corpus": str(corpus),
            "chunks": 128,
            "ctx": 2048,
            "batch": 2048,
            "ubatch": 512,
            "threads": 16,
        },
        "pack": {"enabled": pack_enabled, "expert_order": None},
    }
    if imatrix is not None:
        arm["imatrix"] = str(imatrix)
    path = tmp / "arm.json"
    path.write_text(json.dumps(arm), encoding="utf-8")
    return path


def make_budget_model(tmp: Path) -> Path:
    """Synthetic 40-layer GGUF header with 3 expert tensors per layer.

    ``ne1=616`` makes the tiled expert package ~5.9 GiB, i.e. the Qwen3-MoE-scale
    arm the budget gate is meant to reject.  Only the header is written.
    """
    tensors: list[tuple[str, list[int], int]] = []
    for layer in range(EXPERT_LAYERS):
        for family in ("gate", "up", "down"):
            tensors.append(
                (
                    f"blk.{layer}.ffn_{family}_exps.weight",
                    [EXPERT_NE0, EXPERT_NE1, EXPERT_NE2],
                    1,  # F16 source, as in a bf16/f16 base model
                )
            )
        tensors.append((f"blk.{layer}.attn_q.weight", [EXPERT_NE0, EXPERT_NE0], 1))
        tensors.append((f"blk.{layer}.attn_k.weight", [EXPERT_NE0, 512], 1))
    tensors.append(("token_embd.weight", [EXPERT_NE0, 32000], 1))
    metadata = [
        kv_string("general.architecture", "qwen35moe"),
        kv_u32("qwen35moe.block_count", EXPERT_LAYERS),
        kv_u32("qwen35moe.expert_count", 256),
        kv_u32("general.alignment", 32),
    ]
    path = tmp / "budget-model.gguf"
    path.write_bytes(build_gguf(tensors, metadata))
    return path


def expected_expert_raw_bytes() -> int:
    q1 = qrl.GGML_TYPES_BY_NAME["Q1_0"]
    q2 = qrl.GGML_TYPES_BY_NAME["Q2_K"]
    per_q1 = qrl.row_size(q1, EXPERT_NE0) * EXPERT_NE1 * EXPERT_NE2
    per_q2 = qrl.row_size(q2, EXPERT_NE0) * EXPERT_NE1 * EXPERT_NE2
    return 36 * 3 * per_q1 + 4 * 3 * per_q2


class RuleOrderTests(unittest.TestCase):
    def test_specific_rule_before_catch_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            arm_path = make_arm(tmp_path)
            plan = qa.build_plan(arm_path, tmp_path / "out")
            self.assertEqual(
                [rule["type"] for rule in plan.cfg["rules"]], ["Q2_K", "Q1_0"]
            )
            qa.write_plan(plan)
            lines = plan.rules_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], f"{SPECIFIC_RULE['regex']}=Q2_K")
            self.assertEqual(lines[1], f"{CATCH_ALL_RULE['regex']}=Q1_0")
            self.assertLess(
                lines.index(f"{SPECIFIC_RULE['regex']}=Q2_K"),
                lines.index(f"{CATCH_ALL_RULE['regex']}=Q1_0"),
            )

    def test_catch_all_before_specific_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            arm_path = make_arm(tmp_path, rules=[CATCH_ALL_RULE, SPECIFIC_RULE])
            with self.assertRaises(qa.ArmError) as ctx:
                qa.build_plan(arm_path, tmp_path / "out")
            self.assertIn("shadowed", str(ctx.exception))

    def test_rule_file_permissions_0644(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = qa.build_plan(make_arm(tmp_path), tmp_path / "out")
            qa.write_plan(plan)
            self.assertEqual(stat.S_IMODE(plan.rules_path.stat().st_mode), 0o644)

    def test_unknown_type_and_dotdot_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bad_type = make_arm(
                tmp_path,
                rules=[{"regex": r"blk\.0\.attn_q\.weight", "type": "NOPE"}],
            )
            with self.assertRaises(qa.ArmError):
                qa.build_plan(bad_type, tmp_path / "out")

            raw = json.loads(make_arm(tmp_path).read_text(encoding="utf-8"))
            raw["base_model"] = "/tmp/../etc/passwd"
            dotdot = tmp_path / "dotdot.json"
            dotdot.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(qa.ArmError):
                qa.build_plan(dotdot, tmp_path / "out")


class PplParsingTests(unittest.TestCase):
    def test_regular_estimate(self) -> None:
        text = "[1]4.5,[2]4.4\nFinal estimate: PPL = 6.2539 +/- 0.04590\n"
        self.assertEqual(qa.parse_ppl(text), {"ppl": 6.2539, "stddev": 0.0459})

    def test_scientific_notation(self) -> None:
        text = "Final estimate: PPL = 6.2539e+00 +/- 4.590e-02\n"
        self.assertEqual(qa.parse_ppl(text), {"ppl": 6.2539, "stddev": 0.0459})

    def test_last_estimate_wins(self) -> None:
        text = (
            "Final estimate: PPL = 9.0 +/- 1.0\n"
            "Final estimate: PPL = 6.1 +/- 0.2\n"
        )
        self.assertEqual(qa.parse_ppl(text), {"ppl": 6.1, "stddev": 0.2})

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(qa.parse_ppl("perplexity: nope\n"))


class PlanCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = ROOT / "tools" / "moe" / "arms" / "q1_0-q2k-example.json"

    def test_fixed_flags_and_quantize_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "arm-plan"
            plan = qa.build_plan(self.example, out)
            qa.write_plan(plan)
            text = plan.commands_path.read_text(encoding="utf-8")

            self.assertIn(qa.PPL_FIXED_FLAGS, text)
            self.assertIn(
                "-c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128", text
            )
            quantize = [
                line for line in text.splitlines() if "--tensor-type-file" in line
            ]
            self.assertEqual(len(quantize), 1)
            self.assertIn(" Q6_K 16 > ", quantize[0])
            self.assertIn('--tensor-type-file "$OUT/rules.txt"', quantize[0])

            self.assertIn("python3 tools/moe/experts.py extract", text)
            self.assertIn(
                'LD_LIBRARY_PATH="$LIB" python3 tools/moe/pack_hexagon.py', text
            )
            self.assertIn('--lib-dir "$LIB"', text)

            # Specific layer rule must precede the catch-all in commands.sh too.
            self.assertLess(
                text.index("blk\\.(36|37|38|39)"),
                text.index("ffn_(gate|up|down)_exps\\.weight=Q1_0"),
            )

    def test_commands_shell_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "arm-plan"
            plan = qa.build_plan(self.example, out)
            qa.write_plan(plan)
            result = subprocess.run(
                ["bash", "-n", str(plan.commands_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_gpu_env_is_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = qa.build_plan(self.example, Path(tmp) / "out", gpu="3")
            text = qa.commands_text(plan)
            self.assertIn('CUDA_VISIBLE_DEVICES=3 "$P"', text)
            ppl_step = next(step for step in plan.steps if step.kind == "ppl")
            self.assertEqual(ppl_step.env, {"CUDA_VISIBLE_DEVICES": "3"})


class ImatrixTests(unittest.TestCase):
    """T-0035: optional ``imatrix`` arm field."""

    # Frozen pre-T-0035 ``commands.sh`` for the fixed golden arm below.  The
    # repo path is the only machine-dependent value; build/lib paths derive
    # from HOST_PACK_BUILD and are expanded exactly as the old code did.
    GOLDEN_COMMANDS = """#!/bin/sh
# Generated by tools/moe/quant_arm.py for arm golden-arm
# llama-quantize has no `--default` flag; the `--default Q6_K` role is the positional ftype at the end of the quantize command.
set -eu

cd {repo}
Q=~/workspace/csm/moe-exp/build-cuda/bin/llama-quantize
P=~/workspace/csm/moe-exp/build-cuda/bin/llama-perplexity
B={build_dir}
LIB={lib_dir}
OUT=/tmp/qa-golden/out
SRC=/tmp/qa-golden/base.gguf
MODEL=/tmp/qa-golden/out/models/golden-arm.gguf
CORPUS=/tmp/qa-golden/corpus.txt

mkdir -p "$OUT/models" "$OUT/logs" "$OUT/experts"

# [1/3] rules
cat > "$OUT/rules.txt" <<'ARM_RULES'
blk\\.(36|37|38|39)\\.ffn_(gate|up|down)_exps\\.weight=Q2_K
ffn_(gate|up|down)_exps\\.weight=Q1_0
ARM_RULES

# [2/3] quantize
"$Q" --tensor-type-file "$OUT/rules.txt" "$SRC" "$MODEL" Q6_K 16 > "$OUT/logs/quantize.log" 2>&1

# [3/3] ppl
"$P" -m "$MODEL" -f "$CORPUS" -c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128 > "$OUT/logs/ppl.log" 2>&1
"""

    def _golden_arm(self, tmp: Path) -> Path:
        arm = {
            "id": "golden-arm",
            "base_model": "/tmp/qa-golden/base.gguf",
            "default_type": "Q6_K",
            "rules": [SPECIFIC_RULE, CATCH_ALL_RULE],
            "ppl": {
                "corpus": "/tmp/qa-golden/corpus.txt",
                "chunks": 128,
                "ctx": 2048,
                "batch": 2048,
                "ubatch": 512,
                "threads": 16,
            },
            "pack": {"enabled": False},
            "quantize_threads": 16,
        }
        path = tmp / "golden-arm.json"
        path.write_text(json.dumps(arm), encoding="utf-8")
        return path

    def test_no_imatrix_commands_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = qa.build_plan(
                self._golden_arm(Path(tmp)), Path("/tmp/qa-golden/out")
            )
            expected = self.GOLDEN_COMMANDS.format(
                repo=qa.q(str(qa.REPO_ROOT)),
                build_dir=qa.q(str(Path(qa.HOST_PACK_BUILD).expanduser().resolve())),
                lib_dir=qa.q(
                    str(Path(qa.HOST_PACK_BUILD).expanduser().resolve() / "bin")
                ),
            )
            self.assertEqual(qa.commands_text(plan), expected)

            manifest = qa.plan_manifest(plan)
            self.assertIsNone(manifest["imatrix"])
            quantize = next(
                item for item in manifest["commands"] if item["kind"] == "quantize"
            )
            self.assertNotIn("--imatrix", quantize["shell"])
            self.assertEqual(
                quantize["argv"],
                [
                    str(Path(qa.QUANTIZE_BIN).expanduser()),
                    "--tensor-type-file",
                    "/tmp/qa-golden/out/rules.txt",
                    "/tmp/qa-golden/base.gguf",
                    "/tmp/qa-golden/out/models/golden-arm.gguf",
                    "Q6_K",
                    "16",
                ],
            )
            self.assertEqual(
                plan.rules_text,
                f"{SPECIFIC_RULE['regex']}=Q2_K\n{CATCH_ALL_RULE['regex']}=Q1_0\n",
            )

    def test_imatrix_plan_commands_manifest_and_shell_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            imatrix_dir = tmp_path / "imatrix dir"
            imatrix_dir.mkdir()
            imatrix = imatrix_dir / "my imatrix.gguf"
            payload = b"imatrix-bytes\x00with spaces\n" * 7
            imatrix.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()

            plan = qa.build_plan(
                make_arm(tmp_path, imatrix=imatrix), tmp_path / "out dir"
            )
            qa.write_plan(plan)

            self.assertEqual(plan.imatrix_sha_path, tmp_path / "out dir" / "imatrix.sha256")
            text = plan.commands_path.read_text(encoding="utf-8")
            self.assertIn('--imatrix "$IMATRIX" --tensor-type-file', text)
            self.assertLess(
                text.index('--imatrix "$IMATRIX"'),
                text.index('--tensor-type-file'),
            )
            self.assertIn(f"IMATRIX={qa.q(str(imatrix))}", text)
            self.assertIn('"$Q" --imatrix "$IMATRIX"', text)

            quantize = next(step for step in plan.steps if step.kind == "quantize")
            self.assertEqual(quantize.argv[1:3], ["--imatrix", str(imatrix)])
            self.assertEqual(quantize.argv[3], "--tensor-type-file")

            manifest = json.loads(plan.plan_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["imatrix"],
                {
                    "path": str(imatrix),
                    "sha256": digest,
                    "bytes": len(payload),
                    "mtime": manifest["imatrix"]["mtime"],
                },
            )
            self.assertTrue(manifest["imatrix"]["mtime"])

            bash = subprocess.run(
                ["bash", "-n", str(plan.commands_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(bash.returncode, 0, bash.stderr)

    def test_invalid_imatrix_paths_raise_arm_error_exit_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            directory = tmp_path / "imatrix-dir"
            directory.mkdir()
            cases = {
                "relative": "imatrix.gguf",
                "missing": str(tmp_path / "nope.gguf"),
                "directory": str(directory),
                "dotdot": str(tmp_path / ".." / "imatrix.gguf"),
            }
            for name, value in cases.items():
                raw = json.loads(make_arm(tmp_path).read_text(encoding="utf-8"))
                raw["imatrix"] = value
                arm_path = tmp_path / f"bad-{name}.json"
                arm_path.write_text(json.dumps(raw), encoding="utf-8")
                with self.subTest(case=name):
                    with self.assertRaises(qa.ArmError) as ctx:
                        qa.build_plan(arm_path, tmp_path / f"out-{name}")
                    self.assertIn("imatrix", str(ctx.exception))
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                        stderr
                    ):
                        rc = qa.main(
                            [
                                "plan",
                                "--arm",
                                str(arm_path),
                                "--out",
                                str(tmp_path / f"out-{name}"),
                            ]
                        )
                    self.assertEqual(rc, 1)
                    self.assertIn("error: imatrix", stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())

            # ``null`` is the same as omitting the field.
            raw = json.loads(make_arm(tmp_path).read_text(encoding="utf-8"))
            raw["imatrix"] = None
            null_arm = tmp_path / "null.json"
            null_arm.write_text(json.dumps(raw), encoding="utf-8")
            plan = qa.build_plan(null_arm, tmp_path / "out-null")
            self.assertIsNone(plan.cfg["imatrix"])

    def test_resume_invalidates_quantize_when_imatrix_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            imatrix = tmp_path / "imatrix.gguf"
            imatrix.write_bytes(b"first-version")
            plan = qa.build_plan(make_arm(tmp_path, imatrix=imatrix), tmp_path / "out")
            qa.write_plan(plan)

            plan.model_path.parent.mkdir(parents=True, exist_ok=True)
            plan.model_path.write_bytes(b"quantized-model")
            plan.ppl_log.parent.mkdir(parents=True, exist_ok=True)
            plan.ppl_log.write_text(
                "Final estimate: PPL = 6.2500 +/- 0.0100\n", encoding="utf-8"
            )
            first_sha = hashlib.sha256(b"first-version").hexdigest()
            plan.imatrix_sha_path.write_text(first_sha + "\n", encoding="utf-8")
            plan.result_path.write_text('{"status": "success"}\n', encoding="utf-8")

            self.assertTrue(qa.step_done(next(s for s in plan.steps if s.kind == "quantize"), plan))
            calls: list[str] = []

            def must_not_run(step: qa.Step, plan: qa.Plan) -> int:
                raise AssertionError(f"runner called for {step.name}")

            self.assertEqual(
                qa.run_plan(plan, resume=True, keep_models=True, runner=must_not_run), 0
            )
            self.assertEqual(calls, [])

            imatrix.write_bytes(b"second-version")
            quantize_step = next(s for s in plan.steps if s.kind == "quantize")
            self.assertFalse(qa.step_done(quantize_step, plan))
            self.assertTrue(qa.imatrix_changed(plan))

            def fake(step: qa.Step, plan: qa.Plan) -> tuple[int, float]:
                calls.append(step.name)
                if step.kind == "quantize":
                    plan.model_path.write_bytes(b"re-quantized-model")
                elif step.kind == "ppl":
                    plan.ppl_log.write_text(
                        "Final estimate: PPL = 6.2600 +/- 0.0100\n", encoding="utf-8"
                    )
                return 0, 0.0

            with mock.patch.object(qa, "QUANTIZE_BIN", sys.executable), \
                    mock.patch.object(qa, "PERPLEXITY_BIN", sys.executable):
                rc = qa.run_plan(plan, resume=True, keep_models=True, runner=fake)
            self.assertEqual(rc, 0)
            self.assertEqual(calls, ["quantize", "ppl"])
            second_sha = hashlib.sha256(b"second-version").hexdigest()
            self.assertEqual(
                plan.imatrix_sha_path.read_text(encoding="utf-8").strip(), second_sha
            )
            ledger = json.loads(plan.result_path.read_text(encoding="utf-8"))
            self.assertTrue(ledger["imatrix_changed"])
            self.assertEqual(ledger["imatrix"]["sha256"], second_sha)

    def test_run_writes_sidecar_and_ledger_only_with_imatrix(self) -> None:
        def fake(step: qa.Step, plan: qa.Plan) -> tuple[int, float]:
            if step.kind == "quantize":
                plan.model_path.parent.mkdir(parents=True, exist_ok=True)
                plan.model_path.write_bytes(b"quantized-model")
            elif step.kind == "ppl":
                plan.ppl_log.parent.mkdir(parents=True, exist_ok=True)
                plan.ppl_log.write_text(
                    "Final estimate: PPL = 6.2500 +/- 0.0100\n", encoding="utf-8"
                )
            return 0, 0.0

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            imatrix = tmp_path / "imatrix.gguf"
            imatrix.write_bytes(b"ledger-payload")
            digest = hashlib.sha256(b"ledger-payload").hexdigest()
            plan = qa.build_plan(make_arm(tmp_path, imatrix=imatrix), tmp_path / "out")
            qa.write_plan(plan)
            with mock.patch.object(qa, "QUANTIZE_BIN", sys.executable), \
                    mock.patch.object(qa, "PERPLEXITY_BIN", sys.executable):
                self.assertEqual(
                    qa.run_plan(plan, resume=False, keep_models=True, runner=fake), 0
                )
            self.assertEqual(
                plan.imatrix_sha_path.read_text(encoding="utf-8").strip(), digest
            )
            ledger = json.loads(plan.result_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["imatrix"]["path"], str(imatrix))
            self.assertEqual(ledger["imatrix"]["sha256"], digest)
            self.assertEqual(ledger["imatrix"]["bytes"], len(b"ledger-payload"))
            self.assertFalse(ledger["imatrix_changed"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = qa.build_plan(make_arm(tmp_path), tmp_path / "out")
            qa.write_plan(plan)
            with mock.patch.object(qa, "QUANTIZE_BIN", sys.executable), \
                    mock.patch.object(qa, "PERPLEXITY_BIN", sys.executable):
                self.assertEqual(
                    qa.run_plan(plan, resume=False, keep_models=True, runner=fake), 0
                )
            self.assertFalse(plan.imatrix_sha_path.exists())
            ledger = json.loads(plan.result_path.read_text(encoding="utf-8"))
            self.assertIsNone(ledger["imatrix"])
            self.assertNotIn("imatrix_changed", ledger)


class BudgetTests(unittest.TestCase):
    """T-0027: header-only expert byte prediction and --max-expert-gib.

    The model fixture is a byte-by-byte GGUF *header* with 40 layers x 3 expert
    tensors (Qwen3-MoE shape, ne2=256).  Its tiled expert package is ~5.9 GiB, so
    budgeting does not need a real 69 GB model or any tensor data.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="quant-arm-budget-")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.model = make_budget_model(self.tmp)
        self.arm = make_arm(self.tmp, base_model=self.model)

    def run_cli(self, *extra: str) -> tuple[int, str, str]:
        out = self.tmp / "out"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = qa.main(
                ["plan", "--arm", str(self.arm), "--out", str(out), *extra]
            )
        return rc, stdout.getvalue(), stderr.getvalue()

    def plan_json(self) -> dict:
        return json.loads((self.tmp / "out" / "plan.json").read_text(encoding="utf-8"))

    def test_raw_bytes_match_hand_calc_and_tiled_bounds(self) -> None:
        plan = qa.build_plan(self.arm, self.tmp / "out")
        budget = qa.build_budget(plan, max_expert_gib=100.0)
        plan.budget = budget
        qa.write_plan(plan)

        expected = expected_expert_raw_bytes()
        self.assertEqual(budget["expert_bytes_raw"], expected)
        self.assertEqual(budget["by_type"]["Q1_0"]["tensors"], 36 * 3)
        self.assertEqual(budget["by_type"]["Q2_K"]["tensors"], 4 * 3)
        self.assertEqual(budget["by_type"]["Q1_0"]["bytes_raw"], 36 * 3 * (
            qrl.row_size(qrl.GGML_TYPES_BY_NAME["Q1_0"], EXPERT_NE0)
            * EXPERT_NE1 * EXPERT_NE2
        ))
        self.assertEqual(len(budget["by_layer"]), EXPERT_LAYERS)
        self.assertEqual(
            budget["by_layer"][36]["families"]["gate"]["type"], "Q2_K"
        )
        self.assertEqual(
            budget["by_layer"][0]["families"]["gate"]["type"], "Q1_0"
        )

        raw = budget["expert_bytes_raw"]
        tiled = budget["expert_bytes_tiled_est"]
        self.assertGreaterEqual(tiled, raw)
        self.assertLessEqual(tiled, raw * 1.10)

        manifest = self.plan_json()
        self.assertEqual(manifest["budget"]["expert_bytes_raw"], expected)
        self.assertEqual(
            manifest["budget"]["expert_bytes_tiled_est"], tiled
        )
        self.assertIn("estimate", manifest)
        self.assertIn("tiled = ceil(raw * 1.03", manifest["estimate"]["formula"])
        self.assertFalse(manifest["budget"]["over_budget"])

    def test_over_budget_exits_3_and_prints_top10(self) -> None:
        plan = qa.build_plan(self.arm, self.tmp / "out")
        budget = qa.build_budget(plan, max_expert_gib=100.0)
        limit = budget["expert_bytes_tiled_est"] / qa.GIB - 0.01

        rc, _stdout, stderr = self.run_cli(
            "--max-expert-gib", f"{limit:.6f}"
        )
        self.assertEqual(rc, 3, stderr)
        manifest = self.plan_json()
        self.assertTrue(manifest["budget"]["over_budget"])
        self.assertAlmostEqual(manifest["budget"]["budget_gib"], limit, places=5)
        self.assertEqual(
            manifest["budget"]["expert_bytes_raw"], expected_expert_raw_bytes()
        )
        self.assertIn("largest expert tensors (Top-10)", stderr)
        self.assertIn("blk.36.ffn_gate_exps", stderr)
        self.assertIn("--allow-over-budget", stderr)
        # The plan is still written so the rejection is reproducible.
        self.assertTrue((self.tmp / "out" / "commands.sh").is_file())

    def test_under_budget_exits_0(self) -> None:
        plan = qa.build_plan(self.arm, self.tmp / "out")
        budget = qa.build_budget(plan, max_expert_gib=100.0)
        limit = budget["expert_bytes_tiled_est"] / qa.GIB + 0.5

        rc, stdout, stderr = self.run_cli(
            "--max-expert-gib", f"{limit:.6f}"
        )
        self.assertEqual(rc, 0, stderr)
        self.assertIn("budget: OK", stdout)
        self.assertFalse(self.plan_json()["budget"]["over_budget"])

    def test_allow_over_budget_exits_0_but_flags(self) -> None:
        plan = qa.build_plan(self.arm, self.tmp / "out")
        budget = qa.build_budget(plan, max_expert_gib=100.0)
        limit = budget["expert_bytes_tiled_est"] / qa.GIB - 0.01

        rc, _stdout, stderr = self.run_cli(
            "--max-expert-gib", f"{limit:.6f}", "--allow-over-budget"
        )
        self.assertEqual(rc, 0, stderr)
        manifest = self.plan_json()
        self.assertTrue(manifest["budget"]["over_budget"])
        self.assertTrue(manifest["budget"]["allow_over_budget"])
        self.assertIn("OVER", stderr)

    def test_budget_scope_raw_vs_tiled(self) -> None:
        plan = qa.build_plan(self.arm, self.tmp / "out")
        budget = qa.build_budget(plan, max_expert_gib=100.0)
        raw_gib = budget["expert_bytes_raw"] / qa.GIB
        tiled_gib = budget["expert_bytes_tiled_est"] / qa.GIB
        limit = (raw_gib + tiled_gib) / 2

        rc, _stdout, _stderr = self.run_cli(
            "--max-expert-gib", f"{limit:.6f}", "--budget-scope", "raw"
        )
        self.assertEqual(rc, 0)
        rc, _stdout, _stderr = self.run_cli(
            "--max-expert-gib", f"{limit:.6f}", "--budget-scope", "tiled"
        )
        self.assertEqual(rc, 3)

    def test_budget_json_detail(self) -> None:
        detail = self.tmp / "budget-detail.json"
        plan = qa.build_plan(self.arm, self.tmp / "out")
        budget = qa.build_budget(plan, max_expert_gib=100.0)
        limit = budget["expert_bytes_tiled_est"] / qa.GIB - 0.01

        rc, _stdout, _stderr = self.run_cli(
            "--max-expert-gib",
            f"{limit:.6f}",
            "--budget-json",
            str(detail),
        )
        self.assertEqual(rc, 3)
        payload = json.loads(detail.read_text(encoding="utf-8"))
        self.assertEqual(payload["arm_id"], "test-arm")
        self.assertEqual(payload["expert_bytes_raw"], expected_expert_raw_bytes())
        self.assertEqual(len(payload["by_layer"]), EXPERT_LAYERS)
        self.assertTrue(payload["over_budget"])

    def test_custom_expert_regex(self) -> None:
        plan = qa.build_plan(self.arm, self.tmp / "out")
        budget = qa.build_budget(
            plan,
            max_expert_gib=100.0,
            expert_regex=r"blk\.\d+\.ffn_gate_exps\.weight",
        )
        self.assertEqual(budget["tensor_count"], EXPERT_LAYERS)
        self.assertEqual(len(budget["by_layer"]), EXPERT_LAYERS)
        self.assertLess(budget["expert_bytes_raw"], expected_expert_raw_bytes())

    def test_fallback_chain_is_reused_from_lint(self) -> None:
        # ne0=192 is not divisible by Q6_K's block (256); llama-quant.cpp falls
        # back to Q8_0 (block 32).  The estimator must use the same chain.
        tensors = []
        for layer in range(2):
            for family in ("gate", "up", "down"):
                tensors.append(
                    (f"blk.{layer}.ffn_{family}_exps.weight", [192, 64, 8], 1)
                )
        model = self.tmp / "fallback-model.gguf"
        model.write_bytes(build_gguf(tensors))
        arm_dir = self.tmp / "fallback-arm"
        arm_dir.mkdir()
        arm = make_arm(
            arm_dir,
            rules=[{"regex": CATCH_ALL_RULE["regex"], "type": "Q6_K"}],
            base_model=model,
        )
        plan = qa.build_plan(arm, self.tmp / "fallback-out")
        budget = qa.build_budget(plan, max_expert_gib=100.0)

        self.assertEqual(list(budget["by_type"]), ["Q8_0"])
        self.assertEqual(budget["by_type"]["Q8_0"]["tensors"], 6)
        self.assertEqual(
            budget["by_layer"][0]["families"]["gate"]["fallback_from"], "Q6_K"
        )
        expected = (
            qrl.row_size(qrl.GGML_TYPES_BY_NAME["Q8_0"], 192) * 64 * 8 * 6
        )
        self.assertEqual(budget["expert_bytes_raw"], expected)

    def test_no_budget_keeps_legacy_plan_and_does_not_read_gguf(self) -> None:
        # make_arm() writes b"fake-bf16" as the base model; a header parse here
        # would fail, so this also proves the old path stays header-free.
        legacy_dir = self.tmp / "legacy"
        legacy_dir.mkdir()
        legacy_arm = make_arm(legacy_dir)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = qa.main(
                [
                    "plan",
                    "--arm",
                    str(legacy_arm),
                    "--out",
                    str(self.tmp / "legacy-out"),
                ]
            )
        self.assertEqual(rc, 0, stderr.getvalue())
        manifest = json.loads(
            (self.tmp / "legacy-out" / "plan.json").read_text(encoding="utf-8")
        )
        self.assertIn("budget", manifest)
        self.assertFalse(manifest["budget"]["enabled"])
        self.assertFalse(manifest["budget"]["over_budget"])
        self.assertIsNone(manifest["budget"]["expert_bytes_raw"])
        # Old fields still present and unchanged.
        for key in ("schema_version", "arm_id", "rules", "ppl", "pack", "paths", "commands"):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["rules"], [SPECIFIC_RULE, CATCH_ALL_RULE])


class ResumeTests(unittest.TestCase):
    def test_existing_result_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = qa.build_plan(make_arm(tmp_path), tmp_path / "out")
            qa.write_plan(plan)
            plan.result_path.write_text('{"status": "success"}\n', encoding="utf-8")

            def must_not_run(step: qa.Step, plan: qa.Plan) -> int:
                raise AssertionError(f"runner called for {step.name}")

            rc = qa.run_plan(
                plan, resume=True, keep_models=True, runner=must_not_run
            )
            self.assertEqual(rc, 0)

    def test_existing_model_and_ppl_log_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = qa.build_plan(make_arm(tmp_path), tmp_path / "out")
            qa.write_plan(plan)
            plan.model_path.parent.mkdir(parents=True, exist_ok=True)
            plan.model_path.write_bytes(b"quantized-model")
            plan.ppl_log.parent.mkdir(parents=True, exist_ok=True)
            plan.ppl_log.write_text(
                "Final estimate: PPL = 6.2500 +/- 0.0100\n", encoding="utf-8"
            )

            calls: list[str] = []

            def must_not_run(step: qa.Step, plan: qa.Plan) -> int:
                calls.append(step.name)
                raise AssertionError(f"runner called for {step.name}")

            rc = qa.run_plan(
                plan, resume=True, keep_models=True, runner=must_not_run
            )
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [])
            result = json.loads(plan.result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["ppl"]["estimate"]["ppl"], 6.25)
            self.assertTrue(plan.model_path.is_file())  # keep_models=True

    def test_keep_models_false_deletes_after_ppl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = qa.build_plan(make_arm(tmp_path), tmp_path / "out")
            qa.write_plan(plan)

            def fake(step: qa.Step, plan: qa.Plan) -> tuple[int, float]:
                if step.kind == "quantize":
                    plan.model_path.parent.mkdir(parents=True, exist_ok=True)
                    plan.model_path.write_bytes(b"quantized-bytes")
                elif step.kind == "ppl":
                    plan.ppl_log.parent.mkdir(parents=True, exist_ok=True)
                    plan.ppl_log.write_text(
                        "Final estimate: PPL = 6.2500 +/- 0.0100\n",
                        encoding="utf-8",
                    )
                return 0, 0.1

            with mock.patch.object(qa, "QUANTIZE_BIN", sys.executable), \
                    mock.patch.object(qa, "PERPLEXITY_BIN", sys.executable):
                rc = qa.run_plan(plan, resume=False, keep_models=False, runner=fake)

            self.assertEqual(rc, 0)
            self.assertFalse(plan.model_path.exists())
            result = json.loads(plan.result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["model"]["deleted"])
            self.assertIsNotNone(result["model"]["sha256"])

    def test_no_resume_executes_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = qa.build_plan(make_arm(tmp_path), tmp_path / "out")
            qa.write_plan(plan)
            calls: list[str] = []

            def failing(step: qa.Step, plan: qa.Plan) -> int:
                calls.append(step.name)
                return 9

            with mock.patch.object(qa, "QUANTIZE_BIN", sys.executable):
                rc = qa.run_plan(
                    plan, resume=False, keep_models=True, runner=failing
                )
            self.assertEqual(rc, 1)
            self.assertEqual(calls, ["quantize"])
            result = json.loads(plan.result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failure")
            self.assertEqual(result["commands"][-1]["rc"], 9)


class ReportTests(unittest.TestCase):
    def test_tsv_from_three_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = [
                ("arm-a", 6.2539, 18.82, 13.500, "a" * 64),
                ("arm-b", 6.3001, 16.16, 11.375, "b" * 64),
                ("arm-c", 6.4, 14.84, None, "c" * 64),
            ]
            for arm_id, ppl, gib, mib, sha in fixtures:
                directory = root / arm_id
                directory.mkdir()
                record = {
                    "arm_id": arm_id,
                    "status": "success",
                    "ppl": {"estimate": {"ppl": ppl, "stddev": 0.01}},
                    "model": {"bytes": int(gib * (1 << 30)), "sha256": sha},
                    "experts": {"mib_per_step": mib},
                }
                (directory / "arm-result.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )

            records, warnings = qa.collect_results([str(root)])
            self.assertEqual(warnings, [])
            self.assertEqual(len(records), 3)
            tsv = qa.render_tsv(records)
            lines = tsv.strip().splitlines()
            self.assertEqual(
                lines[0],
                "id\tPPL\tmodel_GiB\texperts_MiB_per_step\tsha256_prefix\timatrix_sha",
            )
            self.assertEqual(len(lines), 4)
            self.assertTrue(
                lines[1].startswith(
                    "arm-a\t6.2539\t18.82\t13.500\taaaaaaaaaaaa\t-"
                )
            )
            self.assertIn("arm-c\t6.4000\t14.84\tNA\tcccccccccccc\t-", tsv)
            markdown = qa.render_markdown(records)
            self.assertIn(
                "| id | PPL | model GiB | experts MiB/step | sha256 | imatrix_sha |",
                markdown,
            )
            self.assertIn("| arm-b | 6.3001 |", markdown)

    def test_report_with_and_without_imatrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {
                    "arm_id": "weighted",
                    "ppl": {"estimate": {"ppl": 6.1}},
                    "model": {"bytes": 1 << 30, "sha256": "a" * 64},
                    "imatrix": {"path": "/tmp/im.gguf", "sha256": "d" * 64, "bytes": 9},
                },
                {
                    "arm_id": "plain",
                    "ppl": {"estimate": {"ppl": 6.2}},
                    "model": {"bytes": 1 << 30, "sha256": "b" * 64},
                },
            ]
            for record in records:
                directory = root / record["arm_id"]
                directory.mkdir()
                (directory / "arm-result.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
            collected, warnings = qa.collect_results([str(root)])
            self.assertEqual(warnings, [])
            tsv = qa.render_tsv(collected)
            self.assertIn("imatrix_sha", tsv.splitlines()[0])
            self.assertIn("weighted\t6.1000\t1.00\tNA\taaaaaaaaaaaa\tdddddddddddd", tsv)
            self.assertIn("plain\t6.2000\t1.00\tNA\tbbbbbbbbbbbb\t-", tsv)
            markdown = qa.render_markdown(collected)
            self.assertIn("imatrix_sha", markdown.splitlines()[0])
            self.assertIn("`dddddddddddd`", markdown)
            self.assertIn("| plain | 6.2000 | 1.00 | NA | `bbbbbbbbbbbb` | - |", markdown)


if __name__ == "__main__":
    unittest.main()
