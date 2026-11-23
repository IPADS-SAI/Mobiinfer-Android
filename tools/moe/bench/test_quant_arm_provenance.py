#!/usr/bin/env python3
"""Tests for T-0088: quant_arm.py + q1_provenance.py integration.

Stdlib unittest only, no GPU/model/phone.  The tiled package is the same
synthetic 2-layer x 3-family x 4-expert fixture used by
``tests/test_q1_provenance.py``; the quantizer/perplexity binaries are replaced
by a fake runner, so only metadata and tiny files are involved.

Run from the repository root:

    python3 -m unittest discover -s tools/moe/bench \\
        -p 'test_quant_arm_provenance.py' -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_q1_provenance import (  # noqa: E402
    file_hashes,
    make_inputs,
    make_tiled,
    sha256,
)
from tests.test_quant_rule_lint import (  # noqa: E402
    build_gguf,
    kv_string,
    kv_u32,
)
from tools.moe import q1_provenance as qp  # noqa: E402
from tools.moe import quant_arm as qa  # noqa: E402

EXPERT_RULE = {
    "regex": r"ffn_(gate|up|down)_exps\.weight",
    "type": "Q1_0",
}
PPL_LINE = "Final estimate: PPL = 6.2500 +/- 0.0100\n"


def sha256_text(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_arm(
    tmp: Path,
    *,
    imatrix: Path | None = None,
    pack_enabled: bool = True,
    provenance: dict | None = None,
) -> Path:
    """Minimal valid arm JSON; pack paths point at a synthetic lib dir."""
    base = tmp / "base.gguf"
    base.write_bytes(b"fake-bf16")
    corpus = tmp / "corpus.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    lib = tmp / "lib"
    lib.mkdir(exist_ok=True)
    (lib / "libggml-base.so").write_bytes(b"")
    (lib / "libggml-hexagon.so").write_bytes(b"")

    arm: dict = {
        "id": "prov-arm",
        "base_model": str(base),
        "default_type": "Q6_K",
        "rules": [EXPERT_RULE],
        "ppl": {
            "corpus": str(corpus),
            "chunks": 128,
            "ctx": 2048,
            "batch": 2048,
            "ubatch": 512,
            "threads": 16,
        },
        "pack": {"enabled": pack_enabled, "lib_dir": str(lib)},
    }
    if imatrix is not None:
        arm["imatrix"] = str(imatrix)
    if provenance is not None:
        arm["provenance"] = provenance
    path = tmp / "arm.json"
    path.write_text(json.dumps(arm), encoding="utf-8")
    return path


def valid_gguf(*, with_imatrix_kv: Path | None = None) -> bytes:
    metadata = [kv_string("general.architecture", "qwen35moe")]
    if with_imatrix_kv is not None:
        metadata.append(kv_string("quantize.imatrix.file", str(with_imatrix_kv)))
        metadata.append(kv_u32("quantize.imatrix.chunks_count", 128))
    return build_gguf([], metadata=metadata)


class Recorder:
    """Fake runner: creates the outputs and forwards provenance to the tool."""

    def __init__(
        self,
        *,
        model: bytes | None = None,
        stamp_rc: int = 0,
        verify_rc: int = 0,
    ) -> None:
        self.model = model if model is not None else valid_gguf()
        self.stamp_rc = stamp_rc
        self.verify_rc = verify_rc
        self.calls: list[str] = []

    def __call__(self, step: qa.Step, plan: qa.Plan):
        self.calls.append(step.name)
        if step.kind == "quantize":
            plan.model_path.parent.mkdir(parents=True, exist_ok=True)
            plan.model_path.write_bytes(self.model)
            return 0, 0.0
        if step.kind == "ppl":
            plan.ppl_log.parent.mkdir(parents=True, exist_ok=True)
            plan.ppl_log.write_text(PPL_LINE, encoding="utf-8")
            return 0, 0.0
        if step.kind == "extract":
            plan.raw_dir.mkdir(parents=True, exist_ok=True)
            (plan.raw_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            return 0, 0.0
        if step.kind == "pack":
            return 0, 0.0
        if step.name == "stamp":
            if self.stamp_rc:
                return self.stamp_rc, 0.0
            return self._run_provenance(step), 0.0
        if step.name == "verify":
            if self.verify_rc:
                return self.verify_rc, 0.0
            return self._run_provenance(step), 0.0
        return 0, 0.0

    @staticmethod
    def _run_provenance(step: qa.Step) -> int:
        """Run q1_provenance's main() with its output in the step log."""
        assert step.log is not None
        step.log.parent.mkdir(parents=True, exist_ok=True)
        with step.log.open("w", encoding="utf-8") as handle, \
                contextlib.redirect_stdout(handle), \
                contextlib.redirect_stderr(handle):
            return int(qp.main(step.argv[2:]))


class Q1ProvenanceNullImatrixTests(unittest.TestCase):
    """T-0088: ``--imatrix`` is optional and the sidecar may record null."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="q1-prov-null-")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.tiled = make_tiled(self.tmp)
        self.inputs = make_inputs(
            self.tmp, imatrix_kv=False, arm_name="unweighted.gguf"
        )
        self.out = self.tmp / "out" / "provenance.json"

    def stamp(self, *, weighted: str = "auto", imatrix: bool = False) -> tuple[int, str, str]:
        argv = [
            "stamp",
            "--tiled",
            str(self.tiled),
            "--arm",
            str(self.inputs["arm"]),
            "--rules",
            str(self.inputs["rules"]),
            "--quantizer-src",
            str(self.inputs["quantizer"]),
            "--weighted-q1-0",
            weighted,
            "--out",
            str(self.out),
        ]
        if imatrix:
            argv += ["--imatrix", str(self.inputs["imatrix"])]
        return self.run_cli(argv)

    def verify(self, *extra: str) -> tuple[int, str, str]:
        return self.run_cli(
            [
                "verify",
                "--tiled",
                str(self.tiled),
                "--provenance",
                str(self.out),
                "--require-provenance",
                *extra,
            ]
        )

    @staticmethod
    def run_cli(argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = qp.main(argv)
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_stamp_without_imatrix_records_null_and_verifies(self) -> None:
        before = file_hashes(self.tiled)
        rc, stdout, stderr = self.stamp()
        self.assertEqual(rc, 0, stderr)
        self.assertIn("imatrix=none (null)", stdout)

        record = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertIsNone(record["imatrix_sha256"])
        self.assertIsNone(record["imatrix_bytes"])
        self.assertIsNone(record["source_paths"]["imatrix"])
        self.assertEqual(record["weighted_q1_0"], "unknown")
        self.assertEqual(record["weighted_q1_0_source"], "missing-imatrix")
        # Never label a no-imatrix package as weighted.
        self.assertNotEqual(record["weighted_q1_0"], "true")

        rc, stdout, stderr = self.verify()
        self.assertEqual(rc, 0, stderr)
        self.assertIn("imatrix=none (null)", stdout)
        self.assertIn("weighted_q1_0=unknown", stdout)

        # The sidecar is additive; manifest/layer bytes stay identical.
        self.assertEqual(file_hashes(self.tiled), before)

    def test_stamp_without_imatrix_explicit_false(self) -> None:
        rc, _, stderr = self.stamp(weighted="false")
        self.assertEqual(rc, 0, stderr)
        record = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(record["weighted_q1_0"], "false")
        self.assertEqual(record["weighted_q1_0_source"], "explicit")
        self.assertIsNone(record["imatrix_sha256"])
        self.assertEqual(self.verify()[0], 0)

    def test_stamp_without_imatrix_rejects_true(self) -> None:
        rc, _, stderr = self.stamp(weighted="true")
        self.assertEqual(rc, 2)
        self.assertIn("requires --imatrix", stderr)
        self.assertFalse(self.out.exists())

    def test_expect_imatrix_sha_on_null_sidecar_fails_exit_3(self) -> None:
        self.assertEqual(self.stamp()[0], 0)
        rc, _, stderr = self.verify("--expect-imatrix-sha", "a" * 64)
        self.assertEqual(rc, 3)
        self.assertIn("imatrix_sha256=null", stderr)

    def test_null_imatrix_sidecar_cannot_claim_weighted_true(self) -> None:
        self.assertEqual(self.stamp()[0], 0)
        record = json.loads(self.out.read_text(encoding="utf-8"))
        record["weighted_q1_0"] = "true"
        self.out.write_text(json.dumps(record), encoding="utf-8")
        rc, _, stderr = self.verify()
        self.assertEqual(rc, 3)
        self.assertIn("non-null imatrix_sha256", stderr)

    def test_old_sidecar_with_imatrix_is_still_compatible(self) -> None:
        inputs = make_inputs(self.tmp, imatrix_kv=True, arm_name="weighted.gguf")
        rc, _, stderr = self.run_cli(
            [
                "stamp",
                "--tiled",
                str(self.tiled),
                "--arm",
                str(inputs["arm"]),
                "--imatrix",
                str(inputs["imatrix"]),
                "--rules",
                str(inputs["rules"]),
                "--quantizer-src",
                str(inputs["quantizer"]),
                "--weighted-q1-0",
                "auto",
                "--out",
                str(self.out),
            ]
        )
        self.assertEqual(rc, 0, stderr)
        record = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(record["imatrix_sha256"], sha256(inputs["imatrix"]))
        self.assertEqual(record["imatrix_bytes"], inputs["imatrix"].stat().st_size)
        self.assertEqual(record["weighted_q1_0"], "true")
        rc, _, stderr = self.verify(
            "--expect-imatrix-sha", record["imatrix_sha256"]
        )
        self.assertEqual(rc, 0, stderr)


class PlanProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="quant-arm-plan-prov-")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.imatrix = self.tmp / "imatrix.gguf"
        self.imatrix.write_bytes(b"synthetic-imatrix-data\n")

    def build(self, arm: Path, name: str = "out") -> qa.Plan:
        plan = qa.build_plan(arm, self.tmp / name)
        qa.write_plan(plan)
        return plan

    def test_stamp_and_verify_follow_pack_with_imatrix(self) -> None:
        plan = self.build(make_arm(self.tmp, imatrix=self.imatrix))
        text = plan.commands_path.read_text(encoding="utf-8")

        self.assertEqual(
            [step.name for step in plan.steps][-3:], ["pack", "stamp", "verify"]
        )
        pack_pos = text.index("pack_hexagon.py")
        stamp_pos = text.index("q1_provenance.py stamp")
        verify_pos = text.index("q1_provenance.py verify")
        self.assertLess(pack_pos, stamp_pos)
        self.assertLess(stamp_pos, verify_pos)
        self.assertIn('--imatrix "$IMATRIX" --weighted-q1-0 true', text)
        self.assertIn('--quantizer-src "$QUANTIZER_SRC"', text)
        self.assertIn('--expect-arm-sha "$ARM_SHA"', text)
        self.assertIn('--expect-imatrix-sha "$IMATRIX_SHA"', text)
        self.assertIn("--require-provenance", text)
        self.assertIn("QUANTIZER_SRC=", text)

        manifest = json.loads(plan.plan_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["provenance"]["enabled"])
        self.assertTrue(manifest["provenance"]["active"])
        self.assertEqual(manifest["provenance"]["weighted_q1_0"], "true")

        bash = subprocess.run(
            ["bash", "-n", str(plan.commands_path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(bash.returncode, 0, bash.stderr)

    def test_unweighted_plan_omits_imatrix_and_stamps_false(self) -> None:
        plan = self.build(make_arm(self.tmp, imatrix=None))
        text = plan.commands_path.read_text(encoding="utf-8")
        self.assertNotIn('--imatrix "$IMATRIX"', text)
        self.assertIn("--weighted-q1-0 false", text)
        self.assertNotIn("--expect-imatrix-sha", text)
        stamp = next(step for step in plan.steps if step.name == "stamp")
        self.assertNotIn("--imatrix", stamp.argv)
        verify = next(step for step in plan.steps if step.name == "verify")
        self.assertNotIn("--expect-imatrix-sha", verify.argv)
        bash = subprocess.run(
            ["bash", "-n", str(plan.commands_path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(bash.returncode, 0, bash.stderr)

    def test_provenance_disabled_has_no_steps(self) -> None:
        plan = self.build(
            make_arm(
                self.tmp, imatrix=self.imatrix, provenance={"enabled": False}
            )
        )
        self.assertNotIn("stamp", [step.name for step in plan.steps])
        self.assertNotIn("verify", [step.name for step in plan.steps])
        text = plan.commands_path.read_text(encoding="utf-8")
        self.assertNotIn("q1_provenance.py", text)


class RunProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="quant-arm-run-prov-")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.imatrix = self.tmp / "imatrix.gguf"
        self.imatrix.write_bytes(b"synthetic-imatrix-data\n")

    def build(
        self,
        *,
        imatrix: Path | None = None,
        provenance: dict | None = None,
        name: str = "out",
    ) -> qa.Plan:
        arm = make_arm(self.tmp, imatrix=imatrix, provenance=provenance)
        plan = qa.build_plan(arm, self.tmp / name)
        qa.write_plan(plan)
        # The fake "pack" step leaves this pre-built package in place.
        if not plan.tiled_dir.exists():
            shutil.copytree(make_tiled(self.tmp), plan.tiled_dir)
        return plan

    def _run(self, plan: qa.Plan, recorder: Recorder, *, resume: bool = False) -> int:
        with mock.patch.object(qa, "QUANTIZE_BIN", sys.executable), \
                mock.patch.object(qa, "PERPLEXITY_BIN", sys.executable), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return qa.run_plan(
                plan, resume=resume, keep_models=True, runner=recorder
            )

    def ledger(self, plan: qa.Plan) -> dict:
        return json.loads(plan.result_path.read_text(encoding="utf-8"))

    def test_success_writes_complete_provenance_and_keeps_tiled_bytes(self) -> None:
        plan = self.build(imatrix=self.imatrix)
        before = file_hashes(plan.tiled_dir)
        recorder = Recorder(model=valid_gguf(with_imatrix_kv=self.imatrix))
        self.assertEqual(self._run(plan, recorder), 0)

        provenance = self.ledger(plan)["provenance"]
        self.assertTrue(provenance["verify_ok"])
        self.assertEqual(provenance["status"], "ok")
        self.assertEqual(provenance["stamp_rc"], 0)
        self.assertEqual(provenance["verify_rc"], 0)
        self.assertEqual(provenance["sidecar"], str(plan.provenance_path))
        self.assertEqual(provenance["sidecar_sha256"], sha256(plan.provenance_path))
        self.assertEqual(provenance["weighted_q1_0"], "true")
        self.assertEqual(provenance["imatrix_sha256"], sha256(self.imatrix))
        self.assertEqual(provenance["arm_sha256"], sha256(plan.model_path))
        self.assertEqual(provenance["rules_sha256"], sha256(plan.rules_path))

        after = file_hashes(plan.tiled_dir)
        self.assertIn("provenance.json", after)
        after.pop("provenance.json")
        before.pop("provenance.json", None)
        self.assertEqual(after, before)
        self.assertLess(
            recorder.calls.index("pack"), recorder.calls.index("stamp")
        )
        self.assertLess(
            recorder.calls.index("stamp"), recorder.calls.index("verify")
        )

    def test_success_without_imatrix_records_null(self) -> None:
        plan = self.build(imatrix=None)
        recorder = Recorder()
        self.assertEqual(self._run(plan, recorder), 0)
        provenance = self.ledger(plan)["provenance"]
        self.assertEqual(provenance["status"], "ok")
        self.assertTrue(provenance["verify_ok"])
        self.assertIsNone(provenance["imatrix_sha256"])
        self.assertEqual(provenance["weighted_q1_0"], "false")
        record = json.loads(plan.provenance_path.read_text(encoding="utf-8"))
        self.assertIsNone(record["imatrix_sha256"])
        self.assertIsNone(record["imatrix_bytes"])

    def test_stamp_failure_marks_failure_and_mentions_stamp(self) -> None:
        plan = self.build(imatrix=self.imatrix)
        recorder = Recorder(stamp_rc=2)
        self.assertEqual(self._run(plan, recorder), 1)
        ledger = self.ledger(plan)
        self.assertEqual(ledger["status"], "failure")
        self.assertIn("stamp", ledger["error"])
        self.assertEqual(ledger["provenance"]["status"], "stamp_failed")
        self.assertEqual(ledger["provenance"]["stamp_rc"], 2)
        self.assertFalse(ledger["provenance"]["verify_ok"])

    def test_missing_quantizer_source_is_stamp_failed_not_skipped(self) -> None:
        missing = self.tmp / "missing-quants.c"
        plan = self.build(
            imatrix=self.imatrix,
            provenance={"quantizer_src": str(missing)},
        )
        self.assertIn("stamp", [step.name for step in plan.steps])
        recorder = Recorder()
        self.assertEqual(self._run(plan, recorder), 1)
        ledger = self.ledger(plan)
        self.assertEqual(ledger["status"], "failure")
        self.assertEqual(ledger["provenance"]["status"], "stamp_failed")
        self.assertIn("stamp", ledger["error"])
        self.assertIn("stamp", recorder.calls)

    def test_verify_failure_marks_verify_failed(self) -> None:
        plan = self.build(imatrix=self.imatrix)
        recorder = Recorder(verify_rc=3)
        self.assertEqual(self._run(plan, recorder), 1)
        ledger = self.ledger(plan)
        self.assertEqual(ledger["status"], "failure")
        self.assertIn("verify", ledger["error"])
        self.assertEqual(ledger["provenance"]["status"], "verify_failed")
        self.assertEqual(ledger["provenance"]["verify_rc"], 3)
        self.assertFalse(ledger["provenance"]["verify_ok"])
        # The stamp itself succeeded; the sidecar path is still recorded.
        self.assertEqual(
            ledger["provenance"]["sidecar_sha256"], sha256(plan.provenance_path)
        )

    def test_resume_skips_stamp_only_after_verify_passes(self) -> None:
        plan = self.build(imatrix=self.imatrix)
        self.assertEqual(self._run(plan, Recorder()), 0)
        self.assertTrue(plan.provenance_path.is_file())

        second = Recorder()
        self.assertEqual(self._run(plan, second, resume=True), 0)
        self.assertIn("verify", second.calls)
        self.assertNotIn("stamp", second.calls)
        self.assertEqual(self.ledger(plan)["provenance"]["status"], "ok")

        plan.provenance_path.unlink()
        third = Recorder()
        self.assertEqual(self._run(plan, third, resume=True), 0)
        self.assertIn("stamp", third.calls)
        self.assertIn("verify", third.calls)
        self.assertTrue(plan.provenance_path.is_file())

    def test_resume_restamps_when_existing_sidecar_is_invalid(self) -> None:
        plan = self.build(imatrix=self.imatrix)
        self.assertEqual(self._run(plan, Recorder()), 0)
        plan.provenance_path.write_text("{not json", encoding="utf-8")

        second = Recorder()
        self.assertEqual(self._run(plan, second, resume=True), 0)
        self.assertIn("stamp", second.calls)
        self.assertEqual(self.ledger(plan)["provenance"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
