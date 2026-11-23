#!/usr/bin/env python3
"""Offline, stdlib-only orchestrator for A100 mixed-precision quantization arms.

An arm is a small JSON description (base GGUF, ``--tensor-type-file`` rules,
fixed perplexity settings and an optional tiled-pack step).  The tool has three
subcommands:

``plan``
    Validate the arm and write ``rules.txt`` / ``commands.sh`` / ``plan.json``
    into ``--out``.  Nothing is executed.  ``--dry-run`` is a synonym.

``run``
    Execute the plan with ``subprocess``, skipping work already present when
    ``--resume`` is given.  Every command's rc/wall time is recorded, the
    quantized model's sha256/size is measured, the PPL estimate is parsed and a
    single ``arm-result.json`` ledger is written.  Unless ``--keep-models`` is
    passed the (large) quantized GGUF is deleted after packing/Perplexity.

``report``
    Aggregate one or more ``arm-result.json`` files into TSV or Markdown.

The fixed command shape is copied from the existing experiments
(``exps/quant-mixed-20261122/README.md`` §2.2-2.3 and
``exps/quant-partial-q4k-20261122/README.md`` §2.1/2.6):

* quantize: ``llama-quantize --tensor-type-file <rules> SRC OUT Q6_K 16``
  (llama-quantize takes the default type positionally; this tree has no
  ``--default`` flag).
* perplexity: ``-c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128``.
* pack: ``experts.py extract`` followed by ``pack_hexagon.py`` with the host
  shim build's ``lib`` directory on ``LD_LIBRARY_PATH``.

Only the Python standard library is used.  No GPU work is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

try:  # ``python3 -m ...`` / tests: repo root is on sys.path
    from tools.moe import quant_rule_lint as qrl
except ImportError:  # pragma: no cover - ``python3 tools/moe/quant_arm.py``
    import quant_rule_lint as qrl  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Constants copied from the experiment READMEs (do not silently drift)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUANTIZER_SRC = REPO_ROOT / "ggml" / "src" / "ggml-quants.c"

QUANTIZE_BIN = "~/workspace/csm/moe-exp/build-cuda/bin/llama-quantize"
PERPLEXITY_BIN = "~/workspace/csm/moe-exp/build-cuda/bin/llama-perplexity"
HOST_PACK_BUILD = "~/workspace/csm/moe-exp/build-host"

# Fixed PPL recipe: 128 chunks x 2048 ctx, deterministic across arms.
PPL_FIXED: dict[str, int] = {
    "chunks": 128,
    "ctx": 2048,
    "batch": 2048,
    "ubatch": 512,
    "threads": 16,
}
PPL_FIXED_FLAGS = "-c 2048 -b 2048 -ub 512 -ngl 99 -fit off -t 16 --chunks 128"
QUANTIZE_THREADS = 16

ARM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
GPU_RE = re.compile(r"^\d+(,\d+)*$")

# ggml type name -> (block, bytes per block).  Mirrors gguf-py constants; kept
# local because gguf-py imports numpy and this tool must stay stdlib-only.
GGML_QUANT_SIZES: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q8_1": (32, 40),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66),
    "IQ2_XS": (256, 74),
    "IQ3_XXS": (256, 98),
    "IQ1_S": (256, 50),
    "IQ4_NL": (32, 18),
    "IQ3_S": (256, 110),
    "IQ2_S": (256, 82),
    "IQ4_XS": (256, 136),
    "I8": (1, 1),
    "I16": (1, 2),
    "I32": (1, 4),
    "I64": (1, 8),
    "F64": (1, 8),
    "IQ1_M": (256, 56),
    "BF16": (1, 2),
    "TQ1_0": (256, 54),
    "TQ2_0": (256, 66),
    "MXFP4": (32, 17),
    "NVFP4": (64, 36),
    "Q1_0": (128, 18),
    "Q2_0": (64, 18),
}

# Probe tensor names used to detect a catch-all rule that shadows a later,
# more specific one.  The real quantizer is first-match-wins, so order matters.
PROBE_LAYERS = tuple(range(0, 48))
PROBE_FAMILIES = (
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "attn_q",
    "attn_k",
    "ffn_gate",
)
PROBE_TENSORS = tuple(
    f"blk.{layer}.{family}.weight"
    for layer in PROBE_LAYERS
    for family in PROBE_FAMILIES
) + ("token_embd.weight", "output.weight")

PPL_RE = re.compile(
    r"Final estimate:\s*PPL\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*\+/-\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# Expert-byte budget (T-0027).  The tiled figure is deliberately a conservative
# *estimate*: the packer aligns/heads each layer, but the exact amount depends on
# the backend layout.  Keep the 1.03 factor in sync with
# notes/quant-arm-budget.md and the synthetic tests.
DEFAULT_EXPERT_REGEX = r"ffn_(gate|up|down)_exps\.weight"
TILED_OVERHEAD_FACTOR = 1.03
TILED_HEADER_BYTES_PER_LAYER = 4096
TILED_FORMULA = (
    "tiled = ceil(raw * 1.03 + 4096 * layers); "
    "raw = sum(row_size(final_type, ne0) * ne1 * ne2 * ne3) over expert tensors"
)
GIB = 1 << 30
LAYER_RE = re.compile(r"blk\.(\d+)\.")
FAMILY_RE = re.compile(r"ffn_(gate|up|down)_exps")
BUDGET_SCOPES = ("tiled", "raw")


class ArmError(Exception):
    """Invalid arm JSON or unsafe path."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def q(text: str) -> str:
    return shlex.quote(text)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, mode: int = 0o644) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", mode)


def git_value(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def resolve_path(value: Any, field_name: str, *, allow_relative: bool = False) -> Path:
    """expanduser + resolve, but reject ``..`` before it can be normalised away."""
    if not isinstance(value, str) or not value.strip():
        raise ArmError(f"{field_name}: expected a non-empty path string")
    if "\x00" in value:
        raise ArmError(f"{field_name}: NUL byte in path")
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ArmError(f"{field_name}: '..' components are not allowed: {value!r}")
    if not allow_relative and not path.is_absolute():
        raise ArmError(f"{field_name}: expected an absolute path, got {value!r}")
    return path.resolve()


def resolve_out_dir(value: Any) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ArmError("--out: expected a path")
    return resolve_path(str(value), "--out", allow_relative=True)


def normalize_type(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ArmError(f"{field_name}: expected a ggml type name")
    name = value.strip().upper()
    if name not in GGML_QUANT_SIZES:
        raise ArmError(
            f"{field_name}: unknown ggml type {value!r} "
            f"(known: {', '.join(sorted(GGML_QUANT_SIZES))})"
        )
    if GGML_QUANT_SIZES[name][0] <= 1:
        raise ArmError(f"{field_name}: {name} is not a quantized type")
    return name


def _stat_or_none(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


# Sidecar recording the imatrix sha256 that produced the quantized model.  It
# lives next to plan.json so ``--resume`` can detect a re-generated imatrix.
IMATRIX_SIDECAR = "imatrix.sha256"


def imatrix_metadata(path: Path) -> dict[str, Any]:
    """Measure an imatrix file once: path/bytes/sha256/mtime."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ArmError(f"imatrix: cannot stat {path}: {exc}") from exc
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise ArmError(f"imatrix: cannot read {path}: {exc}") from exc
    return {
        "path": path,
        "bytes": stat.st_size,
        "sha256": digest,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
    }


def imatrix_json(info: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """JSON-safe view of :func:`imatrix_metadata` (``None`` stays ``None``)."""
    if not info:
        return None
    return {
        "path": str(info["path"]),
        "sha256": info["sha256"],
        "bytes": info["bytes"],
        "mtime": info["mtime"],
    }


# ---------------------------------------------------------------------------
# Arm validation / normalisation
# ---------------------------------------------------------------------------


def parse_ppl(text: str) -> Optional[dict[str, float]]:
    """Return the last ``Final estimate`` in *text*, or None.

    Handles both the regular form (``6.2539 +/- 0.04590``) and scientific
    notation (``6.2539e+00 +/- 4.590e-02``).
    """
    matches = list(PPL_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {"ppl": float(match.group(1)), "stddev": float(match.group(2))}


def _probe_rule(pattern: str) -> frozenset[str]:
    compiled = re.compile(pattern.lower())
    return frozenset(name for name in PROBE_TENSORS if compiled.search(name))


def validate_rule_order(rules: Sequence[dict[str, str]]) -> list[str]:
    """Reject a broad rule that fully shadows any later rule.

    ``llama-quantize`` walks ``--tensor-type-file`` entries in file order and
    stops at the first match.  A catch-all placed before a layer-specific rule
    therefore silently disables the specific rule.
    """
    warnings: list[str] = []
    matches = [_probe_rule(rule["regex"]) for rule in rules]
    for index, hit in enumerate(matches):
        if not hit:
            warnings.append(
                f"rule {index + 1} ({rules[index]['regex']!r}) matches none of the "
                "probe tensors; double-check the regex"
            )
    for earlier in range(len(rules)):
        for later in range(earlier + 1, len(rules)):
            if matches[earlier] and matches[later] and matches[earlier] >= matches[later]:
                raise ArmError(
                    f"rule {later + 1} ({rules[later]['regex']!r}) is fully shadowed by "
                    f"earlier rule {earlier + 1} ({rules[earlier]['regex']!r}); "
                    "put layer/family-specific rules before catch-all rules"
                )
    return warnings


def normalize_arm(raw: dict[str, Any], *, allow_ppl_override: bool = False) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ArmError("arm JSON must be an object")

    arm_id = raw.get("id")
    if not isinstance(arm_id, str) or not ARM_ID_RE.match(arm_id):
        raise ArmError("id: expected [A-Za-z0-9][A-Za-z0-9._-]*")

    if "base_model" not in raw:
        raise ArmError("base_model: missing")
    base_model = resolve_path(raw["base_model"], "base_model")

    default_type = normalize_type(raw.get("default_type", "Q6_K"), "default_type")

    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ArmError("rules: expected a non-empty list")
    rules: list[dict[str, str]] = []
    for index, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            raise ArmError(f"rules[{index}]: expected an object")
        regex = item.get("regex")
        if not isinstance(regex, str) or not regex.strip():
            raise ArmError(f"rules[{index}].regex: expected a non-empty string")
        if "\n" in regex or "\r" in regex:
            raise ArmError(f"rules[{index}].regex: newlines are not allowed")
        try:
            re.compile(regex.lower())
        except re.error as exc:
            raise ArmError(f"rules[{index}].regex: {exc}") from exc
        rules.append(
            {
                "regex": regex.strip(),
                "type": normalize_type(item.get("type"), f"rules[{index}].type"),
            }
        )
    warnings = validate_rule_order(rules)

    raw_ppl = raw.get("ppl")
    if not isinstance(raw_ppl, dict):
        raise ArmError("ppl: expected an object")
    if "corpus" not in raw_ppl:
        raise ArmError("ppl.corpus: missing")
    corpus = resolve_path(raw_ppl["corpus"], "ppl.corpus")
    ppl: dict[str, Any] = {"corpus": corpus}
    for key, expected in PPL_FIXED.items():
        if key not in raw_ppl:
            value = expected
        else:
            try:
                value = int(raw_ppl[key])
            except (TypeError, ValueError) as exc:
                raise ArmError(f"ppl.{key}: expected an integer") from exc
        if value <= 0:
            raise ArmError(f"ppl.{key}: expected a positive integer")
        if value != expected and not allow_ppl_override:
            raise ArmError(
                f"ppl.{key}={value} differs from the fixed arm recipe {expected}; "
                "PPL numbers would not be comparable (pass --allow-ppl-override "
                "only for a deliberate exception)"
            )
        ppl[key] = value

    raw_pack = raw.get("pack") or {}
    if not isinstance(raw_pack, dict):
        raise ArmError("pack: expected an object")
    pack_enabled = bool(raw_pack.get("enabled", False))
    build_dir = resolve_path(
        raw_pack.get("build_dir") or HOST_PACK_BUILD, "pack.build_dir"
    )
    lib_dir = (
        resolve_path(raw_pack["lib_dir"], "pack.lib_dir")
        if raw_pack.get("lib_dir")
        else build_dir / "bin"
    )
    expert_order = None
    if raw_pack.get("expert_order"):
        expert_order = resolve_path(raw_pack["expert_order"], "pack.expert_order")
    pack = {
        "enabled": pack_enabled,
        "build_dir": build_dir,
        "lib_dir": lib_dir,
        "expert_order": expert_order,
    }

    try:
        quantize_threads = int(raw.get("quantize_threads", QUANTIZE_THREADS))
    except (TypeError, ValueError) as exc:
        raise ArmError("quantize_threads: expected an integer") from exc
    if quantize_threads <= 0:
        raise ArmError("quantize_threads: expected a positive integer")

    model_name = raw.get("model_name") or f"{arm_id}.gguf"
    if not isinstance(model_name, str) or "/" in model_name or ".." in Path(model_name).parts:
        raise ArmError("model_name: expected a plain file name")

    imatrix = None
    raw_imatrix = raw.get("imatrix")
    if raw_imatrix is not None:
        imatrix_path = resolve_path(raw_imatrix, "imatrix")
        if imatrix_path.is_dir():
            raise ArmError(f"imatrix: expected a file, got a directory: {imatrix_path}")
        if not imatrix_path.is_file():
            raise ArmError(f"imatrix: file not found: {imatrix_path}")
        imatrix = imatrix_metadata(imatrix_path)

    # T-0088: provenance is opt-out (default true) and only affects the tiled
    # pack path.  The quantizer source is fingerprinted by q1_provenance.py; a
    # missing file is deliberately *not* validated here so ``run`` records
    # ``provenance.status=stamp_failed`` instead of silently skipping stamp.
    raw_provenance = raw.get("provenance") or {}
    if not isinstance(raw_provenance, dict):
        raise ArmError("provenance: expected an object")
    provenance_enabled = raw_provenance.get("enabled", True)
    if not isinstance(provenance_enabled, bool):
        raise ArmError("provenance.enabled: expected a boolean")
    raw_quantizer = raw_provenance.get("quantizer_src")
    quantizer_src = (
        resolve_path(raw_quantizer, "provenance.quantizer_src")
        if raw_quantizer is not None
        else DEFAULT_QUANTIZER_SRC
    )
    weighted_override = raw_provenance.get("weighted_q1_0")
    if weighted_override not in (None, "auto", "true", "false"):
        raise ArmError("provenance.weighted_q1_0: expected auto|true|false")
    provenance = {
        "enabled": provenance_enabled,
        "quantizer_src": quantizer_src,
        "weighted_q1_0_override": weighted_override,
    }

    return {
        "id": arm_id,
        "base_model": base_model,
        "default_type": default_type,
        "rules": rules,
        "ppl": ppl,
        "pack": pack,
        "quantize_threads": quantize_threads,
        "model_name": model_name,
        "imatrix": imatrix,
        "provenance": provenance,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


@dataclass
class Step:
    name: str
    kind: str  # rules | quantize | ppl | extract | pack | provenance
    shell: str
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    log: Optional[Path] = None
    produces: Optional[Path] = None


@dataclass
class Plan:
    cfg: dict[str, Any]
    arm_path: Path
    out_dir: Path
    repo_root: Path
    rules_path: Path
    commands_path: Path
    plan_path: Path
    result_path: Path
    model_path: Path
    ppl_log: Path
    raw_dir: Path
    tiled_dir: Path
    steps: list[Step]
    gpu: Optional[str]
    warnings: list[str]
    budget: Optional[dict[str, Any]] = None

    @property
    def arm_id(self) -> str:
        return self.cfg["id"]

    @property
    def rules_text(self) -> str:
        return "".join(f"{rule['regex']}={rule['type']}\n" for rule in self.cfg["rules"])

    @property
    def imatrix_sha_path(self) -> Path:
        return self.out_dir / IMATRIX_SIDECAR

    @property
    def provenance_path(self) -> Path:
        return self.tiled_dir / "provenance.json"


def _gpu_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not GPU_RE.match(text):
        raise ArmError(f"--gpu: expected N or N,M (comma-separated device ids), got {value!r}")
    return text


def _rule_targets_expert(regex: str) -> bool:
    """True when *regex* (llama-quantize syntax) matches an expert probe."""
    compiled = re.compile(regex.lower())
    return any(
        compiled.search(name)
        for name in PROBE_TENSORS
        if FAMILY_RE.search(name)
    )


def provenance_weighted_flag(cfg: dict[str, Any]) -> str:
    """Choose ``--weighted-q1-0`` for the stamp step.

    * explicit arm JSON override wins;
    * no imatrix -> ``false`` (an unweighted arm; q1_provenance rejects true);
    * imatrix + an expert-family Q1_0 rule -> ``true``;
    * otherwise ``auto`` so the sidecar records ``unknown`` rather than guess.
    """
    override = cfg["provenance"].get("weighted_q1_0_override")
    if override:
        return str(override)
    if not cfg.get("imatrix"):
        return "false"
    expert_q1 = any(
        rule["type"] == "Q1_0" and _rule_targets_expert(rule["regex"])
        for rule in cfg["rules"]
    )
    return "true" if expert_q1 else "auto"


def build_plan(
    arm_path: Path,
    out_dir: Path,
    *,
    gpu: Optional[str] = None,
    allow_ppl_override: bool = False,
) -> Plan:
    arm_path = Path(arm_path).expanduser()
    if ".." in arm_path.parts:
        raise ArmError(f"--arm: '..' components are not allowed: {arm_path}")
    arm_path = arm_path.resolve()
    try:
        raw = json.loads(arm_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArmError(f"cannot read arm JSON {arm_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArmError(f"invalid JSON in {arm_path}: {exc}") from exc

    cfg = normalize_arm(raw, allow_ppl_override=allow_ppl_override)
    out_dir = resolve_out_dir(out_dir)
    gpu = _gpu_value(gpu)

    rules_path = out_dir / "rules.txt"
    commands_path = out_dir / "commands.sh"
    plan_path = out_dir / "plan.json"
    result_path = out_dir / "arm-result.json"
    model_path = out_dir / "models" / cfg["model_name"]
    ppl_log = out_dir / "logs" / "ppl.log"
    raw_dir = out_dir / "experts" / "raw"
    tiled_dir = out_dir / "experts" / "tiled"

    quant_bin = str(Path(QUANTIZE_BIN).expanduser())
    ppl_bin = str(Path(PERPLEXITY_BIN).expanduser())
    lib_dir = cfg["pack"]["lib_dir"]

    ppl_prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu else ""
    ppl_flags = (
        f'-c {cfg["ppl"]["ctx"]} -b {cfg["ppl"]["batch"]} '
        f'-ub {cfg["ppl"]["ubatch"]} -ngl 99 -fit off -t {cfg["ppl"]["threads"]} '
        f'--chunks {cfg["ppl"]["chunks"]}'
    )
    pack_order = cfg["pack"]["expert_order"]
    pack_extra = f' --expert-order "$ORDER"' if pack_order else ""

    imatrix = cfg.get("imatrix")
    # Keep the no-imatrix command byte-for-byte identical to the old output:
    # both fragments are empty unless the arm opts in.
    imatrix_flag = '--imatrix "$IMATRIX" ' if imatrix else ""
    imatrix_argv = ["--imatrix", str(imatrix["path"])] if imatrix else []

    steps: list[Step] = []

    rules_block = (
        'cat > "$OUT/rules.txt" <<\'ARM_RULES\'\n'
        + plan_rules_body(cfg["rules"])
        + "ARM_RULES\n"
    )
    steps.append(
        Step(
            name="rules",
            kind="rules",
            shell=rules_block,
            produces=rules_path,
        )
    )

    steps.append(
        Step(
            name="quantize",
            kind="quantize",
            shell=(
                f'"$Q" {imatrix_flag}--tensor-type-file "$OUT/rules.txt" "$SRC" '
                f'"$MODEL" {cfg["default_type"]} {cfg["quantize_threads"]} '
                f'> "$OUT/logs/quantize.log" 2>&1'
            ),
            argv=[
                quant_bin,
                *imatrix_argv,
                "--tensor-type-file",
                str(rules_path),
                str(cfg["base_model"]),
                str(model_path),
                cfg["default_type"],
                str(cfg["quantize_threads"]),
            ],
            log=out_dir / "logs" / "quantize.log",
            produces=model_path,
        )
    )

    steps.append(
        Step(
            name="ppl",
            kind="ppl",
            shell=(
                f'{ppl_prefix}"$P" -m "$MODEL" -f "$CORPUS" '
                f'{ppl_flags} > "$OUT/logs/ppl.log" 2>&1'
            ),
            argv=[
                ppl_bin,
                "-m",
                str(model_path),
                "-f",
                str(cfg["ppl"]["corpus"]),
                "-c",
                str(cfg["ppl"]["ctx"]),
                "-b",
                str(cfg["ppl"]["batch"]),
                "-ub",
                str(cfg["ppl"]["ubatch"]),
                "-ngl",
                "99",
                "-fit",
                "off",
                "-t",
                str(cfg["ppl"]["threads"]),
                "--chunks",
                str(cfg["ppl"]["chunks"]),
            ],
            env={"CUDA_VISIBLE_DEVICES": gpu} if gpu else {},
            log=ppl_log,
            produces=ppl_log,
        )
    )

    if cfg["pack"]["enabled"]:
        steps.append(
            Step(
                name="extract",
                kind="extract",
                shell=(
                    'python3 tools/moe/experts.py extract '
                    '--model "$MODEL" --out-dir "$OUT/experts/raw" '
                    '> "$OUT/logs/extract.log" 2>&1'
                ),
                argv=[
                    "python3",
                    "tools/moe/experts.py",
                    "extract",
                    "--model",
                    str(model_path),
                    "--out-dir",
                    str(raw_dir),
                ],
                log=out_dir / "logs" / "extract.log",
                produces=raw_dir / "manifest.json",
            )
        )
        steps.append(
            Step(
                name="pack",
                kind="pack",
                shell=(
                    'LD_LIBRARY_PATH="$LIB" python3 tools/moe/pack_hexagon.py '
                    '--experts "$OUT/experts/raw" '
                    '--out-dir "$OUT/experts/tiled" --lib-dir "$LIB"'
                    f"{pack_extra} > \"$OUT/logs/pack.log\" 2>&1"
                ),
                argv=[
                    "python3",
                    "tools/moe/pack_hexagon.py",
                    "--experts",
                    str(raw_dir),
                    "--out-dir",
                    str(tiled_dir),
                    "--lib-dir",
                    str(lib_dir),
                ]
                + (["--expert-order", str(pack_order)] if pack_order else []),
                env={"LD_LIBRARY_PATH": str(lib_dir)},
                log=out_dir / "logs" / "pack.log",
                produces=tiled_dir / "manifest.json",
            )
        )

    # T-0088: provenance stamp/verify run immediately after a successful pack.
    # The sidecar never touches manifest.json or layer_*.bin (enforced by
    # q1_provenance.py and asserted by the tests).
    if cfg["pack"]["enabled"] and cfg["provenance"]["enabled"]:
        quantizer_src = cfg["provenance"]["quantizer_src"]
        weighted_flag = provenance_weighted_flag(cfg)
        provenance_out = tiled_dir / "provenance.json"
        imatrix_shell = '--imatrix "$IMATRIX" ' if imatrix else ""
        imatrix_argv = ["--imatrix", str(imatrix["path"])] if imatrix else []

        steps.append(
            Step(
                name="stamp",
                kind="provenance",
                shell=(
                    'python3 tools/moe/q1_provenance.py stamp '
                    '--tiled "$OUT/experts/tiled" --arm "$MODEL" '
                    '--rules "$OUT/rules.txt" --quantizer-src "$QUANTIZER_SRC" '
                    f'{imatrix_shell}--weighted-q1-0 {weighted_flag} '
                    '--out "$OUT/experts/tiled/provenance.json" '
                    '> "$OUT/logs/provenance-stamp.log" 2>&1'
                ),
                argv=[
                    "python3",
                    "tools/moe/q1_provenance.py",
                    "stamp",
                    "--tiled",
                    str(tiled_dir),
                    "--arm",
                    str(model_path),
                    "--rules",
                    str(rules_path),
                    "--quantizer-src",
                    str(quantizer_src),
                    *imatrix_argv,
                    "--weighted-q1-0",
                    weighted_flag,
                    "--out",
                    str(provenance_out),
                ],
                log=out_dir / "logs" / "provenance-stamp.log",
                produces=provenance_out,
            )
        )

        awk_first = "'{print $1}'"
        verify_shell = (
            f'ARM_SHA=$(sha256sum "$MODEL" | awk {awk_first}) && '
            + (
                f'IMATRIX_SHA=$(sha256sum "$IMATRIX" | awk {awk_first}) && '
                if imatrix
                else ""
            )
            + 'python3 tools/moe/q1_provenance.py verify '
            '--tiled "$OUT/experts/tiled" --require-provenance '
            '--expect-arm-sha "$ARM_SHA" '
            + ('--expect-imatrix-sha "$IMATRIX_SHA" ' if imatrix else "")
            + '> "$OUT/logs/provenance-verify.log" 2>&1'
        )
        steps.append(
            Step(
                name="verify",
                kind="provenance",
                shell=verify_shell,
                # Expected hashes are filled in by run_plan() after quantize
                # (the plan-time model does not exist yet).
                argv=[
                    "python3",
                    "tools/moe/q1_provenance.py",
                    "verify",
                    "--tiled",
                    str(tiled_dir),
                    "--require-provenance",
                ],
                log=out_dir / "logs" / "provenance-verify.log",
            )
        )

    warnings = list(cfg["warnings"])
    if not cfg["base_model"].is_file():
        warnings.append(f"base model not found (fine for plan/dry-run): {cfg['base_model']}")
    if not cfg["ppl"]["corpus"].is_file():
        warnings.append(f"PPL corpus not found (fine for plan/dry-run): {cfg['ppl']['corpus']}")
    if not os.access(quant_bin, os.X_OK):
        warnings.append(f"llama-quantize not executable (fine for plan/dry-run): {quant_bin}")
    if not os.access(ppl_bin, os.X_OK):
        warnings.append(f"llama-perplexity not executable (fine for plan/dry-run): {ppl_bin}")
    if cfg["pack"]["enabled"]:
        if not lib_dir.is_dir():
            warnings.append(f"pack lib dir not found (fine for plan/dry-run): {lib_dir}")
        if pack_order is not None and not pack_order.is_file():
            warnings.append(f"expert order file not found: {pack_order}")
    if cfg["pack"]["enabled"] and cfg["provenance"]["enabled"]:
        if not quantizer_src.is_file():
            warnings.append(
                "provenance quantizer source not found (stamp will fail): "
                f"{quantizer_src}"
            )

    return Plan(
        cfg=cfg,
        arm_path=arm_path,
        out_dir=out_dir,
        repo_root=REPO_ROOT,
        rules_path=rules_path,
        commands_path=commands_path,
        plan_path=plan_path,
        result_path=result_path,
        model_path=model_path,
        ppl_log=ppl_log,
        raw_dir=raw_dir,
        tiled_dir=tiled_dir,
        steps=steps,
        gpu=gpu,
        warnings=warnings,
    )


def plan_rules_body(rules: Sequence[dict[str, str]]) -> str:
    return "".join(f"{rule['regex']}={rule['type']}\n" for rule in rules)


def commands_text(plan: Plan) -> str:
    lines = [
        "#!/bin/sh",
        f"# Generated by tools/moe/quant_arm.py for arm {plan.arm_id}",
        "# llama-quantize has no `--default` flag; the `--default "
        f"{plan.cfg['default_type']}` role is the positional ftype at the end "
        "of the quantize command.",
        "set -eu",
        "",
        f"cd {q(str(plan.repo_root))}",
        f"Q={QUANTIZE_BIN}",
        f"P={PERPLEXITY_BIN}",
        f"B={q(str(plan.cfg['pack']['build_dir']))}",
        f"LIB={q(str(plan.cfg['pack']['lib_dir']))}",
    ]
    if plan.cfg.get("imatrix"):
        lines.append(f"IMATRIX={q(str(plan.cfg['imatrix']['path']))}")
    if any(step.kind == "provenance" for step in plan.steps):
        lines.append(
            f"QUANTIZER_SRC={q(str(plan.cfg['provenance']['quantizer_src']))}"
        )
    lines.extend(
        [
            f"OUT={q(str(plan.out_dir))}",
            f"SRC={q(str(plan.cfg['base_model']))}",
            f"MODEL={q(str(plan.model_path))}",
            f"CORPUS={q(str(plan.cfg['ppl']['corpus']))}",
        ]
    )
    if plan.cfg["pack"]["expert_order"]:
        lines.append(f"ORDER={q(str(plan.cfg['pack']['expert_order']))}")
    lines.extend(
        [
            "",
            'mkdir -p "$OUT/models" "$OUT/logs" "$OUT/experts"',
        ]
    )
    total = len(plan.steps)
    for index, step in enumerate(plan.steps, start=1):
        lines.append("")
        lines.append(f"# [{index}/{total}] {step.name}")
        lines.append(step.shell.rstrip("\n"))
    lines.append("")
    return "\n".join(lines)


def empty_budget() -> dict[str, Any]:
    """The ``budget`` object written when no budget/detail was requested.

    Keeping it present (instead of omitting the key) makes ``plan.json``
    self-describing while leaving the old plan output fields untouched.
    """
    return {
        "schema_version": 1,
        "enabled": False,
        "max_expert_gib": None,
        "budget_gib": None,
        "limit_bytes": None,
        "scope": None,
        "compared_bytes": None,
        "expert_bytes_raw": None,
        "expert_bytes_tiled_est": None,
        "over_budget": False,
        "allow_over_budget": False,
        "expert_regex": DEFAULT_EXPERT_REGEX,
        "layers": 0,
        "tensor_count": 0,
        "by_layer": [],
        "by_type": {},
        "estimate": {
            "formula": TILED_FORMULA,
            "tiled_overhead_factor": TILED_OVERHEAD_FACTOR,
            "tile_header_bytes_per_layer": TILED_HEADER_BYTES_PER_LAYER,
            "exact": False,
            "expert_bytes_raw": None,
            "expert_bytes_tiled_est": None,
        },
    }


def _budget_regex(value: Optional[str]) -> str:
    pattern = value or DEFAULT_EXPERT_REGEX
    if not isinstance(pattern, str) or not pattern.strip():
        raise ArmError("--expert-regex: expected a non-empty regular expression")
    if "\n" in pattern or "\r" in pattern:
        raise ArmError("--expert-regex: newlines are not allowed")
    try:
        re.compile(pattern.lower())
    except re.error as exc:
        raise ArmError(f"--expert-regex: {exc}") from exc
    return pattern.strip()


def _expert_family(name: str) -> str:
    match = FAMILY_RE.search(name)
    if match:
        return match.group(1)
    match = re.search(r"\.(gate|up|down)\.", name)
    if match:
        return match.group(1)
    return "expert"


def build_budget(
    plan: Plan,
    *,
    max_expert_gib: Optional[float] = None,
    allow_over_budget: bool = False,
    scope: str = "tiled",
    expert_regex: Optional[str] = None,
    budget_json: Optional[Path] = None,
) -> dict[str, Any]:
    """Predict expert bytes from the GGUF header and evaluate the budget.

    This runs only after ``build_plan()`` has validated the arm, rules, types
    and paths.  No tensor data is read.  Rule resolution and the
    ``tensor_type_fallback`` chain reuse :mod:`quant_rule_lint` so ``plan`` and
    ``quant_rule_lint`` cannot drift apart.
    """
    if scope not in BUDGET_SCOPES:
        raise ArmError(f"--budget-scope: expected one of {', '.join(BUDGET_SCOPES)}")
    if max_expert_gib is not None:
        if not math.isfinite(float(max_expert_gib)) or float(max_expert_gib) < 0:
            raise ArmError("--max-expert-gib: expected a finite non-negative number")

    pattern = _budget_regex(expert_regex)
    enabled = (
        max_expert_gib is not None
        or budget_json is not None
        or expert_regex is not None
    )
    if not enabled:
        return empty_budget()

    try:
        header = qrl.parse_gguf_header(plan.cfg["base_model"])
    except qrl.LintError as exc:
        raise ArmError(f"cannot estimate expert bytes: {exc}") from exc

    rules = qrl.parse_rule_text(
        plan_rules_body(plan.cfg["rules"]), source="<arm rules>"
    )
    bad = [rule for rule in rules if not rule.valid]
    if bad:  # normalize_arm() already rejects these; keep the guard explicit.
        raise ArmError(f"cannot estimate expert bytes: invalid rule: {bad[0].error}")
    default_type = qrl.GGML_TYPES_BY_NAME.get(plan.cfg["default_type"])
    if default_type is None:
        raise ArmError(
            f"cannot estimate expert bytes: unknown default_type {plan.cfg['default_type']!r}"
        )

    report = qrl.lint(header, rules, default_type, strict=False)
    expert_re = re.compile(pattern.lower())

    entries: list[dict[str, Any]] = []
    for tensor_plan in report.plans:
        tensor = tensor_plan.tensor
        if not expert_re.search(tensor.name.lower()):
            continue
        if tensor_plan.bytes is None:
            raise ArmError(
                f"cannot estimate expert bytes for {tensor.name}: "
                "row size not divisible by the final type block"
            )
        layer_match = LAYER_RE.search(tensor.name)
        layer = int(layer_match.group(1)) if layer_match else -1
        entries.append(
            {
                "layer": layer,
                "family": _expert_family(tensor.name),
                "tensor": tensor.name,
                "type": tensor_plan.final_type,
                "fallback_from": tensor_plan.fallback_from,
                "bytes_raw": int(tensor_plan.bytes),
            }
        )

    if not entries:
        raise ArmError(
            f"--expert-regex {pattern!r} matches no tensor in {plan.cfg['base_model']}; "
            "check the regex against the GGUF tensor names"
        )

    by_layer: dict[int, dict[str, Any]] = {}
    by_type: dict[str, dict[str, Any]] = {}
    for entry in entries:
        layer = by_layer.setdefault(
            entry["layer"], {"layer": entry["layer"], "bytes_raw": 0, "families": {}}
        )
        layer["bytes_raw"] += entry["bytes_raw"]
        layer["families"][entry["family"]] = {
            "tensor": entry["tensor"],
            "type": entry["type"],
            "fallback_from": entry["fallback_from"],
            "bytes_raw": entry["bytes_raw"],
        }
        type_entry = by_type.setdefault(
            entry["type"], {"tensors": 0, "bytes_raw": 0}
        )
        type_entry["tensors"] += 1
        type_entry["bytes_raw"] += entry["bytes_raw"]

    raw_total = sum(entry["bytes_raw"] for entry in entries)
    layers = len(by_layer)
    tiled_total = math.ceil(
        raw_total * TILED_OVERHEAD_FACTOR + TILED_HEADER_BYTES_PER_LAYER * layers
    )
    limit_bytes = (
        int(float(max_expert_gib) * GIB) if max_expert_gib is not None else None
    )
    compared = tiled_total if scope == "tiled" else raw_total
    over_budget = bool(limit_bytes is not None and compared > limit_bytes)

    return {
        "schema_version": 1,
        "enabled": True,
        "max_expert_gib": float(max_expert_gib) if max_expert_gib is not None else None,
        "budget_gib": float(max_expert_gib) if max_expert_gib is not None else None,
        "limit_bytes": limit_bytes,
        "scope": scope,
        "compared_bytes": compared,
        "expert_bytes_raw": raw_total,
        "expert_bytes_tiled_est": tiled_total,
        "over_budget": over_budget,
        "allow_over_budget": bool(allow_over_budget),
        "expert_regex": pattern,
        "layers": layers,
        "tensor_count": len(entries),
        "by_layer": [by_layer[key] for key in sorted(by_layer)],
        "by_type": {key: by_type[key] for key in sorted(by_type)},
        "estimate": {
            "formula": TILED_FORMULA,
            "tiled_overhead_factor": TILED_OVERHEAD_FACTOR,
            "tile_header_bytes_per_layer": TILED_HEADER_BYTES_PER_LAYER,
            "exact": False,
            "expert_bytes_raw": raw_total,
            "expert_bytes_tiled_est": tiled_total,
            "layers": layers,
        },
    }


def budget_detail(plan: Plan, budget: dict[str, Any]) -> dict[str, Any]:
    """Payload for ``--budget-json`` (flattened for easy jq/read use)."""
    return {
        "schema_version": 1,
        "arm_id": plan.arm_id,
        "base_model": str(plan.cfg["base_model"]),
        "generated_at": utc_now(),
        **budget,
    }


def _budget_entries(budget: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for layer in budget.get("by_layer") or []:
        for family, info in (layer.get("families") or {}).items():
            entries.append(
                {
                    "layer": layer.get("layer"),
                    "family": family,
                    "tensor": info.get("tensor"),
                    "type": info.get("type"),
                    "bytes_raw": info.get("bytes_raw", 0),
                }
            )
    entries.sort(key=lambda item: int(item.get("bytes_raw") or 0), reverse=True)
    return entries


def _print_budget_summary(budget: dict[str, Any]) -> None:
    if not budget.get("enabled"):
        return
    raw = int(budget["expert_bytes_raw"])
    tiled = int(budget["expert_bytes_tiled_est"])
    scope = budget["scope"]
    compared = int(budget["compared_bytes"])
    line = (
        f"experts: raw {raw / GIB:.3f} GiB, tiled-est {tiled / GIB:.3f} GiB "
        f"(scope={scope}, compared {compared / GIB:.3f} GiB, "
        f"{budget['layers']} layers / {budget['tensor_count']} tensors)"
    )
    print(line)
    limit = budget.get("limit_bytes")
    if limit is None:
        print("budget: not requested (no --max-expert-gib)")
    elif budget["over_budget"]:
        suffix = ", --allow-over-budget)" if budget["allow_over_budget"] else ")"
        print(
            f"budget: OVER by {(compared - int(limit)) / GIB:.3f} GiB "
            f"(limit {float(budget['max_expert_gib']):.3f} GiB{suffix}",
            file=sys.stderr,
        )
    else:
        print(
            f"budget: OK, {(int(limit) - compared) / GIB:.3f} GiB headroom "
            f"(limit {float(budget['max_expert_gib']):.3f} GiB)"
        )


def _print_over_budget(budget: dict[str, Any]) -> None:
    entries = _budget_entries(budget)
    prefix = "warning" if budget.get("allow_over_budget") else "error"
    print(
        f"{prefix}: expert package exceeds --max-expert-gib "
        f"{float(budget['max_expert_gib']):.3f} GiB "
        f"(scope={budget['scope']}, estimate {int(budget['compared_bytes']) / GIB:.3f} GiB)",
        file=sys.stderr,
    )
    print("largest expert tensors (Top-10):", file=sys.stderr)
    for entry in entries[:10]:
        print(
            f"  blk.{entry['layer']}.{entry['family']}  {entry['type']}  "
            f"{int(entry['bytes_raw']) / GIB:.4f} GiB  ({entry['tensor']})",
            file=sys.stderr,
        )
    if not budget.get("allow_over_budget"):
        print(
            "pass --allow-over-budget to write the plan anyway (exit 0)",
            file=sys.stderr,
        )


def plan_manifest(plan: Plan) -> dict[str, Any]:
    budget = plan.budget or empty_budget()
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "arm_id": plan.arm_id,
        "arm_json": str(plan.arm_path),
        "git_head": git_value("rev-parse", "HEAD"),
        "repo_root": str(plan.repo_root),
        "out_dir": str(plan.out_dir),
        "base_model": str(plan.cfg["base_model"]),
        "imatrix": imatrix_json(plan.cfg.get("imatrix")),
        "default_type": plan.cfg["default_type"],
        "quantize_threads": plan.cfg["quantize_threads"],
        "rules": plan.cfg["rules"],
        "ppl": {**plan.cfg["ppl"], "corpus": str(plan.cfg["ppl"]["corpus"])},
        "pack": {
            "enabled": plan.cfg["pack"]["enabled"],
            "build_dir": str(plan.cfg["pack"]["build_dir"]),
            "lib_dir": str(plan.cfg["pack"]["lib_dir"]),
            "expert_order": (
                str(plan.cfg["pack"]["expert_order"])
                if plan.cfg["pack"]["expert_order"]
                else None
            ),
        },
        "provenance": {
            "enabled": plan.cfg["provenance"]["enabled"],
            "active": (
                plan.cfg["pack"]["enabled"]
                and plan.cfg["provenance"]["enabled"]
            ),
            "quantizer_src": str(plan.cfg["provenance"]["quantizer_src"]),
            "weighted_q1_0": provenance_weighted_flag(plan.cfg),
            "sidecar": str(plan.provenance_path),
        },
        "paths": {
            "rules": str(plan.rules_path),
            "commands": str(plan.commands_path),
            "model": str(plan.model_path),
            "ppl_log": str(plan.ppl_log),
            "experts_raw": str(plan.raw_dir),
            "experts_tiled": str(plan.tiled_dir),
            "result": str(plan.result_path),
        },
        "commands": [
            {"name": step.name, "kind": step.kind, "shell": step.shell, "argv": step.argv}
            for step in plan.steps
        ],
        "warnings": plan.warnings,
        "budget": budget,
        "estimate": budget.get("estimate", empty_budget()["estimate"]),
    }


def write_plan(plan: Plan) -> None:
    atomic_write_text(plan.rules_path, plan.rules_text, 0o644)
    atomic_write_text(plan.commands_path, commands_text(plan), 0o644)
    atomic_write_json(plan.plan_path, plan_manifest(plan), 0o644)


# ---------------------------------------------------------------------------
# Run / resume
# ---------------------------------------------------------------------------


def imatrix_changed(plan: Plan) -> bool:
    """True when the arm's imatrix differs from the recorded sidecar.

    A missing sidecar also counts as changed: a model produced before imatrix
    tracking cannot be proven to have consumed the current file.
    """
    info = plan.cfg.get("imatrix")
    if not info:
        return False
    try:
        current = sha256_file(info["path"])
    except OSError:
        return True
    try:
        recorded = plan.imatrix_sha_path.read_text(encoding="utf-8").strip()
    except OSError:
        return True
    return not recorded or recorded != current


def step_done(step: Step, plan: Plan) -> bool:
    """Return True when *step*'s output already exists and is usable."""
    if step.kind == "rules":
        return plan.rules_path.is_file() and plan.rules_path.stat().st_size > 0
    if step.kind == "quantize":
        if not (plan.model_path.is_file() and plan.model_path.stat().st_size > 0):
            return False
        # The imatrix is an input of the quantized model; if it changed the
        # existing GGUF is stale even though it is still on disk.
        return not imatrix_changed(plan)
    if step.kind == "ppl":
        if not plan.ppl_log.is_file():
            return False
        try:
            return parse_ppl(plan.ppl_log.read_text(encoding="utf-8", errors="replace")) is not None
        except OSError:
            return False
    if step.kind == "extract":
        return (plan.raw_dir / "manifest.json").is_file()
    if step.kind == "pack":
        return (plan.tiled_dir / "manifest.json").is_file()
    if step.kind == "provenance":
        # Never skip provenance just because pack output exists; run_plan has
        # an explicit sidecar-aware resume path instead.
        return False
    return False


def _need_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ArmError(f"{label} not found: {path}")


def _need_executable(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ArmError(f"{label} is not an executable file: {path}")


def preflight(step: Step, plan: Plan) -> None:
    if step.kind == "quantize":
        _need_file(plan.cfg["base_model"], "base model")
        if plan.cfg.get("imatrix"):
            _need_file(plan.cfg["imatrix"]["path"], "imatrix")
        _need_executable(Path(QUANTIZE_BIN).expanduser(), "llama-quantize")
    elif step.kind == "ppl":
        _need_file(plan.model_path, "quantized model")
        _need_file(plan.cfg["ppl"]["corpus"], "PPL corpus")
        _need_executable(Path(PERPLEXITY_BIN).expanduser(), "llama-perplexity")
    elif step.kind in ("extract", "pack"):
        _need_file(plan.model_path, "quantized model")
    if step.kind == "pack":
        lib_dir = plan.cfg["pack"]["lib_dir"]
        if not (lib_dir / "libggml-base.so").is_file():
            raise ArmError(f"pack lib dir lacks libggml-base.so: {lib_dir}")
        if not (lib_dir / "libggml-hexagon.so").is_file():
            raise ArmError(
                f"pack lib dir lacks libggml-hexagon.so: {lib_dir} "
                "(build it with tools/moe/build_host_pack.sh)"
            )


def execute_step(step: Step, plan: Plan) -> tuple[int, float]:
    assert step.log is not None
    step.log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(step.env)
    started = time.monotonic()
    with step.log.open("wb") as handle:
        try:
            completed = subprocess.run(
                step.argv,
                cwd=str(plan.repo_root),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            rc = completed.returncode
        except OSError as exc:
            handle.write(f"failed to exec: {exc}\n".encode())
            rc = 127
    return rc, time.monotonic() - started


Runner = Callable[[Step, Plan], Any]


def _log_tail(path: Optional[Path], limit: int = 1200) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _command_record(
    step: Step,
    *,
    rc: Optional[int],
    duration: float,
    skipped: bool,
    not_run: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": step.name,
        "kind": step.kind,
        "argv": step.argv,
        "shell": step.shell,
        "rc": rc,
        "duration_s": round(duration, 3),
        "skipped": skipped,
    }
    if step.log is not None:
        record["log"] = str(step.log)
    if step.produces is not None:
        record["produces"] = str(step.produces)
    if not_run:
        record["not_run"] = True
    return record


def _provenance_active(plan: Plan) -> bool:
    return bool(
        plan.cfg["pack"]["enabled"] and plan.cfg["provenance"]["enabled"]
    )


def _base_provenance(plan: Plan) -> dict[str, Any]:
    return {
        "enabled": bool(plan.cfg["provenance"]["enabled"]),
        "sidecar": str(plan.provenance_path),
        "sidecar_sha256": None,
        "stamp_rc": None,
        "verify_rc": None,
        "verify_ok": False,
        "weighted_q1_0": None,
        "imatrix_sha256": None,
        "arm_sha256": None,
        "rules_sha256": None,
        "status": "pending" if _provenance_active(plan) else "skipped",
    }


def _base_result(plan: Plan, *, keep_models: bool) -> dict[str, Any]:
    base_bytes = _stat_or_none(plan.cfg["base_model"])
    return {
        "schema_version": 1,
        "arm_id": plan.arm_id,
        "generated_at": utc_now(),
        "status": "running",
        "git_head": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "arm_json": str(plan.arm_path),
        "base_model": {
            "path": str(plan.cfg["base_model"]),
            "bytes": base_bytes,
            "sha256": None,
        },
        "imatrix": imatrix_json(plan.cfg.get("imatrix")),
        "default_type": plan.cfg["default_type"],
        "quantize_threads": plan.cfg["quantize_threads"],
        "rules": plan.cfg["rules"],
        "ppl": {
            "corpus": str(plan.cfg["ppl"]["corpus"]),
            "chunks": plan.cfg["ppl"]["chunks"],
            "ctx": plan.cfg["ppl"]["ctx"],
            "batch": plan.cfg["ppl"]["batch"],
            "ubatch": plan.cfg["ppl"]["ubatch"],
            "threads": plan.cfg["ppl"]["threads"],
            "estimate": None,
            "log": str(plan.ppl_log),
        },
        "model": {
            "path": str(plan.model_path),
            "bytes": _stat_or_none(plan.model_path),
            "sha256": None,
            "deleted": False,
        },
        "experts": None,
        "provenance": _base_provenance(plan),
        "commands": [],
        "artifacts": [],
        "keep_models": keep_models,
        "gpu": plan.gpu,
        "warnings": list(plan.warnings),
        "error": None,
    }


def _expert_stats(manifest_path: Path) -> Optional[dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    model = manifest.get("model") or {}
    top_k = int(model.get("expert_used_count") or 0)
    per_layer: dict[int, list[int]] = {}
    for entry in entries:
        try:
            layer = int(entry["layer"])
            size = int(entry["bytes_per_expert"])
        except (KeyError, TypeError, ValueError):
            continue
        per_layer.setdefault(layer, []).append(size)
    if not per_layer:
        return None
    bytes_per_step = None
    if top_k > 0:
        bytes_per_step = sum(top_k * sum(sizes) for sizes in per_layer.values())
    return {
        "manifest": str(manifest_path),
        "backend_layout": manifest.get("backend_layout"),
        "expert_count": int(model.get("expert_count") or 0),
        "top_k": top_k,
        "layers": len(per_layer),
        "entries": len(entries),
        "bytes_per_step": bytes_per_step,
        "mib_per_step": (
            round(bytes_per_step / (1 << 20), 3) if bytes_per_step is not None else None
        ),
    }


def _write_result(plan: Plan, result: dict[str, Any]) -> None:
    atomic_write_json(plan.result_path, result)


def _fail(plan: Plan, result: dict[str, Any], message: str) -> int:
    result["status"] = "failure"
    result["error"] = message
    provenance = result.get("provenance")
    if isinstance(provenance, dict) and provenance.get("status") == "pending":
        # The run stopped before stamp/verify (e.g. quantize/ppl/pack failed).
        provenance["status"] = "skipped"
    result["generated_at"] = utc_now()
    result["artifacts"] = _artifacts(plan)
    _write_result(plan, result)
    print(f"error: {message}", file=sys.stderr)
    print(f"wrote {plan.result_path}", file=sys.stderr)
    return 1


def _artifacts(plan: Plan) -> list[str]:
    candidates = [
        plan.rules_path,
        plan.commands_path,
        plan.plan_path,
        plan.result_path,
        plan.ppl_log,
        plan.out_dir / "logs" / "quantize.log",
        plan.out_dir / "logs" / "extract.log",
        plan.out_dir / "logs" / "pack.log",
        plan.out_dir / "logs" / "provenance-stamp.log",
        plan.out_dir / "logs" / "provenance-verify.log",
        plan.raw_dir / "manifest.json",
        plan.tiled_dir / "manifest.json",
        plan.provenance_path,
    ]
    return [str(path) for path in candidates if path.exists()]


def _load_sidecar(plan: Plan) -> Optional[dict[str, Any]]:
    try:
        record = json.loads(plan.provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _refresh_provenance(plan: Plan, result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Copy the verified sidecar's hashes/flags into the ledger."""
    provenance = result.setdefault("provenance", _base_provenance(plan))
    provenance["sidecar"] = str(plan.provenance_path)
    if not plan.provenance_path.is_file():
        return None
    try:
        provenance["sidecar_sha256"] = sha256_file(plan.provenance_path)
    except OSError:
        return None
    record = _load_sidecar(plan)
    if record is None:
        return None
    for key in (
        "weighted_q1_0",
        "imatrix_sha256",
        "arm_sha256",
        "rules_sha256",
    ):
        if key in record:
            provenance[key] = record[key]
    return record


def _imatrix_sha_for_verify(plan: Plan, result: dict[str, Any]) -> Optional[str]:
    info = plan.cfg.get("imatrix")
    if not info:
        return None
    recorded = result.get("imatrix")
    if isinstance(recorded, dict) and recorded.get("sha256"):
        return str(recorded["sha256"])
    path = Path(info["path"])
    if path.is_file():
        try:
            return sha256_file(path)
        except OSError:
            return None
    return str(info["sha256"]) if info.get("sha256") else None


def _prepare_verify_argv(
    step: Step,
    plan: Plan,
    *,
    arm_sha: Optional[str],
    imatrix_sha: Optional[str],
) -> None:
    """Fill in run-time hashes for the verify argv (plan time has no model)."""
    argv = [
        "python3",
        "tools/moe/q1_provenance.py",
        "verify",
        "--tiled",
        str(plan.tiled_dir),
        "--require-provenance",
    ]
    if arm_sha:
        argv += ["--expect-arm-sha", arm_sha]
    if imatrix_sha:
        argv += ["--expect-imatrix-sha", imatrix_sha]
    step.argv = argv


def _provenance_failure(
    plan: Plan,
    result: dict[str, Any],
    step: Step,
    *,
    rc: int,
    duration: float,
) -> int:
    provenance = result.setdefault("provenance", _base_provenance(plan))
    provenance["status"] = (
        "stamp_failed" if step.name == "stamp" else "verify_failed"
    )
    provenance["stamp_rc" if step.name == "stamp" else "verify_rc"] = rc
    provenance["verify_ok"] = False
    _refresh_provenance(plan, result)
    tail = _log_tail(step.log)
    message = f"provenance {step.name} failed with rc={rc}"
    if tail:
        message += f"; log tail: {tail.strip()[-600:]}"
    return _fail(plan, result, message)


def run_plan(
    plan: Plan,
    *,
    resume: bool = False,
    keep_models: bool = False,
    runner: Optional[Runner] = None,
) -> int:
    runner = runner or execute_step

    imatrix_info = plan.cfg.get("imatrix")
    imatrix_dirty = bool(imatrix_info) and imatrix_changed(plan)
    imatrix_sidecar_exists = plan.imatrix_sha_path.is_file()
    provenance_active = _provenance_active(plan)
    verify_step = next(
        (s for s in plan.steps if s.kind == "provenance" and s.name == "verify"),
        None,
    )

    if resume and plan.result_path.is_file() and not imatrix_dirty:
        try:
            existing = json.loads(plan.result_path.read_text(encoding="utf-8"))
            status = str(existing.get("status", "unknown"))
        except (OSError, json.JSONDecodeError):
            status = "unreadable"
        if status == "failure":
            print(
                f"resume: found {plan.result_path} with status=failure; "
                "retrying, existing outputs will be reused"
            )
        elif provenance_active:
            # T-0088: a completed ledger is not enough when provenance is on:
            # the sidecar may be missing (must re-stamp) or stale (must verify
            # before it can be trusted).  Continue to the step loop instead of
            # short-circuiting.
            print(
                f"resume: {plan.result_path} exists (status={status}); "
                "provenance sidecar will be re-checked"
            )
        else:
            # Any other existing ledger counts as a completed output, per the
            # --resume contract (including an unreadable/unknown ledger).
            print(f"resume: {plan.result_path} exists (status={status}); nothing to do")
            return 0
    if resume and imatrix_dirty:
        print(
            "resume: imatrix sha256 differs from "
            f"{plan.imatrix_sha_path}; quantize and downstream steps will rerun"
        )

    result = _base_result(plan, keep_models=keep_models)
    if imatrix_info:
        # ``imatrix_changed`` describes a *transition* (there was a previous
        # run/sidecar).  A first run with no sidecar is not a change.
        result["imatrix_changed"] = bool(
            imatrix_dirty and (resume or imatrix_sidecar_exists)
        )

    # Rules are plain metadata, not a subprocess.  Always (re)write them unless
    # an identical file is already present under --resume.
    for step in plan.steps:
        if step.kind != "rules":
            continue
        current = plan.rules_path.read_text(encoding="utf-8") if plan.rules_path.is_file() else None
        if resume and current == plan.rules_text:
            result["commands"].append(
                _command_record(step, rc=0, duration=0.0, skipped=True)
            )
            print(f"[rules] skipped (already present): {plan.rules_path}")
        else:
            atomic_write_text(plan.rules_path, plan.rules_text, 0o644)
            result["commands"].append(
                _command_record(step, rc=0, duration=0.0, skipped=False)
            )
            print(f"[rules] wrote {plan.rules_path} (mode 0644)")

    total = len(plan.steps)
    arm_sha: Optional[str] = None
    verify_done = False
    for index, step in enumerate(plan.steps, start=1):
        if step.kind == "rules":
            continue
        if step.name == "verify" and verify_done:
            # Already verified while deciding whether stamp could be skipped.
            continue

        # T-0088 resume: an existing sidecar is only trusted after verify.  On
        # success stamp is skipped; on failure we fall through and re-stamp.
        if (
            resume
            and provenance_active
            and step.name == "stamp"
            and plan.provenance_path.is_file()
            and verify_step is not None
        ):
            existing = _load_sidecar(plan) or {}
            candidate_arm_sha = arm_sha
            if candidate_arm_sha is None and plan.model_path.is_file():
                candidate_arm_sha = sha256_file(plan.model_path)
            if candidate_arm_sha is None:
                value = existing.get("arm_sha256")
                candidate_arm_sha = str(value) if value else None
            _prepare_verify_argv(
                verify_step,
                plan,
                arm_sha=candidate_arm_sha,
                imatrix_sha=_imatrix_sha_for_verify(plan, result),
            )
            print(
                f"[{index}/{total}] stamp: sidecar exists; verifying before skip",
                flush=True,
            )
            try:
                outcome = runner(verify_step, plan)
            except Exception as exc:  # noqa: BLE001
                return _fail(
                    plan,
                    result,
                    f"provenance verify raised {type(exc).__name__}: {exc}",
                )
            if isinstance(outcome, tuple):
                verify_rc, verify_duration = outcome
            else:
                verify_rc, verify_duration = int(outcome), 0.0
            result["commands"].append(
                _command_record(
                    verify_step,
                    rc=int(verify_rc),
                    duration=float(verify_duration),
                    skipped=False,
                )
            )
            if verify_rc == 0:
                result["provenance"]["verify_rc"] = 0
                result["provenance"]["verify_ok"] = True
                result["provenance"]["status"] = "ok"
                _refresh_provenance(plan, result)
                result["commands"].append(
                    _command_record(step, rc=0, duration=0.0, skipped=True)
                )
                print(
                    f"[{index}/{total}] stamp: skipped (existing sidecar verified)"
                )
                verify_done = True
                continue
            print(
                f"[{index}/{total}] stamp: existing sidecar failed verify "
                f"(rc={verify_rc}); re-stamping",
                flush=True,
            )

        if resume and not imatrix_dirty and step_done(step, plan):
            result["commands"].append(
                _command_record(step, rc=0, duration=0.0, skipped=True)
            )
            print(f"[{index}/{total}] {step.name}: skipped (output exists)")
            continue

        if step.kind == "provenance" and step.name == "verify":
            if arm_sha is None and plan.model_path.is_file():
                arm_sha = sha256_file(plan.model_path)
            if arm_sha is None:
                record = _load_sidecar(plan) or {}
                value = record.get("arm_sha256")
                arm_sha = str(value) if value else None
            _prepare_verify_argv(
                step,
                plan,
                arm_sha=arm_sha,
                imatrix_sha=_imatrix_sha_for_verify(plan, result),
            )

        try:
            preflight(step, plan)
        except ArmError as exc:
            if step.kind == "provenance":
                result["provenance"]["status"] = (
                    "stamp_failed" if step.name == "stamp" else "verify_failed"
                )
            return _fail(plan, result, str(exc))

        print(f"[{index}/{total}] {step.name}: running ...", flush=True)
        try:
            outcome = runner(step, plan)
        except Exception as exc:  # noqa: BLE001 - ledger must survive any runner bug
            result["commands"].append(
                _command_record(step, rc=None, duration=0.0, skipped=False)
            )
            if step.kind == "provenance":
                result["provenance"]["status"] = (
                    "stamp_failed" if step.name == "stamp" else "verify_failed"
                )
                _refresh_provenance(plan, result)
                return _fail(
                    plan,
                    result,
                    f"provenance {step.name} raised {type(exc).__name__}: {exc}",
                )
            return _fail(plan, result, f"{step.name} raised {type(exc).__name__}: {exc}")

        if isinstance(outcome, tuple):
            rc, duration = outcome
        else:
            rc, duration = int(outcome), 0.0
        result["commands"].append(
            _command_record(step, rc=int(rc), duration=float(duration), skipped=False)
        )
        if rc != 0:
            if step.kind == "provenance":
                return _provenance_failure(
                    plan, result, step, rc=int(rc), duration=float(duration)
                )
            tail = _log_tail(step.log)
            message = f"{step.name} failed with rc={rc}"
            if tail:
                message += f"; log tail: {tail.strip()[-600:]}"
            return _fail(plan, result, message)

        if step.kind == "quantize":
            if plan.model_path.is_file():
                # T-0088: the arm sha is resolved by run after quantize and is
                # then passed directly to verify (commands.sh resolves it with
                # sha256sum for manual replay).
                arm_sha = sha256_file(plan.model_path)
                result["model"]["bytes"] = plan.model_path.stat().st_size
                result["model"]["sha256"] = arm_sha
            if imatrix_info:
                # Record the input that actually produced this model, not only
                # the plan-time digest (the file may have changed since plan).
                try:
                    fresh = imatrix_metadata(imatrix_info["path"])
                except ArmError as exc:
                    return _fail(plan, result, str(exc))
                result["imatrix"] = imatrix_json(fresh)
                atomic_write_text(
                    plan.imatrix_sha_path, fresh["sha256"] + "\n", 0o644
                )
        if step.kind == "provenance" and step.name == "stamp":
            result["provenance"]["stamp_rc"] = 0
            _refresh_provenance(plan, result)
        if step.kind == "provenance" and step.name == "verify":
            result["provenance"]["verify_rc"] = 0
            result["provenance"]["verify_ok"] = True
            result["provenance"]["status"] = "ok"
            _refresh_provenance(plan, result)
            verify_done = True
        print(f"[{index}/{total}] {step.name}: ok in {duration:.1f}s")

    if provenance_active and result["provenance"]["status"] != "ok":
        return _fail(
            plan,
            result,
            f"provenance verify did not complete (status="
            f"{result['provenance']['status']})",
        )

    # Perplexity estimate.
    estimate = None
    if plan.ppl_log.is_file():
        estimate = parse_ppl(
            plan.ppl_log.read_text(encoding="utf-8", errors="replace")
        )
    if estimate is None:
        return _fail(
            plan,
            result,
            f"could not parse 'Final estimate: PPL = ...' from {plan.ppl_log}",
        )
    result["ppl"]["estimate"] = estimate

    # Model sha256/bytes, measured before the optional cleanup.  A normal run
    # already hashed it right after quantize (needed for verify).
    if plan.model_path.is_file():
        result["model"]["bytes"] = plan.model_path.stat().st_size
        if result["model"]["sha256"] is None:
            result["model"]["sha256"] = sha256_file(plan.model_path)

    # Expert bytes per decode step (top-k x per-layer expert bytes).
    manifest = (
        plan.tiled_dir / "manifest.json"
        if (plan.tiled_dir / "manifest.json").is_file()
        else plan.raw_dir / "manifest.json"
    )
    if manifest.is_file():
        result["experts"] = _expert_stats(manifest)

    result["status"] = "success"
    result["error"] = None
    result["generated_at"] = utc_now()

    # Cleanup is last: pack needs the GGUF, and the ledger keeps sha/bytes.
    if not keep_models and plan.model_path.is_file():
        removed = plan.model_path.stat().st_size
        try:
            plan.model_path.unlink()
            result["model"]["deleted"] = True
            result["model"]["deleted_bytes"] = removed
            print(f"removed quantized model to save disk: {plan.model_path} ({removed} B)")
        except OSError as exc:
            result["warnings"].append(f"could not delete {plan.model_path}: {exc}")

    result["artifacts"] = _artifacts(plan)
    _write_result(plan, result)
    print(f"wrote {plan.result_path}")
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def collect_results(inputs: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    paths: list[Path] = []
    warnings: list[str] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if ".." in path.parts:
            raise ArmError(f"report input: '..' components are not allowed: {raw!r}")
        path = path.resolve()
        if path.is_dir():
            paths.extend(sorted(path.rglob("arm-result.json")))
        elif path.is_file():
            paths.append(path)
        else:
            warnings.append(f"report input not found: {path}")
    seen: set[Path] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"cannot read {path}: {exc}")
            continue
        if not isinstance(record, dict):
            warnings.append(f"ignoring non-object result: {path}")
            continue
        record["_source"] = str(path)
        records.append(record)
    records.sort(key=lambda item: str(item.get("arm_id", "")))
    return records, warnings


def _ppl_value(record: dict[str, Any]) -> Optional[float]:
    ppl = record.get("ppl")
    estimate = ppl.get("estimate") if isinstance(ppl, dict) else None
    value = estimate.get("ppl") if isinstance(estimate, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _fmt(value: Optional[float], spec: str) -> str:
    return "NA" if value is None else format(value, spec)


def _imatrix_sha_value(record: dict[str, Any]) -> str:
    imatrix = record.get("imatrix")
    sha = imatrix.get("sha256") if isinstance(imatrix, dict) else None
    if isinstance(sha, str) and sha:
        return sha[:12]
    return "-"


def render_tsv(records: Sequence[dict[str, Any]]) -> str:
    lines = ["id\tPPL\tmodel_GiB\texperts_MiB_per_step\tsha256_prefix\timatrix_sha"]
    for record in records:
        model = record.get("model") or {}
        experts = record.get("experts") or {}
        model_bytes = model.get("bytes")
        gib = float(model_bytes) / (1 << 30) if isinstance(model_bytes, (int, float)) else None
        mib = experts.get("mib_per_step") if isinstance(experts, dict) else None
        sha = model.get("sha256") or ""
        lines.append(
            "\t".join(
                [
                    str(record.get("arm_id", "unknown")),
                    _fmt(_ppl_value(record), ".4f"),
                    _fmt(gib, ".2f"),
                    _fmt(float(mib) if isinstance(mib, (int, float)) else None, ".3f"),
                    str(sha)[:12] if sha else "NA",
                    _imatrix_sha_value(record),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def render_markdown(records: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| id | PPL | model GiB | experts MiB/step | sha256 | imatrix_sha |",
        "|---|---:|---:|---:|---|---|",
    ]
    for record in records:
        model = record.get("model") or {}
        experts = record.get("experts") or {}
        model_bytes = model.get("bytes")
        gib = float(model_bytes) / (1 << 30) if isinstance(model_bytes, (int, float)) else None
        mib = experts.get("mib_per_step") if isinstance(experts, dict) else None
        sha = model.get("sha256") or ""
        imatrix_sha = _imatrix_sha_value(record)
        lines.append(
            "| {id} | {ppl} | {gib} | {mib} | `{sha}` | {imatrix} |".format(
                id=record.get("arm_id", "unknown"),
                ppl=_fmt(_ppl_value(record), ".4f"),
                gib=_fmt(gib, ".2f"),
                mib=_fmt(float(mib) if isinstance(mib, (int, float)) else None, ".3f"),
                sha=(str(sha)[:12] if sha else "NA"),
                imatrix=(f"`{imatrix_sha}`" if imatrix_sha != "-" else "-"),
            )
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_plan(plan: Plan) -> None:
    print(f"arm:  {plan.arm_id}")
    print(f"out:  {plan.out_dir}")
    print(f"src:  {plan.cfg['base_model']}")
    print(f"model: {plan.model_path}")
    if plan.gpu:
        print(f"gpu:  CUDA_VISIBLE_DEVICES={plan.gpu}")
    for warning in plan.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print("\nrules (first match wins; specific rules must come before catch-all):")
    for index, rule in enumerate(plan.cfg["rules"], start=1):
        print(f"  {index}. {rule['regex']} -> {rule['type']}")
    print("\ncommands.sh:\n")
    print(commands_text(plan))


def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan(
        Path(args.arm),
        Path(args.out),
        gpu=args.gpu,
        allow_ppl_override=args.allow_ppl_override,
    )
    budget_path = (
        resolve_path(args.budget_json, "--budget-json", allow_relative=True)
        if args.budget_json
        else None
    )
    plan.budget = build_budget(
        plan,
        max_expert_gib=args.max_expert_gib,
        allow_over_budget=args.allow_over_budget,
        scope=args.budget_scope,
        expert_regex=args.expert_regex,
        budget_json=budget_path,
    )
    write_plan(plan)
    _print_plan(plan)
    print(f"wrote {plan.rules_path} (mode 0644)")
    print(f"wrote {plan.commands_path}")
    print(f"wrote {plan.plan_path}")
    _print_budget_summary(plan.budget or {})
    if budget_path is not None:
        atomic_write_json(budget_path, budget_detail(plan, plan.budget or {}), 0o644)
        print(f"wrote {budget_path}")
    if (plan.budget or {}).get("over_budget"):
        _print_over_budget(plan.budget or {})
        if not args.allow_over_budget:
            return 3
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan = build_plan(
        Path(args.arm),
        Path(args.out),
        gpu=args.gpu,
        allow_ppl_override=args.allow_ppl_override,
    )
    write_plan(plan)
    if args.dry_run:
        _print_plan(plan)
        return 0
    return run_plan(
        plan,
        resume=args.resume,
        keep_models=args.keep_models,
    )


def cmd_report(args: argparse.Namespace) -> int:
    records, warnings = collect_results(args.inputs)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not records:
        print("error: no arm-result.json found", file=sys.stderr)
        return 1
    text = render_markdown(records) if args.format == "markdown" else render_tsv(records)
    if args.out:
        out = resolve_out_dir(args.out)
        atomic_write_text(out, text, 0o644)
        print(f"wrote {out} ({len(records)} arms)")
    else:
        sys.stdout.write(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan/run/report mixed-precision quantization arms (stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--arm", required=True, help="arm JSON description")
        p.add_argument("--out", required=True, help="output directory for the arm")
        p.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value (e.g. 0 or 0,1)")
        p.add_argument(
            "--allow-ppl-override",
            action="store_true",
            help="allow non-default PPL chunks/ctx/batch/ubatch/threads",
        )

    plan = sub.add_parser("plan", help="validate an arm and print/write paste-ready commands")
    add_common(plan)
    plan.add_argument("--dry-run", action="store_true", help="synonym for plan (nothing is executed)")
    plan.add_argument(
        "--max-expert-gib",
        type=float,
        default=None,
        help="reject the plan when the estimated expert package exceeds this many GiB",
    )
    plan.add_argument(
        "--allow-over-budget",
        action="store_true",
        help="warn instead of exiting 3 when the expert budget is exceeded",
    )
    plan.add_argument(
        "--budget-scope",
        choices=BUDGET_SCOPES,
        default="tiled",
        help="which byte figure the budget applies to",
    )
    plan.add_argument(
        "--budget-json",
        default=None,
        help="write per-layer/per-family budget detail to this JSON path",
    )
    plan.add_argument(
        "--expert-regex",
        default=None,
        help=f"tensor-name regex for expert tensors (default: {DEFAULT_EXPERT_REGEX})",
    )
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="execute an arm and write arm-result.json")
    add_common(run)
    run.add_argument("--resume", action="store_true", help="skip outputs that already exist")
    run.add_argument(
        "--keep-models",
        action="store_true",
        help="keep the quantized GGUF after PPL/pack (default: delete it)",
    )
    run.add_argument("--dry-run", action="store_true", help="only write/print the plan")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="aggregate arm-result.json files to TSV/Markdown")
    report.add_argument("inputs", nargs="+", help="result files or directories to scan")
    report.add_argument("--format", choices=("tsv", "markdown"), default="tsv")
    report.add_argument("--out", help="write the table here instead of stdout")
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "allow_ppl_override"):
        args.allow_ppl_override = False
    try:
        return int(args.func(args))
    except ArmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
