#!/usr/bin/env python3
"""Unit tests for tools/moe/hot_cold_plan.py (host-only, stdlib unittest)."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import hot_cold_plan as hcp  # noqa: E402

N_EXPERT = 256
LAYERS = (0, 1)
FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
SHAPES = {
    "ffn_gate_exps": [64, 512, N_EXPERT],
    "ffn_up_exps": [512, 64, N_EXPERT],
    "ffn_down_exps": [256, 128, N_EXPERT],
}
N_PARAMS = 64 * 512
COLD_BPE = N_PARAMS // 128 * 18      # Q1_0 = 4608
HOT_BPE = N_PARAMS // 256 * 84       # Q2_K  = 10752


def _entry_bytes(family: str) -> int:
    del family
    return COLD_BPE


class PlanFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="hot-cold-plan-test-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write_manifest(
        self,
        layers: tuple[int, ...] = LAYERS,
        families: tuple[str, ...] = FAMILIES,
        *,
        shared_file: bool = True,
        quant_type: str = "Q1_0",
        mutate_entry=None,
        mutate_document=None,
    ) -> Path:
        entries = []
        for layer in layers:
            stride = sum(_entry_bytes(family) for family in families)
            offset = 0
            for family in families:
                bpe = _entry_bytes(family)
                entry = {
                    "layer": layer,
                    "tensor_family": family,
                    "tensor_name": f"blk.{layer}.{family}.weight",
                    "quant_type": quant_type,
                    "shape": list(SHAPES[family]),
                    "exact_byte_length": bpe * N_EXPERT,
                    "bytes_per_expert": bpe,
                    "repacked_file": (
                        f"layer_{layer:03d}.bin" if shared_file
                        else f"layer_{layer:03d}.{family}.bin"
                    ),
                    "repacked_offset": offset if shared_file else 0,
                    "expert_stride": stride if shared_file else bpe,
                }
                if mutate_entry is not None:
                    mutate_entry(entry)
                entries.append(entry)
                offset += bpe
        document = {
            "schema_version": 1,
            "layout": "layer_major_expert",
            "model": {"arch": "test", "expert_count": N_EXPERT, "expert_used_count": 8},
            "entries": entries,
        }
        if mutate_document is not None:
            mutate_document(document)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def write_groups(self, rows, *, header_override=None) -> Path:
        header = header_override or "layer\tfamily\thot_count\thot_experts_by_frequency_rank"
        lines = [header]
        for row in rows:
            layer, family, count, experts = row
            if isinstance(experts, str):
                experts_text = experts
            else:
                experts_text = ",".join(str(expert) for expert in experts)
            lines.append(f"{layer}\t{family}\t{count}\t{experts_text}")
        path = self.root / "groups.tsv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def rows_for_hot(self, hot_count: int, hot_experts=None, layers=LAYERS,
                     families=FAMILIES):
        if hot_experts is None:
            hot_experts = list(range(hot_count))
        return [
            (layer, family, hot_count, list(hot_experts))
            for layer in layers
            for family in families
        ]

    def plan(self, groups_path: Path, manifest_path: Path | None = None, **kwargs):
        manifest_path = manifest_path or self.write_manifest()
        cold_type = kwargs.pop("cold_type", "Q1_0")
        hot_type = kwargs.pop("hot_type", "Q2_K")
        n_expert = kwargs.pop("n_expert", N_EXPERT)
        entries, manifest_n = hcp.load_manifest(manifest_path, cold_type)
        groups = hcp.parse_groups(groups_path, n_expert)
        return hcp.build_plan_from(
            entries, manifest_n, groups, hot_type, cold_type, **kwargs
        )

    # -- happy paths -------------------------------------------------------

    def test_hot_count_two_partition_order_bytes_offsets(self) -> None:
        groups = self.write_groups(
            self.rows_for_hot(2, [31, 152])
        )
        plan, implicit = self.plan(groups)
        self.assertEqual(implicit, 0)
        self.assertEqual(plan["layout"], "hexagon-tiled-hotcold-v1")
        self.assertEqual(plan["n_expert"], N_EXPERT)
        self.assertEqual(len(plan["entries"]), 6)
        # canonical family order from experts.py: gate, up, down
        self.assertEqual(
            [item["family"] for item in plan["entries"][:3]],
            list(FAMILIES),
        )

        item = plan["entries"][0]
        self.assertEqual(item["hot_experts"], [31, 152])          # TSV frequency order kept
        self.assertEqual(len(item["cold_experts"]), 254)
        self.assertEqual(item["cold_experts"], sorted(item["cold_experts"]))
        self.assertNotIn(31, item["cold_experts"])
        self.assertNotIn(152, item["cold_experts"])
        self.assertEqual(item["order"], [31, 152] + item["cold_experts"])
        self.assertEqual(sorted(item["order"]), list(range(N_EXPERT)))

        self.assertEqual(item["hot"]["file"], "layer_000.ffn_gate_exps.hot.bin")
        self.assertEqual(item["cold"]["file"], "layer_000.ffn_gate_exps.cold.bin")
        self.assertEqual(item["hot"]["offset"], 0)
        self.assertEqual(item["cold"]["offset"], 0)
        self.assertEqual(item["hot"]["bytes_per_expert"], HOT_BPE)
        self.assertEqual(item["cold"]["bytes_per_expert"], COLD_BPE)
        self.assertEqual(item["hot"]["count"], 2)
        self.assertEqual(item["cold"]["count"], 254)
        self.assertEqual(item["hot"]["bytes"], 2 * HOT_BPE)
        self.assertEqual(item["cold"]["bytes"], 254 * COLD_BPE)

        totals = plan["totals"]
        self.assertEqual(totals["hot_bytes"], 6 * 2 * HOT_BPE)
        self.assertEqual(totals["cold_bytes"], 6 * 254 * COLD_BPE)
        self.assertEqual(totals["source_bytes"], 6 * N_EXPERT * COLD_BPE)
        self.assertEqual(totals["planned_bytes"], totals["hot_bytes"] + totals["cold_bytes"])
        self.assertEqual(totals["delta_bytes"], totals["planned_bytes"] - totals["source_bytes"])
        self.assertEqual(totals["delta_bytes"], 6 * 2 * (HOT_BPE - COLD_BPE))

    def test_hot_count_zero(self) -> None:
        groups = self.write_groups(self.rows_for_hot(0, []))
        plan, implicit = self.plan(groups)
        self.assertEqual(implicit, 0)
        for item in plan["entries"]:
            self.assertEqual(item["hot_count"], 0)
            self.assertEqual(item["hot_experts"], [])
            self.assertEqual(item["hot"]["count"], 0)
            self.assertEqual(item["hot"]["bytes"], 0)
            self.assertEqual(item["cold"]["count"], N_EXPERT)
            self.assertEqual(item["cold"]["bytes"], N_EXPERT * COLD_BPE)
            self.assertEqual(item["order"], list(range(N_EXPERT)))
        self.assertEqual(plan["totals"]["hot_bytes"], 0)
        self.assertEqual(plan["totals"]["planned_bytes"], plan["totals"]["source_bytes"])
        self.assertEqual(plan["totals"]["delta_bytes"], 0)

    def test_hot_count_255(self) -> None:
        groups = self.write_groups(self.rows_for_hot(255, list(range(255))))
        plan, _implicit = self.plan(groups)
        for item in plan["entries"]:
            self.assertEqual(item["hot_count"], 255)
            self.assertEqual(item["cold_experts"], [255])
            self.assertEqual(item["hot"]["count"], 255)
            self.assertEqual(item["cold"]["count"], 1)
            self.assertEqual(item["cold"]["bytes"], COLD_BPE)
            self.assertEqual(item["hot"]["bytes"], 255 * HOT_BPE)
            self.assertEqual(item["order"][-1], 255)
        self.assertEqual(plan["totals"]["cold_bytes"], 6 * COLD_BPE)

    def test_hot_count_256(self) -> None:
        groups = self.write_groups(self.rows_for_hot(256, list(range(256))))
        plan, _implicit = self.plan(groups)
        for item in plan["entries"]:
            self.assertEqual(item["hot_count"], 256)
            self.assertEqual(item["cold_experts"], [])
            self.assertEqual(item["cold"]["count"], 0)
            self.assertEqual(item["cold"]["bytes"], 0)
            self.assertEqual(item["hot_experts"], list(range(256)))
            self.assertEqual(item["order"], list(range(256)))
        self.assertEqual(plan["totals"]["cold_bytes"], 0)
        self.assertEqual(plan["totals"]["hot_bytes"], 6 * 256 * HOT_BPE)

    def test_single_family_files_validate(self) -> None:
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        manifest = self.write_manifest(shared_file=False)
        plan, _implicit = self.plan(groups, manifest)
        for item in plan["entries"]:
            self.assertEqual(item["source_offset"], 0)
            self.assertEqual(item["source_expert_stride"], COLD_BPE)
            self.assertIn(item["family"], item["source_file"])
            self.assertTrue(item["source_file"].startswith(f"layer_{item['layer']:03d}."))

    def test_missing_group_rows_default_to_hot_count_zero(self) -> None:
        # Real T-0041 groups omit zero-hot pairs; layer 0 only has ffn_down_exps.
        rows = [
            (layer, "ffn_down_exps", 2, [31, 152])
            for layer in LAYERS
        ]
        groups = self.write_groups(rows)
        plan, implicit = self.plan(groups)
        self.assertEqual(implicit, 4)
        self.assertEqual(len(plan["entries"]), 6)
        by_key = {(item["layer"], item["family"]): item for item in plan["entries"]}
        for layer in LAYERS:
            for family in ("ffn_gate_exps", "ffn_up_exps"):
                self.assertEqual(by_key[(layer, family)]["hot_count"], 0)
                self.assertEqual(by_key[(layer, family)]["hot"]["bytes"], 0)
            self.assertEqual(by_key[(layer, "ffn_down_exps")]["hot_experts"], [31, 152])

    def test_require_groups_exact_rejects_missing_rows(self) -> None:
        groups = self.write_groups([(0, "ffn_down_exps", 2, [31, 152])])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, require_groups_exact=True)

    def test_same_hot_and_cold_type_keeps_source_bytes(self) -> None:
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        plan, _implicit = self.plan(groups, hot_type="Q1_0", cold_type="Q1_0")
        self.assertEqual(plan["totals"]["planned_bytes"], plan["totals"]["source_bytes"])
        self.assertEqual(plan["totals"]["delta_bytes"], 0)
        for item in plan["entries"]:
            self.assertEqual(item["hot"]["bytes_per_expert"], COLD_BPE)

    # -- determinism / CLI -------------------------------------------------

    def test_two_runs_are_byte_identical(self) -> None:
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        manifest = self.write_manifest()
        plan_a, _ = self.plan(groups, manifest)
        plan_b, _ = self.plan(groups, manifest)
        self.assertEqual(
            json.dumps(plan_a, sort_keys=True),
            json.dumps(plan_b, sort_keys=True),
        )
        out_a = self.root / "a.json"
        out_b = self.root / "b.json"
        hcp.write_json_atomic(out_a, plan_a)
        hcp.write_json_atomic(out_b, plan_b)
        self.assertEqual(out_a.read_bytes(), out_b.read_bytes())

    def test_cli_writes_plan_and_check_only(self) -> None:
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        manifest = self.write_manifest()
        out = self.root / "plan.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = hcp.main([
                "--manifest", str(manifest), "--groups", str(groups), "--out", str(out),
            ])
        self.assertEqual(rc, 0)
        summary = json.loads(stdout.getvalue())
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["entries"], 6)
        plan = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(plan["totals"], summary["totals"])

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = hcp.main(["--groups", str(groups), "--check-only"])
        self.assertEqual(rc, 0)
        only = json.loads(stdout.getvalue())
        self.assertEqual(only["mode"], "groups-only")
        self.assertEqual(only["groups"], 6)
        self.assertEqual(only["hot_count_histogram"]["1-16"], 6)

    def test_cli_requires_manifest_without_check_only(self) -> None:
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = hcp.main(["--groups", str(groups), "--out", str(self.root / "x.json")])
        self.assertEqual(rc, 2)
        self.assertIn("--manifest is required", stderr.getvalue())

    # -- groups validation errors -----------------------------------------

    def test_rejects_duplicate_hot_expert(self) -> None:
        groups = self.write_groups([(0, "ffn_down_exps", 2, [5, 5])])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)

    def test_rejects_hot_count_mismatch(self) -> None:
        groups = self.write_groups([(0, "ffn_down_exps", 2, [5])])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)

    def test_rejects_hot_count_out_of_range(self) -> None:
        groups = self.write_groups([(0, "ffn_down_exps", N_EXPERT + 1, list(range(N_EXPERT)))])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)

    def test_rejects_hot_expert_out_of_range(self) -> None:
        groups = self.write_groups([(0, "ffn_down_exps", 1, [N_EXPERT])])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)

    def test_rejects_unknown_family_in_groups(self) -> None:
        groups = self.write_groups([(0, "ffn_foo_exps", 2, [1, 2])])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)

    def test_rejects_duplicate_group_row(self) -> None:
        groups = self.write_groups([
            (0, "ffn_down_exps", 2, [1, 2]),
            (0, "ffn_down_exps", 2, [3, 4]),
        ])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)

    def test_rejects_bad_groups_header(self) -> None:
        groups = self.write_groups(
            [(0, "ffn_down_exps", 2, [1, 2])],
            header_override="layer\tfamily\thot_count",
        )
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)

    # -- manifest validation errors ---------------------------------------

    def test_rejects_type_mismatch(self) -> None:
        manifest = self.write_manifest(quant_type="Q2_K")
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_unknown_family_in_manifest(self) -> None:
        manifest = self.write_manifest(
            mutate_entry=lambda entry: entry.update(tensor_family="ffn_nope_exps")
        )
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_bad_shared_stride(self) -> None:
        def mutate(entry):
            if entry["layer"] == 0 and entry["tensor_family"] == "ffn_down_exps":
                entry["expert_stride"] += 1

        manifest = self.write_manifest(mutate_entry=mutate)
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_noncontiguous_offset(self) -> None:
        def mutate(entry):
            if entry["layer"] == 0 and entry["tensor_family"] == "ffn_down_exps":
                entry["repacked_offset"] += 1

        manifest = self.write_manifest(mutate_entry=mutate)
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_shape_expert_axis_mismatch(self) -> None:
        def mutate(entry):
            entry["shape"] = [64, 512, N_EXPERT - 1]

        manifest = self.write_manifest(mutate_entry=mutate)
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_raw_bytes_not_matching_cold_formula(self) -> None:
        def mutate(entry):
            if entry["layer"] == 0 and entry["tensor_family"] == "ffn_gate_exps":
                entry["bytes_per_expert"] += 18

        manifest = self.write_manifest(mutate_entry=mutate)
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_duplicate_manifest_entry(self) -> None:
        def mutate(document):
            document["entries"].append(dict(document["entries"][0]))

        manifest = self.write_manifest(mutate_document=mutate)
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_inconsistent_expert_count(self) -> None:
        def mutate(document):
            document["entries"][0]["expert_count"] = N_EXPERT - 1

        manifest = self.write_manifest(mutate_document=mutate)
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_group_pair_absent_from_manifest(self) -> None:
        manifest = self.write_manifest(layers=(0,), families=("ffn_down_exps",))
        groups = self.write_groups([(0, "ffn_gate_exps", 2, [31, 152])])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_tiled_manifest_input(self) -> None:
        def mutate(document):
            document["backend_layout"] = "hexagon-tiled-v1"

        manifest = self.write_manifest(mutate_document=mutate)
        groups = self.write_groups(self.rows_for_hot(2, [31, 152]))
        with self.assertRaises(hcp.PlanError):
            self.plan(groups, manifest)

    def test_rejects_empty_groups(self) -> None:
        groups = self.write_groups([])
        with self.assertRaises(hcp.PlanError):
            self.plan(groups)


if __name__ == "__main__":
    unittest.main()
