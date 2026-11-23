#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solve_k import CAPS, UNIT, format_vs_base, k0_baseline, plan_scenario  # noqa: E402


def window_curve(value=0.0):
    return {0: [value] * len(CAPS)}


def window_curves(value=0.0, layers=(0,)):
    return {layer: [value] * len(CAPS) for layer in layers}


def sweep_json(useful=0.0, waste=0.0, capacity=8, demand=8.0, curve_k=8):
    return {
        "capacity": capacity,
        "lam": 0.5,
        "layers": {
            "0": {
                "demand_per_step": demand,
                "curve": [{"k": curve_k, "useful": useful, "waste": waste}],
            }
        },
    }


def two_layer_sweep(capacity=72):
    sweep = sweep_json(capacity=capacity)
    sweep["layers"]["1"] = dict(sweep["layers"]["0"])
    return sweep


TWO_LAYER_TRACE = {0: [(list(range(8)), list(range(8)))],
                   1: [(list(range(8)), list(range(8)))]}


class SolveKAccountingTest(unittest.TestCase):
    def test_successful_prefetch_still_counts_as_physical_read(self):
        scenario = plan_scenario(
            [0], window_curve(), window_curve(), sweep_json(useful=8.0), budget=16, K=8
        )
        self.assertEqual(scenario["capacities"]["0"], UNIT)
        self.assertEqual(scenario["sync"], 0.0)
        self.assertEqual(scenario["prefetch_useful"], 8.0)
        self.assertEqual(scenario["reads"], 8.0)

    def test_physical_reads_are_sync_plus_useful_plus_waste(self):
        scenario = plan_scenario(
            [0],
            window_curve(),
            window_curve(),
            sweep_json(useful=2.0, waste=3.0),
            budget=16,
            K=8,
        )
        self.assertEqual(scenario["sync"], 6.0)
        self.assertEqual(scenario["prefetch_useful"], 2.0)
        self.assertEqual(scenario["prefetch_waste"], 3.0)
        self.assertEqual(scenario["reads"], 11.0)
        self.assertEqual(scenario["reads"], scenario["demand"] + scenario["prefetch_waste"])

    def test_rejects_stale_capacity_without_raw_trace(self):
        with self.assertRaisesRegex(ValueError, "does not match sweep capacity"):
            plan_scenario(
                [0],
                window_curve(),
                window_curve(),
                sweep_json(capacity=72),
                budget=16,
                K=0,
            )

    def test_rejects_missing_exact_curve_point_without_raw_trace(self):
        with self.assertRaisesRegex(ValueError, r"K=4: sweep layer L0 has no exact prefetch curve point"):
            plan_scenario(
                [0],
                window_curve(),
                window_curve(),
                sweep_json(),
                budget=16,
                K=4,
            )

    def test_two_layer_default_plan_is_per_layer_hard_cap(self):
        # Without --max-layer-slots, --budget is the per-layer cap; the old
        # global-pool allocation produced an unsafe 144-slot hot layer here.
        scenario = plan_scenario(
            [0, 1],
            window_curves(layers=(0, 1)),
            window_curves(layers=(0, 1)),
            two_layer_sweep(),
            budget=80,
            K=4,
            trace_layers=TWO_LAYER_TRACE,
        )
        capacities = [scenario["capacities"][str(layer)] for layer in (0, 1)]
        self.assertEqual(scenario["max_layer_slots"], 80)
        self.assertEqual(capacities, [72, 72])
        self.assertTrue(all(cap + 4 <= 80 for cap in capacities))

    def test_explicit_max_layer_slots_caps_capacity_plus_prefetch(self):
        scenario = plan_scenario(
            [0, 1],
            window_curves(layers=(0, 1)),
            window_curves(layers=(0, 1)),
            two_layer_sweep(),
            budget=80,
            K=4,
            trace_layers=TWO_LAYER_TRACE,
            max_layer_slots=96,
        )
        capacities = [scenario["capacities"][str(layer)] for layer in (0, 1)]
        self.assertEqual(scenario["max_layer_slots"], 96)
        self.assertEqual(sum(capacities), 152)  # (80 - 4) slots x 2 layers
        self.assertLessEqual(max(capacities) + 4, 96)
        self.assertGreater(max(capacities) + 4, 80)  # average pool, not hard cap

    def test_lower_max_layer_slots_restricts_below_budget(self):
        scenario = plan_scenario(
            [0, 1],
            window_curves(layers=(0, 1)),
            window_curves(layers=(0, 1)),
            two_layer_sweep(),
            budget=80,
            K=4,
            trace_layers=TWO_LAYER_TRACE,
            max_layer_slots=72,
        )
        capacities = [scenario["capacities"][str(layer)] for layer in (0, 1)]
        self.assertEqual(capacities, [64, 64])
        self.assertTrue(all(cap + 4 <= 72 for cap in capacities))

    def test_rejects_non_positive_max_layer_slots(self):
        with self.assertRaisesRegex(ValueError, "max-layer-slots 0.*positive"):
            plan_scenario(
                [0, 1],
                window_curves(layers=(0, 1)),
                window_curves(layers=(0, 1)),
                two_layer_sweep(),
                budget=80,
                K=4,
                trace_layers=TWO_LAYER_TRACE,
                max_layer_slots=0,
            )

    def test_raw_trace_reanalyzes_at_allocated_capacity(self):
        steps = {0: [(list(range(8)), list(range(8)))]}
        scenario = plan_scenario(
            [0],
            window_curve(),
            window_curve(),
            sweep_json(capacity=72),
            budget=16,
            K=8,
            trace_layers=steps,
        )
        self.assertEqual(scenario["sync"], 0.0)
        self.assertEqual(scenario["prefetch_useful"], 8.0)
        self.assertEqual(scenario["reads"], 8.0)

    def test_raw_trace_uses_requested_k_not_sweep_max(self):
        steps = {0: [(list(range(8)), list(range(8)))]}
        scenario = plan_scenario(
            [0],
            window_curve(),
            window_curve(),
            sweep_json(capacity=72),
            budget=16,
            K=4,
            trace_layers=steps,
        )
        self.assertEqual(scenario["sync"], 4.0)
        self.assertEqual(scenario["prefetch_useful"], 4.0)
        self.assertEqual(scenario["reads"], 8.0)

    def test_baseline_requires_actual_k0_scenario(self):
        # Only running --prefetch 8 must not treat K=8 as the "vs K=0" base.
        scenarios = [{"K": 8, "reads": 10.0}]
        self.assertIsNone(k0_baseline(scenarios))
        self.assertEqual(format_vs_base(10.0, k0_baseline(scenarios)), "n/a")

    def test_baseline_uses_k0_regardless_of_order(self):
        zero = {"K": 0, "reads": 10.0}
        eight = {"K": 8, "reads": 11.0}
        self.assertEqual(k0_baseline([zero, eight]), 10.0)
        self.assertEqual(k0_baseline([eight, zero]), 10.0)
        self.assertEqual(format_vs_base(11.0, 10.0).strip(), "+10.0%")

    def test_baseline_uses_unrounded_physical_reads(self):
        base = plan_scenario(
            [0],
            window_curve(),
            window_curve(),
            sweep_json(demand=0.006),
            budget=10,
            K=0,
        )
        prefetched = plan_scenario(
            [0],
            window_curve(),
            window_curve(),
            sweep_json(waste=0.008, demand=0.006, curve_k=2),
            budget=10,
            K=2,
        )

        self.assertEqual(base["reads"], 0.006)
        self.assertEqual(prefetched["reads"], 0.014)
        self.assertEqual(
            format_vs_base(prefetched["reads"], k0_baseline([prefetched, base])).strip(),
            "+133.3%",
        )

    def test_small_positive_baseline_is_not_reported_as_na(self):
        base = plan_scenario(
            [0],
            window_curve(),
            window_curve(),
            sweep_json(demand=0.004),
            budget=10,
            K=0,
        )
        prefetched = plan_scenario(
            [0],
            window_curve(),
            window_curve(),
            sweep_json(waste=0.001, demand=0.004, curve_k=2),
            budget=10,
            K=2,
        )

        self.assertEqual(base["reads"], 0.004)
        self.assertEqual(
            format_vs_base(prefetched["reads"], k0_baseline([base, prefetched])).strip(),
            "+25.0%",
        )

    def test_rejects_negative_k(self):
        with self.assertRaisesRegex(ValueError, "K=-8.*non-negative"):
            plan_scenario(
                [0],
                window_curve(),
                window_curve(),
                sweep_json(),
                budget=8,
                K=-8,
            )

    def test_cli_skips_budget_infeasible_candidates_and_writes_k0(self):
        header = ["layer"]
        header.extend(f"train{c}" for c in CAPS)
        header.extend(f"held{c}" for c in CAPS)
        row = ["0"]
        row.extend("0.0" for _ in CAPS)
        row.extend("0.0" for _ in CAPS)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            win = tmp_path / "win.csv"
            sweep = tmp_path / "sweep.json"
            out = tmp_path / "plan.json"
            win.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
            sweep.write_text(json.dumps(sweep_json()))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "solve_k.py"),
                    "--win",
                    str(win),
                    "--sweep",
                    str(sweep),
                    "--budget",
                    "8",
                    "--json",
                    str(out),
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out.exists(), proc.stderr)
            plan = json.loads(out.read_text())
            self.assertEqual([scenario["K"] for scenario in plan["scenarios"]], [0])

    def test_cli_rejects_negative_k_without_writing_plan(self):
        header = ["layer"]
        header.extend(f"train{c}" for c in CAPS)
        header.extend(f"held{c}" for c in CAPS)
        row = ["0"]
        row.extend("0.0" for _ in CAPS)
        row.extend("0.0" for _ in CAPS)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            win = tmp_path / "win.csv"
            sweep = tmp_path / "sweep.json"
            out = tmp_path / "plan.json"
            win.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
            sweep.write_text(json.dumps(sweep_json()))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "solve_k.py"),
                    "--win",
                    str(win),
                    "--sweep",
                    str(sweep),
                    "--budget",
                    "8",
                    "--prefetch=-8",
                    "--json",
                    str(out),
                ],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("K=-8", proc.stderr)
            self.assertIn("non-negative", proc.stderr)
            self.assertFalse(out.exists())

    def test_cli_fails_when_all_candidates_are_budget_infeasible(self):
        header = ["layer"]
        header.extend(f"train{c}" for c in CAPS)
        header.extend(f"held{c}" for c in CAPS)
        row = ["0"]
        row.extend("0.0" for _ in CAPS)
        row.extend("0.0" for _ in CAPS)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            win = tmp_path / "win.csv"
            sweep = tmp_path / "sweep.json"
            out = tmp_path / "plan.json"
            win.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
            sweep.write_text(json.dumps(sweep_json()))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "solve_k.py"),
                    "--win",
                    str(win),
                    "--sweep",
                    str(sweep),
                    "--budget",
                    "2",
                    "--prefetch",
                    "8",
                    "--json",
                    str(out),
                ],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("no feasible prefetch candidates", proc.stderr)
            self.assertIn("budget=2", proc.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
