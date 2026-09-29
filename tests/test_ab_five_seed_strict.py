from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ab_five_seed_strict import (
    auc,
    bootstrap_ci,
    exact_sign_flip_pvalue,
    xlsx_cell_value,
)
from run_ab_five_seed_strict import MAP_IDS, SEEDS, command_for, frozen_registration


class StrictFiveSeedProtocolTests(unittest.TestCase):
    def test_registration_locks_maps_seeds_and_effective_global_weight(self):
        registration = frozen_registration(
            ROOT / "configs" / "risk_handover_v1.yaml",
            "outputs/test_ab_five_seed_strict",
        )

        self.assertEqual(tuple(registration["maps"]), MAP_IDS)
        self.assertEqual(tuple(registration["seeds"]), SEEDS)
        self.assertEqual(registration["adaptation_steps"], 200_000)
        self.assertEqual(registration["global_prediction_spatial_weight"], 1.0)
        self.assertTrue(registration["validation_only"])

    def test_global_prediction_command_is_validation_only_and_frozen(self):
        command = command_for(
            "global_prediction",
            2,
            "outputs/test_ab_five_seed_strict",
            "cpu",
            1,
        )

        self.assertIn("--defer-test", command)
        self.assertEqual(command[command.index("--seeds") + 1:command.index("--device")], ["0", "1", "2", "3", "4"])
        self.assertEqual(command[command.index("--prediction-loss-weight") + 1], "0.1")
        self.assertEqual(command[command.index("--prediction-pos-weight") + 1], "20")
        self.assertEqual(command[command.index("--prediction-decision-zone-weight") + 1], "1")


class StrictFiveSeedStatisticsTests(unittest.TestCase):
    def test_auc_uses_all_frozen_validation_checkpoints(self):
        curve = [
            {
                "environment_steps": str(step),
                "conflict_safe_success": str(step / 200_000),
            }
            for step in range(0, 200_001, 10_000)
        ]

        self.assertAlmostEqual(auc(curve), 0.5)

    def test_auc_rejects_changed_validation_schedule(self):
        curve = [
            {"environment_steps": "0", "conflict_safe_success": "0.0"},
            {"environment_steps": "200000", "conflict_safe_success": "1.0"},
        ]

        with self.assertRaisesRegex(ValueError, "frozen 10k schedule"):
            auc(curve)

    def test_paired_statistics_are_deterministic(self):
        differences = [0.01, 0.02, 0.03, 0.04, 0.05]
        first = bootstrap_ci(differences, np.random.default_rng(7), draws=2_000)
        second = bootstrap_ci(differences, np.random.default_rng(7), draws=2_000)

        self.assertEqual(first, second)
        self.assertEqual(exact_sign_flip_pvalue(differences), 0.0625)

    def test_xlsx_collection_values_are_serialized(self):
        self.assertEqual(xlsx_cell_value([1.0, 2.0]), "[1.0, 2.0]")
        self.assertEqual(xlsx_cell_value({"upper": 2, "lower": 1}), '{"lower": 1, "upper": 2}')
        self.assertEqual(xlsx_cell_value(np.float64(0.5)), 0.5)


if __name__ == "__main__":
    unittest.main()
