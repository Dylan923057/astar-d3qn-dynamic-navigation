from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_action_ranking_ablation import (
    threshold_confirmation_step,
    validation_auc,
)


class ActionRankingSummaryTests(unittest.TestCase):
    def test_validation_auc_uses_the_complete_fixed_budget(self):
        curve = [
            {"environment_steps": "0", "conflict_safe_success": "0.0"},
            {"environment_steps": "50", "conflict_safe_success": "0.5"},
            {"environment_steps": "100", "conflict_safe_success": "1.0"},
        ]

        self.assertAlmostEqual(validation_auc(curve, budget=100), 0.5)

    def test_validation_auc_rejects_incomplete_curve(self):
        curve = [
            {"environment_steps": "10", "conflict_safe_success": "0.5"},
            {"environment_steps": "100", "conflict_safe_success": "1.0"},
        ]

        with self.assertRaisesRegex(ValueError, "complete fixed-step budget"):
            validation_auc(curve, budget=100)

    def test_threshold_requires_consecutive_validation_passes(self):
        curve = [
            {"environment_steps": "0", "conflict_safe_success": "0.0"},
            {"environment_steps": "10", "conflict_safe_success": "0.9"},
            {"environment_steps": "20", "conflict_safe_success": "0.8"},
            {"environment_steps": "30", "conflict_safe_success": "0.9"},
            {"environment_steps": "40", "conflict_safe_success": "1.0"},
        ]

        self.assertEqual(
            threshold_confirmation_step(
                curve,
                required_passes=2,
                threshold=0.9,
            ),
            40,
        )


if __name__ == "__main__":
    unittest.main()
