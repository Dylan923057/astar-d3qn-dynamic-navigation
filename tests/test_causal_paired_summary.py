from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from summarize_causal_paired_results import summarize_test_rows


class CausalPairedSummaryTests(unittest.TestCase):
    def test_condition_rates_and_pair_degradation(self) -> None:
        scenarios = {
            1: {"pair_id": "a", "pair_condition": "control"},
            2: {"pair_id": "a", "pair_condition": "conflict"},
            3: {"pair_id": "b", "pair_condition": "control"},
            4: {"pair_id": "b", "pair_condition": "conflict"},
        }
        rows = [
            self._row(1, safe=1, collision=0, steps=10, waits=0),
            self._row(2, safe=0, collision=1, steps=7, waits=1),
            self._row(3, safe=1, collision=0, steps=12, waits=2),
            self._row(4, safe=1, collision=0, steps=13, waits=3),
        ]
        result = summarize_test_rows(rows, scenarios)
        self.assertEqual(result["control_safe_success_rate"], 1.0)
        self.assertEqual(result["conflict_safe_success_rate"], 0.5)
        self.assertEqual(result["safe_success_gap_conflict_minus_control"], -0.5)
        self.assertEqual(result["dynamic_collision_gap_conflict_minus_control"], 0.5)
        self.assertEqual(result["paired_degradation_rate"], 0.5)
        self.assertEqual(result["paired_robust_both_rate"], 0.5)

    @staticmethod
    def _row(
        scenario_id: int, safe: int, collision: int, steps: int, waits: int
    ) -> dict[str, float | int]:
        return {
            "scenario_id": scenario_id,
            "safe_success": safe,
            "success": safe,
            "dynamic_collision": collision,
            "steps": steps,
            "wait_steps": waits,
        }


if __name__ == "__main__":
    unittest.main()
