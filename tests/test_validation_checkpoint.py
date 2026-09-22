from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.evaluation.checkpoints import validation_checkpoint_score


class ValidationCheckpointTests(unittest.TestCase):
    def summary(self, safe: float, success: float, collisions: float, steps: float):
        return {
            "safe_success_rate": safe,
            "success_rate": success,
            "mean_dynamic_collision_count": collisions,
            "mean_steps": steps,
        }

    def test_safe_success_is_the_primary_selection_metric(self) -> None:
        safer = self.summary(0.95, 0.95, 0.0, 100.0)
        faster_but_less_safe = self.summary(0.90, 1.0, 0.0, 76.0)

        self.assertGreater(
            validation_checkpoint_score(safer),
            validation_checkpoint_score(faster_but_less_safe),
        )

    def test_ties_prefer_completion_then_collisions_then_steps(self) -> None:
        baseline = self.summary(0.90, 0.95, 0.2, 90.0)
        self.assertGreater(
            validation_checkpoint_score(self.summary(0.90, 1.0, 0.5, 120.0)),
            validation_checkpoint_score(baseline),
        )
        self.assertGreater(
            validation_checkpoint_score(self.summary(0.90, 0.95, 0.1, 120.0)),
            validation_checkpoint_score(baseline),
        )
        self.assertGreater(
            validation_checkpoint_score(self.summary(0.90, 0.95, 0.2, 80.0)),
            validation_checkpoint_score(baseline),
        )


if __name__ == "__main__":
    unittest.main()
