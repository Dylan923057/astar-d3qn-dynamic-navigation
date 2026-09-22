from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.evaluation.checkpoints import checkpoint_metrics, checkpoint_rank


def evaluation_row(
    *,
    success: float,
    steps: int,
    collision_count: int = 0,
    wait_steps: int = 0,
    revisit_count: int = 0,
    reward: float = 0.0,
) -> dict:
    return {
        "success": success,
        "steps": steps,
        "collision": float(collision_count > 0),
        "collision_count": collision_count,
        "wait_steps": wait_steps,
        "revisit_count": revisit_count,
        "reward": reward,
    }


class CheckpointSelectionTests(unittest.TestCase):
    def test_success_always_ranks_above_failure(self) -> None:
        failure = checkpoint_metrics(
            [evaluation_row(success=0.0, steps=350, reward=1.0)]
        )
        success = checkpoint_metrics(
            [evaluation_row(success=1.0, steps=100, collision_count=2, reward=10.0)]
        )
        self.assertGreater(checkpoint_rank(success), checkpoint_rank(failure))

    def test_shorter_success_path_ranks_first(self) -> None:
        longer = checkpoint_metrics(
            [evaluation_row(success=1.0, steps=82, reward=12.0)]
        )
        shorter = checkpoint_metrics(
            [evaluation_row(success=1.0, steps=78, reward=12.0)]
        )
        self.assertGreater(checkpoint_rank(shorter), checkpoint_rank(longer))

    def test_equal_success_path_prefers_fewer_collisions_then_waits(self) -> None:
        collisions = checkpoint_metrics(
            [evaluation_row(success=1.0, steps=78, collision_count=1, reward=12.0)]
        )
        waits = checkpoint_metrics(
            [evaluation_row(success=1.0, steps=78, wait_steps=2, reward=12.0)]
        )
        clean = checkpoint_metrics(
            [evaluation_row(success=1.0, steps=78, reward=12.0)]
        )
        self.assertGreater(checkpoint_rank(waits), checkpoint_rank(collisions))
        self.assertGreater(checkpoint_rank(clean), checkpoint_rank(waits))


if __name__ == "__main__":
    unittest.main()
