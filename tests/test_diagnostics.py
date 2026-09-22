from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action
from astar_d3qn.envs.types import Observation
from astar_d3qn.evaluation.diagnostics import (
    demonstration_action_diagnostics,
    optimal_action_set,
)
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition


class FixedActionAgent:
    def __init__(self, action: int):
        self.action = action

    def select_action(self, state: Observation, epsilon: float) -> int:
        return self.action


class DiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        path = astar_path((0, 0), (2, 2), set(), 3)
        assert path is not None
        self.problem = NavigationProblem(
            map_id="diagnostic",
            seed=1,
            size=3,
            start=(0, 0),
            goal=(2, 2),
            obstacles=frozenset(),
            nominal_path=tuple(path),
        )

    def observation(self, position: tuple[int, int]) -> Observation:
        scale = self.problem.size - 1
        return Observation(
            spatial=np.zeros((1, 3, 3), dtype=np.float32),
            scalars=np.asarray(
                [
                    (self.problem.goal[0] - position[0]) / scale,
                    (self.problem.goal[1] - position[1]) / scale,
                ],
                dtype=np.float32,
            ),
        )

    def test_optimal_action_set_contains_both_equal_shortest_moves(self) -> None:
        actions = optimal_action_set((0, 0), self.problem)
        self.assertEqual(actions, {int(Action.DOWN), int(Action.RIGHT)})

    def test_diagnostics_accepts_an_alternate_shortest_action(self) -> None:
        state = self.observation((0, 0))
        transition = Transition(state, int(Action.RIGHT), 0.0, state, False)
        metrics = demonstration_action_diagnostics(
            FixedActionAgent(int(Action.DOWN)), [transition], self.problem  # type: ignore[arg-type]
        )
        self.assertEqual(metrics["demo_valid_state_count"], 1.0)
        self.assertEqual(metrics["demo_optimal_action_agreement"], 1.0)
        self.assertEqual(metrics["demo_exact_action_agreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
