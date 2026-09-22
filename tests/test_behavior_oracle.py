from __future__ import annotations

import unittest
from dataclasses import replace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.evaluation.behavior_oracle import shortest_safe_plan
from astar_d3qn.evaluation.conflict import obstacle_positions
from astar_d3qn.core.grid import chebyshev
from astar_d3qn.maps.problem import NavigationProblem


class BehaviorOracleTests(unittest.TestCase):
    def _visible_case(self):
        return DynamicScenario(seed=0, obstacles=(DynamicObstacleSpec(
            route=((1, 2), (2, 2)), start_index=0, direction=1, move_every=1,
        ),))

    def test_an_equal_cost_visible_departure_can_exist_after_invisible_tie(self):
        problem, scenario = self._problem(), self._visible_case()
        original = shortest_safe_plan(problem, scenario, allow_wait=False)
        visible = shortest_safe_plan(problem, scenario, allow_wait=False,
                                     visible_deviation_obstacle_index=0, observation_radius=1)
        self.assertIsNotNone(original)
        self.assertIsNotNone(visible)
        self.assertEqual(original.steps, 6)
        self.assertEqual(visible.steps, original.steps)
        for plan, expected_visible in ((original, False), (visible, True)):
            step = next(t for t in range(1, len(problem.nominal_path))
                        if plan.positions[t] != problem.nominal_path[t])
            cell = obstacle_positions(scenario.obstacles[0], step - 1)[step - 1]
            self.assertEqual(chebyshev(plan.positions[step - 1], cell) <= 1, expected_visible)
        self.assertEqual(visible.positions[:2], ((2, 0), (2, 1)))

    def test_visible_departure_can_require_extra_steps(self):
        problem = replace(self._problem(), obstacles=frozenset(((1, 1), (3, 1))))
        scenario = self._visible_case()
        original = shortest_safe_plan(problem, scenario, allow_wait=False)
        visible = shortest_safe_plan(problem, scenario, allow_wait=False,
                                     visible_deviation_obstacle_index=0, observation_radius=1)
        self.assertIsNotNone(original)
        self.assertIsNotNone(visible)
        self.assertEqual(original.steps, 8)
        self.assertEqual(visible.steps, 10)
        # Returning to the same spatial/time state after departing must not be
        # merged with its before-departure state, or this valid path is lost.
        self.assertEqual(visible.positions[:3], ((2, 0), (2, 1), (2, 0)))

    def test_no_visible_departure_path_is_reported_without_relaxing_radius(self):
        plan = shortest_safe_plan(self._problem(), self._visible_case(), allow_wait=False,
                                  visible_deviation_obstacle_index=0, observation_radius=0)
        self.assertIsNone(plan)

    def test_visibility_uses_current_frame_not_future_obstacle_position(self):
        scenario = DynamicScenario(seed=0, obstacles=(DynamicObstacleSpec(
            route=((1, 2), (1, 1)), start_index=0, direction=1,
        ),))
        # At t=0 the primary is outside radius 1, but after its next move it
        # will be inside. Future visibility must not authorize this action.
        plan = shortest_safe_plan(self._problem(), scenario, allow_wait=False,
                                  additionally_blocked={(2, 1)},
                                  visible_deviation_obstacle_index=0, observation_radius=1)
        self.assertIsNone(plan)

    def test_default_and_explicit_disabled_constraint_are_identical(self):
        original = shortest_safe_plan(self._problem(), self._visible_case())
        disabled = shortest_safe_plan(self._problem(), self._visible_case(),
                                      visible_deviation_obstacle_index=None)
        self.assertEqual(original, disabled)

    def test_invalid_visible_obstacle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "index"):
            shortest_safe_plan(self._problem(), self._visible_case(), visible_deviation_obstacle_index=1)

    def _problem(self):
        return NavigationProblem(
            map_id="test",
            size=5,
            start=(2, 0),
            goal=(2, 4),
            obstacles=frozenset(),
            nominal_path=((2, 0), (2, 1), (2, 2), (2, 3), (2, 4)),
            seed=0,
            metadata={},
        )

    def test_static_shortest_plan_uses_no_wait(self):
        plan = shortest_safe_plan(self._problem(), DynamicScenario(seed=0, obstacles=()))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps, 4)
        self.assertEqual(plan.wait_count, 0)

    def test_additional_blocking_forces_detour(self):
        plan = shortest_safe_plan(
            self._problem(),
            DynamicScenario(seed=0, obstacles=()),
            additionally_blocked={(2, 2)},
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps, 6)

    def test_dynamic_collision_cells_are_avoided(self):
        obstacle = DynamicObstacleSpec(
            route=((1, 2), (2, 2), (3, 2)),
            start_index=0,
            direction=1,
            move_every=1,
        )
        plan = shortest_safe_plan(
            self._problem(), DynamicScenario(seed=0, obstacles=(obstacle,))
        )
        self.assertIsNotNone(plan)
        self.assertNotEqual(plan.positions[2], (2, 2))

    def test_replans_from_current_position_and_dynamic_phase(self):
        obstacle = DynamicObstacleSpec(
            route=((1, 2), (2, 2), (3, 2)),
            start_index=0,
            direction=1,
            move_every=1,
        )
        plan = shortest_safe_plan(
            self._problem(),
            DynamicScenario(seed=0, obstacles=(obstacle,)),
            start_position=(2, 1),
            start_time=1,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.positions[0], (2, 1))
        self.assertNotEqual(plan.positions[1], (2, 2))


if __name__ == "__main__":
    unittest.main()
