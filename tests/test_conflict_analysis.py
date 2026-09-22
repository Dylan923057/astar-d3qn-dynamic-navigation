from __future__ import annotations

import unittest

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.evaluation.conflict import (
    obstacle_conflict_metrics,
    obstacle_positions,
    minimum_collision_free_steps,
    scenario_conflict_metrics,
    scenario_safe_path_metrics,
)
from astar_d3qn.maps.problem import NavigationProblem


def _problem() -> NavigationProblem:
    return NavigationProblem(
        map_id="conflict_test",
        seed=1,
        size=7,
        start=(3, 0),
        goal=(3, 6),
        obstacles=frozenset(),
        nominal_path=tuple((3, column) for column in range(7)),
    )


class ConflictAnalysisTests(unittest.TestCase):
    def test_bouncing_obstacle_positions_match_environment_timing(self) -> None:
        spec = DynamicObstacleSpec(
            route=((1, 3), (2, 3), (3, 3), (4, 3), (5, 3)),
            start_index=0,
            direction=1,
            move_every=1,
        )
        self.assertEqual(
            obstacle_positions(spec, 8),
            (
                (1, 3),
                (2, 3),
                (3, 3),
                (4, 3),
                (5, 3),
                (4, 3),
                (3, 3),
                (2, 3),
                (1, 3),
            ),
        )

    def test_exact_nominal_conflict_is_detected(self) -> None:
        spec = DynamicObstacleSpec(
            route=((1, 3), (2, 3), (3, 3), (4, 3), (5, 3)),
            start_index=0,
            direction=1,
            move_every=1,
        )
        metrics = obstacle_conflict_metrics(_problem(), spec)
        self.assertEqual(metrics["direct_intersection"], 1)
        self.assertEqual(metrics["exact_temporal_conflict_steps"], [3])
        self.assertGreaterEqual(metrics["aligned_temporal_conflict_count"], 1)

    def test_off_path_obstacle_has_no_direct_or_temporal_conflict(self) -> None:
        spec = DynamicObstacleSpec(
            route=((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
            start_index=0,
            direction=1,
            move_every=1,
        )
        scenario = DynamicScenario(seed=99, obstacles=(spec,))
        metrics = scenario_conflict_metrics(_problem(), scenario)
        self.assertEqual(metrics["direct_intersection_route_count"], 0)
        self.assertEqual(metrics["exact_temporal_conflict_count"], 0)
        self.assertEqual(metrics["aligned_temporal_conflict_count"], 0)

    def test_time_expanded_safe_path_oracle_supports_wait_action(self) -> None:
        spec = DynamicObstacleSpec(
            route=((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
            start_index=0,
            direction=1,
            move_every=1,
        )
        scenario = DynamicScenario(seed=100, obstacles=(spec,))
        self.assertEqual(minimum_collision_free_steps(_problem(), scenario), 6)
        self.assertEqual(
            scenario_safe_path_metrics(_problem(), scenario)["safe_detour_steps"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
