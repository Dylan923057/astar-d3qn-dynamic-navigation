from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.collisions import CollisionTracker
from astar_d3qn.core.grid import Action
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.problem import NavigationProblem


class RightAgent:
    def select_action(self, state, epsilon: float) -> int:
        return int(Action.RIGHT)


class StaticMaskRecordingAgent:
    def __init__(self) -> None:
        self.valid_actions: list[tuple[int, ...]] = []

    def select_action(
        self,
        state,
        epsilon: float,
        valid_actions=None,
    ) -> int:
        candidates = tuple(valid_actions or ())
        self.valid_actions.append(candidates)
        return int(Action.RIGHT)


class CollisionMetricTests(unittest.TestCase):
    def test_tracker_separates_collision_types_and_obstacle_indices(self) -> None:
        tracker = CollisionTracker()
        tracker.record({"collision": True, "collision_type": "static"}, step=2)
        tracker.record(
            {
                "collision": True,
                "collision_type": "dynamic",
                "dynamic_collision_indices": (0, 2),
            },
            step=5,
        )
        tracker.record(
            {
                "collision": True,
                "collision_type": "dynamic",
                "dynamic_collision_indices": (2,),
            },
            step=7,
        )

        metrics = tracker.metrics()
        self.assertEqual(tracker.total_count, 3)
        self.assertEqual(metrics["static_collision_count"], 1)
        self.assertEqual(metrics["dynamic_collision_count"], 2)
        self.assertEqual(metrics["first_static_collision_step"], 2)
        self.assertEqual(metrics["first_dynamic_collision_step"], 5)
        self.assertEqual(metrics["dynamic_obstacle_contact_count"], 3)
        self.assertEqual(metrics["unique_dynamic_obstacles_hit"], 2)
        self.assertEqual(metrics["dynamic_obstacle_indices_hit"], "0;2")

    def test_evaluation_reports_dynamic_collisions_separately(self) -> None:
        problem = NavigationProblem(
            map_id="dynamic_collision_test",
            seed=1,
            size=7,
            start=(3, 0),
            goal=(3, 6),
            obstacles=frozenset(),
            nominal_path=tuple((3, column) for column in range(7)),
        )
        spec = DynamicObstacleSpec(
            route=((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
            start_index=2,
            direction=1,
        )

        summary, rows, trajectories = evaluate_agent(
            RightAgent(),  # type: ignore[arg-type]
            [problem],
            max_steps=10,
            window_size=7,
            environment_factory=lambda selected, **kwargs: DynamicGridNavigationEnv(
                selected,
                (spec,),
                **kwargs,
            ),
        )

        row = rows[0]
        self.assertEqual(row["static_collision_count"], 0)
        self.assertEqual(row["dynamic_collision_count"], 1)
        self.assertEqual(row["first_dynamic_collision_step"], 1)
        self.assertEqual(row["dynamic_obstacle_count"], 1)
        self.assertEqual(row["unique_dynamic_obstacles_hit"], 1)
        self.assertEqual(row["safe_success"], 0.0)
        self.assertEqual(row["collision_success"], 1.0)
        self.assertEqual(row["dynamic_collision_success"], 1.0)
        self.assertEqual(row["dynamic_collision_free_success"], 0.0)
        self.assertEqual(summary["static_collision_rate"], 0.0)
        self.assertEqual(summary["dynamic_collision_rate"], 1.0)
        self.assertEqual(summary["mean_dynamic_collision_count"], 1.0)
        self.assertEqual(summary["safe_success_rate"], 0.0)
        self.assertEqual(summary["collision_success_rate"], 1.0)
        self.assertEqual(summary["dynamic_collision_success_rate"], 1.0)
        self.assertEqual(len(trajectories[problem.map_id].dynamic_collision_positions), 1)
        self.assertEqual(trajectories[problem.map_id].static_collision_positions, ())

    def test_evaluation_passes_static_safe_actions_to_agent(self) -> None:
        problem = NavigationProblem(
            map_id="static_mask_test",
            seed=1,
            size=3,
            start=(0, 0),
            goal=(0, 1),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1)),
        )
        agent = StaticMaskRecordingAgent()

        summary, _, _ = evaluate_agent(
            agent,  # type: ignore[arg-type]
            [problem],
            max_steps=3,
            window_size=3,
            mask_static_invalid_actions=True,
        )

        first_actions = agent.valid_actions[0]
        self.assertNotIn(int(Action.UP), first_actions)
        self.assertNotIn(int(Action.LEFT), first_actions)
        self.assertIn(int(Action.RIGHT), first_actions)
        self.assertEqual(summary["safe_success_rate"], 1.0)

    def test_evaluation_preserves_scenario_behavior_labels(self) -> None:
        problem = NavigationProblem(
            map_id="behavior_label_test",
            seed=1,
            size=3,
            start=(0, 0),
            goal=(0, 1),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1)),
        )

        def factory(selected, **kwargs):
            env = DynamicGridNavigationEnv(selected, (), **kwargs)
            env.required_behavior = "wait"
            env.difficulty_stratum = "controlled_wait"
            return env

        _, rows, _ = evaluate_agent(
            RightAgent(),  # type: ignore[arg-type]
            [problem],
            max_steps=3,
            window_size=3,
            environment_factory=factory,
        )

        self.assertEqual(rows[0]["required_behavior"], "wait")
        self.assertEqual(rows[0]["difficulty_stratum"], "controlled_wait")


if __name__ == "__main__":
    unittest.main()
