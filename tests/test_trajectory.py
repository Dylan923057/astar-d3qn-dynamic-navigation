from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.grid import Action
from astar_d3qn.core.trajectory import TrajectoryRecorder
from astar_d3qn.evaluation.rollout import evaluate_agent, path_turn_metrics
from astar_d3qn.maps.problem import NavigationProblem


class StayAgent:
    def select_action(self, state, epsilon: float) -> int:
        return int(Action.STAY)


class TrajectoryRecorderTests(unittest.TestCase):
    def test_path_turn_metrics_ignore_waits_and_count_direction_changes(self) -> None:
        movement_steps, turns, rate = path_turn_metrics(
            ((0, 0), (0, 1), (0, 1), (1, 1), (1, 2), (1, 3))
        )
        self.assertEqual(movement_steps, 4)
        self.assertEqual(turns, 2)
        self.assertAlmostEqual(rate, 2 / 3)

    def test_waits_do_not_count_as_revisits(self) -> None:
        recorder = TrajectoryRecorder((0, 0), (2, 2))
        recorder.record(int(Action.RIGHT), (0, 1))
        recorder.record(int(Action.STAY), (0, 1))
        recorder.record(int(Action.STAY), (0, 1))
        recorder.finish()

        self.assertEqual(recorder.wait_steps, 2)
        self.assertEqual(recorder.wait_event_count, 1)
        self.assertEqual(recorder.max_wait_streak, 2)
        self.assertEqual(recorder.revisit_positions, [])
        self.assertEqual(recorder.wait_events[0].position, (0, 1))
        self.assertEqual(recorder.wait_events[0].start_step, 2)

    def test_return_after_leaving_counts_as_one_revisit(self) -> None:
        recorder = TrajectoryRecorder((0, 0), (2, 2))
        recorder.record(int(Action.RIGHT), (0, 1))
        recorder.record(int(Action.LEFT), (0, 0))
        recorder.record(int(Action.RIGHT), (0, 1))
        recorder.record(int(Action.LEFT), (0, 0))
        recorder.record(int(Action.RIGHT), (0, 1))
        recorder.finish()

        self.assertEqual(recorder.revisit_positions, [(0, 1)])

    def test_collision_hold_is_not_a_wait_or_revisit(self) -> None:
        recorder = TrajectoryRecorder((0, 0), (2, 2))
        recorder.record(
            int(Action.RIGHT),
            (0, 0),
            collision_position=(0, 1),
        )
        recorder.finish()

        self.assertEqual(recorder.collision_positions, [(0, 1)])
        self.assertEqual(recorder.wait_steps, 0)
        self.assertEqual(recorder.revisit_positions, [])

    def test_evaluation_reports_waits_without_false_revisits(self) -> None:
        problem = NavigationProblem(
            map_id="wait_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )

        summary, rows, trajectories = evaluate_agent(
            StayAgent(),  # type: ignore[arg-type]
            [problem],
            max_steps=3,
            window_size=5,
        )

        self.assertEqual(rows[0]["wait_steps"], 3)
        self.assertEqual(rows[0]["wait_event_count"], 1)
        self.assertEqual(rows[0]["max_wait_streak"], 3)
        self.assertEqual(rows[0]["revisit_count"], 0)
        self.assertEqual(rows[0]["movement_steps"], 0)
        self.assertEqual(rows[0]["turn_count"], 0)
        self.assertEqual(rows[0]["turn_rate"], 0.0)
        self.assertEqual(summary["mean_wait_steps"], 3.0)
        self.assertEqual(trajectories[problem.map_id].wait_events[0].duration, 3)
        self.assertGreaterEqual(summary["mean_action_latency_ms"], 0.0)

    def test_repeated_map_evaluations_keep_every_trajectory(self) -> None:
        problem = NavigationProblem(
            map_id="repeated_map",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )

        _, rows, trajectories = evaluate_agent(
            StayAgent(),  # type: ignore[arg-type]
            [problem, problem],
            max_steps=1,
            window_size=5,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(
            [row["trajectory_key"] for row in rows],
            [
                "repeated_map::evaluation_00000",
                "repeated_map::evaluation_00001",
            ],
        )


if __name__ == "__main__":
    unittest.main()
