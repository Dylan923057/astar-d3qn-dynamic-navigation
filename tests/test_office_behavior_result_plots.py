from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from plot_office_behavior_v6_results import (
    BEHAVIOR_LABELS,
    STRATEGIES,
    _plot_grouped_bars,
    _plot_scenario_heatmaps,
    _plot_validation_curves,
    _render_path_galleries,
    _scenario_summary,
    _seed_metric_rows,
)
from train_dynamic_spatial_generalization import (
    _paired_test_metrics,
    _serialize_trajectories,
)
from astar_d3qn.core.trajectory import WaitEvent
from astar_d3qn.evaluation.rollout import EvaluationTrajectory
from astar_d3qn.maps.problem import NavigationProblem


class OfficeBehaviorResultPlotTests(unittest.TestCase):
    def _runs(self):
        runs = {}
        behaviors = ("wait", "avoidance", "reroute")
        for strategy_index, strategy in enumerate(STRATEGIES):
            for seed in range(2):
                validation = []
                for step in (100, 200):
                    validation.append(
                        {
                            "environment_steps": str(step),
                            "safe_success_rate": str(0.5 + 0.05 * strategy_index),
                            "success_rate": "0.8",
                            "dynamic_collision_rate": "0.2",
                            "mean_dynamic_collision_count": "0.3",
                            "mean_wait_steps": "1.0",
                            "mean_steps": "20.0",
                        }
                    )
                test = []
                for offset, behavior in enumerate(behaviors):
                    test.append(
                        {
                            "scenario_id": str(20_000 + offset),
                            "required_behavior": behavior,
                            "safe_success": str(float(offset != 2)),
                            "success": "1.0",
                            "dynamic_collision": str(float(offset == 2)),
                            "dynamic_collision_count": str(float(offset == 2)),
                            "wait_steps": str(float(offset == 0)),
                            "steps": str(20 + offset),
                            "path_efficiency": "0.8",
                        }
                    )
                runs[(strategy, seed)] = {"validation": validation, "test": test}
        return runs

    def test_summary_and_core_figures_render(self) -> None:
        runs = self._runs()
        seeds = (0, 1)
        scenarios = {
            20_000 + index: {
                "scenario_id": 20_000 + index,
                "required_behavior": behavior,
            }
            for index, behavior in enumerate(("wait", "avoidance", "reroute"))
        }
        overall = _seed_metric_rows(runs, seeds)
        behavior = _seed_metric_rows(
            runs, seeds, behaviors=("wait", "avoidance", "reroute")
        )
        scenario = _scenario_summary(runs, seeds, scenarios)

        self.assertEqual(len(overall), 6)
        self.assertEqual(len(behavior), 18)
        self.assertEqual(len(scenario), 9)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _plot_validation_curves(runs, seeds, output / "validation.png")
            _plot_grouped_bars(
                behavior,
                output / "behavior.png",
                groups=("wait", "avoidance", "reroute"),
                group_labels=BEHAVIOR_LABELS,
                title="test",
            )
            _plot_scenario_heatmaps(
                scenario,
                scenarios,
                seeds,
                output / "heatmap.png",
            )
            self.assertTrue((output / "validation.png").is_file())
            self.assertTrue((output / "behavior.png").is_file())
            self.assertTrue((output / "heatmap.png").is_file())

    def test_path_gallery_retains_each_seed_and_strategy(self) -> None:
        scenarios = {
            20_000 + index: {
                "scenario_id": 20_000 + index,
                "required_behavior": behavior,
                "oracle_path": [[0, 0], [0, 1], [0, 2]],
                "obstacles": [{"route_id": "test_route", "start_index": 0}],
            }
            for index, behavior in enumerate(("wait", "avoidance", "reroute"))
        }
        runs = {}
        for strategy in STRATEGIES:
            for seed in range(2):
                runs[(strategy, seed)] = {
                    "trajectory_by_scenario": {
                        scenario_id: {
                            "success": 1.0,
                            "steps": 2,
                            "dynamic_collision_count": 0,
                            "wait_steps": 0,
                            "path": [[0, 0], [0, 1], [0, 2]],
                            "collision_positions": [],
                            "wait_events": [],
                        }
                        for scenario_id in scenarios
                    }
                }
        manifest = {
            "route_pools": {
                "test": {
                    "corridor": [
                        {
                            "route_id": "test_route",
                            "route": [[1, 0], [1, 1], [1, 2]],
                        }
                    ]
                }
            }
        }
        problem = NavigationProblem(
            map_id="plot_test",
            seed=0,
            size=3,
            start=(0, 0),
            goal=(0, 2),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            index = _render_path_galleries(
                runs,
                (0, 1),
                scenarios,
                manifest,
                problem,
                output,
            )
            self.assertEqual(len(index), 6)
            self.assertTrue((output / "paths" / "seed_0" / "test_paths_page_01.png").is_file())
            self.assertTrue((output / "paths" / "seed_1" / "test_paths_page_01.png").is_file())

    def test_test_trajectory_serialization_keeps_path_events(self) -> None:
        trajectory = EvaluationTrajectory(
            path=((0, 0), (0, 0), (0, 1)),
            collision_positions=((1, 1),),
            static_collision_positions=(),
            dynamic_collision_positions=((1, 1),),
            revisit_positions=(),
            wait_events=(WaitEvent((0, 0), 1, 1),),
        )
        details = [
            {
                "map_id": "map",
                "scenario_id": 20_000,
                "trajectory_key": "map::scenario_20000",
                "success": 1.0,
                "termination_reason": "goal",
                "steps": 2,
                "collision_count": 1,
                "static_collision_count": 0,
                "dynamic_collision_count": 1,
                "wait_steps": 1,
            }
        ]
        records = _serialize_trajectories(
            details,
            {"map::scenario_20000": trajectory},
        )
        self.assertEqual(records[0]["path"], [[0, 0], [0, 0], [0, 1]])
        self.assertEqual(records[0]["wait_events"][0]["duration"], 1)
        self.assertEqual(records[0]["dynamic_collision_positions"], [[1, 1]])

    def test_paired_metrics_require_wait_only_in_conflict_member(self) -> None:
        details = [
            {
                "scenario_id": 2,
                "safe_success": 1.0,
                "steps": 70,
                "wait_steps": 0,
            },
            {
                "scenario_id": 3,
                "safe_success": 1.0,
                "steps": 71,
                "wait_steps": 1,
            },
        ]
        sources = {
            2: {
                "scenario_id": 2,
                "pair_id": "test_pair_000",
                "pair_role": "matched_control",
                "difficulty_stratum": "easy",
                "minimum_safe_path_steps": 70,
            },
            3: {
                "scenario_id": 3,
                "pair_id": "test_pair_000",
                "pair_role": "conflict",
                "difficulty_stratum": "easy",
                "minimum_safe_path_steps": 71,
            },
        }
        summary, records = _paired_test_metrics(
            details,
            sources,
            {
                "enabled": True,
                "control_maximum_excess_steps": 0,
                "control_requires_no_wait": True,
                "conflict_requires_wait": True,
            },
        )
        self.assertEqual(summary["mean_paired_response_match"], 1.0)
        self.assertEqual(records[0]["control_excess_steps"], 0)
        self.assertEqual(records[0]["conflict_excess_steps"], 0)
        self.assertEqual(records[0]["wait_step_contrast"], 1)

    def test_paired_metrics_do_not_report_excess_for_failed_rollout(self) -> None:
        details = [
            {
                "scenario_id": 2,
                "safe_success": 0.0,
                "steps": 20,
                "wait_steps": 0,
            },
            {
                "scenario_id": 3,
                "safe_success": 0.0,
                "steps": 20,
                "wait_steps": 0,
            },
        ]
        sources = {
            2: {
                "scenario_id": 2,
                "pair_id": "test_pair_000",
                "pair_role": "matched_control",
                "difficulty_stratum": "easy",
                "minimum_safe_path_steps": 70,
            },
            3: {
                "scenario_id": 3,
                "pair_id": "test_pair_000",
                "pair_role": "conflict",
                "difficulty_stratum": "easy",
                "minimum_safe_path_steps": 71,
            },
        }
        summary, records = _paired_test_metrics(
            details,
            sources,
            {"enabled": True},
        )
        self.assertIsNone(records[0]["control_excess_steps"])
        self.assertIsNone(records[0]["conflict_excess_steps"])
        self.assertIsNone(summary["mean_control_excess_steps"])
        self.assertIsNone(summary["mean_conflict_excess_steps"])


if __name__ == "__main__":
    unittest.main()
