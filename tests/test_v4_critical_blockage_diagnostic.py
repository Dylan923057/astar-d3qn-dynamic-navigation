from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.grid import action_between
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.evaluation.conflict import minimum_collision_free_steps
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.io import load_json


class V4CriticalBlockageDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = ROOT / "outputs" / "v4_critical_blockage_diagnostic"
        cls.manifest = load_json(cls.output / "critical_blockage_scenarios.json")
        cls.problems = {
            problem.map_id: problem
            for problem in load_problem_set(ROOT / "maps" / "structured_main" / "maps.json")
        }

    def test_scenarios_are_safe_but_the_registered_astar_action_is_blocked(self) -> None:
        self.assertTrue(self.manifest["models_are_not_retrained"])
        self.assertEqual(len(self.manifest["maps"]), 3)
        for map_record in self.manifest["maps"].values():
            problem = self.problems[map_record["map_id"]]
            scenarios = map_record["scenarios"]
            self.assertEqual(len(scenarios), 6)
            self.assertEqual(
                {(row["region"], row["direction_variant"]) for row in scenarios},
                {(region, direction) for region in ("early", "middle", "late") for direction in (0, 1)},
            )
            for row in scenarios:
                specs = tuple(
                    DynamicObstacleSpec(
                        route=tuple(tuple(cell) for cell in source["route"]),
                        start_index=int(source["start_index"]),
                        direction=int(source["direction"]),
                        move_every=int(source["move_every"]),
                        label=str(source["label"]),
                    )
                    for source in row["obstacles"]
                )
                self.assertEqual(len(specs), 5)
                self.assertFalse(
                    any(set(spec.route).intersection(problem.nominal_path) for spec in specs[1:])
                )
                scenario = DynamicScenario(int(row["scenario_id"]), specs)
                self.assertIsNotNone(minimum_collision_free_steps(problem, scenario))
                env = DynamicGridNavigationEnv(problem, specs)
                env.reset()
                target_index = int(row["target_path_index"])
                for step in range(1, target_index):
                    result = env.step(
                        int(action_between(problem.nominal_path[step - 1], problem.nominal_path[step]))
                    )
                    self.assertFalse(result.info["collision"])
                result = env.step(
                    int(action_between(problem.nominal_path[target_index - 1], problem.nominal_path[target_index]))
                )
                self.assertEqual(result.info["collision_type"], "dynamic")
                self.assertIn(0, result.info["dynamic_collision_indices"])

    def test_all_frozen_models_have_probe_and_rollout_rows(self) -> None:
        expected = 3 * 5 * 5 * 6
        for filename in ("critical_state_action_probes.csv", "critical_blockage_rollouts.csv"):
            with (self.output / filename).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), expected)
        with (self.output / "critical_blockage_summary.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            summary = list(csv.DictReader(handle))
        self.assertEqual(len(summary), 15)
        self.assertTrue(all(int(row["seed_count"]) == 5 for row in summary))


if __name__ == "__main__":
    unittest.main()
