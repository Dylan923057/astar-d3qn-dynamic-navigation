from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from astar_d3qn.envs.spatial_scenarios import SPLITS, validate_spatial_scenario_manifest
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from generate_causal_paired_benchmark import build_causal_paired_manifest


class CausalPairedBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(
            ROOT / "configs" / "dynamic_spatial_generalization_office_causal_paired_v1.yaml"
        )
        cls.problem = next(
            problem
            for problem in load_problem_set(
                ROOT / cls.config["map_sets"]["train"]["file"]
            )
            if problem.map_id == cls.config["map"]["scene"]
        )
        cls.manifest, cls.audit = build_causal_paired_manifest(
            cls.problem, cls.config
        )

    def test_manifest_is_valid_and_counts_match_config(self) -> None:
        validate_spatial_scenario_manifest(self.problem, self.manifest)
        spatial = self.config["spatial_generalization"]
        for split in SPLITS:
            self.assertEqual(
                len(self.manifest["scenarios"][split]),
                int(spatial["scenario_counts"][split]),
            )
            self.assertTrue(
                all(len(item["obstacles"]) == 1 for item in self.manifest["scenarios"][split])
            )

    def test_pairs_only_change_start_phase_and_are_solvable(self) -> None:
        for split in SPLITS:
            groups = defaultdict(list)
            for scenario in self.manifest["scenarios"][split]:
                groups[scenario["pair_id"]].append(scenario)
            for members in groups.values():
                self.assertEqual(len(members), 2)
                by_condition = {item["pair_condition"]: item for item in members}
                self.assertEqual(set(by_condition), {"control", "conflict"})
                control = by_condition["control"]
                conflict = by_condition["conflict"]
                control_obstacle = control["obstacles"][0]
                conflict_obstacle = conflict["obstacles"][0]
                for key in ("route_id", "direction", "move_every"):
                    self.assertEqual(control_obstacle[key], conflict_obstacle[key])
                self.assertNotEqual(
                    control_obstacle["start_index"], conflict_obstacle["start_index"]
                )
                self.assertEqual(control["exact_temporal_conflict_count"], 0)
                self.assertGreater(conflict["exact_temporal_conflict_count"], 0)
                self.assertIsNotNone(control["minimum_safe_path_steps"])
                self.assertIsNotNone(conflict["minimum_safe_path_steps"])


if __name__ == "__main__":
    unittest.main()

