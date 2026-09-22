from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.spatial_scenarios import (
    SPLITS,
    scenarios_from_spatial_manifest,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.io import load_json


class SpatialScenarioManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = load_problem_set(
            ROOT / "maps" / "structured_calibration_40x40" / "maps.json"
        )[0]
        cls.manifest = load_json(
            ROOT
            / "data"
            / "dynamic_scenarios"
            / "map01_spatial_generalization_manifest.json"
        )

    def test_frozen_manifest_is_valid_and_has_preregistered_counts(self) -> None:
        validate_spatial_scenario_manifest(self.problem, self.manifest)

        self.assertEqual(
            {split: len(self.manifest["scenarios"][split]) for split in SPLITS},
            {"train": 100, "validation": 20, "test": 50},
        )
        self.assertEqual(
            {
                split: {
                    category: len(self.manifest["route_pools"][split][category])
                    for category in ("corridor", "background")
                }
                for split in SPLITS
            },
            {
                "train": {"corridor": 25, "background": 7},
                "validation": {"corridor": 10, "background": 4},
                "test": {"corridor": 15, "background": 6},
            },
        )

    def test_splits_have_a_one_cell_spatial_buffer(self) -> None:
        cells = {}
        for split in SPLITS:
            cells[split] = {
                tuple(cell)
                for category in ("corridor", "background")
                for route in self.manifest["route_pools"][split][category]
                for cell in route["route"]
            }

        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                minimum_distance = min(
                    max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                    for a in cells[left]
                    for b in cells[right]
                )
                self.assertGreaterEqual(minimum_distance, 2)

    def test_each_scenario_materializes_three_corridor_and_two_background_routes(self) -> None:
        for split in SPLITS:
            scenarios = scenarios_from_spatial_manifest(
                self.problem,
                self.manifest,
                split,
            )
            self.assertEqual(len(scenarios), len(self.manifest["scenarios"][split]))
            for scenario in scenarios:
                sources = [item.reference_path_source for item in scenario.obstacles]
                self.assertEqual(len(sources), 5)
                self.assertEqual(
                    sources.count("randomized_shortest_path_corridor"),
                    3,
                )
                self.assertEqual(sources.count("off_corridor_open_space"), 2)

    def test_invalid_obstacle_phase_is_rejected(self) -> None:
        damaged = copy.deepcopy(self.manifest)
        damaged["scenarios"]["validation"][0]["obstacles"][0]["direction"] = 0

        with self.assertRaisesRegex(ValueError, "direction"):
            validate_spatial_scenario_manifest(self.problem, damaged)

    def test_map02_confirmatory_manifest_is_frozen_and_valid(self) -> None:
        problem = load_problem_set(
            ROOT / "maps" / "structured_calibration_40x40" / "maps.json"
        )[1]
        manifest = load_json(
            ROOT
            / "data"
            / "dynamic_scenarios"
            / "map02_spatial_generalization_confirmatory_v2.json"
        )

        validate_spatial_scenario_manifest(problem, manifest)
        self.assertEqual(manifest["map_id"], "calibration_40x40_map_02")
        self.assertEqual(manifest["generation"]["seed"], 20260828)
        self.assertEqual(
            {split: len(manifest["scenarios"][split]) for split in SPLITS},
            {"train": 100, "validation": 20, "test": 50},
        )
        self.assertEqual(
            {
                split: {
                    category: len(manifest["route_pools"][split][category])
                    for category in ("corridor", "background")
                }
                for split in SPLITS
            },
            {
                "train": {"corridor": 25, "background": 7},
                "validation": {"corridor": 10, "background": 4},
                "test": {"corridor": 15, "background": 6},
            },
        )


if __name__ == "__main__":
    unittest.main()
