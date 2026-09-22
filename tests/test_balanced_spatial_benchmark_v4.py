from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.spatial_scenarios import SPLITS, validate_spatial_scenario_manifest
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.io import load_json


CASES = (
    ("office_40x40", "office_40x40_balanced_v4.json", 20260909),
    ("parcel_station_40x40", "parcel_station_40x40_balanced_v4.json", 20260915),
    ("warehouse_40x40", "warehouse_40x40_balanced_v4.json", 20260948),
)
EXPECTED_MEAN_DETOUR = {
    "office_40x40": {"train": 0.0, "validation": 0.0, "test": 0.0},
    "parcel_station_40x40": {"train": 1.8, "validation": 2.0, "test": 2.0},
    "warehouse_40x40": {"train": 1.033333, "validation": 1.0, "test": 1.066667},
}


class BalancedSpatialBenchmarkV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problems = {
            p.map_id: p
            for p in load_problem_set(ROOT / "maps" / "structured_main" / "maps.json")
        }

    def test_v4_manifests_are_valid_and_oracle_difficulty_is_balanced(self) -> None:
        audit_path = ROOT / "outputs" / "spatial_generalization_balanced_v4_design" / "scenario_spatiotemporal_difficulty.csv"
        with audit_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 510)
        for map_id, filename, seed in CASES:
            manifest = load_json(ROOT / "data" / "dynamic_scenarios" / filename)
            validate_spatial_scenario_manifest(self.problems[map_id], manifest)
            self.assertEqual(manifest["generation"]["protocol"], "conflict_balanced_spatial_v4")
            self.assertEqual(manifest["generation"]["seed"], seed)
            map_rows = [row for row in rows if row["map_id"] == map_id]
            for split in SPLITS:
                split_rows = [row for row in map_rows if row["split"] == split]
                for stratum in ("low", "medium", "high"):
                    group = [row for row in split_rows if row["difficulty_stratum"] == stratum]
                    self.assertEqual(len(group), {"train": {"low": 40, "medium": 30, "high": 30}, "validation": {"low": 8, "medium": 6, "high": 6}, "test": {"low": 20, "medium": 15, "high": 15}}[split][stratum])
                    self.assertTrue(all(row["minimum_safe_path_steps"] for row in group))
                high = [float(row["safe_detour_steps"]) for row in split_rows if row["difficulty_stratum"] == "high"]
                self.assertAlmostEqual(sum(high) / len(high), EXPECTED_MEAN_DETOUR[map_id][split], places=5)


if __name__ == "__main__":
    unittest.main()
