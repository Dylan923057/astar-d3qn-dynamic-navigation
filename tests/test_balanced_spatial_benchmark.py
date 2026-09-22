from __future__ import annotations

import sys
import unittest
from collections import Counter
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
from astar_d3qn.evaluation.conflict import (
    route_record_lookup,
    scenario_conflict_metrics,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.io import load_json


CASES = (
    ("office_40x40", "office_40x40_balanced_v3.json", 20260911),
    ("parcel_station_40x40", "parcel_station_40x40_balanced_v3.json", 20260921),
    ("warehouse_40x40", "warehouse_40x40_balanced_v3.json", 20260931),
)
EXPECTED_COUNTS = {"train": 100, "validation": 20, "test": 50}
EXPECTED_POOL_COUNTS = {
    "train": {"corridor": 9, "background": 7},
    "validation": {"corridor": 5, "background": 4},
    "test": {"corridor": 5, "background": 6},
}
EXPECTED_STRATA = {
    "train": {"low": 40, "medium": 30, "high": 30},
    "validation": {"low": 8, "medium": 6, "high": 6},
    "test": {"low": 20, "medium": 15, "high": 15},
}


class BalancedSpatialBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        problems = load_problem_set(ROOT / "maps" / "structured_main" / "maps.json")
        cls.problems = {problem.map_id: problem for problem in problems}

    def _cases(self):
        for map_id, filename, seed in CASES:
            yield (
                self.problems[map_id],
                load_json(ROOT / "data" / "dynamic_scenarios" / filename),
                seed,
            )

    def test_manifests_are_frozen_valid_and_spatially_disjoint(self) -> None:
        for problem, manifest, seed in self._cases():
            with self.subTest(map_id=problem.map_id):
                validate_spatial_scenario_manifest(problem, manifest)
                self.assertEqual(manifest["map_id"], problem.map_id)
                self.assertEqual(manifest["generation"]["seed"], seed)
                self.assertEqual(
                    manifest["generation"]["protocol"],
                    "conflict_balanced_spatial_v3",
                )
                self.assertEqual(
                    {
                        split: len(manifest["scenarios"][split])
                        for split in SPLITS
                    },
                    EXPECTED_COUNTS,
                )
                self.assertEqual(
                    {
                        split: {
                            category: len(manifest["route_pools"][split][category])
                            for category in ("corridor", "background")
                        }
                        for split in SPLITS
                    },
                    EXPECTED_POOL_COUNTS,
                )

                cells = {
                    split: {
                        tuple(cell)
                        for category in ("corridor", "background")
                        for route in manifest["route_pools"][split][category]
                        for cell in route["route"]
                    }
                    for split in SPLITS
                }
                for left_index, left in enumerate(SPLITS):
                    for right in SPLITS[left_index + 1 :]:
                        minimum_distance = min(
                            max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                            for a in cells[left]
                            for b in cells[right]
                        )
                        self.assertGreaterEqual(minimum_distance, 2)

    def test_strata_have_matched_primary_conflict_controls(self) -> None:
        expected_direct = {"low": 0, "medium": 1, "high": 2}
        for problem, manifest, _ in self._cases():
            per_split_exact = {}
            for split in SPLITS:
                raw_scenarios = manifest["scenarios"][split]
                self.assertEqual(
                    Counter(item["difficulty_stratum"] for item in raw_scenarios),
                    Counter(EXPECTED_STRATA[split]),
                )
                materialized = scenarios_from_spatial_manifest(problem, manifest, split)
                records = route_record_lookup(manifest, split)
                exact_by_stratum = {"low": [], "medium": [], "high": []}
                combination_counts = Counter()
                phase_signatures = set()

                for raw, scenario in zip(raw_scenarios, materialized, strict=True):
                    stratum = raw["difficulty_stratum"]
                    metrics = scenario_conflict_metrics(
                        problem,
                        scenario,
                        temporal_window=2,
                        route_records=records,
                    )
                    self.assertEqual(
                        metrics["direct_intersection_route_count"],
                        expected_direct[stratum],
                    )
                    if stratum == "low":
                        self.assertEqual(metrics["exact_temporal_conflict_count"], 0)
                        self.assertEqual(metrics["aligned_temporal_conflict_count"], 0)
                    elif stratum == "medium":
                        self.assertEqual(metrics["exact_temporal_conflict_count"], 0)
                        self.assertEqual(metrics["aligned_temporal_conflict_count"], 1)
                    else:
                        self.assertGreater(metrics["exact_temporal_conflict_count"], 0)
                    exact_by_stratum[stratum].append(
                        metrics["exact_temporal_conflict_count"]
                    )

                    route_ids = tuple(
                        sorted(item["route_id"] for item in raw["obstacles"])
                    )
                    combination_counts[route_ids] += 1
                    phase_signature = tuple(
                        sorted(
                            (
                                item["route_id"],
                                item["start_index"],
                                item["direction"],
                            )
                            for item in raw["obstacles"]
                        )
                    )
                    self.assertNotIn(phase_signature, phase_signatures)
                    phase_signatures.add(phase_signature)

                self.assertLessEqual(max(combination_counts.values()), 2)
                per_split_exact[split] = {
                    stratum: sum(values) / len(values)
                    for stratum, values in exact_by_stratum.items()
                }

            for stratum in ("low", "medium", "high"):
                values = [per_split_exact[split][stratum] for split in SPLITS]
                self.assertAlmostEqual(max(values), min(values))


if __name__ == "__main__":
    unittest.main()
