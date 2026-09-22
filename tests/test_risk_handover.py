from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astar_d3qn.maps.adaptation import collision_step, spec_from_record
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.maps.risk_handover import replay_multi_obstacle_path


class FrozenRiskHandoverDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(
            (ROOT / "data/risk_handover_v1/manifest.json").read_text(encoding="utf-8")
        )

    def test_density_quotas_and_all_oracles_replay(self):
        expected = {
            "train": {3: 18, 5: 18},
            "validation": {3: 6, 5: 6},
            "test": {1: 12, 3: 12, 5: 12, 7: 12},
        }
        for entry in self.payload["maps"]:
            problem = problem_from_record(entry["problem"])
            for split, pairs in entry["scenarios"]["splits"].items():
                self.assertEqual(Counter(pair["obstacle_count"] for pair in pairs), expected[split])
                for pair in pairs:
                    for condition in ("control", "conflict"):
                        record = pair[condition]
                        specs = [spec_from_record(value) for value in record["obstacles"]]
                        self.assertEqual(len(specs), pair["obstacle_count"])
                        self.assertTrue(
                            replay_multi_obstacle_path(problem, specs, record["oracle_path"])
                        )
                        route_cells = [set(spec.route) for spec in specs]
                        self.assertTrue(
                            all(not left & right for index, left in enumerate(route_cells)
                                for right in route_cells[index + 1:])
                        )

    def test_control_reference_is_safe_and_conflict_has_causal_blockers(self):
        for entry in self.payload["maps"]:
            problem = problem_from_record(entry["problem"])
            for pairs in entry["scenarios"]["splits"].values():
                for pair in pairs:
                    controls = [spec_from_record(value) for value in pair["control"]["obstacles"]]
                    conflicts = [spec_from_record(value) for value in pair["conflict"]["obstacles"]]
                    self.assertTrue(all(collision_step(problem.nominal_path, spec) is None
                                        for spec in controls))
                    causal = pair["causal_obstacle_count"]
                    self.assertTrue(all(collision_step(problem.nominal_path, spec) is not None
                                        for spec in conflicts[:causal]))


if __name__ == "__main__":
    unittest.main()
