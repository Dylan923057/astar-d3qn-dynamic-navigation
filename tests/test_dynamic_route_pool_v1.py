from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from astar_d3qn.maps.io import problem_from_record
from design_dynamic_route_pool_v1 import validate_map_entry


class DynamicRoutePoolV1Tests(unittest.TestCase):
    def test_frozen_design_keeps_maps_and_validates_five_obstacle_examples(self):
        source = json.loads((ROOT / "data/risk_handover_v1/manifest.json").read_text(encoding="utf-8"))
        design = json.loads((ROOT / "data/dynamic_route_pool_v1/manifest.json").read_text(encoding="utf-8"))
        source_lookup = {entry["problem"]["map_id"]: entry for entry in source["maps"]}
        self.assertTrue(design["design_only"])
        self.assertFalse(design["training_started"])
        for entry in design["maps"]:
            reference = entry["problem_reference"]
            problem = problem_from_record(source_lookup[reference["map_id"]]["problem"])
            self.assertEqual(reference["grid_sha256"], problem.grid_sha256)
            self.assertEqual(tuple(reference["start"]), problem.start)
            self.assertEqual(tuple(reference["goal"]), problem.goal)
            audit = validate_map_entry(
                problem,
                entry,
                {"high_interaction": 20, "alternative_branch": 24, "background": 12},
                {"high_interaction": 2, "alternative_branch": 2, "background": 1},
                0.55,
            )
            self.assertEqual(audit["minimum_interactive_routes_per_scene"], 5)
            self.assertGreaterEqual(audit["minimum_ensemble_path_coverage_fraction"], 0.55)
            self.assertGreater(entry["independence_audit"]["routes_not_intersecting_nominal_astar"], 0)


if __name__ == "__main__":
    unittest.main()
