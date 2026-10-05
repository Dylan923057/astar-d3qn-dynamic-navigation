from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from astar_d3qn.maps.io import problem_from_record
from design_dynamic_route_pool_v2 import validate_map_entry


class DynamicRoutePoolV2Tests(unittest.TestCase):
    def test_frozen_dual_density_design_and_offline_audits(self):
        config = yaml.safe_load(
            (ROOT / "configs/dynamic_route_pool_v2.yaml").read_text(encoding="utf-8")
        )
        source = json.loads(
            (ROOT / config["source_manifest"]).read_text(encoding="utf-8")
        )
        payload = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
        source_lookup = {entry["problem"]["map_id"]: entry for entry in source["maps"]}
        self.assertTrue(payload["design_only"])
        self.assertFalse(payload["training_started"])
        self.assertFalse(payload["adaptive_modified"])
        for entry in payload["maps"]:
            reference = entry["problem_reference"]
            problem = problem_from_record(source_lookup[reference["map_id"]]["problem"])
            self.assertEqual(reference["grid_sha256"], problem.grid_sha256)
            self.assertEqual(tuple(reference["start"]), problem.start)
            self.assertEqual(tuple(reference["goal"]), problem.goal)
            audit = validate_map_entry(
                problem,
                entry,
                config["route_pool_counts"],
                config["density_schemes"],
                config["scene_acceptance"],
            )
            self.assertTrue(audit["all_ar_non_nominal_and_alternative_supported"])
            self.assertGreaterEqual(
                entry["route_pool_audit"]["minimum_selected_ar_alternative_support_fraction"],
                config["classification"]["alternative_min_path_fraction"],
            )
            for density in ("5", "7"):
                summary = entry["density_summary"][density]
                self.assertAlmostEqual(
                    summary["mean_interacting_path_fraction"]
                    + summary["mean_fully_avoiding_path_fraction"],
                    1.0,
                    places=5,
                )


if __name__ == "__main__":
    unittest.main()
