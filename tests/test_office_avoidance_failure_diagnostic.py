import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_office_v10_avoidance_failures as diagnostic


class AvoidanceFailureDiagnosticTests(unittest.TestCase):
    def test_search_budget_is_not_a_geometry_failure(self):
        self.assertEqual(diagnostic.classify_search(
            {"status": "static_bypass_possible"}, {"status": "budget_exhausted"}),
            "unknown_resource_limit")

    def test_static_cut_and_dynamic_exhaustion_are_distinguished(self):
        search = {"status": "infeasible_under_constraints"}
        self.assertEqual(diagnostic.classify_search({"status": "conflict_cell_is_required"}, search),
                         "geometric_constraint_blocks_bypass")
        self.assertEqual(diagnostic.classify_search({"status": "static_bypass_possible"}, search),
                         "no_path_under_dynamic_visibility_simple_path_constraints")

    def test_generation_retains_capped_examples_for_each_stop_reason(self):
        v10 = diagnostic.v10
        problem = SimpleNamespace(grid_sha256="test", nominal_path=((0, 0), (0, 1)))
        scene = v10.DynamicScenario(seed=0, obstacles=(v10.DynamicObstacleSpec(
            route=((0, 1), (1, 1)), label="test_route"),))
        counts = {}
        with patch.object(v10, "write_json") as save:
            for reason in ("static_disconnection", "generated_states"):
                for _ in range(5):
                    v10._save_spatial_failure(problem, scene, 0,
                                              {"observation_radius": 7, "_search_audit_dir": "unused"},
                                              counts, {"stop_reason": reason})
            self.assertEqual(save.call_count, 6)
        self.assertEqual(counts["spatial_stop_reasons"],
                         {"static_disconnection": 5, "generated_states": 5})
        self.assertEqual(counts["retained_spatial_failures_by_reason"],
                         {"static_disconnection": 3, "generated_states": 3})


if __name__ == "__main__":
    unittest.main()
