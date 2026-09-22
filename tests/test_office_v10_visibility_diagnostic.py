"""Checks for retained evidence selection and diagnostic collision traces."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_office_v10_visibility as diagnostic


class VisibilityDiagnosticTests(unittest.TestCase):
    def test_legacy_examples_are_loaded_without_requiring_new_generation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write = diagnostic.v10.write_json
            write({"groups": [{"behavior": "avoidance", "seed_offset": 0},
                               {"behavior": "wait", "seed_offset": 0}]}, root / "batch_summary.json")
            raw = [{"route_id": "example"}]
            write({"examples": {"primary_not_visible": raw}}, root / "seed_0/avoidance/test_avoidance_search.json")
            cases = diagnostic._saved_cases(root)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0][1]["obstacles"], raw)
            retained = {"obstacles": raw, "primary_obstacle_index": 2}
            write(retained, root / "seed_0/avoidance/visibility_failures/test_avoidance_0001.json")
            cases = diagnostic._saved_cases(root)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0][1], retained)

    def test_trace_flags_old_and_new_dynamic_cells(self):
        spec = diagnostic.v10.DynamicObstacleSpec(route=((1, 1), (1, 2)))
        scene = diagnostic.v10.DynamicScenario(seed=0, obstacles=(spec,))
        for destination in ((1, 1), (1, 2)):
            plan = diagnostic.v9.SafePlan(((0, 1), destination), 1, 0)
            trace = diagnostic._plan_trace(None, scene, plan, 0, 1)
            self.assertEqual(trace[0]["next_action_collision_indices"], [0])
            self.assertTrue(trace[0]["primary_visible"])


if __name__ == "__main__":
    unittest.main()
