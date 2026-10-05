from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.independent_train_monitor import generate_train_monitor_scenes


class IndependentMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((PACKAGE_ROOT / "configs" / "capability_adaptive_handover_v2_independent_monitor.yaml").read_text(encoding="utf-8"))
        cls.manifest = json.loads((PACKAGE_ROOT / "data" / "risk_handover_v1" / "manifest.json").read_text(encoding="utf-8"))

    def test_each_map_has_new_12_plus_12_monitor_scenes(self):
        for entry in self.manifest["maps"]:
            problem = problem_from_record(entry["problem"])
            scenes, audit = generate_train_monitor_scenes(problem, entry["scenarios"]["splits"], self.config["train_monitor"])
            self.assertEqual(len(scenes), 24)
            self.assertEqual({density: sum(scene["obstacle_count"] == density for scene in scenes) for density in (3, 5)}, {3: 12, 5: 12})
            self.assertEqual(len({scene["scene_sha256"] for scene in scenes}), 24)
            self.assertTrue(audit["all_existing_split_scene_fingerprints_excluded"])
            self.assertTrue(all(scene["source_split"] == "train_monitor" for scene in scenes))

    def test_v2_config_keeps_controller_and_validation_protocol(self):
        self.assertEqual(self.config["capability_adaptive_handover"]["rho_max"], 0.25)
        self.assertEqual(self.config["capability_adaptive_handover"]["beta"], 0.3)
        self.assertEqual(self.config["capability_adaptive_handover"]["tau"], 0.9)
        self.assertEqual(self.config["capability_adaptive_handover"]["consecutive_confirmations"], 2)
        self.assertEqual(self.config["validation_recording"]["controls_training"], False)
        self.assertEqual(self.config["validation_recording"]["split"], "validation")

    def test_v2_adapter_does_not_add_monitor_scenes_to_train_or_replay(self):
        source = (PACKAGE_ROOT / "src" / "astar_d3qn" / "training" / "capability_adaptive_v2.py").read_text(encoding="utf-8")
        self.assertIn("base.select_monitor_scenes = v2_monitor_selector", source)
        self.assertIn("monitor_scenes_in_training_stream", source)
        self.assertIn("monitor_scenes_in_replay", source)
        self.assertNotIn("replay.add", source)
        self.assertNotIn("train_batch", source)

if __name__ == "__main__":
    unittest.main()
