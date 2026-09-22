import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from astar_d3qn.utils.config import load_config
from plot_office_behavior_v6_results import _uncertainty_label


class OfficeDevelopmentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/dynamic_spatial_generalization_office_v6_development_v1.yaml")
        cls.manifest_path = ROOT / cls.config["spatial_generalization"]["manifest"]
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_uses_existing_frozen_data_and_matching_counts(self):
        self.assertEqual(hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                         "ce47ee4ef71f9e110c69379b08669d0ab9c311afb10dcbc9d2d8f5a65aacda97")
        spatial = self.config["spatial_generalization"]
        for split in ("train", "validation", "test"):
            self.assertEqual(len(self.manifest["scenarios"][split]), spatial["scenario_counts"][split])
            for kind, count in spatial["route_pool_counts"][split].items():
                self.assertEqual(len(self.manifest["route_pools"][split][kind]), count)

    def test_curriculum_does_not_select_nonexistent_difficulty_labels(self):
        stages = self.config["training"]["dynamic_obstacle_curriculum"]["stages"]
        self.assertEqual([s["start_environment_step"] for s in stages], [0, 60000, 140000, 240000])
        self.assertEqual([s["obstacle_count"] for s in stages], [0, 1, 3, 5])
        for stage in stages:
            self.assertIsNone(stage["allowed_difficulties"])
            self.assertEqual(len(stage["obstacle_indices"]), stage["obstacle_count"])
            for scene in self.manifest["scenarios"]["train"]:
                self.assertTrue(all(0 <= i < len(scene["obstacles"]) for i in stage["obstacle_indices"]))

    def test_preserves_agreed_training_and_isolates_output(self):
        cfg = self.config
        self.assertTrue(cfg["environment"]["terminate_on_collision"])
        self.assertTrue(cfg["environment"]["mask_static_invalid_actions"])
        self.assertEqual(cfg["reward"], {"step": -.01, "progress": .05, "stay": 0., "collision": -1., "goal": 10.})
        self.assertEqual(cfg["training"]["seeds"], [0])
        self.assertEqual(cfg["training"]["max_environment_steps"], 400000)
        self.assertEqual(cfg["training"]["epsilon_decay_environment_steps"], 340000)
        self.assertEqual(cfg["training"]["validation_interval_environment_steps"], 25000)
        self.assertEqual(cfg["training"]["demo_fraction"], .25)
        self.assertTrue(cfg["training"]["select_best_checkpoint"])
        self.assertEqual(cfg["experiment"]["output_root"], "outputs/dynamic_spatial_generalization_office_v6_development_v1")

    def test_single_seed_figures_do_not_claim_a_confidence_interval(self):
        self.assertIn("no across-seed CI", _uncertainty_label([0]))
        self.assertIn("95% CI", _uncertainty_label([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
