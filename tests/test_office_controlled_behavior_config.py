from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from astar_d3qn.utils.config import load_config
from run_office_controlled_behavior_study import METHODS
from train_dynamic_spatial_generalization import _load_problem, _scenario_schedules


class OfficeControlledBehaviorConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            ROOT / "configs/dynamic_office_controlled_behavior_v1.yaml"
        )
        cls.current = load_config(
            ROOT / "configs/dynamic_office_controlled_behavior_current_v1.yaml"
        )
        cls.scheduled_decay = load_config(
            ROOT / "configs/dynamic_office_scheduled_decay_v2.yaml"
        )
        cls.adaptive_decay = load_config(
            ROOT / "configs/dynamic_office_conflict_adaptive_decay_v2.yaml"
        )
        cls.local_counterexample = load_config(
            ROOT / "configs/dynamic_office_local_counterexample_v1.yaml"
        )
        cls.predictive_margin = load_config(
            ROOT / "configs/dynamic_office_predictive_margin_v1.yaml"
        )
        manifest_path = ROOT / cls.config["spatial_generalization"]["manifest"]
        cls.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_frozen_splits_have_preregistered_behavior_counts(self):
        spatial = self.config["spatial_generalization"]
        for split in ("train", "validation", "test"):
            scenarios = self.manifest["scenarios"][split]
            actual = Counter(row["required_behavior"] for row in scenarios)
            self.assertEqual(len(scenarios), spatial["scenario_counts"][split])
            self.assertEqual(actual, Counter(spatial["behavior_counts"][split]))

    def test_every_obstacle_is_an_active_astar_interaction(self):
        for split in ("train", "validation", "test"):
            self.assertEqual(len(self.manifest["route_pools"][split]["background"]), 0)
            for scenario in self.manifest["scenarios"][split]:
                self.assertEqual(len(scenario["obstacles"]), 5)
                self.assertTrue(
                    all(item["active_demo_blocker"] for item in scenario["obstacles"])
                )
                self.assertGreaterEqual(
                    sum(
                        bool(item["counterfactually_decisive"])
                        for item in scenario["obstacles"]
                    ),
                    2,
                )

    def test_wait_scenarios_have_a_strict_verified_wait_advantage(self):
        for split in ("train", "validation", "test"):
            waits = [
                row
                for row in self.manifest["scenarios"][split]
                if row["required_behavior"] == "wait"
            ]
            self.assertTrue(waits)
            for scenario in waits:
                self.assertGreaterEqual(scenario["oracle_wait_count"], 1)
                self.assertGreaterEqual(scenario["wait_advantage_steps"], 1)
                self.assertGreater(
                    scenario["no_wait_safe_path_steps"],
                    scenario["minimum_safe_path_steps"],
                )

    def test_training_removes_extra_wait_penalty_and_uses_curriculum(self):
        self.assertFalse(self.config["environment"]["terminate_on_collision"])
        self.assertTrue(self.config["environment"]["mask_static_invalid_actions"])
        self.assertEqual(self.config["reward"]["stay"], 0.0)
        stages = self.config["training"]["dynamic_obstacle_curriculum"]["stages"]
        self.assertEqual(
            [stage["obstacle_count"] for stage in stages], [0, 1, 3, 5]
        )
        self.assertEqual(
            [stage["start_environment_step"] for stage in stages],
            [0, 60_000, 140_000, 240_000],
        )

    def test_validation_and_test_are_seen_training_scenarios_not_generalization(self):
        spatial = self.config["spatial_generalization"]
        self.assertEqual(
            spatial["evaluation_design"],
            "in_distribution_seen_training_scenarios",
        )
        self.assertEqual(
            spatial["scenario_split_sources"],
            {"train": "train", "validation": "train", "test": "train"},
        )
        problem = _load_problem(self.config)
        schedules, sources = _scenario_schedules(
            problem, self.manifest, spatial
        )
        train_ids = {scenario.seed for scenario in schedules["train"]}
        validation_ids = {scenario.seed for scenario in schedules["validation"]}
        test_ids = {scenario.seed for scenario in schedules["test"]}
        self.assertEqual(sources["validation"], "train")
        self.assertEqual(sources["test"], "train")
        self.assertTrue(validation_ids.issubset(train_ids))
        self.assertTrue(test_ids.issubset(train_ids))
        self.assertFalse(validation_ids.intersection(test_ids))
        for split, expected in spatial["behavior_counts"].items():
            actual = Counter(
                scenario.required_behavior for scenario in schedules[split]
            )
            self.assertEqual(actual, Counter(expected))

    def test_current_ablation_changes_only_conflict_lookahead_output_identity(self):
        self.assertFalse(self.current["training"]["conflict_predict_next"])
        self.assertTrue(self.config["training"]["conflict_predict_next"])
        self.assertNotEqual(
            self.current["experiment"]["output_root"],
            self.config["experiment"]["output_root"],
        )

    def test_new_decay_configs_are_isolated_and_mechanistically_distinct(self):
        scheduled = self.scheduled_decay["training"]
        adaptive = self.adaptive_decay["training"]
        self.assertEqual(scheduled["demo_fraction_final"], 0.0)
        self.assertEqual(
            scheduled["demo_fraction_decay_start_environment_step"],
            200_000,
        )
        self.assertEqual(
            scheduled["demo_fraction_decay_end_environment_step"],
            340_000,
        )
        self.assertTrue(adaptive["conflict_event_binary"])
        self.assertEqual(adaptive["conflict_ema_alpha"], 0.5)
        self.assertEqual(adaptive["conflict_recovery_alpha"], 0.005)
        self.assertEqual(adaptive["conflict_demo_fraction_min"], 0.05)
        roots = {
            self.config["experiment"]["output_root"],
            self.scheduled_decay["experiment"]["output_root"],
            self.adaptive_decay["experiment"]["output_root"],
        }
        self.assertEqual(len(roots), 3)

    def test_runner_contains_baselines_and_two_decay_variants(self):
        self.assertEqual(
            [method.key for method in METHODS],
            [
                "uniform",
                "prefill",
                "persistent",
                "scheduled_decay",
                "ca_current",
                "ca_predictive",
                "local_conflict",
                "ca_adaptive_decay",
                "local_counterexample",
                "predictive_margin",
            ],
        )

    def test_local_counterexample_preserves_demo_quota_and_isolates_output(self):
        training = self.local_counterexample["training"]
        self.assertEqual(training["demo_fraction"], 0.25)
        self.assertEqual(training["local_conflict_min_sampling_weight"], 0.0)
        self.assertEqual(training["local_counterexample_capacity"], 2000)
        self.assertEqual(training["local_counterexample_fraction"], 0.25)
        self.assertNotEqual(
            self.local_counterexample["experiment"]["output_root"],
            self.config["experiment"]["output_root"],
        )

    def test_predictive_margin_preserves_demos_without_counterexample_replay(self):
        training = self.predictive_margin["training"]
        self.assertEqual(training["demo_fraction"], 0.25)
        self.assertEqual(training["conflict_margin"], 0.8)
        self.assertEqual(training["conflict_margin_loss_weight"], 1.0)
        self.assertTrue(training["conflict_margin_predict_next"])
        self.assertNotEqual(
            self.predictive_margin["experiment"]["output_root"],
            self.local_counterexample["experiment"]["output_root"],
        )


if __name__ == "__main__":
    unittest.main()
