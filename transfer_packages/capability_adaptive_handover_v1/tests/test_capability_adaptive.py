from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from astar_d3qn.training.capability_adaptive import (
    CapabilityHandover,
    select_monitor_scenes,
)


def synthetic_pair(pair_id: str, split: str) -> dict:
    scene_common = {
        "obstacles": [],
        "condition": "unused",
        "oracle_steps": 1,
        "split_marker": split,
    }
    return {
        "pair_id": pair_id,
        "first_reference_collision_step": 1,
        "control": {**scene_common, "condition": "control"},
        "conflict": {**scene_common, "condition": "conflict"},
    }


class MonitorSelectionTests(unittest.TestCase):
    def test_monitor_is_deterministic_conflict_only_and_train_only(self):
        splits = {
            "train": [synthetic_pair("pair_b", "train"), synthetic_pair("pair_a", "train")],
            "validation": [synthetic_pair("pair_0", "validation")],
            "test": [synthetic_pair("pair_0", "test")],
        }
        selected = select_monitor_scenes(splits, 2)
        self.assertEqual([scene["pair_id"] for scene in selected], ["pair_a", "pair_b"])
        self.assertTrue(all(scene["condition"] == "conflict" for scene in selected))
        self.assertTrue(all(scene["source_split"] == "train" for scene in selected))
        self.assertTrue(all(scene["split_marker"] == "train" for scene in selected))

    def test_monitor_count_comes_from_config(self):
        config = yaml.safe_load(
            (PACKAGE_ROOT / "configs" / "capability_adaptive_handover_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        count = config["capability_adaptive_handover"]["monitor_scene_count"]
        splits = {
            "train": [synthetic_pair(f"pair_{index:03d}", "train") for index in range(36)],
            "validation": [],
            "test": [],
        }
        self.assertEqual(len(select_monitor_scenes(splits, count)), count)


class CapabilityHandoverTests(unittest.TestCase):
    def test_rho_is_monotone_nonincreasing(self):
        controller = CapabilityHandover()
        values = [float(controller.demo_fraction)]
        for step, success in enumerate((0.0, 0.4, 0.8, 0.2, 1.0), start=1):
            controller.observe(success, step * 10_000)
            values.append(float(controller.demo_fraction))
        self.assertTrue(all(right <= left for left, right in zip(values, values[1:])))

    def test_competence_decline_cannot_raise_rho(self):
        controller = CapabilityHandover()
        controller.observe(1.0, 10_000)
        reduced = float(controller.demo_fraction)
        controller.observe(0.0, 20_000)
        self.assertLess(controller.competence_ema, 0.3)
        self.assertEqual(controller.demo_fraction, reduced)

    def test_k_confirmations_permanently_exit_expert(self):
        controller = CapabilityHandover(beta=1.0, tau=0.9, consecutive_confirmations=2)
        controller.observe(0.95, 10_000)
        self.assertIsNone(controller.expert_exit_step)
        controller.observe(0.91, 20_000)
        self.assertEqual(controller.expert_exit_step, 20_000)
        self.assertEqual(controller.demo_fraction, 0.0)
        controller.observe(0.0, 30_000)
        self.assertEqual(controller.expert_exit_step, 20_000)
        self.assertEqual(controller.demo_fraction, 0.0)


class IsolationTests(unittest.TestCase):
    def test_original_time_decay_and_global_prediction_sources_unchanged(self):
        workspace_root = PACKAGE_ROOT.parents[1]
        expected = json.loads(
            (PACKAGE_ROOT / "protected_source_hashes.json").read_text(encoding="utf-8")
        )
        if not all((workspace_root / relative).is_file() for relative in expected):
            self.skipTest("Original workspace is not present on this transferred copy.")
        actual = {
            relative: hashlib.sha256((workspace_root / relative).read_bytes()).hexdigest()
            for relative in expected
        }
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
