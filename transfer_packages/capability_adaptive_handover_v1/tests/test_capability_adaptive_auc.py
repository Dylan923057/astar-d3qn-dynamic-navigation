from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from astar_d3qn.training import capability_adaptive_auc


ANALYSIS_PATH = (
    PACKAGE_ROOT / "scripts" / "analyze_capability_adaptive_handover_auc.py"
)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_capability_adaptive_handover_auc", ANALYSIS_PATH
)
analysis = importlib.util.module_from_spec(ANALYSIS_SPEC)
assert ANALYSIS_SPEC.loader is not None
ANALYSIS_SPEC.loader.exec_module(analysis)


class FrozenAlgorithmTests(unittest.TestCase):
    def test_original_implementation_files_are_unchanged(self):
        expected = {
            "src/astar_d3qn/training/capability_adaptive.py": (
                "f41f30a61254ee062cd764513c9750b47806b2f1a178890517d28b0a1d8c9a3c"
            ),
            "scripts/run_capability_adaptive_handover.py": (
                "443b78f5851830ae6b352723aa08effd2993e240d2fff82e8722f38d959f0e10"
            ),
            "configs/capability_adaptive_handover_v1.yaml": (
                "3ee3ae2da824cc29a3cc52c8484d482b8a7f795ca7280fe7e3a58fcc267512a2"
            ),
        }
        actual = {
            relative: hashlib.sha256((PACKAGE_ROOT / relative).read_bytes()).hexdigest()
            for relative in expected
        }
        self.assertEqual(actual, expected)

    def test_auc_config_changes_only_namespace_and_validation_recording(self):
        original = yaml.safe_load(
            (PACKAGE_ROOT / "configs/capability_adaptive_handover_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        auc = yaml.safe_load(
            (
                PACKAGE_ROOT
                / "configs/capability_adaptive_handover_v1_auc.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(auc.pop("validation_recording"), {
            "split": "validation",
            "evaluation_interval": 10_000,
            "epsilon": 0.0,
            "controls_training": False,
            "evaluate_step_zero": True,
        })
        self.assertEqual(auc.pop("experiment"), "capability_adaptive_handover_v1_auc")
        self.assertEqual(
            auc.pop("output_root"), "outputs/capability_adaptive_handover_v1_auc"
        )
        original.pop("experiment")
        original.pop("output_root")
        self.assertEqual(auc, original)

    def test_validation_is_observation_only_and_never_reads_test(self):
        source = inspect.getsource(
            capability_adaptive_auc.train_capability_adaptive_auc
        )
        observer = source.split("    def record_validation", 1)[1].split(
            "    record_validation(0)", 1
        )[0]
        self.assertNotIn("controller.observe", observer)
        self.assertNotIn("agent.train_batch", observer)
        self.assertNotIn("replay.set_demo_fraction", observer)
        self.assertNotIn('splits["test"]', source)
        self.assertIn('splits["validation"]', source)
        self.assertIn("record_validation(0)", source)
        self.assertIn("record_validation(step)", source)
        for column in (
            "environment_steps",
            "conflict_safe_success",
            "control_safe_success",
            "all_safe_success",
            "dynamic_collision",
            "timeout",
        ):
            self.assertIn(f'"{column}"', observer)


class BaselineTests(unittest.TestCase):
    def test_all_time_decay_pairs_are_complete(self):
        root = PACKAGE_ROOT / "baselines" / "time_decay"
        results = sorted(root.rglob("result.json"))
        curves = sorted(root.rglob("validation_curve.csv"))
        self.assertEqual(len(results), 15)
        self.assertEqual(len(curves), 15)
        for path in results:
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "validation_complete")
            self.assertEqual(record["replay_schedule"], "decay")
            self.assertEqual(record["adaptation_run"]["steps"], 200_000)
            self.assertTrue(record["test_deferred"])

    def test_time_decay_curves_have_all_auc_checkpoints(self):
        root = PACKAGE_ROOT / "baselines" / "time_decay"
        for path in root.rglob("validation_curve.csv"):
            rows = analysis.load_curve(path, adaptive=False)
            self.assertEqual(
                tuple(row["environment_steps"] for row in rows),
                analysis.EXPECTED_STEPS,
            )


class AnalysisTests(unittest.TestCase):
    def test_normalized_trapezoid_auc(self):
        curve = [
            {
                "environment_steps": step,
                "conflict_safe_success": step / 200_000,
            }
            for step in analysis.EXPECTED_STEPS
        ]
        self.assertAlmostEqual(analysis.normalized_auc(curve), 0.5)

    def test_bootstrap_ci_is_deterministic(self):
        import numpy as np

        first = analysis.bootstrap_mean_ci(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            np.random.default_rng(analysis.BOOTSTRAP_SEED),
        )
        second = analysis.bootstrap_mean_ci(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            np.random.default_rng(analysis.BOOTSTRAP_SEED),
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
