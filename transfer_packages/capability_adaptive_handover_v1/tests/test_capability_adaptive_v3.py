from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astar_d3qn.training import capability_adaptive_auc as base
from astar_d3qn.training import capability_adaptive_v2 as v2
from astar_d3qn.envs.types import Observation
from astar_d3qn.training.capability_adaptive_v3 import (
    CapabilityHandoverV3, METHOD, observe_monitor, train_capability_adaptive_v3,
)


class ControllerTests(unittest.TestCase):
    def test_ema_and_exact_two_stage_exit(self):
        controller = CapabilityHandoverV3()
        logs = [controller.observe(1., 0., 1., step)
                for step in range(10_000, 120_001, 10_000)]
        self.assertAlmostEqual(logs[0]["Sbar_t"], .3)
        self.assertAlmostEqual(logs[1]["Ebar_t"], .51)
        self.assertFalse(logs[5]["qualified"])
        self.assertEqual([r["demo_ratio"] for r in logs],
                         [.25] * 7 + [.10] * 2 + [0.] * 3)
        self.assertEqual(logs[6]["streak"], 1)
        self.assertEqual(logs[7]["streak"], 0)
        self.assertEqual(logs[8]["streak"], 1)
        self.assertEqual(logs[9]["streak"], 0)
        self.assertEqual(controller.expert_exit_step, 100_000)
        controller.observe(0., 1., 0., 130_000)
        self.assertEqual(controller.demo_fraction, 0.)

    def test_each_metric_blocks_and_resets_streak(self):
        for initial in ({"competence_ema": .89}, {"collision_ema": .06},
                        {"efficiency_ema": .89}):
            c = CapabilityHandoverV3(competence_ema=.95, efficiency_ema=.95)
            for key, value in initial.items():
                setattr(c, key, value)
            row = c.observe(c.competence_ema, c.collision_ema, c.efficiency_ema, 10_000)
            self.assertFalse(row["qualified"])
            self.assertEqual(c.demo_fraction, .25)
        c = CapabilityHandoverV3(competence_ema=.95, efficiency_ema=.95)
        c.observe(1., 0., 1., 10_000)
        c.observe(0., 1., 0., 20_000)
        self.assertEqual(c.threshold_streak, 0)
        self.assertEqual(c.demo_fraction, .25)

    def test_thresholds_are_inclusive(self):
        c = CapabilityHandoverV3(competence_ema=.9, collision_ema=.05, efficiency_ema=.9)
        self.assertTrue(c.observe(.9, .05, .9, 10_000)["qualified"])

    def test_middle_stage_never_rebounds_and_collision_ema(self):
        c = CapabilityHandoverV3(competence_ema=.95, efficiency_ema=.95)
        c.observe(1., 0., 1., 10_000)
        c.observe(1., 0., 1., 20_000)
        self.assertEqual(c.demo_fraction, .10)
        log = c.observe(0., 1., 0., 30_000)
        self.assertAlmostEqual(log["Cbar_t"], .3)
        self.assertEqual((c.demo_fraction, c.threshold_streak), (.10, 0))
        log = c.observe(0., 0., 0., 40_000)
        self.assertAlmostEqual(log["Cbar_t"], .21)
        self.assertEqual(c.demo_fraction, .10)

    def test_rollout_is_greedy_and_reports_episode_collisions(self):
        agent = SimpleNamespace(select_action=lambda *args: 0)
        observation = object()
        result = SimpleNamespace(observation=observation, reward=-1., done=True,
                                 truncated=False, info={"reached": False,
                                                        "collision_type": "static"})
        env = SimpleNamespace(reset=lambda: observation, steps=1,
                              action_mask=lambda flag: np.ones(5, dtype=bool),
                              step=lambda action: result)
        with patch.object(agent, "select_action", wraps=agent.select_action) as select, \
             patch.object(base, "make_env", return_value=env):
            row = base.rollout(agent, None, {"max_episode_steps": 300}, {})
        self.assertEqual(select.call_args.args[1], 0.0)
        self.assertEqual((row["safe_success"], row["static_collision"], row["steps"]), (0, 1, 1))

    def test_invalid_metrics_and_cadence(self):
        for metrics in ((float("nan"), 0., 1.), (1., -1., 1.), (1., 0., 1.1)):
            with self.assertRaises(ValueError):
                CapabilityHandoverV3().observe(*metrics, 10_000)
        with self.assertRaises(ValueError):
            CapabilityHandoverV3().observe(1., 0., 1., 20_000)

    def test_episode_aggregation_failure_zero_collision_union_and_clamp(self):
        rows = [dict(safe_success=1, dynamic_collision=0, static_collision=0, steps=20),
                dict(safe_success=0, dynamic_collision=1, static_collision=1, steps=1),
                dict(safe_success=0, dynamic_collision=0, static_collision=1, steps=1),
                dict(safe_success=1, dynamic_collision=0, static_collision=0, steps=5)]
        scenes = [{"source_split": "train_monitor"}] * 4
        log = observe_monitor(CapabilityHandoverV3(), rows, scenes,
                              SimpleNamespace(astar_steps=10), 10_000)
        self.assertEqual((log["S_t"], log["C_t"], log["E_t"]), (.5, .5, .375))
        for split in ("train", "validation", "test"):
            with self.assertRaises(ValueError):
                observe_monitor(CapabilityHandoverV3(), rows,
                                [{"source_split": split}] * 4,
                                SimpleNamespace(astar_steps=10), 10_000)

    def test_config_preserves_v2_conditions(self):
        old = yaml.safe_load((ROOT / "configs/capability_adaptive_handover_v2_independent_monitor.yaml").read_text())
        new = yaml.safe_load((ROOT / "configs/capability_adaptive_handover_v3.yaml").read_text())
        for key in ("experiment", "output_root", "adaptive_v3"):
            new.pop(key)
            old.pop(key, None)
        self.assertEqual(old, new)


class IntegrationTests(unittest.TestCase):
    def test_reused_training_loop_monitor_logs_and_validation_isolation(self):
        config = yaml.safe_load((ROOT / "configs/capability_adaptive_handover_v3.yaml").read_text())
        config["adaptation"]["max_steps"] = 40_000
        problem = SimpleNamespace(map_id="tiny", astar_steps=10)
        agent = SimpleNamespace(_rng=random.Random(1), update_steps=0,
                                training_state_dict=lambda: {},
                                select_action=lambda *args: 0,
                                save_weights=lambda path: None)
        replay = SimpleNamespace(online=SimpleNamespace(state_dict=lambda: {}),
                                 _rng=random.Random(2), demo_fraction=.25,
                                 demonstration_snapshot=lambda: [], add=lambda value: None,
                                 can_sample=lambda size: False,
                                 sample_counts=lambda size: (round(size * replay.demo_fraction), 0))
        replay.set_demo_fraction = lambda value: setattr(replay, "demo_fraction", value)
        observation = Observation(np.zeros((4, 15, 15)), np.zeros(2))
        result = SimpleNamespace(observation=observation, reward=0., done=False,
                                 terminated=False, info={"reached": False, "collision": False})
        env = SimpleNamespace(reset=lambda: observation, action_mask=lambda flag: np.ones(5, dtype=bool),
                              step=lambda action: result)
        pair = {"pair_id": "p", "first_reference_collision_step": 1,
                "control": {"condition": "control"}, "conflict": {"condition": "conflict"}}
        scenes = [dict(scenario_id=f"m{i}", source_split="train_monitor") for i in range(24)]
        calls = []
        def rollout(agent, problem, config, scene):
            calls.append(scene["source_split"])
            # Poor validation must not prevent monitor-triggered exits.
            return dict(safe_success=int(scene["source_split"] == "train_monitor"),
                        dynamic_collision=0, static_collision=0, timeout=0, steps=10, **{"return": 0.})
        checkpoint = {"metadata": dict(qualified=True, smoke=False, seed=0,
                                      map_id="tiny", device="cpu", snapshot_sha256="digest")}
        checkpoint["metadata"]["snapshot_sha256"] = base.state_digest({
            "agent": {}, "online_replay": {}, "demonstrations": [],
            "demo_rng": replay._rng.getstate(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        })
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            with patch.object(base, "_validate_frozen_conditions"), \
                 patch.object(base, "restore_foundation", return_value=(agent, replay)), \
                 patch.object(base, "make_env", return_value=env), \
                 patch.object(base, "rollout", side_effect=rollout), \
                 patch.object(v2, "generate_train_monitor_scenes", return_value=(scenes, {"manifest_sha256": "monitor"})), \
                 patch("astar_d3qn.training.capability_adaptive_v3.CapabilityHandoverV3",
                       return_value=CapabilityHandoverV3(competence_ema=.95, efficiency_ema=.95)):
                returned = train_capability_adaptive_v3(
                    problem, config, {"train": [pair], "validation": [copy.deepcopy(pair)]},
                    0, "cpu", output, checkpoint, {})
            with (output / "training.csv").open(encoding="utf-8-sig", newline="") as handle:
                logs = list(csv.DictReader(handle))
            self.assertEqual([int(row["env_steps"]) for row in logs], [10000, 20000, 30000, 40000])
            self.assertEqual([float(row["demo_ratio"]) for row in logs], [.25, .1, .1, 0.])
            self.assertEqual([int(row["streak"]) for row in logs], [1, 0, 1, 0])
            for key in ("S_t", "C_t", "E_t", "Sbar_t", "Cbar_t", "Ebar_t", "qualified"):
                self.assertIn(key, logs[0])
            self.assertEqual(calls.count("train_monitor"), 24 * 4)
            self.assertEqual(calls.count("validation"), 2 * 5)
            self.assertNotIn("test", calls)
            self.assertEqual(replay.demo_fraction, 0.)
            self.assertEqual(returned["method"], METHOD)
            audit = json.loads((output / "run_audit.json").read_text())
            self.assertEqual(audit["method"], METHOD)
            self.assertFalse(audit["monitor_scenes_in_replay"])
            self.assertFalse(audit["validation_controls_training"])
            self.assertEqual(json.loads((output / "result.json").read_text())["method"], METHOD)


if __name__ == "__main__":
    unittest.main()
