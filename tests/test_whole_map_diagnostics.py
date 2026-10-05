from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from astar_d3qn.replay.demo import PersistentDemoReplay
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.training.demo_collector import collect_astar_demonstrations
from astar_d3qn.training.trainer import build_replay
from astar_d3qn.utils.config import load_config
from evaluate_whole_map_astar_only import _validate_astar_config
from train_whole_map_route_pool_pilot import (
    _load_inputs,
    _reward_config,
    _training_config,
    _validate_prefill_comparison,
    _validate_protocol,
    _validate_time_decay_comparison,
)


class WholeMapDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.astar_config = load_config(
            ROOT / "configs/whole_map_route_pool_91701_astar_only_v1.yaml"
        )
        cls.capacity_config = load_config(
            ROOT
            / "configs/whole_map_route_pool_91701_time_decay_capacity_controlled_v1.yaml"
        )
        cls.prefill_config = load_config(
            ROOT / "configs/whole_map_route_pool_91701_prefill_only_v1.yaml"
        )
        problem, map_entry, pool = _load_inputs(cls.capacity_config)
        cls.problem = problem
        cls.map_entry = map_entry
        cls.pool = pool
        cls.demonstrations = collect_astar_demonstrations(
            [problem],
            episodes=20,
            seed=7400,
            max_steps=300,
            reward_config=_reward_config(cls.capacity_config),
            window_size=15,
            spatial_channels=4,
            mask_static_invalid_actions=True,
        )

    def test_astar_only_is_validation_without_training(self):
        _validate_protocol(
            self.astar_config,
            self.problem,
            self.map_entry,
            self.pool,
        )
        _validate_astar_config(self.astar_config)
        self.assertFalse(self.astar_config["diagnostic"]["training_started"])
        self.assertEqual(self.astar_config["diagnostic"]["evaluation_episodes"], 50)

    def test_demo_source_is_exactly_1320_transitions(self):
        self.assertEqual(len(self.demonstrations), 1320)

    def test_capacity_controlled_time_decay_keeps_10000_online_slots(self):
        _validate_protocol(
            self.capacity_config,
            self.problem,
            self.map_entry,
            self.pool,
        )
        _validate_time_decay_comparison(self.capacity_config)
        training = _training_config(self.capacity_config, seed=0)
        replay = build_replay(training, self.demonstrations)
        self.assertIsInstance(replay, PersistentDemoReplay)
        self.assertEqual(replay.demonstration_size, 1320)
        self.assertEqual(replay.online.capacity, 10000)
        self.assertEqual(training.replay_capacity, 11320)

    def test_prefill_uses_evictable_uniform_capacity_10000(self):
        _validate_protocol(
            self.prefill_config,
            self.problem,
            self.map_entry,
            self.pool,
        )
        _validate_prefill_comparison(self.prefill_config)
        training = _training_config(self.prefill_config, seed=0)
        replay = build_replay(training, self.demonstrations)
        self.assertIsInstance(replay, UniformReplayBuffer)
        self.assertEqual(replay.capacity, 10000)
        self.assertEqual(len(replay), 1320)
        for transition in self.demonstrations:
            replay.add(transition)
        self.assertEqual(len(replay), 2640)
        self.assertEqual(training.demo_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
