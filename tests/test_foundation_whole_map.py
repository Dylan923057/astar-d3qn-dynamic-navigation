"""Verify the scene adapter and the guarded foundation reuse entry."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_foundation_whole_map import (
    load_inputs, load_foundation, make_env, scene_sampler, trajectory_stats,
    decay_demo_fraction, epsilon_at,
)
from train_whole_map_route_pool_pilot import _factory
from astar_d3qn.envs.static_grid import RewardConfig


class FoundationWholeMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs("configs/whole_map_91701_foundation_reuse_v1.yaml")

    def test_adapter_preserves_observations_rewards_masks_and_dynamic_clock(self):
        _, legacy, scene_config, problem, entry, pool, _ = self.inputs
        for seed in (0, 1):
            factory = _factory(scene_config, problem, entry, pool,
                               seed=91701000 + seed, prefix=f"train_seed{seed}")
            sample = scene_sampler(scene_config, problem, entry, pool, legacy, seed)
            for _ in range(5):
                direct = factory(problem, max_steps=300, window_size=15,
                                 reward_config=RewardConfig(**legacy["reward"]), terminate_on_collision=True)
                scene = sample()
                adapted = make_env(problem, legacy, scene)
                self.assertEqual(scene["obstacles"], [asdict(s) for s in direct.dynamic_obstacles])
                self.assertIn(len(adapted.dynamic_obstacles), (3, 4, 5))
                left, right = direct.reset(), adapted.reset()
                for step in range(30):
                    np.testing.assert_array_equal(left.spatial, right.spatial)
                    np.testing.assert_array_equal(left.scalars, right.scalars)
                    np.testing.assert_array_equal(direct.action_mask(True), adapted.action_mask(True))
                    # Include waiting and movement, without adding a dynamic safety mask.
                    valid = np.flatnonzero(direct.action_mask(True)).tolist()
                    action = valid[step % len(valid)]
                    a, b = direct.step(action), adapted.step(action)
                    self.assertEqual(a.reward, b.reward)
                    self.assertEqual(a.terminated, b.terminated)
                    self.assertEqual(a.truncated, b.truncated)
                    self.assertEqual(a.info["collision_type"], b.info["collision_type"])
                    self.assertEqual(direct.position, adapted.position)
                    self.assertEqual(direct.dynamic_positions, adapted.dynamic_positions)
                    left, right = a.observation, b.observation
                    if a.done:
                        break

    def test_missing_weights_cannot_fall_back_to_training(self):
        config, legacy, _, problem, *_ = self.inputs
        with tempfile.TemporaryDirectory() as directory:
            changed = {**config, "foundation_source_root": directory}
            expected = Path(directory) / "seed_0/foundation/foundation.pt"
            with self.assertRaisesRegex(FileNotFoundError, "never auto-trained") as error:
                load_foundation(changed, legacy, problem, 0)
            self.assertIn(str(expected), str(error.exception))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failure_diagnosis_separates_waiting_oscillation_and_nonarrival(self):
        goal = (3, 3)
        waiting = trajectory_stats([(0, 0)] * 301, [4] * 300, goal, False, False)
        repeated = trajectory_stats([(0, i % 2) for i in range(301)], [2, 3] * 150,
                                    goal, False, False)
        nonarrival = trajectory_stats([(0, i) for i in range(40)], [2] * 39,
                                      goal, False, False)
        self.assertEqual(waiting["failure_behavior"], "timeout_waiting")
        self.assertEqual(repeated["failure_behavior"], "timeout_repeated_movement")
        self.assertEqual(nonarrival["failure_behavior"], "timeout_no_arrival")
        self.assertEqual(repeated["immediate_reversals"], 299)
        collision = trajectory_stats([(0, 0), (0, 1)], [2], goal, False, True)
        self.assertEqual(collision["failure_behavior"], "collision")

    def test_registered_environment_clock_and_boundary_allocation(self):
        _, legacy, *_ = self.inputs
        stage = legacy["adaptation"]
        self.assertEqual([decay_demo_fraction(s) for s in (1, 50000, 50001, 100000, 100001, 200000)],
                         [.25, .25, .10, .10, 0.0, 0.0])
        self.assertAlmostEqual(epsilon_at(stage, 0), .30)
        self.assertAlmostEqual(epsilon_at(stage, 150000), .05)
        self.assertAlmostEqual(epsilon_at(stage, 200000), .05)


if __name__ == "__main__":
    unittest.main()
