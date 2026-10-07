from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from astar_d3qn.agents.astar_replan_wait import ObservedAStarReplanWaitPolicy
from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment, PathGuidanceFactory
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.training.trainer import train_d3qn
from astar_d3qn.utils.config import load_config
import run_runtime_path_guidance as entry


def problem(obstacles=frozenset()):
    start, goal = (3, 0), (3, 6)
    path = astar_path(start, goal, obstacles, 7)
    assert path is not None
    return NavigationProblem("path_test", 0, 7, start, goal, obstacles, tuple(path))


def environment(*, enabled=True, specs=(), max_steps=30, record_trace=False):
    return PathGuidanceEnvironment(
        DynamicGridNavigationEnv(problem(), specs, max_steps=max_steps,
                                 window_size=5, terminate_on_collision=True),
        enabled=enabled, lookahead_steps=2, record_trace=record_trace,
    )


class RuntimePathGuidanceTests(unittest.TestCase):
    def test_path_and_subgoal_coordinates_follow_the_agent(self):
        env = environment()
        state = env.reset()
        self.assertEqual(state.spatial.shape, (5, 5, 5))
        self.assertEqual(state.scalars.shape, (4,))
        self.assertEqual(state.spatial.dtype, np.float32)
        np.testing.assert_array_equal(state.spatial[4, 2], [0, 0, 1, 1, 1])
        np.testing.assert_allclose(state.scalars, [0, 1, 0, 2 / 6])
        moved = env.step(int(Action.RIGHT)).observation
        np.testing.assert_array_equal(moved.spatial[4, 2], [0, 1, 1, 1, 1])
        np.testing.assert_allclose(moved.scalars, [0, 5 / 6, 0, 2 / 6])

    def test_control_preserves_base_inputs_with_zero_extra_inputs(self):
        guided, control = environment(), environment(enabled=False)
        left, right = guided.reset(), control.reset()
        np.testing.assert_array_equal(left.spatial[:4], right.spatial[:4])
        np.testing.assert_array_equal(left.scalars[:2], right.scalars[:2])
        self.assertFalse(np.any(right.spatial[4]))
        self.assertFalse(np.any(right.scalars[2:]))
        self.assertEqual(guided.observation_shape, control.observation_shape)
        self.assertEqual(guided.scalar_dim, control.scalar_dim)

    def test_wait_deviation_and_rejoining_do_not_force_actions(self):
        env = environment()
        env.reset()
        target = env.guidance()
        env.step(int(Action.STAY))
        self.assertEqual(target, env.guidance())
        env.step(int(Action.UP))
        self.assertEqual(env.position, (2, 0))
        self.assertEqual(env.guidance()["path_deviation"], 1)
        self.assertEqual(env.guidance()["subgoal"], (3, 2))
        env.step(int(Action.RIGHT))
        env.step(int(Action.DOWN))
        self.assertEqual(env.position, (3, 1))
        self.assertEqual(env.guidance()["subgoal"], (3, 3))

    def test_goal_clamping_reset_and_repeated_reads_have_no_cursor(self):
        env = environment()
        env.reset()
        for _ in range(6):
            result = env.step(int(Action.RIGHT))
        self.assertTrue(result.terminated)
        self.assertEqual(env.guidance()["subgoal"], env.problem.goal)
        np.testing.assert_array_equal(result.observation.scalars, np.zeros(4))
        first = env.reset()
        second = env.observation()
        np.testing.assert_array_equal(first.spatial, second.spatial)
        self.assertEqual(env.guidance()["reference_index"], 0)

    def test_dynamic_collision_reward_mask_and_timeout_are_unchanged(self):
        # The obstacle moves into the next reference cell; it remains an allowed
        # action under the static mask and must still produce a collision.
        specs = (DynamicObstacleSpec(((2, 1), (3, 1)), 0, 1, move_every=1),)
        wrapped = environment(specs=specs)
        base = DynamicGridNavigationEnv(problem(), specs, max_steps=30, window_size=5,
                                        terminate_on_collision=True)
        wrapped.reset()
        base.reset()
        self.assertTrue(wrapped.action_mask(True)[int(Action.RIGHT)])
        observed = wrapped.step(int(Action.RIGHT))
        expected = base.step(int(Action.RIGHT))
        self.assertEqual(observed.info, expected.info)
        self.assertEqual(observed.reward, expected.reward)
        self.assertTrue(observed.terminated)
        self.assertEqual(observed.info["collision_type"], "dynamic")
        env = environment(max_steps=1)
        env.reset()
        timeout = env.step(int(Action.STAY))
        self.assertTrue(timeout.truncated)
        self.assertFalse(timeout.terminated)
        self.assertEqual(timeout.info["termination_reason"], "timeout")

    def test_guidance_does_not_inspect_dynamic_obstacle_motion(self):
        left = environment()
        right = environment(specs=(DynamicObstacleSpec(((2, 1), (3, 1)), 1, -1),))
        ls, rs = left.reset(), right.reset()
        np.testing.assert_array_equal(ls.spatial[4], rs.spatial[4])
        np.testing.assert_array_equal(ls.scalars, rs.scalars)
        self.assertEqual(left.guidance(), right.guidance())

    def test_trace_records_pre_action_obstacles_and_guidance(self):
        env = environment(record_trace=True)
        env.reset()
        env.step(int(Action.UP))
        self.assertEqual(env.trace[0]["before"]["position"], (3, 0))
        self.assertEqual(env.trace[0]["before"]["subgoal"], (3, 2))
        self.assertEqual(env.trace[0]["position"], (2, 0))
        self.assertEqual(env.trace[0]["termination_reason"], "running")
        env.reset()
        self.assertEqual(env.trace, [])

    def test_invalid_reference_and_lookahead_are_rejected(self):
        for lookahead in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                PathGuidanceEnvironment(DynamicGridNavigationEnv(problem(), window_size=5), lookahead_steps=lookahead)
        invalid = replace(problem(), nominal_path=((3, 0), (3, 6)))
        with self.assertRaises(ValueError):
            PathGuidanceEnvironment(DynamicGridNavigationEnv(invalid, window_size=5))

    def test_rule_replans_around_current_observed_occupancy(self):
        specs = (DynamicObstacleSpec(((3, 1), (2, 1)), 0, 1),)
        env = environment(specs=specs)
        state = env.reset()
        policy = ObservedAStarReplanWaitPolicy(env.problem)
        self.assertNotEqual(policy.select_action(state), int(Action.RIGHT))
        # Changing route/subgoal hints has no effect on this baseline.
        changed = replace(state, spatial=state.spatial.copy(), scalars=state.scalars.copy())
        changed.spatial[4] = 0
        changed.scalars[2:] = -1
        self.assertEqual(policy.select_action(state), policy.select_action(changed))

    def test_rule_waits_when_only_corridor_is_currently_blocked(self):
        obstacles = frozenset((row, column) for row in range(7) if row != 3 for column in range(7))
        p = problem(obstacles)
        specs = (DynamicObstacleSpec(((3, 1), (3, 2)), 0, 1),)
        env = PathGuidanceEnvironment(DynamicGridNavigationEnv(p, specs, window_size=5))
        self.assertEqual(ObservedAStarReplanWaitPolicy(p).select_action(env.reset()), int(Action.STAY))

    def test_registered_protocol_and_factory_reset(self):
        config = load_config(entry.CONFIG)
        inputs = entry.validate_config(config)
        p = inputs[0]
        factory = entry.make_factory(config, *inputs, method="path_guided", seed=0)
        first = factory(p, window_size=15)
        factory.reset_schedule()
        repeated = factory(p, window_size=15)
        self.assertEqual(first.scenario_id, repeated.scenario_id)
        self.assertEqual(first.dynamic_obstacles, repeated.dynamic_obstacles)
        self.assertEqual(first.problem.grid_sha256, config["dataset"]["grid_sha256"])
        changed = load_config(entry.CONFIG)
        changed["reward"]["goal"] = 12
        with self.assertRaises(ValueError):
            entry.validate_config(changed)

    def test_paired_networks_and_evaluation_preserve_state(self):
        torch.set_num_threads(1)
        config = load_config(entry.CONFIG)
        left, right = entry.make_agent(config, 0, "cpu"), entry.make_agent(config, 0, "cpu")
        self.assertEqual(entry.state_digest(left.training_state_dict()), entry.state_digest(right.training_state_dict()))
        before = entry.state_digest(left.training_state_dict())
        rng = torch.get_rng_state().clone()
        with entry.preserved_evaluation(left):
            state = PathGuidanceEnvironment(DynamicGridNavigationEnv(problem(), window_size=15)).reset()
            left.select_action(state, epsilon=0.0)
            torch.rand(3)
            left.policy_network.eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(left.policy_network.training)
        self.assertEqual(before, entry.state_digest(left.training_state_dict()))

    def test_eight_step_replay_optimizer_integration_for_both_inputs(self):
        torch.set_num_threads(1)
        config = load_config(entry.CONFIG)
        inputs = entry.validate_config(config)
        for method in entry.METHODS:
            agent = entry.make_agent(config, 0, "cpu")
            training = replace(entry._training_config(config, 0), max_environment_steps=8,
                               batch_size=4, learning_starts=4, epsilon_decay_environment_steps=8,
                               progress_interval_environment_steps=None)
            result = train_d3qn([inputs[0]], agent, training,
                               environment_factory=entry.make_factory(config, *inputs, method=method, seed=0),
                               reward_config=entry._reward_config(config))
            self.assertEqual(result.environment_steps, 8)
            self.assertEqual(result.gradient_updates, 5)
            self.assertEqual(result.episode_records[-1]["demo_retained_count"], 0)

    def test_no_progress_protocol_removes_only_distance_reward(self):
        config = load_config(entry.NO_PROGRESS_CONFIG)
        entry.validate_config(config)
        dense = load_config(entry.CONFIG)
        expected = dict(dense["reward"], progress=0.0)
        self.assertEqual(config["reward"], expected)
        for section in ("environment", "agent", "dataset", "dynamic_route_pool", "guidance"):
            self.assertEqual(config[section], dense[section])
        self.assertEqual({k: v for k, v in config["training"].items() if k != "seeds"},
                         {k: v for k, v in dense["training"].items() if k != "seeds"})
        self.assertEqual(config["training"]["seeds"], list(entry.REGISTERED_SEEDS))
        reward = entry._reward_config(config)
        for enabled in (False, True):
            env = PathGuidanceEnvironment(DynamicGridNavigationEnv(problem(), window_size=5,
                                           max_steps=30, reward_config=reward), enabled=enabled, lookahead_steps=2)
            # Approach, move away and wait all have the same step cost, while goal direction remains visible.
            for action in (Action.RIGHT, Action.UP, Action.STAY):
                state = env.reset()
                self.assertTrue(np.any(state.scalars[:2]))
                result = env.step(int(action))
                self.assertAlmostEqual(result.reward, -0.01)
                self.assertEqual(result.info["reward_progress"], 0.0)
            env.reset()
            for _ in range(6):
                arrived = env.step(int(Action.RIGHT))
            self.assertTrue(arrived.terminated)
            self.assertEqual(arrived.reward, 10.0)
            specs = (DynamicObstacleSpec(((2, 1), (3, 1)), 0, 1, move_every=1),)
            collision_env = PathGuidanceEnvironment(DynamicGridNavigationEnv(problem(), specs,
                                                   window_size=5, reward_config=reward,
                                                   terminate_on_collision=True), enabled=enabled)
            collision_env.reset()
            collision = collision_env.step(int(Action.RIGHT))
            self.assertTrue(collision.terminated)
            self.assertEqual(collision.reward, -1.0)
        for section, key, value in (("reward", "progress", 0.05), ("reward", "collision", -2.0),
                                    ("agent", "learning_rate", 0.001),
                                    ("experiment", "output_root", dense["experiment"]["output_root"])):
            changed = load_config(entry.NO_PROGRESS_CONFIG)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                entry.validate_config(changed)

    def test_no_progress_cli_and_optimizer_integration(self):
        config = load_config(entry.NO_PROGRESS_CONFIG)
        with patch.object(entry, "validate_config", return_value=(None, None, None)), \
             patch.object(entry, "run_training", return_value=Path("unused")) as train:
            entry.main(["--train", "--config", str(entry.NO_PROGRESS_CONFIG),
                        "--seeds", "0", "1", "2", "3", "4", "--methods", "unguided", "path_guided"])
            self.assertEqual(train.call_args.args[0], config)
            self.assertEqual(train.call_args.args[2], list(entry.REGISTERED_SEEDS))
            self.assertEqual(train.call_args.args[3], list(entry.METHODS))
            self.assertFalse(train.call_args.kwargs["smoke"])
        torch.set_num_threads(1)
        inputs = entry.validate_config(config)
        for method in entry.METHODS:
            agent = entry.make_agent(config, 0, "cpu")
            training = replace(entry._training_config(config, 0), max_environment_steps=8,
                               batch_size=4, learning_starts=4, epsilon_decay_environment_steps=8,
                               progress_interval_environment_steps=None)
            result = train_d3qn([inputs[0]], agent, training,
                               environment_factory=entry.make_factory(config, *inputs, method=method, seed=0),
                               reward_config=entry._reward_config(config))
            self.assertEqual(result.environment_steps, 8)
            self.assertEqual(result.gradient_updates, 5)
            self.assertEqual(result.episode_records[-1]["demo_retained_count"], 0)
        with self.assertRaises(SystemExit):
            entry.main(["--pilot", "--config", str(entry.NO_PROGRESS_CONFIG)])

    def test_default_cli_checks_only_and_train_requires_explicit_flag(self):
        with patch.object(entry, "validate_config", return_value=(None, None, None)), \
             patch.object(entry, "unique_check_dir", return_value=Path("unused")), \
             patch.object(entry, "check", return_value=({}, [])) as checks, \
             patch.object(entry, "write_json"), patch.object(entry, "run_training") as train:
            entry.main([])
            checks.assert_called_once()
            train.assert_not_called()
        with patch.object(entry, "validate_config", return_value=(None, None, None)), \
             patch.object(entry, "run_training", return_value=Path("unused")) as train:
            entry.main(["--train", "--seeds", "0", "--methods", "path_guided"])
            train.assert_called_once()
            self.assertFalse(train.call_args.kwargs["smoke"])

    def test_pilot_has_separate_budget_schedule_and_output(self):
        config = load_config(entry.PILOT_CONFIG)
        entry.validate_config(config)
        training = entry._training_config(config, 0)
        self.assertEqual(training.max_environment_steps, 50000)
        self.assertEqual(training.epsilon_decay_environment_steps, 37500)
        self.assertTrue(config["experiment"]["output_root"].endswith("runtime_path_pilot_v1"))
        with patch.object(entry, "validate_config", return_value=(None, None, None)), \
             patch.object(entry, "run_training", return_value=Path("unused")) as train:
            entry.main(["--pilot", "--seeds", "1"])
            args = train.call_args.args
            self.assertEqual(args[0]["training"]["max_environment_steps"], 50000)
            self.assertEqual(args[2], [1])
            self.assertEqual(args[3], list(entry.METHODS))
            self.assertFalse(train.call_args.kwargs["smoke"])
        with self.assertRaises(SystemExit):
            entry.main(["--train", "--config", str(entry.PILOT_CONFIG)])
        with self.assertRaises(SystemExit):
            entry.main(["--pilot", "--config", str(entry.CONFIG)])

    def test_additional_seeds_preserve_formal_protocol_and_both_methods(self):
        with patch.object(entry, "validate_config", return_value=(None, None, None)), \
             patch.object(entry, "run_training", return_value=Path("unused")) as train:
            entry.main(["--train", "--seeds", "2", "3", "4", "--methods", "unguided", "path_guided"])
            config, _, seeds, methods = train.call_args.args
            self.assertEqual(config, load_config(entry.CONFIG))
            self.assertEqual(config["training"]["max_environment_steps"], 200000)
            self.assertEqual(seeds, [2, 3, 4])
            self.assertEqual(methods, ["unguided", "path_guided"])
            self.assertFalse(train.call_args.kwargs["smoke"])
        for seeds in (["2", "2"], ["5"], ["-1"]):
            with self.assertRaises(SystemExit), patch.object(entry, "run_training") as train:
                entry.main(["--train", "--seeds", *seeds])
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
