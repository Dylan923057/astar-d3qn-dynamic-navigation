from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_training_handover as entry
from astar_d3qn.agents.astar_training_handover import AStarTrainingHandoverAgent, METHODS
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.utils.seed import seed_everything


class HandoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = NavigationProblem("test", 0, 7, (3, 0), (3, 6), frozenset(),
                                        tuple(astar_path((3, 0), (3, 6), frozenset(), 7)))

    def env(self):
        return PathGuidanceEnvironment(DynamicGridNavigationEnv(
            self.problem, (), window_size=5, max_steps=30, terminate_on_collision=True), enabled=False)

    def agent(self, method, **kwargs):
        seed_everything(0)
        if method == "astar_budget_matched" and "advice_quota" not in kwargs:
            kwargs["advice_quota"] = [0] * 20
        return AStarTrainingHandoverAgent(D3QNConfig(spatial_shape=(5, 5, 5), scalar_dim=4,
                                                  action_dim=5, hidden_dim=16, device="cpu", seed=0),
                                         self.problem, method=method, **kwargs)

    def test_initial_network_optimizer_and_rng_identical(self):
        hashes = [entry.runtime.state_digest(self.agent(method).training_state_dict()) for method in METHODS]
        self.assertEqual(len(set(hashes)), 1)

    def test_unguided_matches_original_exploration_and_rng(self):
        env = self.env()
        state = env.reset()
        handover = self.agent("unguided")
        seed_everything(0)
        ordinary = D3QNAgent(handover.config)
        for epsilon in (1, 0.5, 0.05, 0):
            valid = list(np.flatnonzero(env.action_mask(True)))
            self.assertEqual(handover.select_action(state, epsilon, valid),
                             ordinary.select_action(state, epsilon, valid))
            self.assertEqual(handover._rng.getstate(), ordinary._rng.getstate())

    def test_full_teacher_reaches_static_goal_and_off_path_replans(self):
        env = self.env()
        state = env.reset()
        agent = self.agent("astar_fixed", probability=1)
        state = env.step(int(Action.UP)).observation
        for _ in range(20):
            action = agent.select_action(state, epsilon=1, valid_actions=list(np.flatnonzero(env.action_mask(True))))
            result = env.step(action)
            state = result.observation
            if result.done:
                break
        self.assertEqual(env.position, self.problem.goal)
        self.assertEqual(result.info["termination_reason"], "goal")
        self.assertFalse(result.info["collision"])
        self.assertTrue(all(row["applied"] == row["steps"] for row in agent.advice_records()))

    def test_risk_handover_returns_learner_action_without_action_masking(self):
        state = self.env().reset()
        # Current occupancy directly on the advised RIGHT cell.
        state.spatial[1, 2, 3] = 1
        agent = self.agent("astar_risk_decay", probability=1)
        with patch.object(D3QNAgent, "select_action", return_value=int(Action.UP)):
            self.assertEqual(agent.select_action(state, epsilon=0.5), int(Action.UP))
        self.assertEqual(agent.advice_records()[0]["risk_veto"], 1)
        self.assertEqual(agent.advice_records()[0]["applied"], 0)
        # The time-only control ignores this veto and takes exactly the same advice.
        control = self.agent("astar_decay", probability=1)
        with patch.object(D3QNAgent, "select_action", return_value=int(Action.UP)):
            self.assertEqual(control.select_action(state, epsilon=0.5), int(Action.RIGHT))

    def test_recent_history_and_next_cell_neighbors_are_used(self):
        agent = self.agent("astar_risk_decay")
        state = self.env().reset()
        self.assertFalse(agent.observed_risk(state, int(Action.RIGHT)))
        state.spatial[3, 1, 3] = 1
        self.assertTrue(agent.observed_risk(state, int(Action.RIGHT)))
        state.spatial[:] = 0
        state.spatial[1, 0, 0] = 1
        self.assertFalse(agent.observed_risk(state, int(Action.RIGHT)))

    def test_greedy_evaluation_never_calls_planner_or_advances_state(self):
        agent = self.agent("astar_fixed", probability=1)
        state = self.env().reset()
        before = entry.runtime.state_digest(agent.training_state_dict())
        with patch.object(agent, "static_advice", side_effect=AssertionError("Planner invoked in eval")):
            with entry.runtime.preserved_evaluation(agent):
                agent.select_action(state, epsilon=0)
        self.assertEqual(entry.runtime.state_digest(agent.training_state_dict()), before)

    def test_exact_advice_withdrawal_at_100k_and_no_clock_compression(self):
        for method in ("astar_decay", "astar_risk_decay"):
            agent = self.agent(method)
            self.assertAlmostEqual(agent.advice_probability(50000), 0.4)
            self.assertEqual(agent.advice_probability(100000), 0)
            self.assertEqual(agent.advice_probability(200000), 0)
            agent.training_action_steps = 99999
            state = self.env().reset()
            with patch.object(D3QNAgent, "select_action", return_value=int(Action.STAY)):
                self.assertEqual(agent.select_action(state, 0.05), int(Action.STAY))
            self.assertEqual(agent.advice_records()[0]["applied"], 0)
        self.assertEqual(self.agent("astar_fixed").advice_probability(200000), 0.8)

    def test_advice_state_can_be_restored(self):
        agent = self.agent("astar_risk_decay")
        state = self.env().reset()
        for _ in range(9):
            agent.select_action(state, 0.5)
        saved = copy.deepcopy(agent.training_state_dict())
        restored = self.agent("astar_risk_decay")
        restored.load_training_state_dict(saved)
        self.assertEqual(entry.runtime.state_digest(saved), entry.runtime.state_digest(restored.training_state_dict()))
        self.assertEqual(agent.select_action(state, 0.5), restored.select_action(state, 0.5))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.agent("unknown")
        with self.assertRaises(ValueError):
            self.agent("astar_decay", probability=2)

    def test_matched_budget_is_exact_and_timing_independent_of_observed_risk(self):
        agent = self.agent("astar_budget_matched", advice_quota=[37, 21] + [0] * 18)
        self.assertEqual(sum(1 <= step <= 10000 for step in agent._quota_steps), 37)
        self.assertEqual(sum(10001 <= step <= 20000 for step in agent._quota_steps), 21)
        self.assertEqual(agent._quota_steps, self.agent("astar_budget_matched", advice_quota=[37, 21] + [0] * 18)._quota_steps)
        state = self.env().reset()
        state.spatial[1, 2, 3] = 1
        agent.training_action_steps = min(agent._quota_steps) - 1
        with patch.object(D3QNAgent, "select_action", return_value=int(Action.UP)):
            self.assertEqual(agent.select_action(state, 0.5), int(Action.RIGHT))
        self.assertEqual(agent.advice_records()[0]["risk_veto"], 0)

    def test_actual_matched_action_counts_across_two_full_bins(self):
        agent = self.agent("astar_budget_matched", advice_quota=[37, 21] + [0] * 18)
        state = self.env().reset()
        with patch.object(D3QNAgent, "select_action", return_value=int(Action.UP)):
            for _ in range(20000):
                agent.select_action(state, 0.5)
        self.assertEqual([row["applied"] for row in agent.advice_records()], [37, 21])
        self.assertEqual(agent.training_action_steps, 20000)

    def test_default_entry_does_not_start_training(self):
        config = entry.load_config(entry.CONFIG)
        with patch.object(entry, "validate_config", return_value=None), \
             patch.object(entry, "check", return_value=({}, [])), \
             patch.object(entry.runtime, "unique_check_dir", return_value=Path("unused")), \
             patch.object(entry, "write_json"), \
             patch.object(entry, "run_training") as training:
            entry.main([])
            training.assert_not_called()

    def test_bad_matched_quota_is_rejected(self):
        with self.assertRaises(ValueError):
            self.agent("astar_budget_matched", advice_quota=[10001] * 20)


if __name__ == "__main__":
    unittest.main()
