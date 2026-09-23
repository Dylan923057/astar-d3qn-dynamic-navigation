from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import (
    D3QNAgent,
    D3QNConfig,
    double_dqn_bootstrap,
    double_dqn_targets,
)
from astar_d3qn.agents.networks import DuelingQNetwork
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.transition import Transition


class DuelingNetworkTests(unittest.TestCase):
    def test_dueling_aggregation_has_value_equal_to_mean_q(self) -> None:
        torch.manual_seed(3)
        network = DuelingQNetwork((1, 7, 7), 2, action_dim=5, hidden_dim=32)
        spatial = torch.randn(4, 1, 7, 7)
        scalars = torch.randn(4, 2)
        value, _ = network.streams(spatial, scalars)
        q_values = network(spatial, scalars)
        self.assertEqual(tuple(q_values.shape), (4, 5))
        torch.testing.assert_close(q_values.mean(dim=1, keepdim=True), value)


class DoubleDqnTests(unittest.TestCase):
    def test_policy_selects_and_target_evaluates(self) -> None:
        policy = torch.tensor([[1.0, 9.0, 3.0], [8.0, 2.0, 1.0]])
        target = torch.tensor([[7.0, 2.0, 6.0], [1.0, 5.0, 4.0]])
        actual = double_dqn_bootstrap(policy, target)
        torch.testing.assert_close(actual, torch.tensor([2.0, 1.0]))

    def test_action_mask_is_applied_before_policy_argmax(self) -> None:
        policy = torch.tensor([[1.0, 9.0, 3.0]])
        target = torch.tensor([[7.0, 2.0, 6.0]])
        mask = torch.tensor([[True, False, True]])
        actual = double_dqn_bootstrap(policy, target, mask)
        torch.testing.assert_close(actual, torch.tensor([6.0]))

    def test_terminal_transition_does_not_bootstrap(self) -> None:
        rewards = torch.tensor([2.0, 3.0])
        terminated = torch.tensor([1.0, 0.0])
        policy = torch.tensor([[0.0, 5.0], [4.0, 1.0]])
        target = torch.tensor([[8.0, 7.0], [6.0, 9.0]])
        actual = double_dqn_targets(
            rewards, terminated, 0.5, policy, target
        )
        torch.testing.assert_close(actual, torch.tensor([2.0, 6.0]))


class D3QNAgentTests(unittest.TestCase):
    def make_agent(self, target_sync_interval: int = 10) -> D3QNAgent:
        return D3QNAgent(
            D3QNConfig(
                spatial_shape=(1, 7, 7),
                scalar_dim=2,
                action_dim=5,
                hidden_dim=32,
                target_sync_interval=target_sync_interval,
                device="cpu",
                seed=5,
            )
        )

    def make_batch(self) -> list[Transition]:
        mask = np.ones(5, dtype=bool)
        return [
            Transition(
                state=Observation(
                    spatial=np.full((1, 7, 7), index % 2, dtype=np.float32),
                    scalars=np.asarray([0.5, -0.5], dtype=np.float32),
                ),
                action=index % 5,
                reward=float(index),
                next_state=Observation(
                    spatial=np.full(
                        (1, 7, 7), (index + 1) % 2, dtype=np.float32
                    ),
                    scalars=np.asarray([0.4, -0.4], dtype=np.float32),
                ),
                terminated=index == 3,
                next_action_mask=mask,
            )
            for index in range(4)
        ]

    def test_train_batch_updates_policy_and_returns_finite_stats(self) -> None:
        agent = self.make_agent()
        before = [parameter.detach().clone() for parameter in agent.policy_network.parameters()]
        stats = agent.train_batch(self.make_batch())
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(before, agent.policy_network.parameters())
            )
        )

    def test_target_sync_uses_configured_update_interval(self) -> None:
        agent = self.make_agent(target_sync_interval=1)
        agent.train_batch(self.make_batch())
        for policy, target in zip(
            agent.policy_network.parameters(), agent.target_network.parameters()
        ):
            torch.testing.assert_close(policy, target)

    def test_demo_margin_loss_accepts_demo_mask(self) -> None:
        agent = self.make_agent()
        stats = agent.train_batch(
            self.make_batch(),
            demonstration_mask=[True, True, False, False],
            demo_margin=0.8,
            demo_loss_weight=1.0,
        )
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertGreaterEqual(stats["demo_margin_loss"], 0.0)
        self.assertEqual(len(stats["td_errors"]), 4)

    def test_safe_guidance_margin_uses_only_labeled_online_states(self) -> None:
        agent = self.make_agent()
        batch = self.make_batch()
        batch[0] = replace(batch[0], safe_demo_action=3)
        batch[2] = replace(batch[2], safe_demo_action=1)

        stats = agent.train_batch(
            batch,
            safe_guidance_margin=0.8,
            safe_guidance_loss_weight=1.0,
        )

        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertGreaterEqual(stats["safe_guidance_margin_loss"], 0.0)
        self.assertEqual(stats["safe_guidance_batch_count"], 2)

    def test_conflict_margin_compares_best_safe_with_blocked_demo_action(self) -> None:
        agent = self.make_agent()
        batch = self.make_batch()
        safe = np.asarray([True, False, False, False, True], dtype=bool)
        blocked = np.asarray([False, False, False, True, False], dtype=bool)
        batch[0] = replace(
            batch[0],
            conflict_safe_action_mask=safe,
            conflict_blocked_action_mask=blocked,
        )
        batch[2] = replace(
            batch[2],
            conflict_safe_action_mask=safe,
            conflict_blocked_action_mask=blocked,
        )

        stats = agent.train_batch(
            batch,
            conflict_margin=0.8,
            conflict_margin_loss_weight=1.0,
        )

        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertGreaterEqual(stats["conflict_margin_loss"], 0.0)
        self.assertEqual(stats["conflict_margin_batch_count"], 2)

    def test_zero_conflict_margin_weight_is_exactly_equivalent_to_td_only(self) -> None:
        reference = self.make_agent()
        treatment = self.make_agent()
        treatment.load_training_state_dict(copy.deepcopy(reference.training_state_dict()))
        plain_batch = self.make_batch()
        labeled_batch = list(plain_batch)
        safe = np.asarray([True, False, False, False, True], dtype=bool)
        blocked = np.asarray([False, True, False, True, False], dtype=bool)
        labeled_batch[0] = replace(
            labeled_batch[0],
            conflict_safe_action_mask=safe,
            conflict_blocked_action_mask=blocked,
        )

        plain_stats = reference.train_batch(plain_batch)
        labeled_stats = treatment.train_batch(
            labeled_batch,
            conflict_margin=0.8,
            conflict_margin_loss_weight=0.0,
        )

        self.assertEqual(plain_stats["loss"], labeled_stats["loss"])
        self.assertEqual(labeled_stats["conflict_margin_batch_count"], 0)
        for plain, labeled in zip(
            reference.policy_network.parameters(),
            treatment.policy_network.parameters(),
        ):
            torch.testing.assert_close(plain, labeled, rtol=0.0, atol=0.0)

    def test_greedy_action_respects_valid_action_subset(self) -> None:
        agent = self.make_agent()
        state = Observation(
            spatial=np.zeros((1, 7, 7), dtype=np.float32),
            scalars=np.zeros(2, dtype=np.float32),
        )
        action = agent.select_action(state, epsilon=0.0, valid_actions=[2, 4])
        self.assertIn(action, {2, 4})


if __name__ == "__main__":
    unittest.main()
