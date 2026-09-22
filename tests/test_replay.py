from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.replay.demo import (
    LocalConflictDemoReplay,
    LocalCounterexampleDemoReplay,
    PersistentDemoReplay,
)
from astar_d3qn.replay.prioritized import PrioritizedReplayBuffer
from astar_d3qn.replay.transition import Transition
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.envs.types import Observation


def transition(value: int, reward: float | None = None) -> Transition:
    state = Observation(
        spatial=np.full((1, 3, 3), value, dtype=np.float32),
        scalars=np.asarray([value, -value], dtype=np.float32),
    )
    next_state = Observation(
        spatial=state.spatial + 1,
        scalars=state.scalars / 2,
    )
    return Transition(
        state=state,
        action=value % 5,
        reward=float(value if reward is None else reward),
        next_state=next_state,
        terminated=False,
        next_action_mask=np.ones(5, dtype=bool),
    )


class UniformReplayTests(unittest.TestCase):
    def test_prefilled_items_can_be_evicted(self) -> None:
        replay = UniformReplayBuffer(capacity=3, seed=1)
        replay.extend([transition(0), transition(1), transition(2)])
        replay.add(transition(3))
        self.assertEqual([item.reward for item in replay.snapshot()], [1.0, 2.0, 3.0])

    def test_snapshot_reports_current_buffer_contents(self) -> None:
        replay = UniformReplayBuffer(capacity=3, seed=1)
        first = transition(0)
        replay.add(first)
        self.assertEqual(replay.snapshot(), (first,))

    def test_state_round_trip_preserves_future_sampling(self) -> None:
        original = UniformReplayBuffer(capacity=6, seed=9)
        original.extend(transition(index) for index in range(6))
        state = original.state_dict()
        expected = [item.reward for item in original.sample(4)]
        restored = UniformReplayBuffer(capacity=6, seed=999)
        restored.load_state_dict(state)
        actual = [item.reward for item in restored.sample(4)]
        self.assertEqual(actual, expected)


class PersistentDemoReplayTests(unittest.TestCase):
    def test_demo_partition_is_not_evicted_by_online_data(self) -> None:
        demos = [transition(index, reward=-1.0) for index in range(4)]
        replay = PersistentDemoReplay(demos, online_capacity=3, demo_fraction=0.25, seed=3)
        for index in range(20):
            replay.add(transition(index, reward=1.0))
        self.assertEqual(replay.demonstration_size, 4)
        self.assertEqual(replay.online_size, 3)
        self.assertEqual(len(replay.demonstration_snapshot()), 4)

    def test_snapshot_keeps_demo_partition_visible(self) -> None:
        demos = [transition(index, reward=-1.0) for index in range(2)]
        replay = PersistentDemoReplay(demos, online_capacity=2, demo_fraction=0.25, seed=3)
        replay.add(transition(8, reward=1.0))
        snapshot = replay.snapshot()
        self.assertEqual(len(snapshot), 3)
        self.assertEqual(snapshot[:2], tuple(demos))

    def test_batch_uses_registered_demo_fraction(self) -> None:
        demos = [transition(index, reward=-1.0) for index in range(8)]
        replay = PersistentDemoReplay(demos, online_capacity=20, demo_fraction=0.25, seed=4)
        for index in range(12):
            replay.add(transition(index, reward=1.0))
        batch = replay.sample(8)
        rewards = [item.reward for item in batch]
        self.assertEqual(rewards.count(-1.0), 2)
        self.assertEqual(rewards.count(1.0), 6)

    def test_demo_fraction_can_decay_to_zero(self) -> None:
        demos = [transition(index, reward=-1.0) for index in range(8)]
        replay = PersistentDemoReplay(demos, online_capacity=20, demo_fraction=0.25, seed=4)
        for index in range(12):
            replay.add(transition(index, reward=1.0))

        replay.set_demo_fraction(0.0)
        batch = replay.sample(8)

        self.assertTrue(all(item.reward == 1.0 for item in batch))


class LocalConflictDemoReplayTests(unittest.TestCase):
    def replay(self, *, suppression_steps: int = 3) -> LocalConflictDemoReplay:
        items = [transition(index, reward=-float(index + 1)) for index in range(4)]
        keys = (
            ((1, 1), 0),
            ((1, 1), 1),
            ((2, 2), 2),
            ((3, 3), 3),
        )
        replay = LocalConflictDemoReplay(
            items,
            keys,
            online_capacity=12,
            demo_fraction=0.5,
            suppression_steps=suppression_steps,
            minimum_sampling_weight=0.0,
            seed=7,
        )
        for index in range(8):
            replay.add(transition(index, reward=1.0))
        return replay

    def test_only_the_risky_local_action_is_suppressed(self) -> None:
        replay = self.replay()
        replay.observe_local_conflicts((1, 1), {0: True, 1: False})
        self.assertEqual(replay.suppressed_key_count, 1)
        self.assertEqual(replay.suppressed_transition_count, 1)
        for _ in range(20):
            rewards = [item.reward for item in replay.sample(4)]
            self.assertNotIn(-1.0, rewards)
            self.assertEqual(sum(reward < 0.0 for reward in rewards), 2)

    def test_suppression_expires_and_safe_observation_restores_immediately(self) -> None:
        replay = self.replay(suppression_steps=2)
        replay.observe_local_conflicts((1, 1), {0: True})
        self.assertEqual(replay.suppressed_key_count, 1)
        replay.observe_local_conflicts((9, 9), {})
        self.assertEqual(replay.suppressed_key_count, 1)
        replay.observe_local_conflicts((9, 9), {})
        self.assertEqual(replay.suppressed_key_count, 0)
        replay.observe_local_conflicts((1, 1), {0: True})
        replay.observe_local_conflicts((1, 1), {0: False})
        self.assertEqual(replay.suppressed_key_count, 0)

    def test_total_demo_fraction_stays_fixed_during_suppression(self) -> None:
        replay = self.replay()
        replay.observe_local_conflicts((1, 1), {0: True})
        batch = replay.sample(4)
        self.assertEqual(sum(item.reward < 0.0 for item in batch), 2)


class LocalCounterexampleDemoReplayTests(unittest.TestCase):
    def replay(self) -> LocalCounterexampleDemoReplay:
        items = [transition(index, reward=-float(index + 1)) for index in range(4)]
        replay = LocalCounterexampleDemoReplay(
            items,
            [((index, index), item.action) for index, item in enumerate(items)],
            online_capacity=12,
            demo_fraction=0.5,
            suppression_steps=3,
            minimum_sampling_weight=0.0,
            counterexample_capacity=3,
            counterexample_fraction=0.5,
            seed=11,
        )
        for index in range(8):
            replay.add(transition(index, reward=1.0))
        return replay

    def test_safe_counterexample_uses_reserved_online_batch_slot(self) -> None:
        replay = self.replay()
        counterexample = transition(99, reward=99.0)
        replay.add_counterexample(counterexample)

        batch = replay.sample(4)

        self.assertTrue(any(item is counterexample for item in batch))
        self.assertEqual(sum(item.reward < 0.0 for item in batch), 2)
        self.assertEqual(replay.counterexample_size, 1)
        self.assertEqual(replay.counterexample_total_added, 1)
        self.assertEqual(replay.counterexample_sample_fraction, 0.25)

    def test_empty_counterexample_ring_falls_back_to_uniform_online(self) -> None:
        replay = self.replay()

        batch = replay.sample(4)

        self.assertEqual(sum(item.reward < 0.0 for item in batch), 2)
        self.assertEqual(sum(item.reward == 1.0 for item in batch), 2)
        self.assertEqual(replay.counterexample_sample_fraction, 0.0)


class PrioritizedReplayTests(unittest.TestCase):
    def test_sampling_returns_normalized_importance_weights(self) -> None:
        replay = PrioritizedReplayBuffer(capacity=5, seed=4)
        items = [transition(index) for index in range(5)]
        replay.extend(items)
        batch = replay.sample(4)
        weights = replay.sample_weights()
        self.assertEqual(len(batch), 4)
        self.assertEqual(weights.shape, (4,))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(weights.max()), 1.0)

    def test_priority_update_changes_registered_transition_priority(self) -> None:
        replay = PrioritizedReplayBuffer(capacity=4, seed=2)
        items = [transition(index) for index in range(4)]
        replay.extend(items)
        replay.update_priorities([items[2]], [9.0])
        self.assertEqual(replay.priority_snapshot()[2], 9.0)

    def test_state_round_trip_preserves_priorities(self) -> None:
        original = PrioritizedReplayBuffer(capacity=4, seed=2)
        original.extend(transition(index) for index in range(4))
        original.update_priorities([original.snapshot()[1]], [3.0])
        state = original.state_dict()
        restored = PrioritizedReplayBuffer(capacity=4, seed=99)
        restored.load_state_dict(state)
        self.assertEqual(restored.priority_snapshot(), original.priority_snapshot())


if __name__ == "__main__":
    unittest.main()
