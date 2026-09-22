from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.grid import Action
from astar_d3qn.envs.static_grid import StaticGridNavigationEnv
from astar_d3qn.envs.types import Observation
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.demo import (
    LocalConflictDemoReplay,
    LocalCounterexampleDemoReplay,
    PersistentDemoReplay,
)
from astar_d3qn.replay.transition import Transition
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.training.trainer import (
    ConflictAdaptiveDemoController,
    TrainingConfig,
    _predictive_conflict_margin_masks,
    _safe_demo_guidance,
    build_replay,
    demo_fraction_at_episode,
    demo_fraction_at_environment_step,
    linear_epsilon,
    linear_epsilon_at_environment_step,
    train_d3qn,
)


def demos(count: int) -> list[Transition]:
    state = Observation(
        spatial=np.zeros((1, 5, 5), dtype=np.float32),
        scalars=np.zeros(2, dtype=np.float32),
    )
    return [
        Transition(state, index % 5, 0.0, state, False, np.ones(5, dtype=bool))
        for index in range(count)
    ]


class EpsilonRecordingAgent:
    def __init__(self) -> None:
        self.config = SimpleNamespace(spatial_shape=(1, 5, 5), scalar_dim=2)
        self.action_dim = 5
        self.epsilons: list[float] = []

    def select_action(self, state: Observation, epsilon: float) -> int:
        self.epsilons.append(epsilon)
        return int(Action.STAY)


class AlwaysRightConflictEnvironment(StaticGridNavigationEnv):
    def dynamic_action_collision_risk(
        self,
        action: int,
        *,
        predict_next: bool,
    ) -> bool:
        del predict_next
        return int(action) == int(Action.RIGHT)


class TrainingStrategyTests(unittest.TestCase):
    def config(self, strategy: str) -> TrainingConfig:
        return TrainingConfig(
            episodes=2,
            max_steps=10,
            replay_capacity=20,
            batch_size=4,
            epsilon_decay_episodes=100,
            replay_strategy=strategy,
            window_size=5,
        )

    def test_only_prefill_starts_uniform_replay_with_demos(self) -> None:
        uniform = build_replay(self.config("uniform"), demos(5))
        prefill = build_replay(self.config("prefill"), demos(5))
        self.assertIsInstance(uniform, UniformReplayBuffer)
        self.assertIsInstance(prefill, UniformReplayBuffer)
        self.assertEqual(len(uniform), 0)
        self.assertEqual(len(prefill), 5)

    def test_per_and_dqfd_replay_strategies_are_constructible(self) -> None:
        per = build_replay(self.config("per"), ())
        dqfd = build_replay(self.config("dqfd"), demos(5))
        self.assertEqual(len(per), 0)
        self.assertEqual(len(dqfd), 5)

    def test_persistent_strategy_uses_separate_demo_partition(self) -> None:
        replay = build_replay(self.config("persistent_demo"), demos(5))
        self.assertIsInstance(replay, PersistentDemoReplay)
        assert isinstance(replay, PersistentDemoReplay)
        self.assertEqual(replay.demonstration_size, 5)
        self.assertEqual(replay.online_size, 0)
        for transition in demos(15):
            replay.add(transition)
        self.assertEqual(len(replay), self.config("persistent_demo").replay_capacity)

    def test_conflict_adaptive_strategy_uses_persistent_partition(self) -> None:
        replay = build_replay(self.config("conflict_adaptive_demo"), demos(5))
        self.assertIsInstance(replay, PersistentDemoReplay)

    def test_local_conflict_strategy_requires_keys_and_uses_local_replay(self) -> None:
        config = self.config("local_conflict_demo")
        items = demos(5)
        with self.assertRaisesRegex(ValueError, "position-action key"):
            build_replay(config, items)
        replay = build_replay(
            config,
            items,
            [((1, index), item.action) for index, item in enumerate(items)],
        )
        self.assertIsInstance(replay, LocalConflictDemoReplay)
        self.assertEqual(replay.demonstration_size, 5)

    def test_local_counterexample_strategy_uses_dedicated_replay(self) -> None:
        config = self.config("local_counterexample_demo")
        items = demos(5)
        replay = build_replay(
            config,
            items,
            [((1, index), item.action) for index, item in enumerate(items)],
        )
        self.assertIsInstance(replay, LocalCounterexampleDemoReplay)
        self.assertEqual(replay.demonstration_size, 5)
        self.assertEqual(replay.counterexample_size, 0)

    def test_predictive_margin_strategy_keeps_persistent_demo_partition(self) -> None:
        replay = build_replay(self.config("predictive_margin_demo"), demos(5))
        self.assertIsInstance(replay, PersistentDemoReplay)
        self.assertNotIsInstance(replay, LocalCounterexampleDemoReplay)

    def test_predictive_margin_masks_blocked_demo_and_keeps_safe_alternatives(self) -> None:
        problem = NavigationProblem(
            map_id="predictive_margin_masks",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1)),
        )
        env = AlwaysRightConflictEnvironment(problem, max_steps=3, window_size=5)
        env.reset()
        safe, blocked, covered, conflict, no_safe, risky_fraction = (
            _predictive_conflict_margin_masks(
                env,
                {(0, 0): (int(Action.RIGHT),)},
                predict_next=True,
            )
        )
        self.assertTrue(covered)
        self.assertTrue(conflict)
        self.assertFalse(no_safe)
        self.assertEqual(risky_fraction, 1.0)
        assert safe is not None and blocked is not None
        self.assertTrue(blocked[int(Action.RIGHT)])
        self.assertFalse(safe[int(Action.RIGHT)])
        self.assertTrue(safe[int(Action.DOWN)])
        self.assertTrue(safe[int(Action.STAY)])

    def test_training_registers_safe_wait_as_local_counterexample(self) -> None:
        problem = NavigationProblem(
            map_id="local_counterexample_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        probe = AlwaysRightConflictEnvironment(problem, max_steps=3, window_size=5)
        state = probe.reset()
        demonstration = Transition(
            state,
            int(Action.RIGHT),
            0.0,
            state,
            False,
            np.ones(5, dtype=bool),
        )
        config = TrainingConfig(
            episodes=1,
            max_steps=3,
            replay_capacity=20,
            batch_size=4,
            learning_starts=1000,
            replay_strategy="local_counterexample_demo",
            window_size=5,
        )

        result = train_d3qn(
            [problem],
            EpsilonRecordingAgent(),  # type: ignore[arg-type]
            config,
            demonstrations=[demonstration],
            environment_factory=AlwaysRightConflictEnvironment,
        )

        record = result.episode_records[0]
        self.assertEqual(record["local_counterexample_candidate_count"], 3)
        self.assertEqual(record["local_counterexample_added_count"], 3)
        self.assertEqual(record["local_counterexample_wait_count"], 3)
        self.assertEqual(record["local_counterexample_buffer_size"], 3)

    def test_training_labels_conflict_states_without_extra_replay(self) -> None:
        problem = NavigationProblem(
            map_id="predictive_margin_training_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        probe = AlwaysRightConflictEnvironment(problem, max_steps=3, window_size=5)
        state = probe.reset()
        demonstration = Transition(
            state,
            int(Action.RIGHT),
            0.0,
            state,
            False,
            np.ones(5, dtype=bool),
        )
        config = TrainingConfig(
            episodes=1,
            max_steps=3,
            replay_capacity=20,
            batch_size=4,
            learning_starts=1000,
            replay_strategy="predictive_margin_demo",
            window_size=5,
        )

        result = train_d3qn(
            [problem],
            EpsilonRecordingAgent(),  # type: ignore[arg-type]
            config,
            demonstrations=[demonstration],
            environment_factory=AlwaysRightConflictEnvironment,
        )

        record = result.episode_records[0]
        self.assertEqual(record["conflict_margin_covered_count"], 3)
        self.assertEqual(record["conflict_margin_conflict_count"], 3)
        self.assertEqual(record["conflict_margin_labeled_count"], 3)
        self.assertEqual(record["conflict_margin_no_safe_count"], 0)
        self.assertEqual(record["local_counterexample_buffer_size"], 0)

    def test_safe_guided_strategy_uses_evictable_prefill(self) -> None:
        replay = build_replay(self.config("safe_guided_demo"), demos(5))
        self.assertIsInstance(replay, UniformReplayBuffer)
        self.assertEqual(len(replay), 5)

    def test_safe_guidance_selects_safe_majority_and_skips_all_blocked(self) -> None:
        class RiskEnvironment:
            position = (2, 3)

            @staticmethod
            def dynamic_action_collision_risk(action: int, *, predict_next: bool) -> bool:
                self_predict = predict_next
                return action == 1 or (self_predict and action == 4)

        actions = {(2, 3): (1, 2, 2, 4)}
        selected, covered, all_blocked, risky_fraction = _safe_demo_guidance(
            RiskEnvironment(), actions, predict_next=True  # type: ignore[arg-type]
        )
        self.assertEqual(selected, 2)
        self.assertTrue(covered)
        self.assertFalse(all_blocked)
        self.assertEqual(risky_fraction, 0.5)

        selected, covered, all_blocked, risky_fraction = _safe_demo_guidance(
            RiskEnvironment(), {(2, 3): (1, 4)}, predict_next=True  # type: ignore[arg-type]
        )
        self.assertIsNone(selected)
        self.assertTrue(covered)
        self.assertTrue(all_blocked)
        self.assertEqual(risky_fraction, 1.0)

    def test_conflict_controller_reduces_and_recovers_demo_fraction(self) -> None:
        controller = ConflictAdaptiveDemoController(
            maximum=0.25,
            minimum=0.0,
            ema_alpha=0.5,
            sensitivity=1.0,
        )
        self.assertEqual(controller.demo_fraction, 0.25)
        self.assertAlmostEqual(controller.observe(1.0), 0.125)
        self.assertAlmostEqual(controller.observe(0.0), 0.1875)

    def test_event_controller_reacts_fast_and_recovers_slowly(self) -> None:
        controller = ConflictAdaptiveDemoController(
            maximum=0.25,
            minimum=0.05,
            ema_alpha=0.5,
            recovery_alpha=0.005,
            sensitivity=1.0,
            event_binary=True,
        )
        self.assertAlmostEqual(controller.observe(0.01), 0.15)
        self.assertAlmostEqual(controller.conflict_ema, 0.5)
        self.assertAlmostEqual(controller.observe(0.0), 0.1505)
        self.assertAlmostEqual(controller.conflict_ema, 0.4975)

    def test_linear_epsilon_reaches_registered_floor(self) -> None:
        config = self.config("uniform")
        self.assertEqual(linear_epsilon(0, config), config.epsilon_start)
        self.assertAlmostEqual(
            linear_epsilon(config.epsilon_decay_episodes, config), config.epsilon_end
        )
        self.assertAlmostEqual(
            linear_epsilon(config.epsilon_decay_episodes * 2, config), config.epsilon_end
        )

    def test_fixed_step_budget_requires_a_step_based_epsilon_schedule(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "epsilon_decay_environment_steps",
        ):
            TrainingConfig(max_environment_steps=10)

    def test_step_based_epsilon_reaches_registered_floor(self) -> None:
        config = TrainingConfig(
            max_environment_steps=10,
            epsilon_decay_environment_steps=8,
        )
        self.assertEqual(
            linear_epsilon_at_environment_step(0, config),
            config.epsilon_start,
        )
        self.assertAlmostEqual(
            linear_epsilon_at_environment_step(8, config),
            config.epsilon_end,
        )
        self.assertAlmostEqual(
            linear_epsilon_at_environment_step(20, config),
            config.epsilon_end,
        )

    def test_demo_fraction_linearly_decays_to_registered_final_value(self) -> None:
        config = TrainingConfig(
            episodes=10,
            replay_strategy="persistent_demo",
            demo_fraction=0.25,
            demo_fraction_final=0.0,
            demo_fraction_decay_start_episode=2,
            demo_fraction_decay_end_episode=6,
        )

        self.assertEqual(demo_fraction_at_episode(1, config), 0.25)
        self.assertEqual(demo_fraction_at_episode(2, config), 0.25)
        self.assertAlmostEqual(demo_fraction_at_episode(4, config), 0.125)
        self.assertEqual(demo_fraction_at_episode(6, config), 0.0)
        self.assertEqual(demo_fraction_at_episode(10, config), 0.0)

    def test_demo_fraction_can_decay_on_environment_step_clock(self) -> None:
        config = TrainingConfig(
            max_environment_steps=100,
            epsilon_decay_environment_steps=80,
            replay_strategy="persistent_demo",
            demo_fraction=0.25,
            demo_fraction_final=0.0,
            demo_fraction_decay_start_environment_step=20,
            demo_fraction_decay_end_environment_step=60,
        )

        self.assertEqual(demo_fraction_at_environment_step(20, config), 0.25)
        self.assertAlmostEqual(
            demo_fraction_at_environment_step(40, config),
            0.125,
        )
        self.assertEqual(demo_fraction_at_environment_step(60, config), 0.0)

    def test_demo_decay_rejects_mixed_episode_and_step_clocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "either episodes or environment steps"):
            TrainingConfig(
                episodes=10,
                max_environment_steps=100,
                epsilon_decay_environment_steps=80,
                demo_fraction_final=0.0,
                demo_fraction_decay_start_episode=1,
                demo_fraction_decay_end_episode=5,
                demo_fraction_decay_start_environment_step=20,
                demo_fraction_decay_end_environment_step=60,
            )

    def test_epsilon_is_constant_within_each_episode(self) -> None:
        problem = NavigationProblem(
            map_id="epsilon_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        config = TrainingConfig(
            episodes=2,
            max_steps=3,
            replay_capacity=20,
            batch_size=4,
            learning_starts=1000,
            epsilon_decay_episodes=2,
            replay_strategy="uniform",
            window_size=5,
        )
        agent = EpsilonRecordingAgent()

        result = train_d3qn([problem], agent, config)  # type: ignore[arg-type]

        self.assertEqual(len(result.episode_records), 2)
        self.assertEqual(agent.epsilons[:3], [1.0, 1.0, 1.0])
        self.assertEqual(agent.epsilons[3:], [0.525, 0.525, 0.525])
        self.assertTrue(
            all(row["wait_steps"] == 3 for row in result.episode_records)
        )
        self.assertTrue(
            all(row["revisit_count"] == 0 for row in result.episode_records)
        )
        self.assertTrue(
            all(row["static_collision_count"] == 0 for row in result.episode_records)
        )
        self.assertTrue(
            all(row["dynamic_collision_count"] == 0 for row in result.episode_records)
        )
        self.assertTrue(
            all(row["dynamic_obstacle_count"] == 0 for row in result.episode_records)
        )
        self.assertTrue(
            all(row["scenario_id"] is None for row in result.episode_records)
        )
        self.assertGreaterEqual(result.training_seconds, 0.0)

    def test_progress_callback_runs_at_registered_interval(self) -> None:
        problem = NavigationProblem(
            map_id="progress_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        config = TrainingConfig(
            episodes=3,
            max_steps=2,
            replay_capacity=20,
            batch_size=4,
            learning_starts=1000,
            replay_strategy="uniform",
            window_size=5,
            progress_interval=2,
        )
        calls: list[tuple[int, int, bool]] = []
        train_d3qn(
            [problem],
            EpsilonRecordingAgent(),  # type: ignore[arg-type]
            config,
            progress_callback=lambda episode, records, full_evaluation: calls.append(
                (episode, len(records), full_evaluation)
            ),
        )
        self.assertEqual(calls, [(2, 2, True)])

    def test_progress_callback_can_stop_training_cleanly(self) -> None:
        problem = NavigationProblem(
            map_id="callback_stop_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        config = TrainingConfig(
            episodes=10,
            max_steps=2,
            replay_capacity=20,
            batch_size=4,
            learning_starts=1000,
            replay_strategy="uniform",
            window_size=5,
            progress_interval=1,
        )

        result = train_d3qn(
            [problem],
            EpsilonRecordingAgent(),  # type: ignore[arg-type]
            config,
            progress_callback=lambda episode, records, full_evaluation: False,
        )

        self.assertTrue(result.stopped_early)
        self.assertEqual(len(result.episode_records), 1)
        self.assertEqual(result.environment_steps, 2)

    def test_fixed_environment_step_budget_stops_exactly_and_drives_progress(self) -> None:
        problem = NavigationProblem(
            map_id="step_budget_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        config = TrainingConfig(
            episodes=1,
            max_environment_steps=5,
            max_steps=3,
            replay_capacity=20,
            batch_size=4,
            learning_starts=1000,
            epsilon_decay_episodes=2,
            epsilon_decay_environment_steps=4,
            replay_strategy="uniform",
            window_size=5,
            progress_interval=100,
            progress_interval_environment_steps=2,
        )
        agent = EpsilonRecordingAgent()
        calls: list[tuple[int, int, int, bool]] = []

        result = train_d3qn(
            [problem],
            agent,  # type: ignore[arg-type]
            config,
            progress_callback=lambda episode, records, full_evaluation: calls.append(
                (
                    episode,
                    len(records),
                    int(records[-1]["environment_steps_total"]),
                    full_evaluation,
                )
            ),
        )

        self.assertEqual(result.environment_steps, 5)
        self.assertEqual(len(result.episode_records), 2)
        self.assertEqual([row["steps"] for row in result.episode_records], [3, 2])
        for actual, expected in zip(
            agent.epsilons,
            [1.0, 0.7625, 0.525, 0.2875, 0.05],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(calls, [(1, 1, 3, True), (2, 2, 5, True)])
        self.assertEqual(result.episode_records[-1]["training_budget_reached"], 1.0)
        self.assertFalse(result.stopped_early)


if __name__ == "__main__":
    unittest.main()
