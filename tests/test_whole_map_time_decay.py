from __future__ import annotations

import sys
from dataclasses import replace
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from astar_d3qn.core.grid import Action
from astar_d3qn.envs.types import Observation
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.replay_adaptation import decay_demo_fraction
from astar_d3qn.training.trainer import TrainingConfig, TrainingWindowAdaptiveController, OnlineReplayCoverage, train_d3qn
from astar_d3qn.envs.static_grid import StaticGridNavigationEnv
from train_whole_map_route_pool_pilot import (
    _training_config,
    _validate_time_decay_comparison,
)


class _StayAgent:
    def __init__(self) -> None:
        self.config = SimpleNamespace(spatial_shape=(1, 5, 5), scalar_dim=2)
        self.action_dim = 5

    def select_action(self, _state, epsilon=0.0, valid_actions=None):
        del epsilon, valid_actions
        return int(Action.STAY)


def _demonstrations(count: int) -> list[Transition]:
    state = Observation(
        spatial=np.zeros((1, 5, 5), dtype=np.float32),
        scalars=np.zeros(2, dtype=np.float32),
    )
    return [
        Transition(state, index % 5, 0.0, state, False, np.ones(5, dtype=bool))
        for index in range(count)
    ]


class WholeMapTimeDecayTests(unittest.TestCase):
    def test_replay_diagnostics_do_not_change_training_or_sample_rng(self):
        class Agent(_StayAgent):
            def __init__(self):
                super().__init__()
                self.batches = []
            def train_batch(self, batch, **kwargs):
                self.batches.append([(item.action, item.reward, item.terminated) for item in batch])
                return {"loss": 0.0, "safe_guidance_margin_loss": 0.0,
                        "safe_guidance_batch_count": 0, "conflict_margin_loss": 0.0,
                        "conflict_margin_batch_count": 0}
        problem = NavigationProblem(map_id="diagnostic_invariance", seed=0, size=5,
                                    start=(0, 0), goal=(4, 4), obstacles=frozenset(),
                                    nominal_path=((0, 0), (0, 1)))
        config = TrainingConfig(episodes=10, max_environment_steps=15, max_steps=3,
                                replay_capacity=10, batch_size=4, learning_starts=1,
                                epsilon_decay_environment_steps=10, replay_strategy="persistent_demo", window_size=5)
        demos = _demonstrations(5)
        plain, diagnostic = Agent(), Agent()
        baseline = train_d3qn([problem], plain, config, demonstrations=demos)
        measured = train_d3qn([problem], diagnostic, replace(config, replay_diagnostics=True), demonstrations=demos)
        self.assertEqual(plain.batches, diagnostic.batches)
        for before, after in zip(baseline.episode_records, measured.episode_records):
            self.assertEqual(before, {key: after[key] for key in before})
        last = measured.episode_records[-1]
        self.assertEqual(last["actual_demo_samples_total"], measured.gradient_updates)
        self.assertEqual(last["actual_online_samples_total"], 3 * measured.gradient_updates)
        self.assertEqual(last["online_replay_size"], 5)
        self.assertEqual(last["actual_demo_batch_count_mean"], 1)

    def test_coverage_tracks_eviction_and_distinct_dynamic_observations(self):
        coverage = OnlineReplayCoverage(2)
        transitions = _demonstrations(3)
        coverage.add(transitions[0], "a", conflict=True, encounter=True)
        coverage.add(transitions[1], "a", conflict=False, encounter=True)
        coverage.add(transitions[2], "b", conflict=False, encounter=False)
        metrics = coverage.metrics()
        self.assertEqual(metrics["online_replay_size"], 2)
        self.assertEqual(metrics["online_unique_states"], 1)
        self.assertEqual(metrics["online_unique_state_actions"], 2)
        self.assertEqual(metrics["online_retained_conflict_transitions"], 0)
        self.assertEqual(metrics["online_retained_encounter_transitions"], 1)
        self.assertEqual(metrics["online_retained_scenarios"], 2)

    def _window(self, controller, step, successes, conflicts=10, nonconflict_failures=0):
        for i in range(conflicts + nonconflict_failures):
            controller.observe_interaction(step - 100 + i, completed=True,
                                           safe_success=i < successes, conflict=i < conflicts)
        controller.observe_interaction(step, completed=False, safe_success=True, conflict=True)
        return controller.records[-1]

    def test_training_window_formula_ema_and_bidirectional_ratio(self):
        controller = TrainingWindowAdaptiveController()
        self.assertEqual(controller(1), 0.25)
        first = self._window(controller, 10000, 8)
        self.assertEqual(first["C_t"], 0.8)
        self.assertEqual(first["Cbar_t"], 0.8)
        self.assertAlmostEqual(first["rho_Astar"], 0.25 * 0.15 / 0.35)
        second = self._window(controller, 20000, 10)
        self.assertAlmostEqual(second["Cbar_t"], 0.86)
        self.assertLess(second["rho_Astar"], first["rho_Astar"])
        third = self._window(controller, 30000, 0)
        self.assertAlmostEqual(third["Cbar_t"], 0.602)
        self.assertGreater(third["rho_Astar"], second["rho_Astar"])
        fourth = self._window(controller, 40000, 0)
        self.assertEqual(fourth["rho_Astar"], 0.25)
        fresh = TrainingWindowAdaptiveController()
        self.assertEqual(self._window(fresh, 10000, 10)["rho_Astar"], 0.0)

    def test_minimum_all_and_conflict_success_and_skipped_windows(self):
        controller = TrainingWindowAdaptiveController()
        skipped = self._window(controller, 10000, 4, conflicts=4)
        self.assertIsNone(skipped["C_t"])
        self.assertIsNone(skipped["Cbar_t"])
        self.assertEqual(skipped["rho_Astar"], 0.25)
        valid = self._window(controller, 20000, 5, conflicts=5, nonconflict_failures=5)
        self.assertEqual(valid["SafeSuccess_all"], 0.5)
        self.assertEqual(valid["SafeSuccess_conflict"], 1.0)
        self.assertEqual(valid["C_t"], 0.5)
        skipped = self._window(controller, 30000, 0, conflicts=0)
        self.assertEqual(skipped["C_t"], 0.5)
        self.assertEqual(skipped["Cbar_t"], 0.5)
        self.assertEqual(skipped["rho_Astar"], 0.25)
        self.assertFalse(skipped["updated"])

    def test_exact_step_window_hook_excludes_unfinished_episode(self):
        problem = NavigationProblem(map_id="window_test", seed=0, size=5,
                                    start=(0, 0), goal=(4, 4), obstacles=frozenset(),
                                    nominal_path=((0, 0), (0, 1)))
        def factory(problem, **kwargs):
            env = StaticGridNavigationEnv(problem, **kwargs)
            env.conflict_opportunity_steps = 1
            return env
        controller = TrainingWindowAdaptiveController()
        result = train_d3qn(
            [problem], _StayAgent(),
            TrainingConfig(episodes=10000, max_environment_steps=10001, max_steps=3,
                           replay_capacity=20, batch_size=4, learning_starts=20000,
                           epsilon_decay_environment_steps=10000,
                           replay_strategy="persistent_demo", window_size=5),
            demonstrations=_demonstrations(5), environment_factory=factory,
            demo_fraction_schedule=controller, training_interaction_controller=controller,
        )
        self.assertEqual(result.environment_steps, 10001)
        self.assertEqual(len(controller.records), 1)
        self.assertEqual(controller.records[0]["environment_steps"], 10000)
        self.assertEqual(controller.records[0]["N_completed"], 3333)
        self.assertEqual(controller.records[0]["N_conflict"], 3333)
        self.assertEqual(controller.completed, 0)

    def test_config_only_changes_registered_demo_replay(self):
        config = yaml.safe_load(
            (ROOT / "configs/whole_map_route_pool_91701_time_decay_v1.yaml").read_text(
                encoding="utf-8"
            )
        )

        _validate_time_decay_comparison(config)
        training = _training_config(config, seed=0)
        self.assertEqual(training.replay_strategy, "persistent_demo")
        self.assertEqual(training.demo_fraction, 0.25)
        self.assertEqual(training.max_environment_steps, 200000)
        self.assertEqual(config["training"]["seeds"], [0, 1])
        self.assertEqual(
            config["experiment"]["output_root"],
            "outputs/whole_map_route_pool_91701_time_decay_v1",
        )

    def test_reuses_exact_registered_piecewise_schedule(self):
        self.assertEqual(
            [
                decay_demo_fraction(step)
                for step in (1, 50000, 50001, 100000, 100001, 200000)
            ],
            [0.25, 0.25, 0.10, 0.10, 0.0, 0.0],
        )

    def test_trainer_schedule_uses_one_based_environment_steps(self):
        problem = NavigationProblem(
            map_id="scheduled_demo_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        config = TrainingConfig(
            episodes=3,
            max_environment_steps=5,
            max_steps=3,
            replay_capacity=20,
            batch_size=4,
            learning_starts=1000,
            epsilon_decay_environment_steps=4,
            replay_strategy="persistent_demo",
            demo_fraction=0.25,
            window_size=5,
        )
        schedule = lambda step: 0.25 if step <= 2 else 0.10 if step <= 4 else 0.0

        result = train_d3qn(
            [problem],
            _StayAgent(),  # type: ignore[arg-type]
            config,
            demonstrations=_demonstrations(5),
            demo_fraction_schedule=schedule,
        )

        self.assertEqual(len(result.episode_records), 2)
        self.assertAlmostEqual(result.episode_records[0]["demo_fraction_mean"], 0.20)
        self.assertAlmostEqual(result.episode_records[1]["demo_fraction_mean"], 0.05)
        self.assertEqual(result.episode_records[-1]["demo_fraction"], 0.0)

    def test_schedule_rejects_uniform_replay(self):
        problem = NavigationProblem(
            map_id="invalid_scheduled_demo_test",
            seed=1,
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
            nominal_path=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        )
        with self.assertRaisesRegex(ValueError, "requires persistent_demo"):
            train_d3qn(
                [problem],
                _StayAgent(),  # type: ignore[arg-type]
                TrainingConfig(replay_strategy="uniform", window_size=5),
                demo_fraction_schedule=lambda _step: 0.25,
            )


if __name__ == "__main__":
    unittest.main()
