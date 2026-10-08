from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import teaching_efficiency_common as common
import run_teaching_efficiency as entry
import evaluate_teaching_efficiency as independent
from analyze_teaching_efficiency import aulc
from astar_d3qn.agents.astar_supervision_only import AStarSupervisionOnlyAgent
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.core.astar import astar_path
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.seed import seed_everything


class TeachingEfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = NavigationProblem('toy', 0, 7, (3, 0), (3, 6), frozenset(),
                                        tuple(astar_path((3, 0), (3, 6), frozenset(), 7)))

    def agent(self, *, original=None):
        seed_everything(0)
        config = D3QNConfig((5, 15, 15), 4, 5, hidden_dim=16, device='cpu')
        if original:
            return AStarValueRepairAgent(config, self.problem, repair_method=original)
        return AStarSupervisionOnlyAgent(config, self.problem)

    def transition(self):
        env = PathGuidanceEnvironment(DynamicGridNavigationEnv(self.problem, (), window_size=15), enabled=False)
        state = env.reset()
        result = env.step(3)
        return Transition(state, 3, result.reward, result.observation, result.terminated, env.action_mask(True))

    def test_exact_initialization_matches_original_combination(self):
        self.assertEqual(common.legacy.runtime.state_digest(self.agent().training_state_dict()),
                         common.legacy.runtime.state_digest(self.agent(original='advice_bound_margin').training_state_dict()))

    def test_actions_and_rng_equal_unguided_control_without_planner(self):
        agent, control = self.agent(), self.agent(original='unguided_bound')
        state = self.transition().state
        with patch.object(agent, 'static_advice', side_effect=AssertionError('Planner selected an action')):
            for epsilon in (1., .5, .05, 0.):
                self.assertEqual(agent.select_action(state, epsilon), control.select_action(state, epsilon))
                self.assertEqual(agent._rng.getstate(), control._rng.getstate())
                self.assertEqual(agent.advice_rng.getstate(), control.advice_rng.getstate())
        self.assertTrue(all(r['applied'] == 0 for r in agent.advice_records()))

    def test_supervised_update_bitwise_equal_original_on_same_batch(self):
        new, original = self.agent(), self.agent(original='advice_bound_margin')
        batch = [self.transition()] * 4
        new.train_batch(batch)
        original.train_batch(batch)
        from astar_d3qn.agents.d3qn import D3QNAgent
        self.assertEqual(common.legacy.runtime.state_digest(D3QNAgent.training_state_dict(new)),
                         common.legacy.runtime.state_digest(D3QNAgent.training_state_dict(original)))

    def test_labels_and_coverage_exclude_mismatch_risk_and_collision(self):
        agent = self.agent()
        clean = self.transition()
        risky = copy.deepcopy(clean)
        risky.state.spatial[2, 7, 8] = 1.
        batch = [clean, replace(clean, action=4), replace(clean, reward=-1., terminated=True), risky]
        metrics = agent.train_batch(batch)
        coverage = agent.coverage_records()[0]
        self.assertEqual(metrics['teacher_label_count'], 1)
        self.assertEqual(coverage['sampled_transitions'], 4)
        self.assertEqual(coverage['eligible_noncollision_match'], 1)
        self.assertEqual(coverage['teacher_label_coverage'], .25)

    def test_no_labels_after_withdrawal_and_greedy_evaluation_preserves_state(self):
        agent = self.agent()
        agent.training_action_steps = 100000
        before = common.legacy.runtime.state_digest(agent.training_state_dict())
        with patch.object(agent, 'static_advice', side_effect=AssertionError('Planner after withdrawal')):
            with common.legacy.runtime.preserved_evaluation(agent):
                agent.select_action(self.transition().state, 0.)
            self.assertEqual(common.legacy.runtime.state_digest(agent.training_state_dict()), before)
            agent.train_batch([self.transition()] * 4)
        self.assertEqual(agent.coverage_records()[0]['supervised_samples'], 0)
        self.assertIsNone(agent.coverage_records()[0]['active_teacher_label_coverage'])

    def test_full_state_restores_optimizer_rng_and_coverage(self):
        agent = self.agent()
        agent.select_action(self.transition().state, .5)
        agent.train_batch([self.transition()] * 4)
        state = copy.deepcopy(agent.training_state_dict())
        restored = self.agent()
        restored.load_training_state_dict(state)
        self.assertEqual(common.legacy.runtime.state_digest(state), common.legacy.runtime.state_digest(restored.training_state_dict()))

    def test_phase_aulc_interpolates_fixed_boundaries(self):
        x, y = np.array([0, 100000, 200000]), np.array([0., 1., 1.])
        self.assertAlmostEqual(aulc(x, y), .75)
        self.assertAlmostEqual(aulc(x, y, 0, 50000), .25)
        self.assertAlmostEqual(aulc(x, y, 50000, 100000), .75)

    def test_physical_identity_ignores_labels_order_and_endpoint_direction(self):
        obstacle = dict(route=[[0, 0], [0, 1], [0, 2]], start_index=0, direction=1, move_every=1)
        left = dict(obstacles=[dict(obstacle, label='left')])
        right = dict(obstacles=[dict(obstacle, label='right', direction=-1)])
        reversed_route = dict(obstacles=[dict(obstacle, route=list(reversed(obstacle['route'])), start_index=2, direction=-1)])
        self.assertEqual(common.physical_key(left), common.physical_key(right))
        self.assertEqual(common.physical_key(left), common.physical_key(reversed_route))
        self.assertEqual(common.combination_key(left), common.combination_key(reversed_route))

    def test_collection_diagnostics_do_not_change_collected_transition_or_actions(self):
        def factory(problem, **kwargs):
            return PathGuidanceEnvironment(DynamicGridNavigationEnv(problem, (), **kwargs), enabled=False)
        agent = self.agent()
        critical = [dict(observation_sha256=common.observation_hash(self.transition().state), position=[3, 0])]
        diagnosed = entry.CollectionDiagnostics(factory, agent, critical)(self.problem, window_size=15)
        ordinary = factory(self.problem, window_size=15)
        left, right = diagnosed.reset(), ordinary.reset()
        for action in (3, 4, 3):
            agent.training_action_steps += 1
            a, b = diagnosed.step(action), ordinary.step(action)
            self.assertEqual(a.info, b.info)
            self.assertEqual(a.reward, b.reward)
            np.testing.assert_array_equal(a.observation.spatial, b.observation.spatial)
            np.testing.assert_array_equal(a.observation.scalars, b.observation.scalars)

    def test_sampled_critical_coverage_distinguishes_draws_and_labels(self):
        agent = self.agent()
        transition = self.transition()
        agent.critical_positions = {(3, 0)}
        agent.critical_observation_hashes = {common.observation_hash(transition.state)}
        agent.train_batch([transition, replace(transition, action=4), transition, transition])
        row = agent.coverage_records()[0]
        self.assertEqual(row['exact_critical_sample_draws'], 4)
        self.assertEqual(row['exact_critical_supervised_draws'], 3)

    def test_frozen_factory_preserves_initial_phase_and_rejects_exhaustion(self):
        specs = [dict(route=[[r, 0], [r, 1], [r, 2]], start_index=1, direction=-1,
                      move_every=1, label=str(r)) for r in (0, 1, 5)]
        scene = dict(scenario_id='frozen', route_ids=['0', '1', '5'], categories=['high_interaction', 'alternative_branch', 'background'], obstacles=specs)
        factory = common.FrozenFactory(self.problem, [scene])
        env = factory(self.problem, window_size=15)
        env.reset()
        self.assertEqual(env.dynamic_positions, ((0, 1), (1, 1), (5, 1)))
        for time in range(4):
            self.assertEqual(tuple(common.obstacle_cycle(spec)[time] for spec in specs), env.dynamic_positions)
            env.step(4)
        with self.assertRaises(IndexError):
            factory(self.problem, window_size=15)
        factory.reset_schedule()
        self.assertEqual(factory(self.problem, window_size=15).scenario_id, 'frozen')

    def test_toy_loop_with_collection_observer_is_bitwise_identical(self):
        # Eight toy steps test actual trainer/replay integration, without formal-map training.
        from astar_d3qn.training.trainer import TrainingConfig, train_d3qn
        def factory(problem, **kwargs):
            return PathGuidanceEnvironment(DynamicGridNavigationEnv(problem, (), **kwargs), enabled=False)
        factory.reset_schedule = lambda: None
        settings = TrainingConfig(episodes=20, max_environment_steps=8, max_steps=10,
                                  batch_size=4, learning_starts=4, epsilon_decay_environment_steps=150000,
                                  allow_partial_epsilon_schedule=True, replay_strategy='uniform',
                                  window_size=15, mask_static_invalid_actions=True,
                                  progress_interval=0, diagnostic_interval=0, seed=0)
        ordinary = self.agent()
        first = train_d3qn([self.problem], ordinary, settings, environment_factory=factory)
        diagnosed = self.agent()
        wrapper = entry.CollectionDiagnostics(factory, diagnosed, [])
        second = train_d3qn([self.problem], diagnosed, settings, environment_factory=wrapper)
        self.assertEqual(first.environment_steps, second.environment_steps)
        self.assertEqual(first.gradient_updates, second.gradient_updates)
        self.assertEqual(first.episode_records, second.episode_records)
        self.assertEqual(common.legacy.runtime.state_digest(ordinary.training_state_dict()),
                         common.legacy.runtime.state_digest(diagnosed.training_state_dict()))
        self.assertEqual(sum(r['collected_transitions'] for r in wrapper.records()), 8)

    def test_defaults_never_train_or_evaluate_500_scene_performance(self):
        with patch.object(entry.common, 'validate_config', return_value=None), \
             patch.object(entry, 'check', return_value={}), \
             patch.object(entry.common.legacy.runtime, 'unique_check_dir', return_value=Path('unused')), \
             patch.object(entry, 'write_json'), patch.object(entry, 'run_training') as training:
            entry.main([])
            training.assert_not_called()
        with patch.object(independent.common, 'validate_config'), \
             patch.object(independent.common, 'load_frozen', return_value=(list(range(500)), [], {})), \
             patch.object(independent, 'evaluate') as evaluation:
            independent.main([])
            evaluation.assert_not_called()

    def test_all_models_preflight_before_independent_evaluation(self):
        config = common.legacy.load_config(common.CONFIG)
        with patch.object(independent.common, 'validate_config', return_value=None), \
             patch.object(independent.common, 'load_frozen', return_value=([], [], {})), \
             patch.object(independent.common, 'audit_run', side_effect=FileNotFoundError('missing final model')), \
             patch.object(independent, 'evaluate_agent') as rollout:
            with self.assertRaises(FileNotFoundError):
                independent.evaluate(config)
            rollout.assert_not_called()

    def test_output_guard_does_not_start_training(self):
        config = copy.deepcopy(common.legacy.load_config(common.CONFIG))
        with tempfile.TemporaryDirectory() as temporary:
            config['experiment']['output_root'] = temporary
            existing = Path(temporary) / common.METHOD / 'seed_0'
            existing.mkdir(parents=True)
            with patch.object(entry.common, 'load_frozen', return_value=([], [], {})), \
                 patch.object(entry.common.legacy, 'train_d3qn') as training:
                with self.assertRaises(FileExistsError):
                    entry.run_training(config, None, [0])
                training.assert_not_called()


if __name__ == '__main__':
    unittest.main()
