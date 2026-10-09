from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import supervision_tail_common as common
import run_supervision_tail as entry
import evaluate_supervision_tail as independent
from analyze_supervision_tail import curve_metrics, describe_delay, failure_counts
from astar_d3qn.agents.astar_supervision_tail import AStarSupervisionTailAgent, supervision_weight
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.core.astar import astar_path
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.seed import seed_everything


class SupervisionTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = NavigationProblem('toy', 0, 7, (3, 0), (3, 6), frozenset(),
                                        tuple(astar_path((3, 0), (3, 6), frozenset(), 7)))

    def agent(self, original=False):
        seed_everything(0)
        config = D3QNConfig((5, 15, 15), 4, 5, hidden_dim=16, device='cpu')
        if original:
            return AStarValueRepairAgent(config, self.problem, repair_method='advice_bound_margin')
        return AStarSupervisionTailAgent(config, self.problem,
            supervision_schedule=dict(original_decay_steps=100000, tail_start_steps=80000, tail_end_steps=140000))

    def transition(self, action=3):
        env = PathGuidanceEnvironment(DynamicGridNavigationEnv(self.problem, (), window_size=15), enabled=False)
        state = env.reset()
        result = env.step(action)
        return Transition(state, action, result.reward, result.observation, result.terminated, env.action_mask(True))

    def test_exact_requested_schedule_and_continuity(self):
        self.assertEqual(supervision_weight(0), 1)
        self.assertAlmostEqual(supervision_weight(80000), .2, places=15)
        self.assertAlmostEqual(supervision_weight(80000+1e-6), .2, places=10)
        self.assertEqual(supervision_weight(100000), .2*40000/60000)
        self.assertEqual(supervision_weight(120000), .2*20000/60000)
        self.assertEqual(supervision_weight(140000), 0)
        self.assertEqual(supervision_weight(200000), 0)
        self.assertGreater(supervision_weight(139999), 0)

    def test_every_early_step_is_identical_and_clock_never_reactivates(self):
        for step in range(80001):
            self.assertEqual(supervision_weight(step), max(0, 1-step/100000))
        values = np.array([supervision_weight(t) for t in range(200001)])
        self.assertTrue(np.all(np.diff(values) <= 0))
        self.assertTrue(np.all(values[140000:] == 0))
        for step in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                supervision_weight(step)

    def test_same_full_initial_training_state(self):
        self.assertEqual(common.legacy.runtime.state_digest(self.agent().training_state_dict()),
                         common.legacy.runtime.state_digest(self.agent(True).training_state_dict()))

    def test_advice_probability_unchanged_and_zero_at_100000(self):
        agent, original = self.agent(), self.agent(True)
        for t in (1, 79999, 80000, 80001, 99999, 100000, 100001, 139999, 140000, 200000):
            self.assertEqual(agent.advice_probability(t), original.advice_probability(t))
        self.assertEqual(agent.decay_steps, 100000)
        self.assertEqual(agent.advice_probability(100000), 0)

    def test_withdrawn_advice_does_not_call_planner_or_replace_actions(self):
        agent = self.agent()
        agent.training_action_steps = 99999
        state = self.transition().state
        with patch.object(agent, 'static_advice', side_effect=AssertionError('A* action selection after cutoff')):
            agent.select_action(state, .05, list(np.flatnonzero(agent.current_static_mask(state))))
        self.assertEqual(agent.training_action_steps, 100000)
        self.assertEqual(agent.advice_records()[0]['applied'], 0)
        self.assertGreater(agent.imitation_weight(), 0)

    def test_identical_early_gradient_updates_not_just_weight_numbers(self):
        batch = [self.transition() for _ in range(4)]
        for t in (0, 79999, 80000):
            agent, original = self.agent(), self.agent(True)
            agent.training_action_steps = original.training_action_steps = t
            first, second = agent.train_batch(batch), original.train_batch(batch)
            self.assertEqual(first, second)
            self.assertEqual(common.legacy.runtime.state_digest(agent.training_state_dict()),
                             common.legacy.runtime.state_digest(original.training_state_dict()))

    def test_tail_keeps_labels_and_margin_but_zero_after_exact_end(self):
        batch = [self.transition() for _ in range(4)]
        agent = self.agent()
        agent.training_action_steps = 110000
        result = agent.train_batch(batch)
        self.assertEqual(result['teacher_label_count'], 4)
        self.assertGreater(result['teacher_margin_weight'], 0)
        agent.training_action_steps = 140000
        with patch.object(agent, 'teacher_label', side_effect=AssertionError('Label after exact zero')):
            result = agent.train_batch(batch)
        self.assertEqual(result['teacher_label_count'], 0)
        self.assertEqual(result['teacher_margin_weight'], 0)

    def test_label_filters_remain_matching_risk_clear_noncollision(self):
        agent = self.agent()
        good = self.transition()
        self.assertTrue(agent.teacher_label(good))
        self.assertFalse(agent.teacher_label(self.transition(4)))
        from dataclasses import replace
        self.assertFalse(agent.teacher_label(replace(good, terminated=True, reward=-1)))
        risky = copy.deepcopy(good)
        radius = risky.state.spatial.shape[1]//2
        risky.state.spatial[1, radius, radius+1] = 1
        self.assertFalse(agent.teacher_label(risky))

    def test_td_bound_is_identical(self):
        agent, original = self.agent(), self.agent(True)
        args = (torch.tensor([50., -50.]), torch.tensor([True, True]),
                torch.zeros((2, 5)), torch.zeros((2, 5)), torch.ones((2, 5), dtype=torch.bool))
        first = agent._td_targets(*args)
        self.assertTrue(torch.equal(first, original._td_targets(*args)))
        self.assertTrue(torch.equal(first, torch.tensor([10., -6.])))

    def test_epsilon_zero_evaluation_preserves_full_state_and_skips_teacher(self):
        agent = self.agent()
        state = self.transition().state
        before = common.legacy.runtime.state_digest(agent.training_state_dict())
        with patch.object(agent, 'static_advice', side_effect=AssertionError('Teacher in autonomous evaluation')):
            with common.legacy.runtime.preserved_evaluation(agent):
                agent.select_action(state, 0, list(np.flatnonzero(agent.current_static_mask(state))))
        self.assertEqual(before, common.legacy.runtime.state_digest(agent.training_state_dict()))

    def test_passive_collection_diagnostics_do_not_change_toy_training(self):
        from astar_d3qn.training.trainer import TrainingConfig, train_d3qn
        from run_teaching_efficiency import CollectionDiagnostics
        def factory(problem, **kwargs):
            return PathGuidanceEnvironment(DynamicGridNavigationEnv(problem, (), **kwargs), enabled=False)
        factory.reset_schedule = lambda: None
        settings = TrainingConfig(episodes=20, max_environment_steps=8, max_steps=10, batch_size=4, learning_starts=4,
            epsilon_decay_environment_steps=150000, allow_partial_epsilon_schedule=True, replay_strategy='uniform',
            window_size=15, mask_static_invalid_actions=True, progress_interval=0, diagnostic_interval=0, seed=0)
        plain, watched = self.agent(True), self.agent()
        a = train_d3qn([self.problem], plain, settings, environment_factory=factory)
        b = train_d3qn([self.problem], watched, settings,
                      environment_factory=CollectionDiagnostics(factory, watched, []))
        self.assertEqual(a.episode_records, b.episode_records)
        self.assertEqual(common.legacy.runtime.state_digest(plain.training_state_dict()),
                         common.legacy.runtime.state_digest(watched.training_state_dict()))

    def test_config_preserves_all_shared_settings_and_rejects_changes(self):
        config = common.legacy.load_config(common.CONFIG)
        common.validate_config(config)
        for key in ('agent', 'reward', 'training_advice', 'training'):
            changed = copy.deepcopy(config)
            changed[key]['unexpected_change'] = True
            with self.assertRaises(ValueError):
                common.validate_config(changed)

    def test_window_metrics_detect_late_dip_even_when_old_window_improves(self):
        settings = common.legacy.load_config(common.CONFIG)['supervision_tail']
        x = np.arange(0, 200001, 10000)
        original, new = np.ones(21), np.ones(21)
        original[10] = .4
        new[14] = .2
        control, treated = curve_metrics(x, original, settings), curve_metrics(x, new, settings)
        self.assertEqual(treated['original_window_min_success'], 1)
        self.assertEqual(treated['new_window_min_success'], .2)
        self.assertLess(treated['aulc_140000_200000'], control['aulc_140000_200000'])
        self.assertIn('后移', describe_delay(treated, control)['descriptive_verdict'])

    def test_waiting_and_looping_timeout_counts_are_disjoint(self):
        rows = [dict(termination_reason='timeout', waiting_dominated=True, repeated_movement=True, tail_period=2),
                dict(termination_reason='timeout', waiting_dominated=False, repeated_movement=True, tail_period=2),
                dict(termination_reason='timeout', waiting_dominated=False, repeated_movement=False, tail_period=None),
                dict(termination_reason='collision', waiting_dominated=False, repeated_movement=False, tail_period=None)]
        self.assertEqual(failure_counts(rows), dict(collision_count=1, timeout_count=3, waiting_timeout_count=1,
                                                   looping_timeout_count=1, other_timeout_count=1))

    def test_existing_output_guard_prevents_training(self):
        config = copy.deepcopy(common.legacy.load_config(common.CONFIG))
        with tempfile.TemporaryDirectory() as temporary:
            config['experiment']['output_root'] = temporary
            (Path(temporary)/common.METHOD/'seed_0').mkdir(parents=True)
            with patch.object(common.legacy, 'train_d3qn') as training:
                with self.assertRaises(FileExistsError):
                    entry.run_training(config, None, [0])
                training.assert_not_called()

    def test_all_ten_models_preflight_before_any_held_out_rollout(self):
        config = common.legacy.load_config(common.CONFIG)
        calls = []
        def auditor(config, method, seed):
            calls.append((method, seed))
            if len(calls) == 10:
                raise FileNotFoundError('last pending final model')
            return Path('unused'), {}
        with patch.object(common, 'validate_config', return_value=None), \
             patch.object(common, 'load_frozen', return_value=([], [], {})), \
             patch.object(common, 'audit_run', side_effect=auditor), \
             patch.object(independent, 'evaluate_agent') as rollout:
            with self.assertRaises(FileNotFoundError):
                independent.evaluate(config)
            rollout.assert_not_called()
            self.assertEqual(len(calls), 10)

    def test_entry_defaults_only_check_without_training_or_final_evaluation(self):
        with patch.object(common, 'validate_config', return_value=None), \
             patch.object(entry, 'check', return_value={}), \
             patch.object(common.legacy.runtime, 'unique_check_dir', return_value=Path('unused')), \
             patch.object(entry, 'write_json'), patch.object(entry, 'write_records_csv'), \
             patch.object(entry, 'run_training') as training:
            entry.main([])
            training.assert_not_called()
        with patch.object(common, 'validate_config'), \
             patch.object(common, 'load_frozen', return_value=(list(range(500)), [], {})), \
             patch.object(independent, 'evaluate') as evaluation:
            independent.main([])
            evaluation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
