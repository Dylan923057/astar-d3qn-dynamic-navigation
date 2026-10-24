from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import test_execution_consistency as fixtures
import persistent_paper_common as core
from astar_d3qn.agents.all_eligible_module_ablation import AllEligibleModuleAblationAgent
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig


class PersistentPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ExecutionConsistencyTests.setUpClass()
        cls.fx = fixtures.ExecutionConsistencyTests()

    def agent(self, method, old=False):
        fixtures.seed_everything(0)
        cls = AllEligibleModuleAblationAgent if old else core.PersistentModuleAgent
        return cls(D3QNConfig((5, 5, 5), 4, 5, hidden_dim=16, device='cpu'),
            self.fx.problem, repair_method='advice_bound_margin', ablation_method=core.OLD_METHODS[method])

    def test_old_ablation_prefix_has_identical_optimizer_target_rng(self):
        for method in core.METHODS:
            old, new = self.agent(method, True), self.agent(method)
            for step in (0, 50000, 89999):
                old.training_action_steps = new.training_action_steps = step
                self.assertEqual(old.advice_probability(), new.advice_probability())
                self.assertEqual(old.imitation_weight(), new.imitation_weight())
                a, b = old.train_batch([self.fx.transition(4)] * 4), new.train_batch([self.fx.transition(4)] * 4)
                self.assertEqual(a, b)
                self.assertEqual(core.legacy.runtime.state_digest(old.training_state_dict()), core.legacy.runtime.state_digest(new.training_state_dict()))

    def test_margin_persists_and_td_uses_the_executed_action(self):
        for method in core.METHODS:
            agent = self.agent(method)
            agent.training_action_steps = 150000
            agent.policy_network = fixtures.FixedQ()
            agent.target_network = fixtures.FixedQ()
            agent.optimizer = torch.optim.SGD(agent.policy_network.parameters(), lr=0)
            transition = self.fx.transition(4)
            metrics = agent.train_batch([transition])
            self.assertEqual(metrics['teacher_margin_weight'], .1)
            self.assertEqual(metrics['teacher_label_count'], 1)
            self.assertEqual(metrics['teacher_label_matching_count'], 0)
            self.assertAlmostEqual(metrics['td_loss'], .5*(transition.reward+.99*5-5)**2, places=6)
            self.assertTrue(agent.supervision_records()[0]['labels_computed'])
            self.assertEqual(transition.action, 4)

    def test_no_advice_keeps_supervision_and_plain_exploration(self):
        agent = self.agent('no_advice')
        plain = self.fx.agent(True)
        transition = self.fx.transition(4)
        for step in (0, 99999, 150000):
            agent.training_action_steps = step
            with patch.object(agent, 'static_advice', side_effect=AssertionError('Teacher selected an action')):
                self.assertEqual(agent.select_action(transition.state, .05, [0, 1, 3, 4]),
                                 D3QNAgent.select_action(plain, transition.state, .05, [0, 1, 3, 4]))
            self.assertEqual(agent.advice_probability(), 0)
            self.assertGreater(agent.train_batch([transition])['teacher_label_count'], 0)
        self.assertTrue(agent.bound_enabled)
        self.assertFalse(any(r['requested'] or r['applied'] for r in agent.advice_records()))

    def test_no_td_only_removes_target_clipping(self):
        unclipped, bounded = self.agent('no_td'), self.agent('no_advice')
        q = torch.tensor([[9., 2., 3., 4., 5.], [9., 2., 3., 4., 5.]])
        target = q * 3
        args = (torch.tensor([0., -12.]), torch.tensor([False, True]), q, target, torch.ones_like(q, dtype=torch.bool))
        self.assertTrue(torch.equal(unclipped._td_targets(*args), torch.tensor([27*.99, -12.])))
        self.assertTrue(torch.equal(bounded._td_targets(*args), torch.tensor([10., -6.])))
        self.assertEqual(unclipped._target_metrics['target_clip_fraction'], 0)
        self.assertEqual(unclipped.advice_probability(50000), .4)
        self.assertEqual(unclipped.advice_probability(100000), 0)

    def test_risk_filter_and_autonomous_evaluation_still_apply(self):
        for method in core.METHODS:
            agent = self.agent(method)
            transition = self.fx.transition(4)
            agent.training_action_steps = 150000
            for channel in (0, 1, 2, 3):
                bad = copy.deepcopy(transition)
                bad.state.spatial[channel, 2, 3] = 1
                self.assertIsNone(agent.supervision_action(bad))
            before = core.legacy.runtime.state_digest(agent.training_state_dict())
            with patch.object(agent, 'static_advice', side_effect=AssertionError('Teacher during evaluation')):
                agent.select_action(transition.state, 0., [0, 1, 3, 4])
            self.assertEqual(before, core.legacy.runtime.state_digest(agent.training_state_dict()))

    def test_resume_keeps_cache_and_persistent_schedule(self):
        for method in core.METHODS:
            agent = self.agent(method)
            agent.training_action_steps = 150000
            agent.train_batch([self.fx.transition(4)] * 4)
            saved = copy.deepcopy(agent.training_state_dict())
            restored = self.agent(method)
            restored.load_training_state_dict(saved)
            self.assertEqual(core.legacy.runtime.state_digest(saved), core.legacy.runtime.state_digest(restored.training_state_dict()))
            self.assertEqual(restored._static_action_cache, agent._static_action_cache)
            self.assertEqual(restored.imitation_weight(), .1)

    def test_nonfinite_target_is_not_silently_restarted(self):
        for method in core.METHODS:
            agent = self.agent(method)
            q = torch.full((1, 5), float('inf'))
            with self.assertRaises(FloatingPointError):
                agent._td_targets(torch.tensor([0.]), torch.tensor([False]), q, q, torch.ones_like(q, dtype=torch.bool))

    def test_lock_blocks_duplicate_worker_and_releases(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            path = Path(temp) / 'run.lock'
            with core.run_lock(path):
                with self.assertRaises(OSError):
                    with core.run_lock(path):
                        pass
            with core.run_lock(path):
                pass

    def test_resume_flag_does_not_require_a_checkpoint_for_new_jobs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            root = Path(temp)
            run = root / 'no_td/seed_0'
            with patch.object(core, 'OUT', root), patch.object(core, 'directory', return_value=run), \
                    patch.object(core, 'ORIGINAL_RUN') as training:
                core.run_one('no_td', 0, resume=True)
                training.assert_called_once_with('no_td', 0, resume=False)
                training.reset_mock()
                run.mkdir(parents=True)
                core.run_one('no_td', 0, resume=True)
                training.assert_called_once_with('no_td', 0, resume=True)
                training.reset_mock()
                core.write_json(dict(environment_steps=123, numerical_failure=True), run / 'nonfinite_failure.json')
                core.run_one('no_td', 0, resume=True)
                training.assert_not_called()


if __name__ == '__main__':
    unittest.main()
