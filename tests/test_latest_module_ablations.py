from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import test_execution_consistency as fixtures
from astar_d3qn.agents.all_eligible_module_ablation import AllEligibleModuleAblationAgent,METHODS
from astar_d3qn.agents.d3qn import D3QNAgent,D3QNConfig


class LatestModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ExecutionConsistencyTests.setUpClass()
        cls.fx=fixtures.ExecutionConsistencyTests()

    def agent(self,method):
        fixtures.seed_everything(0)
        return AllEligibleModuleAblationAgent(D3QNConfig((5,5,5),4,5,hidden_dim=16,device='cpu'),
            self.fx.problem,repair_method='advice_bound_margin',ablation_method=method)

    def test_real_td_and_independent_teacher_for_both_arms(self):
        t=self.fx.transition(4)
        for method in METHODS:
            a=self.agent(method);a.policy_network=fixtures.FixedQ();a.target_network=fixtures.FixedQ()
            a.optimizer=torch.optim.SGD(a.policy_network.parameters(),lr=0)
            m=a.train_batch([t])
            self.assertAlmostEqual(m['q_mean'],5)
            self.assertAlmostEqual(m['td_loss'],.5*(t.reward+a.config.gamma*5-5)**2,places=6)
            self.assertAlmostEqual(m['demo_margin_loss'],1.8,places=5)
            self.assertEqual(m['teacher_label_matching_count'],0);self.assertEqual(t.action,4)

    def test_only_td_clip_changes_raw_double_dqn_target(self):
        a,b=self.agent(METHODS[0]),self.fx.agent(True)
        q=torch.tensor([[9.,2.,3.,4.,5.],[9.,2.,3.,4.,5.]])
        target=torch.tensor([[27.,2.,3.,4.,5.],[27.,2.,3.,4.,5.]])
        rewards=torch.tensor([0.,-12.]);terminated=torch.tensor([False,True]);valid=torch.ones_like(q,dtype=torch.bool)
        raw=a._td_targets(rewards,terminated,q,target,valid)
        bounded=b._td_targets(rewards,terminated,q,target,valid)
        self.assertTrue(torch.equal(raw,torch.tensor([27*a.config.gamma,-12.])))
        self.assertTrue(torch.equal(bounded,torch.tensor([10.,-6.])))
        self.assertEqual(a._target_metrics['target_clip_fraction'],0)
        self.assertEqual(a._target_metrics['raw_target_outside_fraction'],1)
        self.assertEqual(a.advice_probability(60000),b.advice_probability(60000))
        self.assertEqual(a.margin,b.margin);self.assertEqual(a.config,b.config)

    def test_no_override_matches_plain_exploration_but_supervises(self):
        a=self.agent(METHODS[1]);plain=self.fx.agent(True);t=self.fx.transition(4)
        with patch.object(a,'static_advice',side_effect=AssertionError('Teacher in action selection')):
            for epsilon in [1.,.3,.05]*10:
                self.assertEqual(a.select_action(t.state,epsilon,[0,1,3,4]),D3QNAgent.select_action(plain,t.state,epsilon,[0,1,3,4]))
        self.assertEqual(a.training_action_steps,30)
        self.assertGreater(a.imitation_weight(),0)
        m=a.train_batch([t]*4)
        self.assertEqual(m['teacher_label_count'],4);self.assertEqual(m['teacher_label_matching_count'],0)
        self.assertFalse(any(r['requested'] or r['applied'] for r in a.advice_records()))

    def test_filters_and_exit_still_apply(self):
        t=self.fx.transition(4)
        for method in METHODS:
            a=self.agent(method)
            self.assertTrue(a.teacher_label(t))
            for channel in (0,1,2,3):
                bad=copy.deepcopy(t);bad.state.spatial[channel,2,3]=1
                self.assertFalse(a.teacher_label(bad))
            collision=fixtures.Transition(t.state,4,-1,t.next_state,True,t.next_action_mask)
            self.assertFalse(a.teacher_label(collision))
            a.training_action_steps=99999
            with patch.object(a,'static_advice',side_effect=AssertionError('Teacher after exit')):
                a.select_action(t.state,.05,[0,1,3,4]);self.assertEqual(a.training_action_steps,100000)
                self.assertEqual(a.imitation_weight(),0);self.assertEqual(a.advice_probability(),0)
                self.assertEqual(a.train_batch([t]*4)['teacher_label_count'],0)
                before=fixtures.legacy.runtime.state_digest(a.training_state_dict())
                a.select_action(t.state,0,[0,1,3,4])
                self.assertEqual(before,fixtures.legacy.runtime.state_digest(a.training_state_dict()))

    def test_pairing_and_restored_cache_ranges(self):
        for method in METHODS:
            a=self.agent(method);reference=self.fx.agent(True)
            self.assertEqual(fixtures.legacy.runtime.state_digest(a.training_state_dict()),fixtures.legacy.runtime.state_digest(reference.training_state_dict()))
            a.train_batch([self.fx.transition(4)]*4)
            saved=copy.deepcopy(a.training_state_dict());b=self.agent(method);b.load_training_state_dict(saved)
            self.assertEqual(fixtures.legacy.runtime.state_digest(saved),fixtures.legacy.runtime.state_digest(b.training_state_dict()))
            self.assertEqual(a._static_action_cache,b._static_action_cache)
            self.assertIn('raw_target_min',a.supervision_records()[0]);self.assertIn('raw_target_max',a.supervision_records()[0])

    def test_nonfinite_target_is_rejected(self):
        for method in METHODS:
            a=self.agent(method);q=torch.full((1,5),float('inf'))
            with self.assertRaises(FloatingPointError):
                a._td_targets(torch.tensor([0.]),torch.tensor([False]),q,q,torch.ones_like(q,dtype=torch.bool))


if __name__=='__main__':unittest.main()
