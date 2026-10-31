from __future__ import annotations
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests')]
import test_execution_consistency as helpers
import fixed_ce_common as c
from fixed_ce_agent import FixedTeacherCEAgent,masked_teacher_ce,AST_AUDIT
from astar_d3qn.agents.d3qn import D3QNAgent,D3QNConfig
from astar_d3qn.agents.persistent_teaching import PersistentTeachingAgent
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.seed import seed_everything


class FixedCETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers.ExecutionConsistencyTests.setUpClass();cls.helper=helpers.ExecutionConsistencyTests()

    def agent(self,ce=True):
        seed_everything(0)
        cls=FixedTeacherCEAgent if ce else PersistentTeachingAgent
        return cls(D3QNConfig((5,5,5),4,5,hidden_dim=16,device='cpu'),self.helper.problem,
                   repair_method='advice_bound_margin',advice_floor=0.,supervision_floor=.1)

    def test_analytical_ce_static_mask_and_eligible_denominator(self):
        q=torch.tensor([[1.,2.,1000.,4.,5.],[30.,0.,0.,0.,0.]],requires_grad=True)
        legal=torch.tensor([[True,True,False,True,True],[False]*5])
        loss=masked_teacher_ce(q,torch.tensor([3,2]),torch.tensor([True,False]),legal)
        expected=torch.logsumexp(torch.tensor([1.,2.,4.,5.]),0)-4
        self.assertAlmostEqual(loss.item(),expected.item(),places=6)
        loss.backward();self.assertEqual(q.grad[0,2].item(),0.);self.assertEqual(q.grad[1].abs().sum().item(),0.)
        self.assertEqual(masked_teacher_ce(q,torch.tensor([3,2]),torch.zeros(2,dtype=torch.bool),legal).item(),0.)
        with self.assertRaises(ValueError):masked_teacher_ce(q,torch.tensor([2,2]),torch.tensor([True,False]),legal)

    def test_td_uses_executed_action_and_ce_uses_distinct_teacher(self):
        agent=self.agent();item=self.helper.transition(4);agent.training_action_steps=110000
        agent.policy_network=helpers.FixedQ();agent.target_network=helpers.FixedQ()
        with torch.no_grad():agent.policy_network.values[2]=1000.;agent.target_network.values[2]=1000.
        agent.optimizer=torch.optim.SGD(agent.policy_network.parameters(),lr=0)
        before=copy.deepcopy(item)
        metrics=agent.train_batch([item])
        expected_ce=(torch.logsumexp(torch.tensor([1.,2.,4.,5.]),0)-4).item()
        expected_td=.5*(item.reward+.99*5-5)**2
        self.assertAlmostEqual(metrics['td_loss'],expected_td,places=6)
        self.assertAlmostEqual(metrics['supervision_ce_loss'],expected_ce,places=6)
        self.assertAlmostEqual(metrics['loss'],expected_td+.1*expected_ce,places=6)
        self.assertEqual(metrics['teacher_label_count'],1);self.assertEqual(metrics['teacher_label_matching_count'],0)
        self.assertEqual(item.action,4);self.assertEqual(agent.static_advice(item.state),3)
        self.assertTrue(np.array_equal(item.state.spatial,before.state.spatial))
        self.assertTrue(np.array_equal(item.next_state.spatial,before.next_state.spatial))

    def test_no_labels_matches_original_td_parameter_update(self):
        a,b=self.agent(),self.agent(False);item=self.helper.transition(4)
        collision=Transition(item.state,4,-1.,item.next_state,True,item.next_action_mask)
        ma,mb=a.train_batch([collision]*4),b.train_batch([collision]*4)
        self.assertEqual(ma['supervision_ce_loss'],0.)
        self.assertEqual(ma['td_loss'],mb['td_loss'])
        self.assertEqual(c.digest(D3QNAgent.training_state_dict(a)),c.digest(D3QNAgent.training_state_dict(b)))

    def test_risk_three_frames_collision_and_static_filters_unchanged(self):
        a,b=self.agent(),self.agent(False);item=self.helper.transition(4)
        candidates=[item,Transition(item.state,4,-1.,item.next_state,True,item.next_action_mask)]
        for channel in (1,2,3):
            risky=copy.deepcopy(item);risky.state.spatial[channel,2,3]=1;candidates.append(risky)
        blocked=copy.deepcopy(item);blocked.state.spatial[0,2,3]=1;candidates.append(blocked)
        self.assertEqual([a.supervision_action(t) for t in candidates],[b.supervision_action(t) for t in candidates])
        self.assertEqual([a.supervision_action(t) for t in candidates],[3,None,None,None,None,None])

    def test_schedule_and_teacher_exit_preserved(self):
        a,b=self.agent(),self.agent(False)
        for step in (0,1,50000,89999,90000,99999,100000,150000,200000):
            a.training_action_steps=b.training_action_steps=step
            self.assertEqual(a.imitation_weight(),max(1-step/100000,.1))
            self.assertEqual(a.advice_probability(step),b.advice_probability(step))
        with patch.object(a,'static_advice',side_effect=AssertionError('Teacher after advice exit')):
            a.select_action(self.helper.transition(4).state,.05,[0,1,3,4])
        self.assertTrue(AST_AUDIT['non_supervision_statements_identical'])

    def test_autonomous_evaluation_bypasses_teacher_risk_and_rng(self):
        a=self.agent();item=self.helper.transition(4)
        for step in (0,110000,200000):
            a.training_action_steps=step;before=c.digest(a.training_state_dict())
            with patch.object(a,'static_advice',side_effect=AssertionError('Teacher in evaluation')),patch.object(a,'observed_risk',side_effect=AssertionError('Risk in evaluation')):
                a.select_action(item.state,0.,[0,1,3,4])
            self.assertEqual(before,c.digest(a.training_state_dict()))

    def test_ce_checkpoint_restores_all_learning_and_diagnostic_state(self):
        a,b=self.agent(),self.agent();self.assertEqual(c.digest(a.training_state_dict()),c.digest(b.training_state_dict()))
        a.training_action_steps=100001;a.train_batch([self.helper.transition(4)]*4)
        state=copy.deepcopy(a.training_state_dict());b.load_training_state_dict(state)
        self.assertEqual(c.digest(state),c.digest(b.training_state_dict()))

    def test_target_constraint_and_no_route_input(self):
        a=self.agent();item=self.helper.transition(4)
        target=a._td_targets(torch.tensor([100.,-100.]),torch.ones(2),torch.zeros(2,5),torch.zeros(2,5),torch.ones(2,5,dtype=torch.bool))
        self.assertTrue(torch.equal(target,torch.tensor([10.,-6.])))
        self.assertFalse(item.state.spatial[4].any());self.assertFalse(item.state.scalars[2:].any())

    def test_aulc_uses_true_steps_and_does_not_extrapolate(self):
        from analyze_fixed_ce import aulc
        curve=[dict(environment_steps_total=t,safe_success_rate=v) for t,v in ((0,0.),(10062,.5),(200000,1.))]
        self.assertAlmostEqual(aulc(curve,0,10062),.25)
        self.assertIsNone(aulc(curve[:2],0,50000))
        self.assertGreater(aulc(curve,100000,200000),.5)


if __name__=='__main__':unittest.main()
