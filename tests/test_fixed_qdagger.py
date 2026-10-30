"""Scientific invariants of the independent QDagger fixed-task adaptation."""
import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fixed_qdagger_common as q
from fixed_qdagger_agent import ThreeStepCollector,coefficient,distillation,three_step_targets,ObservableTeacher,static_mask
from astar_d3qn.replay.transition import Transition


class QDaggerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.config,cls.shared,cls.inputs=q.configure()

    def transition(self,reward=1.,terminal=False):
        env=q.c.legacy.runtime.make_factory(self.shared,*self.inputs,method='unguided',seed=0)(self.inputs[0],
            max_steps=300,window_size=15,reward_config=q.c.legacy._reward_config(self.shared),terminate_on_collision=True)
        state=env.reset()
        return Transition(state,4,reward,state,terminal,env.action_mask(True))

    def test_three_step_boundaries_and_tail_discounts(self):
        collector=ThreeStepCollector(.5);all_records=[]
        for i,reward in enumerate((1,2,4,8)):
            all_records.extend(collector.add(self.transition(reward),4,7,episode_end=i==3))
        self.assertEqual([r.n_return for r in all_records],[3,6,8,8])
        self.assertEqual([r.n_length for r in all_records],[3,3,2,1])
        self.assertTrue(all(not r.n_terminated for r in all_records));self.assertFalse(collector.pending)
        records=collector.add(self.transition(99,True),4,8)
        self.assertEqual(len(records),1);self.assertEqual(records[0].n_return,99)
        collector.add(self.transition(),4,9)
        with self.assertRaises(AssertionError):collector.add(self.transition(),4,10)

    def test_terminated_vs_truncated_bootstrap_and_full_unclipped_target(self):
        returns=torch.tensor([1.,2.,3.]);lengths=torch.tensor([3.,2.,1.]);terminal=torch.tensor([True,False,False])
        policy=torch.tensor([[0.,10.,3.]]*3);target=torch.tensor([[1.,100.,5.]]*3);mask=torch.tensor([[True,True,False]]*3)
        actual=three_step_targets(returns,lengths,terminal,.5,policy,target,mask)
        torch.testing.assert_close(actual,torch.tensor([1.,27.,53.]))

    def test_masked_one_hot_no_temperature_squared(self):
        values=torch.tensor([[1.,2.,900.]],requires_grad=True);mask=torch.tensor([[True,True,False]])
        actual=distillation(values,torch.tensor([0]),mask)
        expected=torch.nn.functional.cross_entropy(torch.tensor([[10.,20.]]),torch.tensor([0]))
        torch.testing.assert_close(actual,expected);actual.backward();self.assertEqual(values.grad[0,2],0)

    def test_teacher_exit_exact(self):
        self.assertEqual(coefficient(0),3.);self.assertEqual(coefficient(60000),1.5)
        self.assertGreater(coefficient(119999),0);self.assertEqual(coefficient(120000),0);self.assertEqual(coefficient(200000),0)

    def test_true_executed_action_not_teacher_td(self):
        transition=self.transition(reward=2.,terminal=True);collector=ThreeStepCollector()
        record=collector.add(transition,3,0,episode_end=True)[0]
        agent=q.make_agent(0,'cpu')
        with torch.no_grad():
            values=agent.policy_network(torch.tensor(transition.state.spatial)[None],torch.tensor(transition.state.scalars)[None])[0]
            expected=torch.nn.functional.smooth_l1_loss(values[4],torch.tensor(2.)).item()
        metrics=agent.train_update([record],0)
        self.assertAlmostEqual(metrics['td_loss'],expected,places=6)
        self.assertEqual(metrics['distillation_loss'],0)

    def test_teacher_local_only_and_no_environment(self):
        transition=self.transition();teacher=ObservableTeacher(self.inputs[0]);state=transition.state
        expected=teacher.query(state);changed=copy.deepcopy(state)
        changed.spatial[1:4,:3,:]=1-changed.spatial[1:4,:3,:]
        changed.spatial[1:4,-3:,:]=1-changed.spatial[1:4,-3:,:]
        self.assertEqual(expected,teacher.query(changed))
        legacy=q.c.ce.prior.make_agent(self.shared,self.inputs[0],0,'supervision_floor','cpu')
        action=legacy.static_advice(state)
        original=4 if action is None or not static_mask(state)[action] or legacy.observed_risk(state,action) else action
        self.assertEqual(expected,original)

    def test_all_frozen_demos_teacher_and_episode_identity(self):
        records,audit=q.load_demos();self.assertEqual(len(records),7260)
        self.assertEqual(audit['teacher_mismatches'],0);self.assertEqual(audit['retained_episodes'],107)


if __name__=='__main__':unittest.main()
