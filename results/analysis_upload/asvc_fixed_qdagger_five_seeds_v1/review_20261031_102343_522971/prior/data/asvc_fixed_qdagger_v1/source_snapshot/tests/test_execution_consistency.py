from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import run_value_repair as legacy
from astar_d3qn.agents.d3qn import D3QNAgent,D3QNConfig
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent
from astar_d3qn.agents.astar_all_eligible_supervision import AStarAllEligibleSupervisionAgent
from astar_d3qn.agents.independent_teacher_actions import train_batch_with_teacher_actions
from astar_d3qn.core.astar import astar_path
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.seed import seed_everything


class FixedQ(nn.Module):
    def __init__(self):
        super().__init__();self.values=nn.Parameter(torch.tensor([1.,2.,3.,4.,5.]))
    def forward(self,spatial,scalars):return self.values.unsqueeze(0).expand(spatial.shape[0],-1)


class ExecutionConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem=NavigationProblem('toy',0,7,(3,0),(3,6),frozenset(),
            tuple(astar_path((3,0),(3,6),frozenset(),7)))

    def agent(self,new=False):
        seed_everything(0)
        cls=AStarAllEligibleSupervisionAgent if new else AStarValueRepairAgent
        return cls(D3QNConfig((5,5,5),4,5,hidden_dim=16,device='cpu'),self.problem,repair_method='advice_bound_margin')

    def transition(self,action=4):
        env=PathGuidanceEnvironment(DynamicGridNavigationEnv(self.problem,(),window_size=5),enabled=False)
        state=env.reset();result=env.step(action)
        return Transition(state,action,result.reward,result.observation,result.terminated,np.asarray(env.action_mask(True)))

    def test_td_true_action_and_margin_teacher_action(self):
        agent=self.agent(True);t=self.transition(4)
        agent.policy_network=FixedQ();agent.target_network=FixedQ()
        agent.optimizer=torch.optim.SGD(agent.policy_network.parameters(),lr=0)
        metrics=agent.train_batch([t])
        target=t.reward+agent.config.gamma*5
        expected_td=.5*(target-5)**2
        self.assertAlmostEqual(metrics['td_loss'],expected_td,places=6)
        self.assertAlmostEqual(metrics['q_mean'],5.,places=6)
        self.assertAlmostEqual(metrics['demo_margin_loss'],1.8,places=5)
        self.assertEqual(metrics['teacher_label_count'],1)
        self.assertEqual(metrics['teacher_label_matching_count'],0)
        self.assertEqual(t.action,4);self.assertEqual(agent.static_advice(t.state),3)

    def test_only_execution_condition_removed(self):
        old,new=self.agent(),self.agent(True);t=self.transition(4)
        self.assertFalse(old.teacher_label(t));self.assertTrue(new.teacher_label(t))
        collision=Transition(t.state,4,-1,t.next_state,True,t.next_action_mask)
        self.assertFalse(new.teacher_label(collision))
        for channel in (1,2,3):
            risky=copy.deepcopy(t);risky.state.spatial[channel,2,3]=1
            self.assertFalse(new.teacher_label(risky))
        blocked=copy.deepcopy(t);blocked.state.spatial[0,2,3]=1
        self.assertFalse(new.teacher_label(blocked))
        with patch.object(new,'static_advice',return_value=None):self.assertFalse(new.teacher_label(t))
        matched=self.transition(3)
        self.assertTrue(old.teacher_label(matched));self.assertTrue(new.teacher_label(matched))

    def test_default_interface_preserves_old_numerics(self):
        a,b=self.agent(),self.agent();batch=[self.transition(3)]*4
        kwargs=dict(demonstration_mask=[True]*4,demo_margin=.8,demo_loss_weight=1,
            demo_competitor_mask=[a.current_static_mask(t.state) for t in batch])
        m=D3QNAgent.train_batch(a,batch,**kwargs)
        n=train_batch_with_teacher_actions(b,batch,teacher_actions=None,**kwargs)
        self.assertEqual(m,{k:n[k] for k in m})
        self.assertEqual(legacy.runtime.state_digest(D3QNAgent.training_state_dict(a)),
            legacy.runtime.state_digest(D3QNAgent.training_state_dict(b)))

    def test_matched_only_batch_preserves_old_update(self):
        a,b=self.agent(),self.agent(True);batch=[self.transition(3)]*4
        m=a.train_batch(batch);n=b.train_batch(batch)
        self.assertEqual(m,{k:n[k] for k in m})
        self.assertEqual(legacy.runtime.state_digest(D3QNAgent.training_state_dict(a)),
            legacy.runtime.state_digest(D3QNAgent.training_state_dict(b)))

    def test_static_competitor_mask_unchanged(self):
        a=self.agent(True);batch=[self.transition(4)]*4
        m=train_batch_with_teacher_actions(a,batch,teacher_actions=[3]*4,
            demonstration_mask=[True]*4,demo_margin=.8,demo_loss_weight=1,
            demo_competitor_mask=[[False,False,False,True,False]]*4)
        self.assertEqual(m['demo_margin_loss'],0)

    def test_original100k_exit_and_autonomous_evaluation(self):
        a=self.agent(True);t=self.transition(4);a.training_action_steps=99999
        with patch.object(a,'static_advice',side_effect=AssertionError('A* after exit')):
            a.select_action(t.state,.05,[0,1,3,4])
            self.assertEqual(a.training_action_steps,100000)
            self.assertEqual(a.imitation_weight(),0);self.assertEqual(a.advice_probability(),0)
            m=a.train_batch([t]*4)
            self.assertEqual(m['teacher_label_count'],0);self.assertEqual(m['demo_margin_loss'],0)
            before=legacy.runtime.state_digest(a.training_state_dict())
            a.select_action(t.state,0,[0,1,3,4])
            self.assertEqual(before,legacy.runtime.state_digest(a.training_state_dict()))

    def test_restore_new_diagnostics_and_initial_full_pairing(self):
        a,b=self.agent(),self.agent(True)
        self.assertEqual(legacy.runtime.state_digest(a.training_state_dict()),legacy.runtime.state_digest(b.training_state_dict()))
        b.train_batch([self.transition(4)]*4)
        saved=copy.deepcopy(b.training_state_dict());c=self.agent(True);c.load_training_state_dict(saved)
        self.assertEqual(legacy.runtime.state_digest(saved),legacy.runtime.state_digest(c.training_state_dict()))


if __name__=='__main__':unittest.main()
