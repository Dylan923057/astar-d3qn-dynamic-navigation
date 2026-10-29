"""TD-only mechanism and exact recovery checks; no research training runs."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import td_control_common as c
from td_control_engine import TDControlAgent,TDControlSession
from multitask_external_agents import ExternalD3QNAgent
from multitask_external_engine import ExternalSession
from multitask_pilot_engine import state_digest
from multitask_pilot_domain import make_env
from diagnose_multitask_pairs import network_suite
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.utils.seed import seed_everything


class TDControlChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.cfg,cls.data,cls.planner=c.shared()

    def agent(self,device='cpu'):
        values=dict(self.cfg['agent']);values['spatial_shape']=tuple(values['spatial_shape']);values['device']=device
        seed_everything(0);return TDControlAgent(D3QNConfig(seed=0,**values))

    def test_scope_and_shared_frozen_settings(self):
        c.previous.verify();cfg=c.config()
        self.assertEqual(cfg['experiment']['seeds'],[0,1]);self.assertFalse(any(cfg['scope'].values()))
        self.assertEqual(self.cfg['training']['environment_steps'],500000);self.assertEqual(self.cfg['training']['replay_capacity'],20000)
        self.assertEqual(self.cfg['training']['batch_size'],64);self.assertEqual(self.cfg['training']['learning_starts'],500)
        self.assertEqual(self.cfg['agent']['gamma'],.99);self.assertEqual(self.cfg['agent']['target_sync_interval'],250)

    def test_complete_target_clipping_and_terminal_semantics(self):
        agent=self.agent();q=torch.tensor([[100.]*5,[-100.]*5,[100.]*5,[100.]*5,[2.]*5])
        reward=torch.tensor([1.,-2.,10.,-1.,-.01]);terminated=torch.tensor([False,False,True,True,False])
        actual=agent._td_targets(reward,terminated,q,q,None)
        np.testing.assert_allclose(actual.numpy(),[10.,-6.,10.,-1.,1.97],atol=1e-6)
        self.assertGreater(agent.last_targets['raw_target_max'],10);self.assertLess(agent.last_targets['raw_target_min'],-6)
        self.assertEqual(agent.last_targets['clipped_target_min'],-6);self.assertEqual(agent.last_targets['clipped_target_max'],10)
        self.assertAlmostEqual(agent.last_targets['target_clip_fraction'],.4,places=6)

    def test_masked_double_dqn_bootstrap(self):
        agent=self.agent();policy=torch.tensor([[2.,1000.,1.,0.,0.]]);target=torch.tensor([[20.,999.,0.,0.,0.]])
        mask=torch.tensor([[True,False,True,True,True]])
        y=agent._td_targets(torch.tensor([1.]),torch.tensor([False]),policy,target,mask)
        self.assertEqual(y.item(),10);self.assertAlmostEqual(agent.last_targets['raw_target_max'],20.8,places=5)

    def test_q_outputs_not_clipped_and_no_teacher_collection(self):
        session=TDControlSession(self.cfg,self.data,self.planner,0,'cpu');session.start_episode();agent=session.agent
        with torch.no_grad():
            for parameter in agent.policy_network.parameters():parameter.zero_()
            agent.policy_network.value_stream[-1].bias.fill_(42.)
            q=agent.policy_network(torch.tensor(session.state.spatial).unsqueeze(0),torch.tensor(session.state.scalars).unsqueeze(0))
        self.assertTrue(torch.all(q==42))
        with patch('multitask_pilot_domain.hard_gate',side_effect=AssertionError('teacher used')),patch('multitask_pilot_domain.observation_risk',side_effect=AssertionError('risk used')):
            action=agent.collect_action(session.state,object(),object(),0.)
        self.assertIn(action,range(5));self.assertIs(TDControlAgent.collect_action,ExternalD3QNAgent.collect_action)
        self.assertIsInstance(session.replay,UniformReplayBuffer);self.assertEqual(len(session.replay),0)
        self.assertIsNone(agent.prediction_head);self.assertFalse(session.collector.pending)
        self.assertEqual(session.state.spatial[4].sum(),0);self.assertTrue(np.all(session.state.scalars[2:]==0))

    def test_raw_initialization_and_inactive_clip_update_match(self):
        cfg=copy.deepcopy(self.cfg);cfg['training']['learning_starts']=100000
        bounded=TDControlSession(cfg,self.data,self.planner,0,'cpu')
        raw=ExternalSession(cfg,self.data,self.planner,'d3qn',0,c.previous.config()['dqfd'],'cpu')
        left=bounded.agent.training_state_dict();left.pop('td_control')
        self.assertEqual(state_digest(left),state_digest(raw.agent.training_state_dict()))
        for _ in range(64):
            left_transition,_=bounded.advance();right_transition,_=raw.advance()
            self.assertEqual((left_transition.action,left_transition.reward,left_transition.task_id),
                (right_transition.action,right_transition.reward,right_transition.task_id))
        batch=bounded.replay.sample(64);raw_batch=raw.replay.sample(64)
        bounded.agent.train_task_batch(batch);raw.agent.train_task_batch(raw_batch)
        self.assertEqual(bounded.agent.last_targets['target_clip_fraction'],0.)
        left=bounded.agent.training_state_dict();left.pop('td_control')
        self.assertEqual(state_digest(left),state_digest(raw.agent.training_state_dict()))
        self.assertEqual(len(bounded.agent.target_audit_rows),64)

    def test_actual_timeout_keeps_bootstrap(self):
        cfg,data,planner=c.shared();task=data['pools']['validation'][0];scene=c.read(c.previous.pilot.REG/'validation_scenes.json')[0]
        env=make_env(task,scene,planner,cfg);env.reset();env.max_steps=1
        result=env.step(4);self.assertTrue(result.truncated);self.assertFalse(result.terminated)
        agent=self.agent();q=torch.ones((1,5))*2
        target=agent._td_targets(torch.tensor([result.reward]),torch.tensor([result.terminated]),q,q,None)
        self.assertAlmostEqual(target.item(),result.reward+.99*2,places=6)

    def test_cpu_complete_recovery(self):self.recovery('cpu')

    def test_cuda_batch64_complete_recovery(self):
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        self.recovery('cuda')

    def recovery(self,device):
        cfg=copy.deepcopy(self.cfg);cfg['training'].update(batch_size=64,learning_starts=64,replay_capacity=100)
        a=TDControlSession(cfg,self.data,self.planner,0,device)
        for _ in range(70):a.advance()
        path=c.RESULTS/'checks'/f'recovery_{device}.pth';path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(a.snapshot('preflight'),path)
        expected=[]
        for _ in range(4):
            transition,metrics=a.advance();expected.append((transition.action,transition.reward,transition.task_id,metrics))
        digest=state_digest(a.agent.training_state_dict())
        b=TDControlSession(cfg,self.data,self.planner,0,device)
        b.restore(torch.load(path,map_location='cpu',weights_only=False),'preflight')
        actual=[]
        for _ in range(4):
            transition,metrics=b.advance();actual.append((transition.action,transition.reward,transition.task_id,metrics))
        self.assertEqual(expected,actual);self.assertEqual(digest,state_digest(b.agent.training_state_dict()))
        self.assertEqual(a.replay.state_dict()['rng_state'],b.replay.state_dict()['rng_state'])
        self.assertEqual(a.offline_updates,b.offline_updates);self.assertEqual(a.online_updates,b.online_updates)

    def test_evaluation_is_network_only_and_no_update(self):
        cfg,data,planner=c.shared();agent=self.agent();scenes=c.read(c.previous.pilot.REG/'validation_scenes.json')
        with (patch.object(agent,'collect_action',side_effect=AssertionError('collection used')),
              patch('multitask_pilot_domain.hard_gate',side_effect=AssertionError('teacher used'))):
            details,_=network_suite(agent,cfg,data,planner,scenes,'static',{scenes[0]['task_id']})
        self.assertEqual(len(details),1);self.assertEqual(agent.update_steps,0);self.assertEqual(agent.action_steps,0)


if __name__=='__main__':unittest.main(verbosity=2)
