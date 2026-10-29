"""Mechanism/recovery micro checks; no additional research training runs."""
from __future__ import annotations

import copy
from dataclasses import replace
import inspect
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import multitask_external_common as c
from multitask_external_agents import make_agent,StatefulDQfDReplay
from multitask_external_demo_reader import load_records
from multitask_external_engine import ExternalSession
from multitask_pilot_engine import state_digest
from multitask_pilot_domain import make_env
from diagnose_multitask_pairs import behavior_metrics,reference_groups,existing_dynamic,network_suite
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.agents.dqfd import DQfDAgent
from astar_d3qn.replay.transition import Transition
from astar_d3qn.replay.dqfd import NStepCollector
from astar_d3qn.utils.seed import seed_everything


class ExternalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg,cls.data,cls.planner=c.data();cls.settings=c.config()['dqfd']
        cls.demos=load_records(cls.settings)[:128]

    def test_shared_frozen_input_and_scope(self):
        c.pilot.verify_freeze()
        self.assertEqual(c.RUNS,(('d3qn',0),('dqfd',0),('d3qn',1),('dqfd',1)))
        self.assertFalse(any(c.config()['future_training'].values()))
        self.assertEqual(self.cfg['training']['environment_steps'],500000)
        self.assertEqual(self.cfg['training']['replay_capacity'],20000)
        self.assertEqual(self.settings['pretraining_updates'],20000)

    def test_demonstrations_cover_only_frozen_training_tasks(self):
        attempts=c.read(c.REG/'demonstration_attempts.json');report=c.read(c.REG/'demonstration_report.json')
        train={t['task_id']:t for t in self.data['pools']['train']}
        held={tuple(g) for g in self.data['held_out_goals']}
        self.assertEqual(len(attempts),480);counts={key:0 for key in train}
        for a in attempts:
            self.assertIn(a['task_id'],train);counts[a['task_id']]+=1
            self.assertNotIn(tuple(a['goal']),held)
            self.assertEqual(a['goal'],train[a['task_id']]['goal'])
        self.assertTrue(all(v==2 for v in counts.values()))
        self.assertEqual(report['retained_episodes']+report['failed_episodes'],480)
        self.assertFalse(report['validation_or_test_used']);self.assertFalse(report['old_fixed_goal_demonstrations_used'])
        self.assertEqual(report['attempted_environment_steps'],sum(a['steps'] for a in attempts))

    def test_replay_partition_and_state(self):
        replay=StatefulDQfDReplay(self.demos[:64],online_capacity=20,seed=2)
        for record in self.demos[64:94]:replay.add(record)
        self.assertEqual(replay.demo_count,64);self.assertEqual(len(replay),84)
        snapshot=copy.deepcopy(replay.state_dict())
        _,expected,weights,mask=replay.sample(32,.7)
        restored=StatefulDQfDReplay(self.demos[:64],online_capacity=20,seed=99);restored.load_state_dict(snapshot)
        _,actual,w,m=restored.sample(32,.7)
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(w,weights);np.testing.assert_array_equal(m,mask)
        self.assertTrue(np.all(weights<=1));self.assertEqual(restored.demo_count,64)

    def test_n_step_flush_prevents_goal_mixing_and_timeout_bootstraps(self):
        first=self.demos[0].transition;second=replace(first,task_id='different_task',reward=100.)
        collector=NStepCollector(10,.99)
        record=replace(first,reward=1.,terminated=False,truncated=True)
        out=collector.add(record,episode_end=True)
        self.assertEqual(len(out),1);self.assertFalse(out[0].n_terminated);self.assertEqual(out[0].n_return,1.)
        out2=collector.add(second,episode_end=True)
        self.assertEqual(out2[0].transition.task_id,'different_task');self.assertEqual(out[0].n_return,1.)
        self.assertFalse(collector.pending)

    def test_raw_d3qn_matches_original_update_without_bound(self):
        seed_everything(0);agent=make_agent(self.cfg,self.settings,'d3qn',0,'cpu')
        seed_everything(0);original=D3QNAgent(agent.config)
        batch=[r.transition for r in self.demos[:64]]
        plain=[Transition(t.state,t.action,t.reward,t.next_state,t.terminated,t.next_action_mask) for t in batch]
        agent.train_task_batch(batch);original.train_batch(plain)
        adapted=agent.training_state_dict();adapted.pop('external')
        self.assertEqual(state_digest(adapted),state_digest(original.training_state_dict()))
        raw=agent._td_targets(torch.tensor([0.]),torch.tensor([False]),torch.ones((1,5))*100,torch.ones((1,5))*100,None)
        self.assertGreater(raw.item(),10)
        source=inspect.getsource(type(agent).collect_action)
        for prohibited in ('planner','hard_gate','observation_risk','teacher'):self.assertNotIn(prohibited,source)

    def test_dqfd_logging_preserves_original_objective_exactly(self):
        seed_everything(0);agent=make_agent(self.cfg,self.settings,'dqfd',0,'cpu')
        seed_everything(0);original=DQfDAgent(agent.config,self.settings,bound=False)
        records=self.demos[:64];weights=np.ones(64,dtype=np.float32);mask=np.ones(64,dtype=bool)
        a=agent.train_logged(records,weights,mask);b=original.train_dqfd(records,weights,mask)
        adapted=agent.training_state_dict();adapted.pop('external')
        self.assertEqual(state_digest(adapted),state_digest(original.training_state_dict()))
        self.assertEqual(a['loss'],b['loss']);self.assertFalse(agent.bound_enabled)
        target,raw=agent.complete_target(torch.tensor([0.]),torch.tensor([.99]),torch.tensor([False]),torch.ones((1,5))*100,torch.ones((1,5))*100,None)
        self.assertEqual(target.item(),raw.item());self.assertGreater(target.item(),10.)

    def test_complete_cpu_recovery(self):self.recovery('cpu')

    def test_complete_cuda_batch64_recovery(self):
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        self.recovery('cuda',64)

    def recovery(self,device,batch=4):
        for method in ('d3qn','dqfd'):
            cfg=copy.deepcopy(self.cfg);cfg['training'].update(batch_size=batch,learning_starts=batch,replay_capacity=100)
            a=ExternalSession(cfg,self.data,self.planner,method,0,self.settings,device,demos=self.demos[:64])
            if method=='dqfd':
                for _ in range(3):a.update(offline=True)
                a.flush_metrics();a.phase='online'
            for _ in range(max(batch,9)):a.advance()
            path=c.RESULTS/'checks'/f'{method}_{device}_full_checkpoint.pth';path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(a.snapshot('unit_check'),path)
            expected=[]
            for _ in range(4):
                item,metrics=a.advance();expected.append((item.action,item.reward,item.task_id,metrics))
            expected_state=state_digest(a.agent.training_state_dict())
            b=ExternalSession(cfg,self.data,self.planner,method,0,self.settings,device,demos=self.demos[:64])
            b.restore(torch.load(path,map_location='cpu',weights_only=False),'unit_check')
            actual=[]
            for _ in range(4):
                item,metrics=b.advance();actual.append((item.action,item.reward,item.task_id,metrics))
            self.assertEqual(expected,actual);self.assertEqual(expected_state,state_digest(b.agent.training_state_dict()))
            self.assertEqual(a.offline_updates,b.offline_updates);self.assertEqual(a.online_updates,b.online_updates)
            if method=='dqfd':
                np.testing.assert_array_equal(a.replay.sums,b.replay.sums);np.testing.assert_array_equal(a.replay.mins,b.replay.mins)
                self.assertEqual(a.replay.rng.bit_generator.state,b.replay.rng.bit_generator.state)
                self.assertEqual(len(a.collector.pending),len(b.collector.pending))

    def test_movement_loop_and_waiting_are_distinct(self):
        from multitask_pilot_domain import GoalFields
        planner=GoalFields(20,[])
        positions=[(5,5),(5,6)]*6+[(5,5)]
        row=behavior_metrics(positions,[3,2]*6,(10,10),planner)
        self.assertTrue(row['periodic_movement']);self.assertEqual(row['cycle_period'],2)
        self.assertGreater(row['revisit_move_ratio'],.5);self.assertEqual(row['wait_fraction'],0.)
        wait=behavior_metrics([(5,5)]*11,[4]*10,(10,10),planner)
        self.assertFalse(wait['periodic_movement']);self.assertEqual(wait['wait_fraction'],1.)
        self.assertEqual(wait['final_static_distance'],10)

    def test_reference_grouping_uses_same_rule_without_policy(self):
        scenes=c.read(c.pilot.REG/'validation_scenes.json')[:3]
        a=reference_groups(self.cfg,self.data,self.planner,scenes);b=reference_groups(self.cfg,self.data,self.planner,scenes)
        self.assertEqual(a,b);self.assertEqual(len(a),3)
        self.assertNotIn('agent',inspect.signature(reference_groups).parameters)

    def test_static_pair_keeps_task_map_masks_and_timeout(self):
        task=self.data['pools']['validation'][0];scene=c.read(c.pilot.REG/'validation_scenes.json')[0]
        dynamic=make_env(task,scene,self.planner,self.cfg)
        static=make_env(task,dict(scene,obstacles=[]),self.planner,self.cfg)
        x,y=dynamic.reset(),static.reset()
        self.assertEqual(dynamic.problem,static.problem);self.assertEqual(dynamic.max_steps,static.max_steps)
        np.testing.assert_array_equal(dynamic.action_mask(True),static.action_mask(True))
        np.testing.assert_array_equal(x.spatial[0],y.spatial[0]);np.testing.assert_array_equal(x.scalars,y.scalars)
        self.assertEqual(y.spatial[1:4].sum(),0)

    def test_dynamic_scores_reused_without_policy_decisions(self):
        info=c.model_info('C',0);scenes=c.read(c.pilot.REG/'validation_scenes.json')
        original=D3QNAgent.select_action
        def forbidden(*args,**kwargs):raise AssertionError('Dynamic reuse must not run a policy')
        D3QNAgent.select_action=forbidden
        try:details,traces=existing_dynamic(info,self.cfg,self.data,self.planner,scenes)
        finally:D3QNAgent.select_action=original
        self.assertEqual(len(details),50);self.assertLess(len(traces),50)
        missing=[r for r in details if not r['trace_metrics_available']]
        self.assertTrue(missing);self.assertTrue(all(r['periodic_movement_ratio'] is None for r in missing))
        self.assertAlmostEqual(np.mean([r['success'] for r in details]),.76)

    def test_network_evaluation_bypasses_collection(self):
        seed_everything(0);agent=make_agent(self.cfg,self.settings,'d3qn',0,'cpu')
        def forbidden(*args,**kwargs):raise AssertionError('Collection/teacher path used during evaluation')
        agent.collect_action=forbidden
        scenes=c.read(c.pilot.REG/'validation_scenes.json')
        result,_=network_suite(agent,self.cfg,self.data,self.planner,scenes,'static',{scenes[0]['task_id']})
        self.assertEqual(len(result),1);self.assertEqual(agent.action_steps,0);self.assertEqual(agent.update_steps,0)


if __name__=='__main__':unittest.main(verbosity=2)
