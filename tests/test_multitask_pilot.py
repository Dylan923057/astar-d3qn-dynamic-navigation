"""Mechanism and recovery checks; these are not extra research training runs."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import multitask_pilot_common as c
from multitask_pilot_agent import PilotAgent,supervision_loss
from multitask_pilot_domain import (GoalFields,VisibleHistoryEnv,TaskTransition,hard_gate,make_env,
                                    observation_risk,scene_for,static_mask)
from multitask_pilot_engine import TrainSession,evaluate,state_digest
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv,DynamicObstacleSpec
from astar_d3qn.envs.types import Observation
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.agents.astar_training_handover import AStarTrainingHandoverAgent
from astar_d3qn.agents.d3qn import double_dqn_targets


def synthetic(position=(5,5),goal=(5,10),frames=(),size=20):
    spatial=np.zeros((5,15,15),dtype=np.float32)
    for channel,frame in enumerate(frames,1):
        for cell in frame:
            local=cell[0]-position[0]+7,cell[1]-position[1]+7
            if 0<=local[0]<15 and 0<=local[1]<15:spatial[channel,local[0],local[1]]=1
    return Observation(spatial,np.array([(goal[0]-position[0])/(size-1),(goal[1]-position[1])/(size-1),0,0],dtype=np.float32))


class PilotChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg,cls.data,cls.planner=c.load_data()

    def test_pool_isolation_and_endpoint_coverage(self):
        pools=self.data['pools'];seen=set();train=pools['train']
        starts={tuple(t['start']) for t in train};goals={tuple(t['goal']) for t in train}
        held={tuple(x) for x in self.data['held_out_goals']}
        self.assertEqual(len(held),10);self.assertEqual(len(goals),40);self.assertEqual(len(starts),64)
        for name,tasks in pools.items():
            pairs={(tuple(t['start']),tuple(t['goal'])) for t in tasks}
            self.assertTrue(seen.isdisjoint(pairs));seen|=pairs
            if name!='unseen_goal':self.assertTrue(held.isdisjoint({tuple(t['goal']) for t in tasks}))
            for t in tasks:
                self.assertEqual(t['length'],len(self.planner.path(t['start'],t['goal']))-1)
                self.assertGreaterEqual(t['length'],12)
                self.assertEqual(t['max_steps'],max(100,4*t['length']+20))
            if name=='unseen_combination':
                self.assertTrue(all(tuple(t['start']) in starts and tuple(t['goal']) in goals for t in tasks))
            if name=='unseen_goal':self.assertTrue(all(tuple(t['goal']) in held for t in tasks))

    def test_bfs_tie_rule_and_goal_map_cache(self):
        p=GoalFields(20,[])
        self.assertEqual(p.action((4,4),(5,5)),0)
        self.assertEqual(p.action((5,9),(5,5)),3)
        self.assertEqual(p.action((5,1),(5,5)),2)
        self.assertEqual(len(p.cache),3)
        self.assertTrue(all(key[0]==p.map_digest for key in p.cache))
        blocked=GoalFields(20,[(5,6)])
        self.assertNotEqual(p.map_digest,blocked.map_digest)
        self.assertNotEqual(blocked.action((5,9),(5,5)),3)
        self.assertEqual(p.path((5,5),(7,7),4),((5,5),(6,5),(7,5),(7,6),(7,7)))

    def test_historical_unseen_cells_never_appear_later(self):
        p=GoalFields(20,[])
        problem=NavigationProblem('check',0,20,(0,0),(19,19),frozenset(),p.path((0,0),(19,19)))
        spec=DynamicObstacleSpec(((0,8),(0,9)))
        env=VisibleHistoryEnv(problem,[spec],window_size=15,terminate_on_collision=True)
        first=env.reset();self.assertEqual(first.spatial[1:4].sum(),0)
        obs=env.step(3).observation
        self.assertEqual(env.position,(0,1));self.assertEqual(obs.spatial[2,7,14],0)
        self.assertEqual(obs.spatial[2:4].sum(),0)

    def test_visible_history_is_aligned_once(self):
        p=GoalFields(20,[])
        problem=NavigationProblem('check',0,20,(5,5),(19,19),frozenset(),p.path((5,5),(19,19)))
        env=VisibleHistoryEnv(problem,[DynamicObstacleSpec(((5,11),(5,12)))],window_size=15)
        env.reset();obs=env.step(2).observation
        self.assertEqual(env.position,(5,4));self.assertEqual(obs.spatial[2,7,14],1)
        self.assertEqual(obs.spatial[1].sum(),0);self.assertEqual(obs.spatial[3].sum(),0)
        self.assertTrue(np.all(obs.spatial[4]==0));self.assertTrue(np.all(obs.scalars[2:]==0))

    def test_physics_rewards_and_termination_are_legacy(self):
        task=self.data['pools']['train'][0]
        scene=scene_for(task,self.planner,self.cfg['scenes'],1234,3)
        current=make_env(task,scene,self.planner,self.cfg)
        legacy=DynamicGridNavigationEnv(current.problem,current.dynamic_obstacles,max_steps=current.max_steps,
                reward_config=RewardConfig(**self.cfg['reward']),terminate_on_collision=True,window_size=15)
        current.reset();legacy.reset()
        for _ in range(12):
            action=int(np.flatnonzero(current.action_mask(True))[0])
            a,b=current.step(action),legacy.step(action)
            self.assertEqual((a.reward,a.terminated,a.truncated,current.position),(b.reward,b.terminated,b.truncated,legacy.position))
            if a.done:break

    def test_mixed_goal_replay_uses_its_own_task(self):
        cfg=copy.deepcopy(self.cfg);p=GoalFields(20,[])
        tasks=[dict(task_id='left',goal=(5,1)),dict(task_id='right',goal=(5,9))]
        agent=PilotAgent(cfg,p,tasks,'C',0,'cpu')
        transitions=[]
        for t in tasks:
            obs=synthetic(goal=t['goal'])
            transitions.append(TaskTransition(obs,0,0.,obs,False,np.ones(5,dtype=bool),t['task_id'],(5,5),(5,5),False,False,0.,{}))
        self.assertEqual(agent.label(transitions[0]),(2,True));self.assertEqual(agent.label(transitions[1]),(3,True))
        for _ in range(3):self.assertEqual(agent.label(transitions[0]),(2,True))
        collided=TaskTransition(**{name:getattr(transitions[0],name) for name in transitions[0].__dataclass_fields__ if name!='collision'},collision=True)
        self.assertFalse(agent.label(collided)[1])
        agent.action_steps=100000;metrics=agent.train_batch([collided,transitions[1]])
        self.assertEqual(agent.replay_stats['collision_draws'],1);self.assertEqual(metrics['replay_draws'],2)

    def test_common_hard_gate_matches_legacy_observed_veto(self):
        rng=np.random.default_rng(2)
        for _ in range(50):
            obs=synthetic();obs.spatial[1:4]=(rng.random((3,15,15))<.025)
            for action in range(5):
                expected=not AStarTrainingHandoverAgent.observed_risk(SimpleNamespace(risk_radius=1),obs,action)
                self.assertEqual(hard_gate(obs,action),expected)

    def test_risk_four_step_path_and_old_new_swap_occupancy(self):
        p=GoalFields(20,[]);rc=self.cfg['risk']
        obs=synthetic(goal=(5,9),frames=(((5,8),),((5,9),),((5,10),)))
        risk,_=observation_risk(obs,(5,5),(5,9),p,rc)
        self.assertAlmostEqual(risk,.9)
        self.assertTrue(hard_gate(obs,3))
        swap=synthetic(goal=(5,9),frames=(((5,6),),((5,7),),((5,8),)))
        self.assertEqual(observation_risk(swap,(5,5),(5,9),p,rc)[0],1.)
        new=synthetic(goal=(5,9),frames=(((5,7),),((5,8),),((5,9),)))
        self.assertEqual(observation_risk(new,(5,5),(5,9),p,rc)[0],1.)

    def test_first_appearance_and_ambiguity_are_conservative(self):
        p=GoalFields(20,[])
        obs=synthetic(frames=(((5,8),),(),()))
        risk,stats=observation_risk(obs,(5,5),(5,10),p,self.cfg['risk'])
        self.assertGreater(risk,0);self.assertEqual(stats['first_appearance'],1)
        ambiguous=synthetic(frames=(((5,8),),((5,7),(5,9)),()))
        _,stats=observation_risk(ambiguous,(5,5),(5,10),p,self.cfg['risk'])
        self.assertEqual(stats['ambiguous_association'],1)
        empty=synthetic();self.assertEqual(observation_risk(empty,(5,5),(5,10),p,self.cfg['risk'])[0],0.)

    def test_risk_interface_has_no_hidden_future_access(self):
        text=inspect.getsource(observation_risk)
        for forbidden in ('dynamic_positions','_indices','_directions','_move_counters','_next_dynamic_state','dynamic_action_collision_risk'):
            self.assertNotIn(forbidden,text)
        self.assertEqual(list(inspect.signature(observation_risk).parameters),['state','position','goal','planner','config'])

    def test_loss_normalizes_by_eligible_count(self):
        q=torch.zeros((2,5),requires_grad=True);labels=torch.tensor([0,0]);eligible=torch.tensor([True,True])
        loss,_=supervision_loss(q,labels,eligible,torch.tensor([1.,0.]),1.,torch.ones((2,5),dtype=torch.bool))
        self.assertEqual(loss.item(),.5)
        loss,_=supervision_loss(q,labels,~eligible,torch.ones(2),1.,torch.ones((2,5),dtype=torch.bool))
        self.assertEqual(loss.item(),0.)

    def test_c_d_hard_screen_advice_and_unit_weight_update_equivalence(self):
        cfg=copy.deepcopy(self.cfg);session=TrainSession(cfg,self.data,self.planner,'C',0,'cpu')
        for _ in range(8):session.advance()
        left=PilotAgent(cfg,self.planner,self.data['pools']['train'],'C',0,'cpu')
        right=PilotAgent(cfg,self.planner,self.data['pools']['train'],'D',0,'cpu')
        right.load_training_state_dict(copy.deepcopy(left.training_state_dict()))
        for item in session.replay.snapshot():
            self.assertEqual(left.label(item),right.label(item))
            task=left.tasks[item.task_id]
            self.assertEqual(left.collect_action(item.state,task,item.position,.5),right.collect_action(item.state,task,item.position,.5))
        batch=list(session.replay.snapshot())
        a=left.train_batch(batch);b=right.train_batch(batch,force_unit_soft=True)
        self.assertEqual(a,b);self.assertEqual(state_digest(left.training_state_dict()),state_digest(right.training_state_dict()))

    def test_timeout_bootstraps_but_terminal_does_not(self):
        rewards=torch.tensor([-.01,-1.]);terminated=torch.tensor([False,True])
        targets=double_dqn_targets(rewards,terminated,.99,torch.ones((2,5))*2,torch.ones((2,5))*2)
        self.assertAlmostEqual(targets[0].item(),1.97,places=5);self.assertEqual(targets[1].item(),-1.)

    def test_scenario_sequences_are_method_independent_and_stratified(self):
        bands=[]
        for episode in range(12):
            task,seed=c.training_task(self.data,self.cfg,0,episode);bands.append(task['band'])
            a=scene_for(task,self.planner,self.cfg['scenes'],seed,(3,4,5)[episode%3])
            b=scene_for(task,self.planner,self.cfg['scenes'],seed,(3,4,5)[episode%3])
            self.assertEqual(a,b)
            labels=[o['label'] for o in a['obstacles']]
            self.assertIn('nominal',labels);self.assertIn('alternative',labels);self.assertIn('background',labels)
        self.assertEqual({band:bands.count(band) for band in set(bands)},{'short':4,'medium':4,'long':4})

    def test_full_mid_episode_checkpoint_resume_cpu(self):
        self.resume_check('cpu')

    def test_full_mid_episode_checkpoint_resume_cuda(self):
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        self.resume_check('cuda')

    def resume_check(self,device):
        cfg=copy.deepcopy(self.cfg);cfg['training'].update(batch_size=4,learning_starts=4,replay_capacity=30)
        first=TrainSession(cfg,self.data,self.planner,'D',0,device)
        for _ in range(9):first.advance()
        snapshot=copy.deepcopy(first.snapshot('check'))
        trace=[]
        for _ in range(9):
            item,metrics=first.advance();trace.append((item.action,item.reward,item.task_id,metrics))
        expected=state_digest(first.agent.training_state_dict())
        resumed=TrainSession(cfg,self.data,self.planner,'D',0,device);resumed.restore(snapshot,'check')
        actual=[]
        for _ in range(9):
            item,metrics=resumed.advance();actual.append((item.action,item.reward,item.task_id,metrics))
        self.assertEqual(trace,actual);self.assertEqual(expected,state_digest(resumed.agent.training_state_dict()))
        self.assertEqual(state_digest(first.replay.state_dict()),state_digest(resumed.replay.state_dict()))
        self.assertEqual(first.episode,resumed.episode)
        if first.env:self.assertEqual(first.env.state_dict(),resumed.env.state_dict())

    def test_validation_decisions_use_network_only(self):
        session=TrainSession(self.cfg,self.data,self.planner,'C',0,'cpu')
        scenes=c.read(c.REG/'validation_scenes.json');tasks={t['task_id']:t for t in self.data['pools']['validation']}
        envs=[(tasks[s['task_id']],make_env(tasks[s['task_id']],s,self.planner,self.cfg)) for s in scenes]
        original=self.planner.action
        def forbidden(*args,**kwargs):raise AssertionError('Planner was called during evaluation decisions')
        self.planner.action=forbidden
        session.agent.collect_action=forbidden
        try:
            summary,details,_=evaluate(session,scenes,envs)
            self.assertEqual(summary['episodes'],50);self.assertEqual(len(details),50)
            self.assertEqual(session.agent.update_steps,0)
        finally:self.planner.action=original

    def test_only_four_pilot_runs_are_launchable(self):
        self.assertEqual(c.RUNS,(('C',0),('D',0),('C',1),('D',1)))
        for group in ('A','B','E','F','DQfD','C-Matched'):
            with self.assertRaises(ValueError):c.run_directory(group,0)


if __name__=='__main__':unittest.main(verbosity=2)
