from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_staged_withdrawal_pilot as entry
from astar_d3qn.agents.astar_staged_withdrawal import AStarStagedWithdrawalAgent, staged_advice_probability
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.core.astar import astar_path
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.trainer import TrainingConfig, train_d3qn
from astar_d3qn.utils.seed import seed_everything


class StagedWithdrawalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem=NavigationProblem('toy',0,7,(3,0),(3,6),frozenset(),
            tuple(astar_path((3,0),(3,6),frozenset(),7)))

    def agent(self,original=False):
        seed_everything(0)
        cfg=D3QNConfig((5,15,15),4,5,hidden_dim=16,device='cpu')
        if original: return AStarValueRepairAgent(cfg,self.problem,repair_method='advice_bound_margin')
        return AStarStagedWithdrawalAgent(cfg,self.problem)

    def transition(self):
        env=PathGuidanceEnvironment(DynamicGridNavigationEnv(self.problem,(),window_size=15),enabled=False)
        state=env.reset(); result=env.step(3)
        return Transition(state,3,result.reward,result.observation,result.terminated,env.action_mask(True))

    def test_no_advice_at_80k_but_same_supervision_until_100k(self):
        agent,old=self.agent(),self.agent(True)
        for t in (0,60000,79999,80000,90000,99999,100000,200000):
            agent.training_action_steps=old.training_action_steps=t
            self.assertEqual(agent.imitation_weight(),old.imitation_weight())
        agent.training_action_steps=79999
        with patch.object(agent,'static_advice',side_effect=AssertionError('Teacher selection after exit')):
            agent.select_action(self.transition().state,.05,[1,3,4])
        self.assertEqual(agent.advice_records()[0]['requested'],0)
        self.assertGreater(agent.imitation_weight(),0)
        for t in (80000,90000,100000,200000): self.assertEqual(agent.advice_probability(t),0)

    def test_same_initialization_and_same_gradient_updates(self):
        for step in (0,60000,80000,90000,100000):
            agent,old=self.agent(),self.agent(True)
            self.assertEqual(entry.common.legacy.runtime.state_digest(agent.training_state_dict()),
                             entry.common.legacy.runtime.state_digest(old.training_state_dict()))
            agent.training_action_steps=old.training_action_steps=step
            batch=[self.transition() for _ in range(4)]
            self.assertEqual(agent.train_batch(batch),old.train_batch(batch))
            self.assertEqual(entry.common.legacy.runtime.state_digest(agent.training_state_dict()),
                             entry.common.legacy.runtime.state_digest(old.training_state_dict()))
        self.assertEqual(agent.td_records()[0]['updates'],1)

    def test_evaluation_skips_teacher_and_preserves_state(self):
        agent=self.agent(); state=self.transition().state
        before=entry.common.legacy.runtime.state_digest(agent.training_state_dict())
        with patch.object(agent,'static_advice',side_effect=AssertionError('Teacher in evaluation')):
            agent.select_action(state,0,[1,3,4])
        self.assertEqual(before,entry.common.legacy.runtime.state_digest(agent.training_state_dict()))
        self.assertEqual(agent.planner_calls,{})

    def test_schedule_continuity_and_finite_input(self):
        self.assertAlmostEqual(staged_advice_probability(60000),staged_advice_probability(60000+1e-6),places=10)
        self.assertGreater(staged_advice_probability(79999),0)
        self.assertEqual(staged_advice_probability(80000),0)
        values=[staged_advice_probability(t) for t in range(0,200001,100)]
        self.assertTrue(np.all(np.diff(values)<=0))
        for t in (-1,np.nan,np.inf):
            with self.assertRaises(ValueError): staged_advice_probability(t)

    def test_passive_label_census_preserves_updates_and_rng(self):
        def factory(problem,**kwargs):
            return PathGuidanceEnvironment(DynamicGridNavigationEnv(problem,(),**kwargs),enabled=False)
        factory.reset_schedule=lambda:None
        cfg=TrainingConfig(episodes=20,max_environment_steps=8,max_steps=10,batch_size=4,learning_starts=4,
            epsilon_decay_environment_steps=150000,allow_partial_epsilon_schedule=True,replay_strategy='uniform',
            window_size=15,mask_static_invalid_actions=True,progress_interval=0,diagnostic_interval=0,seed=0)
        a,b=self.agent(),self.agent()
        census=entry.LabelCensusFactory(factory,b)
        first=train_d3qn([self.problem],a,cfg,environment_factory=factory)
        second=train_d3qn([self.problem],b,cfg,environment_factory=census)
        self.assertEqual(first.episode_records,second.episode_records)
        self.assertEqual(entry.common.legacy.runtime.state_digest(a.training_state_dict()),
                         entry.common.legacy.runtime.state_digest(b.training_state_dict()))
        self.assertEqual(sum(r['collected'] for r in census.records()),8)
        self.assertGreater(b.planner_calls['diagnostic_lookups'],0)

    def test_default_never_trains_and_train_refuses_existing_outputs(self):
        config=entry.common.legacy.load_config(entry.CONFIG)
        with patch.object(entry,'validate',return_value=None),patch.object(entry,'register',return_value=({}, {}, Path('unused'))), \
             patch.object(entry.common.legacy.runtime,'unique_check_dir',return_value=Path('unused')), \
             patch.object(entry,'write_json'),patch.object(entry,'write_records_csv'),patch.object(entry,'run') as training:
            entry.main([]); training.assert_not_called()
        with patch.object(Path,'exists',return_value=True),patch.object(entry.common.legacy,'train_d3qn') as training:
            with self.assertRaises(FileExistsError): entry.run(config,None,{},Path('unused'))
            training.assert_not_called()


if __name__=='__main__': unittest.main()
