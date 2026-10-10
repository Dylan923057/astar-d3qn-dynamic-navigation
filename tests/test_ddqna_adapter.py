"""Probability/learning/evaluation contracts; eight-step disposable integration only."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ddqna_adapter_common as common
import report_ddqna_adapter as report
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.envs.types import Observation


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg=common.legacy.load_config(common.CONFIG)
        cls.inputs=common.validate_config(cls.cfg)

    def agent_state(self,seed=0):
        agent=common.make_agent(self.cfg,self.inputs[0],seed,'cpu')
        factory=common.legacy.runtime.make_factory(self.cfg,*self.inputs,method='unguided',seed=seed)
        env=factory(self.inputs[0],max_steps=300,window_size=15,
            reward_config=common.legacy._reward_config(self.cfg),terminate_on_collision=True)
        return agent,env,env.reset()

    def test_all_five_paired_initializations_and_shared_config(self):
        for seed in common.SEEDS:
            agent=common.make_agent(self.cfg,self.inputs[0],seed,'cpu')
            self.assertEqual(common.historical.paired_digest(agent),
                common.historical.baseline_digest(self.cfg,self.inputs[0],seed))
        cfg=deepcopy(self.cfg);cfg['agent']['gamma']=.5
        with self.assertRaises(ValueError):common.validate_config(cfg)

    def test_random_priority_conditional_teacher_and_greedy_boundaries(self):
        agent,env,state=self.agent_state()
        teacher=agent.static_advice(state);agent.astar_lookups=0
        valid=list(np.flatnonzero(env.action_mask(True)))
        with patch.object(agent._rng,'random',return_value=.2999),patch.object(agent.advice_rng,'random') as teacher_rng:
            agent.select_action(state,.3,valid);teacher_rng.assert_not_called()
        with patch.object(agent._rng,'random',return_value=.3),patch.object(agent.advice_rng,'random',return_value=.4999):
            self.assertEqual(agent.select_action(state,.3,valid),teacher)
        with patch.object(agent._rng,'random',return_value=.3),patch.object(agent.advice_rng,'random',return_value=.5), \
                patch.object(D3QNAgent,'select_action',return_value=4):
            self.assertEqual(agent.select_action(state,.3,valid),4)
        r=agent.advice_records()[0]
        self.assertEqual([r[k] for k in ('random_branch','teacher_branch','greedy_branch')],[1,1,1])
        self.assertEqual(agent.astar_lookups,1)

    def test_branch_probability_distribution_and_constant_schedule(self):
        for eps in (1.,.3,.05):
            agent,env,state=self.agent_state()
            valid=list(np.flatnonzero(env.action_mask(True)))
            with patch.object(D3QNAgent,'select_action',return_value=4):
                for _ in range(30000):agent.select_action(state,eps,valid)
            bins=agent.advice_records()
            counts=[sum(r[k] for r in bins) for k in ('random_branch','teacher_branch','greedy_branch')]
            for n,p in zip(counts,(eps,(1-eps)*.5,(1-eps)*.5)):
                self.assertLessEqual(abs(n-30000*p),6*np.sqrt(30000*p*(1-p))+1)
            for step in (1,80000,100000,100001,140000,200000):
                self.assertEqual(agent.advice_probability(step),.5)

    def test_no_dynamic_risk_filter_and_static_mask_fallback(self):
        agent,env,state=self.agent_state()
        valid=list(np.flatnonzero(env.action_mask(True)));teacher=agent.static_advice(state)
        state.spatial[1:4]=1 # All observed dynamics risky; no safety veto is allowed.
        with patch.object(agent._rng,'random',return_value=.9),patch.object(agent.advice_rng,'random',return_value=0), \
                patch.object(agent,'observed_risk',side_effect=AssertionError('Risk query prohibited')):
            self.assertEqual(agent.select_action(state,.3,valid),teacher)
            self.assertEqual(agent.select_action(state,.3,[4]),4)
        self.assertEqual(agent.advice_records()[0]['teacher_unavailable'],1)

    def test_autonomous_evaluation_bypasses_teacher_and_preserves_all_state(self):
        agent,env,state=self.agent_state()
        valid=list(np.flatnonzero(env.action_mask(True)))
        with patch.object(agent,'static_advice',side_effect=AssertionError('Teacher prohibited')):
            before=common.legacy.runtime.state_digest(agent.training_state_dict())
            with common.legacy.runtime.preserved_evaluation(agent):agent.select_action(state,0,valid)
            self.assertEqual(common.legacy.runtime.state_digest(agent.training_state_dict()),before)
            agent.teacher_enabled=False
            agent.select_action(state,.3,valid)
        self.assertEqual(agent.astar_lookups,0);self.assertEqual(agent.training_action_steps,0)

    def test_full_rng_cache_and_cost_state_restore(self):
        agent,env,state=self.agent_state()
        valid=list(np.flatnonzero(env.action_mask(True)))
        with patch.object(agent._rng,'random',return_value=.9),patch.object(agent.advice_rng,'random',return_value=0):
            agent.select_action(state,.3,valid)
        saved=deepcopy(agent.training_state_dict());agent.select_action(state,.3,valid)
        agent.load_training_state_dict(saved)
        self.assertEqual(common.legacy.runtime.state_digest(saved),
            common.legacy.runtime.state_digest(agent.training_state_dict()))

    def test_ordinary_unclipped_td_and_no_added_supervision(self):
        agent,_,_=self.agent_state()
        self.assertIs(type(agent)._td_targets,D3QNAgent._td_targets)
        self.assertIs(type(agent).train_batch,D3QNAgent.train_batch)
        target=agent._td_targets(torch.tensor([100.,-100.]),torch.tensor([0.,1.]),
            torch.ones((2,5)),torch.full((2,5),30.),torch.ones((2,5),dtype=torch.bool))
        self.assertTrue(torch.allclose(target,torch.tensor([129.7,-100.])))

    def test_disposable_eight_step_five_update_integration(self):
        agent,_,_=self.agent_state()
        training=replace(common.legacy.effective_training(self.cfg,0,'formal'),max_environment_steps=8,
            learning_starts=4,batch_size=4,progress_interval_environment_steps=None,allow_partial_epsilon_schedule=True)
        factory=common.legacy.runtime.make_factory(self.cfg,*self.inputs,method='unguided',seed=0)
        result=common.legacy.train_d3qn([self.inputs[0]],agent,training,demonstrations=(),
            reward_config=common.legacy._reward_config(self.cfg),environment_factory=factory)
        self.assertEqual((result.environment_steps,result.gradient_updates,agent.training_action_steps),(8,5,8))

    def test_three_success_points_and_censoring(self):
        x=np.arange(7)*10000;y=np.array([.89,.9,.92,.9,.1,.9,.9])
        metrics=report.learning_metrics(x,y,self.cfg['ddqna_adapter'])
        self.assertEqual(metrics['first_three_90_start_step'],10000)
        self.assertEqual(metrics['first_three_90_confirmation_step'],30000)
        metrics=report.learning_metrics(x,np.full(7,.89),self.cfg['ddqna_adapter'])
        self.assertIsNone(metrics['first_three_90_start_step']);self.assertEqual(metrics['three_90_reached'],0)

    def test_final_evaluation_waits_for_all_five_runs(self):
        with patch.object(common.tail,'load_frozen',return_value=([],[],{})), \
                patch.object(common,'audit_run',side_effect=FileNotFoundError('pending')), \
                patch.object(report,'evaluate_agent') as rollout, \
                patch.object(common.legacy.runtime,'unique_check_dir') as output:
            with self.assertRaises(FileNotFoundError):report.evaluate(self.cfg,'cpu')
            rollout.assert_not_called();output.assert_not_called()


if __name__=='__main__':unittest.main()
