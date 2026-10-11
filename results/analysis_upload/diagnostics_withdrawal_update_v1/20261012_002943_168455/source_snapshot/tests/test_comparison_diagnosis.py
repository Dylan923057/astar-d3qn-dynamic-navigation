from pathlib import Path
from dataclasses import replace
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ddqna_adapter_common as common
from diagnose_comparison_evidence import first_crossings
from probe_ddqna_stability import make_agent
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.transition import Transition


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=common.legacy.load_config(common.CONFIG)
        cls.problem=common.validate_config(cls.config)[0]

    def test_first_crossing_is_not_stable_streak(self):
        r=first_crossings(np.arange(6)*10000,np.array([0,.9,0,.9,.9,.9]))
        self.assertEqual(r['first_90_step'],10000)
        self.assertEqual(r['three_90_start_step'],30000)
        self.assertEqual(r['three_90_confirmation_step'],50000)
        r=first_crossings(np.arange(6)*10000,np.zeros(6))
        self.assertIsNone(r['first_90_step']);self.assertIsNone(r['three_90_confirmation_step'])

    def batch(self):
        s=Observation(np.zeros((5,15,15),dtype=np.float32),np.zeros(4,dtype=np.float32))
        return [Transition(s,4,100.,s,True,np.ones(5,dtype=bool))]*4

    def test_huber_probe_matches_original_ordinary_td_update(self):
        probe=make_agent(self.config,self.problem,0,'huber','cpu')
        original=common.make_agent(self.config,self.problem,0,'cpu')
        batch=self.batch();a=probe.train_batch(batch);b=D3QNAgent.train_batch(original,batch)
        self.assertAlmostEqual(a['loss'],b['loss'],places=5)
        for key,value in probe.policy_network.state_dict().items():
            self.assertTrue(torch.allclose(value,original.policy_network.state_dict()[key],atol=1e-7,rtol=1e-6))
        self.assertEqual(probe.update_steps,original.update_steps)

    def test_mse_and_complete_target_bound_are_explicit(self):
        huber=make_agent(self.config,self.problem,0,'huber','cpu')
        mse=make_agent(self.config,self.problem,0,'mse','cpu')
        bound=make_agent(self.config,self.problem,0,'huber_bound','cpu')
        results=[a.train_batch(self.batch()) for a in (huber,mse,bound)]
        self.assertGreater(results[1]['loss'],results[0]['loss']*90)
        self.assertLess(results[2]['loss'],11)
        self.assertEqual(bound.profiles()[0]['raw_target_max_mean'],100)
        self.assertEqual(bound.profiles()[0]['actual_target_max_mean'],10)
        self.assertEqual(huber.profiles()[0]['actual_target_max_mean'],100)

    def test_probe_does_not_allow_auxiliary_supervision(self):
        a=make_agent(self.config,self.problem,0,'huber','cpu')
        with self.assertRaises(ValueError):a.train_batch(self.batch(),demo_loss_weight=1)
        self.assertEqual(a.update_steps,0)

    def test_eight_step_trainer_integration_for_all_three_arms(self):
        inputs=common.validate_config(self.config)
        training=replace(common.legacy.effective_training(self.config,0,'formal'),
            max_environment_steps=8,learning_starts=4,batch_size=4,progress_interval_environment_steps=None,
            progress_interval=0,allow_partial_epsilon_schedule=True)
        for arm in ('huber','mse','huber_bound'):
            agent=make_agent(self.config,self.problem,0,arm,'cpu')
            factory=common.legacy.runtime.make_factory(self.config,*inputs,method='unguided',seed=0)
            result=common.legacy.train_d3qn([self.problem],agent,training,demonstrations=(),
                reward_config=common.legacy._reward_config(self.config),environment_factory=factory)
            self.assertEqual((result.environment_steps,result.gradient_updates),(8,5))


if __name__=='__main__':unittest.main()
