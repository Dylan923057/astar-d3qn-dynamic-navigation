"""Zero-update checks for the independent one-factor DQfD control."""
from pathlib import Path
from copy import deepcopy
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_dqfd_no_pretrain as control
from astar_d3qn.training.dqfd import new_replay


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=control.common.legacy.load_config(control.CONFIG)
        cls.original,cls.inputs=control.validate(cls.config)

    def test_only_offline_algorithm_setting_changed(self):
        base=self.original['comparison'];new=self.config['comparison']
        self.assertEqual({k:v for k,v in base.items() if k!='pretraining_updates'},
                         {k:v for k,v in new.items() if k!='pretraining_updates'})
        self.assertEqual(base['pretraining_updates'],20000)
        self.assertEqual(new['pretraining_updates'],0)
        self.assertNotEqual(self.original['experiment']['output_root'],self.config['experiment']['output_root'])
        for seed in range(5):
            self.assertEqual(control.common.legacy.effective_training(self.original,seed,'formal'),
                             control.common.legacy.effective_training(self.config,seed,'formal'))

    def test_reward_or_per_change_rejected(self):
        for path in (('reward','progress'),('comparison','per_alpha'),('comparison','n_steps')):
            config=deepcopy(self.config);config[path[0]][path[1]]+=1
            with self.assertRaises(ValueError):control.validate(config)

    def test_paired_network_and_permanent_partition_no_updates(self):
        demos,manifest=control.common.load_demonstrations(self.original)
        agent=control.common.make_agent(self.config,self.inputs[0],0,'dqfd','cpu')
        self.assertEqual(control.common.paired_digest(agent),control.common.baseline_digest(self.original,self.inputs[0],0))
        self.assertFalse(agent.bound_enabled);self.assertEqual(agent.update_steps,0)
        self.assertFalse(agent.optimizer.state)
        replay=new_replay(demos,self.config['comparison'],0,10000)
        self.assertEqual(replay.demo_count,7260);self.assertEqual(len(replay),7260)
        self.assertEqual(replay.online_capacity,10000)
        batch,indices,weights,mask=replay.sample(64,.6)
        self.assertTrue(mask.all());self.assertTrue(all(i<7260 for i in indices))
        self.assertEqual(agent.update_steps,0)


if __name__=='__main__':unittest.main()
