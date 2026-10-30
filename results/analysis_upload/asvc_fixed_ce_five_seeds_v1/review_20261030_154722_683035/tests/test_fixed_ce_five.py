from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import fixed_ce_five_common as c
import run_fixed_ce_five as runner
from analyze_fixed_ce_five import summarize, MEASUREMENTS
from astar_d3qn.replay.transition import Transition


class FixedCEFiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config, cls.inputs = c.configure()
        cls.old_config, _ = c.original.configure()

    def test_only_seed_metadata_changes_and_reused_seeds_cannot_train(self):
        for seed in c.ALL_SEEDS:
            self.assertEqual(asdict(c.prior.legacy.effective_training(self.config, seed, 'formal')),
                             asdict(c.prior.legacy.effective_training(self.old_config, seed, 'formal')))
        for seed in (0, 1, 5):
            with self.assertRaises(ValueError):
                runner.run_one(seed)
        self.assertEqual(c.make_agent.__code__, c.original.make_agent.__code__)

    def test_new_seed_actions_updates_and_factory_match_original_ce(self):
        for seed in c.NEW_SEEDS:
            new = c.make_agent(self.config, self.inputs[0], seed, 'cpu')
            with patch.object(c.original, 'SEEDS', c.ALL_SEEDS):
                old = c.original.make_agent(self.old_config, self.inputs[0], seed, 'cpu')
            self.assertEqual(c.digest(new.training_state_dict()), c.digest(old.training_state_dict()))
            kwargs = dict(max_steps=300, window_size=15, reward_config=c.prior.legacy._reward_config(self.config), terminate_on_collision=True)
            environments = [c.prior.legacy.runtime.make_factory(cfg, *self.inputs, method='unguided', seed=seed)(self.inputs[0], **kwargs)
                            for cfg in (self.config, self.old_config)]
            observations = [env.reset() for env in environments]
            for step in (500, 100001, 199999):
                new.training_action_steps = old.training_action_steps = step
                transitions, actions = [], []
                for agent, env, obs in zip((new,old), environments, observations):
                    action = agent.select_action(obs, .05, list(np.flatnonzero(env.action_mask(True))))
                    result = env.step(action)
                    actions.append(action)
                    transitions.append(Transition(obs, action, result.reward, result.observation, result.terminated,
                                                  np.asarray(env.action_mask(True))))
                self.assertEqual(actions[0], actions[1])
                self.assertEqual(c.digest(transitions[0]), c.digest(transitions[1]))
                for agent, transition in zip((new,old), transitions):
                    agent.train_batch([transition]*4)
                self.assertEqual(c.digest(new.training_state_dict()), c.digest(old.training_state_dict()))
                observations = [t.next_state for t in transitions]

    def test_reporting_uses_same_completed_seed_pairs(self):
        records = []
        for method in ('margin','ce'):
            for seed in c.ALL_SEEDS:
                finished = method=='margin' or seed<2
                records.append(dict(method=method,seed=seed,run_complete=finished,curve_complete=finished,
                                    **{key: .8+seed*.01 for key in MEASUREMENTS}))
        matched, summary, _, paired = summarize(records)
        self.assertEqual(matched,[0,1])
        self.assertTrue(all(r['completed_paired_training_seeds']==2 and not r['all_five_complete'] for r in summary))
        self.assertTrue(all(abs(r['mean']-.805)<1e-12 for r in summary))
        self.assertTrue(all(r['mean']==0. for r in paired))

    def test_process_local_binding_restores_original_runner(self):
        import run_fixed_ce
        import analyze_fixed_ce
        original_common, original_analyze = run_fixed_ce.c, analyze_fixed_ce.analyze
        with runner.installed():
            self.assertIs(run_fixed_ce.c,c)
            self.assertIsNot(analyze_fixed_ce.analyze, original_analyze)
        self.assertIs(run_fixed_ce.c,original_common)
        self.assertIs(analyze_fixed_ce.analyze,original_analyze)


if __name__ == '__main__':
    unittest.main()
