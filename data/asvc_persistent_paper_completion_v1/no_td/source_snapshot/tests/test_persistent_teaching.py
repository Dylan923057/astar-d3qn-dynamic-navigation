from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

from test_execution_consistency import ExecutionConsistencyTests, FixedQ
from astar_d3qn.agents.persistent_teaching import PersistentTeachingAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.utils.seed import seed_everything
import run_value_repair as legacy


class PersistentTeachingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ExecutionConsistencyTests.setUpClass()
        cls.helper = ExecutionConsistencyTests()

    def agent(self, **floors):
        seed_everything(0)
        return PersistentTeachingAgent(D3QNConfig((5, 5, 5), 4, 5, hidden_dim=16, device='cpu'),
            self.helper.problem, repair_method='advice_bound_margin', **floors)

    def test_original_prefix_is_identical(self):
        old = self.helper.agent(True)
        new = self.agent(advice_floor=.05, supervision_floor=.1)
        self.assertEqual(legacy.runtime.state_digest(old.training_state_dict()),
                         legacy.runtime.state_digest(new.training_state_dict()))
        for step in (0, 80000, 89999):
            old.training_action_steps = new.training_action_steps = step
            self.assertEqual(old.advice_probability(), new.advice_probability())
            self.assertEqual(old.imitation_weight(), new.imitation_weight())
        batch = [self.helper.transition(4)] * 4
        old.train_batch(batch); new.train_batch(batch)
        self.assertEqual(legacy.runtime.state_digest(old.training_state_dict()),
                         legacy.runtime.state_digest(new.training_state_dict()))

    def test_floors_and_independent_clocks(self):
        for af, sf in ((0, .1), (.05, 0), (.05, .1)):
            agent = self.agent(advice_floor=af, supervision_floor=sf)
            for step in (0, 90000, 93750, 100000, 150000, 200000):
                agent.training_action_steps = step
                self.assertEqual(agent.advice_probability(step), max(af, .8 * max(0, 1-step/100000)))
                self.assertEqual(agent.imitation_weight(), max(sf, max(0, 1-step/100000)))

    def test_evaluation_never_calls_teacher_even_with_floors(self):
        agent = self.agent(advice_floor=.05, supervision_floor=.1)
        transition = self.helper.transition(4)
        for step in (0, 110000, 200000):
            agent.training_action_steps = step
            before = legacy.runtime.state_digest(agent.training_state_dict())
            with patch.object(agent, 'static_advice', side_effect=AssertionError('Teacher during evaluation')):
                agent.select_action(transition.state, 0, [0, 1, 3, 4])
            self.assertEqual(before, legacy.runtime.state_digest(agent.training_state_dict()))

    def test_post100k_teacher_separate_from_true_td_action(self):
        agent = self.agent(supervision_floor=.1)
        transition = self.helper.transition(4)
        agent.training_action_steps = 110000
        agent.policy_network = FixedQ(); agent.target_network = FixedQ()
        agent.optimizer = torch.optim.SGD(agent.policy_network.parameters(), lr=0)
        metrics = agent.train_batch([transition])
        self.assertEqual(metrics['teacher_label_count'], 1)
        self.assertEqual(metrics['teacher_label_matching_count'], 0)
        self.assertEqual(metrics['teacher_margin_weight'], .1)
        self.assertAlmostEqual(metrics['td_loss'], .5*(transition.reward+.99*5-5)**2, places=6)
        self.assertTrue(agent.supervision_records()[0]['labels_computed'])

    def test_risk_veto_still_applies_to_persistent_advice(self):
        agent = self.agent(advice_floor=.05, supervision_floor=.1)
        state = self.helper.transition(4).state
        state.spatial[1, 2, 3] = 1
        agent.training_action_steps = 110000
        with patch.object(agent.advice_rng, 'random', return_value=0):
            agent.select_action(state, .05, [0, 1, 3, 4])
        row = agent.advice_records()[0]
        self.assertEqual(row['risk_veto'], 1)
        self.assertEqual(row['applied'], 0)

    def test_training_state_roundtrip(self):
        agent = self.agent(advice_floor=.05, supervision_floor=.1)
        agent.training_action_steps = 110000
        agent.train_batch([self.helper.transition(4)]*4)
        state = copy.deepcopy(agent.training_state_dict())
        restored = self.agent(advice_floor=.05, supervision_floor=.1)
        restored.load_training_state_dict(state)
        self.assertEqual(legacy.runtime.state_digest(state), legacy.runtime.state_digest(restored.training_state_dict()))
        self.assertEqual(restored.imitation_weight(), .1)


if __name__ == '__main__':
    unittest.main()
