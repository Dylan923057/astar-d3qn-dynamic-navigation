from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_value_repair as entry
from astar_d3qn.agents.astar_training_handover import AStarTrainingHandoverAgent
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent, METHODS
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig, double_dqn_targets
from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.seed import seed_everything


class ValueRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = NavigationProblem("test", 0, 7, (3, 0), (3, 6), frozenset(),
                                       tuple(astar_path((3, 0), (3, 6), frozenset(), 7)))

    def agent(self, method):
        seed_everything(0)
        return AStarValueRepairAgent(D3QNConfig((5, 5, 5), 4, 5, hidden_dim=16, device="cpu"),
                                    self.problem, repair_method=method)

    def transition(self):
        env = PathGuidanceEnvironment(DynamicGridNavigationEnv(self.problem, (), window_size=5), enabled=False)
        state = env.reset()
        result = env.step(int(Action.RIGHT))
        return Transition(state, int(Action.RIGHT), result.reward, result.observation, result.terminated,
                          np.asarray(env.action_mask(True)))

    def test_initial_states_paired_across_all_groups(self):
        hashes = [entry.runtime.state_digest(self.agent(m).training_state_dict()) for m in METHODS]
        self.assertEqual(len(set(hashes)), 1)

    def test_raw_targets_equal_original_and_bounds_preserve_terminals(self):
        rewards = torch.tensor([0.04, -0.06, 10., -1.])
        terminated = torch.tensor([0., 0., 1., 1.])
        next_q = torch.tensor([[1000.] * 5, [-1000.] * 5, [1000.] * 5, [1000.] * 5])
        mask = torch.ones(4, 5, dtype=torch.bool)
        expected = double_dqn_targets(rewards, terminated, .99, next_q, next_q, mask)
        raw = self.agent("advice_raw")._td_targets(rewards, terminated, next_q, next_q, mask)
        bounded = self.agent("advice_bound")._td_targets(rewards, terminated, next_q, next_q, mask)
        self.assertTrue(torch.equal(raw, expected))
        self.assertTrue(torch.equal(bounded, torch.tensor([10., -6., 10., -1.])))

    def test_raw_update_bitwise_equals_legacy_risk_handover(self):
        repaired = self.agent("advice_raw")
        seed_everything(0)
        legacy = AStarTrainingHandoverAgent(repaired.config, self.problem, method="astar_risk_decay")
        batch = [self.transition()] * 4
        repaired.train_batch(batch)
        legacy.train_batch(batch)
        self.assertEqual(entry.runtime.state_digest(D3QNAgent.training_state_dict(repaired)),
                         entry.runtime.state_digest(D3QNAgent.training_state_dict(legacy)))

    def test_labels_require_matching_execution_and_exclude_collision_and_observed_risk(self):
        agent = self.agent("advice_bound_margin")
        transition = self.transition()
        self.assertTrue(agent.teacher_label(transition))
        self.assertFalse(agent.teacher_label(Transition(transition.state, int(Action.STAY), -.01,
                                                       transition.next_state, False)))
        self.assertFalse(agent.teacher_label(Transition(transition.state, transition.action, -1,
                                                       transition.next_state, True)))
        risky = copy.deepcopy(transition)
        risky.state.spatial[3, 2, 3] = 1
        self.assertFalse(agent.teacher_label(risky))

    def test_margin_ignores_masked_competitors(self):
        agent = self.agent("advice_raw")
        result = D3QNAgent.train_batch(agent, [self.transition()] * 4,
                                     demonstration_mask=[True] * 4, demo_margin=.8, demo_loss_weight=1,
                                     demo_competitor_mask=[[False, False, False, True, False]] * 4)
        self.assertEqual(result["demo_margin_loss"], 0)

    def test_auxiliary_loss_has_real_labels_and_updates_parameters(self):
        agent = self.agent("advice_bound_margin")
        before = entry.runtime.state_digest(D3QNAgent.training_state_dict(agent))
        metrics = agent.train_batch([self.transition()] * 4)
        self.assertEqual(metrics["teacher_label_count"], 4)
        self.assertGreater(metrics["demo_margin_loss"], 0)
        self.assertNotEqual(before, entry.runtime.state_digest(D3QNAgent.training_state_dict(agent)))

    def test_action_advice_and_margin_withdraw_on_original_clock(self):
        agent = self.agent("advice_bound_margin")
        agent.training_action_steps = 50000
        self.assertEqual(agent.imitation_weight(), .5)
        self.assertEqual(agent.advice_probability(50000), .4)
        agent.training_action_steps = 100000
        self.assertEqual(agent.imitation_weight(), 0)
        self.assertEqual(agent.advice_probability(100000), 0)
        with patch.object(agent, "teacher_label", side_effect=AssertionError("Labels computed after withdrawal")):
            metrics = agent.train_batch([self.transition()] * 4)
        self.assertEqual(metrics["teacher_label_count"], 0)
        with patch.object(agent, "static_advice", side_effect=AssertionError("Planner called after withdrawal")):
            agent.select_action(self.transition().state, .05)

    def test_unguided_controls_never_call_astar_during_training(self):
        for method in ("unguided_raw", "unguided_bound"):
            agent = self.agent(method)
            with patch.object(agent, "static_advice", side_effect=AssertionError("Planner called in control")):
                agent.select_action(self.transition().state, .5)
                agent.train_batch([self.transition()] * 4)

    def test_evaluation_never_calls_teacher_or_changes_full_state(self):
        agent = self.agent("advice_bound_margin")
        before = entry.runtime.state_digest(agent.training_state_dict())
        with patch.object(agent, "static_advice", side_effect=AssertionError("Teacher used in evaluation")):
            with entry.runtime.preserved_evaluation(agent):
                agent.select_action(self.transition().state, 0)
        self.assertEqual(before, entry.runtime.state_digest(agent.training_state_dict()))

    def test_training_state_restores_optimizer_rng_and_metrics(self):
        agent = self.agent("advice_bound_margin")
        agent.select_action(self.transition().state, .5)
        agent.train_batch([self.transition()] * 4)
        saved = copy.deepcopy(agent.training_state_dict())
        restored = self.agent("advice_bound_margin")
        restored.load_training_state_dict(saved)
        self.assertEqual(entry.runtime.state_digest(saved), entry.runtime.state_digest(restored.training_state_dict()))

    def test_probe_preserves_training_schedule_and_replay(self):
        config = entry.load_config(entry.CONFIG)
        formal = entry.effective_training(config, 0, "formal")
        probe = entry.effective_training(config, 0, "probe")
        self.assertEqual(probe.max_environment_steps, 2000)
        self.assertEqual(probe.epsilon_decay_environment_steps, 150000)
        self.assertEqual(probe.learning_starts, formal.learning_starts)
        self.assertEqual(probe.batch_size, formal.batch_size)
        self.assertEqual(probe.replay_capacity, formal.replay_capacity)

    def test_original_unguided_control_has_identical_actions_and_update(self):
        repaired = self.agent("unguided_raw")
        seed_everything(0)
        original = D3QNAgent(repaired.config)
        state = self.transition().state
        for epsilon in (1, .5, .05, 0):
            self.assertEqual(repaired.select_action(state, epsilon), original.select_action(state, epsilon))
            self.assertEqual(repaired._rng.getstate(), original._rng.getstate())
        batch = [self.transition()] * 4
        repaired.train_batch(batch)
        original.train_batch(batch)
        self.assertEqual(entry.runtime.state_digest(D3QNAgent.training_state_dict(repaired)),
                         entry.runtime.state_digest(original.training_state_dict()))

    def test_pilot_keeps_original_curriculum_clocks(self):
        config = entry.load_config(entry.PILOT_CONFIG)
        pilot = entry.effective_training(config, 0, "pilot")
        self.assertEqual(pilot.max_environment_steps, 20000)
        self.assertEqual(pilot.progress_interval_environment_steps, 5000)
        self.assertEqual(pilot.epsilon_decay_environment_steps, 150000)
        self.assertEqual(config["training_advice"]["decay_environment_steps"], 100000)
        self.assertEqual(pilot.replay_capacity, 10000)
        self.assertEqual(pilot.learning_starts, 500)
        self.assertEqual(pilot.batch_size, 64)

    def test_pilot_cli_does_not_start_formal_training(self):
        with patch.object(entry, "validate_config", return_value=None), \
             patch.object(entry, "run_training", return_value=Path("unused")) as training:
            entry.main(["--pilot", "--seeds", "0", "1"])
            self.assertEqual(training.call_args.kwargs["mode"], "pilot")
            self.assertEqual(training.call_args.args[0]["experiment"]["protocol"], "value_bound_teacher_margin_pilot_v1")

    def test_existing_output_is_preserved_and_training_not_started(self):
        config = copy.deepcopy(entry.load_config(entry.CONFIG))
        with tempfile.TemporaryDirectory() as temporary:
            config["experiment"]["output_root"] = temporary
            existing = Path(temporary) / "advice_raw/seed_0"
            existing.mkdir(parents=True)
            marker = existing / "preserve.txt"
            marker.write_text("unchanged", encoding="utf-8")
            with patch.object(entry, "train_d3qn") as training:
                with self.assertRaises(FileExistsError):
                    entry.run_training(config, None, [0], ["advice_raw"], mode="formal")
                training.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_default_only_checks(self):
        with patch.object(entry, "validate_config", return_value=None), \
             patch.object(entry, "check", return_value=({}, [])), \
             patch.object(entry.runtime, "unique_check_dir", return_value=Path("unused")), \
             patch.object(entry, "write_json"), patch.object(entry, "run_training") as training:
            entry.main([])
            training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
