from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from astar_d3qn.maps.adaptation import (
    occupancy, collision_step, safe_oracle, reactive_wait_witness, replay_witness,
    spec_from_record,
    progress_band, select_coverage_pairs,
)
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.demo import (
    IndexedRiskReplay,
    PersistentDemoReplay,
    RiskCoverageHandoverReplay,
    SafeInterventionReplay,
)
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.replay_adaptation import (
    make_agent, collect_demos, snapshot, restore, state_digest, seed_everything,
    action_risk_margin_masks, evaluate, flatten_pairs, prediction_weight_map,
    train_steps, train_branch,
)


def tiny_problem():
    path = tuple((3, col) for col in range(1, 8))
    return NavigationProblem("tiny_crossing", 0, 10, path[0], path[-1], frozenset(), path)


class ActionRiskMarginTests(unittest.TestCase):
    class RiskEnvironment:
        position = (2, 2)

        @staticmethod
        def action_mask(mask_collisions=True):
            assert mask_collisions
            return np.asarray([True, True, False, True, True], dtype=bool)

        @staticmethod
        def dynamic_action_collision_risk(action, *, predict_next=True):
            assert predict_next
            return action in {1, 3}

    def test_demo_scope_labels_only_risky_reference_action(self):
        safe, blocked = action_risk_margin_masks(
            self.RiskEnvironment(),
            {(2, 2): 3},
            scope="demo_action",
        )

        np.testing.assert_array_equal(safe, [True, False, False, False, True])
        np.testing.assert_array_equal(blocked, [False, False, False, True, False])

    def test_all_action_scope_labels_every_statically_legal_risky_action(self):
        safe, blocked = action_risk_margin_masks(
            self.RiskEnvironment(),
            {},
            scope="all_actions",
        )

        np.testing.assert_array_equal(safe, [True, False, False, False, True])
        np.testing.assert_array_equal(blocked, [False, True, False, True, False])

    def test_demo_scope_skips_online_states_outside_reference_path(self):
        masks = action_risk_margin_masks(
            self.RiskEnvironment(),
            {},
            scope="demo_action",
        )

        self.assertEqual(masks, (None, None))

    def test_prediction_weighting_changes_only_center_five_by_five(self):
        global_weights = prediction_weight_map((15, 15), mode="global_prediction")
        decision_weights = prediction_weight_map(
            (15, 15),
            mode="decision_weighted_prediction",
            decision_zone_size=5,
            decision_zone_weight=3.0,
        )

        self.assertTrue(np.all(global_weights == 1.0))
        self.assertEqual(int(np.sum(decision_weights == 3.0)), 25)
        self.assertEqual(int(np.sum(decision_weights == 1.0)), 200)

    def test_next_dynamic_channel_uses_next_agent_centered_coordinates(self):
        problem = tiny_problem()
        obstacle = DynamicObstacleSpec(
            route=((1, 4), (2, 4), (3, 4), (4, 4), (5, 4)),
            start_index=0,
            direction=1,
        )
        env = DynamicGridNavigationEnv(
            problem,
            [obstacle],
            max_steps=10,
            window_size=5,
        )
        env.reset()
        result = env.step(3)
        target = result.observation.spatial[env.current_dynamic_channel]
        expected = np.zeros((5, 5), dtype=np.float32)
        radius = 2
        for row, column in env.dynamic_positions:
            local_row = row - env.position[0] + radius
            local_column = column - env.position[1] + radius
            if 0 <= local_row < 5 and 0 <= local_column < 5:
                expected[local_row, local_column] = 1.0

        np.testing.assert_array_equal(target, expected)


class DynamicsAuditTests(unittest.TestCase):
    def test_phase_oracle_matches_actual_environment_for_both_directions(self):
        problem = tiny_problem()
        route = tuple((row, 4) for row in range(1, 6))
        for index in range(len(route)):
            for direction in (-1, 1):
                spec = DynamicObstacleSpec(route, index, direction)
                env = DynamicGridNavigationEnv(problem, [spec], max_steps=100)
                env.reset()
                for t in range(25):
                    self.assertEqual(env.dynamic_positions[0], occupancy(spec, t))
                    env.step(4)
                oracle = safe_oracle(problem, spec)
                self.assertIsNotNone(oracle)
                self.assertTrue(replay_witness(problem, spec, oracle))

    def test_conflict_requires_reference_response_and_wait_is_safe(self):
        problem = tiny_problem()
        spec = DynamicObstacleSpec(tuple((r, 4) for r in range(1, 6)))
        self.assertIsNotNone(collision_step(problem.nominal_path, spec))
        wait = reactive_wait_witness(problem, spec)
        self.assertGreater(len(wait), len(problem.nominal_path))
        self.assertTrue(replay_witness(problem, spec, wait))
        self.assertFalse(replay_witness(problem, spec, problem.nominal_path))

    def test_frozen_dataset_pairs_are_disjoint_and_all_witnesses_replay(self):
        self.check_dataset("v1")

    def test_v2_pairs_cover_all_thirds_without_repeated_positions(self):
        self.check_dataset("v2")

    def check_dataset(self, version):
        manifest = json.loads((ROOT / f"data/replay_adaptation_{version}/manifest.json").read_text(encoding="utf-8"))
        old = json.loads((ROOT / "data/replay_adaptation_v1/manifest.json").read_text(encoding="utf-8"))
        old_hashes = {entry["problem"]["map_id"]: entry["problem"]["grid_sha256"] for entry in old["maps"]}
        for entry in manifest["maps"]:
            problem = problem_from_record(entry["problem"])
            self.assertEqual(problem.grid_sha256, old_hashes[problem.map_id])
            seen = set()
            for split, pairs in entry["scenarios"]["splits"].items():
                self.assertEqual(len(pairs), manifest["design"]["pair_counts"][split])
                if version == "v2":
                    steps = sorted(pair["first_reference_collision_step"] for pair in pairs)
                    minimum_gap = manifest["design"]["coverage"]["minimum_decision_step_gap"]
                    self.assertTrue(all(b - a >= minimum_gap for a, b in zip(steps, steps[1:])))
                    bands = Counter(progress_band(step, problem.astar_steps) for step in steps)
                    self.assertEqual(bands, {band: len(pairs) // 3 for band in ("early", "middle", "late")})
                    for pair in pairs:
                        self.assertEqual(pair["progress_band"], progress_band(pair["first_reference_collision_step"], problem.astar_steps))
                for pair in pairs:
                    specs = [spec_from_record(pair[c]["obstacle"]) for c in ("control", "conflict")]
                    self.assertEqual(specs[0].route, specs[1].route)
                    for condition, spec in zip(("control", "conflict"), specs):
                        # Motion sequence canonicalizes endpoint-direction equivalents.
                        signature = tuple(occupancy(spec, t) for t in range(2 * (len(spec.route) - 1)))
                        self.assertNotIn(signature, seen)
                        seen.add(signature)
                        self.assertTrue(replay_witness(problem, spec, pair[condition]["oracle_path"]))
                    self.assertIsNone(collision_step(problem.nominal_path, specs[0]))
                    self.assertIsNotNone(collision_step(problem.nominal_path, specs[1]))
                    self.assertGreater(pair["conflict"]["demo_conflict_count"], pair["control"]["demo_conflict_count"])
                    for key in ("wait_witness", "bypass_witness"):
                        self.assertTrue(replay_witness(problem, specs[1], pair[key]))
                    # Bypass must rejoin within ten reference steps, not choose
                    # an unrelated equal-length global route to the goal.
                    hit = pair["first_reference_collision_step"]
                    bypass = list(map(tuple, pair["bypass_witness"]))
                    reference = list(problem.nominal_path)
                    self.assertTrue(any(bypass[-len(reference[j:]):] == reference[j:]
                                        for j in range(hit + 2, min(hit + 11, len(reference)))))

    def test_missing_late_candidates_fail_instead_of_filling_from_middle(self):
        candidates = [{"first_reference_collision_step": step} for step in (5, 8, 12, 25, 28, 32)]
        with self.assertRaisesRegex(RuntimeError, "late"):
            select_coverage_pairs(candidates, {"train": 3, "validation": 3}, 66, random.Random(0))

    def test_phase_variants_cannot_count_as_different_positions(self):
        candidates = [{"first_reference_collision_step": step} for step in (5, 5, 25, 25, 45, 45)]
        with self.assertRaisesRegex(RuntimeError, "early"):
            select_coverage_pairs(candidates, {"train": 6}, 66, random.Random(0))

    def test_thirds_use_absolute_path_progress_not_candidate_quantiles(self):
        self.assertEqual([progress_band(step, 66) for step in (0, 21, 22, 43, 44, 65)],
                         ["early", "early", "middle", "middle", "late", "late"])


class ForkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = yaml.safe_load((ROOT / "configs/replay_adaptation_v1.yaml").read_text(encoding="utf-8"))
        self.config.update(window_size=7, hidden_dim=16, batch_size=8, demo_episodes=2, replay_capacity=100)
        self.problem = tiny_problem()
        seed_everything(42)
        agent = make_agent(self.config, 42, "cpu")
        demos = collect_demos(self.problem, self.config)
        replay = PersistentDemoReplay(demos, 88, .25, 42)
        replay.online.extend(demos)
        # Populate Adam moments and target/update counter, not only network weights.
        for _ in range(3):
            agent.train_batch(replay.sample(8))
        self.base = snapshot(agent, replay)

    def test_all_fractions_restore_full_identical_state_and_keep_online_capacity(self):
        digest = state_digest(self.base)
        for fraction in (0.0, .1, .25):
            agent, replay = restore(self.config, self.base, fraction, 42, "cpu")
            self.assertEqual(state_digest(snapshot(agent, replay)), digest)
            self.assertEqual(replay.online.capacity, 88)
            self.assertEqual(replay.sample_counts(64), {0.0: (0, 64), .1: (6, 58), .25: (16, 48)}[fraction])

    def test_identical_branch_continuations_reproduce_actions_updates_and_buffers(self):
        results = []
        stage = dict(max_steps=12, evaluation_interval=4, epsilon_start=.5, epsilon_end=.1, epsilon_decay_steps=12)
        for _ in range(2):
            agent, replay = restore(self.config, self.base, .25, 42, "cpu")
            train_steps(agent, replay, self.problem, self.config, stage, [], 42, lambda *args: False)
            results.append(state_digest(snapshot(agent, replay)))
        self.assertEqual(results[0], results[1])

    def test_branch_updates_cannot_mutate_foundation(self):
        digest = state_digest(self.base)
        agent, replay = restore(self.config, self.base, .25, 42, "cpu")
        agent.train_batch(replay.sample(8))
        self.assertEqual(state_digest(self.base), digest)

    def test_evaluation_does_not_change_training_state_or_rng(self):
        agent, replay = restore(self.config, self.base, .25, 42, "cpu")
        config = {**self.config, "max_episode_steps": 12}
        route = [[row, 4] for row in range(1, 6)]
        scenes = [{"pair_id": "a", "scenario_id": condition, "condition": condition,
                   "reference_collision_step": 2, "oracle_steps": 6,
                   "obstacle": {"route": route, "start_index": index, "direction": 1}}
                  for condition, index in (("control", 3), ("conflict", 0))]
        before = state_digest(snapshot(agent, replay))
        evaluate(agent, self.problem, config, scenes)
        self.assertEqual(state_digest(snapshot(agent, replay)), before)

    def test_unqualified_foundation_is_not_adapted(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "not qualified"):
                train_branch(self.problem, self.config, {}, 42, "cpu", Path(temp) / "branch",
                             {"metadata": {"qualified": False}}, .25)

    def test_partial_episode_still_honors_exact_step_and_update_budget(self):
        agent, replay = restore(self.config, self.base, .25, 42, "cpu")
        initial_updates = agent.update_steps
        intervals = []
        stage = dict(max_steps=13, evaluation_interval=5, epsilon_start=.5, epsilon_end=.1, epsilon_decay_steps=13)
        result = train_steps(agent, replay, self.problem, self.config, stage, [], 42,
                             lambda step, records, metrics: intervals.append(step))
        self.assertEqual(result["steps"], 13)
        self.assertEqual(agent.update_steps - initial_updates, 13)
        self.assertEqual(intervals, [5, 10, 13])
        self.assertEqual(result["demo_samples"], 26)
        self.assertEqual(result["online_samples"], 78)


class RiskHandoverReplayTests(unittest.TestCase):
    def setUp(self):
        observation = Observation(
            spatial=np.zeros((4, 3, 3), dtype=np.float32),
            scalars=np.zeros(2, dtype=np.float32),
        )
        self.transition = Transition(observation, 0, 0.0, observation, False)

    def test_risk_coverage_replaces_demo_slots_without_changing_batch_size(self):
        replay = RiskCoverageHandoverReplay(
            [self.transition] * 32,
            online_capacity=100,
            guidance_fraction=.25,
            risk_capacity=50,
            coverage_target=4,
            seed=3,
        )
        replay.online.extend([self.transition] * 80)
        self.assertEqual(replay.partition_counts(64), (16, 0, 48))
        for index in range(2):
            replay.add_risk(self.transition, ("scene", index))
        self.assertEqual(replay.partition_counts(64), (14, 2, 48))
        for index in range(2, 16):
            replay.add_risk(self.transition, ("scene", index))
        self.assertEqual(replay.partition_counts(64), (0, 16, 48))
        self.assertEqual(len(replay.sample(64)), 64)
        self.assertEqual(replay.last_risk_sample_count, 16)

    def test_repeated_risk_key_does_not_inflate_coverage(self):
        replay = RiskCoverageHandoverReplay(
            [self.transition] * 8, 20, .25, 10, 2, seed=1
        )
        replay.add_risk(self.transition, ("same", 1))
        replay.add_risk(self.transition, ("same", 1))
        self.assertEqual(replay.covered_risk_count, 1)
        self.assertEqual(replay.risk_total_added, 2)


class SafeInterventionReplayTests(unittest.TestCase):
    def setUp(self):
        observation = Observation(
            spatial=np.zeros((4, 3, 3), dtype=np.float32),
            scalars=np.zeros(2, dtype=np.float32),
        )
        self.transition = Transition(observation, 0, 0.0, observation, False)

    def test_safe_samples_augment_decayed_demos_without_changing_batch_size(self):
        replay = SafeInterventionReplay(
            [self.transition] * 32,
            online_capacity=100,
            demo_fraction=.25,
            safe_capacity=20,
            safe_samples_per_batch=8,
            seed=3,
        )
        replay.online.extend([self.transition] * 80)
        for index in range(8):
            replay.add_safe(self.transition, (3, 0, index % 5))
        self.assertEqual(replay.partition_counts(64), (16, 8, 40))
        replay.set_demo_fraction(.10)
        self.assertEqual(replay.partition_counts(64), (6, 8, 50))
        replay.set_demo_fraction(0.0)
        self.assertEqual(replay.partition_counts(64), (0, 8, 56))
        self.assertEqual(len(replay.sample(64)), 64)
        self.assertEqual(replay.last_safe_sample_count, 8)

    def test_safe_pool_warms_up_without_duplicate_sampling(self):
        replay = SafeInterventionReplay(
            [self.transition] * 16, 100, .25, 20, 8, seed=1
        )
        replay.online.extend([self.transition] * 80)
        replay.add_safe(self.transition, (5, 0, 1))
        self.assertEqual(replay.partition_counts(64), (16, 1, 47))
        self.assertEqual(len(replay.sample(64)), 64)


class IndexedRiskReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        observation = Observation(
            spatial=np.zeros((4, 3, 3), dtype=np.float32),
            scalars=np.zeros(2, dtype=np.float32),
        )
        self.transition = Transition(observation, 0, 0.0, observation, False)
        self.config = yaml.safe_load(
            (ROOT / "configs/replay_adaptation_v1.yaml").read_text(encoding="utf-8")
        )
        self.config.update(
            window_size=7,
            hidden_dim=16,
            batch_size=8,
            demo_episodes=2,
            replay_capacity=100,
        )
        self.problem = tiny_problem()
        seed_everything(42)
        agent = make_agent(self.config, 42, "cpu")
        demos = collect_demos(self.problem, self.config)
        replay = PersistentDemoReplay(demos, 88, .25, 42)
        replay.online.extend(demos)
        for _ in range(3):
            agent.train_batch(replay.sample(8))
        self.base = snapshot(agent, replay)

    def test_risks_are_indexed_inside_online_capacity(self):
        replay = IndexedRiskReplay(
            [self.transition] * 32, 8, .25, 4, seed=3
        )
        replay.online.extend([self.transition] * 4)
        replay.initialize_online_index()
        tokens = [replay.add(self.transition) for _ in range(4)]
        replay.mark_risk(tokens[0])
        replay.mark_risk(tokens[1])
        self.assertEqual(replay.online_size, 8)
        self.assertEqual(replay.risk_size, 2)
        self.assertEqual(replay.partition_counts(8), (2, 2, 4))
        for _ in range(8):
            replay.add(self.transition)
        self.assertEqual(replay.online_size, 8)
        self.assertEqual(replay.risk_size, 0)

    def test_zero_risk_sampling_matches_persistent_replay(self):
        ordinary = PersistentDemoReplay([self.transition] * 32, 100, .25, seed=7)
        indexed = IndexedRiskReplay([self.transition] * 32, 100, .25, 0, seed=7)
        ordinary.online.extend([self.transition] * 80)
        indexed.online.load_state_dict(ordinary.online.state_dict())
        indexed.initialize_online_index()
        self.assertEqual(ordinary.sample(64), indexed.sample(64))
        self.assertEqual(ordinary._rng.getstate(), indexed._rng.getstate())
        self.assertEqual(ordinary.online._rng.getstate(), indexed.online._rng.getstate())

    def test_zero_risk_training_is_equivalent_to_time_decay(self):
        stage = dict(
            max_steps=12,
            evaluation_interval=4,
            epsilon_start=.5,
            epsilon_end=.1,
            epsilon_decay_steps=12,
        )
        digests = []
        for schedule in ("decay", "risk_sampling"):
            agent, replay = restore(
                self.config,
                self.base,
                .25,
                42,
                "cpu",
                replay_schedule=(schedule if schedule == "risk_sampling" else None),
                risk_sample_count=0,
            )
            train_steps(
                agent,
                replay,
                self.problem,
                self.config,
                stage,
                [],
                42,
                lambda *args: False,
                replay_schedule=schedule,
            )
            digests.append(state_digest(snapshot(agent, replay)))
        self.assertEqual(digests[0], digests[1])


class SummaryTests(unittest.TestCase):
    def test_generic_defer_test_keeps_baseline_and_prediction_validation_only(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_replay_adaptation import should_defer_test

        self.assertTrue(should_defer_test("decay", True))
        self.assertFalse(should_defer_test("decay", False))
        self.assertTrue(should_defer_test("global_prediction", False))

    def test_effect_sign_and_single_seed_uncertainty(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from summarize_replay_adaptation import paired_effect, bootstrap_mean
        reference = {"validation_conflict_auc": .8, "test": {"conflict_safe_success": .9,
                     "conflict_minus_control_failure": .1, "conflict_probe_unsafe_action": .2,
                     "static_safe_success": 1.0}}
        treatment = copy.deepcopy(reference)
        treatment["validation_conflict_auc"] = .6
        treatment["test"].update(conflict_safe_success=.7, conflict_minus_control_failure=.3,
                                  conflict_probe_unsafe_action=.4)
        effect = paired_effect(reference, treatment)
        for key in ("validation_auc_loss", "test_conflict_safe_success_loss",
                    "test_paired_failure_gap_increase", "test_probe_unsafe_increase"):
            self.assertAlmostEqual(effect[key], .2)
        self.assertEqual(effect["test_static_retention_loss"], 0)
        self.assertIsNone(bootstrap_mean([.2])["ci95"])


if __name__ == "__main__":
    unittest.main()
