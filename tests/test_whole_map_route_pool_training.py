from __future__ import annotations

import json
import copy
import random
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.whole_map_route_pool import WholeMapRoutePoolEnvironmentFactory
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.utils.config import load_config
sys.path.insert(0, str(ROOT / "scripts"))
from train_whole_map_route_pool_pilot import _factory, _load_inputs, _validate_protocol, _training_config, _safe_success_aulc, _validation_diagnostics, _verified_batch_segments
from astar_d3qn.training.trainer import build_replay, train_d3qn
from astar_d3qn.training.demo_collector import collect_astar_demonstrations


class _StayAgent:
    def select_action(self, _state, epsilon=0.0, valid_actions=None):
        return 4


class WholeMapRoutePoolTrainingTests(unittest.TestCase):
    def test_fixed_25_full_capacity_matches_adaptive_and_samples_exact_batches(self):
        config = load_config(ROOT / "configs/whole_map_route_pool_91701_fixed_25_full_capacity_v1.yaml")
        config["experiment"]["method"] = "fixed_25_full_capacity"
        _validate_protocol(config, *_load_inputs(config))
        original_root = ROOT / config["diagnostic"]["paired_baseline_root"]
        original = json.loads((original_root / "adaptive/seed_0/result.json").read_text())
        for section in ("training", "dataset", "dynamic_route_pool", "environment", "reward", "agent", "demonstration", "adaptive"):
            self.assertEqual(config[section], original["config"][section])
        problem, entry, pool = _load_inputs(config)
        demos = collect_astar_demonstrations([problem], 20, 7400, max_steps=300, window_size=15, spatial_channels=4)
        demo_ids = {id(item) for item in demos}
        replay = build_replay(_training_config(config, 0), demos)
        self.assertEqual(replay.online.capacity, 10000)
        self.assertEqual(replay.demonstration_size, 1320)
        self.assertEqual(replay.sample_counts(64), (16, 48))
        class Agent:
            config = SimpleNamespace(spatial_shape=(4, 15, 15), scalar_dim=2)
            action_dim = 5
            batches = []
            def select_action(self, state, epsilon=0.0, valid_actions=None):
                return 4
            def train_batch(self, batch, **kwargs):
                self.batches.append(sum(id(item) in demo_ids for item in batch))
                return {"loss":0.0, "safe_guidance_margin_loss":0.0,
                        "safe_guidance_batch_count":0, "conflict_margin_loss":0.0,
                        "conflict_margin_batch_count":0}
        agent = Agent()
        training = replace(_training_config(config, 0), max_environment_steps=512,
                           epsilon_decay_environment_steps=512)
        result = train_d3qn([problem], agent, training, demonstrations=demos,
                            environment_factory=_factory(config, problem, entry, pool, seed=91701000, prefix="train_seed0"))
        self.assertEqual(agent.batches, [16] * 13)
        self.assertEqual(result.episode_records[-1]["actual_demo_samples_total"], 13 * 16)
        self.assertEqual(result.episode_records[-1]["actual_online_samples_total"], 13 * 48)

    def test_capacity_adaptive_actual_batch_audit_has_late_differences(self):
        root = ROOT / "outputs/whole_map_91701_3to5_capacity_controlled_20261001_142502/adaptive"
        for seed, differing in ((0, 20000), (1, 30000)):
            segments = _verified_batch_segments(root / f"seed_{seed}", adaptive=True)
            self.assertEqual(sum(row["gradient_updates"] for row in segments), 199501)
            self.assertEqual(sum(row["gradient_updates"] for row in segments if row["demo_count"] != 16), differing)

    def test_capacity_controlled_changes_only_capacity_and_registered_seeds(self):
        config = load_config(ROOT / "configs/whole_map_route_pool_91701_3to5_capacity_controlled_v1.yaml")
        baseline = load_config(ROOT / "configs/whole_map_route_pool_91701_four_methods_3to5_v1.yaml")
        for section in ("dataset", "environment", "dynamic_route_pool", "reward", "agent", "demonstration", "adaptive"):
            self.assertEqual(config[section], baseline[section])
        self.assertEqual({k:v for k,v in config["training"].items() if k not in {"seeds", "replay_capacity"}},
                         {k:v for k,v in baseline["training"].items() if k not in {"seeds", "replay_capacity"}})
        problem, entry, pool = _load_inputs(config)
        demos = collect_astar_demonstrations([problem], 20, 7400, window_size=15, spatial_channels=4)
        for method in ("time_decay", "adaptive"):
            current = copy.deepcopy(config)
            current["experiment"]["method"] = method
            _validate_protocol(current, problem, entry, pool)
            replay = build_replay(_training_config(current, 0), demos)
            self.assertEqual(replay.demonstration_size, 1320)
            self.assertEqual(replay.online.capacity, 10000)
            self.assertTrue(_training_config(current, 0).replay_diagnostics)
        factory = _factory(config, problem, entry, pool, seed=91701999, prefix="validation")
        original = json.loads((ROOT / baseline["experiment"]["output_root"]).parent.joinpath(
            "whole_map_91701_3to5_four_methods_20261001_002044/fixed_validation_scenarios.json").read_text())
        for scene in original["scenarios"]:
            env = factory(problem, window_size=15)
            self.assertEqual(list(env.dynamic_route_ids), scene["route_ids"])
            self.assertEqual(env.scenario_id, scene["scenario_id"])
            self.assertEqual([spec.start_index for spec in env.dynamic_obstacles], [spec["start_index"] for spec in scene["obstacles"]])
            self.assertEqual([spec.direction for spec in env.dynamic_obstacles], [spec["direction"] for spec in scene["obstacles"]])

    def test_aulc_common_nominal_checkpoints_and_timeout_definition(self):
        curve = [{"environment_steps_total":10012, "safe_success_rate":0.0},
                 {"environment_steps_total":20230, "safe_success_rate":1.0},
                 {"environment_steps_total":30000, "safe_success_rate":1.0}]
        self.assertEqual(_safe_success_aulc(curve), 0.75)
        self.assertIsNone(_safe_success_aulc(curve[:1]))
        summary = _validation_diagnostics({"success_rate":0.6, "collision_rate":0.1})
        self.assertAlmostEqual(summary["timeout_rate"], 0.3)
        self.assertAlmostEqual(summary["not_reached_rate"], 0.4)

    def test_variable_counts_compositions_fairness_and_fixed_validation(self):
        config = load_config(ROOT / "configs/whole_map_route_pool_91701_four_methods_3to5_v1.yaml")
        problem, entry, pool = _load_inputs(config)
        observed = set()
        for seed in range(5):
            left = _factory(config, problem, entry, pool, seed=91701000 + seed, prefix="train")
            right = _factory(config, problem, entry, pool, seed=91701000 + seed, prefix="train")
            count_rng = random.Random(91701000 + seed)
            for _ in range(100):
                expected = count_rng.choice((3, 4, 5))
                hi, ar = (1, 1) if expected == 3 else (2, 2) if expected == 5 else count_rng.choice(((2, 1), (1, 2)))
                a = left(problem, max_steps=3, window_size=15)
                b = right(problem, max_steps=3, window_size=15)
                self.assertEqual(len(a.dynamic_obstacles), expected)
                self.assertEqual(Counter(a.dynamic_route_categories), Counter(high_interaction=hi, alternative_branch=ar, background=1))
                self.assertEqual(a.dynamic_obstacles, b.dynamic_obstacles)
                self.assertEqual(a.dynamic_route_ids, b.dynamic_route_ids)
                observed.add((hi, ar, 1))
            left.reset_schedule()
            right.reset_schedule()
            self.assertEqual(left(problem, window_size=15).dynamic_obstacles, right(problem, window_size=15).dynamic_obstacles)
        self.assertEqual(observed, {(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)})
        validation = _factory(config, problem, entry, pool, seed=config["dynamic_route_pool"]["validation_sampling_seed"], prefix="validation")
        scenes = [validation(problem, window_size=15).dynamic_obstacles for _ in range(50)]
        validation.reset_schedule()
        self.assertEqual(scenes, [validation(problem, window_size=15).dynamic_obstacles for _ in range(50)])

    def test_four_method_protocol_and_unique_directories(self):
        config = load_config(ROOT / "configs/whole_map_route_pool_91701_four_methods_3to5_v1.yaml")
        directories = set()
        for method, replay in (("pure", "uniform"), ("prefill", "prefill"), ("time_decay", "persistent_demo"), ("adaptive", "persistent_demo")):
            current = copy.deepcopy(config)
            current["experiment"]["method"] = method
            current["training"]["replay_strategy"] = replay
            if method == "pure":
                current.pop("demonstration")
            _validate_protocol(current, *_load_inputs(current))
            for seed in range(5):
                directories.add((method, seed))
                self.assertEqual(_training_config(current, seed).replay_strategy, replay)
        self.assertEqual(len(directories), 20)

    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(
            (ROOT / "configs/whole_map_route_pool_91701_pilot_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        source = json.loads(
            (ROOT / cls.config["dataset"]["source_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        cls.pool = json.loads(
            (ROOT / cls.config["dataset"]["route_pool_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        cls.problem = problem_from_record(source["maps"][0]["problem"])
        cls.map_entry = cls.pool["maps"][0]

    def _factory(self, seed=0):
        acceptance = dict(self.pool["design"]["scene_acceptance"])
        acceptance["reasonable_path_count"] = 240
        return WholeMapRoutePoolEnvironmentFactory(
            self.problem,
            self.map_entry,
            composition=self.config["dynamic_route_pool"]["composition"],
            seed=seed,
            acceptance=acceptance,
        )

    def test_pilot_config_is_isolated_and_step_budgeted(self):
        self.assertEqual(self.config["dataset"]["map_id"], "irregular_workcell_91701")
        self.assertEqual(self.config["training"]["seeds"], [0, 1])
        self.assertEqual(self.config["training"]["max_environment_steps"], 200000)
        self.assertEqual(self.config["training"]["replay_strategy"], "uniform")
        self.assertEqual(
            self.config["experiment"]["output_root"],
            "outputs/whole_map_route_pool_91701_pilot_v1",
        )

    def test_factory_samples_seven_once_per_episode(self):
        factory = self._factory(seed=91701000)
        first = factory(self.problem, max_steps=10, window_size=15)
        self.assertEqual(
            Counter(first.dynamic_route_categories),
            Counter(high_interaction=3, alternative_branch=3, background=1),
        )
        route_ids = first.dynamic_route_ids
        specs = first.dynamic_obstacles
        first.reset()
        for _ in range(3):
            first.step(4)
        self.assertEqual(first.dynamic_route_ids, route_ids)
        self.assertEqual(first.dynamic_obstacles, specs)
        second = factory(self.problem, max_steps=10, window_size=15)
        self.assertNotEqual(first.scenario_id, second.scenario_id)

    def test_dynamic_metrics_reach_evaluation_summary(self):
        problem = NavigationProblem(
            map_id="metric_fixture",
            seed=0,
            size=5,
            start=(2, 0),
            goal=(2, 4),
            obstacles=frozenset(),
            nominal_path=((2, 0), (2, 1), (2, 2), (2, 3), (2, 4)),
        )
        spec = DynamicObstacleSpec(route=((2, 1), (1, 1)))

        def factory(current_problem, **kwargs):
            return DynamicGridNavigationEnv(
                current_problem,
                dynamic_obstacles=(spec,),
                **kwargs,
            )

        summary, rows, _ = evaluate_agent(
            _StayAgent(),
            [problem],
            max_steps=1,
            window_size=3,
            environment_factory=factory,
        )
        self.assertEqual(rows[0]["encounter_rate"], 1.0)
        self.assertEqual(rows[0]["conflict_opportunity_rate"], 1.0)
        for key in (
            "encounter_rate",
            "conflict_opportunity_rate",
            "dynamic_collision_rate",
            "safe_success_rate",
        ):
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
