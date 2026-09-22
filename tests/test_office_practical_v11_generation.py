"""Registration and bookkeeping checks for the practical Office v11 dataset."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_office_practical_scenarios_v11 as generator
from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.evaluation.behavior_oracle import SafePlan
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.utils.config import load_config


class OfficePracticalV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            ROOT / "configs/dynamic_spatial_generalization_office_practical_v11.yaml"
        )

    def test_registered_counts_and_training_contract_are_consistent(self):
        generator.validate_config_contract(self.config)
        spatial = self.config["spatial_generalization"]
        design = spatial["scenario_design"]
        self.assertEqual(sum(design["practical_counts"]["train"].values()), 100)
        self.assertEqual(sum(design["practical_counts"]["validation"].values()), 20)
        self.assertEqual(
            sum(design["practical_counts"]["test"].values())
            + 2 * sum(design["diagnostic_pair_counts"].values()),
            50,
        )
        self.assertEqual(spatial["candidate_route_lengths"], [3, 5, 7])
        self.assertEqual(self.config["reward"], {
            "step": -0.01, "progress": 0.05, "stay": 0.0,
            "collision": -1.0, "goal": 10.0,
        })
        self.assertTrue(self.config["environment"]["terminate_on_collision"])
        self.assertTrue(self.config["environment"]["mask_static_invalid_actions"])

    def test_curriculum_activates_primary_first_and_restores_all_obstacles(self):
        stages = self.config["training"]["dynamic_obstacle_curriculum"]["stages"]
        self.assertEqual([row["start_environment_step"] for row in stages], [0, 60000, 140000, 240000])
        self.assertEqual([row["obstacle_indices"] for row in stages], [[], [0], [0, 1, 2], [0, 1, 2, 3, 4]])

    def test_route_assignment_keeps_split_geometries_unique_but_allows_zone_alternatives(self):
        candidates = []
        serial = 0
        for zone in generator.ZONE_ORDER:
            anchors = sorted({
                anchor
                for split in generator.SPLITS
                for anchor in generator._route_requirements(split, zone)
            })
            for anchor in anchors:
                # Four distinct middle variants are needed across the three
                # splits because test intentionally receives two of them.
                for variant in range(5):
                    serial += 1
                    base = serial * 20
                    route = [[base, column] for column in range(5)]
                    if zone == "upper_left_gate" and anchor == "approach_before":
                        route[0] = [999, 999]
                    role = {
                        "wait_clearance": "wait_gate",
                        "bottleneck": "bottleneck",
                    }.get(anchor, "open_approach")
                    candidates.append({
                        "critical_zone": zone,
                        "anchor_name": anchor,
                        "functional_role": role,
                        "route": route,
                        "center": route[len(route) // 2],
                        "orientation": "horizontal",
                        "corridor_score": 1,
                        "anchor_distance": 1,
                        "eligible_reference_indices": [0],
                        "topology_path_coverage_count": 1,
                    })
        trap_route = [[999, 999], [999, 1000], [999, 1001], [999, 1002], [999, 1003]]
        candidates.append({
            "critical_zone": "upper_left_gate", "anchor_name": "bottleneck",
            "functional_role": "bottleneck", "route": trap_route,
            "center": trap_route[2], "orientation": "horizontal", "corridor_score": 1,
            "anchor_distance": 0, "eligible_reference_indices": [0, 1],
            "topology_path_coverage_count": 2,
        })
        pools = generator._select_route_pools(candidates, 7, 100000)
        geometries = []
        for split in generator.SPLITS:
            for row in pools[split]["corridor"]:
                geometries.append(generator.route_key(row["route"]))
            self.assertEqual(
                len(pools[split]["corridor"]),
                self.config["spatial_generalization"]["route_pool_counts"][split]["corridor"],
            )
        self.assertEqual(len(geometries), len(set(geometries)))
        upper_left = [
            row for split in generator.SPLITS
            for row in pools[split]["corridor"]
            if row["critical_zone"] == "upper_left_gate"
        ]
        self.assertTrue(any(
            set(map(tuple, left["route"])).intersection(map(tuple, right["route"]))
            for index, left in enumerate(upper_left)
            for right in upper_left[index + 1:]
        ))

    def test_safe_context_balancing_filters_infeasible_routes_before_usage(self):
        blocked_spec = DynamicObstacleSpec(
            route=((0, 0), (0, 1)), label="unused_but_blocked",
        )
        safe_spec = DynamicObstacleSpec(
            route=((1, 0), (1, 1)), label="used_but_safe",
        )
        routes = [
            {"route_id": "unused_but_blocked"},
            {"route_id": "used_but_safe"},
        ]
        options = {
            "unused_but_blocked": [(blocked_spec, 0b1)],
            "used_but_safe": [(safe_spec, 0b0)],
        }
        result = generator._balanced_reference_safe_option(
            routes,
            options,
            reference_index=0,
            route_use=Counter({"unused_but_blocked": 0, "used_but_safe": 9}),
            rng=generator.random.Random(3),
        )
        self.assertEqual(result[0].label, "used_but_safe")

    def test_safe_context_balancing_also_protects_behavior_witness(self):
        colliding = DynamicObstacleSpec(
            route=((0, 1), (1, 1)), start_index=0, direction=1,
            move_every=2, label="reference_safe_but_witness_blocked",
        )
        harmless = DynamicObstacleSpec(
            route=((2, 0), (2, 1)), start_index=0, direction=1,
            move_every=2, label="reference_and_witness_safe",
        )
        routes = [
            {"route_id": "reference_safe_but_witness_blocked"},
            {"route_id": "reference_and_witness_safe"},
        ]
        options = {
            "reference_safe_but_witness_blocked": [(colliding, 0)],
            "reference_and_witness_safe": [(harmless, 0)],
        }
        result = generator._balanced_reference_safe_option(
            routes, options, reference_index=0, route_use=Counter(),
            rng=generator.random.Random(3),
            required_safe_path=((0, 0), (0, 1), (0, 2)),
        )
        self.assertEqual(result[0].label, "reference_and_witness_safe")

    def test_wait_threshold_is_read_from_config_and_rejection_is_named(self):
        problem = NavigationProblem(
            map_id="unit", seed=0, size=3, start=(0, 0), goal=(0, 2),
            obstacles=frozenset(), nominal_path=((0, 0), (0, 1), (0, 2)),
        )
        scenario = DynamicScenario(seed=0, obstacles=(DynamicObstacleSpec(
            route=((1, 0), (1, 1)), label="primary",
        ),))
        reference = problem.nominal_path
        wait_plan = SafePlan(reference, 5, 1)
        no_wait_plan = SafePlan(reference, 6, 0)
        pair = ("upper_left", "lower_left")
        pair_wait = {name: None for name in generator.GATE_PAIRS}
        pair_no_wait = {name: None for name in generator.GATE_PAIRS}
        pair_wait[pair] = wait_plan
        pair_no_wait[pair] = no_wait_plan
        design = dict(self.config["spatial_generalization"]["scenario_design"])
        design["minimum_wait_advantage_steps"] = 2
        diagnostics = Counter()
        with patch.object(generator, "_gate_pair", return_value=pair), patch.object(
            generator.v9, "_nominal_path_collides", return_value=True,
        ), patch.object(
            generator, "shortest_safe_plan", side_effect=(wait_plan, wait_plan, no_wait_plan),
        ), patch.object(
            generator, "_gate_plans", return_value=pair_no_wait,
        ):
            result = generator.classify_practical(
                problem, scenario, "wait", reference, 1, design, diagnostics,
            )
        self.assertIsNone(result)
        self.assertEqual(diagnostics, {"wait_advantage_below_threshold": 1})

    def test_wait_accepts_when_same_gate_is_impossible_without_stay(self):
        problem = NavigationProblem(
            map_id="unit", seed=0, size=3, start=(0, 0), goal=(0, 2),
            obstacles=frozenset(), nominal_path=((0, 0), (0, 1), (0, 2)),
        )
        scenario = DynamicScenario(seed=0, obstacles=(DynamicObstacleSpec(
            route=((1, 0), (1, 1)), label="primary",
        ),))
        wait_plan = SafePlan(problem.nominal_path, 5, 1)
        global_no_wait = SafePlan(problem.nominal_path, 7, 0)
        pair = ("upper_left", "lower_left")
        pair_wait = {name: None for name in generator.GATE_PAIRS}
        pair_no_wait = {name: None for name in generator.GATE_PAIRS}
        pair_wait[pair] = wait_plan
        observation = {
            "decision_step": 2,
            "decision_position": [0, 1],
            "nearest_dynamic_distance_at_decision": 1,
            "observable_decisive_obstacle_index": 0,
        }
        diagnostics = Counter()
        with patch.object(generator, "_gate_pair", return_value=pair), patch.object(
            generator.v9, "_nominal_path_collides", return_value=True,
        ), patch.object(
            generator, "shortest_safe_plan",
            side_effect=(wait_plan, wait_plan, None, global_no_wait),
        ), patch.object(
            generator, "_gate_plans", return_value=pair_no_wait,
        ), patch.object(
            generator.v9, "_decision_observability", return_value=observation,
        ):
            result = generator.classify_practical(
                problem, scenario, "wait", problem.nominal_path, 1,
                self.config["spatial_generalization"]["scenario_design"], diagnostics,
            )
        self.assertIsNotNone(result)
        self.assertFalse(result["reference_gate_no_wait_reachable"])
        self.assertIsNone(result["reference_gate_no_wait_steps"])
        self.assertIsNone(result["wait_advantage_steps"])
        self.assertEqual(diagnostics, {})

    def test_record_keeps_the_exact_designated_reference(self):
        reference = ((0, 0), (0, 1), (0, 2))
        problem = NavigationProblem(
            map_id="unit", seed=0, size=3, start=(0, 0), goal=(0, 2),
            obstacles=frozenset(), nominal_path=reference,
        )
        spec = DynamicObstacleSpec(route=((1, 0), (1, 1)), label="route")
        scenario = DynamicScenario(seed=0, obstacles=(spec,))
        plan = SafePlan(reference, 2, 0)
        classification = {
            "best_plan": plan, "witness": plan,
            "observation": {"decision_step": None, "decision_position": None,
                            "nearest_dynamic_distance_at_decision": None},
            "reference_gate_pair": ("upper_left", "lower_left"),
            "reference_gate_cost": 2, "best_alternative_gate_cost": 3,
            "reference_gate_no_wait_steps": 2,
            "reference_gate_no_wait_reachable": True,
            "no_wait_steps": 2, "wait_advantage_steps": 0,
            "global_no_wait_penalty_steps": 0,
            "first_primary_conflict_cell": None,
        }
        routes = {"route": {"functional_role": "open_approach", "critical_zone": "goal_approach"}}
        record = generator._record(
            problem, scenario, (0b0011,), classification, "normal", 0,
            routes, demo_count=4, observation_radius=1,
        )
        self.assertEqual(record["reference_path"], [[0, 0], [0, 1], [0, 2]])
        self.assertEqual(record["reference_demo_collision_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
