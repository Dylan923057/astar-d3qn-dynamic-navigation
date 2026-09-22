import sys
import unittest
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pilot_office_v10_route_expansion as pilot


class RouteExpansionTests(unittest.TestCase):
    def test_approach_geometry_is_bounded_and_separates_both_sides(self):
        old = [[7, 8], [7, 9], [7, 10]]
        pools = {"train": {"corridor": [{"route": old}]},
                 "validation": {"corridor": []}, "test": {"corridor": []}}
        problem = SimpleNamespace(size=40, start=(2, 2), goal=(37, 37), obstacles=set())
        rows, audit = pilot.approach_geometry_candidates(problem, pools)
        self.assertLessEqual(len(rows), 72)
        self.assertEqual(len(audit), 4)
        self.assertEqual(len({pilot.route_key(row["route"]) for row in rows}), len(rows))
        self.assertEqual(len({row["route_id"] for row in rows}), len(rows))
        for zone, anchors in pilot.APPROACH_CENTERS.items():
            selected = [row for row in rows if row["critical_zone"] == zone]
            self.assertEqual({row["approach_side"] for row in selected}, {0, 1})
            for row in selected:
                self.assertNotEqual(pilot.route_key(row["route"]), pilot.route_key(old))
                self.assertTrue(pilot._valid_route(tuple(map(tuple, row["route"])), problem, min_length=len(row["route"])))
                self.assertLessEqual(max(abs(a - b) for a, b in zip(row["center"], anchors[row["approach_side"]])), 1)

    def test_static_filter_is_phase_specific_and_keeps_input_unchanged(self):
        v10 = pilot.v10
        problem = SimpleNamespace(nominal_path=((2, 0), (2, 1), (2, 2), (2, 3), (2, 4)))
        spec1 = v10.DynamicObstacleSpec(route=((2, 1), (2, 2)), start_index=0, label="same_route")
        spec2 = v10.DynamicObstacleSpec(route=((2, 1), (2, 2)), start_index=1, label="same_route")
        original = {0: [(spec1, 1), (spec2, 2)], 1: []}
        def trajectory(spec, steps):
            return ((-9, -9), (-9, -9), (2, 2), (-9, -9), (-9, -9)) if spec.start_index == 0 else (
                (-9, -9), (2, 1), (-9, -9), (-9, -9), (-9, -9))
        def geometry(problem, cell, **kwargs):
            return {"status": "conflict_cell_is_required" if cell == (2, 2) else "static_bypass_possible"}
        with patch.object(v10.v9, "obstacle_positions", side_effect=trajectory), patch.object(
            v10, "static_bypass_audit", side_effect=geometry
        ):
            filtered, audit = v10._filter_spatial_primary_options(problem, original)
        self.assertEqual(filtered, {0: [(spec2, 2)], 1: []})
        self.assertEqual(len(original[0]), 2)
        self.assertEqual(audit["phase_counts"], {"conflict_cell_is_required": 1, "static_bypass_possible": 1})

    def test_target_routes_are_not_silently_replaced_with_old_routes(self):
        v10 = pilot.v10
        spec = v10.DynamicObstacleSpec(route=((0, 1), (1, 1)), label="old_route")
        with patch.object(v10.v9, "obstacle_positions") as motion:
            filtered, audit = v10._filter_spatial_primary_options(None, {0: [(spec, 1)]}, allowed_route_ids=[])
            motion.assert_not_called()
        self.assertEqual(filtered, {0: []})
        self.assertEqual(audit["phase_counts"], {"outside_target_approach_routes": 1})

    def test_geometry_is_valid_bounded_and_does_not_copy_split_routes(self):
        old = [[9, 8], [9, 9], [9, 10]]
        pools = {"train": {"corridor": [{"route": old}]},
                 "validation": {"corridor": []}, "test": {"corridor": []}}
        problem = SimpleNamespace(size=40, start=(2, 2), goal=(37, 37), obstacles={(25, 17)})
        rows, audit = pilot.geometry_candidates(problem, pools)
        self.assertTrue(rows)
        for row in rows:
            route = tuple(map(tuple, row["route"]))
            self.assertTrue(pilot._valid_route(route, problem, min_length=len(route)))
            self.assertNotEqual(pilot.route_key(route), pilot.route_key(old))
            self.assertNotEqual(pilot.route_key(tuple(reversed(route))), pilot.route_key(old))
        self.assertTrue(all(info["preflight_geometry_count"] <= 36 for info in audit.values()))
        self.assertEqual(pools["train"]["corridor"][0]["route"], old)

    def test_spatial_budget_unknown_is_not_an_accepted_witness(self):
        v10 = pilot.v10
        problem = SimpleNamespace(nominal_path=((0, 0), (0, 1), (0, 2)))
        scene = v10.DynamicScenario(seed=0, obstacles=(v10.DynamicObstacleSpec(route=((0, 1), (1, 1))),))
        audit = {}
        with patch.object(v10, "shortest_simple_visible_plan", return_value=SimpleNamespace(
            plan=None, status="budget_exhausted", stop_reason="generated_states", expanded=3,
            generated=8, elapsed_seconds=.1,
        )):
            evidence, status = v10._spatial_avoidance_evidence(
                problem, scene, {}, 0, {"observation_radius": 7}, search_audit=audit)
        self.assertIsNone(evidence)
        self.assertEqual(status, "budget_exhausted")
        self.assertEqual(audit["stop_reason"], "generated_states")

    def test_target_zone_without_primary_options_does_not_build_cartesian_product(self):
        v10 = pilot.v10
        records = [{"critical_zone": zone} for zone in v10.ZONE_COLORS]
        spec = v10.DynamicObstacleSpec(route=((0, 1), (1, 1)))
        safe = {i: [(spec, 0)] for i in range(5)}
        primary = {i: ([(spec, 1)] if i == 0 else []) for i in range(5)}
        design = {"move_every_choices": [1], "_constructive_phase_pairs": True,
                  "_factorized_proposals": True, "_proposal_primary_zone": "goal_approach"}
        with patch.object(v10, "_route_options_for_speeds", return_value=[(spec, 1)]), patch.object(
            v10, "_preflight_phase_pairs", return_value=(safe, primary)
        ), patch.object(v10.v9, "_candidate_combinations") as combinations:
            with self.assertRaisesRegex(RuntimeError, "No phase-pair proposal"):
                v10._build_pool(None, "test", "avoidance", 1, records, [], design,
                                random.Random(0), set(), require_pair=True)
            combinations.assert_not_called()

    def test_impossible_spatial_proposals_stop_before_behavior_search(self):
        v10 = pilot.v10
        records = [{"critical_zone": zone} for zone in v10.ZONE_COLORS]
        spec = v10.DynamicObstacleSpec(route=((0, 1), (1, 1)))
        safe = {i: [(spec, 0)] for i in range(5)}
        primary = {i: [(spec, 1)] for i in range(5)}
        design = {"move_every_choices": [1], "_constructive_phase_pairs": True,
                  "_factorized_proposals": True, "_spatial_avoidance_pilot": True}
        with patch.object(v10, "_route_options_for_speeds", return_value=[(spec, 1)]), patch.object(
            v10, "_preflight_phase_pairs", return_value=(safe, primary)
        ), patch.object(v10, "_filter_spatial_primary_options", return_value=(
            {i: [] for i in range(5)}, {"phase_counts": {"conflict_cell_is_required": 5}}
        )), patch.object(v10.v9, "_classify_candidate") as classify:
            with self.assertRaisesRegex(RuntimeError, "No phase-pair proposal"):
                v10._build_pool(None, "test", "avoidance", 1, records, [], design,
                                random.Random(0), set(), require_pair=True)
            classify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
