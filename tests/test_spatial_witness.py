from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.evaluation.spatial_witness import motion_summary, shortest_simple_visible_plan, static_bypass_audit
from astar_d3qn.evaluation.conflict import obstacle_positions
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.core.grid import chebyshev


class SpatialWitnessTests(unittest.TestCase):
    def problem(self):
        return NavigationProblem(map_id="test", seed=0, size=5, start=(2, 0), goal=(2, 4),
                                 obstacles=frozenset(),
                                 nominal_path=((2, 0), (2, 1), (2, 2), (2, 3), (2, 4)))

    def scene(self):
        return DynamicScenario(seed=0, obstacles=(DynamicObstacleSpec(route=((1, 2), (2, 2))),))

    def test_static_bypass_does_not_certify_dynamic_feasibility(self):
        report = static_bypass_audit(self.problem(), (2, 2))
        self.assertEqual(report["status"], "static_bypass_possible")
        self.assertEqual((report["base_steps"], report["bypass_steps"]), (4, 6))

    def test_gate_pair_constraint_can_make_an_ordinary_cell_required(self):
        problem = replace(self.problem(), obstacles=frozenset({(1, 2), (3, 2)}))
        self.assertEqual(static_bypass_audit(problem, (2, 2))["status"], "static_bypass_possible")
        report = static_bypass_audit(problem, (2, 2), additionally_blocked={(0, 2), (4, 2)})
        self.assertEqual(report["status"], "conflict_cell_is_required")
        self.assertEqual(report["base_steps"], 4)
        self.assertIsNone(report["bypass_steps"])

    def test_preexisting_disconnection_is_not_attributed_to_conflict_cell(self):
        problem = replace(self.problem(), obstacles=frozenset((r, 2) for r in range(5)))
        self.assertEqual(static_bypass_audit(problem, (2, 1))["status"], "base_disconnected")
        self.assertEqual(static_bypass_audit(self.problem(), self.problem().goal)["status"], "endpoint_excluded")

    def test_wait_reversal_and_longer_closed_walk_are_distinct_counts(self):
        wait = motion_summary(((0, 0), (0, 0), (0, 1)))
        self.assertEqual(wait["stay_count"], 1)
        self.assertEqual(wait["closed_walk_count"], 0)
        reversal = motion_summary(((0, 0), (0, 1), (0, 0)))
        self.assertEqual(reversal["immediate_reversal_count"], 1)
        self.assertEqual(reversal["closed_walk_count"], 1)
        loop = motion_summary(((0, 0), (0, 1), (1, 1), (1, 0), (0, 0)))
        self.assertEqual(loop["immediate_reversal_count"], 0)
        self.assertEqual(loop["closed_walk_count"], 1)
        self.assertFalse(loop["is_simple_path"])

    def test_visible_simple_bypass_is_safe_and_has_no_repeated_cells(self):
        problem, scene = self.problem(), self.scene()
        result = shortest_simple_visible_plan(problem, scene, primary_index=0,
                                             observation_radius=1, additionally_blocked={(2, 2)})
        self.assertEqual(result.status, "found")
        path = result.plan.positions
        self.assertEqual(result.plan.steps, 6)
        self.assertTrue(motion_summary(path)["is_simple_path"])
        self.assertNotIn((2, 2), path)
        step = next(t for t in range(1, len(problem.nominal_path)) if path[t] != problem.nominal_path[t])
        cells = obstacle_positions(scene.obstacles[0], result.plan.steps)
        self.assertLessEqual(chebyshev(path[step - 1], cells[step - 1]), 1)
        for t in range(1, len(path)):
            self.assertNotIn(path[t], (cells[t - 1], cells[t]))

    def test_backtracking_only_solution_is_not_a_simple_spatial_witness(self):
        problem = replace(self.problem(), obstacles=frozenset(((1, 1), (3, 1))))
        result = shortest_simple_visible_plan(problem, self.scene(), primary_index=0,
                                             observation_radius=1, additionally_blocked={(2, 2)})
        self.assertEqual(result.status, "infeasible_under_constraints")
        self.assertEqual(result.stop_reason, "exhaustive_search")
        self.assertIsNone(result.plan)

    def test_resource_exhaustion_is_unknown_not_infeasible(self):
        for limits in ({"max_expanded": 1}, {"max_generated": 1}, {"max_seconds": 1e-12}):
            with self.subTest(limits=limits):
                result = shortest_simple_visible_plan(self.problem(), self.scene(), primary_index=0,
                                                     observation_radius=1, additionally_blocked={(2, 2)}, **limits)
                self.assertEqual(result.status, "budget_exhausted")
                self.assertIsNone(result.plan)

    def test_visibility_is_not_silently_relaxed(self):
        result = shortest_simple_visible_plan(self.problem(), self.scene(), primary_index=0,
                                             observation_radius=0, additionally_blocked={(2, 2)})
        self.assertEqual(result.status, "infeasible_under_constraints")


if __name__ == "__main__":
    unittest.main()
