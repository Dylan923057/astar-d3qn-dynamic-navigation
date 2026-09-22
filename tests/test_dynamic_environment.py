from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action, action_between
from astar_d3qn.envs.dynamic_grid import (
    DynamicGridNavigationEnv,
    DynamicObstacleSpec,
    build_map01_controlled_bottleneck_obstacle_specs,
    build_map01_controlled_six_obstacle_specs,
    build_map01_controlled_mixed_six_obstacle_specs,
    build_map01_three_crossing_obstacle_specs,
    build_map01_three_obstacle_specs,
    build_crossing_obstacle_spec,
)
from astar_d3qn.envs.dynamic_scenarios import (
    CurriculumScheduledDynamicEnvironmentFactory,
    DynamicObstacleCurriculumStage,
    DynamicScenario,
    ScheduledDynamicEnvironmentFactory,
    build_random_crossing_scenario,
)
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.training.demo_collector import collect_astar_demonstrations


def open_problem(
    *,
    start: tuple[int, int] = (3, 0),
    goal: tuple[int, int] = (3, 6),
    obstacles: frozenset[tuple[int, int]] = frozenset(),
) -> NavigationProblem:
    path = astar_path(start, goal, obstacles, 7)
    assert path is not None
    return NavigationProblem(
        map_id="dynamic_test",
        seed=1,
        size=7,
        start=start,
        goal=goal,
        obstacles=obstacles,
        nominal_path=tuple(path),
    )


class DynamicEnvironmentTests(unittest.TestCase):
    def test_random_scenario_is_seeded_and_strategy_independent(self) -> None:
        problem = open_problem()
        first = build_random_crossing_scenario(problem, 12, obstacle_count=1)
        second = build_random_crossing_scenario(problem, 12, obstacle_count=1)

        self.assertEqual(first, second)
        self.assertEqual(first.obstacles[0].reference_path_source, "nominal_static_path")

    def test_scheduled_factory_cycles_train_and_eval_separately(self) -> None:
        problem = open_problem()
        train = build_random_crossing_scenario(problem, 1, obstacle_count=1)
        evaluation = build_random_crossing_scenario(problem, 2, obstacle_count=1)
        factory = ScheduledDynamicEnvironmentFactory((train,), (evaluation,))

        first = factory(problem, window_size=7)
        second = factory(problem, window_size=7)
        factory.set_mode("eval")
        held_out = factory(problem, window_size=7)

        self.assertEqual(first.scenario_id, 1)
        self.assertEqual(second.scenario_id, 1)
        self.assertEqual(held_out.scenario_id, 2)

    def test_training_schedule_shuffle_is_seeded_and_resettable(self) -> None:
        problem = open_problem()
        train = tuple(
            build_random_crossing_scenario(problem, seed, obstacle_count=1)
            for seed in (1, 2, 3, 4)
        )
        evaluation = build_random_crossing_scenario(
            problem, 20, obstacle_count=1
        )
        factory = ScheduledDynamicEnvironmentFactory(
            train,
            (evaluation,),
            shuffle_train=True,
            seed=8,
        )

        first_order = [
            factory(problem, window_size=7).scenario_id for _ in train
        ]
        factory.reset_schedule()
        reset_order = [
            factory(problem, window_size=7).scenario_id for _ in train
        ]

        self.assertEqual(first_order, reset_order)
        self.assertEqual(set(first_order), {1, 2, 3, 4})
        self.assertNotEqual(first_order, [1, 2, 3, 4])

    def test_static_action_mask_does_not_hide_dynamic_obstacle(self) -> None:
        problem = open_problem(obstacles=frozenset({(2, 0)}))
        spec = DynamicObstacleSpec(
            route=((2, 1), (3, 1), (4, 1)),
            start_index=1,
            direction=1,
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()

        mask = env.action_mask(mask_collisions=True)

        self.assertFalse(mask[int(Action.UP)])
        self.assertTrue(mask[int(Action.RIGHT)])

    def test_curriculum_factory_adds_obstacles_but_keeps_evaluation_full(
        self,
    ) -> None:
        problem = open_problem()
        train = build_random_crossing_scenario(problem, 1, obstacle_count=3)
        evaluation = build_random_crossing_scenario(problem, 2, obstacle_count=3)
        stages = (
            DynamicObstacleCurriculumStage("easy_1", 0, 1),
            DynamicObstacleCurriculumStage("medium_2", 10, 2),
            DynamicObstacleCurriculumStage("full_3", 20, 3),
        )
        factory = CurriculumScheduledDynamicEnvironmentFactory(
            (train,),
            (evaluation,),
            stages,
        )

        easy = factory(problem, window_size=7)
        factory.set_environment_steps(10)
        medium = factory(problem, window_size=7)
        factory.set_environment_steps(20)
        full = factory(problem, window_size=7)
        factory.set_mode("eval")
        held_out = factory(problem, window_size=7)

        self.assertEqual(len(easy.dynamic_obstacles), 1)
        self.assertEqual(easy.curriculum_stage, "easy_1")
        self.assertEqual(len(medium.dynamic_obstacles), 2)
        self.assertEqual(medium.curriculum_stage, "medium_2")
        self.assertEqual(len(full.dynamic_obstacles), 3)
        self.assertEqual(full.curriculum_stage, "full_3")
        self.assertEqual(len(held_out.dynamic_obstacles), 3)
        self.assertEqual(held_out.curriculum_stage, "full_evaluation")

    def test_curriculum_can_start_static_and_select_difficulty_order(self) -> None:
        problem = open_problem()
        train = build_random_crossing_scenario(problem, 1, obstacle_count=3)
        evaluation = build_random_crossing_scenario(problem, 2, obstacle_count=3)
        stages = (
            DynamicObstacleCurriculumStage("static_0", 0, 0, ()),
            DynamicObstacleCurriculumStage("selected_1", 10, 1, (2,)),
            DynamicObstacleCurriculumStage("full_3", 20, 3, (0, 1, 2)),
        )
        factory = CurriculumScheduledDynamicEnvironmentFactory(
            (train,),
            (evaluation,),
            stages,
        )

        static = factory(problem, window_size=7)
        factory.set_environment_steps(10)
        selected = factory(problem, window_size=7)
        factory.set_environment_steps(20)
        full = factory(problem, window_size=7)
        factory.set_mode("eval")
        held_out = factory(problem, window_size=7)

        self.assertEqual(static.dynamic_obstacles, ())
        self.assertEqual(selected.dynamic_obstacles, (train.obstacles[2],))
        self.assertEqual(full.dynamic_obstacles, train.obstacles)
        self.assertEqual(held_out.dynamic_obstacles, evaluation.obstacles)

    def test_curriculum_filters_training_difficulty_only(self) -> None:
        problem = open_problem()
        source = build_random_crossing_scenario(problem, 1, obstacle_count=1)
        easy = DynamicScenario(
            seed=11,
            obstacles=source.obstacles,
            difficulty_stratum="easy",
        )
        hard = DynamicScenario(
            seed=12,
            obstacles=source.obstacles,
            difficulty_stratum="hard",
        )
        evaluation = DynamicScenario(
            seed=20,
            obstacles=source.obstacles,
            difficulty_stratum="hard",
        )
        stages = (
            DynamicObstacleCurriculumStage(
                "easy_only",
                0,
                1,
                (0,),
                ("easy",),
            ),
            DynamicObstacleCurriculumStage(
                "full",
                10,
                1,
                (0,),
                ("easy", "hard"),
            ),
        )
        factory = CurriculumScheduledDynamicEnvironmentFactory(
            (easy, hard),
            (evaluation,),
            stages,
        )

        first = factory(problem, window_size=7)
        second = factory(problem, window_size=7)
        factory.set_mode("eval")
        held_out = factory(problem, window_size=7)

        self.assertEqual(first.scenario_id, easy.seed)
        self.assertEqual(second.scenario_id, easy.seed)
        self.assertEqual(held_out.scenario_id, evaluation.seed)

    def test_manual_curriculum_advancement_and_foundation_rehearsal(self) -> None:
        problem = open_problem()
        train = build_random_crossing_scenario(problem, 1, obstacle_count=1)
        evaluation = build_random_crossing_scenario(problem, 2, obstacle_count=1)
        stages = (
            DynamicObstacleCurriculumStage("static", 0, 0, ()),
            DynamicObstacleCurriculumStage("full", 10, 1, (0,)),
        )
        factory = CurriculumScheduledDynamicEnvironmentFactory(
            (train,),
            (evaluation,),
            stages,
            manual_stage_control=True,
            rehearsal_probabilities=((1.0, 0.0), (1.0, 0.0)),
        )

        factory.set_environment_steps(100)
        before_gate = factory(problem, window_size=7)
        advanced = factory.advance_curriculum_stage(100)
        rehearsal = factory(problem, window_size=7)
        factory.set_mode("eval")
        held_out = factory(problem, window_size=7)

        self.assertEqual(before_gate.curriculum_active_stage, "static")
        self.assertEqual(factory.current_stage_index, 1)
        self.assertTrue(advanced)
        self.assertEqual(factory.current_stage_start_environment_step, 100)
        self.assertEqual(rehearsal.curriculum_active_stage, "full")
        self.assertEqual(rehearsal.curriculum_stage, "static")
        self.assertTrue(rehearsal.curriculum_rehearsal)
        self.assertEqual(rehearsal.dynamic_obstacles, ())
        self.assertEqual(held_out.dynamic_obstacles, evaluation.obstacles)
        self.assertEqual(held_out.curriculum_stage, "full_evaluation")

    def test_crossing_route_is_free_and_intersects_nominal_path(self) -> None:
        problem = open_problem()
        spec = build_crossing_obstacle_spec(problem, route_length=5)

        self.assertEqual(len(spec.route), 5)
        self.assertTrue(set(spec.route).intersection(problem.nominal_path))
        self.assertFalse(set(spec.route).intersection(problem.obstacles))
        self.assertNotIn(problem.start, spec.route)
        self.assertNotIn(problem.goal, spec.route)

    def test_invalid_route_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DynamicObstacleSpec(route=((1, 1), (1, 3)))

    def test_bounce_and_reset_are_reproducible(self) -> None:
        problem = open_problem(start=(5, 0), goal=(5, 6))
        spec = DynamicObstacleSpec(route=((1, 3), (2, 3), (3, 3)))
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()

        positions = []
        for _ in range(3):
            result = env.step(int(Action.STAY))
            self.assertAlmostEqual(result.reward, -0.06)
            positions.append(env.dynamic_positions[0])
        self.assertEqual(positions, [(2, 3), (3, 3), (2, 3)])
        env.reset()
        self.assertEqual(env.dynamic_positions, ((1, 3),))
        self.assertEqual(env.steps, 0)

    def test_dynamic_action_risk_supports_current_and_predicted_ablation(self) -> None:
        problem = open_problem(start=(5, 0), goal=(5, 6))
        spec = DynamicObstacleSpec(
            route=((2, 2), (3, 2), (4, 2)),
            start_index=0,
            direction=1,
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()
        env.position = (3, 1)

        self.assertFalse(
            env.dynamic_action_collision_risk(int(Action.RIGHT), predict_next=False)
        )
        self.assertTrue(
            env.dynamic_action_collision_risk(int(Action.RIGHT), predict_next=True)
        )

    def test_observation_contains_current_and_two_previous_frames(self) -> None:
        problem = open_problem(start=(3, 0), goal=(3, 6))
        spec = DynamicObstacleSpec(route=((1, 3), (2, 3), (3, 3)))
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        initial = env.reset()
        after_move = env.step(int(Action.STAY)).observation
        radius = env.window_size // 2

        def local(cell: tuple[int, int]) -> tuple[int, int]:
            return (
                cell[0] - env.position[0] + radius,
                cell[1] - env.position[1] + radius,
            )

        self.assertEqual(initial.spatial.shape, (4, 7, 7))
        self.assertEqual(initial.scalars.shape, (2,))
        self.assertEqual(after_move.spatial[(1, *local((2, 3)))], 1.0)
        self.assertEqual(after_move.spatial[(2, *local((1, 3)))], 1.0)
        self.assertEqual(after_move.spatial[(3, *local((1, 3)))], 1.0)

    def test_agent_entering_current_dynamic_cell_collides(self) -> None:
        problem = open_problem()
        spec = DynamicObstacleSpec(
            route=((2, 1), (3, 1), (4, 1)), start_index=1
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()

        result = env.step(int(Action.RIGHT))

        self.assertTrue(result.info["collision"])
        self.assertEqual(result.info["collision_type"], "dynamic")
        self.assertEqual(env.position, problem.start)
        self.assertEqual(env.dynamic_positions, ((4, 1),))
        self.assertFalse(result.done)
        self.assertAlmostEqual(result.reward, -1.0)
        self.assertAlmostEqual(result.info["reward_step"], 0.0)
        self.assertAlmostEqual(result.info["reward_collision"], -1.0)

    def test_same_next_cell_collision_keeps_agent_in_place(self) -> None:
        problem = open_problem(start=(3, 1), goal=(3, 6))
        spec = DynamicObstacleSpec(
            route=((1, 2), (2, 2), (3, 2)), start_index=1
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()

        result = env.step(int(Action.RIGHT))

        self.assertTrue(result.info["collision"])
        self.assertEqual(env.position, (3, 1))
        self.assertEqual(env.dynamic_positions, ((3, 2),))
        self.assertFalse(result.done)

    def test_cell_swap_collision_is_detected(self) -> None:
        problem = open_problem()
        spec = DynamicObstacleSpec(
            route=((3, 1), (3, 2), (3, 3)), start_index=2, direction=-1
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()
        first = env.step(int(Action.RIGHT))
        self.assertFalse(first.info["collision"])

        result = env.step(int(Action.RIGHT))

        self.assertTrue(result.info["collision"])
        self.assertEqual(env.position, (3, 1))
        self.assertEqual(env.dynamic_positions, ((3, 1),))
        self.assertAlmostEqual(result.reward, -1.0)
        self.assertAlmostEqual(result.info["reward_stay"], 0.0)

    def test_obstacle_entering_staying_agent_collides(self) -> None:
        problem = open_problem()
        spec = DynamicObstacleSpec(
            route=((3, 1), (3, 2), (3, 3)), start_index=2, direction=-1
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()
        env.step(int(Action.RIGHT))

        result = env.step(int(Action.STAY))

        self.assertTrue(result.info["collision"])
        self.assertEqual(env.position, (3, 1))
        self.assertEqual(env.dynamic_positions, ((3, 1),))

    def test_obstacle_entering_departed_cell_is_not_a_collision(self) -> None:
        problem = open_problem()
        spec = DynamicObstacleSpec(
            route=((3, 1), (3, 2), (3, 3)), start_index=2, direction=-1
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()
        env.step(int(Action.RIGHT))

        result = env.step(int(Action.LEFT))

        self.assertFalse(result.info["collision"])
        self.assertEqual(env.position, (3, 0))
        self.assertEqual(env.dynamic_positions, ((3, 1),))

    def test_static_collision_does_not_stop_dynamic_obstacle(self) -> None:
        problem = open_problem(obstacles=frozenset({(2, 0)}))
        spec = DynamicObstacleSpec(route=((1, 3), (2, 3), (3, 3)))
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()

        result = env.step(int(Action.UP))

        self.assertTrue(result.info["collision"])
        self.assertEqual(result.info["collision_type"], "static")
        self.assertEqual(env.position, problem.start)
        self.assertEqual(env.dynamic_positions, ((2, 3),))
        self.assertFalse(result.done)

    def test_nominal_demo_has_zero_dynamic_channels(self) -> None:
        transitions = collect_astar_demonstrations(
            [open_problem()],
            episodes=1,
            seed=7,
            window_size=7,
            spatial_channels=4,
        )

        self.assertTrue(transitions)
        self.assertTrue(
            all(item.state.spatial.shape == (4, 7, 7) for item in transitions)
        )
        self.assertTrue(
            all((item.state.spatial[1:] == 0.0).all() for item in transitions)
        )
        self.assertTrue(
            all((item.next_state.spatial[1:] == 0.0).all() for item in transitions)
        )

    def test_map01_three_obstacle_scenario_is_reproducible(self) -> None:
        problem = open_problem()
        problem = NavigationProblem(
            map_id="calibration_40x40_map_01",
            seed=1500,
            size=40,
            start=(1, 1),
            goal=(39, 39),
            obstacles=frozenset(),
            nominal_path=tuple((1 + i, 1) for i in range(39))
            + tuple((39, 1 + i) for i in range(1, 39)),
        )
        specs = build_map01_three_obstacle_specs(problem)
        self.assertEqual([spec.label for spec in specs], ["crossing", "head_on", "same_direction_slow"])
        self.assertEqual([spec.move_every for spec in specs], [1, 1, 2])
        self.assertEqual(
            [spec.reference_path_source for spec in specs],
            ["uniform_final_path", "prefill_final_path", "persistent_demo_final_path"],
        )
        self.assertEqual(len({spec.route[spec.start_index] for spec in specs}), 3)

    def test_map01_three_crossing_scenario_is_shared_and_reproducible(self) -> None:
        problem = NavigationProblem(
            map_id="calibration_40x40_map_01",
            seed=1500,
            size=40,
            start=(1, 1),
            goal=(39, 39),
            obstacles=frozenset(),
            nominal_path=tuple((1 + i, 1) for i in range(39))
            + tuple((39, 1 + i) for i in range(1, 39)),
        )

        specs = build_map01_three_crossing_obstacle_specs(problem)

        self.assertEqual(
            [spec.label for spec in specs],
            ["crossing_uniform", "crossing_prefill", "crossing_persistent_demo"],
        )
        self.assertEqual([spec.move_every for spec in specs], [1, 1, 1])
        self.assertEqual(
            [spec.reference_path_index for spec in specs],
            [63, 31, 46],
        )
        route_cells = [cell for spec in specs for cell in spec.route]
        self.assertEqual(len(route_cells), len(set(route_cells)))

    def test_map01_controlled_bottleneck_has_required_interactions(self) -> None:
        problem = load_problem_set(
            ROOT / "maps" / "structured_calibration_40x40" / "maps.json"
        )[0]
        specs = build_map01_controlled_bottleneck_obstacle_specs(problem)

        self.assertEqual(
            [spec.label for spec in specs],
            [
                "mandatory_bottleneck",
                "nominal_crossing_middle",
                "nominal_crossing_late",
            ],
        )
        self.assertEqual(
            [spec.reference_path_index for spec in specs],
            [9, 27, 43],
        )
        self.assertEqual(
            specs[0].route,
            ((6, 5), (6, 6), (6, 7), (6, 8), (6, 9)),
        )
        route_cells = [cell for spec in specs for cell in spec.route]
        self.assertEqual(len(route_cells), len(set(route_cells)))
        self.assertIsNone(
            astar_path(
                problem.start,
                problem.goal,
                problem.obstacles | {(6, 5)},
                problem.size,
            )
        )

        phase_env = DynamicGridNavigationEnv(problem, specs, window_size=15)
        phase_env.reset()
        positions_by_step = {}
        for step in range(1, 45):
            phase_env.step(int(Action.STAY))
            if step in {27, 44}:
                positions_by_step[step] = phase_env.dynamic_positions
        self.assertEqual(positions_by_step[27][1], (22, 7))
        self.assertEqual(positions_by_step[44][2], (35, 10))

        blind_env = DynamicGridNavigationEnv(problem, specs, window_size=15)
        blind_env.reset()
        for path_index in range(1, 10):
            result = blind_env.step(
                int(
                    Action.DOWN
                    if problem.nominal_path[path_index][0]
                    > problem.nominal_path[path_index - 1][0]
                    else Action.RIGHT
                )
            )
        self.assertTrue(result.info["collision"])
        self.assertEqual(result.info["collision_type"], "dynamic")
        self.assertEqual(result.info["dynamic_collision_indices"], (0,))
        self.assertEqual(blind_env.position, (6, 4))

        waiting_env = DynamicGridNavigationEnv(problem, specs, window_size=15)
        waiting_env.reset()
        for path_index in range(1, 9):
            waiting_env.step(
                int(
                    Action.DOWN
                    if problem.nominal_path[path_index][0]
                    > problem.nominal_path[path_index - 1][0]
                    else Action.RIGHT
                )
            )
        wait_result = waiting_env.step(int(Action.STAY))
        enter_result = waiting_env.step(int(Action.RIGHT))
        self.assertFalse(wait_result.info["collision"])
        self.assertFalse(enter_result.info["collision"])
        self.assertEqual(waiting_env.position, (6, 5))

    def test_map01_controlled_six_obstacles_combine_all_crossings(self) -> None:
        problem = load_problem_set(
            ROOT / "maps" / "structured_calibration_40x40" / "maps.json"
        )[0]
        specs = build_map01_controlled_six_obstacle_specs(problem)

        self.assertEqual(len(specs), 6)
        self.assertEqual(
            [spec.reference_path_index for spec in specs],
            [9, 13, 27, 28, 36, 43],
        )
        route_cells = [cell for spec in specs for cell in spec.route]
        self.assertEqual(len(route_cells), len(set(route_cells)))
        self.assertTrue(
            all(
                set(spec.route).intersection(problem.nominal_path)
                for spec in specs
            )
        )

        phase_env = DynamicGridNavigationEnv(problem, specs, window_size=15)
        phase_env.reset()
        expected_interactions = {
            13: (1, (10, 5)),
            28: (2, (22, 7)),
            30: (3, (23, 7)),
            39: (4, (29, 9)),
            47: (5, (35, 10)),
        }
        for step in range(1, 48):
            phase_env.step(int(Action.STAY))
            if step in expected_interactions:
                obstacle_index, interaction_cell = expected_interactions[step]
                self.assertEqual(
                    phase_env.dynamic_positions[obstacle_index],
                    interaction_cell,
                )

        safe_env = DynamicGridNavigationEnv(problem, specs, window_size=15)
        safe_env.reset()
        interaction_indices = {9, 13, 27, 28, 36, 43}
        for path_index in range(1, len(problem.nominal_path)):
            if path_index in interaction_indices:
                wait_result = safe_env.step(int(Action.STAY))
                self.assertFalse(wait_result.info["collision"])
            move_result = safe_env.step(
                int(
                    action_between(
                        safe_env.position,
                        problem.nominal_path[path_index],
                    )
                )
            )
            self.assertFalse(move_result.info["collision"])
        self.assertEqual(safe_env.position, problem.goal)
        self.assertEqual(safe_env.steps, problem.astar_steps + 6)

    def test_map01_controlled_mixed_six_obstacles_keep_three_routes_off_path(self) -> None:
        problem = load_problem_set(
            ROOT / "maps" / "structured_calibration_40x40" / "maps.json"
        )[0]
        specs = build_map01_controlled_mixed_six_obstacle_specs(problem)

        self.assertEqual(len(specs), 6)
        self.assertEqual(
            [spec.reference_path_index for spec in specs[:3]],
            [9, 27, 43],
        )
        self.assertEqual(
            [spec.reference_path_source for spec in specs[3:]],
            ["off_nominal_open_space", "off_nominal_open_space", "static_nominal_path"],
        )
        self.assertTrue(
            all(
                not set(spec.route).intersection(problem.nominal_path)
                for spec in specs[3:5]
            )
        )
        self.assertEqual(set(specs[5].route).intersection(problem.nominal_path), {(38, 19)})
        route_cells = [cell for spec in specs for cell in spec.route]
        self.assertEqual(len(route_cells), len(set(route_cells)))
        self.assertEqual(
            [spec.route for spec in specs[3:]],
            [
                ((11, 19), (12, 19), (13, 19), (14, 19), (15, 19)),
                ((24, 32), (24, 33), (24, 34), (24, 35), (24, 36)),
                ((35, 19), (36, 19), (37, 19), (38, 19), (39, 19)),
            ],
        )

    def test_move_every_slows_dynamic_obstacle(self) -> None:
        problem = open_problem()
        spec = DynamicObstacleSpec(
            route=((1, 3), (2, 3), (3, 3)),
            move_every=2,
        )
        env = DynamicGridNavigationEnv(problem, (spec,), window_size=7)
        env.reset()
        self.assertEqual(env.dynamic_positions, ((1, 3),))
        env.step(int(Action.STAY))
        self.assertEqual(env.dynamic_positions, ((1, 3),))
        env.step(int(Action.STAY))
        self.assertEqual(env.dynamic_positions, ((2, 3),))


if __name__ == "__main__":
    unittest.main()
