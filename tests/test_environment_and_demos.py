from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action, action_between
from astar_d3qn.envs.static_grid import RewardConfig, StaticGridNavigationEnv
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.training.demo_collector import (
    collect_astar_demonstrations,
    demonstration_signature,
    load_demonstrations,
    paths_from_demonstrations,
    save_demonstrations,
)


def simple_problem() -> NavigationProblem:
    obstacles = frozenset({(1, 2), (2, 2), (3, 2)})
    path = astar_path((2, 0), (2, 4), obstacles, 5)
    assert path is not None
    return NavigationProblem(
        map_id="simple",
        seed=1,
        size=5,
        start=(2, 0),
        goal=(2, 4),
        obstacles=obstacles,
        nominal_path=tuple(path),
    )


class StaticEnvironmentTests(unittest.TestCase):
    def test_local_observation_has_padded_occupancy_and_normalized_goal(self) -> None:
        env = StaticGridNavigationEnv(simple_problem(), window_size=5)
        observation = env.reset()

        self.assertEqual(observation.spatial.shape, (1, 5, 5))
        self.assertEqual(observation.scalars.shape, (2,))
        self.assertTrue((observation.spatial[0, :, :2] == 1.0).all())
        self.assertAlmostEqual(float(observation.scalars[0]), 0.0)
        self.assertAlmostEqual(float(observation.scalars[1]), 1.0)

    def test_local_observation_can_pad_empty_dynamic_channels(self) -> None:
        env = StaticGridNavigationEnv(
            simple_problem(),
            window_size=5,
            spatial_channels=4,
        )
        observation = env.reset()

        self.assertEqual(observation.spatial.shape, (4, 5, 5))
        self.assertTrue((observation.spatial[1:] == 0.0).all())
        self.assertTrue((observation.spatial[0, :, :2] == 1.0).all())

    def test_stay_is_valid_and_does_not_move(self) -> None:
        env = StaticGridNavigationEnv(simple_problem())
        env.reset()
        result = env.step(int(Action.STAY))
        self.assertEqual(env.position, (2, 0))
        self.assertFalse(result.done)
        self.assertAlmostEqual(result.reward, -0.06)
        self.assertAlmostEqual(result.info["reward_step"], -0.01)
        self.assertAlmostEqual(result.info["reward_stay"], -0.05)

    def test_terminal_rewards_are_exclusive(self) -> None:
        problem = simple_problem()
        env = StaticGridNavigationEnv(problem)
        env.reset()
        env.step(int(Action.UP))
        env.step(int(Action.RIGHT))
        collision = env.step(int(Action.RIGHT))

        self.assertAlmostEqual(collision.reward, -1.0)
        self.assertAlmostEqual(collision.info["reward_step"], 0.0)
        self.assertAlmostEqual(collision.info["reward_progress"], 0.0)
        self.assertAlmostEqual(collision.info["reward_collision"], -1.0)

        env = StaticGridNavigationEnv(problem)
        env.reset()
        final = None
        for current, following in zip(problem.nominal_path, problem.nominal_path[1:]):
            self.assertEqual(env.position, current)
            final = env.step(int(action_between(current, following)))

        assert final is not None
        self.assertTrue(final.info["reached"])
        self.assertAlmostEqual(final.reward, 10.0)
        self.assertAlmostEqual(final.info["reward_step"], 0.0)
        self.assertAlmostEqual(final.info["reward_progress"], 0.0)
        self.assertAlmostEqual(final.info["reward_goal"], 10.0)

    def test_obstacle_collision_keeps_position_and_episode_running(self) -> None:
        env = StaticGridNavigationEnv(simple_problem())
        env.reset()
        env.step(int(Action.UP))
        env.step(int(Action.RIGHT))
        position_before_collision = env.position
        result = env.step(int(Action.RIGHT))
        self.assertEqual(env.position, position_before_collision)
        self.assertFalse(result.done)
        self.assertTrue(result.info["collision"])
        self.assertEqual(result.info["collision_position"], (1, 2))
        self.assertLess(result.reward, 0.0)

    def test_non_goal_episode_runs_until_max_steps(self) -> None:
        env = StaticGridNavigationEnv(simple_problem(), max_steps=3)
        env.reset()
        first = env.step(int(Action.STAY))
        second = env.step(int(Action.STAY))
        third = env.step(int(Action.STAY))

        self.assertFalse(first.done)
        self.assertFalse(second.done)
        self.assertTrue(third.truncated)
        self.assertEqual(third.info["termination_reason"], "timeout")


class DemonstrationTests(unittest.TestCase):
    def test_astar_demonstrations_are_complete_environment_transitions(self) -> None:
        problem = simple_problem()
        transitions = collect_astar_demonstrations(
            [problem], episodes=3, seed=9, window_size=5
        )
        self.assertEqual(sum(item.terminated for item in transitions), 3)
        self.assertTrue(all(item.action != int(Action.STAY) for item in transitions))
        self.assertTrue(
            all(item.state.spatial.shape == (1, 5, 5) for item in transitions)
        )
        self.assertTrue(all(item.state.scalars.shape == (2,) for item in transitions))

    def test_demo_dataset_round_trip(self) -> None:
        transitions = collect_astar_demonstrations(
            [simple_problem()], episodes=2, seed=4, window_size=5
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "demos.npz"
            save_demonstrations(transitions, path)
            restored = load_demonstrations(path)
        self.assertEqual(len(restored), len(transitions))
        self.assertEqual(
            [item.action for item in restored], [item.action for item in transitions]
        )
        self.assertEqual(
            [item.terminated for item in restored],
            [item.terminated for item in transitions],
        )
        for expected, actual in zip(transitions, restored):
            self.assertTrue((expected.state.spatial == actual.state.spatial).all())
            self.assertTrue((expected.state.scalars == actual.state.scalars).all())

    def test_exact_paths_are_reconstructed_from_frozen_transitions(self) -> None:
        problem = simple_problem()
        transitions = collect_astar_demonstrations(
            [problem], episodes=3, seed=9, window_size=5
        )

        paths = paths_from_demonstrations(problem, transitions)

        self.assertEqual(len(paths), 3)
        self.assertTrue(all(path[0] == problem.start for path in paths))
        self.assertTrue(all(path[-1] == problem.goal for path in paths))
        self.assertEqual(
            sum(len(path) - 1 for path in paths),
            len(transitions),
        )

    def test_demo_signature_includes_map_endpoints(self) -> None:
        signature = demonstration_signature(
            [simple_problem()],
            episodes=2,
            seed=4,
            max_steps=20,
            reward_config=RewardConfig(),
            window_size=5,
        )
        self.assertEqual(signature["maps"][0]["start"], [2, 0])
        self.assertEqual(signature["maps"][0]["goal"], [2, 4])
        self.assertEqual(signature["observation"]["window_size"], 5)


if __name__ == "__main__":
    unittest.main()
