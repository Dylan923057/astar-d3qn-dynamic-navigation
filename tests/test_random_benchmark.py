from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.maps.io import load_problem_set, save_problem_set
from astar_d3qn.maps.random_benchmark import (
    RandomMapConfig,
    build_random_problem,
    build_random_problem_set,
)


def small_config() -> RandomMapConfig:
    return RandomMapConfig(
        size=10,
        density=0.15,
        start=(1, 1),
        goal=(8, 8),
        endpoint_clearance=1,
        min_path_steps=14,
        max_path_steps=40,
        min_turns=1,
        max_attempts=1000,
    )


class RandomBenchmarkTests(unittest.TestCase):
    def test_same_seed_reproduces_exact_problem(self) -> None:
        first = build_random_problem(41, small_config())
        second = build_random_problem(41, small_config())
        self.assertEqual(first.obstacles, second.obstacles)
        self.assertEqual(first.nominal_path, second.nominal_path)
        self.assertEqual(first.grid_sha256, second.grid_sha256)

    def test_problem_meets_registered_gates(self) -> None:
        config = small_config()
        problem = build_random_problem(42, config)
        self.assertEqual(len(problem.obstacles), round(config.size**2 * config.density))
        self.assertGreaterEqual(problem.astar_steps, config.min_path_steps)
        self.assertLessEqual(problem.astar_steps, config.max_path_steps)
        self.assertGreaterEqual(problem.metadata["turn_count"], config.min_turns)

    def test_problem_set_has_unique_maps(self) -> None:
        problems = build_random_problem_set(50, 5, small_config())
        self.assertEqual(len({problem.grid_sha256 for problem in problems}), 5)

    def test_map_set_uses_one_json_file_and_round_trips(self) -> None:
        problems = build_random_problem_set(43, 2, small_config())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "maps.json"
            save_problem_set(problems, path, metadata={"set": "train"})
            restored = load_problem_set(path)
            self.assertEqual([item.name for item in Path(directory).iterdir()], ["maps.json"])
        self.assertEqual(
            [problem.obstacles for problem in restored],
            [problem.obstacles for problem in problems],
        )
        self.assertEqual(
            [problem.nominal_path for problem in restored],
            [problem.nominal_path for problem in problems],
        )
        self.assertEqual(
            [problem.grid_sha256 for problem in restored],
            [problem.grid_sha256 for problem in problems],
        )


if __name__ == "__main__":
    unittest.main()
