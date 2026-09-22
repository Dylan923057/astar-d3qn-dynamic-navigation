from __future__ import annotations

import random
import sys
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.astar import astar_path, randomized_tie_astar_path
from astar_d3qn.core.grid import (
    Action,
    manhattan,
    planner_neighbors,
    valid_execution_actions,
)


def bfs_length(start, goal, obstacles, size):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current, distance = queue.popleft()
        if current == goal:
            return distance
        for neighbor in planner_neighbors(current, size):
            if neighbor not in obstacles and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


class GridActionTests(unittest.TestCase):
    def test_planner_never_expands_stay(self) -> None:
        position = (2, 2)
        neighbors = tuple(planner_neighbors(position, 5))
        self.assertEqual(len(neighbors), 4)
        self.assertNotIn(position, neighbors)

    def test_execution_space_contains_stay(self) -> None:
        actions = valid_execution_actions((2, 2), 5)
        self.assertIn(int(Action.STAY), actions)
        self.assertEqual(len(actions), 5)


class AstarCorrectnessTests(unittest.TestCase):
    def test_known_detour_is_optimal_and_valid(self) -> None:
        obstacles = {(1, 2), (2, 2), (3, 2)}
        path = astar_path((2, 0), (2, 4), obstacles, 5)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(len(path) - 1, 8)
        self.assertTrue(all(cell not in obstacles for cell in path))
        self.assertTrue(
            all(manhattan(left, right) == 1 for left, right in zip(path, path[1:]))
        )

    def test_astar_matches_bfs_on_random_grids(self) -> None:
        for seed in range(30):
            rng = random.Random(seed)
            size = 8
            candidates = [
                (row, column)
                for row in range(size)
                for column in range(size)
                if (row, column) not in {(0, 0), (7, 7)}
            ]
            obstacles = set(rng.sample(candidates, 12))
            expected = bfs_length((0, 0), (7, 7), obstacles, size)
            path = astar_path((0, 0), (7, 7), obstacles, size)
            actual = None if path is None else len(path) - 1
            self.assertEqual(actual, expected, msg=f"seed={seed}")

    def test_random_ties_preserve_optimality_and_create_variants(self) -> None:
        paths = {
            tuple(randomized_tie_astar_path((0, 0), (5, 5), set(), 6, random.Random(seed)) or ())
            for seed in range(20)
        }
        self.assertGreater(len(paths), 1)
        self.assertTrue(all(len(path) - 1 == 10 for path in paths))

    def test_blocked_endpoint_returns_no_path(self) -> None:
        self.assertIsNone(astar_path((0, 0), (2, 2), {(2, 2)}, 3))


if __name__ == "__main__":
    unittest.main()

