from __future__ import annotations

import heapq
import itertools
import random
from collections.abc import Collection

from .grid import Position, in_bounds, manhattan, planner_neighbors


def astar_path(
    start: Position,
    goal: Position,
    obstacles: Collection[Position],
    size: int,
) -> list[Position] | None:
    """Return an optimal four-connected path using standard A*."""

    return _astar_path(start, goal, obstacles, size, tie_rng=None)


def randomized_tie_astar_path(
    start: Position,
    goal: Position,
    obstacles: Collection[Position],
    size: int,
    rng: random.Random,
) -> list[Position] | None:
    """Return an optimal A* path while randomizing only equal-f ties."""

    return _astar_path(start, goal, obstacles, size, tie_rng=rng)


def _astar_path(
    start: Position,
    goal: Position,
    obstacles: Collection[Position],
    size: int,
    tie_rng: random.Random | None,
) -> list[Position] | None:
    if size <= 0:
        raise ValueError("Grid size must be positive.")
    if not in_bounds(start, size) or not in_bounds(goal, size):
        raise ValueError("A* endpoints must be inside the grid.")
    blocked = set(obstacles)
    if start in blocked or goal in blocked:
        return None

    counter = itertools.count()
    start_h = manhattan(start, goal)
    start_tie = tie_rng.random() if tie_rng is not None else 0.0
    open_heap: list[tuple[int, float, int, int, Position]] = [
        (start_h, start_tie, next(counter), 0, start)
    ]
    came_from: dict[Position, Position] = {}
    g_score: dict[Position, int] = {start: 0}

    while open_heap:
        _, _, _, current_g, current = heapq.heappop(open_heap)
        if current_g != g_score.get(current):
            continue
        if current == goal:
            return _reconstruct_path(came_from, goal)

        for neighbor in planner_neighbors(current, size):
            if neighbor in blocked:
                continue
            tentative_g = current_g + 1
            if tentative_g >= g_score.get(neighbor, 10**18):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heuristic = manhattan(neighbor, goal)
            tie = tie_rng.random() if tie_rng is not None else float(heuristic)
            heapq.heappush(
                open_heap,
                (
                    tentative_g + heuristic,
                    tie,
                    next(counter),
                    tentative_g,
                    neighbor,
                ),
            )
    return None


def shortest_path_length(
    start: Position,
    goal: Position,
    obstacles: Collection[Position],
    size: int,
) -> int | None:
    path = astar_path(start, goal, obstacles, size)
    return None if path is None else len(path) - 1


def _reconstruct_path(
    came_from: dict[Position, Position], goal: Position
) -> list[Position]:
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

