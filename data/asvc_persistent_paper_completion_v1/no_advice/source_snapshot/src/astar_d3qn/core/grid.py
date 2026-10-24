from __future__ import annotations

from enum import IntEnum
from typing import Iterable, Iterator, Sequence, TypeAlias

import numpy as np

Position: TypeAlias = tuple[int, int]


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


ACTION_DELTAS: tuple[Position, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (0, 0),
)
ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right", "stay")
PLANNER_ACTIONS: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
)
EXECUTION_ACTIONS: tuple[Action, ...] = tuple(Action)


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def chebyshev(left: Position, right: Position) -> int:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def in_bounds(position: Position, size: int) -> bool:
    row, column = position
    return 0 <= row < size and 0 <= column < size


def move(position: Position, action: int | Action) -> Position:
    index = int(action)
    if not 0 <= index < len(ACTION_DELTAS):
        raise ValueError(f"Invalid action index: {index}")
    row_delta, column_delta = ACTION_DELTAS[index]
    return position[0] + row_delta, position[1] + column_delta


def planner_neighbors(position: Position, size: int) -> Iterator[Position]:
    """Yield four-connected neighbors; STAY is never a planner edge."""

    for action in PLANNER_ACTIONS:
        candidate = move(position, action)
        if in_bounds(candidate, size):
            yield candidate


def valid_execution_actions(
    position: Position,
    size: int,
    obstacles: Iterable[Position] = (),
) -> tuple[int, ...]:
    blocked = set(obstacles)
    return tuple(
        int(action)
        for action in EXECUTION_ACTIONS
        if in_bounds(move(position, action), size)
        and move(position, action) not in blocked
    )


def action_between(start: Position, goal: Position) -> Action:
    delta = goal[0] - start[0], goal[1] - start[1]
    try:
        return Action(ACTION_DELTAS.index(delta))
    except ValueError as exc:
        raise ValueError(f"Positions are not one execution step apart: {start} -> {goal}") from exc


def obstacle_grid(obstacles: Iterable[Position], size: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    for row, column in obstacles:
        if in_bounds((row, column), size):
            grid[row, column] = 1
    return grid


def obstacle_set_from_grid(grid: np.ndarray) -> frozenset[Position]:
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("Obstacle grid must be a square 2D array.")
    rows, columns = np.where(np.asarray(grid) > 0)
    return frozenset(zip(rows.tolist(), columns.tolist()))


def path_length(path: Sequence[Position] | None) -> int | None:
    if path is None:
        return None
    return max(0, len(path) - 1)

