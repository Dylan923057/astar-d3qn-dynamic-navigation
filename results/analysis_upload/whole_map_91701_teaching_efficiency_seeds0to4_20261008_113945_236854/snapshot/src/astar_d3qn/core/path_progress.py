from __future__ import annotations

from collections.abc import Sequence

from .grid import Position, manhattan


def nearest_path_index(position: Position, path: Sequence[Position]) -> int:
    if not path:
        return 0
    return min(range(len(path)), key=lambda index: manhattan(position, path[index]))


def project_path_index(
    position: Position,
    path: Sequence[Position],
    previous_index: int,
    backtrack_steps: int = 2,
    forward_steps: int = 2,
) -> int:
    if not path:
        return 0
    previous_index = max(0, min(int(previous_index), len(path) - 1))
    start = max(0, previous_index - max(0, int(backtrack_steps)))
    end = min(len(path) - 1, previous_index + max(1, int(forward_steps)))
    return min(
        range(start, end + 1),
        key=lambda index: (
            manhattan(position, path[index]),
            abs(index - previous_index),
            -index,
        ),
    )


def path_deviation(position: Position, path: Sequence[Position]) -> int:
    if not path:
        return 0
    return min(manhattan(position, cell) for cell in path)

