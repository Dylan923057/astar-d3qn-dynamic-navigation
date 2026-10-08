from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .grid import Position


def revisit_ratio(path: Sequence[Position]) -> float:
    if not path:
        return 0.0
    counts = Counter(path)
    revisits = sum(max(0, count - 1) for count in counts.values())
    return revisits / len(path)


def has_terminal_two_cell_cycle(
    path: Sequence[Position], tail_length: int = 20
) -> bool:
    if tail_length < 4:
        raise ValueError("tail_length must be at least four.")
    if len(path) < tail_length:
        return False
    tail = path[-tail_length:]
    first, second = tail[0], tail[1]
    if first == second:
        return False
    return all(
        position == (first if index % 2 == 0 else second)
        for index, position in enumerate(tail)
    )


def path_efficiency(actual_steps: int, shortest_steps: int | None) -> float | None:
    if shortest_steps is None or actual_steps <= 0:
        return None
    return shortest_steps / actual_steps

