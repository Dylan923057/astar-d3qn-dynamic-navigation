from __future__ import annotations

from collections.abc import Mapping, Sequence


def total_environment_steps(records: Sequence[Mapping[str, object]]) -> int:
    total = 0
    for record in records:
        steps = int(float(record.get("steps", 0)))
        if steps <= 0:
            raise ValueError("Every completed episode must contain a step.")
        total += steps
    return total


def crossed_interval(before: int, after: int, interval: int) -> bool:
    if interval <= 0 or after <= before:
        return False
    return before // interval < after // interval

