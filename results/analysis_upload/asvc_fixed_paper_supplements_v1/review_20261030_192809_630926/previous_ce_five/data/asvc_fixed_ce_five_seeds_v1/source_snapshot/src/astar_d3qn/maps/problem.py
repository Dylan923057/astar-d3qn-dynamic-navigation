from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from astar_d3qn.core.grid import Position, obstacle_grid


@dataclass(frozen=True, slots=True)
class NavigationProblem:
    map_id: str
    seed: int
    size: int
    start: Position
    goal: Position
    obstacles: frozenset[Position]
    nominal_path: tuple[Position, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def astar_steps(self) -> int:
        return max(0, len(self.nominal_path) - 1)

    @property
    def density(self) -> float:
        return len(self.obstacles) / float(self.size * self.size)

    @property
    def grid_sha256(self) -> str:
        grid = obstacle_grid(self.obstacles, self.size)
        return hashlib.sha256(grid.tobytes(order="C")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "seed": self.seed,
            "size": self.size,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacle_count": len(self.obstacles),
            "density": self.density,
            "astar_steps": self.astar_steps,
            "grid_sha256": self.grid_sha256,
            **dict(self.metadata),
        }

