from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CollisionTracker:
    """Accumulate collision types without changing environment semantics."""

    total_count: int = 0
    static_count: int = 0
    dynamic_count: int = 0
    first_static_step: int | None = None
    first_dynamic_step: int | None = None
    dynamic_obstacle_contact_count: int = 0
    dynamic_obstacle_indices: set[int] = field(default_factory=set)

    def record(self, info: Mapping[str, Any], step: int) -> None:
        if not bool(info.get("collision", False)):
            return

        self.total_count += 1
        collision_type = info.get("collision_type") or "static"
        if collision_type == "static":
            self.static_count += 1
            if self.first_static_step is None:
                self.first_static_step = int(step)
            return
        if collision_type != "dynamic":
            raise ValueError(f"Unknown collision type: {collision_type!r}.")

        self.dynamic_count += 1
        if self.first_dynamic_step is None:
            self.first_dynamic_step = int(step)
        indices = tuple(int(value) for value in info.get("dynamic_collision_indices", ()))
        self.dynamic_obstacle_contact_count += len(indices)
        self.dynamic_obstacle_indices.update(indices)

    def metrics(self) -> dict[str, Any]:
        return {
            "static_collision": float(self.static_count > 0),
            "static_collision_count": self.static_count,
            "dynamic_collision": float(self.dynamic_count > 0),
            "dynamic_collision_count": self.dynamic_count,
            "first_static_collision_step": self.first_static_step,
            "first_dynamic_collision_step": self.first_dynamic_step,
            "dynamic_obstacle_contact_count": self.dynamic_obstacle_contact_count,
            "unique_dynamic_obstacles_hit": len(self.dynamic_obstacle_indices),
            "dynamic_obstacle_indices_hit": ";".join(
                str(index) for index in sorted(self.dynamic_obstacle_indices)
            ),
        }
