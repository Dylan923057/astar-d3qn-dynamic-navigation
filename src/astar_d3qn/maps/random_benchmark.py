from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Position, chebyshev

from .problem import NavigationProblem


@dataclass(frozen=True, slots=True)
class RandomMapConfig:
    size: int = 20
    density: float = 0.20
    start: Position = (1, 1)
    goal: Position = (18, 18)
    endpoint_clearance: int = 1
    min_path_steps: int = 36
    max_path_steps: int = 90
    min_turns: int = 2
    max_attempts: int = 5000

    def __post_init__(self) -> None:
        if self.size < 3:
            raise ValueError("Random benchmark size must be at least three.")
        if not 0.0 <= self.density < 1.0:
            raise ValueError("density must be in [0, 1).")
        if self.start == self.goal:
            raise ValueError("start and goal must differ.")
        if self.endpoint_clearance < 0:
            raise ValueError("endpoint_clearance cannot be negative.")
        if not 0 <= self.min_path_steps <= self.max_path_steps:
            raise ValueError("Invalid path-step range.")
        if self.min_turns < 0 or self.max_attempts <= 0:
            raise ValueError("min_turns and max_attempts are invalid.")
        for name, position in (("start", self.start), ("goal", self.goal)):
            if len(position) != 2 or not all(
                0 <= coordinate < self.size for coordinate in position
            ):
                raise ValueError(f"{name} must be inside the map.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RandomMapConfig:
        size = int(values.get("size", 20))
        return cls(
            size=size,
            density=float(values.get("density", 0.20)),
            start=tuple(values.get("start", [1, 1])),
            goal=tuple(values.get("goal", [size - 2, size - 2])),
            endpoint_clearance=int(values.get("endpoint_clearance", 1)),
            min_path_steps=int(values.get("min_path_steps", 36)),
            max_path_steps=int(values.get("max_path_steps", 90)),
            min_turns=int(values.get("min_turns", 2)),
            max_attempts=int(values.get("max_attempts", 5000)),
        )


def build_random_problem(seed: int, config: RandomMapConfig) -> NavigationProblem:
    """Generate a fixed-count random obstacle map that passes A* gates."""

    rng = random.Random(seed)
    candidates = [
        (row, column)
        for row in range(config.size)
        for column in range(config.size)
        if chebyshev((row, column), config.start) > config.endpoint_clearance
        and chebyshev((row, column), config.goal) > config.endpoint_clearance
    ]
    obstacle_count = round(config.size * config.size * config.density)
    if obstacle_count > len(candidates):
        raise ValueError("Requested density exceeds eligible obstacle cells.")

    for attempt in range(1, config.max_attempts + 1):
        obstacles = frozenset(rng.sample(candidates, obstacle_count))
        path = astar_path(config.start, config.goal, obstacles, config.size)
        if path is None:
            continue
        steps = len(path) - 1
        turns = path_turn_count(path)
        if not config.min_path_steps <= steps <= config.max_path_steps:
            continue
        if turns < config.min_turns:
            continue
        density_tag = f"{config.density:.2f}".replace(".", "p")
        return NavigationProblem(
            map_id=f"random_{config.size}x{config.size}_d{density_tag}_seed_{seed}",
            seed=seed,
            size=config.size,
            start=config.start,
            goal=config.goal,
            obstacles=obstacles,
            nominal_path=tuple(path),
            metadata={
                "generator": "astar_iddqn_style_random_v1",
                "generation_attempt": attempt,
                "turn_count": turns,
                "endpoint_clearance": config.endpoint_clearance,
            },
        )
    raise RuntimeError(
        "Unable to generate a random map satisfying the registered gates: "
        f"seed={seed}, attempts={config.max_attempts}."
    )


def build_random_problem_set(
    seed_start: int,
    count: int,
    config: RandomMapConfig,
) -> list[NavigationProblem]:
    if count <= 0:
        raise ValueError("count must be positive.")
    problems = [
        build_random_problem(seed_start + offset, config) for offset in range(count)
    ]
    hashes = [problem.grid_sha256 for problem in problems]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Random benchmark unexpectedly generated duplicate maps.")
    return problems


def path_turn_count(path: list[Position] | tuple[Position, ...]) -> int:
    if len(path) < 3:
        return 0
    directions = [
        (right[0] - left[0], right[1] - left[1])
        for left, right in zip(path, path[1:])
    ]
    return sum(left != right for left, right in zip(directions, directions[1:]))
