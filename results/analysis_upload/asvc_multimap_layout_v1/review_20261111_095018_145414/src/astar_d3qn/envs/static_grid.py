from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astar_d3qn.core.grid import (
    ACTION_DELTAS,
    Action,
    Position,
    in_bounds,
    manhattan,
    move,
)
from astar_d3qn.maps.problem import NavigationProblem

from .types import StepResult
from .types import Observation


@dataclass(frozen=True, slots=True)
class RewardConfig:
    step: float = -0.01
    progress: float = 0.05
    stay: float = -0.05
    collision: float = -1.0
    goal: float = 10.0


class StaticGridNavigationEnv:
    """Static benchmark with an agent-centered local occupancy observation."""

    def __init__(
        self,
        problem: NavigationProblem,
        max_steps: int = 350,
        reward_config: RewardConfig | None = None,
        terminate_on_collision: bool = False,
        window_size: int | None = 11,
        spatial_channels: int | None = None,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.max_steps = int(max_steps)
        self.reward_config = reward_config or RewardConfig()
        self.terminate_on_collision = bool(terminate_on_collision)
        if window_size is not None and (window_size <= 0 or window_size % 2 == 0):
            raise ValueError("window_size must be a positive odd integer or None.")
        self.window_size = int(window_size) if window_size is not None else None
        default_channels = 3 if self.window_size is None else 1
        self.spatial_channels = (
            default_channels if spatial_channels is None else int(spatial_channels)
        )
        if self.spatial_channels <= 0:
            raise ValueError("spatial_channels must be positive.")
        if self.window_size is None and self.spatial_channels != 3:
            raise ValueError("Global static observations require exactly three channels.")
        self.problem = problem
        self.position: Position = problem.start
        self.steps = 0

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        if self.window_size is None:
            return 3, self.problem.size, self.problem.size
        return self.spatial_channels, self.window_size, self.window_size

    @property
    def scalar_dim(self) -> int:
        return 0 if self.window_size is None else 2

    @property
    def action_dim(self) -> int:
        return len(ACTION_DELTAS)

    def set_problem(self, problem: NavigationProblem) -> None:
        if problem.size != self.problem.size:
            raise ValueError("All benchmark problems must share one grid size.")
        self.problem = problem
        self.reset()

    def reset(self) -> Observation:
        self.position = self.problem.start
        self.steps = 0
        return self.observation()

    def observation(self) -> Observation:
        if self.window_size is None:
            spatial = np.zeros(self.observation_shape, dtype=np.float32)
            for row, column in self.problem.obstacles:
                spatial[0, row, column] = 1.0
            spatial[1, self.position[0], self.position[1]] = 1.0
            spatial[2, self.problem.goal[0], self.problem.goal[1]] = 1.0
            return Observation(spatial=spatial, scalars=np.zeros(0, dtype=np.float32))

        radius = self.window_size // 2
        # Channel 0 is static occupancy. Optional additional channels stay zero,
        # matching an empty dynamic-history observation for retention evaluation.
        spatial = np.zeros(self.observation_shape, dtype=np.float32)
        spatial[0] = 1.0
        for local_row in range(self.window_size):
            map_row = self.position[0] + local_row - radius
            for local_column in range(self.window_size):
                map_column = self.position[1] + local_column - radius
                candidate = (map_row, map_column)
                if in_bounds(candidate, self.problem.size):
                    spatial[0, local_row, local_column] = float(
                        candidate in self.problem.obstacles
                    )

        scale = max(1, self.problem.size - 1)
        scalars = np.asarray(
            [
                (self.problem.goal[0] - self.position[0]) / scale,
                (self.problem.goal[1] - self.position[1]) / scale,
            ],
            dtype=np.float32,
        )
        return Observation(spatial=spatial, scalars=scalars)

    def action_mask(self, mask_collisions: bool = False) -> np.ndarray:
        if not mask_collisions:
            return np.ones(self.action_dim, dtype=bool)
        mask = np.zeros(self.action_dim, dtype=bool)
        for action in Action:
            candidate = move(self.position, action)
            mask[int(action)] = (
                in_bounds(candidate, self.problem.size)
                and candidate not in self.problem.obstacles
            )
        return mask

    def step(self, action: int) -> StepResult:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"Invalid action index: {action}")
        action = int(action)
        old_position = self.position
        candidate = move(old_position, action)
        self.steps += 1
        collision = (
            not in_bounds(candidate, self.problem.size)
            or candidate in self.problem.obstacles
        )
        if not collision:
            self.position = candidate

        reached = not collision and self.position == self.problem.goal
        old_distance = manhattan(old_position, self.problem.goal)
        new_distance = manhattan(self.position, self.problem.goal)
        # Terminal events are exclusive. Progress and waiting costs apply only to
        # ordinary transitions, so collision/goal rewards cannot be double counted.
        if reached:
            components = {
                "step": 0.0,
                "progress": 0.0,
                "stay": 0.0,
                "collision": 0.0,
                "goal": self.reward_config.goal,
            }
        elif collision:
            components = {
                "step": 0.0,
                "progress": 0.0,
                "stay": 0.0,
                "collision": self.reward_config.collision,
                "goal": 0.0,
            }
        else:
            components = {
                "step": self.reward_config.step,
                "progress": self.reward_config.progress * (old_distance - new_distance),
                "stay": self.reward_config.stay if action == int(Action.STAY) else 0.0,
                "collision": 0.0,
                "goal": 0.0,
            }
        terminated = reached or (collision and self.terminate_on_collision)
        truncated = not terminated and self.steps >= self.max_steps
        info = {
            "position": self.position,
            "collision_position": (
                candidate if collision and in_bounds(candidate, self.problem.size) else None
            ),
            "steps": self.steps,
            "collision": collision,
            "collision_type": "static" if collision else None,
            "dynamic_collision_indices": tuple(),
            "reached": reached,
            "termination_reason": (
                "goal"
                if reached
                else "collision"
                if terminated
                else "timeout"
                if truncated
                else "running"
            ),
            **{f"reward_{key}": value for key, value in components.items()},
        }
        return StepResult(
            observation=self.observation(),
            reward=float(sum(components.values())),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
