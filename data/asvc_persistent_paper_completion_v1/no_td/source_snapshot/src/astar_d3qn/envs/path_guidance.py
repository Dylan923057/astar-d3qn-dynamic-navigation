"""Optional runtime path observations; the underlying navigation rules are unchanged."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from astar_d3qn.core.grid import in_bounds, manhattan
from astar_d3qn.core.path_progress import nearest_path_index
from astar_d3qn.envs.types import Observation


class PathGuidanceEnvironment:
    """Append one static A* path channel and two subgoal displacement scalars.

    The subgoal is a fixed number of path edges beyond the nearest reference cell.
    Projection is recomputed from position, with the earliest index breaking ties.
    There is no hidden progress cursor, obstacle prediction, action override, or
    path-following reward. The zero-input control uses the same observation sizes.
    """

    def __init__(self, environment, *, enabled=True, lookahead_steps=4, record_trace=False):
        if isinstance(lookahead_steps, bool) or not isinstance(lookahead_steps, int) or lookahead_steps <= 0:
            raise ValueError("lookahead_steps must be a positive integer.")
        if environment.observation_shape[0] != 4 or environment.scalar_dim != 2:
            raise ValueError("Path guidance requires the original four-channel/two-scalar environment.")
        self.environment = environment
        self.enabled = bool(enabled)
        self.lookahead_steps = lookahead_steps
        self.record_trace = bool(record_trace)
        self.trace = []
        self._validate_reference()

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def _validate_reference(self):
        path = self.problem.nominal_path
        if not path or path[0] != self.problem.start or path[-1] != self.problem.goal:
            raise ValueError("Reference path must connect the registered start and goal.")
        if any(not in_bounds(cell, self.problem.size) or cell in self.problem.obstacles for cell in path):
            raise ValueError("Reference path contains a statically invalid cell.")
        if any(manhattan(left, right) != 1 for left, right in zip(path, path[1:])):
            raise ValueError("Reference path must consist of four-connected movement edges.")

    @property
    def observation_shape(self):
        return (5, *self.environment.observation_shape[1:])

    @property
    def scalar_dim(self):
        return 4

    def guidance(self):
        path = self.problem.nominal_path
        nearest = nearest_path_index(self.position, path)
        target_index = min(nearest + self.lookahead_steps, len(path) - 1)
        return {
            "reference_index": nearest,
            "subgoal_index": target_index,
            "subgoal": path[target_index],
            "path_deviation": manhattan(self.position, path[nearest]),
        }

    def _augment(self, original):
        channel = np.zeros((1, *original.spatial.shape[1:]), dtype=np.float32)
        displacement = np.zeros(2, dtype=np.float32)
        if self.enabled:
            radius = self.window_size // 2
            for row, column in self.problem.nominal_path:
                local_row = row - self.position[0] + radius
                local_column = column - self.position[1] + radius
                if 0 <= local_row < self.window_size and 0 <= local_column < self.window_size:
                    channel[0, local_row, local_column] = 1.0
            target = self.guidance()["subgoal"]
            scale = max(1, self.problem.size - 1)
            displacement[:] = [(target[axis] - self.position[axis]) / scale for axis in (0, 1)]
        return Observation(
            spatial=np.concatenate((original.spatial, channel), axis=0),
            scalars=np.concatenate((original.scalars, displacement)),
        )

    def observation(self):
        return self._augment(self.environment.observation())

    def reset(self):
        original = self.environment.reset()
        self.trace = []
        return self._augment(original)

    def set_problem(self, problem):
        self.environment.set_problem(problem)
        self._validate_reference()
        self.trace = []

    def step(self, action):
        before = {
            "position": self.position,
            "dynamic_positions": self.dynamic_positions,
            **self.guidance(),
        } if self.record_trace else None
        result = self.environment.step(action)
        if self.record_trace:
            self.trace.append({
                "step": self.steps,
                "action": int(action),
                "before": before,
                "position": self.position,
                "dynamic_positions": self.dynamic_positions,
                "reward": result.reward,
                **dict(result.info),
            })
        return replace(result, observation=self._augment(result.observation))


class PathGuidanceFactory:
    """Preserve the route sampler and its shape-probe reset contract."""

    def __init__(self, factory, *, enabled, lookahead_steps=4, record_trace=False):
        self.factory = factory
        self.enabled = enabled
        self.lookahead_steps = lookahead_steps
        self.record_trace = record_trace
        self.environments = []

    def reset_schedule(self):
        self.factory.reset_schedule()
        self.environments = []

    def __call__(self, problem, **kwargs):
        environment = PathGuidanceEnvironment(
            self.factory(problem, **kwargs), enabled=self.enabled,
            lookahead_steps=self.lookahead_steps, record_trace=self.record_trace,
        )
        if self.record_trace:
            self.environments.append(environment)
        return environment
