"""A* replanning against the locally observed current dynamic occupancy."""

from __future__ import annotations

import numpy as np

from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action, action_between, in_bounds


class ObservedAStarReplanWaitPolicy:
    """Known static map + current local frame; wait if the observed graph has no route.

    No access to hidden obstacle routes, phases, next positions, or environment.
    This reactive baseline can collide with an obstacle moving into its next cell.
    """

    def __init__(self, problem):
        self.problem = problem

    def select_action(self, state, epsilon=0.0, valid_actions=None):
        if epsilon != 0.0:
            raise ValueError("The A* rule baseline is deterministic and requires epsilon=0.")
        scale = max(1, self.problem.size - 1)
        position = tuple(
            self.problem.goal[axis] - int(round(float(state.scalars[axis]) * scale))
            for axis in (0, 1)
        )
        radius = state.spatial.shape[1] // 2
        occupied = {
            (position[0] + int(row) - radius, position[1] + int(column) - radius)
            for row, column in zip(*np.where(state.spatial[1] > 0.5))
        }
        occupied = {cell for cell in occupied if in_bounds(cell, self.problem.size)}
        blocked = set(self.problem.obstacles) | occupied
        blocked.discard(position)
        path = astar_path(position, self.problem.goal, blocked, self.problem.size)
        action = int(action_between(path[0], path[1])) if path and len(path) > 1 else int(Action.STAY)
        if valid_actions is not None and action not in valid_actions:
            return int(Action.STAY)
        return action
