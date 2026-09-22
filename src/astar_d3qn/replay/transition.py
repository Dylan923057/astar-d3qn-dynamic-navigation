from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astar_d3qn.envs.types import Observation


@dataclass(frozen=True, slots=True)
class Transition:
    state: Observation
    action: int
    reward: float
    next_state: Observation
    terminated: bool
    next_action_mask: np.ndarray | None = None
    safe_demo_action: int | None = None
    conflict_safe_action_mask: np.ndarray | None = None
    conflict_blocked_action_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.state.spatial.shape != self.next_state.spatial.shape:
            raise ValueError("state spatial arrays must have identical shapes.")
        if self.state.scalars.shape != self.next_state.scalars.shape:
            raise ValueError("state scalar arrays must have identical shapes.")
        if self.action < 0:
            raise ValueError("action must be non-negative.")
        if self.next_action_mask is not None and self.next_action_mask.ndim != 1:
            raise ValueError("next_action_mask must be one-dimensional.")
        if self.safe_demo_action is not None and self.safe_demo_action < 0:
            raise ValueError("safe_demo_action must be non-negative when provided.")
        conflict_masks = (
            self.conflict_safe_action_mask,
            self.conflict_blocked_action_mask,
        )
        if (conflict_masks[0] is None) != (conflict_masks[1] is None):
            raise ValueError("Conflict-margin action masks must be provided together.")
        if conflict_masks[0] is not None and conflict_masks[1] is not None:
            safe = np.asarray(conflict_masks[0], dtype=bool)
            blocked = np.asarray(conflict_masks[1], dtype=bool)
            if safe.ndim != 1 or blocked.ndim != 1 or safe.shape != blocked.shape:
                raise ValueError(
                    "Conflict-margin action masks must be matching one-dimensional arrays."
                )
            if not safe.any() or not blocked.any():
                raise ValueError("Conflict-margin action masks cannot be empty.")
            if np.logical_and(safe, blocked).any():
                raise ValueError("Safe and blocked conflict-margin actions cannot overlap.")
