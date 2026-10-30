"""Opt-in reward experiments; the historical environment is unchanged."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from math import isfinite

import numpy as np

from astar_d3qn.core.grid import planner_neighbors
from astar_d3qn.envs.types import Observation

VARIANTS = ("legacy", "potential", "potential_deadline")


def goal_distances(problem):
    """Exact four-connected distances on the known static map, without dynamics."""
    distances = {problem.goal: 0}
    pending = deque([problem.goal])
    while pending:
        cell = pending.popleft()
        for neighbor in planner_neighbors(cell, problem.size):
            if neighbor not in problem.obstacles and neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                pending.append(neighbor)
    if problem.start not in distances:
        raise ValueError("Start must be statically reachable from the goal.")
    return distances


class RewardStudyEnvironment:
    """Wrap the existing five-channel, four-scalar unguided environment.

    Nonterminal potential is alpha * (D_max - D(position)), hence nonnegative.
    Terminal potential is zero. This convention keeps ordinary waiting negative.
    Only the deadline variant uses scalar 3 (previously zero) for remaining time.
    """

    def __init__(self, environment, *, variant, gamma=0.99, alpha=0.05,
                 timeout_reward=-1.0):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown reward variant: {variant}")
        if not all(isfinite(x) for x in (gamma, alpha, timeout_reward)):
            raise ValueError("Reward settings must be finite.")
        if not 0 < gamma < 1 or alpha <= 0 or timeout_reward > 0:
            raise ValueError("Require 0 < gamma < 1, alpha > 0, timeout_reward <= 0.")
        if not environment.terminate_on_collision:
            raise ValueError("This experiment requires terminal collisions.")
        if environment.scalar_dim != 4 or environment.enabled:
            raise ValueError("Use the unguided four-scalar PathGuidanceEnvironment.")
        self.environment = environment
        self.variant, self.gamma, self.alpha = variant, float(gamma), float(alpha)
        self.timeout_reward = float(timeout_reward)
        self.distances = goal_distances(environment.problem)
        self.distance_max = max(self.distances.values())
        self._done = False

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "environment"), name)

    @property
    def deadline(self):
        return self.variant == "potential_deadline"

    def potential(self, position, *, terminal=False):
        if terminal:
            return 0.0
        return self.alpha * (self.distance_max - self.distances[tuple(position)])

    def return_bounds(self):
        """Conservative discounted-return bounds, not bounds on raw network Q."""
        reward = self.reward_config
        if self.variant == "legacy":
            low = min(reward.step - reward.progress, reward.step + reward.stay)
            high = max(reward.step + reward.progress, reward.step + reward.stay)
            return (min(low / (1 - self.gamma), reward.collision, reward.goal),
                    max(high / (1 - self.gamma), reward.collision, reward.goal))
        base_low = min(reward.step / (1 - self.gamma), reward.collision, reward.goal)
        if self.deadline:
            base_low = min(base_low, self.timeout_reward)
        base_high = max(reward.step / (1 - self.gamma), reward.collision, reward.goal)
        max_phi = max(self.potential(p) for p in self.distances if p != self.problem.goal)
        return base_low - max_phi, base_high

    def _observation(self, original):
        if not self.deadline:
            return original
        scalars = original.scalars.copy()
        scalars[3] = max(0, self.max_steps - self.steps) / self.max_steps
        return Observation(original.spatial, scalars)

    def observation(self):
        return self._observation(self.environment.observation())

    def reset(self):
        self._done = False
        return self._observation(self.environment.reset())

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset the environment after episode end.")
        previous = self.position
        result = self.environment.step(action)
        self._done = result.done
        if self.variant == "legacy":
            return result

        timeout = bool(result.truncated and not result.terminated)
        deadline_reached = self.deadline and timeout
        terminal = bool(result.terminated or deadline_reached)
        info = dict(result.info)
        if info["reached"]:
            base = self.reward_config.goal
        elif info["collision"]:
            base = self.reward_config.collision
        elif deadline_reached:
            base = self.timeout_reward
        else:
            base = self.reward_config.step
        phi = self.potential(previous)
        next_phi = self.potential(self.position, terminal=terminal)
        shaping = self.gamma * next_phi - phi
        for key in tuple(info):
            if key.startswith("reward_"):
                info["legacy_" + key] = info.pop(key)
        info.update(
            reward_base=float(base), reward_shaping=shaping,
            potential_before=phi, potential_after=next_phi,
            static_distance_before=self.distances[previous],
            static_distance_after=self.distances[self.position],
            deadline_reached=deadline_reached,
        )
        # A genuine task deadline terminates bootstrapping. A sampling cutoff
        # in the other variants remains truncated, with its nonzero potential.
        return replace(
            result, reward=float(base + shaping),
            observation=self._observation(result.observation),
            terminated=terminal,
            truncated=bool(result.truncated and not deadline_reached),
            info=info,
        )


class RewardStudyFactory:
    def __init__(self, factory, **settings):
        self.factory, self.settings = factory, settings

    def reset_schedule(self):
        self.factory.reset_schedule()

    def __call__(self, problem, **kwargs):
        return RewardStudyEnvironment(self.factory(problem, **kwargs), **self.settings)
