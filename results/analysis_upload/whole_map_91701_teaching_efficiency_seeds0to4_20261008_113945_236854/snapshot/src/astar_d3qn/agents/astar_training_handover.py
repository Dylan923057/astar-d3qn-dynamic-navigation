"""Training-only static A* advice, with optional observed-risk handover."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import ACTION_DELTAS, action_between


PRIMARY_METHODS = ("unguided", "astar_fixed", "astar_decay", "astar_risk_decay")
METHODS = (*PRIMARY_METHODS, "astar_budget_matched")


class AStarTrainingHandoverAgent(D3QNAgent):
    """Advice changes collected actions, never targets, rewards or evaluation.

    The planner knows the static map. The veto only reads the three dynamic
    occupancy channels also given to D3QN. It cannot read obstacle routes or
    their actual next positions. Epsilon=0 bypasses advice and its entire state.
    """

    def __init__(self, config, problem, *, method, probability=0.8,
                 decay_steps=100000, risk_radius=1, rng_offset=730000, advice_quota=None):
        if method not in METHODS:
            raise ValueError(f"Unknown handover method: {method}")
        if not 0 <= probability <= 1 or decay_steps <= 0 or risk_radius < 0:
            raise ValueError("Invalid advice parameters.")
        super().__init__(config)
        self.problem = problem
        self.method = method
        self.probability = float(probability)
        self.decay_steps = int(decay_steps)
        self.risk_radius = int(risk_radius)
        self.advice_rng = random.Random(config.seed + rng_offset)
        self.training_action_steps = 0
        self.bins = {}
        self._static_action_cache = {}
        self.advice_quota = None if advice_quota is None else tuple(int(count) for count in advice_quota)
        self._quota_steps = set()
        if method == "astar_budget_matched":
            if self.advice_quota is None or len(self.advice_quota) != 20 or any(not 0 <= count <= 10000 for count in self.advice_quota):
                raise ValueError("Budget-matched control requires twenty registered 10000-step quotas.")
            quota_rng = random.Random(config.seed + rng_offset + 1)
            for index, count in enumerate(self.advice_quota):
                self._quota_steps.update(quota_rng.sample(range(index * 10000 + 1, (index + 1) * 10000 + 1), count))

    def advice_probability(self, step=None):
        """Step is the 1-based environment action about to be collected."""
        step = self.training_action_steps + 1 if step is None else step
        if self.method == "unguided":
            return 0.0
        if self.method == "astar_fixed":
            return self.probability
        if self.method == "astar_budget_matched":
            index = (step - 1) // 10000
            return self.advice_quota[index] / 10000 if 0 <= index < len(self.advice_quota) else 0.0
        return self.probability * max(0.0, 1.0 - step / self.decay_steps)

    def static_advice(self, state):
        scale = max(1, self.problem.size - 1)
        position = tuple(self.problem.goal[axis] - int(round(float(state.scalars[axis]) * scale))
                         for axis in (0, 1))
        if position not in self._static_action_cache:
            path = astar_path(position, self.problem.goal, self.problem.obstacles, self.problem.size)
            self._static_action_cache[position] = (
                int(action_between(path[0], path[1])) if path and len(path) > 1 else None
            )
        return self._static_action_cache[position]

    def observed_risk(self, state, action):
        radius = state.spatial.shape[1] // 2
        dr, dc = ACTION_DELTAS[action]
        target = (radius + dr, radius + dc)
        occupied = np.any(state.spatial[1:4] > 0.5, axis=0)
        rows, columns = np.where(occupied)
        return bool(np.any(np.abs(rows - target[0]) + np.abs(columns - target[1]) <= self.risk_radius))

    def select_action(self, state, epsilon, valid_actions=None):
        # Generate the ordinary D3QN candidate even when advice will replace it:
        # every group advances the learner's exploration RNG in the same way.
        learner_action = super().select_action(state, epsilon, valid_actions)
        if epsilon == 0.0:
            return learner_action
        step = self.training_action_steps + 1
        probability = self.advice_probability(step)
        requested = self.advice_rng.random() < probability
        if self.method == "astar_budget_matched":
            requested = step in self._quota_steps
        teacher_action = self.static_advice(state)
        available = teacher_action is not None and (valid_actions is None or teacher_action in valid_actions)
        risk = self.observed_risk(state, teacher_action) if available else False
        veto = requested and available and self.method == "astar_risk_decay" and risk
        applied = requested and available and not veto
        if requested and not available and self.method == "astar_budget_matched":
            raise RuntimeError("An unavailable static action would invalidate the exact advice-budget match.")
        action = teacher_action if applied else learner_action
        counts = self.bins.setdefault((step - 1) // 10000, Counter())
        counts.update({"steps": 1, "requested": int(requested), "applied": int(applied),
                       "risk_veto": int(veto), "unavailable": int(requested and not available),
                       "advice_changes_action": int(applied and action != learner_action),
                       "observed_risk_steps": int(risk)})
        counts["probability_sum"] += probability
        self.training_action_steps = step
        return int(action)

    def advice_records(self):
        return [{"bin_start_step": index * 10000 + 1,
                 "bin_end_step": min((index + 1) * 10000, self.training_action_steps),
                 **dict(counts), "application_rate": counts["applied"] / counts["steps"],
                 "mean_requested_probability": counts["probability_sum"] / counts["steps"]}
                for index, counts in sorted(self.bins.items())]

    def training_state_dict(self):
        state = super().training_state_dict()
        state["advice_state"] = {"rng_state": self.advice_rng.getstate(),
                                 "training_action_steps": self.training_action_steps,
                                 "bins": {key: dict(value) for key, value in self.bins.items()}}
        return state

    def load_training_state_dict(self, state):
        super().load_training_state_dict(state)
        advice = state["advice_state"]
        self.advice_rng.setstate(advice["rng_state"])
        self.training_action_steps = int(advice["training_action_steps"])
        self.bins = {int(key): Counter(value) for key, value in advice["bins"].items()}
