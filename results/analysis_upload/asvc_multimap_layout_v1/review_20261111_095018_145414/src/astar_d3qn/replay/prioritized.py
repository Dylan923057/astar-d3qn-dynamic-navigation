from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

from .transition import Transition


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with importance-sampling weights.

    Priorities are updated by transition identity after a learner step. New
    transitions receive the current maximum priority so they are observed at
    least once before their TD error is known.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        priority_epsilon: float = 1e-6,
        seed: int = 0,
    ):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        if alpha < 0.0:
            raise ValueError("PER alpha must be non-negative.")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("PER beta must be in [0, 1].")
        if priority_epsilon <= 0.0:
            raise ValueError("PER priority epsilon must be positive.")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.priority_epsilon = float(priority_epsilon)
        self._transitions: deque[Transition] = deque(maxlen=self.capacity)
        self._priorities: deque[float] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)
        self._last_sample_weights = np.ones(0, dtype=np.float32)

    def add(self, transition: Transition) -> None:
        maximum = max(self._priorities, default=1.0)
        self._transitions.append(transition)
        self._priorities.append(maximum)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def can_sample(self, batch_size: int) -> bool:
        return batch_size > 0 and len(self) >= batch_size

    def sample(self, batch_size: int) -> list[Transition]:
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from replay size {len(self)}."
            )
        priorities = np.asarray(self._priorities, dtype=np.float64)
        scaled = np.maximum(priorities, self.priority_epsilon) ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = self._rng.choices(
            range(len(self._transitions)),
            weights=probabilities.tolist(),
            k=batch_size,
        )
        selected_probabilities = probabilities[indices]
        weights = (len(self) * selected_probabilities) ** (-self.beta)
        weights /= max(float(weights.max()), 1e-12)
        self._last_sample_weights = weights.astype(np.float32)
        transitions = list(self._transitions)
        return [transitions[index] for index in indices]

    def sample_weights(self) -> np.ndarray:
        if len(self._last_sample_weights) == 0:
            raise RuntimeError("sample_weights() called before sample().")
        return self._last_sample_weights.copy()

    def update_priorities(
        self,
        transitions: Sequence[Transition],
        priorities: Sequence[float],
    ) -> None:
        if len(transitions) != len(priorities):
            raise ValueError("Transitions and priorities must have equal lengths.")
        index_by_id = {id(item): index for index, item in enumerate(self._transitions)}
        values = list(self._priorities)
        for transition, priority in zip(transitions, priorities):
            index = index_by_id.get(id(transition))
            if index is None:
                continue
            value = float(priority)
            if not np.isfinite(value):
                raise ValueError("PER priorities must be finite.")
            values[index] = max(abs(value), self.priority_epsilon)
        self._priorities = deque(values, maxlen=self.capacity)

    def set_beta(self, beta: float) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("PER beta must be in [0, 1].")
        self.beta = float(beta)

    def snapshot(self) -> tuple[Transition, ...]:
        return tuple(self._transitions)

    def priority_snapshot(self) -> tuple[float, ...]:
        return tuple(self._priorities)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "priority_epsilon": self.priority_epsilon,
            "transitions": list(self._transitions),
            "priorities": list(self._priorities),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Replay capacity does not match checkpoint.")
        transitions = list(state.get("transitions", ()))
        priorities = list(state.get("priorities", ()))
        if len(transitions) != len(priorities) or len(transitions) > self.capacity:
            raise ValueError("Invalid prioritized replay checkpoint contents.")
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta = float(state.get("beta", self.beta))
        self.priority_epsilon = float(
            state.get("priority_epsilon", self.priority_epsilon)
        )
        self._transitions = deque(transitions, maxlen=self.capacity)
        self._priorities = deque(
            [max(abs(float(value)), self.priority_epsilon) for value in priorities],
            maxlen=self.capacity,
        )
        self._rng.setstate(state["rng_state"])

    def __len__(self) -> int:
        return len(self._transitions)
