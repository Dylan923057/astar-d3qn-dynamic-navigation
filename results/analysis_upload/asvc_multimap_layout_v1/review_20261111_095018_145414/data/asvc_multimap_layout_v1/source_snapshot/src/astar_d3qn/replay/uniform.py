from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Sequence

from .transition import Transition


class UniformReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self._transitions: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def add(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        self._transitions.extend(transitions)

    def can_sample(self, batch_size: int) -> bool:
        return batch_size > 0 and len(self) >= batch_size

    def sample(self, batch_size: int) -> list[Transition]:
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from replay size {len(self)}."
            )
        return self._rng.sample(list(self._transitions), batch_size)

    def snapshot(self) -> tuple[Transition, ...]:
        return tuple(self._transitions)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "transitions": list(self._transitions),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Replay capacity does not match checkpoint.")
        transitions: Sequence[Transition] = state.get("transitions", ())
        if len(transitions) > self.capacity:
            raise ValueError("Replay checkpoint exceeds configured capacity.")
        self._transitions = deque(transitions, maxlen=self.capacity)
        self._rng.setstate(state["rng_state"])

    def __len__(self) -> int:
        return len(self._transitions)

