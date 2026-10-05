from __future__ import annotations

import random
from collections.abc import Iterable

from .transition import Transition
from .uniform import UniformReplayBuffer


class PersistentDemoReplay:
    """Keep demonstrations immutable while online transitions use a ring buffer."""

    def __init__(
        self,
        demonstrations: Iterable[Transition],
        online_capacity: int,
        demo_fraction: float,
        seed: int = 0,
    ):
        demos = tuple(demonstrations)
        if not demos:
            raise ValueError("Persistent demonstration replay requires demonstrations.")
        if not 0.0 <= demo_fraction <= 1.0:
            raise ValueError("demo_fraction must be in [0, 1].")
        self._demonstrations = demos
        self.demo_fraction = float(demo_fraction)
        self.online = UniformReplayBuffer(online_capacity, seed=seed + 1)
        self._rng = random.Random(seed)

    @property
    def demonstration_size(self) -> int:
        return len(self._demonstrations)

    @property
    def online_size(self) -> int:
        return len(self.online)

    def add(self, transition: Transition) -> None:
        self.online.add(transition)

    def set_demo_fraction(self, demo_fraction: float) -> None:
        if not 0.0 <= demo_fraction <= 1.0:
            raise ValueError("demo_fraction must be in [0, 1].")
        self.demo_fraction = float(demo_fraction)

    def sample_counts(self, batch_size: int) -> tuple[int, int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        demo_count = min(batch_size, int(batch_size * self.demo_fraction + 0.5))
        return demo_count, batch_size - demo_count

    def can_sample(self, batch_size: int) -> bool:
        demo_count, online_count = self.sample_counts(batch_size)
        return (
            self.demonstration_size >= demo_count
            and self.online_size >= online_count
        )

    def sample(self, batch_size: int) -> list[Transition]:
        demo_count, online_count = self.sample_counts(batch_size)
        if not self.can_sample(batch_size):
            raise ValueError(
                "Insufficient replay data for requested persistent-demo batch: "
                f"demo={self.demonstration_size}/{demo_count}, "
                f"online={self.online_size}/{online_count}."
            )
        batch = self._rng.sample(self._demonstrations, demo_count)
        if online_count:
            batch.extend(self.online.sample(online_count))
        self._rng.shuffle(batch)
        return batch

    def demonstration_snapshot(self) -> tuple[Transition, ...]:
        return self._demonstrations

    def snapshot(self) -> tuple[Transition, ...]:
        return self._demonstrations + self.online.snapshot()

    def __len__(self) -> int:
        return self.demonstration_size + self.online_size
