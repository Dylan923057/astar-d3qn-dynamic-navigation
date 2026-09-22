from __future__ import annotations

import random
from collections.abc import Iterable
from math import floor

from .transition import Transition


DemoKey = tuple[tuple[int, int], int]
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
        demo_count = min(
            batch_size,
            int(batch_size * self.demo_fraction + 0.5),
        )
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
        """Return demonstrations and currently retained online transitions."""

        return self._demonstrations + self.online.snapshot()

    def __len__(self) -> int:
        return self.demonstration_size + self.online_size


class RiskCoverageHandoverReplay(PersistentDemoReplay):
    """Replace static-demo batch slots with covered dynamic-risk experience.

    Risk transitions remain in the ordinary online buffer and are additionally
    retained in a small dedicated partition.  The total guidance budget stays
    fixed, so the method changes which guidance source is sampled without
    increasing the batch size or reducing the ordinary-online allocation.
    """

    def __init__(
        self,
        demonstrations: Iterable[Transition],
        online_capacity: int,
        guidance_fraction: float,
        risk_capacity: int,
        coverage_target: int,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= guidance_fraction <= 1.0:
            raise ValueError("guidance_fraction must be in [0, 1].")
        if risk_capacity <= 0:
            raise ValueError("risk_capacity must be positive.")
        if coverage_target <= 0:
            raise ValueError("coverage_target must be positive.")
        super().__init__(
            demonstrations,
            online_capacity=online_capacity,
            demo_fraction=guidance_fraction,
            seed=seed,
        )
        self.guidance_fraction = float(guidance_fraction)
        self.risk = UniformReplayBuffer(risk_capacity, seed=seed + 17)
        self.coverage_target = int(coverage_target)
        self._covered_risk_keys: set[tuple] = set()
        self.risk_total_added = 0
        self.last_demo_sample_count = 0
        self.last_risk_sample_count = 0
        self.last_online_sample_count = 0

    @property
    def risk_size(self) -> int:
        return len(self.risk)

    @property
    def covered_risk_count(self) -> int:
        return len(self._covered_risk_keys)

    @property
    def handover_progress(self) -> float:
        return min(1.0, self.covered_risk_count / self.coverage_target)

    def add_risk(self, transition: Transition, risk_key: tuple) -> None:
        if not risk_key:
            raise ValueError("risk_key cannot be empty.")
        self.risk.add(transition)
        self._covered_risk_keys.add(tuple(risk_key))
        self.risk_total_added += 1

    def partition_counts(self, batch_size: int) -> tuple[int, int, int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        guidance_count = min(
            batch_size,
            int(batch_size * self.guidance_fraction + 0.5),
        )
        requested_risk = int(guidance_count * self.handover_progress + 0.5)
        risk_count = min(requested_risk, self.risk_size, guidance_count)
        demo_count = guidance_count - risk_count
        online_count = batch_size - guidance_count
        return demo_count, risk_count, online_count

    def sample_counts(self, batch_size: int) -> tuple[int, int]:
        demo_count, risk_count, online_count = self.partition_counts(batch_size)
        return demo_count, risk_count + online_count

    def can_sample(self, batch_size: int) -> bool:
        demo_count, risk_count, online_count = self.partition_counts(batch_size)
        return (
            self.demonstration_size >= demo_count
            and self.risk_size >= risk_count
            and self.online_size >= online_count
        )

    def sample(self, batch_size: int) -> list[Transition]:
        demo_count, risk_count, online_count = self.partition_counts(batch_size)
        if not self.can_sample(batch_size):
            raise ValueError(
                "Insufficient replay data for risk-handover batch: "
                f"demo={self.demonstration_size}/{demo_count}, "
                f"risk={self.risk_size}/{risk_count}, "
                f"online={self.online_size}/{online_count}."
            )
        batch = self._rng.sample(self._demonstrations, demo_count)
        if risk_count:
            batch.extend(self.risk.sample(risk_count))
        if online_count:
            batch.extend(self.online.sample(online_count))
        self._rng.shuffle(batch)
        self.last_demo_sample_count = demo_count
        self.last_risk_sample_count = risk_count
        self.last_online_sample_count = online_count
        return batch


class SafeInterventionReplay(PersistentDemoReplay):
    """Augment time-decayed static demos with executed safe interventions."""

    def __init__(
        self,
        demonstrations: Iterable[Transition],
        online_capacity: int,
        demo_fraction: float,
        safe_capacity: int,
        safe_samples_per_batch: int,
        seed: int = 0,
    ) -> None:
        if safe_capacity <= 0:
            raise ValueError("safe_capacity must be positive.")
        if safe_samples_per_batch < 0:
            raise ValueError("safe_samples_per_batch must be nonnegative.")
        super().__init__(
            demonstrations,
            online_capacity=online_capacity,
            demo_fraction=demo_fraction,
            seed=seed,
        )
        self.safe = UniformReplayBuffer(safe_capacity, seed=seed + 29)
        self.safe_samples_per_batch = int(safe_samples_per_batch)
        self._covered_safe_keys: set[tuple] = set()
        self.safe_total_added = 0
        self.last_demo_sample_count = 0
        self.last_safe_sample_count = 0
        self.last_online_sample_count = 0

    @property
    def safe_size(self) -> int:
        return len(self.safe)

    @property
    def safe_coverage_count(self) -> int:
        return len(self._covered_safe_keys)

    def add_safe(self, transition: Transition, safe_key: tuple) -> None:
        if not safe_key:
            raise ValueError("safe_key cannot be empty.")
        self.safe.add(transition)
        self._covered_safe_keys.add(tuple(safe_key))
        self.safe_total_added += 1

    def partition_counts(self, batch_size: int) -> tuple[int, int, int]:
        demo_count, _ = super().sample_counts(batch_size)
        safe_count = min(
            self.safe_samples_per_batch,
            self.safe_size,
            batch_size - demo_count,
        )
        online_count = batch_size - demo_count - safe_count
        return demo_count, safe_count, online_count

    def sample_counts(self, batch_size: int) -> tuple[int, int]:
        demo_count, safe_count, online_count = self.partition_counts(batch_size)
        return demo_count, safe_count + online_count

    def can_sample(self, batch_size: int) -> bool:
        demo_count, safe_count, online_count = self.partition_counts(batch_size)
        return (
            self.demonstration_size >= demo_count
            and self.safe_size >= safe_count
            and self.online_size >= online_count
        )

    def sample(self, batch_size: int) -> list[Transition]:
        demo_count, safe_count, online_count = self.partition_counts(batch_size)
        if not self.can_sample(batch_size):
            raise ValueError(
                "Insufficient replay data for safe-intervention batch: "
                f"demo={self.demonstration_size}/{demo_count}, "
                f"safe={self.safe_size}/{safe_count}, "
                f"online={self.online_size}/{online_count}."
            )
        batch = self._rng.sample(self._demonstrations, demo_count)
        if safe_count:
            batch.extend(self.safe.sample(safe_count))
        if online_count:
            batch.extend(self.online.sample(online_count))
        self._rng.shuffle(batch)
        self.last_demo_sample_count = demo_count
        self.last_safe_sample_count = safe_count
        self.last_online_sample_count = online_count
        return batch


class LocalConflictDemoReplay(PersistentDemoReplay):
    """Persist demonstrations while locally suppressing unsafe state-actions.

    The total demonstration fraction is unchanged.  Only demonstrations whose
    recovered ``(position, action)`` key was recently observed to conflict with
    a moving obstacle receive a lower sampling weight.  Suppression expires on
    an environment-step clock or is removed immediately when the same action is
    observed to be safe again.
    """

    def __init__(
        self,
        demonstrations: Iterable[Transition],
        demonstration_keys: Iterable[DemoKey],
        online_capacity: int,
        demo_fraction: float,
        suppression_steps: int,
        minimum_sampling_weight: float,
        seed: int = 0,
    ) -> None:
        demos = tuple(demonstrations)
        keys = tuple(demonstration_keys)
        if len(keys) != len(demos):
            raise ValueError("Every demonstration requires one position-action key.")
        if suppression_steps <= 0:
            raise ValueError("suppression_steps must be positive.")
        if not 0.0 <= minimum_sampling_weight <= 1.0:
            raise ValueError("minimum_sampling_weight must be in [0, 1].")
        super().__init__(
            demos,
            online_capacity=online_capacity,
            demo_fraction=demo_fraction,
            seed=seed,
        )
        self._demonstration_keys = keys
        self.suppression_steps = int(suppression_steps)
        self.minimum_sampling_weight = float(minimum_sampling_weight)
        self._clock = 0
        self._suppressed_until: dict[DemoKey, int] = {}

    @property
    def suppressed_key_count(self) -> int:
        return len(self._active_suppressed_keys())

    @property
    def suppressed_transition_count(self) -> int:
        active = self._active_suppressed_keys()
        return sum(key in active for key in self._demonstration_keys)

    @property
    def suppression_rate(self) -> float:
        return self.suppressed_transition_count / max(1, self.demonstration_size)

    def _active_suppressed_keys(self) -> set[DemoKey]:
        return {
            key
            for key, expiry in self._suppressed_until.items()
            if expiry > self._clock
        }

    def observe_local_conflicts(
        self,
        position: tuple[int, int],
        action_risks: dict[int, bool],
    ) -> None:
        """Advance one step and update local action suppression."""

        self._clock += 1
        self._suppressed_until = {
            key: expiry
            for key, expiry in self._suppressed_until.items()
            if expiry > self._clock
        }
        normalized_position = (int(position[0]), int(position[1]))
        for action, is_risky in action_risks.items():
            key = (normalized_position, int(action))
            if is_risky:
                self._suppressed_until[key] = self._clock + self.suppression_steps
            else:
                self._suppressed_until.pop(key, None)

    def _sample_demonstrations(self, demo_count: int) -> list[Transition]:
        active = self._active_suppressed_keys()
        preferred = [
            index
            for index, key in enumerate(self._demonstration_keys)
            if key not in active
        ]
        suppressed = [
            index
            for index, key in enumerate(self._demonstration_keys)
            if key in active
        ]
        suppressed_weight = self.minimum_sampling_weight * len(suppressed)
        total_weight = len(preferred) + suppressed_weight
        expected_suppressed = (
            0.0
            if total_weight == 0.0
            else demo_count * suppressed_weight / total_weight
        )
        suppressed_count = floor(expected_suppressed)
        if self._rng.random() < expected_suppressed - suppressed_count:
            suppressed_count += 1
        suppressed_count = min(suppressed_count, len(suppressed), demo_count)
        preferred_count = demo_count - suppressed_count
        if preferred_count > len(preferred):
            shortage = preferred_count - len(preferred)
            preferred_count -= shortage
            suppressed_count += shortage
        if suppressed_count > len(suppressed):
            shortage = suppressed_count - len(suppressed)
            suppressed_count -= shortage
            preferred_count += shortage
        indices = self._rng.sample(preferred, preferred_count)
        indices.extend(self._rng.sample(suppressed, suppressed_count))
        return [self._demonstrations[index] for index in indices]

    def sample(self, batch_size: int) -> list[Transition]:
        demo_count, online_count = self.sample_counts(batch_size)
        if not self.can_sample(batch_size):
            raise ValueError(
                "Insufficient replay data for requested local-conflict batch: "
                f"demo={self.demonstration_size}/{demo_count}, "
                f"online={self.online_size}/{online_count}."
            )
        batch = self._sample_demonstrations(demo_count)
        if online_count:
            batch.extend(self.online.sample(online_count))
        self._rng.shuffle(batch)
        return batch


class LocalCounterexampleDemoReplay(LocalConflictDemoReplay):
    """Locally suppress risky demos and oversample discovered safe responses.

    Counterexamples remain in the ordinary online ring buffer and are also kept
    in a small dedicated ring.  Sampling uses a fixed share of the online part
    of a batch for that dedicated ring whenever enough examples are available.
    The demonstration count itself remains unchanged.
    """

    def __init__(
        self,
        demonstrations: Iterable[Transition],
        demonstration_keys: Iterable[DemoKey],
        online_capacity: int,
        demo_fraction: float,
        suppression_steps: int,
        minimum_sampling_weight: float,
        counterexample_capacity: int,
        counterexample_fraction: float,
        seed: int = 0,
    ) -> None:
        if counterexample_capacity <= 0:
            raise ValueError("counterexample_capacity must be positive.")
        if not 0.0 <= counterexample_fraction <= 1.0:
            raise ValueError("counterexample_fraction must be in [0, 1].")
        super().__init__(
            demonstrations,
            demonstration_keys,
            online_capacity=online_capacity,
            demo_fraction=demo_fraction,
            suppression_steps=suppression_steps,
            minimum_sampling_weight=minimum_sampling_weight,
            seed=seed,
        )
        self.counterexamples = UniformReplayBuffer(
            counterexample_capacity,
            seed=seed + 3,
        )
        self.counterexample_fraction = float(counterexample_fraction)
        self.counterexample_total_added = 0
        self.last_counterexample_sample_count = 0
        self.last_sample_size = 0

    @property
    def counterexample_size(self) -> int:
        return len(self.counterexamples)

    @property
    def counterexample_sample_fraction(self) -> float:
        return self.last_counterexample_sample_count / max(1, self.last_sample_size)

    def add_counterexample(self, transition: Transition) -> None:
        self.counterexamples.add(transition)
        self.counterexample_total_added += 1

    def sample(self, batch_size: int) -> list[Transition]:
        demo_count, online_count = self.sample_counts(batch_size)
        if not self.can_sample(batch_size):
            raise ValueError(
                "Insufficient replay data for requested local-counterexample batch: "
                f"demo={self.demonstration_size}/{demo_count}, "
                f"online={self.online_size}/{online_count}."
            )
        requested = int(online_count * self.counterexample_fraction + 0.5)
        counterexample_count = min(
            requested,
            online_count,
            self.counterexample_size,
        )
        ordinary_count = online_count - counterexample_count
        batch = self._sample_demonstrations(demo_count)
        if ordinary_count:
            batch.extend(self.online.sample(ordinary_count))
        if counterexample_count:
            batch.extend(self.counterexamples.sample(counterexample_count))
        self._rng.shuffle(batch)
        self.last_counterexample_sample_count = counterexample_count
        self.last_sample_size = batch_size
        return batch
