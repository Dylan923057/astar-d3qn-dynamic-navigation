from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class Observation:
    spatial: np.ndarray
    scalars: np.ndarray

    def __post_init__(self) -> None:
        if self.spatial.ndim != 3:
            raise ValueError("spatial observation must have shape (channels, height, width).")
        if self.scalars.ndim != 1:
            raise ValueError("scalar observation must be one-dimensional.")


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
