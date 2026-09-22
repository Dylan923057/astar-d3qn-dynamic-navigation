from __future__ import annotations

from collections.abc import Sequence
from math import prod

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """CNN with separate state-value and action-advantage streams."""

    def __init__(
        self,
        spatial_shape: Sequence[int],
        scalar_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        shape = tuple(int(value) for value in spatial_shape)
        if len(shape) != 3 or min(shape) <= 0:
            raise ValueError("spatial_shape must be (channels, height, width).")
        if scalar_dim < 0:
            raise ValueError("scalar_dim cannot be negative.")
        if action_dim <= 1:
            raise ValueError("action_dim must be greater than one.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        channels = shape[0]
        self.spatial_shape = shape
        self.scalar_dim = int(scalar_dim)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            encoded = self.encoder(torch.zeros(1, *shape))
        encoded_dim = prod(encoded.shape[1:]) + self.scalar_dim
        self.value_stream = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def streams(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial.ndim != 4:
            raise ValueError("Spatial input must have shape (batch, channels, height, width).")
        if scalars.ndim != 2 or scalars.shape != (spatial.shape[0], self.scalar_dim):
            raise ValueError("Scalar input must have shape (batch, scalar_dim).")
        encoded = self.encoder(spatial).flatten(start_dim=1)
        features = torch.cat((encoded, scalars), dim=1)
        return self.value_stream(features), self.advantage_stream(features)

    def forward(self, spatial: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        value, advantage = self.streams(spatial, scalars)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
