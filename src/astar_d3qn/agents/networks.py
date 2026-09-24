from __future__ import annotations

from collections.abc import Sequence
from math import prod

import torch
import torch.nn.functional as functional
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
        self.spatial_feature_dim = prod(encoded.shape[1:])
        encoded_dim = self.spatial_feature_dim + self.scalar_dim
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
        encoded = self.encode_spatial(spatial)
        features = torch.cat((encoded, scalars), dim=1)
        return self.value_stream(features), self.advantage_stream(features)

    def encode_spatial(self, spatial: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or tuple(spatial.shape[1:]) != self.spatial_shape:
            raise ValueError(
                "Spatial input must have shape (batch, channels, height, width)."
            )
        return self.encoder(spatial).flatten(start_dim=1)

    def forward(self, spatial: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        value, advantage = self.streams(spatial, scalars)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DynamicOccupancyPredictionHead(nn.Module):
    """Predict next local dynamic occupancy from shared features and action."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        output_shape: Sequence[int],
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        shape = tuple(int(value) for value in output_shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("output_shape must contain positive height and width.")
        if feature_dim <= 0 or action_dim <= 1 or hidden_dim <= 0:
            raise ValueError("Prediction-head dimensions must be positive.")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.output_shape = shape
        self.layers = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prod(shape)),
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("Prediction features have the wrong shape.")
        if actions.ndim != 1 or actions.shape[0] != features.shape[0]:
            raise ValueError("Prediction actions must match the feature batch.")
        if (actions < 0).any() or (actions >= self.action_dim).any():
            raise ValueError("Prediction actions exceed the action space.")
        one_hot = functional.one_hot(
            actions.long(), num_classes=self.action_dim
        ).to(dtype=features.dtype)
        logits = self.layers(torch.cat((features, one_hot), dim=1))
        return logits.reshape(features.shape[0], *self.output_shape)
