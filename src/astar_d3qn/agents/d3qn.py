from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from astar_d3qn.replay.transition import Transition
from astar_d3qn.envs.types import Observation

from .networks import DuelingQNetwork


@dataclass(frozen=True, slots=True)
class D3QNConfig:
    spatial_shape: tuple[int, int, int]
    scalar_dim: int
    action_dim: int
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_sync_interval: int = 250
    gradient_clip_norm: float = 10.0
    hidden_dim: int = 256
    device: str = "auto"
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.spatial_shape) != 3 or min(self.spatial_shape) <= 0:
            raise ValueError("spatial_shape must contain three positive values.")
        if self.scalar_dim < 0:
            raise ValueError("scalar_dim cannot be negative.")
        if self.action_dim <= 1:
            raise ValueError("action_dim must be greater than one.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.target_sync_interval <= 0:
            raise ValueError("target_sync_interval must be positive.")
        if self.gradient_clip_norm < 0.0:
            raise ValueError("gradient_clip_norm cannot be negative.")


def double_dqn_bootstrap(
    policy_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    valid_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select with the policy network and evaluate with the target network."""

    if policy_next_q.shape != target_next_q.shape or policy_next_q.ndim != 2:
        raise ValueError("Policy and target Q tensors must share a 2D shape.")
    selection_q = policy_next_q
    if valid_action_mask is not None:
        if valid_action_mask.shape != policy_next_q.shape:
            raise ValueError("valid_action_mask shape must match Q tensors.")
        mask = valid_action_mask.bool().clone()
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask[empty_rows] = True
        selection_q = selection_q.masked_fill(~mask, float("-inf"))
    selected_actions = selection_q.argmax(dim=1, keepdim=True)
    return target_next_q.gather(1, selected_actions).squeeze(1)


def double_dqn_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    policy_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    valid_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if rewards.ndim != 1 or terminated.shape != rewards.shape:
        raise ValueError("rewards and terminated must share a one-dimensional shape.")
    if policy_next_q.shape[0] != rewards.shape[0]:
        raise ValueError("Q-value batch size must match rewards.")
    bootstrap = double_dqn_bootstrap(
        policy_next_q, target_next_q, valid_action_mask
    )
    return rewards + float(gamma) * bootstrap * (1.0 - terminated.float())


class D3QNAgent:
    def __init__(self, config: D3QNConfig):
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.policy_network = DuelingQNetwork(
            config.spatial_shape,
            config.scalar_dim,
            config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.target_network = DuelingQNetwork(
            config.spatial_shape,
            config.scalar_dim,
            config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=config.learning_rate
        )
        self._rng = random.Random(config.seed)
        self.update_steps = 0
        self.sync_target()
        self.target_network.eval()

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def select_action(
        self,
        state: Observation,
        epsilon: float,
        valid_actions: tuple[int, ...] | list[int] | None = None,
    ) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        candidates = list(range(self.action_dim)) if valid_actions is None else list(valid_actions)
        if not candidates or any(not 0 <= action < self.action_dim for action in candidates):
            raise ValueError("valid_actions must contain valid action indices.")
        if epsilon > 0.0 and self._rng.random() < epsilon:
            return self._rng.choice(candidates)

        spatial = torch.as_tensor(
            state.spatial, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        scalars = torch.as_tensor(
            state.scalars, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(spatial, scalars).squeeze(0)
        mask = torch.full_like(q_values, float("-inf"))
        mask[candidates] = q_values[candidates]
        return int(mask.argmax().item())

    def train_batch(
        self,
        batch: list[Transition],
        sample_weights: Sequence[float] | None = None,
        demonstration_mask: Sequence[bool] | None = None,
        demo_margin: float = 0.0,
        demo_loss_weight: float = 0.0,
        safe_guidance_margin: float = 0.0,
        safe_guidance_loss_weight: float = 0.0,
        conflict_margin: float = 0.0,
        conflict_margin_loss_weight: float = 0.0,
    ) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot train on an empty batch.")
        if sample_weights is not None and len(sample_weights) != len(batch):
            raise ValueError("sample_weights must match the batch length.")
        if demonstration_mask is not None and len(demonstration_mask) != len(batch):
            raise ValueError("demonstration_mask must match the batch length.")
        if demo_margin < 0.0:
            raise ValueError("demo_margin cannot be negative.")
        if demo_loss_weight < 0.0:
            raise ValueError("demo_loss_weight cannot be negative.")
        if safe_guidance_margin < 0.0 or safe_guidance_loss_weight < 0.0:
            raise ValueError("Safe-guidance margin settings cannot be negative.")
        if conflict_margin < 0.0 or conflict_margin_loss_weight < 0.0:
            raise ValueError("Conflict-margin settings cannot be negative.")
        spatial_states = torch.as_tensor(
            np.stack([item.state.spatial for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        scalar_states = torch.as_tensor(
            np.stack([item.state.scalars for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_spatial_states = torch.as_tensor(
            np.stack([item.next_state.spatial for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_scalar_states = torch.as_tensor(
            np.stack([item.next_state.scalars for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        terminated = torch.as_tensor(
            [item.terminated for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        valid_mask = self._next_action_mask(batch)

        policy_q_values = self.policy_network(spatial_states, scalar_states)
        predicted_q = policy_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            policy_next_q = self.policy_network(
                next_spatial_states, next_scalar_states
            )
            target_next_q = self.target_network(
                next_spatial_states, next_scalar_states
            )
            targets = double_dqn_targets(
                rewards,
                terminated,
                self.config.gamma,
                policy_next_q,
                target_next_q,
                valid_mask,
            )

        td_errors = targets - predicted_q.detach()
        td_losses = functional.smooth_l1_loss(
            predicted_q, targets, reduction="none"
        )
        if sample_weights is not None:
            weights = torch.as_tensor(
                sample_weights, dtype=torch.float32, device=self.device
            )
            td_loss = (td_losses * weights).mean()
        else:
            td_loss = td_losses.mean()

        demo_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if demonstration_mask is not None and demo_margin > 0.0 and demo_loss_weight > 0.0:
            mask = torch.as_tensor(
                demonstration_mask, dtype=torch.bool, device=self.device
            )
            if mask.any():
                margin_values = policy_q_values + float(demo_margin)
                margin_values.scatter_(1, actions.unsqueeze(1), policy_q_values.gather(1, actions.unsqueeze(1)))
                margin_losses = functional.relu(
                    margin_values - predicted_q.unsqueeze(1)
                ).amax(dim=1)
                demo_loss = margin_losses[mask].mean()
        safe_guidance_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if safe_guidance_margin > 0.0 and safe_guidance_loss_weight > 0.0:
            guidance_mask = torch.as_tensor(
                [item.safe_demo_action is not None for item in batch],
                dtype=torch.bool,
                device=self.device,
            )
            if guidance_mask.any():
                guidance_actions = torch.as_tensor(
                    [
                        int(item.safe_demo_action)
                        if item.safe_demo_action is not None
                        else 0
                        for item in batch
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                if (guidance_actions[guidance_mask] >= self.action_dim).any():
                    raise ValueError("safe_demo_action exceeds the agent action space.")
                guided_q = policy_q_values[guidance_mask]
                guided_actions = guidance_actions[guidance_mask]
                selected_guidance_q = guided_q.gather(
                    1, guided_actions.unsqueeze(1)
                ).squeeze(1)
                margin_values = guided_q + float(safe_guidance_margin)
                margin_values.scatter_(
                    1,
                    guided_actions.unsqueeze(1),
                    selected_guidance_q.unsqueeze(1),
                )
                safe_guidance_loss = functional.relu(
                    margin_values.amax(dim=1) - selected_guidance_q
                ).mean()
        conflict_margin_loss = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        conflict_margin_batch_count = 0
        if conflict_margin > 0.0 and conflict_margin_loss_weight > 0.0:
            conflict_mask = torch.as_tensor(
                [
                    item.conflict_safe_action_mask is not None
                    and item.conflict_blocked_action_mask is not None
                    for item in batch
                ],
                dtype=torch.bool,
                device=self.device,
            )
            if conflict_mask.any():
                empty = np.zeros(self.action_dim, dtype=bool)
                safe_masks = np.stack(
                    [
                        np.asarray(item.conflict_safe_action_mask, dtype=bool)
                        if item.conflict_safe_action_mask is not None
                        else empty
                        for item in batch
                    ]
                )
                blocked_masks = np.stack(
                    [
                        np.asarray(item.conflict_blocked_action_mask, dtype=bool)
                        if item.conflict_blocked_action_mask is not None
                        else empty
                        for item in batch
                    ]
                )
                if safe_masks.shape[1:] != (self.action_dim,) or blocked_masks.shape[
                    1:
                ] != (self.action_dim,):
                    raise ValueError(
                        "Conflict-margin action masks must match the agent action space."
                    )
                safe_masks_tensor = torch.as_tensor(
                    safe_masks, dtype=torch.bool, device=self.device
                )
                blocked_masks_tensor = torch.as_tensor(
                    blocked_masks, dtype=torch.bool, device=self.device
                )
                safe_q = policy_q_values.masked_fill(
                    ~safe_masks_tensor, float("-inf")
                ).amax(dim=1)
                blocked_q = policy_q_values.masked_fill(
                    ~blocked_masks_tensor, float("-inf")
                ).amax(dim=1)
                conflict_margin_loss = functional.relu(
                    blocked_q[conflict_mask]
                    + float(conflict_margin)
                    - safe_q[conflict_mask]
                ).mean()
                conflict_margin_batch_count = int(conflict_mask.sum().item())
        loss = (
            td_loss
            + float(demo_loss_weight) * demo_loss
            + float(safe_guidance_loss_weight) * safe_guidance_loss
            + float(conflict_margin_loss_weight) * conflict_margin_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = 0.0
        if self.config.gradient_clip_norm > 0.0:
            norm = nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), self.config.gradient_clip_norm
            )
            gradient_norm = float(norm.item())
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.config.target_sync_interval == 0:
            self.sync_target()
        return {
            "loss": float(loss.item()),
            "q_mean": float(predicted_q.detach().mean().item()),
            "target_mean": float(targets.mean().item()),
            "td_abs_mean": float(td_errors.abs().mean().item()),
            "gradient_norm_before_clip": gradient_norm,
            "td_errors": td_errors.abs().detach().cpu().tolist(),
            "demo_margin_loss": float(demo_loss.detach().item()),
            "safe_guidance_margin_loss": float(
                safe_guidance_loss.detach().item()
            ),
            "safe_guidance_batch_count": sum(
                item.safe_demo_action is not None for item in batch
            ),
            "conflict_margin_loss": float(conflict_margin_loss.detach().item()),
            "conflict_margin_batch_count": conflict_margin_batch_count,
        }

    def _next_action_mask(self, batch: list[Transition]) -> torch.Tensor:
        masks = []
        for transition in batch:
            if transition.next_action_mask is None:
                masks.append(np.ones(self.action_dim, dtype=bool))
                continue
            mask = np.asarray(transition.next_action_mask, dtype=bool)
            if mask.shape != (self.action_dim,):
                raise ValueError(
                    "Transition next_action_mask does not match agent action_dim."
                )
            masks.append(mask)
        return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=self.device)

    def save_weights(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_network.state_dict(), target)

    def load_weights(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_network.load_state_dict(state)
        self.sync_target()

    def training_state_dict(self) -> dict:
        return {
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "rng_state": self._rng.getstate(),
        }

    def load_training_state_dict(self, state: dict) -> None:
        self.policy_network.load_state_dict(state["policy_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_steps = int(state["update_steps"])
        self._rng.setstate(state["rng_state"])
