"""Isolated value-bound and observed-risk-filtered action-learning ablations."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from astar_d3qn.agents.astar_training_handover import AStarTrainingHandoverAgent
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.core.grid import ACTION_DELTAS

METHODS = ("unguided_raw", "unguided_bound", "advice_raw", "advice_bound", "advice_margin", "advice_bound_margin")
FLAGS = {"unguided_raw": (False, False, False),
         "unguided_bound": (False, True, False),
         "advice_raw": (True, False, False),
         "advice_bound": (True, True, False),
         "advice_margin": (True, False, True),
         "advice_bound_margin": (True, True, True)}


class AStarValueRepairAgent(AStarTrainingHandoverAgent):
    def __init__(self, config, problem, *, repair_method, lower=-6.0, upper=10.0,
                 margin=0.8, margin_weight=1.0, **advice_kwargs):
        if repair_method not in FLAGS or not np.isfinite([lower, upper, margin, margin_weight]).all():
            raise ValueError("Invalid value-repair settings.")
        if lower >= upper or margin <= 0 or margin_weight <= 0:
            raise ValueError("Invalid value bounds or teacher margin.")
        advice, bound, imitation = FLAGS[repair_method]
        super().__init__(config, problem, method="astar_risk_decay" if advice else "unguided", **advice_kwargs)
        self.repair_method = repair_method
        self.bound_enabled, self.margin_enabled = bound, imitation
        self.lower, self.upper = float(lower), float(upper)
        self.margin, self.margin_weight = float(margin), float(margin_weight)
        self.repair_bins = {}
        self._target_metrics = {}

    def select_action(self, state, epsilon, valid_actions=None):
        if epsilon == 0 or self.advice_probability() == 0:
            # A control/withdrawn curriculum does not call A* even for logging.
            action = D3QNAgent.select_action(self, state, epsilon, valid_actions)
            if epsilon > 0:
                self.advice_rng.random()
                step = self.training_action_steps + 1
                row = self.bins.setdefault((step - 1) // 10000, Counter())
                row.update(steps=1, requested=0, applied=0, risk_veto=0,
                           unavailable=0, advice_changes_action=0, observed_risk_steps=0,
                           probability_sum=0.0)
                self.training_action_steps = step
            return action
        return super().select_action(state, epsilon, valid_actions)

    def _td_targets(self, rewards, terminated, policy_next_q, target_next_q, valid_mask):
        raw = super()._td_targets(rewards, terminated, policy_next_q, target_next_q, valid_mask)
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("Nonfinite DDQN target; stopping this run.")
        targets = raw.clamp(self.lower, self.upper) if self.bound_enabled else raw
        self._target_metrics = {"raw_target_mean": raw.mean().item(), "raw_target_max": raw.max().item(),
                                "raw_target_min": raw.min().item(),
                                "raw_target_outside_fraction": ((raw < self.lower) | (raw > self.upper)).float().mean().item(),
                                "target_clip_fraction": (targets != raw).float().mean().item()}
        return targets

    @staticmethod
    def current_static_mask(state):
        row, column = state.spatial.shape[1] // 2, state.spatial.shape[2] // 2
        return [bool(state.spatial[0, row + dr, column + dc] < 0.5) for dr, dc in ACTION_DELTAS]

    def teacher_label(self, transition):
        """Label only a matching, observed-risk-clear, noncollision execution.

        No obstacle future/route lookup. Actual collision terminal rewards are
        excluded; observing that a collected action collided is normal feedback.
        """
        action = self.static_advice(transition.state)
        if action is None or action != transition.action:
            return False
        if transition.terminated and transition.reward == -1.0:
            return False
        return self.current_static_mask(transition.state)[action] and not self.observed_risk(transition.state, action)

    def imitation_weight(self):
        return self.margin_weight * max(0.0, 1.0 - self.training_action_steps / self.decay_steps) if self.margin_enabled else 0.0

    def train_batch(self, batch, **kwargs):
        weight = self.imitation_weight()
        labels = [self.teacher_label(t) for t in batch] if weight > 0 else None
        # Uniform online replay is unchanged. The supervision is derived from
        # sampled, actually executed transitions rather than adding more replay.
        kwargs.update(demonstration_mask=labels, demo_margin=self.margin if weight > 0 else 0.0,
                      demo_loss_weight=weight,
                      demo_competitor_mask=[self.current_static_mask(t.state) for t in batch] if weight > 0 else None)
        metrics = super().train_batch(batch, **kwargs)
        metrics.update(self._target_metrics)
        metrics["teacher_label_count"] = sum(labels) if labels is not None else 0
        metrics["teacher_margin_weight"] = weight
        if not np.isfinite([metrics["q_mean"], metrics["q_abs_max"], metrics["loss"]]).all():
            raise FloatingPointError("Nonfinite Q values or loss; stopping this run.")
        index = max(0, self.training_action_steps - 1) // 1000
        row = self.repair_bins.setdefault(index, Counter())
        row["updates"] += 1
        row["q_abs_max"] = max(row["q_abs_max"], metrics["q_abs_max"])
        row["raw_target_max"] = max(row.get("raw_target_max", float("-inf")), metrics["raw_target_max"])
        row["teacher_label_samples"] += metrics["teacher_label_count"]
        for key in ("q_mean", "target_mean", "raw_target_mean", "raw_target_outside_fraction",
                    "target_clip_fraction", "td_loss", "demo_margin_loss", "teacher_margin_weight"):
            row[key + "_sum"] += metrics[key]
        return metrics

    def repair_records(self):
        records = []
        for index, values in sorted(self.repair_bins.items()):
            row = {"bin_start_step": index * 1000 + 1,
                   "bin_end_step": min((index + 1) * 1000, self.training_action_steps),
                   "updates": values["updates"], "q_abs_max": values["q_abs_max"],
                   "raw_target_max": values["raw_target_max"], "teacher_label_samples": values["teacher_label_samples"]}
            row.update({key[:-4] + "_mean": value / values["updates"] for key, value in values.items() if key.endswith("_sum")})
            records.append(row)
        return records

    def training_state_dict(self):
        state = super().training_state_dict()
        state["repair_state"] = {"bins": {key: dict(value) for key, value in self.repair_bins.items()},
                                 "target_metrics": dict(self._target_metrics)}
        return state

    def load_training_state_dict(self, state):
        super().load_training_state_dict(state)
        data = state["repair_state"]
        self.repair_bins = {int(key): Counter(value) for key, value in data["bins"].items()}
        self._target_metrics = dict(data["target_metrics"])
