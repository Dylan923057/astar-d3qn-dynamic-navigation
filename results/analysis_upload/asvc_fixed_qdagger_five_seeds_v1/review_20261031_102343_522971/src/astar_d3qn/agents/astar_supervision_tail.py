"""Change only the auxiliary action-supervision clock, keeping the original advice clock."""
from __future__ import annotations

import math

from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent


def supervision_weight(step, *, original_decay_steps=100000, tail_start_steps=80000,
                       tail_end_steps=140000):
    if (not math.isfinite(step) or step < 0 or not 0 < tail_start_steps < original_decay_steps < tail_end_steps):
        raise ValueError('Invalid independent supervision schedule or step.')
    if step <= tail_start_steps:
        # Keep the original arithmetic exactly, including floating-point rounding at 80000.
        return max(0.0, 1.0 - step / original_decay_steps)
    if step >= tail_end_steps:
        return 0.0
    height = (original_decay_steps - tail_start_steps) / original_decay_steps
    return height * (tail_end_steps - step) / (tail_end_steps - tail_start_steps)


class AStarSupervisionTailAgent(AStarValueRepairAgent):
    def __init__(self, config, problem, *, supervision_schedule, **kwargs):
        super().__init__(config, problem, repair_method='advice_bound_margin', **kwargs)
        self.supervision_schedule = dict(supervision_schedule)
        supervision_weight(0, **self.supervision_schedule)

    def imitation_weight(self):
        # Never assign self.decay_steps: inherited select_action/advice_probability stay at 100000.
        return self.margin_weight * supervision_weight(self.training_action_steps, **self.supervision_schedule)
