"""Earlier action-override withdrawal, with unchanged auxiliary supervision."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent


def staged_advice_probability(step, *, probability=.8, original_end=100000, tail_start=60000, end=80000):
    if (not math.isfinite(step) or step < 0 or not 0 <= probability <= 1 or
            not 0 < tail_start < end < original_end):
        raise ValueError('Invalid staged advice schedule.')
    if step <= tail_start:
        return probability * max(0., 1. - step / original_end)
    if step >= end:
        return 0.
    return probability * (1. - tail_start / original_end) * (end - step) / (end - tail_start)


class AStarStagedWithdrawalAgent(AStarValueRepairAgent):
    def __init__(self, config, problem, *, tail_start=60000, advice_end=80000, **kwargs):
        super().__init__(config, problem, repair_method='advice_bound_margin', **kwargs)
        self.advice_tail_start, self.advice_end = tail_start, advice_end
        self.planner_calls = Counter()
        self.diagnostic_planner = False
        self.td_bins = {}
        staged_advice_probability(0, probability=self.probability, original_end=self.decay_steps,
                                  tail_start=tail_start, end=advice_end)

    def advice_probability(self, step=None):
        return staged_advice_probability(self.training_action_steps + 1 if step is None else step,
            probability=self.probability, original_end=self.decay_steps,
            tail_start=self.advice_tail_start, end=self.advice_end)

    def static_advice(self, state):
        kind = 'diagnostic' if self.diagnostic_planner else 'algorithm'
        size = len(self._static_action_cache)
        action = super().static_advice(state)
        self.planner_calls[kind + '_lookups'] += 1
        self.planner_calls[kind + '_searches'] += len(self._static_action_cache) != size
        return action

    def train_batch(self, batch, **kwargs):
        metrics = super().train_batch(batch, **kwargs)
        values = np.asarray(metrics['td_errors'], dtype=float)
        row = self.td_bins.setdefault((self.training_action_steps - 1) // 1000, Counter())
        row['updates'] += 1; row['td_abs_mean_sum'] += values.mean()
        row['td_abs_p95_sum'] += np.percentile(values, 95)
        row['td_abs_max'] = max(row['td_abs_max'], values.max())
        return metrics

    def td_records(self):
        return [dict(bin_start_step=i*1000+1, bin_end_step=min((i+1)*1000,self.training_action_steps),
            updates=r['updates'], td_abs_mean=r['td_abs_mean_sum']/r['updates'],
            mean_batch_td_abs_p95=r['td_abs_p95_sum']/r['updates'], td_abs_max=r['td_abs_max'])
            for i,r in sorted(self.td_bins.items())]
