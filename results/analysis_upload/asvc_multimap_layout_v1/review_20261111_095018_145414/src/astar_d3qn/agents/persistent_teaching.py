"""Independent ASVC teaching floors; historical agents remain unchanged."""
from __future__ import annotations

import math

from astar_d3qn.agents.astar_all_eligible_supervision import AStarAllEligibleSupervisionAgent


class PersistentTeachingAgent(AStarAllEligibleSupervisionAgent):
    def __init__(self, *args, advice_floor=0.0, supervision_floor=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(advice_floor) or not 0 <= advice_floor <= self.probability:
            raise ValueError("Advice floor must lie between zero and the initial probability.")
        if not math.isfinite(supervision_floor) or not 0 <= supervision_floor <= self.margin_weight:
            raise ValueError("Supervision floor must lie between zero and the initial weight.")
        self.advice_floor = float(advice_floor)
        self.supervision_floor = float(supervision_floor)

    def advice_probability(self, step=None):
        return max(self.advice_floor, super().advice_probability(step))

    def imitation_weight(self):
        return max(self.supervision_floor, super().imitation_weight())

    def supervision_records(self):
        rows = super().supervision_records()
        for row in rows:
            # The historical logger assumes supervision ends at 100k.
            row['labels_computed'] = self.supervision_floor > 0 or row['bin_start_step'] < self.decay_steps
        return rows
