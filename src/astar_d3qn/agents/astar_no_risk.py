"""Disable only the dynamic risk predicate, with passive teacher-cost instrumentation."""
from time import perf_counter

from .astar_value_repair import AStarValueRepairAgent


class AStarNoRiskAgent(AStarValueRepairAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.astar_lookups = self.astar_searches = 0
        self.astar_seconds = 0.
        self.label_checks = self.eligible_labels = 0

    def observed_risk(self, state, action):
        # This same virtual method controls BOTH override veto and teacher_label.
        return False

    def static_advice(self, state):
        started = perf_counter()
        scale = max(1, self.problem.size-1)
        position = tuple(self.problem.goal[i]-int(round(float(state.scalars[i])*scale)) for i in (0, 1))
        self.astar_lookups += 1
        self.astar_searches += int(position not in self._static_action_cache)
        try:
            return super().static_advice(state)
        finally:
            self.astar_seconds += perf_counter()-started

    def teacher_label(self, transition):
        valid = super().teacher_label(transition)
        self.label_checks += 1
        self.eligible_labels += int(valid)
        return valid
