"""A* labels on executed experience, without any teacher action override."""
from __future__ import annotations

from collections import Counter
import hashlib

from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent


class AStarSupervisionOnlyAgent(AStarValueRepairAgent):
    def __init__(self, config, problem, **kwargs):
        super().__init__(config, problem, repair_method='advice_bound_margin', **kwargs)
        self.repair_method = 'supervision_only_bound'
        # The original control select_action path advances the same exploration/advice RNGs,
        # collects only epsilon-greedy actions and never calls the planner to choose an action.
        self.method = 'unguided'
        self.coverage_bins = {}
        self.critical_observation_hashes = set()
        self.critical_positions = set()

    def train_batch(self, batch, **kwargs):
        active = self.imitation_weight() > 0
        metrics = super().train_batch(batch, **kwargs)
        row = self.coverage_bins.setdefault(max(0, self.training_action_steps - 1) // 1000, Counter())
        row['updates'] += 1
        row['sampled_transitions'] += len(batch)
        row['active_sampled_transitions'] += len(batch) if active else 0
        row['supervised_samples'] += metrics['teacher_label_count']
        for transition in batch:
            scale = self.problem.size - 1
            position = tuple(self.problem.goal[a] - round(float(transition.state.scalars[a]) * scale)
                             for a in (0, 1))
            if position in self.critical_positions:
                row['critical_position_sample_draws'] += 1
                labelled = active and self.teacher_label(transition)
                row['critical_position_supervised_draws'] += int(labelled)
                digest = hashlib.sha256(transition.state.spatial.tobytes()
                                        + transition.state.scalars.tobytes()).hexdigest()
                if digest in self.critical_observation_hashes:
                    row['exact_critical_sample_draws'] += 1
                    row['exact_critical_supervised_draws'] += int(labelled)
        if active:
            for transition in batch:
                action = self.static_advice(transition.state)
                available = action is not None and self.current_static_mask(transition.state)[action]
                matching = available and action == transition.action
                clear = matching and not self.observed_risk(transition.state, action)
                noncollision = clear and not (transition.terminated and transition.reward == -1.)
                row.update(teacher_available=int(available), executed_match=int(matching),
                           observed_risk_clear_match=int(clear), eligible_noncollision_match=int(noncollision))
        metrics['teacher_label_fraction'] = metrics['teacher_label_count'] / len(batch)
        return metrics

    def coverage_records(self):
        keys = ('updates', 'sampled_transitions', 'active_sampled_transitions', 'supervised_samples',
                'teacher_available', 'executed_match', 'observed_risk_clear_match', 'eligible_noncollision_match',
                'critical_position_sample_draws', 'critical_position_supervised_draws',
                'exact_critical_sample_draws', 'exact_critical_supervised_draws')
        return [dict(bin_start_step=index * 1000 + 1,
                     bin_end_step=min((index + 1) * 1000, self.training_action_steps),
                     **{k: row[k] for k in keys}, teacher_label_coverage=row['supervised_samples'] / row['sampled_transitions'],
                     active_teacher_label_coverage=(row['supervised_samples'] / row['active_sampled_transitions']
                                                    if row['active_sampled_transitions'] else None))
                for index, row in sorted(self.coverage_bins.items())]

    def training_state_dict(self):
        state = super().training_state_dict()
        # Empty diagnostics add no state, preserving the exact registered paired initialization SHA.
        if self.coverage_bins:
            state['repair_state']['coverage_bins'] = {k: dict(v) for k, v in self.coverage_bins.items()}
        return state

    def load_training_state_dict(self, state):
        super().load_training_state_dict(state)
        self.coverage_bins = {int(k): Counter(v) for k, v in state['repair_state'].get('coverage_bins', {}).items()}
