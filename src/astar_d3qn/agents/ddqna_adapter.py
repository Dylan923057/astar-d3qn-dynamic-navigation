"""DDQNA's action-selection rule in the existing D3QN; not a full reproduction."""
from __future__ import annotations

from collections import Counter
from time import perf_counter

from .astar_training_handover import AStarTrainingHandoverAgent
from .d3qn import D3QNAgent


class DDQNAD3QNAgent(AStarTrainingHandoverAgent):
    """Reuse the static planner/cache only, never the old override/risk rule.

    Training: epsilon random, otherwise p teacher / (1-p) greedy. Evaluation
    explicitly bypasses the teacher at epsilon=0; no evaluation RNG/cost state
    changes. Ordinary D3QN TD targets and Huber loss are inherited unchanged.
    """

    def __init__(self, config, problem, *, probability=0.5, rng_offset=730000):
        super().__init__(config, problem, method='astar_fixed', probability=probability,
                         rng_offset=rng_offset)
        self.teacher_enabled = True
        self.astar_lookups = self.astar_searches = 0
        self.astar_seconds = 0.0

    def static_advice(self, state):
        self.astar_lookups += 1
        before = len(self._static_action_cache)
        start = perf_counter()
        action = super().static_advice(state)
        self.astar_seconds += perf_counter() - start
        self.astar_searches += int(len(self._static_action_cache) > before)
        return action

    def select_action(self, state, epsilon, valid_actions=None):
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon must be in [0,1].')
        candidates = list(range(self.action_dim)) if valid_actions is None else list(valid_actions)
        if not candidates or any(not 0 <= a < self.action_dim for a in candidates):
            raise ValueError('Invalid static legal action mask.')
        if epsilon == 0 or not self.teacher_enabled:
            return D3QNAgent.select_action(self, state, epsilon, candidates)
        step = self.training_action_steps + 1
        counts = self.bins.setdefault((step-1)//10000, Counter())
        counts['steps'] += 1
        counts['random_mass_sum'] += epsilon
        counts['teacher_mass_sum'] += (1-epsilon)*self.probability
        counts['greedy_mass_sum'] += (1-epsilon)*(1-self.probability)
        if self._rng.random() < epsilon:
            counts['random_branch'] += 1
            action = self._rng.choice(candidates)
        elif self.advice_rng.random() < self.probability:
            counts['teacher_branch'] += 1
            action = self.static_advice(state)
            if action is None or action not in candidates:
                counts['teacher_unavailable'] += 1
                action = D3QNAgent.select_action(self, state, 0, candidates)
            else:
                counts['teacher_applied'] += 1
        else:
            counts['greedy_branch'] += 1
            action = D3QNAgent.select_action(self, state, 0, candidates)
        self.training_action_steps = step
        return int(action)

    def advice_records(self):
        keys = ('steps', 'random_branch', 'teacher_branch', 'greedy_branch',
                'teacher_applied', 'teacher_unavailable', 'random_mass_sum',
                'teacher_mass_sum', 'greedy_mass_sum')
        return [dict(bin_start_step=index*10000+1,
                     bin_end_step=min((index+1)*10000, self.training_action_steps),
                     teacher_probability=self.probability, **{k: counts[k] for k in keys})
                for index, counts in sorted(self.bins.items())]

    def training_state_dict(self):
        state = super().training_state_dict()
        state['ddqna_state'] = dict(teacher_enabled=self.teacher_enabled,
            cache=dict(self._static_action_cache), astar_lookups=self.astar_lookups,
            astar_searches=self.astar_searches, astar_seconds=self.astar_seconds)
        return state

    def load_training_state_dict(self, state):
        super().load_training_state_dict(state)
        values = state['ddqna_state']
        self.teacher_enabled = values['teacher_enabled']
        self._static_action_cache = dict(values['cache'])
        for key in ('astar_lookups', 'astar_searches', 'astar_seconds'):
            setattr(self, key, values[key])
