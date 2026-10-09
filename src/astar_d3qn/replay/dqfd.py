"""Permanent expert partition, FIFO online partition, proportional PER and episode-safe n-step records."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .transition import Transition


@dataclass(frozen=True, slots=True)
class NStepRecord:
    transition: Transition
    n_return: float
    n_state: object
    n_terminated: bool
    n_action_mask: object
    n_length: int


class NStepCollector:
    def __init__(self, n=10, gamma=.99):
        if n < 1 or not 0 <= gamma <= 1:
            raise ValueError('Invalid n-step collector.')
        self.n, self.gamma, self.pending = n, gamma, deque()

    def add(self, transition, *, episode_end=False):
        self.pending.append(transition)
        records = []
        while self.pending and (episode_end or len(self.pending) >= self.n):
            items = list(self.pending)[:self.n]
            total, length = 0., 0
            for i, item in enumerate(items):
                total += self.gamma**i * item.reward
                length += 1
                if item.terminated:
                    break
            last = items[length-1]
            records.append(NStepRecord(self.pending.popleft(), total, last.next_state,
                                       last.terminated, last.next_action_mask, length))
        return records


class DQfDReplay:
    """Sum/min segment trees avoid scanning the permanent demo partition on every update.

    Priority is |one-step TD error| + source bonus (paper's proportional PER).
    Global minimum-probability normalization keeps IS weights <=1. Sampling is
    with replacement; duplicate sampled indices receive their maximum error.
    """
    def __init__(self, demonstrations, online_capacity=10000, *, alpha=.4,
                 agent_bonus=.001, demo_bonus=1., seed=0):
        if not demonstrations or online_capacity < 1 or alpha < 0 or min(agent_bonus, demo_bonus) <= 0:
            raise ValueError('Invalid permanent-demo PER.')
        self.demo_count = len(demonstrations)
        self.capacity = self.demo_count + online_capacity
        self.online_capacity = online_capacity
        self.alpha, self.agent_bonus, self.demo_bonus = alpha, agent_bonus, demo_bonus
        self.records = list(demonstrations) + [None]*online_capacity
        self.size, self.cursor, self.max_error = self.demo_count, 0, 1.
        self.base = 1 << (self.capacity-1).bit_length()
        self.sums = np.zeros(2*self.base, dtype=np.float64)
        self.mins = np.full(2*self.base, np.inf, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        for i in range(self.demo_count):
            self._set(i, self.max_error)

    def _set(self, index, error):
        if not np.isfinite(error) or error < 0:
            raise ValueError('PER error must be finite and nonnegative.')
        bonus = self.demo_bonus if index < self.demo_count else self.agent_bonus
        node = self.base+index
        self.sums[node] = self.mins[node] = (float(error)+bonus)**self.alpha
        while node > 1:
            node //= 2
            self.sums[node] = self.sums[2*node]+self.sums[2*node+1]
            self.mins[node] = min(self.mins[2*node], self.mins[2*node+1])

    def add(self, record):
        index = self.demo_count+self.cursor
        if self.records[index] is None:
            self.size += 1
        self.records[index] = record
        self._set(index, self.max_error)
        self.cursor = (self.cursor+1) % self.online_capacity

    def sample(self, batch_size, beta):
        if batch_size < 1 or not 0 <= beta <= 1:
            raise ValueError('Invalid PER sample.')
        masses = self.rng.random(batch_size)*self.sums[1]
        nodes = np.ones(batch_size, dtype=np.int64)
        while np.any(nodes < self.base):
            left_mass = self.sums[2*nodes]
            right = masses >= left_mass
            masses -= right*left_mass
            nodes = 2*nodes+right
        indices = nodes-self.base
        # (N*p)^(-beta) / (N*p_min)^(-beta).
        weights = (self.sums[nodes]/self.mins[1])**(-beta)
        return [self.records[int(i)] for i in indices], indices, weights.astype(np.float32), indices < self.demo_count

    def update_priorities(self, indices, errors):
        if len(indices) != len(errors):
            raise ValueError('PER indices/errors must match.')
        updates = {}
        for i, e in zip(indices, errors):
            if not 0 <= int(i) < self.capacity or self.records[int(i)] is None:
                raise ValueError('Invalid active replay index.')
            if not np.isfinite(e) or e < 0:
                raise ValueError('Invalid TD error.')
            updates[int(i)] = max(updates.get(int(i), 0.), float(e))
        for i, e in updates.items():
            self.max_error = max(self.max_error, e)
            self._set(i, e)

    def __len__(self):
        return self.size
