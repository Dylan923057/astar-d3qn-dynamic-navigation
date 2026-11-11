"""Canonical Dueling Double DQN implementation."""

from .d3qn import D3QNAgent, D3QNConfig
from .networks import DuelingQNetwork

__all__ = ["D3QNAgent", "D3QNConfig", "DuelingQNetwork"]

