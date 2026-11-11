"""Reward-independent terminal filtering for the opt-in ASVC reward study."""
from __future__ import annotations

import numpy as np

from astar_d3qn.agents.astar_all_eligible_supervision import AStarAllEligibleSupervisionAgent


class RewardStudyAgent(AStarAllEligibleSupervisionAgent):
    def supervision_action(self, transition):
        # The first two scalars are goal displacement in every study variant.
        # Collision and genuine deadline failures must not receive teacher labels,
        # regardless of how shaping changes their numerical terminal reward.
        reached = bool(np.allclose(transition.next_state.scalars[:2], 0, atol=1e-7, rtol=0))
        if transition.terminated and not reached:
            return None
        action = self.static_advice(transition.state)
        if action is None:
            return None
        if not self.current_static_mask(transition.state)[action] or self.observed_risk(transition.state, action):
            return None
        return int(action)
