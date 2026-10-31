"""DQfD adaptation using the unchanged D3QN architecture and navigation inputs."""
from __future__ import annotations

import numpy as np
import torch

from .d3qn import D3QNAgent, double_dqn_bootstrap
from .astar_value_repair import AStarValueRepairAgent


class DQfDAgent(D3QNAgent):
    def __init__(self, config, settings, *, bound=False):
        super().__init__(config)
        self.settings, self.bound_enabled = dict(settings), bound

    def complete_target(self, returns, discounts, terminated, policy_q, target_q, mask):
        raw = returns+discounts*double_dqn_bootstrap(policy_q, target_q, mask)*(1-terminated.float())
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError('Nonfinite DQfD target.')
        return (raw.clamp(-6., 10.) if self.bound_enabled else raw), raw

    def train_dqfd(self, records, weights, demonstrations):
        cfg = self.settings
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        def values(states):
            return tensor(np.stack([s.spatial for s in states])), tensor(np.stack([s.scalars for s in states]))
        transitions = [r.transition for r in records]
        actions = tensor([t.action for t in transitions], torch.long)
        q = self.policy_network(*values([t.state for t in transitions]))
        selected = q.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            one_state = values([t.next_state for t in transitions])
            n_state = values([r.n_state for r in records])
            one, raw_one = self.complete_target(tensor([t.reward for t in transitions]),
                tensor([self.config.gamma]*len(records)), tensor([t.terminated for t in transitions]),
                self.policy_network(*one_state), self.target_network(*one_state), self._next_action_mask(transitions))
            n, raw_n = self.complete_target(tensor([r.n_return for r in records]),
                tensor([self.config.gamma**r.n_length for r in records]), tensor([r.n_terminated for r in records]),
                self.policy_network(*n_state), self.target_network(*n_state),
                tensor(np.stack([r.n_action_mask for r in records]), torch.bool))
        mask = tensor(np.stack([AStarValueRepairAgent.current_static_mask(t.state) for t in transitions]), torch.bool)
        margins = q+cfg['expert_margin']
        margins = margins.scatter(1, actions[:, None], selected[:, None]).masked_fill(~mask, -torch.inf)
        expert_loss = margins.amax(1)-selected
        demo = tensor(demonstrations)
        importance = tensor(weights)
        one_loss = ((one-selected).square()*importance).mean()
        n_loss = ((n-selected).square()*importance).mean()
        # Expert loss averaged across the full weighted batch; online entries have zero expert weight.
        margin_loss = (expert_loss*demo*importance).mean()
        l2 = sum(p.square().sum() for p in self.policy_network.parameters())
        loss = one_loss+cfg['n_step_weight']*n_loss+cfg['expert_weight']*margin_loss+cfg['l2_weight']*l2
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite DQfD loss.')
        errors = (one-selected.detach()).abs().cpu().tolist()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.config.target_sync_interval == 0:
            self.sync_target()
        return dict(loss=float(loss.item()), one_td_loss=float(one_loss.item()), n_td_loss=float(n_loss.item()),
                    expert_loss=float(margin_loss.item()), l2_loss=float(l2.item()),
                    demo_samples=int(np.sum(demonstrations)), sample_count=len(records),
                    q_abs_max=float(q.detach().abs().max().item()),
                    raw_one_target_max=float(raw_one.max().item()), raw_n_target_max=float(raw_n.max().item()),
                    one_target_clip_fraction=float((one != raw_one).float().mean().item()),
                    n_target_clip_fraction=float((n != raw_n).float().mean().item()), td_errors=errors)
