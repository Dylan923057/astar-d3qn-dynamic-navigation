"""The pilot groups share all learning code except eligible supervision weights."""
from __future__ import annotations

from collections import Counter
import random

import numpy as np
import torch
import torch.nn.functional as F

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig, double_dqn_targets
from multitask_pilot_domain import hard_gate, static_mask


def supervision_loss(q,teachers,eligible,weights,margin,competitors):
    target=q.gather(1,teachers[:,None])
    margins=q+margin
    margins=margins.scatter(1,teachers[:,None],target)
    per_item=F.relu(margins.masked_fill(~competitors,float('-inf'))-target).amax(dim=1)
    count=eligible.sum()
    # Sum is deliberately divided by label count, not coefficient sum.
    return (per_item*weights*eligible).sum()/count.clamp_min(1),per_item


class PilotAgent(D3QNAgent):
    def __init__(self,cfg,planner,tasks,group,seed,device=None):
        values=dict(cfg['agent']);values['spatial_shape']=tuple(values['spatial_shape'])
        if device:values['device']=device
        super().__init__(D3QNConfig(seed=seed,**values))
        self.protocol=cfg;self.planner=planner;self.tasks={t['task_id']:t for t in tasks}
        self.group=group;self.action_steps=0
        self.advice_rng=random.Random(seed+cfg['training']['advice_rng_offset'])
        self.advice_stats=Counter();self.replay_stats=Counter();self.risk_bins={}

    def label(self,item):
        # The replay item's task is authoritative; no current-episode goal exists here.
        task=self.tasks[item.task_id]
        expected=(np.asarray(task['goal'])-np.asarray(item.position))/(self.planner.size-1)
        if not np.allclose(item.state.scalars[:2],expected,atol=1e-6):
            raise ValueError('Replay task/goal/position mismatch')
        teacher=self.planner.action(task['goal'],item.position)
        valid=hard_gate(item.state,teacher,self.protocol['risk']['hard_gate_radius']) and not item.collision
        return max(teacher,0),valid

    def base_weight(self,step=None):
        t=self.action_steps if step is None else step
        tr=self.protocol['training']
        return max(tr['supervision_floor'],tr['supervision_start']*max(0.,1-t/tr['supervision_decay_steps']))

    def collect_action(self,state,task,position,epsilon):
        candidates=np.flatnonzero(static_mask(state)).tolist()
        action=super().select_action(state,epsilon,candidates)
        self.action_steps+=1
        tr=self.protocol['training']
        probability=tr['advice_probability_start']*max(0.,1-self.action_steps/tr['advice_decay_steps'])
        requested=self.advice_rng.random()<probability
        teacher=self.planner.action(task['goal'],position)
        available=hard_gate(state,teacher,self.protocol['risk']['hard_gate_radius'])
        applied=requested and available
        self.advice_stats.update(steps=1,requested=int(requested),applied=int(applied),
                                 changed=int(applied and teacher!=action),risk_or_unavailable_veto=int(requested and not available),
                                 environment_label_available=int(available))
        return teacher if applied else action

    def train_batch(self,batch,force_unit_soft=False):
        cfg=self.protocol['training'];device=self.device
        spatial=torch.as_tensor(np.stack([x.state.spatial for x in batch]),dtype=torch.float32,device=device)
        scalars=torch.as_tensor(np.stack([x.state.scalars for x in batch]),dtype=torch.float32,device=device)
        next_spatial=torch.as_tensor(np.stack([x.next_state.spatial for x in batch]),dtype=torch.float32,device=device)
        next_scalars=torch.as_tensor(np.stack([x.next_state.scalars for x in batch]),dtype=torch.float32,device=device)
        actions=torch.tensor([x.action for x in batch],dtype=torch.long,device=device)
        rewards=torch.tensor([x.reward for x in batch],dtype=torch.float32,device=device)
        terminated=torch.tensor([x.terminated for x in batch],dtype=torch.float32,device=device)
        mask=torch.as_tensor(np.stack([x.next_action_mask for x in batch]),dtype=torch.bool,device=device)
        q=self.policy_network(spatial,scalars)
        with torch.no_grad():
            raw=double_dqn_targets(rewards,terminated,self.config.gamma,
                                  self.policy_network(next_spatial,next_scalars),self.target_network(next_spatial,next_scalars),mask)
            target=raw.clamp(cfg['td_target_lower'],cfg['td_target_upper'])
        td=F.smooth_l1_loss(q.gather(1,actions[:,None]).squeeze(1),target)
        labels=[self.label(x) for x in batch]
        teachers=torch.tensor([a for a,_ in labels],dtype=torch.long,device=device)
        eligible=torch.tensor([ok for _,ok in labels],dtype=torch.bool,device=device)
        risks=np.array([x.risk for x in batch],dtype=np.float64)
        soft=np.ones(len(batch)) if self.group=='C' or force_unit_soft else (1-risks)**self.protocol['risk']['power']
        coefficients=self.base_weight()*soft
        weights=torch.as_tensor(coefficients,dtype=torch.float32,device=device)
        competitors=torch.as_tensor(np.stack([static_mask(x.state) for x in batch]),dtype=torch.bool,device=device)
        sup,per_item=supervision_loss(q,teachers,eligible,weights,cfg['margin'],competitors)
        loss=td+sup
        if not bool(torch.isfinite(q).all() and torch.isfinite(raw).all() and torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite Q, TD target or loss; retain this failed run')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.config.gradient_clip_norm,error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        self.update_steps+=1
        if self.update_steps%self.config.target_sync_interval==0:self.sync_target()
        eligible_np=np.array([ok for _,ok in labels]);count=int(eligible_np.sum())
        rs=self.replay_stats
        rs.update(updates=1,sampled_draws=len(batch),eligible_draws=count,
                  effective_coefficient_sum=float(coefficients[eligible_np].sum()),
                  base_coefficient_sum=self.base_weight()*count,
                  reduced_draws=int((eligible_np & (soft<1)).sum()),collision_draws=sum(x.collision for x in batch))
        for item,risk,coef,ok in zip(batch,risks,coefficients,eligible_np):
            for key,value in item.risk_stats.items():rs['replay_risk_'+key]+=value
            if ok:
                bucket=min(int(risk*5),4)
                entry=self.risk_bins.setdefault(bucket,Counter())
                entry.update(draws=1,risk_sum=float(risk),coefficient_sum=float(coef),base_sum=self.base_weight())
        return dict(loss=float(loss.detach()),td_loss=float(td.detach()),supervision_loss=float(sup.detach()),
                    raw_margin_loss=float(per_item[eligible].mean().detach()) if count else 0.,
                    q_min=float(q.detach().min()),q_max=float(q.detach().max()),q_abs_max=float(q.detach().abs().max()),
                    raw_target_min=float(raw.min()),raw_target_max=float(raw.max()),
                    target_clip_fraction=float((raw!=target).float().mean()),gradient_norm=float(norm),
                    replay_eligible=count,replay_draws=len(batch),coefficient_sum=float(coefficients[eligible_np].sum()))

    def training_state_dict(self):
        state=super().training_state_dict()
        state['pilot']=dict(action_steps=self.action_steps,advice_rng=self.advice_rng.getstate(),
                            advice_stats=self.advice_stats,replay_stats=self.replay_stats,risk_bins=self.risk_bins)
        return state

    def load_training_state_dict(self,state):
        super().load_training_state_dict(state)
        p=state['pilot'];self.action_steps=p['action_steps'];self.advice_rng.setstate(p['advice_rng'])
        self.advice_stats=p['advice_stats'];self.replay_stats=p['replay_stats'];self.risk_bins=p['risk_bins']
