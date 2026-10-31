"""Independent task-adapted QDagger update; no CE-agent or advice inheritance."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch
import torch.nn.functional as F
from astar_d3qn.agents.d3qn import D3QNAgent, double_dqn_bootstrap
from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import ACTION_DELTAS, action_between


def static_mask(state):
    center = state.spatial.shape[1]//2
    return np.array([state.spatial[0,center+dr,center+dc] < .5 for dr,dc in ACTION_DELTAS],dtype=bool)


def coefficient(step):
    return 3.*max(1.-step/120000.,0.)


class ObservableTeacher:
    """Static A* plus radius-one union of three already aligned occupancy frames.

    Only cells <=2 from the present robot affect the veto. At most two unit
    robot moves separate the frames, so these cells were inside the original
    radius-seven view too. Reprojected unseen peripheral history is never used.
    No environment, route, phase, movement counter or future enters this API.
    """
    def __init__(self, problem):
        self.problem=problem;self.cache={};self.queries=0;self.searches=0;self.risk_waits=0;self.seconds=0.

    def query(self,state):
        start=perf_counter();self.queries+=1
        scale=max(1,self.problem.size-1)
        position=tuple(self.problem.goal[i]-int(round(float(state.scalars[i])*scale)) for i in (0,1))
        key=(self.problem.grid_sha256,self.problem.goal,position)
        if key not in self.cache:
            path=astar_path(position,self.problem.goal,self.problem.obstacles,self.problem.size)
            self.cache[key]=int(action_between(path[0],path[1])) if path and len(path)>1 else None
            self.searches+=1
        action=self.cache[key]
        if action is None or not static_mask(state)[action]: action=4
        else:
            center=state.spatial.shape[1]//2;dr,dc=ACTION_DELTAS[action]
            cells=[(center+dr+r,center+dc+c) for r in (-1,0,1) for c in (-1,0,1) if abs(r)+abs(c)<=1]
            risk=any(np.any(state.spatial[1:4,r,c]>.5) for r,c in cells)
            if risk:self.risk_waits+=1;action=4
        self.seconds+=perf_counter()-start
        return int(action)


@dataclass(frozen=True)
class Record:
    transition: object
    teacher_action: int
    n_return: float
    n_state: object
    n_terminated: bool
    n_action_mask: object
    n_length: int
    episode: int


class ThreeStepCollector:
    def __init__(self,gamma=.99):self.gamma=gamma;self.pending=deque()

    def add(self,transition,teacher_action,episode,*,episode_end=False):
        if self.pending:assert self.pending[0][2]==episode,'Cross-episode n-step input'
        self.pending.append((transition,teacher_action,episode));records=[]
        end=episode_end or transition.terminated
        while self.pending and (end or len(self.pending)>=3):
            items=list(self.pending)[:3];total=0.;length=0
            for i,(item,_,_) in enumerate(items):
                total+=self.gamma**i*item.reward;length+=1
                if item.terminated:break
            first,label,ep=self.pending.popleft();last=items[length-1][0]
            records.append(Record(first,label,total,last.next_state,last.terminated,last.next_action_mask,length,ep))
        return records


def three_step_targets(returns,lengths,terminated,gamma,policy_q,target_q,masks):
    return returns+torch.pow(torch.as_tensor(gamma,device=returns.device),lengths)*double_dqn_bootstrap(policy_q,target_q,masks)*(~terminated).float()


def distillation(q,labels,masks,temperature=.1):
    assert bool(masks.gather(1,labels[:,None]).all()),'Illegal teacher action'
    return F.cross_entropy((q/temperature).masked_fill(~masks,float('-inf')),labels)


class QDaggerAgent(D3QNAgent):
    def train_update(self,batch,weight):
        assert len(batch)>0 and 0<=weight<=3
        device=self.device
        def observations(items):
            return (torch.as_tensor(np.stack([x.spatial for x in items]),device=device,dtype=torch.float32),
                    torch.as_tensor(np.stack([x.scalars for x in items]),device=device,dtype=torch.float32))
        states=[r.transition.state for r in batch]
        q=self.policy_network(*observations(states))
        actions=torch.tensor([r.transition.action for r in batch],device=device,dtype=torch.long)
        chosen=q.gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            next_inputs=observations([r.n_state for r in batch])
            returns=torch.tensor([r.n_return for r in batch],device=device,dtype=torch.float32)
            lengths=torch.tensor([r.n_length for r in batch],device=device,dtype=torch.float32)
            terminal=torch.tensor([r.n_terminated for r in batch],device=device,dtype=torch.bool)
            masks=torch.as_tensor(np.stack([r.n_action_mask for r in batch]),device=device,dtype=torch.bool)
            target=three_step_targets(returns,lengths,terminal,self.config.gamma,
                self.policy_network(*next_inputs),self.target_network(*next_inputs),masks)
        td=F.smooth_l1_loss(chosen,target)
        raw=torch.zeros((),device=device)
        if weight>0:
            labels=torch.tensor([r.teacher_action for r in batch],device=device,dtype=torch.long)
            raw=distillation(q,labels,torch.as_tensor(np.stack([static_mask(s) for s in states]),device=device))
        loss=td+weight*raw
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(target).all()):
            raise FloatingPointError('nonfinite Q/target/TD/distillation; no automatic changes')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.config.gradient_clip_norm,error_if_nonfinite=True)
        self.optimizer.step();self.update_steps+=1
        if self.update_steps%self.config.target_sync_interval==0:self.sync_target()
        metrics=dict(loss=loss.item(),td_loss=td.item(),distillation_loss=raw.item(),
            weighted_distillation_loss=(weight*raw).item(),weight=weight,q_min=q.min().item(),q_max=q.max().item(),
            target_min=target.min().item(),target_max=target.max().item(),gradient_norm_before_clip=norm.item(),
            teacher_labeled_samples=len(batch) if weight>0 else 0,actual_action_td_samples=len(batch),
            n_length_mean=lengths.mean().item())
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('nonfinite update metrics')
        return metrics
