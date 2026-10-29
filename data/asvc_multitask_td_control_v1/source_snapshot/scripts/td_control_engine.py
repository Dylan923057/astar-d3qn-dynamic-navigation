"""The existing raw D3QN update, with only its complete TD-target hook bounded."""
from __future__ import annotations

from collections import Counter,deque
from types import SimpleNamespace
import torch

import td_control_common as c
from multitask_external_agents import ExternalD3QNAgent
from multitask_external_engine import ExternalSession
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.utils.seed import seed_everything


class TDControlAgent(ExternalD3QNAgent):
    def __init__(self,config,bounds=(-6.,10.),audit_interval=1000):
        super().__init__(config)
        self.bounds=tuple(bounds);self.audit_interval=audit_interval;self.target_audit_rows=[]
        self._audit_batch=None

    def _td_targets(self,rewards,terminated,policy_next_q,target_next_q,valid_mask):
        raw=super()._td_targets(rewards,terminated,policy_next_q,target_next_q,valid_mask)
        clipped=raw.clamp(*self.bounds)
        self.last_targets.update(clipped_target_min=float(clipped.min()),clipped_target_max=float(clipped.max()),
            target_clip_fraction=float((raw!=clipped).float().mean()))
        update=self.update_steps+1
        if self._audit_batch is not None and (update==1 or update%self.audit_interval==0):
            for i,item in enumerate(self._audit_batch):
                self.target_audit_rows.append(dict(update=update,environment_steps=self.action_steps,
                    batch_index=i,task_id=item.task_id,position=str(item.position),executed_action=item.action,
                    reward=float(rewards[i]),terminated=bool(terminated[i]),truncated=bool(item.truncated),
                    bootstrap_enabled=not bool(terminated[i]),raw_td_target=float(raw[i]),
                    clipped_td_target=float(clipped[i]),target_was_clipped=bool(raw[i]!=clipped[i])))
        return clipped

    def train_task_batch(self,batch):
        self._audit_batch=batch
        try:return super().train_task_batch(batch)
        finally:self._audit_batch=None

    def training_state_dict(self):
        return super().training_state_dict()|dict(td_control=dict(bounds=self.bounds,audit_interval=self.audit_interval,
            target_audit_rows=self.target_audit_rows))

    def load_training_state_dict(self,state):
        extra=state['td_control']
        if tuple(extra['bounds'])!=self.bounds or extra['audit_interval']!=self.audit_interval:
            raise ValueError('TD-control checkpoint settings differ')
        super().load_training_state_dict(state);self.target_audit_rows=extra['target_audit_rows']


class TDControlSession(ExternalSession):
    def __init__(self,cfg,data,planner,seed,device=None):
        # Same initialization order and seeded network construction as the raw
        # uniform branch. No DQfD object, prefill, teacher, or extra loss is built.
        seed_everything(seed)
        self.cfg,self.data,self.planner,self.group,self.seed=cfg,data,planner,c.METHOD,seed
        values=dict(cfg['agent']);values['spatial_shape']=tuple(values['spatial_shape'])
        if device:values['device']=device
        settings=c.config()
        self.agent=TDControlAgent(D3QNConfig(seed=seed,**values),
            settings['algorithm']['complete_td_target_clip'],settings['logging']['target_sample_interval_updates'])
        self.settings={};self.offline_updates=0;self.online_updates=0;self.phase='online'
        # Parent checkpoint compatibility only; uniform advance never uses an
        # n-step collector, and this empty queue never receives transitions.
        self.collector=SimpleNamespace(pending=deque())
        self.replay=UniformReplayBuffer(cfg['training']['replay_capacity'],seed)
        self.step=0;self.episode=0;self.env=None;self.task=None;self.scene=None;self.state=None
        self.episode_stats=Counter();self.environment_risk_stats=Counter()
        self.episodes=[];self.validation=[];self.validation_details=[];self.update_log=[];self.case_rows=[];self.offline_log=[]
        self.metric_accumulator=Counter();self.metric_count=0;self.sample_statistics=Counter()
        self.next_validation=cfg['training']['validation_interval'];self.next_checkpoint=cfg['training']['checkpoint_interval']

    def update(self,offline=False):
        if offline:raise ValueError('No offline update is authorized for this control')
        return super().update(offline=False)
