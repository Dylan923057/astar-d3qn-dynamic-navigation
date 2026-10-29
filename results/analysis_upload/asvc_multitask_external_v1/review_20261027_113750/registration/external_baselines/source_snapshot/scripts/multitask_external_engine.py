"""Shared frozen environments and exact full-state recovery for external baselines."""
from __future__ import annotations

from collections import Counter,deque

import numpy as np

import multitask_external_common as c
from multitask_external_agents import make_agent,StatefulDQfDReplay
from multitask_external_demo_reader import load_records
from multitask_pilot_engine import TrainSession
from multitask_pilot_domain import TaskTransition
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.replay.dqfd import NStepCollector
from astar_d3qn.utils.seed import seed_everything


class ExternalSession(TrainSession):
    def __init__(self,cfg,data,planner,method,seed,settings,device=None,demos=None):
        seed_everything(seed)
        self.cfg,self.data,self.planner,self.group,self.seed=cfg,data,planner,method,seed
        self.agent=make_agent(cfg,settings,method,seed,device)
        self.settings=settings;self.offline_updates=0;self.online_updates=0;self.phase='offline' if method=='dqfd' else 'online'
        self.collector=NStepCollector(settings['n_steps'],cfg['agent']['gamma'])
        if method=='dqfd':
            demos=load_records(settings) if demos is None else demos
            self.replay=StatefulDQfDReplay(demos,cfg['training']['replay_capacity'],alpha=settings['per_alpha'],
                agent_bonus=settings['per_agent_bonus'],demo_bonus=settings['per_demo_bonus'],seed=seed)
        else:self.replay=UniformReplayBuffer(cfg['training']['replay_capacity'],seed)
        self.step=0;self.episode=0;self.env=None;self.task=None;self.scene=None;self.state=None
        self.episode_stats=Counter();self.environment_risk_stats=Counter()
        self.episodes=[];self.validation=[];self.validation_details=[];self.update_log=[];self.case_rows=[];self.offline_log=[]
        self.metric_accumulator=Counter();self.metric_count=0;self.sample_statistics=Counter()
        self.next_validation=cfg['training']['validation_interval'];self.next_checkpoint=cfg['training']['checkpoint_interval']

    def update(self,offline=False):
        batch_size=self.cfg['training']['batch_size']
        if self.group=='dqfd':
            beta=self.settings['per_beta_start'] if offline else self.settings['per_beta_start']+(self.settings['per_beta_end']-self.settings['per_beta_start'])*min(1.,self.step/500000)
            records,indices,weights,demo=self.replay.sample(batch_size,beta)
            metrics=self.agent.train_logged(records,weights,demo)
            self.replay.update_priorities(indices,metrics.pop('td_errors'))
            metrics.update(raw_target_min=metrics['raw_one_target_min'],raw_target_max=max(metrics['raw_one_target_max'],metrics['raw_n_target_max']),
                           target_clip_fraction=metrics['one_target_clip_fraction'],beta=beta)
            demo_count=int(demo.sum())
        else:
            metrics=self.agent.train_task_batch(self.replay.sample(batch_size));demo_count=0
        phase='offline' if offline else 'online'
        self.sample_statistics.update({phase+'_updates':1,phase+'_sampled_draws':batch_size,phase+'_demonstration_draws':demo_count})
        for key,value in metrics.items():
            if key.endswith('_min'):
                self.metric_accumulator[key]=min(self.metric_accumulator.get(key,float('inf')),value)
            elif key.endswith('_max') or 'gradient_norm' in key:
                self.metric_accumulator[key]=max(self.metric_accumulator.get(key,float('-inf')),value)
            else:self.metric_accumulator[key]+=value
        self.metric_count+=1
        if offline:self.offline_updates+=1
        else:self.online_updates+=1
        return metrics

    def flush_metrics(self):
        if not self.metric_count:return
        row=dict(environment_steps_total=self.step,phase=self.phase,offline_updates=self.offline_updates,
                 online_updates=self.online_updates,total_updates=self.agent.update_steps,replay_size=len(self.replay),
                 bin_updates=self.metric_count)
        for key,value in self.metric_accumulator.items():
            row[key]=value if key.endswith(('_min','_max')) or 'gradient_norm' in key else value/self.metric_count
        row.update({f'cumulative_{k}':v for k,v in self.sample_statistics.items()})
        row['online_demonstration_coverage']=self.sample_statistics['online_demonstration_draws']/max(1,self.sample_statistics['online_sampled_draws'])
        (self.offline_log if self.phase=='offline' else self.update_log).append(row)
        self.metric_accumulator=Counter();self.metric_count=0

    def advance(self):
        if self.phase!='online':raise ValueError('Offline pretraining is incomplete')
        if self.env is None:self.start_episode()
        tr=self.cfg['training']
        epsilon=tr['epsilon_start']+(tr['epsilon_end']-tr['epsilon_start'])*min(1.,self.step/tr['epsilon_decay_steps'])
        position=self.env.position
        # No teacher, hard gate, risk score, or target bound in either online baseline.
        action=self.agent.collect_action(self.state,self.task,position,epsilon)
        outcome=self.env.step(action)
        item=TaskTransition(self.state,action,outcome.reward,outcome.observation,outcome.terminated,
            self.env.action_mask(True),self.task['task_id'],position,self.env.position,
            outcome.truncated,bool(outcome.info['collision']),0.,{})
        self.step+=1;self.state=outcome.observation
        if self.group=='dqfd':
            for record in self.collector.add(item,episode_end=outcome.done or self.step==tr['environment_steps']):self.replay.add(record)
        else:self.replay.add(item)
        metrics=None
        if self.step>=tr['learning_starts']:
            metrics=self.update()
        self.episode_stats.update(steps=1,reward=outcome.reward,wait_steps=int(action==4),
            encounter_steps=int(outcome.info['dynamic_encounter']),conflict_steps=int(outcome.info['dynamic_conflict_opportunity']))
        if outcome.done:
            self.episodes.append(dict(episode=self.episode,environment_steps_total=self.step,task_id=self.task['task_id'],band=self.task['band'],
                length=self.task['length'],scene_seed=self.scene['scene_seed'],obstacle_count=len(self.scene['obstacles']),max_steps=self.task['max_steps'],
                success=int(outcome.info['reached']),collision=int(outcome.info['collision']),timeout=int(outcome.truncated),
                terminated=int(outcome.terminated),truncated=int(outcome.truncated),**dict(self.episode_stats)))
            self.episode+=1;self.env=None;self.state=None
        if self.step%tr['log_interval']==0:self.flush_metrics()
        return item,metrics

    def snapshot(self,identity):
        return super().snapshot(identity)|dict(external=dict(offline_updates=self.offline_updates,online_updates=self.online_updates,
            phase=self.phase,collector_pending=list(self.collector.pending),offline_log=self.offline_log,sample_statistics=self.sample_statistics))

    def restore(self,snapshot,identity):
        super().restore(snapshot,identity)
        state=snapshot['external'];self.offline_updates=state['offline_updates'];self.online_updates=state['online_updates'];self.phase=state['phase']
        self.collector.pending=deque(state['collector_pending']);self.offline_log=state['offline_log'];self.sample_statistics=state['sample_statistics']
        if self.agent.update_steps!=self.offline_updates+self.online_updates:raise ValueError('Offline/online update recovery differs')
