"""Task-aware interaction loop, complete mid-episode recovery and pure evaluation."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import pickle
import random

import numpy as np
import torch

import multitask_pilot_common as c
from multitask_pilot_agent import PilotAgent
from multitask_pilot_domain import TaskTransition, make_env, observation_risk, scene_for, static_mask
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.utils.seed import seed_everything


def state_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=repr):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:h.update(pickle.dumps(x,protocol=4))
    visit(value)
    return h.hexdigest()


def global_rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


class TrainSession:
    def __init__(self,cfg,data,planner,group,seed,device=None):
        seed_everything(seed)
        self.cfg,self.data,self.planner,self.group,self.seed=cfg,data,planner,group,seed
        self.agent=PilotAgent(cfg,planner,data['pools']['train'],group,seed,device)
        self.replay=UniformReplayBuffer(cfg['training']['replay_capacity'],seed)
        self.step=0;self.episode=0;self.env=None;self.task=None;self.scene=None;self.state=None
        self.episode_stats=Counter();self.environment_risk_stats=Counter()
        self.episodes=[];self.validation=[];self.validation_details=[];self.update_log=[];self.case_rows=[]
        self.metric_accumulator=Counter();self.metric_count=0
        self.next_validation=cfg['training']['validation_interval']
        self.next_checkpoint=cfg['training']['checkpoint_interval']

    def start_episode(self):
        self.task,scene_seed=c.training_task(self.data,self.cfg,self.seed,self.episode)
        self.scene=scene_for(self.task,self.planner,self.cfg['scenes'],scene_seed,(3,4,5)[self.episode%3])
        self.env=make_env(self.task,self.scene,self.planner,self.cfg);self.state=self.env.reset()
        self.episode_stats=Counter()

    def advance(self):
        if self.env is None:self.start_episode()
        tr=self.cfg['training']
        epsilon=tr['epsilon_start']+(tr['epsilon_end']-tr['epsilon_start'])*min(1.,self.step/tr['epsilon_decay_steps'])
        old_position=self.env.position
        risk,risk_stats=observation_risk(self.state,old_position,tuple(self.task['goal']),self.planner,self.cfg['risk'])
        self.environment_risk_stats.update(risk_stats)
        action=self.agent.collect_action(self.state,self.task,old_position,epsilon)
        outcome=self.env.step(action)
        item=TaskTransition(self.state,action,outcome.reward,outcome.observation,outcome.terminated,
                            self.env.action_mask(True),self.task['task_id'],old_position,self.env.position,
                            outcome.truncated,bool(outcome.info['collision']),risk,risk_stats)
        self.replay.add(item);self.step+=1;self.state=outcome.observation
        # Collision samples remain in TD replay, even when their teacher label is rejected.
        metrics=None
        if self.step>=tr['learning_starts'] and self.replay.can_sample(tr['batch_size']):
            try:metrics=self.agent.train_batch(self.replay.sample(tr['batch_size']))
            except RuntimeError as error:
                if 'non-finite' in str(error).lower() or 'nonfinite' in str(error).lower():
                    raise FloatingPointError(str(error)) from error
                raise
            for key,value in metrics.items():
                if key in ('q_min','raw_target_min'):
                    self.metric_accumulator[key]=min(self.metric_accumulator.get(key,float('inf')),value)
                elif key in ('q_max','q_abs_max','raw_target_max','gradient_norm'):
                    self.metric_accumulator[key]=max(self.metric_accumulator.get(key,float('-inf')),value)
                else:self.metric_accumulator[key]+=value
            self.metric_count+=1
        self.episode_stats.update(steps=1,reward=outcome.reward,wait_steps=int(action==4),
                                  encounter_steps=int(outcome.info['dynamic_encounter']),
                                  conflict_steps=int(outcome.info['dynamic_conflict_opportunity']))
        if outcome.done:
            steps=self.episode_stats['steps']
            self.episodes.append(dict(episode=self.episode,environment_steps_total=self.step,task_id=self.task['task_id'],
                band=self.task['band'],length=self.task['length'],scene_seed=self.scene['scene_seed'],
                obstacle_count=len(self.scene['obstacles']),max_steps=self.task['max_steps'],
                success=int(outcome.info['reached']),collision=int(outcome.info['collision']),
                timeout=int(outcome.truncated),terminated=int(outcome.terminated),truncated=int(outcome.truncated),
                **dict(self.episode_stats),encounter_rate=self.episode_stats['encounter_steps']/steps,
                conflict_rate=self.episode_stats['conflict_steps']/steps))
            self.episode+=1;self.env=None;self.state=None
        if self.step%tr['log_interval']==0:self.flush_metrics()
        return item,metrics

    def flush_metrics(self):
        if not self.metric_count:return
        row=dict(environment_steps_total=self.step,updates=self.agent.update_steps,
                 supervision_base=self.agent.base_weight(),replay_size=len(self.replay))
        for key,value in self.metric_accumulator.items():
            row[key]=value if key in ('q_min','q_max','q_abs_max','raw_target_min','raw_target_max','gradient_norm') else value/self.metric_count
        row.update({f'cumulative_{k}':v for k,v in self.agent.replay_stats.items()})
        row.update({f'advice_{k}':v for k,v in self.agent.advice_stats.items()})
        row['replay_supervision_coverage']=self.agent.replay_stats['eligible_draws']/max(1,self.agent.replay_stats['sampled_draws'])
        row['environment_label_availability']=self.agent.advice_stats['environment_label_available']/max(1,self.agent.advice_stats['steps'])
        self.update_log.append(row);self.metric_accumulator=Counter();self.metric_count=0

    def snapshot(self,identity):
        return dict(format_version=1,identity=identity,step=self.step,episode=self.episode,
                    agent=self.agent.training_state_dict(),replay=self.replay.state_dict(),rng=global_rng_state(),
                    task=self.task,scene=self.scene,env=self.env.state_dict() if self.env else None,state=self.state,
                    episode_stats=self.episode_stats,environment_risk_stats=self.environment_risk_stats,
                    episodes=self.episodes,validation=self.validation,validation_details=self.validation_details,
                    update_log=self.update_log,case_rows=self.case_rows,metric_accumulator=self.metric_accumulator,
                    metric_count=self.metric_count,next_validation=self.next_validation,next_checkpoint=self.next_checkpoint)

    def restore(self,snapshot,identity):
        if snapshot['identity']!=identity:raise ValueError('Checkpoint identity differs')
        self.agent.load_training_state_dict(snapshot['agent']);self.replay.load_state_dict(snapshot['replay'])
        for key in ('step','episode','task','scene','state','episode_stats','environment_risk_stats','episodes',
                    'validation','validation_details','update_log','case_rows','metric_accumulator','metric_count',
                    'next_validation','next_checkpoint'):
            setattr(self,key,snapshot[key])
        if snapshot['env'] is None:self.env=None
        else:
            self.env=make_env(self.task,self.scene,self.planner,self.cfg);self.env.reset();self.env.load_state_dict(snapshot['env'])
        restore_rng(snapshot['rng'])
        if self.agent.action_steps!=self.step:raise ValueError('Action counter recovery differs')


def evaluate(session,validation_scenes,prepared_envs=None):
    """All 50 decisions use the base inference method; no planner/risk hooks."""
    tasks={t['task_id']:t for t in session.data['pools']['validation']}
    envs=prepared_envs or [(tasks[s['task_id']],make_env(tasks[s['task_id']],s,session.planner,session.cfg)) for s in validation_scenes]
    before=state_digest(session.agent.training_state_dict());rng_before=global_rng_state()
    details=[];retained=[];first_failure_retained=False
    for index,(task,env) in enumerate(envs):
        state=env.reset();trace=[];wait=0
        while True:
            position=env.position
            action=D3QNAgent.select_action(session.agent,state,0.,np.flatnonzero(env.action_mask(True)).tolist())
            trace.append((state,position,action));wait+=int(action==4)
            result=env.step(action);state=result.observation
            if result.done:break
        row=dict(environment_steps_total=session.step,scenario_index=index,task_id=task['task_id'],band=task['band'],
                 length=task['length'],success=int(result.info['reached']),collision=int(result.info['collision']),
                 timeout=int(result.truncated),steps=env.steps,wait_steps=wait,
                 encounter_steps=env.encounter_steps,conflict_steps=env.conflict_opportunity_steps,
                 scene_seed=validation_scenes[index]['scene_seed'])
        details.append(row)
        if index in (0,17,34) or (not row['success'] and not first_failure_retained):
            retained.append((row,trace))
            if not row['success']:first_failure_retained=True
    if before!=state_digest(session.agent.training_state_dict()):raise ValueError('Evaluation modified learner/optimizer state')
    if state_digest(rng_before)!=state_digest(global_rng_state()):raise ValueError('Evaluation consumed training RNG')
    steps=sum(r['steps'] for r in details)
    summary=dict(environment_steps_total=session.step,episodes=len(details),
                 success_rate=float(np.mean([r['success'] for r in details])),
                 collision_rate=float(np.mean([r['collision'] for r in details])),
                 timeout_rate=float(np.mean([r['timeout'] for r in details])),
                 encounter_rate=sum(r['encounter_steps'] for r in details)/steps,
                 conflict_opportunity_rate=sum(r['conflict_steps'] for r in details)/steps,
                 encounter_episode_rate=float(np.mean([r['encounter_steps']>0 for r in details])),
                 conflict_episode_rate=float(np.mean([r['conflict_steps']>0 for r in details])))
    for band in session.cfg['tasks']['length_bands']:
        chosen=[r for r in details if r['band']==band]
        for name in ('success','collision','timeout'):
            summary[f'{band}_{name}_rate']=float(np.mean([r[name] for r in chosen]))
    return summary,details,retained


def offline_cases(session,retained):
    """Called only AFTER all validation decisions and inference-state checks."""
    tasks={t['task_id']:t for t in session.data['pools']['validation']}
    output=[]
    for row,trace in retained:
        task=tasks[row['task_id']]
        for tick,(state,position,action) in enumerate(trace):
            teacher=session.planner.action(task['goal'],position)
            risk,stats=observation_risk(state,position,tuple(task['goal']),session.planner,session.cfg['risk'])
            eligible=hard_available=__import__('multitask_pilot_domain').hard_gate(state,teacher,session.cfg['risk']['hard_gate_radius'])
            soft=1. if session.group=='C' else (1-risk)**session.cfg['risk']['power']
            output.append(dict(environment_steps_total=session.step,task_id=task['task_id'],tick=tick,
                position=str(position),action=action,teacher_action=teacher,agrees_with_astar=int(action==teacher),
                waits=int(action==4),risk=risk,label_available=int(eligible),hypothetical_supervision_weight=session.agent.base_weight()*soft if hard_available else 0.,
                offline_only=True,**stats))
    return output
