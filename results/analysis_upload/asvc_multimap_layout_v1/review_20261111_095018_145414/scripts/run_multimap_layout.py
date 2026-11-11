"""Frozen 90+10 run queue, at most two independent workers, complete recovery."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter,sleep
import traceback
from types import MethodType
from unittest.mock import patch
import numpy as np
import torch
import multimap_layout_common as c
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.replay.dqfd import NStepCollector
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.dqfd import new_replay, update as dqfd_update
from astar_d3qn.training.trainer import linear_epsilon_at_environment_step
from fixed_qdagger_agent import ThreeStepCollector, ObservableTeacher,coefficient

def atomic_torch(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);os.replace(temporary,path)

def timed_advice(agent,state):
    clock=perf_counter();before=len(agent._static_action_cache)
    try:return type(agent).static_advice(agent,state)
    finally:
        agent._multimap_timing['astar_lookups']+=1
        agent._multimap_timing['astar_searches']+=int(len(agent._static_action_cache)>before)
        agent._multimap_timing['astar_seconds']+=perf_counter()-clock

def td_instrument(agent,*args):
    target=type(agent)._td_targets(agent,*args)
    agent._multimap_targets.update(target_min=target.min().item(),target_max=target.max().item())
    if not hasattr(agent,'_target_metrics'):
        agent._multimap_targets.update(raw_target_min=target.min().item(),raw_target_max=target.max().item(),target_clip_fraction=0.)
    return target

def dqfd_target_instrument(agent,*args):
    target,raw=type(agent).complete_target(agent,*args)
    index=agent._multimap_target_calls;agent._multimap_target_calls+=1
    prefix='one' if index%2==0 else 'n'
    agent._multimap_targets.update({f'raw_{prefix}_target_min':raw.min().item(),f'raw_{prefix}_target_max':raw.max().item(),
        f'{prefix}_target_min':target.min().item(),f'{prefix}_target_max':target.max().item()})
    return target,raw

class Experiment:
    def __init__(self,map_seed,method,train_seed,directory,*,resume=False,unit=False,device=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.unit=unit;self.method=method;self.map_seed=map_seed;self.train_seed=train_seed
        self.agent=c.make_agent(method,map_seed,train_seed,device)
        self.training=c._training_config(c.shared_config(),train_seed)
        self.identity=dict(study=c.NAME,map_seed=map_seed,train_seed=train_seed,method=method,
            freeze_sha256='unit' if unit else c.sha(c.DATA/'freeze.json'))
        self.stream=c.training_stream(map_seed,train_seed)
        self.teacher=ObservableTeacher(c.map_data(map_seed)[0]);self.collector=None
        self.sampler=random.Random(train_seed+910300)
        self.demo_count=0
        if method=='dqfd':
            demos=c.load_demos(map_seed,'dqfd');self.demo_count=len(demos)
            self.replay=new_replay(demos,self.agent.settings,train_seed,10000);self.collector=NStepCollector(10,.99)
        elif method=='qdagger':
            self.replay=[];self.replay_position=0;self.collector=ThreeStepCollector()
        else:
            self.replay=UniformReplayBuffer(10000,seed=train_seed)
            if method=='astar_prefill_td':
                demos=c.load_demos(map_seed);self.demo_count=len(demos);self.replay.extend(demos)
        self.phase='offline' if method in ('dqfd','qdagger') else 'online'
        self.total=self.offline_updates=self.online_updates=self.episode=self.last_validation_bucket=0
        self.env=self.state=None;self.episode_return=0.;self.episode_wait=0;self.episode_loss=0.;self.episode_updates=0
        self.rows=[];self.curves=[];self.validation_details=[];self.diagnostics=[];self.bin=None
        self.offline_seconds=self.online_seconds=self.evaluation_seconds=self.checkpoint_seconds=0.
        self.last_checkpoint_step=0;self.initial_evaluated=False;self.after_offline_evaluated=False;self.test_complete=False
        self.agent._multimap_timing=Counter();self.agent._multimap_targets={};self.agent._multimap_target_calls=0
        if hasattr(self.agent,'static_advice'):self.agent.static_advice=MethodType(timed_advice,self.agent)
        if method=='dqfd':self.agent.complete_target=MethodType(dqfd_target_instrument,self.agent)
        elif method!='qdagger':self.agent._td_targets=MethodType(td_instrument,self.agent)
        self.hook=self.agent.policy_network.register_forward_hook(self.capture_q)
        if resume:self.restore(self.directory/'recovery_latest.pth')

    def capture_q(self,module,inputs,output):
        if torch.is_grad_enabled():
            self.agent._multimap_targets.update(q_min=output.detach().min().item(),q_max=output.detach().max().item())

    def snapshot(self):
        names=('phase','stream','teacher','collector','sampler','replay','total','offline_updates','online_updates','episode',
            'last_validation_bucket','state','episode_return','episode_wait','episode_loss','episode_updates',
            'rows','curves','validation_details','diagnostics','bin','offline_seconds','online_seconds','evaluation_seconds',
            'checkpoint_seconds','last_checkpoint_step','initial_evaluated','after_offline_evaluated','test_complete')
        loop={k:getattr(self,k) for k in names}
        if self.method=='qdagger':loop['replay_position']=self.replay_position
        extras={name:getattr(self.agent,name) for name in ('_static_action_cache','_multimap_timing','_multimap_targets','_multimap_target_calls') if hasattr(self.agent,name)}
        environment=None if self.env is None else dict(wrapper={k:v for k,v in vars(self.env).items() if k!='environment'},inner=vars(self.env.environment).copy())
        return dict(version=1,identity=self.identity,agent=self.agent.training_state_dict(),agent_extras=extras,loop=loop,environment=environment,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def checkpoint(self,name='recovery_latest.pth'):
        clock=perf_counter();atomic_torch(self.snapshot(),self.directory/name);self.checkpoint_seconds+=perf_counter()-clock

    def restore(self,path):
        value=torch.load(path,map_location=self.agent.device,weights_only=False)
        assert value['identity']==self.identity,'Complete recovery identity mismatch'
        self.agent.load_training_state_dict(value['agent'])
        for key,val in value['agent_extras'].items():setattr(self.agent,key,val)
        for key,val in value['loop'].items():setattr(self,key,val)
        saved=value['environment']
        if saved is None:self.env=None
        else:
            from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
            from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
            inner=object.__new__(DynamicGridNavigationEnv);vars(inner).update(saved['inner'])
            self.env=object.__new__(PathGuidanceEnvironment);vars(self.env).update(saved['wrapper']);self.env.environment=inner
        random.setstate(value['python_rng']);np.random.set_state(value['numpy_rng']);torch.set_rng_state(value['torch_rng'].cpu())
        if value['cuda_rng'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda_rng']])

    def status(self,**extra):
        c.write(dict(identity=self.identity,pid=os.getpid(),running=self.phase!='complete',complete=self.phase=='complete',
            phase=self.phase,environment_steps=self.total,online_updates=self.online_updates,offline_updates=self.offline_updates,
            recoverable_environment_steps=self.last_checkpoint_step,updated=datetime.now().isoformat(),**extra),self.directory/'status.json')

    def update(self,phase,step,batch=None,weight=0.):
        self.agent._multimap_targets={};self.agent._multimap_target_calls=0
        original_clip=torch.nn.utils.clip_grad_norm_;gradient={}
        def recorded_clip(parameters,*args,**kwargs):
            norm=original_clip(parameters,*args,**kwargs)
            if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient; preserve run without parameter changes')
            gradient['gradient_norm_before_clip']=float(norm.item());return norm
        with patch.object(torch.nn.utils,'clip_grad_norm_',recorded_clip):
            if self.method=='dqfd':
                beta=self.agent.settings['per_beta_start']
                if phase=='online':beta+=(self.agent.settings['per_beta_end']-beta)*min(1.,self.total/200000)
                metrics=dqfd_update(self.agent,self.replay,64 if not self.unit else 4,beta)
            elif self.method=='qdagger':metrics=self.agent.train_update(batch,weight)
            else:metrics=self.agent.train_batch(batch)
        metrics.update(self.agent._multimap_targets);metrics.update(gradient)
        if hasattr(self.agent,'_target_metrics'):metrics.update(self.agent._target_metrics)
        if self.method=='qdagger':
            metrics.update(raw_target_min=metrics['target_min'],raw_target_max=metrics['target_max'],target_clip_fraction=0.)
        metrics={k:float(v) for k,v in metrics.items() if np.isscalar(v)}
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite update diagnostic')
        index=(step-1)//1000
        if self.bin is None or (phase,index)!=(self.bin['phase'],self.bin['index']):
            self.flush();self.bin=dict(phase=phase,index=index,updates=0,sums={},minimum={},maximum={})
        row=self.bin;row['updates']+=1
        for key,value in metrics.items():
            row['sums'][key]=row['sums'].get(key,0.)+value
            row['minimum'][key]=min(row['minimum'].get(key,value),value)
            row['maximum'][key]=max(row['maximum'].get(key,value),value)
        return metrics

    def flush(self):
        if self.bin is None:return
        b=self.bin;row=dict(phase=b['phase'],bin_start=b['index']*1000+1,updates=b['updates'],online_step=self.total)
        row.update({k+'_mean':v/b['updates'] for k,v in b['sums'].items()})
        row.update({k+'_min':v for k,v in b['minimum'].items()});row.update({k+'_max':v for k,v in b['maximum'].items()})
        self.diagnostics.append(row);self.bin=None

    def logs(self):
        c.csv_write(self.rows,self.directory/'training.csv');c.csv_write(self.curves,self.directory/'validation_curve.csv')
        c.csv_write(self.validation_details,self.directory/'validation_details.csv');c.csv_write(self.diagnostics,self.directory/'update_diagnostics.csv')
        for name,file in [('advice_records','advice.csv'),('supervision_records','supervision.csv'),('repair_records','value_learning.csv')]:
            if hasattr(self.agent,name):c.csv_write(getattr(self.agent,name)(),self.directory/file)

    def validate(self,stage='online',scenes=None):
        scenes=scenes or c.heldout(self.map_seed)[0]
        # epsilon=0 bypasses all teacher advice. Patches make that contract executable.
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.teacher,'query',side_effect=AssertionError('Teacher used during evaluation')))
            if hasattr(self.agent,'static_advice'):stack.enter_context(patch.object(self.agent,'static_advice',side_effect=AssertionError('A* used during evaluation')))
            if self.method=='qdagger':stack.enter_context(patch.object(self.agent,'train_update',side_effect=AssertionError('Evaluation update')))
            if self.method=='dqfd':stack.enter_context(patch.object(self.agent,'train_dqfd',side_effect=AssertionError('Evaluation update')))
            summary,details,failures,seconds=c.evaluate(self.agent,self.map_seed,scenes)
        self.evaluation_seconds+=seconds
        row=dict(stage=stage,environment_steps_total=self.total,**summary)
        if stage!='before_pretraining':
            self.curves.append(row);self.validation_details.extend(dict(environment_steps_total=self.total,**r) for r in details)
        else:c.write(dict(summary=summary,details=details),self.directory/'before_pretraining50.json')
        with gzip.open(self.directory/f'validation_failures_{stage}_{self.total:06d}.json.gz','wt',encoding='utf-8') as f:json.dump(failures,f)
        self.logs();self.status(last_validation=row)
        print(f'[{self.map_seed}/{self.method}/s{self.train_seed}] {stage} step={self.total} safe={summary["safe_success_rate"]:.1%}',flush=True)
        return summary

    def offline(self,end=None):
        end=20000 if end is None else end
        records=c.load_demos(self.map_seed,'qdagger') if self.method=='qdagger' else None
        self.demo_count=len(records) if records is not None else self.demo_count
        while self.offline_updates<end:
            clock=perf_counter();batch=self.sampler.sample(records,4 if self.unit else 64) if records is not None else None
            self.update('offline',self.offline_updates+1,batch,3.);self.offline_updates+=1
            self.offline_seconds+=perf_counter()-clock
            if not self.unit and self.offline_updates%1000==0:
                self.flush();self.logs();self.checkpoint();self.status()
                print(f'[{self.map_seed}/{self.method}/s{self.train_seed}] offline={self.offline_updates}/20000',flush=True)

    def begin_online(self):
        self.flush()
        if self.method=='qdagger':self.agent.sync_target()
        self.phase='online'

    def step(self,budget=200000,learning_starts=500,batch_size=64):
        if self.env is None:
            self.env=self.stream(**c.env_kwargs());self.state=self.env.reset();self.episode+=1
            self.episode_return=self.episode_loss=0.;self.episode_wait=self.episode_updates=0
        clock=perf_counter();epsilon=linear_epsilon_at_environment_step(self.total,self.training)
        assert not self.state.spatial[4].any() and not self.state.scalars[2:].any()
        action=self.agent.select_action(self.state,epsilon,list(np.flatnonzero(self.env.action_mask(True))))
        label=self.teacher.query(self.state) if self.method=='qdagger' and coefficient(self.total+1)>0 else -1
        result=self.env.step(action);self.total+=1;end=result.done or self.total==budget
        transition=Transition(self.state,action,result.reward,result.observation,result.terminated,self.env.action_mask(True))
        if self.method=='qdagger':
            for record in self.collector.add(transition,label,self.episode,episode_end=end):
                if len(self.replay)<10000:self.replay.append(record)
                else:self.replay[self.replay_position]=record
                self.replay_position=(self.replay_position+1)%10000
        elif self.method=='dqfd':
            for record in self.collector.add(transition,episode_end=end):self.replay.add(record)
        else:self.replay.add(transition)
        if self.total>=learning_starts:
            batch=None if self.method=='dqfd' else self.sampler.sample(self.replay,batch_size) if self.method=='qdagger' else self.replay.sample(batch_size)
            metrics=self.update('online',self.total,batch,coefficient(self.total));self.online_updates+=1
            self.episode_loss+=metrics['loss'];self.episode_updates+=1
        self.state=result.observation;self.episode_return+=result.reward;self.episode_wait+=int(action==4)
        if end:
            self.rows.append(dict(episode=self.episode,scenario_id=self.env.scenario_id,environment_steps_total=self.total,
                online_updates=self.online_updates,steps=self.env.steps,episode_return=self.episode_return,wait_steps=self.episode_wait,
                safe_success=bool(result.info['reached']) and not bool(result.info['collision']),
                termination_reason=result.info['termination_reason'] if result.done else 'budget_truncation',
                terminated=bool(result.terminated),truncated=bool(result.truncated),budget_end=self.total==budget,
                epsilon=epsilon,loss=self.episode_loss/max(1,self.episode_updates),replay_size=len(self.replay)))
            if self.collector is not None:assert not self.collector.pending
            self.env=self.state=None
        self.online_seconds+=perf_counter()-clock
        return end

    def final_test(self):
        path=self.directory/'final500'
        model_sha=c.sha(self.directory/'model_final.pth');scene_sha=c.sha(c.DATA/f'map_{self.map_seed}'/'test500.json')
        registration=dict(model_sha256=model_sha,scene_sha256=scene_sha,identity=self.identity,epsilon=0,teacher=False,updates=False,static_mask=True)
        if (path/'completion.json').exists():
            saved=c.read(path/'completion.json');assert saved['registration']==registration
            self.test_complete=True;return saved['summary']
        path.mkdir(parents=True,exist_ok=True)
        summary,details,failures,seconds=c.evaluate(self.agent,self.map_seed,c.heldout(self.map_seed)[1])
        self.evaluation_seconds+=seconds
        c.csv_write(details,path/'scene_details.csv')
        with gzip.open(path/'failure_trajectories.json.gz','wt',encoding='utf-8') as f:json.dump(failures,f)
        c.write(dict(complete=True,registration=registration,summary=summary,evaluation_seconds=seconds),path/'completion.json')
        self.test_complete=True;return summary

    def execute(self):
        if not self.initial_evaluated:
            if self.phase=='offline':
                self.checkpoint('before_pretraining_state.pth');self.validate('before_pretraining')
            else:self.validate('online0')
            self.initial_evaluated=True;self.checkpoint()
        if self.phase=='offline':
            self.offline();self.begin_online()
            self.validate('after_pretraining');self.after_offline_evaluated=True
            self.checkpoint('after_pretraining_online0_state.pth');self.checkpoint()
        while self.total<200000:
            end=self.step()
            if end and (self.total//4000>self.last_validation_bucket or self.total==200000):
                self.flush();self.validate();self.last_validation_bucket=self.total//4000
                self.last_checkpoint_step=self.total;self.checkpoint()
            elif self.total-self.last_checkpoint_step>=4000:
                self.last_checkpoint_step=self.total;self.checkpoint();self.logs();self.status()
        self.flush();self.logs()
        assert self.total==200000 and self.online_updates==199501
        assert self.offline_updates==(20000 if self.method in ('dqfd','qdagger') else 0)
        if self.collector is not None:assert not self.collector.pending
        if hasattr(self.agent,'training_action_steps'):assert self.agent.training_action_steps==200000
        if not (self.directory/'model_final.pth').exists():self.agent.save_weights(self.directory/'model_final.pth')
        self.phase='test';self.checkpoint();self.status()
        test=self.final_test();self.phase='complete';self.checkpoint()
        result=dict(complete=True,identity=self.identity,environment_steps=self.total,online_updates=self.online_updates,
            offline_updates=self.offline_updates,total_updates=self.agent.update_steps,final_model_sha256=c.sha(self.directory/'model_final.pth'),
            final_validation=self.curves[-1],final500=test,offline_seconds=self.offline_seconds,online_seconds=self.online_seconds,
            evaluation_seconds=self.evaluation_seconds,checkpoint_seconds=self.checkpoint_seconds,
            demo_registration=c.read(c.DATA/f'map_{self.map_seed}'/'demonstration_registration.json') if self.method in ('astar_prefill_td','dqfd','qdagger') else None,
            online_teacher_queries=self.teacher.queries,online_teacher_searches=self.teacher.searches,online_teacher_seconds=self.teacher.seconds,
            astar_timing=dict(self.agent._multimap_timing),hardware=hardware(),max_parallel_training_workers=2,
            completed=datetime.now().isoformat())
        c.write(result,self.directory/'result.json');c.write(dict(complete=True,identity=self.identity,model_sha256=result['final_model_sha256']),self.directory/'completion.json');self.status()

def hardware():
    import platform
    return dict(platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,torch_threads=torch.get_num_threads())

def run_directory(job):return c.OUT/f'map_{job["map_seed"]}'/job['method']/f'seed_{job["train_seed"]}'
def identity(job):return dict(study=c.NAME,**job,freeze_sha256=c.sha(c.DATA/'freeze.json'))
def completion(job):
    root=run_directory(job)
    if not (root/'completion.json').exists():return False
    record=c.read(root/'completion.json');assert record['identity']==identity(job)
    assert record['model_sha256']==c.sha(root/'model_final.pth')
    result=c.read(root/'result.json');assert result['online_updates']==199501 and result['environment_steps']==200000
    assert result['identity']==identity(job)
    final=c.read(root/'final500/completion.json')
    assert final['complete'] and final['summary']['episodes']==500
    assert final['registration']['identity']==identity(job) and final['registration']['model_sha256']==record['model_sha256']
    assert final['registration']['scene_sha256']==c.sha(c.DATA/f'map_{job["map_seed"]}'/'test500.json')
    return True

@contextmanager
def run_lock(path):
    import msvcrt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError(f'Another process owns {path}')
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def worker(job,resume=False):
    torch.set_num_threads(1);c.verify_freeze();root=run_directory(job)
    if completion(job):print('Verified completed run reused',job);return
    if (root/'failure.json').exists():raise RuntimeError('Recorded failed run retained; no automatic rerun')
    with run_lock(root/'worker.lock'):
        recovery=root/'recovery_latest.pth'
        if recovery.exists() and not resume:raise FileExistsError('Use --resume for a complete training state')
        if (root/'run_manifest.json').exists():assert c.read(root/'run_manifest.json')['identity']==identity(job)
        else:c.write(dict(identity=identity(job),config=c.settings(),effective_config=c.effective_config(job['map_seed'],job['method'],job['train_seed']),hardware=hardware(),created=datetime.now().isoformat()),root/'run_manifest.json')
        experiment=Experiment(job['map_seed'],job['method'],job['train_seed'],root,resume=recovery.exists() and resume)
        try:experiment.execute()
        except BaseException as error:
            atomic_torch(experiment.snapshot(),root/'failure_state.pth')
            failure=dict(identity=identity(job),error=repr(error),traceback=traceback.format_exc(),
                numerical_failure=isinstance(error,FloatingPointError),environment_steps=experiment.total,
                offline_updates=experiment.offline_updates,created=datetime.now().isoformat())
            c.write(failure,root/'failure.json');experiment.status(failed=True,error=repr(error));raise

def progress():
    jobs=c.read(c.DATA/'queue.json')['jobs'];rows=[]
    for job in jobs:
        root=run_directory(job);status=c.read(root/'status.json') if (root/'status.json').exists() else {}
        state='failed' if (root/'failure.json').exists() else 'complete' if (root/'completion.json').exists() else 'running' if status and status.get('running') and pid_alive(status.get('pid',0)) else 'interrupted' if status else 'pending'
        rows.append(dict(**job,status=state,
            environment_steps=status.get('environment_steps',0),offline_updates=status.get('offline_updates',0),pid=status.get('pid')))
    c.csv_write(rows,c.RESULTS/'run_progress.csv')
    counts=Counter(r['status'] for r in rows)
    c.write(dict(planned=100,counts=dict(counts),jobs=rows,updated=datetime.now().isoformat()),c.RESULTS/'progress.json')
    return rows

def queue(resume=False,workers=2):
    c.verify_freeze();check=c.read(c.DATA/'preflight.json');assert check['passed'] and check['freeze_sha256']==c.sha(c.DATA/'freeze.json')
    jobs=c.read(c.DATA/'queue.json')['jobs'];active=[];remaining=iter(jobs);exhausted=False
    logroot=c.OUT/'logs';logroot.mkdir(parents=True,exist_ok=True)
    with run_lock(c.OUT/'scheduler.lock'):
        c.write(dict(pid=os.getpid(),running=True,max_workers=workers,started=datetime.now().isoformat()),c.OUT/'scheduler.json')
        try:
            while active or not exhausted:
                while len(active)<workers and not exhausted:
                    try:job=next(remaining)
                    except StopIteration:exhausted=True;break
                    root=run_directory(job)
                    if completion(job) or (root/'failure.json').exists():continue
                    if (root/'recovery_latest.pth').exists() and not resume:raise RuntimeError('Interrupted run exists: use --resume')
                    if (root/'status.json').exists():
                        status=c.read(root/'status.json')
                        if status.get('running') and status.get('pid') and pid_alive(status['pid']):raise RuntimeError(f'Worker already active: {job}')
                    filename=logroot/f'map{job["map_seed"]}_{job["method"]}_seed{job["train_seed"]}_{datetime.now():%Y%m%d_%H%M%S_%f}.log'
                    log=filename.open('wb');cmd=[sys.executable,'-X','utf8','-u',__file__,'--worker',job['method'],'--map-seed',str(job['map_seed']),'--train-seed',str(job['train_seed'])]
                    if resume:cmd.append('--resume')
                    process=subprocess.Popen(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    active.append((process,log,job));print(f'START {job} PID={process.pid}',flush=True)
                for process,log,job in active[:]:
                    if process.poll() is not None:
                        log.close();active.remove((process,log,job))
                        print(f'EXIT {job} code={process.returncode}',flush=True)
                        from analyze_multimap_layout import analyze
                        analyze()
                progress()
                if active:sleep(15)
        finally:
            for process,log,job in active:
                if process.poll() is None:process.terminate();process.wait(timeout=30)
                log.close()
            progress();c.write(dict(pid=os.getpid(),running=False,finished=datetime.now().isoformat()),c.OUT/'scheduler.json')
    from analyze_multimap_layout import analyze
    analyze()

def pid_alive(pid):
    import ctypes
    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(handle)
    return code.value==259

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',choices=c.settings()['methods']+c.settings()['ablation_methods']);parser.add_argument('--map-seed',type=int);parser.add_argument('--train-seed',type=int)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2);parser.add_argument('--status',action='store_true');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.preflight:
        import unittest
        suite=unittest.defaultTestLoader.discover(str(c.ROOT/'tests'),pattern='test_multimap_layout.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():raise SystemExit(1)
        c.verify_freeze();c.write(dict(passed=True,tests=result.testsRun,freeze_sha256=c.sha(c.DATA/'freeze.json'),hardware=hardware(),created=datetime.now().isoformat()),c.DATA/'preflight.json')
    elif args.worker:
        job=dict(map_seed=args.map_seed,train_seed=args.train_seed,method=args.worker)
        assert job in c.read(c.DATA/'queue.json')['jobs'];worker(job,args.resume)
    elif args.run:queue(args.resume,args.workers)
    else:print(json.dumps(progress(),indent=2))

if __name__=='__main__':main()
