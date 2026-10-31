"""Freeze, check, and run exactly two fixed-task QDagger adapted seeds."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime
import hashlib
import os
from pathlib import Path
import random
import shutil
from time import perf_counter
import traceback
import unittest
from unittest.mock import patch
import numpy as np
import torch
import fixed_qdagger_common as q
from fixed_qdagger_agent import ThreeStepCollector, ObservableTeacher, coefficient
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.trainer import linear_epsilon_at_environment_step
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv

c=q.c


def atomic_torch(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);os.replace(tmp,path)


def environment_state(env):
    if env is None:return None
    return dict(wrapper={k:v for k,v in vars(env).items() if k!='environment'},inner=vars(env.environment).copy())


def restore_environment(state):
    if state is None:return None
    inner=object.__new__(DynamicGridNavigationEnv);vars(inner).update(state['inner'])
    env=object.__new__(PathGuidanceEnvironment);vars(env).update(state['wrapper']);env.environment=inner
    return env


class Experiment:
    def __init__(self,seed,directory,*,resume=False,unit=False,device=None):
        self.seed=seed;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.config,self.shared,self.inputs=q.configure();self.agent=q.make_agent(seed,device)
        self.unit=unit;self.identity=dict(study=q.NAME,seed=seed,freeze_sha256=c.sha(q.DATA/'freeze.json') if not unit else 'recovery_unit')
        self.factory=c.legacy.runtime.make_factory(self.shared,*self.inputs,method='unguided',seed=seed)
        self.factory.reset_schedule();self.teacher=ObservableTeacher(self.inputs[0]);self.collector=ThreeStepCollector()
        self.training=c.legacy.effective_training(self.shared,seed,'formal')
        self.phase='offline';self.offline_updates=0;self.online_updates=0;self.total=0;self.episode=0
        self.replay=[];self.replay_position=0;self.sampler=random.Random(seed+910300)
        self.env=None;self.state=None;self.episode_return=0.;self.episode_wait=0;self.episode_loss=0.;self.episode_updates=0
        self.training_rows=[];self.curves=[];self.details=[];self.logs=[];self.target_samples=[];self.initial_evaluation=None
        self.offline_seconds=0.;self.online_seconds=0.;self.evaluation_seconds=0.;self.bin=None
        self.last_checkpoint_step=0;self.last_validation_bucket=0;self.last_checkpoint_offline=0
        if resume:self.restore(self.directory/'recovery_latest.pth')

    def snapshot(self):
        names=('phase','offline_updates','online_updates','total','episode','replay','replay_position','sampler',
            'state','episode_return','episode_wait','episode_loss','episode_updates','training_rows','curves','details',
            'logs','target_samples','initial_evaluation','offline_seconds','online_seconds','evaluation_seconds','bin',
            'last_checkpoint_step','last_validation_bucket','last_checkpoint_offline','collector')
        return dict(version=1,identity=self.identity,agent=self.agent.training_state_dict(),
            loop={name:getattr(self,name) for name in names},teacher=vars(self.teacher).copy(),
            factory=c.factory_state(self.factory),environment=environment_state(self.env),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def checkpoint(self,name='recovery_latest.pth'):
        atomic_torch(self.snapshot(),self.directory/name)

    def restore(self,path):
        saved=torch.load(path,map_location=self.agent.device,weights_only=False)
        assert saved['identity']==self.identity,'Complete recovery identity differs'
        self.agent.load_training_state_dict(saved['agent'])
        for name,value in saved['loop'].items():setattr(self,name,value)
        vars(self.teacher).update(saved['teacher']);c.restore_factory(self.factory,saved['factory'])
        self.env=restore_environment(saved['environment'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])

    def record_update(self,phase,step,batch,weight):
        metrics=self.agent.train_update(batch,weight)
        if self.bin is None or self.bin['phase']!=phase or self.bin['bin_index']!=(step-1)//1000:
            self.flush_bin();self.bin=dict(phase=phase,bin_index=(step-1)//1000,count=0,sums={},minimum={},maximum={},
                online_step_start=self.total,online_step_end=self.total,teacher_queries_start=self.teacher.queries)
        b=self.bin;b['count']+=1;b['online_step_end']=self.total
        for key,value in metrics.items():
            b['sums'][key]=b['sums'].get(key,0.)+value
            b['minimum'][key]=min(b['minimum'].get(key,value),value)
            b['maximum'][key]=max(b['maximum'].get(key,value),value)
        if self.agent.update_steps==1 or self.agent.update_steps%1000==0:
            # Per-record audit inputs and targets from the same pre-update target are
            # summarized above; detailed n-step return/terminal/action samples below.
            for record in batch:
                self.target_samples.append(dict(phase=phase,online_environment_step=self.total,
                    global_update=self.agent.update_steps,episode=record.episode,actual_action=record.transition.action,
                    teacher_action=record.teacher_action,n_return=record.n_return,n_length=record.n_length,
                    n_terminated=record.n_terminated,bootstrap=not record.n_terminated,
                    bootstrap_discount=self.agent.config.gamma**record.n_length if not record.n_terminated else 0.,
                    distillation_weight=weight))
        return metrics

    def flush_bin(self):
        if self.bin is None:return
        b=self.bin;n=b['count'];row={k:v for k,v in b.items() if k not in ('sums','minimum','maximum')}
        row['teacher_queries_end']=self.teacher.queries
        row.update({key+'_mean':value/n for key,value in b['sums'].items()})
        row.update({key+'_min':value for key,value in b['minimum'].items()})
        row.update({key+'_max':value for key,value in b['maximum'].items()})
        self.logs.append(row);self.bin=None

    def write_logs(self):
        c.csv_write(self.logs,self.directory/'update_diagnostics.csv');c.csv_write(self.target_samples,self.directory/'n_step_samples.csv')
        c.csv_write(self.training_rows,self.directory/'training.csv');c.csv_write(self.curves,self.directory/'validation_curve.csv')
        c.csv_write(self.details,self.directory/'validation_details.csv')

    def validate(self,stage):
        clock=perf_counter()
        with patch.object(self.teacher,'query',side_effect=AssertionError('Teacher used in evaluation')), \
             patch.object(self.agent,'train_update',side_effect=AssertionError('Update used in evaluation')):
            summary,details,failures=c.evaluate(self.agent,c.frozen50(),self.shared,self.inputs)
        if stage=='before_pretraining':
            self.initial_evaluation=summary
            c.write(dict(summary=summary,details=details,failures=failures),self.directory/'before_pretraining50.json')
        else:
            self.curves.append(dict(environment_steps_total=self.total,**summary))
            self.details.extend(dict(environment_steps_total=self.total,**r) for r in details)
        c.write(failures,self.directory/f'validation_failures_{stage}_{self.total:06d}.json')
        self.agent.save_weights(self.directory/f'model_{stage}_{self.total:06d}.pth')
        self.evaluation_seconds+=perf_counter()-clock;self.write_logs()
        print(f'[QDagger seed{self.seed}] {stage} online={self.total} offline={self.offline_updates} safe={summary["safe_success_rate"]:.1%} collision={summary["collision_rate"]:.1%} timeout={summary["timeout_rate"]:.1%}',flush=True)
        c.write(dict(running=True,complete=False,pid=os.getpid(),phase=stage,online_environment_steps=self.total,
            offline_updates=self.offline_updates,online_updates=self.online_updates,teacher_queries=self.teacher.queries,
            final_validation=summary,updated=datetime.now().isoformat()),self.directory/'status.json')

    def offline(self,records,end=None):
        budget=self.config['offline_updates'] if end is None else end
        while self.offline_updates<budget:
            clock=perf_counter();batch=self.sampler.sample(records,self.config['batch_size'])
            self.record_update('offline',self.offline_updates+1,batch,3.);self.offline_updates+=1
            self.offline_seconds+=perf_counter()-clock
            if not self.unit and self.offline_updates%1000==0:
                self.flush_bin();self.write_logs();self.last_checkpoint_offline=self.offline_updates;self.checkpoint()
                print(f'[QDagger seed{self.seed}] offline {self.offline_updates}/20000',flush=True)

    def begin_online(self):
        self.flush_bin();self.agent.sync_target();assert not self.replay and not self.collector.pending
        self.phase='online'

    def step(self,budget=None,learning_starts=None,batch_size=None):
        budget=self.config['environment_steps'] if budget is None else budget
        if self.env is None:
            self.episode+=1
            self.env=self.factory(self.inputs[0],max_steps=300,window_size=15,
                reward_config=c.legacy._reward_config(self.shared),terminate_on_collision=True)
            self.state=self.env.reset();self.episode_return=0.;self.episode_wait=0;self.episode_loss=0.;self.episode_updates=0
        clock=perf_counter();step=self.total+1
        epsilon=linear_epsilon_at_environment_step(self.total,self.training)
        valid=list(np.flatnonzero(self.env.action_mask(True)))
        action=self.agent.select_action(self.state,epsilon,valid)
        label=self.teacher.query(self.state) if coefficient(step)>0 else -1
        result=self.env.step(action);self.total=step;end=result.done or self.total==budget
        transition=Transition(self.state,action,result.reward,result.observation,result.terminated,self.env.action_mask(True))
        for record in self.collector.add(transition,label,self.episode,episode_end=end):
            if len(self.replay)<10000:self.replay.append(record)
            else:self.replay[self.replay_position]=record
            self.replay_position=(self.replay_position+1)%10000
        starts=self.config['learning_starts_online_step'] if learning_starts is None else learning_starts
        batch_size=self.config['batch_size'] if batch_size is None else batch_size
        if self.total>=starts:
            batch=self.sampler.sample(self.replay,batch_size)
            metrics=self.record_update('online',self.total,batch,coefficient(self.total));self.online_updates+=1
            self.episode_loss+=metrics['loss'];self.episode_updates+=1
        self.state=result.observation;self.episode_return+=result.reward;self.episode_wait+=int(action==4)
        if end:
            self.training_rows.append(dict(episode=self.episode,scenario_id=self.env.scenario_id,
                environment_steps_total=self.total,online_updates_total=self.online_updates,steps=self.env.steps,
                epsilon=epsilon,episode_return=self.episode_return,wait_steps=self.episode_wait,
                safe_success=bool(result.info['reached']) and not bool(result.info['collision']),
                termination_reason=result.info['termination_reason'] if result.done else 'budget_truncation',
                terminated=bool(result.terminated),truncated=bool(result.truncated),budget_end=self.total==budget,
                loss=self.episode_loss/max(1,self.episode_updates),replay_size=len(self.replay),
                demo_replay_samples=0,teacher_queries=self.teacher.queries))
            assert not self.collector.pending;self.env=None;self.state=None
        self.online_seconds+=perf_counter()-clock
        return end

    def execute(self,records):
        if self.phase=='offline':
            if not (self.directory/'before_pretraining_state.pth').exists():
                self.checkpoint('before_pretraining_state.pth');self.validate('before_pretraining');self.checkpoint()
            self.offline(records);self.begin_online()
            self.validate('after_pretraining');self.checkpoint('after_pretraining_online0_state.pth');self.checkpoint()
        while self.total<200000:
            end=self.step()
            if end and (self.total//10000>self.last_validation_bucket or self.total==200000):
                self.flush_bin();self.validate('online');self.last_validation_bucket=self.total//10000
                self.last_checkpoint_step=self.total;self.checkpoint()
                from analyze_fixed_qdagger import analyze
                analyze(plot=False)
        assert self.offline_updates==20000 and self.online_updates==199501 and self.agent.update_steps==219501
        assert not self.collector.pending and len(self.replay)==10000
        assert self.total==200000 and self.curves[-1]['environment_steps_total']==200000
        assert all(r.teacher_action==-1 for r in self.replay),'Teacher labels unexpectedly queried after exit'
        self.agent.save_weights(self.directory/'model_final.pth');self.phase='complete';self.checkpoint();self.write_logs()
        result=dict(complete=True,seed=self.seed,method='qdagger_task_adapted',environment_steps=self.total,
            offline_updates=self.offline_updates,online_updates=self.online_updates,total_updates=self.agent.update_steps,
            final_model_sha256=c.sha(self.directory/'model_final.pth'),freeze_sha256=self.identity['freeze_sha256'],
            final_validation=self.curves[-1],teacher_queries=self.teacher.queries,teacher_seconds=self.teacher.seconds,
            offline_seconds=self.offline_seconds,online_seconds=self.online_seconds,evaluation_seconds=self.evaluation_seconds,
            student_replay_only=True,teacher_exit=120000,teacher_advice=False,td_clip=False,supervision_floor=0,
            demo_transitions=7260,offline_dataset_teacher_audit_queries=7260,
            offline_distillation_sample_uses=20000*64,teacher_query_semantics='collected-state queries; offline identity audit separate; labels reused per replay update',
            completed=datetime.now().isoformat())
        c.write(result,self.directory/'result.json');c.write(dict(complete=True),self.directory/'completion.json')
        c.write(dict(running=False,**result),self.directory/'status.json')


def behavioral_fingerprint(experiment):
    saved=experiment.snapshot()
    for key in ('offline_seconds','online_seconds','evaluation_seconds'):
        saved['loop'].pop(key,None)
    saved['teacher'].pop('seconds',None)
    # Make RNG-bearing objects explicit and hash every array byte, including env history.
    def visit(item):
        if isinstance(item,torch.Tensor):return ('tensor',str(item.dtype),tuple(item.shape),item.detach().cpu().numpy().tobytes())
        if isinstance(item,np.ndarray):return ('array',str(item.dtype),item.shape,item.tobytes())
        if isinstance(item,dict):return tuple((visit(k),visit(v)) for k,v in sorted(item.items(),key=lambda kv:str(kv[0])))
        if isinstance(item,(list,tuple,deque)):return tuple(visit(x) for x in item)
        if isinstance(item,random.Random):return visit(item.getstate())
        from dataclasses import is_dataclass, fields
        if is_dataclass(item):return (type(item).__name__,visit({field.name:getattr(item,field.name) for field in fields(item)}))
        if hasattr(item,'__dict__'):return (type(item).__name__,visit(vars(item)))
        return item
    return hashlib.sha256(repr(visit(saved)).encode()).hexdigest()


def recovery_check(root,device):
    demos,_=q.load_demos()
    def offline_prefix(directory,interrupt):
        exp=Experiment(0,directory,unit=True,device=device);exp.offline(demos,end=3)
        if interrupt:exp.checkpoint();exp=Experiment(0,directory,resume=True,unit=True,device=device)
        exp.offline(demos,end=6);exp.begin_online()
        for _ in range(12):exp.step(budget=24,learning_starts=6,batch_size=4)
        assert exp.env is not None and exp.collector.pending,'Recovery should exercise mid-episode pending returns'
        if interrupt:exp.checkpoint();exp=Experiment(0,directory,resume=True,unit=True,device=device)
        for _ in range(12):exp.step(budget=24,learning_starts=6,batch_size=4)
        exp.flush_bin();return behavioral_fingerprint(exp)
    assert offline_prefix(root/'plain',False)==offline_prefix(root/'resumed',True),'Offline or mid-episode recovery differs'
    return dict(device=device,passed=True,offline_updates=6,online_steps=24,offline_and_mid_episode_restored=True)


def prepare():
    if (q.DATA/'freeze.json').exists():q.verify();print('Existing QDagger freeze verified; preserved.',flush=True);return
    config,shared,inputs=q.configure();c.ce.verify_freeze()
    attempt=q.RESULTS/'preflight_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S');attempt.mkdir(parents=True,exist_ok=False)
    try:
        suite=unittest.defaultTestLoader.discover(str(c.ROOT/'tests'),pattern='test_fixed_qdagger.py')
        with (attempt/'unit_tests.log').open('w',encoding='utf-8') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        assert result.wasSuccessful(),'QDagger preflight checks failed; see preserved unit_tests.log'
        demos,audit=q.load_demos();initial=[]
        for seed in (0,1):
            agent=q.make_agent(seed,'cpu');reference=c.legacy.runtime.make_agent(shared,seed,'cpu')
            assert c.digest(agent.training_state_dict())==c.digest(reference.training_state_dict())
            with patch.object(agent,'train_update',side_effect=AssertionError('Evaluation update')):
                summary,_,_=c.evaluate(agent,c.frozen50(),shared,inputs)
            initial.append(dict(seed=seed,initial_base_state_sha256=c.digest(agent.training_state_dict()),
                autonomous50=summary,teacher_calls=0,updates=0))
        recovery=[recovery_check(attempt/device,device) for device in (('cpu','cuda') if torch.cuda.is_available() else ('cpu',))]
        protected_paths=[c.ROOT/'data/asvc_current_method.json',c.ROOT/'data/whole_map_91701_dqfd_comparison_v1',
            c.ROOT/'data/asvc_fixed_ce_v1/freeze.json',c.ROOT/'data/asvc_fixed_ce_five_seeds_v1/freeze.json']
        baseline_audit=[]
        for method in ('margin','d3qn','dqfd'):
            for seed in (0,1):
                directory=q.baseline_directory(method,seed);record=c.read(directory/'result.json')
                assert c.read(directory/'completion.json')['complete'] and record['environment_steps']==200000
                manifest=c.read(directory/'run_manifest.json')
                for key in ('dataset','environment','agent','reward','dynamic_route_pool','guidance'):
                    assert manifest['config'][key]==shared[key],f'Baseline shared {key} differs: {method}/{seed}'
                for name in ('model_final.pth','result.json','run_manifest.json','completion.json','validation_curve.csv','validation_details.csv'):
                    protected_paths.append(directory/name)
                baseline_audit.append(dict(method=method,seed=seed,directory=directory.relative_to(c.ROOT).as_posix(),
                    model_sha256=c.sha(directory/'model_final.pth'),curve_sha256=c.sha(directory/'validation_curve.csv'),
                    original_offline_updates=20000 if method=='dqfd' else 0))
        q.DATA.mkdir(parents=True,exist_ok=True)
        source=c.file_map([c.ROOT/name for name in q.CODE]);source.update(c.ce.sources())
        for name in source:
            target=q.DATA/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(c.ROOT/name,target)
        for item in baseline_audit:
            directory=c.ROOT/item['directory'];target=q.DATA/'baseline_snapshots'/item['method']/f"seed_{item['seed']}"
            target.mkdir(parents=True,exist_ok=False)
            for name in ('result.json','run_manifest.json','completion.json','validation_curve.csv','validation_details.csv'):
                shutil.copyfile(directory/name,target/name)
        scenes=c.frozen50();c.write(dict(scenarios=scenes),q.DATA/'frozen50.json')
        c.write(audit,q.DATA/'demonstration_audit.json');c.csv_write(audit['episode_audit'],q.DATA/'demonstration_episode_audit.csv')
        input_paths=[q.DATA/'official_reference',c.ROOT/config['teacher_dataset'],
            c.ROOT/'data/whole_map_91701_dqfd_comparison_v1/demonstration_episodes.json',
            c.ROOT/shared['dataset']['validation_reference'],c.ROOT/shared['dataset']['source_manifest'],c.ROOT/shared['dataset']['route_pool_manifest']]
        checks=dict(passed=True,unit_tests=result.testsRun,recovery=recovery,initializations=initial,
            demonstration_transitions=7260,teacher_mismatches=0,episode_boundaries=107,
            online_replay_only=True,teacher_after_exit=False,baseline_audit=baseline_audit,
            new_training_runs=2,online_steps_per_seed=200000,offline_updates_per_seed=20000,
            final500_used=False,multitask_results_used=False)
        c.write(checks,attempt/'passed.json');c.write(checks,q.RESULTS/'preflight.json')
        copied=c.file_map([q.DATA/'source_snapshot',q.DATA/'baseline_snapshots',q.DATA/'frozen50.json',
            q.DATA/'demonstration_audit.json',q.DATA/'demonstration_episode_audit.csv'])
        copied={Path(name).relative_to(q.DATA.relative_to(c.ROOT)).as_posix():value for name,value in copied.items()}
        c.write(dict(created=datetime.now().isoformat(),config=config,shared_config=shared,source_sha256=source,
            input_sha256=c.file_map(input_paths),protected_sha256=c.file_map(protected_paths),
            copied_files_sha256=copied,checks=checks),q.DATA/'freeze.json')
        q.verify();print(f'QDagger PREFLIGHT PASS: {result.testsRun} checks; 7260 teacher labels; CPU/CUDA recovery; two seeds authorized.',flush=True)
    except BaseException as error:
        c.write(dict(passed=False,error=repr(error),traceback=traceback.format_exc(),formal_training_started=False),attempt/'failure.json');raise


def run_one(seed,resume):
    frozen=q.verify();directory=q.OUT/f'seed_{seed}'
    if (directory/'completion.json').exists():
        result=c.read(directory/'result.json');assert result['final_model_sha256']==c.sha(directory/'model_final.pth')
        assert result['freeze_sha256']==c.sha(q.DATA/'freeze.json');print(f'QDagger seed{seed} complete; reused.',flush=True);return
    if directory.exists() and not resume:raise FileExistsError('Incomplete QDagger run preserved; explicit --resume required')
    if resume and (directory/'failure.json').exists():
        text=str(c.read(directory/'failure.json')).lower()
        if 'nonfinite' in text or 'non-finite' in text:raise ValueError('No automatic resume of nonfinite failure')
    actual_resume=resume and (directory/'recovery_latest.pth').exists()
    if directory.exists() and not actual_resume:raise FileNotFoundError('Full recovery state missing; do not restart')
    experiment=Experiment(seed,directory,resume=actual_resume)
    if not actual_resume:
        c.write(dict(seed=seed,config=experiment.config,shared_config=experiment.shared,
            freeze_sha256=c.sha(q.DATA/'freeze.json'),initial_state_sha256=c.digest(experiment.agent.training_state_dict()),
            device=str(experiment.agent.device),torch_version=torch.__version__,pid=os.getpid(),
            teacher_advice=False,td_target_clip=False,demonstration_online_prefill=False,
            offline_updates=20000,online_budget=200000,evaluation_teacher=False),directory/'run_manifest.json')
    try:
        demos,_=q.load_demos();experiment.execute(demos);q.verify()
        from analyze_fixed_qdagger import analyze
        analyze();print(f'QDagger seed{seed} COMPLETE',flush=True)
    except BaseException as error:
        experiment.checkpoint('failure_state.pth')
        c.write(dict(complete=False,running=False,error=repr(error),traceback=traceback.format_exc(),
            online_environment_steps=experiment.total,offline_updates=experiment.offline_updates,
            last_complete_checkpoint='recovery_latest.pth',failed_state='failure_state.pth'),directory/'failure.json')
        c.write(c.read(directory/'failure.json'),directory/'status.json');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--train',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    if not args.train:prepare();return
    q.verify();q.OUT.mkdir(parents=True,exist_ok=True);lock=q.OUT/'runner.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        for seed in (0,1):
            c.write(dict(running=True,complete=False,pid=os.getpid(),active_seed=seed,seeds=[0,1],updated=datetime.now().isoformat()),q.RESULTS/'queue_status.json')
            run_one(seed,args.resume)
        c.write(dict(running=False,complete=True,seeds=[0,1],updated=datetime.now().isoformat()),q.RESULTS/'queue_status.json')
    except BaseException as error:
        c.write(dict(running=False,complete=False,error=repr(error),traceback=traceback.format_exc()),q.RESULTS/'queue_status.json');raise
    finally:lock.unlink()


if __name__=='__main__':main()
