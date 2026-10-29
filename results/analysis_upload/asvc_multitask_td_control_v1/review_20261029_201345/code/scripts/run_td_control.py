"""Only two 500000-step TD-only controls. No CE, scans, tests or other training."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter,sleep
import traceback

import td_control_common as c
from run_multitask_pilot import run_lock,resources


def now():return datetime.now().astimezone().isoformat(timespec='seconds')


def prepare():
    if (c.REG/'frozen_protocol.json').exists():c.verify(True);print('Existing TD control freeze verified; preserved.');return
    settings=c.config()
    if settings['experiment']['seeds']!=[0,1] or settings['algorithm']['complete_td_target_clip']!=[-6,10]:raise ValueError('Scope/bounds differ')
    if any((c.directory(s)/'run_manifest.json').exists() for s in c.SEEDS):raise ValueError('Run exists without its expected freeze')
    upstream=c.upstream_identity();protected={};inventory=[]
    for method in c.previous.METHODS:
        for seed in c.SEEDS:
            info=c.model_info(method,seed)
            if info is None:raise ValueError('Expected reusable completed method missing')
            directory=info['directory']
            for path in directory.iterdir():
                if path.is_file() and (path.suffix in ('.csv','.json') or path.name in ('model_final.pth','checkpoint_latest.pth')):
                    protected[str(path.relative_to(c.ROOT)).replace('\\','/')]=c.sha(path)
            inventory.append(dict(method=method,seed=seed,complete=True,step=500000,model_sha256=info['model_sha256']))
    c.immutable(c.REG/'inventory_before.json',inventory)
    c.immutable(c.REG/'dynamic_replay_authorization.json',dict(authorized=True,source='Explicit user instruction 2026-10-29',
        methods=['C','D'],seeds=[0,1],validation_tasks=50,only_missing_trajectories=True,
        epsilon=0,updates=0,preserve_original_scores=True,test_pool_evaluation=False))
    c.RESULTS.mkdir(parents=True,exist_ok=True)
    check=subprocess.run([sys.executable,'-X','utf8','-u','tests/test_td_control.py'],cwd=c.ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (c.RESULTS/'preflight_checks.log').write_text(check.stdout,encoding='utf-8');print(check.stdout,flush=True)
    if check.returncode:raise ValueError('Preflight failed; no research training started')
    frozen=dict(created=now(),config=settings,upstream_identity=upstream,training_source_sha256=c.training_sources(),
        protected_upstream_results_sha256=protected,training_runs=2,online_steps_per_run=500000,
        expected_updates=499501,offline_updates=0,only_algorithm_change='Complete raw TD target clipped to [-6,10]; Q outputs unchanged.',
        replay='Uniform online 20000; no demonstrations or prefill.',test_pool_evaluations=0,extra_training_disabled=True)
    c.immutable(c.REG/'frozen_protocol.json',frozen)
    for name in frozen['training_source_sha256']:
        target=c.REG/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(c.ROOT/name,target)
    c.write(c.RESULTS/'preflight.json',dict(passed=True,freeze_sha256=c.sha(c.REG/'frozen_protocol.json'),
        same_initialization_and_inactive_clip_update_parity=True,complete_cpu_cuda_recovery=True,
        no_teacher_or_prefill=True,terminated_and_truncated_checked=True,created=now()))
    c.verify(True);print('TD-ONLY CONTROL FROZEN: seeds 0,1; 500000 steps each.',flush=True)


def logs(session,path):
    for name,values in (('training_episodes.csv',session.episodes),('validation_curve.csv',session.validation),
        ('validation_details.csv',session.validation_details),('update_log.csv',session.update_log),
        ('td_target_samples.csv',session.agent.target_audit_rows)):
        c.csv_write(path/name,values)
    c.write(path/'sample_statistics.json',dict(method=c.METHOD,online_updates=session.online_updates,offline_updates=0,
        counts=dict(session.sample_statistics),uniform_online_replay=True,demonstration_draws=0,
        online_astar_advice=0,online_teacher_labels=0,td_target_clip=True,td_target_bounds=[-6,10],q_output_clip=False))


def checkpoint(session,path,identity,name='checkpoint_latest.pth'):
    import torch
    temporary=path/(name+'.tmp');torch.save(session.snapshot(identity),temporary);temporary.replace(path/name)


def validate(session,path):
    from multitask_pilot_engine import evaluate
    scenes=c.read(c.previous.pilot.REG/'validation_scenes.json')
    if session.step==500000:
        from diagnose_multitask_pairs import network_suite,summary
        details,traces=network_suite(session.agent,session.cfg,session.data,session.planner,scenes,'dynamic')
        metrics=summary(details)|dict(environment_steps_total=session.step)
        session.validation_details.extend(dict(environment_steps_total=session.step,**r) for r in details)
        c.write(path/'final_validation_trajectories.json',traces)
    else:
        metrics,details,_=evaluate(session,scenes);session.validation_details.extend(details)
    session.validation.append(metrics);logs(session,path);session.agent.save_weights(path/f'model_step_{session.step:06d}.pth')
    print(f'[TD-only seed{session.seed}] {session.step}/500000 success={metrics["success_rate"]:.3f} collision={metrics["collision_rate"]:.3f} timeout={metrics["timeout_rate"]:.3f}',flush=True)


def train(seed,resume=False):
    import torch
    from td_control_engine import TDControlSession
    frozen=c.verify(True);cfg,data,planner=c.shared();path=c.directory(seed)
    torch.set_num_threads(1)
    identity=dict(method=c.METHOD,seed=seed,freeze_sha256=c.sha(c.REG/'frozen_protocol.json'))
    with run_lock(path/'run.lock'):
        if (path/'run_manifest.json').exists():
            if c.read(path/'run_manifest.json')['identity']!=identity:raise ValueError('Existing identity differs')
            if (path/'completion.json').exists():c.model_info(c.METHOD,seed);print('Completed run reused; no retraining.');return
            if not resume:raise ValueError('Existing incomplete run requires --resume')
            if (path/'failure.json').exists():raise ValueError('Retained failure requires review; no automatic retry')
            if not (path/'checkpoint_latest.pth').exists():raise ValueError('No complete recovery checkpoint')
        elif resume:raise ValueError('No existing run to resume')
        session=TDControlSession(cfg,data,planner,seed)
        if resume:
            saved=torch.load(path/'checkpoint_latest.pth',map_location='cpu',weights_only=False);session.restore(saved,identity);del saved
        else:
            c.immutable(path/'run_manifest.json',dict(identity=identity,config=cfg,control_config=c.config(),device=str(session.agent.device),
                torch_version=torch.__version__,created=now(),upstream_identity=frozen['upstream_identity'],
                random_initialization=True,offline_updates=0,teacher_actions=False,teacher_loss=False,risk_weighting=False,
                demonstration_prefill=False,td_target_bounds=[-6,10],q_output_clip=False,expected_online_updates=499501))
        started=perf_counter();first_step=session.step
        def status():
            c.write(path/'status.json',dict(running=True,complete=False,pid=os.getpid(),step=session.step,
                online_updates=session.online_updates,offline_updates=0,
                session_steps_per_second=(session.step-first_step)/max(.001,perf_counter()-started),updated=now()))
        try:
            status()
            if not session.validation:validate(session,path);checkpoint(session,path,identity)
            while session.step<500000:
                session.advance()
                if session.step>=session.next_validation:
                    validate(session,path);session.next_validation+=10000
                if session.step>=session.next_checkpoint:
                    session.next_checkpoint+=10000;checkpoint(session,path,identity)
                if session.step%1000==0:status()
            if session.online_updates!=499501 or session.offline_updates!=0 or session.agent.action_steps!=500000:
                raise ValueError('Fixed budget changed')
            session.flush_metrics();logs(session,path);session.agent.save_weights(path/'model_final.pth');checkpoint(session,path,identity)
            c.write(path/'result.json',dict(identity=identity,complete=True,step=session.step,updates=session.agent.update_steps,
                online_updates=session.online_updates,offline_updates=0,final_validation=session.validation[-1],
                final_model_sha256=c.sha(path/'model_final.pth'),test_pools_evaluated=False,
                completed=now(),session_seconds=perf_counter()-started))
            c.write(path/'completion.json',dict(complete=True,step=500000))
            c.write(path/'status.json',dict(running=False,complete=True,step=500000,updated=now()))
        except BaseException as error:
            failure=dict(failed=True,step=session.step,seed=seed,error=repr(error),traceback=traceback.format_exc(),
                nonfinite=isinstance(error,FloatingPointError) or 'nonfinite' in str(error).lower() or 'non-finite' in str(error).lower(),
                automatic_retry=False,created=now())
            try:logs(session,path);checkpoint(session,path,identity,f'failure_state_step_{session.step:06d}.pth')
            except BaseException as saving_error:failure['failure_state_save_error']=repr(saving_error)
            c.write(path/'failure.json',failure)
            c.write(path/'status.json',dict(running=False,complete=False,step=session.step,error=repr(error),updated=now()))
            raise


def postprocess(methods=None,seeds=None):
    args=[sys.executable,'-X','utf8','-u','scripts/diagnose_multitask_loops.py']
    if methods:args+=['--methods',*methods]
    if seeds:args+=['--seeds',*[str(s) for s in seeds]]
    subprocess.run(args,cwd=c.ROOT,check=True)
    subprocess.run([sys.executable,'-X','utf8','scripts/analyze_td_control.py'],cwd=c.ROOT,check=True)


def pipeline(resume=False,workers=1):
    c.verify(True)
    if workers not in (1,2):raise ValueError('At most two training workers')
    with run_lock(c.OUT/'pipeline.lock'):
        queue=[];active=[];failures=[];post_errors=[]
        for seed in c.SEEDS:
            path=c.directory(seed)
            if (path/'completion.json').exists():c.model_info(c.METHOD,seed);continue
            if (path/'failure.json').exists():failures.append(dict(seed=seed,error='retained failure'));continue
            existing=(path/'run_manifest.json').exists()
            if existing and not resume:raise ValueError('Incomplete run exists; use --resume, do not duplicate')
            queue.append((seed,existing))
        logroot=c.OUT/'logs';logroot.mkdir(parents=True,exist_ok=True)
        while queue or active:
            remaining=[]
            for job in active:
                code=job['process'].poll()
                if code is None:remaining.append(job);continue
                job['stream'].close();print(f'END seed{job["seed"]}: {code}',flush=True)
                if code:failures.append(dict(seed=job['seed'],returncode=code))
                elif (c.ROOT/'scripts/diagnose_multitask_loops.py').exists():
                    try:postprocess([c.METHOD],[job['seed']])
                    except BaseException as error:post_errors.append(repr(error))
            active=remaining
            while queue and len(active)<workers:
                free=resources()
                if active and free is not None and free<c.config()['experiment']['minimum_free_gib_for_second_worker']:break
                seed,restore=queue.pop(0);stream=(logroot/f'seed_{seed}.log').open('a' if restore else 'x',encoding='utf-8')
                args=[sys.executable,'-X','utf8','-u','scripts/run_td_control.py','train','--seed',str(seed)]
                if restore:args+=['--resume']
                process=subprocess.Popen(args,cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append(dict(seed=seed,process=process,stream=stream));print(f'START seed{seed} pid={process.pid}',flush=True)
            c.write(c.OUT/'pipeline_status.json',dict(running=True,complete=False,pid=os.getpid(),updated=now(),
                active=[dict(seed=j['seed'],pid=j['process'].pid) for j in active],queued=[dict(seed=s,resume=r) for s,r in queue],
                failures=failures,postprocessing_errors=post_errors,available_memory_gib=resources(),training_workers_limit=workers,
                extra_training_enabled=False,test_pools_evaluated=False))
            if active:sleep(5)
        try:postprocess()
        except BaseException as error:post_errors.append(repr(error))
        c.verify(True)
        c.write(c.OUT/'pipeline_status.json',dict(running=False,complete=not failures and not post_errors,
            training_complete=all((c.directory(s)/'completion.json').exists() for s in c.SEEDS),
            active=[],queued=[],failures=failures,postprocessing_errors=post_errors,updated=now(),
            extra_training_enabled=False,test_pools_evaluated=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','pipeline','status'))
    parser.add_argument('--seed',type=int,choices=c.SEEDS);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='pipeline':pipeline(args.resume,args.workers)
    elif args.action=='status':print((c.OUT/'pipeline_status.json').read_text(encoding='utf-8') if (c.OUT/'pipeline_status.json').exists() else 'not started')
    elif args.seed is None:parser.error('train requires --seed')
    else:train(args.seed,args.resume)
