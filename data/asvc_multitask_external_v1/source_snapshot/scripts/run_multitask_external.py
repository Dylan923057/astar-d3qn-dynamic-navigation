"""Only D3QN/DQfD x seed0/1; resume safely, reuse C/D, then paired diagnostics."""
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

import multitask_external_common as c
from run_multitask_pilot import run_lock,resources


def now():return datetime.now().astimezone().isoformat(timespec='seconds')


def replay_authorized():
    path=c.REG/'dynamic_replay_authorization.json'
    return path.exists() and c.read(path).get('authorized') is True


def prepare():
    if (c.REG/'frozen_protocol.json').exists():c.verify();print('Existing external freeze verified; preserved.');return
    cfg=c.config()
    if [tuple(x) for x in cfg['experiment']['allowed_runs']]!=list(c.RUNS):raise ValueError('Only the four requested baselines may start')
    if any((c.directory(m,s)/'run_manifest.json').exists() for m,s in c.RUNS):raise ValueError('Existing run without this freeze; inspect instead of overwriting')
    inventory=[]
    for directory in sorted(c.ROOT.joinpath('outputs').glob('*multitask*')):
        for path in directory.glob('*/seed_*/run_manifest.json'):
            inventory.append(dict(path=str(path.relative_to(c.ROOT)),complete=(path.parent/'completion.json').exists()))
    c.immutable(c.REG/'inventory_before.json',inventory)
    from multitask_external_demos import generate
    report=generate();print(f'Demos: {report["retained_episodes"]}/{report["attempted_episodes"]} complete successful attempts; {report["transitions"]} transitions.',flush=True)
    check=subprocess.run([sys.executable,'-X','utf8','-u','tests/test_multitask_external.py'],cwd=c.ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    c.RESULTS.mkdir(parents=True,exist_ok=True);(c.RESULTS/'preflight_checks.log').write_text(check.stdout,encoding='utf-8');print(check.stdout,flush=True)
    if check.returncode:raise ValueError('External preflight failed; no research training started')
    from diagnose_multitask_pairs import ensure_protocol
    ensure_protocol()
    reused={}
    for method in ('C','D'):
        for seed in (0,1):
            info=c.model_info(method,seed)
            if info is None:raise ValueError('Expected completed C/D model is missing')
            reused[f'{method}/seed_{seed}']=dict(model_sha256=info['model_sha256'],
                result_sha256=c.sha(info['directory']/'result.json'),curve_sha256=c.sha(info['directory']/'validation_curve.csv'),
                details_sha256=c.sha(info['directory']/'validation_details.csv'))
    artifacts=['demonstration_plan.json','demonstrations.npz','demonstration_attempts.json','demonstration_task_coverage.csv',
               'demonstration_report.json','paired_diagnostic_protocol.json','paired_reference_groups.json','inventory_before.json']
    frozen=dict(config=cfg,shared_identity=c.shared_identity(),source_sha256=c.sources(),
        artifact_sha256={name:c.sha(c.REG/name) for name in artifacts},reuse_c_d=reused,
        no_test_pool_access=True,additional_training_disabled=True,created=now(),
        budget='500000 online steps / 499501 online updates; DQfD separately has 20000 offline updates and fresh fixed-budget multi-task demonstrations.',
        objectives='Original D3QN Smooth L1 without teacher/target bounds. Original DQfD squared 1/n-step TD, permanent demo/PER/margin/L2, no bounds.',
        replay='20000 online capacity in both new methods; DQfD permanent demo partition is additional and disclosed.',
        diagnostics='Reuse all original dynamic scores; static pairs are new. Additional dynamic policy replay requires a separate explicit authorization record.')
    c.immutable(c.REG/'frozen_protocol.json',frozen)
    for name in frozen['source_sha256']:
        target=c.REG/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(c.ROOT/name,target)
    c.write(c.RESULTS/'preflight.json',dict(passed=True,source_sha256=c.sources(),freeze_sha256=c.sha(c.REG/'frozen_protocol.json'),
        fixed_demo_budget=True,test_pool_evaluations=0,formal_ablations_started=0,created=now()))
    c.verify();print('EXTERNAL BASELINES FROZEN: d3qn/dqfd x seed0/1 only.',flush=True)


def checkpoint(session,path,identity):
    import torch
    temporary=path/'checkpoint_latest.pth.tmp';torch.save(session.snapshot(identity),temporary);temporary.replace(path/'checkpoint_latest.pth')


def logs(session,path):
    c.csv_write(path/'training_episodes.csv',session.episodes);c.csv_write(path/'validation_curve.csv',session.validation)
    c.csv_write(path/'validation_details.csv',session.validation_details);c.csv_write(path/'update_log.csv',session.update_log)
    c.csv_write(path/'offline_update_log.csv',session.offline_log)
    c.write(path/'sample_statistics.json',dict(method=session.group,offline_updates=session.offline_updates,online_updates=session.online_updates,
        counts=dict(session.sample_statistics),online_astar_advice=0,online_teacher_labels=0,td_target_clip=False,
        expert_supervision='Permanent demonstrations only' if session.group=='dqfd' else 'none'))


def validate(session,path):
    from multitask_pilot_engine import evaluate
    scenes=c.read(c.pilot.REG/'validation_scenes.json')
    if session.step==500000:
        from diagnose_multitask_pairs import network_suite,summary
        details,traces=network_suite(session.agent,session.cfg,session.data,session.planner,scenes,'dynamic')
        metrics=summary(details);metrics['environment_steps_total']=session.step
        session.validation_details.extend(dict(environment_steps_total=session.step,**r) for r in details)
        c.write(path/'final_validation_trajectories.json',traces)
    else:
        metrics,details,_=evaluate(session,scenes)
        session.validation_details.extend(details)
    session.validation.append(metrics);logs(session,path)
    session.agent.save_weights(path/f'model_step_{session.step:06d}.pth')
    print(f'[{session.group} seed{session.seed}] online {session.step}/500000 success={metrics["success_rate"]:.3f} collision={metrics["collision_rate"]:.3f} timeout={metrics["timeout_rate"]:.3f}',flush=True)


def train(method,seed,resume=False):
    import torch
    from multitask_external_engine import ExternalSession
    freeze=c.verify();cfg,data,planner=c.data();settings=c.config()['dqfd'];path=c.directory(method,seed)
    if (method,seed) not in c.RUNS:raise ValueError('Unauthorized extra training')
    torch.set_num_threads(1)
    identity=dict(method=method,seed=seed,freeze_sha256=c.sha(c.REG/'frozen_protocol.json'))
    with run_lock(path/'run.lock'):
        manifest=path/'run_manifest.json'
        if manifest.exists():
            if c.read(manifest)['identity']!=identity:raise ValueError('Existing run identity differs')
            if (path/'completion.json').exists():print('Completed baseline verified and reused; no retraining.');return
            if not resume:raise ValueError('Existing run requires explicit --resume')
            if (path/'failure.json').exists():raise ValueError('Retained failure must not be automatically retrained')
            if not (path/'checkpoint_latest.pth').exists():raise ValueError('Weights alone are not a full recovery checkpoint')
        elif resume:raise ValueError('No existing run to resume')
        session=ExternalSession(cfg,data,planner,method,seed,settings)
        if resume:
            state=torch.load(path/'checkpoint_latest.pth',map_location='cpu',weights_only=False);session.restore(state,identity);del state
        else:
            c.immutable(manifest,dict(identity=identity,config=cfg,external_config=c.config(),device=str(session.agent.device),
                torch_version=torch.__version__,created=now(),shared_identity=freeze['shared_identity'],
                demonstration_sha256=c.sha(c.REG/'demonstrations.npz') if method=='dqfd' else None,
                no_td_target_bound=True,no_online_astar_advice=True,test_pools_used=False,
                expected_online_updates=499501,expected_offline_updates=20000 if method=='dqfd' else 0,
                zero_step_validation='after offline pretraining for DQfD, fresh network for D3QN'))
        started=perf_counter();start_step=session.step
        def status():
            c.write(path/'status.json',dict(running=True,complete=False,pid=os.getpid(),phase=session.phase,
                step=session.step,offline_updates=session.offline_updates,online_updates=session.online_updates,
                total_updates=session.agent.update_steps,session_steps_per_second=(session.step-start_step)/max(.001,perf_counter()-started),updated=now()))
        try:
            status()
            while session.phase=='offline' and session.offline_updates<settings['pretraining_updates']:
                session.update(offline=True)
                if session.offline_updates%1000==0:
                    session.flush_metrics();logs(session,path);checkpoint(session,path,identity);status()
                    print(f'[{method} seed{seed}] offline {session.offline_updates}/20000',flush=True)
            if session.phase=='offline':
                session.flush_metrics();session.phase='online';session.agent.save_weights(path/'model_pretrained.pth');checkpoint(session,path,identity)
            if not session.validation:validate(session,path);checkpoint(session,path,identity)
            while session.step<500000:
                session.advance()
                if session.step>=session.next_validation:
                    validate(session,path);session.next_validation+=cfg['training']['validation_interval']
                if session.step>=session.next_checkpoint:
                    session.next_checkpoint+=cfg['training']['checkpoint_interval'];logs(session,path);checkpoint(session,path,identity)
                if session.step%1000==0:status()
            expected_offline=20000 if method=='dqfd' else 0
            if session.online_updates!=499501 or session.offline_updates!=expected_offline or session.agent.action_steps!=500000:
                raise ValueError('Online/offline fixed budget changed')
            if method=='dqfd' and session.collector.pending:raise ValueError('Unflushed final n-step records')
            session.flush_metrics();logs(session,path);session.agent.save_weights(path/'model_final.pth');checkpoint(session,path,identity)
            c.write(path/'result.json',dict(identity=identity,complete=True,step=session.step,updates=session.agent.update_steps,
                online_updates=session.online_updates,offline_updates=session.offline_updates,
                final_validation=session.validation[-1],final_model_sha256=c.sha(path/'model_final.pth'),
                test_pools_evaluated=False,demonstration_cost=c.read(c.REG/'demonstration_report.json') if method=='dqfd' else None,
                completed=now(),session_seconds=perf_counter()-started))
            c.write(path/'completion.json',dict(complete=True,step=500000));c.write(path/'status.json',dict(running=False,complete=True,step=500000,updated=now()))
        except BaseException as error:
            c.write(path/'failure.json',dict(failed=True,method=method,seed=seed,step=session.step,
                offline_updates=session.offline_updates,online_updates=session.online_updates,error=repr(error),
                numerical=isinstance(error,FloatingPointError),traceback=traceback.format_exc(),automatic_retry=False,created=now()))
            c.write(path/'status.json',dict(running=False,complete=False,step=session.step,error=repr(error),updated=now()))
            raise


def pipeline(resume=False,workers=2):
    if workers not in (1,2):raise ValueError('At most two training workers')
    c.verify()
    with run_lock(c.OUT/'pipeline.lock'):
        # C/D final models exist: perform their static pairs before new training.
        from diagnose_multitask_pairs import run as pair
        for method in ('C','D'):
            for seed in (0,1):pair(method,seed,replay_authorized())
        subprocess.run([sys.executable,'scripts/analyze_multitask_external.py'],cwd=c.ROOT,check=True)
        queue=[];active=[];failures=[]
        for method,seed in c.RUNS:
            path=c.directory(method,seed)
            if (path/'completion.json').exists():c.model_info(method,seed);pair(method,seed,replay_authorized());continue
            if (path/'failure.json').exists():failures.append(dict(method=method,seed=seed,error='retained failure'));continue
            existing=(path/'run_manifest.json').exists()
            if existing and not resume:raise ValueError('Existing incomplete baseline requires --resume; no duplicate submission')
            queue.append((method,seed,existing))
        logroot=c.OUT/'logs';logroot.mkdir(parents=True,exist_ok=True)
        while queue or active:
            remaining=[]
            for job in active:
                code=job['process'].poll()
                if code is None:remaining.append(job)
                else:
                    job['stream'].close();print(f'END {job["method"]} seed{job["seed"]}: {code}',flush=True)
                    if code:failures.append(dict(method=job['method'],seed=job['seed'],returncode=code))
                    else:pair(job['method'],job['seed'],replay_authorized())
                    subprocess.run([sys.executable,'scripts/analyze_multitask_external.py'],cwd=c.ROOT,check=True)
            active=remaining
            while queue and len(active)<workers:
                free=resources()
                if active and free is not None and free<3.:break
                method,seed,restore=queue.pop(0)
                stream=(logroot/f'{method}_seed_{seed}.log').open('a' if restore else 'x',encoding='utf-8')
                args=[sys.executable,'-X','utf8','-u','scripts/run_multitask_external.py','train','--method',method,'--seed',str(seed)]
                if restore:args.append('--resume')
                process=subprocess.Popen(args,cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append(dict(method=method,seed=seed,process=process,stream=stream));print(f'START {method} seed{seed} pid={process.pid}',flush=True)
            c.write(c.OUT/'pipeline_status.json',dict(running=True,complete=False,pid=os.getpid(),updated=now(),
                active=[dict(method=j['method'],seed=j['seed'],pid=j['process'].pid) for j in active],
                queued=[dict(method=m,seed=s,resume=r) for m,s,r in queue],failures=failures,
                available_memory_gb=resources(),maximum_training_workers=workers,
                test_pool_evaluation=False,future_training_enabled=False))
            if active:sleep(5)
        for method in c.METHODS:
            for seed in (0,1):pair(method,seed,replay_authorized())
        subprocess.run([sys.executable,'scripts/analyze_multitask_external.py'],cwd=c.ROOT,check=True)
        c.write(c.OUT/'pipeline_status.json',dict(running=False,complete=not failures,failures=failures,active=[],queued=[],updated=now(),
                                                test_pool_evaluation=False,future_training_enabled=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','pipeline','status'))
    parser.add_argument('--method',choices=('d3qn','dqfd'));parser.add_argument('--seed',type=int,choices=(0,1))
    parser.add_argument('--workers',type=int,choices=(1,2),default=2);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='pipeline':pipeline(args.resume,args.workers)
    elif args.action=='status':print((c.OUT/'pipeline_status.json').read_text(encoding='utf-8') if (c.OUT/'pipeline_status.json').exists() else 'not started')
    elif args.method is None or args.seed is None:parser.error('train requires method and seed')
    else:train(args.method,args.seed,args.resume)
