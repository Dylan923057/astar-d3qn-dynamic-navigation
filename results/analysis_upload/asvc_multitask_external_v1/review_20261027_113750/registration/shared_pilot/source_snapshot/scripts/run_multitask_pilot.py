"""Prepare/start/resume exactly C,D x seed 0,1; no formal ablation launch path."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import multitask_pilot_common as c


def now():return datetime.now().astimezone().isoformat(timespec='seconds')


@contextmanager
def run_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    if path.stat().st_size==0:stream.write(b'0');stream.flush()
    stream.seek(0)
    if os.name=='nt':
        import msvcrt
        try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:stream.close();raise RuntimeError('Another process holds '+str(path))
    else:
        import fcntl
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:yield
    finally:
        stream.seek(0)
        if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(stream,fcntl.LOCK_UN)
        stream.close()


def prepare():
    if (c.REG/'frozen_protocol.json').exists():
        c.verify_freeze();print('Existing verified freeze preserved.');return
    if c.OUT.exists() and any(c.OUT.glob('*/seed_*/run_manifest.json')):
        raise ValueError('An existing experiment cannot be prepared again')
    cfg=c.config()
    if [tuple(r) for r in cfg['experiment']['allowed_runs']]!=list(c.RUNS):raise ValueError('Pilot scope changed')
    protection={str(p.relative_to(c.ROOT)).replace('\\','/'):c.sha(p) for p in c.legacy_paths()}
    c.immutable(c.REG/'legacy_protection.json',protection)
    data,validation,previews=c.generate_data(cfg)
    c.immutable(c.REG/'task_pools.json',data)
    c.immutable(c.REG/'validation_scenes.json',validation)
    c.immutable(c.REG/'training_sequence_preview.json',previews)
    checks=subprocess.run([sys.executable,'-u',str(c.ROOT/'tests/test_multitask_pilot.py')],cwd=c.ROOT,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    c.RESULTS.mkdir(parents=True,exist_ok=True)
    (c.RESULTS/'preflight_checks.log').write_text(checks.stdout,encoding='utf-8')
    print(checks.stdout,flush=True)
    if checks.returncode:raise ValueError('Preflight failed; no training was started')
    inputs={name:c.sha(c.REG/name) for name in ('task_pools.json','validation_scenes.json','training_sequence_preview.json','legacy_protection.json')}
    protocol=dict(name=c.NAME,created=now(),config=cfg,source_sha256=c.sources(),input_sha256=inputs,
        preflight_log_sha256=c.sha(c.RESULTS/'preflight_checks.log'),allowed_runs=c.RUNS,
        steps_per_run=500000,expected_updates=499501,test_pools_evaluated=False,
        split_sizes={k:len(v) for k,v in data['pools'].items()},
        hard_screen='Same C/D static legality, observed three-frame radius-1 veto, collision-transition exclusion from supervision only.',
        supervision='C: base schedule; D: base*(1-R)^2. Eligible-count normalization. TD includes every sampled transition.',
        timeout='max(100,4*static_shortest_length+20); goal/collision terminated, timeout truncated; only terminated suppresses TD bootstrap.',
        sequence='Same training seed/episode -> same task, scene seed, routes, phases, directions and count; policy-dependent episode lengths may differ.',
        risk='Observed mutual-unique associations; H4 constant observed velocity; ambiguous/new/turned/lost detections use static-free bounded-speed reachable sets. No hidden obstacle state.',
        limitations='Single-map pilot with two training seeds; heuristic risk is neither probability nor safety guarantee. No unseen-combination/goal evaluation in this pilot.',
        formal_presets_launchable=False)
    c.immutable(c.REG/'frozen_protocol.json',protocol)
    for name in protocol['source_sha256']:
        source=c.ROOT/name;target=c.REG/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and c.sha(target)!=c.sha(source):raise ValueError('Snapshot collision')
        if not target.exists():shutil.copyfile(source,target)
    c.verify_freeze()
    c.write(c.RESULTS/'preflight.json',dict(passed=True,created=now(),freeze_sha256=c.sha(c.REG/'frozen_protocol.json'),
                                         protected_files=len(protection),tests_log='preflight_checks.log'))
    print('FROZEN: four pilot runs only; test pools remain unevaluated.',flush=True)


def save_checkpoint(session,directory,identity):
    import torch
    temporary=directory/'checkpoint_latest.pth.tmp'
    torch.save(session.snapshot(identity),temporary)
    temporary.replace(directory/'checkpoint_latest.pth')


def write_logs(session,directory):
    c.csv_write(directory/'training_episodes.csv',session.episodes)
    c.csv_write(directory/'validation_curve.csv',session.validation)
    c.csv_write(directory/'validation_details.csv',session.validation_details)
    c.csv_write(directory/'update_log.csv',session.update_log)
    c.csv_write(directory/'offline_cases.csv',session.case_rows)
    stats=session.agent.replay_stats;advice=session.agent.advice_stats
    c.write(directory/'supervision_statistics.json',dict(environment_steps=session.step,updates=session.agent.update_steps,
        environment_label_availability=advice['environment_label_available']/max(1,advice['steps']),
        replay_supervision_coverage=stats['eligible_draws']/max(1,stats['sampled_draws']),
        replay_actual_statistics=dict(stats),advice_statistics=dict(advice),
        environment_association_counts=dict(session.environment_risk_stats),
        risk_bins={str(k):dict(v) for k,v in session.agent.risk_bins.items()},
        note='Environment detections and replay draws are different denominators. Risk in replay is the immutable observation-derived score of that transition.'))


def train(group,seed,resume=False):
    import torch
    from multitask_pilot_engine import TrainSession,evaluate,offline_cases
    freeze=c.verify_freeze();cfg,data,planner=c.load_data();directory=c.run_directory(group,seed)
    torch.set_num_threads(1)
    identity=dict(group=group,seed=seed,freeze_sha256=c.sha(c.REG/'frozen_protocol.json'))
    with run_lock(directory/'run.lock'):
        existing=directory/'run_manifest.json'
        if existing.exists():
            if not resume:raise ValueError('Existing run refuses overwrite; use --resume for a complete checkpoint')
            if c.read(existing)['identity']!=identity:raise ValueError('Run identity differs')
            if (directory/'failure.json').exists():raise ValueError('Failed run is retained; no automatic retry/replacement')
            if (directory/'completion.json').exists():print('Completed run preserved.');return
            if not (directory/'checkpoint_latest.pth').exists():raise ValueError('No full recovery checkpoint; weights alone are insufficient')
        elif resume:raise ValueError('No run to resume')
        session=TrainSession(cfg,data,planner,group,seed)
        if resume:
            checkpoint=torch.load(directory/'checkpoint_latest.pth',map_location='cpu',weights_only=False)
            session.restore(checkpoint,identity);del checkpoint
        else:
            c.immutable(existing,dict(identity=identity,created=now(),config=cfg,device=str(session.agent.device),
                                     torch_version=torch.__version__,expected_steps=500000,expected_updates=499501,
                                     evaluation='greedy network only, 50 frozen validation scenes, test pools disabled',
                                     restoration='optimizer, online/target networks, replay and RNG, counters, in-flight environment and local history'))
        started=time.perf_counter();start_step=session.step
        validation=c.read(c.REG/'validation_scenes.json')
        try:
            c.write(directory/'status.json',dict(running=True,complete=False,pid=os.getpid(),step=session.step,started=now()))
            if not resume:
                summary,details,traces=evaluate(session,validation)
                session.validation.append(summary);session.validation_details.extend(details)
                session.case_rows.extend(offline_cases(session,traces))
                write_logs(session,directory);save_checkpoint(session,directory,identity)
            while session.step<cfg['training']['environment_steps']:
                session.advance()
                if session.step>=session.next_validation:
                    summary,details,traces=evaluate(session,validation)
                    session.validation.append(summary);session.validation_details.extend(details)
                    session.case_rows.extend(offline_cases(session,traces))
                    session.next_validation+=cfg['training']['validation_interval']
                    session.agent.save_weights(directory/f'model_step_{session.step:06d}.pth')
                    print(f'[{group} seed {seed}] {session.step}/500000 success={summary["success_rate"]:.3f} collision={summary["collision_rate"]:.3f} timeout={summary["timeout_rate"]:.3f}',flush=True)
                if session.step>=session.next_checkpoint:
                    session.next_checkpoint+=cfg['training']['checkpoint_interval']
                    write_logs(session,directory);save_checkpoint(session,directory,identity)
                if session.step%1000==0:
                    c.write(directory/'status.json',dict(running=True,complete=False,pid=os.getpid(),step=session.step,
                        updates=session.agent.update_steps,checkpoint_step=(session.next_checkpoint-cfg['training']['checkpoint_interval']),
                        session_steps_per_second=(session.step-start_step)/max(.001,time.perf_counter()-started),updated=now()))
            if session.agent.update_steps!=499501 or session.agent.action_steps!=500000:raise ValueError('Fixed budget differs')
            session.flush_metrics();write_logs(session,directory)
            session.agent.save_weights(directory/'model_final.pth');save_checkpoint(session,directory,identity)
            c.write(directory/'result.json',dict(identity=identity,complete=True,step=session.step,updates=session.agent.update_steps,
                final_validation=session.validation[-1],final_model_sha256=c.sha(directory/'model_final.pth'),
                test_pools_evaluated=False,completed=now(),partial_final_episode=session.env is not None))
            c.write(directory/'completion.json',dict(complete=True,step=session.step))
            c.write(directory/'status.json',dict(running=False,complete=True,step=session.step,updated=now()))
        except BaseException as error:
            c.write(directory/'status.json',dict(running=False,complete=False,step=session.step,error=repr(error),updated=now()))
            c.write(directory/'failure.json',dict(failed=True,error=repr(error),nonfinite=isinstance(error,FloatingPointError),
                    step=session.step,traceback=traceback.format_exc(),retain_seed=True,automatic_retry=False,created=now()))
            raise


def resources():
    import ctypes
    import struct
    if os.name!='nt':return None
    data=ctypes.create_string_buffer(64);struct.pack_into('I',data,0,64)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(data):return None
    return struct.unpack_from('Q',data,16)[0]/2**30


def pipeline(resume=False,workers=2):
    if workers not in (1,2):raise ValueError('Only one or two training workers are allowed')
    c.verify_freeze()
    with run_lock(c.OUT/'pipeline.lock'):
        queue=[];active=[];failures=[]
        for group,seed in c.RUNS:
            directory=c.run_directory(group,seed)
            if (directory/'completion.json').exists():
                if not resume:raise ValueError('Refusing to overwrite a completed pilot')
                continue
            if (directory/'failure.json').exists():
                failures.append(dict(group=group,seed=seed,error='retained failure'));continue
            if (directory/'run_manifest.json').exists() and not resume:raise ValueError('Existing run requires explicit --resume')
            queue.append((group,seed,(directory/'run_manifest.json').exists()))
        logs=c.OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
        while queue or active:
            remaining=[]
            for job in active:
                code=job['process'].poll()
                if code is None:remaining.append(job)
                else:
                    job['stream'].close();print(f'END {job["group"]} seed {job["seed"]}: {code}',flush=True)
                    if code:failures.append(dict(group=job['group'],seed=job['seed'],returncode=code))
                    subprocess.run([sys.executable,'scripts/analyze_multitask_pilot.py'],cwd=c.ROOT,check=True)
            active=remaining
            while queue and len(active)<workers:
                available=resources()
                # Resource policy only affects concurrency, never scientific settings.
                if active and available is not None and available<3.0:break
                group,seed,restore=queue.pop(0)
                name=f'{group}_seed_{seed}'
                stream=(logs/f'{name}.log').open('a' if restore else 'x',encoding='utf-8')
                args=[sys.executable,'-u','scripts/run_multitask_pilot.py','train','--group',group,'--seed',str(seed)]
                if restore:args.append('--resume')
                proc=subprocess.Popen(args,cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append(dict(group=group,seed=seed,process=proc,stream=stream))
                print(f'START {name} pid={proc.pid}',flush=True)
            c.write(c.OUT/'pipeline_status.json',dict(running=True,complete=False,pid=os.getpid(),updated=now(),
                active=[dict(group=j['group'],seed=j['seed'],pid=j['process'].pid) for j in active],
                queued=[dict(group=g,seed=s,resume=r) for g,s,r in queue],failures=failures,
                maximum_training_workers=workers,available_memory_gb=resources(),test_evaluation_enabled=False,
                formal_ablations_enabled=False))
            if active:time.sleep(5)
        subprocess.run([sys.executable,'scripts/analyze_multitask_pilot.py'],cwd=c.ROOT,check=True)
        c.write(c.OUT/'pipeline_status.json',dict(running=False,complete=not failures,failures=failures,updated=now(),
                                                active=[],queued=[],test_evaluation_enabled=False,formal_ablations_enabled=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','pipeline','status'))
    parser.add_argument('--group',choices=('C','D'));parser.add_argument('--seed',type=int,choices=(0,1))
    parser.add_argument('--resume',action='store_true');parser.add_argument('--workers',type=int,default=2,choices=(1,2))
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='pipeline':pipeline(args.resume,args.workers)
    elif args.action=='status':
        print((c.OUT/'pipeline_status.json').read_text(encoding='utf-8') if (c.OUT/'pipeline_status.json').exists() else 'not started')
    elif args.group is None or args.seed is None:parser.error('train requires --group and --seed')
    else:train(args.group,args.seed,args.resume)


if __name__=='__main__':main()
