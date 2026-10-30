"""Exactly two new 200k fixed-task CE runs; immutable margin controls reused."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime
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

import fixed_ce_common as c
from fixed_ce_agent import AST_AUDIT,GENERATED_UPDATE_SOURCE,GENERATED_WRAPPER_SOURCE
from replication_recovery import Recovery,STAGE_LOCALS,instrument


def global_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def evaluate_isolated(config,inputs,seed,agent):
    before=c.digest(agent.training_state_dict());rng=c.digest(global_rng())
    with patch.object(agent,'static_advice',side_effect=AssertionError('Teacher called during autonomous evaluation')),\
         patch.object(agent,'observed_risk',side_effect=AssertionError('Risk protection called during autonomous evaluation')),\
         patch.object(agent,'train_batch',side_effect=AssertionError('Update called during evaluation')):
        result=c.prior.legacy.old.evaluate(config,inputs,seed,agent)
    if before!=c.digest(agent.training_state_dict()) or rng!=c.digest(global_rng()):
        raise AssertionError('Evaluation changed learner/target/optimizer/schedules/RNG')
    return result


def recovery_check(config,inputs,root):
    small=deepcopy(config);small['training_advice']['decay_environment_steps']=12
    training=replace(c.prior.legacy.effective_training(config,0,'formal'),max_environment_steps=24,max_steps=6,
        learning_starts=4,batch_size=4,epsilon_decay_environment_steps=24,progress_interval_environment_steps=6)
    def execute(directory,resume=False,interrupt=False):
        agent=c.make_agent(small,inputs[0],0,'cpu')
        factory=c.prior.legacy.runtime.make_factory(small,*inputs,method='unguided',seed=0)
        manager=Recovery(directory,agent,factory,STAGE_LOCALS,resume=resume,identity='fixed-ce-recovery-unit',
                         checkpoint_interval=6,stop_after=12 if interrupt else None)
        result=instrument(c.prior.legacy.train_d3qn,STAGE_LOCALS)([inputs[0]],agent,training,
            reward_config=c.prior.legacy._reward_config(small),environment_factory=factory,
            progress_callback=lambda *args:None,_recovery=manager)
        return c.digest(agent.training_state_dict()),result.episode_records,result.environment_steps,result.gradient_updates
    plain=execute(root/'plain')
    try:execute(root/'resumed',interrupt=True)
    except InterruptedError:pass
    else:raise AssertionError('Recovery unit did not interrupt')
    resumed=execute(root/'resumed',resume=True)
    if plain!=resumed:raise AssertionError('CE interruption/recovery changes training')
    return dict(passed=True,steps=plain[2],updates=plain[3],formal_training=False,
                source='24-step CPU integration unit; not part of the two 200k runs',directory=str(root.relative_to(c.ROOT)))


def preflight():
    if (c.DATA/'freeze.json').exists():
        c.verify_freeze();print('Existing CE preflight/freeze verified; preserved.',flush=True);return
    config,inputs=c.configure();pairs,protected=c.audit_baseline(config)
    attempt=c.RESULTS/'preflight_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S')
    attempt.mkdir(parents=True,exist_ok=False)
    try:
        suite=unittest.defaultTestLoader.discover(str(c.ROOT/'tests'),pattern='test_fixed_ce.py')
        with (attempt/'unit_tests.log').open('w',encoding='utf-8') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        if not result.wasSuccessful():raise AssertionError('CE unit checks failed; see retained log')
        for pair in pairs:
            for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
                agent=c.make_agent(config,inputs[0],pair['seed'],device)
                if c.digest(agent.training_state_dict())!=pair['initial_state_sha256']:
                    raise AssertionError('CE initial policy/target/optimizer/learner and advice RNG differ from original margin')
            pair['initialization_cpu_cuda_verified']=torch.cuda.is_available()
        validation,scenes=c.prior.legacy.runtime.check(config,inputs,list(c.SEEDS))
        if len(scenes)!=50:raise AssertionError('Expected original 50 validation scenes')
        unit_recovery=recovery_check(config,inputs,attempt/'recovery')
        # Exercise the full frozen autonomous evaluation with forbidden hooks.
        evaluation=[]
        for seed in c.SEEDS:
            agent=c.make_agent(config,inputs[0],seed,'cpu')
            summary,details,traces=evaluate_isolated(config,inputs,seed,agent)
            old=c.prior.base.load_csv(c.ROOT/config['ce_control']['baseline_root']/f'seed_{seed}'/'validation_curve.csv')[0]
            for metric in ('safe_success_rate','collision_rate','timeout_rate'):
                if abs(summary[metric]-float(old[metric]))>1e-12:raise AssertionError('Initial validation differs from paired margin seed')
            c.write_csv(details,attempt/f'initial50_seed_{seed}.csv')
            evaluation.append(dict(seed=seed,episodes=len(details),teacher_calls=0,risk_calls=0,updates=0,
                                   learner_and_rng_unchanged=True,summary=summary))
        for name,digest in protected.items():
            if c.sha(c.ROOT/name)!=digest:raise AssertionError('Preflight changed historical evidence')
        report=dict(passed=True,tests=result.testsRun,ast_audit=AST_AUDIT,pairs=pairs,
            recovery=unit_recovery,validation=validation,autonomous_evaluation=evaluation,
            map=inputs[0].manifest(),training_runs_authorized=2,steps_per_run=200000,updates_per_run=199501,
            old_group_retrained=False,independent500_used=False,pretraining_updates=0,
            explanation='CE is the only loss intervention; same eligible labels and true-action TD. Historical boundary-time validation retained.')
        c.DATA.mkdir(parents=True,exist_ok=True)
        for pair in pairs:
            source=c.ROOT/pair['baseline_directory'];target=c.DATA/'baseline_snapshots'/f"seed_{pair['seed']}"
            target.mkdir(parents=True,exist_ok=False)
            for name in ('run_manifest.json','result.json','completion.json','validation_curve.csv','validation_details.csv',
                         'value_learning.csv','supervision_alignment.csv','advice_budget.csv'):
                shutil.copyfile(source/name,target/name)
        validation_source=c.ROOT/config['dataset']['validation_reference']
        shutil.copyfile(validation_source,c.DATA/'fixed_validation_scenarios.json')
        source_hashes=c.sources()
        for name in source_hashes:
            target=c.DATA/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(c.ROOT/name,target)
        (c.DATA/'generated_ce_train_batch.py.txt').write_text(GENERATED_UPDATE_SOURCE+'\n',encoding='utf-8')
        (c.DATA/'generated_teacher_wrapper.py.txt').write_text(GENERATED_WRAPPER_SOURCE+'\n',encoding='utf-8')
        c.write_json(report,attempt/'passed.json')
        c.write_json(dict(config=config,source_sha256=source_hashes,protected_files_sha256=protected,
            copied_input_sha256={str(p.relative_to(c.DATA)):c.sha(p) for p in c.DATA.rglob('*') if p.is_file()},
            checks=report,created=datetime.now().isoformat()),c.DATA/'freeze.json')
        c.write_json(report,c.RESULTS/'preflight.json')
        from analyze_fixed_ce import analyze
        analyze()
        print(f'PREFLIGHT PASSED: {result.testsRun} tests, paired initialization and original50 isolation verified.',flush=True)
    except BaseException as error:
        c.write_json(dict(passed=False,error=repr(error),traceback=traceback.format_exc(),formal_training_runs=0),attempt/'failure.json')
        raise


def run_one(seed,resume=False):
    frozen=c.verify_freeze();config,inputs=c.configure();directory=c.OUT/f'seed_{seed}'
    if (directory/'completion.json').exists():
        if resume:print(f'CE seed{seed} already complete; preserved.',flush=True);return
        raise FileExistsError('Completed CE run exists; do not retrain')
    if directory.exists() and not resume:raise FileExistsError('Incomplete run preserved; explicit --resume required')
    if resume and not (directory/'recovery_latest.pth').exists():raise FileNotFoundError('No complete recovery checkpoint')
    if resume and (directory/'status.json').exists():
        error=c.read(directory/'status.json').get('error','').lower()
        if 'nonfinite' in error or 'non-finite' in error:raise ValueError('Do not resume a nonfinite failure automatically')
    directory.mkdir(parents=True,exist_ok=resume)
    agent=c.make_agent(config,inputs[0],seed)
    pair=next(p for p in frozen['checks']['pairs'] if p['seed']==seed)
    if c.digest(agent.training_state_dict())!=pair['initial_state_sha256']:raise AssertionError('Runtime initialization mismatch')
    factory=c.prior.legacy.runtime.make_factory(config,*inputs,method='unguided',seed=seed)
    manager=Recovery(directory,agent,factory,STAGE_LOCALS,resume=resume,
        identity=dict(study=c.NAME,seed=seed,freeze_sha256=c.sha(c.DATA/'freeze.json')))
    def old_rows(name):
        return [r for r in c.prior.base.load_csv(directory/name) if int(r['environment_steps_total'])<=manager.last_step] if resume else []
    curves,details,behaviors=old_rows('validation_curve.csv'),old_rows('validation_details.csv'),old_rows('failure_behaviors.csv')
    training=c.prior.legacy.effective_training(config,seed,'formal')
    metadata=dict(seed=seed,method='same_teacher_online_ce',temperature=1.,config=config,effective_training=asdict(training),
        initial_state_sha256=pair['initial_state_sha256'],source_sha256=frozen['source_sha256'],
        freeze_sha256=c.sha(c.DATA/'freeze.json'),baseline_model_sha256=pair['baseline_model_sha256'],
        baseline_curve_sha256=pair['baseline_curve_sha256'],fresh_random_initialization=True,
        evaluation_advice=False,independent500_used=False,route_input=False,pretraining_updates=0,
        soft_risk_weight=False,device=str(agent.device),torch_version=torch.__version__,pid=os.getpid())
    if not resume:c.write_json(metadata,directory/'run_manifest.json')
    distances=c.prior.legacy.static_distances(inputs[0])
    def diagnostics():
        c.write_csv(agent.advice_records(),directory/'advice_budget.csv')
        repair=[]
        for row in agent.repair_records():
            row=dict(row);row['supervision_ce_loss_mean']=row.pop('demo_margin_loss_mean')
            row['supervision_weight_mean']=row.pop('teacher_margin_weight_mean');row['supervision_loss_type']='CE_T1'
            repair.append(row)
        c.write_csv(repair,directory/'value_learning.csv');c.write_csv(agent.supervision_records(),directory/'supervision_alignment.csv')
    def validate(step):
        summary,part,traces=evaluate_isolated(config,inputs,seed,agent)
        if len(part)!=50:raise AssertionError('Validation scope changed')
        failures=[t for t in traces if not t['safe_success']]
        curves.append(dict(environment_steps_total=step,**summary));details.extend(dict(environment_steps_total=step,**r) for r in part)
        behaviors.extend(dict(environment_steps_total=step,**c.prior.legacy.trace_behavior(t,distances)) for t in failures)
        c.write_csv(curves,directory/'validation_curve.csv');c.write_csv(details,directory/'validation_details.csv')
        c.write_csv(behaviors,directory/'failure_behaviors.csv');c.write_json(failures,directory/f'validation_failures_{step:06d}.json')
        agent.save_weights(directory/f'model_step_{step:06d}.pth');diagnostics()
        c.write_json(dict(running=True,complete=False,environment_steps=step,recoverable_environment_steps=manager.last_step,
                          pid=os.getpid(),updated=datetime.now().isoformat()),directory/'status.json')
        from analyze_fixed_ce import analyze
        analyze(plot=False)
        print(f'[CE seed{seed}] step={step} safe={summary["safe_success_rate"]:.1%} collision={summary["collision_rate"]:.1%} timeout={summary["timeout_rate"]:.1%}',flush=True)
    def progress(_episode,records,full):
        if full:validate(int(records[-1]['environment_steps_total']));c.write_csv(list(records),directory/'training.csv')
    started=perf_counter()
    c.write_json(dict(running=True,complete=False,environment_steps=manager.last_step,pid=os.getpid()),directory/'status.json')
    try:
        if not resume:validate(0)
        result=instrument(c.prior.legacy.train_d3qn,STAGE_LOCALS)([inputs[0]],agent,training,
            reward_config=c.prior.legacy._reward_config(config),environment_factory=factory,progress_callback=progress,_recovery=manager)
        if result.environment_steps!=200000 or result.gradient_updates!=199501 or agent.training_action_steps!=200000:
            raise AssertionError('CE environment/update budget mismatch')
        if int(curves[-1]['environment_steps_total'])!=200000:validate(200000)
        for row in agent.advice_records():
            if row['bin_start_step']>=100001 and (row['applied']!=0 or row['mean_requested_probability']!=0):raise AssertionError('Advice did not exit')
        for row in agent.repair_records():
            if row['bin_start_step']>=100001 and abs(row['teacher_margin_weight_mean']-.1)>1e-10:raise AssertionError('Supervision floor differs')
        agent.save_weights(directory/'model_final.pth');diagnostics();c.write_csv(list(result.episode_records),directory/'training.csv')
        c.write_json(dict(**metadata,complete=True,environment_steps=200000,gradient_updates=199501,final_validation=curves[-1],
            final_model_sha256=c.sha(directory/'model_final.pth'),online_training_seconds=manager.prior_training_seconds+result.training_seconds,
            session_seconds=perf_counter()-started,completed=datetime.now().isoformat()),directory/'result.json')
        c.write_json(dict(complete=True,environment_steps=200000,gradient_updates=199501),directory/'completion.json')
        c.write_json(dict(complete=True,running=False,environment_steps=200000,pid=os.getpid()),directory/'status.json')
        from analyze_fixed_ce import analyze
        analyze();c.verify_freeze();print(f'[CE seed{seed}] COMPLETE',flush=True)
    except BaseException as error:
        c.write_json(dict(complete=False,running=False,error=repr(error),traceback=traceback.format_exc(),
            recoverable_environment_steps=manager.last_step,pid=os.getpid()),directory/'status.json')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--seeds',nargs='+',type=int,choices=c.SEEDS,default=list(c.SEEDS))
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):parser.error('Duplicate seeds are not allowed')
    torch.set_num_threads(1)
    if not args.train:preflight();return
    c.verify_freeze();c.OUT.mkdir(parents=True,exist_ok=True)
    lock=c.OUT/'runner.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    try:
        for seed in args.seeds:
            c.write_json(dict(running=True,pid=os.getpid(),active_seed=seed,planned_seeds=args.seeds,
                maximum_runs=2,steps_per_run=200000,updated=datetime.now().isoformat()),c.RESULTS/'queue_status.json')
            run_one(seed,args.resume)
        c.write_json(dict(running=False,complete=True,seeds=args.seeds,updated=datetime.now().isoformat()),c.RESULTS/'queue_status.json')
    except BaseException as error:
        c.write_json(dict(running=False,complete=False,error=repr(error),active_seed=seed,updated=datetime.now().isoformat()),c.RESULTS/'queue_status.json')
        raise
    finally:lock.unlink()


if __name__=='__main__':main()
