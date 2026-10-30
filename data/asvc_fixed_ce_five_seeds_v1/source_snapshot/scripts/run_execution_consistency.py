"""Frozen three-seed single-factor consistency ablation; --train starts authorized runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

import teaching_efficiency_common as common
from analyze_runtime_path_pilot import load_csv
from replication_recovery import Recovery,STAGE_LOCALS,instrument
from astar_d3qn.agents.astar_all_eligible_supervision import AStarAllEligibleSupervisionAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.utils.io import write_json,write_records_csv
from astar_d3qn.utils.seed import seed_everything

ROOT,legacy=common.ROOT,common.legacy
CONFIG=ROOT/'configs/whole_map_91701_execution_consistency_v1.yaml'
METHOD='advice_bound_margin_all_eligible'
SEEDS=(0,1,2)


def configure():
    config=legacy.load_config(CONFIG);old=legacy.load_config(legacy.CONFIG)
    if {k:v for k,v in config.items() if k not in ('experiment','execution_consistency')}!={k:v for k,v in old.items() if k!='experiment'}:
        raise ValueError('Only executed-teacher equality eligibility may change.')
    if config['execution_consistency']!=dict(method=METHOD,seeds=list(SEEDS),require_executed_teacher_match=False,
        registration_root='data/whole_map_91701_execution_consistency_v1',teacher_action_separate_from_transition_action=True,
        reweight_or_resample=False,use_independent500=False,pretraining_updates=0,offline_demonstrations=0):
        raise ValueError('Use fixed seed0-2 single-factor protocol; do not compensate label quantity.')
    return config,legacy.validate_config(old)


def make_agent(config,problem,seed,device=None):
    seed_everything(seed);values=dict(config['agent'])
    if device:values['device']=device
    ad,repair=config['training_advice'],config['value_repair']
    return AStarAllEligibleSupervisionAgent(D3QNConfig((5,15,15),4,5,seed=seed,**values),problem,
        repair_method='advice_bound_margin',lower=repair['target_lower'],upper=repair['target_upper'],
        margin=repair['margin'],margin_weight=repair['margin_weight'],probability=ad['probability'],
        decay_steps=ad['decay_environment_steps'],risk_radius=ad['risk_radius'],rng_offset=ad['rng_seed_offset'])


def source_hashes():
    sources=dict(legacy.source_hashes())
    for path in (CONFIG,Path(__file__),ROOT/'scripts/replication_recovery.py',ROOT/'scripts/teaching_efficiency_common.py',
        ROOT/'src/astar_d3qn/agents/independent_teacher_actions.py',
        ROOT/'src/astar_d3qn/agents/astar_all_eligible_supervision.py',ROOT/'tests/test_execution_consistency.py'):
        sources[str(path.relative_to(ROOT))]=common.sha256(path)
    return sources


def recovery_check(config,inputs):
    training=replace(legacy.effective_training(config,0,'formal'),max_environment_steps=24,max_steps=6,
        learning_starts=4,batch_size=4,epsilon_decay_environment_steps=24,progress_interval_environment_steps=6)
    def execute(path,interrupted=False,resume=False):
        agent=make_agent(config,inputs[0],0,'cpu')
        factory=legacy.runtime.make_factory(config,*inputs,method='unguided',seed=0)
        recovery=Recovery(path,agent,factory,STAGE_LOCALS,resume=resume,identity='single-factor-parity',
            checkpoint_interval=6,stop_after=12 if interrupted else None)
        result=instrument(legacy.train_d3qn,STAGE_LOCALS)([inputs[0]],agent,training,
            reward_config=legacy._reward_config(config),environment_factory=factory,progress_callback=lambda *args:None,
            _recovery=recovery)
        return legacy.runtime.state_digest(agent.training_state_dict()),result.episode_records
    with TemporaryDirectory(dir=ROOT/'outputs') as temp:
        p=Path(temp);plain=execute(p/'plain')
        try:execute(p/'resume',interrupted=True)
        except InterruptedError:pass
        else:raise ValueError('Recovery check did not interrupt.')
        if plain!=execute(p/'resume',resume=True):raise ValueError('Recovery changed learner/diagnostics/episode records.')
    return dict(passed=True,environment_steps=24,updates=21,device='cpu',formal_training=False)


def preflight():
    import unittest
    torch.set_num_threads(1)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_execution_consistency.py')
    test_result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not test_result.wasSuccessful():raise ValueError('Single-factor action/eligibility/parity tests failed.')
    config,inputs=configure();old=legacy.load_config(legacy.CONFIG);pairs=[]
    for seed in SEEDS:
        _,reference=common.audit_run(legacy.load_config(common.CONFIG),'advice_bound_margin',seed)
        agent=make_agent(config,inputs[0],seed,'cpu')
        digest=legacy.runtime.state_digest(agent.training_state_dict())
        if digest!=reference['initial_state_sha256']:raise ValueError('Full initial learner/optimizer/target/advice RNG pairing changed.')
        baseline=legacy.make_agent(old,inputs[0],seed,'advice_bound_margin','cpu')
        for t in range(200001):
            agent.training_action_steps=baseline.training_action_steps=t
            if agent.advice_probability(t)!=baseline.advice_probability(t) or agent.imitation_weight()!=baseline.imitation_weight():
                raise ValueError('Do not use staged withdrawal or change supervision clock.')
            if t>=100000 and (agent.advice_probability(t)!=0 or agent.imitation_weight()!=0):raise ValueError('100k exit failed.')
        if legacy.effective_training(config,seed,'formal')!=legacy.effective_training(old,seed,'formal'):
            raise ValueError('Replay/exploration/update/online budget changed.')
        factories=[legacy.runtime.make_factory(c,*inputs,method='unguided',seed=seed) for c in (config,old)]
        for i in range(3):
            envs=[f(inputs[0],max_steps=6,window_size=15,reward_config=legacy._reward_config(config),terminate_on_collision=True) for f in factories]
            if envs[0].dynamic_obstacles!=envs[1].dynamic_obstacles or envs[0].scenario_id!=envs[1].scenario_id:
                raise ValueError('Episode-index scene pairing differs.')
        pairs.append(dict(seed=seed,initial_state_sha256=digest,original_initialization_paired=True))
    validation,_=legacy.runtime.check(config,inputs,list(SEEDS))
    report=dict(passed=True,unit_checks=test_result.testsRun,initializations=pairs,
        small_recovery_check=recovery_check(config,inputs),validation=validation,
        real_transition_action_preserved=True,teacher_action_is_independent=True,
        baseline_source_unchanged=True,advice_and_supervision_exit=100000,independent500_used=False,
        pretraining_updates=0,online_environment_steps=200000,online_updates=199501,
        sample_count_change_is_treatment=True,no_weight_or_sampling_compensation=True,
        random_alignment='Identical per-seed initial learner/target/optimizer and learner/advice RNG. Same physical scenario generation by episode index. Policy divergence changes episode lengths, collected transitions and replay sampling contents.')
    frozen=dict(config=config,source_sha256=source_hashes(),checks=report,
        validation_sha256=common.sha256(ROOT/config['dataset']['validation_reference']))
    target=ROOT/config['execution_consistency']['registration_root']/'freeze_manifest.json'
    if target.exists() and common.read_json(target)!=frozen:raise ValueError('Frozen protocol changed; preserve old files.')
    if not target.exists():write_json(frozen,target)
    directory=legacy.runtime.unique_check_dir(config,'check');write_json(report,directory/'check_report.json')
    print(directory,flush=True);return frozen


def run_one(seed,resume=False):
    config,inputs=configure()
    frozen_path=ROOT/config['execution_consistency']['registration_root']/'freeze_manifest.json'
    frozen=common.read_json(frozen_path)
    if frozen['config']!=config or frozen['source_sha256']!=source_hashes():raise ValueError('Frozen training source/config changed.')
    directory=ROOT/config['experiment']['output_root']/METHOD/f'seed_{seed}'
    if (directory/'completion.json').exists():
        if resume:print(f'[seed{seed}] already complete; preserved.',flush=True);return
        raise FileExistsError(f'Preserve existing result: {directory}')
    if directory.exists() and not resume:raise FileExistsError(f'Use explicit --resume for incomplete run: {directory}')
    if resume and not (directory/'recovery_latest.pth').is_file():raise FileNotFoundError('No full recovery checkpoint.')
    directory.mkdir(parents=True,exist_ok=resume)
    agent=make_agent(config,inputs[0],seed)
    initial=next(r['initial_state_sha256'] for r in frozen['checks']['initializations'] if r['seed']==seed)
    if legacy.runtime.state_digest(agent.training_state_dict())!=initial:raise ValueError('Runtime paired initialization mismatch.')
    factory=legacy.runtime.make_factory(config,*inputs,method='unguided',seed=seed)
    manager=Recovery(directory,agent,factory,STAGE_LOCALS,resume=resume,
        identity=dict(seed=seed,freeze_sha256=common.sha256(frozen_path)))
    step=manager.last_step
    curves=[r for r in load_csv(directory/'validation_curve.csv') if int(r['environment_steps_total'])<=step] if resume else []
    details=[r for r in load_csv(directory/'validation_details.csv') if int(r['environment_steps_total'])<=step] if resume else []
    behaviors=[r for r in load_csv(directory/'failure_behaviors.csv') if int(r['environment_steps_total'])<=step] if resume else []
    training=legacy.effective_training(config,seed,'formal');distances=legacy.static_distances(inputs[0])
    metadata=dict(config=config,method=METHOD,seed=seed,effective_training=asdict(training),
        source_sha256=frozen['source_sha256'],initial_state_sha256=initial,freeze_manifest_sha256=common.sha256(frozen_path),
        model_selection='final_200000_only',test_data_used=False,evaluation_advice=False,independent500_used=False,
        foundation_loaded=False,pretraining_updates=0,demo_transition_count=0,online_replay_capacity=10000,
        teacher_action_separate_from_transition_action=True,random_alignment=frozen['checks']['random_alignment'])
    write_json(metadata,directory/'run_manifest.json')
    write_json(dict(complete=False,running=True,environment_steps=step),directory/'status.json')
    def diagnostics():
        write_records_csv(agent.advice_records(),directory/'advice_budget.csv')
        write_records_csv(agent.repair_records(),directory/'value_learning.csv')
        write_records_csv(agent.supervision_records(),directory/'supervision_alignment.csv')
    def validate_model(current):
        summary,rows,traces=legacy.old.evaluate(config,inputs,seed,agent)
        failures=[t for t in traces if not t['safe_success']]
        curves.append(dict(environment_steps_total=current,**summary))
        details.extend(dict(environment_steps_total=current,**r) for r in rows)
        behaviors.extend(dict(environment_steps_total=current,**legacy.trace_behavior(t,distances)) for t in failures)
        write_records_csv(curves,directory/'validation_curve.csv');write_records_csv(details,directory/'validation_details.csv')
        write_records_csv(behaviors,directory/'failure_behaviors.csv')
        write_json(failures,directory/f'validation_failures_{current:06d}.json')
        agent.save_weights(directory/f'model_step_{current:06d}.pth');diagnostics()
        write_json(dict(complete=False,running=True,environment_steps=current,
            recoverable_environment_steps=manager.last_step),directory/'status.json')
        print(f'[{METHOD} seed{seed}] step={current} autonomous_success={summary["safe_success_rate"]:.1%}',flush=True)
    def progress(_episode,records,full):
        if full:validate_model(records[-1]['environment_steps_total']);write_records_csv(list(records),directory/'training.csv')
    clock=perf_counter()
    if not resume:validate_model(0)
    try:
        result=instrument(legacy.train_d3qn,STAGE_LOCALS)([inputs[0]],agent,training,
            reward_config=legacy._reward_config(config),environment_factory=factory,progress_callback=progress,_recovery=manager)
        if result.environment_steps!=200000 or result.gradient_updates!=199501 or agent.update_steps!=199501 or agent.training_action_steps!=200000:
            raise ValueError('Formal budget mismatch.')
        if any(r['requested'] or r['applied'] for r in agent.advice_records() if r['bin_start_step']>=100001):raise ValueError('Advice persisted after100k.')
        if any(r['teacher_label_samples'] for r in agent.repair_records() if r['bin_start_step']>=100001):raise ValueError('Supervision persisted after100k.')
        if int(curves[-1]['environment_steps_total'])!=200000:validate_model(200000)
        agent.save_weights(directory/'model_final.pth');diagnostics()
        write_records_csv(list(result.episode_records),directory/'training.csv')
        write_json(dict(**metadata,complete=True,environment_steps=200000,gradient_updates=199501,online_updates=199501,
            final_model_sha256=common.sha256(directory/'model_final.pth'),final_validation=curves[-1],
            online_training_seconds=manager.prior_training_seconds+result.training_seconds,
            validation_and_recovery_seconds=result.progress_callback_seconds,session_seconds=perf_counter()-clock),directory/'result.json')
        write_json(dict(complete=True,environment_steps=200000),directory/'completion.json')
        write_json(dict(complete=True,running=False,environment_steps=200000,
            recoverable_environment_steps=manager.last_step),directory/'status.json')
        print(f'[{METHOD} seed{seed}] COMPLETE {directory}',flush=True)
    except BaseException as error:
        write_json(dict(complete=False,running=False,recoverable_environment_steps=manager.last_step,error=repr(error),
            resume_command=f'python scripts/run_execution_consistency.py --train --seeds {seed} --resume'),directory/'status.json')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seeds',nargs='+',type=int,choices=list(SEEDS),default=list(SEEDS));args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):parser.error('No duplicate seeds.')
    if not args.train:preflight();return
    for seed in args.seeds:run_one(seed,args.resume)


if __name__=='__main__':main()
