"""Four paired seed3/4 replications; preserve completed seed0/1/2 and registered sources."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import run_staged_withdrawal_pilot as stage
import run_dqfd_no_pretrain as no_pretrain
import dqfd_comparison_common as dq
from run_dqfd_comparison import UpdateDiagnostics
from replication_recovery import Recovery, STAGE_LOCALS, DQFD_LOCALS, instrument
from astar_d3qn.agents.astar_staged_withdrawal import AStarStagedWithdrawalAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.training.dqfd import train_online, new_replay
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything
from analyze_runtime_path_pilot import load_csv

ROOT = stage.common.ROOT
CONFIGS = dict(staged=ROOT/'configs/whole_map_91701_staged_withdrawal_seeds34_v1.yaml',
    dqfd=ROOT/'configs/whole_map_91701_dqfd_no_pretrain_seeds34_v1.yaml')
METHODS = dict(staged=stage.METHOD, dqfd=no_pretrain.METHOD)
OLD_CONFIGS = dict(staged=stage.CONFIG, dqfd=no_pretrain.CONFIG)
common = stage.common
legacy = common.legacy


def make_stage(config, problem, seed, device=None):
    seed_everything(seed)
    values = dict(config['agent'])
    if device:
        values['device'] = device
    ad, repair, schedule = config['training_advice'], config['value_repair'], config['staged_withdrawal']
    return AStarStagedWithdrawalAgent(D3QNConfig((5,15,15),4,5,seed=seed,**values), problem,
        tail_start=schedule['advice_tail_start_steps'], advice_end=schedule['advice_end_steps'],
        lower=repair['target_lower'], upper=repair['target_upper'], margin=repair['margin'],
        margin_weight=repair['margin_weight'], probability=ad['probability'],
        decay_steps=ad['decay_environment_steps'], risk_radius=ad['risk_radius'], rng_offset=ad['rng_seed_offset'])


def source_hashes():
    sources = dict(stage.source_hashes())
    sources.update(dq.source_hashes())
    for path in (Path(__file__), ROOT/'scripts/replication_recovery.py', no_pretrain.CONFIG,
                 ROOT/'scripts/run_dqfd_no_pretrain.py', *CONFIGS.values()):
        sources[str(path.relative_to(ROOT))] = common.sha256(path)
    return sources


def configure(arm):
    config = legacy.load_config(CONFIGS[arm])
    old = legacy.load_config(OLD_CONFIGS[arm])
    if {k:v for k,v in config.items() if k not in ('experiment','replication')} != {
            k:v for k,v in old.items() if k != 'experiment'}:
        raise ValueError('Replication changes only output paths and seed selection.')
    if config['replication']['new_seeds'] != [3,4] or not config['replication']['reuse_seed0']:
        raise ValueError('Only the four specified seed3/4 replications are authorized.')
    inputs = stage.validate(old) if arm == 'staged' else no_pretrain.validate(old)[1]
    return config, old, inputs


def parity_check(config, inputs):
    """Small CPU check: unchanged loop vs instrumented, interrupted/resumed vs uninterrupted."""
    from tempfile import TemporaryDirectory
    from astar_d3qn.training import trainer
    reports = []
    for arm, names, function in [('staged', STAGE_LOCALS, legacy.train_d3qn),
                                ('dqfd', DQFD_LOCALS, train_online)]:
        cfg, _, scene_inputs = configure(arm)
        training = replace(legacy.effective_training(cfg,3,'formal'), max_environment_steps=24,
            max_steps=6, learning_starts=4, batch_size=4, progress_interval_environment_steps=6,
            epsilon_decay_environment_steps=24)
        def execute(directory, hooked, interrupted=False, resume=False):
            agent = make_stage(cfg,scene_inputs[0],3,'cpu') if arm == 'staged' else dq.make_agent(cfg,scene_inputs[0],3,'dqfd','cpu')
            factory = legacy.runtime.make_factory(cfg,*scene_inputs,method='unguided',seed=3)
            if arm == 'staged':
                factory = stage.LabelCensusFactory(factory,agent)
            manager = Recovery(directory,agent,factory,names,resume=resume,
                identity='parity',checkpoint_interval=6,stop_after=12 if interrupted else None)
            kwargs = dict(_recovery=manager) if hooked else {}
            loop = instrument(function,names) if hooked else function
            if arm == 'staged':
                result = loop([scene_inputs[0]],agent,training,reward_config=legacy._reward_config(cfg),
                    environment_factory=factory,progress_callback=lambda *args:None,**kwargs)
            else:
                frozen_cfg=legacy.load_config(dq.CONFIG)
                demos,_=dq.load_demonstrations(frozen_cfg)
                replay=new_replay(demos,cfg['comparison'],3,training.replay_capacity)
                result=loop(scene_inputs[0],agent,training,replay,legacy._reward_config(cfg),factory,
                    lambda *args:None,**kwargs)
            return legacy.runtime.state_digest(agent.training_state_dict()),result.episode_records,factory
        with TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            path=Path(temporary)
            plain=execute(path/'plain',False)
            hooked=execute(path/'hook',True)
            try:
                execute(path/'resume',True,interrupted=True)
            except InterruptedError:
                pass
            else:
                raise ValueError('Interruption check did not stop.')
            resumed=execute(path/'resume',True,resume=True)
            if plain[:2]!=hooked[:2] or hooked[:2]!=resumed[:2]:
                raise ValueError(f'{arm} hook/resume changed learner or training records.')
            reports.append(dict(arm=arm,environment_steps=24,update_count=21,
                unchanged_loop_parity=True,interrupted_resume_parity=True,device='cpu'))
    return reports


def preflight():
    torch.set_num_threads(1)
    pairs=[]
    for arm in ('staged','dqfd'):
        cfg,old,inputs=configure(arm)
        if arm=='dqfd':
            *_,report=no_pretrain.prepare(old,[3,4])
            pairs.extend(dict(arm=arm,**row) for row in report['initializations'])
        for seed in (3,4):
            reference_config=legacy.load_config(legacy.CONFIG if arm=='staged' else dq.CONFIG)
            if arm=='staged':
                _,result=common.audit_run(legacy.load_config(common.CONFIG),'advice_bound_margin',seed)
                agent=make_stage(cfg,inputs[0],seed,'cpu')
                initial=legacy.runtime.state_digest(agent.training_state_dict())
                if initial != result['initial_state_sha256']:
                    raise ValueError('Historical full initial-state pairing mismatch.')
                baseline=legacy.make_agent(reference_config,inputs[0],seed,'advice_bound_margin','cpu')
                for t in range(200001):
                    agent.training_action_steps=baseline.training_action_steps=t
                    if agent.imitation_weight()!=baseline.imitation_weight() or (
                        t<=60000 and agent.advice_probability(t)!=baseline.advice_probability(t)) or (
                        t>=80000 and agent.advice_probability(t)!=0) or (
                        t>=100000 and agent.imitation_weight()!=0):
                        raise ValueError(f'Exit schedule changed at {t}.')
                # epsilon=0 must neither invoke the teacher nor advance its training clock.
                agent=make_stage(cfg,inputs[0],seed,'cpu')
                env=legacy.runtime.make_factory(cfg,*inputs,method='unguided',seed=seed)(inputs[0],
                    max_steps=cfg['environment']['max_steps'],window_size=15,reward_config=legacy._reward_config(cfg),terminate_on_collision=True)
                s=env.reset();before=legacy.runtime.state_digest(agent.training_state_dict())
                agent.select_action(s,0,list(np.flatnonzero(env.action_mask(True))))
                if before!=legacy.runtime.state_digest(agent.training_state_dict()) or agent.planner_calls:
                    raise ValueError('Autonomous evaluation used teacher or changed training state.')
                pairs.append(dict(arm=arm,seed=seed,initial_state_sha256=initial,full_initial_state_paired=True))
            if legacy.effective_training(cfg,seed,'formal') != legacy.effective_training(reference_config,seed,'formal'):
                raise ValueError('Effective online settings differ.')
            # Same physical scene per episode index; policy divergence can change episode lengths.
            factories=[legacy.runtime.make_factory(c,*inputs,method='unguided',seed=seed) for c in (cfg,reference_config)]
            for i in range(3):
                scenes=[f(inputs[0],max_steps=6,window_size=15,reward_config=legacy._reward_config(cfg),terminate_on_collision=True) for f in factories]
                if scenes[0].dynamic_obstacles != scenes[1].dynamic_obstacles or scenes[0].scenario_id != scenes[1].scenario_id:
                    raise ValueError('Episode-index scene pairing differs.')
    cfg,_,inputs=configure('staged')
    # Original runtime checker verifies all 50 serialized validation scenes and static masks.
    validation_report,_=legacy.runtime.check(cfg,inputs,[3,4])
    report=dict(passed=True,paired_initializations=pairs,
        loop_and_recovery_checks=parity_check(cfg,inputs),
        validation=validation_report,seed0_reused=True,final500_used=False,
        advice_off_from=80000,supervision_off_from=100000,
        scheduled_advice_area_reduction=.08,demonstration_count=7260,
        pretraining_updates=0,expected_online_updates=199501,
        random_alignment='Per seed: identical pre-training network/target/optimizer/learner RNG; episode-index scene sampling RNG identical. DQfD baseline offline updates advance optimizer/target/PER priorities and replay RNG; those differences are part of removing pretraining. Policy divergence changes trajectories and replay contents.')
    for arm in ('staged','dqfd'):
        config,_,_=configure(arm)
        frozen=dict(config=config,source_sha256=source_hashes(),check=report,
            validation_sha256=common.sha256(ROOT/config['dataset']['validation_reference']))
        path=ROOT/config['replication']['registration_root']/'freeze_manifest.json'
        if path.exists() and common.read_json(path)!=frozen:
            raise ValueError('Frozen replication changed.')
        if not path.exists():write_json(frozen,path)
    return report


def run_one(arm,seed,resume=False):
    config,old,inputs=configure(arm)
    frozen_path=ROOT/config['replication']['registration_root']/'freeze_manifest.json'
    frozen=common.read_json(frozen_path)
    if frozen['source_sha256']!=source_hashes() or frozen['config']!=config:
        raise ValueError('Frozen sources/config changed.')
    method=METHODS[arm]
    directory=ROOT/config['experiment']['output_root']/method/f'seed_{seed}'
    if (directory/'completion.json').exists():
        if resume:
            print(f'[{arm} seed{seed}] already complete; preserved.',flush=True);return
        raise FileExistsError(f'Completed run preserved: {directory}')
    if directory.exists() and not resume:raise FileExistsError(f'Use --resume for incomplete run: {directory}')
    if resume and not (directory/'recovery_latest.pth').exists():raise FileNotFoundError('No full recovery checkpoint.')
    directory.mkdir(parents=True,exist_ok=resume)
    agent=make_stage(config,inputs[0],seed) if arm=='staged' else dq.make_agent(config,inputs[0],seed,'dqfd')
    digest=legacy.runtime.state_digest(agent.training_state_dict()) if arm=='staged' else dq.paired_digest(agent)
    pair=next(v for v in frozen['check']['paired_initializations'] if v['arm']==arm and v['seed']==seed)
    expected=pair.get('initial_state_sha256',pair.get('initial_learner_sha256'))
    if digest!=expected:raise ValueError('Runtime initialization pairing mismatch.')
    factory=legacy.runtime.make_factory(config,*inputs,method='unguided',seed=seed)
    diagnostics=None
    if arm=='staged':factory=stage.LabelCensusFactory(factory,agent)
    else:diagnostics=UpdateDiagnostics()
    names=STAGE_LOCALS if arm=='staged' else DQFD_LOCALS
    manager=Recovery(directory,agent,factory,names,diagnostics=diagnostics,resume=resume,
        identity=dict(arm=arm,seed=seed,freeze_sha256=common.sha256(frozen_path)))
    step=manager.last_step
    curves=[r for r in load_csv(directory/'validation_curve.csv') if int(r['environment_steps_total'])<=step] if resume else []
    details=[r for r in load_csv(directory/'validation_details.csv') if int(r['environment_steps_total'])<=step] if resume else []
    behaviors=[r for r in load_csv(directory/'failure_behaviors.csv') if int(r['environment_steps_total'])<=step] if resume else []
    training=legacy.effective_training(config,seed,'formal')
    metadata=dict(config=config,method=method,seed=seed,effective_training=asdict(training),
        initial_state_sha256=digest,initial_learner_sha256=digest,source_sha256=frozen['source_sha256'],
        freeze_manifest_sha256=common.sha256(frozen_path),test_data_used=False,evaluation_advice=False,
        independent500_used=False,model_selection='final_200000_only',
        random_alignment=frozen['check']['random_alignment'],
        recovery='Episode-boundary full learner/optimizer/replay/collector/RNG/factory/diagnostics checkpoint, atomically replaced after validation.')
    write_json(metadata,directory/'run_manifest.json')
    write_json(dict(complete=False,running=True,resumed=resume,environment_steps=step),directory/'status.json')
    distances=legacy.static_distances(inputs[0])
    def save_diagnostics():
        if arm=='staged':
            for records,name in ((agent.advice_records(),'advice_budget'),(agent.repair_records(),'value_learning'),
                (agent.td_records(),'true_training_td'),(factory.records(),'collection_label_census'),(factory.scenes,'training_scenes')):
                write_records_csv(records,directory/f'{name}.csv')
        else:write_records_csv(diagnostics.records(),directory/'dqfd_learning.csv')
    def validate_model(current):
        summary,rows,traces=legacy.old.evaluate(config,inputs,seed,agent)
        failures=[t for t in traces if not t['safe_success']]
        curves.append(dict(environment_steps_total=current,**summary))
        details.extend(dict(environment_steps_total=current,**r) for r in rows)
        behaviors.extend(dict(environment_steps_total=current,**legacy.trace_behavior(t,distances)) for t in failures)
        write_records_csv(curves,directory/'validation_curve.csv')
        write_records_csv(details,directory/'validation_details.csv')
        write_records_csv(behaviors,directory/'failure_behaviors.csv')
        write_json(failures,directory/f'validation_failures_{current:06d}.json')
        agent.save_weights(directory/f'model_step_{current:06d}.pth');save_diagnostics()
        write_json(dict(complete=False,running=True,environment_steps=current,
            recoverable_environment_steps=manager.last_step),directory/'status.json')
        print(f'[{arm} seed{seed}] step={current} autonomous_success={summary["safe_success_rate"]:.1%}',flush=True)
    def progress(_episode,records,full):
        if full:
            validate_model(records[-1]['environment_steps_total'])
            write_records_csv(list(records),directory/'training.csv')
    start=perf_counter()
    if not resume:validate_model(0)
    try:
        if arm=='staged':
            result=instrument(legacy.train_d3qn,names)([inputs[0]],agent,training,demonstrations=(),
                reward_config=legacy._reward_config(config),environment_factory=factory,
                progress_callback=progress,_recovery=manager)
        else:
            demos,demo_manifest=dq.load_demonstrations(legacy.load_config(dq.CONFIG))
            replay=new_replay(demos,config['comparison'],seed,training.replay_capacity)
            result=instrument(train_online,names)(inputs[0],agent,training,replay,legacy._reward_config(config),
                factory,progress,diagnostics,_recovery=manager)
        if result.environment_steps!=200000 or result.gradient_updates!=199501 or agent.update_steps!=199501:
            raise ValueError('Training/update budget mismatch.')
        if arm=='staged':
            if agent.training_action_steps!=200000 or any(r['requested'] or r['applied'] for r in agent.advice_records() if r['bin_start_step']>=80001):
                raise ValueError('Advice exit or training clock mismatch.')
            if any(r['teacher_label_samples'] for r in agent.repair_records() if r['bin_start_step']>=100001):
                raise ValueError('Supervision did not exit at 100k.')
        if int(curves[-1]['environment_steps_total'])!=200000:validate_model(200000)
        agent.save_weights(directory/'model_final.pth');save_diagnostics()
        write_records_csv(list(result.episode_records),directory/'training.csv')
        summary=dict(**metadata,environment_steps=200000,gradient_updates=199501,online_updates=199501,
            pretraining_updates=0,permanent_demo_retained_count=7260 if arm=='dqfd' else 0,
            online_replay_capacity=10000,final_validation=curves[-1],
            final_model_sha256=common.sha256(directory/'model_final.pth'),
            online_training_seconds=manager.prior_training_seconds+result.training_seconds,
            validation_and_recovery_seconds=result.progress_callback_seconds,total_seconds_this_session=perf_counter()-start,
            planner_calls=dict(agent.planner_calls) if arm=='staged' else {},
            scheduled_advice_area_reduction=.08 if arm=='staged' else None)
        if arm=='dqfd':
            f=dq.frozen(legacy.load_config(dq.CONFIG))
            summary.update(demonstration_sha256=f['files_sha256']['demonstrations.npz'],online_astar_lookups=0,online_astar_searches=0)
        write_json(summary,directory/'result.json')
        write_json(dict(complete=True,running=False,environment_steps=200000,gradient_updates=199501,
            recovery_checkpoint=str(manager.path.relative_to(ROOT)),recoverable_environment_steps=manager.last_step),directory/'status.json')
        write_json(dict(complete=True,environment_steps=200000),directory/'completion.json')
        print(f'[{arm} seed{seed}] COMPLETE {directory}',flush=True)
    except BaseException as error:
        write_json(dict(complete=False,running=False,recoverable_environment_steps=manager.last_step,
            error=repr(error),resume_command=f'python scripts/run_replication_seeds34.py --train --arms {arm} --seeds {seed} --resume'),directory/'status.json')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--arms',nargs='+',choices=['staged','dqfd'],default=['staged','dqfd'])
    parser.add_argument('--seeds',nargs='+',type=int,choices=[3,4],default=[3,4])
    args=parser.parse_args()
    if len(set(args.arms))!=len(args.arms) or len(set(args.seeds))!=len(args.seeds):parser.error('No duplicate runs.')
    if not args.train:
        report=preflight()
        config,_,_=configure('staged')
        output=legacy.runtime.unique_check_dir(config,'check')
        write_json(report,output/'check_report.json');print(output);return
    # A preflight freeze is mandatory; do not silently register altered code while training.
    for arm in args.arms:
        for seed in args.seeds:run_one(arm,seed,resume=args.resume)


if __name__=='__main__':main()
