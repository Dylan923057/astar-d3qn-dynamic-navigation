"""One-factor DQfD control: omit offline updates, retain every online component.

Default: zero-update preflight. --train is explicit authorization for online runs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

import dqfd_comparison_common as common
from run_dqfd_comparison import UpdateDiagnostics
from astar_d3qn.training.dqfd import new_replay, train_online
from astar_d3qn.utils.io import write_json, write_records_csv

CONFIG=common.ROOT/'configs/whole_map_91701_dqfd_no_pretrain_v1.yaml'
METHOD='dqfd_no_pretrain'


def validate(config):
    original=common.legacy.load_config(common.CONFIG)
    expected=common.legacy.load_config(CONFIG)
    if config!=expected:raise ValueError('Use the fixed no-pretraining configuration.')
    shared={k:v for k,v in config.items() if k not in ('experiment','no_pretrain_control')}
    baseline={k:v for k,v in original.items() if k!='experiment'}
    baseline['comparison']=dict(baseline['comparison'],pretraining_updates=0)
    if shared!=baseline:raise ValueError('Only offline update count may change.')
    spec=config['no_pretrain_control']
    if (spec['method']!=METHOD or spec['regenerate_demonstrations'] or
        spec['online_environment_steps']!=200000 or spec['online_gradient_updates']!=199501 or
        spec['total_gradient_updates']!=199501 or spec['final_checkpoint']!='final_200000_only'):
        raise ValueError('Unexpected control budget/selection protocol.')
    return original,common.validate_config(original)


def prepare(config,seeds):
    original,inputs=validate(config)
    manifest=common.frozen(original)
    demos,_=common.load_demonstrations(original)
    paired=[]
    for seed in seeds:
        baseline=common.baseline_digest(original,inputs[0],seed)
        agent=common.make_agent(config,inputs[0],seed,'dqfd','cpu')
        if common.paired_digest(agent)!=baseline or agent.update_steps!=0 or agent.optimizer.state:
            raise ValueError('Control must start from paired untrained learner.')
        if agent.bound_enabled:raise ValueError('Do not add target clipping to this control.')
        replay=new_replay(demos,config['comparison'],seed,config['training']['replay_capacity'])
        if replay.demo_count!=7260 or len(replay)!=7260 or replay.cursor!=0:
            raise ValueError('Permanent frozen demos or initial online partition changed.')
        # Verify every effective online setting, not just the declared step count.
        if common.legacy.effective_training(config,seed,'formal')!=common.legacy.effective_training(original,seed,'formal'):
            raise ValueError('Effective online training changed.')
        paired.append(dict(seed=seed,initial_learner_sha256=baseline,updates=0,optimizer_state_empty=True,
            permanent_demo_count=replay.demo_count,online_capacity=replay.online_capacity))
    registration=dict(config_sha256=common.canonical_hash(config),
        scripts_sha256={str(Path(__file__).relative_to(common.ROOT)):common.sha256(Path(__file__)),
                       str(CONFIG.relative_to(common.ROOT)):common.sha256(CONFIG)},
        original_source_sha256=manifest['source_sha256'],
        original_freeze_sha256=common.sha256(common.artifact_root(original)/'freeze_manifest.json'),
        demonstration_sha256=manifest['files_sha256']['demonstrations.npz'],
        final_scene_sha256=manifest['final_scene_sha256'],
        only_algorithm_change='pretraining_updates: 20000 -> 0; no state learned/consumed offline',
        all_online_settings_unchanged=True,evaluation=dict(epsilon=0,teacher=False,
            dynamic_protection=False,static_mask=True,checkpoint='final_200000_only'),
        random_alignment=dict(paired_before_pretraining=True,scene_pairing='episode index',
            replay_rng='same seed; baseline advances RNG/TD priorities through offline updates, control does not',
            optimizer_target_and_priorities='offline-induced differences are part of removing pretraining'),
        paired_initializations={str(s):common.baseline_digest(original,inputs[0],s) for s in common.SEEDS})
    target=common.ROOT/config['no_pretrain_control']['registration_root']/'freeze_manifest.json'
    if target.exists():
        if common.read_json(target)!=registration:raise ValueError('Frozen control changed; create a new protocol.')
    else:write_json(registration,target)
    return inputs,demos,manifest,dict(passed=True,formal_training_started=False,gradient_updates=0,
        original_results_preserved=True,demonstrations_regenerated=False,final500_evaluated=False,
        comparison='same online interactions, 20000 fewer total updates',initializations=paired,
        demonstration_sha256=registration['demonstration_sha256'])


def run(config,seeds):
    planned=[(s,common.ROOT/config['experiment']['output_root']/METHOD/f'seed_{s}') for s in seeds]
    if any(d.exists() for _,d in planned):raise FileExistsError('Refusing to overwrite/resume an existing control run.')
    inputs,demos,manifest,_=prepare(config,seeds)
    for seed,directory in planned:
        agent=common.make_agent(config,inputs[0],seed,'dqfd')
        if common.paired_digest(agent)!=common.baseline_digest(config,inputs[0],seed):
            raise ValueError('Runtime paired initialization mismatch.')
        training=common.legacy.effective_training(config,seed,'formal')
        replay=new_replay(demos,config['comparison'],seed,training.replay_capacity)
        directory.mkdir(parents=True,exist_ok=False)
        diagnostics=UpdateDiagnostics();curves=[];details=[]
        def validate_point(step):
            summary,rows,traces=common.legacy.old.evaluate(config,inputs,seed,agent)
            curves.append(dict(environment_steps_total=step,**summary))
            details.extend(dict(environment_steps_total=step,**r) for r in rows)
            write_records_csv(curves,directory/'validation_curve.csv')
            write_records_csv(details,directory/'validation_details.csv')
            write_json([t for t in traces if not t['safe_success']],directory/f'validation_failures_{step:06d}.json')
            write_records_csv(diagnostics.records(),directory/'dqfd_learning.csv')
            agent.save_weights(directory/f'model_step_{step:06d}.pth')
            print(f'[{METHOD} seed={seed}] step={step} autonomous_success={summary["safe_success_rate"]:.1%}',flush=True)
        clock=perf_counter();validate_point(0)  # Random initialization, unlike original DQfD's post-pretrain point 0.
        def progress(_episode,records,full_evaluation):
            if full_evaluation:
                validate_point(records[-1]['environment_steps_total'])
                write_records_csv(list(records),directory/'training.csv')
        factory=common.legacy.runtime.make_factory(config,*inputs,method='unguided',seed=seed)
        # No call to pretrain(), no reset/fabrication of offline optimizer/PER state.
        result=train_online(inputs[0],agent,training,replay,common.legacy._reward_config(config),factory,progress,diagnostics)
        if (result.environment_steps!=200000 or result.gradient_updates!=199501 or
            agent.update_steps!=199501 or replay.demo_count!=7260):raise ValueError('Online budget/demo retention changed.')
        if curves[-1]['environment_steps_total']!=200000:validate_point(200000)
        agent.save_weights(directory/'model_final.pth')
        write_records_csv(list(result.episode_records),directory/'training.csv')
        write_records_csv(diagnostics.records(),directory/'dqfd_learning.csv')
        write_json(dict(complete=True,config=config,method=METHOD,seed=seed,
            effective_training=asdict(training),environment_steps=200000,pretraining_updates=0,
            online_updates=199501,gradient_updates=199501,permanent_demo_retained_count=replay.demo_count,
            online_replay_capacity=training.replay_capacity,
            initial_learner_sha256=common.baseline_digest(config,inputs[0],seed),
            demonstration_sha256=manifest['files_sha256']['demonstrations.npz'],
            original_demonstration_generation_shared_seconds=manifest['demonstration_generation_seconds'],
            pretraining_seconds=0,online_training_seconds=result.training_seconds,
            online_validation_and_callback_seconds=result.progress_callback_seconds,total_seconds=perf_counter()-clock,
            online_astar_lookups=0,online_astar_searches=0,evaluation_advice=False,
            final500_used_for_training_or_scheduling=False,model_selection='final_200000_only',
            freeze_manifest_sha256=common.sha256(common.ROOT/config['no_pretrain_control']['registration_root']/'freeze_manifest.json'),
            final_model_sha256=common.sha256(directory/'model_final.pth'),final_validation=curves[-1]),directory/'result.json')
        write_json(dict(complete=True,environment_steps=200000),directory/'completion.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=list(common.SEEDS));a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or not set(a.seeds)<=set(common.SEEDS):p.error('Use unique paired seeds0-4.')
    config=common.legacy.load_config(CONFIG)
    if a.train:run(config,a.seeds)
    else:
        torch.set_num_threads(1)
        *_,report=prepare(config,a.seeds)
        directory=common.legacy.runtime.unique_check_dir(config,'check')
        write_json(report,directory/'check_report.json');print(directory)


if __name__=='__main__':main()
