"""Paired ASVC guidance floors. Default freezes/checks; --train runs new arms."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil
from time import perf_counter
import unittest
from unittest.mock import patch

import torch

import run_execution_consistency as base
from replication_recovery import Recovery, STAGE_LOCALS, instrument
from astar_d3qn.agents.persistent_teaching import PersistentTeachingAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.utils.seed import seed_everything

ROOT, common, legacy = base.ROOT, base.common, base.legacy
CONFIG = ROOT / 'configs/asvc_persistent_teaching_v1.yaml'
VARIANTS = ('supervision_floor', 'advice_floor', 'both_floors')
SEEDS = (0, 1, 2)
write_json, write_records_csv = base.write_json, legacy.write_records_csv


def configure():
    config = legacy.load_config(CONFIG)
    original, inputs = base.configure()
    excluded = ('experiment', 'persistent_teaching')
    if {k:v for k,v in config.items() if k not in excluded} != {k:v for k,v in original.items() if k != 'experiment'}:
        raise ValueError('Only guidance floors and independent metadata may change.')
    expected = dict(seeds=list(SEEDS), registration_root='data/asvc_persistent_teaching_v1',
        baseline_root='outputs/whole_map_91701_execution_consistency_v1/advice_bound_margin_all_eligible',
        environment_steps=200000, schedule='maximum_of_original_schedule_and_floor',
        variants=dict(supervision_floor=dict(advice_floor=0., supervision_floor=.1),
                      advice_floor=dict(advice_floor=.05, supervision_floor=0.),
                      both_floors=dict(advice_floor=.05, supervision_floor=.1)),
        evaluation_advice=False, independent500_used=False)
    if config['persistent_teaching'] != expected:
        raise ValueError('Use the registered three-arm, three-seed protocol.')
    return config, inputs


def make_agent(config, problem, seed, variant, device=None):
    seed_everything(seed)
    values = dict(config['agent'])
    if device: values['device'] = device
    ad, repair = config['training_advice'], config['value_repair']
    return PersistentTeachingAgent(D3QNConfig((5,15,15),4,5,seed=seed,**values), problem,
        repair_method='advice_bound_margin', lower=repair['target_lower'], upper=repair['target_upper'],
        margin=repair['margin'], margin_weight=repair['margin_weight'], probability=ad['probability'],
        decay_steps=ad['decay_environment_steps'], risk_radius=ad['risk_radius'], rng_offset=ad['rng_seed_offset'],
        **config['persistent_teaching']['variants'][variant])


def source_hashes():
    sources = base.source_hashes()
    paths = [CONFIG, Path(__file__), ROOT/'tests/test_persistent_teaching.py',
             *sorted((ROOT/'src/astar_d3qn').rglob('*.py'))]
    for path in paths: sources[str(path.relative_to(ROOT))] = common.sha256(path)
    return sources


def preflight():
    config, inputs = configure()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_persistent_teaching.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful(): raise ValueError('Persistent guidance tests failed.')
    initializations, sources = [], base.source_hashes()
    for seed in SEEDS:
        directory = ROOT/config['persistent_teaching']['baseline_root']/f'seed_{seed}'
        manifest = common.read_json(directory/'run_manifest.json')
        if not common.read_json(directory/'completion.json')['complete']:
            raise ValueError('Baseline incomplete.')
        if manifest['source_sha256'] != sources:
            raise ValueError('Historical baseline source differs; do not reuse without an audit.')
        expected = manifest['initial_state_sha256']
        for variant in VARIANTS:
            agent = make_agent(config, inputs[0], seed, variant, 'cpu')
            if legacy.runtime.state_digest(agent.training_state_dict()) != expected:
                raise ValueError('Initial learner/target/optimizer/RNG mismatch.')
        initializations.append(dict(seed=seed, initial_state_sha256=expected,
            baseline_manifest_sha256=common.sha256(directory/'run_manifest.json'),
            baseline_curve_sha256=common.sha256(directory/'validation_curve.csv')))
    validation, _ = legacy.runtime.check(config, inputs, list(SEEDS))
    small = deepcopy(config)
    small['training_advice']['decay_environment_steps'] = 12
    with patch.object(base, 'make_agent', side_effect=lambda c,p,s,d=None: make_agent(c,p,s,'both_floors',d)):
        recovery = base.recovery_check(small, inputs)
    report = dict(passed=True, tests=result.testsRun, initializations=initializations,
        recovery_check_crosses_exit=recovery, validation=validation,
        evaluation_advice=False, independent500_used=False,
        original_schedule_unchanged_until_step=89999,
        supervision_floor_reached_at=90000, advice_floor_reached_at=93750,
        planned_new_runs=9, environment_steps_per_run=200000,
        interpretation='Paired same-seed initialization; trajectories may diverge after intervention. Validation only, not new held-out evidence.')
    frozen = dict(config=config, checks=report, source_sha256=source_hashes(),
        validation_sha256=common.sha256(ROOT/config['dataset']['validation_reference']))
    path = ROOT/config['persistent_teaching']['registration_root']/'freeze_manifest.json'
    if path.exists():
        if common.read_json(path) != frozen: raise ValueError('Frozen protocol changed.')
    else:
        write_json(frozen, path)
        for name in frozen['source_sha256']:
            target = path.parent/'source_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
    write_json(report, ROOT/config['experiment']['check_root']/'preflight.json')
    print('PREFLIGHT PASSED: '+str(path), flush=True)


def run_one(variant, seed, resume=False):
    config, inputs = configure()
    freeze_path = ROOT/config['persistent_teaching']['registration_root']/'freeze_manifest.json'
    frozen = common.read_json(freeze_path)
    if frozen['config'] != config or frozen['source_sha256'] != source_hashes():
        raise ValueError('Frozen source/config changed.')
    directory = ROOT/config['experiment']['output_root']/variant/f'seed_{seed}'
    if (directory/'completion.json').exists():
        if resume: print(f'{variant} seed{seed}: complete, preserved.', flush=True); return
        raise FileExistsError(directory)
    if directory.exists() and not resume: raise FileExistsError('Use --resume for an incomplete run.')
    if resume and not (directory/'recovery_latest.pth').exists(): raise FileNotFoundError('No recovery checkpoint.')
    directory.mkdir(parents=True, exist_ok=resume)
    agent = make_agent(config, inputs[0], seed, variant)
    initial = next(r for r in frozen['checks']['initializations'] if r['seed']==seed)
    if legacy.runtime.state_digest(agent.training_state_dict()) != initial['initial_state_sha256']:
        raise ValueError('Runtime initialization mismatch.')
    factory = legacy.runtime.make_factory(config, *inputs, method='unguided', seed=seed)
    manager = Recovery(directory, agent, factory, STAGE_LOCALS, resume=resume,
        identity=dict(seed=seed, variant=variant, freeze_sha256=common.sha256(freeze_path)))
    def records(name):
        return [r for r in base.load_csv(directory/name) if int(r['environment_steps_total'])<=manager.last_step] if resume else []
    curves, details, behaviors = records('validation_curve.csv'), records('validation_details.csv'), records('failure_behaviors.csv')
    prefix = common.read_json(directory/'prefix_parity.json') if resume and (directory/'prefix_parity.json').exists() else []
    training = legacy.effective_training(config, seed, 'formal')
    metadata = dict(config=config, variant=variant, seed=seed, effective_training=asdict(training),
        source_sha256=frozen['source_sha256'], initial_state_sha256=initial['initial_state_sha256'],
        freeze_manifest_sha256=common.sha256(freeze_path), evaluation_advice=False,
        independent500_used=False, fresh_random_initialization=True, device=str(agent.device),
        torch_version=torch.__version__, baseline_curve_sha256=initial['baseline_curve_sha256'])
    write_json(metadata, directory/'run_manifest.json')
    distances = legacy.static_distances(inputs[0])
    def diagnostics():
        write_records_csv(agent.advice_records(), directory/'advice_budget.csv')
        write_records_csv(agent.repair_records(), directory/'value_learning.csv')
        write_records_csv(agent.supervision_records(), directory/'supervision_alignment.csv')
    def validate(step):
        summary, rows, traces = legacy.old.evaluate(config, inputs, seed, agent)
        failures = [t for t in traces if not t['safe_success']]
        curves.append(dict(environment_steps_total=step, **summary))
        details.extend(dict(environment_steps_total=step, **r) for r in rows)
        behaviors.extend(dict(environment_steps_total=step, **legacy.trace_behavior(t, distances)) for t in failures)
        write_records_csv(curves, directory/'validation_curve.csv')
        write_records_csv(details, directory/'validation_details.csv')
        write_records_csv(behaviors, directory/'failure_behaviors.csv')
        write_json(failures, directory/f'validation_failures_{step:06d}.json')
        agent.save_weights(directory/f'model_step_{step:06d}.pth')
        if step < 90000:
            baseline_path = ROOT/config['persistent_teaching']['baseline_root']/f'seed_{seed}'/f'model_step_{step:06d}.pth'
            same = baseline_path.exists() and legacy.runtime.state_digest(torch.load(baseline_path, map_location='cpu', weights_only=True)) == legacy.runtime.state_digest(agent.policy_network.state_dict())
            prefix.append(dict(step=step, exact_weights_match=bool(same)))
            write_json(prefix, directory/'prefix_parity.json')
            if not same: raise ValueError('Historical prefix differs before intervention; baseline reuse needs review.')
        diagnostics()
        write_json(dict(complete=False, running=True, environment_steps=step,
            recoverable_environment_steps=manager.last_step), directory/'status.json')
        print(f'[{variant} seed{seed}] step={step} success={summary["safe_success_rate"]:.1%} timeout={summary["timeout_rate"]:.1%}', flush=True)
    def progress(_episode, rows, full):
        if full:
            validate(int(rows[-1]['environment_steps_total']))
            write_records_csv(list(rows), directory/'training.csv')
    started = perf_counter()
    write_json(dict(complete=False, running=True, environment_steps=manager.last_step), directory/'status.json')
    try:
        if not resume: validate(0)
        result = instrument(legacy.train_d3qn, STAGE_LOCALS)([inputs[0]], agent, training,
            reward_config=legacy._reward_config(config), environment_factory=factory,
            progress_callback=progress, _recovery=manager)
        if result.environment_steps != 200000 or result.gradient_updates != 199501 or agent.training_action_steps != 200000:
            raise ValueError('Budget mismatch.')
        if int(curves[-1]['environment_steps_total']) != 200000: validate(200000)
        floors = config['persistent_teaching']['variants'][variant]
        for row in agent.advice_records():
            if row['bin_start_step'] >= 100001 and abs(row['mean_requested_probability']-floors['advice_floor']) > 1e-10:
                raise ValueError('Advice floor diagnostics inconsistent.')
        for row in agent.repair_records():
            if row['bin_start_step'] >= 100001 and abs(row['teacher_margin_weight_mean']-floors['supervision_floor']) > 1e-10:
                raise ValueError('Supervision floor diagnostics inconsistent.')
        agent.save_weights(directory/'model_final.pth'); diagnostics()
        write_records_csv(list(result.episode_records), directory/'training.csv')
        write_json(dict(**metadata, complete=True, environment_steps=200000, gradient_updates=199501,
            final_validation=curves[-1], final_model_sha256=common.sha256(directory/'model_final.pth'),
            online_training_seconds=manager.prior_training_seconds+result.training_seconds,
            session_seconds=perf_counter()-started), directory/'result.json')
        write_json(dict(complete=True, environment_steps=200000), directory/'completion.json')
        write_json(dict(complete=True, running=False, environment_steps=200000), directory/'status.json')
        print(f'[{variant} seed{seed}] COMPLETE', flush=True)
    except BaseException as error:
        write_json(dict(complete=False, running=False, error=repr(error),
            recoverable_environment_steps=manager.last_step), directory/'status.json')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error('Duplicate runs.')
    torch.set_num_threads(1)
    if not args.train: preflight(); return
    for seed in args.seeds:
        for variant in args.variants: run_one(variant, seed, args.resume)


if __name__ == '__main__': main()
