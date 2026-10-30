"""Independent registration for exactly three additional, unchanged CE seeds."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import FunctionType

import fixed_ce_common as original

ROOT, prior = original.ROOT, original.prior
NAME = 'asvc_fixed_ce_five_seeds_v1'
CONFIG = ROOT / 'configs' / (NAME + '.yaml')
DATA, OUT, RESULTS = (ROOT / folder / NAME for folder in ('data', 'outputs', 'results'))
SEEDS = NEW_SEEDS = (2, 3, 4)
ALL_SEEDS = (0, 1, 2, 3, 4)
REUSED_SEEDS = (0, 1)
read, sha, write_json, write_csv, digest = original.read, original.sha, original.write_json, original.write_csv, original.digest
# Rebind only the constructor's allowed seed list. Its function bytecode and learner class are unchanged.
make_agent = FunctionType(original.make_agent.__code__, dict(original.make_agent.__globals__, SEEDS=ALL_SEEDS),
                          'make_agent', original.make_agent.__defaults__, original.make_agent.__closure__)


def configure():
    parent, inputs = original.configure()
    expected = deepcopy(parent)
    expected['experiment'].update(name=NAME, protocol='unchanged_fixed_task_ce_extend_seeds234_v1',
        output_root=f'outputs/{NAME}', check_root=f'results/{NAME}')
    expected['ce_control'].update(seeds=list(NEW_SEEDS), maximum_new_training_runs=3,
        registration_root=f'data/{NAME}', statistics='paired_five_seed_mean_and_sample_sd')
    expected['ce_five_seed_extension'] = dict(pooled_seeds=list(ALL_SEEDS), reused_ce_seeds=list(REUSED_SEEDS),
        reused_ce_root='outputs/asvc_fixed_ce_v1', new_ce_seeds=list(NEW_SEEDS),
        baseline_registration='data/asvc_current_method.json', baseline_retrained=False,
        additional_environment_steps=600000, additional_gradient_updates=598503, independent500_used=False,
        parameter_changes='none', analysis_unit='paired_training_seed',
        decision='report_all_five_without_selecting_seeds_or_checkpoints')
    config = prior.legacy.load_config(CONFIG)
    if config != expected:
        raise ValueError('Only new seeds, independent paths and reporting metadata may change')
    return config, inputs


def baseline_directory(seed):
    registry = read(ROOT / 'data/asvc_current_method.json')
    return (ROOT / registry['final_models'][str(seed)]).parent


def analysis_directory(method, seed):
    if method == 'margin':
        return DATA / 'baseline_snapshots' / f'seed_{seed}'
    return DATA / 'reused_ce_snapshots' / f'seed_{seed}' if seed in REUSED_SEEDS else OUT / f'seed_{seed}'


def sources():
    result = original.sources()
    for name in ('configs/asvc_persistent_supervision_five_seeds_v1.yaml',
                 'scripts/run_persistent_supervision_five_seeds.py',
                 f'configs/{NAME}.yaml', 'scripts/fixed_ce_five_common.py',
                 'scripts/run_fixed_ce_five.py', 'scripts/analyze_fixed_ce_five.py',
                 'scripts/start_fixed_ce_five.ps1', 'tests/test_fixed_ce_five.py',
                 'docs/ASVC_FIXED_CE_FIVE_SEEDS.zh-CN.md'):
        result[name] = sha(ROOT / name)
    return result


def audit_baseline(config):
    original.verify_freeze()
    protected, pairs, reused = {}, [], []

    def protect(path):
        paths = path.rglob('*') if path.is_dir() else [path]
        for p in paths:
            if p.is_file():
                protected[str(p.relative_to(ROOT))] = sha(p)

    core = lambda cfg: {k: v for k, v in cfg.items() if k not in
        ('experiment', 'persistent_teaching', 'five_seed_review', 'ce_control', 'ce_five_seed_extension')}
    for seed in ALL_SEEDS:
        directory = baseline_directory(seed)
        manifest, result = read(directory / 'run_manifest.json'), read(directory / 'result.json')
        registration = ROOT / manifest['config']['persistent_teaching']['registration_root'] / 'freeze_manifest.json'
        frozen = read(registration)
        if not read(directory / 'completion.json')['complete'] or not result['complete'] or \
                result['environment_steps'] != 200000 or result['gradient_updates'] != 199501:
            raise ValueError(f'Margin seed{seed} is incomplete')
        if manifest['seed'] != seed or manifest['variant'] != 'supervision_floor' or \
                manifest['config'] != frozen['config'] or manifest['source_sha256'] != frozen['source_sha256'] or \
                manifest['freeze_manifest_sha256'] != sha(registration):
            raise ValueError(f'Margin seed{seed} identity differs')
        for name, value in frozen['source_sha256'].items():
            if sha(ROOT / name) != value:
                raise ValueError('Historical source changed: ' + name)
        if core(manifest['config']) != core(config) or \
                manifest['config']['persistent_teaching']['variants']['supervision_floor'] != dict(advice_floor=0., supervision_floor=.1):
            raise ValueError(f'Margin seed{seed} shared protocol differs')
        if manifest['effective_training'] != asdict(prior.legacy.effective_training(config, seed, 'formal')):
            raise ValueError(f'Margin seed{seed} exploration/replay/update budget differs')
        if sha(directory / 'model_final.pth') != result['final_model_sha256']:
            raise ValueError(f'Margin seed{seed} final model changed')
        protect(directory); protect(registration)
        pairs.append(dict(seed=seed, baseline_directory=str(directory.relative_to(ROOT)),
            initial_state_sha256=manifest['initial_state_sha256'], baseline_model_sha256=result['final_model_sha256'],
            baseline_curve_sha256=sha(directory / 'validation_curve.csv'),
            baseline_online_seconds=result['online_training_seconds']))
        if seed in REUSED_SEEDS:
            ce_dir = original.OUT / f'seed_{seed}'
            ce = read(ce_dir / 'result.json')
            if not ce['complete'] or ce['environment_steps'] != 200000 or ce['gradient_updates'] != 199501 or \
                    ce['config'] != read(original.DATA / 'freeze.json')['config'] or \
                    ce['initial_state_sha256'] != manifest['initial_state_sha256'] or \
                    ce['effective_training'] != manifest['effective_training'] or \
                    ce['baseline_model_sha256'] != result['final_model_sha256'] or \
                    ce['baseline_curve_sha256'] != sha(directory / 'validation_curve.csv') or \
                    ce['final_model_sha256'] != sha(ce_dir / 'model_final.pth'):
                raise ValueError(f'Reused CE seed{seed} identity differs')
            reused.append(dict(seed=seed, directory=str(ce_dir.relative_to(ROOT)),
                model_sha256=ce['final_model_sha256'], result_sha256=sha(ce_dir / 'result.json'),
                curve_sha256=sha(ce_dir / 'validation_curve.csv')))
    for directory in (original.DATA, original.OUT, original.RESULTS):
        protect(directory)
    for path in ('data/asvc_current_method.json', 'docs/ASVC_CURRENT.zh-CN.md',
                 config['dataset']['validation_reference'], config['dataset']['source_manifest'], config['dataset']['route_pool_manifest']):
        protect(ROOT / path)
    return pairs, reused, protected


def verify_freeze(check_baselines=True):
    frozen = read(DATA / 'freeze.json')
    config, _ = configure()
    if frozen['config'] != config or frozen['source_sha256'] != sources():
        raise ValueError('Five-seed extension source/configuration changed')
    for name, value in frozen['copied_input_sha256'].items():
        if sha(DATA / name) != value:
            raise ValueError('Frozen copied input changed: ' + name)
    if check_baselines:
        for name, value in frozen['protected_files_sha256'].items():
            if sha(ROOT / name) != value:
                raise ValueError('Historical artifact changed: ' + name)
    return frozen
