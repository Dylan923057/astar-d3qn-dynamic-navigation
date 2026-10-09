"""Registered independent tail experiment; existing training sources remain untouched."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import teaching_efficiency_common as previous
from astar_d3qn.agents.astar_supervision_tail import AStarSupervisionTailAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.utils.seed import seed_everything

ROOT, legacy = previous.ROOT, previous.legacy
CONFIG = ROOT / 'configs/whole_map_91701_supervision_tail_v1.yaml'
METHOD, BASELINE = 'advice_bound_margin_tail140k', 'advice_bound_margin'
METHODS, SEEDS = (BASELINE, METHOD), previous.SEEDS
sha256, read_json, canonical_hash = previous.sha256, previous.read_json, previous.canonical_hash
FrozenFactory = previous.FrozenFactory


def validate_config(config):
    if config != legacy.load_config(CONFIG):
        raise ValueError('Use the registered independent supervision-tail configuration.')
    original = legacy.load_config(legacy.CONFIG)
    if {k: v for k, v in config.items() if k not in ('experiment', 'supervision_tail')} != {
            k: v for k, v in original.items() if k != 'experiment'}:
        raise ValueError('Shared network, reward, replay, epsilon, advice or training settings changed.')
    if config['experiment'] != dict(original['experiment'], name='whole_map_91701_supervision_tail_v1',
            protocol='independent_action_supervision_tail_140k_v1',
            output_root='outputs/whole_map_91701_supervision_tail_v1',
            check_root='results/whole_map_91701_supervision_tail_v1'):
        raise ValueError('The tail experiment requires independent directories.')
    if config['supervision_tail']['schedule'] != dict(original_decay_steps=100000, tail_start_steps=80000,
                                                     tail_end_steps=140000):
        raise ValueError('Supervision tail must follow the requested 80k/100k/140k schedule.')
    if config['training_advice']['decay_environment_steps'] != 100000:
        raise ValueError('A* action override must still stop at 100000.')
    return legacy.validate_config(original)


def make_agent(config, problem, seed, device=None):
    seed_everything(seed)
    values = dict(config['agent'])
    if device:
        values['device'] = device
    advice, repair = config['training_advice'], config['value_repair']
    window = config['environment']['window_size']
    return AStarSupervisionTailAgent(D3QNConfig((5, window, window), 4, 5, seed=seed, **values), problem,
        supervision_schedule=config['supervision_tail']['schedule'], lower=repair['target_lower'],
        upper=repair['target_upper'], margin=repair['margin'], margin_weight=repair['margin_weight'],
        probability=advice['probability'], decay_steps=advice['decay_environment_steps'],
        risk_radius=advice['risk_radius'], rng_offset=advice['rng_seed_offset'])


def artifact_root(config):
    return ROOT / config['supervision_tail']['artifact_root']


def run_directory(config, method, seed):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('Use the two registered methods and paired seeds 0–4.')
    root = config['experiment']['output_root'] if method == METHOD else config['supervision_tail']['baseline_root']
    return ROOT / root / method / f'seed_{seed}'


def audit_run(config, method, seed):
    if method == BASELINE:
        original_config = legacy.load_config(previous.CONFIG)
        return previous.audit_run(original_config, BASELINE, seed)
    directory = run_directory(config, method, seed)
    result, completion = read_json(directory / 'result.json'), read_json(directory / 'completion.json')
    _, control = audit_run(config, BASELINE, seed)
    if (not completion['complete'] or completion['environment_steps'] != 200000
            or result['environment_steps'] != 200000 or result['gradient_updates'] != 199501
            or result['training_mode'] != 'formal' or result['method'] != method or result['seed'] != seed
            or result['config'] != config or result['test_data_used'] or result['evaluation_advice']
            or result['effective_training'] != asdict(legacy.effective_training(config, seed, 'formal'))
            or result['initial_state_sha256'] != control['initial_state_sha256']
            or result['demo_transition_count'] != 0 or result['online_replay_capacity'] != 10000
            or result['independent_500_used_for_training_or_scheduling']):
        raise ValueError(f'Incomplete or unpaired supervision-tail protocol: {directory}')
    for name, digest in result['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Recorded training source changed: {name}')
    if sha256(directory / 'model_final.pth') != result['final_model_sha256']:
        raise ValueError('New final model changed.')
    frozen = read_json(artifact_root(config) / 'freeze_manifest.json')
    if result['freeze_manifest_sha256'] != sha256(artifact_root(config) / 'freeze_manifest.json'):
        raise ValueError('Run did not use the registered pretraining freeze.')
    if result['source_sha256'] != frozen['registered_training_source_sha256']:
        raise ValueError('Training sources differ from the pretraining registration.')
    return directory, result


def source_hashes():
    files = [CONFIG, Path(__file__), ROOT / 'scripts/run_supervision_tail.py',
             ROOT / 'scripts/run_teaching_efficiency.py', ROOT / 'scripts/teaching_efficiency_common.py',
             ROOT / 'src/astar_d3qn/agents/astar_supervision_tail.py',
             *[ROOT / name for name in legacy.source_hashes()]]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def excluded_scenes(config):
    return read_json(ROOT / config['dataset']['validation_reference'])['scenarios'] + read_json(
        ROOT / config['supervision_tail']['excluded_independent_scenes'])['scenarios']


def check_scene_sets(config, scenes):
    old = excluded_scenes(config)
    combos = [previous.combination_key(s) for s in scenes]
    physics = [previous.physical_key(s) for s in scenes]
    ids = [s['scenario_id'] for s in scenes]
    if (len(scenes) != 500 or len(set(combos)) != 500 or len(set(physics)) != 500 or len(set(ids)) != 500
            or set(combos) & {previous.combination_key(s) for s in old}
            or set(physics) & {previous.physical_key(s) for s in old}
            or set(ids) & {s['scenario_id'] for s in old}
            or any(s['start'] != old[0]['start'] or s['goal'] != old[0]['goal']
                   or not 3 <= len(s['obstacles']) <= 5 for s in scenes)):
        raise ValueError('New 500 scenes must exclude both old 50 and previous 500 physical/route combinations.')


def load_frozen(config):
    directory = artifact_root(config)
    manifest = read_json(directory / 'freeze_manifest.json')
    if (manifest['config_sha256'] != canonical_hash(config) or not manifest['frozen_before_new_training']
            or manifest['training_started'] or manifest['model_performance_used_for_scene_selection']):
        raise ValueError('Independent scenes were not frozen under the pretraining protocol.')
    for name, digest in manifest['source_data_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen scene source changed: {name}')
    for name, digest in manifest['files_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Frozen artifact changed: {name}')
    if source_hashes() != manifest['registered_training_source_sha256']:
        raise ValueError('Training implementation changed after the freeze; preserve the registered experiment.')
    dataset = read_json(directory / 'independent_final_scenarios.json')
    if (dataset['map_id'] != config['dataset']['map_id'] or dataset['grid_sha256'] != config['dataset']['grid_sha256']
            or dataset['route_pool_design_sha256'] != config['dataset']['route_pool_design_sha256']):
        raise ValueError('Frozen map/route pool mismatch.')
    check_scene_sets(config, dataset['scenarios'])
    return dataset['scenarios'], read_json(directory / 'critical_states.json'), manifest
