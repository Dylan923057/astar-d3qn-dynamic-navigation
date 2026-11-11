"""Immutable protocol helpers; the registered six-method source is left untouched."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import run_value_repair as legacy
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.envs.types import Observation

ROOT = legacy.ROOT
CONFIG = ROOT / 'configs/whole_map_91701_teaching_efficiency_v1.yaml'
METHOD = 'supervision_only_bound'
ALL_METHODS = (*legacy.METHODS, METHOD)
SEEDS = (0, 1, 2, 3, 4)
BASELINE_FINGERPRINTS = ROOT / ('results/analysis_upload/'
    'whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/manifest.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config(config):
    if config != legacy.load_config(CONFIG):
        raise ValueError('Use the independent registered teaching-efficiency configuration.')
    baseline = legacy.load_config(legacy.CONFIG)
    if {k: v for k, v in config.items() if k not in ('experiment', 'teaching_efficiency')} != {
            k: v for k, v in baseline.items() if k != 'experiment'}:
        raise ValueError('The ablation changed shared training, labels, margin, rewards or dynamics.')
    expected = dict(baseline['experiment'], name='whole_map_91701_teaching_efficiency_v1',
                    protocol='supervision_without_action_override_v1',
                    output_root='outputs/whole_map_91701_teaching_efficiency_v1',
                    check_root='results/whole_map_91701_teaching_efficiency_v1')
    if config['experiment'] != expected:
        raise ValueError('An independent protocol and output directory are required.')
    if config['value_repair']['margin_decay_environment_steps'] != config['training_advice']['decay_environment_steps']:
        raise ValueError('The original shared 100000-step supervision clock must remain unchanged.')
    return legacy.validate_config(baseline)


def artifact_root(config):
    return ROOT / config['teaching_efficiency']['artifact_root']


def run_directory(config, method, seed):
    field = config['experiment']['output_root'] if method == METHOD else config['teaching_efficiency']['baseline_root']
    return ROOT / field / method / f'seed_{seed}'


def audit_run(config, method, seed, *, require_weight=True):
    directory = run_directory(config, method, seed)
    result = read_json(directory / 'result.json')
    completion = read_json(directory / 'completion.json')
    reference = legacy.load_config(legacy.CONFIG)
    expected_config = config if method == METHOD else reference
    if (not completion['complete'] or completion['environment_steps'] != 200000
            or result['environment_steps'] != 200000 or result['gradient_updates'] != 199501
            or result['training_mode'] != 'formal' or result['method'] != method or result['seed'] != seed
            or result['test_data_used'] or result['evaluation_advice'] or result['config'] != expected_config
            or result['effective_training'] != asdict(legacy.effective_training(reference, seed, 'formal'))
            or result['online_replay_capacity'] != 10000 or result['demo_transition_count'] != 0):
        raise ValueError(f'Run does not match the paired formal protocol: {directory}')
    for name, digest in result['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Recorded source changed: {name}')
    paired = read_json(run_directory(config, 'advice_bound_margin', seed) / 'result.json')
    if result['initial_state_sha256'] != paired['initial_state_sha256']:
        raise ValueError(f'Unpaired initialization: {directory}')
    if require_weight and not (directory / 'model_final.pth').is_file():
        raise FileNotFoundError(directory / 'model_final.pth')
    if require_weight:
        if method == METHOD:
            expected_weight = result['final_model_sha256']
        else:
            registered = read_json(BASELINE_FINGERPRINTS)['excluded_model_fingerprints']
            record = next(r for r in registered if r['method'] == method and r['seed'] == seed)
            if (ROOT / record['source']).resolve() != (directory / 'model_final.pth').resolve():
                raise ValueError('Registered baseline weight path changed.')
            expected_weight = record['sha256']
        if sha256(directory / 'model_final.pth') != expected_weight:
            raise ValueError(f'Final model differs from its registered fingerprint: {directory}')
    return directory, result


def observation_hash(state):
    return hashlib.sha256(np.asarray(state.spatial, dtype=np.float32).tobytes()
                          + np.asarray(state.scalars, dtype=np.float32).tobytes()).hexdigest()


def decode_observation(record):
    return Observation(np.asarray(record['spatial'], dtype=np.float32),
                       np.asarray(record['scalars'], dtype=np.float32))


def route_key(spec):
    route = tuple(tuple(cell) for cell in spec['route'])
    return min(route, route[::-1])


def combination_key(scene):
    # Geometry, not labels or obstacle ordering; phases do not create a new combination.
    return tuple(sorted(route_key(spec) for spec in scene['obstacles']))


def obstacle_cycle(spec):
    route = spec['route']
    index, direction = spec['start_index'], spec['direction']
    cells = []
    for time in range(2 * (len(route) - 1) * spec['move_every']):
        cells.append(tuple(route[index]))
        if (time + 1) % spec['move_every'] == 0:
            if not 0 <= index + direction < len(route):
                direction *= -1
            index += direction
    return tuple(cells)


def physical_key(scene):
    # Equivalent endpoint directions and reversed route encodings have the same physical key.
    return tuple(sorted(obstacle_cycle(spec) for spec in scene['obstacles']))


class FrozenFactory:
    """Exact serialized scenes, no sampling and no evaluation-dependent rejection."""
    def __init__(self, problem, scenes, *, trace=False):
        self.problem, self.scenes, self.trace = problem, scenes, trace
        self.index = 0
        self.environments = []

    def reset_schedule(self):
        self.index = 0
        self.environments = []

    def __call__(self, problem, **kwargs):
        if problem.grid_sha256 != self.problem.grid_sha256 or problem.start != self.problem.start or problem.goal != self.problem.goal:
            raise ValueError('Frozen evaluation map/start/goal mismatch.')
        if self.index >= len(self.scenes):
            raise IndexError('All frozen scenes were consumed; never resample or wrap around.')
        scene = self.scenes[self.index]
        self.index += 1
        specs = tuple(DynamicObstacleSpec(**dict(spec, route=tuple(tuple(c) for c in spec['route'])))
                      for spec in scene['obstacles'])
        env = DynamicGridNavigationEnv(problem, specs, scenario_id=scene['scenario_id'], **kwargs)
        env.dynamic_route_ids = tuple(scene['route_ids'])
        env.dynamic_route_categories = tuple(scene['categories'])
        wrapped = PathGuidanceEnvironment(env, enabled=False, record_trace=self.trace)
        self.environments.append(wrapped)
        return wrapped


def load_frozen(config):
    directory = artifact_root(config)
    manifest = read_json(directory / 'freeze_manifest.json')
    if manifest['config_sha256'] != canonical_hash(config):
        raise ValueError('Frozen configuration mismatch.')
    for name, digest in manifest['files_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Frozen artifact changed: {directory / name}')
    for name, digest in manifest['source_data_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen source input changed: {name}')
    dataset = read_json(directory / 'independent_final_scenarios.json')
    scenes = dataset['scenarios']
    old = read_json(ROOT / config['dataset']['validation_reference'])['scenarios']
    if (dataset['map_id'] != config['dataset']['map_id']
            or dataset['grid_sha256'] != config['dataset']['grid_sha256']
            or dataset['route_pool_design_sha256'] != config['dataset']['route_pool_design_sha256']
            or any(s['start'] != old[0]['start'] or s['goal'] != old[0]['goal']
                   or not 3 <= len(s['obstacles']) <= 5 for s in scenes)):
        raise ValueError('Frozen map/start/goal or obstacle-count protocol mismatch.')
    if (len(scenes) != 500 or len({s['scenario_id'] for s in scenes}) != 500
            or len({physical_key(s) for s in scenes}) != 500
            or len({combination_key(s) for s in scenes}) != 500
            or {physical_key(s) for s in scenes} & {physical_key(s) for s in old}
            or {combination_key(s) for s in scenes} & {combination_key(s) for s in old}
            or {s['scenario_id'] for s in scenes} & {s['scenario_id'] for s in old}):
        raise ValueError('The 500 frozen scenes must have distinct new physical route combinations.')
    return scenes, read_json(directory / 'critical_states.json'), manifest
