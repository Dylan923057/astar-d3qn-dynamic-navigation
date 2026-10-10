"""Independent fixed protocol; dispatch historical audits without modifying registered sources."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

import supervision_tail_common as tail
from astar_d3qn.agents.astar_no_risk import AStarNoRiskAgent
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.agents.dqfd import DQfDAgent
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.dqfd import NStepCollector
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.seed import seed_everything

ROOT, legacy, previous = tail.ROOT, tail.legacy, tail.previous
CONFIG = ROOT / 'configs/whole_map_91701_dqfd_comparison_v1.yaml'
METHODS = ('dqfd', 'dqfd_bound', 'advice_bound_margin_no_risk')
ALL_METHODS = (*previous.ALL_METHODS, tail.METHOD, *METHODS)
SEEDS = tail.SEEDS
read_json, sha256, canonical_hash = tail.read_json, tail.sha256, tail.canonical_hash


def validate_config(config):
    old = legacy.load_config(legacy.CONFIG)
    if config != legacy.load_config(CONFIG) or {k: v for k, v in config.items() if k not in ('experiment', 'comparison')} != {
            k: v for k, v in old.items() if k != 'experiment'}:
        raise ValueError('Shared navigation/training settings changed.')
    expected = dict(old['experiment'], name='whole_map_91701_dqfd_comparison_v1',
                    protocol='frozen_dqfd_and_observed_risk_ablation_v1',
                    output_root='outputs/whole_map_91701_dqfd_comparison_v1',
                    check_root='results/whole_map_91701_dqfd_comparison_v1')
    if config['experiment'] != expected or tuple(config['comparison']['methods']) != METHODS:
        raise ValueError('Use independent registered methods/directories.')
    return legacy.validate_config(old)


def source_hashes():
    paths = [CONFIG, Path(__file__), ROOT/'scripts/prepare_dqfd_comparison.py',
             ROOT/'scripts/run_dqfd_comparison.py', ROOT/'src/astar_d3qn/agents/dqfd.py',
             ROOT/'src/astar_d3qn/agents/astar_no_risk.py', ROOT/'src/astar_d3qn/replay/dqfd.py',
             ROOT/'src/astar_d3qn/training/dqfd.py', ROOT/'scripts/run_teaching_efficiency.py',
             ROOT/'src/astar_d3qn/envs/path_guidance.py', ROOT/'src/astar_d3qn/envs/dynamic_grid.py',
             ROOT/'src/astar_d3qn/replay/transition.py', ROOT/'src/astar_d3qn/agents/networks.py',
             *[ROOT/name for name in legacy.source_hashes()]]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def artifact_root(config):
    return ROOT/config['comparison']['artifact_root']


def run_directory(config, method, seed):
    if seed not in SEEDS or method not in ALL_METHODS:
        raise ValueError('Unknown paired method/seed.')
    if method in METHODS:
        return ROOT/config['experiment']['output_root']/method/f'seed_{seed}'
    if method == tail.METHOD:
        return tail.run_directory(legacy.load_config(tail.CONFIG), method, seed)
    return previous.run_directory(legacy.load_config(previous.CONFIG), method, seed)


def make_agent(config, problem, seed, method, device=None):
    seed_everything(seed)
    values = dict(config['agent'])
    if device:
        values['device'] = device
    ac = D3QNConfig((5, 15, 15), 4, 5, seed=seed, **values)
    if method in ('dqfd', 'dqfd_bound'):
        return DQfDAgent(ac, config['comparison'], bound=method == 'dqfd_bound')
    if method != METHODS[2]:
        raise ValueError(method)
    ad, repair = config['training_advice'], config['value_repair']
    return AStarNoRiskAgent(ac, problem, repair_method='advice_bound_margin',
        lower=repair['target_lower'], upper=repair['target_upper'], margin=repair['margin'],
        margin_weight=repair['margin_weight'], probability=ad['probability'],
        decay_steps=ad['decay_environment_steps'], risk_radius=ad['risk_radius'], rng_offset=ad['rng_seed_offset'])


def paired_digest(agent):
    # New algorithms may have additional replay/cost state; compare the complete base learner state.
    return legacy.runtime.state_digest(D3QNAgent.training_state_dict(agent))


def baseline_digest(config, problem, seed):
    agent = legacy.make_agent(legacy.load_config(legacy.CONFIG), problem, seed, 'advice_bound_margin', 'cpu')
    _, result = previous.audit_run(legacy.load_config(previous.CONFIG), 'advice_bound_margin', seed)
    if legacy.runtime.state_digest(agent.training_state_dict()) != result['initial_state_sha256']:
        raise ValueError('Historical paired initialization no longer reproducible.')
    return paired_digest(agent)


def frozen(config):
    directory = artifact_root(config)
    manifest = read_json(directory/'freeze_manifest.json')
    if manifest['config_sha256'] != canonical_hash(config) or manifest['source_sha256'] != source_hashes():
        raise ValueError('New registered config/training source changed after freeze.')
    for name, digest in manifest['files_sha256'].items():
        if sha256(directory/name) != digest:
            raise ValueError(f'Frozen demo artifact changed: {name}')
    tail_config = legacy.load_config(tail.CONFIG)
    _, _, old_manifest = tail.load_frozen(tail_config)
    if manifest['final_scene_sha256'] != old_manifest['files_sha256']['independent_final_scenarios.json']:
        raise ValueError('Final evaluation scenes changed.')
    return manifest


def load_demonstrations(config):
    manifest = frozen(config)
    dataset = np.load(artifact_root(config)/'demonstrations.npz', allow_pickle=False)
    settings, records = config['comparison'], []
    collector = NStepCollector(settings['n_steps'], config['agent']['gamma'])
    states = [Observation(s.astype(np.float32), a) for s, a in zip(dataset['spatial'], dataset['scalars'])]
    next_states = [Observation(s.astype(np.float32), a) for s, a in zip(dataset['next_spatial'], dataset['next_scalars'])]
    for i, (state, next_state) in enumerate(zip(states, next_states)):
        transition = Transition(state, int(dataset['action'][i]), float(dataset['reward'][i]), next_state,
                                bool(dataset['terminated'][i]), dataset['next_mask'][i])
        records.extend(collector.add(transition, episode_end=bool(dataset['episode_end'][i])))
    if collector.pending or len(records) != manifest['demo_transition_count'] or len(records) != len(states):
        raise ValueError('Demo episode/n-step boundaries mismatch.')
    return records, manifest


def audit_run(config, method, seed):
    if method == tail.METHOD:
        return tail.audit_run(legacy.load_config(tail.CONFIG), method, seed)
    if method not in METHODS:
        return previous.audit_run(legacy.load_config(previous.CONFIG), method, seed)
    manifest = frozen(config)
    directory = run_directory(config, method, seed)
    result, completion = read_json(directory/'result.json'), read_json(directory/'completion.json')
    expected_pretrain = config['comparison']['pretraining_updates'] if method.startswith('dqfd') else 0
    if (not completion['complete'] or completion['environment_steps'] != 200000
            or result['config'] != config or result['method'] != method or result['seed'] != seed
            or result['environment_steps'] != 200000 or result['online_updates'] != 199501
            or result['pretraining_updates'] != expected_pretrain
            or result['gradient_updates'] != 199501+expected_pretrain or result['evaluation_advice']
            or result['test_data_used'] or result['independent_500_used_for_training_or_scheduling']
            or result['online_replay_capacity'] != 10000 or result['training_mode'] != 'formal'
            or result['initial_learner_sha256'] != baseline_digest(config, validate_config(config)[0], seed)
            or result['source_sha256'] != manifest['source_sha256']
            or result['freeze_manifest_sha256'] != sha256(artifact_root(config)/'freeze_manifest.json')
            or result['effective_training'] != asdict(legacy.effective_training(config, seed, 'formal'))):
        raise ValueError(f'Incomplete or unpaired run: {directory}')
    if method.startswith('dqfd') and (result['demonstration_sha256'] != manifest['files_sha256']['demonstrations.npz']
            or result['demo_transition_count'] != manifest['demo_transition_count']
            or result['permanent_demo_retained_count'] != manifest['demo_transition_count']
            or result['online_astar_lookups'] != 0):
        raise ValueError('DQfD permanent demonstration protocol changed.')
    if sha256(directory/'model_final.pth') != result['final_model_sha256']:
        raise ValueError('Final weight fingerprint changed.')
    return directory, result
