"""Independent registration; reuse historical audits without editing their sources."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import dqfd_comparison_common as historical
from astar_d3qn.agents.ddqna_adapter import DDQNAD3QNAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.utils.seed import seed_everything

ROOT, legacy, tail = historical.ROOT, historical.legacy, historical.tail
CONFIG = ROOT/'configs/whole_map_91701_ddqna_adapter_v1.yaml'
METHOD, CONTROL, SEEDS = 'ddqna_d3qn', 'advice_bound_margin', historical.SEEDS
read_json, sha256, canonical_hash = historical.read_json, historical.sha256, historical.canonical_hash


def validate_config(config):
    original = legacy.load_config(legacy.CONFIG)
    if config != legacy.load_config(CONFIG) or {k:v for k,v in config.items() if k not in ('experiment','ddqna_adapter')} != {
            k:v for k,v in original.items() if k != 'experiment'}:
        raise ValueError('Shared environment, network, reward or optimization changed.')
    expected = dict(original['experiment'], name='whole_map_91701_ddqna_adapter_v1',
        protocol='ddqna_action_selection_d3qn_adapter_v1',
        output_root='outputs/whole_map_91701_ddqna_adapter_v1', check_root='results/whole_map_91701_ddqna_adapter_v1')
    spec = config['ddqna_adapter']
    if config['experiment'] != expected or spec['method'] != METHOD or spec['teacher_probability'] != .5 or (
            spec['teacher_schedule'] != 'constant_all_online_steps' or not spec['epsilon_branch_first'] or
            any(spec[k] for k in ('evaluation_advice','dynamic_risk_filter','auxiliary_supervision','td_target_clip'))):
        raise ValueError('DDQNA adapter policy changed.')
    return legacy.validate_config(original)


def make_agent(config, problem, seed, device=None):
    seed_everything(seed)
    values = dict(config['agent'])
    if device:
        values['device'] = device
    return DDQNAD3QNAgent(D3QNConfig((5,15,15), 4, 5, seed=seed, **values), problem,
        probability=config['ddqna_adapter']['teacher_probability'],
        rng_offset=config['training_advice']['rng_seed_offset'])


def run_directory(config, seed):
    if seed not in SEEDS:
        raise ValueError('Unregistered seed.')
    return ROOT/config['experiment']['output_root']/METHOD/f'seed_{seed}'


def source_hashes():
    paths = [CONFIG, Path(__file__), ROOT/'scripts/run_ddqna_adapter.py',
        ROOT/'scripts/report_ddqna_adapter.py', ROOT/'src/astar_d3qn/agents/ddqna_adapter.py',
        ROOT/'src/astar_d3qn/agents/astar_training_handover.py',
        *[ROOT/p for p in legacy.source_hashes()], *[ROOT/p for p in historical.source_hashes()]]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def register(config):
    validate_config(config)
    # Read identities, fingerprints and settings only; no final-test performance.
    _, _, scenes = tail.load_frozen(legacy.load_config(tail.CONFIG))
    historical.frozen(legacy.load_config(historical.CONFIG))
    manifest = dict(config_sha256=canonical_hash(config), source_sha256=source_hashes(),
        map_sha256=config['dataset']['grid_sha256'], pool_sha256=config['dataset']['route_pool_design_sha256'],
        validation_sha256=sha256(ROOT/config['dataset']['validation_reference']),
        final_scene_sha256=scenes['files_sha256']['independent_final_scenarios.json'],
        evaluation=dict(epsilon=0, teacher=False, dynamic_protection=False, static_mask=True,
                        checkpoint='final_200000_only', tuning=False),
        paired_initial_learner_sha256={str(s):historical.baseline_digest(config, validate_config(config)[0], s) for s in SEEDS})
    path = ROOT/config['ddqna_adapter']['artifact_root']/'freeze_manifest.json'
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError('Registered protocol/source changed after freeze; do not overwrite.')
    else:
        from astar_d3qn.utils.io import write_json
        write_json(manifest, path)
    return manifest


def audit_run(config, method, seed):
    if method == CONTROL:
        return historical.audit_run(legacy.load_config(historical.CONFIG), method, seed)
    if method != METHOD:
        raise ValueError(method)
    frozen = register(config)
    directory = run_directory(config, seed)
    run, completion = read_json(directory/'result.json'), read_json(directory/'completion.json')
    if (not completion['complete'] or completion['environment_steps'] != 200000 or
            run['config'] != config or run['method'] != METHOD or run['seed'] != seed or
            run['environment_steps'] != 200000 or run['gradient_updates'] != 199501 or
            run['demo_transition_count'] != 0 or run['online_replay_capacity'] != 10000 or
            run['test_data_used'] or run['evaluation_advice'] or run['foundation_loaded'] or
            run['independent_500_used_for_training_or_scheduling'] or run['training_mode'] != 'formal' or
            run['initial_learner_sha256'] != frozen['paired_initial_learner_sha256'][str(seed)] or
            run['source_sha256'] != frozen['source_sha256'] or
            run['effective_training'] != asdict(legacy.effective_training(config,seed,'formal')) or
            run['freeze_manifest_sha256'] != sha256(ROOT/config['ddqna_adapter']['artifact_root']/'freeze_manifest.json') or
            sha256(directory/'model_final.pth') != run['final_model_sha256']):
        raise ValueError(f'Incomplete, changed or unpaired run: {directory}')
    return directory, run
