"""Frozen, full-state static-foundation forks. Prepare never trains."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_foundation_whole_map as base
import numpy as np
import torch
from astar_d3qn.training.trainer import TrainingWindowAdaptiveController
from astar_d3qn.utils.seed import seed_everything

METHODS = ('foundation_no_demo', 'foundation_fixed_25', 'foundation_time_decay', 'foundation_adaptive')
CONFIG = 'configs/foundation_dynamic_ratios_91701_v1.yaml'


def inputs():
    protocol = base.load_config(base._resolve(CONFIG))
    config, *rest = base.load_inputs(protocol['base_config'])
    if (tuple(protocol['methods']) != METHODS or protocol['seeds'] != list(range(5))
            or protocol['online_replay_capacity'] != 8680
            or protocol['demonstration_transition_count'] != 1320
            or not protocol['deterministic'] or protocol['test_scene_count'] != 200
            or protocol['static_acceptance_max_steps'] != 75):
        raise ValueError('Frozen protocol changed.')
    return {**config, **protocol}, *rest


def source_hashes(config):
    paths = list((ROOT / 'src/astar_d3qn').rglob('*.py'))
    paths += list((ROOT / 'scripts').glob('*.py'))
    paths += list((ROOT / 'configs').glob('*.yaml'))
    # Freeze map and route-pool assets too; historical output files are excluded.
    paths += list((ROOT / 'data').rglob('*.json')) + list((ROOT / 'data').rglob('*.csv'))
    return {p.relative_to(ROOT).as_posix(): base.file_sha(p) for p in sorted(set(paths))}


def scene_signature(scene):
    # IDs/labels are bookkeeping: compare the physical full scene, not names.
    obstacles = [{k: v for k, v in spec.items() if k != 'label'} for spec in scene['obstacles']]
    obstacles.sort(key=lambda x: json.dumps(x, sort_keys=True))
    return base.state_digest(obstacles)


def test_scenes(config, legacy, scene_config, problem, entry, pool, validation):
    factory = base._factory(scene_config, problem, entry, pool,
                            seed=config['test_sampling_seed'], prefix='test')
    seen = {scene_signature(s) for s in validation}
    if len(seen) != len(validation):
        raise ValueError('Validation contains duplicate complete physical scenes.')
    scenes = []
    while len(scenes) < config['test_scene_count']:
        env = factory(problem, max_steps=legacy['max_episode_steps'], window_size=legacy['window_size'])
        scene = {'scenario_id': env.scenario_id, 'condition': 'whole_map_3to5',
                 'start': list(problem.start), 'goal': list(problem.goal),
                 'obstacle_count': len(env.dynamic_obstacles),
                 'obstacles': [asdict(s) for s in env.dynamic_obstacles]}
        signature = scene_signature(scene)
        if signature not in seen:
            seen.add(signature)
            scenes.append(scene)
    return scenes


def preserved_evaluation(agent, replay, problem, legacy, scenes):
    before = base.state_digest(base.snapshot(agent, replay))
    modes = (agent.policy_network.training, agent.target_network.training)
    result = base.evaluate(agent, problem, legacy, scenes)
    if before != base.state_digest(base.snapshot(agent, replay)):
        raise RuntimeError('Evaluation changed network/optimizer/replay/full RNG state.')
    if modes != (agent.policy_network.training, agent.target_network.training):
        raise RuntimeError('Evaluation changed module mode.')
    return result


def prepare(root, device):
    config, legacy, scene_config, problem, entry, pool, frozen = inputs()
    if root.exists():
        raise FileExistsError(f'Prepare requires a new directory: {root}')
    root.mkdir(parents=True)
    hashes = source_hashes(config)
    for relative in hashes:
        dest = root / 'snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    tests = test_scenes(config, legacy, scene_config, problem, entry, pool, frozen['scenarios'])
    base.write_json(frozen, root / 'fixed_validation_scenarios.json')
    base.write_json({'sampling_seed': config['test_sampling_seed'], 'map_id': problem.map_id,
                     'grid_sha256': problem.grid_sha256, 'route_pool_design_sha256': pool['design_sha256'],
                     'scenarios': tests}, root / 'fixed_test_scenarios.json')
    manifest = {'config': config, 'source_hashes': hashes, 'device': device,
                'foundations': {}, 'validation_sha256': base.file_sha(root / 'fixed_validation_scenarios.json'),
                'test_sha256': base.file_sha(root / 'fixed_test_scenarios.json')}
    checks = []
    for seed in config['seeds']:
        row = {'seed': seed, 'passed': False}
        try:
            seed_everything(seed)
            checkpoint = base.load_foundation(config, legacy, problem, seed)
            path = base.foundation_path(config, seed)
            manifest['foundations'][str(seed)] = {'file_sha256': base.file_sha(path),
                                                  'snapshot_sha256': checkpoint['metadata']['snapshot_sha256']}
            for method in METHODS:
                agent, replay, audit = base.fork(config, legacy, problem, checkpoint, seed, method, device)
                base.write_json(audit, root / 'acceptance' / f'seed_{seed}' / f'{method}_restore.json')
            # Each method was verified against the exact same complete source digest.
            static, static_rows, static_traces = preserved_evaluation(agent, replay, problem, legacy, [None])
            dynamic, rows, traces = preserved_evaluation(agent, replay, problem, legacy, frozen['scenarios'])
            directory = root / 'acceptance' / f'seed_{seed}'
            base.write_records_csv(static_rows, directory / 'static_details.csv')
            base.write_records_csv(rows, directory / 'initial_validation_details.csv')
            base.write_json(static_traces, directory / 'static_trajectories.json')
            base.write_json([t for t in traces if not t['safe_success']], directory / 'initial_failures.json')
            row.update(qualified=True, smoke=False, full_restore_verified=True,
                       fingerprint=checkpoint['metadata']['snapshot_sha256'],
                       static_safe_success=static_rows[0]['safe_success'], static_steps=static_rows[0]['steps'],
                       initial_safe_success=dynamic['safe_success_rate'],
                       initial_dynamic_collision=dynamic['dynamic_collision_rate'],
                       initial_timeout=dynamic['timeout_rate'], evaluation_state_unchanged=True)
            row['passed'] = bool(static_rows[0]['safe_success'] and static_rows[0]['steps'] <= 75)
            if not row['passed']:
                row['error'] = f"Static acceptance failed: {static_rows[0]['failure_behavior']}, {static_rows[0]['steps']} steps"
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        checks.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    base.write_records_csv(checks, root / 'acceptance.csv')
    manifest['acceptance_passed'] = all(r['passed'] for r in checks)
    base.write_json(manifest, root / 'manifest.json')
    if not manifest['acceptance_passed']:
        raise RuntimeError('Foundation acceptance failed; formal training is blocked. See acceptance.csv.')


def verify(root):
    config, *rest = inputs()
    manifest = base.read_json(root / 'manifest.json')
    if not manifest['acceptance_passed']:
        raise RuntimeError('Static acceptance has not passed; training blocked.')
    if config != manifest['config'] or source_hashes(config) != manifest['source_hashes']:
        raise RuntimeError('Code/config/data differs from frozen version; training blocked.')
    for name, key in [('fixed_validation_scenarios.json', 'validation_sha256'), ('fixed_test_scenarios.json', 'test_sha256')]:
        if base.file_sha(root / name) != manifest[key]:
            raise RuntimeError(f'Frozen scenarios modified: {name}')
    for seed, expected in manifest['foundations'].items():
        if base.file_sha(base.foundation_path(config, int(seed))) != expected['file_sha256']:
            raise RuntimeError(f'Foundation source modified: seed {seed}')
    return (config, *rest), manifest


def fraction(method, step, controller=None):
    if method == 'foundation_no_demo':
        return 0.0
    if method == 'foundation_time_decay':
        return base.decay_demo_fraction(step)
    if method == 'foundation_adaptive':
        return controller(step)
    return .25


def train(root, seeds):
    loaded, manifest = verify(root)
    config, legacy, scene_config, problem, entry, pool, _ = loaded
    validation = base.read_json(root / 'fixed_validation_scenarios.json')['scenarios']
    tests = base.read_json(root / 'fixed_test_scenarios.json')['scenarios']
    version = base.state_digest(manifest)
    for seed in seeds:
        for method in METHODS:
            directory = root / method / f'seed_{seed}'
            provenance = {'manifest_digest': version, 'seed': seed, 'method': method,
                          'foundation': manifest['foundations'][str(seed)]}
            if directory.exists():
                if (directory / 'completion.json').exists():
                    completed = base.read_json(directory / 'completion.json')
                    if (completed['provenance'] == provenance and all(
                            (directory / f).is_file() and base.file_sha(directory / f) == sha
                            for f, sha in completed['files'].items())):
                        print(f'Reuse verified completed run: {method} seed {seed}', flush=True)
                        continue
                raise RuntimeError(f'Existing incomplete/unverified run cannot be overwritten: {directory}')
            directory.mkdir(parents=True)
            seed_everything(seed)
            checkpoint = base.load_foundation(config, legacy, problem, seed)
            agent, replay, audit = base.fork(config, legacy, problem, checkpoint, seed, method, manifest['device'])
            base.write_json({**audit, 'provenance': provenance}, directory / 'restore_audit.json')
            initial_updates = agent.update_steps
            controller = TrainingWindowAdaptiveController() if method == 'foundation_adaptive' else None
            curve, details, static_details, failures, exposures = [], [], [], [], []
            waits = 0
            demo_total = online_total = 0
            batch_file = (directory / 'demo_sampling.csv').open('w', newline='', encoding='utf-8')
            writer = csv.DictWriter(batch_file, fieldnames=['environment_steps', 'incremental_update',
                                   'requested_ratio', 'demo_count', 'online_count', 'actual_demo_ratio', 'online_replay_size'])
            writer.writeheader()

            def on_batch(step, demo, online):
                nonlocal demo_total, online_total
                demo_total += demo
                online_total += online
                writer.writerow(dict(environment_steps=step, incremental_update=agent.update_steps-initial_updates,
                    requested_ratio=replay.demo_fraction, demo_count=demo, online_count=online,
                    actual_demo_ratio=demo/(demo+online), online_replay_size=replay.online_size))

            def interaction(step, env, result, action, scene):
                nonlocal waits
                waits += int(action == 4)
                if controller:
                    controller.observe_interaction(step, completed=result.done,
                        safe_success=bool(result.info['reached'] and not result.info['collision']),
                        conflict=env.conflict_opportunity_steps > 0)

            def sample():
                scene = sampler()
                exposures.append({'episode_index': len(exposures), **scene})
                return scene

            def check(step, records, metrics):
                dynamic, rows, traces = preserved_evaluation(agent, replay, problem, legacy, validation)
                static, sr, st = preserved_evaluation(agent, replay, problem, legacy, [None])
                curve.append({'environment_steps': step, 'gradient_updates_since_fork': agent.update_steps-initial_updates,
                    **{k: v for k,v in dynamic.items() if k != 'behavior_counts'},
                    'static_safe_success': sr[0]['safe_success'], 'static_steps': sr[0]['steps'],
                    'demo_samples': demo_total, 'online_samples': online_total, 'training_wait_steps': waits})
                details.extend({'environment_steps': step, **r} for r in rows)
                static_details.extend({'environment_steps': step, **r} for r in sr)
                failures.extend({'environment_steps': step, 'split': 'validation', **t} for t in traces if not t['safe_success'])
                failures.extend({'environment_steps': step, 'split': 'static', **t} for t in st if not t['safe_success'])
                base.write_records_csv(curve, directory / 'validation_curve.csv')
                base.write_records_csv(records, directory / 'training.csv')
                if controller:
                    base.write_records_csv(controller.records, directory / 'adaptive_schedule.csv')
                batch_file.flush()
                print(f'{method} seed={seed} step={step} safe={dynamic["safe_success_rate"]:.3f}', flush=True)
                return False

            sampler = base.scene_sampler(scene_config, problem, entry, pool, legacy, seed)
            check(0, [], {})
            try:
                run = base.train_steps(agent, replay, problem, legacy, config['adaptation'], validation, seed, check,
                    scene_sampler=sample, demo_fraction_schedule=lambda step: fraction(method, step, controller),
                    on_interaction=interaction, on_batch=on_batch)
            finally:
                batch_file.close()
            # Test is called only at the fixed final budget, never by controller.
            if run['steps'] != 200000:
                raise RuntimeError('Final test requires exactly 200000 dynamic steps.')
            test, test_rows, test_traces = preserved_evaluation(agent, replay, problem, legacy, tests)
            base.write_records_csv(details, directory / 'validation_details.csv')
            base.write_records_csv(static_details, directory / 'static_retention.csv')
            base.write_records_csv(test_rows, directory / 'test_details.csv')
            failures.extend({'environment_steps': 200000, 'split': 'test', **t} for t in test_traces if not t['safe_success'])
            base.write_json(failures, directory / 'failure_trajectories.json')
            base.write_json(exposures, directory / 'training_scenarios.json')
            torch.save({'state': base.snapshot(agent, replay), 'metadata': provenance}, directory / 'checkpoint_final.pt')
            agent.save_weights(directory / 'model_final.pth')
            x = np.array([r['environment_steps'] for r in curve])
            y = np.array([r['safe_success_rate'] for r in curve])
            result = {'seed': seed, 'method': method, 'provenance': provenance, 'run': run,
                      'incremental_gradient_updates': agent.update_steps-initial_updates,
                      'final_safe_success': float(y[-1]), 'safe_success_aulc': float(np.trapz(y, x)/200000),
                      'final_dynamic_collision': curve[-1]['dynamic_collision_rate'],
                      'final_timeout': curve[-1]['timeout_rate'], 'final_static_safe_success': curve[-1]['static_safe_success'],
                      'training_wait_steps': waits, 'demo_samples': demo_total, 'online_samples': online_total,
                      'test_safe_success': test['safe_success_rate'], 'test_dynamic_collision': test['dynamic_collision_rate'],
                      'test_timeout': test['timeout_rate']}
            base.write_json(result, directory / 'result.json')
            files = {p.name: base.file_sha(p) for p in directory.iterdir() if p.is_file()}
            base.write_json({'provenance': provenance, 'files': files}, directory / 'completion.json')
    summarize(root)


def summarize(root):
    rows = []
    for method in METHODS:
        for seed in range(5):
            file = root / method / f'seed_{seed}/result.json'
            if file.exists():
                result = base.read_json(file)
                rows.append({k:v for k,v in result.items() if isinstance(v, (int,float,str))})
    base.write_records_csv(rows, root / 'summary_per_seed.csv')
    metrics = [k for k in rows[0] if k not in ('seed', 'method')] if rows else []
    aggregate = []
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        for metric in metrics:
            values = [r[metric] for r in group]
            aggregate.append({'method': method, 'metric': metric, 'n': len(values),
                'mean': float(np.mean(values)) if values else None,
                'std': float(np.std(values, ddof=1)) if len(values)>1 else None})
    base.write_records_csv(aggregate, root / 'summary_mean_std.csv')
    paired = []
    for left, right in combinations(METHODS, 2):
        for metric in metrics:
            differences = []
            for seed in range(5):
                a = next((r for r in rows if r['method']==left and r['seed']==seed), None)
                b = next((r for r in rows if r['method']==right and r['seed']==seed), None)
                if a is not None and b is not None:
                    delta = b[metric]-a[metric]
                    differences.append(delta)
                    paired.append({'left':left, 'right':right, 'metric':metric, 'seed':seed, 'difference_right_minus_left':delta})
            if differences:
                paired.append({'left':left, 'right':right, 'metric':metric, 'seed':'mean',
                    'n':len(differences), 'difference_right_minus_left':float(np.mean(differences)),
                    'std':float(np.std(differences,ddof=1)) if len(differences)>1 else None})
    base.write_records_csv(paired, root / 'summary_paired.csv')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=['prepare','train','summarize'], default='prepare')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=[0,1])
    args = parser.parse_args()
    root = base._resolve(args.output_root)
    if args.mode == 'prepare':
        prepare(root, args.device)
    elif args.mode == 'train':
        train(root, args.seeds)
    else:
        verify(root)
        summarize(root)


if __name__ == '__main__':
    main()
