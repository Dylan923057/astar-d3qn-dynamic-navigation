"""Evaluate only ten new ablations and twenty new density suites; audit reused evidence."""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import torch

import persistent_paper_common as core
from astar_d3qn.evaluation.rollout import evaluate_agent

ROOT, REG, OUT, RESULTS = core.ROOT, core.REG, core.OUT, core.RESULTS
read, sha, require = core.common.read_json, core.common.sha256, core.require
DENSITY_REG = ROOT / 'data/asvc_paper_density_v2'
METRICS = ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate')
OLD_DENSITY_NAMES = dict(original_asvc='advice_bound_margin_all_eligible', d3qn='unguided_raw',
                         dqfd='dqfd', astar_prefill='astar_prefill_bound')


@lru_cache(maxsize=5)
def scenes_for(suite):
    if suite == 'final500':
        scenes, _, frozen = core.tail.load_frozen(core.legacy.load_config(core.tail.CONFIG))
        digest = frozen['files_sha256']['independent_final_scenarios.json']
    else:
        n = int(suite.split('_')[-1])
        require(n in (1, 3, 5, 7), 'Unknown density.')
        frozen = read(DENSITY_REG / 'density_registration.json')
        path = DENSITY_REG / f'density_{n}.json'
        digest = sha(path)
        require(digest == frozen['files_sha256'][path.name], 'Original density scenes changed.')
        require(frozen['all_2000_environments_validated'] and not frozen['used_for_tuning'], 'Use the frozen valid density-v2 scenes.')
        scenes = read(path)['scenes']
        require(all(len(s['obstacles']) == n for s in scenes), 'Wrong obstacle count.')
    require(len(scenes) == len({s['scenario_id'] for s in scenes}) == 500, 'Expected 500 unique scenes.')
    return scenes, digest


def summary(method, seed, suite, rows):
    return dict(method=method, seed=seed, suite=suite, scene_count=500,
        safe_success_rate=sum(float(r['safe_success']) for r in rows)/500,
        dynamic_collision_rate=sum(float(r['dynamic_collision']) for r in rows)/500,
        timeout_rate=sum(r['termination_reason'] == 'timeout' for r in rows)/500)


def check_rows(rows, stored, scenes):
    require([r['scenario_id'] for r in rows] == [s['scenario_id'] for s in scenes], 'Scene IDs/order differ.')
    computed = summary('', 0, '', rows)
    require(all(abs(float(stored[k])-computed[k]) < 1e-12 for k in METRICS), 'Summary differs from individual scenes.')


def check_autonomous(meta):
    require(meta.get('epsilon') == 0 and meta.get('static_mask') is True, 'Evaluation epsilon/mask differs.')
    for key in ('teacher', 'route_guidance', 'dynamic_shield', 'dynamic_protection', 'q_output_clip',
                'action_override', 'used_for_tuning', 'used_for_schedule_tuning'):
        require(not meta.get(key, False), f'Evaluation intervention/tuning: {key}')
    require(meta.get('checkpoint_steps') == 200000 or meta.get('checkpoint') == 'final_200000_step_only', 'Not the final checkpoint.')


@lru_cache(maxsize=1)
def no_supervision_cache():
    path = ROOT / read(ROOT / 'results/latest_all_eligible_modules_final500_v1/latest.json')['evaluation']
    return path, read(path / 'verification.json'), core.read_csv(path / 'scene_details.csv'), core.read_csv(path / 'summary.csv')


def cached(info, suite):
    method, seed = info['method'], info['seed']
    scenes, digest = scenes_for(suite)
    if suite == 'final500' and method == 'no_supervision':
        path, meta, all_rows, summaries = no_supervision_cache()
        check_autonomous(meta)
        require(meta['complete'] and meta['frozen_scene_sha256'] == digest, 'Cached final500 scenes differ.')
        hashes = {k.replace('\\', '/'): v for k, v in meta['source_model_sha256'].items()}
        require(hashes.get(core.relative(info['model'])) == info['digest'], 'Cached no-supervision model differs.')
        rows = [r for r in all_rows if r['method'] == 'advice_bound' and int(r['seed']) == seed]
        stored = next(r for r in summaries if r['method'] == 'advice_bound' and int(r['seed']) == seed)
        files = ('verification.json', 'scene_details.csv', 'summary.csv')
    else:
        if suite == 'final500':
            require(method == 'persistent_asvc', 'No cache for a new ablation.')
            path = ROOT / 'results/asvc_persistent_supervision_five_seeds_v1/final500' / f'seed_{seed}'
        else:
            path = ROOT / 'results/asvc_paper_supplement_v1/evaluation_v2' / suite.split('_')[-1] / OLD_DENSITY_NAMES[method] / f'seed_{seed}'
        meta = read(path / 'registration.json')
        check_autonomous(meta)
        require(meta['model_sha256'] == info['digest'] and meta['scene_sha256'] == digest, 'Cached model/scenes differ.')
        require(read(path / 'completion.json')['complete'], 'Cached evaluation incomplete.')
        require(meta['rollout_sha256'] == sha(ROOT / 'src/astar_d3qn/evaluation/rollout.py'), 'Cached rollout implementation differs.')
        rows, stored = core.read_csv(path / 'scene_details.csv'), read(path / 'summary.json')
        files = ('registration.json', 'completion.json', 'summary.json', 'scene_details.csv')
    check_rows(rows, stored, scenes)
    provenance = dict(source=core.relative(path), model_sha256=info['digest'], scene_sha256=digest,
        files_sha256={core.relative(path / name): sha(path / name) for name in files})
    return rows, summary(method, seed, suite, rows), provenance


def audit_reuse():
    """Read-only audit, performed before the ten trainings are launched."""
    sources, cache_records, models = {}, [], {}
    for method in dict.fromkeys((*core.DENSITY_METHODS, 'no_supervision')):
        for seed in core.SEEDS:
            info = core.run_info(method, seed)
            core.assert_final_checkpoint(info)
            models[core.relative(info['model'])] = info['digest']
            suites = ('final500',) if method in ('persistent_asvc', 'no_supervision') else tuple(f'density_{n}' for n in (1, 3, 5, 7))
            for suite in suites:
                _, _, origin = cached(info, suite)
                sources.update(origin['files_sha256'])
                cache_records.append(dict(method=method, seed=seed, suite=suite, **origin))
    seven = scenes_for('density_7')[0]
    for n in (1, 3, 5):
        small = scenes_for(f'density_{n}')[0]
        require(all(a['obstacles'] == b['obstacles'][:n] for a, b in zip(small, seven)), 'Density prefix pairing changed.')
    sources[core.relative(DENSITY_REG / 'density_registration.json')] = sha(DENSITY_REG / 'density_registration.json')
    for n in (1, 3, 5, 7):
        path = DENSITY_REG / f'density_{n}.json'
        sources[core.relative(path)] = sha(path)
    sources[core.relative(Path(__file__))] = sha(Path(__file__))
    value = dict(complete=True, reused_suites=90, reused_model_sha256=models, source_sha256=sources,
        caches=cache_records, new_ablation_final500_suites=10, new_density_suites=20,
        evaluation_device='cpu', new_evaluation_episodes=15000, unchanged_full_final500=True,
        unchanged_existing_comparison_training=True)
    target = REG / 'reuse_registration.json'
    if target.exists():
        require(read(target) == value, 'Frozen evaluation reuse inputs changed.')
    else:
        core.write_json(value, target)
    print('REUSE PASSED: 90 cached suites; only 10 new ablation final500 and 20 new density suites to evaluate.', flush=True)
    return value


def check_reuse_freeze():
    value = read(REG / 'reuse_registration.json')
    for name, digest in (value['source_sha256'] | value['reused_model_sha256']).items():
        require(sha(ROOT / name) == digest, f'Frozen evaluation source changed: {name}')
    return value


def evaluate_one(info, suite):
    scenes, digest = scenes_for(suite)
    method, seed = info['method'], info['seed']
    target = RESULTS / 'evaluation' / suite / method / f'seed_{seed}'
    registration = dict(method=method, seed=seed, suite=suite, scene_sha256=digest,
        model_sha256=info['digest'], checkpoint_steps=200000, epsilon=0, teacher=False,
        route_guidance=False, dynamic_shield=False, q_output_clip=False, static_mask=True,
        used_for_tuning=False, device='cpu', max_steps=300, window_size=15,
        source_sha256=sha(Path(__file__)), rollout_sha256=sha(ROOT / 'src/astar_d3qn/evaluation/rollout.py'))
    if (target / 'completion.json').exists():
        require(read(target / 'completion.json')['complete'], 'Invalid completion marker.')
        require(read(target / 'registration.json') == registration, 'Cached new evaluation registration differs.')
        check_rows(core.read_csv(target / 'scene_details.csv'), read(target / 'summary.json'), scenes)
        return
    reused = (suite == 'final500' and method not in core.METHODS) or (suite != 'final500' and method != 'persistent_asvc')
    clock = perf_counter()
    if reused:
        rows, result, provenance = cached(info, suite)
        core.write_json(provenance, target / 'reuse_provenance.json')
    else:
        agent = core.legacy.runtime.make_agent(info['config'], seed, 'cpu')
        agent.load_weights(info['model'])
        factory = core.tail.FrozenFactory(info['inputs'][0], scenes, trace=True)
        with core.legacy.runtime.preserved_evaluation(agent):
            _, rows, _ = evaluate_agent(agent, [info['inputs'][0]] * 500, max_steps=300,
                window_size=15, reward_config=core.legacy._reward_config(info['config']),
                terminate_on_collision=True, environment_factory=factory, mask_static_invalid_actions=True)
        result = summary(method, seed, suite, rows)
        failures = [dict(scenario_id=r['scenario_id'], safe_success=False, termination_reason=r['termination_reason'], steps=e.trace)
                    for r, e in zip(rows, factory.environments) if not r['safe_success']]
        core.write_json(failures, OUT / 'evaluation' / suite / method / f'seed_{seed}' / 'failure_trajectories.json')
    check_rows(rows, result, scenes)
    require(sha(info['model']) == info['digest'], 'Evaluation modified a model.')
    core.write_json(registration, target / 'registration.json')
    core.write_csv(rows, target / 'scene_details.csv')
    core.write_json(result, target / 'summary.json')
    core.write_json(dict(complete=True, reused=reused, seconds=perf_counter()-clock), target / 'completion.json')
    print(f'[{suite} {method} seed{seed}] {"REUSED" if reused else "EVALUATED"} success={result["safe_success_rate"]:.2%}', flush=True)


def evaluate(suite='all'):
    torch.set_num_threads(1)
    check_reuse_freeze()
    tasks = []
    failed = []
    if suite in ('all', 'ablation'):
        for method in core.ABLATIONS:
            for seed in core.SEEDS:
                info = core.run_info(method, seed)
                if info['failed']:
                    failed.append(info['failure'])
                else:
                    tasks.append((info, 'final500'))
    if suite in ('all', 'density'):
        for method in core.DENSITY_METHODS:
            for seed in core.SEEDS:
                info = core.run_info(method, seed)
                tasks.extend((info, f'density_{n}') for n in (1, 3, 5, 7))
    with core.run_lock(OUT / 'locks/evaluation.lock'):
        for info, name in tasks:
            core.assert_final_checkpoint(info)
            evaluate_one(info, name)
        core.write_json(dict(complete=True, suite=suite, completed_suites=len(tasks),
            numerical_failures=failed, failed_seeds_excluded_from_means=False), RESULTS / f'evaluation_{suite}_status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--suite', choices=('all', 'ablation', 'density'), default='all')
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.suite)
    else:
        audit_reuse()
