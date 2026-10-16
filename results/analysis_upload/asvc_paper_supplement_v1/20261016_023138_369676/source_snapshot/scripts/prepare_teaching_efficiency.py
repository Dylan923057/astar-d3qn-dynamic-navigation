"""Freeze independent final-only scenes and retrospective critical states; never train."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict

import teaching_efficiency_common as common
from astar_d3qn.utils.io import write_json


def critical_states(config, problem):
    reference = common.read_json(common.ROOT / config['dataset']['validation_reference'])
    scenes = {s['scenario_id']: s for s in reference['scenarios']}
    source = common.run_directory(config, 'advice_bound_margin', 3) / 'validation_failures_200000.json'
    traces = common.read_json(source)
    expected = {f'validation_seed91701999_episode{i:06d}' for i in (9, 19, 21, 49)}
    if {t['scenario_id'] for t in traces} != expected:
        raise ValueError('Expected the four original combination seed3 final collisions.')
    states = []
    for trace in traces:
        factory = common.FrozenFactory(problem, [scenes[trace['scenario_id']]])
        env = factory(problem, max_steps=300, window_size=15,
                      reward_config=common.legacy._reward_config(config), terminate_on_collision=True)
        state = env.reset()
        for index, row in enumerate(trace['steps']):
            if list(env.position) != row['before']['position']:
                raise ValueError('The saved failure prefix cannot be replayed exactly.')
            if index == len(trace['steps']) - 1:
                states.append(dict(scenario_id=trace['scenario_id'], step=row['step'],
                                   position=list(env.position), action=row['action'],
                                   collision_route_ids=[scene_spec['label'] for i, scene_spec in
                                                        enumerate(scenes[trace['scenario_id']]['obstacles'])
                                                        if i in row['dynamic_collision_indices']],
                                   observation_sha256=common.observation_hash(state),
                                   spatial=state.spatial.tolist(), scalars=state.scalars.tolist(),
                                   static_action_mask=env.action_mask(True).tolist(),
                                   source_failure_file=str(source.relative_to(common.ROOT))))
            result = env.step(row['action'])
            if (list(env.position) != row['position'] or result.reward != row['reward']
                    or list(map(list, env.dynamic_positions)) != row['dynamic_positions']):
                raise ValueError('The saved failure dynamics/reward cannot be replayed exactly.')
            state = result.observation
        if result.info['termination_reason'] != 'collision':
            raise ValueError('Critical-state replay did not reproduce the recorded collision.')
    return states, source


def prepare(config):
    inputs = common.validate_config(config)
    for seed in common.SEEDS:
        for method in common.legacy.METHODS:
            common.audit_run(config, method, seed)
    directory = common.artifact_root(config)
    if directory.exists():
        common.load_frozen(config)
        print(f'Existing immutable artifacts verified: {directory}', flush=True)
        return directory
    problem = inputs[0]
    states, failure_source = critical_states(config, problem)
    old_reference = common.read_json(common.ROOT / config['dataset']['validation_reference'])
    old_combinations = {common.combination_key(s) for s in old_reference['scenarios']}
    old_physical = {common.physical_key(s) for s in old_reference['scenarios']}
    settings = config['teaching_efficiency']
    factory = common.legacy.runtime._factory(config, *inputs, seed=settings['evaluation_sampling_seed'],
                                             prefix='independent_final_500')
    scenes, combinations, physical, rejected = [], set(), set(), Counter()
    for draw in range(1, settings['evaluation_max_candidate_draws'] + 1):
        env = factory(problem, max_steps=300, window_size=15,
                      reward_config=common.legacy._reward_config(config), terminate_on_collision=True)
        scene = dict(scenario_id=f'independent_final_seed{settings["evaluation_sampling_seed"]}_{len(scenes):06d}',
                     start=list(problem.start), goal=list(problem.goal),
                     route_ids=list(env.dynamic_route_ids), categories=list(env.dynamic_route_categories),
                     obstacles=[asdict(s) for s in env.dynamic_obstacles])
        combo, phase = common.combination_key(scene), common.physical_key(scene)
        if combo in old_combinations or phase in old_physical:
            rejected['old_validation_combination_or_phase'] += 1
            continue
        if combo in combinations or phase in physical:
            rejected['duplicate_new_combination_or_phase'] += 1
            continue
        combinations.add(combo)
        physical.add(phase)
        scenes.append(scene)
        if len(scenes) == settings['evaluation_scene_count']:
            break
    if len(scenes) != 500:
        raise RuntimeError(f'Only {len(scenes)} distinct accepted combinations; no artifact was written.')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(dict(map_id=problem.map_id, grid_sha256=problem.grid_sha256,
                    route_pool_design_sha256=inputs[2]['design_sha256'],
                    sampling_seed=settings['evaluation_sampling_seed'],
                    purpose=settings['evaluation_use'], scenarios=scenes), directory / 'independent_final_scenarios.json')
    write_json(states, directory / 'critical_states.json')
    paths = [config['dataset'][key] for key in ('source_manifest', 'route_pool_manifest', 'validation_reference')]
    paths.append(str(failure_source.relative_to(common.ROOT)))
    write_json(dict(config_sha256=common.canonical_hash(config), scene_count=500,
                    candidate_draws=draw, rejected=dict(rejected),
                    obstacle_count_distribution=dict(Counter(len(s['obstacles']) for s in scenes)),
                    unique_route_combinations=500, unique_physical_scenes=500,
                    old_50_combination_overlap=0, old_50_physical_overlap=0,
                    files_sha256={name: common.sha256(directory / name) for name in
                                  ('independent_final_scenarios.json', 'critical_states.json')},
                    source_data_sha256={name: common.sha256(common.ROOT / name) for name in paths},
                    training_started=False, model_performance_used_for_scene_selection=False,
                    training_access_allowed=False, scheduling_access_allowed=False,
                    scope='Same fixed map/start/goal and route pool; distinct combinations from old validation, not unseen maps or a guaranteed disjoint training distribution.',
                    sampling_note='Original acceptance and 3-5 count sampler; reject repeated or old-50 route combinations without performance filtering. Uniqueness rejection can change count proportions.'),
               directory / 'freeze_manifest.json')
    common.load_frozen(config)
    print(f'Frozen 500 scenes after {draw} draws; rejected={dict(rejected)}; {directory}', flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare(common.legacy.load_config(common.CONFIG))


if __name__ == '__main__':
    main()
