"""Freeze 500 new held-out combinations before training; no policy performance is queried."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

import supervision_tail_common as common
from astar_d3qn.utils.io import write_json


def prepare(config):
    inputs = common.validate_config(config)
    directory = common.artifact_root(config)
    if directory.exists():
        common.load_frozen(config)
        print(f'Existing immutable freeze verified: {directory}')
        return directory
    if any(common.run_directory(config, common.METHOD, s).exists() for s in common.SEEDS):
        raise ValueError('Freeze must precede every new training run.')
    for seed in common.SEEDS:
        common.audit_run(config, common.BASELINE, seed)
    excluded = common.excluded_scenes(config)
    old_combos = {common.previous.combination_key(s) for s in excluded}
    old_physics = {common.previous.physical_key(s) for s in excluded}
    settings, problem = config['supervision_tail'], inputs[0]
    factory = common.legacy.runtime._factory(config, *inputs, seed=settings['evaluation_sampling_seed'],
                                            prefix='supervision_tail_final_500')
    scenes, combinations, physics, rejected = [], set(), set(), Counter()
    for draw in range(1, settings['evaluation_max_candidate_draws'] + 1):
        env = factory(problem, max_steps=300, window_size=15, reward_config=common.legacy._reward_config(config),
                      terminate_on_collision=True)
        scene = dict(scenario_id=f'tail_final_seed{settings["evaluation_sampling_seed"]}_{len(scenes):06d}',
                     start=list(problem.start), goal=list(problem.goal), route_ids=list(env.dynamic_route_ids),
                     categories=list(env.dynamic_route_categories), obstacles=[asdict(s) for s in env.dynamic_obstacles])
        combo, physical = common.previous.combination_key(scene), common.previous.physical_key(scene)
        if combo in old_combos or physical in old_physics:
            rejected['old_50_or_old_500_combination_or_phase'] += 1
            continue
        if combo in combinations or physical in physics:
            rejected['duplicate_new_combination_or_phase'] += 1
            continue
        combinations.add(combo)
        physics.add(physical)
        scenes.append(scene)
        if len(scenes) == 500:
            break
    common.check_scene_sets(config, scenes)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(dict(map_id=problem.map_id, grid_sha256=problem.grid_sha256,
                    route_pool_design_sha256=inputs[2]['design_sha256'], sampling_seed=settings['evaluation_sampling_seed'],
                    purpose=settings['evaluation_use'], scenarios=scenes), directory / 'independent_final_scenarios.json')
    source = common.ROOT / settings['critical_states_reference']
    write_json(common.read_json(source), directory / 'critical_states.json')
    paths = [config['dataset'][k] for k in ('source_manifest', 'route_pool_manifest', 'validation_reference')]
    paths += [settings['excluded_independent_scenes'], settings['critical_states_reference']]
    write_json(dict(config_sha256=common.canonical_hash(config), scene_count=500, candidate_draws=draw,
                    rejected=dict(rejected), obstacle_count_distribution=dict(Counter(len(s['obstacles']) for s in scenes)),
                    unique_route_combinations=500, unique_physical_scenes=500,
                    old_50_overlap=0, previous_500_overlap=0,
                    created_at_utc=datetime.now(timezone.utc).isoformat(), frozen_before_new_training=True,
                    training_started=False, model_performance_used_for_scene_selection=False,
                    training_access_allowed=False, scheduling_access_allowed=False,
                    files_sha256={name: common.sha256(directory / name) for name in
                                  ('independent_final_scenarios.json', 'critical_states.json')},
                    source_data_sha256={name: common.sha256(common.ROOT / name) for name in paths},
                    registered_training_source_sha256=common.source_hashes(),
                    scope='Same fixed map/start/goal/route pool; new combinations excluding old 50 and old 500, not cross-map generalization or guaranteed separation from random training.'),
               directory / 'freeze_manifest.json')
    common.load_frozen(config)
    print(f'Frozen 500 scenes, {draw} draws, rejected={dict(rejected)}: {directory}')
    return directory


if __name__ == '__main__':
    prepare(common.legacy.load_config(common.CONFIG))
