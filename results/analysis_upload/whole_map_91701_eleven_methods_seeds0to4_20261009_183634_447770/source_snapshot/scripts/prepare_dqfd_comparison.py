"""Freeze A* demonstration data and hyperparameters before any new training, without updates."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

import dqfd_comparison_common as common
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.io import write_json


def prepare(config):
    inputs = common.validate_config(config)
    root = common.artifact_root(config)
    if root.exists():
        common.frozen(config)
        print(f'Existing immutable demo registration verified: {root}')
        return root
    if any(common.run_directory(config, m, s).exists() for m in common.METHODS for s in common.SEEDS):
        raise ValueError('Registration must precede all new formal runs.')
    tail_config = common.legacy.load_config(common.tail.CONFIG)
    final_scenes, _, final_manifest = common.tail.load_frozen(tail_config)
    excluded = common.tail.excluded_scenes(tail_config)+final_scenes
    combinations = {common.previous.combination_key(s) for s in excluded}
    physical = {common.previous.physical_key(s) for s in excluded}
    settings, problem = config['comparison'], inputs[0]
    factory = common.legacy.runtime.make_factory(config, *inputs, method='unguided',
                                                seed=settings['demonstration_seed'])
    teacher = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG), problem,
                                       settings['demonstration_seed'], 'advice_bound_margin', 'cpu')
    kwargs = dict(max_steps=300, window_size=15, reward_config=common.legacy._reward_config(config),
                  terminate_on_collision=True)
    kept, episodes, lookup_count, searches, planner_seconds, skipped = [], [], 0, 0, 0., 0
    started = perf_counter()
    for draw in range(1, settings['demonstration_max_scene_draws']+1):
        env = factory(problem, **kwargs)
        scene = dict(scenario_id=env.scenario_id, start=list(problem.start), goal=list(problem.goal),
                     route_ids=list(env.dynamic_route_ids), categories=list(env.dynamic_route_categories),
                     obstacles=[asdict(s) for s in env.dynamic_obstacles])
        if (common.previous.combination_key(scene) in combinations or
                common.previous.physical_key(scene) in physical):
            skipped += 1
            continue
        state, transitions = env.reset(), []
        waits = 0
        while True:
            scale = max(1, problem.size-1)
            position = tuple(problem.goal[i]-int(round(float(state.scalars[i])*scale)) for i in (0, 1))
            searches += int(position not in teacher._static_action_cache)
            clock = perf_counter()
            action = teacher.static_advice(state)
            planner_seconds += perf_counter()-clock
            lookup_count += 1
            if action is None or not env.action_mask(True)[action] or teacher.observed_risk(state, action):
                action = 4
            waits += int(action == 4)
            result = env.step(action)
            transitions.append(Transition(state, action, result.reward, result.observation,
                                           result.terminated, env.action_mask(True)))
            state = result.observation
            if result.done:
                break
        success = bool(result.info['reached']) and not any(t.terminated and t.reward == -1 for t in transitions)
        episodes.append(dict(**scene, steps=len(transitions), retained=success, wait_steps=waits,
                             termination_reason=result.info.get('termination_reason'),
                             retained_offset=len(kept) if success else None))
        if success:
            kept.extend(transitions)
        if len(episodes) == settings['demonstration_attempts']:
            break
    if len(episodes) != settings['demonstration_attempts'] or len(kept) < config['training']['batch_size']:
        raise ValueError('Fixed collection budget yielded insufficient data; no automatic budget adjustment.')
    for t in kept:
        if (np.any((t.state.spatial != 0) & (t.state.spatial != 1)) or
                np.any((t.next_state.spatial != 0) & (t.next_state.spatial != 1))):
            raise ValueError('Binary spatial storage would lose observation information.')
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root/'demonstrations.npz',
        spatial=np.stack([t.state.spatial for t in kept]).astype(np.uint8),
        scalars=np.stack([t.state.scalars for t in kept]),
        next_spatial=np.stack([t.next_state.spatial for t in kept]).astype(np.uint8),
        next_scalars=np.stack([t.next_state.scalars for t in kept]),
        action=np.array([t.action for t in kept], dtype=np.uint8),
        reward=np.array([t.reward for t in kept], dtype=np.float64),
        terminated=np.array([t.terminated for t in kept]),
        next_mask=np.stack([t.next_action_mask for t in kept]),
        episode_end=np.array([t.terminated for t in kept]))
    write_json(dict(episodes=episodes, no_model_or_validation_performance_used=True,
                    collection_filter='Successful executed training trajectories, no future obstacle information.'),
               root/'demonstration_episodes.json')
    digest = common.canonical_hash(config)
    write_json(dict(config=config, config_sha256=digest, source_sha256=common.source_hashes(),
        files_sha256={n: common.sha256(root/n) for n in ('demonstrations.npz', 'demonstration_episodes.json')},
        created_at_utc=datetime.now(timezone.utc).isoformat(), frozen_before_new_training=True,
        final_scene_sha256=final_manifest['files_sha256']['independent_final_scenarios.json'],
        demo_transition_count=len(kept), attempted_episodes=len(episodes),
        retained_episodes=sum(e['retained'] for e in episodes), rejected_heldout_scene_draws=skipped,
        candidate_draws=draw, demo_astar_lookups=lookup_count, demo_astar_searches=searches,
        demo_astar_seconds=planner_seconds, demonstration_generation_seconds=perf_counter()-started,
        gradient_updates=0, formal_training_started=False, independent_500_performance_evaluated=False,
        demo_policy_uses_future_obstacles=False, demo_heldout_combinations_overlap=0,
        pretraining_updates_per_seed=settings['pretraining_updates']), root/'freeze_manifest.json')
    common.load_demonstrations(config)
    print(f'Frozen {len(kept)} transitions from {sum(e["retained"] for e in episodes)}/{len(episodes)} episodes: {root}')
    return root


if __name__ == '__main__':
    prepare(common.legacy.load_config(common.CONFIG))
