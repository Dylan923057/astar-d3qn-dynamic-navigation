"""Fixed-budget multi-task demonstrations, generated before baseline training."""
from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np

import multitask_external_common as c
from multitask_pilot_domain import TaskTransition,hard_gate,make_env,scene_for
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.dqfd import NStepCollector


def generate():
    settings=c.config()['demonstrations'];cfg,data,planner=c.data()
    plan=dict(settings=settings,shared_identity=c.shared_identity(),demonstration_source_sha256={
                  'scripts/multitask_external_demos.py':c.sha(c.ROOT/'scripts/multitask_external_demos.py'),
                  'scripts/multitask_external_common.py':c.sha(c.ROOT/'scripts/multitask_external_common.py'),
                  'configs/asvc_multitask_external_v1.yaml':c.sha(c.CONFIG)},
              generated_before_training=True,validation_or_test_used=False)
    c.immutable(c.REG/'demonstration_plan.json',plan)
    if (c.REG/'demonstration_report.json').exists():
        report=c.read(c.REG/'demonstration_report.json')
        if c.sha(c.REG/'demonstrations.npz')!=report['demonstrations_sha256']:raise ValueError('Demonstration identity changed')
        return report
    if (c.REG/'demonstrations.npz').exists():raise ValueError('Partial demonstration generation retained; inspect before continuing')
    attempts=[];retained=[];coverage=[];clock=perf_counter();lookups=0
    tasks=sorted(data['pools']['train'],key=lambda t:t['task_id'])
    if len(tasks)!=240 or settings['total_attempts']!=480:raise ValueError('Frozen demonstration budget differs')
    for index,task in enumerate(tasks):
        counts=Counter()
        for attempt in range(settings['attempts_per_training_task']):
            ordinal=index*settings['attempts_per_training_task']+attempt
            scene_seed=settings['scene_seed_base']+10*index+attempt
            scene=scene_for(task,planner,cfg['scenes'],scene_seed,(3,4,5)[ordinal%3])
            env=make_env(task,scene,planner,cfg);state=env.reset();episode=[];waits=0
            while True:
                position=env.position;teacher=planner.action(task['goal'],position);lookups+=1
                action=teacher if hard_gate(state,teacher,cfg['risk']['hard_gate_radius']) else 4
                waits+=int(action==4);result=env.step(action)
                episode.append(TaskTransition(state,action,result.reward,result.observation,result.terminated,
                    env.action_mask(True),task['task_id'],position,env.position,result.truncated,bool(result.info['collision']),0.,{}))
                state=result.observation
                if result.done:break
            success=bool(result.info['reached']) and not any(t.collision for t in episode)
            offset=len(retained) if success else None
            if success:retained.extend(episode)
            counts.update(attempts=1,successes=int(success),collisions=int(result.info['collision']),timeouts=int(result.truncated),
                          attempted_steps=len(episode),retained_steps=len(episode) if success else 0)
            attempts.append(dict(ordinal=ordinal,task_id=task['task_id'],goal=task['goal'],start=task['start'],band=task['band'],
                scene=scene,steps=len(episode),wait_steps=waits,retained=success,retained_offset=offset,
                termination_reason=result.info['termination_reason']))
        coverage.append(dict(task_id=task['task_id'],band=task['band'],goal=str(task['goal']),**counts))
    if len(attempts)!=480 or len(retained)<64:raise ValueError('Frozen attempts produced insufficient demonstrations; no budget increase allowed')
    c.REG.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(c.REG/'demonstrations.npz',
        spatial=np.stack([t.state.spatial for t in retained]).astype(np.uint8),
        scalars=np.stack([t.state.scalars for t in retained]),
        next_spatial=np.stack([t.next_state.spatial for t in retained]).astype(np.uint8),
        next_scalars=np.stack([t.next_state.scalars for t in retained]),
        action=np.array([t.action for t in retained],dtype=np.uint8),reward=np.array([t.reward for t in retained],dtype=np.float64),
        terminated=np.array([t.terminated for t in retained]),truncated=np.array([t.truncated for t in retained]),
        next_mask=np.stack([t.next_action_mask for t in retained]),task_id=np.array([t.task_id for t in retained]),
        position=np.array([t.position for t in retained],dtype=np.int16),next_position=np.array([t.next_position for t in retained],dtype=np.int16))
    c.write(c.REG/'demonstration_attempts.json',attempts);c.csv_write(c.REG/'demonstration_task_coverage.csv',coverage)
    report=dict(attempted_episodes=480,retained_episodes=sum(a['retained'] for a in attempts),failed_episodes=sum(not a['retained'] for a in attempts),
        transitions=len(retained),attempted_environment_steps=sum(a['steps'] for a in attempts),
        attempted_training_tasks=len(coverage),retained_training_tasks=sum(r['successes']>0 for r in coverage),
        attempted_goals=len({tuple(a['goal']) for a in attempts}),retained_goals=len({tuple(a['goal']) for a in attempts if a['retained']}),
        teacher_lookups=lookups,optimal_field_cache_entries=len(planner.cache),generation_seconds=perf_counter()-clock,
        demonstrations_sha256=c.sha(c.REG/'demonstrations.npz'),validation_or_test_used=False,
        source='fresh_multi_task_training_pool_only',old_fixed_goal_demonstrations_used=False,
        cost_is_shared_generation_not_multiplied_by_training_seed=True,offline_updates_per_dqfd_seed=20000)
    c.write(c.REG/'demonstration_report.json',report)
    return report


def load_records(settings):
    report=c.read(c.REG/'demonstration_report.json')
    if c.sha(c.REG/'demonstrations.npz')!=report['demonstrations_sha256']:raise ValueError('Demonstration SHA mismatch')
    arrays=np.load(c.REG/'demonstrations.npz',allow_pickle=False)
    collector=NStepCollector(settings['n_steps'],.99);records=[]
    for i in range(len(arrays['action'])):
        item=TaskTransition(Observation(arrays['spatial'][i],arrays['scalars'][i]),int(arrays['action'][i]),float(arrays['reward'][i]),
            Observation(arrays['next_spatial'][i],arrays['next_scalars'][i]),bool(arrays['terminated'][i]),arrays['next_mask'][i],
            str(arrays['task_id'][i]),tuple(map(int,arrays['position'][i])),tuple(map(int,arrays['next_position'][i])),
            bool(arrays['truncated'][i]),False,0.,{})
        records.extend(collector.add(item,episode_end=item.terminated or item.truncated))
    if collector.pending or len(records)!=report['transitions']:raise ValueError('Incomplete demonstration episode/n-step boundary')
    return records
