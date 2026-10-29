"""Network-only static pairs, original dynamic-score reuse, post-execution cases.

Additional dynamic POLICY replay is disabled unless separately authorized. Existing
recorded actions can always be replayed offline without running a policy again.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch

import multitask_external_common as c
from astar_d3qn.agents.d3qn import D3QNAgent,D3QNConfig
from astar_d3qn.utils.seed import seed_everything
from multitask_pilot_domain import make_env,observation_risk,hard_gate
from multitask_pilot_engine import state_digest,global_rng_state


def csv_rows(path):
    import csv
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def behavior_metrics(positions,actions,goal,planner):
    positions=list(map(tuple,positions));actions=list(map(int,actions))
    seen={positions[0]};revisit_all=0;revisit_moves=0;move_positions=[positions[0]];move_count=0
    for old,new in zip(positions,positions[1:]):
        revisit_all+=new in seen
        if new!=old:
            move_count+=1;revisit_moves+=new in seen;move_positions.append(new)
        seen.add(new)
    covered=set();best_period=None;best_repeats=0
    # Three complete movement cycles, period 2..12, with at least two cells.
    for end in range(6,len(move_positions)+1):
        for period in range(2,min(12,end//3)+1):
            block=move_positions[end-period:end]
            if len(set(block))<2:continue
            if block==move_positions[end-2*period:end-period]==move_positions[end-3*period:end-2*period]:
                repetitions=3
                while end-(repetitions+1)*period>=0 and move_positions[end-(repetitions+1)*period:end-repetitions*period]==block:
                    repetitions+=1
                covered.update(range(max(1,end-repetitions*period),end))
                if repetitions>best_repeats:best_repeats=repetitions;best_period=period
    return dict(wait_fraction=sum(a==4 for a in actions)/max(1,len(actions)),
        successful_moves=move_count,revisit_move_count=revisit_moves,revisit_move_ratio=revisit_moves/max(1,move_count),
        revisit_all_ratio=revisit_all/max(1,len(actions)),periodic_movement=bool(covered),
        periodic_movement_ratio=len(covered)/max(1,move_count),cycle_period=best_period,max_cycle_repetitions=best_repeats,
        final_position=str(positions[-1]),final_static_distance=int(planner.fields(goal)[0][positions[-1]]),
        trace_metrics_available=True)


def summary(details):
    result=dict(episodes=len(details))
    for name in ('success','collision','timeout'):result[name+'_rate']=float(np.mean([r[name] for r in details]))
    steps=sum(r['steps'] for r in details)
    result['encounter_rate']=sum(r['encounter_steps'] for r in details)/max(1,steps)
    result['conflict_opportunity_rate']=sum(r['conflict_steps'] for r in details)/max(1,steps)
    result['encounter_episode_rate']=float(np.mean([r['encounter_steps']>0 for r in details]))
    result['conflict_episode_rate']=float(np.mean([r['conflict_steps']>0 for r in details]))
    for band in ('short','medium','long'):
        selected=[r for r in details if r['band']==band]
        for name in ('success','collision','timeout'):result[f'{band}_{name}_rate']=float(np.mean([r[name] for r in selected])) if selected else None
    return result


def network_suite(agent,cfg,data,planner,scenes,condition='dynamic',subset=None):
    tasks={t['task_id']:t for t in data['pools']['validation']}
    before=state_digest(agent.training_state_dict());rng=state_digest(global_rng_state())
    details=[];traces=[]
    for index,original in enumerate(scenes):
        if subset is not None and original['task_id'] not in subset:continue
        task=tasks[original['task_id']]
        scene=dict(original,obstacles=[]) if condition=='static' else original
        env=make_env(task,scene,planner,cfg);state=env.reset();actions=[];positions=[env.position]
        while True:
            # Explicit base inference: no overridden collection, teacher or shield.
            action=D3QNAgent.select_action(agent,state,0.,np.flatnonzero(env.action_mask(True)).tolist())
            outcome=env.step(action);actions.append(action);positions.append(env.position);state=outcome.observation
            if outcome.done:break
        details.append(dict(scenario_index=index,task_id=task['task_id'],band=task['band'],length=task['length'],
            scene_seed=original['scene_seed'],success=int(outcome.info['reached']),collision=int(outcome.info['collision']),
            timeout=int(outcome.truncated),steps=env.steps,wait_steps=sum(a==4 for a in actions),
            encounter_steps=env.encounter_steps,conflict_steps=env.conflict_opportunity_steps,
            **behavior_metrics(positions,actions,tuple(task['goal']),planner)))
        traces.append(dict(task_id=task['task_id'],actions=actions,positions=positions))
    if state_digest(agent.training_state_dict())!=before or state_digest(global_rng_state())!=rng:
        raise ValueError('Network evaluation modified training/RNG state')
    return details,traces


def reference_groups(cfg,data,planner,scenes):
    tasks={t['task_id']:t for t in data['pools']['validation']};groups=[]
    for scene in scenes:
        task=tasks[scene['task_id']];env=make_env(task,scene,planner,cfg);env.reset();path=planner.path(task['start'],task['goal'])
        conflict=0;first=None
        for tick,position in enumerate(path[:-1]):
            env.position=position;action=planner.action(task['goal'],position)
            hit=env.dynamic_action_collision_risk(action,predict_next=True)
            if hit and first is None:first=tick
            conflict+=int(hit)
            env._indices,env._directions,env._move_counters,_=env._next_dynamic_state()
        groups.append(dict(task_id=task['task_id'],band=task['band'],reference_steps=len(path)-1,
            reference_conflict_steps=conflict,first_reference_conflict_tick=first,
            reference_group='reference_conflict' if conflict else 'reference_clear'))
    return groups


def ensure_protocol():
    cfg,data,planner=c.data();c.pilot.verify_freeze()
    scenes=c.read(c.pilot.REG/'validation_scenes.json')
    protocol=dict(shared_identity=c.shared_identity(),source_sha256=c.sha(c.ROOT/'scripts/diagnose_multitask_pairs.py'),
        tasks_sha256=c.sha(c.pilot.REG/'task_pools.json'),scenes_sha256=c.sha(c.pilot.REG/'validation_scenes.json'),
        conditions=['dynamic','static'],teacher=False,epsilon=0,dynamic_shield=False,updates=0,
        grouping='Same frozen optimal-field nominal path and true old/new dynamic occupancy at nominal step; independent of every learned policy.',
        movement_metrics=c.config()['diagnostics'],test_pools_used=False)
    c.immutable(c.REG/'paired_diagnostic_protocol.json',protocol)
    groups=reference_groups(cfg,data,planner,scenes)
    c.immutable(c.REG/'paired_reference_groups.json',groups)
    return cfg,data,planner,scenes,groups


def load_agent(info,cfg):
    seed_everything(info['seed'])
    values=dict(cfg['agent']);values['spatial_shape']=tuple(values['spatial_shape'])
    # Match the original validation device rather than silently changing backend.
    values['device']=c.read(info['directory']/'run_manifest.json')['device']
    agent=D3QNAgent(D3QNConfig(seed=info['seed'],**values));agent.load_weights(info['model'])
    return agent


def trace_from_actions(task,scene,planner,cfg,actions,expected_positions=None):
    env=make_env(task,scene,planner,cfg);env.reset();positions=[env.position]
    for i,action in enumerate(actions):
        if expected_positions is not None and tuple(expected_positions[i])!=env.position:raise ValueError('Saved trajectory position differs')
        outcome=env.step(int(action));positions.append(env.position)
        if outcome.done and i!=len(actions)-1:raise ValueError('Saved actions continue after episode end')
    if not outcome.done:raise ValueError('Saved trajectory is incomplete')
    if expected_positions is not None and len(expected_positions)==len(actions)+1 and tuple(expected_positions[-1])!=env.position:
        raise ValueError('Saved final position differs')
    return dict(task_id=task['task_id'],actions=list(map(int,actions)),positions=positions),outcome,env


def existing_dynamic(info,cfg,data,planner,scenes):
    directory=info['directory'];tasks={t['task_id']:t for t in data['pools']['validation']};scene_by_id={s['task_id']:s for s in scenes}
    curve=[r for r in csv_rows(directory/'validation_curve.csv') if int(r['environment_steps_total'])==500000]
    source=[r for r in csv_rows(directory/'validation_details.csv') if int(r['environment_steps_total'])==500000]
    if len(curve)!=1 or len(source)!=50 or [r['task_id'] for r in source]!=[s['task_id'] for s in scenes]:raise ValueError('Final frozen validation identity/order differs')
    details=[]
    for record,scene in zip(source,scenes):
        if int(record['scene_seed'])!=scene['scene_seed']:raise ValueError('Scene identity differs')
        row={k:int(float(record[k])) for k in ('scenario_index','length','scene_seed','success','collision','timeout','steps','wait_steps','encounter_steps','conflict_steps')}
        row.update(task_id=record['task_id'],band=record['band'],wait_fraction=row['wait_steps']/row['steps'],
            revisit_move_ratio=None,revisit_all_ratio=None,periodic_movement=None,periodic_movement_ratio=None,
            cycle_period=None,max_cycle_repetitions=None,final_position=None,final_static_distance=None,trace_metrics_available=False,
            score_source='reused_original_final_validation')
        details.append(row)
    for metric in ('success','collision','timeout'):
        if abs(np.mean([r[metric] for r in details])-float(curve[0][metric+'_rate']))>1e-12:raise ValueError('Reused dynamic aggregate differs')
    traces={}
    full=directory/'final_validation_trajectories.json'
    if full.exists():traces={r['task_id']:r for r in c.read(full)}
    else:
        cases=directory/'offline_cases.csv'
        if cases.exists():
            records={}
            for row in csv_rows(cases):
                if int(row['environment_steps_total'])==500000:records.setdefault(row['task_id'],[]).append(row)
            for task_id,sequence in records.items():
                sequence=sorted(sequence,key=lambda r:int(r['tick']))
                trace,outcome,env=trace_from_actions(tasks[task_id],scene_by_id[task_id],planner,cfg,
                    [int(r['action']) for r in sequence],[ast.literal_eval(r['position']) for r in sequence])
                traces[task_id]=trace
    for row in details:
        if row['task_id'] in traces:
            trace=traces[row['task_id']]
            replay,outcome,env=trace_from_actions(tasks[row['task_id']],scene_by_id[row['task_id']],planner,cfg,trace['actions'],trace['positions'])
            if (int(outcome.info['reached']),int(outcome.info['collision']),int(outcome.truncated),env.steps)!=(row['success'],row['collision'],row['timeout'],row['steps']):raise ValueError('Recorded-action replay differs from original dynamic score')
            row.update(behavior_metrics(replay['positions'],replay['actions'],tuple(tasks[row['task_id']]['goal']),planner))
    return details,traces


def offline_case(task,scene,trace,cfg,planner):
    env=make_env(task,scene,planner,cfg);state=env.reset();output=[]
    for tick,action in enumerate(trace['actions']):
        position=env.position
        if position!=tuple(trace['positions'][tick]):raise ValueError('Offline case trajectory differs')
        teacher=planner.action(task['goal'],position);risk,stats=observation_risk(state,position,tuple(task['goal']),planner,cfg['risk'])
        teacher_next=env.dynamic_action_collision_risk(teacher,predict_next=True)
        actual_next=env.dynamic_action_collision_risk(action,predict_next=True)
        indices,directions,counters=list(env._indices),list(env._directions),list(env._move_counters)
        robot=position;first=None
        for k in range(1,5):
            old=env.dynamic_positions;next_indices,next_directions,next_counters,new=env._next_dynamic_state()
            planned=planner.action(task['goal'],robot);delta=__import__('astar_d3qn.core.grid',fromlist=['ACTION_DELTAS']).ACTION_DELTAS[planned]
            candidate=robot[0]+delta[0],robot[1]+delta[1]
            if candidate in old or candidate in new:
                if first is None:first=k
            robot=candidate;env._indices,env._directions,env._move_counters=next_indices,next_directions,next_counters
        env._indices,env._directions,env._move_counters=indices,directions,counters
        allowed=hard_gate(state,teacher,cfg['risk']['hard_gate_radius'])
        output.append(dict(task_id=task['task_id'],tick=tick,position=str(position),action=action,
            teacher_action=teacher,agrees_with_teacher=int(action==teacher),waits=int(action==4),risk_R=risk,
            hard_label_available=allowed,hypothetical_C_weight=.1 if allowed else 0.,hypothetical_D_weight=.1*(1-risk)**2 if allowed else 0.,
            actual_true_next_conflict=actual_next,teacher_true_next_conflict=teacher_next,
            teacher_true_first_conflict_within4=first,offline_only=True,**stats))
        outcome=env.step(action);state=outcome.observation
    return output


def representatives(details,traces):
    selected=[]
    for condition in ('success','collision','timeout'):
        choices=[r for r in details if r[condition] and r['task_id'] in traces]
        if not choices:continue
        if condition=='timeout':choice=sorted(choices,key=lambda r:(-(r.get('periodic_movement_ratio') or 0),r['task_id']))[0]
        else:choice=min(choices,key=lambda r:r['task_id'])
        if choice['task_id'] not in selected:selected.append(choice['task_id'])
    return selected


def run(method,seed,augment_dynamic=False):
    cfg,data,planner,scenes,groups=ensure_protocol();info=c.model_info(method,seed)
    if info is None:return False
    if augment_dynamic:
        authorization=c.REG/'dynamic_replay_authorization.json'
        if not authorization.exists() or not c.read(authorization).get('authorized'):
            raise ValueError('Additional dynamic policy replay has not been explicitly authorized')
    tasks={t['task_id']:t for t in data['pools']['validation']};scene_by_id={s['task_id']:s for s in scenes};group_by_id={g['task_id']:g for g in groups}
    base=c.RESULTS/'paired'/method/f'seed_{seed}';agent=None
    for condition in ('dynamic','static'):
        path=base/condition;done=path/'completion.json'
        identity=dict(method=method,seed=seed,condition=condition,model_sha256=info['model_sha256'],
            paired_protocol_sha256=c.sha(c.REG/'paired_diagnostic_protocol.json'),scene_sha256=c.sha(c.pilot.REG/'validation_scenes.json'))
        if done.exists():
            previous=c.read(done)
            if previous['identity']!=identity:raise ValueError('Existing paired result identity differs')
            if condition!='dynamic' or not augment_dynamic or previous.get('full_trace_coverage'):continue
        if condition=='dynamic':
            details,traces=existing_dynamic(info,cfg,data,planner,scenes)
            missing={r['task_id'] for r in details if not r['trace_metrics_available']}
            if missing and augment_dynamic:
                if agent is None:agent=load_agent(info,cfg)
                new,new_traces=network_suite(agent,cfg,data,planner,scenes,'dynamic',missing)
                new_by_id={r['task_id']:r for r in new}
                for row in details:
                    if row['task_id'] in new_by_id:
                        candidate=new_by_id[row['task_id']]
                        for key in ('success','collision','timeout','steps','wait_steps','encounter_steps','conflict_steps'):
                            if candidate[key]!=row[key]:raise ValueError('Supplementary dynamic replay differs from original score: '+row['task_id'])
                        row.update(candidate);row['score_source']='reused_original_score_verified_by_diagnostic_replay'
                traces.update({r['task_id']:r for r in new_traces})
        else:
            if agent is None:agent=load_agent(info,cfg)
            details,trace_list=network_suite(agent,cfg,data,planner,scenes,'static');traces={r['task_id']:r for r in trace_list}
            for row in details:row['score_source']='new_static_pair_only'
        for row in details:row.update(reference_group=group_by_id[row['task_id']]['reference_group'])
        c.csv_write(path/'per_task.csv',details);c.write(path/'summary.json',summary(details))
        c.write(path/'trajectories.json',list(traces.values()))
        case_ids=representatives(details,traces);analysis=[]
        for task_id in case_ids:
            scene=dict(scene_by_id[task_id],obstacles=[]) if condition=='static' else scene_by_id[task_id]
            analysis.extend(offline_case(tasks[task_id],scene,traces[task_id],cfg,planner))
        if analysis:c.csv_write(path/'offline_cases.csv',analysis)
        c.write(path/'representatives.json',dict(task_ids=case_ids,selection_rule=c.config()['diagnostics']['representative_rule'],offline_only=True))
        c.write(done,dict(identity=identity,complete=True,full_trace_coverage=all(r['trace_metrics_available'] for r in details),
            trace_tasks=len(traces),policy_scores_reused=condition=='dynamic',extra_dynamic_policy_episodes=len(missing) if condition=='dynamic' and augment_dynamic else 0,
            completed=datetime.now().astimezone().isoformat(timespec='seconds')))
        print(f'PAIRED {method} seed{seed} {condition}: success={np.mean([r["success"] for r in details]):.3f}, traces={len(traces)}/50',flush=True)
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=c.METHODS,default=list(c.METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,choices=(0,1),default=[0,1]);parser.add_argument('--augment-dynamic',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    for method in args.methods:
        for seed in args.seeds:run(method,seed,args.augment_dynamic)
