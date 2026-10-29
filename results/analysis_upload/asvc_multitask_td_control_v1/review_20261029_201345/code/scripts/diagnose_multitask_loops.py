"""Frozen greedy trajectories first; Q/teacher/risk/future annotations only later."""
from __future__ import annotations

import argparse
from collections import Counter,deque
from datetime import datetime
import gc
import json

import numpy as np
import torch

import td_control_common as c
from diagnose_multitask_pairs import (behavior_metrics,csv_rows,existing_dynamic,load_agent,network_suite,summary)
from multitask_pilot_domain import make_env,hard_gate,observation_risk
from multitask_pilot_engine import state_digest,global_rng_state
from astar_d3qn.core.grid import ACTION_DELTAS
from run_multitask_pilot import run_lock


SCORE_KEYS=('success','collision','timeout','steps','wait_steps','encounter_steps','conflict_steps')


def protocol():
    spec=dict(source_sha256=c.sha(__file__),config_sha256=c.sha(c.CONFIG),
        upstream_identity=c.upstream_identity(),authorization_sha256=c.sha(c.REG/'dynamic_replay_authorization.json'),
        actual_policy='D3QNAgent greedy, original static mask, epsilon0, no update/teacher/shield',
        annotations='Only after every requested policy trajectory finished; Q recomputed with same final weights.',
        q_rank='1 + number of legal actions greater than teacher Q by >1e-6',
        teacher_check='Independent reverse BFS, UP/DOWN/LEFT/RIGHT tie break',
        weights='Actual evaluation supervision is always0. Hypothetical state C=.1*z, D=.1*z*(1-R)^2; replay variant additionally rejects actual collision transitions.',
        static_pair='Remove all dynamic obstacles; this also shifts input distribution and is not static training.')
    import hashlib
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    c.immutable(c.REG/'loop_protocols'/f'{digest}.json',spec)
    root=c.RESULTS/'loop_diagnostics'/digest[:12]
    c.write(c.REG/'loop_latest.json',dict(protocol_sha256=digest,protocol_path=str((c.REG/'loop_protocols'/f'{digest}.json').relative_to(c.ROOT)),
        results_path=str(root.relative_to(c.ROOT))))
    return root,digest


class IndependentFields:
    def __init__(self,data):self.size=data['size'];self.blocked=set(map(tuple,data['obstacles']));self.cache={}
    def distance(self,goal):
        goal=tuple(goal)
        if goal not in self.cache:
            distances={goal:0};queue=deque([goal])
            while queue:
                point=queue.popleft()
                for dr,dc in ACTION_DELTAS[:4]:
                    nxt=point[0]+dr,point[1]+dc
                    if 0<=nxt[0]<self.size and 0<=nxt[1]<self.size and nxt not in self.blocked and nxt not in distances:
                        distances[nxt]=distances[point]+1;queue.append(nxt)
            self.cache[goal]=distances
        return self.cache[goal]
    def action(self,goal,position):
        distances=self.distance(goal);position=tuple(position)
        if position==tuple(goal):return 4
        if position not in distances:return -1
        for action,(dr,dc) in enumerate(ACTION_DELTAS[:4]):
            if distances.get((position[0]+dr,position[1]+dc))==distances[position]-1:return action
        return -1


def q_values(agent,state):
    with torch.no_grad():
        values=agent.policy_network(torch.as_tensor(state.spatial,dtype=torch.float32,device=agent.device).unsqueeze(0),
            torch.as_tensor(state.scalars,dtype=torch.float32,device=agent.device).unsqueeze(0))[0]
    return values.detach().cpu().numpy()


def true_teacher_conflict_within4(env,goal,planner):
    # Offline oracle only. Risk estimation never receives env or this result.
    saved=env.position,list(env._indices),list(env._directions),list(env._move_counters)
    first=None
    try:
        for k in range(1,5):
            action=planner.action(goal,env.position)
            if env.dynamic_action_collision_risk(action,predict_next=True):first=k;break
            indices,directions,counters,_=env._next_dynamic_state()
            delta=ACTION_DELTAS[action];env.position=env.position[0]+delta[0],env.position[1]+delta[1]
            env._indices,env._directions,env._move_counters=indices,directions,counters
            if env.position==tuple(goal):break
    finally:env.position,env._indices,env._directions,env._move_counters=saved
    return first


def annotate_trace(agent,task,scene,trace,cfg,planner,independent,export_steps=True):
    env=make_env(task,scene,planner,cfg);state=env.reset();rows=[];counts=Counter();positions=[env.position]
    goal=tuple(task['goal']);distances=independent.distance(goal);previous_position=None
    for tick,action in enumerate(trace['actions']):
        if env.position!=tuple(trace['positions'][tick]):raise ValueError('Saved-action reconstruction position mismatch')
        position=env.position;legal=env.action_mask(True).astype(bool);q=q_values(agent,state)
        if not np.isfinite(q).all():raise FloatingPointError('Nonfinite final-model Q during offline diagnostic')
        teacher=planner.action(goal,position);expected_teacher=independent.action(goal,position)
        expected_goal=np.array([(goal[0]-position[0])/(planner.size-1),(goal[1]-position[1])/(planner.size-1),0,0],dtype=np.float32)
        goal_ok=bool(np.allclose(state.scalars,expected_goal,rtol=0,atol=1e-6));teacher_ok=teacher==expected_teacher
        gate=hard_gate(state,teacher,cfg['risk']['hard_gate_radius'])
        risk,risk_stats=observation_risk(state,position,goal,planner,cfg['risk'])
        selected=int(np.argmax(np.where(legal,q,-np.inf)))
        rank=1+int(np.sum(q[legal]>q[teacher]+1e-6))
        other=[a for a in range(5) if legal[a] and a!=teacher]
        best_other=max(other,key=lambda a:(q[a],-a)) if other else None
        retreats=[]
        for a,(dr,dc) in enumerate(ACTION_DELTAS[:4]):
            if legal[a] and distances.get((position[0]+dr,position[1]+dc),-1)>distances[position]:retreats.append(a)
        retreat=max(retreats,key=lambda a:(q[a],-a)) if retreats else None
        true_actual=env.dynamic_action_collision_risk(action,predict_next=True)
        true_teacher=env.dynamic_action_collision_risk(teacher,predict_next=True)
        future_conflict=true_teacher_conflict_within4(env,goal,planner)
        outcome=env.step(int(action));next_position=env.position;positions.append(next_position)
        replay_eligible=bool(gate and not outcome.info['collision'])
        counts.update(steps=1,goal_input_mismatch=int(not goal_ok),teacher_label_mismatch=int(not teacher_ok),
            saved_action_not_current_greedy=int(selected!=action),teacher_not_top_rank=int(rank>1),
            teacher_loses_to_retreat=int(retreat is not None and q[retreat]>q[teacher]+1e-6),
            hard_eligible_steps=int(gate),hypothetically_risk_reduced_steps=int(gate and risk>0),
            immediate_backtrack=int(previous_position is not None and next_position==previous_position and next_position!=position))
        if export_steps:
            row=dict(task_id=task['task_id'],tick=tick,position=str(position),next_position=str(next_position),
                goal=str(goal),executed_action=int(action),waits=int(action==4),distance_before=distances[position],
                distance_after=distances[next_position],teacher_action=teacher,independent_teacher_action=expected_teacher,
                teacher_label_correct=teacher_ok,goal_input_correct=goal_ok,goal_dr=float(state.scalars[0]),goal_dc=float(state.scalars[1]),
                current_masked_argmax=selected,saved_action_matches_current_greedy=selected==action,
                teacher_q=float(q[teacher]),teacher_q_legal_rank=rank,
                best_other_legal_action=best_other,teacher_minus_best_other_q=float(q[teacher]-q[best_other]) if best_other is not None else None,
                teacher_minus_executed_q=float(q[teacher]-q[action]),best_retreat_action=retreat,
                teacher_minus_best_retreat_q=float(q[teacher]-q[retreat]) if retreat is not None else None,
                hard_screen_pass=gate,risk_R=risk,replay_eligible_if_used=replay_eligible,
                actual_evaluation_supervision_weight=0.,hypothetical_state_C_weight=.1 if gate else 0.,
                hypothetical_state_D_weight=.1*(1-risk)**2 if gate else 0.,
                hypothetical_replay_C_weight=.1 if replay_eligible else 0.,
                hypothetical_replay_D_weight=.1*(1-risk)**2 if replay_eligible else 0.,
                true_actual_next_conflict=true_actual,true_teacher_next_conflict=true_teacher,
                true_teacher_first_conflict_within4=future_conflict,offline_only=True,
                q_source='same_final_network_offline_reconstruction',**risk_stats)
            row.update({f'q_{a}':float(q[a]) for a in range(5)})
            row.update({f'legal_{a}':bool(legal[a]) for a in range(5)})
            rows.append(row)
        previous_position=position;state=outcome.observation
        if outcome.done and tick!=len(trace['actions'])-1:raise ValueError('Saved trace continues beyond termination')
    if not outcome.done or tuple(trace['positions'][-1])!=env.position:raise ValueError('Incomplete/final trace mismatch')
    behavior=behavior_metrics(positions,trace['actions'],goal,planner)
    score=dict(task_id=task['task_id'],band=task['band'],length=task['length'],scene_seed=scene['scene_seed'],
        success=int(outcome.info['reached']),collision=int(outcome.info['collision']),timeout=int(outcome.truncated),
        steps=env.steps,wait_steps=sum(a==4 for a in trace['actions']),encounter_steps=env.encounter_steps,
        conflict_steps=env.conflict_opportunity_steps,**behavior)
    score.update(dict(counts))
    return score,rows


def previous_trace_candidates(method,seed,condition,model_sha):
    for done_path in sorted((c.RESULTS/'loop_diagnostics').glob(f'*/{method}/seed_{seed}/{condition}/completion.json'),reverse=True):
        done=c.read(done_path)
        if done.get('identity',{}).get('model_sha256')==model_sha and done.get('original_scores_verified'):
            yield done_path.parent


def condition_data(info,condition,cfg,data,planner,scenes,agent):
    method,seed=info['method'],info['seed'];new_policy_episodes=0
    if condition=='dynamic':
        source,traces=existing_dynamic(info,cfg,data,planner,scenes)
    else:
        old=c.previous.RESULTS/'paired'/method/f'seed_{seed}'/'static'
        if method!=c.METHOD and (old/'completion.json').exists():
            done=c.read(old/'completion.json')
            if done['identity']['model_sha256']!=info['model_sha256']:raise ValueError('Old static model differs')
            source=csv_rows(old/'per_task.csv');traces={r['task_id']:r for r in c.read(old/'trajectories.json')}
        else:source=[];traces={}
    for directory in previous_trace_candidates(method,seed,condition,info['model_sha256']):
        for trace in c.read(directory/'trajectories.json'):traces.setdefault(trace['task_id'],trace)
        if not source:source=csv_rows(directory/'per_task.csv')
    missing={s['task_id'] for s in scenes if s['task_id'] not in traces}
    if missing:
        if condition=='dynamic' and method in ('C','D'):
            auth=c.read(c.REG/'dynamic_replay_authorization.json')
            if not auth['authorized']:raise ValueError('Missing dynamic replay not authorized')
        fresh,fresh_traces=network_suite(agent,cfg,data,planner,scenes,condition,missing)
        expected={r['task_id']:r for r in source};mismatches=[]
        for row in fresh:
            if row['task_id'] in expected:
                for key in SCORE_KEYS:
                    if int(float(expected[row['task_id']][key]))!=int(row[key]):
                        mismatches.append(dict(task_id=row['task_id'],field=key,original=expected[row['task_id']][key],replayed=row[key]))
        if mismatches:
            path=c.RESULTS/'reconciliation_failures'/method/f'seed_{seed}'/condition
            c.write(path/'mismatches.json',mismatches);c.write(path/'candidate_trajectories.json',fresh_traces)
            raise ValueError('Greedy diagnostic differs from original scores; original retained; inspect '+str(path))
        if not source:source=fresh
        traces.update({r['task_id']:r for r in fresh_traces});new_policy_episodes=len(missing)
    return source,traces,new_policy_episodes


def run(method,seed):
    info=c.model_info(method,seed)
    if info is None:return False
    cfg,data,planner=c.shared();scenes=c.read(c.previous.pilot.REG/'validation_scenes.json')
    root,digest=protocol();base=root/method/f'seed_{seed}'
    tasks={t['task_id']:t for t in data['pools']['validation']};scene_by={s['task_id']:s for s in scenes}
    groups={r['task_id']:r['reference_group'] for r in c.read(c.previous.REG/'paired_reference_groups.json')}
    independent=IndependentFields(data);agent=None
    with run_lock(base/'diagnostic.lock'):
        for condition in ('dynamic','static'):
            path=base/condition;identity=dict(method=method,seed=seed,condition=condition,
                model_sha256=info['model_sha256'],loop_protocol_sha256=digest,
                scenes_sha256=c.sha(c.previous.pilot.REG/'validation_scenes.json'))
            if (path/'completion.json').exists():
                if c.read(path/'completion.json')['identity']!=identity:raise ValueError('Existing diagnostic identity differs')
                continue
            if agent is None:agent=load_agent(info,cfg)
            before=state_digest(agent.training_state_dict());rng=state_digest(global_rng_state())
            source,traces,new_episodes=condition_data(info,condition,cfg,data,planner,scenes,agent)
            # No teacher/risk annotations have been calculated by this routine
            # during the preceding policy executions. Now all decisions are fixed.
            original={r['task_id']:r for r in source};details=[];export=[];counts=Counter()
            ordered=[traces[s['task_id']] for s in scenes]
            for index,trace in enumerate(ordered):
                task=tasks[trace['task_id']];scene=scene_by[task['task_id']]
                if condition=='static':scene=dict(scene,obstacles=[])
                expected=original[task['task_id']]
                keep_steps=bool(int(float(expected['timeout']))) or index in (0,17,34)
                row,annotations=annotate_trace(agent,task,scene,trace,cfg,planner,independent,keep_steps)
                for key in SCORE_KEYS:
                    if int(float(expected[key]))!=int(row[key]):raise ValueError('Offline saved-action replay differs: '+task['task_id']+'/'+key)
                row.update(scenario_index=index,reference_group=groups[task['task_id']],original_score_verified=True,
                    original_score_source='reused_final_validation' if condition=='dynamic' else 'reused_static' if method!=c.METHOD else 'new_static',
                    actual_evaluation_supervision_weight=0.)
                details.append(row);export.extend(annotations)
                for key in ('goal_input_mismatch','teacher_label_mismatch','saved_action_not_current_greedy','teacher_not_top_rank','teacher_loses_to_retreat','steps'):
                    counts[key]+=row[key]
            if before!=state_digest(agent.training_state_dict()) or rng!=state_digest(global_rng_state()):raise ValueError('Diagnostic changed network/optimizer/RNG state')
            c.csv_write(path/'per_task.csv',details);c.csv_write(path/'timeout_and_reference_steps.csv',export)
            c.write(path/'trajectories.json',ordered);c.write(path/'summary.json',summary(details)|dict(offline_step_audit=dict(counts)))
            failures=[r for r in details if r['timeout']]
            selected=sorted(failures,key=lambda r:(-r['periodic_movement_ratio'],r['task_id']))[:3]
            waiting=sorted(failures,key=lambda r:(-r['wait_fraction'],r['task_id']))[:1]
            chosen=list(dict.fromkeys(r['task_id'] for r in [*selected,*waiting]))
            c.write(path/'representatives.json',dict(task_ids=chosen,rule='three largest cycle-ratio timeouts plus highest-wait timeout; ties by task id'))
            c.write(path/'completion.json',dict(identity=identity,complete=True,original_scores_verified=True,
                full_trace_coverage=True,trace_tasks=50,new_policy_episodes=new_episodes,
                actual_evaluation_supervision_weight=0.,updates=0,completed=datetime.now().astimezone().isoformat(timespec='seconds')))
            print(f'LOOPS {method}/{seed} {condition}: success={np.mean([r["success"] for r in details]):.3f}; all50 traces; new policy episodes={new_episodes}; goal errors={counts["goal_input_mismatch"]}, teacher errors={counts["teacher_label_mismatch"]}',flush=True)
    return True


def audit_replay(method,seed):
    if method not in ('C','D'):return
    root,_=protocol();target=root/'replay_audit'/f'{method}_seed_{seed}.json'
    if target.exists():return
    info=c.model_info(method,seed);cfg,data,planner=c.shared();tasks={t['task_id']:t for t in data['pools']['train']}
    independent=IndependentFields(data);checkpoint=info['directory']/'checkpoint_latest.pth'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);transitions=saved['replay']['transitions'];counts=Counter()
    for item in transitions:
        counts['transitions']+=1
        if item.task_id not in tasks:counts['unknown_task_id']+=1;continue
        task=tasks[item.task_id];goal=tuple(task['goal']);position=tuple(item.position)
        for state,point,key in ((item.state,position,'current_goal_mismatch'),(item.next_state,tuple(item.next_position),'next_goal_mismatch')):
            expected=np.array([(goal[0]-point[0])/(data['size']-1),(goal[1]-point[1])/(data['size']-1),0,0],dtype=np.float32)
            counts[key]+=int(not np.allclose(state.scalars,expected,rtol=0,atol=1e-6))
            counts['nonzero_route_channel']+=int(np.any(state.spatial[4]))
        counts['teacher_label_mismatch']+=int(planner.action(goal,position)!=independent.action(goal,position))
        risk,_=observation_risk(item.state,position,goal,planner,cfg['risk'])
        counts['stored_observation_risk_mismatch']+=int(abs(risk-item.risk)>1e-9)
    c.write(target,dict(method=method,seed=seed,checkpoint_sha256=c.sha(checkpoint),counts=dict(counts),
        scope='Final retained20000 transitions; does not reconstruct every historical replay draw.',
        new_updates=0,created=datetime.now().astimezone().isoformat(timespec='seconds')))
    del saved,transitions;gc.collect()
    print('REPLAY AUDIT',method,seed,dict(counts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=c.METHODS,default=list(c.METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,choices=c.SEEDS,default=list(c.SEEDS))
    parser.add_argument('--skip-replay-audit',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);c.verify(True)
    for method in args.methods:
        for seed in args.seeds:
            run(method,seed)
            if not args.skip_replay_audit:audit_replay(method,seed)
