"""Offline evidence for recurrent observations and fixed greedy action cycles."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from pathlib import Path

import td_control_common as c
from multitask_pilot_domain import make_env


def observation_id(state):
    digest=hashlib.sha256();digest.update(state.spatial.tobytes());digest.update(state.scalars.tobytes())
    return digest.hexdigest()


def functional_cycles(edges):
    """Cycles of length >=2 with a unique observed successor per state."""
    successor={state:next(iter(targets)) for state,targets in edges.items() if len(targets)==1}
    seen=set();cycles=[]
    for first in successor:
        trail=[];index={};state=first
        while state in successor and state not in seen:
            if state in index:
                nodes=trail[index[state]:]
                if len(nodes)>=2:cycles.append(nodes)
                break
            index[state]=len(trail);trail.append(state);state=successor[state]
        seen.update(trail)
    return cycles


def csv_rows(path):
    with path.open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def review(loop_root):
    cfg,data,planner=c.shared();tasks={t['task_id']:t for t in data['pools']['validation']}
    scenes={s['task_id']:s for s in c.read(c.previous.pilot.REG/'validation_scenes.json')}
    code_sha=c.sha(__file__);base=c.RESULTS/'observation_cycle_evidence'/code_sha[:12]
    states_all=[];cycles_all=[]
    for method in c.METHODS:
        for seed in c.SEEDS:
            for condition in ('dynamic','static'):
                source=loop_root/method/f'seed_{seed}'/condition
                if not (source/'completion.json').exists():continue
                destination=base/method/f'seed_{seed}'/condition
                identity=dict(diagnostic_completion_sha256=c.sha(source/'completion.json'),
                    annotations_sha256=c.sha(source/'timeout_and_reference_steps.csv'),source_sha256=code_sha)
                if (destination/'completion.json').exists():
                    if c.read(destination/'completion.json')['identity']!=identity:raise ValueError('Observation evidence identity differs')
                    state_rows=csv_rows(destination/'repeated_states.csv') if (destination/'repeated_states.csv').exists() else []
                    cycle_rows=csv_rows(destination/'closed_cycles.csv') if (destination/'closed_cycles.csv').exists() else []
                else:
                    details={r['task_id']:r for r in csv_rows(source/'per_task.csv')}
                    annotations=defaultdict(dict)
                    for row in csv_rows(source/'timeout_and_reference_steps.csv'):annotations[row['task_id']][int(row['tick'])]=row
                    state_rows=[];cycle_rows=[]
                    for trace in c.read(source/'trajectories.json'):
                        task_id=trace['task_id']
                        if not int(details[task_id]['timeout']):continue
                        task=tasks[task_id];scene=scenes[task_id]
                        if condition=='static':scene=dict(scene,obstacles=[])
                        env=make_env(task,scene,planner,cfg);state=env.reset();groups=defaultdict(list);edges=defaultdict(set)
                        for tick,action in enumerate(trace['actions']):
                            if env.position!=tuple(trace['positions'][tick]):raise ValueError('Offline observation reconstruction differs')
                            key=observation_id(state);row=annotations[task_id][tick]
                            outcome=env.step(action);following=observation_id(outcome.observation)
                            groups[key].append(row);edges[key].add(following);state=outcome.observation
                        for key,part in groups.items():
                            if len(part)<3:continue
                            first=part[0];qkeys=[f'q_{a}' for a in range(5)]
                            same_q=all(all(abs(float(r[k])-float(first[k]))<=1e-6 for k in qkeys) for r in part)
                            row=dict(method=method,seed=seed,condition=condition,task_id=task_id,observation_sha256=key,
                                position=first['position'],visits=len(part),first_tick=int(part[0]['tick']),last_tick=int(part[-1]['tick']),
                                executed_actions=','.join(sorted({r['executed_action'] for r in part})),q_consistent=same_q,
                                teacher_action=first['teacher_action'],teacher_rank=first['teacher_q_legal_rank'],
                                teacher_minus_retreat_q=first['teacher_minus_best_retreat_q'],
                                hard_screen_pass=first['hard_screen_pass'],risk_R=first['risk_R'],
                                actual_evaluation_supervision_weight=0.,hypothetical_C_weight=first['hypothetical_state_C_weight'],
                                hypothetical_D_weight=first['hypothetical_state_D_weight'],**{k:first[k] for k in qkeys})
                            state_rows.append(row)
                        for index,nodes in enumerate(functional_cycles(edges)):
                            minimum_visits=min(len(groups[n]) for n in nodes)
                            if minimum_visits<3:continue
                            first_rows=[groups[n][0] for n in nodes]
                            cycle_rows.append(dict(method=method,seed=seed,condition=condition,task_id=task_id,
                                cycle_id=index,cycle_length=len(nodes),minimum_visits_per_state=minimum_visits,
                                positions=' -> '.join(r['position'] for r in first_rows),
                                executed_actions=' -> '.join(r['executed_action'] for r in first_rows),
                                teacher_actions=' -> '.join(r['teacher_action'] for r in first_rows),
                                teacher_rank_greater_than1_states=sum(int(r['teacher_q_legal_rank'])>1 for r in first_rows),
                                teacher_below_retreat_states=sum(r['teacher_minus_best_retreat_q']!='' and float(r['teacher_minus_best_retreat_q']) < -1e-6 for r in first_rows),
                                observation_states=' -> '.join(nodes),inference='observed closed recurrent-observation graph, not proof of training cause'))
                    c.csv_write(destination/'repeated_states.csv',state_rows);c.csv_write(destination/'closed_cycles.csv',cycle_rows)
                    c.write(destination/'completion.json',dict(identity=identity,complete=True,new_policy_decisions=0,updates=0,
                        repeated_states=len(state_rows),closed_cycles=len(cycle_rows)))
                states_all.extend(state_rows);cycles_all.extend(cycle_rows)
    c.csv_write(c.RESULTS/'repeated_observation_states.csv',states_all)
    c.csv_write(c.RESULTS/'recurrent_observation_cycles.csv',cycles_all)
    return dict(repeated_states=len(states_all),closed_cycles=len(cycles_all),new_policy_decisions=0)
