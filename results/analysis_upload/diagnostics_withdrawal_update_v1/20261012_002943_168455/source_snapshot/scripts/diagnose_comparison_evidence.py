"""Read-only checkpoint diagnosis and paired paper evidence. Never trains."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch

import ddqna_adapter_common as common
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc, mean_records
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.core.grid import move, valid_execution_actions
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.utils.io import write_json, write_records_csv


def q_batch(agent,states):
    with torch.no_grad():
        return agent.policy_network(torch.as_tensor(np.stack([s.spatial for s in states]),device=agent.device),
            torch.as_tensor(np.stack([s.scalars for s in states]),device=agent.device)).cpu().numpy()


def profile(values,prefix):
    data=np.asarray(values,dtype=float).reshape(-1)
    return {prefix+'_'+key:float(value) for key,value in dict(min=np.min(data),max=np.max(data),
        mean=np.mean(data),p95=np.percentile(data,95),abs_p95=np.percentile(np.abs(data),95)).items()}


def rollout(config,problem,agent,scenes,mode,static=False,trials=20):
    # Diagnostic teacher mode is explicitly separate from published epsilon=0
    # evaluation: epsilon=0 but p=.5 enabled by this wrapper, never by agent.
    count=trials if static and mode=='assisted' else 1 if static else len(scenes)
    environments=[]
    if static:
        environments=[PathGuidanceEnvironment(DynamicGridNavigationEnv(problem,(),max_steps=300,
            window_size=15,terminate_on_collision=True,reward_config=common.legacy._reward_config(config)),
            enabled=False) for _ in range(count)]
    else:
        factory=common.tail.FrozenFactory(problem,scenes,trace=False)
        environments=[factory(problem,max_steps=300,window_size=15,terminate_on_collision=True,
            reward_config=common.legacy._reward_config(config)) for _ in scenes]
    states=[e.reset() for e in environments]
    rngs=[random.Random(91703700+i) for i in range(count)]
    traces=[[] for _ in environments];rows=[];samples=[]
    counts=Counter();all_q=[];residuals=[];gaps=[];seen=set()
    before=common.legacy.runtime.state_digest(D3QNAgent.training_state_dict(agent))
    active=list(range(count))
    while active:
        current=[states[i] for i in active];q=q_batch(agent,current)
        masks=np.stack([environments[i].action_mask(True) for i in active])
        greedy=np.where(masks,q,-np.inf).argmax(axis=1)
        decisions=[];outcomes=[]
        for offset,i in enumerate(active):
            env,state=environments[i],states[i]
            decoded=tuple(problem.goal[j]-round(float(state.scalars[j])*(problem.size-1)) for j in (0,1))
            if decoded!=env.position or state.spatial.shape!=(5,15,15) or state.scalars.shape!=(4,) or (
                    np.any(state.spatial[4]) or np.any(state.scalars[2:])):
                raise ValueError('Coordinate/observation contract mismatch.')
            if tuple(np.flatnonzero(masks[offset]))!=valid_execution_actions(env.position,problem.size,problem.obstacles):
                raise ValueError('Static mask mismatch.')
            label=agent.static_advice(state)
            valid=label is not None and masks[offset,label]
            if valid and static_distances_cached[move(env.position,label)]!=static_distances_cached[env.position]-1:
                raise ValueError('A* action mapping or direction is wrong.')
            counts['decisions']+=1;counts['label_valid']+=int(valid)
            counts['agreement']+=int(valid and greedy[offset]==label)
            if valid:gaps.append(float(q[offset,greedy[offset]]-q[offset,label]))
            epsilon=.05 if mode=='training_policy' else 0.
            random_branch=epsilon>0 and rngs[i].random()<epsilon
            requested=False if mode=='autonomous' or random_branch else (
                True if mode=='teacher_only' else rngs[i].random()<.5)
            action=int(greedy[offset])
            if random_branch:action=rngs[i].choice(list(np.flatnonzero(masks[offset])))
            elif requested and valid:action=int(label)
            counts['random_branch']+=int(random_branch);counts['teacher_requested']+=int(requested)
            counts['teacher_applied']+=int(requested and valid)
            counts['requested_valid']+=int(requested and valid)
            counts['teacher_changed_action']+=int(requested and valid and action!=greedy[offset])
            counts['executed_matches_label']+=int(valid and action==label)
            all_q.extend(q[offset,masks[offset]].tolist())
            # Compare original epsilon=0 API with batched decision once per cell.
            key=(i,env.position)
            if mode=='autonomous' and key not in seen:
                expected=D3QNAgent.select_action(agent,state,0,list(np.flatnonzero(masks[offset])))
                counts['original_api_checks']+=1
                counts['original_api_mismatches']+=int(expected!=greedy[offset]);seen.add(key)
            position=env.position
            outcome=env.step(action);states[i]=outcome.observation
            trace=dict(action=action,before=dict(position=position),position=env.position,
                dynamic_encounter=outcome.info['dynamic_encounter'],reward=outcome.reward)
            traces[i].append(trace)
            decisions.append(dict(scenario_id=str(scenes[i]['scenario_id']) if not static else f'static_{i}',
                step=env.steps,position=position,network_action=int(greedy[offset]),teacher_action=label,
                executed_action=action,teacher_requested=requested,
                q_values=q[offset].tolist(),teacher_q_gap=float(q[offset,greedy[offset]]-q[offset,label]) if valid else None))
            outcomes.append(outcome)
        nextq=q_batch(agent,[o.observation for o in outcomes])
        nextmasks=np.stack([environments[i].action_mask(True) for i in active])
        nextvalues=np.max(np.where(nextmasks,nextq,-np.inf),axis=1)
        still=[]
        for offset,(i,outcome) in enumerate(zip(active,outcomes)):
            d=decisions[offset]
            target=outcome.reward+agent.config.gamma*float(nextvalues[offset])*(not outcome.terminated)
            residual=target-float(q[offset,d['executed_action']]);residuals.append(residual)
            d['frozen_self_bootstrap_residual']=residual;d['terminated']=outcome.terminated;d['truncated']=outcome.truncated
            # All decisions are saved: no cherry-picking the largest errors.
            samples.append(d)
            if outcome.done:
                trace=dict(scenario_id=d['scenario_id'],termination_reason=outcome.info['termination_reason'],steps=traces[i])
                b=trace_behavior(trace,static_distances_cached)
                rows.append(dict(scenario_id=d['scenario_id'],safe_success=int(outcome.info['reached']),
                    dynamic_collision=int(outcome.info['collision_type']=='dynamic'),
                    **{k:v for k,v in b.items() if k not in ('tail_60_steps','scenario_id')}))
            else:still.append(i)
        active=still
    after=common.legacy.runtime.state_digest(D3QNAgent.training_state_dict(agent))
    if before!=after:raise ValueError('Diagnostic changed policy/target/optimizer/learner RNG.')
    summary=dict(scene_count=count,safe_success_rate=np.mean([r['safe_success'] for r in rows]),
        dynamic_collision_rate=np.mean([r['dynamic_collision'] for r in rows]),
        timeout_rate=np.mean([r['termination_reason']=='timeout' for r in rows]),
        **failure_counts(rows),**dict(counts),
        teacher_request_rate=counts['teacher_requested']/counts['decisions'],
        suggestion_valid_rate=counts['label_valid']/counts['decisions'],
        requested_suggestion_valid_rate=counts['requested_valid']/counts['teacher_requested'] if counts['teacher_requested'] else None,
        teacher_execution_rate=counts['teacher_applied']/counts['decisions'],
        network_teacher_agreement=counts['agreement']/counts['label_valid'] if counts['label_valid'] else None,
        executed_teacher_agreement=counts['executed_matches_label']/counts['label_valid'] if counts['label_valid'] else None,
        **profile(all_q,'valid_q'),**profile(residuals,'frozen_residual'),
        teacher_q_gap_mean=float(np.mean(gaps)) if gaps else None,
        policy_state_unchanged=True)
    return summary,rows,samples


static_distances_cached={}


def diagnosis(output,seeds,anchors,device):
    global static_distances_cached
    config=common.legacy.load_config(common.CONFIG);inputs=common.validate_config(config)
    common.register(config);problem=inputs[0];static_distances_cached=static_distances(problem)
    scenes=common.read_json(common.ROOT/config['dataset']['validation_reference'])['scenarios']
    summaries=[];costs=[];model_hashes={}
    for seed in seeds:
        directory,run=common.audit_run(config,common.METHOD,seed)
        branches=load_csv(directory/'action_branches.csv')
        for left,right in ((0,80000),(80000,100000),(100000,200000),(0,200000)):
            chosen=[r for r in branches if int(r['bin_start_step'])>left and int(r['bin_end_step'])<=right]
            n=sum(int(r['steps']) for r in chosen);requested=sum(int(r['teacher_branch']) for r in chosen)
            costs.append(dict(seed=seed,left=left,right=right,teacher_request_rate=requested/n,
                suggestion_valid_rate=sum(int(r['teacher_applied']) for r in chosen)/requested if requested else None,
                execution_rate=sum(int(r['teacher_applied']) for r in chosen)/n,
                expected_request_rate=sum(float(r['teacher_mass_sum']) for r in chosen)/n,
                agreement_in_original_training_log=None,teacher_exit_during_training=False))
        curve=load_csv(directory/'validation_curve.csv')
        selected=list(dict.fromkeys(next(int(r['environment_steps_total']) for r in curve
            if int(r['environment_steps_total'])>=anchor) for anchor in anchors))
        for step in selected:
            model=directory/f'model_step_{step:06d}.pth';sha=common.sha256(model)
            agent=common.make_agent(config,problem,seed,device);agent.load_weights(model)
            stored=torch.load(model,map_location=agent.device,weights_only=True)
            if common.legacy.runtime.state_digest(stored)!=common.legacy.runtime.state_digest(agent.policy_network.state_dict()):
                raise ValueError('Model loader changed stored parameters.')
            model_hashes[str(model.relative_to(common.ROOT))]=sha
            for static in (False,True):
                modes=['autonomous','assisted']
                if step==200000:modes+=['teacher_only'] if static else ['training_policy','teacher_only']
                for mode in modes:
                    s,rows,decisions=rollout(config,problem,agent,scenes,mode,static)
                    if not static and mode=='autonomous':
                        expected=next(r for r in curve if int(r['environment_steps_total'])==step)
                        s['matches_saved_success_collision_timeout']=all(np.isclose(s[k],float(expected[k]))
                            for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate'))
                    s.update(method=common.METHOD,seed=seed,checkpoint=step,mode=mode,static=static)
                    summaries.append(s)
                    prefix=f'seed{seed}_step{step:06d}_{"static" if static else "dynamic"}_{mode}'
                    write_records_csv(rows,output/'diagnosis'/f'{prefix}_episodes.csv')
                    write_records_csv(decisions,output/'diagnosis'/f'{prefix}_q_decisions.csv')
                    write_records_csv(summaries,output/'diagnosis'/'summary.csv')
                    print(f'[{prefix}] success={s["safe_success_rate"]:.1%} collision={s["dynamic_collision_rate"]:.1%} '
                          f'timeout={s["timeout_rate"]:.1%} Qmax={s["valid_q_max"]:.2f}',flush=True)
            if common.sha256(model)!=sha:raise ValueError('Checkpoint changed during diagnosis.')
        # Aggregate actual training logs by phase: these were collected with
        # epsilon, teacher and changing weights, not matched evaluation scenes.
        training=load_csv(directory/'training.csv')
        for left,right in ((0,80000),(80000,100000),(100000,150000),(150000,200000)):
            chosen=[r for r in training if left<int(r['environment_steps_total'])<=right]
            costs.append(dict(seed=seed,left=left,right=right,training_episode_count=len(chosen),
                logged_teacher_policy_success_rate=np.mean([float(r['safe_success']) for r in chosen]),
                logged_mean_huber_loss=np.mean([float(r['loss']) for r in chosen if r['loss']]),
                logged_max_huber_loss=max(float(r['loss']) for r in chosen if r['loss']),
                not_same_as_fixed_scene_autonomous_comparison=True))
    # Plain original combination final policies provide the same-state-scale control.
    for seed in seeds:
        directory,_=common.audit_run(config,common.CONTROL,seed)
        agent=common.make_agent(config,problem,seed,device);model=directory/'model_final.pth';agent.load_weights(model)
        model_hashes[str(model.relative_to(common.ROOT))]=common.sha256(model)
        for static in (False,True):
            s,rows,decisions=rollout(config,problem,agent,scenes,'autonomous',static)
            s.update(method=common.CONTROL,seed=seed,checkpoint=200000,mode='autonomous',static=static)
            summaries.append(s);prefix=f'control_seed{seed}_{"static" if static else "dynamic"}'
            write_records_csv(rows,output/'diagnosis'/f'{prefix}_episodes.csv')
            write_records_csv(decisions,output/'diagnosis'/f'{prefix}_q_decisions.csv')
    write_records_csv(summaries,output/'diagnosis'/'summary.csv')
    write_records_csv(costs,output/'diagnosis'/'original_training_audit.csv')
    write_json(dict(complete=True,training_started=False,gradient_updates=0,
        final500_used_for_diagnosis=False,validation_scenes=50,static_assisted_trials=20,
        teacher_diagnostic_probability=.5,teacher_only_probability=1,training_policy_diagnostic_epsilon=.05,
        frozen_residual_definition='r + gamma * max_static_legal Q_loaded(s_next) * (1-terminated) - Q_loaded(s,executed)',
        saved_target_network_and_replay_missing=True,residual_is_not_original_training_td_error=True,
        source_models_sha256=model_hashes,seeds=seeds,anchors=anchors,source_sha256=common.sha256(Path(__file__))),
        output/'diagnosis'/'verification.json')


def first_crossings(x,y):
    hits=np.flatnonzero(y>=.9);triples=[i for i in range(len(y)-2) if np.all(y[i:i+3]>=.9)]
    first=int(x[hits[0]]) if len(hits) else None
    start=int(x[triples[0]]) if triples else None
    confirm=int(x[triples[0]+2]) if triples else None
    return dict(first_90_step=first,three_90_start_step=start,three_90_confirmation_step=confirm,
        first_90_reached=int(first is not None),three_90_reached=int(start is not None))


def evidence(output):
    config=common.legacy.load_config(common.CONFIG);hc=common.legacy.load_config(common.historical.CONFIG)
    methods=(*common.historical.ALL_METHODS,common.METHOD)
    summaries=[];status=[];conditions=[];curves=[]
    for method in methods:
        for seed in common.SEEDS:
            directory,run=(common.audit_run(config,method,seed) if method==common.METHOD else
                common.historical.audit_run(hc,method,seed))
            curve=load_csv(directory/'validation_curve.csv');details=load_csv(directory/'validation_details.csv')
            x=np.array([int(r['environment_steps_total']) for r in curve]);y=np.array([float(r['safe_success_rate']) for r in curve])
            if len(x)!=21 or x[0]!=0 or x[-1]!=200000 or len(details)!=1050:raise ValueError('Missing curve points/scenes.')
            for row in curve:
                d=[r for r in details if r['environment_steps_total']==row['environment_steps_total']]
                if len(d)!=50 or not np.isclose(float(row['safe_success_rate']),np.mean([float(r['safe_success']) for r in d])):
                    raise ValueError('Validation curve/scene disagreement.')
                curves.append(dict(method=method,seed=seed,**row))
            has_advice=method.startswith('advice') or method=='supervision_only_bound'
            withdrawal=140000 if method==common.tail.METHOD else 100000 if has_advice else None
            post=y[x>=withdrawal] if withdrawal is not None else None
            summary=dict(method=method,seed=seed,**first_crossings(x,y),
                early_aulc_0_80000=aulc(x,y,0,80000),full_aulc=aulc(x,y),late_aulc_100000_200000=aulc(x,y,100000,200000),
                late_aulc_140000_200000=aulc(x,y,140000,200000),teaching_fully_off_step=withdrawal,
                post_teaching_min_success=float(np.min(post)) if post is not None else None,
                comparison_100k_120500_min=float(np.min(y[(x>=100000)&(x<=120500)])),
                comparison_140k_160500_min=float(np.min(y[(x>=140000)&(x<=160500)])),
                final_success=y[-1],final_collision=float(curve[-1]['dynamic_collision_rate']),
                final_timeout=float(curve[-1]['timeout_rate']),last_five_mean=float(np.mean(y[-5:])),
                pretraining_before_zero_checkpoint=method.startswith('dqfd'))
            summaries.append(summary)
            status.append(dict(method=method,seed=seed,complete=True,environment_steps=run['environment_steps'],
                gradient_updates=run['gradient_updates'],checkpoints=21,initialization_and_sources_audited=True,
                final_model_sha256=common.sha256(directory/'model_final.pth')))
            if method in ('advice_bound_margin','dqfd','dqfd_bound'):
                cond=dict(method=method,seed=seed,offline_demo_transitions=run['demo_transition_count'],
                    pretraining_updates=run.get('pretraining_updates',0),online_environment_steps=run['environment_steps'],
                    online_updates=run.get('online_updates',run['gradient_updates']),total_updates=run['gradient_updates'],
                    online_replay_capacity=run['online_replay_capacity'],permanent_demo_count=run.get('permanent_demo_retained_count',0),
                    online_astar_lookups=run.get('online_astar_lookups'),online_astar_searches=run.get('online_astar_searches'),
                    online_training_seconds=run.get('online_training_seconds'),pretraining_seconds=run.get('pretraining_seconds'),
                    evaluation_advice=run['evaluation_advice'],checkpoint_selection='final_200000_only',
                    online_teacher_exit=100000 if method=='advice_bound_margin' else None,
                    permanent_demo_exit=None if method.startswith('dqfd') else 100000,
                    initial_learner_pairing_audited=True)
                conditions.append(cond)
    metrics=[k for k in summaries[0] if k not in ('method','seed','pretraining_before_zero_checkpoint')]
    means=mean_records(summaries,metrics)
    for row in means:
        subset=[s for s in summaries if s['method']==row['method']]
        row.update({k+'_observed_seed_count':sum(s[k] is not None for s in subset) for k in metrics})
    pairs=[];lookup={(r['method'],r['seed']):r for r in summaries}
    for a,b,label in (('advice_bound_margin','advice_bound','added_margin'),
            ('advice_bound_margin','supervision_only_bound','added_action_override'),
            ('supervision_only_bound','unguided_bound','added_supervision'),
            ('unguided_bound','unguided_raw','td_bound_without_teacher'),
            ('advice_bound','advice_raw','td_bound_with_override'),
            ('advice_bound_margin','advice_margin','td_bound_with_override_and_margin'),
            ('advice_bound_margin','advice_bound_margin_no_risk','risk_filter'),
            (common.tail.METHOD,'advice_bound_margin','tail_extension'),
            ('dqfd','advice_bound_margin','dqfd_different_total_budget'),
            ('dqfd_bound','dqfd','dqfd_td_bound')):
        for seed in common.SEEDS:
            pairs.append(dict(method=label,treated=a,control=b,seed=seed,
                **{k:lookup[a,seed][k]-lookup[b,seed][k] if lookup[a,seed][k] is not None and lookup[b,seed][k] is not None else None
                   for k in ('early_aulc_0_80000','full_aulc','late_aulc_140000_200000','final_success')}))
    base=output/'evidence'
    for name,records in (('summary',summaries),('method_means',means),('completion_status',status),
            ('dqfd_conditions',conditions),('paired_differences',pairs),('all_curves',curves),
            ('paired_means',mean_records(pairs,['early_aulc_0_80000','full_aulc','late_aulc_140000_200000','final_success']))):
        write_records_csv(records,base/(name+'.csv'))
    dm=common.historical.frozen(hc)
    write_json({k:dm[k] for k in ('demo_transition_count','attempted_episodes','retained_episodes','demo_astar_lookups',
        'demo_astar_searches','demonstration_generation_seconds','demo_policy_uses_future_obstacles')},base/'dqfd_demonstration_cost.json')
    write_json(dict(complete=True,formal_runs=60,methods=12,seeds=5,checkpoints=len(curves),training_started=False,
        final500_used_for_method_selection=False,first_vs_three_point_streak_separate=True,
        missing_threshold_steps_are_censored=True),base/'verification.json')


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnose',action='store_true');parser.add_argument('--evidence',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,2])
    parser.add_argument('--anchors',type=int,nargs='+',default=[20000,80000,90000,100000,110000,200000])
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if not args.diagnose and not args.evidence:parser.error('Choose --diagnose and/or --evidence; these never train.')
    if not set(args.seeds)<=set(common.SEEDS):parser.error('Seeds must be 0..4.')
    output=common.ROOT/'results/comparison_evidence_diagnosis'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False);print(output,flush=True)
    if args.evidence:evidence(output)
    if args.diagnose:diagnosis(output,args.seeds,args.anchors,args.device)
    write_json(dict(complete=True,training_started=False,gradient_updates=0,existing_results_preserved=True),output/'completion.json')
    print(output)


if __name__=='__main__':main()
