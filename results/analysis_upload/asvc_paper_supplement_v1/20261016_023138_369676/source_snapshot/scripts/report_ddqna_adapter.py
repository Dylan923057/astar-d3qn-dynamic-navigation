"""Analyze old-50 learning curves or evaluate the two methods' FINAL models."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

import ddqna_adapter_common as common
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc, mean_records
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json, write_records_csv


METHODS=(common.CONTROL,common.METHOD)


def learning_metrics(x,y,spec):
    triples=[i for i in range(len(y)-2) if np.all(y[i:i+3]>=spec['stable_success_threshold'])]
    first=triples[0] if triples else None
    return dict(safe_success_aulc=aulc(x,y),aulc_100000_200000=aulc(x,y,100000,200000),
        **{f'aulc_{a}_{b}':aulc(x,y,a,b) for a,b in zip(spec['phase_boundaries'],spec['phase_boundaries'][1:])},
        first_three_90_start_step=int(x[first]) if first is not None else None,
        first_three_90_confirmation_step=int(x[first+2]) if first is not None else None,
        three_90_reached=int(first is not None), last_five_checkpoint_mean=float(np.mean(y[-5:])))


def collect(config,method,seed,distances):
    directory,run=common.audit_run(config,method,seed)
    curve=load_csv(directory/'validation_curve.csv')
    x=np.array([int(r['environment_steps_total']) for r in curve])
    y=np.array([float(r['safe_success_rate']) for r in curve])
    if len(x)!=21 or x[0]!=0 or x[-1]!=200000 or np.any(np.diff(x)<=0):
        raise ValueError('All original 21 timed validations are required.')
    scenes=common.read_json(common.ROOT/config['dataset']['validation_reference'])['scenarios']
    grouped=defaultdict(list)
    for row in load_csv(directory/'validation_details.csv'):
        grouped[int(row['environment_steps_total'])].append(row)
    if set(grouped)!=set(x):
        raise ValueError('Missing validation scene details.')
    behaviors,checkpoints=[],[]
    for row in curve:
        step=int(row['environment_steps_total']); rows=grouped[step]
        if ([r['scenario_id'] for r in rows]!=[s['scenario_id'] for s in scenes] or
                any(r['dynamic_route_ids'].split(';')!=s['route_ids'] for r,s in zip(rows,scenes))):
            raise ValueError('Old validation identities/order/routes changed.')
        expected=dict(safe_success_rate=np.mean([float(r['safe_success']) for r in rows]),
            dynamic_collision_rate=np.mean([float(r['dynamic_collision']) for r in rows]),
            timeout_rate=np.mean([r['termination_reason']=='timeout' for r in rows]))
        if any(not np.isclose(float(row[k]),v) for k,v in expected.items()):
            raise ValueError('Curve and scene details disagree.')
        traces=common.read_json(directory/f'validation_failures_{step:06d}.json')
        failed={r['scenario_id'] for r in rows if not float(r['safe_success'])}
        if len(traces)!=len(failed) or {t['scenario_id'] for t in traces}!=failed:
            raise ValueError('Raw failure trajectories missing/duplicated.')
        inspected=[trace_behavior(t,distances) for t in traces]
        behaviors.extend(dict(method=method,seed=seed,environment_steps=step,
            **{k:v for k,v in b.items() if k!='tail_60_steps'}) for b in inspected)
        checkpoints.append(dict(method=method,seed=seed,**row,**failure_counts(inspected)))
    summary=dict(method=method,seed=seed,**learning_metrics(x,y,config['ddqna_adapter']),
        safe_success_rate=y[-1],dynamic_collision_rate=float(curve[-1]['dynamic_collision_rate']),
        timeout_rate=float(curve[-1]['timeout_rate']))
    for prefix,left,right in (('all',0,200000),('final',200000,200000)):
        summary.update({prefix+'_'+k:v for k,v in failure_counts(
            [b for b in behaviors if left<=b['environment_steps']<=right]).items()})
    if method==common.METHOD:
        branches=load_csv(directory/'action_branches.csv')
        if len(branches)!=20 or sum(int(r['steps']) for r in branches)!=200000:
            raise ValueError('Incomplete action branch log.')
        for index,r in enumerate(branches):
            if (int(r['bin_start_step'])!=index*10000+1 or int(r['bin_end_step'])!=(index+1)*10000 or
                    float(r['teacher_probability'])!=.5 or int(r['steps'])!=10000 or
                    sum(int(r[k]) for k in ('random_branch','teacher_branch','greedy_branch'))!=int(r['steps']) or
                    int(r['teacher_applied'])+int(r['teacher_unavailable'])!=int(r['teacher_branch']) or
                    not np.isclose(float(r['teacher_mass_sum']),float(r['greedy_mass_sum'])) or
                    not np.isclose(sum(float(r[k]) for k in ('random_mass_sum','teacher_mass_sum','greedy_mass_sum')),10000)):
                raise ValueError('Action-selection probability/schedule audit failed.')
        if sum(int(r['teacher_branch']) for r in branches)!=run['online_astar_lookups']:
            raise ValueError('A* lookup cost disagrees with action branches.')
    costs=dict(method=method,seed=seed,gradient_updates=run['gradient_updates'],
        **{k:run.get(k) for k in ('online_training_seconds','online_validation_and_callback_seconds',
            'method_wall_seconds','online_astar_lookups','online_astar_searches','online_astar_seconds')})
    return summary,checkpoints,behaviors,costs


def aggregate(rows,output,prefix=''):
    metrics=[k for k in rows[0] if k not in ('method','seed')]
    means=mean_records(rows,metrics)
    for r in means:
        subset=[s for s in rows if s['method']==r['method']]
        r.update({k+'_observed_seed_count':sum(s.get(k) is not None for s in subset) for k in metrics})
    keyed={(r['method'],r['seed']):r for r in rows}
    paired=[]
    for seed in common.SEEDS:
        a,b=keyed[common.CONTROL,seed],keyed[common.METHOD,seed]
        paired.append(dict(method=common.CONTROL+'_minus_'+common.METHOD,seed=seed,
            **{k:a[k]-b[k] if a.get(k) is not None and b.get(k) is not None else None for k in metrics}))
    paired_means=mean_records(paired,metrics)
    for r in paired_means:
        r.update({k+'_observed_seed_count':sum(p.get(k) is not None for p in paired) for k in metrics})
    write_records_csv(rows,output/(prefix+'summary.csv'))
    write_records_csv(means,output/(prefix+'method_means.csv'))
    write_records_csv(paired,output/(prefix+'paired_differences.csv'))
    write_records_csv(paired_means,output/(prefix+'paired_difference_means.csv'))
    return means


def report_text(means,output,title):
    display=[k for k in ('safe_success_aulc','aulc_100000_200000','safe_success_rate',
                        'dynamic_collision_rate','timeout_rate') if k+'_mean' in means[0]]
    lines=[title,'','五个配对训练seed为统计单位，均值±样本标准差（ddof=1）。', '',
        '| 方法 | '+' | '.join(display)+' |','|---|'+'---:|'*len(display)]
    for r in means:
        lines.append('| '+r['method']+' | '+' | '.join(
            f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_sample_sd"]:.2f}%' for k in display)+' |')
    lines.extend(['', '逐seed、配对差值、分阶段AULC和失败分类见CSV。首次连续三个≥90%验证点：主指标为第一点步数，另列第三点确认步数。',
        '未达到者留空并标记three_90_reached=0；步数均值只描述达到者，observed_seed_count必须同时报告，不将未达到者记成200000。',
        '等待超时、绕圈超时、其他超时沿用既有轨迹分类（互斥）；全程失败次数含重复验证场景，不能当作独立场景数量。',
        '旧方法未记录的A*调用/耗时留空；不按文件时间或理论次数补造。',
        '该基线只适配DDQNA动作选择；全程固定教师p=0.5。与完整组合的差值是多项机制差异的总效果，单模块归因需结合旧消融。',''])
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def analyze(config):
    inputs=common.validate_config(config)
    planned=[(m,s,*common.audit_run(config,m,s)) for m in METHODS for s in common.SEEDS]
    distances=static_distances(inputs[0])
    summaries,curves,behaviors,costs=[],[],[],[]
    for method,seed,_,_ in planned:
        summary,c,b,cost=collect(config,method,seed,distances)
        summaries.append(summary);curves.extend(c);behaviors.extend(b);costs.append(cost)
    output=common.legacy.runtime.unique_check_dir(config,'analysis')
    report_text(aggregate(summaries,output),output,'# DDQNA动作机制适配版与完整组合：原50验证场景')
    aggregate(costs,output,'training_cost_')
    write_records_csv(curves,output/'all_validation_checkpoints.csv')
    write_records_csv(behaviors,output/'failure_behaviors.csv')
    write_json(dict(complete=True,models=10,checkpoint_count=len(curves),inspected_failures=len(behaviors),
        training_started=False,independent_500_performance_evaluated=False,statistical_unit='paired seed'),output/'verification.json')
    print(output)


def evaluate(config,device='auto'):
    inputs=common.validate_config(config)
    scenes,_,frozen=common.tail.load_frozen(common.legacy.load_config(common.tail.CONFIG))
    # Audit all ten final models before creating evaluation outputs.
    planned=[(m,s,*common.audit_run(config,m,s)) for m in METHODS for s in common.SEEDS]
    weights={str((d/'model_final.pth').relative_to(common.ROOT)):common.sha256(d/'model_final.pth') for _,_,d,_ in planned}
    output=common.legacy.runtime.unique_check_dir(config,'independent_eval')
    raw=common.ROOT/config['experiment']['output_root']/output.name
    raw.mkdir(parents=True,exist_ok=False)
    distances=static_distances(inputs[0]);summaries,details,failures=[],[],[]
    for method,seed,directory,_ in planned:
        # Plain D3QN has no teacher/protection modules, for both algorithms.
        agent=common.legacy.runtime.make_agent(config,seed,device)
        model=directory/'model_final.pth';agent.load_weights(model)
        factory=common.tail.FrozenFactory(inputs[0],scenes,trace=True)
        with common.legacy.runtime.preserved_evaluation(agent):
            summary,rows,_=evaluate_agent(agent,[inputs[0]]*500,max_steps=300,window_size=15,
                reward_config=common.legacy._reward_config(config),terminate_on_collision=True,
                environment_factory=factory,mask_static_invalid_actions=True)
        if [r['scenario_id'] for r in rows]!=[s['scenario_id'] for s in scenes]:
            raise ValueError('Final scene identity/order changed.')
        traces,inspected=[],[]
        for row,env in zip(rows,factory.environments):
            details.append(dict(method=method,seed=seed,**row))
            if not row['safe_success']:
                trace=dict(scenario_id=row['scenario_id'],safe_success=False,
                    termination_reason=row['termination_reason'],steps=env.trace)
                traces.append(trace);b=trace_behavior(trace,distances);inspected.append(b)
                failures.append(dict(method=method,seed=seed,**{k:v for k,v in b.items() if k!='tail_60_steps'}))
        write_json(traces,raw/method/f'seed_{seed}'/'failure_trajectories.json')
        if common.sha256(model)!=weights[str(model.relative_to(common.ROOT))]:
            raise ValueError('Evaluation changed model.')
        summaries.append(dict(method=method,seed=seed,safe_success_rate=summary['safe_success_rate'],
            dynamic_collision_rate=summary['dynamic_collision_rate'],
            timeout_rate=sum(r['termination_reason']=='timeout' for r in rows)/500,**failure_counts(inspected)))
        write_records_csv(summaries,output/'summary.csv')
        print(f'[{method} seed={seed}] final500 autonomous_safe={summary["safe_success_rate"]:.1%}',flush=True)
    report_text(aggregate(summaries,output),output,'# 同一冻结500场景：只评估200000步最终模型')
    write_records_csv(details,output/'scene_details.csv');write_records_csv(failures,output/'failure_behaviors.csv')
    audit=dict(complete=True,models=10,scene_rows=len(details),scene_count=500,
        evaluation_epsilon=0,action_override=False,dynamic_protection=False,static_legal_mask=True,
        future_obstacles_used_for_actions=False,training_started=False,
        checkpoint_selection='final_200000_only',test_used_for_tuning=False,
        frozen_scene_sha256=frozen['files_sha256']['independent_final_scenarios.json'],
        source_model_sha256=weights,full_failure_trace_root=str(raw.relative_to(common.ROOT)),
        statistical_unit='five paired training seeds')
    write_json(audit,output/'verification.json');write_json(audit,raw/'completion.json')
    print(output)


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--analyze',action='store_true');actions.add_argument('--evaluate',action='store_true')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=parser.parse_args();config=common.legacy.load_config(common.CONFIG)
    if args.analyze:analyze(config)
    else:evaluate(config,args.device)


if __name__=='__main__':
    main()
