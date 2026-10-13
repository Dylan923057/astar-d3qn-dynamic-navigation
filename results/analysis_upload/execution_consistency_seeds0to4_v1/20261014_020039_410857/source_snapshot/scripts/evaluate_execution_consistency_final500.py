"""Final confirmation on the existing frozen500; five new models, no training or tuning."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import analyze_execution_consistency_five_seeds as analysis
import supervision_tail_common as tail
from evaluate_replication_final500 import OLD,METRICS,ProgressFactory,summarize
from analyze_runtime_path_pilot import load_csv,static_distances,trace_behavior
from analyze_teaching_efficiency import mean_records
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json,write_records_csv

ROOT,common,legacy=analysis.ROOT,analysis.common,analysis.entry.legacy
NAME='execution_consistency_final500_v1'
REGISTRATION=ROOT/'data'/NAME/'freeze_manifest.json'
BASELINE='advice_bound_margin'
METHOD=analysis.entry.METHOD


def relative(path):return Path(path).relative_to(ROOT).as_posix()


def prepare():
    torch.set_num_threads(1)
    frozen_config=legacy.load_config(tail.CONFIG)
    scenes,_,frozen=tail.load_frozen(frozen_config)
    scene_path=tail.artifact_root(frozen_config)/'independent_final_scenarios.json'
    scene_sha=frozen['files_sha256']['independent_final_scenarios.json']
    cache=common.read_json(OLD/'verification.json')
    if not (cache['complete'] and cache['scene_count']==500 and cache['evaluation_epsilon']==0
            and not cache['action_override'] and not cache['future_obstacles_used_for_actions']
            and cache['checkpoint_selection']=='final_200000_step_only' and cache['frozen_scene_sha256']==scene_sha):
        raise ValueError('Original comparison results do not use the same frozen500/autonomous protocol.')
    original_config=common.read_json(OLD/'effective_config.json')
    old_hashes={p.replace('\\','/'):digest for p,digest in cache['source_model_sha256'].items()}
    old_rows=load_csv(OLD/'scene_details.csv');old_summaries=load_csv(OLD/'summary.csv')
    ids=[s['scenario_id'] for s in scenes]
    planned=[];sources={relative(Path(__file__)):common.sha256(Path(__file__))}
    for path in (Path(analysis.__file__),ROOT/'scripts/analyze_execution_consistency.py',
                 ROOT/'scripts/evaluate_replication_final500.py',ROOT/'scripts/supervision_tail_common.py',
                 ROOT/'src/astar_d3qn/evaluation/rollout.py'):
        sources[relative(path)]=common.sha256(path)
    for seed in range(5):
        config,inputs,old,new,_=analysis.directories('staged',seed)
        if any(config[k]!=original_config[k] for k in ('dataset','environment','reward','guidance','dynamic_route_pool','agent')):
            raise ValueError('Shared evaluation/map/reward/observation settings changed.')
        distances=static_distances(inputs[0])
        for method,directory in ((BASELINE,old),(METHOD,new)):
            done=common.read_json(directory/'completion.json');result=common.read_json(directory/'result.json')
            if not done['complete'] or done['environment_steps']!=200000 or result['environment_steps']!=200000:
                raise ValueError(f'Missing final200000 model: {directory}')
            model=directory/'model_final.pth'
            if not model.is_file():raise FileNotFoundError(f'Missing weight; do not retrain: {model}')
            digest=common.sha256(model)
            if result.get('final_model_sha256',digest)!=digest:raise ValueError('Recorded final weight changed.')
            a=torch.load(model,map_location='cpu',weights_only=True)
            b=torch.load(directory/'model_step_200000.pth',map_location='cpu',weights_only=True)
            if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
                raise ValueError('Use only final200000 checkpoint, never best validation checkpoint.')
            probe=legacy.runtime.make_agent(config,seed,'cpu')
            probe.load_weights(model)  # Verify compatibility with plain D3QN; no teacher wrapper.
            cached=None
            if method==BASELINE:
                if old_hashes.get(relative(model))!=digest:raise ValueError('Cached original model SHA differs; no silent replacement.')
                rows=[r for r in old_rows if r['method']==method and int(r['seed'])==seed]
                if len(rows)!=500 or [r['scenario_id'] for r in rows]!=ids:raise ValueError('Cached scene IDs/order differ.')
                trace_path=ROOT/cache['full_failure_trace_root']/method/f'seed_{seed}'/'failure_trajectories.json'
                traces=common.read_json(trace_path)
                if {r['scenario_id'] for r in rows if not float(r['safe_success'])}!={t['scenario_id'] for t in traces}:
                    raise ValueError('Cached failures and scene results disagree.')
                behaviors=[trace_behavior(t,distances) for t in traces]
                summary=summarize(method,seed,rows,behaviors)
                stored=next(r for r in old_summaries if r['method']==method and int(r['seed'])==seed)
                if any(not np.isclose(summary[k],float(stored[k])) for k in METRICS):raise ValueError('Cached summary differs.')
                cached=dict(rows=rows,behaviors=behaviors,summary=summary,trace_source=relative(trace_path))
                sources[relative(trace_path)]=common.sha256(trace_path)
            planned.append(dict(method=method,seed=seed,config=config,inputs=inputs,model=model,digest=digest,cached=cached))
    for path in (OLD/'verification.json',OLD/'effective_config.json',OLD/'summary.csv',OLD/'scene_details.csv'):
        sources[relative(path)]=common.sha256(path)
    registration=dict(scene_file=relative(scene_path),frozen_scene_sha256=scene_sha,
        original_scene_freeze_sha256=common.sha256(tail.artifact_root(frozen_config)/'freeze_manifest.json'),
        source_model_sha256={relative(p['model']):p['digest'] for p in planned},source_sha256=sources,
        new_method=METHOD,seeds=list(range(5)),new_models=5,reused_original_models=5,
        epsilon=0,teacher=False,route_guidance=False,dynamic_protection=False,static_mask=True,
        max_steps=300,window_size=15,terminate_on_collision=True,checkpoint='final_200000_step_only',
        used_for_tuning=False,training_started=False,new_scene_generation=False)
    if REGISTRATION.exists() and common.read_json(REGISTRATION)!=registration:
        raise ValueError('Final evaluation registration changed; preserve the existing freeze.')
    if not REGISTRATION.exists():write_json(registration,REGISTRATION)
    return planned,scenes,registration


def evaluate(device='cpu'):
    planned,scenes,registration=prepare()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'results'/NAME/stamp;raw=ROOT/'outputs'/NAME/stamp
    output.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False)
    manifest=dict(registration,device=device,registration_sha256=common.sha256(REGISTRATION))
    write_json(manifest,output/'run_manifest.json')
    summaries=[];details=[];failures=[];origins=[];clock=perf_counter()
    write_json(dict(complete=False,completed_models=0,planned_models=10),output/'status.json')
    for p in planned:
        method,seed=p['method'],p['seed']
        if p['cached'] is not None:
            c=p['cached'];rows=c['rows'];behaviors=c['behaviors'];summary=c['summary']
            trace_source=c['trace_source'];reused=True
        else:
            agent=legacy.runtime.make_agent(p['config'],seed,device)
            agent.load_weights(p['model'])
            factory=ProgressFactory(p['inputs'][0],scenes,trace=True,label=f'{method} seed{seed}')
            with legacy.runtime.preserved_evaluation(agent):
                _,rows,_=evaluate_agent(agent,[p['inputs'][0]]*500,max_steps=300,window_size=15,
                    reward_config=legacy._reward_config(p['config']),terminate_on_collision=True,
                    environment_factory=factory,mask_static_invalid_actions=True)
            if len(rows)!=500 or [r['scenario_id'] for r in rows]!=[s['scenario_id'] for s in scenes]:
                raise ValueError('Frozen scene IDs/order differ.')
            traces=[];behaviors=[];distances=static_distances(p['inputs'][0])
            for row,env in zip(rows,factory.environments):
                if not row['safe_success']:
                    trace=dict(scenario_id=row['scenario_id'],safe_success=False,
                        termination_reason=row['termination_reason'],steps=env.trace)
                    traces.append(trace);behaviors.append(trace_behavior(trace,distances))
            source=raw/method/f'seed_{seed}'/'failure_trajectories.json'
            write_json(traces,source);trace_source=relative(source);reused=False
            summary=summarize(method,seed,rows,behaviors)
        if common.sha256(p['model'])!=p['digest']:raise ValueError('Evaluation changed a model.')
        summaries.append(summary)
        details.extend(dict(r,method=method,seed=seed) for r in rows)
        failures.extend(dict(method=method,seed=seed,**{k:v for k,v in b.items() if k!='tail_60_steps'}) for b in behaviors)
        origins.append(dict(method=method,seed=seed,reused=reused,trace_source=trace_source,
            model_sha256=p['digest'],frozen_scene_sha256=registration['frozen_scene_sha256']))
        for name,records in (('summary',summaries),('scene_details',details),('failure_behaviors',failures),('result_sources',origins)):
            write_records_csv(records,output/f'{name}.csv')
        write_json(dict(complete=False,completed_models=len(summaries),planned_models=10),output/'status.json')
        print(f'[{method} seed{seed}] {"REUSED" if reused else "COMPLETE"} safe={summary["safe_success_rate"]:.1%}, collision={summary["dynamic_collision_rate"]:.1%}, timeout={summary["timeout_rate"]:.1%}',flush=True)
    if len(summaries)!=10 or len(details)!=5000:raise ValueError('Incomplete final confirmation.')
    means=mean_records(summaries,list(METRICS));pairs=[]
    for seed in range(5):
        a=next(r for r in summaries if r['method']==BASELINE and r['seed']==seed)
        b=next(r for r in summaries if r['method']==METHOD and r['seed']==seed)
        pairs.append(dict(method=METHOD+'_minus_'+BASELINE,seed=seed,**{k:b[k]-a[k] for k in METRICS}))
    write_records_csv(means,output/'method_means.csv');write_records_csv(pairs,output/'paired_differences.csv')
    write_records_csv(mean_records(pairs,list(METRICS)),output/'paired_difference_means.csv')
    audit=dict(manifest,complete=True,models=10,scene_count=500,scene_rows=5000,
        newly_evaluated_models=5,reused_models=5,statistical_unit='seed',evaluation_seconds=perf_counter()-clock)
    write_json(audit,output/'verification.json');write_json(audit,raw/'completion.json')
    lines=['# 执行一致性筛选：同一冻结500场景最终确认','',
        '只评估200000步最终模型，ε=0，关闭教师、路线引导和动态保护，仅共同静态合法动作掩码。新增五模型共2500回合，原组五seed同场景/SHA/协议结果复用。统计单位seed，样本标准差ddof=1，不用于调参或选择检查点。','',
        '|方法|成功%|动态碰撞%|超时%|', '|---|---:|---:|---:|']
    for row in means:
        lines.append('|'+row['method']+'|'+ '|'.join(f'{100*row[k+"_mean"]:.2f}±{100*row[k+"_sample_sd"]:.2f}' for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate'))+'|')
    lines.extend(['','|方法|seed|成功%|碰撞次数|等待超时|绕圈超时|其他超时|', '|---|---:|---:|---:|---:|---:|---:|'])
    for row in summaries:
        lines.append(f'|{row["method"]}|{row["seed"]}|{100*row["safe_success_rate"]:.1f}|{row["collision_count"]}|{row["waiting_timeout_count"]}|{row["looping_timeout_count"]}|{row["other_timeout_count"]}|')
    lines.extend(['','配对差见paired_differences.csv及paired_difference_means.csv；全部5000条场景记录见scene_details.csv。失败完整轨迹路径见result_sources.csv，未来障碍信息仅用于事后诊断。',
        '该确认只研究同地图独立动态场景；学习效率与撤教低谷仍以原50验证曲线判断，不主张地图泛化或保证安全。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(dict(complete=True,completed_models=10,planned_models=10),output/'status.json')
    write_json(dict(evaluation=relative(output)),ROOT/'results'/NAME/'latest.json')
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    args=parser.parse_args()
    if args.evaluate:evaluate(args.device)
    else:
        planned,scenes,registration=prepare()
        print(dict(passed=True,frozen_scenes=len(scenes),new_models=sum(p['cached'] is None for p in planned),
            reused_models=sum(p['cached'] is not None for p in planned),
            frozen_scene_sha256=registration['frozen_scene_sha256'],evaluation_started=False,training_started=False),flush=True)
