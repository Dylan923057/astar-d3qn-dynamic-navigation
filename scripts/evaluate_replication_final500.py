"""Final-only independent evaluation; reuse matching old results, never train/tune."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import analyze_replication_seeds01234 as analysis
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import mean_records
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT,common,legacy=analysis.ROOT,analysis.common,analysis.legacy
OLD=ROOT/'results/whole_map_91701_dqfd_comparison_v1/independent_eval_20261009_174147_641239'
METHODS=('advice_bound_margin','staged_withdrawal','dqfd','dqfd_no_pretrain')
METRICS=('safe_success_rate','dynamic_collision_rate','timeout_rate','collision_count',
    'timeout_count','waiting_timeout_count','looping_timeout_count','other_timeout_count')


def relative(path):return Path(path).relative_to(ROOT).as_posix()


def summarize(method,seed,rows,behaviors):
    return dict(method=method,seed=seed,environment_steps=200000,scene_count=500,
        safe_success_rate=float(np.mean([float(r['safe_success']) for r in rows])),
        dynamic_collision_rate=float(np.mean([float(r['dynamic_collision']) for r in rows])),
        timeout_rate=sum(r['termination_reason']=='timeout' for r in rows)/500,
        **failure_counts(behaviors))


def prepare():
    frozen_cfg=legacy.load_config(analysis.entry.dq.tail.CONFIG)
    scenes,_,frozen=analysis.entry.dq.tail.load_frozen(frozen_cfg)
    scene_hash=frozen['files_sha256']['independent_final_scenarios.json']
    original=legacy.load_config(analysis.entry.dq.CONFIG)
    original_registration=analysis.entry.dq.frozen(original)
    if scene_hash!=original_registration['final_scene_sha256']:raise ValueError('Use the original pre-training frozen500, never generate replacement scenes.')
    ids=[s['scenario_id'] for s in scenes]
    cache=common.read_json(OLD/'verification.json')
    protocol_matches=(cache['complete'] and cache['scene_count']==500 and cache['evaluation_epsilon']==0
        and not cache['action_override'] and not cache['future_obstacles_used_for_actions']
        and cache['checkpoint_selection']=='final_200000_step_only' and cache['frozen_scene_sha256']==scene_hash
        and common.read_json(OLD/'effective_config.json')==original)
    old_hashes={p.replace('\\','/'):v for p,v in cache['source_model_sha256'].items()}
    old_details=load_csv(OLD/'scene_details.csv');old_summaries=load_csv(OLD/'summary.csv')
    planned=[]
    for arm in ('staged','dqfd'):
        for seed in range(5):
            config,inputs,baseline,new,new_method=analysis.directories(arm,seed)
            if any(config[k]!=original[k] for k in ('dataset','environment','reward','guidance','dynamic_route_pool','agent')):
                raise ValueError('Shared evaluation/map/reward/observation settings changed.')
            for method,directory in (('advice_bound_margin' if arm=='staged' else 'dqfd',baseline),(new_method,new)):
                result=common.read_json(directory/'result.json');done=common.read_json(directory/'completion.json')
                if not done['complete'] or done['environment_steps']!=200000 or result['environment_steps']!=200000:
                    raise ValueError(f'Incomplete final model: {directory}')
                model=directory/'model_final.pth'
                if not model.is_file():raise FileNotFoundError(f'Missing final weight; no retraining: {model}')
                digest=common.sha256(model)
                if result.get('final_model_sha256') is not None and result['final_model_sha256']!=digest:
                    raise ValueError('Recorded final weight changed.')
                a=torch.load(model,map_location='cpu',weights_only=True)
                b=torch.load(directory/'model_step_200000.pth',map_location='cpu',weights_only=True)
                if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('Final weight is not the200000-step checkpoint.')
                cached=None
                if directory==baseline and protocol_matches and old_hashes.get(relative(model))==digest:
                    rows=[r for r in old_details if r['method']==method and int(r['seed'])==seed]
                    source=ROOT/cache['full_failure_trace_root']/method/f'seed_{seed}'/'failure_trajectories.json'
                    if source.is_file() and len(rows)==500 and [r['scenario_id'] for r in rows]==ids:
                        traces=common.read_json(source)
                        if {r['scenario_id'] for r in rows if not float(r['safe_success'])}!={t['scenario_id'] for t in traces}:
                            raise ValueError('Cached failure traces disagree with episode rows.')
                        behaviors=[trace_behavior(t,static_distances(inputs[0])) for t in traces]
                        computed=summarize(method,seed,rows,behaviors)
                        recorded=next(r for r in old_summaries if r['method']==method and int(r['seed'])==seed)
                        if any(not np.isclose(computed[k],float(recorded[k])) for k in METRICS):
                            raise ValueError('Cached summary disagrees with full500 rows/traces.')
                        cached=dict(rows=rows,behaviors=behaviors,summary=computed,trace_source=relative(source))
                planned.append(dict(method=method,seed=seed,config=config,inputs=inputs,directory=directory,
                    model=model,digest=digest,cached=cached))
    return planned,scenes,frozen_cfg,scene_hash


class ProgressFactory(analysis.entry.dq.tail.FrozenFactory):
    def __init__(self,*args,label,**kwargs):
        super().__init__(*args,**kwargs);self.label=label
    def __call__(self,*args,**kwargs):
        env=super().__call__(*args,**kwargs)
        if self.index%100==0:print(f'[{self.label}] starting frozen scene {self.index}/500',flush=True)
        return env


def evaluate(device='cpu'):
    torch.set_num_threads(1)
    planned,scenes,frozen_cfg,scene_hash=prepare()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'results/replication_final500_v1'/stamp
    raw=ROOT/'outputs/replication_final500_v1'/stamp
    output.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False)
    manifest=dict(epsilon=0,teacher=False,route_guidance=False,dynamic_protection=False,static_mask=True,
        max_steps=300,window_size=15,terminate_on_collision=True,checkpoint='final_200000_step_only',
        frozen_scene_sha256=scene_hash,source_model_sha256={relative(p['model']):p['digest'] for p in planned},
        evaluation_source_sha256={relative(Path(__file__)):common.sha256(Path(__file__)),
            'src/astar_d3qn/evaluation/rollout.py':common.sha256(ROOT/'src/astar_d3qn/evaluation/rollout.py')},
        training_started=False,used_for_tuning=False,device=device)
    write_json(manifest,output/'run_manifest.json')
    summaries=[];details=[];failures=[];origins=[]
    clock=perf_counter()
    for p in planned:
        method,seed=p['method'],p['seed'];label=f'{method} seed{seed}'
        if p['cached'] is not None:
            c=p['cached'];summary=c['summary'];rows=c['rows'];behaviors=c['behaviors']
            origin=dict(method=method,seed=seed,reused=True,source_evaluation=relative(OLD),
                trace_source=c['trace_source'],model_sha256=p['digest'],frozen_scene_sha256=scene_hash)
            print(f'[{label}] REUSED same SHA/protocol; safe={summary["safe_success_rate"]:.1%}',flush=True)
        else:
            agent=legacy.runtime.make_agent(p['config'],seed,device)
            agent.load_weights(p['model'])  # Plain D3QN: no teacher, dynamic protection or Q-output clipping.
            factory=ProgressFactory(p['inputs'][0],scenes,trace=True,label=label)
            with legacy.runtime.preserved_evaluation(agent):
                _,rows,_=evaluate_agent(agent,[p['inputs'][0]]*500,max_steps=300,window_size=15,
                    reward_config=legacy._reward_config(p['config']),terminate_on_collision=True,
                    environment_factory=factory,mask_static_invalid_actions=True)
            if [r['scenario_id'] for r in rows]!=[s['scenario_id'] for s in scenes]:raise ValueError('Frozen scene IDs/order changed.')
            traces=[];behaviors=[];distances=static_distances(p['inputs'][0])
            for row,env in zip(rows,factory.environments):
                if not row['safe_success']:
                    trace=dict(scenario_id=row['scenario_id'],safe_success=False,
                        termination_reason=row['termination_reason'],steps=env.trace)
                    traces.append(trace);behaviors.append(trace_behavior(trace,distances))
            source=raw/method/f'seed_{seed}'/'failure_trajectories.json'
            write_json(traces,source)
            summary=summarize(method,seed,rows,behaviors)
            origin=dict(method=method,seed=seed,reused=False,source_evaluation=relative(output),
                trace_source=relative(source),model_sha256=p['digest'],frozen_scene_sha256=scene_hash)
            print(f'[{label}] COMPLETE final500 safe={summary["safe_success_rate"]:.1%}, collision={summary["dynamic_collision_rate"]:.1%}, timeout={summary["timeout_rate"]:.1%}',flush=True)
        if common.sha256(p['model'])!=p['digest']:raise ValueError('Evaluation modified a source model.')
        summaries.append(summary)
        details.extend(dict(r,method=method,seed=seed) for r in rows)
        failures.extend(dict(method=method,seed=seed,**{k:v for k,v in b.items() if k!='tail_60_steps'}) for b in behaviors)
        origins.append(origin)
        write_records_csv(summaries,output/'summary.csv')
        write_records_csv(details,output/'scene_details.csv')
        write_records_csv(failures,output/'failure_behaviors.csv')
        write_records_csv(origins,output/'result_sources.csv')
        write_json(dict(complete=False,completed_models=len(summaries),planned_models=20),output/'status.json')
    means=mean_records(summaries,list(METRICS));pairs=[]
    for baseline,new in (('advice_bound_margin','staged_withdrawal'),('dqfd','dqfd_no_pretrain')):
        for seed in range(5):
            a=next(r for r in summaries if r['method']==baseline and r['seed']==seed)
            b=next(r for r in summaries if r['method']==new and r['seed']==seed)
            pairs.append(dict(method=new+'_minus_'+baseline,seed=seed,**{k:b[k]-a[k] for k in METRICS}))
    write_records_csv(means,output/'method_means.csv');write_records_csv(pairs,output/'paired_differences.csv')
    write_records_csv(mean_records(pairs,list(METRICS)),output/'paired_difference_means.csv')
    audit=dict(**manifest,complete=True,models=20,scene_count=500,scene_rows=len(details),
        reused_models=sum(r['reused'] for r in origins),newly_evaluated_models=sum(not r['reused'] for r in origins),
        statistical_unit='five paired training seeds',evaluation_seconds=perf_counter()-clock,
        original_evaluation_verification_sha256=common.sha256(OLD/'verification.json'),
        source_table_sha256={relative(OLD/name):common.sha256(OLD/name) for name in ('summary.csv','scene_details.csv','failure_behaviors.csv')})
    write_json(audit,output/'verification.json');write_json(dict(complete=True,completed_models=20),output/'status.json')
    write_json(audit,raw/'completion.json')
    lines=['# 同一冻结500场景：四方法五seed最终确认','',
        '200000步最终模型，ε=0普通D3QN动作，无教师、路线引导、动态保护或Q输出裁剪，仅共同静态合法动作掩码。原结果仅在场景SHA、模型SHA、协议及500条明细/失败轨迹复核一致时复用。统计单位为seed，样本标准差ddof=1。','',
        '|方法|安全成功%|动态碰撞%|超时%|', '|---|---:|---:|---:|']
    for r in means:
        lines.append('|'+r['method']+'|'+ '|'.join(f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_sample_sd"]:.2f}' for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate'))+'|')
    lines.extend(['','|方法|seed|成功%|碰撞次数|等待超时|绕圈超时|其他超时|', '|---|---:|---:|---:|---:|---:|---:|'])
    for r in summaries:
        lines.append(f'|{r["method"]}|{r["seed"]}|{100*r["safe_success_rate"]:.1f}|{r["collision_count"]}|{r["waiting_timeout_count"]}|{r["looping_timeout_count"]}|{r["other_timeout_count"]}|')
    lines.extend(['',f'复用{audit["reused_models"]}个原最终模型结果，补评{audit["newly_evaluated_models"]}个模型；合并10000条场景记录。','',
        '本次评估只做最终确认，不用于修改教学日程、奖励、超参数或选择检查点。学习效率与撤教低谷仍以原50场景曲线判断；同地图结果不代表地图泛化或保证安全。',
        '配对差：[逐seed](paired_differences.csv)、[均值与样本标准差](paired_difference_means.csv)。复用来源与完整失败轨迹位置见result_sources.csv，原始轨迹留在outputs。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:9]));print(output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu');args=parser.parse_args()
    if args.evaluate:evaluate(args.device)
    else:
        torch.set_num_threads(1)
        planned,_,_,scene_hash=prepare()
        print(dict(passed=True,models=20,reusable=sum(p['cached'] is not None for p in planned),
            pending=sum(p['cached'] is None for p in planned),frozen_scene_sha256=scene_hash,
            evaluation_started=False,training_started=False))


if __name__=='__main__':main()
