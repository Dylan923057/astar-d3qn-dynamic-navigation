"""Finalize a complete20-model evaluation ledger without rerunning any episode."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluate_replication_final500 import ROOT,common,OLD,METHODS,METRICS,relative
from analyze_runtime_path_pilot import load_csv
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import mean_records
from astar_d3qn.utils.io import write_json,write_records_csv


def finalize(output):
    output=Path(output).resolve();output.relative_to(ROOT.resolve())
    manifest=common.read_json(output/'run_manifest.json')
    models=manifest.get('source_model_sha256',manifest.get('models'))
    if not isinstance(models,dict) or len(models)!=20:raise ValueError('Need all20 frozen final model SHA values.')
    status=common.read_json(output/'status.json')
    if status['completed_models']!=20:raise ValueError('Evaluations incomplete; do not finalize partial results.')
    summaries=load_csv(output/'summary.csv');details=load_csv(output/'scene_details.csv')
    failures=load_csv(output/'failure_behaviors.csv');origins=load_csv(output/'result_sources.csv')
    if len(summaries)!=20 or len(details)!=10000 or len(origins)!=20:raise ValueError('Incomplete evaluation rows.')
    import analyze_replication_seeds01234 as analysis
    tail=analysis.entry.dq.tail
    cfg=analysis.legacy.load_config(tail.CONFIG)
    scenes,_,freeze=tail.load_frozen(cfg)
    ids=[s['scenario_id'] for s in scenes]
    if manifest['frozen_scene_sha256']!=freeze['files_sha256']['independent_final_scenarios.json']:
        raise ValueError('Frozen scenes changed.')
    for name,sha in models.items():
        if common.sha256(ROOT/name)!=sha:raise ValueError('Final model changed.')
    for r in summaries:
        r.update(seed=int(r['seed']),environment_steps=int(r['environment_steps']),scene_count=int(r['scene_count']))
        for key in METRICS:r[key]=int(r[key]) if key.endswith('_count') else float(r[key])
        rows=[x for x in details if x['method']==r['method'] and int(x['seed'])==r['seed']]
        if [x['scenario_id'] for x in rows]!=ids:raise ValueError('Scene identity/order differs.')
        fail=[x for x in failures if x['method']==r['method'] and int(x['seed'])==r['seed']]
        typed=[dict(x,waiting_dominated=x['waiting_dominated'].lower()=='true',
            repeated_movement=x['repeated_movement'].lower()=='true',
            tail_period=int(x['tail_period']) if x['tail_period'] else None) for x in fail]
        counts=failure_counts(typed)
        expected=dict(safe_success_rate=np.mean([float(x['safe_success']) for x in rows]),
            dynamic_collision_rate=np.mean([float(x['dynamic_collision']) for x in rows]),
            timeout_rate=sum(x['termination_reason']=='timeout' for x in rows)/500,**counts)
        if any(not np.isclose(r[k],v) for k,v in expected.items()):raise ValueError('Summary/rows/behavior counts disagree.')
        if r['method'] not in METHODS or r['seed'] not in range(5):raise ValueError('Unexpected model group.')
    if len({(r['method'],r['seed']) for r in summaries})!=20:raise ValueError('Duplicate or missing model.')
    means=mean_records(summaries,list(METRICS));pairs=[]
    for baseline,new in (('advice_bound_margin','staged_withdrawal'),('dqfd','dqfd_no_pretrain')):
        for seed in range(5):
            a=next(r for r in summaries if r['method']==baseline and r['seed']==seed)
            b=next(r for r in summaries if r['method']==new and r['seed']==seed)
            pairs.append(dict(method=new+'_minus_'+baseline,seed=seed,**{k:b[k]-a[k] for k in METRICS}))
    write_records_csv(means,output/'method_means.csv');write_records_csv(pairs,output/'paired_differences.csv')
    write_records_csv(mean_records(pairs,list(METRICS)),output/'paired_difference_means.csv')
    reused=sum(r['reused'].lower()=='true' for r in origins)
    normalized={k:v for k,v in manifest.items() if k!='models'}
    normalized['source_model_sha256']=models
    audit=dict(**normalized,complete=True,models=20,scene_count=500,scene_rows=10000,
        reused_models=reused,newly_evaluated_models=20-reused,statistical_unit='five paired training seeds',
        evaluation_seconds=None,report_only_recovery=True,
        recovery_reason='All evaluations completed; an audit field name collision affected report writing only. Finalized saved rows without rerunning episodes.',
        finalizer_source_sha256=common.sha256(Path(__file__)),
        original_evaluation_verification_sha256=common.sha256(OLD/'verification.json'),
        source_table_sha256={relative(OLD/name):common.sha256(OLD/name) for name in ('summary.csv','scene_details.csv','failure_behaviors.csv')})
    snapshot=output/'evaluation_source_used.py'
    if snapshot.is_file():
        original_sha=next(v for k,v in manifest['evaluation_source_sha256'].items() if k.endswith('evaluate_replication_final500.py'))
        if common.sha256(snapshot)!=original_sha:raise ValueError('Preserved executed evaluator source hash differs.')
        audit['executed_evaluator_snapshot']=relative(snapshot)
    write_json(audit,output/'verification.json');write_json(dict(complete=True,completed_models=20),output/'status.json')
    raw=ROOT/'outputs/replication_final500_v1'/output.name
    write_json(audit,raw/'completion.json')
    lines=['# 同一冻结500场景：四方法五seed最终确认','',
        '200000步最终模型，ε=0普通D3QN动作，关闭教师、路线引导、动态保护和Q输出裁剪，只保留共有静态合法动作掩码。统计单位seed，样本标准差ddof=1。','',
        '|方法|安全成功%|动态碰撞%|超时%|', '|---|---:|---:|---:|']
    for r in means:
        lines.append('|'+r['method']+'|'+ '|'.join(f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_sample_sd"]:.2f}' for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate'))+'|')
    lines.extend(['','|方法|seed|成功%|碰撞|等待超时|绕圈超时|其他超时|', '|---|---:|---:|---:|---:|---:|---:|'])
    for r in summaries:
        lines.append(f'|{r["method"]}|{r["seed"]}|{100*r["safe_success_rate"]:.1f}|{r["collision_count"]}|{r["waiting_timeout_count"]}|{r["looping_timeout_count"]}|{r["other_timeout_count"]}|')
    lines.extend(['',f'复用{reused}个旧最终模型结果，补评{20-reused}个模型；汇总10000条场景记录。旧结果仅在场景SHA、模型SHA、协议和逐场景/失败轨迹一致时复用。',
        '评估未用于修改日程、超参数、地图、奖励或选择检查点。学习效率与退出低谷仍以原50验证曲线判断；同地图结果不代表地图泛化。',
        '所有回合已正常保存；汇总写入时发生同名审计字段错误，仅修复报告写入，没有重跑或删掉任何seed。实际执行源码的SHA和快照已保留，评估总耗时未捕获，明确记为缺失。',
        '完整轨迹留在result_sources.csv所列outputs路径。逐seed、均值和配对差见本目录CSV，所有表格已与10000条明细及失败分类复核。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:10]));print(output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();finalize(args.output)


if __name__=='__main__':main()
