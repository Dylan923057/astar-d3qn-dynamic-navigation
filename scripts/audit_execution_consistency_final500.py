"""Verify saved final500 rows and all failures without rerunning evaluation or training."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import evaluate_execution_consistency_final500 as evaluation
from analyze_runtime_path_pilot import load_csv,static_distances,trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import mean_records
from astar_d3qn.utils.io import write_json

ROOT,common=evaluation.ROOT,evaluation.common


def audit(output=None):
    if output is None:output=ROOT/common.read_json(ROOT/'results'/evaluation.NAME/'latest.json')['evaluation']
    output=Path(output).resolve();output.relative_to(ROOT.resolve())
    planned,scenes,registration=evaluation.prepare()
    verification=common.read_json(output/'verification.json')
    manifest=common.read_json(output/'run_manifest.json');status=common.read_json(output/'status.json')
    if not verification['complete'] or verification['models']!=10 or verification['scene_rows']!=5000 or not status['complete'] or status['completed_models']!=10:
        raise ValueError('Final500 evaluation is incomplete.')
    for key,value in registration.items():
        if manifest[key]!=value or verification[key]!=value:raise ValueError(f'Frozen protocol differs: {key}')
    if verification['registration_sha256']!=common.sha256(evaluation.REGISTRATION):raise ValueError('Evaluation registration changed.')
    summary=load_csv(output/'summary.csv');details=load_csv(output/'scene_details.csv')
    origins=load_csv(output/'result_sources.csv');logged=load_csv(output/'failure_behaviors.csv')
    expected={(p['method'],p['seed']) for p in planned}
    if len(summary)!=10 or len(origins)!=10 or len(details)!=5000 or {(r['method'],int(r['seed'])) for r in summary}!=expected or {(r['method'],int(r['seed'])) for r in origins}!=expected:
        raise ValueError('Missing or duplicate results.')
    ids=[s['scenario_id'] for s in scenes];computed=[];traces_sha={}
    for p in planned:
        method,seed=p['method'],p['seed']
        rows=[r for r in details if r['method']==method and int(r['seed'])==seed]
        if len(rows)!=500 or [r['scenario_id'] for r in rows]!=ids:raise ValueError('Scene identity/order differs.')
        for row,scene in zip(rows,scenes):
            if int(row['dynamic_obstacle_count'])!=len(scene['obstacles']):raise ValueError('Obstacle count differs.')
            if int(row['steps'])>300:raise ValueError('Episode budget differs.')
        origin=next(r for r in origins if r['method']==method and int(r['seed'])==seed)
        if origin['model_sha256']!=p['digest'] or origin['frozen_scene_sha256']!=registration['frozen_scene_sha256'] or (origin['reused'].lower()=='true')!=(p['cached'] is not None):
            raise ValueError('Model provenance/reuse differs.')
        path=ROOT/origin['trace_source'];traces=common.read_json(path)
        failed={r['scenario_id']:r for r in rows if not float(r['safe_success'])}
        if len(traces)!=len(failed) or {t['scenario_id'] for t in traces}!=set(failed):raise ValueError('Missing or duplicate failure traces.')
        behaviors=[trace_behavior(t,static_distances(p['inputs'][0])) for t in traces]
        logs={r['scenario_id']:r for r in logged if r['method']==method and int(r['seed'])==seed}
        if set(logs)!=set(failed):raise ValueError('Failure behavior records differ.')
        for trace,behavior in zip(traces,behaviors):
            row=failed[trace['scenario_id']];record=logs[trace['scenario_id']]
            if trace['termination_reason']!=row['termination_reason'] or len(trace['steps'])!=int(row['steps']):raise ValueError('Failure rollout and result differ.')
            for key,value in behavior.items():
                if key=='tail_60_steps':continue
                if isinstance(value,bool):equal=record[key].lower()==str(value).lower()
                elif value is None:equal=not record[key]
                elif isinstance(value,(int,float)):equal=np.isclose(float(record[key]),value)
                elif isinstance(value,str):equal=record[key]==value
                else:continue
                if not equal:raise ValueError(f'Failure behavior differs: {method}/seed{seed}/{key}')
        result=evaluation.summarize(method,seed,rows,behaviors)
        stored=next(r for r in summary if r['method']==method and int(r['seed'])==seed)
        if any(not np.isclose(result[k],float(stored[k])) for k in evaluation.METRICS):raise ValueError('Summary disagrees with rows/traces.')
        computed.append(result);traces_sha[evaluation.relative(path)]=common.sha256(path)
    def compare(filename,records):
        saved=load_csv(output/filename)
        if len(saved)!=len(records):raise ValueError(f'Missing statistics: {filename}')
        for record in records:
            row=next(r for r in saved if r['method']==record['method'] and ('seed' not in record or int(r['seed'])==record['seed']))
            for key,value in record.items():
                if key=='method':continue
                if not np.isclose(float(row[key]),value):raise ValueError(f'Statistics differ: {filename}/{key}')
    compare('method_means.csv',mean_records(computed,list(evaluation.METRICS)))
    pairs=[]
    for seed in range(5):
        old=next(r for r in computed if r['method']==evaluation.BASELINE and r['seed']==seed)
        new=next(r for r in computed if r['method']==evaluation.METHOD and r['seed']==seed)
        pairs.append(dict(method=evaluation.METHOD+'_minus_'+evaluation.BASELINE,seed=seed,
            **{key:new[key]-old[key] for key in evaluation.METRICS}))
    compare('paired_differences.csv',pairs)
    compare('paired_difference_means.csv',mean_records(pairs,list(evaluation.METRICS)))
    record=dict(passed=True,models=10,scene_rows=5000,newly_evaluated_models=5,reused_models=5,
        all_failures_checked=len(logged),statistical_unit='seed',sample_sd_ddof=1,
        evaluation_repeated=False,training_started=False,
        audit_source_sha256=common.sha256(Path(__file__)),
        evaluation_verification_sha256=common.sha256(output/'verification.json'),
        source_model_sha256=registration['source_model_sha256'],failure_trace_sha256=traces_sha,
        table_sha256={name:common.sha256(output/name) for name in ('summary.csv','scene_details.csv','failure_behaviors.csv','result_sources.csv','method_means.csv','paired_differences.csv','paired_difference_means.csv')})
    write_json(record,output/'audit_saved_results.json')
    print(dict(passed=True,models=10,scene_rows=5000,all_failures_checked=len(logged),evaluation_repeated=False),flush=True)
    return output,record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation',type=Path)
    args=parser.parse_args();audit(args.evaluation)
