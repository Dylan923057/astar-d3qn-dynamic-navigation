"""Final500 confirmation: evaluate only the ten new final models, reuse audited old methods."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import latest_module_common as protocol
import supervision_tail_common as tail
from evaluate_replication_final500 import OLD,METRICS,ProgressFactory,summarize
from analyze_runtime_path_pilot import load_csv,static_distances,trace_behavior
from analyze_teaching_efficiency import mean_records
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json,write_records_csv

ROOT,common,legacy=protocol.ROOT,protocol.common,protocol.legacy
NAME='latest_all_eligible_modules_final500_v1'


def prepare():
    torch.set_num_threads(1)
    cfg=legacy.load_config(tail.CONFIG);scenes,_,freeze=tail.load_frozen(cfg)
    sha=freeze['files_sha256']['independent_final_scenarios.json'];ids=[s['scenario_id'] for s in scenes]
    latest=ROOT/common.read_json(ROOT/'results/execution_consistency_final500_v1/latest.json')['evaluation']
    caches={OLD:common.read_json(OLD/'verification.json'),latest:common.read_json(latest/'verification.json')}
    tables={path:(load_csv(path/'scene_details.csv'),load_csv(path/'summary.csv')) for path in caches}
    expected_config=legacy.load_config(legacy.CONFIG);planned=[];models={};failed=[]
    for method in protocol.ALL:
        for seed in range(5):
            p=protocol.run(method,seed)
            if p['failed']:
                failed.append(dict(p['failure'],method=method,seed=seed));continue
            if any(p['config'][k]!=expected_config[k] for k in ('dataset','environment','reward','guidance','dynamic_route_pool','agent')):
                raise ValueError('Common evaluation conditions differ.')
            a=torch.load(p['model'],map_location='cpu',weights_only=True)
            b=torch.load(p['directory']/'model_step_200000.pth',map_location='cpu',weights_only=True)
            if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('Not the final200000 checkpoint.')
            models[protocol.relative(p['model'])]=p['digest'];p['cached']=None
            if method not in protocol.core.METHODS:
                path=latest if method==protocol.FULL else OLD;meta=caches[path]
                hashes={key.replace('\\','/'):v for key,v in meta['source_model_sha256'].items()}
                epsilon=meta.get('epsilon',meta.get('evaluation_epsilon'))
                selection=meta.get('checkpoint',meta.get('checkpoint_selection'))
                if not meta['complete'] or meta['scene_count']!=500 or meta['frozen_scene_sha256']!=sha or epsilon!=0 or selection!='final_200000_step_only' or hashes.get(protocol.relative(p['model']))!=p['digest']:
                    raise ValueError('Cached scenes/model/autonomous protocol differ.')
                if meta.get('teacher',False) or meta.get('route_guidance',False) or meta.get('dynamic_protection',False) or meta.get('action_override',False) or meta.get('future_obstacles_used_for_actions',False):
                    raise ValueError('Cached evaluation used intervention/future information.')
                rows=[r for r in tables[path][0] if r['method']==method and int(r['seed'])==seed]
                if len(rows)!=500 or [r['scenario_id'] for r in rows]!=ids:raise ValueError('Cached500 IDs/order differ.')
                if method==protocol.FULL:
                    source=ROOT/next(r['trace_source'] for r in load_csv(path/'result_sources.csv') if r['method']==method and int(r['seed'])==seed)
                else:source=ROOT/meta['full_failure_trace_root']/method/f'seed_{seed}'/'failure_trajectories.json'
                traces=common.read_json(source)
                if len(traces)!=sum(not float(r['safe_success']) for r in rows) or {t['scenario_id'] for t in traces}!={r['scenario_id'] for r in rows if not float(r['safe_success'])}:
                    raise ValueError('Cached failures differ.')
                behaviors=[trace_behavior(t,static_distances(p['inputs'][0])) for t in traces]
                computed=summarize(method,seed,rows,behaviors)
                stored=next(r for r in tables[path][1] if r['method']==method and int(r['seed'])==seed)
                if any(not np.isclose(computed[k],float(stored[k])) for k in METRICS):raise ValueError('Cached summary differs.')
                p['cached']=dict(rows=rows,behaviors=behaviors,summary=computed,trace_source=protocol.relative(source),source_evaluation=protocol.relative(path))
            planned.append(p)
    sources={protocol.relative(Path(__file__)):common.sha256(Path(__file__)),
        'scripts/latest_module_common.py':common.sha256(ROOT/'scripts/latest_module_common.py'),
        'scripts/evaluate_replication_final500.py':common.sha256(ROOT/'scripts/evaluate_replication_final500.py'),
        'src/astar_d3qn/evaluation/rollout.py':common.sha256(ROOT/'src/astar_d3qn/evaluation/rollout.py')}
    value=dict(frozen_scene_sha256=sha,scene_file=protocol.relative(tail.artifact_root(cfg)/'independent_final_scenarios.json'),
        source_model_sha256=models,evaluation_source_sha256=sources,failed_training_runs=failed,
        epsilon=0,teacher=False,route_guidance=False,dynamic_protection=False,static_mask=True,
        max_steps=300,window_size=15,terminate_on_collision=True,checkpoint='final_200000_step_only',
        training_started=False,used_for_tuning=False,methods=list(protocol.ALL),seeds=list(range(5)),
        old_cache_verification_sha256={protocol.relative(p/'verification.json'):common.sha256(p/'verification.json') for p in caches})
    target=ROOT/'data'/protocol.core.NAME/'final500_registration.json'
    if target.exists() and common.read_json(target)!=value:raise ValueError('Final500 registration changed.')
    if not target.exists():write_json(value,target)
    return planned,scenes,value


def evaluate(device='cpu'):
    planned,scenes,registration=prepare();stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'results'/NAME/stamp;raw=ROOT/'outputs'/NAME/stamp
    output.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False)
    write_json(dict(registration,device=device),output/'run_manifest.json')
    summaries=[];details=[];behaviors_all=[];origins=[];clock=perf_counter()
    for p in planned:
        method,seed=p['method'],p['seed']
        if p['cached'] is not None:
            c=p['cached'];rows=c['rows'];behaviors=c['behaviors'];summary=c['summary'];source=c['trace_source'];reused=True
        else:
            agent=legacy.runtime.make_agent(p['config'],seed,device);agent.load_weights(p['model'])
            factory=ProgressFactory(p['inputs'][0],scenes,trace=True,label=f'{method} seed{seed}')
            with legacy.runtime.preserved_evaluation(agent):
                _,rows,_=evaluate_agent(agent,[p['inputs'][0]]*500,max_steps=300,window_size=15,
                    reward_config=legacy._reward_config(p['config']),terminate_on_collision=True,
                    environment_factory=factory,mask_static_invalid_actions=True)
            if len(rows)!=500 or [r['scenario_id'] for r in rows]!=[s['scenario_id'] for s in scenes]:raise ValueError('Scene IDs/order differ.')
            traces=[];behaviors=[];distances=static_distances(p['inputs'][0])
            for row,env in zip(rows,factory.environments):
                if not row['safe_success']:
                    trace=dict(scenario_id=row['scenario_id'],safe_success=False,termination_reason=row['termination_reason'],steps=env.trace)
                    traces.append(trace);behaviors.append(trace_behavior(trace,distances))
            path=raw/method/f'seed_{seed}'/'failure_trajectories.json';write_json(traces,path)
            source=protocol.relative(path);reused=False;summary=summarize(method,seed,rows,behaviors)
        if common.sha256(p['model'])!=p['digest']:raise ValueError('Evaluation modified a model.')
        summaries.append(summary);details.extend(dict(r,method=method,seed=seed) for r in rows)
        behaviors_all.extend(dict(method=method,seed=seed,**{k:v for k,v in b.items() if k!='tail_60_steps'}) for b in behaviors)
        origins.append(dict(method=method,seed=seed,reused=reused,trace_source=source,model_sha256=p['digest'],frozen_scene_sha256=registration['frozen_scene_sha256']))
        for name,records in (('summary',summaries),('scene_details',details),('failure_behaviors',behaviors_all),('result_sources',origins)):
            write_records_csv(records,output/f'{name}.csv')
        write_json(dict(complete=False,completed_models=len(summaries),planned_models=len(planned)),output/'status.json')
        print(f'[{method} seed{seed}] {"REUSED" if reused else "COMPLETE"} final500 success={summary["safe_success_rate"]:.1%}',flush=True)
    # Failed training seeds remain explicit NA rows, never silently removed from means.
    for f in registration['failed_training_runs']:
        summaries.append(dict(method=f['method'],seed=f['seed'],environment_steps=f['environment_steps'],scene_count=0,numerical_failure=True,
            **{key:None for key in METRICS}))
    means=protocol.finite_five_means(summaries,list(METRICS))
    write_records_csv(summaries,output/'summary.csv');write_records_csv(means,output/'method_means.csv')
    audit=dict(registration,complete=True,models=len(planned),planned_models=35,scene_count=500,
        scene_rows=len(details),newly_evaluated_models=sum(p['cached'] is None for p in planned),
        reused_models=sum(p['cached'] is not None for p in planned),statistical_unit='seed',evaluation_seconds=perf_counter()-clock)
    write_json(audit,output/'verification.json');write_json(audit,raw/'completion.json')
    write_json(dict(complete=True,completed_models=len(planned),planned_models=35,failed_training_runs=registration['failed_training_runs']),output/'status.json')
    write_json(dict(evaluation=protocol.relative(output)),ROOT/'results'/NAME/'latest.json')
    print(output,flush=True);return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu');args=parser.parse_args()
    if args.evaluate:evaluate(args.device)
    else:
        planned,scenes,registration=prepare()
        print(dict(passed=True,new_final_models=sum(p['cached'] is None for p in planned),reusable=sum(p['cached'] is not None for p in planned),frozen_scenes=len(scenes),evaluation_started=False))
