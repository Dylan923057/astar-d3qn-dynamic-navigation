"""Standard-library-only verification and recalculation for the map evidence bundle."""
from __future__ import annotations
import argparse
import bisect
from collections import Counter,defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

NAME='asvc_multimap_layout_v1'
MAIN=('d3qn','astar_prefill_td','dqfd','qdagger','ce','asvc')
METRICS=('early_aulc','full_aulc','late_aulc','late_min_success','final_safe_success_rate',
    'final_dynamic_collision_rate','final_timeout_rate')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def local(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Unsafe archive path: '+str(name))
    return path

def archived(root,name):
    path=local(root,name)
    if path.exists():return path
    compressed=Path(str(path)+'.gz')
    if compressed.exists():return compressed
    raise FileNotFoundError(path)

def read(root,name):
    path=archived(root,name)
    with gzip.open(path,'rt',encoding='utf-8-sig') if path.suffix=='.gz' else path.open(encoding='utf-8-sig') as stream:
        return json.load(stream)

def rows(root,name):
    path=archived(root,name)
    with gzip.open(path,'rt',encoding='utf-8-sig',newline='') if path.suffix=='.gz' else path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def write(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_write(values,path):
    values=list(values);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not values:return
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in values for k in row)))
        writer.writeheader();writer.writerows(values)

def flag(value):return value is True or str(value).lower() in ('1','1.0','true')
def close(left,right,label):
    if not math.isclose(float(left),float(right),rel_tol=1e-10,abs_tol=1e-11):raise ValueError(f'{label}: {left} != {right}')

def interpolate(x,y,t):
    if t<x[0] or t>x[-1]:raise ValueError('AULC extrapolation forbidden')
    i=bisect.bisect_left(x,t)
    if i<len(x) and x[i]==t:return y[i]
    return y[i-1]+(y[i]-y[i-1])*(t-x[i-1])/(x[i]-x[i-1])

def auc(curve,start,end):
    x=[int(r['environment_steps_total']) for r in curve];y=[float(r['safe_success_rate']) for r in curve]
    if not x or x[0]>start or x[-1]<end:return None
    if any(b<=a for a,b in zip(x,x[1:])):raise ValueError('Validation steps must strictly increase')
    points=[start]+[t for t in x if start<t<end]+[end]
    values=[interpolate(x,y,t) for t in points]
    return sum((b-a)*(left+right)/2 for a,b,left,right in zip(points,points[1:],values,values[1:]))/(end-start)

def aggregate(records,groups,metrics=METRICS):
    bins=defaultdict(list)
    for row in records:bins[tuple(row[k] for k in groups)].append(row)
    result=[]
    for group,selected in sorted(bins.items(),key=lambda kv:str(kv[0])):
        for metric in metrics:
            values=[float(r[metric]) for r in selected]
            result.append(dict(zip(groups,group),metric=metric,n_seeds=len(values),mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None))
    return result

def compute(root,check_details=True):
    root=Path(root);snapshot=read(root,'provenance/snapshot.json');freeze=read(root,f'data/{NAME}/freeze.json')
    freeze_sha=sha(root/f'data/{NAME}/freeze.json');records=[];curve_points=[];scene_rows=0;failure_count=0
    models={(int(r['map_seed']),r['method'],int(r['train_seed'])):r for r in rows(root,'model_index.csv') if r['scope']=='new_layout'}
    for job in snapshot['included_completed_jobs']:
        map_seed,seed,method=job['map_seed'],job['train_seed'],job['method']
        directory=f'outputs/{NAME}/map_{map_seed}/{method}/seed_{seed}'
        identity=dict(study=NAME,map_seed=map_seed,train_seed=seed,method=method,freeze_sha256=freeze_sha)
        complete=read(root,directory+'/completion.json');result=read(root,directory+'/result.json');run=read(root,directory+'/run_manifest.json')
        assert complete['complete'] and complete['identity']==result['identity']==run['identity']==identity
        assert result['environment_steps']==200000 and result['online_updates']==199501
        assert result['offline_updates']==(20000 if method in ('dqfd','qdagger') else 0)
        assert result['total_updates']==199501+result['offline_updates']
        assert result['final_model_sha256']==complete['model_sha256']==models[map_seed,method,seed]['model_sha256']
        final=read(root,directory+'/final500/completion.json');reg=final['registration']
        assert final['complete'] and reg['identity']==identity and reg['model_sha256']==result['final_model_sha256']
        assert reg['epsilon']==0 and not reg['teacher'] and not reg['updates'] and reg['static_mask']
        assert reg['scene_sha256']==sha(root/f'data/{NAME}/map_{map_seed}/test500.json')
        curve=rows(root,directory+'/validation_curve.csv')
        assert int(curve[0]['environment_steps_total'])==0 and int(curve[-1]['environment_steps_total'])==200000
        assert len(curve)==51
        curve_points.extend(dict(**job,**r) for r in curve)
        details=rows(root,directory+'/final500/scene_details.csv');scenes=read(root,f'data/{NAME}/map_{map_seed}/test500.json')['scenarios']
        assert len(details)==500 and [r['scenario_id'] for r in details]==[str(s['scenario_id']) for s in scenes]
        for row,scene in zip(details,scenes):
            assert int(row['dynamic_obstacle_count'])==len(scene['obstacles'])
            assert row['dynamic_route_ids']==';'.join(scene['route_ids'])
            assert not flag(row['static_collision'])
        final_metrics=dict(final_safe_success_rate=sum(flag(r['safe_success']) for r in details)/500,
            final_dynamic_collision_rate=sum(flag(r['dynamic_collision']) for r in details)/500,
            final_timeout_rate=sum(r['termination_reason']=='timeout' for r in details)/500)
        close(sum(final_metrics.values()),1,'Final outcomes partition')
        for key,value in final_metrics.items():
            close(final['summary'][key.removeprefix('final_')],value,'Final500 summary '+key)
            close(result['final500'][key.removeprefix('final_')],value,'Run result '+key)
        failed={r['scenario_id']:r for r in details if not flag(r['safe_success'])}
        failures=read(root,directory+'/final500/failure_trajectories.json.gz')
        assert len(failures)==len(failed) and {r['scenario_id'] for r in failures}==set(failed)
        for trace in failures:
            row=failed[trace['scenario_id']]
            assert trace['termination_reason']==row['termination_reason'] and len(trace['trace'])==int(row['steps'])
        failure_count+=len(failures);scene_rows+=len(details)
        if check_details:
            val=rows(root,directory+'/validation_details.csv');validation=read(root,f'data/{NAME}/map_{map_seed}/validation50.json')['scenarios']
            assert len(val)==51*50
            for point in curve:
                batch=[r for r in val if int(r['environment_steps_total'])==int(point['environment_steps_total'])]
                assert len(batch)==50 and [r['scenario_id'] for r in batch]==[str(s['scenario_id']) for s in validation]
                close(point['safe_success_rate'],sum(flag(r['safe_success']) for r in batch)/50,'Validation success')
                close(point['dynamic_collision_rate'],sum(flag(r['dynamic_collision']) for r in batch)/50,'Validation collision')
                close(point['timeout_rate'],sum(r['termination_reason']=='timeout' for r in batch)/50,'Validation timeout')
            updates=rows(root,directory+'/update_diagnostics.csv')
            assert sum(int(r['updates']) for r in updates if r['phase']=='online')==199501
            assert sum(int(r['updates']) for r in updates if r['phase']=='offline')==result['offline_updates']
            for row in updates:
                assert all(math.isfinite(float(v)) for k,v in row.items() if k!='phase' and v!='')
            training=rows(root,directory+'/training.csv')
            assert int(training[-1]['environment_steps_total'])==200000 and int(training[-1]['online_updates'])==199501
            assert sum(int(r['steps']) for r in training)==200000
        timing=result['astar_timing']
        records.append(dict(**job,**final_metrics,early_aulc=auc(curve,0,50000),full_aulc=auc(curve,0,200000),late_aulc=auc(curve,100000,200000),
            late_min_success=min(float(r['safe_success_rate']) for r in curve if int(r['environment_steps_total'])>=100000),
            offline_updates=result['offline_updates'],online_updates=result['online_updates'],
            offline_seconds=result['offline_seconds'],online_seconds=result['online_seconds'],evaluation_seconds=result['evaluation_seconds'],
            checkpoint_seconds=result['checkpoint_seconds'],inference_latency_ms=final['summary']['mean_action_latency_ms'],
            astar_calls=timing.get('astar_lookups',0)+result['online_teacher_queries'],
            astar_searches=timing.get('astar_searches',0)+result['online_teacher_searches'],
            astar_seconds=timing.get('astar_seconds',0)+result['online_teacher_seconds'],
            demo_transitions=result['demo_registration']['transitions'] if result['demo_registration'] else 0))
    assert sum(r['method'] in MAIN for r in records)==90
    per_map=aggregate(records,['map_seed','method']);paired=[]
    for row in records:
        if row['method']=='asvc':continue
        reference=next(r for r in records if r['map_seed']==row['map_seed'] and r['train_seed']==row['train_seed'] and r['method']=='asvc')
        for metric in METRICS:
            paired.append(dict(map_seed=row['map_seed'],train_seed=row['train_seed'],comparison='asvc_minus_'+row['method'],metric=metric,difference=reference[metric]-row[metric]))
    bins=defaultdict(list)
    for row in paired:bins[row['map_seed'],row['comparison'],row['metric']].append(row['difference'])
    paired_summary=[dict(map_seed=k[0],comparison=k[1],metric=k[2],n_seeds=len(v),mean=statistics.mean(v),sample_sd=statistics.stdev(v) if len(v)>1 else None) for k,v in bins.items()]
    macro=[]
    for method in MAIN:
        for seed in range(5):
            selected=[r for r in records if r['method']==method and r['train_seed']==seed];assert len(selected)==3
            macro.append(dict(method=method,train_seed=seed,maps=3,**{k:statistics.mean(r[k] for r in selected) for k in METRICS}))
    return dict(per_seed=records,per_map_mean_sample_sd=per_map,paired_differences=paired,paired_mean_sample_sd=paired_summary,
        cross_map_macro_per_seed=macro,cross_map_macro_mean_sample_sd=aggregate(macro,['method']),validation_curves=curve_points,
        audit=dict(completed_runs=len(records),main_runs=90,ablation_runs=len(records)-90,final500_episodes=scene_rows,
            final500_failure_trajectories=failure_count,validation_points=len(curve_points),validation_scene_rows=len(curve_points)*50))

def verify(root):
    root=Path(root).resolve();manifest=read(root,'manifest.json')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(manifest['files_sha256'])|{'manifest.json'},'File list changed'
    for name,digest in manifest['files_sha256'].items():
        path=local(root,name);assert path.stat().st_size<50*1024**2 and sha(path)==digest,name
        assert path.suffix not in ('.pt','.pth','.tmp','.lock'),name
    for source,item in manifest['source_files'].items():
        path=local(root,item['archive']);h=hashlib.sha256()
        with gzip.open(path,'rb') if item['compressed'] else path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        assert h.hexdigest()==item['raw_sha256'],source
    frozen=read(root,f'data/{NAME}/freeze.json')
    for name,digest in frozen['source_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in frozen['data_sha256'].items():assert sha(root/f'data/{NAME}'/name)==digest,name
    preflight=read(root,f'data/{NAME}/preflight.json');assert preflight['passed'] and preflight['tests']==9
    assert preflight['freeze_sha256']==sha(root/f'data/{NAME}/freeze.json')
    calculated=compute(root)
    for table in ('per_seed','per_map_mean_sample_sd','paired_differences','paired_mean_sample_sd','cross_map_macro_per_seed','cross_map_macro_mean_sample_sd'):
        published=rows(root,'summary/'+table+'.csv');expected=calculated[table];assert len(published)==len(expected),table
        for left,right in zip(published,expected):
            for key,value in right.items():
                if value is None:assert left[key]=='',key
                elif isinstance(value,(float,int)):close(left[key],value,table+'/'+key)
                else:assert left[key]==value,table+'/'+key
    assert read(root,'provenance/verification.json')['scientific_audit']==calculated['audit']
    print(json.dumps(dict(passed=True,files=len(actual),**calculated['audit']),ensure_ascii=False))
    return calculated

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify',action='store_true');parser.add_argument('--recompute',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.recompute:
        output=args.output or args.root.parent/('recomputed_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        assert not output.resolve().is_relative_to(args.root.resolve()),'Recompute outside immutable package'
        values=compute(args.root)
        for name,value in values.items():
            if isinstance(value,list):csv_write(value,output/(name+'.csv'))
        write(values['audit'],output/'audit.json');print(output)
    else:verify(args.root)

if __name__=='__main__':main()
