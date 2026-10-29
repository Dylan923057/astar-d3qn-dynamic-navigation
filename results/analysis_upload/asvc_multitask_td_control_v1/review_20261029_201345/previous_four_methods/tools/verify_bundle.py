"""Build/verify the eight-run multitask reading bundle. Standard library only.

No training, policy evaluation, staging, commit or network operation is performed.
This utility's name deliberately stays outside both frozen training-source globs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='asvc_multitask_external_v1'
BASE=ROOT/'results/analysis_upload'/NAME
LIVE=ROOT/'results'/NAME
REG=ROOT/'data'/NAME
PILOT=ROOT/'data/asvc_multitask_risk_pilot_v1'
METHODS=('C','D','d3qn','dqfd')
RUNS=tuple((m,s) for m in METHODS for s in (0,1))
ALLOW={'.json','.csv','.md','.txt','.log','.py','.yaml','.ps1','.png','.pdf','.svg','.npz','.toml'}
RUN_FILES=('run_manifest.json','completion.json','result.json','status.json','training_episodes.csv',
           'validation_curve.csv','validation_details.csv','update_log.csv','offline_update_log.csv',
           'supervision_statistics.json','sample_statistics.json','offline_cases.csv','final_validation_trajectories.json')
EXTRA_CODE=('scripts/start_external_baselines.ps1','scripts/render_multitask_pairs.py',
            'scripts/inspect_multitask_learning_trends.py','scripts/package_multitask_comparison.py',
            'scripts/sync_multitask_comparison_github.ps1','pyproject.toml')
MEASUREMENTS=('final_success_rate','final_collision_rate','final_timeout_rate','aulc_0_500000',
              'aulc_0_100000','aulc_100000_250000','aulc_250000_500000','aulc_400000_500000','last_five_success_mean')
SECRET_PATTERNS=(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9_]{25,}',
                 r'github_pat_[A-Za-z0-9_]{40,}',r'AKIA[A-Z0-9]{16}',r'sk-[A-Za-z0-9_-]{35,}')


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def csv_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        writer.writeheader();writer.writerows(data)


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def local(base,name):
    base=Path(base).resolve();path=(base/name).resolve()
    require(path.is_relative_to(base),'Path escapes package: '+str(name))
    return path


def close(actual,expected,label):
    require(math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-9),'Statistic differs: '+str(label))


def aulc(curve,start,end):
    xs=[int(r['environment_steps_total']) for r in curve];ys=[float(r['success_rate']) for r in curve]
    require(xs[0]<=start<end<=xs[-1],'AULC would extrapolate')
    def at(t):
        i=bisect_left(xs,t)
        return ys[i] if xs[i]==t else ys[i-1]+(ys[i]-ys[i-1])*(t-xs[i-1])/(xs[i]-xs[i-1])
    points=[start]+[x for x in xs if start<x<end]+[end]
    return sum((b-a)*(at(a)+at(b))*.5 for a,b in zip(points,points[1:]))/(end-start)


def verify(package,check_sources=False):
    package=Path(package).resolve();manifest=read(package/'manifest.json')
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual==set(manifest['files_sha256'])|{'manifest.json'},'Bundle file list differs')
    for name,digest in manifest['files_sha256'].items():
        path=local(package,name)
        require(path.suffix.lower() in ALLOW and path.stat().st_size<50*1024**2,'Unsupported/oversized file: '+name)
        require(sha(path)==digest,'Bundle SHA differs: '+name)
    require(manifest['new_training_runs']==4 and manifest['reused_training_runs']==4 and manifest['training_runs']==8,'Run scope differs')
    require(not manifest['weights_included'] and manifest['test_pool_evaluations']==manifest['additional_training_runs']==0,'Unexpected weights/tests/training')
    shared=package/'registration/shared_pilot';external=package/'registration/external_baselines'
    old=read(shared/'frozen_protocol.json');frozen=read(external/'frozen_protocol.json')
    require(sha(shared/'frozen_protocol.json')==frozen['shared_identity']['pilot_freeze_sha256'],'Shared freeze differs')
    require(old['source_sha256']==frozen['shared_identity']['source_sha256'] and old['input_sha256']==frozen['shared_identity']['inputs_sha256'],'Shared source/input identity differs')
    for name,digest in old['input_sha256'].items():
        require(sha(shared/Path(name).name)==digest,'Shared input differs: '+name)
    expected=dict(frozen['source_sha256'])
    for correction_file in sorted((external/'nontraining_corrections').glob('*.json')):
        correction=read(correction_file)
        require(correction['scope']=='reporting_and_verification_only' and correction['base_freeze_sha256']==sha(external/'frozen_protocol.json'),'Correction identity differs')
        for name,change in correction['source_changes'].items():
            require(name in {'scripts/multitask_external_common.py','scripts/analyze_multitask_external.py'},'Training correction prohibited')
            require(expected[name]==change['before_sha256'] and sha(external/'source_snapshot'/name)==change['before_sha256'],'Original snapshot changed')
            expected[name]=change['after_sha256']
    for registration,freeze in ((shared,old),(external,frozen)):
        for name,digest in freeze['source_sha256'].items():
            require(sha(registration/'source_snapshot'/name)==digest,'Original frozen source snapshot differs: '+name)
    for name,digest in (old['source_sha256']|expected).items():
        require(sha(package/'code'/name)==digest,'Current registered code differs: '+name)
    for name,digest in frozen['artifact_sha256'].items():
        require(sha(external/name)==digest,'External frozen artifact differs: '+name)
    pools=read(shared/'task_pools.json');scenes=read(shared/'validation_scenes.json')
    require({k:len(v) for k,v in pools['pools'].items()}==dict(train=240,validation=50,unseen_combination=60,unseen_goal=60),'Pool counts differ')
    train={t['task_id']:t for t in pools['pools']['train']};validation={t['task_id']:t for t in pools['pools']['validation']}
    held={tuple(g) for g in pools['held_out_goals']};require(len(held)==10,'Held-out goal count differs')
    require(all(tuple(t['goal']) not in held for t in train.values()),'Held-out goal present in training')
    seen_pairs=set()
    for split in pools['pools'].values():
        pairs={(tuple(t['start']),tuple(t['goal'])) for t in split}
        require(len(pairs)==len(split) and not seen_pairs.intersection(pairs),'Task pairs overlap')
        seen_pairs.update(pairs)
    require(len(scenes)==50 and len({s['task_id'] for s in scenes})==50,'Frozen validation differs')
    analysis=package/'results'/NAME
    summaries={(r['method'],int(r['seed'])):r for r in rows(analysis/'run_comparison.csv')}
    require(set(summaries)==set(RUNS),'Summary run list differs')
    model_index={(r['method'],int(r['seed'])):r for r in rows(package/'model_index.csv')}
    curves={};final_details={};validation_count=0;training_count=0
    for method,seed in RUNS:
        directory=package/'runs'/method/f'seed_{seed}'
        run=read(directory/'run_manifest.json');done=read(directory/'completion.json');result=read(directory/'result.json')
        require(done['complete'] and result['complete'] and result['step']==500000,'Incomplete/wrong-budget run')
        offline=20000 if method=='dqfd' else 0
        require(result['updates']==499501+offline and result.get('offline_updates',0)==offline,'Update budget differs')
        require(result.get('online_updates',499501)==499501 and result['test_pools_evaluated'] is False,'Online/test scope differs')
        identity={'group' if method in ('C','D') else 'method':method,'seed':seed,'freeze_sha256':sha((shared if method in ('C','D') else external)/'frozen_protocol.json')}
        require(run['identity']==result['identity']==identity,'Run identity differs')
        require(run['config']==old['config'],'Shared environment/hyperparameter config differs')
        require(model_index[method,seed]['model_sha256']==result['final_model_sha256'],'Model index differs')
        if method not in ('C','D'):
            require(run['shared_identity']==frozen['shared_identity'],'Baseline environment identity differs')
            require(run['no_td_target_bound'] and run['no_online_astar_advice'],'Baseline bound/advice differs')
        else:
            reuse=frozen['reuse_c_d'][f'{method}/seed_{seed}']
            for name,key in (('result.json','result_sha256'),('validation_curve.csv','curve_sha256'),('validation_details.csv','details_sha256')):
                require(sha(directory/name)==reuse[key],'Reused C/D artifact differs')
        curve=rows(directory/'validation_curve.csv');details=rows(directory/'validation_details.csv');curves[method,seed]=curve
        require([int(r['environment_steps_total']) for r in curve]==list(range(0,500001,10000)),'Expected 51 checkpoints')
        require(len(details)==2550,'Expected 2550 validation task records')
        grouped={step:[] for step in range(0,500001,10000)}
        for r in details:grouped[int(r['environment_steps_total'])].append(r)
        for point in curve:
            part=grouped[int(point['environment_steps_total'])]
            require([r['task_id'] for r in part]==[s['task_id'] for s in scenes],'Validation task order differs')
            require([int(r['scene_seed']) for r in part]==[s['scene_seed'] for s in scenes],'Validation scene seed differs')
            for r in part:
                require(sum(int(r[k]) for k in ('success','collision','timeout'))==1,'Outcome categories differ')
                require(r['band']==validation[r['task_id']]['band'] and int(r['length'])==validation[r['task_id']]['length'],'Validation task length differs')
            for metric in ('success','collision','timeout'):
                close(statistics.mean(int(r[metric]) for r in part),point[metric+'_rate'],(method,seed,point['environment_steps_total'],metric))
                for band in ('short','medium','long'):
                    selected=[r for r in part if r['band']==band]
                    close(statistics.mean(int(r[metric]) for r in selected),point[f'{band}_{metric}_rate'],'Length stratum')
        final_details[method,seed]={r['task_id']:r for r in grouped[500000]};validation_count+=len(details)
        summary=summaries[method,seed]
        for start,end in ((0,500000),(0,100000),(100000,250000),(250000,500000),(400000,500000)):
            close(aulc(curve,start,end),summary[f'aulc_{start}_{end}'],(method,seed,start,end))
        close(statistics.mean(float(r['success_rate']) for r in curve[-5:]),summary['last_five_success_mean'],'Last five')
        for metric in ('success','collision','timeout'):close(curve[-1][metric+'_rate'],summary['final_'+metric+'_rate'],'Final rate')
        # Verify actual multi-task collection rather than inferring it from a filename.
        episodes=rows(directory/'training_episodes.csv');training_count+=len(episodes)
        for r in episodes:
            require(r['task_id'] in train,'Non-training task used in collection')
            episode=int(r['episode']);settings=old['config']['tasks'];order=list(settings['length_bands'])
            random.Random(settings['sequence_seed_offset']+seed*1000003+episode//3).shuffle(order)
            rng=random.Random(settings['sequence_seed_offset']+seed*1000003+episode*1000033)
            task=rng.choice([t for t in train.values() if t['band']==order[episode%3]])
            require(task['task_id']==r['task_id'] and rng.randrange(2**32)==int(r['scene_seed']),'Training task/scene sequence differs')
        if method in ('C','D'):
            stats=read(directory/'supervision_statistics.json');draws=stats['replay_actual_statistics']
            require(draws['updates']==499501 and draws['sampled_draws']==499501*64,'Replay update count differs')
            close(draws['eligible_draws']/draws['sampled_draws'],stats['replay_supervision_coverage'],'Replay coverage')
            require(sum(v['draws'] for v in stats['risk_bins'].values())==draws['eligible_draws'],'Risk-bin counts differ')
            require(0<=draws['effective_coefficient_sum']<=draws['base_coefficient_sum']+1e-6,'Invalid supervision coefficient')
        else:
            stats=read(directory/'sample_statistics.json')
            require(stats['online_updates']==499501 and stats['offline_updates']==offline and stats['td_target_clip'] is False,'Baseline sample accounting differs')
    require(validation_count==20400,'Validation record count differs')
    for row in rows(analysis/'method_comparison.csv'):
        selected=[summaries[row['method'],s] for s in (0,1)]
        for metric in MEASUREMENTS:
            values=[float(r[metric]) for r in selected]
            close(statistics.mean(values),row[metric+'_mean'],('mean',row['method'],metric))
            close(statistics.stdev(values),row[metric+'_sample_sd'],('sample SD',row['method'],metric))
    groups={r['task_id']:r['reference_group'] for r in read(external/'paired_reference_groups.json')}
    paired_count=0;trace_count=0
    for method,seed in RUNS:
        for condition in ('dynamic','static'):
            directory=analysis/'paired'/method/f'seed_{seed}'/condition
            done=read(directory/'completion.json');part=rows(directory/'per_task.csv');trajectories=read(directory/'trajectories.json')
            require(done['complete'] and len(part)==50,'Paired evaluation incomplete')
            require(done['identity']['model_sha256']==model_index[method,seed]['model_sha256'],'Paired model identity differs')
            require(done['extra_dynamic_policy_episodes']==0,'Unexpected extra old dynamic policy replay')
            available=sum(r['trace_metrics_available']=='True' for r in part)
            require(available==len(trajectories)==done['trace_tasks'],'Trace coverage differs')
            require(available==(4 if method in ('C','D') and condition=='dynamic' else 50),'Missing trace availability changed')
            for r in part:
                require(r['reference_group']==groups[r['task_id']],'Policy-dependent reference grouping')
                require(sum(int(r[k]) for k in ('success','collision','timeout'))==1,'Paired outcomes differ')
                if condition=='dynamic':
                    for key in ('success','collision','timeout','steps','wait_steps','encounter_steps','conflict_steps'):
                        close(r[key],final_details[method,seed][r['task_id']][key],'Dynamic-score reuse')
                if r['trace_metrics_available']!='True':require(r['periodic_movement_ratio']==r['final_static_distance']=='','Unknown trace metric must remain missing')
            for trace in trajectories:require(len(trace['positions'])==len(trace['actions'])+1,'Incomplete saved trace')
            paired_count+=len(part);trace_count+=len(trajectories)
    require(paired_count==800 and trace_count==616,'Paired task/trajectory total differs')
    attempts=read(external/'demonstration_attempts.json');demo=read(external/'demonstration_report.json')
    counts=Counter(r['task_id'] for r in attempts)
    require(len(attempts)==480 and set(counts)==set(train) and all(v==2 for v in counts.values()),'Frozen demo budget differs')
    require(all(r['goal']==train[r['task_id']]['goal'] for r in attempts),'Demonstration goal differs')
    require(sum(r['retained'] for r in attempts)==demo['retained_episodes']==77,'Retained demonstration count differs')
    require(sum(r['steps'] for r in attempts)==demo['attempted_environment_steps'],'Demonstration collection cost differs')
    require(not demo['validation_or_test_used'] and not demo['old_fixed_goal_demonstrations_used'],'Invalid demo source')
    require(sha(external/'demonstrations.npz')==demo['demonstrations_sha256'],'Demonstration NPZ differs')
    with zipfile.ZipFile(external/'demonstrations.npz') as archive:require(archive.testzip() is None,'Corrupt demonstration NPZ')
    if check_sources:
        for name,digest in (manifest['source_files_sha256']|manifest['models_sha256']).items():
            require(sha(local(ROOT,name))==digest,'Original local source/model changed: '+name)
        for name,digest in read(shared/'legacy_protection.json').items():
            require(sha(local(ROOT,name))==digest,'Protected old result changed: '+name)
    for name in manifest['files_sha256']:
        path=package/name
        if path.suffix.lower() in {'.json','.csv','.md','.txt','.log','.py','.yaml','.ps1','.toml'}:
            text=path.read_text(encoding='utf-8-sig')
            require(not any(re.search(pattern,text) for pattern in SECRET_PATTERNS),'Potential secret found; inspect locally: '+name)
    size=sum((package/name).stat().st_size for name in manifest['files_sha256'])
    print(f'PASS: {len(actual)} files / {size/1024**2:.2f} MiB; 8 completed runs, 408 checkpoints, 20400 validation rows; 800 paired scores, 616 saved traces.',flush=True)
    print(f'Verified {training_count} actual multi-task training episodes, full/phase AULC, seed sample SD, demo budget, frozen sources and trace missingness.',flush=True)
    return manifest


def build():
    pipeline=read(ROOT/'outputs'/NAME/'pipeline_status.json')
    require(pipeline['complete'] and not pipeline['running'] and not pipeline['failures'] and not pipeline['active'] and not pipeline['queued'],'Pipeline is incomplete')
    shared=read(PILOT/'frozen_protocol.json');external=read(REG/'frozen_protocol.json')
    # All live frozen-source checks run read-only in a separate scientific module.
    import multitask_external_common as scientific
    scientific.verify()
    published=BASE/datetime.now().strftime('review_%Y%m%d_%H%M%S')
    require(not published.exists(),'Published reading bundle already exists')
    destination=LIVE/'publication_staging'/published.name
    destination.mkdir(parents=True,exist_ok=False)
    originals={};models={}
    def copy(source,target):
        source=Path(source);target=local(destination,target)
        require(source.is_file() and source.suffix.lower() in ALLOW and source.stat().st_size<50*1024**2,'Unsupported publication input: '+str(source))
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        originals[relative(source)]=sha(source)
    for folder,target in ((PILOT,'registration/shared_pilot'),(REG,'registration/external_baselines')):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:copy(path,Path(target)/path.relative_to(folder))
    names=set(shared['source_sha256'])|set(external['source_sha256'])|set(EXTRA_CODE)
    for name in sorted(names):copy(ROOT/name,Path('code')/name)
    for name in ('MULTITASK_EXTERNAL_PROTOCOL.zh-CN.md','MULTITASK_COMPARISON_UPLOAD.zh-CN.md'):
        copy(ROOT/'docs'/name,Path('docs')/name)
    # Keep the repository-relative live layout so copied report links remain valid.
    summary_target=Path('results')/NAME
    for path in sorted(LIVE.iterdir()):
        if path.is_file() and path.suffix.lower() in {'.json','.csv','.md','.png','.svg','.log'}:
            copy(path,summary_target/path.name)
    for path in sorted((LIVE/'paired').rglob('*')):
        if path.is_file():copy(path,summary_target/'paired'/path.relative_to(LIVE/'paired'))
    trend_dirs=sorted(p for p in LIVE.glob('learning_trend_review_*') if p.is_dir())
    require(bool(trend_dirs),'Latest curve review missing')
    trend=trend_dirs[-1]
    for path in sorted(trend.iterdir()):
        if path.is_file():copy(path,summary_target/trend.name/path.name)
    for name in ('preflight.json','preflight_checks.log','production_cuda_checks.json'):
        path=ROOT/'results/asvc_multitask_risk_pilot_v1'/name
        if path.exists():copy(path,Path('checks/shared_pilot')/name)
    index=[]
    for method,seed in RUNS:
        info=scientific.model_info(method,seed);require(info is not None,'Final model missing')
        directory=info['directory']
        for name in RUN_FILES:
            path=directory/name
            if path.exists():copy(path,Path('runs')/method/f'seed_{seed}'/name)
        console=directory.parents[1]/'logs'/f'{method}_seed_{seed}.log'
        if console.exists():copy(console,Path('runs')/method/f'seed_{seed}/console.log')
        model=directory/'model_final.pth';checkpoint=directory/'checkpoint_latest.pth'
        require(sha(model)==info['model_sha256'],'Final model differs')
        models[relative(model)]=sha(model);models[relative(checkpoint)]=sha(checkpoint)
        result=info['result']
        index.append(dict(method=method,seed=seed,online_steps=500000,online_updates=499501,
            offline_updates=result.get('offline_updates',0),model_path=relative(model),model_sha256=sha(model),
            full_checkpoint_path=relative(checkpoint),checkpoint_sha256=models[relative(checkpoint)],weights_included=False))
    csv_write(destination/'model_index.csv',index)
    # A standard-library verifier travels with the bundle for download-only users.
    copy(Path(__file__),Path('tools/verify_bundle.py'))
    write(destination/'registration/pipeline_status.json',pipeline)
    write(destination/'privacy_and_scope.json',dict(no_credentials_or_private_keys_detected=True,
        credential_scan='High-confidence token/private-key patterns on published text; no account configuration copied.',
        local_paths_retained_as_provenance=True,model_and_replay_files_included=False,
        failed_seeds_removed=False,additional_training_runs=0,test_pool_evaluations=0,
        new_policy_evaluations_during_packaging=0))
    scope=sorted(set(names)-{'pyproject.toml'}|{
        '.gitignore','.gitattributes',relative(REG),relative(published),relative(BASE/'latest.json'),relative(BASE/'LATEST.md'),'docs/GITHUB_SYNC.zh-CN.md',
        'docs/MULTITASK_EXTERNAL_PROTOCOL.zh-CN.md','docs/MULTITASK_COMPARISON_UPLOAD.zh-CN.md'})
    # Historical shared sources are already tracked; include their exact bytes
    # in scope because training freezes depend on them and CRLF would change SHA.
    write(destination/'upload_scope.json',scope)
    (destination/'README.md').write_text(README.replace('@TREND@',trend.name),encoding='utf-8')
    manifest=dict(complete=True,created=datetime.now().astimezone().isoformat(timespec='seconds'),
        training_runs=8,new_training_runs=4,reused_training_runs=4,online_steps_per_run=500000,
        online_environment_steps_total=4000000,new_online_environment_steps=2000000,
        validation_checkpoints=408,validation_episode_rows=20400,paired_task_scores=800,
        static_policy_episodes=400,paired_dynamic_scores_reused=400,additional_dynamic_policy_episodes=0,
        test_pool_evaluations=0,additional_training_runs=0,weights_included=False,
        source_files_sha256=originals,models_sha256=models,
        files_sha256={p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    write(destination/'manifest.json',manifest)
    verify(destination,check_sources=True)
    BASE.mkdir(parents=True,exist_ok=True)
    destination.rename(published);destination=published
    write(BASE/'latest.json',dict(package=relative(destination)))
    (BASE/'LATEST.md').write_text('# 四方法多任务预实验：最新完整阅读包\n\n'
        f'[开始阅读]({destination.name}/README.md) · [最终比较]({destination.name}/results/{NAME}/FINAL_SUMMARY.zh-CN.md) · '
        f'[逐seed AULC]({destination.name}/results/{NAME}/run_comparison.csv) · '
        f'[学习曲线]({destination.name}/results/{NAME}/{trend.name}/learning_trends_full.png)\n\n'
        '新增D3QN/DQfD各2seed，复用多任务C/D各2seed，全部500000在线步。仅冻结验证池，未评估两个测试池。C在当前预算表现最好；D碰撞较少、超时更多，无整体增益证据。两个seed不支持正式显著性结论。\n',encoding='utf-8')
    print(relative(destination),flush=True)
    return destination


README='''# 四方法多任务外部基线与静态/动态配对诊断

本轮已于2026-10-27 04:47完成。新增原始D3QN、DQfD各seed0/1，复用此前多任务C/D四次运行，共8条曲线、每次500000在线环境步。D3QN/DQfD均在相同240任务训练池重新训练，没有拿旧固定任务模型直接验证。

## 先看结果

| 方法 | 成功率seed0/1 | 平均成功 | 平均碰撞 | 平均超时 | 全程AULC均值±样本SD |
|---|---:|---:|---:|---:|---:|
| C 持续弱监督 | 76% / 72% | 74% | 13% | 13% | 47.14% ± 1.78% |
| D 风险自适应 | 74% / 64% | 69% | 3% | 28% | 44.17% ± 0.07% |
| 原始D3QN | 2% / 26% | 14% | 20% | 66% | 5.88% ± 4.95% |
| DQfD | 12% / 20% | 16% | 16% | 68% | 14.94% ± 1.02% |

当前相同50万步预算下，C有明显数值优势，D尚无整体收益；仅2seed，不宣称正式显著性或未见目标泛化。去掉动态障碍后的平均成功率为C 74%、D 63%、D3QN 0%、DQfD 18%，基础多目标导航仍有缺口。不能将超时增加直接归因为监督减少。

## 阅读顺序

1. [最终结论、成本和审稿缺口](results/asvc_multitask_external_v1/FINAL_SUMMARY.zh-CN.md)。
2. [逐seed全程/分阶段AULC](results/asvc_multitask_external_v1/run_comparison.csv) 与 [均值/样本SD](results/asvc_multitask_external_v1/method_comparison.csv)。
3. [完整学习曲线](results/asvc_multitask_external_v1/@TREND@/learning_trends_full.png)、[最后20万步](results/asvc_multitask_external_v1/@TREND@/learning_trends_last_200k.png)、同目录窗口统计CSV/PDF。细线是原始成绩，粗线为仅供显示的5点后向平均，不作正式收敛检验。
4. [配对诊断和代表性失败](results/asvc_multitask_external_v1/DIAGNOSIS.zh-CN.md)、[轨迹图](results/asvc_multitask_external_v1/representative_trajectories.png)、[全部配对任务成绩](results/asvc_multitask_external_v1/paired_all_tasks.csv)。
5. [Q/TD及实际监督审计](results/asvc_multitask_external_v1/learning_audit.csv)、[风险区间系数](results/asvc_multitask_external_v1/risk_weight_audit.csv)。
6. [冻结协议、启动/完整续训与未启动方案](docs/MULTITASK_EXTERNAL_PROTOCOL.zh-CN.md)。

## 数据范围与复用

- `runs/<方法>/seed_0或1/`：8次运行的完整51点曲线、各2550条验证明细、实际多任务训练回合、更新/监督/PER成本日志、登记与完成记录；C/D历史逐步离线案例也保留。
- 总计408个检查点、20400条训练期验证记录，反复使用相同50场景，不能当成20400个独立测试任务。其中C/D的10200条复用，新增基线10200条。
- 静态/动态配对表800条，其中400条动态成绩复用原最终验证，400条静态是新增无训练评估，不把800条都当新增评估。包内616份可用完整配对轨迹，旧C/D动态仅各4/50；缺少的循环/终止距离保持NA，未擅自补跑。
- `registration/shared_pilot/`：共同240训练/50验证/两个60测试任务池、50个冻结场景、旧冻结、原始代码快照与旧结果保护登记。保存测试池定义不代表做过测试。
- `registration/external_baselines/`：新冻结、示范计划、480次成功/失败尝试、3472条示范NPZ、覆盖报告、原始快照与单独登记的报表修正。示范成功77次、覆盖70/240任务、31/40目标；没有使用旧固定起终点示范或验证/测试任务，也没有临时增加预算。
- 每个方法499501次在线更新；DQfD额外20000离线更新/seed，并永久保留示范。示范生成15107环境步由两个seed共享，成本另列，不能宣称总成本相同。
- `code/`：原始训练代码、当前通过登记修正后的分析代码、配置、测试和只读工具。训练源字节与原冻结核对；两个报表/校验修正保留原快照，不改训练身份。
- `model_index.csv`：8个最终模型及完整断点的本地路径/SHA。权重和回放本体留在本地；没有权重就不能重新执行网络评估，只有权重也不等于完整续训。已有结果的只读复核无需它们。
- 旧固定任务unguided_bound证据单列于results下legacy_*表，不能混入新多任务成绩。

## 校验命令

下载整个目录后，只需Python标准库：

```powershell
python tools/verify_bundle.py --verify --package .
```

仓库根目录：

```powershell
python scripts/package_multitask_comparison.py --verify
# 仅原工作空间具有模型、完整断点和原始日志时：
python scripts/package_multitask_comparison.py --verify --check-sources
```

校验包括全部文件SHA、原冻结/报表修正、多任务训练任务和场景序列、20400条验证、全程与阶段AULC、seed间样本SD、800条配对成绩与缺失轨迹、示范预算和来源。没有PyTorch也能校验已有结果；不训练、不跑网络、不改Git索引。

## 结论边界与后续

C仍在改善但后期放缓，部分seed已横盘；D3QN有seed长期低水平，DQfD末段不持续上升。50万步预算不足可能是因素，延长收益尚未验证。追加预算、CE监督、监督下限0/0.05/0.1/0.2、五seed和正式消融均未启动。另开预算扩展时必须保留原50万步结果，同预算比较并完整恢复断点，不能挑失败seed或重设日程。

GitHub上传范围由upload_scope.json限定。阅读包不包含账户配置、凭据、文档转换工具、模型、回放、缓存或其他未提交实验；原始日志中本地路径作为来源保留。SHA保证内容一致，不代表结论已独立复现。
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--check-sources',action='store_true')
    parser.add_argument('--package',type=Path);parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    if args.verify:
        package=args.package
        if package is None:
            package=ROOT if (ROOT/'manifest.json').exists() else ROOT/read(BASE/'latest.json')['package']
        verify(package,args.check_sources)
    else:
        package=build()
        if args.archive:
            archive=BASE.parent/f'multitask_comparison_{package.name}.zip'
            require(not archive.exists(),'Archive already exists')
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as target:
                for path in sorted(package.rglob('*')):
                    if path.is_file():target.write(path,Path(package.name)/path.relative_to(package))
            print(f'Archive: {archive} ({archive.stat().st_size/1024**2:.2f} MiB)',flush=True)


if __name__=='__main__':main()
