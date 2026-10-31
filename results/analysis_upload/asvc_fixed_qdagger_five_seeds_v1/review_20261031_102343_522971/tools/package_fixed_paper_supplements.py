"""Archive CE500 confirmation and fixed-task QDagger. Verify with standard library only."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
import zipfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import package_fixed_ce_evidence as base
import package_fixed_ce_five_evidence as previous

NAME='asvc_fixed_paper_supplements_v1'
CE='asvc_fixed_ce_final500_v1';QD='asvc_fixed_qdagger_v1'
BASE=ROOT/'results/analysis_upload'/NAME
read,write,rows,write_csv,sha,local,normal,require,close,integral=(getattr(base,n) for n in
    ('read','write','rows','write_csv','sha','local','normal','require','close','integral'))
CODE=['configs/asvc_fixed_ce_final500_v1.yaml','configs/asvc_fixed_qdagger_v1.yaml',
    'scripts/fixed_supplement_common.py','scripts/run_fixed_ce_final500.py','scripts/fixed_qdagger_agent.py',
    'scripts/fixed_qdagger_common.py','scripts/run_fixed_qdagger.py','scripts/analyze_fixed_qdagger.py',
    'scripts/start_fixed_qdagger.ps1','scripts/watch_fixed_paper_supplements.ps1',
    'scripts/audit_fixed_paper_supplements.py','scripts/review_fixed_paper_supplements.py',
    'scripts/package_fixed_paper_supplements.py','scripts/sync_fixed_paper_supplements_github.ps1',
    'tests/test_fixed_qdagger.py','docs/ASVC_FIXED_QDAGGER.zh-CN.md','docs/ASVC_FIXED_SUPPLEMENTS_UPLOAD.zh-CN.md']
ATTR=['.gitignore','configs/.gitattributes','scripts/.gitattributes','docs/.gitattributes','tests/.gitattributes',
    f'data/{CE}/.gitattributes',f'data/{QD}/.gitattributes',f'results/analysis_upload/{NAME}/.gitattributes']
METRICS=['aulc_0_50000','aulc_0_200000','aulc_100000_200000','late_min_safe_success_rate',
    'final_safe_success_rate','final_collision_rate','final_timeout_rate',
    'late_collision_rate_aulc','late_timeout_rate_aulc','late_collision_rate_point_mean',
    'late_timeout_rate_point_mean','late_collision_rate_point_max','late_timeout_rate_point_max']
FINAL_KEYS=['safe_success_rate','dynamic_collision_rate','timeout_rate','mean_steps','mean_wait_steps',
    'mean_wait_fraction','pooled_wait_fraction','mean_path_efficiency_success_only','safe_success_weighted_path_efficiency']
PATTERNS=(r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}',r'AKIA[0-9A-Z]{16}',
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def flag(value):return value is True or str(value).lower() in ('1','1.0','true')


def archived(package,name):
    path=local(package,name)
    if path.exists():return path
    require(path.suffix=='.json' and Path(str(path)+'.gz').exists(),'Missing archive source: '+str(name))
    return Path(str(path)+'.gz')


def read_archived(package,name):
    path=archived(package,name)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8-sig') as stream:return json.load(stream)
    return read(path)


def scientific_sha(path):
    if path.suffix!='.gz':return sha(path)
    digest=hashlib.sha256()
    with gzip.open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def scene_order(details,scenes):
    require(len(details)==len(scenes),'Scenario count differs')
    require([r['scenario_id'] for r in details]==[s['scenario_id'] for s in scenes],'Scenario order differs')
    for row,scene in zip(details,scenes):
        require(int(row['dynamic_obstacle_count'])==len(scene['obstacles']),'Obstacle count differs')
        require(str(row['dynamic_route_ids'])==';'.join(map(str,scene['route_ids'])),'Route order differs')


def failures_match(traces,details):
    failed={r['scenario_id']:r for r in details if not flag(r['safe_success'])}
    require(len(traces)==len(failed) and {t['scenario_id'] for t in traces}==set(failed),'Failure trajectories differ')
    for trace in traces:
        row=failed[trace['scenario_id']]
        require(len(trace['steps'])==int(row['steps']) and trace['termination_reason']==row['termination_reason'],
            'Failure length/outcome differs')
        require(trace['steps'][-1]['termination_reason']==row['termination_reason'],'Last failure step differs')
    return len(traces)


def final_metrics(details):
    n=len(details);eff=[float(r['path_efficiency']) for r in details if r.get('path_efficiency') not in ('',None)]
    return dict(safe_success_rate=sum(flag(r['safe_success']) for r in details)/n,
        dynamic_collision_rate=sum(flag(r['dynamic_collision']) for r in details)/n,
        timeout_rate=sum(r['termination_reason']=='timeout' for r in details)/n,
        mean_steps=statistics.mean(float(r['steps']) for r in details),
        mean_wait_steps=statistics.mean(float(r['wait_steps']) for r in details),
        mean_wait_fraction=statistics.mean(float(r['wait_steps'])/float(r['steps']) for r in details),
        pooled_wait_fraction=sum(float(r['wait_steps']) for r in details)/sum(float(r['steps']) for r in details),
        mean_path_efficiency_success_only=statistics.mean(eff),
        safe_success_weighted_path_efficiency=sum(float(r['path_efficiency'] or 0)*flag(r['safe_success']) for r in details)/n)


def curve_check(directory,scenes,*,qdagger=False):
    curve=rows(directory/'validation_curve.csv');details=rows(directory/'validation_details.csv')
    steps=[int(r['environment_steps_total']) for r in curve]
    require(len(steps)==21 and steps==sorted(set(steps)) and steps[0]==0 and steps[-1]==200000,'200k curve budget differs')
    require(len(details)==1050,'Expected 21 x 50 validation records')
    grouped=defaultdict(list)
    for row in details:grouped[int(row['environment_steps_total'])].append(row)
    failures=0
    for point in curve:
        step=int(point['environment_steps_total']);part=grouped[step];scene_order(part,scenes)
        close(point['safe_success_rate'],statistics.mean(float(r['safe_success']) for r in part),'success')
        close(point['collision_rate'],statistics.mean(float(r['collision']) for r in part),'collision')
        close(point['timeout_rate'],sum(r['termination_reason']=='timeout' for r in part)/50,'timeout')
        suffix=('after_pretraining_' if step==0 else 'online_') if qdagger else ''
        failures+=failures_match(read_archived(directory,f'validation_failures_{suffix}{step:06d}.json'),part)
    result={f'aulc_{a}_{b}':integral(curve,a,b,'safe_success_rate') for a,b in ((0,50000),(0,200000),(100000,200000))}
    late=[r for r in curve if int(r['environment_steps_total'])>=100000]
    result['late_min_safe_success_rate']=min(float(r['safe_success_rate']) for r in late)
    for field in ('safe_success_rate','collision_rate','timeout_rate'):result['final_'+field]=float(curve[-1][field])
    for field in ('collision_rate','timeout_rate'):
        result['late_'+field+'_aulc']=integral(curve,100000,200000,field)
        result['late_'+field+'_point_mean']=statistics.mean(float(r[field]) for r in late)
        result['late_'+field+'_point_max']=max(float(r[field]) for r in late)
    return result,failures,curve


def verify(package,check_sources=False):
    package=Path(package).resolve();manifest=read(package/'manifest.json')
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual==set(manifest['files_sha256'])|{'manifest.json'},'Archive file list differs')
    allow=base.ALLOW|{'.gin','.npz'}
    for name,digest in manifest['files_sha256'].items():
        path=local(package,name)
        require(path.suffix.lower() in allow or path.name in ('.gitattributes','.gitignore','LICENSE'),'Unsupported artifact: '+name)
        require(path.stat().st_size<50*1024**2 and sha(path)==digest,'Oversized/changed artifact: '+name)
        if path.suffix.lower() not in ('.gz','.png','.pdf','.npz'):
            require(not any(re.search(p,path.read_text(encoding='utf-8-sig')) for p in PATTERNS),'Potential credential: '+name)
    require(manifest['new_training_runs']==2 and manifest['reused_training_runs']==14 and
        manifest['new_online_environment_steps']==400000 and manifest['new_offline_updates']==40000 and
        manifest['new_online_updates']==399002 and not manifest['weights_included'] and not manifest['multitask_used'],
        'Supplement scope differs')
    for source,destination in manifest['source_to_archive'].items():
        path=local(package,destination)
        value=scientific_sha(path) if source.endswith('.json') and destination.endswith('.gz') else sha(path)
        require(value==manifest['source_files_sha256'][source],'Source bytes differ: '+source)
    prior=package/'previous_ce_five';previous.verify(prior,False)
    prior_manifest=read(prior/'manifest.json')
    require(sha(prior/'manifest.json')==manifest['previous_ce_five_manifest_sha256'],'Prior bundle differs')
    def source_sha(name):
        name=normal(name)
        return manifest['source_files_sha256'].get(name,manifest['models_sha256'].get(name))
    for study,protected in ((CE,'protected_sha256'),(QD,'protected_sha256')):
        frozen=read(package/f'data/{study}/freeze.json')
        for group in ('source_sha256',protected):
            for name,value in frozen[group].items():require(source_sha(name)==value,'Frozen scientific source differs: '+name)
        if study==CE:require(sha(package/f'data/{CE}/frozen500.json')==frozen['copied_scene_sha256'],'CE500 copy differs')
        else:
            for name,value in frozen['input_sha256'].items():require(source_sha(name)==value,'QDagger input differs')
            for name,value in frozen['copied_files_sha256'].items():require(sha(local(package/f'data/{QD}',name))==value,'QDagger copied freeze differs')
    ce_frozen=read(package/f'data/{CE}/freeze.json');qf=read(package/f'data/{QD}/freeze.json')
    scene500=read(package/f'data/{CE}/frozen500.json')['scenarios'];scene50=read(package/f'data/{QD}/frozen50.json')['scenarios']
    require(len(scene500)==500 and len(scene50)==50,'50/500 split differs')
    require([s['scenario_id'] for s in scene500]==ce_frozen['scene_order'],'CE500 frozen order differs')
    require(not {s['scenario_id'] for s in scene500}&{s['scenario_id'] for s in scene50},'Validation and confirmation overlap')
    final_analysis=package/f'results/{CE}';reported=rows(final_analysis/'per_seed.csv');final_lookup={};final_rows={}
    failure500=0
    for item in ce_frozen['models']:
        seed=item['seed']
        for method in ('margin','ce'):
            directory=local(package,item['reused_margin']) if method=='margin' else package/f'outputs/{CE}/seed_{seed}'
            require(read(directory/'completion.json')['complete'],'Final500 incomplete')
            registration=read(directory/'registration.json')
            digest=item['margin_model_sha256'] if method=='margin' else item['model_sha256']
            for key,value in dict(model_sha256=digest,scene_sha256=ce_frozen['scene_sha256'],scene_count=500,
                checkpoint_steps=200000,epsilon=0,teacher=False,static_mask=True,dynamic_shield=False,device='cpu').items():
                require(registration[key]==value,'Final500 evaluation protocol differs: '+key)
            details=rows(directory/'scene_details.csv');scene_order(details,scene500)
            failure_name=item['margin_failure_path'] if method=='margin' else f'outputs/{CE}/seed_{seed}/failure_trajectories.json'
            failure500+=failures_match(read_archived(package,failure_name),details)
            expected=final_metrics(details);row=next(r for r in reported if r['method']==method and int(r['seed'])==seed)
            for key in FINAL_KEYS:close(row[key],expected[key],'CE500 '+key)
            final_lookup[method,seed]=expected;final_rows[method,seed]={r['scenario_id']:r for r in details}
    for row in rows(final_analysis/'mean_sample_sd.csv'):
        values=[final_lookup[row['method'],s][row['metric']] for s in range(5)]
        close(row['mean'],statistics.mean(values),'CE500 mean');close(row['sample_sd'],statistics.stdev(values),'CE500 sample SD')
    for row in rows(final_analysis/'paired_per_seed.csv'):
        seed=int(row['seed'])
        for key in FINAL_KEYS:close(row[key],final_lookup['ce',seed][key]-final_lookup['margin',seed][key],'CE500 paired seed')
    for row in rows(final_analysis/'paired_mean_sample_sd.csv'):
        values=[final_lookup['ce',s][row['metric']]-final_lookup['margin',s][row['metric']] for s in range(5)]
        close(row['mean'],statistics.mean(values),'CE500 paired mean');close(row['sample_sd'],statistics.stdev(values),'CE500 paired SD')
    combined=rows(final_analysis/'all_scene_details.csv');require(len(combined)==5000,'CE500 combined count differs')
    for row in combined:
        expected=final_rows[row['method'],int(row['seed'])][row['scenario_id']]
        for key in expected:require(str(row[key])==str(expected[key]),'CE500 combined record differs')
    paired=rows(final_analysis/'paired_scene_differences.csv');require(len(paired)==2500,'CE500 paired scene count differs')
    for row in paired:
        s=int(row['seed']);sid=row['scenario_id'];a=final_rows['margin',s][sid];b=final_rows['ce',s][sid]
        for target,key in (('delta_safe_success','safe_success'),('delta_dynamic_collision','dynamic_collision')):
            close(row[target],int(flag(b[key]))-int(flag(a[key])),'CE500 scene contrast')
        close(row['delta_timeout'],int(b['termination_reason']=='timeout')-int(a['termination_reason']=='timeout'),'CE500 timeout contrast')
        for target,key in (('delta_steps','steps'),('delta_wait_steps','wait_steps')):close(row[target],float(b[key])-float(a[key]),'CE500 path contrast')
        if a['path_efficiency'] and b['path_efficiency']:
            close(row['delta_path_efficiency_success_in_both'],float(b['path_efficiency'])-float(a['path_efficiency']),'CE500 efficiency contrast')
    qanalysis=package/f'results/{QD}';summaries=rows(qanalysis/'per_seed_metrics.csv');metrics={};curves={};failure50=0
    require(len(summaries)==8,'Expected four paired methods x two seeds')
    old_runs={(r['method'],r['seed']):r for r in prior_manifest['runs']}
    baseline={(r['method'],r['seed']):r for r in qf['checks']['baseline_audit']}
    for method in ('margin','d3qn','dqfd','qdagger'):
        for seed in (0,1):
            if method=='margin':directory=local(prior,old_runs[method,seed]['directory'])
            elif method=='qdagger':directory=package/f'outputs/{QD}/seed_{seed}'
            else:directory=local(package,baseline[method,seed]['directory'])
            values,failed,curve=curve_check(directory,scene50,qdagger=(method=='qdagger'));failure50+=failed
            row=next(r for r in summaries if r['method']==method and int(r['seed'])==seed)
            for key,value in values.items():close(row[key],value,'QDagger comparison '+key)
            metrics[method,seed]=values;curves[method,seed]=curve
            result=read(directory/'result.json');require(result['environment_steps']==200000,'Training budget differs')
            if method!='qdagger':continue
            require(read(directory/'completion.json')['complete'] and result['complete'],'QDagger incomplete')
            require(result['offline_updates']==20000 and result['online_updates']==199501 and result['total_updates']==219501,'QDagger update budget differs')
            require(result['freeze_sha256']==sha(package/f'data/{QD}/freeze.json'),'QDagger run freeze differs')
            require(result['teacher_queries']==119999 and result['student_replay_only'] and result['teacher_exit']==120000
                and not result['teacher_advice'] and not result['td_clip'] and result['supervision_floor']==0,'QDagger mechanisms differ')
            run=read(directory/'run_manifest.json')
            require(run['config']==qf['config'] and run['shared_config']==qf['shared_config'],'QDagger run config differs')
            require(source_sha(f'outputs/{QD}/seed_{seed}/model_final.pth')==result['final_model_sha256'],'QDagger final model differs')
            logs=rows(directory/'update_diagnostics.csv')
            for phase,updates in (('offline',20000),('online',199501)):
                part=[r for r in logs if r['phase']==phase];require(sum(int(r['count']) for r in part)==updates,'Update log count differs')
            for log in logs:
                require(all(math.isfinite(float(value)) for key,value in log.items() if key!='phase' and value!=''),'Nonfinite diagnostics')
                if log['phase']=='offline':close(log['weight_mean'],3,'Offline weight')
                elif int(log['bin_index'])>=120:close(log['weight_max'],0,'Teacher exit')
            training=rows(directory/'training.csv')
            require(int(training[-1]['environment_steps_total'])==200000 and int(training[-1]['online_updates_total'])==199501,'Online log budget differs')
            require(all(int(r['demo_replay_samples'])==0 for r in training),'Permanent demonstration mixing found')
            before=read_archived(directory,'before_pretraining50.json');scene_order(before['details'],scene50)
            failure50+=failures_match(before['failures'],before['details'])
            require(before['failures']==read_archived(directory,'validation_failures_before_pretraining_000000.json'),'Pretraining failure export differs')
            for sample in rows(directory/'n_step_samples.csv'):
                n=int(sample['n_length']);require(n in (1,2,3),'n-step tail length differs')
                terminal=flag(sample['n_terminated']);require(flag(sample['bootstrap'])!=terminal,'Terminal bootstrap differs')
                close(sample['bootstrap_discount'],0 if terminal else .99**n,'Actual tail discount')
                step=int(sample['online_environment_step']);close(sample['distillation_weight'],3 if sample['phase']=='offline' else 3*max(1-step/120000,0),'Distillation schedule')
    for row in rows(qanalysis/'mean_sample_sd.csv'):
        values=[metrics[row['method'],s][row['metric']] for s in (0,1)]
        close(row['mean'],statistics.mean(values),'QDagger mean');close(row['sample_sd'],statistics.stdev(values),'QDagger sample SD')
    for row in rows(qanalysis/'paired_per_seed.csv'):
        seed=int(row['seed']);method=row['method'].removeprefix('qdagger_minus_')
        for key in METRICS:close(row[key],metrics['qdagger',seed][key]-metrics[method,seed][key],'QDagger paired contrast')
    for row in rows(qanalysis/'paired_mean_sample_sd.csv'):
        method=row['method'].removeprefix('qdagger_minus_');key=row['metric']
        values=[metrics['qdagger',s][key]-metrics[method,s][key] for s in (0,1)]
        close(row['mean'],statistics.mean(values),'QDagger contrast mean');close(row['sample_sd'],statistics.stdev(values),'QDagger contrast SD')
    points=rows(qanalysis/'all_validation_points.csv');require(len(points)==168,'Combined learning point count differs')
    for row in points:
        curve=curves[row['method'],int(row['seed'])];point=next(r for r in curve if r['environment_steps_total']==row['environment_steps_total'])
        for key in ('safe_success_rate','collision_rate','timeout_rate'):close(row[key],point[key],'Combined learning point')
    require(read(package/f'results/{NAME}/evidence_audit.json')['qdagger_all_complete'],'Full-state audit missing')
    index=rows(package/'model_index.csv');require(len(index)==16,'Expected sixteen final model identities')
    for row in index:require(source_sha(row['model_path'])==row['model_sha256'],'Final model index differs')
    require(manifest['unique_validation_rows']==16800 and manifest['new_confirmation_episodes']==2500 and manifest['reused_confirmation_episodes']==2500,'Episode accounting differs')
    if check_sources:
        for name,value in (manifest['source_files_sha256']|manifest['models_sha256']).items():require(sha(local(ROOT,name))==value,'Local source/model differs: '+name)
    size=sum(local(package,n).stat().st_size for n in actual)
    print(f'PASS SUPPLEMENTS: {len(actual)} files / {size/1024**2:.2f} MiB; 16 final models indexed, no binaries.')
    print(f'CE500: 2500 new + 2500 reused, {failure500} failures checked. QDagger: two 200k + 20k-offline runs.')
    print(f'16800 unique repeated-validation rows; 5000 confirmation rows; all raw bytes, actual-step AULC, sample SD and failure exports checked ({failure50} four-method/pretraining failures).')
    return manifest


README='''# 固定起终点补实验：CE最终500确认与QDagger任务适配

两项补实验均于2026-10-30完成，源结果和旧冻结不覆盖。本轮只有QDagger seed0、1两次新训练，各20000离线更新＋200000在线环境步；CE五个已完成最终模型新增2500回合确认，原大间隔组2500回合确认直接复用。没有50万步多任务研究或其他新增训练。

## 核心结果与阅读顺序

|500场景确认，5训练seed|大间隔|CE|
|---|---:|---:|
|安全成功率（均值±样本SD）|99.28%±0.73%|98.32%±1.48%|
|动态碰撞率|0.72%±0.73%|1.36%±1.34%|
|超时率|0|0.32%±0.72%|

CE−大间隔安全成功配对差−0.96±1.99个百分点；seed0–2更高、seed3–4更低，不能说每个seed都由大间隔胜出。CE共有42次失败（34碰撞、8超时），原组18次碰撞。

|50场景重复验证，仅seed0/1|全程AULC均值±样本SD|最终安全成功|
|---|---:|---:|
|当前持续弱监督大间隔|95.54%±0.06%|100%|
|DQfD|93.10%±0.43%|100%|
|原始D3QN|31.56%±5.76%|91%±1.41%|
|QDagger任务适配版|75.90%±3.79%|90%±5.66%|

QDagger最终seed0/1成功86%/94%，碰撞均2%，超时12%/4%；全程和后半段波动明显。早期0%后来恢复，退出教师附近再回落；时间相邻不是因果证明。数值日志有限，不将成绩低直接归于Q爆炸。

1. [合并总报告](results/asvc_fixed_paper_supplements_v1/REPORT.zh-CN.md)、[CE500详细报告](results/asvc_fixed_ce_final500_v1/REPORT.zh-CN.md)、[QDagger四方法报告](results/asvc_fixed_qdagger_v1/REPORT.zh-CN.md)。
2. CE的[逐seed](results/asvc_fixed_ce_final500_v1/per_seed.csv)、[均值/样本SD](results/asvc_fixed_ce_final500_v1/mean_sample_sd.csv)、[逐场景配对](results/asvc_fixed_ce_final500_v1/paired_scene_differences.csv)；5000条原/新模型逐场景记录与全部失败可核验。
3. QDagger的[逐seed](results/asvc_fixed_qdagger_v1/per_seed_metrics.csv)、[均值/样本SD](results/asvc_fixed_qdagger_v1/mean_sample_sd.csv)、[配对差](results/asvc_fixed_qdagger_v1/paired_per_seed.csv)、[四方法学习曲线](results/asvc_fixed_qdagger_v1/learning_curves.png)，保留PDF和所有真实步数。
4. [原文机制—本次实现—适配差异及复现命令](docs/ASVC_FIXED_QDAGGER.zh-CN.md)、data/asvc_fixed_qdagger_v1中的冻结、官方源码/许可证、107回合示范审计与原7260条训练示范。
5. outputs/asvc_fixed_qdagger_v1中的两个完整训练日志、220个更新诊断区间/seed、三步样本、预训练前和在线0步验证、全部失败轨迹；outputs/asvc_fixed_ce_final500_v1为新确认的全部原始结果。
6. [完成及完整恢复审计](results/asvc_fixed_paper_supplements_v1/evidence_audit.json)、[五seed训练/评估身份补核](results/asvc_fixed_paper_supplements_v1/extended_identity_audit.json)、[16个最终模型身份](model_index.csv)。权重和完整回放断点留在原工作区；另有excluded_model_artifacts.csv记录排除的模型/恢复文件SHA。

## 复用与统计边界

previous_ce_five是上一轮完整CE五seed阅读包，按原字节嵌入，其历史“500尚未评估”仅描述上一轮，本轮以新的CE500报告为准。旧两seed/五seed包本身不覆盖。原始D3QN/DQfD各seed0、1的完整CSV和全部验证失败轨迹也已纳入，不重新训练。

共16条200k学习曲线、336个检查点、16800条50场景重复验证记录；另有5000条500确认记录，及QDagger预训练前100条结果单列。不是16800个独立任务或训练seed。CE是5训练seed配对确认；QDagger是2训练seed预实验，没有运行QDagger最终500，不混合两种统计口径。

QDagger复用原DQfD的7260条示范（107/128成功回合，生成10756环境步），不增加示范。每seed额外20000离线更新，在线199501更新，总219501；教师查询、损失/Q/TD范围和耗时单列。学生三步Double-DQN＋Huber，T=0.1，确定性教师one-hot，λ(t)=3max(1−t/120000,0)。不加本方法的代选、TD裁剪、监督下限、风险软权重或DQfD附加损失。这是任务适配，不能声称原Atari完整复现；与主方法的差值是组合协议差，不能归于单一模块。

预检与中断恢复单元单列，保留未完成预检尝试，不算正式训练。无非有限数值失败，没有换seed或按结果改参数，不宣称正式统计显著性或跨地图泛化。原50最终验证与独立冻结500确认分开；500不是训练/调参依据。

## 数据压缩与独立校验

所有validation_failures、failure_trajectories、CE批次缓存、预训练前详情JSON都按原字节无损gzip。manifest同时记录原始SHA与压缩文件SHA；没有抽样删失败。可用Python标准库gzip解压。权重不能凭本包重建；重跑网络评估需要对应原模型，完整续训需要原完整断点。

仅下载本目录，进入目录后验证（只需Python标准库，无PyTorch/GPU/权重）：

```powershell
python tools/verify_bundle.py --verify --package .
```

完整仓库根目录：

```powershell
python scripts/package_fixed_paper_supplements.py --verify
# 仅保存全部原始文件/模型的原工作区：
python scripts/package_fixed_paper_supplements.py --verify --check-sources
```

验证核对SHA、原冻结、所有逐场景和失败、实际步数AULC、均值/样本SD、配对差、更新预算、蒸馏退出与恢复审计记录，不运行训练、策略评估或联网。upload_scope.json限定本批上传范围；旧根README、其他研究、模型/回放/缓存不混入。具体上传命令见[上传说明](docs/ASVC_FIXED_SUPPLEMENTS_UPLOAD.zh-CN.md)。
'''


def build(new_revision=False,archive=False):
    pointer=BASE/'latest.json'
    if pointer.exists() and not new_revision:
        info=read(pointer);package=local(ROOT,info['package']);verify(package,True)
        print('Existing publication reused:',package.relative_to(ROOT));return package
    import fixed_qdagger_common as q
    import run_fixed_ce_final500 as ce
    q.verify();ce.verify()
    require(read(ROOT/f'results/{QD}/queue_status.json')['complete'],'Wait for QDagger queue')
    require(read(ROOT/f'results/{NAME}/postprocessing_status.json')['complete'],'Wait for final postprocessing')
    old_pointer=read(previous.BASE/'latest.json');old=local(ROOT,old_pointer['package'])
    require(sha(old/'manifest.json')==old_pointer['manifest_sha256'],'Prior CE bundle pointer differs')
    previous.verify(old,False)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');dirname='review_'+stamp
    stage=ROOT/f'results/{NAME}/packaging_staging'/dirname
    final=BASE/dirname;stage.mkdir(parents=True,exist_ok=False)
    sources={};destinations={};models={};excluded=[]
    def copy(name,destination=None):
        name=normal(name);path=local(ROOT,name)
        if path.suffix in ('.pth','.pt'):
            if name not in models:
                models[name]=sha(path);excluded.append(dict(path=name,bytes=path.stat().st_size,sha256=models[name],included=False))
            return
        if not path.is_file() or '__pycache__' in path.parts:return
        # A repository .gitignore copied to the bundle root would hide its raw
        # outputs. Preserve its exact bytes as provenance, without activating it.
        destination=normal(destination or ('provenance/root_gitignore.txt' if name=='.gitignore' else name))
        compress=path.suffix=='.json' and (path.name.startswith(('validation_failures_','batch_')) or path.name in ('failure_trajectories.json','before_pretraining50.json'))
        if compress:destination+='.gz'
        if name in destinations:
            require(destinations[name]==destination,'Conflicting source destination');return
        target=local(stage,destination);target.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with path.open('rb') as stream,target.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as zipped:
                shutil.copyfileobj(stream,zipped)
        else:shutil.copyfile(path,target)
        sources[name]=sha(path);destinations[name]=destination
    for p in sorted(old.rglob('*')):
        if p.is_file():copy(p.relative_to(ROOT).as_posix(),'previous_ce_five/'+p.relative_to(old).as_posix())
    prior_manifest=read(old/'manifest.json');models.update(prior_manifest['models_sha256'])
    qf=read(ROOT/f'data/{QD}/freeze.json');cef=read(ROOT/f'data/{CE}/freeze.json')
    for name in base.dependency_files(list(qf['source_sha256'])+list(cef['source_sha256'])+CODE+['scripts/review_persistent_supervision_five_seeds.py']):copy(name)
    for frozen in (qf,cef):
        for group in ('source_sha256','protected_sha256','input_sha256'):
            for name in frozen.get(group,{}):copy(name)
    for study in (QD,CE):
        for path in sorted((ROOT/f'data/{study}').rglob('*')):
            if path.is_file():copy(path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT/f'outputs/{study}').rglob('*')):
            if path.is_file() and path.suffix.lower() in base.ALLOW|{'.pth','.pt'}:copy(path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT/f'results/{study}').rglob('*')):
            if path.is_file() and path.suffix.lower() in base.ALLOW|{'.pth','.pt'}:copy(path.relative_to(ROOT).as_posix())
    for path in (ROOT/f'results/{NAME}').iterdir():
        if path.is_file() and path.name!='postprocessing.lock':copy(path.relative_to(ROOT).as_posix())
    for item in qf['checks']['baseline_audit']:
        if item['method']=='margin':continue
        for p in sorted(local(ROOT,item['directory']).iterdir()):
            if p.is_file() and p.name not in ('static_final.json','static_final_trace.json'):copy(p.relative_to(ROOT).as_posix())
    for name in ('independent_final_scenarios.json','freeze_manifest.json'):
        copy('data/whole_map_91701_supervision_tail_v1/'+name)
    for name in CODE+ATTR+['pyproject.toml']:copy(name)
    # Portable verifier dependencies are byte-preserved; original scientific files remain at their root paths.
    for name,target in [('scripts/package_fixed_paper_supplements.py','tools/verify_bundle.py'),
        ('scripts/package_fixed_ce_five_evidence.py','tools/package_fixed_ce_five_evidence.py'),
        ('scripts/package_fixed_ce_evidence.py','tools/package_fixed_ce_evidence.py')]:
        p=stage/target;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    model_rows=[]
    registry=read(ROOT/'data/asvc_current_method.json')
    for method in ('margin','ce','d3qn','dqfd','qdagger'):
        for seed in range(5) if method in ('margin','ce') else (0,1):
            if method=='margin':directory=Path(normal(registry['final_models'][str(seed)])).parent
            elif method=='ce':directory=Path('outputs')/('asvc_fixed_ce_v1' if seed<2 else 'asvc_fixed_ce_five_seeds_v1')/f'seed_{seed}'
            elif method=='qdagger':directory=Path('outputs')/QD/f'seed_{seed}'
            else:directory=Path(normal(next(r['directory'] for r in qf['checks']['baseline_audit'] if r['method']==method and r['seed']==seed)))
            model=(directory/'model_final.pth').as_posix();copy(model)
            model_rows.append(dict(method=method,seed=seed,model_path=model,model_sha256=models[model],
                model_bytes=(ROOT/model).stat().st_size,model_included=False,new_training_this_round=(method=='qdagger'),
                online_environment_steps=200000,offline_updates=20000 if method in ('qdagger','dqfd') else 0))
    for name in models:
        if not any(r['path']==name for r in excluded):excluded.append(dict(path=name,bytes=local(ROOT,name).stat().st_size,sha256=models[name],included=False))
    write_csv(stage/'model_index.csv',model_rows);write_csv(stage/'excluded_model_artifacts.csv',sorted(excluded,key=lambda r:r['path']))
    (stage/'README.md').write_text(README,encoding='utf-8');(stage/'.gitattributes').write_text('** -text\n',encoding='utf-8')
    relative=final.relative_to(ROOT).as_posix()
    scope=sorted(set(CODE+ATTR+['docs/GITHUB_SYNC.zh-CN.md',f'data/{QD}',f'data/{CE}',relative,
        f'results/analysis_upload/{NAME}/LATEST.md',f'results/analysis_upload/{NAME}/latest.json']))
    write(stage/'upload_scope.json',scope)
    write(stage/'manifest.json',dict(name=NAME,created=datetime.now().isoformat(),
        files_sha256={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob('*')) if p.is_file()},
        source_files_sha256=sources,source_to_archive=destinations,models_sha256=models,
        previous_ce_five_manifest_sha256=old_pointer['manifest_sha256'],new_training_runs=2,reused_training_runs=14,
        new_online_environment_steps=400000,new_offline_updates=40000,new_online_updates=399002,
        unique_training_curves=16,unique_validation_checkpoints=336,unique_validation_rows=16800,
        qdagger_before_pretraining_episodes=100,new_confirmation_episodes=2500,reused_confirmation_episodes=2500,
        qdagger_final500_used=False,weights_included=False,multitask_used=False,policy_rollouts_in_packaging=0))
    verify(stage,True);BASE.mkdir(parents=True,exist_ok=True);stage.rename(final)
    write(pointer,dict(package=relative,manifest_sha256=sha(final/'manifest.json')))
    (BASE/'LATEST.md').write_text('# 固定任务补实验：最新上传资料\n\n'
        f'[开始阅读]({dirname}/README.md) · [总报告]({dirname}/results/{NAME}/REPORT.zh-CN.md) · '
        f'[QDagger曲线]({dirname}/results/{QD}/learning_curves.png)\n\n'
        'CE五seed500确认：98.32%±1.48%，大间隔99.28%±0.73%；QDagger两seed最终50验证86%/94%，全程AULC均值75.90%。'
        '两类结果分开。两次新训练、14次历史训练复用，未混入多任务研究。\n',encoding='utf-8')
    if archive:
        archive_path=BASE.parent/('fixed_paper_supplements_'+dirname+'.zip')
        with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
            for path in sorted(final.rglob('*')):
                if path.is_file():zipped.write(path,dirname+'/'+path.relative_to(final).as_posix())
        print('ZIP:',archive_path.relative_to(ROOT),archive_path.stat().st_size,'bytes')
    print('Package:',relative);return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    parser.add_argument('--check-sources',action='store_true');parser.add_argument('--package',type=Path)
    parser.add_argument('--new-revision',action='store_true');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.verify:
        if args.package:package=args.package
        elif (ROOT/'manifest.json').exists():package=ROOT
        else:
            pointer=read(BASE/'latest.json');package=local(ROOT,pointer['package'])
            require(sha(package/'manifest.json')==pointer['manifest_sha256'],'Publication pointer differs')
        verify(package,args.check_sources)
    else:build(args.new_revision,args.archive)
