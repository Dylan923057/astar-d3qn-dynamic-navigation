"""Publish the completed fixed-task five-seed QDagger extension; standard-library verification."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import gzip
import io
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
import package_fixed_paper_supplements as prior

NAME='asvc_fixed_qdagger_five_seeds_v1'
BASE=ROOT/'results/analysis_upload'/NAME
read,write,rows,write_csv,sha,local,normal,require,close=(getattr(prior,n) for n in
    ('read','write','rows','write_csv','sha','local','normal','require','close'))
METHODS=('margin','ce','d3qn','dqfd','qdagger')
CODE=[f'configs/{NAME}.yaml','scripts/fixed_qdagger_five_common.py','scripts/run_fixed_qdagger_five.py',
    'scripts/evaluate_fixed_qdagger_five.py','scripts/analyze_fixed_qdagger_five.py','scripts/start_fixed_qdagger_five.ps1',
    'tests/test_fixed_qdagger_five.py','docs/ASVC_FIXED_QDAGGER_FIVE_SEEDS.zh-CN.md',
    'scripts/package_fixed_qdagger_five_evidence.py','scripts/sync_fixed_qdagger_five_github.ps1',
    'docs/ASVC_FIXED_QDAGGER_FIVE_UPLOAD.zh-CN.md']
ATTR=['.gitignore','configs/.gitattributes','scripts/.gitattributes','docs/.gitattributes','tests/.gitattributes',
    f'data/{NAME}/.gitattributes',f'results/analysis_upload/{NAME}/.gitattributes']


def source_digest(path,source):
    return prior.scientific_sha(path) if str(source).endswith('.json') and path.suffix=='.gz' else sha(path)


def verify(package,check_sources=False):
    package=Path(package).resolve();manifest=read(package/'manifest.json')
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual==set(manifest['files_sha256'])|{'manifest.json'},'Publication file list differs')
    for name,value in manifest['files_sha256'].items():
        path=local(package,name)
        require(path.suffix.lower() in prior.base.ALLOW|{'.npz','.gin'} or path.name in ('.gitattributes','LICENSE'),
            'Unsupported publication artifact: '+name)
        require(path.stat().st_size<50*1024**2 and sha(path)==value,'Oversized/changed publication file: '+name)
        if path.suffix.lower() not in ('.gz','.png','.pdf','.npz'):
            require(not any(re.search(pattern,path.read_text(encoding='utf-8-sig')) for pattern in prior.PATTERNS),
                'Potential credential: '+name)
    require(manifest['new_training_seeds']==[2,3,4] and manifest['reused_qdagger_seeds']==[0,1]
        and manifest['new_training_runs']==3 and manifest['comparison_runs']==25,'Training scope differs')
    require(manifest['new_online_environment_steps']==600000 and manifest['new_offline_updates']==60000
        and manifest['new_online_updates']==598503 and not manifest['weights_included'] and not manifest['multitask_used'],
        'Training budget or experiment boundary differs')
    previous=package/'prior';old=read(previous/'manifest.json')
    require(sha(previous/'manifest.json')==manifest['previous_manifest_sha256'],'Prior publication identity differs')
    with redirect_stdout(io.StringIO()):prior.verify(previous,False)
    for source,destination in manifest['source_to_archive'].items():
        # The historical verifier already checks its original-byte exports, including gzip.
        inherited=old['source_to_archive'].get(source)
        if inherited and destination=='prior/'+inherited:
            require(manifest['source_files_sha256'][source]==old['source_files_sha256'][source],'Inherited source identity differs')
        else:require(source_digest(local(package,destination),source)==manifest['source_files_sha256'][source],
            'Scientific bytes differ: '+source)
    def artifact(source):
        source=normal(source);require(source in manifest['source_to_archive'],'Missing scientific source: '+source)
        return local(package,manifest['source_to_archive'][source])
    def document(source):
        path=artifact(source)
        if path.suffix=='.gz':
            with gzip.open(path,'rt',encoding='utf-8-sig') as stream:return prior.json.load(stream)
        return read(path)
    def csv(source):return rows(artifact(source))
    def identity(source):
        source=normal(source)
        return manifest['source_files_sha256'].get(source,manifest['models_sha256'].get(source))
    frozen=document(f'data/{NAME}/freeze.json')
    for group in ('source_sha256','protected_sha256','input_sha256'):
        for name,value in frozen[group].items():require(identity(name)==value,'Frozen source/input differs: '+name)
    for name,value in frozen['copied_files_sha256'].items():
        require(identity(f'data/{NAME}/'+normal(name))==value,'Frozen copied data differs: '+name)
    config=frozen['config'];parent=document('data/asvc_fixed_qdagger_v1/freeze.json')
    metadata={'experiment','protocol','seeds','maximum_new_training_runs','final500_evaluation','qdagger_five_extension'}
    require({k:v for k,v in config.items() if k not in metadata}=={k:v for k,v in parent['config'].items() if k not in metadata},
        'An algorithm setting changed between the two- and five-seed runs')
    require(frozen['checks']['passed'] and frozen['checks']['tests']==5 and
        all(r['passed'] for r in frozen['checks']['recovery']),'Extension preflight/recovery evidence missing')
    scene50=document(f'data/{NAME}/frozen50.json')['scenarios'];scene500=document(f'data/{NAME}/frozen500.json')['scenarios']
    require(len(scene50)==50 and len(scene500)==500 and not {s['scenario_id'] for s in scene50}&{s['scenario_id'] for s in scene500},
        '50 validation / 500 confirmation split differs')
    ce_freeze=document('data/asvc_fixed_ce_final500_v1/freeze.json')
    require(frozen['final500_scene_sha256']==ce_freeze['scene_sha256'] and
        [s['scenario_id'] for s in scene500]==ce_freeze['scene_order'],'Confirmation scene identity/order differs')
    analysis=f'results/{NAME}'
    require(document(analysis+'/queue_status.json')['complete'] and document(analysis+'/analysis_status.json')['complete'],
        'Queue or analysis is incomplete')
    audit=document(analysis+'/completion_audit.json');require(audit['passed'] and audit['final500_episodes']==2500,'Final audit missing')
    reported=csv(analysis+'/per_seed_metrics.csv');require(len(reported)==25,'Expected five methods x five seeds')
    run_index={(r['method'],int(r['seed'])):r for r in rows(package/'model_index.csv')}
    require(set(run_index)=={(m,s) for m in METHODS for s in range(5)},'Twenty-five final model identities required')
    metrics={};curves={};failure50=0;validation_rows=0
    for method in METHODS:
        for seed in range(5):
            item=run_index[method,seed];source=item['directory'];directory=artifact(source+'/validation_curve.csv').parent
            values,failed,curve=prior.curve_check(directory,scene50,qdagger=(method=='qdagger'))
            failure50+=failed;validation_rows+=1050;curves[method,seed]=curve;metrics[method,seed]=values
            result=document(source+'/result.json');require(document(source+'/completion.json')['complete'],'Incomplete reused/new run')
            require(result['environment_steps']==200000 and identity(item['model_path'])==item['model_sha256'],
                'Final model or budget differs')
            row=next(r for r in reported if r['method']==method and int(r['seed'])==seed)
            for key,value in values.items():close(row[key],value,'Five-seed '+key)
            if method!='qdagger':continue
            run=document(source+'/run_manifest.json')
            require(result['offline_updates']==20000 and result['online_updates']==199501 and result['total_updates']==219501,
                'QDagger update budget differs')
            require(result['teacher_queries']==119999 and result['student_replay_only'] and result['teacher_exit']==120000
                and not result['teacher_advice'] and not result['td_clip'] and result['supervision_floor']==0,'QDagger mechanisms differ')
            registration='data/asvc_fixed_qdagger_v1/freeze.json' if seed<2 else f'data/{NAME}/freeze.json'
            require(result['freeze_sha256']==identity(registration) and run['freeze_sha256']==result['freeze_sha256'],
                'QDagger training registration differs')
            require({k:v for k,v in run['config'].items() if k not in metadata}=={k:v for k,v in config.items() if k not in metadata},
                'Per-seed algorithm configuration differs')
            require(result['final_model_sha256']==item['model_sha256'],'QDagger model SHA differs')
            logs=csv(source+'/update_diagnostics.csv')
            for phase,updates in (('offline',20000),('online',199501)):
                require(sum(int(r['count']) for r in logs if r['phase']==phase)==updates,'Update diagnostics count differs')
            for log in logs:
                require(all(math.isfinite(float(v)) for k,v in log.items() if k!='phase' and v!=''),'Nonfinite update diagnostic')
                if log['phase']=='offline':close(log['weight_mean'],3,'Offline distillation weight')
                elif int(log['bin_index'])>=120:close(log['weight_max'],0,'Teacher exit')
            training=csv(source+'/training.csv')
            require(int(training[-1]['environment_steps_total'])==200000 and int(training[-1]['online_updates_total'])==199501,
                'Online training log budget differs')
            require(all(int(r['demo_replay_samples'])==0 for r in training),'Permanent online demonstration mixing found')
            before=document(source+'/before_pretraining50.json');prior.scene_order(before['details'],scene50)
            failure50+=prior.failures_match(before['failures'],before['details'])
            require(before['failures']==document(source+'/validation_failures_before_pretraining_000000.json'),
                'Pretraining failure exports differ')
            for sample in csv(source+'/n_step_samples.csv'):
                n=int(sample['n_length']);terminal=prior.flag(sample['n_terminated'])
                require(n in (1,2,3) and prior.flag(sample['bootstrap'])!=terminal,'n-step/terminal boundary differs')
                close(sample['bootstrap_discount'],0 if terminal else .99**n,'Tail bootstrap discount')
                close(sample['distillation_weight'],3 if sample['phase']=='offline' else 3*max(1-int(sample['online_environment_step'])/120000,0),
                    'Fixed distillation schedule')
    require(validation_rows==26250,'Unique repeated-validation count differs')
    for row in csv(analysis+'/mean_sample_sd.csv'):
        values=[metrics[row['method'],s][row['metric']] for s in range(5)]
        close(row['mean'],statistics.mean(values),'Five-seed mean');close(row['sample_sd'],statistics.stdev(values),'Five-seed sample SD')
        close(row['training_seeds'],5,'Training-seed statistical unit')
    for row in csv(analysis+'/paired_per_seed.csv'):
        method=row['method'].removeprefix('qdagger_minus_');seed=int(row['seed'])
        for key in prior.METRICS:close(row[key],metrics['qdagger',seed][key]-metrics[method,seed][key],'Learning paired difference')
    for row in csv(analysis+'/paired_mean_sample_sd.csv'):
        method=row['method'].removeprefix('qdagger_minus_');key=row['metric']
        values=[metrics['qdagger',s][key]-metrics[method,s][key] for s in range(5)]
        close(row['mean'],statistics.mean(values),'Learning paired mean');close(row['sample_sd'],statistics.stdev(values),'Learning paired SD')
    points=csv(analysis+'/all_validation_points.csv');require(len(points)==525,'Expected 525 actual validation checkpoints')
    for row in points:
        point=next(r for r in curves[row['method'],int(row['seed'])] if r['environment_steps_total']==row['environment_steps_total'])
        for key in ('safe_success_rate','collision_rate','timeout_rate'):close(row[key],point[key],'Merged curve point')
    reported500=csv(analysis+'/final500_per_seed.csv');require(len(reported500)==15,'Expected three final500 groups x five seeds')
    final_values={};details_lookup={};failure500=0
    for method in ('margin','ce','qdagger'):
        for seed in range(5):
            source=f'outputs/{NAME}/final500/seed_{seed}' if method=='qdagger' else next(r['directory'] for r in frozen['baseline500'] if r['method']==method and r['seed']==seed)
            require(document(source+'/completion.json')['complete'],'Final500 not complete')
            reg=document(source+'/registration.json');item=run_index[method,seed]
            for key,value in dict(seed=seed,model_sha256=item['model_sha256'],scene_sha256=frozen['final500_scene_sha256'],
                scene_count=500,checkpoint_steps=200000,epsilon=0,teacher=False,static_mask=True,dynamic_shield=False,device='cpu').items():
                require(reg[key]==value,'Final500 policy protocol differs: '+key)
            if method=='qdagger':
                require(reg['updates']==0 and not reg['used_for_tuning'] and reg['max_steps']==300,'Final500 isolation differs')
                require(reg['extension_freeze_sha256']==identity(f'data/{NAME}/freeze.json'),'Final500 extension identity differs')
                require(reg['training_freeze_sha256']==document(item['directory']+'/result.json')['freeze_sha256'],
                    'Final500 model training identity differs')
            details=csv(source+'/scene_details.csv');prior.scene_order(details,scene500)
            failures=source+'/failure_trajectories.json' if method!='margin' else f'outputs/asvc_persistent_supervision_five_seeds_v1/final500/seed_{seed}/failure_trajectories.json'
            failure500+=prior.failures_match(document(failures),details)
            values=prior.final_metrics(details);final_values[method,seed]=values
            details_lookup[method,seed]={r['scenario_id']:r for r in details}
            row=next(r for r in reported500 if r['method']==method and int(r['seed'])==seed)
            for key,value in values.items():close(row[key],value,'Final500 '+key)
    for row in csv(analysis+'/final500_mean_sample_sd.csv'):
        values=[final_values[row['method'],s][row['metric']] for s in range(5)]
        close(row['mean'],statistics.mean(values),'Final500 mean');close(row['sample_sd'],statistics.stdev(values),'Final500 sample SD')
    for row in csv(analysis+'/final500_paired_per_seed.csv'):
        seed=int(row['seed']);method=row['method'].removeprefix('qdagger_minus_')
        for key in prior.FINAL_KEYS:close(row[key],final_values['qdagger',seed][key]-final_values[method,seed][key],'Final500 paired seed')
    for row in csv(analysis+'/final500_paired_mean_sample_sd.csv'):
        method=row['method'].removeprefix('qdagger_minus_');key=row['metric']
        values=[final_values['qdagger',s][key]-final_values[method,s][key] for s in range(5)]
        close(row['mean'],statistics.mean(values),'Final500 paired mean');close(row['sample_sd'],statistics.stdev(values),'Final500 paired SD')
    combined=csv(analysis+'/final500_scene_details.csv');require(len(combined)==7500,'Final500 combined count differs')
    for row in combined:
        source=details_lookup[row['method'],int(row['seed'])][row['scenario_id']]
        for key in source:require(row[key]==source[key],'Final500 combined scientific row differs')
    scene_pairs=csv(analysis+'/final500_paired_scenes.csv');require(len(scene_pairs)==5000,'Final500 paired-scenario count differs')
    for row in scene_pairs:
        method=row['contrast'].removeprefix('qdagger_minus_');seed=int(row['seed']);sid=row['scenario_id']
        a=details_lookup[method,seed][sid];b=details_lookup['qdagger',seed][sid]
        close(row['delta_safe_success'],int(prior.flag(b['safe_success']))-int(prior.flag(a['safe_success'])),'Paired scene success')
        close(row['delta_collision'],int(prior.flag(b['dynamic_collision']))-int(prior.flag(a['dynamic_collision'])),'Paired scene collision')
        close(row['delta_timeout'],int(b['termination_reason']=='timeout')-int(a['termination_reason']=='timeout'),'Paired scene timeout')
        for target,key in (('delta_steps','steps'),('delta_wait_steps','wait_steps')):close(row[target],float(b[key])-float(a[key]),'Paired scene path')
        if a['path_efficiency'] and b['path_efficiency']:close(row['delta_success_path_efficiency'],float(b['path_efficiency'])-float(a['path_efficiency']),'Paired efficiency')
    if check_sources:
        for source,value in (manifest['source_files_sha256']|manifest['models_sha256']).items():
            require(sha(local(ROOT,source))==value,'Local source/model differs: '+source)
    size=sum(local(package,n).stat().st_size for n in actual)
    print(f'PASS QDAGGER FIVE: {len(actual)} files / {size/1024**2:.2f} MiB; 25 final model identities, no weights.')
    print(f'25 curves / 525 checkpoints / 26250 validation rows; 7500 final500 rows and {failure500} final500 failures checked.')
    print(f'Original bytes, all failure exports ({failure50} validation/pretraining failures), AULC, sample SD and paired differences verified.')
    print('This extension: seeds2/3/4, 600000 online steps + 60000 offline updates; original seeds0/1 preserved. No training/evaluation executed.')
    return manifest


README='''# 固定任务QDagger五seed与最终500确认：完整证据包

全部于2026-10-31 00:59完成。本轮仅新增QDagger seed2、3、4，原seed0、1及其他方法复用。原冻结、模型、原始日志和旧阅读包不覆盖；不混入50万步多任务研究。

## 结果与阅读顺序

|同一500场景，5训练seed|安全成功均值±样本SD|动态碰撞|超时|
|---|---:|---:|---:|
|持续弱监督大间隔|99.28%±0.73%|0.72%±0.73%|0|
|CE|98.32%±1.48%|1.36%±1.34%|0.32%±0.72%|
|QDagger任务适配|94.04%±3.92%|1.28%±0.58%|4.68%±4.03%|

QDagger五个seed成功率88.0%、92.8%、98.2%、94.8%、96.4%，2351/2500安全成功、32次动态碰撞、117次超时。相对大间隔安全成功配对差−5.24±3.43个百分点，相对CE−4.28±4.37个百分点。主要失败是超时；不能仅据这些成绩确定循环/等待的训练原因。

|原50重复验证，5训练seed|全程AULC均值±样本SD|后10万步AULC|
|---|---:|---:|
|持续弱监督大间隔|95.63%±0.38%|99.54%±0.34%|
|CE|94.03%±0.95%|98.14%±0.93%|
|DQfD|93.24%±0.42%|97.11%±0.69%|
|QDagger任务适配|78.92%±4.43%|85.06%±6.53%|
|原始D3QN|29.08%±17.45%|55.57%±32.21%|

本固定任务与适配配置下，大间隔学习过程和最终表现较好；不是单项机制因果结论，也不宣称正式统计显著性或原Atari完整复现。

1. [完整报告](results/asvc_fixed_qdagger_five_seeds_v1/REPORT.zh-CN.md)、[五seed均值曲线](results/asvc_fixed_qdagger_five_seeds_v1/learning_curves.png)、[逐seed曲线](results/asvc_fixed_qdagger_five_seeds_v1/individual_seeds.png)，同目录保留PDF。
2. [25个逐seed指标](results/asvc_fixed_qdagger_five_seeds_v1/per_seed_metrics.csv)、[均值/样本SD](results/asvc_fixed_qdagger_five_seeds_v1/mean_sample_sd.csv)、[学习配对差](results/asvc_fixed_qdagger_five_seeds_v1/paired_per_seed.csv)。
3. [500确认逐seed](results/asvc_fixed_qdagger_five_seeds_v1/final500_per_seed.csv)、[500均值/样本SD](results/asvc_fixed_qdagger_five_seeds_v1/final500_mean_sample_sd.csv)、[500配对差](results/asvc_fixed_qdagger_five_seeds_v1/final500_paired_per_seed.csv)、[7500条原/新逐场景](results/asvc_fixed_qdagger_five_seeds_v1/final500_scene_details.csv)、[5000条逐场景配对差](results/asvc_fixed_qdagger_five_seeds_v1/final500_paired_scenes.csv)。
4. [仅种子/路径/评估元数据变化的扩展协议](docs/ASVC_FIXED_QDAGGER_FIVE_SEEDS.zh-CN.md)、[预检](results/asvc_fixed_qdagger_five_seeds_v1/preflight.json)、[完成/完整恢复审计](results/asvc_fixed_qdagger_five_seeds_v1/completion_audit.json)、[25个最终模型身份](model_index.csv)。
5. data/asvc_fixed_qdagger_five_seeds_v1保留冻结配置、112个源码文件/快照、50/500输入、旧组只读副本与原7260条示范成本。新seed2–4的训练、Q/TD/蒸馏、三步样本及全部失败在outputs/asvc_fixed_qdagger_five_seeds_v1；原seed0/1和旧四组在prior或对应原目录。

## 口径与复用

25条训练曲线=5方法×5seed，525个实际验证点、26250条反复使用同50场景的记录。另有QDagger预训练前250条单列；7500条500确认=3方法×5模型×500场景，其中本轮新增QDagger2500回合，原大间隔/CE5000回合复用。样本SD以训练seed计算，不把场景数当成独立训练次数；50验证与500确认分开。

本轮新增预算为600000在线环境步、60000离线更新、598503在线更新。五个QDagger累计1000000在线步、100000离线更新、997505在线更新。每seed严格使用200000步最终模型，不选最佳中间模型。原7260示范（107/128成功回合，生成10756环境步）不增加；目标裁剪/代选/持续监督下限不引入QDagger，λ在120000步退出，T=0.1。

扩展是在看到原两seed成绩后由用户授权，未据500调参、筛seed或改预算。原两seed和所有旧结果保留。prior按原字节保留上一轮资料，其中“两seed”及“QDagger不跑500”的文字是旧轮边界；本轮以当前五seed协议和报告为准。

## 压缩、完整性与复现

大体积失败、批次缓存和预训练前JSON按原字节无损gzip，不抽样删除失败；manifest记录原始SHA和压缩SHA。权重、完整回放/优化器断点及恢复测试二进制留在原工作区，excluded_model_artifacts.csv登记路径/大小/SHA。只加载权重不等于完整续训，缺少原模型无法重跑网络评估。

只下载本目录，进入目录后即可标准库校验，无PyTorch/GPU/权重：

```powershell
python tools/verify_bundle.py --verify --package .
```

完整仓库根目录：

```powershell
python scripts/package_fixed_qdagger_five_evidence.py --verify
# 原工作区仍保存全部原文件、模型和断点时：
python scripts/package_fixed_qdagger_five_evidence.py --verify --check-sources
```

校验从原始逐场景数据复算成功/碰撞/超时、实际步数AULC、训练seed均值/样本SD、配对差，核对所有失败长度和结局、冻结/预算/教师退出及恢复审计；不训练、不运行策略、不联网。来源映射记录复用的历史文件位置，根目录提供完整当前源码与新实验数据。

GitHub仅提交upload_scope.json列出的本批内容，模型/回放/缓存、根README及其他研究改动不混入。见[限定上传命令](docs/ASVC_FIXED_QDAGGER_FIVE_UPLOAD.zh-CN.md)。
'''


def build(new_revision=False,archive=False):
    pointer=BASE/'latest.json'
    if pointer.exists() and not new_revision:
        info=read(pointer);package=local(ROOT,info['package']);verify(package,True);return package
    import fixed_qdagger_five_common as f
    f.verify();require(read(f.RESULTS/'queue_status.json')['complete'] and read(f.RESULTS/'completion_audit.json')['passed'],
        'Wait for completed training/evaluation/audit')
    old_pointer=read(prior.BASE/'latest.json');old_dir=local(ROOT,old_pointer['package'])
    require(sha(old_dir/'manifest.json')==old_pointer['manifest_sha256'],'Prior publication pointer differs')
    with redirect_stdout(io.StringIO()):prior.verify(old_dir,False)
    old=read(old_dir/'manifest.json');stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');dirname='review_'+stamp
    stage=f.RESULTS/'publication_staging'/dirname;stage.mkdir(parents=True,exist_ok=False);final=BASE/dirname
    sources={};destinations={};models=old['models_sha256'].copy();metadata_changes=[]
    for path in sorted(old_dir.rglob('*')):
        if not path.is_file():continue
        destination='prior/'+path.relative_to(old_dir).as_posix();target=stage/destination
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        name=path.relative_to(ROOT).as_posix();sources[name]=sha(path);destinations[name]=destination
    for source,destination in old['source_to_archive'].items():
        path=local(ROOT,source);expected=old['source_files_sha256'][source]
        if sha(path)!=expected:
            require(path.name in ('.gitattributes','.gitignore'),'Historical scientific source changed: '+source)
            metadata_changes.append(dict(path=source,historical_sha256=expected,current_sha256=sha(path)))
        else:sources[source]=expected;destinations[source]='prior/'+destination
    # The previous supplement kept only some margin/CE files at its root,
    # while their complete runs (including all failures) live in its CE bundle.
    # Point each whole historical run at that complete, byte-preserved directory.
    legacy=read(old_dir/'previous_ce_five/manifest.json')
    legacy_directories=tuple(r['directory']+'/' for r in legacy['runs'])
    for source,destination in legacy['source_to_archive'].items():
        if source.startswith(legacy_directories):
            expected=legacy['source_files_sha256'][source]
            require(sha(local(ROOT,source))==expected,'Historical complete run changed: '+source)
            sources[source]=expected;destinations[source]='prior/previous_ce_five/'+destination
    def copy(name,force_root=False):
        name=normal(name);path=local(ROOT,name)
        if not path.is_file() or '__pycache__' in path.parts:return
        if path.suffix in ('.pth','.pt'):
            value=sha(path)
            require(name not in models or models[name]==value,'Protected model changed: '+name);models[name]=value;return
        value=sha(path)
        if name in sources and not force_root:
            require(sources[name]==value,'Previously copied scientific source differs: '+name);return
        destination='provenance/root_gitignore.txt' if name=='.gitignore' else name
        compress=path.suffix=='.json' and (path.name.startswith(('validation_failures_','batch_')) or
            path.name in ('failure_trajectories.json','before_pretraining50.json'))
        if compress:destination+='.gz'
        target=local(stage,destination);target.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with path.open('rb') as stream,target.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0,compresslevel=6) as zipped:
                shutil.copyfileobj(stream,zipped)
        else:shutil.copyfile(path,target)
        sources[name]=value;destinations[name]=destination
    frozen=read(f.DATA/'freeze.json')
    for name in frozen['source_sha256']:copy(name,True)
    for group in ('protected_sha256','input_sha256'):
        for name in frozen[group]:copy(name)
    for folder in (f.DATA,f.OUT):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in prior.base.ALLOW|{'.pth','.pt','.npz','.gin'} or path.is_file() and path.name in ('.gitattributes','LICENSE'):
                copy(path.relative_to(ROOT).as_posix(),force_root=(folder==f.DATA))
    for path in sorted(f.RESULTS.rglob('*')):
        if path.is_file() and 'publication_staging' not in path.parts and path.suffix.lower() in prior.base.ALLOW|{'.pth','.pt'}:
            copy(path.relative_to(ROOT).as_posix())
    for method in ('d3qn','dqfd'):
        for seed in (2,3,4):
            for path in sorted(f.baseline_directory(method,seed).iterdir()):
                if path.is_file() and path.name not in ('static_final.json','static_final_trace.json'):copy(path.relative_to(ROOT).as_posix())
    # Direct current source/config/input context is available at the usual root paths.
    for name in prior.base.dependency_files(CODE+list(frozen['source_sha256'])):copy(name,True)
    for path in sorted(f.original.DATA.rglob('*')):
        if path.is_file():copy(path.relative_to(ROOT).as_posix(),True)
    old_training=read(f.original.DATA/'freeze.json')
    for name in old_training['input_sha256']:copy(name,True)
    for name in CODE+ATTR+['pyproject.toml']:copy(name,True)
    for source,target in [('scripts/package_fixed_qdagger_five_evidence.py','tools/verify_bundle.py'),
        ('scripts/package_fixed_paper_supplements.py','tools/package_fixed_paper_supplements.py'),
        ('scripts/package_fixed_ce_five_evidence.py','tools/package_fixed_ce_five_evidence.py'),
        ('scripts/package_fixed_ce_evidence.py','tools/package_fixed_ce_evidence.py')]:
        path=stage/target;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,path)
    index=[]
    for method in METHODS:
        for seed in range(5):
            directory=f.baseline_directory(method,seed);model=directory/'model_final.pth';copy(model.relative_to(ROOT).as_posix())
            index.append(dict(method=method,seed=seed,directory=directory.relative_to(ROOT).as_posix(),
                model_path=model.relative_to(ROOT).as_posix(),model_sha256=sha(model),model_bytes=model.stat().st_size,
                included=False,new_training_this_extension=(method=='qdagger' and seed>=2),online_environment_steps=200000,
                offline_updates=20000 if method in ('qdagger','dqfd') else 0))
    write_csv(stage/'model_index.csv',index)
    write_csv(stage/'excluded_model_artifacts.csv',[dict(path=name,bytes=local(ROOT,name).stat().st_size,sha256=value,included=False)
        for name,value in sorted(models.items())])
    write(stage/'provenance/metadata_changes_since_prior.json',metadata_changes)
    (stage/'README.md').write_text(README,encoding='utf-8');(stage/'.gitattributes').write_text('** -text\n',encoding='utf-8')
    relative=final.relative_to(ROOT).as_posix()
    scope=sorted(set(CODE+ATTR+[f'data/{NAME}','docs/GITHUB_SYNC.zh-CN.md',relative,
        f'results/analysis_upload/{NAME}/latest.json',f'results/analysis_upload/{NAME}/LATEST.md']))
    write(stage/'upload_scope.json',scope)
    write(stage/'manifest.json',dict(name=NAME,created=datetime.now().isoformat(),
        files_sha256={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob('*')) if p.is_file()},
        source_files_sha256=sources,source_to_archive=destinations,models_sha256=models,
        previous_manifest_sha256=old_pointer['manifest_sha256'],new_training_runs=3,new_training_seeds=[2,3,4],
        reused_qdagger_seeds=[0,1],comparison_runs=25,reused_comparison_runs=22,
        new_online_environment_steps=600000,new_offline_updates=60000,new_online_updates=598503,
        unique_validation_rows=26250,unique_validation_checkpoints=525,qdagger_before_pretraining_rows=250,
        new_final500_episodes=2500,reused_final500_episodes=5000,weights_included=False,multitask_used=False,
        policy_rollouts_in_packaging=0))
    verify(stage,True);BASE.mkdir(parents=True,exist_ok=True);stage.rename(final)
    write(pointer,dict(package=relative,manifest_sha256=sha(final/'manifest.json')))
    (BASE/'LATEST.md').write_text('# 固定QDagger五seed：最新完整上传包\n\n'
        f'[开始阅读]({dirname}/README.md) · [结果报告]({dirname}/results/{NAME}/REPORT.zh-CN.md) · '
        f'[学习曲线]({dirname}/results/{NAME}/learning_curves.png)\n\n'
        '五seed及最终2500回合均完成。500安全成功：QDagger94.04%±3.92%，大间隔99.28%±0.73%，CE98.32%±1.48%。'
        '新增seed2/3/4、保留seed0/1。25条曲线与三方法500确认分开，未混入多任务。\n',encoding='utf-8')
    if archive:
        archive_path=BASE.parent/('fixed_qdagger_five_'+dirname+'.zip')
        with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
            for p in sorted(final.rglob('*')):
                if p.is_file():zipped.write(p,dirname+'/'+p.relative_to(final).as_posix())
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
