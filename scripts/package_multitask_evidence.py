"""Build/verify completed four-run evidence using only the Python standard library.

This publication utility is deliberately outside the frozen '*multitask_pilot*.py'
training source set. It never trains, evaluates a policy, stages or pushes files.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]
NAME='asvc_multitask_risk_pilot_v1'
REG=ROOT/'data'/NAME
LIVE=ROOT/'results'/NAME
OUT=ROOT/'outputs'/NAME
BASE=ROOT/'results/analysis_upload'/NAME
RUNS=(('C',0),('D',0),('C',1),('D',1))
ALLOW={'.json','.csv','.md','.py','.yaml','.png','.ps1','.txt','.log'}
SUMMARY_FILES=('REPORT.json','run_results.csv','validation_curves_all_seeds.csv',
               'paired_D_minus_C_curves.csv','supervision_by_risk.csv')
FIGURES=('individual_learning_curves.png','length_band_learning_curves.png','paired_differences.png')
RUN_FILES=('run_manifest.json','completion.json','result.json','status.json','supervision_statistics.json',
           'training_episodes.csv','update_log.csv','validation_curve.csv','validation_details.csv','offline_cases.csv')
METRICS=('last_success','last_collision','last_timeout','early_aulc','full_aulc','late_aulc','late_min_success','final_window_success_sd')


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    values=list(values)
    require(bool(values),'Cannot publish an empty result table')
    fields=list(dict.fromkeys(k for row in values for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def local(name,base=ROOT):
    base=Path(base).resolve();path=(base/name).resolve()
    require(path.is_relative_to(base),'Path escapes its declared directory: '+str(name))
    return path


def close(a,b,context):
    require(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),'Statistic differs: '+str(context))


def aulc(curve,start,end):
    xs=[int(r['environment_steps_total']) for r in curve];ys=[float(r['success_rate']) for r in curve]
    require(xs[0]<=start<end<=xs[-1],'No curve extrapolation')
    def at(t):
        index=bisect_left(xs,t)
        if xs[index]==t:return ys[index]
        return ys[index-1]+(ys[index]-ys[index-1])*(t-xs[index-1])/(xs[index]-xs[index-1])
    points=[start]+[x for x in xs if start<x<end]+[end]
    return sum((b-a)*(at(a)+at(b))*.5 for a,b in zip(points,points[1:]))/(end-start)


def verify(package,check_sources=False):
    package=Path(package).resolve();manifest=read(package/'manifest.json')
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual==set(manifest['files_sha256'])|{'manifest.json'},'Bundle file list changed')
    for name,digest in manifest['files_sha256'].items():
        path=local(name,package)
        require(path.suffix.lower() in ALLOW and path.stat().st_size<50*1024**2,'Unexpected/oversized package file: '+name)
        require(sha(path)==digest,'Bundle SHA differs: '+name)
    require(manifest['complete'] and manifest['training_runs']==4 and manifest['environment_steps_total']==2000000,'Wrong pilot scope')
    require(not manifest['weights_included'] and manifest['test_pool_evaluations']==manifest['formal_ablation_runs']==0,'Unexpected model/test/formal scope')
    freeze=read(package/'registration/frozen_protocol.json')
    require(freeze['steps_per_run']==500000 and freeze['expected_updates']==499501,'Budget changed')
    for name,digest in freeze['source_sha256'].items():
        require(sha(local(name,package/'source_snapshot'))==digest,'Frozen training source differs: '+name)
    input_paths=dict(task_pools='inputs/task_pools.json',validation_scenes='inputs/validation_scenes.json',
                     training_sequence_preview='inputs/training_sequence_preview.json',legacy_protection='registration/legacy_protection.json')
    for name,digest in freeze['input_sha256'].items():
        target=input_paths[Path(name).stem]
        require(sha(package/target)==digest,'Frozen input differs: '+name)
    pools=read(package/'inputs/task_pools.json');validation=read(package/'inputs/validation_scenes.json')
    require({k:len(v) for k,v in pools['pools'].items()}=={'train':240,'validation':50,'unseen_combination':60,'unseen_goal':60},'Task pool sizes changed')
    require(len(validation)==50 and len({s['task_id'] for s in validation})==50,'Validation set differs')
    held={tuple(g) for g in pools['held_out_goals']};require(len(held)==10,'Expected 20% held-out candidate goals')
    observed=set()
    for split,tasks in pools['pools'].items():
        pairs={(tuple(t['start']),tuple(t['goal'])) for t in tasks}
        require(len(pairs)==len(tasks) and not observed.intersection(pairs),'Task pairs overlap across pools')
        observed.update(pairs)
        if split!='unseen_goal':require(all(tuple(t['goal']) not in held for t in tasks),'Held-out goal used in training/validation')
    val_ids=[s['task_id'] for s in validation]
    val_seeds=[int(s['scene_seed']) for s in validation]
    val_tasks={t['task_id']:t for t in pools['pools']['validation']}
    report=read(package/'summary/REPORT.json')
    require(report['complete'] and report['completed_runs']==4 and report['test_pool_evaluations']==report['formal_ablation_runs']==0,'Incomplete report')
    run_lookup={(r['group'],int(r['seed'])):r for r in report['runs']}
    require(set(run_lookup)==set(RUNS),'Report run list differs')
    model_index=rows(package/'model_index.csv')
    require(len(model_index)==4,'Final model index incomplete')
    curves={};validation_rows=0
    for group,seed in RUNS:
        path=package/'run_logs'/group/f'seed_{seed}'
        run,done,result=(read(path/name) for name in ('run_manifest.json','completion.json','result.json'))
        require(done['complete'] and result['complete'] and result['step']==500000 and result['updates']==499501,'Run budget/completion differs')
        require(run['identity']==result['identity']=={'group':group,'seed':seed,'freeze_sha256':sha(package/'registration/frozen_protocol.json')},'Run identity mismatch')
        require(run['config']==freeze['config'],'Group configuration differs from frozen common settings')
        require(result['test_pools_evaluated'] is False,'Test pool evaluated')
        indexed=next(r for r in model_index if r['group']==group and int(r['seed'])==seed)
        require(indexed['model_sha256']==result['final_model_sha256'],'Final model SHA differs')
        curve=rows(path/'validation_curve.csv');details=rows(path/'validation_details.csv');curves[group,seed]=curve
        require([int(r['environment_steps_total']) for r in curve]==list(range(0,500001,10000)),'Expected all 51 fixed validation checkpoints')
        require(len(details)==2550,'Expected 51 x 50 validation details per run')
        validation_rows+=len(details)
        for point in curve:
            step=int(point['environment_steps_total'])
            selected=[r for r in details if int(r['environment_steps_total'])==step]
            require([r['task_id'] for r in selected]==val_ids,'Validation task order changed')
            require([int(r['scene_seed']) for r in selected]==val_seeds,'Validation scene seed changed')
            require(all(val_tasks[r['task_id']]['band']==r['band'] and int(r['length'])==val_tasks[r['task_id']]['length'] for r in selected),'Task length/stratum differs')
            for metric in ('success','collision','timeout'):
                close(sum(int(r[metric]) for r in selected)/50,point[metric+'_rate'],(group,seed,step,metric))
                for band in ('short','medium','long'):
                    chosen=[r for r in selected if r['band']==band]
                    close(sum(int(r[metric]) for r in chosen)/len(chosen),point[f'{band}_{metric}_rate'],(group,seed,step,band,metric))
            require(all(sum(int(r[k]) for k in ('success','collision','timeout'))==1 for r in selected),'Validation outcome categories differ')
        summary=run_lookup[group,seed]
        for key,start,end in (('early_aulc',0,100000),('full_aulc',0,500000),('late_aulc',250000,500000)):
            close(aulc(curve,start,end),summary[key],(group,seed,key))
        close(min(float(r['success_rate']) for r in curve if int(r['environment_steps_total'])>=250000),summary['late_min_success'],'Late minimum')
        for metric in ('success','collision','timeout'):
            close(curve[-1][metric+'_rate'],summary['last_'+metric],'Final '+metric)
        stats=read(path/'supervision_statistics.json');rs=stats['replay_actual_statistics']
        require(rs['updates']==499501 and rs['sampled_draws']==499501*64,'Actual replay update counts differ')
        close(rs['eligible_draws']/rs['sampled_draws'],stats['replay_supervision_coverage'],'Replay coverage')
        require(sum(v['draws'] for v in stats['risk_bins'].values())==rs['eligible_draws'],'Eligible risk-bin counts differ')
        require(0<=rs['effective_coefficient_sum']<=rs['base_coefficient_sum']+1e-6,'Supervision coefficient outside baseline')
        if group=='C':close(rs['effective_coefficient_sum'],rs['base_coefficient_sum'],'C supervision strength')
    require(validation_rows==10200,'Validation detail total differs')
    for row in rows(package/'summary/group_means_sample_sd.csv'):
        selected=[r for r in report['runs'] if r['group']==row['group']]
        require(len(selected)==int(row['training_seed_count'])==2,'Training-seed unit changed')
        for key in METRICS:
            close(statistics.mean(r[key] for r in selected),row[key+'_mean'],('mean',key))
            close(statistics.stdev(r[key] for r in selected),row[key+'_sample_sd'],('sample SD',key))
    for row in rows(package/'summary/paired_D_minus_C_per_seed.csv'):
        seed=int(row['seed'])
        for key in METRICS:close(run_lookup['D',seed][key]-run_lookup['C',seed][key],row[key+'_difference'],('paired',seed,key))
    for row in rows(package/'summary/paired_D_minus_C_curves.csv'):
        seed=int(row['seed']);step=int(row['environment_steps_total'])
        left=next(r for r in curves['C',seed] if int(r['environment_steps_total'])==step)
        right=next(r for r in curves['D',seed] if int(r['environment_steps_total'])==step)
        for key in ('success','collision','timeout'):
            close(float(right[key+'_rate'])-float(left[key+'_rate']),row[key+'_difference'],'Paired curve')
    require(sum(1 for _ in rows(package/'summary/validation_curves_all_seeds.csv'))==204,'All-seed curves incomplete')
    if check_sources:
        for name,digest in (manifest['source_files_sha256']|manifest['models_sha256']).items():
            require(sha(local(name))==digest,'Original local source/model changed: '+name)
        for name,digest in read(package/'registration/legacy_protection.json').items():
            require(sha(local(name))==digest,'Protected legacy file changed: '+name)
    total=sum(local(name,package).stat().st_size for name in manifest['files_sha256'])
    print(f'PASS: {len(manifest["files_sha256"])} SHA-verified files; four 500000-step runs; 10200 validation rows; {total/1024**2:.2f} MiB.',flush=True)
    print('Test-pool evaluations: 0. Formal ablation runs: 0. No weights, replay checkpoints or caches included.',flush=True)
    return manifest


def build():
    pipeline=read(OUT/'pipeline_status.json')
    require(pipeline['complete'] and not pipeline['running'] and not pipeline['failures'],'Four-run pipeline is not complete')
    freeze=read(REG/'frozen_protocol.json')
    stamp=datetime.fromisoformat(pipeline['updated']).strftime('%Y%m%d_%H%M%S')
    destination=BASE/('review_'+stamp)
    if destination.exists():
        verify(destination,check_sources=True)
        return destination
    for name,digest in freeze['source_sha256'].items():require(sha(local(name))==digest,'Frozen source changed: '+name)
    destination.mkdir(parents=True)
    source_hashes={};models={}
    def copy(source,target):
        source=Path(source);target=destination/target
        require(source.is_file() and source.suffix.lower() in ALLOW,'Unsupported/missing publication input: '+str(source))
        require(source.stat().st_size<50*1024**2,'Publication input is oversized')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        source_hashes[relative(source)]=sha(source)
    for name in ('frozen_protocol.json','legacy_protection.json','reserved_formal_configs.json'):
        copy(REG/name,Path('registration')/name)
    copy(OUT/'pipeline_status.json','registration/pipeline_status.json')
    for name in ('task_pools.json','validation_scenes.json','training_sequence_preview.json'):
        copy(REG/name,Path('inputs')/name)
    for name,digest in freeze['source_sha256'].items():
        snapshot=REG/'source_snapshot'/name
        require(sha(snapshot)==digest,'Training snapshot differs: '+name)
        copy(snapshot,Path('source_snapshot')/name)
        source_hashes[name]=digest
    for path in sorted((ROOT/'configs/asvc_multitask_risk_pilot_v1_formal').glob('*.yaml')):
        copy(path,Path('formal_reserved')/path.name)
    for name in SUMMARY_FILES:copy(LIVE/name,Path('summary')/name)
    for name in FIGURES:copy(LIVE/name,Path('figures')/name)
    for name in ('preflight.json','preflight_checks.log','production_cuda_checks.json'):
        copy(LIVE/name,Path('checks')/name)
    copy(LIVE/'RUN_GUIDE.zh-CN.txt','RUN_GUIDE.zh-CN.txt')
    model_rows=[];diagnostics=[]
    for group,seed in RUNS:
        directory=OUT/group/f'seed_{seed}'
        for name in RUN_FILES:copy(directory/name,Path('run_logs')/group/f'seed_{seed}'/name)
        copy(OUT/'logs'/f'{group}_seed_{seed}.log',Path('run_logs')/group/f'seed_{seed}'/'console.log')
        result=read(directory/'result.json');stats=read(directory/'supervision_statistics.json')
        model=directory/'model_final.pth';checkpoint=directory/'checkpoint_latest.pth'
        require(sha(model)==result['final_model_sha256'],'Final model identity differs')
        models[relative(model)]=sha(model);models[relative(checkpoint)]=sha(checkpoint)
        model_rows.append(dict(group=group,seed=seed,steps=result['step'],updates=result['updates'],
            model_path=relative(model),model_sha256=models[relative(model)],
            full_checkpoint_path=relative(checkpoint),checkpoint_sha256=models[relative(checkpoint)],weights_included=False))
        rs=stats['replay_actual_statistics'];advice=stats['advice_statistics'];updates=rows(directory/'update_log.csv')
        diagnostics.append(dict(group=group,seed=seed,replay_supervision_coverage=stats['replay_supervision_coverage'],
            environment_pre_action_gate_availability=stats['environment_label_availability'],
            effective_coefficient_sum=rs['effective_coefficient_sum'],base_coefficient_sum=rs['base_coefficient_sum'],
            weighted_soft_factor=rs['effective_coefficient_sum']/rs['base_coefficient_sum'],
            reduced_eligible_fraction=rs['reduced_draws']/rs['eligible_draws'],
            advice_application_rate=advice['applied']/advice['steps'],
            q_abs_peak=max(float(r['q_abs_max']) for r in updates),
            raw_target_min=min(float(r['raw_target_min']) for r in updates),raw_target_max=max(float(r['raw_target_max']) for r in updates)))
    write_csv(destination/'model_index.csv',model_rows)
    write_csv(destination/'summary/diagnostics_per_seed.csv',diagnostics)
    report=read(destination/'summary/REPORT.json')
    means=[];paired=[]
    for group in ('C','D'):
        selected=[r for r in report['runs'] if r['group']==group]
        means.append(dict(group=group,training_seed_count=len(selected),**{name:value for key in METRICS for name,value in
            ((key+'_mean',statistics.mean(r[key] for r in selected)),(key+'_sample_sd',statistics.stdev(r[key] for r in selected)))}))
    for seed in (0,1):
        left=next(r for r in report['runs'] if r['group']=='C' and r['seed']==seed)
        right=next(r for r in report['runs'] if r['group']=='D' and r['seed']==seed)
        paired.append(dict(seed=seed,**{key+'_difference':right[key]-left[key] for key in METRICS}))
    write_csv(destination/'summary/group_means_sample_sd.csv',means)
    write_csv(destination/'summary/paired_D_minus_C_per_seed.csv',paired)
    extras=['.gitignore','.gitattributes','docs/GITHUB_SYNC.zh-CN.md','docs/ASVC_MULTITASK_PILOT.zh-CN.md',
            'scripts/package_multitask_evidence.py','scripts/sync_multitask_evidence_github.ps1',
            relative(REG),'configs/asvc_multitask_risk_pilot_v1_formal',relative(BASE)]
    scope=sorted(set(freeze['source_sha256'])|set(extras))
    write(destination/'upload_scope.json',scope)
    (destination/'README.md').write_text(README,encoding='utf-8')
    manifest=dict(complete=True,training_runs=4,environment_steps_total=2000000,gradient_updates_total=1998004,
        validation_checkpoints_per_run=51,validation_scenes=50,validation_episode_rows=10200,
        test_pool_evaluations=0,formal_ablation_runs=0,weights_included=False,statistical_unit='training_seed',
        validation_reuses_same_50_scenes=True,source_files_sha256=source_hashes,models_sha256=models,
        files_sha256={p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    write(destination/'manifest.json',manifest)
    verify(destination,check_sources=True)
    write(BASE/'latest.json',dict(package=relative(destination)))
    (BASE/'LATEST.md').write_text('# 多起终点与风险自适应监督预实验\n\n'
        f'[结果和文件导航]({destination.name}/README.md) · [四次成绩]({destination.name}/summary/run_results.csv) · '
        f'[逐 seed 学习曲线]({destination.name}/figures/individual_learning_curves.png) · '
        f'[同 seed 差异]({destination.name}/figures/paired_differences.png)\n\n'
        '四次训练均完成 50 万步。D 的平均碰撞率更低，同时成功率更低、超时更多。仅两个 seed 的验证结果，两个测试池与正式消融未运行。\n',encoding='utf-8')
    print(relative(destination),flush=True)
    return destination


README='''# 同地图多起终点与风险自适应监督预实验

2026-10-26 06:19，四次运行全部完成，每次 500000 环境步、499501 次更新，无失败记录。C 为持续弱监督 ASVC，D 仅对合格监督样本增加四步风险软权重。两个测试池和 A～F 正式消融均未运行。

## 结果

| 方法 | seed | 最终成功率 | 碰撞率 | 超时率 | 全程 AULC |
|---|---:|---:|---:|---:|---:|
| C 持续弱监督 | 0 | 76% | 6% | 18% | 45.88% |
| D 风险自适应 | 0 | 74% | 0% | 26% | 44.22% |
| C 持续弱监督 | 1 | 72% | 20% | 8% | 48.40% |
| D 风险自适应 | 1 | 64% | 6% | 30% | 44.12% |

C/D 两 seed 最终平均成功率为 74%/69%，碰撞率为 13%/3%，超时率为 13%/28%。成功率从初始 0% 提升到 64%–76%，说明多任务学习有进展；四次均未达到预登记描述性门槛（总体至少 80%，每档至少 70%）。

全程 AULC 是按环境步积分后的曲线平均成功率，C/D 为 47.14%/44.17%。D−C 全程差在 seed 0、1 分别为 −1.66、−4.28 个百分点，学习速度没有一致增益。后半程最低成功率为 C 46%、54%，D 48%、46%，稳定性也没有一致改善。

四次中三次在最后 10 万步仍有净提升，50 万步预算的充分性尚未得到支持，延长训练的收益仍未知。D 的累计监督系数约为其未降权基准的 67%；不能把超时增加直接归因于监督减少。

统计单位是训练 seed，只有两个。51 次验证反复使用同一组 50 个场景，10200 条记录不是独立测试任务，也不能视为 10200 次训练重复。未见组合和未见目标测试池未评估，不能宣称未见目标泛化或正式显著性结论。

![四次逐 seed 学习曲线](figures/individual_learning_curves.png)

## 文件导航

- [逐次结果](summary/run_results.csv)、[两 seed 均值及样本标准差](summary/group_means_sample_sd.csv)、[配对差](summary/paired_D_minus_C_per_seed.csv)。
- [原始汇总](summary/REPORT.json)、[全部 204 个验证点](summary/validation_curves_all_seeds.csv)、[逐检查点配对差](summary/paired_D_minus_C_curves.csv)。
- [分风险监督](summary/supervision_by_risk.csv)、[回放/代选/Q 值诊断](summary/diagnostics_per_seed.csv)。
- [长度分档曲线](figures/length_band_learning_curves.png)、[同 seed 差异图](figures/paired_differences.png)。
- `run_logs/C/seed_0` 等四个目录：完成标记、运行身份、2550 条验证明细、训练回合、实际回放更新、离线案例和控制台日志。
- [冻结协议](registration/frozen_protocol.json)、[任务池](inputs/task_pools.json)、[50 个验证场景](inputs/validation_scenes.json)、[训练序列预览](inputs/training_sequence_preview.json)。
- `source_snapshot/` 为训练时配置、源码和测试的冻结字节；`formal_reserved/` 的六组配置仅预留，不可通过本轮入口启动。
- [模型索引](model_index.csv)记录四个最终模型及完整断点的本地路径和 SHA，文件本体保留本地。
- [检查记录](checks/preflight_checks.log)、[实际 CUDA batch64 一致性](checks/production_cuda_checks.json)、[本地启动和续跑说明](RUN_GUIDE.zh-CN.txt)。
- `manifest.json` 包含完整文件 SHA；`upload_scope.json` 限定本批 Git 提交范围。

## 实验边界

使用原静态地图、网络、奖励、gamma、五动作、碰撞规则、Smooth L1 和梯度裁剪。C/D 均使用 TD 目标 [-6,10]、相同硬标签筛查、代选及基础监督日程。D 仅增加 `(1-R)^2`，H=4、gamma_r=0.9、d_crit=0、d_safe=2，风险从局部观测关联估计，不读取隐藏未来。风险不是碰撞概率或安全保证。

每条经验记录自己的任务，BFS 动作场按地图/目标缓存并以位置索引；历史动态通道只保留当时实际可见的观测。训练池 240 对，验证池 50 对，两个测试池各 60 对；候选目标 50 个中 10 个用于未见目标测试。短/中/长任务等比例，超时 max(100,4×最短路长度+20)，超时保留 TD 自举。

所有监督统计来自实际回放更新。环境标签可用率为行动前硬门控，回放覆盖率另含碰撞经验的监督筛除。碰撞经验仍参与 TD。案例中的教师动作、风险及权重在验证全部结束后离线计算，不参与决策。

## 校验和复现

仓库根目录执行 `python scripts/package_multitask_evidence.py --verify`。只需标准库，无需模型、PyTorch 或原始 outputs；命令校验 SHA、四次预算、10200 条验证记录、学习曲线积分、均值/样本标准差及同 seed 差异。

原工作空间可加 `--check-sources` 校验原始文件及模型/断点。训练入口保持原有严格冻结和历史保护，完整续训需要本地完整检查点；克隆后的阅读包校验不代表已具备全部训练恢复文件。不要修改冻结协议以绕过检查。

本包整理没有新增训练、模型评估、测试池访问或 Git 推送。旧实验和旧模型保持原样。上传命令见仓库 `docs/ASVC_MULTITASK_PILOT.zh-CN.md` 和 `docs/GITHUB_SYNC.zh-CN.md`。
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--check-sources',action='store_true')
    parser.add_argument('--package',type=Path)
    args=parser.parse_args()
    if args.verify:
        package=args.package or local(read(BASE/'latest.json')['package'])
        verify(package,args.check_sources)
    else:
        require(args.package is None,'--package is a verification-only argument')
        build()


if __name__=='__main__':main()
