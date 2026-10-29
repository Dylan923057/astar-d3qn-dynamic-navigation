"""Publish the completed TD-only control and full loop evidence. No training/network."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import package_multitask_comparison as prior

ROOT=Path(__file__).resolve().parents[1]
NAME='asvc_multitask_td_control_v1'
BASE=ROOT/'results/analysis_upload'/NAME
LIVE=ROOT/'results'/NAME
REG=ROOT/'data'/NAME
METHODS=(*prior.METHODS,'d3qn_bound')
METRICS=prior.MEASUREMENTS
read,write,sha,rows,csv_write,require,close,aulc=prior.read,prior.write,prior.sha,prior.rows,prior.csv_write,prior.require,prior.close,prior.aulc
CODE=('configs/asvc_multitask_td_control_v1.yaml','configs/multitask_teacher_loss_plan_v1.yaml',
    'scripts/td_control_common.py','scripts/td_control_engine.py','scripts/run_td_control.py','scripts/start_td_control.ps1',
    'scripts/diagnose_multitask_loops.py','scripts/cycle_observation_evidence.py','scripts/analyze_td_control.py',
    'scripts/plan_teacher_loss_control.py','scripts/package_td_control_evidence.py','scripts/sync_td_control_github.ps1',
    'tests/test_td_control.py','tests/test_multitask_loop_diagnostics.py','tests/test_cycle_observation_evidence.py')
DOCS=('MULTITASK_TD_CONTROL.zh-CN.md','TEACHER_LOSS_COMPARISON_PLAN.zh-CN.md','TD_CONTROL_GITHUB_UPLOAD.zh-CN.md')


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def norm(path):return str(path).replace('\\','/')
def local(base,name):return prior.local(base,norm(name))


def verify(package,check_sources=False):
    package=Path(package).resolve();manifest=read(package/'manifest.json')
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual==set(manifest['files_sha256'])|{'manifest.json'},'Publication file list differs')
    for name,digest in manifest['files_sha256'].items():
        p=local(package,name)
        require(p.suffix.lower() in prior.ALLOW and p.stat().st_size<50*1024**2,'Unsupported/oversized publication file: '+name)
        require(sha(p)==digest,'File SHA differs: '+name)
    require(manifest['new_training_runs']==2 and manifest['reused_training_runs']==8 and not manifest['weights_included'],'Scope differs')
    previous=package/'previous_four_methods';prior.verify(previous,False)
    registration=package/'data'/NAME;frozen=read(registration/'frozen_protocol.json')
    require(frozen['training_runs']==2 and frozen['online_steps_per_run']==500000 and frozen['expected_updates']==499501,'TD-control budget differs')
    old=read(previous/'registration/shared_pilot/frozen_protocol.json')
    external=previous/'registration/external_baselines/frozen_protocol.json'
    require(frozen['upstream_identity']['external_freeze_sha256']==sha(external),'External freeze differs')
    require(frozen['upstream_identity']['shared_identity']['pilot_freeze_sha256']==sha(previous/'registration/shared_pilot/frozen_protocol.json'),'Shared freeze differs')
    for name,digest in frozen['training_source_sha256'].items():
        require(sha(registration/'source_snapshot'/name)==digest and sha(package/'code'/name)==digest,'TD frozen code differs: '+name)
    analysis=package/'results'/NAME
    summaries={(r['method'],int(r['seed'])):r for r in rows(analysis/'five_method_per_seed.csv')}
    require(set(summaries)=={(m,s) for m in METHODS for s in (0,1)},'Five-method summary scope differs')
    index={(r['method'],int(r['seed'])):r for r in rows(package/'model_index.csv')}
    require(len(index)==10,'Model index must contain ten identities')
    curves={};final_details={};validation_count=0;target_count=0
    pools=read(previous/'registration/shared_pilot/task_pools.json');train={r['task_id']:r for r in pools['pools']['train']}
    scenes=read(previous/'registration/shared_pilot/validation_scenes.json')
    for method in METHODS:
        for seed in (0,1):
            directory=(package if method=='d3qn_bound' else previous)/'runs'/method/f'seed_{seed}'
            result=read(directory/'result.json');run=read(directory/'run_manifest.json')
            require(result['complete'] and result['step']==500000 and run['config']==old['config'],'Run completion/shared config differs')
            require(index[method,seed]['model_sha256']==result['final_model_sha256'],'Model SHA index differs')
            if method=='d3qn_bound':
                require(result['identity']==dict(method=method,seed=seed,freeze_sha256=sha(registration/'frozen_protocol.json')),'TD run identity differs')
                require(result['online_updates']==499501 and result['updates']==499501 and result['offline_updates']==0,'TD update budget differs')
                require(run['td_target_bounds']==[-6,10] and not any(run[k] for k in ('teacher_actions','teacher_loss','risk_weighting','demonstration_prefill','q_output_clip')),'Unexpected TD mechanism')
                for record in rows(directory/'training_episodes.csv'):require(record['task_id'] in train,'TD training used a validation/test task')
                samples=rows(directory/'td_target_samples.csv');require(len(samples)==32000,'Expected 32000 sampled targets per seed')
                for record in samples:
                    close(max(-6.,min(10.,float(record['raw_td_target']))),record['clipped_td_target'],'Complete TD clip')
                    require((record['bootstrap_enabled']=='True')==(record['terminated']=='False'),'Terminated/truncated bootstrap differs')
                target_count+=len(samples)
                log=rows(directory/'update_log.csv')
                require(sum(int(r['bin_updates']) for r in log)==499501,'TD actual update log count differs')
                require(all(-6<=float(r['clipped_target_min'])<=float(r['clipped_target_max'])<=10 for r in log),'TD targets outside bounds')
            curve=rows(directory/'validation_curve.csv');details=rows(directory/'validation_details.csv');curves[method,seed]=curve
            require([int(r['environment_steps_total']) for r in curve]==list(range(0,500001,10000)) and len(details)==2550,'Validation history incomplete')
            for point in curve:
                part=[r for r in details if int(r['environment_steps_total'])==int(point['environment_steps_total'])]
                require([r['task_id'] for r in part]==[r['task_id'] for r in scenes],'Validation tasks differ')
                require([int(r['scene_seed']) for r in part]==[r['scene_seed'] for r in scenes],'Validation scenes differ')
                for metric in ('success','collision','timeout'):
                    close(sum(int(r[metric]) for r in part)/50,point[metric+'_rate'],'Validation aggregate')
            final_details[method,seed]={r['task_id']:r for r in details if int(r['environment_steps_total'])==500000}
            summary=summaries[method,seed]
            for start,end in ((0,500000),(0,100000),(100000,250000),(250000,500000),(400000,500000)):
                close(aulc(curve,start,end),summary[f'aulc_{start}_{end}'],'AULC')
            for metric in ('success','collision','timeout'):close(curve[-1][metric+'_rate'],summary['final_'+metric+'_rate'],'Final outcome')
            close(sum(float(r['success_rate']) for r in curve[-5:])/5,summary['last_five_success_mean'],'Last five')
            validation_count+=len(details)
    require(validation_count==25500 and target_count==64000,'Total validation/target sample count differs')
    import statistics
    for row in rows(analysis/'five_method_mean_sample_sd.csv'):
        for metric in METRICS:
            values=[float(summaries[row['method'],s][metric]) for s in (0,1)]
            close(statistics.mean(values),row[metric+'_mean'],'Seed mean');close(statistics.stdev(values),row[metric+'_sample_sd'],'Seed sample SD')
    effects=rows(analysis/'controlled_effects_per_seed.csv')
    for row in effects:
        seed=int(row['seed']);left,right=('d3qn_bound','d3qn') if row['contrast']=='TD_only_minus_raw' else ('C','d3qn_bound')
        for metric in METRICS:close(float(summaries[left,seed][metric])-float(summaries[right,seed][metric]),row[metric],'Controlled difference')
    for row in rows(analysis/'controlled_effects_mean_sample_sd.csv'):
        part=[r for r in effects if r['contrast']==row['contrast']]
        for metric in METRICS:
            close(statistics.mean(float(r[metric]) for r in part),row[metric+'_mean'],'Effect mean')
            close(statistics.stdev(float(r[metric]) for r in part),row[metric+'_sample_sd'],'Effect sample SD')
    pointer=read(registration/'loop_latest.json');loop_root=local(package,pointer['results_path'])
    protocol_file=registration/'loop_protocols'/Path(norm(pointer['protocol_path'])).name
    spec=read(protocol_file)
    import hashlib
    require(hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()==pointer['protocol_sha256'],'Diagnostic protocol digest differs')
    require(spec['source_sha256']==sha(package/'code/scripts/diagnose_multitask_loops.py'),'Diagnostic code differs')
    paired_count=0;new_policy=0;step_count=0;goal_errors=0;teacher_errors=0
    for method in METHODS:
        for seed in (0,1):
            for condition in ('dynamic','static'):
                directory=loop_root/method/f'seed_{seed}'/condition
                done=read(directory/'completion.json');part=rows(directory/'per_task.csv');traces=read(directory/'trajectories.json')
                require(done['complete'] and done['original_scores_verified'] and len(part)==len(traces)==50,'Missing paired trajectories')
                require(done['identity']['model_sha256']==index[method,seed]['model_sha256'],'Diagnostic model differs')
                require(done['actual_evaluation_supervision_weight']==0 and done['updates']==0,'Evaluation must have no learning')
                expected_new=46 if method in ('C','D') and condition=='dynamic' else 50 if method=='d3qn_bound' and condition=='static' else 0
                require(done['new_policy_episodes']==expected_new,'Unexpected extra policy evaluation')
                new_policy+=done['new_policy_episodes'];paired_count+=len(part)
                for record in part:
                    if condition=='dynamic':
                        for metric in ('success','collision','timeout','steps','wait_steps'):
                            close(record[metric],final_details[method,seed][record['task_id']][metric],'Original dynamic score reuse')
                    goal_errors+=int(record['goal_input_mismatch']);teacher_errors+=int(record['teacher_label_mismatch'])
                for trace in traces:require(len(trace['positions'])==len(trace['actions'])+1,'Incomplete trace')
                for row in rows(directory/'timeout_and_reference_steps.csv'):
                    require(float(row['actual_evaluation_supervision_weight'])==0 and row['offline_only']=='True','Hypothetical weight confused with learning')
                    q=[float(row[f'q_{a}']) for a in range(5)];legal=[a for a in range(5) if row[f'legal_{a}']=='True'];teacher=int(row['teacher_action'])
                    require(teacher in legal and 1+sum(q[a]>q[teacher]+1e-6 for a in legal)==int(row['teacher_q_legal_rank']),'Teacher Q rank differs')
                    close(q[teacher],row['teacher_q'],'Teacher Q value')
                    active=row['hard_screen_pass']=='True';risk=float(row['risk_R']);require(0<=risk<=1,'Invalid heuristic risk')
                    close(.1 if active else 0,row['hypothetical_state_C_weight'],'Hypothetical C coefficient')
                    close(.1*(1-risk)**2 if active else 0,row['hypothetical_state_D_weight'],'Hypothetical D coefficient')
                    step_count+=1
    require(paired_count==1000 and new_policy==284,'Paired/replay budget differs')
    replay_rows=rows(analysis/'replay_goal_teacher_audit.csv');require(sum(int(r['transitions']) for r in replay_rows)==80000,'Final replay audit coverage differs')
    repeated=rows(analysis/'repeated_observation_states.csv');recurrent=rows(analysis/'recurrent_observation_cycles.csv')
    require(all(r['q_consistent']=='True' and int(r['visits'])>=3 for r in repeated),'Repeated-observation evidence differs')
    require(all(int(r['cycle_length'])>=2 and int(r['minimum_visits_per_state'])>=3 for r in recurrent),'Invalid recurrent-observation cycle')
    if check_sources:
        for name,digest in (manifest['source_files_sha256']|manifest['models_sha256']|frozen['protected_upstream_results_sha256']).items():
            require(sha(local(ROOT,name))==digest,'Original local source/model changed: '+name)
    for name in manifest['files_sha256']:
        path=package/name
        if path.suffix in {'.py','.md','.csv','.json','.log','.txt','.yaml','.ps1'}:
            require(not any(re.search(pattern,path.read_text(encoding='utf-8-sig')) for pattern in prior.SECRET_PATTERNS),'Potential secret; inspect locally: '+name)
    size=sum((package/name).stat().st_size for name in manifest['files_sha256'])
    print(f'PASS TD CONTROL: {len(actual)} files / {size/1024**2:.2f} MiB; ten runs, 25500 validation rows, 64000 target samples.',flush=True)
    print(f'20 paired conditions /1000 traces; {step_count} offline Q/risk rows; 80000 retained replay transitions; {len(recurrent)} recurrent observation cycles.',flush=True)
    print(f'Goal/teacher mismatches in paired scores: {goal_errors}/{teacher_errors}. Extra dynamic replay184 + new static100; test pools0; no weights/checkpoints.',flush=True)
    return manifest


def build():
    import td_control_common as c
    c.verify(True);pipeline=read(c.OUT/'pipeline_status.json')
    require(pipeline['complete'] and not pipeline['running'] and not pipeline['failures'] and not pipeline['postprocessing_errors'],'TD round incomplete')
    previous=ROOT/read(prior.BASE/'latest.json')['package'];prior.verify(previous,True)
    published=BASE/datetime.now().strftime('review_%Y%m%d_%H%M%S');require(not published.exists(),'Publication exists')
    destination=LIVE/'publication_staging'/published.name;destination.mkdir(parents=True,exist_ok=False)
    originals={};models={}
    def copy(source,target):
        source=Path(source);target=local(destination,target)
        require(source.is_file() and source.suffix in prior.ALLOW and source.stat().st_size<50*1024**2,'Unsupported file: '+str(source))
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);originals[rel(source)]=sha(source)
    for path in sorted(previous.rglob('*')):
        if path.is_file():copy(path,Path('previous_four_methods')/path.relative_to(previous))
    for path in sorted(REG.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:copy(path,Path('data')/NAME/path.relative_to(REG))
    for name in CODE:copy(ROOT/name,Path('code')/name)
    for name in DOCS:copy(ROOT/'docs'/name,Path('docs')/name)
    for path in sorted(LIVE.iterdir()):
        if path.is_file() and path.suffix in prior.ALLOW:copy(path,Path('results')/NAME/path.name)
    loop=ROOT/read(REG/'loop_latest.json')['results_path']
    for folder in (loop,LIVE/'observation_cycle_evidence',LIVE/'typical_cases'):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in prior.ALLOW:copy(path,Path('results')/NAME/path.relative_to(LIVE))
    model_rows=rows(previous/'model_index.csv');old_manifest=read(previous/'manifest.json');models.update(old_manifest['models_sha256'])
    for seed in (0,1):
        info=c.model_info(c.METHOD,seed);require(info is not None,'TD final model missing')
        for path in sorted(info['directory'].iterdir()):
            if path.suffix in ('.json','.csv'):copy(path,Path('runs')/c.METHOD/f'seed_{seed}'/path.name)
        copy(c.OUT/'logs'/f'seed_{seed}.log',Path('runs')/c.METHOD/f'seed_{seed}/console.log')
        checkpoint=info['directory']/'checkpoint_latest.pth';models[rel(info['model'])]=info['model_sha256'];models[rel(checkpoint)]=sha(checkpoint)
        model_rows.append(dict(method=c.METHOD,seed=seed,online_steps=500000,online_updates=499501,offline_updates=0,
            model_path=rel(info['model']),model_sha256=info['model_sha256'],full_checkpoint_path=rel(checkpoint),checkpoint_sha256=sha(checkpoint),weights_included=False))
    csv_write(destination/'model_index.csv',model_rows)
    copy(Path(__file__),'tools/verify_bundle.py');copy(ROOT/'scripts/package_multitask_comparison.py','tools/package_multitask_comparison.py')
    write(destination/'provenance/pipeline_status.json',pipeline)
    write(destination/'provenance/reused_bundle.json',dict(original_package=rel(previous),original_manifest_sha256=sha(previous/'manifest.json'),copied_byte_for_byte=True,
        note='Historical eight-run evidence retained unchanged; latest C/D dynamic trajectories are complete in this new round.'))
    scope=sorted(set(CODE)|{'.gitignore','.gitattributes',rel(REG),rel(published),rel(BASE/'LATEST.md'),rel(BASE/'latest.json'),'docs/GITHUB_SYNC.zh-CN.md',*[f'docs/{name}' for name in DOCS]})
    write(destination/'upload_scope.json',scope);(destination/'README.md').write_text(README,encoding='utf-8')
    manifest=dict(complete=True,created=datetime.now().astimezone().isoformat(timespec='seconds'),new_training_runs=2,reused_training_runs=8,
        total_methods=5,seeds_per_method=2,total_online_steps=5000000,new_online_steps=1000000,
        validation_checkpoints=510,validation_episode_rows=25500,paired_task_scores=1000,paired_full_trajectories=1000,
        additional_c_d_dynamic_policy_episodes=184,new_td_static_policy_episodes=100,test_pool_evaluations=0,
        additional_training_runs=0,weights_included=False,full_replay_included=False,
        source_files_sha256=originals,models_sha256=models,
        files_sha256={p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    write(destination/'manifest.json',manifest);verify(destination,True)
    BASE.mkdir(parents=True,exist_ok=True);destination.rename(published)
    write(BASE/'latest.json',dict(package=rel(published)))
    (BASE/'LATEST.md').write_text('# TD约束控制与循环诊断：最新完整阅读包\n\n'
        f'[开始阅读]({published.name}/README.md) · [最终发现]({published.name}/results/{NAME}/FINAL_FINDINGS.zh-CN.md) · '
        f'[五方法逐seed]({published.name}/results/{NAME}/five_method_per_seed.csv) · '
        f'[学习曲线]({published.name}/results/{NAME}/five_method_learning_curves.png)\n\n'
        '新增TD-only两个seed均完成50万在线步，复用原四方法8次运行。TD-only成功均值14%→31%，C为74%；C−TD-only是教学组合差，不能归于持续监督单项。两seed冻结验证，无未见目标测试或其他新增训练。\n',encoding='utf-8')
    return published


README='''# 五方法多任务比较：TD约束单独控制与循环诊断

本轮2026-10-29 18:21完成。新增D3QN＋TD约束seed0/1，每次500000在线步、499501更新，无预训练/示范/教师；旧四方法8次运行复用。没有重复训练，没有启动CE、监督下限扫描、五seed、正式消融或两个测试池。

| 方法 | 成功率seed0/1 | 平均成功 | 平均碰撞 | 平均超时 | 全程AULC均值±样本SD |
|---|---:|---:|---:|---:|---:|
| 原始D3QN | 2% / 26% | 14% | 20% | 66% | 5.88% ± 4.95% |
| D3QN＋TD约束 | 26% / 36% | 31% | 11% | 58% | 15.99% ± 2.50% |
| DQfD | 12% / 20% | 16% | 16% | 68% | 14.94% ± 1.02% |
| C 持续弱监督 | 76% / 72% | 74% | 13% | 13% | 47.14% ± 1.78% |
| D 风险自适应 | 74% / 64% | 69% | 3% | 28% | 44.17% ± 0.07% |

TD约束单独令平均成功提高17个百分点，AULC提高10.11个百分点；在其基础上C再提高43个百分点、31.15个百分点。后者只能解释为A*代选＋持续动作监督的教学组合增益，不能归于持续监督单项，也不能把代数差分比例当成独立因果贡献。

## 阅读顺序

1. [最终发现](results/asvc_multitask_td_control_v1/FINAL_FINDINGS.zh-CN.md)、[完整报告](results/asvc_multitask_td_control_v1/REPORT.zh-CN.md)。
2. [逐seed与分阶段AULC](results/asvc_multitask_td_control_v1/five_method_per_seed.csv)、[均值/样本SD](results/asvc_multitask_td_control_v1/five_method_mean_sample_sd.csv)、[TD/教学组合差值](results/asvc_multitask_td_control_v1/controlled_effects_per_seed.csv)。
3. [五方法学习曲线](results/asvc_multitask_td_control_v1/five_method_learning_curves.png)、[PDF](results/asvc_multitask_td_control_v1/five_method_learning_curves.pdf)。
4. [循环与长度分档](results/asvc_multitask_td_control_v1/loop_seed_summary.csv)、[完整1000条配对成绩](results/asvc_multitask_td_control_v1/all_paired_tasks.csv)、[典型案例索引](results/asvc_multitask_td_control_v1/typical_cases.csv)和typical_cases图。
5. [重复观测状态](results/asvc_multitask_td_control_v1/repeated_observation_states.csv)、[观测闭环](results/asvc_multitask_td_control_v1/recurrent_observation_cycles.csv)、[回放目标/教师审计](results/asvc_multitask_td_control_v1/replay_goal_teacher_audit.csv)、[Q/TD及成本](results/asvc_multitask_td_control_v1/learning_and_cost_audit.csv)。
6. [独立冻结与复现命令](docs/MULTITASK_TD_CONTROL.zh-CN.md)、[CE/大间隔未执行方案](docs/TEACHER_LOSS_COMPARISON_PLAN.zh-CN.md)。

## 数据口径与文件

- `runs/d3qn_bound/seed_0或1/`：新增两次完整51点曲线、各2550条验证记录、训练回合、原始/裁剪TD和Q/梯度日志、各32000条TD目标样本、运行登记与完成记录。
- `previous_four_methods/`：上一轮完整阅读包按原字节嵌入，含原8次运行、共同输入、3472条DQfD示范、原始源码及冻结。其manifest不改。旧报告中C/D动态缺失描述是旧轮记录，本轮以新的50/50轨迹为准。
- 本包合计10条训练曲线、510个检查点、25500条验证记录；反复使用同一50任务，不是25500个独立任务或训练seed。
- 新诊断版本目录包含全部20条件、1000份完整轨迹、所有超时及固定参考任务的逐步五动作Q、合法掩码、教师排名/Q差、硬筛查、风险和事后真实冲突。动态原成绩保留，C/D补录184个缺失轨迹，新增控制静态100回合；不是1000次新增评估。
- C/D最终80000条保留回放未发现目标/静态教师/风险存储错配，仅覆盖最终缓冲区。反复出现相同观测、Q及贪心动作的闭环解释执行行为，不能单独确定训练原因。实际评估监督系数始终0，所有C/D权重是事后假设，真实未来只作离线误差分析。
- 静态配对也改变输入分布，不等于单独静态训练。D保持探索方法，不因碰撞低就升级为主方法。
- `data/asvc_multitask_td_control_v1/`：新冻结、原始训练源快照、补录授权、版本化诊断协议及清单。`code/`是本轮代码/测试/禁用CE计划；共同网络/环境源码在previous_four_methods/code，完整仓库中均在原位置。
- `model_index.csv`记录10个最终模型和完整断点的路径/SHA，本体保留本地。无权重不能重跑网络评估，只有权重也不等于完整续训。

DQfD额外20000离线更新/seed、永久3472示范仅成功覆盖70/240训练任务和31/40目标，生成15107环境步成本单列；不因为成绩低就断言Q爆炸。仅两seed验证，不宣称正式显著性或未见目标泛化。

## 只读校验

仅下载本资料目录后，进入目录运行（只需Python标准库，无PyTorch、GPU或权重）：

```powershell
python tools/verify_bundle.py --verify --package .
```

完整仓库根目录：

```powershell
python scripts/package_td_control_evidence.py --verify
# 仅原工作区具有原日志、模型和完整断点时：
python scripts/package_td_control_evidence.py --verify --check-sources
```

校验SHA、原冻结、10次预算、25500条验证、均值/样本SD、AULC及受控差值、64000个原始/裁剪目标、20组完整配对与离线权重语义。不会训练、运行策略、联网、暂存或推送。

## 上传范围

upload_scope.json限定本批代码、输入、文档和阅读包；旧包保留。模型、回放、缓存、无关README/历史轨迹、论文草稿、转换工具和其他未提交实验不混入。upload脚本只在用户自行执行时提交/推送，不强推，不清空范围外暂存。具体命令见docs/TD_CONTROL_GITHUB_UPLOAD.zh-CN.md。
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    parser.add_argument('--check-sources',action='store_true');parser.add_argument('--package',type=Path);parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    if args.verify:
        package=args.package or (ROOT if (ROOT/'manifest.json').exists() else ROOT/read(BASE/'latest.json')['package'])
        verify(package,args.check_sources)
    else:
        package=build();print(rel(package),flush=True)
        if args.archive:
            archive=BASE.parent/f'td_control_{package.name}.zip';require(not archive.exists(),'Archive exists')
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as target:
                for p in sorted(package.rglob('*')):
                    if p.is_file():target.write(p,Path(package.name)/p.relative_to(package))
            print(f'Archive: {archive} / {archive.stat().st_size/1024**2:.2f} MiB',flush=True)


if __name__=='__main__':main()
