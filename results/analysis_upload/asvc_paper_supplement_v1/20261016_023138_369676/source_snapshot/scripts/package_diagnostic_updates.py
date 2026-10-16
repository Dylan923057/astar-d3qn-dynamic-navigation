"""Prepare/verify a compact post-DDQNA GitHub reading package. Never trains or pushes."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

import numpy as np

import run_staged_withdrawal_pilot as staged
import run_dqfd_no_pretrain as no_pretrain
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT=staged.common.ROOT
PACKAGE_ROOT=ROOT/'results/analysis_upload/diagnostics_withdrawal_update_v1'
EVIDENCE=ROOT/'results/comparison_evidence_diagnosis/20261010_162403_833725'
PROBE=ROOT/'results/ddqna_stability_probe_v1/analysis_corrected_20261010_232350_941649'
PROBE_RUN=ROOT/'outputs/ddqna_stability_probe_v1/20261010_222448_137813'
WITHDRAWAL=ROOT/'results/teaching_withdrawal_diagnosis/20261011_012947_389779'
STAGED=ROOT/'results/whole_map_91701_staged_withdrawal_pilot_v1/review_20261011_121016_214692'
NEW_SOURCES=[
    'configs/ddqna_stability_probe_v1.yaml','configs/whole_map_91701_dqfd_no_pretrain_v1.yaml',
    'configs/whole_map_91701_staged_withdrawal_pilot_v1.yaml',
    'scripts/diagnose_comparison_evidence.py','scripts/render_comparison_diagnosis.py',
    'scripts/probe_ddqna_stability.py','scripts/analyze_ddqna_stability_probe.py',
    'scripts/run_dqfd_no_pretrain.py','scripts/diagnose_teaching_withdrawal.py',
    'scripts/summarize_teaching_withdrawal.py','scripts/run_staged_withdrawal_pilot.py',
    'scripts/analyze_staged_withdrawal_result.py','scripts/package_diagnostic_updates.py',
    'scripts/sync_github.ps1','src/astar_d3qn/agents/astar_staged_withdrawal.py',
    'tests/test_comparison_diagnosis.py','tests/test_probe_analysis.py',
    'tests/test_dqfd_no_pretrain.py','tests/test_staged_withdrawal_pilot.py']


def relative(path): return Path(path).relative_to(ROOT).as_posix()


def verify(directory):
    manifest=staged.common.read_json(directory/'manifest.json')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual!=set(manifest['package_files_sha256']): raise ValueError('Package file list changed.')
    for name,digest in manifest['package_files_sha256'].items():
        if staged.common.sha256(directory/name)!=digest: raise ValueError(f'Package checksum mismatch: {name}')
    print(f'PASS: {len(actual)} files; {manifest["package_bytes"]/1048576:.2f} MiB; no training/push.',flush=True)
    return manifest


def dqfd_control(config,output):
    common=no_pretrain.common; original,inputs=no_pretrain.validate(config)
    directory=ROOT/config['experiment']['output_root']/no_pretrain.METHOD/'seed_0'
    result=common.read_json(directory/'result.json'); done=common.read_json(directory/'completion.json')
    registration_path=ROOT/config['no_pretrain_control']['registration_root']/'freeze_manifest.json'
    registration=common.read_json(registration_path)
    frozen=common.frozen(original)
    if (not done['complete'] or done['environment_steps']!=200000 or result['config']!=config
            or result['environment_steps']!=200000 or result['pretraining_updates']!=0
            or result['online_updates']!=199501 or result['gradient_updates']!=199501
            or result['permanent_demo_retained_count']!=7260 or result['evaluation_advice']
            or result['final500_used_for_training_or_scheduling']
            or result['initial_learner_sha256']!=registration['paired_initializations']['0']
            or result['demonstration_sha256']!=frozen['files_sha256']['demonstrations.npz']
            or result['freeze_manifest_sha256']!=common.sha256(registration_path)
            or common.sha256(directory/'model_final.pth')!=result['final_model_sha256']):
        raise ValueError('DQfD no-pretraining seed0 does not match its frozen protocol.')
    for name,digest in registration['scripts_sha256'].items():
        if common.sha256(ROOT/name)!=digest: raise ValueError('DQfD control source changed.')
    baseline,baseline_result=common.audit_run(original,'dqfd',0)
    rows=[]
    for method,path,run in (('dqfd',baseline,baseline_result),(no_pretrain.METHOD,directory,result)):
        curve=load_csv(path/'validation_curve.csv'); details=load_csv(path/'validation_details.csv')
        x=np.asarray([int(r['environment_steps_total']) for r in curve]); y=np.asarray([float(r['safe_success_rate']) for r in curve])
        if len(curve)!=21 or len(details)!=1050 or x[0]!=0 or x[-1]!=200000: raise ValueError('Incomplete DQfD curve.')
        scenes=common.read_json(ROOT/config['dataset']['validation_reference'])['scenarios']
        for point in curve:
            chosen=[r for r in details if r['environment_steps_total']==point['environment_steps_total']]
            if ([r['scenario_id'] for r in chosen]!=[s['scenario_id'] for s in scenes] or
                    not np.isclose(float(point['safe_success_rate']),np.mean([float(r['safe_success']) for r in chosen]))):
                raise ValueError('DQfD validation scene/aggregate mismatch.')
        traces=common.read_json(path/'validation_failures_200000.json')
        behaviors=[trace_behavior(t,static_distances(inputs[0])) for t in traces]
        rows.append(dict(method=method,seed=0,full_aulc=aulc(x,y),aulc_0_80000=aulc(x,y,0,80000),
            aulc_100000_200000=aulc(x,y,100000,200000),last_five_mean=float(y[-5:].mean()),
            final_success=y[-1],final_collision=float(curve[-1]['dynamic_collision_rate']),
            final_timeout=float(curve[-1]['timeout_rate']),**failure_counts(behaviors),
            offline_updates=run['pretraining_updates'],online_steps=200000,online_updates=199501,
            total_updates=run['gradient_updates'],permanent_demos=7260,final500_new_control_evaluated=False))
    write_records_csv(rows,output/'dqfd_no_pretrain_seed0.csv')
    write_records_csv([dict(metric=k,new_minus_old=rows[1][k]-rows[0][k]) for k in
        ('full_aulc','aulc_0_80000','aulc_100000_200000','final_success','final_collision','final_timeout')],output/'dqfd_no_pretrain_paired_seed0.csv')
    return directory,baseline,rows


def package():
    common=staged.common; config=common.legacy.load_config(staged.CONFIG); inputs=staged.validate(config)
    stage_run=ROOT/config['experiment']['output_root']/staged.METHOD/'seed_0'
    run=common.read_json(stage_run/'result.json'); registered=common.read_json(ROOT/config['staged_withdrawal']['artifact_root']/'freeze_manifest.json')
    if (run['source_sha256']!=staged.source_hashes() or run['source_sha256']!=registered['source_sha256']
            or run['initial_state_sha256']!=registered['paired_initial_state_sha256']
            or common.sha256(stage_run/'model_final.pth')!=run['final_model_sha256']
            or not common.read_json(stage_run/'completion.json')['complete']):
        raise ValueError('Staged withdrawal run/source audit failed.')
    for name in ('verification.json','metrics.csv','same_state_rankings.csv'):
        if not (STAGED/name).is_file(): raise FileNotFoundError(STAGED/name)
    output=PACKAGE_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); output.mkdir(parents=True,exist_ok=False)
    copied={}; models={}; trajectories=[]
    def copy(path,destination):
        if not path.is_file(): raise FileNotFoundError(path)
        copied[relative(path)]=common.sha256(path)
        target=output/destination; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    def folder(path,destination,skip=lambda p:False):
        for p in sorted(path.rglob('*')):
            if p.is_file() and not skip(p): copy(p,Path(destination)/p.relative_to(path))
    for verification,key in ((PROBE/'verification.json','model_sha256'),(WITHDRAWAL/'verification.json','checkpoint_hashes'),
                             (STAGED/'verification.json','model_sha256'),(EVIDENCE/'diagnosis/verification.json','source_models_sha256')):
        audit=common.read_json(verification)
        if not audit['complete']: raise ValueError('Incomplete diagnosis.')
        for name,digest in audit[key].items():
            if common.sha256(ROOT/name)!=digest: raise ValueError(f'Changed model: {name}')
            models[name]=digest
    folder(EVIDENCE/'evidence','evidence')
    for p in EVIDENCE.iterdir():
        if p.is_file(): copy(p,Path('comparison_diagnosis')/p.name)
    folder(EVIDENCE/'diagnosis','comparison_diagnosis/records',lambda p:p.name.endswith('_q_decisions.csv'))
    folder(PROBE,'stability_probe')
    folder(WITHDRAWAL,'withdrawal_diagnosis',lambda p:p.suffix=='.npz' or '_q_anchor' in p.name)
    folder(STAGED,'staged_seed0')
    nc=no_pretrain.common.legacy.load_config(no_pretrain.CONFIG)
    dqfd_run,dqfd_baseline,dqfd_rows=dqfd_control(nc,output)
    files_stage=('run_manifest.json','result.json','completion.json','training.csv','validation_curve.csv',
        'validation_details.csv','advice_budget.csv','value_learning.csv','true_training_td.csv',
        'collection_label_census.csv','training_scenes.csv')
    for filename in files_stage: copy(stage_run/filename,Path('run_logs/staged_seed0')/filename)
    for filename in ('result.json','completion.json','training.csv','validation_curve.csv','validation_details.csv','dqfd_learning.csv'):
        copy(dqfd_run/filename,Path('run_logs/dqfd_no_pretrain_seed0')/filename)
    for arm in ('huber','mse','huber_bound'):
        directory=PROBE_RUN/arm/'seed_0'
        arm_run=common.read_json(directory/'result.json')
        if not arm_run['complete'] or arm_run['environment_steps']!=50000 or arm_run['gradient_updates']!=49501:
            raise ValueError('Incomplete 50k probe.')
        for filename in ('result.json','training.csv','validation_curve.csv','action_branches.csv','update_profiles.csv','training_scenes.csv'):
            copy(directory/filename,Path('run_logs/stability_probe')/arm/filename)
    for path in (ROOT/nc['no_pretrain_control']['registration_root'],ROOT/config['staged_withdrawal']['artifact_root']):
        folder(path,Path('registration')/path.name)
    for name in sorted(set(NEW_SOURCES)|set(staged.source_hashes())|set(registered['source_sha256'])):
        copy(ROOT/name,Path('source_snapshot')/name.replace('\\','/'))
    # Deterministic examples only. Aggregate tables include every scene; these never select models.
    candidates=[(f'probe_{arm}',PROBE_RUN/arm/'seed_0'/'failures_050000.json') for arm in ('huber','mse','huber_bound')]
    old,_=common.audit_run(common.legacy.load_config(common.CONFIG),'advice_bound_margin',0)
    candidates.extend((label,path) for label,path in (
        ('staged_new_100k',stage_run/'validation_failures_100100.json'),
        ('staged_old_100k',old/'validation_failures_100107.json'),
        ('dqfd_no_pretrain_final',dqfd_run/'validation_failures_200000.json')))
    for label,path in candidates:
        seen=set(); source_sha=common.sha256(path)
        for trace in common.read_json(path):
            b=trace_behavior(trace,static_distances(inputs[0]))
            kind='collision' if b['termination_reason']=='collision' else 'waiting_timeout' if b['waiting_dominated'] else 'looping_timeout' if b['repeated_movement'] or b['tail_period'] else 'other_timeout'
            if kind in seen: continue
            seen.add(kind); target=Path('trajectories')/f'{label}_{kind}.json'; write_json(trace,output/target)
            trajectories.append(dict(label=label,category=kind,scenario_id=trace['scenario_id'],package_path=target.as_posix(),
                source=relative(path),source_sha256=source_sha,selection='first in saved scene order per category; diagnostic only'))
    write_records_csv(trajectories,output/'trajectory_index.csv')
    for directory in (dqfd_run,dqfd_baseline,stage_run): models[relative(directory/'model_final.pth')]=common.sha256(directory/'model_final.pth')
    write_json(dict(training_started=False,evaluation_rerun=False,push_performed=False,
        statistical_unit='seed',probe_seeds=[0],no_pretrain_completed_seeds=[0],staged_completed_seeds=[0],
        other_seeds_not_started=True,original_formal_runs_reused=60,final500_used_for_new_controls=False,
        corrected_probe_aulc_window=[0,50000],selected_trajectories=len(trajectories),weights_uploaded=False,
        bulk_q_and_updates_uploaded=False),output/'verification.json')
    overview=['# 上次DDQNA上传后的诊断与单seed对照','',
        '阅读范围：固定地图、相同起终点、每回合3–5个动态障碍；研究在线A*教学加快D3QN自主学习。没有改地图或奖励。','',
        '| 工作 | 状态 | 阅读入口 |','|---|---|---|',
        '| 12方法×5seed证据与DDQNA诊断 | 复用60次已有正式运行；核查seed0/2 | [报告](comparison_diagnosis/REPORT.md) |',
        '| DDQNA稳定性探针 | Huber/MSE/Huber＋TD限制，均seed0、5万步 | [更正后报告](stability_probe/REPORT.md) |',
        '| 撤教附近同状态诊断 | 原组合五seed，8–12万步 | [报告](withdrawal_diagnosis/REPORT.txt) |',
        '| DQfD取消离线预训练 | seed0完成20万在线步；取消2万离线更新 | [配对结果](dqfd_no_pretrain_seed0.csv) |',
        '| 先撤代选、后撤监督 | seed0完成20万步；8万代选退出、10万监督退出 | [报告](staged_seed0/REPORT.txt) |','',
        '## 当前证据','',
        '- 原组合五seed全程AULC91.37±1.05%；动作监督＋TD约束组合增益、在线代选额外贡献有配对证据；不主张最终避障全面最优。',
        '- 稳定性探针：Huber/MSE Q值增大；TD限制抑制数值增长，但5万步自主成功率仅0%、0%、2%。单seed适配诊断，不评价原版DDQNA优劣。',
        '- 分阶段撤教seed0：全程AULC89.69%→93.48%，10–20万92.68%→97.79%，低谷70%→88%，两组最终均100%。低谷仍在10万附近；只是一seed证据，代选概率日程面积少8%。',
        f'- DQfD取消预训练seed0：全程AULC{dqfd_rows[0]["full_aulc"]:.2%}→{dqfd_rows[1]["full_aulc"]:.2%}，最终原50验证成功{dqfd_rows[0]["final_success"]:.0%}→{dqfd_rows[1]["final_success"]:.0%}。保留7260条永久示范、PER、多步TD、margin、L2及在线设置。不能由一seed归因全部优势。',
        '- 两个新单seed对照尚无独立500最终确认评估；不要把它们的50场景结果与旧方法500场景结果直接比较。','',
        '## 更正、预算与缺失项','',
        '探针以0–50000步计算AULC，正确值Huber1.60%、MSE1.20%、Huber＋TD限制0.20%。旧短探针汇总曾误外推到20万步，其AULC已被取代，不纳入本阅读包；原20万步实验不受影响。',
        '原DQfD共219501梯度更新（离线20000＋在线199501）；取消预训练为199501。相同在线交互步数不等于相同总训练预算；两者都保留永久示范。',
        '风险筛选不保证安全；教学延期不等于解决退出低谷；DDQNA全0不作为主要优势证据。统计单位是训练seed，单seed没有样本标准差。',
        '旧方法缺少的真实TD误差分布、关键训练状态覆盖、A*计数和耗时继续标为缺失，不以冻结自举残差或Huber损失替代真实TD误差。',
        '早期登记的checks.json只描述当时未启动训练的状态；当前完成状态见run_logs/result.json及completion.json，勿误读为训练尚未运行。','',
        '## 文件与复核','',
        '`run_logs/`保留训练曲线、验证逐场景明细、覆盖率与成本；`source_snapshot/`保留新增实现和冻结依赖；`registration/`保存预注册，`trajectories/`按固定规则选取失败案例。',
        '权重、全量逐更新/逐动作Q日志、重复诊断与重复探针留在本地。更早同参数探针不是新增seed，详细记录复测使用20261010_222448_137813。模型路径和SHA见manifest.json。',
        '仅克隆仓库不能重新加载本地模型，不能因模型缺失自动重训。包内表格与报告可独立阅读；有本地原始outputs时可复跑诊断。','',
        '上传前只校验本包：`python scripts/package_diagnostic_updates.py --verify <本包目录>`。该命令不训练、不推送。','',
        '[给GPT的分析请求](GPT_ANALYSIS_PROMPT.md)']
    (output/'README.md').write_text('\n'.join(overview)+'\n',encoding='utf-8')
    (output/'GPT_ANALYSIS_PROMPT.md').write_text('''请先阅读README.md及manifest.json，基于本包代码和数据分析，保持固定地图、同起终点、3–5动态障碍的范围。
1. 区分已有五seed论文证据、seed0探针和两个seed0单因素对照，不把验证回合当成训练seed。
2. 核查在线代选、动作监督、TD约束的条件增益；保留DQfD强基线并区分在线交互与总更新预算。
3. 检查分阶段撤教是否减轻低谷，结合两个退出窗口、后段AULC、等待/绕圈/碰撞和同状态Q排名；不宣称已经解决。
4. DQfD取消离线预训练仅seed0，示范及在线机制均保留，独立500尚未评估；说明比较边界和最少的补充证据。
5. DDQNA探针AULC只用0–5万步，真实TD误差与冻结残差不同；数值改善不等于导航改善，也不能否定原论文。
6. 提出简洁贡献表述和下一步最少对照。不要建议改地图/奖励迎合结果，不用已有500结果调参，不选择最佳检查点。
缺失的信息明确写“缺失”或“未确认”；给出表格对应来源路径，避免首次融合、保证安全、全面最优等结论。
''',encoding='utf-8')
    git=Path('D:/softwares/Anaconda3/envs/yolov5/Library/cmd/git.exe')
    def git_read(*args):
        return subprocess.check_output([str(git),'-c','safe.directory='+ROOT.as_posix(),*args],cwd=ROOT,text=True).strip()
    package_files={p.relative_to(output).as_posix():common.sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(dict(base_commit=git_read('rev-parse','HEAD'),cached_origin_main=git_read('rev-parse','origin/main'),
        remote_state_live_verified=False,source_files_sha256=copied,source_models_sha256=models,
        package_files_sha256=package_files,package_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        source_weights_uploaded=False,training_started=False,push_performed=False),output/'manifest.json')
    verify(output); print(output); return output


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if args.verify: verify(args.verify)
    else: package()


if __name__=='__main__': main()
