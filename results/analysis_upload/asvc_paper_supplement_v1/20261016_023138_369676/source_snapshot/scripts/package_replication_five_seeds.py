"""Pack audited five-seed curves, run logs and registered sources. Never trains or pushes."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess

import analyze_replication_seeds01234 as analysis
from analyze_runtime_path_pilot import load_csv, trace_behavior, static_distances
from analyze_supervision_tail import failure_counts
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT,common=analysis.ROOT,analysis.common
PACKAGE_ROOT=ROOT/'results/analysis_upload/replication_seeds0to4_v1'


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def verify(directory):
    directory=Path(directory).resolve()
    manifest=common.read_json(directory/'manifest.json')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
    if actual!=set(manifest['package_files_sha256']):raise ValueError('Package file list changed.')
    for name,digest in manifest['package_files_sha256'].items():
        if common.sha256(directory/name)!=digest:raise ValueError(f'Package checksum mismatch: {name}')
    if any(Path(n).suffix.lower() in ('.pth','.pt','.npz','.zip','.tmp') for n in actual):
        raise ValueError('Weights/replay/bulk archives must remain local.')
    if manifest['seed_count']!=5 or manifest['run_count']!=20:
        raise ValueError('Unexpected five-seed package.')
    if manifest['independent500_evaluated']:
        final=common.read_json(directory/'final500/verification.json')
        if not final['complete'] or final['models']!=20 or final['scene_rows']!=10000:
            raise ValueError('Incomplete final500 confirmation.')
        if final['source_model_sha256']!=manifest['source_models_sha256']:
            raise ValueError('Final500 models do not match packaged learning-curve models.')
    print(f'PASS: {len(actual)} files; {manifest["package_bytes"]/1048576:.2f} MiB; five seeds, no weights/training/push.')
    return manifest


def package(summary,evaluation):
    summary=Path(summary).resolve()
    audited=common.read_json(summary/'verification.json')
    if not audited['passed'] or audited['n']!=5 or len(audited['audits'])!=20 or audited['independent500_used']:
        raise ValueError('Require completed five-seed validation-only audit first.')
    registered=dict(analysis.entry.source_hashes());registered.update(analysis.entry34.source_hashes())
    if registered!=audited['source_sha256']:raise ValueError('Registered training sources changed since analysis.')
    evaluation=Path(evaluation).resolve()
    final=common.read_json(evaluation/'verification.json')
    if not final['complete'] or final['models']!=20 or final['scene_rows']!=10000 or final['used_for_tuning']:
        raise ValueError('Require completed final500 confirmation without tuning.')
    for row in audited['audits']:
        arm='staged' if row['method'] in ('advice_bound_margin','staged_withdrawal') else 'dqfd'
        _,_,baseline,new,method=analysis.directories(arm,row['seed'])
        directory=new if row['method']==method else baseline
        if common.sha256(directory/'model_final.pth')!=row['final_model_sha256']:
            raise ValueError('Final model changed since audit.')
    output=PACKAGE_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    copied={};models={};traces=[];runs=[]
    def copy(source,destination):
        source=Path(source)
        if not source.is_file():raise FileNotFoundError(source)
        copied[relative(source)]=common.sha256(source)
        target=output/destination;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    for p in sorted(summary.iterdir()):
        if p.is_file():copy(p,Path('summary')/p.name)
    report=output/'summary/REPORT.md'
    report.write_text('> 本报告在独立500评估前生成；其中“待评估”为当时状态。当前最终确认已完成，见[独立500报告](../final500/REPORT.md)。\n\n'+report.read_text(encoding='utf-8'),encoding='utf-8')
    for p in sorted(evaluation.iterdir()):
        if p.is_file():copy(p,Path('final500')/p.name)
    for arm in ('staged','dqfd'):
        for seed in range(5):
            config,inputs,baseline,new,new_method=analysis.directories(arm,seed)
            distances=static_distances(inputs[0])
            for method,directory in (('advice_bound_margin' if arm=='staged' else 'dqfd',baseline),(new_method,new)):
                result=common.read_json(directory/'result.json')
                done=common.read_json(directory/'completion.json')
                if not done['complete'] or result['environment_steps']!=200000:
                    raise ValueError('Incomplete run.')
                dest=Path('run_logs')/method/f'seed_{seed}'
                required=('result.json','completion.json','validation_curve.csv','validation_details.csv')
                optional=('run_manifest.json','status.json','training.csv','advice_budget.csv','value_learning.csv',
                    'true_training_td.csv','collection_label_census.csv','training_scenes.csv','dqfd_learning.csv','failure_behaviors.csv')
                for name in required:copy(directory/name,dest/name)
                for name in optional:
                    if name=='training.csv' and directory==baseline:continue
                    if (directory/name).is_file():copy(directory/name,dest/name)
                model=directory/'model_final.pth';models[relative(model)]=common.sha256(model)
                runs.append(dict(method=method,seed=seed,source=relative(directory),environment_steps=200000,
                    gradient_updates=result['gradient_updates'],model_sha256=models[relative(model)],
                    final_success=float(result['final_validation']['safe_success_rate']),complete=True))
                # Each example is the first saved scene per failure category, never a model-selection rule.
                candidates=[('final',200000)]
                if arm=='staged':
                    metrics=load_csv(summary/'per_seed_metrics.csv')
                    record=next(r for r in metrics if r['method']==method and int(r['seed'])==seed)
                    candidates.append(('withdrawal_min',int(record['min_step_80000_160500'])))
                for label,step in candidates:
                    source=directory/f'validation_failures_{step:06d}.json'
                    seen=set()
                    for trace in common.read_json(source):
                        behavior=trace_behavior(trace,distances)
                        counts=failure_counts([behavior])
                        kind='collision' if counts['collision_count'] else 'waiting_timeout' if counts['waiting_timeout_count'] else 'looping_timeout' if counts['looping_timeout_count'] else 'other_timeout'
                        if kind in seen:continue
                        seen.add(kind)
                        target=Path('trajectories')/f'{method}_seed{seed}_{label}_{kind}.json'
                        (output/target).parent.mkdir(parents=True,exist_ok=True)
                        # Preserve every trace field while removing JSON whitespace only.
                        (output/target).write_text(json.dumps(trace,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
                        traces.append(dict(method=method,seed=seed,checkpoint=step,kind=kind,
                            scenario_id=trace['scenario_id'],package_path=target.as_posix(),
                            source=relative(source),source_sha256=common.sha256(source),
                            selection='first saved scene per category; diagnostic only'))
    write_records_csv(runs,output/'run_index.csv')
    # Final500 examples use the same deterministic first-per-category selection rule.
    for origin in load_csv(evaluation/'result_sources.csv'):
        method,seed=origin['method'],int(origin['seed'])
        arm='staged' if method in ('advice_bound_margin','staged_withdrawal') else 'dqfd'
        _,inputs,_,_,_=analysis.directories(arm,seed)
        source=ROOT/origin['trace_source'];seen=set()
        for trace in common.read_json(source):
            counts=failure_counts([trace_behavior(trace,static_distances(inputs[0]))])
            kind='collision' if counts['collision_count'] else 'waiting_timeout' if counts['waiting_timeout_count'] else 'looping_timeout' if counts['looping_timeout_count'] else 'other_timeout'
            if kind in seen:continue
            seen.add(kind);target=Path('trajectories')/f'{method}_seed{seed}_final500_{kind}.json'
            (output/target).parent.mkdir(parents=True,exist_ok=True)
            (output/target).write_text(json.dumps(trace,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
            traces.append(dict(method=method,seed=seed,checkpoint=200000,kind=kind,
                scenario_id=trace['scenario_id'],package_path=target.as_posix(),source=relative(source),
                source_sha256=common.sha256(source),selection='first saved scene per category; final500 diagnostic only'))
    write_records_csv(traces,output/'trajectory_index.csv')
    if models!=final['source_model_sha256']:raise ValueError('Evaluation used other checkpoints.')
    for protocol in (analysis.entry,analysis.entry34):
        for arm in ('staged','dqfd'):
            config,_,_=protocol.configure(arm)
            p=ROOT/config['replication']['registration_root']/'freeze_manifest.json'
            frozen=common.read_json(p)
            if frozen['source_sha256']!=protocol.source_hashes():raise ValueError('Registration changed.')
            copy(p,Path('registration')/p.parent.name/p.name)
    # Preserve seed0 and original DQfD registration context, without duplicating demo arrays or final500 scenes.
    for p in (ROOT/'data/whole_map_91701_staged_withdrawal_pilot_v1/freeze_manifest.json',
              ROOT/'data/whole_map_91701_dqfd_no_pretrain_v1/freeze_manifest.json',
              ROOT/'data/whole_map_91701_dqfd_comparison_v1/freeze_manifest.json'):
        copy(p,Path('registration')/p.parent.name/p.name)
    config,_,_=analysis.entry.configure('staged')
    validation=ROOT/config['dataset']['validation_reference']
    copy(validation,Path('inputs')/'fixed_validation_scenarios.json')
    tail=analysis.entry.dq.tail
    tail_cfg=analysis.legacy.load_config(tail.CONFIG)
    scene_file=tail.artifact_root(tail_cfg)/'independent_final_scenarios.json'
    if common.sha256(scene_file)!=final['frozen_scene_sha256']:raise ValueError('Independent500 scenes changed.')
    copy(scene_file,Path('inputs')/'independent_final_scenarios.json')
    extras=['scripts/analyze_replication_seeds012.py','scripts/analyze_replication_seeds01234.py',
        'scripts/evaluate_replication_final500.py','scripts/finalize_replication_final500.py',
        'scripts/package_replication_five_seeds.py','scripts/sync_github.ps1','docs/REPLICATION_FIVE_SEEDS.zh-CN.md']
    sources={str(p).replace('\\','/') for p in registered}|set(extras)
    for name in sorted(sources):copy(ROOT/name,Path('source_snapshot')/name)
    (output/'README.md').write_text('''# 分阶段撤教与DQfD无预训练：五seed阅读包

先读[五seed报告](summary/REPORT.md)、[均值和样本标准差](summary/mean_sample_sd.csv)、[同seed配对差](summary/paired_differences.csv)及[曲线](summary/paired_validation_curves.png)。统计单位为seed，不是50验证回合。

两种新增方法各seed0–4全部完成200000在线步；原完整组合与原DQfD五seed复用。共有20次运行、420验证点、21000条逐场景验证记录；20个最终模型重新复核原50场景。四方法同一冻结500场景最终确认已完成，原两方法按SHA及协议核对复用，新两方法补评5000回合。没有新训练或最佳模型选择。

分阶段日程：0–60000与原组合一致；60000–80000代选概率由0.32线性降至0；80000后代选为0；监督按原日程在100000完全退出。TD约束、风险筛选、标签规则及训练设置不变。预设代选日程总量减少8%，不是等教师量对照。

DQfD对照只取消20000次离线更新。仍使用同一7260条冻结示范、永久示范分区、PER、多步TD、margin、L2及原在线条件，无TD裁剪。原DQfD总219501次更新，对照199501次。原DQfD的0步验证已经预训练；另外报告1–8万AULC。

`run_logs/`包含所有五seed曲线、场景明细、已有诊断与成本日志，以及十次新增对照的训练CSV；原方法训练CSV此前已上传，不重复复制。`source_snapshot/`保存对应实现和配置；`registration/`保留冻结登记；`trajectories/`按固定规则保存失败示例，仅移除JSON空白、不删字段。未来障碍轨迹仅用于事后诊断。全部失败分类在summary/checkpoint_failures.csv，示例不代表独立样本或模型筛选。

新增权重、回放恢复文件、批量失败轨迹和临时文件留在本地。最终模型准确路径和SHA在manifest.json；仅克隆仓库不能重新执行模型评估，缺权重不得自动重训。示范仍在仓库data/whole_map_91701_dqfd_comparison_v1/demonstrations.npz，不重复复制。

独立500最终确认见[最终报告](final500/REPORT.md)、[逐seed结果](final500/summary.csv)、[均值与样本标准差](final500/method_means.csv)和[配对差](final500/paired_difference_means.csv)。同批500场景合并10000条明细，不与50验证成功率混比，也不据此调参。summary中的50场景报告保存了评估前的“待评估”历史状态，当前完成状态以final500/verification.json为准。三seed报告作为历史保留。

[给GPT的分析请求](GPT_ANALYSIS_PROMPT.md)。文件完整性：`python scripts/package_replication_five_seeds.py --verify <本包目录>`，不训练、不推送。
''',encoding='utf-8')
    (output/'GPT_ANALYSIS_PROMPT.md').write_text('''请基于summary/REPORT.md、mean_sample_sd.csv、paired_differences.csv、validation_curves.csv和对应source_snapshot分析两个五seed对照；按需查看日志和轨迹，不逐一展开所有大文件。
1. 分阶段撤教的全程及后半AULC增益是否跨seed一致？seed4接近零的差值不要包装成明显收益。
2. 对比8万、10万退出窗口及8–16万最低点、恢复时钟和等待/绕圈/碰撞；检查seed2、3、4低谷是否加深或移动，不能只看最终成功率。
3. 取消DQfD预训练比较相同在线交互、不同总梯度预算，仍有7260永久示范；区分0步初始化优势与1–8万、10–20万持续表现。保留DQfD强基线。
4. 明确五seed、配对差与样本标准差。原50验证场景反复评估不是独立测试样本；独立500同场景最终确认见final500，检查场景/模型SHA和旧结果复用来源，不与验证曲线混比、不据测试调参。
5. 结合此前动作监督与TD约束的消融证据，提出论文贡献及最少的下一步；不主张保证安全、退出低谷已解决或最终全面最优，不改地图/奖励迎合结果。
缺失项写“缺失/未确认”，引用包内准确路径。不要建议挑最佳检查点或基于500结果调参。
''',encoding='utf-8')
    git=shutil.which('git') or 'D:/softwares/Anaconda3/envs/yolov5/Library/cmd/git.exe'
    def git_read(*args):return subprocess.check_output([git,'-c','safe.directory='+ROOT.as_posix(),*args],cwd=ROOT,text=True).strip()
    files={p.relative_to(output).as_posix():common.sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(dict(base_commit=git_read('rev-parse','HEAD'),cached_origin_main=git_read('rev-parse','origin/main'),
        remote_state_live_verified=False,summary_source=relative(summary),seed_count=5,run_count=20,
        statistical_unit='seed',source_files_sha256=copied,source_models_sha256=models,
        package_files_sha256=files,package_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        independent500_evaluated=True,final_evaluation_source=relative(evaluation),
        frozen_final_scene_sha256=final['frozen_scene_sha256'],
        weights_uploaded=False,training_started=False,push_performed=False),output/'manifest.json')
    verify(output)
    write_json(dict(package=relative(output),summary=relative(summary),evaluation=relative(evaluation)),PACKAGE_ROOT/'latest.json')
    (PACKAGE_ROOT/'LATEST.md').write_text(f'# 最新五seed阅读包\n\n[阅读入口]({output.name}/README.md) · [学习曲线报告]({output.name}/summary/REPORT.md) · [独立500最终确认]({output.name}/final500/REPORT.md) · [GPT分析请求]({output.name}/GPT_ANALYSIS_PROMPT.md)\n\n原50验证与同一冻结500场景最终确认均已完成。\n',encoding='utf-8')
    print(output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',type=Path);parser.add_argument('--summary',type=Path);parser.add_argument('--evaluation',type=Path)
    args=parser.parse_args()
    if args.verify:verify(args.verify);return
    summary=args.summary
    if summary is None:
        candidates=sorted((ROOT/'results/replication_seeds01234_v1').glob('*/verification.json'))
        completed=[p.parent for p in candidates if common.read_json(p).get('passed') and common.read_json(p).get('n')==5]
        if not completed:raise FileNotFoundError('Run scripts/analyze_replication_seeds01234.py first; no auto training/evaluation.')
        summary=completed[-1]
    evaluation=args.evaluation
    if evaluation is None:
        candidates=sorted((ROOT/'results/replication_final500_v1').glob('*/verification.json'))
        completed=[p.parent for p in candidates if common.read_json(p).get('complete') and common.read_json(p).get('models')==20]
        if not completed:raise FileNotFoundError('Complete scripts/evaluate_replication_final500.py --evaluate before packaging/upload; no auto evaluation.')
        evaluation=completed[-1]
    package(summary,evaluation)


if __name__=='__main__':main()
