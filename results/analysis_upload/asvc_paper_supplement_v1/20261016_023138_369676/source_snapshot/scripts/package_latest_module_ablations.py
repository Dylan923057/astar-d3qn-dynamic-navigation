"""Package all completed/failed registered seeds and fixed50/final500 evidence; no training or push."""
from datetime import datetime
import json
from pathlib import Path
import shutil

import latest_module_common as protocol
from analyze_runtime_path_pilot import load_csv,static_distances,trace_behavior
from analyze_supervision_tail import failure_counts
from astar_d3qn.utils.io import write_json,write_records_csv

ROOT,common=protocol.ROOT,protocol.common
PACKAGE_ROOT=ROOT/'results/analysis_upload/latest_all_eligible_modules_v1'


def package():
    summary=ROOT/common.read_json(ROOT/'results/latest_all_eligible_modules_analysis_v1/latest.json')['analysis']
    evaluation=ROOT/common.read_json(ROOT/'results/latest_all_eligible_modules_final500_v1/latest.json')['evaluation']
    audit=common.read_json(summary/'verification.json');final=common.read_json(evaluation/'verification.json')
    if not audit['passed'] or not final['complete']:raise ValueError('Finish fixed analysis/evaluation first.')
    if audit['final500_verification_sha256']!=common.sha256(evaluation/'verification.json'):raise ValueError('Final500 changed since analysis.')
    output=PACKAGE_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    sources={};runs=[];traces=[];models={}
    def relative(path):return Path(path).relative_to(ROOT).as_posix()
    def copy(source,destination):
        source=Path(source);target=output/destination;target.parent.mkdir(parents=True,exist_ok=True)
        if not source.is_file():raise FileNotFoundError(source)
        sources[relative(source)]=common.sha256(source);shutil.copyfile(source,target)
    for folder,destination in ((summary,'learning50'),(evaluation,'final500')):
        for path in folder.iterdir():
            if path.is_file():copy(path,Path(destination)/path.name)
    metrics=load_csv(summary/'per_seed_metrics.csv')
    for method in protocol.ALL:
        for seed in range(5):
            p=protocol.run(method,seed);directory=p['directory'];dest=Path('run_logs')/method/f'seed_{seed}'
            names=('run_manifest.json','result.json','completion.json','status.json','nonfinite_failure.json',
                'validation_curve.csv','validation_details.csv','advice_budget.csv','value_learning.csv','supervision_alignment.csv','dqfd_learning.csv')
            for name in names:
                if (directory/name).is_file():copy(directory/name,dest/name)
            if method in protocol.core.METHODS and (directory/'training.csv').is_file():copy(directory/'training.csv',dest/'training.csv')
            runs.append(dict(method=method,seed=seed,completed=not p['failed'],numerical_failure=p['failed'],source=relative(directory)))
            if p['failed']:continue
            models[relative(p['model'])]=p['digest']
            if method not in protocol.MAIN:continue
            metric=next(r for r in metrics if r['method']==method and int(r['seed'])==seed)
            candidates={200000:'final50',int(metric['min_step_80000_160500']):'withdrawal_min'}
            for step,label in candidates.items():
                source=directory/f'validation_failures_{step:06d}.json';seen=set()
                for trace in common.read_json(source):
                    counts=failure_counts([trace_behavior(trace,static_distances(p['inputs'][0]))])
                    kind='collision' if counts['collision_count'] else 'waiting_timeout' if counts['waiting_timeout_count'] else 'looping_timeout' if counts['looping_timeout_count'] else 'other_timeout'
                    if kind in seen:continue
                    seen.add(kind);dest=Path('trajectories')/f'{method}_seed{seed}_{label}_{kind}.json'
                    (output/dest).parent.mkdir(parents=True,exist_ok=True)
                    (output/dest).write_text(json.dumps(trace,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
                    traces.append(dict(method=method,seed=seed,scene_set='validation50',step=step,kind=kind,
                        scenario_id=trace['scenario_id'],package_path=dest.as_posix(),source_sha256=common.sha256(source)))
    if models!=final['source_model_sha256']:raise ValueError('Model set changed after final500.')
    for origin in load_csv(evaluation/'result_sources.csv'):
        if origin['method'] not in protocol.MAIN:continue
        p=protocol.run(origin['method'],int(origin['seed']));source=ROOT/origin['trace_source'];seen=set()
        for trace in common.read_json(source):
            counts=failure_counts([trace_behavior(trace,static_distances(p['inputs'][0]))])
            kind='collision' if counts['collision_count'] else 'waiting_timeout' if counts['waiting_timeout_count'] else 'looping_timeout' if counts['looping_timeout_count'] else 'other_timeout'
            if kind in seen:continue
            seen.add(kind);dest=Path('trajectories')/f'{p["method"]}_seed{p["seed"]}_final500_{kind}.json'
            (output/dest).parent.mkdir(parents=True,exist_ok=True)
            (output/dest).write_text(json.dumps(trace,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
            traces.append(dict(method=p['method'],seed=p['seed'],scene_set='independent500',step=200000,kind=kind,
                scenario_id=trace['scenario_id'],package_path=dest.as_posix(),source_sha256=common.sha256(source)))
    write_records_csv(runs,output/'run_index.csv');write_records_csv(traces,output/'trajectory_index.csv')
    for path in (ROOT/'data'/protocol.core.NAME).rglob('*.json'):
        if 'source_snapshot' not in path.parts:copy(path,Path('registration')/path.relative_to(ROOT/'data'/protocol.core.NAME))
    cfg,_=protocol.core.configure(protocol.core.METHODS[0])
    copy(ROOT/cfg['dataset']['validation_reference'],Path('inputs/fixed_validation_scenarios.json'))
    copy(ROOT/final['scene_file'],Path('inputs/independent_final_scenarios.json'))
    registered=protocol.core.source_hashes()
    extras={'scripts/run_latest_module_suite.py','scripts/latest_module_common.py','scripts/evaluate_latest_module_ablations.py',
        'scripts/analyze_latest_module_ablations.py','scripts/package_latest_module_ablations.py',
        'scripts/finalize_latest_module_suite.py'}
    for name in sorted(set(registered)|extras):copy(ROOT/name,Path('source_snapshot')/name)
    (output/'README.md').write_text('''# 最新完整方法：模块消融阅读包

[学习过程与模块报告](learning50/REPORT.md)、[逐seed](learning50/per_seed_metrics.csv)、[配对差](learning50/paired_differences.csv)、[配对均值/样本标准差](learning50/paired_means.csv)、[曲线](learning50/paired_validation_curves.png)。独立500结果在final500/summary.csv和method_means.csv，与原50训练曲线分开报告。

主对比：最新all-eligible完整组、新去TD组、新去A*代选组、已有advice_bound、原始D3QN（unguided_raw）。原筛选完整组和DQfD仅作背景并保留；DQfD有永久示范/PER和20000离线更新，总训练预算不同。

只新增两组各seed0–4，固定200000在线环境步；监督独立教师动作、TD真实动作。关闭代选仍推进监督时钟；去TD仅关闭完整目标[-6,10]裁剪；保留Smooth L1、梯度范数裁剪和全部其他设置。监督与有代选组在100000退出，不补偿标签数量，不增加示范或预训练。冻结源码及配置见registration/和source_snapshot/。

全部seed保留，数值失败写明准确发生步数并报告缺失，不能删seed或用剩余样本冒充五seed均值。所有最终评估仅200000模型、ε=0、关闭教师/路线引导/动态保护/Q输出裁剪，共同静态合法动作掩码；同一冻结500不用于调参。旧结果仅模型/场景/协议一致时复用。

阶段监督覆盖、执行一致率、margin、权重、Q、原始TD目标范围、裁剪及实际代选率在learning50/phase_mechanisms.csv；旧未记录字段保持缺失。监督比例分母为回放抽样次数，非独立转移数。首达90%和连续三点90%分开，未达到保持缺失。未来障碍信息仅用于事后轨迹诊断。

失败示例是低谷/最终处每类别首个保存场景，不选择模型；原始轨迹和权重/回放断点留在本地，模型精确路径/SHA见manifest.json。克隆缺权重不自动重训。旧实验与旧阅读包保留，没有参数搜索或额外方法训练。
''',encoding='utf-8')
    hashes={p.relative_to(output).as_posix():common.sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(dict(source_model_sha256=models,source_files_sha256=sources,package_files_sha256=hashes,
        statistical_unit='seed',planned_training_seeds=5,new_training_runs=10,independent500_evaluated=True,
        failed_training_runs=final['failed_training_runs'],summary_source=relative(summary),evaluation_source=relative(evaluation)),output/'manifest.json')
    for name,digest in hashes.items():
        if common.sha256(output/name)!=digest:raise ValueError('Package hash differs.')
    write_json(dict(package=relative(output)),PACKAGE_ROOT/'latest.json')
    (PACKAGE_ROOT/'LATEST.md').write_text(f'# 最新模块消融阅读包\n\n[阅读入口]({output.name}/README.md) · [报告]({output.name}/learning50/REPORT.md)\n',encoding='utf-8')
    print(f'{relative(output)} ({len(hashes)} files)',flush=True);return output


if __name__=='__main__':package()
