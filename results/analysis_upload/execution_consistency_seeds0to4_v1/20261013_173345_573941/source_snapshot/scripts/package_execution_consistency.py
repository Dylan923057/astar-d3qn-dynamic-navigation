"""Pack the audited five-seed ablation for GitHub; no training, commit or push."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import analyze_execution_consistency_five_seeds as analysis
from analyze_runtime_path_pilot import load_csv,trace_behavior,static_distances
from analyze_supervision_tail import failure_counts
from astar_d3qn.utils.io import write_json,write_records_csv

ROOT,common=analysis.ROOT,analysis.common
PACKAGE_ROOT=ROOT/'results/analysis_upload/execution_consistency_seeds0to4_v1'


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def verify(directory):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(PACKAGE_ROOT.resolve()):raise ValueError('Unexpected package location.')
    manifest=common.read_json(directory/'manifest.json')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
    if actual!=set(manifest['package_files_sha256']):raise ValueError('Package file list changed.')
    for name,digest in manifest['package_files_sha256'].items():
        if common.sha256(directory/name)!=digest:raise ValueError(f'Package hash changed: {name}')
        if Path(name).suffix.lower() in ('.pth','.pt','.npz','.pkl','.zip'):raise ValueError('Unnecessary binary in package.')
    runs=load_csv(directory/'run_index.csv')
    expected={(method,seed) for method in ('advice_bound_margin',analysis.entry.METHOD) for seed in range(5)}
    if {(r['method'],int(r['seed'])) for r in runs}!=expected or len(runs)!=10:
        raise ValueError('Expected ten completed paired runs.')
    for row in runs:
        base=directory/'run_logs'/row['method']/f'seed_{row["seed"]}'
        if not common.read_json(base/'completion.json')['complete']:raise ValueError('Incomplete packaged run.')
        result=common.read_json(base/'result.json')
        if result['environment_steps']!=200000 or result['gradient_updates']!=199501:raise ValueError('Training budget differs.')
        if len(load_csv(base/'validation_curve.csv'))!=21 or len(load_csv(base/'validation_details.csv'))!=1050:
            raise ValueError('Incomplete validation records.')
    if manifest['independent500_evaluated'] or manifest['statistical_unit']!='seed':raise ValueError('Incorrect evaluation boundary.')
    audit=common.read_json(directory/'summary/verification.json')
    if not audit['passed'] or audit['n']!=5 or len(audit['audits'])!=10:raise ValueError('Missing five-seed audit.')
    print(f'Package verified: {relative(directory)} ({len(actual)} files)',flush=True)
    return manifest


def package(summary=None):
    if summary is None:
        choices=sorted(p for p in analysis.analysis.ANALYSIS_ROOT.iterdir() if (p/'verification.json').is_file())
        if not choices:raise FileNotFoundError('Run five-seed analysis before packaging.')
        summary=choices[-1]
    summary=Path(summary).resolve()
    audit=common.read_json(summary/'verification.json')
    if not audit['passed'] or audit['n']!=5 or len(audit['audits'])!=10:raise ValueError('Use completed five-seed audit.')
    if audit['analysis_source_sha256']!=common.sha256(ROOT/'scripts/analyze_execution_consistency.py') or audit['analysis_entry_sha256']!=common.sha256(Path(analysis.__file__)):
        raise ValueError('Analysis source changed after audit.')
    output=PACKAGE_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    sources={};runs=[];models={};examples=[];trace_index=[]
    def copy(source,destination):
        source=Path(source)
        if not source.is_file():raise FileNotFoundError(source)
        sources[relative(source)]=common.sha256(source)
        target=output/destination;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    for path in summary.iterdir():
        if path.is_file():copy(path,Path('summary')/path.name)
    metrics=load_csv(summary/'per_seed_metrics.csv')
    for seed in range(5):
        config,inputs,old,new,method=analysis.directories('staged',seed)
        distances=static_distances(inputs[0])
        for name,directory in (('advice_bound_margin',old),(method,new)):
            result=common.read_json(directory/'result.json')
            done=common.read_json(directory/'completion.json')
            if not done['complete'] or result['environment_steps']!=200000:raise ValueError('Incomplete source run.')
            model=directory/'model_final.pth'
            expected=next(r['final_model_sha256'] for r in audit['audits'] if r['method']==name and r['seed']==seed)
            if common.sha256(model)!=expected:raise ValueError('Final model changed after audit.')
            models[relative(model)]=expected
            dest=Path('run_logs')/name/f'seed_{seed}'
            for filename in ('result.json','completion.json','validation_curve.csv','validation_details.csv'):
                copy(directory/filename,dest/filename)
            for filename in ('run_manifest.json','status.json','advice_budget.csv','value_learning.csv','supervision_alignment.csv','failure_behaviors.csv'):
                if (directory/filename).is_file():copy(directory/filename,dest/filename)
            if directory==new:copy(directory/'training.csv',dest/'training.csv')
            runs.append(dict(method=name,seed=seed,source=relative(directory),complete=True,
                environment_steps=200000,gradient_updates=result['gradient_updates'],model_sha256=expected,
                final_success=result['final_validation']['safe_success_rate']))
            metric=next(r for r in metrics if r['method']==name and int(r['seed'])==seed)
            example_steps={200000:'final',int(metric['min_step_80000_160500']):'withdrawal_min'}
            curve=load_csv(directory/'validation_curve.csv')
            for row in curve:
                step=int(row['environment_steps_total']);path=directory/f'validation_failures_{step:06d}.json'
                traces=common.read_json(path)
                trace_index.append(dict(method=name,seed=seed,step=step,count=len(traces),
                    source=relative(path),sha256=common.sha256(path)))
                if step not in example_steps:continue
                seen=set()
                for trace in traces:
                    counts=failure_counts([trace_behavior(trace,distances)])
                    kind='collision' if counts['collision_count'] else 'waiting_timeout' if counts['waiting_timeout_count'] else 'looping_timeout' if counts['looping_timeout_count'] else 'other_timeout'
                    if kind in seen:continue
                    seen.add(kind)
                    dest=Path('trajectories')/f'{name}_seed{seed}_{example_steps[step]}_{kind}.json'
                    (output/dest).parent.mkdir(parents=True,exist_ok=True)
                    (output/dest).write_text(json.dumps(trace,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
                    examples.append(dict(method=name,seed=seed,step=step,kind=kind,scenario_id=trace['scenario_id'],
                        package_path=dest.as_posix(),selection='first saved scenario per category at final/trough; diagnostic only'))
    write_records_csv(runs,output/'run_index.csv')
    write_records_csv(trace_index,output/'raw_trace_index.csv')
    write_records_csv(examples,output/'trajectory_index.csv')
    registered={}
    for seed in (0,3):
        config,freeze=analysis.protocol(seed)
        registered.update(freeze['source_sha256'])
        path=ROOT/config['execution_consistency']['registration_root']/'freeze_manifest.json'
        copy(path,Path('registration')/path.parent.name/path.name)
    config,_=analysis.protocol(0)
    copy(ROOT/config['dataset']['validation_reference'],Path('inputs/fixed_validation_scenarios.json'))
    extras={'scripts/analyze_execution_consistency.py','scripts/analyze_execution_consistency_five_seeds.py',
        'scripts/package_execution_consistency.py','scripts/sync_github.ps1','docs/EXECUTION_CONSISTENCY.zh-CN.md'}
    for name in sorted(set(registered)|extras):copy(ROOT/name,Path('source_snapshot')/name)
    (output/'README.md').write_text('''# 执行一致性筛选：五seed阅读包

先读[五seed报告](summary/REPORT.md)、[逐seed结果](summary/per_seed_metrics.csv)、[配对差](summary/paired_differences.csv)、[配对均值与样本标准差](summary/paired_mean_sample_sd.csv)及[曲线](summary/paired_validation_curves.png)。统计单位seed，样本标准差ddof=1。

原完整组合advice_bound_margin五seed复用；新组只取消辅助监督的实际执行动作=A*建议条件。TD使用真实动作、真实奖励/下一状态/终止标记，margin使用独立教师动作。不改其他标签规则、风险筛选、权重、采样、奖励或日程；代选与监督在10万步同时退出。不是分阶段撤教方案。

五次新训练均200000环境步、199501在线更新，没有离线示范或预训练。全部10个最终模型复评原50场景，210个验证点、10500条场景记录全部核对。不选择最佳检查点。新对照尚未评估独立500场景，本包只支持原50验证协议下的结论。

`run_logs/`保留全部曲线、逐场景结果、已有机制日志和新组训练CSV；`source_snapshot/`保留对应源码/配置；`registration/`保留原三seed和新增两seed冻结登记。监督比例按回放采样次数计算，不能解释为不同转移的覆盖率。原组监督一致率由标签规则推得100%，有符号Q范围未记录，缺失不能填零。

`trajectories/`保留最终和撤教低谷的首个各类失败示例，完整字段仅压缩JSON空白；选择规则不筛选模型。全部失败计数见summary/checkpoint_failures.csv，原始完整轨迹路径及SHA见raw_trace_index.csv。未来障碍信息仅用于事后诊断。

权重和回放断点留在本地，准确最终权重路径/SHA见manifest.json。克隆后缺失权重不能自动重训。旧结果及三seed历史报告保留；五seed结论优先看本包，不沿用三seed结论。

完整性检查：`python scripts/package_execution_consistency.py --verify <本包路径>`。给GPT的分析请求见[GPT_ANALYSIS_PROMPT.md](GPT_ANALYSIS_PROMPT.md)。
''',encoding='utf-8')
    (output/'GPT_ANALYSIS_PROMPT.md').write_text('''请仅依据此包五seed证据分析执行一致性筛选是否有效。原完整组合为10万步代选/监督退出，新组只取消辅助监督实际动作与教师动作相同的条件；TD依然使用真实动作，监督使用独立教师动作。请核对源码、冻结登记、全部逐seed曲线及配对统计，不只看最终点。

比较前5万、全程及10–20万AULC、首次90%与连续三点90%、撤教最低点时间及恢复、最后五次均值、最终碰撞和等待/绕圈超时。关注seed3、4是否改变三seed初步结论。结合监督样本比例、监督中执行一致率、margin、权重、Q值、TD裁剪和实际代选率解释机制，区分证据与假说。

监督数量变化是该单因素的机制变化；不能宣称已分离样本数量与不同动作标签的作用。统计单位seed，重复50场景不能当新增独立样本。新对照未做独立500确认，不据验证结果调参，不推断地图泛化或保证安全。指出论文可支持的贡献及最少还缺的证据，不删任何seed或DQfD强基线。
''',encoding='utf-8')
    files={p.relative_to(output).as_posix():common.sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(dict(statistical_unit='seed',n=5,summary_source=relative(summary),runs=runs,
        independent500_evaluated=False,validation_scenarios=50,validation_points=210,validation_episodes=10500,
        source_model_sha256=models,source_files_sha256=sources,package_files_sha256=files,
        weights_included=False,old_results_preserved=True),output/'manifest.json')
    verify(output)
    write_json(dict(package=relative(output)),PACKAGE_ROOT/'latest.json')
    (PACKAGE_ROOT/'LATEST.md').write_text(f'# 最新五seed阅读包\n\n[阅读入口]({output.name}/README.md) · [报告]({output.name}/summary/REPORT.md) · [GPT分析请求]({output.name}/GPT_ANALYSIS_PROMPT.md)\n',encoding='utf-8')
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    print(f'{relative(output)}; {size/1024**2:.2f} MiB',flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',type=Path);parser.add_argument('--analysis',type=Path)
    args=parser.parse_args()
    if args.verify:verify(args.verify)
    else:package(args.analysis)
