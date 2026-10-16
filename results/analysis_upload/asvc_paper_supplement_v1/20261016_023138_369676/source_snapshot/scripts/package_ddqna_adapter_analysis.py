"""Audit completed DDQNA adaptation and prepare a compact GitHub reading package."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import torch

import ddqna_adapter_common as common
import report_ddqna_adapter as report
from analyze_runtime_path_pilot import static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from package_dqfd_comparison_analysis import read_csv, write_csv, write_json, relative, verify


def latest(prefix):
    root=common.ROOT/'results/whole_map_91701_ddqna_adapter_v1'
    choices=[p for p in sorted(root.glob(prefix+'_*')) if (p/'verification.json').is_file()
             and common.read_json(p/'verification.json').get('complete')]
    if not choices:raise FileNotFoundError(f'No completed {prefix}.')
    return choices[-1]


def category(b):
    return ('collision' if b['termination_reason']=='collision' else
            'waiting_timeout' if b['waiting_dominated'] else
            'looping_timeout' if b['repeated_movement'] or b['tail_period'] else 'other_timeout')


def same_csv_numbers(saved,actual):
    if len(saved)!=len(actual):raise ValueError('Summary row count changed.')
    for a,b in zip(saved,actual):
        if set(a)!=set(b):raise ValueError('Summary columns changed.')
        for key,value in b.items():
            if value is None:
                if a[key]!='':raise ValueError('Missing value changed.')
            elif isinstance(value,(bool,np.bool_)):
                if a[key]!=str(bool(value)):raise ValueError(f'Boolean differs: {key}')
            elif isinstance(value,(int,float,np.number)):
                if not np.isclose(float(a[key]),value):raise ValueError(f'Metric differs: {key}')
            elif str(a[key])!=str(value):raise ValueError(f'Identity differs: {key}')


def audit(config,analysis,evaluation):
    av,ev=(common.read_json(p/'verification.json') for p in (analysis,evaluation))
    if (not av['complete'] or av['models']!=10 or av['checkpoint_count']!=210 or
            not ev['complete'] or ev['models']!=10 or ev['scene_rows']!=5000 or ev['scene_count']!=500 or
            ev['evaluation_epsilon']!=0 or ev['action_override'] or ev['dynamic_protection'] or
            not ev['static_legal_mask'] or ev['future_obstacles_used_for_actions'] or
            ev['checkpoint_selection']!='final_200000_only' or ev['test_used_for_tuning']):
        raise ValueError('Incomplete fixed protocol.')
    scenes,_,frozen=common.tail.load_frozen(common.legacy.load_config(common.tail.CONFIG))
    if ev['frozen_scene_sha256']!=frozen['files_sha256']['independent_final_scenarios.json']:
        raise ValueError('Final scene set changed.')
    raw=common.ROOT/ev['full_failure_trace_root']
    if common.read_json(raw/'completion.json')!=ev:raise ValueError('Raw evaluation is incomplete.')
    distances=static_distances(common.validate_config(config)[0])
    training,checkpoints,failures,costs=[],[],[],[]
    for method in report.METHODS:
        for seed in common.SEEDS:
            summary,curve,behavior,cost=report.collect(config,method,seed,distances)
            training.append(summary);checkpoints.extend(curve);failures.extend(behavior);costs.append(cost)
            print(f'Audited learning {method} seed={seed}: {len(behavior)} failures',flush=True)
    same_csv_numbers(read_csv(analysis/'summary.csv'),training)
    same_csv_numbers(read_csv(analysis/'all_validation_checkpoints.csv'),checkpoints)
    same_csv_numbers(read_csv(analysis/'failure_behaviors.csv'),failures)
    same_csv_numbers(read_csv(analysis/'training_cost_summary.csv'),costs)
    if len(failures)!=av['inspected_failures']:raise ValueError('Training failure audit changed.')
    grouped=defaultdict(list)
    for row in read_csv(evaluation/'scene_details.csv'):grouped[row['method'],int(row['seed'])].append(row)
    if set(grouped)!={(m,s) for m in report.METHODS for s in common.SEEDS}:
        raise ValueError('Extra/missing final scene model groups.')
    saved_behaviors=read_csv(evaluation/'failure_behaviors.csv')
    recomputed_behaviors,summaries,selected,raw_hashes=[],[],[],{}
    for method in report.METHODS:
        for seed in common.SEEDS:
            directory,result=common.audit_run(config,method,seed)
            model=directory/'model_final.pth'
            if ev['source_model_sha256'].get(str(model.relative_to(common.ROOT)))!=common.sha256(model):
                raise ValueError('Final model hash changed.')
            rows=grouped[method,seed]
            if ([r['scenario_id'] for r in rows]!=[s['scenario_id'] for s in scenes] or
                    any(r['dynamic_route_ids'].split(';')!=s['route_ids'] or
                        int(r['dynamic_obstacle_count'])!=len(s['obstacles']) for r,s in zip(rows,scenes)) or
                    any(float(r['static_collision']) for r in rows)):
                raise ValueError('Final scene/mask mismatch.')
            path=raw/method/f'seed_{seed}'/'failure_trajectories.json'
            raw_hashes[relative(path)]=common.sha256(path)
            traces=common.read_json(path)
            failed={r['scenario_id']:r for r in rows if not float(r['safe_success'])}
            if len(traces)!=len(failed) or {t['scenario_id'] for t in traces}!=set(failed):
                raise ValueError('Final full failures missing/duplicated.')
            inspected=[];seen=set()
            for trace in traces:
                b=trace_behavior(trace,distances);row=failed[trace['scenario_id']]
                if (b['termination_reason']!=row['termination_reason'] or b['steps']!=int(row['steps']) or
                        b['steps']>300 or b['termination_reason']=='timeout' and b['steps']!=300):
                    raise ValueError('Final failure reason/length mismatch.')
                inspected.append(b)
                recomputed_behaviors.append(dict(method=method,seed=seed,**{k:v for k,v in b.items() if k!='tail_60_steps'}))
                kind=category(b)
                if kind not in seen:
                    selected.append((method,seed,200000,'final500',kind,path,trace));seen.add(kind)
            summaries.append(dict(method=method,seed=seed,
                safe_success_rate=float(np.mean([float(r['safe_success']) for r in rows])),
                dynamic_collision_rate=float(np.mean([float(r['dynamic_collision']) for r in rows])),
                timeout_rate=float(np.mean([r['termination_reason']=='timeout' for r in rows])),**failure_counts(inspected)))
            del traces,inspected
            print(f'Audited {method} seed={seed}: {len(failed)} final failures',flush=True)
    same_csv_numbers(read_csv(evaluation/'summary.csv'),summaries)
    same_csv_numbers(saved_behaviors,recomputed_behaviors)
    # Predetermined diagnostic selection: first failure of each category per seed,
    # nearest recorded checkpoint >=100k; selection never alters aggregate metrics.
    for seed in common.SEEDS:
        directory,_=common.audit_run(config,common.METHOD,seed)
        step=next(int(r['environment_steps_total']) for r in read_csv(directory/'validation_curve.csv')
                  if int(r['environment_steps_total'])>=100000)
        path=directory/f'validation_failures_{step:06d}.json';seen=set()
        for trace in common.read_json(path):
            kind=category(trace_behavior(trace,distances))
            if kind not in seen:
                selected.append((common.METHOD,seed,step,'original50',kind,path,trace));seen.add(kind)
    return dict(complete=True,new_training_runs=5,final_models=10,training_checkpoints=210,
        training_failures_recomputed=len(failures),final_scene_rows=5000,
        final_failures_recomputed=len(recomputed_behaviors),source_model_sha256=ev['source_model_sha256'],
        frozen_scene_sha256=ev['frozen_scene_sha256'],raw_final_failure_sha256=raw_hashes,
        statistical_unit='five paired training seeds',training_started=False,
        evaluation_rerun=False,source_weights_uploaded=False),selected


def figures(analysis,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped=defaultdict(list)
    for r in read_csv(analysis/'all_validation_checkpoints.csv'):
        grouped[r['method'],int(r['seed'])].append(r)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,method in zip(axes,report.METHODS):
        for seed in common.SEEDS:
            rows=grouped[method,seed]
            ax.plot([int(r['environment_steps_total'])/1000 for r in rows],
                [100*float(r['safe_success_rate']) for r in rows],label=f'seed {seed}')
        ax.set(title=method,xlabel='Online environment steps (thousands)',ylim=(-2,102))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Autonomous safe success on original 50 (%)')
    fig.tight_layout();fig.savefig(output/'paired_seed_learning_curves.png',dpi=160)
    fig.savefig(output/'paired_seed_learning_curves.pdf');plt.close(fig)


def package(analysis,evaluation):
    config=common.legacy.load_config(common.CONFIG)
    common.register(config)
    checked,selected=audit(config,analysis,evaluation)
    name='whole_map_91701_ddqna_adapter_seeds0to4_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=common.ROOT/'results/analysis_upload'/name;output.mkdir(parents=True,exist_ok=False)
    external={}
    def copy(path,target):
        external[relative(path)]=common.sha256(path)
        dest=output/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    for label,directory in (('learning',analysis),('final500',evaluation)):
        for path in directory.iterdir():
            if path.is_file():copy(path,Path(label)/path.name)
    for seed in common.SEEDS:
        directory,_=common.audit_run(config,common.METHOD,seed)
        for filename in ('run_manifest.json','result.json','completion.json','training.csv',
                         'validation_curve.csv','validation_details.csv','action_branches.csv'):
            copy(directory/filename,Path('run_logs')/common.METHOD/f'seed_{seed}'/filename)
    for path in common.ROOT.joinpath(config['ddqna_adapter']['artifact_root']).iterdir():
        if path.is_file():copy(path,Path('registration')/path.name)
    for name in common.source_hashes():copy(common.ROOT/name,Path('source_snapshot')/name)
    for name in ('docs/DDQNA_ADAPTER_EXPERIMENT.md','tests/test_ddqna_adapter.py','scripts/package_ddqna_adapter_analysis.py'):
        copy(common.ROOT/name,Path('source_snapshot')/name)
    index=[]
    for number,(method,seed,step,split,kind,path,trace) in enumerate(selected):
        target=Path('trajectories')/f'{number:03d}_{method}_seed{seed}_{split}_{kind}.json'
        write_json(output/target,trace)
        index.append(dict(method=method,seed=seed,environment_steps=step,split=split,category=kind,
            scenario_id=trace['scenario_id'],source_path=relative(path),
            selection='first recorded failure of each category per seed; original50 uses first checkpoint >=100k',
            package_path=target.as_posix()))
    write_csv(output/'trajectory_index.csv',index)
    figures(analysis,output)
    checked['selected_full_trajectories']=len(index);write_json(output/'verification.json',checked)
    am={r['method']:r for r in read_csv(analysis/'method_means.csv')}
    em={r['method']:r for r in read_csv(evaluation/'method_means.csv')}
    def metric(r,k):return f'{float(r[k+"_mean"])*100:.2f}±{float(r[k+"_sample_sd"])*100:.2f}%'
    lines=['# DDQNA动作机制适配版：五seed完成阅读包','',
        '五次新增训练均从配对初始化运行200000在线环境步；原完整组合复用旧模型。十份最终模型在同一冻结500场景完成5000次自主评估。',
        '该基线仅适配DDQNA动作选择，使用本项目D3QN、奖励、观测及动态场景，不能称为DDQNA原版完整复现。','',
        '| 方法 | 全程AULC | 10–20万步AULC | 最终500成功 | 动态碰撞 | 超时 |',
        '|---|---:|---:|---:|---:|---:|']
    for method in report.METHODS:
        lines.append('| '+method+' | '+' | '.join([metric(am[method],'safe_success_aulc'),
            metric(am[method],'aulc_100000_200000')]+[metric(em[method],k) for k in
            ('safe_success_rate','dynamic_collision_rate','timeout_rate')])+' |')
    lines+=['','均值±样本标准差，统计单位为五个配对seed；数值CSV中的率/AULC为0–1，配对差值乘100得到百分点。',
        '适配基线五seed最终自主成功率均为0；失败轨迹核查显示主要为绕圈/等待超时。不能把适配失败推广为原论文方法普遍无效；差异包含教学概率/退出、监督、TD约束、风险筛选及任务适配。',
        '首次三个连续≥90%验证点：完整组合五seed达到，适配版均未达到；未达到保留空值，不伪造200000步，也不对缺失步数作配对均值。',
        'ε=0，教师与动态保护关闭，仅共同静态合法掩码；只取200000步最终模型。未按最终500结果调参。未来障碍信息仅用于事后诊断。',
        f'核查{checked["training_checkpoints"]}个学习检查点、{checked["training_failures_recomputed"]}条验证失败和{checked["final_failures_recomputed"]}条最终失败；精选{len(index)}条完整轨迹的规则见trajectory_index.csv。',
        '旧方法缺失的A*计数/耗时仍为空；不得声称等计算成本胜出。权重与批量原始轨迹保留本地，包中记录路径与SHA。','',
        '[学习曲线与逐seed结果](learning/REPORT.md)、[最终500结果](final500/REPORT.md)、[文献机制及适配说明](../../../docs/DDQNA_ADAPTER_EXPERIMENT.md)。',
        '[此前十一方法结果与消融](../whole_map_91701_eleven_methods_seeds0to4_20261009_183634_447770/README.md)。',
        'run_logs/含新五seed的训练/验证/动作分支/成本记录；source_snapshot/含源码与配置；registration/含训练前冻结指纹；manifest.json用于校验。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    prompt=('请从README开始，结合learning/、final500/、run_logs/、source_snapshot/和trajectory_index.csv，分析完整组合相比采用DDQNA动作选择机制的D3QN适配版增加了什么有效改进。'
        '这是动作机制适配，不能称为原论文完整复现。检查概率分支、固定p=0.5、普通TD、评估关闭教师及初始化配对，再结合原十一方法消融区分监督、TD约束、风险筛选和退出日程的证据。'
        '五seed适配最终成功均0，请依据等待/循环/碰撞轨迹分析，不能只凭成功率猜原因或断言原论文无效。按seed报告均值±样本SD与配对差值；未达到连续三点90%是删失值。'
        '最终500只作确认，不选最佳检查点、不调参；历史缺失成本不推测。明确哪些是证据、哪些是待核查假设；不要宣称首次融合、保证安全或已解决退出低谷。\n')
    (output/'GPT_ANALYSIS_PROMPT.md').write_text(prompt,encoding='utf-8')
    write_json(output/'external_artifact_sha256.json',external)
    hashes={p.relative_to(output).as_posix():common.sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'manifest.json',dict(complete=True,files_sha256=hashes,analysis=relative(analysis),
        final_evaluation=relative(evaluation),source_weights_uploaded=False,statistical_unit='five paired seeds'))
    verify(output);print(output);return output


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',type=Path);args=parser.parse_args()
    if args.verify:verify(args.verify.resolve())
    else:package(latest('analysis'),latest('independent_eval'))


if __name__=='__main__':main()
