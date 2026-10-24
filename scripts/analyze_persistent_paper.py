"""Version-matched paper tables and figures from completed, audited scalar results."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import persistent_paper_common as core
import evaluate_persistent_paper as evaluation
from analyze_persistent_teaching import auc

LABELS = dict(persistent_asvc='ASVC (persistent supervision)', no_td='Without TD target constraint',
    no_advice='Without A* action advice', no_supervision='Without action supervision',
    original_asvc='ASVC (original withdrawal)', dqfd='DQfD', d3qn='D3QN', astar_prefill='A* prefill + TD constraint')
CN = dict(persistent_asvc='新版完整方法', no_td='新版去TD目标约束', no_advice='新版去A*动作代选',
    no_supervision='去动作监督（复用）', original_asvc='原版ASVC（复用）', dqfd='DQfD（复用）',
    d3qn='D3QN（复用）', astar_prefill='A*初始化＋TD约束（复用）')
COLORS = dict(persistent_asvc='#0072B2', no_td='#CC79A7', no_advice='#D55E00', no_supervision='#009E73',
    original_asvc='#777777', dqfd='#D55E00', d3qn='#999933', astar_prefill='#882255')


def evaluate_record(method, seed, suite, evidence):
    path = core.RESULTS / 'evaluation' / suite / method / f'seed_{seed}'
    reg = core.common.read_json(path / 'registration.json')
    info = core.run_info(method, seed)
    scenes, digest = evaluation.scenes_for(suite)
    core.require(core.common.read_json(path / 'completion.json')['complete'], f'Evaluation incomplete: {path}')
    core.require(reg['model_sha256'] == info['digest'] and reg['scene_sha256'] == digest, 'Evaluation model/scenes changed.')
    evaluation.check_autonomous(reg)
    record = core.common.read_json(path / 'summary.json')
    evaluation.check_rows(core.read_csv(path / 'scene_details.csv'), record, scenes)
    for name in ('registration.json', 'summary.json', 'scene_details.csv', 'completion.json'):
        evidence[core.relative(path / name)] = core.common.sha256(path / name)
    return record


def means(records, methods, keys):
    result = []
    for method in methods:
        selected = [r for r in records if r['method'] == method]
        core.require(len(selected) == 5 and {r['seed'] for r in selected} == set(range(5)), 'Keep all five training seeds.')
        row = dict(method=method, planned_seed_count=5)
        for key in keys:
            values = [r.get(key) for r in selected]
            usable = [v for v in values if v is not None and np.isfinite(v)]
            row[key + '_observed_seeds'] = len(usable)
            row[key + '_mean'] = float(np.mean(usable)) if len(usable) == 5 else None
            row[key + '_sd'] = float(np.std(usable, ddof=1)) if len(usable) == 5 else None
        result.append(row)
    return result


def ablation_records(evidence):
    metrics, curves = [], []
    for method in core.ABLATIONS:
        for seed in core.SEEDS:
            info = core.run_info(method, seed)
            path = info['directory']
            curve_file = path / 'validation_curve.csv'
            rows = core.read_csv(curve_file) if curve_file.exists() else []
            for row in rows:
                curves.append(dict(method=method, seed=seed, step=int(row['environment_steps_total']),
                    success=float(row['safe_success_rate']), timeout=float(row['timeout_rate']),
                    wait=float(row['mean_wait_steps']), numerical_failure=info['failed']))
            if curve_file.exists():
                evidence[core.relative(curve_file)] = core.common.sha256(curve_file)
            if info['failed']:
                metrics.append(dict(method=method, seed=seed, numerical_failure=True,
                    failure_step=info['failure']['environment_steps'], full_aulc=None, late_aulc=None,
                    late_min=None, final_validation=None, final500_success=None,
                    final500_collision=None, final500_timeout=None))
                evidence[core.relative(path / 'nonfinite_failure.json')] = core.common.sha256(path / 'nonfinite_failure.json')
                continue
            x = np.array([int(r['environment_steps_total']) for r in rows])
            y = np.array([float(r['safe_success_rate']) for r in rows])
            core.require(len(x) == 21 and x[0] == 0 and x[-1] == 200000 and np.all(np.diff(x) > 0), 'Expected all 21 validation checkpoints.')
            final = evaluate_record(method, seed, 'final500', evidence)
            metrics.append(dict(method=method, seed=seed, numerical_failure=False, failure_step=None,
                full_aulc=auc(x, y, 0, 200000), late_aulc=auc(x, y, 100000, 200000),
                late_min=float(y[x >= 100000].min()), final_validation=float(y[-1]),
                final500_success=final['safe_success_rate'], final500_collision=final['dynamic_collision_rate'],
                final500_timeout=final['timeout_rate']))
            evidence[core.relative(info['model'])] = info['digest']
            evidence[core.relative(path / 'result.json')] = core.common.sha256(path / 'result.json')
    return metrics, curves


def save(fig, output, name):
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{name}.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def ablation_figures(output, metrics, curves, averages):
    def trace(method, seed):
        rows = [r for r in curves if r['method'] == method and r['seed'] == seed]
        return np.array([r['step'] for r in rows]), np.array([r['success'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for method in core.ABLATIONS:
        if any(r['numerical_failure'] for r in metrics if r['method'] == method):
            continue  # Incomplete seeds are explicitly retained below, never averaged as five.
        for ax, start in zip(axes, (0, 80000)):
            points = np.linspace(start, 200000, 201)
            values = np.array([np.interp(points, *trace(method, s)) * 100 for s in core.SEEDS])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(points/1000, mean, color=COLORS[method], label=LABELS[method], lw=1.7)
            ax.fill_between(points/1000, np.maximum(0, mean-sd), np.minimum(100, mean+sd), color=COLORS[method], alpha=.12)
    for ax, title in zip(axes, ('Full training', 'Late-stage stability')):
        ax.set(title=title, xlabel='Environment steps (thousands)', ylabel='Autonomous safe success (%)', ylim=(0, 102))
        ax.axvline(90, color='#999999', ls=':', lw=1)
        ax.axvline(100, color='#999999', ls='--', lw=1)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='lower right')
    failures = sum(r['numerical_failure'] for r in metrics)
    fig.suptitle(f'Persistent-supervision ablations | five-seed mean ± sample SD | failed runs: {failures}')
    fig.tight_layout(); save(fig, output, '01_module_learning_curves')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for seed, ax in zip(core.SEEDS, axes.flat):
        for method in core.ABLATIONS:
            x, y = trace(method, seed)
            row = next(r for r in metrics if r['method'] == method and r['seed'] == seed)
            ax.plot(x/1000, y*100, color=COLORS[method], label=LABELS[method],
                    ls='--' if row['numerical_failure'] else '-', lw=1.5)
            if row['numerical_failure'] and row['failure_step'] is not None:
                ax.axvline(row['failure_step']/1000, color=COLORS[method], ls=':')
        ax.set(title=f'Seed {seed}', xlim=(0, 200), ylim=(0, 102), xlabel='Steps (thousands)', ylabel='Safe success (%)')
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False, fontsize=9)
    fig.suptitle('Every checkpoint retained; dashed incomplete runs are numerical failures')
    fig.tight_layout(); save(fig, output, '02_module_individual_seeds')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    names = ('Full', 'No TD', 'No advice', 'No supervision')
    for ax, key, title in zip(axes, ('final500_success', 'final500_collision', 'final500_timeout'),
                              ('Final500 safe success', 'Final500 collision', 'Final500 timeout')):
        for i, method in enumerate(core.ABLATIONS):
            row = next(r for r in averages if r['method'] == method)
            if row[key + '_mean'] is None:
                ax.text(i, .5, 'NA', ha='center', transform=ax.get_xaxis_transform())
                continue
            ax.bar(i, row[key + '_mean']*100, yerr=row[key + '_sd']*100, color=COLORS[method], alpha=.8, capsize=4)
        ax.set(title=title, xticks=range(4), xticklabels=names, ylabel='%')
        ax.tick_params(axis='x', rotation=20); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Final 200k models; same frozen 500 scenes | mean ± sample SD')
    fig.tight_layout(); save(fig, output, '03_module_final500')


def density_figures(output, records, averages):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, key, title in zip(axes, evaluation.METRICS, ('Safe success', 'Dynamic collision', 'Timeout')):
        for method in core.DENSITY_METHODS:
            rows = [next(r for r in averages if r['method'] == method and r['density'] == n) for n in (1, 3, 5, 7)]
            ax.errorbar([1, 3, 5, 7], [r[key + '_mean']*100 for r in rows],
                yerr=[r[key + '_sd']*100 for r in rows], marker='o', capsize=3, color=COLORS[method],
                label=LABELS[method], ls='--' if method == 'original_asvc' else '-')
        ax.set(title=title, xlabel='Dynamic obstacles', ylabel='%', xticks=[1, 3, 5, 7])
        ax.grid(alpha=.2)
    axes[0].set_ylim(0, 102)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('Original frozen density-v2 scenes | final models | five-seed mean ± sample SD')
    fig.tight_layout(rect=(0, .13, 1, .94)); save(fig, output, '04_density_comparison')
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for seed in core.SEEDS:
        rows = [next(r for r in records if r['method'] == 'persistent_asvc' and r['seed'] == seed and r['density'] == n) for n in (1, 3, 5, 7)]
        ax.plot([1, 3, 5, 7], [r['safe_success_rate']*100 for r in rows], marker='o', label=f'Seed {seed}')
    ax.set(title='Persistent ASVC: all five seeds', xlabel='Dynamic obstacles', ylabel='Safe success (%)', xticks=[1, 3, 5, 7], ylim=(0, 102))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); save(fig, output, '05_density_individual_seeds')


def format_metric(row, key):
    if row[key + '_mean'] is None:
        return 'NA（未满5个有效seed）'
    return f'{row[key + "_mean"]*100:.2f}±{row[key + "_sd"]*100:.2f}%'


def analyze(suite='all'):
    evaluation.check_reuse_freeze()
    evidence = {}
    metrics, curves, averages, density_rows, density_means = [], [], [], [], []
    # Require all requested inputs before creating an official report directory.
    if suite in ('all', 'ablation'):
        metrics, curves = ablation_records(evidence)
        keys = ('full_aulc', 'late_aulc', 'late_min', 'final_validation', 'final500_success', 'final500_collision', 'final500_timeout')
        averages = means(metrics, core.ABLATIONS, keys)
    if suite in ('all', 'density'):
        for n in (1, 3, 5, 7):
            selected = []
            for method in core.DENSITY_METHODS:
                for seed in core.SEEDS:
                    record = evaluate_record(method, seed, f'density_{n}', evidence)
                    selected.append(dict(record, density=n))
            density_rows.extend(selected)
            density_means.extend(dict(r, density=n) for r in means(selected, core.DENSITY_METHODS, evaluation.METRICS))
    output = core.RESULTS / ('analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    lines = ['# 持续弱监督新版：模块消融与障碍密度测试', '',
        '监督从1下降到0.1后持续保留；完整方法A*代选在10万步退出。去TD仅取消TD目标裁剪；去代选保留监督标签和0.1权重。奖励、观测、网络、风险筛选和训练预算沿用完整方法。', '',
        '只新增两个消融×五seed训练，各20万步；完整方法和去监督对照复用已核对结果。最终评估均为网络自主行动，只有静态非法动作屏蔽，不使用A*、教师、未来障碍信息或动态安全屏蔽。', '']
    if metrics:
        core.write_csv(metrics, output / 'ablation_per_seed.csv')
        core.write_csv(averages, output / 'ablation_means.csv')
        core.write_csv(curves, output / 'ablation_curves.csv')
        paired = []
        for method in core.ABLATIONS[1:]:
            for seed in core.SEEDS:
                full = next(r for r in metrics if r['method'] == 'persistent_asvc' and r['seed'] == seed)
                removed = next(r for r in metrics if r['method'] == method and r['seed'] == seed)
                paired.append(dict(method=method, seed=seed, **{key + '_full_minus_removed':
                    None if removed[key] is None else full[key]-removed[key] for key in ('late_aulc', 'late_min', 'final500_success')}))
        core.write_csv(paired, output / 'ablation_paired_differences.csv')
        ablation_figures(output, metrics, curves, averages)
        lines += ['## 新版模块消融', '', '|方法|全程AULC|后半程AULC|后半程最低成功率|最终500成功率|', '|---|---:|---:|---:|---:|']
        for row in averages:
            lines.append('|' + CN[row['method']] + '|' + '|'.join(format_metric(row, key) for key in ('full_aulc', 'late_aulc', 'late_min', 'final500_success')) + '|')
        failures = [dict(method=r['method'], seed=r['seed'], failure_step=r['failure_step']) for r in metrics if r['numerical_failure']]
        core.write_json(failures, output / 'numerical_failures.json')
        if failures:
            lines += ['', '数值失败的种子保留为NA，不换seed、不调参重跑、不把剩余种子当成五seed平均：' + str(failures)]
        lines += ['', '![模块曲线](01_module_learning_curves.png)', '', '![模块逐seed](02_module_individual_seeds.png)', '']
    if density_rows:
        core.write_csv(density_rows, output / 'density_per_seed.csv')
        core.write_csv(density_means, output / 'density_means.csv')
        density_figures(output, density_rows, density_means)
        lines += ['## 新版障碍密度测试', '', '|方法|1个障碍|3个障碍|5个障碍|7个障碍|', '|---|---:|---:|---:|---:|']
        for method in core.DENSITY_METHODS:
            values = [next(r for r in density_means if r['method'] == method and r['density'] == n) for n in (1, 3, 5, 7)]
            lines.append('|' + CN[method] + '|' + '|'.join(format_metric(r, 'safe_success_rate') for r in values) + '|')
        lines += ['', '每档500个原冻结v2场景，新版5模型×4档共新增10000回合，四个旧对照的40000回合核对后复用。', '', '![密度比较](04_density_comparison.png)', '']
    lines += ['所有均值±样本标准差以训练seed为单位。AULC按实际步数积分；曲线只对齐时间点，不平滑、不选择最好检查点。最低值先按每个seed计算再平均。', '',
        '这是固定地图及既定动态障碍分布的测试，不是跨地图泛化。DQfD另外使用2万次离线更新、永久示范和PER；同在线步数不等于同总计算预算。', '',
        '新旧完整方法五seed对比和新版最终500结果继续保留；本报告没有重跑它们。旧消融和旧密度结果保留原版本标签。所有图提供PNG、PDF和SVG。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    core.write_json(dict(complete=True, suite=suite, sources_sha256=evidence,
        analysis_source_sha256=core.common.sha256(Path(__file__)), statistical_unit='training_seed',
        all_five_seeds_retained=True, evaluation_source_sha256=core.common.sha256(Path(evaluation.__file__))), output / 'evidence.json')
    (core.RESULTS / 'LATEST.md').write_text(f'# 新版补实验\n\n[报告]({output.name}/REPORT.md)\n', encoding='utf-8')
    print(output, flush=True)
    return output / 'REPORT.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('all', 'ablation', 'density'), default='all')
    analyze(parser.parse_args().suite)
