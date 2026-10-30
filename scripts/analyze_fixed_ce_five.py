"""Five-seed paired CE/margin reporting; historical two-seed results remain immutable."""
from __future__ import annotations
import argparse
import statistics

import fixed_ce_five_common as c
from analyze_fixed_ce import rows, metrics, MEASUREMENTS, interpolate


def summarize(records):
    lookup = {(r['method'], r['seed']): r for r in records}
    matched = [seed for seed in c.ALL_SEEDS if all(lookup.get((m, seed), {}).get('run_complete', False)
               and lookup.get((m, seed), {}).get('curve_complete', False) for m in ('margin', 'ce'))]
    summary, differences, paired_summary = [], [], []
    for seed in c.ALL_SEEDS:
        left, right = lookup.get(('ce', seed), {}), lookup.get(('margin', seed), {})
        differences.append(dict(seed=seed, contrast='CE_minus_margin', paired_complete=(seed in matched),
            **{key: left[key]-right[key] if left.get(key) is not None and right.get(key) is not None else None
               for key in MEASUREMENTS}))
    for method in ('margin', 'ce'):
        for key in MEASUREMENTS:
            values = [lookup[method, seed][key] for seed in matched]
            summary.append(dict(method=method, metric=key, completed_paired_training_seeds=len(values),
                requested_training_seeds=5, all_five_complete=len(matched)==5,
                paired_seed_ids=';'.join(map(str, matched)),
                mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None))
    for key in MEASUREMENTS:
        values = [r[key] for r in differences if r['seed'] in matched]
        paired_summary.append(dict(contrast='CE_minus_margin', metric=key, completed_paired_training_seeds=len(values),
            requested_training_seeds=5, all_five_complete=len(matched)==5,
            mean=statistics.mean(values) if values else None, sample_sd=statistics.stdev(values) if len(values)>1 else None))
    return matched, summary, differences, paired_summary


def plots(curves, matched):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#3478b1', '#d67b30', '#3c9865', '#a45eae', '#8a7965')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    grid = list(range(0, 200001, 10000))
    for ax, key in zip(axes, ('safe_success_rate', 'collision_rate', 'timeout_rate')):
        for method, color in (('margin', '#3478b1'), ('ce', '#d67b30')):
            if not matched:
                continue
            values = [[interpolate(curves[method, seed], t, key) for seed in matched] for t in grid]
            means = [statistics.mean(v) for v in values]
            sd = [statistics.stdev(v) if len(v)>1 else 0. for v in values]
            ax.plot([t/1000 for t in grid], means, color=color, label=f'{method}, n={len(matched)} paired seeds')
            ax.fill_between([t/1000 for t in grid], [m-s for m,s in zip(means,sd)],
                            [m+s for m,s in zip(means,sd)], color=color, alpha=.15)
        ax.set(xlabel='Environment steps (thousands)', ylabel=key, xlim=(0,200), ylim=(-.02,1.02))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed-task CE vs margin: mean +/- sample SD; ' +
                 ('FIVE SEEDS COMPLETE' if len(matched)==5 else f'IN PROGRESS ({len(matched)}/5 completed pairs)'))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(c.RESULTS / f'learning_curves.{ext}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for seed, ax in zip(c.ALL_SEEDS, axes.flat):
        for method, style in (('margin', '--'), ('ce', '-')):
            curve = curves.get((method, seed), [])
            if curve:
                ax.plot([int(r['environment_steps_total'])/1000 for r in curve],
                        [float(r['safe_success_rate']) for r in curve], linestyle=style,
                        color='#3478b1' if method=='margin' else '#d67b30', label=method)
        ax.set(title=f'seed {seed}', xlabel='Steps (thousands)', ylabel='Safe success', xlim=(0,200), ylim=(-.02,1.02))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(c.RESULTS / f'individual_seeds.{ext}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for method in ('margin', 'ce'):
        for seed in c.ALL_SEEDS:
            log = rows(c.analysis_directory(method, seed) / 'value_learning.csv')
            if not log:
                continue
            raw = 'demo_margin_loss_mean' if method=='margin' else 'supervision_ce_loss_mean'
            for ax, key in zip(axes, ('q_abs_max', 'td_loss_mean', raw)):
                ax.plot([int(r['bin_end_step'])/1000 for r in log], [float(r[key]) for r in log],
                        linestyle='--' if method=='margin' else '-', color=colors[seed], alpha=.8,
                        label=f'{method} seed{seed}')
    for ax, label in zip(axes, ('Q absolute maximum', 'TD loss', 'Raw eligible supervision loss')):
        ax.set(xlabel='Steps (thousands)', ylabel=label, xlim=(0,200)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(c.RESULTS / f'value_loss_diagnostics.{ext}', dpi=200)
    plt.close(fig)


def analyze(plot=True):
    c.verify_freeze(check_baselines=False)
    records, curves, points, details, cost = [], {}, [], [], []
    for seed in c.ALL_SEEDS:
        for method in ('margin', 'ce'):
            directory = c.analysis_directory(method, seed)
            curve = rows(directory / 'validation_curve.csv')
            curves[method, seed] = curve
            complete = (directory / 'completion.json').exists() and c.read(directory / 'completion.json')['complete']
            if curve:
                xs = [int(r['environment_steps_total']) for r in curve]
                if xs != sorted(set(xs)) or xs[-1]>200000:
                    raise ValueError('Invalid validation clock')
                record = dict(method=method, seed=seed, **metrics(curve))
                if complete and not record['curve_complete']:
                    raise ValueError('Completion marker without final validation')
            else:
                record = dict(method=method, seed=seed, curve_complete=False, validation_points=0,
                              last_validation_step=0, **{key:None for key in MEASUREMENTS})
            record.update(run_complete=bool(complete), source_reused=(method=='margin' or seed in c.REUSED_SEEDS))
            records.append(record)
            points.extend(dict(method=method, seed=seed, **r) for r in curve)
            details.extend(dict(method=method, seed=seed, **r) for r in rows(directory / 'validation_details.csv'))
            result = c.read(directory / 'result.json') if complete else {}
            cost.append(dict(method=method, seed=seed, source_reused=record['source_reused'], complete=bool(complete),
                environment_steps=result.get('environment_steps'), gradient_updates=result.get('gradient_updates'),
                online_training_seconds=result.get('online_training_seconds'), session_seconds=result.get('session_seconds'),
                new_steps_in_this_extension=result.get('environment_steps') if not record['source_reused'] else 0))
    matched, summary, paired, paired_summary = summarize(records)
    for name, content in [('per_seed_metrics', records), ('mean_sample_sd', summary), ('paired_seed_differences', paired),
                          ('paired_mean_sample_sd', paired_summary), ('learning_curve_points', points),
                          ('validation_details', details), ('training_cost', cost)]:
        c.write_csv(content, c.RESULTS / (name+'.csv'))
    complete = len(matched)==5
    c.write_json(dict(all_five_ce_runs_complete=complete, completed_paired_seeds=matched,
        new_training_seeds=list(c.NEW_SEEDS), reused_ce_seeds=list(c.REUSED_SEEDS),
        new_runs_complete=sum(r['run_complete'] for r in records if r['method']=='ce' and r['seed'] in c.NEW_SEEDS),
        baseline_rerun=False, formal_test_used=False, statistics_unit='paired_training_seed',
        aggregate_uses_same_completed_seeds_for_both_methods=True, runs=records), c.RESULTS / 'analysis_status.json')
    if plot:
        plots(curves, matched)
    lines = ['# 固定起终点 CE 对照：五 seed 扩展', '',
        '**状态：五个配对 seed 已完成。**' if complete else f'**状态：已完成 {len(matched)}/5 个配对 seed；五 seed 结论尚不可用。**', '',
        '本轮只新增 CE seed2、3、4 各 200000 步；CE seed0、1 和原大间隔五 seed 复用。参数、损失、训练循环和原50验证池保持不变，未使用最终500场景。', '',
        '| 方法 | seed | 验证步数 | 前5万 AULC | 全程 AULC | 后10万 AULC | 后半最低成功 | 最终成功 | 最终碰撞 | 最终超时 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: '未完成' if x is None else f'{100*x:.2f}%'
    keys = MEASUREMENTS[:7]
    for r in records:
        lines.append(f"| {r['method']} | {r['seed']} | {r['last_validation_step']} | " +
                     ' | '.join(fmt(r.get(key)) for key in keys) + ' |')
    lines += ['', f'下表仅用两组共同完成的 seed {matched}，n={len(matched)}；训练中不把原组5个和CE不足5个混比。均值±样本SD的单位是训练seed。', '',
        '| 指标 | 大间隔监督 | CE | CE−大间隔（百分点） |', '|---|---:|---:|---:|']
    labels = dict(aulc_0_50000='前5万 AULC', aulc_0_200000='全程 AULC', aulc_100000_200000='后10万 AULC',
        late_min_safe_success_rate='后半最低安全成功', final_safe_success_rate='最终安全成功',
        final_collision_rate='最终碰撞', final_timeout_rate='最终超时',
        late_collision_rate_aulc='后半碰撞 AULC', late_timeout_rate_aulc='后半超时 AULC')
    for key, label in labels.items():
        cells = []
        for method in ('margin', 'ce'):
            r = next(r for r in summary if r['method']==method and r['metric']==key)
            cells.append(fmt(r['mean']) + ' ± ' + fmt(r['sample_sd']))
        change = next(r['mean'] for r in paired_summary if r['metric']==key)
        cells.append('未完成' if change is None else f'{100*change:+.2f}')
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '原始曲线按真实步数积分；50k、100k边界线性插值，不外推未完成区间。每万步验证沿用跨过间隔后的回合边界。后半最低值取实际验证点。',
        '', '均值曲线仅为可视化在共同步数网格上线性插值，阴影为训练seed的样本SD，不是置信区间；单seed图同时显示未完成曲线。',
        '', '原50场景反复用于验证，不能当作独立最终测试。五seed扩展是在看到前两个seed结果后由用户决定，未改变参数或挑选seed；保留原两seed报告及全部失败证据。',
        '', 'Q值/TD/监督损失和标签覆盖日志保留；CE与margin原始损失尺度不同，不能凭标量大小判断效果。新旧训练的墙钟耗时不作为效率优势证据。']
    if complete:
        changes = {r['metric']: r['mean'] for r in paired_summary}
        lines += ['', f"观测：五seed平均全程AULC的 CE−margin 为 {100*changes['aulc_0_200000']:+.2f} 个百分点，后半程为 {100*changes['aulc_100000_200000']:+.2f} 个百分点，最终成功为 {100*changes['final_safe_success_rate']:+.2f} 个百分点。",
            '', '解释边界：这些是本固定任务协议中的配对观测；不能直接推出普遍优势、具体梯度因果或收敛性。交叉熵在此作为同教师监督形式对照，不以其本身宣称创新。不自动替换已登记主方法。']
    (c.RESULTS / 'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Five-seed paired analysis: {len(matched)}/5 complete; old outputs preserved.', flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    analyze(not args.no_plots)
