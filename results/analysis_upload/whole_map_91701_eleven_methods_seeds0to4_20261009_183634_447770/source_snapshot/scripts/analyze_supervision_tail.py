"""Compare all autonomous checkpoints and actual failures, including both withdrawal windows."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

import supervision_tail_common as common
from analyze_teaching_efficiency import aulc, mean_records
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from astar_d3qn.agents.astar_supervision_tail import supervision_weight
from astar_d3qn.utils.io import write_json, write_records_csv


def failure_counts(behaviors):
    timeouts = [b for b in behaviors if b['termination_reason'] == 'timeout']
    waiting = sum(b['waiting_dominated'] for b in timeouts)
    looping = sum(not b['waiting_dominated'] and (b['repeated_movement'] or b['tail_period'] is not None)
                  for b in timeouts)
    return dict(collision_count=sum(b['termination_reason'] == 'collision' for b in behaviors),
                timeout_count=len(timeouts), waiting_timeout_count=waiting, looping_timeout_count=looping,
                other_timeout_count=len(timeouts)-waiting-looping)


def curve_metrics(x, y, settings):
    result = dict(safe_success_aulc=aulc(x, y), last_five_checkpoint_mean=float(np.mean(y[-5:])))
    for left, right in zip(settings['phase_boundaries'], settings['phase_boundaries'][1:]):
        result[f'aulc_{left}_{right}'] = aulc(x, y, left, right)
    for prefix, window in (('original', settings['original_withdrawal_window']),
                           ('new', settings['new_withdrawal_window']),
                           ('combined', settings['combined_withdrawal_window']), ('late', [140000, 200000])):
        chosen = np.flatnonzero((x >= window[0]) & (x <= window[1]))
        if not len(chosen):
            raise ValueError('A prespecified window has no recorded checkpoints.')
        worst = chosen[int(np.argmin(y[chosen]))]
        result[prefix + '_window_min_success'] = float(y[worst])
        result[prefix + '_window_min_step'] = int(x[worst])
    return result


def describe_delay(treated, control):
    old = treated['original_window_min_success']-control['original_window_min_success']
    new = treated['new_window_min_success']-control['new_window_min_success']
    combined = treated['combined_window_min_success']-control['combined_window_min_success']
    late = treated['aulc_140000_200000']-control['aulc_140000_200000']
    # Compare nominal 10000-step checkpoints, not a few steps of episode-end timing jitter.
    checkpoint_shift = treated['combined_window_min_step']//10000-control['combined_window_min_step']//10000
    moved = checkpoint_shift > 0
    # Descriptive criteria registered before seeing the new curves; no significance claim.
    if old > 0 and moved and (new <= 0 or combined <= 0) and late <= 0:
        verdict = '低谷后移迹象：旧窗口改善，但新/合并窗口及后段效率未支持持续改善'
    elif combined > 0 and late > 0:
        verdict = '存在持续改善迹象：合并窗口最低值与14–20万步AULC均提高'
    else:
        verdict = '混合或无改善：不能单凭旧窗口最低值判断交接更稳定'
    return dict(original_window_difference=old, new_window_difference=new, combined_window_difference=combined,
                late_aulc_difference=late, lowest_step_shift=treated['combined_window_min_step']-control['combined_window_min_step'],
                lowest_nominal_checkpoint_shift=checkpoint_shift,
                descriptive_verdict=verdict)


def collect_run(config, method, seed):
    directory, result = common.audit_run(config, method, seed)
    curve = load_csv(directory / 'validation_curve.csv')
    details = defaultdict(list)
    for entry in load_csv(directory / 'validation_details.csv'):
        details[int(entry['environment_steps_total'])].append(entry)
    x = np.array([int(r['environment_steps_total']) for r in curve])
    y = np.array([float(r['safe_success_rate']) for r in curve])
    if len(curve) != 21 or x[0] != 0 or x[-1] != 200000 or np.any(np.diff(x) <= 0) or set(details) != set(x):
        raise ValueError('Expected the original 21 checkpoints including zero and final 200000.')
    scenes = common.read_json(common.ROOT / config['dataset']['validation_reference'])['scenarios']
    distances = static_distances(common.validate_config(config)[0])
    behaviors, checkpoints = [], []
    for point in curve:
        step = int(point['environment_steps_total'])
        entries = details[step]
        if [r['scenario_id'] for r in entries] != [s['scenario_id'] for s in scenes]:
            raise ValueError('Autonomous validation scene identities/order differ from the original 50.')
        for entry, scene in zip(entries, scenes):
            if entry['dynamic_route_ids'].split(';') != scene['route_ids']:
                raise ValueError('Validation routes differ.')
        for key in ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate'):
            values = [r['termination_reason'] == 'timeout' if key == 'timeout_rate' else
                      float(r[key.removesuffix('_rate')]) for r in entries]
            if not np.isclose(float(point[key]), np.mean(values)):
                raise ValueError('Validation summary disagrees with scene details.')
        traces = common.read_json(directory / f'validation_failures_{step:06d}.json')
        failed = {r['scenario_id'] for r in entries if not float(r['safe_success'])}
        if len(traces) != len(failed) or {t['scenario_id'] for t in traces} != failed:
            raise ValueError('Missing or duplicate failure trajectories.')
        inspected = [trace_behavior(t, distances) for t in traces]
        behaviors.extend(dict(method=method, seed=seed, environment_steps=step,
                              **{k: v for k, v in b.items() if k != 'tail_60_steps'}) for b in inspected)
        checkpoints.append(dict(method=method, seed=seed, **point, **failure_counts(inspected)))
    learning = load_csv(directory / 'value_learning.csv')
    advice = load_csv(directory / 'advice_budget.csv')
    if sum(int(r['updates']) for r in learning) != 199501 or any(int(r['requested']) or int(r['applied'])
                for r in advice if int(r['bin_start_step']) >= 100001):
        raise ValueError('Incorrect update count or A* override past the original 100000 clock.')
    for value in learning:
        start = max(int(value['bin_start_step']), result['effective_training']['learning_starts'])
        end = int(value['bin_end_step'])
        if int(value['updates']) != end-start+1:
            raise ValueError('Learning log does not contain every original one-step update.')
        expected = np.mean([supervision_weight(t) if method == common.METHOD else max(0., 1-t/100000)
                            for t in range(start, end+1)])
        if not np.isclose(float(value['teacher_margin_weight_mean']), expected, atol=1e-12, rtol=1e-12):
            raise ValueError('Actual learning weight differs from the registered separate supervision clock.')
        cutoff = 140000 if method == common.METHOD else 100000
        if start >= cutoff and (int(value['teacher_label_samples']) or float(value['teacher_margin_weight_mean'])):
            raise ValueError('Supervision labels survived the exact zero boundary.')
    metrics = dict(method=method, seed=seed, **curve_metrics(x, y, config['supervision_tail']),
                   safe_success_rate=y[-1], dynamic_collision_rate=float(curve[-1]['dynamic_collision_rate']),
                   timeout_rate=float(curve[-1]['timeout_rate']))
    for prefix, left, right in (('all', 0, 200000), ('final', 200000, 200000),
                ('original_window', *config['supervision_tail']['original_withdrawal_window']),
                ('new_window', *config['supervision_tail']['new_withdrawal_window']), ('late', 140000, 200000)):
        selected = [b for b in behaviors if left <= b['environment_steps'] <= right]
        metrics.update({prefix + '_' + k: v for k, v in failure_counts(selected).items()})
    return metrics, checkpoints, behaviors, [dict(method=method, seed=seed, **r) for r in learning]


def analyze(config):
    common.validate_config(config)
    common.load_frozen(config)
    # Require every final run before producing a comparison; no selective seeds/checkpoints.
    for method in common.METHODS:
        for seed in common.SEEDS:
            common.audit_run(config, method, seed)
    summaries, curves, behaviors, learning = [], [], [], []
    for method in common.METHODS:
        for seed in common.SEEDS:
            summary, checkpoints, failures, values = collect_run(config, method, seed)
            summaries.append(summary)
            curves.extend(checkpoints)
            behaviors.extend(failures)
            learning.extend(values)
    metrics = [k for k in summaries[0] if k not in ('method', 'seed') and not k.endswith('_step')]
    means = mean_records(summaries, metrics)
    by_key = {(r['method'], r['seed']): r for r in summaries}
    paired = [dict(seed=seed, **describe_delay(by_key[common.METHOD, seed], by_key[common.BASELINE, seed]),
                   **{k + '_difference': by_key[common.METHOD, seed][k]-by_key[common.BASELINE, seed][k] for k in metrics})
              for seed in common.SEEDS]
    paired_means = [dict(metric=k, mean_difference=float(np.mean([r[k+'_difference'] for r in paired])),
                        sample_sd_difference=float(np.std([r[k+'_difference'] for r in paired], ddof=1))) for k in metrics]
    output = common.legacy.runtime.unique_check_dir(config, 'analysis')
    for name, rows in (('summary.csv', summaries), ('method_means.csv', means), ('all_validation_checkpoints.csv', curves),
                       ('all_failure_behaviors.csv', behaviors), ('value_learning.csv', learning),
                       ('paired_differences.csv', paired), ('paired_difference_means.csv', paired_means)):
        write_records_csv(rows, output / name)
    write_json(dict(complete=True, training_started=False, independent_500_performance_evaluated=False,
                    methods=list(common.METHODS), seeds=list(common.SEEDS), all_50_scenes_and_failures_audited=True,
                    validation_checkpoint_count=len(curves), original_window=config['supervision_tail']['original_withdrawal_window'],
                    new_window=config['supervision_tail']['new_withdrawal_window'],
                    statistical_unit='Five paired training seeds; counts across checkpoints are repeated observations.',
                    historical_td_targets='Use actual value_learning logs; checkpoint policy weights do not recover old target networks.'),
               output / 'verification.json')
    plot_curves(output, curves)
    lines = ['# 动作监督尾段延期：完整组合配对分析', '',
             '仅改变辅助监督权重；A*代选仍在100000步为零，监督在140000步为零。所有曲线为原50场景epsilon=0自主验证；未使用新的500场景调参或选择检查点。统计单位是5个配对seed，以下均值±样本标准差单位%。', '',
             '| 方法 | 全程AULC | 原退出窗最低 | 新退出窗最低 | 14–20万AULC | 最后五次均值 | 最终成功 | 最终碰撞 | 最终超时 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    report_metrics = ('safe_success_aulc', 'original_window_min_success', 'new_window_min_success', 'aulc_140000_200000',
                      'last_five_checkpoint_mean', 'safe_success_rate', 'dynamic_collision_rate', 'timeout_rate')
    for r in means:
        lines.append('| ' + r['method'] + ' | ' + ' | '.join(
            f'{r[k+"_mean"]*100:.2f}±{r[k+"_sample_sd"]*100:.2f}' for k in report_metrics) + ' |')
    lines += ['', '原窗口90000–120500，新窗口130000–160500；同宽整体后移40000步。合并窗口90000–160500和14–20万后段也完整报告，不能只展示旧窗口改善。最低值是记录的检查点最低值，AULC按实际步数线性插值积分。', '',
              '| seed | 原窗最低变化（百分点） | 新窗最低变化 | 合并窗最低变化 | 14–20万AULC变化 | 合并窗最低步数变化 | 描述性判断 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for r in paired:
        lines.append(f'| {r["seed"]} | {r["original_window_difference"]*100:.2f} | {r["new_window_difference"]*100:.2f} | '
                     f'{r["combined_window_difference"]*100:.2f} | {r["late_aulc_difference"]*100:.2f} | '
                     f'{r["lowest_step_shift"]} | {r["descriptive_verdict"]} |')
    lines += ['', '配对表及全程/分阶段指标逐seed数值见summary.csv，所有均值/样本标准差见method_means.csv，配对差值见paired_difference_means.csv。描述性判断不是显著性检验；是否持续改善须联合全部seed曲线、两窗口、合并窗口、后段AULC及失败类型判断。', '',
              '碰撞、等待超时、循环超时和其他超时计数分别保存于summary.csv及all_validation_checkpoints.csv；final前缀是最后50场景，original_window/new_window/late前缀为各窗口检查点失败次数之和，all包括0步全部21次检查点，不能当作独立失败场景数。',
              '等待超时定义为WAIT比例≥50%；循环超时定义为非等待主导且反复移动占比≥50%（至少20次移动），或尾部60位置出现2–12周期；其余归其他超时，三类互斥。原始布尔行为可重叠，完整保留在all_failure_behaviors.csv。静态剩余距离只表明静态可达，不证明动态300步内可达。',
              '真实Q和TD目标诊断来自value_learning.csv。未来障碍信息不用于动作选择或教学标签。新的500场景仅在两方法全部最终模型完成后统一评估，不选择最佳权重。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output)
    return output


def plot_curves(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    grid = np.arange(0, 200001, 1000)
    for method in common.METHODS:
        interpolated = []
        for seed in common.SEEDS:
            entries = [r for r in rows if r['method'] == method and r['seed'] == seed]
            x = [int(r['environment_steps_total']) for r in entries]
            y = [float(r['safe_success_rate'])*100 for r in entries]
            axes.flat[seed].plot(x, y, label=method, linewidth=1.2)
            interpolated.append(np.interp(grid, x, y))
        values = np.array(interpolated)
        axes.flat[5].plot(grid, values.mean(axis=0), label=method)
        axes.flat[5].fill_between(grid, values.mean(axis=0)-values.std(axis=0, ddof=1),
                                 values.mean(axis=0)+values.std(axis=0, ddof=1), alpha=.12)
    for index, ax in enumerate(axes.flat):
        for step in (80000, 100000, 140000):
            ax.axvline(step, color='gray', linestyle='--', alpha=.5)
        ax.set(title=f'Seed {index}' if index < 5 else 'Mean and sample SD', xlabel='Environment steps',
               ylabel='Autonomous safe success (%)', ylim=(-2, 102))
    axes.flat[5].legend(fontsize=7)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'learning_curves.{extension}', dpi=160)
    plt.close(fig)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    analyze(common.legacy.load_config(common.CONFIG))


if __name__ == '__main__':
    main()
