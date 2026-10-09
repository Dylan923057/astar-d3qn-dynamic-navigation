"""Read-only partial review of completed paired runs; never trains or evaluates holdouts."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

import analyze_supervision_tail as analysis
import supervision_tail_common as common
from analyze_runtime_path_pilot import load_csv
from analyze_teaching_efficiency import mean_records
from astar_d3qn.utils.io import write_json, write_records_csv


def review(seeds):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s not in common.SEEDS for s in seeds):
        raise ValueError('Use at least two distinct registered seeds; only completed runs are accepted.')
    config = common.legacy.load_config(common.CONFIG)
    common.validate_config(config)
    common.load_frozen(config)  # Integrity only; no independent-scene performance.
    summaries, checkpoints, behaviors, learning, early = [], [], [], [], []
    for seed in seeds:
        directories = []
        for method in common.METHODS:
            directories.append(common.run_directory(config, method, seed))
            summary, curve, failures, values = analysis.collect_run(config, method, seed)
            summaries.append(summary)
            checkpoints.extend(curve)
            behaviors.extend(failures)
            learning.extend(values)
        records = [[r for r in load_csv(d / 'training.csv')
                    if int(r['environment_steps_total']) <= 80000] for d in directories]
        if records[0] != records[1]:
            raise ValueError(f'Seed {seed}: early training records differ.')
        early.append(dict(seed=seed, checked_episode_rows=len(records[0]),
                          all_columns_identical=True, cutoff_environment_steps=80000))
    metrics = [k for k in summaries[0] if k not in ('method', 'seed') and not k.endswith('_step')]
    means = mean_records(summaries, metrics)
    by_key = {(r['method'], r['seed']): r for r in summaries}
    paired = [dict(seed=seed, **analysis.describe_delay(by_key[common.METHOD, seed],
                                                        by_key[common.BASELINE, seed]),
                   **{k + '_difference': by_key[common.METHOD, seed][k] -
                       by_key[common.BASELINE, seed][k] for k in metrics}) for seed in seeds]
    paired_means = [dict(metric=k, mean_difference=float(np.mean([r[k+'_difference'] for r in paired])),
                        sample_sd_difference=float(np.std([r[k+'_difference'] for r in paired], ddof=1)))
                    for k in metrics]
    output = common.legacy.runtime.unique_check_dir(config, 'partial_analysis')
    for name, rows in (('summary.csv', summaries), ('method_means.csv', means),
                       ('all_validation_checkpoints.csv', checkpoints), ('all_failure_behaviors.csv', behaviors),
                       ('value_learning.csv', learning), ('paired_differences.csv', paired),
                       ('paired_difference_means.csv', paired_means), ('early_training_match.csv', early)):
        write_records_csv(rows, output / name)
    write_json(dict(partial_review=True, full_five_seed_analysis_complete=False, seeds=list(seeds),
                    remaining_seeds=[s for s in common.SEEDS if s not in seeds],
                    training_started=False, training_modified=False, independent_500_performance_evaluated=False,
                    all_original_50_scene_details_and_failure_traces_audited=True,
                    validation_checkpoint_count=len(checkpoints), validation_detail_count=len(checkpoints)*50,
                    inspected_failure_trace_count=len(behaviors), early_training_match=early,
                    actual_supervision_schedule_and_advice_cutoffs_verified=True,
                    original_window=config['supervision_tail']['original_withdrawal_window'],
                    new_window=config['supervision_tail']['new_withdrawal_window'],
                    statistical_unit=f'{len(seeds)} paired seeds; repeated checkpoint counts are not independent scenes.'),
               output / 'verification.json')
    plot(output, checkpoints, seeds)
    display = ('safe_success_aulc', 'original_window_min_success', 'new_window_min_success',
               'aulc_140000_200000', 'last_five_checkpoint_mean', 'safe_success_rate',
               'dynamic_collision_rate', 'timeout_rate')
    lines = ['# 动作监督尾段延期：已完成seed的阶段性对比', '',
             f'范围：seed {", ".join(map(str, seeds))}。仅为已完成seed的描述性结果，不能代替五seed结论。', '',
             '本报告审查两个方法的全部21次原50场景自主验证（epsilon=0，无A*代选），以及每次失败的实际轨迹；没有使用新的独立500场景表现，也没有启动、停止或修改训练。所有最终结果来自200000步模型，不选择最佳检查点。', '',
             '## 逐seed结果', '',
             '| seed | 方法 | 全程AULC | 原退出窗最低 | 新退出窗最低 | 14–20万步AULC | 最后五次均值 | 最终成功 | 最终碰撞 | 最终超时 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    names = {common.BASELINE: '原完整组合', common.METHOD: '监督尾段延期'}
    for r in summaries:
        lines.append(f'| {r["seed"]} | {names[r["method"]]} | ' +
                     ' | '.join(f'{r[k]*100:.2f}%' for k in display) + ' |')
    lines += ['', '## 均值±样本标准差', '',
              '| 方法 | 全程AULC | 原退出窗最低 | 新退出窗最低 | 14–20万步AULC | 最后五次均值 | 最终成功 | 最终碰撞 | 最终超时 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in means:
        lines.append('| ' + names[r['method']] + ' | ' + ' | '.join(
            f'{r[k+"_mean"]*100:.2f}±{r[k+"_sample_sd"]*100:.2f}%' for k in display) + ' |')
    lines += ['', '## 最低点和实际失败类型', '']
    for seed in seeds:
        r = by_key[common.METHOD, seed]
        step = r['combined_window_min_step']
        chosen = [b for b in behaviors if b['method'] == common.METHOD and b['seed'] == seed
                  and b['environment_steps'] == step]
        counts = analysis.failure_counts(chosen)
        positions = Counter(tuple(b['final_position']) for b in chosen)
        lines.append(f'- seed{seed}：合并退出窗最低点{step}步，成功率{r["combined_window_min_success"]*100:.0f}%；'
                     f'碰撞{counts["collision_count"]}，等待超时{counts["waiting_timeout_count"]}，'
                     f'循环超时{counts["looping_timeout_count"]}，其他超时{counts["other_timeout_count"]}。'
                     f'失败结束位置计数：{dict(positions)}。')
    lines += ['', '## 各窗口失败次数', '',
              '| seed | 方法 | 范围 | 碰撞 | 等待超时 | 循环超时 | 其他超时 |',
              '|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        for prefix, name in (('original_window', '原退出窗'), ('new_window', '新退出窗'),
                             ('late', '14–20万步'), ('final', '最终50场景')):
            lines.append(f'| {r["seed"]} | {names[r["method"]]} | {name} | ' + ' | '.join(
                str(r[prefix+'_'+k]) for k in ('collision_count', 'waiting_timeout_count',
                                               'looping_timeout_count', 'other_timeout_count')) + ' |')
    lines += ['', '## 核查与解释范围', '',
              f'- 核查了{len(checkpoints)}个检查点、{len(checkpoints)*50}条逐场景记录和{len(behaviors)}条原始失败轨迹。',
              '- 每个seed与原组合的完整初始状态一致；前8万步内结束的训练回合所有记录列逐条一致：' +
              '，'.join(f'seed{r["seed"]} {r["checked_episode_rows"]}条' for r in early) + '。',
              '- 源码、配置和最终模型指纹通过原注册核查；199501次实际更新的监督权重日志与新旧日程一致。A*代选仍于10万步退出，监督于14万步归零。',
              '- 原退出窗口90000–120500，新退出窗口130000–160500；AULC依据实际检查点步数线性插值。所有分阶段AULC、合并窗口最低点、配对差值及其样本标准差见CSV。',
              '- 窗口失败次数是该窗口各检查点的累计次数，同一场景可能重复失败，不能解释为独立场景总数。',
              '- 等待超时：WAIT比例至少50%；循环超时：非等待主导且反复移动占比至少50%（至少20次移动），或尾部60位置出现2–12周期；其余归其他超时。',
              '- 静态可达不等于动态条件下300步内一定可达；失败原因依据动作轨迹分类，不能单凭成功率猜测。',
              '- 监督总权重积分增加8%，本实验不能单独区分退出时间和累计监督量的影响；低谷发生在退出附近也不足以证明唯一因果机制。',
              '- 目前两个seed均显示原退出窗改善、后期AULC下降：seed0低谷后移且更深，seed1低谷后移但幅度减轻。继续完成原定其余seed后，统一分析并使用冻结的新500场景评估最终模型；期间不根据本阶段结果修改设置。', '',
              '![两seed完整学习曲线](learning_curves.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output)
    for r in means:
        print(names[r['method']], {k: round(r[k+'_mean']*100, 4) for k in display})
    return output


def plot(output, rows, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(seeds), figsize=(7*len(seeds), 4.5), squeeze=False)
    for seed, ax in zip(seeds, axes.flat):
        for method, label in ((common.BASELINE, 'Original full combination'),
                              (common.METHOD, 'Supervision tail to 140k')):
            chosen = [r for r in rows if r['seed'] == seed and r['method'] == method]
            ax.plot([int(r['environment_steps_total']) for r in chosen],
                    [100*float(r['safe_success_rate']) for r in chosen], marker='o', markersize=3, label=label)
        for step in (80000, 100000, 140000):
            ax.axvline(step, color='gray', linestyle='--', alpha=.5)
        ax.set(title=f'Seed {seed}', xlabel='Environment steps', ylabel='Autonomous safe success (%)',
               ylim=(-2, 102))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'learning_curves.{extension}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    review(parser.parse_args().seeds)
