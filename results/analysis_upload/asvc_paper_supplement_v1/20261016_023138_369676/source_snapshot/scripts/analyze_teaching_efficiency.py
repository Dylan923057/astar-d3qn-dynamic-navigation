"""Unified paired metrics and retrospective withdrawal diagnostics; never trains."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

import teaching_efficiency_common as common
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from astar_d3qn.utils.io import write_json, write_records_csv


def aulc(x, y, left=0, right=200000):
    inside = x[(x > left) & (x < right)]
    steps = np.concatenate(([left], inside, [right]))
    values = np.interp(steps, x, y)
    return float(sum(np.diff(steps) * (values[:-1] + values[1:]) / 2) / (right - left))


def mean_records(rows, metrics):
    methods = list(dict.fromkeys(row['method'] for row in rows))
    result = []
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        row = dict(method=method, seed_count=len(subset))
        for key in metrics:
            values = [float(r[key]) for r in subset if r.get(key) is not None]
            row[key + '_mean'] = float(np.mean(values)) if values else None
            row[key + '_sample_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(row)
    return result


def collect_metrics(config, methods):
    summaries, curves, withdrawals, values, coverage, failures = [], [], [], [], [], []
    settings = config['teaching_efficiency']
    problem = common.validate_config(config)[0]
    distances = static_distances(problem)
    expected_scene_ids = [s['scenario_id'] for s in common.read_json(common.ROOT / config['dataset']['validation_reference'])['scenarios']]
    for seed in common.SEEDS:
        for method in methods:
            directory, result = common.audit_run(config, method, seed)
            rows = load_csv(directory / 'validation_curve.csv')
            x = np.array([int(r['environment_steps_total']) for r in rows])
            y = np.array([float(r['safe_success_rate']) for r in rows])
            if len(rows) != 21 or x[0] != 0 or x[-1] != 200000 or np.any(np.diff(x) <= 0):
                raise ValueError('Expected all 21 original-timed formal checkpoints.')
            details = defaultdict(list)
            for r in load_csv(directory / 'validation_details.csv'):
                details[int(r['environment_steps_total'])].append(r)
            if set(details) != set(x):
                raise ValueError('Missing or extra validation detail checkpoints.')
            for r in rows:
                step = int(r['environment_steps_total'])
                entries = details[step]
                if [s['scenario_id'] for s in entries] != expected_scene_ids:
                    raise ValueError('Old 50 validation scene identities/order changed.')
                expected = dict(safe_success_rate=np.mean([float(s['safe_success']) for s in entries]),
                                dynamic_collision_rate=np.mean([float(s['dynamic_collision']) for s in entries]),
                                timeout_rate=np.mean([s['termination_reason'] == 'timeout' for s in entries]))
                if any(not np.isclose(float(r[k]), v) for k, v in expected.items()):
                    raise ValueError('Validation aggregates disagree with all scene details.')
                curves.append(dict(method=method, seed=seed, **r))
                if settings['withdrawal_window'][0] <= step <= settings['withdrawal_window'][1]:
                    traces = common.read_json(directory / f'validation_failures_{step:06d}.json')
                    failed_ids = {s['scenario_id'] for s in entries if not float(s['safe_success'])}
                    if len(traces) != len(failed_ids) or {t['scenario_id'] for t in traces} != failed_ids:
                        raise ValueError('Missing or duplicate withdrawal failures.')
                    behavior = [trace_behavior(t, distances) for t in traces]
                    withdrawals.append(dict(method=method, seed=seed, environment_steps=step,
                                            **expected, failure_count=len(behavior),
                                            waiting_dominated=sum(b['waiting_dominated'] for b in behavior),
                                            repeated_movement=sum(b['repeated_movement'] for b in behavior),
                                            periodic_tail=sum(b['tail_period'] is not None for b in behavior),
                                            static_remaining_one_step=sum(b['final_static_remaining_steps'] == 1 for b in behavior)))
                    failures.extend(dict(method=method, seed=seed, environment_steps=step,
                                         **{k: v for k, v in b.items() if k != 'tail_60_steps'}) for b in behavior)
            learning = load_csv(directory / 'value_learning.csv')
            if sum(int(r['updates']) for r in learning) != 199501 or int(learning[-1]['bin_end_step']) != 200000:
                raise ValueError('Value logs do not cover all 199501 updates.')
            if any(int(r['teacher_label_samples']) or float(r['teacher_margin_weight_mean'])
                   for r in learning if int(r['bin_start_step']) >= 100001):
                raise ValueError('Supervision persisted past the registered clock.')
            for r in learning:
                denom = int(r['updates']) * result['effective_training']['batch_size']
                count = int(r['teacher_label_samples'])
                if not 0 <= count <= denom:
                    raise ValueError('Impossible label coverage.')
                coverage.append(dict(method=method, seed=seed, bin_start_step=int(r['bin_start_step']),
                                     bin_end_step=int(r['bin_end_step']), sampled_transitions=denom,
                                     teacher_label_samples=count, teacher_label_coverage=count / denom,
                                     unit='Replay sample draws; the same collected transition can repeat.'))
                if settings['withdrawal_window'][0] <= int(r['bin_end_step']) <= settings['withdrawal_window'][1]:
                    values.append(dict(method=method, seed=seed, **r))
            near = (x >= settings['withdrawal_window'][0]) & (x <= settings['withdrawal_window'][1])
            phase = {f'aulc_{left}_{right}': aulc(x, y, left, right) for left, right in
                     zip(settings['phase_boundaries'], settings['phase_boundaries'][1:])}
            active_coverage = [r for r in coverage if r['method'] == method and r['seed'] == seed and r['bin_start_step'] < 100001]
            summaries.append(dict(method=method, seed=seed, safe_success_rate=y[-1],
                                  dynamic_collision_rate=float(rows[-1]['dynamic_collision_rate']),
                                  timeout_rate=float(rows[-1]['timeout_rate']), safe_success_aulc=aulc(x, y),
                                  **phase, last_five_checkpoint_mean=float(np.mean(y[-5:])),
                                  withdrawal_min_safe_success=float(np.min(y[near])),
                                  withdrawal_min_checkpoint_step=int(x[near][np.argmin(y[near])]),
                                  active_replay_label_coverage=(sum(r['teacher_label_samples'] for r in active_coverage)
                                                               / sum(r['sampled_transitions'] for r in active_coverage)),
                                  final_static_success=result['final_static']['safe_success']))
    return summaries, curves, withdrawals, values, coverage, failures


def critical_q_diagnostics(config, inputs, methods, critical):
    rows, weights = [], {}
    agent = common.legacy.runtime.make_agent(config, 3, 'cpu')
    teacher = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG), inputs[0], 3,
                                       'advice_bound_margin', 'cpu')
    for method in methods:
        directory, _ = common.audit_run(config, method, 3)
        curve = load_csv(directory / 'validation_curve.csv')
        steps = [int(r['environment_steps_total']) for r in curve if
                 90000 <= int(r['environment_steps_total']) <= 120500] + [200000]
        for step in steps:
            path = directory / f'model_step_{step:06d}.pth'
            if step == 200000:
                path = directory / 'model_final.pth'
            before = common.sha256(path)
            agent.load_weights(path)
            with common.legacy.runtime.preserved_evaluation(agent):
                for record in critical:
                    state = common.decode_observation(record)
                    with torch.no_grad():
                        q = agent.policy_network(torch.as_tensor(state.spatial[None], device=agent.device),
                                                 torch.as_tensor(state.scalars[None], device=agent.device))[0].cpu().numpy()
                    mask = np.asarray(record['static_action_mask'], dtype=bool)
                    action = int(np.argmax(np.where(mask, q, -np.inf)))
                    teacher_action = teacher.static_advice(state)
                    row = dict(method=method, seed=3, environment_steps=step,
                               scenario_id=record['scenario_id'], position=record['position'],
                               observation_sha256=record['observation_sha256'],
                               chosen_action=action, q_right_minus_down=float(q[3] - q[1]),
                               static_teacher_action=teacher_action,
                               teacher_observed_risk=teacher.observed_risk(state, teacher_action),
                               right_observed_risk=teacher.observed_risk(state, 3),
                               down_observed_risk=teacher.observed_risk(state, 1),
                               actual_training_target_network_available=False,
                               scope='Same-state candidate Q comparison, not a counterfactual closed-loop rollout.')
                    row.update({f'q_{a}': float(value) for a, value in enumerate(q)})
                    rows.append(row)
            if common.sha256(path) != before:
                raise RuntimeError('Read-only Q inspection changed a source checkpoint.')
            weights[str(path.relative_to(common.ROOT))] = before
    return rows, weights


def training_route_exposure(config, methods, critical):
    routes = {route for state in critical for route in state['collision_route_ids']}
    rows = []
    for method in methods:
        directory, _ = common.audit_run(config, method, 3)
        training = load_csv(directory / 'training.csv')
        previous, counts = 0, defaultdict(lambda: [0, 0])
        for record in training:
            end = int(record['environment_steps_total'])
            start = previous + 1
            previous = end
            relevant = bool(routes.intersection(record['dynamic_route_ids'].split(';')))
            for left, right in ((0, 50000), (50000, 100000), (100000, 200000), (90000, 120500)):
                overlap = max(0, min(end, right) - max(start - 1, left))
                if relevant and overlap:
                    counts[left, right][0] += 1
                    counts[left, right][1] += overlap
        for (left, right), (episodes, steps) in counts.items():
            rows.append(dict(method=method, seed=3, start_step_exclusive=left, end_step_inclusive=right,
                             critical_route_ids=';'.join(sorted(routes)), episodes_with_route=episodes,
                             environment_steps_in_route_scenes=steps,
                             exact_state_action_coverage_available=method == common.METHOD,
                             initial_phase_logged=False,
                             limitation='Route inclusion only; it does not prove visits, actions or supervision at a critical state.'))
    return rows


def analyze(config, *, existing_only=False):
    torch.set_num_threads(1)
    inputs = common.validate_config(config)
    _, critical, _ = common.load_frozen(config)
    methods = common.legacy.METHODS if existing_only else common.ALL_METHODS
    for method in methods:
        for seed in common.SEEDS:
            common.audit_run(config, method, seed)
    summaries, curves, withdrawals, values, coverage, failures = collect_metrics(config, methods)
    critical_rows, weights = critical_q_diagnostics(config, inputs, methods, critical)
    exposure = training_route_exposure(config, methods, critical)
    directory = common.legacy.runtime.unique_check_dir(config, 'analysis')
    metrics = ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate', 'safe_success_aulc',
               'aulc_0_50000', 'aulc_50000_100000', 'aulc_100000_200000',
               'last_five_checkpoint_mean', 'withdrawal_min_safe_success', 'active_replay_label_coverage')
    means = mean_records(summaries, metrics)
    for filename, records in (('summary.csv', summaries), ('method_means.csv', means),
                              ('all_validation_checkpoints.csv', curves), ('withdrawal_failures.csv', failures),
                              ('withdrawal_behavior_summary.csv', withdrawals), ('withdrawal_value_learning.csv', values),
                              ('supervision_coverage.csv', coverage), ('critical_state_q.csv', critical_rows),
                              ('critical_training_route_exposure.csv', exposure)):
        write_records_csv(records, directory / filename)
    if not existing_only:
        collections, sampled = [], []
        for seed in common.SEEDS:
            run = common.run_directory(config, common.METHOD, seed)
            collected = load_csv(run / 'collection_coverage.csv')
            draws = load_csv(run / 'supervision_coverage.csv')
            values_by_start = {int(r['bin_start_step']): int(r['teacher_label_samples'])
                               for r in load_csv(run / 'value_learning.csv')}
            if (sum(int(r['collected_transitions']) for r in collected) != 200000
                    or sum(int(r['sampled_transitions']) for r in draws) != 199501 * 64
                    or any(int(r['supervised_samples']) != values_by_start[int(r['bin_start_step'])] for r in draws)):
                raise ValueError('New ablation collected/sample coverage does not match the training/value logs.')
            collections.extend(dict(method=common.METHOD, seed=seed, **r) for r in collected)
            sampled.extend(dict(method=common.METHOD, seed=seed, **r) for r in draws)
        write_records_csv(collections, directory / 'new_ablation_collection_coverage.csv')
        write_records_csv(sampled, directory / 'new_ablation_sampled_coverage.csv')
    paired = []
    pairs = [('unguided_bound', 'unguided_raw'), ('advice_bound', 'unguided_bound'),
             ('advice_bound_margin', 'advice_bound')]
    if not existing_only:
        pairs += [(common.METHOD, 'unguided_bound'), ('advice_bound_margin', common.METHOD)]
    by_key = {(r['method'], r['seed']): r for r in summaries}
    for treated, control in pairs:
        for metric in metrics:
            differences = [by_key[treated, seed][metric] - by_key[control, seed][metric] for seed in common.SEEDS]
            paired.append(dict(treated=treated, control=control, metric=metric,
                               **{f'seed_{s}_difference': d for s, d in zip(common.SEEDS, differences)},
                               mean_difference=float(np.mean(differences)),
                               sample_sd_difference=float(np.std(differences, ddof=1))))
    write_records_csv(paired, directory / 'paired_differences.csv')
    write_json(dict(methods=list(methods), seeds=list(common.SEEDS), source_checkpoints_sha256=weights,
                    training_started=False, independent_500_performance_evaluated=False,
                    all_50_validation_checkpoints_audited=True,
                    statistical_unit='Paired training seed; scene and checkpoint observations are correlated.',
                    exact_old_training_state_action_coverage='Unavailable: no per-transition observations/actions or final replay snapshot was saved.',
                    historical_td_targets='Use recorded value_learning.csv only; weight files contain policy only, not historical target networks.'),
               directory / 'verification.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for method in methods:
        runs = [[r for r in curves if r['method'] == method and r['seed'] == seed] for seed in common.SEEDS]
        x = np.mean([[int(r['environment_steps_total']) for r in run] for run in runs], axis=0)
        y = np.array([[float(r['safe_success_rate']) * 100 for r in run] for run in runs])
        avg, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
        for ax in axes:
            ax.plot(x, avg, label=method)
            ax.fill_between(x, avg - sd, avg + sd, alpha=.12)
            ax.axvline(100000, color='grey', linestyle='--')
            ax.set(xlabel='Environment steps', ylabel='Autonomous safe success (%)', ylim=(-2, 102))
    axes[1].set_xlim(80000, 130000)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / ('mean_validation_curves.' + suffix), dpi=160)
    plt.close(fig)
    lines = ['# A*教学效率：配对汇总与退出诊断', '',
             '固定地图、固定起终点、3–5动态障碍；全部旧50场景验证均为网络自主动作。未评估新500场景，未启动训练。', '',
             '| 方法 | 最终安全均值±样本SD | AULC均值±样本SD | 退出窗口最低成功率均值 |', '|---|---:|---:|---:|']
    lines.extend(f'| {r["method"]} | {r["safe_success_rate_mean"]:.2%}±{r["safe_success_rate_sample_sd"]:.2%} | {r["safe_success_aulc_mean"]:.2%}±{r["safe_success_aulc_sample_sd"]:.2%} | {r["withdrawal_min_safe_success_mean"]:.2%} |' for r in means)
    lines += ['', '逐seed与全部要求指标见summary.csv；均值和样本SD见method_means.csv；阶段边界为0/50000/100000/200000，退出窗口为90000–120500，按实际检查点计算。',
              'withdrawal_behavior_summary.csv与withdrawal_failures.csv区分等待、反复移动、周期尾段及静态剩余距离；行为标签可重叠。',
              'withdrawal_value_learning.csv保留真实训练Q与TD目标日志；critical_state_q.csv比较四个相同观测状态的候选Q与向右减向下差值。历史权重只有策略网络，不能用加载后同步的网络冒充原训练目标网络。',
              '四次失败对应两个不同的局部观测SHA。静态A*在这些状态均建议向下；观测筛选判定向下风险通过、向右风险未通过。失败是网络自主向右，不是当时教师建议向右；这仍不能单独证明为何价值排序学错。',
              'supervision_coverage.csv按监督标签样本数/(updates×batch64)计算；分母是回放抽样次数，不是独立经验条数。',
              'critical_training_route_exposure.csv只表示含AR17等碰撞路线的场景曝光，不能证明机器人访问或监督过碰撞状态。旧逐步状态、动作与回放未保存，精确经历覆盖无法事后恢复。新组另记录实际收集标签及关键位置、迎面观测、精确观测SHA和动作/碰撞覆盖。',
              '以配对训练seed为单位，不把场景或检查点当独立样本；当前论文定位为学习效率，不主张最终避障全面最好。',
              '新消融尚未训练，当前仅复用原六组。' if existing_only else '新消融与原六组按seed配对；原实验数据保留。']
    (directory / 'REPORT.md').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    print(directory, flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-only', action='store_true', help='Diagnose the completed original six methods before the new ablation is trained.')
    args = parser.parse_args()
    analyze(common.legacy.load_config(common.CONFIG), existing_only=args.existing_only)


if __name__ == '__main__':
    main()
