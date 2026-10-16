"""Unified seed-level learning efficiency, cost and final-only confirmation; never trains."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

import dqfd_comparison_common as common
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from analyze_supervision_tail import curve_metrics, describe_delay, failure_counts, collect_run as collect_tail
from analyze_teaching_efficiency import mean_records
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json, write_records_csv


def collect(config, method, seed):
    directory, run = common.audit_run(config, method, seed)
    curve = load_csv(directory/'validation_curve.csv')
    x = np.array([int(r['environment_steps_total']) for r in curve])
    y = np.array([float(r['safe_success_rate']) for r in curve])
    if len(curve) != 21 or x[0] != 0 or x[-1] != 200000 or np.any(np.diff(x) <= 0):
        raise ValueError('All original 21 checkpoints are required.')
    scenes = common.read_json(common.ROOT/config['dataset']['validation_reference'])['scenarios']
    grouped = defaultdict(list)
    for r in load_csv(directory/'validation_details.csv'):
        grouped[int(r['environment_steps_total'])].append(r)
    if set(grouped) != set(x):
        raise ValueError('Missing scene detail checkpoints.')
    distances = static_distances(common.validate_config(config)[0])
    behaviors, checkpoints = [], []
    for r in curve:
        step = int(r['environment_steps_total'])
        details = grouped[step]
        if [d['scenario_id'] for d in details] != [s['scenario_id'] for s in scenes]:
            raise ValueError('Original 50 identities/order changed.')
        if any(d['dynamic_route_ids'].split(';') != s['route_ids'] for d, s in zip(details, scenes)):
            raise ValueError('Original validation routes changed.')
        expected = dict(safe_success_rate=np.mean([float(d['safe_success']) for d in details]),
            dynamic_collision_rate=np.mean([float(d['dynamic_collision']) for d in details]),
            timeout_rate=np.mean([d['termination_reason'] == 'timeout' for d in details]))
        if any(not np.isclose(float(r[k]), v) for k, v in expected.items()):
            raise ValueError('Scene results disagree with curve.')
        traces = common.read_json(directory/f'validation_failures_{step:06d}.json')
        failed = {d['scenario_id'] for d in details if not float(d['safe_success'])}
        if len(traces) != len(failed) or {t['scenario_id'] for t in traces} != failed:
            raise ValueError('Incomplete raw failure trajectories.')
        inspected = [trace_behavior(t, distances) for t in traces]
        behaviors.extend(dict(method=method, seed=seed, environment_steps=step,
                              **{k: v for k, v in b.items() if k != 'tail_60_steps'}) for b in inspected)
        checkpoints.append(dict(method=method, seed=seed, **r, **failure_counts(inspected)))
    summary = dict(method=method, seed=seed, **curve_metrics(x, y, config['comparison']),
                   safe_success_rate=y[-1], dynamic_collision_rate=float(curve[-1]['dynamic_collision_rate']),
                   timeout_rate=float(curve[-1]['timeout_rate']))
    for prefix, left, right in (('all', 0, 200000), ('final', 200000, 200000),
                ('original_window', *config['comparison']['original_withdrawal_window']),
                ('new_window', *config['comparison']['new_withdrawal_window']), ('late', 140000, 200000)):
        summary.update({prefix+'_'+k: v for k, v in failure_counts(
            [b for b in behaviors if left <= b['environment_steps'] <= right]).items()})
    if method.startswith('dqfd'):
        learning = load_csv(directory/'dqfd_learning.csv')
        if (sum(int(r['updates']) for r in learning if r['phase'] == 'pretrain') != run['pretraining_updates'] or
                sum(int(r['updates']) for r in learning if r['phase'] == 'online') != 199501):
            raise ValueError('DQfD update logs incomplete.')
        if method == 'dqfd' and any(float(r['one_target_clip_fraction_mean']) or
                                    float(r['n_target_clip_fraction_mean']) for r in learning):
            raise ValueError('A unexpectedly clipped TD targets.')
    elif method == common.METHODS[2]:
        learning = load_csv(directory/'value_learning.csv')
        advice = load_csv(directory/'advice_budget.csv')
        if sum(int(r['updates']) for r in learning) != 199501 or any(int(r['risk_veto']) for r in advice):
            raise ValueError('C update/risk-veto mismatch.')
        for row in learning:
            left, right = max(500, int(row['bin_start_step'])), int(row['bin_end_step'])
            expected = np.mean([max(0., 1-t/100000) for t in range(left, right+1)])
            if not np.isclose(expected, float(row['teacher_margin_weight_mean']), atol=1e-12, rtol=1e-12):
                raise ValueError('C supervision clock changed.')
            if left >= 100000 and int(row['teacher_label_samples']):
                raise ValueError('C supervision survived exit.')
        if any(int(r['applied']) or int(r['requested']) for r in advice if int(r['bin_start_step']) >= 100001):
            raise ValueError('C advice survived exit.')
    elif method in (common.tail.METHOD, common.tail.BASELINE):
        # Original tail diagnostic includes every actual supervision weight/update check.
        collect_tail(common.legacy.load_config(common.tail.CONFIG), method, seed)
    costs = dict(method=method, seed=seed,
        online_updates=run.get('online_updates', run['gradient_updates']),
        pretraining_updates=run.get('pretraining_updates', 0), demo_transition_count=run['demo_transition_count'],
        **{k: run.get(k) for k in ('pretraining_seconds', 'online_training_seconds', 'method_wall_seconds',
            'online_astar_lookups', 'online_astar_searches', 'online_astar_seconds',
            'demo_generation_shared_seconds', 'demo_generation_shared_astar_lookups', 'demo_generation_shared_astar_searches')})
    return summary, checkpoints, behaviors, costs


def aggregate(config, summaries, output, filename='REPORT.md', extra=''):
    metrics = [k for k in summaries[0] if k not in ('method', 'seed', 'environment_steps', 'scene_count')
               and not k.endswith('_step')]
    means = mean_records(summaries, metrics)
    by_key = {(r['method'], r['seed']): r for r in summaries}
    comparisons = [(m, 'advice_bound_margin') for m in common.ALL_METHODS if m != 'advice_bound_margin']
    comparisons.append(('dqfd_bound', 'dqfd'))
    paired = [dict(method=f'{method}_minus_{control}', treated=method, control=control, seed=seed,
                   **{k: by_key[method, seed][k]-by_key[control, seed][k] for k in metrics})
              for method, control in comparisons for seed in common.SEEDS]
    write_records_csv(summaries, output/'summary.csv')
    write_records_csv(means, output/'method_means.csv')
    write_records_csv(paired, output/'paired_differences.csv')
    write_records_csv(mean_records(paired, metrics), output/'paired_difference_means.csv')
    display = [k for k in ('safe_success_aulc', 'aulc_0_80000', 'aulc_80000_100000', 'aulc_100000_140000',
                'aulc_140000_200000', 'last_five_checkpoint_mean', 'safe_success_rate',
                'dynamic_collision_rate', 'timeout_rate') if k in metrics]
    lines = ['# 统一配对seed0–4对比', '', extra, '',
             '统计单位为五个训练seed，均值±样本标准差（ddof=1）；配对差值按同seed相减，不以场景数扩大样本量。', '',
             '| 方法 | '+' | '.join(display)+' |', '|---|'+'---:|'*len(display)]
    for r in means:
        lines.append('| '+r['method']+' | '+' | '.join(
            f'{r[k+"_mean"]*100:.2f}±{r[k+"_sample_sd"]*100:.2f}%' for k in display)+' |')
    lines += ['', '逐seed、分阶段AULC、窗口最低、最终碰撞、等待/循环/其他超时见summary.csv；配对逐seed差值及均值±样本标准差见paired_differences.csv与paired_difference_means.csv。',
              '等待/循环分类沿用原轨迹分析，窗口累计次数包含重复场景检查，不能当成独立失败场景总数。', '']
    (output/filename).write_text('\n'.join(lines), encoding='utf-8')
    return means


def analyze(config):
    common.frozen(config)
    for method in common.ALL_METHODS:
        for seed in common.SEEDS:
            common.audit_run(config, method, seed)
    summaries, checkpoints, behaviors, costs = [], [], [], []
    for method in common.ALL_METHODS:
        for seed in common.SEEDS:
            s, c, b, cost = collect(config, method, seed)
            summaries.append(s); checkpoints.extend(c); behaviors.extend(b); costs.append(cost)
    output = common.legacy.runtime.unique_check_dir(config, 'analysis')
    aggregate(config, summaries, output, extra='原50固定验证场景，epsilon=0，无教学/风险保护。DQfD的0步点是离线预训练完成后；其20000次额外更新和示范生成成本单列，在线AULC不代表等总计算成本。未读取新500表现。')
    write_records_csv(checkpoints, output/'all_validation_checkpoints.csv')
    write_records_csv(behaviors, output/'all_failure_behaviors.csv')
    write_records_csv(costs, output/'training_costs.csv')
    cost_metrics = [k for k in costs[0] if k not in ('method', 'seed')]
    write_records_csv(mean_records(costs, cost_metrics), output/'training_cost_means.csv')
    cost_by_key = {(r['method'], r['seed']): r for r in costs}
    cost_pairs = [(m, 'advice_bound_margin') for m in common.ALL_METHODS if m != 'advice_bound_margin']
    cost_pairs.append(('dqfd_bound', 'dqfd'))
    paired_costs = []
    for treated, control in cost_pairs:
        for seed in common.SEEDS:
            a, b = cost_by_key[treated, seed], cost_by_key[control, seed]
            paired_costs.append(dict(method=f'{treated}_minus_{control}', seed=seed,
                **{k: a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in cost_metrics}))
    write_records_csv(paired_costs, output/'training_cost_paired_differences.csv')
    write_records_csv(mean_records(paired_costs, cost_metrics), output/'training_cost_paired_difference_means.csv')
    by_key = {(r['method'], r['seed']): r for r in summaries}
    delays = [dict(seed=s, **describe_delay(by_key[common.tail.METHOD, s], by_key['advice_bound_margin', s]))
              for s in common.SEEDS]
    write_records_csv(delays, output/'supervision_tail_delay_diagnosis.csv')
    with (output/'REPORT.md').open('a', encoding='utf-8') as handle:
        handle.write('\n尾段是否只是推迟低谷：同时比较原窗90000–120500、新窗130000–160500、合并窗和14–20万步AULC；逐seed判据见supervision_tail_delay_diagnosis.csv。原历史方法未保存的耗时/A*计数留空，不能伪造或用文件时间代替；共享示范成本只发生一次，单列、不重复相加。\n')
    write_json(dict(complete=True, methods=list(common.ALL_METHODS), seeds=list(common.SEEDS),
        checkpoint_count=len(checkpoints), inspected_failures=len(behaviors), training_started=False,
        independent_500_performance_evaluated=False, statistical_unit='paired training seed'), output/'verification.json')
    print(output)
    return output


def evaluate(config, device='auto'):
    torch.set_num_threads(1)
    inputs = common.validate_config(config)
    common.frozen(config)
    scenes, _, frozen = common.tail.load_frozen(common.legacy.load_config(common.tail.CONFIG))
    planned = [(m, s, *common.audit_run(config, m, s)) for m in common.ALL_METHODS for s in common.SEEDS]
    weights = {str((d/'model_final.pth').relative_to(common.ROOT)): common.sha256(d/'model_final.pth')
               for _, _, d, _ in planned}
    output = common.legacy.runtime.unique_check_dir(config, 'independent_eval')
    raw = common.ROOT/config['experiment']['output_root']/output.name
    raw.mkdir(parents=True, exist_ok=False)
    summaries, details, failures = [], [], []
    distances = static_distances(inputs[0])
    for method, seed, directory, _ in planned:
        agent = common.legacy.runtime.make_agent(config, seed, device)
        model = directory/'model_final.pth'
        agent.load_weights(model)
        factory = common.tail.FrozenFactory(inputs[0], scenes, trace=True)
        with common.legacy.runtime.preserved_evaluation(agent):
            summary, rows, _ = evaluate_agent(agent, [inputs[0]]*500, max_steps=300, window_size=15,
                reward_config=common.legacy._reward_config(config), terminate_on_collision=True,
                environment_factory=factory, mask_static_invalid_actions=True)
        if [r['scenario_id'] for r in rows] != [s['scenario_id'] for s in scenes]:
            raise ValueError('Frozen final scene identities/order changed.')
        traces, behavior = [], []
        for row, env in zip(rows, factory.environments):
            details.append(dict(method=method, seed=seed, **row))
            if not row['safe_success']:
                trace = dict(scenario_id=row['scenario_id'], safe_success=False,
                             termination_reason=row['termination_reason'], steps=env.trace)
                traces.append(trace)
                b = trace_behavior(trace, distances)
                behavior.append(b)
                failures.append(dict(method=method, seed=seed, **{k: v for k, v in b.items() if k != 'tail_60_steps'}))
        write_json(traces, raw/method/f'seed_{seed}'/'failure_trajectories.json')
        if common.sha256(model) != weights[str(model.relative_to(common.ROOT))]:
            raise ValueError('Evaluation modified model.')
        summaries.append(dict(method=method, seed=seed, environment_steps=200000, scene_count=500,
            safe_success_rate=summary['safe_success_rate'], dynamic_collision_rate=summary['dynamic_collision_rate'],
            timeout_rate=sum(r['termination_reason'] == 'timeout' for r in rows)/500,
            **failure_counts(behavior)))
        write_records_csv(summaries, output/'summary.csv')
        print(f'[{method} seed={seed}] final new500 safe={summary["safe_success_rate"]:.1%}', flush=True)
    aggregate(config, summaries, output, extra='尾段实验新冻结的同一批500场景，仅200000在线步最终模型，epsilon=0普通D3QN动作，无A*/动态风险保护，不选择最佳检查点，不调参。')
    write_records_csv(details, output/'scene_details.csv')
    write_records_csv(failures, output/'failure_behaviors.csv')
    audit = dict(complete=True, models=len(planned), scene_count=500, scene_rows=len(details),
        methods=list(common.ALL_METHODS), seeds=list(common.SEEDS), training_started=False,
        evaluation_epsilon=0, future_obstacles_used_for_actions=False, action_override=False,
        frozen_scene_sha256=frozen['files_sha256']['independent_final_scenarios.json'],
        source_model_sha256=weights, full_failure_trace_root=str(raw.relative_to(common.ROOT)),
        checkpoint_selection='final_200000_step_only', statistical_unit='five paired training seeds')
    write_json(audit, output/'verification.json'); write_json(audit, raw/'completion.json')
    print(output)
    return output


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--analyze', action='store_true')
    actions.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    config = common.legacy.load_config(common.CONFIG)
    common.validate_config(config); common.frozen(config)
    if args.analyze:
        analyze(config)
    elif args.evaluate:
        evaluate(config, args.device)
    else:
        pending = [str(common.run_directory(config, m, s)) for m in common.ALL_METHODS for s in common.SEEDS
                   if not (common.run_directory(config, m, s)/'completion.json').is_file()]
        print(dict(methods=len(common.ALL_METHODS), total_models=len(common.ALL_METHODS)*5,
                   pending_final_runs=pending, training_started=False, performance_evaluated=False))


if __name__ == '__main__':
    main()
