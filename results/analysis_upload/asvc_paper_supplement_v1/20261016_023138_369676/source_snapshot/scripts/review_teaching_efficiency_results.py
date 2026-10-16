"""Audit completed runs and traces, then summarize evidence; never train or evaluate a suite."""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import teaching_efficiency_common as common
from analyze_teaching_efficiency import aulc, mean_records
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from astar_d3qn.utils.io import write_json, write_records_csv


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(np.isclose(float(actual), expected, rtol=1e-9, atol=1e-10), context)


def stream_array(path):
    """Read a JSON array one object at a time, keeping large trace files off the heap."""
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as handle:
        buffer = ''
        exhausted = False

        def refill():
            nonlocal buffer, exhausted
            chunk = handle.read(1024 * 1024)
            buffer += chunk
            exhausted = not chunk

        def token():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not exhausted:
                refill()
                buffer = buffer.lstrip()
            require(bool(buffer), f'Truncated JSON: {path}')
            value, buffer = buffer[0], buffer[1:]
            return value

        require(token() == '[', f'Expected JSON array: {path}')
        while True:
            first = token()
            if first == ']':
                require(not (buffer + handle.read()).strip(), f'Trailing JSON: {path}')
                return
            buffer = first + buffer
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    break
                except json.JSONDecodeError:
                    require(not exhausted, f'Malformed JSON: {path}')
                    refill()
            yield value
            separator = token()
            require(separator in ',]', f'Missing JSON delimiter: {path}')
            if separator == ']':
                require(not (buffer + handle.read()).strip(), f'Trailing JSON: {path}')
                return


def latest(root, prefix):
    candidates = sorted(p for p in root.glob(prefix + '_*') if (p / 'verification.json').is_file())
    require(bool(candidates), f'No completed {prefix} results under {root}')
    return candidates[-1]


def review(analysis, evaluation):
    torch.set_num_threads(1)
    config = common.legacy.load_config(common.CONFIG)
    inputs = common.validate_config(config)
    scenes, critical, frozen = common.load_frozen(config)
    distances = static_distances(inputs[0])
    old_scenes = common.read_json(common.ROOT / config['dataset']['validation_reference'])['scenarios']
    expected_keys = {(m, s) for m in common.ALL_METHODS for s in common.SEEDS}
    source_metrics = {(r['method'], int(r['seed'])): r for r in load_csv(analysis / 'summary.csv')}
    final_metrics = {(r['method'], int(r['seed'])): r for r in load_csv(evaluation / 'summary.csv')}
    require(set(source_metrics) == set(final_metrics) == expected_keys, 'Expected all 35 model summaries.')
    verification = common.read_json(evaluation / 'verification.json')
    raw_root = common.ROOT / verification['full_failure_trace_root']
    completion = common.read_json(raw_root / 'completion.json')
    require(completion == dict(complete=True, **verification), 'Evaluation completion differs from verification.')
    require(verification['models'] == 35 and verification['scene_count'] == 500
            and verification['scene_rows'] == 17500 and verification['evaluation_epsilon'] == 0
            and not any(verification[k] for k in ('training_started', 'action_override',
                       'oracle_used_to_choose_actions', 'independent_eval_used_for_training_or_schedule')),
            'Final evaluation protocol mismatch.')
    require(verification['frozen_scene_file_sha256'] == frozen['files_sha256']['independent_final_scenarios.json'],
            'Frozen evaluation scene hash mismatch.')
    details = defaultdict(list)
    for row in load_csv(evaluation / 'scene_details.csv'):
        details[row['method'], int(row['seed'])].append(row)
    require(set(details) == expected_keys, 'Unexpected final scene groups.')
    fingerprints, combined, coverage = {}, [], []
    for method in common.ALL_METHODS:
        for seed in common.SEEDS:
            directory, result = common.audit_run(config, method, seed)
            path = directory / 'model_final.pth'
            fingerprints[str(path.relative_to(common.ROOT))] = common.sha256(path)
            curve = load_csv(directory / 'validation_curve.csv')
            checkpoints = defaultdict(list)
            for row in load_csv(directory / 'validation_details.csv'):
                checkpoints[int(row['environment_steps_total'])].append(row)
            x = np.array([int(r['environment_steps_total']) for r in curve])
            y = np.array([float(r['safe_success_rate']) for r in curve])
            require(len(curve) == 21 and x[0] == 0 and x[-1] == 200000 and np.all(np.diff(x) > 0)
                    and set(checkpoints) == set(x), 'Incomplete validation curve.')
            for point in curve:
                entries = checkpoints[int(point['environment_steps_total'])]
                require([r['scenario_id'] for r in entries] == [s['scenario_id'] for s in old_scenes],
                        'Old 50 validation scenes/order changed.')
                for metric in ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate'):
                    values = [r['termination_reason'] == 'timeout' if metric == 'timeout_rate' else
                              float(r[metric.removesuffix('_rate')]) for r in entries]
                    close(point[metric], np.mean(values), 'Validation detail aggregate mismatch.')
            row = source_metrics[method, seed]
            computed = dict(safe_success_aulc=aulc(x, y), last_five_checkpoint_mean=np.mean(y[-5:]),
                            withdrawal_min_safe_success=min(y[(x >= 90000) & (x <= 120500)]))
            for left, right in ((0, 50000), (50000, 100000), (100000, 200000)):
                computed[f'aulc_{left}_{right}'] = aulc(x, y, left, right)
            for key, value in computed.items():
                close(row[key], value, 'Learning metric differs from raw curve.')
            for metric in ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate'):
                close(row[metric], float(curve[-1][metric]), 'Final 50-scene metric mismatch.')
            learning = load_csv(directory / 'value_learning.csv')
            require(sum(int(r['updates']) for r in learning) == 199501, 'Incorrect number of updates.')
            require(not any(int(r['teacher_label_samples']) or float(r['teacher_margin_weight_mean'])
                            for r in learning if int(r['bin_start_step']) >= 100001),
                    'Supervision persisted past 100000 steps.')
            active = [r for r in learning if int(r['bin_start_step']) < 100001]
            close(row['active_replay_label_coverage'], sum(int(r['teacher_label_samples']) for r in active)
                  / (sum(int(r['updates']) for r in active) * 64), 'Replay label coverage mismatch.')
            if method == common.METHOD:
                collection = load_csv(directory / 'collection_coverage.csv')
                sampled = load_csv(directory / 'supervision_coverage.csv')
                advice = load_csv(directory / 'advice_budget.csv')
                require(sum(int(r['collected_transitions']) for r in collection) == 200000,
                        'Collection log does not cover the training budget.')
                # The original linear weight reaches exactly zero at step 100000.
                require(sum(int(r['active_collected_transitions']) for r in collection) == 99999,
                        'Active collection log does not cover the teaching budget.')
                require(not any(int(r['applied']) or int(r['requested']) for r in advice),
                        'New ablation performed an A* override.')
                require(len(sampled) == len(learning) == 200, 'Missing coverage bins.')
                for sample, value in zip(sampled, learning):
                    require(int(sample['bin_end_step']) == int(value['bin_end_step'])
                            and int(sample['updates']) == int(value['updates'])
                            and int(sample['sampled_transitions']) == int(value['updates']) * 64
                            and int(sample['supervised_samples']) == int(value['teacher_label_samples'])
                            == int(sample['eligible_noncollision_match'])
                            <= int(sample['observed_risk_clear_match']) <= int(sample['executed_match'])
                            <= int(sample['teacher_available']) <= int(sample['sampled_transitions']),
                            'Supervision funnel disagrees with actual learning labels.')
                coverage.append(dict(seed=seed,
                    exact_critical_visits=sum(int(r['exact_critical_observation_visits']) for r in collection),
                    exact_visits_before_exit=sum(int(r['exact_critical_observation_visits']) for r in collection
                                                 if int(r['bin_end_step']) <= 100000),
                    exact_visits_after_exit=sum(int(r['exact_critical_observation_visits']) for r in collection
                                                if int(r['bin_start_step']) >= 100001),
                    exact_critical_down=sum(int(r['exact_critical_action_1']) for r in collection),
                    exact_critical_right=sum(int(r['exact_critical_action_3']) for r in collection),
                    exact_critical_collisions=sum(int(r['exact_critical_collisions']) for r in collection),
                    exact_replay_draws=sum(int(r['exact_critical_sample_draws']) for r in sampled),
                    exact_supervised_draws=sum(int(r['exact_critical_supervised_draws']) for r in sampled)))
            entries = details[method, seed]
            require(len(entries) == 500 and [r['scenario_id'] for r in entries] == [s['scenario_id'] for s in scenes],
                    '500 final scenes missing, duplicated or reordered.')
            for entry, scene in zip(entries, scenes):
                require(entry['dynamic_route_ids'].split(';') == scene['route_ids']
                        and int(entry['dynamic_obstacle_count']) == len(scene['obstacles']),
                        'Final scene routes/count disagree with frozen input.')
                close(float(entry['safe_success']) + float(entry['dynamic_collision'])
                      + (entry['termination_reason'] == 'timeout'), 1, 'Unclassified final outcome.')
            final = final_metrics[method, seed]
            require(int(final['scene_count']) == 500 and int(final['environment_steps']) == 200000,
                    'Final evaluation did not use full-budget weights.')
            for key in ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate', 'mean_steps', 'mean_wait_steps'):
                values = [r['termination_reason'] == 'timeout' if key == 'timeout_rate' else
                          float(r[key.removesuffix('_rate').removeprefix('mean_')]) for r in entries]
                close(final[key], np.mean(values), '500-scene summary differs from scene details.')
            combined.append(dict(row, **{'independent_' + k: v for k, v in final.items()
                                         if k not in ('method', 'seed', 'environment_steps', 'scene_count')}))
    require(fingerprints == verification['source_model_sha256'], 'Evaluated weight fingerprints changed.')
    for folder, source_rows in ((analysis, list(source_metrics.values())),
                                (evaluation, list(final_metrics.values()))):
        stored_means = {r['method']: r for r in load_csv(folder / 'method_means.csv')}
        require(set(stored_means) == set(common.ALL_METHODS), 'Expected seven method mean rows.')
        for method in common.ALL_METHODS:
            for key, actual in stored_means[method].items():
                if key.endswith('_mean') or key.endswith('_sample_sd'):
                    suffix = '_sample_sd' if key.endswith('_sample_sd') else '_mean'
                    metric = key.removesuffix(suffix)
                    values = [float(r[metric]) for r in source_rows if r['method'] == method]
                    close(actual, np.std(values, ddof=1) if suffix == '_sample_sd' else np.mean(values),
                          'Stored method mean/sample SD differs from seed results.')
    print('Verified 35 paired completed runs, 735 checkpoints, and 17500 frozen-scene outcomes.', flush=True)
    behaviors = load_csv(evaluation / 'failure_behaviors.csv')
    behavior_index = {(r['method'], int(r['seed']), r['scenario_id']): r for r in behaviors}
    failed_keys = {(m, s, r['scenario_id']) for (m, s), entries in details.items()
                   for r in entries if not float(r['safe_success'])}
    require(set(behavior_index) == failed_keys and len(behavior_index) == len(behaviors),
            'Failed scene/behavior identities differ.')
    selected, audited_keys, failure_counts = [], set(), Counter()
    for method in common.ALL_METHODS:
        for seed in common.SEEDS:
            for trace in stream_array(raw_root / method / f'seed_{seed}' / 'failure_trajectories.json'):
                key = method, seed, trace['scenario_id']
                require(key in failed_keys and key not in audited_keys, 'Unexpected or duplicate failure trace.')
                computed = trace_behavior(trace, distances)
                for name, value in computed.items():
                    if name == 'tail_60_steps':
                        continue
                    stored = behavior_index[key][name]
                    if isinstance(value, bool):
                        require(stored == str(value), 'Failure boolean disagrees with raw trajectory.')
                    elif value is None:
                        require(stored == '', 'Failure period disagrees with raw trajectory.')
                    elif isinstance(value, tuple):
                        require(ast.literal_eval(stored) == value, 'Failure position differs from raw trajectory.')
                    elif isinstance(value, (int, float)):
                        close(stored, value, 'Failure metric differs from raw trajectory.')
                    else:
                        require(stored == value, 'Failure identity/type mismatch.')
                audited_keys.add(key)
                failure_counts[method, trace['termination_reason']] += 1
                if (method == 'advice_bound_margin' and seed == 3) or (
                        method == common.METHOD and seed == 0 and trace['termination_reason'] == 'timeout'):
                    selected.append((method, seed, trace))
            print(f'Audited raw failures: {method} seed={seed}.', flush=True)
    require(audited_keys == failed_keys, 'Missing raw final failure trajectories.')
    exit_failures = []
    for seed in common.SEEDS:
        directory = common.run_directory(config, common.METHOD, seed)
        for point in load_csv(directory / 'validation_curve.csv'):
            step = int(point['environment_steps_total'])
            if not (90000 <= step <= 120500 or (step > 120500 and float(point['safe_success_rate']) < .9)):
                continue
            traces = common.read_json(directory / f'validation_failures_{step:06d}.json')
            require(len(traces) == round(50 * (1 - float(point['safe_success_rate'])))
                    and len({t['scenario_id'] for t in traces}) == len(traces), 'Missing new exit/late failure traces.')
            computed = [trace_behavior(t, distances) for t in traces]
            exit_failures.append(dict(seed=seed, environment_steps=step,
                safe_success_rate=point['safe_success_rate'], failure_count=len(traces),
                dynamic_collisions=sum(t['steps'][-1]['collision_type'] == 'dynamic' for t in traces),
                timeouts=sum(t['termination_reason'] == 'timeout' for t in traces),
                waiting_dominated=sum(t['waiting_dominated'] for t in computed),
                repeated_movement=sum(t['repeated_movement'] for t in computed),
                periodic_tail=sum(t['tail_period'] is not None for t in computed)))
    worst = next(r for r in exit_failures if r['seed'] == 2 and r['environment_steps'] == 100085)
    require(worst['waiting_dominated'] == worst['timeouts'] == 36 and worst['dynamic_collisions'] == 1,
            'Registered worst-exit diagnosis no longer matches raw failures.')
    # Reconstruct existing recorded prefixes only, using recorded actions. No new policy rollout.
    critical_hashes = {r['observation_sha256'] for r in critical}
    scene_lookup = {s['scenario_id']: s for s in scenes}
    retrospective = []
    for method, seed in sorted({(m, s) for m, s, _ in selected}):
        agent = common.legacy.runtime.make_agent(config, seed, 'cpu')
        model = common.run_directory(config, method, seed) / 'model_final.pth'
        agent.load_weights(model)
        with common.legacy.runtime.preserved_evaluation(agent), torch.no_grad():
            for m, s, trace in selected:
                if (m, s) != (method, seed):
                    continue
                factory = common.FrozenFactory(inputs[0], [scene_lookup[trace['scenario_id']]], trace=True)
                env = factory(inputs[0], max_steps=300, reward_config=common.legacy._reward_config(config),
                              terminate_on_collision=True, window_size=15)
                state = env.reset()
                last = trace['steps'][-1]
                for index, recorded in enumerate(trace['steps']):
                    require(list(env.position) == recorded['before']['position']
                            and [list(p) for p in env.dynamic_positions] == recorded['before']['dynamic_positions'],
                            'Saved prefix state differs from frozen environment.')
                    if index == len(trace['steps']) - 1:
                        q = agent.policy_network(torch.as_tensor(state.spatial[None], device=agent.device),
                                                 torch.as_tensor(state.scalars[None], device=agent.device))[0].cpu().numpy()
                        mask = env.action_mask(True)
                        action = int(np.argmax(np.where(mask, q, -np.inf)))
                        require(action == recorded['action'], 'Recorded final action differs from unchanged model.')
                        retrospective.append(dict(method=method, seed=seed, scenario_id=trace['scenario_id'],
                            observation_sha256=common.observation_hash(state),
                            matches_old_critical_state=common.observation_hash(state) in critical_hashes,
                            position=tuple(env.position), chosen_action=action,
                            q_right_minus_down=float(q[3] - q[1]),
                            wait_minus_best_valid_move=float(q[4] - max(q[a] for a in range(4) if mask[a])),
                            collision_route_ids=';'.join(env.dynamic_route_ids[i] for i in last['dynamic_collision_indices']),
                            **{f'q_{a}': float(v) for a, v in enumerate(q)},
                            future_information_used_for_action=False))
                    result = env.step(recorded['action'])
                    state = result.observation
                    actual = env.trace[-1]
                    for field in ('position', 'dynamic_positions', 'collision_type', 'collision_position', 'termination_reason'):
                        require(json.loads(json.dumps(actual[field])) == recorded[field], 'Prefix replay outcome mismatch.')
                    close(actual['reward'], recorded['reward'], 'Prefix replay reward mismatch.')
        require(common.sha256(model) == fingerprints[str(model.relative_to(common.ROOT))], 'Q inspection changed model.')
    output = common.ROOT / config['experiment']['check_root'] / ('review_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    metrics = [k for k in combined[0] if k not in ('method', 'seed', 'withdrawal_min_checkpoint_step', 'final_static_success')]
    means = mean_records(combined, metrics)
    write_records_csv(combined, output / 'per_seed.csv')
    write_records_csv(means, output / 'method_means.csv')
    write_records_csv(coverage, output / 'critical_training_coverage.csv')
    write_records_csv(exit_failures, output / 'new_exit_and_late_failures.csv')
    write_records_csv(retrospective, output / 'reconstructed_failure_q.csv')
    paired = []
    for a, b in ((common.METHOD, 'unguided_bound'), ('advice_bound_margin', common.METHOD),
                 ('advice_bound_margin', 'advice_bound')):
        for seed in common.SEEDS:
            left = next(r for r in combined if r['method'] == a and int(r['seed']) == seed)
            right = next(r for r in combined if r['method'] == b and int(r['seed']) == seed)
            paired.append(dict(method_a=a, method_b=b, seed=seed,
                               **{k + '_difference_pp': (float(left[k]) - float(right[k])) * 100
                                  for k in ('safe_success_aulc', 'aulc_0_50000', 'aulc_50000_100000',
                                            'aulc_100000_200000', 'independent_safe_success_rate')}))
    write_records_csv(paired, output / 'paired_differences.csv')
    paired_means = []
    for a, b in dict.fromkeys((r['method_a'], r['method_b']) for r in paired):
        group = [r for r in paired if (r['method_a'], r['method_b']) == (a, b)]
        item = dict(method_a=a, method_b=b, paired_seed_count=len(group))
        for metric in group[0]:
            if metric.endswith('_difference_pp'):
                values = [r[metric] for r in group]
                item[metric + '_mean'] = float(np.mean(values))
                item[metric + '_sample_sd'] = float(np.std(values, ddof=1))
        paired_means.append(item)
    write_records_csv(paired_means, output / 'paired_difference_means.csv')
    audit = dict(complete=True, training_started=False, suite_evaluation_started=False,
                 input_analysis=str(analysis.relative_to(common.ROOT)), input_evaluation=str(evaluation.relative_to(common.ROOT)),
                 models=35, paired_seeds=5, completed_steps_per_run=200000, gradient_updates_per_run=199501,
                 validation_checkpoints=735, validation_scene_rows=36750, independent_scene_rows=17500,
                 full_raw_failure_trajectories_checked=len(audited_keys), reconstructed_saved_prefixes=len(selected),
                 new_exit_and_late_failure_checkpoints_checked=len(exit_failures),
                 active_collection_steps_per_new_run=99999, zero_teacher_weight_from_step=100000,
                 frozen_scene_sha256=verification['frozen_scene_file_sha256'], source_model_sha256=fingerprints,
                 registered_training_sources_unchanged=True,
                 statistical_unit='Five paired training seeds; sample standard deviation uses ddof=1.',
                 scope='Failure/Q replay is retrospective only; no future information selected any action.')
    write_json(audit, output / 'verification.json')
    write_report(output, analysis, evaluation, combined, means, coverage, retrospective, failure_counts)
    print(output, flush=True)
    return output


def write_report(output, analysis, evaluation, rows, means, coverage, q_rows, failure_counts):
    names = {'unguided_raw': '无A*，普通TD', 'unguided_bound': '无A*＋TD限制',
             'advice_raw': 'A*代选，普通TD', 'advice_bound': 'A*代选＋TD限制',
             'advice_margin': 'A*代选＋动作监督', 'advice_bound_margin': 'A*代选＋动作监督＋TD限制',
             common.METHOD: '无代选＋动作监督＋TD限制（新）'}
    def fmt(row, metric):
        return f'{float(row[metric + "_mean"]) * 100:.2f}±{float(row[metric + "_sample_sd"]) * 100:.2f}'
    lines = ['# 七方法五seed完成结果复核', '',
        '结论：动作监督在没有A*代选时仍明显加速自主学习；代选主要进一步提升前5万步效率。退出低谷、局部错误动作和长期等待仍存在，论文继续定位为学习效率提升，不主张最终避障效果最好。', '',
        '35份模型均完成200000环境步、199501次更新，源码SHA、配对初始化和最终权重SHA通过核对。735个检查点的36750条原50场景记录及17500条独立500场景结果均与汇总一致；全部3496条独立失败轨迹重新计算等待/循环/碰撞指标通过。原训练与评估结果保留，本次没有启动训练或重新评估整套场景。', '',
        '固定40×40地图91701，起点(3,3)、终点(36,36)，每回合3–5个动态障碍；奖励、观测、网络、回放容量10000及衰减规则保持原协议。均从配对随机网络开始，没有复用静态foundation。评估epsilon=0，无A*代选或动态保护，仅屏蔽静态非法动作。', '',
        '500场景在训练前冻结，与原50场景的物理场景及几何路线组合不重复，所有方法使用同一批最终权重和场景。它们来自同一地图、同一路线池，不代表跨地图泛化，也不保证与随机训练组合完全不重合。', '',
        '下表为五seed均值±样本标准差（单位%，ddof=1）；统计单位是配对seed。AULC按原50场景实际检查步数线性插值积分，包含0步，不是训练中A*帮助执行的成功率。', '',
        '| 方法 | 全程AULC | 500场景成功 | 动态碰撞 | 超时 |', '|---|---:|---:|---:|---:|']
    for r in means:
        lines.append(f'| {names[r["method"]]} | {fmt(r,"safe_success_aulc")} | {fmt(r,"independent_safe_success_rate")} | {fmt(r,"independent_dynamic_collision_rate")} | {fmt(r,"independent_timeout_rate")} |')
    lines += ['', '新组相对无A*＋TD限制的AULC提高34.66个百分点，五seed分别提高47.46、33.28、21.10、26.81、44.65个百分点。组合组相对新组再提高8.35个百分点，五seed均提高；收益主要集中在前5万步（阶段AULC 84.65%对49.93%）。50–100k阶段为93.83%对94.73%，100–200k为93.50%对93.70%，并非每阶段都占优。', '',
        '机制支持：无需让A*实际替换动作，匹配实际执行且通过观测风险筛选、非碰撞的动作监督仍有效。前10万步新组回放标签覆盖率43.28%±0.15%，组合组69.71%±0.23%。覆盖率是回放抽样次数比例，同一经验可重复被抽取；行为不同也会改变训练分布，不能仅凭覆盖率证明8.35个百分点来自标签数量。', '',
        '各seed分阶段与退出结果（原50场景，%）：', '',
        '| 方法 | seed | 全程AULC | 0–50k | 50–100k | 100–200k | 最后五次均值 | 退出附近最低 | 最低步数 | 最终成功 | 最终碰撞 | 最终超时 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fields = [names[r['method']], str(r['seed'])] + [f'{float(r[k])*100:.2f}' for k in
            ('safe_success_aulc', 'aulc_0_50000', 'aulc_50000_100000', 'aulc_100000_200000',
             'last_five_checkpoint_mean', 'withdrawal_min_safe_success')] + [str(r['withdrawal_min_checkpoint_step'])] + [f'{float(r[k])*100:.2f}' for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate')]
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', '分阶段等指标的均值及样本标准差均完整保存在method_means.csv；per_seed.csv合并原50场景学习指标与独立500场景最终指标。退出窗口预先固定为90000–120500步，最低值只对应记录的检查点。', '',
        '| 方法 | 500场景seed0–4安全成功率（%） | 失败数：碰撞 / 超时 |', '|---|---|---:|']
    for method in common.ALL_METHODS:
        values = [r for r in rows if r['method'] == method]
        lines.append(f'| {names[method]} | ' + ', '.join(f'{float(r["independent_safe_success_rate"])*100:.1f}' for r in values)
                     + f' | {failure_counts[method,"collision"]} / {failure_counts[method,"timeout"]} |')
    matches = [r for r in q_rows if r['method'] == 'advice_bound_margin' and r['matches_old_critical_state']]
    waits = [r for r in q_rows if r['method'] == common.METHOD]
    lines += ['', '失败原因与证据：', '',
        '- 新组seed2在100085步成功率降到26%：37条失败中36条是等待占主导的超时、1条动态碰撞；110173步仍有29条超时，其中21条等待占主导、8条反复移动。120053步恢复到90%。这组从未使用动作代选，因此低谷不能只归因于停止A*替它选动作。新组seed0/1在约140k还有72%/54%的较晚低谷，分别14/23条失败全部为周期性反复移动导致超时；固定退出窗口不能概括所有训练波动。',
        '- 独立500场景新组seed0有9次动态碰撞和12次超时；12次超时均等待占主导，停在(22,34)，静态仍距终点16步，尾部连续等待249–250步。静态可达不保证动态300步内可达；不把静态BFS剩余距离当作动态可解性证明。',
        f'- 重放上述12条保存的前缀并查询原最终网络，最后一步均选择WAIT；WAIT比最佳静态合法移动的Q高{min(r["wait_minus_best_valid_move"] for r in waits):.5f}–{max(r["wait_minus_best_valid_move"] for r in waits):.5f}。这直接说明当时网络偏好等待，不只是从成功率猜测原因；仍不能把它单独归因于奖励或监督强度。',
        f'- 原组合组seed3在500场景中9次动态碰撞，其中{len(matches)}次碰撞前观测SHA精确匹配原四次失败的两个局部状态，碰撞路线仍为AR17。原网络向右Q比向下高0.02468或0.02044，于是选择危险向右；冻结场景的新组合没有消除此问题。',
        '- 新组seed3在相同两状态最终选择向下，向右减向下Q为−1.96828/−1.30433；但它在500场景仍有6次其他位置的动态碰撞，不能宣称已解决全部避障。',
        '- 新组seed3训练实际访问这两类精确观测7次，均向下、零碰撞；回放抽到439次，其中321次被监督。原组合组没有记录相同细粒度训练经历，不能倒推出它未见过这些状态，也不能据此确定碰撞因果。',
        '- 新组seed2低谷附近Q绝对最大值约11–12，真实TD目标受限于[-6,10]。目标限制不限制网络输出；此次低谷没有出现旧普通TD组上百的Q发散，局部动作排序、等待与回放变化仍需进一步区分。', '',
        '新组精确临界观测覆盖（两个唯一观测SHA；回放次数不等于唯一经验数）：', '',
        '| seed | 访问数 | 退出前 / 后 | 向下 / 向右 | 碰撞 | 回放抽样数 | 被监督抽样数 |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in coverage:
        lines.append(f'| {r["seed"]} | {r["exact_critical_visits"]} | {r["exact_visits_before_exit"]} / {r["exact_visits_after_exit"]} | {r["exact_critical_down"]} / {r["exact_critical_right"]} | {r["exact_critical_collisions"]} | {r["exact_replay_draws"]} | {r["exact_supervised_draws"]} |')
    lines += ['', '临界状态与等待Q仅从记录的真实前缀事后重建；保存动作驱动环境，没有重新选择前缀动作。历史检查点只有策略网络，历史真实TD目标只取原value_learning日志，不用加载后同步的目标网络冒充历史训练目标。未来障碍位置仅用于检查已发生碰撞，不参与动作选择、监督或调度。', '',
        '现在足以形成学习效率论文的内部消融证据；尚不能凭这些结果证明相对其他论文的方法最好。下一步先使用训练经历与原50场景诊断“过度等待”和监督退出的局部动作排序，不自动改时钟或加大监督，不依据这500场景调参。若以后据已查看的500结果提出并调整方法，确认实验应另行冻结未查看的独立场景。', '',
        f'原统一分析：[REPORT.md](../{analysis.name}/REPORT.md)；原独立评估：[REPORT.md](../{evaluation.name}/REPORT.md)。本目录的verification.json记录完整审核范围及35份模型指纹，reconstructed_failure_q.csv保留21条保存前缀最后状态的Q证据。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--evaluation', type=Path)
    args = parser.parse_args()
    root = common.ROOT / 'results/whole_map_91701_teaching_efficiency_v1'
    review((args.analysis or latest(root, 'analysis')).resolve(),
           (args.evaluation or latest(root, 'independent_eval')).resolve())


if __name__ == '__main__':
    main()
