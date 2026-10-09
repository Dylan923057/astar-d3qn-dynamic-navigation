"""Final-only evaluation on 500 frozen scenes. Default checks; --evaluate is explicit."""
from __future__ import annotations

import argparse
from datetime import datetime

import torch

import teaching_efficiency_common as common
from analyze_teaching_efficiency import mean_records
from analyze_runtime_path_pilot import static_distances, trace_behavior
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json, write_records_csv


def evaluate(config, *, device='auto'):
    torch.set_num_threads(1)
    inputs = common.validate_config(config)
    scenes, _, frozen = common.load_frozen(config)
    # Preflight ALL 35 runs before inspecting any held-out performance.
    planned = [(method, seed, *common.audit_run(config, method, seed))
               for seed in common.SEEDS for method in common.ALL_METHODS]
    fingerprints = {str((directory / 'model_final.pth').relative_to(common.ROOT)):
                    common.sha256(directory / 'model_final.pth') for _, _, directory, _ in planned}
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    source_output = common.ROOT / config['experiment']['output_root'] / ('independent_eval_' + stamp)
    source_output.mkdir(parents=True, exist_ok=False)
    report_output = common.legacy.runtime.unique_check_dir(config, 'independent_eval')
    distances = static_distances(inputs[0])
    summaries, details, failures = [], [], []
    for method, seed, directory, _ in planned:
        agent = common.legacy.runtime.make_agent(config, seed, device)
        weight = directory / 'model_final.pth'
        agent.load_weights(weight)  # Plain D3QN only; no teacher, margin wrapper or risk shield.
        factory = common.FrozenFactory(inputs[0], scenes, trace=True)
        with common.legacy.runtime.preserved_evaluation(agent):
            summary, rows, _ = evaluate_agent(agent, [inputs[0]] * 500, max_steps=300,
                                             reward_config=common.legacy._reward_config(config),
                                             terminate_on_collision=True, window_size=15,
                                             environment_factory=factory, mask_static_invalid_actions=True)
        if [r['scenario_id'] for r in rows] != [s['scenario_id'] for s in scenes] or len(rows) != 500:
            raise RuntimeError('Independent scene identity/order changed.')
        if common.sha256(weight) != fingerprints[str(weight.relative_to(common.ROOT))]:
            raise RuntimeError('Evaluation changed a source model.')
        traces = []
        for row, env in zip(rows, factory.environments):
            details.append(dict(method=method, seed=seed, **row))
            if not row['safe_success']:
                trace = dict(scenario_id=row['scenario_id'], safe_success=False,
                             termination_reason=row['termination_reason'], steps=env.trace)
                traces.append(trace)
                failures.append(dict(method=method, seed=seed,
                                     **{k: v for k, v in trace_behavior(trace, distances).items() if k != 'tail_60_steps'}))
        write_json(traces, source_output / method / f'seed_{seed}' / 'failure_trajectories.json')
        summaries.append(dict(method=method, seed=seed, environment_steps=200000, scene_count=500,
                              safe_success_rate=summary['safe_success_rate'],
                              dynamic_collision_rate=summary['dynamic_collision_rate'],
                              timeout_rate=sum(r['termination_reason'] == 'timeout' for r in rows) / 500,
                              mean_steps=summary['mean_steps'], mean_wait_steps=summary['mean_wait_steps']))
        write_records_csv(summaries, report_output / 'summary.csv')
        print(f'[{method} seed={seed}] 500-scene final evaluation: safe={summaries[-1]["safe_success_rate"]:.1%}', flush=True)
    means = mean_records(summaries, ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate', 'mean_steps', 'mean_wait_steps'))
    write_records_csv(means, report_output / 'method_means.csv')
    write_records_csv(details, report_output / 'scene_details.csv')
    write_records_csv(failures, report_output / 'failure_behaviors.csv')
    audit = dict(seeds=list(common.SEEDS), methods=list(common.ALL_METHODS), models=35, scene_count=500,
                 scene_rows=17500, evaluation_epsilon=0, training_started=False,
                 action_override=False, oracle_used_to_choose_actions=False,
                 independent_eval_used_for_training_or_schedule=False,
                 source_model_sha256=fingerprints,
                 frozen_scene_file_sha256=frozen['files_sha256']['independent_final_scenarios.json'],
                 full_failure_trace_root=str(source_output.relative_to(common.ROOT)),
                 device=device, statistical_unit='Paired training seed; all methods share the same 500 scenes.')
    write_json(audit, report_output / 'verification.json')
    write_json(dict(complete=True, **audit), source_output / 'completion.json')
    lines = ['# 同地图500个冻结独立场景：200000步最终模型', '',
             '所有35份模型均epsilon=0、普通网络独立选动作，静态非法动作屏蔽，无A*或风险保护。场景未参与训练、调参或教学时钟。', '',
             '| 方法 | 安全成功均值±样本SD | 动态碰撞均值±样本SD | 超时均值±样本SD |', '|---|---:|---:|---:|']
    lines.extend(f'| {r["method"]} | {r["safe_success_rate_mean"]:.2%}±{r["safe_success_rate_sample_sd"]:.2%} | {r["dynamic_collision_rate_mean"]:.2%}±{r["dynamic_collision_rate_sample_sd"]:.2%} | {r["timeout_rate_mean"]:.2%}±{r["timeout_rate_sample_sd"]:.2%} |' for r in means)
    lines += ['', 'summary.csv为逐seed结果，scene_details.csv保留17500条结果，failure_behaviors.csv区分真实失败类型。完整失败轨迹留在verification.json记录的独立outputs目录。',
              '这是同地图、同起终点、同路线池的独立组合评估，不是跨地图泛化。500场景不能作为500个训练重复；统计单位仍为5个seed。仅评估最终权重，不选择最佳检查点，不据此改变当前教学调度。',
              '学习效率、全程/阶段AULC、最后五次和退出低谷从旧50验证场景的统一分析读取，不能把最终一次500场景评估伪装成学习曲线。']
    (report_output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(report_output, flush=True)
    return report_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    config = common.legacy.load_config(common.CONFIG)
    if args.evaluate:
        evaluate(config, device=args.device)
    else:
        common.validate_config(config)
        scenes, _, frozen = common.load_frozen(config)
        missing = [str(common.run_directory(config, method, seed) / 'model_final.pth')
                   for method in common.ALL_METHODS for seed in common.SEEDS
                   if not (common.run_directory(config, method, seed) / 'completion.json').is_file()
                   or not (common.run_directory(config, method, seed) / 'model_final.pth').is_file()]
        print(dict(frozen_scenes=len(scenes), performance_evaluated=False, training_started=False,
                   pending_final_models=missing, evaluation_requires_all_35_completed_runs=True))


if __name__ == '__main__':
    main()
