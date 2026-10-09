"""Final-only evaluation of both five-seed methods on a newly frozen shared 500 scenes."""
from __future__ import annotations

import argparse

import torch

import supervision_tail_common as common
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import mean_records
from analyze_runtime_path_pilot import static_distances, trace_behavior
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json, write_records_csv


def evaluate(config, device='auto'):
    torch.set_num_threads(1)
    inputs = common.validate_config(config)
    scenes, _, frozen = common.load_frozen(config)
    # No output/performance until ALL TEN final models have passed preflight.
    planned = [(method, seed, *common.audit_run(config, method, seed))
               for seed in common.SEEDS for method in common.METHODS]
    weights = {str((d / 'model_final.pth').relative_to(common.ROOT)): common.sha256(d / 'model_final.pth')
               for _, _, d, _ in planned}
    output = common.legacy.runtime.unique_check_dir(config, 'independent_eval')
    raw_root = common.ROOT / config['experiment']['output_root'] / output.name
    raw_root.mkdir(parents=True, exist_ok=False)
    summaries, details, behaviors = [], [], []
    distances = static_distances(inputs[0])
    for method, seed, directory, _ in planned:
        agent = common.legacy.runtime.make_agent(config, seed, device)
        model = directory / 'model_final.pth'
        agent.load_weights(model)  # Plain network; no A* or supervision wrapper at evaluation.
        factory = common.FrozenFactory(inputs[0], scenes, trace=True)
        with common.legacy.runtime.preserved_evaluation(agent):
            summary, rows, _ = evaluate_agent(agent, [inputs[0]]*500, max_steps=300, window_size=15,
                reward_config=common.legacy._reward_config(config), terminate_on_collision=True,
                environment_factory=factory, mask_static_invalid_actions=True)
        if [r['scenario_id'] for r in rows] != [s['scenario_id'] for s in scenes]:
            raise RuntimeError('New independent scene identity/order changed.')
        traces, inspected = [], []
        for row, env in zip(rows, factory.environments):
            details.append(dict(method=method, seed=seed, **row))
            if not row['safe_success']:
                trace = dict(scenario_id=row['scenario_id'], safe_success=False,
                             termination_reason=row['termination_reason'], steps=env.trace)
                traces.append(trace)
                b = trace_behavior(trace, distances)
                inspected.append(b)
                behaviors.append(dict(method=method, seed=seed, **{k: v for k, v in b.items() if k != 'tail_60_steps'}))
        write_json(traces, raw_root / method / f'seed_{seed}' / 'failure_trajectories.json')
        if common.sha256(model) != weights[str(model.relative_to(common.ROOT))]:
            raise RuntimeError('Final evaluation changed a source weight.')
        summaries.append(dict(method=method, seed=seed, environment_steps=200000, scene_count=500,
                              safe_success_rate=summary['safe_success_rate'], dynamic_collision_rate=summary['dynamic_collision_rate'],
                              timeout_rate=sum(r['termination_reason']=='timeout' for r in rows)/500,
                              **failure_counts(inspected)))
        write_records_csv(summaries, output / 'summary.csv')
        print(f'[{method} seed={seed}] new500 safe={summary["safe_success_rate"]:.1%}', flush=True)
    metrics = [k for k in summaries[0] if k not in ('method','seed','environment_steps','scene_count')]
    means = mean_records(summaries, metrics)
    paired = [dict(seed=seed, **{k: next(r for r in summaries if r['seed']==seed and r['method']==common.METHOD)[k]
                               - next(r for r in summaries if r['seed']==seed and r['method']==common.BASELINE)[k]
                               for k in metrics}) for seed in common.SEEDS]
    write_records_csv(means, output / 'method_means.csv')
    write_records_csv(paired, output / 'paired_differences.csv')
    write_records_csv(mean_records([dict(method='new_minus_baseline', **r) for r in paired], metrics),
                      output / 'paired_difference_means.csv')
    write_records_csv(details, output / 'scene_details.csv')
    write_records_csv(behaviors, output / 'failure_behaviors.csv')
    audit = dict(complete=True, methods=list(common.METHODS), seeds=list(common.SEEDS), models=10,
                 scene_count=500, scene_rows=5000, evaluation_epsilon=0, action_override=False,
                 oracle_used_to_choose_actions=False, training_started=False,
                 independent_eval_used_for_training_or_schedule=False, checkpoint_selection='final_200000_step_only',
                 source_model_sha256=weights, full_failure_trace_root=str(raw_root.relative_to(common.ROOT)),
                 frozen_scene_sha256=frozen['files_sha256']['independent_final_scenarios.json'],
                 statistical_unit='Five paired training seeds; all ten models share the same 500 scenes.')
    write_json(audit, output / 'verification.json')
    write_json(audit, raw_root / 'completion.json')
    lines = ['# 新500独立场景：原组合与监督尾段延期最终模型', '',
             '相同冻结场景、200000步最终模型、epsilon=0自主动作，无A*代选或动态保护，仅屏蔽静态非法动作。未选择最佳检查点，不据此调参或改变教学日程。', '',
             '| 方法 | 安全成功均值±样本SD（%） | 动态碰撞 | 超时 |', '|---|---:|---:|---:|']
    for r in means:
        lines.append('| ' + r['method'] + ' | ' + ' | '.join(f'{r[k+"_mean"]*100:.2f}±{r[k+"_sample_sd"]*100:.2f}'
                      for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate')) + ' |')
    lines += ['', '逐seed结果与互斥失败分类见summary.csv，所有均值/样本标准差见method_means.csv，配对差值见paired_difference_means.csv，5000条场景明细及全部失败行为完整保存。完整轨迹保留在verification.json记录的outputs目录。',
              '学习效率、两次退出窗口及是否只是推迟低谷，使用原50场景的analyze_supervision_tail报告判断；最终独立评估不能替代学习曲线。新场景与旧50及已查看的旧500均不重合，但仍是同地图同路线池，不代表跨地图泛化，也不保证不与随机训练组合重合。统计单位为5个配对seed。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    config = common.legacy.load_config(common.CONFIG)
    if args.evaluate:
        evaluate(config, args.device)
    else:
        common.validate_config(config)
        scenes, _, _ = common.load_frozen(config)
        pending = [str(common.run_directory(config, common.METHOD, seed)) for seed in common.SEEDS
                   if not (common.run_directory(config, common.METHOD, seed) / 'completion.json').is_file()]
        print(dict(frozen_scenes=len(scenes), pending_new_runs=pending, training_started=False,
                   performance_evaluated=False, evaluation_requires_all_10_final_models=True))


if __name__ == '__main__':
    main()
