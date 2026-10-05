"""Audit completed runtime-guidance runs; evaluate final weights without training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import analyze_runtime_path_pilot as helpers
import run_runtime_path_guidance as entry
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/whole_map_91701_runtime_path_v1"


def audit_run(directory):
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    budget = result["config"]["training"]["max_environment_steps"]
    if budget != 200000 or result["training_mode"] != "formal" or result["integration_only"]:
        raise ValueError(f"Not the registered formal run: {directory}")
    if not completion["complete"] or result["environment_steps"] != budget or completion["environment_steps"] != budget:
        raise ValueError(f"Incomplete run: {directory}")
    if result["gradient_updates"] != 199501 or result["demo_transition_count"] != 0 or result["online_replay_capacity"] != 10000:
        raise ValueError(f"Budget/replay mismatch: {directory}")
    if not (directory / "model_final.pth").is_file():
        raise FileNotFoundError(directory / "model_final.pth")
    curves = helpers.load_csv(directory / "validation_curve.csv")
    details = helpers.load_csv(directory / "validation_details.csv")
    groups = defaultdict(list)
    for detail in details:
        groups[int(detail["environment_steps_total"])].append(detail)
    steps = [int(row["environment_steps_total"]) for row in curves]
    if len(steps) != 21 or steps[0] != 0 or steps[-1] != budget or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Expected initial, nineteen intermediate, and final validations.")
    if set(groups) != set(steps):
        raise ValueError("Validation curves and detail checkpoints differ.")
    scenes = None
    for curve in curves:
        values = groups[int(curve["environment_steps_total"])]
        if len(values) != 50 or len({value["scenario_id"] for value in values}) != 50:
            raise ValueError("Missing or duplicated validation scene.")
        current = [(v["scenario_id"], v["dynamic_route_ids"], v["dynamic_obstacle_count"]) for v in values]
        if scenes is None:
            scenes = current
        elif current != scenes:
            raise ValueError("Validation scenarios changed across checkpoints.")
        expected = {
            "safe_success_rate": sum(float(v["safe_success"]) for v in values) / 50,
            "dynamic_collision_rate": sum(float(v["dynamic_collision"]) for v in values) / 50,
            "timeout_rate": sum(v["termination_reason"] == "timeout" for v in values) / 50,
        }
        for key, value in expected.items():
            if not np.isclose(float(curve[key]), value):
                raise ValueError(f"Aggregate/detail mismatch: {directory}/{key}")
            if int(curve["environment_steps_total"]) == budget and not np.isclose(result["final_validation"][key], value):
                raise ValueError("Final result differs from final validation.")
    training = helpers.load_csv(directory / "training.csv")
    if int(training[-1]["environment_steps_total"]) != budget or int(training[-1]["gradient_updates_total"]) != 199501:
        raise ValueError("Training log does not match completion metadata.")
    return result, curves, groups, scenes


def learning_metrics(curves):
    x = [int(row["environment_steps_total"]) for row in curves]
    y = [float(row["safe_success_rate"]) for row in curves]
    metrics = {"safe_success_aulc_0to200k": sum((b - a) * (u + v) / 2 for a, b, u, v in zip(x, x[1:], y, y[1:])) / 200000}
    for threshold, label in ((0.9, "90"), (0.95, "95")):
        metrics[f"first_observed_ge{label}_step"] = next((step for step, rate in zip(x, y) if rate >= threshold), None)
        metrics[f"all_remaining_checks_ge{label}_from_step"] = next((x[i] for i in range(len(x)) if all(rate >= threshold for rate in y[i:])), None)
    metrics["best_observed_safe_success_rate"] = max(y)
    metrics["last_five_checks_mean_safe_success_rate"] = float(np.mean(y[-5:]))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, choices=entry.REGISTERED_SEEDS,
                        default=list(entry.REGISTERED_SEEDS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use unique seeds.")
    seeds = tuple(sorted(args.seeds))
    torch.set_num_threads(1)
    output = ROOT / "results/whole_map_91701_runtime_path_v1" / f"analysis_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    summaries, curves_out, behaviors, static_rows, final_failures = [], [], [], [], []
    runs, common_scenes, config, common_initial_frames = {}, None, None, None
    for method in entry.METHODS:
        for seed in seeds:
            directory = SOURCE / method / f"seed_{seed}"
            result, curves, details, scenes = audit_run(directory)
            if config is None:
                config = result["config"]
                inputs = entry.validate_config(config)
                registered_check, registered_scenes = entry.check(config, inputs, seeds)
                write_json(registered_check, output / "registered_check.json")
                write_json(registered_scenes, output / "fixed_validation_scenarios.json")
            elif result["config"] != config:
                raise ValueError("Registered configuration differs between runs.")
            if common_scenes is None:
                common_scenes = scenes
            elif scenes != common_scenes:
                raise ValueError("Methods/seeds used different validation sequences.")
            problem = inputs[0]
            if result["map"]["grid_sha256"] != problem.grid_sha256 or result["route_pool_design_sha256"] != inputs[2]["design_sha256"]:
                raise ValueError("Run metadata does not match registered map and route pool.")
            runs[method, seed] = result
            distances = helpers.static_distances(problem)
            final = result["final_validation"]
            summary = {"method": method, "seed": seed, "environment_steps": result["environment_steps"],
                       "gradient_updates": result["gradient_updates"],
                       "final_safe_success_rate": final["safe_success_rate"],
                       "final_dynamic_collision_rate": final["dynamic_collision_rate"],
                       "final_timeout_rate": final["timeout_rate"], **learning_metrics(curves)}
            for curve in curves:
                step = int(curve["environment_steps_total"])
                traces = json.loads((directory / f"validation_failures_{step:06d}.json").read_text(encoding="utf-8"))
                indexed = {v["scenario_id"]: v for v in details[step]}
                expected = {v["scenario_id"] for v in details[step] if not float(v["safe_success"])}
                if len(traces) != len(expected) or {t["scenario_id"] for t in traces} != expected:
                    raise ValueError("Failure traces do not match validation details.")
                # All step-zero episodes fail, allowing direct physical-scene comparison across all runs.
                if step == 0:
                    initial_frames = [(t["scenario_id"], t["obstacles"], t["steps"][0]["before"]["dynamic_positions"]) for t in traces]
                    if len(initial_frames) != 50:
                        raise ValueError("Expected fifty recorded initial evaluation trajectories.")
                    if common_initial_frames is None:
                        common_initial_frames = initial_frames
                    elif initial_frames != common_initial_frames:
                        raise ValueError("Actual initial obstacle states differ between runs.")
                checkpoint = []
                for trace in traces:
                    behavior = helpers.trace_behavior(trace, distances)
                    detail = indexed[trace["scenario_id"]]
                    if behavior["steps"] != int(detail["steps"]) or behavior["wait_steps"] != int(detail["wait_steps"]) or behavior["termination_reason"] != detail["termination_reason"]:
                        raise ValueError("Trace and detail behavior counts disagree.")
                    behavior.update(method=method, seed=seed, environment_steps_total=step)
                    behaviors.append({key: value for key, value in behavior.items() if key != "tail_60_steps"})
                    checkpoint.append(behavior)
                    if step == 200000:
                        write_json(trace, output / f"final_{method}_seed{seed}_{trace['scenario_id']}.json")
                        final_failures.append({**behavior, "terminal_transition": trace["steps"][-1]})
                curves_out.append({"method": method, "seed": seed, **curve,
                                   "failure_waiting_dominated_count": sum(b["waiting_dominated"] for b in checkpoint),
                                   "failure_repeated_movement_count": sum(b["repeated_movement"] for b in checkpoint),
                                   "failure_periodic_tail_count": sum(b["tail_period"] is not None for b in checkpoint)})
                if step == 200000:
                    timeouts = [b for b in checkpoint if b["termination_reason"] == "timeout"]
                    summary.update(final_failure_count=len(checkpoint), final_timeout_count=len(timeouts),
                                   final_waiting_dominated_timeouts=sum(b["waiting_dominated"] for b in timeouts),
                                   final_repeated_movement_timeouts=sum(b["repeated_movement"] for b in timeouts),
                                   final_periodic_tail_timeouts=sum(b["tail_period"] is not None for b in timeouts))
            static_summary, static_detail, static_trace = helpers.static_evaluation(config, problem, method, seed, directory / "model_final.pth")
            static_behavior = helpers.trace_behavior(static_trace, distances)
            write_json(static_trace, output / f"static_{method}_seed{seed}.json")
            static_rows.append({"method": method, "seed": seed, **static_detail,
                                **{key: value for key, value in static_behavior.items() if key not in {"tail_60_steps", "scenario_id"}}})
            summary.update(static_original_task_safe_success=static_summary["safe_success_rate"],
                           static_steps=static_detail["steps"], static_termination_reason=static_detail["termination_reason"],
                           static_wait_steps=static_behavior["wait_steps"])
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    initializations = {value["seed"]: value for value in registered_check["initializations"]}
    for seed in seeds:
        first, second = runs["unguided", seed], runs["path_guided", seed]
        if first["initial_state_sha256"] != second["initial_state_sha256"] or first["effective_training"] != second["effective_training"]:
            raise ValueError("Paired initial states or effective training parameters differ.")
        if first["initial_state_sha256"] != initializations[seed]["initial_state_sha256"]:
            raise ValueError("Run initialization fingerprint differs from the registered check.")
    write_records_csv(summaries, output / "summary.csv")
    write_records_csv(curves_out, output / "validation_curves.csv")
    write_records_csv(behaviors, output / "failure_behavior.csv")
    write_records_csv(static_rows, output / "static_diagnostics.csv")
    write_json(final_failures, output / "final_failure_examples.json")
    write_json({"all_runs_complete": True, "run_count": len(summaries), "seeds": seeds,
                "validation_checkpoint_count": len(curves_out), "same_initial_states_per_seed": True,
                "same_registered_config_and_effective_parameters_per_seed": True,
                "same_fixed_validation_sequence_and_initial_obstacle_states": True,
                "aggregate_detail_and_trace_counts_match": True, "test_data_used": False,
                "training_started_by_analysis": False, "static_evaluation_epsilon": 0,
                "scope": f"One map, fixed endpoints, {len(seeds)} training seeds, fifty fixed validation scenes; no generalization or statistical superiority established.",
                "aulc_definition": "Trapezoidal integral at actual evaluation steps, divided by 200000.",
                "threshold_definition": "First observed checkpoint reaching threshold; later checkpoints may fall below it. Remaining-check threshold is retrospective, not a continuous guarantee.",
                "failure_categories_overlap": True}, output / "verification.json")
    plot(curves_out, output)
    report(summaries, final_failures, output)
    print(f"Analysis output: {output}", flush=True)


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
    seeds = sorted({row["seed"] for row in rows})
    for method in entry.METHODS:
        for seed in seeds:
            values = [r for r in rows if r["method"] == method and r["seed"] == seed]
            for axis, key in zip(axes, ("safe_success_rate", "dynamic_collision_rate", "timeout_rate")):
                axis.plot([int(r["environment_steps_total"]) / 1000 for r in values],
                          [float(r[key]) for r in values], marker="o", label=f"{method}, seed {seed}")
    for axis, title in zip(axes, ("Safe success", "Dynamic collision", "Timeout")):
        axis.set_title(title)
        axis.set_xlabel("Environment steps (thousands)")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Fraction of 50 validation episodes")
    axes[-1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / "validation_curves.png", dpi=180)
    figure.savefig(output / "validation_curves.pdf")
    plt.close(figure)

    # Matched seeds are easier to compare in separate panels than in ten overlapping curves.
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    colors = {"unguided": "tab:blue", "path_guided": "tab:orange"}
    for axis, seed in zip(axes.flat, seeds):
        for method in entry.METHODS:
            values = [r for r in rows if r["method"] == method and r["seed"] == seed]
            axis.plot([int(r["environment_steps_total"]) / 1000 for r in values],
                      [float(r["safe_success_rate"]) for r in values], marker="o", markersize=3,
                      color=colors[method], label=method)
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Environment steps (thousands)")
        axis.set_ylabel("Safe success fraction")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(seeds):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "paired_seed_success.png", dpi=180)
    figure.savefig(output / "paired_seed_success.pdf")
    plt.close(figure)


def report(summaries, failures, output):
    seeds = sorted({row["seed"] for row in summaries})
    lines = ["# 全图动态场景：运行时A*路线输入，20万步结果", "",
             f"seed {seeds}，{len(summaries)}组均完成200000环境步、199501次更新。每组21次验证，包含完整训练曲线。无A*示范经验、无旧foundation加载。",
             "同seed的初始网络、优化器和随机数状态一致；配置一致；验证使用相同50个固定场景，epsilon=0。未使用test、未启动训练。", "",
             "| 方法 | seed | 最终安全成功 | 动态碰撞 | 超时 | 首次≥90%步数 | 安全成功曲线面积/200000 | 静态终局 |", 
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        label = "无引导" if row["method"] == "unguided" else "A*路线输入"
        lines.append(f"| {label} | {row['seed']} | {row['final_safe_success_rate']:.0%} | {row['final_dynamic_collision_rate']:.0%} | {row['final_timeout_rate']:.0%} | {row['first_observed_ge90_step']} | {row['safe_success_aulc_0to200k']:.4f} | {row['static_termination_reason']}，{row['static_steps']}步 |")
    means, paired = [], []
    indexed = {(row["method"], row["seed"]): row for row in summaries}
    for method in entry.METHODS:
        values = [row for row in summaries if row["method"] == method]
        average_keys = ("final_safe_success_rate", "final_dynamic_collision_rate", "final_timeout_rate",
                        "safe_success_aulc_0to200k", "last_five_checks_mean_safe_success_rate")
        means.append({"method": method, "training_seed_count": len(values),
                      **{key: float(np.mean([row[key] for row in values])) for key in average_keys}})
    for seed in seeds:
        plain, guided = indexed["unguided", seed], indexed["path_guided", seed]
        paired.append({"seed": seed,
                       "guided_minus_unguided_final_safe_success": guided["final_safe_success_rate"] - plain["final_safe_success_rate"],
                       "guided_minus_unguided_aulc": guided["safe_success_aulc_0to200k"] - plain["safe_success_aulc_0to200k"],
                       "unguided_first_ge90_step": plain["first_observed_ge90_step"],
                       "guided_first_ge90_step": guided["first_observed_ge90_step"]})
    write_records_csv(means, output / "method_means.csv")
    write_records_csv(paired, output / "paired_seed_differences.csv")
    lines += ["", "按训练seed计算的描述性均值：", "",
              "| 方法 | 最终安全成功 | 动态碰撞 | 超时 | 全程安全成功曲线面积/200000 | 最后5次验证平均安全成功 |",
              "|---|---:|---:|---:|---:|---:|"]
    for value in means:
        lines.append(f"| {value['method']} | {value['final_safe_success_rate']:.1%} | {value['final_dynamic_collision_rate']:.1%} | {value['final_timeout_rate']:.1%} | {value['safe_success_aulc_0to200k']:.4f} | {value['last_five_checks_mean_safe_success_rate']:.1%} |")
    lines += ["", "逐seed比较首次达到90%和全程学习表现："]
    for value in paired:
        lines.append(f"- seed{value['seed']}：无引导首次达到90%为{value['unguided_first_ge90_step']}步，引导为{value['guided_first_ge90_step']}步；引导减去无引导的全程曲线面积差{value['guided_minus_unguided_aulc']:+.4f}，最终安全成功率差{value['guided_minus_unguided_final_safe_success']:+.1%}。")
    final_wins = sum(value["guided_minus_unguided_final_safe_success"] > 1e-9 for value in paired)
    curve_wins = sum(value["guided_minus_unguided_aulc"] > 1e-9 for value in paired)
    lines += ["", f"引导组最终成功率在{final_wins}/{len(seeds)}个seed更高，全程曲线面积在{curve_wins}/{len(seeds)}个seed更高。收益方向随seed改变；均值优势不等于稳定或统计显著的优势。",
              "最终成功率和全程学习速度必须分别比较，不能根据最终98%推断前期是否学得更快。",
              "", "最终模型的静态诊断："]
    for row in summaries:
        lines.append(f"- {row['method']} seed{row['seed']}：{row['static_termination_reason']}，{row['static_steps']}步，安全成功{row['static_original_task_safe_success']:.0%}。")
    lines += ["", "首次达到阈值仅为稀疏检查点结果，不能称为稳定收敛。均值按训练seed计算，重复使用同一批50场景不增加独立训练seed数量。",
              "这次比较同时加入路线通道和前方引导点，不能单独归因于其中某一项。训练起点、探索和回放与旧foundation复用实验不同，不能把两批差异单独归因于引导形式。",
              "没有保存中间权重，不能把曲线中的最好检查点当作可加载的最好模型，也不能在分析时选最好检查点替代预先固定的终局比较。", "", "最终失败轨迹："]
    for value in failures:
        end = value["terminal_transition"]
        lines.append(f"- {value['method']} seed{value['seed']} / {value['scenario_id']}: {value['termination_reason']}；等待{value['wait_steps']}步，移动{value['movement_steps']}步，重复移动{value['repeat_move_count']}次，末段周期{value['tail_period']}；终局位置{value['final_position']}，静态剩余距离{value['final_static_remaining_steps']}。碰撞类型{end['collision_type']}，碰撞目标位置{end['collision_position']}；最后动作{end['action']}，动作前位置{end['before']['position']}，动作后位置{end['position']}；动态障碍动作前{end['before']['dynamic_positions']}，动作后{end['dynamic_positions']}。")
    lines += ["", "证据边界：静态剩余距离忽略动态障碍，只用于轨迹诊断。等待和反复移动分类可重叠。所有终局失败完整轨迹、曲线、静态诊断、核对记录均保存于本目录。",
              "这批五seed实验已足够用于描述当前设置的收益是否一致，不为了得到优势继续挑seed。若做机制消融，分别比较仅路线、仅引导点、两者同时输入，并预先固定预算和评价标准。未自动启动这些实验。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
