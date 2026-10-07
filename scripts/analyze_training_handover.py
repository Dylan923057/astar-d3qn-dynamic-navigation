"""Audit completed training-advice experiments; network-only validation analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

import run_training_handover as entry
import analyze_runtime_path_formal as formal
import analyze_runtime_path_pilot as helpers
from astar_d3qn.utils.io import write_json, write_records_csv


def analyze(config, inputs, seeds, methods):
    source = entry._resolve(config["experiment"]["output_root"])
    # Read and validate completion before creating a report directory.
    runs = {}
    common_scenes = None
    for seed in seeds:
        paired_hashes = set()
        for method in methods:
            directory = source / method / f"seed_{seed}"
            result, curves, details, scenes = formal.audit_run(directory)
            if result["config"] != config or result["method"] != method or result["seed"] != seed or result["evaluation_advice"]:
                raise ValueError(f"Protocol/method/evaluation mismatch: {directory}")
            if common_scenes is None:
                common_scenes = scenes
            elif common_scenes != scenes:
                raise ValueError("Different fixed validation scenarios across methods/seeds.")
            paired_hashes.add(result["initial_state_sha256"])
            counts = result["advice_budget"]
            if len(counts) != 20 or sum(row["steps"] for row in counts) != 200000:
                raise ValueError("Incomplete intervention diagnostics.")
            if method in ("astar_decay", "astar_risk_decay", "astar_budget_matched") and any(row["applied"] != 0 for row in counts[10:]):
                raise ValueError("Advice remained active after withdrawal.")
            if method == "astar_budget_matched":
                expected = entry.load_matching_quota(config, seed)
                if [row["applied"] for row in counts] != expected or result["matching_advice_quota"] != expected:
                    raise ValueError("Unmatched intervention budget.")
            runs[method, seed] = (result, curves, details, directory)
        if len(paired_hashes) != 1:
            raise ValueError(f"Different paired initialization for seed {seed}.")

    output = entry.runtime.unique_check_dir(config, "analysis")
    summaries, all_curves, behaviors, statics = [], [], [], []
    distances = helpers.static_distances(inputs[0])
    for (method, seed), (result, curves, details, directory) in runs.items():
        training = helpers.load_csv(directory / "training.csv")
        successes = sum(float(row["safe_success"]) for row in training)
        budgets = result["advice_budget"]
        summaries.append({"method": method, "seed": seed,
                          "final_safe_success_rate": float(curves[-1]["safe_success_rate"]),
                          "final_dynamic_collision_rate": float(curves[-1]["dynamic_collision_rate"]),
                          "final_timeout_rate": float(curves[-1]["timeout_rate"]),
                          **formal.learning_metrics(curves),
                          "training_safe_success_episodes": int(successes),
                          "advice_requested": sum(row["requested"] for row in budgets),
                          "advice_applied": sum(row["applied"] for row in budgets),
                          "risk_veto": sum(row["risk_veto"] for row in budgets)})
        all_curves.extend({"method": method, "seed": seed, **row} for row in curves)
        for step, rows in details.items():
            filename = directory / f"validation_failures_{step:06d}.json"
            traces = json.loads(filename.read_text(encoding="utf-8"))
            failures = {row["scenario_id"]: row for row in rows if not bool(float(row["safe_success"]))}
            if len(traces) != len(failures) or {trace["scenario_id"] for trace in traces} != set(failures):
                raise ValueError(f"Missing or duplicated failure trajectories: {filename}")
            for trace in traces:
                detail = failures[trace["scenario_id"]]
                behavior = helpers.trace_behavior(trace, distances)
                if behavior["steps"] != int(detail["steps"]) or behavior["wait_steps"] != int(detail["wait_steps"]) or behavior["termination_reason"] != detail["termination_reason"]:
                    raise ValueError(f"Failure trajectory differs from validation detail: {filename}")
                behaviors.append({"method": method, "seed": seed, "environment_steps_total": step, **behavior})
            if step == 200000:
                write_json(traces, output / f"final_failures_{method}_seed{seed}.json")
        # These models share the existing unguided architecture and observation.
        _, static_row, static_trace = helpers.static_evaluation(config, inputs[0], "unguided", seed,
                                                                directory / "model_final.pth")
        statics.append({"method": method, "seed": seed, **static_row})
        write_json(static_trace, output / f"static_{method}_seed{seed}.json")

    means = []
    for method in methods:
        rows = [row for row in summaries if row["method"] == method]
        keys = ("final_safe_success_rate", "final_dynamic_collision_rate", "final_timeout_rate",
                "safe_success_aulc_0to200k", "last_five_checks_mean_safe_success_rate", "advice_applied")
        means.append({"method": method, "seed_count": len(rows),
                      **{key: float(np.mean([row[key] for row in rows])) for key in keys},
                      **{f"{key}_sample_std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else None for key in keys}})
    paired = []
    lookup = {(row["method"], row["seed"]): row for row in summaries}
    for reference in ("unguided", "astar_decay", "astar_budget_matched"):
        if "astar_risk_decay" not in methods or reference not in methods:
            continue
        for seed in seeds:
            proposed, control = lookup["astar_risk_decay", seed], lookup[reference, seed]
            paired.append({"seed": seed, "comparison": f"astar_risk_decay_minus_{reference}",
                           **{key: proposed[key] - control[key] for key in
                              ("final_safe_success_rate", "safe_success_aulc_0to200k",
                               "final_dynamic_collision_rate", "final_timeout_rate", "advice_applied")}})
    write_records_csv(summaries, output / "summary.csv")
    write_records_csv(means, output / "means.csv")
    write_records_csv(paired, output / "paired_differences.csv")
    write_records_csv(all_curves, output / "curves.csv")
    write_records_csv(behaviors, output / "failure_behavior.csv")
    write_records_csv(statics, output / "static_diagnostics.csv")
    write_json({"complete_runs_audited": len(runs), "seeds": seeds, "methods": methods,
                "all_checkpoints_audited": True, "paired_initialization_equal": True,
                "same_fixed_50_validation_scenarios": True, "evaluation_advice": False,
                "formal_training_started_by_analysis": False, "test_data_used": False}, output / "verification.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric, title in zip(axes, ("safe_success_rate", "dynamic_collision_rate", "timeout_rate"),
                                 ("Network-only safe success", "Dynamic collision", "Timeout")):
        for method in methods:
            series = [runs[method, seed][1] for seed in seeds]
            # Checkpoints occur at episode boundaries and therefore have slightly
            # different steps. Interpolate on a common axis before averaging.
            x = np.arange(0, 200001, 10000)
            values = np.array([np.interp(x, [int(r["environment_steps_total"]) for r in rows],
                                           [float(r[metric]) for r in rows]) for rows in series])
            axis.plot(x / 1000, values.mean(axis=0), label=method)
            if len(seeds) > 1:
                spread = values.std(axis=0, ddof=1)
                axis.fill_between(x / 1000, np.maximum(0, values.mean(axis=0) - spread),
                                  np.minimum(1, values.mean(axis=0) + spread), alpha=0.12)
        axis.axvline(100, color="gray", linestyle="--", linewidth=1)
        axis.set(xlabel="Environment steps (thousands)", title=title, ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=180)
    figure.savefig(output / "learning_curves.pdf")
    plt.close(figure)
    report = ["# A*训练引导退出实验", "", f"已核查 {len(seeds)} 个seed、{len(methods)} 种方法的完整200000步记录。",
              "全部验证由网络单独选动作，epsilon=0，不调用A*；使用原固定50个验证场景，不使用test。", "",
              "|方法|最终安全成功率|动态碰撞率|超时率|学习曲线面积|A*介入次数均值|",
              "|---|---:|---:|---:|---:|---:|"]
    for row in means:
        report.append(f"|{row['method']}|{row['final_safe_success_rate']:.1%}|{row['final_dynamic_collision_rate']:.1%}|{row['final_timeout_rate']:.1%}|{row['safe_success_aulc_0to200k']:.4f}|{row['advice_applied']:.0f}|")
    report.extend(["", "曲线阴影为seed间标准差，不是置信区间；50个验证场景不能替代独立训练seed。",
                   "summary.csv包含首达阈值、后续检查始终达标起点和最后5次验证均值。只看最后一次不足以判断收敛速度或稳定性。",
                   "failure_behavior.csv核对全部检查点的失败轨迹：碰撞、等待占主导、反复移动及剩余静态路程；类型允许重叠。",
                   "astar_budget_matched逐个seed、逐10000步区间匹配risk组的实际A*介入次数，只随机改变介入时刻。",
                   "若risk组只胜过time-decay，却不胜过匹配次数组，不能把提升归因于风险判断。",
                   "本批只检验固定地图与起终点；跨地图结论及论文方法贡献需要后续独立验证。", ""])
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(entry.CONFIG))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=entry.METHODS, default=list(entry.METHODS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(entry.runtime.REGISTERED_SEEDS) or len(set(args.methods)) != len(args.methods):
        parser.error("Use unique registered seeds and methods.")
    torch.set_num_threads(1)
    config = entry.load_config(entry._resolve(args.config))
    inputs = entry.validate_config(config)
    print(f"Output: {analyze(config, inputs, args.seeds, args.methods)}", flush=True)


if __name__ == "__main__":
    main()
