"""Audit every recorded repair checkpoint, compare autonomous validation, and inspect failures."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_value_repair as entry
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from astar_d3qn.utils.io import write_json, write_records_csv


def analyze(root, seeds, methods):
    summaries, curves, failures = [], [], []
    common_config = common_scenes = None
    initial_hashes = {}
    mode = None
    for seed in seeds:
        for method in methods:
            directory = root / method / f"seed_{seed}"
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
            budget = result["effective_training"]["max_environment_steps"]
            if not completion["complete"] or result["environment_steps"] != budget or completion["environment_steps"] != budget:
                raise ValueError(f"Incomplete run: {directory}")
            if result["method"] != method or result["seed"] != seed or result["training_mode"] == "smoke":
                raise ValueError("Incorrect run identity or smoke output.")
            expected_updates = budget - result["effective_training"]["learning_starts"] + 1
            if result["gradient_updates"] != expected_updates or result["demo_transition_count"] != 0 or result["online_replay_capacity"] != 10000 or result["test_data_used"] or result["evaluation_advice"]:
                raise ValueError("Budget, replay, or evaluation protocol mismatch.")
            if common_config is None:
                common_config, mode = result["config"], result["training_mode"]
            if result["config"] != common_config or result["training_mode"] != mode:
                raise ValueError("Cannot mix different configurations or formal/probe outputs.")
            if seed in initial_hashes and initial_hashes[seed] != result["initial_state_sha256"]:
                raise ValueError("Paired initial training states differ.")
            initial_hashes[seed] = result["initial_state_sha256"]
            details = defaultdict(list)
            for row in load_csv(directory / "validation_details.csv"):
                details[int(row["environment_steps_total"])].append(row)
            curve = load_csv(directory / "validation_curve.csv")
            x = [int(r["environment_steps_total"]) for r in curve]
            if x[0] != 0 or x[-1] != budget or any(a >= b for a, b in zip(x, x[1:])) or set(x) != set(details):
                raise ValueError("Missing, duplicated, or unordered checkpoints.")
            if len(curve) != {"formal": 21, "pilot": 5, "probe": 2}[mode]:
                raise ValueError("Unexpected number of validation checkpoints.")
            if mode == "formal" and (budget != 200000 or result["effective_training"]["epsilon_decay_environment_steps"] != 150000):
                raise ValueError("Formal budget or epsilon schedule changed.")
            if mode == "pilot" and (budget != 20000 or result["effective_training"]["epsilon_decay_environment_steps"] != 150000):
                raise ValueError("Pilot budget or epsilon schedule changed.")
            training_rows = load_csv(directory / "training.csv")
            if int(training_rows[-1]["environment_steps_total"]) != budget or int(training_rows[-1]["gradient_updates_total"]) != expected_updates:
                raise ValueError("Training log does not match the result.")
            problem = entry.old.validate_config(entry.load_config(entry.old.CONFIG))[0]
            distances = static_distances(problem)
            for record in curve:
                step = int(record["environment_steps_total"])
                rows = details[step]
                scenes = [(r["scenario_id"], r["dynamic_route_ids"], r["dynamic_obstacle_count"]) for r in rows]
                if len(rows) != 50 or len({r["scenario_id"] for r in rows}) != 50:
                    raise ValueError("Expected all 50 distinct fixed validation scenes.")
                if common_scenes is None:
                    common_scenes = scenes
                if scenes != common_scenes:
                    raise ValueError("Validation scene sequence differs across methods/seeds/checkpoints.")
                expected = {"safe_success_rate": sum(float(r["safe_success"]) for r in rows) / 50,
                            "dynamic_collision_rate": sum(float(r["dynamic_collision"]) for r in rows) / 50,
                            "timeout_rate": sum(r["termination_reason"] == "timeout" for r in rows) / 50}
                if any(not np.isclose(float(record[k]), v) for k, v in expected.items()):
                    raise ValueError("Validation aggregates differ from scene details.")
                traces = json.loads((directory / f"validation_failures_{step:06d}.json").read_text(encoding="utf-8"))
                failed_ids = {r["scenario_id"] for r in rows if not float(r["safe_success"])}
                if len(traces) != len(failed_ids) or {t["scenario_id"] for t in traces} != failed_ids:
                    raise ValueError("Missing or duplicated failure trajectories.")
                failures.extend({"method": method, "seed": seed, "environment_steps_total": step,
                                 **trace_behavior(t, distances)} for t in traces)
                curves.append({"method": method, "seed": seed, **record})
            final = result["final_validation"]
            if any(not np.isclose(final[k], float(curve[-1][k])) for k in ("safe_success_rate", "dynamic_collision_rate", "timeout_rate")):
                raise ValueError("Final result and validation curve disagree.")
            y = [float(r["safe_success_rate"]) for r in curve]
            aulc = sum((b - a) * (yb + ya) / 2 for a, b, ya, yb in zip(x, x[1:], y, y[1:])) / budget
            final_failures = [r for r in failures if r["method"] == method and r["seed"] == seed and r["environment_steps_total"] == budget]
            values = load_csv(directory / "value_learning.csv")
            if sum(int(r["updates"]) for r in values) != expected_updates or int(values[-1]["bin_end_step"]) != budget:
                raise ValueError("Value-learning records do not cover all gradient updates.")
            if any(float(r["teacher_margin_weight_mean"]) != 0 or int(r["teacher_label_samples"]) != 0
                   for r in values if int(r["bin_start_step"]) >= 100001):
                raise ValueError("Teacher supervision did not withdraw after 100000 steps.")
            if any(r["applied"] != 0 for r in result["advice_budget"] if r["bin_start_step"] >= 100001):
                raise ValueError("A* action advice did not withdraw after 100000 steps.")
            static = result["final_static"]
            summaries.append({"method": method, "seed": seed, "mode": mode, "environment_steps": budget,
                              "safe_success_rate": final["safe_success_rate"], "dynamic_collision_rate": final["dynamic_collision_rate"],
                              "timeout_rate": final["timeout_rate"], "safe_success_aulc": aulc,
                              "last_five_checkpoint_mean": sum(y[-5:]) / len(y[-5:]) if mode == "formal" else None,
                              "static_success": bool(static["safe_success"]), "static_steps": static["steps"],
                              "waiting_dominated_failures": sum(r["waiting_dominated"] for r in final_failures),
                              "repeated_movement_failures": sum(r["repeated_movement"] for r in final_failures),
                              "periodic_tail_failures": sum(r["tail_period"] is not None for r in final_failures),
                              "max_observed_training_q_abs": max(float(r["q_abs_max"]) for r in values),
                              "max_observed_raw_target": max(float(r["raw_target_max"]) for r in values)})
    output = entry.runtime.unique_check_dir(common_config, "analysis")
    write_records_csv(summaries, output / "summary.csv")
    write_records_csv(curves, output / "all_validation_checkpoints.csv")
    write_records_csv(failures, output / "all_failure_behaviors.csv")
    write_json({"source_root": str(root), "mode": mode, "seeds": seeds, "methods": methods,
                "all_recorded_checkpoints_verified": True, "all_50_scenes_identical": True,
                "paired_initial_state_sha256": initial_hashes, "all_failure_traces_inspected": True,
                "test_data_used": False}, output / "verification.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 4), squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        for method in methods:
            rows = [r for r in curves if r["seed"] == seed and r["method"] == method]
            ax.plot([int(r["environment_steps_total"]) for r in rows], [float(r["safe_success_rate"]) * 100 for r in rows], label=method)
        ax.set(title=f"{mode} seed {seed}: network only", xlabel="Environment steps", ylabel="Safe success (%)", ylim=(-2, 102))
        ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("validation_curves." + suffix), dpi=180)
    plt.close(fig)
    lines = [f"# A*动作学习与价值目标修正：{mode}", "", f"原始数据：`{root}`", "",
             "已核对所有记录的检查点和失败轨迹。各组使用同样的50个固定验证场景，epsilon=0，由网络独立选择动作；不调用A*，不使用test。", "",
             "| 方法 | Seed | 安全成功 | 动态碰撞 | 超时 | 静态到达 |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['seed']} | {r['safe_success_rate']:.0%} | {r['dynamic_collision_rate']:.0%} | {r['timeout_rate']:.0%} | {r['static_success']} |" for r in summaries]
    lines += ["", "等待占主导、反复移动与周期尾段标签可重叠。每条失败轨迹都记录终点的剩余静态BFS距离；静态有路不保证动态场景300步内一定可通行。", "",
              "这是一项2000步接入诊断，可以检查早期自主动作和数值行为；不能证明20万步收敛、撤出引导后的稳定性或论文方法优势。" if mode == "probe" else "结论应基于全部检查点、配对seed及失败轨迹；避免只选择最好的检查点。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/whole_map_91701_value_repair_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=entry.METHODS, default=list(entry.METHODS))
    args = parser.parse_args()
    print(analyze(args.root.resolve(), args.seeds, args.methods))


if __name__ == "__main__":
    main()
