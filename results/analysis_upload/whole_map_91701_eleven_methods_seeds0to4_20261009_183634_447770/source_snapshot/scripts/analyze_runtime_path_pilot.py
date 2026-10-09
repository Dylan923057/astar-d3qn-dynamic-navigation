"""Read completed pilot runs, inspect recorded failures, and evaluate static navigation only."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_runtime_path_guidance as entry
from astar_d3qn.core.grid import planner_neighbors
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.io import write_json, write_records_csv


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def static_distances(problem):
    distances = {problem.goal: 0}
    pending = deque([problem.goal])
    while pending:
        cell = pending.popleft()
        for neighbor in planner_neighbors(cell, problem.size):
            if neighbor not in problem.obstacles and neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                pending.append(neighbor)
    return distances


def trace_behavior(trace, distances):
    steps = trace["steps"]
    positions = [tuple(steps[0]["before"]["position"])] + [tuple(step["position"]) for step in steps]
    waits = [int(step["action"]) == 4 for step in steps]
    max_wait = current_wait = 0
    for waiting in waits:
        current_wait = current_wait + 1 if waiting else 0
        max_wait = max(max_wait, current_wait)
    seen = {positions[0]}
    moves = repeat_moves = 0
    for previous, position in zip(positions, positions[1:]):
        if position != previous:
            moves += 1
            repeat_moves += int(position in seen)
            seen.add(position)
    tail = positions[-60:]
    period = None
    if len(tail) == 60 and len(set(tail)) > 1:
        for candidate in range(2, 13):
            if all(tail[index] == tail[index % candidate] for index in range(len(tail))):
                period = candidate
                break
    encounter_steps = sum(bool(step["dynamic_encounter"]) for step in steps)
    goal_distances = [distances[position] for position in positions]
    wait_fraction = sum(waits) / len(steps)
    repeated_move_fraction = repeat_moves / max(1, moves)
    return {
        "scenario_id": trace["scenario_id"], "termination_reason": trace["termination_reason"],
        "steps": len(steps), "wait_steps": sum(waits), "max_wait_streak": max_wait,
        "trailing_wait_steps": current_wait, "wait_fraction": wait_fraction,
        "movement_steps": moves, "repeat_move_count": repeat_moves,
        "repeat_move_fraction": repeated_move_fraction, "unique_visited_cells": len(seen),
        "tail_period": period, "final_position": positions[-1],
        "closest_static_remaining_steps": min(goal_distances),
        "final_static_remaining_steps": goal_distances[-1], "encounter_steps": encounter_steps,
        "waiting_dominated": wait_fraction >= 0.5,
        "repeated_movement": moves >= 20 and repeated_move_fraction >= 0.5,
        "tail_60_steps": positions[-60:],
    }


def audit_run(directory):
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    if not completion["complete"] or result["environment_steps"] != 50000 or completion["environment_steps"] != 50000:
        raise ValueError(f"Incomplete pilot: {directory}")
    if result["gradient_updates"] != 49501 or result["demo_transition_count"] != 0 or result["online_replay_capacity"] != 10000:
        raise ValueError(f"Training budget/replay mismatch: {directory}")
    if not (directory / "model_final.pth").is_file():
        raise FileNotFoundError(directory / "model_final.pth")
    curves = load_csv(directory / "validation_curve.csv")
    details = load_csv(directory / "validation_details.csv")
    groups = defaultdict(list)
    for row in details:
        groups[int(row["environment_steps_total"])].append(row)
    if len(curves) != 6 or int(curves[0]["environment_steps_total"]) != 0 or int(curves[-1]["environment_steps_total"]) != 50000:
        raise ValueError("Expected step zero, four intermediate checks, and final 50000-step validation.")
    scenes = None
    for row in curves:
        step = int(row["environment_steps_total"])
        values = groups[step]
        if len(values) != 50 or len({value["scenario_id"] for value in values}) != 50:
            raise ValueError("Incomplete or duplicate validation scenes.")
        this_scenes = [(value["scenario_id"], value["dynamic_route_ids"], value["dynamic_obstacle_count"]) for value in values]
        if scenes is None:
            scenes = this_scenes
        elif this_scenes != scenes:
            raise ValueError("Validation scene sequence changed across checkpoints.")
        expected = {
            "safe_success_rate": sum(float(value["safe_success"]) for value in values) / 50,
            "dynamic_collision_rate": sum(float(value["dynamic_collision"]) for value in values) / 50,
            "timeout_rate": sum(value["termination_reason"] == "timeout" for value in values) / 50,
        }
        for key, value in expected.items():
            if not np.isclose(float(row[key]), value):
                raise ValueError(f"Aggregate/detail mismatch: {directory}/{step}/{key}")
    training = load_csv(directory / "training.csv")
    if int(training[-1]["environment_steps_total"]) != 50000 or int(training[-1]["gradient_updates_total"]) != 49501:
        raise ValueError("Final training log does not match the result.")
    return result, curves, groups, scenes, training


def static_evaluation(config, problem, method, seed, weights):
    agent = entry.make_agent(config, seed, "cpu")
    agent.load_weights(weights)
    environments = []
    def factory(p, **kwargs):
        env = PathGuidanceEnvironment(
            DynamicGridNavigationEnv(p, dynamic_obstacles=(), scenario_id="static_original_task", **kwargs),
            enabled=method == "path_guided", lookahead_steps=config["guidance"]["lookahead_steps"], record_trace=True,
        )
        environments.append(env)
        return env
    environment = config["environment"]
    with entry.preserved_evaluation(agent):
        summary, rows, _ = evaluate_agent(
            agent, [problem], max_steps=environment["max_steps"],
            reward_config=entry._reward_config(config), terminate_on_collision=environment["terminate_on_collision"],
            window_size=environment["window_size"], environment_factory=factory,
            mask_static_invalid_actions=environment["mask_static_invalid_actions"],
        )
    trace = {"scenario_id": "static_original_task", "termination_reason": rows[0]["termination_reason"],
             "safe_success": bool(rows[0]["safe_success"]), "steps": environments[0].trace}
    return summary, rows[0], trace


def main():
    torch.set_num_threads(1)
    output = ROOT / "results/whole_map_91701_runtime_path_pilot_v1" / f"analysis_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "outputs/whole_map_91701_runtime_path_pilot_v1"
    summaries, all_curves, behaviors, static_rows, examples = [], [], [], [], []
    runs = {}
    common_scenes = None
    config = None
    for method in entry.METHODS:
        for seed in (0, 1):
            directory = source / method / f"seed_{seed}"
            result, curves, details, scenes, training = audit_run(directory)
            if config is None:
                config = result["config"]
            elif result["config"] != config:
                raise ValueError("Paired methods used different registered configurations.")
            if common_scenes is None:
                common_scenes = scenes
            elif common_scenes != scenes:
                raise ValueError("Methods/seeds received different fixed validation sequences.")
            runs[method, seed] = result
            problem, _, _ = entry.validate_config(config)
            distances = static_distances(problem)
            best = max(curves, key=lambda row: float(row["safe_success_rate"]))
            x = np.asarray([int(row["environment_steps_total"]) for row in curves])
            y = np.asarray([float(row["safe_success_rate"]) for row in curves])
            aulc = sum((x[index + 1] - x[index]) * (y[index + 1] + y[index]) / 2 for index in range(len(x) - 1)) / 50000
            final = result["final_validation"]
            summary = {"method": method, "seed": seed,
                       "final_safe_success_rate": final["safe_success_rate"],
                       "final_dynamic_collision_rate": final["dynamic_collision_rate"],
                       "final_timeout_rate": final["timeout_rate"], "safe_success_aulc_0to50k": aulc,
                       "best_observed_safe_success_rate": float(best["safe_success_rate"]),
                       "best_observed_step": int(best["environment_steps_total"]),
                       "final_mean_wait_steps": final["mean_wait_steps"],
                       "environment_steps": result["environment_steps"], "gradient_updates": result["gradient_updates"]}
            last_training = [row for row in training if int(row["environment_steps_total"]) > 40000]
            summary["last_10k_training_episode_success_rate"] = sum(float(row["safe_success"]) for row in last_training) / len(last_training)
            for row in curves:
                step = int(row["environment_steps_total"])
                file = directory / f"validation_failures_{step:06d}.json"
                failures = json.loads(file.read_text(encoding="utf-8"))
                expected_ids = {detail["scenario_id"] for detail in details[step] if not float(detail["safe_success"])}
                if {trace["scenario_id"] for trace in failures} != expected_ids or len(failures) != len(expected_ids):
                    raise ValueError("Recorded failure trajectories do not match validation outcomes.")
                checkpoint_behaviors = []
                indexed_details = {value["scenario_id"]: value for value in details[step]}
                for trace in failures:
                    behavior = trace_behavior(trace, distances)
                    detail = indexed_details[trace["scenario_id"]]
                    if behavior["steps"] != int(detail["steps"]) or behavior["wait_steps"] != int(detail["wait_steps"]) or behavior["termination_reason"] != detail["termination_reason"]:
                        raise ValueError("Trace/detail count mismatch.")
                    behavior.update(method=method, seed=seed, environment_steps_total=step)
                    behaviors.append({key: value for key, value in behavior.items() if key != "tail_60_steps"})
                    checkpoint_behaviors.append(behavior)
                all_curves.append({"method": method, "seed": seed, **row,
                                   "failure_waiting_dominated_count": sum(value["waiting_dominated"] for value in checkpoint_behaviors),
                                   "failure_repeated_movement_count": sum(value["repeated_movement"] for value in checkpoint_behaviors),
                                   "failure_periodic_tail_count": sum(value["tail_period"] is not None for value in checkpoint_behaviors)})
                if step == 50000:
                    timeout = [value for value in checkpoint_behaviors if value["termination_reason"] == "timeout"]
                    summary.update(
                        timeout_waiting_dominated_count=sum(value["waiting_dominated"] for value in timeout),
                        timeout_repeated_movement_count=sum(value["repeated_movement"] for value in timeout),
                        timeout_periodic_tail_count=sum(value["tail_period"] is not None for value in timeout),
                        mean_closest_static_remaining_steps=float(np.mean([value["closest_static_remaining_steps"] for value in checkpoint_behaviors])),
                    )
                    candidates = [max(timeout, key=lambda value: value["wait_steps"]),
                                  max(timeout, key=lambda value: value["repeat_move_count"])] if timeout else []
                    candidates += [value for value in checkpoint_behaviors if value["termination_reason"] == "collision"][:1]
                    selected = set()
                    for value in candidates:
                        if value["scenario_id"] not in selected:
                            examples.append(value)
                            selected.add(value["scenario_id"])
            static_summary, static_detail, static_trace = static_evaluation(config, problem, method, seed, directory / "model_final.pth")
            static_behavior = trace_behavior(static_trace, distances)
            static_rows.append({"method": method, "seed": seed, **static_detail,
                                **{key: value for key, value in static_behavior.items() if key not in {"tail_60_steps", "scenario_id"}}})
            write_json(static_trace, output / f"static_{method}_seed{seed}.json")
            summary["static_original_task_safe_success"] = static_summary["safe_success_rate"]
            summary["static_termination_reason"] = static_detail["termination_reason"]
            summary["static_final_position"] = static_behavior["final_position"]
            summary["static_tail_period"] = static_behavior["tail_period"]
            summary["static_wait_steps"] = static_behavior["wait_steps"]
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    for seed in (0, 1):
        if runs["unguided", seed]["initial_state_sha256"] != runs["path_guided", seed]["initial_state_sha256"]:
            raise ValueError("Initial paired networks/optimizer/RNG differ.")
    write_records_csv(summaries, output / "summary.csv")
    write_records_csv(all_curves, output / "validation_curves.csv")
    write_records_csv(behaviors, output / "failure_behavior.csv")
    write_records_csv(static_rows, output / "static_diagnostics.csv")
    write_json(examples, output / "failure_examples.json")
    write_json({"all_four_runs_complete": True, "same_initial_states_per_seed": True,
                "same_registered_config": True, "same_fixed_validation_sequence": True,
                "aggregate_detail_and_trace_counts_match": True,
                "training_started_by_analysis": False, "test_data_used": False,
                "failure_pattern_definitions": {
                    "waiting_dominated": "STAY actions at least half of episode steps",
                    "repeated_movement": "at least 20 movements; at least half arrive at a previously visited cell",
                    "periodic_tail": "last 60 positions repeat exactly with a period of 2..12 and include multiple cells",
                    "categories_overlap": True,
                    "closest_static_remaining_steps": "BFS distance to final goal, ignoring dynamic obstacles; a diagnostic, not a safe dynamic path length",
                }, "aulc_definition": "Trapezoidal integral at actual sparse evaluation steps, normalized by 50000; not continuously observed performance."}, output / "verification.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
    for method in entry.METHODS:
        for seed in (0, 1):
            values = [row for row in all_curves if row["method"] == method and row["seed"] == seed]
            for axis, key in zip(axes, ("safe_success_rate", "dynamic_collision_rate", "timeout_rate")):
                axis.plot([int(row["environment_steps_total"]) / 1000 for row in values],
                          [float(row[key]) for row in values], marker="o", label=f"{method}, seed {seed}")
    for axis, title in zip(axes, ("Safe success", "Dynamic collision", "Timeout")):
        axis.set_title(title)
        axis.set_xlabel("Environment steps (thousands)")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation episode fraction")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "validation_curves.png", dpi=180)
    figure.savefig(output / "validation_curves.pdf")
    plt.close(figure)
    print(f"Analysis output: {output}", flush=True)


if __name__ == "__main__":
    main()
