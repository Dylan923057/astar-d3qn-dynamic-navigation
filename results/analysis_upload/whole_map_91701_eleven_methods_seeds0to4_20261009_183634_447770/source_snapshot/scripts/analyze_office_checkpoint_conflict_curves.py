"""Measure A*-conflict handling throughout completed Office training runs.

This is evaluation-only.  It reconstructs the 20 static demonstration paths,
extracts the first blocked demonstration action in each in-distribution test
scenario, and probes every saved 25k-step checkpoint on the exact same states.
Oracle wait-decision states are evaluated separately.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from astar_d3qn.core.astar import randomized_tie_astar_path
from astar_d3qn.core.grid import Action, action_between
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.spatial_scenarios import scenarios_from_spatial_manifest
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv
from run_office_controlled_behavior_study import MAIN_CONFIG, METHODS
from run_office_controlled_blockage_diagnostic import _agent


BEHAVIORS = ("wait", "avoidance", "reroute")
COLORS = {
    "uniform": "#546e7a",
    "prefill": "#1976d2",
    "persistent": "#d84315",
    "scheduled_decay": "#f9a825",
    "ca_current": "#8e24aa",
    "ca_predictive": "#43a047",
    "local_conflict": "#00838f",
    "ca_adaptive_decay": "#c62828",
    "local_counterexample": "#6a1b9a",
    "predictive_margin": "#ad1457",
}
CHECKPOINT_PATTERN = re.compile(r"step_(\d+)_episode_(\d+)\.pth$")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _problem_context():
    config = load_config(_resolve(MAIN_CONFIG))
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    matches = [problem for problem in problems if problem.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one Office problem.")
    problem = matches[0]
    manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))
    scenarios = {
        scenario.seed: scenario
        for scenario in scenarios_from_spatial_manifest(problem, manifest, "train")
    }
    records = {
        int(row["scenario_id"]): row for row in manifest["scenarios"]["train"]
    }
    test_ids = tuple(
        int(value) for value in config["spatial_generalization"]["scenario_ids"]["test"]
    )
    return config, problem, scenarios, records, test_ids


def _static_demo_paths(problem, config) -> tuple[tuple[tuple[int, int], ...], ...]:
    episodes = int(config["demonstrations"]["episodes"])
    seed = int(config["demonstrations"]["seed"])
    paths = []
    for episode in range(episodes):
        path = randomized_tie_astar_path(
            problem.start,
            problem.goal,
            problem.obstacles,
            problem.size,
            random.Random(seed + episode),
        )
        if path is None:
            raise RuntimeError("Could not reconstruct a registered A* demonstration.")
        paths.append(tuple(path))
    return tuple(paths)


def _valid_and_safe_masks(env: DynamicGridNavigationEnv):
    valid = np.asarray(env.action_mask(mask_collisions=True), dtype=bool)
    safe = np.asarray(
        [
            bool(valid[action])
            and not env.dynamic_action_collision_risk(action, predict_next=True)
            for action in range(env.action_dim)
        ],
        dtype=bool,
    )
    return valid, safe


def _build_probe_bank():
    config, problem, scenarios, records, test_ids = _problem_context()
    paths = _static_demo_paths(problem, config)
    conflict = []
    for scenario_id in test_ids:
        scenario = scenarios[scenario_id]
        behavior = str(records[scenario_id]["required_behavior"])
        for demonstration_index, path in enumerate(paths):
            env = DynamicGridNavigationEnv(
                problem,
                scenario.obstacles,
                scenario_id=scenario_id,
                max_steps=int(config["environment"]["max_steps"]),
                window_size=int(config["environment"]["window_size"]),
            )
            state = env.reset()
            for step in range(1, len(path)):
                action = int(action_between(path[step - 1], path[step]))
                if env.dynamic_action_collision_risk(action, predict_next=True):
                    valid, safe = _valid_and_safe_masks(env)
                    if not safe.any():
                        raise RuntimeError("A conflict probe has no safe execution action.")
                    conflict.append(
                        {
                            "scenario_id": scenario_id,
                            "required_behavior": behavior,
                            "demonstration_index": demonstration_index,
                            "conflict_step": step,
                            "spatial": np.asarray(state.spatial, dtype=np.float32),
                            "scalars": np.asarray(state.scalars, dtype=np.float32),
                            "valid_mask": valid,
                            "safe_mask": safe,
                            "demo_action": action,
                        }
                    )
                    break
                result = env.step(action)
                if result.info["collision"]:
                    raise RuntimeError("Demonstration collided before its first probe.")
                state = result.observation

    wait = []
    for scenario_id in test_ids:
        record = records[scenario_id]
        if record["required_behavior"] != "wait":
            continue
        scenario = scenarios[scenario_id]
        oracle = tuple(tuple(cell) for cell in record["oracle_path"])
        wait_index = next(
            index
            for index in range(1, len(oracle))
            if oracle[index] == oracle[index - 1]
        )
        env = DynamicGridNavigationEnv(
            problem,
            scenario.obstacles,
            scenario_id=scenario_id,
            max_steps=int(config["environment"]["max_steps"]),
            window_size=int(config["environment"]["window_size"]),
        )
        state = env.reset()
        for step in range(1, wait_index):
            result = env.step(int(action_between(oracle[step - 1], oracle[step])))
            if result.info["collision"]:
                raise RuntimeError("Oracle collided before its wait-decision probe.")
            state = result.observation
        valid, safe = _valid_and_safe_masks(env)
        if not safe[int(Action.STAY)]:
            raise RuntimeError("Registered oracle wait action is unexpectedly unsafe.")
        wait.append(
            {
                "scenario_id": scenario_id,
                "wait_step": wait_index,
                "spatial": np.asarray(state.spatial, dtype=np.float32),
                "scalars": np.asarray(state.scalars, dtype=np.float32),
                "valid_mask": valid,
                "safe_mask": safe,
            }
        )
    if not conflict or not wait:
        raise RuntimeError("Probe construction produced an empty bank.")
    return conflict, wait


def _tensor_bank(records, device):
    return (
        torch.as_tensor(
            np.stack([record["spatial"] for record in records]),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.stack([record["scalars"] for record in records]),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.stack([record["valid_mask"] for record in records]),
            dtype=torch.bool,
            device=device,
        ),
        torch.as_tensor(
            np.stack([record["safe_mask"] for record in records]),
            dtype=torch.bool,
            device=device,
        ),
    )


def _q_values(agent, tensors, batch_size: int) -> torch.Tensor:
    spatial, scalars, _, _ = tensors
    batches = []
    with torch.no_grad():
        for start in range(0, len(spatial), batch_size):
            batches.append(
                agent.policy_network(
                    spatial[start : start + batch_size],
                    scalars[start : start + batch_size],
                )
            )
    return torch.cat(batches, dim=0)


def _probe_checkpoint(agent, conflict, wait, conflict_tensors, wait_tensors, batch_size):
    conflict_q = _q_values(agent, conflict_tensors, batch_size)
    valid = conflict_tensors[2]
    safe = conflict_tensors[3]
    masked_q = conflict_q.masked_fill(~valid, -torch.inf)
    selected = masked_q.argmax(dim=1)
    safe_selected = safe.gather(1, selected.unsqueeze(1)).squeeze(1)
    demo_actions = torch.as_tensor(
        [row["demo_action"] for row in conflict],
        dtype=torch.long,
        device=agent.device,
    )
    demo_selected = selected == demo_actions
    demo_q = conflict_q.gather(1, demo_actions.unsqueeze(1)).squeeze(1)
    best_safe_q = conflict_q.masked_fill(~safe, -torch.inf).max(dim=1).values
    result = {
        "conflict_probe_count": len(conflict),
        "conflict_safe_action_rate": float(safe_selected.float().mean().item()),
        "blocked_demo_action_rate": float(demo_selected.float().mean().item()),
        "safe_q_margin": float((best_safe_q - demo_q).mean().item()),
    }
    behavior_indices = {
        behavior: torch.as_tensor(
            [row["required_behavior"] == behavior for row in conflict],
            dtype=torch.bool,
            device=agent.device,
        )
        for behavior in BEHAVIORS
    }
    for behavior, indices in behavior_indices.items():
        result[f"conflict_safe_action_rate_{behavior}"] = float(
            safe_selected[indices].float().mean().item()
        )

    wait_q = _q_values(agent, wait_tensors, batch_size)
    wait_valid = wait_tensors[2]
    wait_safe = wait_tensors[3]
    wait_selected = wait_q.masked_fill(~wait_valid, -torch.inf).argmax(dim=1)
    wait_safe_selected = wait_safe.gather(
        1, wait_selected.unsqueeze(1)
    ).squeeze(1)
    non_wait_safe = wait_safe.clone()
    non_wait_safe[:, int(Action.STAY)] = False
    best_non_wait_q = wait_q.masked_fill(~non_wait_safe, -torch.inf).max(dim=1).values
    result.update(
        {
            "wait_probe_count": len(wait),
            "oracle_wait_action_rate": float(
                (wait_selected == int(Action.STAY)).float().mean().item()
            ),
            "oracle_wait_safe_action_rate": float(
                wait_safe_selected.float().mean().item()
            ),
            "wait_q_margin": float(
                (wait_q[:, int(Action.STAY)] - best_non_wait_q).mean().item()
            ),
        }
    )
    return result


def _run_dir(method, seed: int) -> Path:
    config = load_config(_resolve(method.config_path))
    return _resolve(config["experiment"]["output_root"]) / (
        f"{config['experiment']['run_name_prefix']}_seed_{seed}_{method.strategy}"
    )


def _checkpoints(run_dir: Path):
    result = []
    for path in (run_dir / "checkpoints").glob("step_*.pth"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            result.append((int(match.group(1)), int(match.group(2)), path))
    result.sort()
    if len(result) != 16:
        raise RuntimeError(f"Expected 16 checkpoints in {run_dir}, found {len(result)}.")
    return result


def _nearest_training_row(rows, environment_steps: int):
    return min(
        rows,
        key=lambda row: abs(int(float(row["environment_steps_total"])) - environment_steps),
    )


def _recent_weighted_metric(rows, previous_step: int, current_step: int, key: str):
    selected = [
        row
        for row in rows
        if previous_step < int(float(row["environment_steps_total"])) <= current_step
    ]
    if not selected:
        return math.nan
    weights = [float(row["steps"]) for row in selected]
    return sum(
        float(row.get(key, 0.0)) * weight
        for row, weight in zip(selected, weights)
    ) / sum(weights)


def _evaluate_all(conflict, wait, device: str, batch_size: int):
    base_config = load_config(_resolve(MAIN_CONFIG))
    seeds = tuple(int(value) for value in base_config["training"]["seeds"])
    rows = []
    for method in METHODS:
        config = load_config(_resolve(method.config_path))
        for seed in seeds:
            run_dir = _run_dir(method, seed)
            training = _read_csv(run_dir / "training.csv")
            validation = _read_csv(run_dir / "validation_summary.csv")
            validation_by_step = {
                int(float(row["environment_steps"])): row for row in validation
            }
            agent = _agent(config, seed, device)
            conflict_tensors = _tensor_bank(conflict, agent.device)
            wait_tensors = _tensor_bank(wait, agent.device)
            previous_step = 0
            for checkpoint_index, (environment_steps, episode, path) in enumerate(
                _checkpoints(run_dir), start=1
            ):
                agent.load_weights(path)
                metrics = _probe_checkpoint(
                    agent,
                    conflict,
                    wait,
                    conflict_tensors,
                    wait_tensors,
                    batch_size,
                )
                validation_row = validation_by_step.get(environment_steps)
                if validation_row is None:
                    validation_row = min(
                        validation,
                        key=lambda row: abs(
                            int(float(row["environment_steps"])) - environment_steps
                        ),
                    )
                training_row = _nearest_training_row(training, environment_steps)
                rows.append(
                    {
                        "method": method.key,
                        "method_label": method.label,
                        "training_seed": seed,
                        "checkpoint_index": checkpoint_index,
                        "nominal_environment_steps": checkpoint_index * 25_000,
                        "environment_steps": environment_steps,
                        "episode": episode,
                        "curriculum_stage": training_row["curriculum_stage"],
                        "validation_safe_success": float(
                            validation_row["safe_success_rate"]
                        ),
                        "validation_success": float(validation_row["success_rate"]),
                        "validation_dynamic_collision": float(
                            validation_row["dynamic_collision_rate"]
                        ),
                        "recent_demo_fraction": _recent_weighted_metric(
                            training,
                            previous_step,
                            environment_steps,
                            "demo_fraction_mean",
                        ),
                        "recent_demo_conflict_rate": _recent_weighted_metric(
                            training,
                            previous_step,
                            environment_steps,
                            "demo_action_conflict_rate",
                        ),
                        "recent_local_suppression_rate": _recent_weighted_metric(
                            training,
                            previous_step,
                            environment_steps,
                            "local_demo_suppression_rate",
                        ),
                        "recent_local_counterexample_add_rate": _recent_weighted_metric(
                            training,
                            previous_step,
                            environment_steps,
                            "local_counterexample_add_rate",
                        ),
                        "recent_local_counterexample_sample_fraction": (
                            _recent_weighted_metric(
                                training,
                                previous_step,
                                environment_steps,
                                "local_counterexample_sample_fraction",
                            )
                        ),
                        "recent_conflict_margin_label_rate": _recent_weighted_metric(
                            training,
                            previous_step,
                            environment_steps,
                            "conflict_margin_label_rate",
                        ),
                        "recent_conflict_margin_batch_fraction": (
                            _recent_weighted_metric(
                                training,
                                previous_step,
                                environment_steps,
                                "conflict_margin_batch_fraction",
                            )
                        ),
                        **metrics,
                    }
                )
                previous_step = environment_steps
            print(f"[{method.label} seed={seed}] 16 checkpoints probed", flush=True)
    return rows, seeds


def _mean_ci95(values: Sequence[float]):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    mean = fmean(finite)
    return mean, 1.96 * stdev(finite) / math.sqrt(len(finite)) if len(finite) > 1 else 0.0


def _summarize(rows, seeds):
    metrics = (
        "validation_safe_success",
        "validation_success",
        "validation_dynamic_collision",
        "recent_demo_fraction",
        "recent_demo_conflict_rate",
        "recent_local_suppression_rate",
        "recent_local_counterexample_add_rate",
        "recent_local_counterexample_sample_fraction",
        "recent_conflict_margin_label_rate",
        "recent_conflict_margin_batch_fraction",
        "conflict_safe_action_rate",
        "blocked_demo_action_rate",
        "safe_q_margin",
        "conflict_safe_action_rate_wait",
        "conflict_safe_action_rate_avoidance",
        "conflict_safe_action_rate_reroute",
        "oracle_wait_action_rate",
        "oracle_wait_safe_action_rate",
        "wait_q_margin",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["checkpoint_index"]))].append(row)
    summary = []
    for method in METHODS:
        for checkpoint_index in range(1, 17):
            group = grouped[(method.key, checkpoint_index)]
            if len(group) != len(seeds):
                raise RuntimeError("A checkpoint summary is missing one or more seeds.")
            record = {
                "method": method.key,
                "method_label": method.label,
                "checkpoint_index": checkpoint_index,
                "nominal_environment_steps": checkpoint_index * 25_000,
                "actual_environment_steps_mean": fmean(
                    float(row["environment_steps"]) for row in group
                ),
                "seed_count": len(group),
            }
            for metric in metrics:
                mean, ci95 = _mean_ci95([float(row[metric]) for row in group])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_ci95"] = ci95
            summary.append(record)
    return summary


def _plot(summary, output: Path):
    panels = (
        ("validation_safe_success", "Validation collision-free success", (0, 1.05)),
        ("conflict_safe_action_rate", "Exact A* conflict: safe action", (0, 1.05)),
        ("blocked_demo_action_rate", "Selects blocked A* action", (0, 1.05)),
        ("oracle_wait_action_rate", "Oracle wait state: selects STAY", (0, 1.05)),
        ("safe_q_margin", "Best-safe Q minus blocked-A* Q", (None, None)),
        ("recent_demo_fraction", "Actual demonstration fraction", (0, 0.28)),
        ("recent_local_suppression_rate", "Locally suppressed demo fraction", (0, None)),
        (
            "recent_conflict_margin_batch_fraction",
            "Conflict-margin labeled share of replay batch",
            (0, 0.25),
        ),
    )
    lookup = defaultdict(list)
    for row in summary:
        lookup[row["method"]].append(row)
    figure, axes = plt.subplots(4, 2, figsize=(14, 17), dpi=180)
    for axis, (metric, title, limits) in zip(axes.flat, panels):
        for method in METHODS:
            records = sorted(lookup[method.key], key=lambda row: row["checkpoint_index"])
            x = np.asarray([row["nominal_environment_steps"] for row in records])
            mean = np.asarray([row[f"{metric}_mean"] for row in records], dtype=float)
            ci = np.asarray([row[f"{metric}_ci95"] for row in records], dtype=float)
            axis.plot(x, mean, color=COLORS[method.key], linewidth=2, label=method.label)
            axis.fill_between(x, mean - ci, mean + ci, color=COLORS[method.key], alpha=0.12)
        for boundary in (60_000, 140_000, 240_000):
            axis.axvline(boundary, color="#9e9e9e", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if limits[0] is not None:
            axis.set_ylim(*limits)
    axes.flat[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Office checkpoint conflict curves | mean and 95% CI across 5 seeds")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _auc(records, metric: str):
    ordered = sorted(records, key=lambda row: row["nominal_environment_steps"])
    x = np.asarray([float(row["nominal_environment_steps"]) for row in ordered])
    y = np.asarray([float(row[f"{metric}_mean"]) for row in ordered])
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def _write_report(summary, conflict_count: int, wait_count: int, output: Path):
    by_method = {
        method.key: sorted(
            [row for row in summary if row["method"] == method.key],
            key=lambda row: row["checkpoint_index"],
        )
        for method in METHODS
    }
    lines = [
        "# Office checkpoint conflict analysis",
        "",
        f"Evaluation-only analysis of 16 saved checkpoints per run, 5 seeds, {conflict_count} in-distribution A* conflict states, and {wait_count} oracle wait states.",
        "Vertical dotted lines in the figure mark the 60k, 140k, and 240k obstacle-curriculum boundaries.",
        "",
        "| Method | Validation-safe AUC | Conflict-safe AUC | Final conflict-safe | Final blocked A* | Final oracle STAY | Final demo fraction | Final CE sample | Final margin labels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        records = by_method[method.key]
        final = records[-1]
        lines.append(
            f"| {method.label} | {_auc(records, 'validation_safe_success'):.3f} | "
            f"{_auc(records, 'conflict_safe_action_rate'):.3f} | "
            f"{100 * float(final['conflict_safe_action_rate_mean']):.1f}% | "
            f"{100 * float(final['blocked_demo_action_rate_mean']):.1f}% | "
            f"{100 * float(final['oracle_wait_action_rate_mean']):.1f}% | "
            f"{100 * float(final['recent_demo_fraction_mean']):.2f}% | "
            f"{100 * float(final['recent_local_counterexample_sample_fraction_mean']):.2f}% | "
            f"{100 * float(final['recent_conflict_margin_batch_fraction_mean']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "AUC is normalized over the saved 25k–400k checkpoint interval. The conflict bank uses scenarios that belong to the training schedule; it measures learning dynamics rather than map or route generalization.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir", default="outputs/office_checkpoint_conflict_analysis_v1"
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conflict, wait = _build_probe_bank()
    write_json(
        {
            "conflict_probe_count": len(conflict),
            "conflict_scenario_count": len({row["scenario_id"] for row in conflict}),
            "wait_probe_count": len(wait),
            "conflict_definition": "first predicted collision on each registered static A* demonstration path",
            "evaluation_distribution": "seen training scenarios; frozen checkpoints; no updates",
        },
        output_dir / "probe_manifest.json",
    )
    rows, seeds = _evaluate_all(conflict, wait, args.device, args.batch_size)
    summary = _summarize(rows, seeds)
    write_records_csv(rows, output_dir / "checkpoint_probe_per_run.csv")
    write_records_csv(summary, output_dir / "checkpoint_probe_summary.csv")
    _plot(summary, output_dir / "checkpoint_conflict_curves.png")
    _write_report(summary, len(conflict), len(wait), output_dir / "REPORT.md")
    print(f"Wrote checkpoint conflict analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
