"""Create complete Office v6 result figures from finished training runs.

This script never trains or changes a checkpoint.  It requires all three
preregistered strategies and all configured training seeds to be complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv


DEFAULT_CONFIG = (
    "configs/dynamic_spatial_generalization_office_behavior_v6_terminal_v1.yaml"
)
STRATEGIES = ("uniform", "prefill", "persistent_demo")
STRATEGY_LABELS = {
    "uniform": "Uniform",
    "prefill": "Prefill",
    "persistent_demo": "Persistent 25%",
}
STRATEGY_COLORS = {
    "uniform": "#546e7a",
    "prefill": "#1976d2",
    "persistent_demo": "#d84315",
}
BEHAVIOR_LABELS = {
    "normal": "Normal control",
    "wait": "Wait",
    "avoidance": "Local avoidance",
    "reroute": "Global reroute",
}
BEHAVIOR_COLORS = {
    "normal": "#546e7a",
    "wait": "#7b1fa2",
    "avoidance": "#00897b",
    "reroute": "#ef6c00",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return math.nan if value in (None, "") else float(value)


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return fmean(finite) if finite else math.nan


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan
    mean = fmean(finite)
    if len(finite) < 2:
        return mean, 0.0
    return mean, 1.96 * stdev(finite) / math.sqrt(len(finite))


def _uncertainty_label(seeds) -> str:
    if len(seeds) == 1:
        return "single-seed development result; no across-seed CI"
    return f"mean ± 95% CI across {len(seeds)} seeds"


def _load_problem(config: Mapping[str, Any]):
    source = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(source["file"])))
    map_id = str(config["map"]["scene"])
    matches = [problem for problem in problems if problem.map_id == map_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one map named {map_id!r}.")
    return matches[0]


def _load_complete_runs(config, manifest_path: Path):
    output_root = _resolve(str(config["experiment"]["output_root"]))
    prefix = str(config["experiment"]["run_name_prefix"])
    seeds = tuple(int(seed) for seed in config["training"]["seeds"])
    expected_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    missing = []
    for strategy in STRATEGIES:
        for seed in seeds:
            run_dir = output_root / f"{prefix}_seed_{seed}_{strategy}"
            required = (
                "run_metadata.json",
                "training.csv",
                "validation_summary.csv",
                "test_evaluation.csv",
                "test_trajectories.json",
                "model_selected.pth",
            )
            absent = [name for name in required if not (run_dir / name).exists()]
            if absent:
                missing.append(f"{run_dir}: {', '.join(absent)}")
                continue
            metadata = load_json(run_dir / "run_metadata.json")
            if bool(metadata.get("smoke", False)):
                missing.append(f"{run_dir}: smoke run is not a formal result")
                continue
            actual_digest = str(metadata["scenario_manifest"]["sha256"])
            if actual_digest != expected_digest:
                raise ValueError(
                    f"{run_dir} used manifest {actual_digest[:12]}, expected "
                    f"{expected_digest[:12]}."
                )
            runs[(strategy, seed)] = {
                "dir": run_dir,
                "metadata": metadata,
                "training": _read_csv(run_dir / "training.csv"),
                "validation": _read_csv(run_dir / "validation_summary.csv"),
                "test": _read_csv(run_dir / "test_evaluation.csv"),
                "trajectories": load_json(run_dir / "test_trajectories.json"),
            }
    if missing:
        joined = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            f"All {len(STRATEGIES) * len(seeds)} configured runs must finish "
            "before aggregate plotting. Missing:\n"
            + joined
        )
    return runs, seeds, expected_digest


def _attach_behavior_and_validate(runs, manifest):
    scenarios = {
        int(row["scenario_id"]): row for row in manifest["scenarios"]["test"]
    }
    expected_ids = set(scenarios)
    for (strategy, seed), run in runs.items():
        test_rows = run["test"]
        actual_ids = {int(float(row["scenario_id"])) for row in test_rows}
        if actual_ids != expected_ids or len(test_rows) != len(expected_ids):
            raise ValueError(
                f"{strategy} seed {seed} does not contain exactly the frozen "
                f"{len(expected_ids)} test scenarios."
            )
        trajectory_rows = run["trajectories"]
        trajectory_ids = {int(row["scenario_id"]) for row in trajectory_rows}
        if trajectory_ids != expected_ids or len(trajectory_rows) != len(expected_ids):
            raise ValueError(
                f"{strategy} seed {seed} is missing retained test trajectories."
            )
        for row in test_rows:
            scenario_id = int(float(row["scenario_id"]))
            row["strategy"] = strategy
            row["training_seed"] = str(seed)
            row["required_behavior"] = str(
                scenarios[scenario_id]["required_behavior"]
            )
            row["difficulty_stratum"] = str(
                scenarios[scenario_id].get("difficulty_stratum", "unassigned")
            )
            row["pair_id"] = str(scenarios[scenario_id].get("pair_id") or "")
            row["pair_role"] = str(
                scenarios[scenario_id].get("pair_role") or "unpaired"
            )
            row["matched_behavior"] = str(
                scenarios[scenario_id].get("matched_behavior") or ""
            )
            row["matched_conflict_difficulty"] = str(
                scenarios[scenario_id].get("matched_conflict_difficulty") or ""
            )
        run["trajectory_by_scenario"] = {
            int(row["scenario_id"]): row for row in trajectory_rows
        }
    return scenarios


def _style_axis(axis) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_validation_curves(runs, seeds, output: Path) -> None:
    panels = (
        ("safe_success_rate", "Collision-free success", (0.0, 1.0)),
        ("success_rate", "Goal success", (0.0, 1.0)),
        ("dynamic_collision_rate", "Dynamic collision episode rate", (0.0, 1.0)),
        ("mean_dynamic_collision_count", "Mean dynamic collisions", (0.0, None)),
        ("mean_wait_steps", "Mean wait steps", (0.0, None)),
        ("mean_steps", "Mean episode steps", (0.0, None)),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), dpi=180)
    for axis, (metric, title, limits) in zip(axes.flat, panels):
        for strategy in STRATEGIES:
            per_seed = []
            common_steps = None
            by_seed = {}
            for seed in seeds:
                series = {
                    int(float(row["environment_steps"])): _number(row, metric)
                    for row in runs[(strategy, seed)]["validation"]
                }
                by_seed[seed] = series
                common_steps = (
                    set(series)
                    if common_steps is None
                    else common_steps.intersection(series)
                )
            steps = sorted(common_steps or ())
            if not steps:
                raise ValueError(f"No shared validation steps for {strategy}.")
            for seed in seeds:
                per_seed.append([by_seed[seed][step] for step in steps])
            matrix = np.asarray(per_seed, dtype=float)
            mean = np.nanmean(matrix, axis=0)
            ci = (
                1.96 * np.nanstd(matrix, axis=0, ddof=1) / math.sqrt(len(seeds))
                if len(seeds) > 1
                else np.zeros_like(mean)
            )
            color = STRATEGY_COLORS[strategy]
            axis.plot(
                steps,
                mean,
                color=color,
                linewidth=2.2,
                label=STRATEGY_LABELS[strategy],
            )
            axis.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.16)
        axis.set_title(title)
        axis.set_xlabel("Environment interaction steps")
        axis.set_ylabel("Rate" if limits[1] == 1.0 else "Count")
        axis.set_ylim(bottom=limits[0], top=limits[1])
        _style_axis(axis)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"Office validation learning curves | {_uncertainty_label(seeds)}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _seed_metric_rows(
    runs,
    seeds,
    behaviors=None,
    *,
    group_field: str = "required_behavior",
):
    metrics = (
        "safe_success",
        "success",
        "dynamic_collision",
        "dynamic_collision_count",
        "wait_steps",
        "steps",
        "path_efficiency",
    )
    result = []
    for strategy in STRATEGIES:
        for seed in seeds:
            rows = runs[(strategy, seed)]["test"]
            groups = {"all": rows} if behaviors is None else {
                behavior: [
                    row for row in rows if row[group_field] == behavior
                ]
                for behavior in behaviors
            }
            for group_name, group in groups.items():
                record = {
                    "strategy": strategy,
                    "training_seed": seed,
                    group_field: group_name,
                    "scenario_count": len(group),
                }
                for metric in metrics:
                    record[metric] = _finite_mean(
                        [_number(row, metric) for row in group]
                    )
                result.append(record)
    return result


def _plot_grouped_bars(
    seed_rows,
    output: Path,
    *,
    groups: Sequence[str],
    group_labels: Mapping[str, str],
    title: str,
    group_field: str = "required_behavior",
) -> None:
    panels = (
        ("safe_success", "Collision-free success", (0.0, 1.0)),
        ("success", "Goal success", (0.0, 1.0)),
        ("dynamic_collision", "Dynamic collision episode rate", (0.0, 1.0)),
        ("dynamic_collision_count", "Mean dynamic collisions", (0.0, None)),
        ("wait_steps", "Mean wait steps", (0.0, None)),
        ("path_efficiency", "Path efficiency (successful only)", (0.0, 1.0)),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180)
    x = np.arange(len(groups), dtype=float)
    width = 0.24
    for axis, (metric, panel_title, limits) in zip(axes.flat, panels):
        for strategy_index, strategy in enumerate(STRATEGIES):
            means = []
            cis = []
            for group in groups:
                values = [
                    float(row[metric])
                    for row in seed_rows
                    if row["strategy"] == strategy
                    and row[group_field] == group
                ]
                mean, ci = _mean_ci95(values)
                means.append(mean)
                cis.append(ci)
            positions = x + (strategy_index - 1) * width
            axis.bar(
                positions,
                means,
                width=width,
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                alpha=0.9,
            )
            axis.errorbar(
                positions,
                means,
                yerr=cis,
                fmt="none",
                ecolor="#263238",
                elinewidth=1.0,
                capsize=3,
            )
        axis.set_title(panel_title)
        axis.set_xticks(x, [group_labels[group] for group in groups])
        axis.set_ylim(bottom=limits[0], top=limits[1])
        _style_axis(axis)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _scenario_summary(runs, seeds, scenarios):
    rows = []
    for scenario_id in sorted(scenarios):
        behavior = str(scenarios[scenario_id]["required_behavior"])
        for strategy in STRATEGIES:
            source = []
            for seed in seeds:
                matches = [
                    row
                    for row in runs[(strategy, seed)]["test"]
                    if int(float(row["scenario_id"])) == scenario_id
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Missing scenario {scenario_id} for {strategy} seed {seed}."
                    )
                source.append(matches[0])
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "required_behavior": behavior,
                    "pair_id": str(scenarios[scenario_id].get("pair_id") or ""),
                    "pair_role": str(
                        scenarios[scenario_id].get("pair_role") or "unpaired"
                    ),
                    "matched_behavior": str(
                        scenarios[scenario_id].get("matched_behavior") or ""
                    ),
                    "difficulty_stratum": str(
                        scenarios[scenario_id].get(
                            "difficulty_stratum", "unassigned"
                        )
                    ),
                    "strategy": strategy,
                    "safe_success_rate": _finite_mean(
                        [_number(row, "safe_success") for row in source]
                    ),
                    "success_rate": _finite_mean(
                        [_number(row, "success") for row in source]
                    ),
                    "dynamic_collision_rate": _finite_mean(
                        [_number(row, "dynamic_collision") for row in source]
                    ),
                    "mean_dynamic_collision_count": _finite_mean(
                        [_number(row, "dynamic_collision_count") for row in source]
                    ),
                    "mean_wait_steps": _finite_mean(
                        [_number(row, "wait_steps") for row in source]
                    ),
                }
            )
    return rows


def _plot_scenario_heatmaps(rows, scenarios, seeds, output: Path) -> None:
    scenario_ids = sorted(scenarios)
    lookup = {
        (int(row["scenario_id"]), str(row["strategy"])): row for row in rows
    }
    panels = (
        ("safe_success_rate", "Collision-free success rate", 0.0, 1.0, "YlGn"),
        ("dynamic_collision_rate", "Dynamic collision rate", 0.0, 1.0, "YlOrRd"),
        (
            "mean_dynamic_collision_count",
            "Mean dynamic collision count",
            0.0,
            None,
            "YlOrRd",
        ),
        ("mean_wait_steps", "Mean wait steps", 0.0, None, "PuBu"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(18, 10.5), dpi=180)
    for axis, (metric, title, vmin, vmax, cmap) in zip(axes, panels):
        matrix = np.asarray(
            [
                [float(lookup[(scenario_id, strategy)][metric]) for scenario_id in scenario_ids]
                for strategy in STRATEGIES
            ],
            dtype=float,
        )
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_yticks(range(len(STRATEGIES)), [STRATEGY_LABELS[s] for s in STRATEGIES])
        axis.set_xticks(range(len(scenario_ids)))
        axis.set_xticklabels([str(value) for value in scenario_ids], rotation=90, fontsize=6)
        first_source = scenarios[scenario_ids[0]]
        previous = str(
            first_source.get("matched_behavior")
            or first_source["required_behavior"]
        )
        for index, scenario_id in enumerate(scenario_ids[1:], start=1):
            source = scenarios[scenario_id]
            current = str(
                source.get("matched_behavior") or source["required_behavior"]
            )
            if current != previous:
                axis.axvline(index - 0.5, color="black", linewidth=1.3)
                previous = current
        fig.colorbar(image, ax=axis, fraction=0.012, pad=0.01)
    fig.suptitle(
        f"Office: all {len(scenario_ids)} frozen test scenarios | "
        f"values averaged across {len(seeds)} seeds",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _paired_effect_rows(runs, seeds, scenarios):
    paired_sources = [source for source in scenarios.values() if source.get("pair_id")]
    if not paired_sources:
        return [], []
    pairs = defaultdict(dict)
    for source in paired_sources:
        pairs[str(source["pair_id"])][str(source["pair_role"])] = source
    if any(set(pair) != {"conflict", "matched_control"} for pair in pairs.values()):
        raise ValueError("Every paired test case needs one conflict and one control.")
    metrics = (
        "safe_success",
        "success",
        "dynamic_collision",
        "dynamic_collision_count",
        "wait_steps",
        "steps",
    )
    pair_rows = []
    for strategy in STRATEGIES:
        for seed in seeds:
            result_lookup = {
                int(float(row["scenario_id"])): row
                for row in runs[(strategy, seed)]["test"]
            }
            for pair_id, pair in sorted(pairs.items()):
                conflict = pair["conflict"]
                control = pair["matched_control"]
                conflict_result = result_lookup[int(conflict["scenario_id"])]
                control_result = result_lookup[int(control["scenario_id"])]
                row = {
                    "strategy": strategy,
                    "training_seed": seed,
                    "pair_id": pair_id,
                    "matched_behavior": str(conflict["matched_behavior"]),
                    "difficulty_stratum": str(conflict["difficulty_stratum"]),
                    "conflict_scenario_id": int(conflict["scenario_id"]),
                    "control_scenario_id": int(control["scenario_id"]),
                }
                for metric in metrics:
                    conflict_value = _number(conflict_result, metric)
                    control_value = _number(control_result, metric)
                    row[f"conflict_{metric}"] = conflict_value
                    row[f"control_{metric}"] = control_value
                    row[f"delta_{metric}"] = conflict_value - control_value
                pair_rows.append(row)
    seed_rows = []
    groups = ("all", "wait", "avoidance", "reroute")
    for strategy in STRATEGIES:
        for seed in seeds:
            source = [
                row
                for row in pair_rows
                if row["strategy"] == strategy and row["training_seed"] == seed
            ]
            for group in groups:
                selected = (
                    source
                    if group == "all"
                    else [row for row in source if row["matched_behavior"] == group]
                )
                if not selected:
                    continue
                record = {
                    "strategy": strategy,
                    "training_seed": seed,
                    "matched_behavior": group,
                    "pair_count": len(selected),
                }
                for metric in metrics:
                    record[f"delta_{metric}"] = _finite_mean(
                        [row[f"delta_{metric}"] for row in selected]
                    )
                seed_rows.append(record)
    return pair_rows, seed_rows


def _plot_paired_effects(seed_rows, seeds, output: Path) -> None:
    groups = ("all", "wait", "avoidance", "reroute")
    labels = {
        "all": "All pairs",
        "wait": "Wait",
        "avoidance": "Avoidance",
        "reroute": "Reroute",
    }
    panels = (
        ("delta_safe_success", "Safe-success: conflict minus control"),
        ("delta_success", "Goal success: conflict minus control"),
        ("delta_dynamic_collision", "Collision episode: conflict minus control"),
        (
            "delta_dynamic_collision_count",
            "Collision count: conflict minus control",
        ),
        ("delta_wait_steps", "Wait steps: conflict minus control"),
        ("delta_steps", "Episode steps: conflict minus control"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), dpi=180)
    x = np.arange(len(groups), dtype=float)
    width = 0.24
    for axis, (metric, title) in zip(axes.flat, panels):
        for strategy_index, strategy in enumerate(STRATEGIES):
            means = []
            cis = []
            for group in groups:
                values = [
                    float(row[metric])
                    for row in seed_rows
                    if row["strategy"] == strategy
                    and row["matched_behavior"] == group
                ]
                mean, ci = _mean_ci95(values)
                means.append(mean)
                cis.append(ci)
            positions = x + (strategy_index - 1) * width
            axis.bar(
                positions,
                means,
                width=width,
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                alpha=0.9,
            )
            axis.errorbar(
                positions,
                means,
                yerr=cis,
                fmt="none",
                ecolor="#263238",
                elinewidth=1.0,
                capsize=3,
            )
        axis.axhline(0.0, color="#263238", linewidth=0.9)
        axis.set_xticks(x, [labels[group] for group in groups], rotation=12)
        axis.set_title(title)
        _style_axis(axis)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Office matched-phase test effects | {_uncertainty_label(seeds)}\n"
        "Each value is the within-pair conflict result minus its control",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _draw_path_axis(axis, problem, scenario, route_lookup, trajectory, strategy):
    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1
    axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)
    for obstacle in scenario["obstacles"]:
        route = route_lookup[str(obstacle["route_id"])]["route"]
        is_primary = obstacle["route_id"] == scenario.get("primary_route_id")
        primary_color = (
            "#1565c0"
            if scenario.get("pair_role") == "matched_control"
            else "#c62828"
        )
        axis.plot(
            [cell[1] for cell in route],
            [cell[0] for cell in route],
            color=primary_color if is_primary else "#90a4ae",
            linewidth=2.2 if is_primary else 1.0,
            linestyle="-" if is_primary else ":",
            zorder=2,
        )
        initial = route[int(obstacle["start_index"])]
        axis.scatter(
            initial[1],
            initial[0],
            color=primary_color if is_primary else "#424242",
            s=18 if is_primary else 9,
            zorder=4,
        )
    oracle = scenario["oracle_path"]
    axis.plot(
        [cell[1] for cell in oracle],
        [cell[0] for cell in oracle],
        color=BEHAVIOR_COLORS[str(scenario["required_behavior"])],
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
        zorder=2,
    )
    path = trajectory["path"]
    axis.plot(
        [cell[1] for cell in path],
        [cell[0] for cell in path],
        color=STRATEGY_COLORS[strategy],
        linewidth=1.8,
        zorder=3,
    )
    collisions = trajectory["collision_positions"]
    if collisions:
        axis.scatter(
            [cell[1] for cell in collisions],
            [cell[0] for cell in collisions],
            color="#e53935",
            marker="x",
            linewidths=1.2,
            s=20,
            zorder=6,
        )
    waits = trajectory["wait_events"]
    if waits:
        axis.scatter(
            [event["position"][1] for event in waits],
            [event["position"][0] for event in waits],
            color="#6a1b9a",
            marker="D",
            edgecolors="white",
            linewidths=0.4,
            s=18,
            zorder=7,
        )
    axis.scatter(problem.start[1], problem.start[0], color="#2e7d32", s=14, zorder=7)
    axis.scatter(
        problem.goal[1],
        problem.goal[0],
        color="#fbc02d",
        edgecolors="black",
        linewidths=0.4,
        marker="*",
        s=28,
        zorder=7,
    )
    success = bool(float(trajectory["success"]))
    axis.set_title(
        f"{STRATEGY_LABELS[strategy]} | {'OK' if success else 'FAIL'} | "
        f"coll={trajectory['dynamic_collision_count']} | "
        f"wait={trajectory['wait_steps']} | steps={trajectory['steps']}",
        fontsize=7.2,
        color="#1b5e20" if success and not collisions else "#b71c1c",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _render_path_galleries(
    runs,
    seeds,
    scenarios,
    manifest,
    problem,
    output_dir: Path,
):
    route_lookup = {
        str(record["route_id"]): record
        for record in manifest["route_pools"]["test"]["corridor"]
    }
    scenario_ids = sorted(scenarios)
    index_rows = []
    page_size = 10
    total_pages = math.ceil(len(scenario_ids) / page_size)
    for seed in seeds:
        for page_index, start in enumerate(range(0, len(scenario_ids), page_size), start=1):
            page_ids = scenario_ids[start : start + page_size]
            fig, axes = plt.subplots(
                len(page_ids),
                len(STRATEGIES),
                figsize=(11.5, 3.25 * len(page_ids)),
                dpi=150,
                squeeze=False,
            )
            for row_index, scenario_id in enumerate(page_ids):
                scenario = scenarios[scenario_id]
                for column_index, strategy in enumerate(STRATEGIES):
                    trajectory = runs[(strategy, seed)]["trajectory_by_scenario"][scenario_id]
                    axis = axes[row_index, column_index]
                    _draw_path_axis(
                        axis,
                        problem,
                        scenario,
                        route_lookup,
                        trajectory,
                        strategy,
                    )
                    if column_index == 0:
                        pair_text = (
                            f"{scenario['pair_id']} | "
                            f"{scenario['pair_role']}\n"
                            if scenario.get("pair_id")
                            else ""
                        )
                        axis.set_ylabel(
                            f"{pair_text}Scenario {scenario_id}\n"
                            f"{BEHAVIOR_LABELS[str(scenario['required_behavior'])]} | "
                            f"{str(scenario.get('difficulty_stratum', '')).title()}",
                            fontsize=8,
                            color=BEHAVIOR_COLORS[str(scenario["required_behavior"])],
                        )
            page_dir = output_dir / "paths" / f"seed_{seed}"
            page_dir.mkdir(parents=True, exist_ok=True)
            target = page_dir / f"test_paths_page_{page_index:02d}.png"
            legend = (
                Line2D([0], [0], color="#90a4ae", linestyle=":", label="obstacle route"),
                Line2D([0], [0], color="#7b1fa2", linestyle="--", label="oracle safe path"),
                Line2D([0], [0], color="#546e7a", label="learned path"),
                Line2D([0], [0], color="#e53935", marker="x", linestyle="", label="collision"),
                Line2D([0], [0], color="#6a1b9a", marker="D", linestyle="", label="wait"),
            )
            fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=8)
            fig.suptitle(
                f"Office behavior test paths | training seed {seed} | "
                f"page {page_index}/{total_pages}",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0.025, 1, 0.98))
            fig.savefig(target, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            for scenario_id in page_ids:
                index_rows.append(
                    {
                        "training_seed": seed,
                        "scenario_id": scenario_id,
                        "required_behavior": scenarios[scenario_id]["required_behavior"],
                        "page": page_index,
                        "figure": str(target),
                    }
                )
    return index_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-paths", action="store_true")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    manifest_path = _resolve(config["spatial_generalization"]["manifest"])
    manifest = load_json(manifest_path)
    problem = _load_problem(config)
    runs, seeds, digest = _load_complete_runs(config, manifest_path)
    scenarios = _attach_behavior_and_validate(runs, manifest)
    behaviors = tuple(
        behavior
        for behavior in BEHAVIOR_LABELS
        if any(
            str(scenario["required_behavior"]) == behavior
            for scenario in scenarios.values()
        )
    )
    difficulty_order = ("control", "easy", "medium", "hard")
    difficulties = tuple(
        difficulty
        for difficulty in difficulty_order
        if any(
            str(scenario.get("difficulty_stratum", "unassigned")) == difficulty
            for scenario in scenarios.values()
        )
    )
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else _resolve(config["experiment"]["output_root"]) / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = _seed_metric_rows(runs, seeds)
    behavior_rows = _seed_metric_rows(
        runs,
        seeds,
        behaviors=behaviors,
    )
    difficulty_rows = (
        _seed_metric_rows(
            runs,
            seeds,
            behaviors=difficulties,
            group_field="difficulty_stratum",
        )
        if difficulties
        else []
    )
    scenario_rows = _scenario_summary(runs, seeds, scenarios)
    pair_rows, pair_seed_rows = _paired_effect_rows(runs, seeds, scenarios)
    write_records_csv(overall_rows, output_dir / "test_overall_per_seed.csv")
    write_records_csv(behavior_rows, output_dir / "test_behavior_per_seed.csv")
    if difficulty_rows:
        write_records_csv(
            difficulty_rows,
            output_dir / "test_difficulty_per_seed.csv",
        )
    write_records_csv(scenario_rows, output_dir / "test_scenario_strategy_summary.csv")
    if pair_rows:
        write_records_csv(pair_rows, output_dir / "test_paired_effect_per_pair.csv")
        write_records_csv(
            pair_seed_rows,
            output_dir / "test_paired_effect_per_seed.csv",
        )

    _plot_validation_curves(runs, seeds, output_dir / "validation_learning_curves.png")
    _plot_grouped_bars(
        overall_rows,
        output_dir / "test_overall_metrics.png",
        groups=("all",),
        group_labels={"all": f"All {len(scenarios)} test scenarios"},
        title=(
            f"Office test split | {_uncertainty_label(seeds)}"
        ),
    )
    _plot_grouped_bars(
        behavior_rows,
        output_dir / "test_behavior_metrics.png",
        groups=behaviors,
        group_labels=BEHAVIOR_LABELS,
        title=f"Office test by required behavior | {_uncertainty_label(seeds)}",
    )
    if difficulties:
        _plot_grouped_bars(
            difficulty_rows,
            output_dir / "test_difficulty_metrics.png",
            groups=difficulties,
            group_labels={name: name.title() for name in difficulties},
            title=f"Office test by scenario difficulty | {_uncertainty_label(seeds)}",
            group_field="difficulty_stratum",
        )
    _plot_scenario_heatmaps(
        scenario_rows,
        scenarios,
        seeds,
        output_dir / f"test_{len(scenarios)}_scenario_heatmaps.png",
    )
    if pair_seed_rows:
        _plot_paired_effects(
            pair_seed_rows,
            seeds,
            output_dir / "test_paired_phase_effects.png",
        )
    path_index = []
    if not args.skip_paths:
        path_index = _render_path_galleries(
            runs,
            seeds,
            scenarios,
            manifest,
            problem,
            output_dir,
        )
        write_records_csv(path_index, output_dir / "path_gallery_index.csv")

    write_json(
        {
            "config": str(_resolve(args.config)),
            "manifest": str(manifest_path),
            "manifest_sha256": digest,
            "strategies": list(STRATEGIES),
            "training_seeds": list(seeds),
            "test_scenarios": len(scenarios),
            "matched_test_pairs": len(
                {
                    str(source.get("pair_id"))
                    for source in scenarios.values()
                    if source.get("pair_id")
                }
            ),
            "path_gallery_pages": len({row["figure"] for row in path_index}),
            "uncertainty": (
                "not_estimated_single_training_seed" if len(seeds) == 1 else
                f"normal_approximation_95pct_ci_across_{len(seeds)}_training_seeds"
            ),
        },
        output_dir / "analysis_metadata.json",
    )
    print(
        f"Office figures complete: strategies={len(STRATEGIES)} "
        f"seeds={len(seeds)} scenarios={len(scenarios)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
