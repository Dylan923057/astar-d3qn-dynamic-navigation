"""Validate, analyze, plot, tabulate, and archive the strict five-seed A/B study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_ab_five_seed_strict import (
    MAP_IDS,
    OUTPUT_ROOT,
    SEEDS,
    current_code_sha256,
    verify_no_test_artifacts,
)


METHODS = {
    "A_time_decay": ("schedule_decay", "decay"),
    "B_global_prediction": ("global_prediction", "global_prediction"),
}
PHASE = "10_ab_five_seed_strict_validation"
RAW_FILES = (
    "training.csv",
    "validation_curve.csv",
    "validation_details.csv",
    "validation_trajectories.json",
    "result.json",
    "fork_audit.json",
)
INDEX_FIELDS = (
    "phase", "completed_at", "map_id", "method", "seed", "status",
    "replay_schedule", "validation_conflict_auc",
    "test_conflict_safe_success", "test_conflict_dynamic_collision",
    "test_conflict_timeout", "static_safe_success", "config_sha256",
    "manifest_sha256", "code_sha256", "fork_sha256", "result_path",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    return None if raw in (None, "") else float(raw)


def mean_std(values) -> tuple[float, float]:
    data = [float(value) for value in values]
    return statistics.fmean(data), statistics.stdev(data) if len(data) > 1 else 0.0


def formatted(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None or std is None:
        return "NA"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def auc(curve: list[dict[str, str]], budget: int = 200_000) -> float:
    points = [
        (int(row["environment_steps"]), float(row["conflict_safe_success"]))
        for row in curve
    ]
    expected = list(range(0, budget + 1, 10_000))
    if [step for step, _ in points] != expected:
        raise ValueError("Validation checkpoints differ from the frozen 10k schedule.")
    return sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value)
        in zip(points, points[1:])
    ) / budget


def bootstrap_ci(values, rng: np.random.Generator, draws: int = 100_000):
    data = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(data), size=(draws, len(data)))
    means = data[indices].mean(axis=1)
    lower, upper = np.percentile(means, (2.5, 97.5))
    return float(lower), float(upper)


def exact_sign_flip_pvalue(values) -> float:
    data = np.asarray(values, dtype=np.float64)
    observed = abs(float(data.mean()))
    permuted = [
        abs(float((data * np.asarray(signs)).mean()))
        for signs in itertools.product((-1.0, 1.0), repeat=len(data))
    ]
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def save_figure(figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def verify_and_load(run_root: Path, registration: dict):
    verify_no_test_artifacts(str(run_root.relative_to(ROOT)))
    manifest_path = ROOT / "data" / "risk_handover_v1" / "manifest.json"
    config_path = ROOT / registration.get(
        "config_path", "configs/risk_handover_v1.yaml"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if registration.get("protocol") != "ab_five_seed_strict_v1":
        raise SystemExit("Unexpected or missing frozen protocol registration.")
    if registration.get("training_code_sha256") != current_code_sha256():
        raise SystemExit("Current training code differs from the registered frozen code.")
    if (
        not config_path.is_file()
        or registration.get("config_file_sha256")
        != hashlib.sha256(config_path.read_bytes()).hexdigest()
    ):
        raise SystemExit("Current config file differs from the registered frozen config.")
    if registration.get("manifest_sha256") != hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest():
        raise SystemExit("Current manifest differs from the registered manifest.")
    if tuple(registration.get("maps", ())) != MAP_IDS or tuple(
        registration.get("seeds", ())
    ) != SEEDS:
        raise SystemExit("Registration map or seed set changed.")

    loaded = {}
    provenance = set()
    for map_entry in manifest["maps"]:
        map_id = map_entry["problem"]["map_id"]
        if map_id not in MAP_IDS:
            continue
        for method, (branch, schedule) in METHODS.items():
            for seed in SEEDS:
                directory = run_root / "formal" / map_id / f"seed_{seed}" / branch
                for name in RAW_FILES:
                    if not (directory / name).is_file():
                        raise SystemExit(f"Missing raw artifact: {directory / name}")
                forbidden = list(directory.glob("test_*"))
                if forbidden:
                    raise SystemExit(f"Test artifact is forbidden: {forbidden[0]}")
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                if (
                    result.get("status") != "validation_complete"
                    or not result.get("test_deferred")
                    or "test" in result
                    or result.get("adaptation_run", {}).get("steps") != 200_000
                    or result.get("replay_schedule") != schedule
                    or result.get("code_sha256") != registration["training_code_sha256"]
                    or result.get("manifest_sha256") != registration["manifest_sha256"]
                    or result.get("map_id") != map_id
                ):
                    raise SystemExit(f"Run violates the registered protocol: {directory}")
                if method == "A_time_decay" and (
                    result.get("prediction_mode") is not None
                    or result.get("prediction_loss_weight") != 0.0
                ):
                    raise SystemExit(f"A is not unchanged time_decay: {directory}")
                if method == "B_global_prediction" and (
                    result.get("prediction_mode") != "global_prediction"
                    or result.get("prediction_loss_weight") != 0.1
                    or result.get("prediction_pos_weight") != 20.0
                    or result.get("prediction_decision_zone_weight") != 1.0
                    or result.get("prediction_gradient_strategy", "none") != "none"
                ):
                    raise SystemExit(f"B prediction settings changed: {directory}")
                curve = read_csv(directory / "validation_curve.csv")
                computed_auc = auc(curve)
                if not math.isclose(
                    computed_auc,
                    float(result["validation_conflict_auc"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise SystemExit(f"Stored and recomputed AUC differ: {directory}")
                trajectories = json.loads(
                    (directory / "validation_trajectories.json").read_text(
                        encoding="utf-8"
                    )
                )
                if trajectories.get("environment_steps") != 200_000:
                    raise SystemExit(f"Trajectories are not from the final checkpoint: {directory}")
                loaded[(map_id, method, seed)] = {
                    "directory": directory,
                    "result": result,
                    "curve": curve,
                    "trajectories": trajectories["trajectories"],
                }
                provenance.add((
                    result["code_sha256"],
                    result["config_sha256"],
                    result["manifest_sha256"],
                ))
        for seed in SEEDS:
            a = loaded[(map_id, "A_time_decay", seed)]["result"]
            b = loaded[(map_id, "B_global_prediction", seed)]["result"]
            if a["fork_sha256"] != b["fork_sha256"]:
                raise SystemExit(f"A/B foundation fork differs: {map_id} seed {seed}")
    if len(provenance) != 1:
        raise SystemExit("The 30 runs do not share one code/config/manifest provenance group.")
    return manifest, loaded, next(iter(provenance))


def per_seed_metrics(loaded):
    rows = []
    for map_id in MAP_IDS:
        for seed in SEEDS:
            method_rows = {}
            for method in METHODS:
                item = loaded[(map_id, method, seed)]
                curve = item["curve"]
                final = curve[-1]
                late = [
                    float(row["conflict_safe_success"])
                    for row in curve
                    if int(row["environment_steps"]) >= 150_000
                ]
                threshold = item["result"].get("threshold_confirmation_step")
                method_rows[method] = {
                    "auc": auc(curve),
                    "threshold_confirmation_step": threshold,
                    "threshold_right_censored": int(threshold is None),
                    "final_conflict_safe_success": number(
                        final, "conflict_safe_success"
                    ),
                    "final_dynamic_collision": number(
                        final, "conflict_dynamic_collision"
                    ),
                    "final_timeout": number(final, "conflict_timeout"),
                    "last_50k_safe_success_mean": statistics.fmean(late),
                    "last_50k_safe_success_std": (
                        statistics.pstdev(late) if len(late) > 1 else 0.0
                    ),
                    "prediction_precision": number(
                        final, "prediction_positive_precision"
                    ),
                    "prediction_recall": number(final, "prediction_positive_recall"),
                    "prediction_f1": number(final, "prediction_positive_f1"),
                    "decision_zone_f1": number(final, "decision_zone_f1"),
                }
            row = {"map_id": map_id, "seed": seed}
            for method, prefix in (
                ("A_time_decay", "A"),
                ("B_global_prediction", "B"),
            ):
                row.update({f"{prefix}_{key}": value for key, value in method_rows[method].items()})
            row["delta_auc_B_minus_A"] = method_rows["B_global_prediction"]["auc"] - method_rows["A_time_decay"]["auc"]
            rows.append(row)
    return rows


def summarize(rows):
    map_summaries = []
    main_rows = []
    prediction_rows = []
    rng = np.random.default_rng(20260928)
    for map_id in MAP_IDS:
        selected = [row for row in rows if row["map_id"] == map_id]
        deltas = [row["delta_auc_B_minus_A"] for row in selected]
        ci_low, ci_high = bootstrap_ci(deltas, rng)
        summary = {
            "map_id": map_id,
            "paired_differences": deltas,
            "B_greater_than_A_seed_count": sum(delta > 0.0 for delta in deltas),
            "mean_paired_difference": statistics.fmean(deltas),
            "paired_difference_bootstrap_ci95": [ci_low, ci_high],
            "paired_sign_flip_pvalue_two_sided": exact_sign_flip_pvalue(deltas),
        }
        if summary["mean_paired_difference"] > 0.0:
            summary["effect_interpretation"] = (
                "stable_positive"
                if summary["B_greater_than_A_seed_count"] >= 4 and ci_low > 0.0
                else "positive_direction_but_uncertain"
            )
        elif summary["mean_paired_difference"] < 0.0:
            summary["effect_interpretation"] = (
                "stable_negative"
                if summary["B_greater_than_A_seed_count"] <= 1 and ci_high < 0.0
                else "negative_direction_but_uncertain"
            )
        else:
            summary["effect_interpretation"] = "no_mean_effect"
        for method, prefix in (("A_time_decay", "A"), ("B_global_prediction", "B")):
            auc_mean, auc_std = mean_std(row[f"{prefix}_auc"] for row in selected)
            summary[f"{prefix}_auc_mean"] = auc_mean
            summary[f"{prefix}_auc_std"] = auc_std
            thresholds = [
                row[f"{prefix}_threshold_confirmation_step"]
                for row in selected
                if row[f"{prefix}_threshold_confirmation_step"] is not None
            ]
            threshold_mean, threshold_std = (
                mean_std(thresholds) if thresholds else (None, None)
            )
            table_row = {
                "map_id": map_id,
                "method": method,
                "validation_auc_mean": auc_mean,
                "validation_auc_std": auc_std,
                "validation_auc_mean_std": formatted(auc_mean, auc_std),
                "threshold_confirmation_step_mean": threshold_mean,
                "threshold_confirmation_step_std": threshold_std,
                "threshold_confirmed_seeds": len(thresholds),
            }
            for metric in (
                "final_conflict_safe_success",
                "final_dynamic_collision",
                "final_timeout",
                "last_50k_safe_success_mean",
                "last_50k_safe_success_std",
            ):
                metric_mean, metric_std = mean_std(
                    row[f"{prefix}_{metric}"] for row in selected
                )
                table_row[f"{metric}_mean"] = metric_mean
                table_row[f"{metric}_std"] = metric_std
                table_row[f"{metric}_mean_std"] = formatted(metric_mean, metric_std)
            main_rows.append(table_row)
        b_rows = selected
        prediction_row = {"map_id": map_id, "method": "B_global_prediction"}
        for metric in (
            "prediction_precision", "prediction_recall", "prediction_f1",
            "decision_zone_f1",
        ):
            metric_mean, metric_std = mean_std(row[f"B_{metric}"] for row in b_rows)
            prediction_row[f"{metric}_mean"] = metric_mean
            prediction_row[f"{metric}_std"] = metric_std
            prediction_row[f"{metric}_mean_std"] = formatted(metric_mean, metric_std)
        prediction_rows.append(prediction_row)
        map_summaries.append(summary)

    equal_a = statistics.fmean(row["A_auc_mean"] for row in map_summaries)
    equal_b = statistics.fmean(row["B_auc_mean"] for row in map_summaries)
    map_deltas = [row["paired_differences"] for row in map_summaries]
    bootstrap_equal = []
    for _ in range(100_000):
        bootstrap_equal.append(statistics.fmean(
            statistics.fmean(rng.choice(delta, size=len(delta), replace=True))
            for delta in map_deltas
        ))
    cross_ci = np.percentile(bootstrap_equal, (2.5, 97.5))
    signs = [math.copysign(1.0, row["mean_paired_difference"])
             if row["mean_paired_difference"] != 0.0 else 0.0
             for row in map_summaries]
    if all(sign > 0 for sign in signs) and float(cross_ci[0]) > 0.0:
        conclusion = "stable_cross_map_improvement"
    elif any(sign > 0 for sign in signs) and any(sign < 0 for sign in signs):
        conclusion = "scenario_dependent"
    else:
        conclusion = "insufficient_evidence"
    cross_map = {
        "aggregation": "equal weight per map; seeds are paired only within each map",
        "A_equal_weight_mean_auc": equal_a,
        "B_equal_weight_mean_auc": equal_b,
        "equal_weight_mean_paired_difference": equal_b - equal_a,
        "hierarchical_within_map_bootstrap_ci95": [
            float(cross_ci[0]), float(cross_ci[1])
        ],
        "classification": conclusion,
    }
    return map_summaries, main_rows, prediction_rows, cross_map


def plot_learning_curves(loaded, destination: Path) -> None:
    colors = {"A_time_decay": "#3b6fb6", "B_global_prediction": "#d95f02"}
    labels = {"A_time_decay": "A: time_decay", "B_global_prediction": "B: global_prediction"}
    for map_id in MAP_IDS:
        for late_only in (False, True):
            figure, axis = plt.subplots(figsize=(7.2, 4.6))
            displayed = []
            for method in METHODS:
                curves = loaded[(map_id, method, 0)]["curve"]
                steps = np.asarray([int(row["environment_steps"]) for row in curves])
                values = np.asarray([
                    [float(row["conflict_safe_success"]) for row in loaded[(map_id, method, seed)]["curve"]]
                    for seed in SEEDS
                ])
                means = values.mean(axis=0)
                stds = values.std(axis=0, ddof=1)
                mask = steps >= 150_000 if late_only else np.ones_like(steps, dtype=bool)
                displayed.extend((means[mask] - stds[mask]).tolist())
                displayed.extend((means[mask] + stds[mask]).tolist())
                axis.plot(steps[mask], means[mask], color=colors[method], linewidth=2.0, label=labels[method])
                axis.fill_between(steps[mask], means[mask] - stds[mask], means[mask] + stds[mask], color=colors[method], alpha=0.20)
            axis.set_xlabel("Training steps")
            axis.set_ylabel("Validation conflict safe-success")
            axis.set_title(f"{map_id} | {'last 50k' if late_only else 'full learning curve'}")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
            axis.set_xlim((150_000, 200_000) if late_only else (0, 200_000))
            if late_only:
                lower = max(0.0, min(displayed) - 0.03)
                upper = min(1.02, max(displayed) + 0.03)
                if upper - lower < 0.15:
                    center = (upper + lower) / 2.0
                    lower, upper = max(0.0, center - 0.075), min(1.02, center + 0.075)
                axis.set_ylim(lower, upper)
            else:
                axis.set_ylim(0.0, 1.02)
            figure.tight_layout()
            suffix = "learning_curve_last50k" if late_only else "learning_curve_full"
            save_figure(figure, destination / f"{map_id}_{suffix}")


def plot_statistics(map_summaries, main_rows, per_seed_rows, destination: Path) -> None:
    x = np.arange(len(MAP_IDS))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for offset, prefix, label, color in (
        (-width / 2, "A", "A: time_decay", "#3b6fb6"),
        (width / 2, "B", "B: global_prediction", "#d95f02"),
    ):
        means = [row[f"{prefix}_auc_mean"] for row in map_summaries]
        stds = [row[f"{prefix}_auc_std"] for row in map_summaries]
        axis.bar(x + offset, means, width, yerr=stds, capsize=4, color=color, label=label)
    axis.set_xticks(x, ("map01", "map02", "map03"))
    axis.set_ylabel("Validation conflict safe-success AUC")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Five-seed validation AUC (mean ± std)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, destination / "three_map_auc_comparison")

    for map_id in MAP_IDS:
        selected = [row for row in per_seed_rows if row["map_id"] == map_id]
        deltas = [row["delta_auc_B_minus_A"] for row in selected]
        figure, axis = plt.subplots(figsize=(6.6, 4.2))
        axis.axhline(0.0, color="#444444", linewidth=1.0)
        axis.bar(SEEDS, deltas, color=["#2ca25f" if value > 0 else "#de2d26" for value in deltas])
        axis.scatter(SEEDS, deltas, color="#111111", s=25, zorder=3)
        axis.set_xticks(SEEDS)
        axis.set_xlabel("Paired seed")
        axis.set_ylabel("Δ AUC (B - A)")
        axis.set_title(f"{map_id} | paired AUC differences")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        save_figure(figure, destination / f"{map_id}_paired_auc_difference")

    metrics = (
        ("final_conflict_safe_success_mean", "Final safe success"),
        ("final_dynamic_collision_mean", "Dynamic collision"),
        ("final_timeout_mean", "Timeout"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    for axis, (metric, title) in zip(axes, metrics):
        for offset, method, color, label in (
            (-width / 2, "A_time_decay", "#3b6fb6", "A"),
            (width / 2, "B_global_prediction", "#d95f02", "B"),
        ):
            selected = [row for row in main_rows if row["method"] == method]
            std_metric = metric.replace("_mean", "_std")
            axis.bar(
                x + offset,
                [row[metric] for row in selected],
                width,
                yerr=[row[std_metric] for row in selected],
                capsize=3,
                color=color,
                label=label,
            )
        axis.set_xticks(x, ("map01", "map02", "map03"))
        axis.set_ylim(0.0, 1.02)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False)
    figure.suptitle("Final validation outcomes (five-seed means)")
    figure.tight_layout()
    save_figure(figure, destination / "three_map_final_outcomes")


def trajectory_lookup(loaded, map_id: str, method: str, seed: int):
    return {
        row["scenario_id"]: row
        for row in loaded[(map_id, method, seed)]["trajectories"]
    }


def scenario_lookup(manifest):
    result = {}
    for entry in manifest["maps"]:
        map_id = entry["problem"]["map_id"]
        for pair in entry["scenarios"]["splits"]["validation"]:
            result[(map_id, pair["pair_id"] + "_conflict")] = {
                **pair["conflict"],
                "pair_id": pair["pair_id"],
                "scenario_id": pair["pair_id"] + "_conflict",
                "progress_band": pair["progress_band"],
            }
    return result


def draw_trajectory(axis, problem, scene, trajectory, label: str) -> None:
    grid = np.zeros((problem["size"], problem["size"]), dtype=np.float32)
    for row, column in problem["obstacles"]:
        grid[row, column] = 1.0
    axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1, interpolation="none")
    for index, obstacle in enumerate(scene["obstacles"]):
        route = np.asarray(obstacle["route"])
        axis.plot(route[:, 1], route[:, 0], "--", color="#ef8a62", alpha=0.65, linewidth=1.0, label="dynamic route" if index == 0 else None)
        initial = route[int(obstacle["start_index"])]
        axis.scatter(initial[1], initial[0], marker="s", s=28, color="#d7301f", zorder=4, label="dynamic initial" if index == 0 else None)
    path = np.asarray(trajectory["positions"])
    axis.plot(path[:, 1], path[:, 0], color="#2166ac", linewidth=2.0, label="robot path")
    axis.scatter(problem["start"][1], problem["start"][0], s=55, color="#1a9850", label="start", zorder=5)
    axis.scatter(problem["goal"][1], problem["goal"][0], s=100, marker="*", color="#ffd92f", edgecolor="black", label="goal", zorder=5)
    critical = None
    best_distance = math.inf
    for step, (position, dynamic_positions) in enumerate(zip(trajectory["positions"], trajectory["dynamic_positions"])):
        for dynamic_position in dynamic_positions:
            distance = abs(position[0] - dynamic_position[0]) + abs(position[1] - dynamic_position[1])
            if distance < best_distance:
                best_distance = distance
                critical = (step, position, dynamic_position)
    if critical is not None:
        _, robot_position, dynamic_position = critical
        axis.scatter(dynamic_position[1], dynamic_position[0], marker="D", s=55, color="#f46d43", edgecolor="black", label="critical dynamic", zorder=6)
        axis.scatter(robot_position[1], robot_position[0], marker="o", s=45, facecolor="none", edgecolor="#542788", linewidth=1.5, label="critical robot", zorder=6)
    if trajectory["dynamic_collision"] or trajectory["static_collision"]:
        axis.scatter(path[-1, 1], path[-1, 0], marker="X", s=110, color="#b2182b", label="collision", zorder=7)
    outcome = "success" if trajectory["safe_success"] else "dynamic collision" if trajectory["dynamic_collision"] else "timeout" if trajectory["timeout"] else "static collision"
    axis.set_title(f"{label} | {outcome} | steps={trajectory['steps']}")
    axis.set_xlim(-0.5, problem["size"] - 0.5)
    axis.set_ylim(problem["size"] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.grid(alpha=0.12)


def plot_pair(problem, scene, a_trajectory, b_trajectory, title: str, target: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.2))
    draw_trajectory(axes[0], problem, scene, a_trajectory, "A: time_decay")
    draw_trajectory(axes[1], problem, scene, b_trajectory, "B: global_prediction")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=5, frameon=False, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(figure, target)


def plot_trajectories(manifest, loaded, destination: Path):
    scenes = scenario_lookup(manifest)
    selections = []
    problems = {entry["problem"]["map_id"]: entry["problem"] for entry in manifest["maps"]}
    for map_id in MAP_IDS:
        candidates = [
            value for (candidate_map, _), value in scenes.items()
            if candidate_map == map_id and value.get("obstacle_count") == 5
        ]
        chosen = []
        for band in ("early", "late"):
            matches = sorted(
                (scene for scene in candidates if scene["progress_band"] == band),
                key=lambda scene: scene["scenario_id"],
            )
            if matches:
                chosen.append(matches[0])
        for scene in chosen:
            scenario_id = scene["scenario_id"]
            a = trajectory_lookup(loaded, map_id, "A_time_decay", 0)[scenario_id]
            b = trajectory_lookup(loaded, map_id, "B_global_prediction", 0)[scenario_id]
            name = f"{map_id}_seed0_{scene['progress_band']}_{scenario_id}_trajectory"
            plot_pair(problems[map_id], scene, a, b, f"{map_id} | seed 0 | {scenario_id}", destination / name)
            selections.append({"figure": name, "selection_rule": "seed0_first_5_obstacle_scene_in_progress_band", "map_id": map_id, "seed": 0, "scenario_id": scenario_id})

    map_id = "irregular_workcell_91703"
    failure_candidates = []
    fallback_candidates = []
    for seed in SEEDS:
        a_lookup = trajectory_lookup(loaded, map_id, "A_time_decay", seed)
        b_lookup = trajectory_lookup(loaded, map_id, "B_global_prediction", seed)
        for scenario_id in sorted(set(a_lookup) & set(b_lookup)):
            if not scenario_id.endswith("_conflict"):
                continue
            a, b = a_lookup[scenario_id], b_lookup[scenario_id]
            if a["safe_success"] and not b["safe_success"]:
                failure_candidates.append((seed, scenario_id, a, b))
            fallback_candidates.append((
                b["steps"] - a["steps"],
                b["wait_steps"] - a["wait_steps"],
                -seed,
                scenario_id,
                seed,
                a,
                b,
            ))
    if failure_candidates:
        seed, scenario_id, a, b = sorted(failure_candidates, key=lambda item: (item[0], item[1]))[0]
        rule = "first_lexicographic_case_where_A_succeeds_and_B_fails"
    else:
        _, _, _, scenario_id, seed, a, b = max(fallback_candidates)
        rule = "largest_B_minus_A_steps_then_waits_when_no_A_success_B_failure_exists"
    scene = scenes[(map_id, scenario_id)]
    name = f"{map_id}_representative_degradation_{scenario_id}_seed{seed}"
    plot_pair(problems[map_id], scene, a, b, f"map03 diagnostic | seed {seed} | {scenario_id}", destination / name)
    selections.append({"figure": name, "selection_rule": rule, "map_id": map_id, "seed": seed, "scenario_id": scenario_id, "A_safe_success": a["safe_success"], "B_safe_success": b["safe_success"], "A_steps": a["steps"], "B_steps": b["steps"]})
    return selections


def write_text_tables(destination: Path, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    markdown = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    (destination / f"{name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    escaped_columns = [column.replace("_", r"\_") for column in columns]
    latex = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        " & ".join(escaped_columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        latex.append(" & ".join(str(row.get(column, "")).replace("_", r"\_") for column in columns) + r" \\")
    latex.append(r"\end{tabular}")
    (destination / f"{name}.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def xlsx_cell_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    return value


def write_xlsx(path: Path, sheets: dict[str, list[dict]]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as error:
        raise SystemExit(
            "XLSX output requires openpyxl. Run: python -m pip install openpyxl"
        ) from error
    workbook = Workbook()
    workbook.remove(workbook.active)
    for sheet_name, rows in sheets.items():
        worksheet = workbook.create_sheet(sheet_name[:31])
        columns = list(rows[0]) if rows else []
        worksheet.append(columns)
        for row in rows:
            worksheet.append([
                xlsx_cell_value(row.get(column)) for column in columns
            ])
        worksheet.freeze_panes = "A2"
    workbook.save(path)


def copy_raw(loaded, destination: Path) -> None:
    for (map_id, method, seed), item in loaded.items():
        target = destination / map_id / method / f"seed_{seed}"
        target.mkdir(parents=True, exist_ok=True)
        for name in RAW_FILES:
            shutil.copy2(item["directory"] / name, target / name)


def update_global_archive(destination: Path, loaded, provenance) -> None:
    evidence = ROOT / "results" / "paper_evidence"
    index_path = evidence / "experiment_index.json"
    index_rows = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if any(row["phase"] == PHASE for row in index_rows):
        raise SystemExit(f"Global index already contains {PHASE}.")
    new_rows = []
    completed = []
    for (map_id, method, seed), item in loaded.items():
        result = item["result"]
        result_path = item["directory"] / "result.json"
        timestamp = datetime.fromtimestamp(result_path.stat().st_mtime)
        completed.append(timestamp)
        archived_result = destination / "raw_data" / map_id / method / f"seed_{seed}" / "result.json"
        new_rows.append({
            "phase": PHASE,
            "completed_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "map_id": map_id,
            "method": method,
            "seed": seed,
            "status": result["status"],
            "replay_schedule": result["replay_schedule"],
            "validation_conflict_auc": result["validation_conflict_auc"],
            "test_conflict_safe_success": "",
            "test_conflict_dynamic_collision": "",
            "test_conflict_timeout": "",
            "static_safe_success": "",
            "config_sha256": result["config_sha256"],
            "manifest_sha256": result["manifest_sha256"],
            "code_sha256": result["code_sha256"],
            "fork_sha256": result["fork_sha256"],
            "result_path": str(archived_result.relative_to(evidence)),
        })
    index_rows.extend(new_rows)
    index_rows.sort(key=lambda row: (row["completed_at"], row["phase"], row["method"], int(row["seed"])))
    write_json(index_path, index_rows)
    write_csv(evidence / "experiment_index.csv", index_rows, INDEX_FIELDS)
    provenance_path = evidence / "provenance_groups.csv"
    provenance_rows = read_csv(provenance_path)
    provenance_rows.append({
        "phase": PHASE,
        "first_completed": min(completed).strftime("%Y-%m-%d %H:%M:%S"),
        "last_completed": max(completed).strftime("%Y-%m-%d %H:%M:%S"),
        "run_count": 30,
        "code_sha256": provenance[0],
        "config_sha256": provenance[1],
        "manifest_sha256": provenance[2],
    })
    write_csv(provenance_path, provenance_rows, provenance_rows[0].keys())
    readme = evidence / "README.md"
    lines = readme.read_text(encoding="utf-8-sig").splitlines()
    lines = [f"- `experiment_index.csv`：{len(index_rows)} 个正式运行的可读索引。" if "`experiment_index.csv`" in line else line for line in lines]
    lines.extend(["", "### 10：三地图 A/B 严格 5-seed validation", "", f"- 目录：`{PHASE}/`", "- 当前同一代码版本下重新训练三个地图、A/B 各 5 个 seed，共 30 次 validation-only 运行。", "- 目录内包含原始结果、PNG/PDF 图、CSV/XLSX/Markdown/LaTeX 表格和统计报告。", "- 未生成、读取或归档 test 结果；结论不用于结果驱动调参。"])
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checklist = evidence / "UPLOAD_CHECKLIST.md"
    checklist.write_text(checklist.read_text(encoding="utf-8-sig").rstrip() + "\n- [x] 阶段 10 已归档严格 5-seed A/B 的 raw data、论文图表、统计表和 validation-only 报告。\n", encoding="utf-8")
    checksum_rows = []
    for phase_directory in sorted(path for path in evidence.iterdir() if path.is_dir()):
        for path in sorted(phase_directory.rglob("*")):
            if path.is_file():
                checksum_rows.append({"path": str(path.relative_to(evidence)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(evidence / "SHA256SUMS.csv", checksum_rows, ("path", "bytes", "sha256"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=OUTPUT_ROOT)
    parser.add_argument(
        "--archive-root",
        default=f"results/paper_evidence/{PHASE}",
    )
    args = parser.parse_args()
    run_root = ROOT / args.run_root
    registration_path = run_root / "protocol_registration.json"
    if not registration_path.exists():
        raise SystemExit(f"Missing protocol registration: {registration_path}")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    manifest, loaded, provenance = verify_and_load(run_root, registration)
    destination = ROOT / args.archive_root
    temporary = destination.with_name(destination.name + ".tmp")
    if destination.exists():
        raise SystemExit(f"Archive destination already exists: {destination}")
    if temporary.exists():
        expected_parent = destination.parent.resolve()
        if temporary.is_symlink() or temporary.resolve().parent != expected_parent:
            raise SystemExit(f"Refusing to clean unexpected temporary path: {temporary}")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        raw = temporary / "raw_data"
        figures = temporary / "figures"
        tables = temporary / "tables"
        analysis = temporary / "analysis"
        analysis.mkdir(parents=True)
        copy_raw(loaded, raw)
        shutil.copy2(registration_path, analysis / "protocol_registration.json")
        per_seed_rows = per_seed_metrics(loaded)
        map_summaries, main_rows, prediction_rows, cross_map = summarize(per_seed_rows)
        write_csv(tables / "per_seed_details.csv", per_seed_rows)
        write_csv(tables / "main_results.csv", main_rows)
        write_csv(tables / "prediction_metrics.csv", prediction_rows)
        paired_rows = [{**row, "paired_differences": json.dumps(row["paired_differences"]), "paired_difference_bootstrap_ci95": json.dumps(row["paired_difference_bootstrap_ci95"])} for row in map_summaries]
        cross_map_rows = [cross_map]
        write_csv(tables / "paired_auc_statistics.csv", paired_rows)
        write_csv(tables / "cross_map_equal_weight_summary.csv", cross_map_rows)
        for name, rows in (
            ("main_results", main_rows),
            ("prediction_metrics", prediction_rows),
            ("per_seed_details", per_seed_rows),
            ("paired_auc_statistics", paired_rows),
            ("cross_map_equal_weight_summary", cross_map_rows),
        ):
            write_text_tables(tables, name, rows)
        write_xlsx(tables / "paper_tables.xlsx", {
            "main_results": main_rows,
            "prediction_metrics": prediction_rows,
            "per_seed_details": per_seed_rows,
            "paired_auc": paired_rows,
            "cross_map": cross_map_rows,
        })
        plot_learning_curves(loaded, figures / "learning_curves")
        plot_statistics(map_summaries, main_rows, per_seed_rows, figures / "statistics")
        trajectory_selections = plot_trajectories(manifest, loaded, figures / "trajectories")
        write_json(analysis / "trajectory_selection_audit.json", trajectory_selections)
        summary = {
            "validation_only": True,
            "test_files_read": False,
            "training_runs": 30,
            "maps": list(MAP_IDS),
            "seeds": list(SEEDS),
            "map_summaries": map_summaries,
            "cross_map_equal_weight_summary": cross_map,
            "trajectory_selection": trajectory_selections,
            "provenance": {
                "code_sha256": provenance[0],
                "config_sha256": provenance[1],
                "manifest_sha256": provenance[2],
            },
        }
        write_json(analysis / "summary.json", summary)
        map_lines = []
        for row in map_summaries:
            map_lines.append(
                f"- {row['map_id']}: A={row['A_auc_mean']:.4f}±{row['A_auc_std']:.4f}, "
                f"B={row['B_auc_mean']:.4f}±{row['B_auc_std']:.4f}, "
                f"Δ={row['mean_paired_difference']:+.4f}, "
                f"95% bootstrap CI=[{row['paired_difference_bootstrap_ci95'][0]:+.4f}, {row['paired_difference_bootstrap_ci95'][1]:+.4f}], "
                f"B>A={row['B_greater_than_A_seed_count']}/5."
            )
        map01 = next(row for row in map_summaries if row["map_id"] == MAP_IDS[0])
        map02 = next(row for row in map_summaries if row["map_id"] == MAP_IDS[1])
        map03 = next(row for row in map_summaries if row["map_id"] == MAP_IDS[2])
        report = "\n".join([
            "# Strict three-map A/B five-seed validation",
            "",
            "## Protocol",
            "",
            "All 30 adaptation runs use the same frozen code/config/manifest provenance and paired per-seed foundations. Only validation artifacts were read; test artifacts were forbidden.",
            "",
            "## Per-map effects",
            "",
            *map_lines,
            "",
            "## Direct answers",
            "",
            f"1. map01 positive-gain assessment: {map01['effect_interpretation']}.",
            f"   map02 positive-gain assessment: {map02['effect_interpretation']}.",
            f"2. map03 negative-gain assessment: {map03['effect_interpretation']}.",
            f"3. Cross-map assessment: {cross_map['classification']}.",
            "",
            "## Cross-map equal-weight result",
            "",
            f"- A equal-weight mean AUC: {cross_map['A_equal_weight_mean_auc']:.4f}",
            f"- B equal-weight mean AUC: {cross_map['B_equal_weight_mean_auc']:.4f}",
            f"- Equal-weight paired effect: {cross_map['equal_weight_mean_paired_difference']:+.4f}",
            f"- Hierarchical 95% bootstrap CI: [{cross_map['hierarchical_within_map_bootstrap_ci95'][0]:+.4f}, {cross_map['hierarchical_within_map_bootstrap_ci95'][1]:+.4f}]",
            f"- Classification: {cross_map['classification']}",
            "",
            "Map-level effects are weighted equally. Seeds from different maps are not pooled as one sample. Exact paired sign-flip tests are supplementary; effect sizes and bootstrap intervals remain primary.",
        ])
        (analysis / "report.md").write_text(report + "\n", encoding="utf-8")
        (temporary / "README.md").write_text(
            "# Strict A/B five-seed paper artifacts\n\n"
            "- `raw_data/`: all required training and validation files for 30 runs.\n"
            "- `figures/`: 350-dpi PNG and vector PDF learning, statistical, and trajectory figures.\n"
            "- `tables/`: CSV, XLSX, Markdown, and LaTeX tables.\n"
            "- `analysis/`: protocol audit, statistical summary, trajectory selection audit, and report.\n"
            "- No test data were generated, read, or archived.\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        update_global_archive(destination, loaded, provenance)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    print(json.dumps({
        "archive": str(destination),
        "training_runs": 30,
        "validation_only": True,
        "classification": cross_map["classification"],
        "png_figures": len(list((destination / "figures").rglob("*.png"))),
        "pdf_figures": len(list((destination / "figures").rglob("*.pdf"))),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
