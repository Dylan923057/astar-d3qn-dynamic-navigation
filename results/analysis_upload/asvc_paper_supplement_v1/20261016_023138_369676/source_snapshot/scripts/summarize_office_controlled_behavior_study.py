"""Summarize the complete controlled Office behavior study.

Results are reported both overall and separately for the oracle-verified wait,
local-avoidance, and global-reroute scenario strata.
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
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_records_csv
from run_office_controlled_behavior_study import MAIN_CONFIG, METHODS


BEHAVIORS = ("wait", "avoidance", "reroute")
BEHAVIOR_LABELS = {
    "wait": "Wait",
    "avoidance": "Local avoidance",
    "reroute": "Global reroute",
}
METHOD_COLORS = {
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
GATE_CELLS = {
    "upper_left": frozenset(((10, 9), (11, 9))),
    "upper_right": frozenset(((10, 27), (11, 27))),
    "lower_left": frozenset(((25, 17), (26, 17))),
    "lower_right": frozenset(((25, 31), (26, 31))),
}
NOMINAL_GATE_PAIR = "upper_left+lower_left"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_dir(method, seed: int) -> Path:
    config = load_config(_resolve(method.config_path))
    return _resolve(config["experiment"]["output_root"]) / (
        f"{config['experiment']['run_name_prefix']}_seed_{seed}_{method.strategy}"
    )


def _gate_pair(path: Sequence[Sequence[int]]) -> str:
    cells = {tuple(int(value) for value in cell) for cell in path}
    upper = [
        name
        for name in ("upper_left", "upper_right")
        if cells.intersection(GATE_CELLS[name])
    ]
    lower = [
        name
        for name in ("lower_left", "lower_right")
        if cells.intersection(GATE_CELLS[name])
    ]
    if len(upper) != 1 or len(lower) != 1:
        return "unclassified"
    return f"{upper[0]}+{lower[0]}"


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan
    mean = fmean(finite)
    if len(finite) < 2:
        return mean, 0.0
    return mean, 1.96 * stdev(finite) / math.sqrt(len(finite))


def _load_scenario_rows() -> tuple[dict[int, dict[str, Any]], str]:
    config = load_config(_resolve(MAIN_CONFIG))
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(spatial["manifest"])
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = load_json(manifest_path)
    source = str(spatial.get("scenario_split_sources", {}).get("test", "test"))
    requested_ids = spatial.get("scenario_ids", {}).get("test")
    available = {
        int(row["scenario_id"]): row for row in manifest["scenarios"][source]
    }
    ids = (
        tuple(int(value) for value in requested_ids)
        if requested_ids is not None
        else tuple(available)
    )
    scenarios = {
        scenario_id: available[scenario_id] for scenario_id in ids
    }
    return scenarios, digest


def _scenario_results() -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    config = load_config(_resolve(MAIN_CONFIG))
    seeds = tuple(int(value) for value in config["training"]["seeds"])
    scenarios, expected_digest = _load_scenario_rows()
    expected_ids = set(scenarios)
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in METHODS:
        for seed in seeds:
            run_dir = _run_dir(method, seed)
            required = (
                run_dir / "run_metadata.json",
                run_dir / "test_evaluation.csv",
                run_dir / "test_trajectories.json",
                run_dir / "model_selected.pth",
            )
            absent = [path.name for path in required if not path.exists()]
            if absent:
                missing.append(f"{run_dir}: {', '.join(absent)}")
                continue
            metadata = load_json(run_dir / "run_metadata.json")
            if bool(metadata.get("smoke", False)):
                missing.append(f"{run_dir}: smoke output is not a formal run")
                continue
            digest = str(metadata["scenario_manifest"]["sha256"])
            if digest != expected_digest:
                raise ValueError(
                    f"{run_dir} used manifest {digest[:12]}, expected "
                    f"{expected_digest[:12]}."
                )
            evaluation = _read_csv(run_dir / "test_evaluation.csv")
            trajectories = load_json(run_dir / "test_trajectories.json")
            evaluation_by_id = {
                int(float(row["scenario_id"])): row for row in evaluation
            }
            trajectory_by_id = {
                int(row["scenario_id"]): row for row in trajectories
            }
            if (
                set(evaluation_by_id) != expected_ids
                or set(trajectory_by_id) != expected_ids
            ):
                raise ValueError(
                    f"{run_dir} does not contain the exact frozen test split."
                )
            for scenario_id in sorted(expected_ids):
                scenario = scenarios[scenario_id]
                row = evaluation_by_id[scenario_id]
                trajectory = trajectory_by_id[scenario_id]
                behavior = str(scenario["required_behavior"])
                safe_success = float(row["safe_success"])
                wait_steps = float(row["wait_steps"])
                actual_gate_pair = _gate_pair(trajectory["path"])
                oracle_gate_pair = _gate_pair(scenario["oracle_path"])
                oracle_gate_match = float(
                    safe_success > 0
                    and actual_gate_pair != "unclassified"
                    and actual_gate_pair == oracle_gate_pair
                )
                if behavior == "wait":
                    behavior_match = float(safe_success > 0 and wait_steps >= 1)
                elif behavior == "avoidance":
                    behavior_match = float(
                        safe_success > 0
                        and wait_steps == 0
                        and actual_gate_pair == NOMINAL_GATE_PAIR
                    )
                elif behavior == "reroute":
                    behavior_match = oracle_gate_match
                else:
                    raise ValueError(f"Unknown required behavior {behavior!r}.")
                minimum_steps = float(scenario["minimum_safe_path_steps"])
                results.append(
                    {
                        "method": method.key,
                        "method_label": method.label,
                        "training_seed": seed,
                        "scenario_id": scenario_id,
                        "required_behavior": behavior,
                        "safe_success": safe_success,
                        "success": float(row["success"]),
                        "dynamic_collision": float(row["dynamic_collision"]),
                        "dynamic_collision_count": float(
                            row["dynamic_collision_count"]
                        ),
                        "wait_steps": wait_steps,
                        "wait_used": float(wait_steps >= 1),
                        "steps": float(row["steps"]),
                        "minimum_safe_path_steps": minimum_steps,
                        "safe_excess_steps": (
                            float(row["steps"]) - minimum_steps
                            if safe_success > 0
                            else math.nan
                        ),
                        "actual_gate_pair": actual_gate_pair,
                        "oracle_gate_pair": oracle_gate_pair,
                        "oracle_gate_match": oracle_gate_match,
                        "behavior_match": behavior_match,
                    }
                )
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            f"All {len(METHODS) * len(seeds)} formal runs are required:\n{details}"
        )
    return results, seeds


def _aggregate(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = (
        "safe_success",
        "success",
        "dynamic_collision",
        "dynamic_collision_count",
        "wait_used",
        "wait_steps",
        "behavior_match",
        "oracle_gate_match",
        "safe_excess_steps",
    )
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["training_seed"]), "all")].append(row)
        grouped[
            (
                str(row["method"]),
                int(row["training_seed"]),
                str(row["required_behavior"]),
            )
        ].append(row)
    per_seed: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in seeds:
            for behavior in ("all", *BEHAVIORS):
                group = grouped[(method.key, seed, behavior)]
                record: dict[str, Any] = {
                    "method": method.key,
                    "method_label": method.label,
                    "training_seed": seed,
                    "required_behavior": behavior,
                    "scenario_count": len(group),
                }
                for metric in metrics:
                    values = [float(row[metric]) for row in group]
                    finite = [value for value in values if math.isfinite(value)]
                    record[metric] = fmean(finite) if finite else math.nan
                per_seed.append(record)

    summary: list[dict[str, Any]] = []
    for method in METHODS:
        for behavior in ("all", *BEHAVIORS):
            group = [
                row
                for row in per_seed
                if row["method"] == method.key
                and row["required_behavior"] == behavior
            ]
            record = {
                "method": method.key,
                "method_label": method.label,
                "required_behavior": behavior,
                "seed_count": len(group),
                "scenarios_per_seed": group[0]["scenario_count"],
            }
            for metric in metrics:
                mean, ci95 = _mean_ci95([float(row[metric]) for row in group])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_ci95"] = ci95
            summary.append(record)
    return per_seed, summary


def _plot(summary: Sequence[Mapping[str, Any]], output: Path) -> None:
    panels = (
        ("safe_success", "Collision-free success"),
        ("behavior_match", "Required behavior + safe success"),
        ("dynamic_collision", "Dynamic-collision episode rate"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.3), dpi=180)
    x = np.arange(len(BEHAVIORS), dtype=float)
    width = 0.082
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width
    lookup = {
        (str(row["method"]), str(row["required_behavior"])): row
        for row in summary
    }
    for axis, (metric, title) in zip(axes, panels):
        for offset, method in zip(offsets, METHODS):
            values = [
                float(lookup[(method.key, behavior)][f"{metric}_mean"])
                for behavior in BEHAVIORS
            ]
            errors = [
                float(lookup[(method.key, behavior)][f"{metric}_ci95"])
                for behavior in BEHAVIORS
            ]
            axis.bar(
                x + offset,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                color=METHOD_COLORS[method.key],
                label=method.label,
            )
        axis.set_title(title)
        axis.set_xticks(x, [BEHAVIOR_LABELS[item] for item in BEHAVIORS])
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Controlled Office behavior test | mean and 95% CI across seeds")
    figure.tight_layout()
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _percent(value: Any) -> str:
    number = float(value)
    return "n/a" if not math.isfinite(number) else f"{100 * number:.1f}%"


def _write_report(summary: Sequence[Mapping[str, Any]], output: Path) -> None:
    lookup = {
        (str(row["method"]), str(row["required_behavior"])): row
        for row in summary
    }
    lines = [
        "# Controlled Office behavior study",
        "",
        "All values are seed means on the frozen test split. Every scenario has five verified A*-interaction obstacles and no off-path filler obstacles.",
        "",
        "| Method | Overall safe success | Dynamic collision | Any wait |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = lookup[(method.key, "all")]
        lines.append(
            f"| {method.label} | {_percent(row['safe_success_mean'])} | "
            f"{_percent(row['dynamic_collision_mean'])} | "
            f"{_percent(row['wait_used_mean'])} |"
        )
    lines.extend(
        [
            "",
            "| Method | Wait: safe + waits | Avoidance: safe + nominal gates + no wait | Reroute: safe + oracle gate pair |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        values = [
            _percent(lookup[(method.key, behavior)]["behavior_match_mean"])
            for behavior in BEHAVIORS
        ]
        lines.append(f"| {method.label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "`behavior_match` is intentionally stricter than goal success: waiting must contain at least one STAY action; local avoidance must remain on the nominal gate pair without waiting; rerouting must use the oracle-selected alternative gate pair. `safe_success` remains the primary task metric.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/office_controlled_behavior_study_v1"
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, seeds = _scenario_results()
    per_seed, summary = _aggregate(rows, seeds)
    write_records_csv(rows, output_dir / "test_scenario_results.csv")
    write_records_csv(per_seed, output_dir / "test_per_seed.csv")
    write_records_csv(summary, output_dir / "test_summary.csv")
    _plot(summary, output_dir / "behavior_comparison.png")
    _write_report(summary, output_dir / "REPORT.md")
    print(f"Wrote controlled behavior summary to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
