"""Summarize control/conflict phase pairs for the causal Office benchmark.

The script deliberately excludes smoke runs.  Test metrics are computed within
each seed before aggregating across seeds, so every seed has equal weight.
Validation curves are joined to the frozen scenario manifest by scenario id.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv


DEFAULT_CONFIG = "configs/dynamic_spatial_generalization_office_causal_paired_v1.yaml"
STRATEGIES = ("uniform", "prefill", "persistent_demo")
LABELS = {
    "uniform": "Uniform",
    "prefill": "Prefill",
    "persistent_demo": "Persistent",
}
COLORS = {
    "uniform": "#64748b",
    "prefill": "#0ea5e9",
    "persistent_demo": "#e76f51",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else float("nan")


def _sd(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def _scenario_lookup(
    manifest: Mapping[str, Any], split: str
) -> dict[int, Mapping[str, Any]]:
    return {
        int(record["scenario_id"]): record
        for record in manifest["scenarios"][split]
    }


def summarize_test_rows(
    rows: Iterable[Mapping[str, Any]],
    scenarios: Mapping[int, Mapping[str, Any]],
) -> dict[str, float]:
    """Return condition and within-pair metrics for one trained seed."""
    joined: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for row in rows:
        scenario_id = int(row["scenario_id"])
        if scenario_id not in scenarios:
            raise ValueError(f"Unknown test scenario id {scenario_id}.")
        joined.append((row, scenarios[scenario_id]))
    if len(joined) != len(scenarios):
        raise ValueError(
            f"Test evaluation has {len(joined)} rows; expected {len(scenarios)}."
        )

    result: dict[str, float] = {}
    for condition in ("control", "conflict"):
        selected = [
            row
            for row, scenario in joined
            if str(scenario["pair_condition"]) == condition
        ]
        if not selected:
            raise ValueError(f"No {condition!r} test scenarios were evaluated.")
        result[f"{condition}_safe_success_rate"] = _mean(
            float(row["safe_success"]) for row in selected
        )
        result[f"{condition}_success_rate"] = _mean(
            float(row["success"]) for row in selected
        )
        result[f"{condition}_dynamic_collision_rate"] = _mean(
            float(row["dynamic_collision"]) for row in selected
        )
        result[f"{condition}_mean_steps"] = _mean(
            float(row["steps"]) for row in selected
        )
        result[f"{condition}_mean_wait_steps"] = _mean(
            float(row["wait_steps"]) for row in selected
        )

    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row, scenario in joined:
        pairs[str(scenario["pair_id"])][str(scenario["pair_condition"])] = row
    if any(set(pair) != {"control", "conflict"} for pair in pairs.values()):
        raise ValueError("Every evaluated pair must contain one control and one conflict.")
    degradation = []
    robust_both = []
    for pair in pairs.values():
        control = float(pair["control"]["safe_success"])
        conflict = float(pair["conflict"]["safe_success"])
        degradation.append(float(control > conflict))
        robust_both.append(float(control == 1.0 and conflict == 1.0))

    result["safe_success_gap_conflict_minus_control"] = (
        result["conflict_safe_success_rate"]
        - result["control_safe_success_rate"]
    )
    result["dynamic_collision_gap_conflict_minus_control"] = (
        result["conflict_dynamic_collision_rate"]
        - result["control_dynamic_collision_rate"]
    )
    result["paired_degradation_rate"] = _mean(degradation)
    result["paired_robust_both_rate"] = _mean(robust_both)
    result["pair_count"] = float(len(pairs))
    return result


def _validation_records(
    run_dir: Path,
    scenarios: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = _read_csv(run_dir / "validation_summary.csv")
    step_by_episode = {
        int(float(row["episode"])): int(float(row["environment_steps"]))
        for row in summaries
    }
    grouped: dict[tuple[int, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in _read_csv(run_dir / "validation_evaluation.csv"):
        scenario_id = int(row["scenario_id"])
        if scenario_id not in scenarios:
            raise ValueError(f"Unknown validation scenario id {scenario_id}.")
        episode = int(float(row["episode"]))
        condition = str(scenarios[scenario_id]["pair_condition"])
        grouped[(step_by_episode[episode], condition)].append(row)
    return [
        {
            "environment_steps": step,
            "condition": condition,
            "safe_success_rate": _mean(float(row["safe_success"]) for row in rows),
            "dynamic_collision_rate": _mean(
                float(row["dynamic_collision"]) for row in rows
            ),
        }
        for (step, condition), rows in sorted(grouped.items())
    ]


def _aggregate_test(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        name
        for name in run_rows[0]
        if name not in {"strategy", "seed", "run_dir"}
    ]
    records = []
    for strategy in STRATEGIES:
        selected = [row for row in run_rows if row["strategy"] == strategy]
        record: dict[str, Any] = {"strategy": strategy, "seed_count": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected]
            record[f"{metric}_mean"] = _mean(values)
            record[f"{metric}_sd"] = _sd(values)
        records.append(record)
    return records


def _aggregate_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["strategy"], int(row["environment_steps"]), row["condition"])
        ].append(row)
    records = []
    for (strategy, step, condition), selected in sorted(grouped.items()):
        safe = [float(row["safe_success_rate"]) for row in selected]
        collision = [float(row["dynamic_collision_rate"]) for row in selected]
        records.append(
            {
                "strategy": strategy,
                "environment_steps": step,
                "condition": condition,
                "seed_count": len(selected),
                "safe_success_rate_mean": _mean(safe),
                "safe_success_rate_sd": _sd(safe),
                "dynamic_collision_rate_mean": _mean(collision),
                "dynamic_collision_rate_sd": _sd(collision),
            }
        )
    return records


def _plot(
    test_summary: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    for strategy in STRATEGIES:
        for condition, linestyle in (("control", "-"), ("conflict", "--")):
            selected = [
                row
                for row in validation
                if row["strategy"] == strategy and row["condition"] == condition
            ]
            axes[0].plot(
                [row["environment_steps"] for row in selected],
                [100 * row["safe_success_rate_mean"] for row in selected],
                color=COLORS[strategy],
                linestyle=linestyle,
                linewidth=2,
                label=f"{LABELS[strategy]} / {condition}",
            )
    axes[0].set_title("Validation learning curve")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Safe success (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    x = np.arange(len(STRATEGIES))
    width = 0.34
    for offset, condition, hatch in (
        (-width / 2, "control", ""),
        (width / 2, "conflict", "//"),
    ):
        values = [
            100 * float(row[f"{condition}_safe_success_rate_mean"])
            for row in test_summary
        ]
        errors = [
            100 * float(row[f"{condition}_safe_success_rate_sd"])
            for row in test_summary
        ]
        axes[1].bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=[COLORS[strategy] for strategy in STRATEGIES],
            hatch=hatch,
            alpha=0.9,
            label=condition.title(),
        )
    axes[1].set_title("Untouched paired test")
    axes[1].set_ylabel("Safe success (%)")
    axes[1].set_ylim(0, 110)
    axes[1].set_xticks(x, [LABELS[strategy] for strategy in STRATEGIES])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.suptitle("Causal phase-paired dynamic-obstacle benchmark", fontsize=14)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Summarize available formal runs instead of requiring every seed.",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = load_config(config_path)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(str(spatial["manifest"]))
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = load_json(manifest_path)
    output_root = _resolve(str(config["experiment"]["output_root"]))
    prefix = str(config["experiment"]["run_name_prefix"])
    planned_seeds = tuple(int(seed) for seed in config["training"]["seeds"])
    test_lookup = _scenario_lookup(manifest, "test")
    validation_lookup = _scenario_lookup(manifest, "validation")

    run_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    missing = []
    for strategy in STRATEGIES:
        for seed in planned_seeds:
            run_dir = output_root / f"{prefix}_seed_{seed}_{strategy}"
            metadata_path = run_dir / "run_metadata.json"
            if not metadata_path.exists():
                missing.append(f"{strategy}/seed={seed}")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if bool(metadata.get("smoke")):
                raise ValueError(f"Formal run path contains smoke metadata: {run_dir}")
            if metadata["scenario_manifest"]["sha256"] != manifest_sha256:
                raise ValueError(f"Manifest hash mismatch in {run_dir}.")
            metrics = summarize_test_rows(
                _read_csv(run_dir / "test_evaluation.csv"), test_lookup
            )
            run_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    **metrics,
                }
            )
            validation_rows.extend(
                {"strategy": strategy, "seed": seed, **record}
                for record in _validation_records(run_dir, validation_lookup)
            )
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "Missing formal runs: " + ", ".join(missing) + ". Run the 3x3 command first."
        )
    if not run_rows:
        raise FileNotFoundError("No completed formal causal-paired runs were found.")

    test_summary = _aggregate_test(run_rows)
    validation_summary = _aggregate_validation(validation_rows)
    analysis_dir = output_root / "causal_paired_analysis"
    write_records_csv(run_rows, analysis_dir / "run_condition_metrics.csv")
    write_records_csv(test_summary, analysis_dir / "strategy_summary.csv")
    write_records_csv(
        validation_summary, analysis_dir / "validation_condition_curve.csv"
    )
    _plot(test_summary, validation_summary, analysis_dir / "causal_paired_summary.png")
    write_json(
        {
            "config": str(config_path),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "planned_seeds": list(planned_seeds),
            "missing_runs": missing,
            "interpretation": {
                "primary_metric": "conflict_safe_success_rate",
                "causal_penalty": "safe_success_gap_conflict_minus_control; negative is worse",
                "adaptive_gate": (
                    "Proceed only if persistent_demo learns earlier than both baselines "
                    "and has a consistently larger late conflict penalty across seeds."
                ),
            },
        },
        analysis_dir / "analysis_metadata.json",
    )
    print(f"Wrote causal paired analysis: {analysis_dir}")
    for row in test_summary:
        print(
            f"{LABELS[row['strategy']]:10s} "
            f"control={100 * row['control_safe_success_rate_mean']:5.1f}% "
            f"conflict={100 * row['conflict_safe_success_rate_mean']:5.1f}% "
            f"gap={100 * row['safe_success_gap_conflict_minus_control_mean']:+5.1f} pp"
        )


if __name__ == "__main__":
    main()
