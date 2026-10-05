"""Paired, map-stratified analysis of validation conflict adaptation AUC."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_IDS = (
    "irregular_workcell_91701",
    "irregular_workcell_91702",
    "irregular_workcell_91703",
)
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_STEPS = tuple(range(0, 200_001, 10_000))
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260929


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_curve(path: Path, *, adaptive: bool) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    required = {
        "environment_steps",
        "conflict_safe_success",
        "control_safe_success",
        "all_safe_success",
    }
    if adaptive:
        required.update({"dynamic_collision", "timeout"})
    if not source or not required.issubset(source[0]):
        raise SystemExit(f"Curve schema mismatch: {path}")
    rows = [
        {
            key: float(row[key]) if key != "environment_steps" else int(row[key])
            for key in required
        }
        for row in source
    ]
    steps = tuple(int(row["environment_steps"]) for row in rows)
    if steps != EXPECTED_STEPS:
        raise SystemExit(f"Expected 0:10k:200k checkpoints: {path} has {steps}")
    return rows


def normalized_auc(curve: list[dict[str, float]]) -> float:
    times = np.asarray([row["environment_steps"] for row in curve], dtype=float)
    values = np.asarray([row["conflict_safe_success"] for row in curve], dtype=float)
    return float(
        np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2.0)
        / EXPECTED_STEPS[-1]
    )


def mean_std(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def bootstrap_mean_ci(values, rng: np.random.Generator) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive-root",
        default="outputs/capability_adaptive_handover_v1_auc",
    )
    parser.add_argument("--baseline-root", default="baselines/time_decay")
    parser.add_argument(
        "--analysis-root",
        default="outputs/capability_adaptive_handover_v1_auc/analysis",
    )
    args = parser.parse_args()
    adaptive_root = ROOT / args.adaptive_root / "formal"
    baseline_root = ROOT / args.baseline_root
    analysis_root = ROOT / args.analysis_root
    if analysis_root.exists():
        raise SystemExit(
            f"Analysis output already exists and will not be overwritten: {analysis_root}"
        )

    per_seed = []
    for map_id in MAP_IDS:
        for seed in SEEDS:
            adaptive_dir = (
                adaptive_root
                / map_id
                / f"seed_{seed}"
                / "capability_adaptive_handover"
            )
            baseline_dir = baseline_root / map_id / f"seed_{seed}"
            adaptive_result = load_json(adaptive_dir / "result.json")
            baseline_result = load_json(baseline_dir / "result.json")
            if (
                adaptive_result.get("status") != "validation_complete"
                or adaptive_result.get("adaptation_steps") != 200_000
                or adaptive_result.get("test_deferred") is not True
                or adaptive_result.get("test_read_or_generated") is not False
                or adaptive_result.get("validation_controls_training") is not False
            ):
                raise SystemExit(f"Invalid adaptive result: {adaptive_dir}")
            if (
                baseline_result.get("status") != "validation_complete"
                or baseline_result.get("replay_schedule") != "decay"
                or baseline_result.get("adaptation_run", {}).get("steps") != 200_000
                or baseline_result.get("test_deferred") is not True
            ):
                raise SystemExit(f"Invalid time_decay baseline: {baseline_dir}")
            if (
                adaptive_result.get("map_id") != map_id
                or adaptive_result.get("seed") != seed
                or baseline_result.get("map_id") != map_id
                or baseline_result.get("seed") != seed
                or adaptive_result.get("snapshot_sha256")
                != baseline_result.get("snapshot_sha256")
            ):
                raise SystemExit(f"Map/seed/foundation pairing mismatch: {map_id} seed={seed}")

            adaptive_curve = load_curve(
                adaptive_dir / "validation_curve.csv", adaptive=True
            )
            baseline_curve = load_curve(
                baseline_dir / "validation_curve.csv", adaptive=False
            )
            adaptive_auc = normalized_auc(adaptive_curve)
            baseline_auc = normalized_auc(baseline_curve)
            last_50k = [
                row["conflict_safe_success"]
                for row in adaptive_curve
                if row["environment_steps"] > 150_000
            ]
            if len(last_50k) != 5:
                raise SystemExit("Last-50k window must contain checkpoints 160k..200k.")
            last_mean, last_std = mean_std(last_50k)
            per_seed.append(
                {
                    "map_id": map_id,
                    "seed": seed,
                    "adaptive_conflict_auc": adaptive_auc,
                    "time_decay_conflict_auc": baseline_auc,
                    "paired_auc_difference": adaptive_auc - baseline_auc,
                    "adaptive_gt_time_decay": int(adaptive_auc > baseline_auc),
                    "adaptive_final_conflict_safe_success": adaptive_curve[-1][
                        "conflict_safe_success"
                    ],
                    "time_decay_final_conflict_safe_success": baseline_curve[-1][
                        "conflict_safe_success"
                    ],
                    "adaptive_last_50k_mean": last_mean,
                    "adaptive_last_50k_std": last_std,
                    "expert_exit_step": adaptive_result.get("expert_exit_step"),
                    "effective_demo_fraction": adaptive_result[
                        "effective_demo_fraction"
                    ],
                    "foundation_snapshot_sha256": adaptive_result[
                        "snapshot_sha256"
                    ],
                }
            )

    map_summaries = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for map_id in MAP_IDS:
        rows = [row for row in per_seed if row["map_id"] == map_id]
        adaptive_auc_mean, adaptive_auc_std = mean_std(
            row["adaptive_conflict_auc"] for row in rows
        )
        baseline_auc_mean, baseline_auc_std = mean_std(
            row["time_decay_conflict_auc"] for row in rows
        )
        differences = [row["paired_auc_difference"] for row in rows]
        paired_mean, paired_std = mean_std(differences)
        ci_low, ci_high = bootstrap_mean_ci(differences, rng)
        final_mean, final_std = mean_std(
            row["adaptive_final_conflict_safe_success"] for row in rows
        )
        last_mean, last_std = mean_std(
            row["adaptive_last_50k_mean"] for row in rows
        )
        effective_mean, effective_std = mean_std(
            row["effective_demo_fraction"] for row in rows
        )
        exit_values = [
            float(row["expert_exit_step"])
            for row in rows
            if row["expert_exit_step"] is not None
        ]
        exit_mean, exit_std = (
            mean_std(exit_values) if exit_values else (math.nan, math.nan)
        )
        map_summaries.append(
            {
                "map_id": map_id,
                "seed_count": len(rows),
                "adaptive_conflict_auc_mean": adaptive_auc_mean,
                "adaptive_conflict_auc_std": adaptive_auc_std,
                "time_decay_conflict_auc_mean": baseline_auc_mean,
                "time_decay_conflict_auc_std": baseline_auc_std,
                "mean_paired_auc_difference": paired_mean,
                "paired_auc_difference_std": paired_std,
                "paired_difference_ci95_low": ci_low,
                "paired_difference_ci95_high": ci_high,
                "adaptive_gt_time_decay_seed_count": sum(
                    row["adaptive_gt_time_decay"] for row in rows
                ),
                "adaptive_final_conflict_safe_success_mean": final_mean,
                "adaptive_final_conflict_safe_success_std": final_std,
                "adaptive_last_50k_mean": last_mean,
                "adaptive_last_50k_across_seed_std": last_std,
                "expert_exit_count": len(exit_values),
                "expert_exit_step_mean_among_exits": exit_mean,
                "expert_exit_step_std_among_exits": exit_std,
                "effective_demo_fraction_mean": effective_mean,
                "effective_demo_fraction_std": effective_std,
            }
        )

    # Equal-map aggregation: first average the three maps within each matched
    # seed, then bootstrap the five seed-level averages. The 15 runs are never
    # treated as an exchangeable pooled sample.
    equal_map_seed_differences = []
    for seed in SEEDS:
        differences = [
            row["paired_auc_difference"]
            for row in per_seed
            if row["seed"] == seed
        ]
        if len(differences) != len(MAP_IDS):
            raise SystemExit(f"Missing map for equal-weight seed aggregation: {seed}")
        equal_map_seed_differences.append(float(np.mean(differences)))
    overall_ci_low, overall_ci_high = bootstrap_mean_ci(
        equal_map_seed_differences, rng
    )
    adaptive_auc_map_mean, adaptive_auc_map_std = mean_std(
        row["adaptive_conflict_auc_mean"] for row in map_summaries
    )
    baseline_auc_map_mean, baseline_auc_map_std = mean_std(
        row["time_decay_conflict_auc_mean"] for row in map_summaries
    )
    paired_map_mean, paired_map_std = mean_std(
        row["mean_paired_auc_difference"] for row in map_summaries
    )
    final_map_mean, final_map_std = mean_std(
        row["adaptive_final_conflict_safe_success_mean"]
        for row in map_summaries
    )
    last_50k_map_mean, last_50k_map_std = mean_std(
        row["adaptive_last_50k_mean"] for row in map_summaries
    )
    effective_map_mean, effective_map_std = mean_std(
        row["effective_demo_fraction_mean"] for row in map_summaries
    )
    overall = {
        "aggregation": "three map means weighted equally; no 15-run pooling",
        "last_50k_definition": "five checkpoints at 160k,170k,180k,190k,200k",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "map_level_ci_unit": "five paired seeds within each map",
            "overall_ci_unit": "five paired seeds after equal-weight map averaging",
        },
        "adaptive_conflict_auc_equal_map_mean": adaptive_auc_map_mean,
        "adaptive_conflict_auc_across_map_std": adaptive_auc_map_std,
        "time_decay_conflict_auc_equal_map_mean": baseline_auc_map_mean,
        "time_decay_conflict_auc_across_map_std": baseline_auc_map_std,
        "mean_paired_auc_difference_equal_map": paired_map_mean,
        "paired_auc_difference_across_map_std": paired_map_std,
        "paired_difference_equal_map_ci95_low": overall_ci_low,
        "paired_difference_equal_map_ci95_high": overall_ci_high,
        "adaptive_gt_time_decay_map_count": sum(
            row["mean_paired_auc_difference"] > 0.0 for row in map_summaries
        ),
        "adaptive_gt_time_decay_equal_map_seed_count": sum(
            value > 0.0 for value in equal_map_seed_differences
        ),
        "adaptive_final_conflict_safe_success_equal_map_mean": final_map_mean,
        "adaptive_final_conflict_safe_success_across_map_std": final_map_std,
        "adaptive_last_50k_equal_map_mean": last_50k_map_mean,
        "adaptive_last_50k_across_map_std": last_50k_map_std,
        "effective_demo_fraction_equal_map_mean": effective_map_mean,
        "effective_demo_fraction_across_map_std": effective_map_std,
        "expert_exit_by_map": {
            row["map_id"]: {
                "exit_count": row["expert_exit_count"],
                "mean_step_among_exits": (
                    None
                    if math.isnan(row["expert_exit_step_mean_among_exits"])
                    else row["expert_exit_step_mean_among_exits"]
                ),
            }
            for row in map_summaries
        },
        "per_seed_equal_map_paired_differences": {
            str(seed): value
            for seed, value in zip(SEEDS, equal_map_seed_differences)
        },
    }

    analysis_root.mkdir(parents=True, exist_ok=False)
    write_csv(analysis_root / "per_seed_metrics.csv", per_seed)
    write_csv(analysis_root / "per_map_summary.csv", map_summaries)
    (analysis_root / "equal_weight_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"maps": map_summaries, "equal_weight": overall}, indent=2))


if __name__ == "__main__":
    main()
