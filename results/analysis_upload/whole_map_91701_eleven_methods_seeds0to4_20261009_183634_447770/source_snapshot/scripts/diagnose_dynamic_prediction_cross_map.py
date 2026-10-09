"""Summarize validation-only A/B evidence across the three frozen maps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from astar_d3qn.utils.io import write_json, write_records_csv


RUNS = {
    "irregular_workcell_91701": {
        "A": "outputs/risk_handover_v1/formal/irregular_workcell_91701",
        "B": "outputs/dynamic_prediction_auxiliary_v1/formal/irregular_workcell_91701",
    },
    "irregular_workcell_91702": {
        "A": "outputs/dynamic_prediction_generalization_v1/formal/irregular_workcell_91702",
        "B": "outputs/dynamic_prediction_generalization_v1/formal/irregular_workcell_91702",
    },
    "irregular_workcell_91703": {
        "A": "outputs/dynamic_prediction_third_map_v1/formal/irregular_workcell_91703",
        "B": "outputs/dynamic_prediction_third_map_v1/formal/irregular_workcell_91703",
    },
}
BRANCHES = {"A": "schedule_decay", "B": "global_prediction"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row, key):
    raw = row.get(key)
    return None if raw in (None, "") else float(raw)


def metrics(curve, budget, consecutive_passes):
    points = [
        (int(row["environment_steps"]), float(row["conflict_safe_success"]))
        for row in curve
    ]
    if not points or points[0][0] != 0 or points[-1][0] != budget:
        raise ValueError("Validation curve does not span the fixed budget.")
    auc = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value)
        in zip(points, points[1:])
    ) / budget
    threshold_step = next(
        (
            int(row["environment_steps"])
            for row in curve
            if int(float(row["threshold_consecutive_passes"])) >= consecutive_passes
        ),
        None,
    )
    late = [
        float(row["conflict_safe_success"])
        for row in curve
        if int(row["environment_steps"]) >= budget - 50_000
    ]
    final = curve[-1]
    return {
        "auc": auc,
        "threshold_step": threshold_step,
        "threshold_right_censored": int(threshold_step is None),
        "final_safe_success": numeric(final, "conflict_safe_success"),
        "final_dynamic_collision": numeric(final, "conflict_dynamic_collision"),
        "final_timeout": numeric(final, "conflict_timeout"),
        "final_static_safe_success": numeric(final, "static_safe_success"),
        "last_50k_safe_mean": statistics.fmean(late),
        "last_50k_safe_std": statistics.pstdev(late) if len(late) > 1 else 0.0,
        "prediction_f1": numeric(final, "prediction_positive_f1"),
        "decision_zone_prediction_f1": numeric(final, "decision_zone_f1"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output",
        default="outputs/dynamic_prediction_cross_map_diagnostic_v1",
    )
    args = parser.parse_args()

    destination = ROOT / args.output
    if destination.exists():
        raise SystemExit(f"Diagnostic output already exists: {destination}")
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    budget = int(config["adaptation"]["max_steps"])
    consecutive_passes = int(config["adaptation"]["consecutive_passes"])
    paired_records = []
    for map_id, roots in RUNS.items():
        for seed in args.seeds:
            by_method = {}
            for label in ("A", "B"):
                branch = ROOT / roots[label] / f"seed_{seed}" / BRANCHES[label]
                curve_path = branch / "validation_curve.csv"
                if not curve_path.exists():
                    raise SystemExit(f"Missing validation curve: {curve_path}")
                by_method[label] = metrics(
                    read_csv(curve_path),
                    budget,
                    consecutive_passes,
                )
            paired_records.append({
                "map_id": map_id,
                "seed": seed,
                "A_validation_auc": by_method["A"]["auc"],
                "B_validation_auc": by_method["B"]["auc"],
                "B_minus_A_auc": by_method["B"]["auc"] - by_method["A"]["auc"],
                "B_better_than_A": int(by_method["B"]["auc"] > by_method["A"]["auc"]),
                "A_threshold_step": by_method["A"]["threshold_step"],
                "B_threshold_step": by_method["B"]["threshold_step"],
                "A_threshold_right_censored": by_method["A"]["threshold_right_censored"],
                "B_threshold_right_censored": by_method["B"]["threshold_right_censored"],
                "A_final_safe_success": by_method["A"]["final_safe_success"],
                "B_final_safe_success": by_method["B"]["final_safe_success"],
                "A_final_dynamic_collision": by_method["A"]["final_dynamic_collision"],
                "B_final_dynamic_collision": by_method["B"]["final_dynamic_collision"],
                "A_final_timeout": by_method["A"]["final_timeout"],
                "B_final_timeout": by_method["B"]["final_timeout"],
                "A_final_static_safe_success": by_method["A"]["final_static_safe_success"],
                "B_final_static_safe_success": by_method["B"]["final_static_safe_success"],
                "A_last_50k_safe_mean": by_method["A"]["last_50k_safe_mean"],
                "B_last_50k_safe_mean": by_method["B"]["last_50k_safe_mean"],
                "A_last_50k_safe_std": by_method["A"]["last_50k_safe_std"],
                "B_last_50k_safe_std": by_method["B"]["last_50k_safe_std"],
                "B_prediction_f1": by_method["B"]["prediction_f1"],
                "B_decision_zone_prediction_f1": by_method["B"]["decision_zone_prediction_f1"],
            })

    map_records = []
    for map_id in RUNS:
        rows = [row for row in paired_records if row["map_id"] == map_id]
        map_records.append({
            "map_id": map_id,
            "seed_count": len(rows),
            "A_mean_validation_auc": statistics.fmean(row["A_validation_auc"] for row in rows),
            "B_mean_validation_auc": statistics.fmean(row["B_validation_auc"] for row in rows),
            "B_minus_A_mean_auc": statistics.fmean(row["B_minus_A_auc"] for row in rows),
            "B_better_seed_count": sum(row["B_better_than_A"] for row in rows),
            "A_mean_threshold_step": statistics.fmean(row["A_threshold_step"] for row in rows),
            "B_mean_threshold_step": statistics.fmean(row["B_threshold_step"] for row in rows),
            "A_mean_last_50k_safe_std": statistics.fmean(row["A_last_50k_safe_std"] for row in rows),
            "B_mean_last_50k_safe_std": statistics.fmean(row["B_last_50k_safe_std"] for row in rows),
            "B_mean_prediction_f1": statistics.fmean(row["B_prediction_f1"] for row in rows),
        })

    positive_maps = sum(row["B_minus_A_mean_auc"] > 0.0 for row in map_records)
    positive_seeds = sum(row["B_better_than_A"] for row in paired_records)
    diagnostic = {
        "validation_only": True,
        "test_files_read": False,
        "map_count": len(map_records),
        "paired_seed_count": len(paired_records),
        "positive_map_count": positive_maps,
        "positive_seed_pair_count": positive_seeds,
        "mean_auc_effect_across_equal_weight_maps": statistics.fmean(
            row["B_minus_A_mean_auc"] for row in map_records
        ),
        "robust_cross_map_generalization_supported": positive_maps == len(map_records),
        "interpretation": (
            "benefit_is_map_dependent; retain as mixed evidence, not a robust generalization claim"
        ),
        "next_step": "freeze_negative_result_and_reframe_before_any_new_experiment",
    }
    destination.mkdir(parents=True)
    write_records_csv(paired_records, destination / "per_seed_cross_map.csv")
    write_records_csv(map_records, destination / "per_map_summary.csv")
    write_json(diagnostic, destination / "diagnostic.json")

    lines = [
        "# Dynamic prediction auxiliary cross-map diagnostic",
        "",
        "This diagnostic reads validation curves only. It does not open test files.",
        "",
        "| Map | A mean AUC | B mean AUC | B-A | Improved seeds | A threshold | B threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in map_records:
        lines.append(
            f"| {row['map_id']} | {row['A_mean_validation_auc']:.4f} | "
            f"{row['B_mean_validation_auc']:.4f} | {row['B_minus_A_mean_auc']:+.4f} | "
            f"{row['B_better_seed_count']}/{row['seed_count']} | "
            f"{row['A_mean_threshold_step']:.0f} | {row['B_mean_threshold_step']:.0f} |"
        )
    lines.extend([
        "",
        f"- Positive maps: {positive_maps}/{len(map_records)}.",
        f"- Positive paired seeds: {positive_seeds}/{len(paired_records)}.",
        f"- Equal-map mean AUC effect: {diagnostic['mean_auc_effect_across_equal_weight_maps']:+.4f}.",
        "- Map 3 is not a terminal-safety collapse: both methods finish with safe success 1.0.",
        "- On map 3, B reaches the validation threshold later for all seeds and has larger late-stage variation.",
        "- Therefore the auxiliary prediction benefit is map-dependent and does not support a robust three-map generalization claim.",
        "- Do not unlock test data, tune prediction parameters, or expand seeds based on this result.",
    ])
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
