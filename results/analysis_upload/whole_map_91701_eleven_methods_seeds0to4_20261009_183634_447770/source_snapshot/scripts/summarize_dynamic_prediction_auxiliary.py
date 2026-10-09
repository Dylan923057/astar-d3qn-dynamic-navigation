"""Create a validation-only A/B/C decision for dynamic prediction auxiliary learning."""

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


METHODS = {
    "A_time_decay": ("baseline", "schedule_decay"),
    "B_global_prediction": ("prediction", "global_prediction"),
    "C_decision_weighted_prediction": (
        "prediction",
        "decision_weighted_prediction",
    ),
}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row, key):
    raw = row.get(key)
    return None if raw in (None, "") else float(raw)


def validation_auc(curve, budget):
    points = [
        (int(row["environment_steps"]), float(row["conflict_safe_success"]))
        for row in curve
    ]
    if not points or points[0][0] != 0 or points[-1][0] != budget:
        raise ValueError("Validation curve does not span the fixed budget.")
    return sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value)
        in zip(points, points[1:])
    ) / budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--prediction-output-root",
        default="outputs/dynamic_prediction_auxiliary_v1",
    )
    parser.add_argument(
        "--output",
        default="outputs/dynamic_prediction_auxiliary_v1/analysis_validation",
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    map_id = manifest["maps"][args.map_index]["problem"]["map_id"]
    roots = {
        "baseline": ROOT / config["output_root"] / "formal" / map_id,
        "prediction": ROOT / args.prediction_output_root / "formal" / map_id,
    }
    budget = int(config["adaptation"]["max_steps"])
    records = []
    for method, (root_key, branch_name) in METHODS.items():
        for seed in args.seeds:
            branch = roots[root_key] / f"seed_{seed}" / branch_name
            curve_path = branch / "validation_curve.csv"
            if not curve_path.exists():
                raise SystemExit(f"Missing validation curve: {curve_path}")
            if root_key == "prediction" and (branch / "test_evaluation.csv").exists():
                raise SystemExit(f"Prediction test was read before validation freeze: {branch}")
            curve = read_csv(curve_path)
            final = curve[-1]
            threshold_step = next(
                (
                    int(row["environment_steps"])
                    for row in curve
                    if int(float(row["threshold_consecutive_passes"]))
                    >= int(config["adaptation"]["consecutive_passes"])
                ),
                None,
            )
            late = [
                float(row["conflict_safe_success"])
                for row in curve
                if int(row["environment_steps"]) >= budget - 50_000
            ]
            records.append({
                "method": method,
                "seed": seed,
                "validation_conflict_auc": validation_auc(curve, budget),
                "threshold_confirmation_step": threshold_step,
                "threshold_right_censored": int(threshold_step is None),
                "final_conflict_safe_success": value(final, "conflict_safe_success"),
                "final_conflict_dynamic_collision": value(
                    final, "conflict_dynamic_collision"
                ),
                "final_conflict_timeout": value(final, "conflict_timeout"),
                "final_static_safe_success": value(final, "static_safe_success"),
                "last_50k_conflict_safe_mean": statistics.fmean(late),
                "last_50k_conflict_safe_std": (
                    statistics.pstdev(late) if len(late) > 1 else 0.0
                ),
                "prediction_f1": value(final, "prediction_positive_f1"),
                "decision_zone_prediction_f1": value(final, "decision_zone_f1"),
            })

    indexed = {(row["method"], row["seed"]): row for row in records}
    means = {
        method: statistics.fmean(
            row["validation_conflict_auc"]
            for row in records
            if row["method"] == method
        )
        for method in METHODS
    }
    b_minus_a = means["B_global_prediction"] - means["A_time_decay"]
    c_minus_a = means["C_decision_weighted_prediction"] - means["A_time_decay"]
    c_minus_b = (
        means["C_decision_weighted_prediction"] - means["B_global_prediction"]
    )
    c_better_a = sum(
        indexed[("C_decision_weighted_prediction", seed)]["validation_conflict_auc"]
        > indexed[("A_time_decay", seed)]["validation_conflict_auc"]
        for seed in args.seeds
    )
    c_better_b = sum(
        indexed[("C_decision_weighted_prediction", seed)]["validation_conflict_auc"]
        > indexed[("B_global_prediction", seed)]["validation_conflict_auc"]
        for seed in args.seeds
    )
    no_regression = all(
        indexed[("C_decision_weighted_prediction", seed)]["final_conflict_dynamic_collision"]
        <= indexed[("A_time_decay", seed)]["final_conflict_dynamic_collision"] + 0.05
        and indexed[("C_decision_weighted_prediction", seed)]["final_conflict_timeout"]
        <= indexed[("A_time_decay", seed)]["final_conflict_timeout"] + 0.05
        and indexed[("C_decision_weighted_prediction", seed)]["final_static_safe_success"]
        >= indexed[("A_time_decay", seed)]["final_static_safe_success"] - 0.05
        for seed in args.seeds
    )
    decision = {
        "validation_only": True,
        "test_files_read": False,
        "mean_validation_auc": means,
        "B_minus_A": b_minus_a,
        "C_minus_A": c_minus_a,
        "C_minus_B": c_minus_b,
        "C_better_than_A_seed_count": c_better_a,
        "C_better_than_B_seed_count": c_better_b,
        "no_clear_final_regression": no_regression,
        "continue": c_minus_a >= 0.03 and c_better_a >= 2 and no_regression,
        "decision_weighting_adds_value": c_minus_b > 0.0,
    }
    destination = ROOT / args.output / map_id
    if destination.exists():
        raise SystemExit(f"Decision output already exists: {destination}")
    destination.mkdir(parents=True)
    write_records_csv(records, destination / "per_seed_metrics.csv")
    write_json(decision, destination / "decision.json")
    lines = [
        "# Dynamic prediction auxiliary validation decision",
        "",
        "This report reads validation curves only. Test files were neither required nor opened.",
        "",
        f"- B - A mean validation AUC: {b_minus_a:.4f}",
        f"- C - A mean validation AUC: {c_minus_a:.4f}",
        f"- C - B mean validation AUC: {c_minus_b:.4f}",
        f"- C better than A seeds: {c_better_a}/{len(args.seeds)}",
        f"- C better than B seeds: {c_better_b}/{len(args.seeds)}",
        f"- Continue: {decision['continue']}",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
