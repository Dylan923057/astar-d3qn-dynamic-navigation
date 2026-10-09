"""Freeze a validation-only A/B dynamic-prediction generalization decision."""

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
    "A_time_decay": "schedule_decay",
    "B_global_prediction": "global_prediction",
}


def read_csv(path):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--map-index", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--run-root",
        default="outputs/dynamic_prediction_generalization_v1",
    )
    parser.add_argument(
        "--output",
        default="outputs/dynamic_prediction_generalization_v1/analysis_validation",
    )
    parser.add_argument(
        "--safety-regression-tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed per-seed increase in final collision or timeout rate",
    )
    parser.add_argument(
        "--late-stability-gate",
        choices=("required", "report_only"),
        default="required",
    )
    parser.add_argument(
        "--next-step-on-pass",
        default="expand_to_5_seeds",
    )
    args = parser.parse_args()
    if args.safety_regression_tolerance < 0.0:
        raise SystemExit("Safety regression tolerance cannot be negative.")

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    map_id = manifest["maps"][args.map_index]["problem"]["map_id"]
    run_root = ROOT / args.run_root / "formal" / map_id
    budget = int(config["adaptation"]["max_steps"])
    records = []
    provenance = {}
    for method, branch_name in METHODS.items():
        for seed in args.seeds:
            branch = run_root / f"seed_{seed}" / branch_name
            result_path = branch / "result.json"
            curve_path = branch / "validation_curve.csv"
            if not result_path.exists() or not curve_path.exists():
                raise SystemExit(f"Missing completed validation run: {branch}")
            if (branch / "test_evaluation.csv").exists():
                raise SystemExit(f"Test was generated before validation freeze: {branch}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "validation_complete" or not result.get("test_deferred"):
                raise SystemExit(f"Run is not validation-only: {branch}")
            if result.get("adaptation_run", {}).get("steps") != budget:
                raise SystemExit(f"Run does not use the frozen 200k budget: {branch}")
            if result.get("replay_schedule") != (
                "decay" if method == "A_time_decay" else "global_prediction"
            ):
                raise SystemExit(f"Unexpected schedule: {branch}")
            if method == "A_time_decay" and (
                result.get("prediction_mode") is not None
                or result.get("prediction_loss_weight") != 0.0
            ):
                raise SystemExit(f"A is not the unchanged time-decay baseline: {branch}")
            if method == "B_global_prediction" and (
                result.get("prediction_mode") != "global_prediction"
                or result.get("prediction_loss_weight") != 0.1
                or result.get("prediction_pos_weight") != 20.0
            ):
                raise SystemExit(f"B parameters differ from the frozen protocol: {branch}")
            provenance[(method, seed)] = tuple(
                result[key]
                for key in (
                    "fork_sha256",
                    "config_sha256",
                    "manifest_sha256",
                    "code_sha256",
                )
            )

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
                "final_conflict_dynamic_collision": value(final, "conflict_dynamic_collision"),
                "final_conflict_timeout": value(final, "conflict_timeout"),
                "final_static_safe_success": value(final, "static_safe_success"),
                "last_50k_conflict_safe_mean": statistics.fmean(late),
                "last_50k_conflict_safe_std": statistics.pstdev(late) if len(late) > 1 else 0.0,
                "prediction_f1": value(final, "prediction_positive_f1"),
                "decision_zone_prediction_f1": value(final, "decision_zone_f1"),
            })

    for seed in args.seeds:
        if provenance[("A_time_decay", seed)] != provenance[("B_global_prediction", seed)]:
            raise SystemExit(f"A/B provenance or foundation fork differs for seed {seed}.")

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
    b_better_seed_count = sum(
        indexed[("B_global_prediction", seed)]["validation_conflict_auc"]
        > indexed[("A_time_decay", seed)]["validation_conflict_auc"]
        for seed in args.seeds
    )
    no_final_safety_regression = all(
        indexed[("B_global_prediction", seed)]["final_conflict_dynamic_collision"]
        <= indexed[("A_time_decay", seed)]["final_conflict_dynamic_collision"]
        + args.safety_regression_tolerance
        and indexed[("B_global_prediction", seed)]["final_conflict_timeout"]
        <= indexed[("A_time_decay", seed)]["final_conflict_timeout"]
        + args.safety_regression_tolerance
        for seed in args.seeds
    )
    late_stability = {
        method: statistics.fmean(
            row["last_50k_conflict_safe_std"]
            for row in records
            if row["method"] == method
        )
        for method in METHODS
    }
    late_stability_not_worse = (
        late_stability["B_global_prediction"]
        <= late_stability["A_time_decay"]
    )
    generalizes = (
        b_minus_a > 0.0
        and b_better_seed_count >= 2
        and no_final_safety_regression
        and (
            args.late_stability_gate == "report_only"
            or late_stability_not_worse
        )
    )
    decision = {
        "validation_only": True,
        "test_files_read": False,
        "matched_ab_forks_and_provenance": True,
        "map_id": map_id,
        "mean_validation_auc": means,
        "B_minus_A": b_minus_a,
        "B_better_than_A_seed_count": b_better_seed_count,
        "safety_regression_tolerance": args.safety_regression_tolerance,
        "no_final_collision_or_timeout_regression": no_final_safety_regression,
        "mean_last_50k_conflict_safe_std": late_stability,
        "late_stability_not_worse": late_stability_not_worse,
        "late_stability_gate": args.late_stability_gate,
        "generalizes": generalizes,
        "next_step": args.next_step_on_pass if generalizes else "stop_and_diagnose",
    }
    destination = ROOT / args.output / map_id
    if destination.exists():
        raise SystemExit(f"Decision output already exists: {destination}")
    destination.mkdir(parents=True)
    write_records_csv(records, destination / "per_seed_metrics.csv")
    write_json(decision, destination / "decision.json")
    lines = [
        "# Dynamic prediction cross-map validation decision",
        "",
        "This report reads validation curves only. Test files were neither required nor opened.",
        "",
        f"- Map: {map_id}",
        f"- B - A mean validation AUC: {b_minus_a:.4f}",
        f"- B better than A seeds: {b_better_seed_count}/{len(args.seeds)}",
        f"- Safety regression tolerance: {args.safety_regression_tolerance:.4f}",
        f"- No final collision/timeout regression: {no_final_safety_regression}",
        f"- Late stability not worse: {late_stability_not_worse}",
        f"- Late stability gate: {args.late_stability_gate}",
        f"- Generalizes: {generalizes}",
        f"- Next step: {decision['next_step']}",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
