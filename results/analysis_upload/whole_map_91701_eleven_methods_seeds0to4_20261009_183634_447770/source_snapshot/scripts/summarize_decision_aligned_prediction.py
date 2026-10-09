"""Freeze a validation-only A/B/D decision for gradient-aligned prediction."""

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
    "A_time_decay": ("reference", "schedule_decay"),
    "B_global_prediction": ("reference", "global_prediction"),
    "D_decision_aligned_prediction": (
        "aligned",
        "decision_aligned_prediction",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    return None if raw in (None, "") else float(raw)


def validation_auc(curve: list[dict[str, str]], budget: int) -> float:
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
    parser.add_argument("--map-index", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--reference-root",
        default="outputs/dynamic_prediction_third_map_v1",
    )
    parser.add_argument(
        "--aligned-root",
        default="outputs/decision_aligned_prediction_v1",
    )
    parser.add_argument(
        "--output",
        default="outputs/decision_aligned_prediction_v1/analysis_validation",
    )
    parser.add_argument("--clear-improvement-margin", type=float, default=0.03)
    args = parser.parse_args()
    if args.clear_improvement_margin < 0.0:
        raise SystemExit("Clear-improvement margin cannot be negative.")

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    map_id = manifest["maps"][args.map_index]["problem"]["map_id"]
    roots = {
        "reference": ROOT / args.reference_root / "formal" / map_id,
        "aligned": ROOT / args.aligned_root / "formal" / map_id,
    }
    budget = int(config["adaptation"]["max_steps"])
    consecutive_passes = int(config["adaptation"]["consecutive_passes"])
    records: list[dict[str, object]] = []
    results: dict[tuple[str, int], dict[str, object]] = {}

    for method, (root_name, branch_name) in METHODS.items():
        for seed in args.seeds:
            branch = roots[root_name] / f"seed_{seed}" / branch_name
            result_path = branch / "result.json"
            curve_path = branch / "validation_curve.csv"
            if not result_path.exists() or not curve_path.exists():
                raise SystemExit(f"Missing completed validation run: {branch}")
            if (branch / "test_evaluation.csv").exists():
                raise SystemExit(f"Test was generated before validation freeze: {branch}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "validation_complete" or not result.get(
                "test_deferred"
            ):
                raise SystemExit(f"Run is not validation-only: {branch}")
            if result.get("adaptation_run", {}).get("steps") != budget:
                raise SystemExit(f"Run does not use the frozen 200k budget: {branch}")
            expected_schedule = {
                "A_time_decay": "decay",
                "B_global_prediction": "global_prediction",
                "D_decision_aligned_prediction": "decision_aligned_prediction",
            }[method]
            if result.get("replay_schedule") != expected_schedule:
                raise SystemExit(f"Unexpected schedule: {branch}")
            if method == "A_time_decay":
                if result.get("prediction_mode") is not None:
                    raise SystemExit(f"A is not the unchanged baseline: {branch}")
            else:
                if (
                    result.get("prediction_mode") != "global_prediction"
                    or result.get("prediction_loss_weight") != 0.1
                    or result.get("prediction_pos_weight") != 20.0
                    or result.get("prediction_decision_zone_weight") != 1.0
                ):
                    raise SystemExit(f"Prediction settings are not frozen: {branch}")
            expected_strategy = (
                "project_conflicting"
                if method == "D_decision_aligned_prediction"
                else "none"
            )
            if result.get("prediction_gradient_strategy", "none") != expected_strategy:
                raise SystemExit(f"Unexpected gradient strategy: {branch}")

            curve = read_csv(curve_path)
            final = curve[-1]
            late_values = [
                float(row["conflict_safe_success"])
                for row in curve
                if int(row["environment_steps"]) >= budget - 50_000
            ]
            threshold_step = next(
                (
                    int(row["environment_steps"])
                    for row in curve
                    if int(float(row["threshold_consecutive_passes"]))
                    >= consecutive_passes
                ),
                None,
            )
            run = result["adaptation_run"]
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
                "last_50k_conflict_safe_mean": statistics.fmean(late_values),
                "last_50k_conflict_safe_std": (
                    statistics.pstdev(late_values) if len(late_values) > 1 else 0.0
                ),
                "prediction_f1": value(final, "prediction_positive_f1"),
                "gradient_cosine_before_mean": run.get(
                    "td_prediction_gradient_cosine_before_mean"
                ),
                "gradient_cosine_after_mean": run.get(
                    "td_prediction_gradient_cosine_after_mean"
                ),
                "conflict_batch_ratio": run.get(
                    "prediction_gradient_conflict_ratio"
                ),
                "corrected_prediction_gradient_ratio": run.get(
                    "prediction_gradient_corrected_ratio"
                ),
                "removed_prediction_gradient_norm_ratio_mean": run.get(
                    "prediction_gradient_removed_norm_ratio_mean"
                ),
                "td_loss_before_update_mean": run.get(
                    "aligned_td_loss_before_update_mean"
                ),
                "td_loss_after_update_mean": run.get(
                    "aligned_td_loss_after_update_mean"
                ),
                "td_loss_update_delta_mean": run.get(
                    "aligned_td_loss_update_delta_mean"
                ),
                "prediction_loss_before_update_mean": run.get(
                    "aligned_prediction_loss_before_update_mean"
                ),
                "prediction_loss_after_update_mean": run.get(
                    "aligned_prediction_loss_after_update_mean"
                ),
                "prediction_loss_update_delta_mean": run.get(
                    "aligned_prediction_loss_update_delta_mean"
                ),
            })
            results[(method, seed)] = result

    for seed in args.seeds:
        a = results[("A_time_decay", seed)]
        b = results[("B_global_prediction", seed)]
        d = results[("D_decision_aligned_prediction", seed)]
        for key in ("fork_sha256", "manifest_sha256", "map_id", "grid_sha256"):
            if not a.get(key) == b.get(key) == d.get(key):
                raise SystemExit(f"A/B/D provenance differs for seed {seed}: {key}")

    indexed = {(row["method"], row["seed"]): row for row in records}
    mean_auc = {
        method: statistics.fmean(
            float(row["validation_conflict_auc"])
            for row in records
            if row["method"] == method
        )
        for method in METHODS
    }
    mean_late = {
        method: statistics.fmean(
            float(row["last_50k_conflict_safe_mean"])
            for row in records
            if row["method"] == method
        )
        for method in METHODS
    }
    mean_late_std = {
        method: statistics.fmean(
            float(row["last_50k_conflict_safe_std"])
            for row in records
            if row["method"] == method
        )
        for method in METHODS
    }
    mean_final_metrics = {
        method: {
            key: statistics.fmean(
                float(row[key])
                for row in records
                if row["method"] == method and row[key] is not None
            )
            for key in (
                "final_conflict_safe_success",
                "final_conflict_dynamic_collision",
                "final_conflict_timeout",
                "final_static_safe_success",
            )
        }
        for method in METHODS
    }
    baseline = "A_time_decay"
    prediction = "B_global_prediction"
    aligned = "D_decision_aligned_prediction"
    d_minus_b = mean_auc[aligned] - mean_auc[prediction]
    d_minus_a = mean_auc[aligned] - mean_auc[baseline]
    d_better_b_seed_count = sum(
        float(indexed[(aligned, seed)]["validation_conflict_auc"])
        > float(indexed[(prediction, seed)]["validation_conflict_auc"])
        for seed in args.seeds
    )
    d_not_below_a_seed_count = sum(
        float(indexed[(aligned, seed)]["validation_conflict_auc"])
        >= float(indexed[(baseline, seed)]["validation_conflict_auc"])
        for seed in args.seeds
    )
    no_collision_or_timeout_regression = all(
        float(indexed[(aligned, seed)]["final_conflict_dynamic_collision"])
        <= float(indexed[(baseline, seed)]["final_conflict_dynamic_collision"])
        and float(indexed[(aligned, seed)]["final_conflict_timeout"])
        <= float(indexed[(baseline, seed)]["final_conflict_timeout"])
        for seed in args.seeds
    )
    late_performance_not_worse = mean_late[aligned] >= mean_late[baseline]
    late_stability_not_worse = mean_late_std[aligned] <= mean_late_std[baseline]
    mechanism_active = all(
        int(
            results[(aligned, seed)]["adaptation_run"].get(
                "gradient_alignment_update_count", 0
            )
        ) > 0
        for seed in args.seeds
    )
    mechanism_keys = (
        "gradient_cosine_before_mean",
        "gradient_cosine_after_mean",
        "conflict_batch_ratio",
        "corrected_prediction_gradient_ratio",
        "removed_prediction_gradient_norm_ratio_mean",
        "td_loss_before_update_mean",
        "td_loss_after_update_mean",
        "td_loss_update_delta_mean",
        "prediction_loss_before_update_mean",
        "prediction_loss_after_update_mean",
        "prediction_loss_update_delta_mean",
    )
    mean_mechanism_metrics = {
        key: statistics.fmean(
            float(indexed[(aligned, seed)][key]) for seed in args.seeds
        )
        for key in mechanism_keys
    }
    passes = (
        d_minus_b >= args.clear_improvement_margin
        and d_better_b_seed_count >= 2
        and d_minus_a >= 0.0
        and d_not_below_a_seed_count >= 2
        and no_collision_or_timeout_regression
        and late_performance_not_worse
        and late_stability_not_worse
        and mechanism_active
    )
    decision = {
        "validation_only": True,
        "test_files_read": False,
        "map_id": map_id,
        "fixed_budget": budget,
        "mean_validation_auc": mean_auc,
        "mean_final_metrics": mean_final_metrics,
        "D_minus_B": d_minus_b,
        "D_minus_A": d_minus_a,
        "clear_improvement_margin": args.clear_improvement_margin,
        "D_better_than_B_seed_count": d_better_b_seed_count,
        "D_not_below_A_seed_count": d_not_below_a_seed_count,
        "no_final_collision_or_timeout_regression": (
            no_collision_or_timeout_regression
        ),
        "mean_last_50k_conflict_safe_success": mean_late,
        "mean_last_50k_conflict_safe_std": mean_late_std,
        "late_performance_not_worse_than_A": late_performance_not_worse,
        "late_stability_not_worse_than_A": late_stability_not_worse,
        "gradient_alignment_mechanism_active": mechanism_active,
        "decision_aligned_mean_mechanism_metrics": mean_mechanism_metrics,
        "phase_one_passes": passes,
        "next_step": (
            "freeze_method_and_create_unseen_map"
            if passes
            else "stop_without_tuning_and_report_negative_or_inconclusive_result"
        ),
    }
    destination = ROOT / args.output / map_id
    if destination.exists():
        raise SystemExit(f"Decision output already exists: {destination}")
    destination.mkdir(parents=True)
    write_records_csv(records, destination / "per_seed_metrics.csv")
    write_json(decision, destination / "decision.json")
    lines = [
        "# Decision-aligned prediction validation decision",
        "",
        "This report reads validation artifacts only; test artifacts are forbidden.",
        "",
        f"- Map: {map_id}",
        f"- D - B validation AUC: {d_minus_b:.4f}",
        f"- D - A validation AUC: {d_minus_a:.4f}",
        f"- D better than B seeds: {d_better_b_seed_count}/{len(args.seeds)}",
        f"- D not below A seeds: {d_not_below_a_seed_count}/{len(args.seeds)}",
        f"- No collision/timeout regression: {no_collision_or_timeout_regression}",
        f"- Late performance not worse: {late_performance_not_worse}",
        f"- Late stability not worse: {late_stability_not_worse}",
        f"- Gradient-alignment mechanism active: {mechanism_active}",
        f"- Phase one passes: {passes}",
        f"- Next step: {decision['next_step']}",
    ]
    (destination / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
