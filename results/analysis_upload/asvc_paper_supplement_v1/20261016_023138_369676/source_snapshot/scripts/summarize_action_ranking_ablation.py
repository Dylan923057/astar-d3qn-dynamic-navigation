"""Summarize the validation-only A/B/C action-ranking ablation."""

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
    "B_demo_action_margin": "demo_action_margin",
    "C_all_action_margin": "all_action_margin",
}
EXPECTED_BASELINE_AUC = 0.7042
BASELINE_AUC_TOLERANCE = 0.0001


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def validation_auc(curve, *, budget):
    points = [
        (int(row["environment_steps"]), float(row["conflict_safe_success"]))
        for row in curve
    ]
    if not points or points[0][0] != 0 or points[-1][0] != budget:
        raise ValueError("Validation curve must span the complete fixed-step budget.")
    return sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value)
        in zip(points, points[1:])
    ) / budget


def threshold_confirmation_step(curve, *, required_passes, threshold):
    streak = 0
    for row in curve:
        if float(row["conflict_safe_success"]) >= threshold:
            streak += 1
        else:
            streak = 0
        if streak >= required_passes:
            return int(row["environment_steps"])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output",
        default="outputs/action_ranking_ablation_summary_v1",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Add fixed-final test metrics only after the validation decision is frozen.",
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    map_id = manifest["maps"][args.map_index]["problem"]["map_id"]
    source = ROOT / config["output_root"] / "formal" / map_id
    destination = ROOT / args.output / map_id
    if destination.exists():
        raise SystemExit(f"Summary output already exists: {destination}")

    records = []
    budget = int(config["adaptation"]["max_steps"])
    for method, branch_name in METHODS.items():
        for seed in args.seeds:
            branch = source / f"seed_{seed}" / branch_name
            curve_path = branch / "validation_curve.csv"
            if not curve_path.exists() or not (branch / "model_final.pth").exists():
                raise SystemExit(f"Missing completed branch: {branch}")
            curve = read_csv(curve_path)
            auc = validation_auc(curve, budget=budget)
            threshold_step = threshold_confirmation_step(
                curve,
                required_passes=int(config["adaptation"]["consecutive_passes"]),
                threshold=float(config["adaptation"]["safe_success_threshold"]),
            )
            final = curve[-1]
            late = [
                number(row, "conflict_safe_success")
                for row in curve
                if int(row["environment_steps"])
                >= budget - 50_000
            ]
            late = [value for value in late if value is not None]
            record = {
                "method": method,
                "seed": seed,
                "validation_conflict_auc": auc,
                "threshold_confirmation_step": threshold_step,
                "threshold_right_censored": int(threshold_step is None),
                "final_conflict_safe_success": number(final, "conflict_safe_success"),
                "final_conflict_dynamic_collision": number(
                    final, "conflict_dynamic_collision"
                ),
                "final_conflict_timeout": number(final, "conflict_timeout"),
                "final_static_safe_success": number(final, "static_safe_success"),
                "final_conflict_probe_unsafe_action": number(
                    final, "conflict_probe_unsafe_action"
                ),
                "final_conflict_probe_safe_q_margin": number(
                    final, "conflict_probe_safe_q_margin"
                ),
                "last_50k_conflict_safe_mean": statistics.fmean(late),
                "last_50k_conflict_safe_std": (
                    statistics.pstdev(late) if len(late) > 1 else 0.0
                ),
            }
            if args.include_test:
                result_path = branch / "result.json"
                if not result_path.exists():
                    raise SystemExit(f"Missing fixed-final test result: {result_path}")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                record.update({
                    f"test_{key}": value
                    for key, value in result["test"].items()
                })
            records.append(record)

    by_method = {
        method: [row for row in records if row["method"] == method]
        for method in METHODS
    }
    mean_auc = {
        method: statistics.fmean(row["validation_conflict_auc"] for row in rows)
        for method, rows in by_method.items()
    }
    indexed = {
        (row["method"], row["seed"]): row
        for row in records
    }
    c_better_seed_count = sum(
        indexed[("C_all_action_margin", seed)]["validation_conflict_auc"]
        > indexed[("A_time_decay", seed)]["validation_conflict_auc"]
        for seed in args.seeds
    )
    c_minus_a = mean_auc["C_all_action_margin"] - mean_auc["A_time_decay"]
    c_minus_b = mean_auc["C_all_action_margin"] - mean_auc["B_demo_action_margin"]
    baseline_matches_registered = (
        abs(mean_auc["A_time_decay"] - EXPECTED_BASELINE_AUC)
        <= BASELINE_AUC_TOLERANCE
    )
    no_clear_final_regression = all(
        indexed[("C_all_action_margin", seed)]["final_conflict_dynamic_collision"]
        <= indexed[("A_time_decay", seed)]["final_conflict_dynamic_collision"] + 0.05
        and indexed[("C_all_action_margin", seed)]["final_conflict_timeout"]
        <= indexed[("A_time_decay", seed)]["final_conflict_timeout"] + 0.05
        and indexed[("C_all_action_margin", seed)]["final_static_safe_success"]
        >= indexed[("A_time_decay", seed)]["final_static_safe_success"] - 0.05
        for seed in args.seeds
    )
    decision = {
        "validation_only": not args.include_test,
        "mean_validation_auc": mean_auc,
        "C_minus_A_mean_auc": c_minus_a,
        "C_minus_B_mean_auc": c_minus_b,
        "C_better_than_A_seed_count": c_better_seed_count,
        "required_seed_count": 2,
        "registered_baseline_auc": EXPECTED_BASELINE_AUC,
        "baseline_matches_registered": baseline_matches_registered,
        "no_clear_final_regression": no_clear_final_regression,
        "continue": (
            baseline_matches_registered
            and c_minus_a >= 0.03
            and c_better_seed_count >= 2
            and c_minus_b > 0.0
            and no_clear_final_regression
        ),
    }
    destination.mkdir(parents=True)
    write_records_csv(records, destination / "per_seed_metrics.csv")
    write_json(decision, destination / "decision.json")
    lines = [
        "# Action-ranking ablation decision",
        "",
        f"- C - A mean validation AUC: {c_minus_a:.3f}",
        f"- C - B mean validation AUC: {c_minus_b:.3f}",
        f"- C improves over A seeds: {c_better_seed_count}/{len(args.seeds)}",
        f"- A matches registered AUC {EXPECTED_BASELINE_AUC:.4f}: {baseline_matches_registered}",
        f"- No clear final regression: {no_clear_final_regression}",
        f"- Continue to larger study: {decision['continue']}",
        "",
        "This is a predeclared engineering gate, not a statistical-significance test.",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
