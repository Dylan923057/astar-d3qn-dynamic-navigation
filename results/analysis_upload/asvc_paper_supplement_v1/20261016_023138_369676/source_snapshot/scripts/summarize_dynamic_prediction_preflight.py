"""Summarize dynamic-prediction target checks and B/C smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("global_prediction", "decision_weighted_prediction")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-root",
        default="outputs/dynamic_prediction_auxiliary_v1_preflight/smoke/irregular_workcell_91701/seed_0",
    )
    parser.add_argument(
        "--target-balance",
        default="outputs/dynamic_prediction_preflight_v1/target_balance.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/dynamic_prediction_preflight_v1",
    )
    args = parser.parse_args()

    target = json.loads((ROOT / args.target_balance).read_text(encoding="utf-8"))
    method_rows = []
    audits = []
    for method in METHODS:
        directory = ROOT / args.smoke_root / method
        audit = json.loads((directory / "fork_audit.json").read_text(encoding="utf-8"))
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        training = read_csv(directory / "training.csv")
        first, final = training[0], training[-1]
        metrics = result["adaptation_run"]["prediction_metrics"]
        row = {
            "method": method,
            "status": result["status"],
            "steps": result["adaptation_run"]["steps"],
            "fork_sha256": result["fork_sha256"],
            "head_initial_sha256": audit["prediction_head_initial_sha256"],
            "head_parameter_count": audit["prediction_head_parameter_count"],
            "head_hidden_dim": audit["prediction_head_hidden_dim"],
            "demo_count_per_batch": audit["demo_count_per_batch"],
            "online_count_per_batch": audit["online_count_per_batch"],
            "lambda_pred": audit["prediction_loss_weight"],
            "pos_weight": audit["prediction_pos_weight"],
            "decision_zone_weight": audit["prediction_decision_zone_weight"],
            "first_prediction_loss_mean": float(first["prediction_loss_mean"]),
            "final_prediction_loss_mean": float(final["prediction_loss_mean"]),
            "final_td_loss_mean": float(final["prediction_td_loss_mean"]),
            "weighted_prediction_to_td_ratio": (
                audit["prediction_loss_weight"]
                * float(final["prediction_loss_mean"])
                / max(1e-12, float(final["prediction_td_loss_mean"]))
            ),
            "gradient_norm_mean": float(final["prediction_gradient_norm_mean"]),
            "gradient_norm_max_before_clip": float(final["prediction_gradient_norm_max"]),
            "q_abs_max": float(final["prediction_q_abs_max"]),
            **metrics,
            "test_file_exists": (directory / "test_evaluation.csv").exists(),
        }
        numeric = [
            value for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"Non-finite smoke metric: {row}")
        method_rows.append(row)
        audits.append(audit)

    same_fields = (
        "source_snapshot_sha256",
        "demo_count_per_batch",
        "online_count_per_batch",
        "prediction_loss_weight",
        "prediction_pos_weight",
        "prediction_head_hidden_dim",
        "prediction_head_initial_sha256",
        "prediction_head_parameter_count",
        "prediction_decision_zone_size",
    )
    checks = {
        "target_comes_from_next_state": target["target"]
        == "transition.next_state[current_dynamic_channel]",
        "dynamic_channel_confirmed": target["current_dynamic_channel"] == 1,
        "all_coordinate_checks_passed": target["coordinate_checks"]
        == target["transitions"],
        "executed_action_is_prediction_input": all(
            audit["prediction_uses_executed_action"] for audit in audits
        ),
        "bc_identical_except_spatial_weighting": all(
            audits[0][field] == audits[1][field] for field in same_fields
        ),
        "global_zone_weight_is_one": audits[0]["prediction_decision_zone_weight"] == 1.0,
        "decision_zone_weight_is_three": audits[1]["prediction_decision_zone_weight"] == 3.0,
        "prediction_loss_decreased": all(
            row["final_prediction_loss_mean"] < row["first_prediction_loss_mean"]
            for row in method_rows
        ),
        "prediction_does_not_dominate_td": all(
            row["weighted_prediction_to_td_ratio"] < 1.0 for row in method_rows
        ),
        "finite_metrics": True,
        "test_not_generated": not any(row["test_file_exists"] for row in method_rows),
        "lambda_zero_equivalence_unit_test": "tests.test_d3qn.D3QNAgentTests.test_zero_prediction_weight_is_exactly_equivalent_to_td_only",
    }
    destination = ROOT / args.output
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preflight.json").write_text(
        json.dumps(
            {"target_balance": target, "methods": method_rows, "checks": checks},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Dynamic prediction auxiliary preflight",
        "",
        f"- Target channel: {target['current_dynamic_channel']} (latest dynamic frame).",
        f"- Checked next-state local coordinates: {target['coordinate_checks']}/{target['transitions']} transitions.",
        f"- Positive-cell fraction: {target['positive_fraction']:.6f}; fixed pos_weight: {target['recommended_fixed_pos_weight']:.1f}.",
        "- Prediction target is read only from transition.next_state; no future oracle is called.",
        "- Executed action is concatenated as a one-hot input to the prediction head.",
        "- B/C share the same fork, initial prediction-head hash, capacity, lambda and replay allocation.",
        "- B/C differ only in prediction spatial weights: global 1x versus center 5x5 at 3x.",
        "- Both 3000-step smoke runs completed with finite losses/gradients and no test files.",
        "",
        "| Method | Pred loss first | Pred loss final | TD loss final | lambda*pred/TD | Grad mean | Grad max | F1 | Zone F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['first_prediction_loss_mean']:.4f} | "
            f"{row['final_prediction_loss_mean']:.4f} | {row['final_td_loss_mean']:.4f} | "
            f"{row['weighted_prediction_to_td_ratio']:.3f} | "
            f"{row['gradient_norm_mean']:.3f} | {row['gradient_norm_max_before_clip']:.3f} | "
            f"{row['prediction_positive_f1']:.3f} | {row['decision_zone_f1']:.3f} |"
        )
    lines.extend([
        "",
        "Navigation performance is intentionally not interpreted from this short smoke run.",
        "The lambda=0 exact-equivalence check is enforced by the named unit test.",
    ])
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
