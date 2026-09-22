"""Summarize the preregistered validation-only indexed-risk replay ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astar_d3qn.utils.io import write_json, write_records_csv


METHODS = {
    "risk_sample_00_reused_time_decay": "schedule_decay",
    "risk_sample_04": "risk_sample_04",
    "risk_sample_16": "risk_sample_16",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_validation_run(root: Path, seed: int, method: str, branch: str):
    directory = root / f"seed_{seed}" / branch
    result_path = directory / "result.json"
    curve_path = directory / "validation_curve.csv"
    if not result_path.exists() or not curve_path.exists():
        raise SystemExit(f"Missing completed validation branch: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("smoke"):
        raise SystemExit(f"Not a complete formal result: {result_path}")
    if method == "risk_sample_00_reused_time_decay":
        if result.get("replay_schedule") != "decay":
            raise SystemExit("The zero-risk reference must be the existing time-decay branch.")
    elif result.get("risk_sample_count") != int(method.rsplit("_", 1)[1]):
        raise SystemExit(f"Risk-sample allocation mismatch: {result_path}")
    return result, read_csv(curve_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--map-index", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = args.output_root
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    map_id = manifest["maps"][args.map_index]["problem"]["map_id"]
    root = ROOT / config["output_root"] / "formal" / map_id

    per_seed = []
    curves = []
    fork_by_seed: dict[int, set[str]] = {seed: set() for seed in args.seeds}
    common_provenance = set()
    for seed in args.seeds:
        for method, branch in METHODS.items():
            result, curve = load_validation_run(root, seed, method, branch)
            fork_by_seed[seed].add(result["fork_sha256"])
            common_provenance.add((
                result["config_sha256"],
                result["manifest_sha256"],
                result["torch_version"],
                result["device"],
            ))
            run = result["adaptation_run"]
            per_seed.append({
                "method": method,
                "seed": seed,
                "validation_conflict_auc": result["validation_conflict_auc"],
                "threshold_confirmation_step": result["threshold_confirmation_step"],
                "risk_samples": run.get("risk_samples", 0),
                "risk_marked_total": run.get("risk_marked_total", 0),
                "active_risk_index_size": run.get("risk_buffer_size", 0),
                "demo_samples": run.get("demo_samples", 0),
                "online_samples": run.get("online_samples", 0),
                "code_sha256": result["code_sha256"],
                "fork_sha256": result["fork_sha256"],
            })
            curves.extend({"method": method, "seed": seed, **row} for row in curve)

    if any(len(values) != 1 for values in fork_by_seed.values()):
        raise SystemExit("Methods do not share one frozen foundation within each seed.")
    if len(common_provenance) != 1:
        raise SystemExit("Methods do not share config/manifest/torch/device provenance.")

    baseline = {
        row["seed"]: row
        for row in per_seed
        if row["method"] == "risk_sample_00_reused_time_decay"
    }
    paired = []
    means = {}
    for method in METHODS:
        selected = [row for row in per_seed if row["method"] == method]
        means[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "validation_conflict_auc",
                "risk_samples",
                "risk_marked_total",
                "active_risk_index_size",
            )
        }
        if method == "risk_sample_00_reused_time_decay":
            continue
        for row in selected:
            delta = (
                float(row["validation_conflict_auc"])
                - float(baseline[row["seed"]]["validation_conflict_auc"])
            )
            paired.append({
                "method": method,
                "seed": row["seed"],
                "validation_conflict_auc_delta_vs_zero": delta,
            })

    destination = root / "analysis_risk_sampling_ablation"
    destination.mkdir(parents=True, exist_ok=True)
    write_records_csv(per_seed, destination / "per_seed_validation.csv")
    write_records_csv(curves, destination / "validation_curves.csv")
    write_records_csv(paired, destination / "paired_validation_differences.csv")
    write_json({
        "map_id": map_id,
        "seeds": args.seeds,
        "methods": list(METHODS),
        "method_means": means,
        "selection_protocol": {
            "primary_metric": "mean validation_conflict_auc across paired seeds",
            "continue_rule": "positive mean delta versus zero and improvement on at least 2 of 3 seeds",
            "test_metrics_used_for_selection": False,
            "zero_risk_source": "existing schedule_decay after code-equivalence test",
        },
        "inference_unit": "paired training seed",
    }, destination / "summary.json")
    print(json.dumps({"map": map_id, "means": means, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
