"""Summarize replay baselines and optional safe-intervention branches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astar_d3qn.utils.io import write_json, write_records_csv


METHODS = {
    "fixed_00": "demo_00",
    "fixed_10": "demo_10",
    "fixed_25": "demo_25",
    "time_decay": "schedule_decay",
    "risk_handover": "risk_handover",
}

OPTIONAL_METHODS = {
    "intervention_only": "intervention_only",
    "safe_replay_04": "safe_replay_04",
    "safe_replay_08": "safe_replay_08",
}


def load_branch(root, seed, branch):
    directory = root / f"seed_{seed}" / branch
    result_path = directory / "result.json"
    test_path = directory / "test_evaluation.csv"
    if not result_path.exists() or not test_path.exists():
        raise SystemExit(f"Missing completed branch: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("smoke"):
        raise SystemExit(f"Not a complete formal result: {result_path}")
    with test_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return result, rows


def density_rows(method, seed, rows):
    groups = defaultdict(list)
    for row in rows:
        if row["condition"] == "conflict":
            groups[int(row["obstacle_count"])].append(row)
    output = []
    for density, selected in sorted(groups.items()):
        output.append({
            "method": method,
            "seed": seed,
            "obstacle_count": density,
            "n_scenarios": len(selected),
            "safe_success": float(np.mean([int(row["safe_success"]) for row in selected])),
            "dynamic_collision": float(np.mean([int(row["dynamic_collision"]) for row in selected])),
            "timeout": float(np.mean([int(row["timeout"]) for row in selected])),
            "wait_steps": float(np.mean([float(row["wait_steps"]) for row in selected])),
        })
    return output


def main():
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
    by_density = []
    provenance = []
    methods = dict(METHODS)
    for method, branch in OPTIONAL_METHODS.items():
        availability = [
            (root / f"seed_{seed}" / branch / "result.json").exists()
            for seed in args.seeds
        ]
        if any(availability) and not all(availability):
            raise SystemExit(f"Optional branch is incomplete across seeds: {branch}")
        if all(availability):
            methods[method] = branch
    for seed in args.seeds:
        fork_digests = set()
        for method, branch in methods.items():
            result, rows = load_branch(root, seed, branch)
            provenance.append(tuple(result[key] for key in (
                "config_sha256", "manifest_sha256", "torch_version", "device"
            )))
            fork_digests.add(result["fork_sha256"])
            run = result["adaptation_run"]
            per_seed.append({
                "method": method,
                "seed": seed,
                "validation_conflict_auc": result["validation_conflict_auc"],
                "test_conflict_safe_success": result["test"]["conflict_safe_success"],
                "test_conflict_dynamic_collision": result["test"]["conflict_dynamic_collision"],
                "test_conflict_timeout": result["test"]["conflict_timeout"],
                "risk_samples": run.get("risk_samples", 0),
                "risk_coverage_count": run.get("risk_coverage_count", 0),
                "handover_progress": run.get("handover_progress", 0.0),
                "proposed_risk_steps": run.get("proposed_risk_steps", 0),
                "intervention_steps": run.get("intervention_steps", 0),
                "teacher_fallback_steps": run.get("teacher_fallback_steps", 0),
                "teacher_unavailable_steps": run.get("teacher_unavailable_steps", 0),
                "safe_samples": run.get("safe_samples", 0),
                "safe_buffer_size": run.get("safe_buffer_size", 0),
                "safe_coverage_count": run.get("safe_coverage_count", 0),
            })
            by_density.extend(density_rows(method, seed, rows))
        if len(fork_digests) != 1:
            raise SystemExit(f"Seed {seed} branches do not share one foundation fork.")
    if len(set(provenance)) != 1:
        raise SystemExit("Branches do not share one config/manifest/torch/device provenance.")
    method_means = {}
    for method in methods:
        selected = [row for row in per_seed if row["method"] == method]
        method_means[method] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "validation_conflict_auc", "test_conflict_safe_success",
                "test_conflict_dynamic_collision", "test_conflict_timeout",
                "risk_samples", "risk_coverage_count", "handover_progress",
                "proposed_risk_steps", "intervention_steps",
                "teacher_fallback_steps", "teacher_unavailable_steps",
                "safe_samples", "safe_buffer_size", "safe_coverage_count",
            )
        }
    density_means = []
    for method in methods:
        densities = sorted({row["obstacle_count"] for row in by_density if row["method"] == method})
        for density in densities:
            selected = [row for row in by_density
                        if row["method"] == method and row["obstacle_count"] == density]
            density_means.append({
                "method": method,
                "obstacle_count": density,
                **{key: float(np.mean([row[key] for row in selected]))
                   for key in ("safe_success", "dynamic_collision", "timeout", "wait_steps")},
            })
    destination = root / "analysis_safe_intervention"
    destination.mkdir(parents=True, exist_ok=True)
    write_records_csv(per_seed, destination / "per_seed_metrics.csv")
    write_records_csv(by_density, destination / "per_seed_density_metrics.csv")
    write_records_csv(density_means, destination / "density_summary.csv")
    write_json({
        "map_id": map_id,
        "seeds": args.seeds,
        "methods": list(methods),
        "method_means": method_means,
        "density_means": density_means,
        "inference_unit": "paired training seed",
    }, destination / "summary.json")
    print(json.dumps({"map": map_id, "means": method_means, "output": str(destination)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
