"""Summarize the scheduled replay branch against existing fixed branches."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astar_d3qn.utils.io import write_json, write_records_csv


def load_result(root: Path, seed: int, branch: str) -> dict:
    path = root / f"seed_{seed}" / branch / "result.json"
    if not path.exists():
        raise SystemExit(f"Missing result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("smoke"):
        raise SystemExit(f"Result is not a complete formal run: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/replay_adaptation_v2.yaml")
    parser.add_argument("--map-index", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = args.output_root
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    map_id = manifest["maps"][args.map_index]["problem"]["map_id"]
    root = ROOT / config["output_root"] / "formal" / map_id
    destination = root / "analysis_schedule"
    rows = []
    for seed in args.seeds:
        scheduled = load_result(root, seed, "schedule_decay")
        row = {"seed": seed, "method": "decay", "validation_conflict_auc": scheduled["validation_conflict_auc"],
               "test_conflict_safe_success": scheduled["test"]["conflict_safe_success"],
               "test_conflict_dynamic_collision": scheduled["test"]["conflict_dynamic_collision"],
               "test_conflict_timeout": scheduled["test"]["conflict_timeout"],
               "effective_fraction": scheduled["effective_fraction"]}
        for fraction in (0.0, 0.10, 0.25):
            fixed = load_result(root, seed, f"demo_{round(fraction * 100):02d}")
            prefix = f"fixed_{round(fraction * 100):02d}"
            row[f"{prefix}_validation_conflict_auc"] = fixed["validation_conflict_auc"]
            row[f"{prefix}_test_conflict_safe_success"] = fixed["test"]["conflict_safe_success"]
            row[f"{prefix}_test_conflict_dynamic_collision"] = fixed["test"]["conflict_dynamic_collision"]
            row[f"{prefix}_test_conflict_timeout"] = fixed["test"]["conflict_timeout"]
        rows.append(row)
    metrics = ["validation_conflict_auc", "test_conflict_safe_success",
               "test_conflict_dynamic_collision", "test_conflict_timeout"]
    means = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
    fixed_means = {str(f): {key: float(np.mean([row[f"fixed_{round(f * 100):02d}_{key}"] for row in rows]))
                            for key in metrics} for f in (0.0, 0.10, 0.25)}
    report = {"map_id": map_id, "seeds": args.seeds, "method": "25% -> 10% -> 0%",
              "means": means, "fixed_means": fixed_means,
              "per_seed": rows,
              "interpretation": "Higher validation AUC and test success are better; lower collision and timeout are better."}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(report, destination / "summary.json")
    write_records_csv(rows, destination / "per_seed_comparison.csv")
    print(json.dumps({"map": map_id, "scheduled": means, "fixed": fixed_means,
                      "output": str(destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
