"""Analyze v2 against strict A and v1 AUC without reading test data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_IDS = ("irregular_workcell_91701", "irregular_workcell_91702", "irregular_workcell_91703")
SEEDS = (0, 1, 2, 3, 4)
STEPS = tuple(range(0, 200_001, 10_000))
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260930


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_curve(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"environment_steps", "conflict_safe_success", "control_safe_success", "all_safe_success", "dynamic_collision", "timeout"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Validation curve schema mismatch: {path}")
    result = [{key: (int(row[key]) if key == "environment_steps" else float(row[key])) for key in required} for row in rows]
    if tuple(int(row["environment_steps"]) for row in result) != STEPS:
        raise SystemExit(f"Expected step 0..200k in {path}")
    return result


def load_a_curve(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "environment_steps" not in rows[0] or "conflict_safe_success" not in rows[0]:
        raise SystemExit(f"A curve schema mismatch: {path}")
    result = [{"environment_steps": int(row["environment_steps"]), "conflict_safe_success": float(row["conflict_safe_success"]), "control_safe_success": float(row.get("control_safe_success", 0.0)), "all_safe_success": float(row.get("all_safe_success", 0.0)), "dynamic_collision": float(row.get("all_dynamic_collision", row.get("dynamic_collision", 0.0))), "timeout": float(row.get("all_timeout", row.get("timeout", 0.0)))} for row in rows]
    if tuple(int(row["environment_steps"]) for row in result) != STEPS:
        raise SystemExit(f"Expected step 0..200k in {path}")
    return result


def normalized_auc(curve):
    times = np.asarray([row["environment_steps"] for row in curve], dtype=float)
    values = np.asarray([row["conflict_safe_success"] for row in curve], dtype=float)
    return float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2.0) / STEPS[-1])


def mean_std(values):
    values = np.asarray(list(values), dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def bootstrap_ci(values, rng):
    values = np.asarray(list(values), dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def load_monitor_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"environment_steps", "monitor_safe_success", "demo_fraction", "monitor_environment_steps_current", "monitor_environment_steps_cumulative", "total_algorithm_environment_interactions"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"v2 training schema mismatch: {path}")
    if tuple(int(row["environment_steps"]) for row in rows) != tuple(range(10_000, 200_001, 10_000)):
        raise SystemExit(f"v2 monitor checkpoints mismatch: {path}")
    return rows


def run_dir(root: Path, map_id: str, seed: int) -> Path:
    return root / "formal" / map_id / f"seed_{seed}" / "capability_adaptive_handover"


def validate_v2(result, audit, monitor_manifest, map_id, seed):
    required = {
        "status": "validation_complete",
        "method": "capability_adaptive_handover_v2_independent_monitor",
        "adaptation_steps": 200_000,
        "test_deferred": True,
        "test_read_or_generated": False,
        "validation_controls_training": False,
        "source_protocol": "ab_five_seed_strict_v1",
    }
    for key, value in required.items():
        if result.get(key) != value or audit.get(key) != value and key in {"source_protocol", "test_read_or_generated", "validation_controls_training"}:
            raise SystemExit(f"v2 provenance mismatch {map_id} seed={seed}: {key}")
    for key, value in {"verified_full_state_equal": True, "training_split_unchanged": True, "monitor_scenes_remain_in_training_stream": False, "monitor_scenes_in_training_stream": False, "monitor_scenes_in_replay": False, "validation_controls_training": False, "test_read_or_generated": False}.items():
        if audit.get(key) != value:
            raise SystemExit(f"v2 audit mismatch {map_id} seed={seed}: {key}")
    if monitor_manifest.get("scene_count") != 24 or monitor_manifest.get("density_counts") != {"3": 12, "5": 12}:
        raise SystemExit(f"v2 monitor manifest mismatch {map_id} seed={seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", default="results/capability_adaptive_handover_v2_independent_monitor")
    parser.add_argument("--a-root", default="baselines/time_decay")
    parser.add_argument("--v1-root", default="results/capability_adaptive_handover_v1_auc")
    parser.add_argument("--analysis-root", default="results/capability_adaptive_handover_v2_independent_monitor/analysis")
    args = parser.parse_args()
    v2_root, a_root, v1_root, analysis_root = [ROOT / value for value in (args.v2_root, args.a_root, args.v1_root, args.analysis_root)]
    if analysis_root.exists():
        raise SystemExit(f"Analysis exists and will not be overwritten: {analysis_root}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_seed = []
    mismatch_rows = []
    for map_id in MAP_IDS:
        for seed in SEEDS:
            v2_dir = run_dir(v2_root, map_id, seed)
            v1_dir = run_dir(v1_root, map_id, seed)
            a_dir = a_root / map_id / f"seed_{seed}"
            v2_result = load_json(v2_dir / "result.json")
            v2_audit = load_json(v2_dir / "run_audit.json")
            monitor_manifest = load_json(v2_dir / "monitor_scene_manifest.json")
            validate_v2(v2_result, v2_audit, monitor_manifest, map_id, seed)
            v2_curve = load_curve(v2_dir / "validation_curve.csv")
            v1_result = load_json(v1_dir / "result.json")
            if v1_result.get("status") != "validation_complete" or v1_result.get("test_deferred") is not True:
                raise SystemExit(f"Invalid v1 result: {v1_dir}")
            v1_curve = load_curve(v1_dir / "validation_curve.csv")
            a_result = load_json(a_dir / "result.json")
            if a_result.get("status") != "validation_complete" or a_result.get("test_deferred") is not True:
                raise SystemExit(f"Invalid A result: {a_dir}")
            a_curve = load_a_curve(a_dir / "validation_curve.csv")
            for other in (v1_result, a_result):
                if other.get("map_id") != map_id or other.get("seed") != seed:
                    raise SystemExit(f"Map/seed mismatch for {map_id} seed={seed}")
            if v2_result.get("snapshot_sha256") != a_result.get("snapshot_sha256") or v1_result.get("snapshot_sha256") != a_result.get("snapshot_sha256"):
                raise SystemExit(f"Foundation pairing mismatch for {map_id} seed={seed}")
            monitor_rows = load_monitor_rows(v2_dir / "training.csv")
            v2_auc, v1_auc, a_auc = normalized_auc(v2_curve), normalized_auc(v1_curve), normalized_auc(a_curve)
            v2_last = [row["conflict_safe_success"] for row in v2_curve if row["environment_steps"] > 150_000]
            v2_monitor = {int(row["environment_steps"]): float(row["monitor_safe_success"]) for row in monitor_rows}
            for curve_row in v2_curve[1:]:
                step = int(curve_row["environment_steps"])
                monitor_value = v2_monitor[step]
                validation_value = float(curve_row["conflict_safe_success"])
                mismatch_rows.append({"map_id": map_id, "seed": seed, "environment_steps": step, "monitor_safe_success": monitor_value, "validation_conflict_safe_success": validation_value, "monitor_minus_validation": monitor_value - validation_value, "absolute_gap": abs(monitor_value - validation_value), "severe_overestimation": int(monitor_value >= 0.9 and validation_value < 0.5)})
            overhead = int(monitor_rows[-1]["monitor_environment_steps_cumulative"])
            per_seed.append({"map_id": map_id, "seed": seed, "v2_conflict_auc": v2_auc, "time_decay_conflict_auc": a_auc, "v1_conflict_auc": v1_auc, "v2_minus_a_auc": v2_auc - a_auc, "v2_minus_v1_auc": v2_auc - v1_auc, "v2_gt_a": int(v2_auc > a_auc), "v2_gt_v1": int(v2_auc > v1_auc), "v2_final_conflict_safe_success": v2_curve[-1]["conflict_safe_success"], "v2_final_conflict_collision": v2_curve[-1]["dynamic_collision"], "v2_final_conflict_timeout": v2_curve[-1]["timeout"], "v2_last50k_conflict_mean": float(np.mean(v2_last)), "v2_last50k_conflict_std": float(np.std(v2_last, ddof=1)), "expert_exit_step": v2_result.get("expert_exit_step"), "effective_demo_fraction": v2_result.get("effective_demo_fraction"), "monitor_environment_steps_cumulative": overhead, "monitor_overhead_fraction": overhead / (200_000 + overhead), "total_algorithm_environment_interactions": 200_000 + overhead})
    map_summaries = []
    for map_id in MAP_IDS:
        rows = [row for row in per_seed if row["map_id"] == map_id]
        out = {"map_id": map_id, "seed_count": 5}
        for label in ("v2_conflict_auc", "v2_minus_a_auc", "v2_minus_v1_auc", "v2_final_conflict_safe_success", "v2_final_conflict_collision", "v2_final_conflict_timeout", "v2_last50k_conflict_mean", "effective_demo_fraction", "monitor_environment_steps_cumulative", "monitor_overhead_fraction"):
            mean, std = mean_std(row[label] for row in rows)
            out[f"{label}_mean"], out[f"{label}_std"] = mean, std
        for label in ("v2_minus_a_auc", "v2_minus_v1_auc"):
            low, high = bootstrap_ci([row[label] for row in rows], rng)
            out[f"{label}_ci95_low"], out[f"{label}_ci95_high"] = low, high
            out[f"{label}_win_count"] = sum(row[label] > 0 for row in rows)
        exits = [float(row["expert_exit_step"]) for row in rows if row["expert_exit_step"] is not None]
        out["expert_exit_count"] = len(exits)
        out["expert_exit_step_mean"] = float(np.mean(exits)) if exits else None
        map_summaries.append(out)
    equal = {"aggregation": "three map means weighted equally; never pooled 15 runs", "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}
    for label in ("v2_conflict_auc", "v2_minus_a_auc", "v2_minus_v1_auc", "v2_final_conflict_safe_success", "v2_last50k_conflict_mean", "effective_demo_fraction", "monitor_overhead_fraction"):
        values = [row[f"{label}_mean"] for row in map_summaries]
        equal[f"{label}_equal_map_mean"], equal[f"{label}_across_map_std"] = mean_std(values)
    for label in ("v2_minus_a_auc", "v2_minus_v1_auc"):
        seed_values = [float(np.mean([row[label] for row in per_seed if row["seed"] == seed])) for seed in SEEDS]
        equal[f"{label}_equal_map_ci95_low"], equal[f"{label}_equal_map_ci95_high"] = bootstrap_ci(seed_values, rng)
        equal[f"{label}_equal_map_win_seed_count"] = sum(value > 0 for value in seed_values)
    mismatch_summary = []
    for map_id in MAP_IDS:
        for seed in SEEDS:
            rows = [row for row in mismatch_rows if row["map_id"] == map_id and row["seed"] == seed]
            mismatch_summary.append({"map_id": map_id, "seed": seed, "mean_absolute_gap": float(np.mean([row["absolute_gap"] for row in rows])), "max_absolute_gap": float(np.max([row["absolute_gap"] for row in rows])), "severe_overestimation_count": sum(row["severe_overestimation"] for row in rows)})
    analysis_root.mkdir(parents=True, exist_ok=False)
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_csv(analysis_root / "per_seed_metrics.csv", per_seed)
    write_csv(analysis_root / "per_map_summary.csv", map_summaries)
    write_csv(analysis_root / "monitor_validation_mismatch.csv", mismatch_summary)
    write_csv(analysis_root / "monitor_validation_mismatch_checkpoints.csv", mismatch_rows)
    (analysis_root / "equal_weight_summary.json").write_text(json.dumps(equal, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"maps": map_summaries, "equal_weight": equal}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
