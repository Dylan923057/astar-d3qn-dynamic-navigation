"""Analyze cross-map environment, prediction, and navigation relationships."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astar_d3qn.utils.io import write_json, write_records_csv


SPLITS = ("train", "validation")
CORE_CONTROL_FIELDS = (
    "mean_obstacle_count",
    "mean_causal_obstacle_count",
    "mean_move_every",
    "designed_conflict_scenario_fraction",
    "early_pair_count",
    "middle_pair_count",
    "late_pair_count",
)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def population_std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def summarize_pairs(map_id, split, pairs, astar_steps):
    obstacle_counts = [int(pair["obstacle_count"]) for pair in pairs]
    causal_counts = [int(pair["causal_obstacle_count"]) for pair in pairs]
    collision_steps = [int(pair["first_reference_collision_step"]) for pair in pairs]
    risk_counts = [len(pair["risk_positions"]) for pair in pairs]
    progress = Counter(pair["progress_band"] for pair in pairs)
    directions = []
    move_every = []
    route_lengths = []
    for pair in pairs:
        for obstacle in pair["conflict"]["obstacles"]:
            directions.append(int(obstacle["direction"]))
            move_every.append(int(obstacle.get("move_every", 1)))
            route_lengths.append(len(obstacle["route"]))
    density_counts = Counter(obstacle_counts)
    causal_density_counts = Counter(causal_counts)
    return {
        "map_id": map_id,
        "split": split,
        "pair_count": len(pairs),
        "scenario_count": 2 * len(pairs),
        "obstacle_count_distribution": json.dumps(dict(sorted(density_counts.items()))),
        "mean_obstacle_count": mean(obstacle_counts),
        "min_obstacle_count": min(obstacle_counts),
        "max_obstacle_count": max(obstacle_counts),
        "causal_obstacle_distribution": json.dumps(dict(sorted(causal_density_counts.items()))),
        "mean_causal_obstacle_count": mean(causal_counts),
        "mean_causal_obstacle_fraction": mean(
            causal / total for causal, total in zip(causal_counts, obstacle_counts)
        ),
        "early_pair_count": progress["early"],
        "middle_pair_count": progress["middle"],
        "late_pair_count": progress["late"],
        "mean_first_collision_step": mean(collision_steps),
        "std_first_collision_step": population_std(collision_steps),
        "mean_first_collision_progress": mean(collision_steps) / astar_steps,
        "mean_risk_positions_per_conflict_scenario": mean(risk_counts),
        "risk_positions_per_100_reference_steps": mean(risk_counts) * 100.0 / astar_steps,
        "positive_direction_fraction": sum(direction > 0 for direction in directions) / len(directions),
        "negative_direction_fraction": sum(direction < 0 for direction in directions) / len(directions),
        "mean_move_every": mean(move_every),
        "min_move_every": min(move_every),
        "max_move_every": max(move_every),
        "mean_cells_per_step": mean(1.0 / interval for interval in move_every),
        "mean_route_length": mean(route_lengths),
        "std_route_length": population_std(route_lengths),
        "designed_conflict_scenario_fraction": 0.5,
    }


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/risk_handover_v1/manifest.json")
    parser.add_argument(
        "--cross-map-diagnostic",
        default="outputs/dynamic_prediction_cross_map_diagnostic_v1",
    )
    parser.add_argument(
        "--output",
        default="outputs/dynamic_prediction_mechanism_analysis_v1",
    )
    args = parser.parse_args()

    destination = ROOT / args.output
    if destination.exists():
        raise SystemExit(f"Mechanism analysis output already exists: {destination}")
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    split_records = []
    map_records = []
    for entry in manifest["maps"]:
        map_id = entry["problem"]["map_id"]
        astar_steps = len(entry["problem"]["nominal_path"]) - 1
        combined = []
        for split in SPLITS:
            pairs = entry["scenarios"]["splits"][split]
            combined.extend(pairs)
            split_records.append(summarize_pairs(map_id, split, pairs, astar_steps))
        map_records.append(summarize_pairs(map_id, "train+validation", combined, astar_steps))

    diagnostic_root = ROOT / args.cross_map_diagnostic
    paired = read_csv(diagnostic_root / "per_seed_cross_map.csv")
    performance = {
        row["map_id"]: row
        for row in read_csv(diagnostic_root / "per_map_summary.csv")
    }
    prediction_navigation_records = []
    for environment in map_records:
        map_id = environment["map_id"]
        seed_rows = [row for row in paired if row["map_id"] == map_id]
        prediction_navigation_records.append({
            "map_id": map_id,
            "mean_prediction_f1": mean(float(row["B_prediction_f1"]) for row in seed_rows),
            "mean_decision_zone_prediction_f1": mean(
                float(row["B_decision_zone_prediction_f1"]) for row in seed_rows
            ),
            "A_mean_validation_auc": float(performance[map_id]["A_mean_validation_auc"]),
            "B_mean_validation_auc": float(performance[map_id]["B_mean_validation_auc"]),
            "B_minus_A_mean_auc": float(performance[map_id]["B_minus_A_mean_auc"]),
            "B_better_seed_count": int(performance[map_id]["B_better_seed_count"]),
            "A_mean_threshold_step": float(performance[map_id]["A_mean_threshold_step"]),
            "B_mean_threshold_step": float(performance[map_id]["B_mean_threshold_step"]),
            "A_mean_last_50k_safe_std": float(performance[map_id]["A_mean_last_50k_safe_std"]),
            "B_mean_last_50k_safe_std": float(performance[map_id]["B_mean_last_50k_safe_std"]),
            "mean_obstacle_count": environment["mean_obstacle_count"],
            "mean_causal_obstacle_count": environment["mean_causal_obstacle_count"],
            "mean_first_collision_progress": environment["mean_first_collision_progress"],
            "risk_positions_per_100_reference_steps": environment["risk_positions_per_100_reference_steps"],
            "mean_cells_per_step": environment["mean_cells_per_step"],
            "mean_route_length": environment["mean_route_length"],
        })

    map_prediction = [row["mean_prediction_f1"] for row in prediction_navigation_records]
    map_zone_prediction = [
        row["mean_decision_zone_prediction_f1"] for row in prediction_navigation_records
    ]
    map_navigation = [row["B_minus_A_mean_auc"] for row in prediction_navigation_records]
    seed_prediction = [float(row["B_prediction_f1"]) for row in paired]
    seed_navigation = [float(row["B_minus_A_auc"]) for row in paired]
    core_controls_matched = all(
        max(float(row[field]) for row in map_records)
        - min(float(row[field]) for row in map_records)
        < 1e-12
        for field in CORE_CONTROL_FIELDS
    )
    map3 = next(
        row for row in prediction_navigation_records
        if row["map_id"] == "irregular_workcell_91703"
    )
    map1 = next(
        row for row in prediction_navigation_records
        if row["map_id"] == "irregular_workcell_91701"
    )
    analysis = {
        "splits_used": list(SPLITS),
        "test_split_used": False,
        "training_rerun": False,
        "core_dynamic_controls_matched_across_maps": core_controls_matched,
        "map_level_prediction_f1_navigation_effect_pearson": pearson(
            map_prediction, map_navigation
        ),
        "map_level_decision_zone_f1_navigation_effect_pearson": pearson(
            map_zone_prediction, map_navigation
        ),
        "seed_level_prediction_f1_navigation_effect_pearson": pearson(
            seed_prediction, seed_navigation
        ),
        "map3_prediction_f1_not_below_map1": (
            map3["mean_prediction_f1"] >= map1["mean_prediction_f1"]
        ),
        "map3_decision_zone_f1_not_below_map1": (
            map3["mean_decision_zone_prediction_f1"]
            >= map1["mean_decision_zone_prediction_f1"]
        ),
        "supported_interpretation": (
            "prediction_is_learned_but_navigation_benefit_is_not_consistently_transferred"
        ),
        "unsupported_interpretation": (
            "map3_is_simpler_because_of_fewer_or_slower_obstacles"
        ),
        "paper_claim": (
            "conditional learning benefit on two maps with an explicit third-map boundary case; "
            "no robust universal generalization claim"
        ),
        "recommended_next_experiment_if_required": (
            "pre-registered dynamic-complexity sweep varying obstacle count, speed, and conflict "
            "frequency on a newly frozen validation set"
        ),
        "immediate_action": "write mechanism and limitation sections before any new training",
    }

    destination.mkdir(parents=True)
    write_records_csv(split_records, destination / "environment_statistics_by_split.csv")
    write_records_csv(map_records, destination / "environment_statistics_by_map.csv")
    write_records_csv(
        prediction_navigation_records,
        destination / "prediction_navigation_by_map.csv",
    )
    write_records_csv(paired, destination / "prediction_navigation_by_seed.csv")
    write_json(analysis, destination / "analysis.json")

    lines = [
        "# Dynamic prediction cross-map mechanism analysis",
        "",
        "## Scope",
        "",
        "This analysis uses the frozen train and validation scenario definitions plus existing validation curves. It does not use the test split and does not rerun training.",
        "",
        "## Environment controls",
        "",
        "| Map | Obstacles | Causal obstacles | Conflict progress | Risk positions / 100 steps | Speed | Route length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in map_records:
        lines.append(
            f"| {row['map_id']} | {row['mean_obstacle_count']:.2f} | "
            f"{row['mean_causal_obstacle_count']:.2f} | "
            f"{row['mean_first_collision_progress']:.3f} | "
            f"{row['risk_positions_per_100_reference_steps']:.3f} | "
            f"{row['mean_cells_per_step']:.2f} | {row['mean_route_length']:.2f} |"
        )
    lines.extend([
        "",
        "Obstacle-count distribution, causal-obstacle count, speed, designed conflict frequency, and early/middle/late allocation are matched by construction across the three maps. Therefore the existing evidence cannot support the claim that map03 is simply a lower-complexity dynamic environment.",
        "",
        "## Prediction and navigation",
        "",
        "| Map | Prediction F1 | Decision-zone F1 | B-A AUC | Improved seeds | A threshold | B threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in prediction_navigation_records:
        lines.append(
            f"| {row['map_id']} | {row['mean_prediction_f1']:.3f} | "
            f"{row['mean_decision_zone_prediction_f1']:.3f} | "
            f"{row['B_minus_A_mean_auc']:+.4f} | {row['B_better_seed_count']}/3 | "
            f"{row['A_mean_threshold_step']:.0f} | {row['B_mean_threshold_step']:.0f} |"
        )
    lines.extend([
        "",
        f"Map-level prediction-F1/effect correlation is {analysis['map_level_prediction_f1_navigation_effect_pearson']:.3f}; with only three maps this is descriptive, not inferential.",
        f"Across nine paired seeds, prediction-F1/effect correlation is {analysis['seed_level_prediction_f1_navigation_effect_pearson']:.3f}.",
        "Map03 prediction F1 and decision-zone F1 are both higher than map01, yet its navigation AUC effect is negative. This rejects a simple 'prediction head failed on map03' explanation.",
        "The evidence is most consistent with the prediction task being learned while its representation benefit is not reliably converted into improved Q-policy learning under every map geometry and conflict structure.",
        "",
        "## Paper interpretation",
        "",
        "- Do not claim universal cross-map improvement or that the method is proven to work only in more complex dynamic scenes.",
        "- Report positive learning-efficiency evidence on map01 and map02, with map03 as a formal boundary/negative case.",
        "- Frame the contribution as a conditional auxiliary-learning effect and explicitly discuss representation-to-control transfer limitations.",
        "- The current three maps do not vary obstacle count or speed independently, so complexity causality remains untested.",
        "",
        "## Next action",
        "",
        "Write the method, mechanism analysis, and limitation sections now. If one additional experiment is later required, pre-register a small dynamic-complexity sweep on a newly frozen validation set; do not tune prediction parameters on the current validation results.",
    ])
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
