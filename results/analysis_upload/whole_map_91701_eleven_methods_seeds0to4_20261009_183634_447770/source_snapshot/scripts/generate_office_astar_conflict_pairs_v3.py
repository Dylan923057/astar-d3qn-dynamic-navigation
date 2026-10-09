"""Generate five-obstacle, phase-paired A*-conflict scenarios for clean_v3.

Every pair keeps all route geometries, directions, speeds, and four context
obstacles fixed. Only the primary obstacle's start phase changes. The control
member keeps the registered static A* route safe and optimal; the conflict
member is oracle-verified to require a visible STAY action that is strictly
better than every no-wait solution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import (
    SPLITS,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv

import generate_office_behavior_scenarios_v9 as v9
import generate_office_behavior_scenarios_v10 as v10


DEFAULT_CONFIG = "configs/dynamic_office_clean_v3.yaml"
OFFSETS = {"train": 0, "validation": 10_000, "test": 20_000}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _shared_route_pools(source_manifest: Mapping[str, Any]) -> dict[str, dict]:
    if "route_pools" in source_manifest:
        source_records = source_manifest["route_pools"]["train"]["corridor"]
    elif "route_pool" in source_manifest:
        source_records = source_manifest["route_pool"]["corridor"]
    else:
        raise ValueError("Route-template source has no registered route pool.")
    pools: dict[str, dict] = {}
    for split in SPLITS:
        records = []
        for index, source in enumerate(source_records):
            record = copy.deepcopy(source)
            record["shared_template_source_route_id"] = str(source["route_id"])
            record["route_id"] = f"{split}_v3_shared_corridor_{index:02d}"
            records.append(record)
        pools[split] = {"corridor": records, "background": []}
    return pools


def _design(config: Mapping[str, Any]) -> dict[str, Any]:
    spatial = config["spatial_generalization"]
    values = dict(spatial["scenario_design"])
    values.update(spatial["paired_astar_conflict"])
    values["observation_radius"] = int(config["environment"]["window_size"]) // 2
    values["move_every_choices"] = tuple(
        int(value) for value in values.get("move_every_choices", (1,))
    )
    values["primary_obstacle_index"] = int(
        values.get("primary_obstacle_index", 0)
    )
    values["require_primary_demo_conflict"] = True
    values["require_strict_demo_conflict_reduction_in_pair"] = True
    values["_constructive_phase_pairs"] = True
    if values["primary_obstacle_index"] != 0:
        raise ValueError("clean_v3 requires the causal obstacle at index zero.")
    if values["move_every_choices"] != (1,):
        raise ValueError("clean_v3 preregisters move_every_choices=[1].")
    return values


def _pair_records(
    selected: Sequence[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    scenarios = []
    for pair_index, source in enumerate(
        sorted(selected, key=lambda row: (v9.DIFFICULTIES.index(row["difficulty_stratum"]), row["difficulty_score"]))
    ):
        conflict = copy.deepcopy(source)
        control = conflict.pop("_matched_control")
        conflict.pop("_matched_control_signature")
        pair_id = f"{split}_astar_wait_{pair_index:03d}"
        conflict_id = OFFSETS[split] + 2 * pair_index
        control_id = conflict_id + 1
        phase_delta = int(control["primary_start_index"]) - int(
            conflict["primary_start_index"]
        )
        conflict.update(
            scenario_id=conflict_id,
            pair_id=pair_id,
            pair_role="conflict",
            matched_behavior="wait",
            matched_conflict_difficulty=conflict["difficulty_stratum"],
            paired_scenario_id=control_id,
            pair_phase_delta=phase_delta,
        )
        control.update(
            scenario_id=control_id,
            difficulty_stratum="control",
            pair_id=pair_id,
            pair_role="matched_control",
            matched_behavior="wait",
            matched_conflict_difficulty=conflict["difficulty_stratum"],
            paired_scenario_id=conflict_id,
            pair_phase_delta=-phase_delta,
        )
        scenarios.extend((conflict, control))
    return scenarios


def _fingerprint(source: Mapping[str, Any], lookup: Mapping[str, Any]) -> tuple:
    return tuple(
        sorted(
            (
                tuple(tuple(cell) for cell in lookup[item["route_id"]]["route"]),
                int(item["start_index"]),
                int(item["direction"]),
                int(item["move_every"]),
            )
            for item in source["obstacles"]
        )
    )


def validate_manifest(problem, manifest: Mapping[str, Any], config) -> list[dict]:
    validate_spatial_scenario_manifest(problem, manifest)
    design = _design(config)
    pair_counts = {
        split: int(design["pair_counts"][split]) for split in SPLITS
    }
    all_fingerprints: dict[str, set[tuple]] = {}
    audit_rows = []
    nominal_steps = len(problem.nominal_path) - 1
    for split in SPLITS:
        sources = manifest["scenarios"][split]
        if len(sources) != 2 * pair_counts[split]:
            raise ValueError(f"{split} does not contain the registered pair count.")
        lookup = {
            record["route_id"]: record
            for record in manifest["route_pools"][split]["corridor"]
        }
        fingerprints = {_fingerprint(source, lookup) for source in sources}
        if len(fingerprints) != len(sources):
            raise ValueError(f"{split} contains duplicate complete scenarios.")
        all_fingerprints[split] = fingerprints
        groups: dict[str, list[dict]] = defaultdict(list)
        for source in sources:
            groups[str(source["pair_id"])].append(source)
        if len(groups) != pair_counts[split]:
            raise ValueError(f"{split} has an unexpected number of pair ids.")
        for pair_id, members in groups.items():
            if len(members) != 2:
                raise ValueError(f"Pair {pair_id} does not contain two members.")
            conflict = next(
                (row for row in members if row["pair_role"] == "conflict"), None
            )
            control = next(
                (
                    row
                    for row in members
                    if row["pair_role"] == "matched_control"
                ),
                None,
            )
            if conflict is None or control is None:
                raise ValueError(f"Pair {pair_id} lacks a conflict or control.")
            v10._validate_pair(conflict, control, design)
            conflict_scenario = v10._scenario_from_record(conflict, lookup)
            control_scenario = v10._scenario_from_record(control, lookup)
            verified_conflict = v9._classify_candidate(
                problem, conflict_scenario, "wait", design
            )
            verified_control = v9._classify_candidate(
                problem, control_scenario, "normal", design
            )
            if verified_conflict is None or verified_control is None:
                raise ValueError(f"Pair {pair_id} failed oracle revalidation.")
            if (
                verified_control["best_plan"].steps != nominal_steps
                or verified_control["best_plan"].wait_count != 0
                or not verified_conflict["nominal_path_collision"]
                or verified_conflict["best_plan"].wait_count < 1
                or (
                    verified_conflict["no_wait_steps"] is not None
                    and verified_conflict["no_wait_steps"]
                    <= verified_conflict["best_plan"].steps
                )
            ):
                raise ValueError(f"Pair {pair_id} is not action-discriminative.")
            audit_rows.append(
                {
                    "split": split,
                    "pair_id": pair_id,
                    "conflict_scenario_id": conflict["scenario_id"],
                    "control_scenario_id": control["scenario_id"],
                    "primary_route_id": conflict["primary_route_id"],
                    "conflict_phase": conflict["primary_start_index"],
                    "control_phase": control["primary_start_index"],
                    "direction": conflict["primary_direction"],
                    "move_every": conflict["primary_move_every"],
                    "difficulty": conflict["difficulty_stratum"],
                    "conflict_safe_steps": verified_conflict["best_plan"].steps,
                    "conflict_wait_count": verified_conflict["best_plan"].wait_count,
                    "conflict_no_wait_steps": verified_conflict["no_wait_steps"],
                    "control_safe_steps": verified_control["best_plan"].steps,
                    "control_wait_count": verified_control["best_plan"].wait_count,
                    "conflict_primary_demo_collisions": conflict[
                        "primary_reference_demo_collision_count"
                    ],
                    "control_primary_demo_collisions": control[
                        "primary_reference_demo_collision_count"
                    ],
                }
            )
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = all_fingerprints[left].intersection(all_fingerprints[right])
            if overlap:
                raise ValueError(
                    f"Complete-scenario leakage between {left} and {right}."
                )
    return audit_rows


def build_manifest(problem, config) -> tuple[dict, list[dict]]:
    spatial = config["spatial_generalization"]
    paired = spatial["paired_astar_conflict"]
    source_path = _resolve(str(paired["route_template_source_manifest"]))
    source_bytes = source_path.read_bytes()
    source_manifest = load_json(source_path)
    if "route_pools" in source_manifest:
        validate_spatial_scenario_manifest(problem, source_manifest)
    elif (
        source_manifest.get("map_id") != problem.map_id
        or source_manifest.get("grid_sha256") != problem.grid_sha256
    ):
        raise ValueError("Development route-template source does not match the map.")
    route_pools = _shared_route_pools(source_manifest)
    design = _design(config)
    demo_paths, demo_dataset = v9._reference_demonstrations(problem, config)
    scenarios: dict[str, list[dict]] = {}
    search_seed = int(spatial["behavior_search_seed"])
    for split_index, split in enumerate(SPLITS):
        target = int(design["pair_counts"][split])
        pool_target = max(2 * target, target + 6)
        rng = random.Random(search_seed + 10_000 * (split_index + 1))
        pool = v10._build_pool(
            problem,
            split,
            "wait",
            pool_target,
            route_pools[split]["corridor"],
            demo_paths,
            design,
            rng,
            set(),
            require_pair=True,
        )
        selected = v9._select_stratified(
            pool,
            target,
            design["difficulty_fractions"],
            random.Random(search_seed + 100_000 * (split_index + 1)),
            design.get("difficulty_score_targets", {}).get("wait"),
        )
        scenarios[split] = _pair_records(selected, split)
    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "office_clean_v3_phase_paired_astar_wait",
            "seed": int(spatial["generation_seed"]),
            "behavior_search_seed": search_seed,
            "route_length": int(spatial["route_length"]),
            "reference_path_count": int(spatial["reference_path_count"]),
            "reference_demo_episodes": int(config["demonstrations"]["episodes"]),
            "reference_demo_seed": int(config["demonstrations"]["seed"]),
            "reference_demo_dataset": demo_dataset,
            "separation_radius": int(spatial["separation_radius"]),
            "move_every": 1,
            "corridor_per_scenario": 5,
            "background_per_scenario": 0,
            "cross_split_spatial_disjoint": False,
            "cross_split_route_geometry_disjoint": False,
            "shared_route_templates_across_splits": True,
            "complete_scenarios_cross_split_disjoint": True,
            "pair_counts": {
                split: int(design["pair_counts"][split]) for split in SPLITS
            },
            "pair_invariant": (
                "only primary start_index changes; all five routes, directions, "
                "speeds, and four context-obstacle phases are fixed"
            ),
            "control_requirement": (
                "registered static A* route remains safe, optimal, and wait-free"
            ),
            "conflict_requirement": (
                "registered A* route collides; visible STAY is strictly better "
                "than every no-wait solution"
            ),
            "route_template_source_manifest": str(source_path),
            "route_template_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "route_pools": route_pools,
        "scenarios": scenarios,
    }
    audit_rows = validate_manifest(problem, manifest, config)
    return manifest, audit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--audit-output", default="outputs/office_clean_v3_dataset_design"
    )
    args = parser.parse_args()
    config = load_config(_resolve(args.config))
    problem = v9._load_problem(config)
    manifest, audit_rows = build_manifest(problem, config)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(spatial["manifest"])
    write_json(manifest, manifest_path)
    v9.render_route_pools(problem, manifest, _resolve(spatial["route_pool_preview"]))
    for split, key in (
        ("train", "training_preview"),
        ("validation", "validation_preview"),
        ("test", "test_preview"),
    ):
        v9.render_gallery(problem, manifest, split, _resolve(spatial[key]))
    audit_dir = _resolve(args.audit_output)
    write_records_csv(audit_rows, audit_dir / "phase_pair_audit.csv")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(
        {
            "status": "passed",
            "manifest": str(manifest_path),
            "manifest_sha256": digest,
            "pair_counts": manifest["generation"]["pair_counts"],
            "complete_scenarios_cross_split_disjoint": True,
            "shared_route_templates_across_splits": True,
            "phase_pair_audit": str(audit_dir / "phase_pair_audit.csv"),
        },
        audit_dir / "protocol_audit.json",
    )
    print(
        f"Generated clean_v3 paired benchmark: sha256={digest[:12]} "
        f"pairs={manifest['generation']['pair_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
