"""Generate a minimal phase-paired dynamic-obstacle benchmark.

Each pair uses exactly one route, direction, and speed.  The two scenes differ
only in the obstacle start phase: the control phase does not collide with the
registered nominal A* rollout, while the conflict phase does.  Both phases must
remain solvable under the environment's exact move-before-agent semantics.

The route geometry is reused from a previously frozen spatially-disjoint v4
manifest.  This avoids another subjective obstacle-placement search and keeps
train/validation/test route geometry separated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import (
    SPLITS,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.evaluation.conflict import (
    obstacle_conflict_metrics,
    scenario_safe_path_metrics,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv
from generate_spatial_dynamic_scenarios import _draw_base, render_route_pool_overview


DEFAULT_CONFIG = "configs/dynamic_spatial_generalization_office_causal_paired_v1.yaml"
CONDITION_COLORS = {"control": "#2e7d32", "conflict": "#c62828"}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_problem(config: Mapping[str, Any]):
    map_set = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(map_set["file"])))
    if len(problems) != int(map_set["count"]):
        raise ValueError("Configured map count does not match maps.json.")
    matches = [problem for problem in problems if problem.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise ValueError("Configured scene is not uniquely registered.")
    return matches[0]


def _spec(record: Mapping[str, Any], start_index: int, direction: int, move_every: int):
    return DynamicObstacleSpec(
        route=tuple(tuple(int(value) for value in cell) for cell in record["route"]),
        start_index=int(start_index),
        direction=int(direction),
        move_every=int(move_every),
        label=str(record["route_id"]),
        reference_path_source="frozen_v4_direct_corridor",
    )


def _phase_options(problem, record: Mapping[str, Any], move_every: int):
    options: dict[int, list[dict[str, Any]]] = {-1: [], 1: []}
    for direction in (-1, 1):
        for start_index in range(len(record["route"])):
            spec = _spec(record, start_index, direction, move_every)
            conflict = obstacle_conflict_metrics(problem, spec, temporal_window=2)
            safe = scenario_safe_path_metrics(
                problem,
                DynamicScenario(seed=0, obstacles=(spec,)),
            )
            if safe["minimum_safe_path_steps"] is None:
                continue
            options[direction].append(
                {"spec": spec, "conflict": conflict, "safe": safe}
            )
    return options


def _ordered_pairs(
    problem,
    record: Mapping[str, Any],
    move_every: int,
    pairs_per_route: int,
    rng: random.Random,
):
    by_direction = _phase_options(problem, record, move_every)
    direction_pairs: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for direction in (-1, 1):
        controls = [
            item
            for item in by_direction[direction]
            if int(item["conflict"]["exact_temporal_conflict_count"]) == 0
        ]
        conflicts = [
            item
            for item in by_direction[direction]
            if int(item["conflict"]["exact_temporal_conflict_count"]) > 0
        ]
        controls.sort(
            key=lambda item: (
                int(item["conflict"]["aligned_temporal_conflict_count"]),
                -int(item["conflict"]["minimum_phase_offset"] or 0),
                int(item["spec"].start_index),
            )
        )
        conflicts.sort(
            key=lambda item: (
                -int(item["conflict"]["exact_temporal_conflict_count"]),
                int(item["spec"].start_index),
            )
        )
        # Randomization only changes which equally valid phase is paired.  It is
        # seeded and never changes route geometry or acceptance requirements.
        rng.shuffle(controls)
        rng.shuffle(conflicts)
        direction_pairs[direction] = list(zip(controls, conflicts))

    selected = []
    while len(selected) < pairs_per_route:
        added = False
        for direction in (-1, 1):
            if direction_pairs[direction]:
                selected.append(direction_pairs[direction].pop(0))
                added = True
                if len(selected) == pairs_per_route:
                    break
        if not added:
            break
    if not selected:
        raise ValueError(
            f"Route {record['route_id']!r} has no same-direction conflict/control phase pair."
        )
    return selected


def _direct_route_pools(source: Mapping[str, Any]):
    pools = {}
    for split in SPLITS:
        direct = [
            copy.deepcopy(record)
            for record in source["route_pools"][split]["corridor"]
            if int(record.get("nominal_intersection_cell_count", 0)) > 0
        ]
        if not direct:
            raise ValueError(f"Source manifest has no direct route in {split!r}.")
        pools[split] = {"corridor": direct, "background": []}
    return pools


def build_causal_paired_manifest(problem, config: Mapping[str, Any]):
    spatial = config["spatial_generalization"]
    paired = spatial["causal_paired"]
    source_path = _resolve(str(paired["source_manifest"]))
    source_bytes = source_path.read_bytes()
    source = load_json(source_path)
    validate_spatial_scenario_manifest(problem, source)
    pools = _direct_route_pools(source)
    move_every = int(spatial["move_every"])
    pairs_per_route = int(paired["pairs_per_route"])
    if pairs_per_route <= 0:
        raise ValueError("pairs_per_route must be positive.")
    offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    scenarios: dict[str, list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    generation_seed = int(spatial["generation_seed"])

    for split_index, split in enumerate(SPLITS):
        rng = random.Random(generation_seed + 10_000 * (split_index + 1))
        split_scenarios = []
        pair_number = 0
        for record in pools[split]["corridor"]:
            for control, conflict in _ordered_pairs(
                problem,
                record,
                move_every,
                pairs_per_route,
                rng,
            ):
                pair_id = f"{split}_pair_{pair_number:03d}"
                scenario_ids = {}
                for condition, item in (("control", control), ("conflict", conflict)):
                    scenario_id = offsets[split] + len(split_scenarios)
                    scenario_ids[condition] = scenario_id
                    spec = item["spec"]
                    split_scenarios.append(
                        {
                            "scenario_id": scenario_id,
                            "pair_id": pair_id,
                            "pair_condition": condition,
                            "required_behavior": condition,
                            "difficulty_stratum": condition,
                            "minimum_safe_path_steps": int(
                                item["safe"]["minimum_safe_path_steps"]
                            ),
                            "safe_detour_steps": int(item["safe"]["safe_detour_steps"]),
                            "exact_temporal_conflict_count": int(
                                item["conflict"]["exact_temporal_conflict_count"]
                            ),
                            "exact_temporal_conflict_steps": list(
                                item["conflict"]["exact_temporal_conflict_steps"]
                            ),
                            "obstacles": [
                                {
                                    "route_id": record["route_id"],
                                    "start_index": int(spec.start_index),
                                    "direction": int(spec.direction),
                                    "move_every": int(spec.move_every),
                                }
                            ],
                        }
                    )
                audit_rows.append(
                    {
                        "map_id": problem.map_id,
                        "split": split,
                        "pair_id": pair_id,
                        "route_id": record["route_id"],
                        "route_center": str(tuple(record["center"])),
                        "corridor_frequency": float(record["corridor_frequency"]),
                        "direction": int(control["spec"].direction),
                        "control_scenario_id": scenario_ids["control"],
                        "control_start_index": int(control["spec"].start_index),
                        "control_exact_conflicts": int(
                            control["conflict"]["exact_temporal_conflict_count"]
                        ),
                        "control_safe_steps": int(
                            control["safe"]["minimum_safe_path_steps"]
                        ),
                        "conflict_scenario_id": scenario_ids["conflict"],
                        "conflict_start_index": int(conflict["spec"].start_index),
                        "conflict_exact_conflicts": int(
                            conflict["conflict"]["exact_temporal_conflict_count"]
                        ),
                        "conflict_steps": str(
                            conflict["conflict"]["exact_temporal_conflict_steps"]
                        ),
                        "conflict_safe_steps": int(
                            conflict["safe"]["minimum_safe_path_steps"]
                        ),
                    }
                )
                pair_number += 1
        scenarios[split] = split_scenarios
        expected = int(spatial["scenario_counts"][split])
        if len(split_scenarios) != expected:
            raise ValueError(
                f"Generated {len(split_scenarios)} {split} scenarios; config expects {expected}."
            )
        expected_routes = int(spatial["route_pool_counts"][split]["corridor"])
        if len(pools[split]["corridor"]) != expected_routes:
            raise ValueError(
                f"Selected {len(pools[split]['corridor'])} {split} routes; "
                f"config expects {expected_routes}."
            )

    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "causal_phase_paired_single_obstacle_v1",
            "seed": generation_seed,
            "source_manifest": str(source_path),
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "route_length": int(source["generation"]["route_length"]),
            "reference_path_count": int(source["generation"]["reference_path_count"]),
            "separation_radius": int(source["generation"]["separation_radius"]),
            "cross_split_spatial_disjoint": bool(
                source["generation"].get("cross_split_spatial_disjoint", True)
            ),
            "move_every": move_every,
            "corridor_per_scenario": 1,
            "background_per_scenario": 0,
            "pairs_per_route_cap": pairs_per_route,
            "pair_invariant": "same route, direction and speed; start_index only",
            "control_requirement": "zero exact nominal A* temporal conflicts",
            "conflict_requirement": "at least one exact nominal A* temporal conflict",
            "oracle": "periodic_time_expanded_bfs_with_wait_action",
        },
        "route_pools": pools,
        "scenarios": scenarios,
    }
    validate_spatial_scenario_manifest(problem, manifest)
    return manifest, audit_rows


def render_pair_gallery(problem, manifest: Mapping[str, Any], output: Path) -> None:
    split = "test"
    lookup = {
        str(record["route_id"]): record
        for record in manifest["route_pools"][split]["corridor"]
    }
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for scenario in manifest["scenarios"][split]:
        groups[str(scenario["pair_id"])].append(scenario)
    pair_ids = sorted(groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(pair_ids), 2, figsize=(9, max(3, 3.2 * len(pair_ids))), dpi=160
    )
    axes = np.asarray(axes).reshape(len(pair_ids), 2)
    for row, pair_id in enumerate(pair_ids):
        members = {str(item["pair_condition"]): item for item in groups[pair_id]}
        for column, condition in enumerate(("control", "conflict")):
            scenario = members[condition]
            axis = axes[row, column]
            _draw_base(
                axis,
                problem,
                title=(
                    f"{pair_id} | {condition}\n"
                    f"exact={scenario['exact_temporal_conflict_count']} "
                    f"safe={scenario['minimum_safe_path_steps']}"
                ),
            )
            obstacle = scenario["obstacles"][0]
            route = lookup[str(obstacle["route_id"])]["route"]
            axis.plot(
                [cell[1] for cell in route],
                [cell[0] for cell in route],
                color=CONDITION_COLORS[condition],
                linewidth=2.4,
            )
            initial = route[int(obstacle["start_index"])]
            axis.scatter(
                initial[1], initial[0], c=CONDITION_COLORS[condition], s=30, zorder=6
            )
    figure.suptitle(
        f"{problem.map_id}: causal test pairs (only obstacle phase changes)",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = load_config(config_path)
    problem = _load_problem(config)
    manifest, audit_rows = build_causal_paired_manifest(problem, config)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(str(spatial["manifest"]))
    write_json(manifest, manifest_path)
    render_route_pool_overview(
        problem, manifest, _resolve(str(spatial["route_pool_preview"]))
    )
    render_pair_gallery(
        problem, manifest, _resolve(str(spatial["validation_preview"]))
    )
    audit_dir = _resolve(str(spatial["causal_paired"]["audit_output"]))
    write_records_csv(audit_rows, audit_dir / "phase_pairs.csv")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(
        {
            "status": "complete",
            "training_ready": True,
            "config": str(config_path),
            "manifest": str(manifest_path),
            "manifest_sha256": digest,
            "scenario_counts": {
                split: len(manifest["scenarios"][split]) for split in SPLITS
            },
            "pair_counts": {
                split: len(manifest["scenarios"][split]) // 2 for split in SPLITS
            },
            "audit": str(audit_dir / "phase_pairs.csv"),
        },
        audit_dir / "generation_status.json",
    )
    print(
        f"Generated causal paired benchmark: sha256={digest[:12]} "
        f"pairs=train:{len(manifest['scenarios']['train']) // 2},"
        f"validation:{len(manifest['scenarios']['validation']) // 2},"
        f"test:{len(manifest['scenarios']['test']) // 2}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Audit: {audit_dir}", flush=True)


if __name__ == "__main__":
    main()

