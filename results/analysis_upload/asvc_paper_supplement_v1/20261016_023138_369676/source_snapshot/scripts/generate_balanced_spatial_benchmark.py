"""Generate spatially disjoint scenarios with matched conflict strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import (
    SPLITS,
    _corridor_frequency,
    _enumerate_routes,
    _pack_routes,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.evaluation.conflict import (
    obstacle_conflict_metrics,
    route_record_lookup,
    scenario_conflict_metrics,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_problem_layout, render_problem_overview
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv

from generate_spatial_dynamic_scenarios import (
    render_route_pool_overview,
    render_validation_gallery,
)


DEFAULT_CONFIGS = (
    "configs/dynamic_spatial_generalization_office_balanced_v3.yaml",
    "configs/dynamic_spatial_generalization_parcel_balanced_v3.yaml",
    "configs/dynamic_spatial_generalization_warehouse_balanced_v3.yaml",
)


def _resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def _load_problem(config: Mapping[str, Any]):
    map_set = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(map_set["file"])))
    if len(problems) != int(map_set["count"]):
        raise ValueError("Configured map count does not match maps.json.")
    matches = [item for item in problems if item.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise ValueError("Configured scene is not uniquely registered.")
    return matches[0]


def _even_sample(
    records: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count < 0 or count > len(records):
        raise ValueError("Requested balanced sample is unavailable.")
    if count == 0:
        return []
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item["corridor_score"]),
            int(item["center"][0]),
            int(item["center"][1]),
            str(item["orientation"]),
        ),
    )
    if count == len(ordered):
        return ordered
    # Even rank sampling preserves the full score range instead of selecting only
    # the easiest or hardest routes.
    positions = [min(len(ordered) - 1, int((index + 0.5) * len(ordered) / count)) for index in range(count)]
    return [ordered[position] for position in positions]


def _proportional_quotas(
    total: int,
    targets: Mapping[str, int],
    minimum: int = 0,
) -> dict[str, int]:
    target_total = sum(int(targets[split]) for split in SPLITS)
    raw = {split: total * int(targets[split]) / target_total for split in SPLITS}
    quotas = {split: max(minimum, math.floor(raw[split])) for split in SPLITS}
    if sum(quotas.values()) > total:
        raise ValueError("The minimum per-split quota exceeds available routes.")
    while sum(quotas.values()) < total:
        eligible = [
            split
            for split in SPLITS
            if quotas[split] < int(targets[split])
        ]
        split = max(
            eligible,
            key=lambda name: (raw[name] - quotas[name], int(targets[name]), -SPLITS.index(name)),
        )
        quotas[split] += 1
    return quotas


def _allocate_records(
    records: Sequence[dict[str, Any]],
    quotas: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    result = {split: [] for split in SPLITS}
    ordered = list(records)
    cursor = 0
    while cursor < len(ordered):
        eligible = [split for split in SPLITS if len(result[split]) < int(quotas[split])]
        if not eligible:
            break
        split = min(
            eligible,
            key=lambda name: (
                len(result[name]) / max(1, int(quotas[name])),
                SPLITS.index(name),
            ),
        )
        result[split].append(ordered[cursor])
        cursor += 1
    if cursor != len(ordered) or any(
        len(result[split]) != int(quotas[split]) for split in SPLITS
    ):
        raise RuntimeError("Balanced route allocation failed.")
    return result


def _select_direct_routes(
    records: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Prefer phase-controllable routes with fewer repeated path intersections."""

    selected: list[dict[str, Any]] = []
    maximum_values = sorted({int(item["_maximum_exact_conflicts"]) for item in records})
    for maximum in maximum_values:
        group = [
            item for item in records if int(item["_maximum_exact_conflicts"]) == maximum
        ]
        remaining = count - len(selected)
        if remaining <= 0:
            break
        selected.extend(
            group if len(group) <= remaining else _even_sample(group, remaining)
        )
    if len(selected) != count:
        raise ValueError("Not enough phase-controllable direct routes were selected.")
    return selected


def _allocate_balanced_direct_routes(
    records: Sequence[dict[str, Any]],
    quotas: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Minimize split differences in maximum temporal-conflict potential."""

    indexed = tuple(records)
    train_count = int(quotas["train"])
    validation_count = int(quotas["validation"])
    best: tuple[tuple[float, ...], dict[str, list[dict[str, Any]]]] | None = None
    indices = tuple(range(len(indexed)))
    import itertools

    for train_indices in itertools.combinations(indices, train_count):
        train_set = set(train_indices)
        remaining = tuple(index for index in indices if index not in train_set)
        for validation_indices in itertools.combinations(remaining, validation_count):
            validation_set = set(validation_indices)
            test_indices = tuple(
                index for index in remaining if index not in validation_set
            )
            groups = {
                "train": [indexed[index] for index in train_indices],
                "validation": [indexed[index] for index in validation_indices],
                "test": [indexed[index] for index in test_indices],
            }
            maximum_means = {
                split: sum(float(item["_maximum_exact_conflicts"]) for item in group)
                / len(group)
                for split, group in groups.items()
            }
            aligned_means = {
                split: sum(
                    float(item["_minimum_nonexact_aligned_conflicts"])
                    for item in group
                )
                / len(group)
                for split, group in groups.items()
            }
            frequency_means = {
                split: sum(float(item["corridor_score"]) for item in group) / len(group)
                for split, group in groups.items()
            }
            objective = (
                max(maximum_means.values()) - min(maximum_means.values()),
                max(aligned_means.values()) - min(aligned_means.values()),
                max(frequency_means.values()) - min(frequency_means.values()),
                sum(
                    abs(value - sum(maximum_means.values()) / len(SPLITS))
                    for value in maximum_means.values()
                ),
                tuple(
                    tuple(tuple(item["center"]) for item in groups[split])
                    for split in SPLITS
                ),
            )
            if best is None or objective < best[0]:
                best = (objective, groups)
    if best is None:
        raise RuntimeError("Could not allocate balanced direct routes.")
    return best[1]


def _phase_conflict_range(
    problem,
    record: Mapping[str, Any],
    move_every: int,
) -> tuple[int, int, int]:
    route = tuple(tuple(cell) for cell in record["route"])
    counts = []
    nonexact_aligned = []
    for start_index in range(len(route)):
        for direction in (-1, 1):
            spec = DynamicObstacleSpec(
                route=route,
                start_index=start_index,
                direction=direction,
                move_every=move_every,
            )
            metrics = obstacle_conflict_metrics(problem, spec, temporal_window=2)
            exact = int(metrics["exact_temporal_conflict_count"])
            aligned = int(metrics["aligned_temporal_conflict_count"])
            counts.append(exact)
            if exact == 0 and aligned > 0:
                nonexact_aligned.append(aligned)
    return min(counts), max(counts), min(nonexact_aligned or (0,))


def _balanced_route_pools(
    problem,
    packed: Sequence[dict[str, Any]],
    route_pool_counts: Mapping[str, Mapping[str, int]],
    direct_fraction: float,
    minimum_direct: int,
    reference_path_count: int,
    move_every: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    corridor_targets = {
        split: int(route_pool_counts[split]["corridor"]) for split in SPLITS
    }
    background_targets = {
        split: int(route_pool_counts[split]["background"]) for split in SPLITS
    }
    required_corridor = sum(corridor_targets.values())
    required_background = sum(background_targets.values())
    corridor = [item for item in packed if item["category"] == "corridor"]
    background = [
        item
        for item in packed
        if item["category"] == "background"
        and not set(item["route"]).intersection(problem.nominal_path)
    ]
    direct = []
    for item in corridor:
        if not set(item["route"]).intersection(problem.nominal_path):
            continue
        minimum_exact, maximum_exact, minimum_nonexact_aligned = _phase_conflict_range(
            problem, item, move_every
        )
        if minimum_exact == 0 and maximum_exact > 0:
            direct.append(
                {
                    **item,
                    "_minimum_exact_conflicts": minimum_exact,
                    "_maximum_exact_conflicts": maximum_exact,
                    "_minimum_nonexact_aligned_conflicts": minimum_nonexact_aligned,
                }
            )
    indirect = [
        item
        for item in corridor
        if not set(item["route"]).intersection(problem.nominal_path)
    ]
    desired_direct = max(
        minimum_direct * len(SPLITS),
        round(required_corridor * direct_fraction),
    )
    if len(direct) < desired_direct:
        raise ValueError(
            f"{problem.map_id} has only {len(direct)} spatially separated, phase-"
            "controllable direct routes; "
            f"{desired_direct} are required."
        )
    if len(indirect) < required_corridor - desired_direct:
        raise ValueError(f"{problem.map_id} has too few indirect corridor routes.")
    if len(background) < required_background:
        raise ValueError(
            f"{problem.map_id} has too few background routes that are also "
            "disjoint from the registered nominal A* path."
        )

    selected_direct = _select_direct_routes(direct, desired_direct)
    selected_indirect = _even_sample(indirect, required_corridor - desired_direct)
    selected_background = _even_sample(background, required_background)
    direct_quotas = _proportional_quotas(
        desired_direct, corridor_targets, minimum=minimum_direct
    )
    indirect_quotas = {
        split: corridor_targets[split] - direct_quotas[split] for split in SPLITS
    }
    direct_split = _allocate_balanced_direct_routes(selected_direct, direct_quotas)
    indirect_split = _allocate_records(selected_indirect, indirect_quotas)
    background_split = _allocate_records(selected_background, background_targets)

    route_pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in SPLITS:
        route_pools[split] = {"corridor": [], "background": []}
        combined_corridor = direct_split[split] + indirect_split[split]
        combined_corridor.sort(
            key=lambda item: (-int(bool(set(item["route"]).intersection(problem.nominal_path))), -float(item["corridor_score"]))
        )
        for category, records in (
            ("corridor", combined_corridor),
            ("background", background_split[split]),
        ):
            for index, record in enumerate(records):
                intersections = sorted(set(record["route"]).intersection(problem.nominal_path))
                minimum_distance = min(
                    abs(route_cell[0] - path_cell[0]) + abs(route_cell[1] - path_cell[1])
                    for route_cell in record["route"]
                    for path_cell in problem.nominal_path
                )
                route_pools[split][category].append(
                    {
                        "route_id": f"{split}_{category}_{index:02d}",
                        "category": category,
                        "center": list(record["center"]),
                        "orientation": record["orientation"],
                        "route": [list(cell) for cell in record["route"]],
                        "corridor_score": int(record["corridor_score"]),
                        "corridor_frequency": float(record["corridor_score"]) / reference_path_count,
                        "nominal_intersection_cell_count": len(intersections),
                        "nominal_intersection_cells": [list(cell) for cell in intersections],
                        "minimum_nominal_distance": minimum_distance,
                        "minimum_exact_conflicts_over_phases": int(
                            record.get("_minimum_exact_conflicts", 0)
                        ),
                        "maximum_exact_conflicts_over_phases": int(
                            record.get("_maximum_exact_conflicts", 0)
                        ),
                        "minimum_nonexact_aligned_conflicts_over_phases": int(
                            record.get("_minimum_nonexact_aligned_conflicts", 0)
                        ),
                    }
                )
    return route_pools


def _phase_options(problem, record: Mapping[str, Any], move_every: int):
    route = tuple(tuple(cell) for cell in record["route"])
    options = []
    for start_index in range(len(route)):
        for direction in (-1, 1):
            spec = DynamicObstacleSpec(
                route=route,
                start_index=start_index,
                direction=direction,
                move_every=move_every,
                label=str(record["route_id"]),
                reference_path_source=str(record["category"]),
            )
            metrics = obstacle_conflict_metrics(problem, spec, temporal_window=2)
            options.append((spec, metrics))
    return options


def _phase_key(item) -> tuple[int, int, int]:
    _, metrics = item
    offset = metrics["minimum_phase_offset"]
    return (
        int(metrics["exact_temporal_conflict_count"]),
        int(metrics["aligned_temporal_conflict_count"]),
        -(int(offset) if offset is not None else 10_000),
    )


def _choose_phase(options, target: str, rng: random.Random) -> DynamicObstacleSpec:
    ordered = sorted(options, key=_phase_key)
    if target == "low":
        best = _phase_key(ordered[0])
        candidates = [item for item in ordered if _phase_key(item) == best]
        return rng.choice(candidates)[0]
    if target == "high":
        best = _phase_key(ordered[-1])
        candidates = [item for item in ordered if _phase_key(item) == best]
        return rng.choice(candidates)[0]
    if target == "medium":
        non_exact_aligned = [
            item
            for item in ordered
            if int(item[1]["exact_temporal_conflict_count"]) == 0
            and int(item[1]["aligned_temporal_conflict_count"]) > 0
        ]
        if non_exact_aligned:
            minimum_aligned = min(
                int(item[1]["aligned_temporal_conflict_count"])
                for item in non_exact_aligned
            )
            candidates = [
                item
                for item in non_exact_aligned
                if int(item[1]["aligned_temporal_conflict_count"])
                == minimum_aligned
            ]
            return rng.choice(candidates)[0]
        return ordered[len(ordered) // 2][0]
    if target == "random":
        return rng.choice(ordered)[0]
    raise ValueError(f"Unknown phase target {target!r}.")


def _stratum_counts(
    scenario_count: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    names = ("low", "medium", "high")
    raw = {name: scenario_count * float(fractions[name]) for name in names}
    result = {name: math.floor(raw[name]) for name in names}
    while sum(result.values()) < scenario_count:
        name = max(names, key=lambda item: (raw[item] - result[item], -names.index(item)))
        result[name] += 1
    return result


def _balanced_high_direct_sample(
    records: Sequence[Mapping[str, Any]],
    count: int,
    created: int,
    rng: random.Random,
) -> list[Mapping[str, Any]]:
    """Alternate below/above-target route pairs to match conflict potential."""

    import itertools

    combinations = list(itertools.combinations(records, count))
    target = count * sum(
        float(item["maximum_exact_conflicts_over_phases"]) for item in records
    ) / len(records)
    with_values = [
        (
            sum(float(item["maximum_exact_conflicts_over_phases"]) for item in combo),
            combo,
        )
        for combo in combinations
    ]
    exact = [combo for value, combo in with_values if abs(value - target) < 1.0e-9]
    if exact:
        return list(rng.choice(exact))
    lower_value = max(value for value, _ in with_values if value < target)
    upper_value = min(value for value, _ in with_values if value > target)
    candidates = [
        combo
        for value, combo in with_values
        if abs(value - (lower_value if created % 2 == 0 else upper_value)) < 1.0e-9
    ]
    return list(rng.choice(candidates))


def _balanced_medium_direct_sample(
    records: Sequence[Mapping[str, Any]],
    created: int,
) -> list[Mapping[str, Any]]:
    minimum_aligned = min(
        int(item["minimum_nonexact_aligned_conflicts_over_phases"])
        for item in records
    )
    eligible = [
        item
        for item in records
        if int(item["minimum_nonexact_aligned_conflicts_over_phases"])
        == minimum_aligned
    ]
    ordered = sorted(
        eligible,
        key=lambda item: (
            int(item["minimum_nonexact_aligned_conflicts_over_phases"]),
            str(item["route_id"]),
        ),
    )
    return [ordered[created % len(ordered)]]


def _build_balanced_scenarios(
    problem,
    route_pools: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    scenario_counts: Mapping[str, int],
    corridor_per_scenario: int,
    background_per_scenario: int,
    move_every: int,
    balance: Mapping[str, Any],
    generation_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if corridor_per_scenario != 3 or background_per_scenario != 2:
        raise ValueError("Balanced v3 currently requires three corridor and two background routes.")
    fractions = balance["scenario_strata"]
    direct_targets = {
        name: int(value)
        for name, value in balance["direct_routes_per_scenario"].items()
    }
    offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    result: dict[str, list[dict[str, Any]]] = {}
    for split_index, split in enumerate(SPLITS):
        rng = random.Random(generation_seed + (split_index + 1) * 10_000)
        lookup = {
            str(record["route_id"]): record
            for category in ("corridor", "background")
            for record in route_pools[split][category]
        }
        direct = [
            record for record in route_pools[split]["corridor"]
            if int(record["nominal_intersection_cell_count"]) > 0
        ]
        indirect = [
            record for record in route_pools[split]["corridor"]
            if int(record["nominal_intersection_cell_count"]) == 0
        ]
        background = list(route_pools[split]["background"])
        phase_options = {
            str(record["route_id"]): _phase_options(problem, record, move_every)
            for record in lookup.values()
        }
        target_counts = _stratum_counts(int(scenario_counts[split]), fractions)
        scenarios: list[dict[str, Any]] = []
        combination_counts: dict[tuple[str, ...], int] = {}
        used_configurations: set[tuple[tuple[str, int, int], ...]] = set()
        for stratum in ("low", "medium", "high"):
            direct_count = direct_targets[stratum]
            indirect_count = corridor_per_scenario - direct_count
            if len(direct) < direct_count or len(indirect) < indirect_count:
                raise ValueError(f"{problem.map_id} {split} cannot build {stratum} scenarios.")
            attempts = 0
            created = 0
            while created < target_counts[stratum]:
                attempts += 1
                if attempts > target_counts[stratum] * 20_000:
                    raise RuntimeError(f"Could not generate unique {split} {stratum} scenarios.")
                selected_direct = (
                    _balanced_high_direct_sample(
                        direct, direct_count, created, rng
                    )
                    if stratum == "high"
                    else _balanced_medium_direct_sample(direct, created)
                    if stratum == "medium"
                    else rng.sample(direct, direct_count)
                )
                selected = selected_direct + rng.sample(indirect, indirect_count)
                selected += rng.sample(background, background_per_scenario)
                combination = tuple(sorted(str(item["route_id"]) for item in selected))
                maximum_per_combination = int(
                    balance.get("maximum_scenarios_per_route_combination", 1)
                )
                if combination_counts.get(combination, 0) >= maximum_per_combination:
                    continue
                obstacles = []
                for record in selected:
                    is_direct = int(record.get("nominal_intersection_cell_count", 0)) > 0
                    phase_target = (
                        "low"
                        if stratum == "low"
                        else "medium"
                        if stratum == "medium" and is_direct
                        else "high"
                        if stratum == "high" and is_direct
                        else "random"
                    )
                    spec = _choose_phase(
                        phase_options[str(record["route_id"])], phase_target, rng
                    )
                    obstacles.append(
                        {
                            "route_id": record["route_id"],
                            "start_index": spec.start_index,
                            "direction": spec.direction,
                            "move_every": spec.move_every,
                        }
                    )
                configuration = tuple(
                    sorted(
                        (
                            str(item["route_id"]),
                            int(item["start_index"]),
                            int(item["direction"]),
                        )
                        for item in obstacles
                    )
                )
                if configuration in used_configurations:
                    continue
                used_configurations.add(configuration)
                combination_counts[combination] = (
                    combination_counts.get(combination, 0) + 1
                )
                scenarios.append(
                    {
                        "scenario_id": offsets[split] + len(scenarios),
                        "difficulty_stratum": stratum,
                        "obstacles": obstacles,
                    }
                )
                created += 1
        result[split] = scenarios
    return result


def build_balanced_manifest(problem, config: Mapping[str, Any]) -> dict[str, Any]:
    spatial = config["spatial_generalization"]
    balance = spatial["conflict_balance"]
    generation_seed = int(spatial["generation_seed"])
    route_length = int(spatial["route_length"])
    reference_path_count = int(spatial["reference_path_count"])
    separation_radius = int(spatial["separation_radius"])
    candidates = _enumerate_routes(problem, route_length)
    frequency = _corridor_frequency(
        problem, reference_path_count, generation_seed + 100_000
    )
    packed = _pack_routes(candidates, frequency, separation_radius, generation_seed)
    route_pools = _balanced_route_pools(
        problem,
        packed,
        spatial["route_pool_counts"],
        float(balance["direct_route_fraction"]),
        int(balance["minimum_direct_routes_per_split"]),
        reference_path_count,
        int(spatial["move_every"]),
    )
    scenarios = _build_balanced_scenarios(
        problem,
        route_pools,
        spatial["scenario_counts"],
        int(spatial["corridor_per_scenario"]),
        int(spatial["background_per_scenario"]),
        int(spatial["move_every"]),
        balance,
        generation_seed,
    )
    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "conflict_balanced_spatial_v3",
            "seed": generation_seed,
            "route_length": route_length,
            "reference_path_count": reference_path_count,
            "separation_radius": separation_radius,
            "move_every": int(spatial["move_every"]),
            "corridor_per_scenario": int(spatial["corridor_per_scenario"]),
            "background_per_scenario": int(spatial["background_per_scenario"]),
            "legal_route_count": len(candidates),
            "packed_route_count": len(packed),
            "conflict_balance": dict(balance),
        },
        "route_pools": route_pools,
        "scenarios": scenarios,
    }
    validate_spatial_scenario_manifest(problem, manifest)
    return manifest


def _audit(problem, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summaries = []
    for split in SPLITS:
        lookup = route_record_lookup(manifest, split)
        scenario_sources = manifest["scenarios"][split]
        for source in scenario_sources:
            specs = []
            for item in source["obstacles"]:
                record = lookup[str(item["route_id"])]
                specs.append(
                    DynamicObstacleSpec(
                        route=tuple(tuple(cell) for cell in record["route"]),
                        start_index=int(item["start_index"]),
                        direction=int(item["direction"]),
                        move_every=int(item["move_every"]),
                        label=str(item["route_id"]),
                    )
                )
            metrics = scenario_conflict_metrics(
                problem,
                DynamicScenario(seed=int(source["scenario_id"]), obstacles=tuple(specs)),
                temporal_window=2,
                route_records=lookup,
            )
            rows.append(
                {
                    "map_id": problem.map_id,
                    "split": split,
                    "difficulty_stratum": source["difficulty_stratum"],
                    **{key: value for key, value in metrics.items() if key != "obstacle_metrics"},
                }
            )
    for split in SPLITS:
        for stratum in ("low", "medium", "high"):
            group = [row for row in rows if row["split"] == split and row["difficulty_stratum"] == stratum]
            summaries.append(
                {
                    "map_id": problem.map_id,
                    "split": split,
                    "difficulty_stratum": stratum,
                    "scenario_count": len(group),
                    "mean_direct_routes": sum(float(row["direct_intersection_route_count"]) for row in group) / len(group),
                    "exact_conflict_rate": sum(int(row["exact_temporal_conflict_count"]) > 0 for row in group) / len(group),
                    "mean_exact_conflicts": sum(float(row["exact_temporal_conflict_count"]) for row in group) / len(group),
                    "mean_aligned_conflicts": sum(float(row["aligned_temporal_conflict_count"]) for row in group) / len(group),
                    "mean_conflict_score": sum(float(row["conflict_score"]) for row in group) / len(group),
                }
            )
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate three structurally distinct, conflict-balanced spatial benchmarks."
    )
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument(
        "--audit-output",
        default="outputs/spatial_generalization_balanced_v3_design",
    )
    args = parser.parse_args()
    config_names = tuple(args.configs or DEFAULT_CONFIGS)
    audit_dir = _resolve(args.audit_output)
    audit_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_summaries = []
    problems = []
    manifest_records = []
    for config_name in config_names:
        config = load_config(_resolve(config_name))
        problem = _load_problem(config)
        manifest = build_balanced_manifest(problem, config)
        target = _resolve(config["spatial_generalization"]["manifest"])
        write_json(manifest, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        render_problem_layout(problem, _resolve(f"maps/previews/{problem.map_id}_balanced_v3_static.png"))
        render_route_pool_overview(
            problem,
            manifest,
            _resolve(config["spatial_generalization"]["route_pool_preview"]),
        )
        render_validation_gallery(
            problem,
            manifest,
            _resolve(config["spatial_generalization"]["validation_preview"]),
        )
        rows, summaries = _audit(problem, manifest)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        problems.append(problem)
        manifest_records.append(
            {
                "map_id": problem.map_id,
                "config": str(_resolve(config_name)),
                "manifest": str(target),
                "sha256": digest,
                "astar_steps": problem.astar_steps,
                "turn_count": problem.metadata["turn_count"],
                "obstacle_count": len(problem.obstacles),
            }
        )
        print(
            f"{problem.map_id}: manifest={target.name} sha256={digest[:12]} "
            f"A*={problem.astar_steps} turns={problem.metadata['turn_count']}",
            flush=True,
        )
    render_problem_overview(
        problems, _resolve("maps/previews/spatial_generalization_balanced_v3_static_overview.png")
    )
    write_records_csv(all_rows, audit_dir / "scenario_conflicts.csv")
    write_records_csv(all_summaries, audit_dir / "split_stratum_summary.csv")
    write_json(manifest_records, audit_dir / "manifests.json")
    print(f"Balanced v3 design audit saved to {audit_dir}", flush=True)


if __name__ == "__main__":
    main()
