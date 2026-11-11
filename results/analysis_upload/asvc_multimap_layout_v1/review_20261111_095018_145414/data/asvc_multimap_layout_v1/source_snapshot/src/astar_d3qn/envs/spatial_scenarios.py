"""Spatially disjoint dynamic-obstacle scenario generation."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from astar_d3qn.core.astar import randomized_tie_astar_path
from astar_d3qn.core.grid import Position
from astar_d3qn.maps.problem import NavigationProblem

from .dynamic_grid import DynamicObstacleSpec, _valid_route
from .dynamic_scenarios import DynamicScenario


SPLITS = ("train", "validation", "test")


def _enumerate_routes(
    problem: NavigationProblem,
    route_length: int,
) -> list[dict[str, Any]]:
    if route_length < 3 or route_length % 2 == 0:
        raise ValueError("route_length must be an odd integer of at least three.")
    half = route_length // 2
    routes: list[dict[str, Any]] = []
    for row in range(half, problem.size - half):
        for column in range(half, problem.size - half):
            candidates = (
                (
                    "horizontal",
                    tuple(
                        (row, column + offset)
                        for offset in range(-half, half + 1)
                    ),
                ),
                (
                    "vertical",
                    tuple(
                        (row + offset, column)
                        for offset in range(-half, half + 1)
                    ),
                ),
            )
            for orientation, route in candidates:
                if _valid_route(route, problem, min_length=route_length):
                    routes.append(
                        {
                            "center": (row, column),
                            "orientation": orientation,
                            "route": route,
                        }
                    )
    return routes


def _corridor_frequency(
    problem: NavigationProblem,
    reference_path_count: int,
    seed: int,
) -> dict[Position, int]:
    if reference_path_count <= 0:
        raise ValueError("reference_path_count must be positive.")
    frequency: dict[Position, int] = {}
    for offset in range(reference_path_count):
        path = randomized_tie_astar_path(
            problem.start,
            problem.goal,
            problem.obstacles,
            problem.size,
            random.Random(seed + offset),
        )
        if path is None:
            raise RuntimeError("Randomized A* failed on the registered map.")
        for cell in path:
            frequency[cell] = frequency.get(cell, 0) + 1
    return frequency


def _pack_routes(
    candidates: Sequence[dict[str, Any]],
    frequency: Mapping[Position, int],
    separation_radius: int,
    seed: int,
) -> list[dict[str, Any]]:
    if separation_radius < 0:
        raise ValueError("separation_radius must be non-negative.")
    rng = random.Random(seed)
    ranked = []
    for candidate in candidates:
        score = max(frequency.get(cell, 0) for cell in candidate["route"])
        ranked.append((-score, rng.random(), candidate))
    ranked.sort(key=lambda item: (item[0], item[1]))

    selected: list[dict[str, Any]] = []
    occupied: set[Position] = set()
    for negative_score, _, candidate in ranked:
        expanded = {
            (row + row_offset, column + column_offset)
            for row, column in candidate["route"]
            for row_offset in range(-separation_radius, separation_radius + 1)
            for column_offset in range(-separation_radius, separation_radius + 1)
        }
        if occupied.intersection(expanded):
            continue
        selected.append(
            {
                **candidate,
                "corridor_score": -negative_score,
                "category": "corridor" if negative_score < 0 else "background",
            }
        )
        occupied.update(candidate["route"])
    return selected


def _allocate_route_pool(
    records: Sequence[dict[str, Any]],
    targets: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    required = sum(int(targets[split]) for split in SPLITS)
    if len(records) < required:
        raise ValueError(
            f"Only {len(records)} routes are available, but {required} are required."
        )
    result = {split: [] for split in SPLITS}
    for record in records[:required]:
        eligible = [
            split
            for split in SPLITS
            if len(result[split]) < int(targets[split])
        ]
        split = min(
            eligible,
            key=lambda name: (
                len(result[name]) / max(1, int(targets[name])),
                SPLITS.index(name),
            ),
        )
        result[split].append(record)
    return result


def _build_scenarios(
    route_pool: Mapping[str, Sequence[dict[str, Any]]],
    scenario_count: int,
    corridor_per_scenario: int,
    background_per_scenario: int,
    move_every: int,
    seed: int,
    scenario_id_start: int,
) -> list[dict[str, Any]]:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")
    corridor = list(route_pool["corridor"])
    background = list(route_pool["background"])
    if len(corridor) < corridor_per_scenario:
        raise ValueError("The route pool has too few corridor routes.")
    if len(background) < background_per_scenario:
        raise ValueError("The route pool has too few background routes.")
    rng = random.Random(seed)
    scenarios: list[dict[str, Any]] = []
    combinations: set[tuple[str, ...]] = set()
    attempts = 0
    while len(scenarios) < scenario_count:
        attempts += 1
        if attempts > scenario_count * 1000:
            raise RuntimeError("Could not generate enough unique route combinations.")
        selected = rng.sample(corridor, corridor_per_scenario) + rng.sample(
            background, background_per_scenario
        )
        combination = tuple(sorted(str(item["route_id"]) for item in selected))
        if combination in combinations:
            continue
        combinations.add(combination)
        obstacles = []
        for route in selected:
            obstacles.append(
                {
                    "route_id": route["route_id"],
                    "start_index": rng.randrange(len(route["route"])),
                    "direction": rng.choice((-1, 1)),
                    "move_every": move_every,
                }
            )
        scenarios.append(
            {
                "scenario_id": scenario_id_start + len(scenarios),
                "obstacles": obstacles,
            }
        )
    return scenarios


def build_spatial_scenario_manifest(
    problem: NavigationProblem,
    *,
    generation_seed: int,
    route_length: int,
    reference_path_count: int,
    separation_radius: int,
    route_pool_counts: Mapping[str, Mapping[str, int]],
    scenario_counts: Mapping[str, int],
    corridor_per_scenario: int,
    background_per_scenario: int,
    move_every: int,
) -> dict[str, Any]:
    candidates = _enumerate_routes(problem, route_length)
    frequency = _corridor_frequency(
        problem,
        reference_path_count,
        generation_seed + 100_000,
    )
    packed = _pack_routes(
        candidates,
        frequency,
        separation_radius,
        generation_seed,
    )
    corridor_records = [item for item in packed if item["category"] == "corridor"]
    background_records = [
        item for item in packed if item["category"] == "background"
    ]
    corridor_targets = {
        split: int(route_pool_counts[split]["corridor"]) for split in SPLITS
    }
    background_targets = {
        split: int(route_pool_counts[split]["background"]) for split in SPLITS
    }
    corridor_split = _allocate_route_pool(corridor_records, corridor_targets)
    background_split = _allocate_route_pool(background_records, background_targets)

    route_pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in SPLITS:
        route_pools[split] = {"corridor": [], "background": []}
        for category, records in (
            ("corridor", corridor_split[split]),
            ("background", background_split[split]),
        ):
            for index, record in enumerate(records):
                route_pools[split][category].append(
                    {
                        "route_id": f"{split}_{category}_{index:02d}",
                        "category": category,
                        "center": list(record["center"]),
                        "orientation": record["orientation"],
                        "route": [list(cell) for cell in record["route"]],
                        "corridor_score": int(record["corridor_score"]),
                        "corridor_frequency": float(record["corridor_score"])
                        / reference_path_count,
                    }
                )

    scenario_offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    scenarios = {}
    for split_index, split in enumerate(SPLITS):
        scenarios[split] = _build_scenarios(
            route_pools[split],
            int(scenario_counts[split]),
            corridor_per_scenario,
            background_per_scenario,
            move_every,
            generation_seed + (split_index + 1) * 10_000,
            scenario_offsets[split],
        )

    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "seed": generation_seed,
            "route_length": route_length,
            "reference_path_count": reference_path_count,
            "separation_radius": separation_radius,
            "move_every": move_every,
            "corridor_per_scenario": corridor_per_scenario,
            "background_per_scenario": background_per_scenario,
            "legal_route_count": len(candidates),
            "packed_route_count": len(packed),
        },
        "route_pools": route_pools,
        "scenarios": scenarios,
    }
    validate_spatial_scenario_manifest(problem, manifest)
    return manifest


def _route_lookup(
    manifest: Mapping[str, Any],
    split: str,
) -> dict[str, Mapping[str, Any]]:
    pool = manifest["route_pools"][split]
    return {
        str(record["route_id"]): record
        for category in ("corridor", "background")
        for record in pool[category]
    }


def validate_spatial_scenario_manifest(
    problem: NavigationProblem,
    manifest: Mapping[str, Any],
) -> None:
    if int(manifest.get("format_version", -1)) != 1:
        raise ValueError("Unsupported spatial scenario manifest format.")
    if manifest.get("map_id") != problem.map_id:
        raise ValueError("Spatial scenario manifest map_id does not match.")
    if manifest.get("grid_sha256") != problem.grid_sha256:
        raise ValueError("Spatial scenario manifest grid hash does not match.")
    separation_radius = int(manifest["generation"]["separation_radius"])
    if separation_radius < 0:
        raise ValueError("Manifest separation_radius must be non-negative.")
    corridor_per_scenario = int(
        manifest["generation"]["corridor_per_scenario"]
    )
    background_per_scenario = int(
        manifest["generation"]["background_per_scenario"]
    )
    expected_obstacle_count = corridor_per_scenario + background_per_scenario
    split_cells: dict[str, set[Position]] = {}
    all_route_ids: set[str] = set()
    all_scenario_ids: set[int] = set()
    for split in SPLITS:
        lookup = _route_lookup(manifest, split)
        if len(lookup) != sum(
            len(manifest["route_pools"][split][category])
            for category in ("corridor", "background")
        ):
            raise ValueError("Route identifiers must be globally unique within a split.")
        cells: set[Position] = set()
        registered_routes: list[tuple[set[Position], str | None]] = []
        for category in ("corridor", "background"):
            for record in manifest["route_pools"][split][category]:
                route_id = str(record["route_id"])
                if route_id in all_route_ids:
                    raise ValueError("Route identifiers must be unique across splits.")
                all_route_ids.add(route_id)
                if record.get("category") != category:
                    raise ValueError(
                        f"Route {route_id!r} is stored in the wrong category."
                    )
                route = tuple(
                    tuple(int(value) for value in cell) for cell in record["route"]
                )
                if not _valid_route(route, problem, min_length=len(route)):
                    raise ValueError(f"Invalid route {route_id!r} in manifest.")
                center = tuple(int(value) for value in record["center"])
                if center != route[len(route) // 2]:
                    raise ValueError(f"Route {route_id!r} has an invalid center.")
                if len({cell[0] for cell in route}) == 1:
                    expected_orientation = "horizontal"
                elif len({cell[1] for cell in route}) == 1:
                    expected_orientation = "vertical"
                else:
                    expected_orientation = "turning"
                if record.get("orientation") != expected_orientation:
                    raise ValueError(f"Route {route_id!r} has an invalid orientation.")
                score = int(record["corridor_score"])
                if (category == "corridor" and score <= 0) or (
                    category == "background" and score != 0
                ):
                    raise ValueError(f"Route {route_id!r} has an invalid corridor score.")
                expanded = {
                    (row + row_offset, column + column_offset)
                    for row, column in route
                    for row_offset in range(-separation_radius, separation_radius + 1)
                    for column_offset in range(
                        -separation_radius, separation_radius + 1
                    )
                }
                alternative_group = record.get("alternative_group")
                for previous_cells, previous_group in registered_routes:
                    if not previous_cells.intersection(expanded):
                        continue
                    if not alternative_group or alternative_group != previous_group:
                        raise ValueError(
                            f"Routes are not spatially separated within split {split!r}."
                        )
                registered_routes.append((set(route), alternative_group))
                cells.update(route)
        split_cells[split] = cells
        scenario_ids: set[int] = set()
        for scenario in manifest["scenarios"][split]:
            scenario_id = int(scenario["scenario_id"])
            if scenario_id in scenario_ids or scenario_id in all_scenario_ids:
                raise ValueError("Scenario identifiers must be unique across splits.")
            scenario_ids.add(scenario_id)
            all_scenario_ids.add(scenario_id)
            if len(scenario["obstacles"]) != expected_obstacle_count:
                raise ValueError(
                    f"Scenario {scenario_id} has an unexpected obstacle count."
                )
            route_ids = [str(item["route_id"]) for item in scenario["obstacles"]]
            if len(route_ids) != len(set(route_ids)):
                raise ValueError("A scenario cannot reuse a route.")
            if any(route_id not in lookup for route_id in route_ids):
                raise ValueError("Scenario references an unknown route_id.")
            alternative_groups = [
                lookup[route_id].get("alternative_group") for route_id in route_ids
                if lookup[route_id].get("alternative_group")
            ]
            if len(alternative_groups) != len(set(alternative_groups)):
                raise ValueError(
                    "A scenario cannot combine mutually exclusive route alternatives."
                )
            categories = [lookup[route_id]["category"] for route_id in route_ids]
            if categories.count("corridor") != corridor_per_scenario or categories.count(
                "background"
            ) != background_per_scenario:
                raise ValueError(
                    f"Scenario {scenario_id} has an invalid route-category composition."
                )
            for obstacle in scenario["obstacles"]:
                route = lookup[str(obstacle["route_id"])]["route"]
                if not 0 <= int(obstacle["start_index"]) < len(route):
                    raise ValueError("Dynamic obstacle start_index is outside its route.")
                if int(obstacle["direction"]) not in (-1, 1):
                    raise ValueError("Dynamic obstacle direction must be -1 or 1.")
                if int(obstacle["move_every"]) <= 0:
                    raise ValueError("Dynamic obstacle move_every must be positive.")

    if bool(manifest["generation"].get("cross_split_spatial_disjoint", True)):
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                for row, column in split_cells[left]:
                    for row_offset in range(-separation_radius, separation_radius + 1):
                        for column_offset in range(
                            -separation_radius, separation_radius + 1
                        ):
                            if (row + row_offset, column + column_offset) in split_cells[right]:
                                raise ValueError(
                                    f"Route pools {left!r} and {right!r} are not spatially disjoint."
                                )


def scenarios_from_spatial_manifest(
    problem: NavigationProblem,
    manifest: Mapping[str, Any],
    split: str,
) -> tuple[DynamicScenario, ...]:
    if split not in SPLITS:
        raise ValueError(f"Unknown spatial scenario split: {split!r}.")
    validate_spatial_scenario_manifest(problem, manifest)
    lookup = _route_lookup(manifest, split)
    result = []
    for scenario in manifest["scenarios"][split]:
        obstacles = []
        for index, item in enumerate(scenario["obstacles"]):
            record = lookup[str(item["route_id"])]
            route = tuple(
                tuple(int(value) for value in cell) for cell in record["route"]
            )
            obstacles.append(
                DynamicObstacleSpec(
                    route=route,
                    start_index=int(item["start_index"]),
                    direction=int(item["direction"]),
                    move_every=int(item["move_every"]),
                    label=str(item["route_id"]),
                    reference_path_source=(
                        "randomized_shortest_path_corridor"
                        if record["category"] == "corridor"
                        else "off_corridor_open_space"
                    ),
                    reference_path_index=None,
                )
            )
        result.append(
            DynamicScenario(
                seed=int(scenario["scenario_id"]),
                obstacles=tuple(obstacles),
                required_behavior=(
                    str(scenario["required_behavior"])
                    if scenario.get("required_behavior") is not None
                    else None
                ),
                difficulty_stratum=(
                    str(scenario["difficulty_stratum"])
                    if scenario.get("difficulty_stratum") is not None
                    else None
                ),
            )
        )
    return tuple(result)
