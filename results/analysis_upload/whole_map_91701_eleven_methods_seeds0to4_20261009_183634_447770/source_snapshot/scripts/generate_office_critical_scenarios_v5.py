"""Generate the Office v5 topology-critical dynamic-obstacle benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from astar_d3qn.core.astar import randomized_tie_astar_path
from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import (
    SPLITS,
    _enumerate_routes,
    _pack_routes,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.evaluation.conflict import (
    obstacle_positions,
    scenario_conflict_metrics,
    scenario_safe_path_metrics,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json


DEFAULT_CONFIG = "configs/dynamic_spatial_generalization_office_critical_v5_terminal_v1.yaml"
ZONE_COLORS = {
    "upper_left_gate": "#1f77b4",
    "upper_right_gate": "#ff7f0e",
    "lower_left_gate": "#2ca02c",
    "lower_right_gate": "#9467bd",
    "goal_approach": "#d62728",
}
ZONE_LABELS = {
    "upper_left_gate": "Z1 upper-left gate",
    "upper_right_gate": "Z2 upper-right gate",
    "lower_left_gate": "Z3 lower-left gate",
    "lower_right_gate": "Z4 lower-right gate",
    "goal_approach": "Z5 goal approach",
}

ZONE_ROUTE_QUOTAS = {
    "upper_left_gate": {"train": 3, "validation": 2, "test": 2},
    "upper_right_gate": {"train": 3, "validation": 2, "test": 3},
    "lower_left_gate": {"train": 3, "validation": 2, "test": 2},
    "lower_right_gate": {"train": 3, "validation": 2, "test": 3},
    "goal_approach": {"train": 3, "validation": 2, "test": 3},
}
MIN_TOPOLOGY_PATH_COVERAGE = 0.10
CANDIDATE_PHASE_SAMPLES_PER_ROUTE_COMBINATION = 1000

# Difficulty is defined by the fraction of the actual 20 static A* demonstration
# paths that would collide if replayed without reacting.  Route placement itself
# remains topology-based, so this score controls timing rather than geometry.
STRATUM_TARGETS = {
    "low": (6, 8, 10),
    "medium": (12, 14, 16),
    "high": (18, 19, 20),
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_problem(config: Mapping[str, Any]):
    source = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(source["file"])))
    matches = [problem for problem in problems if problem.map_id == config["map"]["scene"]]
    if len(matches) != 1 or matches[0].map_id != "office_40x40":
        raise ValueError("Office critical v5 requires the registered office_40x40 map.")
    return matches[0]


def _reference_demo_paths(problem, episodes: int, seed: int):
    paths = []
    for episode in range(episodes):
        path = randomized_tie_astar_path(
            problem.start,
            problem.goal,
            problem.obstacles,
            problem.size,
            random.Random(seed + episode),
        )
        if path is None:
            raise RuntimeError("Could not regenerate the configured static A* path family.")
        paths.append(tuple(path))
    return tuple(paths)


def _orientation(route: Sequence[tuple[int, int]]) -> str:
    return "horizontal" if len({cell[0] for cell in route}) == 1 else "vertical"


def _critical_zone(record: Mapping[str, Any]) -> str:
    row, column = record["center"]
    if row <= 12:
        return "upper_left_gate" if column <= 17 else "upper_right_gate"
    if row <= 29:
        return "lower_left_gate" if column <= 24 else "lower_right_gate"
    return "goal_approach"


def _allocate_zone_records(records, quotas):
    """Split one functional zone while matching mean path coverage."""

    required = sum(int(quotas[split]) for split in SPLITS)
    if len(records) < required:
        raise ValueError(f"Only {len(records)} routes are available; {required} are required.")
    selected = sorted(
        records,
        key=lambda item: (
            -int(item["topology_path_coverage_count"]),
            -int(item["corridor_score"]),
            tuple(item["center"]),
        ),
    )[:required]
    train_count = int(quotas["train"])
    validation_count = int(quotas["validation"])
    indices = tuple(range(required))
    best = None
    for train_indices in itertools.combinations(indices, train_count):
        train_set = set(train_indices)
        remaining = tuple(index for index in indices if index not in train_set)
        for validation_indices in itertools.combinations(remaining, validation_count):
            validation_set = set(validation_indices)
            groups = {
                "train": [selected[index] for index in train_indices],
                "validation": [selected[index] for index in validation_indices],
                "test": [
                    selected[index]
                    for index in remaining
                    if index not in validation_set
                ],
            }
            coverage_means = {
                split: sum(float(item["topology_path_coverage_count"]) for item in group)
                / len(group)
                for split, group in groups.items()
            }
            demo_means = {
                split: sum(float(item["reference_demo_path_coverage_count"]) for item in group)
                / len(group)
                for split, group in groups.items()
            }
            score_means = {
                split: sum(float(item["corridor_score"]) for item in group) / len(group)
                for split, group in groups.items()
            }
            objective = (
                max(coverage_means.values()) - min(coverage_means.values()),
                max(demo_means.values()) - min(demo_means.values()),
                max(score_means.values()) - min(score_means.values()),
                tuple(tuple(tuple(item["center"]) for item in groups[split]) for split in SPLITS),
            )
            if best is None or objective < best[0]:
                best = (objective, groups)
    if best is None:
        raise RuntimeError("Could not allocate topology-critical routes.")
    return best[1]


def _build_route_pools(problem, config, demo_paths):
    spatial = config["spatial_generalization"]
    reference_count = int(spatial["reference_path_count"])
    topology_seed = int(spatial["generation_seed"]) + 100_000
    topology_paths = _reference_demo_paths(problem, reference_count, topology_seed)
    topology_sets = tuple(set(path) for path in topology_paths)
    frequencies: dict[tuple[int, int], int] = {}
    for path in topology_paths:
        for cell in path:
            frequencies[cell] = frequencies.get(cell, 0) + 1
    packed = _pack_routes(
        _enumerate_routes(problem, int(spatial["route_length"])),
        frequencies,
        int(spatial["separation_radius"]),
        int(spatial["generation_seed"]),
    )
    minimum_coverage = round(reference_count * MIN_TOPOLOGY_PATH_COVERAGE)
    candidates_by_zone = {zone: [] for zone in ZONE_COLORS}
    for packed_record in packed:
        route = tuple(packed_record["route"])
        coverage = sum(bool(set(route).intersection(path)) for path in topology_sets)
        if coverage < minimum_coverage:
            continue
        if min(abs(cell[0] - problem.start[0]) + abs(cell[1] - problem.start[1]) for cell in route) < 3:
            continue
        demo_intersections = [sum(cell in path for cell in route) for path in demo_paths]
        record = {
            "center": list(packed_record["center"]),
            "orientation": _orientation(route),
            "route": [list(cell) for cell in route],
            "corridor_score": int(packed_record["corridor_score"]),
            "corridor_frequency": float(packed_record["corridor_score"]) / reference_count,
            "topology_path_coverage_count": coverage,
            "topology_path_coverage_rate": coverage / reference_count,
            "reference_demo_path_coverage_count": sum(value > 0 for value in demo_intersections),
            "reference_demo_intersection_cell_count": sum(demo_intersections),
        }
        candidates_by_zone[_critical_zone(record)].append(record)

    allocated = {split: [] for split in SPLITS}
    for zone, quotas in ZONE_ROUTE_QUOTAS.items():
        groups = _allocate_zone_records(candidates_by_zone[zone], quotas)
        for split in SPLITS:
            for source in groups[split]:
                allocated[split].append({**source, "critical_zone": zone})

    route_pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in SPLITS:
        records = []
        zone_counters = Counter()
        for source in sorted(
            allocated[split],
            key=lambda item: (list(ZONE_COLORS).index(item["critical_zone"]), tuple(item["center"])),
        ):
            zone = str(source["critical_zone"])
            zone_counters[zone] += 1
            zone_number = list(ZONE_COLORS).index(zone) + 1
            records.append(
                {
                    **source,
                    "route_id": f"{split}_critical_z{zone_number}_{zone_counters[zone]:02d}",
                    "category": "corridor",
                    "functional_role": "topology_critical_route_variant",
                    "zone_variant": zone_counters[zone],
                }
            )
        route_pools[split] = {"corridor": records, "background": []}
    return route_pools


def _option_collision_mask(
    route: tuple[tuple[int, int], ...],
    start_index: int,
    direction: int,
    move_every: int,
    demo_paths,
) -> int:
    horizon = max(len(path) - 1 for path in demo_paths)
    positions = obstacle_positions(
        DynamicObstacleSpec(
            route=route,
            start_index=start_index,
            direction=direction,
            move_every=move_every,
        ),
        horizon,
    )
    mask = 0
    for path_index, path in enumerate(demo_paths):
        collided = any(
            path[step] == positions[step - 1] or path[step] == positions[step]
            for step in range(1, len(path))
        )
        if collided:
            mask |= 1 << path_index
    return mask


def _route_options(record, move_every: int, demo_paths):
    route = tuple(tuple(cell) for cell in record["route"])
    result = []
    for start_index in range(len(route)):
        for direction in (-1, 1):
            result.append(
                {
                    "start_index": start_index,
                    "direction": direction,
                    "move_every": move_every,
                    "collision_mask": _option_collision_mask(
                        route, start_index, direction, move_every, demo_paths
                    ),
                }
            )
    return result


def _stratum_counts(total: int) -> dict[str, int]:
    low = round(total * 0.40)
    medium = round(total * 0.30)
    return {"low": low, "medium": medium, "high": total - low - medium}


def _scenario_from_configuration(
    scenario_id: int,
    stratum: str,
    records,
    options,
    configuration,
    reference_collision_count: int,
    problem,
):
    obstacle_items = []
    specs = []
    for record_index, option_index in configuration:
        record = records[record_index]
        option = options[record_index][option_index]
        obstacle_items.append(
            {
                "route_id": record["route_id"],
                "start_index": option["start_index"],
                "direction": option["direction"],
                "move_every": option["move_every"],
            }
        )
        specs.append(
            DynamicObstacleSpec(
                route=tuple(tuple(cell) for cell in record["route"]),
                start_index=option["start_index"],
                direction=option["direction"],
                move_every=option["move_every"],
                label=str(record["route_id"]),
                reference_path_source=str(record["critical_zone"]),
            )
        )
    dynamic_scenario = DynamicScenario(seed=scenario_id, obstacles=tuple(specs))
    safe_metrics = scenario_safe_path_metrics(problem, dynamic_scenario)
    if safe_metrics["minimum_safe_path_steps"] is None:
        return None
    return {
        "scenario_id": scenario_id,
        "difficulty_stratum": stratum,
        "reference_demo_collision_count": reference_collision_count,
        "reference_demo_collision_rate": reference_collision_count / 20.0,
        **safe_metrics,
        "obstacles": obstacle_items,
    }


def _build_scenarios(problem, config, route_pools, demo_paths):
    spatial = config["spatial_generalization"]
    move_every = int(spatial["move_every"])
    generation_seed = int(spatial["generation_seed"])
    offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    scenarios: dict[str, list[dict[str, Any]]] = {}
    for split_index, split in enumerate(SPLITS):
        rng = random.Random(generation_seed + (split_index + 1) * 10_000)
        records = route_pools[split]["corridor"]
        options = [_route_options(record, move_every, demo_paths) for record in records]
        indices_by_zone = {
            zone: [index for index, record in enumerate(records) if record["critical_zone"] == zone]
            for zone in ZONE_COLORS
        }
        route_combinations = list(itertools.product(*(indices_by_zone[zone] for zone in ZONE_COLORS)))
        required_combinations = int(spatial["scenario_counts"][split])
        if len(route_combinations) < required_combinations:
            raise ValueError(
                f"{split} has only {len(route_combinations)} route combinations; "
                f"{required_combinations} unique layouts are required."
            )
        buckets: dict[int, list[tuple[tuple[int, int], ...]]] = {value: [] for value in range(21)}
        for route_combination in route_combinations:
            retained: dict[int, tuple[tuple[int, int], ...]] = {}
            phase_codes = rng.sample(
                range(100_000),
                CANDIDATE_PHASE_SAMPLES_PER_ROUTE_COMBINATION,
            )
            for phase_code in phase_codes:
                value = phase_code
                phase_indices = []
                for _ in range(5):
                    phase_indices.append(value % 10)
                    value //= 10
                configuration = tuple(zip(route_combination, phase_indices))
                mask = 0
                for record_index, option_index in configuration:
                    mask |= int(options[record_index][option_index]["collision_mask"])
                collision_count = mask.bit_count()
                if collision_count in retained:
                    continue
                retained[collision_count] = configuration
            for collision_count, configuration in retained.items():
                buckets[collision_count].append(configuration)
        for values in buckets.values():
            rng.shuffle(values)

        split_scenarios = []
        used_route_combinations = set()
        route_use = Counter()
        counts = _stratum_counts(int(spatial["scenario_counts"][split]))
        for stratum in ("low", "medium", "high"):
            targets = STRATUM_TARGETS[stratum]
            for item_index in range(counts[stratum]):
                target = targets[item_index % len(targets)]
                accepted = None
                while accepted is None:
                    eligible = [
                        configuration
                        for configuration in buckets[target]
                        if tuple(record_index for record_index, _ in configuration)
                        not in used_route_combinations
                    ]
                    if not eligible:
                        raise RuntimeError(
                            f"No unused {split} route combination remained for "
                            f"{stratum} target {target}; bucket sizes are "
                            f"{ {key: len(value) for key, value in buckets.items() if value} }."
                        )
                    configuration = min(
                        eligible,
                        key=lambda item: (
                            max(route_use[record_index] for record_index, _ in item),
                            sum(route_use[record_index] for record_index, _ in item),
                            rng.random(),
                        ),
                    )
                    buckets[target].remove(configuration)
                    accepted = _scenario_from_configuration(
                        offsets[split] + len(split_scenarios),
                        stratum,
                        records,
                        options,
                        configuration,
                        target,
                        problem,
                    )
                route_combination = tuple(record_index for record_index, _ in configuration)
                used_route_combinations.add(route_combination)
                for record_index in route_combination:
                    route_use[record_index] += 1
                split_scenarios.append(accepted)
        scenarios[split] = split_scenarios
    return scenarios


def build_manifest(problem, config):
    spatial = config["spatial_generalization"]
    demo_episodes = int(config["demonstrations"]["episodes"])
    demo_seed = int(config["demonstrations"]["seed"])
    if demo_episodes != 20:
        raise ValueError("Office critical v5 is calibrated against exactly 20 A* demonstrations.")
    demo_paths = _reference_demo_paths(problem, demo_episodes, demo_seed)
    route_pools = _build_route_pools(problem, config, demo_paths)
    scenarios = _build_scenarios(problem, config, route_pools, demo_paths)
    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "office_topology_critical_v5",
            "seed": int(spatial["generation_seed"]),
            "route_length": 5,
            "reference_path_count": int(spatial["reference_path_count"]),
            "reference_demo_episodes": demo_episodes,
            "reference_demo_seed": demo_seed,
            "separation_radius": int(spatial["separation_radius"]),
            "move_every": int(spatial["move_every"]),
            "corridor_per_scenario": 5,
            "background_per_scenario": 0,
            "critical_zones": list(ZONE_COLORS),
            "placement_rule": "five_topology_critical_regions_no_background_routes",
            "route_pool_counts": {
                split: len(route_pools[split]["corridor"]) for split in SPLITS
            },
            "zone_route_quotas": ZONE_ROUTE_QUOTAS,
            "minimum_topology_path_coverage_rate": MIN_TOPOLOGY_PATH_COVERAGE,
            "split_rule": "same_functional_regions_spatially_disjoint_route_pools",
            "scenario_rule": "one_random_route_per_critical_zone_unique_route_combination",
            "difficulty_rule": "static_demo_collision_rate_controlled_by_phase_and_direction",
            "difficulty_targets": {key: list(value) for key, value in STRATUM_TARGETS.items()},
        },
        "route_pools": route_pools,
        "scenarios": scenarios,
    }
    validate_spatial_scenario_manifest(problem, manifest)
    return manifest


def _grid(problem):
    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1
    return grid


def _draw_base(axis, problem, title: str, *, ticks: bool = False):
    axis.imshow(_grid(problem), cmap="Greys", origin="upper", vmin=0, vmax=1)
    axis.plot(
        [cell[1] for cell in problem.nominal_path],
        [cell[0] for cell in problem.nominal_path],
        color="#64b5f6",
        linewidth=0.9,
        linestyle="--",
        alpha=0.55,
    )
    axis.scatter(problem.start[1], problem.start[0], c="#2e7d32", s=18, zorder=6)
    axis.scatter(problem.goal[1], problem.goal[0], c="#fbc02d", edgecolors="black", marker="*", s=35, zorder=6)
    axis.set_xlim(-0.5, problem.size - 0.5)
    axis.set_ylim(problem.size - 0.5, -0.5)
    if ticks:
        axis.set_xticks(range(0, problem.size, 5))
        axis.set_yticks(range(0, problem.size, 5))
        axis.tick_params(labelsize=6)
        axis.grid(color="#cfd8dc", linewidth=0.25, alpha=0.6)
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title(title, fontsize=8)


def render_route_pool_overview(problem, manifest, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), dpi=180)
    for axis, split in zip(axes, SPLITS):
        route_count = len(manifest["route_pools"][split]["corridor"])
        _draw_base(
            axis,
            problem,
            f"{split.title()}: {route_count} critical routes; five selected per layout",
            ticks=True,
        )
        for record in manifest["route_pools"][split]["corridor"]:
            route = record["route"]
            color = ZONE_COLORS[record["critical_zone"]]
            axis.plot([cell[1] for cell in route], [cell[0] for cell in route], color=color, linewidth=2.6)
            row, column = record["center"]
            zone_number = list(ZONE_COLORS).index(record["critical_zone"]) + 1
            variant = int(record["zone_variant"])
            axis.scatter(column, row, c=color, edgecolors="white", linewidths=0.5, s=24, zorder=5)
            axis.text(column + 0.5, row - 0.6, f"Z{zone_number}{chr(96 + variant)} ({column},{row})", color=color, fontsize=5.7, weight="bold")
    legend = [
        Line2D([0], [0], color=color, linewidth=2.6, label=ZONE_LABELS[zone])
        for zone, color in ZONE_COLORS.items()
    ]
    legend.append(Line2D([0], [0], color="#64b5f6", linestyle="--", label="one nominal A* path"))
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Office v5: topology-critical and spatially disjoint route pools", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _next_route_cell(route, start_index: int, direction: int):
    candidate = start_index + direction
    if candidate < 0 or candidate >= len(route):
        candidate = start_index - direction
    return route[candidate]


def render_scenario_gallery(problem, manifest, split: str, output: Path):
    scenarios = manifest["scenarios"][split]
    records = {record["route_id"]: record for record in manifest["route_pools"][split]["corridor"]}
    columns = 10 if len(scenarios) >= 50 else 5
    rows = (len(scenarios) + columns - 1) // columns
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, columns, figsize=(2.55 * columns, 2.65 * rows), dpi=140)
    axes = np.asarray(axes).reshape(-1)
    for axis, scenario in zip(axes, scenarios):
        short_stratum = str(scenario["difficulty_stratum"])[0].upper()
        detour = int(scenario["safe_detour_steps"])
        title = f"{scenario['scenario_id']} | {short_stratum} | demo {scenario['reference_demo_collision_count']}/20 | safe +{detour}"
        _draw_base(axis, problem, title)
        for obstacle in scenario["obstacles"]:
            record = records[obstacle["route_id"]]
            route = record["route"]
            color = ZONE_COLORS[record["critical_zone"]]
            axis.plot([cell[1] for cell in route], [cell[0] for cell in route], color=color, linewidth=1.6, alpha=0.9)
            start_index = int(obstacle["start_index"])
            initial = route[start_index]
            next_cell = _next_route_cell(route, start_index, int(obstacle["direction"]))
            axis.scatter(initial[1], initial[0], c="#d32f2f", edgecolors="white", linewidths=0.35, s=16, zorder=6)
            axis.annotate(
                "",
                xy=(next_cell[1], next_cell[0]),
                xytext=(initial[1], initial[0]),
                arrowprops={"arrowstyle": "->", "color": "#212121", "lw": 0.7},
                zorder=7,
            )
    for axis in axes[len(scenarios):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Office v5 {split}: {len(scenarios)} layouts | red=start, arrow=first movement | L/M/H=demo conflict stratum",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _audit_rows(problem, manifest):
    rows = []
    for split in SPLITS:
        lookup = {record["route_id"]: record for record in manifest["route_pools"][split]["corridor"]}
        for source in manifest["scenarios"][split]:
            specs = []
            for item in source["obstacles"]:
                record = lookup[item["route_id"]]
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
                    "split": split,
                    "scenario_id": source["scenario_id"],
                    "difficulty_stratum": source["difficulty_stratum"],
                    "reference_demo_collision_count": source["reference_demo_collision_count"],
                    "reference_demo_collision_rate": source["reference_demo_collision_rate"],
                    "minimum_safe_path_steps": source["minimum_safe_path_steps"],
                    "safe_detour_steps": source["safe_detour_steps"],
                    "nominal_exact_temporal_conflicts": metrics["exact_temporal_conflict_count"],
                    "nominal_aligned_temporal_conflicts": metrics["aligned_temporal_conflict_count"],
                }
            )
    return rows


def _write_csv(rows, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-output", default="outputs/office_critical_v5_design")
    args = parser.parse_args()
    config = load_config(_resolve(args.config))
    problem = _load_problem(config)
    manifest = build_manifest(problem, config)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(spatial["manifest"])
    write_json(manifest, manifest_path)
    render_route_pool_overview(problem, manifest, _resolve(spatial["route_pool_preview"]))
    render_scenario_gallery(problem, manifest, "train", _resolve(spatial["training_preview"]))
    render_scenario_gallery(problem, manifest, "validation", _resolve(spatial["validation_preview"]))
    render_scenario_gallery(problem, manifest, "test", _resolve(spatial["test_preview"]))
    rows = _audit_rows(problem, manifest)
    audit_root = _resolve(args.audit_output)
    _write_csv(rows, audit_root / "scenario_audit.csv")
    summary = []
    for split in SPLITS:
        group = [row for row in rows if row["split"] == split]
        source_scenarios = manifest["scenarios"][split]
        route_combinations = {
            tuple(sorted(item["route_id"] for item in source["obstacles"]))
            for source in source_scenarios
        }
        route_usage = Counter(
            item["route_id"]
            for source in source_scenarios
            for item in source["obstacles"]
        )
        summary.append(
            {
                "split": split,
                "scenario_count": len(group),
                "unique_route_combination_count": len(route_combinations),
                "route_usage_minimum": min(route_usage.values()),
                "route_usage_maximum": max(route_usage.values()),
                "stratum_counts": dict(Counter(row["difficulty_stratum"] for row in group)),
                "mean_reference_demo_collision_rate": sum(float(row["reference_demo_collision_rate"]) for row in group) / len(group),
                "mean_safe_detour_steps": sum(float(row["safe_detour_steps"]) for row in group) / len(group),
                "maximum_safe_detour_steps": max(int(row["safe_detour_steps"]) for row in group),
            }
        )
    write_json(summary, audit_root / "split_summary.json")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"manifest={manifest_path} sha256={digest[:12]}")
    for item in summary:
        print(
            f"{item['split']}: scenarios={item['scenario_count']} strata={item['stratum_counts']} "
            f"demo_collision={item['mean_reference_demo_collision_rate']:.1%} "
            f"safe_detour_mean={item['mean_safe_detour_steps']:.2f} "
            f"safe_detour_max={item['maximum_safe_detour_steps']}"
        )


if __name__ == "__main__":
    main()
