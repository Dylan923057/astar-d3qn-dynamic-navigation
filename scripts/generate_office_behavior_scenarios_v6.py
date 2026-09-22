"""Generate Office scenarios verified to require wait, avoidance, or rerouting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec, _valid_route
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import SPLITS, validate_spatial_scenario_manifest
from astar_d3qn.evaluation.behavior_oracle import SafePlan, shortest_safe_plan
from astar_d3qn.evaluation.conflict import obstacle_positions
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json

from generate_office_critical_scenarios_v5 import (
    ZONE_COLORS,
    ZONE_LABELS,
    _build_route_pools,
    _draw_base,
    _load_problem,
    _reference_demo_paths,
    _resolve,
)


DEFAULT_CONFIG = "configs/dynamic_spatial_generalization_office_behavior_v6_terminal_v1.yaml"
BEHAVIOR_COUNTS = {
    "train": {"wait": 40, "avoidance": 30, "reroute": 30},
    "validation": {"wait": 8, "avoidance": 6, "reroute": 6},
    "test": {"wait": 20, "avoidance": 15, "reroute": 15},
}
BEHAVIOR_COLORS = {
    "wait": "#7b1fa2",
    "avoidance": "#00897b",
    "reroute": "#ef6c00",
}
GATE_CELLS = {
    "upper_left": frozenset(((10, 9), (11, 9))),
    "upper_right": frozenset(((10, 27), (11, 27))),
    "lower_left": frozenset(((25, 17), (26, 17))),
    "lower_right": frozenset(((25, 31), (26, 31))),
}
GATE_PAIRS = (
    ("upper_left", "lower_left"),
    ("upper_left", "lower_right"),
    ("upper_right", "lower_left"),
    ("upper_right", "lower_right"),
)
NOMINAL_GATE_PAIR = ("upper_left", "lower_left")
MAX_ROUTE_COMBINATION_REUSE = {"wait": 3, "avoidance": 1, "reroute": 1}
MAX_SEARCH_ATTEMPTS_PER_ACCEPTED_SCENARIO = 4_000
MIN_COUNTERFACTUALLY_DECISIVE_OBSTACLES = 2


def _route_metrics(route, topology_paths, demo_paths):
    topology_sets = tuple(set(path) for path in topology_paths)
    demo_sets = tuple(set(path) for path in demo_paths)
    cell_frequency: dict[tuple[int, int], int] = {}
    for path in topology_paths:
        for cell in path:
            cell_frequency[cell] = cell_frequency.get(cell, 0) + 1
    topology_coverage = sum(bool(set(route).intersection(path)) for path in topology_sets)
    demo_intersections = [sum(cell in path for cell in route) for path in demo_sets]
    return {
        "center": list(route[len(route) // 2]),
        "orientation": "horizontal" if len({cell[0] for cell in route}) == 1 else "vertical",
        "route": [list(cell) for cell in route],
        "corridor_score": max(cell_frequency.get(cell, 0) for cell in route),
        "corridor_frequency": max(cell_frequency.get(cell, 0) for cell in route) / len(topology_paths),
        "topology_path_coverage_count": topology_coverage,
        "topology_path_coverage_rate": topology_coverage / len(topology_paths),
        "reference_demo_path_coverage_count": sum(value > 0 for value in demo_intersections),
        "reference_demo_intersection_cell_count": sum(demo_intersections),
    }


def _replace_with_anchor(
    problem,
    records,
    *,
    zone: str,
    route: Sequence[tuple[int, int]],
    topology_paths,
    demo_paths,
):
    route = tuple(route)
    if not _valid_route(route, problem, min_length=len(route)):
        raise ValueError(f"Invalid behavior anchor route: {zone} {route}")
    candidates = [record for record in records if record["critical_zone"] == zone]
    source = min(candidates, key=lambda item: int(item["topology_path_coverage_count"]))
    replacement = {
        **source,
        **_route_metrics(route, topology_paths, demo_paths),
        "behavior_anchor": True,
        "functional_role": "verified_behavior_anchor",
    }
    records[records.index(source)] = replacement


def _install_behavior_anchors(problem, config, route_pools, demo_paths):
    spatial = config["spatial_generalization"]
    topology_paths = _reference_demo_paths(
        problem,
        int(spatial["reference_path_count"]),
        int(spatial["generation_seed"]) + 100_000,
    )
    for split in SPLITS:
        for record in route_pools[split]["corridor"]:
            record["behavior_anchor"] = False
    # Training learns waiting at the two upper doors.
    _replace_with_anchor(
        problem,
        route_pools["train"]["corridor"],
        zone="upper_left_gate",
        route=tuple((row, 9) for row in range(6, 11)),
        topology_paths=topology_paths,
        demo_paths=demo_paths,
    )
    _replace_with_anchor(
        problem,
        route_pools["train"]["corridor"],
        zone="upper_right_gate",
        route=tuple((row, 27) for row in range(6, 11)),
        topology_paths=topology_paths,
        demo_paths=demo_paths,
    )
    # Validation checks transfer to a horizontal upper-left approach combined
    # with a vertical lower-right door.  An exhaustive pilot showed that strict
    # waiting exists for this cross-region pair, whereas fixing both lower doors
    # makes explicit STAY no better than spending the same time moving sideways.
    upper_left_approach = next(
        record
        for record in route_pools["validation"]["corridor"]
        if record["critical_zone"] == "upper_left_gate"
        and tuple(record["center"]) == (6, 9)
    )
    upper_left_approach["behavior_anchor"] = True
    upper_left_approach["functional_role"] = "verified_behavior_anchor"
    lower_right = max(
        (
            record
            for record in route_pools["validation"]["corridor"]
            if record["critical_zone"] == "lower_right_gate"
        ),
        key=lambda item: len(set(map(tuple, item["route"])).intersection(GATE_CELLS["lower_right"])),
    )
    lower_right["behavior_anchor"] = True
    lower_right["functional_role"] = "verified_behavior_anchor"
    # Test uses unseen upper-door route extents; these routes were selected by the
    # spatial generator and are not copied from the training pool.
    for zone, gate_cells in (
        ("upper_left_gate", GATE_CELLS["upper_left"]),
        ("upper_right_gate", GATE_CELLS["upper_right"]),
    ):
        anchor = max(
            (record for record in route_pools["test"]["corridor"] if record["critical_zone"] == zone),
            key=lambda item: len(set(map(tuple, item["route"])).intersection(gate_cells)),
        )
        anchor["behavior_anchor"] = True
        anchor["functional_role"] = "verified_behavior_anchor"
    return route_pools


def _route_options(record, move_every: int, demo_paths):
    route = tuple(tuple(cell) for cell in record["route"])
    horizon = max(len(path) - 1 for path in demo_paths)
    options = []
    for start_index in range(len(route)):
        for direction in (-1, 1):
            spec = DynamicObstacleSpec(
                route=route,
                start_index=start_index,
                direction=direction,
                move_every=move_every,
                label=str(record["route_id"]),
                reference_path_source=str(record["critical_zone"]),
            )
            positions = obstacle_positions(spec, horizon)
            mask = 0
            for path_index, path in enumerate(demo_paths):
                if any(
                    path[step] == positions[step - 1] or path[step] == positions[step]
                    for step in range(1, len(path))
                ):
                    mask |= 1 << path_index
            if mask:
                options.append((spec, mask))
    if not options:
        raise ValueError(f"Route {record['route_id']} cannot affect any configured A* demonstration.")
    return options


def _gate_pair_plans(problem, scenario):
    plans = {}
    for pair in GATE_PAIRS:
        blocked = set().union(
            *(cells for name, cells in GATE_CELLS.items() if name not in pair)
        )
        plans[pair] = shortest_safe_plan(
            problem,
            scenario,
            additionally_blocked=blocked,
        )
    return plans


def _plan_cost(plan: SafePlan | None) -> int:
    return 10**9 if plan is None else int(plan.steps)


def _nominal_path_collides(problem, scenario) -> bool:
    horizon = len(problem.nominal_path) - 1
    dynamic_positions = [obstacle_positions(spec, horizon) for spec in scenario.obstacles]
    return any(
        problem.nominal_path[step] in (positions[step - 1], positions[step])
        for positions in dynamic_positions
        for step in range(1, len(problem.nominal_path))
    )


def _classify_behavior(problem, scenario, desired_behavior: str | None = None):
    best = shortest_safe_plan(problem, scenario)
    without_wait = shortest_safe_plan(problem, scenario, allow_wait=False)
    if best is None:
        return None
    no_wait_cost = _plan_cost(without_wait)
    wait_advantage = no_wait_cost - best.steps
    if wait_advantage >= 1 and best.wait_count >= 1:
        behavior = "wait"
    else:
        if desired_behavior == "wait":
            return None
        # Gate-pair planning is unnecessary for the many candidates rejected
        # during wait-scenario search, so defer it until waiting is ruled out.
        pair_plans = _gate_pair_plans(problem, scenario)
        pair_costs = {pair: _plan_cost(plan) for pair, plan in pair_plans.items()}
        nominal_cost = pair_costs[NOMINAL_GATE_PAIR]
        alternative_cost = min(
            cost for pair, cost in pair_costs.items() if pair != NOMINAL_GATE_PAIR
        )
        if wait_advantage == 0 and alternative_cost < nominal_cost:
            behavior = "reroute"
        elif (
            wait_advantage == 0
            and nominal_cost == best.steps
            and _nominal_path_collides(problem, scenario)
            and tuple(best.positions) != tuple(problem.nominal_path)
        ):
            behavior = "avoidance"
        else:
            return None
    if behavior == "wait":
        pair_plans = _gate_pair_plans(problem, scenario)
        pair_costs = {pair: _plan_cost(plan) for pair, plan in pair_plans.items()}
        nominal_cost = pair_costs[NOMINAL_GATE_PAIR]
        alternative_cost = min(
            cost for pair, cost in pair_costs.items() if pair != NOMINAL_GATE_PAIR
        )
    return {
        "behavior": behavior,
        "best_plan": best,
        "no_wait_steps": None if without_wait is None else without_wait.steps,
        "wait_advantage_steps": wait_advantage,
        "pair_costs": pair_costs,
        "nominal_gate_cost": nominal_cost,
        "best_alternative_gate_cost": alternative_cost,
    }


def _counterfactual_relevance(problem, scenario, masks, full_pair_costs):
    full_mask = 0
    for mask in masks:
        full_mask |= mask
    details = []
    for obstacle_index in range(len(scenario.obstacles)):
        reduced_mask = 0
        for index, mask in enumerate(masks):
            if index != obstacle_index:
                reduced_mask |= mask
        unique_demo_contribution = full_mask.bit_count() - reduced_mask.bit_count()
        changed_gate_cost = False
        # A unique blocked demonstration already proves relevance.  Only run the
        # more expensive counterfactual planner when that direct proof is absent.
        if unique_demo_contribution <= 0:
            reduced = DynamicScenario(
                seed=scenario.seed,
                obstacles=tuple(
                    obstacle
                    for index, obstacle in enumerate(scenario.obstacles)
                    if index != obstacle_index
                ),
            )
            for pair in GATE_PAIRS:
                blocked = set().union(
                    *(cells for name, cells in GATE_CELLS.items() if name not in pair)
                )
                reduced_cost = _plan_cost(
                    shortest_safe_plan(
                        problem,
                        reduced,
                        additionally_blocked=blocked,
                    )
                )
                if reduced_cost != full_pair_costs[pair]:
                    changed_gate_cost = True
                    break
        details.append(
            {
                "individual_demo_collision_count": masks[obstacle_index].bit_count(),
                "unique_demo_collision_contribution": unique_demo_contribution,
                "changes_gate_pair_cost": changed_gate_cost,
                "active_demo_blocker": bool(masks[obstacle_index].bit_count() > 0),
                "counterfactually_decisive": bool(
                    unique_demo_contribution > 0 or changed_gate_cost
                ),
            }
        )
    return details


def _candidate_combinations(records, behavior):
    by_zone = {
        zone: [index for index, record in enumerate(records) if record["critical_zone"] == zone]
        for zone in ZONE_COLORS
    }
    combinations = list(itertools.product(*(by_zone[zone] for zone in ZONE_COLORS)))
    if behavior != "wait":
        return combinations
    anchors = {index for index, record in enumerate(records) if record.get("behavior_anchor")}
    return [
        combination
        for combination in combinations
        if len(anchors.intersection(combination)) >= 2
    ]


def _build_split_scenarios(problem, config, split, records, demo_paths):
    spatial = config["spatial_generalization"]
    search_seed = int(spatial.get("behavior_search_seed", int(spatial["generation_seed"]) + 10))
    rng = random.Random(search_seed + 10_000 * (SPLITS.index(split) + 1))
    move_every = int(config["spatial_generalization"]["move_every"])
    options = [_route_options(record, move_every, demo_paths) for record in records]
    scenario_offset = {"train": 0, "validation": 10_000, "test": 20_000}[split]
    accepted = []
    signatures = set()
    for target_behavior in ("wait", "avoidance", "reroute"):
        # Reuse limits are behavior-specific.  A combination used to teach
        # waiting may still be useful with a different phase to teach rerouting.
        combination_use = Counter()
        target_count = BEHAVIOR_COUNTS[split][target_behavior]
        combinations = _candidate_combinations(records, target_behavior)
        if not combinations:
            raise RuntimeError(f"No {split} combinations are available for {target_behavior}.")
        attempts = 0
        created = 0
        rejected = Counter()
        while created < target_count:
            attempts += 1
            if attempts > target_count * MAX_SEARCH_ATTEMPTS_PER_ACCEPTED_SCENARIO:
                raise RuntimeError(
                    f"Could not find enough verified {split} {target_behavior} scenarios; "
                    f"created={created}/{target_count}; rejected={dict(rejected)}."
                )
            if attempts % 100 == 0:
                print(
                    f"[{split} {target_behavior}] searching: accepted={created}/{target_count} "
                    f"attempts={attempts} rejected={dict(rejected)}",
                    flush=True,
                )
            minimum_use = min(combination_use[combination] for combination in combinations)
            eligible_combinations = [
                combination
                for combination in combinations
                if combination_use[combination] == minimum_use
                and combination_use[combination] < MAX_ROUTE_COMBINATION_REUSE[target_behavior]
            ]
            if not eligible_combinations:
                eligible_combinations = [
                    combination
                    for combination in combinations
                    if combination_use[combination] < MAX_ROUTE_COMBINATION_REUSE[target_behavior]
                ]
            if not eligible_combinations:
                raise RuntimeError(
                    f"{split} exhausted route combinations while building {target_behavior}."
                )
            combination = rng.choice(eligible_combinations)
            chosen = [rng.choice(options[record_index]) for record_index in combination]
            specs = tuple(item[0] for item in chosen)
            masks = tuple(int(item[1]) for item in chosen)
            signature = tuple(
                (spec.label, spec.start_index, spec.direction) for spec in specs
            )
            if signature in signatures:
                rejected["duplicate"] += 1
                continue
            scenario = DynamicScenario(
                seed=scenario_offset + len(accepted),
                obstacles=specs,
            )
            classification = _classify_behavior(
                problem,
                scenario,
                desired_behavior=target_behavior,
            )
            if classification is None:
                rejected["unclassified"] += 1
                continue
            if classification["behavior"] != target_behavior:
                rejected[f"classified_{classification['behavior']}"] += 1
                continue
            relevance = _counterfactual_relevance(
                problem,
                scenario,
                masks,
                classification["pair_costs"],
            )
            if not all(item["active_demo_blocker"] for item in relevance):
                rejected["not_all_five_active_demo_blockers"] += 1
                continue
            decisive_count = sum(
                item["counterfactually_decisive"] for item in relevance
            )
            if decisive_count < MIN_COUNTERFACTUALLY_DECISIVE_OBSTACLES:
                rejected["fewer_than_two_counterfactually_decisive"] += 1
                continue
            full_mask = 0
            for mask in masks:
                full_mask |= mask
            plan = classification["best_plan"]
            obstacle_rows = []
            for spec, relevance_row in zip(specs, relevance):
                obstacle_rows.append(
                    {
                        "route_id": spec.label,
                        "start_index": spec.start_index,
                        "direction": spec.direction,
                        "move_every": spec.move_every,
                        **relevance_row,
                    }
                )
            accepted.append(
                {
                    "scenario_id": scenario.seed,
                    "difficulty_stratum": target_behavior,
                    "required_behavior": target_behavior,
                    "reference_demo_collision_count": full_mask.bit_count(),
                    "reference_demo_collision_rate": full_mask.bit_count() / len(demo_paths),
                    "minimum_safe_path_steps": plan.steps,
                    "safe_detour_steps": plan.steps - (len(problem.nominal_path) - 1),
                    "oracle_wait_count": plan.wait_count,
                    "no_wait_safe_path_steps": classification["no_wait_steps"],
                    "wait_advantage_steps": classification["wait_advantage_steps"],
                    "nominal_gate_cost": classification["nominal_gate_cost"],
                    "best_alternative_gate_cost": classification["best_alternative_gate_cost"],
                    "oracle_path": [list(cell) for cell in plan.positions],
                    "obstacles": obstacle_rows,
                }
            )
            signatures.add(signature)
            combination_use[combination] += 1
            created += 1
            if created == 1 or created % 5 == 0 or created == target_count:
                print(
                    f"[{split} {target_behavior}] {created}/{target_count} "
                    f"attempts={attempts} rejected={dict(rejected)}",
                    flush=True,
                )
    return accepted


def build_manifest(problem, config):
    demo_count = int(config["demonstrations"]["episodes"])
    demo_seed = int(config["demonstrations"]["seed"])
    if demo_count != 20:
        raise ValueError("Office behavior v6 is calibrated against exactly 20 A* demonstrations.")
    demo_paths = _reference_demo_paths(problem, demo_count, demo_seed)
    route_pools = _build_route_pools(problem, config, demo_paths)
    _install_behavior_anchors(problem, config, route_pools, demo_paths)
    scenarios = {
        split: _build_split_scenarios(
            problem,
            config,
            split,
            route_pools[split]["corridor"],
            demo_paths,
        )
        for split in SPLITS
    }
    spatial = config["spatial_generalization"]
    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "office_behavior_verified_v6",
            "seed": int(spatial["generation_seed"]),
            "behavior_search_seed": int(
                spatial.get("behavior_search_seed", int(spatial["generation_seed"]) + 10)
            ),
            "route_length": int(spatial["route_length"]),
            "reference_path_count": int(spatial["reference_path_count"]),
            "reference_demo_episodes": demo_count,
            "reference_demo_seed": demo_seed,
            "separation_radius": int(spatial["separation_radius"]),
            "move_every": int(spatial["move_every"]),
            "corridor_per_scenario": 5,
            "background_per_scenario": 0,
            "cross_split_spatial_disjoint": False,
            "split_scope": "unseen_route_phase_combinations_on_shared_office_topology",
            "behavior_counts": BEHAVIOR_COUNTS,
            "behavior_definitions": {
                "wait": "shortest_safe_plan_is_strictly_shorter_with_wait_action",
                "avoidance": "no_wait_penalty_and_nominal_gate_pair_remains_optimal_but_nominal_path_collides",
                "reroute": "no_wait_penalty_and_an_alternative_gate_pair_is_strictly_shorter_than_nominal_gate_pair",
            },
            "obstacle_activity_rule": "each_obstacle_temporally_collides_with_at_least_one_of_the_20_actual_astar_demonstrations",
            "counterfactual_rule": "at_least_two_obstacles_uniquely_block_a_demo_or_change_a_gate_pair_safe_cost",
        },
        "route_pools": route_pools,
        "scenarios": scenarios,
    }
    validate_spatial_scenario_manifest(problem, manifest)
    return manifest


def _next_cell(route, start_index, direction):
    candidate = start_index + direction
    if candidate < 0 or candidate >= len(route):
        candidate = start_index - direction
    return route[candidate]


def render_route_pools(problem, manifest, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=180)
    for axis, split in zip(axes, SPLITS):
        records = manifest["route_pools"][split]["corridor"]
        _draw_base(axis, problem, f"{split.title()}: {len(records)} critical routes", ticks=True)
        for record in records:
            route = record["route"]
            color = ZONE_COLORS[record["critical_zone"]]
            width = 3.4 if record.get("behavior_anchor") else 1.8
            axis.plot(
                [cell[1] for cell in route],
                [cell[0] for cell in route],
                color=color,
                linewidth=width,
                alpha=0.95,
            )
            row, column = record["center"]
            marker = "D" if record.get("behavior_anchor") else "o"
            axis.scatter(column, row, color=color, marker=marker, s=22, zorder=5)
    legend = [
        Line2D([0], [0], color=color, linewidth=2.3, label=ZONE_LABELS[zone])
        for zone, color in ZONE_COLORS.items()
    ]
    legend.append(Line2D([0], [0], marker="D", color="#424242", linestyle="", label="behavior anchor"))
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Office v6 route pools: critical zones only; thick routes anchor verified behavior", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_gallery(problem, manifest, split: str, output: Path):
    scenarios = manifest["scenarios"][split]
    lookup = {
        record["route_id"]: record
        for record in manifest["route_pools"][split]["corridor"]
    }
    columns = 10 if len(scenarios) >= 50 else 5
    rows = math.ceil(len(scenarios) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.7 * rows), dpi=140)
    axes = np.asarray(axes).reshape(-1)
    for axis, source in zip(axes, scenarios):
        behavior = source["required_behavior"]
        title = (
            f"{source['scenario_id']} | {behavior.upper()} | "
            f"wait={source['oracle_wait_count']} | demo={source['reference_demo_collision_count']}/20"
        )
        _draw_base(axis, problem, title)
        oracle_path = source["oracle_path"]
        axis.plot(
            [cell[1] for cell in oracle_path],
            [cell[0] for cell in oracle_path],
            color=BEHAVIOR_COLORS[behavior],
            linewidth=1.25,
            linestyle="--",
            alpha=0.85,
        )
        wait_cells = [
            oracle_path[index]
            for index in range(1, len(oracle_path))
            if oracle_path[index] == oracle_path[index - 1]
        ]
        if wait_cells:
            axis.scatter(
                [cell[1] for cell in wait_cells],
                [cell[0] for cell in wait_cells],
                marker="s",
                color=BEHAVIOR_COLORS[behavior],
                edgecolors="white",
                linewidths=0.4,
                s=22,
                zorder=8,
            )
        for obstacle in source["obstacles"]:
            record = lookup[obstacle["route_id"]]
            route = record["route"]
            color = ZONE_COLORS[record["critical_zone"]]
            axis.plot([cell[1] for cell in route], [cell[0] for cell in route], color=color, linewidth=1.5)
            initial = route[int(obstacle["start_index"])]
            next_cell = _next_cell(route, int(obstacle["start_index"]), int(obstacle["direction"]))
            axis.scatter(initial[1], initial[0], color="#d32f2f", edgecolors="white", linewidths=0.3, s=15, zorder=7)
            axis.annotate(
                "",
                xy=(next_cell[1], next_cell[0]),
                xytext=(initial[1], initial[0]),
                arrowprops={"arrowstyle": "->", "color": "#212121", "lw": 0.65},
            )
    for axis in axes[len(scenarios):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Office v6 {split}: verified behavior | dashed=oracle safe path, square=wait",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _write_audit(manifest, output: Path):
    rows = []
    summaries = []
    for split in SPLITS:
        for source in manifest["scenarios"][split]:
            rows.append(
                {
                    "split": split,
                    "scenario_id": source["scenario_id"],
                    "required_behavior": source["required_behavior"],
                    "reference_demo_collision_count": source["reference_demo_collision_count"],
                    "minimum_safe_path_steps": source["minimum_safe_path_steps"],
                    "safe_detour_steps": source["safe_detour_steps"],
                    "oracle_wait_count": source["oracle_wait_count"],
                    "no_wait_safe_path_steps": source["no_wait_safe_path_steps"],
                    "wait_advantage_steps": source["wait_advantage_steps"],
                    "nominal_gate_cost": source["nominal_gate_cost"],
                    "best_alternative_gate_cost": source["best_alternative_gate_cost"],
                    "active_demo_blocker_count": sum(
                        item["active_demo_blocker"] for item in source["obstacles"]
                    ),
                    "counterfactually_decisive_obstacle_count": sum(
                        item["counterfactually_decisive"] for item in source["obstacles"]
                    ),
                }
            )
        group = [row for row in rows if row["split"] == split]
        summaries.append(
            {
                "split": split,
                "scenario_count": len(group),
                "behavior_counts": dict(Counter(row["required_behavior"] for row in group)),
                "all_five_active_demo_blockers_rate": sum(
                    row["active_demo_blocker_count"] == 5 for row in group
                ) / len(group),
                "mean_counterfactually_decisive_obstacles": sum(
                    row["counterfactually_decisive_obstacle_count"] for row in group
                ) / len(group),
                "mean_safe_detour_steps": sum(row["safe_detour_steps"] for row in group) / len(group),
                "mean_oracle_wait_count_in_wait_scenarios": sum(
                    row["oracle_wait_count"] for row in group if row["required_behavior"] == "wait"
                ) / max(1, sum(row["required_behavior"] == "wait" for row in group)),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scenario_behavior_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(summaries, output / "split_summary.json")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-output", default="outputs/office_behavior_v6_design")
    args = parser.parse_args()
    config = load_config(_resolve(args.config))
    problem = _load_problem(config)
    manifest = build_manifest(problem, config)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(spatial["manifest"])
    write_json(manifest, manifest_path)
    render_route_pools(problem, manifest, _resolve(spatial["route_pool_preview"]))
    for split, key in (("train", "training_preview"), ("validation", "validation_preview"), ("test", "test_preview")):
        render_gallery(problem, manifest, split, _resolve(spatial[key]))
    summaries = _write_audit(manifest, _resolve(args.audit_output))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"manifest={manifest_path} sha256={digest[:12]}")
    for summary in summaries:
        print(
            f"{summary['split']}: scenarios={summary['scenario_count']} "
            f"behaviors={summary['behavior_counts']} "
            f"all5_active={summary['all_five_active_demo_blockers_rate']:.1%} "
            f"mean_decisive={summary['mean_counterfactually_decisive_obstacles']:.2f} "
            f"mean_detour={summary['mean_safe_detour_steps']:.2f} "
            f"waits_in_wait={summary['mean_oracle_wait_count_in_wait_scenarios']:.2f}"
        )


if __name__ == "__main__":
    main()
