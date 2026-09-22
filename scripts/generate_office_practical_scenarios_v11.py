"""Build practical Office training data and a small causal diagnostic subset.

Training/validation scenes use useful, observable interaction witnesses rather
than requiring one exclusive behavior. Only ten test conflicts receive strict
phase-matched controls. The static Office map is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import generate_office_behavior_scenarios_v10 as v10
from generate_office_behavior_scenarios_v6 import GATE_CELLS, GATE_PAIRS, _route_metrics
from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec, _valid_route
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import SPLITS, validate_spatial_scenario_manifest
from astar_d3qn.evaluation.behavior_oracle import SafePlan, shortest_safe_plan
from astar_d3qn.evaluation.conflict import obstacle_positions
from astar_d3qn.evaluation.spatial_witness import motion_summary, static_bypass_audit
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json

v9 = v10.v9
DEFAULT_CONFIG = "configs/dynamic_spatial_generalization_office_practical_v11.yaml"
ZONE_ORDER = tuple(v10.ZONE_COLORS)
BEHAVIORS = ("normal", "wait", "avoidance", "reroute")
COLORS = {"normal": "#546e7a", "wait": "#7b1fa2", "avoidance": "#00897b", "reroute": "#ef6c00"}

# Coordinates are (row, column). Straight gate routes create sustained blockage
# for rerouting; turning gate-clearance routes vacate a doorway for waiting;
# approach routes cross open portions of paths for local avoidance.
ANCHORS = {
    "upper_left_gate": (
        ("bottleneck", "bottleneck", (10, 9), "vertical"),
        ("approach_before", "open_approach", (7, 9), "horizontal"),
        ("approach_after", "open_approach", (14, 9), "horizontal"),
    ),
    "upper_right_gate": (
        ("bottleneck", "bottleneck", (10, 27), "vertical"),
        ("approach_before", "open_approach", (7, 27), "horizontal"),
        ("approach_after", "open_approach", (14, 27), "horizontal"),
    ),
    "lower_left_gate": (
        ("bottleneck", "bottleneck", (25, 17), "vertical"),
        ("approach_before", "open_approach", (22, 17), "horizontal"),
        ("approach_after", "open_approach", (29, 17), "horizontal"),
    ),
    "lower_right_gate": (
        ("bottleneck", "bottleneck", (25, 31), "vertical"),
        ("approach_before", "open_approach", (22, 31), "horizontal"),
        ("approach_after", "open_approach", (29, 34), "horizontal"),
    ),
    "goal_approach": (
        ("goal_west", "open_approach", (35, 18), "vertical"),
        ("goal_middle", "open_approach", (35, 24), "vertical"),
        ("goal_east", "open_approach", (35, 34), "vertical"),
    ),
}


def route_key(route):
    cells = tuple(map(tuple, route))
    return min(cells, tuple(reversed(cells)))


def _gate_pair(path):
    cells = set(map(tuple, path))
    upper = [name for name in ("upper_left", "upper_right") if cells.intersection(GATE_CELLS[name])]
    lower = [name for name in ("lower_left", "lower_right") if cells.intersection(GATE_CELLS[name])]
    return (upper[0], lower[0]) if len(upper) == len(lower) == 1 else None


def _blocked_gates(pair):
    return set().union(*(cells for name, cells in GATE_CELLS.items() if name not in pair))


def _candidate_routes(problem, topology_paths, demo_paths):
    candidates = []
    bypass_cache = {}
    gate_cells_by_zone = {
        "upper_left_gate": GATE_CELLS["upper_left"],
        "upper_right_gate": GATE_CELLS["upper_right"],
        "lower_left_gate": GATE_CELLS["lower_left"],
        "lower_right_gate": GATE_CELLS["lower_right"],
        "goal_approach": frozenset(),
    }
    gate_name_by_zone = {
        "upper_left_gate": "upper_left",
        "upper_right_gate": "upper_right",
        "lower_left_gate": "lower_left",
        "lower_right_gate": "lower_right",
    }
    demo_pairs = [_gate_pair(path) for path in demo_paths]
    seen = set()
    for zone, anchors in ANCHORS.items():
        for anchor_name, role, anchor, orientation in anchors:
            dr, dc = ((1, 0) if orientation == "vertical" else (0, 1))
            for row in range(anchor[0] - 1, anchor[0] + 2):
                for column in range(anchor[1] - 1, anchor[1] + 2):
                    for length in (3, 5, 7):
                        route = tuple((row + dr * offset, column + dc * offset)
                                      for offset in range(-(length // 2), length // 2 + 1))
                        key = route_key(route)
                        if key in seen or not _valid_route(route, problem, min_length=length):
                            continue
                        gate_intersection = set(route).intersection(gate_cells_by_zone[zone])
                        if role == "bottleneck" and not gate_intersection:
                            continue
                        if role == "open_approach" and any(set(route).intersection(cells) for cells in GATE_CELLS.values()):
                            continue
                        metrics = _route_metrics(route, topology_paths, demo_paths)
                        if int(metrics["corridor_score"]) <= 0:
                            continue
                        eligible = []
                        bypass_cells = {}
                        for index, path in enumerate(demo_paths):
                            intersections = sorted(set(route).intersection(path))
                            if not intersections or demo_pairs[index] is None:
                                continue
                            if role == "bottleneck":
                                eligible.append(index)
                                continue
                            pair = demo_pairs[index]
                            feasible = []
                            for cell in intersections:
                                cache_key = (pair, cell)
                                if cache_key not in bypass_cache:
                                    bypass_cache[cache_key] = static_bypass_audit(
                                        replace(problem, nominal_path=tuple(path)), cell,
                                        additionally_blocked=_blocked_gates(pair),
                                    )
                                if bypass_cache[cache_key]["status"] == "static_bypass_possible":
                                    feasible.append(cell)
                            if feasible:
                                eligible.append(index)
                                bypass_cells[str(index)] = [list(cell) for cell in feasible]
                        if not eligible:
                            continue
                        seen.add(key)
                        candidates.append({
                            **metrics, "route": [list(cell) for cell in route],
                            "critical_zone": zone, "functional_role": role,
                            "anchor_name": anchor_name, "design_anchor": list(anchor),
                            "anchor_distance": max(abs(row - anchor[0]), abs(column - anchor[1])),
                            "eligible_reference_indices": eligible,
                            "statically_bypassable_reference_cells": bypass_cells,
                        })
    # A straight patrol spanning a one-cell-wide gate never leaves enough room
    # for an agent and obstacle to pass one another. It is a useful reroute
    # blocker, but not a waiting interaction. Add a separate family that enters
    # the two-cell gate and then turns into the open room below it, periodically
    # vacating the doorway completely.
    for zone, gate_cells in gate_cells_by_zone.items():
        if zone == "goal_approach":
            continue
        ordered = sorted(gate_cells)
        if len(ordered) != 2 or ordered[0][1] != ordered[1][1]:
            raise RuntimeError(f"{zone} does not have the expected two-cell vertical gate.")
        entrance, exit_cell = ordered
        clearance_cell = (exit_cell[0] + 1, exit_cell[1])
        for side in (-1, 1):
            for tail_length in (2, 3, 4):
                route = (entrance, exit_cell, clearance_cell) + tuple(
                    (clearance_cell[0], clearance_cell[1] + side * offset)
                    for offset in range(1, tail_length + 1)
                )
                key = route_key(route)
                if key in seen or not _valid_route(route, problem, min_length=len(route)):
                    continue
                metrics = _route_metrics(route, topology_paths, demo_paths)
                gate_name = gate_name_by_zone[zone]
                eligible = [
                    index for index, path in enumerate(demo_paths)
                    if demo_pairs[index] is not None
                    and gate_name in demo_pairs[index]
                    and set(gate_cells).intersection(path)
                ]
                if not eligible:
                    continue
                seen.add(key)
                candidates.append({
                    **metrics,
                    "orientation": "turning",
                    "route": [list(cell) for cell in route],
                    "critical_zone": zone,
                    "functional_role": "wait_gate",
                    "anchor_name": "wait_clearance",
                    "design_anchor": list(clearance_cell),
                    "anchor_distance": 0,
                    "eligible_reference_indices": eligible,
                    "statically_bypassable_reference_cells": {},
                })
    return candidates, {
        "candidate_count": len(candidates),
        "by_zone_role": {f"{zone}:{role}": sum(
            row["critical_zone"] == zone and row["functional_role"] == role for row in candidates)
            for zone in ZONE_ORDER for role in ("wait_gate", "bottleneck", "open_approach")},
        "scope": "bounded straight approach/blocking routes plus gate-clearing turns; all intersect at least one A* demonstration",
    }


def _route_requirements(split, zone):
    if zone == "goal_approach":
        # Only one legal demo-intersecting route exists at goal_west. Keep that
        # unique geometry in training, then use distinct middle/east variants
        # for held-out validation and test rather than copying it across splits.
        return {
            "train": ("goal_west", "goal_middle", "goal_east"),
            "validation": ("goal_middle", "goal_east"),
            "test": ("goal_middle", "goal_middle", "goal_east"),
        }[split]
    if split == "train":
        return ("wait_clearance", "bottleneck", "approach_before", "approach_after")
    if split == "validation":
        return ("wait_clearance", "bottleneck", "approach_before")
    return ("wait_clearance", "bottleneck", "approach_after")


def _select_route_pools(candidates, seed, max_trials):
    """Jointly assign all functional routes without order-dependent failures."""

    rng = random.Random(seed)
    tie_break = {id(row): rng.random() for row in candidates}
    by_group = {
        (split, zone, anchor): [
            row for row in candidates
            if row["critical_zone"] == zone and row["anchor_name"] == anchor
        ]
        for split in SPLITS
        for zone in ZONE_ORDER
        for anchor in _route_requirements(split, zone)
    }
    missing = [key for key, rows in by_group.items() if not rows]
    if missing:
        raise RuntimeError(f"Functional route candidates are missing before assignment: {missing}.")

    def route_rank(row):
        # Five cells is the preferred motion extent; length 3/7 remain legal
        # alternatives when needed to keep neighboring functional routes apart.
        return (
            abs(len(row["route"]) - 5),
            int(row["anchor_distance"]),
            -len(row["eligible_reference_indices"]),
            -int(row["topology_path_coverage_count"]),
            tie_break[id(row)],
            route_key(row["route"]),
        )

    selected = {}
    search = {"trials": 0, "started": time.monotonic()}
    print(
        f"[route assignment] candidates={len(candidates)} zones={len(ZONE_ORDER)} "
        f"max_trials={max_trials}",
        flush=True,
    )

    def compatible_combinations(split, zone, used_geometries):
        anchors = _route_requirements(split, zone)

        def extend(position, chosen, chosen_geometries):
            if position == len(anchors):
                yield tuple(chosen)
                return
            key = (split, zone, anchors[position])
            for row in sorted(by_group[key], key=route_rank):
                search["trials"] += 1
                if search["trials"] > max_trials:
                    raise RuntimeError(
                        "Route assignment reached its bounded trial limit "
                        f"({max_trials}) at {split}:{zone}:{anchors[position]}."
                    )
                if search["trials"] % 10000 == 0:
                    print(
                        f"[route assignment] trials={search['trials']} "
                        f"zone={zone} split={split} "
                        f"elapsed={time.monotonic() - search['started']:.1f}s",
                        flush=True,
                    )
                geometry = route_key(row["route"])
                if geometry in used_geometries or geometry in chosen_geometries:
                    continue
                yield from extend(
                    position + 1,
                    chosen + [row],
                    chosen_geometries.union((geometry,)),
                )

        # Routes inside one functional zone are alternatives: every scenario
        # selects exactly one route from that zone. They may therefore share a
        # physical gate, while exact geometry remains unique across all splits.
        return extend(0, [], set())

    # Anchor regions are spatially separate by construction. Solve the three
    # split allocations jointly inside each zone, instead of taking a Cartesian
    # product with choices from unrelated zones.
    split_order = ("train", "test", "validation")
    for zone_index, zone in enumerate(ZONE_ORDER):
        used_geometries = set()
        announced = set()

        def assign_split(split_index):
            if split_index == len(split_order):
                return True
            split = split_order[split_index]
            if split not in announced:
                announced.add(split)
                print(
                    f"[route assignment] zone={zone_index + 1}/{len(ZONE_ORDER)} "
                    f"{zone} checking={split}",
                    flush=True,
                )
            for combination in compatible_combinations(split, zone, used_geometries):
                geometries = {route_key(row["route"]) for row in combination}
                selected[(split, zone)] = combination
                used_geometries.update(geometries)
                if assign_split(split_index + 1):
                    return True
                used_geometries.difference_update(geometries)
                del selected[(split, zone)]
            return False

        if not assign_split(0):
            counts = {
                anchor: len(by_group[("train", zone, anchor)])
                for anchor in sorted({
                    name for split in SPLITS
                    for name in _route_requirements(split, zone)
                })
            }
            raise RuntimeError(
                f"No joint split assignment exists for {zone}; "
                f"anchor candidate counts={counts}, trials={search['trials']}."
            )
    all_geometries = []
    for split in SPLITS:
        occupied = set()
        for zone in ZONE_ORDER:
            zone_cells = set()
            for row in selected[(split, zone)]:
                cells = set(map(tuple, row["route"]))
                zone_cells.update(cells)
                all_geometries.append(route_key(row["route"]))
            if occupied.intersection(zone_cells):
                raise RuntimeError(
                    f"Functional anchor regions overlap across zones in {split}."
                )
            occupied.update(zone_cells)
    if len(all_geometries) != len(set(all_geometries)):
        raise RuntimeError("Whole route geometry was reused across split assignments.")
    print(
        f"[route assignment] complete trials={search['trials']} "
        f"elapsed={time.monotonic() - search['started']:.1f}s",
        flush=True,
    )

    pools = {split: {"corridor": [], "background": []} for split in SPLITS}
    for split in SPLITS:
        for zone in ZONE_ORDER:
            for source in selected[(split, zone)]:
                number = len(pools[split]["corridor"]) + 1
                pools[split]["corridor"].append({
                    **source,
                    "route_id": f"{split}_functional_{number:02d}",
                    "category": "corridor",
                    "alternative_group": source["critical_zone"],
                    "split_variant": number,
                    "placement_reason": (
                        "turns out of the doorway to create a genuine wait window"
                        if source["functional_role"] == "wait_gate"
                        else (
                            "sustained single-lane blockage for rerouting"
                            if source["functional_role"] == "bottleneck"
                            else "open area with a statically verified lateral bypass"
                        )
                    ),
                })
    return pools


def build_route_pools(problem, config, demo_paths):
    spatial = config["spatial_generalization"]
    topology_paths = v9._reference_demo_paths(
        problem, int(spatial["reference_path_count"]), int(spatial["generation_seed"]) + 100_000)
    candidates, audit = _candidate_routes(problem, topology_paths, demo_paths)
    pools = _select_route_pools(
        candidates,
        int(spatial["generation_seed"]),
        int(spatial["max_route_assignment_trials"]),
    )
    expected = spatial["route_pool_counts"]
    for split in SPLITS:
        for category in ("corridor", "background"):
            if len(pools[split][category]) != int(expected[split][category]):
                raise ValueError(f"{split} {category} route count disagrees with config.")
    return pools, candidates, audit


def validate_config_contract(config):
    """Reject internally inconsistent registrations before an expensive search."""

    spatial = config["spatial_generalization"]
    design = spatial["scenario_design"]
    if int(spatial["obstacle_count"]) != 5 or (
        int(spatial["corridor_per_scenario"]), int(spatial["background_per_scenario"])
    ) != (5, 0):
        raise ValueError("Office v11 requires five functional-route obstacles per scene.")
    route_lengths = tuple(int(value) for value in spatial["candidate_route_lengths"])
    if route_lengths != (3, 5, 7):
        raise ValueError("Office v11 candidate_route_lengths must remain [3, 5, 7].")
    if int(spatial["max_route_assignment_trials"]) <= 0:
        raise ValueError("max_route_assignment_trials must be positive.")
    expected_route_counts = {
        split: sum(len(_route_requirements(split, zone)) for zone in ZONE_ORDER)
        for split in SPLITS
    }
    for split in SPLITS:
        pool = spatial["route_pool_counts"][split]
        if int(pool["corridor"]) != expected_route_counts[split] or int(pool["background"]) != 0:
            raise ValueError(f"{split} route_pool_counts disagree with the functional blueprint.")
        practical_total = sum(int(value) for value in design["practical_counts"][split].values())
        pair_scenes = 0
        if split == "test":
            pair_scenes = 2 * sum(int(value) for value in design["diagnostic_pair_counts"].values())
        if practical_total + pair_scenes != int(spatial["scenario_counts"][split]):
            raise ValueError(f"{split} scenario counts disagree with the practical/pair quotas.")
    fractions = tuple(float(design["difficulty_fractions"][name]) for name in ("easy", "medium", "hard"))
    if any(value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("difficulty_fractions must be positive and sum to one.")
    if any(int(value) <= 0 for value in design["move_every_choices"]):
        raise ValueError("move_every_choices must contain positive integers.")
    for key in (
        "minimum_wait_advantage_steps", "minimum_reroute_margin_steps",
        "maximum_visible_avoidance_extra_steps", "max_attempts_per_requested_scene",
        "max_attempts_without_acceptance",
    ):
        if int(design[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if int(config["demonstrations"]["episodes"]) != 20:
        raise ValueError("Office v11 is registered against exactly 20 static A* demonstrations.")


def _route_options(records, speeds, demo_paths):
    return {row["route_id"]: v10._route_options_for_speeds(row, speeds, demo_paths) for row in records}


def _signature(specs):
    return tuple(sorted((spec.label, spec.start_index, spec.direction, spec.move_every) for spec in specs))


def _spec_collides_path(spec, path):
    """Return whether one obstacle conflicts with an explicitly timed path."""

    path = tuple(map(tuple, path))
    positions = obstacle_positions(spec, len(path) - 1)
    return any(
        path[step] in (positions[step - 1], positions[step])
        for step in range(1, len(path))
    )


def _balanced_reference_safe_option(
    routes,
    options_by_route,
    reference_index,
    route_use,
    rng,
    required_safe_path=None,
):
    """Choose a least-used route only after proving its phase is harmless."""

    reference_bit = 1 << reference_index
    feasible = []
    for route in routes:
        safe = [
            item for item in options_by_route[route["route_id"]]
            if not (int(item[1]) & reference_bit)
            and (
                required_safe_path is None
                or not _spec_collides_path(item[0], required_safe_path)
            )
        ]
        if safe:
            feasible.append((route, safe))
    if not feasible:
        return None
    minimum_use = min(route_use[route["route_id"]] for route, _ in feasible)
    balanced = [
        (route, safe) for route, safe in feasible
        if route_use[route["route_id"]] == minimum_use
    ]
    route, safe = rng.choice(balanced)
    return rng.choice(safe)


def _gate_plans(problem, scenario, *, allow_wait):
    return {pair: shortest_safe_plan(problem, scenario, allow_wait=allow_wait,
                                     additionally_blocked=_blocked_gates(pair)) for pair in GATE_PAIRS}


def _plan_cost(plan):
    return math.inf if plan is None else int(plan.steps)


def _first_primary_conflict(problem, scenario, primary_index=0):
    primary = obstacle_positions(scenario.obstacles[primary_index], len(problem.nominal_path) - 1)
    return next((cell for step, cell in enumerate(problem.nominal_path)
                 if step and cell in (primary[step - 1], primary[step])), None)


def classify_practical(
    problem,
    scenario,
    behavior,
    reference_path,
    radius,
    design,
    diagnostics=None,
):
    """Return a reproducible witness, or record the exact rejection reason.

    The behavior name describes one verified option available in the scene.  It
    is deliberately not interpreted as the only acceptable policy response.
    """

    def reject(reason):
        if diagnostics is not None:
            diagnostics[reason] += 1
        return None

    minimum_wait_advantage = int(design["minimum_wait_advantage_steps"])
    minimum_reroute_margin = int(design["minimum_reroute_margin_steps"])
    maximum_avoidance_extra = int(design["maximum_visible_avoidance_extra_steps"])
    context = replace(problem, nominal_path=tuple(reference_path))
    pair = _gate_pair(reference_path)
    if pair is None:
        return reject("reference_has_no_unique_gate_pair")
    reference_collision = v9._nominal_path_collides(context, scenario)
    best = shortest_safe_plan(context, scenario)
    if best is None:
        return reject("no_safe_plan")
    # Compute only the plans needed to reject a candidate.  Complete diagnostic
    # metrics are filled after it passes its behavior contract.  The previous
    # eager version solved all eight gate-pair variants for every rejected
    # candidate and made a failed 2,000-attempt search take over twenty minutes.
    no_wait = None
    pair_no_wait = None
    reference_plan = None
    reference_no_wait = None
    alternative_pair, alternative = None, None
    witness = best
    observation = {"decision_step": None, "decision_position": None,
                   "nearest_dynamic_distance_at_decision": None,
                   "observable_decisive_obstacle_index": None}
    conflict_cell = None
    spatial_check = None
    if behavior == "normal":
        if reference_collision:
            return reject("control_reference_collision")
        if best.steps != len(reference_path) - 1:
            return reject("control_not_static_shortest_cost")
        if best.wait_count:
            return reject("control_requires_wait")
        witness = SafePlan(tuple(reference_path), len(reference_path) - 1, 0)
    elif not reference_collision:
        return reject("target_has_no_reference_collision")
    elif behavior == "wait":
        blocked = _blocked_gates(pair)
        reference_plan = shortest_safe_plan(
            context, scenario, additionally_blocked=blocked)
        if reference_plan is None:
            return reject("wait_no_reference_gate_plan")
        reference_no_wait = shortest_safe_plan(
            context, scenario, allow_wait=False, additionally_blocked=blocked)
        if reference_plan.wait_count < 1:
            return reject("wait_witness_has_no_stay")
        # A missing no-STAY plan is the strongest possible evidence that an
        # explicit wait is useful on this gate pair, not a reason to reject the
        # scene.  This matches the established v9 behavior contract.  Only
        # apply the finite step threshold when a no-STAY route actually exists.
        advantage = (
            None
            if reference_no_wait is None
            else int(reference_no_wait.steps - reference_plan.steps)
        )
        if advantage is not None and advantage < minimum_wait_advantage:
            return reject("wait_advantage_below_threshold")
        if best.steps != reference_plan.steps:
            return reject("wait_reference_gate_not_globally_optimal")
        witness = reference_plan
        observation = v9._decision_observability(context, scenario, witness, "wait", radius, (0,))
        if observation is None:
            return reject("wait_primary_not_visible_at_decision")
    elif behavior == "avoidance":
        conflict_cell = _first_primary_conflict(context, scenario)
        if conflict_cell is None:
            return reject("avoidance_has_no_primary_conflict")
        spatial_check = static_bypass_audit(
            context, conflict_cell, additionally_blocked=_blocked_gates(pair))
        if spatial_check["status"] != "static_bypass_possible":
            return reject("avoidance_conflict_has_no_static_bypass")
        witness = shortest_safe_plan(
            context, scenario, allow_wait=False, additionally_blocked=_blocked_gates(pair),
            visible_deviation_obstacle_index=0, observation_radius=radius)
        if witness is None:
            return reject("avoidance_no_visible_same_gate_no_wait_plan")
        if tuple(witness.positions) == tuple(reference_path):
            return reject("avoidance_witness_does_not_deviate")
        motion = motion_summary(witness.positions)
        if not motion["is_simple_path"]:
            return reject("avoidance_witness_repeats_cells")
        if witness.steps - best.steps > maximum_avoidance_extra:
            return reject("avoidance_extra_cost_above_threshold")
        observation = v9._decision_observability(context, scenario, witness, "avoidance", radius, (0,))
        if observation is None:
            return reject("avoidance_primary_not_visible_at_decision")
    elif behavior == "reroute":
        pair_no_wait = _gate_plans(context, scenario, allow_wait=False)
        reference_no_wait = pair_no_wait[pair]
        alternatives = [
            (name, plan) for name, plan in pair_no_wait.items()
            if name != pair and plan is not None
        ]
        alternative_pair, alternative = (
            min(alternatives, key=lambda item: item[1].steps)
            if alternatives else (None, None)
        )
        if reference_no_wait is None:
            return reject("reroute_no_reference_gate_no_wait_plan")
        if alternative is None:
            return reject("reroute_no_alternative_gate_plan")
        if reference_no_wait.steps - alternative.steps < minimum_reroute_margin:
            return reject("reroute_margin_below_threshold")
        if best.steps != alternative.steps:
            return reject("reroute_alternative_not_globally_optimal")
        witness = alternative
        observation = v9._decision_observability(context, scenario, witness, "reroute", radius, (0,))
        if observation is None:
            return reject("reroute_primary_not_visible_at_decision")
    else:
        raise ValueError(f"Unknown practical behavior {behavior!r}.")

    # The candidate is accepted.  Compute the remaining values needed in the
    # manifest without paying this cost for every rejected proposal.
    no_wait = shortest_safe_plan(context, scenario, allow_wait=False)
    if reference_plan is None:
        reference_plan = shortest_safe_plan(
            context, scenario, additionally_blocked=_blocked_gates(pair))
    if pair_no_wait is None:
        pair_no_wait = _gate_plans(context, scenario, allow_wait=False)
    reference_no_wait = pair_no_wait[pair]
    alternatives = [
        (name, plan) for name, plan in pair_no_wait.items()
        if name != pair and plan is not None
    ]
    alternative_pair, alternative = (
        min(alternatives, key=lambda item: item[1].steps)
        if alternatives else (None, None)
    )
    global_wait_advantage = None if no_wait is None else int(no_wait.steps - best.steps)
    reference_wait_advantage = (
        None
        if reference_plan is None or reference_no_wait is None
        else int(reference_no_wait.steps - reference_plan.steps)
    )
    return {
        "best_plan": best, "witness": witness, "observation": observation,
        "reference_gate_pair": pair, "alternative_gate_pair": alternative_pair,
        "reference_gate_cost": _plan_cost(reference_plan),
        "reference_gate_no_wait_steps": (
            None if reference_no_wait is None else int(reference_no_wait.steps)
        ),
        "reference_gate_no_wait_reachable": reference_no_wait is not None,
        "best_alternative_gate_cost": _plan_cost(alternative),
        "no_wait_steps": None if no_wait is None else no_wait.steps,
        "wait_advantage_steps": reference_wait_advantage,
        "global_no_wait_penalty_steps": global_wait_advantage,
        "reference_path_collision": reference_collision,
        "first_primary_conflict_cell": None if conflict_cell is None else list(conflict_cell),
        "spatial_bypass_precheck": spatial_check,
    }


def _record(
    problem,
    scenario,
    masks,
    classification,
    behavior,
    reference_index,
    records_by_id,
    demo_count,
    observation_radius,
):
    reference_bit = 1 << reference_index
    full_mask = 0
    for mask in masks:
        full_mask |= int(mask)
    best, witness = classification["best_plan"], classification["witness"]
    obstacle_rows = []
    for index, (spec, mask) in enumerate(zip(scenario.obstacles, masks)):
        route = records_by_id[spec.label]
        obstacle_rows.append({
            "route_id": spec.label, "start_index": spec.start_index,
            "direction": spec.direction, "move_every": spec.move_every,
            "functional_role": route["functional_role"],
            "critical_zone": route["critical_zone"],
            "scene_role": "primary" if behavior != "normal" and index == 0 else "context",
            "individual_demo_collision_count": int(mask).bit_count(),
            "collides_designated_reference": bool(int(mask) & reference_bit),
        })
    distance = classification["observation"]["nearest_dynamic_distance_at_decision"]
    primary_demo = 0 if behavior == "normal" else int(masks[0]).bit_count()
    score = (
        int(full_mask).bit_count()
        + max(0, best.steps - (len(problem.nominal_path) - 1))
        + (0 if distance is None else max(0, observation_radius - int(distance)))
    )
    return {
        "scenario_id": None, "required_behavior": behavior,
        "interaction_definition": "verified opportunity/witness; not exclusive policy intent",
        "difficulty_stratum": "control" if behavior == "normal" else None,
        "difficulty_score": float(score), "reference_path_index": reference_index,
        "reference_gate_pair": list(classification["reference_gate_pair"]),
        "reference_path": [list(cell) for cell in problem.nominal_path],
        "reference_demo_collision_count": int(full_mask).bit_count(),
        "reference_demo_collision_rate": int(full_mask).bit_count() / float(demo_count),
        "minimum_safe_path_steps": best.steps,
        "safe_detour_steps": best.steps - (len(problem.nominal_path) - 1),
        "oracle_wait_count": witness.wait_count,
        "no_wait_safe_path_steps": classification["no_wait_steps"],
        "wait_advantage_steps": classification["wait_advantage_steps"],
        "reference_gate_no_wait_steps": classification["reference_gate_no_wait_steps"],
        "reference_gate_no_wait_reachable": classification["reference_gate_no_wait_reachable"],
        "global_no_wait_penalty_steps": classification["global_no_wait_penalty_steps"],
        "nominal_gate_cost": classification["reference_gate_cost"],
        "best_alternative_gate_cost": classification["best_alternative_gate_cost"],
        "decision_step": classification["observation"]["decision_step"],
        "decision_position": classification["observation"]["decision_position"],
        "nearest_dynamic_distance_at_decision": distance,
        "primary_obstacle_index": 0 if behavior != "normal" else None,
        "primary_route_id": scenario.obstacles[0].label if behavior != "normal" else None,
        "primary_critical_zone": records_by_id[scenario.obstacles[0].label]["critical_zone"] if behavior != "normal" else None,
        "primary_functional_role": records_by_id[scenario.obstacles[0].label]["functional_role"] if behavior != "normal" else None,
        "primary_reference_demo_collision_count": primary_demo,
        "first_primary_conflict_cell": classification["first_primary_conflict_cell"],
        "witness_steps": witness.steps, "witness_extra_steps": witness.steps - best.steps,
        "oracle_path": [list(cell) for cell in witness.positions],
        "pair_id": None, "pair_role": "unpaired", "paired_scenario_id": None,
        "matched_behavior": behavior,
        "obstacles": obstacle_rows,
    }


class ScenarioBuilder:
    def __init__(self, problem, config, split, records, demo_paths, signatures):
        self.problem, self.config, self.split = problem, config, split
        self.records, self.demo_paths, self.signatures = records, demo_paths, signatures
        self.lookup = {row["route_id"]: row for row in records}
        design = config["spatial_generalization"]["scenario_design"]
        self.design = design
        self.radius = int(config["environment"]["window_size"]) // 2
        self.max_attempts = int(design["max_attempts_per_requested_scene"])
        self.max_stalled = int(design["max_attempts_without_acceptance"])
        self.speeds = tuple(int(value) for value in design["move_every_choices"])
        self.options = _route_options(records, self.speeds, demo_paths)
        seed = int(config["spatial_generalization"]["behavior_search_seed"]) + 10000 * (SPLITS.index(split) + 1)
        self.rng = random.Random(seed)
        self.primary_use, self.primary_trials, self.route_use = Counter(), Counter(), Counter()
        self.primary_witness_cache = {}
        self.audit = defaultdict(Counter)
        self.progress_root = Path(
            design.get("_audit_root", ROOT / "outputs/office_practical_v11_design")
        ) / "search"

    def _save_progress(self, behavior, accepted, requested, attempts, started, status, *, paired=False):
        write_json({
            "status": status,
            "split": self.split,
            "behavior": behavior,
            "paired": paired,
            "accepted": accepted,
            "requested": requested,
            "attempts": attempts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "rejections": dict(self.audit[behavior]),
            "primary_route_acceptance": dict(self.primary_use),
            "primary_route_trials": dict(self.primary_trials),
        }, self.progress_root / f"{self.split}_{behavior}{'_paired' if paired else ''}.json")

    def _scene(self, chosen):
        specs = tuple(row[0] for row in chosen)
        masks = tuple(int(row[1]) for row in chosen)
        return DynamicScenario(seed=0, obstacles=specs), masks

    def _target_proposals(self, behavior):
        role = {
            "wait": "wait_gate",
            "avoidance": "open_approach",
            "reroute": "bottleneck",
        }[behavior]
        proposals = []
        for route in self.records:
            if route["functional_role"] != role:
                continue
            eligible = set(int(value) for value in route["eligible_reference_indices"])
            for spec, mask in self.options[route["route_id"]]:
                for reference_index in eligible:
                    if int(mask) & (1 << reference_index):
                        proposals.append((route, spec, int(mask), reference_index))
        self.rng.shuffle(proposals)
        return proposals

    def _primary_behavior_witness(self, behavior, route, spec, reference_index):
        key = (
            behavior, spec.label, int(spec.start_index), int(spec.direction),
            int(spec.move_every), int(reference_index),
        )
        if key in self.primary_witness_cache:
            return self.primary_witness_cache[key]
        diagnostics = Counter()
        primary_only = DynamicScenario(seed=0, obstacles=(spec,))
        classification = classify_practical(
            self.problem, primary_only, behavior, self.demo_paths[reference_index],
            self.radius, self.design, diagnostics)
        if classification is None:
            self.audit[behavior].update({
                f"primary_only_{name}": value for name, value in diagnostics.items()
            })
            self.primary_witness_cache[key] = None
            return None
        witness = tuple(classification["witness"].positions)
        reference_bit = 1 << reference_index
        for zone in ZONE_ORDER:
            if zone == route["critical_zone"]:
                continue
            has_safe_context = any(
                not (int(mask) & reference_bit) and not _spec_collides_path(option, witness)
                for candidate in self.records
                if candidate["critical_zone"] == zone
                for option, mask in self.options[candidate["route_id"]]
            )
            if not has_safe_context:
                self.audit[behavior][f"primary_only_no_witness_safe_context:{zone}"] += 1
                self.primary_witness_cache[key] = None
                return None
        self.primary_witness_cache[key] = witness
        return witness

    def _context(self, primary_route, reference_index, primary, required_safe_path=None):
        chosen = [primary]
        for zone in ZONE_ORDER:
            if zone == primary_route["critical_zone"]:
                continue
            routes = [row for row in self.records if row["critical_zone"] == zone]
            option = _balanced_reference_safe_option(
                routes, self.options, reference_index, self.route_use, self.rng,
                required_safe_path=required_safe_path)
            if option is None:
                return None
            chosen.append(option)
        return chosen

    def _matched_control(self, conflict, masks, reference_index, behavior):
        primary = conflict.obstacles[0]
        strict = bool(self.config["spatial_generalization"]["scenario_design"][
            "require_paired_primary_demo_conflict_reduction"])
        replacements = []
        for spec, mask in self.options[primary.label]:
            if (spec.direction != primary.direction or spec.move_every != primary.move_every
                    or spec.start_index == primary.start_index or int(mask) & (1 << reference_index)):
                continue
            if strict and int(mask).bit_count() >= int(masks[0]).bit_count():
                continue
            replacements.append((spec, int(mask)))
        replacements.sort(key=lambda item: (item[1].bit_count(), item[0].start_index))
        diagnostics = Counter()
        for replacement, mask in replacements:
            specs = (replacement,) + conflict.obstacles[1:]
            control = DynamicScenario(seed=0, obstacles=specs)
            if _signature(control.obstacles) in self.signatures:
                self.audit[behavior]["pair_control_duplicate"] += 1
                continue
            classification = classify_practical(
                self.problem, control, "normal", self.demo_paths[reference_index],
                self.radius, self.design, diagnostics)
            if classification is not None:
                return control, (mask,) + masks[1:], classification
        self.audit[behavior].update({f"pair_control_{key}": value for key, value in diagnostics.items()})
        return None

    def build_targets(self, behavior, count, *, paired=False):
        proposals = self._target_proposals(behavior)
        if not proposals:
            raise RuntimeError(f"No {self.split} {behavior} primary proposals.")
        print(
            f"[{self.split} {behavior}{' paired' if paired else ''}] "
            f"primary_proposals={len(proposals)} causal_precheck=enabled",
            flush=True,
        )
        accepted, attempts = [], 0
        attempts_without_acceptance = 0
        started = time.monotonic()
        while len(accepted) < count:
            attempts += 1
            attempts_without_acceptance += 1
            if attempts > count * self.max_attempts:
                self._save_progress(
                    behavior, len(accepted), count, attempts, started, "failed_total_budget", paired=paired)
                raise RuntimeError(
                    f"Could not build {count} {self.split} {behavior} scenes; "
                    f"accepted={len(accepted)}, rejected={dict(self.audit[behavior])}.")
            if attempts_without_acceptance > self.max_stalled:
                self._save_progress(
                    behavior, len(accepted), count, attempts, started, "failed_stalled", paired=paired)
                raise RuntimeError(
                    f"No new {self.split} {behavior} scene in {self.max_stalled} attempts; "
                    f"accepted={len(accepted)}, rejected={dict(self.audit[behavior])}. "
                    f"See {self.progress_root} for the audit.")
            if attempts % 500 == 0:
                print(f"[{self.split} {behavior}{' paired' if paired else ''}] "
                      f"{len(accepted)}/{count} attempts={attempts} elapsed={time.monotonic() - started:.1f}s", flush=True)
                self._save_progress(
                    behavior, len(accepted), count, attempts, started, "running", paired=paired)
            # Failed route variants are gradually deprioritized, so one impossible
            # anchor cannot monopolize the search merely because its accepted-use
            # count remains zero.
            score = lambda row: (
                self.primary_use[row[0]["route_id"]]
                + self.primary_trials[row[0]["route_id"]] // 25
            )
            minimum = min(score(row) for row in proposals)
            eligible_indices = [
                index for index, row in enumerate(proposals)
                if score(row) == minimum
            ]
            proposal_index = self.rng.choice(eligible_indices)
            route, spec, mask, reference_index = proposals[proposal_index]
            self.primary_trials[route["route_id"]] += 1
            witness = self._primary_behavior_witness(
                behavior, route, spec, reference_index)
            if witness is None:
                # This exact primary phase/reference pair cannot independently
                # cause the requested behavior.  Remove it permanently instead
                # of retrying it with thousands of random context combinations.
                proposals.pop(proposal_index)
                self.audit[behavior]["primary_only_proposal_removed"] += 1
                if not proposals:
                    self._save_progress(
                        behavior, len(accepted), count, attempts, started,
                        "failed_no_causal_primary", paired=paired)
                    raise RuntimeError(
                        f"No causally valid {self.split} {behavior} primary proposal remains; "
                        f"accepted={len(accepted)}, rejected={dict(self.audit[behavior])}. "
                        f"See {self.progress_root} for the audit.")
                continue
            chosen = self._context(
                route, reference_index, (spec, mask),
                required_safe_path=witness)
            if chosen is None:
                self.audit[behavior]["no_reference_safe_context"] += 1
                continue
            scene, masks = self._scene(chosen)
            signature = _signature(scene.obstacles)
            if signature in self.signatures:
                self.audit[behavior]["duplicate"] += 1
                continue
            classification = classify_practical(
                self.problem, scene, behavior, self.demo_paths[reference_index],
                self.radius, self.design, self.audit[behavior])
            if classification is None:
                continue
            context = replace(self.problem, nominal_path=tuple(self.demo_paths[reference_index]))
            reduced = DynamicScenario(seed=0, obstacles=scene.obstacles[1:])
            if v9._nominal_path_collides(context, reduced):
                self.audit[behavior]["context_not_reference_safe"] += 1
                continue
            record = _record(
                context, scene, masks, classification, behavior, reference_index,
                self.lookup, len(self.demo_paths), self.radius)
            if paired:
                control = self._matched_control(scene, masks, reference_index, behavior)
                if control is None:
                    self.audit[behavior]["no_phase_control"] += 1
                    continue
                control_scene, control_masks, control_classification = control
                control_record = _record(
                    context, control_scene, control_masks, control_classification,
                    "normal", reference_index, self.lookup, len(self.demo_paths), self.radius)
                accepted.append((record, control_record))
                self.signatures.add(_signature(control_scene.obstacles))
            else:
                accepted.append(record)
            self.signatures.add(signature)
            self.primary_use[route["route_id"]] += 1
            self.route_use.update(spec.label for spec in scene.obstacles)
            attempts_without_acceptance = 0
            if len(accepted) == 1 or len(accepted) % 5 == 0 or len(accepted) == count:
                print(f"[{self.split} {behavior}{' paired' if paired else ''}] "
                      f"{len(accepted)}/{count} attempts={attempts} rejected={dict(self.audit[behavior])}", flush=True)
                self._save_progress(
                    behavior, len(accepted), count, attempts, started,
                    "complete" if len(accepted) == count else "running", paired=paired)
        return accepted

    def build_controls(self, count):
        accepted, attempts = [], 0
        attempts_without_acceptance = 0
        started = time.monotonic()
        references = []
        for reference_index, path in enumerate(self.demo_paths):
            if _gate_pair(path) is None:
                continue
            reference_bit = 1 << reference_index
            if all(any(
                not (int(mask) & reference_bit)
                for route in self.records
                if route["critical_zone"] == zone
                for _, mask in self.options[route["route_id"]]
            ) for zone in ZONE_ORDER):
                references.append(reference_index)
        if not references:
            raise RuntimeError(
                f"No {self.split} A* reference has a safe phase option in every functional zone."
            )
        print(
            f"[{self.split} normal] eligible_references={len(references)}/{len(self.demo_paths)}",
            flush=True,
        )
        while len(accepted) < count:
            attempts += 1
            attempts_without_acceptance += 1
            if attempts > count * self.max_attempts:
                self._save_progress("normal", len(accepted), count, attempts, started, "failed_total_budget")
                raise RuntimeError(
                    f"Could not build {count} {self.split} normal controls; "
                    f"accepted={len(accepted)}, rejected={dict(self.audit['normal'])}.")
            if attempts_without_acceptance > self.max_stalled:
                self._save_progress("normal", len(accepted), count, attempts, started, "failed_stalled")
                raise RuntimeError(
                    f"No new {self.split} normal control in {self.max_stalled} attempts; "
                    f"accepted={len(accepted)}, rejected={dict(self.audit['normal'])}.")
            if attempts % 500 == 0:
                print(f"[{self.split} normal] {len(accepted)}/{count} attempts={attempts} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
                self._save_progress("normal", len(accepted), count, attempts, started, "running")
            reference_index = self.rng.choice(references)
            chosen = []
            for zone in ZONE_ORDER:
                routes = [row for row in self.records if row["critical_zone"] == zone]
                option = _balanced_reference_safe_option(
                    routes, self.options, reference_index, self.route_use, self.rng)
                if option is None:
                    chosen = []
                    self.audit["normal"][f"no_reference_safe_zone:{zone}"] += 1
                    break
                chosen.append(option)
            if not chosen:
                self.audit["normal"]["no_reference_safe_combination"] += 1
                continue
            scene, masks = self._scene(chosen)
            signature = _signature(scene.obstacles)
            if signature in self.signatures:
                self.audit["normal"]["duplicate"] += 1
                continue
            classification = classify_practical(
                self.problem, scene, "normal", self.demo_paths[reference_index],
                self.radius, self.design, self.audit["normal"])
            if classification is None:
                continue
            context = replace(self.problem, nominal_path=tuple(self.demo_paths[reference_index]))
            accepted.append(_record(
                context, scene, masks, classification, "normal", reference_index,
                self.lookup, len(self.demo_paths), self.radius))
            self.signatures.add(signature)
            self.route_use.update(spec.label for spec in scene.obstacles)
            attempts_without_acceptance = 0
            if len(accepted) == 1 or len(accepted) % 5 == 0 or len(accepted) == count:
                print(f"[{self.split} normal] {len(accepted)}/{count} attempts={attempts} "
                      f"rejected={dict(self.audit['normal'])}", flush=True)
                self._save_progress(
                    "normal", len(accepted), count, attempts, started,
                    "complete" if len(accepted) == count else "running")
        return accepted


def _fraction_targets(total, fractions):
    raw = {name: total * float(value) for name, value in fractions.items()}
    result = {name: int(math.floor(value)) for name, value in raw.items()}
    for name in sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))[:total - sum(result.values())]:
        result[name] += 1
    return result


def _assign_difficulty(records, fractions):
    for behavior in ("wait", "avoidance", "reroute"):
        group = sorted((row for row in records if row["required_behavior"] == behavior),
                       key=lambda row: (row["difficulty_score"], row["primary_route_id"]))
        targets = _fraction_targets(len(group), fractions)
        cursor = 0
        for label in ("easy", "medium", "hard"):
            for row in group[cursor:cursor + targets[label]]:
                row["difficulty_stratum"] = label
            cursor += targets[label]


def _assign_ids(split, general, pairs):
    offset = {"train": 0, "validation": 10_000, "test": 20_000}[split]
    result = list(general)
    for pair_index, (conflict, control) in enumerate(pairs):
        pair_id = f"v11_test_pair_{pair_index:02d}"
        conflict.update(pair_id=pair_id, pair_role="conflict")
        control.update(pair_id=pair_id, pair_role="matched_control",
                       matched_behavior=conflict["required_behavior"],
                       matched_conflict_difficulty=conflict["difficulty_stratum"])
        result.extend((conflict, control))
    for index, row in enumerate(result):
        row["scenario_id"] = offset + index
    for conflict, control in pairs:
        conflict["paired_scenario_id"] = control["scenario_id"]
        control["paired_scenario_id"] = conflict["scenario_id"]
    return result


def build_scenarios(problem, config, route_pools, demo_paths):
    design = config["spatial_generalization"]["scenario_design"]
    fractions = design["difficulty_fractions"]
    signatures = set()
    scenarios, audits = {}, {}
    for split in ("train", "validation"):
        builder = ScenarioBuilder(problem, config, split, route_pools[split]["corridor"], demo_paths, signatures)
        counts = design["practical_counts"][split]
        records = builder.build_controls(int(counts["normal"]))
        for behavior in ("wait", "avoidance", "reroute"):
            records.extend(builder.build_targets(behavior, int(counts[behavior])))
        _assign_difficulty(records, fractions)
        scenarios[split] = _assign_ids(split, records, [])
        audits[split] = {key: dict(value) for key, value in builder.audit.items()}
    split = "test"
    builder = ScenarioBuilder(problem, config, split, route_pools[split]["corridor"], demo_paths, signatures)
    counts = design["practical_counts"][split]
    general = builder.build_controls(int(counts["normal"]))
    for behavior in ("wait", "avoidance", "reroute"):
        general.extend(builder.build_targets(behavior, int(counts[behavior])))
    pair_groups = []
    for behavior, count in design["diagnostic_pair_counts"].items():
        pair_groups.extend(builder.build_targets(str(behavior), int(count), paired=True))
    conflicts = [pair[0] for pair in pair_groups]
    _assign_difficulty(general + conflicts, fractions)
    scenarios[split] = _assign_ids(split, general, pair_groups)
    audits[split] = {key: dict(value) for key, value in builder.audit.items()}
    return scenarios, audits


def _scenario_from_record(source, lookup):
    return DynamicScenario(seed=int(source["scenario_id"]), obstacles=tuple(
        DynamicObstacleSpec(route=tuple(map(tuple, lookup[row["route_id"]]["route"])),
                            start_index=int(row["start_index"]), direction=int(row["direction"]),
                            move_every=int(row["move_every"]), label=row["route_id"],
                            reference_path_source=lookup[row["route_id"]]["critical_zone"])
        for row in source["obstacles"]))


def _replay_path(problem, scenario, path):
    path = tuple(map(tuple, path))
    trajectories = [obstacle_positions(spec, len(path) - 1) for spec in scenario.obstacles]
    if not path or path[0] != problem.start or path[-1] != problem.goal:
        raise ValueError("Witness path has incorrect endpoints.")
    for step, cell in enumerate(path):
        if cell in problem.obstacles or not all(0 <= value < problem.size for value in cell):
            raise ValueError("Witness path intersects static geometry.")
        if step:
            if sum(abs(a - b) for a, b in zip(cell, path[step - 1])) > 1:
                raise ValueError("Witness path has an invalid action.")
            if any(cell in (positions[step - 1], positions[step]) for positions in trajectories):
                raise ValueError("Witness path collides with a dynamic obstacle.")


def validate_v11(problem, manifest, config, demo_paths):
    validate_spatial_scenario_manifest(problem, manifest)
    expected = config["spatial_generalization"]["scenario_counts"]
    design = config["spatial_generalization"]["scenario_design"]
    signatures, route_keys = set(), set()
    for split in SPLITS:
        rows = manifest["scenarios"][split]
        if len(rows) != int(expected[split]):
            raise ValueError(f"{split} scenario count mismatch.")
        expected_behaviors = Counter({
            name: int(count)
            for name, count in design["practical_counts"][split].items()
        })
        if split == "test":
            for name, count in design["diagnostic_pair_counts"].items():
                expected_behaviors[str(name)] += int(count)
                expected_behaviors["normal"] += int(count)
        actual_behaviors = Counter(str(row["required_behavior"]) for row in rows)
        if actual_behaviors != expected_behaviors:
            raise ValueError(
                f"{split} behavior counts {dict(actual_behaviors)} disagree with "
                f"the registered design {dict(expected_behaviors)}."
            )
        lookup = {row["route_id"]: row for row in manifest["route_pools"][split]["corridor"]}
        for route in lookup.values():
            key = route_key(route["route"])
            if key in route_keys:
                raise ValueError("Whole route geometry is repeated across splits.")
            route_keys.add(key)
        used = Counter()
        for source in rows:
            scene = _scenario_from_record(source, lookup)
            signature = _signature(scene.obstacles)
            if signature in signatures:
                raise ValueError("Duplicate full dynamic scene across splits.")
            signatures.add(signature)
            used.update(row["route_id"] for row in source["obstacles"])
            reference_index = int(source["reference_path_index"])
            if not 0 <= reference_index < len(demo_paths):
                raise ValueError("Scenario reference_path_index is out of range.")
            if source["reference_path"] != [list(cell) for cell in demo_paths[reference_index]]:
                raise ValueError("Stored reference_path does not match reference_path_index.")
            behavior = source["required_behavior"]
            verified = classify_practical(
                problem, scene, behavior, demo_paths[reference_index],
                int(config["environment"]["window_size"]) // 2,
                config["spatial_generalization"]["scenario_design"])
            if verified is None:
                raise ValueError(f"Scenario {source['scenario_id']} no longer has its witness.")
            if source["oracle_path"] != [list(cell) for cell in verified["witness"].positions]:
                raise ValueError(f"Scenario {source['scenario_id']} stores a stale witness path.")
            if int(source["minimum_safe_path_steps"]) != int(verified["best_plan"].steps):
                raise ValueError(f"Scenario {source['scenario_id']} stores a stale safe-path cost.")
            _replay_path(problem, scene, source["oracle_path"])
            designated = [row["collides_designated_reference"] for row in source["obstacles"]]
            if behavior == "normal":
                if any(designated) or source["primary_obstacle_index"] is not None:
                    raise ValueError("A normal control collides with its designated reference.")
            elif designated != [True, False, False, False, False]:
                raise ValueError("A target scene must isolate one designated-reference primary at index 0.")
        if set(used) != set(lookup):
            raise ValueError(f"{split} contains unused route variants: {sorted(set(lookup) - set(used))}")
    pairs = defaultdict(list)
    for source in manifest["scenarios"]["test"]:
        if source["pair_id"]:
            pairs[source["pair_id"]].append(source)
    if len(pairs) != sum(int(v) for v in config["spatial_generalization"]["scenario_design"]["diagnostic_pair_counts"].values()):
        raise ValueError("Test diagnostic pair count mismatch.")
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"Pair {pair_id} does not contain two scenes.")
        conflict = next(row for row in pair if row["pair_role"] == "conflict")
        control = next(row for row in pair if row["pair_role"] == "matched_control")
        if conflict["paired_scenario_id"] != control["scenario_id"] or control["paired_scenario_id"] != conflict["scenario_id"]:
            raise ValueError(f"Pair {pair_id} has inconsistent IDs.")
        changed = [index for index, (left, right) in enumerate(zip(conflict["obstacles"], control["obstacles"]))
                   if left["start_index"] != right["start_index"]]
        if changed != [0] or any(left[key] != right[key]
                                 for left, right in zip(conflict["obstacles"], control["obstacles"])
                                 for key in ("route_id", "direction", "move_every")):
            raise ValueError(f"Pair {pair_id} changes more than primary phase.")
        if conflict["matched_behavior"] != control["matched_behavior"]:
            raise ValueError(f"Pair {pair_id} does not preserve its matched behavior label.")
        if conflict["primary_reference_demo_collision_count"] <= control["primary_reference_demo_collision_count"]:
            raise ValueError(f"Pair {pair_id} does not reduce primary A* conflict exposure.")
        if not conflict["obstacles"][0]["collides_designated_reference"]:
            raise ValueError(f"Pair {pair_id} conflict does not hit its designated reference.")
        if control["obstacles"][0]["collides_designated_reference"]:
            raise ValueError(f"Pair {pair_id} control still hits its designated reference.")


def build_manifest(problem, config, route_pools, demo_paths):
    scenarios, search_audit = build_scenarios(problem, config, route_pools, demo_paths)
    spatial = config["spatial_generalization"]
    design = spatial["scenario_design"]
    manifest = {
        "format_version": 1, "map_id": problem.map_id, "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "office_practical_behavior_v11", "seed": int(spatial["generation_seed"]),
            "behavior_search_seed": int(spatial["behavior_search_seed"]),
            "candidate_route_lengths": [int(value) for value in spatial["candidate_route_lengths"]],
            "reference_path_count": int(spatial["reference_path_count"]),
            "reference_demo_episodes": len(demo_paths),
            "reference_demo_seed": int(config["demonstrations"]["seed"]),
            "separation_radius": int(spatial["separation_radius"]),
            "move_every": int(spatial["move_every"]),
            "move_every_choices": list(design["move_every_choices"]),
            "corridor_per_scenario": 5, "background_per_scenario": 0,
            "cross_split_spatial_disjoint": False,
            "split_scope": "held-out whole route geometry and phase configurations on one Office topology",
            "placement_rule": "one route in each functional zone; one designated-reference primary plus four reference-safe context obstacles",
            "reference_rule": "scenario reference selected from the same 20 randomized static A* demonstrations used to define replay exposure; not fixed to the left gates",
            "practical_counts": design["practical_counts"],
            "diagnostic_pair_counts": design["diagnostic_pair_counts"],
            "search_audit": search_audit,
            "acceptance": {
                "all": "legal five-obstacle motion and replay-verified collision-free witness",
                "normal": "designated static A* reference remains safe",
                "wait": "visible primary and an optimal reference-gate path with strictly useful STAY",
                "avoidance": "visible no-STAY simple path in the same gate pair with a statically bypassable conflict cell",
                "reroute": "visible primary before a no-STAY alternative gate pair improves reference gates",
                "paired_test": "only primary initial phase changes; designated reference conflict disappears and primary demo-conflict count decreases",
            },
            "caveat": "behavior labels certify available witnesses/opportunities, not a unique required policy action",
        },
        "route_pools": route_pools, "scenarios": scenarios,
    }
    validate_v11(problem, manifest, config, demo_paths)
    return manifest


def render_route_pools(problem, route_pools, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=180)
    for axis, split in zip(axes, SPLITS):
        v9._draw_base(axis, problem, f"{split.title()}: functional routes", ticks=True)
        for row in route_pools[split]["corridor"]:
            route = row["route"]
            style = {
                "wait_gate": "-.",
                "bottleneck": "-",
                "open_approach": "--",
            }[row["functional_role"]]
            axis.plot([cell[1] for cell in route], [cell[0] for cell in route], style,
                      color=v10.ZONE_COLORS[row["critical_zone"]],
                      linewidth=2.6 if style == "-" else 1.7)
            axis.scatter(row["center"][1], row["center"][0], s=16,
                         color=v10.ZONE_COLORS[row["critical_zone"]])
    fig.suptitle(
        "Office v11 placement blueprint | dash-dot=wait clearance, "
        "solid=reroute blocker, dashed=open bypass",
        fontweight="bold",
    )
    handles = [
        Line2D([0], [0], color=v10.ZONE_COLORS[zone], linewidth=2.5,
               label=zone.replace("_gate", "").replace("_approach", ""))
        for zone in ZONE_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.text(.5, .045, "Every route intersects at least one of 20 static A* demonstrations; panels hold out whole route geometries", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_gallery(problem, manifest, split, output):
    sources = manifest["scenarios"][split]
    lookup = {row["route_id"]: row for row in manifest["route_pools"][split]["corridor"]}
    columns = 10 if len(sources) >= 50 else 5
    rows = math.ceil(len(sources) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.55 * columns, 2.65 * rows), dpi=135)
    axes = np.asarray(axes).reshape(-1)
    for axis, source in zip(axes, sources):
        behavior = source["required_behavior"]
        pair_control = source["pair_role"] == "matched_control"
        witness_behavior = source["matched_behavior"] if pair_control else behavior
        v9._draw_base(axis, problem, f"{source['scenario_id']} {behavior} | {source['pair_role']}")
        path = source["oracle_path"]
        axis.plot([cell[1] for cell in path], [cell[0] for cell in path], "--",
                  color=COLORS[witness_behavior], linewidth=1.2)
        for index, obstacle in enumerate(source["obstacles"]):
            route = lookup[obstacle["route_id"]]["route"]
            highlight = index == 0 and (behavior != "normal" or pair_control)
            route_color = (
                "#1565c0" if highlight and pair_control
                else "#c62828" if highlight
                else "#888888"
            )
            axis.plot([cell[1] for cell in route], [cell[0] for cell in route],
                      color=route_color, linewidth=2.2 if highlight else 1)
            start_index = int(obstacle["start_index"])
            initial = route[start_index]
            axis.scatter(initial[1], initial[0], s=13, color=route_color)
            next_index = start_index + int(obstacle["direction"])
            if not 0 <= next_index < len(route):
                next_index = start_index - int(obstacle["direction"])
            following = route[next_index]
            axis.annotate(
                "", xy=(following[1], following[0]), xytext=(initial[1], initial[0]),
                arrowprops={"arrowstyle": "->", "color": route_color, "linewidth": .75},
            )
    for axis in axes[len(sources):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Office v11 {split}: dashed=safe witness | red=conflict primary, blue=paired control primary",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_audit(manifest, output):
    rows = []
    summaries = []
    for split in SPLITS:
        group = manifest["scenarios"][split]
        for source in group:
            rows.append({key: source.get(key) for key in (
                "scenario_id", "required_behavior", "difficulty_stratum", "pair_id", "pair_role",
                "reference_path_index", "primary_critical_zone", "primary_functional_role",
                "reference_demo_collision_count", "minimum_safe_path_steps", "safe_detour_steps",
                "oracle_wait_count", "reference_gate_no_wait_reachable",
                "reference_gate_no_wait_steps", "wait_advantage_steps",
                "witness_extra_steps", "decision_step",
                "nearest_dynamic_distance_at_decision") } | {"split": split})
        summaries.append({
            "split": split, "scenario_count": len(group),
            "behavior_counts": dict(Counter(row["required_behavior"] for row in group)),
            "primary_zone_counts": dict(Counter(row["primary_critical_zone"] for row in group if row["primary_critical_zone"])),
            "pair_scene_count": sum(bool(row["pair_id"]) for row in group),
            "safe_reachable_rate": 1.0,
            "designated_reference_conflict_rate": sum(row["required_behavior"] != "normal" for row in group) / len(group),
        })
    output.mkdir(parents=True, exist_ok=True)
    fields = ["split"] + [key for key in rows[0] if key != "split"]
    with (output / "scenario_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(summaries, output / "split_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    x = np.arange(len(SPLITS))
    bottom = np.zeros(len(SPLITS))
    for behavior in BEHAVIORS:
        values = [sum(row["required_behavior"] == behavior for row in manifest["scenarios"][split]) for split in SPLITS]
        axes[0].bar(x, values, bottom=bottom, label=behavior, color=COLORS[behavior])
        bottom += values
    axes[0].set_xticks(x, SPLITS); axes[0].set_title("Scene composition"); axes[0].legend(frameon=False)
    zones = list(ZONE_ORDER)
    width = .25
    for index, split in enumerate(SPLITS):
        values = Counter(row["primary_critical_zone"] for row in manifest["scenarios"][split] if row["primary_critical_zone"])
        axes[1].bar(np.arange(len(zones)) + (index - 1) * width, [values[z] for z in zones], width, label=split)
    axes[1].set_xticks(np.arange(len(zones)), [z.replace("_gate", "").replace("_approach", "") for z in zones], rotation=25)
    axes[1].set_title("Primary interaction locations"); axes[1].legend(frameon=False)
    fig.suptitle("Office v11 dataset audit | opportunities, not exclusive action labels", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "dataset_summary.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = v9._resolve(args.config)
    config = load_config(config_path)
    validate_config_contract(config)
    problem = v9._load_problem(config)
    demo_paths = v9._reference_demo_paths(
        problem, int(config["demonstrations"]["episodes"]), int(config["demonstrations"]["seed"]))
    if len(demo_paths) != 20:
        raise ValueError("Office v11 is registered against exactly 20 static A* demonstrations.")
    design_root = ROOT / "outputs/office_practical_v11_design"
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    output = design_root / "runs" / run_id
    config["spatial_generalization"]["scenario_design"]["_audit_root"] = str(output)
    manifest_path = v9._resolve(config["spatial_generalization"]["manifest"])
    if manifest_path.exists() and not args.overwrite and not args.preview_only:
        raise FileExistsError(f"Frozen manifest already exists: {manifest_path}. Use --overwrite only intentionally.")
    status = {"status": "running", "config": str(config_path), "manifest": str(manifest_path),
              "audit_directory": str(output),
              "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(status, output / "generation_status.json")
    write_json(status, design_root / "generation_status.json")
    try:
        route_pools, candidates, route_audit = build_route_pools(problem, config, demo_paths)
        write_json({"anchors": ANCHORS, "route_audit": route_audit,
                    "candidate_routes": candidates, "selected_route_pools": route_pools,
                    "note": "Route feasibility only; no scenario behavior is claimed here."},
                   output / "placement_audit.json")
        render_route_pools(problem, route_pools, output / "placement_blueprint.png")
        render_route_pools(problem, route_pools, v9._resolve(config["spatial_generalization"]["route_pool_preview"]))
        if args.preview_only:
            status.update(status="preview_complete", training_ready=False,
                          note="No manifest or scenarios generated in preview-only mode.")
            return
        manifest = build_manifest(problem, config, route_pools, demo_paths)
        for split, key in (("train", "training_preview"), ("validation", "validation_preview"), ("test", "test_preview")):
            render_gallery(problem, manifest, split, v9._resolve(config["spatial_generalization"][key]))
        summaries = write_audit(manifest, output)
        dataset_preview = v9._resolve(config["spatial_generalization"]["dataset_summary_preview"])
        dataset_preview.parent.mkdir(parents=True, exist_ok=True)
        # Copying through matplotlib is unnecessary; the audit path remains the
        # canonical figure and this small explicit byte copy preserves identity.
        dataset_preview.write_bytes((output / "dataset_summary.png").read_bytes())
        # Freeze the manifest only after every validation and preview succeeds.
        # A failed run therefore cannot leave a partial dataset that training
        # later mistakes for a completed registration.
        write_json(manifest, manifest_path)
        status.update(status="complete", training_ready=True, split_summary=summaries,
                      note="Ready for a one-seed training pilot; test is development data, not final independent confirmation.")
        print(f"Office v11 complete: {manifest_path}", flush=True)
    except KeyboardInterrupt:
        status.update(status="interrupted", training_ready=False)
        raise SystemExit(130) from None
    except Exception as error:
        status.update(status="error", training_ready=False, error=str(error))
        raise
    finally:
        status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(status, output / "generation_status.json")
        write_json(status, design_root / "generation_status.json")


if __name__ == "__main__":
    main()
