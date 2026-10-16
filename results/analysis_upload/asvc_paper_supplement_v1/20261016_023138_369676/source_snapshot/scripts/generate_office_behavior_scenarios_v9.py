"""Generate balanced Office controls plus wait, avoidance, and reroute scenarios."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

from astar_d3qn.core.grid import chebyshev
from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import SPLITS, validate_spatial_scenario_manifest
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.evaluation.behavior_oracle import SafePlan, shortest_safe_plan
from astar_d3qn.evaluation.conflict import obstacle_positions
from astar_d3qn.training.demo_collector import (
    demonstration_file_sha256,
    demonstration_signature,
    load_demonstrations,
    paths_from_demonstrations,
    validate_demonstration_dataset,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json

from generate_office_behavior_scenarios_v6 import (
    NOMINAL_GATE_PAIR,
    _build_route_pools,
    _candidate_combinations,
    _counterfactual_relevance,
    _draw_base,
    _gate_pair_plans,
    _install_behavior_anchors,
    _load_problem,
    _nominal_path_collides,
    _plan_cost,
    _reference_demo_paths,
    _resolve,
)
from generate_office_critical_scenarios_v5 import ZONE_COLORS, ZONE_LABELS


DEFAULT_CONFIG = (
    "configs/"
    "dynamic_spatial_generalization_office_behavior_v9_balanced_dataset_v1.yaml"
)
BEHAVIORS = ("normal", "wait", "avoidance", "reroute")
DIFFICULTIES = ("easy", "medium", "hard")
BEHAVIOR_COLORS = {
    "normal": "#546e7a",
    "wait": "#7b1fa2",
    "avoidance": "#00897b",
    "reroute": "#ef6c00",
}
BEHAVIOR_LABELS = {
    "normal": "Normal control",
    "wait": "Temporal wait",
    "avoidance": "Local avoidance",
    "reroute": "Global reroute",
}
SCENARIO_OFFSETS = {"train": 0, "validation": 10_000, "test": 20_000}


def _dataset_label(manifest: Mapping[str, Any]) -> str:
    protocol = str(manifest.get("generation", {}).get("protocol", "office"))
    version = protocol.rsplit("_v", 1)[-1] if "_v" in protocol else ""
    return f"Office v{version}" if version else "Office"


def _scenario_design(config: Mapping[str, Any]) -> dict[str, Any]:
    spatial = config["spatial_generalization"]
    design = dict(spatial["scenario_design"])
    counts = {
        split: {
            behavior: int(design["behavior_counts"][split][behavior])
            for behavior in BEHAVIORS
        }
        for split in SPLITS
    }
    for split in SPLITS:
        expected = int(spatial["scenario_counts"][split])
        actual = sum(counts[split].values())
        if actual != expected:
            raise ValueError(
                f"{split} behavior counts total {actual}, expected {expected}."
            )
        if any(value <= 0 for value in counts[split].values()):
            raise ValueError("Every split must contain every configured behavior.")
    fractions = {
        name: float(design["difficulty_fractions"][name])
        for name in DIFFICULTIES
    }
    if any(value <= 0.0 for value in fractions.values()) or not math.isclose(
        sum(fractions.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("Difficulty fractions must be positive and sum to one.")
    design["behavior_counts"] = counts
    design["difficulty_fractions"] = fractions
    design["observation_radius"] = int(config["environment"]["window_size"]) // 2
    return design


def _route_options(record, move_every: int, demo_paths):
    """Return every phase so the control group can select low-conflict cases."""

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
                    path[step] in (positions[step - 1], positions[step])
                    for step in range(1, len(path))
                ):
                    mask |= 1 << path_index
            options.append((spec, mask))
    return options


def _full_mask(masks: Sequence[int]) -> int:
    result = 0
    for mask in masks:
        result |= int(mask)
    return result


def _classify_candidate(
    problem,
    scenario: DynamicScenario,
    desired_behavior: str,
    design: Mapping[str, Any],
) -> dict[str, Any] | None:
    nominal_path_collision = _nominal_path_collides(problem, scenario)
    # Most normal-control candidates can be rejected by this inexpensive,
    # deterministic check before invoking the time-expanded safe planner.
    if desired_behavior == "normal" and nominal_path_collision:
        return None
    best = shortest_safe_plan(problem, scenario)
    if best is None:
        return None
    without_wait = shortest_safe_plan(problem, scenario, allow_wait=False)
    no_wait_steps = None if without_wait is None else int(without_wait.steps)
    wait_advantage = (
        None if without_wait is None else int(without_wait.steps - best.steps)
    )
    nominal_steps = len(problem.nominal_path) - 1
    safe_detour_steps = int(best.steps - nominal_steps)

    if desired_behavior == "normal":
        if best.wait_count != 0 or safe_detour_steps > int(
            design["normal_max_safe_detour_steps"]
        ):
            return None

    pair_plans = _gate_pair_plans(problem, scenario)
    pair_costs = {pair: _plan_cost(plan) for pair, plan in pair_plans.items()}
    nominal_cost = pair_costs[NOMINAL_GATE_PAIR]
    alternative_cost = min(
        cost for pair, cost in pair_costs.items() if pair != NOMINAL_GATE_PAIR
    )

    if desired_behavior == "wait":
        minimum_advantage = int(design["minimum_wait_advantage_steps"])
        minimum_gate_margin = int(
            design.get("minimum_wait_gate_margin_steps", 0)
        )
        wait_is_strictly_useful = without_wait is None or (
            wait_advantage is not None and wait_advantage >= minimum_advantage
        )
        if not (
            wait_is_strictly_useful
            and best.wait_count >= int(design["minimum_oracle_wait_steps"])
            and nominal_cost == best.steps
            and alternative_cost - nominal_cost >= minimum_gate_margin
        ):
            return None
    elif desired_behavior == "avoidance":
        minimum_gate_margin = int(
            design.get("minimum_avoidance_gate_margin_steps", 0)
        )
        if not (
            wait_advantage == 0
            and best.wait_count == 0
            and nominal_cost == best.steps
            and alternative_cost - nominal_cost >= minimum_gate_margin
            and nominal_path_collision
            and tuple(best.positions) != tuple(problem.nominal_path)
        ):
            return None
    elif desired_behavior == "reroute":
        reroute_margin = int(design["minimum_reroute_margin_steps"])
        if not (
            wait_advantage == 0
            and best.wait_count == 0
            and best.steps == alternative_cost
            and nominal_cost - alternative_cost >= reroute_margin
        ):
            return None
    elif desired_behavior != "normal":
        raise ValueError(f"Unknown behavior {desired_behavior!r}.")

    nominal_cells = set(problem.nominal_path)
    path_deviation_steps = sum(cell not in nominal_cells for cell in best.positions)
    return {
        "behavior": desired_behavior,
        "best_plan": best,
        "no_wait_steps": no_wait_steps,
        "wait_advantage_steps": wait_advantage,
        "pair_costs": pair_costs,
        "nominal_gate_cost": nominal_cost,
        "best_alternative_gate_cost": alternative_cost,
        "safe_detour_steps": safe_detour_steps,
        "path_deviation_steps": path_deviation_steps,
        "nominal_path_collision": nominal_path_collision,
    }


def _decision_step(
    problem,
    plan: SafePlan,
    behavior: str,
) -> int | None:
    if behavior == "wait":
        return next(
            (
                step
                for step in range(1, len(plan.positions))
                if plan.positions[step] == plan.positions[step - 1]
            ),
            None,
        )
    compared = min(len(plan.positions), len(problem.nominal_path))
    return next(
        (
            step
            for step in range(1, compared)
            if plan.positions[step] != problem.nominal_path[step]
        ),
        None,
    )


def _decision_observability(
    problem,
    scenario: DynamicScenario,
    plan: SafePlan,
    behavior: str,
    observation_radius: int,
    relevant_obstacle_indices: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    if behavior == "normal":
        return {
            "decision_step": None,
            "decision_position": None,
            "nearest_dynamic_distance_at_decision": None,
            "observable_decisive_obstacle_index": None,
        }
    step = _decision_step(problem, plan, behavior)
    if step is None:
        return None
    observation_time = step - 1
    decision_position = plan.positions[observation_time]
    selected_indices = (
        tuple(range(len(scenario.obstacles)))
        if relevant_obstacle_indices is None
        else tuple(relevant_obstacle_indices)
    )
    if not selected_indices:
        return None
    dynamic_cells = [
        (index, obstacle_positions(spec, observation_time)[observation_time])
        for index, spec in enumerate(scenario.obstacles)
        if index in selected_indices
    ]
    nearest_index, nearest = min(
        (
            (index, chebyshev(decision_position, cell))
            for index, cell in dynamic_cells
        ),
        key=lambda item: item[1],
    )
    if nearest > observation_radius:
        return None
    return {
        "decision_step": step,
        "decision_position": list(decision_position),
        "nearest_dynamic_distance_at_decision": nearest,
        "observable_decisive_obstacle_index": nearest_index,
    }


def _difficulty_score(
    record: Mapping[str, Any],
    observation_radius: int,
) -> float:
    decision_distance = record["nearest_dynamic_distance_at_decision"]
    reaction_pressure = (
        0.0
        if decision_distance is None
        else max(0.0, observation_radius - float(decision_distance))
    )
    return float(
        2.0 * int(record["reference_demo_collision_count"])
        + 2.0 * int(record["counterfactually_decisive_obstacle_count"])
        + max(0, int(record["safe_detour_steps"]))
        + 0.25 * int(record["path_deviation_steps"])
        + reaction_pressure
    )


def _candidate_record(
    scenario: DynamicScenario,
    masks: Sequence[int],
    classification: Mapping[str, Any],
    observability: Mapping[str, Any],
    relevance: Sequence[Mapping[str, Any]],
    demo_count: int,
    observation_radius: int,
) -> dict[str, Any]:
    full_mask = _full_mask(masks)
    plan = classification["best_plan"]
    obstacle_rows = []
    for spec, relevance_row in zip(scenario.obstacles, relevance):
        obstacle_rows.append(
            {
                "route_id": spec.label,
                "start_index": spec.start_index,
                "direction": spec.direction,
                "move_every": spec.move_every,
                **relevance_row,
            }
        )
    primary_original_index = observability.get(
        "observable_decisive_obstacle_index"
    )
    primary_route_id = None
    if classification["behavior"] != "normal":
        if primary_original_index is None:
            raise ValueError("A behavioral scenario must have an observable cause.")
        primary_original_index = int(primary_original_index)
        order = [primary_original_index] + [
            index
            for index in range(len(obstacle_rows))
            if index != primary_original_index
        ]
        obstacle_rows = [obstacle_rows[index] for index in order]
        primary_route_id = obstacle_rows[0]["route_id"]
        observability = {
            **observability,
            "observable_decisive_obstacle_index": 0,
        }
    record = {
        "scenario_id": -1,
        "difficulty_stratum": "unassigned",
        "difficulty_score": 0.0,
        "required_behavior": classification["behavior"],
        "primary_obstacle_index": (
            0 if classification["behavior"] != "normal" else None
        ),
        "primary_route_id": primary_route_id,
        "primary_original_obstacle_index": primary_original_index,
        "reference_demo_collision_count": full_mask.bit_count(),
        "reference_demo_collision_rate": full_mask.bit_count() / demo_count,
        "nominal_path_collision": classification["nominal_path_collision"],
        "minimum_safe_path_steps": plan.steps,
        "safe_detour_steps": classification["safe_detour_steps"],
        "path_deviation_steps": classification["path_deviation_steps"],
        "oracle_wait_count": plan.wait_count,
        "no_wait_safe_path_steps": classification["no_wait_steps"],
        "wait_advantage_steps": classification["wait_advantage_steps"],
        "nominal_gate_cost": classification["nominal_gate_cost"],
        "best_alternative_gate_cost": classification[
            "best_alternative_gate_cost"
        ],
        **observability,
        "active_demo_blocker_count": sum(
            bool(item["active_demo_blocker"]) for item in relevance
        ),
        "counterfactually_decisive_obstacle_count": sum(
            bool(item["counterfactually_decisive"]) for item in relevance
        ),
        "oracle_path": [list(cell) for cell in plan.positions],
        "obstacles": obstacle_rows,
    }
    record["difficulty_score"] = _difficulty_score(record, observation_radius)
    return record


def _build_behavior_pool(
    problem,
    split: str,
    behavior: str,
    target_size: int,
    records,
    demo_paths,
    design: Mapping[str, Any],
    rng: random.Random,
    signatures: set[tuple],
) -> list[dict[str, Any]]:
    move_every = int(design["move_every"])
    options = [_route_options(record, move_every, demo_paths) for record in records]
    combinations = _candidate_combinations(records, behavior)
    if not combinations:
        raise RuntimeError(f"No {split} route combinations exist for {behavior}.")
    combination_use = Counter()
    accepted = []
    rejected = Counter()
    attempts = 0
    maximum_attempts = target_size * int(design["max_search_attempts_per_candidate"])
    while len(accepted) < target_size:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError(
                f"Could not build {target_size} {split} {behavior} candidates; "
                f"accepted={len(accepted)} rejected={dict(rejected)}."
            )
        if attempts % 1000 == 0:
            print(
                f"[{split} {behavior}] searching pool={len(accepted)}/{target_size} "
                f"attempts={attempts} rejected={dict(rejected)}",
                flush=True,
            )
        minimum_use = min(combination_use[item] for item in combinations)
        eligible = [
            item for item in combinations if combination_use[item] == minimum_use
        ]
        combination = rng.choice(eligible)
        combination_use[combination] += 1
        chosen = [rng.choice(options[index]) for index in combination]
        specs = tuple(item[0] for item in chosen)
        masks = tuple(int(item[1]) for item in chosen)
        signature = tuple(
            (spec.label, spec.start_index, spec.direction) for spec in specs
        )
        if signature in signatures:
            rejected["duplicate"] += 1
            continue
        full_mask = _full_mask(masks)
        if behavior != "normal" and full_mask == 0:
            rejected["no_astar_demo_conflict"] += 1
            continue
        scenario = DynamicScenario(seed=0, obstacles=specs)
        classification = _classify_candidate(
            problem,
            scenario,
            behavior,
            design,
        )
        if classification is None:
            rejected["behavior_rule"] += 1
            continue
        relevance = _counterfactual_relevance(
            problem,
            scenario,
            masks,
            classification["pair_costs"],
        )
        decisive_count = sum(
            bool(item["counterfactually_decisive"]) for item in relevance
        )
        if behavior != "normal" and decisive_count < int(
            design["minimum_decisive_obstacles"]
        ):
            rejected["no_decisive_obstacle"] += 1
            continue
        relevant_indices = tuple(
            index
            for index, item in enumerate(relevance)
            if bool(item["counterfactually_decisive"])
        )
        observability = _decision_observability(
            problem,
            scenario,
            classification["best_plan"],
            behavior,
            int(design["observation_radius"]),
            relevant_indices if behavior != "normal" else None,
        )
        if observability is None:
            rejected["decisive_obstacle_not_observable"] += 1
            continue
        accepted.append(
            _candidate_record(
                scenario,
                masks,
                classification,
                observability,
                relevance,
                len(demo_paths),
                int(design["observation_radius"]),
            )
        )
        signatures.add(signature)
        if len(accepted) == 1 or len(accepted) % 10 == 0:
            print(
                f"[{split} {behavior}] pool={len(accepted)}/{target_size} "
                f"attempts={attempts} rejected={dict(rejected)}",
                flush=True,
            )
    return accepted


def _largest_remainder_targets(
    total: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    raw = {name: total * float(fractions[name]) for name in DIFFICULTIES}
    result = {name: int(math.floor(raw[name])) for name in DIFFICULTIES}
    remaining = total - sum(result.values())
    order = sorted(
        DIFFICULTIES,
        key=lambda name: (raw[name] - result[name], -DIFFICULTIES.index(name)),
        reverse=True,
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


def _select_stratified(
    pool: Sequence[dict[str, Any]],
    target_count: int,
    fractions: Mapping[str, float],
    rng: random.Random,
    score_targets: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    targets = _largest_remainder_targets(target_count, fractions)
    if score_targets is not None:
        absolute_targets = tuple(
            float(score_targets[difficulty]) for difficulty in DIFFICULTIES
        )
        ranked = sorted(
            (
                (
                    min(
                        abs(float(record["difficulty_score"]) - target)
                        for target in absolute_targets
                    ),
                    rng.random(),
                    record,
                )
                for record in pool
            ),
            key=lambda item: (item[0], item[1]),
        )
        selected = sorted(
            (item[2] for item in ranked[:target_count]),
            key=lambda item: float(item["difficulty_score"]),
        )
        cursor = 0
        for difficulty in DIFFICULTIES:
            next_cursor = cursor + targets[difficulty]
            for record in selected[cursor:next_cursor]:
                record["difficulty_stratum"] = difficulty
            cursor = next_cursor
        return selected

    ranked = sorted(pool, key=lambda item: float(item["difficulty_score"]))
    first = len(ranked) // 3
    second = 2 * len(ranked) // 3
    bins = {
        "easy": ranked[:first],
        "medium": ranked[first:second],
        "hard": ranked[second:],
    }
    selected = []
    for difficulty in DIFFICULTIES:
        if len(bins[difficulty]) < targets[difficulty]:
            raise RuntimeError(
                f"Difficulty pool {difficulty} has {len(bins[difficulty])}, "
                f"needs {targets[difficulty]}."
            )
        chosen = rng.sample(bins[difficulty], targets[difficulty])
        for record in chosen:
            record["difficulty_stratum"] = difficulty
        selected.extend(sorted(chosen, key=lambda item: item["difficulty_score"]))
    return selected


def _build_split_scenarios(problem, config, split, records, demo_paths, design):
    search_seed = int(config["spatial_generalization"]["behavior_search_seed"])
    rng = random.Random(search_seed + 10_000 * (SPLITS.index(split) + 1))
    signatures: set[tuple] = set()
    selected = []
    multiplier = int(design["candidate_pool_multiplier"])
    for behavior in BEHAVIORS:
        target_count = int(design["behavior_counts"][split][behavior])
        behavior_multiplier = (
            int(design["normal_candidate_pool_multiplier"])
            if behavior == "normal"
            else multiplier
        )
        pool_target = max(target_count * behavior_multiplier, target_count + 6)
        pool = _build_behavior_pool(
            problem,
            split,
            behavior,
            pool_target,
            records,
            demo_paths,
            {**design, "move_every": config["spatial_generalization"]["move_every"]},
            rng,
            signatures,
        )
        if behavior == "normal":
            # No zero-conflict phase was observed in the preceding 480,000-candidate
            # search when all five topology-critical routes were active. Controls
            # are therefore feasible nominal-path cases with the lowest observed
            # demonstration conflict, selected without consulting a learned policy.
            normal_target = design.get(
                "normal_reference_demo_collision_target"
            )
            chosen = sorted(
                pool,
                key=lambda item: (
                    (
                        int(item["reference_demo_collision_count"])
                        if normal_target is None
                        else abs(
                            int(item["reference_demo_collision_count"])
                            - int(normal_target)
                        )
                    ),
                    float(item["difficulty_score"]),
                ),
            )[:target_count]
            for record in chosen:
                record["difficulty_stratum"] = "control"
            selected.extend(chosen)
        else:
            selection_rng = random.Random(
                search_seed
                + 1_000_000 * (SPLITS.index(split) + 1)
                + 10_000 * (BEHAVIORS.index(behavior) + 1)
            )
            selected.extend(
                _select_stratified(
                    pool,
                    target_count,
                    design["difficulty_fractions"],
                    selection_rng,
                    design.get("difficulty_score_targets", {}).get(behavior),
                )
            )
    behavior_order = {name: index for index, name in enumerate(BEHAVIORS)}
    difficulty_order = {
        "control": 0,
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }
    selected.sort(
        key=lambda item: (
            behavior_order[item["required_behavior"]],
            difficulty_order[item["difficulty_stratum"]],
            float(item["difficulty_score"]),
        )
    )
    for index, record in enumerate(selected):
        record["scenario_id"] = SCENARIO_OFFSETS[split] + index
    return selected


def _validate_dataset_semantics(manifest, design: Mapping[str, Any]) -> None:
    radius = int(design["observation_radius"])
    for split in SPLITS:
        scenarios = manifest["scenarios"][split]
        actual_behaviors = Counter(
            source["required_behavior"] for source in scenarios
        )
        if actual_behaviors != Counter(design["behavior_counts"][split]):
            raise ValueError(f"{split} behavior distribution does not match its design.")
        route_use = Counter(item["route_id"] for source in scenarios for item in source["obstacles"])
        expected_routes = {
            record["route_id"]
            for record in manifest["route_pools"][split]["corridor"]
        }
        if set(route_use) != expected_routes or min(route_use.values()) <= 0:
            raise ValueError(f"{split} leaves at least one route variant unused.")
        for behavior in BEHAVIORS:
            group = [
                source
                for source in scenarios
                if source["required_behavior"] == behavior
            ]
            if behavior == "normal":
                if any(
                    source["difficulty_stratum"] != "control"
                    or source["oracle_wait_count"] != 0
                    or source["safe_detour_steps"] > 0
                    or source["nominal_path_collision"]
                    for source in group
                ):
                    raise ValueError(f"{split} contains an invalid normal control.")
                continue
            expected_difficulty = _largest_remainder_targets(
                len(group), design["difficulty_fractions"]
            )
            actual_difficulty = Counter(
                source["difficulty_stratum"] for source in group
            )
            if actual_difficulty != Counter(expected_difficulty):
                raise ValueError(
                    f"{split} {behavior} difficulty distribution is unbalanced."
                )
            if any(
                source["decision_step"] is None
                or source["nearest_dynamic_distance_at_decision"] is None
                or source["nearest_dynamic_distance_at_decision"] > radius
                or source["counterfactually_decisive_obstacle_count"]
                < int(design["minimum_decisive_obstacles"])
                for source in group
            ):
                raise ValueError(
                    f"{split} {behavior} contains an unobservable or irrelevant case."
                )
            if any(
                source.get("primary_obstacle_index") != 0
                or source.get("observable_decisive_obstacle_index") != 0
                or source.get("primary_route_id")
                != source["obstacles"][0]["route_id"]
                or not source["obstacles"][0]["counterfactually_decisive"]
                for source in group
            ):
                raise ValueError(
                    f"{split} {behavior} does not place its observable causal "
                    "obstacle first."
                )
            if behavior == "wait" and any(
                source["oracle_wait_count"]
                < int(design["minimum_oracle_wait_steps"])
                or (
                    source["no_wait_safe_path_steps"] is not None
                    and source["wait_advantage_steps"]
                    < int(design["minimum_wait_advantage_steps"])
                )
                or source["best_alternative_gate_cost"]
                - source["nominal_gate_cost"]
                < int(design.get("minimum_wait_gate_margin_steps", 0))
                for source in group
            ):
                raise ValueError(f"{split} contains an invalid wait case.")
            if behavior == "avoidance" and any(
                source["oracle_wait_count"] != 0
                or source["wait_advantage_steps"] != 0
                or source["best_alternative_gate_cost"]
                - source["nominal_gate_cost"]
                < int(design.get("minimum_avoidance_gate_margin_steps", 0))
                for source in group
            ):
                raise ValueError(f"{split} contains an invalid avoidance case.")
            if behavior == "reroute" and any(
                source["oracle_wait_count"] != 0
                or source["wait_advantage_steps"] != 0
                or source["nominal_gate_cost"]
                - source["best_alternative_gate_cost"]
                < int(design["minimum_reroute_margin_steps"])
                for source in group
            ):
                raise ValueError(f"{split} contains an invalid reroute case.")
            means = {
                difficulty: sum(
                    float(source["difficulty_score"])
                    for source in group
                    if source["difficulty_stratum"] == difficulty
                )
                / actual_difficulty[difficulty]
                for difficulty in DIFFICULTIES
                if actual_difficulty[difficulty]
            }
            if any(
                means[right] < means[left]
                for left, right in zip(DIFFICULTIES, DIFFICULTIES[1:])
            ):
                raise ValueError(f"{split} {behavior} difficulty scores are not ordered.")
        normal_group = [
            source
            for source in scenarios
            if source["required_behavior"] == "normal"
        ]
        behavioral_group = [
            source
            for source in scenarios
            if source["required_behavior"] != "normal"
        ]
        normal_mean = sum(
            source["reference_demo_collision_count"] for source in normal_group
        ) / len(normal_group)
        behavioral_mean = sum(
            source["reference_demo_collision_count"] for source in behavioral_group
        ) / len(behavioral_group)
        if normal_mean >= behavioral_mean:
            raise ValueError(
                f"{split} controls are not lower-conflict than behavioral cases."
            )


def _reference_demonstrations(problem, config):
    demo_config = config["demonstrations"]
    demo_count = int(demo_config["episodes"])
    demo_seed = int(demo_config["seed"])
    if not bool(demo_config.get("use_frozen_dataset", False)):
        return _reference_demo_paths(problem, demo_count, demo_seed), {
            "frozen": False,
            "path": None,
            "sha256": None,
        }

    demo_path = _resolve(str(demo_config["file"]))
    metadata_path = demo_path.with_suffix(".json")
    if not demo_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing frozen A* demonstration dataset {demo_path}. Collect it "
            "before generating dynamic scenarios."
        )
    transitions = load_demonstrations(demo_path)
    reward_config = RewardConfig(
        **{key: float(value) for key, value in config["reward"].items()}
    )
    environment = config["environment"]
    expected_signature = demonstration_signature(
        [problem],
        episodes=demo_count,
        seed=demo_seed,
        max_steps=int(environment["max_steps"]),
        reward_config=reward_config,
        window_size=int(environment["window_size"]),
        spatial_channels=int(environment["spatial_channels"]),
        observation_mode=str(environment["observation"]),
        environment_id=str(
            demo_config.get(
                "environment_id", "static_nominal_zero_dynamic_channels"
            )
        ),
    )
    metadata = load_json(metadata_path)
    validate_demonstration_dataset(
        transitions,
        metadata,
        expected_signature,
        demo_path,
    )
    paths = paths_from_demonstrations(problem, transitions)
    if len(paths) != demo_count:
        raise ValueError(
            f"Frozen dataset contains {len(paths)} paths, expected {demo_count}."
        )
    return paths, {
        "frozen": True,
        "path": str(demo_path),
        "metadata_path": str(metadata_path),
        "sha256": demonstration_file_sha256(demo_path),
    }


def _validate_route_geometry_isolation(route_pools) -> None:
    routes = {
        split: {
            tuple(tuple(cell) for cell in record["route"])
            for category in ("corridor", "background")
            for record in route_pools[split][category]
        }
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = routes[left].intersection(routes[right])
            if overlap:
                raise ValueError(
                    f"Route-geometry leakage between {left} and {right}: "
                    f"{len(overlap)} exact routes."
                )


def _share_training_route_templates(route_pools) -> None:
    """Use one route-template distribution with split-specific identifiers.

    This keeps the causal-conflict study in-distribution while the independently
    sampled phases/directions/combinations keep complete scenarios disjoint.
    Route-geometry holdout remains available as a separate stress benchmark.
    """

    source = route_pools["train"]
    for split in ("validation", "test"):
        cloned = {"corridor": [], "background": []}
        for category in ("corridor", "background"):
            for index, record in enumerate(source[category]):
                item = copy.deepcopy(record)
                item["route_id"] = f"{split}_shared_{category}_{index:02d}"
                item["shared_template_source_route_id"] = record["route_id"]
                cloned[category].append(item)
        route_pools[split] = cloned


def _validate_exact_scenario_isolation(route_pools, scenarios) -> None:
    fingerprints = {}
    for split in SPLITS:
        lookup = {
            record["route_id"]: tuple(tuple(cell) for cell in record["route"])
            for category in ("corridor", "background")
            for record in route_pools[split][category]
        }
        fingerprints[split] = {
            tuple(
                sorted(
                    (
                        lookup[obstacle["route_id"]],
                        int(obstacle["start_index"]),
                        int(obstacle["direction"]),
                        int(obstacle["move_every"]),
                    )
                    for obstacle in scenario["obstacles"]
                )
            )
            for scenario in scenarios[split]
        }
        if len(fingerprints[split]) != len(scenarios[split]):
            raise ValueError(f"{split} contains duplicate complete scenarios.")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = fingerprints[left].intersection(fingerprints[right])
            if overlap:
                raise ValueError(
                    f"Complete-scenario leakage between {left} and {right}: "
                    f"{len(overlap)} duplicate configurations."
                )


def build_manifest(problem, config):
    design = _scenario_design(config)
    if not math.isclose(float(config["reward"]["stay"]), 0.0, abs_tol=1e-12):
        raise ValueError(
            "The v9 wait benchmark requires reward.stay=0 so waiting costs one "
            "ordinary environment step instead of receiving an extra penalty."
        )
    demo_count = int(config["demonstrations"]["episodes"])
    demo_seed = int(config["demonstrations"]["seed"])
    demo_paths, demo_dataset = _reference_demonstrations(problem, config)
    route_pools = _build_route_pools(problem, config, demo_paths)
    _install_behavior_anchors(problem, config, route_pools, demo_paths)
    share_route_templates = bool(
        config["spatial_generalization"].get(
            "shared_route_templates_across_splits", False
        )
    )
    if share_route_templates:
        _share_training_route_templates(route_pools)
    else:
        _validate_route_geometry_isolation(route_pools)
    scenarios = {
        split: _build_split_scenarios(
            problem,
            config,
            split,
            route_pools[split]["corridor"],
            demo_paths,
            design,
        )
        for split in SPLITS
    }
    _validate_exact_scenario_isolation(route_pools, scenarios)
    spatial = config["spatial_generalization"]
    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "office_behavior_balanced_v9",
            "seed": int(spatial["generation_seed"]),
            "behavior_search_seed": int(spatial["behavior_search_seed"]),
            "route_length": int(spatial["route_length"]),
            "reference_path_count": int(spatial["reference_path_count"]),
            "reference_demo_episodes": demo_count,
            "reference_demo_seed": demo_seed,
            "reference_demo_dataset": demo_dataset,
            "separation_radius": int(spatial["separation_radius"]),
            "move_every": int(spatial["move_every"]),
            "corridor_per_scenario": 5,
            "background_per_scenario": 0,
            "cross_split_spatial_disjoint": False,
            "cross_split_route_geometry_disjoint": not share_route_templates,
            "shared_route_templates_across_splits": share_route_templates,
            "complete_scenarios_cross_split_disjoint": True,
            "split_scope": (
                "shared route templates with held-out motion phases and complete "
                "five-obstacle configurations"
                if share_route_templates
                else "held-out route variants and motion phases on shared Office topology"
            ),
            "behavior_counts": design["behavior_counts"],
            "difficulty_fractions": design["difficulty_fractions"],
            "difficulty_score_targets": design.get(
                "difficulty_score_targets"
            ),
            "normal_reference_demo_collision_target": design.get(
                "normal_reference_demo_collision_target"
            ),
            "behavior_definitions": {
                "normal": (
                    "the registered nominal A* path remains collision-free and the "
                    "dynamic safe path needs no wait or detour; select the lowest "
                    "reference-demo-conflict candidates as controls"
                ),
                "wait": (
                    "a shortest nominal-gate safe plan uses STAY and is strictly "
                    "shorter than every no-STAY plan"
                ),
                "avoidance": (
                    "a no-wait local deviation keeps the nominal gate pair optimal"
                ),
                "reroute": (
                    "a no-wait alternative gate pair is strictly better than the "
                    "nominal gate pair"
                ),
            },
            "difficulty_rule": (
                "behavior-specific score quantiles combining demo conflicts, "
                "decisive obstacles, detour, deviation, and reaction distance"
            ),
            "observability_rule": (
                "the causal dynamic obstacle is inside the configured local window "
                "when the oracle first waits or deviates"
            ),
            "placement_rule": (
                "one route in each of five topology-critical zones; relevance is "
                "required across the dataset, not from all five obstacles in every case"
            ),
            "control_selection_rule": (
                "lowest reference-demo conflict among feasible nominal-path cases; "
                "zero conflict across all 20 randomized A* paths was not observed "
                "in the preceding 480000-candidate search with five critical routes"
            ),
        },
        "route_pools": route_pools,
        "scenarios": scenarios,
    }
    _validate_dataset_semantics(manifest, design)
    validate_spatial_scenario_manifest(problem, manifest)
    return manifest


def _next_cell(route, start_index: int, direction: int):
    candidate = start_index + direction
    if candidate < 0 or candidate >= len(route):
        candidate = start_index - direction
    return route[candidate]


def render_route_pools(problem, manifest, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=180)
    for axis, split in zip(axes, SPLITS):
        records = manifest["route_pools"][split]["corridor"]
        _draw_base(axis, problem, f"{split.title()}: {len(records)} critical routes", ticks=True)
        for record in records:
            route = record["route"]
            color = ZONE_COLORS[record["critical_zone"]]
            axis.plot(
                [cell[1] for cell in route],
                [cell[0] for cell in route],
                color=color,
                linewidth=2.2,
                alpha=0.95,
            )
            row, column = record["center"]
            axis.scatter(column, row, color=color, s=22, zorder=5)
    legend = [
        Line2D([0], [0], color=color, linewidth=2.3, label=ZONE_LABELS[zone])
        for zone, color in ZONE_COLORS.items()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        f"{_dataset_label(manifest)} route pools: five critical zones with held-out route variants",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_gallery(problem, manifest, split: str, output: Path) -> None:
    scenarios = manifest["scenarios"][split]
    lookup = {
        record["route_id"]: record
        for record in manifest["route_pools"][split]["corridor"]
    }
    columns = 10 if len(scenarios) >= 50 else 5
    rows = math.ceil(len(scenarios) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.8 * rows), dpi=140)
    axes = np.asarray(axes).reshape(-1)
    for axis, source in zip(axes, scenarios):
        behavior = source["required_behavior"]
        difficulty = source["difficulty_stratum"]
        pair_id = source.get("pair_id")
        pair_role = source.get("pair_role")
        pair_prefix = ""
        if pair_id:
            role_label = "X" if pair_role == "conflict" else "C"
            pair_prefix = f"{pair_id}:{role_label} | "
        title = (
            f"{pair_prefix}{source['scenario_id']} | "
            f"{behavior.upper()}-{difficulty.upper()} | "
            f"demo={source['reference_demo_collision_count']}/20"
        )
        _draw_base(axis, problem, title)
        oracle_path = source["oracle_path"]
        axis.plot(
            [cell[1] for cell in oracle_path],
            [cell[0] for cell in oracle_path],
            color=BEHAVIOR_COLORS[behavior],
            linewidth=1.25,
            linestyle="--",
            alpha=0.9,
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
            is_primary = (
                source.get("primary_route_id") is not None
                and obstacle["route_id"] == source["primary_route_id"]
            )
            if is_primary:
                color = (
                    "#c62828"
                    if source.get("pair_role") != "matched_control"
                    else "#1565c0"
                )
            axis.plot(
                [cell[1] for cell in route],
                [cell[0] for cell in route],
                color=color,
                linewidth=2.5 if is_primary else 1.25,
                alpha=1.0 if is_primary else 0.72,
            )
            initial = route[int(obstacle["start_index"])]
            following = _next_cell(
                route,
                int(obstacle["start_index"]),
                int(obstacle["direction"]),
            )
            axis.scatter(
                initial[1], initial[0], color="#d32f2f", s=15, zorder=7
            )
            axis.annotate(
                "",
                xy=(following[1], following[0]),
                xytext=(initial[1], initial[0]),
                arrowprops={"arrowstyle": "->", "color": "#212121", "lw": 0.65},
            )
    for axis in axes[len(scenarios) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"{_dataset_label(manifest)} {split}: control and behavior-balanced dynamic scenarios",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_dataset_summary(manifest, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    split_colors = {"train": "#1976d2", "validation": "#7b1fa2", "test": "#ef6c00"}
    x = np.arange(len(BEHAVIORS), dtype=float)
    width = 0.24
    for split_index, split in enumerate(SPLITS):
        counts = Counter(
            source["required_behavior"] for source in manifest["scenarios"][split]
        )
        axes[0].bar(
            x + (split_index - 1) * width,
            [counts[name] for name in BEHAVIORS],
            width,
            color=split_colors[split],
            label=split.title(),
        )
    axes[0].set_xticks(x, [BEHAVIOR_LABELS[name] for name in BEHAVIORS], rotation=18)
    axes[0].set_title("Behavior composition")
    axes[0].legend(frameon=False, fontsize=8)

    strata = ("control", *DIFFICULTIES)
    x = np.arange(len(strata), dtype=float)
    for split_index, split in enumerate(SPLITS):
        counts = Counter(
            source["difficulty_stratum"] for source in manifest["scenarios"][split]
        )
        axes[1].bar(
            x + (split_index - 1) * width,
            [counts[name] for name in strata],
            width,
            color=split_colors[split],
        )
    axes[1].set_xticks(x, [name.title() for name in strata])
    axes[1].set_title("Difficulty composition")

    x = np.arange(len(BEHAVIORS), dtype=float)
    for split_index, split in enumerate(SPLITS):
        values = []
        for behavior in BEHAVIORS:
            group = [
                source
                for source in manifest["scenarios"][split]
                if source["required_behavior"] == behavior
            ]
            values.append(
                sum(source["reference_demo_collision_count"] for source in group)
                / len(group)
            )
        axes[2].bar(
            x + (split_index - 1) * width,
            values,
            width,
            color=split_colors[split],
        )
    axes[2].set_xticks(x, [BEHAVIOR_LABELS[name] for name in BEHAVIORS], rotation=18)
    axes[2].set_ylim(0, int(manifest["generation"]["reference_demo_episodes"]))
    axes[2].set_title("Mean A* demo conflicts")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        f"{_dataset_label(manifest)} dataset audit before training",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _write_audit(manifest, output: Path):
    rows = []
    summaries = []
    for split in SPLITS:
        route_use = Counter(
            {
                record["route_id"]: 0
                for record in manifest["route_pools"][split]["corridor"]
            }
        )
        for source in manifest["scenarios"][split]:
            route_use.update(item["route_id"] for item in source["obstacles"])
            rows.append(
                {
                    "split": split,
                    "scenario_id": source["scenario_id"],
                    "required_behavior": source["required_behavior"],
                    "difficulty_stratum": source["difficulty_stratum"],
                    "difficulty_score": source["difficulty_score"],
                    "reference_demo_collision_count": source[
                        "reference_demo_collision_count"
                    ],
                    "nominal_path_collision": source["nominal_path_collision"],
                    "minimum_safe_path_steps": source["minimum_safe_path_steps"],
                    "safe_detour_steps": source["safe_detour_steps"],
                    "path_deviation_steps": source["path_deviation_steps"],
                    "oracle_wait_count": source["oracle_wait_count"],
                    "no_wait_safe_path_steps": source["no_wait_safe_path_steps"],
                    "wait_advantage_steps": source["wait_advantage_steps"],
                    "decision_step": source["decision_step"],
                    "nearest_dynamic_distance_at_decision": source[
                        "nearest_dynamic_distance_at_decision"
                    ],
                    "observable_decisive_obstacle_index": source[
                        "observable_decisive_obstacle_index"
                    ],
                    "active_demo_blocker_count": source[
                        "active_demo_blocker_count"
                    ],
                    "counterfactually_decisive_obstacle_count": source[
                        "counterfactually_decisive_obstacle_count"
                    ],
                }
            )
        group = [row for row in rows if row["split"] == split]
        summaries.append(
            {
                "split": split,
                "scenario_count": len(group),
                "behavior_counts": dict(
                    Counter(row["required_behavior"] for row in group)
                ),
                "difficulty_counts": dict(
                    Counter(row["difficulty_stratum"] for row in group)
                ),
                "mean_reference_demo_collisions": sum(
                    row["reference_demo_collision_count"] for row in group
                )
                / len(group),
                "normal_mean_reference_demo_collisions": sum(
                    row["reference_demo_collision_count"]
                    for row in group
                    if row["required_behavior"] == "normal"
                )
                / max(1, sum(
                    row["required_behavior"] == "normal" for row in group
                )),
                "normal_nominal_path_safe_rate": sum(
                    row["required_behavior"] == "normal"
                    and not row["nominal_path_collision"]
                    for row in group
                )
                / max(1, sum(
                    row["required_behavior"] == "normal" for row in group
                )),
                "behavior_decision_observable_rate": sum(
                    row["required_behavior"] == "normal"
                    or row["nearest_dynamic_distance_at_decision"] is not None
                    for row in group
                )
                / len(group),
                "minimum_route_usage": min(route_use.values()),
                "maximum_route_usage": max(route_use.values()),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scenario_dataset_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(summaries, output / "split_summary.json")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--audit-output",
        default="outputs/office_behavior_v9_dataset_design",
    )
    args = parser.parse_args()
    config = load_config(_resolve(args.config))
    problem = _load_problem(config)
    manifest = build_manifest(problem, config)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(spatial["manifest"])
    write_json(manifest, manifest_path)
    render_route_pools(problem, manifest, _resolve(spatial["route_pool_preview"]))
    render_dataset_summary(manifest, _resolve(spatial["dataset_summary_preview"]))
    for split, key in (
        ("train", "training_preview"),
        ("validation", "validation_preview"),
        ("test", "test_preview"),
    ):
        render_gallery(problem, manifest, split, _resolve(spatial[key]))
    summaries = _write_audit(manifest, _resolve(args.audit_output))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"manifest={manifest_path} sha256={digest[:12]}")
    for summary in summaries:
        print(
            f"{summary['split']}: scenarios={summary['scenario_count']} "
            f"behaviors={summary['behavior_counts']} "
            f"difficulty={summary['difficulty_counts']} "
            f"mean_demo_conflicts={summary['mean_reference_demo_collisions']:.2f} "
            f"observable={summary['behavior_decision_observable_rate']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
