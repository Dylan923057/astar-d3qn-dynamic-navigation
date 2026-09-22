"""Generate causal Office behavior scenarios and phase-matched test controls.

The v10 protocol keeps the structured Office topology, places one moving route
in each of five critical zones, and assigns exactly one obstacle as the causal
primary obstacle in every wait, avoidance, or reroute case.  Removing that
primary obstacle must turn the case into a strict normal control.  Test cases
are paired with a second scenario that changes only the primary obstacle phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
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

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import (
    SPLITS,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json

import generate_office_behavior_scenarios_v9 as v9
from generate_office_critical_scenarios_v5 import ZONE_COLORS
from generate_office_behavior_scenarios_v6 import GATE_CELLS, NOMINAL_GATE_PAIR
from astar_d3qn.evaluation.spatial_witness import motion_summary, shortest_simple_visible_plan, static_bypass_audit


DEFAULT_CONFIG = (
    "configs/"
    "dynamic_spatial_generalization_office_behavior_v10_paired_v1.yaml"
)
BEHAVIORS = v9.BEHAVIORS
CAUSAL_BEHAVIORS = ("wait", "avoidance", "reroute")
DIFFICULTIES = v9.DIFFICULTIES
SCENARIO_OFFSETS = v9.SCENARIO_OFFSETS


def _signature(specs: Sequence[DynamicObstacleSpec]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (spec.label, spec.start_index, spec.direction, spec.move_every)
            for spec in specs
        )
    )


def _route_options_for_speeds(record, move_every_choices, demo_paths):
    options = []
    for move_every in move_every_choices:
        options.extend(v9._route_options(record, int(move_every), demo_paths))
    return options


def _preflight_phase_pairs(problem, records, options, design, output):
    """Enumerate necessary phase conditions before expensive behavior planning.

    A normal full scene requires every obstacle to leave the canonical path
    safe.  Independent deterministic obstacle motion lets us check that cheaply
    per obstacle.  This is necessary, not sufficient, for a verified wait pair.
    """
    safe_options = {}
    primary_options = {}
    audit = []
    for index, (record, route_options) in enumerate(zip(records, options)):
        classified = [
            (spec, mask, v9._nominal_path_collides(
                problem, DynamicScenario(seed=0, obstacles=(spec,))
            ))
            for spec, mask in route_options
        ]
        safe_options[index] = [(spec, mask) for spec, mask, collision in classified if not collision]
        possible = []
        pairs = []
        for conflict, conflict_mask, collision in classified:
            if not collision:
                continue
            if design["require_primary_demo_conflict"] and not conflict_mask:
                continue
            for control, control_mask in safe_options[index]:
                if (
                    conflict.direction != control.direction
                    or conflict.move_every != control.move_every
                    or conflict.start_index == control.start_index
                ):
                    continue
                if design["require_strict_demo_conflict_reduction_in_pair"] and (
                    control_mask.bit_count() >= conflict_mask.bit_count()
                ):
                    continue
                pairs.append({
                    "conflict_start_index": conflict.start_index,
                    "control_start_index": control.start_index,
                    "direction": conflict.direction,
                    "move_every": conflict.move_every,
                    "conflict_demo_count": conflict_mask.bit_count(),
                    "control_demo_count": control_mask.bit_count(),
                })
            if any(
                pair["conflict_start_index"] == conflict.start_index
                and pair["direction"] == conflict.direction
                and pair["move_every"] == conflict.move_every
                for pair in pairs
            ):
                possible.append((conflict, conflict_mask))
        primary_options[index] = possible
        audit.append({
            "route_id": record["route_id"], "zone": record["critical_zone"],
            "route": record["route"], "motion_options": len(route_options),
            "nominal_safe_options": len(safe_options[index]),
            "potential_primary_options": len(possible),
            "phase_pairs_passing_necessary_conditions": pairs,
        })
    if output is not None:
        write_json({
            "scope": "exhaustive phases/directions/configured speeds for the supplied routes only",
            "wait_behavior_verified": False,
            "routes": audit,
        }, Path(output) / "route_phase_preflight.json")
    print(
        f"Route preflight: {sum(bool(value) for value in primary_options.values())}"
        f"/{len(records)} routes have potential phase pairs; behavior not yet verified.",
        flush=True,
    )
    return safe_options, primary_options


def _save_candidate(record, design, split, behavior, number):
    output = design.get("_search_audit_dir")
    if output is not None:
        # Save each accepted candidate, including a matched control if present,
        # before another search attempt or split can fail.  These are candidate
        # records, not a validated full training manifest or a resume cache.
        write_json(record, Path(output) / "accepted_candidates" /
                   f"{split}_{behavior}_{number:04d}.json")


def _save_visibility_failure(problem, scenario, classification, primary_index,
                             design, split, behavior, diagnostics):
    output = design.get("_search_audit_dir")
    retained = int(diagnostics.get("retained_visibility_failures", 0))
    if output is None or retained >= 5:
        return
    plan = classification["best_plan"]
    step = v9._decision_step(problem, plan, behavior)
    observation_time = None if step is None else step - 1
    spec = scenario.obstacles[primary_index]
    primary_cell = None if observation_time is None else v9.obstacle_positions(
        spec, observation_time
    )[observation_time]
    write_json({
        "purpose": "legacy_visibility_failure_not_an_acceptance_verdict",
        "grid_sha256": problem.grid_sha256, "required_behavior": behavior,
        "primary_obstacle_index": primary_index, "primary_route_id": spec.label,
        "obstacles": [{"route_id": item.label, "route": list(item.route),
                       "start_index": item.start_index, "direction": item.direction,
                       "move_every": item.move_every} for item in scenario.obstacles],
        "nominal_path": list(problem.nominal_path), "oracle_path": list(plan.positions),
        "oracle_steps": plan.steps, "oracle_wait_count": plan.wait_count,
        "legacy_decision_step": step, "observation_time": observation_time,
        "decision_position": None if observation_time is None else plan.positions[observation_time],
        "primary_position_at_decision": primary_cell,
        "distance_at_decision": None if primary_cell is None else v9.chebyshev(
            plan.positions[observation_time], primary_cell
        ),
        "observation_radius": int(design["observation_radius"]),
        "nominal_gate_cost": classification["nominal_gate_cost"],
        "best_alternative_gate_cost": classification["best_alternative_gate_cost"],
    }, Path(output) / "visibility_failures" / f"{split}_{behavior}_{retained + 1:04d}.json")
    diagnostics["retained_visibility_failures"] = retained + 1


def _raw_plan_observation(problem, scenario, plan, behavior, primary_index):
    """Describe the original plan's decision even when it is outside view."""
    step = v9._decision_step(problem, plan, behavior)
    if step is None:
        return {"decision_step": None, "decision_position": None,
                "nearest_dynamic_distance_at_decision": None,
                "observable_decisive_obstacle_index": None}
    position = plan.positions[step - 1]
    cell = v9.obstacle_positions(scenario.obstacles[primary_index], step - 1)[step - 1]
    return {"decision_step": step, "decision_position": list(position),
            "nearest_dynamic_distance_at_decision": v9.chebyshev(position, cell),
            "observable_decisive_obstacle_index": None}


def _visible_avoidance_evidence(problem, scenario, classification, primary_index, design):
    """Separate feasibility from full-information optimal cost; no detour cutoff."""
    radius = int(design["observation_radius"])
    blocked = set().union(*(cells for name, cells in GATE_CELLS.items() if name not in NOMINAL_GATE_PAIR))
    plan = v9.shortest_safe_plan(
        problem, scenario, allow_wait=False, additionally_blocked=blocked,
        visible_deviation_obstacle_index=primary_index, observation_radius=radius,
    )
    if plan is None:
        return None
    observation = v9._decision_observability(problem, scenario, plan, "avoidance", radius, (primary_index,))
    if observation is None:
        raise ValueError("Visible-avoidance witness failed visibility replay.")
    if plan.steps != len(plan.positions) - 1 or plan.wait_count != 0:
        raise ValueError("Visible-avoidance witness has inconsistent steps/waits.")
    if plan.positions[0] != problem.start or plan.positions[-1] != problem.goal:
        raise ValueError("Visible-avoidance witness has incorrect endpoints.")
    trajectories = [v9.obstacle_positions(spec, plan.steps) for spec in scenario.obstacles]
    for t, cell in enumerate(plan.positions):
        if cell in blocked or cell in problem.obstacles or not all(0 <= c < problem.size for c in cell):
            raise ValueError("Visible-avoidance witness left the allowed static/gate space.")
        if t:
            if sum(abs(a - b) for a, b in zip(cell, plan.positions[t - 1])) != 1:
                raise ValueError("Visible-avoidance witness waits or has an invalid move.")
            if any(cell in (positions[t - 1], positions[t]) for positions in trajectories):
                raise ValueError("Visible-avoidance witness collides during replay.")
    optimum = int(classification["best_plan"].steps)
    if plan.steps < optimum:
        raise ValueError("Constrained cost is below the full-information optimum.")
    return {"verified": True, "steps": plan.steps, "extra_steps": plan.steps - optimum,
            "wait_count": plan.wait_count, "path": [list(cell) for cell in plan.positions],
            "observation": observation, "observation_radius": radius,
            "scope": "nominal prefix until visible first departure; same gate pair; no STAY",
            "extra_steps_limit": None}


def _check_spatial_evidence(problem, scenario, primary_index, evidence, radius):
    path = tuple(map(tuple, evidence["path"]))
    primary = v9.obstacle_positions(scenario.obstacles[primary_index], len(problem.nominal_path) - 1)
    conflict_cell = next((cell for t, cell in enumerate(problem.nominal_path)
                          if t and cell in (primary[t - 1], primary[t])), None)
    blocked = set(problem.obstacles).union(*(cells for name, cells in GATE_CELLS.items() if name not in NOMINAL_GATE_PAIR))
    if conflict_cell is None or evidence["bypassed_cell"] != list(conflict_cell):
        raise ValueError("Spatial witness has an incorrect conflict cell.")
    blocked.add(conflict_cell)
    if not path or path[0] != problem.start or path[-1] != problem.goal or not motion_summary(path)["is_simple_path"]:
        raise ValueError("Spatial witness is not a simple start-to-goal path.")
    trajectories = [v9.obstacle_positions(spec, len(path) - 1) for spec in scenario.obstacles]
    for t, cell in enumerate(path):
        if cell in blocked or not all(0 <= c < problem.size for c in cell):
            raise ValueError("Spatial witness violates its geometric constraints.")
        if t and (sum(abs(a - b) for a, b in zip(cell, path[t - 1])) != 1
                  or any(cell in (cells[t - 1], cells[t]) for cells in trajectories)):
            raise ValueError("Spatial witness contains an invalid/unsafe action.")
    plan = v9.SafePlan(path, len(path) - 1, 0)
    observation = v9._decision_observability(problem, scenario, plan, "avoidance", radius, (primary_index,))
    if observation is None or observation != evidence["observation"] or evidence["steps"] != plan.steps:
        raise ValueError("Spatial witness visibility/step metadata is inconsistent.")


def _spatial_avoidance_evidence(problem, scenario, classification, primary_index, design, *, search_audit=None):
    # Preserve the solver's precise stop reason; aggregate status alone cannot
    # distinguish a static cut from exhaustive dynamic failure or a time limit.
    audit = search_audit if search_audit is not None else {}
    audit.clear()
    primary = v9.obstacle_positions(scenario.obstacles[primary_index], len(problem.nominal_path) - 1)
    cell = next((cell for t, cell in enumerate(problem.nominal_path)
                 if t and cell in (primary[t - 1], primary[t])), None)
    if cell is None:
        audit.update(status="no_nominal_primary_conflict", stop_reason="no_nominal_primary_conflict")
        return None, "no_nominal_primary_conflict"
    audit["bypassed_cell"] = list(cell)
    blocked = set().union(*(cells for name, cells in GATE_CELLS.items() if name not in NOMINAL_GATE_PAIR))
    remaining = min(5.0, float(design.get("_search_deadline", math.inf)) - time.monotonic())
    if remaining <= 0:
        audit.update(status="budget_exhausted", stop_reason="group_deadline_before_spatial_search")
        return None, "budget_exhausted"
    result = shortest_simple_visible_plan(problem, scenario, primary_index=primary_index,
                                         observation_radius=int(design["observation_radius"]),
                                         additionally_blocked=blocked | {cell}, max_seconds=remaining)
    audit.update(status=result.status, stop_reason=result.stop_reason,
                 expanded=result.expanded, generated=result.generated,
                 elapsed_seconds=result.elapsed_seconds)
    if result.plan is None:
        return None, result.status
    plan = result.plan
    evidence = {"path": [list(c) for c in plan.positions], "steps": plan.steps,
                "extra_steps": plan.steps - classification["best_plan"].steps,
                "bypassed_cell": list(cell), "motion": motion_summary(plan.positions),
                "observation": v9._decision_observability(problem, scenario, plan, "avoidance",
                                                         int(design["observation_radius"]), (primary_index,)),
                "scope": "visible first departure; same gates; no STAY/revisits; bypass first nominal primary-conflict cell"}
    _check_spatial_evidence(problem, scenario, primary_index, evidence, int(design["observation_radius"]))
    return evidence, "found"


def _save_spatial_failure(problem, scenario, primary_index, design, diagnostics, search_audit):
    output = design.get("_search_audit_dir")
    reason = search_audit["stop_reason"]
    counts = diagnostics.setdefault("spatial_stop_reasons", {})
    counts[reason] = counts.get(reason, 0) + 1
    retained = diagnostics.setdefault("retained_spatial_failures_by_reason", {})
    number = retained.get(reason, 0)
    if output is None or number >= 3:
        return
    write_json({
        "purpose": "spatial_witness_failure_not_a_general_avoidance_impossibility",
        "grid_sha256": problem.grid_sha256, "search": search_audit,
        "primary_obstacle_index": primary_index,
        "primary_route_id": scenario.obstacles[primary_index].label,
        "observation_radius": int(design["observation_radius"]),
        "nominal_path": list(problem.nominal_path),
        "obstacles": [{"route_id": spec.label, "route": list(spec.route),
                       "start_index": spec.start_index, "direction": spec.direction,
                       "move_every": spec.move_every} for spec in scenario.obstacles],
    }, Path(output) / "spatial_failures" / f"{reason}_{number + 1:04d}.json")
    retained[reason] = number + 1


def _empty_relevance(
    masks: Sequence[int],
    *,
    primary_index: int | None = None,
    primary_role: str = "distractor",
) -> list[dict[str, Any]]:
    full_mask = v9._full_mask(masks)
    rows = []
    for index, mask in enumerate(masks):
        reduced_mask = v9._full_mask(
            value for other_index, value in enumerate(masks) if other_index != index
        )
        is_primary = index == primary_index
        rows.append(
            {
                "individual_demo_collision_count": int(mask).bit_count(),
                "unique_demo_collision_contribution": (
                    full_mask.bit_count() - reduced_mask.bit_count()
                ),
                "changes_gate_pair_cost": False,
                "active_demo_blocker": bool(int(mask).bit_count()),
                "counterfactually_decisive": False,
                "behavior_causal": None,
                "causal_role": (
                    primary_role if is_primary else "controlled_covariate"
                ),
            }
        )
    return rows


def _causal_relevance(
    masks: Sequence[int],
    classification: Mapping[str, Any],
    reduced_classification: Mapping[str, Any],
    primary_index: int,
) -> list[dict[str, Any]]:
    rows = _empty_relevance(masks)
    changed_gate_cost = any(
        classification["pair_costs"][pair]
        != reduced_classification["pair_costs"][pair]
        for pair in classification["pair_costs"]
    )
    rows[primary_index].update(
        {
            "changes_gate_pair_cost": changed_gate_cost,
            "counterfactually_decisive": True,
            "behavior_causal": True,
            "causal_role": "primary",
        }
    )
    return rows


def _record_with_causal_metadata(
    scenario: DynamicScenario,
    masks: Sequence[int],
    classification: Mapping[str, Any],
    observability: Mapping[str, Any],
    relevance: Sequence[Mapping[str, Any]],
    demo_count: int,
    observation_radius: int,
    *,
    primary_index: int | None,
    reduced_classification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record = v9._candidate_record(
        scenario,
        masks,
        classification,
        observability,
        relevance,
        demo_count,
        observation_radius,
    )
    primary = None if primary_index is None else scenario.obstacles[primary_index]
    primary_mask = 0 if primary_index is None else int(masks[primary_index])
    record.update(
        {
            "primary_obstacle_index": primary_index,
            "primary_route_id": None if primary is None else primary.label,
            "primary_critical_zone": (
                None if primary is None else primary.reference_path_source
            ),
            "primary_start_index": None if primary is None else primary.start_index,
            "primary_direction": None if primary is None else primary.direction,
            "primary_move_every": None if primary is None else primary.move_every,
            "primary_reference_demo_collision_count": primary_mask.bit_count(),
            "causal_behavior_verified": primary_index is not None,
            "counterfactual_without_primary_behavior": (
                None
                if reduced_classification is None
                else reduced_classification["behavior"]
            ),
            "counterfactual_without_primary_steps": (
                None
                if reduced_classification is None
                else reduced_classification["best_plan"].steps
            ),
            "pair_id": None,
            "pair_role": "unpaired",
            "matched_behavior": None,
            "matched_conflict_difficulty": None,
            "paired_scenario_id": None,
            "pair_phase_delta": None,
        }
    )
    return record


def _normal_record(
    scenario: DynamicScenario,
    masks: Sequence[int],
    classification: Mapping[str, Any],
    demo_count: int,
    observation_radius: int,
    *,
    primary_index: int | None = None,
    primary_role: str = "distractor",
) -> dict[str, Any]:
    relevance = _empty_relevance(
        masks,
        primary_index=primary_index,
        primary_role=primary_role,
    )
    observability = {
        "decision_step": None,
        "decision_position": None,
        "nearest_dynamic_distance_at_decision": None,
        "observable_decisive_obstacle_index": None,
    }
    return _record_with_causal_metadata(
        scenario,
        masks,
        classification,
        observability,
        relevance,
        demo_count,
        observation_radius,
        primary_index=primary_index,
        reduced_classification=None,
    )


def _classify_without_obstacle(
    problem,
    scenario: DynamicScenario,
    obstacle_index: int,
    design: Mapping[str, Any],
) -> dict[str, Any] | None:
    reduced = DynamicScenario(
        seed=scenario.seed,
        obstacles=tuple(
            obstacle
            for index, obstacle in enumerate(scenario.obstacles)
            if index != obstacle_index
        ),
    )
    return v9._classify_candidate(problem, reduced, "normal", design)


def _matched_control(
    problem,
    conflict_scenario: DynamicScenario,
    conflict_masks: Sequence[int],
    classification: Mapping[str, Any],
    primary_index: int,
    primary_options: Sequence[tuple[DynamicObstacleSpec, int]],
    demo_count: int,
    design: Mapping[str, Any],
    reserved_signatures: set[tuple],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple] | None:
    conflict_primary = conflict_scenario.obstacles[primary_index]
    conflict_primary_count = int(conflict_masks[primary_index]).bit_count()
    candidates = []
    phase_counts = Counter() if diagnostics is None else diagnostics["phase_trials"]
    for replacement, replacement_mask in primary_options:
        if (
            replacement.label != conflict_primary.label
            or replacement.direction != conflict_primary.direction
            or replacement.move_every != conflict_primary.move_every
            or replacement.start_index == conflict_primary.start_index
        ):
            continue
        phase_counts["alternate_phase"] += 1
        specs = list(conflict_scenario.obstacles)
        specs[primary_index] = replacement
        candidate_signature = _signature(specs)
        if candidate_signature in reserved_signatures:
            phase_counts["duplicate"] += 1
            continue
        masks = list(conflict_masks)
        masks[primary_index] = int(replacement_mask)
        replacement_count = int(replacement_mask).bit_count()
        full_count = v9._full_mask(masks).bit_count()
        if bool(design["require_strict_demo_conflict_reduction_in_pair"]) and (
            replacement_count >= conflict_primary_count
        ):
            phase_counts["demo_conflict_not_reduced"] += 1
            continue
        scenario = DynamicScenario(seed=0, obstacles=tuple(specs))
        if v9._nominal_path_collides(problem, scenario):
            phase_counts["nominal_path_still_collides"] += 1
            continue
        normal = v9._classify_candidate(problem, scenario, "normal", design)
        if normal is None:
            phase_counts["normal_oracle_failed"] += 1
            continue
        phase_counts["valid_normal_control"] += 1
        phase_distance = abs(replacement.start_index - conflict_primary.start_index)
        candidates.append(
            (
                replacement_count,
                full_count,
                phase_distance,
                scenario,
                tuple(masks),
                normal,
                candidate_signature,
            )
        )
    if not candidates:
        return None
    (
        _,
        _,
        _,
        scenario,
        masks,
        normal,
        candidate_signature,
    ) = min(candidates, key=lambda item: item[:3])
    control = _normal_record(
        scenario,
        masks,
        normal,
        demo_count,
        int(design["observation_radius"]),
        primary_index=primary_index,
        primary_role="matched_control_primary",
    )
    control["causal_behavior_verified"] = False
    control["counterfactual_without_primary_behavior"] = "normal"
    control["counterfactual_without_primary_steps"] = classification[
        "counterfactual_without_primary_steps"
    ]
    return control, candidate_signature


def _build_pool(
    problem,
    split: str,
    behavior: str,
    target_size: int,
    records,
    demo_paths,
    design: Mapping[str, Any],
    rng: random.Random,
    signatures: set[tuple],
    *,
    require_pair: bool,
) -> list[dict[str, Any]]:
    diagnostics = {
        "split": split,
        "behavior": behavior,
        "target": target_size,
        "attempts": 0,
        "accepted": 0,
        "scenario_rejections": Counter(),
        "primary_trials": Counter(),
        "phase_trials": Counter(),
        "examples": {},
        "status": "running",
    }
    started = time.monotonic()
    seconds = design.get("_search_seconds")
    if seconds is not None:
        if not math.isfinite(float(seconds)) or float(seconds) <= 0:
            raise ValueError("Search seconds must be finite and positive.")
        design = {**design, "_search_deadline": started + float(seconds)}

    def report():
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 2)
        output = design.get("_search_audit_dir")
        if output is not None:
            write_json(diagnostics, Path(output) / f"{split}_{behavior}_search.json")

    try:
        result = _search_pool(
            problem, split, behavior, target_size, records, demo_paths, design,
            rng, signatures, require_pair=require_pair,
            diagnostics=diagnostics, report=report,
        )
        diagnostics["status"] = "complete"
        return result
    except KeyboardInterrupt:
        diagnostics["status"] = "interrupted"
        raise
    except Exception as error:
        diagnostics["status"] = "failed"
        diagnostics["error"] = str(error)
        raise
    finally:
        report()
        print(
            f"[{split} {behavior}] status={diagnostics['status']} "
            f"accepted={diagnostics['accepted']}/{target_size} "
            f"attempts={diagnostics['attempts']} "
            f"elapsed={diagnostics['elapsed_seconds']:.1f}s "
            f"primary={dict(diagnostics['primary_trials'])} "
            f"phase={dict(diagnostics['phase_trials'])}", flush=True,
        )


def _filter_spatial_primary_options(problem, primary_options, *, allowed_route_ids=None):
    """Necessary geometry test per PHASE, never a behavior/safety certificate.

    Do not remove entire routes: different phases can first conflict at
    different cells. Safe covariate options and wait/reroute proposals stay intact.
    """
    blocked = set().union(*(cells for name, cells in GATE_CELLS.items() if name not in NOMINAL_GATE_PAIR))
    cache, rows, filtered = {}, [], {}
    counts = Counter()
    for index, options in primary_options.items():
        filtered[index] = []
        for spec, mask in options:
            cell = None
            if allowed_route_ids is not None and spec.label not in allowed_route_ids:
                reason = "outside_target_approach_routes"
            else:
                primary = v9.obstacle_positions(spec, len(problem.nominal_path) - 1)
                cell = next((c for t, c in enumerate(problem.nominal_path)
                             if t and c in (primary[t - 1], primary[t])), None)
                if cell is None:
                    reason = "no_nominal_primary_conflict"
                else:
                    if cell not in cache:
                        cache[cell] = static_bypass_audit(problem, cell, additionally_blocked=blocked)
                    reason = cache[cell]["status"]
            counts[reason] += 1
            if reason == "static_bypass_possible":
                filtered[index].append((spec, mask))
            rows.append({"route_id": spec.label, "start_index": spec.start_index,
                         "direction": spec.direction, "move_every": spec.move_every,
                         "first_conflict_cell": None if cell is None else list(cell), "status": reason})
    return filtered, {"phase_counts": dict(counts), "phases": rows, "cell_geometry": list(cache.values()),
                      "scope": "necessary static bypass under fixed gates; no dynamic/visibility/behavior certification"}


def _search_pool(
    problem, split, behavior, target_size, records, demo_paths, design,
    rng, signatures, *, require_pair, diagnostics, report,
):
    move_every_choices = tuple(int(value) for value in design["move_every_choices"])
    options = [
        _route_options_for_speeds(record, move_every_choices, demo_paths)
        for record in records
    ]
    constructive = bool(design.get("_constructive_phase_pairs", False))
    if design.get("_visible_avoidance_pilot", False) and (not constructive or not require_pair):
        raise ValueError("Separated visible-avoidance acceptance is restricted to paired constructive pilots.")
    if constructive and (not require_pair or behavior == "normal"):
        raise ValueError("Constructive phase proposals require behavioral paired search.")
    # The one-pair feasibility pilot considers every route combination in the
    # existing split pool; the historical two-anchor filter is not imposed.
    factorized = constructive and bool(design.get("_factorized_proposals", False))
    combinations = [] if factorized else v9._candidate_combinations(
        records, "avoidance" if constructive else behavior
    )
    proposals = []
    if constructive:
        safe_options, primary_options = _preflight_phase_pairs(
            problem, records, options, design, design.get("_search_audit_dir")
        )
        if behavior == "avoidance" and design.get("_spatial_avoidance_pilot", False):
            primary_options, geometry_audit = _filter_spatial_primary_options(
                problem, primary_options, allowed_route_ids=design.get("_avoidance_primary_route_ids"))
            diagnostics["spatial_proposal_precheck"] = geometry_audit["phase_counts"]
            if design.get("_search_audit_dir") is not None:
                write_json(geometry_audit, Path(design["_search_audit_dir"]) / "spatial_proposal_precheck.json")
            print(f"[{split} avoidance] static proposal precheck: {geometry_audit['phase_counts']}", flush=True)
        if factorized:
            zone_order = list(ZONE_COLORS)
            safe_by_zone = {zone: [i for i, record in enumerate(records)
                                   if record["critical_zone"] == zone and safe_options[i]] for zone in zone_order}
            requested_zone = design.get("_proposal_primary_zone")
            for index, record in enumerate(records):
                zone = record["critical_zone"]
                if requested_zone is not None and zone != requested_zone:
                    continue
                if all(safe_by_zone[z] for z in zone_order if z != zone):
                    for option in primary_options[index]:
                        proposals.append((index, zone_order.index(zone), option))
        for combination in combinations:
            for primary_position, route_index in enumerate(combination):
                if all(
                    safe_options[index]
                    for position, index in enumerate(combination)
                    if position != primary_position
                ):
                    for primary_option in primary_options[route_index]:
                        proposals.append((combination, primary_position, primary_option))
        diagnostics["constructive_proposal_count"] = len(proposals)
        if not proposals:
            raise RuntimeError(
                "No phase-pair proposal satisfies the necessary conditions in the "
                "existing route pool. Inspect route_phase_preflight.json and, for spatial avoidance, spatial_proposal_precheck.json; "
                "full behavior search was not started."
            )
        rng.shuffle(proposals)
    if not combinations and not factorized:
        raise RuntimeError(f"No {split} route combinations exist for {behavior}.")
    combination_use = Counter()
    accepted = []
    rejected = diagnostics["scenario_rejections"]
    primary_counts = diagnostics["primary_trials"]
    attempts = 0
    maximum_attempts = target_size * int(design["max_search_attempts_per_candidate"])
    maximum_attempts = min(
        maximum_attempts, int(design.get("_diagnostic_max_attempts", maximum_attempts))
    )
    stall_limit = int(design.get("max_attempts_without_acceptance", 3000))
    if stall_limit <= 0 or maximum_attempts <= 0:
        raise ValueError("Search attempt and stall limits must be positive.")
    last_accept_attempt = 0
    last_report = time.monotonic()
    diagnostics["route_combination_count"] = None if factorized else len(combinations)
    diagnostics["proposal_mode"] = "factorized_by_primary_zone" if factorized else "enumerated_combinations"
    diagnostics["requested_primary_zone"] = design.get("_proposal_primary_zone")
    diagnostics["wait_route_rule"] = (
        "all_zone_combinations_with_preflight_phase_pairs" if constructive else
        "at_least_two_behavior_anchors" if behavior == "wait" else "all_zone_combinations"
    )
    diagnostics["maximum_attempts"] = maximum_attempts
    diagnostics["stall_limit"] = stall_limit
    diagnostics["primary_zone_attempts"] = Counter()
    diagnostics["accepted_primary_zones"] = Counter()

    def example(reason, scenario):
        if reason not in diagnostics["examples"]:
            diagnostics["examples"][reason] = [
                {"route_id": spec.label, "route": list(spec.route),
                 "start_index": spec.start_index, "direction": spec.direction,
                 "move_every": spec.move_every}
                for spec in scenario.obstacles
            ]

    while len(accepted) < target_size:
        # Cooperative limit: do not interrupt an oracle mid-computation.
        if time.monotonic() >= float(design.get("_search_deadline", math.inf)):
            raise RuntimeError(
                f"Search time budget reached for {split} {behavior}; "
                f"accepted={len(accepted)}/{target_size} attempts={attempts}."
            )
        if attempts >= maximum_attempts or attempts - last_accept_attempt >= stall_limit:
            raise RuntimeError(
                f"Could not build {target_size} {split} {behavior} candidates; "
                f"accepted={len(accepted)} attempts={attempts} "
                f"without_acceptance={attempts - last_accept_attempt} "
                f"rejected={dict(rejected)}. Inspect the search audit before retrying."
            )
        if attempts and (attempts % 100 == 0 or time.monotonic() - last_report >= 30):
            report()
            last_report = time.monotonic()
            print(
                f"[{split} {behavior}] pool={len(accepted)}/{target_size} "
                f"attempts={attempts} elapsed={diagnostics['elapsed_seconds']:.1f}s "
                f"rejected={dict(rejected)} "
                f"primary={dict(primary_counts)} phase={dict(diagnostics['phase_trials'])}",
                flush=True,
            )
        attempts += 1
        diagnostics["attempts"] = attempts
        if constructive:
            combination, proposed_primary, primary_option = proposals[(attempts - 1) % len(proposals)]
            if factorized:
                primary_route_index = combination
                combination = tuple(primary_route_index if position == proposed_primary else rng.choice(safe_by_zone[zone])
                                    for position, zone in enumerate(zone_order))
            zone = records[combination[proposed_primary]]["critical_zone"]
            diagnostics["primary_zone_attempts"][zone] += 1
            chosen = [
                primary_option if position == proposed_primary else rng.choice(safe_options[index])
                for position, index in enumerate(combination)
            ]
        else:
            minimum_use = min(combination_use[item] for item in combinations)
            eligible = [item for item in combinations if combination_use[item] == minimum_use]
            combination = rng.choice(eligible)
            combination_use[combination] += 1
            chosen = [rng.choice(options[index]) for index in combination]
        specs = tuple(item[0] for item in chosen)
        masks = tuple(int(item[1]) for item in chosen)
        scenario_signature = _signature(specs)
        if scenario_signature in signatures:
            rejected["duplicate"] += 1
            continue
        if behavior != "normal" and v9._full_mask(masks) == 0:
            rejected["no_astar_demo_conflict"] += 1
            continue
        scenario = DynamicScenario(seed=0, obstacles=specs)
        classification = v9._classify_candidate(
            problem,
            scenario,
            behavior,
            design,
        )
        if classification is None:
            rejected["behavior_rule"] += 1
            continue
        if behavior == "normal":
            record = _normal_record(
                scenario,
                masks,
                classification,
                len(demo_paths),
                int(design["observation_radius"]),
            )
            accepted.append(record)
            _save_candidate(record, design, split, behavior, len(accepted))
            diagnostics["accepted"] = len(accepted)
            last_accept_attempt = attempts
            signatures.add(scenario_signature)
            continue
        primary_candidates = [proposed_primary] if constructive else list(range(len(specs)))
        rng.shuffle(primary_candidates)
        found = None
        original_scenario = scenario
        original_masks = masks
        primary_passed = False
        visible_passed = False
        spatial_failed = False
        for primary_index in primary_candidates:
            # Each trial uses the original index-to-route mapping.  A failed
            # visibility or pairing check must not leave reordered obstacles
            # behind for the next primary candidate.
            scenario = original_scenario
            masks = original_masks
            primary_counts["tested"] += 1
            if bool(design["require_primary_demo_conflict"]) and not int(
                masks[primary_index]
            ).bit_count():
                primary_counts["no_demo_conflict"] += 1
                continue
            reduced = _classify_without_obstacle(
                problem,
                scenario,
                primary_index,
                design,
            )
            if reduced is None:
                primary_counts["removal_not_normal"] += 1
                continue
            primary_passed = True
            primary_counts["removal_normal"] += 1
            primary_options = options[combination[primary_index]]
            configured_primary_index = int(design["primary_obstacle_index"])
            if primary_index != configured_primary_index:
                reordered_specs = list(scenario.obstacles)
                reordered_masks = list(masks)
                reordered_specs[primary_index], reordered_specs[configured_primary_index] = (
                    reordered_specs[configured_primary_index],
                    reordered_specs[primary_index],
                )
                reordered_masks[primary_index], reordered_masks[configured_primary_index] = (
                    reordered_masks[configured_primary_index],
                    reordered_masks[primary_index],
                )
                scenario = DynamicScenario(seed=0, obstacles=tuple(reordered_specs))
                masks = tuple(reordered_masks)
                primary_index = configured_primary_index
            observability = v9._decision_observability(
                problem,
                scenario,
                classification["best_plan"],
                behavior,
                int(design["observation_radius"]),
                (primary_index,),
            )
            legacy_observable = observability is not None
            visible_evidence = None
            spatial_evidence = None
            separated_avoidance = behavior == "avoidance" and bool(design.get("_visible_avoidance_pilot", False))
            if separated_avoidance:
                if not legacy_observable:
                    primary_counts["legacy_not_visible"] += 1
                    _save_visibility_failure(problem, scenario, classification, primary_index,
                                             design, split, behavior, diagnostics)
                visible_evidence = _visible_avoidance_evidence(
                    problem, scenario, classification, primary_index, design,
                )
                if visible_evidence is None:
                    primary_counts["no_visible_avoidance"] += 1
                    continue
                primary_counts["visible_avoidance_feasible"] += 1
                if design.get("_spatial_avoidance_pilot", False):
                    spatial_search_audit = {}
                    spatial_evidence, spatial_status = _spatial_avoidance_evidence(
                        problem, scenario, classification, primary_index, design,
                        search_audit=spatial_search_audit,
                    )
                    primary_counts[f"spatial_{spatial_status}"] += 1
                    if spatial_evidence is None:
                        _save_spatial_failure(problem, scenario, primary_index, design,
                                              diagnostics, spatial_search_audit)
                        spatial_failed = True
                        continue
                # Keep metadata tied to the ORIGINAL oracle_path. The witness
                # and its different decision/steps have their own namespace.
                if observability is None:
                    observability = _raw_plan_observation(
                        problem, scenario, classification["best_plan"], behavior, primary_index,
                    )
            if observability is None:
                primary_counts["not_visible"] += 1
                _save_visibility_failure(
                    problem, scenario, classification, primary_index,
                    design, split, behavior, diagnostics,
                )
                continue
            visible_passed = True
            if legacy_observable:
                primary_counts["visible"] += 1
            relevance = _causal_relevance(
                masks,
                classification,
                reduced,
                primary_index,
            )
            record = _record_with_causal_metadata(
                scenario,
                masks,
                classification,
                observability,
                relevance,
                len(demo_paths),
                int(design["observation_radius"]),
                primary_index=primary_index,
                reduced_classification=reduced,
            )
            if separated_avoidance:
                record.update(acceptance_rule="visible_avoidance_feasibility_v1",
                              full_information_behavior=behavior,
                              legacy_oracle_observable=legacy_observable,
                              visible_avoidance=visible_evidence)
                if spatial_evidence is not None:
                    record["spatial_avoidance"] = spatial_evidence
                    record["spatial_acceptance_rule"] = "cycle_free_conflict_cell_bypass_v1"
            if require_pair:
                # Retain a few fully verified conflict cases even if no normal
                # phase control is found, rather than losing all useful work.
                retained = int(diagnostics.get("retained_unpaired_examples", 0))
                if retained < 5:
                    _save_candidate(record, design, split, f"{behavior}_unpaired_verified", retained + 1)
                    diagnostics["retained_unpaired_examples"] = retained + 1
                match = _matched_control(
                    problem,
                    scenario,
                    masks,
                    {
                        **classification,
                        "counterfactual_without_primary_steps": reduced[
                            "best_plan"
                        ].steps,
                    },
                    primary_index,
                    primary_options,
                    len(demo_paths),
                    design,
                    signatures | {scenario_signature},
                    diagnostics=diagnostics,
                )
                if match is None:
                    primary_counts["no_phase_control"] += 1
                    example("visible_primary_without_phase_control", scenario)
                    continue
                control, control_signature = match
                record["_matched_control"] = control
                record["_matched_control_signature"] = control_signature
            found = record
            break
        if found is None:
            reason = (
                "no_causal_primary" if not primary_passed else
                "spatial_witness_not_found" if not visible_passed and spatial_failed else
                "no_visible_avoidance" if (not visible_passed and behavior == "avoidance"
                                           and design.get("_visible_avoidance_pilot", False)) else
                "primary_not_visible" if not visible_passed else
                "no_phase_matched_control"
            )
            rejected[reason] += 1
            example(reason, original_scenario)
            continue
        accepted.append(found)
        diagnostics["accepted_primary_zones"][found.get("primary_critical_zone", "unknown")] += 1
        _save_candidate(found, design, split, behavior, len(accepted))
        diagnostics["accepted"] = len(accepted)
        last_accept_attempt = attempts
        signatures.add(scenario_signature)
        if require_pair:
            signatures.add(found["_matched_control_signature"])
        if len(accepted) == 1 or len(accepted) % 10 == 0:
            print(
                f"[{split} {behavior}] pool={len(accepted)}/{target_size} "
                f"attempts={attempts} rejected={dict(rejected)}",
                flush=True,
            )
    return accepted


def _select_normal(pool, count: int) -> list[dict[str, Any]]:
    selected = sorted(
        pool,
        key=lambda item: (
            int(item["reference_demo_collision_count"]),
            float(item["difficulty_score"]),
        ),
    )[:count]
    for record in selected:
        record["difficulty_stratum"] = "control"
    return selected


def _difficulty_bins(pool: Sequence[dict[str, Any]]):
    ranked = sorted(pool, key=lambda item: float(item["difficulty_score"]))
    first = len(ranked) // 3
    second = 2 * len(ranked) // 3
    return {
        "easy": ranked[:first],
        "medium": ranked[first:second],
        "hard": ranked[second:],
    }


def _select_causal_pools(
    pools: Mapping[str, Sequence[dict[str, Any]]],
    counts: Mapping[str, int],
    fractions: Mapping[str, float],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bins = {
        (behavior, difficulty): group
        for behavior, pool in pools.items()
        for difficulty, group in _difficulty_bins(pool).items()
    }
    remaining = {
        (behavior, difficulty): count
        for behavior in CAUSAL_BEHAVIORS
        for difficulty, count in v9._largest_remainder_targets(
            int(counts[behavior]), fractions
        ).items()
    }
    for key, count in remaining.items():
        if len(bins[key]) < count:
            raise RuntimeError(
                f"Difficulty pool {key} has {len(bins[key])}, needs {count}."
            )
    options_by_zone = {
        zone: [
            (key, record)
            for key, group in bins.items()
            for record in group
            if record["primary_critical_zone"] == zone
        ]
        for zone in ZONE_COLORS
    }
    eligible_zones = [zone for zone in ZONE_COLORS if options_by_zone[zone]]
    zone_order = sorted(
        eligible_zones,
        key=lambda zone: len(options_by_zone[zone]),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def choose_zone(position: int) -> bool:
        if position == len(zone_order):
            return True
        zone = zone_order[position]
        choices = list(options_by_zone[zone])
        rng.shuffle(choices)
        choices.sort(key=lambda item: remaining[item[0]], reverse=True)
        for key, record in choices:
            record_id = id(record)
            if remaining[key] <= 0 or record_id in selected_ids:
                continue
            remaining[key] -= 1
            selected.append(record)
            selected_ids.add(record_id)
            if choose_zone(position + 1):
                return True
            selected_ids.remove(record_id)
            selected.pop()
            remaining[key] += 1
        return False

    # Geographic coverage is a selection preference, not a behavior-validity
    # condition.  Backtracking restores quotas if coverage is infeasible; fill
    # the prescribed behavior/difficulty quotas and report any omissions.
    choose_zone(0)
    for key, count in remaining.items():
        available = [record for record in bins[key] if id(record) not in selected_ids]
        if len(available) < count:
            raise RuntimeError(f"Not enough unused candidates remain in {key}.")
        chosen = rng.sample(available, count)
        selected.extend(chosen)
        selected_ids.update(id(record) for record in chosen)
    for (behavior, difficulty), group in bins.items():
        group_ids = {id(record) for record in group}
        for record in selected:
            if id(record) in group_ids:
                record["difficulty_stratum"] = difficulty
    candidate_zone_counts = Counter(
        record["primary_critical_zone"]
        for pool in pools.values()
        for record in pool
    )
    selected_zone_counts = Counter(
        record["primary_critical_zone"] for record in selected
    )
    selection_audit = {
        "candidate_primary_zone_counts": {
            zone: int(candidate_zone_counts[zone]) for zone in ZONE_COLORS
        },
        "observed_candidate_primary_zones": eligible_zones,
        "zones_without_accepted_candidates": [
            zone for zone in ZONE_COLORS if zone not in eligible_zones
        ],
        "selected_primary_zone_counts": {
            zone: int(selected_zone_counts[zone]) for zone in ZONE_COLORS
        },
        "unselected_candidate_zones": [
            zone for zone in eligible_zones if not selected_zone_counts[zone]
        ],
        "limited_primary_zone_coverage": len(selected_zone_counts) < 2,
        "interpretation": (
            "finite accepted-candidate pool, not an exhaustive zone feasibility proof; "
            "review zone concentration and split differences before training"
        ),
    }
    print(
        f"[primary-zone audit] candidates={dict(candidate_zone_counts)} "
        f"selected={dict(selected_zone_counts)} "
        f"limited_coverage={selection_audit['limited_primary_zone_coverage']}",
        flush=True,
    )
    return selected, selection_audit


def _sort_records(records):
    behavior_order = {name: index for index, name in enumerate(BEHAVIORS)}
    difficulty_order = {"control": 0, "easy": 1, "medium": 2, "hard": 3}
    return sorted(
        records,
        key=lambda item: (
            behavior_order[item["required_behavior"]],
            difficulty_order[item["difficulty_stratum"]],
            float(item["difficulty_score"]),
        ),
    )


def _build_train_or_validation(
    problem,
    config,
    split: str,
    records,
    demo_paths,
    design,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    search_seed = int(config["spatial_generalization"]["behavior_search_seed"])
    rng = random.Random(search_seed + 10_000 * (SPLITS.index(split) + 1))
    signatures: set[tuple] = set()
    normal_count = int(design["behavior_counts"][split]["normal"])
    normal_multiplier = int(design["normal_candidate_pool_multiplier"])
    normal_pool = _build_pool(
        problem,
        split,
        "normal",
        max(normal_count * normal_multiplier, normal_count + 6),
        records,
        demo_paths,
        design,
        rng,
        signatures,
        require_pair=False,
    )
    selected = _select_normal(normal_pool, normal_count)
    causal_pools = {}
    for behavior in CAUSAL_BEHAVIORS:
        count = int(design["behavior_counts"][split][behavior])
        multiplier = int(design["candidate_pool_multiplier"])
        pool_target = max(count * multiplier, count + 6)
        pool = _build_pool(
            problem,
            split,
            behavior,
            pool_target,
            records,
            demo_paths,
            design,
            rng,
            signatures,
            require_pair=False,
        )
        causal_pools[behavior] = pool
    causal_selected, selection_audit = _select_causal_pools(
        causal_pools,
        design["behavior_counts"][split],
        design["difficulty_fractions"],
        rng,
    )
    selected.extend(causal_selected)
    selected = _sort_records(selected)
    for index, record in enumerate(selected):
        record["scenario_id"] = SCENARIO_OFFSETS[split] + index
    return selected, selection_audit


def _build_test_pairs(problem, config, records, demo_paths, design):
    split = "test"
    search_seed = int(config["spatial_generalization"]["behavior_search_seed"])
    rng = random.Random(search_seed + 10_000 * (SPLITS.index(split) + 1))
    signatures: set[tuple] = set()
    pools = {}
    multiplier = int(design["candidate_pool_multiplier"])
    for behavior in CAUSAL_BEHAVIORS:
        count = int(design["behavior_counts"][split][behavior])
        pool_target = max(count * multiplier, count + 6)
        pool = _build_pool(
            problem,
            split,
            behavior,
            pool_target,
            records,
            demo_paths,
            design,
            rng,
            signatures,
            require_pair=True,
        )
        pools[behavior] = pool
    selected_conflicts, selection_audit = _select_causal_pools(
        pools,
        design["behavior_counts"][split],
        design["difficulty_fractions"],
        rng,
    )
    selected_conflicts = _sort_records(selected_conflicts)
    scenarios = []
    behavior_pair_indices = Counter()
    for pair_index, conflict in enumerate(selected_conflicts):
        control = conflict.pop("_matched_control")
        conflict.pop("_matched_control_signature")
        behavior = conflict["required_behavior"]
        difficulty = conflict["difficulty_stratum"]
        behavior_pair_index = behavior_pair_indices[behavior]
        behavior_pair_indices[behavior] += 1
        pair_id = f"test_{behavior}_{behavior_pair_index:02d}"
        conflict_id = SCENARIO_OFFSETS[split] + 2 * pair_index
        control_id = conflict_id + 1
        phase_delta = int(control["primary_start_index"]) - int(
            conflict["primary_start_index"]
        )
        conflict.update(
            {
                "scenario_id": conflict_id,
                "pair_id": pair_id,
                "pair_role": "conflict",
                "matched_behavior": behavior,
                "matched_conflict_difficulty": difficulty,
                "paired_scenario_id": control_id,
                "pair_phase_delta": phase_delta,
            }
        )
        control.update(
            {
                "scenario_id": control_id,
                "difficulty_stratum": "control",
                "pair_id": pair_id,
                "pair_role": "matched_control",
                "matched_behavior": behavior,
                "matched_conflict_difficulty": difficulty,
                "paired_scenario_id": conflict_id,
                "pair_phase_delta": -phase_delta,
            }
        )
        scenarios.extend((conflict, control))
    expected_controls = int(design["behavior_counts"][split]["normal"])
    if expected_controls != len(selected_conflicts):
        raise ValueError(
            "Test normal count must equal the total causal-behavior count so every "
            "conflict receives exactly one matched control."
        )
    return scenarios, selection_audit


def _scenario_from_record(source, lookup) -> DynamicScenario:
    specs = []
    for obstacle in source["obstacles"]:
        route_record = lookup[str(obstacle["route_id"])]
        specs.append(
            DynamicObstacleSpec(
                route=tuple(tuple(cell) for cell in route_record["route"]),
                start_index=int(obstacle["start_index"]),
                direction=int(obstacle["direction"]),
                move_every=int(obstacle["move_every"]),
                label=str(obstacle["route_id"]),
                reference_path_source=str(route_record["critical_zone"]),
            )
        )
    return DynamicScenario(seed=int(source["scenario_id"]), obstacles=tuple(specs))


def _validate_pair(conflict, control, design) -> None:
    if len(conflict["obstacles"]) != 5 or len(control["obstacles"]) != 5:
        raise ValueError("A phase pair must contain five obstacles in each scene.")
    if conflict["paired_scenario_id"] != control["scenario_id"] or control[
        "paired_scenario_id"
    ] != conflict["scenario_id"]:
        raise ValueError(f"Pair {conflict['pair_id']} has inconsistent scenario IDs.")
    primary_index = int(conflict["primary_obstacle_index"])
    if primary_index != int(control["primary_obstacle_index"]):
        raise ValueError(f"Pair {conflict['pair_id']} changes the primary index.")
    changed_start_indices = []
    for index, (left, right) in enumerate(
        zip(conflict["obstacles"], control["obstacles"])
    ):
        for key in ("route_id", "direction", "move_every"):
            if left[key] != right[key]:
                raise ValueError(
                    f"Pair {conflict['pair_id']} changes {key}, not only phase."
                )
        if left["start_index"] != right["start_index"]:
            changed_start_indices.append(index)
    if changed_start_indices != [primary_index]:
        raise ValueError(
            f"Pair {conflict['pair_id']} must change only the primary start_index."
        )
    if bool(design["require_strict_demo_conflict_reduction_in_pair"]) and (
        int(control["primary_reference_demo_collision_count"])
        >= int(conflict["primary_reference_demo_collision_count"])
    ):
        raise ValueError(f"Pair {conflict['pair_id']} lacks strict demo-conflict reduction.")


def _validate_semantics(problem, manifest, design) -> None:
    expected_counts = design["behavior_counts"]
    radius = int(design["observation_radius"])
    all_signatures = set()
    for split in SPLITS:
        sources = manifest["scenarios"][split]
        actual = Counter(source["required_behavior"] for source in sources)
        if actual != Counter(expected_counts[split]):
            raise ValueError(f"{split} behavior distribution does not match its design.")
        lookup = {
            record["route_id"]: record
            for record in manifest["route_pools"][split]["corridor"]
        }
        route_use = Counter()
        for source in sources:
            route_use.update(item["route_id"] for item in source["obstacles"])
            scenario = _scenario_from_record(source, lookup)
            scenario_signature = _signature(scenario.obstacles)
            if scenario_signature in all_signatures:
                raise ValueError(f"Duplicate dynamic scenario in {split}.")
            all_signatures.add(scenario_signature)
            behavior = source["required_behavior"]
            verified = v9._classify_candidate(problem, scenario, behavior, design)
            if verified is None:
                raise ValueError(
                    f"Scenario {source['scenario_id']} no longer satisfies {behavior}."
                )
            if behavior == "normal":
                if (
                    source["difficulty_stratum"] != "control"
                    or source["causal_behavior_verified"]
                ):
                    raise ValueError(f"Scenario {source['scenario_id']} is not a valid control.")
                continue
            primary_index = int(source["primary_obstacle_index"])
            if primary_index != int(design["primary_obstacle_index"]):
                raise ValueError("Primary obstacle is not at the configured curriculum index.")
            primary_spec = scenario.obstacles[primary_index]
            if (
                source["primary_route_id"] != primary_spec.label
                or source["primary_critical_zone"] != primary_spec.reference_path_source
            ):
                raise ValueError("Primary obstacle metadata does not match its actual route.")
            reduced = _classify_without_obstacle(
                problem,
                scenario,
                primary_index,
                design,
            )
            if reduced is None:
                raise ValueError(
                    f"Scenario {source['scenario_id']} has no causal primary obstacle."
                )
            observable = v9._decision_observability(
                problem,
                scenario,
                verified["best_plan"],
                behavior,
                radius,
                (primary_index,),
            )
            if observable is None:
                raise ValueError(
                    f"Scenario {source['scenario_id']} hides its causal obstacle."
                )
            if int(source["primary_reference_demo_collision_count"]) <= 0:
                raise ValueError(
                    f"Scenario {source['scenario_id']} primary does not affect an A* demo."
                )
        expected_routes = set(lookup)
        if set(route_use) != expected_routes:
            missing = sorted(expected_routes - set(route_use))
            raise ValueError(f"{split} leaves route variants unused: {missing}.")
        causal_sources = [
            source
            for source in sources
            if source["required_behavior"] in CAUSAL_BEHAVIORS
        ]
        primary_zones = {
            source["primary_critical_zone"] for source in causal_sources
        }
        zone_audit = manifest["generation"]["causal_primary_zone_search"][split]
        observed_zones = set(zone_audit["observed_candidate_primary_zones"])
        actual_zone_counts = Counter(
            source["primary_critical_zone"] for source in causal_sources
        )
        if not primary_zones.issubset(observed_zones) or any(
            int(zone_audit["selected_primary_zone_counts"][zone])
            != actual_zone_counts[zone]
            for zone in ZONE_COLORS
        ):
            raise ValueError(f"{split} primary-zone audit does not match selected scenarios.")
        used_speeds = {
            int(obstacle["move_every"])
            for source in sources
            for obstacle in source["obstacles"]
        }
        if used_speeds != set(design["move_every_choices"]):
            raise ValueError(
                f"{split} does not use every configured obstacle speed."
            )
        for behavior in CAUSAL_BEHAVIORS:
            group = [
                source
                for source in sources
                if source["required_behavior"] == behavior
            ]
            expected_difficulties = v9._largest_remainder_targets(
                len(group), design["difficulty_fractions"]
            )
            actual_difficulties = Counter(
                source["difficulty_stratum"] for source in group
            )
            if actual_difficulties != Counter(expected_difficulties):
                raise ValueError(
                    f"{split} {behavior} difficulty distribution is unbalanced."
                )
    test_pairs = defaultdict(list)
    for source in manifest["scenarios"]["test"]:
        if not source.get("pair_id"):
            raise ValueError("Every v10 test scenario must belong to a matched pair.")
        test_pairs[source["pair_id"]].append(source)
    expected_pair_count = sum(expected_counts["test"][name] for name in CAUSAL_BEHAVIORS)
    if len(test_pairs) != expected_pair_count:
        raise ValueError("The test set has an unexpected number of pairs.")
    for pair_id, pair in test_pairs.items():
        if len(pair) != 2:
            raise ValueError(f"Pair {pair_id} does not contain exactly two scenarios.")
        conflict = next((item for item in pair if item["pair_role"] == "conflict"), None)
        control = next(
            (item for item in pair if item["pair_role"] == "matched_control"),
            None,
        )
        if conflict is None or control is None:
            raise ValueError(f"Pair {pair_id} lacks a conflict or matched control.")
        if control["required_behavior"] != "normal" or conflict[
            "required_behavior"
        ] != conflict["matched_behavior"]:
            raise ValueError(f"Pair {pair_id} has inconsistent behavior labels.")
        _validate_pair(conflict, control, design)


def _prepare_generation(problem, config):
    design = v9._scenario_design(config)
    move_every_choices = tuple(
        int(value) for value in design.get("move_every_choices", ())
    )
    if not move_every_choices or any(value <= 0 for value in move_every_choices):
        raise ValueError("move_every_choices must contain positive integers.")
    if len(set(move_every_choices)) != len(move_every_choices):
        raise ValueError("move_every_choices cannot contain duplicates.")
    design["move_every_choices"] = move_every_choices
    primary_obstacle_index = int(design.get("primary_obstacle_index", -1))
    if not 0 <= primary_obstacle_index < 5:
        raise ValueError("primary_obstacle_index must be between 0 and 4.")
    design["primary_obstacle_index"] = primary_obstacle_index
    if not math.isclose(float(config["reward"]["stay"]), 0.0, abs_tol=1e-12):
        raise ValueError("The v10 wait benchmark requires reward.stay=0.")
    demo_count = int(config["demonstrations"]["episodes"])
    demo_seed = int(config["demonstrations"]["seed"])
    demo_paths = v9._reference_demo_paths(problem, demo_count, demo_seed)
    route_pools = v9._build_route_pools(problem, config, demo_paths)
    v9._install_behavior_anchors(problem, config, route_pools, demo_paths)
    return design, demo_paths, route_pools


def build_manifest(problem, config, *, search_audit_dir: Path | None = None):
    design, demo_paths, route_pools = _prepare_generation(problem, config)
    design["_search_audit_dir"] = search_audit_dir
    demo_count = int(config["demonstrations"]["episodes"])
    demo_seed = int(config["demonstrations"]["seed"])
    primary_obstacle_index = int(design["primary_obstacle_index"])
    # The paired test is the restrictive part of this protocol.  Check it
    # first; each split has its own RNG, so this does not change its samples.
    test_scenarios, test_zone_audit = _build_test_pairs(
        problem,
        config,
        route_pools["test"]["corridor"],
        demo_paths,
        design,
    )
    train_scenarios, train_zone_audit = _build_train_or_validation(
        problem,
        config,
        "train",
        route_pools["train"]["corridor"],
        demo_paths,
        design,
    )
    validation_scenarios, validation_zone_audit = _build_train_or_validation(
        problem,
        config,
        "validation",
        route_pools["validation"]["corridor"],
        demo_paths,
        design,
    )
    scenarios = {
        "train": train_scenarios,
        "validation": validation_scenarios,
        "test": test_scenarios,
    }
    zone_audits = {
        "train": train_zone_audit,
        "validation": validation_zone_audit,
        "test": test_zone_audit,
    }
    spatial = config["spatial_generalization"]
    manifest = {
        "format_version": 1,
        "map_id": problem.map_id,
        "map_seed": problem.seed,
        "grid_sha256": problem.grid_sha256,
        "generation": {
            "protocol": "office_behavior_paired_v10",
            "seed": int(spatial["generation_seed"]),
            "behavior_search_seed": int(spatial["behavior_search_seed"]),
            "route_length": int(spatial["route_length"]),
            "reference_path_count": int(spatial["reference_path_count"]),
            "reference_demo_episodes": demo_count,
            "reference_demo_seed": demo_seed,
            "separation_radius": int(spatial["separation_radius"]),
            "move_every": int(spatial["move_every"]),
            "move_every_choices": list(design["move_every_choices"]),
            "corridor_per_scenario": 5,
            "background_per_scenario": 0,
            "cross_split_spatial_disjoint": False,
            "split_scope": (
                "held-out route variants and phases on one structured Office topology"
            ),
            "behavior_counts": design["behavior_counts"],
            "difficulty_fractions": design["difficulty_fractions"],
            "causal_primary_zone_search": zone_audits,
            "behavior_definitions": {
                "normal": (
                    "the registered nominal path remains collision-free and needs "
                    "neither waiting nor a detour"
                ),
                "wait": (
                    "the shortest nominal-gate safe plan explicitly waits and is "
                    "strictly shorter than every no-wait plan"
                ),
                "avoidance": (
                    "a no-wait local deviation preserves the nominal gate pair"
                ),
                "reroute": (
                    "a no-wait alternative gate pair is strictly shorter"
                ),
            },
            "causal_primary_rule": (
                "every behavioral scenario declares one verified primary obstacle; "
                "removing it turns the unchanged four-obstacle case into normal; "
                f"it is stored at curriculum index {primary_obstacle_index}"
            ),
            "paired_test_rule": (
                "each conflict has a matched normal control with identical routes, "
                "directions, speeds, and distractors; only primary start phase changes"
            ),
            "observability_rule": (
                "the primary obstacle is inside the local observation window when "
                "the oracle first waits or departs from the nominal path"
            ),
            "placement_rule": (
                "one moving obstacle in each of five topology-critical zones; one is "
                "causal and the other four are fixed covariates in paired comparisons; "
                "primary-zone coverage is limited to zones supported by the oracle"
            ),
            "difficulty_rule": (
                "behavior-specific score quantiles based on demo conflicts, detour, "
                "deviation, and reaction distance"
            ),
        },
        "route_pools": route_pools,
        "scenarios": scenarios,
    }
    _validate_semantics(problem, manifest, design)
    validate_spatial_scenario_manifest(problem, manifest)
    return manifest


def _draw_pair_scenario(axis, problem, source, lookup) -> None:
    role = "conflict" if source["pair_role"] == "conflict" else "control"
    title = (
        f"{source['matched_behavior'].upper()} {source['matched_conflict_difficulty']} "
        f"{role}\nID {source['scenario_id']} | demo "
        f"{source['reference_demo_collision_count']}/20"
    )
    witness = source.get("spatial_avoidance", source.get("visible_avoidance"))
    if witness is not None:
        title += f"\nfull-info={source['minimum_safe_path_steps']} visible={witness['steps']} (+{witness['extra_steps']})"
    v9._draw_base(axis, problem, title)
    oracle_path = source["oracle_path"]
    path_color = "#c62828" if role == "conflict" else "#1565c0"
    axis.plot(
        [cell[1] for cell in oracle_path],
        [cell[0] for cell in oracle_path],
        color=path_color,
        linewidth=1.35,
        linestyle="--",
        alpha=0.9,
        label="Full-information optimum",
    )
    if witness is not None:
        axis.plot([cell[1] for cell in witness["path"]], [cell[0] for cell in witness["path"]],
                  color="#168543", linewidth=1.6, linestyle=":",
                  label="Spatial bypass" if "spatial_avoidance" in source else "Visible-departure path")
        decision = witness["observation"]["decision_position"]
        axis.scatter(decision[1], decision[0], color="#168543", marker="s", s=20, zorder=8)
        axis.legend(loc="upper right", fontsize=6)
    for obstacle in source["obstacles"]:
        route_record = lookup[obstacle["route_id"]]
        route = route_record["route"]
        is_primary = obstacle["route_id"] == source["primary_route_id"]
        color = path_color if is_primary else ZONE_COLORS[route_record["critical_zone"]]
        axis.plot(
            [cell[1] for cell in route],
            [cell[0] for cell in route],
            color=color,
            linewidth=2.8 if is_primary else 1.15,
            alpha=1.0 if is_primary else 0.65,
        )
        initial = route[int(obstacle["start_index"])]
        following = v9._next_cell(
            route,
            int(obstacle["start_index"]),
            int(obstacle["direction"]),
        )
        axis.scatter(
            initial[1],
            initial[0],
            color=path_color if is_primary else "#424242",
            s=23 if is_primary else 11,
            zorder=7,
        )
        axis.annotate(
            "",
            xy=(following[1], following[0]),
            xytext=(initial[1], initial[0]),
            arrowprops={
                "arrowstyle": "->",
                "color": path_color if is_primary else "#424242",
                "lw": 1.0 if is_primary else 0.55,
            },
        )


def render_representative_pairs(problem, manifest, output: Path) -> None:
    lookup = {
        record["route_id"]: record
        for record in manifest["route_pools"]["test"]["corridor"]
    }
    pair_lookup = defaultdict(dict)
    for source in manifest["scenarios"]["test"]:
        pair_lookup[source["pair_id"]][source["pair_role"]] = source
    representatives = []
    for behavior in CAUSAL_BEHAVIORS:
        for difficulty in DIFFICULTIES:
            pair = next(
                pair
                for pair in pair_lookup.values()
                if pair["conflict"]["required_behavior"] == behavior
                and pair["conflict"]["difficulty_stratum"] == difficulty
            )
            representatives.append(
                (pair["conflict"], pair["matched_control"])
            )
    fig, axes = plt.subplots(3, 6, figsize=(17.5, 9.2), dpi=160)
    for row, behavior in enumerate(CAUSAL_BEHAVIORS):
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            conflict, control = representatives[row * len(DIFFICULTIES) + difficulty_index]
            _draw_pair_scenario(
                axes[row, difficulty_index * 2],
                problem,
                conflict,
                lookup,
            )
            _draw_pair_scenario(
                axes[row, difficulty_index * 2 + 1],
                problem,
                control,
                lookup,
            )
    fig.suptitle(
        "Office v10 matched test pairs: red conflict vs blue phase control",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _write_audit(manifest, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = []
    for split in SPLITS:
        route_use = Counter(
            {
                record["route_id"]: 0
                for record in manifest["route_pools"][split]["corridor"]
            }
        )
        split_rows = []
        for source in manifest["scenarios"][split]:
            route_use.update(item["route_id"] for item in source["obstacles"])
            row = {
                "split": split,
                "scenario_id": source["scenario_id"],
                "pair_id": source["pair_id"],
                "pair_role": source["pair_role"],
                "paired_scenario_id": source["paired_scenario_id"],
                "required_behavior": source["required_behavior"],
                "matched_behavior": source["matched_behavior"],
                "difficulty_stratum": source["difficulty_stratum"],
                "matched_conflict_difficulty": source[
                    "matched_conflict_difficulty"
                ],
                "difficulty_score": source["difficulty_score"],
                "primary_obstacle_index": source["primary_obstacle_index"],
                "primary_route_id": source["primary_route_id"],
                "primary_critical_zone": source["primary_critical_zone"],
                "primary_start_index": source["primary_start_index"],
                "primary_direction": source["primary_direction"],
                "primary_move_every": source["primary_move_every"],
                "pair_phase_delta": source["pair_phase_delta"],
                "primary_reference_demo_collision_count": source[
                    "primary_reference_demo_collision_count"
                ],
                "reference_demo_collision_count": source[
                    "reference_demo_collision_count"
                ],
                "causal_behavior_verified": source["causal_behavior_verified"],
                "counterfactual_without_primary_behavior": source[
                    "counterfactual_without_primary_behavior"
                ],
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
            }
            rows.append(row)
            split_rows.append(row)
        summaries.append(
            {
                "split": split,
                "scenario_count": len(split_rows),
                "primary_zone_search": manifest["generation"][
                    "causal_primary_zone_search"
                ][split],
                "pair_count": len(
                    {
                        row["pair_id"]
                        for row in split_rows
                        if row["pair_id"] is not None
                    }
                ),
                "behavior_counts": dict(
                    Counter(row["required_behavior"] for row in split_rows)
                ),
                "difficulty_counts": dict(
                    Counter(row["difficulty_stratum"] for row in split_rows)
                ),
                "primary_zone_counts": dict(
                    Counter(
                        row["primary_critical_zone"]
                        for row in split_rows
                        if row["causal_behavior_verified"]
                    )
                ),
                "move_every_counts": dict(
                    Counter(
                        int(obstacle["move_every"])
                        for source in manifest["scenarios"][split]
                        for obstacle in source["obstacles"]
                    )
                ),
                "mean_reference_demo_collisions": sum(
                    row["reference_demo_collision_count"] for row in split_rows
                )
                / len(split_rows),
                "causal_behavior_rate": sum(
                    row["required_behavior"] == "normal"
                    or row["causal_behavior_verified"]
                    for row in split_rows
                )
                / len(split_rows),
                "minimum_route_usage": min(route_use.values()),
                "maximum_route_usage": max(route_use.values()),
            }
        )
    with (output / "scenario_dataset_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = defaultdict(dict)
    for source in manifest["scenarios"]["test"]:
        pairs[source["pair_id"]][source["pair_role"]] = source
    pair_rows = []
    for pair_id, pair in sorted(pairs.items()):
        conflict = pair["conflict"]
        control = pair["matched_control"]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "behavior": conflict["required_behavior"],
                "difficulty": conflict["difficulty_stratum"],
                "conflict_scenario_id": conflict["scenario_id"],
                "control_scenario_id": control["scenario_id"],
                "primary_route_id": conflict["primary_route_id"],
                "direction": conflict["primary_direction"],
                "move_every": conflict["primary_move_every"],
                "conflict_start_index": conflict["primary_start_index"],
                "control_start_index": control["primary_start_index"],
                "phase_delta": conflict["pair_phase_delta"],
                "conflict_primary_demo_collisions": conflict[
                    "primary_reference_demo_collision_count"
                ],
                "control_primary_demo_collisions": control[
                    "primary_reference_demo_collision_count"
                ],
                "conflict_total_demo_collisions": conflict[
                    "reference_demo_collision_count"
                ],
                "control_total_demo_collisions": control[
                    "reference_demo_collision_count"
                ],
                "conflict_oracle_steps": conflict["minimum_safe_path_steps"],
                "control_oracle_steps": control["minimum_safe_path_steps"],
                "conflict_waits": conflict["oracle_wait_count"],
                "control_waits": control["oracle_wait_count"],
            }
        )
    with (output / "paired_test_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    write_json(summaries, output / "split_summary.json")
    return summaries


def diagnose_test_wait(problem, config, output: Path, attempts: int) -> None:
    """Bounded reproduction of the failing pool, without generating other splits."""
    design, demo_paths, route_pools = _prepare_generation(problem, config)
    design["_search_audit_dir"] = output
    design["_diagnostic_max_attempts"] = attempts
    spatial = config["spatial_generalization"]
    search_seed = int(spatial["behavior_search_seed"]) + 30_000
    count = int(design["behavior_counts"]["test"]["wait"])
    target = max(count * int(design["candidate_pool_multiplier"]), count + 6)
    write_json(
        {
            "purpose": "bounded_test_wait_search_not_a_training_manifest",
            "map_id": problem.map_id,
            "grid_sha256": problem.grid_sha256,
            "search_seed": search_seed,
            "attempt_budget": attempts,
            "target_candidates": target,
            "config": config,
            "test_route_pool": route_pools["test"],
            "interpretation": "Failure within this budget does not prove infeasibility.",
        },
        output / "diagnostic_context.json",
    )
    print(f"Diagnostic only: test wait, attempts<={attempts}, output={output}", flush=True)
    try:
        pool = _build_pool(
            problem, "test", "wait", target, route_pools["test"]["corridor"],
            demo_paths, design, random.Random(search_seed), set(), require_pair=True,
        )
    except RuntimeError as error:
        print(f"Diagnostic stopped: {error}", flush=True)
        print(f"Read {output / 'test_wait_search.json'}", flush=True)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print(f"Diagnostic interrupted; audit saved in {output}", flush=True)
        raise SystemExit(130) from None
    write_json(pool, output / "accepted_test_wait_candidates.json")
    print(f"Diagnostic pool complete; review {output} before full generation.", flush=True)


def _verified_pilot_pair(problem, candidate, behavior, pair_id, first_id, design, lookup):
    """Revalidate serialized candidates without mutating the retained evidence."""
    conflict = dict(candidate)
    control = dict(conflict.pop("_matched_control"))
    conflict.pop("_matched_control_signature")
    delta = int(control["primary_start_index"]) - int(conflict["primary_start_index"])
    for record, role, scenario_id, paired_id, phase_delta in (
        (conflict, "conflict", first_id, first_id + 1, delta),
        (control, "matched_control", first_id + 1, first_id, -delta),
    ):
        record.update({
            "scenario_id": scenario_id, "paired_scenario_id": paired_id,
            "pair_id": pair_id, "pair_role": role,
            "matched_behavior": behavior, "matched_conflict_difficulty": "pilot",
            "difficulty_stratum": "pilot" if role == "conflict" else "control",
            "pair_phase_delta": phase_delta,
        })
    _validate_pair(conflict, control, design)
    if conflict["required_behavior"] != behavior or control["required_behavior"] != "normal":
        raise ValueError("Pilot pair has incorrect behavior labels.")
    for source in (conflict, control):
        scenario = _scenario_from_record(source, lookup)
        verified = v9._classify_candidate(problem, scenario, source["required_behavior"], design)
        if verified is None:
            raise ValueError("Serialized pilot failed behavior revalidation.")
        if source is conflict:
            index = int(source["primary_obstacle_index"])
            if _classify_without_obstacle(problem, scenario, index, design) is None:
                raise ValueError("Serialized pilot failed primary-removal revalidation.")
            legacy_observation = v9._decision_observability(
                problem, scenario, verified["best_plan"], behavior,
                int(design["observation_radius"]), (index,),
            )
            if behavior == "avoidance" and design.get("_visible_avoidance_pilot", False):
                if source.get("acceptance_rule") != "visible_avoidance_feasibility_v1":
                    raise ValueError("Visible-avoidance pilot is missing its acceptance version.")
                evidence = _visible_avoidance_evidence(problem, scenario, verified, index, design)
                if evidence is None or source.get("visible_avoidance") != evidence:
                    raise ValueError("Serialized visible-avoidance evidence failed revalidation.")
                if design.get("_spatial_avoidance_pilot", False):
                    if "spatial_avoidance" not in source or source.get("spatial_acceptance_rule") != "cycle_free_conflict_cell_bypass_v1":
                        raise ValueError("Spatial pilot lacks a spatial witness.")
                    _check_spatial_evidence(problem, scenario, index, source["spatial_avoidance"],
                                            int(design["observation_radius"]))
                    if source["spatial_avoidance"]["extra_steps"] != source["spatial_avoidance"]["steps"] - verified["best_plan"].steps:
                        raise ValueError("Spatial extra cost is inconsistent.")
                if (source.get("full_information_behavior") != behavior
                    or source.get("legacy_oracle_observable") != (legacy_observation is not None)
                    or source["minimum_safe_path_steps"] != verified["best_plan"].steps
                    or source["oracle_path"] != [list(cell) for cell in verified["best_plan"].positions]):
                    raise ValueError("Pilot mixed original optimal metrics with visible-path metrics.")
            elif legacy_observation is None:
                raise ValueError("Serialized pilot failed visibility revalidation.")
    return conflict, control


def find_first_wait_pair(problem, config, output: Path, attempts: int) -> None:
    """Find one reviewable pair before committing to a complete dataset."""
    design, demo_paths, route_pools = _prepare_generation(problem, config)
    design.update({
        "_search_audit_dir": output,
        "_diagnostic_max_attempts": attempts,
        "_constructive_phase_pairs": True,
    })
    seed = int(config["spatial_generalization"]["behavior_search_seed"]) + 30_000
    write_json({
        "purpose": "one_pair_development_pilot_not_a_training_dataset",
        "map_id": problem.map_id, "grid_sha256": problem.grid_sha256,
        "config": config, "search_seed": seed, "attempt_budget": attempts,
        "test_route_pool": route_pools["test"],
        "proposal_rule": "all_existing_test_routes; four nominal-safe covariates; phase-qualified primary",
    }, output / "pilot_context.json")
    # Provide the actual geometry even if no full pair can be found.
    preview_manifest = {
        "generation": {"protocol": "office_behavior_paired_v10"},
        "route_pools": route_pools,
    }
    v9.render_route_pools(problem, preview_manifest, output / "route_pools.png")
    print(f"One-pair pilot: output={output}", flush=True)
    try:
        pool = _build_pool(
            problem, "test", "wait", 1, route_pools["test"]["corridor"],
            demo_paths, design, random.Random(seed), set(), require_pair=True,
        )
    except RuntimeError as error:
        print(f"Pilot stopped: {error}\nSaved evidence: {output}", flush=True)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print(f"Pilot interrupted; saved evidence: {output}", flush=True)
        raise SystemExit(130) from None
    lookup = {record["route_id"]: record for record in route_pools["test"]["corridor"]}
    conflict, control = _verified_pilot_pair(
        problem, pool[0], "wait", "pilot_wait_00", 20000, design, lookup,
    )
    write_json({
        "purpose": "one_verified_wait_pair_not_a_training_dataset",
        "map_id": problem.map_id, "grid_sha256": problem.grid_sha256,
        "route_pool": route_pools["test"], "scenarios": [conflict, control],
    }, output / "verified_wait_pair.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), dpi=180)
    for axis, source in zip(axes, (conflict, control)):
        _draw_pair_scenario(axis, problem, source, lookup)
    fig.suptitle("Office pilot: verified wait (left) / phase control (right)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.text(0.5, 0.015, "Thick route: primary obstacle | dashed path: safe oracle | dots: initial positions", ha="center")
    fig.savefig(output / "verified_wait_pair.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Verified pair saved: {output / 'verified_wait_pair.png'}", flush=True)


def _read_saved_pairs(output: Path, behavior: str):
    # Only complete matched candidates from this run; never include the
    # *_unpaired_verified_* diagnostic examples or another run's cache.
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
        (output / "accepted_candidates").glob(f"test_{behavior}_[0-9][0-9][0-9][0-9].json")
    )]


def _write_batch_report(output, problem, route_pool, groups, scenarios, run_status):
    conflicts = [row for row in scenarios if row["pair_role"] == "conflict"]
    zones = list(dict.fromkeys(row["critical_zone"] for row in route_pool["corridor"]))
    by_behavior = {
        behavior: dict(Counter(row["primary_critical_zone"] for row in conflicts
                               if row["matched_behavior"] == behavior))
        for behavior in CAUSAL_BEHAVIORS
    }
    def motion_signature(row):
        return tuple(sorted((spec["route_id"], spec["start_index"], spec["direction"],
                             spec["move_every"]) for spec in row["obstacles"]))
    unique = len({motion_signature(row) for row in conflicts})
    visible_cases = [row for row in conflicts if "visible_avoidance" in row]
    requested_behaviors = list(dict.fromkeys(group["behavior"] for group in groups))
    warnings = [f"{behavior}: fewer than two observed primary zones"
                for behavior in requested_behaviors if len(by_behavior[behavior]) < 2]
    if unique < len(conflicts):
        warnings.append("Repeated conflict scenes across generation seeds; inspect unique count.")
    write_json({
        "purpose": "small_batch_generation_feasibility_not_a_training_dataset",
        "status": run_status, "training_ready": False,
        "map_id": problem.map_id, "grid_sha256": problem.grid_sha256,
        "scope": "test route pool only; generation seeds, not trained policy seeds",
        "target_pairs": sum(row["target"] for row in groups),
        "verified_pairs": len(conflicts), "unique_conflict_scenes": unique,
        "all_groups_complete": all(row["status"] == "complete" for row in groups),
        "requested_behaviors": requested_behaviors,
        "primary_zones_by_behavior": by_behavior, "warnings": warnings,
        "avoidance_acceptance": groups[0].get("avoidance_acceptance", "legacy_optimal_path_visibility"),
        "visible_avoidance_extra_steps": [row["visible_avoidance"]["extra_steps"] for row in visible_cases],
        "visible_avoidance_legacy_visibility_failures": sum(not row["legacy_oracle_observable"] for row in visible_cases),
        "spatial_witness_count": sum("spatial_avoidance" in row for row in conflicts),
        "spatial_extra_steps": [row["spatial_avoidance"]["extra_steps"] for row in conflicts if "spatial_avoidance" in row],
        "efficiency_approval": "pending human review; feasibility alone is not a difficulty qualification",
        "interpretation": "Finite search absence is not impossibility; completion is not dataset diversity approval.",
        "groups": groups,
    }, output / "batch_summary.json")
    write_json({
        "purpose": "verified_development_pairs_not_a_training_manifest",
        "map_id": problem.map_id, "grid_sha256": problem.grid_sha256,
        "route_pool": route_pool, "scenarios": scenarios,
    }, output / "verified_pairs.json")
    fields = ["pair_id", "generation_seed", "required_behavior", "primary_critical_zone",
              "primary_route_id", "minimum_safe_path_steps", "oracle_wait_count",
              "no_wait_safe_path_steps", "wait_advantage_steps",
              "reference_demo_collision_count", "primary_reference_demo_collision_count",
              "control_safe_steps", "control_demo_collisions", "control_primary_demo_collisions",
              "acceptance_rule", "full_information_behavior", "legacy_oracle_observable",
              "visible_avoidance_steps", "visible_avoidance_extra_steps",
              "visible_avoidance_decision_step", "visible_avoidance_distance", "spatial_steps", "spatial_extra_steps"]
    controls = {row["pair_id"]: row for row in scenarios if row["pair_role"] == "matched_control"}
    with (output / "pair_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in conflicts:
            control = controls[row["pair_id"]]
            writer.writerow({**{key: row.get(key) for key in fields},
                             "control_safe_steps": control["minimum_safe_path_steps"],
                             "control_demo_collisions": control["reference_demo_collision_count"],
                             "control_primary_demo_collisions": control["primary_reference_demo_collision_count"],
                             "visible_avoidance_steps": row.get("visible_avoidance", {}).get("steps"),
                             "visible_avoidance_extra_steps": row.get("visible_avoidance", {}).get("extra_steps"),
                             "visible_avoidance_decision_step": row.get("visible_avoidance", {}).get("observation", {}).get("decision_step"),
                             "visible_avoidance_distance": row.get("visible_avoidance", {}).get("observation", {}).get("nearest_dynamic_distance_at_decision"),
                             "spatial_steps": row.get("spatial_avoidance", {}).get("steps"),
                             "spatial_extra_steps": row.get("spatial_avoidance", {}).get("extra_steps")})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    labels = [f"{row['behavior']} / seed {row['seed_offset']}" for row in groups]
    if any(row.get("requested_primary_zone") for row in groups):
        labels = [f"{row['behavior']} / {row['requested_primary_zone']}" for row in groups]
    axes[0].barh(labels, [row["target"] for row in groups], color="#eeeeee", label="Target")
    axes[0].barh(labels, [row.get("verified", 0) for row in groups], color="#237b9f", label="Verified")
    axes[0].invert_yaxis()
    axes[0].set_title("Verified pairs by generation seed")
    axes[0].set_xlabel("Pairs (partial/failed groups are retained)")
    axes[0].legend()
    counts = [[by_behavior[behavior].get(zone, 0) for zone in zones] for behavior in CAUSAL_BEHAVIORS]
    axes[1].imshow(counts, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, max(map(max, counts))))
    axes[1].set_xticks(range(len(zones)), [zone.replace("_", "\n") for zone in zones])
    axes[1].set_yticks(range(len(CAUSAL_BEHAVIORS)), CAUSAL_BEHAVIORS)
    for y, row in enumerate(counts):
        for x, count in enumerate(row):
            axes[1].text(x, y, str(count), ha="center", va="center", color="#e06400")
    axes[1].set_title("Primary zones of verified conflict cases")
    fig.suptitle("Office development pilot: generation feasibility, NOT policy performance")
    fig.tight_layout()
    fig.savefig(output / "batch_summary.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    if visible_cases:
        fig, axis = plt.subplots(figsize=(10, 4), dpi=150)
        x = list(range(len(visible_cases)))
        axis.bar([i - 0.18 for i in x], [row["minimum_safe_path_steps"] for row in visible_cases],
                 width=0.36, label="Full-information optimum", color="#777777")
        axis.bar([i + 0.18 for i in x], [row["visible_avoidance"]["steps"] for row in visible_cases],
                 width=0.36, label="Visible-departure optimum", color="#168543")
        spatial_points = [(i, row["spatial_avoidance"]["steps"]) for i, row in enumerate(visible_cases) if "spatial_avoidance" in row]
        if spatial_points:
            axis.scatter([i for i, _ in spatial_points], [steps for _, steps in spatial_points],
                         marker="D", color="#8e24aa", label="Cycle-free spatial bypass", zorder=5)
        for i, row in enumerate(visible_cases):
            axis.text(i, row["visible_avoidance"]["steps"] + 0.5,
                      f"+{row['visible_avoidance']['extra_steps']}", ha="center")
        axis.set_xticks(x, [row["pair_id"] for row in visible_cases], rotation=25, ha="right")
        axis.set_ylabel("Safe path steps")
        axis.set_title("Avoidance: feasibility and efficiency reported separately (not policy results)")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "visible_avoidance_costs.png", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def run_pair_batch_pilot(problem, config, output, attempts, pairs_per_group, seed_offsets, seconds,
                         *, visible_avoidance=False, prepared=None, primary_zones=None, spatial_avoidance=False,
                         behaviors=CAUSAL_BEHAVIORS):
    """Bounded independent-seed feasibility searches; preserve partial results."""
    design, demo_paths, route_pools = _prepare_generation(problem, config) if prepared is None else prepared
    design["_visible_avoidance_pilot"] = visible_avoidance
    design["_spatial_avoidance_pilot"] = spatial_avoidance
    if spatial_avoidance and not visible_avoidance:
        raise ValueError("Spatial pilot requires visible-avoidance acceptance.")
    if not behaviors or len(set(behaviors)) != len(behaviors) or any(b not in CAUSAL_BEHAVIORS for b in behaviors):
        raise ValueError("Pilot behaviors must be a nonempty unique subset of causal behaviors.")
    route_pool = route_pools["test"]
    lookup = {row["route_id"]: row for row in route_pool["corridor"]}
    base_seed = int(config["spatial_generalization"]["behavior_search_seed"]) + 30_000
    groups = [{"behavior": behavior, "seed_offset": offset,
               "search_seed": base_seed + offset + index * 1_000_000,
               "target": pairs_per_group, "verified": 0, "status": "not_started",
               "requested_primary_zone": None if primary_zones is None else primary_zones[offset],
               "avoidance_acceptance": ("spatial_bypass_feasibility_v1" if spatial_avoidance else "visible_avoidance_feasibility_v1" if visible_avoidance
                                        else "legacy_optimal_path_visibility")}
              for offset in seed_offsets for index, behavior in enumerate(CAUSAL_BEHAVIORS) if behavior in behaviors]
    write_json({
        "purpose": "small_batch_test_route_feasibility_only", "config": config,
        "grid_sha256": problem.grid_sha256, "groups": groups,
        "attempts_per_group": attempts, "seconds_per_group": seconds,
        "time_limit": "cooperative at candidate boundaries; excludes revalidation and rendering",
        "avoidance_acceptance": groups[0]["avoidance_acceptance"],
        "requested_behaviors": list(behaviors),
        "avoidance_primary_route_ids": design.get("_avoidance_primary_route_ids"),
        "visible_avoidance_extra_steps_limit": None,
        "unchanged_rules": "full-information behavior labels; primary removal; primary demo conflict; phase-only normal control; wait/reroute visibility",
        "sampling": "phase-qualified primary and four nominal-safe covariates; no wait anchors",
        "deduplication": "within each seed across behaviors; cross-seed repeats reported, not suppressed",
        "test_route_pool": route_pool,
    }, output / "batch_context.json")
    v9.render_route_pools(problem, {"generation": {"protocol": "office_behavior_paired_v10"},
                                 "route_pools": route_pools}, output / "route_pools.png")
    scenarios = []
    signatures_by_seed = {offset: set() for offset in seed_offsets}
    run_status = "running"
    active_group = None
    print(f"Batch pilot: {len(groups)} groups, {pairs_per_group} pairs/group, output={output}", flush=True)
    try:
        for group_index, group in enumerate(groups):
            active_group = group
            behavior, offset = group["behavior"], group["seed_offset"]
            group_dir = output / f"seed_{offset}" / behavior
            group_design = {**design, "_search_audit_dir": group_dir,
                            "_diagnostic_max_attempts": attempts, "_search_seconds": seconds,
                            "_constructive_phase_pairs": True}
            if primary_zones is not None:
                group_design.update(_factorized_proposals=True, _proposal_primary_zone=primary_zones[offset])
            group["status"] = "running"
            interrupted = False
            try:
                pool = _build_pool(problem, "test", behavior, pairs_per_group,
                                   route_pool["corridor"], demo_paths, group_design,
                                   random.Random(group["search_seed"]), signatures_by_seed[offset],
                                   require_pair=True)
            except RuntimeError as error:
                group.update(status="search_incomplete", error=str(error))
                pool = _read_saved_pairs(group_dir, behavior)
                print(f"Group incomplete; keeping {len(pool)} pairs and continuing: {error}", flush=True)
            except KeyboardInterrupt:
                group["status"] = "interrupted"
                pool = _read_saved_pairs(group_dir, behavior)
                interrupted = True
            audit_path = group_dir / f"test_{behavior}_search.json"
            if audit_path.exists():
                group["search_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
            for number, candidate in enumerate(pool):
                pair_id = f"pilot_s{offset}_{behavior}_{number:02d}"
                conflict, control = _verified_pilot_pair(
                    problem, candidate, behavior, pair_id,
                    20000 + 2 * (group_index * pairs_per_group + number), design, lookup,
                )
                for source in (conflict, control):
                    source["generation_seed"] = group["search_seed"]
                write_json({"scenarios": [conflict, control]}, group_dir / f"{pair_id}.json")
                scenarios.extend((conflict, control))
                group["verified"] += 1
                fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), dpi=150)
                for axis, source in zip(axes, (conflict, control)):
                    _draw_pair_scenario(axis, problem, source, lookup)
                fig.suptitle(f"{pair_id}: conflict (left) / phase control (right)")
                fig.tight_layout(rect=(0, 0.05, 1, 0.94))
                fig.text(0.5, 0.02, "Dashed: full-info oracle | Green (if shown): visible-departure path | Dots: initial obstacle positions", ha="center", fontsize=9)
                fig.savefig(group_dir / f"{pair_id}.png", facecolor="white", bbox_inches="tight")
                plt.close(fig)
            if group["status"] == "running":
                group["status"] = "complete"
            _write_batch_report(output, problem, route_pool, groups, scenarios, run_status)
            if interrupted:
                raise KeyboardInterrupt
        run_status = "complete" if all(g["status"] == "complete" for g in groups) else "incomplete"
    except KeyboardInterrupt:
        run_status = "interrupted"
        if active_group is not None:
            active_group["status"] = "interrupted"
        raise SystemExit(130) from None
    except Exception as error:
        run_status = "error"
        if active_group is not None:
            active_group.update(status="error", error=str(error))
        raise
    finally:
        _write_batch_report(output, problem, route_pool, groups, scenarios, run_status)
        print(f"Batch {run_status}; verified={len(scenarios) // 2}. Review {output / 'batch_summary.png'}", flush=True)
    if run_status != "complete":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--audit-output",
        default="outputs/office_behavior_v10_dataset_design",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnose-test-wait", action="store_true")
    mode.add_argument("--find-first-wait-pair", action="store_true")
    mode.add_argument("--pair-batch-pilot", action="store_true")
    parser.add_argument("--visible-avoidance", action="store_true",
                        help="With --pair-batch-pilot only: separate visible avoidance feasibility from optimal cost.")
    parser.add_argument("--diagnostic-attempts", type=int, default=2000)
    parser.add_argument("--pilot-pairs-per-group", type=int, default=2)
    parser.add_argument("--pilot-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--pilot-seconds-per-group", type=float, default=60)
    args = parser.parse_args()
    if args.visible_avoidance and not args.pair_batch_pilot:
        parser.error("--visible-avoidance is only supported with --pair-batch-pilot")
    if args.diagnostic_attempts <= 0:
        parser.error("--diagnostic-attempts must be positive")
    if args.pilot_pairs_per_group <= 0:
        parser.error("--pilot-pairs-per-group must be positive")
    if len(set(args.pilot_seeds)) != len(args.pilot_seeds) or min(args.pilot_seeds) < 0:
        parser.error("--pilot-seeds must be distinct nonnegative integers")
    if not math.isfinite(args.pilot_seconds_per_group) or args.pilot_seconds_per_group <= 0:
        parser.error("--pilot-seconds-per-group must be finite and positive")
    config = load_config(v9._resolve(args.config))
    problem = v9._load_problem(config)
    audit_root = v9._resolve(args.audit_output)
    run_stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000000:09d}"
    if args.pair_batch_pilot:
        pilot_folder = "visible_avoidance_batch_pilots" if args.visible_avoidance else "batch_pilots"
        run_pair_batch_pilot(
            problem, config, audit_root / pilot_folder / run_stamp,
            args.diagnostic_attempts, args.pilot_pairs_per_group,
            args.pilot_seeds, args.pilot_seconds_per_group,
            visible_avoidance=args.visible_avoidance,
        )
        return
    if args.find_first_wait_pair:
        find_first_wait_pair(
            problem, config, audit_root / "pair_pilots" / run_stamp,
            args.diagnostic_attempts,
        )
        return
    if args.diagnose_test_wait:
        diagnose_test_wait(
            problem, config, audit_root / "diagnostics" / run_stamp,
            args.diagnostic_attempts,
        )
        return
    manifest = build_manifest(
        problem, config, search_audit_dir=audit_root / "search_runs" / run_stamp,
    )
    spatial = config["spatial_generalization"]
    manifest_path = v9._resolve(spatial["manifest"])
    write_json(manifest, manifest_path)
    v9.render_route_pools(
        problem,
        manifest,
        v9._resolve(spatial["route_pool_preview"]),
    )
    v9.render_dataset_summary(
        manifest,
        v9._resolve(spatial["dataset_summary_preview"]),
    )
    for split, key in (
        ("train", "training_preview"),
        ("validation", "validation_preview"),
        ("test", "test_preview"),
    ):
        v9.render_gallery(problem, manifest, split, v9._resolve(spatial[key]))
    render_representative_pairs(
        problem,
        manifest,
        v9._resolve(spatial["paired_test_preview"]),
    )
    summaries = _write_audit(manifest, v9._resolve(args.audit_output))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"manifest={manifest_path} sha256={digest[:12]}")
    for summary in summaries:
        print(
            f"{summary['split']}: scenarios={summary['scenario_count']} "
            f"pairs={summary['pair_count']} "
            f"behaviors={summary['behavior_counts']} "
            f"causal={summary['causal_behavior_rate']:.1%} "
            f"primary_zones={summary['primary_zone_counts']} "
            f"limited_zone_coverage="
            f"{summary['primary_zone_search']['limited_primary_zone_coverage']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
