"""Intrinsic conflict metrics for dynamic-obstacle navigation scenarios."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from math import gcd
from typing import Any

from astar_d3qn.core.grid import Position, manhattan
from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.maps.problem import NavigationProblem


def obstacle_positions(
    spec: DynamicObstacleSpec,
    horizon: int,
) -> tuple[Position, ...]:
    """Return obstacle positions at reset (t=0) and after every environment step."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    index = int(spec.start_index)
    direction = int(spec.direction)
    move_counter = 0
    positions = [spec.route[index]]
    for _ in range(horizon):
        move_counter += 1
        if move_counter >= spec.move_every:
            move_counter = 0
            candidate = index + direction
            if candidate >= len(spec.route) or candidate < 0:
                direction *= -1
                candidate = index + direction
            index = candidate
        positions.append(spec.route[index])
    return tuple(positions)


def _obstacle_period(spec: DynamicObstacleSpec) -> int:
    """Return the period of one deterministic bouncing obstacle."""

    index = int(spec.start_index)
    direction = int(spec.direction)
    move_counter = 0
    seen = {(index, direction, move_counter): 0}
    period = 0
    while True:
        move_counter += 1
        if move_counter >= int(spec.move_every):
            move_counter = 0
            candidate = index + direction
            if candidate >= len(spec.route) or candidate < 0:
                direction *= -1
                candidate = index + direction
            index = candidate
        period += 1
        state = (index, direction, move_counter)
        if state in seen:
            return period - seen[state]
        seen[state] = period
        # The implementation keeps the direction at a boundary until the next
        # attempted move, so the full internal state can take twice the usual
        # back-and-forth position period to repeat.
        if period > 4 * max(1, len(spec.route)) * max(1, int(spec.move_every)) + 4:
            raise RuntimeError("Could not determine dynamic obstacle period.")


def minimum_collision_free_steps(
    problem: NavigationProblem,
    scenario: DynamicScenario,
) -> int | None:
    """Find the shortest collision-free path in the periodic time-expanded grid.

    The dynamic obstacles move before the agent action, and both their old and
    proposed cells are treated as occupied, matching ``DynamicGridNavigationEnv``.
    ``None`` means that the goal is unreachable without collision under the
    deterministic obstacle motion.
    """

    if not scenario.obstacles:
        return len(problem.nominal_path) - 1
    period = 1
    for spec in scenario.obstacles:
        obstacle_period = _obstacle_period(spec)
        period = period * obstacle_period // gcd(period, obstacle_period)
    positions = [obstacle_positions(spec, period) for spec in scenario.obstacles]
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
    queue = deque([(problem.start, 0, 0)])
    visited = {(problem.start, 0)}
    while queue:
        position, time_mod, distance = queue.popleft()
        if position == problem.goal:
            return distance
        next_time = (time_mod + 1) % period
        forbidden = {sequence[time_mod] for sequence in positions}
        forbidden.update(sequence[next_time] for sequence in positions)
        for row_delta, column_delta in moves:
            candidate = (
                position[0] + row_delta,
                position[1] + column_delta,
            )
            state = (candidate, next_time)
            if not (0 <= candidate[0] < problem.size and 0 <= candidate[1] < problem.size):
                continue
            if candidate in problem.obstacles or candidate in forbidden:
                continue
            if state in visited:
                continue
            visited.add(state)
            queue.append((candidate, next_time, distance + 1))
    return None


def obstacle_conflict_metrics(
    problem: NavigationProblem,
    spec: DynamicObstacleSpec,
    *,
    temporal_window: int = 2,
) -> dict[str, Any]:
    """Measure geometry and nominal-arrival timing for one moving obstacle.

    Timing assumes that the agent traverses the registered nominal A* path at one
    cell per step without reacting. This is an intrinsic scenario descriptor, not
    a claim about any learned policy's actual trajectory.
    """

    if temporal_window < 0:
        raise ValueError("temporal_window must be non-negative.")
    path = tuple(problem.nominal_path)
    path_set = set(path)
    route_set = set(spec.route)
    intersection_cells = tuple(sorted(path_set.intersection(route_set)))
    min_route_distance = min(
        manhattan(route_cell, path_cell)
        for route_cell in spec.route
        for path_cell in path
    )
    horizon = max(1, len(path) - 1 + temporal_window)
    positions = obstacle_positions(spec, horizon)

    exact_steps: list[int] = []
    near_steps: list[int] = []
    minimum_phase_offset: int | None = None
    for step in range(1, len(path)):
        candidate = path[step]
        # DynamicGridNavigationEnv moves the obstacle first and treats both its
        # old and proposed cell as occupied for collision purposes.
        if candidate == positions[step - 1] or candidate == positions[step]:
            exact_steps.append(step)
        if candidate not in route_set:
            continue
        offsets = [
            abs(obstacle_time - step)
            for obstacle_time, position in enumerate(positions)
            if position == candidate
        ]
        if not offsets:
            continue
        cell_offset = min(offsets)
        minimum_phase_offset = (
            cell_offset
            if minimum_phase_offset is None
            else min(minimum_phase_offset, cell_offset)
        )
        if cell_offset <= temporal_window:
            near_steps.append(step)

    return {
        "direct_intersection": int(bool(intersection_cells)),
        "direct_intersection_cell_count": len(intersection_cells),
        "intersection_cells": [list(cell) for cell in intersection_cells],
        "min_route_distance": int(min_route_distance),
        "near_nominal": int(min_route_distance <= 1),
        "exact_temporal_conflict_count": len(set(exact_steps)),
        "exact_temporal_conflict_steps": sorted(set(exact_steps)),
        "aligned_temporal_conflict_count": len(set(near_steps)),
        "aligned_temporal_conflict_steps": sorted(set(near_steps)),
        "minimum_phase_offset": minimum_phase_offset,
    }


def scenario_conflict_metrics(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    *,
    temporal_window: int = 2,
    route_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate intrinsic conflict metrics for a complete scenario."""

    obstacle_rows = [
        obstacle_conflict_metrics(problem, spec, temporal_window=temporal_window)
        for spec in scenario.obstacles
    ]
    corridor_frequency_sum = 0.0
    if route_records is not None:
        for spec in scenario.obstacles:
            record = route_records.get(str(spec.label), {})
            corridor_frequency_sum += float(record.get("corridor_frequency", 0.0))
    direct_routes = sum(int(row["direct_intersection"]) for row in obstacle_rows)
    direct_cells = sum(
        int(row["direct_intersection_cell_count"]) for row in obstacle_rows
    )
    exact_conflicts = sum(
        int(row["exact_temporal_conflict_count"]) for row in obstacle_rows
    )
    aligned_conflicts = sum(
        int(row["aligned_temporal_conflict_count"]) for row in obstacle_rows
    )
    near_routes = sum(int(row["near_nominal"]) for row in obstacle_rows)
    min_distance = min(int(row["min_route_distance"]) for row in obstacle_rows)
    phase_offsets = [
        int(row["minimum_phase_offset"])
        for row in obstacle_rows
        if row["minimum_phase_offset"] is not None
    ]
    # The score is used only to order scenarios for stratification. Raw component
    # metrics remain the scientific outputs and must be reported alongside it.
    score = (
        20.0 * exact_conflicts
        + 6.0 * aligned_conflicts
        + 3.0 * direct_routes
        + 1.0 * direct_cells
        + 0.5 * near_routes
        + 0.25 * corridor_frequency_sum
    )
    return {
        "scenario_id": int(scenario.seed),
        "obstacle_count": len(scenario.obstacles),
        "direct_intersection_route_count": direct_routes,
        "direct_intersection_cell_count": direct_cells,
        "near_nominal_route_count": near_routes,
        "minimum_route_distance": min_distance,
        "exact_temporal_conflict_count": exact_conflicts,
        "aligned_temporal_conflict_count": aligned_conflicts,
        "minimum_phase_offset": min(phase_offsets) if phase_offsets else None,
        "corridor_frequency_sum": corridor_frequency_sum,
        "conflict_score": score,
        "obstacle_metrics": obstacle_rows,
    }


def scenario_safe_path_metrics(
    problem: NavigationProblem,
    scenario: DynamicScenario,
) -> dict[str, int | None]:
    """Return oracle safe-path length and detour relative to nominal A*."""

    safe_steps = minimum_collision_free_steps(problem, scenario)
    nominal_steps = len(problem.nominal_path) - 1
    return {
        "nominal_path_steps": nominal_steps,
        "minimum_safe_path_steps": safe_steps,
        "safe_detour_steps": None if safe_steps is None else safe_steps - nominal_steps,
    }


def route_record_lookup(
    manifest: Mapping[str, Any],
    split: str,
) -> dict[str, Mapping[str, Any]]:
    """Return route metadata indexed by route id for one manifest split."""

    pool = manifest["route_pools"][split]
    return {
        str(record["route_id"]): record
        for category in ("corridor", "background")
        for record in pool[category]
    }


def aggregate_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    """Return an arithmetic mean for a numeric scenario metric."""

    if not rows:
        raise ValueError("rows must not be empty.")
    return sum(float(row[key]) for row in rows) / len(rows)
