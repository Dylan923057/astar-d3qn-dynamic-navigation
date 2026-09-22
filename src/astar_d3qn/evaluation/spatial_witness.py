"""Bounded spatial-bypass witnesses, separate from optimal behavior labels."""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Collection

from astar_d3qn.core.grid import ACTION_DELTAS, Position, chebyshev, in_bounds
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.maps.problem import NavigationProblem

from .behavior_oracle import SafePlan
from .conflict import _obstacle_period, obstacle_positions


def static_bypass_audit(problem, conflict_cell, *, additionally_blocked=()):
    """Exact geometric necessary condition, NOT a dynamic feasibility verdict.

    Compare connectivity before/after permanently removing the conflict cell.
    A cut under a fixed gate pair need not be a cut in the whole map.
    """
    blocked = set(problem.obstacles).union(additionally_blocked)

    def distance(removed):
        if problem.start in removed or problem.goal in removed:
            return None
        pending = deque([(problem.start, 0)])
        seen = {problem.start}
        while pending:
            cell, steps = pending.popleft()
            if cell == problem.goal:
                return steps
            for dr, dc in ACTION_DELTAS[:4]:
                nxt = (cell[0] + dr, cell[1] + dc)
                if in_bounds(nxt, problem.size) and nxt not in removed and nxt not in seen:
                    seen.add(nxt)
                    pending.append((nxt, steps + 1))
        return None

    base = distance(blocked)
    bypass = distance(blocked | {conflict_cell})
    status = ("base_disconnected" if base is None else
              "endpoint_excluded" if conflict_cell in (problem.start, problem.goal) else
              "conflict_cell_is_required" if bypass is None else "static_bypass_possible")
    return {"status": status, "conflict_cell": list(conflict_cell),
            "base_steps": base, "bypass_steps": bypass,
            "scope": "static four-connected graph under supplied gate blocks; ignores dynamics and visibility"}


def motion_summary(path):
    """Geometry only: closed walks indicate delay capacity, not policy intent."""
    path = tuple(map(tuple, path))
    waits, reversals, returns = [], [], []
    last_visit = {}
    for t, cell in enumerate(path):
        if t and cell == path[t - 1]:
            waits.append(t)
        elif cell in last_visit:
            returns.append({"departure_time": last_visit[cell], "return_time": t,
                            "cell": list(cell), "closed_walk_steps": t - last_visit[cell]})
        if t >= 2 and cell == path[t - 2] and cell != path[t - 1]:
            reversals.append({"departure_time": t - 2, "return_time": t,
                              "cell": list(cell), "via": list(path[t - 1])})
        last_visit[cell] = t
    return {"steps": max(0, len(path) - 1), "stay_count": len(waits), "stay_steps": waits,
            "immediate_reversal_count": len(reversals), "immediate_reversals": reversals,
            "closed_walk_count": len(returns), "closed_walks": returns,
            "is_simple_path": len(set(path)) == len(path),
            "note": "Counts overlap; a reversal is also a closed walk. Geometry is not a causal behavior label."}


@dataclass(frozen=True)
class SpatialSearchResult:
    plan: SafePlan | None
    status: str
    expanded: int
    generated: int
    elapsed_seconds: float
    stop_reason: str


def shortest_simple_visible_plan(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    *,
    primary_index: int,
    observation_radius: int,
    additionally_blocked: Collection[Position] = (),
    max_expanded: int = 50_000,
    max_generated: int = 150_000,
    max_seconds: float = 15,
    progress=None,
) -> SpatialSearchResult:
    """A* over position, phase, reference prefix and the ENTIRE visited set.

    No STAY or spatial revisits. Before first departure follow the reference;
    departure requires current-frame primary visibility. All path-history
    variants are retained (merging just position/time would be incorrect).
    Found plans are shortest under these constraints. Exhaustion is scoped to
    them; a resource limit returns unknown, never an infeasibility claim.
    """
    if not 0 <= primary_index < len(scenario.obstacles) or observation_radius < 0:
        raise ValueError("Invalid primary index or observation radius.")
    if max_expanded <= 0 or max_generated <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Search budgets must be finite and positive.")
    if not problem.nominal_path or problem.nominal_path[0] != problem.start:
        raise ValueError("A nominal reference starting at the robot is required.")
    started = time.monotonic()
    # Compare against an absolute deadline.  For sub-clock-resolution budgets,
    # ``started + max_seconds`` can round back to ``started``; in that case the
    # first budget check deterministically expires instead of allowing a tiny
    # search to finish before the monotonic clock advances.
    deadline = started + max_seconds
    expanded, generated = 0, 0

    def finish(status, reason, plan=None):
        return SpatialSearchResult(plan, status, expanded, generated,
                                   round(time.monotonic() - started, 3), reason)

    blocked = set(problem.obstacles).union(additionally_blocked)
    if problem.start in blocked or problem.goal in blocked:
        return finish("infeasible_under_constraints", "blocked_endpoint")
    # Static reverse distances are an admissible, consistent lower bound.
    distance = {problem.goal: 0}
    pending = deque([problem.goal])
    while pending:
        if time.monotonic() >= deadline:
            return finish("budget_exhausted", "time_during_static_lower_bound")
        cell = pending.popleft()
        for dr, dc in ACTION_DELTAS[:4]:
            nxt = (cell[0] + dr, cell[1] + dc)
            if in_bounds(nxt, problem.size) and nxt not in blocked and nxt not in distance:
                distance[nxt] = distance[cell] + 1
                pending.append(nxt)
    if problem.start not in distance:
        return finish("infeasible_under_constraints", "static_disconnection")
    period = 1
    for spec in scenario.obstacles:
        period = math.lcm(period, _obstacle_period(spec))
    # Precompute independent small-period trajectories, avoiding a potentially
    # large joint-period array.
    trajectories = [obstacle_positions(spec, _obstacle_period(spec)) for spec in scenario.obstacles]
    periods = [len(cells) - 1 for cells in trajectories]

    def bit(cell):
        return 1 << (cell[0] * problem.size + cell[1])

    start = (problem.start, 0, 0, bit(problem.start))
    serial = itertools.count()
    queue = [(distance[problem.start], 0, next(serial), start)]
    parent = {start: None}
    generated = 1
    last_report = started
    while queue:
        if time.monotonic() >= deadline:
            return finish("budget_exhausted", "time")
        _, steps, _, state = heapq.heappop(queue)
        position, phase, prefix, visited = state
        if position == problem.goal:
            states = [state]
            while parent[states[-1]] is not None:
                states.append(parent[states[-1]])
            path = tuple(item[0] for item in reversed(states))
            return finish("found", "shortest_in_constraint_class", SafePlan(path, steps, 0))
        if expanded >= max_expanded:
            return finish("budget_exhausted", "expanded_states")
        expanded += 1
        if progress and time.monotonic() - last_report >= 5:
            progress(expanded, generated, len(queue))
            last_report = time.monotonic()
        forbidden = set()
        for cells, motion_period in zip(trajectories, periods):
            forbidden.update((cells[phase % motion_period], cells[(phase + 1) % motion_period]))
        for dr, dc in ACTION_DELTAS[:4]:
            nxt = (position[0] + dr, position[1] + dc)
            if nxt not in distance or nxt in forbidden or visited & bit(nxt):
                continue
            next_prefix = prefix
            if prefix is not None:
                if prefix + 1 < len(problem.nominal_path) and nxt == problem.nominal_path[prefix + 1]:
                    next_prefix += 1
                else:
                    primary = trajectories[primary_index][phase % periods[primary_index]]
                    if chebyshev(position, primary) > observation_radius:
                        continue
                    next_prefix = None
            successor = (nxt, (phase + 1) % period, next_prefix, visited | bit(nxt))
            if successor in parent:
                continue
            if generated >= max_generated:
                return finish("budget_exhausted", "generated_states")
            parent[successor] = state
            generated += 1
            heapq.heappush(queue, (steps + 1 + distance[nxt], steps + 1, next(serial), successor))
    return finish("infeasible_under_constraints", "exhaustive_search")
