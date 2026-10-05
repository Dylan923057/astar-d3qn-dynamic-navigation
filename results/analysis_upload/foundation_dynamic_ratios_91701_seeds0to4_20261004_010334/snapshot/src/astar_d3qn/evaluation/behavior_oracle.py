"""Exact time-expanded planning diagnostics for dynamic navigation behavior."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Collection

from astar_d3qn.core.grid import ACTION_DELTAS, Position, chebyshev, in_bounds
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.maps.problem import NavigationProblem

from .conflict import _obstacle_period, obstacle_positions


@dataclass(frozen=True)
class SafePlan:
    """One shortest safe plan, breaking equal-step ties by fewer waits."""

    positions: tuple[Position, ...]
    steps: int
    wait_count: int


def shortest_safe_plan(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    *,
    allow_wait: bool = True,
    additionally_blocked: Collection[Position] = (),
    visible_deviation_obstacle_index: int | None = None,
    observation_radius: int = 7,
    start_position: Position | None = None,
    start_time: int = 0,
) -> SafePlan | None:
    """Return a collision-free plan matching the environment's motion order.

    Dynamic obstacles move before the agent.  Their cells immediately before and
    after that movement are both forbidden, which also prevents edge swaps.
    Search cost is lexicographic: minimum environment steps, then minimum waits.

    Optional diagnostic constraint: follow the nominal path until its first
    deviation, which is allowed only when the specified obstacle is within the
    observation square BEFORE that action. Afterwards normal oracle planning
    resumes. This tests existence across all tied paths, not a learned local
    policy. A plan with no deviation is also permitted. Defaults are unchanged.
    """

    if start_time < 0:
        raise ValueError("start_time must be nonnegative.")
    start = problem.start if start_position is None else tuple(start_position)
    blocked = set(problem.obstacles).union(additionally_blocked)
    constrained = visible_deviation_obstacle_index is not None
    if constrained:
        if start != problem.start or start_time != 0:
            raise ValueError("Visible-deviation planning must start from the episode origin.")
        if not 0 <= visible_deviation_obstacle_index < len(scenario.obstacles):
            raise ValueError("Visible-deviation obstacle index is out of range.")
        if observation_radius < 0:
            raise ValueError("Observation radius must be nonnegative.")
        if not problem.nominal_path or problem.nominal_path[0] != problem.start:
            raise ValueError("Visible-deviation planning requires a nominal path from start.")
    if not in_bounds(start, problem.size):
        raise ValueError("start_position is outside the map.")
    if start in blocked or problem.goal in blocked:
        return None
    period = 1
    for spec in scenario.obstacles:
        obstacle_period = _obstacle_period(spec)
        period = period * obstacle_period // __import__("math").gcd(period, obstacle_period)
    positions = [obstacle_positions(spec, period) for spec in scenario.obstacles]
    moves = ACTION_DELTAS if allow_wait else ACTION_DELTAS[:4]
    # Prefix index, rather than just a Boolean, also handles references that
    # revisit a position at the same obstacle phase without merging states.
    start_state = (start, start_time % period, 0 if constrained else None)
    counter = itertools.count()
    queue: list[tuple[int, int, int, tuple[Position, int, int | None]]] = [
        (0, 0, next(counter), start_state)
    ]
    best = {start_state: (0, 0)}
    came_from: dict[tuple[Position, int, int | None], tuple[Position, int, int | None]] = {}
    while queue:
        steps, waits, _, state = heapq.heappop(queue)
        if (steps, waits) != best.get(state):
            continue
        position, time_mod, prefix_index = state
        if position == problem.goal:
            states = [state]
            while states[-1] in came_from:
                states.append(came_from[states[-1]])
            states.reverse()
            path = tuple(item[0] for item in states)
            return SafePlan(path, steps, waits)
        next_time = (time_mod + 1) % period
        forbidden = {sequence[time_mod] for sequence in positions}
        forbidden.update(sequence[next_time] for sequence in positions)
        for row_delta, column_delta in moves:
            candidate = (position[0] + row_delta, position[1] + column_delta)
            if not in_bounds(candidate, problem.size):
                continue
            if candidate in blocked or candidate in forbidden:
                continue
            next_prefix = prefix_index
            if prefix_index is not None:
                if prefix_index + 1 < len(problem.nominal_path) and (
                    candidate == problem.nominal_path[prefix_index + 1]
                ):
                    next_prefix = prefix_index + 1
                else:
                    primary_cell = positions[visible_deviation_obstacle_index][time_mod]
                    if chebyshev(position, primary_cell) > observation_radius:
                        continue
                    next_prefix = None
            next_state = (candidate, next_time, next_prefix)
            next_cost = (steps + 1, waits + int(candidate == position))
            if next_cost >= best.get(next_state, (10**18, 10**18)):
                continue
            best[next_state] = next_cost
            came_from[next_state] = state
            heapq.heappush(queue, (*next_cost, next(counter), next_state))
    return None
