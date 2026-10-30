"""Multi-obstacle construction and exact audits for risk-handover studies."""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import asdict

from astar_d3qn.core.grid import ACTION_DELTAS, in_bounds, manhattan, move
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec

from .adaptation import collision_step, occupancy, spec_from_record


def replay_multi_obstacle_path(problem, specs, path) -> bool:
    if not path or tuple(path[0]) != problem.start or tuple(path[-1]) != problem.goal:
        return False
    env = DynamicGridNavigationEnv(
        problem,
        list(specs),
        max_steps=len(path) + 1,
        terminate_on_collision=True,
        window_size=15,
    )
    env.reset()
    for left, right in zip(path, path[1:]):
        delta = (right[0] - left[0], right[1] - left[1])
        try:
            action = ACTION_DELTAS.index(delta)
        except ValueError:
            return False
        result = env.step(action)
        if result.info["collision"]:
            return False
    return env.position == problem.goal


def multi_obstacle_safe_oracle(problem, specs, *, max_steps=300):
    """Shortest safe path in position-time space under simulator semantics."""

    specs = tuple(specs)
    initial = (problem.start, 0)
    costs = {initial: 0}
    parents = {}
    counter = itertools.count()
    pending = [(manhattan(problem.start, problem.goal), 0, next(counter), initial)]
    while pending:
        _, cost, _, state = heapq.heappop(pending)
        if costs.get(state) != cost:
            continue
        position, time = state
        if position == problem.goal:
            path = [position]
            while state in parents:
                state = parents[state]
                path.append(state[0])
            return list(reversed(path))
        if time >= max_steps:
            continue
        old_dynamic = {occupancy(spec, time) for spec in specs}
        next_dynamic = {occupancy(spec, time + 1) for spec in specs}
        for action in range(5):
            following = move(position, action)
            if (
                not in_bounds(following, problem.size)
                or following in problem.obstacles
                or following in old_dynamic
                or following in next_dynamic
            ):
                continue
            next_time = time + 1
            if next_time + manhattan(following, problem.goal) > max_steps:
                continue
            next_state = (following, next_time)
            if cost + 1 >= costs.get(next_state, 10**9):
                continue
            costs[next_state] = cost + 1
            parents[next_state] = state
            heapq.heappush(
                pending,
                (
                    cost + 1 + manhattan(following, problem.goal),
                    cost + 1,
                    next(counter),
                    next_state,
                ),
            )
    return None


def _route_cells(pair):
    return frozenset(map(tuple, pair["route"]))


def _compatible_companions(anchor, candidates, count, rng):
    if count == 0:
        return []
    anchor_step = anchor["first_reference_collision_step"]
    ranked = list(candidates)
    rng.shuffle(ranked)
    ranked.sort(
        key=lambda pair: (
            -abs(pair["first_reference_collision_step"] - anchor_step),
            pair["pair_id"],
        )
    )
    chosen = []
    occupied = set(_route_cells(anchor))
    for pair in ranked:
        cells = _route_cells(pair)
        if cells & occupied:
            continue
        chosen.append(pair)
        occupied.update(cells)
        if len(chosen) == count:
            return chosen
    raise RuntimeError(
        f"Could not place {count + 1} disjoint dynamic routes around {anchor['pair_id']}."
    )


def compose_pairs(problem, source_pairs, densities, causal_counts, seed, max_steps):
    """Compose paired multi-obstacle scenes without policy-dependent filtering."""

    rng = random.Random(seed)
    composed = []
    for anchor_index, anchor in enumerate(source_pairs):
        for density in densities:
            causal_count = int(causal_counts[str(density)])
            if not 1 <= causal_count <= density:
                raise ValueError("Each density requires between one and density causal obstacles.")
            candidates = [pair for pair in source_pairs if pair is not anchor]
            companions = _compatible_companions(anchor, candidates, density - 1, rng)
            selected = [anchor, *companions]
            causal = selected[:causal_count]
            risk_positions = sorted(
                {pair["first_reference_collision_step"] for pair in causal}
            )
            condition_records = {}
            for condition in ("control", "conflict"):
                specs = []
                for index, pair in enumerate(selected):
                    source_condition = (
                        "conflict"
                        if condition == "conflict" and index < causal_count
                        else "control"
                    )
                    specs.append(spec_from_record(pair[source_condition]["obstacle"]))
                if condition == "control":
                    oracle = list(problem.nominal_path)
                else:
                    oracle = multi_obstacle_safe_oracle(
                        problem,
                        specs,
                        max_steps=max_steps,
                    )
                    if oracle is None:
                        raise RuntimeError(
                            f"No safe multi-obstacle path for {anchor['pair_id']} density={density}."
                        )
                if not replay_multi_obstacle_path(problem, specs, oracle):
                    raise RuntimeError("Multi-obstacle oracle/environment disagreement.")
                condition_records[condition] = {
                    "condition": condition,
                    "obstacles": [asdict(spec) for spec in specs],
                    "obstacle_count": density,
                    "causal_obstacle_count": causal_count,
                    "oracle_steps": len(oracle) - 1,
                    "oracle_path": oracle,
                    "risk_positions": risk_positions,
                }
            composed.append(
                {
                    "pair_id": f"{anchor['pair_id']}_n{density}_{anchor_index:03d}",
                    "source_pair_id": anchor["pair_id"],
                    "obstacle_count": density,
                    "causal_obstacle_count": causal_count,
                    "first_reference_collision_step": risk_positions[0],
                    "risk_positions": risk_positions,
                    "progress_band": anchor.get("progress_band"),
                    "control": condition_records["control"],
                    "conflict": condition_records["conflict"],
                }
            )
    return composed
