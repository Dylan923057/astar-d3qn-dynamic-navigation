"""Frozen irregular maps and phase-paired, single-obstacle mechanism scenarios.

No learned policy is used to accept maps/scenarios. Oracles are dataset audits,
never agent inputs or training labels. Coordinates are (row, column).
"""
from __future__ import annotations

import heapq
import itertools
import random
from collections import Counter, deque
from dataclasses import asdict

from astar_d3qn.core.astar import astar_path, randomized_tie_astar_path
from astar_d3qn.core.grid import action_between, in_bounds, manhattan, move
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.maps.problem import NavigationProblem


def irregular_problem(seed: int, size: int = 40, demo_seed: int = 7400) -> NavigationProblem:
    """Scatter separated equipment islands, without spatial density quotas."""
    if size < 20:
        raise ValueError("Use size >= 20 for equipment islands and crossing routes.")
    rng = random.Random(seed)
    start, goal = (3, 3), (size - 4, size - 4)
    for attempt in range(200):
        blocked: set[tuple[int, int]] = set()
        islands = []
        for _ in range(2000):
            if len(blocked) >= int(size * size * 0.23):
                break
            height, width = rng.randint(2, 5), rng.randint(2, 6)
            row, col = rng.randrange(1, size - height), rng.randrange(1, size - width)
            shape = rng.choice(("rectangle", "rectangle", "L", "T"))
            cells = {(row + r, col + c) for r in range(height) for c in range(width)
                     if shape == "rectangle" or
                     (shape == "L" and (r >= height - 2 or c < 2)) or
                     (shape == "T" and (r < 2 or abs(c - width // 2) <= 0))}
            if any(manhattan(p, start) <= 3 or manhattan(p, goal) <= 3 for p in cells):
                continue
            if any((r + dr, c + dc) in blocked for r, c in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                continue
            blocked.update(cells)
            islands.append({"shape": shape, "cells": sorted(cells)})
        if len(blocked) < int(size * size * 0.20):
            continue
        path = randomized_tie_astar_path(start, goal, blocked, size, random.Random(demo_seed))
        if not path or len(path) - 1 > manhattan(start, goal) + 12:
            continue
        # Reject sealed free pockets; boundary travel remains legal and is audited.
        seen, queue = {start}, deque([start])
        while queue:
            p = queue.popleft()
            for a in range(4):
                q = move(p, a)
                if in_bounds(q, size) and q not in blocked and q not in seen:
                    seen.add(q)
                    queue.append(q)
        if len(seen) + len(blocked) != size * size:
            continue
        return NavigationProblem(
            f"irregular_workcell_{seed}", seed, size, start, goal, frozenset(blocked), tuple(path),
            {"design": "synthetic irregular equipment islands", "generation_attempt": attempt,
             "islands": islands, "manhattan_steps": manhattan(start, goal),
             "static_detour_steps": len(path) - 1 - manhattan(start, goal),
             "physical_scale": "abstract grid; no real-robot clearance claim"},
        )
    raise RuntimeError(f"Map {seed}: bounded generation exhausted; no acceptance rules relaxed.")


def spec_from_record(record: dict) -> DynamicObstacleSpec:
    return DynamicObstacleSpec(route=tuple(map(tuple, record["route"])),
                               start_index=record["start_index"], direction=record["direction"],
                               move_every=1)


def occupancy(spec: DynamicObstacleSpec, t: int) -> tuple[int, int]:
    if spec.move_every != 1:
        raise ValueError("This protocol freezes obstacle speed to one cell per step.")
    extent = len(spec.route) - 1
    phase = spec.start_index if spec.direction == 1 else 2 * extent - spec.start_index
    index = (phase + t) % (2 * extent)
    return spec.route[min(index, 2 * extent - index)]


def collision_step(path, spec) -> int | None:
    for t, position in enumerate(path[1:]):
        if position in (occupancy(spec, t), occupancy(spec, t + 1)):
            return t
    return None


def safe_oracle(problem, spec, *, allow_wait=True):
    """Shortest path in position x obstacle phase, using the environment's rule.

    The simulator protects BOTH old and next obstacle cells, more conservatively
    than vertex/edge-swap-only MAPF. Every accepted witness is replayed separately.
    """
    period = 2 * (len(spec.route) - 1)
    initial = (problem.start, 0)
    costs, parents = {initial: 0}, {}
    counter = itertools.count()
    heap = [(manhattan(problem.start, problem.goal), 0, next(counter), initial)]
    while heap:
        _, cost, _, state = heapq.heappop(heap)
        if costs.get(state) != cost:
            continue
        position, phase = state
        if position == problem.goal:
            path = [position]
            while state in parents:
                state = parents[state]
                path.append(state[0])
            return list(reversed(path))
        blocked_dynamic = {occupancy(spec, phase), occupancy(spec, phase + 1)}
        for action in range(5 if allow_wait else 4):
            following = move(position, action)
            if (not in_bounds(following, problem.size) or following in problem.obstacles
                    or following in blocked_dynamic):
                continue
            next_state = (following, (phase + 1) % period)
            if cost + 1 >= costs.get(next_state, 10**9):
                continue
            costs[next_state] = cost + 1
            parents[next_state] = state
            heapq.heappush(heap, (cost + 1 + manhattan(following, problem.goal),
                                  cost + 1, next(counter), next_state))
    return None


def replay_witness(problem, spec, path) -> bool:
    if not path or tuple(path[0]) != problem.start or tuple(path[-1]) != problem.goal:
        return False
    env = DynamicGridNavigationEnv(problem, [spec], max_steps=len(path) + 1,
                                   terminate_on_collision=True, window_size=15)
    env.reset()
    for left, right in zip(path, path[1:]):
        result = env.step(int(action_between(tuple(left), tuple(right))))
        if result.info["collision"]:
            return False
    return env.position == problem.goal


def reactive_wait_witness(problem, spec, radius=7):
    """Follow reference; wait only when a visible obstacle blocks the next move."""
    path, index = [problem.start], 0
    reference = problem.nominal_path
    for t in range(len(reference) + 12):
        if index == len(reference) - 1:
            return path
        danger = {occupancy(spec, t), occupancy(spec, t + 1)}
        following = reference[index + 1]
        if following in danger:
            here = reference[index]
            if here in danger or max(abs(here[d] - occupancy(spec, t)[d]) for d in (0, 1)) > radius:
                return None
            path.append(here)
        else:
            path.append(following)
            index += 1
    return None


def local_bypass_witness(problem, spec, hit, radius=7):
    """Deviate at the last safe reference state; never use an invisible decision."""
    reference = problem.nominal_path
    here = reference[hit]
    if max(abs(here[d] - occupancy(spec, hit)[d]) for d in (0, 1)) > radius:
        return None
    candidates = []
    for rejoin in range(hit + 2, min(hit + 11, len(reference))):
        segment = astar_path(here, reference[rejoin],
                             problem.obstacles | frozenset(spec.route) | frozenset(reference[:hit]),
                             problem.size)
        if segment is None or any(max(abs(p[d] - here[d]) for d in (0, 1)) > radius for p in segment):
            continue
        path = list(reference[:hit]) + segment + list(reference[rejoin + 1:])
        if len(set(path)) != len(path) or len(path) - len(reference) > 8:
            continue
        if collision_step(path, spec) is None:
            candidates.append((len(path), len(segment), path))
    return min(candidates)[2] if candidates else None


PROGRESS_BANDS = ("early", "middle", "late")


def progress_band(step, total_steps):
    return PROGRESS_BANDS[min(2, 3 * step // total_steps)]


def coverage_routes(problem):
    """Short perpendicular crossings and one-turn clearance routes, including corners."""
    reference = problem.nominal_path
    routes = set()
    # Keep two history frames before a decision. Include the goal approach;
    # the intersection check below still forbids touching start/goal themselves.
    for index in range(3, len(reference) - 1):
        center = reference[index]
        for delta in ((1, 0), (0, 1)):
            tangent = (delta[1], delta[0])
            candidates = []
            for length in (3, 5, 7, 9):
                half = length // 2
                candidates.append(tuple((center[0] + k * delta[0], center[1] + k * delta[1])
                                        for k in range(-half, half + 1)))
            # At a reference corner the unused free neighbors are perpendicular:
            # a route turning AT the crossing can fit where a straight route cannot.
            for incoming in (-1, 1):
                for outgoing in (-1, 1):
                    for approach in (1, 2):
                        for clearance in (1, 2):
                            first = tuple((center[0] + incoming * k * delta[0],
                                           center[1] + incoming * k * delta[1])
                                          for k in range(approach, -1, -1))
                            second = tuple((center[0] + outgoing * k * tangent[0],
                                            center[1] + outgoing * k * tangent[1])
                                           for k in range(1, clearance + 1))
                            candidates.append(first + second)
            for side in (-1, 1):
                for approach in (1, 2):
                    for clearance in (1, 2):
                        stem = tuple((center[0] + side * k * delta[0], center[1] + side * k * delta[1])
                                     for k in range(-approach, clearance + 1))
                        for turn in (-1, 1):
                            for tail_length in (1, 2):
                                tail = tuple((stem[-1][0] + turn * k * tangent[0],
                                              stem[-1][1] + turn * k * tangent[1])
                                             for k in range(1, tail_length + 1))
                                candidates.append(stem + tail)
            for route in candidates:
                if (len(set(route)) == len(route)
                        and all(in_bounds(p, problem.size) and p not in problem.obstacles for p in route)
                        and set(route).intersection(reference) == {center}):
                    # Reversed geometry is the same periodic physical route.
                    routes.add(min(route, tuple(reversed(route))))
    return routes


def select_coverage_pairs(accepted, counts, total_steps, rng, minimum_gap=2):
    """Hard thirds quotas and distinct, spaced decision positions within each split.

    Bounded backtracking allocates pairs jointly; no fallback to a different band.
    Selection uses geometry/phase metadata only, never policy performance.
    """
    if any(count <= 0 or count % 3 for count in counts.values()):
        raise ValueError("Balanced pair counts must be positive multiples of three.")
    splits = {name: [] for name in counts}
    for band in PROGRESS_BANDS:
        pool = [p for p in accepted if progress_band(p["first_reference_collision_step"], total_steps) == band]
        grouped = {}
        for pair in pool:
            grouped.setdefault(pair["first_reference_collision_step"], []).append(pair)
        for values in grouped.values():
            rng.shuffle(values)
        tasks = [name for name in sorted(counts, key=lambda k: -counts[k]) for _ in range(counts[name] // 3)]
        selected = {name: [] for name in counts}
        used = set()
        visits = 0
        def search(task_index):
            nonlocal visits
            visits += 1
            if visits > 50000:
                return False
            if task_index == len(tasks):
                return True
            split = tasks[task_index]
            chosen_steps = [p["first_reference_collision_step"] for p in selected[split] + splits[split]]
            candidates = [step for step in grouped
                          if all(abs(step - prior) >= minimum_gap for prior in chosen_steps)
                          and any(id(pair) not in used for pair in grouped[step])]
            # Prefer spread-out positions, including both ends of each third.
            low = total_steps * PROGRESS_BANDS.index(band) / 3
            high = total_steps * (PROGRESS_BANDS.index(band) + 1) / 3 - 1
            candidates.sort(key=lambda step: (-min(abs(step - prior) for prior in chosen_steps)
                                              if chosen_steps else abs(step - (low + high) / 2), step))
            if len(candidates) < tasks[task_index:].count(split):
                return False
            for step in candidates:
                # Pairs from this generator are already disjoint within a route.
                pair = next(pair for pair in grouped[step] if id(pair) not in used)
                selected[split].append(pair)
                used.add(id(pair))
                if search(task_index + 1):
                    return True
                selected[split].pop()
                used.remove(id(pair))
            return False
        if not search(0):
            raise RuntimeError(f"Coverage quota failed for {band}: {len(pool)} pairs at "
                               f"positions {sorted(grouped)}; required per split "
                               f"{ {k: v // 3 for k, v in counts.items()} }, min gap={minimum_gap}. "
                               "Other bands will not fill this shortfall.")
        for name in counts:
            splits[name].extend(selected[name])
    return splits


def scenario_pairs(problem, counts, demo_seed=7400, demo_episodes=20, radius=7, coverage=None):
    """Split complete phase pairs; routes may be shared across splits.

    Reference A* is one of the exact demonstration routes. A label describes
    available responses, NOT the only acceptable behavior. Neither a mandatory
    wait nor worse learned-policy performance is an acceptance requirement.
    """
    if coverage and (coverage.get("mode") != "equal_path_thirds"
                     or coverage.get("minimum_decision_step_gap", 0) < 1):
        raise ValueError("Coverage requires equal_path_thirds and a positive decision-step gap.")
    rng = random.Random(problem.seed + 9000)
    demo_rng = random.Random(demo_seed)
    demos = [randomized_tie_astar_path(problem.start, problem.goal, problem.obstacles,
                                      problem.size, demo_rng) for _ in range(demo_episodes)]
    reference = problem.nominal_path
    routes = set()
    for index in range(8, len(reference) - 8):
        before, center, after = reference[index - 1:index + 2]
        if before[0] == after[0]:
            delta = (1, 0)
        elif before[1] == after[1]:
            delta = (0, 1)
        else:
            continue
        for length in (5, 7, 9):
            half = length // 2
            route = tuple((center[0] + k * delta[0], center[1] + k * delta[1])
                          for k in range(-half, half + 1))
            if (all(in_bounds(p, problem.size) and p not in problem.obstacles for p in route)
                    and set(route).intersection(reference) == {center}):
                routes.add(route)
    candidates = sorted(coverage_routes(problem) if coverage else routes)
    rng.shuffle(candidates)
    if coverage:
        candidates.sort(key=len)
    accepted = []
    accepted_positions = Counter()
    for route in candidates:
        encounter_step = next(i - 1 for i, p in enumerate(reference) if p in route)
        kind = "straight" if len({p[0] for p in route}) == 1 or len({p[1] for p in route}) == 1 else "turning"
        if coverage and accepted_positions[(encounter_step, kind)] >= 4:
            continue
        controls, conflicts = [], []
        extent = len(route) - 1
        for phase in range(2 * extent):
            spec = DynamicObstacleSpec(route, min(phase, 2 * extent - phase),
                                       1 if phase < extent else -1)
            hit = collision_step(reference, spec)
            if hit is None:
                controls.append(spec)
                continue
            wait = reactive_wait_witness(problem, spec, radius)
            bypass = local_bypass_witness(problem, spec, hit, radius)
            if wait is None or bypass is None:
                continue
            optimal = safe_oracle(problem, spec)
            if optimal is None or not all(replay_witness(problem, spec, p)
                                          for p in (wait, bypass, optimal)):
                raise RuntimeError("Oracle/environment disagreement.")
            conflicts.append((spec, hit, wait, bypass, optimal))
        demo_hits = lambda spec: sum(collision_step(p, spec) is not None for p in demos)
        rng.shuffle(controls)
        rng.shuffle(conflicts)
        controls.sort(key=demo_hits)
        conflicts.sort(key=lambda item: demo_hits(item[0]))
        for conflict, hit, wait, bypass, optimal in conflicts:
            eligible = [spec for spec in controls if demo_hits(spec) < demo_hits(conflict)]
            if not eligible:
                continue
            control = eligible[0]
            controls.remove(control)
            if not replay_witness(problem, control, reference):
                raise RuntimeError("Control is not safe under actual simulator semantics.")
            def record(spec, condition, oracle):
                return {"condition": condition, "obstacle": asdict(spec),
                        "oracle_steps": len(oracle) - 1, "oracle_path": oracle,
                        "demo_conflict_count": demo_hits(spec)}
            accepted.append({"route": route, "first_reference_collision_step": hit,
                             "progress_band": progress_band(hit, problem.astar_steps),
                             "route_kind": kind,
                             "decision_position": reference[hit],
                             "wait_witness": wait, "bypass_witness": bypass,
                             "wait_steps": len(wait) - len(reference),
                             "bypass_excess_steps": len(bypass) - len(reference),
                             "control": record(control, "control", reference),
                             "conflict": record(conflict, "conflict", optimal)})
            accepted_positions[(hit, kind)] += 1
    required = sum(counts.values())
    if len(accepted) < required:
        raise RuntimeError(f"{problem.map_id}: only {len(accepted)} valid pairs for {required}; "
                           "do not train or silently relax the protocol.")
    if coverage:
        splits = select_coverage_pairs(accepted, counts, problem.astar_steps, rng,
                                       coverage["minimum_decision_step_gap"])
        for index, pair in enumerate(p for group in splits.values() for p in group):
            pair["pair_id"] = f"pair_{index:03d}"
    else:
        # Legacy v1 selection, retained only for reproducing old datasets.
        accepted.sort(key=lambda p: p["first_reference_collision_step"])
        selected = [accepted[min(len(accepted) - 1, int((i + .5) * len(accepted) / required))]
                    for i in range(required)]
        split_names = [name for name, count in counts.items() for _ in range(count)]
        rng.shuffle(split_names)
        splits = {name: [] for name in counts}
        for index, (pair, split) in enumerate(zip(selected, split_names)):
            pair["pair_id"] = f"pair_{index:03d}"
            splits[split].append(pair)
    return {"splits": splits, "audit": {"candidate_routes": len(candidates),
            "valid_candidate_pairs": len(accepted), "selected_pairs": required,
            "route_geometry_may_be_shared": True, "exact_scenarios_disjoint": True,
            "reference_is_demo_0": reference == tuple(demos[0]),
            "both_wait_and_local_bypass_verified": True,
            "coverage": {name: {"bands": dict(Counter(progress_band(p["first_reference_collision_step"], problem.astar_steps)
                                                        for p in pairs)),
                                  "unique_decision_positions": len({p["first_reference_collision_step"] for p in pairs}),
                                  "decision_steps": sorted(p["first_reference_collision_step"] for p in pairs),
                                  "route_kinds": dict(Counter(p["route_kind"] for p in pairs))}
                         for name, pairs in splits.items()},
            "purpose": "adaptation mechanism, not unseen-route or cross-map transfer"}}
