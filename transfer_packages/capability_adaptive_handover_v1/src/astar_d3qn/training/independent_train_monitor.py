"""Deterministic independent train-monitor scene construction for v2."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from astar_d3qn.core.grid import ACTION_DELTAS, in_bounds, manhattan, move
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.maps.adaptation import collision_step, occupancy, spec_from_record


def _scene_fingerprint(scene: Mapping[str, Any]) -> str:
    payload = {
        "condition": scene.get("condition", "conflict"),
        "obstacles": scene.get("obstacles", []),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _route_cells(record: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(cell) for cell in record["route"])


def _multi_obstacle_safe_oracle(problem: Any, specs: Sequence[DynamicObstacleSpec], max_steps: int):
    import heapq
    import itertools

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
                (cost + 1 + manhattan(following, problem.goal), cost + 1, next(counter), next_state),
            )
    return None


def _replay_path(problem: Any, specs: Sequence[DynamicObstacleSpec], path: Sequence[Sequence[int]]) -> bool:
    if not path or tuple(path[0]) != problem.start or tuple(path[-1]) != problem.goal:
        return False
    env = DynamicGridNavigationEnv(
        problem, list(specs), max_steps=len(path) + 1, terminate_on_collision=True, window_size=15
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


def _source_obstacles(problem: Any, splits: Mapping[str, Sequence[Mapping[str, Any]]]):
    records = []
    for split_name, pairs in splits.items():
        for pair in pairs:
            for index, conflict_record in enumerate(pair["conflict"]["obstacles"]):
                control_record = pair["control"]["obstacles"][index]
                conflict_spec = spec_from_record(conflict_record)
                hit = collision_step(problem.nominal_path, conflict_spec)
                if hit is None:
                    continue
                records.append(
                    {
                        "source_split": split_name,
                        "source_pair_id": pair["pair_id"],
                        "source_index": index,
                        "route": conflict_record["route"],
                        "first_reference_collision_step": hit,
                        "conflict": conflict_record,
                        "control": control_record,
                    }
                )
    return records


def generate_train_monitor_scenes(
    problem: Any,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate new conflict scenes, excluding exact scenes from every split."""

    densities = {int(key): int(value) for key, value in settings["densities"].items()}
    causal_counts = {int(key): int(value) for key, value in settings["causal_obstacles"].items()}
    expected_count = int(settings["independent_scene_count"])
    if sum(densities.values()) != expected_count or set(densities) != {3, 5}:
        raise ValueError("v2 monitor densities must be exactly 12 at density 3 and 12 at density 5.")

    existing = set()
    for pairs in splits.values():
        for pair in pairs:
            for condition in ("control", "conflict"):
                record = pair[condition]
                existing.add(_scene_fingerprint({"condition": condition, **record}))

    source = _source_obstacles(problem, splits)
    source = [item for item in source if item["source_split"] == settings["source_split"]]
    if len(source) < 30:
        raise ValueError("Not enough independent source obstacle components for v2 monitor generation.")
    rng = random.Random(int(settings["generation_seed"]) + problem.seed)
    selected: list[dict[str, Any]] = []
    selected_fingerprints = set()
    attempts = 0
    max_attempts = 20000
    for density in (3, 5):
        target = densities[density]
        causal_count = causal_counts[density]
        while sum(scene["obstacle_count"] == density for scene in selected) < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Unable to construct {target} independent density-{density} monitor scenes.")
            anchor = rng.choice(source)
            candidates = list(source)
            rng.shuffle(candidates)
            chosen = [anchor]
            occupied = set(_route_cells(anchor["conflict"]))
            for candidate in candidates:
                if candidate is anchor:
                    continue
                cells = _route_cells(candidate["conflict"])
                if cells & occupied:
                    continue
                chosen.append(candidate)
                occupied.update(cells)
                if len(chosen) == density:
                    break
            if len(chosen) != density:
                continue
            conflict_specs = [spec_from_record(item["conflict"]) for item in chosen[:causal_count]]
            conflict_specs.extend(spec_from_record(item["control"]) for item in chosen[causal_count:])
            if not any(collision_step(problem.nominal_path, spec) is not None for spec in conflict_specs[:causal_count]):
                continue
            oracle = _multi_obstacle_safe_oracle(problem, conflict_specs, 300)
            if oracle is None or not _replay_path(problem, conflict_specs, oracle):
                continue
            scene = {
                "condition": "conflict",
                "source_split": "train_monitor",
                "pair_id": f"train_monitor_n{density}_{len(selected):03d}",
                "scenario_id": f"{problem.map_id}_train_monitor_n{density}_{len(selected):03d}",
                "obstacle_count": density,
                "causal_obstacle_count": causal_count,
                "obstacles": [asdict(spec) for spec in conflict_specs],
                "oracle_steps": len(oracle) - 1,
                "oracle_path": oracle,
                "source_component_ids": [
                    f"{item['source_split']}:{item['source_pair_id']}:{item['source_index']}"
                    for item in chosen
                ],
            }
            fingerprint = _scene_fingerprint(scene)
            if fingerprint in existing or fingerprint in selected_fingerprints:
                continue
            scene["scene_sha256"] = fingerprint
            selected.append(scene)
            selected_fingerprints.add(fingerprint)

    selected.sort(key=lambda scene: scene["scenario_id"])
    for index, scene in enumerate(selected):
        density = scene["obstacle_count"]
        scene["pair_id"] = f"train_monitor_n{density}_{index:03d}"
        scene["scenario_id"] = f"{problem.map_id}_train_monitor_n{density}_{index:03d}"
    manifest = {
        "map_id": problem.map_id,
        "generation_seed": int(settings["generation_seed"]) + problem.seed,
        "source_split": "train",
        "scene_count": len(selected),
        "density_counts": {str(density): sum(s["obstacle_count"] == density for s in selected) for density in (3, 5)},
        "scene_ids": [scene["scenario_id"] for scene in selected],
        "scene_sha256": [scene["scene_sha256"] for scene in selected],
        "all_existing_split_scene_fingerprints_excluded": True,
        "scenes_in_training_stream": False,
        "scenes_in_replay": False,
        "used_for_astar_demo": False,
        "scenes": selected,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return selected, manifest
