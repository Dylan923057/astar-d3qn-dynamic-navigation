from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from astar_d3qn.utils.io import load_json, write_json

from .problem import NavigationProblem


def problem_record(problem: NavigationProblem) -> dict[str, Any]:
    return {
        "map_id": problem.map_id,
        "seed": problem.seed,
        "size": problem.size,
        "start": list(problem.start),
        "goal": list(problem.goal),
        "obstacles": [list(cell) for cell in sorted(problem.obstacles)],
        "nominal_path": [list(cell) for cell in problem.nominal_path],
        "obstacle_count": len(problem.obstacles),
        "density": problem.density,
        "astar_steps": problem.astar_steps,
        "grid_sha256": problem.grid_sha256,
        "metadata": dict(problem.metadata),
    }


def problem_from_record(record: Mapping[str, Any]) -> NavigationProblem:
    problem = NavigationProblem(
        map_id=str(record["map_id"]),
        seed=int(record["seed"]),
        size=int(record["size"]),
        start=tuple(int(value) for value in record["start"]),
        goal=tuple(int(value) for value in record["goal"]),
        obstacles=frozenset(
            tuple(int(value) for value in cell) for cell in record["obstacles"]
        ),
        nominal_path=tuple(
            tuple(int(value) for value in cell) for cell in record["nominal_path"]
        ),
        metadata=dict(record.get("metadata", {})),
    )
    if problem.grid_sha256 != str(record["grid_sha256"]):
        raise ValueError(f"Map hash mismatch for {problem.map_id}.")
    if problem.astar_steps != int(record["astar_steps"]):
        raise ValueError(f"A* path length mismatch for {problem.map_id}.")
    return problem


def save_problem_set(
    problems: Sequence[NavigationProblem],
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not problems:
        raise ValueError("Cannot save an empty map set.")
    payload = {
        "format_version": 1,
        **dict(metadata or {}),
        "count": len(problems),
        "maps": [problem_record(problem) for problem in problems],
    }
    write_json(payload, path)
    return payload


def load_problem_set(path: str | Path) -> list[NavigationProblem]:
    payload = load_json(path)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported map-set format.")
    records = payload.get("maps")
    if not isinstance(records, list) or not records:
        raise ValueError("Map set must contain at least one map.")
    if int(payload.get("count", -1)) != len(records):
        raise ValueError("Map-set count does not match its map records.")
    problems = [problem_from_record(record) for record in records]
    map_ids = [problem.map_id for problem in problems]
    if len(map_ids) != len(set(map_ids)):
        raise ValueError("Map set contains duplicate map IDs.")
    return problems
