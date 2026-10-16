"""Audit spatial conflict and evaluate selected models on paired stress strata."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import (
    DynamicScenario,
    ScheduledDynamicEnvironmentFactory,
)
from astar_d3qn.envs.spatial_scenarios import (
    scenarios_from_spatial_manifest,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.evaluation.conflict import (
    obstacle_conflict_metrics,
    route_record_lookup,
    scenario_conflict_metrics,
)
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything


MAP_CONFIGS = {
    "map01": "configs/dynamic_spatial_generalization_map01_checkpoint_v2.yaml",
    "map02": "configs/dynamic_spatial_generalization_map02_confirmatory_v2.yaml",
}


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    map_key: str
    method: str
    config_path: str
    strategy: str


EVALUATIONS = (
    EvaluationSpec("map01", "uniform", MAP_CONFIGS["map01"], "uniform"),
    EvaluationSpec("map01", "prefill", MAP_CONFIGS["map01"], "prefill"),
    EvaluationSpec(
        "map01", "persistent25", MAP_CONFIGS["map01"], "persistent_demo"
    ),
    EvaluationSpec(
        "map01",
        "persistent10",
        "configs/dynamic_spatial_generalization_map01_demo10_v2.yaml",
        "persistent_demo",
    ),
    EvaluationSpec(
        "map01",
        "decay25to0",
        "configs/dynamic_spatial_generalization_map01_demo_decay_v2.yaml",
        "persistent_demo",
    ),
    EvaluationSpec("map02", "uniform", MAP_CONFIGS["map02"], "uniform"),
    EvaluationSpec("map02", "prefill", MAP_CONFIGS["map02"], "prefill"),
    EvaluationSpec(
        "map02", "persistent25", MAP_CONFIGS["map02"], "persistent_demo"
    ),
)


def _resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def _load_problem(config: Mapping[str, Any]):
    map_set = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(map_set["file"])))
    map_id = str(config["map"]["scene"])
    matches = [problem for problem in problems if problem.map_id == map_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered problem named {map_id!r}.")
    return matches[0]


def _reward_config(values: Mapping[str, Any]) -> RewardConfig:
    return RewardConfig(**{key: float(value) for key, value in values.items()})


def _agent(config: Mapping[str, Any], seed: int, device: str) -> D3QNAgent:
    environment = config["environment"]
    values = config["agent"]
    return D3QNAgent(
        D3QNConfig(
            spatial_shape=(
                int(environment["spatial_channels"]),
                int(environment["window_size"]),
                int(environment["window_size"]),
            ),
            scalar_dim=2,
            action_dim=int(environment["action_count"]),
            learning_rate=float(values["learning_rate"]),
            gamma=float(values["gamma"]),
            target_sync_interval=int(values["target_sync_interval"]),
            gradient_clip_norm=float(values["gradient_clip_norm"]),
            hidden_dim=int(values["hidden_dim"]),
            device=device,
            seed=seed,
        )
    )


def _flatten_conflict_row(
    metrics: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        **extra,
        **{key: value for key, value in metrics.items() if key != "obstacle_metrics"},
    }


def _audit_manifests(
    output_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    contexts: dict[str, dict[str, Any]] = {}
    scenario_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for map_key, config_name in MAP_CONFIGS.items():
        config_path = _resolve(config_name)
        config = load_config(config_path)
        problem = _load_problem(config)
        manifest_path = _resolve(config["spatial_generalization"]["manifest"])
        manifest = load_json(manifest_path)
        validate_spatial_scenario_manifest(problem, manifest)
        contexts[map_key] = {
            "config": config,
            "config_path": config_path,
            "problem": problem,
            "manifest": manifest,
            "manifest_path": manifest_path,
        }
        for split in ("train", "validation", "test"):
            lookup = route_record_lookup(manifest, split)
            scenarios = scenarios_from_spatial_manifest(problem, manifest, split)
            for scenario in scenarios:
                metrics = scenario_conflict_metrics(
                    problem,
                    scenario,
                    temporal_window=2,
                    route_records=lookup,
                )
                scenario_rows.append(
                    _flatten_conflict_row(
                        metrics,
                        map_key=map_key,
                        map_id=problem.map_id,
                        split=split,
                    )
                )
            for category in ("corridor", "background"):
                for record in manifest["route_pools"][split][category]:
                    route = tuple(tuple(cell) for cell in record["route"])
                    phase_metrics = []
                    for start_index in range(len(route)):
                        for direction in (-1, 1):
                            spec = DynamicObstacleSpec(
                                route=route,
                                start_index=start_index,
                                direction=direction,
                                move_every=int(manifest["generation"]["move_every"]),
                                label=str(record["route_id"]),
                            )
                            phase_metrics.append(
                                obstacle_conflict_metrics(problem, spec, temporal_window=2)
                            )
                    route_rows.append(
                        {
                            "map_key": map_key,
                            "map_id": problem.map_id,
                            "split": split,
                            "route_id": record["route_id"],
                            "category": category,
                            "center_row": record["center"][0],
                            "center_column": record["center"][1],
                            "orientation": record["orientation"],
                            "corridor_frequency": record["corridor_frequency"],
                            "direct_intersection_cell_count": max(
                                int(row["direct_intersection_cell_count"])
                                for row in phase_metrics
                            ),
                            "minimum_route_distance": min(
                                int(row["min_route_distance"])
                                for row in phase_metrics
                            ),
                            "minimum_exact_conflicts_over_phases": min(
                                int(row["exact_temporal_conflict_count"])
                                for row in phase_metrics
                            ),
                            "maximum_exact_conflicts_over_phases": max(
                                int(row["exact_temporal_conflict_count"])
                                for row in phase_metrics
                            ),
                            "minimum_aligned_conflicts_over_phases": min(
                                int(row["aligned_temporal_conflict_count"])
                                for row in phase_metrics
                            ),
                            "maximum_aligned_conflicts_over_phases": max(
                                int(row["aligned_temporal_conflict_count"])
                                for row in phase_metrics
                            ),
                        }
                    )
    write_records_csv(scenario_rows, output_dir / "manifest_scenario_conflicts.csv")
    write_records_csv(route_rows, output_dir / "manifest_route_conflicts.csv")
    return contexts, scenario_rows, route_rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _manifest_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["map_key"]), str(row["split"]))].append(row)
    result = []
    for (map_key, split), group in sorted(groups.items()):
        result.append(
            {
                "map_key": map_key,
                "split": split,
                "scenario_count": len(group),
                "no_direct_intersection_rate": sum(
                    int(row["direct_intersection_route_count"]) == 0 for row in group
                )
                / len(group),
                "two_or_more_direct_routes_rate": sum(
                    int(row["direct_intersection_route_count"]) >= 2 for row in group
                )
                / len(group),
                "exact_temporal_conflict_rate": sum(
                    int(row["exact_temporal_conflict_count"]) > 0 for row in group
                )
                / len(group),
                "aligned_temporal_conflict_rate": sum(
                    int(row["aligned_temporal_conflict_count"]) > 0 for row in group
                )
                / len(group),
                "mean_direct_intersection_routes": _mean(
                    group, "direct_intersection_route_count"
                ),
                "mean_exact_temporal_conflicts": _mean(
                    group, "exact_temporal_conflict_count"
                ),
                "mean_aligned_temporal_conflicts": _mean(
                    group, "aligned_temporal_conflict_count"
                ),
                "mean_conflict_score": _mean(group, "conflict_score"),
            }
        )
    return result


def _phase_options(
    problem,
    record: Mapping[str, Any],
    move_every: int,
) -> list[tuple[DynamicObstacleSpec, dict[str, Any]]]:
    route = tuple(tuple(cell) for cell in record["route"])
    result = []
    for start_index in range(len(route)):
        for direction in (-1, 1):
            spec = DynamicObstacleSpec(
                route=route,
                start_index=start_index,
                direction=direction,
                move_every=move_every,
                label=str(record["route_id"]),
                reference_path_source=str(record["category"]),
            )
            metrics = obstacle_conflict_metrics(problem, spec, temporal_window=2)
            result.append((spec, metrics))
    return result


def _phase_rank(item: tuple[DynamicObstacleSpec, Mapping[str, Any]]) -> tuple[int, int, int]:
    _, metrics = item
    offset = metrics["minimum_phase_offset"]
    return (
        int(metrics["exact_temporal_conflict_count"]),
        int(metrics["aligned_temporal_conflict_count"]),
        -(int(offset) if offset is not None else 10_000),
    )


def _select_phase(
    options: Sequence[tuple[DynamicObstacleSpec, dict[str, Any]]],
    level: str,
) -> DynamicObstacleSpec:
    ordered = sorted(options, key=_phase_rank)
    if level == "low":
        return ordered[0][0]
    if level == "high":
        return ordered[-1][0]
    if level == "medium":
        return ordered[len(ordered) // 2][0]
    raise ValueError(f"Unknown phase level: {level}")


def _build_stress_scenarios(
    contexts: Mapping[str, Mapping[str, Any]],
    scenarios_per_stratum: int,
    generation_seed: int,
    output_dir: Path,
) -> tuple[
    dict[str, tuple[DynamicScenario, ...]],
    dict[tuple[str, int], dict[str, Any]],
]:
    schedules: dict[str, tuple[DynamicScenario, ...]] = {}
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    serialized: dict[str, Any] = {
        "format_version": 1,
        "purpose": "post_hoc_supplemental_paired_phase_conflict_stress",
        "generation_seed": generation_seed,
        "scenarios_per_stratum": scenarios_per_stratum,
        "maps": {},
    }
    for map_index, (map_key, context) in enumerate(sorted(contexts.items())):
        problem = context["problem"]
        manifest = context["manifest"]
        pool = manifest["route_pools"]["test"]
        records = {
            str(record["route_id"]): record
            for category in ("corridor", "background")
            for record in pool[category]
        }
        move_every = int(manifest["generation"]["move_every"])
        phase_options = {
            route_id: _phase_options(problem, record, move_every)
            for route_id, record in records.items()
        }
        corridor_ids = [str(item["route_id"]) for item in pool["corridor"]]
        background_ids = [str(item["route_id"]) for item in pool["background"]]
        combinations = list(
            itertools.product(
                itertools.combinations(corridor_ids, 3),
                itertools.combinations(background_ids, 2),
            )
        )
        rng = random.Random(generation_seed + map_index * 10_000)
        rng.shuffle(combinations)
        candidates = []
        for corridor_combo, background_combo in combinations:
            route_ids = tuple(corridor_combo) + tuple(background_combo)
            level_scenarios = {}
            level_metrics = {}
            for level in ("low", "medium", "high"):
                specs = tuple(
                    _select_phase(phase_options[route_id], level)
                    for route_id in route_ids
                )
                provisional = DynamicScenario(seed=-1, obstacles=specs)
                level_scenarios[level] = provisional
                level_metrics[level] = scenario_conflict_metrics(
                    problem,
                    provisional,
                    temporal_window=2,
                    route_records=records,
                )
            delta = (
                float(level_metrics["high"]["conflict_score"])
                - float(level_metrics["low"]["conflict_score"])
            )
            direct_routes = int(
                level_metrics["high"]["direct_intersection_route_count"]
            )
            if delta <= 0 or direct_routes <= 0:
                continue
            candidates.append(
                (
                    direct_routes,
                    delta,
                    rng.random(),
                    route_ids,
                    level_scenarios,
                    level_metrics,
                )
            )
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if len(candidates) < scenarios_per_stratum:
            raise RuntimeError(
                f"Only {len(candidates)} paired stress blocks available for {map_key}."
            )
        # Select from the strongest eligible half with an even stride so that one
        # exceptional route combination cannot dominate all blocks.
        eligible = candidates[: max(scenarios_per_stratum, len(candidates) // 2)]
        selected = []
        used_combinations: set[tuple[str, ...]] = set()
        for index in range(scenarios_per_stratum):
            position = min(
                len(eligible) - 1,
                int(index * len(eligible) / scenarios_per_stratum),
            )
            while position < len(eligible) and eligible[position][3] in used_combinations:
                position += 1
            if position >= len(eligible):
                raise RuntimeError("Could not choose unique stress route combinations.")
            selected.append(eligible[position])
            used_combinations.add(eligible[position][3])

        map_scenarios: list[DynamicScenario] = []
        map_json = {"map_id": problem.map_id, "strata": {}}
        base = 30_000 + map_index * 10_000
        for stratum_index, level in enumerate(("low", "medium", "high"), start=1):
            records_json = []
            for block_index, candidate in enumerate(selected):
                route_ids = candidate[3]
                provisional = candidate[4][level]
                scenario_id = base + stratum_index * 1_000 + block_index
                scenario = DynamicScenario(
                    seed=scenario_id,
                    obstacles=provisional.obstacles,
                )
                metrics = scenario_conflict_metrics(
                    problem,
                    scenario,
                    temporal_window=2,
                    route_records=records,
                )
                map_scenarios.append(scenario)
                metadata[(map_key, scenario_id)] = {
                    "map_key": map_key,
                    "map_id": problem.map_id,
                    "scenario_id": scenario_id,
                    "stratum": level,
                    "block_id": block_index,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "obstacle_metrics"
                    },
                }
                records_json.append(
                    {
                        "scenario_id": scenario_id,
                        "block_id": block_index,
                        "route_ids": list(route_ids),
                        "obstacles": [
                            {
                                "route_id": spec.label,
                                "start_index": spec.start_index,
                                "direction": spec.direction,
                                "move_every": spec.move_every,
                            }
                            for spec in scenario.obstacles
                        ],
                        "conflict_metrics": {
                            key: value
                            for key, value in metrics.items()
                            if key != "obstacle_metrics"
                        },
                    }
                )
            map_json["strata"][level] = records_json
        schedules[map_key] = tuple(map_scenarios)
        serialized["maps"][map_key] = map_json
    write_json(serialized, output_dir / "stress_scenarios.json")
    write_records_csv(
        list(metadata.values()), output_dir / "stress_scenario_conflicts.csv"
    )
    return schedules, metadata


def _run_model_evaluations(
    contexts: Mapping[str, Mapping[str, Any]],
    schedules: Mapping[str, Sequence[DynamicScenario]],
    metadata: Mapping[tuple[str, int], Mapping[str, Any]],
    device: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in EVALUATIONS:
        config = load_config(_resolve(spec.config_path))
        context = contexts[spec.map_key]
        problem = context["problem"]
        scenarios = tuple(schedules[spec.map_key])
        output_root = _resolve(config["experiment"]["output_root"])
        prefix = str(
            config["experiment"].get(
                "run_name_prefix", "dynamic_spatial_generalization_map01"
            )
        )
        seeds = tuple(int(value) for value in config["training"]["seeds"])
        environment = config["environment"]
        reward = _reward_config(config["reward"])
        for seed in seeds:
            seed_everything(seed)
            model_path = (
                output_root
                / f"{prefix}_seed_{seed}_{spec.strategy}"
                / "model_selected.pth"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Missing selected model: {model_path}")
            agent = _agent(config, seed, device)
            agent.load_weights(model_path)
            factory = ScheduledDynamicEnvironmentFactory(scenarios, scenarios)
            factory.set_mode("eval")
            summary, details, _ = evaluate_agent(
                agent,
                [problem] * len(scenarios),
                max_steps=int(environment["max_steps"]),
                reward_config=reward,
                terminate_on_collision=bool(environment["terminate_on_collision"]),
                window_size=int(environment["window_size"]),
                environment_factory=factory,
            )
            for detail in details:
                scenario_id = int(detail["scenario_id"])
                intrinsic = metadata[(spec.map_key, scenario_id)]
                rows.append(
                    {
                        "map_key": spec.map_key,
                        "map_id": problem.map_id,
                        "method": spec.method,
                        "training_seed": seed,
                        "stratum": intrinsic["stratum"],
                        "block_id": intrinsic["block_id"],
                        **detail,
                        **{
                            f"intrinsic_{key}": value
                            for key, value in intrinsic.items()
                            if key
                            not in {
                                "map_key",
                                "map_id",
                                "scenario_id",
                                "stratum",
                                "block_id",
                            }
                        },
                    }
                )
            print(
                f"[{spec.map_key} {spec.method} seed={seed}] "
                f"safe={summary['safe_success_rate']:.1%} "
                f"dynamic_collision={summary['dynamic_collision_rate']:.1%}",
                flush=True,
            )
    write_records_csv(rows, output_dir / "stress_evaluation.csv")
    return rows


def _seed_summary(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    by_seed: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields) + (row["training_seed"],)
        by_seed[key].append(row)
    per_seed = []
    for key, group in sorted(by_seed.items()):
        per_seed.append(
            {
                **dict(zip((*group_fields, "training_seed"), key)),
                "scenario_count": len(group),
                "success_rate": _mean(group, "success"),
                "safe_success_rate": _mean(group, "safe_success"),
                "dynamic_collision_rate": _mean(group, "dynamic_collision"),
                "mean_dynamic_collision_count": _mean(
                    group, "dynamic_collision_count"
                ),
                "mean_wait_steps": _mean(group, "wait_steps"),
                "mean_steps": _mean(group, "steps"),
            }
        )
    return per_seed


def _aggregate_seed_summary(
    per_seed: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        groups[tuple(row[field] for field in group_fields)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        record = dict(zip(group_fields, key))
        record["seed_count"] = len(group)
        record["scenarios_per_seed"] = int(group[0]["scenario_count"])
        for metric in (
            "success_rate",
            "safe_success_rate",
            "dynamic_collision_rate",
            "mean_dynamic_collision_count",
            "mean_wait_steps",
            "mean_steps",
        ):
            values = [float(row[metric]) for row in group]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            half_width = 2.776 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_ci95_low"] = mean - half_width
            record[f"{metric}_ci95_high"] = mean + half_width
        result.append(record)
    return result


def _assign_rank_strata(rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    ordered = sorted(rows, key=lambda row: (float(row["conflict_score"]), int(row["scenario_id"])))
    result = {}
    for index, row in enumerate(ordered):
        fraction = index / max(1, len(ordered))
        stratum = "low" if fraction < 1 / 3 else "medium" if fraction < 2 / 3 else "high"
        result[int(row["scenario_id"])] = stratum
    return result


def _formal_test_conflict_analysis(
    audit_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_metrics: dict[tuple[str, int], Mapping[str, Any]] = {
        (str(row["map_key"]), int(row["scenario_id"])): row
        for row in audit_rows
        if row["split"] == "test"
    }
    strata = {
        map_key: _assign_rank_strata(
            [
                row
                for row in audit_rows
                if row["split"] == "test" and row["map_key"] == map_key
            ]
        )
        for map_key in MAP_CONFIGS
    }
    rows = []
    for spec in EVALUATIONS:
        config = load_config(_resolve(spec.config_path))
        output_root = _resolve(config["experiment"]["output_root"])
        prefix = str(
            config["experiment"].get(
                "run_name_prefix", "dynamic_spatial_generalization_map01"
            )
        )
        for seed in (int(value) for value in config["training"]["seeds"]):
            path = (
                output_root
                / f"{prefix}_seed_{seed}_{spec.strategy}"
                / "test_evaluation.csv"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing formal test evaluation: {path}")
            with path.open("r", encoding="utf-8", newline="") as handle:
                for source in csv.DictReader(handle):
                    scenario_id = int(float(source["scenario_id"]))
                    intrinsic = test_metrics[(spec.map_key, scenario_id)]
                    rows.append(
                        {
                            "map_key": spec.map_key,
                            "method": spec.method,
                            "training_seed": seed,
                            "scenario_id": scenario_id,
                            "conflict_stratum": strata[spec.map_key][scenario_id],
                            "conflict_score": intrinsic["conflict_score"],
                            "direct_intersection_route_count": intrinsic[
                                "direct_intersection_route_count"
                            ],
                            "exact_temporal_conflict_count": intrinsic[
                                "exact_temporal_conflict_count"
                            ],
                            "aligned_temporal_conflict_count": intrinsic[
                                "aligned_temporal_conflict_count"
                            ],
                            "success": float(source["success"]),
                            "safe_success": float(source["safe_success"]),
                            "dynamic_collision": float(source["dynamic_collision"]),
                            "dynamic_collision_count": float(
                                source["dynamic_collision_count"]
                            ),
                            "wait_steps": float(source["wait_steps"]),
                            "steps": float(source["steps"]),
                        }
                    )
    write_records_csv(rows, output_dir / "formal_test_by_conflict.csv")
    renamed = [
        {**row, "stratum": row["conflict_stratum"]}
        for row in rows
    ]
    per_seed = _seed_summary(renamed, ("map_key", "method", "stratum"))
    summary = _aggregate_seed_summary(
        per_seed, ("map_key", "method", "stratum")
    )
    write_records_csv(per_seed, output_dir / "formal_test_conflict_per_seed.csv")
    write_records_csv(summary, output_dir / "formal_test_conflict_summary.csv")
    return rows, summary


def _stress_summary(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    per_seed = _seed_summary(rows, ("map_key", "method", "stratum"))
    summary = _aggregate_seed_summary(
        per_seed, ("map_key", "method", "stratum")
    )
    write_records_csv(per_seed, output_dir / "stress_per_seed.csv")
    write_records_csv(summary, output_dir / "stress_summary.csv")
    baseline = {
        (str(row["map_key"]), str(row["stratum"]), int(row["training_seed"])): row
        for row in per_seed
        if row["method"] == "persistent25"
    }
    differences = []
    for row in per_seed:
        base = baseline[(str(row["map_key"]), str(row["stratum"]), int(row["training_seed"]))]
        differences.append(
            {
                "map_key": row["map_key"],
                "method": row["method"],
                "stratum": row["stratum"],
                "training_seed": row["training_seed"],
                "safe_success_difference_vs_persistent25": float(
                    row["safe_success_rate"]
                )
                - float(base["safe_success_rate"]),
                "dynamic_collision_difference_vs_persistent25": float(
                    row["dynamic_collision_rate"]
                )
                - float(base["dynamic_collision_rate"]),
            }
        )
    write_records_csv(differences, output_dir / "stress_paired_differences.csv")
    return summary


def _percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _write_report(
    manifest_summary: Sequence[Mapping[str, Any]],
    stress_metadata: Mapping[tuple[str, int], Mapping[str, Any]],
    formal_summary: Sequence[Mapping[str, Any]],
    stress_summary: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    lines = [
        "# Spatial conflict audit and supplemental stress evaluation",
        "",
        "This is a post-hoc supplemental analysis. It does not replace the frozen Map 1/Map 2 test results and must not be used to retroactively tune their checkpoint selection.",
        "",
        "## Frozen-manifest audit",
        "",
        "| Map | Split | Scenarios | No direct route | >=2 direct routes | Exact nominal-time conflict | Mean score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in manifest_summary:
        lines.append(
            f"| {row['map_key']} | {row['split']} | {row['scenario_count']} | "
            f"{_percentage(float(row['no_direct_intersection_rate']))} | "
            f"{_percentage(float(row['two_or_more_direct_routes_rate']))} | "
            f"{_percentage(float(row['exact_temporal_conflict_rate']))} | "
            f"{float(row['mean_conflict_score']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired supplemental stress-set intrinsic difficulty",
            "",
            "Low/medium/high use the same route combinations within each block and differ only in obstacle start phase/direction.",
            "",
            "| Map | Stratum | Scenarios | Mean direct routes | Exact-conflict rate | Mean exact conflicts | Mean score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    intrinsic_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in stress_metadata.values():
        intrinsic_groups[(str(row["map_key"]), str(row["stratum"]))].append(row)
    for (map_key, stratum), group in sorted(intrinsic_groups.items()):
        lines.append(
            f"| {map_key} | {stratum} | {len(group)} | "
            f"{_mean(group, 'direct_intersection_route_count'):.2f} | "
            f"{_percentage(sum(int(row['exact_temporal_conflict_count']) > 0 for row in group) / len(group))} | "
            f"{_mean(group, 'exact_temporal_conflict_count'):.2f} | "
            f"{_mean(group, 'conflict_score'):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Existing frozen-test outcomes by conflict-score tertile",
            "",
            "| Map | Method | Stratum | Safe success mean | Dynamic collision mean |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in formal_summary:
        lines.append(
            f"| {row['map_key']} | {row['method']} | {row['stratum']} | "
            f"{_percentage(float(row['safe_success_rate_mean']))} | "
            f"{_percentage(float(row['dynamic_collision_rate_mean']))} |"
        )
    lines.extend(
        [
            "",
            "## Selected-model paired stress results",
            "",
            "Means and 95% t intervals use five training-seed means; scenarios sharing a trained model are not treated as independent training replicates.",
            "",
            "| Map | Method | Stratum | Safe success mean ± SD | 95% CI | Dynamic collision | Mean contacts |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stress_summary:
        lines.append(
            f"| {row['map_key']} | {row['method']} | {row['stratum']} | "
            f"{_percentage(float(row['safe_success_rate_mean']))} ± "
            f"{_percentage(float(row['safe_success_rate_sd']))} | "
            f"[{_percentage(float(row['safe_success_rate_ci95_low']))}, "
            f"{_percentage(float(row['safe_success_rate_ci95_high']))}] | "
            f"{_percentage(float(row['dynamic_collision_rate_mean']))} | "
            f"{float(row['mean_dynamic_collision_count_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The stress set is deliberately constructed after the formal experiments and therefore supports mechanism diagnosis only. A conflict-aware adaptive replay method is justified only if the relative advantage of persistent25 systematically decreases from low to high conflict, preferably on more than one map. Otherwise adaptive replay should not be promoted as the next algorithmic contribution.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen spatial manifests and evaluate selected models under paired conflict stress."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/spatial_conflict_stress_v1",
    )
    parser.add_argument("--scenarios-per-stratum", type=int, default=20)
    parser.add_argument("--generation-seed", type=int, default=20260901)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.scenarios_per_stratum <= 0:
        raise ValueError("scenarios-per-stratum must be positive.")

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts, audit_rows, _ = _audit_manifests(output_dir)
    manifest_summary = _manifest_summary(audit_rows)
    write_records_csv(manifest_summary, output_dir / "manifest_split_summary.csv")
    _, formal_summary = _formal_test_conflict_analysis(audit_rows, output_dir)
    schedules, stress_metadata = _build_stress_scenarios(
        contexts,
        args.scenarios_per_stratum,
        args.generation_seed,
        output_dir,
    )
    if args.audit_only:
        _write_report(
            manifest_summary,
            stress_metadata,
            formal_summary,
            [],
            output_dir,
        )
        print(f"Audit saved to {output_dir}", flush=True)
        return
    evaluation_rows = _run_model_evaluations(
        contexts,
        schedules,
        stress_metadata,
        args.device,
        output_dir,
    )
    stress_summary = _stress_summary(evaluation_rows, output_dir)
    _write_report(
        manifest_summary,
        stress_metadata,
        formal_summary,
        stress_summary,
        output_dir,
    )
    print(f"Conflict stress analysis saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
