"""Measure nominal A* action agreement specifically at imminent conflict states."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.core.grid import ACTION_DELTAS
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import validate_spatial_scenario_manifest
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything

from run_spatial_conflict_stress import EVALUATIONS, MAP_CONFIGS, _resolve


def _load_problem(config: Mapping[str, Any]):
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    map_id = str(config["map"]["scene"])
    matches = [problem for problem in problems if problem.map_id == map_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered problem named {map_id!r}.")
    return matches[0]


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


def _stress_scenarios(
    map_key: str,
    problem,
    manifest: Mapping[str, Any],
    stress_manifest: Mapping[str, Any],
) -> tuple[tuple[DynamicScenario, str, int], ...]:
    lookup = {
        str(record["route_id"]): record
        for category in ("corridor", "background")
        for record in manifest["route_pools"]["test"][category]
    }
    result = []
    for stratum in ("low", "medium", "high"):
        for source in stress_manifest["maps"][map_key]["strata"][stratum]:
            obstacles = []
            for obstacle in source["obstacles"]:
                record = lookup[str(obstacle["route_id"])]
                obstacles.append(
                    DynamicObstacleSpec(
                        route=tuple(tuple(cell) for cell in record["route"]),
                        start_index=int(obstacle["start_index"]),
                        direction=int(obstacle["direction"]),
                        move_every=int(obstacle["move_every"]),
                        label=str(obstacle["route_id"]),
                        reference_path_source=str(record["category"]),
                    )
                )
            scenario = DynamicScenario(
                seed=int(source["scenario_id"]),
                obstacles=tuple(obstacles),
            )
            # Construction validation is delegated to the real environment.
            DynamicGridNavigationEnv(problem, scenario.obstacles)
            result.append((scenario, stratum, int(source["block_id"])))
    return tuple(result)


def _nominal_action_lookup(problem) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    result = {}
    for current, following in zip(problem.nominal_path, problem.nominal_path[1:]):
        delta = following[0] - current[0], following[1] - current[1]
        result[current] = ACTION_DELTAS.index(delta), following
    return result


def _evaluate_one(
    agent: D3QNAgent,
    problem,
    scenario: DynamicScenario,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    environment = config["environment"]
    reward = RewardConfig(
        **{key: float(value) for key, value in config["reward"].items()}
    )
    env = DynamicGridNavigationEnv(
        problem,
        dynamic_obstacles=scenario.obstacles,
        scenario_id=scenario.seed,
        max_steps=int(environment["max_steps"]),
        reward_config=reward,
        terminate_on_collision=bool(environment["terminate_on_collision"]),
        window_size=int(environment["window_size"]),
    )
    state = env.reset()
    nominal = _nominal_action_lookup(problem)
    nominal_state_count = 0
    nominal_agreement_count = 0
    imminent_conflict_state_count = 0
    conflict_agreement_count = 0
    conflict_avoidance_count = 0
    collision_after_conflict_agreement = 0
    collision_after_conflict_avoidance = 0
    dynamic_collisions = 0
    while True:
        action = agent.select_action(state, epsilon=0.0)
        conflict_state = False
        nominal_action = None
        if env.position in nominal:
            nominal_state_count += 1
            nominal_action, nominal_candidate = nominal[env.position]
            if action == nominal_action:
                nominal_agreement_count += 1
            _, _, _, proposed_dynamic = env._next_dynamic_state()
            conflict_state = nominal_candidate in env.dynamic_positions or nominal_candidate in proposed_dynamic
            if conflict_state:
                imminent_conflict_state_count += 1
                if action == nominal_action:
                    conflict_agreement_count += 1
                else:
                    conflict_avoidance_count += 1
        result = env.step(action)
        state = result.observation
        dynamic_collision = result.info.get("collision_type") == "dynamic"
        if dynamic_collision:
            dynamic_collisions += 1
        if conflict_state:
            if action == nominal_action and dynamic_collision:
                collision_after_conflict_agreement += 1
            elif action != nominal_action and dynamic_collision:
                collision_after_conflict_avoidance += 1
        if result.done:
            break
    return {
        "scenario_id": scenario.seed,
        "success": float(bool(result.info["reached"])),
        "steps": env.steps,
        "dynamic_collision_count": dynamic_collisions,
        "nominal_state_count": nominal_state_count,
        "nominal_action_agreement_count": nominal_agreement_count,
        "nominal_action_agreement_rate": (
            nominal_agreement_count / nominal_state_count if nominal_state_count else None
        ),
        "imminent_demo_conflict_state_count": imminent_conflict_state_count,
        "demo_action_agreement_in_conflict_count": conflict_agreement_count,
        "demo_action_avoidance_in_conflict_count": conflict_avoidance_count,
        "demo_action_agreement_in_conflict_rate": (
            conflict_agreement_count / imminent_conflict_state_count
            if imminent_conflict_state_count
            else None
        ),
        "collision_after_demo_agreement_count": collision_after_conflict_agreement,
        "collision_after_demo_avoidance_count": collision_after_conflict_avoidance,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[
            (
                str(row["map_key"]),
                str(row["method"]),
                str(row["stratum"]),
                int(row["training_seed"]),
            )
        ].append(row)
    per_seed = []
    for key, group in sorted(by_seed.items()):
        conflict_states = sum(int(row["imminent_demo_conflict_state_count"]) for row in group)
        agreements = sum(int(row["demo_action_agreement_in_conflict_count"]) for row in group)
        per_seed.append(
            {
                "map_key": key[0],
                "method": key[1],
                "stratum": key[2],
                "training_seed": key[3],
                "scenario_count": len(group),
                "scenarios_with_conflict_state": sum(
                    int(row["imminent_demo_conflict_state_count"]) > 0 for row in group
                ),
                "imminent_demo_conflict_state_count": conflict_states,
                "demo_action_agreement_in_conflict_rate": (
                    agreements / conflict_states if conflict_states else None
                ),
                "collision_after_demo_agreement_rate": (
                    sum(int(row["collision_after_demo_agreement_count"]) for row in group)
                    / max(1, agreements)
                ),
            }
        )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        groups[(row["map_key"], row["method"], row["stratum"])].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        rates = [
            float(row["demo_action_agreement_in_conflict_rate"])
            for row in group
            if row["demo_action_agreement_in_conflict_rate"] is not None
        ]
        mean = statistics.fmean(rates) if rates else math.nan
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        summary.append(
            {
                "map_key": key[0],
                "method": key[1],
                "stratum": key[2],
                "seed_count_with_conflict_states": len(rates),
                "mean_scenarios_with_conflict_state": statistics.fmean(
                    float(row["scenarios_with_conflict_state"]) for row in group
                ),
                "mean_conflict_state_count": statistics.fmean(
                    float(row["imminent_demo_conflict_state_count"]) for row in group
                ),
                "demo_action_agreement_in_conflict_rate_mean": mean,
                "demo_action_agreement_in_conflict_rate_sd": sd,
                "collision_after_demo_agreement_rate_mean": statistics.fmean(
                    float(row["collision_after_demo_agreement_rate"]) for row in group
                ),
            }
        )
    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate static-demo action agreement at imminent dynamic conflicts."
    )
    parser.add_argument(
        "--analysis-dir", default="outputs/spatial_conflict_stress_v1"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    analysis_dir = _resolve(args.analysis_dir)
    stress_manifest = load_json(analysis_dir / "stress_scenarios.json")
    contexts = {}
    schedules = {}
    for map_key, config_name in MAP_CONFIGS.items():
        config = load_config(_resolve(config_name))
        problem = _load_problem(config)
        manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))
        validate_spatial_scenario_manifest(problem, manifest)
        contexts[map_key] = (config, problem, manifest)
        schedules[map_key] = _stress_scenarios(
            map_key, problem, manifest, stress_manifest
        )

    rows = []
    for spec in EVALUATIONS:
        config = load_config(_resolve(spec.config_path))
        _, problem, _ = contexts[spec.map_key]
        output_root = _resolve(config["experiment"]["output_root"])
        prefix = str(
            config["experiment"].get(
                "run_name_prefix", "dynamic_spatial_generalization_map01"
            )
        )
        for seed in (int(value) for value in config["training"]["seeds"]):
            seed_everything(seed)
            agent = _agent(config, seed, args.device)
            model_path = (
                output_root
                / f"{prefix}_seed_{seed}_{spec.strategy}"
                / "model_selected.pth"
            )
            agent.load_weights(model_path)
            seed_rows = []
            for scenario, stratum, block_id in schedules[spec.map_key]:
                row = _evaluate_one(agent, problem, scenario, config)
                seed_rows.append(
                    {
                        "map_key": spec.map_key,
                        "method": spec.method,
                        "training_seed": seed,
                        "stratum": stratum,
                        "block_id": block_id,
                        **row,
                    }
                )
            rows.extend(seed_rows)
            conflict_states = sum(
                int(row["imminent_demo_conflict_state_count"]) for row in seed_rows
            )
            agreements = sum(
                int(row["demo_action_agreement_in_conflict_count"]) for row in seed_rows
            )
            print(
                f"[{spec.map_key} {spec.method} seed={seed}] "
                f"conflict_states={conflict_states} "
                f"demo_agreement={agreements / max(1, conflict_states):.1%}",
                flush=True,
            )
    per_seed, summary = _summarize(rows)
    write_records_csv(rows, analysis_dir / "astar_conflict_agreement.csv")
    write_records_csv(
        per_seed, analysis_dir / "astar_conflict_agreement_per_seed.csv"
    )
    write_records_csv(
        summary, analysis_dir / "astar_conflict_agreement_summary.csv"
    )
    print(f"A* conflict agreement saved to {analysis_dir}", flush=True)


if __name__ == "__main__":
    main()
