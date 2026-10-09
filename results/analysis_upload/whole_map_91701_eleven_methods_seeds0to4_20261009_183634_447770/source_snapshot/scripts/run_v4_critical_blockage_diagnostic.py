"""Evaluate frozen v4 models on controlled critical-path blockage states.

This is a post-hoc, evaluation-only diagnostic.  Each scenario contains one
phase-aligned obstacle that makes the next nominal A* action collide and four
off-path background obstacles.  No model is trained or selected by this script.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.core.astar import astar_path, randomized_tie_astar_path
from astar_d3qn.core.grid import ACTION_NAMES, Action, Position, action_between, in_bounds, move
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.evaluation.conflict import minimum_collision_free_steps, obstacle_positions
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything


CONFIGS = {
    "office": "configs/dynamic_spatial_generalization_office_balanced_v4.yaml",
    "parcel": "configs/dynamic_spatial_generalization_parcel_balanced_v4.yaml",
    "warehouse": "configs/dynamic_spatial_generalization_warehouse_balanced_v4.yaml",
}
STRATEGIES = ("uniform", "prefill", "persistent_demo")
REGIONS = ((0.10, 0.35, "early"), (0.35, 0.65, "middle"), (0.65, 0.90, "late"))


def _evaluation_specs(map_key: str) -> tuple[tuple[str, str, str], ...]:
    specs = [
        ("uniform", CONFIGS[map_key], "uniform"),
        ("prefill", CONFIGS[map_key], "prefill"),
        ("persistent_demo", CONFIGS[map_key], "persistent_demo"),
        (
            "ca_predictive",
            f"configs/dynamic_spatial_generalization_{map_key}_conflict_adaptive_v1.yaml",
            "conflict_adaptive_demo",
        ),
        (
            "ca_current",
            f"configs/dynamic_spatial_generalization_{map_key}_conflict_current_v1.yaml",
            "conflict_adaptive_demo",
        ),
    ]
    if map_key == "office":
        specs.append(
            (
                "local_conflict_demo",
                "configs/dynamic_office_local_conflict_demo_v1.yaml",
                "local_conflict_demo",
            )
        )
    return tuple(specs)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_context(config_name: str) -> dict[str, Any]:
    config = load_config(_resolve(config_name))
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    matches = [problem for problem in problems if problem.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one problem for {config_name}.")
    manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))
    return {"config": config, "problem": matches[0], "manifest": manifest}


def _valid_route(problem, route: Sequence[Position]) -> bool:
    return (
        len(route) == 5
        and len(set(route)) == 5
        and all(in_bounds(cell, problem.size) for cell in route)
        and not set(route).intersection(problem.obstacles)
        and problem.start not in route
        and problem.goal not in route
    )


def _crossing_candidates(problem) -> list[tuple[int, tuple[Position, ...]]]:
    path = problem.nominal_path
    candidates = []
    for index in range(1, len(path) - 1):
        center = path[index]
        previous, following = path[index - 1], path[index + 1]
        path_is_horizontal = previous[0] == center[0] == following[0]
        path_is_vertical = previous[1] == center[1] == following[1]
        orientations = (
            ("vertical",) if path_is_horizontal else
            ("horizontal",) if path_is_vertical else
            ("horizontal", "vertical")
        )
        for orientation in orientations:
            if orientation == "horizontal":
                route = tuple((center[0], center[1] + offset) for offset in range(-2, 3))
            else:
                route = tuple((center[0] + offset, center[1]) for offset in range(-2, 3))
            if _valid_route(problem, route) and set(route).intersection(path) == {center}:
                candidates.append((index, route))
    return candidates


def _path_frequencies(problem, samples: int, seed: int) -> Counter[Position]:
    rng = random.Random(seed)
    frequencies: Counter[Position] = Counter()
    for _ in range(samples):
        path = randomized_tie_astar_path(
            problem.start, problem.goal, problem.obstacles, problem.size, rng
        )
        if path is None:
            raise RuntimeError(f"Randomized A* failed for {problem.map_id}.")
        frequencies.update(path)
    return frequencies


def _select_key_routes(problem, *, samples: int, seed: int) -> list[dict[str, Any]]:
    frequencies = _path_frequencies(problem, samples, seed)
    candidates = []
    for path_index, route in _crossing_candidates(problem):
        cell = problem.nominal_path[path_index]
        blocked_path = astar_path(
            problem.start,
            problem.goal,
            set(problem.obstacles) | {cell},
            problem.size,
        )
        disconnected = blocked_path is None
        detour = (
            problem.size * problem.size
            if disconnected
            else len(blocked_path) - 1 - problem.astar_steps
        )
        candidates.append(
            {
                "path_index": path_index,
                "cell": cell,
                "route": route,
                "disconnected_if_blocked": disconnected,
                "static_block_detour_steps": detour,
                "astar_path_frequency": frequencies[cell] / samples,
            }
        )
    selected = []
    path_steps = problem.astar_steps
    for lower, upper, label in REGIONS:
        eligible = [
            row
            for row in candidates
            if math.ceil(lower * path_steps) <= row["path_index"] <= math.floor(upper * path_steps)
        ]
        if not eligible:
            raise RuntimeError(f"No {label} crossing route for {problem.map_id}.")
        eligible.sort(
            key=lambda row: (
                int(row["disconnected_if_blocked"]),
                int(row["static_block_detour_steps"]),
                float(row["astar_path_frequency"]),
                -abs(row["path_index"] - (lower + upper) * path_steps / 2),
            ),
            reverse=True,
        )
        selected.append({**eligible[0], "region": label})
    return selected


def _aligned_phases(
    route: tuple[Position, ...], target: Position, target_step: int, label: str
) -> list[DynamicObstacleSpec]:
    options: dict[Position, DynamicObstacleSpec] = {}
    for start_index in range(len(route)):
        for direction in (-1, 1):
            spec = DynamicObstacleSpec(
                route=route,
                start_index=start_index,
                direction=direction,
                move_every=1,
                label=label,
                reference_path_source="static_nominal_path_critical_probe",
                reference_path_index=target_step,
            )
            positions = obstacle_positions(spec, target_step + 1)
            if (
                positions[target_step - 1] == target
                and positions[target_step] != target
                and positions[target_step + 1] != target
            ):
                options.setdefault(positions[target_step], spec)
    if len(options) < 2:
        raise RuntimeError(f"Could not align two departure directions for {label}.")
    return [options[key] for key in sorted(options)[:2]]


def _background_specs(
    problem,
    manifest: Mapping[str, Any],
    target_route: Sequence[Position],
    scenario_offset: int,
) -> tuple[DynamicObstacleSpec, ...]:
    records = [
        record
        for record in manifest["route_pools"]["test"]["background"]
        if not set(tuple(cell) for cell in record["route"]).intersection(target_route)
    ]
    if len(records) < 4:
        raise RuntimeError(f"Not enough off-path background routes for {problem.map_id}.")
    result = []
    for index, record in enumerate(records[:4]):
        route = tuple(tuple(cell) for cell in record["route"])
        result.append(
            DynamicObstacleSpec(
                route=route,
                start_index=(scenario_offset + 2 * index) % len(route),
                direction=1 if (scenario_offset + index) % 2 == 0 else -1,
                move_every=1,
                label=str(record["route_id"]),
                reference_path_source="v4_test_background",
            )
        )
    return tuple(result)


def _verify_forced_probe(problem, scenario: DynamicScenario, target_index: int) -> None:
    env = DynamicGridNavigationEnv(problem, scenario.obstacles, scenario_id=scenario.seed)
    state = env.reset()
    del state
    for step in range(1, target_index):
        action = action_between(problem.nominal_path[step - 1], problem.nominal_path[step])
        result = env.step(int(action))
        if result.info["collision"] or env.position != problem.nominal_path[step]:
            raise RuntimeError(f"Forced probe collided before target in scenario {scenario.seed}.")
    target_action = action_between(
        problem.nominal_path[target_index - 1], problem.nominal_path[target_index]
    )
    result = env.step(int(target_action))
    if result.info["collision_type"] != "dynamic" or 0 not in result.info["dynamic_collision_indices"]:
        raise RuntimeError(f"Target A* action is not blocked in scenario {scenario.seed}.")


def _build_scenarios(
    contexts: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> tuple[dict[str, tuple[DynamicScenario, ...]], dict[tuple[str, int], dict[str, Any]]]:
    schedules = {}
    metadata = {}
    serialized: dict[str, Any] = {
        "format_version": 1,
        "purpose": "post_hoc_v4_critical_path_blockage_diagnostic",
        "models_are_not_retrained": True,
        "maps": {},
    }
    for map_offset, (map_key, context) in enumerate(contexts.items()):
        problem, manifest = context["problem"], context["manifest"]
        keys = _select_key_routes(problem, samples=500, seed=20260930 + map_offset)
        scenarios = []
        map_records = []
        for key_index, key in enumerate(keys):
            phases = _aligned_phases(
                key["route"], key["cell"], key["path_index"],
                f"critical_{key['region']}",
            )
            for direction_index, target_spec in enumerate(phases):
                scenario_id = 50_000 + map_offset * 100 + key_index * 10 + direction_index
                backgrounds = _background_specs(
                    problem, manifest, target_spec.route, key_index * 2 + direction_index
                )
                scenario = DynamicScenario(scenario_id, (target_spec, *backgrounds))
                _verify_forced_probe(problem, scenario, key["path_index"])
                safe_steps = minimum_collision_free_steps(problem, scenario)
                if safe_steps is None:
                    raise RuntimeError(f"Scenario {scenario_id} has no safe path.")
                record = {
                    "map_key": map_key,
                    "map_id": problem.map_id,
                    "scenario_id": scenario_id,
                    "region": key["region"],
                    "direction_variant": direction_index,
                    "target_path_index": key["path_index"],
                    "target_cell": list(key["cell"]),
                    "static_block_detour_steps": key["static_block_detour_steps"],
                    "disconnected_if_blocked": key["disconnected_if_blocked"],
                    "astar_path_frequency": key["astar_path_frequency"],
                    "minimum_safe_path_steps": safe_steps,
                    "safe_detour_steps": safe_steps - problem.astar_steps,
                    "obstacles": [
                        {
                            "label": spec.label,
                            "route": [list(cell) for cell in spec.route],
                            "start_index": spec.start_index,
                            "direction": spec.direction,
                            "move_every": spec.move_every,
                        }
                        for spec in scenario.obstacles
                    ],
                }
                scenarios.append(scenario)
                map_records.append(record)
                metadata[(map_key, scenario_id)] = record
        schedules[map_key] = tuple(scenarios)
        serialized["maps"][map_key] = {
            "map_id": problem.map_id,
            "scenarios": map_records,
        }
    write_json(serialized, output_dir / "critical_blockage_scenarios.json")
    write_records_csv(
        [
            {key: value for key, value in row.items() if key != "obstacles"}
            for row in metadata.values()
        ],
        output_dir / "critical_blockage_scenario_audit.csv",
    )
    return schedules, metadata


def _agent(config: Mapping[str, Any], seed: int, device: str) -> D3QNAgent:
    environment, values = config["environment"], config["agent"]
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


def _q_values(agent: D3QNAgent, state) -> list[float]:
    spatial = torch.as_tensor(state.spatial, dtype=torch.float32, device=agent.device).unsqueeze(0)
    scalars = torch.as_tensor(state.scalars, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        values = agent.policy_network(spatial, scalars).squeeze(0)
    return [float(value) for value in values.detach().cpu().tolist()]


def _action_risks(env: DynamicGridNavigationEnv) -> list[bool]:
    old_dynamic = set(env.dynamic_positions)
    proposed_dynamic = set(env._next_dynamic_state()[3])
    risks = []
    for action in Action:
        candidate = move(env.position, action)
        risks.append(
            not in_bounds(candidate, env.problem.size)
            or candidate in env.problem.obstacles
            or candidate in old_dynamic
            or candidate in proposed_dynamic
        )
    return risks


def _probe_model(agent, problem, scenario, meta, config) -> dict[str, Any]:
    environment = config["environment"]
    env = DynamicGridNavigationEnv(
        problem,
        scenario.obstacles,
        scenario_id=scenario.seed,
        max_steps=int(environment["max_steps"]),
        window_size=int(environment["window_size"]),
    )
    state = env.reset()
    target_index = int(meta["target_path_index"])
    for step in range(1, target_index):
        action = action_between(problem.nominal_path[step - 1], problem.nominal_path[step])
        result = env.step(int(action))
        if result.info["collision"]:
            raise RuntimeError("Unexpected pre-target collision during policy probe.")
        state = result.observation
    demo_action = int(
        action_between(problem.nominal_path[target_index - 1], problem.nominal_path[target_index])
    )
    risks = _action_risks(env)
    if not risks[demo_action]:
        raise RuntimeError("Controlled A* action is unexpectedly safe.")
    values = _q_values(agent, state)
    selected = agent.select_action(state, epsilon=0.0)
    safe_actions = [index for index, risky in enumerate(risks) if not risky]
    best_safe = max(safe_actions, key=lambda index: values[index])
    return {
        "probe_selected_action": selected,
        "probe_selected_action_name": ACTION_NAMES[selected],
        "probe_demo_action": demo_action,
        "probe_demo_action_name": ACTION_NAMES[demo_action],
        "probe_selected_demo_action": float(selected == demo_action),
        "probe_selected_risky_action": float(risks[selected]),
        "probe_selected_safe_avoidance": float(not risks[selected]),
        "probe_q_demo": values[demo_action],
        "probe_q_best_safe": values[best_safe],
        "probe_safe_q_margin": values[best_safe] - values[demo_action],
    }


def _autonomous_rollout(agent, problem, scenario, meta, config) -> dict[str, Any]:
    environment = config["environment"]
    reward = RewardConfig(**{key: float(value) for key, value in config["reward"].items()})
    env = DynamicGridNavigationEnv(
        problem,
        scenario.obstacles,
        scenario_id=scenario.seed,
        max_steps=int(environment["max_steps"]),
        reward_config=reward,
        terminate_on_collision=bool(environment["terminate_on_collision"]),
        window_size=int(environment["window_size"]),
    )
    state = env.reset()
    target_index = int(meta["target_path_index"])
    pre_target = problem.nominal_path[target_index - 1]
    demo_action = int(action_between(pre_target, problem.nominal_path[target_index]))
    target_opportunity = False
    target_demo_selected = False
    collision_count = 0
    target_collision_count = 0
    wait_steps = 0
    while True:
        action = agent.select_action(state, epsilon=0.0)
        if env.position == pre_target and _action_risks(env)[demo_action]:
            target_opportunity = True
            target_demo_selected = target_demo_selected or action == demo_action
        wait_steps += int(action == int(Action.STAY))
        result = env.step(action)
        collision_count += int(bool(result.info["collision"]))
        target_collision_count += int(0 in result.info.get("dynamic_collision_indices", ()))
        state = result.observation
        if result.done:
            break
    success = bool(result.info["reached"])
    return {
        "success": float(success),
        "safe_success": float(success and collision_count == 0),
        "dynamic_collision": float(collision_count > 0),
        "dynamic_collision_count": collision_count,
        "target_collision": float(target_collision_count > 0),
        "target_collision_count": target_collision_count,
        "target_opportunity_reached": float(target_opportunity),
        "target_demo_action_selected": float(target_demo_selected),
        "wait_steps": wait_steps,
        "steps": env.steps,
    }


def _evaluate(
    contexts: Mapping[str, Mapping[str, Any]],
    schedules: Mapping[str, Sequence[DynamicScenario]],
    metadata: Mapping[tuple[str, int], Mapping[str, Any]],
    output_dir: Path,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probe_rows, rollout_rows = [], []
    for map_key, context in contexts.items():
        problem = context["problem"]
        for method, config_name, strategy in _evaluation_specs(map_key):
            config = load_config(_resolve(config_name))
            root = _resolve(config["experiment"]["output_root"])
            prefix = config["experiment"]["run_name_prefix"]
            for seed in (int(value) for value in config["training"]["seeds"]):
                seed_everything(seed)
                model_path = root / f"{prefix}_seed_{seed}_{strategy}" / "model_selected.pth"
                if not model_path.exists():
                    raise FileNotFoundError(f"Missing selected model {model_path}.")
                agent = _agent(config, seed, device)
                agent.load_weights(model_path)
                for scenario in schedules[map_key]:
                    meta = metadata[(map_key, scenario.seed)]
                    common = {
                        "map_key": map_key,
                        "map_id": problem.map_id,
                        "strategy": method,
                        "training_seed": seed,
                        "scenario_id": scenario.seed,
                        "region": meta["region"],
                        "direction_variant": meta["direction_variant"],
                    }
                    probe_rows.append(
                        {**common, **_probe_model(agent, problem, scenario, meta, config)}
                    )
                    rollout_rows.append(
                        {**common, **_autonomous_rollout(agent, problem, scenario, meta, config)}
                    )
                selected_demo = sum(row["probe_selected_demo_action"] for row in probe_rows if row["map_key"] == map_key and row["strategy"] == method and row["training_seed"] == seed)
                print(
                    f"[{map_key} {method} seed={seed}] "
                    f"probe_A*= {selected_demo:.0f}/{len(schedules[map_key])}",
                    flush=True,
                )
    write_records_csv(probe_rows, output_dir / "critical_state_action_probes.csv")
    write_records_csv(rollout_rows, output_dir / "critical_blockage_rollouts.csv")
    return probe_rows, rollout_rows


def _summaries(
    probe_rows: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    joined = {
        (row["map_key"], row["strategy"], row["training_seed"], row["scenario_id"]): row
        for row in rollout_rows
    }
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for probe in probe_rows:
        key = (probe["map_key"], probe["strategy"], probe["training_seed"], probe["scenario_id"])
        groups[(str(probe["map_key"]), str(probe["strategy"]), int(probe["training_seed"]))].append(
            {**probe, **joined[key]}
        )
    metrics = (
        "probe_selected_demo_action",
        "probe_selected_safe_avoidance",
        "probe_safe_q_margin",
        "success",
        "safe_success",
        "dynamic_collision",
        "target_collision",
        "target_opportunity_reached",
        "target_demo_action_selected",
        "wait_steps",
        "steps",
    )
    per_seed = []
    for (map_key, strategy, seed), rows in sorted(groups.items()):
        per_seed.append(
            {
                "map_key": map_key,
                "strategy": strategy,
                "training_seed": seed,
                "scenario_count": len(rows),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in rows)
                    for metric in metrics
                },
            }
        )
    write_records_csv(per_seed, output_dir / "critical_blockage_per_seed.csv")
    grouped_seed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped_seed[(str(row["map_key"]), str(row["strategy"]))].append(row)
    summary = []
    for (map_key, strategy), rows in sorted(grouped_seed.items()):
        record: dict[str, Any] = {
            "map_key": map_key,
            "strategy": strategy,
            "seed_count": len(rows),
            "scenarios_per_seed": rows[0]["scenario_count"],
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    write_records_csv(summary, output_dir / "critical_blockage_summary.csv")
    return summary


def _render_previews(contexts, schedules, metadata, output_dir: Path) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for map_key, scenarios in schedules.items():
        problem = contexts[map_key]["problem"]
        figure, axes = plt.subplots(2, 3, figsize=(13, 9), squeeze=False)
        for axis, scenario in zip(axes.flat, scenarios):
            grid = [[0] * problem.size for _ in range(problem.size)]
            for row, column in problem.obstacles:
                grid[row][column] = 1
            axis.imshow(grid, cmap="binary", origin="upper", vmin=0, vmax=1)
            path_rows = [cell[0] for cell in problem.nominal_path]
            path_cols = [cell[1] for cell in problem.nominal_path]
            axis.plot(path_cols, path_rows, "--", color="#64b5f6", linewidth=1.2)
            for index, spec in enumerate(scenario.obstacles):
                route_rows = [cell[0] for cell in spec.route]
                route_cols = [cell[1] for cell in spec.route]
                color = "#ff7f0e" if index == 0 else "#009688"
                axis.plot(route_cols, route_rows, ":", color=color, linewidth=2)
                initial = spec.route[spec.start_index]
                axis.scatter(initial[1], initial[0], color="#d32f2f", s=25, zorder=4)
            meta = metadata[(map_key, scenario.seed)]
            target = meta["target_cell"]
            axis.scatter(target[1], target[0], marker="X", color="#ff0000", s=75, zorder=5)
            axis.set_title(f"{meta['region']} / direction {meta['direction_variant']}")
            axis.set_xlim(-0.5, problem.size - 0.5)
            axis.set_ylim(problem.size - 0.5, -0.5)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"{problem.map_id}: controlled critical blockage probes")
        figure.tight_layout()
        figure.savefig(preview_dir / f"{map_key}_critical_blockage.png", dpi=180)
        plt.close(figure)


def _write_report(summary: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    lines = [
        "# V4 critical blockage diagnostic",
        "",
        "Post-hoc evaluation only: the frozen v4 selected models were not retrained or reselected.",
        "Each map has six scenarios (three path regions, two obstacle departure directions), with one exact A* blocker and four off-path background obstacles.",
        "",
        "| Map | Strategy | Probe chooses blocked A* | Probe chooses safe action | Autonomous safe success | Target collision |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['map_key']} | {row['strategy']} | "
            f"{100 * float(row['probe_selected_demo_action_mean']):.1f}% | "
            f"{100 * float(row['probe_selected_safe_avoidance_mean']):.1f}% | "
            f"{100 * float(row['safe_success_mean']):.1f}% | "
            f"{100 * float(row['target_collision_mean']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "The forced-state probe is the primary mechanism measure. Autonomous results also depend on whether a learned policy reaches the registered nominal pre-blockage cell.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v4_critical_blockage_diagnostic")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--maps",
        nargs="+",
        choices=tuple(CONFIGS),
        default=tuple(CONFIGS),
        help="Evaluate only the named maps; use '--maps office' for the focused study.",
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts = {key: _load_context(CONFIGS[key]) for key in args.maps}
    schedules, metadata = _build_scenarios(contexts, output_dir)
    _render_previews(contexts, schedules, metadata, output_dir)
    probe_rows, rollout_rows = _evaluate(
        contexts, schedules, metadata, output_dir, args.device
    )
    summary = _summaries(probe_rows, rollout_rows, output_dir)
    _write_report(summary, output_dir)
    print(f"Wrote v4 critical blockage diagnostic to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
