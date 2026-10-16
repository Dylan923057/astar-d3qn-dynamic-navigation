"""Evaluate trained policies in a near-blockage, single-obstacle diagnostic.

Each policy starts four nominal-path steps before a phase-aligned blocker.  A
matched no-obstacle rollout uses the identical local start.  This removes the
old diagnostic's off-path filler obstacles and prevents a policy from choosing
an unrelated global route long before it can observe the blockage.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.core.grid import ACTION_NAMES, Action, action_between
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything
from run_office_controlled_behavior_study import MAIN_CONFIG, METHODS
from run_v4_critical_blockage_diagnostic import _aligned_phases, _select_key_routes


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_problem():
    config = load_config(_resolve(MAIN_CONFIG))
    source = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(source["file"]))
    matches = [problem for problem in problems if problem.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one registered Office problem.")
    return matches[0]


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
    spatial = torch.as_tensor(
        state.spatial, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    scalars = torch.as_tensor(
        state.scalars, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        values = agent.policy_network(spatial, scalars).squeeze(0)
    return [float(value) for value in values.detach().cpu().tolist()]


def _valid_actions(env: DynamicGridNavigationEnv) -> list[int]:
    return [
        index
        for index, valid in enumerate(env.action_mask(mask_collisions=True))
        if bool(valid)
    ]


def _dynamic_risks(env: DynamicGridNavigationEnv) -> list[bool]:
    static_valid = set(_valid_actions(env))
    return [
        index not in static_valid
        or env.dynamic_action_collision_risk(index, predict_next=True)
        for index in range(env.action_dim)
    ]


def _build_scenarios(lead_steps: int, output_dir: Path):
    if not 3 <= lead_steps <= 5:
        raise ValueError("lead_steps must be between three and five.")
    problem = _load_problem()
    keys = _select_key_routes(problem, samples=500, seed=20261001)
    scenarios = []
    serialized = {
        "format_version": 1,
        "purpose": "near_blockage_single_obstacle_causal_diagnostic",
        "lead_steps": lead_steps,
        "matched_no_obstacle_control": True,
        "off_path_filler_obstacles": 0,
        "scenarios": [],
    }
    for key_index, key in enumerate(keys):
        absolute_target_index = int(key["path_index"])
        start_index = absolute_target_index - lead_steps
        if start_index < 0:
            raise RuntimeError("Selected diagnostic target is too close to map start.")
        path = tuple(problem.nominal_path[start_index:])
        diagnostic_problem = replace(
            problem,
            map_id=f"{problem.map_id}_near_{key['region']}",
            start=path[0],
            nominal_path=path,
        )
        phases = _aligned_phases(
            tuple(key["route"]),
            tuple(key["cell"]),
            lead_steps,
            f"near_blocker_{key['region']}",
        )
        for direction_index, spec in enumerate(phases):
            scenario_id = 60_000 + key_index * 10 + direction_index
            scenario = DynamicScenario(scenario_id, (spec,))
            env = DynamicGridNavigationEnv(
                diagnostic_problem,
                scenario.obstacles,
                scenario_id=scenario_id,
            )
            env.reset()
            for step in range(1, lead_steps):
                action = action_between(path[step - 1], path[step])
                result = env.step(int(action))
                if result.info["collision"] or env.position != path[step]:
                    raise RuntimeError(
                        f"Scenario {scenario_id} collides before the registered target."
                    )
            blocked_action = int(action_between(path[lead_steps - 1], path[lead_steps]))
            if not env.dynamic_action_collision_risk(blocked_action, predict_next=True):
                raise RuntimeError(
                    f"Scenario {scenario_id} does not block the registered A* action."
                )
            target_result = env.step(blocked_action)
            if target_result.info["collision_type"] != "dynamic":
                raise RuntimeError(
                    f"Scenario {scenario_id} failed the exact collision witness."
                )
            metadata = {
                "scenario_id": scenario_id,
                "region": str(key["region"]),
                "direction_variant": direction_index,
                "lead_steps": lead_steps,
                "start": list(diagnostic_problem.start),
                "goal": list(diagnostic_problem.goal),
                "target_cell": list(key["cell"]),
                "pre_target_cell": list(path[lead_steps - 1]),
                "blocked_action": blocked_action,
                "blocked_action_name": ACTION_NAMES[blocked_action],
                "absolute_target_path_index": absolute_target_index,
                "obstacle": {
                    "route": [list(cell) for cell in spec.route],
                    "start_index": spec.start_index,
                    "direction": spec.direction,
                    "move_every": spec.move_every,
                },
            }
            scenarios.append((diagnostic_problem, scenario, metadata))
            serialized["scenarios"].append(metadata)
    write_json(serialized, output_dir / "controlled_blockage_scenarios.json")
    return scenarios


def _forced_probe(agent, problem, scenario, metadata, config) -> dict[str, Any]:
    env = DynamicGridNavigationEnv(
        problem,
        scenario.obstacles,
        scenario_id=scenario.seed,
        max_steps=int(config["environment"]["max_steps"]),
        window_size=int(config["environment"]["window_size"]),
    )
    state = env.reset()
    lead_steps = int(metadata["lead_steps"])
    path = problem.nominal_path
    for step in range(1, lead_steps):
        action = int(action_between(path[step - 1], path[step]))
        result = env.step(action)
        if result.info["collision"]:
            raise RuntimeError("Unexpected pre-target collision in forced probe.")
        state = result.observation
    blocked_action = int(metadata["blocked_action"])
    risks = _dynamic_risks(env)
    if not risks[blocked_action]:
        raise RuntimeError("Forced probe A* action is unexpectedly safe.")
    valid_actions = _valid_actions(env)
    selected = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
    values = _q_values(agent, state)
    safe_actions = [index for index in valid_actions if not risks[index]]
    if not safe_actions:
        raise RuntimeError("Forced probe contains no safe action.")
    best_safe = max(safe_actions, key=lambda index: values[index])
    return {
        "probe_selected_action": selected,
        "probe_selected_action_name": ACTION_NAMES[selected],
        "probe_blocked_astar_action": blocked_action,
        "probe_blocked_astar_action_name": ACTION_NAMES[blocked_action],
        "probe_selected_blocked_astar": float(selected == blocked_action),
        "probe_selected_safe_action": float(not risks[selected]),
        "probe_selected_wait": float(selected == int(Action.STAY)),
        "probe_q_blocked_astar": values[blocked_action],
        "probe_q_best_safe": values[best_safe],
        "probe_safe_q_margin": values[best_safe] - values[blocked_action],
    }


def _rollout(agent, problem, scenario, metadata, config, condition: str):
    obstacle_specs = scenario.obstacles if condition == "blocked" else ()
    reward = RewardConfig(
        **{key: float(value) for key, value in config["reward"].items()}
    )
    env = DynamicGridNavigationEnv(
        problem,
        obstacle_specs,
        scenario_id=scenario.seed,
        max_steps=int(config["environment"]["max_steps"]),
        reward_config=reward,
        terminate_on_collision=bool(config["environment"]["terminate_on_collision"]),
        window_size=int(config["environment"]["window_size"]),
    )
    state = env.reset()
    pre_target = tuple(metadata["pre_target_cell"])
    blocked_action = int(metadata["blocked_action"])
    collision_count = 0
    dynamic_collision_count = 0
    target_collision_count = 0
    wait_steps = 0
    wait_at_decision = False
    target_opportunity = False
    path = [env.position]
    while True:
        valid_actions = _valid_actions(env)
        action = agent.select_action(
            state, epsilon=0.0, valid_actions=valid_actions
        )
        if condition == "blocked" and env.position == pre_target:
            if env.dynamic_action_collision_risk(blocked_action, predict_next=True):
                target_opportunity = True
                wait_at_decision = wait_at_decision or action == int(Action.STAY)
        wait_steps += int(action == int(Action.STAY))
        result = env.step(action)
        collision_count += int(bool(result.info["collision"]))
        dynamic_collision_count += int(result.info["collision_type"] == "dynamic")
        target_collision_count += int(
            tuple(result.info.get("collision_position") or ())
            == tuple(metadata["target_cell"])
            and result.info["collision_type"] == "dynamic"
        )
        state = result.observation
        path.append(env.position)
        if result.done:
            break
    success = bool(result.info["reached"])
    row = {
        "condition": condition,
        "success": float(success),
        "safe_success": float(success and collision_count == 0),
        "collision": float(collision_count > 0),
        "collision_count": collision_count,
        "dynamic_collision": float(dynamic_collision_count > 0),
        "dynamic_collision_count": dynamic_collision_count,
        "target_collision": float(target_collision_count > 0),
        "target_collision_count": target_collision_count,
        "target_opportunity_reached": float(target_opportunity),
        "wait_at_blockage_decision": float(wait_at_decision),
        "wait_steps": wait_steps,
        "steps": env.steps,
    }
    return row, path


def _run_evaluation(scenarios, device: str, output_dir: Path):
    base_config = load_config(_resolve(MAIN_CONFIG))
    seeds = tuple(int(value) for value in base_config["training"]["seeds"])
    probe_rows = []
    rollout_rows = []
    trajectories = []
    for method in METHODS:
        config = load_config(_resolve(method.config_path))
        root = _resolve(config["experiment"]["output_root"])
        prefix = str(config["experiment"]["run_name_prefix"])
        for seed in seeds:
            seed_everything(seed)
            model_path = (
                root
                / f"{prefix}_seed_{seed}_{method.strategy}"
                / "model_selected.pth"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Missing selected model {model_path}.")
            agent = _agent(config, seed, device)
            agent.load_weights(model_path)
            for problem, scenario, metadata in scenarios:
                common = {
                    "method": method.key,
                    "method_label": method.label,
                    "training_seed": seed,
                    "scenario_id": scenario.seed,
                    "region": metadata["region"],
                    "direction_variant": metadata["direction_variant"],
                }
                probe_rows.append(
                    {
                        **common,
                        **_forced_probe(agent, problem, scenario, metadata, config),
                    }
                )
                for condition in ("blocked", "control"):
                    rollout, path = _rollout(
                        agent, problem, scenario, metadata, config, condition
                    )
                    rollout_rows.append({**common, **rollout})
                    trajectories.append(
                        {**common, "condition": condition, "path": [list(cell) for cell in path]}
                    )
            print(f"[{method.label} seed={seed}] diagnostic complete", flush=True)
    write_records_csv(probe_rows, output_dir / "forced_state_probes.csv")
    write_records_csv(rollout_rows, output_dir / "near_blockage_rollouts.csv")
    write_json(trajectories, output_dir / "near_blockage_trajectories.json")
    return probe_rows, rollout_rows, seeds


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    mean = fmean(values)
    return (
        mean,
        1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
    )


def _summarize(probe_rows, rollout_rows, seeds, output_dir: Path):
    metrics = (
        "probe_selected_blocked_astar",
        "probe_selected_safe_action",
        "probe_selected_wait",
        "probe_safe_q_margin",
        "blocked_safe_success",
        "control_safe_success",
        "blocker_safe_success_drop",
        "blocked_target_collision",
        "blocked_wait_at_decision",
        "blocked_wait_steps",
    )
    probe_group = defaultdict(list)
    rollout_group = defaultdict(list)
    for row in probe_rows:
        probe_group[(row["method"], int(row["training_seed"]))].append(row)
    for row in rollout_rows:
        rollout_group[
            (row["method"], int(row["training_seed"]), row["condition"])
        ].append(row)
    per_seed = []
    for method in METHODS:
        for seed in seeds:
            probes = probe_group[(method.key, seed)]
            blocked = rollout_group[(method.key, seed, "blocked")]
            control = rollout_group[(method.key, seed, "control")]
            blocked_safe = fmean(float(row["safe_success"]) for row in blocked)
            control_safe = fmean(float(row["safe_success"]) for row in control)
            per_seed.append(
                {
                    "method": method.key,
                    "method_label": method.label,
                    "training_seed": seed,
                    "scenario_count": len(probes),
                    "probe_selected_blocked_astar": fmean(
                        float(row["probe_selected_blocked_astar"]) for row in probes
                    ),
                    "probe_selected_safe_action": fmean(
                        float(row["probe_selected_safe_action"]) for row in probes
                    ),
                    "probe_selected_wait": fmean(
                        float(row["probe_selected_wait"]) for row in probes
                    ),
                    "probe_safe_q_margin": fmean(
                        float(row["probe_safe_q_margin"]) for row in probes
                    ),
                    "blocked_safe_success": blocked_safe,
                    "control_safe_success": control_safe,
                    "blocker_safe_success_drop": control_safe - blocked_safe,
                    "blocked_target_collision": fmean(
                        float(row["target_collision"]) for row in blocked
                    ),
                    "blocked_wait_at_decision": fmean(
                        float(row["wait_at_blockage_decision"]) for row in blocked
                    ),
                    "blocked_wait_steps": fmean(
                        float(row["wait_steps"]) for row in blocked
                    ),
                }
            )
    summary = []
    for method in METHODS:
        group = [row for row in per_seed if row["method"] == method.key]
        record = {
            "method": method.key,
            "method_label": method.label,
            "seed_count": len(group),
            "scenarios_per_seed": group[0]["scenario_count"],
        }
        for metric in metrics:
            mean, ci95 = _mean_ci95([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95"] = ci95
        summary.append(record)
    write_records_csv(per_seed, output_dir / "diagnostic_per_seed.csv")
    write_records_csv(summary, output_dir / "diagnostic_summary.csv")
    return summary


def _render_scenarios(scenarios, output: Path) -> None:
    figure, axes = plt.subplots(
        2, 3, figsize=(15, 10), dpi=180, constrained_layout=True
    )
    for axis, (problem, scenario, metadata) in zip(axes.flat, scenarios):
        grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
        for row, column in problem.obstacles:
            grid[row, column] = 1
        axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)
        axis.plot(
            [cell[1] for cell in problem.nominal_path],
            [cell[0] for cell in problem.nominal_path],
            "--",
            color="#1976d2",
            linewidth=1.4,
        )
        spec = scenario.obstacles[0]
        axis.plot(
            [cell[1] for cell in spec.route],
            [cell[0] for cell in spec.route],
            ":",
            color="#e53935",
            linewidth=2.0,
        )
        initial = spec.route[spec.start_index]
        axis.scatter(initial[1], initial[0], color="#e53935", s=45)
        axis.scatter(problem.start[1], problem.start[0], color="#43a047", s=45)
        target = tuple(metadata["target_cell"])
        axis.scatter(target[1], target[0], marker="X", color="#b71c1c", s=75)
        axis.set_title(
            f"{metadata['region']} / direction {metadata['direction_variant']} / "
            f"start {metadata['lead_steps']} steps before",
            fontsize=10,
        )
        axis.set_xlim(-0.5, problem.size - 0.5)
        axis.set_ylim(problem.size - 0.5, -0.5)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Controlled blockage: one causal obstacle, no filler obstacles")
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _plot_summary(summary: Sequence[Mapping[str, Any]], output: Path) -> None:
    panels = (
        ("probe_selected_safe_action_mean", "Exact state: selects safe action"),
        ("blocked_safe_success_mean", "Near-blockage safe success"),
        ("blocked_target_collision_mean", "Collision at target cell"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    colors = (
        "#546e7a",
        "#1976d2",
        "#d84315",
        "#f9a825",
        "#8e24aa",
        "#43a047",
        "#00838f",
        "#c62828",
        "#6a1b9a",
        "#ad1457",
    )
    x = np.arange(len(METHODS))
    for axis, (metric, title) in zip(axes, panels):
        values = [float(row[metric]) for row in summary]
        error_key = metric.replace("_mean", "_ci95")
        errors = [float(row[error_key]) for row in summary]
        axis.bar(x, values, yerr=errors, color=colors, capsize=3)
        axis.set_xticks(x, [method.label for method in METHODS], rotation=35, ha="right")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Near-blockage causal diagnostic | mean and 95% CI across seeds")
    figure.tight_layout()
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _write_report(summary: Sequence[Mapping[str, Any]], output: Path) -> None:
    lines = [
        "# Near-blockage causal diagnostic",
        "",
        "The agent starts four path steps before one phase-aligned blocker. There are no filler obstacles. Every blocked rollout has a matched no-obstacle control from the same start.",
        "",
        "| Method | Exact-state safe action | Exact-state blocked A* | Blocked safe success | Control safe success | Target collision | Wait at decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        percent = lambda key: f"{100 * float(row[key]):.1f}%"
        lines.append(
            f"| {row['method_label']} | {percent('probe_selected_safe_action_mean')} | "
            f"{percent('probe_selected_blocked_astar_mean')} | "
            f"{percent('blocked_safe_success_mean')} | "
            f"{percent('control_safe_success_mean')} | "
            f"{percent('blocked_target_collision_mean')} | "
            f"{percent('blocked_wait_at_decision_mean')} |"
        )
    lines.extend(
        [
            "",
            "The forced-state safe-action rate is the direct A*-conflict measure. The blocked-versus-control difference isolates the obstacle effect; the rollout no longer treats a route chosen near the original map start as a response to a distant blocker.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-steps", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--output-dir", default="outputs/office_controlled_blockage_diagnostic_v1"
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _build_scenarios(args.lead_steps, output_dir)
    _render_scenarios(scenarios, output_dir / "controlled_blockage_layouts.png")
    if args.build_only:
        print(f"Built and verified controlled blockage scenarios in {output_dir}")
        return
    probe_rows, rollout_rows, seeds = _run_evaluation(
        scenarios, args.device, output_dir
    )
    summary = _summarize(probe_rows, rollout_rows, seeds, output_dir)
    _plot_summary(summary, output_dir / "diagnostic_comparison.png")
    _write_report(summary, output_dir / "REPORT.md")
    print(f"Wrote controlled blockage diagnostic to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
