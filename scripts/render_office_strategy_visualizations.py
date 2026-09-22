"""Render the fixed Office map and multi-policy dynamic rollout comparisons.

The script is intentionally evaluation-only.  It loads each run's frozen
``model_selected.pth`` and places every policy in the same deterministic
training, validation, or critical-blockage scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import scenarios_from_spatial_manifest
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.maps.render import (
    render_dynamic_obstacle_animation,
    render_problem_layout,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json
from render_dynamic_replay_comparison import PolicyRollout, _agent, _rollout


BASE_CONFIG = "configs/dynamic_spatial_generalization_office_balanced_v4.yaml"
LOCAL_CONFIG = "configs/dynamic_office_local_conflict_demo_v1.yaml"
CRITICAL_MANIFEST = (
    "outputs/office_local_conflict_study_v1/critical_blockage_scenarios.json"
)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    key: str
    label: str
    config_path: str
    training_strategy: str
    color: str


METHODS = (
    MethodSpec("uniform", "Uniform", BASE_CONFIG, "uniform", "#2563eb"),
    MethodSpec("prefill", "Prefill", BASE_CONFIG, "prefill", "#f59e0b"),
    MethodSpec(
        "persistent_demo",
        "Persistent",
        BASE_CONFIG,
        "persistent_demo",
        "#059669",
    ),
    MethodSpec(
        "ca_current",
        "CA-current",
        "configs/dynamic_spatial_generalization_office_conflict_current_v1.yaml",
        "conflict_adaptive_demo",
        "#a855f7",
    ),
    MethodSpec(
        "ca_predictive",
        "CA-predictive",
        "configs/dynamic_spatial_generalization_office_conflict_adaptive_v1.yaml",
        "conflict_adaptive_demo",
        "#ef4444",
    ),
    MethodSpec(
        "local_conflict_demo",
        "Local-conflict",
        LOCAL_CONFIG,
        "local_conflict_demo",
        "#0891b2",
    ),
)
METHOD_BY_KEY = {item.key: item for item in METHODS}
OBSTACLE_COLORS = ("#dc2626", "#9333ea", "#ea580c", "#0d9488", "#db2777")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _problem_and_scenarios() -> tuple[
    NavigationProblem,
    tuple[DynamicScenario, ...],
    tuple[DynamicScenario, ...],
]:
    config = load_config(_resolve(BASE_CONFIG))
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    matches = [item for item in problems if item.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one Office training problem.")
    problem = matches[0]
    manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))
    return (
        problem,
        scenarios_from_spatial_manifest(problem, manifest, "train"),
        scenarios_from_spatial_manifest(problem, manifest, "validation"),
    )


def _select_scenario(
    scenarios: tuple[DynamicScenario, ...], scenario_id: int, split: str
) -> DynamicScenario:
    matches = [item for item in scenarios if item.seed == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown {split} scenario id: {scenario_id}.")
    return matches[0]


def _load_critical_scenario(scenario_id: int) -> DynamicScenario:
    payload = load_json(_resolve(CRITICAL_MANIFEST))
    records = payload["maps"]["office"]["scenarios"]
    matches = [row for row in records if int(row["scenario_id"]) == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown Office critical scenario id: {scenario_id}.")
    specs = []
    for row in matches[0]["obstacles"]:
        specs.append(
            DynamicObstacleSpec(
                route=tuple(tuple(int(value) for value in cell) for cell in row["route"]),
                start_index=int(row["start_index"]),
                direction=int(row["direction"]),
                move_every=int(row["move_every"]),
                label=str(row["label"]),
            )
        )
    return DynamicScenario(scenario_id, tuple(specs))


def _find_run_dir(config: dict, strategy: str, seed: int) -> Path:
    output_root = _resolve(config["experiment"]["output_root"])
    matches = []
    for metadata_path in output_root.glob("*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not bool(metadata.get("smoke", False))
            and str(metadata.get("strategy")) == strategy
            and int(metadata.get("training_seed", -1)) == seed
        ):
            matches.append(metadata_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run for strategy={strategy}, seed={seed}; found {len(matches)}."
        )
    return matches[0]


def _load_rollouts(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    training_seed: int,
    device: str,
) -> tuple[PolicyRollout, ...]:
    rollouts = []
    for method in METHODS:
        config = load_config(_resolve(method.config_path))
        run_dir = _find_run_dir(config, method.training_strategy, training_seed)
        model_path = run_dir / "model_selected.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing selected model: {model_path}")
        agent = _agent(config, training_seed, device)
        agent.load_weights(model_path)
        rollouts.append(_rollout(agent, problem, scenario, config, method.key))
    return tuple(rollouts)


def _base_axis(
    axis,
    problem: NavigationProblem,
    scenario: DynamicScenario,
    *,
    show_initial_obstacles: bool,
) -> None:
    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1
    axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)
    axis.plot(
        [cell[1] for cell in problem.nominal_path],
        [cell[0] for cell in problem.nominal_path],
        color="#1677b8",
        linewidth=1.5,
        linestyle="--",
        alpha=0.75,
        zorder=2,
    )
    for index, spec in enumerate(scenario.obstacles):
        color = OBSTACLE_COLORS[index % len(OBSTACLE_COLORS)]
        axis.plot(
            [cell[1] for cell in spec.route],
            [cell[0] for cell in spec.route],
            color=color,
            linewidth=1.3,
            linestyle=":",
            alpha=0.9,
            zorder=3,
        )
        if show_initial_obstacles:
            position = spec.route[spec.start_index]
            axis.scatter(
                position[1],
                position[0],
                c=color,
                edgecolors="white",
                linewidths=0.8,
                s=70,
                zorder=7,
            )
    axis.scatter(
        problem.start[1],
        problem.start[0],
        c="#16a34a",
        edgecolors="white",
        linewidths=0.8,
        s=65,
        zorder=8,
    )
    axis.scatter(
        problem.goal[1],
        problem.goal[0],
        c="#facc15",
        edgecolors="black",
        linewidths=0.7,
        marker="*",
        s=100,
        zorder=8,
    )
    axis.set_xlim(-0.5, problem.size - 0.5)
    axis.set_ylim(problem.size - 0.5, -0.5)
    axis.set_xticks(np.arange(-0.5, problem.size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, problem.size, 1), minor=True)
    axis.grid(which="minor", color="#64748b", linewidth=0.25, alpha=0.2)
    axis.set_xticks([])
    axis.set_yticks([])


def _render_scenario_layout(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    output_path: Path,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 8))
    _base_axis(axis, problem, scenario, show_initial_obstacles=True)
    legend = [
        Line2D([0], [0], color="#1677b8", linestyle="--", label="Static A* path"),
        Line2D([0], [0], color="#dc2626", linestyle=":", marker="o", label="Dynamic routes/start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a", label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#facc15", label="Goal"),
    ]
    axis.legend(handles=legend, loc="lower right", fontsize=8)
    axis.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def _render_final_paths(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    rollouts: tuple[PolicyRollout, ...],
    output_path: Path,
    title: str,
) -> None:
    columns = 4 if len(rollouts) > 6 else 3
    rows = math.ceil(len(rollouts) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, rollout in zip(axes.flat, rollouts):
        _base_axis(axis, problem, scenario, show_initial_obstacles=False)
        method = METHOD_BY_KEY[rollout.strategy]
        axis.plot(
            [cell[1] for cell in rollout.path],
            [cell[0] for cell in rollout.path],
            color=method.color,
            linewidth=2.2,
            zorder=5,
        )
        if rollout.collisions:
            axis.scatter(
                [cell[1] for cell in rollout.collisions],
                [cell[0] for cell in rollout.collisions],
                c="#ef4444",
                marker="x",
                s=65,
                linewidths=1.8,
                zorder=9,
            )
        safe = rollout.success and not rollout.collisions
        axis.set_title(
            f"{method.label} | safe={safe} | steps={rollout.steps} | "
            f"collisions={len(rollout.collisions)} | waits={rollout.wait_steps}",
            fontsize=9,
        )
    for axis in axes.flat[len(rollouts) :]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)


def _render_rollout_animation(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    rollouts: tuple[PolicyRollout, ...],
    output_path: Path,
    interval_ms: int,
    title: str,
    max_frames: int,
) -> None:
    columns = 4 if len(rollouts) > 6 else 3
    rows = math.ceil(len(rollouts) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    panels = []
    for axis, rollout in zip(axes.flat, rollouts):
        _base_axis(axis, problem, scenario, show_initial_obstacles=False)
        method = METHOD_BY_KEY[rollout.strategy]
        trail, = axis.plot([], [], color=method.color, linewidth=2.2, zorder=5)
        agent = axis.scatter(
            [], [], c=method.color, edgecolors="white", linewidths=1.0, s=75, zorder=9
        )
        obstacles = [
            axis.scatter(
                [], [],
                c=OBSTACLE_COLORS[index % len(OBSTACLE_COLORS)],
                edgecolors="white",
                linewidths=0.8,
                s=65,
                zorder=8,
            )
            for index in range(len(scenario.obstacles))
        ]
        collisions = axis.scatter(
            [], [], c="#ef4444", marker="x", linewidths=1.8, s=60, zorder=10
        )
        status = axis.text(
            0.02,
            0.02,
            "",
            transform=axis.transAxes,
            fontsize=7.5,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.9},
            zorder=11,
        )
        axis.set_title(method.label, fontsize=10)
        panels.append((trail, agent, obstacles, collisions, status))
    for axis in axes.flat[len(rollouts) :]:
        axis.set_visible(False)

    frame_count = min(max(len(item.frames) for item in rollouts), max_frames)

    def update(frame_index: int):
        changed = []
        for rollout, panel in zip(rollouts, panels):
            trail, agent, obstacles, collisions, status = panel
            local_index = min(frame_index, len(rollout.frames) - 1)
            frame = rollout.frames[local_index]
            visited = rollout.path[: local_index + 1]
            trail.set_data(
                [cell[1] for cell in visited], [cell[0] for cell in visited]
            )
            agent.set_offsets(
                np.asarray([[frame.agent_position[1], frame.agent_position[0]]])
            )
            for marker, position in zip(obstacles, frame.dynamic_positions):
                marker.set_offsets(np.asarray([[position[1], position[0]]]))
            collision_positions = [
                item.collision_position
                for item in rollout.frames[: local_index + 1]
                if item.collision_position is not None
            ]
            collisions.set_offsets(
                np.asarray(
                    [[cell[1], cell[0]] for cell in collision_positions], dtype=float
                ).reshape((-1, 2))
            )
            waits_so_far = sum(
                item.action_name == "stay" for item in rollout.frames[1 : local_index + 1]
            )
            done = local_index == len(rollout.frames) - 1
            status.set_text(
                f"step {local_index}/{rollout.steps} | {frame.action_name}\n"
                f"collisions {len(collision_positions)} | waits {waits_so_far} | "
                f"{'done' if done else 'running'}"
            )
            changed.extend([trail, agent, *obstacles, collisions, status])
        return tuple(changed)

    fig.suptitle(title, fontsize=14)
    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    animation.save(
        output_path,
        writer=PillowWriter(fps=max(1, round(1000 / interval_ms))),
        dpi=90,
    )
    plt.close(fig)


def _rollout_record(rollout: PolicyRollout) -> dict:
    return {
        "strategy": rollout.strategy,
        "label": METHOD_BY_KEY[rollout.strategy].label,
        "success": rollout.success,
        "safe_success": rollout.success and not rollout.collisions,
        "steps": rollout.steps,
        "collision_count": len(rollout.collisions),
        "wait_steps": rollout.wait_steps,
        "path": [list(cell) for cell in rollout.path],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Office training/validation maps and six-policy GIFs."
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--training-scenario-id", type=int, default=0)
    parser.add_argument("--validation-scenario-id", type=int, default=10013)
    parser.add_argument("--critical-training-seed", type=int, default=2)
    parser.add_argument("--critical-scenario-id", type=int, default=50011)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interval-ms", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--output-dir", default="outputs/office_strategy_visualizations_v1"
    )
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem, training_scenarios, validation_scenarios = _problem_and_scenarios()
    training_scenario = _select_scenario(
        training_scenarios, args.training_scenario_id, "training"
    )
    validation_scenario = _select_scenario(
        validation_scenarios, args.validation_scenario_id, "validation"
    )
    critical_scenario = _load_critical_scenario(args.critical_scenario_id)

    render_problem_layout(problem, output_dir / "office_static_map.png")
    _render_scenario_layout(
        problem,
        training_scenario,
        output_dir / f"training_scenario_{training_scenario.seed}_layout.png",
        f"Office training scenario {training_scenario.seed}: initial state and routes",
    )
    render_dynamic_obstacle_animation(
        problem,
        training_scenario.obstacles,
        output_dir / f"training_scenario_{training_scenario.seed}_obstacles.gif",
        frames=32,
        interval_ms=args.interval_ms,
    )
    _render_scenario_layout(
        problem,
        validation_scenario,
        output_dir / f"validation_scenario_{validation_scenario.seed}_layout.png",
        f"Held-out validation scenario {validation_scenario.seed}: initial state and routes",
    )
    render_dynamic_obstacle_animation(
        problem,
        validation_scenario.obstacles,
        output_dir / f"validation_scenario_{validation_scenario.seed}_obstacles.gif",
        frames=32,
        interval_ms=args.interval_ms,
    )

    validation_rollouts = _load_rollouts(
        problem, validation_scenario, args.training_seed, args.device
    )
    _render_final_paths(
        problem,
        validation_scenario,
        validation_rollouts,
        output_dir / f"validation_scenario_{validation_scenario.seed}_all_strategies.png",
        f"Selected checkpoints, training seed {args.training_seed}, held-out validation scenario {validation_scenario.seed}",
    )
    _render_rollout_animation(
        problem,
        validation_scenario,
        validation_rollouts,
        output_dir / f"validation_scenario_{validation_scenario.seed}_all_strategies.gif",
        args.interval_ms,
        f"Six frozen policies on the same validation scenario {validation_scenario.seed} (seed {args.training_seed})",
        args.max_frames,
    )

    critical_rollouts = _load_rollouts(
        problem, critical_scenario, args.critical_training_seed, args.device
    )
    _render_scenario_layout(
        problem,
        critical_scenario,
        output_dir / f"critical_scenario_{critical_scenario.seed}_layout.png",
        f"Purpose-built A* blockage scenario {critical_scenario.seed}",
    )
    _render_final_paths(
        problem,
        critical_scenario,
        critical_rollouts,
        output_dir / f"critical_scenario_{critical_scenario.seed}_all_strategies.png",
        f"Critical A* blockage, training seed {args.critical_training_seed}, scenario {critical_scenario.seed}",
    )
    _render_rollout_animation(
        problem,
        critical_scenario,
        critical_rollouts,
        output_dir / f"critical_scenario_{critical_scenario.seed}_all_strategies.gif",
        args.interval_ms,
        f"Six frozen policies on critical A* blockage scenario {critical_scenario.seed} (seed {args.critical_training_seed})",
        args.max_frames,
    )

    metadata = {
        "map_id": problem.map_id,
        "static_map_is_shared_across_splits": True,
        "training_scenario_id": training_scenario.seed,
        "validation": {
            "scenario_id": validation_scenario.seed,
            "training_seed": args.training_seed,
            "rollouts": [_rollout_record(item) for item in validation_rollouts],
        },
        "critical_blockage": {
            "scenario_id": critical_scenario.seed,
            "training_seed": args.critical_training_seed,
            "rollouts": [_rollout_record(item) for item in critical_rollouts],
        },
    }
    write_json(metadata, output_dir / "visualization_metadata.json")
    for group, rollouts in (
        ("validation", validation_rollouts),
        ("critical", critical_rollouts),
    ):
        for rollout in rollouts:
            print(
                f"{group} {rollout.strategy}: success={rollout.success} "
                f"steps={rollout.steps} collisions={len(rollout.collisions)} "
                f"waits={rollout.wait_steps}"
            )
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
