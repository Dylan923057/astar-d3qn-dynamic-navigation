"""Render one held-out dynamic scenario for three trained replay policies."""

from __future__ import annotations

import argparse
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

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.core.grid import ACTION_NAMES, Action, Position
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario, build_random_crossing_scenario
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.maps.render import (
    render_dynamic_obstacle_animation,
    render_problem_layout,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json


STRATEGIES = ("uniform", "prefill", "persistent_demo")
STRATEGY_LABELS = {
    "uniform": "Uniform",
    "prefill": "Prefill",
    "persistent_demo": "Persistent Demo",
}
STRATEGY_COLORS = {
    "uniform": "#2563eb",
    "prefill": "#f59e0b",
    "persistent_demo": "#059669",
}
OBSTACLE_COLORS = ("#dc2626", "#9333ea", "#ea580c")


@dataclass(frozen=True, slots=True)
class RolloutFrame:
    agent_position: Position
    dynamic_positions: tuple[Position, ...]
    action_name: str
    collision_position: Position | None


@dataclass(frozen=True, slots=True)
class PolicyRollout:
    strategy: str
    frames: tuple[RolloutFrame, ...]
    path: tuple[Position, ...]
    collisions: tuple[Position, ...]
    success: bool
    steps: int
    wait_steps: int


def _reward_config(values: dict) -> RewardConfig:
    return RewardConfig(**{key: float(value) for key, value in values.items()})


def _agent(config: dict, seed: int, device: str) -> D3QNAgent:
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


def _rollout(
    agent: D3QNAgent,
    problem: NavigationProblem,
    scenario: DynamicScenario,
    config: dict,
    strategy: str,
) -> PolicyRollout:
    environment = config["environment"]
    env = DynamicGridNavigationEnv(
        problem,
        dynamic_obstacles=scenario.obstacles,
        scenario_id=scenario.seed,
        max_steps=int(environment["max_steps"]),
        reward_config=_reward_config(config["reward"]),
        terminate_on_collision=bool(environment["terminate_on_collision"]),
        window_size=int(environment["window_size"]),
    )
    state = env.reset()
    frames = [
        RolloutFrame(
            agent_position=env.position,
            dynamic_positions=env.dynamic_positions,
            action_name="start",
            collision_position=None,
        )
    ]
    path = [env.position]
    collisions: list[Position] = []
    wait_steps = 0
    while True:
        if bool(environment.get("mask_static_invalid_actions", False)):
            action_mask = env.action_mask(mask_collisions=True)
            valid_actions = [
                index for index, valid in enumerate(action_mask) if bool(valid)
            ]
            action = agent.select_action(
                state, epsilon=0.0, valid_actions=valid_actions
            )
        else:
            action = agent.select_action(state, epsilon=0.0)
        result = env.step(action)
        state = result.observation
        collision_position = None
        if bool(result.info["collision"]):
            collision_position = tuple(
                result.info["collision_position"] or env.position
            )
            collisions.append(collision_position)
        if action == int(Action.STAY):
            wait_steps += 1
        path.append(env.position)
        frames.append(
            RolloutFrame(
                agent_position=env.position,
                dynamic_positions=env.dynamic_positions,
                action_name=ACTION_NAMES[action],
                collision_position=collision_position,
            )
        )
        if result.done:
            break
    return PolicyRollout(
        strategy=strategy,
        frames=tuple(frames),
        path=tuple(path),
        collisions=tuple(collisions),
        success=bool(result.info["reached"]),
        steps=env.steps,
        wait_steps=wait_steps,
    )


def _draw_base(
    axis,
    problem: NavigationProblem,
    scenario: DynamicScenario,
) -> None:
    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1
    axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)
    axis.plot(
        [cell[1] for cell in problem.nominal_path],
        [cell[0] for cell in problem.nominal_path],
        color="#0f766e",
        linewidth=1.2,
        linestyle="--",
        alpha=0.55,
        zorder=2,
    )
    for index, spec in enumerate(scenario.obstacles):
        axis.plot(
            [cell[1] for cell in spec.route],
            [cell[0] for cell in spec.route],
            color=OBSTACLE_COLORS[index % len(OBSTACLE_COLORS)],
            linewidth=1.4,
            linestyle=":",
            alpha=0.85,
            zorder=3,
        )
    axis.scatter(
        problem.start[1], problem.start[0], c="#16a34a", s=42, zorder=7
    )
    axis.scatter(
        problem.goal[1],
        problem.goal[0],
        c="#facc15",
        edgecolors="black",
        linewidths=0.7,
        marker="*",
        s=85,
        zorder=7,
    )
    axis.set_xlim(-0.5, problem.size - 0.5)
    axis.set_ylim(problem.size - 0.5, -0.5)
    axis.set_xticks(np.arange(-0.5, problem.size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, problem.size, 1), minor=True)
    axis.grid(which="minor", color="#64748b", linewidth=0.25, alpha=0.2)
    axis.set_xticks([])
    axis.set_yticks([])


def _render_final_comparison(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    rollouts: tuple[PolicyRollout, ...],
    output_path: Path,
    title: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    for axis, rollout in zip(axes, rollouts):
        _draw_base(axis, problem, scenario)
        color = STRATEGY_COLORS[rollout.strategy]
        axis.plot(
            [cell[1] for cell in rollout.path],
            [cell[0] for cell in rollout.path],
            color=color,
            linewidth=2.2,
            zorder=5,
        )
        if rollout.collisions:
            axis.scatter(
                [cell[1] for cell in rollout.collisions],
                [cell[0] for cell in rollout.collisions],
                c="#ef4444",
                marker="x",
                s=80,
                linewidths=2.2,
                zorder=8,
            )
        axis.set_title(
            f"{STRATEGY_LABELS[rollout.strategy]}\n"
            f"success={rollout.success} | steps={rollout.steps} | "
            f"collisions={len(rollout.collisions)} | waits={rollout.wait_steps}",
            fontsize=10,
        )
    fig.suptitle(
        title or f"Map 1 held-out dynamic scenario {scenario.seed}: final greedy paths",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def _render_rollout_animation(
    problem: NavigationProblem,
    scenario: DynamicScenario,
    rollouts: tuple[PolicyRollout, ...],
    output_path: Path,
    interval_ms: int,
    title: str | None = None,
    max_frames: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    artists = []
    for axis, rollout in zip(axes, rollouts):
        _draw_base(axis, problem, scenario)
        color = STRATEGY_COLORS[rollout.strategy]
        trail, = axis.plot([], [], color=color, linewidth=2.3, zorder=5)
        agent_marker = axis.scatter(
            [], [], c=color, edgecolors="white", linewidths=1.2, s=90, zorder=9
        )
        obstacle_markers = [
            axis.scatter(
                [],
                [],
                c=OBSTACLE_COLORS[index % len(OBSTACLE_COLORS)],
                edgecolors="white",
                linewidths=1.0,
                s=85,
                zorder=8,
            )
            for index in range(len(scenario.obstacles))
        ]
        collision_markers = axis.scatter(
            [], [], c="#ef4444", marker="x", linewidths=2.0, s=70, zorder=10
        )
        status = axis.text(
            0.02,
            0.02,
            "",
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.9},
            zorder=11,
        )
        axis.set_title(STRATEGY_LABELS[rollout.strategy], fontsize=11)
        artists.append(
            (trail, agent_marker, obstacle_markers, collision_markers, status)
        )

    legend = [
        Line2D([0], [0], color="#0f766e", linestyle="--", label="A* nominal path"),
        Line2D([0], [0], color="#dc2626", linestyle=":", marker="o", label="Dynamic obstacle"),
        Line2D([0], [0], color="#ef4444", marker="x", linestyle="none", label="Collision"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(
        title or f"Map 1 held-out scenario {scenario.seed}: frozen greedy policy rollout",
        fontsize=14,
    )
    frame_count = max(len(rollout.frames) for rollout in rollouts)
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))

    def update(frame_index: int):
        changed = []
        for rollout, panel in zip(rollouts, artists):
            trail, agent_marker, obstacle_markers, collision_markers, status = panel
            local_index = min(frame_index, len(rollout.frames) - 1)
            frame = rollout.frames[local_index]
            visited = rollout.path[: local_index + 1]
            trail.set_data(
                [cell[1] for cell in visited], [cell[0] for cell in visited]
            )
            agent_marker.set_offsets(
                np.asarray([[frame.agent_position[1], frame.agent_position[0]]])
            )
            for marker, position in zip(obstacle_markers, frame.dynamic_positions):
                marker.set_offsets(np.asarray([[position[1], position[0]]]))
            collision_positions = tuple(
                item.collision_position
                for item in rollout.frames[: local_index + 1]
                if item.collision_position is not None
            )
            collision_markers.set_offsets(
                np.asarray(
                    [[cell[1], cell[0]] for cell in collision_positions],
                    dtype=float,
                ).reshape((-1, 2))
            )
            state = "done" if local_index == len(rollout.frames) - 1 else "running"
            status.set_text(
                f"step {local_index:02d}/{rollout.steps:02d} | "
                f"action {frame.action_name}\n"
                f"collisions {len(collision_positions)} | waits {rollout.wait_steps} | {state}"
            )
            changed.extend(
                [trail, agent_marker, *obstacle_markers, collision_markers, status]
            )
        return tuple(changed)

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
        dpi=105,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render three final replay policies on one held-out scenario."
    )
    parser.add_argument("--config", default="configs/dynamic_generalization_map01.yaml")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--scenario-seed", type=int, default=108)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default="outputs/dynamic_generalization_map01/visualizations/scenario_108_seed_0",
    )
    parser.add_argument("--interval-ms", type=int, default=140)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    map_path = Path(config["map_sets"]["train"]["file"])
    if not map_path.is_absolute():
        map_path = ROOT / map_path
    problem = load_problem_set(map_path)[0]
    dynamic = config["dynamic_generalization"]
    scenario = build_random_crossing_scenario(
        problem,
        args.scenario_seed,
        obstacle_count=int(dynamic["obstacle_count"]),
        route_length=int(dynamic["route_length"]),
        move_every=int(dynamic["move_every"]),
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_root = Path(config["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    rollouts = []
    for strategy in STRATEGIES:
        agent = _agent(config, args.training_seed, args.device)
        model_path = output_root / (
            f"dynamic_generalization_map01_seed_{args.training_seed}_{strategy}"
        ) / "model_final.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}")
        agent.load_weights(model_path)
        rollouts.append(_rollout(agent, problem, scenario, config, strategy))
    rollout_tuple = tuple(rollouts)

    render_problem_layout(problem, output_dir / "map1_static_layout.png")
    render_dynamic_obstacle_animation(
        problem,
        scenario.obstacles,
        output_dir / "scenario_108_dynamic_obstacles.gif",
        frames=32,
        interval_ms=args.interval_ms,
    )
    _render_final_comparison(
        problem,
        scenario,
        rollout_tuple,
        output_dir / "replay_final_paths.png",
    )
    _render_rollout_animation(
        problem,
        scenario,
        rollout_tuple,
        output_dir / "replay_path_comparison.gif",
        args.interval_ms,
    )
    write_json(
        {
            "map_id": problem.map_id,
            "training_seed": args.training_seed,
            "scenario_seed": scenario.seed,
            "dynamic_obstacles": [
                {
                    "route": [list(cell) for cell in spec.route],
                    "start_index": spec.start_index,
                    "direction": spec.direction,
                    "move_every": spec.move_every,
                }
                for spec in scenario.obstacles
            ],
            "rollouts": [
                {
                    "strategy": rollout.strategy,
                    "success": rollout.success,
                    "steps": rollout.steps,
                    "collision_count": len(rollout.collisions),
                    "wait_steps": rollout.wait_steps,
                    "path": [list(cell) for cell in rollout.path],
                }
                for rollout in rollout_tuple
            ],
        },
        output_dir / "visualization_metadata.json",
    )
    for rollout in rollout_tuple:
        print(
            f"{rollout.strategy}: success={rollout.success} steps={rollout.steps} "
            f"collisions={len(rollout.collisions)} waits={rollout.wait_steps}"
        )
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
