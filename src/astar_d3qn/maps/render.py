from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np

from astar_d3qn.core.grid import Action
from astar_d3qn.core.trajectory import WaitEvent
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec

from .problem import NavigationProblem


def _draw_problem_layout(
    axis,
    problem: NavigationProblem,
    *,
    show_legend: bool,
) -> None:
    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1

    axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)
    path_rows = [cell[0] for cell in problem.nominal_path]
    path_columns = [cell[1] for cell in problem.nominal_path]
    axis.plot(
        path_columns,
        path_rows,
        color="#1677b8",
        linewidth=2.0,
        linestyle="--",
        label="A* nominal path",
        zorder=3,
    )
    axis.scatter(
        problem.start[1],
        problem.start[0],
        c="#2ca02c",
        s=65,
        label="Start",
        zorder=4,
    )
    axis.scatter(
        problem.goal[1],
        problem.goal[0],
        c="#f2c94c",
        edgecolors="black",
        marker="*",
        s=110,
        label="Goal",
        zorder=4,
    )
    axis.set_xticks(range(problem.size))
    axis.set_yticks(range(problem.size))
    axis.tick_params(labelsize=5 if problem.size >= 30 else 7, pad=1)
    axis.grid(color="#64748b", linewidth=0.35, alpha=0.22)
    axis.set_xlim(-0.5, problem.size - 0.5)
    axis.set_ylim(problem.size - 0.5, -0.5)
    scene = str(problem.metadata.get("scene", problem.map_id)).replace("_", " ").title()
    axis.set_title(
        f"{scene}\n{problem.size}x{problem.size} | obstacles {len(problem.obstacles)} | "
        f"A* {problem.astar_steps} | turns {problem.metadata.get('turn_count', '?')}",
        fontsize=10,
    )
    if show_legend:
        axis.legend(loc="lower right", fontsize=8)


def render_problem_layout(
    problem: NavigationProblem,
    output_path: str | Path,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 7))
    _draw_problem_layout(axis, problem, show_legend=True)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def render_problem_overview(
    problems: Sequence[NavigationProblem],
    output_path: str | Path,
) -> None:
    if not problems:
        raise ValueError("Cannot render an empty problem overview.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows = (len(problems) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 6 * rows), squeeze=False)
    for axis, problem in zip(axes.flat, problems):
        _draw_problem_layout(axis, problem, show_legend=False)
    for axis in list(axes.flat)[len(problems):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Structured static map layouts", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(target, dpi=180)
    plt.close(fig)


def render_dynamic_obstacle_animation(
    problem: NavigationProblem,
    dynamic_obstacles: Sequence[DynamicObstacleSpec],
    output_path: str | Path,
    *,
    frames: int = 96,
    interval_ms: int = 125,
) -> None:
    """Render one full deterministic obstacle-motion cycle as an animated GIF."""

    if not dynamic_obstacles:
        raise ValueError("At least one dynamic obstacle is required for animation.")
    if frames <= 0 or interval_ms <= 0:
        raise ValueError("frames and interval_ms must be positive.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    environment = DynamicGridNavigationEnv(
        problem,
        dynamic_obstacles=dynamic_obstacles,
        max_steps=frames + 1,
        window_size=15,
    )
    environment.reset()
    positions_by_frame: list[tuple[tuple[int, int], ...]] = []
    for _ in range(frames):
        positions_by_frame.append(environment.dynamic_positions)
        environment.step(int(Action.STAY))

    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1

    colors = (
        "#d32f2f",
        "#7b1fa2",
        "#ef6c00",
        "#00897b",
        "#c2185b",
        "#455a64",
    )
    display_labels = {
        "crossing": "Crossing",
        "head_on": "Head-on",
        "same_direction_slow": "Same-direction slow",
    }
    fig, axis = plt.subplots(figsize=(8.5, 8.5))
    axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)
    axis.plot(
        [cell[1] for cell in problem.nominal_path],
        [cell[0] for cell in problem.nominal_path],
        color="#1677b8",
        linewidth=1.6,
        linestyle="--",
        alpha=0.65,
        label="A* nominal path",
        zorder=2,
    )

    moving_markers = []
    for index, spec in enumerate(dynamic_obstacles):
        color = colors[index % len(colors)]
        label = display_labels.get(spec.label, spec.label.replace("_", " ").title())
        axis.plot(
            [cell[1] for cell in spec.route],
            [cell[0] for cell in spec.route],
            color=color,
            linewidth=2.0,
            linestyle=":",
            alpha=0.85,
            zorder=3,
        )
        marker = axis.scatter(
            [],
            [],
            s=135,
            c=color,
            edgecolors="white",
            linewidths=1.4,
            marker="o",
            zorder=6,
        )
        moving_markers.append(marker)
        center = spec.route[len(spec.route) // 2]
        axis.annotate(
            label,
            xy=(center[1], center[0]),
            xytext=(5, -9 if index == 0 else 7),
            textcoords="offset points",
            fontsize=8,
            color=color,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.82, "pad": 1.5},
            zorder=7,
        )

    axis.scatter(
        problem.start[1],
        problem.start[0],
        c="#2ca02c",
        edgecolors="white",
        linewidths=1.0,
        s=80,
        marker="o",
        zorder=5,
    )
    axis.scatter(
        problem.goal[1],
        problem.goal[0],
        c="#f2c94c",
        edgecolors="black",
        linewidths=0.8,
        s=130,
        marker="*",
        zorder=5,
    )
    axis.set_xticks(range(problem.size))
    axis.set_yticks(range(problem.size))
    axis.tick_params(labelsize=5, pad=1)
    axis.set_xticks(np.arange(-0.5, problem.size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, problem.size, 1), minor=True)
    axis.grid(which="minor", color="#64748b", linewidth=0.35, alpha=0.35)
    axis.grid(which="major", visible=False)
    axis.set_xlim(-0.5, problem.size - 0.5)
    axis.set_ylim(problem.size - 0.5, -0.5)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    title = axis.set_title("", fontsize=11)
    legend_handles = [
        Line2D([0], [0], color="#1677b8", linestyle="--", label="A* nominal path"),
        *[
            Line2D(
                [0],
                [0],
                color=colors[index % len(colors)],
                linestyle=":",
                marker="o",
                markerfacecolor=colors[index % len(colors)],
                label=display_labels.get(spec.label, spec.label),
            )
            for index, spec in enumerate(dynamic_obstacles)
        ],
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#f2c94c", label="Goal"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

    def update(frame_index: int):
        for marker, position in zip(moving_markers, positions_by_frame[frame_index]):
            marker.set_offsets(np.asarray([[position[1], position[0]]]))
        title.set_text(
            f"{problem.map_id} dynamic obstacle scenario | "
            f"step {frame_index:02d}/{frames - 1:02d} | dotted lines are routes"
        )
        return (*moving_markers, title)

    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    fig.tight_layout()
    animation.save(
        target,
        writer=PillowWriter(fps=max(1, round(1000 / interval_ms))),
        dpi=110,
    )
    plt.close(fig)


def render_final_path(
    problem: NavigationProblem,
    greedy_path: list[tuple[int, int]],
    output_path: str | Path,
    strategy: str,
    success: bool,
    collision_positions: Sequence[tuple[int, int]] = (),
    revisit_positions: Sequence[tuple[int, int]] = (),
    dynamic_routes: Sequence[Sequence[tuple[int, int]]] = (),
    wait_events: Sequence[WaitEvent] = (),
    path_label: str = "FINAL PATH",
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)

    for route_index, route in enumerate(dynamic_routes):
        route_rows = [cell[0] for cell in route]
        route_columns = [cell[1] for cell in route]
        axis.plot(
            route_columns,
            route_rows,
            color="#00897b",
            linewidth=1.8,
            linestyle=":",
            marker="s",
            markersize=3.5,
            alpha=0.9,
            label="Dynamic obstacle route" if route_index == 0 else None,
            zorder=2,
        )

    if strategy in {"prefill", "persistent_demo"}:
        astar_rows = [cell[0] for cell in problem.nominal_path]
        astar_columns = [cell[1] for cell in problem.nominal_path]
        axis.plot(
            astar_columns,
            astar_rows,
            color="#1677b8",
            linewidth=3.5,
            linestyle="--",
            alpha=0.75,
            label="A* reference",
        )

    greedy_rows = [cell[0] for cell in greedy_path]
    greedy_columns = [cell[1] for cell in greedy_path]
    axis.plot(
        greedy_columns,
        greedy_rows,
        color="#d62728",
        linewidth=2.0,
        label="Greedy policy",
        zorder=3,
    )

    if revisit_positions:
        revisit_rows = [cell[0] for cell in revisit_positions]
        revisit_columns = [cell[1] for cell in revisit_positions]
        axis.scatter(
            revisit_columns,
            revisit_rows,
            s=48,
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=1.5,
            marker="o",
            label="Revisited cell",
            zorder=4,
        )

    if collision_positions:
        collision_rows = [cell[0] for cell in collision_positions]
        collision_columns = [cell[1] for cell in collision_positions]
        axis.scatter(
            collision_columns,
            collision_rows,
            c="#e63946",
            s=70,
            linewidths=2.0,
            marker="x",
            label="Collision",
            zorder=5,
        )

    waits_by_position: dict[tuple[int, int], int] = defaultdict(int)
    for event in wait_events:
        waits_by_position[event.position] += event.duration
    if waits_by_position:
        wait_positions = list(waits_by_position)
        axis.scatter(
            [cell[1] for cell in wait_positions],
            [cell[0] for cell in wait_positions],
            c="#3f51b5",
            edgecolors="white",
            linewidths=0.8,
            marker="D",
            s=45,
            label="Wait",
            zorder=6,
        )
        for index, position in enumerate(wait_positions):
            duration = waits_by_position[position]
            axis.annotate(
                f"Wait {duration} step{'s' if duration != 1 else ''}",
                xy=(position[1], position[0]),
                xytext=(6, 8 if index % 2 == 0 else -12),
                textcoords="offset points",
                fontsize=7,
                color="#283593",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "#9fa8da",
                    "alpha": 0.85,
                },
                zorder=7,
            )

    axis.scatter(
        problem.start[1],
        problem.start[0],
        c="#2ca02c",
        s=70,
        label="Start",
        zorder=5,
    )
    axis.scatter(
        problem.goal[1],
        problem.goal[0],
        c="#f2c94c",
        edgecolors="black",
        marker="*",
        s=120,
        label="Goal",
        zorder=5,
    )

    status = "success" if success else "failure"
    axis.set_title(f"{problem.map_id} | {path_label} | {strategy} | {status}")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
