"""Render an exact visual comparison of the clean_v2 and clean_v3 designs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.evaluation.conflict import obstacle_positions
from astar_d3qn.utils.config import load_config
from generate_office_behavior_scenarios_v9 import _draw_base, _load_problem


OLD_COLORS = {
    "normal": "#546e7a",
    "wait": "#7b1fa2",
    "avoidance": "#00897b",
    "reroute": "#ef6c00",
}
PRIMARY_CONFLICT = "#c62828"
PRIMARY_CONTROL = "#1565c0"
CONTEXT = "#78909c"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _route_lookup(manifest: dict, split: str) -> dict[str, dict]:
    return {
        record["route_id"]: record
        for record in manifest["route_pools"][split]["corridor"]
    }


def _next_cell(route, start_index: int, direction: int):
    candidate = start_index + direction
    if candidate < 0 or candidate >= len(route):
        candidate = start_index - direction
    return route[candidate]


def _wait_cells(path) -> list[tuple[int, int]]:
    return [
        tuple(path[index])
        for index in range(1, len(path))
        if path[index] == path[index - 1]
    ]


def _draw_scene(
    axis,
    problem,
    manifest: dict,
    split: str,
    scene: dict,
    title: str,
    *,
    annotate_primary: bool = False,
    zoom: bool = False,
) -> None:
    _draw_base(axis, problem, title)
    lookup = _route_lookup(manifest, split)
    primary_id = scene.get("primary_route_id")
    role = scene.get("pair_role")
    path_color = (
        PRIMARY_CONFLICT
        if role == "conflict"
        else PRIMARY_CONTROL
        if role == "matched_control"
        else OLD_COLORS[scene["required_behavior"]]
    )
    oracle_path = scene["oracle_path"]
    axis.plot(
        [cell[1] for cell in oracle_path],
        [cell[0] for cell in oracle_path],
        color=path_color,
        linewidth=2.0,
        linestyle="--",
        alpha=0.9,
        zorder=4,
    )
    for obstacle_index, obstacle in enumerate(scene["obstacles"]):
        route = lookup[obstacle["route_id"]]["route"]
        is_primary = obstacle["route_id"] == primary_id
        route_color = path_color if is_primary else CONTEXT
        axis.plot(
            [cell[1] for cell in route],
            [cell[0] for cell in route],
            color=route_color,
            linewidth=4.0 if is_primary else 1.4,
            alpha=1.0 if is_primary else 0.55,
            zorder=5 if is_primary else 3,
        )
        initial = route[int(obstacle["start_index"])]
        following = _next_cell(
            route,
            int(obstacle["start_index"]),
            int(obstacle["direction"]),
        )
        axis.scatter(
            initial[1],
            initial[0],
            s=70 if is_primary else 28,
            color=route_color if is_primary else "#ef5350",
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
        )
        axis.annotate(
            "",
            xy=(following[1], following[0]),
            xytext=(initial[1], initial[0]),
            arrowprops={
                "arrowstyle": "->",
                "color": route_color if is_primary else "#455a64",
                "lw": 1.3 if is_primary else 0.7,
            },
            zorder=9,
        )
        if annotate_primary and is_primary:
            axis.annotate(
                f"关键障碍初始位置\nindex={obstacle['start_index']}",
                xy=(initial[1], initial[0]),
                xytext=(initial[1] + 2.2, initial[0] - 2.0),
                fontsize=9,
                color=route_color,
                fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "color": route_color, "lw": 1.2},
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": route_color},
                zorder=12,
            )
    waits = _wait_cells(oracle_path)
    if waits:
        wait = waits[0]
        axis.scatter(
            wait[1],
            wait[0],
            marker="s",
            s=85,
            color="#8e24aa",
            edgecolors="white",
            linewidths=1.0,
            zorder=11,
        )
        if annotate_primary:
            axis.annotate(
                "Oracle在这里等待1步",
                xy=(wait[1], wait[0]),
                xytext=(wait[1] + 2.0, wait[0] - 2.5),
                fontsize=9,
                color="#6a1b9a",
                fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "color": "#6a1b9a", "lw": 1.2},
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#8e24aa"},
                zorder=12,
            )
    if zoom:
        axis.set_xlim(5.5, 16.0)
        axis.set_ylim(14.0, 4.5)
        axis.set_xticks(range(6, 17, 2))
        axis.set_yticks(range(6, 15, 2))
        axis.grid(color="#b0bec5", linewidth=0.35, alpha=0.7)
        axis.tick_params(labelsize=7)


def _assert_pair_is_phase_only(conflict: dict, control: dict) -> None:
    physical_keys = ("route_id", "start_index", "direction", "move_every")
    for index, (left, right) in enumerate(
        zip(conflict["obstacles"], control["obstacles"], strict=True)
    ):
        differences = [key for key in physical_keys if left[key] != right[key]]
        expected = ["start_index"] if index == 0 else []
        if differences != expected:
            raise ValueError(
                f"Pair is not phase-only at obstacle {index}: {differences}."
            )


def _render_overview(problem, old: dict, new: dict, output: Path) -> None:
    labels = {
        "normal": "普通通过",
        "wait": "等待",
        "avoidance": "局部绕行",
        "reroute": "全局改道",
    }
    representatives = {
        behavior: next(
            scene
            for scene in old["scenarios"]["test"]
            if scene["required_behavior"] == behavior
        )
        for behavior in labels
    }
    conflict, control = new["scenarios"]["test"][:2]
    _assert_pair_is_phase_only(conflict, control)

    fig = plt.figure(figsize=(17, 10.5), dpi=180)
    grid = fig.add_gridspec(2, 4, height_ratios=(0.88, 1.15), hspace=0.28, wspace=0.08)
    for column, behavior in enumerate(labels):
        scene = representatives[behavior]
        axis = fig.add_subplot(grid[0, column])
        _draw_scene(
            axis,
            problem,
            old,
            "test",
            scene,
            f"旧版 clean_v2：{labels[behavior]}\n独立场景 #{scene['scenario_id']}",
        )
    conflict_axis = fig.add_subplot(grid[1, :2])
    control_axis = fig.add_subplot(grid[1, 2:])
    _draw_scene(
        conflict_axis,
        problem,
        new,
        "test",
        conflict,
        "新版 clean_v3：冲突成员\n静态A*会受阻，Oracle等待1步",
        annotate_primary=True,
    )
    _draw_scene(
        control_axis,
        problem,
        new,
        "test",
        control,
        "新版 clean_v3：匹配控制成员\n同组障碍、仅改关键障碍相位，直接通过",
        annotate_primary=True,
    )
    fig.suptitle(
        "Office clean_v2 与 clean_v3 动态场景设计对比",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        "静态墙体、起点、终点和每场动态障碍数量（5个）均未改变；变化的是障碍路线池、相位抽样与因果对照方式",
        ha="center",
        fontsize=12,
        color="#37474f",
    )
    handles = [
        Line2D([0], [0], color="#64b5f6", linestyle="--", label="静态A*路径"),
        Line2D([0], [0], color=CONTEXT, linewidth=2, label="其余4个控制障碍轨迹"),
        Line2D([0], [0], color=PRIMARY_CONFLICT, linewidth=4, label="冲突关键障碍轨迹"),
        Line2D([0], [0], color=PRIMARY_CONTROL, linewidth=4, label="控制关键障碍轨迹"),
        Line2D([0], [0], marker="s", color="#8e24aa", linestyle="None", label="Oracle等待位置"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.text(
        0.5,
        0.035,
        "v2：100/20/50个训练/验证/测试独立样本，覆盖4类行为　　"
        "v3：18/6/12个训练/验证/测试配对，专门检验A*示范与动态等待的冲突",
        ha="center",
        fontsize=10.5,
        color="#263238",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _render_pair_detail(problem, manifest: dict, output: Path) -> None:
    conflict, control = manifest["scenarios"]["test"][:2]
    _assert_pair_is_phase_only(conflict, control)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12), dpi=180)
    _draw_scene(
        axes[0, 0],
        problem,
        manifest,
        "test",
        conflict,
        "冲突成员：完整地图",
        annotate_primary=True,
    )
    _draw_scene(
        axes[0, 1],
        problem,
        manifest,
        "test",
        control,
        "匹配控制成员：完整地图",
        annotate_primary=True,
    )
    _draw_scene(
        axes[1, 0],
        problem,
        manifest,
        "test",
        conflict,
        "冲突局部：index=4，Oracle 71步（含等待1步）",
        annotate_primary=True,
        zoom=True,
    )
    _draw_scene(
        axes[1, 1],
        problem,
        manifest,
        "test",
        control,
        "控制局部：index=2，Oracle 70步（不等待）",
        annotate_primary=True,
        zoom=True,
    )
    fig.suptitle(
        "clean_v3 测试配对示例：物理输入只改变关键障碍的起始相位",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "相同项：静态地图、起终点、5条障碍轨迹、移动方向、速度、其余4个障碍相位；"
        "唯一改动：第1个关键障碍 start_index",
        ha="center",
        fontsize=10.5,
        color="#263238",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _specs(scene: dict, lookup: dict[str, dict]) -> list[DynamicObstacleSpec]:
    return [
        DynamicObstacleSpec(
            route=tuple(tuple(cell) for cell in lookup[item["route_id"]]["route"]),
            start_index=int(item["start_index"]),
            direction=int(item["direction"]),
            move_every=int(item["move_every"]),
        )
        for item in scene["obstacles"]
    ]


def _render_pair_animation(problem, manifest: dict, output: Path) -> None:
    conflict, control = manifest["scenarios"]["test"][:2]
    lookup = _route_lookup(manifest, "test")
    scenes = (conflict, control)
    position_tables = [
        [obstacle_positions(spec, 18) for spec in _specs(scene, lookup)]
        for scene in scenes
    ]
    frames = list(range(13)) + [13] * 4 + list(range(14, 19)) + [18] * 4
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), dpi=120)

    def update(time_step: int):
        for axis, scene, tables in zip(axes, scenes, position_tables, strict=True):
            axis.clear()
            role = scene["pair_role"]
            is_conflict = role == "conflict"
            color = PRIMARY_CONFLICT if is_conflict else PRIMARY_CONTROL
            _draw_base(
                axis,
                problem,
                ("冲突：等待通过" if is_conflict else "控制：直接通过")
                + f"　t={time_step}",
            )
            for index, (obstacle, positions) in enumerate(
                zip(scene["obstacles"], tables, strict=True)
            ):
                route = lookup[obstacle["route_id"]]["route"]
                route_color = color if index == 0 else CONTEXT
                axis.plot(
                    [cell[1] for cell in route],
                    [cell[0] for cell in route],
                    color=route_color,
                    linewidth=4.0 if index == 0 else 1.3,
                    alpha=1.0 if index == 0 else 0.45,
                )
                row, column = positions[time_step]
                axis.scatter(
                    column,
                    row,
                    s=85 if index == 0 else 35,
                    color=route_color if index == 0 else "#ef5350",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=10,
                )
            path = scene["oracle_path"]
            agent_index = min(time_step, len(path) - 1)
            traversed = path[: agent_index + 1]
            axis.plot(
                [cell[1] for cell in traversed],
                [cell[0] for cell in traversed],
                color="#6a1b9a" if is_conflict else "#0d47a1",
                linewidth=2.6,
                zorder=7,
            )
            agent = path[agent_index]
            waiting = agent_index > 0 and path[agent_index] == path[agent_index - 1]
            axis.scatter(
                agent[1],
                agent[0],
                marker="s" if waiting else "o",
                s=110,
                color="#8e24aa" if waiting else "#ffb300",
                edgecolors="black",
                linewidths=1.0,
                zorder=12,
            )
            if waiting:
                axis.text(
                    agent[1] + 0.45,
                    agent[0] - 0.45,
                    "等待1步",
                    color="#6a1b9a",
                    fontsize=10,
                    fontweight="bold",
                    bbox={"boxstyle": "round", "fc": "white", "ec": "#8e24aa"},
                    zorder=13,
                )
            axis.set_xlim(5.5, 16.0)
            axis.set_ylim(14.0, 4.5)
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(
            "同一路线、同样5个障碍，仅关键障碍初始相位不同",
            fontsize=14,
            fontweight="bold",
        )
        return tuple(axes)

    animation = FuncAnimation(fig, update, frames=frames, interval=500, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=2))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="maps/previews/office_clean_v2_v3_comparison"
    )
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    config = load_config(ROOT / "configs/dynamic_office_clean_v3.yaml")
    problem = _load_problem(config)
    old = _load_json(ROOT / "data/dynamic_scenarios/office_40x40_clean_v2.json")
    new = _load_json(
        ROOT / "data/dynamic_scenarios/office_40x40_clean_v3_pairs.json"
    )
    overview = output_dir / "office_clean_v2_v3_design_comparison.png"
    detail = output_dir / "office_clean_v3_pair_detail.png"
    animation = output_dir / "office_clean_v3_pair_dynamics.gif"
    _render_overview(problem, old, new, overview)
    _render_pair_detail(problem, new, detail)
    _render_pair_animation(problem, new, animation)
    print(f"[rendered] {overview}")
    print(f"[rendered] {detail}")
    print(f"[rendered] {animation}")


if __name__ == "__main__":
    main()
