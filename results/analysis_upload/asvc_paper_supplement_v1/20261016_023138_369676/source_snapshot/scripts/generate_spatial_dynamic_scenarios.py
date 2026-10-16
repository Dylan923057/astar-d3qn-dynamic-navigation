from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from astar_d3qn.envs.spatial_scenarios import (
    SPLITS,
    build_spatial_scenario_manifest,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json


COLORS = {"train": "#1976d2", "validation": "#ef6c00", "test": "#c2185b"}


def _resolve(path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def _grid(problem) -> np.ndarray:
    grid = np.zeros((problem.size, problem.size), dtype=np.uint8)
    for row, column in problem.obstacles:
        grid[row, column] = 1
    return grid


def _draw_base(axis, problem, *, title: str) -> None:
    axis.imshow(_grid(problem), cmap="Greys", origin="upper", vmin=0, vmax=1)
    axis.plot(
        [cell[1] for cell in problem.nominal_path],
        [cell[0] for cell in problem.nominal_path],
        color="#5dade2",
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
    )
    axis.scatter(problem.start[1], problem.start[0], c="#2e7d32", s=18, zorder=5)
    axis.scatter(
        problem.goal[1],
        problem.goal[0],
        c="#fbc02d",
        edgecolors="black",
        marker="*",
        s=35,
        zorder=5,
    )
    axis.set_xlim(-0.5, problem.size - 0.5)
    axis.set_ylim(problem.size - 0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=8)


def render_route_pool_overview(problem, manifest: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=180)
    for axis, split in zip(axes, SPLITS):
        pool = manifest["route_pools"][split]
        _draw_base(
            axis,
            problem,
            title=(
                f"{split.title()} route pool | "
                f"corridor={len(pool['corridor'])}, background={len(pool['background'])}"
            ),
        )
        for category in ("corridor", "background"):
            for record in pool[category]:
                route = record["route"]
                axis.plot(
                    [cell[1] for cell in route],
                    [cell[0] for cell in route],
                    color=COLORS[split],
                    linewidth=2.2 if category == "corridor" else 1.5,
                    linestyle="-" if category == "corridor" else ":",
                    alpha=0.9,
                )
                center = record["center"]
                axis.scatter(
                    center[1],
                    center[0],
                    c=COLORS[split],
                    marker="o" if category == "corridor" else "s",
                    s=12,
                    zorder=4,
                )
    legend = [
        Line2D([0], [0], color="#455a64", linewidth=2.2, label="Corridor route"),
        Line2D([0], [0], color="#455a64", linestyle=":", label="Background route"),
        Line2D([0], [0], color="#5dade2", linestyle="--", label="Nominal A*"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{problem.map_id} spatially disjoint dynamic route pools",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_validation_gallery(problem, manifest: dict, output: Path) -> None:
    scenarios = manifest["scenarios"]["validation"]
    lookup = {
        record["route_id"]: record
        for category in ("corridor", "background")
        for record in manifest["route_pools"]["validation"][category]
    }
    columns = 5
    rows = (len(scenarios) + columns - 1) // columns
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    for axis, scenario in zip(axes, scenarios):
        _draw_base(
            axis,
            problem,
            title=f"Validation scenario {scenario['scenario_id']}",
        )
        for obstacle in scenario["obstacles"]:
            record = lookup[obstacle["route_id"]]
            route = record["route"]
            axis.plot(
                [cell[1] for cell in route],
                [cell[0] for cell in route],
                color=("#ef6c00" if record["category"] == "corridor" else "#00897b"),
                linewidth=1.8,
                linestyle="-" if record["category"] == "corridor" else ":",
            )
            initial = route[int(obstacle["start_index"])]
            axis.scatter(initial[1], initial[0], c="#d32f2f", s=14, zorder=5)
    for axis in axes[len(scenarios) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"{problem.map_id}: twenty validation layouts unseen during training",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate spatially disjoint dynamic scenarios for the configured map."
    )
    parser.add_argument(
        "--config",
        default="configs/dynamic_spatial_generalization_map01.yaml",
    )
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Render the existing manifest without regenerating or overwriting it.",
    )
    args = parser.parse_args()
    config = load_config(_resolve(args.config))
    map_set = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(map_set["file"])))
    if len(problems) != int(map_set["count"]):
        raise ValueError("Configured map count does not match maps.json.")
    map_id = str(config["map"]["scene"])
    matches = [problem for problem in problems if problem.map_id == map_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registered map named {map_id!r}.")
    problem = matches[0]
    values = config["spatial_generalization"]
    manifest_path = _resolve(str(values["manifest"]))
    if args.render_existing:
        manifest = load_json(manifest_path)
        validate_spatial_scenario_manifest(problem, manifest)
    else:
        manifest = build_spatial_scenario_manifest(
            problem,
            generation_seed=int(values["generation_seed"]),
            route_length=int(values["route_length"]),
            reference_path_count=int(values["reference_path_count"]),
            separation_radius=int(values["separation_radius"]),
            route_pool_counts=values["route_pool_counts"],
            scenario_counts=values["scenario_counts"],
            corridor_per_scenario=int(values["corridor_per_scenario"]),
            background_per_scenario=int(values["background_per_scenario"]),
            move_every=int(values["move_every"]),
        )
        write_json(manifest, manifest_path)
    route_preview = _resolve(str(values["route_pool_preview"]))
    validation_preview = _resolve(str(values["validation_preview"]))
    render_route_pool_overview(problem, manifest, route_preview)
    render_validation_gallery(problem, manifest, validation_preview)
    if args.render_existing:
        print(f"Rendered existing manifest from {manifest_path}")
    else:
        print(f"Saved manifest to {manifest_path}")
    print(f"Saved route-pool preview to {route_preview}")
    print(f"Saved validation preview to {validation_preview}")
    print(
        "Scenario counts: "
        + ", ".join(
            f"{split}={len(manifest['scenarios'][split])}" for split in SPLITS
        )
    )


if __name__ == "__main__":
    main()
