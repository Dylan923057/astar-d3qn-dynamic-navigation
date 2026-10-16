"""Plot stored selected-policy trajectories for the causal paired benchmark."""

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
from matplotlib.lines import Line2D

from astar_d3qn.envs.spatial_scenarios import scenarios_from_spatial_manifest
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json
from render_dynamic_replay_comparison import (
    STRATEGIES,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    _draw_base,
)


DEFAULT_CONFIG = "configs/dynamic_spatial_generalization_office_causal_paired_v1.yaml"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _problem(config: dict):
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    map_id = str(config["map"]["scene"])
    return next(problem for problem in problems if problem.map_id == map_id)


def _trajectory(path: Path, scenario_id: int) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))
    matches = [record for record in records if int(record["scenario_id"]) == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one trajectory for scenario {scenario_id} in {path}.")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--scenario-id", type=int, default=20001)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    problem = _problem(config)
    manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))
    test_scenarios = scenarios_from_spatial_manifest(problem, manifest, "test")
    matches = [scenario for scenario in test_scenarios if scenario.seed == args.scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown test scenario {args.scenario_id}.")
    scenario = matches[0]
    scenario_record = next(
        record
        for record in manifest["scenarios"]["test"]
        if int(record["scenario_id"]) == args.scenario_id
    )

    output_root = _resolve(config["experiment"]["output_root"])
    prefix = str(config["experiment"]["run_name_prefix"])
    seeds = tuple(int(seed) for seed in config["training"]["seeds"])
    output = (
        _resolve(args.output)
        if args.output
        else output_root
        / "causal_paired_analysis"
        / f"scenario_{args.scenario_id}_all_seeds_paths.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(seeds), len(STRATEGIES), figsize=(15, 4.8 * len(seeds)), dpi=170
    )
    for row, seed in enumerate(seeds):
        for column, strategy in enumerate(STRATEGIES):
            axis = axes[row, column]
            run_dir = output_root / f"{prefix}_seed_{seed}_{strategy}"
            trajectory = _trajectory(
                run_dir / "test_trajectories.json", args.scenario_id
            )
            path = [tuple(cell) for cell in trajectory["path"]]
            _draw_base(axis, problem, scenario)
            color = STRATEGY_COLORS[strategy]
            axis.plot(
                [cell[1] for cell in path],
                [cell[0] for cell in path],
                color=color,
                linewidth=2.4,
                alpha=0.95,
                zorder=5,
            )
            for event in trajectory["wait_events"]:
                position = tuple(event["position"])
                axis.scatter(
                    position[1],
                    position[0],
                    s=min(180, 35 + int(event["duration"]) / 4),
                    marker="o",
                    facecolors="none",
                    edgecolors="#7c3aed",
                    linewidths=2,
                    zorder=8,
                )
            for position in trajectory["collision_positions"]:
                axis.scatter(
                    position[1],
                    position[0],
                    s=90,
                    marker="x",
                    c="#dc2626",
                    linewidths=2.2,
                    zorder=9,
                )
            if not bool(trajectory["success"]):
                endpoint = path[-1]
                axis.scatter(
                    endpoint[1],
                    endpoint[0],
                    s=55,
                    marker="s",
                    c="#dc2626",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=9,
                )
            outcome = "SUCCESS" if bool(trajectory["success"]) else "TIMEOUT"
            axis.set_title(
                f"{STRATEGY_LABELS[strategy]} | seed {seed}\n"
                f"{outcome} | steps={trajectory['steps']} | "
                f"waits={trajectory['wait_steps']} | "
                f"collisions={trajectory['collision_count']}",
                fontsize=10,
                color="#166534" if bool(trajectory["success"]) else "#991b1b",
            )

    condition = str(scenario_record.get("pair_condition", "unknown"))
    exact_steps = scenario_record.get("exact_temporal_conflict_steps", [])
    figure.suptitle(
        f"Office causal-paired test scenario {args.scenario_id} ({condition})\n"
        f"Stored selected-checkpoint trajectories; nominal conflict steps={exact_steps}",
        fontsize=15,
        fontweight="bold",
    )
    legend = [
        Line2D([0], [0], color="#0f766e", linestyle="--", label="Nominal A*"),
        Line2D([0], [0], color="#dc2626", linestyle=":", label="Dynamic route"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#7c3aed",
            linestyle="none",
            label="Wait location",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="#dc2626",
            linestyle="none",
            label="Timeout endpoint",
        ),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved policy path comparison: {output}")


if __name__ == "__main__":
    main()
