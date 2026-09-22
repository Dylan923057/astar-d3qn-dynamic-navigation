from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.envs.dynamic_grid import (
    DynamicGridNavigationEnv,
    build_crossing_obstacle_spec,
    build_strategy_crossing_obstacle_spec,
    build_map01_three_crossing_obstacle_specs,
)
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_final_path
from astar_d3qn.plotting.training_plots import plot_training_curves
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_records_csv


def read_records(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reward_from_config(values: dict) -> RewardConfig:
    return RewardConfig(**{key: float(value) for key, value in values.items()})


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def selected_run_dirs(root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [resolve_path(ROOT, item) for item in requested]
    result: list[Path] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if not bool(summary.get("smoke", False)):
            result.append(directory)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot completed benchmark runs without retraining."
    )
    parser.add_argument("--config", default="configs/random_benchmark.yaml")
    parser.add_argument("--root", default="outputs/random_benchmark")
    parser.add_argument("--run-dir", action="append")
    parser.add_argument("--device")
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    config = load_config(config_path)
    train_set = config["map_sets"]["train"]
    problems = load_problem_set(resolve_path(ROOT, train_set["file"]))
    if len(problems) != int(train_set["count"]):
        raise ValueError("Configured train-map count does not match maps.json.")

    reward_config = reward_from_config(config["reward"])
    environment_values = config["environment"]
    environment_kind = str(environment_values.get("kind", "static"))
    dynamic_specs = {}
    environment_factory = None
    if environment_kind == "dynamic":
        if int(environment_values.get("spatial_channels", 1)) != 4:
            raise ValueError("Dynamic benchmark requires four spatial channels.")
        scenario = str(config["dynamic_obstacle"].get("scenario", "single_crossing"))
        if scenario == "strategy_crossing":
            dynamic_specs = {}
        else:
            if int(config["dynamic_obstacle"]["count"]) != 1:
                raise ValueError("Dynamic benchmark v1 requires exactly one obstacle.")
            route_length = int(config["dynamic_obstacle"]["route_length"])
            dynamic_specs = {
                problem.map_id: (build_crossing_obstacle_spec(problem, route_length),)
                for problem in problems
            }

        def environment_factory(problem, **kwargs):
            return DynamicGridNavigationEnv(
                problem,
                dynamic_obstacles=dynamic_specs[problem.map_id],
                **kwargs,
            )
    elif environment_kind != "static":
        raise ValueError(f"Unknown environment kind: {environment_kind}")
    agent_values = config["agent"]
    run_root = resolve_path(ROOT, args.root)
    run_dirs = selected_run_dirs(run_root, args.run_dir)
    if not run_dirs:
        raise ValueError("No completed non-smoke run directories were found.")

    for run_dir in run_dirs:
        summary = load_json(run_dir / "summary.json")
        strategy = str(summary["strategy"])
        if environment_kind == "dynamic" and scenario == "strategy_crossing":
            if len(problems) != 1 or problems[0].map_id != "calibration_40x40_map_01":
                raise ValueError("strategy_crossing plotting requires Map 1.")
            dynamic_specs = {
                problems[0].map_id: build_map01_three_crossing_obstacle_specs(
                    problems[0]
                )
            }
        run_problems = problems
        map_selection = summary.get("map_selection")
        if map_selection is not None:
            selected_map_id = str(map_selection["map_id"])
            run_problems = [
                problem for problem in problems if problem.map_id == selected_map_id
            ]
            if len(run_problems) != 1:
                raise ValueError(
                    f"Run {run_dir} refers to unknown map {selected_map_id!r}."
                )
        records = read_records(run_dir / "training.csv")
        plot_training_curves(
            records,
            run_dir / "plots" / "training_curves.png",
            title=(
                f"{strategy} | {run_problems[0].map_id} training curves"
                if len(run_problems) == 1
                else f"{strategy} training curves"
            ),
        )

        observation = summary.get("observation")
        if observation is None:
            spatial_shape = (3, problems[0].size, problems[0].size)
            scalar_dim = 0
            window_size = None
        else:
            spatial_shape = tuple(int(value) for value in observation["spatial_shape"])
            scalar_dim = int(observation["scalar_dim"])
            window_size = int(observation["window_size"])

        agent = D3QNAgent(
            D3QNConfig(
                spatial_shape=spatial_shape,
                scalar_dim=scalar_dim,
                action_dim=int(config["environment"]["action_count"]),
                learning_rate=float(agent_values["learning_rate"]),
                gamma=float(agent_values["gamma"]),
                target_sync_interval=int(agent_values["target_sync_interval"]),
                gradient_clip_norm=float(agent_values["gradient_clip_norm"]),
                hidden_dim=int(agent_values["hidden_dim"]),
                device=args.device or str(agent_values["device"]),
                seed=int(config["experiment"]["seed"]),
            )
        )
        checkpoints = [
            (
                "model_final.pth",
                "FINAL PATH",
                "final_path",
                "training_map_evaluation.csv",
            )
        ]
        if (run_dir / "model_best.pth").exists():
            checkpoints.append(
                (
                    "model_best.pth",
                    "BEST PATH",
                    "best_path",
                    "best_map_evaluation.csv",
                )
            )
        for model_name, path_label, path_suffix, evaluation_name in checkpoints:
            agent.load_weights(run_dir / model_name)
            _, details, trajectories = evaluate_agent(
                agent,
                run_problems,
                max_steps=int(config["environment"]["max_steps"]),
                reward_config=reward_config,
                terminate_on_collision=bool(
                    config["environment"]["terminate_on_collision"]
                ),
                window_size=window_size,
                environment_factory=environment_factory,
            )
            detail_by_map = {row["map_id"]: row for row in details}
            for problem in run_problems:
                trajectory = trajectories[problem.map_id]
                render_final_path(
                    problem=problem,
                    greedy_path=trajectory.path,
                    output_path=run_dir
                    / "paths"
                    / f"{problem.map_id}_{path_suffix}.png",
                    strategy=strategy,
                    success=bool(detail_by_map[problem.map_id]["success"]),
                    path_label=path_label,
                    collision_positions=trajectory.collision_positions,
                    revisit_positions=trajectory.revisit_positions,
                    wait_events=trajectory.wait_events,
                    dynamic_routes=tuple(spec.route for spec in dynamic_specs[problem.map_id])
                    if problem.map_id in dynamic_specs
                    else (),
                )
            write_records_csv(details, run_dir / evaluation_name)
        print(f"Updated plots and paths: {run_dir}")


if __name__ == "__main__":
    main()
