"""Render trained replay policies on one frozen spatial test scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.spatial_scenarios import scenarios_from_spatial_manifest
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_dynamic_obstacle_animation, render_problem_layout
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json
from render_dynamic_replay_comparison import (
    STRATEGIES,
    _agent,
    _render_final_comparison,
    _render_rollout_animation,
    _rollout,
)


MODEL_FILES = {
    "final": "model_final.pth",
    "best": "model_best_validation.pth",
    "selected": "model_selected.pth",
}


def _resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three replay policies on a frozen spatial test scenario."
    )
    parser.add_argument(
        "--config",
        default="configs/dynamic_spatial_generalization_map01.yaml",
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--scenario-id", type=int, default=20023)
    parser.add_argument("--model", choices=tuple(MODEL_FILES), default="final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    parser.add_argument("--interval-ms", type=int, default=140)
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    map_id = str(config["map"]["scene"])
    problem = next(item for item in problems if item.map_id == map_id)
    manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))
    test_scenarios = scenarios_from_spatial_manifest(problem, manifest, "test")
    matches = [item for item in test_scenarios if item.seed == args.scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown test scenario_id: {args.scenario_id}.")
    scenario = matches[0]

    output_root = _resolve(config["experiment"]["output_root"])
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else output_root
        / "visualizations"
        / f"scenario_{args.scenario_id}_seed_{args.training_seed}_{args.model}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rollouts = []
    run_name_prefix = str(
        config["experiment"].get(
            "run_name_prefix", "dynamic_spatial_generalization_map01"
        )
    )
    for strategy in STRATEGIES:
        agent = _agent(config, args.training_seed, args.device)
        run_dir = output_root / f"{run_name_prefix}_seed_{args.training_seed}_{strategy}"
        model_path = run_dir / MODEL_FILES[args.model]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}")
        agent.load_weights(model_path)
        rollouts.append(_rollout(agent, problem, scenario, config, strategy))
    rollout_tuple = tuple(rollouts)

    render_problem_layout(problem, output_dir / f"{problem.map_id}_static_layout.png")
    render_dynamic_obstacle_animation(
        problem,
        scenario.obstacles,
        output_dir / f"scenario_{scenario.seed}_dynamic_obstacles.gif",
        frames=32,
        interval_ms=args.interval_ms,
    )
    _render_final_comparison(
        problem,
        scenario,
        rollout_tuple,
        output_dir / "replay_paths.png",
        title=f"Spatial test scenario {scenario.seed}: {args.model} checkpoints",
    )
    _render_rollout_animation(
        problem,
        scenario,
        rollout_tuple,
        output_dir / "replay_rollouts.gif",
        args.interval_ms,
        title=f"Spatial test scenario {scenario.seed}: frozen greedy rollouts",
    )
    write_json(
        {
            "map_id": problem.map_id,
            "training_seed": args.training_seed,
            "scenario_id": scenario.seed,
            "model": args.model,
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
