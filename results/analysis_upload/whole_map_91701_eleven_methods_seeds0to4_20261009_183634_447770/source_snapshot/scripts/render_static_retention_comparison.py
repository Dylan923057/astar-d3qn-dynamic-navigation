"""Render the static retained policy for the three dynamic replay strategies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.core.grid import ACTION_NAMES, Action
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.static_grid import StaticGridNavigationEnv
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_problem_layout
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json
from render_dynamic_replay_comparison import (
    STRATEGIES,
    PolicyRollout,
    RolloutFrame,
    _agent,
    _render_final_comparison,
    _render_rollout_animation,
    _reward_config,
)
from render_spatial_replay_comparison import MODEL_FILES, _resolve


def _static_rollout(agent, problem, config: dict, strategy: str) -> PolicyRollout:
    environment = config["environment"]
    env = StaticGridNavigationEnv(
        problem,
        max_steps=int(environment["max_steps"]),
        reward_config=_reward_config(config["reward"]),
        terminate_on_collision=bool(environment["terminate_on_collision"]),
        window_size=int(environment["window_size"]),
        spatial_channels=int(environment["spatial_channels"]),
    )
    state = env.reset()
    frames = [RolloutFrame(env.position, tuple(), "start", None)]
    path = [env.position]
    collisions = []
    wait_steps = 0
    while True:
        action = agent.select_action(state, epsilon=0.0)
        result = env.step(action)
        state = result.observation
        collision_position = None
        if bool(result.info["collision"]):
            collision_position = tuple(result.info["collision_position"] or env.position)
            collisions.append(collision_position)
        if action == int(Action.STAY):
            wait_steps += 1
        path.append(env.position)
        frames.append(
            RolloutFrame(env.position, tuple(), ACTION_NAMES[action], collision_position)
        )
        if result.done:
            break
    return PolicyRollout(
        strategy,
        tuple(frames),
        tuple(path),
        tuple(collisions),
        bool(result.info["reached"]),
        env.steps,
        wait_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retained static policies.")
    parser.add_argument(
        "--config",
        default="configs/dynamic_spatial_generalization_map01.yaml",
    )
    parser.add_argument("--training-seed", type=int, default=3)
    parser.add_argument("--model", choices=tuple(MODEL_FILES), default="final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    parser.add_argument("--interval-ms", type=int, default=80)
    parser.add_argument("--max-animation-frames", type=int, default=200)
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    problem = next(item for item in problems if item.map_id == config["map"]["scene"])
    output_root = _resolve(config["experiment"]["output_root"])
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else output_root
        / "visualizations"
        / f"static_seed_{args.training_seed}_{args.model}"
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
        rollouts.append(_static_rollout(agent, problem, config, strategy))
    rollout_tuple = tuple(rollouts)
    empty_scenario = DynamicScenario(seed=-1, obstacles=tuple())

    render_problem_layout(problem, output_dir / f"{problem.map_id}_static_layout.png")
    _render_final_comparison(
        problem,
        empty_scenario,
        rollout_tuple,
        output_dir / "static_retention_paths.png",
        title=f"Static retention after dynamic training: seed {args.training_seed}",
    )
    _render_rollout_animation(
        problem,
        empty_scenario,
        rollout_tuple,
        output_dir / "static_retention_rollouts.gif",
        args.interval_ms,
        title=f"Static retention rollouts: seed {args.training_seed}",
        max_frames=args.max_animation_frames,
    )
    write_json(
        {
            "map_id": problem.map_id,
            "training_seed": args.training_seed,
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
