from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.envs.dynamic_grid import (
    DynamicGridNavigationEnv,
    build_map01_three_crossing_obstacle_specs,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.training.demo_collector import (
    collect_astar_demonstrations,
    demonstration_file_sha256,
    demonstration_signature,
    save_demonstrations,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json


def reward_from_config(values: dict) -> RewardConfig:
    return RewardConfig(**{key: float(value) for key, value in values.items()})


def select_problem(problems: list, map_index: int | None) -> list:
    if map_index is None:
        return problems
    if not 0 <= map_index < len(problems):
        raise ValueError(
            f"map-index must be between 0 and {len(problems) - 1}; got {map_index}."
        )
    return [problems[map_index]]


def per_map_path(value: str, map_index: int | None, problem, strategy: str | None = None) -> Path:
    if map_index is None:
        if "{" in value or "}" in value:
            raise ValueError(
                "The configured demonstration path is map-specific; pass --map-index."
            )
        return Path(value)
    return Path(
        value.format(
            map_index=map_index,
            map_number=map_index + 1,
            map_id=problem.map_id,
            map_seed=problem.seed,
            strategy=strategy or "static",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect A* replay demonstrations.")
    parser.add_argument("--config", default="configs/random_benchmark.yaml")
    parser.add_argument("--output")
    parser.add_argument(
        "--strategy",
        choices=("uniform", "prefill", "persistent_demo", "dqfd"),
        help="Replay-specific dynamic crossing scenario.",
    )
    parser.add_argument(
        "--map-index",
        type=int,
        help="Collect demonstrations for one zero-based map index.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    train_set = config["map_sets"]["train"]
    map_path = Path(train_set["file"])
    if not map_path.is_absolute():
        map_path = ROOT / map_path
    all_problems = load_problem_set(map_path)
    if len(all_problems) != int(train_set["count"]):
        raise ValueError("Configured train-map count does not match maps.json.")
    if bool(config["experiment"].get("independent_maps", False)) and args.map_index is None:
        parser.error("this experiment requires --map-index so maps stay independent")
    problems = select_problem(all_problems, args.map_index)
    demo_config = config["demonstrations"]
    reward_config = reward_from_config(config["reward"])
    max_steps = int(config["environment"]["max_steps"])
    window_size = int(config["environment"]["window_size"])
    spatial_channels = int(config["environment"].get("spatial_channels", 1))
    observation_mode = str(config["environment"]["observation"])
    environment_kind = str(config["environment"].get("kind", "static"))
    collection_environment = str(
        demo_config.get(
            "collection_environment",
            "dynamic" if environment_kind == "dynamic" else "static",
        )
    )
    strategy = args.strategy or str(config["training"].get("replay_strategy", "uniform"))
    dynamic_specs = {}
    environment_factory = None
    environment_id = None
    if environment_kind == "dynamic":
        if strategy not in {"uniform", "prefill", "persistent_demo"}:
            raise ValueError("Dynamic demonstration collection requires a replay strategy.")
        if collection_environment == "static_nominal":
            environment_id = str(
                demo_config.get(
                    "environment_id", "static_nominal_zero_dynamic_channels"
                )
            )
        elif collection_environment == "dynamic":
            dynamic_specs = {
                problem.map_id: build_map01_three_crossing_obstacle_specs(problem)
                for problem in problems
            }

            def environment_factory(problem, **kwargs):
                return DynamicGridNavigationEnv(
                    problem,
                    dynamic_obstacles=dynamic_specs[problem.map_id],
                    **kwargs,
                )

            environment_id = str(
                demo_config.get("environment_id", "dynamic_three_crossing_shared")
            )
        else:
            raise ValueError(
                "Dynamic demonstration collection_environment must be "
                "'static_nominal' or 'dynamic'."
            )
    elif environment_kind != "static":
        raise ValueError(f"Unknown environment kind: {environment_kind}")
    collection_started_at = perf_counter()
    transitions = collect_astar_demonstrations(
        problems,
        episodes=int(demo_config["episodes"]),
        seed=int(demo_config["seed"]),
        max_steps=max_steps,
        reward_config=reward_config,
        window_size=window_size,
        spatial_channels=spatial_channels,
        environment_factory=environment_factory,
        mask_static_invalid_actions=bool(
            config["environment"].get("mask_static_invalid_actions", False)
        ),
    )
    collection_seconds = perf_counter() - collection_started_at
    output = per_map_path(
        str(args.output or demo_config["file"]), args.map_index, problems[0], strategy
    )
    if not output.is_absolute():
        output = ROOT / output
    save_demonstrations(transitions, output)
    signature = demonstration_signature(
        problems,
        episodes=int(demo_config["episodes"]),
        seed=int(demo_config["seed"]),
        max_steps=max_steps,
        reward_config=reward_config,
        window_size=window_size,
        spatial_channels=spatial_channels,
        observation_mode=observation_mode,
        environment_id=environment_id,
    )
    write_json(
        {
            "format_version": 3,
            **signature,
            "transitions": len(transitions),
            "dataset": output.name,
            "dataset_sha256": demonstration_file_sha256(output),
            "collection_seconds": collection_seconds,
        },
        output.with_suffix(".json"),
    )
    print(
        f"Saved {len(transitions)} transitions from "
        f"{demo_config['episodes']} A* episodes to {output}"
    )
    print(f"A* demonstration collection time: {collection_seconds:.3f} seconds")


if __name__ == "__main__":
    main()
