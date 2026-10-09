from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.maps.io import save_problem_set
from astar_d3qn.maps.random_benchmark import (
    RandomMapConfig,
    build_random_problem_set,
)
from astar_d3qn.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible A*-IDDQN-style random maps."
    )
    parser.add_argument("--config", default="configs/random_benchmark.yaml")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    map_config = RandomMapConfig.from_mapping(config["map"])
    set_config = config["map_sets"]["train"]
    problems = build_random_problem_set(
        seed_start=int(set_config["seed_start"]),
        count=int(set_config["count"]),
        config=map_config,
    )
    output = (
        Path(args.output).resolve()
        if args.output
        else ROOT / str(set_config["file"])
    )
    try:
        config_label = str(config_path.relative_to(ROOT))
    except ValueError:
        config_label = str(config_path)
    save_problem_set(
        problems,
        output,
        metadata={
            "generator": "astar_iddqn_style_random_v1",
            "set": "train",
            "config": config_label,
            "seed_start": int(set_config["seed_start"]),
        },
    )
    print(f"Generated {len(problems)} maps in {output}")
    for problem in problems:
        print(
            f"  {problem.map_id}: obstacles={len(problem.obstacles)}, "
            f"A*={problem.astar_steps}, turns={problem.metadata['turn_count']}"
        )


if __name__ == "__main__":
    main()
