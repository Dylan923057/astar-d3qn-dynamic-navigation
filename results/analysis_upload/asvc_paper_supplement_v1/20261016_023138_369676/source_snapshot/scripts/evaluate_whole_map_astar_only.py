"""Evaluate the frozen static A* path in whole-map dynamic scenes, without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from astar_d3qn.core.grid import action_between
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from train_whole_map_route_pool_pilot import (
    _factory,
    _load_inputs,
    _resolve,
    _reward_config,
    _validate_protocol,
)


class StaticAStarPolicy:
    """Replay the registered static nominal A* path from local goal scalars."""

    def __init__(self, problem) -> None:
        self.problem = problem
        self._next_action = {
            current: int(action_between(current, following))
            for current, following in zip(
                problem.nominal_path,
                problem.nominal_path[1:],
            )
        }

    def select_action(self, state, epsilon=0.0, valid_actions=None) -> int:
        del epsilon
        scale = max(1, self.problem.size - 1)
        position = (
            self.problem.goal[0] - int(round(float(state.scalars[0]) * scale)),
            self.problem.goal[1] - int(round(float(state.scalars[1]) * scale)),
        )
        if position not in self._next_action:
            raise RuntimeError(
                f"Static A* policy left its registered path at {position}."
            )
        action = self._next_action[position]
        if valid_actions is not None and action not in valid_actions:
            raise RuntimeError("Registered static A* action became statically invalid.")
        return action


def _validate_astar_config(config: dict) -> None:
    diagnostic = config.get("diagnostic", {})
    if diagnostic != {
        "kind": "astar_only_dynamic_validation",
        "evaluation_episodes": 50,
        "training_started": False,
    }:
        raise ValueError("A*-only diagnostic protocol changed.")
    if int(config["training"]["max_environment_steps"]) != 200000:
        raise ValueError("Comparison training budget registration must remain 200000.")
    if [int(seed) for seed in config["training"]["seeds"]] != [0, 1]:
        raise ValueError("A*-only comparison must retain registered seeds 0 and 1.")


def run(config: dict, seeds: list[int]) -> None:
    problem, map_entry, pool = _load_inputs(config)
    _validate_protocol(config, problem, map_entry, pool)
    _validate_astar_config(config)
    environment = config["environment"]
    dynamic = config["dynamic_route_pool"]
    reward = _reward_config(config)
    output_root = _resolve(config["experiment"]["output_root"])

    for seed in seeds:
        factory = _factory(
            config,
            problem,
            map_entry,
            pool,
            seed=int(dynamic["validation_sampling_seed"]),
            prefix=f"astar_validation_seed{seed}",
        )
        summary, rows, _ = evaluate_agent(
            StaticAStarPolicy(problem),  # type: ignore[arg-type]
            [problem] * int(config["diagnostic"]["evaluation_episodes"]),
            max_steps=int(environment["max_steps"]),
            reward_config=reward,
            terminate_on_collision=bool(environment["terminate_on_collision"]),
            window_size=int(environment["window_size"]),
            environment_factory=factory,
            mask_static_invalid_actions=bool(
                environment["mask_static_invalid_actions"]
            ),
        )
        run_dir = output_root / f"{config['experiment']['run_name_prefix']}_seed_{seed}"
        if run_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing A*-only output: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_records_csv(rows, run_dir / "evaluation_episodes.csv")
        write_json(summary, run_dir / "summary.json")
        write_json(
            {
                "experiment": config["experiment"]["name"],
                "method": "static_astar_only",
                "comparison_seed": seed,
                "training_started": False,
                "registered_training_budget_environment_steps": int(
                    config["training"]["max_environment_steps"]
                ),
                "evaluation_scene_count": len(rows),
                "map": problem.manifest(),
                "route_pool_protocol": pool["protocol"],
                "composition": dynamic["composition"],
                "summary": summary,
                "config": config,
            },
            run_dir / "result.json",
        )
        print(
            f"[A*-only seed={seed}] safe_success={summary['safe_success_rate']:.1%} "
            f"dynamic_collision={summary['dynamic_collision_rate']:.1%} "
            f"encounter={summary['encounter_rate']:.1%} "
            f"conflict_opportunity={summary['conflict_opportunity_rate']:.1%} "
            f"output={run_dir}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate static A* on the registered whole-map dynamic scenes."
    )
    parser.add_argument(
        "--config",
        default="configs/whole_map_route_pool_91701_astar_only_v1.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if args.seeds != [0, 1]:
        raise ValueError("A*-only diagnostic is registered for seeds 0 and 1.")
    run(load_config(_resolve(args.config)), args.seeds)


if __name__ == "__main__":
    main()
