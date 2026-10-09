"""Train on sampled dynamic scenarios and evaluate on held-out scenarios."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.envs.dynamic_scenarios import (
    ScheduledDynamicEnvironmentFactory,
    build_scenario_schedule,
)
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.training.demo_collector import collect_astar_demonstrations
from astar_d3qn.training.trainer import TrainingConfig, train_d3qn
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything


def _reward_config(values: dict) -> RewardConfig:
    return RewardConfig(**{key: float(value) for key, value in values.items()})


def _training_config(config: dict, strategy: str, seed: int) -> TrainingConfig:
    values = config["training"]
    environment = config["environment"]
    return TrainingConfig(
        episodes=int(values["episodes"]),
        max_steps=int(environment["max_steps"]),
        replay_capacity=int(values["replay_capacity"]),
        batch_size=int(values["batch_size"]),
        learning_starts=int(values["learning_starts"]),
        updates_per_step=int(values["updates_per_step"]),
        epsilon_start=float(values["epsilon_start"]),
        epsilon_end=float(values["epsilon_end"]),
        epsilon_decay_episodes=int(values["epsilon_decay_episodes"]),
        terminate_on_collision=bool(environment["terminate_on_collision"]),
        replay_strategy=strategy,
        demo_fraction=float(values["demo_fraction"]),
        window_size=int(environment["window_size"]),
        progress_interval=int(values.get("progress_interval", 100)),
        diagnostic_interval=0,
        per_alpha=float(values.get("per_alpha", 0.6)),
        per_beta_start=float(values.get("per_beta_start", 0.4)),
        per_beta_end=float(values.get("per_beta_end", 1.0)),
        per_priority_epsilon=float(values.get("per_priority_epsilon", 1e-6)),
        demo_margin=float(values.get("demo_margin", 0.8)),
        demo_loss_weight=float(values.get("demo_loss_weight", 1.0)),
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on dynamic train scenarios and evaluate held-out scenarios."
    )
    parser.add_argument("--config", default="configs/dynamic_generalization_map01.yaml")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("uniform", "prefill", "persistent_demo"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    seed = int(args.seed)
    seed_everything(seed)

    train_set = config["map_sets"]["train"]
    map_path = Path(train_set["file"])
    if not map_path.is_absolute():
        map_path = ROOT / map_path
    problems = load_problem_set(map_path)
    if len(problems) != int(train_set["count"]):
        raise ValueError("Configured train-map count does not match maps.json.")
    if not problems or problems[0].map_id != "calibration_40x40_map_01":
        raise ValueError("Dynamic generalization requires Map 1 at train-map index 0.")
    problem = problems[0]

    dynamic = config["dynamic_generalization"]
    train_seed_start = int(dynamic["train_seed_start"])
    train_seed_count = int(dynamic["train_seed_count"])
    eval_seed_start = int(dynamic["eval_seed_start"])
    eval_seed_count = int(dynamic["eval_seed_count"])
    train_seed_values = set(range(train_seed_start, train_seed_start + train_seed_count))
    eval_seed_values = set(range(eval_seed_start, eval_seed_start + eval_seed_count))
    if not train_seed_values or not eval_seed_values:
        raise ValueError("Train and eval scenario seed ranges must be non-empty.")
    if train_seed_values.intersection(eval_seed_values):
        raise ValueError("Train and held-out scenario seed ranges must be disjoint.")
    scenario_kwargs = {
        "obstacle_count": int(dynamic["obstacle_count"]),
        "route_length": int(dynamic["route_length"]),
        "move_every": int(dynamic["move_every"]),
    }
    train_scenarios = build_scenario_schedule(
        problem,
        sorted(train_seed_values),
        **scenario_kwargs,
    )
    eval_scenarios = build_scenario_schedule(
        problem,
        sorted(eval_seed_values),
        **scenario_kwargs,
    )
    environment_factory = ScheduledDynamicEnvironmentFactory(
        train_scenarios,
        eval_scenarios,
    )
    train_eval_factory = ScheduledDynamicEnvironmentFactory(
        train_scenarios,
        eval_scenarios,
    )
    train_eval_count = min(
        len(train_scenarios), int(dynamic.get("train_eval_count", 20))
    )
    train_eval_scenarios = train_scenarios[:train_eval_count]

    reward_config = _reward_config(config["reward"])
    environment = config["environment"]
    agent_values = config["agent"]
    agent = D3QNAgent(
        D3QNConfig(
            spatial_shape=(
                int(environment["spatial_channels"]),
                int(environment["window_size"]),
                int(environment["window_size"]),
            ),
            scalar_dim=2,
            action_dim=int(environment["action_count"]),
            learning_rate=float(agent_values["learning_rate"]),
            gamma=float(agent_values["gamma"]),
            target_sync_interval=int(agent_values["target_sync_interval"]),
            gradient_clip_norm=float(agent_values["gradient_clip_norm"]),
            hidden_dim=int(agent_values["hidden_dim"]),
            device=args.device or str(agent_values["device"]),
            seed=seed,
        )
    )
    training_config = _training_config(config, args.strategy, seed)
    if args.smoke:
        training_config = replace(
            training_config,
            episodes=4,
            max_steps=20,
            batch_size=4,
            learning_starts=4,
            epsilon_decay_episodes=2,
        )

    demonstrations = []
    if args.strategy in {"prefill", "persistent_demo"}:
        print(
            f"[{args.strategy} seed={seed}] collecting nominal A* demonstrations...",
            flush=True,
        )
        environment_factory.set_mode("train")
        demonstrations = collect_astar_demonstrations(
            [problem],
            episodes=int(config["demonstrations"]["episodes"]),
            seed=int(config["demonstrations"]["seed"]),
            max_steps=training_config.max_steps,
            reward_config=reward_config,
            window_size=training_config.window_size,
            spatial_channels=int(environment["spatial_channels"]),
            environment_factory=environment_factory,
        )
        print(
            f"[{args.strategy} seed={seed}] demonstrations={len(demonstrations)}",
            flush=True,
        )

    environment_factory.set_mode("train")
    progress_evaluations: list[dict] = []
    print(
        f"[{args.strategy} seed={seed}] training episodes={training_config.episodes} "
        f"train_scenarios={len(train_scenarios)} heldout_scenarios={len(eval_scenarios)}",
        flush=True,
    )

    def report_progress(episode: int, records: list[dict], full_evaluation: bool) -> None:
        if not full_evaluation:
            return
        train_eval_factory.set_mode("train")
        summary, _, _ = evaluate_agent(
            agent,
            [problem] * len(train_eval_scenarios),
            max_steps=training_config.max_steps,
            reward_config=reward_config,
            terminate_on_collision=training_config.terminate_on_collision,
            window_size=training_config.window_size,
            environment_factory=train_eval_factory,
        )
        progress_evaluations.append({"episode": episode, **summary})
        latest = records[-1]
        print(
            f"[{args.strategy} seed={seed}] episode={episode:4d}/{training_config.episodes} "
            f"env_steps={latest['environment_steps_total']:7.0f} "
            f"epsilon={latest['epsilon']:.3f} "
            f"train_greedy_success={summary['success_rate']:.1%} "
            f"train_static_collision={summary['static_collision_rate']:.1%} "
            f"train_dynamic_collision={summary['dynamic_collision_rate']:.1%} "
            f"train_greedy_steps={summary['mean_steps']:.1f}",
            flush=True,
        )

    result = train_d3qn(
        [problem],
        agent,
        training_config,
        demonstrations=demonstrations,
        reward_config=reward_config,
        progress_callback=report_progress,
        environment_factory=environment_factory,
    )

    environment_factory.set_mode("eval")
    eval_summary, eval_details, _ = evaluate_agent(
        agent,
        [problem] * len(eval_scenarios),
        max_steps=training_config.max_steps,
        reward_config=reward_config,
        terminate_on_collision=training_config.terminate_on_collision,
        window_size=training_config.window_size,
        environment_factory=environment_factory,
    )

    output_root = Path(config["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = output_root / f"dynamic_generalization_map01_seed_{seed}_{args.strategy}"
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.save_weights(run_dir / "model_final.pth")
    write_records_csv(list(result.episode_records), run_dir / "training.csv")
    write_records_csv(progress_evaluations, run_dir / "train_scenario_evaluation.csv")
    write_records_csv(eval_details, run_dir / "heldout_evaluation.csv")
    write_json(
        {
            "strategy": args.strategy,
            "training_seed": seed,
            "map": problem.manifest(),
            "dynamic_scenarios": environment_factory.scenario_manifest(),
            "demonstration_transitions": len(demonstrations),
            "demonstration_semantics": "nominal_static_astar_under_dynamic_interference",
            "training": {
                "environment_steps": result.environment_steps,
                "gradient_updates": result.gradient_updates,
                "training_seconds": result.training_seconds,
            },
            "heldout_summary": eval_summary,
        },
        run_dir / "run_metadata.json",
    )
    print(
        f"{args.strategy} seed={seed} held-out success={eval_summary['success_rate']:.1%} "
        f"static_collision={eval_summary['static_collision_rate']:.1%} "
        f"dynamic_collision={eval_summary['dynamic_collision_rate']:.1%} "
        f"scenarios={len(eval_scenarios)} output={run_dir}"
    )


if __name__ == "__main__":
    main()
