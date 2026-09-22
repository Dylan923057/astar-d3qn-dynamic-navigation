from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.envs.dynamic_grid import (
    DynamicGridNavigationEnv,
    build_map01_three_obstacle_specs,
    build_map01_controlled_bottleneck_obstacle_specs,
    build_map01_controlled_six_obstacle_specs,
    build_map01_controlled_mixed_six_obstacle_specs,
    build_crossing_obstacle_spec,
    build_strategy_crossing_obstacle_spec,
    build_map01_three_crossing_obstacle_specs,
)
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.evaluation.checkpoints import checkpoint_metrics, checkpoint_rank
from astar_d3qn.evaluation.diagnostics import demonstration_action_diagnostics
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_final_path
from astar_d3qn.plotting.training_plots import (
    plot_evaluation_curves,
    plot_training_curves,
)
from astar_d3qn.training.demo_collector import (
    demonstration_signature,
    load_demonstrations,
    validate_demonstration_dataset,
)
from astar_d3qn.training.trainer import TrainingConfig, train_d3qn
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything


def reward_from_config(values: dict) -> RewardConfig:
    return RewardConfig(**{key: float(value) for key, value in values.items()})


def select_training_problems(
    problems: list,
    map_index: int | None,
) -> list:
    if map_index is None:
        return problems
    if not 0 <= map_index < len(problems):
        raise ValueError(
            f"map-index must be between 0 and {len(problems) - 1}; got {map_index}."
        )
    return [problems[map_index]]


def per_map_path(
    value: str,
    map_index: int | None,
    problem,
    strategy: str | None = None,
) -> Path:
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


def training_from_config(config: dict, strategy: str) -> TrainingConfig:
    values = config["training"]
    return TrainingConfig(
        episodes=int(values["episodes"]),
        max_steps=int(config["environment"]["max_steps"]),
        replay_capacity=int(values["replay_capacity"]),
        batch_size=int(values["batch_size"]),
        learning_starts=int(values["learning_starts"]),
        updates_per_step=int(values["updates_per_step"]),
        epsilon_start=float(values["epsilon_start"]),
        epsilon_end=float(values["epsilon_end"]),
        epsilon_decay_episodes=int(values["epsilon_decay_episodes"]),
        terminate_on_collision=bool(config["environment"]["terminate_on_collision"]),
        replay_strategy=strategy,
        demo_fraction=float(values["demo_fraction"]),
        window_size=int(config["environment"]["window_size"]),
        progress_interval=int(values.get("progress_interval", 100)),
        diagnostic_interval=int(values.get("diagnostic_interval", 0)),
        per_alpha=float(values.get("per_alpha", 0.6)),
        per_beta_start=float(values.get("per_beta_start", 0.4)),
        per_beta_end=float(values.get("per_beta_end", 1.0)),
        per_priority_epsilon=float(values.get("per_priority_epsilon", 1e-6)),
        demo_margin=float(values.get("demo_margin", 0.8)),
        demo_loss_weight=float(values.get("demo_loss_weight", 1.0)),
        seed=int(config["experiment"]["seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one controlled replay variant.")
    parser.add_argument("--config", default="configs/random_benchmark.yaml")
    parser.add_argument(
        "--strategy",
        choices=("uniform", "prefill", "persistent_demo", "per", "dqfd"),
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the configured training seed for multi-seed experiments.",
    )
    parser.add_argument(
        "--map-index",
        type=int,
        help="Train only one zero-based map index from the configured fixed map set.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    config["experiment"]["seed"] = seed
    seed_everything(seed)
    strategy = args.strategy or str(config["training"]["replay_strategy"])
    train_set = config["map_sets"]["train"]
    map_path = Path(train_set["file"])
    if not map_path.is_absolute():
        map_path = ROOT / map_path
    all_train_problems = load_problem_set(map_path)
    if len(all_train_problems) != int(train_set["count"]):
        raise ValueError("Configured train-map count does not match maps.json.")
    if bool(config["experiment"].get("independent_maps", False)) and args.map_index is None:
        parser.error("this experiment requires --map-index so maps train independently")
    train_problems = select_training_problems(all_train_problems, args.map_index)
    reward_config = reward_from_config(config["reward"])
    environment_values = config["environment"]
    environment_kind = str(environment_values.get("kind", "static"))
    spatial_channels = int(environment_values.get("spatial_channels", 1))
    environment_factory = None
    dynamic_specs = {}
    if environment_kind == "dynamic":
        if spatial_channels != 4:
            raise ValueError("Dynamic benchmark requires four spatial channels.")
        dynamic_config = config["dynamic_obstacle"]
        scenario = str(dynamic_config.get("scenario", "single_crossing"))
        if scenario == "single_crossing":
            if int(dynamic_config["count"]) != 1:
                raise ValueError("single_crossing requires exactly one obstacle.")
            route_length = int(dynamic_config["route_length"])
            dynamic_specs = {
                problem.map_id: (build_crossing_obstacle_spec(problem, route_length),)
                for problem in train_problems
            }
        elif scenario == "map01_three_obstacles":
            if len(train_problems) != 1 or train_problems[0].map_id != "calibration_40x40_map_01":
                raise ValueError(
                    "map01_three_obstacles requires --map-index 0 for calibration_40x40_map_01."
                )
            if int(dynamic_config["count"]) != 3:
                raise ValueError("map01_three_obstacles requires exactly three obstacles.")
            dynamic_specs = {
                train_problems[0].map_id: build_map01_three_obstacle_specs(train_problems[0])
            }
        elif scenario == "strategy_crossing":
            if len(train_problems) != 1 or train_problems[0].map_id != "calibration_40x40_map_01":
                raise ValueError(
                    "strategy_crossing requires --map-index 0 for calibration_40x40_map_01."
                )
            dynamic_specs = {
                train_problems[0].map_id: build_map01_three_crossing_obstacle_specs(
                    train_problems[0]
                )
            }
        elif scenario == "map01_controlled_bottleneck":
            if len(train_problems) != 1 or train_problems[0].map_id != "calibration_40x40_map_01":
                raise ValueError(
                    "map01_controlled_bottleneck requires --map-index 0 for "
                    "calibration_40x40_map_01."
                )
            if int(dynamic_config["count"]) != 3:
                raise ValueError(
                    "map01_controlled_bottleneck requires exactly three obstacles."
                )
            dynamic_specs = {
                train_problems[0].map_id: (
                    build_map01_controlled_bottleneck_obstacle_specs(
                        train_problems[0]
                    )
                )
            }
        elif scenario == "map01_controlled_six_obstacles":
            if len(train_problems) != 1 or train_problems[0].map_id != "calibration_40x40_map_01":
                raise ValueError(
                    "map01_controlled_six_obstacles requires --map-index 0 for "
                    "calibration_40x40_map_01."
                )
            if int(dynamic_config["count"]) != 6:
                raise ValueError(
                    "map01_controlled_six_obstacles requires exactly six obstacles."
                )
            dynamic_specs = {
                train_problems[0].map_id: build_map01_controlled_six_obstacle_specs(
                    train_problems[0]
                )
            }
        elif scenario == "map01_controlled_mixed_six_obstacles":
            if len(train_problems) != 1 or train_problems[0].map_id != "calibration_40x40_map_01":
                raise ValueError(
                    "map01_controlled_mixed_six_obstacles requires --map-index 0 for "
                    "calibration_40x40_map_01."
                )
            if int(dynamic_config["count"]) != 6:
                raise ValueError(
                    "map01_controlled_mixed_six_obstacles requires exactly six obstacles."
                )
            dynamic_specs = {
                train_problems[0].map_id: build_map01_controlled_mixed_six_obstacle_specs(
                    train_problems[0]
                )
            }
        else:
            raise ValueError(f"Unknown dynamic obstacle scenario: {scenario}")

        def environment_factory(problem, **kwargs):
            return DynamicGridNavigationEnv(
                problem,
                dynamic_obstacles=dynamic_specs[problem.map_id],
                scenario_id=scenario,
                **kwargs,
            )
    elif environment_kind != "static":
        raise ValueError(f"Unknown environment kind: {environment_kind}")
    elif spatial_channels != 1:
        raise ValueError("Static local benchmark requires one spatial channel.")

    demonstrations = []
    demonstrations_by_map: dict[str, list] = {
        problem.map_id: [] for problem in train_problems
    }
    demo_generation_seconds = 0.0
    if strategy in {"prefill", "persistent_demo", "dqfd"}:
        demo_config = config["demonstrations"]
        demo_path = per_map_path(
            str(demo_config["file"]), args.map_index, train_problems[0], strategy
        )
        if not demo_path.is_absolute():
            demo_path = ROOT / demo_path
        demonstrations = load_demonstrations(demo_path)
        metadata = load_json(demo_path.with_suffix(".json"))
        demo_generation_seconds = float(metadata.get("collection_seconds", 0.0))
        expected_signature = demonstration_signature(
            train_problems,
            episodes=int(demo_config["episodes"]),
            seed=int(demo_config["seed"]),
            max_steps=int(config["environment"]["max_steps"]),
            reward_config=reward_config,
            window_size=int(config["environment"]["window_size"]),
            spatial_channels=spatial_channels,
            observation_mode=str(config["environment"]["observation"]),
            environment_id=(
                str(demo_config["environment_id"])
                if demo_config.get("environment_id") is not None
                else "dynamic_three_crossing_shared"
                if environment_kind == "dynamic"
                else None
            ),
        )
        validate_demonstration_dataset(
            demonstrations,
            metadata,
            expected_signature,
            demo_path,
        )
        problem_index = 0
        for transition in demonstrations:
            problem = train_problems[problem_index]
            demonstrations_by_map[problem.map_id].append(transition)
            if transition.terminated:
                problem_index = (problem_index + 1) % len(train_problems)
    agent_values = config["agent"]
    agent = D3QNAgent(
        D3QNConfig(
            spatial_shape=(
                spatial_channels,
                int(config["environment"]["window_size"]),
                int(config["environment"]["window_size"]),
            ),
            scalar_dim=2,
            action_dim=int(config["environment"]["action_count"]),
            learning_rate=float(agent_values["learning_rate"]),
            gamma=float(agent_values["gamma"]),
            target_sync_interval=int(agent_values["target_sync_interval"]),
            gradient_clip_norm=float(agent_values["gradient_clip_norm"]),
            hidden_dim=int(agent_values["hidden_dim"]),
            device=args.device or ("cpu" if args.smoke else str(agent_values["device"])),
            seed=seed,
        )
    )
    training_config = training_from_config(config, strategy)
    if args.smoke:
        training_config = replace(
            training_config,
            episodes=4,
            max_steps=20,
            replay_capacity=max(1024, len(demonstrations) + 64),
            batch_size=4,
            learning_starts=64,
            epsilon_decay_episodes=2,
            progress_interval=2,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{strategy}"
    if args.map_index is not None:
        selected_problem = train_problems[0]
        run_name = (
            f"{timestamp}_map_{args.map_index + 1:02d}_seed_"
            f"{selected_problem.seed}_trainseed_{seed}_{strategy}"
        )
    run_dir = ROOT / str(config["experiment"]["output_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_dir / "model_best.pth"
    progress_evaluations: list[dict] = []
    diagnostic_evaluations: list[dict] = []
    best_rank: tuple[float, ...] | None = None
    best_episode: int | None = None

    if len(train_problems) == 1:
        print(
            f"Training {strategy} on map index {args.map_index}: "
            f"{train_problems[0].map_id}",
            flush=True,
        )

    def report_progress(
        episode: int,
        records: list[dict] | tuple[dict, ...],
        full_evaluation: bool,
    ) -> None:
        nonlocal best_episode, best_rank
        report_window = (
            training_config.progress_interval
            if full_evaluation
            else training_config.diagnostic_interval
        )
        recent = list(records[-max(1, report_window) :])
        reward = fmean(float(row["reward"]) for row in recent)
        success = fmean(float(row["success"]) for row in recent)
        collision_episodes = sum(float(row["collision"]) for row in recent)
        collision_count = sum(int(row.get("collision_count", 0)) for row in recent)
        static_collision_count = sum(
            int(row.get("static_collision_count", 0)) for row in recent
        )
        dynamic_collision_count = sum(
            int(row.get("dynamic_collision_count", 0)) for row in recent
        )
        wait_steps = sum(int(row.get("wait_steps", 0)) for row in recent)
        steps = fmean(float(row["steps"]) for row in recent)
        greedy_summary: dict[str, float] = {}
        greedy_details: list[dict] = []
        greedy_trajectories = {}
        metrics: dict[str, float | None] = {}
        checkpoint_saved = False
        periodic_checkpoint_path: Path | None = None
        periodic_path_files: list[str] = []
        if full_evaluation:
            greedy_summary, greedy_details, greedy_trajectories = evaluate_agent(
                agent,
                train_problems,
                max_steps=training_config.max_steps,
                reward_config=reward_config,
                terminate_on_collision=training_config.terminate_on_collision,
                window_size=training_config.window_size,
                environment_factory=environment_factory,
            )
            metrics = checkpoint_metrics(greedy_details)
            rank = checkpoint_rank(metrics)
            checkpoint_saved = best_rank is None or rank > best_rank
            if checkpoint_saved:
                agent.save_weights(best_model_path)
                best_rank = rank
                best_episode = episode
            periodic_checkpoint_path = (
                run_dir / "checkpoints" / f"model_episode_{episode:04d}.pth"
            )
            agent.save_weights(periodic_checkpoint_path)
            greedy_detail_by_map = {row["map_id"]: row for row in greedy_details}
            for problem in train_problems:
                trajectory = greedy_trajectories[problem.map_id]
                periodic_path = (
                    run_dir
                    / "paths"
                    / "periodic"
                    / f"{problem.map_id}_episode_{episode:04d}_greedy_path.png"
                )
                render_final_path(
                    problem=problem,
                    greedy_path=trajectory.path,
                    output_path=periodic_path,
                    strategy=strategy,
                    success=bool(greedy_detail_by_map[problem.map_id]["success"]),
                    path_label=f"EPISODE {episode} GREEDY PATH",
                    collision_positions=trajectory.collision_positions,
                    revisit_positions=trajectory.revisit_positions,
                    wait_events=trajectory.wait_events,
                    dynamic_routes=tuple(spec.route for spec in dynamic_specs[problem.map_id])
                    if problem.map_id in dynamic_specs
                    else (),
                )
                periodic_path_files.append(str(periodic_path.relative_to(run_dir)))
        diagnostics = {
            "demo_transition_count": float(len(demonstrations)),
            "demo_valid_state_count": 0.0,
            "demo_optimal_action_agreement": 0.0,
            "demo_exact_action_agreement": 0.0,
        }
        if demonstrations:
            per_map_diagnostics = []
            for problem in train_problems:
                map_demos = demonstrations_by_map[problem.map_id]
                if map_demos:
                    per_map_diagnostics.append(
                        demonstration_action_diagnostics(agent, map_demos, problem)
                    )
            if per_map_diagnostics:
                diagnostics["demo_valid_state_count"] = sum(
                    item["demo_valid_state_count"] for item in per_map_diagnostics
                )
                diagnostics["demo_optimal_action_agreement"] = sum(
                    item["demo_optimal_action_agreement"]
                    * item["demo_valid_state_count"]
                    for item in per_map_diagnostics
                ) / max(1.0, diagnostics["demo_valid_state_count"])
                diagnostics["demo_exact_action_agreement"] = sum(
                    item["demo_exact_action_agreement"]
                    * item["demo_valid_state_count"]
                    for item in per_map_diagnostics
                ) / max(1.0, diagnostics["demo_valid_state_count"])
        diagnostic_evaluations.append(
            {
                "episode": episode,
                **diagnostics,
                "periodic_checkpoint": (
                    str(periodic_checkpoint_path.relative_to(run_dir))
                    if periodic_checkpoint_path is not None
                    else ""
                ),
                "periodic_greedy_paths": ";".join(periodic_path_files),
            }
        )
        if full_evaluation:
            progress_evaluations.append(
                {
                    "episode": episode,
                    **metrics,
                    "checkpoint_saved": float(checkpoint_saved),
                    "best_checkpoint_episode": best_episode,
                }
            )
        write_records_csv(progress_evaluations, run_dir / "progress_evaluation.csv")
        write_records_csv(diagnostic_evaluations, run_dir / "diagnostics.csv")
        print(
            f"[episode {episode:4d}/{training_config.episodes}] "
            f"recent({len(recent)}): reward={reward:8.2f} "
            f"success={success:6.1%} collision_episodes={collision_episodes:.0f} "
            f"collision_count={collision_count:.0f} "
            f"static={static_collision_count} dynamic={dynamic_collision_count} "
            f"wait_steps={wait_steps} "
            f"steps={steps:6.1f} "
            f"epsilon={float(recent[-1]['epsilon']):.3f} | "
            + (
                f"greedy: success={greedy_summary['success_rate']:.1%} "
                f"safe={greedy_summary['safe_success_rate']:.1%} "
                f"collision_success={greedy_summary['collision_success_rate']:.1%} "
                f"static_collision={greedy_summary['static_collision_rate']:.1%} "
                f"dynamic_collision={greedy_summary['dynamic_collision_rate']:.1%} "
                f"collision={greedy_summary['collision_rate']:.1%} "
                f"steps={greedy_summary['mean_steps']:.1f} "
                f"turns={greedy_summary['mean_turn_count']:.1f} "
                f"wait={greedy_summary['mean_wait_steps']:.1f} "
                f"best_checkpoint={best_episode}"
                f"{' (updated)' if checkpoint_saved else ''}"
                if full_evaluation
                else "diagnostics-only"
            ),
            flush=True,
        )

    result = train_d3qn(
        train_problems,
        agent,
        training_config,
        demonstrations,
        reward_config,
        progress_callback=report_progress,
        environment_factory=environment_factory,
    )
    final_evaluation_started_at = perf_counter()
    final_summary, final_details, final_trajectories = evaluate_agent(
        agent,
        train_problems,
        training_config.max_steps,
        reward_config,
        terminate_on_collision=training_config.terminate_on_collision,
        window_size=training_config.window_size,
        environment_factory=environment_factory,
    )
    final_evaluation_seconds = perf_counter() - final_evaluation_started_at
    final_metrics = checkpoint_metrics(final_details)
    final_rank = checkpoint_rank(final_metrics)
    final_was_progress_evaluated = bool(
        progress_evaluations
        and int(progress_evaluations[-1]["episode"]) == training_config.episodes
    )
    if not final_was_progress_evaluated:
        checkpoint_saved = best_rank is None or final_rank > best_rank
        if checkpoint_saved:
            agent.save_weights(best_model_path)
            best_rank = final_rank
            best_episode = training_config.episodes
        progress_evaluations.append(
            {
                "episode": training_config.episodes,
                **final_metrics,
                "checkpoint_saved": float(checkpoint_saved),
                "best_checkpoint_episode": best_episode,
            }
        )
    agent.save_weights(run_dir / "model_final.pth")

    if not best_model_path.exists() or best_episode is None:
        agent.save_weights(best_model_path)
        best_rank = final_rank
        best_episode = training_config.episodes

    final_detail_by_map = {row["map_id"]: row for row in final_details}
    for problem in train_problems:
        render_final_path(
            problem=problem,
            greedy_path=final_trajectories[problem.map_id].path,
            output_path=run_dir / "paths" / f"{problem.map_id}_final_path.png",
            strategy=strategy,
            success=bool(final_detail_by_map[problem.map_id]["success"]),
            path_label="FINAL PATH",
            collision_positions=final_trajectories[
                problem.map_id
            ].collision_positions,
            revisit_positions=final_trajectories[problem.map_id].revisit_positions,
            wait_events=final_trajectories[problem.map_id].wait_events,
            dynamic_routes=tuple(spec.route for spec in dynamic_specs[problem.map_id])
            if problem.map_id in dynamic_specs
            else (),
        )

    best_evaluation_started_at = perf_counter()
    agent.load_weights(best_model_path)
    best_summary, best_details, best_trajectories = evaluate_agent(
        agent,
        train_problems,
        training_config.max_steps,
        reward_config,
        terminate_on_collision=training_config.terminate_on_collision,
        window_size=training_config.window_size,
        environment_factory=environment_factory,
    )
    best_evaluation_seconds = perf_counter() - best_evaluation_started_at
    best_detail_by_map = {row["map_id"]: row for row in best_details}
    for problem in train_problems:
        render_final_path(
            problem=problem,
            greedy_path=best_trajectories[problem.map_id].path,
            output_path=run_dir / "paths" / f"{problem.map_id}_best_path.png",
            strategy=strategy,
            success=bool(best_detail_by_map[problem.map_id]["success"]),
            path_label="BEST PATH",
            collision_positions=best_trajectories[problem.map_id].collision_positions,
            revisit_positions=best_trajectories[problem.map_id].revisit_positions,
            wait_events=best_trajectories[problem.map_id].wait_events,
            dynamic_routes=tuple(spec.route for spec in dynamic_specs[problem.map_id])
            if problem.map_id in dynamic_specs
            else (),
        )
    write_records_csv(list(result.episode_records), run_dir / "training.csv")
    write_records_csv(progress_evaluations, run_dir / "progress_evaluation.csv")
    write_records_csv(diagnostic_evaluations, run_dir / "diagnostics.csv")
    write_records_csv(final_details, run_dir / "training_map_evaluation.csv")
    write_records_csv(best_details, run_dir / "best_map_evaluation.csv")
    plot_training_curves(
        list(result.episode_records),
        run_dir / "plots" / "training_curves.png",
        title=(
            f"{strategy} | {train_problems[0].map_id} training curves"
            if len(train_problems) == 1
            else f"{strategy} training curves"
        ),
    )
    plot_evaluation_curves(
        progress_evaluations,
        run_dir / "plots" / "greedy_evaluation_curves.png",
        title=(
            f"{strategy} | {train_problems[0].map_id} greedy evaluation"
            if len(train_problems) == 1
            else f"{strategy} greedy evaluation"
        ),
    )
    write_json(
        {
            "strategy": strategy,
            "training_seed": seed,
            "smoke": args.smoke,
            "map_selection": (
                {
                    "index": args.map_index,
                    "number": args.map_index + 1,
                    "map_id": train_problems[0].map_id,
                    "map_seed": train_problems[0].seed,
                    "grid_sha256": train_problems[0].grid_sha256,
                }
                if args.map_index is not None
                else None
            ),
            "demonstration_transitions": len(demonstrations),
            "replay": {
                "strategy": strategy,
                "total_capacity": training_config.replay_capacity,
                "demo_fraction": training_config.demo_fraction,
                "per_alpha": training_config.per_alpha,
                "per_beta_start": training_config.per_beta_start,
                "per_beta_end": training_config.per_beta_end,
                "per_priority_epsilon": training_config.per_priority_epsilon,
                "demo_margin": training_config.demo_margin,
                "demo_loss_weight": training_config.demo_loss_weight,
            },
            "environment_steps": result.environment_steps,
            "gradient_updates": result.gradient_updates,
            "timing": {
                "demo_generation_seconds": demo_generation_seconds,
                "training_seconds": result.training_seconds,
                "progress_evaluation_seconds": result.progress_callback_seconds,
                "final_evaluation_seconds": final_evaluation_seconds,
                "best_evaluation_seconds": best_evaluation_seconds,
                "mean_action_latency_ms": final_summary[
                    "mean_action_latency_ms"
                ],
                "total_algorithm_seconds": demo_generation_seconds
                + result.training_seconds
                + result.progress_callback_seconds
                + final_evaluation_seconds
                + best_evaluation_seconds,
            },
            "checkpointing": {
                "evaluation_interval_episodes": training_config.progress_interval,
                "diagnostic_interval_episodes": training_config.diagnostic_interval,
                "best_episode": best_episode,
                "selection_order": [
                    "higher_success_rate",
                    "shorter_success_path",
                    "fewer_collisions",
                    "fewer_wait_steps",
                    "fewer_revisits",
                    "higher_reward",
                    "earlier_episode_on_exact_tie",
                ],
                "best_model": "model_best.pth",
                "final_model": "model_final.pth",
            },
            "observation": {
                "mode": str(config["environment"]["observation"]),
                "window_size": training_config.window_size,
                "spatial_shape": [
                    spatial_channels,
                    training_config.window_size,
                    training_config.window_size,
                ],
                "scalar_dim": 2,
                "goal_normalization": "map_extent",
            },
            "dynamic_obstacles": {
                map_id: [
                    {
                        "label": spec.label,
                        "route": [list(cell) for cell in spec.route],
                        "start_index": spec.start_index,
                        "direction": spec.direction,
                        "move_every": spec.move_every,
                        "reference_path_source": spec.reference_path_source,
                        "reference_path_index": spec.reference_path_index,
                    }
                    for spec in specs
                ]
                for map_id, specs in dynamic_specs.items()
            },
            "dynamic_scenario": dict(config.get("dynamic_obstacle", {})),
            "time_scale": dict(config.get("time_scale", {})),
            "best_checkpoint_evaluation": best_summary,
            "final_checkpoint_evaluation": final_summary,
            "train_map_evaluation": final_summary,
        },
        run_dir / "summary.json",
    )
    print(f"Run saved to {run_dir}")
    print(f"Best checkpoint: episode {best_episode} | {best_summary}")
    print(f"Final checkpoint: {final_summary}")
    print(
        "Timing: "
        f"demo={demo_generation_seconds:.3f}s, "
        f"training={result.training_seconds:.3f}s, "
        f"progress_evaluation={result.progress_callback_seconds:.3f}s, "
        f"final_evaluation={final_evaluation_seconds:.3f}s, "
        f"best_evaluation={best_evaluation_seconds:.3f}s, "
        f"action_latency={final_summary['mean_action_latency_ms']:.3f}ms"
    )


if __name__ == "__main__":
    main()
