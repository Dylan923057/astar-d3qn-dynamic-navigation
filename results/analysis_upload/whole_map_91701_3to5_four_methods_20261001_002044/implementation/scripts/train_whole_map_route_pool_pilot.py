"""Train the unchanged D3QN loop with episode-sampled whole-map obstacle routes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.envs.whole_map_route_pool import (
    POOL_CATEGORIES,
    WholeMapRoutePoolEnvironmentFactory,
)
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.demo_collector import collect_astar_demonstrations
from astar_d3qn.training.replay_adaptation import decay_demo_fraction
from astar_d3qn.training.trainer import TrainingConfig, TrainingWindowAdaptiveController, train_d3qn
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_inputs(config: dict):
    dataset = config["dataset"]
    source = json.loads(_resolve(dataset["source_manifest"]).read_text(encoding="utf-8"))
    pool = json.loads(_resolve(dataset["route_pool_manifest"]).read_text(encoding="utf-8"))
    if pool.get("protocol") != dataset["route_pool_protocol"]:
        raise ValueError("Unexpected whole-map route-pool protocol.")
    if pool.get("design_sha256") != dataset["route_pool_design_sha256"]:
        raise ValueError("Whole-map route-pool design hash mismatch.")
    map_id = str(dataset["map_id"])
    source_entries = [entry for entry in source["maps"] if entry["problem"]["map_id"] == map_id]
    pool_entries = [
        entry for entry in pool["maps"]
        if entry["problem_reference"]["map_id"] == map_id
    ]
    if len(source_entries) != 1 or len(pool_entries) != 1:
        raise ValueError(f"Expected exactly one source and route-pool entry for {map_id}.")
    problem = problem_from_record(source_entries[0]["problem"])
    return problem, pool_entries[0], pool


def _reward_config(config: dict) -> RewardConfig:
    return RewardConfig(
        **{key: float(value) for key, value in config["reward"].items()}
    )


def _training_config(config: dict, seed: int) -> TrainingConfig:
    values = config["training"]
    environment = config["environment"]
    replay_strategy = str(values["replay_strategy"])
    return TrainingConfig(
        episodes=int(values["episodes"]),
        max_environment_steps=int(values["max_environment_steps"]),
        max_steps=int(environment["max_steps"]),
        replay_capacity=int(values["replay_capacity"]),
        batch_size=int(values["batch_size"]),
        learning_starts=int(values["learning_starts"]),
        updates_per_step=int(values["updates_per_step"]),
        epsilon_start=float(values["epsilon_start"]),
        epsilon_end=float(values["epsilon_end"]),
        epsilon_decay_episodes=int(values["epsilon_decay_episodes"]),
        epsilon_decay_environment_steps=int(
            values["epsilon_decay_environment_steps"]
        ),
        terminate_on_collision=bool(environment["terminate_on_collision"]),
        mask_static_invalid_actions=bool(
            environment.get("mask_static_invalid_actions", False)
        ),
        replay_strategy=replay_strategy,
        demo_fraction=(
            float(values.get("demo_fraction", 0.25))
            if replay_strategy == "persistent_demo"
            else 0.0
        ),
        window_size=int(environment["window_size"]),
        progress_interval=0,
        progress_interval_environment_steps=int(
            values["validation_interval_environment_steps"]
        ),
        diagnostic_interval=0,
        seed=int(seed),
    )


def _factory(config: dict, problem, map_entry: dict, pool: dict, *, seed: int, prefix: str):
    dynamic = config["dynamic_route_pool"]
    acceptance = dict(pool["design"]["scene_acceptance"])
    acceptance["reasonable_path_count"] = int(map_entry["path_ensemble"]["sample_count"])
    return WholeMapRoutePoolEnvironmentFactory(
        problem,
        map_entry,
        composition=dynamic["composition"],
        seed=seed,
        acceptance=acceptance,
        move_every=int(dynamic["move_every"]),
        sampling_trials=int(dynamic["sampling_trials_per_episode"]),
        scenario_prefix=prefix,
        obstacle_counts=tuple(dynamic["obstacle_counts"]) if "obstacle_counts" in dynamic else None,
    )


def _validate_protocol(config: dict, problem, map_entry: dict, pool: dict) -> None:
    training = config["training"]
    dynamic = config["dynamic_route_pool"]
    composition = {key: int(dynamic["composition"].get(key, 0)) for key in POOL_CATEGORIES}
    if problem.map_id != "irregular_workcell_91701":
        raise ValueError("This pilot is locked to irregular_workcell_91701.")
    if int(training["max_environment_steps"]) != 200000:
        raise ValueError("Pilot budget must remain exactly 200000 environment steps.")
    replay_strategy = str(training["replay_strategy"])
    if replay_strategy not in {"uniform", "persistent_demo", "prefill"}:
        raise ValueError(
            "Pilot replay must be uniform, frozen persistent_demo, or prefill."
        )
    if config["experiment"].get("protocol") == "four_method_training_window_v1":
        baseline = load_config(_resolve("configs/whole_map_route_pool_91701_pilot_v1.yaml"))
        for section in ("dataset", "environment", "reward", "agent"):
            if config[section] != baseline[section]:
                raise ValueError(f"Fixed {section} changed.")
        for key, value in baseline["training"].items():
            if key not in {"seeds", "replay_strategy"} and training[key] != value:
                raise ValueError(f"Fixed training setting changed: {key}")
        if training["seeds"] != [0, 1, 2, 3, 4]:
            raise ValueError("All five seeds must be registered.")
        if dynamic["obstacle_counts"] != [3, 4, 5] or composition != dict(zip(POOL_CATEGORIES, (2, 2, 1))):
            raise ValueError("Variable count protocol changed.")
        for key, value in baseline["dynamic_route_pool"].items():
            if key != "composition" and dynamic[key] != value:
                raise ValueError(f"Fixed scene setting changed: {key}")
        method = config["experiment"]["method"]
        if replay_strategy != {"pure": "uniform", "prefill": "prefill", "time_decay": "persistent_demo", "adaptive": "persistent_demo"}[method]:
            raise ValueError("Method/replay mismatch.")
        if method == "pure" and "demonstration" in config:
            raise ValueError("Pure must not collect demonstrations.")
        if method != "pure":
            expected = load_config(_resolve("configs/whole_map_route_pool_91701_time_decay_v1.yaml"))["demonstration"]
            if config["demonstration"] != expected:
                raise ValueError("Existing demo/schedule changed.")
        if config["adaptive"] != {"interval_environment_steps": 10000, "minimum_conflict_episodes": 5, "ema_alpha": 0.3, "lower": 0.60, "upper": 0.95, "rho_max": 0.25, "source": "training_interaction"}:
            raise ValueError("Adaptive specification changed.")
        if int(map_entry["path_ensemble"]["sample_count"]) != 240 or pool.get("training_started") is not False:
            raise ValueError("Frozen pool changed.")
        if [decay_demo_fraction(s) for s in (1, 50000, 50001, 100000, 100001, 200000)] != [0.25, 0.25, 0.1, 0.1, 0, 0]:
            raise ValueError("Time-decay schedule changed.")
        return
    if composition != {
        "high_interaction": 3,
        "alternative_branch": 3,
        "background": 1,
    }:
        raise ValueError("Pilot composition must be exactly 3 HI + 3 AR + 1 BG.")
    if [int(seed) for seed in training["seeds"]] != [0, 1]:
        raise ValueError("Pilot config must register training seeds 0 and 1.")
    if int(dynamic["move_every"]) != 1:
        raise ValueError("Pilot obstacles must move continuously every environment step.")
    for category, expected in composition.items():
        if len(map_entry["route_pool"][category]) < expected:
            raise ValueError(f"Insufficient {category} routes for pilot composition.")
    if int(map_entry["path_ensemble"]["sample_count"]) != 240:
        raise ValueError("Pilot expects the frozen 240-path reasonable-path ensemble.")
    if pool.get("training_started") is not False:
        raise ValueError("The source route-pool artifact must remain the frozen design artifact.")
    if replay_strategy == "persistent_demo":
        _validate_time_decay_comparison(config)
    elif replay_strategy == "prefill":
        _validate_prefill_comparison(config)


def _validate_time_decay_comparison(config: dict) -> None:
    experiment = config["experiment"]
    baseline = load_config(_resolve(experiment["comparison_baseline_config"]))
    for section in ("dataset", "dynamic_route_pool", "environment", "reward"):
        if config[section] != baseline[section]:
            raise ValueError(f"Time-decay {section} differs from the No-demo baseline.")
    current_agent = dict(config["agent"])
    baseline_agent = dict(baseline["agent"])
    current_agent.pop("device", None)
    baseline_agent.pop("device", None)
    if current_agent != baseline_agent:
        raise ValueError("Time-decay agent settings differ from the No-demo baseline.")
    current_training = dict(config["training"])
    baseline_training = dict(baseline["training"])
    if baseline_training.pop("replay_strategy") != "uniform":
        raise ValueError("Comparison baseline is no longer the registered No-demo run.")
    if current_training.pop("replay_strategy") != "persistent_demo":
        raise ValueError("Time-decay must use persistent_demo replay.")
    diagnostic = config.get("diagnostic", {})
    if diagnostic.get("kind") == "capacity_controlled_time_decay":
        expected_demos = int(diagnostic["expected_demo_transitions"])
        online_capacity = int(diagnostic["online_replay_capacity"])
        total_capacity = int(diagnostic["total_replay_capacity"])
        if expected_demos != 1320 or online_capacity != 10000:
            raise ValueError("Capacity-controlled Time-decay protocol changed.")
        if total_capacity != online_capacity + expected_demos:
            raise ValueError("Total replay capacity must equal online plus demos.")
        if int(current_training.pop("replay_capacity")) != total_capacity:
            raise ValueError("Capacity-controlled total replay must be 11320.")
        if int(baseline_training.pop("replay_capacity")) != online_capacity:
            raise ValueError("No-demo comparison replay capacity must remain 10000.")
    if current_training != baseline_training:
        raise ValueError("Time-decay training settings differ from No-demo baseline.")
    demonstration = config.get("demonstration", {})
    if demonstration != {
        "source": "static_randomized_tie_astar",
        "episodes": 20,
        "seed": 7400,
        "schedule": {
            "steps_1_to_50000": 0.25,
            "steps_50001_to_100000": 0.10,
            "steps_100001_to_200000": 0.0,
        },
    }:
        raise ValueError("Time-decay A* demonstration protocol changed.")
    checkpoints = (1, 50_000, 50_001, 100_000, 100_001, 200_000)
    if [decay_demo_fraction(step) for step in checkpoints] != [
        0.25, 0.25, 0.10, 0.10, 0.0, 0.0
    ]:
        raise ValueError(
            "Registered Time-decay implementation is no longer 25% -> 10% -> 0%."
        )


def _validate_prefill_comparison(config: dict) -> None:
    experiment = config["experiment"]
    baseline = load_config(_resolve(experiment["comparison_baseline_config"]))
    for section in ("dataset", "dynamic_route_pool", "environment", "reward"):
        if config[section] != baseline[section]:
            raise ValueError(f"Prefill-only {section} differs from No-demo baseline.")
    current_agent = dict(config["agent"])
    baseline_agent = dict(baseline["agent"])
    current_agent.pop("device", None)
    baseline_agent.pop("device", None)
    if current_agent != baseline_agent:
        raise ValueError("Prefill-only agent settings differ from No-demo baseline.")
    current_training = dict(config["training"])
    baseline_training = dict(baseline["training"])
    if current_training.pop("replay_strategy") != "prefill":
        raise ValueError("Prefill-only diagnostic must use ordinary prefill replay.")
    if baseline_training.pop("replay_strategy") != "uniform":
        raise ValueError("Comparison baseline is no longer the registered No-demo run.")
    if current_training != baseline_training:
        raise ValueError("Prefill-only training settings differ from No-demo baseline.")
    if config.get("demonstration") != {
        "source": "static_randomized_tie_astar",
        "episodes": 20,
        "seed": 7400,
    }:
        raise ValueError("Prefill-only A* demonstration protocol changed.")
    diagnostic = config.get("diagnostic", {})
    if diagnostic != {
        "kind": "prefill_only",
        "expected_demo_transitions": 1320,
        "online_replay_capacity": 10000,
        "persistent_partition": False,
        "forced_demo_sampling": False,
    }:
        raise ValueError("Prefill-only replay semantics changed.")


def _preflight(config: dict, seeds: list[int]) -> None:
    problem, map_entry, pool = _load_inputs(config)
    _validate_protocol(config, problem, map_entry, pool)
    environment = config["environment"]
    expected = {"high_interaction": 3, "alternative_branch": 3, "background": 1}
    for seed in seeds:
        factory = _factory(
            config,
            problem,
            map_entry,
            pool,
            seed=int(config["dynamic_route_pool"]["training_sampling_seed_offset"]) + seed,
            prefix=f"preflight_seed{seed}",
        )
        first = factory(
            problem,
            max_steps=int(environment["max_steps"]),
            window_size=int(environment["window_size"]),
        )
        before = tuple(first.dynamic_route_ids)
        first.reset()
        first.step(4)
        if tuple(first.dynamic_route_ids) != before:
            raise AssertionError("Obstacle routes changed within an episode.")
        counts = {category: first.dynamic_route_categories.count(category) for category in expected}
        allowed = {(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)}
        variable = "obstacle_counts" in config["dynamic_route_pool"]
        if (tuple(counts.values()) not in allowed if variable else counts != expected or len(first.dynamic_obstacles) != 7):
            raise AssertionError("Sampled pilot scene does not contain 3 HI + 3 AR + 1 BG.")
        second = factory(
            problem,
            max_steps=int(environment["max_steps"]),
            window_size=int(environment["window_size"]),
        )
        if first.scenario_id == second.scenario_id:
            raise AssertionError("Episode factory did not advance to a new scene ID.")
    print(
        "Preflight passed: map/hash fixed, 240-path v2 pool loaded, each episode "
        "samples configured HI/AR/BG counts, routes remain fixed within the episode, "
        "and no training was started.",
        flush=True,
    )


def _run_seed(config: dict, seed: int) -> None:
    problem, map_entry, pool = _load_inputs(config)
    _validate_protocol(config, problem, map_entry, pool)
    seed_everything(seed)
    training_config = _training_config(config, seed)
    reward_config = _reward_config(config)
    environment = config["environment"]
    dynamic = config["dynamic_route_pool"]
    agent_values = config["agent"]
    method = config["experiment"].get("method")
    uses_adaptive = method == "adaptive"
    uses_persistent = training_config.replay_strategy == "persistent_demo"
    uses_time_decay = uses_persistent and not uses_adaptive
    controller = TrainingWindowAdaptiveController() if uses_adaptive else None
    uses_demonstrations = training_config.replay_strategy in {
        "persistent_demo",
        "prefill",
    }
    demonstrations = (
        collect_astar_demonstrations(
            [problem],
            int(config["demonstration"]["episodes"]),
            int(config["demonstration"]["seed"]),
            max_steps=training_config.max_steps,
            reward_config=reward_config,
            window_size=training_config.window_size,
            spatial_channels=int(environment["spatial_channels"]),
            mask_static_invalid_actions=training_config.mask_static_invalid_actions,
        )
        if uses_demonstrations
        else ()
    )
    expected_demo_count = int(
        config.get("diagnostic", {}).get("expected_demo_transitions", 1320)
    )
    if uses_demonstrations and len(demonstrations) != expected_demo_count:
        raise ValueError(
            f"Expected {expected_demo_count} A* demo transitions, got "
            f"{len(demonstrations)}."
        )

    train_factory = _factory(
        config,
        problem,
        map_entry,
        pool,
        seed=int(dynamic["training_sampling_seed_offset"]) + seed,
        prefix=f"train_seed{seed}",
    )
    validation_factory = _factory(
        config,
        problem,
        map_entry,
        pool,
        seed=int(dynamic["validation_sampling_seed"]),
        prefix="validation",
    )
    device = str(agent_values["device"])
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
            device=device,
            seed=seed,
        )
    )

    output_root = _resolve(config["experiment"]["output_root"])
    run_dir = output_root / method / f"seed_{seed}" if method else output_root / f"{config['experiment']['run_name_prefix']}_seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot output: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    validations: list[dict] = []
    validation_details: list[dict] = []
    write_json(config, run_dir / "effective_config.json")

    def report_progress(episode: int, records, full_evaluation: bool):
        if not full_evaluation:
            return None
        validation_factory.reset_schedule()
        summary, details, _ = evaluate_agent(
            agent,
            [problem] * int(dynamic["validation_episodes"]),
            max_steps=training_config.max_steps,
            reward_config=reward_config,
            terminate_on_collision=training_config.terminate_on_collision,
            window_size=training_config.window_size,
            environment_factory=validation_factory,
            mask_static_invalid_actions=training_config.mask_static_invalid_actions,
        )
        latest = records[-1]
        row = {
            "episode": episode,
            "environment_steps_total": latest["environment_steps_total"],
            **summary,
        }
        validations.append(row)
        validation_details.extend({"environment_steps_total": latest["environment_steps_total"], **detail} for detail in details)
        write_records_csv(validations, run_dir / "validation_curve.csv")
        write_records_csv(validation_details, run_dir / "validation_details.csv")
        write_records_csv(list(records), run_dir / "training.csv")
        if controller is not None:
            write_records_csv(controller.records, run_dir / "adaptive_schedule.csv")
        write_records_csv(validations, run_dir / "validation_summary.csv")
        print(
            f"[seed={seed}] env_steps={latest['environment_steps_total']:.0f}/"
            f"{training_config.max_environment_steps} "
            f"encounter={summary['encounter_rate']:.1%} "
            f"conflict_opportunity={summary['conflict_opportunity_rate']:.1%} "
            f"dynamic_collision={summary['dynamic_collision_rate']:.1%} "
            f"safe_success={summary['safe_success_rate']:.1%}",
            flush=True,
        )
        return None

    print(
        f"[seed={seed}] starting 200000-step pilot on {problem.map_id}; "
        f"method={method or training_config.replay_strategy}; counts={dynamic.get('obstacle_counts', [7])}",
        flush=True,
    )
    result = train_d3qn(
        [problem],
        agent,
        training_config,
        demonstrations=demonstrations,
        reward_config=reward_config,
        progress_callback=report_progress,
        environment_factory=train_factory,
        demo_fraction_schedule=controller if uses_adaptive else decay_demo_fraction if uses_time_decay else None,
        training_interaction_controller=controller,
    )

    validation_factory.reset_schedule()
    final_summary, final_details, _ = evaluate_agent(
        agent,
        [problem] * int(dynamic["validation_episodes"]),
        max_steps=training_config.max_steps,
        reward_config=reward_config,
        terminate_on_collision=training_config.terminate_on_collision,
        window_size=training_config.window_size,
        environment_factory=validation_factory,
        mask_static_invalid_actions=training_config.mask_static_invalid_actions,
    )
    agent.save_weights(run_dir / "model_final.pth")
    write_records_csv(list(result.episode_records), run_dir / "training.csv")
    write_records_csv(validations, run_dir / "validation_summary.csv")
    write_records_csv(validations, run_dir / "validation_curve.csv")
    write_records_csv(validation_details, run_dir / "validation_details.csv")
    if controller is not None:
        write_records_csv(controller.records, run_dir / "adaptive_schedule.csv")
    write_records_csv(final_details, run_dir / "final_validation_episodes.csv")
    write_json(final_summary, run_dir / "final_validation_summary.json")
    result_record = {
        "experiment": config["experiment"]["name"],
        "method": method or (
            "astar_time_decay_capacity_controlled"
            if config.get("diagnostic", {}).get("kind")
            == "capacity_controlled_time_decay"
            else "astar_time_decay"
            if uses_time_decay
            else "astar_prefill_only"
            if training_config.replay_strategy == "prefill"
            else "no_demo"
        ),
        "training_seed": seed,
        "map": problem.manifest(),
        "route_pool_protocol": pool["protocol"],
        "route_pool_design_sha256": pool["design_sha256"],
        "composition": dynamic["composition"],
        "obstacle_counts": dynamic.get("obstacle_counts", [7]),
        "adaptive": config.get("adaptive") if uses_adaptive else None,
        "environment_steps": result.environment_steps,
        "gradient_updates": result.gradient_updates,
        "demonstration_transition_count": len(demonstrations),
        "configured_replay_capacity": training_config.replay_capacity,
        "effective_online_replay_capacity": (
            training_config.replay_capacity - len(demonstrations)
            if uses_persistent
            else training_config.replay_capacity
        ),
        "persistent_demo_partition": uses_persistent,
        "forced_demo_sampling": uses_persistent,
        "demo_fraction_schedule": (
            config["demonstration"]["schedule"] if uses_time_decay else None
        ),
        "metrics": {
            "encounter_rate": "fraction of decision steps with a dynamic obstacle in the 15x15 local view",
            "conflict_opportunity_rate": "fraction of decision steps where at least one static-valid action conflicts with current/predicted dynamic occupancy",
            "dynamic_collision_rate": "fraction of evaluation episodes with one or more dynamic collisions",
            "safe_success_rate": "fraction of evaluation episodes reaching the goal with no collision",
        },
        "final_validation": final_summary,
        "config": config,
        "artifacts": {
            "training": "training.csv",
            "validation_summary": "validation_summary.csv",
            "final_validation_episodes": "final_validation_episodes.csv",
            "final_validation_summary": "final_validation_summary.json",
            "model": "model_final.pth",
            "validation_curve": "validation_curve.csv",
            "validation_details": "validation_details.csv",
            "adaptive_schedule": "adaptive_schedule.csv" if uses_adaptive else None,
        },
    }
    write_json(result_record, run_dir / "run_metadata.json")
    write_json(result_record, run_dir / "result.json")
    print(
        f"[seed={seed}] complete: environment_steps={result.environment_steps} "
        f"output={run_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated irregular_workcell_91701 whole-map-pool pilot."
    )
    parser.add_argument(
        "--config",
        default="configs/whole_map_route_pool_91701_pilot_v1.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--methods", nargs="+", choices=("pure", "prefill", "time_decay", "adaptive"))
    parser.add_argument("--output-root")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate sampling and lifecycle without creating outputs or training.",
    )
    args = parser.parse_args()
    config = load_config(_resolve(args.config))
    seeds = list(args.seeds if args.seeds is not None else config["training"]["seeds"])
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    registered = {int(seed) for seed in config["training"]["seeds"]}
    if any(seed not in registered for seed in seeds):
        raise ValueError(f"Requested seeds must be registered pilot seeds: {sorted(registered)}")
    if args.device is not None:
        config["agent"]["device"] = args.device
    if args.output_root:
        config["experiment"]["output_root"] = args.output_root
    if config["experiment"].get("protocol") == "four_method_training_window_v1":
        import torch
        torch.set_num_threads(2)
        methods = args.methods or ["pure", "prefill", "time_decay", "adaptive"]
        if len(methods) != len(set(methods)):
            raise ValueError("Duplicate methods.")
        for method in methods:
            current = copy.deepcopy(config)
            current["experiment"]["method"] = method
            current["training"]["replay_strategy"] = {"pure": "uniform", "prefill": "prefill", "time_decay": "persistent_demo", "adaptive": "persistent_demo"}[method]
            if method == "pure":
                current.pop("demonstration", None)
            _validate_protocol(current, *_load_inputs(current))
            for seed in seeds:
                if (_resolve(current["experiment"]["output_root"]) / method / f"seed_{seed}").exists():
                    raise FileExistsError("Refusing to overwrite an existing run.")
        for method in methods:
            current = copy.deepcopy(config)
            current["experiment"]["method"] = method
            current["training"]["replay_strategy"] = {"pure": "uniform", "prefill": "prefill", "time_decay": "persistent_demo", "adaptive": "persistent_demo"}[method]
            if method == "pure":
                current.pop("demonstration", None)
            if args.preflight:
                _preflight(current, seeds)
            else:
                for seed in seeds:
                    _run_seed(current, seed)
        return
    if args.preflight:
        _preflight(config, seeds)
        return
    for seed in seeds:
        _run_seed(config, seed)


if __name__ == "__main__":
    main()
