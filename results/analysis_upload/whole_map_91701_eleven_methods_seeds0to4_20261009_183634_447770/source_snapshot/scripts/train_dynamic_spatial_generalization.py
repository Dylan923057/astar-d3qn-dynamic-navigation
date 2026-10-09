"""Train and test D3QN on spatially disjoint dynamic-obstacle scenarios."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.envs.dynamic_scenarios import (
    CurriculumScheduledDynamicEnvironmentFactory,
    DynamicObstacleCurriculumStage,
    DynamicScenario,
    ScheduledDynamicEnvironmentFactory,
)
from astar_d3qn.envs.spatial_scenarios import (
    scenarios_from_spatial_manifest,
    validate_spatial_scenario_manifest,
)
from astar_d3qn.envs.static_grid import RewardConfig, StaticGridNavigationEnv
from astar_d3qn.evaluation.checkpoints import validation_checkpoint_score
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.plotting.training_plots import (
    plot_evaluation_curves,
    plot_training_curves,
)
from astar_d3qn.training.demo_collector import (
    collect_astar_demonstrations,
    demonstration_file_sha256,
    demonstration_signature,
    load_demonstrations,
    validate_demonstration_dataset,
)
from astar_d3qn.training.trainer import TrainingConfig, train_d3qn
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything


STRATEGIES = (
    "uniform",
    "prefill",
    "persistent_demo",
    "conflict_adaptive_demo",
    "local_conflict_demo",
    "local_counterexample_demo",
    "predictive_margin_demo",
    "safe_guided_demo",
)


def _resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def _reward_config(values: dict) -> RewardConfig:
    return RewardConfig(**{key: float(value) for key, value in values.items()})


def _dynamic_obstacle_curriculum(
    config: dict,
    expected_obstacle_count: int,
) -> tuple[DynamicObstacleCurriculumStage, ...]:
    values = config["training"].get("dynamic_obstacle_curriculum", {})
    if not bool(values.get("enabled", False)):
        return ()
    if config["training"].get("max_environment_steps") is None:
        raise ValueError(
            "The dynamic-obstacle curriculum requires a fixed environment-step budget."
        )
    stages = tuple(
        DynamicObstacleCurriculumStage(
            name=str(item["name"]),
            start_environment_step=int(item["start_environment_step"]),
            obstacle_count=int(item["obstacle_count"]),
            obstacle_indices=(
                tuple(int(index) for index in item["obstacle_indices"])
                if item.get("obstacle_indices") is not None
                else None
            ),
            allowed_difficulties=(
                tuple(str(value) for value in item["allowed_difficulties"])
                if item.get("allowed_difficulties") is not None
                else None
            ),
        )
        for item in values.get("stages", ())
    )
    if not stages:
        raise ValueError("Enabled dynamic-obstacle curriculum has no stages.")
    budget = int(config["training"]["max_environment_steps"])
    if stages[-1].start_environment_step >= budget:
        raise ValueError(
            "The full-difficulty curriculum stage must start before training ends."
        )
    if stages[-1].obstacle_count != expected_obstacle_count:
        raise ValueError(
            "The final curriculum stage must match the configured obstacle count."
        )
    return stages


def _smoke_curriculum(
    stages: tuple[DynamicObstacleCurriculumStage, ...],
    source_budget: int,
    smoke_budget: int,
) -> tuple[DynamicObstacleCurriculumStage, ...]:
    if not stages:
        return ()
    return tuple(
        replace(
            stage,
            start_environment_step=round(
                smoke_budget * stage.start_environment_step / source_budget
            ),
        )
        for stage in stages
    )


def _training_config(config: dict, strategy: str, seed: int) -> TrainingConfig:
    values = config["training"]
    environment = config["environment"]
    return TrainingConfig(
        episodes=int(values["episodes"]),
        max_environment_steps=(
            int(values["max_environment_steps"])
            if values.get("max_environment_steps") is not None
            else None
        ),
        max_steps=int(environment["max_steps"]),
        replay_capacity=int(values["replay_capacity"]),
        batch_size=int(values["batch_size"]),
        learning_starts=int(values["learning_starts"]),
        updates_per_step=int(values["updates_per_step"]),
        epsilon_start=float(values["epsilon_start"]),
        epsilon_end=float(values["epsilon_end"]),
        epsilon_decay_episodes=int(values["epsilon_decay_episodes"]),
        epsilon_decay_environment_steps=(
            int(values["epsilon_decay_environment_steps"])
            if values.get("epsilon_decay_environment_steps") is not None
            else None
        ),
        terminate_on_collision=bool(environment["terminate_on_collision"]),
        mask_static_invalid_actions=bool(
            environment.get("mask_static_invalid_actions", False)
        ),
        replay_strategy=strategy,
        demo_fraction=float(values["demo_fraction"]),
        demo_fraction_final=(
            float(values["demo_fraction_final"])
            if values.get("demo_fraction_final") is not None
            else None
        ),
        demo_fraction_decay_start_episode=int(
            values.get("demo_fraction_decay_start_episode", 0)
        ),
        demo_fraction_decay_end_episode=int(
            values.get("demo_fraction_decay_end_episode", 0)
        ),
        demo_fraction_decay_start_environment_step=(
            int(values["demo_fraction_decay_start_environment_step"])
            if values.get("demo_fraction_decay_start_environment_step") is not None
            else None
        ),
        demo_fraction_decay_end_environment_step=(
            int(values["demo_fraction_decay_end_environment_step"])
            if values.get("demo_fraction_decay_end_environment_step") is not None
            else None
        ),
        window_size=int(environment["window_size"]),
        progress_interval=int(values["validation_interval"]),
        progress_interval_environment_steps=(
            int(values["validation_interval_environment_steps"])
            if values.get("validation_interval_environment_steps") is not None
            else None
        ),
        diagnostic_interval=int(values.get("diagnostic_interval", 0)),
        per_alpha=float(values.get("per_alpha", 0.6)),
        per_beta_start=float(values.get("per_beta_start", 0.4)),
        per_beta_end=float(values.get("per_beta_end", 1.0)),
        per_priority_epsilon=float(values.get("per_priority_epsilon", 1e-6)),
        demo_margin=float(values.get("demo_margin", 0.8)),
        demo_loss_weight=float(values.get("demo_loss_weight", 1.0)),
        conflict_demo_fraction_min=float(
            values.get("conflict_demo_fraction_min", 0.0)
        ),
        conflict_ema_alpha=float(values.get("conflict_ema_alpha", 0.2)),
        conflict_recovery_alpha=(
            float(values["conflict_recovery_alpha"])
            if values.get("conflict_recovery_alpha") is not None
            else None
        ),
        conflict_sensitivity=float(values.get("conflict_sensitivity", 1.0)),
        conflict_event_binary=bool(values.get("conflict_event_binary", False)),
        conflict_predict_next=bool(values.get("conflict_predict_next", True)),
        local_conflict_suppression_steps=int(
            values.get("local_conflict_suppression_steps", 2500)
        ),
        local_conflict_min_sampling_weight=float(
            values.get("local_conflict_min_sampling_weight", 0.05)
        ),
        local_conflict_predict_next=bool(
            values.get("local_conflict_predict_next", True)
        ),
        local_counterexample_capacity=int(
            values.get("local_counterexample_capacity", 2000)
        ),
        local_counterexample_fraction=float(
            values.get("local_counterexample_fraction", 0.25)
        ),
        conflict_margin=float(values.get("conflict_margin", 0.8)),
        conflict_margin_loss_weight=float(
            values.get("conflict_margin_loss_weight", 1.0)
        ),
        conflict_margin_predict_next=bool(
            values.get("conflict_margin_predict_next", True)
        ),
        safe_guidance_margin=float(values.get("safe_guidance_margin", 0.8)),
        safe_guidance_loss_weight=float(
            values.get("safe_guidance_loss_weight", 1.0)
        ),
        safe_guidance_predict_next=bool(
            values.get("safe_guidance_predict_next", True)
        ),
        seed=seed,
    )


def _load_problem(config: dict):
    map_set = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(map_set["file"])))
    if len(problems) != int(map_set["count"]):
        raise ValueError("Configured map count does not match maps.json.")
    map_id = str(config["map"]["scene"])
    matches = [problem for problem in problems if problem.map_id == map_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registered map named {map_id!r}.")
    return matches[0]


def _static_factory(spatial_channels: int):
    def factory(problem, **kwargs):
        return StaticGridNavigationEnv(
            problem,
            spatial_channels=spatial_channels,
            **kwargs,
        )

    return factory


def _scenario_schedules(problem, manifest: dict, spatial: dict):
    """Resolve logical splits, including explicit in-distribution subsets.

    Legacy experiments use each manifest split directly.  A controlled study
    can instead point validation/test at registered training scenarios so
    spatial generalization is not mixed into the mechanism comparison.
    """

    sources = {
        split: str(spatial.get("scenario_split_sources", {}).get(split, split))
        for split in ("train", "validation", "test")
    }
    invalid_sources = sorted(set(sources.values()).difference(("train", "validation", "test")))
    if invalid_sources:
        raise ValueError(f"Unknown scenario split sources: {invalid_sources}.")
    materialized = {
        source: scenarios_from_spatial_manifest(problem, manifest, source)
        for source in set(sources.values())
    }
    configured_ids = spatial.get("scenario_ids", {})
    schedules = {}
    for split in ("train", "validation", "test"):
        source_scenarios = materialized[sources[split]]
        requested = configured_ids.get(split)
        if requested is None:
            schedules[split] = source_scenarios
            continue
        ids = tuple(int(value) for value in requested)
        if len(ids) != len(set(ids)):
            raise ValueError(f"Configured {split!r} scenario ids must be unique.")
        lookup = {scenario.seed: scenario for scenario in source_scenarios}
        missing = [scenario_id for scenario_id in ids if scenario_id not in lookup]
        if missing:
            raise ValueError(
                f"Configured {split!r} scenario ids are absent from source "
                f"{sources[split]!r}: {missing}."
            )
        schedules[split] = tuple(lookup[scenario_id] for scenario_id in ids)
    return schedules, sources


def _scenario_fingerprint(scenario) -> tuple:
    return tuple(
        sorted(
            (
                tuple(spec.route),
                int(spec.start_index),
                int(spec.direction),
                int(spec.move_every),
            )
            for spec in scenario.obstacles
        )
    )


def _validate_split_isolation(
    schedules: dict,
    sources: dict[str, str],
    settings: dict | None,
) -> dict:
    values = dict(settings or {})
    enabled = bool(values.get("enabled", False))
    level = str(values.get("level", "exact_scenario"))
    if level not in {"exact_scenario", "route_geometry"}:
        raise ValueError(
            "split_isolation.level must be 'exact_scenario' or 'route_geometry'."
        )
    audit = {
        "enabled": enabled,
        "level": level,
        "require_distinct_sources": bool(
            values.get("require_distinct_sources", False)
        ),
        "passed": True,
        "pairwise": {},
    }
    if not enabled:
        return audit
    if audit["require_distinct_sources"] and len(set(sources.values())) != 3:
        raise ValueError(
            "Training, validation, and test must use distinct manifest sources."
        )

    split_names = ("train", "validation", "test")
    scenario_ids = {
        split: {int(scenario.seed) for scenario in schedules[split]}
        for split in split_names
    }
    fingerprints = {
        split: {_scenario_fingerprint(scenario) for scenario in schedules[split]}
        for split in split_names
    }
    route_geometries = {
        split: {
            tuple(spec.route)
            for scenario in schedules[split]
            for spec in scenario.obstacles
        }
        for split in split_names
    }
    for left_index, left in enumerate(split_names):
        if len(scenario_ids[left]) != len(schedules[left]):
            raise ValueError(f"The {left!r} schedule contains duplicate scenario ids.")
        if len(fingerprints[left]) != len(schedules[left]):
            raise ValueError(f"The {left!r} schedule contains duplicate scenarios.")
        for right in split_names[left_index + 1 :]:
            shared_ids = scenario_ids[left].intersection(scenario_ids[right])
            shared_scenarios = fingerprints[left].intersection(fingerprints[right])
            shared_routes = route_geometries[left].intersection(
                route_geometries[right]
            )
            audit["pairwise"][f"{left}_vs_{right}"] = {
                "shared_scenario_ids": len(shared_ids),
                "shared_exact_scenarios": len(shared_scenarios),
                "shared_route_geometries": len(shared_routes),
            }
            if shared_ids or shared_scenarios:
                raise ValueError(
                    f"Scenario leakage detected between {left} and {right}: "
                    f"ids={len(shared_ids)}, exact={len(shared_scenarios)}."
                )
            if level == "route_geometry" and shared_routes:
                raise ValueError(
                    f"Route-geometry leakage detected between {left} and {right}: "
                    f"{len(shared_routes)} shared dynamic-obstacle routes."
                )
    return audit


def _serialize_trajectories(details, trajectories) -> list[dict]:
    """Join per-scenario metrics to every retained evaluation trajectory."""

    records = []
    for detail in details:
        key = str(detail["trajectory_key"])
        if key not in trajectories:
            raise RuntimeError(f"Missing evaluation trajectory {key!r}.")
        trajectory = trajectories[key]
        records.append(
            {
                "map_id": detail["map_id"],
                "scenario_id": detail["scenario_id"],
                "trajectory_key": key,
                "success": detail["success"],
                "termination_reason": detail["termination_reason"],
                "steps": detail["steps"],
                "collision_count": detail["collision_count"],
                "static_collision_count": detail["static_collision_count"],
                "dynamic_collision_count": detail["dynamic_collision_count"],
                "wait_steps": detail["wait_steps"],
                "path": [list(cell) for cell in trajectory.path],
                "collision_positions": [
                    list(cell) for cell in trajectory.collision_positions
                ],
                "static_collision_positions": [
                    list(cell) for cell in trajectory.static_collision_positions
                ],
                "dynamic_collision_positions": [
                    list(cell) for cell in trajectory.dynamic_collision_positions
                ],
                "revisit_positions": [
                    list(cell) for cell in trajectory.revisit_positions
                ],
                "wait_events": [
                    {
                        "position": list(event.position),
                        "start_step": event.start_step,
                        "duration": event.duration,
                    }
                    for event in trajectory.wait_events
                ],
            }
        )
    if len(records) != len(trajectories):
        raise RuntimeError(
            "Evaluation trajectory count does not match per-scenario details."
        )
    return records


def _paired_test_metrics(
    details: list[dict],
    scenario_records: dict[int, dict],
    settings: dict | None,
) -> tuple[dict | None, list[dict]]:
    values = dict(settings or {})
    if not bool(values.get("enabled", False)):
        return None, []
    detail_by_id = {int(row["scenario_id"]): row for row in details}
    groups: dict[str, list[dict]] = {}
    for scenario_id, source in scenario_records.items():
        pair_id = source.get("pair_id")
        if not pair_id or scenario_id not in detail_by_id:
            raise ValueError("Paired evaluation requires pair metadata for every test case.")
        groups.setdefault(str(pair_id), []).append(source)
    maximum_control_excess = int(values.get("control_maximum_excess_steps", 0))
    control_requires_no_wait = bool(values.get("control_requires_no_wait", True))
    conflict_requires_wait = bool(values.get("conflict_requires_wait", True))
    records = []
    for pair_id, members in sorted(groups.items()):
        if len(members) != 2:
            raise ValueError(f"Paired evaluation group {pair_id!r} is not a pair.")
        conflict_source = next(
            (row for row in members if row.get("pair_role") == "conflict"), None
        )
        control_source = next(
            (
                row
                for row in members
                if row.get("pair_role") == "matched_control"
            ),
            None,
        )
        if conflict_source is None or control_source is None:
            raise ValueError(f"Pair {pair_id!r} lacks a conflict or control member.")
        conflict = detail_by_id[int(conflict_source["scenario_id"])]
        control = detail_by_id[int(control_source["scenario_id"])]
        control_excess = (
            int(control["steps"])
            - int(control_source["minimum_safe_path_steps"])
            if bool(control["safe_success"])
            else None
        )
        conflict_excess = (
            int(conflict["steps"])
            - int(conflict_source["minimum_safe_path_steps"])
            if bool(conflict["safe_success"])
            else None
        )
        control_efficient = bool(control["safe_success"]) and (
            control_excess is not None
            and control_excess <= maximum_control_excess
        )
        if control_requires_no_wait:
            control_efficient = control_efficient and int(control["wait_steps"]) == 0
        conflict_response = bool(conflict["safe_success"])
        if conflict_requires_wait:
            conflict_response = conflict_response and int(conflict["wait_steps"]) >= 1
        records.append(
            {
                "pair_id": pair_id,
                "difficulty_stratum": conflict_source["difficulty_stratum"],
                "control_scenario_id": control_source["scenario_id"],
                "conflict_scenario_id": conflict_source["scenario_id"],
                "control_safe_success": float(control["safe_success"]),
                "conflict_safe_success": float(conflict["safe_success"]),
                "paired_safe_success": float(
                    bool(control["safe_success"]) and bool(conflict["safe_success"])
                ),
                "control_wait_steps": int(control["wait_steps"]),
                "conflict_wait_steps": int(conflict["wait_steps"]),
                "control_excess_steps": control_excess,
                "conflict_excess_steps": conflict_excess,
                "control_efficient_match": float(control_efficient),
                "conflict_wait_match": float(conflict_response),
                "paired_response_match": float(
                    control_efficient and conflict_response
                ),
                "wait_step_contrast": int(conflict["wait_steps"])
                - int(control["wait_steps"]),
            }
        )
    summary = {"pair_count": len(records)}
    for key in (
        "control_safe_success",
        "conflict_safe_success",
        "paired_safe_success",
        "control_efficient_match",
        "conflict_wait_match",
        "paired_response_match",
        "control_wait_steps",
        "conflict_wait_steps",
        "wait_step_contrast",
    ):
        summary[f"mean_{key}"] = sum(float(row[key]) for row in records) / len(
            records
        )
    for key in ("control_excess_steps", "conflict_excess_steps"):
        observed = [float(row[key]) for row in records if row[key] is not None]
        summary[f"mean_{key}"] = (
            sum(observed) / len(observed) if observed else None
        )
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train on fixed dynamic layouts, validate on unseen positions, "
            "and report the untouched spatial test split."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/dynamic_spatial_generalization_map01_checkpoint_v2.yaml",
    )
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = load_config(config_path)
    seed = int(args.seed)
    planned_seeds = tuple(int(value) for value in config["training"]["seeds"])
    if seed not in planned_seeds:
        raise ValueError(
            f"Training seed {seed} is outside the preregistered seeds {planned_seeds}."
        )
    seed_everything(seed)

    problem = _load_problem(config)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(str(spatial["manifest"]))
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing frozen scenario manifest {manifest_path}. Run "
            "scripts/generate_spatial_dynamic_scenarios.py first."
        )
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = load_json(manifest_path)
    validate_spatial_scenario_manifest(problem, manifest)

    schedules, schedule_sources = _scenario_schedules(
        problem, manifest, spatial
    )
    split_isolation_audit = _validate_split_isolation(
        schedules,
        schedule_sources,
        spatial.get("split_isolation"),
    )
    for split, expected in spatial["route_pool_counts"].items():
        for category in ("corridor", "background"):
            actual_count = len(manifest["route_pools"][split][category])
            if actual_count != int(expected[category]):
                raise ValueError(
                    f"Manifest {split!r} {category!r} route count does not match config."
                )
    for split, expected in spatial["scenario_counts"].items():
        if len(schedules[split]) != int(expected):
            raise ValueError(f"Manifest {split!r} scenario count does not match config.")
    expected_obstacles = int(spatial["obstacle_count"])
    if any(
        len(scenario.obstacles) != expected_obstacles
        for scenarios in schedules.values()
        for scenario in scenarios
    ):
        raise ValueError("Manifest obstacle count does not match config.")

    gate_values = dict(
        config["training"].get("dynamic_obstacle_curriculum_gate", {})
    )
    gated_curriculum_enabled = bool(gate_values.get("enabled", False))
    curriculum_probe_scenarios: tuple[DynamicScenario, ...] = ()
    train_scenarios = schedules["train"]
    if gated_curriculum_enabled:
        probe_ids = tuple(int(value) for value in gate_values["probe_scenario_ids"])
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("Curriculum probe scenario ids must be unique.")
        lookup = {scenario.seed: scenario for scenario in train_scenarios}
        missing = [scenario_id for scenario_id in probe_ids if scenario_id not in lookup]
        if missing:
            raise ValueError(
                f"Curriculum probe scenarios are absent from train split: {missing}."
            )
        curriculum_probe_scenarios = tuple(lookup[value] for value in probe_ids)
        probe_set = set(probe_ids)
        train_scenarios = tuple(
            scenario for scenario in train_scenarios if scenario.seed not in probe_set
        )
        if not train_scenarios:
            raise ValueError("Curriculum probes cannot consume the full train split.")
    validation_scenarios = schedules["validation"]
    test_scenarios = schedules["test"]
    if args.smoke:
        train_scenarios = train_scenarios[:4]
        if bool(config.get("paired_evaluation", {}).get("enabled", False)):
            validation_scenarios = validation_scenarios[:4]
            test_scenarios = test_scenarios[:4]
        else:
            validation_scenarios = validation_scenarios[:3]
            test_scenarios = test_scenarios[:3]

    validation_factory = ScheduledDynamicEnvironmentFactory(
        train_scenarios,
        validation_scenarios,
    )
    test_factory = ScheduledDynamicEnvironmentFactory(train_scenarios, test_scenarios)

    reward_config = _reward_config(config["reward"])
    environment = config["environment"]
    agent_values = config["agent"]
    spatial_channels = int(environment["spatial_channels"])
    agent = D3QNAgent(
        D3QNConfig(
            spatial_shape=(
                spatial_channels,
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
    curriculum_stages = _dynamic_obstacle_curriculum(config, expected_obstacles)
    if bool(gate_values.get("enabled", False)):
        if not curriculum_stages:
            raise ValueError(
                "dynamic_obstacle_curriculum_gate requires an enabled curriculum."
            )
        registered_gates = tuple(dict(item) for item in gate_values.get("stages", ()))
        expected_gate_names = tuple(stage.name for stage in curriculum_stages[:-1])
        actual_gate_names = tuple(str(item.get("name")) for item in registered_gates)
        if actual_gate_names != expected_gate_names:
            raise ValueError(
                "Curriculum gate names must match every non-final stage in order: "
                f"expected {expected_gate_names}, got {actual_gate_names}."
            )
        for gate in registered_gates:
            threshold = float(gate["safe_success_threshold"])
            minimum_steps = int(gate["minimum_environment_steps"])
            maximum_steps = int(gate["maximum_environment_steps"])
            required_passes = int(gate.get("required_consecutive_passes", 2))
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Curriculum safe-success thresholds must be in [0, 1].")
            if minimum_steps < 0 or maximum_steps <= minimum_steps:
                raise ValueError(
                    "Each curriculum gate needs 0 <= minimum steps < maximum steps."
                )
            if required_passes <= 0:
                raise ValueError(
                    "Curriculum required_consecutive_passes must be positive."
                )
        failure_policy = str(gate_values.get("failure_policy", "force_advance"))
        if failure_policy not in {"force_advance", "stop_run"}:
            raise ValueError(
                "Curriculum gate failure_policy must be 'force_advance' or "
                "'stop_run'."
            )
    if args.smoke:
        smoke_has_episode_demo_decay = (
            training_config.demo_fraction_final is not None
            and training_config.demo_fraction_decay_start_environment_step is None
        )
        smoke_has_step_demo_decay = (
            training_config.demo_fraction_final is not None
            and training_config.demo_fraction_decay_start_environment_step is not None
        )
        smoke_uses_step_budget = training_config.max_environment_steps is not None
        source_step_budget = int(training_config.max_environment_steps or 80)
        training_config = replace(
            training_config,
            episodes=4,
            max_environment_steps=(80 if smoke_uses_step_budget else None),
            max_steps=20,
            batch_size=4,
            learning_starts=4,
            epsilon_decay_episodes=2,
            epsilon_decay_environment_steps=(
                40 if smoke_uses_step_budget else None
            ),
            progress_interval=2,
            progress_interval_environment_steps=(
                (20 if gated_curriculum_enabled else 40)
                if smoke_uses_step_budget
                else None
            ),
            demo_fraction_decay_start_episode=(
                1
                if smoke_has_episode_demo_decay
                else training_config.demo_fraction_decay_start_episode
            ),
            demo_fraction_decay_end_episode=(
                3
                if smoke_has_episode_demo_decay
                else training_config.demo_fraction_decay_end_episode
            ),
            demo_fraction_decay_start_environment_step=(
                round(
                    80
                    * int(training_config.demo_fraction_decay_start_environment_step)
                    / source_step_budget
                )
                if smoke_has_step_demo_decay
                else training_config.demo_fraction_decay_start_environment_step
            ),
            demo_fraction_decay_end_environment_step=(
                round(
                    80
                    * int(training_config.demo_fraction_decay_end_environment_step)
                    / source_step_budget
                )
                if smoke_has_step_demo_decay
                else training_config.demo_fraction_decay_end_environment_step
            ),
        )
        if curriculum_stages:
            curriculum_stages = _smoke_curriculum(
                curriculum_stages,
                int(config["training"]["max_environment_steps"]),
                int(training_config.max_environment_steps or 80),
            )
        if gated_curriculum_enabled:
            scaled_gates = []
            for source_gate in gate_values["stages"]:
                scaled_gate = dict(source_gate)
                scaled_minimum = max(
                    0,
                    round(
                        80
                        * int(source_gate["minimum_environment_steps"])
                        / source_step_budget
                    ),
                )
                scaled_maximum = max(
                    scaled_minimum + 1,
                    round(
                        80
                        * int(source_gate["maximum_environment_steps"])
                        / source_step_budget
                    ),
                )
                scaled_gate["minimum_environment_steps"] = scaled_minimum
                scaled_gate["maximum_environment_steps"] = scaled_maximum
                scaled_gates.append(scaled_gate)
            gate_values = {
                **gate_values,
                "stages": scaled_gates,
                # Smoke tests exercise the complete train/select/test pipeline;
                # formal runs retain the preregistered stop-on-failure policy.
                "failure_policy": "force_advance",
            }

    shuffle_training_scenarios = bool(
        spatial.get("shuffle_training_scenarios", False)
    )
    training_schedule_seed = seed + int(
        spatial.get("training_schedule_seed_offset", 100_000)
    )
    rehearsal_probabilities = config["training"].get(
        "dynamic_obstacle_curriculum_rehearsal_probabilities"
    )
    train_factory = (
        CurriculumScheduledDynamicEnvironmentFactory(
            train_scenarios,
            validation_scenarios,
            curriculum_stages,
            shuffle_train=shuffle_training_scenarios,
            seed=training_schedule_seed,
            manual_stage_control=gated_curriculum_enabled,
            rehearsal_probabilities=rehearsal_probabilities,
        )
        if curriculum_stages
        else ScheduledDynamicEnvironmentFactory(
            train_scenarios,
            validation_scenarios,
            shuffle_train=shuffle_training_scenarios,
            seed=training_schedule_seed,
        )
    )

    output_root = _resolve(str(config["experiment"]["output_root"]))
    suffix = "_smoke" if args.smoke else ""
    run_name_prefix = str(
        config["experiment"].get(
            "run_name_prefix", "dynamic_spatial_generalization_map01"
        )
    )
    run_dir = output_root / f"{run_name_prefix}_seed_{seed}_{args.strategy}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    save_validation_checkpoints = bool(
        config["training"].get("save_validation_checkpoints", False)
    )
    select_best_checkpoint = bool(
        config["training"].get("select_best_checkpoint", False)
    )

    demonstrations = []
    demonstration_dataset: dict | None = None
    if args.strategy in {
        "prefill",
        "persistent_demo",
        "conflict_adaptive_demo",
        "local_conflict_demo",
        "local_counterexample_demo",
        "predictive_margin_demo",
        "safe_guided_demo",
    }:
        if config["demonstrations"].get("collection_environment") != "static_nominal":
            raise ValueError("This experiment requires static_nominal demonstrations.")
        demo_config = config["demonstrations"]
        demo_episodes = int(demo_config["episodes"])
        if bool(demo_config.get("use_frozen_dataset", False)):
            demo_path = _resolve(str(demo_config["file"]))
            metadata_path = demo_path.with_suffix(".json")
            if not demo_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(
                    f"Missing frozen A* demonstrations {demo_path} and metadata. "
                    "Run the clean-study preparation command first."
                )
            print(
                f"[{args.strategy} seed={seed}] loading frozen A* demonstrations...",
                flush=True,
            )
            demonstrations = load_demonstrations(demo_path)
            demo_metadata = load_json(metadata_path)
            expected_signature = demonstration_signature(
                [problem],
                episodes=demo_episodes,
                seed=int(demo_config["seed"]),
                # The smoke flag shortens rollout horizons but must not change
                # the registered signature of the shared frozen dataset.
                max_steps=int(environment["max_steps"]),
                reward_config=reward_config,
                window_size=training_config.window_size,
                spatial_channels=spatial_channels,
                observation_mode=str(environment["observation"]),
                environment_id=str(
                    demo_config.get(
                        "environment_id",
                        "static_nominal_zero_dynamic_channels",
                    )
                ),
            )
            validate_demonstration_dataset(
                demonstrations,
                demo_metadata,
                expected_signature,
                demo_path,
            )
            demonstration_dataset = {
                "frozen": True,
                "path": str(demo_path),
                "metadata_path": str(metadata_path),
                "sha256": demonstration_file_sha256(demo_path),
                "signature": expected_signature,
            }
        else:
            if args.smoke:
                demo_episodes = min(2, demo_episodes)
            print(
                f"[{args.strategy} seed={seed}] collecting static A* demonstrations...",
                flush=True,
            )
            demonstrations = collect_astar_demonstrations(
                [problem],
                episodes=demo_episodes,
                seed=int(demo_config["seed"]),
                max_steps=training_config.max_steps,
                reward_config=reward_config,
                window_size=training_config.window_size,
                spatial_channels=spatial_channels,
                mask_static_invalid_actions=(
                    training_config.mask_static_invalid_actions
                ),
            )
            demonstration_dataset = {
                "frozen": False,
                "path": None,
                "metadata_path": None,
                "sha256": None,
                "signature": None,
            }

    print(
        f"[{args.strategy} seed={seed}] training={len(train_scenarios)} "
        f"validation={len(validation_scenarios)} test={len(test_scenarios)} "
        f"budget="
        f"{training_config.max_environment_steps or training_config.episodes} "
        f"budget_unit="
        f"{'environment_steps' if training_config.max_environment_steps is not None else 'episodes'} "
        f"manifest_sha256={manifest_sha256[:12]}",
        flush=True,
    )
    if curriculum_stages:
        curriculum_text = ", ".join(
            f"{stage.name}@{stage.start_environment_step}:"
            f"{stage.obstacle_count}_obstacles"
            f"{list(stage.obstacle_indices) if stage.obstacle_indices is not None else ''}"
            f":difficulties="
            f"{list(stage.allowed_difficulties) if stage.allowed_difficulties is not None else 'all'}"
            for stage in curriculum_stages
        )
        print(
            f"[{args.strategy} seed={seed}] curriculum={curriculum_text} "
            f"gate={'probe_controlled' if gated_curriculum_enabled else 'fixed_steps'} "
            f"rehearsal={'enabled' if rehearsal_probabilities is not None else 'disabled'} "
            f"validation_test=full_{expected_obstacles}_obstacles",
            flush=True,
        )
    validation_summaries: list[dict] = []
    validation_details: list[dict] = []
    curriculum_probe_summaries: list[dict] = []
    curriculum_transitions: list[dict] = []
    curriculum_gate_streaks: dict[str, int] = {}
    curriculum_gate_failure: dict | None = None
    best_score: tuple[float, ...] | None = None
    best_checkpoint_episode: int | None = None
    best_validation_summary: dict | None = None
    best_checkpoint_path = run_dir / "model_best_validation.pth"

    def run_validation(episode: int, environment_steps: int) -> None:
        nonlocal best_score, best_checkpoint_episode, best_validation_summary
        validation_factory.set_mode("eval")
        summary, details, _ = evaluate_agent(
            agent,
            [problem] * len(validation_scenarios),
            max_steps=training_config.max_steps,
            reward_config=reward_config,
            terminate_on_collision=training_config.terminate_on_collision,
            window_size=training_config.window_size,
            environment_factory=validation_factory,
            mask_static_invalid_actions=(
                training_config.mask_static_invalid_actions
            ),
        )
        score = validation_checkpoint_score(summary)
        is_best = best_score is None or score > best_score
        checkpoint_path = checkpoint_dir / (
            f"step_{environment_steps:07d}_episode_{episode:05d}.pth"
            if training_config.max_environment_steps is not None
            else f"episode_{episode:04d}.pth"
        )
        if save_validation_checkpoints:
            agent.save_weights(checkpoint_path)
        if is_best:
            best_score = score
            best_checkpoint_episode = episode
            best_validation_summary = {
                "episode": episode,
                "environment_steps": environment_steps,
                **summary,
            }
            if select_best_checkpoint:
                agent.save_weights(best_checkpoint_path)
        validation_summaries.append(
            {
                "episode": episode,
                "environment_steps": environment_steps,
                "is_best_checkpoint": float(is_best),
                "checkpoint_path": (
                    str(checkpoint_path) if save_validation_checkpoints else ""
                ),
                **summary,
            }
        )
        validation_details.extend(
            {"episode": episode, **record} for record in details
        )
        if training_config.max_environment_steps is not None:
            progress_text = (
                f"episode={episode:5d} env_steps={environment_steps:7d}/"
                f"{training_config.max_environment_steps}"
            )
        else:
            progress_text = f"episode={episode:4d}/{training_config.episodes}"
        print(
            f"[{args.strategy} seed={seed}] {progress_text} "
            f"validation_success={summary['success_rate']:.1%} "
            f"safe_success={summary['safe_success_rate']:.1%} "
            f"dynamic_collision={summary['dynamic_collision_rate']:.1%}",
            flush=True,
        )

    def run_curriculum_probe(
        episode: int, environment_steps: int
    ) -> bool | None:
        nonlocal curriculum_gate_failure
        if not gated_curriculum_enabled:
            return None
        if not isinstance(
            train_factory, CurriculumScheduledDynamicEnvironmentFactory
        ):
            raise RuntimeError("Gated curriculum requires a curriculum factory.")
        stage_index = train_factory.current_stage_index
        if stage_index >= len(curriculum_stages) - 1:
            return None
        stage = curriculum_stages[stage_index]
        stage_gate_lookup = {
            str(item["name"]): dict(item)
            for item in gate_values.get("stages", ())
        }
        if stage.name not in stage_gate_lookup:
            raise ValueError(f"Missing curriculum gate for stage {stage.name!r}.")
        gate = stage_gate_lookup[stage.name]
        eligible = curriculum_probe_scenarios
        if stage.allowed_difficulties is not None:
            allowed = set(stage.allowed_difficulties)
            eligible = tuple(
                scenario
                for scenario in eligible
                if scenario.difficulty_stratum in allowed
            )
        if not eligible:
            raise ValueError(
                f"Curriculum stage {stage.name!r} has no eligible probe scenarios."
            )
        if stage.obstacle_indices is None:
            select_obstacles = lambda source: source[: stage.obstacle_count]
        else:
            select_obstacles = lambda source: tuple(
                source[index] for index in stage.obstacle_indices
            )
        probe_schedule = tuple(
            DynamicScenario(
                seed=scenario.seed,
                obstacles=select_obstacles(scenario.obstacles),
                required_behavior=scenario.required_behavior,
                difficulty_stratum=scenario.difficulty_stratum,
            )
            for scenario in eligible
        )
        probe_factory = ScheduledDynamicEnvironmentFactory(
            probe_schedule,
            probe_schedule,
        )
        probe_factory.set_mode("eval")
        summary, _, _ = evaluate_agent(
            agent,
            [problem] * len(probe_schedule),
            max_steps=training_config.max_steps,
            reward_config=reward_config,
            terminate_on_collision=training_config.terminate_on_collision,
            window_size=training_config.window_size,
            environment_factory=probe_factory,
            mask_static_invalid_actions=training_config.mask_static_invalid_actions,
        )
        stage_start = train_factory.current_stage_start_environment_step
        elapsed = environment_steps - stage_start
        threshold = float(gate["safe_success_threshold"])
        passed = summary["safe_success_rate"] >= threshold
        streak = curriculum_gate_streaks.get(stage.name, 0)
        streak = streak + 1 if passed else 0
        curriculum_gate_streaks[stage.name] = streak
        minimum_steps = int(gate["minimum_environment_steps"])
        maximum_steps = int(gate["maximum_environment_steps"])
        required_streak = int(gate.get("required_consecutive_passes", 2))
        qualified = elapsed >= minimum_steps and streak >= required_streak
        maximum_reached = elapsed >= maximum_steps and not qualified
        failure_policy = str(gate_values.get("failure_policy", "force_advance"))
        gate_failed = maximum_reached and failure_policy == "stop_run"
        forced = maximum_reached and failure_policy == "force_advance"
        advanced = qualified or forced
        record = {
            "episode": episode,
            "environment_steps": environment_steps,
            "stage": stage.name,
            "stage_index": stage_index,
            "stage_elapsed_environment_steps": elapsed,
            "scenario_count": len(probe_schedule),
            "safe_success_threshold": threshold,
            "required_consecutive_passes": required_streak,
            "consecutive_passes": streak,
            "qualified": float(qualified),
            "maximum_reached": float(maximum_reached),
            "gate_failed": float(gate_failed),
            "forced": float(forced),
            "advanced": float(advanced),
            **summary,
        }
        curriculum_probe_summaries.append(record)
        if advanced:
            next_stage = curriculum_stages[stage_index + 1]
            train_factory.advance_curriculum_stage(environment_steps)
            curriculum_transitions.append(
                {
                    "from_stage": stage.name,
                    "to_stage": next_stage.name,
                    "environment_steps": environment_steps,
                    "episode": episode,
                    "qualified": qualified,
                    "forced": forced,
                    "probe_safe_success_rate": summary["safe_success_rate"],
                }
            )
        elif gate_failed:
            curriculum_gate_failure = {
                "stage": stage.name,
                "stage_index": stage_index,
                "episode": episode,
                "environment_steps": environment_steps,
                "stage_elapsed_environment_steps": elapsed,
                "safe_success_rate": summary["safe_success_rate"],
                "safe_success_threshold": threshold,
                "consecutive_passes": streak,
                "required_consecutive_passes": required_streak,
                "reason": "maximum_stage_duration_reached_without_qualification",
            }
        print(
            f"[{args.strategy} seed={seed}] curriculum_probe={stage.name} "
            f"elapsed={elapsed} safe_success={summary['safe_success_rate']:.1%} "
            f"threshold={threshold:.1%} streak={streak}/{required_streak} "
            f"advanced={advanced} forced={forced} failed={gate_failed}",
            flush=True,
        )
        return not gate_failed

    def report_progress(
        episode: int, records: list[dict], full_evaluation: bool
    ) -> bool | None:
        if full_evaluation:
            environment_steps = int(records[-1]["environment_steps_total"])
            run_validation(episode, environment_steps)
            return run_curriculum_probe(episode, environment_steps)
        return None

    train_factory.set_mode("train")
    result = train_d3qn(
        [problem],
        agent,
        training_config,
        demonstrations=demonstrations,
        reward_config=reward_config,
        progress_callback=report_progress,
        environment_factory=train_factory,
    )
    if result.stopped_early and curriculum_gate_failure is None:
        raise RuntimeError("Training stopped early without a registered gate failure.")
    if curriculum_gate_failure is not None:
        failed_model_path = run_dir / "model_curriculum_failed.pth"
        agent.save_weights(failed_model_path)
        write_records_csv(list(result.episode_records), run_dir / "training.csv")
        write_records_csv(validation_summaries, run_dir / "validation_summary.csv")
        write_records_csv(validation_details, run_dir / "validation_evaluation.csv")
        write_records_csv(
            curriculum_probe_summaries, run_dir / "curriculum_probe_summary.csv"
        )
        plot_training_curves(
            list(result.episode_records),
            run_dir / "plots" / "training_curves.png",
            title=f"{problem.map_id} | {args.strategy} | seed {seed} | failed gate",
        )
        plot_evaluation_curves(
            validation_summaries,
            run_dir / "plots" / "validation_curves.png",
            title=f"{problem.map_id} | {args.strategy} | seed {seed} | validation",
        )
        write_json(
            {
                "status": "curriculum_failed",
                "test_evaluated": False,
                "config_path": str(config_path),
                "strategy": args.strategy,
                "training_seed": seed,
                "smoke": bool(args.smoke),
                "scenario_manifest_sha256": manifest_sha256,
                "environment_steps": result.environment_steps,
                "gradient_updates": result.gradient_updates,
                "completed_episodes": len(result.episode_records),
                "failure": curriculum_gate_failure,
                "gate_configuration": gate_values,
                "transitions": curriculum_transitions,
                "partial_model": str(failed_model_path),
            },
            run_dir / "curriculum_failure.json",
        )
        print(
            f"[{args.strategy} seed={seed}] CURRICULUM FAILED at "
            f"stage={curriculum_gate_failure['stage']} "
            f"environment_steps={result.environment_steps}; test was not evaluated. "
            f"output={run_dir}",
            flush=True,
        )
        return
    final_episode = len(result.episode_records)
    if (
        not validation_summaries
        or int(validation_summaries[-1]["environment_steps"])
        != result.environment_steps
    ):
        run_validation(final_episode, result.environment_steps)

    # Preserve the final policy, then load the validation-selected policy for all
    # untouched test and static-retention evaluations.
    final_model_path = run_dir / "model_final.pth"
    agent.save_weights(final_model_path)
    selected_model_path = final_model_path
    if select_best_checkpoint:
        if best_checkpoint_episode is None or not best_checkpoint_path.exists():
            raise RuntimeError("Best-checkpoint selection was enabled but no model was saved.")
        agent.load_weights(best_checkpoint_path)
        selected_model_path = best_checkpoint_path
    agent.save_weights(run_dir / "model_selected.pth")

    # The test schedule is first evaluated here, after all gradient updates and
    # validation reporting have finished. It never affects learning or selection.
    test_factory.set_mode("eval")
    test_summary, test_details, test_trajectories = evaluate_agent(
        agent,
        [problem] * len(test_scenarios),
        max_steps=training_config.max_steps,
        reward_config=reward_config,
        terminate_on_collision=training_config.terminate_on_collision,
        window_size=training_config.window_size,
        environment_factory=test_factory,
        mask_static_invalid_actions=training_config.mask_static_invalid_actions,
    )
    test_manifest_records = {
        int(source["scenario_id"]): source
        for source in manifest["scenarios"][schedule_sources["test"]]
        if int(source["scenario_id"])
        in {int(scenario.seed) for scenario in test_scenarios}
    }
    for detail in test_details:
        source = test_manifest_records[int(detail["scenario_id"])]
        for key in (
            "pair_id",
            "pair_role",
            "paired_scenario_id",
            "matched_behavior",
            "matched_conflict_difficulty",
            "primary_obstacle_index",
            "primary_route_id",
            "minimum_safe_path_steps",
            "oracle_wait_count",
        ):
            if key in source:
                detail[key] = source[key]
    paired_test_summary, paired_test_records = _paired_test_metrics(
        test_details,
        test_manifest_records,
        config.get("paired_evaluation"),
    )

    static_summary = None
    static_details: list[dict] = []
    if bool(spatial.get("static_retention_evaluation", True)):
        static_summary, static_details, _ = evaluate_agent(
            agent,
            [problem],
            max_steps=training_config.max_steps,
            reward_config=reward_config,
            terminate_on_collision=training_config.terminate_on_collision,
            window_size=training_config.window_size,
            environment_factory=_static_factory(spatial_channels),
            mask_static_invalid_actions=(
                training_config.mask_static_invalid_actions
            ),
        )

    write_records_csv(list(result.episode_records), run_dir / "training.csv")
    write_records_csv(validation_summaries, run_dir / "validation_summary.csv")
    write_records_csv(validation_details, run_dir / "validation_evaluation.csv")
    write_records_csv(
        curriculum_probe_summaries, run_dir / "curriculum_probe_summary.csv"
    )
    write_records_csv(test_details, run_dir / "test_evaluation.csv")
    write_records_csv(paired_test_records, run_dir / "paired_test_evaluation.csv")
    write_records_csv(static_details, run_dir / "static_retention_evaluation.csv")
    write_json(
        _serialize_trajectories(test_details, test_trajectories),
        run_dir / "test_trajectories.json",
    )
    plot_training_curves(
        list(result.episode_records),
        run_dir / "plots" / "training_curves.png",
        title=f"{problem.map_id} | {args.strategy} | seed {seed} | training",
    )
    plot_evaluation_curves(
        validation_summaries,
        run_dir / "plots" / "validation_curves.png",
        title=f"{problem.map_id} | {args.strategy} | seed {seed} | validation",
    )
    write_json(
        {
            "experiment": dict(config["experiment"]),
            "config_path": str(config_path),
            "strategy": args.strategy,
            "training_seed": seed,
            "planned_training_seeds": list(planned_seeds),
            "smoke": bool(args.smoke),
            "map": problem.manifest(),
            "scenario_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "generation": manifest["generation"],
                "evaluation_design": spatial.get(
                    "evaluation_design", "manifest_held_out_splits"
                ),
                "schedule_sources": schedule_sources,
                "split_isolation": split_isolation_audit,
                "shuffle_training_scenarios": shuffle_training_scenarios,
                "training_schedule_seed": training_schedule_seed,
                "scenario_ids": {
                    split: [scenario.seed for scenario in schedules[split]]
                    for split in schedules
                },
                "scenario_counts": {
                    split: len(schedules[split]) for split in schedules
                },
                "optimization_scenario_ids": [
                    scenario.seed for scenario in train_scenarios
                ],
                "optimization_scenario_count": len(train_scenarios),
                "curriculum_probe_scenario_ids": [
                    scenario.seed for scenario in curriculum_probe_scenarios
                ],
                "curriculum_probe_scenario_count": len(
                    curriculum_probe_scenarios
                ),
            },
            "demonstration_transitions": len(demonstrations),
            "demonstration_semantics": "static_nominal_astar_zero_dynamic_channels",
            "demonstration_dataset": demonstration_dataset,
            "demo_fraction_schedule": {
                "initial": training_config.demo_fraction,
                "final": training_config.demo_fraction_final,
                "decay_start_episode": training_config.demo_fraction_decay_start_episode,
                "decay_end_episode": training_config.demo_fraction_decay_end_episode,
                "decay_start_environment_step": (
                    training_config.demo_fraction_decay_start_environment_step
                ),
                "decay_end_environment_step": (
                    training_config.demo_fraction_decay_end_environment_step
                ),
                "conflict_adaptive": args.strategy == "conflict_adaptive_demo",
                "conflict_minimum": training_config.conflict_demo_fraction_min,
                "conflict_ema_alpha": training_config.conflict_ema_alpha,
                "conflict_recovery_alpha": training_config.conflict_recovery_alpha,
                "conflict_sensitivity": training_config.conflict_sensitivity,
                "conflict_event_binary": training_config.conflict_event_binary,
                "conflict_predict_next": training_config.conflict_predict_next,
                "local_conflict_enabled": (
                    args.strategy
                    in {"local_conflict_demo", "local_counterexample_demo"}
                ),
                "local_conflict_suppression_steps": (
                    training_config.local_conflict_suppression_steps
                ),
                "local_conflict_min_sampling_weight": (
                    training_config.local_conflict_min_sampling_weight
                ),
                "local_conflict_predict_next": (
                    training_config.local_conflict_predict_next
                ),
                "local_counterexample_enabled": (
                    args.strategy == "local_counterexample_demo"
                ),
                "local_counterexample_capacity": (
                    training_config.local_counterexample_capacity
                ),
                "local_counterexample_fraction": (
                    training_config.local_counterexample_fraction
                ),
                "conflict_margin_enabled": (
                    args.strategy == "predictive_margin_demo"
                ),
                "conflict_margin": training_config.conflict_margin,
                "conflict_margin_loss_weight": (
                    training_config.conflict_margin_loss_weight
                ),
                "conflict_margin_predict_next": (
                    training_config.conflict_margin_predict_next
                ),
                "safe_guidance_enabled": args.strategy == "safe_guided_demo",
                "safe_guidance_margin": training_config.safe_guidance_margin,
                "safe_guidance_loss_weight": training_config.safe_guidance_loss_weight,
                "safe_guidance_predict_next": training_config.safe_guidance_predict_next,
            },
            "training": {
                "budget_unit": (
                    "environment_steps"
                    if training_config.max_environment_steps is not None
                    else "episodes"
                ),
                "max_environment_steps": training_config.max_environment_steps,
                "configured_episodes": training_config.episodes,
                "completed_episodes": len(result.episode_records),
                "epsilon_decay_environment_steps": (
                    training_config.epsilon_decay_environment_steps
                ),
                "validation_interval_environment_steps": (
                    training_config.progress_interval_environment_steps
                ),
                "environment_steps": result.environment_steps,
                "gradient_updates": result.gradient_updates,
                "training_seconds": result.training_seconds,
                "mask_static_invalid_actions": (
                    training_config.mask_static_invalid_actions
                ),
                "dynamic_obstacle_curriculum": {
                    "enabled": bool(curriculum_stages),
                    "switch_semantics": (
                        "training_probe_gate_with_registered_maximum_stage_duration"
                        if gated_curriculum_enabled
                        else "first_episode_starting_at_or_after_boundary"
                    ),
                    "gate_enabled": gated_curriculum_enabled,
                    "gate_configuration": gate_values if gated_curriculum_enabled else None,
                    "probe_scenario_ids": [
                        scenario.seed for scenario in curriculum_probe_scenarios
                    ],
                    "probe_summary_path": (
                        str(run_dir / "curriculum_probe_summary.csv")
                        if curriculum_probe_summaries
                        else None
                    ),
                    "rehearsal_probabilities": rehearsal_probabilities,
                    "transitions": curriculum_transitions,
                    "forced_transition_count": sum(
                        bool(record["forced"]) for record in curriculum_transitions
                    ),
                    "reached_final_stage": (
                        isinstance(
                            train_factory,
                            CurriculumScheduledDynamicEnvironmentFactory,
                        )
                        and train_factory.current_stage_index
                        == len(curriculum_stages) - 1
                    ),
                    "evaluation_uses_full_obstacle_scenarios": True,
                    "stages": [
                        {
                            "name": stage.name,
                            "start_environment_step": stage.start_environment_step,
                            "obstacle_count": stage.obstacle_count,
                            "obstacle_indices": (
                                list(stage.obstacle_indices)
                                if stage.obstacle_indices is not None
                                else None
                            ),
                            "allowed_difficulties": (
                                list(stage.allowed_difficulties)
                                if stage.allowed_difficulties is not None
                                else None
                            ),
                        }
                        for stage in curriculum_stages
                    ],
                },
            },
            "final_validation_summary": validation_summaries[-1],
            "checkpoint_selection": {
                "enabled": select_best_checkpoint,
                "criterion": [
                    "max_safe_success_rate",
                    "max_success_rate",
                    "min_mean_dynamic_collision_count",
                    "min_mean_steps",
                ],
                "best_episode": best_checkpoint_episode,
                "best_validation_summary": best_validation_summary,
                "selected_model": str(selected_model_path),
                "saved_all_validation_checkpoints": save_validation_checkpoints,
            },
            "test_summary": test_summary,
            "paired_test_summary": paired_test_summary,
            "static_retention_summary": static_summary,
        },
        run_dir / "run_metadata.json",
    )
    print(
        f"{args.strategy} seed={seed} test_success={test_summary['success_rate']:.1%} "
        f"test_safe_success={test_summary['safe_success_rate']:.1%} "
        f"dynamic_collision={test_summary['dynamic_collision_rate']:.1%} "
        f"selected_episode={best_checkpoint_episode} "
        f"output={run_dir}"
    )


if __name__ == "__main__":
    main()
