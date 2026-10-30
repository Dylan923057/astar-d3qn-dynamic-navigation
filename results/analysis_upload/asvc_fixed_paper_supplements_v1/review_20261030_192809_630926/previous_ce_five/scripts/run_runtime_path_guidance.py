"""Independent runtime A* guidance entry. Default: checks only, never formal training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from astar_d3qn.agents.astar_replan_wait import ObservedAStarReplanWaitPolicy
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.core.grid import Action, action_between
from astar_d3qn.envs.path_guidance import PathGuidanceFactory
from astar_d3qn.evaluation.rollout import evaluate_agent
from astar_d3qn.training.trainer import train_d3qn
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything
from train_whole_map_route_pool_pilot import _factory, _load_inputs, _resolve, _reward_config, _training_config

CONFIG = ROOT / "configs/whole_map_91701_runtime_path_v1.yaml"
PILOT_CONFIG = ROOT / "configs/whole_map_91701_runtime_path_pilot_v1.yaml"
NO_PROGRESS_CONFIG = ROOT / "configs/whole_map_91701_runtime_path_no_progress_v1.yaml"
METHODS = ("unguided", "path_guided")
REGISTERED_SEEDS = (0, 1, 2, 3, 4)


def validate_config(config):
    baseline = load_config(ROOT / "configs/whole_map_route_pool_91701_pilot_v1.yaml")
    protocol = config["experiment"]["protocol"]
    output_names = {
        "runtime_path_observation_v1": "whole_map_91701_runtime_path_v1",
        "runtime_path_observation_pilot_v1": "whole_map_91701_runtime_path_pilot_v1",
        "runtime_path_observation_no_progress_v1": "whole_map_91701_runtime_path_no_progress_v1",
    }
    if protocol not in output_names:
        raise ValueError("Unexpected runtime guidance protocol.")
    pilot = protocol == "runtime_path_observation_pilot_v1"
    no_progress = protocol == "runtime_path_observation_no_progress_v1"
    if config["experiment"]["initialization"] != "paired_random":
        raise ValueError("This entry does not load legacy foundation weights or replay.")
    for section in ("environment", "agent"):
        if config[section] != baseline[section]:
            raise ValueError(f"Registered {section} changed.")
    expected_reward = dict(baseline["reward"])
    if no_progress:
        expected_reward["progress"] = 0.0
    if config["reward"] != expected_reward:
        raise ValueError("Registered reward changed; the no-progress protocol only removes the progress term.")
    for key, value in baseline["training"].items():
        if no_progress and key == "seeds":
            value = list(REGISTERED_SEEDS)
        if pilot and key in {"max_environment_steps", "epsilon_decay_environment_steps"}:
            value = {"max_environment_steps": 50000, "epsilon_decay_environment_steps": 37500}[key]
        if config["training"].get(key) != value:
            raise ValueError(f"Registered training setting changed: {key}")
    if config["training"]["demo_fraction"] != 0 or any(key in config for key in ("demonstration", "adaptive", "foundation")):
        raise ValueError("Runtime guidance v1 isolates observations; demonstration/adaptive/foundation loading is disabled.")
    for key, value in baseline["dataset"].items():
        if config["dataset"].get(key) != value:
            raise ValueError(f"Registered dataset changed: {key}")
    dynamic = config["dynamic_route_pool"]
    for key, value in baseline["dynamic_route_pool"].items():
        if key != "composition" and dynamic.get(key) != value:
            raise ValueError(f"Registered dynamic setting changed: {key}")
    if dynamic["obstacle_counts"] != [3, 4, 5] or dynamic["composition"] != {"high_interaction": 2, "alternative_branch": 2, "background": 1}:
        raise ValueError("The latest 3-5 obstacle scene protocol changed.")
    if config["guidance"] != {
        "methods": list(METHODS), "lookahead_steps": 4,
        "projection": "nearest_manhattan_earliest_tie",
        "path_source": "frozen_nominal_static_astar", "added_spatial_channels": 1,
        "added_scalar_dim": 2, "control": "zero_added_inputs_same_network",
    }:
        raise ValueError("The fixed guidance v1 specification changed.")
    if config["integration"] != {"steps_per_branch": 8, "batch_size": 4, "learning_starts": 4, "device": "cpu"}:
        raise ValueError("Integration mode must remain bounded to eight CPU steps per branch.")
    if config["rule_baseline"] != {
        "method": "astar_replan_wait",
        "information": "known_static_map_and_current_local_dynamic_frame",
    }:
        raise ValueError("Rule baseline information contract changed.")
    for key, directory in (("output_root", "outputs"), ("check_root", "results")):
        expected = ROOT / directory / output_names[protocol]
        if _resolve(config["experiment"][key]).resolve() != expected.resolve():
            raise ValueError("Runtime guidance must use its own registered output directories.")
    problem, entry, pool = _load_inputs(config)
    if problem.grid_sha256 != config["dataset"]["grid_sha256"]:
        raise ValueError("Registered map grid SHA mismatch.")
    # Constructor also verifies the route pool's grid SHA and start/goal.
    _factory(config, problem, entry, pool, seed=0, prefix="verify")
    return problem, entry, pool


def make_factory(config, problem, entry, pool, *, method, seed, validation=False, trace=False):
    if method not in (*METHODS, "astar_replan_wait"):
        raise ValueError(f"Unknown method: {method}")
    sampling_seed = int(config["dynamic_route_pool"]["validation_sampling_seed"]) if validation else int(config["dynamic_route_pool"]["training_sampling_seed_offset"]) + seed
    factory = _factory(config, problem, entry, pool, seed=sampling_seed, prefix="validation" if validation else f"train_seed{seed}")
    return PathGuidanceFactory(factory, enabled=method == "path_guided",
                               lookahead_steps=config["guidance"]["lookahead_steps"], record_trace=trace)


def make_agent(config, seed, device=None):
    seed_everything(seed)
    environment = config["environment"]
    values = dict(config["agent"])
    if device is not None:
        values["device"] = device
    window = environment["window_size"]
    return D3QNAgent(D3QNConfig(spatial_shape=(5, window, window), scalar_dim=4,
                                  action_dim=environment["action_count"], seed=seed, **values))


def state_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


@contextmanager
def preserved_evaluation(agent):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    agent_state = agent._rng.getstate() if isinstance(agent, D3QNAgent) else None
    modules = list(agent.policy_network.modules()) + list(agent.target_network.modules()) if isinstance(agent, D3QNAgent) else []
    modes = [module.training for module in modules]
    before = state_digest(agent.training_state_dict()) if isinstance(agent, D3QNAgent) else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if agent_state is not None:
            agent._rng.setstate(agent_state)
        for module, mode in zip(modules, modes):
            module.training = mode
        if before is not None and state_digest(agent.training_state_dict()) != before:
            raise RuntimeError("Evaluation changed the agent's network, optimizer, update counter, or RNG.")


def evaluate(config, problem, entry, pool, method, seed, agent):
    factory = make_factory(config, problem, entry, pool, method=method, seed=seed, validation=True, trace=True)
    environment = config["environment"]
    with preserved_evaluation(agent):
        summary, rows, _ = evaluate_agent(
            agent, [problem] * config["dynamic_route_pool"]["validation_episodes"],
            max_steps=environment["max_steps"], reward_config=_reward_config(config),
            terminate_on_collision=environment["terminate_on_collision"],
            window_size=environment["window_size"], environment_factory=factory,
            mask_static_invalid_actions=environment["mask_static_invalid_actions"],
        )
    summary["timeout_rate"] = sum(row["termination_reason"] == "timeout" for row in rows) / len(rows)
    traces = []
    for row, env in zip(rows, factory.environments):
        traces.append({
            "scenario_id": env.scenario_id, "safe_success": bool(row["safe_success"]),
            "termination_reason": row["termination_reason"],
            "wait_steps": row["wait_steps"], "revisit_count": row["revisit_count"],
            "obstacles": [asdict(spec) for spec in env.dynamic_obstacles], "steps": env.trace,
        })
    return summary, rows, traces


def check(config, inputs, seeds):
    problem, entry, pool = inputs
    environment = config["environment"]
    kwargs = dict(max_steps=environment["max_steps"], window_size=environment["window_size"],
                  reward_config=_reward_config(config), terminate_on_collision=environment["terminate_on_collision"])
    factories = {method: make_factory(config, *inputs, method=method, seed=0, validation=True) for method in METHODS}
    reference_file = _resolve(config["dataset"]["validation_reference"])
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    if reference["grid_sha256"] != problem.grid_sha256 or reference["route_pool_design_sha256"] != pool["design_sha256"] or len(reference["scenarios"]) != config["dynamic_route_pool"]["validation_episodes"]:
        raise ValueError("Existing fixed validation reference does not match this protocol.")
    def physical_scene(specs):
        return [{key: spec[key] for key in ("route", "start_index", "direction", "move_every")} for spec in specs]
    scenes = []
    for scene_index in range(config["dynamic_route_pool"]["validation_episodes"]):
        left, right = (factories[method](problem, **kwargs) for method in METHODS)
        if left.dynamic_obstacles != right.dynamic_obstacles or left.scenario_id != right.scenario_id:
            raise RuntimeError("The methods received different validation scenes.")
        actual_specs = json.loads(json.dumps([asdict(spec) for spec in left.dynamic_obstacles]))
        if physical_scene(actual_specs) != physical_scene(reference["scenarios"][scene_index]["obstacles"]):
            raise RuntimeError("Sampled validation scene differs from the existing frozen foundation experiment.")
        ls, rs = left.reset(), right.reset()
        np.testing.assert_array_equal(ls.spatial[:4], rs.spatial[:4])
        np.testing.assert_array_equal(ls.scalars[:2], rs.scalars[:2])
        np.testing.assert_array_equal(left.action_mask(True), right.action_mask(True))
        if np.any(ls.spatial[4]) or np.any(ls.scalars[2:]) or not np.any(rs.spatial[4]):
            raise RuntimeError("Guidance/control inputs are incorrect.")
        for current, following in zip(problem.nominal_path[:17], problem.nominal_path[1:17]):
            action = int(action_between(current, following))
            lr, rr = left.step(action), right.step(action)
            if (lr.reward, lr.terminated, lr.truncated, lr.info) != (rr.reward, rr.terminated, rr.truncated, rr.info):
                raise RuntimeError("Guidance changed reward, movement, or collision rules.")
            np.testing.assert_array_equal(lr.observation.spatial[:4], rr.observation.spatial[:4])
            if lr.done:
                break
        scenes.append({"scenario_id": left.scenario_id, "obstacle_count": len(left.dynamic_obstacles),
                       "route_ids": left.dynamic_route_ids, "obstacles": [asdict(spec) for spec in left.dynamic_obstacles]})
    initializations = []
    for seed in seeds:
        hashes = []
        for method in METHODS:
            agent = make_agent(config, seed, "cpu")
            hashes.append(state_digest(agent.training_state_dict()))
            env = make_factory(config, *inputs, method=method, seed=seed)(problem, **kwargs)
            state = env.reset()
            agent.select_action(state, epsilon=0.0, valid_actions=list(np.flatnonzero(env.action_mask(True))))
        if hashes[0] != hashes[1]:
            raise RuntimeError("Paired initial networks/optimizers/RNG differ.")
        initializations.append({"seed": seed, "initial_state_sha256": hashes[0], "paired_identical": True})
    return {
        "passed": True, "formal_training_started": False, "gradient_updates": 0,
        "map": problem.manifest(), "route_pool_design_sha256": pool["design_sha256"],
        "observation_shape": [5, environment["window_size"], environment["window_size"]],
        "scalar_dim": 4, "validation_scene_count": len(scenes),
        "validation_scene_sha256": state_digest(scenes), "initializations": initializations,
        "existing_validation_reference": str(reference_file.relative_to(ROOT)),
        "existing_validation_file_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "matches_existing_fixed_50_validation_scenes": True,
        "original_dynamics_reward_and_masks_preserved": config["experiment"]["protocol"] != "runtime_path_observation_no_progress_v1",
        "original_dynamics_and_masks_preserved": True,
        "configured_rewards_identical_between_methods": True,
        "reward": config["reward"],
        "only_progress_reward_removed": config["experiment"]["protocol"] == "runtime_path_observation_no_progress_v1",
        "legacy_weight_compatible": False,
        "scope": "Fixed original map/start/goal integration; no multi-goal or generalization claim.",
    }, scenes


def unique_check_dir(config, action):
    root = _resolve(config["experiment"]["check_root"])
    directory = root / f"{action}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(config, directory / "effective_config.json")
    return directory


def run_training(config, inputs, seeds, methods, *, smoke=False):
    problem, entry, pool = inputs
    pilot = config["experiment"]["protocol"] == "runtime_path_observation_pilot_v1"
    run_root = unique_check_dir(config, "smoke") if smoke else _resolve(config["experiment"]["output_root"])
    planned = [run_root / method / f"seed_{seed}" for seed in seeds for method in methods]
    if any(directory.exists() for directory in planned):
        raise FileExistsError("Refusing to overwrite an existing runtime guidance run.")
    for seed in seeds:
        for method in methods:
            directory = run_root / method / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            training = _training_config(config, seed)
            if smoke:
                integration = config["integration"]
                training = replace(training, max_environment_steps=integration["steps_per_branch"],
                                   batch_size=integration["batch_size"], learning_starts=integration["learning_starts"],
                                   epsilon_decay_environment_steps=integration["steps_per_branch"],
                                   progress_interval_environment_steps=None)
            agent = make_agent(config, seed, "cpu" if smoke else None)
            initial_hash = state_digest(agent.training_state_dict())
            factory = make_factory(config, *inputs, method=method, seed=seed)
            validations, details = [], []
            def record_validation(step):
                summary, rows, traces = evaluate(config, *inputs, method, seed, agent)
                validations.append({"environment_steps_total": step, **summary})
                details.extend({"environment_steps_total": step, **row} for row in rows)
                write_records_csv(validations, directory / "validation_curve.csv")
                write_records_csv(details, directory / "validation_details.csv")
                write_json([trace for trace in traces if not trace["safe_success"]], directory / f"validation_failures_{step:06d}.json")
                print(f"[{method} seed={seed}] step={step} safe={summary['safe_success_rate']:.1%} collision={summary['dynamic_collision_rate']:.1%} timeout={summary['timeout_rate']:.1%}", flush=True)
                return summary
            if not smoke:
                record_validation(0)
            def progress(_episode, records, full_evaluation):
                if full_evaluation:
                    record_validation(records[-1]["environment_steps_total"])
                    write_records_csv(list(records), directory / "training.csv")
            result = train_d3qn([problem], agent, training, demonstrations=(), reward_config=_reward_config(config),
                               environment_factory=factory, progress_callback=progress if not smoke else None)
            write_records_csv(list(result.episode_records), directory / "training.csv")
            if not smoke:
                if not validations or validations[-1]["environment_steps_total"] != result.environment_steps:
                    record_validation(result.environment_steps)
                agent.save_weights(directory / "model_final.pth")
            if smoke and result.gradient_updates <= 0:
                raise RuntimeError("Integration check did not exercise replay sampling and an optimizer update.")
            write_json({
                "method": method, "seed": seed, "config": config,
                "effective_training": asdict(training),
                "formal_training_started": not smoke and not pilot,
                "pilot_training_started": not smoke and pilot, "integration_only": smoke,
                "training_mode": "integration" if smoke else "pilot" if pilot else "formal",
                "initial_state_sha256": initial_hash, "initialization": "paired_random",
                "environment_steps": result.environment_steps, "gradient_updates": result.gradient_updates,
                "demo_transition_count": 0, "online_replay_capacity": training.replay_capacity,
                "map": problem.manifest(), "route_pool_design_sha256": pool["design_sha256"],
                "final_validation": validations[-1] if validations else None,
            }, directory / "result.json")
            write_json({"complete": True, "environment_steps": result.environment_steps}, directory / "completion.json")
            print(f"[{method} seed={seed}] {'integration' if smoke else 'training'} complete: {result.environment_steps} steps, {result.gradient_updates} updates; {directory}", flush=True)
    return run_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="Default: input/scene checks only, no gradient updates.")
    actions.add_argument("--smoke", action="store_true", help="Eight CPU integration steps per method/seed; not formal training.")
    actions.add_argument("--evaluate-rules", action="store_true", help="Evaluate the A* replan/wait baseline on 50 validation scenes only.")
    actions.add_argument("--train", action="store_true", help="Explicitly start registered 200000-step formal runs.")
    actions.add_argument("--pilot", action="store_true", help="Explicitly start separate 50000-step pilot runs; epsilon decays over 37500 steps.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(REGISTERED_SEEDS) or len(set(args.methods)) != len(args.methods):
        parser.error("Use unique registered seeds 0/1/2/3/4 and unique methods.")
    config = load_config(_resolve(args.config or (PILOT_CONFIG if args.pilot else CONFIG)))
    inputs = validate_config(config)
    pilot_config = config["experiment"]["protocol"] == "runtime_path_observation_pilot_v1"
    if args.train and pilot_config:
        parser.error("Use --pilot for the short-run config; --train is reserved for the 200000-step config.")
    if args.pilot and not pilot_config:
        parser.error("--pilot requires the registered 50000-step pilot config.")
    if not args.train and not args.pilot:
        torch.set_num_threads(1)
    if args.smoke or args.train or args.pilot:
        directory = run_training(config, inputs, args.seeds, args.methods, smoke=args.smoke)
    elif args.evaluate_rules:
        directory = unique_check_dir(config, "rules")
        summary, rows, traces = evaluate(config, *inputs, "astar_replan_wait", 0, ObservedAStarReplanWaitPolicy(inputs[0]))
        write_json({"method": "astar_replan_wait", "formal_training_started": False, "summary": summary}, directory / "result.json")
        write_records_csv(rows, directory / "validation_details.csv")
        write_json(traces, directory / "validation_trajectories.json")
    else:
        directory = unique_check_dir(config, "check")
        report, scenes = check(config, inputs, args.seeds)
        write_json(report, directory / "check_report.json")
        write_json(scenes, directory / "fixed_validation_scenarios.json")
    print(f"Output: {directory}", flush=True)


if __name__ == "__main__":
    main()
