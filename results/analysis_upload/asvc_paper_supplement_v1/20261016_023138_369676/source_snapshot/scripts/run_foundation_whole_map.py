"""Evaluate existing models or fork qualified foundations; defaults to preflight only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from astar_d3qn.core.grid import manhattan
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.replay_adaptation import (
    decay_demo_fraction, epsilon_at, make_agent, make_env, restore,
    seed_everything, snapshot, state_digest, train_steps,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from train_whole_map_route_pool_pilot import _factory, _load_inputs, _resolve

METHODS = ("foundation_no_demo", "foundation_fixed_25", "foundation_time_decay")
LATEST_METHODS = ("pure", "prefill", "time_decay", "adaptive")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(path):
    config = load_config(_resolve(path))
    legacy = load_config(_resolve(config["legacy_config"]))
    scene_config = load_config(_resolve(config["scene_config"]))
    problem, entry, pool = _load_inputs(scene_config)
    old_manifest = read_json(_resolve(legacy["dataset"]))
    old_problem = problem_from_record(next(m["problem"] for m in old_manifest["maps"]
                                          if m["problem"]["map_id"] == problem.map_id))
    if (old_problem.grid_sha256 != problem.grid_sha256 or old_problem.start != problem.start
            or old_problem.goal != problem.goal or old_problem.nominal_path != problem.nominal_path):
        raise ValueError("Old/new map, fixed endpoints or nominal A* route differ.")
    # The scene substitution must not silently introduce a different old stage.
    if config["adaptation"] != legacy["adaptation"]:
        raise ValueError("Adaptation settings differ from the old dynamic stage.")
    expected = {"max_steps": 200000, "evaluation_interval": 10000,
                "epsilon_start": .30, "epsilon_end": .05, "epsilon_decay_steps": 150000,
                "safe_success_threshold": .90, "consecutive_passes": 2}
    if config["adaptation"] != expected or tuple(config["methods"]) != METHODS:
        raise ValueError("Frozen reuse protocol changed.")
    if scene_config["reward"] != legacy["reward"]:
        raise ValueError("Old/new rewards differ.")
    expected_env = {"window_size": legacy["window_size"], "spatial_channels": 4,
                    "action_count": 5, "max_steps": legacy["max_episode_steps"],
                    "terminate_on_collision": True, "mask_static_invalid_actions": True}
    if scene_config["environment"] != expected_env:
        raise ValueError("Old/new observations or collision settings differ.")
    for key in ("learning_rate", "gamma", "target_sync_interval", "hidden_dim"):
        if scene_config["agent"][key] != legacy[key]:
            raise ValueError(f"Old/new agent setting differs: {key}")
    if scene_config["agent"]["gradient_clip_norm"] != 10.0:
        raise ValueError("Gradient clipping changed.")
    if scene_config["dynamic_route_pool"]["obstacle_counts"] != [3, 4, 5]:
        raise ValueError("Expected the latest 3/4/5 scene pool.")
    latest_root = _resolve(config["latest_experiment_root"])
    frozen_path = latest_root / "fixed_validation_scenarios.json"
    frozen = read_json(frozen_path)
    if (frozen["map_id"] != problem.map_id or frozen["grid_sha256"] != problem.grid_sha256
            or frozen["route_pool_design_sha256"] != pool["design_sha256"]
            or len(frozen["scenarios"]) != 50):
        raise ValueError("Frozen validation identity mismatch.")
    for scene in frozen["scenarios"]:
        if (tuple(scene["start"]) != problem.start or tuple(scene["goal"]) != problem.goal
                or len(scene["obstacles"]) not in (3, 4, 5)):
            raise ValueError("Frozen validation scene changed.")
    # Reconstruct exactly the recorded sequence, including phases and directions.
    validation_factory = _factory(scene_config, problem, entry, pool,
                                  seed=frozen["sampling_seed"], prefix="validation")
    for scene in frozen["scenarios"]:
        env = validation_factory(problem, max_steps=legacy["max_episode_steps"],
                                  window_size=legacy["window_size"])
        if (env.scenario_id != scene["scenario_id"] or
                state_digest([asdict(s) for s in env.dynamic_obstacles]) != state_digest(scene["obstacles"])):
            raise ValueError("Saved validation differs from regenerated frozen pool.")
    return config, legacy, scene_config, problem, entry, pool, frozen


def foundation_path(config, seed):
    return _resolve(config["foundation_source_root"]) / f"seed_{seed}/foundation/foundation.pt"


def load_foundation(config, legacy, problem, seed):
    path = foundation_path(config, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing foundation (never auto-trained): {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta, state = checkpoint["metadata"], checkpoint["state"]
    if (not meta["qualified"] or meta["smoke"] or meta["seed"] != seed
            or meta["map_id"] != problem.map_id or meta["grid_sha256"] != problem.grid_sha256):
        raise ValueError(f"Invalid/unqualified foundation: {path}")
    effective = read_json(path.parent / "effective_config.json")
    for key in ("foundation", "adaptation", "reward", "window_size", "max_episode_steps",
                "batch_size", "learning_rate", "gamma", "target_sync_interval", "hidden_dim",
                "demo_seed", "demo_episodes", "replay_capacity"):
        if effective[key] != legacy[key]:
            raise ValueError(f"Foundation config mismatch: {key}")
    if (state["online_replay"]["capacity"] != config["online_replay_capacity"]
            or len(state["demonstrations"]) != config["demonstration_transition_count"]
            or state_digest(state) != meta["snapshot_sha256"]):
        raise ValueError("Foundation full-state content/capacity mismatch.")
    return checkpoint


def fork(config, legacy, problem, checkpoint, seed, method, device):
    if device.split(":")[0] != checkpoint["metadata"]["device"].split(":")[0]:
        raise ValueError("Training/preflight must use the foundation device family to restore RNG exactly.")
    fraction = 0.0 if method == "foundation_no_demo" else .25
    agent, replay = restore(legacy, checkpoint["state"], fraction, seed, device)
    digest = state_digest(snapshot(agent, replay))
    if digest != checkpoint["metadata"]["snapshot_sha256"]:
        raise RuntimeError("Network/target/optimizer/replay/RNG restore mismatch.")
    return agent, replay, {"method": method, "seed": seed, "verified_full_state_equal": True,
                           "source_snapshot_sha256": digest,
                           "foundation_path": str(foundation_path(config, seed)),
                           "online_capacity": replay.online.capacity,
                           "online_size": replay.online_size, "demo_size": replay.demonstration_size,
                           "initial_sample_counts": list(replay.sample_counts(legacy["batch_size"])),
                           "epsilon_clock": "environment steps since dynamic fork"}


def trajectory_stats(positions, actions, goal, safe, collision):
    moves = [(a, b) for a, b in zip(positions, positions[1:]) if a != b]
    seen, revisits = set(), 0
    for a, b in moves:
        seen.add(a)
        revisits += int(b in seen)
    waits = sum(a == 4 for a in actions)
    streak = longest = 0
    for action in actions:
        streak = streak + 1 if action == 4 else 0
        longest = max(longest, streak)
    reversals = sum(a == c and a != b for a, b, c in zip(positions, positions[1:], positions[2:]))
    tail = positions[-min(51, len(positions)):]
    if safe:
        behavior = "safe_arrival"
    elif collision:
        behavior = "collision"
    elif len(set(tail)) == 1 and longest >= 50:
        behavior = "timeout_waiting"
    elif len(moves) > 0 and revisits / len(moves) >= .5:
        behavior = "timeout_repeated_movement"
    else:
        behavior = "timeout_no_arrival"
    return {"failure_behavior": behavior, "wait_steps": waits, "max_wait_streak": longest,
            "movement_steps": len(moves), "movement_revisits": revisits,
            "immediate_reversals": reversals, "unique_positions": len(set(positions)),
            "tail_unique_positions": len(set(tail)), "final_position": list(positions[-1]),
            "final_goal_manhattan": manhattan(positions[-1], goal)}


def rollout(agent, problem, legacy, scene):
    env = make_env(problem, legacy, scene)
    obs = env.reset()
    positions, actions, frames = [tuple(env.position)], [], [list(env.dynamic_positions)]
    total_reward = 0.0
    while True:
        action = agent.select_action(obs, epsilon=0.0,
                                     valid_actions=np.flatnonzero(env.action_mask(True)).tolist())
        result = env.step(action)
        actions.append(action)
        positions.append(tuple(env.position))
        frames.append(list(env.dynamic_positions))
        total_reward += result.reward
        obs = result.observation
        if result.done:
            break
    collision = bool(result.info["collision"])
    safe = bool(result.info["reached"]) and not collision
    row = {"scenario_id": scene["scenario_id"] if scene else "static_original_map",
           "obstacle_count": len(env.dynamic_obstacles), "safe_success": int(safe),
           "dynamic_collision": int(result.info["collision_type"] == "dynamic"),
           "static_collision": int(result.info["collision_type"] == "static"),
           "timeout": int(result.truncated), "steps": env.steps, "return": total_reward,
           "termination_reason": result.info["termination_reason"],
           "collision_position": result.info.get("collision_position"),
           **trajectory_stats(positions, actions, problem.goal, safe, collision)}
    return row, {**row, "positions": positions, "actions": actions, "dynamic_positions": frames}


def evaluate(agent, problem, legacy, scenes):
    rows, trajectories = [], []
    for scene in scenes:
        row, trace = rollout(agent, problem, legacy, scene)
        rows.append(row)
        trajectories.append(trace)
    return {"episodes": len(rows),
            **{key + "_rate": sum(r[key] for r in rows) / len(rows)
               for key in ("safe_success", "dynamic_collision", "static_collision", "timeout")},
            "behavior_counts": dict(Counter(r["failure_behavior"] for r in rows))}, rows, trajectories


def scene_sampler(scene_config, problem, entry, pool, legacy, seed):
    factory = _factory(scene_config, problem, entry, pool,
                       seed=scene_config["dynamic_route_pool"]["training_sampling_seed_offset"] + seed,
                       prefix=f"train_seed{seed}")
    def sample():
        env = factory(problem, max_steps=legacy["max_episode_steps"], window_size=legacy["window_size"])
        return {"scenario_id": env.scenario_id, "condition": "whole_map_3to5",
                "obstacle_count": len(env.dynamic_obstacles),
                "route_ids": list(env.dynamic_route_ids),
                "categories": list(env.dynamic_route_categories),
                "obstacles": [asdict(spec) for spec in env.dynamic_obstacles]}
    return sample


def preflight(inputs, seeds, device, directory):
    config, legacy, scene_config, problem, entry, pool, frozen = inputs
    audits = []
    for seed in seeds:
        checkpoint = load_foundation(config, legacy, problem, seed)
        equivalent_path = _resolve(config["equivalent_foundation_root"]) / f"seed_{seed}/foundation/foundation.pt"
        if equivalent_path.exists():
            equivalent = torch.load(equivalent_path, map_location="cpu", weights_only=False)
            if state_digest(equivalent["state"]) != checkpoint["metadata"]["snapshot_sha256"]:
                raise ValueError("The two old foundation sources are not equivalent.")
        else:
            equivalent = None
        for method in METHODS:
            agent, replay, audit = fork(config, legacy, problem, checkpoint, seed, method, device)
            sample = scene_sampler(scene_config, problem, entry, pool, legacy, seed)
            scenes = [sample() for _ in range(3)]
            old_env = make_env(problem, legacy, scenes[0])
            obs = old_env.reset()
            assert obs.spatial.shape == (4, 15, 15) and obs.scalars.shape == (2,)
            assert agent.policy_network(torch.tensor(obs.spatial, device=device).unsqueeze(0),
                                        torch.tensor(obs.scalars, device=device).unsqueeze(0)).shape == (1, 5)
            audit.update({"equivalent_old_foundation": equivalent is not None,
                          "equivalent_old_foundation_path": str(equivalent_path),
                          "sampled_scene_prefix_sha256": state_digest(scenes),
                          "observation_shape": list(obs.spatial.shape), "scalar_shape": list(obs.scalars.shape),
                          "map_sha256": problem.grid_sha256,
                          "source_weights_sha256": file_sha(foundation_path(config, seed))})
            audits.append(audit)
        assert len({a["sampled_scene_prefix_sha256"] for a in audits if a["seed"] == seed}) == 1
    write_json({"formal_training_started": False, "audits": audits,
                "validation_scenarios_sha256": file_sha(_resolve(config["latest_experiment_root"]) / "fixed_validation_scenarios.json"),
                "validation_count": len(frozen["scenarios"]),
                "decay_boundary_steps": [1, 50000, 50001, 100000, 100001, 200000],
                "decay_boundary_fractions": [decay_demo_fraction(s) for s in (1, 50000, 50001, 100000, 100001, 200000)]},
               directory / "preflight.json")
    print(f"Preflight passed: {len(audits)} identical full-state forks; no formal training.", flush=True)


def evaluate_existing(inputs, seeds, device, directory):
    config, legacy, _, problem, _, _, frozen = inputs
    summaries, missing = [], []
    for seed in seeds:
        for method in (*LATEST_METHODS, "old_foundation"):
            path = (foundation_path(config, seed) if method == "old_foundation" else
                    _resolve(config["latest_experiment_root"]) / method / f"seed_{seed}/model_final.pth")
            if not path.is_file():
                missing.append(str(path))
                continue
            agent = make_agent(legacy, seed, device)
            if method == "old_foundation":
                checkpoint = load_foundation(config, legacy, problem, seed)
                agent.load_training_state_dict(checkpoint["state"]["agent"])
            else:
                effective = read_json(path.parent / "effective_config.json")
                registered = read_json(path.parent / "result.json")["config"]
                for section in ("dataset", "dynamic_route_pool", "environment", "reward", "agent"):
                    if effective[section] != registered[section]:
                        raise ValueError(f"Latest model configuration changed: {path}: {section}")
                    if section != "agent" and effective[section] != inputs[2][section]:
                        raise ValueError(f"Latest model differs from frozen evaluation: {section}")
                if effective["environment"] != inputs[2]["environment"] or effective["reward"] != legacy["reward"]:
                    raise ValueError("Latest model evaluation inputs incompatible.")
                agent.load_weights(path)
            before = state_digest({k: v for k, v in agent.training_state_dict().items() if k != "rng_state"})
            for kind, scenes in (("static", [None]), ("fixed_dynamic_validation", frozen["scenarios"])):
                summary, rows, traces = evaluate(agent, problem, legacy, scenes)
                model_dir = directory / "evaluation" / method / f"seed_{seed}"
                model_dir.mkdir(parents=True, exist_ok=True)
                write_records_csv(rows, model_dir / f"{kind}_episodes.csv")
                write_json(traces, model_dir / f"{kind}_trajectories.json")
                if method != "old_foundation" and kind == "fixed_dynamic_validation":
                    original = read_json(path.parent / "final_validation_summary.json")
                    for metric in ("safe_success_rate", "dynamic_collision_rate"):
                        if abs(original[metric] - summary[metric]) > 1e-12:
                            raise ValueError(f"Re-evaluation differs from the original: {method} seed {seed}: {metric}")
                summaries.append({"method": method, "seed": seed, "evaluation": kind,
                                  "weight_path": str(path), "weight_sha256": file_sha(path), **summary})
                print(json.dumps({"method": method, "seed": seed, "evaluation": kind, **summary}), flush=True)
            # Greedy action selection advances only its private RNG; learning state stays fixed.
            after = state_digest({k: v for k, v in agent.training_state_dict().items() if k != "rng_state"})
            if before != after:
                raise RuntimeError("Evaluation changed network/target/optimizer/update count.")
    write_json({"epsilon": 0.0, "no_training": True, "no_test_data": True,
                "missing_weight_paths": missing, "summaries": summaries,
                "static_note": "One deterministic original-map episode per model; rates describe this task, not a generalization sample."},
               directory / "evaluation_summary.json")
    write_records_csv(summaries, directory / "evaluation_summary.csv")


def train(inputs, seeds, methods, device, output_root, integration_steps):
    config, legacy, scene_config, problem, entry, pool, frozen = inputs
    mode = "integration" if integration_steps else "formal"
    root = output_root / mode
    destinations = [root / f"seed_{s}" / m for s in seeds for m in methods]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {path}")
    # Validate every requested source before creating any training destination.
    for seed in seeds:
        load_foundation(config, legacy, problem, seed)
    for seed in seeds:
        checkpoint = load_foundation(config, legacy, problem, seed)
        for method in methods:
            agent, replay, audit = fork(config, legacy, problem, checkpoint, seed, method, device)
            directory = root / f"seed_{seed}" / method
            directory.mkdir(parents=True, exist_ok=False)
            stage = copy.deepcopy(legacy["adaptation"])
            if integration_steps:
                stage["max_steps"] = integration_steps
                stage["evaluation_interval"] = integration_steps
            write_json({**audit, "stage": stage, "no_test_data": True,
                        "scene_config": scene_config, "legacy_config": legacy,
                        "source_weight_sha256": file_sha(foundation_path(config, seed)),
                        "scene_sampling_seed": scene_config["dynamic_route_pool"]["training_sampling_seed_offset"] + seed},
                       directory / "fork_audit.json")
            write_json(frozen, directory / "fixed_validation_scenarios.json")
            curve, details = [], []
            def check(step, training, _prediction):
                summary, rows, traces = evaluate(agent, problem, legacy, frozen["scenarios"])
                curve.append({"environment_steps": step, **summary})
                details.extend({"environment_steps": step, **row} for row in rows)
                write_records_csv(training, directory / "training.csv")
                write_records_csv(curve, directory / "validation_curve.csv")
                write_records_csv(details, directory / "validation_details.csv")
                write_json(traces, directory / "validation_latest_trajectories.json")
                print(f"{method} seed={seed} steps={step} safe={summary['safe_success_rate']:.2f} collision={summary['dynamic_collision_rate']:.2f}", flush=True)
                return False  # Old dynamic-stage budget remains fixed; no validation stopping.
            check(0, [], {})
            sampler = scene_sampler(scene_config, problem, entry, pool, legacy, seed)
            drawn = []
            def logged_sampler():
                scene = sampler()
                drawn.append(scene)
                return scene
            start_updates = agent.update_steps
            run = train_steps(agent, replay, problem, legacy, stage, [], seed, check,
                              replay_schedule="decay" if method == "foundation_time_decay" else None,
                              scene_sampler=logged_sampler)
            write_json(drawn, directory / "training_scenarios.json")
            torch.save({"state": snapshot(agent, replay), "metadata": {**audit, "run": run,
                       "integration_only": bool(integration_steps)}}, directory / "checkpoint_final.pt")
            agent.save_weights(directory / "model_final.pth")
            write_json({"method": method, "seed": seed, "run": run, "gradient_updates": agent.update_steps - start_updates,
                        "final_validation": curve[-1], "online_capacity": replay.online.capacity,
                        "demo_size": replay.demonstration_size, "no_test_data": True,
                        "integration_only": bool(integration_steps)}, directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/whole_map_91701_foundation_reuse_v1.yaml")
    parser.add_argument("--mode", choices=("preflight", "evaluate", "integration", "train"), default="preflight")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-root")
    parser.add_argument("--integration-steps", type=int, default=8)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s not in (0, 1) for s in args.seeds):
        parser.error("Only distinct registered seeds 0/1 are allowed.")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate methods are not allowed.")
    if not 1 <= args.integration_steps <= 32:
        parser.error("Integration checks are limited to 1..32 steps per branch.")
    torch.set_num_threads(args.threads)
    seed_everything(0)
    inputs = load_inputs(args.config)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.mode in ("preflight", "evaluate"):
        directory = _resolve(args.output_root or inputs[0]["audit_root"])
        directory.mkdir(parents=True, exist_ok=True)
        if args.mode == "preflight":
            preflight(inputs, args.seeds, device, directory)
        else:
            evaluate_existing(inputs, args.seeds, device, directory)
    else:
        train(inputs, args.seeds, args.methods, device,
              _resolve(args.output_root or inputs[0]["output_root"]),
              args.integration_steps if args.mode == "integration" else 0)


if __name__ == "__main__":
    main()
