"""Independent value-repair ablations. Default checks; no automatic formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_runtime_path_guidance as runtime
import run_training_handover as old
from analyze_runtime_path_pilot import static_distances, static_evaluation, trace_behavior
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent, METHODS
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.training.trainer import train_d3qn
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything
from train_whole_map_route_pool_pilot import _resolve, _reward_config, _training_config

CONFIG = ROOT / "configs/whole_map_91701_value_repair_v1.yaml"
PILOT_CONFIG = ROOT / "configs/whole_map_91701_value_repair_pilot_v1.yaml"


def validate_config(config):
    pilot = config["experiment"]["protocol"] == "value_bound_teacher_margin_pilot_v1"
    if config != load_config(PILOT_CONFIG if pilot else CONFIG):
        raise ValueError("Use the registered repair v1 configuration; new settings require a new protocol.")
    baseline = load_config(old.CONFIG)
    for key in baseline:
        if key != "experiment" and config[key] != baseline[key]:
            raise ValueError(f"Repair changed shared {key} settings.")
    if config["value_repair"] != {
        "methods": list(METHODS), "target_lower": -6.0, "target_upper": 10.0,
        "margin": 0.8, "margin_weight": 1.0, "margin_decay_environment_steps": 100000,
        "labels": "matching_executed_astar_action_observed_risk_clear_noncollision",
        "competitor_mask": "static_valid_only", "evaluation_advice": False,
        "probe_environment_steps": 2000, "probe_device": "cpu",
        "probe_epsilon_decay_environment_steps": 150000}:
        raise ValueError("Repair specification changed.")
    for key, parent in (("output_root", "outputs"), ("check_root", "results")):
        name = "whole_map_91701_value_repair_pilot_v1" if pilot else "whole_map_91701_value_repair_v1"
        if _resolve(config["experiment"][key]).resolve() != (ROOT / parent / name).resolve():
            raise ValueError("Repair requires independent directories.")
    if pilot and config["pilot"] != {"environment_steps": 20000, "validation_interval_environment_steps": 5000,
                                     "epsilon_decay_environment_steps": 150000, "advice_decay_environment_steps": 100000,
                                     "auto_start_formal_training": False}:
        raise ValueError("Pilot budget or curriculum changed.")
    if config["reward"] != {"step": -.01, "progress": .05, "stay": 0., "collision": -1., "goal": 10.} or config["agent"]["gamma"] != .99 or not config["environment"]["terminate_on_collision"]:
        raise ValueError("The [-6,10] target bound requires the registered reward, gamma, and collision termination.")
    # Registered gamma/reward/terminal rules are necessary for the [-6,10] bound.
    return old.validate_config(baseline)


def make_agent(config, problem, seed, method, device=None):
    seed_everything(seed)
    values = dict(config["agent"])
    if device:
        values["device"] = device
    advice, repair = config["training_advice"], config["value_repair"]
    window = config["environment"]["window_size"]
    return AStarValueRepairAgent(
        D3QNConfig(spatial_shape=(5, window, window), scalar_dim=4,
                   action_dim=config["environment"]["action_count"], seed=seed, **values),
        problem, repair_method=method, lower=repair["target_lower"], upper=repair["target_upper"],
        margin=repair["margin"], margin_weight=repair["margin_weight"],
        probability=advice["probability"], decay_steps=advice["decay_environment_steps"],
        risk_radius=advice["risk_radius"], rng_offset=advice["rng_seed_offset"])


def effective_training(config, seed, mode):
    training = _training_config(config, seed)
    if mode == "smoke":
        return replace(training, max_environment_steps=8, learning_starts=4, batch_size=4,
                       progress_interval_environment_steps=None, progress_interval=0,
                       allow_partial_epsilon_schedule=True)
    if mode == "probe":
        # Keep the formal epsilon/advice clocks: this is an integration diagnosis,
        # not a compressed curriculum and not evidence of long-run performance.
        return replace(training, max_environment_steps=2000,
                       progress_interval_environment_steps=None, progress_interval=0,
                       allow_partial_epsilon_schedule=True)
    if mode == "pilot":
        return replace(training, max_environment_steps=config["pilot"]["environment_steps"],
                       progress_interval_environment_steps=config["pilot"]["validation_interval_environment_steps"],
                       allow_partial_epsilon_schedule=True)
    return training


def check(config, inputs, seeds):
    report, scenes = runtime.check(load_config(runtime.CONFIG), inputs, seeds)
    hashes = []
    factory = runtime.make_factory(config, *inputs, method="unguided", seed=0)
    env = factory(inputs[0], max_steps=300, window_size=15,
                  reward_config=_reward_config(config), terminate_on_collision=True)
    state = env.reset()
    valid = list(np.flatnonzero(env.action_mask(True)))
    for seed in seeds:
        agents = [make_agent(config, inputs[0], seed, method, "cpu") for method in METHODS]
        initial = [runtime.state_digest(a.training_state_dict()) for a in agents]
        if len(set(initial)) != 1:
            raise RuntimeError("Initial network, optimizer, exploration/advice RNG states differ.")
        baseline = old.make_agent(load_config(old.CONFIG), inputs[0], seed, "astar_risk_decay", "cpu")
        if runtime.state_digest(D3QNAgent.training_state_dict(agents[0])) != runtime.state_digest(D3QNAgent.training_state_dict(baseline)):
            raise RuntimeError("Learner initialization differs from the existing experiment.")
        for agent in agents:
            before = runtime.state_digest(agent.training_state_dict())
            with runtime.preserved_evaluation(agent):
                agent.select_action(state, 0, valid)
            if runtime.state_digest(agent.training_state_dict()) != before:
                raise RuntimeError("Evaluation changed learner or advice state.")
        hashes.append({"seed": seed, "initial_state_sha256": initial[0],
                       "groups_identical": True, "group_count": len(METHODS),
                       "matches_previous_learner_initialization": True})
    report.update(protocol=config["experiment"]["protocol"], initializations=hashes,
                  evaluation_advice=False, observation_guidance=False, test_data_used=False,
                  formal_training_started=False, value_bound=[-6, 10])
    return report, scenes


def source_hashes():
    paths = [Path(__file__), CONFIG, PILOT_CONFIG, ROOT / "src/astar_d3qn/agents/astar_value_repair.py",
             ROOT / "src/astar_d3qn/agents/astar_training_handover.py",
             ROOT / "src/astar_d3qn/agents/d3qn.py", ROOT / "src/astar_d3qn/training/trainer.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_training(config, inputs, seeds, methods, *, mode):
    root = _resolve(config["experiment"]["output_root"]) if mode in ("formal", "pilot") else runtime.unique_check_dir(config, mode)
    planned = [root / method / f"seed_{seed}" for seed in seeds for method in methods]
    if any(p.exists() for p in planned):
        raise FileExistsError("Refusing to overwrite an existing repair run. Interrupted runs stay preserved.")
    distances = static_distances(inputs[0])
    comparisons = []
    for seed in seeds:
        for method in methods:
            directory = root / method / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            training = effective_training(config, seed, mode)
            agent = make_agent(config, inputs[0], seed, method, "cpu" if mode != "formal" else None)
            initial_hash = runtime.state_digest(agent.training_state_dict())
            metadata = dict(config=config, method=method, seed=seed, training_mode=mode,
                            effective_training=asdict(training), initial_state_sha256=initial_hash,
                            source_sha256=source_hashes(), test_data_used=False, evaluation_advice=False)
            write_json(metadata, directory / "run_manifest.json")
            validations, details, behaviors = [], [], []

            def record_validation(step):
                summary, rows, traces = old.evaluate(config, inputs, seed, agent)
                validations.append({"environment_steps_total": step, **summary})
                details.extend({"environment_steps_total": step, **r} for r in rows)
                failures = [t for t in traces if not t["safe_success"]]
                behaviors.extend({"environment_steps_total": step, **trace_behavior(t, distances)} for t in failures)
                write_json(failures, directory / f"validation_failures_{step:06d}.json")
                write_records_csv(behaviors, directory / "failure_behaviors.csv")
                write_records_csv(validations, directory / "validation_curve.csv")
                write_records_csv(details, directory / "validation_details.csv")
                write_records_csv(agent.advice_records(), directory / "advice_budget.csv")
                write_records_csv(agent.repair_records(), directory / "value_learning.csv")
                agent.save_weights(directory / f"model_step_{step:06d}.pth")
                print(f"[{method} seed={seed}] step={step} autonomous safe={summary['safe_success_rate']:.1%} collision={summary['dynamic_collision_rate']:.1%} timeout={summary['timeout_rate']:.1%}", flush=True)

            if mode != "smoke":
                record_validation(0)

            def progress(_episode, records, full_evaluation):
                if full_evaluation:
                    record_validation(records[-1]["environment_steps_total"])
                    write_records_csv(list(records), directory / "training.csv")

            factory = runtime.make_factory(config, *inputs, method="unguided", seed=seed)
            result = train_d3qn([inputs[0]], agent, training, demonstrations=(),
                               reward_config=_reward_config(config), environment_factory=factory,
                               progress_callback=progress if mode in ("formal", "pilot") else None)
            expected_updates = result.environment_steps - training.learning_starts + 1
            if result.environment_steps != training.max_environment_steps or agent.training_action_steps != result.environment_steps or result.gradient_updates != expected_updates:
                raise RuntimeError("Training/advice/update clocks do not match the registered budget.")
            if mode != "smoke" and validations[-1]["environment_steps_total"] != result.environment_steps:
                record_validation(result.environment_steps)
            write_records_csv(list(result.episode_records), directory / "training.csv")
            write_records_csv(agent.advice_records(), directory / "advice_budget.csv")
            write_records_csv(agent.repair_records(), directory / "value_learning.csv")
            agent.save_weights(directory / "model_final.pth")
            final_static = None
            if mode != "smoke":
                _, final_static, trace = static_evaluation(config, inputs[0], "unguided", seed, directory / "model_final.pth")
                write_json(trace, directory / "static_final_trace.json")
                write_json({"result": final_static, "behavior": trace_behavior(trace, distances)}, directory / "static_final.json")
            row = dict(method=method, seed=seed, training_mode=mode, environment_steps=result.environment_steps,
                       gradient_updates=result.gradient_updates,
                       **({k: validations[-1][k] for k in ("safe_success_rate", "dynamic_collision_rate", "timeout_rate")} if validations else {}),
                       static_success=bool(final_static["safe_success"]) if final_static else None,
                       static_steps=final_static["steps"] if final_static else None)
            comparisons.append(row)
            write_records_csv(comparisons, root / "comparison.csv")
            write_json({**metadata, "environment_steps": result.environment_steps,
                        "gradient_updates": result.gradient_updates, "map": inputs[0].manifest(),
                        "route_pool_design_sha256": inputs[2]["design_sha256"],
                        "formal_training_started": mode == "formal", "integration_only": mode in ("smoke", "probe"),
                        "demo_transition_count": 0, "online_replay_capacity": training.replay_capacity,
                        "advice_budget": agent.advice_records(), "value_learning": agent.repair_records(),
                        "final_validation": validations[-1] if validations else None, "final_static": final_static}, directory / "result.json")
            write_json({"complete": True, "environment_steps": result.environment_steps}, directory / "completion.json")
            print(f"[{method} seed={seed}] complete: {result.environment_steps} steps, {result.gradient_updates} updates; {directory}", flush=True)
    write_json({"mode": mode, "seeds": seeds, "methods": methods, "runs": comparisons,
                "formal_training_started": mode == "formal", "test_data_used": False,
                "interpretation": "2000-step probes check integration and numerical behavior only; they do not establish final performance." if mode == "probe" else "Inspect all recorded checkpoints and failures before drawing performance conclusions."}, root / "batch_summary.json")
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    actions = parser.add_mutually_exclusive_group()
    for action in ("check", "smoke", "probe", "pilot", "train"):
        actions.add_argument("--" + action, action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(runtime.REGISTERED_SEEDS) or len(set(args.methods)) != len(args.methods):
        parser.error("Use unique seeds 0-4 and unique methods.")
    config = load_config(_resolve(args.config) if args.config else PILOT_CONFIG if args.pilot else CONFIG)
    inputs = validate_config(config)
    pilot_config = config["experiment"]["protocol"] == "value_bound_teacher_margin_pilot_v1"
    if args.pilot != pilot_config and (args.pilot or args.train or args.probe or args.smoke):
        parser.error("Use --pilot with the independent pilot configuration, and --train with the formal configuration.")
    if not (args.train or args.pilot):
        torch.set_num_threads(1)
    if args.train or args.smoke or args.probe or args.pilot:
        directory = run_training(config, inputs, args.seeds, args.methods,
                                 mode="formal" if args.train else "pilot" if args.pilot else "smoke" if args.smoke else "probe")
    else:
        directory = runtime.unique_check_dir(config, "check")
        report, scenes = check(config, inputs, args.seeds)
        write_json(report, directory / "check_report.json")
        write_json(scenes, directory / "fixed_validation_scenarios.json")
    print(f"Output: {directory}", flush=True)


if __name__ == "__main__":
    main()
