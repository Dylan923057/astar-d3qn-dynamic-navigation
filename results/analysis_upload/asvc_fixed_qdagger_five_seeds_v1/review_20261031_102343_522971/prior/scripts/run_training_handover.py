"""Independent A* action-advice experiment. Default: checks; --train is explicit."""

from __future__ import annotations

import argparse
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
from astar_d3qn.agents.astar_training_handover import AStarTrainingHandoverAgent, METHODS, PRIMARY_METHODS
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.training.trainer import train_d3qn
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything
from train_whole_map_route_pool_pilot import _resolve, _reward_config, _training_config

CONFIG = ROOT / "configs/whole_map_91701_training_handover_v1.yaml"


def validate_config(config):
    expected = load_config(CONFIG)
    if config != expected:
        raise ValueError("Use the registered handover v1 configuration; new settings need a new protocol.")
    baseline = load_config(runtime.CONFIG)
    for section in ("dataset", "dynamic_route_pool", "environment", "reward", "agent", "guidance", "integration"):
        if config[section] != baseline[section]:
            raise ValueError(f"Handover v1 changed shared {section} settings.")
    training = dict(baseline["training"], seeds=list(runtime.REGISTERED_SEEDS))
    if config["training"] != training:
        raise ValueError("Shared training settings changed.")
    if config["experiment"]["protocol"] != "training_action_handover_v1":
        raise ValueError("Wrong experiment protocol.")
    for key, directory in (("output_root", "outputs"), ("check_root", "results")):
        if _resolve(config["experiment"][key]).resolve() != (ROOT / directory / "whole_map_91701_training_handover_v1").resolve():
            raise ValueError("Handover requires its own output directory.")
    advice = config["training_advice"]
    if advice != {"methods": list(METHODS), "probability": 0.8, "decay_environment_steps": 100000,
                  "risk_radius": 1, "rng_seed_offset": 730000,
                  "teacher": "static_astar_replan_from_current_position",
                  "risk_information": "current_and_two_previous_observed_dynamic_frames",
                  "evaluation_advice": False, "observation_guidance": False}:
        raise ValueError("Registered advice specification changed.")
    # Reuse the existing full protocol verification for unchanged physical inputs.
    return runtime.validate_config(baseline)


def make_agent(config, problem, seed, method, device=None, advice_quota=None):
    seed_everything(seed)
    values = dict(config["agent"])
    if device is not None:
        values["device"] = device
    window = config["environment"]["window_size"]
    advice = config["training_advice"]
    return AStarTrainingHandoverAgent(
        D3QNConfig(spatial_shape=(5, window, window), scalar_dim=4,
                   action_dim=config["environment"]["action_count"], seed=seed, **values),
        problem, method=method, probability=advice["probability"],
        decay_steps=advice["decay_environment_steps"], risk_radius=advice["risk_radius"],
        rng_offset=advice["rng_seed_offset"], advice_quota=advice_quota)


def evaluate(config, inputs, seed, agent):
    # All branches use the same zero-added-input environment. Evaluation calls
    # epsilon=0, so the network alone selects every action, including STAY.
    return runtime.evaluate(config, *inputs, "unguided", seed, agent)


def check(config, inputs, seeds):
    base = load_config(runtime.CONFIG)
    report, scenes = runtime.check(base, inputs, seeds)
    kwargs = dict(max_steps=config["environment"]["max_steps"],
                  window_size=config["environment"]["window_size"],
                  reward_config=_reward_config(config), terminate_on_collision=True)
    paired = []
    for seed in seeds:
        agents = [make_agent(config, inputs[0], seed, method, "cpu",
                             advice_quota=[0] * 20 if method == "astar_budget_matched" else None)
                  for method in METHODS]
        hashes = [runtime.state_digest(agent.training_state_dict()) for agent in agents]
        if len(set(hashes)) != 1:
            raise RuntimeError("Initial learner/optimizer/exploration/advice RNG states differ.")
        env = runtime.make_factory(config, *inputs, method="unguided", seed=seed)(inputs[0], **kwargs)
        state = env.reset()
        valid = list(np.flatnonzero(env.action_mask(True)))
        for agent in agents:
            before = runtime.state_digest(agent.training_state_dict())
            with runtime.preserved_evaluation(agent):
                agent.select_action(state, epsilon=0, valid_actions=valid)
            if before != runtime.state_digest(agent.training_state_dict()):
                raise RuntimeError("Greedy evaluation advanced advice state.")
        if any(agent.advice_probability(100000) != 0 or agent.advice_probability(200000) != 0
               for agent in agents[2:]):
            raise RuntimeError("Advice did not fully withdraw for the second half.")
        learner_hash = runtime.state_digest(D3QNAgent.training_state_dict(agents[0]))
        legacy = runtime.make_agent(base, seed, "cpu")
        if learner_hash != runtime.state_digest(legacy.training_state_dict()):
            raise RuntimeError("Learner initialization differs from the existing dense unguided baseline.")
        paired.append({"seed": seed, "initial_state_sha256": hashes[0], "five_groups_identical": True,
                       "initial_learner_state_sha256": learner_hash,
                       "learner_matches_existing_dense_initialization": True})
    report.update({"protocol": config["experiment"]["protocol"], "initializations": paired,
                   "advice_off_at_evaluation": True, "observation_guidance_off": True,
                   "reward_identical_across_methods": True, "test_data_used": False,
                   "formal_training_started": False})
    return report, scenes


def load_matching_quota(config, seed):
    directory = _resolve(config["experiment"]["output_root"]) / "astar_risk_decay" / f"seed_{seed}"
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    if not completion["complete"] or result["config"] != config or result["environment_steps"] != 200000 or result["method"] != "astar_risk_decay" or result["seed"] != seed:
        raise ValueError("Budget-matched control requires its own seed's completed registered risk-handover run.")
    records = result["advice_budget"]
    if len(records) != 20 or any(row["bin_start_step"] != i * 10000 + 1 or row["steps"] != 10000 for i, row in enumerate(records)):
        raise ValueError("Incomplete per-bin advice budget in risk-handover reference.")
    return [int(row["applied"]) for row in records]


def run_training(config, inputs, seeds, methods, *, smoke=False):
    root = runtime.unique_check_dir(config, "smoke") if smoke else _resolve(config["experiment"]["output_root"])
    planned = [root / method / f"seed_{seed}" for seed in seeds for method in methods]
    if any(path.exists() for path in planned):
        raise FileExistsError("Refusing to overwrite an existing handover run.")
    # Resolve a missing prerequisite before creating any output directory.
    if not smoke and "astar_budget_matched" in methods and "astar_risk_decay" not in methods:
        for seed in seeds:
            load_matching_quota(config, seed)
    # The matched control depends on training action counts, never validation scores.
    methods = [method for method in METHODS if method in methods]
    for seed in seeds:
        for method in methods:
            directory = root / method / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            training = _training_config(config, seed)
            if smoke:
                training = replace(training, max_environment_steps=8, batch_size=4,
                                   learning_starts=4, epsilon_decay_environment_steps=8,
                                   progress_interval_environment_steps=None)
            quota = ([10000] + [0] * 19 if smoke else load_matching_quota(config, seed)) if method == "astar_budget_matched" else None
            agent = make_agent(config, inputs[0], seed, method, "cpu" if smoke else None, advice_quota=quota)
            initial_hash = runtime.state_digest(agent.training_state_dict())
            validations, details = [], []
            write_json({"config": config, "method": method, "seed": seed,
                        "initial_state_sha256": initial_hash, "effective_training": asdict(training),
                        "matching_advice_quota": quota},
                       directory / "run_manifest.json")

            def record_validation(step):
                summary, rows, traces = evaluate(config, inputs, seed, agent)
                validations.append({"environment_steps_total": step, **summary})
                details.extend({"environment_steps_total": step, **row} for row in rows)
                write_records_csv(validations, directory / "validation_curve.csv")
                write_records_csv(details, directory / "validation_details.csv")
                write_json([trace for trace in traces if not trace["safe_success"]],
                           directory / f"validation_failures_{step:06d}.json")
                write_records_csv(agent.advice_records(), directory / "advice_budget.csv")
                agent.save_weights(directory / f"model_step_{step:06d}.pth")
                print(f"[{method} seed={seed}] step={step} network-only safe={summary['safe_success_rate']:.1%} collision={summary['dynamic_collision_rate']:.1%} timeout={summary['timeout_rate']:.1%}", flush=True)

            if not smoke:
                record_validation(0)

            def progress(_episode, records, full_evaluation):
                if full_evaluation:
                    record_validation(records[-1]["environment_steps_total"])
                    write_records_csv(list(records), directory / "training.csv")

            factory = runtime.make_factory(config, *inputs, method="unguided", seed=seed)
            result = train_d3qn([inputs[0]], agent, training, demonstrations=(),
                               reward_config=_reward_config(config), environment_factory=factory,
                               progress_callback=progress if not smoke else None)
            if agent.training_action_steps != result.environment_steps or result.gradient_updates <= 0:
                raise RuntimeError("Advice clock/training steps or optimizer integration did not match.")
            if not smoke and quota is not None and [row["applied"] for row in agent.advice_records()] != quota:
                raise RuntimeError("Actual A* interventions do not match the risk-handover reference.")
            if not smoke and (not validations or validations[-1]["environment_steps_total"] != result.environment_steps):
                record_validation(result.environment_steps)
            write_records_csv(list(result.episode_records), directory / "training.csv")
            write_records_csv(agent.advice_records(), directory / "advice_budget.csv")
            if not smoke:
                agent.save_weights(directory / "model_final.pth")
            write_json({"method": method, "seed": seed, "config": config,
                        "initial_state_sha256": initial_hash, "effective_training": asdict(training),
                        "environment_steps": result.environment_steps, "gradient_updates": result.gradient_updates,
                        "map": inputs[0].manifest(), "route_pool_design_sha256": inputs[2]["design_sha256"],
                        "demo_transition_count": 0, "online_replay_capacity": training.replay_capacity,
                        "formal_training_started": not smoke, "integration_only": smoke,
                        "training_mode": "integration" if smoke else "formal",
                        "test_data_used": False, "evaluation_advice": False,
                        "advice_budget": agent.advice_records(),
                        "matching_advice_quota": quota,
                        "final_validation": validations[-1] if validations else None}, directory / "result.json")
            write_json({"complete": True, "environment_steps": result.environment_steps}, directory / "completion.json")
            print(f"[{method} seed={seed}] complete: {result.environment_steps} steps, {result.gradient_updates} updates; {directory}", flush=True)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--smoke", action="store_true", help="Eight CPU integration steps per branch.")
    actions.add_argument("--train", action="store_true", help="Explicit 200000-step training.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(runtime.REGISTERED_SEEDS) or len(set(args.methods)) != len(args.methods):
        parser.error("Use unique seeds 0-4 and unique methods.")
    config = load_config(_resolve(args.config))
    inputs = validate_config(config)
    if not args.train:
        torch.set_num_threads(1)
    if args.train or args.smoke:
        directory = run_training(config, inputs, args.seeds, args.methods, smoke=args.smoke)
    else:
        directory = runtime.unique_check_dir(config, "check")
        report, scenes = check(config, inputs, args.seeds)
        write_json(report, directory / "check_report.json")
        write_json(scenes, directory / "fixed_validation_scenarios.json")
    print(f"Output: {directory}", flush=True)


if __name__ == "__main__":
    main()
