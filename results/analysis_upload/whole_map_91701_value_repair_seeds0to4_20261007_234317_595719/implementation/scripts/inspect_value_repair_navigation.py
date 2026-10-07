"""Read-only greedy replay and observation counterfactuals for audited completed models.

Only the real observation selects the executed action. The obstacle-free
observation and environment collision oracle are retrospective diagnostics.
No training, planner, action shield, or test scenes are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_value_repair as entry
from analyze_runtime_path_pilot import load_csv, static_distances
from astar_d3qn.envs.types import Observation
from astar_d3qn.utils.io import write_json, write_records_csv


def inspect(source, output, seeds=(0, 1), methods=("advice_margin", "advice_bound_margin")):
    summaries, changes, scene_rows = [], [], []
    prior_collision_ids = {f"validation_seed91701999_episode{i:06d}" for i in (15, 29, 30, 39)}
    torch.set_num_threads(1)
    for seed in seeds:
        for method in methods:
            directory = source / method / f"seed_{seed}"
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            budget = result["effective_training"]["max_environment_steps"]
            if (result["environment_steps"] != budget or result["test_data_used"]
                    or result["evaluation_advice"]):
                raise ValueError("Expected a completed network-only validation run.")
            config = result["config"]
            inputs = entry.validate_config(config)
            distances = static_distances(inputs[0])
            agent = entry.runtime.make_agent(config, seed, "cpu")  # Ordinary D3QN: no advice wrapper.
            weight = directory / "model_final.pth"
            original_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
            agent.load_weights(weight)
            expected = {r["scenario_id"]: r for r in load_csv(directory / "validation_details.csv")
                        if int(r["environment_steps_total"]) == budget}
            if len(expected) != 50:
                raise ValueError("Expected all 50 final validation scenes.")
            factory = entry.runtime.make_factory(config, *inputs, method="unguided", seed=seed,
                                                 validation=True, trace=True)
            safe = collision = timeout = wait_steps = differing = avoided = 0
            selected_traces = []
            with entry.runtime.preserved_evaluation(agent):
                for _ in range(50):
                    env = factory(inputs[0], max_steps=300, window_size=15,
                                  reward_config=entry._reward_config(config), terminate_on_collision=True)
                    state = env.reset()
                    scene_changes, waits = [], 0
                    total_reward = 0.
                    for step in range(1, 301):
                        valid = list(np.flatnonzero(env.action_mask(True)))
                        action = agent.select_action(state, 0, valid)
                        spatial = state.spatial.copy()
                        spatial[1:4] = 0
                        counterfactual_action = agent.select_action(Observation(spatial, state.scalars.copy()), 0, valid)
                        if action != counterfactual_action:
                            actual_risk = env.dynamic_action_collision_risk(action, predict_next=True)
                            alternative_risk = env.dynamic_action_collision_risk(counterfactual_action, predict_next=True)
                            row = {"method": method, "seed": seed, "scenario_id": env.scenario_id,
                                   "step": step, "position": env.position,
                                   "dynamic_positions": env.dynamic_positions,
                                   "actual_action": action, "zero_dynamic_observation_action": counterfactual_action,
                                   "actual_one_step_collision_risk": actual_risk,
                                   "counterfactual_one_step_collision_risk": alternative_risk,
                                   "avoided_one_step_collision": alternative_risk and not actual_risk}
                            changes.append(row)
                            scene_changes.append(row)
                            differing += 1
                            avoided += int(row["avoided_one_step_collision"])
                        transition = env.step(action)  # Execute the original greedy choice only.
                        total_reward += transition.reward
                        waits += int(action == 4)
                        state = transition.observation
                        if transition.done:
                            break
                    reason = transition.info["termination_reason"]
                    success = bool(transition.info["reached"] and not transition.info["collision"])
                    original = expected[env.scenario_id]
                    if (success != bool(float(original["safe_success"])) or reason != original["termination_reason"]
                            or step != int(original["steps"]) or waits != int(original["wait_steps"])
                            or not np.isclose(total_reward, float(original["reward"]))):
                        raise RuntimeError("Ordinary-D3QN greedy replay differs from original final validation.")
                    safe += int(success)
                    collision += int(reason == "collision")
                    timeout += int(reason == "timeout")
                    wait_steps += waits
                    scene_rows.append({"method": method, "seed": seed, "scenario_id": env.scenario_id,
                                       "safe_success": success, "termination_reason": reason, "steps": step,
                                       "wait_steps": waits, "dynamic_sensitive_action_steps": len(scene_changes),
                                       "avoided_one_step_collision_steps": sum(r["avoided_one_step_collision"] for r in scene_changes),
                                       "final_static_remaining_steps": distances[env.position],
                                       "failed_in_2000_step_probe": env.scenario_id in prior_collision_ids})
                    if (not success or env.scenario_id in prior_collision_ids
                            or any(r["avoided_one_step_collision"] for r in scene_changes)):
                        selected_traces.append({"scenario_id": env.scenario_id, "safe_success": success,
                                                "termination_reason": reason, "steps": env.trace})
            if hashlib.sha256(weight.read_bytes()).hexdigest() != original_sha:
                raise RuntimeError("Source weights changed during read-only replay.")
            row = {"method": method, "seed": seed, "environment_steps": budget,
                   "safe_success_rate": safe / 50,
                   "dynamic_collision_rate": collision / 50, "timeout_rate": timeout / 50,
                   "total_wait_steps": wait_steps, "dynamic_sensitive_action_steps": differing,
                   "avoided_one_step_collision_steps": avoided,
                   "ordinary_d3qn_matches_all_original_50_scene_outcomes": True,
                   "source_weights_sha256": original_sha}
            summaries.append(row)
            write_json(selected_traces, output / f"navigation_examples_{method}_seed{seed}.json")
            print(json.dumps(row), flush=True)
    write_records_csv(summaries, output / "navigation_replay_summary.csv")
    write_records_csv(scene_rows, output / "navigation_replay_scenes.csv")
    write_records_csv(changes, output / "dynamic_observation_action_changes.csv")
    write_json({"source": str(source), "seeds": list(seeds), "methods": list(methods),
                "evaluation_epsilon": 0, "ordinary_d3qn_only": True,
                "test_data_used": False, "training_started": False,
                "counterfactual": "Zero only the three observed dynamic channels at the same real state; retain static channels, goal scalars and static action mask.",
                "collision_oracle": "Used after action selection solely to diagnose the two candidate actions; never given to the policy or used to select an executed action.",
                "limitation": "These are one-step action counterfactuals, not closed-loop rollouts with missing observations or proof of generalization.",
                "runs": summaries}, output / "navigation_replay_verification.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/whole_map_91701_value_repair_pilot_v1")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=entry.METHODS,
                        default=["advice_margin", "advice_bound_margin"])
    args = parser.parse_args()
    output = args.analysis.resolve()
    if not (output / "verification.json").is_file():
        parser.error("Use an existing successfully audited analysis directory.")
    if (output / "navigation_replay_verification.json").exists():
        parser.error("Read-only replay already exists; preserve that report.")
    audit = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if (Path(audit["source_root"]).resolve() != args.source.resolve()
            or not set(args.seeds).issubset(audit["seeds"])
            or not set(args.methods).issubset(audit["methods"])
            or audit["test_data_used"] or not audit["all_recorded_checkpoints_verified"]):
        parser.error("Source, seeds and methods must belong to this verified analysis.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("Use unique seeds and methods.")
    inspect(args.source.resolve(), output, args.seeds, args.methods)


if __name__ == "__main__":
    main()
