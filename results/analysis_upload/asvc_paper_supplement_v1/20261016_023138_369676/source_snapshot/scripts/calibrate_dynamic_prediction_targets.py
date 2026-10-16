"""Estimate next-state dynamic occupancy balance from frozen training scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.replay_adaptation import flatten_pairs, make_agent, make_env
from astar_d3qn.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-transitions", type=int, default=10000)
    parser.add_argument(
        "--output",
        default="outputs/dynamic_prediction_preflight_v1/target_balance.json",
    )
    args = parser.parse_args()
    if args.max_transitions <= 0:
        raise SystemExit("--max-transitions must be positive.")

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    entry = manifest["maps"][args.map_index]
    problem = problem_from_record(entry["problem"])
    scenes = flatten_pairs(entry["scenarios"]["splits"]["train"])
    branch = (
        ROOT / config["output_root"] / "formal" / problem.map_id
        / f"seed_{args.seed}" / "schedule_decay"
    )
    agent = make_agent(config, args.seed, "cpu")
    agent.load_weights(branch / "model_final.pth")

    positive = 0
    total = 0
    transitions = 0
    coordinate_checks = 0
    for scene in scenes:
        env = make_env(problem, config, scene)
        observation = env.reset()
        target_channel = env.current_dynamic_channel
        for _ in range(config["max_episode_steps"]):
            valid = np.flatnonzero(env.action_mask(True)).tolist()
            action = agent.select_action(observation, 0.0, valid)
            result = env.step(action)
            target = result.observation.spatial[target_channel]
            positive += int(target.sum())
            total += int(target.size)
            transitions += 1

            expected = np.zeros_like(target)
            radius = target.shape[0] // 2
            for row, column in env.dynamic_positions:
                local_row = row - env.position[0] + radius
                local_column = column - env.position[1] + radius
                if 0 <= local_row < target.shape[0] and 0 <= local_column < target.shape[1]:
                    expected[local_row, local_column] = 1.0
            if not np.array_equal(target, expected):
                raise RuntimeError("next_state dynamic target disagrees with the next local frame.")
            coordinate_checks += 1
            observation = result.observation
            if result.done or transitions >= args.max_transitions:
                break
        if transitions >= args.max_transitions:
            break

    if positive == 0:
        raise SystemExit("No positive dynamic target cells were observed.")
    negative = total - positive
    raw_pos_weight = negative / positive
    recommended = min(20.0, max(1.0, raw_pos_weight))
    output = {
        "source_split": "train",
        "policy": str((branch / "model_final.pth").relative_to(ROOT)),
        "target": "transition.next_state[current_dynamic_channel]",
        "current_dynamic_channel": target_channel,
        "transitions": transitions,
        "coordinate_checks": coordinate_checks,
        "positive_cells": positive,
        "negative_cells": negative,
        "positive_fraction": positive / total,
        "raw_negative_to_positive_ratio": raw_pos_weight,
        "recommended_fixed_pos_weight": recommended,
        "recommendation_rule": "clip negative/positive ratio to [1, 20]",
    }
    destination = ROOT / args.output
    write_json(output, destination)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
