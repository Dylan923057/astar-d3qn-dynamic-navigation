"""Inspect action values and margin-label coverage at saved validation failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from astar_d3qn.core.grid import ACTION_NAMES, action_between
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.replay_adaptation import (
    action_risk_margin_masks,
    make_agent,
    make_env,
)
from astar_d3qn.utils.io import write_records_csv


def flatten_validation_pairs(pairs):
    return [
        {
            **pair[condition],
            "pair_id": pair["pair_id"],
            "reference_collision_step": pair["first_reference_collision_step"],
            "scenario_id": f"{pair['pair_id']}_{condition}",
            "condition": condition,
        }
        for pair in pairs
        for condition in ("control", "conflict")
    ]


def capture_dynamic_collision(agent, problem, config, scene, reference_actions):
    env = make_env(problem, config, scene)
    observation = env.reset()
    for step in range(1, config["max_episode_steps"] + 1):
        valid = np.flatnonzero(env.action_mask(True)).tolist()
        action = agent.select_action(observation, 0.0, valid)
        position = tuple(env.position)
        safe, all_blocked = action_risk_margin_masks(
            env,
            reference_actions,
            scope="all_actions",
        )
        demo_safe, demo_blocked = action_risk_margin_masks(
            env,
            reference_actions,
            scope="demo_action",
        )
        valid_mask = np.asarray(env.action_mask(True), dtype=bool)
        risks = np.asarray(
            [
                bool(env.dynamic_action_collision_risk(candidate))
                if valid_mask[candidate]
                else False
                for candidate in range(len(valid_mask))
            ]
        )
        result = env.step(action)
        if result.info["collision_type"] == "dynamic":
            return {
                "step": step,
                "position": position,
                "observation": observation,
                "failing_action": action,
                "safe": safe,
                "all_blocked": all_blocked,
                "demo_safe": demo_safe,
                "demo_blocked": demo_blocked,
                "valid": valid_mask,
                "risks": risks,
            }
        observation = result.observation
        if result.done:
            break
    return None


def q_values(agent, observation):
    with torch.no_grad():
        values = agent.policy_network(
            torch.as_tensor(
                observation.spatial,
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0),
            torch.as_tensor(
                observation.scalars,
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0),
        )[0]
    return values.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--source-failure-seed", type=int, default=1)
    parser.add_argument("--network-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output",
        default="outputs/action_ranking_preflight_v1",
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    entry = manifest["maps"][args.map_index]
    problem = problem_from_record(entry["problem"])
    scenes = flatten_validation_pairs(entry["scenarios"]["splits"]["validation"])
    source_root = ROOT / config["output_root"] / "formal" / problem.map_id
    destination = ROOT / args.output / problem.map_id
    if destination.exists():
        raise SystemExit(f"Preflight output already exists: {destination}")

    agents = {}
    for seed in sorted(set(args.network_seeds + [args.source_failure_seed])):
        branch = source_root / f"seed_{seed}" / "schedule_decay"
        result = json.loads((branch / "result.json").read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("smoke"):
            raise SystemExit(f"Not a complete formal time-decay run: {branch}")
        agent = make_agent(config, seed, "cpu")
        agent.load_weights(branch / "model_final.pth")
        agents[seed] = agent

    reference_actions = {
        position: int(action_between(position, following))
        for position, following in zip(
            problem.nominal_path,
            problem.nominal_path[1:],
        )
    }
    records = []
    source_agent = agents[args.source_failure_seed]
    for scene in scenes:
        failure = capture_dynamic_collision(
            source_agent,
            problem,
            config,
            scene,
            reference_actions,
        )
        if failure is None:
            continue
        safe = failure["safe"]
        all_blocked = failure["all_blocked"]
        demo_safe = failure["demo_safe"]
        demo_blocked = failure["demo_blocked"]
        if safe is None or all_blocked is None:
            raise RuntimeError("A captured dynamic collision lacks an action-risk label.")
        valid = failure["valid"]
        risks = failure["risks"]
        for seed in args.network_seeds:
            values = q_values(agents[seed], failure["observation"])
            masked = np.where(valid, values, -np.inf)
            greedy = int(masked.argmax())
            records.append({
                "scenario_id": scene["scenario_id"],
                "source_failure_seed": args.source_failure_seed,
                "critical_step": failure["step"],
                "position": list(failure["position"]),
                "network_seed": seed,
                "failing_network_action": failure["failing_action"],
                "greedy_action": greedy,
                "greedy_action_name": ACTION_NAMES[greedy],
                "greedy_action_risky": int(risks[greedy]),
                "q_up": float(values[0]),
                "q_down": float(values[1]),
                "q_left": float(values[2]),
                "q_right": float(values[3]),
                "q_stay": float(values[4]),
                "best_safe_q": float(values[safe].max()),
                "best_risky_q": float(values[all_blocked].max()),
                "safe_minus_risky_q": float(
                    values[safe].max() - values[all_blocked].max()
                ),
                "demo_action_margin_covers_state": int(demo_safe is not None),
                "demo_blocked_actions": (
                    np.flatnonzero(demo_blocked).tolist()
                    if demo_blocked is not None
                    else []
                ),
                "all_action_blocked_actions": np.flatnonzero(all_blocked).tolist(),
                "all_action_safe_actions": np.flatnonzero(safe).tolist(),
            })

    if not records:
        raise SystemExit("The source model produced no validation dynamic collisions.")
    destination.mkdir(parents=True)
    write_records_csv(records, destination / "action_value_comparison.csv")
    failure_count = len({row["scenario_id"] for row in records})
    old_covered = len({
        row["scenario_id"]
        for row in records
        if row["demo_action_margin_covers_state"]
    })
    risky_top = {
        seed: sum(
            row["greedy_action_risky"]
            for row in records
            if row["network_seed"] == seed
        )
        for seed in args.network_seeds
    }
    report = [
        "# Action-ranking preflight",
        "",
        "Validation-only inspection; these states are never inserted into training.",
        "",
        f"- Captured dynamic-collision states: {failure_count}",
        f"- Existing demo-action margin covers: {old_covered}/{failure_count}",
        f"- Full-action margin covers: {failure_count}/{failure_count}",
        "- Risky greedy actions on identical observations: "
        + ", ".join(
            f"seed {seed} = {risky_top[seed]}/{failure_count}"
            for seed in args.network_seeds
        ),
        "",
        "The preflight verifies label timing and scope only. It does not establish",
        "training benefit and does not read or select on the test split.",
    ]
    (destination / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "failure_states": failure_count,
        "demo_scope_covered": old_covered,
        "full_scope_covered": failure_count,
        "risky_top_by_seed": risky_top,
    }, indent=2))


if __name__ == "__main__":
    main()
