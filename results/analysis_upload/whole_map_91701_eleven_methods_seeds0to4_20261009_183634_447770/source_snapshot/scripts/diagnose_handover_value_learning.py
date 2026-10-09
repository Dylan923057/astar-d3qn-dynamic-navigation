"""Bounded value-learning diagnostics; never starts the formal training loop.

Existing checkpoint weights are read only. A small, shared dataset is collected
from the training scene pool with a frozen behavior policy. Optional local TD
probes use fresh optimizers and synchronized targets: they are mechanism tests,
not continuations or exact reproductions of the original training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_training_handover as entry
from astar_d3qn.agents.d3qn import double_dqn_targets
from astar_d3qn.core.grid import ACTION_NAMES, Action, action_between
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.replay.transition import Transition
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.training.trainer import linear_epsilon_at_environment_step

METHODS = ("unguided", "astar_fixed", "astar_decay")


def tensors(batch):
    spatial = torch.as_tensor(np.stack([t.state.spatial for t in batch]))
    scalars = torch.as_tensor(np.stack([t.state.scalars for t in batch]))
    next_spatial = torch.as_tensor(np.stack([t.next_state.spatial for t in batch]))
    next_scalars = torch.as_tensor(np.stack([t.next_state.scalars for t in batch]))
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.long)
    rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
    terminated = torch.as_tensor([t.terminated for t in batch], dtype=torch.float32)
    mask = torch.as_tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.bool)
    return spatial, scalars, next_spatial, next_scalars, actions, rewards, terminated, mask


def inspect_batch(agent, batch, next_advice):
    s, x, ns, nx, actions, rewards, terminated, mask = tensors(batch)
    with torch.no_grad():
        q = agent.policy_network(s, x)
        nq = agent.policy_network(ns, nx)
        tq = agent.target_network(ns, nx)
        next_actions = nq.masked_fill(~mask, -torch.inf).argmax(dim=1)
        targets = double_dqn_targets(rewards, terminated, agent.config.gamma, nq, tq, mask)
        executed = q.gather(1, actions[:, None])[:, 0]
        active = terminated == 0
        teacher = torch.as_tensor(next_advice, dtype=torch.long)
        terminal = terminated == 1
    current_mask = torch.as_tensor([[t.state.spatial[0, t.state.spatial.shape[1] // 2 + dr,
                                                     t.state.spatial.shape[2] // 2 + dc] < 0.5
                                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))]
                                   for t in batch], dtype=torch.bool)
    current_max = q.masked_fill(~current_mask, -torch.inf).amax(dim=1)
    return {"q_executed_mean": executed.mean().item(), "q_executed_max": executed.max().item(),
            "current_policy_max_valid_q_mean": current_max.mean().item(),
            "current_policy_max_valid_q_max": current_max.max().item(),
            "target_mean": targets.mean().item(), "target_max": targets.max().item(),
            "fraction_targets_above_10": (targets > 10.001).float().mean().item(),
            "nonterminal_next_greedy_action_counts": dict(Counter(ACTION_NAMES[i] for i in next_actions[active].tolist())),
            "next_greedy_differs_from_static_teacher_fraction": (next_actions[active] != teacher[active]).float().mean().item(),
            "terminal_count": int(terminal.sum()),
            "terminal_targets_equal_actual_rewards": bool(torch.allclose(targets[terminal], rewards[terminal])),
            "synchronized_target_surrogate": True}


def collect(config, inputs, weights, step, count):
    collector = entry.make_agent(config, inputs[0], 0, "astar_fixed", "cpu")
    collector.load_weights(weights)
    collector.training_action_steps = step
    factory = entry.runtime.make_factory(config, *inputs, method="unguided", seed=0)
    kwargs = dict(max_steps=300, window_size=15, terminate_on_collision=True,
                  reward_config=entry._reward_config(config))
    factory.reset_schedule()
    env = factory(inputs[0], **kwargs)
    state = env.reset()
    transitions, advice, contexts, imitation_labels = [], [], [], []
    counts = Counter()
    training = entry._training_config(config, 0)
    for index in range(count):
        before = state.spatial.copy(), state.scalars.copy()
        epsilon = linear_epsilon_at_environment_step(step + index, training)
        candidates = np.flatnonzero(env.action_mask(True)).tolist()
        action = collector.select_action(state, epsilon, candidates)
        safe_teacher_label = (action == collector.static_advice(state)
                              and not collector.observed_risk(state, action))
        result = env.step(action)
        if not np.array_equal(before[0], state.spatial) or not np.array_equal(before[1], state.scalars):
            raise RuntimeError("Environment mutated a previously returned observation.")
        transition = Transition(state, action, result.reward, result.observation,
                                result.terminated, env.action_mask(True))
        next_action = collector.static_advice(result.observation)
        if not result.done and next_action is None:
            raise RuntimeError("No next static route in a nonterminal state.")
        transitions.append(transition)
        imitation_labels.append(safe_teacher_label and not result.info["collision"])
        advice.append(int(Action.STAY) if next_action is None else next_action)
        contexts.append({"scenario_id": env.scenario_id, "position": tuple(env.position),
                         "termination_reason": result.info["termination_reason"],
                         "collision_type": result.info["collision_type"]})
        counts[ACTION_NAMES[action]] += 1
        state = result.observation
        if result.done:
            counts[result.info["termination_reason"]] += 1
            if not result.terminated and result.info["termination_reason"] != "timeout":
                raise RuntimeError("Unexpected episode-end flag.")
            env = factory(inputs[0], **kwargs)
            state = env.reset()
    goals = [t for t, c in zip(transitions, contexts) if c["termination_reason"] == "goal"]
    collisions = [t for t, c in zip(transitions, contexts) if c["termination_reason"] == "collision"]
    if not goals or not collisions or any(not t.terminated or t.reward != 10 for t in goals) or any(not t.terminated or t.reward != -1 for t in collisions):
        raise RuntimeError("Goal/collision transition auditing failed.")
    return transitions, advice, contexts, dict(counts), imitation_labels


def static_path_scan(config, problem, weights, seed=0):
    agent = entry.runtime.make_agent(config, seed, "cpu")
    agent.load_weights(weights)
    env = PathGuidanceEnvironment(DynamicGridNavigationEnv(problem, (), window_size=15,
                                                         max_steps=300, terminate_on_collision=True,
                                                         reward_config=entry._reward_config(config)), enabled=False)
    state = env.reset()
    output = []
    for current, following in zip(problem.nominal_path, problem.nominal_path[1:]):
        action = int(action_between(current, following))
        with torch.no_grad():
            q = agent.policy_network(torch.as_tensor(state.spatial)[None], torch.as_tensor(state.scalars)[None])[0].tolist()
        valid = np.flatnonzero(env.action_mask(True)).tolist()
        selected = max(valid, key=lambda i: q[i])
        output.append({"position": current, "static_teacher_action": ACTION_NAMES[action],
                       "greedy_action": ACTION_NAMES[selected], "teacher_q": q[action],
                       "max_valid_q": q[selected], "q": dict(zip(ACTION_NAMES, q))})
        state = env.step(action).observation
    return output


def local_probes(config, weights, batch, advice, updates, imitation_labels):
    # Each branch starts with identical policy/target weights and a fresh Adam
    # optimizer, and sees the exact same preselected mini-batches.
    replay = UniformReplayBuffer(len(batch), seed=9281)
    replay.extend(batch)
    batches = [replay.sample(64) for _ in range(updates)]
    next_teacher_by_id = {id(t): a for t, a in zip(batch, advice)}
    imitation_by_id = {id(t): label for t, label in zip(batch, imitation_labels)}
    outcomes = []
    for variant in ("original_max_bootstrap", "static_teacher_bootstrap", "return_bound_target",
                    "return_bound_plus_observed_safe_teacher_margin"):
        agent = entry.runtime.make_agent(config, 0, "cpu")
        agent.load_weights(weights)
        before = inspect_batch(agent, batch, advice)
        losses = []
        for mini_batch in batches:
            teacher_actions = torch.as_tensor([next_teacher_by_id[id(t)] for t in mini_batch])

            def diagnostic_targets(rewards, terminated, gamma, policy_next_q, target_next_q, valid_action_mask=None):
                if variant == "static_teacher_bootstrap":
                    bootstrap = target_next_q.gather(1, teacher_actions[:, None])[:, 0]
                    return rewards + gamma * bootstrap * (1 - terminated)
                value = double_dqn_targets(rewards, terminated, gamma, policy_next_q, target_next_q, valid_action_mask)
                return value.clamp(-6.0, 10.0) if variant.startswith("return_bound") else value

            with patch("astar_d3qn.agents.d3qn.double_dqn_targets", diagnostic_targets):
                use_margin = variant == "return_bound_plus_observed_safe_teacher_margin"
                metrics = agent.train_batch(mini_batch,
                    demonstration_mask=[imitation_by_id[id(t)] for t in mini_batch] if use_margin else None,
                    demo_margin=0.8 if use_margin else 0.0, demo_loss_weight=1.0 if use_margin else 0.0)
            losses.append(metrics["td_loss"])
        # Targets remain frozen during <250 local updates, matching the original
        # sync interval. Sync once for the post-fit diagnostic so comparisons use
        # the newly fitted critic, rather than interpreting a stale target copy.
        agent.sync_target()
        after = inspect_batch(agent, batch, advice)
        outcomes.append({"variant": variant, "updates": updates, "before": before, "after": after,
                         "mean_probe_td_loss": float(np.mean(losses)),
                         "max_bootstrap_surrogate_target_change": after["target_mean"] - before["target_mean"]})
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-steps", type=int, default=1000)
    parser.add_argument("--probe-updates", type=int, default=64)
    args = parser.parse_args(argv)
    if not 500 <= args.collection_steps <= 1500 or not 1 <= args.probe_updates <= 128:
        parser.error("Bounded diagnostic: 500-1500 frozen-policy steps and 1-128 local probe updates.")
    torch.set_num_threads(1)
    config = entry.load_config(entry.CONFIG)
    inputs = entry.validate_config(config)
    source = entry._resolve(config["experiment"]["output_root"])
    weights = next(iter(sorted((source / "astar_fixed/seed_0").glob("model_step_01*.pth"))))
    step = int(weights.stem.split("_")[-1])
    paths = [source / method / "seed_0/model_final.pth" for method in METHODS] + [weights]
    original_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output = entry.runtime.unique_check_dir(config, "value_diagnosis")
    print(f"Collecting {args.collection_steps} frozen-policy training-pool steps; no updates...", flush=True)
    batch, advice, contexts, counts, imitation_labels = collect(config, inputs, weights, step, args.collection_steps)
    snapshots = []
    for method in METHODS:
        directory = source / method / "seed_0"
        checkpoints = sorted(directory.glob("model_step_*.pth"))
        selected = [checkpoints[1], min(checkpoints, key=lambda p: abs(int(p.stem.split('_')[-1]) - 50000)), checkpoints[-1]]
        for checkpoint in selected:
            agent = entry.runtime.make_agent(config, 0, "cpu")
            agent.load_weights(checkpoint)
            snapshots.append({"method": method, "step": int(checkpoint.stem.split('_')[-1]),
                              "shared_training_pool_observations": inspect_batch(agent, batch, advice)})
            scan = static_path_scan(config, inputs[0], checkpoint)
            entry.write_json(scan, output / f"static_path_{method}_{checkpoint.stem}.json")
            print(f"{method} {checkpoint.stem}: mean target={snapshots[-1]['shared_training_pool_observations']['target_mean']:.2f}", flush=True)
    print(f"Running {args.probe_updates} paired local updates per probe variant...", flush=True)
    probes = local_probes(config, weights, batch, advice, args.probe_updates, imitation_labels)
    unchanged = all(hashlib.sha256(path.read_bytes()).hexdigest() == original_hashes[str(path)] for path in paths)
    if not unchanged:
        raise RuntimeError("A source checkpoint changed during diagnostics.")
    report = {"formal_training_started": False, "test_data_used": False,
              "source_weights_unchanged": unchanged, "seed": 0, "source_checkpoint": str(weights),
              "frozen_collection_steps": len(batch), "collected_action_and_terminal_counts": counts,
              "collection_gradient_updates": 0, "diagnostic_probe_updates_per_variant": args.probe_updates,
              "probe_optimizer": "fresh_Adam_no_original_optimizer_checkpoint",
              "probe_target": "initially_synchronized_no_original_target_checkpoint",
              "post_probe_target_resynchronized_for_diagnostic": True,
              "probe_imitation_label_count": sum(imitation_labels),
              "data_scope": "new_train_pool_probe_not_original_replay_reconstruction",
              "transitions_goal_and_collision_flags_and_rewards_correct": True,
              "observations_not_mutated_after_step": True, "checkpoint_diagnostics": snapshots,
              "local_probe_outcomes": probes,
              "source_hashes": original_hashes}
    entry.write_json(report, output / "diagnosis.json")
    for result in probes:
        print(json.dumps({"variant": result["variant"], "target_before": result["before"]["target_mean"],
                          "target_after": result["after"]["target_mean"],
                          "executed_q_after": result["after"]["q_executed_mean"]}), flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
