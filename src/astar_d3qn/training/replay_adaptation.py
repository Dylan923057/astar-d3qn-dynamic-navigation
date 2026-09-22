"""Matched-checkpoint experiment for the TOTAL effect of persistent replay.

This is deliberately separate from the legacy curriculum runner. No adaptive
replay, imitation loss, oracle action labels, or dynamic action masks are used.
"""
from __future__ import annotations

import copy
import hashlib
import random
from collections import deque
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.core.grid import Action, action_between, manhattan, move
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.evaluation.behavior_oracle import shortest_safe_plan
from astar_d3qn.maps.adaptation import spec_from_record
from astar_d3qn.replay.demo import (
    IndexedRiskReplay,
    PersistentDemoReplay,
    RiskCoverageHandoverReplay,
    SafeInterventionReplay,
)
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.demo_collector import collect_astar_demonstrations
from astar_d3qn.utils.io import write_json, write_records_csv


def state_digest(value) -> str:
    """Stable content fingerprint including tensor values, optimizer and RNG state."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif is_dataclass(item):
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}".encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\x00")
    visit(value)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_agent(config, seed, device):
    return D3QNAgent(D3QNConfig(
        spatial_shape=(4, config["window_size"], config["window_size"]), scalar_dim=2, action_dim=5,
        learning_rate=config["learning_rate"], gamma=config["gamma"],
        target_sync_interval=config["target_sync_interval"], hidden_dim=config["hidden_dim"],
        device=device, seed=seed))


def make_env(problem, config, scene=None):
    if scene is None:
        obstacle_records = []
    elif "obstacles" in scene:
        obstacle_records = scene["obstacles"]
    else:
        obstacle_records = [scene["obstacle"]]
    return DynamicGridNavigationEnv(
        problem, [spec_from_record(record) for record in obstacle_records],
        max_steps=config["max_episode_steps"], reward_config=RewardConfig(**config["reward"]),
        terminate_on_collision=True, window_size=config["window_size"])


def collect_demos(problem, config):
    return collect_astar_demonstrations(
        [problem], config["demo_episodes"], config["demo_seed"],
        max_steps=config["max_episode_steps"], reward_config=RewardConfig(**config["reward"]),
        window_size=config["window_size"], spatial_channels=4, mask_static_invalid_actions=True)


def snapshot(agent, replay):
    return copy.deepcopy({
        "agent": agent.training_state_dict(),
        "online_replay": replay.online.state_dict(),
        "demonstrations": replay.demonstration_snapshot(),
        "demo_rng": replay._rng.getstate(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    })


def restore(config, base, fraction, seed, device, *, replay_schedule=None,
            safe_sample_count=0, risk_sample_count=0):
    state = copy.deepcopy(base)
    agent = make_agent(config, seed, device)
    agent.load_training_state_dict(state["agent"])
    if replay_schedule == "risk_handover":
        risk = config["risk_replay"]
        replay = RiskCoverageHandoverReplay(
            state["demonstrations"],
            state["online_replay"]["capacity"],
            risk["guidance_fraction"],
            risk["capacity"],
            risk["coverage_target"],
            seed,
        )
    elif replay_schedule == "safe_intervention":
        replay = SafeInterventionReplay(
            state["demonstrations"],
            state["online_replay"]["capacity"],
            fraction,
            config.get("safe_intervention", {}).get("capacity", 3000),
            safe_sample_count,
            seed,
        )
    elif replay_schedule == "risk_sampling":
        replay = IndexedRiskReplay(
            state["demonstrations"],
            state["online_replay"]["capacity"],
            fraction,
            risk_sample_count,
            seed,
        )
    else:
        replay = PersistentDemoReplay(state["demonstrations"],
                                      state["online_replay"]["capacity"], fraction, seed)
    replay.online.load_state_dict(state["online_replay"])
    if isinstance(replay, IndexedRiskReplay):
        replay.initialize_online_index()
    replay._rng.setstate(state["demo_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("Use the foundation's device family for matched adaptation runs.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return agent, replay


def flatten_pairs(pairs):
    return [{**pair[condition], "pair_id": pair["pair_id"],
             "reference_collision_step": pair["first_reference_collision_step"],
             "scenario_id": pair["pair_id"] + "_" + condition}
            for pair in pairs for condition in ("control", "conflict")]


def conflict_probe(agent, problem, config, scene):
    """Common reference states isolate response from learned route selection.

    Prefix is executed in the real simulator to retain the correct history and
    obstacle clock. Oracle risk is logged ONLY; it never changes policy actions.
    """
    env = make_env(problem, config, scene)
    obs = env.reset()
    steps = scene["reference_collision_step"]
    for left, right in zip(problem.nominal_path[:steps], problem.nominal_path[1:steps + 1]):
        result = env.step(int(action_between(left, right)))
        if result.done:
            raise RuntimeError("Probe prefix must remain safe and nonterminal.")
        obs = result.observation
    valid = np.flatnonzero(env.action_mask(True)).tolist()
    action = agent.select_action(obs, 0.0, valid)
    blocked = [a for a in valid if env.dynamic_action_collision_risk(a)]
    safe = [a for a in valid if a not in blocked]
    with torch.no_grad():
        q = agent.policy_network(torch.as_tensor(obs.spatial, device=agent.device).unsqueeze(0),
                                 torch.as_tensor(obs.scalars, device=agent.device).unsqueeze(0))[0]
    margin = float(q[safe].max() - q[blocked].max()) if safe and blocked else None
    return {"probe_unsafe_action": int(action in blocked), "probe_wait_action": int(action == 4),
            "probe_safe_q_margin": margin}


def rollout(agent, problem, config, scene=None):
    env = make_env(problem, config, scene)
    obs = env.reset()
    positions, actions, dynamic = [list(env.position)], [], [list(env.dynamic_positions)]
    reward, waits = 0.0, 0
    for _ in range(config["max_episode_steps"]):
        action = agent.select_action(obs, 0.0, np.flatnonzero(env.action_mask(True)).tolist())
        result = env.step(action)
        actions.append(action)
        positions.append(list(env.position))
        dynamic.append(list(env.dynamic_positions))
        reward += result.reward
        waits += action == 4
        obs = result.observation
        if result.done:
            break
    return {"safe_success": int(result.info["reached"]),
            "dynamic_collision": int(result.info["collision_type"] == "dynamic"),
            "static_collision": int(result.info["collision_type"] == "static"),
            "timeout": int(result.truncated), "steps": env.steps, "wait_steps": waits,
            "return": reward, "positions": positions, "actions": actions, "dynamic_positions": dynamic}


def evaluate(agent, problem, config, scenes):
    rows, trajectories = [], []
    for scene in scenes:
        result = rollout(agent, problem, config, scene)
        trajectory = {key: result.pop(key) for key in ("positions", "actions", "dynamic_positions")}
        row = {"scenario_id": scene["scenario_id"], "pair_id": scene["pair_id"],
               "condition": scene["condition"],
               "obstacle_count": scene.get("obstacle_count", len(scene.get("obstacles", ())) or 1),
               "causal_obstacle_count": scene.get("causal_obstacle_count", 1),
               **result, **conflict_probe(agent, problem, config, scene),
               "successful_excess_steps": (result["steps"] - scene["oracle_steps"]
                                            if result["safe_success"] else None)}
        rows.append(row)
        trajectories.append({**row, **trajectory})
    summary = {}
    for condition in ("all", "control", "conflict"):
        selected = [row for row in rows if condition == "all" or row["condition"] == condition]
        if not selected:
            continue
        for key in ("safe_success", "dynamic_collision", "timeout", "wait_steps"):
            summary[f"{condition}_{key}"] = float(np.mean([row[key] for row in selected]))
        for key in ("probe_unsafe_action", "probe_wait_action", "probe_safe_q_margin"):
            values = [row[key] for row in selected if row[key] is not None]
            summary[f"{condition}_{key}"] = float(np.mean(values)) if values else None
    summary["conflict_minus_control_failure"] = (
        summary["control_safe_success"] - summary["conflict_safe_success"])
    static = rollout(agent, problem, config)
    summary["static_safe_success"] = static["safe_success"]
    summary["static_steps"] = static["steps"]
    return summary, rows, trajectories


def epsilon_at(stage, step):
    fraction = min(1.0, step / stage["epsilon_decay_steps"])
    return stage["epsilon_start"] + fraction * (stage["epsilon_end"] - stage["epsilon_start"])


def decay_demo_fraction(step: int, *, first_boundary: int = 50_000,
                        second_boundary: int = 100_000) -> float:
    """Simple predeclared schedule used by the adaptation ablation."""
    if step <= first_boundary:
        return 0.25
    if step <= second_boundary:
        return 0.10
    return 0.0


def safe_teacher_action(problem, env):
    """Return an exact current-phase safe action, with a one-step fallback."""

    scenario = DynamicScenario(seed=0, obstacles=env.dynamic_obstacles)
    plan = shortest_safe_plan(
        problem,
        scenario,
        start_position=tuple(env.position),
        start_time=env.steps,
    )
    if plan is not None and len(plan.positions) >= 2:
        return int(action_between(plan.positions[0], plan.positions[1])), False
    valid = np.flatnonzero(env.action_mask(True)).tolist()
    safe = [action for action in valid if not env.dynamic_action_collision_risk(action)]
    if not safe:
        return None, True
    action = min(
        safe,
        key=lambda candidate: (
            manhattan(move(env.position, candidate), problem.goal),
            candidate == int(Action.STAY),
            candidate,
        ),
    )
    return int(action), True


def train_steps(agent, replay, problem, config, stage, scenes, seed, on_evaluation,
                replay_schedule=None):
    """Exact step budget, one gradient update per interaction after warm-up.

    Scene order is identical by episode across branches; visits and episode
    lengths can differ as a consequence of treatment. The audit records exposure.
    All branches reset at the fork; evaluation uses separate environments.
    """
    rng = random.Random(seed + 123000)
    sequence, cursor = [], 0
    def next_scene():
        nonlocal sequence, cursor
        if not scenes:
            return None
        if cursor >= len(sequence):
            sequence = list(scenes)
            rng.shuffle(sequence)
            cursor = 0
        scene = sequence[cursor]
        cursor += 1
        return scene
    scene = next_scene()
    env = make_env(problem, config, scene)
    obs = env.reset()
    records, episode, episode_start, reward, demo_samples, online_samples = [], 0, 0, 0.0, 0, 0
    visible_steps, reference_risk_steps, sampled_visible = 0, 0, 0
    risk_samples = 0
    safe_samples = 0
    proposed_risk_steps = 0
    intervention_steps = 0
    teacher_fallback_steps = 0
    teacher_unavailable_steps = 0
    teacher_cache_hits = 0
    teacher_planner_calls = 0
    teacher_cache = {}
    risk_history = deque(maxlen=int(config.get("risk_replay", {}).get("history_steps", 3)) + 1)
    risk_token_history = deque(maxlen=int(config.get("risk_replay", {}).get("history_steps", 3)) + 1)
    reference_actions = {p: int(action_between(p, q)) for p, q in zip(problem.nominal_path, problem.nominal_path[1:])}
    start_time = perf_counter()
    for step in range(1, stage["max_steps"] + 1):
        if replay_schedule in {"decay", "safe_intervention", "risk_sampling"}:
            replay.set_demo_fraction(decay_demo_fraction(step))
        epsilon = epsilon_at(stage, step - 1)
        visible_steps += int(bool(obs.spatial[1:].any()))
        reference_risk = bool(
            scene
            and env.position in reference_actions
            and env.dynamic_action_collision_risk(reference_actions[env.position])
        )
        reference_risk_steps += int(reference_risk)
        proposed_action = agent.select_action(
            obs, epsilon, np.flatnonzero(env.action_mask(True)).tolist()
        )
        selected_action_risk = bool(scene and env.dynamic_action_collision_risk(proposed_action))
        action = proposed_action
        teacher_fallback = False
        if replay_schedule == "safe_intervention" and selected_action_risk:
            proposed_risk_steps += 1
            teacher_key = (
                scene["scenario_id"],
                tuple(env.position),
                int(env.steps),
            )
            if teacher_key in teacher_cache:
                teacher_action, teacher_fallback = teacher_cache[teacher_key]
                teacher_cache_hits += 1
            else:
                teacher_action, teacher_fallback = safe_teacher_action(problem, env)
                teacher_cache[teacher_key] = (teacher_action, teacher_fallback)
                teacher_planner_calls += 1
            if teacher_action is None:
                teacher_unavailable_steps += 1
            else:
                action = teacher_action
                intervention_steps += 1
                teacher_fallback_steps += int(teacher_fallback)
        decision_position = tuple(env.position)
        result = env.step(action)
        transition = Transition(obs, action, result.reward, result.observation,
                                result.terminated, env.action_mask(True))
        online_token = replay.add(transition)
        if replay_schedule == "safe_intervention" and action != proposed_action:
            if not isinstance(replay, SafeInterventionReplay):
                raise RuntimeError("safe_intervention requires SafeInterventionReplay.")
            obstacle_count = scene.get("obstacle_count", len(scene.get("obstacles", ())))
            replay.add_safe(
                transition,
                (int(obstacle_count), int(proposed_action), int(action)),
            )
        risk_history.append(transition)
        if isinstance(replay, IndexedRiskReplay):
            risk_token_history.append(online_token)
        if replay_schedule == "risk_handover" and (
            reference_risk
            or selected_action_risk
            or result.info["collision_type"] == "dynamic"
        ):
            if not isinstance(replay, RiskCoverageHandoverReplay):
                raise RuntimeError("risk_handover requires RiskCoverageHandoverReplay.")
            risk_key = (
                scene.get("obstacle_count", len(scene.get("obstacles", ()))),
                decision_position,
            )
            for item in risk_history:
                replay.add_risk(item, risk_key)
        if replay_schedule == "risk_sampling" and (
            reference_risk
            or selected_action_risk
            or result.info["collision_type"] == "dynamic"
        ):
            if not isinstance(replay, IndexedRiskReplay):
                raise RuntimeError("risk_sampling requires IndexedRiskReplay.")
            for token in risk_token_history:
                replay.mark_risk(token)
        obs = result.observation
        reward += result.reward
        if replay.can_sample(config["batch_size"]) and replay.online_size >= config["batch_size"]:
            batch = replay.sample(config["batch_size"])
            sampled_visible += sum(bool(item.state.spatial[1:].any()) for item in batch)
            agent.train_batch(batch)
            d, o = replay.sample_counts(config["batch_size"])
            demo_samples += d
            online_samples += o
            if isinstance(replay, RiskCoverageHandoverReplay):
                risk_samples += replay.last_risk_sample_count
            if isinstance(replay, IndexedRiskReplay):
                risk_samples += replay.last_risk_sample_count
            if isinstance(replay, SafeInterventionReplay):
                safe_samples += replay.last_safe_sample_count
        done = result.done or step == stage["max_steps"]
        if done:
            records.append({"episode": episode, "environment_steps": step,
                            "episode_steps": step - episode_start,
                            "scenario_id": scene["scenario_id"] if scene else "static",
                            "condition": scene["condition"] if scene else "static",
                            "success": int(result.info["reached"]),
                            "collision": int(result.info["collision"]),
                            "budget_cutoff": int(not result.done), "return": reward,
                            "epsilon": epsilon, "demo_samples_cumulative": demo_samples,
                            "online_samples_cumulative": online_samples,
                            "visible_dynamic_steps_cumulative": visible_steps,
                            "reference_risk_steps_cumulative": reference_risk_steps,
                            "sampled_visible_dynamic_cumulative": sampled_visible,
                            "risk_samples_cumulative": risk_samples,
                            "risk_buffer_size": getattr(replay, "risk_size", 0),
                            "risk_marked_total": getattr(replay, "risk_total_marked", 0),
                            "risk_coverage_count": getattr(replay, "covered_risk_count", 0),
                            "handover_progress": getattr(replay, "handover_progress", 0.0),
                            "proposed_risk_steps_cumulative": proposed_risk_steps,
                            "intervention_steps_cumulative": intervention_steps,
                            "teacher_fallback_steps_cumulative": teacher_fallback_steps,
                            "teacher_unavailable_steps_cumulative": teacher_unavailable_steps,
                            "teacher_cache_hits_cumulative": teacher_cache_hits,
                            "teacher_planner_calls_cumulative": teacher_planner_calls,
                            "safe_samples_cumulative": safe_samples,
                            "safe_buffer_size": getattr(replay, "safe_size", 0),
                            "safe_coverage_count": getattr(replay, "safe_coverage_count", 0)})
        if step % stage["evaluation_interval"] == 0 or step == stage["max_steps"]:
            if on_evaluation(step, records):
                return {"steps": step, "early_stop": True, "seconds": perf_counter() - start_time,
                        "demo_samples": demo_samples, "online_samples": online_samples,
                        "visible_dynamic_steps": visible_steps, "reference_risk_steps": reference_risk_steps,
                        "sampled_visible_dynamic": sampled_visible, "risk_samples": risk_samples,
                        "risk_buffer_size": getattr(replay, "risk_size", 0),
                        "risk_marked_total": getattr(replay, "risk_total_marked", 0),
                        "risk_coverage_count": getattr(replay, "covered_risk_count", 0),
                        "handover_progress": getattr(replay, "handover_progress", 0.0),
                        "proposed_risk_steps": proposed_risk_steps,
                        "intervention_steps": intervention_steps,
                        "teacher_fallback_steps": teacher_fallback_steps,
                        "teacher_unavailable_steps": teacher_unavailable_steps,
                        "teacher_cache_hits": teacher_cache_hits,
                        "teacher_planner_calls": teacher_planner_calls,
                        "safe_samples": safe_samples,
                        "safe_buffer_size": getattr(replay, "safe_size", 0),
                        "safe_coverage_count": getattr(replay, "safe_coverage_count", 0)}
        if done and step < stage["max_steps"]:
            episode += 1
            episode_start, reward = step, 0.0
            risk_history.clear()
            risk_token_history.clear()
            scene = next_scene()
            env = make_env(problem, config, scene)
            obs = env.reset()
    return {"steps": stage["max_steps"], "early_stop": False, "seconds": perf_counter() - start_time,
            "demo_samples": demo_samples, "online_samples": online_samples,
            "visible_dynamic_steps": visible_steps, "reference_risk_steps": reference_risk_steps,
            "sampled_visible_dynamic": sampled_visible, "risk_samples": risk_samples,
            "risk_buffer_size": getattr(replay, "risk_size", 0),
            "risk_marked_total": getattr(replay, "risk_total_marked", 0),
            "risk_coverage_count": getattr(replay, "covered_risk_count", 0),
            "handover_progress": getattr(replay, "handover_progress", 0.0),
            "proposed_risk_steps": proposed_risk_steps,
            "intervention_steps": intervention_steps,
            "teacher_fallback_steps": teacher_fallback_steps,
            "teacher_unavailable_steps": teacher_unavailable_steps,
            "teacher_cache_hits": teacher_cache_hits,
            "teacher_planner_calls": teacher_planner_calls,
            "safe_samples": safe_samples,
            "safe_buffer_size": getattr(replay, "safe_size", 0),
            "safe_coverage_count": getattr(replay, "safe_coverage_count", 0)}


def train_foundation(problem, config, seed, device, directory, provenance, *, smoke=False):
    seed_everything(seed)
    agent = make_agent(config, seed, device)
    demos = collect_demos(problem, config)
    replay = PersistentDemoReplay(demos, config["replay_capacity"] - len(demos),
                                  config["foundation"]["demo_fraction"], seed)
    stage = config["foundation"]
    rows, streak = [], 0
    def check(step, training):
        nonlocal streak
        result = rollout(agent, problem, config)
        passed = result["safe_success"] and result["steps"] <= problem.astar_steps * stage["maximum_path_ratio"]
        streak = streak + 1 if passed else 0
        qualified = step >= stage["minimum_steps"] and streak >= stage["consecutive_passes"]
        rows.append({"environment_steps": step, "safe_success": result["safe_success"],
                     "steps": result["steps"], "consecutive_passes": streak, "qualified": qualified})
        write_records_csv(rows, directory / "foundation_evaluation.csv")
        write_records_csv(training, directory / "training.csv")
        print(f"foundation seed={seed} steps={step} safe={result['safe_success']} "
              f"path={result['steps']} streak={streak}", flush=True)
        return qualified
    run = train_steps(agent, replay, problem, config, stage, [], seed, check)
    qualified = bool(rows[-1]["qualified"])
    # Smoke bypass is explicit and cannot be used by a formal invocation.
    state = snapshot(agent, replay)
    content_hash = state_digest(state)
    metadata = {**provenance, "seed": seed, "device": str(agent.device), "smoke": smoke,
                "qualified": qualified, "status": "qualified" if qualified else "foundation_unqualified",
                "snapshot_sha256": content_hash, "demo_count": len(demos),
                "online_capacity": replay.online.capacity, "run": run}
    torch.save({"metadata": metadata, "state": state}, directory / "foundation.pt")
    write_json(metadata, directory / "foundation_status.json")
    return metadata


def train_branch(problem, config, scenarios, seed, device, directory, checkpoint, fraction,
                 *, smoke=False, replay_schedule=None, safe_sample_count=0,
                 risk_sample_count=0):
    if not checkpoint["metadata"]["qualified"] and not smoke:
        raise ValueError("Foundation is not qualified; adaptation must not start.")
    if bool(checkpoint["metadata"]["smoke"]) != smoke:
        raise ValueError("Smoke and formal artifacts cannot be mixed.")
    initial_fraction = 0.25 if replay_schedule in {
        "decay", "risk_handover", "safe_intervention", "risk_sampling"
    } else fraction
    agent, replay = restore(
        config,
        checkpoint["state"],
        initial_fraction,
        seed,
        device,
        replay_schedule=replay_schedule,
        safe_sample_count=safe_sample_count,
        risk_sample_count=risk_sample_count,
    )
    digest = state_digest(snapshot(agent, replay))
    if digest != checkpoint["metadata"]["snapshot_sha256"]:
        raise RuntimeError("Fork did not restore identical model/optimizer/replay/RNG state.")
    directory.mkdir(parents=True, exist_ok=False)
    d, o = replay.sample_counts(config["batch_size"])
    write_json({"source_snapshot_sha256": digest, "verified_full_state_equal": True,
                "demo_fraction_requested": fraction, "replay_schedule": replay_schedule,
                 "demo_count_per_batch": d,
                 "online_count_per_batch": o, "effective_demo_fraction": d / (d + o),
                 "safe_samples_per_batch": safe_sample_count,
                 "risk_samples_per_batch": risk_sample_count,
                 "risk_storage_mode": (
                     "online_index_only" if replay_schedule == "risk_sampling" else None
                 ),
                "online_capacity": replay.online.capacity, "fork_environment": "fresh static start; dynamic clock reset",
                "epsilon_clock": "steps since fork", "smoke": smoke}, directory / "fork_audit.json")
    validation = flatten_pairs(scenarios["validation"])
    stage = config["adaptation"]
    curve, details, streak, first_threshold = [], [], 0, None
    initial_updates = agent.update_steps
    def check(step, training):
        nonlocal streak, first_threshold
        summary, rows, _ = evaluate(agent, problem, config, validation)
        passed = summary["conflict_safe_success"] >= stage["safe_success_threshold"]
        streak = streak + 1 if passed else 0
        if streak >= stage["consecutive_passes"] and first_threshold is None:
            first_threshold = step
        curve.append({"environment_steps": step, "gradient_updates_since_fork": agent.update_steps - initial_updates,
                      **summary, "threshold_consecutive_passes": streak})
        details.extend({"environment_steps": step, **row} for row in rows)
        write_records_csv(curve, directory / "validation_curve.csv")
        write_records_csv(details, directory / "validation_details.csv")
        write_records_csv(training, directory / "training.csv")
        label = replay_schedule or f"fraction={fraction:.2f}"
        print(f"{label} seed={seed} step={step} "
              f"conflict={summary['conflict_safe_success']:.3f} control={summary['control_safe_success']:.3f} "
              f"static={summary['static_safe_success']}", flush=True)
        return False
    check(0, [])
    run = train_steps(agent, replay, problem, config, stage, flatten_pairs(scenarios["train"]),
                       seed, check, replay_schedule=replay_schedule)
    # Fixed-budget final model only. Test never drives stopping or model selection.
    summary, test_rows, trajectories = evaluate(agent, problem, config, flatten_pairs(scenarios["test"]))
    write_records_csv(test_rows, directory / "test_evaluation.csv")
    write_json(trajectories, directory / "test_trajectories.json")
    agent.save_weights(directory / "model_final.pth")
    times = np.asarray([row["environment_steps"] for row in curve])
    values = np.asarray([row["conflict_safe_success"] for row in curve])
    auc = float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2) / stage["max_steps"])
    total_samples = run["demo_samples"] + run["online_samples"]
    effective_fraction = (run["demo_samples"] / total_samples
                          if total_samples else d / (d + o))
    write_json({**checkpoint["metadata"], "status": "complete", "fraction": fraction,
                 "replay_schedule": replay_schedule or "fixed",
                 "safe_sample_count": safe_sample_count,
                 "risk_sample_count": risk_sample_count,
                "effective_fraction": effective_fraction, "fork_sha256": digest,
                "adaptation_run": run, "gradient_updates_since_fork": agent.update_steps - initial_updates,
                "validation_conflict_auc": auc, "threshold_confirmation_step": first_threshold,
                "threshold_right_censored": first_threshold is None, "test": summary,
                "interpretation": "total replay-allocation effect; not isolated gradient interference"},
               directory / "result.json")
