"""Capability-adaptive handover with observation-only validation curves.

The foundation model, reward, epsilon schedule, D3QN architecture, TD update,
batch size, replay capacity, and training scenes match the frozen time-decay
experiment. Only the A* demonstration fraction during adaptation is changed.
Greedy validation rollouts are recorded every 10k steps and are never provided
to the competence controller or any training decision.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.maps.adaptation import spec_from_record
from astar_d3qn.replay.demo import PersistentDemoReplay
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.io import write_json, write_records_csv


def state_digest(value: Any) -> str:
    """Match the frozen experiment's full-state fingerprint."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
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


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_agent(config: Mapping[str, Any], seed: int, device: str) -> D3QNAgent:
    return D3QNAgent(
        D3QNConfig(
            spatial_shape=(4, config["window_size"], config["window_size"]),
            scalar_dim=2,
            action_dim=5,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            target_sync_interval=config["target_sync_interval"],
            hidden_dim=config["hidden_dim"],
            device=device,
            seed=seed,
        )
    )


def make_env(problem: Any, config: Mapping[str, Any], scene: Mapping[str, Any] | None = None):
    if scene is None:
        obstacle_records = []
    elif "obstacles" in scene:
        obstacle_records = scene["obstacles"]
    else:
        obstacle_records = [scene["obstacle"]]
    return DynamicGridNavigationEnv(
        problem,
        [spec_from_record(record) for record in obstacle_records],
        max_steps=config["max_episode_steps"],
        reward_config=RewardConfig(**config["reward"]),
        terminate_on_collision=True,
        window_size=config["window_size"],
    )


def restore_foundation(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    seed: int,
    device: str,
    demo_fraction: float,
) -> tuple[D3QNAgent, PersistentDemoReplay]:
    """Restore the exact foundation state, then change only replay allocation."""

    state = copy.deepcopy(checkpoint["state"])
    agent = make_agent(config, seed, device)
    agent.load_training_state_dict(state["agent"])
    replay = PersistentDemoReplay(
        state["demonstrations"],
        state["online_replay"]["capacity"],
        demo_fraction,
        seed,
    )
    replay.online.load_state_dict(state["online_replay"])
    replay._rng.setstate(state["demo_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("This CUDA foundation requires a CUDA-capable run device.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return agent, replay


def _scene_from_pair(pair: Mapping[str, Any], condition: str, source_split: str) -> dict[str, Any]:
    return {
        **copy.deepcopy(pair[condition]),
        "pair_id": pair["pair_id"],
        "reference_collision_step": pair["first_reference_collision_step"],
        "scenario_id": f"{pair['pair_id']}_{condition}",
        "source_split": source_split,
    }


def flatten_train_pairs(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Return the unchanged control+conflict adaptation stream from train only."""

    return [
        _scene_from_pair(pair, condition, "train")
        for pair in splits["train"]
        for condition in ("control", "conflict")
    ]


def select_monitor_scenes(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    monitor_scene_count: int,
) -> list[dict[str, Any]]:
    """Select a fixed, deterministic conflict-only monitor from the train split."""

    if monitor_scene_count <= 0:
        raise ValueError("monitor_scene_count must be positive.")
    train_pairs = sorted(splits["train"], key=lambda pair: pair["pair_id"])
    if monitor_scene_count > len(train_pairs):
        raise ValueError("monitor_scene_count exceeds the train pair count.")
    return [
        _scene_from_pair(pair, "conflict", "train")
        for pair in train_pairs[:monitor_scene_count]
    ]


def epsilon_at(stage: Mapping[str, Any], step: int) -> float:
    fraction = min(1.0, step / stage["epsilon_decay_steps"])
    return stage["epsilon_start"] + fraction * (
        stage["epsilon_end"] - stage["epsilon_start"]
    )


@dataclass
class CapabilityHandover:
    rho_max: float = 0.25
    beta: float = 0.3
    tau: float = 0.90
    consecutive_confirmations: int = 2
    competence_ema: float = 0.0
    demo_fraction: float | None = None
    threshold_streak: int = 0
    expert_exit_step: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho_max <= 1.0:
            raise ValueError("rho_max must be in [0, 1].")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be in (0, 1].")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1].")
        if self.consecutive_confirmations <= 0:
            raise ValueError("consecutive_confirmations must be positive.")
        if not 0.0 <= self.competence_ema <= 1.0:
            raise ValueError("competence_ema must be in [0, 1].")
        if self.demo_fraction is None:
            self.demo_fraction = self.rho_max
        if not 0.0 <= self.demo_fraction <= self.rho_max:
            raise ValueError("demo_fraction must be in [0, rho_max].")

    @property
    def permanently_exited(self) -> bool:
        return self.expert_exit_step is not None

    def observe(self, monitor_safe_success: float, environment_steps: int) -> None:
        if not 0.0 <= monitor_safe_success <= 1.0:
            raise ValueError("monitor_safe_success must be in [0, 1].")
        if environment_steps <= 0:
            raise ValueError("environment_steps must be positive.")
        self.competence_ema = (
            (1.0 - self.beta) * self.competence_ema
            + self.beta * monitor_safe_success
        )
        if self.permanently_exited:
            self.demo_fraction = 0.0
            return
        self.threshold_streak = (
            self.threshold_streak + 1 if self.competence_ema >= self.tau else 0
        )
        if self.threshold_streak >= self.consecutive_confirmations:
            self.demo_fraction = 0.0
            self.expert_exit_step = environment_steps
            return
        candidate = self.rho_max * (1.0 - self.competence_ema)
        self.demo_fraction = min(float(self.demo_fraction), candidate)


def rollout(agent: D3QNAgent, problem: Any, config: Mapping[str, Any], scene=None) -> dict[str, Any]:
    """Greedy epsilon=0 evaluation; it cannot affect the training schedule RNG."""

    env = make_env(problem, config, scene)
    observation = env.reset()
    total_return = 0.0
    for _ in range(config["max_episode_steps"]):
        action = agent.select_action(
            observation,
            0.0,
            np.flatnonzero(env.action_mask(True)).tolist(),
        )
        result = env.step(action)
        total_return += result.reward
        observation = result.observation
        if result.done:
            break
    return {
        "safe_success": int(result.info["reached"]),
        "dynamic_collision": int(result.info["collision_type"] == "dynamic"),
        "static_collision": int(result.info["collision_type"] == "static"),
        "timeout": int(result.truncated),
        "steps": env.steps,
        "return": total_return,
    }


def evaluate_scenes(agent, problem, config, scenes) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for scene in scenes:
        rows.append(
            {
                "scenario_id": scene["scenario_id"],
                "pair_id": scene["pair_id"],
                "condition": scene["condition"],
                "source_split": scene["source_split"],
                **rollout(agent, problem, config, scene),
            }
        )
    summary: dict[str, float] = {}
    for condition in ("all", "control", "conflict"):
        selected = [
            row for row in rows if condition == "all" or row["condition"] == condition
        ]
        if selected:
            for key in ("safe_success", "dynamic_collision", "timeout"):
                summary[f"{condition}_{key}"] = float(
                    np.mean([row[key] for row in selected])
                )
    return summary, rows


def _validate_frozen_conditions(config: Mapping[str, Any]) -> None:
    expected = {
        "window_size": 15,
        "max_episode_steps": 300,
        "replay_capacity": 10_000,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "target_sync_interval": 250,
        "hidden_dim": 256,
        "reward": {
            "step": -0.01,
            "progress": 0.05,
            "stay": 0.0,
            "collision": -1.0,
            "goal": 10.0,
        },
        "adaptation": {
            "max_steps": 200_000,
            "evaluation_interval": 10_000,
            "epsilon_start": 0.30,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 150_000,
            "safe_success_threshold": 0.90,
            "consecutive_passes": 2,
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen condition mismatch for {key}: {config.get(key)!r}")


def train_capability_adaptive_auc(
    problem: Any,
    config: Mapping[str, Any],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    device: str,
    output_dir: Path,
    checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    controller=None,
    monitor_observer=None,
) -> dict[str, Any]:
    """Run fixed-budget adaptation with validation as a read-only observer."""

    _validate_frozen_conditions(config)
    expected_recording = {
        "split": "validation",
        "evaluation_interval": 10_000,
        "epsilon": 0.0,
        "controls_training": False,
        "evaluate_step_zero": True,
    }
    if config.get("validation_recording") != expected_recording:
        raise ValueError(
            "validation_recording must be the frozen observation-only protocol."
        )
    if not checkpoint["metadata"].get("qualified"):
        raise ValueError("Only a qualified foundation may be adapted.")
    if checkpoint["metadata"].get("smoke"):
        raise ValueError("Smoke foundations are not valid for the formal run.")
    if checkpoint["metadata"].get("seed") != seed:
        raise ValueError("Foundation seed mismatch.")
    if checkpoint["metadata"].get("map_id") != problem.map_id:
        raise ValueError("Foundation map mismatch.")
    if checkpoint["metadata"].get("device") != device:
        raise ValueError(
            f"Foundation device is {checkpoint['metadata'].get('device')!r}; "
            f"run with --device {checkpoint['metadata'].get('device')}."
        )

    settings = config["capability_adaptive_handover"]
    controller = controller if controller is not None else CapabilityHandover(
        rho_max=float(settings["rho_max"]),
        beta=float(settings["beta"]),
        tau=float(settings["tau"]),
        consecutive_confirmations=int(settings["consecutive_confirmations"]),
        competence_ema=float(settings["initial_competence"]),
    )
    agent, replay = restore_foundation(
        config, checkpoint, seed, device, float(controller.demo_fraction)
    )
    restored_digest = state_digest(
        {
            "agent": agent.training_state_dict(),
            "online_replay": replay.online.state_dict(),
            "demonstrations": replay.demonstration_snapshot(),
            "demo_rng": replay._rng.getstate(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    )
    if restored_digest != checkpoint["metadata"]["snapshot_sha256"]:
        raise RuntimeError("Restored foundation is not bitwise state-identical.")

    monitor_scenes = select_monitor_scenes(
        splits, int(settings["monitor_scene_count"])
    )
    training_scenes = flatten_train_pairs(splits)
    validation_scenes = [
        _scene_from_pair(pair, condition, "validation")
        for pair in splits["validation"]
        for condition in ("control", "conflict")
    ]
    monitor_ids = [scene["scenario_id"] for scene in monitor_scenes]
    monitor_digest = hashlib.sha256(
        "\n".join(monitor_ids).encode("utf-8")
    ).hexdigest()
    stage = config["adaptation"]

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        {
            **provenance,
            "source_snapshot_sha256": restored_digest,
            "verified_full_state_equal": True,
            "method": "capability_adaptive_handover",
            "only_treatment_change": "adaptation A* demo replay fraction",
            "monitor_source_split": "train",
            "monitor_condition": "conflict",
            "monitor_scene_count": len(monitor_scenes),
            "monitor_scenario_ids": monitor_ids,
            "monitor_scenario_ids_sha256": monitor_digest,
            "monitor_scenes_remain_in_training_stream": True,
            "validation_controls_training": False,
            "validation_source_split": "validation",
            "validation_epsilon": 0.0,
            "validation_evaluation_interval": stage["evaluation_interval"],
            "validation_evaluates_step_zero": True,
            "test_read_or_generated": False,
        },
        output_dir / "run_audit.json",
    )

    rng = random.Random(seed + 123000)
    shuffled_scenes: list[dict[str, Any]] = []
    scene_cursor = 0

    def next_scene() -> dict[str, Any]:
        nonlocal shuffled_scenes, scene_cursor
        if scene_cursor >= len(shuffled_scenes):
            shuffled_scenes = list(training_scenes)
            rng.shuffle(shuffled_scenes)
            scene_cursor = 0
        selected = shuffled_scenes[scene_cursor]
        scene_cursor += 1
        return selected

    scene = next_scene()
    env = make_env(problem, config, scene)
    observation = env.reset()
    monitor_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    validation_curve_rows: list[dict[str, Any]] = []
    validation_final_rows: list[dict[str, Any]] = []
    validation_summary: dict[str, float] = {}
    episode = 0
    episode_start = 0
    episode_return = 0.0
    demo_samples_cumulative = 0
    online_samples_cumulative = 0
    demonstration_ids = {id(item) for item in replay.demonstration_snapshot()}
    prediction_target_channel = getattr(env, "current_dynamic_channel", None)
    if prediction_target_channel is None:
        prediction_target_channel = 1
    start_time = perf_counter()

    def record_validation(environment_steps: int) -> None:
        """Record greedy validation without changing controller or RNG state."""

        nonlocal validation_final_rows, validation_summary
        protected_before = state_digest(
            {
                "agent_update_steps": agent.update_steps,
                "agent_rng": agent._rng.getstate(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "controller": vars(controller),
                "replay_demo_fraction": replay.demo_fraction,
            }
        )
        summary, rows = evaluate_scenes(
            agent, problem, config, validation_scenes
        )
        protected_after = state_digest(
            {
                "agent_update_steps": agent.update_steps,
                "agent_rng": agent._rng.getstate(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "controller": vars(controller),
                "replay_demo_fraction": replay.demo_fraction,
            }
        )
        if protected_after != protected_before:
            raise RuntimeError("Validation evaluation changed protected training state.")
        validation_summary = summary
        validation_final_rows = rows
        validation_curve_rows.append(
            {
                "environment_steps": environment_steps,
                "conflict_safe_success": summary["conflict_safe_success"],
                "control_safe_success": summary["control_safe_success"],
                "all_safe_success": summary["all_safe_success"],
                "dynamic_collision": summary["all_dynamic_collision"],
                "timeout": summary["all_timeout"],
                "conflict_dynamic_collision": summary[
                    "conflict_dynamic_collision"
                ],
                "control_dynamic_collision": summary[
                    "control_dynamic_collision"
                ],
                "conflict_timeout": summary["conflict_timeout"],
                "control_timeout": summary["control_timeout"],
            }
        )
        write_records_csv(
            validation_curve_rows, output_dir / "validation_curve.csv"
        )

    record_validation(0)

    for step in range(1, stage["max_steps"] + 1):
        epsilon = epsilon_at(stage, step - 1)
        action = agent.select_action(
            observation,
            epsilon,
            np.flatnonzero(env.action_mask(True)).tolist(),
        )
        result = env.step(action)
        replay.add(
            Transition(
                observation,
                action,
                result.reward,
                result.observation,
                result.terminated,
                env.action_mask(True),
            )
        )
        observation = result.observation
        episode_return += result.reward

        if replay.can_sample(config["batch_size"]) and replay.online_size >= config["batch_size"]:
            batch = replay.sample(config["batch_size"])
            # These explicit inactive arguments match the frozen time_decay call.
            # No auxiliary loss, margin, prediction, or intervention is enabled.
            update = agent.train_batch(
                batch,
                conflict_margin=0.0,
                conflict_margin_loss_weight=0.0,
                prediction_loss_weight=0.0,
                prediction_pos_weight=20.0,
                prediction_target_channel=prediction_target_channel,
                prediction_spatial_weights=None,
                prediction_mask=[id(item) not in demonstration_ids for item in batch],
                prediction_gradient_strategy="none",
            )
            finite_keys = ("loss", "td_loss", "q_abs_max", "gradient_norm_before_clip")
            if not all(np.isfinite(update[key]) for key in finite_keys):
                raise RuntimeError(f"Non-finite training statistic: {update}")
            demo_count, online_count = replay.sample_counts(config["batch_size"])
            demo_samples_cumulative += demo_count
            online_samples_cumulative += online_count

        done = result.done or step == stage["max_steps"]
        if done:
            episode_rows.append(
                {
                    "episode": episode,
                    "environment_steps": step,
                    "episode_steps": step - episode_start,
                    "scenario_id": scene["scenario_id"],
                    "condition": scene["condition"],
                    "success": int(result.info["reached"]),
                    "collision": int(result.info["collision"]),
                    "budget_cutoff": int(not result.done),
                    "return": episode_return,
                    "epsilon": epsilon,
                    "demo_samples_cumulative": demo_samples_cumulative,
                    "online_samples_cumulative": online_samples_cumulative,
                }
            )

        if step % stage["evaluation_interval"] == 0:
            monitor_results = [
                rollout(agent, problem, config, monitor)
                for monitor in monitor_scenes
            ]
            safe_success = float(np.mean([row["safe_success"] for row in monitor_results]))
            extra_monitor_fields = {}
            if monitor_observer is None:
                controller.observe(safe_success, step)
            else:
                extra_monitor_fields = monitor_observer(
                    controller, monitor_results, monitor_scenes, problem, step
                )
            replay.set_demo_fraction(float(controller.demo_fraction))
            demo_count, online_count = replay.sample_counts(config["batch_size"])
            monitor_rows.append(
                {
                    "environment_steps": step,
                    "monitor_safe_success": safe_success,
                    "competence_ema": controller.competence_ema,
                    "demo_fraction": controller.demo_fraction,
                    "demo_sample_count": demo_count,
                    "online_sample_count": online_count,
                    "expert_exit_step": controller.expert_exit_step,
                    "threshold_streak": controller.threshold_streak,
                    "demo_samples_cumulative": demo_samples_cumulative,
                    "online_samples_cumulative": online_samples_cumulative,
                    **extra_monitor_fields,
                }
            )
            write_records_csv(monitor_rows, output_dir / "training.csv")
            write_records_csv(episode_rows, output_dir / "episodes.csv")
            record_validation(step)
            print(
                f"map={problem.map_id} seed={seed} step={step} "
                f"monitor={safe_success:.3f} competence={controller.competence_ema:.3f} "
                f"rho={controller.demo_fraction:.6f}",
                flush=True,
            )

        if done and step < stage["max_steps"]:
            episode += 1
            episode_start = step
            episode_return = 0.0
            scene = next_scene()
            env = make_env(problem, config, scene)
            observation = env.reset()

    agent.save_weights(output_dir / "model_final.pth")
    write_records_csv(validation_final_rows, output_dir / "validation_final.csv")
    times = np.asarray(
        [row["environment_steps"] for row in validation_curve_rows], dtype=float
    )
    conflict_values = np.asarray(
        [row["conflict_safe_success"] for row in validation_curve_rows],
        dtype=float,
    )
    validation_conflict_auc = float(
        np.sum(
            np.diff(times)
            * (conflict_values[:-1] + conflict_values[1:])
            / 2.0
        )
        / stage["max_steps"]
    )
    total_samples = demo_samples_cumulative + online_samples_cumulative
    result_record = {
        **checkpoint["metadata"],
        **provenance,
        "status": "validation_complete",
        "method": "capability_adaptive_handover",
        "adaptation_steps": stage["max_steps"],
        "test_deferred": True,
        "test_read_or_generated": False,
        "monitor_source_split": "train",
        "monitor_scene_count": len(monitor_scenes),
        "monitor_scenario_ids_sha256": monitor_digest,
        "rho_max": controller.rho_max,
        "beta": controller.beta,
        "tau": controller.tau,
        "consecutive_confirmations": controller.consecutive_confirmations,
        "final_competence_ema": controller.competence_ema,
        "final_demo_fraction": controller.demo_fraction,
        "expert_exit_step": controller.expert_exit_step,
        "demo_samples": demo_samples_cumulative,
        "online_samples": online_samples_cumulative,
        "effective_demo_fraction": (
            demo_samples_cumulative / total_samples if total_samples else 0.0
        ),
        "seconds": perf_counter() - start_time,
        "validation": validation_summary,
        "validation_curve_rows": len(validation_curve_rows),
        "validation_conflict_auc": validation_conflict_auc,
        "validation_controls_training": False,
    }
    write_json(result_record, output_dir / "result.json")
    return result_record
