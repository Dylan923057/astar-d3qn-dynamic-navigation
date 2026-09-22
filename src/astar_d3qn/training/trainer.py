from __future__ import annotations

import random
from time import perf_counter
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Callable, Literal, Protocol, Sequence

import numpy as np

from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.core.collisions import CollisionTracker
from astar_d3qn.core.grid import Action
from astar_d3qn.envs.static_grid import RewardConfig, StaticGridNavigationEnv
from astar_d3qn.core.trajectory import TrajectoryRecorder
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.demo import (
    DemoKey,
    LocalConflictDemoReplay,
    LocalCounterexampleDemoReplay,
    PersistentDemoReplay,
)
from astar_d3qn.replay.prioritized import PrioritizedReplayBuffer
from astar_d3qn.replay.transition import Transition
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.training.budget import crossed_interval

ReplayStrategy = Literal[
    "uniform",
    "prefill",
    "persistent_demo",
    "conflict_adaptive_demo",
    "local_conflict_demo",
    "local_counterexample_demo",
    "predictive_margin_demo",
    "safe_guided_demo",
    "per",
    "dqfd",
]
ProgressCallback = Callable[[int, Sequence[dict], bool], bool | None]


class Replay(Protocol):
    def add(self, transition: Transition) -> None: ...

    def can_sample(self, batch_size: int) -> bool: ...

    def sample(self, batch_size: int) -> list[Transition]: ...

    def snapshot(self) -> tuple[Transition, ...]: ...


class NavigationEnvironment(Protocol):
    position: tuple[int, int]
    steps: int

    @property
    def observation_shape(self) -> tuple[int, int, int]: ...

    @property
    def scalar_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    def reset(self): ...

    def step(self, action: int): ...

    def action_mask(self, mask_collisions: bool = False): ...


EnvironmentFactory = Callable[..., NavigationEnvironment]

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    episodes: int = 1500
    max_environment_steps: int | None = None
    max_steps: int = 350
    replay_capacity: int = 10000
    batch_size: int = 64
    learning_starts: int = 500
    updates_per_step: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1200
    epsilon_decay_environment_steps: int | None = None
    terminate_on_collision: bool = False
    # Eliminate actions that are known to hit a static wall or leave the map.
    # Dynamic obstacles are deliberately not masked: avoiding them remains a
    # learned part of the task.
    mask_static_invalid_actions: bool = False
    replay_strategy: ReplayStrategy = "persistent_demo"
    demo_fraction: float = 0.25
    demo_fraction_final: float | None = None
    demo_fraction_decay_start_episode: int = 0
    demo_fraction_decay_end_episode: int = 0
    demo_fraction_decay_start_environment_step: int | None = None
    demo_fraction_decay_end_environment_step: int | None = None
    window_size: int = 11
    progress_interval: int = 100
    progress_interval_environment_steps: int | None = None
    diagnostic_interval: int = 10
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_priority_epsilon: float = 1e-6
    demo_margin: float = 0.8
    demo_loss_weight: float = 1.0
    conflict_demo_fraction_min: float = 0.0
    conflict_ema_alpha: float = 0.2
    conflict_recovery_alpha: float | None = None
    conflict_sensitivity: float = 1.0
    conflict_event_binary: bool = False
    conflict_predict_next: bool = True
    local_conflict_suppression_steps: int = 2500
    local_conflict_min_sampling_weight: float = 0.05
    local_conflict_predict_next: bool = True
    local_counterexample_capacity: int = 2000
    local_counterexample_fraction: float = 0.25
    conflict_margin: float = 0.8
    conflict_margin_loss_weight: float = 1.0
    conflict_margin_predict_next: bool = True
    safe_guidance_margin: float = 0.8
    safe_guidance_loss_weight: float = 1.0
    safe_guidance_predict_next: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if min(
            self.episodes,
            self.max_steps,
            self.replay_capacity,
            self.batch_size,
            self.updates_per_step,
            self.epsilon_decay_episodes,
        ) <= 0:
            raise ValueError("Positive training settings must be greater than zero.")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative.")
        if self.max_environment_steps is not None:
            if self.max_environment_steps <= 0:
                raise ValueError("max_environment_steps must be positive.")
            if self.epsilon_decay_environment_steps is None:
                raise ValueError(
                    "A fixed environment-step budget requires "
                    "epsilon_decay_environment_steps."
                )
        if self.epsilon_decay_environment_steps is not None:
            if self.epsilon_decay_environment_steps <= 0:
                raise ValueError(
                    "epsilon_decay_environment_steps must be positive."
                )
            if (
                self.max_environment_steps is not None
                and self.epsilon_decay_environment_steps
                > self.max_environment_steps
            ):
                raise ValueError(
                    "epsilon_decay_environment_steps cannot exceed the "
                    "environment-step budget."
                )
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")
        if self.progress_interval < 0:
            raise ValueError("progress_interval cannot be negative.")
        if (
            self.progress_interval_environment_steps is not None
            and self.progress_interval_environment_steps <= 0
        ):
            raise ValueError(
                "progress_interval_environment_steps must be positive."
            )
        if self.diagnostic_interval < 0:
            raise ValueError("diagnostic_interval cannot be negative.")
        if self.per_alpha < 0.0:
            raise ValueError("per_alpha cannot be negative.")
        if not 0.0 <= self.per_beta_start <= self.per_beta_end <= 1.0:
            raise ValueError("PER beta range must satisfy 0 <= start <= end <= 1.")
        if self.per_priority_epsilon <= 0.0:
            raise ValueError("per_priority_epsilon must be positive.")
        if self.demo_margin < 0.0 or self.demo_loss_weight < 0.0:
            raise ValueError("DQfD margin settings cannot be negative.")
        if not 0.0 <= self.demo_fraction <= 1.0:
            raise ValueError("demo_fraction must be in [0, 1].")
        if not 0.0 <= self.conflict_demo_fraction_min <= self.demo_fraction:
            raise ValueError(
                "conflict_demo_fraction_min must be between zero and demo_fraction."
            )
        if not 0.0 < self.conflict_ema_alpha <= 1.0:
            raise ValueError("conflict_ema_alpha must be in (0, 1].")
        if (
            self.conflict_recovery_alpha is not None
            and not 0.0 < self.conflict_recovery_alpha <= 1.0
        ):
            raise ValueError("conflict_recovery_alpha must be in (0, 1].")
        if self.conflict_sensitivity <= 0.0:
            raise ValueError("conflict_sensitivity must be positive.")
        if self.local_conflict_suppression_steps <= 0:
            raise ValueError("local_conflict_suppression_steps must be positive.")
        if not 0.0 <= self.local_conflict_min_sampling_weight <= 1.0:
            raise ValueError(
                "local_conflict_min_sampling_weight must be in [0, 1]."
            )
        if self.local_counterexample_capacity <= 0:
            raise ValueError("local_counterexample_capacity must be positive.")
        if not 0.0 <= self.local_counterexample_fraction <= 1.0:
            raise ValueError("local_counterexample_fraction must be in [0, 1].")
        if self.conflict_margin < 0.0 or self.conflict_margin_loss_weight < 0.0:
            raise ValueError("Conflict-margin loss settings cannot be negative.")
        if self.safe_guidance_margin < 0.0 or self.safe_guidance_loss_weight < 0.0:
            raise ValueError("Safe-guidance loss settings cannot be negative.")
        if self.demo_fraction_final is not None:
            if not 0.0 <= self.demo_fraction_final <= self.demo_fraction:
                raise ValueError(
                    "demo_fraction_final must be between zero and demo_fraction."
                )
            uses_step_schedule = (
                self.demo_fraction_decay_start_environment_step is not None
                or self.demo_fraction_decay_end_environment_step is not None
            )
            uses_episode_schedule = (
                self.demo_fraction_decay_start_episode != 0
                or self.demo_fraction_decay_end_episode != 0
            )
            if uses_step_schedule and uses_episode_schedule:
                raise ValueError(
                    "Demo-fraction decay must use either episodes or environment "
                    "steps, not both."
                )
            if uses_step_schedule:
                start = self.demo_fraction_decay_start_environment_step
                end = self.demo_fraction_decay_end_environment_step
                if (
                    start is None
                    or end is None
                    or self.max_environment_steps is None
                    or not 0 <= start < end <= self.max_environment_steps
                ):
                    raise ValueError(
                        "Environment-step demo decay requires 0 <= start < end "
                        "<= max_environment_steps."
                    )
            elif not (
                0 <= self.demo_fraction_decay_start_episode
                < self.demo_fraction_decay_end_episode
                <= self.episodes
            ):
                raise ValueError(
                    "Demo-fraction decay requires 0 <= start < end <= episodes."
                )
        elif (
            self.demo_fraction_decay_start_environment_step is not None
            or self.demo_fraction_decay_end_environment_step is not None
        ):
            raise ValueError(
                "Environment-step demo decay requires demo_fraction_final."
            )
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("Invalid epsilon range.")
        if self.replay_strategy not in {
            "uniform",
            "prefill",
            "persistent_demo",
            "conflict_adaptive_demo",
            "local_conflict_demo",
            "local_counterexample_demo",
            "predictive_margin_demo",
            "safe_guided_demo",
            "per",
            "dqfd",
        }:
            raise ValueError(f"Unknown replay strategy: {self.replay_strategy}")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    episode_records: tuple[dict, ...]
    environment_steps: int
    gradient_updates: int
    training_seconds: float
    progress_callback_seconds: float
    stopped_early: bool


def linear_epsilon(episode: int, config: TrainingConfig) -> float:
    fraction = min(1.0, max(0, episode) / config.epsilon_decay_episodes)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def linear_epsilon_at_environment_step(
    environment_step: int,
    config: TrainingConfig,
) -> float:
    """Return epsilon on the environment-step clock.

    This clock is used only by fixed-transition-budget experiments.  Keeping it
    separate from ``linear_epsilon`` preserves the exact schedule of historical
    episode-budget configurations.
    """

    if config.epsilon_decay_environment_steps is None:
        raise ValueError("epsilon_decay_environment_steps is not configured.")
    fraction = min(
        1.0,
        max(0, environment_step) / config.epsilon_decay_environment_steps,
    )
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def demo_fraction_at_episode(episode: int, config: TrainingConfig) -> float:
    """Return the persistent-demo fraction scheduled for a one-based episode."""

    if config.demo_fraction_final is None:
        return config.demo_fraction
    if episode <= config.demo_fraction_decay_start_episode:
        return config.demo_fraction
    if episode >= config.demo_fraction_decay_end_episode:
        return config.demo_fraction_final
    progress = (
        episode - config.demo_fraction_decay_start_episode
    ) / (
        config.demo_fraction_decay_end_episode
        - config.demo_fraction_decay_start_episode
    )
    return config.demo_fraction + progress * (
        config.demo_fraction_final - config.demo_fraction
    )


def demo_fraction_at_environment_step(
    environment_step: int,
    config: TrainingConfig,
) -> float:
    """Return the persistent-demo fraction on an environment-step schedule."""

    if config.demo_fraction_final is None:
        return config.demo_fraction
    start = config.demo_fraction_decay_start_environment_step
    end = config.demo_fraction_decay_end_environment_step
    if start is None or end is None:
        return config.demo_fraction
    if environment_step <= start:
        return config.demo_fraction
    if environment_step >= end:
        return config.demo_fraction_final
    progress = (environment_step - start) / (end - start)
    return config.demo_fraction + progress * (
        config.demo_fraction_final - config.demo_fraction
    )


class ConflictAdaptiveDemoController:
    """Convert observed online A*-action conflicts into a demo fraction."""

    def __init__(
        self,
        maximum: float,
        minimum: float,
        ema_alpha: float,
        sensitivity: float,
        recovery_alpha: float | None = None,
        event_binary: bool = False,
    ) -> None:
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError("Adaptive demo fractions must satisfy 0 <= min <= max <= 1.")
        if (
            not 0.0 < ema_alpha <= 1.0
            or (
                recovery_alpha is not None
                and not 0.0 < recovery_alpha <= 1.0
            )
            or sensitivity <= 0.0
        ):
            raise ValueError("Invalid conflict controller parameters.")
        self.maximum = float(maximum)
        self.minimum = float(minimum)
        self.ema_alpha = float(ema_alpha)
        self.recovery_alpha = float(
            ema_alpha if recovery_alpha is None else recovery_alpha
        )
        self.sensitivity = float(sensitivity)
        self.event_binary = bool(event_binary)
        self.conflict_ema = 0.0

    @property
    def demo_fraction(self) -> float:
        safe_weight = (1.0 - self.conflict_ema) ** self.sensitivity
        return self.minimum + (self.maximum - self.minimum) * safe_weight

    def observe(self, conflict_rate: float) -> float:
        if not 0.0 <= conflict_rate <= 1.0:
            raise ValueError("conflict_rate must be in [0, 1].")
        target = float(conflict_rate > 0.0) if self.event_binary else float(conflict_rate)
        alpha = self.ema_alpha if target > self.conflict_ema else self.recovery_alpha
        self.conflict_ema += alpha * (target - self.conflict_ema)
        return self.demo_fraction


def _demonstration_actions_by_position(
    problem: NavigationProblem,
    demonstrations: Sequence[Transition],
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Recover the demonstrated action distribution from goal-offset scalars."""

    actions: dict[tuple[int, int], list[int]] = defaultdict(list)
    scale = max(1, problem.size - 1)
    for transition in demonstrations:
        if transition.state.scalars.shape != (2,):
            raise ValueError("Conflict-adaptive replay requires two goal-offset scalars.")
        row = problem.goal[0] - round(float(transition.state.scalars[0]) * scale)
        column = problem.goal[1] - round(float(transition.state.scalars[1]) * scale)
        position = (int(row), int(column))
        if not (0 <= position[0] < problem.size and 0 <= position[1] < problem.size):
            raise ValueError("Could not recover a valid demonstration position.")
        actions[position].append(int(transition.action))
    return {position: tuple(values) for position, values in actions.items()}


def _demonstration_keys(
    problem: NavigationProblem,
    demonstrations: Sequence[Transition],
) -> tuple[DemoKey, ...]:
    """Recover a position-action key for each static A* transition."""

    scale = max(1, problem.size - 1)
    keys = []
    for transition in demonstrations:
        if transition.state.scalars.shape != (2,):
            raise ValueError("Local-conflict replay requires two goal-offset scalars.")
        row = problem.goal[0] - round(float(transition.state.scalars[0]) * scale)
        column = problem.goal[1] - round(float(transition.state.scalars[1]) * scale)
        position = (int(row), int(column))
        if not (0 <= position[0] < problem.size and 0 <= position[1] < problem.size):
            raise ValueError("Could not recover a valid demonstration position.")
        keys.append((position, int(transition.action)))
    return tuple(keys)


def _online_demo_action_conflict_rate(
    env: NavigationEnvironment,
    actions_by_position: dict[tuple[int, int], tuple[int, ...]],
    *,
    predict_next: bool,
) -> float | None:
    actions = actions_by_position.get(tuple(env.position))
    if not actions:
        return None
    risk = getattr(env, "dynamic_action_collision_risk", None)
    if not callable(risk):
        raise ValueError(
            "conflict_adaptive_demo requires an environment with dynamic-action risk."
        )
    return sum(bool(risk(action, predict_next=predict_next)) for action in actions) / len(actions)


def _online_demo_action_risks(
    env: NavigationEnvironment,
    actions_by_position: dict[tuple[int, int], tuple[int, ...]],
    *,
    predict_next: bool,
) -> tuple[dict[int, bool], float | None]:
    """Return unique action risks and the demonstration-frequency-weighted rate."""

    actions = actions_by_position.get(tuple(env.position))
    if not actions:
        return {}, None
    risk = getattr(env, "dynamic_action_collision_risk", None)
    if not callable(risk):
        raise ValueError(
            "local_conflict_demo requires an environment with dynamic-action risk."
        )
    action_risks = {
        int(action): bool(risk(action, predict_next=predict_next))
        for action in set(actions)
    }
    conflict_rate = sum(action_risks[int(action)] for action in actions) / len(actions)
    return action_risks, conflict_rate


def _predictive_conflict_margin_masks(
    env: NavigationEnvironment,
    actions_by_position: dict[tuple[int, int], tuple[int, ...]],
    *,
    predict_next: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, bool, bool, bool, float]:
    """Label safe alternatives only when a demonstrated action is blocked.

    The returned masks let the network choose the highest-valued safe action;
    the trainer never invents a single pseudo-expert action.  The tuple is
    ``(safe_mask, blocked_mask, covered, conflict, no_safe, risky_fraction)``.
    """

    actions = actions_by_position.get(tuple(env.position))
    if not actions:
        return None, None, False, False, False, 0.0
    risk = getattr(env, "dynamic_action_collision_risk", None)
    if not callable(risk):
        raise ValueError(
            "predictive_margin_demo requires an environment with dynamic-action risk."
        )
    demonstrated_risks = [
        bool(risk(action, predict_next=predict_next)) for action in actions
    ]
    risky_fraction = sum(demonstrated_risks) / len(demonstrated_risks)
    blocked_actions = {
        int(action)
        for action, is_risky in zip(actions, demonstrated_risks)
        if is_risky
    }
    if not blocked_actions:
        return None, None, True, False, False, risky_fraction

    static_safe = np.asarray(env.action_mask(mask_collisions=True), dtype=bool)
    safe_mask = static_safe.copy()
    for action in range(len(safe_mask)):
        if bool(risk(action, predict_next=predict_next)):
            safe_mask[action] = False
    blocked_mask = np.zeros(len(safe_mask), dtype=bool)
    for action in blocked_actions:
        if not 0 <= action < len(blocked_mask):
            raise ValueError("A demonstrated action exceeds the environment action space.")
        blocked_mask[action] = True
    safe_mask[blocked_mask] = False
    if not safe_mask.any():
        return None, None, True, True, True, risky_fraction
    return safe_mask, blocked_mask, True, True, False, risky_fraction


def _safe_demo_guidance(
    env: NavigationEnvironment,
    actions_by_position: dict[tuple[int, int], tuple[int, ...]],
    *,
    predict_next: bool,
) -> tuple[int | None, bool, bool, float]:
    """Choose the most frequent currently safe demonstrated action.

    Returns ``(action, covered, all_blocked, risky_fraction)``.  No avoidance
    action is invented when every demonstrated action is unsafe; TD learning is
    left to discover waiting or detours from real online experience.
    """

    actions = actions_by_position.get(tuple(env.position))
    if not actions:
        return None, False, False, 0.0
    risk = getattr(env, "dynamic_action_collision_risk", None)
    if not callable(risk):
        raise ValueError(
            "safe_guided_demo requires an environment with dynamic-action risk."
        )
    risky = [bool(risk(action, predict_next=predict_next)) for action in actions]
    safe_actions = [action for action, is_risky in zip(actions, risky) if not is_risky]
    if not safe_actions:
        return None, True, True, 1.0
    counts = Counter(safe_actions)
    selected = min(counts, key=lambda action: (-counts[action], action))
    return selected, True, False, sum(risky) / len(risky)


def build_replay(
    config: TrainingConfig,
    demonstrations: Sequence[Transition],
    demonstration_keys: Sequence[DemoKey] = (),
) -> Replay:
    if config.replay_strategy == "uniform":
        return UniformReplayBuffer(config.replay_capacity, seed=config.seed)
    if config.replay_strategy == "per":
        return PrioritizedReplayBuffer(
            config.replay_capacity,
            alpha=config.per_alpha,
            beta=config.per_beta_start,
            priority_epsilon=config.per_priority_epsilon,
            seed=config.seed,
        )
    if not demonstrations:
        raise ValueError(
            f"Replay strategy {config.replay_strategy!r} requires demonstrations."
        )
    if config.replay_strategy in {"prefill", "safe_guided_demo", "dqfd"}:
        if config.replay_strategy == "dqfd":
            online_capacity = config.replay_capacity - len(demonstrations)
            if online_capacity <= 0:
                raise ValueError(
                    "Replay capacity must exceed demonstration count for dqfd."
                )
            return PersistentDemoReplay(
                demonstrations,
                online_capacity=online_capacity,
                demo_fraction=config.demo_fraction,
                seed=config.seed,
            )
        if len(demonstrations) > config.replay_capacity:
            raise ValueError(
                "One-time prefill demonstrations exceed replay capacity; "
                "increase replay_capacity or reduce demonstrations."
            )
        replay = UniformReplayBuffer(config.replay_capacity, seed=config.seed)
        replay.extend(demonstrations)
        return replay
    
    online_capacity = config.replay_capacity - len(demonstrations)
    if online_capacity <= 0:
        raise ValueError(
            "Replay capacity must exceed demonstration count for persistent demo strategies."
        )
    if config.replay_strategy in {
        "local_conflict_demo",
        "local_counterexample_demo",
    }:
        if len(demonstration_keys) != len(demonstrations):
            raise ValueError(
                f"{config.replay_strategy} requires one position-action key per "
                "demonstration."
            )
        if config.replay_strategy == "local_counterexample_demo":
            return LocalCounterexampleDemoReplay(
                demonstrations,
                demonstration_keys,
                online_capacity=online_capacity,
                demo_fraction=config.demo_fraction,
                suppression_steps=config.local_conflict_suppression_steps,
                minimum_sampling_weight=config.local_conflict_min_sampling_weight,
                counterexample_capacity=config.local_counterexample_capacity,
                counterexample_fraction=config.local_counterexample_fraction,
                seed=config.seed,
            )
        return LocalConflictDemoReplay(
            demonstrations,
            demonstration_keys,
            online_capacity=online_capacity,
            demo_fraction=config.demo_fraction,
            suppression_steps=config.local_conflict_suppression_steps,
            minimum_sampling_weight=config.local_conflict_min_sampling_weight,
            seed=config.seed,
        )
    return PersistentDemoReplay(
        demonstrations,
        online_capacity=online_capacity,
        demo_fraction=config.demo_fraction,
        seed=config.seed,
    )


def train_d3qn(
    problems: Sequence[NavigationProblem],
    agent: D3QNAgent,
    config: TrainingConfig,
    demonstrations: Sequence[Transition] = (),
    reward_config: RewardConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> TrainingResult:
    if not problems:
        raise ValueError("At least one training problem is required.")
    factory = environment_factory or StaticGridNavigationEnv

    def create_environment(problem: NavigationProblem) -> NavigationEnvironment:
        return factory(
            problem,
            max_steps=config.max_steps,
            reward_config=reward_config,
            terminate_on_collision=config.terminate_on_collision,
            window_size=config.window_size,
        )

    probe = create_environment(problems[0])
    expected_shape = probe.observation_shape
    expected_scalar_dim = probe.scalar_dim
    if (
        agent.config.spatial_shape != expected_shape
        or agent.config.scalar_dim != expected_scalar_dim
    ):
        raise ValueError(
            f"Agent observation spec {(agent.config.spatial_shape, agent.config.scalar_dim)} "
            f"does not match environment spec {(expected_shape, expected_scalar_dim)}."
        )
    if agent.action_dim != probe.action_dim:
        raise ValueError("Agent action count does not match the environment.")
    # Scheduled dynamic factories use the probe only for shape validation. Reset
    # their cursor so the first training episode consumes the first train scenario.
    reset_schedule = getattr(factory, "reset_schedule", None)
    if callable(reset_schedule):
        reset_schedule()

    local_conflict_keys = (
        _demonstration_keys(problems[0], demonstrations)
        if config.replay_strategy
        in {"local_conflict_demo", "local_counterexample_demo"}
        else ()
    )
    replay = build_replay(config, demonstrations, local_conflict_keys)
    demonstration_ids = {id(item) for item in demonstrations}
    adaptive_controller = (
        ConflictAdaptiveDemoController(
            maximum=config.demo_fraction,
            minimum=config.conflict_demo_fraction_min,
            ema_alpha=config.conflict_ema_alpha,
            sensitivity=config.conflict_sensitivity,
            recovery_alpha=config.conflict_recovery_alpha,
            event_binary=config.conflict_event_binary,
        )
        if config.replay_strategy == "conflict_adaptive_demo"
        else None
    )
    uses_position_guidance = config.replay_strategy in {
        "conflict_adaptive_demo",
        "local_conflict_demo",
        "local_counterexample_demo",
        "predictive_margin_demo",
        "safe_guided_demo",
    }
    demo_actions_by_position = (
        _demonstration_actions_by_position(problems[0], demonstrations)
        if uses_position_guidance
        else {}
    )
    if uses_position_guidance and len(problems) != 1:
        raise ValueError(
            f"{config.replay_strategy} currently requires exactly one training map."
        )
    rng = random.Random(config.seed)
    order = list(range(len(problems)))
    rng.shuffle(order)
    order_position = 0
    environment_steps = 0
    gradient_updates = 0
    records: list[dict] = []
    started_at = perf_counter()
    progress_callback_seconds = 0.0
    stopped_early = False

    episode = 0
    while (
        environment_steps < config.max_environment_steps
        if config.max_environment_steps is not None
        else episode < config.episodes
    ):
        episode += 1
        environment_steps_before_episode = environment_steps
        current_demo_fraction = (
            adaptive_controller.demo_fraction
            if adaptive_controller is not None
            else (
                demo_fraction_at_environment_step(environment_steps, config)
                if config.demo_fraction_decay_start_environment_step is not None
                else demo_fraction_at_episode(episode, config)
            )
        )
        set_demo_fraction = getattr(replay, "set_demo_fraction", None)
        if callable(set_demo_fraction):
            set_demo_fraction(current_demo_fraction)
        if order_position >= len(order):
            rng.shuffle(order)
            order_position = 0
        problem = problems[order[order_position]]
        order_position += 1
        set_environment_steps = getattr(factory, "set_environment_steps", None)
        if callable(set_environment_steps):
            set_environment_steps(environment_steps)
        env = create_environment(problem)
        state = env.reset()
        episode_return = 0.0
        losses: list[float] = []
        final_info: dict = {}
        collisions = CollisionTracker()
        trajectory = TrajectoryRecorder(problem.start, problem.goal)
        epsilon = (
            linear_epsilon_at_environment_step(environment_steps, config)
            if config.max_environment_steps is not None
            else linear_epsilon(episode - 1, config)
        )
        episode_demo_fractions: list[float] = []
        episode_demo_conflicts: list[float] = []
        guidance_candidate_count = 0
        guidance_applied_count = 0
        guidance_blocked_count = 0
        guidance_risky_fractions: list[float] = []
        guidance_losses: list[float] = []
        guidance_batch_counts: list[int] = []
        local_conflict_event_count = 0
        local_conflict_covered_count = 0
        local_counterexample_candidate_count = 0
        local_counterexample_added_count = 0
        local_counterexample_wait_count = 0
        conflict_margin_covered_count = 0
        conflict_margin_conflict_count = 0
        conflict_margin_labeled_count = 0
        conflict_margin_no_safe_count = 0
        conflict_margin_risky_fractions: list[float] = []
        conflict_margin_losses: list[float] = []
        conflict_margin_batch_counts: list[int] = []

        while True:
            local_conflict_rate = None
            if config.max_environment_steps is not None:
                epsilon = linear_epsilon_at_environment_step(
                    environment_steps,
                    config,
                )
            if (
                adaptive_controller is None
                and config.demo_fraction_decay_start_environment_step is not None
            ):
                current_demo_fraction = demo_fraction_at_environment_step(
                    environment_steps,
                    config,
                )
                if callable(set_demo_fraction):
                    set_demo_fraction(current_demo_fraction)
                episode_demo_fractions.append(current_demo_fraction)
            if adaptive_controller is not None:
                conflict_rate = _online_demo_action_conflict_rate(
                    env,
                    demo_actions_by_position,
                    predict_next=config.conflict_predict_next,
                )
                if conflict_rate is not None:
                    episode_demo_conflicts.append(conflict_rate)
                    current_demo_fraction = adaptive_controller.observe(conflict_rate)
                    if callable(set_demo_fraction):
                        set_demo_fraction(current_demo_fraction)
                episode_demo_fractions.append(current_demo_fraction)
            if config.replay_strategy in {
                "local_conflict_demo",
                "local_counterexample_demo",
            }:
                if not isinstance(replay, LocalConflictDemoReplay):
                    raise RuntimeError("Local-conflict strategy has the wrong replay type.")
                action_risks, local_conflict_rate = _online_demo_action_risks(
                    env,
                    demo_actions_by_position,
                    predict_next=config.local_conflict_predict_next,
                )
                replay.observe_local_conflicts(tuple(env.position), action_risks)
                if local_conflict_rate is not None:
                    episode_demo_conflicts.append(local_conflict_rate)
                    local_conflict_covered_count += 1
                    local_conflict_event_count += int(local_conflict_rate > 0.0)
                episode_demo_fractions.append(current_demo_fraction)
            safe_demo_action = None
            conflict_safe_action_mask = None
            conflict_blocked_action_mask = None
            if config.replay_strategy == "predictive_margin_demo":
                (
                    conflict_safe_action_mask,
                    conflict_blocked_action_mask,
                    margin_covered,
                    margin_conflict,
                    margin_no_safe,
                    margin_risky_fraction,
                ) = _predictive_conflict_margin_masks(
                    env,
                    demo_actions_by_position,
                    predict_next=config.conflict_margin_predict_next,
                )
                conflict_margin_covered_count += int(margin_covered)
                conflict_margin_conflict_count += int(margin_conflict)
                conflict_margin_no_safe_count += int(margin_no_safe)
                conflict_margin_labeled_count += int(
                    conflict_safe_action_mask is not None
                )
                if margin_covered:
                    conflict_margin_risky_fractions.append(margin_risky_fraction)
            if config.replay_strategy == "safe_guided_demo":
                (
                    safe_demo_action,
                    guidance_covered,
                    guidance_all_blocked,
                    guidance_risky_fraction,
                ) = _safe_demo_guidance(
                    env,
                    demo_actions_by_position,
                    predict_next=config.safe_guidance_predict_next,
                )
                guidance_candidate_count += int(guidance_covered)
                guidance_applied_count += int(safe_demo_action is not None)
                guidance_blocked_count += int(guidance_all_blocked)
                if guidance_covered:
                    guidance_risky_fractions.append(guidance_risky_fraction)
            if config.mask_static_invalid_actions:
                static_safe_mask = env.action_mask(mask_collisions=True)
                valid_actions = [
                    index
                    for index, is_valid in enumerate(static_safe_mask)
                    if bool(is_valid)
                ]
                action = agent.select_action(
                    state,
                    epsilon=epsilon,
                    valid_actions=valid_actions,
                )
            else:
                action = agent.select_action(state, epsilon=epsilon)
            counterexample_candidate = (
                isinstance(replay, LocalCounterexampleDemoReplay)
                and local_conflict_rate is not None
                and local_conflict_rate > 0.0
            )
            counterexample_action_safe = False
            if counterexample_candidate:
                local_counterexample_candidate_count += 1
                risk = getattr(env, "dynamic_action_collision_risk", None)
                if not callable(risk):
                    raise ValueError(
                        "local_counterexample_demo requires dynamic-action risk."
                    )
                counterexample_action_safe = not bool(
                    risk(action, predict_next=config.local_conflict_predict_next)
                )
            result = env.step(action)
            transition = Transition(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.observation,
                terminated=result.terminated,
                next_action_mask=env.action_mask(
                    mask_collisions=config.mask_static_invalid_actions
                ),
                safe_demo_action=safe_demo_action,
                conflict_safe_action_mask=conflict_safe_action_mask,
                conflict_blocked_action_mask=conflict_blocked_action_mask,
            )
            replay.add(transition)
            if (
                isinstance(replay, LocalCounterexampleDemoReplay)
                and counterexample_action_safe
                and not bool(result.info["collision"])
            ):
                replay.add_counterexample(transition)
                local_counterexample_added_count += 1
                local_counterexample_wait_count += int(action == int(Action.STAY))
            environment_steps += 1
            episode_return += result.reward
            state = result.observation
            final_info = dict(result.info)
            collisions.record(result.info, env.steps)
            collision_position = None
            if bool(result.info["collision"]):
                collision_position = tuple(
                    result.info["collision_position"] or env.position
                )
            trajectory.record(action, env.position, collision_position)

            if environment_steps >= config.learning_starts:
                for _ in range(config.updates_per_step):
                    if not replay.can_sample(config.batch_size):
                        break
                    if hasattr(replay, "set_beta"):
                        training_step_budget = (
                            config.max_environment_steps
                            if config.max_environment_steps is not None
                            else config.episodes * config.max_steps
                        )
                        progress = min(
                            1.0,
                            environment_steps
                            / max(1, training_step_budget),
                        )
                        beta = config.per_beta_start + progress * (
                            config.per_beta_end - config.per_beta_start
                        )
                        replay.set_beta(beta)  # type: ignore[attr-defined]
                    batch = replay.sample(config.batch_size)
                    sample_weights = (
                        replay.sample_weights()  # type: ignore[attr-defined]
                        if hasattr(replay, "sample_weights")
                        else None
                    )
                    demonstration_mask = (
                        [id(item) in demonstration_ids for item in batch]
                        if config.replay_strategy == "dqfd"
                        else None
                    )
                    update = agent.train_batch(
                        batch,
                        sample_weights=sample_weights,
                        demonstration_mask=demonstration_mask,
                        demo_margin=(
                            config.demo_margin
                            if config.replay_strategy == "dqfd"
                            else 0.0
                        ),
                        demo_loss_weight=(
                            config.demo_loss_weight
                            if config.replay_strategy == "dqfd"
                            else 0.0
                        ),
                        safe_guidance_margin=(
                            config.safe_guidance_margin
                            if config.replay_strategy == "safe_guided_demo"
                            else 0.0
                        ),
                        safe_guidance_loss_weight=(
                            config.safe_guidance_loss_weight
                            if config.replay_strategy == "safe_guided_demo"
                            else 0.0
                        ),
                        conflict_margin=(
                            config.conflict_margin
                            if config.replay_strategy == "predictive_margin_demo"
                            else 0.0
                        ),
                        conflict_margin_loss_weight=(
                            config.conflict_margin_loss_weight
                            if config.replay_strategy == "predictive_margin_demo"
                            else 0.0
                        ),
                    )
                    if hasattr(replay, "update_priorities"):
                        replay.update_priorities(  # type: ignore[attr-defined]
                            batch, update["td_errors"]
                        )
                    losses.append(update["loss"])
                    guidance_losses.append(update["safe_guidance_margin_loss"])
                    guidance_batch_counts.append(update["safe_guidance_batch_count"])
                    conflict_margin_losses.append(update["conflict_margin_loss"])
                    conflict_margin_batch_counts.append(
                        update["conflict_margin_batch_count"]
                    )
                    gradient_updates += 1
            if result.done or (
                config.max_environment_steps is not None
                and environment_steps >= config.max_environment_steps
            ):
                break

        trajectory.finish()
        episode_success = bool(final_info.get("reached", False))
        records.append(
            {
                "episode": episode,
                "map_id": problem.map_id,
                "map_seed": problem.seed,
                "scenario_id": getattr(env, "scenario_id", None),
                "required_behavior": getattr(env, "required_behavior", None),
                "difficulty_stratum": getattr(
                    env, "difficulty_stratum", None
                ),
                "curriculum_stage": getattr(env, "curriculum_stage", ""),
                "curriculum_active_stage": getattr(
                    env,
                    "curriculum_active_stage",
                    getattr(env, "curriculum_stage", ""),
                ),
                "curriculum_rehearsal": float(
                    bool(getattr(env, "curriculum_rehearsal", False))
                ),
                "curriculum_stage_start_environment_step": getattr(
                    env,
                    "curriculum_stage_start_environment_step",
                    "",
                ),
                "dynamic_obstacle_count": len(
                    getattr(env, "dynamic_obstacles", ())
                ),
                "reward": episode_return,
                "steps": env.steps,
                "success": float(episode_success),
                "collision": float(collisions.total_count > 0),
                "collision_count": collisions.total_count,
                **collisions.metrics(),
                "safe_success": float(
                    episode_success and collisions.total_count == 0
                ),
                "collision_success": float(
                    episode_success and collisions.total_count > 0
                ),
                "static_collision_success": float(
                    episode_success and collisions.static_count > 0
                ),
                "dynamic_collision_success": float(
                    episode_success and collisions.dynamic_count > 0
                ),
                "dynamic_collision_free_success": float(
                    episode_success and collisions.dynamic_count == 0
                ),
                "revisit_count": len(trajectory.revisit_positions),
                "wait_steps": trajectory.wait_steps,
                "wait_event_count": trajectory.wait_event_count,
                "max_wait_streak": trajectory.max_wait_streak,
                "wait_ratio": trajectory.wait_steps / max(1, env.steps),
                "termination_reason": (
                    "training_budget"
                    if config.max_environment_steps is not None
                    and environment_steps >= config.max_environment_steps
                    and not bool(result.done)
                    else final_info.get("termination_reason", "unknown")
                ),
                "training_budget_reached": float(
                    config.max_environment_steps is not None
                    and environment_steps >= config.max_environment_steps
                    and not bool(result.done)
                ),
                "epsilon": epsilon,
                "demo_fraction": (
                    current_demo_fraction
                    if config.replay_strategy
                    in {
                        "persistent_demo",
                        "conflict_adaptive_demo",
                        "local_conflict_demo",
                        "local_counterexample_demo",
                        "predictive_margin_demo",
                        "dqfd",
                    }
                    else 0.0
                ),
                "demo_fraction_mean": (
                    sum(episode_demo_fractions) / len(episode_demo_fractions)
                    if episode_demo_fractions
                    else current_demo_fraction
                    if config.replay_strategy
                    in {
                        "persistent_demo",
                        "conflict_adaptive_demo",
                        "local_conflict_demo",
                        "local_counterexample_demo",
                        "predictive_margin_demo",
                        "dqfd",
                    }
                    else 0.0
                ),
                "demo_action_conflict_rate": (
                    sum(episode_demo_conflicts) / len(episode_demo_conflicts)
                    if episode_demo_conflicts
                    else 0.0
                ),
                "demo_conflict_observation_count": len(episode_demo_conflicts),
                "demo_conflict_ema": (
                    adaptive_controller.conflict_ema
                    if adaptive_controller is not None
                    else 0.0
                ),
                "local_conflict_covered_count": local_conflict_covered_count,
                "local_conflict_event_count": local_conflict_event_count,
                "local_suppressed_demo_keys": (
                    replay.suppressed_key_count
                    if isinstance(replay, LocalConflictDemoReplay)
                    else 0
                ),
                "local_suppressed_demo_transitions": (
                    replay.suppressed_transition_count
                    if isinstance(replay, LocalConflictDemoReplay)
                    else 0
                ),
                "local_demo_suppression_rate": (
                    replay.suppression_rate
                    if isinstance(replay, LocalConflictDemoReplay)
                    else 0.0
                ),
                "local_counterexample_candidate_count": (
                    local_counterexample_candidate_count
                ),
                "local_counterexample_added_count": local_counterexample_added_count,
                "local_counterexample_wait_count": local_counterexample_wait_count,
                "local_counterexample_add_rate": (
                    local_counterexample_added_count / max(1, env.steps)
                ),
                "local_counterexample_wait_share": (
                    local_counterexample_wait_count
                    / max(1, local_counterexample_added_count)
                ),
                "local_counterexample_buffer_size": (
                    replay.counterexample_size
                    if isinstance(replay, LocalCounterexampleDemoReplay)
                    else 0
                ),
                "local_counterexample_sample_fraction": (
                    replay.counterexample_sample_fraction
                    if isinstance(replay, LocalCounterexampleDemoReplay)
                    else 0.0
                ),
                "conflict_margin_covered_count": conflict_margin_covered_count,
                "conflict_margin_conflict_count": conflict_margin_conflict_count,
                "conflict_margin_labeled_count": conflict_margin_labeled_count,
                "conflict_margin_no_safe_count": conflict_margin_no_safe_count,
                "conflict_margin_label_rate": (
                    conflict_margin_labeled_count / max(1, env.steps)
                ),
                "conflict_margin_conflict_coverage_rate": (
                    conflict_margin_labeled_count
                    / max(1, conflict_margin_conflict_count)
                ),
                "conflict_margin_mean_risky_fraction": (
                    sum(conflict_margin_risky_fractions)
                    / len(conflict_margin_risky_fractions)
                    if conflict_margin_risky_fractions
                    else 0.0
                ),
                "conflict_margin_loss": (
                    sum(conflict_margin_losses) / len(conflict_margin_losses)
                    if conflict_margin_losses
                    else 0.0
                ),
                "conflict_margin_batch_count_mean": (
                    sum(conflict_margin_batch_counts)
                    / len(conflict_margin_batch_counts)
                    if conflict_margin_batch_counts
                    else 0.0
                ),
                "conflict_margin_batch_fraction": (
                    sum(conflict_margin_batch_counts)
                    / (len(conflict_margin_batch_counts) * config.batch_size)
                    if conflict_margin_batch_counts
                    else 0.0
                ),
                "safe_guidance_candidate_count": guidance_candidate_count,
                "safe_guidance_applied_count": guidance_applied_count,
                "safe_guidance_blocked_count": guidance_blocked_count,
                "safe_guidance_coverage_rate": (
                    guidance_candidate_count / max(1, env.steps)
                ),
                "safe_guidance_application_rate": (
                    guidance_applied_count / max(1, guidance_candidate_count)
                ),
                "safe_guidance_mean_risky_fraction": (
                    sum(guidance_risky_fractions) / len(guidance_risky_fractions)
                    if guidance_risky_fractions
                    else 0.0
                ),
                "safe_guidance_margin_loss": (
                    sum(guidance_losses) / len(guidance_losses)
                    if guidance_losses
                    else 0.0
                ),
                "safe_guidance_batch_count_mean": (
                    sum(guidance_batch_counts) / len(guidance_batch_counts)
                    if guidance_batch_counts
                    else 0.0
                ),
                "loss": sum(losses) / len(losses) if losses else 0.0,
                "environment_steps_total": environment_steps,
                "gradient_updates_total": gradient_updates,
                "replay_size": len(replay),
                "demo_retained_count": sum(
                    id(item) in demonstration_ids for item in replay.snapshot()
                ),
                "demo_retention_rate": (
                    sum(id(item) in demonstration_ids for item in replay.snapshot())
                    / max(1, len(demonstrations))
                    if demonstrations
                    else 0.0
                ),
            }
        )
        full_evaluation = (
            crossed_interval(
                environment_steps_before_episode,
                environment_steps,
                config.progress_interval_environment_steps,
            )
            if config.progress_interval_environment_steps is not None
            else (
                config.progress_interval > 0
                and episode % config.progress_interval == 0
            )
        )
        diagnostic_evaluation = (
            config.diagnostic_interval > 0
            and episode % config.diagnostic_interval == 0
        )
        if progress_callback is not None and (full_evaluation or diagnostic_evaluation):
            callback_started_at = perf_counter()
            continue_training = progress_callback(
                episode, tuple(records), full_evaluation
            )
            progress_callback_seconds += perf_counter() - callback_started_at
            if continue_training is False:
                stopped_early = True
                break
    training_seconds = perf_counter() - started_at - progress_callback_seconds
    return TrainingResult(
        tuple(records),
        environment_steps,
        gradient_updates,
        training_seconds,
        progress_callback_seconds,
        stopped_early,
    )
