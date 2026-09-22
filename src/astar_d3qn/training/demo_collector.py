from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from astar_d3qn.core.astar import randomized_tie_astar_path
from astar_d3qn.core.grid import Position, action_between, move
from astar_d3qn.envs.static_grid import RewardConfig, StaticGridNavigationEnv
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.types import Observation
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition


REWARD_SEMANTICS_VERSION = "exclusive_terminal_events_v1"


def demonstration_signature(
    problems: Sequence[NavigationProblem],
    episodes: int,
    seed: int,
    max_steps: int,
    reward_config: RewardConfig,
    window_size: int,
    spatial_channels: int = 1,
    observation_mode: str = "local_static_grid_goal_vector",
    environment_id: str | None = None,
) -> dict[str, Any]:
    if spatial_channels <= 0:
        raise ValueError("spatial_channels must be positive.")
    signature = {
        "episodes": int(episodes),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "observation": {
            "mode": str(observation_mode),
            "window_size": int(window_size),
            "spatial_shape": [
                int(spatial_channels),
                int(window_size),
                int(window_size),
            ],
            "scalar_dim": 2,
            "goal_normalization": "map_extent",
        },
        "maps": [
            {
                "map_id": problem.map_id,
                "grid_sha256": problem.grid_sha256,
                "start": list(problem.start),
                "goal": list(problem.goal),
            }
            for problem in problems
        ],
        "reward": asdict(reward_config),
        "reward_semantics": REWARD_SEMANTICS_VERSION,
    }
    if environment_id is not None:
        signature["environment_id"] = str(environment_id)
    return signature


def demonstration_file_sha256(path: str | Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_demonstration_dataset(
    transitions: Sequence[Transition],
    metadata: Mapping[str, Any],
    expected_signature: Mapping[str, Any],
    dataset_path: str | Path,
) -> None:
    if metadata.get("format_version") != 3:
        raise ValueError("Unsupported demonstration metadata format.")
    for key, expected_value in expected_signature.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Demonstration metadata field {key!r} does not match the "
                "current experiment. Regenerate the demonstration dataset."
            )
    if int(metadata.get("transitions", -1)) != len(transitions):
        raise ValueError("Demonstration transition count does not match metadata.")
    if metadata.get("dataset_sha256") != demonstration_file_sha256(dataset_path):
        raise ValueError("Demonstration dataset hash does not match metadata.")


def paths_from_demonstrations(
    problem: NavigationProblem,
    transitions: Sequence[Transition],
) -> list[tuple[Position, ...]]:
    """Reconstruct and validate the exact paths stored in a frozen demo file.

    Scenario generation must use the same paths that replay training consumes.
    Re-running randomized A* with a superficially similar seed is not equivalent:
    even a different RNG reset convention changes tie-broken paths.
    """

    if not transitions:
        raise ValueError("Cannot reconstruct paths from an empty demonstration set.")
    scale = max(1, problem.size - 1)

    def scalar_position(values: np.ndarray) -> Position:
        if values.shape != (2,):
            raise ValueError("Frozen demonstrations must contain two goal scalars.")
        row = problem.goal[0] - int(round(float(values[0]) * scale))
        column = problem.goal[1] - int(round(float(values[1]) * scale))
        return row, column

    paths: list[tuple[Position, ...]] = []
    current = problem.start
    path: list[Position] = [current]
    for index, transition in enumerate(transitions):
        encoded_current = scalar_position(transition.state.scalars)
        if encoded_current != current:
            raise ValueError(
                f"Demonstration transition {index} is discontinuous: state encodes "
                f"{encoded_current}, expected {current}."
            )
        following = move(current, transition.action)
        encoded_following = scalar_position(transition.next_state.scalars)
        if encoded_following != following:
            raise ValueError(
                f"Demonstration transition {index} next state encodes "
                f"{encoded_following}, but action reaches {following}."
            )
        path.append(following)
        current = following
        if transition.terminated:
            if current != problem.goal:
                raise ValueError(
                    f"Demonstration episode ending at transition {index} does not "
                    "terminate at the registered goal."
                )
            paths.append(tuple(path))
            current = problem.start
            path = [current]
        elif current == problem.goal:
            raise ValueError(
                f"Demonstration transition {index} reaches the goal without a "
                "terminal flag."
            )
    if len(path) != 1:
        raise ValueError("Frozen demonstration dataset ends during an episode.")
    return paths


def collect_astar_demonstrations(
    problems: Sequence[NavigationProblem],
    episodes: int,
    seed: int,
    max_steps: int = 350,
    reward_config: RewardConfig | None = None,
    window_size: int = 11,
    spatial_channels: int = 1,
    environment_factory=None,
    mask_static_invalid_actions: bool = False,
) -> list[Transition]:
    if not problems:
        raise ValueError("At least one problem is required.")
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if spatial_channels <= 0:
        raise ValueError("spatial_channels must be positive.")
    rng = random.Random(seed)
    demonstrations: list[Transition] = []

    for episode in range(episodes):
        problem = problems[episode % len(problems)]
        path = randomized_tie_astar_path(
            problem.start,
            problem.goal,
            problem.obstacles,
            problem.size,
            rng,
        )
        if path is None:
            raise RuntimeError(f"A* failed on registered solvable map {problem.map_id}.")
        if environment_factory is None:
            env = StaticGridNavigationEnv(
                problem,
                max_steps=max(max_steps, len(path)),
                reward_config=reward_config,
                window_size=window_size,
            )
        else:
            env = environment_factory(
                problem,
                max_steps=max(max_steps, len(path) + 100),
                reward_config=reward_config,
                terminate_on_collision=False,
                window_size=window_size,
            )
        state = env.reset()
        for current, following in zip(path, path[1:]):
            action = int(action_between(current, following))
            while env.position != following:
                if env.position != current:
                    raise RuntimeError(
                        "Environment diverged from the A* demonstration path."
                    )
                result = env.step(action)
                state_for_replay = _pad_spatial_channels(state, spatial_channels)
                next_state_for_replay = _pad_spatial_channels(
                    result.observation, spatial_channels
                )
                demonstrations.append(
                    Transition(
                        state=state_for_replay,
                        action=action,
                        reward=result.reward,
                        next_state=next_state_for_replay,
                        terminated=result.terminated,
                        next_action_mask=env.action_mask(
                            mask_collisions=mask_static_invalid_actions
                        ),
                    )
                )
                state = result.observation
                if result.done:
                    if env.position != problem.goal:
                        raise RuntimeError(
                            "A* dynamic demonstration terminated before reaching its target cell."
                        )
                    break
            if env.position != following:
                break
        if env.position != problem.goal or not result.terminated:
            raise RuntimeError("A* demonstration did not terminate at the goal.")
    return demonstrations


def _pad_spatial_channels(
    observation: Observation,
    spatial_channels: int,
) -> Observation:
    source_channels = observation.spatial.shape[0]
    if source_channels > spatial_channels:
        raise ValueError(
            "Observation has more spatial channels than the configured replay input."
        )
    if source_channels == spatial_channels:
        return observation
    spatial = np.zeros(
        (spatial_channels, *observation.spatial.shape[1:]), dtype=np.float32
    )
    spatial[0] = observation.spatial[0]
    if source_channels > 1:
        spatial[1:source_channels] = observation.spatial[1:]
    return Observation(spatial=spatial, scalars=observation.scalars.copy())


def save_demonstrations(
    transitions: Sequence[Transition], path: str | Path
) -> None:
    if not transitions:
        raise ValueError("Cannot save an empty demonstration dataset.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    masks = np.stack(
        [
            np.asarray(item.next_action_mask, dtype=bool)
            if item.next_action_mask is not None
            else np.ones(0, dtype=bool)
            for item in transitions
        ]
    )
    np.savez_compressed(
        target,
        state_spatial=np.stack([item.state.spatial for item in transitions]),
        state_scalars=np.stack([item.state.scalars for item in transitions]),
        actions=np.asarray([item.action for item in transitions], dtype=np.int64),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        next_state_spatial=np.stack(
            [item.next_state.spatial for item in transitions]
        ),
        next_state_scalars=np.stack(
            [item.next_state.scalars for item in transitions]
        ),
        terminated=np.asarray([item.terminated for item in transitions], dtype=bool),
        next_action_masks=masks,
    )


def load_demonstrations(path: str | Path) -> list[Transition]:
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "state_spatial",
            "state_scalars",
            "actions",
            "rewards",
            "next_state_spatial",
            "next_state_scalars",
            "terminated",
            "next_action_masks",
        }
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(
                "Demonstration dataset uses an incompatible observation format; "
                "regenerate it for the configured local observation. Missing: "
                + ", ".join(sorted(missing))
            )
        return [
            Transition(
                state=Observation(spatial=state_spatial, scalars=state_scalars),
                action=int(action),
                reward=float(reward),
                next_state=Observation(
                    spatial=next_state_spatial,
                    scalars=next_state_scalars,
                ),
                terminated=bool(terminated),
                next_action_mask=mask,
            )
            for (
                state_spatial,
                state_scalars,
                action,
                reward,
                next_state_spatial,
                next_state_scalars,
                terminated,
                mask,
            ) in zip(
                dataset["state_spatial"],
                dataset["state_scalars"],
                dataset["actions"],
                dataset["rewards"],
                dataset["next_state_spatial"],
                dataset["next_state_scalars"],
                dataset["terminated"],
                dataset["next_action_masks"],
            )
        ]
