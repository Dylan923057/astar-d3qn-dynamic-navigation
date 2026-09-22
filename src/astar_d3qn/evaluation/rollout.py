from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.core.collisions import CollisionTracker
from astar_d3qn.core.grid import Position
from astar_d3qn.core.metrics import path_efficiency
from astar_d3qn.core.trajectory import TrajectoryRecorder, WaitEvent
from astar_d3qn.envs.static_grid import RewardConfig, StaticGridNavigationEnv
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.training.trainer import EnvironmentFactory


@dataclass(frozen=True, slots=True)
class EvaluationTrajectory:
    path: tuple[Position, ...]
    collision_positions: tuple[Position, ...]
    static_collision_positions: tuple[Position, ...]
    dynamic_collision_positions: tuple[Position, ...]
    revisit_positions: tuple[Position, ...]
    wait_events: tuple[WaitEvent, ...]


def path_turn_metrics(path: Sequence[Position]) -> tuple[int, int, float]:
    """Return movement steps, direction changes, and turn rate.

    Repeated positions from STAY actions or collision holds are excluded from
    direction changes. This measures geometric path smoothness separately from
    success and path length.
    """

    movements = [
        (current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(path, path[1:])
        if current != previous
    ]
    turns = sum(
        previous != current for previous, current in zip(movements, movements[1:])
    )
    return len(movements), turns, turns / max(1, len(movements) - 1)


def evaluate_agent(
    agent: D3QNAgent,
    problems: Sequence[NavigationProblem],
    max_steps: int,
    reward_config: RewardConfig | None = None,
    terminate_on_collision: bool = False,
    window_size: int | None = 11,
    environment_factory: EnvironmentFactory | None = None,
    mask_static_invalid_actions: bool = False,
) -> tuple[dict, list[dict], dict[str, EvaluationTrajectory]]:
    rows: list[dict] = []
    trajectories: dict[str, EvaluationTrajectory] = {}
    factory = environment_factory or StaticGridNavigationEnv
    map_id_counts = Counter(problem.map_id for problem in problems)
    for evaluation_index, problem in enumerate(problems):
        env = factory(
            problem,
            max_steps=max_steps,
            reward_config=reward_config,
            terminate_on_collision=terminate_on_collision,
            window_size=window_size,
        )
        state = env.reset()
        total_reward = 0.0
        collisions = CollisionTracker()
        static_collision_positions: list[Position] = []
        dynamic_collision_positions: list[Position] = []
        action_selection_seconds = 0.0
        trajectory = TrajectoryRecorder(problem.start, problem.goal)
        while True:
            action_started_at = perf_counter()
            if mask_static_invalid_actions:
                static_safe_mask = env.action_mask(mask_collisions=True)
                valid_actions = [
                    index
                    for index, is_valid in enumerate(static_safe_mask)
                    if bool(is_valid)
                ]
                action = agent.select_action(
                    state,
                    epsilon=0.0,
                    valid_actions=valid_actions,
                )
            else:
                action = agent.select_action(state, epsilon=0.0)
            action_selection_seconds += perf_counter() - action_started_at
            result = env.step(action)
            state = result.observation
            total_reward += result.reward
            collision_position = None
            if bool(result.info["collision"]):
                collision_position = tuple(
                    result.info["collision_position"] or env.position
                )
                collision_type = result.info.get("collision_type") or "static"
                if collision_type == "dynamic":
                    dynamic_collision_positions.append(collision_position)
                else:
                    static_collision_positions.append(collision_position)
            collisions.record(result.info, env.steps)
            trajectory.record(action, env.position, collision_position)
            if result.done:
                break
        trajectory.finish()
        success = bool(result.info["reached"])
        scenario_id = getattr(env, "scenario_id", None)
        if map_id_counts[problem.map_id] == 1:
            trajectory_key = problem.map_id
        elif scenario_id is not None:
            trajectory_key = f"{problem.map_id}::scenario_{scenario_id}"
        else:
            trajectory_key = f"{problem.map_id}::evaluation_{evaluation_index:05d}"
        if trajectory_key in trajectories:
            trajectory_key = f"{trajectory_key}::evaluation_{evaluation_index:05d}"
        trajectories[trajectory_key] = EvaluationTrajectory(
            path=tuple(trajectory.path),
            collision_positions=tuple(trajectory.collision_positions),
            static_collision_positions=tuple(static_collision_positions),
            dynamic_collision_positions=tuple(dynamic_collision_positions),
            revisit_positions=tuple(trajectory.revisit_positions),
            wait_events=tuple(trajectory.wait_events),
        )
        movement_steps, turn_count, turn_rate = path_turn_metrics(trajectory.path)
        rows.append(
            {
                "map_id": problem.map_id,
                "map_seed": problem.seed,
                "scenario_id": scenario_id,
                "required_behavior": getattr(env, "required_behavior", None),
                "difficulty_stratum": getattr(env, "difficulty_stratum", None),
                "trajectory_key": trajectory_key,
                "dynamic_obstacle_count": len(
                    getattr(env, "dynamic_obstacles", ())
                ),
                "success": float(success),
                "collision": float(collisions.total_count > 0),
                "collision_count": collisions.total_count,
                **collisions.metrics(),
                "revisit_count": len(trajectory.revisit_positions),
                "wait_steps": trajectory.wait_steps,
                "wait_event_count": trajectory.wait_event_count,
                "max_wait_streak": trajectory.max_wait_streak,
                "wait_ratio": trajectory.wait_steps / max(1, env.steps),
                "steps": env.steps,
                "movement_steps": movement_steps,
                "turn_count": turn_count,
                "turn_rate": turn_rate,
                "reward": total_reward,
                "safe_success": float(success and collisions.total_count == 0),
                "collision_success": float(success and collisions.total_count > 0),
                "static_collision_success": float(
                    success and collisions.static_count > 0
                ),
                "dynamic_collision_success": float(
                    success and collisions.dynamic_count > 0
                ),
                "dynamic_collision_free_success": float(
                    success and collisions.dynamic_count == 0
                ),
                "path_efficiency": (
                    path_efficiency(env.steps, problem.astar_steps) if success else None
                ),
                "termination_reason": result.info["termination_reason"],
                "action_selection_seconds": action_selection_seconds,
                "mean_action_latency_ms": 1000.0
                * action_selection_seconds
                / max(1, env.steps),
            }
        )
    summary = {
        "maps": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "collision_rate": sum(row["collision"] for row in rows) / len(rows),
        "mean_collision_count": sum(row["collision_count"] for row in rows)
        / len(rows),
        "static_collision_rate": sum(row["static_collision"] for row in rows)
        / len(rows),
        "mean_static_collision_count": sum(
            row["static_collision_count"] for row in rows
        )
        / len(rows),
        "dynamic_collision_rate": sum(row["dynamic_collision"] for row in rows)
        / len(rows),
        "mean_dynamic_collision_count": sum(
            row["dynamic_collision_count"] for row in rows
        )
        / len(rows),
        "safe_success_rate": sum(row["safe_success"] for row in rows) / len(rows),
        "collision_success_rate": sum(row["collision_success"] for row in rows)
        / len(rows),
        "static_collision_success_rate": sum(
            row["static_collision_success"] for row in rows
        )
        / len(rows),
        "dynamic_collision_success_rate": sum(
            row["dynamic_collision_success"] for row in rows
        )
        / len(rows),
        "dynamic_collision_free_success_rate": sum(
            row["dynamic_collision_free_success"] for row in rows
        )
        / len(rows),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "mean_wait_steps": sum(row["wait_steps"] for row in rows) / len(rows),
        "mean_wait_event_count": sum(row["wait_event_count"] for row in rows)
        / len(rows),
        "mean_max_wait_streak": sum(row["max_wait_streak"] for row in rows)
        / len(rows),
        "mean_movement_steps": sum(row["movement_steps"] for row in rows) / len(rows),
        "mean_turn_count": sum(row["turn_count"] for row in rows) / len(rows),
        "mean_turn_rate": sum(row["turn_rate"] for row in rows) / len(rows),
        "mean_revisit_count": sum(row["revisit_count"] for row in rows) / len(rows),
        "mean_reward": sum(row["reward"] for row in rows) / len(rows),
        "mean_path_efficiency": (
            sum(
                float(row["path_efficiency"])
                for row in rows
                if row["path_efficiency"] is not None
            )
            / sum(row["path_efficiency"] is not None for row in rows)
            if any(row["path_efficiency"] is not None for row in rows)
            else None
        ),
        "mean_first_static_collision_step": (
            sum(
                float(row["first_static_collision_step"])
                for row in rows
                if row["first_static_collision_step"] is not None
            )
            / max(1, sum(row["first_static_collision_step"] is not None for row in rows))
            if any(row["first_static_collision_step"] is not None for row in rows)
            else None
        ),
        "mean_first_dynamic_collision_step": (
            sum(
                float(row["first_dynamic_collision_step"])
                for row in rows
                if row["first_dynamic_collision_step"] is not None
            )
            / max(1, sum(row["first_dynamic_collision_step"] is not None for row in rows))
            if any(row["first_dynamic_collision_step"] is not None for row in rows)
            else None
        ),
        "mean_action_latency_ms": 1000.0
        * sum(row["action_selection_seconds"] for row in rows)
        / max(1, sum(row["steps"] for row in rows)),
    }
    return summary, rows, trajectories
