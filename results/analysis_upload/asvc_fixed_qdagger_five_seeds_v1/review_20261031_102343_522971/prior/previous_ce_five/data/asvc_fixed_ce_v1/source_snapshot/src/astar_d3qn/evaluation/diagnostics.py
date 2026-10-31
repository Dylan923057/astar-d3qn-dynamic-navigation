from __future__ import annotations

from collections.abc import Iterable

from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.core.astar import astar_path
from astar_d3qn.core.grid import Action, in_bounds, move
from astar_d3qn.envs.types import Observation
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition


def _position_from_observation(
    observation: Observation,
    problem: NavigationProblem,
) -> tuple[int, int] | None:
    """Recover the agent cell from the normalized goal displacement scalars."""

    if observation.scalars.shape != (2,):
        return None
    scale = max(1, problem.size - 1)
    row = int(round(problem.goal[0] - float(observation.scalars[0]) * scale))
    column = int(round(problem.goal[1] - float(observation.scalars[1]) * scale))
    position = (row, column)
    return position if in_bounds(position, problem.size) else None


def optimal_action_set(
    position: tuple[int, int],
    problem: NavigationProblem,
) -> frozenset[int]:
    """Return actions that preserve an A*-shortest route to the goal."""

    current_path = astar_path(
        position, problem.goal, problem.obstacles, problem.size
    )
    if current_path is None:
        return frozenset()
    current_steps = len(current_path) - 1
    actions: set[int] = set()
    for action in Action:
        if action == Action.STAY:
            continue
        candidate = move(position, action)
        if not in_bounds(candidate, problem.size) or candidate in problem.obstacles:
            continue
        candidate_path = astar_path(
            candidate, problem.goal, problem.obstacles, problem.size
        )
        if candidate_path is not None and 1 + len(candidate_path) - 1 == current_steps:
            actions.add(int(action))
    return frozenset(actions)


def demonstration_action_diagnostics(
    agent: D3QNAgent,
    demonstrations: Iterable[Transition],
    problem: NavigationProblem,
) -> dict[str, float]:
    """Measure greedy agreement with the set of shortest-path actions."""

    total = 0
    valid = 0
    agreement = 0
    exact_agreement = 0
    for transition in demonstrations:
        total += 1
        position = _position_from_observation(transition.state, problem)
        if position is None:
            continue
        optimal = optimal_action_set(position, problem)
        if not optimal:
            continue
        valid += 1
        predicted = agent.select_action(transition.state, epsilon=0.0)
        agreement += int(predicted in optimal)
        exact_agreement += int(predicted == transition.action)
    return {
        "demo_transition_count": float(total),
        "demo_valid_state_count": float(valid),
        "demo_optimal_action_agreement": agreement / max(1, valid),
        "demo_exact_action_agreement": exact_agreement / max(1, valid),
    }
