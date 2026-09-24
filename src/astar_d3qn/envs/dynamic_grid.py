from __future__ import annotations

from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from astar_d3qn.core.grid import (
    ACTION_DELTAS,
    Action,
    Position,
    in_bounds,
    manhattan,
    move,
)
from astar_d3qn.maps.problem import NavigationProblem

from .static_grid import RewardConfig
from .types import Observation, StepResult


@dataclass(frozen=True, slots=True)
class DynamicObstacleSpec:
    """A deterministic obstacle moving back and forth on a grid route."""

    route: tuple[Position, ...]
    start_index: int = 0
    direction: int = 1
    move_every: int = 1
    label: str = "dynamic_obstacle"
    reference_path_source: str = "nominal_path"
    reference_path_index: int | None = None

    def __post_init__(self) -> None:
        if len(self.route) < 2:
            raise ValueError("A dynamic obstacle route must contain at least two cells.")
        if not 0 <= self.start_index < len(self.route):
            raise ValueError("start_index must refer to a route cell.")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be either -1 or 1.")
        if self.move_every <= 0:
            raise ValueError("move_every must be positive.")
        for left, right in zip(self.route, self.route[1:]):
            if manhattan(left, right) != 1:
                raise ValueError("Dynamic obstacle routes must be four-connected.")


def _valid_route(
    route: Sequence[Position],
    problem: NavigationProblem,
    *,
    min_length: int,
) -> bool:
    if len(route) < min_length or len(set(route)) != len(route):
        return False
    if any(not in_bounds(cell, problem.size) for cell in route):
        return False
    if any(cell in problem.obstacles for cell in route):
        return False
    if problem.start in route or problem.goal in route:
        return False
    return all(manhattan(left, right) == 1 for left, right in zip(route, route[1:]))


def build_crossing_obstacle_spec(
    problem: NavigationProblem,
    route_length: int = 5,
    center_index: int | None = None,
    start_index: int = 0,
    direction: int = 1,
    move_every: int = 1,
    label: str = "crossing",
) -> DynamicObstacleSpec:
    """Build a reproducible free horizontal/vertical route crossing nominal A*.

    The route is selected from interior nominal-path cells and contains no static
    obstacle, start, or goal. This keeps the first dynamic benchmark deterministic
    while guaranteeing that the moving obstacle can interfere with the reference.
    """

    if route_length < 3 or route_length % 2 == 0:
        raise ValueError("route_length must be an odd integer of at least three.")
    half = route_length // 2
    path_cells = tuple(problem.nominal_path[1:-1])
    if center_index is not None:
        if not 1 <= center_index < len(problem.nominal_path) - 1:
            raise ValueError("center_index must refer to an interior nominal-path cell.")
        path_cells = (problem.nominal_path[center_index],)
    for center in path_cells:
        row, column = center
        candidates = (
            tuple((row, column + offset) for offset in range(-half, half + 1)),
            tuple((row + offset, column) for offset in range(-half, half + 1)),
        )
        for route in candidates:
            if _valid_route(route, problem, min_length=route_length):
                if not 0 <= start_index < len(route):
                    raise ValueError("start_index must refer to a route cell.")
                return DynamicObstacleSpec(
                    route=route,
                    start_index=start_index,
                    direction=direction,
                    move_every=move_every,
                    label=label,
                )
    raise ValueError(
        f"Could not construct a free route crossing nominal A* path for {problem.map_id}."
    )


def build_crossing_obstacle_spec_from_path(
    problem: NavigationProblem,
    reference_path: Sequence[Position],
    *,
    rng,
    route_length: int = 5,
    label: str = "crossing",
    move_every: int = 1,
) -> DynamicObstacleSpec:
    """Build a perpendicular crossing route from a frozen greedy path.

    The candidate is selected reproducibly from interior path cells.  A route is
    accepted only when it is free of static obstacles and intersects the reference
    path at the selected cell, so the scenario cannot silently become a parallel
    or unrelated obstacle path.
    """

    if route_length < 3 or route_length % 2 == 0:
        raise ValueError("route_length must be an odd integer of at least three.")
    if len(reference_path) < 3:
        raise ValueError("reference_path must contain at least one interior cell.")
    half = route_length // 2
    path = tuple(reference_path)
    candidates: list[tuple[int, tuple[Position, ...]]] = []
    for index in range(1, len(path) - 1):
        center = path[index]
        previous = path[index - 1]
        following = path[index + 1]
        delta = (following[0] - previous[0], following[1] - previous[1])
        orientations: tuple[str, ...]
        if delta[0] == 0 and delta[1] != 0:
            orientations = ("vertical",)
        elif delta[1] == 0 and delta[0] != 0:
            orientations = ("horizontal",)
        else:
            orientations = ("horizontal", "vertical")
        for orientation in orientations:
            if orientation == "horizontal":
                route = tuple(
                    (center[0], center[1] + offset)
                    for offset in range(-half, half + 1)
                )
            else:
                route = tuple(
                    (center[0] + offset, center[1])
                    for offset in range(-half, half + 1)
                )
            if not _valid_route(route, problem, min_length=route_length):
                continue
            if set(route).intersection(path) != {center}:
                continue
            candidates.append((index, route))
    if not candidates:
        raise ValueError(
            f"Could not construct a perpendicular crossing route for {problem.map_id}."
        )
    path_index, route = rng.choice(candidates)
    start_index = 0
    direction = 1
    return DynamicObstacleSpec(
        route=route,
        start_index=start_index,
        direction=direction,
        move_every=move_every,
        label=label,
        reference_path_source=f"{label}_reference_path",
        reference_path_index=path_index,
    )


_STRATEGY_CROSSING_ROUTES: dict[str, tuple[int, Position, tuple[Position, ...]]] = {
    # The routes are frozen results of one reproducible random draw (seed 8400)
    # from the completed Map 1 static final greedy paths.
    "uniform": (
        63,
        (30, 33),
        ((30, 31), (30, 32), (30, 33), (30, 34), (30, 35)),
    ),
    "prefill": (
        31,
        (21, 12),
        ((21, 10), (21, 11), (21, 12), (21, 13), (21, 14)),
    ),
    "persistent_demo": (
        46,
        (17, 31),
        ((15, 31), (16, 31), (17, 31), (18, 31), (19, 31)),
    ),
}


def build_strategy_crossing_obstacle_spec(
    problem: NavigationProblem,
    strategy: str,
) -> DynamicObstacleSpec:
    """Return the frozen Map 1 crossing scenario for one replay strategy."""

    if problem.map_id != "calibration_40x40_map_01" or problem.size != 40:
        raise ValueError(
            "The strategy crossing calibration scenario is registered only for "
            "calibration_40x40_map_01."
        )
    try:
        path_index, _interaction_cell, route = _STRATEGY_CROSSING_ROUTES[strategy]
    except KeyError as exc:
        raise ValueError(f"Unknown replay strategy: {strategy!r}.") from exc
    if not _valid_route(route, problem, min_length=5):
        raise ValueError(f"Frozen crossing route is invalid for {problem.map_id}.")
    center_index = len(route) // 2
    # Choose a deterministic initial phase so the obstacle occupies the
    # interaction cell when the reference greedy path reaches its selected index.
    for start_index in range(len(route)):
        for direction in (1, -1):
            indices = start_index
            current_direction = direction
            for _ in range(path_index):
                candidate = indices + current_direction
                if candidate >= len(route) or candidate < 0:
                    current_direction *= -1
                    candidate = indices + current_direction
                indices = candidate
            if indices == center_index:
                return DynamicObstacleSpec(
                    route=route,
                    start_index=start_index,
                    direction=direction,
                    move_every=1,
                    label="crossing",
                    reference_path_source=f"{strategy}_final_greedy_path",
                    reference_path_index=path_index,
                )
    raise ValueError(
        f"Could not align crossing phase for {problem.map_id} and {strategy}."
    )


def build_map01_three_crossing_obstacle_specs(
    problem: NavigationProblem,
) -> tuple[DynamicObstacleSpec, ...]:
    """Build the shared Map 1 environment containing all three crossings."""

    return tuple(
        DynamicObstacleSpec(
            route=spec.route,
            start_index=spec.start_index,
            direction=spec.direction,
            move_every=spec.move_every,
            label=f"crossing_{strategy}",
            reference_path_source=spec.reference_path_source,
            reference_path_index=spec.reference_path_index,
        )
        for strategy in ("uniform", "prefill", "persistent_demo")
        for spec in (build_strategy_crossing_obstacle_spec(problem, strategy),)
    )


def _path_exists_without_cell(
    problem: NavigationProblem,
    blocked: Position,
) -> bool:
    if blocked in {problem.start, problem.goal}:
        return False
    visited = {problem.start}
    pending = deque([problem.start])
    while pending:
        current = pending.popleft()
        for delta in ACTION_DELTAS[:4]:
            candidate = (current[0] + delta[0], current[1] + delta[1])
            if (
                candidate == blocked
                or candidate in visited
                or not in_bounds(candidate, problem.size)
                or candidate in problem.obstacles
            ):
                continue
            if candidate == problem.goal:
                return True
            visited.add(candidate)
            pending.append(candidate)
    return False


def _phase_aligned_crossing_spec(
    route: tuple[Position, ...],
    interaction_cell: Position,
    reference_path_index: int,
    label: str,
    alignment_step: int | None = None,
) -> DynamicObstacleSpec:
    interaction_index = route.index(interaction_cell)
    target_step = (
        reference_path_index if alignment_step is None else int(alignment_step)
    )
    if target_step <= 0:
        raise ValueError("Crossing alignment_step must be positive.")
    # The obstacle occupies the interaction cell immediately before target_step.
    # Entering blindly then collides with its old position, while waiting once lets
    # it move far enough away for a safe entry on the following step.
    phase_step = target_step - 1
    for start_index in range(len(route)):
        for direction in (1, -1):
            index = start_index
            current_direction = direction
            for _ in range(phase_step):
                candidate = index + current_direction
                if candidate >= len(route) or candidate < 0:
                    current_direction *= -1
                    candidate = index + current_direction
                index = candidate
            if index == interaction_index:
                return DynamicObstacleSpec(
                    route=route,
                    start_index=start_index,
                    direction=direction,
                    move_every=1,
                    label=label,
                    reference_path_source="static_nominal_path",
                    reference_path_index=reference_path_index,
                )
    raise ValueError(f"Could not align dynamic obstacle phase for {label!r}.")


def build_map01_controlled_bottleneck_obstacle_specs(
    problem: NavigationProblem,
) -> tuple[DynamicObstacleSpec, ...]:
    """Build a strategy-independent scenario with one unavoidable interaction cell."""

    if problem.map_id != "calibration_40x40_map_01" or problem.size != 40:
        raise ValueError(
            "The controlled bottleneck scenario is registered only for "
            "calibration_40x40_map_01."
        )

    bottleneck = (6, 5)
    if bottleneck not in problem.nominal_path:
        raise ValueError("The registered bottleneck is absent from the nominal path.")
    if _path_exists_without_cell(problem, bottleneck):
        raise ValueError("The registered bottleneck is not mandatory on this map.")

    primary = DynamicObstacleSpec(
        route=((6, 5), (6, 6), (6, 7), (6, 8), (6, 9)),
        start_index=0,
        direction=1,
        move_every=1,
        label="mandatory_bottleneck",
        reference_path_source="map_graph_cut",
        reference_path_index=problem.nominal_path.index(bottleneck),
    )
    secondary_routes = (
        (
            "nominal_crossing_middle",
            ((22, 5), (22, 6), (22, 7), (22, 8), (22, 9)),
            (22, 7),
            1,
        ),
        (
            "nominal_crossing_late",
            ((35, 8), (35, 9), (35, 10), (35, 11), (35, 12)),
            (35, 10),
            2,
        ),
    )
    secondary = tuple(
        _phase_aligned_crossing_spec(
            route,
            interaction_cell,
            problem.nominal_path.index(interaction_cell),
            label,
            problem.nominal_path.index(interaction_cell) + preceding_delays,
        )
        for label, route, interaction_cell, preceding_delays in secondary_routes
    )
    specs = (primary, *secondary)
    for spec in specs:
        if not _valid_route(spec.route, problem, min_length=5):
            raise ValueError(f"Controlled route {spec.label!r} is invalid.")
    route_cells = [cell for spec in specs for cell in spec.route]
    if len(route_cells) != len(set(route_cells)):
        raise ValueError("Controlled dynamic obstacle routes must be disjoint.")
    return specs


def build_map01_controlled_six_obstacle_specs(
    problem: NavigationProblem,
) -> tuple[DynamicObstacleSpec, ...]:
    """Combine the mandatory bottleneck with all five Map 1 crossing routes."""

    primary = build_map01_controlled_bottleneck_obstacle_specs(problem)[0]
    crossing_routes = (
        ("nominal_crossing_early", ((10, 3), (10, 4), (10, 5), (10, 6), (10, 7)), (10, 5)),
        ("nominal_crossing_middle", ((22, 5), (22, 6), (22, 7), (22, 8), (22, 9)), (22, 7)),
        ("nominal_crossing_middle_adjacent", ((23, 5), (23, 6), (23, 7), (23, 8), (23, 9)), (23, 7)),
        ("nominal_crossing_late_middle", ((29, 7), (29, 8), (29, 9), (29, 10), (29, 11)), (29, 9)),
        ("nominal_crossing_late", ((35, 8), (35, 9), (35, 10), (35, 11), (35, 12)), (35, 10)),
    )
    crossings = tuple(
        _phase_aligned_crossing_spec(
            route,
            interaction_cell,
            problem.nominal_path.index(interaction_cell),
            label,
            problem.nominal_path.index(interaction_cell) + preceding_interactions,
        )
        for preceding_interactions, (label, route, interaction_cell) in enumerate(
            crossing_routes,
            start=1,
        )
    )
    specs = (primary, *crossings)
    for spec in specs:
        if not _valid_route(spec.route, problem, min_length=5):
            raise ValueError(f"Controlled route {spec.label!r} is invalid.")
    route_cells = [cell for spec in specs for cell in spec.route]
    if len(route_cells) != len(set(route_cells)):
        raise ValueError("Controlled dynamic obstacle routes must be disjoint.")
    return specs


def build_map01_controlled_mixed_six_obstacle_specs(
    problem: NavigationProblem,
) -> tuple[DynamicObstacleSpec, ...]:
    """Combine the three path interactions with three off-path moving obstacles."""

    path_specs = build_map01_controlled_bottleneck_obstacle_specs(problem)
    off_path_specs = (
        DynamicObstacleSpec(
            route=((11, 19), (12, 19), (13, 19), (14, 19), (15, 19)),
            start_index=0,
            direction=1,
            move_every=1,
            label="off_path_west_open_space",
            reference_path_source="off_nominal_open_space",
        ),
        DynamicObstacleSpec(
            route=((24, 32), (24, 33), (24, 34), (24, 35), (24, 36)),
            start_index=4,
            direction=-1,
            move_every=1,
            label="off_path_southeast_open_space",
            reference_path_source="off_nominal_open_space",
        ),
    )
    if any(set(spec.route).intersection(problem.nominal_path) for spec in off_path_specs):
        raise ValueError("Off-path dynamic routes must not intersect the nominal path.")
    terminal_crossing = _phase_aligned_crossing_spec(
        ((35, 19), (36, 19), (37, 19), (38, 19), (39, 19)),
        (38, 19),
        problem.nominal_path.index((38, 19)),
        "nominal_crossing_terminal",
    )
    specs = (*path_specs, *off_path_specs, terminal_crossing)
    route_cells = [cell for spec in specs for cell in spec.route]
    if len(route_cells) != len(set(route_cells)):
        raise ValueError("Mixed dynamic obstacle routes must be disjoint.")
    for spec in off_path_specs:
        if not _valid_route(spec.route, problem, min_length=5):
            raise ValueError(f"Off-path route {spec.label!r} is invalid.")
    if not _valid_route(terminal_crossing.route, problem, min_length=5):
        raise ValueError("Terminal nominal crossing route is invalid.")
    return specs


def build_nominal_path_obstacle_spec(
    problem: NavigationProblem,
    *,
    path_start_index: int,
    path_end_index: int,
    start_index: int,
    direction: int,
    move_every: int = 1,
    label: str = "nominal_path_obstacle",
) -> DynamicObstacleSpec:
    """Create a bouncing obstacle on a fixed segment of the nominal A* path."""

    if not 0 <= path_start_index < path_end_index <= len(problem.nominal_path):
        raise ValueError("Invalid nominal path segment indices.")
    route = tuple(problem.nominal_path[path_start_index:path_end_index])
    if not _valid_route(route, problem, min_length=2):
        raise ValueError("Nominal path segment is not a valid dynamic obstacle route.")
    if not 0 <= start_index < len(route):
        raise ValueError("start_index must refer to a route cell.")
    return DynamicObstacleSpec(
        route=route,
        start_index=start_index,
        direction=direction,
        move_every=move_every,
        label=label,
    )


def build_map01_three_obstacle_specs(
    problem: NavigationProblem,
) -> tuple[DynamicObstacleSpec, ...]:
    """Build the fixed Map 1 crossing, head-on, and slow-following scenario."""

    if problem.map_id != "calibration_40x40_map_01" or problem.size != 40:
        raise ValueError(
            "The three-obstacle calibration scenario is registered only for "
            "calibration_40x40_map_01."
        )
    # These routes are frozen from the completed Map 1 static greedy paths:
    # Uniform supplies the crossing interaction, Prefill the head-on segment,
    # and Persistent Demo the slow same-direction segment.
    crossing = DynamicObstacleSpec(
        route=((5, 13), (6, 13), (7, 13), (8, 13), (9, 13)),
        start_index=0,
        direction=1,
        label="crossing",
        reference_path_source="uniform_final_path",
    )
    head_on = DynamicObstacleSpec(
        route=((31, 17), (31, 18), (31, 19), (31, 20), (31, 21), (31, 22), (31, 23)),
        start_index=6,
        direction=-1,
        label="head_on",
        reference_path_source="prefill_final_path",
    )
    same_direction_slow = DynamicObstacleSpec(
        route=((20, 35), (21, 35), (22, 35), (23, 35), (24, 35), (25, 35), (26, 35), (27, 35), (28, 35)),
        start_index=4,
        direction=1,
        move_every=2,
        label="same_direction_slow",
        reference_path_source="persistent_demo_final_path",
    )
    specs = (crossing, head_on, same_direction_slow)
    for spec in specs:
        if not _valid_route(spec.route, problem, min_length=2):
            raise ValueError(
                f"Configured {spec.label} route is invalid for {problem.map_id}."
            )
    occupied_initial = [spec.route[spec.start_index] for spec in specs]
    if len(occupied_initial) != len(set(occupied_initial)):
        raise ValueError("Calibration dynamic obstacles have duplicate initial cells.")
    return specs


class DynamicGridNavigationEnv:
    current_dynamic_channel = 1

    """Local-observation grid environment with deterministic moving obstacles.

    Spatial channels are static occupancy, current dynamic occupancy, and the two
    preceding dynamic occupancy frames. Dynamic obstacles are deliberately excluded
    from the static problem used by A* demonstrations in the first experiment.
    """

    def __init__(
        self,
        problem: NavigationProblem,
        dynamic_obstacles: Collection[DynamicObstacleSpec] | None = None,
        scenario_id: int | str | None = None,
        max_steps: int = 350,
        reward_config: RewardConfig | None = None,
        terminate_on_collision: bool = False,
        window_size: int | None = 11,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if window_size is None or window_size <= 0 or window_size % 2 == 0:
            raise ValueError("Dynamic environment requires a positive odd window_size.")
        self.problem = problem
        self.scenario_id = scenario_id
        self.max_steps = int(max_steps)
        self.reward_config = reward_config or RewardConfig()
        self.terminate_on_collision = bool(terminate_on_collision)
        self.window_size = int(window_size)
        self.dynamic_obstacles = tuple(dynamic_obstacles or ())
        self._validate_specs()
        self.position: Position = problem.start
        self.steps = 0
        self._indices: list[int] = []
        self._directions: list[int] = []
        self._move_counters: list[int] = []
        self._history: list[tuple[Position, ...]] = []

    def _validate_specs(self) -> None:
        initial = []
        for spec in self.dynamic_obstacles:
            if any(not in_bounds(cell, self.problem.size) for cell in spec.route):
                raise ValueError("Dynamic obstacle route contains an out-of-bounds cell.")
            if any(cell in self.problem.obstacles for cell in spec.route):
                raise ValueError("Dynamic obstacle route intersects a static obstacle.")
            if problem_start_or_goal := {self.problem.start, self.problem.goal}.intersection(spec.route):
                raise ValueError(
                    f"Dynamic obstacle route intersects protected cell {next(iter(problem_start_or_goal))}."
                )
            initial.append(spec.route[spec.start_index])
        if len(initial) != len(set(initial)):
            raise ValueError("Dynamic obstacles cannot share their initial cell.")

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return 4, self.window_size, self.window_size

    @property
    def scalar_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return len(ACTION_DELTAS)

    def set_problem(self, problem: NavigationProblem) -> None:
        self.problem = problem
        self._validate_specs()
        self.reset()

    @property
    def dynamic_positions(self) -> tuple[Position, ...]:
        if not self._indices:
            return tuple()
        return tuple(spec.route[index] for spec, index in zip(self.dynamic_obstacles, self._indices))

    def reset(self) -> Observation:
        self.position = self.problem.start
        self.steps = 0
        self._indices = [spec.start_index for spec in self.dynamic_obstacles]
        self._directions = [spec.direction for spec in self.dynamic_obstacles]
        self._move_counters = [0 for _ in self.dynamic_obstacles]
        current = self.dynamic_positions
        self._history = [current, current, current]
        return self.observation()

    def observation(self) -> Observation:
        radius = self.window_size // 2
        spatial = np.zeros(self.observation_shape, dtype=np.float32)
        for local_row in range(self.window_size):
            map_row = self.position[0] + local_row - radius
            for local_column in range(self.window_size):
                map_column = self.position[1] + local_column - radius
                candidate = (map_row, map_column)
                if not in_bounds(candidate, self.problem.size):
                    spatial[0, local_row, local_column] = 1.0
                    continue
                spatial[0, local_row, local_column] = float(candidate in self.problem.obstacles)
                for channel, frame in enumerate(self._history, start=1):
                    spatial[channel, local_row, local_column] = float(candidate in frame)
        scale = max(1, self.problem.size - 1)
        scalars = np.asarray(
            [
                (self.problem.goal[0] - self.position[0]) / scale,
                (self.problem.goal[1] - self.position[1]) / scale,
            ],
            dtype=np.float32,
        )
        return Observation(spatial=spatial, scalars=scalars)

    def action_mask(self, mask_collisions: bool = False) -> np.ndarray:
        """Return actions valid against known static geometry.

        Dynamic positions are intentionally absent from this mask so collision
        prediction, waiting, and avoidance remain decisions learned by the agent.
        """
        if not mask_collisions:
            return np.ones(self.action_dim, dtype=bool)
        mask = np.zeros(self.action_dim, dtype=bool)
        for action in Action:
            candidate = move(self.position, action)
            mask[int(action)] = in_bounds(candidate, self.problem.size) and candidate not in self.problem.obstacles
        return mask

    def _next_dynamic_state(
        self,
    ) -> tuple[list[int], list[int], list[int], tuple[Position, ...]]:
        indices = list(self._indices)
        directions = list(self._directions)
        move_counters = list(self._move_counters)
        for offset, spec in enumerate(self.dynamic_obstacles):
            move_counters[offset] += 1
            if move_counters[offset] < spec.move_every:
                continue
            move_counters[offset] = 0
            candidate = indices[offset] + directions[offset]
            if candidate >= len(spec.route) or candidate < 0:
                directions[offset] *= -1
                candidate = indices[offset] + directions[offset]
            indices[offset] = candidate
        return indices, directions, move_counters, tuple(
            spec.route[index] for spec, index in zip(self.dynamic_obstacles, indices)
        )

    def dynamic_action_collision_risk(
        self,
        action: int,
        *,
        predict_next: bool = True,
    ) -> bool:
        """Return whether an action conflicts with current/predicted dynamics.

        ``predict_next=True`` exactly matches the dynamic part of ``step``:
        obstacles move before the agent and both their old and proposed cells are
        protected.  ``False`` is a deliberately weaker current-occupancy-only
        signal used as an ablation for conflict-adaptive demonstration replay.
        """

        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"Invalid action index: {action}")
        candidate = move(self.position, int(action))
        old_dynamic = self.dynamic_positions
        if not predict_next:
            return candidate in old_dynamic
        proposed_dynamic = self._next_dynamic_state()[3]
        agent_stays = candidate == self.position
        return any(
            candidate == old_cell
            or candidate == next_cell
            or (agent_stays and self.position == next_cell)
            or (candidate == old_cell and next_cell == self.position)
            for old_cell, next_cell in zip(old_dynamic, proposed_dynamic)
        )

    def step(self, action: int) -> StepResult:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"Invalid action index: {action}")
        action = int(action)
        old_position = self.position
        old_dynamic = self.dynamic_positions
        (
            next_indices,
            next_directions,
            next_move_counters,
            proposed_dynamic,
        ) = self._next_dynamic_state()
        candidate = move(old_position, action)
        static_collision = not in_bounds(candidate, self.problem.size) or candidate in self.problem.obstacles
        agent_stays = candidate == old_position
        dynamic_collisions = {
            index
            for index, (old_cell, next_cell) in enumerate(zip(old_dynamic, proposed_dynamic))
            if candidate == old_cell
            or candidate == next_cell
            or (agent_stays and old_position == next_cell)
            or (candidate == old_cell and next_cell == old_position)
        }
        dynamic_collision = bool(dynamic_collisions)
        collision = static_collision or dynamic_collision
        if not static_collision and not dynamic_collision:
            self.position = candidate
        self._indices = next_indices
        self._directions = next_directions
        self._move_counters = next_move_counters
        self._history = [self.dynamic_positions, old_dynamic, self._history[1]]
        self.steps += 1
        reached = not collision and self.position == self.problem.goal
        old_distance = manhattan(old_position, self.problem.goal)
        new_distance = manhattan(self.position, self.problem.goal)
        # Terminal events are exclusive. Progress and waiting costs apply only to
        # ordinary transitions, so collision/goal rewards cannot be double counted.
        if reached:
            components = {
                "step": 0.0,
                "progress": 0.0,
                "stay": 0.0,
                "collision": 0.0,
                "goal": self.reward_config.goal,
            }
        elif collision:
            components = {
                "step": 0.0,
                "progress": 0.0,
                "stay": 0.0,
                "collision": self.reward_config.collision,
                "goal": 0.0,
            }
        else:
            components = {
                "step": self.reward_config.step,
                "progress": self.reward_config.progress * (old_distance - new_distance),
                "stay": self.reward_config.stay if action == int(Action.STAY) else 0.0,
                "collision": 0.0,
                "goal": 0.0,
            }
        terminated = reached or (collision and self.terminate_on_collision)
        truncated = not terminated and self.steps >= self.max_steps
        info = {
            "position": self.position,
            "scenario_id": self.scenario_id,
            "dynamic_positions": self.dynamic_positions,
            "collision_position": candidate if collision and in_bounds(candidate, self.problem.size) else None,
            "collision_type": "static" if static_collision else "dynamic" if dynamic_collision else None,
            "dynamic_collision_indices": tuple(sorted(dynamic_collisions)),
            "steps": self.steps,
            "collision": collision,
            "reached": reached,
            "termination_reason": (
                "goal" if reached else "collision" if terminated else "timeout" if truncated else "running"
            ),
            **{f"reward_{key}": value for key, value in components.items()},
        }
        return StepResult(
            observation=self.observation(),
            reward=float(sum(components.values())),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
