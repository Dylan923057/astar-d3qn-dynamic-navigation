"""Reproducible dynamic-obstacle scenarios and train/eval scheduling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from astar_d3qn.maps.problem import NavigationProblem

from .dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec, _valid_route


@dataclass(frozen=True, slots=True)
class DynamicScenario:
    """One deterministic dynamic-obstacle configuration for one episode."""

    seed: int
    obstacles: tuple[DynamicObstacleSpec, ...]
    required_behavior: str | None = None
    difficulty_stratum: str | None = None


@dataclass(frozen=True, slots=True)
class DynamicObstacleCurriculumStage:
    """One fixed environment-step stage of a dynamic-obstacle curriculum."""

    name: str
    start_environment_step: int
    obstacle_count: int
    obstacle_indices: tuple[int, ...] | None = None
    allowed_difficulties: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Curriculum stage name cannot be empty.")
        if self.start_environment_step < 0:
            raise ValueError("Curriculum stage start step cannot be negative.")
        if self.obstacle_count < 0:
            raise ValueError("Curriculum obstacle count cannot be negative.")
        if self.obstacle_indices is not None:
            if len(self.obstacle_indices) != self.obstacle_count:
                raise ValueError(
                    "Curriculum obstacle_indices must match obstacle_count."
                )
            if len(set(self.obstacle_indices)) != len(self.obstacle_indices):
                raise ValueError("Curriculum obstacle indices must be unique.")
            if any(index < 0 for index in self.obstacle_indices):
                raise ValueError("Curriculum obstacle indices cannot be negative.")
        if self.allowed_difficulties is not None:
            if not self.allowed_difficulties or any(
                not value.strip() for value in self.allowed_difficulties
            ):
                raise ValueError(
                    "Curriculum allowed_difficulties cannot be empty."
                )
            if len(set(self.allowed_difficulties)) != len(
                self.allowed_difficulties
            ):
                raise ValueError(
                    "Curriculum allowed difficulties must be unique."
                )


def _crossing_candidates(
    problem: NavigationProblem,
    route_length: int,
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    if route_length < 3 or route_length % 2 == 0:
        raise ValueError("route_length must be an odd integer of at least three.")
    half = route_length // 2
    path = tuple(problem.nominal_path)
    candidates: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for path_index in range(1, len(path) - 1):
        center = path[path_index]
        previous = path[path_index - 1]
        following = path[path_index + 1]
        delta = (following[0] - previous[0], following[1] - previous[1])
        orientations = ("vertical",) if delta[0] == 0 else ("horizontal",)
        if delta[0] != 0 and delta[1] != 0:
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
            candidates.append((path_index, route))
    return candidates


def build_random_crossing_scenario(
    problem: NavigationProblem,
    seed: int,
    *,
    obstacle_count: int = 3,
    route_length: int = 5,
    move_every: int = 1,
) -> DynamicScenario:
    """Build one strategy-independent scenario from a stable seed.

    Routes are sampled from the static nominal path before training starts. The
    replay strategy and training seed are never consulted, which prevents the
    dynamic test distribution from depending on a learned policy.
    """
    if obstacle_count <= 0:
        raise ValueError("obstacle_count must be positive.")
    rng = random.Random(int(seed))
    candidates = _crossing_candidates(problem, route_length)
    rng.shuffle(candidates)
    selected: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    occupied: set[tuple[int, int]] = set()
    for path_index, route in candidates:
        if occupied.intersection(route):
            continue
        selected.append((path_index, route))
        occupied.update(route)
        if len(selected) == obstacle_count:
            break
    if len(selected) != obstacle_count:
        raise ValueError(
            f"Could not build {obstacle_count} disjoint crossing routes for {problem.map_id}."
        )
    specs = tuple(
        DynamicObstacleSpec(
            route=route,
            start_index=rng.randrange(len(route)),
            direction=rng.choice((-1, 1)),
            move_every=move_every,
            label=f"random_crossing_{index}",
            reference_path_source="nominal_static_path",
            reference_path_index=path_index,
        )
        for index, (path_index, route) in enumerate(selected)
    )
    return DynamicScenario(seed=int(seed), obstacles=specs)


def build_scenario_schedule(
    problem: NavigationProblem,
    seeds: Iterable[int],
    *,
    obstacle_count: int,
    route_length: int,
    move_every: int,
) -> tuple[DynamicScenario, ...]:
    return tuple(
        build_random_crossing_scenario(
            problem,
            int(seed),
            obstacle_count=obstacle_count,
            route_length=route_length,
            move_every=move_every,
        )
        for seed in seeds
    )


class ScheduledDynamicEnvironmentFactory:
    """Callable environment factory with deterministic train/eval schedules."""

    def __init__(
        self,
        train_scenarios: Sequence[DynamicScenario],
        eval_scenarios: Sequence[DynamicScenario],
        *,
        shuffle_train: bool = False,
        seed: int = 0,
    ) -> None:
        if not train_scenarios or not eval_scenarios:
            raise ValueError("Both train and eval scenario schedules are required.")
        self._schedules = {"train": tuple(train_scenarios), "eval": tuple(eval_scenarios)}
        self._mode: Literal["train", "eval"] = "train"
        self._shuffle_train = bool(shuffle_train)
        self._schedule_seed = int(seed)
        self._train_rng = random.Random(self._schedule_seed)
        self._train_orders: dict[tuple[int, ...], list[int]] = {}
        self._train_cursors: dict[tuple[int, ...], int] = {}
        self._cursor = 0

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: Literal["train", "eval"]) -> None:
        if mode not in self._schedules:
            raise ValueError(f"Unknown dynamic schedule mode: {mode!r}.")
        self._mode = mode
        self.reset_schedule()

    def reset_schedule(self) -> None:
        self._cursor = 0
        self._train_rng = random.Random(self._schedule_seed)
        self._train_orders = {}
        self._train_cursors = {}

    def _next_scenario(
        self, schedule: Sequence[DynamicScenario]
    ) -> DynamicScenario:
        if self._mode != "train" or not self._shuffle_train:
            scenario = schedule[self._cursor % len(schedule)]
            self._cursor += 1
            return scenario

        schedule_key = tuple(scenario.seed for scenario in schedule)
        if schedule_key not in self._train_orders:
            order = list(range(len(schedule)))
            self._train_rng.shuffle(order)
            self._train_orders[schedule_key] = order
            self._train_cursors[schedule_key] = 0
        order = self._train_orders[schedule_key]
        cursor = self._train_cursors[schedule_key]
        if cursor >= len(order):
            self._train_rng.shuffle(order)
            cursor = 0
        scenario = schedule[order[cursor]]
        self._train_cursors[schedule_key] = cursor + 1
        return scenario

    def __call__(self, problem: NavigationProblem, **kwargs):
        schedule = self._schedules[self._mode]
        scenario = self._next_scenario(schedule)
        env = DynamicGridNavigationEnv(
            problem,
            dynamic_obstacles=scenario.obstacles,
            scenario_id=scenario.seed,
            **kwargs,
        )
        env.required_behavior = scenario.required_behavior
        env.difficulty_stratum = scenario.difficulty_stratum
        return env

    def scenario_manifest(self) -> dict[str, list[dict]]:
        return {
            mode: [
                {
                    "seed": scenario.seed,
                    "obstacles": [
                        {
                            "route": [list(cell) for cell in spec.route],
                            "start_index": spec.start_index,
                            "direction": spec.direction,
                            "move_every": spec.move_every,
                            "reference_path_index": spec.reference_path_index,
                        }
                        for spec in scenario.obstacles
                    ],
                }
                for scenario in scenarios
            ]
            for mode, scenarios in self._schedules.items()
        }


class CurriculumScheduledDynamicEnvironmentFactory(
    ScheduledDynamicEnvironmentFactory
):
    """Activate selected obstacles at fixed environment-step boundaries.

    A stage can name exact obstacle indices, which lets an experiment order
    blockers by measured difficulty instead of their position in a manifest.
    Omitting indices preserves the legacy prefix behavior. Evaluation always
    uses every obstacle in the held-out scenario.
    """

    def __init__(
        self,
        train_scenarios: Sequence[DynamicScenario],
        eval_scenarios: Sequence[DynamicScenario],
        stages: Sequence[DynamicObstacleCurriculumStage],
        *,
        shuffle_train: bool = False,
        seed: int = 0,
        manual_stage_control: bool = False,
        rehearsal_probabilities: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(
            train_scenarios,
            eval_scenarios,
            shuffle_train=shuffle_train,
            seed=seed,
        )
        if not stages:
            raise ValueError(
                "At least one dynamic-obstacle curriculum stage is required."
            )
        self._curriculum_stages = tuple(stages)
        starts = [stage.start_environment_step for stage in self._curriculum_stages]
        counts = [stage.obstacle_count for stage in self._curriculum_stages]
        names = [stage.name for stage in self._curriculum_stages]
        if starts[0] != 0:
            raise ValueError(
                "The first curriculum stage must start at environment step 0."
            )
        if any(right <= left for left, right in zip(starts, starts[1:])):
            raise ValueError("Curriculum stage start steps must be strictly increasing.")
        if any(right < left for left, right in zip(counts, counts[1:])):
            raise ValueError("Curriculum obstacle counts cannot decrease.")
        if len(set(names)) != len(names):
            raise ValueError("Curriculum stage names must be unique.")
        available_counts = {len(scenario.obstacles) for scenario in train_scenarios}
        if len(available_counts) != 1:
            raise ValueError(
                "Every curriculum training scenario must have the same obstacle count."
            )
        available = next(iter(available_counts))
        if counts[-1] != available:
            raise ValueError(
                "The final curriculum stage must restore every training obstacle."
            )
        if any(count > available for count in counts):
            raise ValueError(
                "Curriculum requests more obstacles than a scenario contains."
            )
        for stage in self._curriculum_stages:
            if stage.obstacle_indices is not None and any(
                index >= available for index in stage.obstacle_indices
            ):
                raise ValueError(
                    "Curriculum obstacle index exceeds the scenario obstacle count."
                )
        final_indices = self._curriculum_stages[-1].obstacle_indices
        if final_indices is not None and set(final_indices) != set(range(available)):
            raise ValueError(
                "The final curriculum stage must restore every obstacle index."
            )
        self._environment_steps = 0
        self._manual_stage_control = bool(manual_stage_control)
        self._manual_stage_index = 0
        self._manual_stage_start_environment_step = 0
        self._rehearsal_seed = int(seed) + 700_001
        self._rehearsal_rng = random.Random(self._rehearsal_seed)
        if rehearsal_probabilities is None:
            self._rehearsal_probabilities = None
        else:
            matrix = tuple(
                tuple(float(value) for value in row)
                for row in rehearsal_probabilities
            )
            if len(matrix) != len(self._curriculum_stages) or any(
                len(row) != len(self._curriculum_stages) for row in matrix
            ):
                raise ValueError(
                    "Curriculum rehearsal probabilities must form one square "
                    "row per stage."
                )
            for active_index, row in enumerate(matrix):
                if any(value < 0.0 for value in row) or abs(sum(row) - 1.0) > 1e-9:
                    raise ValueError(
                        "Each curriculum rehearsal row must be non-negative and sum to one."
                    )
                if any(row[index] > 0.0 for index in range(active_index + 1, len(row))):
                    raise ValueError(
                        "A curriculum stage cannot rehearse a future stage."
                    )
            self._rehearsal_probabilities = matrix

    @property
    def curriculum_stages(self) -> tuple[DynamicObstacleCurriculumStage, ...]:
        return self._curriculum_stages

    @property
    def manual_stage_control(self) -> bool:
        return self._manual_stage_control

    @property
    def current_stage_index(self) -> int:
        if self._manual_stage_control:
            return self._manual_stage_index
        active = self._active_curriculum_stage()
        return self._curriculum_stages.index(active)

    @property
    def current_stage_start_environment_step(self) -> int:
        if self._manual_stage_control:
            return self._manual_stage_start_environment_step
        return self._active_curriculum_stage().start_environment_step

    def advance_curriculum_stage(self, environment_steps: int) -> bool:
        if not self._manual_stage_control:
            raise ValueError("Manual curriculum advancement is not enabled.")
        if self._manual_stage_index >= len(self._curriculum_stages) - 1:
            return False
        self._manual_stage_index += 1
        self._manual_stage_start_environment_step = int(environment_steps)
        return True

    def reset_schedule(self) -> None:
        super().reset_schedule()
        if hasattr(self, "_rehearsal_seed"):
            self._rehearsal_rng = random.Random(self._rehearsal_seed)

    def set_environment_steps(self, environment_steps: int) -> None:
        if environment_steps < 0:
            raise ValueError("Environment steps cannot be negative.")
        self._environment_steps = int(environment_steps)

    def _active_curriculum_stage(self) -> DynamicObstacleCurriculumStage:
        if self._manual_stage_control:
            return self._curriculum_stages[self._manual_stage_index]
        active = self._curriculum_stages[0]
        for stage in self._curriculum_stages[1:]:
            if self._environment_steps < stage.start_environment_step:
                break
            active = stage
        return active

    def _effective_train_stage(
        self, active: DynamicObstacleCurriculumStage
    ) -> tuple[DynamicObstacleCurriculumStage, bool]:
        if self._rehearsal_probabilities is None:
            return active, False
        active_index = self._curriculum_stages.index(active)
        weights = self._rehearsal_probabilities[active_index]
        draw = self._rehearsal_rng.random()
        cumulative = 0.0
        chosen_index = active_index
        for index, weight in enumerate(weights):
            cumulative += weight
            if draw <= cumulative:
                chosen_index = index
                break
        return self._curriculum_stages[chosen_index], chosen_index != active_index

    def __call__(self, problem: NavigationProblem, **kwargs):
        schedule = self._schedules[self._mode]
        active_stage = (
            self._active_curriculum_stage() if self._mode == "train" else None
        )
        stage = active_stage
        is_rehearsal = False
        if active_stage is not None:
            stage, is_rehearsal = self._effective_train_stage(active_stage)
        if stage is not None and stage.allowed_difficulties is not None:
            allowed = set(stage.allowed_difficulties)
            schedule = tuple(
                scenario
                for scenario in schedule
                if scenario.difficulty_stratum in allowed
            )
            if not schedule:
                raise ValueError(
                    f"Curriculum stage {stage.name!r} has no eligible scenarios."
                )
        scenario = self._next_scenario(schedule)
        if stage is None:
            obstacles = scenario.obstacles
        elif stage.obstacle_indices is None:
            obstacles = scenario.obstacles[: stage.obstacle_count]
        else:
            obstacles = tuple(
                scenario.obstacles[index] for index in stage.obstacle_indices
            )
        env = DynamicGridNavigationEnv(
            problem,
            dynamic_obstacles=obstacles,
            scenario_id=scenario.seed,
            **kwargs,
        )
        env.curriculum_stage = stage.name if stage is not None else "full_evaluation"
        env.curriculum_active_stage = (
            active_stage.name if active_stage is not None else "full_evaluation"
        )
        env.curriculum_rehearsal = is_rehearsal
        env.curriculum_stage_start_environment_step = (
            self.current_stage_start_environment_step
            if active_stage is not None
            else None
        )
        env.required_behavior = scenario.required_behavior
        env.difficulty_stratum = scenario.difficulty_stratum
        return env
