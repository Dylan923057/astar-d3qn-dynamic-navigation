"""Episode-level sampling from a frozen whole-map dynamic-route candidate pool."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from astar_d3qn.core.grid import manhattan
from astar_d3qn.maps.problem import NavigationProblem

from .dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec


POOL_CATEGORIES = (
    "high_interaction",
    "alternative_branch",
    "background",
)


class WholeMapRoutePoolEnvironmentFactory:
    """Sample one accepted obstacle layout when each episode environment is made.

    A returned environment owns immutable route specifications. Calling ``reset``
    replays those routes from their sampled phases; only the next factory call
    samples a new layout. This matches the trainer's one-environment-per-episode
    lifecycle and prevents within-episode refreshes or teleports.
    """

    def __init__(
        self,
        problem: NavigationProblem,
        map_entry: Mapping[str, Any],
        *,
        composition: Mapping[str, int],
        seed: int,
        acceptance: Mapping[str, Any],
        move_every: int = 1,
        sampling_trials: int = 1000,
        scenario_prefix: str = "whole_map_pool_v2",
        obstacle_counts: tuple[int, ...] | None = None,
    ) -> None:
        reference = map_entry.get("problem_reference", {})
        if reference.get("map_id") != problem.map_id:
            raise ValueError("Route-pool map ID does not match the training problem.")
        if reference.get("grid_sha256") != problem.grid_sha256:
            raise ValueError("Route-pool grid hash does not match the training problem.")
        if tuple(reference.get("start", ())) != problem.start:
            raise ValueError("Route-pool start does not match the training problem.")
        if tuple(reference.get("goal", ())) != problem.goal:
            raise ValueError("Route-pool goal does not match the training problem.")
        if move_every <= 0 or sampling_trials <= 0:
            raise ValueError("move_every and sampling_trials must be positive.")

        route_pool = map_entry.get("route_pool", {})
        self._route_pool: dict[str, tuple[Mapping[str, Any], ...]] = {}
        self.composition = {key: int(composition.get(key, 0)) for key in POOL_CATEGORIES}
        for category in POOL_CATEGORIES:
            records = tuple(route_pool.get(category, ()))
            if self.composition[category] <= 0:
                raise ValueError(f"Composition for {category} must be positive.")
            if len(records) < self.composition[category]:
                raise ValueError(f"Insufficient {category} candidates in route pool.")
            if any(record.get("category") != category for record in records):
                raise ValueError(f"Candidate category mismatch in {category} pool.")
            self._route_pool[category] = records

        self.problem = problem
        self.seed = int(seed)
        self.acceptance = dict(acceptance)
        self.move_every = int(move_every)
        self.sampling_trials = int(sampling_trials)
        self.scenario_prefix = str(scenario_prefix)
        self._rng = random.Random(self.seed)
        self._scenario_index = 0
        if obstacle_counts is not None and tuple(obstacle_counts) != (3, 4, 5):
            raise ValueError("Variable obstacle counts must be (3, 4, 5).")
        self.obstacle_counts = obstacle_counts
        self._count_rng = random.Random(self.seed)

    def reset_schedule(self) -> None:
        """Undo the trainer's shape-probe draw before the first real episode."""

        self._rng = random.Random(self.seed)
        self._scenario_index = 0
        self._count_rng = random.Random(self.seed)

    @staticmethod
    def _cells(record: Mapping[str, Any]) -> set[tuple[int, int]]:
        return {tuple(int(value) for value in cell) for cell in record["route"]}

    @staticmethod
    def _center(record: Mapping[str, Any]) -> tuple[int, int]:
        return tuple(int(value) for value in record["center"])

    def _accepted(self, selected: list[Mapping[str, Any]]) -> bool:
        minimum_gap = int(self.acceptance["minimum_center_manhattan_gap"])
        minimum_span = int(self.acceptance["minimum_spatial_span"])
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                if self._cells(left).intersection(self._cells(right)):
                    return False
                if manhattan(self._center(left), self._center(right)) < minimum_gap:
                    return False
        if len({record["spatial_region"] for record in selected}) < int(
            self.acceptance["minimum_spatial_regions"]
        ):
            return False
        span = max(
            manhattan(self._center(left), self._center(right))
            for index, left in enumerate(selected)
            for right in selected[index + 1 :]
        )
        if span < minimum_span:
            return False
        if sum(
            float(record["minimum_distance_to_nominal_astar"])
            >= float(self.acceptance["far_from_nominal_distance"])
            for record in selected
        ) < int(self.acceptance["minimum_far_from_nominal_routes"]):
            return False
        hi = [record for record in selected if record["category"] == "high_interaction"]
        ar = [record for record in selected if record["category"] == "alternative_branch"]
        if not any(
            float(record["ensemble_intersection_fraction"])
            >= float(self.acceptance["high_support_path_fraction"])
            for record in hi
        ):
            return False
        if len({record["interaction_progress_band"] for record in hi}) < min(2, len(hi)):
            return False
        if not any(
            int(record["alternative_path_support_count"]) > 0
            and not bool(record["intersects_nominal_astar"])
            for record in ar
        ):
            return False
        covered_paths = set().union(
            *(set(record["ensemble_path_indices"]) for record in selected)
        )
        path_count = int(self.acceptance.get("reasonable_path_count", 240))
        return len(covered_paths) / path_count >= float(
            self.acceptance["minimum_scene_path_coverage"]
        )

    def _sample_records(self) -> list[Mapping[str, Any]]:
        composition = self.composition
        if self.obstacle_counts is not None:
            count = self._count_rng.choice(self.obstacle_counts)
            hi, ar = (1, 1) if count == 3 else (2, 2) if count == 5 else self._count_rng.choice(((2, 1), (1, 2)))
            composition = dict(zip(POOL_CATEGORIES, (hi, ar, 1)))
        for _ in range(self.sampling_trials):
            selected = [
                record
                for category in POOL_CATEGORIES
                for record in self._rng.sample(
                    self._route_pool[category], composition[category]
                )
            ]
            if self._accepted(selected):
                return selected
        raise RuntimeError(
            "Could not sample an accepted whole-map route-pool scene within "
            f"{self.sampling_trials} trials."
        )

    def __call__(self, problem: NavigationProblem, **environment_kwargs):
        if problem.map_id != self.problem.map_id or problem.grid_sha256 != self.problem.grid_sha256:
            raise ValueError("Whole-map route-pool factory received an unexpected map.")
        selected = self._sample_records()
        specs = tuple(
            DynamicObstacleSpec(
                route=tuple(
                    tuple(int(value) for value in cell) for cell in record["route"]
                ),
                start_index=self._rng.randrange(len(record["route"])),
                direction=self._rng.choice((-1, 1)),
                move_every=self.move_every,
                label=str(record["route_id"]),
                reference_path_source="whole_map_reasonable_path_ensemble_v2",
            )
            for record in selected
        )
        self._scenario_index += 1
        scenario_id = (
            f"{self.scenario_prefix}_seed{self.seed}_"
            f"episode{self._scenario_index:06d}"
        )
        env = DynamicGridNavigationEnv(
            problem,
            dynamic_obstacles=specs,
            scenario_id=scenario_id,
            **environment_kwargs,
        )
        env.dynamic_route_ids = tuple(str(record["route_id"]) for record in selected)
        env.dynamic_route_categories = tuple(
            str(record["category"]) for record in selected
        )
        env.dynamic_route_spatial_regions = tuple(
            str(record["spatial_region"]) for record in selected
        )
        return env
