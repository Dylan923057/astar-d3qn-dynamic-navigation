from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from astar_d3qn.core.astar import astar_path, randomized_tie_astar_path
from astar_d3qn.core.grid import (
    Position,
    chebyshev,
    in_bounds,
    manhattan,
    planner_neighbors,
)

from .problem import NavigationProblem


def _rectangle(top: int, left: int, bottom: int, right: int) -> set[Position]:
    if top > bottom or left > right:
        raise ValueError("Rectangle bounds must be ordered.")
    return {
        (row, column)
        for row in range(top, bottom + 1)
        for column in range(left, right + 1)
    }


def _remove_cells(cells: set[Position], openings: Iterable[Position]) -> None:
    cells.difference_update(openings)


def _obstacle_component_sizes(
    obstacles: Iterable[Position], size: int
) -> list[int]:
    remaining = set(obstacles)
    sizes: list[int] = []
    while remaining:
        queue = deque([remaining.pop()])
        component_size = 1
        while queue:
            current = queue.popleft()
            for neighbor in planner_neighbors(current, size):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component_size += 1
        sizes.append(component_size)
    return sizes


def _obstacle_components(obstacles: Iterable[Position], size: int) -> int:
    return len(_obstacle_component_sizes(obstacles, size))


def _turn_count(path: tuple[Position, ...]) -> int:
    if len(path) < 3:
        return 0
    directions = [
        (right[0] - left[0], right[1] - left[1])
        for left, right in zip(path, path[1:])
    ]
    return sum(left != right for left, right in zip(directions, directions[1:]))


def _sample_optimal_path_diversity(
    start: Position,
    goal: Position,
    obstacles: Iterable[Position],
    size: int,
    samples: int = 100,
) -> tuple[int, int]:
    paths = {
        tuple(
            randomized_tie_astar_path(
                start,
                goal,
                obstacles,
                size,
                random.Random(seed),
            )
            or ()
        )
        for seed in range(samples)
    }
    paths.discard(())
    union = set().union(*paths) if paths else set()
    return len(paths), len(union)


def _random_scattered_calibration_obstacles(
    seed: int,
    *,
    size: int,
    start: Position,
    goal: Position,
    target_count: int,
    endpoint_clearance: int = 1,
    min_detour_steps: int = 0,
    max_detour_steps: int = 10**9,
    min_turns: int = 6,
    max_component_size: int = 9,
    min_path_variants: int = 80,
    min_path_union_cells: int = 100,
    shape_profile: str = "small",
    min_quadrant_obstacles: int = 20,
    max_turns: int | None = None,
    spatial_bins: int = 1,
    min_bin_obstacles: int | None = None,
    max_bin_obstacles: int | None = None,
) -> tuple[set[Position], int, int, int]:
    small_shapes: tuple[tuple[Position, ...], ...] = (
        ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
        ((0, -1), (0, 0), (0, 1), (1, 0)),
        ((-1, 0), (0, -1), (0, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0), (0, -1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0),),
    )
    large_shapes: tuple[tuple[Position, ...], ...] = (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        tuple((row, column) for row in range(3) for column in range(3)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0),),
    )
    if shape_profile == "small":
        shapes = small_shapes
    elif shape_profile == "large":
        shapes = large_shapes
    else:
        raise ValueError("shape_profile must be 'small' or 'large'.")
    if spatial_bins < 1 or size % spatial_bins != 0:
        raise ValueError("spatial_bins must be a positive divisor of size.")
    if (
        min_bin_obstacles is not None
        and max_bin_obstacles is not None
        and min_bin_obstacles > max_bin_obstacles
    ):
        raise ValueError("min_bin_obstacles cannot exceed max_bin_obstacles.")

    def bin_index(cell: Position) -> tuple[int, int]:
        return (
            min(spatial_bins - 1, cell[0] * spatial_bins // size),
            min(spatial_bins - 1, cell[1] * spatial_bins // size),
        )

    def bin_counts(cells: Iterable[Position]) -> list[int]:
        counts = [0] * (spatial_bins * spatial_bins)
        for row, column in cells:
            bin_row, bin_column = bin_index((row, column))
            counts[bin_row * spatial_bins + bin_column] += 1
        return counts
    if endpoint_clearance < 0:
        raise ValueError("endpoint_clearance cannot be negative.")
    protected = {
        cell
        for row in range(size)
        for column in range(size)
        if chebyshev((row, column), start) <= endpoint_clearance
        or chebyshev((row, column), goal) <= endpoint_clearance
        for cell in ((row, column),)
    }
    for generation_attempt in range(1, 1001):
        rng = random.Random(seed * 1009 + generation_attempt)
        obstacles: set[Position] = set()
        placement_attempts = 0
        while len(obstacles) < target_count and placement_attempts < 20000:
            placement_attempts += 1
            shape = rng.choice(shapes)
            anchor = (rng.randrange(size), rng.randrange(size))
            candidate = {
                (anchor[0] + row, anchor[1] + column) for row, column in shape
            }
            if (
                not candidate
                or any(not in_bounds(cell, size) for cell in candidate)
                or candidate.intersection(protected)
                or candidate.issubset(obstacles)
            ):
                continue
            proposed = obstacles | candidate
            if len(proposed) > target_count:
                continue
            if max_bin_obstacles is not None and max(
                bin_counts(proposed)
            ) > max_bin_obstacles:
                continue
            component_sizes = _obstacle_component_sizes(proposed, size)
            if component_sizes and max(component_sizes) > max_component_size:
                continue
            obstacles = proposed
        if len(obstacles) != target_count:
            continue
        quadrant_counts = (
            sum(row < size // 2 and column < size // 2 for row, column in obstacles),
            sum(row < size // 2 and column >= size // 2 for row, column in obstacles),
            sum(row >= size // 2 and column < size // 2 for row, column in obstacles),
            sum(row >= size // 2 and column >= size // 2 for row, column in obstacles),
        )
        if min(quadrant_counts) < min_quadrant_obstacles:
            continue
        spatial_counts = bin_counts(obstacles)
        if min_bin_obstacles is not None and min(spatial_counts) < min_bin_obstacles:
            continue
        if max_bin_obstacles is not None and max(spatial_counts) > max_bin_obstacles:
            continue
        path = astar_path(start, goal, obstacles, size)
        if path is None:
            continue
        path_steps = len(path) - 1
        detour_steps = path_steps - manhattan(start, goal)
        turn_count = _turn_count(tuple(path))
        if (
            detour_steps < min_detour_steps
            or detour_steps > max_detour_steps
            or turn_count < min_turns
            or (max_turns is not None and turn_count > max_turns)
        ):
            continue
        path_variants, path_union_cells = _sample_optimal_path_diversity(
            start, goal, obstacles, size
        )
        if (
            path_variants < min_path_variants
            or path_union_cells < min_path_union_cells
        ):
            continue
        return obstacles, generation_attempt, path_variants, path_union_cells
    raise RuntimeError(
        f"Unable to generate calibration map seed={seed} with the registered gates."
    )


def _uniform_scattered_calibration_obstacles(
    seed: int,
    *,
    size: int,
    start: Position,
    goal: Position,
    target_count: int,
    endpoint_clearance: int = 1,
    max_component_size: int = 24,
    min_turns: int = 10,
    max_turns: int = 30,
    min_path_variants: int = 50,
    min_path_union_cells: int = 180,
) -> tuple[set[Position], int, int, int, tuple[int, ...]]:
    """Place mixed shapes independently in 10x10 bins for even occupancy."""
    if size % 4 != 0:
        raise ValueError("Uniform calibration layout requires a size divisible by 4.")
    if target_count < 16:
        raise ValueError("Uniform calibration layout needs at least one cell per bin.")
    shapes: tuple[tuple[Position, ...], ...] = (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        tuple((row, column) for row in range(3) for column in range(3)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0),),
    )
    bins = 4
    bin_size = size // bins
    base_count, remainder = divmod(target_count, bins * bins)
    target_counts = [base_count + int(index < remainder) for index in range(bins * bins)]
    protected = {
        (row, column)
        for row in range(size)
        for column in range(size)
        if chebyshev((row, column), start) <= endpoint_clearance
        or chebyshev((row, column), goal) <= endpoint_clearance
    }
    for generation_attempt in range(1, 1001):
        rng = random.Random(seed * 1009 + generation_attempt)
        shuffled_targets = target_counts[:]
        rng.shuffle(shuffled_targets)
        obstacles: set[Position] = set()
        success = True
        for bin_index in rng.sample(range(bins * bins), bins * bins):
            bin_row, bin_column = divmod(bin_index, bins)
            row_start, row_end = bin_row * bin_size, (bin_row + 1) * bin_size
            column_start, column_end = bin_column * bin_size, (bin_column + 1) * bin_size
            target_bin_count = shuffled_targets[bin_index]
            local_obstacles: set[Position] = set()
            placement_attempts = 0
            while len(local_obstacles) < target_bin_count and placement_attempts < 3000:
                placement_attempts += 1
                shape = rng.choice(shapes)
                max_shape_row = max(row for row, _ in shape)
                max_shape_column = max(column for _, column in shape)
                anchor = (
                    rng.randrange(row_start, row_end - max_shape_row),
                    rng.randrange(column_start, column_end - max_shape_column),
                )
                candidate = {
                    (anchor[0] + row, anchor[1] + column) for row, column in shape
                }
                if (
                    candidate.intersection(protected)
                    or candidate.intersection(local_obstacles)
                    or len(local_obstacles | candidate) > target_bin_count
                ):
                    continue
                local_obstacles.update(candidate)
            if len(local_obstacles) != target_bin_count:
                success = False
                break
            obstacles.update(local_obstacles)
        if not success:
            continue
        component_sizes = _obstacle_component_sizes(obstacles, size)
        if component_sizes and max(component_sizes) > max_component_size:
            continue
        path = astar_path(start, goal, obstacles, size)
        if path is None:
            continue
        path_tuple = tuple(path)
        turn_count = _turn_count(path_tuple)
        if turn_count < min_turns or turn_count > max_turns:
            continue
        path_variants, path_union_cells = _sample_optimal_path_diversity(
            start, goal, obstacles, size
        )
        if (
            path_variants < min_path_variants
            or path_union_cells < min_path_union_cells
        ):
            continue
        return (
            obstacles,
            generation_attempt,
            path_variants,
            path_union_cells,
            tuple(
                len(
                    {
                        cell
                        for cell in obstacles
                        if (cell[0] // bin_size, cell[1] // bin_size)
                        == (row, column)
                    }
                )
                for row in range(bins)
                for column in range(bins)
            ),
        )
    raise RuntimeError(
        f"Unable to generate uniform calibration map seed={seed} with the registered gates."
    )


def _build_problem(
    *,
    map_id: str,
    seed: int,
    size: int,
    start: Position,
    goal: Position,
    obstacles: Iterable[Position],
    metadata: Mapping[str, Any],
) -> NavigationProblem:
    obstacle_set = frozenset(obstacles)
    if not in_bounds(start, size) or not in_bounds(goal, size):
        raise ValueError(f"Endpoints must be inside the {size}x{size} map.")
    if start in obstacle_set or goal in obstacle_set:
        raise ValueError("Structured layout blocks its start or goal.")
    if any(not in_bounds(cell, size) for cell in obstacle_set):
        raise ValueError("Structured layout contains an out-of-bounds obstacle.")
    path = astar_path(start, goal, obstacle_set, size)
    if path is None:
        raise ValueError(f"Structured layout {map_id} has no path.")
    nominal_path = tuple(path)
    manhattan_steps = manhattan(start, goal)
    values = {
        "generator": "structured_layout_v1",
        "layout_version": 1,
        "turn_count": _turn_count(nominal_path),
        "manhattan_steps": manhattan_steps,
        "detour_steps": len(nominal_path) - 1 - manhattan_steps,
        "detour_ratio": (len(nominal_path) - 1) / max(1, manhattan_steps),
        "obstacle_components": _obstacle_components(obstacle_set, size),
        **dict(metadata),
    }
    return NavigationProblem(
        map_id=map_id,
        seed=seed,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacle_set,
        nominal_path=nominal_path,
        metadata=values,
    )


def _calibration_20_map_01() -> NavigationProblem:
    size = 20
    start, goal = (1, 1), (19, 19)
    obstacles: set[Position] = set()

    def place(center: Position, offsets: Iterable[Position]) -> None:
        obstacles.update(
            (center[0] + row, center[1] + column) for row, column in offsets
        )

    plus = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
    tee = ((0, -1), (0, 0), (0, 1), (1, 0))
    corner = ((0, 0), (1, 0), (1, 1), (2, 1))
    place((3, 4), plus)
    place((4, 11), tee)
    place((3, 16), plus)
    obstacles |= _rectangle(6, 1, 7, 3)
    obstacles |= _rectangle(7, 8, 9, 10)
    place((8, 14), plus)
    place((10, 5), corner)
    obstacles |= _rectangle(10, 10, 11, 12)
    place((11, 17), plus)
    place((14, 2), plus)
    obstacles |= _rectangle(13, 7, 15, 9)
    place((15, 14), tee)
    obstacles |= _rectangle(17, 5, 17, 7)
    place((16, 11), plus)
    # Additional scattered pieces raise occupancy to the registered 30% target
    # without turning the calibration map into long wall corridors.
    obstacles |= _rectangle(2, 7, 3, 9)
    place((6, 17), plus)
    obstacles |= _rectangle(9, 1, 11, 3)
    place((12, 15), plus)
    obstacles |= _rectangle(13, 17, 15, 18)
    place((17, 2), plus)
    obstacles |= _rectangle(16, 16, 18, 18)
    # Break the otherwise trivial bottom-edge shortest route.
    obstacles.add((19, 4))
    obstacles.discard((17, 2))
    obstacles.discard((13, 17))
    # Remove bridge cells where nearby shapes accidentally merged into large walls.
    obstacles.difference_update(
        {(9, 10), (11, 16), (12, 17), (15, 17), (15, 18), (16, 2)}
    )
    # Redistribute the removed occupancy as isolated cells, preserving exactly 30%.
    obstacles.update({(1, 13), (5, 14), (9, 6), (12, 4), (13, 12), (18, 8), (18, 13)})
    path_variants, path_union_cells = _sample_optimal_path_diversity(
        start, goal, obstacles, size
    )
    component_sizes = _obstacle_component_sizes(obstacles, size)
    return _build_problem(
        map_id="calibration_20x20_map_01",
        seed=1100,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        metadata={
            "scene": "calibration",
            "structure": "scattered_mixed_shape_obstacles",
            "layout_version": 2,
            "obstacle_blocks": 21,
            "obstacle_shape_count": 4,
            "layout_mode": "fixed_seed_scattered_v2",
            "route_options": 3,
            "route_sampling_seeds": 100,
            "sampled_optimal_path_variants": path_variants,
            "sampled_optimal_path_union_cells": path_union_cells,
            "largest_obstacle_component": max(component_sizes),
            "nominal_corridor_width": 2,
            "bottleneck_width": 2,
            "purpose": "uniform_learning_calibration",
        },
    )


def _generated_calibration_20(map_number: int) -> NavigationProblem:
    if not 2 <= map_number <= 5:
        raise ValueError("Generated calibration map number must be from 2 to 5.")
    size = 20
    start, goal = (1, 1), (19, 19)
    seed = 1099 + map_number
    obstacles, attempt, path_variants, path_union_cells = (
        _random_scattered_calibration_obstacles(
            seed,
            size=size,
            start=start,
            goal=goal,
            target_count=120,
        )
    )
    component_sizes = _obstacle_component_sizes(obstacles, size)
    return _build_problem(
        map_id=f"calibration_20x20_map_{map_number:02d}",
        seed=seed,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        metadata={
            "scene": "calibration",
            "structure": "scattered_mixed_shape_obstacles",
            "layout_version": 2,
            "generation_attempt": attempt,
            "obstacle_shape_count": 10,
            "layout_mode": "fixed_seed_scattered_v2",
            "route_sampling_seeds": 100,
            "sampled_optimal_path_variants": path_variants,
            "sampled_optimal_path_union_cells": path_union_cells,
            "largest_obstacle_component": max(component_sizes),
            "nominal_corridor_width": 2,
            "bottleneck_width": 2,
            "purpose": "uniform_learning_calibration",
        },
    )


def _generated_hard_calibration_20(map_number: int) -> NavigationProblem:
    if not 1 <= map_number <= 5:
        raise ValueError("Hard calibration map number must be from 1 to 5.")
    size = 20
    start, goal = (1, 1), (19, 19)
    seed = 1199 + map_number
    obstacles, attempt, path_variants, path_union_cells = (
        _random_scattered_calibration_obstacles(
            seed,
            size=size,
            start=start,
            goal=goal,
            target_count=120,
            endpoint_clearance=0,
            min_detour_steps=4,
            max_detour_steps=14,
            min_turns=8,
            max_component_size=12,
            min_path_variants=40,
            min_path_union_cells=90,
        )
    )
    component_sizes = _obstacle_component_sizes(obstacles, size)
    return _build_problem(
        map_id=f"calibration_hard_20x20_map_{map_number:02d}",
        seed=seed,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        metadata={
            "scene": "calibration_hard",
            "structure": "scattered_mixed_shape_obstacles_with_detours",
            "layout_version": 3,
            "generation_attempt": attempt,
            "obstacle_shape_count": 10,
            "layout_mode": "fixed_seed_scattered_detour_v1",
            "endpoint_clearance": 0,
            "route_sampling_seeds": 100,
            "sampled_optimal_path_variants": path_variants,
            "sampled_optimal_path_union_cells": path_union_cells,
            "largest_obstacle_component": max(component_sizes),
            "nominal_corridor_width": 2,
            "bottleneck_width": 2,
            "purpose": "uniform_learning_calibration_harder",
        },
    )


_CALIBRATION_40_SEEDS = (1500, 1501, 1502, 1503, 1504)


def _generated_calibration_40(map_number: int) -> NavigationProblem:
    if not 1 <= map_number <= len(_CALIBRATION_40_SEEDS):
        raise ValueError("40x40 calibration map number must be from 1 to 5.")
    size = 40
    start, goal = (1, 1), (39, 39)
    seed = _CALIBRATION_40_SEEDS[map_number - 1]
    (
        obstacles,
        attempt,
        path_variants,
        path_union_cells,
        spatial_bin_counts,
    ) = (
        _uniform_scattered_calibration_obstacles(
            seed,
            size=size,
            start=start,
            goal=goal,
            target_count=360,
            endpoint_clearance=1,
            max_component_size=24,
            min_turns=10,
            max_turns=30,
            min_path_variants=50,
            min_path_union_cells=180,
        )
    )
    component_sizes = _obstacle_component_sizes(obstacles, size)
    return _build_problem(
        map_id=f"calibration_40x40_map_{map_number:02d}",
        seed=seed,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        metadata={
            "scene": "calibration_40x40",
            "structure": "uniformly_scattered_mixed_shape_obstacles",
            "layout_version": 5,
            "generation_attempt": attempt,
            "obstacle_shape_count": 10,
            "shape_profile": "large",
            "layout_mode": "fixed_seed_uniform_bins_v1",
            "target_density": 0.225,
            "endpoint_clearance": 1,
            "spatial_bins": 4,
            "spatial_bin_size": 10,
            "spatial_bin_counts": list(spatial_bin_counts),
            "route_sampling_seeds": 100,
            "sampled_optimal_path_variants": path_variants,
            "sampled_optimal_path_union_cells": path_union_cells,
            "largest_obstacle_component": max(component_sizes),
            "nominal_corridor_width": 2,
            "bottleneck_width": 2,
            "purpose": "uniform_learning_calibration_40x40",
        },
    )


def _office_40() -> NavigationProblem:
    size = 40
    start, goal = (2, 2), (37, 37)
    obstacles: set[Position] = set()
    # Two long walls create rooms above/below the main corridor. Each has two doors.
    for row in (10, 11):
        obstacles |= _rectangle(row, 0, row, 39)
    _remove_cells(obstacles, {(10, 9), (11, 9), (10, 27), (11, 27)})
    for row in (25, 26):
        obstacles |= _rectangle(row, 0, row, 39)
    _remove_cells(obstacles, {(25, 17), (26, 17), (25, 31), (26, 31)})
    # Office partitions leave a single-door passage from each room to its corridor.
    obstacles |= _rectangle(3, 7, 9, 8)
    _remove_cells(obstacles, {(6, 7), (6, 8)})
    obstacles |= _rectangle(3, 29, 9, 30)
    _remove_cells(obstacles, {(7, 29), (7, 30)})
    obstacles |= _rectangle(27, 7, 36, 8)
    _remove_cells(obstacles, {(31, 7), (31, 8)})
    obstacles |= _rectangle(27, 29, 36, 30)
    _remove_cells(obstacles, {(33, 29), (33, 30)})
    # A central partition makes the two corridors meaningfully different.
    obstacles |= _rectangle(12, 20, 24, 21)
    _remove_cells(obstacles, {(18, 20), (18, 21)})
    return _build_problem(
        map_id="office_40x40",
        seed=2100,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        metadata={
            "scene": "office",
            "structure": "rooms_walls_doors_main_corridor",
            "obstacle_blocks": 9,
            "door_count": 11,
            "route_options": 2,
            "nominal_corridor_width": 2,
            "bottleneck_width": 2,
            "purpose": "structured_static_main_experiment",
        },
    )


def _parcel_station_40() -> NavigationProblem:
    size = 40
    start, goal = (2, 2), (37, 37)
    obstacles: set[Position] = set()
    # Sorting area: a solid island with clear loading lanes on all four sides.
    obstacles |= _rectangle(13, 12, 26, 28)
    _remove_cells(
        obstacles,
        {
            (13, 16), (13, 17), (13, 23), (13, 24),
            (26, 16), (26, 17), (26, 23), (26, 24),
            (18, 12), (19, 12), (22, 28), (23, 28),
        },
    )
    # Cross-traffic wall above the sorting area, with two doors.
    obstacles |= _rectangle(8, 0, 9, 39)
    _remove_cells(obstacles, {(8, 13), (9, 13), (8, 28), (9, 28)})
    # A dispatch divider creates a choice between the left and right loading lanes.
    obstacles |= _rectangle(10, 20, 32, 21)
    _remove_cells(obstacles, {(16, 20), (16, 21), (29, 20), (29, 21)})
    # Workbenches are small obstacles inside the open station zones.
    obstacles |= _rectangle(17, 4, 19, 8)
    obstacles |= _rectangle(29, 5, 31, 10)
    obstacles |= _rectangle(17, 31, 19, 35)
    obstacles |= _rectangle(29, 29, 31, 34)
    return _build_problem(
        map_id="parcel_station_40x40",
        seed=2200,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        metadata={
            "scene": "parcel_station",
            "structure": "sorting_island_workbenches_cross_traffic",
            "obstacle_blocks": 8,
            "door_count": 12,
            "route_options": 2,
            "nominal_corridor_width": 2,
            "bottleneck_width": 2,
            "purpose": "structured_static_main_experiment",
        },
    )


def _warehouse_40() -> NavigationProblem:
    size = 40
    start, goal = (2, 2), (37, 37)
    obstacles: set[Position] = set()
    # Four parallel racks. Three horizontal cross-aisles keep the layout connected.
    for left in (7, 15, 23, 31):
        obstacles |= _rectangle(5, left, 34, left + 2)
        _remove_cells(
            obstacles,
            {
                (11, left), (12, left), (11, left + 1), (12, left + 1),
                (11, left + 2), (12, left + 2),
                (20, left), (21, left), (20, left + 1), (21, left + 1),
                (20, left + 2), (21, left + 2),
                (29, left), (30, left), (29, left + 1), (30, left + 1),
                (29, left + 2), (30, left + 2),
            },
        )
    # A two-cell bottleneck in the middle cross-aisle creates a controlled detour.
    obstacles |= _rectangle(17, 10, 18, 30)
    _remove_cells(obstacles, {(17, 19), (17, 20), (18, 19), (18, 20)})
    # Endcaps force the agent to use the cross-aisles rather than skirt every rack.
    obstacles |= _rectangle(4, 0, 4, 39)
    obstacles |= _rectangle(35, 0, 35, 39)
    _remove_cells(
        obstacles,
        {
            (4, 11), (4, 21), (4, 29),
            (35, 11), (35, 21), (35, 29),
        },
    )
    return _build_problem(
        map_id="warehouse_40x40",
        seed=2300,
        size=size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        metadata={
            "scene": "warehouse",
            "structure": "parallel_racks_cross_aisles_bottleneck",
            "obstacle_blocks": 7,
            "door_count": 25,
            "route_options": 3,
            "nominal_corridor_width": 5,
            "bottleneck_width": 2,
            "purpose": "structured_static_main_experiment",
        },
    )


def build_structured_problem(scene: str) -> NavigationProblem:
    builders = {
        "calibration_20x20": _calibration_20_map_01,
        "calibration_20x20_map_01": _calibration_20_map_01,
        "calibration_20x20_map_02": lambda: _generated_calibration_20(2),
        "calibration_20x20_map_03": lambda: _generated_calibration_20(3),
        "calibration_20x20_map_04": lambda: _generated_calibration_20(4),
        "calibration_20x20_map_05": lambda: _generated_calibration_20(5),
        "calibration_hard_20x20_map_01": lambda: _generated_hard_calibration_20(1),
        "calibration_hard_20x20_map_02": lambda: _generated_hard_calibration_20(2),
        "calibration_hard_20x20_map_03": lambda: _generated_hard_calibration_20(3),
        "calibration_hard_20x20_map_04": lambda: _generated_hard_calibration_20(4),
        "calibration_hard_20x20_map_05": lambda: _generated_hard_calibration_20(5),
        "calibration_40x40_map_01": lambda: _generated_calibration_40(1),
        "calibration_40x40_map_02": lambda: _generated_calibration_40(2),
        "calibration_40x40_map_03": lambda: _generated_calibration_40(3),
        "calibration_40x40_map_04": lambda: _generated_calibration_40(4),
        "calibration_40x40_map_05": lambda: _generated_calibration_40(5),
        "office_40x40": _office_40,
        "parcel_station_40x40": _parcel_station_40,
        "warehouse_40x40": _warehouse_40,
    }
    try:
        return builders[scene]()
    except KeyError as exc:
        choices = ", ".join(sorted(builders))
        raise ValueError(f"Unknown structured scene {scene!r}; choose from {choices}.") from exc


def build_structured_problem_set(scenes: Iterable[str]) -> list[NavigationProblem]:
    problems = [build_structured_problem(scene) for scene in scenes]
    if not problems:
        raise ValueError("At least one structured scene is required.")
    map_ids = [problem.map_id for problem in problems]
    if len(map_ids) != len(set(map_ids)):
        raise ValueError("Structured scene list contains duplicate maps.")
    return problems


__all__ = ["build_structured_problem", "build_structured_problem_set"]
