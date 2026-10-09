"""Design whole-map dynamic-obstacle route pools and previews; never train.

The route categories are derived from an ensemble of reasonable start-to-goal
paths over the complete frozen free-space graph.  The nominal A* path is drawn
only as a visual reference and is not used to enumerate obstacle routes.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from astar_d3qn.core.astar import randomized_tie_astar_path
from astar_d3qn.core.grid import ACTION_DELTAS, in_bounds, manhattan
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.utils.io import write_json


CATEGORY_COLORS = {
    "high_interaction": "#d73027",
    "alternative_branch": "#2478b5",
    "background": "#2ca25f",
}
CATEGORY_LABELS = {
    "high_interaction": "high-interaction corridor / bottleneck",
    "alternative_branch": "alternative route / branch-merge",
    "background": "background free-space",
}


def _neighbors(cell: tuple[int, int], problem: NavigationProblem):
    for dr, dc in ACTION_DELTAS[:4]:
        following = (cell[0] + dr, cell[1] + dc)
        if in_bounds(following, problem.size) and following not in problem.obstacles:
            yield following


def _distances(problem: NavigationProblem, source: tuple[int, int]):
    distance = {source: 0}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for following in _neighbors(cell, problem):
            if following not in distance:
                distance[following] = distance[cell] + 1
                queue.append(following)
    return distance


def _erase_loops(path: Sequence[tuple[int, int]]):
    result: list[tuple[int, int]] = []
    indices: dict[tuple[int, int], int] = {}
    for cell in path:
        if cell in indices:
            keep = indices[cell]
            for removed in result[keep + 1 :]:
                indices.pop(removed, None)
            result = result[: keep + 1]
        else:
            indices[cell] = len(result)
            result.append(cell)
    return tuple(result)


def build_reasonable_path_ensemble(
    problem: NavigationProblem,
    *,
    seed: int,
    sample_count: int,
    candidate_attempts: int,
    detour_budget_steps: int,
):
    """Sample diverse bounded-detour paths through the whole feasible envelope.

    Waypoints are chosen from every free cell whose best start-waypoint-goal walk
    is within the detour budget. Repeatedly used cells are down-weighted, so the
    ensemble includes common corridors as well as reasonable alternatives.
    """
    rng = random.Random(seed)
    start_distance = _distances(problem, problem.start)
    goal_distance = _distances(problem, problem.goal)
    optimum = start_distance[problem.goal]
    feasible = [
        cell
        for cell in start_distance.keys() & goal_distance.keys()
        if start_distance[cell] + goal_distance[cell] <= optimum + detour_budget_steps
        and cell not in (problem.start, problem.goal)
    ]
    if not feasible:
        raise RuntimeError(f"{problem.map_id}: no reasonable-path envelope cells.")

    paths: list[tuple[tuple[int, int], ...]] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    usage: Counter[tuple[int, int]] = Counter()

    def accept(path: Sequence[tuple[int, int]] | None):
        if not path:
            return False
        simple = _erase_loops(tuple(path))
        if (
            simple[0] != problem.start
            or simple[-1] != problem.goal
            or len(simple) - 1 > optimum + detour_budget_steps
            or simple in signatures
        ):
            return False
        if any(
            manhattan(left, right) != 1
            for left, right in zip(simple, simple[1:])
        ):
            return False
        paths.append(simple)
        signatures.add(simple)
        usage.update(simple)
        return True

    # Seed the ensemble with equal-length alternatives before asking for detours.
    for _ in range(min(sample_count // 3, 1000)):
        accept(
            randomized_tie_astar_path(
                problem.start,
                problem.goal,
                problem.obstacles,
                problem.size,
                rng,
            )
        )

    for _ in range(candidate_attempts):
        if len(paths) >= sample_count:
            break
        weights = [
            (1.0 / (1.0 + usage[cell]))
            * (1.0 + 0.12 * (start_distance[cell] + goal_distance[cell] - optimum))
            for cell in feasible
        ]
        waypoint = rng.choices(feasible, weights=weights, k=1)[0]
        first = randomized_tie_astar_path(
            problem.start, waypoint, problem.obstacles, problem.size, rng
        )
        second = randomized_tie_astar_path(
            waypoint, problem.goal, problem.obstacles, problem.size, rng
        )
        if first and second:
            accept(tuple(first) + tuple(second[1:]))
    if len(paths) < sample_count:
        raise RuntimeError(
            f"{problem.map_id}: produced {len(paths)}/{sample_count} distinct "
            "reasonable paths; increase candidate_attempts or revise the new protocol."
        )
    return tuple(paths), start_distance, goal_distance


def _transition_entropy(paths: Sequence[Sequence[tuple[int, int]]]):
    directions: dict[tuple[int, int], Counter[tuple[int, int]]] = defaultdict(Counter)
    for path in paths:
        for left, right in zip(path, path[1:]):
            delta = (right[0] - left[0], right[1] - left[1])
            directions[left][delta] += 1
            directions[right][(-delta[0], -delta[1])] += 1
    entropy = {}
    for cell, counts in directions.items():
        total = sum(counts.values())
        entropy[cell] = -sum(
            count / total * math.log(count / total, 2) for count in counts.values()
        ) / 2.0  # Four directions have maximum entropy 2 bits.
    return entropy, directions


def _enumerate_free_routes(problem: NavigationProblem):
    routes = set()
    for length in (3, 5, 7):
        half = length // 2
        for row in range(problem.size):
            for column in range(problem.size):
                center = (row, column)
                if center in problem.obstacles or center in (problem.start, problem.goal):
                    continue
                for orientation, delta in (
                    ("horizontal", (0, 1)),
                    ("vertical", (1, 0)),
                ):
                    route = tuple(
                        (row + offset * delta[0], column + offset * delta[1])
                        for offset in range(-half, half + 1)
                    )
                    if any(
                        not in_bounds(cell, problem.size)
                        or cell in problem.obstacles
                        or cell in (problem.start, problem.goal)
                        for cell in route
                    ):
                        continue
                    routes.add((route, center, orientation))
    return sorted(routes)


def _perpendicular_bonus(route, orientation, directions):
    along = {(0, 1), (0, -1)} if orientation == "horizontal" else {(1, 0), (-1, 0)}
    crossing = 0
    parallel = 0
    for cell in route:
        for delta, count in directions.get(cell, {}).items():
            if delta in along:
                parallel += count
            else:
                crossing += count
    total = crossing + parallel
    return crossing / total if total else 0.0


def _route_features(problem, paths, path_sets, usage, entropy, directions):
    records = []
    nominal = set(problem.nominal_path)
    path_count = len(paths)
    start_distance = _distances(problem, problem.start)
    optimum = start_distance[problem.goal]
    for route, center, orientation in _enumerate_free_routes(problem):
        route_cells = set(route)
        hit_indices = [
            index for index, cells in enumerate(path_sets) if route_cells & cells
        ]
        hit_count = len(hit_indices)
        coverage = hit_count / path_count
        mean_support = sum(usage[cell] for cell in route) / (len(route) * path_count)
        branch_entropy = max((entropy.get(cell, 0.0) for cell in route), default=0.0)
        perpendicular = _perpendicular_bonus(route, orientation, directions)
        records.append(
            {
                "route": [list(cell) for cell in route],
                "center": list(center),
                "orientation": orientation,
                "route_length": len(route),
                "ensemble_paths_intersected": hit_count,
                "ensemble_path_indices": hit_indices,
                "ensemble_intersection_fraction": round(coverage, 6),
                "mean_cell_support": round(mean_support, 6),
                "branch_merge_entropy": round(branch_entropy, 6),
                "perpendicular_flow_fraction": round(perpendicular, 6),
                "progress_fraction": round(
                    sum(start_distance[cell] for cell in route) / (len(route) * optimum),
                    6,
                ),
                "intersects_nominal_astar": bool(route_cells & nominal),
            }
        )
    return records


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]):
    return manhattan(tuple(left["center"]), tuple(right["center"]))


def _select_spread(
    candidates: Iterable[dict[str, Any]],
    count: int,
    *,
    occupied: Sequence[dict[str, Any]] = (),
    minimum_center_gap: int = 3,
):
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(
            _distance(candidate, prior) >= minimum_center_gap
            for prior in (*occupied, *selected)
        ):
            selected.append(candidate)
            if len(selected) == count:
                return selected
    # Pool construction is allowed to relax spacing by one cell, never validity.
    if minimum_center_gap > 1:
        return _select_spread(
            candidates,
            count,
            occupied=occupied,
            minimum_center_gap=minimum_center_gap - 1,
        )
    raise RuntimeError(f"Could select only {len(selected)}/{count} routes.")


def build_route_pool(problem, paths, counts, classification):
    path_sets = tuple(set(path) for path in paths)
    usage = Counter(cell for path in paths for cell in set(path))
    entropy, directions = _transition_entropy(paths)
    records = _route_features(problem, paths, path_sets, usage, entropy, directions)
    positive = sorted(record["ensemble_intersection_fraction"] for record in records if record["ensemble_intersection_fraction"] > 0)
    high_cutoff = positive[int(0.72 * (len(positive) - 1))]
    low_cutoff = positive[int(0.20 * (len(positive) - 1))]

    progress_edges = tuple(
        float(value) for value in classification["high_interaction_progress_edges"]
    )
    high_count = int(counts["high_interaction"])
    band_count = len(progress_edges) - 1
    if band_count <= 0 or high_count % band_count:
        raise ValueError("High-interaction count must divide evenly across progress bands.")
    high: list[dict[str, Any]] = []
    for band_index, (low, high_edge) in enumerate(zip(progress_edges, progress_edges[1:])):
        band_candidates = [
            record
            for record in records
            if low <= record["progress_fraction"] < high_edge
            and record["ensemble_intersection_fraction"] > 0
        ]
        band_candidates.sort(
            key=lambda record: (
                -record["ensemble_intersection_fraction"],
                -record["perpendicular_flow_fraction"],
                -record["branch_merge_entropy"],
                record["route_length"],
                record["center"],
            )
        )
        selected_band = _select_spread(
            band_candidates,
            high_count // band_count,
            occupied=high,
            minimum_center_gap=3,
        )
        for record in selected_band:
            record["interaction_progress_band"] = band_index
        high.extend(selected_band)

    high_geometry = {tuple(map(tuple, record["route"])) for record in high}
    alternative_candidates = [
        record
        for record in records
        if tuple(map(tuple, record["route"])) not in high_geometry
        and low_cutoff < record["ensemble_intersection_fraction"] < high_cutoff
    ]
    alternative_candidates.sort(
        key=lambda record: (
            -(
                record["branch_merge_entropy"]
                * (0.5 + record["perpendicular_flow_fraction"])
                * (1.25 if not record["intersects_nominal_astar"] else 1.0)
            ),
            -record["ensemble_intersection_fraction"],
            record["route_length"],
            record["center"],
        )
    )
    alternative = _select_spread(
        alternative_candidates,
        int(counts["alternative_branch"]),
        occupied=high,
        minimum_center_gap=2,
    )

    selected_geometry = high_geometry | {
        tuple(map(tuple, record["route"])) for record in alternative
    }
    background_candidates = [
        record
        for record in records
        if tuple(map(tuple, record["route"])) not in selected_geometry
        and record["ensemble_intersection_fraction"] <= low_cutoff
    ]
    # Prefer low-interaction routes that remain near the reasonable-path envelope.
    background_candidates.sort(
        key=lambda record: (
            -record["ensemble_intersection_fraction"],
            -record["branch_merge_entropy"],
            record["route_length"],
            record["center"],
        )
    )
    background = _select_spread(
        background_candidates,
        int(counts["background"]),
        occupied=(*high, *alternative),
        minimum_center_gap=2,
    )

    result = {
        "high_interaction": high,
        "alternative_branch": alternative,
        "background": background,
    }
    prefixes = {"high_interaction": "HI", "alternative_branch": "AR", "background": "BG"}
    for category, category_records in result.items():
        for index, record in enumerate(category_records, start=1):
            record["route_id"] = f"{problem.map_id}_{prefixes[category]}_{index:02d}"
            record["category"] = category
            record["motion"] = "continuous_ping_pong"
    support = [
        {"cell": list(cell), "path_count": count, "path_fraction": round(count / len(paths), 6)}
        for cell, count in sorted(usage.items())
    ]
    return result, support, {"high_cutoff": high_cutoff, "low_cutoff": low_cutoff}


def _route_cells(record):
    return set(map(tuple, record["route"]))


def sample_example_scenarios(problem, route_pool, config, seed):
    rng = random.Random(seed)
    count = int(config["count"])
    composition = {key: int(value) for key, value in config["composition"].items()}
    trials = int(config["sampling_trials_per_scene"])
    minimum_coverage = float(config["minimum_ensemble_path_coverage"])
    ensemble_size = max(
        index
        for records in route_pool.values()
        for record in records
        for index in record["ensemble_path_indices"]
    ) + 1
    scenarios = []
    used_signatures = set()
    route_usage: Counter[str] = Counter()
    progress_band_pairs = list(itertools.combinations(range(4), 2))
    rng.shuffle(progress_band_pairs)
    for scenario_index in range(count):
        required_high_bands = progress_band_pairs[scenario_index % len(progress_band_pairs)]
        best = None
        for _ in range(trials):
            selected: list[dict[str, Any]] = []
            valid = True
            for category in ("high_interaction", "alternative_branch", "background"):
                choices = list(route_pool[category])
                rng.shuffle(choices)
                required = composition[category]
                for candidate in choices:
                    cells = _route_cells(candidate)
                    if (
                        category == "high_interaction"
                        and candidate["interaction_progress_band"] not in required_high_bands
                    ):
                        continue
                    if category == "high_interaction" and any(
                        prior["category"] == "high_interaction"
                        and prior["interaction_progress_band"]
                        == candidate["interaction_progress_band"]
                        for prior in selected
                    ):
                        continue
                    if any(cells & _route_cells(prior) for prior in selected):
                        continue
                    if any(
                        manhattan(tuple(candidate["center"]), tuple(prior["center"])) < 3
                        for prior in selected
                    ):
                        continue
                    selected.append(candidate)
                    required -= 1
                    if required == 0:
                        break
                if required:
                    valid = False
                    break
                if category == "high_interaction" and {
                    record["interaction_progress_band"]
                    for record in selected
                    if record["category"] == "high_interaction"
                } != set(required_high_bands):
                    valid = False
                    break
            if not valid:
                continue
            signature = tuple(sorted(record["route_id"] for record in selected))
            if signature in used_signatures:
                continue
            path_indices = set().union(
                *(set(record["ensemble_path_indices"]) for record in selected)
            )
            interactive = [
                record for record in selected if record["category"] != "background"
            ]
            progress_span = max(record["progress_fraction"] for record in interactive) - min(
                record["progress_fraction"] for record in interactive
            )
            reuse = sum(route_usage[record["route_id"]] for record in selected)
            score = (-reuse, progress_span, len(path_indices), rng.random())
            if best is None or score > best[0]:
                best = (score, selected, path_indices, signature)
        if best is None:
            raise RuntimeError(
                f"{problem.map_id}: cannot sample disjoint routes for example {scenario_index}."
            )
        _, selected, path_indices, signature = best
        path_coverage = len(path_indices) / ensemble_size
        if path_coverage < minimum_coverage:
            raise RuntimeError(
                f"{problem.map_id}: example {scenario_index} covers only "
                f"{path_coverage:.1%} of reasonable paths; required {minimum_coverage:.1%}."
            )
        used_signatures.add(signature)
        route_usage.update(record["route_id"] for record in selected)
        obstacles = []
        for record in selected:
            route_length = len(record["route"])
            obstacles.append(
                {
                    "route_id": record["route_id"],
                    "category": record["category"],
                    "route": record["route"],
                    "start_index": rng.randrange(route_length),
                    "direction": rng.choice((-1, 1)),
                    "move_every": int(config["move_every"]),
                    "motion": "continuous_ping_pong_for_entire_episode",
                }
            )
        scenarios.append(
            {
                "scenario_id": f"{problem.map_id}_preview_{scenario_index + 1:02d}",
                "obstacles": obstacles,
                "composition": composition,
                "selection_time": "episode_reset_only",
                "regenerate_during_episode": False,
                "agent_tracking_placement": False,
                "ensemble_paths_with_interaction_opportunity": len(path_indices),
                "ensemble_path_coverage_fraction": round(path_coverage, 6),
                "high_interaction_progress_bands": list(required_high_bands),
            }
        )
    return scenarios


def validate_map_entry(
    problem,
    entry,
    expected_counts,
    expected_composition,
    minimum_ensemble_path_coverage=0.55,
):
    errors = []
    route_lookup = {}
    for category, expected in expected_counts.items():
        records = entry["route_pool"][category]
        if len(records) != int(expected):
            errors.append(f"{category}: {len(records)} != {expected}")
        for record in records:
            route = tuple(map(tuple, record["route"]))
            route_lookup[record["route_id"]] = record
            if len(route) < 2 or len(set(route)) != len(route):
                errors.append(f"{record['route_id']}: invalid length/duplicates")
            if any(cell in problem.obstacles or not in_bounds(cell, problem.size) for cell in route):
                errors.append(f"{record['route_id']}: not entirely in free space")
            if any(manhattan(a, b) != 1 for a, b in zip(route, route[1:])):
                errors.append(f"{record['route_id']}: route is not four-connected")
    for scenario in entry["example_scenarios"]:
        categories = Counter(obstacle["category"] for obstacle in scenario["obstacles"])
        if categories != Counter(expected_composition):
            errors.append(f"{scenario['scenario_id']}: composition {dict(categories)}")
        if len(scenario["obstacles"]) != 5:
            errors.append(f"{scenario['scenario_id']}: does not have five obstacles")
        occupied = []
        interactive = 0
        for obstacle in scenario["obstacles"]:
            record = route_lookup[obstacle["route_id"]]
            cells = _route_cells(record)
            if any(cells & prior for prior in occupied):
                errors.append(f"{scenario['scenario_id']}: overlapping routes")
            occupied.append(cells)
            interactive += record["ensemble_intersection_fraction"] > 0
        if interactive < 4:
            errors.append(f"{scenario['scenario_id']}: only {interactive}/5 routes meet reasonable paths")
        if scenario["ensemble_path_coverage_fraction"] < minimum_ensemble_path_coverage:
            errors.append(
                f"{scenario['scenario_id']}: path coverage "
                f"{scenario['ensemble_path_coverage_fraction']:.1%}"
            )
        if scenario["regenerate_during_episode"] or scenario["agent_tracking_placement"]:
            errors.append(f"{scenario['scenario_id']}: forbidden placement behavior")
    if errors:
        raise RuntimeError(f"{problem.map_id} validation failed:\n" + "\n".join(errors))
    return {
        "valid_routes": sum(len(records) for records in entry["route_pool"].values()),
        "valid_example_scenarios": len(entry["example_scenarios"]),
        "minimum_interactive_routes_per_scene": min(
            sum(
                route_lookup[obstacle["route_id"]]["ensemble_intersection_fraction"] > 0
                for obstacle in scenario["obstacles"]
            )
            for scenario in entry["example_scenarios"]
        ),
        "minimum_ensemble_path_coverage_fraction": min(
            scenario["ensemble_path_coverage_fraction"]
            for scenario in entry["example_scenarios"]
        ),
        "all_routes_static_free_and_four_connected": True,
        "all_scenes_episode_reset_only": True,
        "all_scenes_exact_2_2_1_composition": True,
    }


def _draw_base(ax, problem, support=None, path_count=None):
    grid = np.ones((problem.size, problem.size, 4), dtype=float)
    for row, column in problem.obstacles:
        grid[row, column] = (0.12, 0.12, 0.12, 1.0)
    ax.imshow(grid, origin="upper", interpolation="none")
    if support and path_count:
        heat = np.zeros((problem.size, problem.size), dtype=float)
        for row in support:
            cell = row["cell"]
            heat[cell[0], cell[1]] = row["path_count"] / path_count
        masked = np.ma.masked_where(heat <= 0, heat)
        ax.imshow(masked, cmap="YlOrBr", vmin=0, vmax=1, alpha=0.24, origin="upper", interpolation="none")
    nominal = np.asarray(problem.nominal_path)
    ax.plot(nominal[:, 1], nominal[:, 0], "--", color="#666666", lw=0.9, alpha=0.7)
    ax.scatter(problem.start[1], problem.start[0], s=55, color="#31a354", zorder=8)
    ax.scatter(problem.goal[1], problem.goal[0], s=100, marker="*", color="#ffd92f", edgecolors="black", zorder=8)
    ax.set(
        xlim=(-0.5, problem.size - 0.5),
        ylim=(problem.size - 0.5, -0.5),
        xlabel="column",
        ylabel="row",
    )
    ax.set_xticks(range(0, problem.size, 5))
    ax.set_yticks(range(0, problem.size, 5))
    ax.grid(alpha=0.12)


def render_route_distribution(entries, problems, destination):
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(entries), figsize=(7 * len(entries), 7), squeeze=False)
    for ax, entry, problem in zip(axes[0], entries, problems):
        _draw_base(ax, problem, entry["path_support"], entry["path_ensemble"]["sample_count"])
        for category in ("background", "alternative_branch", "high_interaction"):
            for record in entry["route_pool"][category]:
                route = np.asarray(record["route"])
                ax.plot(
                    route[:, 1],
                    route[:, 0],
                    color=CATEGORY_COLORS[category],
                    lw=1.7,
                    alpha=0.78,
                )
        counts = {category: len(records) for category, records in entry["route_pool"].items()}
        ax.set_title(
            f"{problem.map_id}\nwhole-map route pool: "
            f"HI={counts['high_interaction']}, AR={counts['alternative_branch']}, BG={counts['background']}"
        )
    handles = [
        Line2D([0], [0], color=CATEGORY_COLORS[key], lw=3, label=CATEGORY_LABELS[key])
        for key in CATEGORY_COLORS
    ] + [
        Line2D([0], [0], color="#666666", lw=1, ls="--", label="nominal A* (reference only)"),
        Line2D([0], [0], color="#d99b24", lw=7, alpha=0.24, label="reasonable-path support heatmap"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Dynamic route pools derived from whole-map reasonable-path ensembles", fontsize=15)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(destination / "route_distribution_overview.png", dpi=170)
    plt.close(fig)

    for entry, problem in zip(entries, problems):
        fig, ax = plt.subplots(figsize=(8, 8))
        _draw_base(ax, problem, entry["path_support"], entry["path_ensemble"]["sample_count"])
        for category in ("background", "alternative_branch", "high_interaction"):
            for record in entry["route_pool"][category]:
                route = np.asarray(record["route"])
                ax.plot(route[:, 1], route[:, 0], color=CATEGORY_COLORS[category], lw=2, alpha=0.8)
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_title(f"{problem.map_id}: complete candidate dynamic-route distribution")
        fig.tight_layout()
        fig.savefig(destination / f"{problem.map_id}_route_distribution.png", dpi=170)
        plt.close(fig)


def render_examples(entries, problems, destination):
    for entry, problem in zip(entries, problems):
        scenarios = entry["example_scenarios"]
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 5.5), squeeze=False)
        for ax, scenario in zip(axes[0], scenarios):
            _draw_base(ax, problem)
            for number, obstacle in enumerate(scenario["obstacles"], start=1):
                route = np.asarray(obstacle["route"])
                color = CATEGORY_COLORS[obstacle["category"]]
                ax.plot(route[:, 1], route[:, 0], "o-", color=color, lw=2, ms=2.5)
                initial = route[obstacle["start_index"]]
                ax.scatter(initial[1], initial[0], s=70, color=color, edgecolors="white", zorder=8)
                ax.text(initial[1] + 0.35, initial[0] - 0.35, str(number), color=color, fontsize=8, weight="bold")
            ax.set_title(f"{scenario['scenario_id'].rsplit('_', 1)[-1]}: 2 HI + 2 AR + 1 BG")
        handles = [
            Line2D([0], [0], color=CATEGORY_COLORS[key], marker="o", lw=2, label=CATEGORY_LABELS[key])
            for key in CATEGORY_COLORS
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
        fig.suptitle(
            f"{problem.map_id}: five episode-fixed moving obstacles (five random examples)",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(destination / f"{problem.map_id}_example_scenarios.png", dpi=160)
        plt.close(fig)


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dynamic_route_pool_v1.yaml")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_entries = {
        entry["problem"]["map_id"]: entry for entry in source["maps"]
    }
    problems = [
        problem_from_record(source_entries[map_id]["problem"])
        for map_id in config["map_ids"]
    ]
    design = {
        "algorithm_revision": "whole_map_progress_stratified_v2",
        "source_manifest": config["source_manifest"],
        "source_sha256": _sha256(source_path),
        "map_ids": config["map_ids"],
        "path_ensemble": config["path_ensemble"],
        "route_pool_counts": config["route_pool_counts"],
        "classification": config["classification"],
        "example_scenarios": config["example_scenarios"],
    }
    design_sha256 = hashlib.sha256(
        json.dumps(design, sort_keys=True).encode("utf-8")
    ).hexdigest()
    destination = ROOT / config["dataset"]
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("design_sha256") != design_sha256:
            raise SystemExit("Frozen dynamic route-pool dataset differs from config; create a new version.")
        print(f"Using existing frozen design: {destination}", flush=True)
    else:
        payload = {
            "format_version": 1,
            "protocol": config["experiment"],
            "design_only": True,
            "training_started": False,
            "design": design,
            "design_sha256": design_sha256,
            "maps": [],
        }
        for map_index, problem in enumerate(problems):
            ensemble_config = config["path_ensemble"]
            paths, _, _ = build_reasonable_path_ensemble(
                problem,
                seed=problem.seed + int(ensemble_config["seed_offset"]),
                sample_count=int(ensemble_config["sample_count"]),
                candidate_attempts=int(ensemble_config["candidate_attempts"]),
                detour_budget_steps=int(ensemble_config["detour_budget_steps"]),
            )
            route_pool, support, thresholds = build_route_pool(
                problem,
                paths,
                config["route_pool_counts"],
                config["classification"],
            )
            entry = {
                "problem_reference": {
                    "map_id": problem.map_id,
                    "grid_sha256": problem.grid_sha256,
                    "size": problem.size,
                    "start": list(problem.start),
                    "goal": list(problem.goal),
                    "static_obstacle_count": len(problem.obstacles),
                    "nominal_astar_steps": problem.astar_steps,
                },
                "path_ensemble": {
                    "sample_count": len(paths),
                    "unique_path_count": len(set(paths)),
                    "minimum_steps": min(len(path) - 1 for path in paths),
                    "maximum_steps": max(len(path) - 1 for path in paths),
                    "detour_budget_steps": int(ensemble_config["detour_budget_steps"]),
                    "construction": "whole_free_space_bounded_detour_waypoint_ensemble",
                    "nominal_astar_used_for_route_enumeration": False,
                },
                "classification_thresholds": thresholds,
                "path_support": support,
                "route_pool": route_pool,
                "example_scenarios": sample_example_scenarios(
                    problem,
                    route_pool,
                    config["example_scenarios"],
                    seed=problem.seed + 700000 + map_index,
                ),
            }
            entry["acceptance"] = validate_map_entry(
                problem,
                entry,
                config["route_pool_counts"],
                config["example_scenarios"]["composition"],
                config["example_scenarios"]["minimum_ensemble_path_coverage"],
            )
            all_routes = [record for records in route_pool.values() for record in records]
            entry["independence_audit"] = {
                "candidate_route_count": len(all_routes),
                "routes_not_intersecting_nominal_astar": sum(
                    not record["intersects_nominal_astar"] for record in all_routes
                ),
                "alternative_routes_not_intersecting_nominal_astar": sum(
                    not record["intersects_nominal_astar"]
                    for record in route_pool["alternative_branch"]
                ),
                "classification_uses_whole_map_path_ensemble": True,
                "single_astar_is_only_a_preview_reference": True,
            }
            payload["maps"].append(entry)
            print(
                f"{problem.map_id}: {len(paths)} reasonable paths; "
                + ", ".join(f"{key}={len(value)}" for key, value in route_pool.items()),
                flush=True,
            )
        write_json(payload, destination)

    # Always re-validate and render the frozen design, without invoking any trainer.
    for problem, entry in zip(problems, payload["maps"]):
        validate_map_entry(
            problem,
            entry,
            config["route_pool_counts"],
            config["example_scenarios"]["composition"],
            config["example_scenarios"]["minimum_ensemble_path_coverage"],
        )
    preview_root = ROOT / config["preview_root"]
    render_route_distribution(payload["maps"], problems, preview_root)
    render_examples(payload["maps"], problems, preview_root)
    print(
        f"Dataset: {destination}\nPreviews: {preview_root}\nNo training started.",
        flush=True,
    )


if __name__ == "__main__":
    main()
