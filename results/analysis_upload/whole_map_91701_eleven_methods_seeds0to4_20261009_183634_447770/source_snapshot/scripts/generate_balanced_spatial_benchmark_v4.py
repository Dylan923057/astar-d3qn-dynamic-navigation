"""Generate v4 spatial benchmarks with time-expanded safe-path matching."""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_problem_overview
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv

from audit_spatiotemporal_difficulty import audit_manifest
from generate_balanced_spatial_benchmark import (
    _resolve,
    build_balanced_manifest,
    render_route_pool_overview,
    render_validation_gallery,
)
from astar_d3qn.maps.render import render_problem_layout


DEFAULT_CONFIGS = (
    "configs/dynamic_spatial_generalization_office_balanced_v4.yaml",
    "configs/dynamic_spatial_generalization_parcel_balanced_v4.yaml",
    "configs/dynamic_spatial_generalization_warehouse_balanced_v4.yaml",
)


def _problem_for_config(config: dict[str, Any]):
    map_set = config["map_sets"]["train"]
    problems = load_problem_set(_resolve(str(map_set["file"])))
    matches = [problem for problem in problems if problem.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise ValueError("Configured scene is not uniquely registered.")
    return matches[0]


def _difficulty_objective(rows: list[dict[str, object]]) -> tuple[float, ...]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["difficulty_stratum"] == "high":
            groups[(str(row["split"]), "high")].append(float(row["safe_detour_steps"]))
    means = [statistics.fmean(groups[key]) for key in sorted(groups)]
    maxima = [max(groups[key]) for key in sorted(groups)]
    # Primary objective is cross-split mean matching; the next terms avoid
    # selecting a uniformly harder candidate when several seeds tie.
    return (
        max(means) - min(means),
        max(maxima) - min(maxima),
        max(means),
        statistics.pvariance(means),
    )


def _build_best_manifest(problem, config: dict[str, Any], search_window: int):
    base_seed = int(config["spatial_generalization"]["generation_seed"])
    candidates = [base_seed + offset for offset in range(-search_window, search_window + 1)]
    best = None
    for seed in candidates:
        candidate_config = copy.deepcopy(config)
        candidate_config["spatial_generalization"]["generation_seed"] = seed
        try:
            manifest = build_balanced_manifest(problem, candidate_config)
        except ValueError:
            # Some packing seeds do not leave enough phase-controllable routes
            # after enforcing the spatial buffer; those seeds are not candidates.
            continue
        rows = audit_manifest(problem, manifest, only_strata={"high"})
        objective = _difficulty_objective(rows)
        key = (objective, seed)
        if best is None or key < best[0]:
            best = (key, manifest, rows)
    if best is None:
        raise RuntimeError("No v4 generation candidate was available.")
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--search-window", type=int, default=3)
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument(
        "--audit-output",
        default="outputs/spatial_generalization_balanced_v4_design",
    )
    args = parser.parse_args()
    audit_dir = _resolve(args.audit_output)
    audit_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    problems = []
    manifest_records = []
    for config_name in tuple(args.configs or DEFAULT_CONFIGS):
        config = load_config(_resolve(config_name))
        problem = _problem_for_config(config)
        if args.no_search:
            manifest = build_balanced_manifest(problem, config)
            rows = audit_manifest(problem, manifest)
        else:
            manifest, rows = _build_best_manifest(problem, config, args.search_window)
        # Search uses only high-stratum rows for speed; the frozen output always
        # receives a complete audit covering every scenario.
        rows = audit_manifest(problem, manifest)
        manifest["generation"]["protocol"] = "conflict_balanced_spatial_v4"
        manifest["generation"]["difficulty_matching"] = {
            "metric": "minimum_collision_free_steps_minus_nominal_path_steps",
            "oracle": "periodic_time_expanded_bfs_with_wait_action",
            "objective": "minimize_cross_split_high_stratum_mean_detour_range",
        }
        target = _resolve(config["spatial_generalization"]["manifest"])
        write_json(manifest, target)
        render_problem_layout(problem, _resolve(f"maps/previews/{problem.map_id}_balanced_v4_static.png"))
        render_route_pool_overview(problem, manifest, _resolve(config["spatial_generalization"]["route_pool_preview"]))
        render_validation_gallery(problem, manifest, _resolve(config["spatial_generalization"]["validation_preview"]))
        all_rows.extend(rows)
        problems.append(problem)
        digest = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
        manifest_records.append(
            {
                "map_id": problem.map_id,
                "config": str(_resolve(config_name)),
                "manifest": str(target),
                "generation_seed": manifest["generation"]["seed"],
                "sha256": digest,
                "difficulty_objective": _difficulty_objective(rows),
            }
        )
        print(
            f"{problem.map_id}: seed={manifest['generation']['seed']} "
            f"sha256={digest[:12]} objective={_difficulty_objective(rows)}",
            flush=True,
        )
    render_problem_overview(problems, _resolve("maps/previews/spatial_generalization_balanced_v4_static_overview.png"))
    write_records_csv(all_rows, audit_dir / "scenario_spatiotemporal_difficulty.csv")
    write_json(manifest_records, audit_dir / "manifests.json")
    print(f"Balanced v4 design audit saved to {audit_dir}", flush=True)


if __name__ == "__main__":
    main()
