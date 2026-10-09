"""Audit frozen spatial manifests with a time-expanded safe-path oracle."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
from astar_d3qn.envs.dynamic_scenarios import DynamicScenario
from astar_d3qn.envs.spatial_scenarios import SPLITS, scenarios_from_spatial_manifest
from astar_d3qn.evaluation.conflict import (
    route_record_lookup,
    scenario_conflict_metrics,
    scenario_safe_path_metrics,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_records_csv


DEFAULT_CONFIGS = (
    "configs/dynamic_spatial_generalization_office_balanced_v3.yaml",
    "configs/dynamic_spatial_generalization_parcel_balanced_v3.yaml",
    "configs/dynamic_spatial_generalization_warehouse_balanced_v3.yaml",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _problem_for_config(config: Mapping[str, object]):
    map_sets = config["map_sets"]["train"]  # type: ignore[index]
    problems = load_problem_set(_resolve(str(map_sets["file"])))  # type: ignore[index]
    scene = str(config["map"]["scene"])  # type: ignore[index]
    matches = [problem for problem in problems if problem.map_id == scene]
    if len(matches) != 1:
        raise ValueError(f"Scene {scene!r} is not uniquely registered.")
    return matches[0]


def audit_manifest(
    problem,
    manifest: Mapping[str, object],
    only_strata: set[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        lookup = route_record_lookup(manifest, split)
        raw_scenarios = manifest["scenarios"][split]  # type: ignore[index]
        scenarios = scenarios_from_spatial_manifest(problem, manifest, split)
        for raw, scenario in zip(raw_scenarios, scenarios, strict=True):  # type: ignore[arg-type]
            if only_strata is not None and str(raw["difficulty_stratum"]) not in only_strata:  # type: ignore[index]
                continue
            specs = []
            for item in raw["obstacles"]:  # type: ignore[index]
                record = lookup[str(item["route_id"])]
                specs.append(
                    DynamicObstacleSpec(
                        route=tuple(tuple(cell) for cell in record["route"]),
                        start_index=int(item["start_index"]),
                        direction=int(item["direction"]),
                        move_every=int(item["move_every"]),
                        label=str(item["route_id"]),
                    )
                )
            materialized = DynamicScenario(
                seed=int(raw["scenario_id"]), obstacles=tuple(specs)  # type: ignore[index]
            )
            rows.append(
                {
                    "map_id": problem.map_id,
                    "split": split,
                    "scenario_id": int(raw["scenario_id"]),  # type: ignore[index]
                    "difficulty_stratum": str(raw["difficulty_stratum"]),  # type: ignore[index]
                    **{
                        key: value
                        for key, value in scenario_conflict_metrics(
                            problem,
                            materialized,
                            temporal_window=2,
                            route_records=lookup,
                        ).items()
                        if key != "obstacle_metrics"
                    },
                    **scenario_safe_path_metrics(problem, scenario),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument(
        "--output",
        default="outputs/spatiotemporal_difficulty_audit.csv",
    )
    args = parser.parse_args()
    rows = []
    for config_name in tuple(args.configs or DEFAULT_CONFIGS):
        config = load_config(_resolve(config_name))
        problem = _problem_for_config(config)
        manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))  # type: ignore[index]
        rows.extend(audit_manifest(problem, manifest))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_records_csv(rows, output)
    print(f"Wrote {len(rows)} scenario rows to {output}", flush=True)


if __name__ == "__main__":
    main()
