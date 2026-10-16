from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.envs.dynamic_grid import (
    build_map01_controlled_bottleneck_obstacle_specs,
    build_map01_controlled_six_obstacle_specs,
    build_map01_controlled_mixed_six_obstacle_specs,
    build_map01_three_crossing_obstacle_specs,
)
from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import (
    render_dynamic_obstacle_animation,
    render_problem_layout,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the fixed Map 1 dynamic-obstacle scenario as a GIF."
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "map01_controlled_bottleneck",
            "map01_controlled_six_obstacles",
            "map01_controlled_mixed_six_obstacles",
            "strategy_crossing",
        ),
        default="map01_controlled_mixed_six_obstacles",
        help="Dynamic scenario to preview.",
    )
    parser.add_argument(
        "--output",
        help="Output GIF path. Defaults to the selected scenario preview path.",
    )
    parser.add_argument(
        "--map-output",
        default="maps/previews/calibration_40x40_map_01.png",
        help="Output path for the corresponding static map PNG.",
    )
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--interval-ms", type=int, default=160)
    args = parser.parse_args()

    problems = load_problem_set(ROOT / "maps/structured_calibration_40x40/maps.json")
    problem = problems[0]
    if args.scenario == "map01_controlled_bottleneck":
        specs = build_map01_controlled_bottleneck_obstacle_specs(problem)
    elif args.scenario == "map01_controlled_six_obstacles":
        specs = build_map01_controlled_six_obstacle_specs(problem)
    elif args.scenario == "map01_controlled_mixed_six_obstacles":
        specs = build_map01_controlled_mixed_six_obstacle_specs(problem)
    else:
        specs = build_map01_three_crossing_obstacle_specs(problem)
    output = Path(
        args.output
        or f"maps/previews/calibration_40x40_map_01_{args.scenario}.gif"
    )
    if not output.is_absolute():
        output = ROOT / output
    map_output = Path(args.map_output)
    if not map_output.is_absolute():
        map_output = ROOT / map_output
    render_problem_layout(problem, map_output)
    render_dynamic_obstacle_animation(
        problem,
        specs,
        output,
        frames=args.frames,
        interval_ms=args.interval_ms,
    )
    print(f"Saved static map preview to {map_output}")
    print(f"Saved dynamic obstacle animation to {output}")


if __name__ == "__main__":
    main()
