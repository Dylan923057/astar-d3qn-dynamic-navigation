from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.maps.io import save_problem_set
from astar_d3qn.maps.structured import build_structured_problem_set


GROUPS = {
    "calibration": tuple(
        f"calibration_20x20_map_{map_number:02d}" for map_number in range(1, 6)
    ),
    "calibration_hard": tuple(
        f"calibration_hard_20x20_map_{map_number:02d}" for map_number in range(1, 6)
    ),
    "calibration_40x40": tuple(
        f"calibration_40x40_map_{map_number:02d}" for map_number in range(1, 6)
    ),
    "main": ("office_40x40", "parcel_station_40x40", "warehouse_40x40"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic structured static navigation maps."
    )
    parser.add_argument("--group", choices=tuple(GROUPS), default="calibration")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = GROUPS[args.group]
    output = (
        Path(args.output).resolve()
        if args.output
        else ROOT / "maps" / f"structured_{args.group}" / "maps.json"
    )
    problems = build_structured_problem_set(scenes)
    set_version = "v2" if args.group == "calibration_40x40" else "v1"
    save_problem_set(
        problems,
        output,
        metadata={
            "generator": "structured_layout_v1",
            "set": f"structured_{args.group}_{set_version}",
            "scenes": list(scenes),
            "independent_maps": len(problems) > 1,
        },
    )
    print(f"Generated {len(problems)} structured map(s) in {output}")
    for problem in problems:
        print(
            f"  {problem.map_id}: size={problem.size}, obstacles={len(problem.obstacles)}, "
            f"A*={problem.astar_steps}, turns={problem.metadata['turn_count']}, "
            f"detour={problem.metadata['detour_ratio']:.3f}, "
            f"bottleneck={problem.metadata['bottleneck_width']}"
        )


if __name__ == "__main__":
    main()
