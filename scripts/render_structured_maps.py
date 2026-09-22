from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_problem_layout, render_problem_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render structured static map previews.")
    parser.add_argument("--output-dir", default="maps/previews")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    calibration_problems = load_problem_set(
        ROOT / "maps" / "structured_calibration" / "maps.json"
    )
    hard_calibration_problems = load_problem_set(
        ROOT / "maps" / "structured_calibration_hard" / "maps.json"
    )
    large_calibration_problems = load_problem_set(
        ROOT / "maps" / "structured_calibration_40x40" / "maps.json"
    )
    main_problems = load_problem_set(ROOT / "maps" / "structured_main" / "maps.json")
    problems = [
        *calibration_problems,
        *hard_calibration_problems,
        *large_calibration_problems,
        *main_problems,
    ]
    for problem in problems:
        output = output_dir / f"{problem.map_id}.png"
        render_problem_layout(problem, output)
        print(f"Rendered {output}")
    overview = output_dir / "structured_maps_overview.png"
    render_problem_overview(problems, overview)
    print(f"Rendered {overview}")
    calibration_overview = output_dir / "calibration_maps_overview.png"
    render_problem_overview(calibration_problems, calibration_overview)
    print(f"Rendered {calibration_overview}")
    hard_overview = output_dir / "calibration_hard_maps_overview.png"
    render_problem_overview(hard_calibration_problems, hard_overview)
    print(f"Rendered {hard_overview}")
    large_overview = output_dir / "calibration_40x40_maps_overview.png"
    render_problem_overview(large_calibration_problems, large_overview)
    print(f"Rendered {large_overview}")


if __name__ == "__main__":
    main()
