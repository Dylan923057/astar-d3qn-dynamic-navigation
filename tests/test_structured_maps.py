from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.maps.io import load_problem_set, save_problem_set
from astar_d3qn.core.astar import randomized_tie_astar_path
from astar_d3qn.maps.structured import (
    build_structured_problem,
    build_structured_problem_set,
)


class StructuredMapTests(unittest.TestCase):
    def test_registered_scenes_are_reproducible_and_solvable(self) -> None:
        scenes = (
            *(f"calibration_20x20_map_{number:02d}" for number in range(1, 6)),
            "office_40x40",
            "parcel_station_40x40",
            "warehouse_40x40",
        )
        first = build_structured_problem_set(scenes)
        second = build_structured_problem_set(scenes)
        self.assertEqual(
            [problem.grid_sha256 for problem in first],
            [problem.grid_sha256 for problem in second],
        )
        for problem in first:
            self.assertGreater(problem.astar_steps, 0)
            self.assertTrue(all(cell not in problem.obstacles for cell in problem.nominal_path))
            self.assertEqual(problem.metadata["manhattan_steps"] + problem.metadata["detour_steps"], problem.astar_steps)
            self.assertGreaterEqual(problem.metadata["turn_count"], 1)

    def test_five_calibration_maps_are_independent_fixed_problems(self) -> None:
        scenes = tuple(
            f"calibration_20x20_map_{number:02d}" for number in range(1, 6)
        )
        problems = build_structured_problem_set(scenes)
        self.assertEqual(len(problems), 5)
        self.assertEqual(len({problem.grid_sha256 for problem in problems}), 5)
        for problem in problems:
            self.assertEqual(problem.start, (1, 1))
            self.assertEqual(problem.goal, (19, 19))
            self.assertEqual(len(problem.obstacles), 120)
            self.assertAlmostEqual(problem.density, 0.30)
            self.assertLessEqual(problem.metadata["largest_obstacle_component"], 9)
            self.assertGreaterEqual(problem.metadata["sampled_optimal_path_variants"], 80)
            self.assertGreaterEqual(problem.metadata["sampled_optimal_path_union_cells"], 100)

    def test_hard_calibration_maps_add_detours_without_single_routes(self) -> None:
        scenes = tuple(
            f"calibration_hard_20x20_map_{number:02d}"
            for number in range(1, 6)
        )
        problems = build_structured_problem_set(scenes)
        self.assertEqual(len(problems), 5)
        self.assertEqual(len({problem.grid_sha256 for problem in problems}), 5)
        for problem in problems:
            self.assertEqual(problem.start, (1, 1))
            self.assertEqual(problem.goal, (19, 19))
            self.assertEqual(len(problem.obstacles), 120)
            self.assertAlmostEqual(problem.density, 0.30)
            self.assertGreaterEqual(problem.metadata["detour_steps"], 4)
            self.assertLessEqual(problem.metadata["detour_steps"], 14)
            self.assertGreaterEqual(problem.metadata["turn_count"], 8)
            self.assertLessEqual(problem.metadata["largest_obstacle_component"], 12)
            self.assertGreaterEqual(problem.metadata["sampled_optimal_path_variants"], 40)
            self.assertGreaterEqual(problem.metadata["sampled_optimal_path_union_cells"], 90)

    def test_40x40_calibration_maps_use_uniform_scattered_obstacles(self) -> None:
        scenes = tuple(
            f"calibration_40x40_map_{number:02d}" for number in range(1, 6)
        )
        problems = build_structured_problem_set(scenes)
        self.assertEqual(len(problems), 5)
        self.assertEqual(len({problem.grid_sha256 for problem in problems}), 5)
        for problem in problems:
            self.assertEqual(problem.size, 40)
            self.assertEqual(problem.start, (1, 1))
            self.assertEqual(problem.goal, (39, 39))
            self.assertEqual(len(problem.obstacles), 360)
            self.assertAlmostEqual(problem.density, 0.225)
            self.assertGreaterEqual(problem.metadata["detour_steps"], 0)
            self.assertLessEqual(problem.metadata["detour_steps"], 8)
            self.assertGreaterEqual(problem.metadata["turn_count"], 10)
            self.assertLessEqual(problem.metadata["turn_count"], 30)
            self.assertLessEqual(problem.metadata["largest_obstacle_component"], 24)
            self.assertGreaterEqual(problem.metadata["sampled_optimal_path_variants"], 50)
            self.assertGreaterEqual(problem.metadata["sampled_optimal_path_union_cells"], 180)
            self.assertEqual(problem.metadata["spatial_bins"], 4)
            self.assertEqual(problem.metadata["spatial_bin_size"], 10)
            self.assertTrue(
                all(count in {22, 23} for count in problem.metadata["spatial_bin_counts"])
            )
            self.assertEqual(sum(problem.metadata["spatial_bin_counts"]), 360)
            self.assertTrue(all(cell not in problem.obstacles for cell in problem.nominal_path))

    def test_paths_use_the_registered_structural_openings(self) -> None:
        calibration = build_structured_problem("calibration_20x20")
        self.assertEqual(calibration.start, (1, 1))
        self.assertEqual(calibration.goal, (19, 19))
        self.assertEqual(calibration.metadata["layout_version"], 2)
        self.assertEqual(calibration.metadata["structure"], "scattered_mixed_shape_obstacles")
        self.assertGreaterEqual(calibration.metadata["obstacle_shape_count"], 4)
        self.assertGreaterEqual(calibration.metadata["obstacle_components"], 25)
        sampled_paths = {
            tuple(
                randomized_tie_astar_path(
                    calibration.start,
                    calibration.goal,
                    calibration.obstacles,
                    calibration.size,
                    random.Random(seed),
                )
                or ()
            )
            for seed in range(100)
        }
        sampled_paths.discard(())
        self.assertGreaterEqual(len(sampled_paths), 80)

        office = build_structured_problem("office_40x40")
        self.assertTrue({(10, 9), (11, 9), (25, 17), (26, 17)}.issubset(office.nominal_path))

        parcel = build_structured_problem("parcel_station_40x40")
        self.assertTrue({(8, 28), (9, 28)}.issubset(parcel.nominal_path))

        warehouse = build_structured_problem("warehouse_40x40")
        self.assertTrue({(4, 11), (35, 21)}.issubset(warehouse.nominal_path))

    def test_structured_manifest_round_trip_preserves_metadata(self) -> None:
        problems = build_structured_problem_set(("office_40x40", "warehouse_40x40"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "maps.json"
            save_problem_set(problems, path, metadata={"set": "test"})
            restored = load_problem_set(path)
        self.assertEqual([item.map_id for item in restored], [item.map_id for item in problems])
        self.assertEqual(
            [item.metadata["structure"] for item in restored],
            [item.metadata["structure"] for item in problems],
        )
        self.assertEqual(
            [item.grid_sha256 for item in restored],
            [item.grid_sha256 for item in problems],
        )


if __name__ == "__main__":
    unittest.main()
