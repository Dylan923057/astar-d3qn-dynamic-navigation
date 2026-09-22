"""Regression checks for candidate selection and independent primary trials."""

from __future__ import annotations

import random
import json
import sys
import unittest
from types import SimpleNamespace
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_office_behavior_scenarios_v10 as generator
from astar_d3qn.maps.problem import NavigationProblem


class OfficeV10GenerationTests(unittest.TestCase):
    def test_visible_avoidance_records_extra_cost_without_replacing_optimum(self):
        problem = NavigationProblem(
            map_id="test", seed=0, size=5, start=(2, 0), goal=(2, 4),
            obstacles=frozenset(((1, 1), (3, 1))),
            nominal_path=((2, 0), (2, 1), (2, 2), (2, 3), (2, 4)),
        )
        scenario = generator.DynamicScenario(seed=0, obstacles=(generator.DynamicObstacleSpec(
            route=((1, 2), (2, 2)), start_index=0, direction=1,
        ),))
        original = generator.v9.shortest_safe_plan(problem, scenario, allow_wait=False)
        classification = {"best_plan": original}
        evidence = generator._visible_avoidance_evidence(
            problem, scenario, classification, 0, {"observation_radius": 1},
        )
        self.assertEqual(evidence["steps"], 10)
        self.assertEqual(evidence["extra_steps"], 2)
        self.assertIsNone(evidence["extra_steps_limit"])
        self.assertEqual(classification["best_plan"].steps, 8)
        self.assertEqual(evidence["observation"]["nearest_dynamic_distance_at_decision"], 1)

    def test_visible_avoidance_rejects_no_feasible_path_and_bad_replay(self):
        with patch.object(generator.v9, "shortest_safe_plan", return_value=None):
            self.assertIsNone(generator._visible_avoidance_evidence(None, None, {}, 0, {"observation_radius": 7}))
        invalid = generator.v9.SafePlan(((0, 0), (0, 0)), 1, 1)
        with patch.object(generator.v9, "shortest_safe_plan", return_value=invalid), patch.object(
            generator.v9, "_decision_observability", return_value={"decision_step": 1},
        ):
            with self.assertRaisesRegex(ValueError, "steps/waits"):
                generator._visible_avoidance_evidence(None, None, {}, 0, {"observation_radius": 7})

    def test_raw_original_observation_does_not_borrow_witness_decision(self):
        scenario = generator.DynamicScenario(seed=0, obstacles=(generator.DynamicObstacleSpec(
            route=((0, 9), (1, 9)),
        ),))
        plan = generator.v9.SafePlan(((0, 0), (1, 0)), 1, 0)
        with patch.object(generator.v9, "_decision_step", return_value=1):
            observation = generator._raw_plan_observation(None, scenario, plan, "avoidance", 0)
        self.assertEqual(observation["decision_position"], [0, 0])
        self.assertEqual(observation["nearest_dynamic_distance_at_decision"], 9)
        self.assertIsNone(observation["observable_decisive_obstacle_index"])

    def test_visible_pilot_revalidates_both_separate_records(self):
        plan = generator.v9.SafePlan(((0, 0), (0, 1)), 1, 0)
        candidate = {"required_behavior": "avoidance", "primary_start_index": 0,
                     "primary_obstacle_index": 2, "acceptance_rule": "visible_avoidance_feasibility_v1",
                     "visible_avoidance": {"steps": 3, "extra_steps": 2},
                     "full_information_behavior": "avoidance", "legacy_oracle_observable": False,
                     "minimum_safe_path_steps": 1, "oracle_path": [[0, 0], [0, 1]],
                     "_matched_control": {"required_behavior": "normal", "primary_start_index": 1},
                     "_matched_control_signature": ()}
        with ExitStack() as stack:
            stack.enter_context(patch.object(generator, "_validate_pair"))
            stack.enter_context(patch.object(generator, "_scenario_from_record", return_value=None))
            stack.enter_context(patch.object(generator.v9, "_classify_candidate", return_value={"best_plan": plan}))
            stack.enter_context(patch.object(generator, "_classify_without_obstacle", return_value={}))
            stack.enter_context(patch.object(generator.v9, "_decision_observability", return_value=None))
            evidence = stack.enter_context(patch.object(generator, "_visible_avoidance_evidence", return_value={"steps": 3, "extra_steps": 2}))
            design = {"_visible_avoidance_pilot": True, "observation_radius": 7}
            conflict, _ = generator._verified_pilot_pair(None, candidate, "avoidance", "p", 0, design, {})
            self.assertEqual(conflict["minimum_safe_path_steps"], 1)
            self.assertEqual(conflict["visible_avoidance"]["steps"], 3)
            self.assertIn("_matched_control", candidate)
            evidence.return_value = {"steps": 4, "extra_steps": 3}
            with self.assertRaisesRegex(ValueError, "evidence failed"):
                generator._verified_pilot_pair(None, candidate, "avoidance", "p", 0, design, {})
            with self.assertRaisesRegex(ValueError, "visibility revalidation"):
                generator._verified_pilot_pair(None, candidate, "avoidance", "p", 0,
                                              {"observation_radius": 7}, {})

    def test_saved_pair_recovery_excludes_unpaired_examples(self):
        with TemporaryDirectory() as folder:
            design = {"_search_audit_dir": folder}
            generator._save_candidate({"pair": 1}, design, "test", "wait", 1)
            generator._save_candidate({"unpaired": True}, design, "test", "wait_unpaired_verified", 1)
            generator._save_candidate({"different_behavior": True}, design, "test", "reroute", 1)
            self.assertEqual(generator._read_saved_pairs(Path(folder), "wait"), [{"pair": 1}])

    def test_time_budget_stops_before_next_candidate_and_preserves_audit(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            spec = generator.DynamicObstacleSpec(route=((0, 0), (0, 1)))
            stack.enter_context(patch.object(generator, "_route_options_for_speeds", return_value=[(spec, 0)]))
            stack.enter_context(patch.object(generator.v9, "_candidate_combinations", return_value=[(0,)]))
            stack.enter_context(patch.object(generator.time, "monotonic", side_effect=[0, 2, 2, 2]))
            design = {"move_every_choices": [1], "max_search_attempts_per_candidate": 100,
                      "_search_seconds": 1, "_search_audit_dir": folder}
            with self.assertRaisesRegex(RuntimeError, "time budget reached"):
                generator._build_pool(None, "test", "wait", 1, [{}], [()], design,
                                      random.Random(0), set(), require_pair=True)
            audit = json.loads((Path(folder) / "test_wait_search.json").read_text())
            self.assertEqual(audit["attempts"], 0)
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(audit["elapsed_seconds"], 2)

    def test_batch_continues_after_incomplete_behavior_groups(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            problem = SimpleNamespace(map_id="office", grid_sha256="hash")
            pools = {"test": {"corridor": []}}
            stack.enter_context(patch.object(generator, "_prepare_generation", return_value=({}, [], pools)))
            stack.enter_context(patch.object(generator.v9, "render_route_pools"))
            search = stack.enter_context(patch.object(generator, "_build_pool", side_effect=RuntimeError("budget")))
            report = stack.enter_context(patch.object(generator, "_write_batch_report"))
            with self.assertRaises(SystemExit) as stopped:
                generator.run_pair_batch_pilot(
                    problem, {"spatial_generalization": {"behavior_search_seed": 1}},
                    Path(folder), 5, 2, [0], 1,
                )
            self.assertEqual(stopped.exception.code, 2)
            self.assertEqual(search.call_count, 3)
            self.assertEqual([call.args[2] for call in search.call_args_list], list(generator.CAUSAL_BEHAVIORS))
            self.assertEqual(report.call_args.args[-1], "incomplete")
            self.assertTrue(all(group["status"] == "search_incomplete" for group in report.call_args.args[3]))

    def test_pair_validation_rejects_extra_obstacles(self):
        with self.assertRaisesRegex(ValueError, "five obstacles"):
            generator._validate_pair({"obstacles": [{}] * 6}, {"obstacles": [{}] * 5}, {})

    def test_phase_preflight_requires_safe_control_and_preserves_motion(self):
        route = ((0, 0), (0, 1), (0, 2))
        records = [{"route_id": "r", "critical_zone": "upper_left_gate", "route": route}]
        options = [[
            (generator.DynamicObstacleSpec(route=route, label="r", start_index=phase,
                                           direction=direction, move_every=speed),
             3 if phase == 0 else 1)
            for phase, direction, speed in ((0, 1, 1), (1, 1, 1), (2, -1, 1), (2, 1, 2))
        ]]
        design = {"require_primary_demo_conflict": True,
                  "require_strict_demo_conflict_reduction_in_pair": True}
        with TemporaryDirectory() as folder, patch.object(
            generator.v9, "_nominal_path_collides",
            side_effect=lambda problem, scenario: scenario.obstacles[0].start_index == 0,
        ):
            safe, primaries = generator._preflight_phase_pairs(None, records, options, design, folder)
            report = json.loads((Path(folder) / "route_phase_preflight.json").read_text())
        self.assertEqual(len(safe[0]), 3)
        self.assertEqual(len(primaries[0]), 1)
        pairs = report["routes"][0]["phase_pairs_passing_necessary_conditions"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["control_start_index"], 1)
        self.assertFalse(report["wait_behavior_verified"])
        with patch.object(generator.v9, "_nominal_path_collides", return_value=True):
            safe, primaries = generator._preflight_phase_pairs(None, records, options, design, None)
        self.assertEqual(safe[0], [])
        self.assertEqual(primaries[0], [])

    def test_zero_progress_stops_and_saves_actual_rejection_reason(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            spec = generator.DynamicObstacleSpec(route=((0, 0), (0, 1)))
            stack.enter_context(patch.object(
                generator, "_route_options_for_speeds", return_value=[(spec, 0)],
            ))
            stack.enter_context(patch.object(
                generator.v9, "_candidate_combinations", return_value=[(0,)],
            ))
            design = {
                "move_every_choices": [1],
                "max_search_attempts_per_candidate": 100,
                "max_attempts_without_acceptance": 2,
                "_search_audit_dir": folder,
            }
            with self.assertRaisesRegex(RuntimeError, "without_acceptance=2"):
                generator._build_pool(
                    None, "test", "wait", 20, [{}], [()], design,
                    random.Random(0), set(), require_pair=True,
                )
            audit = json.loads((Path(folder) / "test_wait_search.json").read_text())
            self.assertEqual(audit["attempts"], 2)
            self.assertEqual(audit["accepted"], 0)
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(audit["scenario_rejections"], {"no_astar_demo_conflict": 2})
            self.assertEqual(audit["phase_trials"], {})

    def test_one_observed_zone_keeps_behavior_and_difficulty_quotas(self):
        pools = {
            behavior: [
                {
                    "required_behavior": behavior,
                    "primary_critical_zone": "upper_left_gate",
                    "difficulty_score": float(index),
                }
                for index in range(11)
            ]
            for behavior in generator.CAUSAL_BEHAVIORS
        }
        selected, audit = generator._select_causal_pools(
            pools,
            {behavior: 5 for behavior in generator.CAUSAL_BEHAVIORS},
            {"easy": 0.40, "medium": 0.35, "hard": 0.25},
            random.Random(1),
        )
        self.assertEqual(len(selected), 15)
        self.assertTrue(audit["limited_primary_zone_coverage"])
        self.assertEqual(audit["observed_candidate_primary_zones"], ["upper_left_gate"])
        for behavior in generator.CAUSAL_BEHAVIORS:
            self.assertEqual(
                Counter(
                    row["difficulty_stratum"]
                    for row in selected
                    if row["required_behavior"] == behavior
                ),
                {"easy": 2, "medium": 2, "hard": 1},
            )

    def test_failed_primary_trial_does_not_reorder_the_next_trial(self):
        # Fail after reordering, once at visibility and once at pair matching.
        for fail_at_pair in (False, True):
            with self.subTest(fail_at_pair=fail_at_pair), ExitStack() as stack:
                specs = tuple(
                    generator.DynamicObstacleSpec(
                        route=((index, 0), (index, 1)), label=f"route_{index}"
                    )
                    for index in range(5)
                )
                records = [{"index": index} for index in range(5)]
                design = {
                    "move_every_choices": [1],
                    "max_search_attempts_per_candidate": 1,
                    "require_primary_demo_conflict": True,
                    "primary_obstacle_index": 2,
                    "observation_radius": 7,
                }
                rng = random.Random(1)
                stack.enter_context(patch.object(rng, "shuffle", lambda values: None))
                stack.enter_context(patch.object(
                    generator, "_route_options_for_speeds",
                    side_effect=lambda record, *_: [(specs[record["index"]], 1)],
                ))
                stack.enter_context(patch.object(
                    generator.v9, "_candidate_combinations", return_value=[tuple(range(5))],
                ))
                plan = type("Plan", (), {"steps": 4})()
                classification = {"best_plan": plan}
                stack.enter_context(patch.object(
                    generator.v9, "_classify_candidate", return_value=classification,
                ))
                reduced_calls = []

                def reduced(problem, scenario, index, design):
                    reduced_calls.append((tuple(spec.label for spec in scenario.obstacles), index))
                    return classification

                stack.enter_context(patch.object(
                    generator, "_classify_without_obstacle", side_effect=reduced,
                ))
                stack.enter_context(patch.object(
                    generator.v9, "_decision_observability",
                    side_effect=([{}, {}] if fail_at_pair else [None, {}]),
                ))
                stack.enter_context(patch.object(generator, "_causal_relevance", return_value=[]))
                captured = []

                def record(scenario, *args, **kwargs):
                    captured.append(scenario.obstacles[kwargs["primary_index"]].label)
                    return {}

                stack.enter_context(patch.object(
                    generator, "_record_with_causal_metadata", side_effect=record,
                ))
                match_calls = []

                def match(*args, **kwargs):
                    scenario, primary_index, options = args[1], args[4], args[5]
                    match_calls.append((scenario.obstacles[primary_index].label, options[0][0].label))
                    if len(match_calls) == 1:
                        return None
                    return {}, (("control", 0, 1, 1),)

                stack.enter_context(patch.object(generator, "_matched_control", side_effect=match))
                result = generator._build_pool(
                    None, "train", "wait", 1, records, [()], design, rng, set(),
                    require_pair=fail_at_pair,
                )
                self.assertEqual(len(result), 1)
                original_labels = tuple(spec.label for spec in specs)
                self.assertEqual(reduced_calls, [(original_labels, 0), (original_labels, 1)])
                self.assertEqual(captured[-1], "route_1")
                if fail_at_pair:
                    self.assertEqual(match_calls, [("route_0", "route_0"), ("route_1", "route_1")])


if __name__ == "__main__":
    unittest.main()
