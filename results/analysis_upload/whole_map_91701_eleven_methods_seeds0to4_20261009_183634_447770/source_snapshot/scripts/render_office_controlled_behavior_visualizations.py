"""Render maps and policy GIFs for the controlled Office study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from astar_d3qn.maps.io import load_problem_set
from astar_d3qn.maps.render import render_dynamic_obstacle_animation, render_problem_layout
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json
import render_office_strategy_visualizations as render
from run_office_controlled_behavior_study import MAIN_CONFIG, METHODS
from run_office_controlled_blockage_diagnostic import _build_scenarios
from train_dynamic_spatial_generalization import _scenario_schedules


COLORS = {
    "uniform": "#2563eb",
    "prefill": "#f59e0b",
    "persistent": "#059669",
    "scheduled_decay": "#ca8a04",
    "ca_current": "#a855f7",
    "ca_predictive": "#ef4444",
    "local_conflict": "#0891b2",
    "ca_adaptive_decay": "#991b1b",
    "local_counterexample": "#7e22ce",
    "predictive_margin": "#db2777",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _configure_renderer() -> None:
    configured = tuple(
        render.MethodSpec(
            method.key,
            method.label,
            method.config_path,
            method.strategy,
            COLORS[method.key],
        )
        for method in METHODS
    )
    render.METHODS = configured
    render.METHOD_BY_KEY = {method.key: method for method in configured}


def _problem_and_schedules():
    config = load_config(_resolve(MAIN_CONFIG))
    problems = load_problem_set(_resolve(config["map_sets"]["train"]["file"]))
    matches = [problem for problem in problems if problem.map_id == config["map"]["scene"]]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one Office problem.")
    problem = matches[0]
    manifest = load_json(_resolve(config["spatial_generalization"]["manifest"]))
    schedules, _ = _scenario_schedules(
        problem, manifest, config["spatial_generalization"]
    )
    return problem, schedules


def _select(scenarios, scenario_id: int, split: str):
    matches = [scenario for scenario in scenarios if scenario.seed == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Scenario {scenario_id} is not in logical {split} split.")
    return matches[0]


def _record(rollout):
    return {
        "method": rollout.strategy,
        "success": rollout.success,
        "safe_success": rollout.success and not rollout.collisions,
        "steps": rollout.steps,
        "collision_count": len(rollout.collisions),
        "wait_steps": rollout.wait_steps,
        "path": [list(cell) for cell in rollout.path],
    }


def _render_policy_scenario(
    problem,
    scenario,
    *,
    name: str,
    title: str,
    training_seed: int,
    device: str,
    interval_ms: int,
    max_frames: int,
    output_dir: Path,
    layouts_only: bool = False,
):
    render._render_scenario_layout(
        problem, scenario, output_dir / f"{name}_layout.png", title
    )
    render_dynamic_obstacle_animation(
        problem,
        scenario.obstacles,
        output_dir / f"{name}_obstacles.gif",
        frames=32,
        interval_ms=interval_ms,
    )
    if layouts_only:
        return []
    rollouts = render._load_rollouts(problem, scenario, training_seed, device)
    render._render_final_paths(
        problem,
        scenario,
        rollouts,
        output_dir / f"{name}_all_strategies.png",
        f"{title} | training seed {training_seed}",
    )
    render._render_rollout_animation(
        problem,
        scenario,
        rollouts,
        output_dir / f"{name}_all_strategies.gif",
        interval_ms,
        f"{len(METHODS)} frozen policies | {title} | seed {training_seed}",
        max_frames,
    )
    return [_record(rollout) for rollout in rollouts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--interval-ms", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=140)
    parser.add_argument("--layouts-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="outputs/office_controlled_behavior_visualizations_v1",
    )
    parser.add_argument(
        "--diagnostic-dir",
        default="outputs/office_controlled_blockage_diagnostic_v1",
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_renderer()
    problem, schedules = _problem_and_schedules()
    render_problem_layout(problem, output_dir / "office_static_map.png")

    training = _select(schedules["train"], 40, "train")
    validation = _select(schedules["validation"], 0, "validation")
    render._render_scenario_layout(
        problem,
        training,
        output_dir / "training_scenario_40_layout.png",
        "Training scenario 40 | local avoidance",
    )
    render_dynamic_obstacle_animation(
        problem,
        training.obstacles,
        output_dir / "training_scenario_40_obstacles.gif",
        frames=32,
        interval_ms=args.interval_ms,
    )
    render._render_scenario_layout(
        problem,
        validation,
        output_dir / "validation_scenario_0_layout.png",
        "Validation scenario 0 | seen training distribution | wait",
    )

    records = {}
    for behavior, scenario_id in (("wait", 8), ("avoidance", 46), ("reroute", 76)):
        scenario = _select(schedules["test"], scenario_id, "test")
        name = f"test_{behavior}_scenario_{scenario_id}"
        records[name] = _render_policy_scenario(
            problem,
            scenario,
            name=name,
            title=f"In-distribution test scenario {scenario_id} | {behavior}",
            training_seed=args.training_seed,
            device=args.device,
            interval_ms=args.interval_ms,
            max_frames=args.max_frames,
            output_dir=output_dir,
            layouts_only=args.layouts_only,
        )

    diagnostic_dir = _resolve(args.diagnostic_dir)
    controlled = _build_scenarios(4, diagnostic_dir)[0]
    controlled_problem, controlled_scenario, metadata = controlled
    records["controlled_blockage_60000"] = _render_policy_scenario(
        controlled_problem,
        controlled_scenario,
        name="controlled_blockage_scenario_60000",
        title="Controlled blockage | one obstacle | start four steps before conflict",
        training_seed=args.training_seed,
        device=args.device,
        interval_ms=args.interval_ms,
        max_frames=args.max_frames,
        output_dir=output_dir,
        layouts_only=args.layouts_only,
    )
    write_json(
        {
            "training_seed": args.training_seed,
            "evaluation_design": "in_distribution_seen_training_scenarios",
            "training_scenario_id": training.seed,
            "validation_scenario_id": validation.seed,
            "controlled_blockage": metadata,
            "rollouts": records,
        },
        output_dir / "visualization_metadata.json",
    )
    print(f"Saved controlled-study visualizations to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
