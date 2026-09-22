"""Replay retained avoidance failures; do not search scenes or change acceptance."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import generate_office_behavior_scenarios_v10 as v10
from generate_office_behavior_scenarios_v6 import GATE_CELLS, NOMINAL_GATE_PAIR
from matplotlib.patches import Rectangle

v9 = v10.v9
plt = v10.plt


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _saved_cases(batch):
    summary = _read(batch / "batch_summary.json")
    cases = []
    for group in summary["groups"]:
        if group["behavior"] != "avoidance":
            continue
        directory = batch / f"seed_{group['seed_offset']}" / "avoidance"
        retained = sorted((directory / "visibility_failures").glob("test_avoidance_*.json"))
        if retained:
            for path in retained:
                cases.append((f"seed_{group['seed_offset']}_{path.stem}", _read(path), str(path)))
        else:
            # Historical batches kept one raw example, not all rejected cases.
            audit = _read(directory / "test_avoidance_search.json")
            raw = audit.get("examples", {}).get("primary_not_visible")
            if raw:
                cases.append((f"seed_{group['seed_offset']}_legacy_example",
                              {"obstacles": raw}, str(directory / "test_avoidance_search.json")))
    return cases


def _plan_trace(problem, scenario, plan, primary_index, radius):
    if plan is None:
        return []
    trajectories = [v9.obstacle_positions(spec, plan.steps) for spec in scenario.obstacles]
    trace = []
    for t, position in enumerate(plan.positions):
        primary = trajectories[primary_index][t]
        next_cell = plan.positions[t + 1] if t < plan.steps else None
        colliders = [] if next_cell is None else [
            index for index, cells in enumerate(trajectories)
            if next_cell in (cells[t], cells[t + 1])
        ]
        trace.append({
            "time": t, "position": list(position), "primary_position": list(primary),
            "distance": v9.chebyshev(position, primary),
            "primary_visible": v9.chebyshev(position, primary) <= radius,
            "next_cell": next_cell, "next_action_collision_indices": colliders,
        })
    return trace


def _assert_safe_plan(problem, plan, trace, blocked):
    if plan is None:
        return
    if plan.positions[0] != problem.start or plan.positions[-1] != problem.goal:
        raise ValueError("Diagnostic plan has incorrect endpoints.")
    if plan.steps != len(plan.positions) - 1 or plan.wait_count != 0:
        raise ValueError("Diagnostic no-wait plan has inconsistent length or waits.")
    for t, position in enumerate(plan.positions):
        if position in problem.obstacles or position in blocked or not all(
            0 <= coordinate < problem.size for coordinate in position
        ):
            raise ValueError("Diagnostic plan enters a blocked/out-of-bounds cell.")
        if t and sum(abs(a - b) for a, b in zip(position, plan.positions[t - 1])) != 1:
            raise ValueError("Diagnostic plan contains a non-cardinal action.")
    if any(row["next_action_collision_indices"] for row in trace):
        raise ValueError("Diagnostic plan failed independent collision replay.")


def diagnose_case(problem, design, raw, lookup):
    specs = tuple(v10.DynamicObstacleSpec(
        route=tuple(tuple(cell) for cell in item["route"]),
        start_index=int(item["start_index"]), direction=int(item["direction"]),
        move_every=int(item["move_every"]), label=item["route_id"],
        reference_path_source=lookup[item["route_id"]]["critical_zone"],
    ) for item in raw["obstacles"])
    if len(specs) != 5:
        raise ValueError("Expected the original five-obstacle scene.")
    for spec in specs:
        if spec.route != tuple(map(tuple, lookup[spec.label]["route"])):
            raise ValueError("Saved example route disagrees with the batch snapshot.")
    scenario = v10.DynamicScenario(seed=0, obstacles=specs)
    classification = v9._classify_candidate(problem, scenario, "avoidance", design)
    if classification is None:
        raise ValueError("Saved failure no longer passes the original avoidance definition.")
    # Constructive search requires exactly one nominal-colliding primary, with
    # four nominal-safe covariates. Recover that primary in old raw examples.
    primaries = [index for index, spec in enumerate(specs) if v9._nominal_path_collides(
        problem, v10.DynamicScenario(seed=0, obstacles=(spec,))
    )]
    if len(primaries) != 1:
        raise ValueError("Cannot unambiguously recover the constructive primary.")
    index = primaries[0]
    if "primary_obstacle_index" in raw and int(raw["primary_obstacle_index"]) != index:
        raise ValueError("Recorded primary disagrees with the reconstructed primary.")
    reduced = v10._classify_without_obstacle(problem, scenario, index, design)
    if reduced is None:
        raise ValueError("Saved failure no longer passes primary-removal normal check.")
    radius = int(design["observation_radius"])
    original = classification["best_plan"]
    legacy_step = v9._decision_step(problem, original, "avoidance")
    legacy_visible = v9._decision_observability(
        problem, scenario, original, "avoidance", radius, (index,),
    )
    blocked = set().union(*(cells for name, cells in GATE_CELLS.items() if name not in NOMINAL_GATE_PAIR))
    constrained = v9.shortest_safe_plan(
        problem, scenario, allow_wait=False, additionally_blocked=blocked,
        visible_deviation_obstacle_index=index, observation_radius=radius,
    )
    original_trace = _plan_trace(problem, scenario, original, index, radius)
    constrained_trace = _plan_trace(problem, scenario, constrained, index, radius)
    _assert_safe_plan(problem, original, original_trace, set())
    _assert_safe_plan(problem, constrained, constrained_trace, blocked)
    visible = None if constrained is None else v9._decision_observability(
        problem, scenario, constrained, "avoidance", radius, (index,),
    )
    if constrained is not None and visible is None:
        raise ValueError("Constrained witness failed the original visibility check.")
    extra = None if constrained is None else constrained.steps - original.steps
    if extra is not None and extra < 0:
        raise ValueError("Constrained plan cannot be shorter than the unrestricted optimum.")
    outcome = (
        "legacy_visibility_now_passes" if legacy_visible is not None else
        "no_visible_departure_plan_under_constraints" if constrained is None else
        "equal_cost_visible_witness" if extra == 0 else
        "visible_departure_requires_extra_steps"
    )
    return {
        "outcome": outcome, "primary_obstacle_index": index,
        "primary_route_id": specs[index].label,
        "primary_zone": specs[index].reference_path_source,
        "observation_radius": radius, "original_steps": original.steps,
        "constrained_steps": None if constrained is None else constrained.steps,
        "extra_steps": extra, "legacy_decision_step": legacy_step,
        "legacy_observation": None if legacy_step is None else original_trace[legacy_step - 1],
        "constrained_observation": visible,
        "removal_normal_steps": reduced["best_plan"].steps,
        "nominal_gate_cost": classification["nominal_gate_cost"],
        "best_alternative_gate_cost": classification["best_alternative_gate_cost"],
        "obstacles": raw["obstacles"], "nominal_path": list(problem.nominal_path),
        "original_path": list(original.positions),
        "constrained_path": None if constrained is None else list(constrained.positions),
        "original_trace": original_trace, "constrained_trace": constrained_trace,
        "scope": "same nominal prefix until visible first deviation; no STAY; nominal gate pair only",
        "caveat": "Oracle witness is not proof of a learnable partially observed policy or a phase-matched pair.",
    }, scenario


def render_case(problem, report, scenario, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    legacy_t = max(0, (report["legacy_decision_step"] or 1) - 1)
    witness = report["constrained_observation"]
    witness_t = max(0, int(witness["decision_step"]) - 1) if witness else legacy_t
    for axis, path, t, title in (
        (axes[0], report["original_path"], legacy_t, "Original shortest path"),
        (axes[1], report["constrained_path"], witness_t, "Visible-departure shortest path"),
    ):
        v9._draw_base(axis, problem, f"{title}\nobservation time={t}")
        for index, spec in enumerate(scenario.obstacles):
            color = "#c62828" if index == report["primary_obstacle_index"] else "#999999"
            axis.plot([c[1] for c in spec.route], [c[0] for c in spec.route], color=color, linewidth=2)
            cell = v9.obstacle_positions(spec, t)[t]
            axis.scatter(cell[1], cell[0], color=color, s=28, zorder=6)
        if path is None:
            axis.text(0.5, 0.5, "No safe path under these constraints", transform=axis.transAxes,
                      ha="center", bbox={"facecolor": "white", "alpha": 0.9})
            continue
        axis.plot([c[1] for c in path], [c[0] for c in path], "--", color="#1976d2")
        position = path[t]
        radius = report["observation_radius"]
        axis.add_patch(Rectangle((position[1] - radius - 0.5, position[0] - radius - 0.5),
                                 2 * radius + 1, 2 * radius + 1,
                                 facecolor="#ffb300", edgecolor="#ff8f00", alpha=0.18))
        axis.scatter(position[1], position[0], marker="s", color="#ff8f00", s=35, zorder=7)
    fig.suptitle(f"{report['case_id']} | {report['outcome']}\n"
                 f"original={report['original_steps']} steps, constrained={report['constrained_steps']} steps")
    fig.text(0.5, 0.02, "Red: primary at observation time | Orange square: observation window | Dashed: oracle, not policy", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    batch = v9._resolve(args.batch_dir)
    context = _read(batch / "batch_context.json")
    config = context["config"]
    problem = v9._load_problem(config)
    if problem.grid_sha256 != context["grid_sha256"]:
        raise ValueError("Static map changed since this batch; refusing a mismatched replay.")
    design = v9._scenario_design(config)
    lookup = {row["route_id"]: row for row in context["test_route_pool"]["corridor"]}
    cases = _saved_cases(batch)
    if not cases:
        raise ValueError("No retained avoidance visibility failures were found.")
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000000:09d}"
    output = batch / "visibility_diagnostics" / stamp
    v10.write_json({"source_batch": str(batch), "config": config,
                    "grid_sha256": problem.grid_sha256,
                    "retained_case_count": len(cases),
                    "note": "Retained examples only; not all 59 rejected candidates, no new random search."},
                   output / "context.json")
    reports = []
    status = "running"
    try:
        for case_id, raw, source in cases:
            print(f"Visibility replay: {case_id}", flush=True)
            report, scenario = diagnose_case(problem, design, raw, lookup)
            report.update(case_id=case_id, source_file=source)
            v10.write_json(report, output / f"{case_id}.json")
            reports.append(report)
            render_case(problem, report, scenario, output / f"{case_id}.png")
            print(f"  {report['outcome']}: original={report['original_steps']} "
                  f"constrained={report['constrained_steps']}", flush=True)
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise SystemExit(130) from None
    except Exception:
        status = "error"
        raise
    finally:
        fields = ("case_id", "outcome", "primary_zone", "primary_route_id", "original_steps",
                  "constrained_steps", "extra_steps", "legacy_decision_step")
        v10.write_json({"status": status, "cases_expected": len(cases),
                        "cases_checked": len(reports), "outcomes": dict(Counter(r["outcome"] for r in reports)),
                        "training_ready": False, "acceptance_rules_changed": False,
                        "cases": [{key: r[key] for key in fields} for r in reports]}, output / "summary.json")
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: r[key] for key in fields} for r in reports)
        print(f"Diagnostic {status}; saved to {output}", flush=True)


if __name__ == "__main__":
    main()
