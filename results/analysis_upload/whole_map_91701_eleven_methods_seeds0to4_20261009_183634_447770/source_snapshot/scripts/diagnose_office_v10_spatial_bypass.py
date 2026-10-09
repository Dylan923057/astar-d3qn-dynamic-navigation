"""Check saved avoidance pairs for visible, cycle-free spatial bypass witnesses."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from dataclasses import asdict

import diagnose_office_v10_visibility as visibility
from astar_d3qn.evaluation.spatial_witness import motion_summary, shortest_simple_visible_plan

v10, v9, plt = visibility.v10, visibility.v9, visibility.plt


def primary_conflicts(problem, scenario, index):
    positions = v9.obstacle_positions(scenario.obstacles[index], len(problem.nominal_path) - 1)
    return [{"step": t, "cell": list(cell)} for t, cell in enumerate(problem.nominal_path)
            if t and cell in (positions[t - 1], positions[t])]


def audit_case(problem, design, source, control, lookup, budgets):
    v10._validate_pair(source, control, design)
    scenario = v10._scenario_from_record(source, lookup)
    original = v9._classify_candidate(problem, scenario, "avoidance", design)
    if original is None or v10._classify_without_obstacle(problem, scenario, source["primary_obstacle_index"], design) is None:
        raise ValueError("Saved case no longer passes its original behavior/removal checks.")
    control_scene = v10._scenario_from_record(control, lookup)
    if v9._classify_candidate(problem, control_scene, "normal", design) is None:
        raise ValueError("Saved matched control is not normal.")
    index, radius = int(source["primary_obstacle_index"]), int(design["observation_radius"])
    witness = source["visible_avoidance"]
    if witness != v10._visible_avoidance_evidence(problem, scenario, original, index, design):
        raise ValueError("Saved visible-departure path no longer reproduces.")
    events = primary_conflicts(problem, scenario, index)
    if not events:
        raise ValueError("Primary never collides with the nominal reference.")
    bypass_cell = tuple(events[0]["cell"])
    gate_blocks = set().union(*(cells for name, cells in visibility.GATE_CELLS.items()
                               if name not in visibility.NOMINAL_GATE_PAIR))
    # A constructive sufficient witness, not a universal definition of human
    # avoidance: bypass the earliest nominal primary-conflict cell spatially,
    # without any repeated grid cell or STAY, keeping the original gate pair.
    result = shortest_simple_visible_plan(
        problem, scenario, primary_index=index, observation_radius=radius,
        additionally_blocked=gate_blocks | {bypass_cell}, **budgets,
        progress=lambda expanded, generated, queued: print(
            f"  expanded={expanded} generated={generated} queued={queued}", flush=True),
    )
    plan = result.plan
    trace = visibility._plan_trace(problem, scenario, plan, index, radius)
    visibility._assert_safe_plan(problem, plan, trace, gate_blocks | {bypass_cell})
    observation = None
    if plan is not None:
        if not motion_summary(plan.positions)["is_simple_path"]:
            raise ValueError("Spatial witness contains a repeated cell.")
        observation = v9._decision_observability(problem, scenario, plan, "avoidance", radius, (index,))
        if observation is None:
            raise ValueError("Spatial witness departs before seeing its primary.")
        if plan.steps < witness["steps"]:
            raise ValueError("Spatial constraint cannot improve the less restricted optimum.")
    search_audit = asdict(result)
    search_audit.pop("plan")
    saved_motion = motion_summary(witness["path"])
    return {
        "pair_id": source["pair_id"], "search": search_audit,
        "primary_route_id": source["primary_route_id"], "primary_zone": source["primary_critical_zone"],
        "primary_conflict_events": events, "bypassed_cell": list(bypass_cell),
        "full_information_steps": original["best_plan"].steps,
        "saved_visible_steps": witness["steps"], "saved_visible_path": witness["path"],
        "saved_visible_motion": saved_motion,
        "saved_path_visits_conflict_cell": bypass_cell in set(map(tuple, witness["path"])),
        "saved_observation": witness["observation"],
        "spatial_steps": None if plan is None else plan.steps,
        "extra_over_saved_visible": None if plan is None else plan.steps - witness["steps"],
        "extra_over_full_information": None if plan is None else plan.steps - original["best_plan"].steps,
        "spatial_path": None if plan is None else [list(cell) for cell in plan.positions],
        "spatial_observation": observation, "spatial_trace": trace,
        "spatial_motion": None if plan is None else motion_summary(plan.positions),
        "observation_radius": radius,
        "scope": "visible first departure from nominal prefix; no STAY/repeated cells; nominal gate pair; omit first nominal primary-conflict cell",
        "interpretation": "A found path certifies this spatial-bypass class, not learnability or absence of timing dependence. Resource exhaustion is unknown.",
    }, scenario


def render_case(problem, report, scenario, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    for axis, path, observation, title, color in (
        (axes[0], report["saved_visible_path"], report["saved_observation"], "Saved visible path", "#d17a00"),
        (axes[1], report["spatial_path"], report["spatial_observation"], "Cycle-free spatial bypass", "#168543"),
    ):
        t = observation["decision_step"] - 1 if observation else report["saved_observation"]["decision_step"] - 1
        v9._draw_base(axis, problem, f"{title}\nobservation time={t}")
        for spec in scenario.obstacles:
            primary = spec.label == report["primary_route_id"]
            obstacle_color = "#c62828" if primary else "#999999"
            axis.plot([c[1] for c in spec.route], [c[0] for c in spec.route], color=obstacle_color)
            cell = v9.obstacle_positions(spec, t)[t]
            axis.scatter(cell[1], cell[0], color=obstacle_color, s=25, zorder=6)
        bypass = report["bypassed_cell"]
        axis.scatter(bypass[1], bypass[0], marker="x", color="#222222", s=50, zorder=8)
        if path is None:
            axis.text(0.5, 0.5, report["search"]["status"].replace("_", " "),
                      transform=axis.transAxes, ha="center", bbox={"facecolor": "white", "alpha": 0.9})
            continue
        axis.plot([c[1] for c in path], [c[0] for c in path], "--", color=color, linewidth=1.6)
        for closed_walk in motion_summary(path)["closed_walks"]:
            segment = path[closed_walk["departure_time"]:closed_walk["return_time"] + 1]
            axis.plot([c[1] for c in segment], [c[0] for c in segment], color="#b017ac", linewidth=3)
        position = path[t]
        radius = report["observation_radius"]
        axis.add_patch(visibility.Rectangle((position[1] - radius - 0.5, position[0] - radius - 0.5),
                                            2 * radius + 1, 2 * radius + 1,
                                            facecolor="#ffb300", edgecolor="#ff8f00", alpha=0.15))
        axis.scatter(position[1], position[0], marker="s", color=color, s=30, zorder=7)
    fig.suptitle(f"{report['pair_id']} | saved={report['saved_visible_steps']} steps | spatial={report['spatial_steps']} steps")
    fig.text(0.5, 0.02, "Purple: closed walk | X: nominal conflict cell to bypass | Dots: obstacles at panel observation time | Oracle, not policy", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_summary(output, reports, status, expected):
    rows = [{"pair_id": r["pair_id"], "search_status": r["search"]["status"],
             "stop_reason": r["search"]["stop_reason"], "primary_zone": r["primary_zone"],
             "saved_visible_steps": r["saved_visible_steps"],
             "saved_stay_count": r["saved_visible_motion"]["stay_count"],
             "saved_reversal_count": r["saved_visible_motion"]["immediate_reversal_count"],
             "saved_closed_walk_count": r["saved_visible_motion"]["closed_walk_count"],
             "spatial_steps": r["spatial_steps"], "extra_over_saved_visible": r["extra_over_saved_visible"]}
            for r in reports]
    v10.write_json({"status": status, "expected_cases": expected, "checked_cases": len(rows),
                    "search_outcomes": dict(Counter(row["search_status"] for row in rows)),
                    "training_ready": False, "generation_rules_changed": False, "cases": rows}, output / "summary.json")
    if rows:
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--max-seconds-per-case", type=float, default=15)
    parser.add_argument("--max-expanded", type=int, default=50000)
    parser.add_argument("--max-generated", type=int, default=150000)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds_per_case) or args.max_seconds_per_case <= 0 or min(args.max_expanded, args.max_generated) <= 0:
        parser.error("All budgets must be finite and positive")
    batch = v9._resolve(args.batch_dir)
    context = visibility._read(batch / "batch_context.json")
    data = visibility._read(batch / "verified_pairs.json")
    problem = v9._load_problem(context["config"])
    if problem.grid_sha256 != context["grid_sha256"] or problem.grid_sha256 != data["grid_sha256"]:
        raise ValueError("Map hash changed; refusing to mix contexts.")
    design = v9._scenario_design(context["config"])
    lookup = {row["route_id"]: row for row in data["route_pool"]["corridor"]}
    by_id = {row["scenario_id"]: row for row in data["scenarios"]}
    if len(by_id) != len(data["scenarios"]):
        raise ValueError("Duplicate scenario IDs in source batch.")
    cases = [row for row in data["scenarios"] if row["pair_role"] == "conflict"
             and row["required_behavior"] == "avoidance" and "visible_avoidance" in row]
    if not cases:
        raise ValueError("No saved visible-avoidance pairs in this batch.")
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000000:09d}"
    output = batch / "spatial_bypass_diagnostics" / stamp
    budgets = {"max_expanded": args.max_expanded, "max_generated": args.max_generated, "max_seconds": args.max_seconds_per_case}
    v10.write_json({"source_batch": str(batch), "grid_sha256": problem.grid_sha256,
                    "config": context["config"], "budgets_per_case": budgets,
                    "note": "Budget excludes source revalidation/rendering. No map, route or training changes."}, output / "context.json")
    reports, status = [], "running"
    try:
        for source in cases:
            print(f"Spatial bypass audit: {source['pair_id']}", flush=True)
            report, scenario = audit_case(problem, design, source, by_id[source["paired_scenario_id"]], lookup, budgets)
            v10.write_json(report, output / f"{source['pair_id']}.json")
            reports.append(report)
            write_summary(output, reports, status, len(cases))
            render_case(problem, report, scenario, output / f"{source['pair_id']}.png")
            print(f"  {report['search']['status']}: saved={report['saved_visible_steps']} spatial={report['spatial_steps']}", flush=True)
        status = "complete_with_unknowns" if any(r["search"]["status"] == "budget_exhausted" for r in reports) else "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise SystemExit(130) from None
    except Exception:
        status = "error"
        raise
    finally:
        write_summary(output, reports, status, len(cases))
        print(f"Audit {status}; output={output}", flush=True)
    if status == "complete_with_unknowns":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
