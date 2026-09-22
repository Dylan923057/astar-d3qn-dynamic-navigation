"""Replay retained failed-group examples; separate geometry, constraints and budget.

No random scene generation, acceptance relaxation, or training. Historical
visibility examples are a capped convenience sample, not all spatial failures.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from types import SimpleNamespace

import diagnose_office_v10_visibility as visibility
from astar_d3qn.evaluation.spatial_witness import static_bypass_audit

v10, v9, plt = visibility.v10, visibility.v9, visibility.plt


def gate_blocks():
    return set().union(*(cells for name, cells in visibility.GATE_CELLS.items()
                         if name not in visibility.NOMINAL_GATE_PAIR))


def classify_search(geometry, search):
    if geometry["status"] != "static_bypass_possible":
        return "geometric_constraint_blocks_bypass"
    if search["status"] == "found":
        return "spatial_witness_found"
    if search["status"] == "budget_exhausted":
        return "unknown_resource_limit"
    if search["status"] == "infeasible_under_constraints":
        return "no_path_under_dynamic_visibility_simple_path_constraints"
    return "not_applicable"


def audit_case(problem, design, raw, lookup):
    report, scene = visibility.diagnose_case(problem, design, raw, lookup)
    index = report["primary_obstacle_index"]
    primary = v9.obstacle_positions(scene.obstacles[index], len(problem.nominal_path) - 1)
    cell = next(cell for t, cell in enumerate(problem.nominal_path)
                if t and cell in (primary[t - 1], primary[t]))
    geometry = static_bypass_audit(problem, cell, additionally_blocked=gate_blocks())
    search = {}
    evidence, _ = v10._spatial_avoidance_evidence(
        problem, scene, {"best_plan": SimpleNamespace(steps=report["original_steps"])},
        index, design, search_audit=search,
    )
    if evidence is not None and geometry["status"] != "static_bypass_possible":
        raise ValueError("Dynamic witness contradicts static connectivity audit.")
    if evidence is not None:
        # This is a witness, not a certified matched pair. The helper has
        # independently replayed geometry, dynamics and first-departure view.
        if report["constrained_steps"] is None or evidence["steps"] < report["constrained_steps"]:
            raise ValueError("More restrictive witness improves the less restricted optimum.")
    report.update(geometry=geometry, spatial_search=search, spatial_avoidance=evidence,
                  diagnosis=classify_search(geometry, search))
    return report, scene


def render_case(problem, report, scene, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    spatial = report["spatial_avoidance"]
    for axis, path, title, color in (
        (axes[0], report["constrained_path"], "Visible path (revisits allowed)", "#d17a00"),
        (axes[1], None if spatial is None else spatial["path"], "Cycle-free conflict-cell bypass", "#168543"),
    ):
        v9._draw_base(axis, problem, title)
        for spec in scene.obstacles:
            primary = spec.label == report["primary_route_id"]
            axis.plot([c[1] for c in spec.route], [c[0] for c in spec.route],
                      color="#c62828" if primary else "#999999", linewidth=2 if primary else 1)
        cell = report["geometry"]["conflict_cell"]
        axis.scatter(cell[1], cell[0], marker="x", color="#111111", s=70, zorder=9)
        if path is not None:
            axis.plot([c[1] for c in path], [c[0] for c in path], "--", color=color)
        else:
            axis.text(.5, .5, report["spatial_search"]["stop_reason"].replace("_", " "),
                      transform=axis.transAxes, ha="center", bbox={"facecolor": "white", "alpha": .9})
    fig.suptitle(f"{report['case_id']}\n{report['diagnosis']}", fontsize=10)
    fig.text(.5, .02, "Red: primary route | X: excluded conflict cell | Oracle paths, not learned policy", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_geometry(problem, output):
    rows = [static_bypass_audit(problem, cell, additionally_blocked=gate_blocks())
            for cell in problem.nominal_path[1:-1]]
    v10.write_json({"cells": rows, "training_ready": False,
                    "note": "Green means static connectivity only, NOT dynamic safety or local/observable bypass."},
                   output / "reference_cell_geometry.json")
    fig, axis = plt.subplots(figsize=(8, 8), dpi=150)
    v9._draw_base(axis, problem, "Static bypass precheck under the fixed gate pair")
    for status, color, label in (("static_bypass_possible", "#168543", "Static bypass possible"),
                                 ("conflict_cell_is_required", "#c62828", "Cannot omit this cell")):
        cells = [r["conflict_cell"] for r in rows if r["status"] == status]
        axis.scatter([c[1] for c in cells], [c[0] for c in cells], s=18, color=color, label=label, zorder=7)
    axis.legend(loc="upper right", fontsize=8)
    fig.text(.5, .02, "Cell removal test; other gate pairs blocked. Green is necessary, not sufficient for avoidance.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / "reference_cell_geometry.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    batch = v9._resolve(args.batch_dir)
    context = visibility._read(batch / "batch_context.json")
    problem = v9._load_problem(context["config"])
    if problem.grid_sha256 != context["grid_sha256"]:
        raise ValueError("Map differs from the batch snapshot.")
    design = v9._scenario_design(context["config"])
    lookup = {r["route_id"]: r for r in context["test_route_pool"]["corridor"]}
    summary = visibility._read(batch / "batch_summary.json")
    failed_seeds = {g["seed_offset"] for g in summary["groups"]
                    if g["behavior"] == "avoidance" and g["verified"] == 0}
    cases = [(case_id, raw, path) for case_id, raw, path in visibility._saved_cases(batch)
             if any(case_id.startswith(f"seed_{seed}_") for seed in failed_seeds)]
    if not cases:
        raise ValueError("No retained examples from unsuccessful avoidance groups.")
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000000:09d}"
    output = batch / "avoidance_failure_diagnostics" / stamp
    v10.write_json({"source_batch": str(batch), "grid_sha256": problem.grid_sha256,
                    "config": context["config"], "cases_expected": len(cases),
                    "sampling": "retained visibility examples from failed groups; NOT all spatial failures",
                    "budget": "5 seconds per spatial search; classification/replay/plots are additional",
                    "generation_rules_relaxed": False, "training_ready": False}, output / "context.json")
    print(f"Failure replay: {len(cases)} saved examples; no scene generation; output={output}", flush=True)
    reports, status = [], "running"
    try:
        render_geometry(problem, output)
        for case_id, raw, source in cases:
            print(f"Checking {case_id}", flush=True)
            report, scene = audit_case(problem, design, raw, lookup)
            report.update(case_id=case_id, source_file=source)
            v10.write_json(report, output / f"{case_id}.json")
            reports.append(report)
            render_case(problem, report, scene, output / f"{case_id}.png")
            print(f"  {report['diagnosis']}; reason={report['spatial_search']['stop_reason']}", flush=True)
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise SystemExit(130) from None
    except Exception:
        status = "error"
        raise
    finally:
        fields = ("case_id", "primary_zone", "primary_route_id", "diagnosis", "geometry", "spatial_search")
        v10.write_json({"status": status, "cases_expected": len(cases), "cases_checked": len(reports),
                        "outcomes": dict(Counter(r["diagnosis"] for r in reports)),
                        "training_ready": False, "cases": [{k: r[k] for k in fields} for r in reports],
                        "interpretation": "Complete means replay finished, NOT that all cases are solvable or a dataset is ready."},
                       output / "summary.json")
        print(f"Replay {status}; saved to {output}", flush=True)


if __name__ == "__main__":
    main()
