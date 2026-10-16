"""Expand local critical-zone routes and audit each primary zone separately."""

from __future__ import annotations

import argparse
import copy
import math
import time
from collections import defaultdict

import generate_office_behavior_scenarios_v10 as v10
from generate_office_behavior_scenarios_v6 import _route_metrics
from astar_d3qn.envs.dynamic_grid import _valid_route

v9 = v10.v9
CENTERS = {
    "upper_left_gate": ((10, 9),), "upper_right_gate": ((10, 27),),
    "lower_left_gate": ((25, 17),), "lower_right_gate": ((25, 31),),
    "goal_approach": ((37, 24), (37, 31)),
}
# Coordinates are (row, column). These are approach candidates, not certified
# safe positions. The per-phase connectivity filter below still has to pass.
APPROACH_CENTERS = {
    "upper_left_gate": ((7, 9), (14, 9)),
    "lower_left_gate": ((23, 14), (29, 17)),
}


def route_key(route):
    route = tuple(map(tuple, route))
    return min(route, tuple(reversed(route)))


def geometry_candidates(problem, pools, *, centers_by_zone=None, radius=2, per_bucket=6, prefix="test_expanded"):
    """At most 36 per zone, balanced over length/orientation before phase tests."""
    occupied = {route_key(row["route"]) for pool in pools.values() for row in pool["corridor"]}
    result, audit = [], {}
    for zone, centers in (CENTERS if centers_by_zone is None else centers_by_zone).items():
        buckets = defaultdict(dict)
        for anchor_row, anchor_col in centers:
            for row in range(anchor_row - radius, anchor_row + radius + 1):
                for col in range(anchor_col - radius, anchor_col + radius + 1):
                    for length in (3, 5, 7):
                        for orientation, (dr, dc) in (("horizontal", (0, 1)), ("vertical", (1, 0))):
                            route = tuple((row + dr * offset, col + dc * offset)
                                          for offset in range(-(length // 2), length // 2 + 1))
                            key = route_key(route)
                            if key in occupied or not _valid_route(route, problem, min_length=length):
                                continue
                            distance = min(max(abs(row - r), abs(col - c)) for r, c in centers)
                            buckets[(length, orientation)][key] = (distance, row, col, route)
        available = sum(len(values) for values in buckets.values())
        chosen = []
        for bucket in sorted(buckets):
            chosen.extend(sorted(buckets[bucket].values())[:per_bucket])
        audit[zone] = {"valid_new_geometry_count": available, "preflight_geometry_count": len(chosen),
                       "center_radius": radius, "per_length_orientation_cap": per_bucket,
                       "scope": "bounded geometry around anchors; not a behavior certificate"}
        for number, (_, row, col, route) in enumerate(chosen):
            result.append({"route_id": f"{prefix}_{zone}_{number:03d}", "critical_zone": zone,
                           "center": [row, col], "route": [list(cell) for cell in route],
                           "orientation": "horizontal" if route[0][0] == route[-1][0] else "vertical",
                           "behavior_anchor": False, "pool": "corridor", "functional_role": "critical_zone_expansion_candidate"})
    return result, audit


def approach_geometry_candidates(problem, pools):
    """Keep separate budgets for both sides so one anchor cannot crowd out the other."""
    rows, audit, seen = [], {}, set()
    for zone, anchors in APPROACH_CENTERS.items():
        for side, anchor in enumerate(anchors):
            candidates, details = geometry_candidates(
                problem, pools, centers_by_zone={zone: (anchor,)}, radius=1, per_bucket=3,
                prefix=f"test_approach_side{side}")
            audit[f"{zone}_side{side}"] = {**details[zone], "anchor": anchor}
            for row in candidates:
                key = route_key(row["route"])
                if key not in seen:
                    seen.add(key)
                    row["approach_side"] = side
                    row["functional_role"] = "local_avoidance_approach_candidate"
                    rows.append(row)
    return rows, audit


def render_approach_preview(problem, pools, selection, spatial_audit, output):
    fig, axes = v10.plt.subplots(1, 2, figsize=(12, 6), dpi=160)
    for axis, zone in zip(axes, APPROACH_CENTERS):
        v9._draw_base(axis, problem, f"{zone}: candidate motion routes")
        route_ids = set(selection[zone]["added_route_ids"])
        for row in pools["test"]["corridor"]:
            if row["route_id"] not in route_ids:
                continue
            route = row["route"]
            axis.plot([c[1] for c in route], [c[0] for c in route],
                      color=("#ef6c00", "#1565c0")[row["approach_side"]], alpha=.7)
        cells = {tuple(row["first_conflict_cell"]) for row in spatial_audit["phases"]
                 if row["route_id"] in route_ids and row["status"] == "static_bypass_possible"}
        axis.scatter([c[1] for c in cells], [c[0] for c in cells], color="#168543", s=15, zorder=8)
        if not route_ids:
            axis.text(.5, .5, "No eligible approach routes", transform=axis.transAxes, ha="center")
    fig.text(.5, .02, "Orange/blue: two approaches | Green: statically bypassable conflict cells | Not certified dynamic scenes", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / "avoidance_approaches.png", facecolor="white", bbox_inches="tight")
    v10.plt.close(fig)


def expand_pools(problem, config, prepared, output, *, avoidance_approaches=False):
    design, demos, original = prepared
    pools = copy.deepcopy(original)
    candidates, geometry_audit = (approach_geometry_candidates(problem, original) if avoidance_approaches
                                  else geometry_candidates(problem, original))
    options = [v10._route_options_for_speeds(row, design["move_every_choices"], demos) for row in candidates]
    safe, primary = v10._preflight_phase_pairs(problem, candidates, options, design, output / "geometry_preflight")
    if avoidance_approaches:
        primary, spatial_audit = v10._filter_spatial_primary_options(problem, primary)
        v10.write_json(spatial_audit, output / "geometry_preflight" / "spatial_proposal_precheck.json")
    topology = v9._reference_demo_paths(problem, int(config["spatial_generalization"]["reference_path_count"]),
                                       int(config["spatial_generalization"]["generation_seed"]) + 100_000)
    selection_audit = {}
    active_centers = APPROACH_CENTERS if avoidance_approaches else CENTERS
    for zone in active_centers:
        indices = [i for i, row in enumerate(candidates) if row["critical_zone"] == zone and safe[i]
                   and (not avoidance_approaches or primary[i])]
        # Necessary phase-qualified primary first, then length/orientation
        # coverage. This selection does not certify any behavior or any pair.
        ranked = sorted(indices, key=lambda i: (not bool(primary[i]), candidates[i]["route_id"]))
        if avoidance_approaches:
            # Round-robin sides before the six-route cap, keeping room for
            # both approaches when both have qualifying phase proposals.
            sides = [[i for i in ranked if candidates[i]["approach_side"] == side] for side in (0, 1)]
            ranked = [side[j] for j in range(max(map(len, sides), default=0)) for side in sides if j < len(side)]
        selected, kinds = [], set()
        for i in ranked:
            kind = (bool(primary[i]), len(candidates[i]["route"]), candidates[i]["orientation"])
            if avoidance_approaches:
                kind += (candidates[i]["approach_side"],)
            if kind not in kinds and len(selected) < 6:
                kinds.add(kind)
                selected.append(i)
        for i in ranked:
            if len(selected) >= 6:
                break
            if i not in selected:
                selected.append(i)
        for i in selected:
            row = candidates[i]
            row.update(_route_metrics(tuple(map(tuple, row["route"])), topology, demos))
            pools["test"]["corridor"].append(row)
        selection_audit[zone] = {"added_route_ids": [candidates[i]["route_id"] for i in selected],
                                 "selected_approach_sides": sorted({candidates[i]["approach_side"] for i in selected}) if avoidance_approaches else None,
                                 "phase_qualified_primary_candidates": sum(bool(primary[i]) for i in indices),
                                 "phase_qualified_primary_selected": sum(bool(primary[i]) for i in selected)}
    # Whole-route equality across splits remains forbidden. Shared gate cells
    # are allowed, as in the existing same-Office protocol.
    other_keys = {route_key(row["route"]) for split in ("train", "validation") for row in pools[split]["corridor"]}
    if any(route_key(row["route"]) in other_keys for row in pools["test"]["corridor"]):
        raise ValueError("Expanded test pool duplicates a whole train/validation route.")
    if avoidance_approaches:
        design = {**design, "_avoidance_primary_route_ids": [
            route_id for info in selection_audit.values() for route_id in info["added_route_ids"]]}
    v10.write_json({"grid_sha256": problem.grid_sha256, "anchors": active_centers,
                    "mode": "avoidance_approaches" if avoidance_approaches else "all_zone_expansion",
                    "geometry_audit": geometry_audit, "selection_audit": selection_audit,
                    "original_test_routes": original["test"]["corridor"],
                    "considered_new_routes": candidates, "route_pools": pools,
                    "note": "Only test/development pool expanded; no training split redesign or behavior success claimed."},
                   output / "route_expansion.json")
    v9.render_route_pools(problem, {"route_pools": pools}, output / "expanded_route_pools.png")
    if avoidance_approaches:
        render_approach_preview(problem, pools, selection_audit, spatial_audit, output)
    return design, demos, pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=v10.DEFAULT_CONFIG)
    parser.add_argument("--attempts-per-group", type=int, default=300)
    parser.add_argument("--seconds-per-group", type=float, default=30)
    parser.add_argument("--avoidance-approaches", action="store_true",
                        help="Only test local avoidance at two left-gate approaches; keep walls and wait/reroute rules.")
    args = parser.parse_args()
    if args.attempts_per_group <= 0 or not math.isfinite(args.seconds_per_group) or args.seconds_per_group <= 0:
        parser.error("Budgets must be finite and positive")
    config = v10.load_config(v9._resolve(args.config))
    problem = v9._load_problem(config)
    prepared = v10._prepare_generation(problem, config)
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000000:09d}"
    folder = "avoidance_approach_pilots" if args.avoidance_approaches else "route_expansion_pilots"
    output = v10.ROOT / "outputs/office_behavior_v10_dataset_design" / folder / stamp
    print(f"Route expansion pilot: {output}", flush=True)
    expanded = expand_pools(problem, config, prepared, output, avoidance_approaches=args.avoidance_approaches)
    zones = dict(enumerate(APPROACH_CENTERS if args.avoidance_approaches else CENTERS))
    v10.run_pair_batch_pilot(problem, config, output, args.attempts_per_group,
                            2 if args.avoidance_approaches else 1, list(zones),
                            args.seconds_per_group, visible_avoidance=True, spatial_avoidance=True,
                            prepared=expanded, primary_zones=zones,
                            behaviors=("avoidance",) if args.avoidance_approaches else v10.CAUSAL_BEHAVIORS)


if __name__ == "__main__":
    main()
