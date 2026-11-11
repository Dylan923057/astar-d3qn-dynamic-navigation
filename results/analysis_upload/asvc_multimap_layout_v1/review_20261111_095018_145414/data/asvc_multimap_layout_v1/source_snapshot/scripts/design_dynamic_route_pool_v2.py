"""Refine whole-map dynamic route pools and audit 5/7-obstacle scenes; never train."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

import design_dynamic_route_pool_v1 as v1
from astar_d3qn.core.grid import in_bounds, manhattan
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.utils.io import write_json


CATEGORY_COLORS = v1.CATEGORY_COLORS
CATEGORY_LABELS = v1.CATEGORY_LABELS
CATEGORIES = ("high_interaction", "alternative_branch", "background")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensemble_sha256(paths: Sequence[Sequence[tuple[int, int]]]) -> str:
    serializable = [[[int(r), int(c)] for r, c in path] for path in paths]
    return hashlib.sha256(
        json.dumps(serializable, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _route_cells(record: Mapping[str, Any]) -> set[tuple[int, int]]:
    return set(map(tuple, record["route"]))


def _center_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return manhattan(tuple(left["center"]), tuple(right["center"]))


def _strict_spread_select(candidates, count, *, occupied=(), minimum_gap=3):
    selected = []
    for candidate in candidates:
        if all(
            _center_distance(candidate, prior) >= minimum_gap
            for prior in (*occupied, *selected)
        ):
            selected.append(candidate)
            if len(selected) == count:
                return selected
    raise RuntimeError(
        f"Strict route-pool spacing selected {len(selected)}/{count}; "
        "the configured filter must be revised explicitly."
    )


def _spatial_region(center, region_size):
    return f"r{int(center[0]) // region_size}_c{int(center[1]) // region_size}"


def _enrich_route_features(problem, paths):
    path_sets = tuple(set(path) for path in paths)
    usage = Counter(cell for path in paths for cell in set(path))
    entropy, directions = v1._transition_entropy(paths)
    records = v1._route_features(
        problem, paths, path_sets, usage, entropy, directions
    )
    nominal = set(problem.nominal_path)
    nominal_tuple = tuple(problem.nominal_path)
    alternative_path_indices = {
        index for index, path in enumerate(paths) if tuple(path) != nominal_tuple
    }
    for record in records:
        cells = _route_cells(record)
        alternative_hits = sorted(
            set(record["ensemble_path_indices"]) & alternative_path_indices
        )
        record["alternative_path_indices"] = alternative_hits
        record["alternative_path_support_count"] = len(alternative_hits)
        record["alternative_path_support_fraction"] = round(
            len(alternative_hits) / len(paths), 6
        )
        record["minimum_distance_to_nominal_astar"] = min(
            manhattan(cell, nominal_cell) for cell in cells for nominal_cell in nominal
        )
        record["route_edge_clearance"] = min(
            min(row, column, problem.size - 1 - row, problem.size - 1 - column)
            for row, column in cells
        )
    support = [
        {
            "cell": list(cell),
            "path_count": count,
            "path_fraction": round(count / len(paths), 6),
        }
        for cell, count in sorted(usage.items())
    ]
    return records, support


def _select_by_progress_bands(
    candidates,
    *,
    count,
    edges,
    minimum_gap,
    score,
    band_field,
):
    band_count = len(edges) - 1
    if count % band_count:
        raise ValueError(f"{count} candidates cannot be balanced across {band_count} bands.")
    selected = []
    for band_index, (low, high) in enumerate(zip(edges, edges[1:])):
        band = [
            record
            for record in candidates
            if float(low) <= record["progress_fraction"] < float(high)
        ]
        band.sort(key=score)
        chosen = _strict_spread_select(
            band,
            count // band_count,
            occupied=selected,
            minimum_gap=minimum_gap,
        )
        for record in chosen:
            record[band_field] = band_index
        selected.extend(chosen)
    return selected


def build_refined_pool(problem, paths, counts, classification, region_size):
    records, support = _enrich_route_features(problem, paths)

    hi_candidates = [
        record
        for record in records
        if record["ensemble_intersection_fraction"]
        >= float(classification["high_interaction_min_path_fraction"])
        and record["mean_cell_support"]
        >= float(classification["high_interaction_min_mean_cell_support"])
    ]
    hi = _select_by_progress_bands(
        hi_candidates,
        count=int(counts["high_interaction"]),
        edges=classification["high_interaction_progress_edges"],
        minimum_gap=3,
        band_field="interaction_progress_band",
        score=lambda record: (
            -record["ensemble_intersection_fraction"],
            -record["mean_cell_support"],
            -record["perpendicular_flow_fraction"],
            -record["branch_merge_entropy"],
            record["route_length"],
            record["center"],
        ),
    )

    hi_geometry = {tuple(map(tuple, record["route"])) for record in hi}
    ar_candidates = [
        record
        for record in records
        if tuple(map(tuple, record["route"])) not in hi_geometry
        and not record["intersects_nominal_astar"]
        and record["alternative_path_support_fraction"]
        >= float(classification["alternative_min_path_fraction"])
        and record["mean_cell_support"]
        >= float(classification["alternative_min_mean_cell_support"])
        and record["branch_merge_entropy"]
        >= float(classification["alternative_min_branch_entropy"])
        and record["minimum_distance_to_nominal_astar"]
        >= int(classification["alternative_min_distance_from_nominal"])
        and record["route_edge_clearance"]
        >= int(classification["alternative_min_edge_clearance"])
    ]
    ar = _select_by_progress_bands(
        ar_candidates,
        count=int(counts["alternative_branch"]),
        edges=classification["alternative_progress_edges"],
        minimum_gap=4,
        band_field="alternative_progress_band",
        score=lambda record: (
            -record["alternative_path_support_fraction"],
            -record["branch_merge_entropy"],
            -record["perpendicular_flow_fraction"],
            -record["mean_cell_support"],
            record["route_length"],
            record["center"],
        ),
    )

    used_geometry = {
        tuple(map(tuple, record["route"])) for record in (*hi, *ar)
    }
    bg_candidates = [
        record
        for record in records
        if tuple(map(tuple, record["route"])) not in used_geometry
        and float(classification["background_min_path_fraction"])
        <= record["ensemble_intersection_fraction"]
        <= float(classification["background_max_path_fraction"])
        and record["route_edge_clearance"]
        >= int(classification["background_min_edge_clearance"])
    ]
    bg_candidates.sort(
        key=lambda record: (
            -record["ensemble_intersection_fraction"],
            -record["branch_merge_entropy"],
            -record["perpendicular_flow_fraction"],
            record["route_length"],
            record["center"],
        )
    )
    bg = _strict_spread_select(
        bg_candidates, int(counts["background"]), minimum_gap=3
    )

    route_pool = {
        "high_interaction": hi,
        "alternative_branch": ar,
        "background": bg,
    }
    prefixes = {
        "high_interaction": "HI",
        "alternative_branch": "AR",
        "background": "BG",
    }
    for category, category_records in route_pool.items():
        for index, record in enumerate(category_records, start=1):
            record["route_id"] = (
                f"{problem.map_id}_v2_{prefixes[category]}_{index:02d}"
            )
            record["category"] = category
            record["motion"] = "continuous_ping_pong"
            record["spatial_region"] = _spatial_region(
                record["center"], region_size
            )
            record["selection_basis"] = (
                "whole_map_reasonable_path_ensemble_not_single_nominal_astar"
            )
    audit = {
        "enumerated_free_routes": len(records),
        "eligible_high_interaction_routes": len(hi_candidates),
        "eligible_alternative_routes_after_cleanup": len(ar_candidates),
        "eligible_background_routes": len(bg_candidates),
        "all_ar_avoid_nominal_astar": all(
            not record["intersects_nominal_astar"] for record in ar
        ),
        "minimum_selected_ar_alternative_support_fraction": min(
            record["alternative_path_support_fraction"] for record in ar
        ),
        "minimum_selected_ar_edge_clearance": min(
            record["route_edge_clearance"] for record in ar
        ),
        "minimum_selected_hi_path_fraction": min(
            record["ensemble_intersection_fraction"] for record in hi
        ),
    }
    return route_pool, support, audit


def _scene_geometry_ok(selected, acceptance):
    minimum_gap = int(acceptance["minimum_center_manhattan_gap"])
    minimum_span = int(acceptance["minimum_spatial_span"])
    minimum_regions = int(acceptance["minimum_spatial_regions"])
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            if _route_cells(left) & _route_cells(right):
                return False
            if _center_distance(left, right) < minimum_gap:
                return False
    regions = {record["spatial_region"] for record in selected}
    if len(regions) < minimum_regions:
        return False
    span = max(
        _center_distance(left, right)
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    )
    if span < minimum_span:
        return False
    far_count = sum(
        record["minimum_distance_to_nominal_astar"]
        >= int(acceptance["far_from_nominal_distance"])
        for record in selected
    )
    if far_count < int(acceptance["minimum_far_from_nominal_routes"]):
        return False
    hi = [r for r in selected if r["category"] == "high_interaction"]
    ar = [r for r in selected if r["category"] == "alternative_branch"]
    if not any(
        record["ensemble_intersection_fraction"]
        >= float(acceptance["high_support_path_fraction"])
        for record in hi
    ):
        return False
    if not any(record["alternative_path_support_count"] > 0 for record in ar):
        return False
    if len({record["interaction_progress_band"] for record in hi}) < 2:
        return False
    return True


def _coverage_audit(selected, path_count):
    category_indices = {}
    for category in CATEGORIES:
        records = [record for record in selected if record["category"] == category]
        category_indices[category] = set().union(
            *(set(record["ensemble_path_indices"]) for record in records)
        )
    all_indices = set().union(*category_indices.values())
    return {
        "reasonable_path_count": path_count,
        "interacting_path_count": len(all_indices),
        "interacting_path_fraction": round(len(all_indices) / path_count, 6),
        "fully_avoiding_path_count": path_count - len(all_indices),
        "fully_avoiding_path_fraction": round(
            1.0 - len(all_indices) / path_count, 6
        ),
        "category_path_coverage": {
            category: {
                "path_count": len(indices),
                "path_fraction": round(len(indices) / path_count, 6),
            }
            for category, indices in category_indices.items()
        },
    }


def sample_scenes(problem, route_pool, paths, density, composition, acceptance, seed):
    rng = random.Random(seed)
    scene_count = int(acceptance["examples_per_density"])
    trial_count = int(acceptance["sampling_trials_per_scene"])
    minimum_coverage = float(acceptance["minimum_scene_path_coverage"])
    route_usage: Counter[str] = Counter()
    used_signatures = set()
    scenes = []
    for scene_index in range(scene_count):
        best = None
        for _ in range(trial_count):
            selected = []
            for category in CATEGORIES:
                selected.extend(
                    rng.sample(
                        route_pool[category], int(composition[category])
                    )
                )
            signature = tuple(sorted(record["route_id"] for record in selected))
            if signature in used_signatures or not _scene_geometry_ok(selected, acceptance):
                continue
            audit = _coverage_audit(selected, len(paths))
            if audit["interacting_path_fraction"] < minimum_coverage:
                continue
            reuse = sum(route_usage[record["route_id"]] for record in selected)
            # Once acceptance is satisfied, keep the examples sampled rather
            # than selecting the maximum-coverage layouts. Route reuse is the
            # only deterministic preference, so the offline density comparison
            # is not inflated by an interaction-maximizing preview search.
            score = (-reuse, rng.random())
            if best is None or score > best[0]:
                best = (score, selected, audit, signature)
        if best is None:
            raise RuntimeError(
                f"{problem.map_id} density {density}: no scene passed the frozen acceptance rules."
            )
        _, selected, audit, signature = best
        used_signatures.add(signature)
        route_usage.update(record["route_id"] for record in selected)
        obstacles = []
        for record in selected:
            obstacles.append(
                {
                    "route_id": record["route_id"],
                    "category": record["category"],
                    "route": record["route"],
                    "start_index": rng.randrange(len(record["route"])),
                    "direction": rng.choice((-1, 1)),
                    "move_every": int(acceptance["move_every"]),
                    "motion": "continuous_ping_pong_for_entire_episode",
                }
            )
        scenes.append(
            {
                "scenario_id": (
                    f"{problem.map_id}_v2_density{density}_preview_{scene_index + 1:02d}"
                ),
                "density": int(density),
                "composition": dict(composition),
                "obstacles": obstacles,
                "spatial_regions": sorted(
                    {record["spatial_region"] for record in selected}
                ),
                "selection_time": "episode_reset_only",
                "regenerate_during_episode": False,
                "agent_tracking_placement": False,
                "coverage_audit": audit,
            }
        )
    return scenes


def sample_audit_cohort(problem, route_pool, paths, density, composition, acceptance, seed):
    """Draw an independent accepted-scene cohort without optimizing coverage."""
    rng = random.Random(seed)
    sample_count = int(acceptance["offline_audit_scenes_per_density"])
    trial_count = int(acceptance["sampling_trials_per_audit_scene"])
    minimum_coverage = float(acceptance["minimum_scene_path_coverage"])
    samples = []
    for sample_index in range(sample_count):
        accepted = None
        for _ in range(trial_count):
            selected = []
            for category in CATEGORIES:
                selected.extend(
                    rng.sample(route_pool[category], int(composition[category]))
                )
            if not _scene_geometry_ok(selected, acceptance):
                continue
            audit = _coverage_audit(selected, len(paths))
            if audit["interacting_path_fraction"] < minimum_coverage:
                continue
            accepted = (selected, audit)
            break
        if accepted is None:
            raise RuntimeError(
                f"{problem.map_id} density {density}: offline audit sample "
                f"{sample_index} exhausted {trial_count} trials."
            )
        selected, audit = accepted
        samples.append(
            {
                "sample_id": f"{problem.map_id}_density{density}_audit_{sample_index + 1:04d}",
                "density": int(density),
                "route_ids": [record["route_id"] for record in selected],
                "spatial_regions": sorted(
                    {record["spatial_region"] for record in selected}
                ),
                "coverage_audit": audit,
            }
        )
    return samples


def summarize_density(scenes):
    def values(key):
        return [scene["coverage_audit"][key] for scene in scenes]

    interaction = values("interacting_path_fraction")
    bypass = values("fully_avoiding_path_fraction")
    summary = {
        "scene_count": len(scenes),
        "mean_interacting_path_fraction": round(float(np.mean(interaction)), 6),
        "minimum_interacting_path_fraction": round(min(interaction), 6),
        "maximum_interacting_path_fraction": round(max(interaction), 6),
        "mean_fully_avoiding_path_fraction": round(float(np.mean(bypass)), 6),
    }
    for category in CATEGORIES:
        fractions = [
            scene["coverage_audit"]["category_path_coverage"][category][
                "path_fraction"
            ]
            for scene in scenes
        ]
        summary[f"mean_{category}_path_coverage_fraction"] = round(
            float(np.mean(fractions)), 6
        )
    return summary


def validate_map_entry(problem, entry, counts, schemes, acceptance):
    lookup = {}
    for category, expected in counts.items():
        records = entry["route_pool"][category]
        if len(records) != int(expected):
            raise AssertionError(f"{category}: {len(records)} != {expected}")
        for record in records:
            route = tuple(map(tuple, record["route"]))
            if any(
                not in_bounds(cell, problem.size) or cell in problem.obstacles
                for cell in route
            ):
                raise AssertionError(f"{record['route_id']}: route is not free")
            if any(manhattan(a, b) != 1 for a, b in zip(route, route[1:])):
                raise AssertionError(f"{record['route_id']}: route is not continuous")
            lookup[record["route_id"]] = record
    if not all(
        not record["intersects_nominal_astar"]
        and record["alternative_path_support_count"] > 0
        for record in entry["route_pool"]["alternative_branch"]
    ):
        raise AssertionError("AR cleanup invariant failed")

    for density, composition in schemes.items():
        scenes = entry["sampled_scenes"][str(density)]
        if len(scenes) != int(acceptance["examples_per_density"]):
            raise AssertionError("Incorrect example scene count")
        for scene in scenes:
            selected = [lookup[o["route_id"]] for o in scene["obstacles"]]
            actual_composition = Counter(r["category"] for r in selected)
            expected_composition = Counter(composition)
            if actual_composition != expected_composition:
                raise AssertionError(
                    f"{scene['scenario_id']}: composition mismatch "
                    f"{dict(actual_composition)} != {dict(expected_composition)}"
                )
            if not _scene_geometry_ok(selected, acceptance):
                raise AssertionError(f"{scene['scenario_id']}: geometry acceptance failed")
            expected_audit = _coverage_audit(
                selected, entry["path_ensemble"]["sample_count"]
            )
            if expected_audit != scene["coverage_audit"]:
                raise AssertionError(f"{scene['scenario_id']}: coverage audit mismatch")
            if scene["regenerate_during_episode"] or scene["agent_tracking_placement"]:
                raise AssertionError(f"{scene['scenario_id']}: forbidden refresh behavior")
        audit_samples = entry["offline_audit_scenes"][str(density)]
        if len(audit_samples) != int(acceptance["offline_audit_scenes_per_density"]):
            raise AssertionError("Incorrect offline audit cohort size")
        for sample in audit_samples:
            selected = [lookup[route_id] for route_id in sample["route_ids"]]
            if Counter(r["category"] for r in selected) != Counter(composition):
                raise AssertionError(f"{sample['sample_id']}: audit composition mismatch")
            if not _scene_geometry_ok(selected, acceptance):
                raise AssertionError(f"{sample['sample_id']}: audit geometry failed")
            expected_audit = _coverage_audit(
                selected, entry["path_ensemble"]["sample_count"]
            )
            if expected_audit != sample["coverage_audit"]:
                raise AssertionError(f"{sample['sample_id']}: audit mismatch")
    return {
        "all_routes_static_free_and_four_connected": True,
        "all_ar_non_nominal_and_alternative_supported": True,
        "all_scenes_episode_reset_only": True,
        "all_scenes_pass_spatial_and_coverage_acceptance": True,
    }


def _draw_base(ax, problem, entry, include_heat=True):
    support = entry["path_support"] if include_heat else None
    path_count = entry["path_ensemble"]["sample_count"] if include_heat else None
    v1._draw_base(ax, problem, support, path_count)


def render_candidate_pool(problem, entry, output):
    fig, ax = plt.subplots(figsize=(8.5, 8.2))
    _draw_base(ax, problem, entry)
    for category in ("background", "alternative_branch", "high_interaction"):
        for record in entry["route_pool"][category]:
            route = np.asarray(record["route"])
            ax.plot(
                route[:, 1],
                route[:, 0],
                color=CATEGORY_COLORS[category],
                lw=2.1,
                alpha=0.82,
            )
    handles = [
        Line2D([0], [0], color=CATEGORY_COLORS[key], lw=3, label=CATEGORY_LABELS[key])
        for key in CATEGORIES
    ] + [
        Line2D([0], [0], color="#666666", lw=1, ls="--", label="nominal A* (reference only)"),
        Line2D([0], [0], color="#d99b24", lw=7, alpha=0.24, label="240-path support"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5)
    counts = {key: len(value) for key, value in entry["route_pool"].items()}
    ax.set_title(
        f"{problem.map_id}: refined whole-map candidate pool\n"
        f"HI={counts['high_interaction']}  AR={counts['alternative_branch']}  BG={counts['background']}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=175)
    plt.close(fig)


def render_scene_examples(problem, entry, density, output):
    scenes = entry["sampled_scenes"][str(density)]
    fig, axes = plt.subplots(1, len(scenes), figsize=(5.1 * len(scenes), 5.5), squeeze=False)
    for ax, scene in zip(axes[0], scenes):
        _draw_base(ax, problem, entry, include_heat=False)
        for number, obstacle in enumerate(scene["obstacles"], start=1):
            route = np.asarray(obstacle["route"])
            color = CATEGORY_COLORS[obstacle["category"]]
            ax.plot(route[:, 1], route[:, 0], "o-", color=color, lw=1.8, ms=2.3)
            initial = route[obstacle["start_index"]]
            ax.scatter(
                initial[1], initial[0], s=55, color=color, edgecolors="white", zorder=8
            )
            ax.text(
                initial[1] + 0.3,
                initial[0] - 0.3,
                str(number),
                color=color,
                fontsize=7,
                weight="bold",
            )
        audit = scene["coverage_audit"]
        ax.set_title(
            f"scene {scene['scenario_id'].rsplit('_', 1)[-1]} | "
            f"interact {audit['interacting_path_fraction']:.1%}\n"
            f"bypass {audit['fully_avoiding_path_fraction']:.1%} | "
            f"regions {len(scene['spatial_regions'])}",
            fontsize=9,
        )
    handles = [
        Line2D([0], [0], color=CATEGORY_COLORS[key], marker="o", lw=2, label=CATEGORY_LABELS[key])
        for key in CATEGORIES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5)
    composition = scenes[0]["composition"]
    fig.suptitle(
        f"{problem.map_id}: {density} episode-fixed obstacles "
        f"({composition['high_interaction']} HI + {composition['alternative_branch']} AR + "
        f"{composition['background']} BG)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_summary_table(problem, entry, output):
    counts = {key: len(value) for key, value in entry["route_pool"].items()}
    columns = [
        "scheme",
        "interacting paths\nmean [min, max]",
        "fully avoiding\nmean",
        "HI coverage\nmean",
        "AR coverage\nmean",
        "BG coverage\nmean",
    ]
    rows = []
    for density in (5, 7):
        summary = entry["density_summary"][str(density)]
        rows.append(
            [
                f"{density} obstacles\n(n={summary['scene_count']})",
                f"{summary['mean_interacting_path_fraction']:.1%} "
                f"[{summary['minimum_interacting_path_fraction']:.1%}, "
                f"{summary['maximum_interacting_path_fraction']:.1%}]",
                f"{summary['mean_fully_avoiding_path_fraction']:.1%}",
                f"{summary['mean_high_interaction_path_coverage_fraction']:.1%}",
                f"{summary['mean_alternative_branch_path_coverage_fraction']:.1%}",
                f"{summary['mean_background_path_coverage_fraction']:.1%}",
            ]
        )
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.75)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#e7eef7")
        table[(0, column)].set_text_props(weight="bold")
    ax.set_title(
        f"{problem.map_id}: offline 240-path spatial coverage audit\n"
        f"candidate pool HI={counts['high_interaction']}, "
        f"AR={counts['alternative_branch']}, BG={counts['background']} | "
        "category coverages may overlap",
        fontsize=12,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_audit_csv(payload, path):
    rows = []
    for entry in payload["maps"]:
        map_id = entry["problem_reference"]["map_id"]
        counts = {key: len(value) for key, value in entry["route_pool"].items()}
        for density, samples in entry["offline_audit_scenes"].items():
            for sample in samples:
                audit = sample["coverage_audit"]
                rows.append(
                    {
                        "map_id": map_id,
                        "density": density,
                        "scenario_id": sample["sample_id"],
                        "candidate_hi": counts["high_interaction"],
                        "candidate_ar": counts["alternative_branch"],
                        "candidate_bg": counts["background"],
                        "spatial_region_count": len(sample["spatial_regions"]),
                        "interacting_path_count": audit["interacting_path_count"],
                        "interacting_path_fraction": audit["interacting_path_fraction"],
                        "fully_avoiding_path_count": audit["fully_avoiding_path_count"],
                        "fully_avoiding_path_fraction": audit["fully_avoiding_path_fraction"],
                        "hi_path_coverage_fraction": audit["category_path_coverage"]["high_interaction"]["path_fraction"],
                        "ar_path_coverage_fraction": audit["category_path_coverage"]["alternative_branch"]["path_fraction"],
                        "bg_path_coverage_fraction": audit["category_path_coverage"]["background"]["path_fraction"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dynamic_route_pool_v2.yaml")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_lookup = {
        entry["problem"]["map_id"]: entry for entry in source["maps"]
    }
    problems = [
        problem_from_record(source_lookup[map_id]["problem"])
        for map_id in config["map_ids"]
    ]
    design = {
        "algorithm_revision": "refined_ar_dual_density_offline_audit_v3",
        "source_manifest": config["source_manifest"],
        "source_sha256": _sha256(source_path),
        "map_ids": config["map_ids"],
        "path_ensemble": config["path_ensemble"],
        "route_pool_counts": config["route_pool_counts"],
        "classification": config["classification"],
        "scene_acceptance": config["scene_acceptance"],
        "density_schemes": config["density_schemes"],
    }
    design_sha256 = hashlib.sha256(
        json.dumps(design, sort_keys=True).encode("utf-8")
    ).hexdigest()
    destination = ROOT / config["dataset"]
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("design_sha256") != design_sha256:
            raise SystemExit("Frozen v2 design differs from config; create a new version.")
        print(f"Using existing frozen design: {destination}", flush=True)
    else:
        payload = {
            "format_version": 2,
            "protocol": config["experiment"],
            "design_only": True,
            "training_started": False,
            "adaptive_modified": False,
            "design": design,
            "design_sha256": design_sha256,
            "maps": [],
        }
        ensemble_config = config["path_ensemble"]
        for map_index, problem in enumerate(problems):
            paths, _, _ = v1.build_reasonable_path_ensemble(
                problem,
                seed=problem.seed + int(ensemble_config["seed_offset"]),
                sample_count=int(ensemble_config["sample_count"]),
                candidate_attempts=int(ensemble_config["candidate_attempts"]),
                detour_budget_steps=int(ensemble_config["detour_budget_steps"]),
            )
            route_pool, support, pool_audit = build_refined_pool(
                problem,
                paths,
                config["route_pool_counts"],
                config["classification"],
                int(config["scene_acceptance"]["spatial_region_size"]),
            )
            sampled = {}
            audit_samples = {}
            summaries = {}
            for density, composition in config["density_schemes"].items():
                sampled[str(density)] = sample_scenes(
                    problem,
                    route_pool,
                    paths,
                    int(density),
                    composition,
                    config["scene_acceptance"],
                    seed=problem.seed + 820000 + 1000 * map_index + int(density),
                )
                audit_samples[str(density)] = sample_audit_cohort(
                    problem,
                    route_pool,
                    paths,
                    int(density),
                    composition,
                    config["scene_acceptance"],
                    seed=problem.seed + 920000 + 1000 * map_index + int(density),
                )
                summaries[str(density)] = summarize_density(
                    audit_samples[str(density)]
                )
            entry = {
                "problem_reference": {
                    "map_id": problem.map_id,
                    "grid_sha256": problem.grid_sha256,
                    "size": problem.size,
                    "start": list(problem.start),
                    "goal": list(problem.goal),
                    "static_obstacle_count": len(problem.obstacles),
                    "nominal_astar_steps": problem.astar_steps,
                },
                "path_ensemble": {
                    "sample_count": len(paths),
                    "unique_path_count": len(set(paths)),
                    "minimum_steps": min(len(path) - 1 for path in paths),
                    "maximum_steps": max(len(path) - 1 for path in paths),
                    "sha256": _ensemble_sha256(paths),
                    "construction": "whole_free_space_bounded_detour_waypoint_ensemble",
                    "nominal_astar_used_for_route_enumeration": False,
                },
                "path_support": support,
                "route_pool": route_pool,
                "route_pool_audit": pool_audit,
                "sampled_scenes": sampled,
                "offline_audit_scenes": audit_samples,
                "density_summary": summaries,
            }
            entry["acceptance"] = validate_map_entry(
                problem,
                entry,
                config["route_pool_counts"],
                config["density_schemes"],
                config["scene_acceptance"],
            )
            payload["maps"].append(entry)
            print(
                f"{problem.map_id}: HI={len(route_pool['high_interaction'])}, "
                f"AR={len(route_pool['alternative_branch'])}, "
                f"BG={len(route_pool['background'])}; "
                f"5-interact={summaries['5']['mean_interacting_path_fraction']:.1%}; "
                f"7-interact={summaries['7']['mean_interacting_path_fraction']:.1%}",
                flush=True,
            )
        write_json(payload, destination)

    for problem, entry in zip(problems, payload["maps"]):
        validate_map_entry(
            problem,
            entry,
            config["route_pool_counts"],
            config["density_schemes"],
            config["scene_acceptance"],
        )
    write_audit_csv(payload, ROOT / config["audit_csv"])
    preview_root = ROOT / config["preview_root"]
    preview_root.mkdir(parents=True, exist_ok=True)
    for problem, entry in zip(problems, payload["maps"]):
        render_candidate_pool(
            problem, entry, preview_root / f"{problem.map_id}_candidate_pool.png"
        )
        render_scene_examples(
            problem,
            entry,
            5,
            preview_root / f"{problem.map_id}_5_obstacle_examples.png",
        )
        render_scene_examples(
            problem,
            entry,
            7,
            preview_root / f"{problem.map_id}_7_obstacle_examples.png",
        )
        render_summary_table(
            problem, entry, preview_root / f"{problem.map_id}_coverage_summary.png"
        )
    print(
        f"Dataset: {destination}\nAudit CSV: {config['audit_csv']}\n"
        f"Previews: {preview_root}\nNo training started. Adaptive unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    main()
