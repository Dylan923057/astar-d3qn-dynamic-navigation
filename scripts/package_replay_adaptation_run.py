"""Package replay-adaptation outputs into one self-contained comparison folder."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astar_d3qn.maps.adaptation import spec_from_record
from astar_d3qn.maps.io import problem_from_record
from render_replay_adaptation import draw_map, font, marker, point

METHODS = (("fixed_00", "fixed 0%", "demo_00"),
           ("fixed_10", "fixed 10%", "demo_10"),
           ("fixed_25", "fixed 25%", "demo_25"),
           ("decay", "25% -> 10% -> 0%", "schedule_decay"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def status(row):
    if row["safe_success"]:
        return "success"
    if row["dynamic_collision"]:
        return "dynamic_collision"
    if row["timeout"]:
        return "timeout"
    return "other"


def comparison_image(problem, pair, trajectories, destination, scenario_id):
    panel_w, panel_h = 390, 440
    image = Image.new("RGB", (panel_w * len(METHODS), panel_h * 5), "#f6f8fb")
    for seed_index, seed in enumerate(sorted(trajectories)):
        for method_index, (key, label, _) in enumerate(METHODS):
            row = trajectories[seed][key]
            panel = Image.new("RGB", (panel_w, panel_h), "#f6f8fb")
            draw = ImageDraw.Draw(panel)
            draw.text((12, 8), f"seed {seed} | {label}", font=font(16), fill="#263344")
            draw.text((12, 31), f"{status(row)} | steps={row['steps']}", font=font(14), fill="#263344")
            origin, scale = (38, 53), 8.2
            draw_map(draw, problem, origin, scale, pair["route"])
            route = row["positions"]
            if len(route) > 1:
                draw.line([point(p, origin, scale) for p in route], fill="#176cc1", width=2)
            marker(draw, route[0], origin, scale, "#176cc1", radius=scale * .35)
            marker(draw, route[-1], origin, scale, "#176cc1", radius=scale * .35)
            spec = spec_from_record(pair["conflict"]["obstacle"])
            final_t = min(row["steps"], len(row["dynamic_positions"]) - 1)
            for position in row["dynamic_positions"][final_t]:
                marker(draw, position, origin, scale, "#df4242", square=True)
            image.paste(panel, (method_index * panel_w, seed_index * panel_h))
    image.save(destination / f"final_paths_{scenario_id}.png")


def package_map(map_index, seeds, scenario_id, manifest, source_root, destination_root):
    entry = manifest["maps"][map_index]
    problem = problem_from_record(entry["problem"])
    source = source_root / "formal" / problem.map_id
    destination = destination_root / f"map_{map_index + 1:02d}_{problem.map_id}"
    destination.mkdir(parents=True, exist_ok=True)
    pair_id, condition = scenario_id.rsplit("_", 1)
    pair = next(p for p in entry["scenarios"]["splits"]["test"] if p["pair_id"] == pair_id)
    metrics, curves, final_routes = [], [], []
    trajectories = {seed: {} for seed in seeds}
    for seed in seeds:
        for key, label, branch_name in METHODS:
            branch = source / f"seed_{seed}" / branch_name
            result = read_json(branch / "result.json")
            curve_path = branch / "validation_curve.csv"
            with curve_path.open(encoding="utf-8", newline="") as handle:
                curve_rows = list(csv.DictReader(handle))
            for curve_row in curve_rows:
                curves.append({"seed": seed, "method": key, "method_label": label,
                               "map_id": problem.map_id, **curve_row})
            metrics.append({"map_id": problem.map_id, "seed": seed, "method": key,
                            "method_label": label,
                            "validation_conflict_auc": result["validation_conflict_auc"],
                            "threshold_confirmation_step": result["threshold_confirmation_step"],
                            "effective_fraction": result["effective_fraction"],
                            "test_conflict_safe_success": result["test"]["conflict_safe_success"],
                            "test_conflict_dynamic_collision": result["test"]["conflict_dynamic_collision"],
                            "test_conflict_timeout": result["test"]["conflict_timeout"],
                            "test_control_safe_success": result["test"]["control_safe_success"],
                            "adaptation_steps": result["adaptation_run"]["steps"]})
            trajectories[seed][key] = next(r for r in read_json(branch / "test_trajectories.json")
                                           if r["scenario_id"] == scenario_id)
            row = trajectories[seed][key]
            final_routes.append({"map_id": problem.map_id, "seed": seed, "method": key,
                                 "method_label": label, "scenario_id": scenario_id,
                                 "status": status(row), "steps": row["steps"],
                                 "safe_success": row["safe_success"],
                                 "dynamic_collision": row["dynamic_collision"],
                                 "timeout": row["timeout"]})
    write_csv(metrics, destination / "metrics_comparison.csv")
    write_csv(curves, destination / "validation_curves.csv")
    write_csv(final_routes, destination / "final_routes.csv")
    comparison_image(problem, pair, trajectories, destination, scenario_id)
    shutil.copy2(ROOT / "configs/replay_adaptation_v2.yaml", destination / "config.yaml")
    shutil.copy2(ROOT / "data/replay_adaptation_v2/manifest.json", destination / "manifest.json")
    (destination / "run_info.json").write_text(json.dumps({
        "map_index": map_index, "map_id": problem.map_id, "seeds": seeds,
        "scenario_id": scenario_id, "methods": [label for _, label, _ in METHODS],
        "dynamic_obstacles_per_episode": 1,
        "training_sampling": "uniform shuffled cycle over 36 train scenarios; no curriculum",
    }, indent=2), encoding="utf-8")
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--map-index", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--scenario-id", default="pair_032_conflict")
    parser.add_argument("--source-root", default="outputs/replay_adaptation_v2")
    args = parser.parse_args()
    manifest = read_json(ROOT / "data/replay_adaptation_v2/manifest.json")
    source_root = ROOT / args.source_root
    destination_root = source_root / "runs" / args.run_id
    destination_root.mkdir(parents=True, exist_ok=True)
    for map_index in args.map_index:
        package_map(map_index, args.seeds, args.scenario_id, manifest, source_root, destination_root)
    (destination_root / "run_info.json").write_text(json.dumps({
        "run_id": args.run_id, "maps": args.map_index, "seeds": args.seeds,
        "scenario_id": args.scenario_id, "source_root": str(source_root),
    }, indent=2), encoding="utf-8")
    print(destination_root)


if __name__ == "__main__":
    main()
