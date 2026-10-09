"""Render saved final test routes for the scheduled replay branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.maps.adaptation import occupancy, spec_from_record
from render_replay_adaptation import draw_map, point, marker, font


def status(row):
    if row["safe_success"]:
        return "success"
    if row["dynamic_collision"]:
        return "dynamic collision"
    if row["timeout"]:
        return "timeout"
    return "other termination"


def render_map(map_index, seeds, scenario_id, manifest):
    entry = manifest["maps"][map_index]
    problem = problem_from_record(entry["problem"])
    map_root = ROOT / "outputs/replay_adaptation_v2/formal" / problem.map_id
    destination = map_root / "visualizations_schedule"
    destination.mkdir(parents=True, exist_ok=True)
    pair_id, condition = scenario_id.rsplit("_", 1)
    pair = next(p for p in entry["scenarios"]["splits"]["test"] if p["pair_id"] == pair_id)
    panels = []
    metadata = []
    for seed in seeds:
        source = map_root / f"seed_{seed}/schedule_decay/test_trajectories.json"
        trajectories = json.loads(source.read_text(encoding="utf-8"))
        row = next(r for r in trajectories if r["scenario_id"] == scenario_id)
        image = Image.new("RGB", (760, 760), "#f6f8fb")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), f"{problem.map_id} | seed {seed} | schedule 25%→10%→0%", font=font(23), fill="#263344")
        draw.text((24, 52), f"{scenario_id} | {status(row)} | steps={row['steps']}", font=font(18), fill="#263344")
        origin, scale = (75, 95), 15
        draw_map(draw, problem, origin, scale, pair["route"])
        route = row["positions"]
        if len(route) > 1:
            draw.line([point(p, origin, scale) for p in route], fill="#176cc1", width=4)
        for t, position in enumerate(route):
            if t in (0, len(route) - 1):
                marker(draw, position, origin, scale, "#176cc1", radius=scale * .33)
        obstacle = spec_from_record(pair[condition]["obstacle"])
        final_t = min(row["steps"], len(row["dynamic_positions"]) - 1)
        for position in row["dynamic_positions"][final_t]:
            marker(draw, position, origin, scale, "#df4242", square=True)
        draw.text((24, 710), "blue: robot route | orange: obstacle route | red: final obstacle position", font=font(16), fill="#263344")
        image.save(destination / f"seed_{seed}_{scenario_id}.png")
        panels.append(image)
        metadata.append({"seed": seed, "scenario_id": scenario_id, "status": status(row), "steps": row["steps"]})
    sheet = Image.new("RGB", (1520, 1520), "#ffffff")
    for index, image in enumerate(panels):
        sheet.paste(image.resize((760, 760)), ((index % 2) * 760, (index // 2) * 760))
    sheet.save(destination / f"all_seeds_{scenario_id}.png")
    (destination / "metadata.json").write_text(json.dumps({"map_id": problem.map_id,
        "scenario_id": scenario_id, "seeds": seeds, "routes": metadata}, indent=2), encoding="utf-8")
    print(destination / f"all_seeds_{scenario_id}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-index", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--scenario-id", default="pair_032_conflict")
    args = parser.parse_args()
    manifest_path = ROOT / "data/replay_adaptation_v2/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for map_index in args.map_index:
        render_map(map_index, args.seeds, args.scenario_id, manifest)


if __name__ == "__main__":
    main()
