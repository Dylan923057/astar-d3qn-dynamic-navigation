"""Generate and audit the independent replay-adaptation dataset; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from astar_d3qn.core.grid import obstacle_grid
from astar_d3qn.maps.adaptation import irregular_problem, scenario_pairs
from astar_d3qn.maps.io import problem_record, problem_from_record
from astar_d3qn.utils.io import write_json, write_records_csv


def draw_map(ax, problem):
    ax.imshow(obstacle_grid(problem.obstacles, problem.size), cmap="Greys", vmin=0, vmax=1,
              origin="upper", interpolation="none")
    path = np.asarray(problem.nominal_path)
    ax.plot(path[:, 1], path[:, 0], "--", color="#2478b5", lw=1.4, label="Static A* reference")
    ax.scatter(problem.start[1], problem.start[0], s=65, c="#269a42", zorder=5, label="Start")
    ax.scatter(problem.goal[1], problem.goal[0], s=120, marker="*", c="#efbf27",
               edgecolors="black", zorder=5, label="Goal")
    ax.set(xlim=(-.5, problem.size - .5), ylim=(problem.size - .5, -.5), xlabel="column", ylabel="row")
    ax.set_xticks(range(0, problem.size, 5))
    ax.set_yticks(range(0, problem.size, 5))
    ax.grid(alpha=.15)


def previews(payload, destination):
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(payload["maps"]), figsize=(7 * len(payload["maps"]), 7), squeeze=False)
    for ax, entry in zip(axes[0], payload["maps"]):
        problem = problem_from_record(entry["problem"])
        draw_map(ax, problem)
        ax.set_title(f"{problem.map_id}\n{problem.size}x{problem.size} | occupancy {problem.density:.1%} | A* {problem.astar_steps}")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "map_overview.png", dpi=150)
    plt.close(fig)
    for entry in payload["maps"]:
        problem = problem_from_record(entry["problem"])
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        for ax, (split, pairs) in zip(axes, entry["scenarios"]["splits"].items()):
            draw_map(ax, problem)
            unique = {tuple(map(tuple, pair["route"])) for pair in pairs}
            colors = {"early": "#cf7930", "middle": "#9763b9", "late": "#13948e"}
            drawn = set()
            for pair in pairs:
                route = np.asarray(pair["route"])
                band = pair.get("progress_band", "middle")
                ax.plot(route[:, 1], route[:, 0], color=colors[band], lw=2, alpha=.75,
                        label=band if band not in drawn else None)
                drawn.add(band)
                position = pair["decision_position"]
                ax.scatter(position[1], position[0], c=colors[band], s=12, zorder=6)
            band_counts = [sum(p.get("progress_band") == b for p in pairs) for b in colors]
            ax.set_title(f"{split}: {len(pairs)} pairs, {len(unique)} routes\n"
                         f"early/middle/late={band_counts}; ONE obstacle per episode")
            ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(destination / f"{problem.map_id}_routes.png", dpi=150)
        plt.close(fig)
        # A pair from each split: same route, different initial phase; witness overlays.
        fig, axes = plt.subplots(3, 2, figsize=(13, 18))
        for row, (split, pairs) in enumerate(entry["scenarios"]["splits"].items()):
            target_band = ("early", "middle", "late")[row]
            examples = [p for p in pairs if p.get("progress_band") == target_band] or pairs
            pair = examples[len(examples) // 2]
            for col, condition in enumerate(("control", "conflict")):
                ax = axes[row, col]
                draw_map(ax, problem)
                spec = pair[condition]["obstacle"]
                route = np.asarray(spec["route"])
                ax.plot(route[:, 1], route[:, 0], "o-", color="#e28335", markersize=3, label="Obstacle route")
                initial = route[spec["start_index"]]
                ax.scatter(initial[1], initial[0], c="#c23434", s=80, zorder=6, label="Obstacle at t=0")
                if condition == "conflict":
                    bypass = np.asarray(pair["bypass_witness"])
                    ax.plot(bypass[:, 1], bypass[:, 0], color="#8b52a1", lw=1.2, alpha=.8, label="Safe local bypass")
                    p = pair["decision_position"]
                    ax.scatter(p[1], p[0], marker="s", facecolors="none", edgecolors="#269a42", s=100,
                               zorder=7, label="Visible wait/bypass decision")
                ax.set_title(f"{split} / {pair.get('progress_band', '')} / {pair['pair_id']} / {condition}\n"
                             f"reference wait delay {pair['wait_steps'] if col else 0}; "
                             f"demo conflicts {pair[condition]['demo_conflict_count']}/{payload['demo_episodes']}")
                ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(destination / f"{problem.map_id}_examples.png", dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/replay_adaptation_v2.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    path = ROOT / config["dataset"]
    design = {k: config[k] for k in ("map_seeds", "map_size", "demo_seed", "demo_episodes", "pair_counts", "window_size")}
    if "coverage" in config:
        design["coverage"] = config["coverage"]
    signature = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["design_sha256"] != signature:
            raise SystemExit("Frozen dataset differs from config. Use a NEW dataset path/version.")
        print(f"Using existing frozen dataset: {path}", flush=True)
    else:
        payload = {"format_version": 1, "protocol": config["experiment"], "design": design,
                   "design_sha256": signature, "demo_episodes": config["demo_episodes"], "maps": []}
        for seed in config["map_seeds"]:
            problem = irregular_problem(seed, config["map_size"], config["demo_seed"])
            print(f"Auditing {problem.map_id}: {problem.astar_steps} steps, {problem.density:.1%} occupancy", flush=True)
            scenes = scenario_pairs(problem, config["pair_counts"], config["demo_seed"], config["demo_episodes"],
                                    config["window_size"] // 2, coverage=config.get("coverage"))
            payload["maps"].append({"problem": problem_record(problem), "scenarios": scenes})
            print(scenes["audit"], flush=True)
        # Publish only after every map and its full scenario set pass.
        write_json(payload, path)
    destination = ROOT / config.get("preview_root", "maps/previews/replay_adaptation_v1")
    previews(payload, destination)
    audit_rows = [{"map_id": entry["problem"]["map_id"], "split": split,
                   "pair_id": pair["pair_id"], "decision_step": pair["first_reference_collision_step"],
                   "decision_row": pair["decision_position"][0], "decision_column": pair["decision_position"][1],
                   "progress_band": pair.get("progress_band"), "route_kind": pair.get("route_kind"),
                   "route_length": len(pair["route"])}
                  for entry in payload["maps"] for split, pairs in entry["scenarios"]["splits"].items() for pair in pairs]
    write_records_csv(audit_rows, path.parent / "coverage_audit.csv")
    print(f"Dataset: {path}\nPreviews: {destination}\nNo training has been started.", flush=True)


if __name__ == "__main__":
    main()
