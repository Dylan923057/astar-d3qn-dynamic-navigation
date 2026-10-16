"""Compose and audit the multi-obstacle risk-handover benchmark; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from astar_d3qn.core.grid import obstacle_grid
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.maps.risk_handover import compose_pairs
from astar_d3qn.utils.io import write_json, write_records_csv


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_map(ax, problem):
    ax.imshow(
        obstacle_grid(problem.obstacles, problem.size),
        cmap="Greys",
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="none",
    )
    path = np.asarray(problem.nominal_path)
    ax.plot(path[:, 1], path[:, 0], "--", color="#2478b5", lw=1.2, label="Static A* reference")
    ax.scatter(problem.start[1], problem.start[0], s=55, c="#269a42", zorder=5, label="Start")
    ax.scatter(problem.goal[1], problem.goal[0], s=100, marker="*", c="#efbf27",
               edgecolors="black", zorder=5, label="Goal")
    ax.set(xlim=(-.5, problem.size - .5), ylim=(problem.size - .5, -.5), xlabel="column", ylabel="row")
    ax.grid(alpha=.15)


def render_previews(payload, destination):
    destination.mkdir(parents=True, exist_ok=True)
    colors = ["#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#e69f00", "#7b61a8", "#555555"]
    for entry in payload["maps"]:
        problem = problem_from_record(entry["problem"])
        fig, axes = plt.subplots(2, 3, figsize=(19, 12))
        for column, split in enumerate(("train", "validation", "test")):
            pairs = entry["scenarios"]["splits"][split]
            density_counts = Counter(pair["obstacle_count"] for pair in pairs)
            ax = axes[0, column]
            draw_map(ax, problem)
            for pair in pairs:
                primary = pair["conflict"]["obstacles"][0]
                route = np.asarray(primary["route"])
                ax.plot(route[:, 1], route[:, 0], color=colors[pair["obstacle_count"] - 1], alpha=.45)
            ax.set_title(f"{split}: {len(pairs)} paired scenes\ndensity={dict(sorted(density_counts.items()))}")
            ax.legend(fontsize=7, loc="upper right")

            example = max(pairs, key=lambda pair: pair["obstacle_count"])
            ax = axes[1, column]
            draw_map(ax, problem)
            for index, spec in enumerate(example["conflict"]["obstacles"]):
                route = np.asarray(spec["route"])
                ax.plot(route[:, 1], route[:, 0], "o-", ms=2.5, lw=1.5,
                        color=colors[index], label=f"obstacle {index + 1}")
                initial = route[spec["start_index"]]
                ax.scatter(initial[1], initial[0], color=colors[index], s=35, zorder=6)
            oracle = np.asarray(example["conflict"]["oracle_path"])
            ax.plot(oracle[:, 1], oracle[:, 0], color="#6a3d9a", lw=1.5, alpha=.8, label="safe oracle")
            ax.set_title(f"{split} example: {example['obstacle_count']} obstacles, "
                         f"{example['causal_obstacle_count']} causal")
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{problem.map_id}: multi-obstacle risk-handover benchmark")
        fig.tight_layout()
        fig.savefig(destination / f"{problem.map_id}_multi_obstacle.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_dataset"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    design = {
        key: config[key]
        for key in (
            "source_dataset", "map_seeds", "map_size", "demo_seed", "demo_episodes",
            "pair_counts", "scenario_densities", "causal_obstacles", "window_size",
        )
    }
    design["source_sha256"] = sha_file(source_path)
    signature = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    destination = ROOT / config["dataset"]
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload["design_sha256"] != signature:
            raise SystemExit("Frozen risk-handover dataset differs from config; use a new version.")
        print(f"Using existing frozen dataset: {destination}", flush=True)
    else:
        payload = {
            "format_version": 1,
            "protocol": config["experiment"],
            "design": design,
            "design_sha256": signature,
            "maps": [],
        }
        for map_index, source_entry in enumerate(source["maps"]):
            problem = problem_from_record(source_entry["problem"])
            splits = {}
            for split, source_pairs in source_entry["scenarios"]["splits"].items():
                print(f"{problem.map_id} {split}: composing densities "
                      f"{config['scenario_densities'][split]}", flush=True)
                splits[split] = compose_pairs(
                    problem,
                    source_pairs,
                    config["scenario_densities"][split],
                    config["causal_obstacles"],
                    seed=problem.seed + 81000 + map_index,
                    max_steps=config["max_episode_steps"],
                )
                expected = config["pair_counts"][split]
                if len(splits[split]) != expected:
                    raise RuntimeError(f"{split}: expected {expected} pairs, got {len(splits[split])}.")
            payload["maps"].append({
                "problem": source_entry["problem"],
                "scenarios": {
                    "splits": splits,
                    "audit": {
                        "source_protocol": source["protocol"],
                        "paired_control_conflict": True,
                        "all_scenes_oracle_replayed": True,
                        "train_validation_densities": config["scenario_densities"]["train"],
                        "test_density_sweep": config["scenario_densities"]["test"],
                    },
                },
            })
        write_json(payload, destination)

    render_previews(payload, ROOT / config["preview_root"])
    rows = []
    for entry in payload["maps"]:
        for split, pairs in entry["scenarios"]["splits"].items():
            for pair in pairs:
                rows.append({
                    "map_id": entry["problem"]["map_id"],
                    "split": split,
                    "pair_id": pair["pair_id"],
                    "source_pair_id": pair["source_pair_id"],
                    "obstacle_count": pair["obstacle_count"],
                    "causal_obstacle_count": pair["causal_obstacle_count"],
                    "first_reference_collision_step": pair["first_reference_collision_step"],
                    "control_oracle_steps": pair["control"]["oracle_steps"],
                    "conflict_oracle_steps": pair["conflict"]["oracle_steps"],
                })
    write_records_csv(rows, destination.parent / "scenario_audit.csv")
    print(f"Dataset: {destination}\nPreviews: {config['preview_root']}\nNo training started.", flush=True)


if __name__ == "__main__":
    main()
