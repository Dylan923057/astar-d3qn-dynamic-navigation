"""Audit the two completed maps without training or checkpoint selection."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.replay_adaptation import (
    evaluate, flatten_pairs, make_agent, seed_everything, state_digest,
)
from astar_d3qn.utils.io import write_json, write_records_csv
from summarize_replay_adaptation import bootstrap_mean


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(curve, key, budget):
    times = np.array([int(row["environment_steps"]) for row in curve])
    values = np.array([float(row[key]) for row in curve])
    return float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2) / budget)


def main():
    torch.set_num_threads(1)
    config = yaml.safe_load((ROOT / "configs/replay_adaptation_v2.yaml").read_text(encoding="utf-8"))
    manifest_path = ROOT / config["dataset"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    code_files = sorted((ROOT / "src/astar_d3qn").rglob("*.py")) + [ROOT / "scripts/run_replay_adaptation.py"]
    provenance = {
        "config_sha256": state_digest(config), "manifest_sha256": sha(manifest_path),
        "code_sha256": state_digest({str(p.relative_to(ROOT)): sha(p) for p in code_files}),
        "torch_version": str(torch.__version__), "device": "cuda",
    }
    budget = config["adaptation"]["max_steps"]
    destination = ROOT / config["output_root"] / "replication_map02"
    records, checks, details_out, sources, curves = [], [], [], {}, {}
    for map_index in (0, 1):
        entry = manifest["maps"][map_index]
        problem = problem_from_record(entry["problem"])
        scenes = flatten_pairs(entry["scenarios"]["splits"]["validation"])
        root = ROOT / config["output_root"] / "formal" / problem.map_id
        for seed in config["training_seeds"]:
            foundation = json.loads((root / f"seed_{seed}/foundation/foundation_status.json").read_text())
            assert foundation["qualified"] and not foundation["smoke"]
            for fraction in config["demo_fractions"]:
                branch = root / f"seed_{seed}/demo_{round(100 * fraction):02d}"
                result = json.loads((branch / "result.json").read_text())
                assert result["status"] == "complete" and not result["smoke"]
                for key, value in provenance.items():
                    assert result[key] == value, (branch, key)
                assert result["fork_sha256"] == foundation["snapshot_sha256"]
                assert result["map_id"] == problem.map_id and result["grid_sha256"] == problem.grid_sha256
                assert result["seed"] == seed and result["fraction"] == fraction
                assert result["adaptation_run"]["steps"] == result["gradient_updates_since_fork"] == budget
                curve = read_csv(branch / "validation_curve.csv")
                assert [int(r["environment_steps"]) for r in curve] == list(range(0, budget + 1, config["adaptation"]["evaluation_interval"]))
                identity = {"map_index": map_index, "seed": seed, "fraction": fraction}
                record = {**identity, "effective_fraction": result["effective_fraction"]}
                for condition in ("conflict", "control"):
                    for metric in ("safe_success", "dynamic_collision", "timeout"):
                        key = f"{condition}_{metric}"
                        record[f"auc_{key}"] = auc(curve, key, budget)
                        record[f"final_validation_{key}"] = float(curve[-1][key])
                        record[f"test_{key}"] = result["test"][key]
                    assert all(np.isclose(sum(float(row[f"{condition}_{k}"]) for k in ("safe_success", "dynamic_collision", "timeout")), 1) for row in curve)
                record["auc_static_safe_success"] = auc(curve, "static_safe_success", budget)
                record["threshold_confirmation_step"] = result["threshold_confirmation_step"]
                assert np.isclose(record["auc_conflict_safe_success"], result["validation_conflict_auc"])
                final_rows = [r for r in read_csv(branch / "validation_details.csv") if int(r["environment_steps"]) == budget]
                assert {r["scenario_id"] for r in final_rows} == {s["scenario_id"] for s in scenes}
                assert len(final_rows) == len(scenes)
                # The final validation row is evaluated before test with no intervening updates.
                # Reload map 2's saved final weights to independently confirm that identity.
                if map_index == 1:
                    seed_everything(seed)
                    agent = make_agent(config, seed, "cuda")
                    agent.load_weights(branch / "model_final.pth")
                    summary, fresh_rows, _ = evaluate(agent, problem, config, scenes)
                    for key, value in summary.items():
                        saved = curve[-1][key]
                        matches = (not saved) if value is None else np.isclose(float(saved), value, rtol=1e-5, atol=1e-6)
                        assert matches, (branch, key, saved, value)
                    by_id = {r["scenario_id"]: r for r in final_rows}
                    for row in fresh_rows:
                        for key in ("safe_success", "dynamic_collision", "static_collision", "timeout", "steps", "wait_steps", "probe_unsafe_action", "probe_wait_action"):
                            assert float(by_id[row["scenario_id"]][key]) == row[key], (branch, row["scenario_id"], key)
                        details_out.append({**identity, **row})
                    checks.append({**identity, "final_weights_match_logged_validation": True, "scenarios": len(fresh_rows)})
                    print(f"Verified map 2 seed={seed} fraction={fraction:.0%}: final validation matches", flush=True)
                records.append(record)
                curves[(map_index, seed, fraction)] = curve
                for filename in ("result.json", "validation_curve.csv", "validation_details.csv", "test_evaluation.csv", "model_final.pth"):
                    path = branch / filename
                    sources[str(path.relative_to(ROOT))] = sha(path)
    metric_keys = [k for k in records[0] if k.startswith(("auc_", "final_validation_", "test_"))]
    paired, summary = [], {}
    for map_index in (0, 1):
        rows = [r for r in records if r["map_index"] == map_index]
        summary[str(map_index)] = {"means": {}, "paired": {}}
        for fraction in config["demo_fractions"]:
            subset = [r for r in rows if r["fraction"] == fraction]
            summary[str(map_index)]["means"][str(fraction)] = {k: float(np.mean([r[k] for r in subset])) for k in metric_keys}
        for fraction in config["demo_fractions"][1:]:
            diffs = []
            for seed in config["training_seeds"]:
                base = next(r for r in rows if r["seed"] == seed and r["fraction"] == 0)
                treatment = next(r for r in rows if r["seed"] == seed and r["fraction"] == fraction)
                diff = {"map_index": map_index, "seed": seed, "fraction": fraction, **{k: treatment[k] - base[k] for k in metric_keys}}
                diffs.append(diff)
                paired.append(diff)
            summary[str(map_index)]["paired"][str(fraction)] = {
                k: {**bootstrap_mean([d[k] for d in diffs]),
                    "positive": sum(d[k] > 1e-10 for d in diffs),
                    "tie": sum(abs(d[k]) <= 1e-10 for d in diffs),
                    "negative": sum(d[k] < -1e-10 for d in diffs)} for k in metric_keys}
    write_records_csv(records, destination / "per_seed_metrics.csv")
    write_records_csv(paired, destination / "paired_differences.csv")
    write_records_csv(details_out, destination / "map02_final_validation_details.csv")
    write_json({"provenance": provenance, "source_sha256": sources, "final_validation_checks": checks,
                "difference_direction": "replay minus 0%; higher success is better; higher collision/timeout is worse",
                "inference_unit": "paired training seed; 5 per map; descriptive bootstrap CI is fragile at n=5",
                "maps": summary}, destination / "summary.json")
    plot(records, curves, destination)
    print(f"Audit complete: {destination}")


def plot(records, curves, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, map_index in enumerate((0, 1)):
        for fraction in (0., .1, .25):
            subset = [r for r in records if r["map_index"] == map_index and r["fraction"] == fraction]
            for col, metric in enumerate(("safe_success", "dynamic_collision", "timeout")):
                data = [curves[(map_index, r["seed"], fraction)] for r in subset]
                x = [int(r["environment_steps"]) for r in data[0]]
                y = np.array([[float(r[f"conflict_{metric}"]) for r in curve] for curve in data])
                line, = axes[row, col].plot(x, y.mean(axis=0), label=f"{fraction:.0%}")
                for individual in y:
                    axes[row, col].plot(x, individual, color=line.get_color(), lw=.6, alpha=.2)
                axes[row, col].set(title=f"Map {map_index + 1}: {metric}", ylim=(-.03, 1.03), xlabel="Post-fork environment steps")
                axes[row, col].grid(alpha=.2)
                axes[row, col].legend()
    fig.suptitle("Same frozen conflict validation set across checkpoints | bold: mean, faint: seeds")
    fig.tight_layout()
    fig.savefig(destination / "conflict_outcome_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
