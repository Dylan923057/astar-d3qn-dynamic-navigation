"""Plot matched-seed adaptation results, retaining failures and censoring."""
from __future__ import annotations

import argparse
import csv
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

from astar_d3qn.utils.io import write_json, write_records_csv


def paired_effect(reference, treatment):
    """Positive loss means persistent replay performed worse than 0% replay."""
    return {
        "validation_auc_loss": reference["validation_conflict_auc"] - treatment["validation_conflict_auc"],
        "test_conflict_safe_success_loss": reference["test"]["conflict_safe_success"] - treatment["test"]["conflict_safe_success"],
        "test_paired_failure_gap_increase": treatment["test"]["conflict_minus_control_failure"] - reference["test"]["conflict_minus_control_failure"],
        "test_probe_unsafe_increase": treatment["test"]["conflict_probe_unsafe_action"] - reference["test"]["conflict_probe_unsafe_action"],
        "test_static_retention_loss": reference["test"]["static_safe_success"] - treatment["test"]["static_safe_success"],
    }


def bootstrap_mean(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"mean": float(values[0]), "ci95": None, "n": len(values)}
    rng = np.random.default_rng(917)
    means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "n": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/replay_adaptation_v2.yaml")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = args.output_root
    manifest = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    entry = manifest["maps"][args.map_index]
    map_id = entry["problem"]["map_id"]
    root = ROOT / config["output_root"] / ("smoke" if args.smoke else "formal") / map_id
    seeds = args.seeds if args.seeds is not None else ([0] if args.smoke else config["training_seeds"])
    fractions = config["demo_fractions"]
    status, complete = [], {}
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        status_path = seed_root / "foundation/foundation_status.json"
        foundation = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        branches = {}
        for fraction in fractions:
            path = seed_root / f"demo_{round(100 * fraction):02d}/result.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("status") == "complete" and result.get("smoke") == args.smoke:
                    branches[fraction] = result
        matched = (len(branches) == len(fractions) and bool(foundation.get("qualified") or args.smoke))
        if matched:
            provenance_keys = ("fork_sha256", "config_sha256", "manifest_sha256", "code_sha256", "torch_version", "device")
            for key in provenance_keys:
                if len({result[key] for result in branches.values()}) != 1:
                    raise SystemExit(f"Unmatched {key} in seed {seed}; refusing aggregation.")
            if next(iter(branches.values()))["fork_sha256"] != foundation.get("snapshot_sha256"):
                raise SystemExit("Branch does not match its recorded foundation.")
            complete[seed] = branches
        status.append({"seed": seed, "foundation_status": foundation.get("status", "missing"),
                       "completed_branches": len(branches), "matched_comparison_available": matched})
    # Cross-seed provenance must also match, except foundation/RNG and seed itself.
    for key in ("config_sha256", "manifest_sha256", "code_sha256", "torch_version", "device"):
        if len({branches[fractions[0]][key] for branches in complete.values()}) > 1:
            raise SystemExit(f"Incompatible {key} across seeds.")
    destination = root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    write_records_csv(status, destination / "seed_status.csv")
    report = {"map_id": map_id, "smoke": args.smoke, "requested_seeds": seeds,
              "matched_seeds": list(complete), "seed_status": status,
              "status": ("smoke_only" if args.smoke else "complete" if len(complete) == len(seeds) else "incomplete"),
              "inference_unit": "paired training seed, conditional on a qualified static foundation",
              "interpretation": "Positive loss indicates a replay allocation cost. Not proof of gradient conflict.",
              "small_sample_warning": len(complete) < 5,
              "primary_metric": "validation conflict safe-success AUC over equal post-fork interaction budget"}
    if not complete:
        write_json(report, destination / "summary.json")
        print(f"No matched completed seeds; inspect {destination / 'seed_status.csv'}")
        return
    effects, methods = [], []
    for seed, branches in complete.items():
        for fraction, result in branches.items():
            methods.append({"seed": seed, "fraction": fraction,
                            "effective_fraction": result["effective_fraction"],
                            "validation_conflict_auc": result["validation_conflict_auc"],
                            "threshold_confirmation_step": result["threshold_confirmation_step"],
                            "threshold_right_censored": result["threshold_right_censored"],
                            "visible_dynamic_steps": result["adaptation_run"]["visible_dynamic_steps"],
                            "reference_risk_steps": result["adaptation_run"]["reference_risk_steps"],
                            "sampled_visible_dynamic": result["adaptation_run"]["sampled_visible_dynamic"],
                            **{f"test_{k}": v for k, v in result["test"].items()}})
            if fraction:
                effects.append({"seed": seed, "fraction": fraction, **paired_effect(branches[0.0], result)})
    report["paired_effects"] = {}
    for fraction in fractions:
        if not fraction:
            continue
        rows = [row for row in effects if row["fraction"] == fraction]
        report["paired_effects"][str(fraction)] = {
            key: bootstrap_mean([row[key] for row in rows]) for key in rows[0] if key not in ("seed", "fraction")}
    write_json(report, destination / "summary.json")
    write_records_csv(methods, destination / "per_seed_metrics.csv")
    write_records_csv(effects, destination / "paired_effects.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [("conflict_safe_success", "Conflict safe success"),
               ("control_safe_success", "Control safe success"),
               ("conflict_probe_unsafe_action", "Unsafe action at common conflict states"),
               ("static_safe_success", "Static retention")]
    for fraction in fractions:
        curves = []
        for seed in complete:
            path = root / f"seed_{seed}/demo_{round(100 * fraction):02d}/validation_curve.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                curves.append(list(csv.DictReader(handle)))
        x = np.asarray([int(row["environment_steps"]) for row in curves[0]])
        if any([int(row["environment_steps"]) for row in curve] != x.tolist() for curve in curves[1:]):
            raise SystemExit("Checkpoint grids differ; cannot average curves.")
        for ax, (key, title) in zip(axes.flat, metrics):
            y = np.asarray([[float(row[key]) for row in curve] for curve in curves])
            line, = ax.plot(x, y.mean(axis=0), label=f"{fraction:.0%} requested demo")
            for individual in y:
                ax.plot(x, individual, color=line.get_color(), alpha=.18, lw=.7)
            ax.set(title=title, xlabel="Environment steps since common fork", ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle(f"{'SMOKE — no scientific inference' if args.smoke else map_id} | matched seeds={len(complete)}/{len(seeds)}\n"
                 "Heavy lines: mean; faint lines: individual seeds")
    fig.tight_layout()
    fig.savefig(destination / "adaptation_curves.png", dpi=150)
    plt.close(fig)
    print(f"Analysis: {destination}\nStatus: {report['status']}; matched seeds: {list(complete)}")


if __name__ == "__main__":
    main()
