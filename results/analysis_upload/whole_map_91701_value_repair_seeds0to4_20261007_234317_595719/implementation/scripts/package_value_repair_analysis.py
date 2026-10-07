"""Package completed formal results for reading; never trains or changes source runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_runtime_path_pilot import load_csv

METHODS = ("unguided_raw", "unguided_bound", "advice_raw", "advice_bound",
           "advice_margin", "advice_bound_margin")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def package(analysis):
    source = ROOT / "outputs/whole_map_91701_value_repair_v1"
    verification = json.loads((analysis / "verification.json").read_text(encoding="utf-8"))
    seeds = tuple(verification["seeds"])
    if (verification["mode"] != "formal" or seeds not in ((0, 1), (0, 1, 2, 3, 4))
            or verification["methods"] != list(METHODS) or verification["test_data_used"]
            or not all(verification[key] for key in ("all_recorded_checkpoints_verified",
                                                     "all_50_scenes_identical", "all_failure_traces_inspected"))
            or Path(verification["source_root"]).resolve() != source.resolve()):
        raise ValueError("Use a fully audited six-method formal analysis (seeds 0,1 or 0-4).")
    seed_label = "seeds01" if seeds == (0, 1) else "seeds0to4"
    destination = ROOT / "results/analysis_upload" / (
        "whole_map_91701_value_repair_" + seed_label + "_" + analysis.name.removeprefix("analysis_"))
    if destination.exists():
        raise FileExistsError(f"Preserve the existing package: {destination}")
    results = {}
    common_hashes = None
    for seed in seeds:
        for method in METHODS:
            directory = source / method / f"seed_{seed}"
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
            if (not completion["complete"] or completion["environment_steps"] != 200000
                    or result["training_mode"] != "formal" or result["environment_steps"] != 200000
                    or result["gradient_updates"] != 199501 or result["test_data_used"]
                    or result["evaluation_advice"] or result["method"] != method or result["seed"] != seed
                    or result["initial_state_sha256"] != verification["paired_initial_state_sha256"][str(seed)]):
                raise ValueError(f"Run identity, completion or protocol mismatch: {directory}")
            hashes = result["source_sha256"]
            if common_hashes is None:
                common_hashes = hashes
            if hashes != common_hashes or any(sha256(ROOT / name) != digest for name, digest in hashes.items()):
                raise ValueError("Recorded implementation hashes differ or source files changed.")
            results[method, seed] = result
    destination.mkdir(parents=True)
    provenance = {}

    def copy(original, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        provenance[Path(relative).as_posix()] = original.relative_to(ROOT).as_posix()

    analysis_files = ("REPORT.md", "summary.csv", "method_means.csv", "all_validation_checkpoints.csv",
                      "effective_config.json", "verification.json", "supplementary_verification.json",
                      "navigation_replay_summary.csv", "navigation_replay_scenes.csv",
                      "dynamic_observation_action_changes.csv", "navigation_replay_verification.json",
                      "validation_curves.png", "validation_curves.pdf")
    for name in analysis_files:
        copy(analysis / name, name)
    supplemental = ("withdrawal_summary.json", "withdrawal_checkpoints.csv", "phase_metrics.csv",
                    "paired_differences.csv", "seed3_collision_action_inspection.json",
                    "mean_validation_curves.png", "mean_validation_curves.pdf",
                    "value_learning_curves.png", "value_learning_curves.pdf")
    for name in supplemental:
        if (analysis / name).is_file():
            copy(analysis / name, name)
    failures = load_csv(analysis / "all_failure_behaviors.csv")
    # Keep every failure and every behavior metric; repeated 60-position tails are unnecessary here.
    compact_failures = [{k: v for k, v in row.items() if k != "tail_60_steps"} for row in failures]
    write_csv(destination / "all_failure_behaviors.csv", compact_failures)
    values, index, weight_hashes = [], [], []
    run_files = ("training.csv", "validation_curve.csv", "validation_details.csv", "value_learning.csv",
                 "advice_budget.csv", "result.json", "run_manifest.json", "completion.json", "static_final.json")

    def select_trace(method, seed, step, criterion, rationale):
        directory = source / method / f"seed_{seed}"
        candidates = [r for r in failures if r["method"] == method and int(r["seed"]) == seed
                      and int(r["environment_steps_total"]) == step and criterion(r)]
        if not candidates:
            raise ValueError(f"No trajectory matches the declared selection: {method}/{seed}/{step}")
        chosen = candidates[0]
        original = directory / f"validation_failures_{step:06d}.json"
        trace = next(t for t in json.loads(original.read_text(encoding="utf-8"))
                     if t["scenario_id"] == chosen["scenario_id"])
        store_trace(method, seed, step, trace, original, rationale)

    def store_trace(method, seed, step, trace, original, rationale):
        filename = f"trajectories/{method}_seed{seed}_step{step}_{trace['scenario_id']}.json"
        if any(row["file"] == filename for row in index):
            return
        write_json(destination / filename, {"method": method, "seed": seed,
                   "environment_steps_total": step, "selection_reason": rationale,
                   "source_file": original.relative_to(ROOT).as_posix(), "source_sha256": sha256(original),
                   "trace": trace})
        index.append({"method": method, "seed": seed, "environment_steps_total": step,
                      "scenario_id": trace["scenario_id"], "termination_reason": trace["termination_reason"],
                      "selection_reason": rationale, "file": filename})

    for seed in seeds:
        for method in METHODS:
            directory = source / method / f"seed_{seed}"
            for name in run_files:
                copy(directory / name, f"runs/{method}/seed_{seed}/{name}")
            values.extend({"method": method, "seed": seed, **row}
                          for row in load_csv(directory / "value_learning.csv"))
            weight_hashes.append({"method": method, "seed": seed,
                                  "source": (directory / "model_final.pth").relative_to(ROOT).as_posix(),
                                  "sha256": sha256(directory / "model_final.pth"), "included": False})
    write_csv(destination / "all_value_learning.csv", values)
    # First recorded example of each observed final failure phenotype, in every run.
    phenotypes = (("dynamic collision", lambda r: r["termination_reason"] == "collision"),
                  ("waiting-dominated timeout", lambda r: r["termination_reason"] == "timeout"
                   and r["waiting_dominated"] == "True"),
                  ("repeated-movement timeout", lambda r: r["termination_reason"] == "timeout"
                   and r["repeated_movement"] == "True"),
                  ("other timeout", lambda r: r["termination_reason"] == "timeout"
                   and r["waiting_dominated"] != "True" and r["repeated_movement"] != "True"))
    for seed in seeds:
        for method in METHODS:
            final_rows = [r for r in failures if r["method"] == method and int(r["seed"]) == seed
                          and int(r["environment_steps_total"]) == 200000]
            for label, criterion in phenotypes:
                if any(criterion(r) for r in final_rows):
                    select_trace(method, seed, 200000, criterion, "First final " + label + " in this run")
    if 3 in seeds:
        original = source / "advice_bound_margin/seed_3/validation_failures_200000.json"
        for trace in json.loads(original.read_text(encoding="utf-8")):
            store_trace("advice_bound_margin", 3, 200000, trace, original,
                        "All four final seed3 combination collisions, including adverse counterexamples")
    checkpoints = load_csv(analysis / "all_validation_checkpoints.csv")
    for seed in seeds:
        near_exit = [r for r in checkpoints if r["method"] == "advice_bound_margin"
                     and int(r["seed"]) == seed and 90000 <= int(r["environment_steps_total"]) <= 120500]
        worst = min(near_exit, key=lambda r: (float(r["safe_success_rate"]), int(r["environment_steps_total"])))
        step = int(worst["environment_steps_total"])
        select_trace("advice_bound_margin", seed, step, lambda r: True,
                     "First failure at the lowest-success recorded checkpoint within 90000-120500 steps")
    changes = load_csv(analysis / "dynamic_observation_action_changes.csv")
    for method in ("advice_bound", "advice_bound_margin"):
        for seed in seeds:
            original = analysis / f"navigation_examples_{method}_seed{seed}.json"
            avoided = {r["scenario_id"] for r in changes if r["method"] == method and int(r["seed"]) == seed
                       and r["avoided_one_step_collision"] == "True"}
            trace = next(t for t in json.loads(original.read_text(encoding="utf-8"))
                         if t["safe_success"] and t["scenario_id"] in avoided)
            store_trace(method, seed, 200000, trace, original,
                        "First final success with a documented dynamic-sensitive one-step collision avoidance")
    write_csv(destination / "trajectory_index.csv", index)
    for name in common_hashes:
        copy(ROOT / name, "implementation/" + Path(name).as_posix())
    # These supporting sources are packaging-time snapshots, not additional recorded run hashes.
    for original in sorted((ROOT / "src/astar_d3qn").rglob("*.py")):
        copy(original, "implementation/" + original.relative_to(ROOT).as_posix())
    for name in ("scripts/analyze_value_repair.py", "scripts/inspect_value_repair_navigation.py",
                 "scripts/analyze_runtime_path_pilot.py", "scripts/package_value_repair_analysis.py"):
        copy(ROOT / name, "implementation/" + name)
    import yaml
    config_files = set()
    for name in common_hashes:
        if Path(name).suffix != ".yaml":
            continue
        current = ROOT / name
        while current not in config_files:
            config_files.add(current)
            copy(current, "implementation/" + current.relative_to(ROOT).as_posix())
            parent = yaml.safe_load(current.read_text(encoding="utf-8")).get("inherits")
            if parent is None:
                break
            current = current.parent / parent
    config = next(iter(results.values()))["config"]
    copy(ROOT / config["dataset"]["validation_reference"], "fixed_validation_scenarios.json")
    original_map = json.loads((ROOT / config["dataset"]["source_manifest"]).read_text(encoding="utf-8"))
    problem = next(m["problem"] for m in original_map["maps"] if m["problem"]["map_id"] == config["dataset"]["map_id"])
    write_json(destination / "map_problem.json", problem)
    payload = {p.relative_to(destination).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
               for p in sorted(destination.rglob("*")) if p.is_file()}
    write_json(destination / "manifest.json", {"source_root": source.relative_to(ROOT).as_posix(),
               "analysis_root": analysis.relative_to(ROOT).as_posix(), "methods": list(METHODS), "seeds": list(seeds),
               "mode": "formal", "environment_steps_per_run": 200000, "training_started_by_packager": False,
               "test_data_used": False, "raw_source_outputs_unchanged": True,
               "failure_rows_retained": len(compact_failures), "omitted_failure_column": "tail_60_steps",
               "selected_trajectory_count": len(index), "copied_file_provenance": provenance,
               "source_sha256": common_hashes, "excluded_model_fingerprints": weight_hashes,
               "supporting_source_scope": "Additional implementation files are packaging-time snapshots; only source_sha256 records run-time hashes.",
               "files": payload, "manifest_scope": "Payload before README and GPT prompt are added; manifest excludes itself."})
    print(destination)
    print(f"Payload: {len(payload)} files, {sum(r['bytes'] for r in payload.values())} bytes; no training or Git mutation.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    package(args.analysis.resolve())


if __name__ == "__main__":
    main()
