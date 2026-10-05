"""Run v2 with an independent train-monitor; preflight never trains."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.capability_adaptive_auc import sha_file, state_digest
from astar_d3qn.training.capability_adaptive_v2 import train_capability_adaptive_v2


MAP_IDS = ("irregular_workcell_91701", "irregular_workcell_91702", "irregular_workcell_91703")
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_STEPS = tuple(range(0, 200_001, 10_000))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def package_code_sha256() -> str:
    paths = sorted((ROOT / "src" / "astar_d3qn").rglob("*.py")) + [Path(__file__)]
    records = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha_file(path) for path in paths
    }
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def foundation_path(map_id: str, seed: int) -> Path:
    return ROOT / "foundations" / map_id / f"seed_{seed}" / "foundation.pt"


def validate_config(config: dict, manifest: dict, source_protocol: dict) -> None:
    expected = {
        "experiment": "capability_adaptive_handover_v2_independent_monitor",
        "dataset": "data/risk_handover_v1/manifest.json",
        "output_root": "results/capability_adaptive_handover_v2_independent_monitor",
        "map_seeds": [91701, 91702, 91703],
        "map_size": 40,
        "demo_seed": 7400,
        "demo_episodes": 20,
        "pair_counts": {"train": 36, "validation": 12, "test": 48},
        "window_size": 15,
        "max_episode_steps": 300,
        "replay_capacity": 10_000,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "target_sync_interval": 250,
        "hidden_dim": 256,
        "reward": {"step": -0.01, "progress": 0.05, "stay": 0.0, "collision": -1.0, "goal": 10.0},
        "adaptation": {"max_steps": 200_000, "evaluation_interval": 10_000, "epsilon_start": 0.30, "epsilon_end": 0.05, "epsilon_decay_steps": 150_000, "safe_success_threshold": 0.90, "consecutive_passes": 2},
        "capability_adaptive_handover": {"monitor_scene_count": 24, "rho_max": 0.25, "beta": 0.3, "tau": 0.90, "consecutive_confirmations": 2, "initial_competence": 0.0},
        "train_monitor": {"generation_seed": 521001, "source_split": "train", "independent_scene_count": 24, "densities": {3: 12, 5: 12}, "causal_obstacles": {3: 1, 5: 2}, "epsilon": 0.0},
        "validation_recording": {"split": "validation", "evaluation_interval": 10_000, "epsilon": 0.0, "controls_training": False, "evaluate_step_zero": True},
        "training_seeds": [0, 1, 2, 3, 4],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"Frozen v2 config mismatch for {key}: {config.get(key)!r}")
    if sha_file(ROOT / config["dataset"]) != source_protocol["manifest_sha256"]:
        raise SystemExit("Copied dataset manifest does not match strict A/B protocol.")
    if tuple(entry["problem"]["map_id"] for entry in manifest["maps"]) != MAP_IDS:
        raise SystemExit("Map order mismatch.")
    for entry in manifest["maps"]:
        if {key: len(value) for key, value in entry["scenarios"]["splits"].items()} != config["pair_counts"]:
            raise SystemExit(f"Scenario split count mismatch: {entry['problem']['map_id']}")


def validate_foundation(map_id: str, seed: int, inventory: dict, source_protocol: dict, deep: bool):
    path = foundation_path(map_id, seed)
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    expected = inventory["files"].get(relative)
    if expected is None or not path.is_file() or path.stat().st_size != expected["bytes"] or sha_file(path) != expected["sha256"]:
        raise SystemExit(f"Foundation checksum mismatch: {relative}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    required = {"qualified": True, "smoke": False, "seed": seed, "map_id": map_id, "manifest_sha256": source_protocol["manifest_sha256"], "code_sha256": source_protocol["training_code_sha256"], "torch_version": str(torch.__version__)}
    for key, value in required.items():
        if metadata.get(key) != value:
            raise SystemExit(f"Foundation metadata mismatch {relative}: {key}")
    if deep and state_digest(checkpoint["state"]) != metadata.get("snapshot_sha256"):
        raise SystemExit(f"Foundation full-state fingerprint mismatch: {relative}")
    return checkpoint


def validate_smoke(output_dir: Path) -> None:
    import csv

    run_dir = output_dir / "formal" / MAP_IDS[0] / "seed_0" / "capability_adaptive_handover"
    required = ["training.csv", "episodes.csv", "validation_curve.csv", "validation_final.csv", "result.json", "run_audit.json", "monitor_scene_manifest.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Smoke output missing: {missing}")
    audit = load_json(run_dir / "run_audit.json")
    for key, value in {"source_protocol": "ab_five_seed_strict_v1", "verified_full_state_equal": True, "training_split_unchanged": True, "monitor_scenes_remain_in_training_stream": False, "monitor_scenes_in_training_stream": False, "monitor_scenes_in_replay": False, "validation_controls_training": False, "test_read_or_generated": False}.items():
        if audit.get(key) != value:
            raise SystemExit(f"Smoke audit mismatch: {key}={audit.get(key)!r}")
    manifest = load_json(run_dir / "monitor_scene_manifest.json")
    if manifest.get("scene_count") != 24 or manifest.get("density_counts") != {"3": 12, "5": 12}:
        raise SystemExit("Smoke monitor manifest count mismatch.")
    with (run_dir / "validation_curve.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        validation = list(csv.DictReader(handle))
    if tuple(int(row["environment_steps"]) for row in validation) != EXPECTED_STEPS:
        raise SystemExit("Smoke validation curve must contain step 0..200k.")
    with (run_dir / "training.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        training = list(csv.DictReader(handle))
    if len(training) != 20:
        raise SystemExit("Smoke training.csv must contain 20 monitor checkpoints.")
    previous_rho = 0.25
    for row in training:
        rho = float(row["demo_fraction"])
        if rho > previous_rho + 1e-12:
            raise SystemExit("Smoke demo fraction is not monotone.")
        previous_rho = rho
        step = int(row["training_environment_steps"])
        cumulative = int(row["monitor_environment_steps_cumulative"])
        if int(row["total_algorithm_environment_interactions"]) != step + cumulative:
            raise SystemExit("Smoke interaction accounting mismatch.")
    if not any(int(row["monitor_environment_steps_current"]) > 0 for row in training):
        raise SystemExit("Smoke monitor interaction count is zero.")
    print(f"smoke_passed run={run_dir}", flush=True)


def main(*, config_default="configs/capability_adaptive_handover_v2_independent_monitor.yaml",
         method="capability_adaptive_handover_v2_independent_monitor",
         config_validator=validate_config, trainer=train_capability_adaptive_v2) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "run"), required=True)
    parser.add_argument("--config", default=config_default)
    parser.add_argument("--map-indices", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--deep-foundation-check", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        raise SystemExit("threads must be positive")
    if args.stage == "smoke":
        map_indices = [0] if args.map_indices is None else args.map_indices
        seeds = [0] if args.seeds is None else args.seeds
    else:
        map_indices = [0, 1, 2] if args.map_indices is None else args.map_indices
        seeds = list(SEEDS) if args.seeds is None else args.seeds
    if len(set(map_indices)) != len(map_indices) or any(index not in (0, 1, 2) for index in map_indices):
        raise SystemExit("map-indices must be distinct members of 0,1,2")
    if len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        raise SystemExit("seeds must be distinct members of 0..4")
    torch.set_num_threads(args.threads)
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["dataset"]
    manifest = load_json(manifest_path)
    source_protocol = load_json(ROOT / "foundations" / "source_protocol_registration.json")
    inventory = load_json(ROOT / "foundations" / "inventory.json")
    config_validator(config, manifest, source_protocol)
    foundation_count = 0
    for index in map_indices:
        for seed in seeds:
            checkpoint = validate_foundation(MAP_IDS[index], seed, inventory, source_protocol, args.deep_foundation_check)
            foundation_count += 1
            print(f"foundation_ok map={MAP_IDS[index]} seed={seed}", flush=True)
            del checkpoint
    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        resolved_device = "cpu"
    summary = {"stage": args.stage, "maps": [MAP_IDS[i] for i in map_indices], "seeds": seeds, "foundation_count": foundation_count, "requested_device": resolved_device, "cuda_available": torch.cuda.is_available(), "package_code_sha256": package_code_sha256(), "manifest_sha256": sha_file(manifest_path), "source_protocol": "ab_five_seed_strict_v1", "method": config["capability_adaptive_handover"], "train_monitor": config["train_monitor"], "validation_recording": config["validation_recording"], "formal_training_started": args.stage != "preflight"}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "preflight":
        return
    if resolved_device != "cuda":
        raise SystemExit("The copied foundations require CUDA; use --device cuda on the training computer.")
    output_root = ROOT / config["output_root"]
    provenance = {"package_code_sha256": summary["package_code_sha256"], "config_sha256": sha_file(config_path), "manifest_sha256": summary["manifest_sha256"], "source_training_code_sha256": source_protocol["training_code_sha256"], "source_protocol": "ab_five_seed_strict_v1"}
    for index in map_indices:
        entry = manifest["maps"][index]
        problem = problem_from_record(entry["problem"])
        splits = copy.deepcopy(entry["scenarios"]["splits"])
        for seed in seeds:
            output_dir = output_root / "formal" / problem.map_id / f"seed_{seed}" / "capability_adaptive_handover"
            result_path = output_dir / "result.json"
            if result_path.is_file():
                result = load_json(result_path)
                if result.get("method") == method and result.get("adaptation_steps") == 200_000:
                    print(f"already_complete {output_dir}", flush=True)
                    continue
                raise SystemExit(f"Existing incompatible output preserved: {result_path}")
            if output_dir.exists():
                raise SystemExit(f"Incomplete output preserved; use a clean package copy: {output_dir}")
            checkpoint = validate_foundation(problem.map_id, seed, inventory, source_protocol, False)
            trainer(problem, config, splits, seed, resolved_device, output_dir, checkpoint, provenance)
            del checkpoint
    if args.stage == "smoke":
        validate_smoke(output_root)


if __name__ == "__main__":
    main()
