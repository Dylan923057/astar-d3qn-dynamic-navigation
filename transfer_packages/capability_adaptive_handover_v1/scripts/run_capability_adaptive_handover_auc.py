"""Run capability-adaptive handover with observation-only validation curves."""

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
from astar_d3qn.training.capability_adaptive_auc import (
    sha_file,
    state_digest,
    train_capability_adaptive_auc,
)


MAP_IDS = (
    "irregular_workcell_91701",
    "irregular_workcell_91702",
    "irregular_workcell_91703",
)
SEEDS = (0, 1, 2, 3, 4)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def directory_digest(root: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\x00")
        digest.update(bytes.fromhex(sha_file(path)))
        digest.update(b"\x00")
    return len(files), sum(path.stat().st_size for path in files), digest.hexdigest()


def package_code_sha256() -> str:
    paths = sorted((ROOT / "src" / "astar_d3qn").rglob("*.py")) + [Path(__file__)]
    records = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha_file(path)
        for path in paths
    }
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_config_and_manifest(config, manifest, source_protocol):
    expected = {
        "experiment": "capability_adaptive_handover_v1_auc",
        "dataset": "data/risk_handover_v1/manifest.json",
        "output_root": "outputs/capability_adaptive_handover_v1_auc",
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
        "reward": {
            "step": -0.01,
            "progress": 0.05,
            "stay": 0.0,
            "collision": -1.0,
            "goal": 10.0,
        },
        "adaptation": {
            "max_steps": 200_000,
            "evaluation_interval": 10_000,
            "epsilon_start": 0.30,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 150_000,
            "safe_success_threshold": 0.90,
            "consecutive_passes": 2,
        },
        "capability_adaptive_handover": {
            "monitor_scene_count": 12,
            "rho_max": 0.25,
            "beta": 0.3,
            "tau": 0.90,
            "consecutive_confirmations": 2,
            "initial_competence": 0.0,
        },
        "validation_recording": {
            "split": "validation",
            "evaluation_interval": 10_000,
            "epsilon": 0.0,
            "controls_training": False,
            "evaluate_step_zero": True,
        },
        "training_seeds": [0, 1, 2, 3, 4],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"Frozen config mismatch for {key}: {config.get(key)!r}")
    map_ids = tuple(entry["problem"]["map_id"] for entry in manifest["maps"])
    if map_ids != MAP_IDS:
        raise SystemExit(f"Frozen map order mismatch: {map_ids}")
    for entry in manifest["maps"]:
        splits = entry["scenarios"]["splits"]
        counts = {name: len(splits[name]) for name in ("train", "validation", "test")}
        if counts != config["pair_counts"]:
            raise SystemExit(f"Scenario split count mismatch for {entry['problem']['map_id']}: {counts}")
    manifest_path = ROOT / config["dataset"]
    if sha_file(manifest_path) != source_protocol["manifest_sha256"]:
        raise SystemExit("Copied dataset manifest does not match the frozen A/B protocol.")


def foundation_path(map_id: str, seed: int) -> Path:
    return ROOT / "foundations" / map_id / f"seed_{seed}" / "foundation.pt"


def validate_foundation_file(
    map_id: str,
    seed: int,
    inventory: dict,
    source_protocol: dict,
    *,
    deep: bool,
):
    path = foundation_path(map_id, seed)
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    expected = inventory["files"].get(relative)
    if expected is None or not path.is_file():
        raise SystemExit(f"Missing inventoried foundation: {relative}")
    if path.stat().st_size != expected["bytes"] or sha_file(path) != expected["sha256"]:
        raise SystemExit(f"Foundation copy checksum mismatch: {relative}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    required = {
        "qualified": True,
        "smoke": False,
        "seed": seed,
        "map_id": map_id,
        "manifest_sha256": source_protocol["manifest_sha256"],
        "code_sha256": source_protocol["training_code_sha256"],
        "torch_version": str(torch.__version__),
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise SystemExit(
                f"Foundation metadata mismatch for {relative} ({key}={metadata.get(key)!r})."
            )
    if deep and state_digest(checkpoint["state"]) != metadata.get("snapshot_sha256"):
        raise SystemExit(f"Foundation state fingerprint mismatch: {relative}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--config", default="configs/capability_adaptive_handover_v1_auc.yaml"
    )
    parser.add_argument("--map-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--deep-foundation-check",
        action="store_true",
        help="Also recompute each checkpoint's internal full-state fingerprint.",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        raise SystemExit("threads must be positive.")
    if len(set(args.map_indices)) != len(args.map_indices) or any(
        index < 0 or index >= len(MAP_IDS) for index in args.map_indices
    ):
        raise SystemExit("map-indices must be distinct members of 0, 1, 2.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in SEEDS for seed in args.seeds):
        raise SystemExit("seeds must be distinct members of 0, 1, 2, 3, 4.")

    torch.set_num_threads(args.threads)
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["dataset"]
    manifest = load_json(manifest_path)
    source_protocol = load_json(ROOT / "foundations" / "source_protocol_registration.json")
    inventory = load_json(ROOT / "foundations" / "inventory.json")
    validate_config_and_manifest(config, manifest, source_protocol)
    baseline_root = ROOT / "baselines" / "time_decay"
    baseline_inventory = load_json(
        ROOT / "baselines" / "time_decay_inventory.json"
    )
    baseline_count, baseline_bytes, baseline_sha256 = directory_digest(baseline_root)
    if (
        baseline_count != baseline_inventory["file_count"]
        or baseline_bytes != baseline_inventory["total_bytes"]
        or baseline_sha256 != baseline_inventory["combined_sha256"]
    ):
        raise SystemExit("Copied time_decay baseline checksum mismatch.")

    foundation_count = 0
    for map_index in args.map_indices:
        map_id = MAP_IDS[map_index]
        for seed in args.seeds:
            checkpoint = validate_foundation_file(
                map_id,
                seed,
                inventory,
                source_protocol,
                deep=args.deep_foundation_check,
            )
            foundation_count += 1
            baseline_result = load_json(
                baseline_root / map_id / f"seed_{seed}" / "result.json"
            )
            if (
                baseline_result.get("status") != "validation_complete"
                or baseline_result.get("replay_schedule") != "decay"
                or baseline_result.get("adaptation_run", {}).get("steps")
                != 200_000
                or baseline_result.get("test_deferred") is not True
                or baseline_result.get("map_id") != map_id
                or baseline_result.get("seed") != seed
                or baseline_result.get("snapshot_sha256")
                != checkpoint["metadata"].get("snapshot_sha256")
            ):
                raise SystemExit(
                    f"time_decay baseline pairing mismatch: {map_id} seed={seed}"
                )
            print(f"foundation_ok map={map_id} seed={seed}", flush=True)
            del checkpoint

    resolved_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    summary = {
        "stage": args.stage,
        "maps": [MAP_IDS[index] for index in args.map_indices],
        "seeds": args.seeds,
        "foundation_count": foundation_count,
        "foundation_device": "cuda",
        "requested_device": resolved_device,
        "cuda_available": torch.cuda.is_available(),
        "package_code_sha256": package_code_sha256(),
        "manifest_sha256": sha_file(manifest_path),
        "time_decay_baseline_sha256": baseline_sha256,
        "method": config["capability_adaptive_handover"],
        "validation_recording": config["validation_recording"],
        "formal_training_started": args.stage == "run",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "preflight":
        return
    if resolved_device != "cuda":
        raise SystemExit("The copied foundations were created on cuda; run with CUDA available.")

    output_root = ROOT / config["output_root"]
    provenance = {
        "package_code_sha256": summary["package_code_sha256"],
        "config_sha256": sha_file(config_path),
        "manifest_sha256": summary["manifest_sha256"],
        "source_training_code_sha256": source_protocol["training_code_sha256"],
        "source_protocol": source_protocol["protocol"],
    }
    for map_index in args.map_indices:
        entry = manifest["maps"][map_index]
        problem = problem_from_record(entry["problem"])
        splits = copy.deepcopy(entry["scenarios"]["splits"])
        for seed in args.seeds:
            output_dir = output_root / "formal" / problem.map_id / f"seed_{seed}" / "capability_adaptive_handover"
            result_path = output_dir / "result.json"
            if result_path.is_file():
                result = load_json(result_path)
                if (
                    result.get("status") == "validation_complete"
                    and result.get("method") == "capability_adaptive_handover"
                    and result.get("adaptation_steps") == config["adaptation"]["max_steps"]
                    and result.get("test_deferred") is True
                ):
                    print(f"already_complete {output_dir}", flush=True)
                    continue
                raise SystemExit(f"Existing result has incompatible provenance: {result_path}")
            if output_dir.exists():
                raise SystemExit(
                    f"Incomplete output is preserved; choose a clean package copy: {output_dir}"
                )
            checkpoint = validate_foundation_file(
                problem.map_id,
                seed,
                inventory,
                source_protocol,
                deep=False,
            )
            train_capability_adaptive_auc(
                problem,
                config,
                splits,
                seed,
                resolved_device,
                output_dir,
                checkpoint,
                provenance,
            )
            del checkpoint


if __name__ == "__main__":
    main()
