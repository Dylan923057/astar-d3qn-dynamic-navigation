"""Orchestrate the frozen three-map, five-seed A/B validation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from astar_d3qn.training.replay_adaptation import decay_demo_fraction, state_digest
from astar_d3qn.utils.io import write_json


MAP_IDS = (
    "irregular_workcell_91701",
    "irregular_workcell_91702",
    "irregular_workcell_91703",
)
SEEDS = (0, 1, 2, 3, 4)
OUTPUT_ROOT = "outputs/ab_five_seed_strict_v1"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_code_sha256() -> str:
    runner = ROOT / "scripts" / "run_replay_adaptation.py"
    files = sorted((ROOT / "src" / "astar_d3qn").rglob("*.py")) + [runner]
    return state_digest({
        str(path.relative_to(ROOT)): sha_file(path)
        for path in files
    })


def frozen_registration(config_path: Path, output_root: str) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "data/risk_handover_v1/manifest.json",
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
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"Frozen config mismatch for {key}: {config.get(key)!r}")
    if [decay_demo_fraction(step) for step in (1, 50_000, 50_001, 100_000, 100_001, 200_000)] != [
        0.25, 0.25, 0.10, 0.10, 0.0, 0.0
    ]:
        raise SystemExit("The time-decay schedule is no longer 25% -> 10% -> 0%.")
    manifest_path = ROOT / config["dataset"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    map_ids = tuple(entry["problem"]["map_id"] for entry in manifest["maps"])
    if map_ids != MAP_IDS:
        raise SystemExit(f"Frozen map order changed: {map_ids}")
    return {
        "protocol": "ab_five_seed_strict_v1",
        "validation_only": True,
        "test_generation_forbidden": True,
        "maps": list(MAP_IDS),
        "seeds": list(SEEDS),
        "methods": {
            "A": "time_decay",
            "B": "global_prediction",
        },
        "foundation_steps": int(config["foundation"]["max_steps"]),
        "adaptation_steps": 200_000,
        "evaluation_interval": 10_000,
        "time_decay_boundaries": {
            "steps_1_to_50000": 0.25,
            "steps_50001_to_100000": 0.10,
            "steps_100001_to_200000": 0.0,
        },
        "prediction_loss_weight": 0.1,
        "prediction_pos_weight": 20.0,
        "global_prediction_spatial_weight": 1.0,
        "prediction_head_hidden_dim": 128,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_file_sha256": sha_file(config_path),
        "manifest_sha256": sha_file(manifest_path),
        "training_code_sha256": current_code_sha256(),
        "orchestrator_sha256": sha_file(Path(__file__)),
        "output_root": output_root,
        "raw_files_required": [
            "training.csv",
            "validation_curve.csv",
            "validation_details.csv",
            "validation_trajectories.json",
            "result.json",
        ],
    }


def registration_path(output_root: str) -> Path:
    return ROOT / output_root / "protocol_registration.json"


def verify_no_test_artifacts(output_root: str) -> None:
    root = ROOT / output_root
    if not root.exists():
        return
    forbidden = [
        path for path in root.rglob("*")
        if path.is_file()
        and (path.name.startswith("test_") or path.name == "test_trajectories.json")
    ]
    if forbidden:
        raise SystemExit(f"Test artifacts are forbidden: {forbidden[0]}")
    for result_path in root.rglob("result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "test" in result or not result.get("test_deferred"):
            raise SystemExit(f"Non-validation-only result found: {result_path}")


def verify_registration(registration: dict, *, create: bool, dry_run: bool) -> None:
    path = registration_path(registration["output_root"])
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != registration:
            raise SystemExit(
                "Protocol registration differs from current code/config; use a new output root."
            )
    elif create:
        if not dry_run:
            write_json(registration, path)
    else:
        raise SystemExit("Run --stage preflight before any training stage.")
    verify_no_test_artifacts(registration["output_root"])


def verify_foundations(registration: dict) -> None:
    root = ROOT / registration["output_root"] / "formal"
    for map_id in MAP_IDS:
        for seed in SEEDS:
            checkpoint = root / map_id / f"seed_{seed}" / "foundation" / "foundation.pt"
            if not checkpoint.exists():
                raise SystemExit(f"Missing strict foundation: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metadata = payload["metadata"]
            if (
                not metadata.get("qualified")
                or metadata.get("code_sha256") != registration["training_code_sha256"]
                or metadata.get("manifest_sha256") != registration["manifest_sha256"]
                or metadata.get("map_id") != map_id
            ):
                raise SystemExit(f"Foundation provenance mismatch: {checkpoint}")


def verify_method_complete(registration: dict, branch: str, schedule: str) -> None:
    root = ROOT / registration["output_root"] / "formal"
    for map_id in MAP_IDS:
        for seed in SEEDS:
            directory = root / map_id / f"seed_{seed}" / branch
            result_path = directory / "result.json"
            if not result_path.exists():
                raise SystemExit(f"Missing completed {schedule} result: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("status") != "validation_complete"
                or not result.get("test_deferred")
                or "test" in result
                or result.get("adaptation_run", {}).get("steps") != 200_000
                or result.get("replay_schedule") != schedule
                or result.get("code_sha256") != registration["training_code_sha256"]
                or result.get("manifest_sha256") != registration["manifest_sha256"]
            ):
                raise SystemExit(f"Completed result violates the frozen protocol: {result_path}")
            for name in registration["raw_files_required"]:
                if not (directory / name).is_file():
                    raise SystemExit(f"Missing required raw artifact: {directory / name}")


def command_for(
    stage: str,
    map_index: int,
    output_root: str,
    device: str,
    threads: int,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_replay_adaptation.py"),
        "--config", "configs/risk_handover_v1.yaml",
        "--map-index", str(map_index),
        "--seeds", *(str(seed) for seed in SEEDS),
        "--device", device,
        "--threads", str(threads),
        "--output-root", output_root,
        "--foundation-source-root", output_root,
        "--defer-test",
    ]
    if stage == "foundation":
        command.extend(("--schedule", "fixed", "--stage", "foundation"))
    elif stage == "time_decay":
        command.extend(("--schedule", "decay", "--stage", "adapt"))
    elif stage == "global_prediction":
        command.extend((
            "--schedule", "global_prediction",
            "--stage", "adapt",
            "--prediction-loss-weight", "0.1",
            "--prediction-pos-weight", "20",
            "--prediction-head-hidden-dim", "128",
            "--prediction-decision-zone-size", "5",
            "--prediction-decision-zone-weight", "1",
        ))
    else:
        raise ValueError(stage)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "foundation", "time_decay", "global_prediction"),
        required=True,
    )
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        raise SystemExit("threads must be positive.")
    registration = frozen_registration(ROOT / args.config, args.output_root)
    verify_registration(
        registration,
        create=args.stage == "preflight",
        dry_run=args.dry_run,
    )
    if args.stage == "preflight":
        print(json.dumps(registration, ensure_ascii=False, indent=2))
        return
    if args.stage in {"time_decay", "global_prediction"}:
        verify_foundations(registration)
    if args.stage == "global_prediction":
        verify_method_complete(registration, "schedule_decay", "decay")
    commands = [
        command_for(
            args.stage,
            map_index,
            args.output_root,
            args.device,
            args.threads,
        )
        for map_index in range(len(MAP_IDS))
    ]
    for command in commands:
        print("COMMAND", shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        if args.stage == "foundation":
            verify_foundations(registration)
        elif args.stage == "time_decay":
            verify_method_complete(registration, "schedule_decay", "decay")
        elif args.stage == "global_prediction":
            verify_method_complete(
                registration, "global_prediction", "global_prediction"
            )


if __name__ == "__main__":
    main()
