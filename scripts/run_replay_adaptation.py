"""Train one static foundation per seed, then fork fixed or scheduled replay."""
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
from astar_d3qn.training.replay_adaptation import train_foundation, train_branch, state_digest
from astar_d3qn.utils.io import write_json


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/replay_adaptation_v2.yaml")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--fractions", nargs="+", type=float)
    parser.add_argument(
        "--schedule",
        choices=("fixed", "decay", "risk_handover", "safe_intervention"),
        default="fixed",
        help="Use fixed fractions, time decay, risk handover, or safe intervention replay",
    )
    parser.add_argument(
        "--safe-samples",
        nargs="+",
        type=int,
        help="Safe transitions per batch; safe_intervention defaults to 0 4 8",
    )
    parser.add_argument("--stage", choices=("foundation", "adapt", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", help="Separate run namespace; never overwrites an existing experiment")
    parser.add_argument("--reuse-foundation", action="store_true",
                        help="Reuse an existing qualified foundation when only the adaptation code changed")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = args.output_root
    manifest_path = ROOT / config["dataset"]
    if not manifest_path.exists():
        raise SystemExit("First run: python scripts/prepare_replay_adaptation.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in manifest["design"].items():
        if key == "source_sha256":
            continue
        if key not in config or config[key] != value:
            raise SystemExit(f"Frozen dataset/config mismatch: {key}")
    if not 0 <= args.map_index < len(manifest["maps"]):
        raise SystemExit("map-index out of range")
    entry = manifest["maps"][args.map_index]
    problem = problem_from_record(entry["problem"])
    scenarios = copy.deepcopy(entry["scenarios"]["splits"])
    seeds = args.seeds if args.seeds is not None else ([0] if args.smoke else config["training_seeds"])
    if args.schedule != "fixed" and args.fractions is not None:
        raise SystemExit("--fractions can only be combined with --schedule fixed.")
    if args.schedule != "safe_intervention" and args.safe_samples is not None:
        raise SystemExit("--safe-samples requires --schedule safe_intervention.")
    safe_samples = args.safe_samples if args.safe_samples is not None else [0, 4, 8]
    if args.schedule == "safe_intervention" and (
        len(set(safe_samples)) != len(safe_samples)
        or any(count < 0 or count >= config["batch_size"] for count in safe_samples)
    ):
        raise SystemExit("Safe sample counts must be distinct and in [0, batch_size).")
    fractions = args.fractions if args.fractions is not None else config["demo_fractions"]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise SystemExit("Seeds must be distinct nonnegative integers.")
    if args.schedule == "fixed" and (len(set(fractions)) != len(fractions)
                                      or any(f not in config["demo_fractions"] for f in fractions)):
        raise SystemExit("Fractions must be distinct members of the registered protocol.")
    if args.smoke:
        config["batch_size"], config["hidden_dim"] = 8, 32
        config["max_episode_steps"] = 80
        for stage in ("foundation", "adaptation"):
            config[stage]["max_steps"] = 16
            config[stage]["evaluation_interval"] = 8
            config[stage]["epsilon_decay_steps"] = 16
        scenarios = {key: value[:1] for key, value in scenarios.items()}
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    code_files = sorted((ROOT / "src/astar_d3qn").rglob("*.py")) + [Path(__file__)]
    code_hash = state_digest({str(p.relative_to(ROOT)): sha_file(p) for p in code_files})
    provenance = {"config_sha256": state_digest(config), "manifest_sha256": sha_file(manifest_path),
                  "code_sha256": code_hash, "torch_version": str(torch.__version__),
                  "map_id": problem.map_id, "grid_sha256": problem.grid_sha256}
    destination = ROOT / config["output_root"] / ("smoke" if args.smoke else "formal") / problem.map_id
    print(json.dumps({"output": str(destination), "seeds": seeds, "fractions": fractions,
                      "schedule": args.schedule,
                      "safe_samples": safe_samples if args.schedule == "safe_intervention" else None,
                      "static_steps_max": config["foundation"]["max_steps"],
                      "adaptation_steps_per_branch": config["adaptation"]["max_steps"],
                      "device": device, "smoke": args.smoke,
                      "fork": "same policy, target, optimizer, replay contents and RNG states",
                      "test_checkpoint": "fixed-budget final; no test-based selection"}, indent=2), flush=True)
    if args.dry_run:
        return
    for seed in seeds:
        root = destination / f"seed_{seed}"
        foundation_dir = root / "foundation"
        checkpoint_path = foundation_dir / "foundation.pt"
        if not checkpoint_path.exists():
            if args.stage == "adapt":
                raise SystemExit(f"Missing foundation: {checkpoint_path}")
            foundation_dir.mkdir(parents=True, exist_ok=False)
            write_json(config, foundation_dir / "effective_config.json")
            train_foundation(problem, config, seed, device, foundation_dir, provenance, smoke=args.smoke)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Only locally generated trusted checkpoints are loaded (contain replay dataclasses).
        for key, value in provenance.items():
            if checkpoint["metadata"].get(key) != value:
                if args.reuse_foundation and key == "code_sha256":
                    continue
                raise SystemExit(f"Existing foundation provenance changed ({key}); use a NEW output_root.")
        if checkpoint["metadata"]["device"] != device:
            raise SystemExit("Restore with the same device as the foundation for exact matched branches.")
        if state_digest(checkpoint["state"]) != checkpoint["metadata"]["snapshot_sha256"]:
            raise SystemExit("Foundation content fingerprint mismatch.")
        if args.stage == "foundation":
            continue
        if not checkpoint["metadata"]["qualified"] and not args.smoke:
            print(f"seed={seed}: static foundation unqualified; no adaptation comparison for this seed.", flush=True)
            continue
        runs = [(fraction, None, f"demo_{round(fraction * 100):02d}", 0) for fraction in fractions]
        if args.schedule == "decay":
            runs = [(None, "decay", "schedule_decay", 0)]
        elif args.schedule == "risk_handover":
            if "risk_replay" not in config:
                raise SystemExit("The selected config does not define risk_replay.")
            runs = [(None, "risk_handover", "risk_handover", 0)]
        elif args.schedule == "safe_intervention":
            runs = [
                (
                    None,
                    "safe_intervention",
                    "intervention_only" if count == 0 else f"safe_replay_{count:02d}",
                    count,
                )
                for count in safe_samples
            ]
        for fraction, schedule, branch_name, safe_sample_count in runs:
            branch = root / branch_name
            result_path = branch / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (all(result.get(k) == v for k, v in provenance.items())
                        and result.get("fork_sha256") == checkpoint["metadata"]["snapshot_sha256"]
                        and result.get("fraction") == fraction
                        and result.get("replay_schedule") == (schedule or "fixed")
                        and result.get("safe_sample_count", 0) == safe_sample_count
                        and result.get("smoke") == args.smoke
                        and result.get("status") == "complete"):
                    print(f"Already complete: {branch}", flush=True)
                    continue
                raise SystemExit(f"Existing branch provenance differs: {branch}")
            if branch.exists():
                raise SystemExit(f"Incomplete branch preserved at {branch}. Use a NEW output_root for a clean rerun.")
            train_branch(problem, config, scenarios, seed, device, branch, checkpoint, fraction,
                         smoke=args.smoke, replay_schedule=schedule,
                         safe_sample_count=safe_sample_count)


if __name__ == "__main__":
    main()
