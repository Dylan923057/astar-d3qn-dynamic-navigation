"""Prepare and run the preregistered clean Office comparison."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json


MAIN_CONFIG = "configs/dynamic_office_clean_v1.yaml"
TIME_DECAY_CONFIG = "configs/dynamic_office_clean_time_decay_v1.yaml"
CA_ORACLE_CONFIG = "configs/dynamic_office_clean_ca_oracle_v1.yaml"


@dataclass(frozen=True, slots=True)
class Method:
    key: str
    label: str
    config: str
    strategy: str


METHODS = (
    Method("uniform", "Uniform", MAIN_CONFIG, "uniform"),
    Method("prefill", "Prefill", MAIN_CONFIG, "prefill"),
    Method("persistent", "Persistent", MAIN_CONFIG, "persistent_demo"),
    Method("time_decay", "Time-decay", TIME_DECAY_CONFIG, "persistent_demo"),
    Method("ca_oracle", "CA-oracle", CA_ORACLE_CONFIG, "conflict_adaptive_demo"),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _complete(method: Method, seed: int, smoke: bool) -> bool:
    config = load_config(_resolve(method.config))
    suffix = "_smoke" if smoke else ""
    run_dir = _resolve(config["experiment"]["output_root"]) / (
        f"{config['experiment']['run_name_prefix']}_seed_{seed}_"
        f"{method.strategy}{suffix}"
    )
    required = (
        run_dir / "run_metadata.json",
        run_dir / "model_selected.pth",
        run_dir / "test_evaluation.csv",
        run_dir / "test_trajectories.json",
    )
    if not all(path.exists() for path in required):
        return False
    metadata = load_json(run_dir / "run_metadata.json")
    isolation = metadata.get("scenario_manifest", {}).get("split_isolation", {})
    manifest_path = _resolve(config["spatial_generalization"]["manifest"])
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    recorded_manifest_hash = metadata.get("scenario_manifest", {}).get("sha256")
    demo_dataset = metadata.get("demonstration_dataset")
    demo_matches = True
    if method.strategy != "uniform":
        demo_path = _resolve(config["demonstrations"]["file"])
        demo_matches = (
            isinstance(demo_dataset, dict)
            and demo_dataset.get("sha256")
            == hashlib.sha256(demo_path.read_bytes()).hexdigest()
        )
    return (
        int(metadata.get("training_seed", -1)) == seed
        and str(metadata.get("strategy")) == method.strategy
        and bool(metadata.get("smoke", False)) == smoke
        and bool(isolation.get("passed", False))
        and recorded_manifest_hash == manifest_hash
        and demo_matches
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_BY_KEY), default=tuple(METHOD_BY_KEY)
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prepare = [sys.executable, str(ROOT / "scripts" / "prepare_office_clean_v1.py")]
    if args.force_prepare:
        prepare.append("--force")
    _run(prepare, args.dry_run)
    if args.prepare_only:
        return

    main_config = load_config(_resolve(MAIN_CONFIG))
    registered = tuple(int(seed) for seed in main_config["training"]["seeds"])
    seeds = registered if args.seeds is None else tuple(args.seeds)
    unknown = sorted(set(seeds).difference(registered))
    if unknown:
        raise ValueError(f"Seeds {unknown} are outside registered seeds {registered}.")

    for key in args.methods:
        method = METHOD_BY_KEY[key]
        for seed in seeds:
            if not args.force and _complete(method, seed, args.smoke):
                print(f"[skip] {method.label} seed={seed}: complete", flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "train_dynamic_spatial_generalization.py"),
                "--config",
                method.config,
                "--strategy",
                method.strategy,
                "--seed",
                str(seed),
            ]
            if args.device:
                command.extend(("--device", args.device))
            if args.smoke:
                command.append("--smoke")
            _run(command, args.dry_run)
    print("[done] clean Office runs completed.", flush=True)


if __name__ == "__main__":
    main()
