"""Run the complete controlled Office dynamic-obstacle study.

The command is deliberately resumable: completed formal runs are skipped unless
``--force`` is supplied.  Ablations with changed controller settings use isolated
configs and output roots, while all methods share the frozen map and scenarios.
"""

from __future__ import annotations

import argparse
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


MAIN_CONFIG = "configs/dynamic_office_controlled_behavior_v1.yaml"
CURRENT_CONFIG = "configs/dynamic_office_controlled_behavior_current_v1.yaml"
SCHEDULED_DECAY_CONFIG = "configs/dynamic_office_scheduled_decay_v2.yaml"
ADAPTIVE_DECAY_CONFIG = "configs/dynamic_office_conflict_adaptive_decay_v2.yaml"
LOCAL_COUNTEREXAMPLE_CONFIG = "configs/dynamic_office_local_counterexample_v1.yaml"
PREDICTIVE_MARGIN_CONFIG = "configs/dynamic_office_predictive_margin_v1.yaml"


@dataclass(frozen=True, slots=True)
class MethodSpec:
    key: str
    label: str
    config_path: str
    strategy: str


METHODS = (
    MethodSpec("uniform", "Uniform", MAIN_CONFIG, "uniform"),
    MethodSpec("prefill", "Prefill", MAIN_CONFIG, "prefill"),
    MethodSpec("persistent", "Persistent", MAIN_CONFIG, "persistent_demo"),
    MethodSpec(
        "scheduled_decay",
        "Time-decay",
        SCHEDULED_DECAY_CONFIG,
        "persistent_demo",
    ),
    MethodSpec(
        "ca_current",
        "CA-current",
        CURRENT_CONFIG,
        "conflict_adaptive_demo",
    ),
    MethodSpec(
        "ca_predictive",
        "CA-predictive",
        MAIN_CONFIG,
        "conflict_adaptive_demo",
    ),
    MethodSpec(
        "local_conflict",
        "Local-conflict",
        MAIN_CONFIG,
        "local_conflict_demo",
    ),
    MethodSpec(
        "ca_adaptive_decay",
        "CA-decay-v2",
        ADAPTIVE_DECAY_CONFIG,
        "conflict_adaptive_demo",
    ),
    MethodSpec(
        "local_counterexample",
        "Local-CE",
        LOCAL_COUNTEREXAMPLE_CONFIG,
        "local_counterexample_demo",
    ),
    MethodSpec(
        "predictive_margin",
        "CA-margin",
        PREDICTIVE_MARGIN_CONFIG,
        "predictive_margin_demo",
    ),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _run_dir(method: MethodSpec, seed: int, smoke: bool) -> Path:
    config = load_config(_resolve(method.config_path))
    suffix = "_smoke" if smoke else ""
    return _resolve(config["experiment"]["output_root"]) / (
        f"{config['experiment']['run_name_prefix']}_seed_{seed}_"
        f"{method.strategy}{suffix}"
    )


def _is_complete(method: MethodSpec, seed: int, smoke: bool) -> bool:
    run_dir = _run_dir(method, seed, smoke)
    config = load_config(_resolve(method.config_path))
    required = (
        run_dir / "run_metadata.json",
        run_dir / "model_selected.pth",
        run_dir / "test_evaluation.csv",
        run_dir / "test_trajectories.json",
    )
    if not all(path.exists() for path in required):
        return False
    metadata = load_json(run_dir / "run_metadata.json")
    complete = (
        bool(metadata.get("smoke", False)) == smoke
        and str(metadata.get("strategy")) == method.strategy
        and int(metadata.get("training_seed", -1)) == seed
    )
    expected_sources = config["spatial_generalization"].get(
        "scenario_split_sources"
    )
    if expected_sources is not None:
        recorded_sources = metadata.get("scenario_manifest", {}).get(
            "schedule_sources"
        )
        complete = complete and recorded_sources == expected_sources
    return complete


def _call(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_BY_KEY),
        default=tuple(METHOD_BY_KEY),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun entries even when their completed outputs already exist.",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main_config = load_config(_resolve(MAIN_CONFIG))
    registered_seeds = tuple(int(value) for value in main_config["training"]["seeds"])
    seeds = tuple(args.seeds) if args.seeds is not None else registered_seeds
    unknown_seeds = sorted(set(seeds).difference(registered_seeds))
    if unknown_seeds:
        raise ValueError(
            f"Seeds {unknown_seeds} are outside the preregistered set "
            f"{registered_seeds}."
        )

    for method_key in args.methods:
        method = METHOD_BY_KEY[method_key]
        for seed in seeds:
            if not args.force and _is_complete(method, seed, args.smoke):
                print(
                    f"[skip] {method.label} seed={seed}: completed output exists",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "train_dynamic_spatial_generalization.py"),
                "--config",
                str(ROOT / method.config_path),
                "--strategy",
                method.strategy,
                "--seed",
                str(seed),
            ]
            if args.device:
                command.extend(("--device", args.device))
            if args.smoke:
                command.append("--smoke")
            _call(command, dry_run=args.dry_run)

    full_matrix = (
        tuple(args.methods) == tuple(METHOD_BY_KEY)
        and seeds == registered_seeds
        and not args.smoke
    )
    if args.skip_analysis or not full_matrix:
        if not args.skip_analysis:
            print(
                "Aggregate analysis skipped because this is not the complete "
                "formal method-by-seed matrix.",
                flush=True,
            )
        return

    summary_command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_office_controlled_behavior_study.py"),
        "--output-dir",
        "outputs/office_controlled_behavior_study_v4",
    ]
    diagnostic_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_office_controlled_blockage_diagnostic.py"),
        "--output-dir",
        "outputs/office_controlled_blockage_diagnostic_v4",
    ]
    checkpoint_command = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_office_checkpoint_conflict_curves.py"),
        "--output-dir",
        "outputs/office_checkpoint_conflict_analysis_v4",
    ]
    visualization_command = [
        sys.executable,
        str(
            ROOT
            / "scripts"
            / "render_office_controlled_behavior_visualizations.py"
        ),
        "--output-dir",
        "outputs/office_controlled_behavior_visualizations_v4",
        "--diagnostic-dir",
        "outputs/office_controlled_blockage_diagnostic_v4",
    ]
    if args.device:
        diagnostic_command.extend(("--device", args.device))
        checkpoint_command.extend(("--device", args.device))
        visualization_command.extend(("--device", args.device))
    _call(summary_command, dry_run=args.dry_run)
    _call(diagnostic_command, dry_run=args.dry_run)
    _call(checkpoint_command, dry_run=args.dry_run)
    _call(visualization_command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
