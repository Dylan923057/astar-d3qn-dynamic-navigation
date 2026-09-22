"""Freeze A* demonstrations, build clean scenarios, and write design audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.envs.spatial_scenarios import validate_spatial_scenario_manifest
from astar_d3qn.training.demo_collector import (
    collect_astar_demonstrations,
    demonstration_file_sha256,
    demonstration_signature,
    save_demonstrations,
)
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json
from generate_office_behavior_scenarios_v9 import (
    _load_problem,
    _reference_demonstrations,
    _resolve,
    _scenario_design,
    _validate_dataset_semantics,
    _validate_exact_scenario_isolation,
    _validate_route_geometry_isolation,
    _write_audit,
    build_manifest,
    render_dataset_summary,
    render_gallery,
    render_route_pools,
)


DEFAULT_CONFIG = "configs/dynamic_office_clean_v1.yaml"
DEFAULT_AUDIT = "outputs/office_clean_v1_dataset_design"


def _config_sha256(config) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _difficulty_matching_audit(manifest, maximum_gap: float) -> list[dict]:
    rows = []
    for behavior in ("normal", "wait", "avoidance", "reroute"):
        means = {}
        for split in ("train", "validation", "test"):
            values = [
                float(source["difficulty_score"])
                for source in manifest["scenarios"][split]
                if source["required_behavior"] == behavior
            ]
            means[split] = sum(values) / len(values)
            rows.append(
                {
                    "behavior": behavior,
                    "split": split,
                    "count": len(values),
                    "minimum_difficulty_score": min(values),
                    "mean_difficulty_score": means[split],
                    "maximum_difficulty_score": max(values),
                }
            )
        gap = max(means.values()) - min(means.values())
        if gap > maximum_gap:
            raise ValueError(
                f"Cross-split mean difficulty gap for {behavior} is {gap:.3f}, "
                f"above the registered limit {maximum_gap:.3f}."
            )
        for row in rows:
            if row["behavior"] == behavior:
                row["cross_split_mean_gap"] = gap
    return rows


def _collect_frozen_demonstrations(problem, config, force: bool) -> Path:
    demo = config["demonstrations"]
    environment = config["environment"]
    output = _resolve(str(demo["file"]))
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists() and not force:
        print(f"[prepare] keeping frozen demonstrations: {output}", flush=True)
        return output

    reward = RewardConfig(
        **{key: float(value) for key, value in config["reward"].items()}
    )
    started = perf_counter()
    transitions = collect_astar_demonstrations(
        [problem],
        episodes=int(demo["episodes"]),
        seed=int(demo["seed"]),
        max_steps=int(environment["max_steps"]),
        reward_config=reward,
        window_size=int(environment["window_size"]),
        spatial_channels=int(environment["spatial_channels"]),
        mask_static_invalid_actions=bool(
            environment.get("mask_static_invalid_actions", False)
        ),
    )
    save_demonstrations(transitions, output)
    signature = demonstration_signature(
        [problem],
        episodes=int(demo["episodes"]),
        seed=int(demo["seed"]),
        max_steps=int(environment["max_steps"]),
        reward_config=reward,
        window_size=int(environment["window_size"]),
        spatial_channels=int(environment["spatial_channels"]),
        observation_mode=str(environment["observation"]),
        environment_id=str(demo["environment_id"]),
    )
    write_json(
        {
            "format_version": 3,
            **signature,
            "transitions": len(transitions),
            "dataset": output.name,
            "dataset_sha256": demonstration_file_sha256(output),
            "collection_seconds": perf_counter() - started,
        },
        metadata_path,
    )
    print(
        f"[prepare] froze {len(transitions)} A* transitions: {output}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-manifest",
        action="store_true",
        help="Regenerate scenarios while retaining the frozen demonstration file.",
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    config_sha256 = _config_sha256(config)
    problem = _load_problem(config)
    demo_path = _collect_frozen_demonstrations(problem, config, args.force)
    # This call checks the metadata signature, file hash, transition continuity,
    # and exact number of paths even when a frozen manifest is reused.
    _reference_demonstrations(problem, config)
    spatial = config["spatial_generalization"]
    manifest_path = _resolve(str(spatial["manifest"]))
    audit_output = _resolve(args.audit_output)
    protocol_audit_path = audit_output / "protocol_audit.json"
    if manifest_path.exists() and not (args.force or args.force_manifest):
        if protocol_audit_path.exists():
            previous_audit = load_json(protocol_audit_path)
            previous_config_hash = previous_audit.get("config_sha256")
            if (
                previous_config_hash is not None
                and previous_config_hash != config_sha256
            ):
                raise ValueError(
                    "The clean-study config changed after the manifest was "
                    "frozen; rerun with --force-manifest after review."
                )
        manifest = load_json(manifest_path)
        validate_spatial_scenario_manifest(problem, manifest)
        if bool(spatial.get("shared_route_templates_across_splits", False)):
            _validate_exact_scenario_isolation(
                manifest["route_pools"], manifest["scenarios"]
            )
        else:
            _validate_route_geometry_isolation(manifest["route_pools"])
        _validate_dataset_semantics(manifest, _scenario_design(config))
        recorded_demo = manifest["generation"].get("reference_demo_dataset", {})
        if recorded_demo.get("sha256") != demonstration_file_sha256(demo_path):
            raise ValueError(
                "Frozen scenario manifest was generated from a different A* "
                "demonstration dataset; rerun with --force-prepare."
            )
        print(f"[prepare] validated frozen scenario manifest: {manifest_path}")
    else:
        manifest = build_manifest(problem, config)
        write_json(manifest, manifest_path)
        render_route_pools(
            problem, manifest, _resolve(spatial["route_pool_preview"])
        )
        render_dataset_summary(
            manifest, _resolve(spatial["dataset_summary_preview"])
        )
        for split, key in (
            ("train", "training_preview"),
            ("validation", "validation_preview"),
            ("test", "test_preview"),
        ):
            render_gallery(problem, manifest, split, _resolve(spatial[key]))
    summaries = _write_audit(manifest, audit_output)
    difficulty_matching = _difficulty_matching_audit(
        manifest,
        float(
            spatial["scenario_design"].get(
                "maximum_cross_split_mean_difficulty_gap", 6.0
            )
        ),
    )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(
        {
            "status": "passed",
            "config_sha256": config_sha256,
            "static_map_scope": "same_registered_office_topology",
            "dynamic_split_scope": (
                "shared_route_templates_with_held_out_complete_scenarios"
                if bool(spatial.get("shared_route_templates_across_splits", False))
                else "held_out_exact_route_geometries_and_phases"
            ),
            "demonstration_path": str(demo_path),
            "demonstration_sha256": demonstration_file_sha256(demo_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "split_summaries": summaries,
            "difficulty_matching": difficulty_matching,
        },
        protocol_audit_path,
    )
    print(
        f"[prepare] clean protocol passed; manifest sha256={manifest_hash[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
