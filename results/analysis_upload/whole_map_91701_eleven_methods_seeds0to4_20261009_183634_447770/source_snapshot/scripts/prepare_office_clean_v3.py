"""Freeze, validate, and render the clean_v3 paired benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from astar_d3qn.envs.spatial_scenarios import validate_spatial_scenario_manifest
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.io import load_json, write_json, write_records_csv
import generate_office_astar_conflict_pairs_v3 as v3
import generate_office_behavior_scenarios_v9 as v9


def _config_hash(config) -> str:
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=v3.DEFAULT_CONFIG)
    parser.add_argument(
        "--audit-output", default="outputs/office_clean_v3_dataset_design"
    )
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args()
    config = load_config(v3._resolve(args.config))
    problem = v9._load_problem(config)
    # This validates the frozen file hash, metadata signature, path continuity,
    # and exact demonstration count before scenarios are accepted.
    v9._reference_demonstrations(problem, config)
    spatial = config["spatial_generalization"]
    manifest_path = v3._resolve(spatial["manifest"])
    audit_dir = v3._resolve(args.audit_output)
    audit_path = audit_dir / "protocol_audit.json"
    config_hash = _config_hash(config)
    if manifest_path.exists() and not args.force_manifest:
        if audit_path.exists():
            previous = load_json(audit_path)
            if previous.get("config_sha256") not in {None, config_hash}:
                raise ValueError(
                    "clean_v3 config changed after the paired manifest was frozen; "
                    "rerun with --force-manifest after review."
                )
        manifest = load_json(manifest_path)
        validate_spatial_scenario_manifest(problem, manifest)
        audit_rows = v3.validate_manifest(problem, manifest, config)
        print(f"[prepare] validated paired manifest: {manifest_path}", flush=True)
    else:
        manifest, audit_rows = v3.build_manifest(problem, config)
        write_json(manifest, manifest_path)
        v9.render_route_pools(
            problem, manifest, v3._resolve(spatial["route_pool_preview"])
        )
        for split, key in (
            ("train", "training_preview"),
            ("validation", "validation_preview"),
            ("test", "test_preview"),
        ):
            v9.render_gallery(problem, manifest, split, v3._resolve(spatial[key]))
    write_records_csv(audit_rows, audit_dir / "phase_pair_audit.csv")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(
        {
            "status": "passed",
            "config_sha256": config_hash,
            "manifest": str(manifest_path),
            "manifest_sha256": digest,
            "pair_counts": manifest["generation"]["pair_counts"],
            "complete_scenarios_cross_split_disjoint": True,
            "shared_route_templates_across_splits": True,
            "test_pair_scoring": (
                "both safe; control follows an optimal no-wait response; conflict "
                "uses a safe wait response"
            ),
            "phase_pair_audit": str(audit_dir / "phase_pair_audit.csv"),
        },
        audit_path,
    )
    print(
        f"[prepare] clean_v3 protocol passed; manifest sha256={digest[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
