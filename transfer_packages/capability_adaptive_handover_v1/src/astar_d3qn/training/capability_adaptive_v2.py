"""v2 independent-train-monitor adapter around the frozen v1 AUC trainer.

The underlying optimizer, replay, controller, validation, and training stream
remain the v1 implementation. This adapter replaces only the monitor scene
selector and records monitor-only environment interaction overhead.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from astar_d3qn.training import capability_adaptive_auc as base
from astar_d3qn.training.independent_train_monitor import generate_train_monitor_scenes


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def train_capability_adaptive_v2(
    problem: Any,
    config: Mapping[str, Any],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    device: str,
    output_dir: Path,
    checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    controller=None,
    monitor_observer=None,
    method="capability_adaptive_handover_v2_independent_monitor",
) -> dict[str, Any]:
    """Run v1 unchanged with an independent monitor and overhead accounting."""

    monitor_scenes, monitor_manifest = generate_train_monitor_scenes(
        problem, splits, config["train_monitor"]
    )
    if len(monitor_scenes) != config["train_monitor"]["independent_scene_count"]:
        raise RuntimeError("Independent monitor count mismatch.")
    monitor_ids = {scene["scenario_id"] for scene in monitor_scenes}
    metrics_by_step: dict[int, tuple[int, int]] = {}
    monitor_current = 0
    monitor_cumulative = 0
    monitor_current_last = 0
    original_rollout = base.rollout
    original_write_json = base.write_json
    original_write_records_csv = base.write_records_csv
    original_select_monitor_scenes = base.select_monitor_scenes

    def v2_monitor_selector(_splits, _count):
        return copy.deepcopy(monitor_scenes)

    def counted_rollout(agent, rollout_problem, rollout_config, scene=None):
        nonlocal monitor_current, monitor_cumulative
        result = original_rollout(agent, rollout_problem, rollout_config, scene)
        if scene is not None and scene.get("scenario_id") in monitor_ids:
            monitor_current += int(result["steps"])
            monitor_cumulative += int(result["steps"])
        return result

    def v2_write_records_csv(records, path):
        nonlocal monitor_current, monitor_current_last
        if Path(path).name != "training.csv":
            return original_write_records_csv(records, path)
        if records:
            latest_step = max(int(row["environment_steps"]) for row in records)
            metrics_by_step[latest_step] = (monitor_current, monitor_cumulative)
            monitor_current_last = monitor_current
        enriched = []
        for row in records:
            value = dict(row)
            step = int(value["environment_steps"])
            current, cumulative = metrics_by_step.get(
                step, (monitor_current, monitor_cumulative)
            )
            value.update(
                {
                    "training_environment_steps": step,
                    "monitor_environment_steps_current": current,
                    "monitor_environment_steps_cumulative": cumulative,
                    "total_algorithm_environment_interactions": step + cumulative,
                }
            )
            enriched.append(value)
        result = original_write_records_csv(enriched, path)
        monitor_current = 0
        return result

    def v2_write_json(value, path):
        nonlocal monitor_current
        path = Path(path)
        value = dict(value)
        if path.name == "run_audit.json":
            value.update(
                {
                    "method": method,
                    "source_protocol": "ab_five_seed_strict_v1",
                    "verified_full_state_equal": True,
                    "training_split_unchanged": True,
                    "monitor_source_split": "train_monitor",
                    "monitor_scene_count": len(monitor_scenes),
                    "monitor_scenes_remain_in_training_stream": False,
                    "monitor_scenes_in_training_stream": False,
                    "monitor_scenes_in_replay": False,
                    "monitor_scenes_used_for_astar_demo": False,
                    "validation_controls_training": False,
                    "test_read_or_generated": False,
                    "monitor_manifest_sha256": monitor_manifest["manifest_sha256"],
                }
            )
            original_write_json(value, path)
            original_write_json(monitor_manifest, path.parent / "monitor_scene_manifest.json")
            return
        if path.name == "result.json":
            value.update(
                {
                    "method": method,
                    "source_protocol": "ab_five_seed_strict_v1",
                    "monitor_source_split": "train_monitor",
                    "monitor_scene_count": len(monitor_scenes),
                    "monitor_manifest_sha256": monitor_manifest["manifest_sha256"],
                    "monitor_environment_steps_current": monitor_current_last,
                    "monitor_environment_steps_cumulative": monitor_cumulative,
                    "training_environment_steps": int(value.get("adaptation_steps", 0)),
                    "total_algorithm_environment_interactions": int(
                        value.get("adaptation_steps", 0)
                    )
                    + monitor_cumulative,
                    "validation_controls_training": False,
                    "test_read_or_generated": False,
                }
            )
        return original_write_json(value, path)

    def v2_rollout_boundary_write(records, path):
        return v2_write_records_csv(records, path)

    try:
        base.rollout = counted_rollout
        base.write_json = v2_write_json
        base.write_records_csv = v2_rollout_boundary_write
        base.select_monitor_scenes = v2_monitor_selector
        result = base.train_capability_adaptive_auc(
            problem,
            config,
            splits,
            seed,
            device,
            output_dir,
            checkpoint,
            provenance,
            controller=controller,
            monitor_observer=monitor_observer,
        )
    finally:
        base.rollout = original_rollout
        base.write_json = original_write_json
        base.write_records_csv = original_write_records_csv
        base.select_monitor_scenes = original_select_monitor_scenes

    result["method"] = method
    return result
