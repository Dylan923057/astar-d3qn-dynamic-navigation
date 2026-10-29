from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


def checkpoint_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    if not details:
        raise ValueError("Checkpoint evaluation details cannot be empty.")

    successful = [row for row in details if float(row["success"]) > 0.0]
    return {
        "success_rate": fmean(float(row["success"]) for row in details),
        "mean_steps": fmean(float(row["steps"]) for row in details),
        "mean_success_steps": (
            fmean(float(row["steps"]) for row in successful)
            if successful
            else None
        ),
        "collision_rate": fmean(float(row["collision"]) for row in details),
        "mean_collision_count": fmean(
            float(row["collision_count"]) for row in details
        ),
        "static_collision_rate": fmean(
            float(row.get("static_collision", row["collision"])) for row in details
        ),
        "mean_static_collision_count": fmean(
            float(row.get("static_collision_count", row["collision_count"]))
            for row in details
        ),
        "dynamic_collision_rate": fmean(
            float(row.get("dynamic_collision", 0.0)) for row in details
        ),
        "mean_dynamic_collision_count": fmean(
            float(row.get("dynamic_collision_count", 0.0)) for row in details
        ),
        "safe_success_rate": fmean(
            float(
                row.get(
                    "safe_success",
                    float(row["success"]) > 0.0 and float(row["collision"]) == 0.0,
                )
            )
            for row in details
        ),
        "collision_success_rate": fmean(
            float(
                row.get(
                    "collision_success",
                    float(row["success"]) > 0.0 and float(row["collision"]) > 0.0,
                )
            )
            for row in details
        ),
        "static_collision_success_rate": fmean(
            float(
                row.get(
                    "static_collision_success",
                    float(row["success"]) > 0.0
                    and float(row.get("static_collision", 0.0)) > 0.0,
                )
            )
            for row in details
        ),
        "dynamic_collision_success_rate": fmean(
            float(
                row.get(
                    "dynamic_collision_success",
                    float(row["success"]) > 0.0
                    and float(row.get("dynamic_collision", 0.0)) > 0.0,
                )
            )
            for row in details
        ),
        "dynamic_collision_free_success_rate": fmean(
            float(
                row.get(
                    "dynamic_collision_free_success",
                    float(row["success"]) > 0.0,
                )
            )
            for row in details
        ),
        "mean_wait_steps": fmean(float(row["wait_steps"]) for row in details),
        "mean_movement_steps": fmean(
            float(
                row.get(
                    "movement_steps",
                    float(row["steps"]) - float(row.get("wait_steps", 0.0)),
                )
            )
            for row in details
        ),
        "mean_turn_count": fmean(float(row.get("turn_count", 0.0)) for row in details),
        "mean_turn_rate": fmean(float(row.get("turn_rate", 0.0)) for row in details),
        "mean_revisit_count": fmean(
            float(row["revisit_count"]) for row in details
        ),
        "mean_reward": fmean(float(row["reward"]) for row in details),
    }


def checkpoint_rank(metrics: Mapping[str, float | None]) -> tuple[float, ...]:
    success_steps = metrics["mean_success_steps"]
    success_step_rank = (
        -float(success_steps) if success_steps is not None else float("-inf")
    )
    return (
        float(metrics["success_rate"]),
        success_step_rank,
        -float(metrics["mean_collision_count"]),
        -float(metrics["mean_wait_steps"]),
        -float(metrics["mean_revisit_count"]),
        float(metrics["mean_reward"]),
    )


def validation_checkpoint_score(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Rank validation checkpoints without consulting the held-out test split."""

    return (
        float(summary["safe_success_rate"]),
        float(summary["success_rate"]),
        -float(summary["mean_dynamic_collision_count"]),
        -float(summary["mean_steps"]),
    )
