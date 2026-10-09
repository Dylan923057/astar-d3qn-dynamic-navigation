"""Aggregate action and event counts from the formal Office v4 test logs."""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from astar_d3qn.utils.config import load_config
from run_office_controlled_behavior_study import MAIN_CONFIG, METHODS, _run_dir


OUTPUT_DIR = ROOT / "outputs" / "office_strategy_comprehensive_v4"
BEHAVIORS = ("wait", "avoidance", "reroute")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else math.nan


def _sum(rows: Iterable[dict[str, str]], key: str) -> int:
    return int(sum(_number(row, key) for row in rows))


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = [_number(row, key) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return fmean(finite) if finite else math.nan


def _aggregate(
    method_key: str,
    method_label: str,
    behavior: str,
    rows: list[dict[str, str]],
) -> dict[str, int | float | str]:
    successful = [row for row in rows if _number(row, "success") > 0]
    waiting = [row for row in rows if _number(row, "wait_steps") > 0]
    return {
        "method": method_key,
        "method_label": method_label,
        "required_behavior": behavior,
        "episode_count": len(rows),
        "goal_episode_count": _sum(rows, "success"),
        "safe_goal_episode_count": _sum(rows, "safe_success"),
        "timeout_episode_count": sum(
            row.get("termination_reason", "") != "goal" for row in rows
        ),
        "collision_episode_count": _sum(rows, "collision"),
        "collision_event_count": _sum(rows, "collision_count"),
        "static_collision_episode_count": _sum(rows, "static_collision"),
        "static_collision_event_count": _sum(rows, "static_collision_count"),
        "dynamic_collision_episode_count": _sum(rows, "dynamic_collision"),
        "dynamic_collision_event_count": _sum(rows, "dynamic_collision_count"),
        "wait_episode_count": len(waiting),
        "wait_action_count": _sum(rows, "wait_steps"),
        "wait_event_count": _sum(rows, "wait_event_count"),
        "max_wait_streak_observed": int(
            max((_number(row, "max_wait_streak") for row in rows), default=0)
        ),
        "step_count": _sum(rows, "steps"),
        "mean_steps_per_episode": _mean(rows, "steps"),
        "mean_steps_per_goal": _mean(successful, "steps"),
        "mean_wait_actions_per_episode": _mean(rows, "wait_steps"),
        "mean_wait_actions_when_used": _mean(waiting, "wait_steps"),
        "movement_action_count": _sum(rows, "movement_steps"),
        "turn_count": _sum(rows, "turn_count"),
        "revisit_count": _sum(rows, "revisit_count"),
        "mean_path_efficiency": _mean(rows, "path_efficiency"),
    }


def _write(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seeds = tuple(
        int(seed)
        for seed in load_config(ROOT / MAIN_CONFIG)["training"]["seeds"]
    )
    overall: list[dict[str, int | float | str]] = []
    by_behavior: list[dict[str, int | float | str]] = []
    for method in METHODS:
        rows: list[dict[str, str]] = []
        for seed in seeds:
            path = _run_dir(method, seed, False) / "test_evaluation.csv"
            seed_rows = _read_csv(path)
            if len(seed_rows) != 50:
                raise ValueError(f"Expected 50 test rows in {path}, got {len(seed_rows)}.")
            rows.extend(seed_rows)
        overall.append(_aggregate(method.key, method.label, "all", rows))
        for behavior in BEHAVIORS:
            selected = [
                row for row in rows if row["required_behavior"] == behavior
            ]
            by_behavior.append(
                _aggregate(method.key, method.label, behavior, selected)
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write(OUTPUT_DIR / "formal_test_behavior_counts.csv", overall)
    _write(OUTPUT_DIR / "formal_test_behavior_counts_by_scene.csv", by_behavior)
    print(
        "Wrote formal test action/event counts for "
        f"{len(METHODS)} methods x {len(seeds)} seeds."
    )


if __name__ == "__main__":
    main()
