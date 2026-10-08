from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: Iterable[float], window: int = 50) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return array
    window = max(1, int(window))
    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    result = np.empty(len(array), dtype=np.float64)
    for index in range(len(array)):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (
            index + 1 - start
        )
    return result


def _series(records: Sequence[dict], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key, 0.0)
        values.append(0.0 if value in (None, "") else float(value))
    return values


def plot_training_curves(
    records: Sequence[dict],
    save_path: str | Path,
    title: str,
    moving_window: int = 50,
) -> None:
    if not records:
        raise ValueError("Cannot plot an empty training record set.")

    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x = _series(records, "episode")
    has_collision_count = any("collision_count" in record for record in records)
    panels = [
        ("success", "Success rate", "#2a9d8f", (0.0, 1.0)),
        ("safe_success", "Collision-free success", "#238b45", (0.0, 1.0)),
        ("collision_success", "Success after collision", "#d1495b", (0.0, 1.0)),
        ("static_collision_count", "Static collision count", "#7b2cbf", (0.0, None)),
        ("dynamic_collision_count", "Dynamic collision count", "#f77f00", (0.0, None)),
        ("reward", "Episode reward", "#146c94", None),
        ("steps", "Steps per episode", "#e08e0b", (0.0, None)),
    ]
    if not has_collision_count:
        panels[3] = ("collision", "Collision episodes", "#d1495b", (0.0, None))
        panels[4] = ("collision", "Collision episodes", "#d1495b", (0.0, None))

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.2 * rows), dpi=180)
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#263238")
    for axis, (key, label, color, limits) in zip(axes, panels):
        values = _series(records, key)
        axis.plot(x, values, color=color, alpha=0.20, linewidth=0.8)
        axis.plot(
            x,
            moving_average(values, moving_window),
            color=color,
            linewidth=2.4,
            label=f"MA-{moving_window}",
        )
        axis.set_title(label, color="#263238")
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if limits is not None:
            lower, upper = limits
            axis.set_ylim(bottom=lower, top=upper)
        axis.legend(loc="best", frameon=False, fontsize=8)

    for axis in axes[len(panels) :]:
        axis.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(target, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_evaluation_curves(
    records: Sequence[dict],
    save_path: str | Path,
    title: str,
) -> None:
    """Plot frozen-greedy checkpoint metrics separately from training episodes."""

    if not records:
        raise ValueError("Cannot plot an empty evaluation record set.")
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x = _series(records, "episode")
    panels = [
        ("success_rate", "Greedy success rate", "#2a9d8f", (0.0, 1.0)),
        ("safe_success_rate", "Greedy collision-free success", "#238b45", (0.0, 1.0)),
        ("collision_success_rate", "Greedy success after collision", "#d1495b", (0.0, 1.0)),
        ("static_collision_rate", "Greedy static collision rate", "#7b2cbf", (0.0, 1.0)),
        ("dynamic_collision_rate", "Greedy dynamic collision rate", "#f77f00", (0.0, 1.0)),
        ("mean_steps", "Greedy mean steps", "#146c94", (0.0, None)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), dpi=180)
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#263238")
    for axis, (key, label, color, limits) in zip(axes.flat, panels):
        values = _series(records, key)
        axis.plot(x, values, color=color, marker="o", linewidth=2.0, markersize=3.5)
        axis.set_title(label, color="#263238")
        axis.set_xlabel("Training episode")
        axis.grid(True, alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        lower, upper = limits
        axis.set_ylim(bottom=lower, top=upper)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(target, facecolor="white", bbox_inches="tight")
    plt.close(fig)
