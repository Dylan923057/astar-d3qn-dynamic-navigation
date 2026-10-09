"""Plot the completed conflict-adaptive v1 experiment in one compact figure."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "conflict_adaptive_v1_analysis"
MAPS = ("office", "parcel", "warehouse")
METHODS = (
    ("Uniform", "uniform"),
    ("Prefill", "prefill"),
    ("Fixed 25%", "persistent_demo"),
    ("Adaptive+prediction", "ca_predictive"),
    ("Adaptive current-only", "ca_current"),
)
COLORS = ("#64748b", "#0ea5e9", "#16a34a", "#f97316", "#a855f7")


def _run_files(map_key: str, method: str) -> list[Path]:
    if method in {"uniform", "prefill", "persistent_demo"}:
        root = ROOT / "outputs" / f"dynamic_spatial_generalization_{map_key}_balanced_v4"
        strategy = method
    else:
        mode = "adaptive" if method == "ca_predictive" else "current"
        root = ROOT / "outputs" / f"dynamic_spatial_generalization_{map_key}_conflict_{mode}_v1"
        strategy = "conflict_adaptive_demo"
    return sorted(
        path
        for path in root.glob(f"*_{strategy}/run_metadata.json")
        if "_smoke" not in str(path)
    )


def _safe_success() -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(METHODS), len(MAPS)))
    deviations = np.zeros_like(means)
    for method_index, (_, method) in enumerate(METHODS):
        for map_index, map_key in enumerate(MAPS):
            values = [
                float(json.loads(path.read_text(encoding="utf-8"))["test_summary"]["safe_success_rate"])
                for path in _run_files(map_key, method)
            ]
            means[method_index, map_index] = statistics.fmean(values)
            deviations[method_index, map_index] = statistics.stdev(values)
    return means, deviations


def _dangerous_astar_rates() -> np.ndarray:
    path = ROOT / "outputs" / "v4_critical_blockage_diagnostic" / "critical_blockage_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        (row["map_key"], row["strategy"]): float(row["probe_selected_demo_action_mean"])
        for row in rows
    }
    return np.asarray(
        [[lookup[(map_key, method)] for map_key in MAPS] for _, method in METHODS]
    )


def _fraction_curve(map_key: str, method: str, window: int = 100) -> np.ndarray:
    curves = []
    for metadata_path in _run_files(map_key, method):
        with (metadata_path.parent / "training.csv").open(encoding="utf-8-sig", newline="") as handle:
            values = [float(row["demo_fraction_mean"]) for row in csv.DictReader(handle)]
        curves.append(
            [statistics.fmean(values[index:index + window]) for index in range(0, len(values), window)]
        )
    return np.mean(np.asarray(curves), axis=0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    safe_mean, safe_sd = _safe_success()
    danger = _dangerous_astar_rates()
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(len(MAPS))
    width = 0.15

    for index, ((label, _), color) in enumerate(zip(METHODS, COLORS)):
        offset = (index - 2) * width
        axes[0].bar(
            x + offset,
            100 * safe_mean[index],
            width,
            yerr=100 * safe_sd[index],
            capsize=2,
            color=color,
            label=label,
        )
        axes[1].bar(x + offset, 100 * danger[index], width, color=color, label=label)

    axes[0].set_title("Formal v4 test: safe success")
    axes[0].set_ylabel("Safe success (%)")
    axes[0].set_ylim(0, 110)
    axes[0].set_xticks(x, [name.title() for name in MAPS])
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_title("Critical probe: chooses blocked A* action")
    axes[1].set_ylabel("Dangerous A* choice (%)  [lower is better]")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(x, [name.title() for name in MAPS])
    axes[1].grid(axis="y", alpha=0.25)

    episodes = np.arange(100, 1501, 100)
    for map_key, color in zip(MAPS, ("#2563eb", "#dc2626", "#16a34a")):
        axes[2].plot(
            episodes,
            100 * _fraction_curve(map_key, "ca_predictive"),
            marker="o",
            markersize=3,
            color=color,
            label=map_key.title(),
        )
    axes[2].axhline(25, color="#111827", linestyle="--", linewidth=1, label="Fixed 25%")
    axes[2].set_title("Adaptive demo fraction (100-episode mean)")
    axes[2].set_xlabel("Training episode")
    axes[2].set_ylabel("Demo fraction (%)")
    axes[2].set_ylim(23.5, 25.1)
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8, loc="lower right")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle("Conflict-adaptive demonstration replay v1: completed 5-seed results", fontsize=15)
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    figure.savefig(OUTPUT / "conflict_adaptive_v1_summary.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
