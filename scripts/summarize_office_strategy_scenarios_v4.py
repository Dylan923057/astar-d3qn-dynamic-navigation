"""Build one dashboard and a Chinese index for the complete Office v4 study."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_office_controlled_behavior_study import MAIN_CONFIG, METHODS, _run_dir
from astar_d3qn.utils.config import load_config


BEHAVIOR_DIR = ROOT / "outputs" / "office_controlled_behavior_study_v4"
BLOCKAGE_DIR = ROOT / "outputs" / "office_controlled_blockage_diagnostic_v4"
CHECKPOINT_DIR = ROOT / "outputs" / "office_checkpoint_conflict_analysis_v4"
VISUAL_DIR = ROOT / "outputs" / "office_controlled_behavior_visualizations_v4"
OUTPUT_DIR = ROOT / "outputs" / "office_strategy_comprehensive_v4"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _auc(rows: list[dict[str, str]], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row["nominal_environment_steps"]))
    x = np.asarray(
        [float(row["nominal_environment_steps"]) for row in ordered], dtype=float
    )
    y = np.asarray([float(row[f"{metric}_mean"]) for row in ordered], dtype=float)
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def _collect() -> list[dict[str, float | str]]:
    behavior = {
        (row["method"], row["required_behavior"]): row
        for row in _read_csv(BEHAVIOR_DIR / "test_summary.csv")
    }
    blockage = {
        row["method"]: row
        for row in _read_csv(BLOCKAGE_DIR / "diagnostic_summary.csv")
    }
    checkpoint = defaultdict(list)
    for row in _read_csv(CHECKPOINT_DIR / "checkpoint_probe_summary.csv"):
        checkpoint[row["method"]].append(row)
    seeds = tuple(
        int(seed)
        for seed in load_config(ROOT / MAIN_CONFIG)["training"]["seeds"]
    )
    records: list[dict[str, float | str]] = []
    for method in METHODS:
        static_values = []
        for seed in seeds:
            metadata_path = _run_dir(method, seed, False) / "run_metadata.json"
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            static_values.append(
                float(metadata["static_retention_summary"]["safe_success_rate"])
            )
        all_row = behavior[(method.key, "all")]
        wait_row = behavior[(method.key, "wait")]
        avoidance_row = behavior[(method.key, "avoidance")]
        reroute_row = behavior[(method.key, "reroute")]
        blockage_row = blockage[method.key]
        checkpoint_rows = checkpoint[method.key]
        records.append(
            {
                "method": method.key,
                "method_label": method.label,
                "static_safe": fmean(static_values),
                "overall_safe": float(all_row["safe_success_mean"]),
                "wait_safe": float(wait_row["safe_success_mean"]),
                "avoidance_safe": float(avoidance_row["safe_success_mean"]),
                "reroute_safe": float(reroute_row["safe_success_mean"]),
                "overall_behavior": float(all_row["behavior_match_mean"]),
                "wait_behavior": float(wait_row["behavior_match_mean"]),
                "avoidance_behavior": float(
                    avoidance_row["behavior_match_mean"]
                ),
                "reroute_behavior": float(reroute_row["behavior_match_mean"]),
                "exact_conflict_safe": float(
                    blockage_row["probe_selected_safe_action_mean"]
                ),
                "blockage_safe": float(
                    blockage_row["blocked_safe_success_mean"]
                ),
                "target_avoidance": 1.0
                - float(blockage_row["blocked_target_collision_mean"]),
                "target_collision": float(
                    blockage_row["blocked_target_collision_mean"]
                ),
                "validation_auc": _auc(
                    checkpoint_rows, "validation_safe_success"
                ),
                "conflict_auc": _auc(
                    checkpoint_rows, "conflict_safe_action_rate"
                ),
            }
        )
    return records


def _write_matrix(records: list[dict[str, float | str]]) -> None:
    fields = tuple(records[0])
    with (OUTPUT_DIR / "strategy_scenario_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot(records: list[dict[str, float | str]]) -> None:
    metrics = (
        ("static_safe", "Static\nsafe"),
        ("overall_safe", "All dynamic\nsafe"),
        ("wait_safe", "Wait scene\nsafe"),
        ("avoidance_safe", "Avoidance\nsafe"),
        ("reroute_safe", "Reroute\nsafe"),
        ("wait_behavior", "Wait behavior\nmatch"),
        ("avoidance_behavior", "Avoid behavior\nmatch"),
        ("reroute_behavior", "Reroute behavior\nmatch"),
        ("exact_conflict_safe", "Exact conflict\nsafe action"),
        ("blockage_safe", "Blocked rollout\nsafe"),
        ("target_avoidance", "Avoid target\ncollision"),
        ("validation_auc", "Validation\nsafe AUC"),
        ("conflict_auc", "Conflict-safe\nAUC"),
    )
    values = np.asarray(
        [[float(record[key]) for key, _ in metrics] for record in records]
    )
    figure, axis = plt.subplots(figsize=(19, 7.5), dpi=180)
    image = axis.imshow(values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(metrics)), [label for _, label in metrics])
    axis.set_yticks(
        range(len(records)), [str(record["method_label"]) for record in records]
    )
    axis.tick_params(axis="x", labelrotation=28)
    best = values.max(axis=0)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{100 * value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight=(
                    "bold" if np.isclose(value, best[column_index]) else "normal"
                ),
                color="white" if value < 0.28 or value > 0.82 else "black",
            )
    for boundary in (4.5, 7.5, 10.5):
        axis.axvline(boundary, color="white", linewidth=3)
    axis.set_title(
        "Office strategies across static, dynamic-behavior, blockage, and learning tests\n"
        "Cell values are percentages; bold marks the best result in each column"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    colorbar.set_label("Score")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "strategy_scenario_dashboard.png",
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)


def _percent(value: float | str) -> str:
    return f"{100 * float(value):.1f}%"


def _write_report(records: list[dict[str, float | str]]) -> None:
    lines = [
        "# Office 十种策略跨场景结果总览（v4）",
        "",
        "![十种策略跨场景热力图](strategy_scenario_dashboard.png)",
        "",
        "所有数字均为 5 个训练种子的均值。常规动态测试每个种子包含 50 个场景：20 个等待、15 个局部避障、15 个全局换路；每个场景有 5 个经过验证的 A* 路径交互障碍。堵路诊断每个种子包含 6 个单障碍场景，并配有同起点无障碍对照。",
        "",
        "| 策略 | 静态安全 | 动态安全 | 总行为匹配 | 等待行为 | 局部避障行为 | 换路行为 | 堵路点安全动作 | 堵路安全成功 | 目标碰撞 | 验证 AUC | 冲突 AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record['method_label']} | {_percent(record['static_safe'])} | "
            f"{_percent(record['overall_safe'])} | "
            f"{_percent(record['overall_behavior'])} | "
            f"{_percent(record['wait_behavior'])} | "
            f"{_percent(record['avoidance_behavior'])} | "
            f"{_percent(record['reroute_behavior'])} | "
            f"{_percent(record['exact_conflict_safe'])} | "
            f"{_percent(record['blockage_safe'])} | "
            f"{_percent(record['target_collision'])} | "
            f"{float(record['validation_auc']):.3f} | "
            f"{float(record['conflict_auc']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 地图与代表性动图",
            "",
            "这些 GIF 使用相同的 Office 静态地图和训练 seed 0 的冻结策略，便于逐帧比较动作；表格才是 5 个种子的统计结论。",
            "",
            "- [Office 静态地图](../office_controlled_behavior_visualizations_v4/office_static_map.png)",
            "- [训练场景 40：动态障碍运动](../office_controlled_behavior_visualizations_v4/training_scenario_40_obstacles.gif)",
            "- [验证场景 0：地图布局](../office_controlled_behavior_visualizations_v4/validation_scenario_0_layout.png)",
            "- [等待场景 8：十种策略动图](../office_controlled_behavior_visualizations_v4/test_wait_scenario_8_all_strategies.gif)",
            "- [局部避障场景 46：十种策略动图](../office_controlled_behavior_visualizations_v4/test_avoidance_scenario_46_all_strategies.gif)",
            "- [全局换路场景 76：十种策略动图](../office_controlled_behavior_visualizations_v4/test_reroute_scenario_76_all_strategies.gif)",
            "- [单障碍堵路场景 60000：十种策略动图](../office_controlled_behavior_visualizations_v4/controlled_blockage_scenario_60000_all_strategies.gif)",
            "",
            "## 原始分析图",
            "",
            "- [常规动态测试柱状图](../office_controlled_behavior_study_v4/behavior_comparison.png)",
            "- [近距离堵路诊断柱状图](../office_controlled_blockage_diagnostic_v4/diagnostic_comparison.png)",
            "- [16 个检查点的训练与冲突曲线](../office_checkpoint_conflict_analysis_v4/checkpoint_conflict_curves.png)",
            "",
            "本实验不测试地图泛化：所有方法使用同一张 40×40 Office 地图、相同起终点和同一动态场景库。训练、验证和测试使用不同场景编号/相位组合；近距离堵路是额外构造的因果诊断。",
        ]
    )
    (OUTPUT_DIR / "RESULT_SUMMARY.zh-CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _collect()
    _write_matrix(records)
    _plot(records)
    _write_report(records)
    print(f"Wrote comprehensive Office summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
