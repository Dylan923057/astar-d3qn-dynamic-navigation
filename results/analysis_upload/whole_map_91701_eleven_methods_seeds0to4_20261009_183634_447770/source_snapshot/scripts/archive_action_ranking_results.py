"""Archive the completed action-ranking ablation for GitHub/paper evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "paper_evidence"
PHASE = "04_2026-09-23_action_ranking_ablation"
MAP_ID = "irregular_workcell_91701"
TRAINING_CODE_SHA256 = "90e46b30a5b1a1d95c55d2c758da7ed599d57b768c645dfd01d7cb050922004e"
METHODS = ("demo_action_margin", "all_action_margin")
RUN_FILES = (
    "fork_audit.json",
    "result.json",
    "training.csv",
    "validation_curve.csv",
    "validation_details.csv",
    "test_evaluation.csv",
)
INDEX_FIELDS = (
    "phase",
    "completed_at",
    "map_id",
    "method",
    "seed",
    "status",
    "replay_schedule",
    "validation_conflict_auc",
    "test_conflict_safe_success",
    "test_conflict_dynamic_collision",
    "test_conflict_timeout",
    "static_safe_success",
    "config_sha256",
    "manifest_sha256",
    "code_sha256",
    "fork_sha256",
    "result_path",
)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    destination = EVIDENCE / PHASE
    if destination.exists():
        raise SystemExit(f"Archive already exists: {destination}")

    source_root = ROOT / "outputs" / "risk_handover_v1" / "formal" / MAP_ID
    summary_root = ROOT / "outputs" / "action_ranking_ablation_summary_v1" / MAP_ID
    decision = json.loads((summary_root / "decision.json").read_text(encoding="utf-8"))
    if not decision.get("validation_only") or decision.get("continue"):
        raise SystemExit("Expected a frozen validation-only stop decision before archiving test files.")

    copy_file(ROOT / "configs" / "risk_handover_v1.yaml", destination / "config" / "risk_handover_v1.yaml")
    copy_file(ROOT / "docs" / "ACTION_RANKING_ABLATION_V1.zh-CN.md", destination / "protocol" / "ACTION_RANKING_ABLATION_V1.zh-CN.md")
    for name in ("decision.json", "per_seed_metrics.csv", "report.md"):
        copy_file(summary_root / name, destination / "analysis" / "validation_decision" / name)
    preflight = ROOT / "outputs" / "action_ranking_preflight_v1" / MAP_ID
    for name in ("action_value_comparison.csv", "report.md"):
        copy_file(preflight / name, destination / "analysis" / "preflight" / name)
    diagnostic = ROOT / "outputs" / "time_decay_failure_diagnostic_v2" / MAP_ID
    for name in (
        "report.md",
        "failures_only.csv",
        "cross_seed_critical_comparison.csv",
        "consistency_check.csv",
        "provenance.csv",
    ):
        copy_file(diagnostic / name, destination / "analysis" / "failure_diagnostic" / name)
    for figure in sorted((diagnostic / "figures").glob("*.png")):
        copy_file(figure, destination / "analysis" / "failure_diagnostic" / "figures" / figure.name)

    new_index_rows = []
    completed_times = []
    for method in METHODS:
        for seed in (0, 1, 2):
            source = source_root / f"seed_{seed}" / method
            target = destination / "runs" / MAP_ID / method / f"seed_{seed}"
            result = json.loads((source / "result.json").read_text(encoding="utf-8"))
            if result.get("status") != "complete" or result.get("smoke"):
                raise SystemExit(f"Not a completed formal run: {source}")
            for name in RUN_FILES:
                copy_file(source / name, target / name)
            provenance = {
                "training_code_sha256": TRAINING_CODE_SHA256,
                "foundation_code_sha256_in_result_json": result["code_sha256"],
                "config_sha256": result["config_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "fork_sha256": result["fork_sha256"],
                "note": (
                    "The legacy result writer inherited code_sha256 from the reused foundation. "
                    "training_code_sha256 is the reconstructed branch-code digest used by this run."
                ),
            }
            (target / "branch_provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            completed = datetime.fromtimestamp((source / "result.json").stat().st_mtime)
            completed_times.append(completed)
            test = result["test"]
            relative_result = (target / "result.json").relative_to(EVIDENCE)
            new_index_rows.append({
                "phase": PHASE,
                "completed_at": completed.strftime("%Y-%m-%d %H:%M:%S"),
                "map_id": MAP_ID,
                "method": method,
                "seed": seed,
                "status": result["status"],
                "replay_schedule": result["replay_schedule"],
                "validation_conflict_auc": result["validation_conflict_auc"],
                "test_conflict_safe_success": test["conflict_safe_success"],
                "test_conflict_dynamic_collision": test["conflict_dynamic_collision"],
                "test_conflict_timeout": test["conflict_timeout"],
                "static_safe_success": test["static_safe_success"],
                "config_sha256": result["config_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "code_sha256": TRAINING_CODE_SHA256,
                "fork_sha256": result["fork_sha256"],
                "result_path": str(relative_result),
            })

    readme = f"""# 动态风险动作排序消融（2026-09-23）

## 内容

- A：阶段 02 已归档的 `time_decay`，本阶段不重复复制。
- B：`demo_action_margin`，seed 0/1/2。
- C：`all_action_margin`，seed 0/1/2。
- 每次训练使用第一张地图和 20 万动态交互步。
- 模型权重与大型逐步测试轨迹仍只保留在本地 `outputs/`。

## 验证集结论

- A 平均安全成功 AUC：0.7042。
- B 平均安全成功 AUC：0.5590。
- C 平均安全成功 AUC：0.4646。
- C-A：-0.2396；C-B：-0.0944；C 优于 A 的 seed 数为 0/3。
- 预登记继续条件未通过，本实现停止，不继续搜索间隔或损失权重。

测试文件是在验证集停止决策写入 `analysis/validation_decision/decision.json` 后归档，未参与方法选择。本阶段属于正式负结果，可用于消融和失败分析，不能表述为性能提升。

## 目录

- `analysis/validation_decision/`：严格基于验证曲线的 A/B/C 决策。
- `analysis/preflight/`：相同碰撞前观测的三个网络动作评分及标签覆盖。
- `analysis/failure_diagnostic/`：时间衰减最终模型的失败定位及代表图。
- `runs/`：B/C 六次正式运行的配置审计、训练、验证和固定最终测试结果。
- `branch_provenance.json`：区分复用基础模型的旧代码哈希与本轮实际分支代码哈希。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    index_json_path = EVIDENCE / "experiment_index.json"
    index_rows = json.loads(index_json_path.read_text(encoding="utf-8-sig"))
    index_rows.extend(new_index_rows)
    index_rows.sort(key=lambda row: (row["completed_at"], row["phase"], row["method"], int(row["seed"])))
    index_json_path.write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(EVIDENCE / "experiment_index.csv", index_rows, INDEX_FIELDS)

    provenance_path = EVIDENCE / "provenance_groups.csv"
    provenance_rows = read_csv(provenance_path)
    provenance_rows.append({
        "phase": PHASE,
        "first_completed": min(completed_times).strftime("%Y-%m-%d %H:%M:%S"),
        "last_completed": max(completed_times).strftime("%Y-%m-%d %H:%M:%S"),
        "run_count": len(new_index_rows),
        "code_sha256": TRAINING_CODE_SHA256,
        "config_sha256": new_index_rows[0]["config_sha256"],
        "manifest_sha256": new_index_rows[0]["manifest_sha256"],
    })
    write_csv(provenance_path, provenance_rows, provenance_rows[0].keys())

    root_readme = EVIDENCE / "README.md"
    existing = root_readme.read_text(encoding="utf-8-sig")
    section = f"""

### 04（2026-09-23，动态风险动作排序消融）

- 目录：`{PHASE}/`
- 方法：`demo_action_margin`、`all_action_margin`，各 3 个 seed；A 组复用阶段 02。
- 结论：B、C 的平均验证 AUC 均低于时间衰减基线，C 也低于 B；预登记继续条件未通过。
- 用途：正式负结果、动作排序机制消融与毕业论文失败分析，不作为小论文性能提升证据。
"""
    if PHASE not in existing:
        root_readme.write_text(existing.rstrip() + section + "\n", encoding="utf-8")

    checksum_rows = []
    for phase_directory in sorted(path for path in EVIDENCE.iterdir() if path.is_dir()):
        for path in sorted(phase_directory.rglob("*")):
            if not path.is_file():
                continue
            checksum_rows.append({
                "path": str(path.relative_to(EVIDENCE)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    write_csv(EVIDENCE / "SHA256SUMS.csv", checksum_rows, ("path", "bytes", "sha256"))
    print(json.dumps({
        "archive": str(destination),
        "new_runs": len(new_index_rows),
        "indexed_runs": len(index_rows),
        "archived_files": sum(1 for path in destination.rglob("*") if path.is_file()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
