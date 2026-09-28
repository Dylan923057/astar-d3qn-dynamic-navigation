"""Archive the validation-only decision-aligned prediction experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "paper_evidence"
PHASE = "09_2026-09-28_decision_aligned_prediction"
MAP_ID = "irregular_workcell_91703"
METHOD = "decision_aligned_prediction"
RUN_FILES = (
    "fork_audit.json",
    "result.json",
    "training.csv",
    "validation_curve.csv",
    "validation_details.csv",
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
    run_root = (
        ROOT
        / "outputs"
        / "decision_aligned_prediction_v1"
        / "formal"
        / MAP_ID
    )
    summary_root = (
        ROOT
        / "outputs"
        / "decision_aligned_prediction_v1"
        / "analysis_validation"
        / MAP_ID
    )
    decision = json.loads((summary_root / "decision.json").read_text(encoding="utf-8"))
    if (
        not decision.get("validation_only")
        or decision.get("test_files_read")
        or decision.get("phase_one_passes")
        or decision.get("next_step")
        != "stop_without_tuning_and_report_negative_or_inconclusive_result"
    ):
        raise SystemExit("Expected a validation-only phase-one stop decision.")

    copy_file(
        ROOT / "configs" / "risk_handover_v1.yaml",
        destination / "config" / "risk_handover_v1.yaml",
    )
    copy_file(
        ROOT / "docs" / "DECISION_ALIGNED_PREDICTION_V1.zh-CN.md",
        destination / "protocol" / "DECISION_ALIGNED_PREDICTION_V1.zh-CN.md",
    )
    for name in ("decision.json", "per_seed_metrics.csv", "report.md"):
        copy_file(summary_root / name, destination / "analysis" / name)

    index_path = EVIDENCE / "experiment_index.json"
    index_rows = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if any(row["phase"] == PHASE for row in index_rows):
        raise SystemExit(f"Index already contains phase: {PHASE}")

    completed_times = []
    code_hashes = set()
    config_hashes = set()
    manifest_hashes = set()
    new_index_rows = []
    for seed in (0, 1, 2):
        source = run_root / f"seed_{seed}" / METHOD
        target = destination / "runs" / MAP_ID / METHOD / f"seed_{seed}"
        result_path = source / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "validation_complete"
            or not result.get("test_deferred")
            or result.get("adaptation_run", {}).get("steps") != 200_000
            or result.get("replay_schedule") != METHOD
            or result.get("prediction_mode") != "global_prediction"
            or result.get("prediction_loss_weight") != 0.1
            or result.get("prediction_pos_weight") != 20.0
            or result.get("prediction_gradient_strategy") != "project_conflicting"
        ):
            raise SystemExit(f"Run differs from the frozen protocol: {source}")
        if (source / "test_evaluation.csv").exists() or "test" in result:
            raise SystemExit(f"Unexpected test evidence: {source}")
        if result["adaptation_run"].get("gradient_alignment_update_count") != 200_000:
            raise SystemExit(f"Gradient alignment was not active for every update: {source}")
        for name in RUN_FILES:
            copy_file(source / name, target / name)

        completed = datetime.fromtimestamp(result_path.stat().st_mtime)
        completed_times.append(completed)
        code_hashes.add(result["code_sha256"])
        config_hashes.add(result["config_sha256"])
        manifest_hashes.add(result["manifest_sha256"])
        new_index_rows.append({
            "phase": PHASE,
            "completed_at": completed.strftime("%Y-%m-%d %H:%M:%S"),
            "map_id": MAP_ID,
            "method": METHOD,
            "seed": seed,
            "status": result["status"],
            "replay_schedule": result["replay_schedule"],
            "validation_conflict_auc": result["validation_conflict_auc"],
            "test_conflict_safe_success": "",
            "test_conflict_dynamic_collision": "",
            "test_conflict_timeout": "",
            "static_safe_success": "",
            "config_sha256": result["config_sha256"],
            "manifest_sha256": result["manifest_sha256"],
            "code_sha256": result["code_sha256"],
            "fork_sha256": result["fork_sha256"],
            "result_path": str(
                (target / "result.json").relative_to(EVIDENCE)
            ),
        })
    if len(code_hashes) != 1 or len(config_hashes) != 1 or len(manifest_hashes) != 1:
        raise SystemExit("The three aligned runs do not share one provenance group.")

    readme = """# Decision-Aligned Prediction 第一阶段结果

## 实验范围

- 地图：`irregular_workcell_91703`。
- A：既有 `time_decay`；B：既有 `global_prediction`；D：新增 `decision_aligned_prediction`。
- D 使用 seed 0/1/2，每次固定训练 20 万步，并复用与 A/B 相同的逐 seed foundation。
- prediction loss weight 固定为 0.1，pos_weight 固定为 20；没有调整奖励、探索率、回放、网络规模或其他辅助模块。
- 本阶段只使用 validation，未生成、读取或归档 test 结果，也未归档模型权重。

## 主要结果

- A 平均 validation AUC：0.8674。
- B 平均 validation AUC：0.8368。
- D 平均 validation AUC：0.8743，比 B 高 0.0375，比 A 高 0.0069。
- D 在 2/3 seed 上高于 B，也在 2/3 seed 上不低于 A。
- 三种方法最终平均安全成功率均为 1.0，最终动态碰撞率和超时率均为 0。
- D 的 last-50k 平均安全成功率为 0.9213，低于 A 的 1.0；平均标准差为 0.1219，高于 A 的 0.0 和 B 的 0.0589。

## 机制结果

- TD 与 prediction 梯度冲突 batch 比例为 0.4878，修正比例同为 0.4878。
- 投影平均移除了 prediction 梯度范数的 0.0772。
- 平均 cosine similarity 从 0.0056 提高到 0.0828。
- 同批更新后的 TD loss 和 prediction loss 平均都下降，说明实现按预期工作。

## 冻结结论

梯度对齐消除了原始 global prediction 在平均 AUC 上的负迁移，并略高于基线，但没有满足预先登记的后期性能和稳定性条件。因此第一阶段判定为未通过：不进入新地图、不扩到 5 seed、不解锁 test，也不根据结果继续调参。该结果适合作为“机制有效但整体方法尚不稳定”的正式消融和失败分析。

## 目录

- `analysis/`：A/B/D 冻结判定、逐 seed 指标和报告。
- `config/`：实验配置快照。
- `protocol/`：预先冻结的方法、指标和继续条件。
- `runs/`：D 组三次正式运行的训练与验证证据，不含权重和 test。

A/B 原始运行已归档在 `07_dynamic_prediction_third_map_validation/`，此处不重复复制。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    index_rows.extend(new_index_rows)
    index_rows.sort(
        key=lambda row: (
            row["completed_at"],
            row["phase"],
            row["method"],
            int(row["seed"]),
        )
    )
    index_path.write_text(
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
        "code_sha256": next(iter(code_hashes)),
        "config_sha256": next(iter(config_hashes)),
        "manifest_sha256": next(iter(manifest_hashes)),
    })
    write_csv(provenance_path, provenance_rows, provenance_rows[0].keys())

    root_readme = EVIDENCE / "README.md"
    root_lines = root_readme.read_text(encoding="utf-8-sig").splitlines()
    root_lines = [
        (
            f"- `experiment_index.csv`：{len(index_rows)} 个正式运行的可读索引。"
            if "`experiment_index.csv`" in line
            else line
        )
        for line in root_lines
    ]
    root_lines.extend([
        "",
        "### 09：Decision-Aligned Prediction 梯度冲突消融",
        "",
        f"- 目录：`{PHASE}/`",
        "- 地图：`irregular_workcell_91703`；新增方法为 `decision_aligned_prediction`，共 3 个 seed。",
        "- 结论：平均 AUC 高于原始 prediction 和基线，但后 50k 性能与稳定性未达到冻结门槛。",
        "- 用途：梯度冲突机制证据及正式消融；不支持进入新地图、扩 seed 或解锁 test。",
    ])
    root_readme.write_text("\n".join(root_lines) + "\n", encoding="utf-8")

    checklist = EVIDENCE / "UPLOAD_CHECKLIST.md"
    checklist_text = checklist.read_text(encoding="utf-8-sig").rstrip()
    checklist_text += (
        "\n- [x] 阶段 09 已归档三次 validation-only 梯度对齐运行，"
        "未包含权重或 test 结果。\n"
    )
    checklist.write_text(checklist_text, encoding="utf-8")

    checksum_rows = []
    for phase_directory in sorted(path for path in EVIDENCE.iterdir() if path.is_dir()):
        for path in sorted(phase_directory.rglob("*")):
            if path.is_file():
                checksum_rows.append({
                    "path": str(path.relative_to(EVIDENCE)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
    write_csv(
        EVIDENCE / "SHA256SUMS.csv",
        checksum_rows,
        ("path", "bytes", "sha256"),
    )
    print(json.dumps({
        "archive": str(destination),
        "new_runs": len(new_index_rows),
        "indexed_runs": len(index_rows),
        "archived_files": sum(1 for path in destination.rglob("*") if path.is_file()),
        "test_files_archived": 0,
        "weight_files_archived": 0,
        "phase_one_passes": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
