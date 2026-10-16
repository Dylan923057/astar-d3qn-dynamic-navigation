"""Archive dynamic-prediction validation evidence for GitHub and paper use."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "paper_evidence"
PHASE = "05_2026-09-24_dynamic_prediction_auxiliary"
MAP_ID = "irregular_workcell_91701"
METHODS = ("global_prediction", "decision_weighted_prediction")
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

    source_root = ROOT / "outputs" / "dynamic_prediction_auxiliary_v1" / "formal" / MAP_ID
    summary_root = (
        ROOT
        / "outputs"
        / "dynamic_prediction_auxiliary_v1"
        / "analysis_validation"
        / MAP_ID
    )
    decision = json.loads((summary_root / "decision.json").read_text(encoding="utf-8"))
    if not decision.get("validation_only") or decision.get("test_files_read"):
        raise SystemExit("Expected a validation-only decision that did not read test files.")

    copy_file(ROOT / "configs" / "risk_handover_v1.yaml", destination / "config" / "risk_handover_v1.yaml")
    copy_file(
        ROOT / "docs" / "DYNAMIC_PREDICTION_AUXILIARY_V1.zh-CN.md",
        destination / "protocol" / "DYNAMIC_PREDICTION_AUXILIARY_V1.zh-CN.md",
    )
    for name in ("decision.json", "per_seed_metrics.csv", "report.md"):
        copy_file(summary_root / name, destination / "analysis" / "validation_decision" / name)

    index_path = EVIDENCE / "experiment_index.json"
    index_rows = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if any(row["phase"] == PHASE for row in index_rows):
        raise SystemExit(f"Index already contains phase: {PHASE}")

    new_index_rows = []
    completed_times = []
    code_hashes = set()
    config_hashes = set()
    manifest_hashes = set()
    for method in METHODS:
        for seed in (0, 1, 2):
            source = source_root / f"seed_{seed}" / method
            target = destination / "runs" / MAP_ID / method / f"seed_{seed}"
            result_path = source / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "validation_complete" or result.get("smoke"):
                raise SystemExit(f"Not a completed formal validation run: {source}")
            if result.get("adaptation_run", {}).get("steps") != 200_000:
                raise SystemExit(f"Unexpected adaptation length: {source}")
            if (source / "test_evaluation.csv").exists() or "test" in result:
                raise SystemExit(f"Unexpected test evidence in validation-only run: {source}")
            for name in RUN_FILES:
                copy_file(source / name, target / name)

            completed = datetime.fromtimestamp(result_path.stat().st_mtime)
            completed_times.append(completed)
            code_hashes.add(result["code_sha256"])
            config_hashes.add(result["config_sha256"])
            manifest_hashes.add(result["manifest_sha256"])
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
                "test_conflict_safe_success": "",
                "test_conflict_dynamic_collision": "",
                "test_conflict_timeout": "",
                "static_safe_success": "",
                "config_sha256": result["config_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "code_sha256": result["code_sha256"],
                "fork_sha256": result["fork_sha256"],
                "result_path": str(relative_result),
            })

    if len(code_hashes) != 1 or len(config_hashes) != 1 or len(manifest_hashes) != 1:
        raise SystemExit("Formal runs do not share one code/config/manifest provenance group.")

    readme = """# 动态占用预测辅助消融（2026-09-24）

## 实验内容

- A：已归档的 `time_decay` 基线，本阶段不重复复制。
- B：`global_prediction`，seed 0/1/2。
- C：`decision_weighted_prediction`，seed 0/1/2。
- 使用第一张地图，每次进行 20 万动态交互步。
- 本阶段严格只归档验证集证据，没有生成或读取测试集结果。
- 模型权重与 smoke 结果继续保留在本地 `outputs/`。

## 验证集结论

- A 的平均安全成功 AUC 为 0.7042。
- B 的平均安全成功 AUC 为 0.7424，相对 A 提高 0.0382。
- C 的平均安全成功 AUC 为 0.7104，相对 A 提高 0.0063，但比 B 低 0.0319。
- B 在 2/3 个 seed 上优于 A，且三个 seed 的最终验证均无碰撞和超时。
- C 的 seed 1 出现安全成功率下降及碰撞、超时退化，预注册继续门槛未通过。
- 当前证据支持保留全局预测辅助 B 进入独立确认，不支持继续调节决策区加权 C。

## 目录

- `analysis/validation_decision/`：验证集汇总、逐 seed 指标和冻结决策。
- `config/`：本轮实验配置快照。
- `protocol/`：实验设计、判定标准与测试集隔离规则。
- `runs/`：六次正式运行的训练审计和验证结果，不含模型权重与测试结果。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    index_rows.extend(new_index_rows)
    index_rows.sort(key=lambda row: (row["completed_at"], row["phase"], row["method"], int(row["seed"])))
    index_path.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    existing = root_readme.read_text(encoding="utf-8-sig")
    section = f"""

### 05：2026-09-24，动态占用预测辅助消融

- 目录：`{PHASE}/`
- 方法：`global_prediction`、`decision_weighted_prediction`，各 3 个 seed；A 组复用既有 `time_decay`。
- 结论：全局预测辅助 B 显示值得独立确认的验证增益；决策区加权 C 未增加价值并出现单 seed 退化。
- 用途：B 可作为后续候选改进，C 作为消融和失败分析；本阶段未解锁测试集。
"""
    existing = re.sub(
        r"`experiment_index\.csv`：\d+ 个正式运行的可读索引。",
        f"`experiment_index.csv`：{len(index_rows)} 个正式运行的可读索引。",
        existing,
    )
    root_readme.write_text(existing.rstrip() + section + "\n", encoding="utf-8")

    checklist = EVIDENCE / "UPLOAD_CHECKLIST.md"
    checklist_text = checklist.read_text(encoding="utf-8-sig")
    checklist_text = checklist_text.replace("分为四个阶段", "分为五个阶段")
    checklist_text = checklist_text.replace(
        "已包含配置、训练审计、验证结果和固定最终测试结果。",
        "已包含配置、训练审计和验证结果；仅在协议允许时归档最终测试结果。",
    )
    checklist_text = checklist_text.replace(
        "阶段 04 已记录复用基础模型哈希与实际分支代码哈希的区别。",
        "阶段 04 已记录复用基础模型哈希与实际分支代码哈希的区别。\n"
        "- [x] 阶段 05 保持验证集决策与测试集隔离，未归档测试结果。",
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
    write_csv(EVIDENCE / "SHA256SUMS.csv", checksum_rows, ("path", "bytes", "sha256"))
    print(json.dumps({
        "archive": str(destination),
        "new_runs": len(new_index_rows),
        "indexed_runs": len(index_rows),
        "archived_files": sum(1 for path in destination.rglob("*") if path.is_file()),
        "test_files_archived": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
