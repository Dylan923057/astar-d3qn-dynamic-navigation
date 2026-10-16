"""Archive the completed cross-map dynamic-prediction validation experiment."""

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
PHASE = "06_2026-09-25_dynamic_prediction_generalization"
MAP_ID = "irregular_workcell_91702"
METHODS = {
    "time_decay": "schedule_decay",
    "global_prediction": "global_prediction",
}
RUN_FILES = (
    "fork_audit.json",
    "result.json",
    "training.csv",
    "validation_curve.csv",
    "validation_details.csv",
)
FOUNDATION_FILES = (
    "effective_config.json",
    "foundation_evaluation.csv",
    "foundation_status.json",
    "training.csv",
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

    run_root = ROOT / "outputs" / "dynamic_prediction_generalization_v1" / "formal" / MAP_ID
    foundation_root = ROOT / "outputs" / "risk_handover_v1" / "formal" / MAP_ID
    summary_root = (
        ROOT
        / "outputs"
        / "dynamic_prediction_generalization_v1"
        / "analysis_validation"
        / MAP_ID
    )
    decision = json.loads((summary_root / "decision.json").read_text(encoding="utf-8"))
    if (
        not decision.get("validation_only")
        or decision.get("test_files_read")
        or not decision.get("matched_ab_forks_and_provenance")
        or not decision.get("generalizes")
    ):
        raise SystemExit("Expected a matched validation-only cross-map pass decision.")

    copy_file(
        ROOT / "configs" / "risk_handover_v1.yaml",
        destination / "config" / "risk_handover_v1.yaml",
    )
    copy_file(
        ROOT / "docs" / "DYNAMIC_PREDICTION_GENERALIZATION_V1.zh-CN.md",
        destination / "protocol" / "DYNAMIC_PREDICTION_GENERALIZATION_V1.zh-CN.md",
    )
    for name in ("decision.json", "per_seed_metrics.csv", "report.md"):
        copy_file(summary_root / name, destination / "analysis" / "validation_decision" / name)

    for seed in (0, 1, 2):
        source = foundation_root / f"seed_{seed}" / "foundation"
        target = destination / "foundations" / MAP_ID / f"seed_{seed}"
        for name in FOUNDATION_FILES:
            copy_file(source / name, target / name)

    index_path = EVIDENCE / "experiment_index.json"
    index_rows = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if any(row["phase"] == PHASE for row in index_rows):
        raise SystemExit(f"Index already contains phase: {PHASE}")

    new_index_rows = []
    completed_times = []
    code_hashes = set()
    config_hashes = set()
    manifest_hashes = set()
    fork_hashes = {}
    for method, branch_name in METHODS.items():
        for seed in (0, 1, 2):
            source = run_root / f"seed_{seed}" / branch_name
            target = destination / "runs" / MAP_ID / method / f"seed_{seed}"
            result_path = source / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("status") != "validation_complete"
                or not result.get("test_deferred")
                or result.get("adaptation_run", {}).get("steps") != 200_000
            ):
                raise SystemExit(f"Not a completed 200k validation-only run: {source}")
            if (source / "test_evaluation.csv").exists() or "test" in result:
                raise SystemExit(f"Unexpected test evidence: {source}")
            for name in RUN_FILES:
                copy_file(source / name, target / name)

            completed = datetime.fromtimestamp(result_path.stat().st_mtime)
            completed_times.append(completed)
            code_hashes.add(result["code_sha256"])
            config_hashes.add(result["config_sha256"])
            manifest_hashes.add(result["manifest_sha256"])
            fork_hashes.setdefault(seed, set()).add(result["fork_sha256"])
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
        raise SystemExit("A/B runs do not share one code/config/manifest provenance group.")
    if any(len(hashes) != 1 for hashes in fork_hashes.values()):
        raise SystemExit("A/B runs did not fork from the same per-seed foundation.")

    readme = """# 动态占用预测辅助跨地图验证（2026-09-25）

## 实验内容

- 地图：`irregular_workcell_91702`。
- A：`time_decay`，seed 0/1/2。
- B：`global_prediction`，seed 0/1/2。
- 每次适应训练固定 20 万步；每个 seed 的 A/B 从同一基础快照分叉。
- B 固定 `prediction_loss_weight=0.1`、`prediction_pos_weight=20`，没有继续调参。
- 本阶段严格只归档验证证据，没有生成或读取测试集结果。
- foundation 和最终模型权重继续保留在本地 `outputs/`，归档只保存审计文件。

## 验证集结论

- A 的平均安全成功 AUC 为 0.9271。
- B 的平均安全成功 AUC 为 0.9590，相对 A 提高 0.0319。
- B 在 3/3 个 seed 上均优于对应 A。
- B 的最终碰撞率、超时率没有退化，后三 seed 后 5 万步波动均值与 A 相同。
- 预先登记的跨地图门槛全部通过，下一步为扩展到 5 个 seed，而不是立即查看测试集。

## 目录

- `analysis/validation_decision/`：冻结的跨地图验证判定和逐 seed 指标。
- `config/`：本轮配置快照。
- `protocol/`：实验设计、继续门槛和测试隔离规则。
- `foundations/`：三个基础模型的配置、训练和资格审计，不含权重。
- `runs/`：A/B 六次正式运行的训练与验证证据，不含权重和测试结果。
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

### 06：2026-09-25，动态占用预测辅助跨地图验证

- 目录：`{PHASE}/`
- 地图：`irregular_workcell_91702`；方法为 `time_decay` 和 `global_prediction`，各 3 个 seed。
- 结论：B 平均验证 AUC 提高 0.0319，3/3 seed 提升且安全性、后期稳定性不退化，跨地图门槛全部通过。
- 用途：支持全局动态预测辅助的初步跨地图有效性；下一步扩展到 5 个 seed，本阶段仍未解锁测试集。
"""
    existing = re.sub(
        r"`experiment_index\.csv`：\d+ 个正式运行的可读索引。",
        f"`experiment_index.csv`：{len(index_rows)} 个正式运行的可读索引。",
        existing,
    )
    root_readme.write_text(existing.rstrip() + section + "\n", encoding="utf-8")

    checklist = EVIDENCE / "UPLOAD_CHECKLIST.md"
    checklist_text = checklist.read_text(encoding="utf-8-sig")
    checklist_text = checklist_text.replace("分为五个阶段", "分为六个阶段")
    checklist_text = checklist_text.replace(
        "- [x] 阶段 05 保持验证集决策与测试集隔离，未归档测试结果。",
        "- [x] 阶段 05 保持验证集决策与测试集隔离，未归档测试结果。\n"
        "- [x] 阶段 06 的 A/B 来源和基础快照匹配，未归档权重或测试结果。",
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
        "weight_files_archived": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
