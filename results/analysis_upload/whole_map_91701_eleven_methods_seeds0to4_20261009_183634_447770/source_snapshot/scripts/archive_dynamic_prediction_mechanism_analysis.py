"""Archive the validation-only cross-map mechanism analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "paper_evidence"
PHASE = "08_dynamic_prediction_cross_map_analysis"
ANALYSIS_FILES = (
    "analysis.json",
    "environment_statistics_by_map.csv",
    "environment_statistics_by_split.csv",
    "prediction_navigation_by_map.csv",
    "prediction_navigation_by_seed.csv",
    "report.md",
)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    destination = EVIDENCE / PHASE
    if destination.exists():
        raise SystemExit(f"Archive already exists: {destination}")
    source = ROOT / "outputs" / "dynamic_prediction_mechanism_analysis_v1"
    analysis = json.loads((source / "analysis.json").read_text(encoding="utf-8"))
    if (
        analysis.get("test_split_used")
        or analysis.get("training_rerun")
        or not analysis.get("core_dynamic_controls_matched_across_maps")
    ):
        raise SystemExit("Mechanism analysis does not match the frozen validation-only protocol.")

    for name in ANALYSIS_FILES:
        copy_file(source / name, destination / "analysis" / name)
    copy_file(
        ROOT / "configs" / "risk_handover_v1.yaml",
        destination / "config" / "risk_handover_v1.yaml",
    )
    copy_file(
        ROOT / "docs" / "DYNAMIC_PREDICTION_CROSS_MAP_MECHANISM_ANALYSIS.zh-CN.md",
        destination / "protocol" / "DYNAMIC_PREDICTION_CROSS_MAP_MECHANISM_ANALYSIS.zh-CN.md",
    )

    readme = """# 动态占用预测辅助跨地图机制分析

## 范围

- 仅分析三个冻结地图的 train、validation 场景定义和已有验证曲线。
- 未读取 test split，未重新训练，未调整预测或强化学习参数。
- 目的不是寻找新的最佳配置，而是解释 B 在不同地图上的作用边界。

## 环境统计结论

- 三张地图的训练与验证设计均为平均 4 个动态障碍，其中平均 1.5 个为因果冲突障碍。
- 三图障碍速度均为 1 格/步，冲突/control 比例均为 1:1，early/middle/late 位置分层相同。
- 每 100 个参考路径步的平均风险位置数均为 2.273。
- 因此，现有三地图不能支持“map03 因障碍更少、更慢或冲突更少而过于简单”的解释。

## 预测与导航关系

- map01：prediction F1 0.309，B-A AUC +0.0382。
- map02：prediction F1 0.443，B-A AUC +0.0319。
- map03：prediction F1 0.406，B-A AUC -0.0306。
- map03 的 prediction F1 和 decision-zone F1 均高于 map01，但导航收益为负。
- 这说明预测任务本身已经学到可测信号，但预测表示没有在所有地图上稳定转化为 Q 策略学习收益。

## 论文结论

- 不能声称该方法具有普适或稳健的三地图泛化提升。
- 可以报告 map01、map02 的条件性学习效率收益，同时把 map03 作为正式边界/负结果。
- 不能依据当前三图宣称“复杂动态环境越复杂，B 优势越大”，因为关键复杂度变量被配平而没有独立变化。
- 论文应突出辅助表示到控制策略的转化边界，并在局限性中明确地图依赖性。

## 下一步

当前优先写方法、机制分析和局限性，不继续训练。如果后续必须补一个实验，应预先登记新的动态复杂度梯度，独立改变障碍数量、速度和冲突频率，并使用新冻结的验证集；不得根据当前结果继续调预测参数。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    root_readme = EVIDENCE / "README.md"
    existing = root_readme.read_text(encoding="utf-8-sig")
    section = f"""

### 08：动态占用预测辅助跨地图机制分析

- 目录：`{PHASE}/`
- 数据：三个地图的 train/validation 场景定义、A/B 验证曲线和预测 F1；未使用测试划分。
- 结论：三图关键动态复杂度变量被配平；map03 预测质量不低但导航收益为负，说明预测到控制的收益转化具有地图依赖性。
- 用途：论文机制解释、适用边界与局限性，不作为新实验或普适泛化证据。
"""
    root_readme.write_text(existing.rstrip() + section + "\n", encoding="utf-8")

    checklist = EVIDENCE / "UPLOAD_CHECKLIST.md"
    checklist_text = checklist.read_text(encoding="utf-8-sig")
    checklist_text = checklist_text.replace("分为七个阶段", "分为八个阶段")
    checklist_text = checklist_text.replace(
        "- [x] 阶段 07 冻结第三地图负结果和三地图诊断，未解锁测试集。",
        "- [x] 阶段 07 冻结第三地图负结果和三地图诊断，未解锁测试集。\n"
        "- [x] 阶段 08 仅使用训练/验证数据完成跨地图机制分析，未重新训练。",
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
        "archived_files": sum(1 for path in destination.rglob("*") if path.is_file()),
        "test_split_used": False,
        "training_rerun": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
